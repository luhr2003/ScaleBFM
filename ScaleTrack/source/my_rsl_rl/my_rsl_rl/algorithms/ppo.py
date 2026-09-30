from __future__ import annotations

import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from my_rsl_rl.modules import ActorCritic, ActorCriticHumanoidTransformer
from my_rsl_rl.storage import RolloutStorage
from my_rsl_rl.utils import string_to_callable


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic | ActorCriticHumanoidTransformer
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticHumanoidTransformer,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        actor_learning_rate: float = 0.001,
        critic_learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        # Fine-tuning options (all off by default: the original training recipe is unchanged)
        anchor_coef: float = 0.0,
        actor_freeze_iters: int = 0,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # Create optimizer
        self.actor_optimizer = optim.Adam(self.policy.actor_parameters, lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.policy.critic_parameters, lr=critic_learning_rate)

        # Create rollout storage
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

        # Anchor to a frozen reference policy: penalise KL(pi_ref || pi) on the flat-rehearsal samples (obs group "group" == 0)
        self.anchor_coef = anchor_coef
        self.anchor_policy = None
        self.actor_freeze_iters = actor_freeze_iters
        self.actor_frozen = False
        self.num_skipped_updates = 0

    def set_anchor(self, checkpoint: str | None = None) -> None:
        """Freeze the reference policy of the anchor loss: the weights in `checkpoint` if given (always use this when a
        run may be resumed from its own checkpoints), otherwise a copy of the current policy."""
        self.anchor_policy = copy.deepcopy(self.policy)
        if checkpoint:
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)["model_state_dict"]
            self.anchor_policy.load_state_dict(state)
        self.anchor_policy.eval()
        for p in self.anchor_policy.parameters():
            p.requires_grad_(False)

    def apply_learning_rates(self) -> None:
        """Write the configured learning rates into the optimizers (a resumed checkpoint carries the old ones)."""
        for g in self.actor_optimizer.param_groups:
            g["lr"] = self.actor_learning_rate
        for g in self.critic_optimizer.param_groups:
            g["lr"] = self.critic_learning_rate

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        # Create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        # Compute value for the last step
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, gpu_global_rank=self.gpu_global_rank, gpu_world_size=self.gpu_world_size,
        )

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_anchor_kl = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            original_batch_size = obs_batch.batch_size[0]

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.actor_learning_rate = max(1e-5, self.actor_learning_rate/1.5)
                            self.critic_learning_rate = max(1e-5, self.critic_learning_rate/1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.actor_learning_rate = min(1e-2, self.actor_learning_rate*1.5)
                            self.critic_learning_rate = min(1e-2, self.critic_learning_rate*1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        actor_lr_tensor = torch.tensor(self.actor_learning_rate, device=self.device)
                        critic_lr_tensor = torch.tensor(self.critic_learning_rate, device=self.device)
                        torch.distributed.broadcast(actor_lr_tensor, src=0)
                        torch.distributed.broadcast(critic_lr_tensor, src=0)
                        self.actor_learning_rate = actor_lr_tensor.item()
                        self.critic_learning_rate = critic_lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.actor_optimizer.param_groups:
                        param_group["lr"] = self.actor_learning_rate
                    for param_group in self.critic_optimizer.param_groups:
                        param_group["lr"] = self.critic_learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.anchor_policy is not None and self.anchor_coef > 0.0:
                with torch.no_grad():
                    self.anchor_policy.act(obs_batch)  # sets the reference distribution (mask/mode handled inside)
                    mu_ref = self.anchor_policy.action_mean[:original_batch_size]
                    sigma_ref = self.anchor_policy.action_std[:original_batch_size]
                kl_ref = (
                    torch.log(sigma_batch / sigma_ref)
                    + (torch.square(sigma_ref) + torch.square(mu_ref - mu_batch)) / (2.0 * torch.square(sigma_batch))
                    - 0.5
                ).sum(dim=-1)
                if "group" in obs_batch.keys():
                    flat_mask = (obs_batch["group"][:, 0] < 0.5).float()
                else:
                    flat_mask = torch.ones_like(kl_ref)
                anchor_kl = (kl_ref * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
                loss = loss + self.anchor_coef * anchor_kl
                mean_anchor_kl += anchor_kl.item()

            # Compute the gradients for PPO
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()

            if os.environ.get("PPO_DEBUG_NONFINITE") and self.num_skipped_updates < 3:
                bad = [n for n, p in self.policy.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                nograd = [n for n, p in self.policy.named_parameters() if p.grad is None]
                print(f"[PPO-debug rank {self.gpu_global_rank}] before reduce: non-finite grads in {bad[:8]} ({len(bad)}), params without grad: {nograd[:8]} ({len(nograd)})", flush=True)
                if bad:
                    for key in ("critic", "critic_task", "action", "policy", "policy_task"):
                        x = obs_batch[key]
                        print(f"[PPO-debug rank {self.gpu_global_rank}] obs[{key}] finite={bool(torch.isfinite(x).all())} max|x|={x.abs().max().item():.4g} "
                              f"argmax sample={int(x.abs().reshape(x.shape[0], -1).amax(1).argmax())}", flush=True)
                    print(f"[PPO-debug rank {self.gpu_global_rank}] returns max {returns_batch.abs().max().item():.4g} values max {target_values_batch.abs().max().item():.4g} value_batch max {value_batch.abs().max().item():.4g}", flush=True)
                    if not getattr(self, "_debug_located", False) and not torch.isfinite(value_batch).all():
                        self._debug_located = True
                        self._debug_locate_nan(obs_batch)

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            if os.environ.get("PPO_DEBUG_NONFINITE") and self.num_skipped_updates < 3:
                bad = [n for n, p in self.policy.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                print(f"[PPO-debug rank {self.gpu_global_rank}] after reduce: non-finite grads in {bad[:8]} ({len(bad)})", flush=True)

            # Apply the gradients for PPO. A non-finite gradient (norm computed AFTER the multi-GPU reduction, so all ranks take the
            # same decision) would turn the parameters into NaN for good: skip that minibatch instead and say why.
            grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            if not torch.isfinite(grad_norm):
                self.num_skipped_updates += 1
                if self.num_skipped_updates <= 20 and self.gpu_global_rank == 0:
                    print(
                        f"[PPO] skipped a minibatch with non-finite gradients (total {self.num_skipped_updates}): "
                        f"loss {loss.item():.4g} surrogate {surrogate_loss.item():.4g} value {value_loss.item():.4g} "
                        f"mu finite {bool(torch.isfinite(mu_batch).all())} sigma [{sigma_batch.min().item():.4g}, {sigma_batch.max().item():.4g}] "
                        f"ratio max {ratio.max().item():.4g} adv finite {bool(torch.isfinite(advantages_batch).all())}",
                        flush=True,
                    )
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                continue
            if not self.actor_frozen:
                self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            
        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_anchor_kl /= num_updates

        if os.environ.get("PPO_DEBUG_NONFINITE") and self.is_multi_gpu:
            lo, hi, own = self._param_checksum_spread()
            if self.gpu_global_rank == 0:
                print(f"[PPO-debug] parameter checksum across ranks after update: min {lo.item():.9g} max {hi.item():.9g}", flush=True)

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.anchor_policy is not None and self.anchor_coef > 0.0:
            loss_dict["anchor_kl"] = mean_anchor_kl
        loss_dict["skipped_updates_total"] = float(self.num_skipped_updates)

        return loss_dict

    def _debug_locate_nan(self, obs_batch) -> None:
        """Debug helper (PPO_DEBUG_NONFINITE): find where a non-finite critic output comes from and whether the CUDA device matters."""
        import copy

        param_device = next(self.policy.parameters()).device
        report = []

        def make_hook(name):
            def hook(module, inputs, output):
                outs = [output] if torch.is_tensor(output) else [o for o in (output if isinstance(output, (tuple, list)) else []) if torch.is_tensor(o)]
                for o in outs:
                    if o.is_floating_point() and not torch.isfinite(o).all() and len(report) < 4:
                        ins = [bool(torch.isfinite(i).all()) for i in inputs if torch.is_tensor(i) and i.is_floating_point()]
                        report.append(f"{name}({type(module).__name__}) out-device {o.device} inputs-finite {ins}")
                        break
            return hook

        hooks = [m.register_forward_hook(make_hook(n)) for n, m in self.policy.named_modules()]
        try:
            with torch.no_grad():
                v_nograd = self.policy.evaluate(obs_batch)
            first_modules = list(report)
            with torch.no_grad(), torch.cuda.device(param_device):
                v_guard = self.policy.evaluate(obs_batch)
        finally:
            for h in hooks:
                h.remove()
        print(
            f"[PPO-debug rank {self.gpu_global_rank}] locate: current_device {torch.cuda.current_device()} param device {param_device} "
            f"obs devices {sorted({str(v.device) for v in obs_batch.values()})} | critic finite: no_grad {bool(torch.isfinite(v_nograd).all())}, "
            f"with device guard {bool(torch.isfinite(v_guard).all())} | first non-finite modules: {first_modules}",
            flush=True,
        )
        try:
            with torch.no_grad():
                critic = copy.deepcopy(self.policy.critic).cpu()
                embedder = copy.deepcopy(self.policy.critic_task_embedder).cpu()
                cpu_obs = obs_batch.to("cpu")
                prop_obs, task_obs, action_obs = self.policy.get_critic_obs(cpu_obs)
                v_cpu = critic(prop_obs, action_obs, embedder(task_obs))
            print(f"[PPO-debug rank {self.gpu_global_rank}] locate: cpu copy of the critic finite {bool(torch.isfinite(v_cpu).all())}, max {v_cpu.abs().max().item():.4g}", flush=True)
        except Exception as exc:  # debugging aid only
            print(f"[PPO-debug rank {self.gpu_global_rank}] locate: cpu check failed: {exc!r}", flush=True)

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters (and buffers) of rank 0 to all GPUs.

        Every tensor is broadcast in place with NCCL on the rank's own device. The former implementation pickled the CUDA state_dict
        (`broadcast_object_list`): the unpickled tensors land on rank 0's device (cuda:0) and `load_state_dict` then copies GPU->GPU,
        which silently yields all-zero tensors on machines with broken peer-to-peer access (IOMMU): every rank but 0 trained a zeroed policy.
        """
        for tensor in self.policy.state_dict().values():
            if not torch.is_tensor(tensor):
                continue
            if tensor.is_contiguous():
                torch.distributed.broadcast(tensor, src=0)
            else:
                tmp = tensor.contiguous()
                torch.distributed.broadcast(tmp, src=0)
                tensor.copy_(tmp)
        # Sanity check: all ranks must now hold identical, finite parameters.
        lo, hi, own = self._param_checksum_spread()
        if not (torch.isfinite(lo) and torch.isfinite(hi) and lo.item() == hi.item()):
            raise RuntimeError(f"[PPO] parameters differ across ranks after broadcast (checksum min {lo.item()}, max {hi.item()}, rank {self.gpu_global_rank}: {own.item()})")
        if self.gpu_global_rank == 0:
            print(f"[PPO] parameters synchronized across {self.gpu_world_size} ranks (checksum {own.item():.6f})", flush=True)

    def _gathered_checksums(self, own: torch.Tensor) -> torch.Tensor:
        """All ranks' scalar checksums as a (world_size,) float64 tensor. Done with a SUM all_reduce on purpose: NCCL MIN/MAX ignore
        NaN, so a rank holding NaN parameters would go unnoticed."""
        gathered = torch.zeros(self.gpu_world_size, dtype=torch.float64, device=own.device)
        gathered[self.gpu_global_rank] = own
        torch.distributed.all_reduce(gathered, op=torch.distributed.ReduceOp.SUM)
        return gathered

    def _param_checksum_spread(self):
        """(min over ranks, max over ranks, own) of the sum of all policy parameters and buffers, in float64 (NaN propagates)."""
        own = torch.stack([p.detach().double().sum() for p in self.policy.state_dict().values() if torch.is_tensor(p)]).sum()
        gathered = self._gathered_checksums(own)
        return gathered.min(), gathered.max(), own

    def check_optimizer_sync(self) -> None:
        """All ranks must hold the same, finite optimizer state (Adam moments and step counters) after a resume."""
        own = torch.zeros((), dtype=torch.float64, device=self.device)
        for opt in (self.actor_optimizer, self.critic_optimizer):
            for state in opt.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        own = own + value.detach().double().sum().to(self.device)
        gathered = self._gathered_checksums(own)
        if not (torch.isfinite(gathered).all() and gathered.min().item() == gathered.max().item()):
            raise RuntimeError(f"[PPO] optimizer state differs across ranks or is not finite after loading the checkpoint: {gathered.tolist()}")
        if self.gpu_global_rank == 0:
            print(f"[PPO] optimizer state identical across {self.gpu_world_size} ranks (checksum {own.item():.6g})", flush=True)

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
