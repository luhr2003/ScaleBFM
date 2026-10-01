"""Paper-protocol evaluation of a ScaleTrack BFM checkpoint over control modes and tracking types.

For every requested (mode, tracking) configuration each clip of a motion set is replayed once in IsaacLab with the
actor driven by the masked task observation of that mode (the same path the policy sees in training). Metrics follow
arXiv:2607.15163 App. B.2 and are computed ONLY over the links activated by the mode:

  Succ      fraction of clips where no activated link is ever farther than --fail_threshold (0.5 m) from its
            reference in the global frame
  G-MPKPE   mean over frames and activated links of the global position error (m)
  G-MPKRE   same for the rotation error (rad)
  L-MPKPE   position error after removing the horizontal offset and heading difference at the root (m)
  L-MPKRE   rotation error after removing the heading difference at the root (rad)

`global` tracking uses the measured robot root. `local` tracking emulates the deployment "reference forcing" of
ScaleBridge: the robot root position is replaced by the reference root position (3-D) and the measured link positions
are shifted rigidly with it, so the policy never sees horizontal drift; it is only defined for modes that include the
pelvis. All errors are computed from the TRUE simulated robot state, never through the overridden properties.

Time alignment follows TRAINING: there every reset happens inside env.step(), after which the command has advanced by
one step (target 0 is the frame AFTER the robot's current frame). `env.reset()` alone would leave the command one step
behind, i.e. the policy would track a reference delayed by 20 ms. So after each `env.reset()` the command is advanced
once and the observation recomputed, and errors compare the robot state at step k with reference frame k.

Results are deterministic given (checkpoint, motion set, --seed, --num_envs): clips are sorted by length and mapped to
env slots in a fixed order, so two checkpoints evaluated with identical arguments are paired clip by clip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a BFM checkpoint over control modes.")
parser.add_argument("--task", type=str, default="G1-BFM-Transformer-Tracking")
parser.add_argument("--checkpoint", type=str, required=True, help="Absolute path of the checkpoint (model_XXXX.pt).")
parser.add_argument("--motion_file", type=str, required=True, help="YAML listing the processed .npz clips to evaluate.")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--modes", type=int, nargs="+", default=list(range(8)))
parser.add_argument("--tracking", type=str, nargs="+", default=["global", "local"], choices=["global", "local"])
parser.add_argument("--max_clips", type=int, default=0, help="Evaluate a fixed random subset of this size (0 = all).")
parser.add_argument("--subset_seed", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=0, help="Cap the evaluated clip length in frames (0 = full).")
parser.add_argument("--min_frames", type=int, default=10, help="Skip clips shorter than this many frames.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fail_threshold", type=float, default=0.5)
parser.add_argument("--reset_disturbance", action="store_true", help="Keep the random reset disturbance of training.")
parser.add_argument("--no_dr", action="store_true", help="Disable startup domain randomization.")
parser.add_argument("--out", type=str, required=True, help="Output JSON with aggregate results.")
parser.add_argument("--per_clip", type=str, default=None, help="Optional .npz with per-clip results.")
parser.add_argument("--video_dir", type=str, default=None,
                    help="Record an mp4 of env 0 (the first clip of every batch) into this directory; use with a handful of clips.")
parser.add_argument("--video_stride", type=int, default=2, help="Record every n-th control step (50 Hz / n fps).")
parser.add_argument("--future_idx", type=int, nargs="+", default=None,
                    help="Override the future frame offsets of the actor's task observations. Training uses 0 1 2 3 4 -1 (-1 = last frame of the clip), "
                         "the deployment export (play_export_check_humanoid_transformer*.py) uses 0 1 2 3 4 5.")
parser.add_argument("--embedding_dim", type=int, default=None)
parser.add_argument("--num_heads", type=int, default=None)
parser.add_argument("--ff_dim", type=int, default=None)
parser.add_argument("--num_layers", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video_dir:
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
import yaml

from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

import scaletrack.tasks  # noqa: F401
from my_rsl_rl.runners.on_policy_runner import OnPolicyRunner


def make_eval_command_class(base_cls):
    """Subclass of MotionCommand with a switchable deployment-style reference forcing (`local_mode`).

    A plain subclass is required so that `cmd.__class__ = ...` keeps the same object layout as the base class.
    """

    class EvalMotionCommand(base_cls):
        local_mode: bool = False

        @property
        def robot_anchor_pos_w(self):
            if self.local_mode:
                return self.anchor_pos_w
            return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

        @property
        def robot_body_pos_w(self):
            pos = self.robot.data.body_pos_w[:, self.body_indexes]
            if self.local_mode:
                true_anchor = self.robot.data.body_pos_w[:, self.robot_anchor_body_index]
                pos = pos + (self.anchor_pos_w - true_anchor).unsqueeze(1)
            return pos

    return EvalMotionCommand


def per_link_errors(cmd):
    """Per-link errors (num_envs, num_links) of the true robot against the reference at the current time step."""
    rb_pos = cmd.robot.data.body_pos_w[:, cmd.body_indexes]
    rb_quat = cmd.robot.data.body_quat_w[:, cmd.body_indexes]
    ref_pos = cmd.body_pos_w
    ref_quat = cmd.body_quat_w
    num_links = ref_pos.shape[1]

    g_pos = torch.norm(ref_pos - rb_pos, dim=-1)
    g_rot = quat_error_magnitude(ref_quat, rb_quat)

    root_pos = cmd.robot.data.body_pos_w[:, cmd.robot_anchor_body_index]
    root_quat = cmd.robot.data.body_quat_w[:, cmd.robot_anchor_body_index]
    ref_root_pos = ref_pos[:, cmd.motion_anchor_body_index]
    ref_root_quat = ref_quat[:, cmd.motion_anchor_body_index]
    delta_ori = yaw_quat(quat_mul(root_quat, quat_inv(ref_root_quat)))[:, None].expand(-1, num_links, -1)
    delta_pos = root_pos.clone()
    delta_pos[:, 2] = ref_root_pos[:, 2]
    rel_pos = delta_pos[:, None] + quat_apply(delta_ori, ref_pos - ref_root_pos[:, None])
    l_pos = torch.norm(rel_pos - rb_pos, dim=-1)
    l_rot = quat_error_magnitude(quat_mul(delta_ori, ref_quat), rb_quat)
    return g_pos, g_rot, l_pos, l_rot


def file_sha1(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    out_dir = os.path.dirname(os.path.abspath(args_cli.out))
    os.makedirs(out_dir, exist_ok=True)

    motion_file = args_cli.motion_file
    if args_cli.max_clips:
        with open(motion_file) as f:
            motions = yaml.safe_load(f)
        names = sorted(motions.keys())
        rng = np.random.RandomState(args_cli.subset_seed)
        pick = sorted(rng.choice(len(names), min(args_cli.max_clips, len(names)), replace=False).tolist())
        subset = {names[i]: motions[names[i]] for i in pick}
        motion_file = os.path.join(out_dir, os.path.basename(args_cli.out) + f".subset{len(subset)}_s{args_cli.subset_seed}.yaml")
        with open(motion_file, "w") as f:
            yaml.safe_dump(subset, f, sort_keys=False)
        print(f"[eval] subset of {len(subset)} clips -> {motion_file}", flush=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    env_cfg.commands.motion.motion_file = motion_file
    env_cfg.commands.motion.enable_reset_disturbance = args_cli.reset_disturbance
    if args_cli.future_idx is not None:
        for term_name in ("target_body_pos", "target_body_pos_rel", "target_body_rot", "target_body_rot_rel", "timestamp"):
            getattr(env_cfg.observations.policy_task, term_name).params["future_idx"] = list(args_cli.future_idx)
        print(f"[eval] actor future offsets overridden: {args_cli.future_idx}", flush=True)
    env_cfg.commands.motion.debug_vis = bool(args_cli.video_dir)  # reference markers are drawn in the videos
    if args_cli.video_dir:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (2.8, 2.8, 1.4)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.6)
    if args_cli.no_dr:
        for name in ("physics_material", "add_joint_default_pos", "base_com", "hand_mass"):
            setattr(env_cfg.events, name, None)
    for key in ("embedding_dim", "num_heads", "ff_dim", "num_layers"):
        val = getattr(args_cli, key)
        if val is not None:
            setattr(agent_cfg.policy, key, val)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video_dir else None)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint, load_optimizer=False)
    runner._set_env_is_evaluating()  # no terminations except clip end, no pushes, no observation noise
    ac = runner.alg.policy
    ac.eval()

    unwrapped = env.unwrapped
    cmd = unwrapped.command_manager.get_term("motion")
    cmd.__class__ = make_eval_command_class(type(cmd))
    device = unwrapped.device
    mode_names = list(cmd.cfg.mode_candidates.keys())
    mode_table = cmd._mode_table.clone()  # (num_modes, num_links)
    anchor_idx = cmd.cfg.body_names.index(cmd.cfg.anchor_body_name)
    num_envs = args_cli.num_envs

    @torch.no_grad()
    def act(obs):
        prop, task, action = ac.get_actor_obs(obs, inference=False)  # applies the per-mode mask and mode vector
        return ac.actor(prop, action, ac.actor_task_embedder(task))

    lengths_all = cmd.time_totals.clone()
    order = torch.argsort(lengths_all, stable=True)
    order = order[lengths_all[order] >= args_cli.min_frames]
    num_clips = len(order)
    clip_names = [cmd.motion_names[i] for i in order.tolist()]
    print(f"[eval] {num_clips} clips, {int(lengths_all[order].sum())} frames, {num_envs} envs", flush=True)

    results = {}
    per_clip = {"names": np.array(clip_names)}
    t_start = time.time()

    for mode_idx in args_cli.modes:
        mode_row = mode_table[mode_idx]
        mask = mode_row.bool()
        mask_f = mode_row.float()
        for tracking in args_cli.tracking:
            if tracking == "local" and not bool(mask[anchor_idx]):
                continue
            cfg_name = f"mode{mode_idx}_{tracking}"
            cmd.local_mode = tracking == "local"
            cmd._mode_table = mode_row[None].clone()  # every reset re-samples exactly this mode
            cmd._mode[:] = mode_row

            succ = np.zeros(num_clips, dtype=bool)
            acc = {k: np.zeros(num_clips, dtype=np.float64) for k in ("g_pos", "g_rot", "l_pos", "l_rot", "max_g", "max_l", "fail_step", "fail_link")}
            t_cfg = time.time()
            for b0 in range(0, num_clips, num_envs):
                idx = order[b0 : b0 + num_envs]
                n = len(idx)
                cmd.motion_ids[:n] = idx
                cmd.motion_ids[n:] = idx[0]
                clip_len = lengths_all[cmd.motion_ids].to(device)
                if args_cli.max_steps:
                    clip_len = torch.clamp(clip_len, max=args_cli.max_steps + 2)
                last_counted = clip_len - 2  # no reset has happened before time step L-1
                max_iters = int(last_counted[:n].max())

                env.reset()
                cmd.time_steps += 1  # training convention (see module docstring)
                cmd._future_manual_cache.clear()
                cmd._mode[:] = mode_row
                obs = env.get_observations()
                elapsed = torch.zeros(num_envs, dtype=torch.long, device=device)
                sums = {k: torch.zeros(num_envs, device=device) for k in ("g_pos", "g_rot", "l_pos", "l_rot")}
                max_g = torch.zeros(num_envs, device=device)
                max_l = torch.zeros(num_envs, device=device)
                fail_step = torch.full((num_envs,), -1.0, device=device)  # first frame with a link > threshold away (global), -1 = never
                fail_link = torch.full((num_envs,), -1.0, device=device)
                counted = torch.zeros(num_envs, device=device)
                frames = []
                for step_i in range(max_iters):
                    obs, _, _, _ = env.step(act(obs))
                    if args_cli.video_dir and step_i % args_cli.video_stride == 0:
                        frames.append(np.asarray(unwrapped.render()).copy())
                    elapsed += 1
                    live = (elapsed <= last_counted).float()
                    cmd.time_steps -= 1  # robot state at step k <-> reference frame k
                    g_pos, g_rot, l_pos, l_rot = per_link_errors(cmd)
                    cmd.time_steps += 1
                    denom = mask_f.sum()
                    sums["g_pos"] += live * (g_pos * mask_f).sum(-1) / denom
                    sums["g_rot"] += live * (g_rot * mask_f).sum(-1) / denom
                    sums["l_pos"] += live * (l_pos * mask_f).sum(-1) / denom
                    sums["l_rot"] += live * (l_rot * mask_f).sum(-1) / denom
                    max_g = torch.maximum(max_g, live * g_pos.masked_fill(~mask, 0.0).amax(-1))
                    max_l = torch.maximum(max_l, live * l_pos.masked_fill(~mask, 0.0).amax(-1))
                    g_act = g_pos.masked_fill(~mask, 0.0)
                    newly = (live > 0) & (fail_step < 0) & (g_act.amax(-1) > args_cli.fail_threshold)
                    fail_step = torch.where(newly, elapsed.float(), fail_step)
                    fail_link = torch.where(newly, g_act.argmax(-1).float(), fail_link)
                    counted += live
                if args_cli.video_dir and frames:
                    import imageio

                    os.makedirs(args_cli.video_dir, exist_ok=True)
                    vpath = os.path.join(args_cli.video_dir, f"{cfg_name}_{clip_names[b0]}.mp4")
                    imageio.mimsave(vpath, frames, fps=int(round(50 / args_cli.video_stride)))
                    print(f"[eval] wrote {vpath} ({len(frames)} frames)", flush=True)
                counted = counted.clamp_min(1.0)
                sl = slice(b0, b0 + n)
                for k in ("g_pos", "g_rot", "l_pos", "l_rot"):
                    acc[k][sl] = (sums[k] / counted)[:n].double().cpu().numpy()
                acc["max_g"][sl] = max_g[:n].double().cpu().numpy()
                acc["max_l"][sl] = max_l[:n].double().cpu().numpy()
                acc["fail_step"][sl] = fail_step[:n].double().cpu().numpy()
                acc["fail_link"][sl] = fail_link[:n].double().cpu().numpy()
                succ[sl] = acc["max_g"][sl] <= args_cli.fail_threshold
                done = min(b0 + n, num_clips)
                print(f"[eval] {cfg_name}: {done}/{num_clips} clips, running succ {succ[:done].mean():.4f}, "
                      f"{time.time() - t_cfg:.0f}s", flush=True)

            def agg(v):
                return {"mean_all": float(v.mean()), "mean_success": float(v[succ].mean()) if succ.any() else float("nan")}

            results[cfg_name] = {
                "mode": mode_idx,
                "mode_name": mode_names[mode_idx],
                "tracking": tracking,
                "n_clips": int(num_clips),
                "success_rate": float(succ.mean()),
                "success_rate_local_error": float((acc["max_l"] <= args_cli.fail_threshold).mean()),  # no link ever > 0.5 m away from its target relative to the root
                "G-MPKPE": agg(acc["g_pos"]),
                "G-MPKRE": agg(acc["g_rot"]),
                "L-MPKPE": agg(acc["l_pos"]),
                "L-MPKRE": agg(acc["l_rot"]),
                "seconds": time.time() - t_cfg,
            }
            per_clip[f"{cfg_name}/succ"] = succ
            for k, v in acc.items():
                per_clip[f"{cfg_name}/{k}"] = v.astype(np.float32)
            r = results[cfg_name]
            print(f"[eval] === {cfg_name} ({r['mode_name']}): succ {r['success_rate']:.4f}  "
                  f"G-MPKPE {r['G-MPKPE']['mean_all']:.4f}  G-MPKRE {r['G-MPKRE']['mean_all']:.4f}  "
                  f"L-MPKPE {r['L-MPKPE']['mean_all']:.4f}  L-MPKRE {r['L-MPKRE']['mean_all']:.4f}", flush=True)

            summary = {
                "meta": {
                    "checkpoint": args_cli.checkpoint,
                    "checkpoint_sha1": file_sha1(args_cli.checkpoint),
                    "motion_file": motion_file,
                    "num_envs": num_envs,
                    "seed": args_cli.seed,
                    "max_steps": args_cli.max_steps,
                    "fail_threshold": args_cli.fail_threshold,
                    "reset_disturbance": args_cli.reset_disturbance,
                    "no_dr": args_cli.no_dr,
                    "alignment": "training (in-step reset convention)",
                    "torch": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0),
                    "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
                    "wall_seconds": time.time() - t_start,
                },
                "results": results,
            }
            with open(args_cli.out, "w") as f:
                json.dump(summary, f, indent=2)
            if args_cli.per_clip:
                np.savez_compressed(args_cli.per_clip, **per_clip)

    print(f"[eval] done in {time.time() - t_start:.0f}s -> {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
