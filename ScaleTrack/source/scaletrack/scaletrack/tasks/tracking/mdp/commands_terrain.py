"""Motion command with two env groups: flat rehearsal clips on the ground plane and terrain clips on recorded layouts.

Compared with `MotionCommand` (which stays untouched and is what the original flat training uses) this subclass adds

* a fixed split of the envs into a *flat* group and a *terrain* group (`terrain_env_fraction`),
* per-group clip sampling (flat envs only replay flat clips, terrain envs only terrain clips) and per-group control-mode
  probabilities (terrain envs mostly use the modes a planner drives: whole body and root+end-effectors),
* per-env reference origins: a flat env adds its grid origin to the clip positions as before, a terrain env adds the world
  offset of the layout the clip was recorded on,
* a new clip is drawn when a clip has been played to its end (the original repeats the same clip), which keeps the
  coverage of a large library high; after a failure the env resumes the same clip where it failed, exactly as before,
* the ground height below the robot (critic observation) taken from the layout height grids.

Clips and layouts are tied together by a JSON file mapping clip name -> layout seed (missing / -1 = flat clip).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

from scaletrack.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from scaletrack.tasks.tracking.terrain_layouts import LayoutHeightSampler


class LayoutMotionCommand(MotionCommand):
    cfg: "LayoutMotionCommandCfg"

    local_forcing: torch.Tensor | None = None
    """(num_envs,) bool: envs whose observations use the deployment-style reference forcing (the reference root position replaces
    the measured one, all robot link positions shift rigidly with it); resampled at every reset with the per-group probabilities."""

    def __init__(self, cfg: "LayoutMotionCommandCfg", env):
        self._grouped = False
        super().__init__(cfg, env)  # loads the whole library; its final resample_motions() sees _grouped == False

        num_envs = self.num_envs
        layout_seeds = list(cfg.layout_seeds)
        meta = json.load(open(cfg.clip_meta_file)) if cfg.clip_meta_file else {}
        seed_to_idx = {s: i for i, s in enumerate(layout_seeds)}
        clip_layout, clip_max_step = [], []
        for name in self.motion_names_train:
            entry = meta.get(name, -1)
            seed = entry["layout"] if isinstance(entry, dict) else entry
            clip_max_step.append(float(entry.get("max_step", 0.0)) if isinstance(entry, dict) else 0.0)
            if seed >= 0 and seed not in seed_to_idx:
                raise ValueError(f"clip {name} belongs to layout {seed}, which is not loaded (layouts: {layout_seeds})")
            clip_layout.append(seed_to_idx[seed] if seed >= 0 else -1)
        self.clip_layout = torch.tensor(clip_layout, dtype=torch.long)  # CPU, (num_clips,)
        self.clip_max_step = torch.tensor(clip_max_step, dtype=torch.float)  # highest step on the clip's path (m)
        self.clip_is_terrain = self.clip_layout >= 0
        n_terrain_clips = int(self.clip_is_terrain.sum())
        n_flat_clips = int((~self.clip_is_terrain).sum())
        terrain_frac = cfg.terrain_env_fraction if n_terrain_clips > 0 else 0.0
        if n_flat_clips == 0:
            terrain_frac = 1.0
        n_terrain_envs = int(round(terrain_frac * num_envs))
        self.env_is_terrain = torch.zeros(num_envs, dtype=torch.bool)  # CPU
        if n_terrain_envs > 0:
            self.env_is_terrain[num_envs - n_terrain_envs :] = True
        self.env_is_terrain_dev = self.env_is_terrain.to(self.device)
        print(f"[terrain] {n_flat_clips} flat clips, {n_terrain_clips} terrain clips; "
              f"{num_envs - n_terrain_envs} flat envs, {n_terrain_envs} terrain envs", flush=True)

        # layouts
        importer = env.scene.terrain
        self.layout_offsets = torch.as_tensor(getattr(importer, "layout_offsets", np.zeros((0, 3))), dtype=torch.float32, device=self.device)
        self.ground = LayoutHeightSampler(cfg.layout_root, layout_seeds, self.layout_offsets.cpu().numpy(), self.device) if layout_seeds else None
        self.ref_origins = env.scene.env_origins.clone()
        self.env_layout = torch.full((num_envs,), -1, dtype=torch.long)  # CPU
        self.env_layout_dev = self.env_layout.to(self.device)

        # per-group control-mode sampling probabilities over the rows of the mode table
        names = list(self.cfg.mode_candidates.keys())
        p_flat = torch.ones(len(names), dtype=torch.float, device=self.device)
        p_terrain = torch.zeros(len(names), dtype=torch.float, device=self.device)
        if cfg.terrain_mode_probs:
            for n, p in cfg.terrain_mode_probs.items():
                p_terrain[names.index(n)] = p
        else:
            p_terrain[:] = 1.0
        self._mode_probs_flat = p_flat / p_flat.sum()
        self._mode_probs_terrain = p_terrain / p_terrain.sum()
        self.local_forcing = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        self._grouped = True
        self.resample_motions()

    # ------------------------------------------------------------------ sampling
    def terrain_step_cap(self) -> float:
        """Current step-height cap of the terrain curriculum (1e9 when the curriculum is off)."""
        c = self.cfg
        if c.terrain_curriculum_steps <= 0:
            return 1e9
        frac = min(1.0, float(self._env.common_step_counter) / float(c.terrain_curriculum_steps))
        return c.terrain_step_cap_init + frac * (c.terrain_step_cap_final - c.terrain_step_cap_init)

    def _sample_clips(self, terrain: bool, n: int) -> torch.Tensor:
        mask = self.clip_is_terrain if terrain else ~self.clip_is_terrain
        w = self.motion_sampling_prob * mask.float()
        if terrain:
            allowed = w * (self.clip_max_step <= self.terrain_step_cap()).float()
            if allowed.sum() > 0:  # clips of low relief (flat, slopes, rough) have max_step 0 and are always allowed
                w = allowed
            if self.cfg.terrain_hard_boost > 0:  # emphasise the clips with high steps (their success lags behind)
                difficulty = ((self.clip_max_step - 0.10) / 0.15).clamp(0.0, 1.0)
                w = w * (1.0 + self.cfg.terrain_hard_boost * difficulty)
        return torch.multinomial(w, num_samples=n, replacement=True)

    def resample_motions(self):
        if not self._grouped:
            return super().resample_motions()
        ids = torch.zeros(self.num_envs, dtype=torch.long)
        n_terrain = int(self.env_is_terrain.sum())
        if n_terrain < self.num_envs:
            ids[~self.env_is_terrain] = self._sample_clips(False, self.num_envs - n_terrain)
        if n_terrain > 0:
            ids[self.env_is_terrain] = self._sample_clips(True, n_terrain)
        self.motion_ids[:] = ids
        self._sync_env_layout(torch.arange(self.num_envs))

    def _sync_env_layout(self, env_ids_cpu: torch.Tensor):
        """Refresh the layout index and the reference origin of the given envs from their current clips."""
        lay = self.clip_layout[self.motion_ids[env_ids_cpu]]
        self.env_layout[env_ids_cpu] = lay
        lay_dev = lay.to(self.device)
        ids_dev = env_ids_cpu.to(self.device)
        self.env_layout_dev[ids_dev] = lay_dev
        grid = self._env.scene.env_origins[ids_dev]
        if self.layout_offsets.shape[0] > 0:
            off = self.layout_offsets[lay_dev.clamp_min(0)]
            self.ref_origins[ids_dev] = torch.where((lay_dev >= 0)[:, None], off, grid)
        else:
            self.ref_origins[ids_dev] = grid

    # ------------------------------------------------------------------ reference with per-env origins
    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.cat_body_pos_w[self._global_time_index()].to(self.device) + self.ref_origins[:, None, :]

    @property
    def body_pos_w_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        global_indices = self._global_time_indices(frame_offsets)
        return self.cat_body_pos_w[global_indices].to(self.device) + self.ref_origins[:, None, None, :]

    @property
    def anchor_pos_w_future(self) -> torch.Tensor:
        frame_offsets = self._future_frame_offsets()
        global_indices = self._global_time_indices(frame_offsets)
        pos = self.cat_body_pos_w[global_indices, self.motion_anchor_body_index].to(self.device)
        return pos + self.ref_origins[:, None, :]

    def body_pos_w_future_manual(self, future_idx: Sequence[int]) -> torch.Tensor:
        key = tuple(future_idx)
        cache = self._future_manual_cache.setdefault(key, {})
        if "body_pos_w_future" not in cache:
            global_indices = cache.get("global_indices")
            if global_indices is None:
                global_indices = self._global_time_indices_manual(future_idx)
                cache["global_indices"] = global_indices
                cache["frame_offsets"] = self._future_frame_offsets_manual(future_idx)
            body_pos_all = self._gather_cat_by_global_indices(self.cat_body_pos_w, global_indices).to(self.device)
            cache["body_pos_w_future"] = body_pos_all + self.ref_origins[:, None, None, :]
        return cache["body_pos_w_future"]

    # ------------------------------------------------------------------ reference forcing (deployment style)
    def _ref_anchor_pos_cached(self) -> torch.Tensor:
        """Reference anchor position of the current time step (cached like the future frames: cleared when time_steps change)."""
        cache = self._future_manual_cache.setdefault(("anchor",), {})
        if "pos" not in cache:
            cache["pos"] = self.anchor_pos_w
        return cache["pos"]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        true = self.robot.data.body_pos_w[:, self.robot_anchor_body_index]
        if self.local_forcing is None:
            return true
        return torch.where(self.local_forcing[:, None], self._ref_anchor_pos_cached(), true)

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        pos = self.robot.data.body_pos_w[:, self.body_indexes]
        if self.local_forcing is None:
            return pos
        true_anchor = self.robot.data.body_pos_w[:, self.robot_anchor_body_index]
        shift = (self._ref_anchor_pos_cached() - true_anchor) * self.local_forcing[:, None].float()
        return pos + shift.unsqueeze(1)

    # ------------------------------------------------------------------ terrain queries
    def ground_height_below(self, xy_w: torch.Tensor) -> torch.Tensor:
        """Ground height (world z) below world xy of every env; 0 for envs on the flat plane."""
        if self.ground is None:
            return torch.zeros(xy_w.shape[0], device=xy_w.device)
        return self.ground.height(xy_w, self.env_layout_dev)

    # ------------------------------------------------------------------ reset
    def _resample_command(self, env_ids: Sequence[int]):
        if not self._grouped:
            return super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return

        env_ids_cpu = env_ids.cpu()
        if not self.is_evaluating:
            motion_len = self.time_totals.gather(0, self.motion_ids[env_ids_cpu])
            if self.randomize_next_resampling:
                phase = torch.rand(self.motion_ids[env_ids_cpu].shape)
                self.time_steps[env_ids_cpu] = (phase * (motion_len.float() - 1)).long()
                self.randomize_next_resampling = False
            else:
                ended = self.time_steps[env_ids_cpu] >= motion_len - 1
                nxt = (self.time_steps[env_ids_cpu] + 1) % motion_len  # after a failure: resume where it failed
                if bool(ended.any()):
                    ended_ids = env_ids_cpu[ended]
                    is_t = self.env_is_terrain[ended_ids]
                    new = torch.zeros(len(ended_ids), dtype=torch.long)
                    if bool((~is_t).any()):
                        new[~is_t] = self._sample_clips(False, int((~is_t).sum()))
                    if bool(is_t.any()):
                        new[is_t] = self._sample_clips(True, int(is_t.sum()))
                    self.motion_ids[ended_ids] = new  # a finished clip is replaced by a fresh one, played from frame 0
                    nxt = torch.where(ended, torch.zeros_like(nxt), nxt)
                self.time_steps[env_ids_cpu] = nxt
        else:
            self.time_steps[env_ids_cpu] = 0
        self._sync_env_layout(env_ids_cpu)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()
        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        if self.cfg.enable_reset_disturbance:
            range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_pos[env_ids] += rand_samples[:, 0:3]
            orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
            range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
            ranges = torch.tensor(range_list, device=self.device)
            rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
            root_lin_vel[env_ids] += rand_samples[:, :3]
            root_ang_vel[env_ids] += rand_samples[:, 3:]

            joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
            soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos[env_ids] = torch.clip(joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])

        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

        if self.cfg.mode_candidates:
            if self._mode_table.shape[0] == self._mode_probs_flat.shape[0]:
                probs = torch.where(
                    self.env_is_terrain_dev[env_ids][:, None], self._mode_probs_terrain[None], self._mode_probs_flat[None]
                )
            else:  # the mode table was restricted (e.g. a single fixed mode during evaluation): uniform over its rows
                probs = torch.ones(len(env_ids), self._mode_table.shape[0], device=self.device)
            sampled_mode_ids = torch.multinomial(probs, 1).squeeze(-1)
            self._mode[env_ids] = self._mode_table[sampled_mode_ids].clone()
        else:
            self._mode[env_ids] = torch.bernoulli(
                torch.ones(len(env_ids), len(self.cfg.body_names), dtype=torch.float32, device=self.device) * 0.5
            )

        if self.local_forcing is not None:
            # reference forcing needs a reference root: only for envs whose control mode activates the anchor link (the pelvis)
            p_local = torch.where(
                self.env_is_terrain_dev[env_ids],
                torch.full((len(env_ids),), self.cfg.local_forcing_prob_terrain, device=self.device),
                torch.full((len(env_ids),), self.cfg.local_forcing_prob_flat, device=self.device),
            )
            anchor_active = self._mode[env_ids][:, self.cfg.body_names.index(self.cfg.anchor_body_name)] > 0.5
            self.local_forcing[env_ids] = (torch.rand(len(env_ids), device=self.device) < p_local) & anchor_active

        self._rand_timestep[env_ids_cpu] = torch.randint(
            low=self.cfg.rand_timestep_range[0], high=self.cfg.rand_timestep_range[1], size=(len(env_ids_cpu), 1), requires_grad=False
        )
        self._future_manual_cache.clear()


@configclass
class LayoutMotionCommandCfg(MotionCommandCfg):
    class_type: type = LayoutMotionCommand

    clip_meta_file: str = ""
    """JSON mapping clip name -> layout seed of the terrain it was recorded on (missing or -1: flat clip)."""
    layout_root: str = ""
    layout_seeds: list[int] = []
    """Layout seeds loaded by the terrain importer, in the same order."""
    terrain_env_fraction: float = 0.35
    terrain_mode_probs: dict[str, float] = {}
    """Control-mode probabilities of terrain envs by mode name (empty: uniform over all modes)."""
    local_forcing_prob_flat: float = 0.0
    """Probability that a flat-group env runs an episode with reference forcing (0: never, e.g. evaluation)."""
    local_forcing_prob_terrain: float = 0.0
    """Same for terrain-group envs."""
    terrain_hard_boost: float = 0.0
    """Sampling weight of a terrain clip is 1 + boost * clamp((max_step - 0.10) / 0.15, 0, 1): 0 = uniform."""
    terrain_curriculum_steps: int = 0
    """Env steps (per process) over which the step-height cap of the terrain clips ramps up; 0 disables the curriculum."""
    terrain_step_cap_init: float = 0.14
    terrain_step_cap_final: float = 1.0
