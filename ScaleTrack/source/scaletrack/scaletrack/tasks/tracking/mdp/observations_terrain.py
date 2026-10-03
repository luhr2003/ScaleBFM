"""Observation terms that need the terrain-aware motion command (`LayoutMotionCommand`)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_DBG = {"n": 0}


def root_height_above_ground(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Pelvis height above the terrain surface below it (critic only). Equals the world z on the flat plane."""
    command = env.command_manager.get_term(command_name)
    pos = command.robot.data.body_pos_w[:, command.robot_anchor_body_index]  # the TRUE pelvis (reference-forced envs report the reference)
    return (pos[:, 2] - command.ground_height_below(pos[:, :2]))[:, None]


def env_group(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """0.0 for the flat rehearsal group, 1.0 for the terrain group, 2.0 for flat envs that currently play an anchor-free clip
    (squat references). Used by the anchor loss (only value 0 is anchored) and for logging (values > 0.5 are logged as 'terrain')."""
    command = env.command_manager.get_term(command_name)
    g = command.env_is_terrain_dev.float()
    free_dev = getattr(command, "clip_anchor_free_dev", None)
    if free_dev is not None:
        free = free_dev[command.motion_ids.to(free_dev.device)] & ~command.env_is_terrain_dev
        g = torch.where(free, torch.full_like(g, 2.0), g)
    if os.environ.get("SCALETRACK_DEBUG_GROUP"):
        _DBG["n"] += 1
        if _DBG["n"] % 200 == 1:
            print(f"[group] envs per group value 0 / 1 / 2: {[int((g == v).sum()) for v in (0.0, 1.0, 2.0)]}", flush=True)
    return g[:, None]
