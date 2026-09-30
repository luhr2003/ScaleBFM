"""Observation terms that need the terrain-aware motion command (`LayoutMotionCommand`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def root_height_above_ground(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Pelvis height above the terrain surface below it (critic only). Equals the world z on the flat plane."""
    command = env.command_manager.get_term(command_name)
    pos = command.robot.data.body_pos_w[:, command.robot_anchor_body_index]  # the TRUE pelvis (reference-forced envs report the reference)
    return (pos[:, 2] - command.ground_height_below(pos[:, :2]))[:, None]


def env_group(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """1.0 for envs of the terrain group, 0.0 for the flat rehearsal group (used by the anchor loss and for logging)."""
    command = env.command_manager.get_term(command_name)
    return command.env_is_terrain_dev.float()[:, None]
