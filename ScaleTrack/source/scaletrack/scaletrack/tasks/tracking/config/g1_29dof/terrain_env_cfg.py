"""Terrain fine-tuning task: flat rehearsal envs on the ground plane + terrain envs on recorded layouts.

Paths are taken from environment variables so that launch scripts can switch data sets without editing code:
  SCALETRACK_TERRAIN_ROOT   directory with `layouts/layout_<seed>/{terrain.npz,heightmap.npz}` (default terrain_raw)
  SCALETRACK_CLIP_META      JSON clip name -> layout seed
  SCALETRACK_LAYOUT_SEEDS   comma separated layout seeds to load (default 0,1,2,3,4,5)
  SCALETRACK_TERRAIN_FRAC   fraction of envs that replay terrain clips (default 0.35)
"""

import os
from dataclasses import fields

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import scaletrack.tasks.tracking.mdp as mdp
from scaletrack.tasks.tracking.config.g1_29dof.flat_env_cfg import G1BFMTrackingEnvCfg
from scaletrack.tasks.tracking.mdp.commands_terrain import LayoutMotionCommandCfg
from scaletrack.tasks.tracking.terrain_layouts import LayoutTerrainImporterCfg
from scaletrack.tasks.tracking.tracking_env_cfg import BFM_CONTEXT_SIZE

TERRAIN_ROOT = os.environ.get("SCALETRACK_TERRAIN_ROOT", "/home/vcj9002/scalebfm_ws/motions/terrain_raw")
CLIP_META = os.environ.get("SCALETRACK_CLIP_META", "/home/vcj9002/scalebfm_ws/motions/yaml/clip_meta.json")
LAYOUT_SEEDS = [int(x) for x in os.environ.get("SCALETRACK_LAYOUT_SEEDS", "0,1,2,3,4,5").split(",") if x != ""]
TERRAIN_FRAC = float(os.environ.get("SCALETRACK_TERRAIN_FRAC", "0.35"))

# Control modes of terrain envs: what a planner that provides foot targets drives (whole body, root + end effectors),
# plus small shares of a few sparser modes.
TERRAIN_MODE_PROBS = {
    "WholeBody-14": 0.65,
    "VR-5": 0.20,
    "UpperBody-Mobile-7": 0.05,
    "VR-3": 0.05,
    "UMI-4": 0.05,
}


@configclass
class GroupObsCfg(ObsGroup):
    env_group = ObsTerm(func=mdp.env_group, params={"command_name": "motion"})


@configclass
class G1BFMTerrainTrackingEnvCfg(G1BFMTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        old = self.scene.terrain
        self.scene.terrain = LayoutTerrainImporterCfg(
            prim_path=old.prim_path,
            terrain_type="plane",
            collision_group=-1,
            physics_material=old.physics_material,
            visual_material=old.visual_material,
            layout_root=TERRAIN_ROOT,
            layout_seeds=list(LAYOUT_SEEDS),
        )
        old_motion = self.commands.motion
        kwargs = {f.name: getattr(old_motion, f.name) for f in fields(old_motion) if f.name != "class_type"}
        motion = LayoutMotionCommandCfg(**kwargs)
        motion.debug_vis = False
        motion.clip_meta_file = CLIP_META
        motion.layout_root = TERRAIN_ROOT
        motion.layout_seeds = list(LAYOUT_SEEDS)
        motion.terrain_env_fraction = TERRAIN_FRAC
        motion.terrain_mode_probs = dict(TERRAIN_MODE_PROBS)
        self.commands.motion = motion

        # critic: height above the terrain instead of the world height (identical on the flat plane)
        self.observations.critic.root_height = ObsTerm(
            func=mdp.root_height_above_ground,
            params={"command_name": "motion"},
            history_length=BFM_CONTEXT_SIZE,
            flatten_history_dim=False,
        )
        self.observations.group = GroupObsCfg()
        # more contacts against mesh terrain
        self.sim.physx.gpu_max_rigid_patch_count = 2 * 10 * 2**15
