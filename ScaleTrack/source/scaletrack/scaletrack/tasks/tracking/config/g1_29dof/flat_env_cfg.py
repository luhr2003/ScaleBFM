import os

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from scaletrack.robots.g1_29dof import G1_29DOF_ACTION_SCALE, G1_29DOF_CYLINDER_CFG

from scaletrack.tasks.tracking.tracking_env_cfg import (
    BFMTrackingEnvCfg,
)

@configclass
class G1BFMTrackingEnvCfg(BFMTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_29DOF_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot_usd = os.environ.get("SCALETRACK_ROBOT_USD")
        if robot_usd:  # sim-gap experiments: same articulation, other collision geometry (e.g. MagicSim-style disc feet)
            self.scene.robot = self.scene.robot.replace(spawn=self.scene.robot.spawn.replace(usd_path=robot_usd))
        robot_usd_mix = os.environ.get("SCALETRACK_ROBOT_USD_MIX")
        if robot_usd_mix:
            # Several collision-geometry variants of the robot (comma separated .usda paths with the same articulation and the same
            # number of collision shapes per link), one chosen at random per environment: robustness to the foot model.
            spawn = self.scene.robot.spawn
            self.scene.robot = self.scene.robot.replace(
                spawn=sim_utils.MultiUsdFileCfg(
                    usd_path=robot_usd_mix.split(","),
                    random_choice=True,
                    activate_contact_sensors=spawn.activate_contact_sensors,
                    rigid_props=spawn.rigid_props,
                    articulation_props=spawn.articulation_props,
                )
            )
            self.scene.replicate_physics = False  # required when the environments do not share one asset
        self.actions.joint_pos.scale = G1_29DOF_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

        self.commands.motion.mode_candidates = {
            "Pelvis-1": ["pelvis"],
            "UMI-2": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
            "VR-3": ["pelvis","left_wrist_yaw_link","right_wrist_yaw_link"],
            "UMI-4": ["left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"],
            "VR-5": ["pelvis","left_wrist_yaw_link","right_wrist_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"],
            "UpperBody-6": [
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_yaw_link",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_yaw_link"
            ],
            "UpperBody-Mobile-7": [
                "pelvis",
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_yaw_link",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_yaw_link"
            ],
            "WholeBody-14":  [
                "pelvis",
                "left_hip_roll_link",
                "left_knee_link",
                "left_ankle_roll_link",
                "right_hip_roll_link",
                "right_knee_link",
                "right_ankle_roll_link",
                "torso_link",
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_yaw_link",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_yaw_link",
            ]
        }