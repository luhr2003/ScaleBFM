import os
import time
import torch
import imageio
import threading
import mujoco
import mujoco.viewer as mjv
import numpy as np
from loguru import logger
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from scalebridge.simulator.base_simulator import BaseSimulator
from scalebridge.utils.merge_robot_object_xml import merge_robot_object_xml

def draw_marker(pos,v):
    geom = v.user_scn.geoms[v.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.03,0.03,0.03],
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=[1,0,0,1]
    )
    v.user_scn.ngeom += 1

from scalebridge.utils.legged_estimator.depth_ground import D435_PITCH, D435_POSITION


class MujocoSimulator(BaseSimulator):
    def __init__(self, config, metadata_dict):
        
        self.record_video = config.get('record_video', False)
        self.marker = config.get('marker', False)
        self.use_joystick = config.get('joystick', False)
        self.camera_follow = config.get('camera_follow', False) or self.record_video
        self.estimate_root_pos = config.get('estimate_root_pos', False)
        self.lidar_odom = config.get('lidar_odom', None)  # simulated FAST-LIO pose stream fused into the estimator
        self.depth_ground = config.get('depth_ground', None)  # simulated D435 depth camera: ground height under the feet

        super().__init__(config, metadata_dict)

        self._setup_joystick()

    def _setup_backbone(self):
        super()._setup_backbone()

        xml_path = self.cfg.asset.xml_path

        # pase object from metadata if exists
        self.has_object = "object_names" in self.metadata_dict
        if self.has_object:
            xml_path = merge_robot_object_xml(xml_path, self.metadata_dict)

        if self.depth_ground:
            # Mount the D435i on the torso as in the URDF (d435_joint): 47.6 deg downward pitch, MuJoCo camera axes x right, y up.
            spec = mujoco.MjSpec.from_file(xml_path)
            pitch = D435_PITCH
            spec.body('torso_link').add_camera(name='d435', pos=list(D435_POSITION), xyaxes=[0.0, -1.0, 0.0, np.sin(pitch), 0.0, np.cos(pitch)], fovy=float(self.depth_ground.get('fovy', 58.0)))
            self.mujoco_model = spec.compile()
        else:
            self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        self.mujoco_model.opt.timestep=self.low_dt
        
        self.viewer = mjv.launch_passive(
            model=self.mujoco_model,
            data=self.mujoco_data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.marker_pos = None
        if self.record_video:
            save_dir = HydraConfig.get().runtime.output_dir
            video_name = os.path.join(save_dir, 'recording.mp4')
            self.video_writer = imageio.get_writer(video_name, fps=50)
            self.renderer = mujoco.Renderer(self.mujoco_model, height=480, width=640)
            self.render_scene = mujoco.MjvScene(self.mujoco_model, maxgeom=1000)


    def _setup_asset(self):
        super()._setup_asset()

        self.default_qpos = self.mujoco_data.qpos.copy()
        self.default_qvel = self.mujoco_data.qvel.copy()

        self.default_dof_pos = self.mujoco_data.qpos[7:7+self.num_joints].copy()

        self.root_estimator = None
        if self.estimate_root_pos and self.metadata_dict.get("enable_root_localization", False):
            self._setup_root_estimator()

    def _setup_root_estimator(self):
        # Run the onboard state estimator on simulated sensors so global tracking can be checked without ground truth.
        from scalebridge.utils.legged_estimator import LeggedStateEstimator

        localization_cfg = self.cfg.asset.localization_module
        assert "estimator" in localization_cfg, "simulator.config.estimate_root_pos requires localization=legged_estimator."
        self.root_estimator = LeggedStateEstimator(**localization_cfg.estimator)

        sim_joint_names = self._get_joint_names()
        self.estimator_joint_idx = np.array([sim_joint_names.index(name) for name in self.root_estimator.joint_names], dtype=np.int64)
        self.imu_site_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SITE, self.cfg.get('imu_site', 'imu_in_pelvis'))
        self.imu_acc = np.zeros(6)
        self.root_pos_offset = np.zeros(3)
        self.estimator_log_counter = 0
        logger.info(f'[Simulator] Root position comes from the legged state estimator instead of the ground truth.')
        if self.lidar_odom:
            self._setup_lidar_odometry()
        if self.depth_ground:
            self._setup_depth_ground()

    def _setup_depth_ground(self):
        from scalebridge.utils.legged_estimator import CameraIntrinsics, DepthGroundHeight
        from scalebridge.utils.legged_estimator.depth_ground import d435_in_torso

        cfg = self.depth_ground
        width, height = int(cfg.get('width', 320)), int(cfg.get('height', 240))
        intrinsics = CameraIntrinsics.from_fovy(cfg.get('fovy', 58.0), width, height)
        self.depth_fusion = DepthGroundHeight(self.root_estimator, intrinsics, d435_in_torso())
        self.depth_renderer = mujoco.Renderer(self.mujoco_model, height=height, width=width)
        self.depth_renderer.enable_depth_rendering()
        self.depth_period = max(1, int(round(1.0 / cfg.get('hz', 10.0) / self.low_dt)))
        self.depth_noise = cfg.get('noise_frac', 0.0)
        self.depth_rng = np.random.default_rng(cfg.get('seed', 0))
        self.depth_step = 0
        logger.info(f'[Simulator] Simulated D435 depth camera fused: {dict(cfg)}')

    def _update_depth_ground(self):
        self.depth_step += 1
        if self.depth_step % self.depth_period:
            return
        self.depth_renderer.update_scene(self.mujoco_data, camera='d435')
        depth = self.depth_renderer.render().copy()
        if self.depth_noise:
            depth = depth * (1.0 + self.depth_rng.normal(0.0, self.depth_noise, depth.shape))
        self.depth_fusion.update(depth)

    def _setup_lidar_odometry(self):
        # Emulate FAST-LIO on the Mid-360: the pose of the LiDAR body in an odom frame with arbitrary origin and heading,
        # at lidar_odom.hz, with white noise and a slow drift, fed through the same fusion code as the real robot.
        import pinocchio as pin
        from scalebridge.utils.legged_estimator import LidarOdometryFusion

        cfg = self.lidar_odom
        self.lidar_fusion = LidarOdometryFusion(self.root_estimator, cfg.get('frame', 'mid360_link'), latency=cfg.get('latency', 0.0), gate=cfg.get('gate', 0.3), realign_after=cfg.get('realign_after', 1.0))
        self.lidar_queue = []  # (delivery time, position, quat): the bridge delivers each pose `delay` seconds late
        self.lidar_delay = cfg.get('delay', 0.0)
        self.lidar_jump = cfg.get('jump', None)  # e.g. {time: 8.0, dz: 0.2, dxy: 0.0}: the odometry jumps once, as on a relocalization
        estimator_model = self.root_estimator.model
        estimator_model.update()
        torso = estimator_model.get_frame_placement(estimator_model.model.getFrameId('torso_link'))
        lidar = estimator_model.get_frame_placement(self.lidar_fusion.frame_id)
        relative = torso.inverse() * lidar
        self.lidar_in_torso = (relative.translation.copy(), relative.rotation.copy())
        self.torso_body_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
        self.lidar_period = max(1, int(round(1.0 / cfg.get('hz', 10.0) / self.low_dt)))
        self.lidar_rng = np.random.default_rng(cfg.get('seed', 0))
        yaw, shift = cfg.get('odom_yaw', 0.7), np.array(cfg.get('odom_origin', [3.0, -2.0, 1.0]), dtype=np.float64)
        self.lidar_odom_rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        self.lidar_odom_shift = shift
        direction = self.lidar_rng.normal(size=2)
        self.lidar_drift_xy = cfg.get('xy_drift', 0.0) * direction / np.linalg.norm(direction)
        self.lidar_drift_z = cfg.get('z_drift', 0.0)
        self.lidar_sigma = cfg.get('sigma', 0.01)
        self.lidar_step = 0
        self.lidar_t0 = 0.0
        logger.info(f'[Simulator] Simulated LiDAR odometry fused: {dict(cfg)}')

    def _update_lidar_odometry(self):
        self.lidar_step += 1
        if self.lidar_step % self.lidar_period:
            return
        data = self.mujoco_data
        rotation_torso = data.xmat[self.torso_body_id].reshape(3, 3)
        position = data.xpos[self.torso_body_id] + rotation_torso @ self.lidar_in_torso[0]
        rotation = rotation_torso @ self.lidar_in_torso[1]
        t = data.time - self.lidar_t0
        position = self.lidar_odom_rotation @ position + self.lidar_odom_shift
        position = position + np.array([*(self.lidar_drift_xy * t), self.lidar_drift_z * t]) + self.lidar_rng.normal(0.0, self.lidar_sigma, 3)
        if self.lidar_jump and t >= self.lidar_jump['time']:
            position = position + np.array([self.lidar_jump.get('dxy', 0.0), 0.0, self.lidar_jump.get('dz', 0.0)])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (self.lidar_odom_rotation @ rotation).flatten())
        self.lidar_queue.append((data.time + self.lidar_delay, position, quat))
        while self.lidar_queue and self.lidar_queue[0][0] <= data.time:
            _, delivered_position, delivered_quat = self.lidar_queue.pop(0)
            self.lidar_fusion.update(delivered_position, delivered_quat)

    def _update_root_estimator(self, qpos, qvel):
        # mj_step integrates qpos/qvel but leaves kinematics, accelerations and actuator forces at the pre-step state,
        # so the accelerometer reading (specific force at the IMU site, IMU frame) pairs with the qpos/qvel sampled before it.
        mujoco.mj_rnePostConstraint(self.mujoco_model, self.mujoco_data)
        mujoco.mj_objectAcceleration(self.mujoco_model, self.mujoco_data, mujoco.mjtObj.mjOBJ_SITE, self.imu_site_id, self.imu_acc, 1)
        joint_idx = self.estimator_joint_idx
        self.root_estimator.update(
            self.low_dt,
            qpos[7:7+self.num_joints][joint_idx],
            qvel[6:6+self.num_joints][joint_idx],
            self.mujoco_data.actuator_force[joint_idx],
            quat_wxyz=qpos[3:7],
            gyro=qvel[3:6],
            acc=self.imu_acc[3:],
        )

    def _reset_root_estimator(self):
        qpos, qvel = self.mujoco_data.qpos, self.mujoco_data.qvel
        joint_idx = self.estimator_joint_idx
        self.root_estimator.set_sensors(
            qpos[7:7+self.num_joints][joint_idx], qvel[6:6+self.num_joints][joint_idx], np.zeros(len(joint_idx)), quat_wxyz=qpos[3:7], gyro=qvel[3:6]
        )
        self.root_estimator.reset()
        if self.lidar_odom:
            self.lidar_fusion.reset_alignment()
            self.lidar_queue.clear()
            self.lidar_t0 = self.mujoco_data.time
        if self.depth_ground:
            self.depth_fusion.reset()
        # Anchor the estimate to the simulated start in xy (matters for reference state initialization); keep its own height.
        self.root_pos_offset[:2] = qpos[:2] - self.root_estimator.position[:2]

    def _log_root_estimation_error(self, root_pos):
        self.estimator_log_counter += 1
        if self.estimator_log_counter % 50 == 0:
            error = root_pos - self.mujoco_data.qpos[:3]
            logger.info(f'[Simulator] Root estimate error: xy {np.linalg.norm(error[:2]):.3f} m, z {error[2]:+.3f} m')

    def _setup_joystick(self):
        if self.use_joystick:
            import pygame
            logger.info(f'[Simulator] Using Joystick as command sender; Make sure you have connected the joystick to the PC!')
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                logger.info(f"[Simulator] Joystick detected: {self.joystick.get_name()}")
            else:
                pygame.quit()
                logger.error(f"[Simulator] Joystick undetected!")
            self.lin_vel_x_tmp = 0
            self.lin_vel_y_tmp = 0
            self.ang_vel_z_tmp = 0
            self.joystick_thread = threading.Thread(target=self._handle_joystick, args=(pygame,),daemon=True)
            self.joystick_thread.start()
    
    def _handle_joystick(self, pygame):
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        break
                self.lin_vel_x_tmp = self.joystick.get_axis(1) * -1
                self.lin_vel_y_tmp = self.joystick.get_axis(0) * -1
                self.ang_vel_z_tmp = self.joystick.get_axis(3) * -1
                time.sleep(0.001)
        except:
            pygame.quit()

    def _render(self):
        if not self.viewer:
            logger.info(f"[Simulator] Viewer has been closed; Automatically exit!")
            exit()

        if self.marker_pos is not None and self.marker:
            self.viewer.user_scn.ngeom = 0
            for i in range(self.marker_pos.shape[0]):
                draw_marker(
                    self.marker_pos[i], self.viewer
                )
            if self.record_video:
                self.render_scene.ngeom = 0
                for i in range(self.marker_pos.shape[0]):
                    geom = self.render_scene.geoms[self.render_scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.03, 0.03, 0.03],
                        pos=self.marker_pos[i],
                        mat=np.eye(3).flatten(),
                        rgba=[1,0,0,1]
                    )
                    self.render_scene.ngeom += 1
        if self.camera_follow:
            self.viewer.cam.lookat = self.mujoco_data.qpos[:3].copy()
            self.viewer.cam.elevation = 0
            self.viewer.cam.azimuth = 180
            self.viewer.cam.distance = 3.0

        self.viewer.sync()

        if self.record_video:
            self.renderer.update_scene(self.mujoco_data, camera=self.viewer.cam)
        
            # Then manually add markers to the renderer's scene
            if self.marker_pos is not None and self.marker:
                # The renderer's scene might need to be updated after update_scene
                # So we add markers after the update
                for i in range(self.marker_pos.shape[0]):
                    geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.03, 0.03, 0.03],
                        pos=self.marker_pos[i],
                        mat=np.eye(3).flatten(),
                        rgba=[1,0,0,1]
                    )
                    self.renderer.scene.ngeom += 1

            img = self.renderer.render()
            self.video_writer.append_data(img)

    def update_marker_pos(self, marker_pos):
        self.marker_pos = marker_pos.cpu().numpy()

    def refresh_sim(self):

        root_pos = self.mujoco_data.qpos[:3]
        if self.root_estimator is not None:
            root_pos = self.root_estimator.position + self.root_pos_offset
            self._log_root_estimation_error(root_pos)

        state_dict = { # weishuai: FIXME
            "root_pos": root_pos,
            "root_quat_wxyz": self.mujoco_data.qpos[3:7],
            "base_ang_vel": self.mujoco_data.qvel[3:6],
            "dof_pos": self.mujoco_data.qpos[7:7+self.num_joints][self.sim_to_env_joint_idx],
            "dof_vel": self.mujoco_data.qvel[6:6+self.num_joints][self.sim_to_env_joint_idx],
        }

        if self.has_object:
            object_root_pose = self.mujoco_data.qpos[7+self.num_joints:].reshape(-1, 7)
            state_dict.update({
                "object_root_pos": object_root_pose[:, :3],
                "object_root_quat_wxyz": object_root_pose[:, 3:7],
            })

        if self.use_joystick:
            self.commands = np.array([self.lin_vel_x_tmp, self.lin_vel_y_tmp, self.ang_vel_z_tmp], dtype=np.float32)
            self.commands = np.where(np.abs(self.commands) < 0.05, 0, self.commands)
            state_dict.update({'commands': self.commands})
        
        return {k:torch.from_numpy(v).float() for k,v in state_dict.items()}

    def calibrate(self, init_state_dict = {}):
        init_qpos = self.default_qpos.copy()
        init_qvel = self.default_qvel.copy()

        if "root_pos" in init_state_dict:
            init_qpos[:3] = init_state_dict["root_pos"]
        if "root_quat" in init_state_dict:
            init_qpos[3:7] = init_state_dict["root_quat"]
        if "dof_pos" in init_state_dict:
            init_qpos[7:7+self.num_joints][self.env_action_to_sim_idx] = init_state_dict["dof_pos"]
        if "root_lin_vel" in init_state_dict:
            init_qvel[:3] = init_state_dict["root_lin_vel"]
        if "root_ang_vel" in init_state_dict:
            init_qvel[3:6] = init_state_dict["root_ang_vel"]
        if "dof_vel" in init_state_dict:
            init_qvel[6:6+self.num_joints][self.env_action_to_sim_idx] = init_state_dict["dof_vel"]
        
        if self.has_object:
            
            init_obj_pose = init_qpos[7+self.num_joints:].copy().reshape(-1, 7)
            init_obj_vel = init_qvel[6+self.num_joints:].copy().reshape(-1, 6)
            
            if "object_root_pos" in init_state_dict:
                init_obj_pose[:, :3] = init_state_dict["object_root_pos"]
            if "object_root_quat" in init_state_dict:
                init_obj_pose[:, 3:7] = init_state_dict["object_root_quat"]
            if "object_root_lin_vel" in init_state_dict:
                init_obj_vel[:, :3] = init_state_dict["object_root_lin_vel"]
            if "object_root_ang_vel" in init_state_dict:
                init_obj_vel[:, 3:] = init_state_dict["object_root_ang_vel"]

            init_qpos[7+self.num_joints:] = init_obj_pose.flatten()
            init_qvel[6+self.num_joints:] = init_obj_vel.flatten()

        self.mujoco_data.qpos[:] = init_qpos
        self.mujoco_data.qvel[:] = init_qvel
        self.mujoco_data.ctrl[:] = 0

        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)

        if self.root_estimator is not None:
            self._reset_root_estimator()

        return init_qpos[:3], init_qpos[3:7]

    def apply_action(self, tgt_dof_pos):
        tgt_dof_pos = tgt_dof_pos.squeeze()

        target_dof_pos_in_sim = self.default_dof_pos.copy()
        target_dof_pos_in_sim[self.env_action_to_sim_idx] = tgt_dof_pos

        for _ in range(self.decimation):
            torque = (target_dof_pos_in_sim - self.mujoco_data.qpos[7:7+self.num_joints]) * self.stiffness - self.mujoco_data.qvel[6:6+self.num_joints] * self.damping # no clip here
            # torque = np.clip(torque, -self.torque_limit, self.torque_limit)
            self.mujoco_data.ctrl[:] = torque # weishuai: no clip applied here
            if self.root_estimator is not None:
                qpos, qvel = self.mujoco_data.qpos.copy(), self.mujoco_data.qvel.copy()
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)
            if self.root_estimator is not None:
                self._update_root_estimator(qpos, qvel)
                if self.lidar_odom:
                    self._update_lidar_odometry()
                if self.depth_ground:
                    self._update_depth_ground()

        self._render()

    def _get_joint_names(self):
        joint_names = []
        for i in range(self.mujoco_model.nu):
            dof_name = mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            joint_names.append(dof_name)
        return joint_names
