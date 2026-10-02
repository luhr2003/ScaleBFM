import select
import threading
import time
from collections import deque

import lcm
import numpy as np
from loguru import logger

from scalebridge.utils.legged_estimator.depth_ground import DepthGroundHeight, d435_in_torso
from scalebridge.utils.legged_estimator.depth_source import DepthWireSubscriber, RealSenseDepthSource
from scalebridge.utils.legged_estimator.lidar_odometry import LidarOdometryFusion
from scalebridge.utils.legged_estimator.pose_wire import BODY_FRAMES, PoseSubscriber
from scalebridge.utils.legged_estimator.state_estimator import LeggedStateEstimator


class LeggedEstimatorOnlineClient:
    """Root localization from onboard sensors; a drop-in replacement for the Vive tracker client.

    Listens to the robot state that the low-level controller publishes over LCM (joint encoders, joint torques and
    the pelvis IMU at 200 Hz) and runs the legged state estimator on every message, so global tracking needs no
    external tracking hardware. Positions are expressed relative to the pose at calibration, with the IMU heading.
    """

    def __init__(
        self,
        estimator,
        joint_names,
        state_decoder,
        lcm_url: str = "udpm://239.255.76.67:7667?ttl=255",
        channel: str = "robot_state_data",
        max_dt: float = 0.02,
        max_buffer_frames: int = 500,
        log_period: float = 1.0,
        lidar=None,
        depth=None,
    ):
        self.estimator = LeggedStateEstimator(**estimator)
        # LCM arrays follow the robot's motor order (which may include hand joints); map them to the model order.
        joint_names = list(joint_names)
        self.joint_idx = np.array([joint_names.index(name) for name in self.estimator.joint_names], dtype=np.int64)
        self.state_decoder = state_decoder
        self.lcm_url = lcm_url
        self.channel = channel
        self.max_dt = float(max_dt)
        self.log_period = float(log_period)
        self.buffer = deque(maxlen=max_buffer_frames)
        # Optional LiDAR odometry (FAST-LIO on the Mid-360, published by MagicLoco's fastlio_bridge.py): its absolute pose is
        # fused into the estimate, which removes the height and position drift of the legs alone on stairs.
        self.lidar_cfg = dict(lidar) if lidar else None
        # Optional D435i depth camera: the ground height under the feet from a rolling height map (see DepthGroundHeight).
        self.depth_cfg = dict(depth) if depth else None
        self.depth_ground = None
        self.depth_source = None
        self.lidar_fusions = {}
        self.lidar_subscriber = None

        self._lock = threading.Lock()
        self._first_state = threading.Event()
        self._running = threading.Event()
        self._last_time = None
        self._last_log_time = None
        self._lcm = None
        self._poll_thread = None

    def listen(self) -> None:
        self._lcm = lcm.LCM(self.lcm_url)
        self._lcm.subscribe(self.channel, self._state_handler)
        self._running.set()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="LeggedEstimatorPoll")
        self._poll_thread.start()
        if self.lidar_cfg:
            self.lidar_subscriber = PoseSubscriber(self.lidar_cfg.get('endpoint', 'tcp://192.168.123.164:5606'), self._lidar_handler)
            self.lidar_subscriber.start()
            logger.info(f"[LeggedEstimator] Fusing LiDAR odometry from {self.lidar_subscriber.endpoint}.")
        if self.depth_cfg:
            hz = self.depth_cfg.get('hz', 10.0)
            if self.depth_cfg.get('source', 'wire') == 'realsense':
                self.depth_source = RealSenseDepthSource(self._depth_handler, hz=hz, width=self.depth_cfg.get('width', 424), height=self.depth_cfg.get('height', 240), fps=self.depth_cfg.get('fps', 30))
            else:
                self.depth_source = DepthWireSubscriber(self.depth_cfg.get('endpoint', 'tcp://192.168.123.164:5609'), self._depth_handler, hz=hz)
            self.depth_source.start()
            logger.info("[LeggedEstimator] Using the D435i depth camera for the ground height under the feet.")

    def accept_blocking(self) -> str:
        self._first_state.wait()
        return f"{self.channel} over LCM"

    def start(self) -> None:
        logger.info(f"[LeggedEstimator] Estimating the root position from joint encoders, joint torques and the IMU.")

    def stop(self) -> None:
        self._running.clear()
        if self.lidar_subscriber is not None:
            self.lidar_subscriber.stop()
            self.lidar_subscriber = None
        if self.depth_source is not None:
            self.depth_source.stop()
            self.depth_source = None
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        self._poll_thread = None

    def _poll_loop(self) -> None:
        while self._running.is_set():
            readable, _, _ = select.select([self._lcm.fileno()], [], [], 0.01)
            if readable:
                self._lcm.handle()

    def _lidar_handler(self, position, quat_wxyz, body, receive_time) -> None:
        frame = BODY_FRAMES.get(body)
        if frame is None:
            return
        with self._lock:
            fusion = self.lidar_fusions.get(frame)
            if fusion is None:
                fusion = self.lidar_fusions[frame] = LidarOdometryFusion(
                    self.estimator, frame, latency=self.lidar_cfg.get('latency', 0.0), gate=self.lidar_cfg.get('gate', 0.3), realign_after=self.lidar_cfg.get('realign_after', 1.0)
                )
            fusion.update(position, quat_wxyz, receive_time)

    def _depth_handler(self, depth, intrinsics, receive_time) -> None:
        with self._lock:
            if self.depth_ground is None:
                self.depth_ground = DepthGroundHeight(self.estimator, intrinsics, d435_in_torso())
                if self.estimator.initialized:
                    self.depth_ground.reset()
            if self.estimator.initialized:
                self.depth_ground.update(depth, intrinsics)

    def _state_handler(self, channel, data) -> None:
        msg = self.state_decoder.decode(data)
        now = time.monotonic()
        dt = 0.0 if self._last_time is None else min(now - self._last_time, self.max_dt)
        self._last_time = now

        with self._lock:
            position = self.estimator.update(
                dt,
                np.asarray(msg.q)[self.joint_idx],
                np.asarray(msg.qd)[self.joint_idx],
                np.asarray(msg.tau_est)[self.joint_idx],
                quat_wxyz=np.asarray(msg.quat),
                gyro=np.asarray(msg.omegaBody),
                acc=np.asarray(msg.aBody),
            )
            if self.estimator.initialized:
                self.buffer.append((now, position))
                if self.log_period > 0 and (self._last_log_time is None or now - self._last_log_time >= self.log_period):
                    self._last_log_time = now
                    probabilities = self.estimator.kalman_filter.contact_probabilities
                    logger.debug(f"[LeggedEstimator] Root position {np.round(position, 3).tolist()} m, contact probabilities {np.round(probabilities, 2).tolist()}")
        self._first_state.set()

    def calibrate(self, robot_root_quat) -> None:
        # The robot stands still on flat ground here: restart the filter so the current pelvis xy becomes the origin.
        # No heading correction is needed because the estimate already uses the IMU heading that aligns the motion.
        with self._lock:
            self.estimator.reset()
            for fusion in self.lidar_fusions.values():
                fusion.reset_alignment()  # the next LiDAR pose defines the odom origin and heading
            if self.depth_ground is not None:
                self.depth_ground.reset()  # restart the height map on the flat patch under the robot
            self.buffer.append((time.monotonic(), self.estimator.position))

    def get_root_pos(self) -> np.ndarray:
        with self._lock:
            return self.estimator.position.astype(np.float32)
