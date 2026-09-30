import select
import threading
import time
from collections import deque

import lcm
import numpy as np
from loguru import logger

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

    def accept_blocking(self) -> str:
        self._first_state.wait()
        return f"{self.channel} over LCM"

    def start(self) -> None:
        logger.info(f"[LeggedEstimator] Estimating the root position from joint encoders, joint torques and the IMU.")

    def stop(self) -> None:
        self._running.clear()
        if self._poll_thread is not None and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        self._poll_thread = None

    def _poll_loop(self) -> None:
        while self._running.is_set():
            readable, _, _ = select.select([self._lcm.fileno()], [], [], 0.01)
            if readable:
                self._lcm.handle()

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
            self.buffer.append((time.monotonic(), self.estimator.position))

    def get_root_pos(self) -> np.ndarray:
        with self._lock:
            return self.estimator.position.astype(np.float32)
