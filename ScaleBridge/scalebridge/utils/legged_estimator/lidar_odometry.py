from collections import deque

import numpy as np
import pinocchio as pin
from loguru import logger


def yaw_of(rotation):
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class LidarOdometryFusion:
    """Fuses a LiDAR-inertial odometry pose (FAST-LIO on the Livox Mid-360) into the legged state estimator.

    The legs alone give a relative position: without the flat-ground height measurement there is no absolute
    reference for the common height and xy of base and feet, so the estimate drifts. A LiDAR odometry measures the
    absolute pose (in its own odom frame) at about 10 Hz with a small, slowly growing error, which is exactly what is
    missing. The legs still supply the high-rate, jitter-free motion between LiDAR frames and bridge dropouts.

    The odom frame of the LiDAR has an arbitrary origin and heading. At calibration (`reset_alignment`) the first LiDAR
    pose is aligned to the estimator frame (which uses the IMU heading) by the yaw difference of the LiDAR body
    orientation and a translation; later poses are mapped through that fixed transform. The pelvis position is the
    LiDAR position minus the lever from pelvis to LiDAR, taken from the estimator's own forward kinematics (the URDF
    has the `mid360_link` frame), so the waist joints are accounted for. The measurement noise
    (`position_noise_density`, `position_noise_density_z` of the filter) sets how fast the estimate follows the LiDAR.

    Latency: a pose arrives `latency` seconds after it was measured. The estimator's own motion since then is added to
    the pose (the legs know how far the robot moved), so the delayed measurement is compared with the state it describes.
    The wire timestamp cannot be used (the sender's clock is another machine's), so `latency` is a parameter.

    Outliers: a LiDAR odometry can jump (relocalization, degenerate geometry on a staircase). A pose further than `gate`
    from the estimate is ignored; if that persists for `realign_after` seconds the new pose is accepted as the truth
    and only the alignment is shifted, so the estimate itself never jumps.
    """

    def __init__(self, estimator, frame_name="mid360_link", latency=0.0, gate=0.3, realign_after=1.0):
        self.estimator = estimator
        model = estimator.model
        if not model.model.existFrame(frame_name):
            raise ValueError(f"Frame {frame_name} not found in the estimator model.")
        self.frame_id = model.model.getFrameId(frame_name)
        self.frame_name = frame_name
        self.latency = float(latency)
        self.gate = float(gate)
        self.realign_after = float(realign_after)
        self._history = deque(maxlen=max(400, int(2.0 * 200)))  # (time, LiDAR position in the estimator frame)
        estimator.update_hooks.append(self._record)
        self.reset_alignment()

    def reset_alignment(self):
        self._aligned = False
        self._last_time = None
        self._rejected_since = None

    @property
    def aligned(self):
        return self._aligned

    def _record(self):
        self._history.append((self.estimator.time, self.estimator.model.get_frame_placement(self.frame_id).translation.copy()))

    def _position_at(self, time_s):
        """The LiDAR position the estimator had at `time_s` (linear interpolation of the recorded history)."""
        if not self._history:
            return None
        times = np.fromiter((t for t, _ in self._history), dtype=np.float64, count=len(self._history))
        index = int(np.searchsorted(times, time_s))
        if index <= 0:
            return self._history[0][1]
        if index >= len(times):
            return self._history[-1][1]
        (t0, p0), (t1, p1) = self._history[index - 1], self._history[index]
        w = 0.0 if t1 == t0 else (time_s - t0) / (t1 - t0)
        return (1.0 - w) * p0 + w * p1

    def update(self, lidar_position, lidar_quat_wxyz, time_s=None):
        """Fuse one LiDAR odometry pose of the `frame_name` body, given in the LiDAR odom frame (quat wxyz).

        Returns True if the pose was used. `time_s` is unused: the estimator's own clock timestamps the arrival.
        """
        estimator = self.estimator
        if not estimator.initialized:
            return False
        lidar_position = np.asarray(lidar_position, dtype=np.float64)
        if not (np.all(np.isfinite(lidar_position)) and np.all(np.isfinite(lidar_quat_wxyz))):
            return False
        model = estimator.model
        now = estimator.time
        placement = model.get_frame_placement(self.frame_id)
        lever = placement.translation - model.get_frame_placement(model.base_frame_id).translation  # pelvis -> LiDAR, world axes

        if not self._aligned:
            quat = np.asarray(lidar_quat_wxyz, dtype=np.float64)
            rotation_odom = pin.Quaternion(quat[0], quat[1], quat[2], quat[3]).toRotationMatrix()
            self._yaw = yaw_of(placement.rotation) - yaw_of(rotation_odom)
            self._rotation = rot_z(self._yaw)
            self._odom_origin = lidar_position.copy()
            self._estimator_origin = placement.translation.copy()
            self._aligned = True
            self._last_time = now
            logger.info(f"[LidarOdometry] Aligned to the estimator frame, yaw offset {np.degrees(self._yaw):.1f} deg.")
            return True

        lidar_in_estimator = self._estimator_origin + self._rotation @ (lidar_position - self._odom_origin)
        if self.latency > 0.0:
            stamped = self._position_at(now - self.latency)
            if stamped is not None:
                lidar_in_estimator = lidar_in_estimator + (placement.translation - stamped)  # motion since the measurement
        measurement = lidar_in_estimator - lever
        dt = max(now - self._last_time, 1e-3)
        self._last_time = now

        if np.linalg.norm(measurement - estimator.position) > self.gate:
            if self._rejected_since is None:
                self._rejected_since = now
            if now - self._rejected_since < self.realign_after:
                return False
            # The odometry has been somewhere else for a while: trust it, move the alignment, keep the estimate continuous.
            self._odom_origin = lidar_position.copy()
            self._estimator_origin = placement.translation.copy()
            self._rejected_since = None
            logger.warning("[LidarOdometry] Pose jumped away from the estimate and stayed there; alignment shifted.")
            return False
        self._rejected_since = None
        estimator.update_position_measurement(dt, measurement)
        return True
