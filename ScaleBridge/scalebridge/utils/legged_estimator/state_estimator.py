import numpy as np
from loguru import logger

from scalebridge.utils.legged_estimator.gm_observer import GmObserver
from scalebridge.utils.legged_estimator.legged_model import LeggedModel
from scalebridge.utils.legged_estimator.linear_kalman_filter import LinearKalmanFilter, LinearKalmanFilterConfig


class LeggedStateEstimator:
    """Base position/velocity estimation from joint encoders, joint torques and the pelvis IMU.

    ROS-free port of the legged_control2 state estimator (legged_controllers::StateEstimator), which BeyondMimic
    deploys without an external tracker. Each update runs the same sequence as the original controller: write the
    sensor readings into the model, update kinematics, run the momentum observer, feed its contact wrenches and the
    raw IMU acceleration to the Kalman filter, then write the estimated base position and local velocity back into
    the model for the next cycle. Orientation is taken from the IMU as-is; only translation is estimated, so the
    result is odometry and drifts slowly with foot slip.
    """

    def __init__(self, urdf_path, base_name="pelvis", contact_names=("LL_FOOT", "LR_FOOT"), cutoff_frequency=10.0, **kalman_filter_config):
        self.model = LeggedModel(urdf_path, base_name, six_dof_contact_names=list(contact_names))
        self.gm_observer = GmObserver(self.model, cutoff_frequency)
        self.kalman_filter = LinearKalmanFilter(self.model, LinearKalmanFilterConfig(**kalman_filter_config))
        self.initialized = False
        # Optional callable(foot_positions) -> list of ground heights (None = unknown), e.g. DepthGroundHeight.foot_heights.
        self.ground_height = None
        self.time = 0.0  # seconds of sensor time accumulated by update(); the clock of the fusion modules
        self.update_hooks = []  # callables run at the end of every update (after the base state is written back)

    @property
    def joint_names(self):
        return self.model.joint_names

    @property
    def position(self):
        return self.kalman_filter.get_position()

    @property
    def velocity(self):
        return self.kalman_filter.get_velocity_global()

    def set_sensors(self, joint_pos, joint_vel, joint_tau, quat_wxyz, gyro):
        """Joint arrays follow `joint_names`; the IMU quaternion is wxyz and the gyro is in the base frame."""
        q, v = self.model.q, self.model.v
        q[3:7] = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        q[3:7] /= np.linalg.norm(q[3:7])
        q[7:] = joint_pos
        v[3:6] = gyro
        v[6:] = joint_vel
        self.model.tau[:] = joint_tau

    def reset(self):
        """Restart the filter from the current pose (feet on flat ground); the base xy becomes the origin."""
        self.model.q[0:3] = 0.0
        self.model.v[0:3] = 0.0
        # Match a freshly constructed filter: zero contact probabilities inflate the contact process noise on the first
        # prediction, so the contact states (reset to zero) rather than the base absorb the current foot positions.
        self.kalman_filter.contact_probabilities[:] = 0.0
        self.kalman_filter.reset()
        self._write_base_state()
        self.initialized = True

    def update(self, dt, joint_pos, joint_vel, joint_tau, quat_wxyz, gyro, acc):
        inputs = (joint_pos, joint_vel, joint_tau, quat_wxyz, gyro, acc)
        if dt <= 0.0 or not all(np.all(np.isfinite(x)) for x in inputs):
            return self.position

        self.time += dt
        self.set_sensors(joint_pos, joint_vel, joint_tau, quat_wxyz, gyro)
        if not self.initialized:
            self.reset()

        self.model.update()
        self.gm_observer.update(dt)
        self.kalman_filter.set_contact_wrenches(self.gm_observer.get_contact_wrenches())
        self.kalman_filter.set_acceleration_local(acc)
        self.kalman_filter.update_imu_process(dt)
        if self.ground_height is not None:
            foot_positions = [self.model.get_frame_placement(f).translation for f in self.model.end_effector_frame_ids]
            self.kalman_filter.set_feet_heights(self.ground_height(foot_positions))
        self.kalman_filter.update_contacts_measurement(dt)

        if not (np.all(np.isfinite(self.kalman_filter.x_hat)) and np.all(np.isfinite(self.kalman_filter.p))):
            logger.warning("[LeggedEstimator] Non-finite estimate; restarting the filter from the current pose.")
            self.gm_observer.reset()
            self.reset()
        self._write_base_state()
        for hook in self.update_hooks:
            hook()
        return self.position

    def update_position_measurement(self, dt, position):
        self.kalman_filter.update_position_measurement(dt, position)
        self._write_base_state()

    def _write_base_state(self):
        self.model.q[0:3] = self.kalman_filter.get_position()
        self.model.v[0:3] = self.kalman_filter.get_velocity_local()
