from dataclasses import dataclass

import numpy as np
from scipy.special import expit

GRAVITY = 9.81


def sigmoid(x, threshold, scale):
    return expit((x - threshold) / scale)  # 1 / (1 + exp(-(x - threshold) / scale)) without overflow warnings


@dataclass
class LinearKalmanFilterConfig:
    imu_acceleration_noise_density: float = 1e-2
    imu_acceleration_bias_noise_density: float = 5e-3
    contact_process_noise_position: float = 0.002
    contact_sensor_noise_position: float = 0.002
    contact_height_sensor_noise: float = 0.002
    contact_radius: float = 0.0
    contact_force_threshold: float = 150.0
    contact_force_scale: float = 15.0
    contact_zmp_length_x: float = 0.08
    contact_zmp_length_y: float = 0.025
    position_noise_density: float = 1e-2


class LinearKalmanFilter:
    """Contact-aided linear Kalman filter for the floating-base position and velocity.

    Port of legged_control2's legged_estimation::LinearKalmanFilter. The state is
    [base_pos(3), base_vel(3), contact_pos(3 * n_contacts), imu_acc_bias(3)] in the world frame. The IMU
    acceleration drives the prediction; contacts are weighted by a probability built from the estimated
    normal force and the center of pressure, and each contact measures base - contact position (from
    forward kinematics) plus a flat-ground height.
    """

    HIGH_SUSPECT_NUMBER = 1e-3
    MIN_CONTACT_PROBABILITY = 0.01

    def __init__(self, legged_model, config=None):
        self.legged_model = legged_model
        self.cfg = config if config is not None else LinearKalmanFilterConfig()

        self.num_contacts = legged_model.num_contacts
        self.dim_contacts = 3 * self.num_contacts
        self.num_state = 6 + self.dim_contacts + 3
        self.dim_contact_observe = self.dim_contacts + self.num_contacts
        self.bias_index = 6 + self.dim_contacts

        self.a = np.eye(self.num_state)
        self.b = np.zeros((self.num_state, 3))
        self.c_contact = np.zeros((self.dim_contact_observe, self.num_state))
        for i in range(self.num_contacts):
            self.c_contact[3 * i:3 * i + 3, 0:3] = np.eye(3)
            self.c_contact[3 * i:3 * i + 3, 6 + 3 * i:9 + 3 * i] = -np.eye(3)
            self.c_contact[self.dim_contacts + i, 6 + 3 * i + 2] = 1.0
        self.c_position = np.zeros((3, self.num_state))
        self.c_position[:, 0:3] = np.eye(3)

        self.acceleration_local = np.zeros(3)
        self.contact_wrenches = [np.zeros(6) for _ in range(self.num_contacts)]
        self.feet_heights = np.zeros(self.num_contacts)  # never updated upstream: flat ground
        self.contact_probabilities = np.zeros(self.num_contacts)
        self.zmps = [np.zeros(2) for _ in range(self.num_contacts)]

        self.x_hat = np.zeros(self.num_state)
        self.p = np.zeros((self.num_state, self.num_state))

    def set_acceleration_local(self, acceleration_local):
        self.acceleration_local = np.asarray(acceleration_local, dtype=np.float64)

    def set_contact_wrenches(self, wrenches):
        self.contact_wrenches = wrenches

    def get_position(self):
        return self.x_hat[0:3].copy()

    def get_velocity_global(self):
        return self.x_hat[3:6].copy()

    def get_velocity_local(self):
        return self.legged_model.get_base_rotation().T @ self.x_hat[3:6]

    def reset(self):
        """Put the base above the mean contact height and restart the covariance (legged model must hold the current pose)."""
        self.x_hat = np.zeros(self.num_state)
        self.legged_model.update()
        feet_z = [self.legged_model.get_frame_placement(frame_id).translation[2] for frame_id in self.legged_model.end_effector_frame_ids]
        self.x_hat[2] = -np.sum(feet_z) / len(feet_z)

        cfg = self.cfg
        self.p = np.eye(self.num_state)
        self.p[0:6, 0:6] = self.get_imu_process_noise_covariance(0.002)
        self.p[6:self.bias_index, 6:self.bias_index] *= 0.002 * cfg.contact_process_noise_position ** 2
        self.p[self.bias_index:, self.bias_index:] *= 0.002 * cfg.imu_acceleration_bias_noise_density ** 2

    def get_imu_process_noise_covariance(self, dt):
        q = np.zeros((6, 6))
        q[0:3, 0:3] = dt ** 3 / 3.0 * np.eye(3)
        q[0:3, 3:6] = 0.5 * dt ** 2 * np.eye(3)
        q[3:6, 0:3] = 0.5 * dt ** 2 * np.eye(3)
        q[3:6, 3:6] = dt * np.eye(3)
        return q * self.cfg.imu_acceleration_noise_density ** 2

    def get_kalman_gain(self, c, r):
        innovation_covariance = c @ self.p @ c.T + r
        return self.p @ c.T @ np.linalg.inv(innovation_covariance)

    def update_imu_process(self, dt):
        cfg = self.cfg
        rotation = self.legged_model.get_base_rotation()
        self.a[0:3, 3:6] = dt * np.eye(3)
        self.b[0:3, 0:3] = 0.5 * dt * dt * np.eye(3)
        self.b[3:6, 0:3] = dt * np.eye(3)
        self.a[0:3, self.bias_index:] = -0.5 * dt * dt * rotation
        self.a[3:6, self.bias_index:] = -dt * rotation

        q = dt * np.eye(self.num_state)
        q[0:6, 0:6] = self.get_imu_process_noise_covariance(dt)
        q[6:self.bias_index, 6:self.bias_index] *= cfg.contact_process_noise_position ** 2
        q[self.bias_index:, self.bias_index:] *= cfg.imu_acceleration_bias_noise_density ** 2
        for i in range(self.num_contacts):
            index = 6 + 3 * i
            q[index:index + 3, index:index + 3] /= self.contact_probabilities[i] + self.HIGH_SUSPECT_NUMBER

        acceleration_global = rotation @ self.acceleration_local + np.array([0.0, 0.0, -GRAVITY])
        self.x_hat = self.a @ self.x_hat + self.b @ acceleration_global
        self.p = self.a @ self.p @ self.a.T + q

    def update_contacts_measurement(self, dt):
        cfg = self.cfg
        self.update_contact_probabilities()

        r = dt * np.eye(self.dim_contact_observe)
        r[0:self.dim_contacts, 0:self.dim_contacts] *= cfg.contact_sensor_noise_position ** 2
        r[self.dim_contacts:, self.dim_contacts:] *= cfg.contact_height_sensor_noise ** 2

        base_position = self.legged_model.q[0:3]
        relative_positions = np.zeros(self.dim_contacts)
        for i, frame_id in enumerate(self.legged_model.end_effector_frame_ids):
            scale = self.contact_probabilities[i] + self.HIGH_SUSPECT_NUMBER
            r[3 * i:3 * i + 3, 3 * i:3 * i + 3] /= scale
            r[self.dim_contacts + i, self.dim_contacts + i] /= scale
            relative_positions[3 * i:3 * i + 3] = base_position - self.legged_model.get_frame_placement(frame_id).translation
            relative_positions[3 * i + 2] += cfg.contact_radius

        y = np.concatenate([relative_positions, self.feet_heights])
        innovation = y - self.c_contact @ self.x_hat
        gain = self.get_kalman_gain(self.c_contact, r)
        for i in range(self.num_contacts):
            if not self.contact_probabilities[i] >= self.MIN_CONTACT_PROBABILITY:
                gain[:, 3 * i:3 * i + 3] = 0.0
                gain[:, self.dim_contacts + i] = 0.0

        self.x_hat = self.x_hat + gain @ innovation
        self.p = (np.eye(self.num_state) - gain @ self.c_contact) @ self.p

    def update_position_measurement(self, dt, position):
        """Fuse an external base-position measurement (e.g. motion capture or LiDAR odometry)."""
        r = self.cfg.position_noise_density ** 2 * dt * np.eye(3)
        innovation = np.asarray(position, dtype=np.float64) - self.c_position @ self.x_hat
        gain = self.get_kalman_gain(self.c_position, r)
        self.x_hat = self.x_hat + gain @ innovation
        self.p = (np.eye(self.num_state) - gain @ self.c_position) @ self.p

    def update_contact_probabilities(self):
        cfg = self.cfg
        force_probabilities = np.array([sigmoid(wrench[2], cfg.contact_force_threshold, cfg.contact_force_scale) for wrench in self.contact_wrenches])
        zmp_probabilities = np.ones(self.num_contacts)
        zmp_limit = np.array([cfg.contact_zmp_length_x, cfg.contact_zmp_length_y])

        for i, frame_id in enumerate(self.legged_model.end_effector_frame_ids):
            rotation = self.legged_model.get_frame_placement(frame_id).rotation
            force_local = rotation.T @ self.contact_wrenches[i][0:3]
            torque_local = rotation.T @ self.contact_wrenches[i][-3:]
            if not abs(force_local[2]) > 1e-6:
                # No normal force: veto the contact and park the ZMP outside the sole for visualization.
                self.zmps[i] = 2.0 * zmp_limit
                zmp_probabilities[i] = 0.0
                continue
            self.zmps[i] = np.array([-torque_local[1], torque_local[0]]) / force_local[2]
            zmp_probabilities[i] = float(abs(self.zmps[i][0]) <= zmp_limit[0] and abs(self.zmps[i][1]) <= zmp_limit[1])

        probabilities = force_probabilities * zmp_probabilities
        probabilities[~np.isfinite(probabilities)] = 0.0
        self.contact_probabilities = probabilities
