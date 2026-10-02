from dataclasses import dataclass

import numpy as np

from scalebridge.utils.legged_estimator.height_map import RollingHeightMap


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_fovy(fovy_deg, width, height):
        fy = height / (2.0 * np.tan(np.radians(fovy_deg) / 2.0))
        return CameraIntrinsics(fx=fy, fy=fy, cx=width / 2.0, cy=height / 2.0)


def deproject(depth, intrinsics, z_min=0.28, z_max=4.0, stride=2):
    """Planar-Z depth image -> points in the camera frame (looks down -Z, +X right, +Y up, as MuJoCo's cameras).

    z_min is the D435i's minimum range, so the simulation cannot see closer than the hardware.
    """
    d = np.asarray(depth, dtype=np.float32)[::stride, ::stride]
    h, w = d.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32) * stride, np.arange(h, dtype=np.float32) * stride)
    ok = np.isfinite(d) & (d > z_min) & (d < z_max)
    if not ok.any():
        return np.zeros((0, 3), dtype=np.float32)
    z = d[ok]
    return np.stack([(uu[ok] - intrinsics.cx) * z / intrinsics.fx, (intrinsics.cy - vv[ok]) * z / intrinsics.fy, -z], axis=-1)


D435_POSITION = np.array([0.0576, 0.0175, 0.42987])  # D435i in torso_link, from the URDF d435_joint
D435_PITCH = 0.8308  # rad, looking down


def d435_in_torso():
    """Pose of the torso-mounted D435i in torso_link: (position, rotation with the MuJoCo camera axes x right, y up, z backward)."""
    s, c = np.sin(D435_PITCH), np.cos(D435_PITCH)
    return D435_POSITION.copy(), np.array([[0.0, s, -c], [-1.0, 0.0, 0.0], [0.0, c, s]])


class DepthGroundHeight:
    """Ground height under the feet from a torso-mounted depth camera (Intel RealSense D435i), for the legged estimator.

    The legged estimator measures every planted foot at z = 0 (flat ground) in its default configuration. On stairs that
    is wrong; a rolling height map built from the depth camera in the estimator's own frame knows how high the ground is
    where the foot actually lands, and replaces the zero. The map is built with the estimator's pose, so what it provides
    is consistency over the few seconds a cell stays in memory: when a foot arrives on a cell that was seen from a
    well-estimated pose, the height error accumulated since is removed. This is the same idea as the FK z-anchor and the
    whole-map z bias of MagicLoco's perception stack, expressed as a Kalman filter measurement.

    The camera is described by its pose in the torso frame (D435i on the G1: 47.6 deg down, see the URDF d435_joint).
    """

    def __init__(self, estimator, intrinsics, camera_in_torso, torso_frame="torso_link", extent=4.0, cell=0.05,
                 self_filter_radius=0.7, self_filter_below_torso=0.45, seed_radius=0.5, memory_frames=50):
        self.estimator = estimator
        self.intrinsics = intrinsics
        self.camera_position, self.camera_rotation = (np.asarray(camera_in_torso[0], dtype=np.float64), np.asarray(camera_in_torso[1], dtype=np.float64))
        model = estimator.model
        self.torso_id = model.model.getFrameId(torso_frame)
        self.map = RollingHeightMap(extent=extent, cell=cell, memory_frames=memory_frames)
        self.self_filter_radius = self_filter_radius
        self.self_filter_below_torso = self_filter_below_torso
        self.seed_radius = seed_radius
        estimator.ground_height = self.foot_heights

    def reset(self):
        """Restart the map at calibration: the robot stands on flat ground, so seed the patch under it at the feet height."""
        self.map.clear()
        model = self.estimator.model
        model.update()  # placements must reflect the base pose written back by the estimator reset
        feet = [model.get_frame_placement(f).translation for f in model.end_effector_frame_ids]
        base = model.get_frame_placement(model.base_frame_id).translation
        self.map.recenter(base[0], base[1])
        z = float(np.mean([f[2] for f in feet]))
        self.map.seed(base[0], base[1], z, self.seed_radius)

    def update(self, depth, intrinsics=None):
        """Fold one depth image (planar-Z, metres) into the map using the estimator's current pose."""
        estimator = self.estimator
        if intrinsics is not None:
            self.intrinsics = intrinsics
        if not estimator.initialized:
            return
        model = estimator.model
        torso = model.get_frame_placement(self.torso_id)
        torso_position, torso_rotation = torso.translation, torso.rotation
        camera_position = torso_position + torso_rotation @ self.camera_position
        camera_rotation = torso_rotation @ self.camera_rotation
        points = deproject(depth, self.intrinsics)
        if points.shape[0] == 0:
            return
        points = points @ camera_rotation.T + camera_position
        # The camera sees the robot's own forearms and hands; ground is always well below the torso, so inside a radius
        # reject anything high.
        near = (points[:, 0] - torso_position[0]) ** 2 + (points[:, 1] - torso_position[1]) ** 2 < self.self_filter_radius ** 2
        high = points[:, 2] > torso_position[2] - self.self_filter_below_torso
        points = points[~(near & high)]
        base = model.get_frame_placement(model.base_frame_id).translation
        self.map.recenter(base[0], base[1])
        self.map.update(points)

    def foot_heights(self, foot_positions):
        """Ground height under each foot (world xy of the sole centers), None where the map has no trusted cell."""
        return [self.map.query(p[0], p[1]) for p in foot_positions]
