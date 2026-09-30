"""Pure-numpy helpers for the MagicLoco terrain reference recorder (no Isaac / torch imports).

Used by ``write_clips.py`` (segmentation + pkl writing) and ``smoke_checks.py`` / ``dataset_stats.py``.

Raw rollout layout (one ``.npz`` per rollout, written by ``record_magicloco_refs.py``):
    root_pos   (T, N, 3)  float32  pelvis (articulation root link) position, world frame [m]
    root_quat  (T, N, 4)  float32  pelvis orientation, world frame, **wxyz** (IsaacLab convention)
    dof_pos    (T, N, 29) float32  body joints in ScaleTrack G1_29DOF_JOINT_NAMES order
    cmd        (T, N, 7)  float32  command in force at frame t (vx, vy, wz, height, roll, pitch, yaw)
    term       (T, N)     bool     step t ended in a termination (fall-like)  -> reset inside step t
    trunc      (T, N)     bool     step t ended in a time-out                 -> reset inside step t
    term_code  (T, N)     uint8    bitmask of the termination terms that fired at step t
    env_row, env_col (N,) int      spawn tile of each env (fixed for the whole rollout)
    arm_train  (N,)       bool     True = training arm distribution (rho_a=1), False = nominal (rho_a=0)
Frame t is the state BEFORE env.step() number t, i.e. the state the policy observation at t describes.
If step t reports done, frame t+1 is the first frame of a new episode (the env was reset inside step t).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
COMMAND_NAMES = ["vx", "vy", "wz", "height", "roll", "pitch", "yaw"]
TERM_BITS = {"fall": 1, "feet_crossed": 2, "knees_close": 4}

FPS = 50
FALL_DROP_FRAMES = 75          # 1.5 s dropped before a termination
MIN_FRAMES = 150               # 3 s minimum clip length
BAD_TILT_DEG = 60.0            # |roll| or |pitch| above this -> bad frame
MIN_PELVIS_HEIGHT = 0.30       # pelvis height above local terrain below this (after GRACE) -> bad frame
HEIGHT_GRACE_FRAMES = 25       # first 0.5 s of a segment exempt from the height test


# ----------------------------------------------------------------------------------------------
# rotations
# ----------------------------------------------------------------------------------------------
def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def roll_pitch_from_wxyz(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Heading-invariant roll/pitch (intrinsic Z-Y-X / yaw-pitch-roll convention), radians."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


# ----------------------------------------------------------------------------------------------
# terrain lookup (layout meta + raster heightmap)
# ----------------------------------------------------------------------------------------------
class TerrainLookup:
    def __init__(self, layout_dir: str):
        with open(os.path.join(layout_dir, "meta.json")) as f:
            self.meta = json.load(f)
        hm = np.load(os.path.join(layout_dir, "heightmap.npz"))
        self.h = hm["height"]
        self.x0, self.y0, self.res = float(hm["x0"]), float(hm["y0"]), float(hm["res"])
        g = self.meta["grid"]
        self.size = float(g["tile_size"][0])
        self.nrows, self.ncols = int(g["num_rows"]), int(g["num_cols"])
        self.gx0, self.gy0 = float(g["tile_grid_origin_xy"][0]), float(g["tile_grid_origin_xy"][1])
        self.type_names = list(self.meta["sub_terrain_names"])
        self.tile_type = np.array(self.meta["tile_type_index"], dtype=np.int64)        # (rows, cols)
        self.tile_step = np.array(self.meta["tile_step_height"], dtype=np.float64)     # stairs only, else 0
        self.tile_box = np.array(self.meta["tile_box_height"], dtype=np.float64)       # boxes only, else 0
        self.tile_param = np.array(self.meta["tile_param_value"], dtype=np.float64)
        self.tile_diff = np.array(self.meta["tile_difficulty"], dtype=np.float64)

    def height(self, xy: np.ndarray) -> np.ndarray:
        """Terrain surface height (upper envelope, ray cast from above) at xy, nearest raster cell."""
        i = np.clip(np.rint((xy[..., 0] - self.x0) / self.res).astype(np.int64), 0, self.h.shape[0] - 1)
        j = np.clip(np.rint((xy[..., 1] - self.y0) / self.res).astype(np.int64), 0, self.h.shape[1] - 1)
        return self.h[i, j]

    # the policy's height scan: 17 x 11 grid, 0.1 m, 1.6 m (forward) x 1.0 m (lateral), yaw-aligned, centred
    # under the torso (~ pelvis xy). The trained "height" command is  root_z - mean(scan hit z)
    # (magicloco rewards_homie.track_base_height_rough_exp / _terrain_z_under_robot).
    _SCAN = np.stack(np.meshgrid(np.linspace(-0.8, 0.8, 17), np.linspace(-0.5, 0.5, 11), indexing="ij"), -1).reshape(-1, 2)

    def scan_mean_height(self, xy: np.ndarray, yaw: np.ndarray, chunk: int = 20000) -> np.ndarray:
        """Mean terrain height over the policy's yaw-aligned 17x11 scan grid centred at xy (n,2)."""
        out = np.empty(xy.shape[0], np.float64)
        for i0 in range(0, xy.shape[0], chunk):
            c, s_ = np.cos(yaw[i0:i0 + chunk]), np.sin(yaw[i0:i0 + chunk])
            gx = xy[i0:i0 + chunk, None, 0] + c[:, None] * self._SCAN[None, :, 0] - s_[:, None] * self._SCAN[None, :, 1]
            gy = xy[i0:i0 + chunk, None, 1] + s_[:, None] * self._SCAN[None, :, 0] + c[:, None] * self._SCAN[None, :, 1]
            out[i0:i0 + chunk] = self.height(np.stack([gx, gy], -1)).mean(axis=1)
        return out

    def tile_of(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, col, inside) of the sub-terrain tile under xy; inside=False on the outer border."""
        r = np.floor((xy[..., 0] - self.gx0) / self.size).astype(np.int64)
        c = np.floor((xy[..., 1] - self.gy0) / self.size).astype(np.int64)
        inside = (r >= 0) & (r < self.nrows) & (c >= 0) & (c < self.ncols)
        return np.clip(r, 0, self.nrows - 1), np.clip(c, 0, self.ncols - 1), inside

    def tile_info(self, r: int, c: int) -> dict:
        return dict(self.meta["tiles"][int(r) * self.ncols + int(c)])


# ----------------------------------------------------------------------------------------------
# segmentation
# ----------------------------------------------------------------------------------------------
def env_segments(term_e: np.ndarray, trunc_e: np.ndarray) -> list[tuple[int, int, str]]:
    """Reset-free intervals of one env: list of (start, end_inclusive, reason).

    reason: 'fall' (step `end` reported terminated), 'timeout' (step `end` reported time-out only),
    'rollout_end' (no done before the rollout ended).
    """
    T = term_e.shape[0]
    done = term_e | trunc_e
    ks = np.nonzero(done)[0]
    segs, s = [], 0
    for k in ks:
        segs.append((s, int(k), "fall" if term_e[k] else "timeout"))
        s = int(k) + 1
    if s <= T - 1:
        segs.append((s, T - 1, "rollout_end"))
    return segs


def trim_segment(start: int, end: int, reason: str, root_pos: np.ndarray, root_quat: np.ndarray,
                 lookup: TerrainLookup) -> tuple[int, str, dict]:
    """Apply the drop rules to one reset-free interval of one env.

    root_pos/root_quat are the env's full (T,3)/(T,4) arrays. Returns (end_kept_inclusive or -1 if
    dropped, end_reason, info).
    """
    info = {"raw_start": start, "raw_end": end, "raw_reason": reason}
    end_keep = end
    end_reason = reason
    if reason == "fall":
        end_keep = end - FALL_DROP_FRAMES
        end_reason = "fall_trimmed"
    if end_keep < start:
        info["drop"] = "too_short"
        return -1, end_reason, info
    sl = slice(start, end_keep + 1)
    q = root_quat[sl].astype(np.float64)
    roll, pitch = roll_pitch_from_wxyz(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    # pelvis height above the LOCAL terrain = the policy's own definition (root_z - mean of its scan)
    hgt = root_pos[sl, 2].astype(np.float64) - lookup.scan_mean_height(root_pos[sl, :2].astype(np.float64), yaw)
    lim = math.radians(BAD_TILT_DEG)
    bad_tilt = (np.abs(roll) > lim) | (np.abs(pitch) > lim)
    idx = np.arange(bad_tilt.shape[0])
    bad_h = (idx >= HEIGHT_GRACE_FRAMES) & (hgt < MIN_PELVIS_HEIGHT)
    bad = bad_tilt | bad_h
    if bad.any():
        b = int(np.argmax(bad))
        info["first_bad_frame"] = b
        info["bad_criterion"] = ("tilt" if bad_tilt[b] else "") + ("height" if bad_h[b] else "")
        end_keep = start + b - 1
        end_reason = end_reason + "+bad_frame_trimmed"
    n = end_keep - start + 1
    if n < MIN_FRAMES:
        info["drop"] = "too_short"
        return -1, end_reason, info
    info["min_pelvis_height"] = float(hgt[: n].min()) if n > 0 else None
    return end_keep, end_reason, info


# ----------------------------------------------------------------------------------------------
# per-clip labels for index.jsonl
# ----------------------------------------------------------------------------------------------
def clip_labels(pos: np.ndarray, lookup: TerrainLookup) -> dict:
    xy = pos[:, :2].astype(np.float64)
    r, c, inside = lookup.tile_of(xy)
    types = np.where(inside, lookup.tile_type[r, c], -1)
    n = len(types)
    frac = {}
    for ti, name in enumerate(lookup.type_names):
        f = float((types == ti).sum()) / n
        if f > 0:
            frac[name] = round(f, 4)
    fb = float((types < 0).sum()) / n
    if fb > 0:
        frac["border"] = round(fb, 4)
    step = np.where(inside, lookup.tile_step[r, c], 0.0)
    box = np.where(inside, lookup.tile_box[r, c], 0.0)
    v = np.linalg.norm(np.diff(xy, axis=0), axis=1) * FPS if n > 1 else np.zeros(1)
    visited = sorted({(int(a), int(b)) for a, b, ins in zip(r, c, inside) if ins})
    return {
        "terrain_frac": frac,
        "max_step_height": round(float(step.max()), 4),
        "max_box_height": round(float(box.max()), 4),
        "mean_speed": round(float(v.mean()), 4),
        "max_speed": round(float(v.max()), 4),
        "path_length": round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 4),
        "z_range": [round(float(pos[:, 2].min()), 4), round(float(pos[:, 2].max()), 4)],
        "tiles_visited": len(visited),
    }
