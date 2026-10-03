"""Augment planner-recorded deep-squat clips for the squat-aware fine-tune (pure numpy, no simulator needed).

Input: joblib pkl clips in the terrain-clip format (root_pos (T,3), root_rot (T,4) quaternion xyzw, dof_pos (T,29) in ScaleTrack joint order,
fps 50). Every training clip gets variants with other speeds (time scaling: linear interpolation of positions / joints, slerp of the root
orientation) and a random heading (rotation about the vertical axis around the start position). Clip names of the variants start with
`sq_`, which the terrain command uses to recognise them (anchor-free squat clips, see LayoutMotionCommandCfg.anchor_free_clip_prefix).

usage: make_squat_aug.py --inputs a.pkl b.pkl ... --out_dir squat_aug_raw/clips [--scales 0.8 1.0 1.25] [--headings 2] [--seed 0]
"""
import argparse
import os

import joblib
import numpy as np


def quat_mul_xyzw(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def slerp_xyzw(q0, q1, t):
    """Vectorised slerp between quaternion arrays q0, q1 (N,4) with weights t (N,)."""
    dot = np.sum(q0 * q1, axis=-1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)
    dot = np.abs(dot).clip(0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-6
    w0 = np.where(near, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(near, 1.0, sin_theta))
    w1 = np.where(near, t, np.sin(t * theta) / np.where(near, 1.0, sin_theta))
    out = w0[:, None] * q0 + w1[:, None] * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def time_scale(clip, speed):
    """Replay the clip `speed` times faster (speed > 1: shorter) at the same fps."""
    n = len(clip["root_pos"])
    t_new = np.arange(0.0, n - 1 + 1e-9, speed)  # sample positions in source frames
    i0 = np.floor(t_new).astype(int).clip(0, n - 2)
    w = (t_new - i0).astype(np.float64)
    lerp = lambda x: (x[i0] * (1 - w)[:, None] + x[i0 + 1] * w[:, None]).astype(np.float32)
    rot = slerp_xyzw(clip["root_rot"][i0].astype(np.float64), clip["root_rot"][i0 + 1].astype(np.float64), w)
    return {"root_pos": lerp(clip["root_pos"]), "root_rot": rot.astype(np.float32), "dof_pos": lerp(clip["dof_pos"])}


def rotate_heading(clip, yaw):
    """Rotate the whole clip about the vertical axis through its first root position."""
    c, s = np.cos(yaw), np.sin(yaw)
    p = clip["root_pos"].copy()
    xy0 = p[0, :2].copy()
    d = p[:, :2] - xy0
    p[:, 0] = xy0[0] + c * d[:, 0] - s * d[:, 1]
    p[:, 1] = xy0[1] + s * d[:, 0] + c * d[:, 1]
    qz = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=np.float64)
    q = quat_mul_xyzw(np.broadcast_to(qz, clip["root_rot"].shape), clip["root_rot"].astype(np.float64))
    return {"root_pos": p.astype(np.float32), "root_rot": q.astype(np.float32), "dof_pos": clip["dof_pos"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.8, 1.0, 1.25])
    ap.add_argument("--headings", type=int, default=2, help="variants per speed: the first keeps the heading, the others get a random heading")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    count = 0
    for path in args.inputs:
        clip = joblib.load(path)
        base = os.path.splitext(os.path.basename(path))[0].replace("deep_squat_clips_unified_", "")
        for s in args.scales:
            scaled = time_scale(clip, s) if abs(s - 1.0) > 1e-6 else {k: clip[k] for k in ("root_pos", "root_rot", "dof_pos")}
            for h in range(args.headings):
                yaw = 0.0 if h == 0 else float(rng.uniform(-np.pi, np.pi))
                out = rotate_heading(scaled, yaw) if yaw != 0.0 else scaled
                name = f"sq_{base}_s{int(round(s * 100)):03d}_h{h}"
                joblib.dump(
                    {**{k: np.asarray(v, dtype=np.float32) for k, v in out.items()}, "fps": 50,
                     "meta": {"source": os.path.basename(path), "speed": s, "yaw": yaw, "min_pelvis_z": float(out["root_pos"][:, 2].min())}},
                    os.path.join(args.out_dir, name + ".pkl"),
                )
                count += 1
    print(f"wrote {count} clips to {args.out_dir}")
