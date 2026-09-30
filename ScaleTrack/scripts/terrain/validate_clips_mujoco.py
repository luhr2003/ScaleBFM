"""Kinematic / terrain consistency check of recorded terrain clips, using ScaleTrack's G1 MJCF and the exported mesh.

For every frame of a random sample of clips the recorded (root_pos, root_rot xyzw, dof_pos[29]) is loaded into MuJoCo
(g1_29dof.xml, the same kinematics ScaleTrack replays), the lowest point of every foot collision capsule is computed by
forward kinematics, and the terrain surface height straight below each point is obtained by ray casting against the
exported terrain mesh. If pelvis frame, joint order, quaternion convention and terrain alignment are all consistent,
then during stance the gap between sole and terrain is ~0 and the feet rarely penetrate the mesh.

Reported (all in metres), per local terrain roughness bin and overall:
  stance gap      lowest foot point minus terrain height, for stance frames (foot slow and close to the ground)
  penetration     fraction of frames where a foot point is more than 2 cm below the terrain surface
  support gap     min over both feet of the foot gap, over all frames
  pelvis height   pelvis z minus terrain height below the pelvis
--diagnose additionally evaluates alternative hypotheses (root_rot interpreted as wxyz; constant z offsets) so a
convention bug shows up as a clear improvement of the stance gap.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import joblib
import mujoco
import numpy as np
import trimesh

XML = "/home/vcj9002/magicloco/ScaleBFM/ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml"
JOINTS = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
JOINTS = [j + "_joint" for j in JOINTS]


class Kinematics:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        assert self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "first joint must be the free base joint"
        self.qadr = np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self.foot_geoms = {}
        for side in ("left", "right"):
            ids = [self.model.geom(f"{side}_foot{i}_collision").id for i in range(1, 8)]
            self.foot_geoms[side] = np.array(ids)
        self.pelvis_id = self.model.body("pelvis").id
        self.ankle_ids = {s: self.model.body(f"{s}_ankle_roll_link").id for s in ("left", "right")}

    def foot_points(self, root_pos, root_quat_wxyz, dof_pos):
        """Lowest-sphere centres of every foot capsule endpoint (14 per foot) and their radii, world frame."""
        m, d = self.model, self.data
        d.qpos[:3] = root_pos
        d.qpos[3:7] = root_quat_wxyz
        d.qpos[self.qadr] = dof_pos
        mujoco.mj_kinematics(m, d)
        out = {}
        for side, ids in self.foot_geoms.items():
            centers = d.geom_xpos[ids]  # (7,3)
            axes = d.geom_xmat[ids].reshape(-1, 3, 3)[:, :, 2]  # capsule axis = local z
            half = m.geom_size[ids, 1][:, None]
            r = m.geom_size[ids, 0]
            pts = np.concatenate([centers + axes * half, centers - axes * half], axis=0)  # (14,3)
            out[side] = (pts, np.concatenate([r, r]))
        out["ankle"] = {s: d.xpos[i].copy() for s, i in self.ankle_ids.items()}
        return out


def load_mesh(root, layout):
    path = os.path.join(root, "layouts", f"layout_{layout}", "terrain.npz")
    z = np.load(path)
    keys = list(z.keys())
    v = z["vertices"] if "vertices" in keys else z[keys[0]]
    f = z["faces"] if "faces" in keys else z[keys[1]]
    return trimesh.Trimesh(vertices=v.astype(np.float64), faces=f.astype(np.int64), process=False)


class TerrainHeight:
    def __init__(self, mesh):
        self.mesh = mesh
        self.ray_z = float(mesh.bounds[1, 2]) + 1.0
        self.floor = float(mesh.bounds[0, 2]) - 1.0

    def __call__(self, xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        origins = np.concatenate([xy, np.full((len(xy), 1), self.ray_z)], axis=1)
        dirs = np.tile(np.array([[0.0, 0.0, -1.0]]), (len(xy), 1))
        loc, idx_ray, _ = self.mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
        h = np.full(len(xy), np.nan)
        h[idx_ray] = loc[:, 2]
        return h


def quat_xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def stats(v):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "p5": float(np.percentile(v, 5)),
            "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}


def analyse_clip(kin, terrain, clip, quat_mode="xyzw"):
    rp = np.asarray(clip["root_pos"], dtype=np.float64)
    rq = np.asarray(clip["root_rot"], dtype=np.float64)
    dof = np.asarray(clip["dof_pos"], dtype=np.float64)
    fps = float(clip["fps"])
    quat = quat_xyzw_to_wxyz(rq) if quat_mode == "xyzw" else rq
    n = len(rp)
    pts_all = {"left": np.zeros((n, 14, 3)), "right": np.zeros((n, 14, 3))}
    rad = None
    ankle = {"left": np.zeros((n, 3)), "right": np.zeros((n, 3))}
    for t in range(n):
        fp = kin.foot_points(rp[t], quat[t], dof[t])
        for s in ("left", "right"):
            pts_all[s][t] = fp[s][0]
            ankle[s][t] = fp["ankle"][s]
        rad = fp["left"][1]
    res = {"n": n, "fps": fps}
    # terrain heights below all foot points and below the pelvis
    heights = {}
    for s in ("left", "right"):
        h = terrain(pts_all[s][:, :, :2].reshape(-1, 2)).reshape(n, 14)
        gap_pts = pts_all[s][:, :, 2] - rad[None] - h  # (n,14): sphere bottom minus surface
        res[f"{s}_gap"] = np.nanmin(np.where(np.isfinite(gap_pts), gap_pts, np.inf), axis=1)
        res[f"{s}_gap"][~np.isfinite(res[f"{s}_gap"])] = np.nan
        # foot speed from the ankle link
        v = np.gradient(ankle[s], axis=0) * fps
        res[f"{s}_speed"] = np.linalg.norm(v, axis=1)
        # local terrain relief around the lowest point (5 x 5 grid, +-0.3 m) to bin terrain roughness
        idx = np.nanargmin(np.where(np.isfinite(gap_pts), gap_pts, np.inf), axis=1)
        low_xy = pts_all[s][np.arange(n), idx, :2]
        offs = np.stack(np.meshgrid(np.linspace(-0.3, 0.3, 5), np.linspace(-0.3, 0.3, 5)), -1).reshape(-1, 2)
        grid = (low_xy[:, None, :] + offs[None]).reshape(-1, 2)
        hg = terrain(grid).reshape(n, -1)
        res[f"{s}_relief"] = np.nanmax(hg, axis=1) - np.nanmin(hg, axis=1)
    res["pelvis_h"] = rp[:, 2] - terrain(rp[:, :2])
    return res


def summarize(results, args):
    def cat(key):
        return np.concatenate([r[key] for r in results])

    out = {}
    bins = [("flat(<1cm)", 0.0, 0.01), ("subtle(1-4cm)", 0.01, 0.04), ("step(4-12cm)", 0.04, 0.12), ("high(>=12cm)", 0.12, 9.0)]
    gap = np.concatenate([cat("left_gap"), cat("right_gap")])
    spd = np.concatenate([cat("left_speed"), cat("right_speed")])
    rel = np.concatenate([cat("left_relief"), cat("right_relief")])
    stance = (spd < args.stance_speed) & (gap < args.stance_gap_max)
    out["stance_gap_all"] = stats(gap[stance])
    for name, lo, hi in bins:
        sel = stance & (rel >= lo) & (rel < hi)
        out[f"stance_gap[{name}]"] = stats(gap[sel])
    out["frames"] = int(len(results) and sum(r["n"] for r in results))
    out["feet_frames"] = int(len(gap))
    out["stance_fraction"] = float(stance.mean())
    out["penetration_gt2cm_all_frames"] = float(np.nanmean(gap < -0.02))
    out["penetration_gt2cm_stance"] = float(np.nanmean(gap[stance] < -0.02)) if stance.any() else float("nan")
    sup = np.fmin(cat("left_gap"), cat("right_gap"))
    out["support_gap_min_over_feet"] = stats(sup)
    out["pelvis_height_above_terrain"] = stats(cat("pelvis_h"))
    out["nan_fraction_gap"] = float(np.isnan(gap).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/vcj9002/scalebfm_ws/motions/terrain_raw")
    ap.add_argument("--layout", type=int, default=0)
    ap.add_argument("--num_clips", type=int, default=60)
    ap.add_argument("--max_frames", type=int, default=1000, help="max frames per clip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stance_speed", type=float, default=0.15, help="ankle speed below this (m/s) counts as stance")
    ap.add_argument("--stance_gap_max", type=float, default=0.05)
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "clips", f"layout_{args.layout}", "*.pkl")))
    assert files, f"no clips found for layout {args.layout} under {args.root}"
    random.Random(args.seed).shuffle(files)
    files = files[: args.num_clips]
    print(f"[validate] {len(files)} clips of layout {args.layout}", flush=True)

    mesh = load_mesh(args.root, args.layout)
    print(f"[validate] mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, bounds {mesh.bounds.round(2).tolist()}", flush=True)
    terrain = TerrainHeight(mesh)
    kin = Kinematics()

    clips = []
    for f in files:
        c = joblib.load(f)
        if c.get("meta", {}).get("smoke_pushed_env"):
            continue  # smoke-run clips with an injected push are not representative
        n = min(len(c["root_pos"]), args.max_frames)
        clips.append({k: (c[k][:n] if k != "fps" else c[k]) for k in ("root_pos", "root_rot", "dof_pos", "fps")})

    report = {}
    hypotheses = [("as_recorded(xyzw)", "xyzw", 0.0)]
    if args.diagnose:
        hypotheses += [("root_rot_as_wxyz", "wxyz", 0.0)]
    for name, qmode, _ in hypotheses:
        results = [analyse_clip(kin, terrain, c, qmode) for c in clips]
        report[name] = summarize(results, args)
        s = report[name]
        print(f"\n=== hypothesis: {name}", flush=True)
        print(f"  frames {s['frames']}, stance fraction {s['stance_fraction']:.3f}, NaN gap fraction {s['nan_fraction_gap']:.4f}")
        for key in ["stance_gap_all"] + [k for k in s if k.startswith("stance_gap[")]:
            v = s[key]
            if v.get("n", 0):
                print(f"  {key:26s} n={v['n']:7d}  mean {v['mean']*100:6.2f} cm  p5 {v['p5']*100:6.2f}  p50 {v['p50']*100:6.2f}  p95 {v['p95']*100:6.2f}")
        print(f"  penetration>2cm: all frames {s['penetration_gt2cm_all_frames']*100:.2f}%  stance {s['penetration_gt2cm_stance']*100:.2f}%")
        v = s["support_gap_min_over_feet"]
        print(f"  support gap (min over feet, all frames): mean {v['mean']*100:.2f} cm  p5 {v['p5']*100:.2f}  p95 {v['p95']*100:.2f}")
        v = s["pelvis_height_above_terrain"]
        print(f"  pelvis height above terrain: mean {v['mean']:.3f} m  p5 {v['p5']:.3f}  p95 {v['p95']:.3f}")
    if args.diagnose:
        base = report["as_recorded(xyzw)"]["stance_gap_all"]
        for dz in (-0.05, -0.03, 0.03, 0.05):
            print(f"[diagnose] a constant z offset of {dz:+.2f} m would move the mean stance gap from {base['mean']*100:.2f} to {(base['mean'] + dz)*100:.2f} cm")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "report": report}, f, indent=2)
        print(f"[validate] wrote {args.out}")


if __name__ == "__main__":
    main()
