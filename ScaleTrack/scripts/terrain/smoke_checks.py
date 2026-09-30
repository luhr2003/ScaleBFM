#!/usr/bin/env python3
"""D1 logic checks of a recorded rollout: raw rollout npz (scratch) + written clips.

(i)   no frame belongs to two clips, no clip spans a reset, clip arrays == raw arrays (slicing),
(ii)  jump bounds inside clips (pelvis speed, joint speed, pelvis angular speed) vs across resets,
(iii) reset handling: every split, the 1.5 s fall drop and short-segment drops, with counts,
(iv)  command fidelity: recorder checks (set vs policy-observed) + constancy inside command segments.

    python smoke_checks.py --raw <scratch>/raw/L0_r0.npz --out <out_root> [--json report.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrain_clip_utils as U  # noqa: E402


def quat_angle_rate(q_xyzw: np.ndarray, fps: int) -> np.ndarray:
    d = np.abs(np.einsum("ij,ij->i", q_xyzw[1:], q_xyzw[:-1])).clip(0, 1)
    return 2.0 * np.arccos(d) * fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--joblib_path", default="/home/vcj9002/scalebfm_ws/tmp/pylibs_joblib")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max_clips", type=int, default=100000)
    a = ap.parse_args()
    if os.path.isdir(a.joblib_path):
        sys.path.insert(0, a.joblib_path)
    import joblib

    raw = np.load(a.raw)
    info = json.loads(str(raw["info_json"]))
    seed, r = info["layout_seed"], info["rollout"]
    term, trunc = raw["term"], raw["trunc"]
    done = term | trunc
    T, N = term.shape
    rp, rq, dof, cmd = raw["root_pos"], raw["root_quat"], raw["dof_pos"], raw["cmd"]
    rep = {"raw": a.raw, "T": int(T), "N": int(N), "recorder_checks": info["checks"]}

    files = sorted(glob.glob(os.path.join(a.out, "clips", f"layout_{seed}", f"L{seed}_r{r}_e*_s*.pkl")))[: a.max_clips]
    owner = -np.ones((T, N), np.int64)
    viol = {"overlap": 0, "spans_reset": 0, "slice_mismatch": 0, "quat_sign_or_convention": 0, "keys": 0, "fps": 0}
    in_clip = {"pelvis_speed": [], "joint_speed": [], "ang_speed": []}
    cmd_seg_lengths, n_frames = [], 0
    for ci, fpath in enumerate(files):
        d = joblib.load(fpath)
        if set(d.keys()) != {"root_pos", "root_rot", "dof_pos", "fps", "meta"}:
            viol["keys"] += 1
        if not (isinstance(d["fps"], int) and d["fps"] == 50):
            viol["fps"] += 1
        m = d["meta"]
        e, (s, t) = m["env_id"], m["rollout_frames"]
        n = t - s + 1
        n_frames += n
        assert d["root_pos"].shape == (n, 3) and d["dof_pos"].shape == (n, 29) and d["root_rot"].shape == (n, 4)
        if (owner[s:t + 1, e] >= 0).any():
            viol["overlap"] += 1
        owner[s:t + 1, e] = ci
        if done[s:t, e].any():                      # a done strictly before the last kept frame -> spans a reset
            viol["spans_reset"] += 1
        if not (np.array_equal(d["root_pos"], rp[s:t + 1, e]) and np.array_equal(d["dof_pos"], dof[s:t + 1, e])
                and np.array_equal(m["commands"], cmd[s:t + 1, e])):
            viol["slice_mismatch"] += 1
        q_raw = U.wxyz_to_xyzw(rq[s:t + 1, e].astype(np.float64))
        q_raw /= np.linalg.norm(q_raw, axis=1, keepdims=True)
        dots = np.abs(np.einsum("ij,ij->i", q_raw, d["root_rot"].astype(np.float64)))
        if dots.min() < 1 - 1e-6:
            viol["quat_sign_or_convention"] += 1
        in_clip["pelvis_speed"].append(np.linalg.norm(np.diff(d["root_pos"], axis=0), axis=1).max() * 50)
        in_clip["joint_speed"].append(np.abs(np.diff(d["dof_pos"], axis=0)).max() * 50)
        in_clip["ang_speed"].append(quat_angle_rate(d["root_rot"].astype(np.float64), 50).max())
        c = m["commands"]
        ch = np.nonzero(np.any(np.abs(np.diff(c, axis=0)) > 0, axis=1))[0]
        bounds = np.r_[0, ch + 1, n]
        seg = np.diff(bounds)
        if len(seg) > 2:
            cmd_seg_lengths.extend(seg[1:-1].tolist())    # interior command segments (not cut by clip ends)
    rep["n_clips_checked"] = len(files)
    rep["n_frames_in_clips"] = int(n_frames)
    rep["violations"] = viol
    for k, v in in_clip.items():
        v = np.asarray(v)
        rep[f"in_clip_max_{k}"] = {"max": float(v.max()), "p99": float(np.percentile(v, 99)), "median": float(np.median(v))}
    # across resets (frame k -> k+1 with done[k]): what a joined clip would contain
    ks, es = np.nonzero(done[:-1])
    if len(ks):
        jump = np.linalg.norm(rp[ks + 1, es] - rp[ks, es], axis=1) * 50
        jj = np.abs(dof[ks + 1, es] - dof[ks, es]).max(axis=1) * 50
        rep["across_reset_pelvis_speed"] = {"n": int(len(ks)), "min": float(jump.min()), "median": float(np.median(jump)),
                                            "max": float(jump.max())}
        rep["across_reset_joint_speed"] = {"min": float(jj.min()), "median": float(np.median(jj)), "max": float(jj.max())}
    # (iii) reset events and drops
    ev = []
    for e in range(N):
        for s, t, why in U.env_segments(term[:, e], trunc[:, e]):
            if why in ("fall", "timeout"):
                kept = [os.path.basename(f) for f in files if f"_e{e}_" in os.path.basename(f)]
                ev.append({"env": int(e), "segment": [int(s), int(t)], "reason": why,
                           "term_code": int(raw["term_code"][t, e]) if "term_code" in raw.files else None})
    rep["reset_events"] = ev
    stats_path = os.path.join(a.out, "rollout_stats", f"L{seed}_r{r}.json")
    if os.path.exists(stats_path):
        st = json.load(open(stats_path))
        rep["writer_stats"] = {k: st[k] for k in ("raw_segments", "raw_segment_end_reasons", "clips_written", "dropped",
                                                  "end_reasons_kept", "kept_hours")}
        segs = st["segments"]
        falls = [s_ for s_ in segs if s_[4] == "fall"]
        rep["fall_segments"] = [{"env": s_[0], "seg": s_[1], "raw": [s_[2], s_[3]], "kept_end": s_[5],
                                 "dropped_frames": (s_[3] - s_[5]) if s_[5] >= 0 else s_[3] - s_[2] + 1,
                                 "end_reason": s_[6]} for s_ in falls]
    cl = np.asarray(cmd_seg_lengths)
    rep["interior_command_segment_frames"] = ({"n": int(cl.size), "min": int(cl.min()), "max": int(cl.max()),
                                               "mean": float(cl.mean())} if cl.size else {"n": 0})
    # frames not in any clip: which reasons
    rep["frames_not_in_clips"] = int((owner < 0).sum())
    print(json.dumps({k: v for k, v in rep.items() if k != "reset_events"}, indent=1))
    print("reset events:", json.dumps(rep["reset_events"]))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    ok = all(v == 0 for v in viol.values())
    print("D1 VIOLATIONS: NONE" if ok else f"D1 VIOLATIONS: {viol}")


if __name__ == "__main__":
    main()
