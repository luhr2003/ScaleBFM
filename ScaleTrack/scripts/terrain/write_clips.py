#!/usr/bin/env python3
"""Segment one raw MagicLoco terrain rollout into ScaleTrack clips (joblib .pkl) + index lines.

Launched in the background by ``record_magicloco_refs.py`` after every rollout (one host transfer per
rollout -> one raw .npz), so the simulator never waits on disk I/O. Pure numpy + joblib; runs in a
clean process, so the fork-based worker pool shares the raw arrays copy-on-write.

    python write_clips.py --raw <scratch>/raw/L0_r0.npz --layout_dir <out>/layouts/layout_0 --out <out>
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import terrain_clip_utils as U  # noqa: E402

_G: dict = {}


def _write_one(job):
    import joblib

    e, seg_id, s, t, end_reason = job
    rp, rq, dof, cmd = _G["root_pos"], _G["root_quat"], _G["dof_pos"], _G["cmd"]
    sl = slice(s, t + 1)
    pos = np.ascontiguousarray(rp[sl, e], dtype=np.float32)
    quat_xyzw = np.ascontiguousarray(U.wxyz_to_xyzw(rq[sl, e]), dtype=np.float64)
    # re-normalise (float32 storage); IsaacLab quats are unit up to ~1e-7
    quat_xyzw /= np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    # sign continuity (q and -q are the same rotation): first frame w >= 0, then no flips between frames
    if quat_xyzw[0, 3] < 0:
        quat_xyzw[0] *= -1.0
    flips = np.cumsum(np.r_[0, (np.einsum("ij,ij->i", quat_xyzw[1:], quat_xyzw[:-1]) < 0).astype(np.int64)])
    # dot sign relative to the ORIGINAL previous frame; accumulate parity of flips
    quat_xyzw *= np.where(flips % 2 == 1, -1.0, 1.0)[:, None]
    quat_xyzw = quat_xyzw.astype(np.float32)
    dofs = np.ascontiguousarray(dof[sl, e], dtype=np.float32)
    cmds = np.ascontiguousarray(cmd[sl, e], dtype=np.float32)
    lk: U.TerrainLookup = _G["lookup"]
    r0, c0 = int(_G["env_row"][e]), int(_G["env_col"][e])
    tile = lk.tile_info(r0, c0)
    seed, rollout = int(_G["layout_seed"]), int(_G["rollout"])
    name = f"L{seed}_r{rollout}_e{e}_s{seg_id}"
    arm_mode = "train_rho1" if bool(_G["arm_train"][e]) else "nominal_rho0"
    meta = {
        "layout_seed": seed,
        "rollout": rollout,
        "rollout_seed": int(_G["rollout_seed"]),
        "env_id": int(e),
        "segment_id": int(seg_id),
        "start_tile": tile,
        "arm_mode": arm_mode,
        "commands": cmds,
        "command_names": list(U.COMMAND_NAMES),
        "end_reason": end_reason,
        "rollout_frames": [int(s), int(t)],
        "split": _G["split"],
        "source": "MagicLoco pi_L v4 terrain (model_champion_tc2_it750), IsaacLab Play task, clean eval caliber",
    }
    if _G.get("smoke_pushed") is not None and bool(_G["smoke_pushed"][e]):
        meta["smoke_pushed_env"] = True
    d = {"root_pos": pos, "root_rot": quat_xyzw, "dof_pos": dofs, "fps": int(U.FPS), "meta": meta}
    path = os.path.join(_G["clip_dir"], name + ".pkl")
    joblib.dump(d, path)
    lab = U.clip_labels(pos, lk)
    line = {
        "name": name,
        "file": os.path.relpath(path, _G["out"]),
        "layout": seed,
        "split": _G["split"],
        "rollout": rollout,
        "env_id": int(e),
        "segment_id": int(seg_id),
        "n_frames": int(pos.shape[0]),
        "seconds": round(pos.shape[0] / U.FPS, 3),
        "start_tile_type": tile["type"],
        "start_tile_row": tile["row"],
        "start_tile_col": tile["col"],
        "start_tile_difficulty": tile["difficulty"],
        "start_tile_step_height": tile["step_height"],
        "start_tile_param": tile["param_name"],
        "start_tile_param_value": tile["param_value"],
        "arm_mode": arm_mode,
        "end_reason": end_reason,
        "cmd_vx_mean": round(float(cmds[:, 0].mean()), 4),
        **lab,
    }
    if meta.get("smoke_pushed_env"):
        line["smoke_pushed_env"] = True
    return line


def _segment_envs(env_ids):
    res = []
    for e in env_ids:
        segs = U.env_segments(_G["term"][:, e], _G["trunc"][:, e])
        for seg_id, (s, t, why) in enumerate(segs):
            t_keep, end_reason, sinfo = U.trim_segment(s, t, why, _G["root_pos"][:, e], _G["root_quat"][:, e],
                                                       _G["lookup"])
            res.append((e, seg_id, s, t, why, t_keep, end_reason, sinfo))
    return res


def merge_index(out: str) -> int:
    parts = sorted(glob.glob(os.path.join(out, "index_parts", "*.jsonl")))
    lock_path = os.path.join(out, ".index.lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        n = 0
        tmp = os.path.join(out, "index.jsonl.tmp")
        with open(tmp, "w") as fo:
            for p in parts:
                with open(p) as fi:
                    for ln in fi:
                        if ln.strip():
                            fo.write(ln if ln.endswith("\n") else ln + "\n")
                            n += 1
        os.replace(tmp, os.path.join(out, "index.jsonl"))
        fcntl.flock(lf, fcntl.LOCK_UN)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--layout_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--joblib_path", default="/home/vcj9002/scalebfm_ws/tmp/pylibs_joblib")
    ap.add_argument("--delete_raw", action="store_true")
    ap.add_argument("--merge_only", action="store_true")
    a = ap.parse_args()
    if a.joblib_path and os.path.isdir(a.joblib_path):
        sys.path.insert(0, a.joblib_path)
    import joblib  # noqa: F401  (fail early)

    if a.merge_only:
        print(f"[write_clips] merged {merge_index(a.out)} index lines", flush=True)
        return

    t0 = time.time()
    raw = np.load(a.raw)
    for k in ("root_pos", "root_quat", "dof_pos", "cmd", "term", "trunc", "env_row", "env_col", "arm_train"):
        _G[k] = raw[k]
    _G["term_code"] = raw["term_code"] if "term_code" in raw.files else None
    _G["smoke_pushed"] = raw["smoke_pushed"] if "smoke_pushed" in raw.files else None
    info = json.loads(str(raw["info_json"]))
    _G["layout_seed"], _G["rollout"], _G["rollout_seed"] = info["layout_seed"], info["rollout"], info["rollout_seed"]
    _G["split"] = info["split"]
    _G["lookup"] = U.TerrainLookup(a.layout_dir)
    _G["out"] = a.out
    _G["clip_dir"] = os.path.join(a.out, "clips", f"layout_{info['layout_seed']}")
    os.makedirs(_G["clip_dir"], exist_ok=True)
    os.makedirs(os.path.join(a.out, "index_parts"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "rollout_stats"), exist_ok=True)

    T, N = _G["term"].shape
    jobs, drops, n_raw = [], {"too_short": 0}, 0
    reasons, seg_log, bad_crit = {}, [], {}
    ctx = mp.get_context("fork")
    chunks = [list(range(i, min(N, i + 64))) for i in range(0, N, 64)]
    with ctx.Pool(a.workers) as pool:
        for res in pool.imap(_segment_envs, chunks):
            for e, seg_id, s, t, why, t_keep, end_reason, sinfo in res:
                n_raw += 1
                reasons[why] = reasons.get(why, 0) + 1
                if "bad_criterion" in sinfo:
                    bad_crit[sinfo["bad_criterion"]] = bad_crit.get(sinfo["bad_criterion"], 0) + 1
                if t_keep < 0:
                    drops[sinfo["drop"]] = drops.get(sinfo["drop"], 0) + 1
                else:
                    jobs.append((e, seg_id, s, t_keep, end_reason))
                seg_log.append([e, seg_id, s, t, why, t_keep, end_reason])
    # a rewrite of the same rollout must not leave clips behind that the new segmentation no longer produces
    tag = f"L{info['layout_seed']}_r{info['rollout']}"
    keep_names = {f"{tag}_e{e}_s{sid}.pkl" for e, sid, _s, _t, _r in jobs}
    stale = [f for f in glob.glob(os.path.join(_G["clip_dir"], f"{tag}_e*_s*.pkl")) if os.path.basename(f) not in keep_names]
    for f in stale:
        os.remove(f)
    lines = []
    with ctx.Pool(a.workers) as pool:
        for ln in pool.imap_unordered(_write_one, jobs, chunksize=16):
            lines.append(ln)
    lines.sort(key=lambda d: (d["env_id"], d["segment_id"]))
    part = os.path.join(a.out, "index_parts", tag + ".jsonl")
    with open(part + ".tmp", "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    os.replace(part + ".tmp", part)
    kept_frames = int(sum(ln["n_frames"] for ln in lines))
    stats = {
        "tag": tag, **info, "num_envs": int(N), "rollout_frames": int(T),
        "raw_segments": n_raw, "raw_segment_end_reasons": reasons, "clips_written": len(lines),
        "dropped": drops, "kept_frames": kept_frames, "kept_hours": kept_frames / U.FPS / 3600.0,
        "raw_frames": int(T * N), "end_reasons_kept": {}, "bad_frame_criterion": bad_crit,
        "stale_clips_removed": len(stale),
        "segments": seg_log,
    }
    for ln in lines:
        stats["end_reasons_kept"][ln["end_reason"]] = stats["end_reasons_kept"].get(ln["end_reason"], 0) + 1
    # D3 aggregates over KEPT frames: commanded vx / actual horizontal pelvis speed histograms, env tiles
    owner = -np.ones((T, N), np.int64)
    for ji, (e, _sid, s, t, _why) in enumerate(jobs):
        owner[s:t + 1, e] = ji
    kept = owner >= 0
    vx_edges = np.round(np.arange(-0.6, 1.0001, 0.05), 3)
    sp_edges = np.round(np.arange(0.0, 2.0001, 0.05), 3)
    stats["hist_cmd_vx"] = {"edges": vx_edges.tolist(),
                            "counts": np.histogram(_G["cmd"][..., 0][kept], bins=vx_edges)[0].tolist()}
    same = (owner[1:] == owner[:-1]) & (owner[1:] >= 0)
    spd = np.linalg.norm(np.diff(_G["root_pos"][..., :2], axis=0), axis=-1) * U.FPS
    stats["hist_speed"] = {"edges": sp_edges.tolist(), "counts": np.histogram(np.clip(spd[same], 0, 1.9999),
                                                                               bins=sp_edges)[0].tolist()}
    stats["env_row"] = _G["env_row"].astype(int).tolist()
    stats["env_col"] = _G["env_col"].astype(int).tolist()
    stats["arm_train"] = _G["arm_train"].astype(bool).tolist()
    with open(os.path.join(a.out, "rollout_stats", tag + ".json"), "w") as f:
        json.dump(stats, f)
    n_idx = merge_index(a.out)
    if a.delete_raw:
        os.remove(a.raw)
    print(f"[write_clips] {tag}: raw_segments={n_raw} clips={len(lines)} dropped={drops} "
          f"kept_hours={stats['kept_hours']:.2f} index_lines={n_idx} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
