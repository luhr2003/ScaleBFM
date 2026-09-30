#!/usr/bin/env python3
"""D3 dataset statistics for the MagicLoco terrain reference set.

Reads <out>/index.jsonl, <out>/rollout_stats/L*_r*.json (writer stats) and <out>/layouts/layout_*/meta.json;
writes <out>/dataset_stats.json and <out>/dataset_stats.md (and prints the markdown).

    python dataset_stats.py --out /home/vcj9002/scalebfm_ws/motions/terrain_raw
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
from collections import defaultdict

import numpy as np

FPS = 50


def du_gb(path: str) -> float:
    try:
        out = subprocess.check_output(["du", "-sb", path], text=True)
        return int(out.split()[0]) / 1e9
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = a.out
    lines = [json.loads(ln) for ln in open(os.path.join(out, "index.jsonl")) if ln.strip()]
    metas = {}
    for mp in sorted(glob.glob(os.path.join(out, "layouts", "layout_*", "meta.json"))):
        m = json.load(open(mp))
        metas[int(m["layout_seed"])] = m
    names = next(iter(metas.values()))["sub_terrain_names"] if metas else []
    nrows = next(iter(metas.values()))["grid"]["num_rows"] if metas else 10

    S: dict = {}
    # ---------------- overall
    by_split = defaultdict(lambda: {"clips": 0, "frames": 0})
    layouts = defaultdict(set)
    for ln in lines:
        by_split[ln["split"]]["clips"] += 1
        by_split[ln["split"]]["frames"] += ln["n_frames"]
        layouts[ln["split"]].add((ln["layout"], ln["rollout"]))
    S["overall"] = {sp: {"clips": v["clips"], "frames": v["frames"], "hours": round(v["frames"] / FPS / 3600, 3),
                         "layout_rollouts": sorted(layouts[sp])} for sp, v in by_split.items()}
    S["total_clips"] = len(lines)
    S["total_frames"] = int(sum(ln["n_frames"] for ln in lines))
    S["total_hours"] = round(S["total_frames"] / FPS / 3600, 3)
    S["disk_gb"] = {"clips": round(du_gb(os.path.join(out, "clips")), 3),
                    "layouts": round(du_gb(os.path.join(out, "layouts")), 3),
                    "total_out": round(du_gb(out), 3)}
    # ---------------- hours per terrain type (frames spent over each tile type) and per start type x row
    over = defaultdict(float)
    start_type_row = defaultdict(float)
    for ln in lines:
        for t, f in ln["terrain_frac"].items():
            over[t] += f * ln["n_frames"]
        start_type_row[(ln["start_tile_type"], ln["start_tile_row"])] += ln["n_frames"]
    S["hours_over_terrain_type"] = {t: round(v / FPS / 3600, 3) for t, v in sorted(over.items(), key=lambda x: -x[1])}
    tab = {t: [round(start_type_row.get((t, r), 0.0) / FPS / 3600, 3) for r in range(nrows)] for t in names}
    S["hours_by_start_type_and_row"] = tab
    # ---------------- step heights of the stair tiles per row (layout 0 as reference)
    # ---------------- non-fall rate per start tile type x row (env-rollouts with zero falls) from writer stats
    nf = defaultdict(lambda: [0, 0])          # (type,row) -> [env_rollouts_without_fall, env_rollouts]
    falls_total, env_rollouts_total, fall_events = 0, 0, 0
    drops = defaultdict(int)
    raw_frames, kept_frames = 0, 0
    hist_vx, hist_sp, vx_edges, sp_edges = None, None, None, None
    raw_reasons = defaultdict(int)
    term_code_counts = defaultdict(int)
    for sp in sorted(glob.glob(os.path.join(out, "rollout_stats", "L*_r*.json"))):
        if sp.endswith("_recorder_checks.json"):
            continue
        st = json.load(open(sp))
        seed = int(st["layout_seed"])
        m = metas.get(seed)
        if m is None or "env_row" not in st:
            continue
        ttype = np.array(m["tile_type_index"])
        rows, cols = np.array(st["env_row"]), np.array(st["env_col"])
        fell = np.zeros(len(rows), bool)
        for e, _sid, s, t, why, _keep, _er in st["segments"]:
            if why == "fall":
                fell[e] = True
                fall_events += 1
        for e in range(len(rows)):
            key = (names[ttype[rows[e], cols[e]]], int(rows[e]))
            nf[key][1] += 1
            nf[key][0] += int(not fell[e])
        falls_total += int(fell.sum())
        env_rollouts_total += len(rows)
        for k, v in st["dropped"].items():
            drops[k] += v
        for k, v in st["raw_segment_end_reasons"].items():
            raw_reasons[k] += v
        raw_frames += st["raw_frames"]
        kept_frames += st["kept_frames"]
        hv, hs = np.array(st["hist_cmd_vx"]["counts"]), np.array(st["hist_speed"]["counts"])
        hist_vx = hv if hist_vx is None else hist_vx + hv
        hist_sp = hs if hist_sp is None else hist_sp + hs
        vx_edges, sp_edges = st["hist_cmd_vx"]["edges"], st["hist_speed"]["edges"]
    S["non_fall_rate_by_type_row"] = {t: [round(nf[(t, r)][0] / nf[(t, r)][1], 4) if nf[(t, r)][1] else None
                                          for r in range(nrows)] for t in names}
    S["non_fall_rate_by_type"] = {t: round(sum(nf[(t, r)][0] for r in range(nrows)) /
                                           max(1, sum(nf[(t, r)][1] for r in range(nrows))), 4) for t in names}
    S["env_rollouts"] = env_rollouts_total
    S["env_rollouts_with_fall"] = falls_total
    S["fall_events"] = fall_events
    S["non_fall_rate_overall"] = round(1 - falls_total / max(1, env_rollouts_total), 4)
    S["raw_segment_end_reasons"] = dict(raw_reasons)
    S["dropped_segments"] = dict(drops)
    S["kept_fraction_of_simulated_frames"] = round(kept_frames / max(1, raw_frames), 4)
    if hist_vx is not None:
        e = np.array(vx_edges)
        tot = hist_vx.sum()
        coarse = [(-0.61, -0.001, "backward (<0)"), (-0.001, 0.05, "stand (0)"), (0.05, 0.25, "0.05-0.25"),
                  (0.25, 0.5, "0.25-0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 1.01, "0.7-0.9")]
        cen = 0.5 * (e[:-1] + e[1:])
        S["cmd_vx_distribution_frames"] = {lab: round(float(hist_vx[(cen > lo) & (cen <= hi)].sum() / tot), 4)
                                           for lo, hi, lab in coarse}
        S["cmd_vx_hist"] = {"edges": vx_edges, "counts": hist_vx.tolist()}
        s_e = np.array(sp_edges)
        s_c = 0.5 * (s_e[:-1] + s_e[1:])
        S["actual_speed_distribution_frames"] = {lab: round(float(hist_sp[(s_c > lo) & (s_c <= hi)].sum() / hist_sp.sum()), 4)
                                                 for lo, hi, lab in [(0, 0.1, "<0.1 m/s"), (0.1, 0.3, "0.1-0.3"),
                                                                     (0.3, 0.5, "0.3-0.5"), (0.5, 0.7, "0.5-0.7"),
                                                                     (0.7, 0.9, "0.7-0.9"), (0.9, 2.1, ">0.9")]}
    arm = defaultdict(float)
    endr = defaultdict(lambda: [0, 0.0])
    for ln in lines:
        arm[ln["arm_mode"]] += ln["n_frames"] / FPS / 3600
        endr[ln["end_reason"]][0] += 1
        endr[ln["end_reason"]][1] += ln["n_frames"] / FPS / 3600
    S["hours_by_arm_mode"] = {k: round(v, 3) for k, v in arm.items()}
    S["clips_by_end_reason"] = {k: {"clips": v[0], "hours": round(v[1], 3)} for k, v in endr.items()}
    nfr = np.array([ln["n_frames"] for ln in lines])
    S["clip_length_s"] = {"mean": round(float(nfr.mean() / FPS), 2), "min": round(float(nfr.min() / FPS), 2),
                          "p50": round(float(np.median(nfr) / FPS), 2), "max": round(float(nfr.max() / FPS), 2)}
    # step heights per row (from layout meta, stairs tiles, all layouts)
    sh = defaultdict(list)
    for m in metas.values():
        for tl in m["tiles"]:
            if tl["param_name"] == "step_height":
                sh[tl["row"]].append(tl["step_height"])
    S["stair_step_height_by_row"] = {r: [round(min(v), 3), round(max(v), 3)] for r, v in sorted(sh.items())}
    with open(os.path.join(out, "dataset_stats.json"), "w") as f:
        json.dump(S, f, indent=1)

    # ---------------- markdown
    md = ["# MagicLoco terrain reference set - statistics", ""]
    md.append(f"clips {S['total_clips']}, frames {S['total_frames']}, hours {S['total_hours']}, disk {S['disk_gb']}")
    for sp, v in S["overall"].items():
        md.append(f"- {sp}: {v['clips']} clips, {v['hours']} h, layout/rollouts {v['layout_rollouts']}")
    md.append(f"- non-fall rate (20 s env-rollouts without any fall): {S['non_fall_rate_overall']} "
              f"({S['env_rollouts_with_fall']}/{S['env_rollouts']} env-rollouts had a fall; {S['fall_events']} fall events)")
    md.append(f"- kept fraction of simulated frames: {S['kept_fraction_of_simulated_frames']}; dropped segments {S['dropped_segments']}")
    md.append(f"- hours by arm mode: {S['hours_by_arm_mode']}")
    md.append(f"- clips by end reason: {S['clips_by_end_reason']}")
    md.append(f"- clip length (s): {S['clip_length_s']}")
    md.append("")
    md.append("## Hours spent over each terrain type (tile under the pelvis)")
    md.append("| type | hours |")
    md.append("|---|---|")
    for t, h in S["hours_over_terrain_type"].items():
        md.append(f"| {t} | {h} |")
    md.append("")
    md.append("## Hours by start tile type x difficulty row (row 0 easiest)")
    md.append("| type | " + " | ".join(f"r{r}" for r in range(nrows)) + " | total |")
    md.append("|---|" + "---|" * (nrows + 1))
    for t, v in tab.items():
        md.append(f"| {t} | " + " | ".join(f"{x:.2f}" for x in v) + f" | {sum(v):.2f} |")
    md.append("")
    md.append("stair step height range per row (m): " + ", ".join(f"r{r}: {v[0]}-{v[1]}" for r, v in S["stair_step_height_by_row"].items()))
    md.append("")
    md.append("## Non-fall rate by start tile type x row (fraction of 20 s env-rollouts without a fall)")
    md.append("| type | " + " | ".join(f"r{r}" for r in range(nrows)) + " | all |")
    md.append("|---|" + "---|" * (nrows + 1))
    for t, v in S["non_fall_rate_by_type_row"].items():
        md.append(f"| {t} | " + " | ".join("-" if x is None else f"{x:.3f}" for x in v) + f" | {S['non_fall_rate_by_type'][t]:.3f} |")
    md.append("")
    if "cmd_vx_distribution_frames" in S:
        md.append(f"commanded vx (fraction of kept frames): {S['cmd_vx_distribution_frames']}")
        md.append(f"actual horizontal pelvis speed (fraction of kept frames): {S['actual_speed_distribution_frames']}")
    txt = "\n".join(md)
    with open(os.path.join(out, "dataset_stats.md"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
