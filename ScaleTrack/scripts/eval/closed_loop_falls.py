"""Recount falls of the MagicSim closed-loop cells (TestOutput/bfm_planner/<queue>/<cell>/{report.json,trace.npz}) with a pose criterion.
The harness flag 'pelvis < 0.40 m above the ground under the pelvis' gives false alarms when stairs are descended (the pelvis z is already on the
lower step while its xy is still over the upper tread). A draw is a TRUE fall here if: min cos(tilt) < 0.5 (tilt > 60 deg), or the pelvis stays
below 0.45 m above the ground for more than 0.5 s, or the final pelvis height above the ground is below 0.5 m.
usage: closed_loop_falls.py <cell> <queue> [<queue> ...]     e.g. closed_loop_falls.py down_h26_open_K16 queue queue_soupV3a queue_soupV4a
"""
import json, sys
import numpy as np

ROOT = "/home/vcj9002/magicloco/MagicSim/TestOutput/bfm_planner"
cell, queues = sys.argv[1], sys.argv[2:]
for q in queues:
    z = np.load(f"{ROOT}/{q}/{cell}/trace.npz", allow_pickle=True)
    r = json.load(open(f"{ROOT}/{q}/{cell}/report.json"))
    rows = []
    for p in r["pairs_result"]:
        s = int(p["name"][1:])
        root, quat, ground = z["main_root"][:, s], z["main_quat"][:, s], z["main_ground"][:, s]
        cos_tilt = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)  # quaternion wxyz
        above = root[:, 2] - ground
        run = best = 0
        for v in above < 0.45:
            run = run + 1 if v else 0
            best = max(best, run)
        true_fall = bool(cos_tilt.min() < 0.5 or best * 0.02 > 0.5 or above[-10:].mean() < 0.5)
        rows.append((p["name"], p["main"]["fell"], true_fall, above.min(), cos_tilt.min(), above[-10:].mean(), p["ghost"]["fell"]))
    print(f"== {q}/{cell}: harness falls {sum(x[1] for x in rows)}/{len(rows)}, true falls {sum(x[2] for x in rows)}/{len(rows)}, ghost falls {sum(x[6] for x in rows)}")
    for n, rep, tf, mn, mc, fa, gf in rows:
        print(f"   {n}: harness={rep!s:5} true={tf!s:5} min_above={mn:.3f} min_cos_tilt={mc:.3f} final_above={fa:.3f} ghost_fell={gf}")
