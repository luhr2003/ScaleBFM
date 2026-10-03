"""Summarise a squat probe: pelvis (anchor) height of the robot above the reference during the hold phase, per clip, from the traces written by
`eval_modes.py --trace_out <prefix>` (files <prefix>.<config>.npz). Prints one compact line.
usage: squat_summary.py <trace prefix> [config ...]       (default configs: mode7_global mode4_global)
"""
import sys

import numpy as np

prefix = sys.argv[1]
configs = sys.argv[2:] or ["mode7_global", "mode4_global"]
VAL = ["val_squat032", "val_squat027", "val_squat022", "val_squat017"]
TRAIN = ["squat035", "squat030", "squat025", "squat020"]
parts = []
for cfg in configs:
    try:
        z = np.load(f"{prefix}.{cfg}.npz", allow_pickle=True)
    except Exception:
        continue
    rz, ez, names = z["robot_z"], z["ref_z"], [str(n) for n in z["names"]]
    def offset(key):
        i = [k for k, n in enumerate(names) if key in n]
        if not i:
            return float("nan")
        i = i[0]
        hold = ez[:, i] < ez[:, i].min() + 0.01
        return 100.0 * float((rz[:, i][hold] - ez[:, i][hold]).mean())
    val = [offset(k) for k in VAL]
    train = [offset(k) for k in TRAIN]
    parts.append(f"{cfg.replace('_global', '')}: held-out val {'/'.join(f'{v:+.1f}' for v in val)} (mean {np.nanmean(val):+.1f}) | train depths {'/'.join(f'{v:+.1f}' for v in train)} (mean {np.nanmean(train):+.1f}) cm")
print("squat hold-phase pelvis above reference: " + " ;; ".join(parts))
