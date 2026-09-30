"""Print the latest tensorboard scalars of a training run: python tb_summary.py <run name or dir> [tag substrings...]"""
import glob, os, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
d = run if os.path.isdir(run) else f"/home/vcj9002/magicloco/ScaleBFM/ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp/{run}"
ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
tags = sorted(ea.Tags()["scalars"])
want = sys.argv[2:]
for t in tags:
    if want and not any(w in t for w in want):
        continue
    ev = ea.Scalars(t)
    last = ev[-1]
    hist = [e.value for e in ev[-5:]]
    print(f"{t:44s} it {last.step:6d}  last {last.value:10.4f}   last5 mean {sum(hist)/len(hist):10.4f}   (n={len(ev)})")
