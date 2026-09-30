"""Convert the BONES-SEED G1 CSV release to ScaleRetarget .pkl files in parallel, excluding the held-out test clips.

Uses ScaleRetarget's own `convert_motion` (120 fps CSV -> 30 fps pkl with root_pos/root_rot(xyzw)/dof_pos/fps) so the
output is identical to the documented single-process recipe. Output files are named `<date>-<stem>.pkl`, the same
naming as the official BONES Test Set, which is used to drop the test clips.
"""
import argparse
import importlib.util
import multiprocessing as mp
import os
import sys
from pathlib import Path

import yaml

SCALERETARGET = "/home/vcj9002/magicloco/ScaleBFM/ScaleRetarget/scaleretarget/utils/convert_bones_to_ours.py"


def _load_convert_motion():
    spec = importlib.util.spec_from_file_location("convert_bones_to_ours", SCALERETARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.convert_motion


def _work(job):
    src, dst = job
    try:
        _load_convert_motion()(Path(src), Path(dst), 120, 30)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{src}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_root", required=True, help="dir containing <date>/<name>.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_yaml", required=True, help="YAML of the BONES test set (names to exclude)")
    ap.add_argument("--workers", type=int, default=72)
    args = ap.parse_args()

    test_names = set(yaml.safe_load(open(args.test_yaml)).keys())
    os.makedirs(args.out_dir, exist_ok=True)
    jobs, excluded, seen = [], 0, set()
    for src in sorted(Path(args.csv_root).glob("*/*.csv")):
        name = f"{src.parent.name}-{src.stem}"
        if name in test_names:
            excluded += 1
            seen.add(name)
            continue
        dst = Path(args.out_dir) / f"{name}.pkl"
        if not dst.exists():
            jobs.append((str(src), str(dst)))
    missing = test_names - seen
    print(f"jobs {len(jobs)}, excluded test clips {excluded}/{len(test_names)}, test clips without csv: {len(missing)}", flush=True)
    with mp.Pool(args.workers) as pool:
        errs = [e for e in pool.imap_unordered(_work, jobs, chunksize=64) if e]
    print(f"done, {len(errs)} errors", flush=True)
    for e in errs[:20]:
        print("ERR", e)
