"""Assemble the YAML files of the terrain fine-tuning run.

Inputs
  --flat_dirs       directories with processed flat clips (.npz), optionally `dir:N` to keep a fixed random sample of N
                    clips of that directory, e.g. processed/bones:60000 processed/lafan
  --terrain_root    terrain recorder output (index.jsonl with `name`, `layout`, `split`)
  --terrain_dir     directory with the processed terrain clips (.npz named like the recorder clips)
Outputs (in --out_dir)
  train_all.yaml            flat clips + terrain clips of the training layouts
  clip_meta.json            terrain clip name -> layout seed (flat clips are absent)
  eval_terrain_test.yaml    terrain clips of the held-out layouts, clip_meta_test.json
  eval_terrain_train_sample.yaml  a fixed random sample of training terrain clips (sanity), clip_meta_train_sample.json
Terrain clips that are missing on disk, shorter than --min_frames, or listed as instrumented smoke clips are skipped.
"""

import argparse
import glob
import json
import os
import random

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat_dirs", nargs="+", required=True)
    ap.add_argument("--terrain_root", required=True)
    ap.add_argument("--terrain_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_frames", type=int, default=100)
    ap.add_argument("--train_sample", type=int, default=2000)
    ap.add_argument("--terrain_max_clips", type=int, default=0,
                    help="keep at most this many training terrain clips (equal share per layout, fixed random sample)")
    ap.add_argument("--train_layouts", type=int, nargs="*", default=None,
                    help="only use training terrain clips of these layout seeds (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    flat = {}
    for spec in args.flat_dirs:
        d, _, n = spec.partition(":")
        paths = sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True))
        if n:
            paths = sorted(random.Random(args.seed).sample(paths, min(int(n), len(paths))))
        for p in paths:
            name = os.path.basename(p)[:-4]
            if name in flat:
                raise ValueError(f"duplicate flat clip name {name}")
            flat[name] = p
        print(f"  {d}: {len(paths)} clips")
    print(f"flat clips: {len(flat)}")

    train_t, test_t, meta_train, meta_test = {}, {}, {}, {}
    skipped = {"missing": 0, "short": 0, "smoke": 0}
    with open(os.path.join(args.terrain_root, "index.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("smoke_pushed_env"):
                skipped["smoke"] += 1
                continue
            if rec["n_frames"] < args.min_frames:
                skipped["short"] += 1
                continue
            path = os.path.join(args.terrain_dir, rec["name"] + ".npz")
            if not os.path.exists(path):
                skipped["missing"] += 1
                continue
            if args.train_layouts is not None and rec["split"] != "test" and rec["layout"] not in args.train_layouts:
                continue
            if rec["split"] == "test":
                test_t[rec["name"]] = path
                meta_test[rec["name"]] = {"layout": rec["layout"], "max_step": rec.get("max_step_height", 0.0)}
            else:
                train_t[rec["name"]] = path
                meta_train[rec["name"]] = {"layout": rec["layout"], "max_step": rec.get("max_step_height", 0.0)}
    print(f"terrain clips: train {len(train_t)}, test {len(test_t)}, skipped {skipped}")

    if args.terrain_max_clips and len(train_t) > args.terrain_max_clips:
        layouts = sorted(set(v["layout"] for v in meta_train.values()))
        per = args.terrain_max_clips // len(layouts)
        keep = {}
        for lay in layouts:
            names = sorted(n for n, m in meta_train.items() if m["layout"] == lay)
            for n in random.Random(args.seed + lay).sample(names, min(per, len(names))):
                keep[n] = train_t[n]
        train_t = keep
        meta_train = {n: meta_train[n] for n in train_t}
        print(f"terrain training clips reduced to {len(train_t)} ({per} per layout)")

    def dump(name, d):
        with open(os.path.join(args.out_dir, name), "w") as f:
            yaml.safe_dump(d, f, sort_keys=False)

    dump("train_all.yaml", {**flat, **train_t})
    json.dump(meta_train, open(os.path.join(args.out_dir, "clip_meta.json"), "w"))
    if test_t:
        dump("eval_terrain_test.yaml", test_t)
        json.dump(meta_test, open(os.path.join(args.out_dir, "clip_meta_test.json"), "w"))
    names = sorted(train_t)
    random.Random(0).shuffle(names)
    sample = {n: train_t[n] for n in sorted(names[: args.train_sample])}
    dump("eval_terrain_train_sample.yaml", sample)
    json.dump({n: meta_train[n] for n in sample}, open(os.path.join(args.out_dir, "clip_meta_train_sample.json"), "w"))
    print("wrote", os.listdir(args.out_dir))


if __name__ == "__main__":
    main()
