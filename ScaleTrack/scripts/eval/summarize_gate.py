"""Summarise / compare gate results written by eval_modes.py (through run_gate.sh).

  python summarize_gate.py table   <eval_dir> [--level quick] [--seed 0]
  python summarize_gate.py compare <ref_eval_dir> <new_eval_dir> [--level quick] [--seed 0]

`compare` pairs the two runs clip by clip (same clip set, same seed, same env-slot mapping) and reports, per test set and
(mode, tracking) configuration: success rate of both, the paired difference with an exact McNemar test, and the relative
change of the four tracking errors with a paired-bootstrap 95% CI. A configuration is flagged REGRESSION when the
success rate drops significantly by more than 0.3 pp (or by more than 1 pp at all), or an error grows by more than 3% with
a CI that excludes zero.
"""

import argparse
import glob
import json
import math
import os
import re

import numpy as np

ERR = [("G-MPKPE", "g_pos"), ("G-MPKRE", "g_rot"), ("L-MPKPE", "l_pos"), ("L-MPKRE", "l_rot")]
SETS = ["bones", "ours", "terrain"]


def load_dir(d, level, seed):
    """-> {set: {"results": {cfg: {...}}, "clips": {cfg: {key: array}}, "names": array}}"""
    out = {}
    for s in SETS:
        res, clips, names = {}, {}, None
        for jf in sorted(glob.glob(os.path.join(d, f"{s}_{level}_s{seed}_*.json"))):
            data = json.load(open(jf))
            res.update(data["results"])
            nf = jf[:-5] + ".npz"
            if os.path.exists(nf):
                z = np.load(nf, allow_pickle=True)
                if names is None:
                    names = z["names"]
                for cfg in data["results"]:
                    clips[cfg] = {k: z[f"{cfg}/{k}"] for k in ["succ", "g_pos", "g_rot", "l_pos", "l_rot", "max_g", "max_l"] if f"{cfg}/{k}" in z}
        if res:
            out[s] = {"results": res, "clips": clips, "names": names}
    return out


def cfg_sort_key(c):
    m = re.match(r"mode(\d+)_(\w+)", c)
    return (int(m.group(1)), m.group(2))


def table(d, level, seed):
    data = load_dir(d, level, seed)
    for s, v in data.items():
        print(f"\n### {s} ({level}, seed {seed}) {d}")
        print(f"{'config':16s} {'mode name':22s} {'clips':>6s} {'Succ':>7s} {'G-MPKPE':>8s} {'G-MPKRE':>8s} {'L-MPKPE':>8s} {'L-MPKRE':>8s}")
        for cfg in sorted(v["results"], key=cfg_sort_key):
            r = v["results"][cfg]
            print(f"{cfg:16s} {r['mode_name']:22s} {r['n_clips']:6d} {r['success_rate']:7.4f} "
                  + " ".join(f"{r[k]['mean_all']:8.4f}" for k, _ in ERR))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * p)


def compare(ref_dir, new_dir, level, seed, n_boot=2000, ref_seed=None, new_seed=None):
    ref = load_dir(ref_dir, level, seed if ref_seed is None else ref_seed)
    new = load_dir(new_dir, level, seed if new_seed is None else new_seed)
    rng = np.random.RandomState(0)
    flagged = []
    agg = {}  # (set, "global"/"local") -> lists of d_succ and of the relative error changes, for the one-line aggregate below
    for s in SETS:
        if s not in ref or s not in new:
            continue
        print(f"\n### {s} ({level}, seed {seed})   ref={os.path.basename(ref_dir)}  new={os.path.basename(new_dir)}")
        print(f"{'config':14s} {'Succ ref':>8s} {'Succ new':>8s} {'d(pp)':>7s} {'p(McN)':>7s} | "
              + " | ".join(f"{k} ref->new (rel, 95% CI)" for k, _ in ERR) + " | flag")
        if ref[s]["names"] is not None and new[s]["names"] is not None:
            assert list(ref[s]["names"]) == list(new[s]["names"]), "clip sets differ between the two runs"
        for cfg in sorted(set(ref[s]["clips"]) & set(new[s]["clips"]), key=cfg_sort_key):
            a, b = ref[s]["clips"][cfg], new[s]["clips"][cfg]
            sa, sb = a["succ"].astype(bool), b["succ"].astype(bool)
            d_succ = sb.mean() - sa.mean()
            p = mcnemar_exact(int((sa & ~sb).sum()), int((~sa & sb).sum()))
            flag = []
            # tolerances calibrated on the base model re-run with another seed (paired by clip, 1000 clips):
            # global configs move by up to ~0.5 pp, local configs (they depend on drift) by up to ~2 pp
            is_local = cfg.endswith("_local")
            tol, hard = (0.010, 0.030) if is_local else (0.003, 0.010)
            if (d_succ < -tol and p < 0.05) or d_succ < -hard:
                flag.append("SUCC")
            cols = []
            both = sa & sb  # errors are compared on clips both runs completed (failed clips have inflated errors)
            for k, key in ERR:
                x, y = a[key][both].astype(np.float64), b[key][both].astype(np.float64)
                base = x.mean()
                rel = y.mean() / base - 1.0
                idx = rng.randint(0, len(x), size=(n_boot, len(x)))
                boots = (y[idx].mean(1) / x[idx].mean(1)) - 1.0
                lo, hi = np.percentile(boots, [2.5, 97.5])
                if rel > 0.03 and lo > 0:
                    flag.append(k)
                cols.append(f"{base:.4f}->{y.mean():.4f} ({rel*100:+5.1f}%, [{lo*100:+5.1f},{hi*100:+5.1f}])")
            a_ = agg.setdefault((s, "local" if is_local else "global"), {"d": [], "g": [], "l": []})
            a_["d"].append(d_succ * 100)
            a_["g"].append((b["g_pos"][both].astype(np.float64).mean() / a["g_pos"][both].astype(np.float64).mean() - 1.0) * 100)
            a_["l"].append((b["l_pos"][both].astype(np.float64).mean() / a["l_pos"][both].astype(np.float64).mean() - 1.0) * 100)
            tag = "REGRESSION " + ",".join(flag) if flag else "ok"
            if flag:
                flagged.append((s, cfg, flag))
            print(f"{cfg:14s} {sa.mean():8.4f} {sb.mean():8.4f} {d_succ*100:+7.2f} {p:7.3f} | " + " | ".join(cols) + f" | {tag}")
    print("\nAGGREGATE over configs (mean d(Succ) in pp | mean rel. change of G-MPKPE / L-MPKPE in %, clips both runs completed):")
    for (sname, kind), v in sorted(agg.items()):
        print(f"  {sname:8s} {kind:6s} n_cfg={len(v['d'])}  dSucc {np.mean(v['d']):+6.2f} pp (min {np.min(v['d']):+.2f})  G-MPKPE {np.mean(v['g']):+5.2f}%  L-MPKPE {np.mean(v['l']):+5.2f}%")
    print("\nSUMMARY:", "no regression flagged" if not flagged else f"{len(flagged)} flagged: {flagged}")
    return flagged


def noise_table(d, level, seeds):
    """Spread of the base model across evaluation seeds (only the simulator randomization changes)."""
    runs = {sd: load_dir(d, level, sd) for sd in seeds}
    for s in SETS:
        cfgs = sorted(set.intersection(*[set(runs[sd][s]["results"]) for sd in seeds if s in runs[sd]]), key=cfg_sort_key) if all(s in runs[sd] for sd in seeds) else []
        if not cfgs:
            continue
        print(f"\n### noise floor {s} ({level}), seeds {seeds}: success per seed and spread; error spreads use the mean over SUCCESSFUL clips")
        print(f"{'config':14s} {'Succ per seed':26s} {'spread(pp)':>10s} | {'G-MPKPE spread %':>16s} {'L-MPKPE spread %':>16s}")
        for cfg in cfgs:
            succ = [runs[sd][s]["results"][cfg]["success_rate"] for sd in seeds]
            g = [runs[sd][s]["results"][cfg]["G-MPKPE"]["mean_success"] for sd in seeds]
            l = [runs[sd][s]["results"][cfg]["L-MPKPE"]["mean_success"] for sd in seeds]
            print(f"{cfg:14s} {' '.join(f'{x:.4f}' for x in succ):26s} {100*(max(succ)-min(succ)):10.2f} | "
                  f"{100*(max(g)-min(g))/np.mean(g):16.2f} {100*(max(l)-min(l))/np.mean(l):16.2f}")


# ----------------------------------------------------------------------------- terrain breakdown
STEP_BINS = [("flat/rough/slope", -1.0, 0.001), ("0.001-0.10", 0.001, 0.10), ("0.10-0.15", 0.10, 0.15),
             ("0.15-0.20", 0.15, 0.20), ("0.20-0.25", 0.20, 0.25), ("0.25-0.31", 0.25, 0.31)]


def terrain_breakdown(dirs, level="quick", seed=0, index="/home/vcj9002/scalebfm_ws/motions/terrain_raw/index.jsonl"):
    """Success rate and mean errors of the held-out terrain gate per maximum step height on the clip's path."""
    meta = {}
    with open(index) as f:
        for line in f:
            r = json.loads(line)
            meta[r["name"]] = r
    for d in dirs:
        data = load_dir(d, level, seed).get("terrain")
        if not data:
            print(f"no terrain results in {d}")
            continue
        names = [str(n) for n in data["names"]]
        msh = np.array([meta[n]["max_step_height"] for n in names])
        print(f"\n### terrain breakdown {os.path.basename(d)} ({level}, seed {seed}), {len(names)} clips")
        for cfg in sorted(data["clips"], key=cfg_sort_key):
            c = data["clips"][cfg]
            sl = f"  SuccL {(c['max_l'] <= 0.5).mean():.3f}" if "max_l" in c else ""
            print(f"  {cfg}: overall Succ {c['succ'].mean():.3f}{sl}   G-MPKPE {c['g_pos'].mean():.3f}   L-MPKPE {c['l_pos'].mean():.3f}")
            for name, lo, hi in STEP_BINS:
                sel = (msh >= lo) & (msh < hi)
                if sel.sum() == 0:
                    continue
                print(f"      max step {name:18s} n={int(sel.sum()):4d}  Succ {c['succ'][sel].mean():.3f}  "
                      f"G-MPKPE {c['g_pos'][sel].mean():.3f}  L-MPKPE {c['l_pos'][sel].mean():.3f}")


def terrain_paired(ref_dir, new_dir, level="quick", ref_seed=0, new_seed=0, index="/home/vcj9002/scalebfm_ws/motions/terrain_raw/index.jsonl"):
    """Paired (same clips) comparison of two terrain-gate runs per maximum step height: success rates, difference in pp and exact
    McNemar p-value, plus the mean global position error over the clips both runs completed."""
    meta = {}
    with open(index) as f:
        for line in f:
            r = json.loads(line)
            meta[r["name"]] = r
    ref = load_dir(ref_dir, level, ref_seed).get("terrain")
    new = load_dir(new_dir, level, new_seed).get("terrain")
    if not ref or not new:
        print("terrain results missing in one of the runs")
        return
    assert list(ref["names"]) == list(new["names"]), "clip sets differ between the two runs"
    msh = np.array([meta[str(n)]["max_step_height"] for n in ref["names"]])
    print(f"\n### terrain paired comparison ref={os.path.basename(ref_dir)} (seed {ref_seed})  new={os.path.basename(new_dir)} (seed {new_seed}), {len(msh)} clips")
    for cfg in sorted(set(ref["clips"]) & set(new["clips"]), key=cfg_sort_key):
        a, b = ref["clips"][cfg], new["clips"][cfg]
        sa, sb = a["succ"].astype(bool), b["succ"].astype(bool)
        la = (a["max_l"] <= 0.5) if "max_l" in a else None
        lb = (b["max_l"] <= 0.5) if "max_l" in b else None
        lsucc = f"  [SuccL {la.mean():.3f} -> {lb.mean():.3f} ({(lb.mean()-la.mean())*100:+.1f} pp)]" if (la is not None and lb is not None) else ""
        print(f"  {cfg}: overall Succ {sa.mean():.3f} -> {sb.mean():.3f}  ({(sb.mean()-sa.mean())*100:+.2f} pp, p={mcnemar_exact(int((sa & ~sb).sum()), int((~sa & sb).sum())):.3f}){lsucc}"
              f"   G-MPKPE {a['g_pos'].mean():.3f} -> {b['g_pos'].mean():.3f}")
        for name, lo, hi in STEP_BINS:
            sel = (msh >= lo) & (msh < hi)
            if sel.sum() == 0:
                continue
            x, y = sa[sel], sb[sel]
            p = mcnemar_exact(int((x & ~y).sum()), int((~x & y).sum()))
            lbin = f"  SuccL {la[sel].mean():.3f} -> {lb[sel].mean():.3f}" if (la is not None and lb is not None) else ""
            print(f"      max step {name:18s} n={int(sel.sum()):4d}  Succ {x.mean():.3f} -> {y.mean():.3f}  ({(y.mean()-x.mean())*100:+6.2f} pp, p={p:.3f}){lbin}"
                  f"  G-MPKPE {a['g_pos'][sel].mean():.3f} -> {b['g_pos'][sel].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("dir")
    nz = sub.add_parser("noise")
    nz.add_argument("dir")
    nz.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    nz.add_argument("--level", default="quick")
    b = sub.add_parser("terrain")
    b.add_argument("dirs", nargs="+")
    b.add_argument("--level", default="quick")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--paired", action="store_true", help="two dirs (ref new): paired comparison per step-height bin")
    b.add_argument("--ref_seed", type=int, default=None)
    b.add_argument("--new_seed", type=int, default=None)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("new")
    c.add_argument("--ref_seed", type=int, default=None)
    c.add_argument("--new_seed", type=int, default=None)
    for p in (t, c):
        p.add_argument("--level", default="quick")
        p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "noise":
        noise_table(a.dir, a.level, a.seeds)
    elif a.cmd == "terrain" and a.paired:
        assert len(a.dirs) == 2, "--paired needs exactly two directories (ref new)"
        terrain_paired(a.dirs[0], a.dirs[1], a.level, a.seed if a.ref_seed is None else a.ref_seed, a.seed if a.new_seed is None else a.new_seed)
    elif a.cmd == "terrain":
        terrain_breakdown(a.dirs, a.level, a.seed)
        if len(a.dirs) == 2:  # ref, new: also the paired comparison (its "overall Succ" lines are what the gate watcher reports)
            terrain_paired(a.dirs[0], a.dirs[1], a.level, a.seed if a.ref_seed is None else a.ref_seed, a.seed if a.new_seed is None else a.new_seed)
    elif a.cmd == "table":
        table(a.dir, a.level, a.seed)
    else:
        compare(a.ref, a.new, a.level, a.seed, ref_seed=a.ref_seed, new_seed=a.new_seed)


