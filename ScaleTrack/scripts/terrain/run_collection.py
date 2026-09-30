#!/usr/bin/env python3
"""Keep GPU worker slots busy with a queue of terrain layouts (one record_magicloco_refs.py process per layout).

* one process per layout seed, CUDA_VISIBLE_DEVICES=<gpu>, own log file;
* before each launch checks nvidia-smi free memory on that GPU (--min_free_mb);
* resumable: record_magicloco_refs.py skips finished rollouts; a failed layout is retried (--retries);
* a per-layout lock (scratch/locks) prevents two drivers from running the same layout;
* at the end merges <out>/index_parts/*.jsonl into <out>/index.jsonl and writes dataset statistics.

    nohup python run_collection.py --gpus 0 --slots_per_gpu 2 > driver.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/vcj9002/magicloco/MagicLoco/.venv/bin/python"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[driver]", *a, flush=True)


def gpu_free_mb(gpu: int) -> int:
    out = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.total,memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    tot, used = [int(x) for x in out.strip().split(",")]
    return tot - used


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--slots_per_gpu", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--heldout_seeds", type=int, nargs="+", default=[6, 7])
    ap.add_argument("--num_rollouts", type=int, default=3)
    ap.add_argument("--heldout_rollouts", type=int, default=1)
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--out", default="/home/vcj9002/scalebfm_ws/motions/terrain_raw")
    ap.add_argument("--scratch", default="/home/vcj9002/scalebfm_ws/tmp/terrain_rec/full")
    ap.add_argument("--min_free_mb", type=int, default=20000)
    ap.add_argument("--launch_gap_s", type=float, default=90.0,
                    help="min seconds between two launches on the same GPU (lets memory settle before the next check)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--poll_s", type=float, default=15.0)
    ap.add_argument("--extra", type=str, default="", help="extra args passed to record_magicloco_refs.py")
    a = ap.parse_args()

    log_dir = os.path.join(a.scratch, "logs")
    lock_dir = os.path.join(a.scratch, "locks")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(lock_dir, exist_ok=True)
    queue = list(a.seeds)
    tries = {s: 0 for s in queue}
    slots = [(g, k) for g in a.gpus for k in range(a.slots_per_gpu)]
    running: dict = {}                      # slot -> (seed, Popen, logfile handle, t_launch)
    last_launch = {g: 0.0 for g in a.gpus}
    failed = []
    log(f"queue={queue} slots={slots} out={a.out}")
    while queue or running:
        # reap
        for slot, (seed, p, fh, t0) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            fh.close()
            try:
                os.remove(os.path.join(lock_dir, f"L{seed}.lock"))
            except FileNotFoundError:
                pass
            del running[slot]
            log(f"layout {seed} on gpu {slot[0]} finished rc={rc} after {time.time() - t0:.0f}s")
            if rc != 0:
                tries[seed] += 1
                if tries[seed] <= a.retries:
                    log(f"layout {seed}: retry {tries[seed]}/{a.retries} (resume skips finished rollouts)")
                    queue.insert(0, seed)
                else:
                    failed.append(seed)
        # launch
        for slot in slots:
            if slot in running or not queue:
                continue
            g = slot[0]
            if time.time() - last_launch[g] < a.launch_gap_s:
                continue
            free = gpu_free_mb(g)
            if free < a.min_free_mb:
                continue
            seed = queue[0]
            lock = os.path.join(lock_dir, f"L{seed}.lock")
            if os.path.exists(lock):
                try:
                    pid = int(open(lock).read().strip() or 0)
                except ValueError:
                    pid = 0
                if pid and pid_alive(pid):
                    log(f"layout {seed} locked by live pid {pid}; skipping it in this driver")
                    queue.pop(0)
                    continue
            queue.pop(0)
            cmd = [PY, "-u", os.path.join(HERE, "record_magicloco_refs.py"), "--layout_seeds", str(seed),
                   "--heldout_seeds", *[str(s) for s in a.heldout_seeds], "--num_rollouts", str(a.num_rollouts),
                   "--heldout_rollouts", str(a.heldout_rollouts), "--num_envs", str(a.num_envs),
                   "--out", a.out, "--scratch", a.scratch, "--headless", *a.extra.split()]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMNI_KIT_ACCEPT_EULA="YES")
            fh = open(os.path.join(log_dir, f"L{seed}_gpu{g}_try{tries[seed]}.log"), "w")
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            with open(lock, "w") as lf:
                lf.write(str(p.pid))
            running[slot] = (seed, p, fh, time.time())
            last_launch[g] = time.time()
            log(f"launched layout {seed} on gpu {g} (slot {slot[1]}, free {free} MB) pid {p.pid}")
        time.sleep(a.poll_s)
    log(f"all layouts processed; failed={failed}")
    subprocess.call([PY, os.path.join(HERE, "write_clips.py"), "--merge_only", "--raw", "-", "--layout_dir", "-",
                     "--out", a.out])
    subprocess.call([PY, os.path.join(HERE, "dataset_stats.py"), "--out", a.out])
    log("done")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
