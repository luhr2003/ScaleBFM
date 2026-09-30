#!/usr/bin/env python3
"""Record MagicLoco pi_L v4 terrain rollouts as ScaleTrack reference clips, paired with the EXACT terrain.

One layout (terrain seed) per process. For a layout seed S this script
  1. builds the IsaacLab terrain deterministically (generator seed S; numpy/torch/cuda re-seeded with S
     right before generation; env seed S) and PROVES determinism (two standalone builds + the env build
     must hash-identically), exports it (terrain.npz / terrain.obj / heightmap.npz / meta.json) in the
     simulator world frame (Z up, metres; the frame root_pos_w is expressed in);
  2. runs the Play task of pi_L v4 (model_champion_tc2_it750) with every robustness DR / push / noise /
     delay / curriculum disabled, spawns env i on a stratified tile (all 200 tiles ~equally used) at the
     tile origin with a random yaw, and drives it with a piecewise-constant command schedule that the
     env can NOT resample (the command term is patched; verified against the policy observation);
  3. records the state BEFORE every env.step (pelvis pos/quat in world, 29 body DoF in ScaleTrack
     order, command in force), keeps the buffers on the GPU, does one host transfer per rollout, and
     hands the raw rollout to ``write_clips.py`` (background process pool -> joblib pkls + index).

Run from anywhere with the MagicLoco venv (the script chdirs to the MagicLoco repo root):

    CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES /home/vcj9002/magicloco/MagicLoco/.venv/bin/python \
        record_magicloco_refs.py --layout_seeds 0 --num_envs 4096 --headless
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ML_ROOT_DEFAULT = "/home/vcj9002/magicloco/MagicLoco"
CKPT_DEFAULT = "checkpoints/G1/final/terrain/v4_terrain_rpy/model_champion_tc2_it750.pt"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--layout_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
parser.add_argument("--heldout_seeds", type=int, nargs="*", default=[6, 7])
parser.add_argument("--num_rollouts", type=int, default=3, help="rollouts per TRAINING layout")
parser.add_argument("--heldout_rollouts", type=int, default=1, help="rollouts per held-out layout")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seconds", type=float, default=20.0, help="rollout length (s); env episode stays 20 s")
parser.add_argument("--out", type=str, default="/home/vcj9002/scalebfm_ws/motions/terrain_raw")
parser.add_argument("--scratch", type=str, default="/home/vcj9002/scalebfm_ws/tmp/terrain_rec")
parser.add_argument("--task", type=str, default="Magicloco-HomieV4Rough-G1-Rpy-Play-v0")
parser.add_argument("--magicloco_root", type=str, default=ML_ROOT_DEFAULT)
parser.add_argument("--arm_train_frac", type=float, default=0.5)
parser.add_argument("--robot_friction", type=float, default=1.0,
                    help="fixed robot static=dynamic friction (training DR range 0.1-3.0; terrain 1.0, multiply)")
parser.add_argument("--workers", type=int, default=12, help="pkl writer processes per rollout")
parser.add_argument("--joblib_path", type=str, default="/home/vcj9002/scalebfm_ws/tmp/pylibs_joblib")
parser.add_argument("--heightmap_res", type=float, default=0.05)
parser.add_argument("--settle_steps", type=int, default=50, help="standing steps of the physical check phase")
parser.add_argument("--skip_determinism_check", action="store_true")
parser.add_argument("--no_obj", action="store_true", help="skip terrain.obj export")
parser.add_argument("--delete_raw", action="store_true", help="writer deletes the raw rollout npz after use")
parser.add_argument("--smoke", action="store_true",
                    help="smoke mode: 1 rollout, extra D1 instrumentation (8 early time-outs, 8 pushed envs)")
parser.add_argument("--rollout_seed_base", type=int, default=100000)

from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, os.path.join(ML_ROOT_DEFAULT, "src"))
import magicloco.core.cli_args as cli_args  # noqa: E402

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown = parser.parse_known_args()

# ---------------------------------------------------------------- one layout per process (driver mode)
if len(args_cli.layout_seeds) > 1:
    argv = sys.argv[1:]
    base, skip = [], False
    for tok in argv:
        if tok == "--layout_seeds":
            skip = True
            continue
        if skip and (tok.lstrip("-").isdigit() and not tok.startswith("--")):
            continue
        skip = False
        base.append(tok)
    for s in args_cli.layout_seeds:
        cmd = [sys.executable, os.path.abspath(__file__), *base, "--layout_seeds", str(s)]
        print(f"[driver] layout {s}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[driver] layout {s} failed rc={rc}", flush=True)
            sys.exit(rc)
    sys.exit(0)

if args_cli.checkpoint is None:
    args_cli.checkpoint = CKPT_DEFAULT
args_cli.headless = True
os.chdir(args_cli.magicloco_root)
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.managers import EventTermCfg  # noqa: E402
from isaaclab.terrains.terrain_generator import TerrainGenerator  # noqa: E402
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul  # noqa: E402
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import magicloco.tasks  # noqa: F401, E402
from magicloco.core.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from magicloco.pi_L.v1.runtime import homie_constants as HC  # noqa: E402

sys.path.insert(0, HERE)
import terrain_clip_utils as U  # noqa: E402

assert U.G1_29DOF_JOINT_NAMES == list(HC.WBC_BODY_29_NAMES), "ScaleTrack order != MagicLoco WBC_BODY_29_NAMES"

LOG_PREFIX = "[rec]"


def log(*a):
    print(LOG_PREFIX, *a, flush=True)


# ================================================================================ terrain generator
class CapturingTerrainGenerator(TerrainGenerator):
    """TerrainGenerator that (a) re-seeds every RNG the sub-terrain functions use right before it
    generates (numpy global: random_rough; torch CUDA default generator: boxes; the generator's own
    np_rng comes from cfg.seed), and (b) keeps itself + the per-tile (type, difficulty) record so the
    exact mesh the TerrainImporter imports can be exported (the importer discards the generator)."""

    instances: list = []

    def __init__(self, cfg, device: str = "cpu"):
        if cfg.seed is None:
            raise ValueError("CapturingTerrainGenerator needs an explicit cfg.seed")
        s = int(cfg.seed)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        self.tile_log: dict = {}
        self._last_difficulty = None
        super().__init__(cfg, device)
        CapturingTerrainGenerator.instances.append(self)

    def _get_terrain_mesh(self, difficulty, cfg):
        self._last_difficulty = float(difficulty)
        return super()._get_terrain_mesh(difficulty, cfg)

    def _add_sub_terrain(self, mesh, origin, row, col, sub_terrain_cfg):
        names = list(self.cfg.sub_terrains.keys())
        vals = list(self.cfg.sub_terrains.values())
        k = [i for i, v in enumerate(vals) if v is sub_terrain_cfg]
        assert len(k) == 1
        self.tile_log[(int(row), int(col))] = (names[k[0]], self._last_difficulty, np.array(origin, dtype=np.float64))
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)


def mesh_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
    return h.hexdigest()


# ================================================================================ recorder state
class RecState:
    def __init__(self):
        self.ready = False


REC = RecState()


def ev_reset_root_to_tile(env, env_ids):
    """RESET event (replaces reset_root_state_uniform): tile origin + default root offset, yaw ~ U(-pi, pi)
    from the recorder's private generator, zero velocity."""
    robot = env.scene["robot"]
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    root = robot.data.default_root_state[ids].clone()
    pos = root[:, :3] + env.scene.env_origins[ids]
    if REC.ready:
        yaw = (torch.rand(len(ids), generator=REC.yaw_gen, device=env.device) * 2.0 - 1.0) * math.pi
    else:
        yaw = torch.zeros(len(ids), device=env.device)
    z = torch.zeros_like(yaw)
    quat = quat_mul(root[:, 3:7], quat_from_euler_xyz(z, z, yaw))
    robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
    robot.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
    if REC.ready:
        REC.spawn_pos[ids] = pos
        REC.spawn_quat[ids] = quat


def ev_reset_joints_default(env, env_ids):
    """RESET event (replaces reset_joints_scale_and_offset): exact default pose (clamped to the soft
    limits exactly like the training term), zero joint velocity."""
    robot = env.scene["robot"]
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    q = robot.data.default_joint_pos[ids].clone()
    lim = robot.data.soft_joint_pos_limits[ids]
    q = torch.maximum(torch.minimum(q, lim[..., 1]), lim[..., 0])
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)


# ================================================================================ command schedule
class CommandSchedule:
    """Piecewise-constant per-env commands; segment length U(2, 6) s. Mask-based draws (no host sync)."""

    # segment mix: 15% stand, 5% backward, 21% fast forward vx U(0.5, 0.9) (~25% of the moving segments),
    # 59% normal forward (70% U(0.25, 0.5), 30% U(0.15, 0.6)); all inside the trained box (vx -0.6..1.0)
    P_STAND, P_BACK, P_FAST = 0.15, 0.05, 0.21

    def __init__(self, n: int, device: str, dt: float, ranges: dict, seed: int, default_height: float):
        self.n, self.dev, self.dt = n, device, dt
        self.ranges = ranges
        self.default_height = default_height
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed))
        self.cmd = torch.zeros(n, 7, device=device)
        self.cmd[:, 3] = default_height
        self.left = torch.zeros(n, dtype=torch.long, device=device)
        self.forced = None
        self.in_reset = False
        lo = torch.tensor([ranges[k][0] for k in ("vx", "vy", "wz", "height", "roll", "pitch", "yaw")], device=device)
        hi = torch.tensor([ranges[k][1] for k in ("vx", "vy", "wz", "height", "roll", "pitch", "yaw")], device=device)
        self.lo, self.hi = lo, hi

    def reseed(self, seed: int):
        self.gen.manual_seed(int(seed))

    def draw(self, mask: torch.Tensor):
        n, dev = self.n, self.dev
        r = torch.rand(n, 16, generator=self.gen, device=dev)
        stand = r[:, 0] < self.P_STAND
        back = (r[:, 0] >= self.P_STAND) & (r[:, 0] < self.P_STAND + self.P_BACK)
        fast = r[:, 0] >= 1.0 - self.P_FAST
        vx_f = torch.where(r[:, 1] < 0.7, 0.25 + 0.25 * r[:, 2], 0.15 + 0.45 * r[:, 2])   # bias 0.25-0.5
        vx_f = torch.where(fast, 0.5 + 0.4 * r[:, 14], vx_f)                                 # U(0.5, 0.9)
        vx_b = -0.15 - 0.25 * r[:, 2]                                                      # U(-0.4, -0.15)
        zero = torch.zeros_like(vx_f)
        vx = torch.where(stand, zero, torch.where(back, vx_b, vx_f))
        vy = torch.where((r[:, 3] < 0.5) | stand, zero, (2.0 * r[:, 4] - 1.0) * 0.3)
        wz = torch.where((r[:, 5] < 0.6) | stand, zero, (2.0 * r[:, 6] - 1.0) * 0.3)
        h = torch.where(r[:, 7] < 0.65, torch.full_like(vx, self.default_height), 0.55 + 0.19 * r[:, 8])
        tilt = (r[:, 9] < 0.25).unsqueeze(1)
        rpy = torch.where(tilt, (2.0 * r[:, 10:13] - 1.0) * 0.2, torch.zeros(n, 3, device=dev))
        new = torch.cat([vx[:, None], vy[:, None], wz[:, None], h[:, None], rpy], dim=1)
        new = torch.maximum(torch.minimum(new, self.hi), self.lo)       # never outside the trained box
        dur = torch.round((2.0 + 4.0 * r[:, 13]) / self.dt).long()
        m = mask.bool()
        self.cmd = torch.where(m[:, None], new, self.cmd)
        self.left = torch.where(m, dur, self.left)

    def tick(self, active: torch.Tensor):
        """Called once per env.step (inside the command manager, after resets, before the observation)."""
        self.left = torch.where(active, self.left - 1, self.left)
        self.draw(active & (self.left <= 0))

    def value(self) -> torch.Tensor:
        if self.forced is not None:
            return self.forced.expand(self.n, 7)
        return self.cmd


# ================================================================================ helpers
def trained_command_ranges(bc) -> dict:
    return {"vx": tuple(bc.lin_vel_x_range), "vy": tuple(bc.lin_vel_y_range), "wz": tuple(bc.ang_vel_z_range),
            "height": tuple(bc.height_range), "roll": tuple(bc.roll_range), "pitch": tuple(bc.pitch_range),
            "yaw": tuple(bc.yaw_range)}


def cfg_to_jsonable(x, depth=0):
    if depth > 6:
        return str(x)
    if hasattr(x, "to_dict"):
        try:
            x = x.to_dict()
        except Exception:
            return str(x)
    if isinstance(x, dict):
        return {str(k): cfg_to_jsonable(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [cfg_to_jsonable(v, depth + 1) for v in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def raycast_down(wmesh, xy: torch.Tensor, z_start: torch.Tensor | float) -> torch.Tensor:
    """z of the first mesh hit straight below (x, y, z_start); +inf/nan on miss."""
    n = xy.shape[0]
    starts = torch.zeros(n, 3, device=xy.device)
    starts[:, :2] = xy
    starts[:, 2] = z_start
    dirs = torch.zeros(n, 3, device=xy.device)
    dirs[:, 2] = -1.0
    hits = raycast_mesh(starts.unsqueeze(0), dirs.unsqueeze(0), wmesh, max_dist=1e4)[0][0]
    return hits[:, 2]


def stats(x: torch.Tensor | np.ndarray) -> dict:
    x = np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": round(float(x.mean()), 6), "std": round(float(x.std()), 6),
            "min": round(float(x.min()), 6), "p5": round(float(np.percentile(x, 5)), 6),
            "p50": round(float(np.percentile(x, 50)), 6), "p95": round(float(np.percentile(x, 95)), 6),
            "max": round(float(x.max()), 6)}


# ================================================================================ main
def main():
    t_start = time.time()
    seed = int(args_cli.layout_seeds[0])
    split = "test" if seed in args_cli.heldout_seeds else "train"
    n_roll = 1 if args_cli.smoke else (args_cli.heldout_rollouts if split == "test" else args_cli.num_rollouts)
    out = os.path.abspath(args_cli.out)
    layout_dir = os.path.join(out, "layouts", f"layout_{seed}")
    raw_dir = os.path.join(os.path.abspath(args_cli.scratch), "raw")
    log_dir = os.path.join(os.path.abspath(args_cli.scratch), "logs")
    for d in (layout_dir, raw_dir, log_dir, os.path.join(out, "rollout_stats")):
        os.makedirs(d, exist_ok=True)
    dev = "cuda:0"

    def rollout_done(r_):
        return (os.path.exists(os.path.join(out, "index_parts", f"L{seed}_r{r_}.jsonl"))
                and os.path.exists(os.path.join(out, "rollout_stats", f"L{seed}_r{r_}.json")))

    def raw_file(r_):
        return os.path.join(raw_dir, f"L{seed}_r{r_}.npz")

    def launch_writer(r_, writers_):
        wcmd = [sys.executable, os.path.join(HERE, "write_clips.py"), "--raw", raw_file(r_), "--layout_dir", layout_dir,
                "--out", out, "--workers", str(args_cli.workers), "--joblib_path", args_cli.joblib_path]
        if args_cli.delete_raw:
            wcmd.append("--delete_raw")
        wlog = open(os.path.join(log_dir, f"write_L{seed}_r{r_}.log"), "w")
        writers_.append((subprocess.Popen(wcmd, stdout=wlog, stderr=subprocess.STDOUT), wlog, raw_file(r_)))
        log(f"writer launched for {raw_file(r_)}")

    def wait_writers(writers_):
        rc_ = 0
        for p_, wl_, rp_ in writers_:
            rc1 = p_.wait()
            wl_.close()
            log(f"writer for {rp_} finished rc={rc1}")
            rc_ |= rc1
        return rc_

    # atomic per-layout claim shared by every driver (lock lives in the output root; released on process exit)
    import fcntl
    os.makedirs(os.path.join(out, ".locks"), exist_ok=True)
    _lock_fh = open(os.path.join(out, ".locks", f"layout_{seed}.flock"), "w")
    try:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log(f"layout {seed} is being processed by another live process (flock held) -> exiting without work")
        return 0
    REC.lock_fh = _lock_fh
    todo = [r_ for r_ in range(n_roll) if not rollout_done(r_)]
    if not todo:
        log(f"layout {seed}: all {n_roll} rollouts already finished -> nothing to do (resume)")
        return 0
    have_layout = os.path.exists(os.path.join(layout_dir, "meta.json")) and os.path.exists(
        os.path.join(layout_dir, "heightmap.npz"))
    if have_layout and all(os.path.exists(raw_file(r_)) for r_ in todo):
        log(f"layout {seed}: rollouts {todo} simulated earlier, only (re)writing clips (resume)")
        ws = []
        for r_ in todo:
            launch_writer(r_, ws)
        return wait_writers(ws)
    log(f"layout_seed={seed} split={split} rollouts={n_roll} todo={todo} num_envs={args_cli.num_envs} out={out} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    for k in ("V3_DELAY_STEPS", "V3_NOISE_RHO", "SCAN_DR_SCALE", "G4DR_ALPHA", "G4_ARM_RHO", "SP_ARM_RHO", "SP_ARM_STATIC"):
        if os.environ.get(k) is not None:
            raise RuntimeError(f"env var {k} is set; it would override the clean caliber")

    # ------------------------------------------------------------------ env cfg
    env_cfg = parse_env_cfg(args_cli.task, device=dev, num_envs=args_cli.num_envs)
    decisions = []
    env_cfg.seed = seed
    decisions.append(f"env_cfg.seed = {seed} (configure_seed before scene/terrain creation)")
    assert abs(env_cfg.episode_length_s - 20.0) < 1e-9
    assert env_cfg.scene.filter_collisions, "robots of different envs must not collide"
    tcfg = env_cfg.scene.terrain
    tg = tcfg.terrain_generator
    tg.seed = seed
    tg.use_cache = False
    tg.curriculum = True          # rows ordered by difficulty (generator layout, NOT the level curriculum)
    tg.class_type = CapturingTerrainGenerator
    tcfg.max_init_terrain_level = None
    decisions.append("terrain generator: seed=layout seed, use_cache=False, row-difficulty layout kept; "
                     "terrain_levels curriculum removed; env tiles assigned by the recorder")
    # curriculum: every term off (terrain level promotion, precision/scan/g4 ramps, delay/noise pins)
    for name in list(vars(env_cfg.curriculum)):
        if not name.startswith("_") and getattr(env_cfg.curriculum, name) is not None:
            decisions.append(f"curriculum.{name}: removed")
            setattr(env_cfg.curriculum, name, None)
    ev = env_cfg.events
    keep = {"physics_material", "reset_base", "reset_joints"}
    for name in list(vars(ev)):
        if name.startswith("_") or getattr(ev, name) is None or name in keep:
            continue
        decisions.append(f"events.{name} ({getattr(ev, name).mode}): removed")
        setattr(ev, name, None)
    f = float(args_cli.robot_friction)
    ev.physics_material.params.update({"static_friction_range": (f, f), "dynamic_friction_range": (f, f),
                                       "restitution_range": (0.0, 0.0), "num_buckets": 1})
    decisions.append(f"events.physics_material: pinned robot friction {f}/{f}, restitution 0 (terrain 1.0, multiply)")
    ev.reset_base = EventTermCfg(func=ev_reset_root_to_tile, mode="reset", params={})
    ev.reset_joints = EventTermCfg(func=ev_reset_joints_default, mode="reset", params={})
    decisions.append("events.reset_base -> tile origin + (0,0,0.8), yaw U(-pi,pi), zero velocity")
    decisions.append("events.reset_joints -> exact default joint pose (was x U(0.8,1.2) + U(-0.1,0.1))")
    bc = env_cfg.commands.base_command
    ranges = trained_command_ranges(bc)
    bc.resampling_time_range = (1.0e9, 1.0e9)
    decisions.append("commands.base_command: resampling disabled (1e9 s); _resample_command/_update_command "
                     "patched to the recorder schedule; smooth_height/smooth_vel/heading: off in cfg")
    assert not bc.smooth_height and not bc.smooth_vel and not bc.scale_yaw_by_height and bc.turn_in_place_ratio == 0.0
    assert env_cfg.actions.lower.max_delay_steps == 0
    assert not env_cfg.observations.policy.enable_corruption

    # ------------------------------------------------------------------ determinism proof (standalone builds)
    det = {}
    if not args_cli.skip_determinism_check:
        hs = []
        for i in range(2):
            t0 = time.time()
            g = CapturingTerrainGenerator(copy.deepcopy(tg), device=dev)
            hs.append(mesh_hash(g.terrain_mesh.vertices, g.terrain_mesh.faces))
            log(f"determinism build {i}: {hs[-1][:16]} ({time.time() - t0:.1f}s, V={len(g.terrain_mesh.vertices)} "
                f"F={len(g.terrain_mesh.faces)})")
        det = {"standalone_build_a": hs[0], "standalone_build_b": hs[1]}
        CapturingTerrainGenerator.instances.clear()
        if hs[0] != hs[1]:
            raise RuntimeError("terrain generation is NOT deterministic for this seed")

    # ------------------------------------------------------------------ env
    t0 = time.time()
    env = gym.make(args_cli.task, cfg=env_cfg)
    core = env.unwrapped
    log(f"env created in {time.time() - t0:.1f}s; step_dt={core.step_dt} max_episode_length={core.max_episode_length}")
    assert abs(core.step_dt - 0.02) < 1e-9, core.step_dt
    assert core.max_episode_length == 1000
    assert len(CapturingTerrainGenerator.instances) == 1, len(CapturingTerrainGenerator.instances)
    gen = CapturingTerrainGenerator.instances[0]
    V = np.asarray(gen.terrain_mesh.vertices)
    F = np.asarray(gen.terrain_mesh.faces)
    h_env = mesh_hash(V, F)
    det["env_build"] = h_env
    if "standalone_build_a" in det:
        det["identical"] = bool(h_env == det["standalone_build_a"] == det["standalone_build_b"])
        if not det["identical"]:
            raise RuntimeError(f"env terrain hash differs from standalone builds: {det}")
    log(f"terrain determinism: {det}")

    robot = core.scene["robot"]
    terrain = core.scene.terrain
    scanner = core.scene.sensors["height_scanner"]
    N = core.num_envs

    # managers sanity (clean caliber)
    act_lower = core.action_manager.get_term("lower")
    act_upper = core.action_manager.get_term("upper")
    from magicloco.pi_L.v3.tasks import obs_noise as _obs_noise
    from magicloco.pi_L.v2.tasks import scan_dr as _scan_dr
    delay_caps = {k: getattr(a, "active_max_delay", None) for k, a in robot.actuators.items()}
    sanity = {
        "active_curriculum_terms": list(core.curriculum_manager.active_terms),
        "active_event_terms": {m: core.event_manager.active_terms.get(m, []) for m in core.event_manager.available_modes},
        "actuator_active_max_delay": delay_caps,
        "lower_action_max_delay_steps": int(act_lower._max_delay_steps),
        "lower_action_scale": float(env_cfg.actions.lower.scale),
        "obs_noise_rho": float(_obs_noise.noise_rho(core)),
        "scan_dr_scale": float(_scan_dr.scan_dr_scale(core)),
        "homie_rho_a_scalar": float(getattr(core, "_homie_rho_a", 0.0)),
        "termination_terms": list(core.termination_manager.active_terms),
        "filter_collisions": bool(env_cfg.scene.filter_collisions),
        "trained_command_ranges": ranges,
    }
    assert all(v == 0 for v in delay_caps.values() if v is not None), delay_caps
    assert sanity["obs_noise_rho"] == 0.0 and sanity["scan_dr_scale"] == 0.0
    assert len(sanity["active_curriculum_terms"]) == 0, sanity["active_curriculum_terms"]
    log("sanity:", json.dumps(sanity))

    # body / joint indices
    assert robot.body_names[0] == "pelvis", robot.body_names[0]
    idx29 = robot.find_joints(U.G1_29DOF_JOINT_NAMES, preserve_order=True)[0]
    assert [robot.joint_names[i] for i in idx29] == U.G1_29DOF_JOINT_NAMES
    idx29_t = torch.tensor(idx29, device=dev, dtype=torch.long)
    feet_ids = robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0]
    log(f"articulation: {robot.num_joints} joints, {robot.num_bodies} bodies; idx29={idx29}")

    # material check
    try:
        mat = robot.root_physx_view.get_material_properties()
        sanity["robot_material_minmax"] = [float(mat.min()), float(mat[..., :2].max()), float(mat[..., 2].max())]
        log(f"robot material (min, max friction, max restitution) = {sanity['robot_material_minmax']}")
    except Exception as e:  # pragma: no cover
        log(f"material readback failed: {e}")

    # ------------------------------------------------------------------ terrain export
    tg_cfg = gen.cfg
    names = list(tg_cfg.sub_terrains.keys())
    nrows, ncols = int(tg_cfg.num_rows), int(tg_cfg.num_cols)
    size = tuple(float(s) for s in tg_cfg.size)
    grid_origin = (-size[0] * nrows * 0.5, -size[1] * ncols * 0.5)
    origins = np.asarray(gen.terrain_origins, dtype=np.float64)
    t_orig = terrain.terrain_origins.detach().cpu().numpy().astype(np.float64)
    assert np.abs(t_orig - origins).max() < 1e-4, "importer origins != generator origins"
    tiles, type_idx = [], np.zeros((nrows, ncols), np.int64)
    step_h, box_h = np.zeros((nrows, ncols)), np.zeros((nrows, ncols))
    param_v, diff_a = np.zeros((nrows, ncols)), np.zeros((nrows, ncols))
    aliases = {
        "pyramid_stairs": "stairs pyramid, spawn on the TOP platform: walking away from the spawn goes DOWN",
        "pyramid_stairs_inv": "inverted stairs pyramid, spawn in the PIT: walking away from the spawn goes UP",
        "boxes": "random grid of boxes (heights +-h) around a raised flat platform",
        "random_rough": "random uniform rough height field (noise 0.02-0.10 m)",
        "hf_pyramid_slope": "slope pyramid, spawn on top: walking away goes DOWN the slope",
        "hf_pyramid_slope_inv": "inverted slope pyramid, spawn in the pit: walking away goes UP the slope",
    }
    for r in range(nrows):
        for c in range(ncols):
            tname, d, _o = gen.tile_log[(r, c)]
            st = tg_cfg.sub_terrains[tname]
            k = names.index(tname)
            type_idx[r, c] = k
            diff_a[r, c] = d
            info = {"row": r, "col": c, "type": tname, "difficulty": round(d, 6),
                    "origin": [round(float(v), 6) for v in origins[r, c]],
                    "center_xy": [grid_origin[0] + (r + 0.5) * size[0], grid_origin[1] + (c + 0.5) * size[1]],
                    "step_height": 0.0, "param_name": None, "param_value": None}
            if hasattr(st, "step_height_range"):
                sh = st.step_height_range[0] + d * (st.step_height_range[1] - st.step_height_range[0])
                n_steps = int(min((size[0] - 2 * st.border_width - st.platform_width) // (2 * st.step_width) + 1,
                                  (size[1] - 2 * st.border_width - st.platform_width) // (2 * st.step_width) + 1))
                # geometry cross-check: origin z = +-(num_steps + 1) * step_height
                assert abs(abs(origins[r, c, 2]) - (n_steps + 1) * sh) < 1e-6, (r, c, origins[r, c], sh, n_steps)
                info.update(step_height=round(sh, 6), param_name="step_height", param_value=round(sh, 6),
                            num_steps=n_steps, step_width=float(st.step_width))
                step_h[r, c] = sh
                param_v[r, c] = sh
            elif hasattr(st, "grid_height_range"):
                gh = st.grid_height_range[0] + d * (st.grid_height_range[1] - st.grid_height_range[0])
                info.update(param_name="box_height", param_value=round(gh, 6))
                box_h[r, c] = gh
                param_v[r, c] = gh
            elif hasattr(st, "slope_range"):
                sl = st.slope_range[0] + d * (st.slope_range[1] - st.slope_range[0])
                info.update(param_name="slope", param_value=round(sl, 6))
                param_v[r, c] = sl
            elif hasattr(st, "noise_range"):
                info.update(param_name="noise_max", param_value=float(st.noise_range[1]))
                param_v[r, c] = st.noise_range[1]
            info["type_alias"] = aliases.get(tname, tname)
            tiles.append(info)
    V32 = V.astype(np.float32)
    F32 = F.astype(np.int32)
    assert F.max() < 2 ** 31
    np.savez(os.path.join(layout_dir, "terrain.npz"), vertices=V32, faces=F32)
    exported_hash = mesh_hash(V32.astype(np.float64), F32)
    if not args_cli.no_obj:
        t0 = time.time()
        with open(os.path.join(layout_dir, "terrain.obj"), "w") as fobj:
            fobj.write(f"# MagicLoco terrain layout {seed}; world frame, Z up, metres\n")
            np.savetxt(fobj, V32, fmt="v %.6f %.6f %.6f")
            np.savetxt(fobj, F32 + 1, fmt="f %d %d %d")
        log(f"terrain.obj written ({time.time() - t0:.1f}s)")

    # USD readback: the prim the simulator/raycaster use must carry exactly these points, identity xform
    usd_check = {}
    try:
        from pxr import Usd, UsdGeom
        stage = core.sim.stage
        prim = stage.GetPrimAtPath(tcfg.prim_path + "/terrain/mesh")
        pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        fvi = np.asarray(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=np.float64)
        usd_check = {"prim": str(prim.GetPath()), "num_points": int(len(pts)),
                     "max_abs_point_diff_vs_export": float(np.abs(pts - V32.astype(np.float64)).max()),
                     "faces_identical": bool(fvi.shape == F.shape and np.array_equal(fvi, F)),
                     "local_to_world_is_identity": bool(np.allclose(xf, np.eye(4), atol=1e-9)),
                     "local_to_world": xf.tolist()}
        # foot collision geometry of g1_new.usd (diagnostic for the kinematic check)
        foot = {}
        link = stage.GetPrimAtPath("/World/envs/env_0/Robot/left_ankle_roll_link")
        xc = UsdGeom.XformCache()
        from pxr import UsdPhysics
        for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                rel = np.asarray(xc.ComputeRelativeTransform(p, link)[0], dtype=np.float64)
                attrs = {}
                for an in ("radius", "height", "axis", "size", "extent"):
                    a = p.GetAttribute(an)
                    if a and a.HasValue():
                        v = a.Get()
                        attrs[an] = str(v) if an in ("axis", "extent") else float(v)
                foot[str(p.GetPath())] = {"type": p.GetTypeName(), **attrs,
                                          "translation_in_link": rel[3, :3].tolist(), "rot_in_link": rel[:3, :3].tolist()}
        usd_check["left_foot_collision_prims"] = foot
    except Exception as e:  # pragma: no cover
        usd_check["error"] = repr(e)
    log(f"USD check: { {k: v for k, v in usd_check.items() if k not in ('local_to_world', 'left_foot_collision_prims')} }")
    log(f"left foot collision prims: {json.dumps(usd_check.get('left_foot_collision_prims', {}))}")

    # heightmap (upper envelope, ray cast straight down on the EXPORTED mesh) for labels / filters
    wmesh = convert_to_warp_mesh(V32.astype(np.float32), F32, device=dev)
    bb_min, bb_max = V32.min(0), V32.max(0)
    res = float(args_cli.heightmap_res)
    xs = np.arange(bb_min[0], bb_max[0] + 1e-6, res, dtype=np.float64)
    ys = np.arange(bb_min[1], bb_max[1] + 1e-6, res, dtype=np.float64)
    hm = np.empty((len(xs), len(ys)), np.float32)
    ys_t = torch.tensor(ys, device=dev, dtype=torch.float32)
    chunk = max(1, 4_000_000 // len(ys))
    for i0 in range(0, len(xs), chunk):
        xx = torch.tensor(xs[i0:i0 + chunk], device=dev, dtype=torch.float32)
        gx, gy = torch.meshgrid(xx, ys_t, indexing="ij")
        z = raycast_down(wmesh, torch.stack([gx.reshape(-1), gy.reshape(-1)], -1), float(bb_max[2]) + 1.0)
        hm[i0:i0 + chunk] = z.reshape(len(xx), len(ys)).cpu().numpy()
    n_miss = int((~np.isfinite(hm)).sum())
    np.savez(os.path.join(layout_dir, "heightmap.npz"), height=hm, x0=xs[0], y0=ys[0], res=res)
    log(f"heightmap {hm.shape} res={res} misses={n_miss}")

    meta = {
        "layout_seed": seed, "split": split, "task": args_cli.task, "checkpoint": os.path.abspath(args_cli.checkpoint),
        "generator_seed": int(tg_cfg.seed), "env_seed": seed,
        "rng_seeding": "np.random/torch/torch.cuda/random re-seeded with the layout seed right before generation; "
                       "generator np_rng = default_rng(seed); env_cfg.seed = layout seed",
        "frame": {"up_axis": "z", "units": "m", "handedness": "right",
                  "mesh_frame": "IsaacLab/USD world frame. The mesh prim /World/ground/terrain/mesh has identity "
                                "local-to-world transform; clip root_pos are root_link_pos_w in the SAME frame "
                                "(env origins included), so no offset is needed to place clips on the mesh.",
                  "tile_axes": "rows along +x, columns along +y; tile (r,c) center = tile_grid_origin_xy + "
                               "((r+0.5)*size_x, (c+0.5)*size_y)"},
        "grid": {"tile_size": list(size), "num_rows": nrows, "num_cols": ncols, "border_width": float(tg_cfg.border_width),
                 "border_height": float(tg_cfg.border_height), "tile_grid_origin_xy": list(grid_origin),
                 "difficulty_range": list(tg_cfg.difficulty_range),
                 "row_difficulty": "difficulty = (row + U(0,1)) / num_rows  (generator row layout)"},
        "sub_terrain_names": names, "type_aliases": aliases,
        "generator_cfg": cfg_to_jsonable(tg_cfg),
        "terrain_origins": origins.round(6).tolist(),
        "tile_type_index": type_idx.tolist(), "tile_difficulty": diff_a.round(6).tolist(),
        "tile_step_height": step_h.round(6).tolist(), "tile_box_height": box_h.round(6).tolist(),
        "tile_param_value": param_v.round(6).tolist(), "tiles": tiles,
        "mesh": {"num_vertices": int(len(V)), "num_faces": int(len(F)), "bbox_min": bb_min.tolist(),
                 "bbox_max": bb_max.tolist(), "sha256_float64_generator": h_env, "sha256_exported_float32": exported_hash,
                 "files": {"terrain.npz": "vertices float32 (V,3), faces int32 (F,3)", "terrain.obj": "same mesh, 1-based",
                           "heightmap.npz": f"upper-envelope height raster, res {res} m: height[i,j] at (x0+i*res, y0+j*res)"}},
        "determinism": det, "usd_check": usd_check, "sanity": sanity, "clean_caliber_decisions": decisions,
    }

    # ------------------------------------------------------------------ tiles for the envs (stratified)
    rng_tiles = np.random.default_rng(10_000 + seed)
    perm = rng_tiles.permutation(nrows * ncols)
    env_tile = perm[np.arange(N) % (nrows * ncols)]
    env_rank = np.arange(N) // (nrows * ncols)
    env_row, env_col = env_tile // ncols, env_tile % ncols
    terrain.terrain_levels[:] = torch.as_tensor(env_row, device=dev)
    terrain.terrain_types[:] = torch.as_tensor(env_col, device=dev)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    assert core.scene.env_origins.data_ptr() == terrain.env_origins.data_ptr()
    counts = np.bincount(env_tile, minlength=nrows * ncols)
    log(f"env->tile stratification: envs per tile min={counts.min()} max={counts.max()}")
    meta["env_tiles"] = {"rule": "env i -> tile perm[i % 200] (perm = default_rng(10000+seed).permutation(200)); "
                                 "tile index = row*num_cols + col", "envs_per_tile_min": int(counts.min()),
                         "envs_per_tile_max": int(counts.max())}

    # ------------------------------------------------------------------ recorder state + patches
    REC.yaw_gen = torch.Generator(device=dev)
    REC.yaw_gen.manual_seed(20_000 + seed)
    REC.spawn_pos = torch.zeros(N, 3, device=dev)
    REC.spawn_quat = torch.zeros(N, 4, device=dev)
    REC.ready = True
    sched = CommandSchedule(N, dev, core.step_dt, ranges, 30_000 + seed, float(bc.default_height))
    term = core.command_manager.get_term("base_command")

    def _resample_command(env_ids):
        if not sched.in_reset:
            raise RuntimeError("base_command tried to resample outside a reset")
        m = torch.zeros(N, dtype=torch.bool, device=dev)
        m[torch.as_tensor(env_ids, device=dev, dtype=torch.long)] = True
        sched.draw(m)
        term.command_b[m] = sched.value()[m]

    def _update_command():
        sched.tick(core.episode_length_buf > 0)
        term.command_b[:] = sched.value()

    term._resample_command = _resample_command
    term._update_command = _update_command
    _orig_reset_idx = core._reset_idx

    def _reset_idx_wrapped(env_ids):
        sched.in_reset = True
        try:
            _orig_reset_idx(env_ids)
        finally:
            sched.in_reset = False

    core._reset_idx = _reset_idx_wrapped
    # arms: per-row rho (training distribution rho=1 vs nominal rho=0 == eval default). The per-row path zeroes
    # rows with rho == 0 EXACTLY; 1e-6 reproduces the scalar rho_a=0 distribution (slight wiggle) bit-for-bit
    # in distribution (eps = 20*(1-0.99*1e-6)).
    core._homie_rho_a_row = torch.full((N,), 1e-6, device=dev)

    # ------------------------------------------------------------------ policy
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=dev)
    ck = torch.load(args_cli.checkpoint, map_location=dev, weights_only=False)
    sd = ck.get("model_state_dict", ck)
    own = runner.alg.policy.state_dict()
    missing = sorted(set(own) - set(sd))
    unexpected = sorted(set(sd) - set(own))
    log(f"checkpoint {args_cli.checkpoint}: missing={missing} unexpected={unexpected}")
    if missing:
        raise RuntimeError(f"checkpoint misses policy keys: {missing}")
    runner.alg.policy.load_state_dict(sd, strict=False)
    policy = runner.get_inference_policy(device=dev)

    # observation layout (decode the command the policy actually receives)
    om = core.observation_manager
    pnames = list(om.active_terms["policy"])
    pdims = [int(np.prod(d)) for d in om.group_obs_term_dim["policy"]]
    offs = dict(zip(pnames, np.concatenate([[0], np.cumsum(pdims)[:-1]]).tolist()))
    frame_dim = HC.SINGLE_OBS_DIM
    newest = offs["obs"] + pdims[pnames.index("obs")] - frame_dim
    ter_off = offs["terrain"]
    cmd_scale = torch.tensor(HC.CMD_SCALE, device=dev)
    default29 = torch.tensor(HC.DEFAULT_ANGLES_29, device=dev)
    log(f"policy obs terms {list(zip(pnames, pdims))}; newest frame at {newest}")

    def expected_obs_cmd(cmd):
        nav = cmd[:, :3] * cmd_scale
        nav = torch.where((nav.norm(dim=1, keepdim=True) < 0.1), torch.zeros_like(nav), nav)
        return torch.cat([nav, cmd[:, 3:7]], dim=1)

    # ------------------------------------------------------------------ physical check (standing at tile origins)
    phys = {}
    with torch.inference_mode():
        torch.manual_seed(seed)
        sched.forced = torch.tensor([0.0, 0.0, 0.0, float(bc.default_height), 0.0, 0.0, 0.0], device=dev)
        obs, _ = wrapped.reset()
        fell_t = torch.zeros((), device=dev)
        for _ in range(int(args_cli.settle_steps)):
            obs, _, dones, _ = wrapped.step(policy(obs))
            fell_t = fell_t + dones.sum()
        fell = int(fell_t.item())
        feet = robot.data.body_link_pos_w[:, feet_ids]                     # (N,2,3)
        z_top = float(bb_max[2]) + 1.0
        mz = raycast_down(wmesh, feet[..., :2].reshape(-1, 2), z_top).reshape(N, 2)
        foot_off = (feet[..., 2] - mz)                                      # ankle_roll_link origin above mesh
        pel = robot.data.root_link_pos_w
        pz = raycast_down(wmesh, pel[:, :2], z_top)
        hits = scanner.data.ray_hits_w                                      # (N,187,3)
        hflat = hits.reshape(-1, 3)
        ok = torch.isfinite(hflat).all(dim=1)
        mz_h = raycast_down(wmesh, hflat[ok, :2], hflat[ok, 2] + 0.5)
        dz_hits = (mz_h - hflat[ok, 2]).abs()
        # policy's scan (clean, no DR) == sensor formula
        scan_expected = (scanner.data.pos_w[:, 2:3] - hits[..., 2] - 0.5).clamp(-1.0, 1.0)
        scan_obs = obs[:, ter_off:ter_off + hits.shape[1]]
        scan_err = (scan_obs - scan_expected).abs().max().item()
        # root link == pelvis body
        pel_body_err = (robot.data.body_link_pos_w[:, 0] - pel).abs().max().item()
        types_env = type_idx[env_row, env_col]
        per_type = {}
        fo = foot_off.detach().cpu().numpy()
        ph = (pel[:, 2] - pz).detach().cpu().numpy()
        for k_, nm in enumerate(names):
            sel = types_env == k_
            if sel.any():
                per_type[nm] = {"foot_ankle_above_mesh": stats(fo[sel]), "pelvis_above_mesh": stats(ph[sel])}
        phys = {"settle_steps": int(args_cli.settle_steps), "fell_during_settle": fell,
                "foot_ankle_link_above_mesh_all": stats(fo), "per_type": per_type,
                "scan_ray_hits_vs_exported_mesh_abs_dz": stats(dz_hits), "scan_ray_misses": int((~ok).sum().item()),
                "policy_scan_obs_vs_clean_formula_max_abs": scan_err,
                "root_link_vs_pelvis_body_max_abs": pel_body_err}
        sched.forced = None
    log("physical check:", json.dumps({k: v for k, v in phys.items() if k != "per_type"}))
    for nm, v in phys.get("per_type", {}).items():
        log(f"  {nm:22s} ankle-above-mesh mean={v['foot_ankle_above_mesh'].get('mean')} "
            f"std={v['foot_ankle_above_mesh'].get('std')} | pelvis-above-mesh mean={v['pelvis_above_mesh'].get('mean')}")
    meta["physical_check"] = phys
    with open(os.path.join(layout_dir, "meta.json"), "w") as fm:
        json.dump(meta, fm, indent=1)
    log(f"layout exported to {layout_dir} ({time.time() - t_start:.0f}s since start)")

    # ------------------------------------------------------------------ rollouts
    T = int(round(args_cli.seconds / core.step_dt))
    writers = []
    B = None
    for r in range(n_roll):
        if rollout_done(r):
            log(f"rollout {r}: already finished (resume) -> skipped")
            continue
        if os.path.exists(raw_file(r)):
            log(f"rollout {r}: raw file exists (resume) -> writing clips only")
            launch_writer(r, writers)
            continue
        rseed = args_cli.rollout_seed_base + 1000 * seed + r
        t_roll = time.time()
        torch.manual_seed(rseed)
        np.random.seed(rseed % (2 ** 32))
        random.seed(rseed)
        sched.reseed(rseed)
        REC.yaw_gen.manual_seed(rseed + 7)
        # arms: stratified per tile, alternating across rollouts
        arm_train = ((env_rank + env_tile + r) % 2 == 0)
        if args_cli.arm_train_frac != 0.5:
            arm_train = np.random.default_rng(rseed).random(N) < args_cli.arm_train_frac
        core._homie_rho_a_row = torch.where(torch.as_tensor(arm_train, device=dev),
                                            torch.ones(N, device=dev), torch.full((N,), 1e-6, device=dev))
        if B is None:       # allocated ONCE and reused (a second set next to the first OOM'd on a shared GPU)
            B = {"root_pos": torch.zeros(T, N, 3, device=dev), "root_quat": torch.zeros(T, N, 4, device=dev),
                 "dof_pos": torch.zeros(T, N, 29, device=dev), "cmd": torch.zeros(T, N, 7, device=dev),
                 "term": torch.zeros(T, N, dtype=torch.bool, device=dev),
                 "trunc": torch.zeros(T, N, dtype=torch.bool, device=dev),
                 "term_code": torch.zeros(T, N, dtype=torch.uint8, device=dev)}
        else:
            for v_ in B.values():
                v_.zero_()
        chk = {k: torch.zeros((), device=dev) for k in
               ("cmd_obs_err", "dof_obs_err", "spawn_pos_err", "spawn_quat_err", "spawn_dof_err", "n_starts",
                "cmd_out_of_range")}
        smoke_pushed = np.zeros(N, bool)
        with torch.inference_mode():
            obs, _ = wrapped.reset()
            if args_cli.smoke:
                # D1 instrumentation: envs 0-7 time out at step 199 (episode clock advanced),
                # envs 8-15 get a hard lateral push at step 150 (-> falls -> terminated)
                core.episode_length_buf[0:8] = core.max_episode_length - 200
                smoke_pushed[8:16] = True
            prev_done = torch.ones(N, dtype=torch.bool, device=dev)       # frame 0 follows the full reset
            for k in range(T):
                pos = robot.data.root_link_pos_w
                quat = robot.data.root_link_quat_w
                dof = robot.data.joint_pos[:, idx29_t]
                cmd = term.command_b
                B["root_pos"][k] = pos
                B["root_quat"][k] = quat
                B["dof_pos"][k] = dof
                B["cmd"][k] = cmd
                # --- live checks (no host sync)
                got = obs[:, newest:newest + 7]
                chk["cmd_obs_err"] = torch.maximum(chk["cmd_obs_err"], (got - expected_obs_cmd(cmd)).abs().max())
                q_obs = obs[:, newest + 13:newest + 13 + 29] + default29
                chk["dof_obs_err"] = torch.maximum(chk["dof_obs_err"], (q_obs - dof).abs().max())
                oor = ((cmd < sched.lo - 1e-6) | (cmd > sched.hi + 1e-6)).any(dim=1).sum()
                chk["cmd_out_of_range"] = chk["cmd_out_of_range"] + oor
                pd_ = prev_done.float()
                chk["spawn_pos_err"] = torch.maximum(chk["spawn_pos_err"], ((pos - REC.spawn_pos).abs().max(1)[0] * pd_).max())
                qd = 1.0 - (quat * REC.spawn_quat).sum(1).abs()
                chk["spawn_quat_err"] = torch.maximum(chk["spawn_quat_err"], (qd * pd_).max())
                chk["spawn_dof_err"] = torch.maximum(
                    chk["spawn_dof_err"], ((dof - robot.data.default_joint_pos[:, idx29_t]).abs().max(1)[0] * pd_).max())
                chk["n_starts"] = chk["n_starts"] + pd_.sum()
                if args_cli.smoke and k == 150:
                    v = robot.data.root_link_vel_w.clone()
                    v[8:16, 1] += 3.0
                    robot.write_root_velocity_to_sim(v[8:16], env_ids=torch.arange(8, 16, device=dev))
                # --- step
                obs, _, dones, _ = wrapped.step(policy(obs))
                B["term"][k] = core.reset_terminated
                B["trunc"][k] = core.reset_time_outs
                code = torch.zeros(N, dtype=torch.uint8, device=dev)
                for tn, bit in U.TERM_BITS.items():
                    if tn in core.termination_manager.active_terms:
                        code |= core.termination_manager.get_term(tn).to(torch.uint8) * bit
                B["term_code"][k] = code
                prev_done = dones.bool()
                if (k + 1) % 250 == 0:
                    log(f"rollout {r} step {k + 1}/{T} ({time.time() - t_roll:.0f}s) "
                        f"terminated so far={int(B['term'][:k + 1].sum().item())}")
        sim_s = time.time() - t_roll
        host = {kk: v.cpu().numpy() for kk, v in B.items()}               # one host transfer per rollout
        torch.cuda.empty_cache()
        chk_h = {kk: float(v.item()) for kk, v in chk.items()}
        info = {"layout_seed": seed, "rollout": r, "rollout_seed": rseed, "split": split, "num_envs": N,
                "frames": T, "fps": 50, "sim_seconds": round(sim_s, 1), "checks": chk_h,
                "arm_train_frac": float(np.mean(arm_train)), "smoke": bool(args_cli.smoke)}
        log(f"rollout {r} done in {sim_s:.0f}s; checks {json.dumps(chk_h)}; "
            f"terminated={int(host['term'].sum())} timeouts={int(host['trunc'].sum())}")
        raw_path = raw_file(r)
        extra = {"smoke_pushed": smoke_pushed} if args_cli.smoke else {}
        with open(raw_path + ".tmp", "wb") as fraw:                       # atomic: a crash never leaves a
            np.savez(fraw, **host, env_row=env_row.astype(np.int32),      # truncated raw file behind
                     env_col=env_col.astype(np.int32), arm_train=arm_train, info_json=json.dumps(info), **extra)
        os.replace(raw_path + ".tmp", raw_path)
        with open(os.path.join(out, "rollout_stats", f"L{seed}_r{r}_recorder_checks.json"), "w") as fc:
            json.dump(info, fc, indent=1)
        launch_writer(r, writers)
    rc_all = wait_writers(writers)
    log(f"layout {seed} complete in {time.time() - t_start:.0f}s rc={rc_all}")
    return rc_all


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        import traceback

        traceback.print_exc()
        rc = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)  # Kit's close() can hang with background threads; all outputs are flushed above
