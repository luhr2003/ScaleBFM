"""Vectorised equivalent of scripts/pretrain/data_process/package_motions.py.

Same numerics and the same call sequence per frame as the reference script (write root/joint state, sim.render(),
scene.update(), read the articulation), but all per-env Python loops and per-tensor `.cpu()` copies are replaced by
batched tensor operations, so packaging a large corpus takes minutes instead of hours. Outputs are byte-compatible
`.npz` files (joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, fps).

Additionally supports:
  --shard i/n     process only files with (stable hash of relative path) % n == i, so several GPUs can share a corpus
  --keep_origin   do NOT add the env-grid xy origin (default: same as the reference script, which adds it and never
                  subtracts it). Terrain clips MUST use --subtract_origin so that positions stay in the terrain frame.
  --subtract_origin  subtract the env-grid xy origin again before saving (positions then equal the input frame)
"""

import argparse
import glob
import hashlib
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Package retargeted motions (fast).")
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--data_format", type=str, default="pkl")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--output_fps", type=int, default=50)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--robot_type", type=str, default="g1_29dof")
parser.add_argument("--shard", type=str, default="0/1")
parser.add_argument("--subtract_origin", action="store_true")
parser.add_argument("--max_files", type=int, default=0)
parser.add_argument("--max_env_frames", type=float, default=5e6, help="GPU buffer budget: envs x frames per batch")
parser.add_argument("--skip_existing", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from scaletrack.robots.g1_29dof import G1_29DOF_CYLINDER_CFG as ROBOT_CFG  # noqa: E402
from scaletrack.robots.g1_29dof import G1_29DOF_JOINT_NAMES as ROBOT_JOINT_NAMES  # noqa: E402


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_conjugate(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def axis_angle_from_quat(quat, eps=1.0e-6):
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48)
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)


def quat_slerp_batched(q1, q2, tau):
    """Element-wise replica of isaaclab.utils.math.quat_slerp (scalar tau, one pair) for (N,4) inputs and (N,) tau,
    including its early-return branches, so results match the per-frame loop of the reference script."""
    eps4 = torch.finfo(q1.dtype).eps * 4.0
    d = (q1 * q2).sum(-1)
    identical = torch.abs(torch.abs(d) - 1.0) < eps4
    neg = d < 0.0
    d = torch.where(neg, -d, d)
    q2f = torch.where(neg[:, None], -q2, q2)
    angle = torch.acos(torch.clamp(d, -1, 1))
    tiny = torch.abs(angle) < eps4
    isin = 1.0 / torch.sin(angle)
    out = q1 * (torch.sin((1.0 - tau) * angle) * isin)[:, None] + q2f * (torch.sin(tau * angle) * isin)[:, None]
    out = torch.where((identical | tiny)[:, None], q1, out)
    out = torch.where((tau == 0.0)[:, None], q1, out)
    out = torch.where((tau == 1.0)[:, None], q2, out)
    return out


def load_single_motion(input_dir, motion_file, output_dt):
    """Same interpolation and finite differences as the reference script; the slerp loop is vectorised."""
    try:
        torch.set_num_threads(1)
        motion_dict = joblib.load(motion_file)
        fps = motion_dict["fps"]
        base_pos = torch.from_numpy(np.asarray(motion_dict["root_pos"])).float()
        base_rot = torch.from_numpy(np.asarray(motion_dict["root_rot"]))[:, [3, 0, 1, 2]].float()
        dof_pos = torch.from_numpy(np.asarray(motion_dict["dof_pos"])).float()

        input_frames = base_pos.shape[0]
        input_dt = 1.0 / fps
        duration = (input_frames - 1) * input_dt
        times = torch.arange(0, duration, output_dt, dtype=torch.float32)
        output_frames = times.shape[0]

        phase = times / duration
        index_0 = (phase * (input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(input_frames - 1))
        blend = phase * (input_frames - 1) - index_0

        motion_base_pos = base_pos[index_0] * (1 - blend.unsqueeze(1)) + base_pos[index_1] * blend.unsqueeze(1)
        motion_base_rot = quat_slerp_batched(base_rot[index_0], base_rot[index_1], blend)
        motion_dof_pos = dof_pos[index_0] * (1 - blend.unsqueeze(1)) + dof_pos[index_1] * blend.unsqueeze(1)

        base_lin_vel = torch.gradient(motion_base_pos, spacing=output_dt, dim=0)[0]
        dof_vel = torch.gradient(motion_dof_pos, spacing=output_dt, dim=0)[0]
        q_prev, q_next = motion_base_rot[:-2], motion_base_rot[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * output_dt)
        base_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

        return {
            "base_pos": motion_base_pos, "base_rot": motion_base_rot, "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel, "dof_pos": motion_dof_pos, "dof_vel": dof_vel,
            "output_frames": output_frames,
            "file_name": "_".join(os.path.relpath(os.path.splitext(motion_file)[0], input_dir).split("/")),
        }
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load {motion_file}: {e}", flush=True)
        return None


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def stable_shard(path: str, n: int) -> int:
    return int(hashlib.md5(path.encode()).hexdigest(), 16) % n


def pad_to(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[0] >= length:
        return x[:length]
    return torch.cat([x, x[-1:].expand(length - x.shape[0], *x.shape[1:])], dim=0)


def run(sim: SimulationContext, scene: InteractiveScene):
    device = sim.device
    files = sorted(glob.glob(f"{args_cli.data_dir}/**/*.{args_cli.data_format}", recursive=True))
    i_shard, n_shard = (int(x) for x in args_cli.shard.split("/"))
    files = [f for f in files if stable_shard(os.path.relpath(f, args_cli.data_dir), n_shard) == i_shard]
    if args_cli.max_files:
        files = files[: args_cli.max_files]
    os.makedirs(args_cli.output_dir, exist_ok=True)
    if args_cli.skip_existing:
        def out_name(f):
            return "_".join(os.path.relpath(os.path.splitext(f)[0], args_cli.data_dir).split("/")) + ".npz"
        files = [f for f in files if not os.path.exists(os.path.join(args_cli.output_dir, out_name(f)))]
    print(f"[pack] shard {args_cli.shard}: {len(files)} files", flush=True)
    if not files:
        return

    output_dt = 1.0 / args_cli.output_fps
    motions = joblib.Parallel(n_jobs=-1, verbose=0)(joblib.delayed(load_single_motion)(args_cli.data_dir, f, output_dt) for f in files)
    motions = [m for m in motions if m is not None and m["output_frames"] > 2]
    motions.sort(key=lambda m: m["output_frames"])
    print(f"[pack] loaded {len(motions)} motions, {sum(m['output_frames'] for m in motions)} frames", flush=True)

    robot = scene["robot"]
    joint_idx = torch.as_tensor(robot.find_joints(ROBOT_JOINT_NAMES, preserve_order=True)[0], device=device)
    num_envs = args_cli.num_envs
    origins_xy = scene.env_origins[:, :2].clone()
    t0 = time.time()
    done = 0
    batches, i = [], 0
    while i < len(motions):  # motions are sorted by length: shrink the batch until envs x longest <= budget
        j = min(i + num_envs, len(motions))
        while j - i > 1 and (j - i) * motions[j - 1]["output_frames"] > args_cli.max_env_frames:
            j = i + max(1, (j - i) * 3 // 4)
        batches.append(motions[i:j])
        i = j
    print(f"[pack] {len(batches)} batches", flush=True)
    for batch in batches:
        n = len(batch)
        T = max(m["output_frames"] for m in batch)

        def stack(key):
            arr = torch.stack([pad_to(m[key], T) for m in batch], dim=0)  # (n,T,...)
            if n < num_envs:
                arr = torch.cat([arr, arr[-1:].expand(num_envs - n, *arr.shape[1:])], dim=0)
            return arr.to(device)

        base_pos, base_rot = stack("base_pos"), stack("base_rot")
        base_lin, base_ang = stack("base_lin_vel"), stack("base_ang_vel")
        dof_pos, dof_vel = stack("dof_pos"), stack("dof_vel")

        nb = robot.data.body_pos_w.shape[1]
        nj = robot.data.joint_pos.shape[1]
        buf = {
            "joint_pos": torch.empty(T, n, nj, device=device), "joint_vel": torch.empty(T, n, nj, device=device),
            "body_pos_w": torch.empty(T, n, nb, 3, device=device), "body_quat_w": torch.empty(T, n, nb, 4, device=device),
            "body_lin_vel_w": torch.empty(T, n, nb, 3, device=device), "body_ang_vel_w": torch.empty(T, n, nb, 3, device=device),
        }
        default_root = robot.data.default_root_state.clone()
        default_jp, default_jv = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        for t in range(T):
            root_states = default_root.clone()
            root_states[:, :3] = base_pos[:, t]
            root_states[:, :2] += origins_xy
            root_states[:, 3:7] = base_rot[:, t]
            root_states[:, 7:10] = base_lin[:, t]
            root_states[:, 10:] = base_ang[:, t]
            robot.write_root_state_to_sim(root_states)
            jp, jv = default_jp.clone(), default_jv.clone()
            jp[:, joint_idx] = dof_pos[:, t]
            jv[:, joint_idx] = dof_vel[:, t]
            robot.write_joint_state_to_sim(jp, jv)
            sim.render()
            scene.update(sim.get_physics_dt())
            buf["joint_pos"][t] = robot.data.joint_pos[:n]
            buf["joint_vel"][t] = robot.data.joint_vel[:n]
            buf["body_pos_w"][t] = robot.data.body_pos_w[:n]
            buf["body_quat_w"][t] = robot.data.body_quat_w[:n]
            buf["body_lin_vel_w"][t] = robot.data.body_lin_vel_w[:n]
            buf["body_ang_vel_w"][t] = robot.data.body_ang_vel_w[:n]
        if args_cli.subtract_origin:
            buf["body_pos_w"][..., :2] -= origins_xy[:n][None, :, None, :]
        host = {k: v.permute(1, 0, *range(2, v.dim())).cpu().numpy() for k, v in buf.items()}  # (n,T,...)
        for i, m in enumerate(batch):
            L = m["output_frames"]
            np.savez(
                os.path.join(args_cli.output_dir, f"{m['file_name']}.npz"),
                fps=args_cli.output_fps,
                **{k: host[k][i, :L] for k in host},
            )
        done += n
        print(f"[pack] {done}/{len(motions)} motions, {time.time() - t0:.0f}s", flush=True)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplayMotionsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
    sim.reset()
    run(sim, scene)


if __name__ == "__main__":
    main()
    # simulation_app.close() can hang for minutes on exit (the reference script skips it too); all files are written.
    import sys

    sys.stdout.flush()
    os._exit(0)
