# ScaleBridge on the CRB G1

> [!IMPORTANT]
> This guide is **specific to the Unitree G1 in the CRB**. It records what we had to install and change on top of the
> [ScaleBridge README](README.md) to run ScaleBFM onboard that robot, including global tracking with the LiDAR. Read the
> README first; this file only covers the differences. Other G1s (different firmware, PC2 image or sensors) may not need
> all of it.

Robot setup this guide was written against:

| Item | Value |
| --- | --- |
| PC2 | Jetson Orin NX 16 GB, L4T R36.4.3 (JetPack 6.2), CUDA 12.6, TensorRT 10.3, Ubuntu 22.04, Python 3.10 |
| ROS | ROS 2 Humble (preinstalled on PC2) |
| Internal network | `enP8p1s0`, PC2 `192.168.123.164`, Livox Mid-360 `192.168.123.120` |
| Where everything runs | onboard PC2 (policy, controller, LiDAR stack) |

## Contents

1. [Python environment](#1-python-environment)
2. [Models](#2-models)
3. [Robot controller](#3-robot-controller)
4. [LiDAR stack (Livox driver + FAST-LIO)](#4-lidar-stack-livox-driver--fast-lio)
5. [Running](#5-running)
6. [Terminal rules](#6-terminal-rules)
7. [Troubleshooting](#7-troubleshooting)

## 1. Python environment

Use the README's **Option 1** (prebuilt environment). It is a `conda-pack` archive, so **conda itself is not needed**:

```bash
cd ~/ScaleBFM
curl -L -o scalebridge.tar.gz https://huggingface.co/WeishuaiZeng/ScaleBFM/resolve/main/environment/scalebridge.tar.gz
mkdir -p scalebridge_env && tar -xzf scalebridge.tar.gz -C scalebridge_env
source scalebridge_env/bin/activate
conda-unpack                      # once: rewrites the archive's install paths (a script inside the env, not conda)
cd ScaleBridge && pip install -e .   # adds pin (pinocchio) and pyzmq for the legged estimator
python -c "import torch, torch_tensorrt; print(torch.__version__, torch.cuda.is_available())"   # expect 2.8.0 True
```

- To leave it: `source ~/ScaleBFM/scalebridge_env/bin/deactivate`.
- cuSPARSELt (README step) was **not** needed: the env's torch has no missing libraries on this image.
- `sudo apt install liblcm-dev` is still needed for the controller.

## 2. Models

### Pretrained model (from Hugging Face)

```bash
D=~/ScaleBFM/ScaleBridge/scalebridge/data/model/g1_29dof/humanoid_transformer_m/aarch64; mkdir -p $D
B=https://huggingface.co/WeishuaiZeng/ScaleBFM/resolve/main/compiled_checkpoint/humanoid_transformer_m/aarch64
for f in model_22200_tensorrt.pt model_22200_tensorrt_metadata.json; do curl -L -o $D/$f $B/$f; done
```

### Terrain fine-tune `ft_v4_soupV4a` (compiled onboard)

The checkpoints in `ScaleTrack/checkpoints/` are **Git LFS** files. A plain clone on PC2 only has 133-byte pointer
stubs: run `git lfs pull` or copy the real file (72 MB, sha256 `8069ce97…c3e98`) over. Then:

```bash
cd ~/ScaleBFM
D=ScaleBridge/scalebridge/data/model/g1_29dof/ft_v4_soupV4a/aarch64; mkdir -p $D
cp ScaleTrack/checkpoints/ft_v4_soupV4a_it28300-28600.pt $D/ft_v4_soupV4a.pt
# mode_table.pt only exists in the Hugging Face *linux* folder; the architecture is the same, so it applies
curl -L -o $D/mode_table.pt https://huggingface.co/WeishuaiZeng/ScaleBFM/resolve/main/compiled_checkpoint/humanoid_transformer_m/linux/mode_table.pt
cp ScaleBridge/scalebridge/data/model/g1_29dof/humanoid_transformer_m/aarch64/model_22200_tensorrt_metadata.json $D/ft_v4_soupV4a_tensorrt_metadata.json
source scalebridge_env/bin/activate
env -u LD_LIBRARY_PATH python ScaleTrack/scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer_onboard.py --checkpoint $D/ft_v4_soupV4a.pt --mode_table $D/mode_table.pt --metadata $D/ft_v4_soupV4a_tensorrt_metadata.json --xml_path ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml
```

- Paste the compile command as **one line**; a line break after `python` opens an interactive prompt instead.
- It writes `ft_v4_soupV4a_tensorrt.pt` next to the checkpoint. Expect **20–40 min** on the Orin NX (one CPU core at
  100 %, empty lines and many `IUnsqueezeLayer … TensorRT 10.7` warnings are normal). Inference afterwards: ~9 ms.

## 3. Robot controller

Build as in the README (`cmake .. -DROBOT_TYPE=g1_29dof && make` in `third_party/unitree_sdk2/build`). Differences:

- **Run it without the ROS environment.** PC2's `~/.bashrc` sources ROS 2, whose CycloneDDS `libddsc` replaces the
  Unitree SDK's and crashes the controller at start-up (`corrupted size vs. prev_size`):
  ```bash
  cd ~/ScaleBFM/ScaleBridge/third_party/unitree_sdk2/build/bin
  env -u LD_LIBRARY_PATH -u CYCLONEDDS_URI -u RMW_IMPLEMENTATION ./g1_29dof_controller enP8p1s0
  ```
- The controller **releases Unitree's built-in motion service itself** at start-up (the software equivalent of
  L2+R2 / debug mode), so the remote is not needed for that. The robot goes limp: it must be hanging.
- **E-stop (damping, kp = 0):** `L2 + B` on the remote, or **Enter** / the first **Ctrl+C** in the controller
  terminal. A second Ctrl+C exits. After the start-up Enter, any Enter in that terminal is the e-stop.
- ScaleBridge's calibrate/start also accept the keyboard: **Enter** = calibrate (R2), typing **`start`** + Enter =
  start the policy (second R2).

## 4. LiDAR stack (Livox driver + FAST-LIO)

Only needed for **global tracking** with `localization=legged_estimator_lidar` (the default localization).

> [!NOTE]
> On this G1 the Unitree DDS LiDAR topics that the [deployment manual](docs/real_robot_deployment_zh.md) assumes
> (`rt/utlidar/cloud_livox_mid360`, `rt/utlidar/imu_livox_mid360`) **are not published**; only `/utlidar/range_info`
> exists. The Mid-360 itself is up at `192.168.123.120`. So we run the Livox driver ourselves to get `/livox/lidar` and
> `/livox/imu`. This step is specific to the CRB G1.

### 4.1 Livox-SDK2

[Livox-SDK2](https://github.com/Livox-SDK/Livox-SDK2), with two missing `#include <cstdint>` added (needed with newer
GCC, harmless on PC2's GCC 11):

```bash
cd ~ && git clone https://github.com/Livox-SDK/Livox-SDK2.git && cd Livox-SDK2
sed -i 's|#include <string>|#include <string>\n#include <cstdint>|' sdk_core/comm/define.h
sed -i 's|#include <map>|#include <map>\n#include <cstdint>|' sdk_core/logger_handler/file_manager.h
mkdir -p build && cd build && cmake .. && make -j4 && sudo make install && sudo ldconfig
```

### 4.2 Workspace `~/livox_ws`

Create a separate workspace with a `src` folder and clone both repos into it:

- [livox_ros_driver2](https://github.com/Livox-SDK/livox_ros_driver2) (official, `master`)
- [FAST_LIO, `ROS2` branch](https://github.com/hku-mars/FAST_LIO/tree/ROS2) (official hku-mars repo; `--recursive` for
  the ikd-Tree submodule)

```bash
mkdir -p ~/livox_ws/src && cd ~/livox_ws/src
git clone https://github.com/Livox-SDK/livox_ros_driver2.git
git clone -b ROS2 --recursive https://github.com/hku-mars/FAST_LIO.git
```

**Livox driver: do not use its `build.sh`.** It runs `rm -rf ../../build ../../install` on the whole workspace first.
Do its two preparation steps by hand, then set the network in `config/MID360_config.json`:

```bash
cd ~/livox_ws/src/livox_ros_driver2
cp -f package_ROS2.xml package.xml && cp -rf launch_ROS2/ launch/
cd config
sed -i 's/"192.168.1.5"/"192.168.123.164"/g; s/"192.168.1.12"/"192.168.123.120"/' MID360_config.json
grep -nE '"(cmd_data_ip|push_msg_ip|point_data_ip|imu_data_ip|ip)"|roll' MID360_config.json   # 4x .164, 1x .120, "roll": 0.0
```

Keep **`"roll": 0.0`** (see [4.5](#45-frames-the-lidar-data-is-upside-down)).

**FAST-LIO:** the stock `config/mid360.yaml` already has the right topics (`/livox/lidar`, `/livox/imu`),
`lidar_type: 1` and the Mid-360's internal IMU offset `extrinsic_T [-0.011, -0.02329, 0.04412]`. We only turned off
outputs we don't need (backup kept as `mid360.yaml.orig`):

```yaml
publish:
  map_en: false
  scan_publish_en: false
pcd_save:
  pcd_save_en: false   # otherwise a map file keeps growing on disk
```

### 4.3 Build

Use a terminal **without `scalebridge_env` active**: that env has empy 4, which breaks ROS 2 Humble's message
generation (`module 'em' has no attribute 'BUFFERED_OPT'`).

```bash
source /opt/ros/humble/setup.bash
cd ~/livox_ws && colcon build --packages-select livox_ros_driver2 --cmake-args -DROS_EDITION=ROS2 -DDISTRO_ROS=humble
source ~/livox_ws/install/setup.bash
rosdep install --from-paths src --ignore-src -y
MAKEFLAGS=-j2 colcon build --symlink-install --packages-select fast_lio --parallel-workers 1   # -j2: avoids running out of RAM
```

### 4.4 Run

Every LiDAR terminal starts with:

```bash
source /opt/ros/humble/setup.bash && source ~/livox_ws/install/setup.bash
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="enP8p1s0"/></Interfaces></General></Domain></CycloneDDS>'
```

(PC2's own `~/.local/cyclone_config.xml` binds ROS 2 to the Wi-Fi dongle, where the robot's topics are not visible.)

| Terminal | Command | Check |
| --- | --- | --- |
| A | `ros2 launch livox_ros_driver2 msg_MID360_launch.py` | `ros2 topic hz /livox/lidar` ≈ 10 Hz, `/livox/imu` ≈ 200 Hz |
| B | `ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml rviz:=false` (robot **still** while it starts) | `ros2 topic hz /Odometry` ≈ 10 Hz |
| C | `python3 ~/ScaleBFM/ScaleBridge/scalebridge/utils/legged_estimator/fastlio_bridge_ros2.py` | logs `… poses forwarded` every 10 s |

Once the driver (A) and FAST-LIO (B) are up, start the bridge (C). It lives in this repository, in
[Haoran's ScaleBFM fork](https://github.com/luhr2003/ScaleBFM), at
`ScaleBridge/scalebridge/utils/legged_estimator/fastlio_bridge_ros2.py` (it is not in the original authors' repository).
It is the ROS 2 port of MagicLoco's ROS 1 `fastlio_bridge.py`: it forwards `/Odometry` to the ZeroMQ pose wire on
`tcp://*:5606` that the legged estimator reads. Run it with the system Python and ROS 2 sourced (needs `rclpy`,
`numpy`, `pyzmq`). **Keep it running**: without it ScaleBridge waits 30 s for the first pose and exits.

The driver and FAST-LIO need only the configuration changes listed above (no forks): `MID360_config.json`
host/LiDAR IPs, `roll: 0.0`, and the three output switches in FAST-LIO's `mid360.yaml`.

**Acceptance:** robot standing still for 5 min, the bridge's `last pos` z stays within ±2 cm (we measured ~2 cm over
12 min; x/y jitter of 1–3 cm is normal scan-matching noise).

### 4.5 Frames: the LiDAR data is upside down

The Mid-360 is mounted rolled 180° on the head, and FAST-LIO's map frame (`camera_init`) is simply the sensor's own
pose when FAST-LIO started. So `/Odometry` is in an **upside-down frame** (z down, y mirrored); at start-up it reports
identity orientation. **This is expected and fine**: at calibrate, the legged estimator
(`scalebridge/utils/legged_estimator/lidar_odometry.py`) aligns FAST-LIO's frame to its own z-up pelvis frame with the
**full 3D rotation** between the two (from the pelvis IMU, joint angles and the URDF's `mid360_link`), and converts the
LiDAR position to a pelvis position with the head-to-pelvis lever from kinematics. Do **not** flip the data in the
driver (`roll: 180`): the upstream driver rotates the points but not its IMU, which breaks FAST-LIO.

> [!NOTE]
> The full-rotation alignment is a fix on top of the original code, which only aligned the heading (yaw) and therefore
> read an upside-down FAST-LIO frame mirrored (up to 0.84 m error in simulation; 0.11 m after the fix). Verified on this
> robot: moving it forward / left / up increases the estimate's x / y / z. In MuJoCo,
> `simulator.config.lidar_odom.body_frame_odom=true` emulates FAST-LIO's upside-down frame.

## 5. Running

Robot hanging, Unitree's other controllers (e.g. SONIC) stopped. Terminal 1: the controller (Section 3).

**Local tracking** (no localization; start here):

```bash
cd ~/ScaleBFM && source scalebridge_env/bin/activate && cd ScaleBridge
env -u LD_LIBRARY_PATH python scalebridge/run.py \
  agent.config.control_mode=7 \
  agent.config.checkpoint=$PWD/scalebridge/data/model/g1_29dof/ft_v4_soupV4a/aarch64/ft_v4_soupV4a_tensorrt.pt \
  asset=g1_29dof env=motion_tracking \
  env.config.reference_forcing=True \
  env.config.motion_path=$PWD/scalebridge/data/motion/g1_29dof/squat.npz \
  simulator=real_world
```

**Global tracking with the LiDAR:** start terminals A–C (Section 4.4), then replace
`env.config.reference_forcing=True` with `env.config.reference_forcing=False localization=legged_estimator_lidar`.
Add `LOGURU_LEVEL=DEBUG` in front to see the estimated pelvis position once a second. After Enter (calibrate) expect
`[LidarOdometry] Aligned to the estimator frame, yaw offset … deg`. Without the LiDAR, use
`localization=legged_estimator` (legs + IMU, flat ground only).

Sequence: Enter (calibrate) → lower the robot until it stands → `start` → give slack gradually. When done: take the
weight on the gantry, **L2+B** (or Enter in the controller terminal), then Ctrl+C both. Do not Ctrl+C the policy first
while the robot stands: the controller would freeze it stiffly in its last pose.

Motion clips we used besides the bundled `squat.npz` (not tracked by git, `.gitignore` excludes `*.npz`) start with
~20 s of standing so the policy is balancing before the motion begins:
`stand20_then_squat.npz` (3 squats) and `stand20_walkshort_armsforward.npz` (~0.9 m straight walk, arms held forward,
waist straight, starts from the controller's hold pose with a 3 s blend).

## 6. Terminal rules

| Terminal | Environment | Why |
| --- | --- | --- |
| Controller | `env -u LD_LIBRARY_PATH -u CYCLONEDDS_URI -u RMW_IMPLEMENTATION` | ROS 2's DDS libraries crash the Unitree SDK |
| ScaleBridge (`run.py`, TensorRT compile) | `scalebridge_env` + `env -u LD_LIBRARY_PATH` | keep ROS libraries out of the policy process |
| LiDAR (driver, FAST-LIO, bridge), `colcon build` | system Python + ROS 2 sourced + `CYCLONEDDS_URI` on `enP8p1s0`, **no** `scalebridge_env` | empy 4 in the env breaks ROS builds; Wi-Fi-bound DDS sees nothing |

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Controller aborts with `corrupted size vs. prev_size` | ROS 2's `libddsc` loaded; start it with the `env -u …` command (Section 3) |
| `colcon build` fails with `em` / `BUFFERED_OPT` | `scalebridge_env` is active; `source ~/ScaleBFM/scalebridge_env/bin/deactivate` or open a new terminal, then `rm -rf build/<pkg>` |
| ScaleBridge stuck at "Waiting up to 30 s for the first LiDAR odometry pose" | the bridge (terminal C) is not running |
| Estimated position runs away by metres while hanging | no LiDAR poses and no planted foot, so only the IMU is integrated; start the bridge. Calibrate resets the estimate; never `start` global mode without the LiDAR flowing or the robot standing |
| `ros2 topic list` shows no robot topics | DDS on the wrong interface; export the `CYCLONEDDS_URI` above |
| No internet on PC2 (`No route to host` to GitHub) | the internal Ethernet's default route wins over Wi-Fi; `sudo ip route del default via 192.168.123.1 dev enP8p1s0`, or `sudo nmcli con mod "<enP8p1s0 connection>" ipv4.never-default yes` |
| Robot buzzes / fights while the controller runs | another program also sends `rt/lowcmd` (Unitree's service or SONIC); stop it |
| Slight jolt at the first policy command | known: the controller's hold pose and gains (`g1_29dof.hpp`) differ from the policy's defaults and switch at `start`. SONIC ramps to the policy's own default pose and gains instead; fix pending |
