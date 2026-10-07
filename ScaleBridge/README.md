# ScaleBridge

ScaleBridge provides an integrated Sim2Sim and Sim2Real deployment pipeline for Behavior Foundation Models on humanoid robots.

This guide covers workstation and Jetson setup, policy evaluation, policy deployment, and migration instructions. Run all commands from the ScaleBridge root unless stated otherwise.

> [!IMPORTANT]
> Real-robot deployment can cause sudden or unexpected motion. Keep an emergency stop within reach before launching the policy.

> [!NOTE]
> Deploying on the **CRB G1**? Follow [CRB-G1-Deploy.md](CRB-G1-Deploy.md) alongside this README. It covers that robot's extra setup: building the Livox driver and FAST-LIO in a separate `livox_ws` (without the driver's `build.sh`, which wipes the workspace), the `MID360_config.json` network settings, the FAST-LIO → ScaleBridge bridge for LiDAR-based global tracking, and the environment pitfalls we hit on that robot.

## 🧭 Table of contents

- [🛠️ 1. Prepare the environment](#️-1-prepare-the-environment)
- [📡 2. Configure optional tracking devices](#-2-configure-optional-tracking-devices)
- [🦾 3. Build the robot controller](#-3-build-the-robot-controller)
- [🎮 4. Launch the low-level controller](#-4-launch-the-low-level-controller)
- [🚀 5. Launch a policy](#-5-launch-a-policy)
- [💡 6. Examples](#-6-examples)
- [📦 7. Migrate the policy](#-7-migrate-the-policy)

## 🛠️ 1. Prepare the environment

Choose the setup instructions for the computer that will run the policy.

### Linux workstation

ScaleBridge require a **Python 3.11** environment on Linux workstation. (You may use the same environment with ScaleTrack.)

```bash
git clone https://github.com/zengweishuai/ScaleBFM.git
cd ScaleBFM/ScaleBridge
# Activate your environment with Python 3.11
pip install -e .
```

Install LCM development files for communication between the policy and the low-level robot controller:

```bash
sudo apt install liblcm-dev
```

Connect the workstation to the robot and assign its network interface an address in the `192.168.123.xxx` subnet, for example `192.168.123.222`. Verify connectivity with the G1 PC2 onboard computer:

```bash
ping 192.168.123.164
```

### NVIDIA Jetson (aarch64)

Clone ScaleBridge on the Jetson:

```bash
git clone https://github.com/zengweishuai/ScaleBFM.git
cd ScaleBFM/ScaleBridge
```

Update the device to JetPack 6.2 by following the [NVIDIA GR00T Whole-Body Control JetPack guide](https://nvlabs.github.io/GR00T-WholeBodyControl/references/jetpack6.html), then install JetPack and CUDA 12.6:

```bash
sudo apt-get update
sudo apt-get install nvidia-jetpack
sudo apt-get install cuda-toolkit-12-6
```

Install cuSPARSELt:

```bash
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
tar xf libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/include/. /usr/local/cuda/include/
sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib/. /usr/local/cuda/lib64/
```

Install LCM development files for communication between the policy and the low-level robot controller:

```bash
sudo apt install liblcm-dev
```

Choose one of the following methods to prepare the Python environment.

#### Option 1: Use the prebuilt Conda environment

Download the prebuilt aarch64 Conda environment, `scalebridge.tar.gz`, from [ScaleBFM on Hugging Face](https://huggingface.co/WeishuaiZeng/ScaleBFM/tree/main/environment). Transfer the archive to the Jetson and extract it into the Conda environments directory:

```bash
scp scalebridge.tar.gz unitree@192.168.123.164:~/anaconda3/envs
ssh unitree@192.168.123.164
cd ~/anaconda3/envs
tar -xzf scalebridge.tar.gz
conda activate scalebridge
```

From the ScaleBridge root, run the editable installation to update the dependencies required by ScaleBridge and register the current checkout:

```bash
cd ~/ScaleBFM/ScaleBridge
pip install -e .
```

#### Option 2: Install from scratch

Activate a **Python 3.10** environment on the Jetson.

Install the JetPack-compatible PyTorch packages:

> [!WARNING]
> At the time this code was collected, the package source below appeared to be broken. This installation method may therefore not work as expected.

```bash
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

Install the bundled TensorRT and Torch-TensorRT wheels:

```bash
pip install dependencies/tensorrt-10.3.0-cp310-none-linux_aarch64.whl
pip install dependencies/torch_tensorrt-2.8.0+cu126-cp310-cp310-linux_aarch64.whl
```

Finally, install the ScaleBridge package:

```bash
pip install -e .
```

## 📡 2. Configure optional tracking devices

### Xsens for real-time teleoperation

1. Install and launch the Xsens software on a Windows computer.
2. Configure network streaming to the computer that runs the policy.
3. Select TCP and the default position-and-orientation data set.
4. Set the same port in Xsens and `env.config.xsens_port`.
5. If Xsens uses Manus gloves, enable them before streaming; hand data is included automatically.

### Onboard state estimator for root localization (default)

Global tracking (`env.config.reference_forcing=False`) needs the pelvis position in the world frame. By default, ScaleBridge estimates it onboard without extra hardware: a Kalman filter fuses the pelvis IMU with leg kinematics, weighting each foot by a contact probability derived from contact forces that a momentum observer estimates from joint torques. It is a ROS-free port of the `legged_control2` state estimator that BeyondMimic deploys, and it runs on every 200 Hz robot state from the low-level controller.

- Parameters are in `scalebridge/config/localization/legged_estimator.yaml`, and the model is `scalebridge/data/robot/g1_29dof/g1_29dof.urdf`.
- It is odometry: the estimate drifts slowly with foot slip, assumes flat ground, and only uses the feet as contacts. Jumps, kneeling, sitting and falls degrade it, and a foot whose center of pressure leaves the sole bounds, for example under a strong lateral push, is ignored until it settles.
- The BFM was trained with ground-truth root positions. Validate each motion in MuJoCo with `simulator.config.estimate_root_pos=True` before running it on the robot; the simulator then uses the estimate and logs its error against the ground truth.

See the [legged state estimator manual](docs/legged_estimator.md) for how it works, tuning, validation results and troubleshooting. For stairs and uneven ground the default estimator is not enough (it assumes flat ground); the manual's section 10 describes the experimental `legged_estimator_stairs`, `legged_estimator_lidar` (Mid-360 / FAST-LIO) and `legged_estimator_depth` (D435i) variants. To use Vive trackers instead, add `localization=vive_tracker` to the launch command and follow the next section.

### Vive Ultimate Tracker for root localization

1. Obtain [Vive Ultimate Trackers](https://www.vive.com/us/accessory/vive-ultimate-tracker/).
2. Print the pelvis connector from `accessories/vive_tracker_for_G1.stl`.
3. Launch Vive Hub and SteamVR, map the environment, and confirm that at least two trackers are available.
4. Place tracker 0 on the floor and attach tracker 1 to the G1 pelvis.
5. Check `port` in `scalebridge/config/localization/vive_tracker.yaml`. The default receiver port is `5000`.
6. Copy the `scalebridge/utils/vive_sender` directory to your Windows device.
7. Launch the tracker sender `tcp_send_raw_tracker_array.py` on the Windows computer using the policy computer's IP address and the same port.

## 🦾 3. Build the robot controller

The robot controller serves as the bridge between the policy and the physical robot. During Sim2Real deployment, it reads the robot's sensor and joint state, forwards that state to the high-level policy, and applies the policy's joint commands to the robot. It must run alongside the high-level policy when using `simulator=real_world`; it is not needed for MuJoCo Sim2Sim evaluation.

### General command

Set `ROBOT_TYPE` to `g1_29dof` for the standard G1 or `g1_29dof_dex3` for the G1 with Dex3 hands, then build the matching controller.

```bash
cd third_party/unitree_sdk2
mkdir -p build
cd build
cmake .. -DROBOT_TYPE=ROBOT_TYPE
make
```

> [!IMPORTANT]
> Use `g1_29dof_dex3` only when you want to control the hands. Otherwise, use `g1_29dof` even if the robot is equipped with hands.

## 🎮 4. Launch the low-level controller

> [!WARNING]
> Suspend the robot securely before launching the controller. It automatically moves the robot to its default pose.

Use the same `ROBOT_TYPE` selected during compilation and pass the robot-facing network interface:

```bash
cd third_party/unitree_sdk2/build/bin
./"${ROBOT_TYPE}_controller" NETWORK_INTERFACE
```

On a Linux workstation, use the interface configured above. For onboard deployment, use the interface associated with the G1 PC2 address, `192.168.123.164`.

## 🚀 5. Launch a policy

Place each compiled checkpoint and its metadata file in the same model directory:

```text
scalebridge/data/model/robot_type
└── MODEL_NAME/
    ├── linux/
    │   ├── CHECKPOINT_tensorrt.pt
    │   └── CHECKPOINT_tensorrt_metadata.json
    └── aarch64/
        ├── CHECKPOINT_tensorrt.pt
        └── CHECKPOINT_tensorrt_metadata.json
```

The checkpoint and metadata filenames must share the same prefix. For example, `model_22200_tensorrt.pt` must be paired with `model_22200_tensorrt_metadata.json`.

> [!NOTE]
> Download the compiled models for Unitree G1 from [ScaleBFM on Hugging Face](https://huggingface.co/WeishuaiZeng/ScaleBFM/tree/main/compiled_checkpoint/), then place the checkpoint and metadata files in the structure shown above under `scalebridge/data/model/g1_29dof`. Each model includes a `linux` version compiled on an NVIDIA GeForce RTX 4090 Linux workstation and an `aarch64` version compiled onboard. Use the version that matches your deployment platform because TensorRT checkpoints are platform- and device-specific.

### General command

```bash
python scalebridge/run.py \
  agent.config.control_mode=MODE_INDEX \
  agent.config.checkpoint=/path/to/checkpoint.pt \
  asset=ROBOT_TYPE \
  env=TASK \
  env.config.reference_forcing=REFERENCE_FORCING \
  simulator=BACKEND
```

The command is composed from four Hydra configuration groups:

| Group | Purpose | Available values | Default |
| --- | --- | --- | --- |
| `agent` | Loads the BFM checkpoint and performs inference | `bfm_agent` | `bfm_agent` |
| `asset` | Selects the robot embodiment | `g1_29dof`, `g1_29dof_dex3` | `g1_29dof` |
| `env` | Selects the corresponding task | `motion_tracking`, `motion_tracking_xsens` | `motion_tracking` |
| `simulator` | Selects the execution backend | `mujoco_simulator`, `real_world` | `mujoco_simulator` |

> [!NOTE]
> Test every policy with `simulator=mujoco_simulator` before using `simulator=real_world`.
>
> Regardless of the selected control mode, the markers always show the whole-body target in global coordinates. When `reference_forcing=True`, they are not expected to align with the execution results.

### Configuration details

Here we provide a brief explanation of each configuration group. For more details, refer to the configuration files under `scalebridge/config`.

<details>
<summary><strong>🤖 Agent configuration</strong></summary>

Set `agent.config.checkpoint` to the compiled policy checkpoint. TensorRT artifacts are platform-specific: a checkpoint compiled on Linux x86_64 cannot be loaded directly on aarch64. Recompile it for the deployment platform by following the export instructions in the training codebase.

Set `agent.config.control_mode` to one of the following modes:

| Mode index | Control mode | Activated links |
| ---: | --- | --- |
| `0` | Root Mode | Pelvis |
| `1` | Bimanual Mode | Left Hand, Right Hand |
| `2` | Root-and-Hand Mode | Pelvis, Left Hand, Right Hand |
| `3` | End-Effector Mode | Left Hand, Right Hand, Left Foot, Right Foot |
| `4` | Root-and-End-Effector Mode | Pelvis, Left Hand, Right Hand, Left Foot, Right Foot |
| `5` | Upper-Body Mode | Left and Right Shoulders, Elbows, and Hands |
| `6` | Root-and-Upper-Body Mode | Pelvis, Left and Right Shoulders, Elbows, and Hands |
| `7` | Whole-Body Mode | Pelvis, Torso, Left and Right Shoulders, Elbows, Hands, Hips, Knees, and Feet |

ScaleBridge currently uses one fixed control mode for the full deployment session. Applications that require mode switching must implement that logic separately.

</details>

<details>
<summary><strong>🦾 Asset configuration</strong></summary>

The default asset is `g1_29dof`. Use the Dex3 configuration with:

```bash
asset=g1_29dof_dex3
```

When using `g1_29dof_dex3`, Manus gloves, and `env=motion_tracking_xsens`, also enable hand tracking:

```bash
env.config.has_hand=true
```

</details>

<details>
<summary><strong>🎯 Environment configuration</strong></summary>

The default option is `env=motion_tracking`, which replays an offline reference motion. Specify the motion file with `env.config.motion_path`

For real-time teleoperation, set `env=motion_tracking_xsens` and configure the Xsens receiver with `env.config.xsens_port`. ScaleBridge directly scales the human skeleton and does not include a separate retargeting pipeline. Adjust `env.config.xsens_scale_factor` to improve alignment.

Both tasks support local and global tracking through `env.config.reference_forcing`:

| Value | Tracking mode | Behavior |
| --- | --- | --- |
| `True` | Local | Uses the reference root position as the current root position |
| `False` | Global | Uses the root-localization module: the onboard state estimator (`localization=legged_estimator_lidar`, default; `localization=legged_estimator` for flat ground without a LiDAR) or Vive trackers (`localization=vive_tracker`) |

</details>

<details>
<summary><strong>🎮 Simulator configuration</strong></summary>

The simulator configuration controls execution timing and optional visualization features. Override an option from the command line with `simulator.config.OPTION=VALUE`.

| Option | Backend | Default | Purpose |
| --- | --- | --- | --- |
| `low_dt` | All | `0.005` | Low-level control timestep in seconds |
| `decimation` | All | `4` | Number of low-level steps per policy step |
| `record_video` | MuJoCo | `False` | Records `recording.mp4` in the Hydra output directory |
| `marker` | MuJoCo | `True` | Displays reference-body markers |
| `camera_follow` | MuJoCo | `True` | Keeps the camera centered on the robot |
| `estimate_root_pos` | MuJoCo | `False` | In global tracking, uses the onboard state estimator on simulated sensors instead of the ground-truth root position |

For example, enable video recording during MuJoCo evaluation with:

```bash
simulator=mujoco_simulator simulator.config.record_video=True
```

</details>

## 💡 6. Examples

### Replay an offline motion in Whole-Body Mode with local tracking

1. Validate the compiled policy and reference motion with MuJoCo Sim2Sim on the Linux workstation:

   ```bash
   python scalebridge/run.py \
     agent.config.control_mode=7 \
     agent.config.checkpoint=/path/to/checkpoint.pt \
     asset=g1_29dof \
     env=motion_tracking \
     env.config.reference_forcing=True \
     env.config.motion_path=/path/to/motion.npz \
     simulator=mujoco_simulator
   ```

2. Proceed only after confirming that the policy performs as expected in simulation. Suspend the robot securely, then launch the low-level controller:

   ```bash
   cd third_party/unitree_sdk2/build/bin
   ./g1_29dof_controller NETWORK_INTERFACE
   ```

3. In a separate terminal, launch the policy with the real-world backend:

   ```bash
   python scalebridge/run.py \
     agent.config.control_mode=7 \
     agent.config.checkpoint=/path/to/checkpoint.pt \
     asset=g1_29dof \
     env=motion_tracking \
     env.config.reference_forcing=True \
     env.config.motion_path=/path/to/motion.npz \
     simulator=real_world
   ```

4. Carefully lower the robot until its feet contact the ground and it reaches a stable standing configuration. If the balance point is difficult to locate, gradually release tension from the suspension system while keeping the robot supported.

5. Follow the instructions in the terminal and press `R2` once on the remote controller to calibrate the robot state.

6. After confirming that the robot and surrounding area are ready, press `R2` again to establish communication between the policy and the controller and begin execution.

> [!IMPORTANT]
> In any emergency, press `L2 + B`. A **single press** immediately switches the robot to damping mode and stops policy-command execution. A **double press** terminates the controller after it enters damping mode, which may cause the robot to suddenly lose actuator support and go limp.


### Replay an offline motion in Whole-Body Mode with global tracking

1. Validate the compiled policy and reference motion with MuJoCo Sim2Sim on the Linux workstation:

   ```bash
   python scalebridge/run.py \
     agent.config.control_mode=7 \
     agent.config.checkpoint=/path/to/checkpoint.pt \
     asset=g1_29dof \
     env=motion_tracking \
     env.config.reference_forcing=False \
     env.config.motion_path=/path/to/motion.npz \
     simulator=mujoco_simulator
   ```

   Repeat with `simulator.config.estimate_root_pos=True` to run the onboard state estimator in the loop, as on the robot.

2. Proceed only after confirming that the policy performs as expected in simulation. Suspend the robot securely, then launch the low-level controller:

   ```bash
   cd third_party/unitree_sdk2/build/bin
   ./g1_29dof_controller NETWORK_INTERFACE
   ```

3. Skip steps 3 and 4 unless you use Vive trackers (`localization=vive_tracker`). Connect the Vive Tracker and put the one with index 0 on the ground, with its logo up. Mount another one with index 1 on the robot's hip **in the same orientation shown in the figure**.

   <p align="center">
     <img src="assets/vive_install.png" alt="Vive Tracker mounted on the robot's hip">
   </p>


4. Copy the `scalebridge/utils/vive_sender` directory to your Windows device. This is independent from `scalebridge`.:
    ```bash
    pip install argparse openvr
    cd vive_sender/
    python tcp_send_raw_tracker_array.py <deploy_device_ip> --port <deploy_device_port>
    ```

5. In a separate terminal, launch the policy with the real-world backend:

   ```bash
   python scalebridge/run.py \
     agent.config.control_mode=7 \
     agent.config.checkpoint=/path/to/checkpoint.pt \
     asset=g1_29dof \
     env=motion_tracking \
     env.config.reference_forcing=False \
     env.config.motion_path=/path/to/motion.npz \
     simulator=real_world
   ```

6. Carefully lower the robot until its feet contact the ground and it reaches a stable standing configuration. If the balance point is difficult to locate, gradually release tension from the suspension system while keeping the robot supported.

7. Follow the instructions in the terminal and press `R2` once on the remote controller to calibrate the robot state. Ensure the robot stands still on flat ground; the state estimator restarts from this pose. With Vive trackers, also ensure the Vive Tracker faces directly to the bottom in a well-aligned pose.

8. After confirming that the robot and surrounding area are ready, press `R2` again to establish communication between the policy and the controller and begin execution.

### Teleoperate the robot with Xsens

1. Validate the compiled policy and reference motion with MuJoCo Sim2Sim on the Linux workstation:

   ```bash
   python scalebridge/run.py \
     agent.config.control_mode=7 \
     agent.config.checkpoint=/path/to/checkpoint.pt \
     asset=g1_29dof \
     env=motion_tracking_xsens \
     env.config.reference_forcing=True \
     simulator=mujoco_simulator
   ```

2. Proceed only after confirming that the policy performs as expected in simulation. Suspend the robot securely, then launch the low-level controller:

   ```bash
   cd third_party/unitree_sdk2/build/bin
   ./g1_29dof_controller NETWORK_INTERFACE
   ```

3. In a separate terminal, launch the policy with the real-world backend:

   ```bash
   python scalebridge/run.py \
     agent.config.control_mode=7 \
     agent.config.checkpoint=/path/to/checkpoint.pt \
     asset=g1_29dof \
     env=motion_tracking_xsens \
     env.config.reference_forcing=True \
     simulator=real_world
   ```

4. Carefully lower the robot until its feet contact the ground and it reaches a stable standing configuration. If the balance point is difficult to locate, gradually release tension from the suspension system while keeping the robot supported.

5. Follow the terminal prompts and press `R2` once on the remote controller to calibrate the robot state. During calibration, the human operator should stand upright, preferably facing the same direction as the robot.

6. After confirming that the robot and surrounding area are ready, press `R2` again to establish communication between the policy and the controller and begin execution.


## 📦 7. Migrate the policy

The exported policy consumes raw sensor observations, so it can be integrated into another deployment codebase.

### Load the policy

Confirm that compatible versions of PyTorch and Torch-TensorRT are installed, then load the checkpoint:

```python
import torch
import torch_tensorrt

policy = torch.jit.load(ckpt_path).to("cuda")
policy.eval()
```

If the checkpoint and runtime versions are incompatible, recompile the checkpoint by following the training codebase instructions.

### Run inference

```python
tgt_dof_pos, action = policy(
    root_quat_buffer,
    base_ang_vel_buffer,
    dof_pos_buffer,
    dof_vel_buffer,
    actions_buffer,
    target_body_pos_future_to_robot_base,
    target_body_rot_future_to_robot_base,
    control_mode,
    future_time_offsets,
)
```

The policy inputs are:

| Input | Shape | Description |
| --- | --- | --- |
| `root_quat_buffer` | `(1, 3, 4)` | Pelvis IMU quaternion history in `wxyz` order |
| `base_ang_vel_buffer` | `(1, 3, 3)` | Pelvis IMU angular-velocity history in `xyz` order |
| `dof_pos_buffer` | `(1, 3, num_joints)` | Raw joint-position history; order follows metadata `joint_names` |
| `dof_vel_buffer` | `(1, 3, num_joints)` | Raw joint-velocity history; order follows metadata `joint_names` |
| `actions_buffer` | `(1, 3, num_actions)` | Raw action history; order follows metadata `action_names` |
| `target_body_pos_future_to_robot_base` | `(1, 6, 14, 3)` | Target body positions expressed in the root frame |
| `target_body_rot_future_to_robot_base` | `(1, 6, 14, 4)` | Target body rotations in `wxyz` order, expressed in the root frame |
| `control_mode` | `(1,)` | Control-mode index |
| `future_time_offsets` | `(6,)` | Time offsets corresponding to the target-body future frames |

For both target-body tensors, link order follows metadata `selected_body_names`. The six future frames are `[0, 1, 2, 3, 4, X]`, where `X` can be any step from 5 through 33 and each step represents `0.02 s`.

The policy returns:

| Output | Shape | Description |
| --- | --- | --- |
| `tgt_dof_pos` | `(1, num_actions)` | Target joint positions after action scale and offset are applied |
| `action` | `(1, num_actions)` | Raw actions before scale and offset are applied |

Both output orders follow metadata `action_names`. The checkpoint metadata also contains:

| Key | Description |
| --- | --- |
| `stiffness` | Execution stiffness values in `action_names` order |
| `damping` | Execution damping values in `action_names` order |
