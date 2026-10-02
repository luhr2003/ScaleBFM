# Legged State Estimator Manual

Global tracking (`env.config.reference_forcing=False`) needs the pelvis position in a fixed world frame. The legged state estimator provides it from the robot's own sensors (joint encoders, joint torques and the pelvis IMU), so no external tracker is needed. It is a ROS-free Python port of the `legged_control2` state estimator that BeyondMimic deploys on the Unitree G1, reconstructed from the released binaries and checked against them.

- [1. Usage](#1-usage)
- [2. How it works](#2-how-it-works)
- [3. Configuration](#3-configuration)
- [4. Validation](#4-validation)
- [5. Limitations and safety](#5-limitations-and-safety)
- [6. Troubleshooting](#6-troubleshooting)
- [7. Should the policy be fine-tuned?](#7-should-the-policy-be-fine-tuned)
- [8. Provenance](#8-provenance)
- [9. File map](#9-file-map)
- [10. Stairs and uneven ground](#10-stairs-and-uneven-ground)

## 1. Usage

The default localization is `legged_estimator_lidar`: this estimator plus the pose of a LiDAR odometry (FAST-LIO on the Mid-360), which works on flat ground and on stairs and needs FAST-LIO and its bridge running (section 10 and [real_robot_deployment_zh.md](real_robot_deployment_zh.md)). `localization=legged_estimator`, described in sections 1 to 9, is the original estimator without a LiDAR for flat ground. Both are used whenever global tracking is enabled (`env.config.reference_forcing=False`). Local tracking (`reference_forcing=True`) does not use it.

### Install

Run `pip install -e .` from the ScaleBridge root. It installs `pin` (pinocchio) along with the other dependencies. Wheels exist for Python 3.10 and 3.11 on x86_64 and aarch64, so the same command works on the Jetson, including in the prebuilt Conda environment.

### Step 1: validate the motion in MuJoCo

Run the motion with global tracking and the estimator in the loop. The simulator feeds simulated joint states, actuator torques and IMU readings to the estimator, and the policy sees the estimate instead of the ground truth:

```bash
python scalebridge/run.py \
  agent.config.control_mode=7 \
  agent.config.checkpoint=/path/to/checkpoint.pt \
  asset=g1_29dof \
  env=motion_tracking \
  env.config.reference_forcing=False \
  env.config.motion_path=/path/to/motion.npz \
  simulator=mujoco_simulator \
  simulator.config.estimate_root_pos=True
```

Once per second the simulator logs the estimate against the ground truth:

```text
[Simulator] Root estimate error: xy 0.043 m, z +0.003 m
```

Then run the same command with `simulator.config.estimate_root_pos=False` (ground truth). The motion is a good candidate for the robot when both runs look alike. In the tests of section 4, the xy error stayed below about 0.3 m and the z error below 1 cm on long walking and running motions. Errors that grow steadily, or a z error of several centimeters, mean the motion breaks the estimator's assumptions (section 5).

### Step 2: run it on the robot

1. Suspend the robot securely and start the low-level controller, as in the README:

   ```bash
   cd third_party/unitree_sdk2/build/bin
   ./g1_29dof_controller NETWORK_INTERFACE
   ```

2. In a separate terminal, launch the policy with global tracking. No tracker or sender is needed:

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

   The log shows `Localization module client connected: robot_state_data over LCM` once the first robot state arrives. The estimator subscribes to the `robot_state_data` LCM channel that the low-level controller already publishes at 200 Hz.

3. Lower the robot until both feet are flat on the ground and it stands stably.
4. Press `R2` once to calibrate. The robot must stand still: the estimator restarts from this pose, which becomes the origin, and the motion is aligned to the same origin and to the IMU heading.
5. Press `R2` again to start. In an emergency, press `L2 + B`.

Start with in-place motions (for example `scalebridge/data/motion/g1_29dof/squat.npz`), then short walks, before long locomotion. Compare against local tracking (`env.config.reference_forcing=True`) on the same motion.

The same steps work with `asset=g1_29dof_dex3`. The hand joints are ignored, but their mass is not in the estimator's model (see section 6).

### Read the estimate

Once per second the client logs the estimated root position and the contact probability of each foot at DEBUG level:

```text
[LeggedEstimator] Root position [0.121, -0.0, 0.752] m, contact probabilities [0.74, 0.74]
```

`scalebridge/run.py` writes DEBUG logs to `run.log` in the Hydra output directory. Set `LOGURU_LEVEL=DEBUG` to also print them in the terminal, and set `localization.log_period` to change the period (0 disables it). On flat ground the height should stay close to the standing pelvis height (about 0.75 to 0.79 m). The contact probabilities should be about 0.6 to 0.8 per foot in double stance, about 1 on a stance foot, and 0 on a swing foot.

### Change parameters

Every key in `scalebridge/config/localization/legged_estimator.yaml` can be overridden on the command line, for example:

```bash
python scalebridge/run.py ... localization.estimator.contact_zmp_length_y=0.035 localization.estimator.contact_force_threshold=120
```

Section 3 describes each parameter.

### Use Vive trackers instead

Add `localization=vive_tracker` to the launch command and follow the Vive section of the README.

### Use the estimator from Python

The estimator does not depend on ScaleBridge's runtime and can be used directly:

```python
import numpy as np
from scalebridge.utils.legged_estimator import LeggedStateEstimator

estimator = LeggedStateEstimator("scalebridge/data/robot/g1_29dof/g1_29dof.urdf")
print(estimator.joint_names)  # joint order expected by update()

# Call at every sensor sample, standing still first:
position = estimator.update(
    dt,                   # seconds since the previous sample
    joint_pos, joint_vel, # (29,) each, in estimator.joint_names order
    joint_tau,            # (29,) measured joint torques (Unitree tau_est)
    quat_wxyz=imu_quat,   # pelvis IMU orientation, w first
    gyro=imu_gyro,        # pelvis angular velocity in the IMU frame
    acc=imu_acc,          # raw accelerometer reading (about +9.81 on z at rest)
)
estimator.reset()         # restart from the current pose; the pelvis xy becomes the origin
```

`estimator.velocity` gives the world-frame velocity, and `estimator.kalman_filter.contact_probabilities` gives the per-foot contact probabilities.

## 2. How it works

### Data flow

```text
low-level controller (C++, 200 Hz)
  └─ LCM robot_state_data: q, qd, tau_est (29 joints), IMU quat / gyro / accelerometer
       └─ LeggedEstimatorOnlineClient (own LCM thread, one update per message)
            └─ LeggedStateEstimator
                 ├─ LeggedModel          pinocchio model of the G1 (free-flyer base)
                 ├─ GmObserver           external joint torques -> contact wrenches
                 └─ LinearKalmanFilter   base position / velocity
RealWorld.refresh_sim() ── get_root_pos() ──> root_pos_buffer ──> policy observation
```

### Per-update sequence

This is the order of `legged_controllers::StateEstimator::update_and_write_commands`, confirmed in the decompiled code:

1. Write the sensors into the model: `q[3:7]` = IMU quaternion (xyzw), `q[7:]` = joint positions, `v[3:6]` = gyro (base frame), `v[6:]` = joint velocities, `tau` = measured joint torques. `q[0:3]` and `v[0:3]` keep the previous estimate.
2. `model.update()`: forward kinematics, joint Jacobians, frame placements.
3. Momentum observer update, then contact wrenches.
4. Kalman filter: set contact wrenches and the raw accelerometer reading, predict with the IMU, update with the contacts.
5. Write the estimated position and the base-frame velocity back into `q[0:3]` and `v[0:3]`.

Orientation is never estimated; it is taken from the IMU as-is. Only translation is estimated.

### Robot model

`scalebridge/data/robot/g1_29dof/g1_29dof.urdf` is the description `legged_control2` loads, with a free-flyer root. Contacts are the 6-DoF frames `LL_FOOT` / `LR_FOOT` at the sole centers, `(0.04, 0, -0.037)` in the ankle roll links; the base frame is `pelvis`. Frame Jacobians are expressed in `LOCAL_WORLD_ALIGNED`.

### Generalized-momentum observer

With `p = M(q) v`, cutoff `f_c` and step `dt`:

```text
gamma  = 1 / (1 + 2 pi f_c dt)
beta   = (1 - gamma) / (gamma dt)          (= 2 pi f_c)
alpha  = beta p + C(q, v)^T v + S^T tau - g(q)
y      = alpha                             first update
y      = gamma y + (1 - gamma) alpha       afterwards
tau_ext = beta p - y
```

`M`, `C` and `g` come from pinocchio (`crba`, `computeCoriolisMatrix`, `computeGeneralizedGravity`) and `S = [0 | I]` selects the actuated joints.

### Contact wrenches

The contact Jacobians of both feet are stacked with the pelvis Jacobian (18 x 35), so disturbances on the base are not attributed to the feet:

```text
(J J^T + lambda I) f = J tau_ext,    lambda = 1e-8 max(diag(J J^T))
```

The damping keeps the solve bounded when a knee reaches full extension. `f` holds one world-aligned wrench `[force; torque]` per foot, about the sole center.

### Contact probability

For each foot:

```text
p_force = sigmoid((F_z - force_threshold) / force_scale)          F_z in world axes
f, tau  = R_foot^T force, R_foot^T torque                          foot frame
cop     = (-tau_y / f_z, tau_x / f_z)                              if |f_z| > 1e-6, else vetoed
p_zmp   = 1 if |cop_x| <= zmp_length_x and |cop_y| <= zmp_length_y else 0
p       = p_force * p_zmp          (non-finite values become 0)
```

With the defaults, a foot carrying half the robot's weight (about 160 N) gets `p` of about 0.66, and a foot carrying all of it about 1.

### Kalman filter

State (15): `[base_pos, base_vel, left_foot_pos, right_foot_pos, acc_bias]`, all in the world frame.

Prediction, with `R` the IMU orientation and `a` the raw accelerometer reading:

```text
u = R a + (0, 0, -9.81)
x = A x + B u,     A: pos += dt vel,  pos -= 0.5 dt^2 R bias,  vel -= dt R bias
                   B: pos += 0.5 dt^2 u,  vel += dt u
P = A P A^T + Q
Q = dt I, then
    Q[0:6, 0:6]   = sigma_a^2 [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]]
    Q[feet]      *= sigma_c^2,  and each foot block /= (p_i + 1e-3)
    Q[bias]      *= sigma_b^2
```

Contact update (after recomputing the probabilities):

```text
y_i   = q[0:3] - p_foot_i(FK),  plus contact_radius on z     base minus foot, world axes
h_i   = 0                                                     flat ground
R_m   = dt I, scaled by sigma_s^2 (positions) and sigma_h^2 (heights), each foot's rows /= (p_i + 1e-3)
K     = P C^T (C P C^T + R_m)^-1
K[:, foot i columns] = 0   if not p_i >= 0.01
x    += K (y - C x)
P     = (I - K C) P
```

A foot with high probability pins its position state and constrains the base through leg kinematics; a swinging foot gets huge process noise and zero gain, so its state follows whatever the next touchdown reports.

### Reset and calibration

`reset()` zeroes the state, places the base at `-mean(foot z)` above the ground (from kinematics) with `x = y = 0`, and sets `P` to the nominal `Q` at `dt = 0.002`. The port also zeroes the contact probabilities, which makes a mid-run reset behave exactly like the first activation in `legged_control2`: the first prediction inflates the foot process noise so the foot states, not the base, absorb the current foot positions.

The online client resets at calibration (first `R2`) and reports positions relative to that pose. The estimate shares the IMU heading, which is the heading ScaleBridge uses to align the motion, so no rotation is applied.

## 3. Configuration

`scalebridge/config/localization/legged_estimator.yaml` (keys under `estimator:`). Defaults equal the `legged_control2` defaults, which BeyondMimic's G1 deployment does not override.

| Key | `legged_control2` parameter | Default | Effect |
| --- | --- | --- | --- |
| `urdf_path` | robot description | `g1_29dof.urdf` | Kinematics and dynamics of the estimator |
| `base_name` | `model.base_name` | `pelvis` | Frame whose position is estimated |
| `contact_names` | `model.six_dof_contact_names` | `[LL_FOOT, LR_FOOT]` | Sole-center contact frames |
| `cutoff_frequency` | `gm_observer.cut_off_frequency` | `10.0` Hz | Contact force bandwidth; higher reacts faster but is noisier |
| `imu_acceleration_noise_density` | `estimation.imu.acceleration_noise_density` | `0.01` | Trust in the accelerometer for prediction |
| `imu_acceleration_bias_noise_density` | `estimation.imu.acceleration_bias_noise_density` | `0.005` | How fast the accelerometer bias may wander |
| `contact_process_noise_position` | `estimation.contact.process_noise_position` | `0.002` | How much a planted foot may move (slip) |
| `contact_sensor_noise_position` | `estimation.contact.sensor_noise_position` | `0.002` | Trust in leg kinematics |
| `contact_height_sensor_noise` | `estimation.contact.height_sensor_noise` | `0.002` | Trust in the flat-ground assumption |
| `contact_radius` | `estimation.contact.radius` | `0.0` | Added to the kinematic foot height |
| `contact_force_threshold` | `estimation.contact.force_threshold` | `150` N | Normal force at which `p_force = 0.5` |
| `contact_force_scale` | `estimation.contact.force_scale` | `15` N | Width of the force sigmoid |
| `contact_zmp_length_x` | `estimation.contact.zmp_length_x` | `0.08` m | Center-of-pressure bound along the foot |
| `contact_zmp_length_y` | `estimation.contact.zmp_length_y` | `0.025` m | Center-of-pressure bound across the foot |
| (unused) `position_noise_density` | `estimation.position.noise` | `0.01` | Only for fusing an external position (not wired) |

Client options (`channel`, `lcm_url`, `max_dt`) are top-level keys of the same file. The update step uses the measured interval between LCM messages, clamped to `max_dt`, because the controller does not timestamp them.

Tuning hints:

- The G1 weighs about 327 N, so the default force threshold puts each foot of a double stance near `p = 0.66`. Lowering it (for example to 100 N) makes double stance more decisive; raising it makes the filter rely more on the IMU.
- `zmp_length_y = 0.025` m is narrow. Under strong lateral pushes or wide-stance motions both feet can be vetoed at once, and the filter then dead-reckons on the IMU. Widening it (for example to 0.035 m, roughly the sole half-width) keeps contacts during lateral loading.
- On soft or slippery floors, raise `contact_process_noise_position` so slipping feet are not trusted as fixed.

## 4. Validation

All results below are from MuJoCo. Simulated sensors are noiseless and foot slip is limited to what the contact model produces, so expect larger errors on the real robot.

### Port versus the original library

The real `legged_control2` library, linked in a ROS 2 Jazzy container, and the port were given the same model and pose. Frame indices, the transition, input and measurement matrices, the selection matrix, the IMU noise block, the reset state and the full reset covariance are identical to the last digit. The per-update sequence, parameter names and defaults were confirmed in the decompiled code.

### Contact forces

With the robot standing (about 160 N per foot), the observer's vertical foot forces match MuJoCo's contact normal forces within about 2 N.

### Estimator alone

A scripted balance controller stood the robot for 8 s, squatted it (6 cm of pelvis travel) for 12 s and pushed it forward ten times with 40 N for 20 s. Pelvis xy error stayed at or below 0.7 cm and z error near 0.3 cm, with 0.5 cm xy error after 40 s. Fed through the LCM client with encoded `robot_state_lcmt` messages and calibrated standing still, the error was 0.02 to 0.08 cm in xy and 0.25 cm in z.

### Policy in the loop

`humanoid_transformer_m` (compiled checkpoint, whole-body mode 7) in ScaleBridge's MuJoCo simulator, on the repository squat, three Xsens clips and eight 100STYLE "Neutral" clips from the released Ours test set. Each motion ran three times: global tracking with the ground-truth root, global tracking with the estimator (`simulator.config.estimate_root_pos=True`), and local tracking (`reference_forcing=True`). G-MPKPE is the mean global position error of the 14 tracked bodies; L-MPKPE removes the pelvis xy offset.

| Motion | Length | Path walked | G-MPKPE ground truth | G-MPKPE estimator | G-MPKPE local | L-MPKPE estimator | Estimate xy error mean / max | Estimate z error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Squat (repo sample) | 8 s | 1.4 m | 4.4 cm | 4.6 cm | 5.6 cm | 3.8 cm | 1.9 / 3.5 cm | 0.3 cm |
| Xsens walk (furui-002) | 13 s | 5.9 m | 3.8 cm | 6.9 cm | 18.7 cm | 2.8 cm | 5.7 / 12.0 cm | 0.3 cm |
| Xsens clip (shunlin-148) | 9 s | 2.2 m | 4.9 cm | 4.9 cm | 12.0 cm | 3.1 cm | 1.9 / 3.6 cm | 0.2 cm |
| Xsens clip (weishuai-154) | 8 s | 0.7 m | 7.2 cm | 7.3 cm | 15.7 cm | 4.6 cm | 1.6 / 4.1 cm | 0.4 cm |
| 100Style idle | 35 s | 5.7 m | 5.5 cm | 6.7 cm | 44.2 cm | 2.5 cm | 7.3 / 8.3 cm | 0.3 cm |
| 100Style forward walk | 132 s | 79.1 m | 4.0 cm | 6.4 cm | 36.2 cm | 2.7 cm | 4.9 / 10.5 cm | 0.3 cm |
| 100Style backward walk | 144 s | 71.1 m | 6.4 cm | 9.9 cm | 32.2 cm | 2.8 cm | 7.0 / 14.7 cm | 0.3 cm |
| 100Style sidestep walk | 104 s | 43.6 m | 5.2 cm | 8.2 cm | 55.9 cm | 2.5 cm | 6.2 / 15.4 cm | 0.4 cm |
| 100Style forward run | 77 s | 75.0 m | 8.3 cm | 12.1 cm | 31.3 cm | 3.2 cm | 8.1 / 16.4 cm | 0.5 cm |
| 100Style backward run | 98 s | 71.6 m | 7.9 cm | 11.5 cm | 49.9 cm | 4.2 cm | 8.2 / 15.6 cm | 0.4 cm |
| 100Style sidestep run | 59 s | 44.1 m | 8.2 cm | 14.3 cm | 38.4 cm | 4.0 cm | 11.5 / 24.9 cm | 0.3 cm |
| 100Style transitions | 119 s | 84.1 m | 6.6 cm | 12.5 cm | 74.5 cm | 3.3 cm | 10.4 / 26.9 cm | 0.4 cm |

- No run fell, with any root source.
- The estimator costs 0.1 to 6 cm of global error compared with the ground truth, while local tracking drifts to 12 to 75 cm on every motion except the in-place squat.
- Pose tracking (L-MPKPE 2.5 to 4.6 cm) is unaffected.
- The estimate error stays bounded (at most 27 cm, sidestep running and transitions being the hardest) and does not grow with distance (about 0.1 cm per meter). It is smooth: 99.9 % of 20 ms policy steps change it by at most 1.4 cm.

### Stress test

To see how the policy reacts to estimator failures, the estimate given to the policy was offset by a sudden jump or a constant drift (`est` mode, first 45 s of a motion).

| Motion | Injected into the policy's root | Fell | G-MPKPE | Pose error (L-MPKPE) before / worst 2 s after | Extra pelvis speed after jump |
| --- | --- | --- | --- | --- | --- |
| 100Style forward walk | drift 1 cm/s x, 0 cm/s y | no | 22.9 cm | n/a | n/a |
| 100Style forward walk | drift 2 cm/s x, 2 cm/s y | no | 61.6 cm | n/a | n/a |
| 100Style forward walk | +5 cm x, +0 cm y jump at 15 s | no | 7.2 cm | 2.8 / 4.0 cm | 0.22 m/s |
| 100Style forward walk | +10 cm x, +0 cm y jump at 15 s | no | 10.5 cm | 2.8 / 4.2 cm | 0.25 m/s |
| 100Style forward walk | +20 cm x, +0 cm y jump at 15 s | no | 16.1 cm | 2.8 / 4.5 cm | 0.21 m/s |
| 100Style forward walk | +0 cm x, +10 cm y jump at 15 s | no | 8.5 cm | 2.8 / 3.5 cm | 0.19 m/s |
| 100Style forward walk | +0 cm x, +20 cm y jump at 15 s | no | 15.0 cm | 2.8 / 4.8 cm | 0.16 m/s |
| 100Style idle | drift 1 cm/s x, 0 cm/s y | no | 16.7 cm | n/a | n/a |
| 100Style idle | +10 cm x, +0 cm y jump at 10 s | no | 9.9 cm | 2.4 / 2.7 cm | 0.12 m/s |
| 100Style idle | +20 cm x, +0 cm y jump at 10 s | no | 16.5 cm | 2.4 / 3.1 cm | 0.31 m/s |
| 100Style idle | +0 cm x, +20 cm y jump at 10 s | no | 10.0 cm | 2.4 / 2.6 cm | 0.18 m/s |

- Nothing fell. Pose error rises from 2.4 to 2.8 cm to at most 4.8 cm, and the pelvis moves at most 0.31 m/s faster than the reference.
- While walking, the robot absorbs 80 % of a 10 to 20 cm jump within 0.4 to 1.1 s. While standing, it closes a 20 cm forward offset in 0.4 s but keeps its stance after a 20 cm sideways jump and closes that offset only gradually (about 20 s).
- Under drift the pose error stays at the undisturbed 2.7 cm: the robot follows the drifted reference, and the global error equals the injected drift.

## 5. Limitations and safety

- **Odometry, not localization.** Errors accumulate with foot slip; nothing corrects them over long runs. External position fusion exists in `legged_control2` but is not ported (see section 8).
- **Flat ground, feet only.** The height measurement assumes the feet are at z = 0, and only the soles are contacts. Stairs, slopes, kneeling, sitting, lying down and falls invalidate the estimate. After a fall, stop the policy.
  - Stairs and uneven ground need the variants of section 10; do not use the default estimator there.
- **Flight phases.** With no contact the filter integrates the IMU; short jumps are fine, long aerial phases drift quickly.
- **Calibrate standing still.** `reset()` zeroes the velocity estimate. Calibrating while the robot moves leaves an offset (about 1 cm in tests).
- **Policy sensitivity.** The BFM was trained with ground-truth root positions. Slow drift is harmless (the robot follows a slightly shifted reference), but a sudden jump in the estimate looks like a real tracking error and the policy reacts to it. See section 7.
- **Keep the emergency stop within reach**, exactly as with Vive-based global tracking.

## 6. Troubleshooting

| Symptom | Likely cause | What to check |
| --- | --- | --- |
| `ModuleNotFoundError: pinocchio` | `pin` not installed | `pip install pin` or `pip install -e .` |
| Startup hangs at "Waiting for the localization module client" | No robot state on LCM | Low-level controller running; same `lcm_url` and network interface |
| Estimate drifts while standing still | Contacts not detected | Log `kalman_filter.contact_probabilities`; both near 0 means low estimated forces (check `tau_est` sign and joint order) or the CoP bounds |
| Height wrong by a few centimeters | Model or ground mismatch | Hand or payload mass not in the URDF; uneven floor; `contact_radius` |
| Height or lateral position biased while standing on flat ground | Leg encoder zero offsets | `legged_control2` applies measured per-robot encoder offsets (one maintainer robot had arm joints off by up to 4 degrees); ScaleBridge applies none, so miscalibrated leg encoders bias the kinematics |
| Robot lunges right after the second `R2` | Calibration while moving, or estimate jump | Calibrate standing still; watch the estimate during the first seconds |
| Both feet lose contact during lateral motion | CoP bound too narrow | Increase `contact_zmp_length_y` |
| Large error in sim with `estimate_root_pos=True` only on some motions | Motion leaves the flat-ground, feet-only assumptions | Kneeling, lying, jumping motions; use local mode for them |

## 7. Should the policy be fine-tuned?

Not now. In simulation the policy already tolerates what the estimator does, and what it cannot tolerate, noise training would not fix.

The BFM was trained with the ground-truth root position. The only perturbation was independent uniform noise of ±5 cm on each target-position coordinate (`target_body_pos` and `target_body_pos_rel` in ScaleTrack), never a shared offset, drift or jump. To measure the headroom, the estimate given to the policy was perturbed further (45 s, whole-body mode). Action rate is the mean change of the policy output per step (jitter):

| Motion | Root seen by the policy | G-MPKPE | L-MPKPE | Action rate |
| --- | --- | --- | --- | --- |
| Forward walk | Ground truth | 4.1 cm | 2.7 cm | 0.385 |
| Forward walk | Estimator | 4.7 cm | 2.7 cm | 0.386 |
| Forward walk | Estimator + 2 cm white noise | 4.8 cm | 2.7 cm | 0.427 |
| Forward walk | Estimator + 5 cm white noise | 7.2 cm | 2.8 cm | 0.573 |
| Forward walk | Estimator + 10 cm white noise | 10.7 cm | 3.0 cm | 0.889 |
| Forward walk | Estimator + 2 cm/√s random walk + 2 cm white noise | 11.7 cm | 2.7 cm | 0.425 |
| Forward run | Estimator | 11.1 cm | 3.2 cm | 0.706 |
| Forward run | Estimator + 5 cm white noise | 16.5 cm | 3.3 cm | 0.804 |
| Forward run | Estimator + 10 cm white noise | 17.9 cm | 3.5 cm | 1.032 |

Nothing fell. The estimator adds no jitter compared with the ground truth. The policy only becomes jittery when high-frequency noise reaches about 5 cm, far above what the Kalman filter produces. Drift raises global error but adds no jitter.

- **The estimator's errors are slow and smooth.** The policy treats them as a slightly shifted reference: global error grows by a few centimeters while pose error is unchanged.
- **Sudden errors are handled gracefully.** Jumps of up to 20 cm, far larger than anything the estimator produced in simulation, cause no falls and only a small transient in pose error.
- **Drift is unobservable.** No policy can tell a drifting estimate from a drifting reference, so training with noise cannot remove drift-induced position error; only a better estimate can (external position fusion, section 8).
- **Noise training has a cost.** White noise on the root position, like ScaleTrack's current ±5 cm on target positions or BeyondMimic's ±25 cm on the anchor position, teaches the policy to trust position less, which lowers global accuracy whenever localization is good.

Reconsider fine-tuning only if real-robot logs show one of these:

- Jumps larger than about 20 cm, or frequent ones, for example contact misclassification on a particular floor.
- High-frequency estimate noise of a few centimeters or more, which the table shows is where the policy starts to jitter.
- Failures that trace back to the estimate.

Measure the real estimator first (log `get_root_pos()` during standing and walking), and match the fine-tuning noise to that measurement. Mix unperturbed episodes into the fine-tune so accuracy with good localization (Vive, motion capture) is kept.

Try the deployment-side fixes first: contact parameters (section 3), smoothing the estimate in the client, or external position fusion.

If fine-tuning becomes necessary, use an error model that looks like the estimator rather than white noise:

1. Keep a per-environment root error `e` (world frame), reset to zero at episode start. Make `e_xy` a random walk plus a drift proportional to distance walked (0.5 to 2 %) plus occasional steps (2 to 15 cm), and `e_z` a small Ornstein-Uhlenbeck process (about 0.5 cm).
2. Apply it to the actor only, through `robot_anchor_pos_w + e` and `robot_body_pos_w + e` in `target_body_pos_future_to_robot_base_manual` and `target_body_pos_future_rel_to_robot_base_manual` (`ScaleTrack/source/scaletrack/scaletrack/tasks/tracking/mdp/observations.py`). Keep the critic on the ground truth.
3. Compute the global position rewards against the reference shifted into the same perceived frame, so the policy is not punished for offsets it cannot observe. Keep the 0.5 m termination on the true state, with `e` clipped well below it.
4. Resume from the released `checkpoint/humanoid_transformer_m/model_22200.pt` (`--resume True --load_run ... --checkpoint model_22200.pt`) on the same motion set to avoid forgetting. Evaluate with both the ground truth and the estimator, then export TensorRT again.

A more faithful but heavier option is a batched GPU version of this Kalman filter inside training, driven by the simulator's contact forces.

## 8. Provenance

The estimator was reconstructed from the Debian packages that `legged_control2` publishes for ROS 2 Jazzy (the source repository is private). The packages are Apache-2.0 licensed; the robot description comes from `unitree_description` (BSD).

| Package | Version | Used for |
| --- | --- | --- |
| `ros-jazzy-legged-estimation` (+ `-dbgsym`) | 3.5.0 (2026-09-18) | `GmObserver`, `LinearKalmanFilter` |
| `ros-jazzy-legged-controllers` (+ `-dbgsym`) | 3.5.0 | `StateEstimator`, ROS wrappers, parameters |
| `ros-jazzy-legged-model` (+ `-dbgsym`) | 3.5.0 | `LeggedModel` (pinocchio usage, Jacobian frames) |
| `ros-jazzy-unitree-description` | 1.3.0 | G1 URDF and contact frames |

Method:

1. Headers, build changelogs and exported symbols gave the class layout, parameters and design history.
2. `gdb` disassembly with the debug-symbol packages (source line tables, PLT resolution) recovered every formula, constant and branch of the estimator.
3. Ghidra 12.1 decompiled the estimator, model and controller functions to C with the DWARF merged back into the libraries; this confirmed the per-update sequence, the IMU-to-model mapping and the parameter names.
4. The Unitree hardware interface (`ros-jazzy-unitree-systems`, also decompiled) shows what `legged_control2` feeds the estimator on the real G1: the IMU is `LowState.imu_state` (pelvis), the joint torque is the motor `tau_est`, and joint positions get optional per-robot encoder offsets. ScaleBridge's C++ controller forwards the same IMU and torques over LCM, at 200 Hz instead of 500 Hz.
5. The real library was linked in a ROS 2 Jazzy container once to dump its matrices and reset state. The port reproduces them exactly: frame indices, `A`, `B`, both measurement matrices, the selection matrix, the IMU noise block, the reset state (base height 0.7938637524222113 m) and the full reset covariance.

Function map:

| `legged_control2` (C++) | Port (Python) |
| --- | --- |
| `legged::LeggedModel::{update, getMassMatrix, getCoriolisMatrix, getGeneralizedGravity, getSelectionMatrix, getContactJacobian, getBaseJacobian}` | `LeggedModel.{update, get_*}` |
| `legged::GmObserver::{update, getContactWrenches}` | `GmObserver.{update, get_contact_wrenches}` |
| `legged::sigmoid` | `linear_kalman_filter.sigmoid` |
| `legged::LinearKalmanFilter::{reset, getImuProcessNoiseCovariance, getKalmanGain, updateImuProcess, updateContactsMeasurement, updatePositionMeasurement, updateContactProbabilities}` | `LinearKalmanFilter.{same names in snake_case}` |
| `legged_controllers::StateEstimator::update_and_write_commands` | `LeggedStateEstimator.update` |

Not ported:

- ROS publishing: odometry, TF, joint states, contact wrench topics, visualization markers.
- External position fusion. `StateEstimator` can subscribe to an odometry topic (for example LiDAR-inertial odometry of the head MID-360). It converts each new message to a pelvis position through the sensor frame's kinematics, estimates a yaw-only rotation between the odometry map and the estimator's world, low-pass filters it with a slerp, and calls `updatePositionMeasurement` with `dt` equal to the time between messages. `LinearKalmanFilter.update_position_measurement` is ported and ready for this.
- 3-DoF (point) contacts are ported but untested.

Two deliberate deviations: `sigmoid` uses `scipy.special.expit` (same values without overflow warnings), and `reset()` also zeroes the contact probabilities (section 2).

## 9. File map

| Path | Content |
| --- | --- |
| `scalebridge/utils/legged_estimator/legged_model.py` | pinocchio model wrapper |
| `scalebridge/utils/legged_estimator/gm_observer.py` | Momentum observer and contact wrenches |
| `scalebridge/utils/legged_estimator/linear_kalman_filter.py` | Kalman filter and contact probabilities |
| `scalebridge/utils/legged_estimator/state_estimator.py` | Per-update sequence, reset, NaN guard |
| `scalebridge/utils/legged_estimator/online_client.py` | LCM client with the localization-module interface |
| `scalebridge/config/localization/legged_estimator.yaml` | Estimator parameters (default localization) |
| `scalebridge/config/localization/vive_tracker.yaml` | Vive tracker alternative |
| `scalebridge/simulator/mujoco_simulator.py` | `estimate_root_pos` hook for sim validation |
| `scalebridge/data/robot/g1_29dof/g1_29dof.urdf` | Estimator model |

## 10. Stairs and uneven ground

The default estimator assumes flat ground: every planted foot is measured at z = 0. On stairs that pins the pelvis height to the floor level of the start, and the whole climb is lost (in MuJoCo, 0 of 10 stair clips pass with the default estimator against 9 of 10 with ground-truth position; see the table). Three variants remove the assumption. All are experimental and were evaluated only in MuJoCo with the policy in the loop; none of the real-robot parts (D435, FAST-LIO bridge) was run on hardware.

| Variant (`localization=`) | What it adds | Needs |
| --- | --- | --- |
| `legged_estimator_stairs` | Drops the flat-ground height measurement, widens the center-of-pressure window to the real sole (0.10 x 0.04 m) and detects contact at a lower load (100 N). The height follows the stance foot through leg kinematics. | nothing |
| `legged_estimator_lidar` | The above plus the absolute pose of a LiDAR odometry (FAST-LIO on the Livox Mid-360), fused as a 10 Hz position measurement (`LidarOdometryFusion`). | MagicLoco's `sim2real/perception/fastlio_bridge.py` on PC2 (pose wire, port 5606) |
| `legged_estimator_depth` | The above stairs variant plus the ground height under each planted foot from a rolling height map of a torso D435i depth camera (`DepthGroundHeight`), replacing the zero. | a depth source feeding `DepthGroundHeight.update` (MuJoCo only for now) |

Why the legs alone drift: once the flat-ground height is removed, the common height of base and feet has no absolute reference and is only held by IMU integration, so a gravity residual of 0.01 m/s^2 moves it by centimeters per second, and every step with a lost contact adds more. With the stairs variant alone the height error after a 4 m climb is about 0.2 to 0.4 m, and it fails when jogging on flat ground (flight phases), so use it only for stairs.

**LiDAR.** The LiDAR odom frame has an arbitrary origin and heading. At calibration (first `R2`) the first pose defines the alignment: yaw from the LiDAR body orientation against the IMU heading, position against the estimate. The pelvis position is the LiDAR position minus the pelvis-to-`mid360_link` lever from the estimator's own kinematics (the URDF has the frame), so the waist joints are accounted for. The fused estimate follows the LiDAR, so its error is the LiDAR drift: FAST-LIO has to be healthy (MagicLoco asks for a height drift below 2 cm over 5 minutes standing still). `body` in the wire message selects the frame (0 `mid360_link`, 1 `torso_link`, 2 `pelvis`).

Latency and outliers are handled in `LidarOdometryFusion`. A pose arrives `latency` seconds late (the wire timestamp is the sender's clock, so it is a parameter, set 0.1 s by default; measure it on the robot): the estimator's own motion since then is added, so the delayed pose is compared with the state it describes. A pose further than `gate` (0.3 m) from the estimate is ignored; if that lasts `realign_after` (1 s) the new pose is trusted and only the alignment is shifted, so a FAST-LIO relocalization never makes the estimate jump. In MuJoCo (9 clean stair clips, 20 s, 0.5 cm/s drift; estimate against ground truth, max / mean): no delay z 0.14 / 0.05 m and xy 0.13 / 0.05 m; 0.1 s delay uncompensated z 0.31 / 0.11 m, xy 0.51 / 0.20 m; compensated z 0.23 / 0.05 m, xy 0.26 / 0.11 m; a 0.5 m jump of the odometry at 8 s z 0.28 / 0.13 m (a 0.2 m jump is inside the gate and followed: z 0.42 / 0.18 m).

**D435i.** The map is built in the estimator's own frame, so it provides consistency over the seconds a cell stays in memory, not an absolute reference; the same idea as the FK z-anchor and the whole-map z bias of MagicLoco's perception stack. The cell update gain decays with the number of frames that saw the cell (`memory_frames`); without that the map follows the estimate's drift and feeds it back (measured: the height drifted 0.4 m in 6 s while standing).

**Real robot D435i.** `depth.source: wire` receives the frames of `python3 rs_probe.py --serve` (MagicLoco `sim2real/perception/tools`) running on PC2, where the camera is plugged in (ZeroMQ port 5609, topic `cam1`, intrinsics in the header); `depth.source: realsense` opens the camera with pyrealsense2 in the same process, when ScaleBridge runs on PC2. The camera pose in the torso is the URDF `d435_joint` (`d435_in_torso()`), the map restarts at calibration. Use `localization=legged_estimator_lidar_depth` for both sensors. The wire decoding was checked in loopback against MagicLoco's own packer and the client handlers on recorded data; the camera itself was not run on hardware.

**MuJoCo.** `simulator.config.lidar_odom={hz: 10, sigma: 0.01, xy_drift: 0.005, z_drift: 0.005, delay: 0.0, latency: 0.0, jump: {time: 8.0, dz: 0.5}}` emulates FAST-LIO (the LiDAR pose in a rotated and shifted odom frame, with noise and a linear drift) through the same fusion code as the robot. `simulator.config.depth_ground={hz: 10, width: 320, height: 240, fovy: 58, noise_frac: 0.01}` mounts the D435i on the torso as in the URDF and renders depth (needs `MUJOCO_GL=egl` without a display). The stock MuJoCo feet are four 5 mm spheres per foot, which catch on stair edges and make even ground-truth tracking fail; for stairs replace them by two flat discs per foot (radius 0.033 m at x = 0.1089 m and radius 0.030 m at x = -0.0355 m in the ankle roll link, half height 0.0075 m, z = -0.0275 m), which is the foot model of the policy's final training.

Policy in the loop, held-out terrain clips (0.10 to 0.26 m steps, up to 4 m of climb, global tracking, lookahead `future_idx=[0,1,2,3,4,16]`), pelvis world error below 0.5 m over the whole clip (below 1 m in brackets), ten clips; clip L7_r0_e1719 fails even with ground truth, so 9 is the ceiling:

| Position source | Clips passed |
| --- | --- |
| ground truth | 9 (9) |
| default estimator | 0 (0) |
| `legged_estimator_stairs` | 3 (6) |
| `legged_estimator_depth` | 5 (8) |
| `legged_estimator_lidar`, ideal LiDAR (1 cm noise, no drift) | 9 (9) |
| `legged_estimator_lidar`, 0.5 cm/s drift in x, y, z | 8 (8) |
| `legged_estimator_lidar`, 2 cm/s drift | 2 (5) |
| `legged_estimator_depth` plus LiDAR with 0.5 cm/s drift | 8 (8) |

The LiDAR fusion must be combined with a variant that has no flat-ground height; with the default estimator it passed 0 of 10. Not modelled: real FAST-LIO behaviour on stairs, real depth noise and holes. Without a LiDAR the cleanest route on stairs is `legged_estimator_depth`; with one, fuse it.
