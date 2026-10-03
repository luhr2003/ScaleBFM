# ScaleBFM 真机部署手册（默认定位器：legged_estimator_lidar）

适用：Unitree G1（29 自由度），ScaleBFM 地形微调检验点，全局跟踪（`env.config.reference_forcing=False`），骨盆位置由 `legged_estimator_lidar` 给出。这个定位器把腿的状态估计器和 Mid-360 上 FAST-LIO 的位姿融合起来，平地和台阶都能用。

> **状态说明**：下面的软件部分在 MuJoCo 里验证过，包括模拟的 FAST-LIO。**真机上的 FAST-LIO 桥、位姿线、D435 取图、`R2` 标定流程都没有在硬件上跑过**，所以第 7 节有一份“第一次上机”的分级清单，请按顺序做，不要跳级。
> 算法、数据和测试方法见 [legged_estimator.md](legged_estimator.md) 第 10 节和 [terrain_localization_manual_zh.md](terrain_localization_manual_zh.md)。通用的 ScaleBridge 安装、低层控制器编译见 [README](../README.md) 第 1 到 5 节。

- [1. 系统怎么连](#1-系统怎么连)
- [2. 准备：检验点和环境](#2-准备检验点和环境)
- [3. PC2：FAST-LIO 和位姿桥](#3-pc2fast-lio-和位姿桥)
- [4. 启动顺序](#4-启动顺序)
- [5. 标定和开始](#5-标定和开始)
- [6. 运行中看什么](#6-运行中看什么)
- [7. 第一次上机的分级清单](#7-第一次上机的分级清单)
- [8. 台阶任务的注意事项](#8-台阶任务的注意事项)
- [9. 没有雷达怎么办](#9-没有雷达怎么办)
- [10. 排错](#10-排错)

## 1. 系统怎么连

```
                       ┌────────────────────────── 机器人 ──────────────────────────┐
                       │ PC1 .161  运动控制（不要碰）                                   │
 工作站或 PC2          │ PC2 .164  Jetson：FAST-LIO、fastlio_bridge、（可选）D435、      │
 ScaleBridge run.py ◀──┤           ScaleBridge 也可以跑在这里                            │
  ▲   ▲                │ Mid-360   .120  点云和 IMU 已经发布在 DDS 上                    │
  │   └ LCM robot_state_data（200 Hz，关节、力矩、IMU）                                 │
  └──── ZeroMQ 位姿线 tcp://PC2:5606（FAST-LIO 位姿，约 10 Hz）                         │
                       └──────────────────────────────────────────────────────────────┘
```

- **ScaleBridge 在哪里跑**：工作站（x86，用 `linux` 版 TensorRT 文件）或 PC2 本机（aarch64，`aarch64` 版，要在板子上编译）。
- **低层控制器**（`g1_29dof_controller`）已经通过 LCM 发布 `robot_state_data`，ScaleBridge 的估计器直接订阅它，不需要额外的传感器接线。
- **雷达位姿**从 PC2 上的 FAST-LIO 来，经 `fastlio_bridge.py` 变成 ZeroMQ 位姿线。**这条线是默认定位器的必要条件**：启动时如果 30 秒内收不到位姿，ScaleBridge 会报错退出，而不是悄悄退化。
- 工作站网卡配静态 IP（例如 `192.168.123.222/24`），和 `.161`、`.164`、`.120` 不冲突；先 `ping 192.168.123.164` 和 `ping 192.168.123.120`。

## 2. 准备：检验点和环境

### 2.1 选检验点

`ScaleTrack/checkpoints/README.md` 推荐地形用 **`ft_v2_soupA_it26000-27000.pt`**（26000 到 27000 迭代的权重平均）：留出地形 mode 7 全局成功率 83.8%（预训练 16.3%），平地 26 种配置没有退化。`ft_v2_it25200.pt` 是 MagicSim 闭环验证过的单个检验点。`ft_v3_it26800.pt`（已入库 `ScaleTrack/checkpoints/`）门限更高（留出地形 mode 7 全局 87.1%，平地无退化），但还没做 MagicSim 闭环验证，我也没有在 MuJoCo 里测过它（测过的是 soupA 和 `ft_v2_it26400`）。

> 更新（2026-10-03）：`ScaleTrack/checkpoints/README.md` 现在推荐最终检验点 **`ft_v4_soupV4a_it28300-28600.pt`**（留出地形 mode 7 全局成功率 87.7%，三个评测种子均值；平地全测试集没有退化；深蹲比预训练模型更深）。它还没有在 MuJoCo 和实机上测过，本手册里的数字都是 soupA 的。想稳妥就继续用 soupA；换成 V4a 时按 2.2 节重新编译，先在 MuJoCo 里把台阶和平地各走一遍。

### 2.2 编译成 TensorRT

TensorRT 文件和平台、显卡绑定，不能互相拷贝。Python 3.11，`torch==2.8.0`，`torch-tensorrt==2.8.0`。

```bash
cd ScaleTrack
python scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer_onboard.py \
  --checkpoint /path/to/model_soupA.pt --mode_table /path/to/mode_table.pt \
  --metadata /path/to/model_soupA_tensorrt_metadata.json --xml_path /path/to/g1_29dof.xml
```

放到 `ScaleBridge/scalebridge/data/model/g1_29dof/<名字>/{linux,aarch64}/`，同目录要有同名前缀的 `_tensorrt_metadata.json`。编译约 12 分钟。`mode_table.pt` 和 metadata 来自 Isaac 里的导出脚本，同一网络结构通用。Jetson 上要用板子本机重新编译（README 第 5 节）。这些文件不要提交。

### 2.3 安装

```bash
cd ScaleBridge && pip install -e .     # 含 pin（pinocchio）、pyzmq
sudo apt install liblcm-dev
```

## 3. PC2：FAST-LIO 和位姿桥

这部分来自 MagicLoco 的 `sim2real/docs/24_terrain_odometry_real_runbook.md`（FAST-LIO 在 PC2 上怎么上），请以那份文档为准，这里只列要点。

- Mid-360 点云和 IMU 已经在 DDS 上：`rt/utlidar/cloud_livox_mid360`（10 Hz）、`rt/utlidar/imu_livox_mid360`（200 Hz）。**雷达是倒装的**（俯仰 −2.3°），FAST-LIO 配置里要对应。
- FAST-LIO 配置（Unitree 的话题）：

```yaml
common:
  lid_topic: /utlidar/cloud_livox_mid360
  imu_topic: /utlidar/imu_livox_mid360
preprocess: { lidar_type: 1, scan_line: 4, blind: 0.5 }
mapping:
  extrinsic_T: [-0.011, -0.02329, 0.04412]
  extrinsic_R: [1,0,0, 0,1,0, 0,0,1]
```

- 位姿桥（只有这个文件碰 ROS），在 PC2 上：

```bash
source /opt/ros/noetic/setup.bash          # 加上 FAST-LIO 的工作空间
python3 sim2real/perception/fastlio_bridge.py --mode odom --odom-topic /Odometry --endpoint tcp://*:5606
```

  位姿线格式：`pos`（雷达本体在 FAST-LIO 里程计系中的位置）、`quat`（wxyz）、`body`（0 `mid360_link`，1 `torso_link`，2 `pelvis`）。ScaleBridge 用 `body` 选对应的 URDF 坐标系，所以桥用默认的 odom 模式（`body=0`）即可。
- **验收**：机器人静止站着 5 分钟，FAST-LIO 的 z 漂移绝对值要小于 2 cm；超过 ±5 cm 先修里程计再上策略。这个数直接决定台阶上的精度（融合后的误差就是雷达的漂移）。
- **已知的小误差来源（没有补偿）**：FAST-LIO 在 `extrinsic_T` 里把位姿定义在雷达的 IMU 系，它和 URDF 里的 `mid360_link` 相差约 5 cm（`(0.011, 0.023, −0.044) m`）。ScaleBridge 把位姿当成 `mid360_link` 的位置，这个固定偏移会随躯干姿态在位置上引入最多几厘米的误差。几厘米的偏差策略可以接受；要消除的话在桥里改发 `body=torso_link`（桥的 `--mode tf`，先把偏移用 TF 修正）。
- 在 PC2 或工作站上都能看位姿：MagicLoco 的 `tools/dds_probe.py`、`tools/lidar_probe.py` 看雷达点云和 IMU 是否活着（点云 ≥8 Hz、IMU ≥150 Hz 为正常）。

## 4. 启动顺序

1. **机器人悬挂**，周围清场。低层控制器会自动把机器人移到默认姿态。
2. **低层控制器**（PC1 通过 PC2 或工作站的 LCM）：

```bash
cd third_party/unitree_sdk2/build/bin
./g1_29dof_controller NETWORK_INTERFACE
```

3. **PC2 上 FAST-LIO，再起位姿桥**（第 3 节）。确认位姿线有数据：在工作站上能收到 `tcp://192.168.123.164:5606`。
4. （可选）D435：PC2 上 `python3 rs_probe.py --serve`（见第 8 节）。
5. **ScaleBridge**（工作站或 PC2）：

```bash
python scalebridge/run.py \
  agent.config.control_mode=7 \
  agent.config.checkpoint=/path/to/model_soupA_tensorrt.pt \
  asset=g1_29dof env=motion_tracking \
  env.config.reference_forcing=False \
  env.config.future_idx=[0,1,2,3,4,16] \
  env.config.motion_path=/path/to/motion.npz \
  simulator=real_world
```

- 默认定位器就是 `legged_estimator_lidar`，不用写 `localization=`。
- 位姿线端点默认 `tcp://192.168.123.164:5606`；改的话用 `localization.lidar.endpoint=tcp://...`。
- **`future_idx=[0,1,2,3,4,16]`**：地形上前瞻 K=16 比默认 K=5 好约 6 个百分点。平地动作可以保持默认。不要用 `-1`。
- 启动日志里应该先看到 `Waiting up to 30 s for the first LiDAR odometry pose`，然后 `Localization module client connected: robot_state_data over LCM`。收不到位姿会报 “No LiDAR odometry pose arrived …” 并退出：先检查第 3 节，不要为了启动去关掉 `required`。

## 5. 标定和开始

1. **把机器人放低，直到双脚平放在地上、站稳**。台阶任务要站在**一块平地**上标定（高度图和对齐都以这里为起点）。
2. **按 `R2` 标定一次。机器人必须站着不动**：估计器从当前姿态重启，当前骨盆 xy 成为原点；**下一帧雷达位姿定义里程计系的原点和朝向**（日志里出现 `Aligned to the estimator frame, yaw offset ... deg`）；参考动作对齐到同一个原点和 IMU 航向。如果标定时机器人在动，朝向对齐会有误差，位置误差会随离原点的距离按比例增大，这时重新按 `R2`。
3. **再按 `R2` 开始。紧急停止 `L2 + B`。**

没有重新对齐的情况：任何时候再按一次标定键，估计器和雷达对齐都会重新开始。

## 6. 运行中看什么

日志（`run.log` 在 hydra 输出目录里，`LOGURU_LEVEL=DEBUG` 在终端也打印）：

| 日志 | 含义和处理 |
|---|---|
| `[LeggedEstimator] Root position [x, y, z] ..., contact probabilities [...]` | 每秒一行（DEBUG）。平地上站立高度约 0.75 到 0.79 m；双脚支撑接触概率约 0.6 到 0.8，单脚约 1，摆动脚 0 |
| `[LidarOdometry] Aligned ... yaw offset` | 标定后雷达对齐成功；yaw offset 是 FAST-LIO 里程计系和 IMU 航向的差，任意值都正常 |
| `LiDAR odometry pose is N s old: ... drifting on the legs only` | **位姿线断了**。腿估计单独没有绝对高度，几秒到十几秒后会明显漂；立刻停（`L2 + B`），先检查 FAST-LIO 和桥 |
| `Pose jumped away from the estimate and stayed there; alignment shifted.` | FAST-LIO 重定位或跳变，估计器没有跳，但这说明里程计不可信，检查雷达环境（空旷、玻璃、剧烈运动） |

雷达健康的标志：位姿线约 10 Hz 稳定；`pose is N s old` 不出现；站着不动时估计 z 基本不变。

## 7. 第一次上机的分级清单

每一级通过再下一级。任何一级异常就停。

1. **只看雷达**（不开机器人）：`lidar_probe.py` 点云 ≥8 Hz、IMU ≥150 Hz；室内 z 直方图有地板和天花板两个峰。
2. **FAST-LIO 静态验收**：站着不动 5 分钟，z 漂移 |bias| < 2 cm。
3. **位姿线**：工作站能收到 `tcp://PC2:5606`，频率约 10 Hz，位置在站着不动时稳定。
4. **悬挂 + 站立动作**（例如 `scalebridge/data/motion/g1_29dof/squat.npz`）：看估计 z 合理，不发散。
5. **落地 + 原地动作**：标定后做原地动作，估计位置不应该漂。
6. **短距离行走**（平地，几米）：对照局部跟踪（`reference_forcing=True`）同一动作。
7. **走回起点**：估计位置回到起点附近（误差应在几厘米到十几厘米内），这检查 yaw 对齐和漂移。
8. **台阶**：先 0.05 到 0.10 m 的低台阶，再逐级加高；参考轨迹的 z 要和真实台阶一致（规划器要用同一个原点和朝向）。

## 8. 台阶任务的注意事项

- **参考的坐标系**：参考动作在标定时对齐到估计器的原点和朝向，参考的 z 是相对**标定点所在的地面**的。标定点不在平地上，z 会整体偏。
- **前瞻 K=16**（第 4 节）。
- **能力边界**：留出地形 mode 7 全局成功率 83.8%（soupA）；0.25 m 以上的台阶仍然是弱项，下楼梯比上楼梯难。MagicSim 闭环在 0.26 m 上台阶时有较高的摔倒率，别在没有保护的情况下做高台阶。
- **可选的 D435**：`localization=legged_estimator_lidar_depth`。PC2 上跑 `python3 rs_probe.py --serve`（MagicLoco `sim2real/perception/tools/`，发深度到 5609 端口），ScaleBridge 直接接收；ScaleBridge 在 PC2 上跑时可用 `localization.depth.source=realsense` 本机读相机。D435 在 MuJoCo 里额外降低了腿估计的高度误差，但**有雷达时雷达已经提供了绝对高度**，D435 的收益较小，真机首次上机建议只用雷达。
- **同一时刻只有一个进程能打开 D435**；USB 要是 3.x。

## 9. 没有雷达怎么办

- **平地任务**：`localization=legged_estimator`（原来的默认），不需要雷达。不要用于台阶。
- **台阶但没有雷达**：`localization=legged_estimator_depth`（D435）或 `legged_estimator_stairs`（只靠腿）。MuJoCo 里分别 5/10 和 3/10 通过，远低于有雷达的 8/10 到 9/10，只能做短距离、低台阶，并且要有保护。
- **想用 Vive**：`localization=vive_tracker`，按 README 的 Vive 一节。

## 10. 排错

| 现象 | 处理 |
|---|---|
| 启动时报 `No LiDAR odometry pose arrived` | FAST-LIO 或桥没起来，或端点不对；按第 3 节检查，`localization.lidar.endpoint` 要指向 PC2 |
| 标定后机器人往一边走偏，距离越远偏得越多 | 标定时在动，或 FAST-LIO 初始化朝向有问题；站稳后重新 `R2`；走回起点检查 |
| 站着不动估计 z 或位置慢慢变 | 位姿线断了（看 `pose is N s old`），或 FAST-LIO 在静止时就漂；回到第 7 节第 2 步 |
| 估计突然跳一下 | 日志看有没有 `Pose jumped`；FAST-LIO 在空旷或镜面环境会退化；调 `localization.lidar.gate` 和 `realign_after` |
| 平地上走得比以前差 | 雷达位姿有延迟误差：`localization.lidar.latency` 要和实测一致（默认 0.1 s）；暂时可换回 `localization=legged_estimator` 对比 |
| 日志里 `Waiting for the localization module client` 一直卡住 | 低层控制器没起或 LCM 网卡不对（`robot_state_data` 没收到） |
| 机器人在台阶上摔 | 先用真值位置在 MuJoCo 里验证检验点和动作；检查 `future_idx` 的最后一个是 16；检查标定点是否平地 |
| TensorRT 加载失败 | 文件是别的平台或显卡编译的，在本机重新编译（第 2.2 节） |
