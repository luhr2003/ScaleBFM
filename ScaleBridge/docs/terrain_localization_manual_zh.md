# ScaleBFM 地形能力与定位器手册

这份手册回答两件事：ScaleBFM 现在有没有地形能力、能到什么程度；以及全局跟踪需要的骨盆位置（定位器）是怎么来的、台阶上为什么要换方案。定位器的算法细节和英文数据表在 [legged_estimator.md](legged_estimator.md)（第 10 节是台阶部分），本手册是面向使用的中文总览。

- [0. 结论](#0-结论)
- [1. ScaleBFM 的地形能力](#1-scalebfm-的地形能力)
- [2. 整体链路：谁给参考，谁来跟踪，谁给位置](#2-整体链路)
- [3. 定位器是怎么搞的](#3-定位器是怎么搞的)
- [4. 怎么用](#4-怎么用)
- [5. 在 MuJoCo 里测台阶](#5-在-mujoco-里测台阶)
- [6. 测试结果](#6-测试结果)
- [7. 限制与没验证的部分](#7-限制与没验证的部分)
- [8. 排错](#8-排错)
- [9. 文件地图](#9-文件地图)

## 0. 结论

1. **ScaleBFM 有地形能力。** 在预训练模型 `model_22200` 上用台阶、斜坡、方块、粗糙地形的参考轨迹做了微调（`ft_v2`、`ft_v3`），平地跟踪没有退化。留出地形（训练时没见过的布局）上的全身模式（mode 7）全局跟踪成功率：预训练 16.3%，微调后 81% 到 87%。
2. **它是“盲”的**：策略没有地形观测，不看高度图。它跟的是参考轨迹，而参考轨迹的 z 已经带着地形起伏（来自 MagicLoco 的地形规划器 pi_L）。所以地形信息是通过参考进来的，不是策略自己感知的。
3. **全局跟踪才可靠，而且它要骨盆的绝对 x、y、z。** 这就是定位器的作用。局部跟踪（参考强制，没有位置反馈）在地形上机器人会站稳但漂移。
4. **平地用的默认定位器在台阶上不行**：它假设脚踩的永远是 z=0 的平地，台阶上整段爬升都会丢掉。MuJoCo 里 10 段台阶 clip 通过 0/10，而用真值位置是 9/10。
5. **默认定位器现在是 `legged_estimator_lidar`**（腿估计 + Mid-360 的 FAST-LIO 位姿），平地和台阶都用它；MuJoCo 里台阶 8/10 到 9/10 通过，平地也比原来的纯腿估计更准。其余是可选配置：只靠腿的台阶配置（3/10）、加 D435 深度（5/10）。没有雷达的平地任务用 `localization=legged_estimator`。真机部署步骤见 [real_robot_deployment_zh.md](real_robot_deployment_zh.md)。
6. **全部是 MuJoCo 里的结果，真机没有测过。** D435 取图和 FAST-LIO 桥只做了通信回环和录制数据的测试。

## 1. ScaleBFM 的地形能力

### 1.1 训练做了什么

- 起点：预训练 `model_22200`（humanoid_transformer_m），网络结构和部署接口都不变，没有新增输入。
- 数据：平地的排练数据（BONES、LAFAN 等）加地形 clip。地形 clip 来自 MagicLoco pi_L v4 地形策略当规划器在多种地形上走出来的轨迹，共约 8.3 万条、455 小时，布局 0 到 5 训练，6、7 留出。
- 脚模型：和 MagicSim 的 `g1_new.usd` 一致，每只脚两个平圆盘碰撞体（不是原来的 7 根细胶囊）。这点很重要，圆盘脚和胶囊脚在台阶上差别可达 10 个百分点。
- 防退化：平地样本上对 `model_22200` 做 KL 锚定；地形课程逐步放开台阶高度；高台阶 clip 加权；一部分 episode 用参考强制（让局部跟踪也有地形经验）。
- 检验点在 `ScaleTrack/checkpoints/`（Git LFS），`checkpoints/README.md` 有逐个检验点的门限表。`ft_v2_it25200` 是经过闭环验证的文件，`ft_v3_it26800` 和权重平均的 `soupA` 是后来的更强版本。
- 更新（2026-10-03）：最终检验点是 `ft_v4_soupV4a_it28300-28600.pt`（留出地形 mode 7 全局 87.7%、mode 4 86.1%，三个评测种子均值；平地全测试集没有退化），成绩见 `checkpoints/README.md`。本手册里所有 MuJoCo 结果都是用 soupA 和 `ft_v2_it26400` 得到的，`soupV4a` 还没有在这里测过。

### 1.2 能到什么程度

留出地形（1500 条 20 s 的 clip，MagicSim 的脚模型），成功率 = 任何被激活的链接与参考的世界系位置误差从未超过 0.5 m：

| 检验点 | mode 7 全局 | mode 4 全局 | 局部跟踪 SuccL（mode 7） |
|---|---|---|---|
| 预训练 `model_22200` | 16.3% | — | 17.5% |
| `ft_v2_it25200` | 80.1% | 80.5% | 58.8% |
| `soupA`（26000 到 27000 权重平均） | 83.8% | 82.3% | 59.6% |
| `ft_v3_it26800` | 87.1% | 84.7% | 62.9% |

- 台阶高度越高越难：25200 上大约 0.15 m 以下 88%，0.15 到 0.20 m 75%，0.20 m 以上 58% 到 60%；下楼梯比上楼梯难。
- 平地：26 种“模式×跟踪方式”的完整门限没有退化（BONES 约 ±0.3 pp 的噪声内，Ours 约 ±1.5 pp）。
- SuccL 是局部跟踪的指标（相对根的链接误差不超过 0.5 m）。局部跟踪没有位置反馈，所以世界系的 Succ 在地形上没意义。

### 1.3 重要的部署细节：前瞻帧 K

策略的任务观测里有 6 个未来帧偏移 `[0,1,2,3,4,K]`。训练时最后一个 K 在 5 到 32 之间随机。标准导出和 ScaleBridge 默认的 `future_idx=[0,1,2,3,4,5]` 是 K=5，在台阶上比训练分布差。留出地形上的扫描：

| K | 5 | 10 | 16 | 24 | 32 | 训练分布（随机） |
|---|---|---|---|---|---|---|
| mode 7 全局 | 73.9% | 80.0% | **82.5%** | 81.1% | 80.4% | 80.1% |

所以台阶上用 `env.config.future_idx=[0,1,2,3,4,16]`。平地上 K=5 没问题。注意不要用 `-1`：训练配置里的 `-1` 是“随机前瞻”的占位，在 ScaleBridge 里字面的 `-1` 是“上一帧”。

## 2. 整体链路

```
地形规划器 (MagicLoco pi_L v4，看得见地形)
        │  参考轨迹：骨盆、躯干、腿、手的位置姿态，z 已带地形起伏
        ▼
ScaleBFM 策略 (盲，跟踪参考)   ◀── 观测：自身本体感觉 + 参考相对机器人的偏差
        │  关节目标
        ▼
ScaleBridge (50 Hz 推理，PD 控制)  ──▶  机器人 / MuJoCo
        ▲
        └── 定位器：给出骨盆在参考坐标系里的 x、y、z
```

- **全局跟踪**（`env.config.reference_forcing=False`）：策略观测里有“参考位置 − 当前骨盆位置”，所以需要骨盆的绝对位置。位置错了，策略就以为自己偏离了参考，会做出反应。
- **局部跟踪**（`reference_forcing=True`）：用参考位置代替测量位置，不需要定位器，但没有位置反馈，漂移会累积；地形上 SuccL 约 60%。
- **对 z 误差很敏感**：同一段台阶，z 偏 0.15 m 通过率就掉到一半（见 6.4）。慢漂移影响小，突然跳变影响大。

## 3. 定位器是怎么搞的

### 3.1 默认估计器（平地）

`scalebridge/utils/legged_estimator` 是 `legged_control2`（BeyondMimic 部署用的）状态估计器的无 ROS 移植：

- 输入：关节编码器、关节力矩（`tau_est`）、骨盆 IMU，来自低层控制器的 LCM `robot_state_data`（200 Hz）。
- 广义动量观测器：从关节力矩和动力学算出外力，再解出两只脚的接触力和压力中心（ZMP）。
- 接触概率 = 力的 sigmoid（中点 150 N）× ZMP 是否在脚底窗口（x ±0.08 m，y ±0.025 m）内。
- 15 维卡尔曼滤波：状态是骨盆位置、速度、两只脚位置和加速度计偏置。IMU 加速度做预测，接触脚做更新（“骨盆 − 脚 = 运动学算出的相对位置”，外加“脚高度 = 0”）。
- 只估计平移；姿态直接用 IMU。是里程计，会随滑移慢慢漂。

### 3.2 为什么台阶上不行，也不是随便改一下就行

- **“脚高度 = 0”这一条**把脚钉在平地上。台阶上脚踩的是 0.1 到 0.26 m 的台面，骨盆高度就一直被拉回起点的地面高度。
- **去掉它以后，整体高度没有绝对参考**：骨盆和脚的共同高度只由 IMU 积分维持，0.01 m/s² 的重力残差就让它每秒漂几毫米，丢接触的时刻还会多漂。
- **接触检测也要改**：下楼梯时脚跟或脚尖悬在台阶边上，压力中心会超出 ±0.08 m，接触被一票否决，估计就漏掉下降。真实脚底范围约 −0.108 到 +0.10 m。

### 3.3 台阶上的四种配置

都用 `localization=<名字>` 选择；不写时用默认的 `legged_estimator_lidar`。

| 配置 | 做了什么 | 需要什么 | MuJoCo 台阶 clip 通过（10 段） |
|---|---|---|---|
| `legged_estimator`（原来的默认，平地用） | 平地假设 | 无 | 0/10 |
| `legged_estimator_stairs` | 去掉“脚高度=0”，ZMP 窗口放宽到 0.10×0.04 m，接触阈值降到 100 N，高度靠腿的运动学跟随支撑脚 | 无 | 3/10 |
| `legged_estimator_depth` | 在 stairs 的基础上，用 D435 的滚动高度图查每只落脚点的地面高度，替换“=0” | D435i 深度来源 | 5/10 |
| `legged_estimator_lidar`（**现在的默认**） | 在 stairs 的基础上，融合 Mid-360 上 FAST-LIO 的位姿作为 10 Hz 的绝对位置测量 | MagicLoco `fastlio_bridge.py` | 8/10（0.5 cm/s 漂移），9/10（无漂移） |
| `legged_estimator_lidar_depth` | 两者都用 | 两者 | 8/10 |

真值位置是 9/10（有一段连真值也过不了，所以 9 是上限）。

**雷达融合**（`LidarOdometryFusion`）：

- FAST-LIO 的里程计系原点和朝向是任意的。标定时（按 R2）用第一帧位姿对齐：yaw 差取雷达本体朝向与 IMU 航向之差，位置对当前估计。之后的位姿都过这一个固定变换。
- 骨盆位置 = 雷达位置 − 骨盆到 `mid360_link` 的杠杆臂；杠杆臂用估计器自己的运动学算（URDF 里有 `mid360_link`），腰关节也算进去了。
- 延迟补偿：位姿晚到 `latency`（默认 0.1 s，要在机器人上测），就加上估计器在这段时间内自己走的距离。
- 异常门限：位姿离估计超过 `gate`（0.3 m）就忽略；持续 `realign_after`（1 s）就接受并只平移对齐，所以 FAST-LIO 重定位不会让估计跳。
- 融合后误差等于雷达自己的漂移，腿估计既修不了它也不会恶化它。FAST-LIO 要健康：MagicLoco 要求静止 5 分钟 z 漂移小于 2 cm。

**D435 融合**（`DepthGroundHeight`）：

- 地图用估计器自己的位姿建，所以它提供的是“几秒内的一致性”，不是绝对参考。思路和 MagicLoco 的 FK z 锚定、整图 z 偏置一样，只是写成了卡尔曼滤波的测量。
- 地图每个格子的更新增益随观测次数衰减（`memory_frames`）。没有这个的话，地图会跟着估计的漂移走再反馈回去：站着不动 6 秒 z 漂了 0.4 m，加了以后消失。
- 没看见的格子没有测量；标定时在脚下 0.5 m 圆内按“平地”播种。

**VideoMimic 和 MagicLoco 的做法对照**：VideoMimic 只用 FAST-LIO 对齐点云和算“躯干高于地面多少”，两者在同一个里程计系，z 的整体漂移互相抵消，所以不需要绝对 z；MagicLoco 的 z 来自腿运动学锚定，再用 D435 地图修整图偏置。我们的情况不一样：全局跟踪要绝对 z，所以把雷达位姿当测量融进去。

## 4. 怎么用

### 4.1 检验点编译成 TensorRT

策略要导出成 TensorRT 才能被 ScaleBridge 加载（`scalebridge/agent/bfm_agent.py`）。TensorRT 文件和平台、显卡绑定。

环境：Python 3.11，`torch==2.8.0`、`torch-tensorrt==2.8.0`。我用的是 `~/scalebfm_ws/envs/deploy`（和训练环境 `envs/scaletrack` 是分开的，别往训练环境里装包）。

```bash
cd ScaleTrack
python scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer_onboard.py \
  --checkpoint <检验点>.pt --mode_table <mode_table.pt> \
  --metadata <检验点>_tensorrt_metadata.json --xml_path <g1_29dof.xml>
```

- 输出 `<检验点>_tensorrt.pt`，放到 `ScaleBridge/scalebridge/data/model/g1_29dof/<名字>/linux/`，同目录放同名前缀的 `_tensorrt_metadata.json`。
- `mode_table.pt` 和 metadata 来自 Isaac 里的导出脚本（`play_export_check_humanoid_transformer.py`）；同一个模型结构（humanoid_transformer_m）的这两个文件对不同检验点是通用的。
- 编译约 12 分钟，需要几 GB 显存。这些大文件不要提交（`.gitignore` 已忽略 `*.pt`、`*.json`）。

### 4.2 MuJoCo 里先测

平地、真值位置（基线）：

```bash
python scalebridge/run.py agent.config.control_mode=7 \
  agent.config.checkpoint=/path/to/model_tensorrt.pt \
  asset=g1_29dof env=motion_tracking env.config.reference_forcing=False \
  env.config.motion_path=/path/to/motion.npz simulator=mujoco_simulator
```

平地、腿估计器（默认）：加 `simulator.config.estimate_root_pos=True`，日志每秒一行 `Root estimate error`。

台阶：参考轨迹的 z 要和 MuJoCo 里的地形对上，MuJoCo 的 xml 要有地形，脚要换成圆盘脚（见第 5 节），用 RSI 从参考起点起步：

```bash
python scalebridge/run.py ... env.config.future_idx=[0,1,2,3,4,16] +env.config.rsi=True \
  asset.xml_path=/path/to/terrain.xml \
  simulator.config.estimate_root_pos=True localization=legged_estimator_stairs
```

仿真传感器（都在 `simulator.config` 下）：

- 模拟 FAST-LIO：`lidar_odom={hz: 10, sigma: 0.01, xy_drift: 0.005, z_drift: 0.005, delay: 0.1, latency: 0.1, jump: {time: 8.0, dz: 0.5}}`，估计器配置用 `localization=legged_estimator_stairs`（雷达融合要配无平地高度的配置）。
- 模拟 D435：`depth_ground={hz: 10, width: 320, height: 240, fovy: 58, noise_frac: 0.01}`，估计器配置用 `localization=legged_estimator_depth`；没有显示器时要 `MUJOCO_GL=egl`。

### 4.3 真机

先按 README 起低层控制器。台阶上先做悬挂测试，再落地。

```bash
python scalebridge/run.py agent.config.control_mode=7 \
  agent.config.checkpoint=/path/to/model_tensorrt.pt \
  asset=g1_29dof env=motion_tracking env.config.reference_forcing=False \
  env.config.motion_path=/path/to/motion.npz env.config.future_idx=[0,1,2,3,4,16] \
  simulator=real_world
```

默认定位器是 `legged_estimator_lidar`，不用写 `localization=`。完整的真机步骤、分级清单和排错见 [real_robot_deployment_zh.md](real_robot_deployment_zh.md)。

- 雷达：PC2 上跑 FAST-LIO 和 MagicLoco 的 `sim2real/perception/fastlio_bridge.py`（发位姿到 `tcp://*:5606`），在 `legged_estimator_lidar.yaml` 里改 `endpoint`、`latency`。
- D435：在 PC2 上跑 `python3 rs_probe.py --serve`（MagicLoco `sim2real/perception/tools`，发深度到 5609 端口）；或者 ScaleBridge 直接跑在 PC2 上时把 `depth.source` 改成 `realsense`。同一时刻只有一个进程能打开相机。
- 机器人站稳在平地上，按 R2 标定（估计器重启、雷达重新对齐、高度图重新播种），再按 R2 开始。紧急停止 `L2 + B`。
- 想再加 D435：`localization=legged_estimator_lidar_depth`。没有雷达时平地用 `localization=legged_estimator`，台阶用 `legged_estimator_depth`（精度低很多，见 7）。

### 4.4 配置选择

| 场景 | 配置 |
|---|---|
| 默认（平地和台阶，有 Mid-360 + FAST-LIO） | `legged_estimator_lidar`（有 D435 想加用 `legged_estimator_lidar_depth`） |
| 平地、跑步、一般动作，没有雷达 | `legged_estimator`（原来的默认） |
| 台阶，只有 D435 | `legged_estimator_depth` |
| 台阶，什么都没有 | `legged_estimator_stairs`（只试一两段，z 会漂，平地快跑会摔） |
| 要验证策略本身 | 仿真里用真值位置作对照 |

## 5. 在 MuJoCo 里测台阶

我用的测试台在 `~/scalebfm_ws/tools/stairs_mujoco/`（不在仓库里；脚本里的路径指向写它们的临时目录，用之前要替换）。方法和踩过的坑：

- **地形**：用留出布局 7 的高度图（`motions/terrain_raw/layouts/layout_7/heightmap.npz`，5 cm 分辨率）裁成 MuJoCo 的 hfield。hfield 的行方向和 y 相反，需要翻转，验证过与高度图的最大差为 0.12 m（台阶边缘的插值）。
- **参考**：用地形 clip 的 npz，把 z 平移到“起点的地面高度为 0”，地形同样平移。起点用 RSI（`+env.config.rsi=True`），因为 clip 起点不在原点。
- **脚模型必须换**：ScaleBridge 自带的 MuJoCo xml 每只脚是 4 个半径 5 mm 的小球，会卡在台阶边上，连真值也 40 次全挂。换成 MagicSim 的圆盘脚（每只脚两个圆柱：前 r=0.033 @x=+0.1089、后 r=0.030 @x=−0.0355，半高 0.0075，z=−0.0275）后真值就通过了。
- **相机 fovy**：MuJoCo 相机默认垂直视场角是 45°，要设成 D435 的 58°，否则点云系统性偏低。
- **成功判据**：骨盆与参考的世界系 3D 误差在整段 clip 内始终小于 0.5 m（宽松版 1 m）。比论文的 Succ 宽松，只看骨盆。
- **时间单位**：低层步长 0.005 s，每个策略步 4 个低层步。
- **离线重放**：`record_run.py` 让策略吃真值（机器人不摔）并录下估计器所有输入，`replay*.py` 离线重放估计器，改参数几分钟一轮，比整段策略在环快得多，也不被摔倒干扰。

## 6. 测试结果

全部在 MuJoCo，检验点 `ft_v2_it26400`（TensorRT），mode 7，全局跟踪，10 段留出地形 clip（台阶 0.10 到 0.26 m，爬升或下降约 4 m，约 20 s）。

### 6.1 平地

| 动作 | 配置 | 路径误差 均值 / 最大 | 定位器 xy 误差 均值 / 最大 |
|---|---|---|---|
| walk 28 s | global + 定位器 | 0.083 / 0.146 m | 0.020 / 0.045 m |
| walk 28 s | global + 真值 | 0.069 / 0.136 m | — |
| walk 28 s | local | 0.294 / 0.662 m | — |
| jog 87 s，68 m | global + 定位器 | 0.191 / 0.627 m | 0.216 / 0.437 m |
| jog 87 s，68 m | global + 真值 | 0.123 / 0.410 m | — |
| jog 87 s，68 m | local | 0.677 / 1.186 m | — |

新默认 `legged_estimator_lidar`（模拟 FAST-LIO：1 cm 噪声，0.2 cm/s 漂移，0.1 s 延迟并补偿）在同样两段上，策略跟踪误差比原来的纯腿估计更好：

| 动作 | 估计器 xy 误差 均值 / 最大 | 策略跟踪误差 均值 / 最大 | 原来的纯腿估计（上表 global + 定位器） |
|---|---|---|---|
| walk 28 s | 0.053 / 0.118 m | 0.054 / 0.128 m | 0.083 / 0.146 m |
| jog 87 s | 0.107 / 0.332 m | 0.113 / 0.359 m | 0.191 / 0.627 m |

（上表前几行是原来的纯腿估计器 `localization=legged_estimator`。）local（没有位置反馈）路径误差是 global 的 4 到 5 倍，jog 末端偏约 0.9 m，但没有摔倒。

### 6.2 台阶：定位方案对比

| 位置来源 | 通过（误差始终小于 0.5 m） | 误差始终小于 1 m |
|---|---|---|
| 真值位置 | 9/10 | 9/10 |
| 默认估计器 | 0/10 | 0/10 |
| 腿估计（`legged_estimator_stairs`） | 3/10 | 6/10 |
| 腿估计 + D435 | 5/10 | 8/10 |
| 腿估计 + 模拟 Mid-360（无漂移） | 9/10 | 9/10 |
| 腿估计 + 模拟 Mid-360（0.5 cm/s 漂移） | 8/10 | 8/10 |
| 腿估计 + 模拟 Mid-360（2 cm/s 漂移） | 2/10 | 5/10 |
| 腿估计 + D435 + Mid-360（0.5 cm/s 漂移） | 8/10 | 8/10 |
| 默认估计器 + 雷达（带“脚高度=0”） | 0/10 | 0/10 |
| **新默认 `legged_estimator_lidar`**（模拟 FAST-LIO：1 cm 噪声、0.2 cm/s 漂移、0.1 s 延迟补偿），`ft_v2_it26400` | 9/10 | 9/10 |
| 新默认，`soupA`（手册推荐检验点） | 8/10 | 9/10 |
| 真值位置，`soupA` | 8/10 | 9/10 |
| 局部跟踪（参考强制） | 1/10 | — |

- 用真值位置时，`ft_v2_it26400` 前瞻 K=16 通过 9/10，K=5 通过 8/10；`soupA` 用真值位置是 8/10，新默认和它持平。
- 新默认下的估计器误差（9 段正常数据，20 s，估计 vs 真值，最大 / 均值）：`ft_v2_it26400` z 0.17 / 0.03 m，xy 0.15 / 0.04 m；`soupA` z 0.13 / 0.03 m，xy 0.13 / 0.04 m。
- 每组只跑一遍，单次通过数有 ±1 到 2 段的随机性，所以更可靠的是下面按估计器误差比较的表。

### 6.3 台阶：估计器自身误差（估计 vs 真值，9 段正常数据，20 s 均值）

| 条件 | z 最大 / 均值 | xy 最大 / 均值 |
|---|---|---|
| 台阶配置，只靠腿 | 0.65 / 0.25 m | 0.72 / 0.30 m |
| 雷达无延迟，0.5 cm/s 漂移 | 0.14 / 0.05 m | 0.13 / 0.05 m |
| 雷达延迟 0.1 s，不补偿 | 0.31 / 0.11 m | 0.51 / 0.20 m |
| 雷达延迟 0.1 s，补偿 | 0.23 / 0.05 m | 0.26 / 0.11 m |
| 雷达延迟 0.2 s，补偿 | 0.18 / 0.04 m | 0.39 / 0.18 m |
| 8 s 时里程计跳变 0.5 m（超门限） | 0.28 / 0.13 m | 0.16 / 0.06 m |
| 8 s 时里程计跳变 0.2 m（门限内） | 0.42 / 0.18 m | 0.20 / 0.05 m |
| 无漂移的理想雷达 | 0.12 / 0.02 m | 0.09 / 0.02 m |

（另外，离线录制的 8 段数据里，用 D435 把 z 最大误差从纯腿的 0.21 m 降到 0.16 m，平均绝对误差从 0.087 m 降到 0.052 m。）

### 6.4 策略对 z 误差的容忍度

在台阶上给策略的位置里人为加 z 误差：

| 给策略的位置 | 通过 |
|---|---|
| 真值 | 9/10 |
| xy 真值，z 固定偏移 0.15 m | 5/10 |
| xy 真值，z 漂移 2 cm/s 加 1 cm 噪声（20 s 漂 0.4 m） | 6/10 |
| xy 腿估计，z 取真值 | 5/10 |
| xy 用原估计器，z 直接取参考 | 6/10（宽松 9/10） |

最后一行说明：xy 腿估计本身在台阶上也有误差，这是 z 取真值也只有 5/10 的原因之一。

结论：z 误差需要控制在 0.1 m 以内才不明显影响策略；慢漂可以容忍，突变不行。

## 7. 限制与没验证的部分

- **全部是 MuJoCo 结果。** 真机上的 D435 取图、FAST-LIO 桥、`R2` 标定流程、锁竞争都没跑过。在线客户端里深度处理和 200 Hz 的状态更新共用一把锁，每帧几毫秒；真机上如果状态更新被拖慢，需要把深度反投影挪到锁外。
- **模拟的雷达是理想化的**：真值加慢漂、1 cm 白噪声、固定延迟、一次跳变。真实 FAST-LIO 在楼梯上的退化、多路径、点云稀疏都没有模拟；真实 D435 的深度噪声和空洞也没有（只加了 1% 的乘性噪声）。
- **`legged_estimator_stairs` 平地快跑会摔**（jog 测试中 pelvis 最低 0.06 m）。雷达和 D435 两个配置都以它为基础，所以只在台阶场景、且没有雷达时使用。
- **腿估计的高度没有绝对参考**，会随爬升慢慢漂（约爬升的 5% 到 10%）。只能做几段台阶，不能做长时间运行；有雷达时用雷达压住。
- **D435 的地图只提供短时一致性**，不能去除已经进入地图的误差。
- **前瞻 K=16 的结论来自 Isaac 的地形门限**，MuJoCo 里只在 9 段 clip 上看到方向一致，样本小。
- **闭环验收（MagicLoco 地形策略当规划器、ScaleBFM 当跟踪器，在 MagicSim 里走通）由另一个 session 在做**，这份手册里没有它的结果。
- **机载版本没编译**：TensorRT 文件和平台绑定，aarch64 的 Jetson 上要用 `play_export_check_humanoid_transformer_onboard.py` 在板子上重新编译，我只编译了 x86 工作站版。

## 8. 排错

| 现象 | 可能原因和处理 |
|---|---|
| 台阶上估计 z 一直是起点高度 | 用了默认估计器（带“脚高度=0”）；换 `legged_estimator_stairs` 或带雷达、D435 的配置 |
| 融合雷达后估计更差 | 配置还是默认的平地高度测量，和雷达冲突；换成无平地高度的配置 |
| 雷达融合后 yaw 对不上，位置在转圈 | 标定时机器人在动，或 IMU 航向与 FAST-LIO 初始朝向差得多；站稳后重新按 R2，日志里看 `Aligned ... yaw offset` |
| 估计偶尔突然跳 | 看日志有没有 `Pose jumped away from the estimate`；调 `gate` 和 `realign_after`，检查 FAST-LIO 本身是否重定位 |
| 站着不动 z 缓慢上升或下降 | 腿估计的整体漂移（IMU 重力残差）；用雷达，或带 D435（`memory_frames` 不要太小） |
| D435 融合后 z 先对后漂 | 地图正反馈，确认 `memory_frames`≥20；确认标定时脚下是平地 |
| 深度点云比地形低或高 | MuJoCo 里 fovy 没设成 58°；真机上内参要用相机报告的（线上帧头里有） |
| 地图高度偏差随距离变大 | 相机俯仰或外参不对，核对 URDF 的 `d435_joint`（47.6° 下俯） |
| 台阶上机器人直接摔 | 先用真值位置验证策略和场景本身；MuJoCo 里确认脚是圆盘脚、`future_idx` 的最后一个是 16 |
| TensorRT 编译失败 | torch 和 torch-tensorrt 版本都要是 2.8.0；显存不足时换一张卡；训练环境里不要装 |
| `jog` 里估计器发散（上百米） | 平地快跑别用台阶配置；默认配置在跑动里 xy 误差 0.2 到 0.4 m 属正常 |

## 9. 文件地图

`ScaleBridge/scalebridge/utils/legged_estimator/`：

| 文件 | 内容 |
|---|---|
| `state_estimator.py` | 估计器主循环，含时钟和更新钩子、可选的脚下地面高度提供者 |
| `linear_kalman_filter.py` | 卡尔曼滤波，含脚高度测量的有效位和按轴的外部位置噪声 |
| `lidar_odometry.py` | 雷达融合：对齐、杠杆臂、延迟补偿、门限和重对齐 |
| `depth_ground.py` | D435 融合：反投影、自体过滤、地图更新、落脚高度查询；D435 外参 |
| `height_map.py` | 滚动高度图，按观测次数衰减的更新增益 |
| `pose_wire.py` | FAST-LIO 位姿线（MagicLoco 格式）接收 |
| `depth_source.py` | 深度线（`rs_probe.py --serve`）接收；pyrealsense2 本机取图 |
| `online_client.py` | 真机在线客户端：LCM 机器人状态，加可选的雷达、深度线程 |

配置：`scalebridge/config/localization/legged_estimator{,_stairs,_lidar,_depth,_lidar_depth}.yaml`；仿真传感器：`scalebridge/simulator/mujoco_simulator.py` 里的 `lidar_odom`、`depth_ground`。

相关资料：`ScaleTrack/checkpoints/README.md`（检验点与门限）、`scalebfm_terrain_training_plan.md`（训练方案和决策日志）、`MagicLoco/sim2real/perception/`（FAST-LIO 桥、D435 取图、高度图）、`VideoMimic/sim2real/`（部署参考）。
