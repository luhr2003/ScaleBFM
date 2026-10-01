# ScaleBFM 地形能力训练计划（2026-09-29）

接续 [scalebfm_terrain_problem_and_plan.md](scalebfm_terrain_problem_and_plan.md)（问题分析）。本文是执行计划。依据有四类：ScaleTrack 代码、论文 arXiv 2607.15163（含附录）、本机 MagicLoco / MagicSim 仓库，以及已核实可下载的公开数据。

## TL;DR

- **验收标准**（两条都要满足）
  1. MagicLoco π_L v4 在地形上当 planner，ScaleBFM 当 tracker（mode 7，兼顾 mode 4），闭环走通 0.05–0.30 m 台阶的上和下，以及 boxes、rough、slopes。
  2. 原有能力不回退：8 个 mode × global/local，在论文的两套官方测试集上不超出噪声带；MagicSim 平地回放保持 31/32。
- **怎么训**：从 `model_22200` 继续训练，4 卡 DDP。
  - 地形数据只用 π_L v4 自己在它的地形分布上的 rollout，和部署时 planner 的输出同分布。
  - 平地环境与原训练逐项一致，用重建的原训练分布做 rehearsal。
  - 冻结的 `model_22200` 在平地样本上当锚（KL 正则），覆盖 8 个 mode。
- **分两步**
  - A：盲走微调，网络结构和部署接口都不改。planner 在 mode 7/4 里已经给出脚的目标，落脚点就在参考里。
  - B：只在 A 过不了闭环时做。给 actor 加 π_L v4 同款 187 射线 height scan，新输入零初始化，初始化时和原模型逐位等价。
- **新工作量主要在两处**
  - MagicLoco 没有可用的地形录制器和地形导出，需要新写。
  - ScaleTrack 要补：地形场景、按组采样、全 mode 评测器、锚损失。
- **时间**：约 3–4 周，其中 B 是否需要视 A 的结果而定。

## 1. 核查结论

### 1.1 ScaleTrack 代码里要改的地方

| 位置 | 现状 | 影响 |
|---|---|---|
| `tracking_env_cfg.py:49` | `terrain_type="plane"` | 需要地形场景 |
| `tracking_env_cfg.py:140` | critic 的 `root_height` 是世界系绝对高度（注释 "only applicable when no terrain"） | 改成离地高度，平地上数值不变 |
| `tracking_env_cfg.py:115-120` | actor 只有 IMU 和关节观测，没有地形输入 | A 阶段不动；B 阶段加 scan |
| `commands.py:284,352` | 参考位置 = 数据 + `scene.env_origins` | 需要按 env 覆盖原点，地形 clip 用录制时的坐标 |
| `commands.py:228` | 所有 clip 共用一个 multinomial 采样 | 需要按组（flat / terrain）采样，否则地形 clip 只占百分之几 |
| `package_motions.py:375,396,430` | 回放时 root 加了打包 env 的 xy 原点（`env_spacing=2.0`），保存时没减。示例 clip 的 pelvis 初始 xy 是 (87.0, −71.0) | 平面上无害；地形 clip 必须减掉，否则和地形错位 |
| `on_policy_runner.py:441,477` + `actor_critic_humanoid_transformer.py:230-233` | 训练中的 eval 走 `act_inference`：mode 全 1、不做 mask，所以只测 whole-body；误差是 14 个 link 的平均 | 需要独立的全 mode 评测器，按论文 B.2 只在激活的 link 上算 |
| `on_policy_runner.py:297-325` | checkpoint 里没有 adaptive sampling 权重；按 strict 加载 | 微调时采样权重从均匀开始；B 阶段需要 checkpoint 手术 |
| `ppo.py:187-191` | adaptive KL 可以把 lr 每次 ×1.5，最高到 1e-2 | 微调要给 actor lr 加上限 |

### 1.2 数据

- **论文两套测试集都在官方 HF**，路径 `WeishuaiZeng/ScaleBFM/test_set/`，可以直接当回归基准。`model_22200` 也在同一仓库。
  - `BONES_Test_Set_processed.zip`：5.8 GB，1 万条。
  - `Ours_Test_Set_processed.zip`：7.3 GB，1,649 条，另有 retargeted pkl。
- **更正**：100STYLE（810 条）属于 Ours Test Set，不在训练集里。
  - 训练集是 LAFAN、AMASS、OMOMO、GRAB、SnapMoGen、FineDance、BONES-SEED、Embody3D（论文 §3.3、表 8）。
  - 100STYLE 和 Xsens 必须保持 held-out。
- **原训练集规模**（论文表 8）
  - BONES-SEED：132,220 条 / 48.3M 帧。
  - 加上其余六个数据集：165,028 条 / 71.1M 帧。
  - 再加 Embody3D：184,206 条 / 102M 帧。
- **BONES-SEED 官方直接提供 G1 轨迹**：`g1.tar.gz` 23.5 GB，在 HF 的 gated 数据集里，同意条款即自动通过。ScaleRetarget 自带转换脚本，不用重新 retarget。

### 1.3 本机的 MagicLoco / MagicSim

- **能在地形上走的只有 π_L v4**
  - checkpoint：`MagicLoco/checkpoints/G1/final/terrain/v4_terrain_rpy/model_champion_tc2_it750.pt`
  - task：`Magicloco-HomieV4Rough-G1-Rpy-Play-v0`
  - 成功率（vx 0.6）：台阶上 ≈97%、下 ≈97%；最高档（0.275–0.30 m）上 0.80、下 0.90。
  - 输入含 187 射线 height scan：torso 上 17×11、0.1 m 间距、随 yaw 转。
  - 只输出腿和腰，手臂是随机动作，wz 基本没有控制力（转向靠 vy）。π_A v7 只能走平地。
- **缺录制器和地形导出**
  - MagicLoco 没有能导出 root 位姿 + 关节角的地形录制器，也不导出地形。
  - 生成器配置是 `seed: null`、`use_cache: false`，每次运行地形都不一样。
- **地形分布**
  - 8×8 m tile，10 行难度 × 20 列。
  - pyramid stairs：下 0.38、上 0.28，台阶高 0.05–0.30 m，踏面 0.30 m。
  - boxes 0.08、rough 0.12、上坡和下坡各 0.07。
  - 一个 20 s episode 大约走 7 m，clip 会跨 tile，所以要和整块地形配对，不能只配单个 tile。
- **脚底碰撞体不一致**
  - MagicSim 和 MagicLoco 用同一个 `g1_new.usd`：每只脚 2 个圆柱，前 r=0.033、后 r=0.030，中间约 5 cm 没有覆盖。
  - ScaleTrack 每只脚是 7 个 r=0.01 的胶囊。
  - 台阶边缘的接触会不同。
- **Python 环境冲突**
  - MagicLoco `.venv`：pip 版 IsaacLab 2.3.2 + torch 2.7。
  - ScaleTrack 要求源码版 IsaacLab `18c7c58` + torch_tensorrt 2.8（torch 2.8）。
  - 所以 ScaleTrack 单独建环境，两边通过文件交换数据。
- **之前的 ScaleBFM 集成不在本机**
  - 包括 `scripts/wbc/replay_with_scalebfm.py`、`scripts/wbc/magicloco_planner_scalebfm.py`、bundle、`TestOutput/bfm_*`，以及 31/32 回放套件。
  - 它们在 `/home/magics` 那台机器上。本机的 MagicSim 是新 clone，远端可能有这些文件，但还没 fetch。
  - 闭环验收要么把这些搬过来，要么在那台机器上跑。

## 2. 总体方案

```text
MagicLoco π_L v4 (planner) --rollout--> 地形参考 clip + 导出的地形 mesh --------+
原训练分布 (BONES-SEED / LAFAN / AMASS / ...) --retarget/打包--> 平地 rehearsal --+--> ScaleTrack 4 卡微调
冻结的 model_22200 ------------------------- 平地样本上的 KL 锚 ----------------+    (从 model_22200 开始)
                                                                                      |
          全 mode 回归门限（两套官方测试集）+ 地形 held-out + MagicSim 闭环验收 <-----+
```

设计原则：

1. **平地条件逐项不变。** 平地 env 沿用原来的平面、env 网格、DR、观测噪声、push、RSI、termination、reward，8 个 mode 均匀采样。
2. **地形数据和部署同分布。** 只用 planner 本身的 rollout。
3. **锚住原模型。** 在平地 env 的 on-policy 状态上加 KL(π_θ‖π_22200)，8 个 mode 都覆盖。
4. **先盲后视。** A 不改网络和接口；B 的改动在初始化时和原模型逐位等价。
5. **门限先于训练。** 先测出原模型各 mode 的噪声带，再开始训练。

不从头训：原训练用了 64 卡、22,200 次迭代，没必要也做不到。

## 3. 数据

### 3.1 地形参考：新写 MagicLoco 录制器

在 MagicLoco `.venv` 里，照 `eval_rough_g1.py` 的写法新写 `record_terrain_refs.py`。

- **地形**
  - 显式设 seed，关掉 curriculum，env 均匀铺到所有行和列。
  - 导出整块地形 mesh（dump `/World/ground/terrain`，或者 `use_cache=True` 得到 per-tile obj 加布局），同时导出 `terrain_origins`、`terrain_levels`、`terrain_types`。
  - 共做 5 个 layout（5 个 seed），其中 1 个留作 held-out 测试。
  - 以后不靠 seed 复现地形，直接把 mesh 和数据一起存档。
- **指令**（不超出 π_L v4 的训练范围）
  - vx −0.6…1.0（以 0.2–0.8 为主），vy ±0.4，wz 取小值，height 0.50–0.74，torso rpy ±0.45。
  - 每 2–5 s 分段切换一次，包括起步、停下、原地站立、在台阶上停下再走。
  - 起步段放在 layout 20 m 宽的平地边框上，这样也覆盖 planner 在平地上的步态。
- **手臂**
  - 打开 `HomieUpperRandomAction`（rho_a>0），让上身姿态更多样。
  - 如果最终系统用 Pink 驱动手臂，就换成 Pink 目标。
- **记录内容**，50 Hz（π_L 本身就是 50 Hz）：
  - pelvis 位置和四元数（wxyz 转成 xyzw）。
  - 29 个关节：从 43 个关节里按名字取 `WBC_BODY_29_NAMES`，顺序和 ScaleTrack 的 `G1_29DOF_JOINT_NAMES` 一致。
  - 每帧所在的 sub-terrain 类型和难度、指令，以及 reset / 摔倒标记。
  - 按 reset 切段，丢掉摔倒前 1.5 s，丢掉短于 3 s 的片段。
- **数据量**：训练用 20–40 h（约 3.6–7.2M 帧），按地形类型 × 难度均衡。π_L 开 4096 个 env，几分钟到几十分钟就能录完。
- **打包**
  - pkl 经 `package_motions.py` 转成 npz，地形 clip 要减去打包 env 的原点。
  - YAML 里加上 `group: terrain` 和 `layout_id`。

### 3.2 平地 rehearsal：重建原训练分布

| 优先级 | 数据集 | 获取方式 | 处理 |
|---|---|---|---|
| 1 | BONES-SEED（48.3M 帧，原训练集近一半） | HF `bones-studio/seed`，同意条款后用 token 下载 `g1.tar.gz` | `convert_bones_to_ours.py`；按 BONES Test Set 的文件名剔除那 1 万条测试 clip |
| 1 | LAFAN1 | HF `lvhaidong/LAFAN1_Retargeting_Dataset` 里的 G1 CSV | `convert_lafan_hf_to_ours.py` |
| 2 | AMASS（SMPL-X 版）+ SMPL-X 模型 | 官网注册 | `retarget.py +loader=amass +formatter=kinematic` |
| 2 | SnapMoGen、OMOMO、GRAB、FineDance | HF / Google Drive / 官网 | 按 ScaleRetarget README 里对应的 recipe |
| 3 | Embody3D（30.8M 帧） | facebookresearch/embody-3d | 许可要先确认；量大，放最后 |
| — | 100STYLE、Xsens | — | 不进训练，只做测试 |

- 统一用 `package_motions.py` 打包成 50 fps 的 npz。平地 clip 带打包原点偏移无妨。
- 用 `model_22200` 在整个 rehearsal 集上评一次，得到原模型本来就失败的 clip 列表，这些不计入回归门限。
  - 例如平面上本来就不可能完成的攀爬动作：ScaleRetarget 的 kinematic formatter 是整个 clip 统一减去最低点，所以上楼梯、攀爬类动作在原训练里一直是悬空的。
- **规模估算**
  - 处理后的 npz 约 1.6 KB/帧，全量约 170 GB。
  - 训练时的 motion library（14 个 link）约 1 KB/帧，全量约 100 GB，放共享内存，4 个 rank 共用一份。本机 376 GB 内存够用。

### 3.3 评测集（全部不进训练）

- BONES Test Set、Ours Test Set（HF 现成）。
- MagicSim 的 32 条平地 clip 和 8 条技能 clip（在 `/home/magics`）。
- held-out layout 上的 π_L v4 rollout。

## 4. ScaleTrack 改造

1. **环境**：Python 3.11 + Isaac Sim 5.1 + 源码版 IsaacLab `18c7c58`，再 `pip install -e source/scaletrack source/my_rsl_rl`，和 MagicLoco 的 `.venv` 分开。
2. **场景**
   - 平地 env 保留原来的平面和 env 网格。
   - 导出的 MagicLoco 地形 mesh 作为静态碰撞体放到远处一块区域（比如 x=+2 km），地形 env 在那里按录制坐标运行。
   - 不同 env 之间本来就不互相碰撞，多个 env 可以同时跑同一段地形。
3. **MotionCommand**（`commands.py`）
   - env 固定分组：约 65% flat、35% terrain，每个 rank 内部同样划分；组内按 adaptive 权重采 clip。
   - 每个 env 各自的参考原点：flat 用原网格原点，terrain 用 layout 偏移。
   - mode 采样：flat 组 8 个 mode 均匀（和原来一样）；terrain 组只采 planner 用的 mode 7 和 4。
   - 在地形上做 RSI 时加穿模修正：按脚下地形高度把 root 抬起来。
   - adaptive 权重存进 checkpoint。
4. **Critic 观测**
   - `root_height` 改为离地高度，平地上数值不变。
   - 加特权 height scan：17×11、0.1 m，和 π_L v4 一致。
   - 新输入对应的 `prop_projection` 列零初始化，critic 输出一开始不变；critic 的优化器重新初始化。
5. **机器人资产**
   - 把 ScaleTrack G1 的脚底碰撞体换成 `g1_new.usd` 的双圆柱，因为最终部署在 MagicSim。
   - Stage 0 先用原模型测一次换脚前后的平地指标；如果明显掉点，改成 flat 组保留胶囊、terrain 组用圆柱。
6. **PPO**（`ppo.py`）
   - 锚损失：在 flat 组样本上加 β·KL(π_θ‖π_22200)，σ 相同时就是 Σ(Δμ)²/2σ²。π_22200 冻结，输入同样的 masked 观测。
   - actor lr 1e-5，上限 3e-5；desired_kl 0.005；entropy_coef 0。
   - 前 300–500 次迭代冻结 actor，只训 critic。原因是 critic 有新输入，地形上的回报分布也不同。
7. **评测器**：新写 `scripts/eval/eval_all_modes.py`。
   - 固定 mode、走 masked 观测（沿用 `play.py --mode_index` 的做法）、用确定性动作、关掉观测噪声和 DR。
   - 按论文 B.2 只在激活的 link 上算 Succ、G-MPKPE、G-MPKRE、L-MPKPE、L-MPKRE。
   - global 和 local（`--local_tracking` 的 reference forcing）都测。
   - 输出逐 clip 的失败列表，方便和原模型对比。
   - 8 个 mode 可以放进同一批 env 里并行评。

## 5. 训练阶段

### Stage 0：基线（不训练，约 3 天，和数据准备并行）

- 装 ScaleTrack 环境，下载 `model_22200` 和两套测试集。
- 写评测器，测原模型：8 个 mode × global/local × 两套测试集 × 3 个 seed，得到噪声带；换脚前后各测一次。
- 在 ScaleTrack 里开环回放 π_L v4 的地形 clip，复现问题文档里"在第一个台阶处失败"的现象，作为训练前的地形基线。

### Stage A：盲走微调（结构和接口不变）

- **起点**：4 卡 × 8192 env，从 `model_22200` 加载，actor 的优化器状态一并加载。
- **训练顺序**：先做 critic warm-up，再联合训练。锚的 β 从 1.0 开始，按平地门限调。
- **地形课程**：先开放 ≤0.15 m 的台阶和 rough/slope；地形成功率稳定后再放开到 0.30 m。adaptive sampling 会把权重自动集中到失败的 clip 上。
- **评测节奏**
  - 每 500 次迭代跑一次快速门限：每个 mode 1,000 条测试 clip，加上地形 held-out。
  - 每 2,000 次迭代跑一次完整门限。
- **时长**：预计 3–6K 次迭代，按每次约 20–25 s 估，1–2 天。
- **出口**：全 mode 门限通过，且 held-out 地形开环 mode 7/4 的成功率 ≥ planner 自身的 90%，然后去 MagicSim 做闭环验收。

### Stage B：加 height scan（只在 A 过不了闭环时做，比如卡在台阶边缘或高台阶上）

- **输入**：actor 加一个 187 维 scan（17×11、0.1 m、随 yaw、挂在 torso，和 π_L v4 完全相同），再加一个 valid 位。
- **checkpoint 手术**：`prop_projection` 的权重扩列，新列置零，初始化时输出和原模型逐位相同。
- **scan 随机化**：加噪声、偏移、1–2 帧延迟、随机缺失。
  - flat 组 50% 概率 valid=0，terrain 组 20%。
  - 锚在 flat 组 valid=0 和 valid=1 两种情况下都加。
- **部署侧**
  - 导出脚本多一个 scan 输入，TensorRT 重新编译，metadata 更新。
  - MagicSim 给 tracker 机器人挂一个 RayCaster（移植 MagicLoco 的配置）。
  - valid=0 时就是盲走，接口可以向下兼容。

### 备选：双教师 DAgger 蒸馏（锚控制不住遗忘时）

- 教师：flat clip 用 `model_22200`；terrain clip 用 Stage A/B 训出的地形模型，可以带特权观测。
- 学生：从 `model_22200` 初始化，按 env 所在组选教师做 action MSE，学生自己 rollout。
- 纯监督，平地行为最稳。

## 6. 门限与验收

**回归门限**：每个 checkpoint 都和同条件下的 `model_22200` 对比。

| 指标 | 范围 | 门限 |
|---|---|---|
| Succ | 8 个 mode × global/local × 两套测试集 | 下降 ≤ max(0.3 pp, 2σ) |
| G-/L-MPKPE、G-/L-MPKRE | 同上 | 上升 ≤ max(3%, 2σ) |
| 新增失败 clip | 同上 | 逐条过目，不能集中在某一类动作或某个数据来源 |
| MagicSim 平地回放 | 32 条平地 + 8 条技能 | 不低于 31/32、7/8 |

**地形验收**：

1. **ScaleTrack 开环**：在 held-out layout 的 π_L v4 clip 上，按地形类型 × 台阶高度分桶，mode 7 和 4 的成功率 ≥ planner 自身的 90%。
2. **MagicSim 闭环**（最终标准）
   - π_L v4 ghost 做 planner，ScaleBFM 做 tracker。
   - 场景：0.05–0.30 m 台阶上/下、boxes、rough、slopes，每格至少 20 次。
   - 成功率 ≥ planner 单独运行的 90%。
   - 问题文档里开环回放 0/24、闭环 0/14 的那些场景全部重跑。

## 7. 算力、存储、时间线

- **GPU**
  - 本机有 5 张 A6000（48 GB）。目前 0/2/4 号被另一位用户的 vLLM 占着（各 43 GB），1/3 号在跑 MagicLoco eval。4 卡训练要先协调出 4 张空卡。
  - 每卡 8192 个 env：rollout 缓冲约 11 GB，B 阶段的 scan 再加约 2 GB，48 GB 够用。
- **CPU**：80 核，负责 retarget 和打包。
- **存储**：`/home` 还剩 1.4 TB。下载（BONES 23.5 GB、两套测试集 13 GB、AMASS 等）加上约 170 GB 的处理后 npz，放得下。

| 周 | 内容 |
|---|---|
| 1 | 建 ScaleTrack 环境；下载、retarget、打包（后台跑）；写评测器并出基线；写 MagicLoco 地形录制器、导出地形 |
| 2 | ScaleTrack 地形场景、分组采样、critic 观测、锚损失；小规模冒烟测试；启动 Stage A |
| 3 | Stage A 训练并过门限；把 MagicSim 闭环 harness 搬到本机（或在 `/home/magics` 上跑）做验收；决定是否做 B |
| 4 | （需要时）Stage B，包括导出和 MagicSim 接入 scan |

## 8. 风险

1. **凑不齐 4 张卡**：目前本机只有 2 张可用。
2. **Rehearsal 覆盖不全**：拿不到的数据集（许可或注册问题）只能靠锚和泛化兜底。门限按数据来源分项看，找出受影响的动作类型。
3. **小 batch 微调更新噪声大**：原训练用 64 卡，我们只有 4 卡。对策是低 lr、锚和 critic warm-up。
4. **脚底接触不一致**：先测换脚的影响，闭环验收放在 MagicSim 里做。
5. **Planner 本身的局限**：π_L v4 转向能力弱、手臂是随机动作，所以地形数据里没有"台阶上转身""拿着东西上台阶"。如果验收场景需要，就得换 planner 或给 planner 补数据。
6. **闭环 harness 不在本机**：如果搬不过来，可以把 π_L v4 的 ONNX（`MagicLoco/sim2real/deploy_packs/g1_terrain_v4_rpy/`）作为 ghost 放进 ScaleTrack 的评测里。

## 9. 本周可以开始的事

1. 在 HF 上同意 BONES-SEED 的条款并准备 token；注册 AMASS、SMPL-X、GRAB。
2. 下载 `model_22200`、两套测试集、LAFAN1 G1。
3. 建 ScaleTrack 独立环境，跑通 README 里的 example play。
4. 写 `eval_all_modes.py`，出原模型的全 mode 基线。
5. 写 MagicLoco 地形录制器，先录一个 layout 联调格式。
6. 从 `/home/magics` 拿回 `scripts/wbc/`、bundle、`TestOutput/bfm_*`，或者 fetch MagicSim 远端确认。

---

## 附录：执行记录与已确定的决策（2026-09-29 更新）

### 已验证的事实

- **数据管线与官方一致**：新写的向量化打包脚本 `scripts/data/package_motions_fast.py` 在 256 条 Ours 测试 clip 上与官方 processed 数据对比，关节位置/速度逐位相同（差 0），body 位姿在 99.9% 的元素上一致到 1e-5，仅在个别帧有 ≤1.4 mm 的读回差异。全量 BONES 训练集 132,220 条（原 142,220 条去掉 10,000 条测试）已全部打包。
- **评测器已就位**（`scripts/eval/eval_modes.py`、`run_gate.sh`、`summarize_gate.py`）：按论文 B.2 的定义只在被激活 link 上算 Succ / G-MPKPE / G-MPKRE / L-MPKPE / L-MPKRE，local 模式复现部署里的 reference forcing；时间对齐采用训练时的约定（reset 之后 command 已前进一步），否则会带入 20 ms 的系统性滞后。原模型 `model_22200` 的基线（BONES 1000 条 / Ours 300 条子集，mode 7 global）：Succ 0.999 / 0.993，G-MPKPE 5.2 / 5.9 cm，L-MPKPE 3.6 / 3.6 cm，与论文表 3（0.040 / 0.040）同量级。
- **地形采集与 ScaleTrack 的机器人模型完全一致**：用 MuJoCo 加载 ScaleTrack 的 G1 MJCF，对冒烟数据做正运动学，再用射线查网格高度，站立帧的脚底—地面间距均值 +0.02 cm（p5/p95 约 ±0.55 cm），各档台阶高度都一致；把四元数误读成 wxyz 时，站立帧占比从 52% 掉到 1.3%，说明检验足够灵敏。
- **训练管线端到端跑通**：冒烟训练从 `model_22200` 续训，冻结 actor 阶段的锚定 KL 恰好为 0，说明锚点与加载的策略一致；解冻后 KL 稳定在 0.009 左右。

### 决策与修正

1. **原模型训练结束时 actor 学习率已被 adaptive 调度推到 8.6e-4**（配置里的 2e-5 只是初值）。微调用固定的小学习率（默认 5e-5，可调），并在加载 checkpoint 后强制覆盖 optimizer 里保存的学习率。
2. **训练库控制在约 6000 万帧**（约 58 GB）：加载多卡共享内存库时峰值内存约为库大小的 3 倍，全量数据会逼近机器的 376 GB。flat：BONES 随机 6 万条 + LAFAN + SnapMoGen 等；terrain：从 6 个训练 layout 里每个抽约 5000 条。依据论文表 1 与 4.2.2：同源数据从 XXS 扩到 S 收益很小，多样性比数量重要。
3. **多卡启动前必须清理 `/dev/shm/shared_motionlib_*`**：否则 `MotionCommand` 会静默复用上次遗留的共享内存库。
4. **不使用 runner 内置的 adaptive sampling 与训练中评测**（每 200 次迭代要评一遍整个库，太贵）。改为外部的 `watch_gate.sh`：按固定间隔对新 checkpoint 自动跑快速门限（flat）和留出地形门限，并与基线做逐 clip 配对比较。
5. **暂不改机器人资产**：ScaleTrack 的脚是 7 根 r=0.01 的胶囊，MagicSim 的 `g1_new.usd` 是两个圆柱（r=0.033/0.030），站立平面高度一致，但台阶边缘的接触不同。先用现有资产训练，闭环验收时如果发现 sim 间差距，再做短阶段的脚型对齐。
6. **不加 height scan**（阶段 A）：网络结构与部署接口完全不变，导出的 TensorRT 模型和 MagicSim 适配器直接可用。只有留出地形的开环成功率仍不达标时才进入阶段 B。

### 运行约定

- GPU：2026-09-29 晚你明确说“用卡 0–3 训练、GPU 4 做评测”，据此 ft_v2 四卡训练在 GPU 0–3、门限评测在 GPU 4；GPU 1 上如果有你自己的 MagicLoco 评测任务会和训练共用这张卡（显存够用，但会拖慢同步训练）。之前只用 0/2/4 是因为自动权限检查拦截了对 1/3 的使用，直到你点名授权。
- 所有脚本与代码改动都在工作区里，未提交 git；数据、环境和日志在 `/home/vcj9002/scalebfm_ws/`。

### 噪声基线与门限标定（2026-09-29 晚）

同一个原模型 `model_22200`、只换评测随机种子（0/1/2，DR 不同），快速门限（BONES 1000 条 / Ours 300 条子集）下的波动，作为判断“是否退化”的参照：

| | 成功率波动（三个种子 max−min） | 误差均值波动（只算成功 clip） |
|---|---|---|
| BONES，global 配置 | ≤ 0.4 pp | ≤ 1.2% |
| BONES，local 配置 | ≤ 1.9 pp | ≤ 6.5% |
| Ours（小样本 300 条），global | ≤ 1.7 pp | ≤ 3.9% |
| Ours，local | ≤ 3.7 pp | ≤ 6.0% |

据此把 `summarize_gate.py compare` 的判据定为：成功率显著下降（McNemar p<0.05）超过 0.3 pp（local 配置 1 pp），或无论显著与否下降超过 1 pp（local 3 pp）；误差均值上升超过 3% 且配对 bootstrap 的 95% CI 不含 0。注意误差只在“新旧两个模型都成功”的 clip 上比较，否则失败 clip 的米级误差会把均值带乱。

### 正式训练 `ft_v1` 的运行记录

- 16:26 起在 GPU 0–3 上四卡训练，GPU 4 专做门限评测；每张卡 2048 个 env，稳态每次迭代约 24 s（收集 12 s + 学习 12 s），合计约 2.1 万步/秒。
- 超参：actor 学习率 1e-4（固定）、critic 3e-4、锚定系数 2、前 40 次迭代只训 critic、地形 env 占 40%、地形课程在 800 次迭代内把台阶高度上限从 0.14 m 放开到全高度。
- 踩到的坑：这台机器上多卡 NCCL 的 P2P 传输会死锁（IOMMU 开启），需要 `NCCL_P2P_DISABLE=1`；多卡启动时 rank 0 加载 9 万条 clip 超过 10 分钟会让其他卡的 NCCL 初始化超时，已把超时加到 120 分钟；动作库改为缓存到磁盘，重启只要几分钟。
- 结果：`ft_v1` 在第 40 次迭代（actor 第一次更新）崩溃，`normal expects all elements of std >= 0.0`。**这次运行作废**（它的 rank 1–3 一直在训练一个被清零的策略，见下）。

### 多卡训练事故：这台机器上 GPU→GPU 拷贝会静默失败（2026-09-29 晚）

- **现象**：单卡冒烟/pilot 完全正常；多卡时 rank 1..N−1 的 critic 每个 minibatch 都出现 NaN 梯度，经 all-reduce 传染所有 rank，随后 actor 更新时 `std` 变成 NaN，报上面的错误。
- **不是某张卡坏了**：把 GPU 顺序换成 `2,0`（rank 1 变成物理 GPU 0）失败依旧出现在 rank 1；GPU 2/3 作为 `cuda:0` 单卡跑完全正常。问题跟着“rank≥1（`cuda:k`, k≥1）”走。
- **根因**：这台机器（IOMMU 开启）上 PyTorch 的 GPU→GPU 直接拷贝（`copy_` / `.to("cuda:k")`）**不报错但拷不过去**（拿到 0 或分配器里的旧内存），走 host 中转是对的，NCCL 在 `NCCL_P2P_DISABLE=1` 下也是对的（4 rank 广播/all_reduce 与 CPU 参考逐位一致）。训练代码里两处踩到：
  1. rsl_rl 原版 `broadcast_parameters` 用 `broadcast_object_list` 广播 CUDA `state_dict`，反序列化后的张量都落在 `cuda:0` 上，`load_state_dict` 再拷到本 rank 的 GPU → rank≥1 整个策略（actor、critic、`std`）被清零。
  2. `runner.load(path)` 的 `map_location=None`：预训练 checkpoint 里 431 个 CUDA 张量（含全部 Adam 动量）都标着 `cuda:0`，rank≥1 加载优化器状态时同样走 GPU→GPU 拷贝，得到垃圾 Adam 状态，第一次 step 就把权重写成 NaN。
- **修复**（都在工作区）：`PPO.broadcast_parameters` 改成逐张量的 NCCL in-place 广播 + 全 rank 校验和（用 SUM all_reduce 收集，因为 NCCL 的 MIN/MAX 会忽略 NaN）；`train.py` 里 `runner.load(..., map_location=<本 rank 设备>)`；启动时校验所有 rank 的参数与优化器状态一致且有限，不一致直接报错；每 50 次迭代重新广播一次参数并报告漂移；每 10 次迭代打印每个 rank 各自的平均 episode reward/长度（只有 rank 0 写日志，防止某个 rank 坏了却看不出来）；再加了 NaN 梯度保护（跳过该 minibatch 并计数，`skipped_updates_total` 应恒为 0）。
- **验证**：4 卡冒烟（GPU 0–3，每卡 512 env）——跳过的 minibatch 为 0，各 rank 参数/优化器状态校验和逐位相同，actor 解冻后锚定 KL≈0.012，value loss、reward 与单卡冒烟一致。
- **教训**：这台机器上多卡代码里不能有任何 GPU 之间的直接张量搬运；跨 rank 只用 NCCL 集合通信（`NCCL_P2P_DISABLE=1`），读 checkpoint 一律 `map_location=<本卡>`。

### 正式训练 `ft_v2` 的运行记录

- 17:14 起，GPU 0–3 四卡训练（每卡 2048 env），GPU 4 专做门限评测；配置同 `ft_v1`：actor lr 1e-4（固定）、critic 3e-4、锚定系数 2（锚 = `model_22200`）、前 40 次迭代只训 critic、地形 env 40%、地形课程 800 次迭代放开台阶高度；共 3000 次迭代，每 50 次存一个 checkpoint，门限每 200 次迭代跑一次（首个被门限的 checkpoint 是 `model_22400`）。
- 启动命令：`ENVS=2048 ITERS=3000 FREEZE=40 SAVE=50 TERRAIN_FRAC=0.4 ANCHOR=2 ACTOR_LR=1e-4 CRITIC_LR=3e-4 EXTRA="env.commands.motion.terrain_curriculum_steps=51200" scripts/train/launch_finetune.sh ft_v2 0,1,2,3`；门限：`scripts/eval/watch_gate2.sh ft_v2 4 100`（你说“可以多训练一会儿、多 eval”，所以门限改为每 100 次迭代一次，平地 modes 0–3、modes 4–7 和留出地形三部分在 GPU 4 上并行跑；首个被门限的 checkpoint 是 `model_22300`）。训练可以按需要续训延长（FREEZE=0，锚仍是 `model_22200`，地形课程直接开到全高度）。

### 门限的噪声基线补充（2026-09-29 晚，用来读 ft_v2 的门限结果）

- **留出地形门限**（1500 条 clip，mode 7 / mode 4，global）：原模型换 3 个评测种子，mode 7 总成功率 18.4 / 17.5 / 16.9%（波动 1.5 pp），mode 4 为 15.3 / 15.2 / 14.9%（0.4 pp）；按台阶高度分箱的成功率每箱波动约 ±1–4 pp（每箱 n≈200–450）。所以总成功率变化超过约 3 pp、分箱变化超过约 5–6 pp 才算真实变化。
- **平地门限的“聚合”读数**：同一个权重（`ft_v1_it22200` 的 actor 与原模型完全相同，只是换了跑评测的 GPU 与并行方式）与原模型的配对差异：BONES global 平均 ΔSucc −0.06 pp、local +0.06 pp；Ours（300 条）global −0.25 pp、local −0.33 pp；误差均值变化都在 ±0.5% 以内。`compare.txt` 末尾新增了这些聚合行，判断平地是否退化时优先看聚合读数（BONES 平均 ΔSucc 低于 −0.5 pp 或误差均值上升超过 1% 才值得警惕），单个配置的 REGRESSION 标记在 300 条的 Ours 集合上会有噪声触发。

### 无人迭代运行手册（2026-09-29 晚起；你说“那你就无人迭代模式”）

**机制**：后台脚本 `tmp/wait_event.py`（`Bash run_in_background`）在“新门限结果 / 训练崩溃、NaN 保护触发、某个 rank 不健康、日志停滞 7 分钟 / 3 小时无事发生”时唤醒我；每次唤醒后读 `runs/eval/ft_v2_it<N>/{compare,terrain}.txt`，按下面规则决策、写入“决策日志”，再重新挂上 `wait_event.py`。GPU 0–3 训练、GPU 4 评测（你已授权）；不碰你自己的任务；git 提交/发布等对外动作等你开口。

**规则**（数字来自上面的噪声基线）：
1. **平地退化**（最高优先级）：`compare.txt` 末尾的聚合行——BONES global 或 local 的平均 ΔSucc ≤ −0.5 pp 且连续两个门限如此，或 BONES 任一配置 ΔSucc ≤ −1.5 pp 且 p < 0.01 并在下一个门限重现，或聚合误差均值上升 > 2% → 停训，从最近一个“好”checkpoint 用 `resume_ft.sh` 续训，锚系数 ×2（上限 16）、actor 学习率 ×0.5，并记录；仍退化就加大平地 rehearsal 比例（`TERRAIN_FRAC` 0.4→0.3）。
2. **地形停滞**：课程放开到全高度（约 22999 之后）且 mode 7 总成功率 400 次迭代内提升 < 2 pp → 先看分箱（高台阶是否还在涨）；仍停滞则依次尝试：提高地形 env 比例、提高 actor 学习率（≤2e-4，同时锚系数 ×1.5）、延长训练；仍不够才做阶段 B（height scan，要改网络与部署接口，需要先告诉你）。
3. **崩溃/停滞/OOM**：`scripts/train/resume_ft.sh ft_v2 0,1,2,3` 从最新 checkpoint 续训（同一目录、actor 不冻结、课程恢复）。GPU 1/3 上你的任务占显存时，训练的 OOM 也用它恢复。
4. **训到 25199 次迭代结束时**：如果地形成功率仍在涨就续训（`ITERS` 再加 3000）；否则进入验收：在 5 张卡上并行跑完整门限（10,000 条 BONES + 全部 Ours + 全部留出地形，多个种子），在候选 checkpoint 里选“地形最好且平地不退化”的；如果最优 checkpoint 有轻微平地退化，用 `interpolate_ckpt.py` 做权重插值（α 扫描）折中。
5. **闭环验收**（MagicLoco 地形策略当 planner、ScaleBFM 当 tracker，在 MagicSim 里走通）：等你的回放脚本；训练期间我自己先搭一套闭环验证（见下）作为后备，你的脚本到了就用你的。
6. **汇报**：每个里程碑（门限结果、规则触发、训练结束、验收）发一条简短消息；最终结果整理成一个页面（表格 + 视频）。

**决策日志**
- 2026-09-29 18:43 `ft_v2_it22300`（首个门限）：平地无退化（BONES global −0.03 pp / local +0.44 pp，Ours −0.25 / −0.27 pp，误差 ±1.3% 以内）；留出地形 mode 7 成功率 18.4% → 45.7%，mode 4 15.3% → 39.1%。决策：继续训练，无调整。
- 2026-09-29 19:26 `ft_v2_it22400`：平地无退化（BONES global −0.09 / local −0.26 pp，Ours −0.21 / −0.47 pp；Ours 单配置 mode7_local −4.0 pp、p=0.09，方向与其他 Ours local 配置混杂，判为噪声）；留出地形 mode 7 68.4%、mode 4 61.7%。决策：继续。
- 2026-09-29 20:22 `ft_v2_it22500`：平地无退化（BONES global −0.01 / local +0.30 pp，Ours −0.13 / +0.33 pp，误差 ≤0.6%）；留出地形 mode 7 **72.5%**（基线 18.4%）、mode 4 71.5%（基线 15.3%），G-MPKPE 1.40 → 0.24。决策：继续；因平地稳定，把常规门限改为每 200 次迭代（奇数百的门限用标记文件跳过），空出 GPU 4 做仿真差异检查。
- 2026-09-29 20:16 **仿真差异检查（脚型）**：MagicSim 的 `g1_new.usd` 每只脚是两个圆盘碰撞体（前 r=0.033 @x=+0.109、后 r=0.030 @x=−0.036，厚 1.5 cm，脚底高度与我们一致），ScaleTrack 资产是 7 根 r=1 cm 的细胶囊；两套资产的 29 个身体关节与 30 个 body 名字一致，质量相差约 4%（腕/躯干），PD 增益与默认姿态和 MagicSim 的 `G1_Sonic` 完全相同。做了“圆盘脚”版本资产（`assets/g1_29dof_discfeet/`，只换脚的碰撞体，每脚仍保持 7 个碰撞形状以兼容物理材质随机化），环境变量 `SCALETRACK_ROBOT_USD` 可切换。结果（留出地形 1500 clip）：基线 mode 7 18.4% → 16.3%；`ft_v2_it22500` mode 7 72.5% → **61.1%**、mode 4 71.5% → 57.8%，损失集中在高台阶（0.15–0.20 m: 75.6% → 54.2%，0.20–0.25 m: 53.2% → 35.5%，0.25–0.31 m: 42.7% → 28.0%），低台阶几乎无损；圆盘脚下仍比基线高 45 pp。结论：脚型差异是真实的（约 11–14 pp），进 MagicSim 会掉点但不致命，值得在训练里加入脚型多样性。
- 2026-09-29 20:30 **决策：混合脚型续训**。给环境加了 `SCALETRACK_ROBOT_USD_MIX`（`MultiUsdFileCfg`，每个 env 随机取一种脚型，`replicate_physics=False`）；冒烟测试无速度损失（512 env 单卡：收集 26.6 s vs 26.0 s）。杀掉 `ft_v2`（迭代 22657），从 `model_22650` 用 `resume_ft.sh` 续训（capsule / 圆盘脚各约 50%，其余超参不变，地形课程恢复到 0.625，ITERS=3000），旧 checkpoint 备份在 `runs/backup/ft_v2_it22650_capsule_only.pt`，旧日志 `tmp/train_ft_v2_part1.log`。新增圆盘脚门限 `scripts/eval/watch_gate_disc.sh`（地形每 200 次、平地每 400 次迭代，对照 `runs/eval/base22200_disc`）。
- 2026-09-29 22:09 `ft_v2_it22600`（混合脚型之前的最后一个纯胶囊脚 checkpoint）：平地无退化（BONES global −0.05 / local +0.10 pp，Ours −0.17 / +0.80 pp）；留出地形 mode 7 **76.3%**、mode 4 72.0%；圆盘脚（21:04）mode 7 62.6%、mode 4 60.0%。决策：继续。
- 2026-09-29 22:11 GPU 4 过载（常规门限 22600 用了 6382 s）。决策：把门限并成一个队列 `scripts/eval/watch_all.sh ft_v2 4 200`：每 200 次迭代做留出地形（mode 7/4，global + local，胶囊脚与圆盘脚各一份）；平地在 it%800==0 做胶囊脚完整门限、it%800==400 做圆盘脚完整门限，其余检查点只做 BONES 哨兵（modes 0/4/7，global+local）。`run_gate.sh` 新增 `SETS`/`TRACKING` 环境变量。旧的两个监视器只收尾 22800 的门限，之后由标记文件跳过。首个混合脚型训练出的 checkpoint 是 `model_22800`（22800 的胶囊脚 + 圆盘脚门限由旧监视器跑）。
- 2026-09-29 22:55 `ft_v2_it22800_disc`（混合脚型训练 150 次迭代后的第一个 checkpoint，圆盘脚）：平地无退化（BONES global −0.23 / local +0.32 pp，Ours −0.04 / +0.67 pp）；留出地形 global mode 7 **69.7%**、mode 4 68.3%（混合前 22600 是 62.6% / 60.0%），分箱 0.15–0.20 m 66%、0.20–0.25 m 46%、0.25–0.31 m 37%。local（参考强制）地形“成功率”只有 7.5% / 8.1%（基线 1.7% / 1.5%）。
- 2026-09-29 23:30 **local 地形结果的解读（重要）**：`Succ` 的定义是“任何被激活的链接的世界系位置误差从未超过 0.5 m”，参考强制下机器人没有位置反馈，一条要走 7–8 m 的地形轨迹光靠漂移就会超过 0.5 m，所以 local 的 `Succ` 主要量的是漂移；相对根的姿态误差（L-MPKPE）平均只有 0.20 m。local 模式演示视频（平缓粗糙地形、0.09 m 台阶）里机器人姿态稳定、没有摔倒，只是相对参考轨迹漂移约 1 m。结论：评 local 用“相对根误差成功率”`SuccL`（任何链接相对根的位置误差从未超过 0.5 m）和 L-MPKPE，而不是 `Succ`；评测器新增了 `success_rate_local_error` / `max_l`，`summarize_gate.py` 的地形对比同时显示 SuccL；基础模型与 `model_22800` 的 local 地形（胶囊脚 / 圆盘脚）在用新指标重跑（后缀 `c`）。如果 SuccL 也低才需要在训练里加参考强制的环境。
- 2026-09-29 23:51 `ft_v2_it22800`（胶囊脚，混合脚型训练后的第一个 checkpoint）：平地无退化（BONES global −0.06 / local +0.20 pp，Ours −0.08 / +0.13 pp，误差略降）；留出地形 mode 7 global **82.2%**（基线 18.4%）；mode 7 local 的 Succ 9.4%（基线 1.4%），**SuccL 59.5%（基线 18.5%）**。local 模式（部署风格，无位置反馈）明显落后于 global（59.5% vs 82.2%），因此决定训练里加入参考强制。
- 2026-09-29 23:57 **决策：加入参考强制环境续训**。`LayoutMotionCommand` 新增 `local_forcing_prob_flat / local_forcing_prob_terrain`：每个 episode 开始时按概率让该 env 用参考根位置代替机器人测得的位置（机器人各链接随之刚性平移，与 `eval_modes.py --tracking local` 一致），仅对控制模式激活了骨盆的 env 生效；奖励与终止随之只看相对误差；critic 的离地高度改用真实骨盆位置。取值：地形组 0.5、平地组 0.3（平地组也进入锚定 KL，保护平地 local 跟踪）。冒烟通过后，从 `model_23150`（备份 `runs/backup/ft_v2_it23150_before_forcing.pt`）续训，同时保留混合脚型；旧日志 `tmp/train_ft_v2_part2.log`。启动命令：`SCALETRACK_ROBOT_USD_MIX="<胶囊>,<圆盘>" EXTRA="env.commands.motion.local_forcing_prob_terrain=0.5 env.commands.motion.local_forcing_prob_flat=0.3" ITERS=3000 scripts/train/resume_ft.sh ft_v2 0,1,2,3 model_23150.pt`。
- 2026-09-30 02:31 **决策：地形“难 clip”加权续训**：地形成功率在 23000 之后趋平（胶囊脚 mode 7 global 84.7% → 83.5%，圆盘脚 73.4% → 71.7%），短板是高台阶和下楼梯（0.22–0.31 m 下行 53%、上行 71%）。新增 `terrain_hard_boost`（地形 clip 的采样权重 1 + boost × clamp((max_step−0.10)/0.15, 0, 1)），取 3.0，使 ≥0.15 m 台阶的 clip 采样占比 44% → 70%；从 `model_23500` 续训（备份 `runs/backup/ft_v2_it23500_before_hardboost.pt`）。结果：到 24600–24800，胶囊脚 mode 7 global 87.5–89.0%，圆盘脚 77–79%。
- 2026-09-30 02:31–14:34 **疏漏**：这段时间我没有把 `wait_event.py` 重新挂上（处理完 23400 之后忘了），所以 23600–24800 的门限是 14:34 回头集中复查的；训练和门限本身一直正常，复查结论：平地无退化（BONES global −0.06…−0.20 pp、local −0.22…+0.60 pp，Ours global −0.29…+0.17、local −0.53…+0.73 pp，均在噪声内），没有触发任何运行手册规则。local（参考强制）地形 SuccL 一直在 52–61% 徘徊，25% 左右的参考强制训练没有让它变好（失败集中在高台阶，机器人没有位置反馈、漂移后踩错台阶）。
- 2026-09-30 14:37 **用户指示：脚的模型要和 MagicSim 用的一样；其余任务不能掉**。核对了 MagicSim `g1_new.usd` 的脚踝连杆：两个圆柱碰撞体（前 r=0.033 @x=+0.1089、后 r=0.030 @x=−0.0355，高 0.015，z=−0.0275，接触/静止偏移均未设置），连杆质量 0.608 kg、质心 (0.0265, 0, −0.0164)、惯量与 ScaleTrack 资产一致——圆盘脚版本就是 MagicSim 的脚模型。**决策**：停止混合脚型，从 `model_25050`（备份 `runs/backup/ft_v2_it25050_before_disc_only.pt`，旧日志 `tmp/train_ft_v2_part4.log`）改成只用圆盘脚续训（`SCALETRACK_ROBOT_USD=<disc>`），难 clip 加权保留 3.0，参考强制降到地形 0.25 / 平地 0.15；门限改成圆盘脚优先（`scripts/eval/watch_all2.sh`：圆盘脚地形 global+local 每 200 次迭代、圆盘脚平地完整 26 配置每 400 次迭代、其余为 BONES 哨兵；胶囊脚只在每 800 次迭代做地形 global + BONES 哨兵当原资产的健全检查），一律对照同一脚型上的基础模型（`base22200_disc`）。
- 2026-09-30 19:25 `ft_v2_it25400_disc`：BONES 哨兵 mode 4 global 99.8% → 97.7%（−2.1 pp，p<0.001），其余配置不变，地形 mode 7 83.7%。用 25300/25350/25450/25500/25600 的 BONES mode 4 复查：99.7 / 99.7 / 99.8 / 99.7 / 99.8%，25600 的完整平地门限也是干净的（BONES −0.18 / −0.04 pp，Ours −0.17 / −0.33 pp）。结论：25400 是一次性的抖动，没有趋势，不触发规则 1；25400 已从 `releases/` 撤掉。
- 2026-09-30 21:51 `ft_v2_it25600`（完整平地门限，圆盘脚）：干净；地形 mode 7 80.5%、mode 4 80.9%（胶囊脚 89.9% / 88.0%）。K 扫描（25200，mode 7 global，圆盘脚）：K=5 73.9%、K=10 80.0%、K=16 **82.5%**、K=24 81.1%、K=32 80.4%、训练分布 80.1%。**更正**：训练里 actor 未来偏移里的 `-1` 不是“片段最后一帧”，而是每个 episode 在 [5, 32] 帧里随机取的前瞻 K（`rand_timestep_range=(5,33)`），部署导出的 `[0..5]` 就是 K=5，是最弱的取值。高台阶（≥0.22 m）的失败 34%，中后段为主（中位 10.9 s），先出问题的是膝/踝，下行楼梯比上行失败多。
- 2026-10-01 MagicSim 闭环（另一个 session，`ft_v2_it25200`，mode 7、真实根位姿、MagicLoco 规划器在影子 G1 上）：0.08–0.22 m 台阶 K=5/16/32 全部无摔倒；0.26 m 上行不等待规划器 K=16 摔 6/8、规划器等待跟踪器（leash）摔 1/8；0.26 m 下行 2/8（leash 4/8，规划器自己也摔 1–2/8）；0.30 m 规划器自己都很难上。平地操作类技能（跪、劈叉、下蹲等）无摔倒。结论：≤0.22 m 走通，≥0.26 m 上行时 tracker 是短板。
- 2026-10-01 01:36 **决策**：(1) 清掉三个遗留的旧门限监视器（有一个仍在按每 100 次迭代评胶囊脚，挤占 GPU 4），只留 `watch_all2.sh`；(2) 因为 ScaleBridge 已加上用于 global tracking 的板载状态估计器、MagicSim 闭环也用真实根位姿，地形环境不再做参考强制（地形 0，平地保留 0.1 保护 local 平地）；(3) 地形控制模式比例 mode 7 0.55 → 0.65、mode 4 0.30 → 0.20（planner 用 mode 7）；(4) 难 clip 加权 boost 3 → 5；从 `model_25900`（备份 `runs/backup/ft_v2_it25900_before_modemix.pt`）续训，ITERS=2300，结束于约 28200。
- 2026-10-01 08:41 **（接上）评测器的“进程内顺序”问题**：BONES mode 4 的周期性“掉点”（25400、25800、26200、26600）只出现在轻量哨兵门限（一个进程里依次跑 mode 0、4、7）里；对干净的 `ft_v2_it25600` 做对照实验：哨兵顺序 0→4→7 时 mode 4 global 99.8% → 98.3%，把 mode 4 放第一个时 99.8%，4→0→7 时是 mode 0 掉到 98.4%。结论：同一进程里排第二的配置会在几条脆弱的动态 BONES clip 上少约 1.5 pp（评测器的状态没有完全复位），**不是策略退化**。从此每个检查点都做完整 26 配置平地门限、进程结构和基线一致（`watch_all3.sh`）。
- 2026-10-01 09:51 **门限静默失败**：`ft_v2_it26800` 的平地门限因为四个 Kit 进程同时启动（`Failed to create simulation view backend`，`run_gate.sh` 仍然打印 GATE_DONE）没有产生任何结果，门限行里 `flat agg:` 为空。新的 `scripts/eval/watch_all4.sh`：每个进程错开 15 s 启动，门限结束后校验 6 个结果文件，缺哪个补跑哪个（最多 3 次）；`--repair <it>..` 补跑旧检查点，`--eval <tag> <ckpt>` 评测任意 checkpoint 文件（例如权重平均）。26800、26600、26200 的平地门限已排队补跑。
- 2026-10-01 12:00 **地形成功率自 26200 起单调下降**（圆盘脚，留出地形，mode 7 global：83.6 → **85.5**（26200）→ 83.1 → 82.4 → 81.5 → 81.1（27000）；mode 4：83.5 → **84.1** → 80.4 → 77.9 → 78.8 → 78.3）。26250 起的设置（锚定系数 5、actor 学习率 6e-5）本来是为压住“平地 mode 4 抖动”，而那是评测假象，所以这个理由不成立，而且更强的锚定会限制地形学习。**决策**：停掉 `ft_v2`（迭代 27079），从峰值 `ft_v2/model_26200.pt`（备份 `runs/backup/ft_v2_it26200_peak.pt`）分支出新的运行 `ft_v3`，恢复 25900–26250 的设置（锚定系数 2、actor 学习率 1e-4、critic 3e-4、地形 0.4、模式 7:0.65 / 4:0.20、难 clip boost 5、地形不做参考强制、平地参考强制 0.1、圆盘脚、课程已满），1800 次迭代（到 28000）。`ft_v2` 目录原样保留。门限：`watch_all4.sh ft_v3 4 200`；`ft_v2` 的 26000–27000 做了权重平均（`scripts/eval/average_ckpts.py`）：soup A = 26000/26200/…/27000 六个检查点，soup B = 26500..27000 每 50 次迭代 11 个，soup A 正在评测。发布脚本支持多个 run（`<run>_it<N>.pt`）。
- 2026-10-01 13:11 **权重平均（soup A）**：`ft_v2` 的 26000/26200/26400/26600/26800/27000 六个检查点的权重平均（`scripts/eval/average_ckpts.py`），圆盘脚、留出地形：mode 7 global **83.8%**（25200 是 80.1%，+3.7 pp，p=0.001；成员平均 82.9%）、mode 4 **82.3%**（成员平均 80.5%），0.25–0.31 m 台阶 62.7% → 73.8%（+11 pp，p=0.002）；完整平地门限 BONES −0.08 / +0.04 pp、Ours 0.00 / +0.20 pp（所有检查点里最干净）。结论：同一条轨迹上的检查点权重平均没有障碍，是便宜而稳定的提升；最终交付也用这个办法（`ft_v3` 的最佳窗口平均）。soup A 已发给 MagicSim 闭环 session 复测 0.26 m 上行/下行。
- 2026-10-01 14:30 **提交**：按用户要求把当前最新一版提交到 main：`ScaleTrack/checkpoints/ft_v2_soupA_it26000-27000.pt`（Git LFS，和 `ft_v2_it25200.pt` 并存，README 里有两者的对照表）、`average_ckpts.py`、带校验的 `watch_all4.sh`（取代 `watch_all2.sh`）、评测器的 `--future_idx` 选项、地形模式比例（0.65 / 0.20）的配置、文档。`ft_v3`（从 `ft_v2` 26200 分出的新运行）仍在训练，最终检查点选定后再更新一次。

