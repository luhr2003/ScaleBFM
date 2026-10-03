# ScaleBFM 地形微调手册（交接手册）

状态日期：2026-10-03 15:10。机器过载卡死，所有评测已暂停；训练早已结束。本手册把“现在手里有什么、怎么用、怎么复现、怎么接着做、哪些没做完”写在一处。完整的决策记录和事故记录在仓库根目录的 `scalebfm_terrain_training_plan.md`，检验点的英文说明和成绩表在 `ScaleTrack/checkpoints/README.md`，部署到实机的流程在 `ScaleBridge/docs/real_robot_deployment_zh.md`。

## 交接清单（接手的人先看这里）

1. **机器现在很卡**：负载平均约 100，5 张 GPU 都被占满（我们的评测、你的 MagicLoco 训练、别的用户的作业），`nvidia-smi` 可能没有响应。**不要马上开新的评测或训练。** 先看 `uptime` 和 `pgrep -af '[-]-future_idx'`（应为空；如果还有处于 `D` 状态的评测进程，等它们自己退出，`kill -9` 对 `D` 状态没有用）。
2. **ScaleBFM 没有训练在跑**，也不需要再训练，除非 7.2 的决策树要求。
3. **最好的检验点已经在 GitHub main**：`ScaleTrack/checkpoints/ft_v4_soupV4a_it28300-28600.pt`（Git LFS）。取回：`git pull && git lfs pull`；校验 `sha256sum` 以 `8069ce97` 开头、以 `c3e98` 结尾。工作区副本和 `recommended.pt` 见第 1 节。
4. **中断在半路的东西**（续跑方法都在 7.1）：
   - 部署偏移 K=5 的完整平地门限：2026-10-03 15:05 被我停掉，已完成的文件保留在 `runs/eval/soupV4a_disc_k5/` 和 `runs/eval/base22200_disc_k5/`；续跑脚本 `ScaleTrack/scripts/eval/chain_k5_final.sh`。
   - MagicSim 的闭环第二批（0.30 m、K=5、广谱场景、再加 8 组抽样）：被打断，等用户同意后由 MagicSim 的 session 续。
5. **接手后最值钱的三件事**：(a) 闭环第二批里的 0.30 m 上/下行，对照 2.5 节 25200 的基线，决定要不要做 7.2 的选项 A 或 B；(b) 把 K=5 平地门限补完，才能说“部署偏移下无退化”；(c) 用 soupV4a 重做 ScaleBridge 的 MuJoCo 台阶测试和实机检查。
6. **不要做的事**：削弱 KL 锚定；用和基线不同的进程结构做平地门限；相信闭环里单独的 0.40 m 骨盆标志；在负载高的时候同时开很多评测；`pkill -f` 一个出现在自己命令行里的模式；动别人的进程。
7. 谁在做什么：MagicSim 闭环的 session（名字 `test`）已暂停，等用户说继续；ScaleBridge 的 session 没人通知过 soupV4a；用户有最终决定权（推送、height scan、是否继续）。

## 0. 一页纸摘要

- **做了什么**：从预训练 `model_22200`（`humanoid_transformer_m`，8 个控制模式）继续训练，网络结构、观测和部署接口都不变，让 G1 能走台阶（0.05–0.30 m）、箱子、粗糙地面和斜坡。平地的 8 个模式和 general tracking 用“平地样本上对 `model_22200` 做 KL 锚定”来保住。
- **训练已经结束**，没有 ScaleBFM 的训练在跑。
- **最终检验点**：`ScaleTrack/checkpoints/ft_v4_soupV4a_it28300-28600.pt`（Git LFS，sha256 `8069ce977b0a1456e852fbd48ce0d8ab8f75218e81600abad2ea6e38176c3e98`），GitHub main 的提交 `f1e3712`。工作区副本 `/home/vcj9002/scalebfm_ws/releases/soupV4a_ft_v4_it28300-28600.pt`，同目录的 `recommended.pt`、`latest.pt` 指向它。它是 `ft_v4` 的 28300、28350、…、28600 七个检查点的权重平均。
- **一句话成绩**（MagicSim 圆盘脚，对照预训练模型）：留出地形 mode 7 全局成功率 16.0% → 87.7%（3 个种子均值），mode 4 13.4% → 86.1%；平地各模式在全测试集（10000 条 BONES + 1648 条 Ours）上变化 −0.49 … +0.91 pp，在噪声带内；深蹲比预训练模型更深（骨盆高于参考 +6.7 → +4.0 cm）；MagicSim 闭环里 0.26 m 上行和下行、0.22 m 上行都是 0/8 真实摔倒（25200 是 6/8、1/8、0/8）。
- **没做完的**（第 7 节有续跑方法）：
  1. 闭环的第二批：0.30 m 上/下行、部署偏移 K=5、广谱场景、0.26 m 再加 8 组抽样，机器卡死时被打断，MagicSim 的 session 在等你同意后再续。
  2. 部署偏移 K=5 的完整平地门限：我这边只跑了一半就停了。
  3. ScaleBridge 的 MuJoCo 和实机还没用 soupV4a 测过。
  4. 文档更新还没推送到 GitHub（见第 7.4 节）。

## 1. 文件在哪里

| 东西 | 位置 |
|---|---|
| GitHub | `github.com-luhr:luhr2003/ScaleBFM.git`，分支 main，最终检验点提交 `f1e3712` |
| 本地仓库 | `/home/vcj9002/magicloco/ScaleBFM`（训练代码在 `ScaleTrack/`） |
| 工作区 | `/home/vcj9002/scalebfm_ws/` |
| Python 环境 | `source /home/vcj9002/scalebfm_ws/envs/scaletrack/bin/activate`（Python 3.11，Isaac Sim 5.1，torch 2.7） |
| 预训练模型 | `ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt` |
| 各次训练的检查点 | `ScaleTrack/logs/rsl_rl/g1_bfm_tracking_exp/{ft_v2,ft_v3,soupV3d_base,ft_v4}/`（`ft_v1` 作废，见 7.5） |
| 发布目录 | `scalebfm_ws/releases/`：`MANIFEST.md`（逐个检验点的门限表）、`manifest.json`、`extra.json`（平均检验点的登记，`recommended: true` 标记最终版）、`soupA/soupV3a/soupV4a_*.pt`、单个检验点 `ft_v2_it*`、`ft_v3_it*`、`ft_v4_it28400.pt` |
| 权重平均 | `scalebfm_ws/soups/`（`soupV4a_28300-28600_step50.pt`、`soupV3d_27000-27999_step50.pt` 等） |
| 评测结果 | `scalebfm_ws/runs/eval/<tag>/`，标签规则见 4.1 |
| 分析脚本 | 仓库里 `ScaleTrack/scripts/eval/closed_loop_falls.py`（按姿态判据重算闭环摔倒）；工作区 `scalebfm_ws/runs/analysis/` 还有 `terrain_features.py`、`descent_analysis.py`、`flips.py`（开环门限里按上/下台阶事件分类的分析，没发现下行特有的差别） |
| 动作数据 | `scalebfm_ws/motions/processed/{bones,lafan,terrain,squat_aug,deepsquat,deepsquat_val}`，列表 `motions/yaml/`（见第 8 节） |
| 圆盘脚资产 | `scalebfm_ws/assets/g1_29dof_discfeet/g1_29dof_discfeet.usda`，生成脚本 `ScaleTrack/scripts/assets/make_discfeet_usd.py` |
| 闭环结果（MagicSim 的 session） | `/home/vcj9002/magicloco/MagicSim/TestOutput/bfm_planner/queue_soupV4a/<格子>/{report.json,trace.npz}`；软连接包 `outputs/wbc_bundle_terrain_soupV4a`（模型 sha256 与上面相同） |
| 脚本 | `ScaleTrack/scripts/{train,eval,data,terrain,assets}`；临时链脚本在 `scalebfm_ws/tmp/` |

## 2. 成绩

所有数字都是 MagicSim 的脚模型（每只脚两个平圆盘，等同 `g1_new.usd`），对照同一资产上的预训练模型。留出地形 = 布局 6、7（训练从没见过），快速门限 1500 条 20 s 的片段，完整 8268 条。成功 = 任何被激活的链接与参考的世界系位置误差从未超过 0.5 m。

### 2.1 开环门限（快速，种子 0）

| | 预训练 | `ft_v2_it25200` | `ft_v2_soupA` | `ft_v3_it26800` | **`soupV4a`** |
|---|---|---|---|---|---|
| 地形 mode 7 全局 | 16.3% | 80.1% | 83.8% | 87.1% | **88.1%** |
| 地形 mode 4 全局 | 12.7% | 80.5% | 82.3% | 84.7% | **85.7%** |
| 0.25–0.31 m 台阶，mode 7 全局 | 4.0% | 62.7% | 73.8% | – | **81.3%** |
| 地形 mode 7 局部（参考强制），相对根成功率 | 17.5% | 58.8% | 59.6% | 62.9% | **63.7%** |
| 平地 26 配置，BONES global / local 成功率变化 | – | −0.11 / −0.10 pp | −0.08 / +0.04 pp | −0.13 / +0.10 pp | −0.31 / −0.10 pp |
| 平地 Ours global / local | – | +0.12 / +0.27 pp | +0.00 / +0.20 pp | +0.00 / −0.33 pp | −0.21 / +0.40 pp |

### 2.2 soupV4a 的三个种子和全测试集

- 三个评测种子（各自对照同种子的预训练模型）：地形 mode 7 全局 88.1 / 86.8 / 88.3%（均值 **87.7%**，预训练 16.0%），mode 4 均值 **86.1%**（13.4%），mode 7 局部相对根成功率 64.3%（17.6%）。平地 26 配置的三种子均值：BONES global −0.10 / local +0.20 pp，Ours global −0.04 / local −0.09 pp。快速门限里最差的单个配置是 Ours local −2.67 pp（300 条子集，噪声约 ±1.5 pp）。
- 全测试集（种子 0）：BONES 13 个配置 −0.41 … +0.27 pp（global −0.09 … +0.04），Ours 13 个配置 −0.49 … +0.91 pp，平均位置误差变化不超过 0.3%（Ours local 的全局误差 +1.2%）。地形全部 8268 条：mode 7 全局 **15.3% → 87.3%**，mode 4 全局 **12.3% → 86.9%**，局部相对根成功率 17.1% → 64.4%（mode 7）、14.0% → 60.6%（mode 4）。
- 汇总脚本标出两处，**没有深究**：Ours mode 6 local 的全局位置误差高 3.2%（同配置成功率 +0.91 pp，local 没有位置反馈，那是漂移）；地形 mode 4 的旋转误差（MPKRE）比预训练高 7%（位置误差低 24%，成功率 +74 pp）。

### 2.3 部署偏移 K=5 的部分结果（已中断）

部署的标准导出用未来帧偏移 `[0,1,2,3,4,5]`（最后一个偏移 K=5）；训练时 K 在 5…32 里随机，所以 K=5 是训练范围的下限。

- 地形 K=5（种子 0，1500 条，全局，对照 2026-09-30 测的预训练 K=5 结果 `runs/eval/base22200_disc_dep`）：soupV4a mode 7 **82.7%**（预训练 16.1%），mode 4 **81.0%**（13.3%）。同一条件下 `ft_v2_it25200` 是 73.9% / 74.2%。
- 平地 K=5：只有 BONES mode 0 global 一项对照完成（−0.10 pp）。`runs/eval/soupV4a_disc_k5/` 里 soupV4a 的 BONES 两部分和地形两部分已完成；`base22200_disc_k5/` 里只有 BONES 第一部分（部分）和地形局部。**完整的 K=5 平地门限没有跑完，不能声称部署偏移下平地无退化**，续跑方法见 7.1。

### 2.4 深蹲

开环探测：机器人跟踪 MagicLoco 站立规划器导出的 4 条留出深度（骨盆最低 0.315 / 0.262 / 0.222 / 0.205 m）的深蹲参考，全局跟踪；保持阶段骨盆高于参考的平均值，理想值 0，越低越好。

| | 预训练 | `ft_v2_it25200` | `soupV3d`（`ft_v4` 的起点） | **soupV4a** |
|---|---|---|---|---|
| mode 7 | +6.7 cm（5.1 / 6.2 / 7.1 / 8.2） | +7.8 cm | +8.6 cm（7.8 / 8.6 / 8.1 / 10.0） | **+4.0 cm**（3.3 / 3.4 / 4.1 / 5.2） |
| mode 4 | +6.7 cm（4.7 / 6.1 / 7.4 / 8.6） | +11.1 cm | +11.7 cm（9.9 / 11.4 / 12.0 / 13.3） | **+3.1 cm**（2.5 / 2.0 / 3.6 / 4.2） |

发现：地形微调会让深蹲变浅 1–5 cm，BONES/Ours 门限看不出来（成功标准是 0.5 m 阈值）；预训练模型自己也有 7–9 cm 的下限，是策略本身的缺口。`ft_v4` 用“不参与锚定的深蹲片段”修了它，mode 7 在 28300 之后进入平台（约 +4 cm）。

闭环（MagicSim，K=8，17 s；跟踪器最低骨盆减规划器影子最低骨盆）：留出深度 +2.1 / +2.6 / +4.1 / +4.8 cm（25200 是 +6.4 / +6.9 / +8.7 / +9.1 cm）；规划器的 0.20 m 深蹲现在能到 0.25 m（25200 只到 0.29 m）。

### 2.5 MagicSim 闭环：台阶

设置：MagicLoco 地形规划器（`ml_terrain`）带一个影子 G1，ScaleBFM 做跟踪器，mode 7，真值根位姿，K=16（偏移 `[0,1,2,3,4,16]`），`MeshPyramidStairs` 踏面 0.30 m，每格 8 组随机抽样，16 s。

**摔倒判据很重要。** 对方的测试程序原来的标志是“骨盆离脚下地面 < 0.40 m”，下台阶时会误报：soupV4a 的骨盆比旧检验点更紧地跟随规划器下台阶前的下蹲，骨盆高度先落到下一级而水平位置还在上一级的踏面上，脚下地面高度多算 0.26 m，出现 0.36–0.39 m 的瞬时凹陷（5/8 个被标成摔倒），但这 8 组全部直立结束、和影子在同一高度。现行判据（`fell_pose`，MagicSim 提交 `dd11109c2`，只对 walk / rpy 配对使用；平地和深蹲仍用 0.40 m）：倾角超过 60°，或骨盆离地低于 0.45 m 持续超过 0.5 s，或结束时骨盆离地低于 0.5 m。“成功”再要求至少完成影子攀爬量的 75%。

| 摔 / 卡住 / 成功（每格 8 组） | `ft_v2_it25200` | `soupV3a` | **soupV4a** |
|---|---|---|---|
| 0.26 m 上行，规划器不等跟踪器 | 6 / 0 / 2 | 4 / 0 / 4 | **0 / 0 / 8** |
| 0.26 m 上行，规划器等跟踪器（leash） | 1 / 1 / 6 | 1 / 0 / 7 | **0 / 0 / 8** |
| 0.26 m 下行，规划器不等 | 1 / 0 / 7 | 2 / 0 / 6 | **0 / 0 / 8** |
| 0.22 m 上行 | 0 / 0 / 8 | 0 / 0 / 8 | **0 / 0 / 8** |

25200 在 0.30 m 和 K=5 上的旧格子（用同一姿态判据重算，**这是 soupV4a 要对照的基线，soupV4a 的这些格子还没有结果**）：

| 25200，真实摔倒 / 8 | K=16 不等 | K=16 leash | K=5 不等 |
|---|---|---|---|
| 0.30 m 上行 | 2 | 2 | 2 |
| 0.30 m 下行 | 3 | 4 | 6 |
| 0.26 m 上行 | 6 | 1 | 7 |
| 0.26 m 下行 | 1 | 4 | 0 |

注意：每格只有 8 组，差一两个不显著；闭环里的大差别（6/8 → 0/8）才是可信的信号。

## 3. 怎么使用检验点

### 3.1 加载和导出

检验点是原始的 rsl_rl 格式（`model_state_dict`、优化器状态、`iter`），加载方式和 `model_22200.pt` 完全相同，网络和观测没变。导出 TensorRT 用 `ScaleTrack/scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py`，**要对原始资产（胶囊脚）导出**（运动学树相同，导出脚本读资产旁边的 MJCF），需要 `torch_tensorrt`；板载（aarch64）重新编译用 `play_export_check_humanoid_transformer_onboard.py`（`--checkpoint --mode_table --metadata --xml_path`）。流程和注意事项见 `ScaleTrack/README.md` 第 5 节和 `ScaleBridge/docs/real_robot_deployment_zh.md` 的 2.2 节。TensorRT 文件和平台、显卡绑定，不能互相拷贝，编译约 12 分钟。

### 3.2 部署时的选择

- **前瞻帧 K**：训练时最后一个偏移 K 在 5…32 里随机。台阶上规划器如果能给前瞻帧，用 **K=16**（25200 上的开环对照：K=5 73.9%、K=10 80.0%、K=16 82.5%、K=24 81.1%、K=32 80.4%）；标准导出的 `[0..5]` 即 K=5，soupV4a 的地形成功率仍有 82.7%（mode 7，见 2.3），但闭环 K=5 的结果还没测。
- **跟踪方式**：地形上要用 **global** 跟踪（真值或估计的根位姿）。局部（参考强制）跟踪没有位置反馈，世界系位置会漂，地形上相对根成功率只有约 64%。
- **控制模式**：地形训练的模式比例是 mode 7（全身 14）65%、VR-5 20%、其余稀疏模式各 5%；闭环验收主要看 mode 7，其次 mode 4。
- **脚模型**：训练用的是 MagicSim 的圆盘脚。原始胶囊脚上的成功率更高（约 +10 pp），但那不是 MagicSim 里的脚，别拿来比。
- **能力边界**：0.26 m 以内的台阶在闭环里 8 组都没摔；0.30 m 没有 soupV4a 的结果；开环门限里 0.25–0.31 m 的台阶仍有约 19% 的片段失败。别在没有保护的情况下做高台阶。

### 3.3 在 MagicSim 里用

MagicSim 的 session 已经用 soupV4a 编了一个 bundle（`outputs/wbc_bundle_terrain_soupV4a`，配置 `configs/scalebfm_mode7.json`，与 25200 的 bundle 用同一份 metadata、mode_table 和 XML）；对应文档在 MagicSim 的分支 `feat/g1-scalebfm-terrain-decoupled` 的 `docs/wbc_planner_scalebfm.md`。续跑闭环用 `.../TestOutput/planner_scalebfm/_archive_20261003/tools/soupV4a_queue2.py`（见 7.1）。

## 4. 怎么评测

### 4.1 约定

- 基线：`runs/eval/base22200_disc`（圆盘脚，快速门限，种子 0/1/2 都在这里）、`base22200_disc_full`（全测试集）、`base22200_disc_dep`（K=5，只有地形和 3 个模式的 BONES）、`base22200`（胶囊脚）。
- 标签：`<检验点>_disc` = 圆盘脚快速门限，`_disc_full` = 全测试集，`_disc_k5` = 部署偏移，`squat8/<标签>` = 深蹲探测。
- **只做同结构的对照**：平地门限固定为两个进程，`MODES="0 1 2 3"` 和 `MODES="4 5 6 7"`（与基线一致）。一个进程里的第二个模式在几条脆弱的 BONES 片段上会少约 1.5 pp，用别的结构和基线比会得到假的“退化”。
- 噪声量级：BONES 约 ±0.3 pp，Ours 约 ±1.5 pp（300 条子集），地形单个种子约 ±1 pp。小于噪声的差别用多个种子（`eval_seed.sh` 的种子 0/1/2）再判断。
- 完整级别的 Ours 测试集有一个损坏的文件（`LegsApart_LegsApart_SW`），所以 `run_gate.sh` 的完整级别用 `motions/yaml/test_ours_clean.yaml`（1648 条）；快速子集不变。

### 4.2 命令

```bash
cd /home/vcj9002/magicloco/ScaleBFM/ScaleTrack
# 一个检验点、一个种子的完整快速门限（平地 26 配置 + 留出地形全局/局部），带启动错峰、结果校验和重试
scripts/eval/eval_seed.sh <检验点绝对路径> <标签> <种子> 1 <gpu>            # 1 = 圆盘脚
EVAL_ARGS="--future_idx 0 1 2 3 4 5" scripts/eval/eval_seed.sh <路径> <标签>_k5 0 1 <gpu>   # 部署偏移 K=5
# 汇总（对照目录 新目录），每个种子一个文件
python scripts/eval/summarize_gate.py compare runs/eval/base22200_disc runs/eval/<标签>
python scripts/eval/summarize_gate.py terrain runs/eval/base22200_disc runs/eval/<标签>
python scripts/eval/summarize_gate.py compare <基线目录> <新目录> --level full      # 全测试集
# 训练中的门限队列（每 200 次迭代自动门限；watch_all5.sh 另外做深蹲探测）
scripts/eval/watch_all5.sh <运行名> <gpu> 200
# 只评一个任意检验点文件（例如权重平均）
scripts/eval/watch_all5.sh <运行名> <gpu> --eval <标签> <检查点文件>
# 深蹲探测（8 条规划器深蹲，输出骨盆高于参考的厘米数）
scripts/eval/squat_probe8.sh <gpu> <标签> <检验点>
# 权重平均
python scripts/eval/average_ckpts.py --ckpts model_28300.pt model_28350.pt ... --out soup.pt
```

`eval_seed.sh` 要求评测 GPU 至少有 7 GB 空闲才启动每个进程，同一块卡上的评测进程不超过 6 个，失败后等 1 分钟重试，最多 4 次。

### 4.3 闭环评测

闭环由 MagicSim 的 session 跑（见 3.3）。重算摔倒：

```bash
python scripts/eval/closed_loop_falls.py <格子名> <队列目录名> [<队列目录名> ...]
# 例如 closed_loop_falls.py down_h26_open_K16 queue queue_soupV3a queue_soupV4a
```

先看 trace 里的结束姿态和倾角，再相信任何“摔倒”标志；标志在两个方向上都可能出错。

## 5. 怎么接着训练

### 5.1 配方的演进

| 运行 | 起点 | 要点 |
|---|---|---|
| `ft_v1` | `model_22200` | **作废**：多卡参数加载把 1–3 号卡的参数置零（5.3 第 1 条） |
| `ft_v2` | `model_22200` | 4 卡，锚定系数 2、actor 学习率 1e-4（固定）、critic 3e-4、地形占 0.4；地形课程 800 次迭代放开；25050 起换成 MagicSim 圆盘脚；难（高台阶）片段加权；26250 起改成锚定 5、学习率 6e-5，结果地形成功率单调下降（那个改动的理由是评测假象），所以停掉 |
| `ft_v3` | `ft_v2` 的 26200（峰值） | 恢复 25900–26250 的设置：锚定 2、学习率 1e-4、难片段 boost 5、模式 7:0.65 / VR-5:0.20、地形不做参考强制、平地参考强制 0.1、圆盘脚、课程已满，到 28000 |
| `soupV3d` | `ft_v3` 的 27000…27999 每 50 次迭代 21 个检查点 | 权重平均，地形 mode 7 88.1%，平地干净；深蹲比预训练更浅 |
| `ft_v4` | `soupV3d`（暂存为 `soupV3d_base/model_27999.pt`） | 加 54 条“不参与锚定”的深蹲片段（占平地抽样 8%），其余同 `ft_v3`；28000–28600，共 600 次迭代，之后进入平台，停训 |
| **`soupV4a`** | `ft_v4` 的 28300…28600 | 7 个检查点权重平均，最终检验点 |

规律：同一条轨迹上的检查点做权重平均便宜而稳定，几乎总是不比最好的单个检查点差，还能去掉单个检查点在脆弱片段上的抖动。

### 5.2 启动命令（`ft_v4` 原样）

```bash
cd /home/vcj9002/scalebfm_ws
SCALETRACK_ROBOT_USD=/home/vcj9002/scalebfm_ws/assets/g1_29dof_discfeet/g1_29dof_discfeet.usda \
BASE_RUN=soupV3d_base BASE_CKPT=model_27999.pt FREEZE=10 ENVS=2048 ITERS=1500 SAVE=50 \
TERRAIN_FRAC=0.4 ANCHOR=2 ACTOR_LR=1e-4 CRITIC_LR=3e-4 \
EXTRA="env.commands.motion.terrain_curriculum_steps=0 env.commands.motion.local_forcing_prob_terrain=0.0 env.commands.motion.local_forcing_prob_flat=0.1 env.commands.motion.terrain_hard_boost=5.0 env.commands.motion.anchor_free_clip_prefix=sq_ env.commands.motion.anchor_free_share=0.08" \
nohup /home/vcj9002/magicloco/ScaleBFM/ScaleTrack/scripts/train/launch_finetune.sh ft_v4 1,2,4 \
  /home/vcj9002/scalebfm_ws/motions/yaml/train_squat.yaml > tmp/train_ft_v4.log 2>&1 &
```

（实际在 28100 处用 `1,2` 重启成 `1,2,4`，`ITERS=1500` 本来要到 29500，在 28600 停掉。）续训用 `scripts/train/resume_ft.sh <运行名> <gpu列表> [model_N.pt]`，它会恢复地形课程的位置。`launch_finetune.sh` 的环境变量：`ENVS ITERS ANCHOR ACTOR_LR CRITIC_LR FREEZE SAVE TERRAIN_FRAC BASE_RUN BASE_CKPT EXTRA`。

训练监控的东西：每 10 次迭代打印各卡的平均回报，每 50 次迭代重新广播参数并报告漂移；日志里“平地”和“地形”（含不参与锚定的深蹲片段）分开记。

### 5.3 必须遵守的规矩（都是踩过的坑）

1. **多卡**：这台机器上 GPU→GPU 的拷贝会静默返回零，NCCL 的 P2P 也会死锁。多卡必须 `NCCL_P2P_DISABLE=1`，参数逐个张量广播，加载检查点时 `map_location` 用各卡自己的设备，并做跨卡校验和。`launch_finetune.sh` 已经做了这些，别绕开。
2. **共享内存动作库**：多卡启动前要 `rm -f /dev/shm/shared_motionlib_*`（脚本已做），否则会静默用上上一次的库。动作库缓存按片段列表的 md5 命名，片段列表一变就要重建，约 15 分钟；进程组超时设为 120 分钟。
3. **不要削弱 KL 锚定**。锚定系数 2 和 actor 学习率 1e-4 是防止遗忘的核心；锚定 5 + 学习率 6e-5 的试验证明更强的锚定限制了地形学习。平地环境与原训练逐项一致，才能保住平地。
4. **GPU 共享**：GPU 0/1/3 常被 MagicLoco 训练占（每个作业约 27 GB），另有其他用户的作业；训练 OOM 过一次。先看 `nvidia-smi`，不要杀别人的进程。
5. **后台任务**：不要对包含自己命令行的模式用 `pkill -f`；用 PID 或 `[x]xx` 括号写法。等待训练完成时先确认训练进程还在，停滞阈值 30 分钟。
6. **改脚本不要原地改**正在运行的 shell 脚本（bash 会增量读取），用 cp + mv。
7. Isaac Sim 脚本关闭时可能挂几分钟：写完输出后 `os._exit(0)`。

## 6. 出问题怎么办

| 现象 | 原因 / 对策 |
|---|---|
| 门限结果文件缺了，但脚本打印 `GATE_DONE` | 多个 Kit 进程同时启动或 GPU 显存不足，报 `Failed to create simulation view backend`。用 `eval_seed.sh`（错峰 15 s、等 7 GB 空闲、校验文件、重试）；每卡评测进程 ≤ 6 个 |
| 平地 BONES 某个 mode 掉 1.5 pp | 评测进程结构和基线不同（第二个模式在同一进程里会少）。按 4.1 的两进程结构重做 |
| 完整 Ours 评测整个进程死掉 | 损坏文件 `LegsApart_LegsApart_SW`，用 `test_ours_clean.yaml` |
| 闭环“摔倒”但机器人其实直立 | 下台阶的 0.40 m 骨盆阈值误报，用姿态判据（2.5 节、4.3 节） |
| 地形局部（参考强制）成功率很低 | 没有位置反馈会漂；用相对根成功率 SuccL，部署用 global |
| 机器卡死 / 命令超时 / `nvidia-smi` 无响应 | 负载平均可到 100–250（别人的作业加我们的评测），GPU 都满。先停自己的评测（见 7.1 第 2 条），被杀的 Kit 进程可能长时间处于 `D` 状态（SIGKILL 也不退出），等内核调用返回，不要重启机器上别人的东西 |
| 训练突然 NaN | 先看是不是多卡参数没同步（各卡校验和），再看梯度保护计数（`num_skipped_updates`） |
| 两个脚本都在改同一个 `releases/` | 发布脚本 `tmp/publish_releases.py --loop` 已停；`recommended.pt` 现在由 `extra.json` 里 `"recommended": true` 固定 |

## 7. 没做完的事和下一步

### 7.1 被打断的验证，怎么续

1. **闭环第二批**（MagicSim 的 session）：0.30 m 上/下行（V4a、V3a）、0.26 m 上/下行 K=5（V4a）、25200 时的广谱场景（ml_terrain h08/h12/h18/h22、ml_unified skills、homie 平地）、0.26 m 再加 s8–s15 抽样（V4a、25200、V3a）。第一批的 0.30 m 两个格子被打断，`queue_soupV4a/*h30*` 目录是不完整的，续之前先删掉；续跑脚本 `/home/vcj9002/magicloco/MagicSim/TestOutput/planner_scalebfm/_archive_20261003/tools/soupV4a_queue2.py`（跳过已完成的格子）。**该 session 说等你同意后才继续。**
2. **K=5 完整平地门限**（我这边）：2026-10-03 15:05 我停掉了它。残留的 6 个评测进程处于 `D` 状态，等它们退出（`pgrep -af '[-]-future_idx'` 为空）后：
   ```bash
   nohup /home/vcj9002/magicloco/ScaleBFM/ScaleTrack/scripts/eval/chain_k5_final.sh > /home/vcj9002/scalebfm_ws/tmp/chain_k5_final.log 2>&1 &
   ```
   它调用 `eval_seed.sh`，只补缺失的部分（soupV4a 在 GPU 3，预训练在 GPU 2，写死在脚本里，可改；最低优先级运行），完成时打印 `K5_DONE` 和汇总。机器负载平均高于 100 时不要开。
3. **ScaleBridge 的 MuJoCo / 实机**：用 soupV4a 重新编译（`ScaleBridge/docs/real_robot_deployment_zh.md` 的 2.2 节），按 `terrain_localization_manual_zh.md` 的 6.2 节重做台阶测试。手册里现有的数字都是 soupA 的。

### 7.2 等结果之后怎么决定

- **0.30 m 闭环**对照 25200 的基线（2.5 节第二张表）：
  - soupV4a 持平或更好，且 K=5 平地无退化 → 地形部分收工。
  - 0.30 m 仍然有 ≥ 3/8 摔倒 → 选项 A：从 soupV4a 继续微调，加重 ≥ 0.28 m 台阶片段（提高 `terrain_hard_boost`、加大下行高台阶的采样），保持锚定 2、学习率 1e-4 和深蹲片段，不超过 600 次迭代，再做权重平均并重跑完整门限；选项 B：Stage B 加 height scan（给 actor 加 187 射线高度扫描输入，新输入零初始化，改网络输入和部署接口），**需要用户决定**。
- **K=5 平地门限**出现退化 → 先查是哪个模式、哪个种子，再多跑两个种子；如果确实是 K=5 特有的，在 `ft_v4` 的平地部分里把 K 的分布向 5 倾斜再微调。
- **深蹲 mode 7 的 +4 cm 下限**：想再压低需要更多样的深蹲数据（现在只有 9 条规划器轨迹增强出的 54 条），不是继续训同一批数据。

### 7.3 已知的小问题

- 开环门限里 0.25–0.31 m 台阶约 19% 的片段失败（20 s 全局跟踪）。
- 地形局部跟踪相对根成功率约 64%。
- 地形 mode 4 旋转误差 +7%、Ours mode 6 local 的全局位置误差 +3.2% 被汇总脚本标出，没深究。
- 只用 mode 7 做了闭环验收；mode 4 的闭环没跑。

### 7.4 提交状态

GitHub main 在 `f1e3712`：包含最终检验点、训练评测脚本、检验点 README、计划文档和两本 ScaleBridge 手册里的一句更新说明。**之后的改动都只在本地**：检验点 README 的闭环表、计划文档的闭环条目、`eval_seed.sh` 的 `EVAL_ARGS` 参数，以及本手册。是否推送由你决定。论文 PDF 和被取代的监视器脚本（`watch_all.sh`、`watch_all3.sh`、`watch_gate*.sh`）一直没有提交。

### 7.5 其他人和进程

- MagicSim 闭环的 session（名字 `test`）：已按你的要求暂停，等你说继续。
- ScaleBridge / MuJoCo 的 session：之前的名字 `scalebfm-dd` 的连接已失效，没有人告诉它 soupV4a 的存在。
- 你自己的 MagicLoco 训练（`Magicloco-WaypointWBC-G1-V7-Stab`）和别人的作业占着 GPU 和 CPU。

## 8. 附录

### 8.1 数据

| 文件 | 内容 |
|---|---|
| `motions/yaml/train_all.yaml` | 60,040 条平地（60k BONES + LAFAN）+ 30,000 条地形（布局 0–5） |
| `motions/yaml/train_squat.yaml` | 上面 + 54 条 `sq_*` 深蹲增强片段（`motions/processed/squat_aug`，9 条规划器深蹲 × 速度 0.8/1.0/1.25 × 2 种朝向，`scripts/data/make_squat_aug.py`） |
| `motions/yaml/test_bones.yaml` / `test_ours.yaml` / `test_ours_clean.yaml` | 官方测试集（10000 条 BONES-SEED；Ours 1649 条，clean 版 1648 条） |
| `motions/yaml/eval_terrain_test.yaml` + `clip_meta_test.json` | 留出地形 8268 条（布局 6、7）；快速门限用其中固定的 1500 条 |
| `motions/yaml/deepsquat_all8.yaml`、`deepsquat_val.yaml` | 深蹲探测的 8 条（4 条训练深度 + 4 条留出深度，留出的从未训练） |

地形片段来自 MagicLoco π_L v4 地形策略在 IsaacLab 地形布局上的 rollout（共约 8.3 万条、455 小时），录制与打包脚本在 `ScaleTrack/scripts/terrain/` 和 `scripts/data/`。

### 8.2 代码改动点

- `ScaleTrack/source/my_rsl_rl/.../ppo.py`：KL 锚定（`anchor_coef`，只对 `env_group == 0` 的平地样本）、actor 冻结、非有限梯度保护、多卡广播与校验和。
- `.../runners/on_policy_runner.py`：分组（平地 / 地形）日志，各卡回报报告，周期性重广播。
- `.../tracking/mdp/commands_terrain.py`（`LayoutMotionCommand`）：按组采样、地形课程、难片段加权（`terrain_hard_boost`）、部署式参考强制（`local_forcing_prob_*`）、不参与锚定的片段（`anchor_free_clip_prefix`、`anchor_free_share`）。
- `.../tracking/mdp/observations_terrain.py`：critic 用的真实离地高度，`env_group`（0 平地 / 1 地形 / 2 不参与锚定的深蹲片段）。
- `.../config/g1_29dof/flat_env_cfg.py`：`SCALETRACK_ROBOT_USD`（换机器人资产）；`terrain_env_cfg.py`：地形模式比例 `TERRAIN_MODE_PROBS`。
- `ScaleTrack/scripts/eval/eval_modes.py`：论文协议的评测器，加了 `--future_idx`、`--trace_out`、`--video_dir`、相对根成功率 SuccL。

### 8.3 决策记录

完整的时间线（含每个决策的理由和每次事故）在 `scalebfm_terrain_training_plan.md` 的附录“执行记录与已确定的决策”。
