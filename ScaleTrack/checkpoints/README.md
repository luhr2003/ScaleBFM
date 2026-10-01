# Terrain fine-tuned checkpoints

Both files are the pretrained `humanoid_transformer_m` (`model_22200.pt`) fine-tuned for terrain (stairs 0.05–0.30 m, boxes, rough ground, slopes)
**without touching the network or the deployment interface** (same observations, same export flow, same 8 control modes). They have the original
rsl_rl format (`model_state_dict`, optimizer states, `iter`) and load exactly like `model_22200.pt`.

| file | what it is |
|---|---|
| `ft_v2_soupA_it26000-27000.pt` | **recommended for terrain.** Weight average of the run's checkpoints at iterations 26000, 26200, 26400, 26600, 26800 and 27000 (`scripts/eval/average_ckpts.py`). |
| `ft_v2_it25200.pt` | A single checkpoint of the same run, the one validated so far in the MagicSim closed loop (planner + tracker). |

How they were trained (see `scalebfm_terrain_training_plan.md` in the repository root for the full story, the decision log and the incidents):

* 4 GPUs, 2048 envs each, from `model_22200`, PPO with a fixed small actor learning rate and a KL anchor to `model_22200` on the flat rehearsal
  samples (so that the flat tracking of all control modes is kept);
* data: 60k flat BONES-SEED / LAFAN clips (rehearsal) + 30k terrain clips recorded from the MagicLoco pi_L v4 terrain policy on IsaacLab terrain
  layouts; terrain step-height curriculum complete, hard (high-step) clips up-weighted (x3 from iteration 23500, x5 from 25900), terrain control
  modes: whole body 55 % -> 65 %, VR-5 30 % -> 20 % (from iteration 25900) plus a few sparse modes;
* the robot uses **MagicSim's foot collision model** (two flat discs per foot, `g1_new.usd`; see `scripts/assets/make_discfeet_usd.py` and
  `SCALETRACK_ROBOT_USD`) from iteration 25050 on (earlier a mix with the original capsule feet); during the early phase a part of the episodes
  ran with deployment-style reference forcing, in the soup's iterations only 10 % of the flat episodes.

Gate results against the pretrained model on the same (MagicSim) foot model — quick gates, 1000 BONES + 300 Ours clips per configuration, held-out
terrain layouts (never trained on, 1500 clips of 20 s), training future offsets:

| | pretrained `model_22200` | `ft_v2_it25200` | `ft_v2_soupA_it26000-27000` |
|---|---|---|---|
| terrain, mode 7 (whole body), global tracking, success | 16.3 % | 80.1 % | **83.8 %** |
| terrain, mode 4 (VR-5), global tracking, success | 12.7 % | 80.5 % | **82.3 %** |
| terrain, steps 0.25–0.31 m, mode 7 global | 4.0 % | 62.7 % | **73.8 %** |
| terrain, mode 7, local tracking (reference forcing), relative-error success | 17.5 % | 58.8 % | 59.6 % |
| flat, 26 mode x tracking configurations (BONES + Ours): change of success rate, BONES global / local | – | −0.11 / −0.10 pp | −0.08 / +0.04 pp |
| flat, Ours global / local | – | +0.12 / +0.27 pp | +0.00 / +0.20 pp |

The flat changes are within the evaluation noise (about ±0.3 pp for BONES, ±1.5 pp for Ours): no regression of any control mode.

Notes
* Success = no activated link farther than 0.5 m from the reference (world frame); local-tracking success uses the error relative to the root,
  because without position feedback the world position drifts. Use global tracking (true or estimated root pose) for terrain.
* The last future offset of the actor is drawn at random in 5..32 frames during training (the `-1` entry of `future_idx`). Fixed values on the terrain
  gate (25200): K=5 (the offsets `[0..5]` of the standard export) 73.9 %, K=10 80.0 %, **K=16 82.5 %**, K=24 81.1 %, K=32 80.4 %: use K=16 for stairs
  if the planner can provide a lookahead frame.
* MagicSim closed loop (MagicLoco terrain planner on a ghost G1, ScaleBFM as tracker, `ft_v2_it25200`, mode 7, true root pose): no falls for
  0.08–0.22 m stairs; at 0.26 m up the tracker is still the weak link when the planner does not wait for it (6/8 falls open loop, 1/8 with the
  planner waiting). The soup is being re-tested on those cells.
* Export: use `scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py` on the **original** asset (the kinematic tree is identical, the
  exporter reads the MJCF next to the asset), it needs `torch_tensorrt`.
