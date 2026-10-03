# Terrain fine-tuned checkpoints

The files are the pretrained `humanoid_transformer_m` (`model_22200.pt`) fine-tuned for terrain (stairs 0.05–0.30 m, boxes, rough ground, slopes)
**without touching the network or the deployment interface** (same observations, same export flow, same 8 control modes). They have the original
rsl_rl format (`model_state_dict`, optimizer states, `iter`) and load exactly like `model_22200.pt`.

| file | what it is |
|---|---|
| `ft_v4_soupV4a_it28300-28600.pt` | **final, recommended.** Weight average (`scripts/eval/average_ckpts.py`) of the `ft_v4` checkpoints 28300, 28350, …, 28600 (7 files). `ft_v4` is a short squat-aware fine-tune (600 iterations) of `soupV3d` (the average of the 21 `ft_v3` checkpoints 27000…27999). Highest terrain numbers of all files here, full flat gate and full test sets clean, and the only file with deeper squats than the pretrained model (see the squat table). |
| `ft_v3_it26800.pt` | A single checkpoint of `ft_v3`, a branch of the `ft_v2` run taken at its terrain peak (iteration 26200) with the 25900-26250 settings restored (anchor coefficient 2, actor learning rate 1e-4); full 26-configuration flat gate clean. |
| `ft_v2_soupA_it26000-27000.pt` | Weight average of the `ft_v2` checkpoints 26000, 26200, …, 27000. Superseded by `soupV4a`; the MuJoCo sim-to-sim checks in the ScaleBridge terrain manual were done with it, `soupV4a` has not been through them yet. |
| `ft_v2_it25200.pt` | A single checkpoint of the first run, the one validated first in the MagicSim closed loop (planner + tracker). |

How they were trained (see `scalebfm_terrain_training_plan.md` in the repository root for the full story, the decision log and the incidents):

* 2–4 GPUs, 2048 envs each, from `model_22200`, PPO with a fixed small actor learning rate and a KL anchor to `model_22200` on the flat rehearsal
  samples (so that the flat tracking of all control modes is kept);
* data: 60k flat BONES-SEED / LAFAN clips (rehearsal) + 30k terrain clips recorded from the MagicLoco pi_L v4 terrain policy on IsaacLab terrain
  layouts; terrain step-height curriculum complete, hard (high-step) clips up-weighted (x3 from iteration 23500, x5 from 25900), terrain control
  modes: whole body 55 % -> 65 %, VR-5 30 % -> 20 % (from iteration 25900) plus a few sparse modes;
* the robot uses **MagicSim's foot collision model** (two flat discs per foot, `g1_new.usd`; see `scripts/assets/make_discfeet_usd.py` and
  `SCALETRACK_ROBOT_USD`) from iteration 25050 on (earlier a mix with the original capsule feet); during the early phase a part of the episodes
  ran with deployment-style reference forcing, later only 10 % of the flat episodes;
* `ft_v4` (squat-aware, iterations 28000-28600): the terrain fine-tunes made deep squats 1–5 cm *shallower* than the pretrained model (the pelvis
  stays higher than the reference; the gates cannot see it because their success criterion is a 0.5 m threshold). `ft_v4` adds 54 "anchor-free"
  squat clips: 9 deep-squat trajectories (pelvis down to 0.20 m, several speeds / foot placements) recorded from MagicLoco's standing planner,
  augmented by `scripts/data/make_squat_aug.py` (speed x0.8 / 1.0 / 1.25, random headings). They are drawn for 8 % of the flat episodes and are
  excluded from the KL anchor (`anchor_free_clip_prefix` / `anchor_free_share` of the terrain command); everything else is as in `ft_v3`. Four planner
  squats of other depths (pelvis 0.315 / 0.262 / 0.222 / 0.205 m) were held out for validation.

## Results

All numbers are on MagicSim's foot model, against the pretrained model on the same asset. Terrain = held-out layouts (never trained on, 1500 clips of 20 s
for the quick gate, 8268 clips for the full one), training future offsets, success = no activated link farther than 0.5 m from the reference (world
frame).

Quick gate, seed 0 (1000 BONES + 300 Ours clips per flat configuration):

| | pretrained `model_22200` | `ft_v2_it25200` | `ft_v2_soupA` | `ft_v3_it26800` | **`ft_v4_soupV4a`** |
|---|---|---|---|---|---|
| terrain, mode 7 (whole body), global tracking | 16.3 % | 80.1 % | 83.8 % | 87.1 % | **88.1 %** |
| terrain, mode 4 (VR-5), global tracking | 12.7 % | 80.5 % | 82.3 % | 84.7 % | **85.7 %** |
| terrain, steps 0.25–0.31 m, mode 7 global | 4.0 % | 62.7 % | 73.8 % | – | **81.3 %** |
| terrain, mode 7, local tracking (reference forcing), relative-error success | 17.5 % | 58.8 % | 59.6 % | 62.9 % | **63.7 %** |
| flat, 26 mode x tracking configs, BONES global / local, change of success | – | −0.11 / −0.10 pp | −0.08 / +0.04 pp | −0.13 / +0.10 pp | −0.31 / −0.10 pp |
| flat, Ours global / local, change of success | – | +0.12 / +0.27 pp | +0.00 / +0.20 pp | +0.00 / −0.33 pp | −0.21 / +0.40 pp |

`ft_v4_soupV4a`, three evaluation seeds (each against the pretrained model with the same seed): terrain mode 7 global 88.1 / 86.8 / 88.3 % (mean **87.7 %**,
pretrained 16.0 %), mode 4 global mean **86.1 %** (13.4 %), mode 7 local relative-error success 64.3 % (17.6 %); flat gate (all 26 configurations), mean
change of success: BONES global −0.10 / local +0.20 pp, Ours global −0.04 / local −0.09 pp (the worst single configuration of the quick gates is
Ours local −2.67 pp, on 300-clip subsets whose noise is about ±1.5 pp; on the full Ours set the worst configuration is −0.49 pp).

`ft_v4_soupV4a` on the **full test sets** (10000 BONES-SEED test clips, the 1648 readable Ours clips, all 8268 held-out terrain clips; 13 BONES and 13 Ours
configurations): BONES change of success −0.41 … +0.27 pp (global configurations −0.09 … +0.04), Ours −0.49 … +0.91 pp, mean position errors (MPKPE)
changed by at most 0.3 % (except the global error of Ours local tracking, +1.2 %, see below); terrain mode 7 global **15.3 % → 87.3 %**, mode 4 global **12.3 % → 86.9 %**, local relative-error success 17.1 % → 64.4 % (mode 7)
and 14.0 % → 60.6 % (mode 4). The summarizer flags two things: Ours mode 6 local has a global position error 3.2 % higher (success +0.91 pp; the local
tracking mode has no position feedback, so its global error is drift), and the rotation errors (MPKRE) of mode 4 on terrain are +7 % while the
position errors are −24 % and the success rate +74 pp; neither was investigated further.

Deep squat (the robot is given the reference trajectories of MagicLoco's squat planner, global tracking, MagicSim feet; mean height of the pelvis above the
reference during the hold phase at the four held-out depths 0.315 / 0.262 / 0.222 / 0.205 m, ideal = 0 cm, lower is better):

| | pretrained | `ft_v2_it25200` | `soupV3d` (the start of `ft_v4`) | **`ft_v4_soupV4a`** |
|---|---|---|---|---|
| mode 7 (whole body) | +6.7 cm (5.1 / 6.2 / 7.1 / 8.2) | +7.8 cm | +8.6 cm (7.8 / 8.6 / 8.1 / 10.0) | **+4.0 cm** (3.3 / 3.4 / 4.1 / 5.2) |
| mode 4 (VR-5) | +6.7 cm (4.7 / 6.1 / 7.4 / 8.6) | +11.1 cm | +11.7 cm (9.9 / 11.4 / 12.0 / 13.3) | **+3.1 cm** (2.5 / 2.0 / 3.6 / 4.2) |

Notes
* Success = no activated link farther than 0.5 m from the reference (world frame); local-tracking success uses the error relative to the root,
  because without position feedback the world position drifts. Use global tracking (true or estimated root pose) for terrain.
* The last future offset of the actor is drawn at random in 5..32 frames during training (the `-1` entry of `future_idx`). Fixed values on the terrain
  gate (measured on `ft_v2_it25200`): K=5 (the offsets `[0..5]` of the standard export) 73.9 %, K=10 80.0 %, **K=16 82.5 %**, K=24 81.1 %,
  K=32 80.4 %: use K=16 for stairs if the planner can provide a lookahead frame.
* Weak point that remains: ascending steps of 0.26 m and more. MagicSim closed loop (MagicLoco terrain planner on a ghost G1, ScaleBFM as tracker, mode 7,
  true root pose, 8 randomised draws per cell): no falls for 0.08–0.22 m stairs; 0.26 m up, planner not waiting for the tracker: 6/8 falls with
  `ft_v2_it25200`, 4/8 with the `ft_v3` average (`soupV3a`), 1/8 with either when the planner waits for the tracker (the planner alone: 0/8). A blind
  tracker is probably near its limit here; the next step would be a height-scan input (changes the network input and the deployment interface). The
  closed-loop runs of `soupV4a` (tall steps and the deep-squat planner runs) are in progress in the MagicSim session; this file will be updated.
* Export: use `scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py` on the **original** asset (the kinematic tree is identical, the
  exporter reads the MJCF next to the asset), it needs `torch_tensorrt`.
* Reproduce the gates: `scripts/eval/eval_seed.sh <checkpoint> <tag> <seed> <1 = disc feet>` (needs the disc-feet asset from
  `scripts/assets/make_discfeet_usd.py`, the motion lists and the baseline results of `model_22200`), `scripts/eval/summarize_gate.py compare|terrain`,
  squat probe: `eval_modes.py --trace_out` + `scripts/eval/squat_summary.py`.
