# Terrain fine-tuned checkpoint

`ft_v2_it25200.pt` — the pretrained `humanoid_transformer_m` (`model_22200.pt`) fine-tuned for terrain (stairs 0.05–0.30 m, boxes, rough
ground, slopes) **without touching the network or the deployment interface** (same observations, same export flow, same 8 control modes).
The checkpoint has the original rsl_rl format (`model_state_dict`, optimizer states, `iter`) and loads exactly like `model_22200.pt`.

How it was trained (see `scalebfm_terrain_training_plan.md` in the repository root for the full story and the decision log):

* 4 GPUs, 2048 envs each, from `model_22200`, PPO with a fixed small actor learning rate and a KL anchor to `model_22200` on the flat
  rehearsal samples (so that the flat tracking of all control modes is kept);
* data: 60k flat BONES-SEED / LAFAN clips (rehearsal) + 30k terrain clips recorded from the MagicLoco pi_L v4 terrain policy on IsaacLab
  terrain layouts; the terrain step-height curriculum is complete and hard (high-step) clips are up-weighted;
* the robot uses **MagicSim's foot collision model** (two flat discs per foot, `g1_new.usd`; see `scripts/assets/make_discfeet_usd.py` and
  `SCALETRACK_ROBOT_USD`), 25 % / 15 % of the terrain / flat episodes run with deployment-style reference forcing.

Gate results against the pretrained model on the same (MagicSim) foot model — quick gates, 1000 BONES + 300 Ours clips per configuration, held-out
terrain layouts (never trained on, 1500 clips of 20 s):

| | pretrained `model_22200` | `ft_v2_it25200` |
|---|---|---|
| terrain, mode 7 (whole body), global tracking, success | 16.3 % | **80.1 %** |
| terrain, mode 4 (VR-5), global tracking, success | 12.7 % | **80.5 %** |
| terrain, mode 7, local tracking (reference forcing), relative-error success | 17.5 % | 58.8 % |
| flat, 26 mode x tracking configurations (BONES + Ours), change of success rate | – | BONES global −0.11 pp / local −0.10 pp, Ours global +0.12 pp / local +0.27 pp (noise about ±0.3 pp / ±1.5 pp: no regression) |

Notes
* Success = no activated link farther than 0.5 m from the reference (world frame); local-tracking success uses the error relative to the root, because
  without position feedback the world position drifts.
* Export: use `scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py` on the **original** asset (the kinematic tree is identical, the
  exporter reads the MJCF next to the asset), it needs `torch_tensorrt`.
