# ScaleBFM on terrain: the problem and the plan (2026-09-28)

## TL;DR

ScaleBFM (bundle `wbc_bundle_20260917_profiles_002`, checkpoint `model_22200`, mode 7) reproduces data collected with
the decoupled MagicLoco / HTD controllers on flat ground (31/32 clips, 0 falls), but it cannot walk over stairs, boxes
or rough ground: 0/24 open-loop replays, and 0/14 closed-loop runs even with MagicLoco as a terrain-aware planner feeding
it. SONIC v1.1 fails identically (0/4). Both trackers were trained for flat ground and have no terrain input; the fix is
training, not integration. Plan: fine-tune ScaleBFM on terrain with MagicLoco-generated motions while rehearsing its
original skills, gated by a flat-ground regression suite.

## 1. What works and what does not

| Data re-tracked by ScaleBFM | Result |
|---|---|
| Two real Collect LocoGrasp episodes (HTD, MagicLoco) | 2/2, root error 1–8 cm |
| Walks (MagicLoco, HTD), HTD squat-walk to torso 0.35 m | 22/22, 0 falls |
| MagicLoco skills (squat, kneels, place) | 7/8; splits not reproduced; deep poses 3–7 cm shallower (2–4 cm with pelvis-height feedback) |
| Terrain: slopes, standing on boxes/slopes/rough, feet on two steps | 3/4, 5/6, 4/6 |
| **Terrain: walking over stairs, boxes, rough ground** | **0/24** |

## 2. Where exactly the problem is

### 2.1 The failure, step by step (0.08 m stairs, MagicLoco ghost as planner, ScaleBFM tracking it 5 frames behind)

| t (s) | MagicLoco (the plan) | ScaleBFM robot |
|---|---|---|
| 2.9–3.4 | walks on the pit floor, pelvis upright (±3°) | follows the pelvis within 5–8 cm, but pelvis pitched **back 6–10°** and feet trailing |
| 3.46–3.64 | left foot swings 0.37 → 0.55 m and lands **on the first tread** (ankle z 0.115) | lifts the left foot to the same height (0.115) but moves it only **0.31 → 0.35 m**, then puts it back on the floor |
| 3.64–3.76 | right foot follows onto the stairs | pelvis pitches **−3.5° → +29°** within 0.2 s |
| 3.9–4.1 | keeps climbing | right foot's big swing lands on the tread too late; pitch +46° → down |

The same step fails in every run: the foot is lifted high enough (0.11–0.15 m for a 0.08 m riser) but advanced only
4–15 cm instead of the planned 27–35 cm, lands on the riser edge or the floor, slips back, and the robot falls forward.
Open-loop replays of the recorded stairs clips fail at the same riser (3.7–4.0 s on all four step heights).

### 2.2 The underlying behaviour: ScaleBFM plays its own gait

Measured on the replays, with the recorded motion as reference:

| Recorded motion | Recorded swing height | ScaleBFM swing height | ScaleBFM pelvis lag at 0.5 m/s |
|---|---|---|---|
| MagicLoco terrain walker, flat/slope part | 0.155 m | 0.10 m | 10–12 cm |
| MagicLoco stairs-up | 0.18–0.39 m | 0.08–0.15 m | 8–10 cm |
| MagicLoco flat walk | 0.10 m | 0.04–0.08 m | 3–8 cm |
| HTD shuffle | 0.02–0.03 m | 0.035–0.07 m (lifts *more*) | 9–12 cm |

ScaleBFM follows the root trajectory well (flat root error 2–9 cm) but chooses its own step length and swing height,
about half of what was recorded. It also holds the pelvis 5–11° further back than MagicLoco's references in mode 7.
On flat ground none of this matters, because any foothold is fine. On stairs the footholds are fixed by the terrain, and
a step that comes up short lands on the edge. SONIC v1.1 shows the same compression (about half the step length and
lift on the flat pit floor) and fails at the same riser.

### 2.3 What has been ruled out

| Suspect | How it was excluded |
|---|---|
| Integration / harness bug | the same harness replays 31/32 flat clips; gains, FK, joint order and offsets match the reference adapter |
| Different terrain in replay | terrain mesh hash identical between recording and replay (seeded generator fix) |
| Absolute height on elevated tiles | ScaleBFM's input is the goal links relative to the measured pelvis; absolute height cancels |
| Open-loop drift | closed loop with the plan waiting for the robot (leash), and replay re-anchored to the measured pelvis: both 0 |
| Too little look-ahead | far keyframe 15/25/32 frames (0.3–0.64 s; training samples 5–32, we had fed 5): 0/4; SONIC with 0.9 s: 0/4 |
| Plan too fast / steps too long | MagicLoco planner at 0.25 and 0.35 m/s: 0/4 |
| Too many constraints (knees, torso) | mode 4 (pelvis, hands, feet only): 0 |
| Foot target not ambitious enough | pushing the foot targets by the tracking error: 0/4 (overshoot and forward fall) |
| Robots interfering | every variant on its own identical tile (co-located robots do collide; that run was discarded) |

### 2.4 Root cause

The tracker has never learned to put a foot down on a surface higher than the one it stands on. ScaleTrack trains on a
flat plane (`tracking_env_cfg.py`: `terrain_type="plane"`; the critic's root height is commented "only applicable when
no terrain"), and the actor has no terrain input. It reproduces the root motion with its own flat-ground stepping, which
cannot hit terrain-fixed footholds. Standing on uneven ground works because no new foothold is needed. This is a
training gap, so it has to be fixed in training (§3).

### 2.5 All runs (0.08 m steps, the lowest height; MagicLoco ghost on an identical tile as the planner)

| Variant | ScaleBFM climbs | Notes |
|---|---|---|
| Open-loop replay of recorded stairs clips (L0–L3) | 0/4 | falls at the first riser, 3.7–4.0 s |
| Closed loop: plan waits for the tracker (leash), open | 0/2 | |
| Planner speed 0.25 / 0.35 m/s, mode 4 | 0/4 | falls later, never climbs |
| Foot-target feedback (xy, xyz, mode 4) | 0/4 | best one climbs one step, then lunges and falls |
| Far keyframe 15 / 25 / 32 frames (0.3–0.64 s look-ahead; training samples 5–32, we had fed 5) | 0/4 | |
| Replay re-anchored to the measured pelvis | 0/4 | |
| SONIC v1.1 as the tracker (0.9 s look-ahead) | 0/4 | same step compression |

Evidence: `TestOutput/bfm_replay/`, `TestOutput/bfm_planner/`; details in
[decoupled_wbc_integration_plan.md §10–11](decoupled_wbc_integration_plan.md). Harnesses:
`scripts/wbc/replay_with_scalebfm.py`, `scripts/wbc/magicloco_planner_scalebfm.py`.

## 3. Plan: terrain fine-tuning without losing existing skills

Training code: `/home/magics/magicsim/ScaleBFM/ScaleTrack` (IsaacLab + PPO; actor lr 2e-5, critic 1e-3, 8192 envs,
6M-parameter transformer; our checkpoint stores both optimizer states, so it can resume). The original training motions
(AMASS, LAFAN, OMOMO, SnapMoGen, FineDance, Embody3D, GRAB, 100STYLE, BONES) are **not** on this machine.

| Phase | Work | Output | Estimate |
|---|---|---|---|
| 1. Baseline | Separate env for ScaleTrack (Isaac Sim 5.1, IsaacLab `18c7c58`); evaluate `model_22200` on a flat benchmark (examples + rebuilt subset + our 32 flat clips) | the yardstick for forgetting | 1–2 days |
| 2. Data | Terrain: MagicLoco rollouts on stairs up/down (0.05–0.30 m), boxes, rough, slopes, many speeds, stops and stances, each with its terrain tile (seeded generator). Rehearsal: rebuild LAFAN / 100STYLE / BONES (+ AMASS if licensed) via ScaleRetarget | paired terrain clips + flat rehearsal set | 2–4 days |
| 3. Environment | Generator terrain instead of the plane; each terrain clip placed on its own tile; critic height above ground + privileged height map; actor unchanged at first (foot targets already encode footholds). If that plateaus: a height-map token for the actor, zero-initialised so the model starts unchanged | ScaleTrack terrain task | 3–5 days |
| 4. Fine-tune | From `model_22200`; about 60 % flat rehearsal / 40 % terrain with a step-height curriculum; penalty for drifting from the original actor on flat clips; low actor lr; all 8 modes; stop on flat regression | terrain-capable checkpoint | 3–5k iterations; hours on a dedicated GPU |
| 5. Validate in MagicSim | Flat replay suite must stay 31/32; terrain replay suite; MagicLoco-planner stairs test at 0.08–0.26 m | go / no-go | 1 day |

Fallback if forgetting is hard to control: train one fresh student from two teachers (the original ScaleBFM on flat
motions, a terrain-tuned copy on terrain) with DAgger.

**Risks.** Rebuilding the rehearsal data (dataset licences); blind tracking may cap performance on edges (then the
height-map token); ScaleTrack trains with a cylinder-collision G1 while MagicSim uses `g1_new.usd`, so foot contact on
edges should be aligned or randomised; compute: the current GPU is shared (24 GB) and faults under contention.

## 4. Until then

- Terrain data collection: MagicLoco owns the legs (decoupled WBC, validated on stairs, boxes, rough, slopes, H0T), Pink
  the arms; strict replay reproduces the episodes.
- Data meant to be re-tracked by ScaleBFM: flat ground and slopes only.
