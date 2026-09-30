#!/bin/bash
# Resume a fine-tuning run from its latest (or a given) checkpoint: same run directory, actor unfrozen, anchor unchanged
# (still model_22200), terrain curriculum restored to where it had got to (the env step counter restarts at 0 on resume).
# usage: [ITERS=3000] [ENVS=2048] [SAVE=50] [TERRAIN_FRAC=0.4] [ANCHOR=2] [ACTOR_LR=1e-4] [CRITIC_LR=3e-4] [EXTRA="hydra overrides"] \
#        resume_ft.sh <run_name> <gpu list> [model_<it>.pt]
# Note: the resumed loop repeats the checkpoint's own iteration once, so model_<it>.pt is overwritten if <it> % SAVE == 0.
RUN=$1; GPUS=$2; CK=${3:-}
ROOT=/home/vcj9002/magicloco/ScaleBFM/ScaleTrack
D=$ROOT/logs/rsl_rl/g1_bfm_tracking_exp/$RUN
[ -z "$CK" ] && CK=$(ls $D/model_*.pt | sort -V | tail -1 | xargs basename)
X=$(echo $CK | sed 's/model_\([0-9]*\).pt/\1/')
DONE_ITERS=$((X - 22199)); [ $DONE_ITERS -lt 0 ] && DONE_ITERS=0
CAP=$(python3 -c "print(round(min(1.0, 0.14 + min(1.0, $DONE_ITERS*64/51200)*0.86), 4))")
echo "[resume] run $RUN from $CK (iteration $X), terrain curriculum cap $CAP, ITERS=${ITERS:-3000}"
BASE_RUN=$RUN BASE_CKPT=$CK FREEZE=0 ENVS=${ENVS:-2048} ITERS=${ITERS:-3000} SAVE=${SAVE:-50} TERRAIN_FRAC=${TERRAIN_FRAC:-0.4} \
ANCHOR=${ANCHOR:-2} ACTOR_LR=${ACTOR_LR:-1e-4} CRITIC_LR=${CRITIC_LR:-3e-4} \
EXTRA="env.commands.motion.terrain_curriculum_steps=51200 env.commands.motion.terrain_step_cap_init=$CAP $EXTRA" \
exec $ROOT/scripts/train/launch_finetune.sh $RUN $GPUS
