#!/bin/bash
# Full quick gate (held-out terrain global + local, flat 26 configs in the baseline's two-part structure) of ONE checkpoint file for ONE evaluation
# seed, with start-up staggering and result-file verification (retries). Used for multi-seed comparisons of final candidates against the base model.
# usage: eval_seed.sh <abs checkpoint> <tag> <seed> <disc 1 | capsule 0> [gpu=4]
#   results: /home/vcj9002/scalebfm_ws/runs/eval/<tag>/{terrain,bones,ours}_quick_s<seed>_{a,b}.json
CKPT=$1; TAG=$2; SEED=$3; DISCFEET=$4; GPU=${5:-4}
ROOT=/home/vcj9002/magicloco/ScaleBFM/ScaleTrack
EVAL=/home/vcj9002/scalebfm_ws/runs/eval
DISC=/home/vcj9002/scalebfm_ws/assets/g1_29dof_discfeet/g1_29dof_discfeet.usda
source /home/vcj9002/scalebfm_ws/envs/scaletrack/bin/activate
EV=(); [ "$DISCFEET" = 1 ] && EV=(SCALETRACK_ROBOT_USD=$DISC)
mkdir -p $EVAL/$TAG

wait_mem() {  # wait until the evaluation GPU has at least 7 GB free (max 60 min): too many Kit processes on one GPU die at start-up
  local i free
  for i in $(seq 120); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | head -1)
    [ -n "$free" ] && [ "$free" -ge 7000 ] && return 0
    sleep 30
  done
  return 0
}

missing() { local f; for f in terrain_quick_s${SEED}_a terrain_quick_s${SEED}_b bones_quick_s${SEED}_a bones_quick_s${SEED}_b ours_quick_s${SEED}_a ours_quick_s${SEED}_b; do [ -e $EVAL/$TAG/$f.json ] || echo $f; done; }

launch_missing() {
  local m=" $(missing | tr '\n' ' ') "
  if [[ $m == *" terrain_quick_s${SEED}_a "* ]]; then
    wait_mem
    env "${EV[@]}" MODES="7 4" TRACKING=global SUFFIX=a $ROOT/scripts/eval/run_terrain_gate.sh $CKPT $TAG $GPU quick $SEED > $EVAL/$TAG.s$SEED.terrain_g.log 2>&1 &
    sleep 15
  fi
  if [[ $m == *" terrain_quick_s${SEED}_b "* ]]; then
    wait_mem
    env "${EV[@]}" MODES="7 4" TRACKING=local SUFFIX=b $ROOT/scripts/eval/run_terrain_gate.sh $CKPT $TAG $GPU quick $SEED > $EVAL/$TAG.s$SEED.terrain_l.log 2>&1 &
    sleep 15
  fi
  if [[ $m == *" bones_quick_s${SEED}_a "* || $m == *" ours_quick_s${SEED}_a "* ]]; then
    wait_mem
    env "${EV[@]}" MODES="0 1 2 3" SUFFIX=a $ROOT/scripts/eval/run_gate.sh $CKPT $TAG $GPU quick $SEED > $EVAL/$TAG.s$SEED.gate_a.log 2>&1 &
    sleep 15
  fi
  if [[ $m == *" bones_quick_s${SEED}_b "* || $m == *" ours_quick_s${SEED}_b "* ]]; then
    wait_mem
    env "${EV[@]}" MODES="4 5 6 7" SUFFIX=b $ROOT/scripts/eval/run_gate.sh $CKPT $TAG $GPU quick $SEED > $EVAL/$TAG.s$SEED.gate_b.log 2>&1 &
    sleep 15
  fi
}

for attempt in 1 2 3 4; do
  [ -z "$(missing)" ] && break
  [ $attempt -gt 1 ] && echo "[eval_seed] $(date +%H:%M:%S) $TAG seed $SEED: missing $(missing | tr '\n' ' ')-> attempt $attempt"
  launch_missing
  wait
  [ -n "$(missing)" ] && sleep 60
done
if [ -z "$(missing)" ]; then echo "[eval_seed] $(date +%H:%M:%S) $TAG seed $SEED complete"; else echo "[eval_seed] WARNING $TAG seed $SEED incomplete: $(missing | tr '\n' ' ')"; fi
