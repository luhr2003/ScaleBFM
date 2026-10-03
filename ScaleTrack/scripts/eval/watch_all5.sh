#!/bin/bash
# Gate queue for the MagicSim foot model (two flat discs per foot = the collision model of MagicSim's g1_new.usd), robust version.
# For every checkpoint model_<it>.pt with it % STEP == 0 (default 200):
#   disc feet (the target): held-out terrain gate (modes 7 and 4, global AND local) + the FULL flat gate (26 configs, BONES + Ours)
#   capsule feet (original asset, sanity check, it % 800 == 0): the same.
# The flat gate always has the process structure of the baseline (parts "0 1 2 3" / "4 5 6 7"): results of a configuration depend on its
# position inside a process (the 2nd mode of a process scores ~1.5 pp lower on a few fragile BONES clips), so only like-for-like
# comparisons are valid.
# Every Kit process is started 15 s after the previous one and every expected result file is verified afterwards; missing parts are re-run
# (up to 3 attempts). Reason: at 09:51 on 2026-10-01 four processes started at the same time and the flat parts died silently
# ("Failed to create simulation view backend", run_gate.sh still printed GATE_DONE), which left an empty gate line.
# Every result is compared with the base model on the same asset (runs/eval/base22200_disc, runs/eval/base22200).
# usage: watch_all4.sh <run_name> <gpu> [STEP=200]            watch forever
#        watch_all4.sh <run_name> <gpu> --repair <it> [<it>..] (re)run the missing parts of the gates of these checkpoints once and print their lines
#        watch_all4.sh <run_name> <gpu> --eval <tag> <checkpoint file>  gate one arbitrary checkpoint file (e.g. an average) as <tag>_disc
RUN=$1; GPU=$2; shift 2
STEP=200; REPAIR=(); EVAL_TAG=""; EVAL_CKPT=""
if [ "$1" = "--repair" ]; then shift; REPAIR=("$@")
elif [ "$1" = "--eval" ]; then EVAL_TAG=$2; EVAL_CKPT=$3
elif [ -n "$1" ]; then STEP=$1; fi
ROOT=/home/vcj9002/magicloco/ScaleBFM/ScaleTrack
DIR=$ROOT/logs/rsl_rl/g1_bfm_tracking_exp/$RUN
EVAL=/home/vcj9002/scalebfm_ws/runs/eval
DONE=$EVAL/.watchall4_$RUN; mkdir -p $DONE
DISC=/home/vcj9002/scalebfm_ws/assets/g1_29dof_discfeet/g1_29dof_discfeet.usda
source /home/vcj9002/scalebfm_ws/envs/scaletrack/bin/activate

summ() {  # <ref dir> <new dir> -> one line
  python $ROOT/scripts/eval/summarize_gate.py compare $1 $2 > $2/compare.txt 2>&1
  python $ROOT/scripts/eval/summarize_gate.py terrain $1 $2 > $2/terrain.txt 2>&1
  local TER AGG
  TER=$(grep "overall Succ" $2/terrain.txt | tail -4 | sed 's/^ *//' | tr -s ' ' | sed 's/G-MPKPE.*//' | tr '\n' ';')
  AGG=$(grep -E "^  (bones|ours) " $2/compare.txt | sed 's/^ *//' | tr -s ' ' | sed 's/n_cfg=[0-9]* //; s/G-MPKPE.*//' | tr '\n' ';')
  echo "$(tail -1 $2/compare.txt | cut -c1-200) | flat agg: $AGG | terrain: $TER"
}

wait_mem() {  # wait until the evaluation GPU has at least 7 GB free (max 60 min): too many Kit processes on one GPU die at start-up
  local i free
  for i in $(seq 120); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | head -1)
    [ -n "$free" ] && [ "$free" -ge 7000 ] && return 0
    sleep 30
  done
  return 0
}

missing() {  # <tag> -> prints the expected result files that do not exist
  local t=$1 f
  for f in terrain_quick_s0_a terrain_quick_s0_b bones_quick_s0_a bones_quick_s0_b ours_quick_s0_a ours_quick_s0_b; do
    [ -e $EVAL/$t/$f.json ] || echo $f
  done
}

launch_missing() {  # <tag> <ckpt> <1 = disc feet, 0 = original asset>
  local tag=$1 ck=$2 disc=$3 m
  m=" $(missing $tag | tr '\n' ' ') "
  local ev=(); [ "$disc" = 1 ] && ev=(SCALETRACK_ROBOT_USD=$DISC)
  mkdir -p $EVAL/$tag
  if [[ $m == *" terrain_quick_s0_a "* ]]; then
    wait_mem
    env "${ev[@]}" MODES="7 4" TRACKING=global SUFFIX=a $ROOT/scripts/eval/run_terrain_gate.sh $ck $tag $GPU quick 0 > $EVAL/$tag.terrain_g.log 2>&1 &
    sleep 15
  fi
  if [[ $m == *" terrain_quick_s0_b "* ]]; then
    wait_mem
    env "${ev[@]}" MODES="7 4" TRACKING=local SUFFIX=b $ROOT/scripts/eval/run_terrain_gate.sh $ck $tag $GPU quick 0 > $EVAL/$tag.terrain_l.log 2>&1 &
    sleep 15
  fi
  if [[ $m == *" bones_quick_s0_a "* || $m == *" ours_quick_s0_a "* ]]; then
    wait_mem
    env "${ev[@]}" MODES="0 1 2 3" SUFFIX=a $ROOT/scripts/eval/run_gate.sh $ck $tag $GPU quick 0 > $EVAL/$tag.gate_a.log 2>&1 &
    sleep 15
  fi
  if [[ $m == *" bones_quick_s0_b "* || $m == *" ours_quick_s0_b "* ]]; then
    wait_mem
    env "${ev[@]}" MODES="4 5 6 7" SUFFIX=b $ROOT/scripts/eval/run_gate.sh $ck $tag $GPU quick 0 > $EVAL/$tag.gate_b.log 2>&1 &
    sleep 15
  fi
}

ensure() {  # <tag> <ckpt> <disc> : run until all expected results exist (3 attempts)
  local tag=$1 ck=$2 disc=$3 a
  for a in 1 2 3 4; do
    [ -z "$(missing $tag)" ] && return 0
    [ $a -gt 1 ] && echo "[watch-all4] $(date +%H:%M:%S) $tag: missing $(missing $tag | tr '\n' ' ')-> attempt $a"
    launch_missing $tag $ck $disc
    wait
    [ -n "$(missing $tag)" ] && sleep 60
  done
  [ -z "$(missing $tag)" ]
}

squat_probe() {  # <tag> <ckpt> <disc> -> one summary line (8 planner squat clips: 4 train depths + 4 held-out validation depths)
  local tag=$1 ck=$2 disc=$3 ev=()
  [ "$disc" = 1 ] && ev=(SCALETRACK_ROBOT_USD=$DISC)
  mkdir -p $EVAL/$tag
  wait_mem
  ( cd $ROOT && env "${ev[@]}" CUDA_VISIBLE_DEVICES=$GPU python scripts/eval/eval_modes.py --headless --checkpoint $ck \
      --motion_file /home/vcj9002/scalebfm_ws/motions/yaml/deepsquat_all8.yaml --num_envs 8 --modes 7 4 --tracking global --seed 0 \
      --out $EVAL/$tag/squat_s0.json --per_clip $EVAL/$tag/squat_s0.npz --trace_out $EVAL/$tag/squat_trace > $EVAL/$tag.squat.log 2>&1 )
  python $ROOT/scripts/eval/squat_summary.py $EVAL/$tag/squat_trace
}

do_checkpoint() {  # <it>
  local it=$1 ck=$DIR/model_$1.pt T0=$(date +%s)
  local TAG=${RUN}_it${it} TAGD=${RUN}_it${it}_disc
  echo "[watch-all4] $(date +%H:%M:%S) gating $TAGD"
  ensure $TAGD $ck 1 || echo "[watch-all4] WARNING $TAGD still incomplete: $(missing $TAGD | tr '\n' ' ')"
  echo "[watch-all4] $(date +%H:%M:%S) $TAGD ($(( $(date +%s) - T0 ))s): disc: $(summ $EVAL/base22200_disc $EVAL/$TAGD) | $(squat_probe $TAGD $ck 1)"
  if (( it % 800 == 0 )); then
    ensure $TAG $ck 0 || echo "[watch-all4] WARNING $TAG still incomplete: $(missing $TAG | tr '\n' ' ')"
    echo "[watch-all4] $(date +%H:%M:%S) $TAG ($(( $(date +%s) - T0 ))s): capsule: $(summ $EVAL/base22200 $EVAL/$TAG)"
  fi
}

if [ -n "$EVAL_TAG" ]; then   # evaluate one arbitrary checkpoint file (e.g. an average of checkpoints) on the disc-feet asset
  T0=$(date +%s)
  echo "[watch-all4] $(date +%H:%M:%S) gating ${EVAL_TAG}_disc ($EVAL_CKPT)"
  ensure ${EVAL_TAG}_disc $EVAL_CKPT 1 || echo "[watch-all4] WARNING ${EVAL_TAG}_disc still incomplete: $(missing ${EVAL_TAG}_disc | tr '\n' ' ')"
  echo "[watch-all4] $(date +%H:%M:%S) ${EVAL_TAG}_disc ($(( $(date +%s) - T0 ))s): disc: $(summ $EVAL/base22200_disc $EVAL/${EVAL_TAG}_disc)"
  exit 0
fi

if [ ${#REPAIR[@]} -gt 0 ]; then
  for it in "${REPAIR[@]}"; do do_checkpoint $it; done
  exit 0
fi

echo "[watch-all4] run $RUN gpu $GPU step $STEP (disc feet first, full flat gate, verified)"
while true; do
  for ck in $(ls $DIR/model_*.pt 2>/dev/null | sort -V); do
    it=$(basename $ck .pt | cut -d_ -f2)
    if (( it % STEP == 0 )) && [ ! -e $DONE/$it ]; then
      sleep 20   # let torch.save finish
      do_checkpoint $it
      touch $DONE/$it
    fi
  done
  sleep 60
done
