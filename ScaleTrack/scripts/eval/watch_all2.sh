#!/bin/bash
# Gate queue for the MagicSim foot model (two flat discs per foot = the collision model of MagicSim's g1_new.usd).
# For every checkpoint model_<it>.pt with it % STEP == 0 (default 200):
#   disc feet (the target): held-out terrain gate, modes 7 and 4, global AND local tracking;
#                           flat gate: full (26 configs, BONES + Ours) when it % 400 == 0, otherwise a BONES sentinel (modes 0 4 7);
#   capsule feet (original asset, sanity only, it % 800 == 0): terrain global + BONES sentinel.
# Every result is compared with the base model on the same asset (runs/eval/base22200_disc, runs/eval/base22200).
# usage: watch_all2.sh <run_name> <gpu> [STEP=200]
RUN=$1; GPU=$2; STEP=${3:-200}
ROOT=/home/vcj9002/magicloco/ScaleBFM/ScaleTrack
DIR=$ROOT/logs/rsl_rl/g1_bfm_tracking_exp/$RUN
EVAL=/home/vcj9002/scalebfm_ws/runs/eval
DONE=$EVAL/.watchall2_$RUN; mkdir -p $DONE
DISC=/home/vcj9002/scalebfm_ws/assets/g1_29dof_discfeet/g1_29dof_discfeet.usda
source /home/vcj9002/scalebfm_ws/envs/scaletrack/bin/activate
echo "[watch-all2] run $RUN gpu $GPU step $STEP (disc feet first)"
summ() {  # <ref dir> <new dir> -> one line
  python $ROOT/scripts/eval/summarize_gate.py compare $1 $2 > $2/compare.txt 2>&1
  python $ROOT/scripts/eval/summarize_gate.py terrain $1 $2 > $2/terrain.txt 2>&1
  local TER AGG
  TER=$(grep "overall Succ" $2/terrain.txt | tail -4 | sed 's/^ *//' | tr -s ' ' | sed 's/G-MPKPE.*//' | tr '\n' ';')
  AGG=$(grep -E "^  (bones|ours) " $2/compare.txt | sed 's/^ *//' | tr -s ' ' | sed 's/n_cfg=[0-9]* //; s/G-MPKPE.*//' | tr '\n' ';')
  echo "$(tail -1 $2/compare.txt | cut -c1-200) | flat agg: $AGG | terrain: $TER"
}
while true; do
  for ck in $(ls $DIR/model_*.pt 2>/dev/null | sort -V); do
    it=$(basename $ck .pt | cut -d_ -f2)
    if (( it % STEP == 0 )) && [ ! -e $DONE/$it ]; then
      TAG=${RUN}_it${it}; TAGD=${TAG}_disc
      sleep 15
      T0=$(date +%s)
      echo "[watch-all2] $(date +%H:%M:%S) gating $TAGD"
      SCALETRACK_ROBOT_USD=$DISC MODES="7 4" TRACKING="global" SUFFIX=a $ROOT/scripts/eval/run_terrain_gate.sh $ck $TAGD $GPU quick 0 > $EVAL/$TAGD.terrain_g.log 2>&1 &
      SCALETRACK_ROBOT_USD=$DISC MODES="7 4" TRACKING="local" SUFFIX=b $ROOT/scripts/eval/run_terrain_gate.sh $ck $TAGD $GPU quick 0 > $EVAL/$TAGD.terrain_l.log 2>&1 &
      if (( it % 400 == 0 )); then
        SCALETRACK_ROBOT_USD=$DISC MODES="0 1 2 3" SUFFIX=a $ROOT/scripts/eval/run_gate.sh $ck $TAGD $GPU quick 0 > $EVAL/$TAGD.gate_a.log 2>&1 &
        SCALETRACK_ROBOT_USD=$DISC MODES="4 5 6 7" SUFFIX=b $ROOT/scripts/eval/run_gate.sh $ck $TAGD $GPU quick 0 > $EVAL/$TAGD.gate_b.log 2>&1 &
      else
        SCALETRACK_ROBOT_USD=$DISC SETS="bones" MODES="0 4 7" SUFFIX=s $ROOT/scripts/eval/run_gate.sh $ck $TAGD $GPU quick 0 > $EVAL/$TAGD.sentinel.log 2>&1 &
      fi
      if (( it % 800 == 0 )); then
        MODES="7 4" TRACKING="global" SUFFIX=a $ROOT/scripts/eval/run_terrain_gate.sh $ck $TAG $GPU quick 0 > $EVAL/$TAG.terrain_g.log 2>&1 &
        SETS="bones" MODES="0 4 7" SUFFIX=s $ROOT/scripts/eval/run_gate.sh $ck $TAG $GPU quick 0 > $EVAL/$TAG.sentinel.log 2>&1 &
      fi
      wait
      L2=$(summ $EVAL/base22200_disc $EVAL/$TAGD)
      echo "[watch-all2] $(date +%H:%M:%S) $TAGD ($(( $(date +%s) - T0 ))s): disc: $L2"
      if (( it % 800 == 0 )); then
        L1=$(summ $EVAL/base22200 $EVAL/$TAG)
        echo "[watch-all2] $(date +%H:%M:%S) $TAG ($(( $(date +%s) - T0 ))s): capsule: $L1"
      fi
      touch $DONE/$it
    fi
  done
  sleep 60
done
