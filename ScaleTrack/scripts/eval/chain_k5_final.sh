#!/bin/bash
# Deployment-style future offsets [0..5] (K=5): complete quick gate (held-out terrain global+local, flat 26 configs) of the pretrained model and soupV4a,
# seed 0, MagicSim disc feet. Low priority (nice 19) because the machine is oversubscribed.
ROOT=/home/vcj9002/magicloco/ScaleBFM/ScaleTrack
BASE=$ROOT/logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt
V4A=/home/vcj9002/scalebfm_ws/releases/soupV4a_ft_v4_it28300-28600.pt
export EVAL_ARGS="--future_idx 0 1 2 3 4 5"
echo "[k5] $(date +%H:%M:%S) start"
nice -n 19 $ROOT/scripts/eval/eval_seed.sh $V4A soupV4a_disc_k5 0 1 3 > /home/vcj9002/scalebfm_ws/tmp/k5_v4a.log 2>&1 &
sleep 60
nice -n 19 $ROOT/scripts/eval/eval_seed.sh $BASE base22200_disc_k5 0 1 2 > /home/vcj9002/scalebfm_ws/tmp/k5_base.log 2>&1 &
wait
echo "[k5] $(date +%H:%M:%S) evaluations finished"
source /home/vcj9002/scalebfm_ws/envs/scaletrack/bin/activate
E=/home/vcj9002/scalebfm_ws/runs/eval
python $ROOT/scripts/eval/summarize_gate.py compare $E/base22200_disc_k5 $E/soupV4a_disc_k5 > $E/soupV4a_disc_k5/compare.txt 2>&1
python $ROOT/scripts/eval/summarize_gate.py terrain $E/base22200_disc_k5 $E/soupV4a_disc_k5 > $E/soupV4a_disc_k5/terrain.txt 2>&1
echo "[k5] K5_DONE $(date +%H:%M:%S)"
grep -E "AGGREGATE|^  (bones|ours|terrain)|SUMMARY" $E/soupV4a_disc_k5/compare.txt
grep "overall Succ" $E/soupV4a_disc_k5/terrain.txt | tail -4
