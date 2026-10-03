#!/bin/bash
# usage: squat_probe8.sh <gpu> <tag> <checkpoint> : track the 8 planner squat clips on the MagicSim foot model, print the summary line
GPU=$1; TAG=$2; CK=$3
ROOT=/home/vcj9002/magicloco/ScaleBFM/ScaleTrack
source /home/vcj9002/scalebfm_ws/envs/scaletrack/bin/activate
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=$GPU
export SCALETRACK_ROBOT_USD=/home/vcj9002/scalebfm_ws/assets/g1_29dof_discfeet/g1_29dof_discfeet.usda
OUT=/home/vcj9002/scalebfm_ws/runs/eval/squat8/$TAG; mkdir -p $OUT
cd $ROOT
python scripts/eval/eval_modes.py --headless --checkpoint $CK --motion_file /home/vcj9002/scalebfm_ws/motions/yaml/deepsquat_all8.yaml \
  --num_envs 8 --modes 7 4 --tracking global --seed 0 --out $OUT/squat.json --per_clip $OUT/squat.npz --trace_out $OUT/trace > $OUT/log.txt 2>&1
python scripts/eval/squat_summary.py $OUT/trace
