#!/bin/bash
# Regression gate for a BFM checkpoint over (mode, tracking) configs on the two official test sets.
# usage: [MODES="7 6 4"] [SUFFIX=a] run_gate.sh <abs checkpoint> <tag> <gpu index> [quick|full] [seed] [extra eval_modes.py args...]
#   quick: 1000 BONES + 300 Ours clips (fixed subsets), clips capped at 1000 frames
#   full : all 10000 BONES + 1649 Ours clips, full length
# MODES/SUFFIX let several GPUs share one gate (each writes <set>_<level>_s<seed>_<suffix>.json).
# SETS="bones" restricts the gate to the BONES test set (a light sentinel); default "bones ours".
# TRACKING="global local" (default) restricts the tracking types.
CKPT=$1; TAG=$2; GPU=$3; LEVEL=${4:-quick}; SEED=${5:-0}; shift 5 2>/dev/null || shift $#
MODES=${MODES:-"0 1 2 3 4 5 6 7"}; SUFFIX=${SUFFIX:-all}; SETS=${SETS:-"bones ours"}; TRACKING=${TRACKING:-"global local"}
source /home/vcj9002/scalebfm_ws/envs/scaletrack/bin/activate
# 80 logical cores are shared by many jobs: PyTorch's default thread count only burns CPU on the small CPU-side ops
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=$GPU
OUT=/home/vcj9002/scalebfm_ws/runs/eval/$TAG
YAML=/home/vcj9002/scalebfm_ws/motions/yaml
mkdir -p $OUT
cd /home/vcj9002/magicloco/ScaleBFM/ScaleTrack
if [ "$LEVEL" = quick ]; then
  B="--max_clips 1000 --max_steps 1000 --num_envs 1024"; O="--max_clips 300 --max_steps 1000 --num_envs 320"
else
  B="--num_envs 2048"; O="--num_envs 1024"
fi
if [[ " $SETS " == *" bones "* ]]; then
python scripts/eval/eval_modes.py --headless --checkpoint $CKPT --motion_file $YAML/test_bones.yaml \
  $B --modes $MODES --tracking $TRACKING --seed $SEED --out $OUT/bones_${LEVEL}_s${SEED}_${SUFFIX}.json --per_clip $OUT/bones_${LEVEL}_s${SEED}_${SUFFIX}.npz "$@" || exit 1
fi
if [[ " $SETS " == *" ours "* ]]; then
python scripts/eval/eval_modes.py --headless --checkpoint $CKPT --motion_file $YAML/test_ours.yaml \
  $O --modes $MODES --tracking $TRACKING --seed $SEED --out $OUT/ours_${LEVEL}_s${SEED}_${SUFFIX}.json --per_clip $OUT/ours_${LEVEL}_s${SEED}_${SUFFIX}.npz "$@" || exit 1
fi
echo GATE_DONE
