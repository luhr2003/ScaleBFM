#!/bin/bash
# Open-loop terrain tracking gate on the HELD-OUT terrain layouts (clips of pi_L v4 the model never trained on).
# usage: [MODES="7 4"] [TRACKING="global"] [SUFFIX=a] run_terrain_gate.sh <abs checkpoint> <tag> <gpu> [quick|full] [seed] [extra args]
#   quick: 1500 clips (fixed subset) of the test layouts, 1000 frames; full: all test clips
CKPT=$1; TAG=$2; GPU=$3; LEVEL=${4:-quick}; SEED=${5:-0}; shift 5 2>/dev/null || shift $#
MODES=${MODES:-"7 4"}; TRACKING=${TRACKING:-"global"}; SUFFIX=${SUFFIX:-all}
source /home/vcj9002/scalebfm_ws/envs/scaletrack/bin/activate
# 80 logical cores are shared by many jobs: PyTorch's default thread count only burns CPU on the small CPU-side ops
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=$GPU
export SCALETRACK_TERRAIN_ROOT=${SCALETRACK_TERRAIN_ROOT:-/home/vcj9002/scalebfm_ws/motions/terrain_raw}
export SCALETRACK_LAYOUT_SEEDS=${SCALETRACK_LAYOUT_SEEDS:-6,7}
export SCALETRACK_CLIP_META=${SCALETRACK_CLIP_META:-/home/vcj9002/scalebfm_ws/motions/yaml/clip_meta_test.json}
export SCALETRACK_TERRAIN_FRAC=1.0
YAML=${TERRAIN_EVAL_YAML:-/home/vcj9002/scalebfm_ws/motions/yaml/eval_terrain_test.yaml}
OUT=/home/vcj9002/scalebfm_ws/runs/eval/$TAG
mkdir -p $OUT
cd /home/vcj9002/magicloco/ScaleBFM/ScaleTrack
if [ "$LEVEL" = quick ]; then B="--max_clips 1500 --max_steps 1000 --num_envs 750"; else B="--num_envs 1024"; fi
python scripts/eval/eval_modes.py --task G1-BFM-Transformer-Tracking-Terrain --headless --checkpoint $CKPT --motion_file $YAML \
  $B --modes $MODES --tracking $TRACKING --seed $SEED --out $OUT/terrain_${LEVEL}_s${SEED}_${SUFFIX}.json \
  --per_clip $OUT/terrain_${LEVEL}_s${SEED}_${SUFFIX}.npz "$@"
echo TERRAIN_GATE_DONE
