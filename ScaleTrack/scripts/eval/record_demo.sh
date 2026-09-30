#!/bin/bash
# Record videos of a checkpoint tracking the three demo stair clips (held-out layouts 6/7, max step ~0.09 / 0.20 / 0.27 m; the
# reference markers are drawn) in mode 7, global tracking. One process per clip.
# usage: [EVAL_PY=path/to/eval_modes.py] [MODE=7] [TRACKING=global|local] [CLIPS="low mid high"] record_demo.sh <abs checkpoint> <tag> <gpu>
CKPT=$1; TAG=$2; GPU=$3
MODE=${MODE:-7}; TRACKING=${TRACKING:-global}; CLIPS=${CLIPS:-"low mid high"}
PY=${EVAL_PY:-/home/vcj9002/magicloco/ScaleBFM/ScaleTrack/scripts/eval/eval_modes.py}
source /home/vcj9002/scalebfm_ws/envs/scaletrack/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=$GPU
export SCALETRACK_TERRAIN_ROOT=/home/vcj9002/scalebfm_ws/motions/terrain_raw
export SCALETRACK_LAYOUT_SEEDS=6,7
export SCALETRACK_CLIP_META=/home/vcj9002/scalebfm_ws/motions/yaml/clip_meta_test.json
export SCALETRACK_TERRAIN_FRAC=1.0
OUT=/home/vcj9002/scalebfm_ws/runs/eval/videos/$TAG
mkdir -p $OUT
cd /home/vcj9002/magicloco/ScaleBFM/ScaleTrack
for D in $CLIPS; do
  python $PY --task G1-BFM-Transformer-Tracking-Terrain --headless --checkpoint $CKPT \
    --motion_file /home/vcj9002/scalebfm_ws/motions/yaml/demo_$D.yaml --num_envs 1 --modes $MODE --tracking $TRACKING \
    --max_steps 1000 --seed 0 --video_dir $OUT --out $OUT/demo_$D.json > $OUT/demo_$D.log 2>&1
  grep -E "=== mode|wrote" $OUT/demo_$D.log
done
echo DEMO_DONE
