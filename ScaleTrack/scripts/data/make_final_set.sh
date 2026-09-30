#!/bin/bash
# After the terrain collection is complete: package all terrain layouts, then assemble the training / eval YAMLs.
# usage: make_final_set.sh <gpu> [bones clips=60000] [terrain clips=30000]
GPU=${1:-2}; NB=${2:-60000}; NT=${3:-30000}
W=/home/vcj9002/scalebfm_ws/motions
ROOT=/home/vcj9002/magicloco/ScaleBFM/ScaleTrack
$ROOT/scripts/data/package_terrain.sh $GPU 0 1 2 3 4 5 6 7
source /home/vcj9002/scalebfm_ws/envs/scaletrack/bin/activate
FLAT="$W/processed/bones:$NB $W/processed/lafan"
[ -d $W/processed/snapmogen ] && FLAT="$FLAT $W/processed/snapmogen"
[ -d $W/processed/amass ] && FLAT="$FLAT $W/processed/amass"
python $ROOT/scripts/data/build_training_set.py --flat_dirs $FLAT --terrain_root $W/terrain_raw --terrain_dir $W/processed/terrain \
  --out_dir $W/yaml --terrain_max_clips $NT --train_layouts 0 1 2 3 4 5
echo FINAL_SET_DONE
