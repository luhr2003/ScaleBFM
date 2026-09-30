#!/bin/bash
# Package recorded terrain clips (pkl, layout frame) into processed npz for ScaleTrack, one layout at a time.
# usage: package_terrain.sh <gpu> <layout seed> [<layout seed> ...]
# Positions stay in the layout frame (--subtract_origin); already packaged clips are skipped.
GPU=$1; shift
source /home/vcj9002/scalebfm_ws/envs/scaletrack/bin/activate
# 80 logical cores are shared by many jobs: PyTorch's default thread count only burns CPU on the small CPU-side ops
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=$GPU
cd /home/vcj9002/magicloco/ScaleBFM/ScaleTrack
mkdir -p /home/vcj9002/scalebfm_ws/motions/processed/terrain
for L in "$@"; do
  python scripts/data/package_motions_fast.py --headless --subtract_origin --skip_existing \
    --data_dir /home/vcj9002/scalebfm_ws/motions/terrain_raw/clips/layout_$L \
    --output_dir /home/vcj9002/scalebfm_ws/motions/processed/terrain --num_envs 2048 --max_env_frames 3e6
done
echo PACKAGE_TERRAIN_DONE
