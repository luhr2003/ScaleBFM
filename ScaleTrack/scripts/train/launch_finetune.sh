#!/bin/bash
# Terrain fine-tuning of the pretrained BFM (resumes from model_22200) with flat rehearsal + KL anchor.
# usage: [ENVS=4096] [TERRAIN_FRAC=0.35] [ITERS=2000] [ANCHOR=1.0] [ACTOR_LR=5e-5] [FREEZE=100] [SAVE=100] \
#        [EXTRA="agent.algorithm.entropy_coef=0.001"] launch_finetune.sh <run_name> <gpu list e.g. 0,2,4> [train yaml]
RUN=$1; GPUS=$2; YAML=${3:-/home/vcj9002/scalebfm_ws/motions/yaml/train_all.yaml}
ENVS=${ENVS:-4096}; ITERS=${ITERS:-2000}; ANCHOR=${ANCHOR:-1.0}; ACTOR_LR=${ACTOR_LR:-5e-5}; CRITIC_LR=${CRITIC_LR:-3e-4}
FREEZE=${FREEZE:-100}; SAVE=${SAVE:-100}; BASE_RUN=${BASE_RUN:-humanoid_transformer_m}; BASE_CKPT=${BASE_CKPT:-model_22200.pt}
source /home/vcj9002/scalebfm_ws/envs/scaletrack/bin/activate
ANCHOR=$(python3 -c "print(float('$ANCHOR'))")  # Hydra parses `2` as int; the config field is a float
# 80 logical cores are shared by many jobs: PyTorch's default thread count only burns CPU on the small CPU-side ops
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=$GPUS PYTHONUNBUFFERED=1
# cache of the concatenated motion library: the first launch builds it, every relaunch loads it in about a minute
export SCALETRACK_MOTION_CACHE=${SCALETRACK_MOTION_CACHE:-/home/vcj9002/scalebfm_ws/motions/cache}
export SCALETRACK_TERRAIN_ROOT=${SCALETRACK_TERRAIN_ROOT:-/home/vcj9002/scalebfm_ws/motions/terrain_raw}
export SCALETRACK_LAYOUT_SEEDS=${SCALETRACK_LAYOUT_SEEDS:-0,1,2,3,4,5}
export SCALETRACK_CLIP_META=${SCALETRACK_CLIP_META:-/home/vcj9002/scalebfm_ws/motions/yaml/clip_meta.json}
export SCALETRACK_TERRAIN_FRAC=${TERRAIN_FRAC:-0.35}
NGPU=$(echo $GPUS | tr ',' '\n' | wc -l)
ROOT_DIR=/home/vcj9002/magicloco/ScaleBFM/ScaleTrack
cd $ROOT_DIR
ARGS="--task G1-BFM-Transformer-Tracking-Terrain --motion_file $YAML --num_envs $ENVS --max_iterations $ITERS \
  --resume True --load_run $BASE_RUN --checkpoint $BASE_CKPT --run_name $RUN --headless \
  agent.save_interval=$SAVE agent.algorithm.anchor_coef=$ANCHOR agent.algorithm.actor_learning_rate=$ACTOR_LR \
  agent.algorithm.critic_learning_rate=$CRITIC_LR agent.algorithm.actor_freeze_iters=$FREEZE \
  agent.algorithm.anchor_checkpoint=$ROOT_DIR/logs/rsl_rl/g1_bfm_tracking_exp/humanoid_transformer_m/model_22200.pt $EXTRA"
if [ "$NGPU" -gt 1 ]; then
  # NCCL's default peer-to-peer transport deadlocks between these GPUs (IOMMU is enabled on this machine, a plain
  # 4-rank all_reduce hangs, see the notes); host-staged transport works and is plenty fast for the small gradients.
  export NCCL_P2P_DISABLE=1
  # MotionCommand attaches to an existing shared-memory motion library if one is left over from an earlier run
  # (it only creates one when none exists), which would silently give the wrong library: always start clean.
  rm -f /dev/shm/shared_motionlib_train /dev/shm/shared_motionlib_test
  exec python -m torch.distributed.run --nnodes 1 --nproc_per_node $NGPU scripts/pretrain/rsl_rl/train.py $ARGS --distributed
else
  exec python scripts/pretrain/rsl_rl/train.py $ARGS
fi
