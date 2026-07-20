#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

cd /data00/users/wanglikun/ProjWormLK/Vision-OPD

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MODEL_PATH="${MODEL_PATH:-/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B}"
export DATA_DIR="${DATA_DIR:-/data00/users/wanglikun/ProjWormLK/Vision-OPD/data}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${DATA_DIR}/train.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-Qwen3.5-4B-paper-repro}"

# Paper experimental settings:
# one epoch on Vision-OPD-6K, batch 96, on-policy rollout n=8,
# max generation length 1024, top-K=100, EMA teacher, JSD alpha/beta=0.5.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-96}"
export PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-96}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-9216}"
export ALPHA="${ALPHA:-0.5}"
export TEACHER_MODEL_SOURCE="${TEACHER_MODEL_SOURCE:-legacy}"
export TEACHER_REGULARIZATION="${TEACHER_REGULARIZATION:-ema}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-20}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-3}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"

bash scripts/run_vision_opd.sh "$@"
