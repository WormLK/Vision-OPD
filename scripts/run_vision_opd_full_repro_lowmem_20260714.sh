#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

cd /data00/users/wanglikun/ProjWormLK/Vision-OPD

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
unset PYTORCH_CUDA_ALLOC_CONF
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-Qwen3.5-4B-full-repro-lowmem-20260714}"
export MODEL_PATH="${MODEL_PATH:-/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B}"

# Full-data reproduction with lower activation pressure than the 20260713 safe
# wrapper. The 32x2x128 run reached step 3 and then OOMed in actor backward.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-96}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-5120}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-5216}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-5216}"
export DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-1}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-2}"

bash scripts/run_vision_opd.sh
