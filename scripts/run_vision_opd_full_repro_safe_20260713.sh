#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

cd /data00/users/wanglikun/ProjWormLK/Vision-OPD

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
unset PYTORCH_CUDA_ALLOC_CONF
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-Qwen3.5-4B-full-repro-safe-20260713}"
export MODEL_PATH="${MODEL_PATH:-/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B}"

# The upstream defaults (batch 96, rollout 8, 1024 response tokens) OOM on this
# 8x46GB L40S node. Keep the full 6241-sample dataset, but use the same
# conservative scale that has been validated locally for Vision-OPD pilots.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-32}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-1}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-20}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-2}"

bash scripts/run_vision_opd.sh
