#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

cd /data00/users/wanglikun/ProjWormLK/Vision-OPD

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# vLLM's CuMem memory pool is incompatible with
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
unset PYTORCH_CUDA_ALLOC_CONF
export MODEL_PATH="${MODEL_PATH:-/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B}"
export DATA_DIR="${DATA_DIR:-/data00/users/wanglikun/ProjWormLK/Vision-OPD/data}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${DATA_DIR}/train.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-Qwen3.5-4B-full-repro-lowmem-20260714}"

# Local 8xL40S continuation settings. These preserve the paper objective
# choices (JSD, top-K=100, EMA teacher, one epoch, no reward/KL) while reducing
# rollout and sequence pressure after the full paper setting OOMed locally.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-8}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-48}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-6192}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-6192}"
# A small number of image-heavy samples exceed 6144 tokens (observed: 7880).
# Filter them with the multimodal processor so token IDs and RoPE grids remain
# aligned. Raising the sequence limit to 8192 previously OOMed actor backward.
export FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-True}"
export FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-8}"
export DATA_TRUNCATION="${DATA_TRUNCATION:-error}"
export ALPHA="${ALPHA:-0.5}"
export TEACHER_MODEL_SOURCE="${TEACHER_MODEL_SOURCE:-legacy}"
export TEACHER_REGULARIZATION="${TEACHER_REGULARIZATION:-ema}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
export DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-1}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-2}"

bash scripts/run_vision_opd.sh "$@"
