#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

cd /data00/users/wanglikun/ProjWormLK/Vision-OPD

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-Qwen3.5-4B-full-repro-lowmem-20260714}"
export MODEL_PATH="${MODEL_PATH:-/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B}"

# Resume after the batch=8 / 8192-token run reached global_step_100 and then
# OOMed during actor backward at step 109. Keep the same experiment directory so
# the trainer auto-loads the latest checkpoint, but lower per-step activation
# pressure and vLLM cache size.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-48}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-6192}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-6192}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
export DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-1}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-2}"

bash scripts/run_vision_opd.sh
