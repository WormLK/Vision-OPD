#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

cd /data00/users/wanglikun/ProjWormLK/Vision-OPD

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# vLLM's CuMem memory pool rejects expandable CUDA segments.
unset PYTORCH_CUDA_ALLOC_CONF
export MODEL_PATH="${MODEL_PATH:-/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B}"
export DATA_DIR="${DATA_DIR:-/data00/users/wanglikun/ProjWormLK/Vision-OPD/data}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${DATA_DIR}/train.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4}"

# Preserve the released global update: 96 prompts x 8 on-policy rollouts.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-96}"
export PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-96}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
export FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-False}"
export DATA_TRUNCATION="${DATA_TRUNCATION:-error}"

# A 2304-token per-GPU budget with SP4 yields one full 9216-token sequence
# per dynamic micro-batch. The actor accumulates all micro-batches before the
# single optimizer and EMA update for the released 768-trajectory batch.
export ACTOR_USE_DYNAMIC_BSZ="${ACTOR_USE_DYNAMIC_BSZ:-True}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-2304}"
export ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE="${ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE:-4}"
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-4}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}"
export ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-True}"
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"

export ALPHA="${ALPHA:-0.5}"
export TEACHER_MODEL_SOURCE="${TEACHER_MODEL_SOURCE:-legacy}"
export TEACHER_REGULARIZATION="${TEACHER_REGULARIZATION:-ema}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
export DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-0}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
# Bound host-memory use during multimodal preprocessing. The manager concatenates
# all 768 trajectories before the one released-semantics PPO/EMA update.
export ROLLOUT_AGENT_DISPATCH_BATCH_SIZE="${ROLLOUT_AGENT_DISPATCH_BATCH_SIZE:-96}"
export MULTIMODAL_STORAGE_DTYPE="${MULTIMODAL_STORAGE_DTYPE:-bfloat16}"
export DEFER_MULTIMODAL_PROCESSING="${DEFER_MULTIMODAL_PROCESSING:-True}"
export MULTIMODAL_CACHE_DIR="${MULTIMODAL_CACHE_DIR:-/data00/users/wanglikun/ProjWormLK/Vision-OPD/rollouts/${EXPERIMENT_NAME}/image_cache}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-1}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-2}"

bash scripts/run_vision_opd.sh \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  +actor_rollout_ref.rollout.agent.dispatch_batch_size="${ROLLOUT_AGENT_DISPATCH_BATCH_SIZE}" \
  +actor_rollout_ref.rollout.agent.multimodal_storage_dtype="${MULTIMODAL_STORAGE_DTYPE}" \
  +actor_rollout_ref.rollout.agent.defer_multimodal_processing="${DEFER_MULTIMODAL_PROCESSING}" \
  +actor_rollout_ref.rollout.agent.multimodal_cache_dir="${MULTIMODAL_CACHE_DIR}" \
  "$@"
