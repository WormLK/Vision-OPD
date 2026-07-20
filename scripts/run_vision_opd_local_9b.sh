#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

cd /data00/users/wanglikun/ProjWormLK/Vision-OPD

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# vLLM's CuMem memory pool is incompatible with
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
unset PYTORCH_CUDA_ALLOC_CONF
export MODEL_PATH="${MODEL_PATH:-/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-9b}"
export DATA_DIR="${DATA_DIR:-/data00/users/wanglikun/ProjWormLK/Vision-OPD/data}"
export TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${DATA_DIR}/train.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-Qwen3.5-9B-local-lowmem-20260714}"

# 9B local starting point. It keeps the Vision-OPD loss/configuration aligned
# with the paper and starts from conservative memory settings for 8x46GB L40S.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MIMI_BATCH_SIZE="${PPO_MIMI_BATCH_SIZE:-8}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-4}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-48}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-6192}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-6192}"
export ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE="${ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE:-2}"
export FILTER_OVERLONG_PROMPTS="${FILTER_OVERLONG_PROMPTS:-True}"
export FILTER_OVERLONG_PROMPTS_WORKERS="${FILTER_OVERLONG_PROMPTS_WORKERS:-8}"
export DATA_TRUNCATION="${DATA_TRUNCATION:-error}"
export ALPHA="${ALPHA:-0.5}"
export TEACHER_MODEL_SOURCE="${TEACHER_MODEL_SOURCE:-legacy}"
export TEACHER_REGULARIZATION="${TEACHER_REGULARIZATION:-ema}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
export ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-True}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.30}"
export DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-1}"
export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-8}"
export TRAINER_SAVE_FREQ="${TRAINER_SAVE_FREQ:-10}"
export TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS:-1}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-2}"

bash scripts/run_vision_opd.sh "$@"

EXPECTED_FINAL_STEP="${EXPECTED_FINAL_STEP:-779}"
CHECKPOINT_ROOT="${TRAINER_DEFAULT_LOCAL_DIR:-/data00/users/wanglikun/ProjWormLK/Vision-OPD/checkpoints/${EXPERIMENT_NAME}}"
MARKER="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
if [[ -f "${MARKER}" ]]; then
    STEP="$(tr -cd '0-9' < "${MARKER}")"
    DATA_STATE="${CHECKPOINT_ROOT}/global_step_${STEP}/data.pt"
    if [[ -n "${STEP}" ]] && (( STEP < EXPECTED_FINAL_STEP )) && [[ -f "${DATA_STATE}" ]]; then
        BACKUP="${DATA_STATE}.resume_original"
        if [[ -e "${BACKUP}" ]]; then
            BACKUP="${DATA_STATE}.resume_original.$(date -u +'%Y%m%dT%H%M%SZ')"
        fi
        mv "${DATA_STATE}" "${BACKUP}"
        echo "Archived incomplete dataloader state ${DATA_STATE} -> ${BACKUP}."
    fi
fi
