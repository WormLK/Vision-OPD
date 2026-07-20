#!/usr/bin/env bash
set -uo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
TRAINING_MARKER="${PROJECT_ROOT}/outputs/vision_opd_paper_explicit_training_complete"
COMPLETE_MARKER="${PROJECT_ROOT}/outputs/vision_opd_paper_explicit_reproduction_complete"
PREPARED_DIR="${PROJECT_ROOT}/benchmark/prepared"
LOG_DIR="${PROJECT_ROOT}/logs"
RESULTS_DIR="${PROJECT_ROOT}/benchmark/results"
PORT="${PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
MAX_EVAL_RETRIES="${MAX_EVAL_RETRIES:-100}"
BASE_JUDGE_PATH="/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B"
BASE_JUDGE_NAME="Qwen3.5-4B-base-judge"
FOUR_B_EVAL_MARKER="${PROJECT_ROOT}/outputs/vision_opd_paper_explicit_4b_evaluation_complete"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/benchmark/model_answer" \
  "${PROJECT_ROOT}/benchmark/judge" "${RESULTS_DIR}" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

wait_for_training() {
  while [[ ! -f "${TRAINING_MARKER}" ]]; do
    local four_step=0
    local nine_step=0
    [[ -f checkpoints/Vision-OPD-Qwen3.5-4B-paper-explicit-local/latest_checkpointed_iteration.txt ]] && \
      four_step="$(tr -cd '0-9' < checkpoints/Vision-OPD-Qwen3.5-4B-paper-explicit-local/latest_checkpointed_iteration.txt)"
    [[ -f checkpoints/Vision-OPD-Qwen3.5-9B-paper-explicit-local/latest_checkpointed_iteration.txt ]] && \
      nine_step="$(tr -cd '0-9' < checkpoints/Vision-OPD-Qwen3.5-9B-paper-explicit-local/latest_checkpointed_iteration.txt)"
    log "Waiting for paper-explicit training and merge: 4B=${four_step:-0}/780, 9B=${nine_step:-0}/780."
    sleep 300
  done
}

wait_for_server() {
  local port="$1"
  local attempt
  for attempt in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

evaluate_model() {
  local model_path="$1"
  local served_name="$2"
  local cuda_devices="$3"
  local tp_size="$4"
  local dp_size="$5"
  local judge_cuda_device="$6"
  local server_log="$7"
  local eval_log="$8"
  local attempt=0

  while (( attempt < MAX_EVAL_RETRIES )); do
    attempt=$((attempt + 1))
    log "Starting ${served_name} benchmark attempt ${attempt}/${MAX_EVAL_RETRIES}."
    ray stop --force >> "${eval_log}" 2>&1 || true
    CUDA_VISIBLE_DEVICES="${cuda_devices}" MODEL_PATH="${model_path}" \
      SERVED_MODEL_NAME="${served_name}" PORT="${PORT}" \
      TENSOR_PARALLEL_SIZE="${tp_size}" DATA_PARALLEL_SIZE="${dp_size}" \
      MAX_MODEL_LEN=32768 \
      GPU_MEMORY_UTILIZATION=0.85 bash scripts/serve_qwen_vllm.sh \
      >> "${server_log}" 2>&1 &
    local server_pid=$!
    CUDA_VISIBLE_DEVICES="${judge_cuda_device}" MODEL_PATH="${BASE_JUDGE_PATH}" \
      SERVED_MODEL_NAME="${BASE_JUDGE_NAME}" PORT="${JUDGE_PORT}" \
      TENSOR_PARALLEL_SIZE=1 DATA_PARALLEL_SIZE=1 MAX_MODEL_LEN=8192 \
      GPU_MEMORY_UTILIZATION=0.85 bash scripts/serve_qwen_vllm.sh \
      >> "${LOG_DIR}/vllm_base_judge_eval.log" 2>&1 &
    local judge_server_pid=$!
    local eval_status=1

    if wait_for_server "${PORT}" && wait_for_server "${JUDGE_PORT}"; then
      API_BASE="http://127.0.0.1:${PORT}/v1/" \
      OPENAI_MODEL_ID="${served_name}" MODEL_NAME="${served_name}" \
      ENABLE_THINKING=False \
      BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
      BENCHMARK_DATA_DIR="${PREPARED_DIR}" PREPARE_DATA=false \
      OUT_DIR="${PROJECT_ROOT}/benchmark/model_answer" \
      JUDGE_OUT_DIR="${PROJECT_ROOT}/benchmark/judge" \
      RESULTS_DIR="${RESULTS_DIR}" MAX_TOKENS=1024 MAX_RETRIES=40 \
      PARALLEL_WORKERS=64 JUDGE_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1/" \
      JUDGE_MODEL="${BASE_JUDGE_NAME}" JUDGE_MAX_RETRIES=40 \
      JUDGE_MAX_TOKENS=16 JUDGE_PARALLEL_WORKERS=32 JUDGE_ENABLE_THINKING=False \
      bash eval/run_eval.sh >> "${eval_log}" 2>&1
      eval_status=$?
      if (( eval_status == 0 )); then
        python scripts/verify_reproduction.py --project-root "${PROJECT_ROOT}" \
          --profile paper-explicit --model-name "${served_name}" \
          >> "${eval_log}" 2>&1 || eval_status=1
      fi
    else
      log "${served_name} vLLM server did not become ready."
    fi

    kill "${server_pid}" 2>/dev/null || true
    kill "${judge_server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    wait "${judge_server_pid}" 2>/dev/null || true
    pkill -f "vllm serve ${model_path}" 2>/dev/null || true
    if (( eval_status == 0 )); then
      log "${served_name} completed all six benchmarks."
      return 0
    fi
    log "${served_name} benchmark attempt failed; resumable outputs were retained."
    sleep $(( attempt < 6 ? attempt * 15 : 90 ))
  done
  return 1
}

wait_for_training

status=0
if [[ -f "${FOUR_B_EVAL_MARKER}" ]] && \
  python scripts/verify_reproduction.py --project-root "${PROJECT_ROOT}" \
    --profile paper-explicit --model-name Vision-OPD-Qwen3.5-4B-paper-explicit; then
  log "Verified completed 4B stage; skipping duplicate 4B evaluation."
else
  evaluate_model \
    "${PROJECT_ROOT}/merged_models/Vision-OPD-Qwen3.5-4B-paper-explicit" \
    Vision-OPD-Qwen3.5-4B-paper-explicit "0,1,2,3,4,5" "1" "6" "6" \
    "${LOG_DIR}/vllm_4b_paper_explicit_eval.log" \
    "${LOG_DIR}/benchmark_4b_paper_explicit_eval.log" || status=1
fi

evaluate_model \
  "${PROJECT_ROOT}/merged_models/Vision-OPD-Qwen3.5-9B-paper-explicit" \
  Vision-OPD-Qwen3.5-9B-paper-explicit "0,1,2,3,4,5" "2" "3" "6" \
  "${LOG_DIR}/vllm_9b_paper_explicit_eval.log" \
  "${LOG_DIR}/benchmark_9b_paper_explicit_eval.log" || status=1

python scripts/summarize_reproduction.py --project-root "${PROJECT_ROOT}" \
  --profile paper-explicit \
  --output "${PROJECT_ROOT}/docs/reproduction_results_paper_explicit.md" || status=1

python scripts/verify_reproduction.py --project-root "${PROJECT_ROOT}" \
  --profile paper-explicit --full || status=1

python scripts/summarize_goal_reproduction.py --project-root "${PROJECT_ROOT}" \
  --output "${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md" || status=1

if (( status == 0 )); then
  touch "${COMPLETE_MARKER}"
  python scripts/summarize_goal_reproduction.py --project-root "${PROJECT_ROOT}" \
    --output "${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md" || true
  log "Paper-explicit training, evaluation, verification, and report are complete."
else
  python scripts/summarize_goal_reproduction.py --project-root "${PROJECT_ROOT}" \
    --output "${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md" || true
  log "Paper-explicit evaluation pipeline ended with failures."
  exit 1
fi
