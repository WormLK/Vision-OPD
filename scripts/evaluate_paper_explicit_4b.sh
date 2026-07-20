#!/usr/bin/env bash
set -uo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
MODEL_PATH="${PROJECT_ROOT}/merged_models/Vision-OPD-Qwen3.5-4B-paper-explicit"
SERVED_MODEL_NAME="Vision-OPD-Qwen3.5-4B-paper-explicit"
PREPARED_DIR="${PROJECT_ROOT}/benchmark/prepared"
RESULTS_DIR="${PROJECT_ROOT}/benchmark/results"
MODEL_ANSWER_DIR="${PROJECT_ROOT}/benchmark/model_answer"
JUDGE_DIR="${PROJECT_ROOT}/benchmark/judge"
LOG_DIR="${PROJECT_ROOT}/logs"
PORT="${PORT:-8000}"
MAX_ATTEMPTS="${MAX_EVAL_RETRIES:-100}"

mkdir -p "${RESULTS_DIR}" "${MODEL_ANSWER_DIR}" "${JUDGE_DIR}" "${LOG_DIR}" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

wait_for_server() {
  local attempt
  for attempt in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

run_eval_attempt() {
  local attempt="$1"
  local server_log="${LOG_DIR}/vllm_4b_paper_explicit_eval.log"
  local eval_log="${LOG_DIR}/benchmark_4b_paper_explicit_eval.log"
  log "Starting 4B benchmark attempt ${attempt}/${MAX_ATTEMPTS}."

  CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" MODEL_PATH="${MODEL_PATH}" \
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" PORT="${PORT}" \
    TENSOR_PARALLEL_SIZE=1 DATA_PARALLEL_SIZE=8 \
    MAX_MODEL_LEN=32768 GPU_MEMORY_UTILIZATION=0.85 \
    bash scripts/serve_qwen_vllm.sh >> "${server_log}" 2>&1 &
  local server_pid=$!
  local status=1

  if wait_for_server; then
    API_BASE="http://127.0.0.1:${PORT}/v1/" \
    OPENAI_MODEL_ID="${SERVED_MODEL_NAME}" MODEL_NAME="${SERVED_MODEL_NAME}" \
    ENABLE_THINKING=False \
    BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
    BENCHMARK_DATA_DIR="${PREPARED_DIR}" PREPARE_DATA=false \
    OUT_DIR="${MODEL_ANSWER_DIR}" JUDGE_OUT_DIR="${JUDGE_DIR}" \
    RESULTS_DIR="${RESULTS_DIR}" MAX_TOKENS=1024 MAX_RETRIES=40 \
    PARALLEL_WORKERS=64 JUDGE_API_BASE="http://127.0.0.1:${PORT}/v1/" \
    JUDGE_MODEL="${SERVED_MODEL_NAME}" JUDGE_MAX_RETRIES=40 \
    JUDGE_PARALLEL_WORKERS=64 JUDGE_ENABLE_THINKING=False \
    bash eval/run_eval.sh >> "${eval_log}" 2>&1
    status=$?
  else
    log "4B vLLM server did not become ready."
  fi

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
  ray stop --force >> "${eval_log}" 2>&1 || true
  return "${status}"
}

status=1
for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  run_eval_attempt "${attempt}" && { status=0; break; }
  log "4B benchmark attempt failed; existing JSONL outputs will be resumed."
  sleep $(( attempt < 6 ? attempt * 15 : 90 ))
done

if (( status == 0 )); then
  python scripts/verify_reproduction.py --project-root "${PROJECT_ROOT}" \
    --profile paper-explicit --model-name "${SERVED_MODEL_NAME}" || status=1
fi
python scripts/summarize_reproduction.py --project-root "${PROJECT_ROOT}" \
  --profile paper-explicit \
  --output "${PROJECT_ROOT}/docs/reproduction_results_paper_explicit.md" || status=1
python scripts/summarize_goal_reproduction.py --project-root "${PROJECT_ROOT}" \
  --output "${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md" || status=1

if (( status == 0 )); then
  touch "${PROJECT_ROOT}/outputs/vision_opd_paper_explicit_4b_evaluation_complete"
  log "4B six-benchmark inference, judge, aggregation, and partial report completed."
else
  log "4B evaluation did not complete after ${MAX_ATTEMPTS} attempts."
  exit 1
fi
