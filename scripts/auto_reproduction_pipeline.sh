#!/usr/bin/env bash
set -uo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
BENCHMARK_DIR="${PROJECT_ROOT}/benchmark"
PREPARED_DIR="${BENCHMARK_DIR}/prepared"
LOG_DIR="${PROJECT_ROOT}/logs"
FOUR_B_CKPT_DIR="${PROJECT_ROOT}/checkpoints/Vision-OPD-Qwen3.5-4B-full-repro-lowmem-20260714"
NINE_B_CKPT_DIR="${PROJECT_ROOT}/checkpoints/Vision-OPD-Qwen3.5-9B-local-lowmem-20260714"
MERGED_ROOT="${PROJECT_ROOT}/merged_models"
FOUR_B_MERGED_DIR="${MERGED_ROOT}/Vision-OPD-Qwen3.5-4B"
NINE_B_MERGED_DIR="${MERGED_ROOT}/Vision-OPD-Qwen3.5-9B"
FOUR_B_EXPECTED_STEP="${FOUR_B_EXPECTED_STEP:-779}"
NINE_B_EXPECTED_STEP="${NINE_B_EXPECTED_STEP:-779}"
MAX_TRAIN_RETRIES="${MAX_TRAIN_RETRIES:-100}"
MAX_EVAL_RETRIES="${MAX_EVAL_RETRIES:-20}"
PORT="${PORT:-8000}"

mkdir -p "${LOG_DIR}" "${BENCHMARK_DIR}/model_answer" \
  "${BENCHMARK_DIR}/judge" "${BENCHMARK_DIR}/results" "${MERGED_ROOT}"
cd "${PROJECT_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

latest_step() {
  local checkpoint_dir="$1"
  local marker="${checkpoint_dir}/latest_checkpointed_iteration.txt"
  if [[ -f "${marker}" ]]; then
    tr -cd '0-9' < "${marker}"
  else
    printf '0'
  fi
}

reset_incomplete_dataloader_state() {
  local checkpoint_dir="$1"
  local step="$2"
  local data_file="${checkpoint_dir}/global_step_${step}/data.pt"
  [[ -f "${data_file}" ]] || return 0
  local backup="${data_file}.resume_original"
  if [[ -e "${backup}" ]]; then
    backup="${data_file}.resume_original.$(date -u +'%Y%m%dT%H%M%SZ')"
  fi
  mv "${data_file}" "${backup}"
  log "Archived exhausted dataloader state ${data_file} -> ${backup}."
}

screen_is_running() {
  local session="$1"
  screen -ls 2>/dev/null | rg -q "[.]${session}[[:space:]]"
}

wait_for_existing_training() {
  local session="$1"
  [[ -z "${session}" ]] && return 0
  while screen_is_running "${session}"; do
    log "Waiting for existing training session ${session}; latest checkpoint is $(latest_step "${FOUR_B_CKPT_DIR}")."
    sleep 60
  done
}

run_training_until_complete() {
  local label="$1"
  local run_script="$2"
  local checkpoint_dir="$3"
  local expected_step="$4"
  local train_log="$5"
  local wait_session="${6:-}"

  wait_for_existing_training "${wait_session}"
  local attempt=0
  local step
  step="$(latest_step "${checkpoint_dir}")"
  while (( step < expected_step )); do
    attempt=$((attempt + 1))
    if (( attempt > MAX_TRAIN_RETRIES )); then
      log "${label} exceeded ${MAX_TRAIN_RETRIES} retries at checkpoint ${step}."
      return 1
    fi
    log "Starting ${label} attempt ${attempt} from checkpoint ${step}/${expected_step}."
    ray stop --force >> "${train_log}" 2>&1 || true
    unset PYTORCH_CUDA_ALLOC_CONF
    local succeeded=0
    if bash "${run_script}" >> "${train_log}" 2>&1; then
      succeeded=1
      log "${label} process exited successfully."
    else
      log "${label} process failed; it will be resumed from the latest saved checkpoint."
    fi
    step="$(latest_step "${checkpoint_dir}")"
    if (( succeeded == 1 && step >= expected_step )); then
      log "${label} completed its configured epoch at checkpoint ${step}."
      return 0
    fi
    if (( succeeded == 1 )); then
      log "${label} exited successfully but checkpoint ${step}/${expected_step} is incomplete; retrying."
      reset_incomplete_dataloader_state "${checkpoint_dir}" "${step}"
    fi
    if (( step < expected_step )); then
      sleep $(( attempt < 6 ? attempt * 10 : 60 ))
    fi
  done
  log "${label} reached checkpoint ${step}."
}

wait_for_benchmarks() {
  local required=(
    vstar.json zoombench.json hr_bench_4k.json hr_bench_8k.json
    MME_RealWorld.json MME_RealWorld_CN.json
  )
  while true; do
    local missing=0
    local item
    for item in "${required[@]}"; do
      [[ -f "${PREPARED_DIR}/${item}" ]] || missing=$((missing + 1))
    done
    (( missing == 0 )) && break
    if ! screen_is_running prepare_benchmarks_local_hr_20260714; then
      log "Preparing ${missing} missing benchmark artifacts."
      if ! bash scripts/prepare_benchmarks.sh >> "${LOG_DIR}/prepare_benchmarks_auto.log" 2>&1; then
        log "Benchmark preparation failed; retrying in 60 seconds."
        sleep 60
      fi
    else
      log "Benchmark preparation is still running; ${missing} artifacts remain."
      sleep 60
    fi
  done
  log "All six benchmark JSON files are ready."
}

merge_checkpoint() {
  local checkpoint="$1"
  local target="$2"
  if [[ -f "${target}/config.json" ]] && compgen -G "${target}/model*.safetensors" > /dev/null; then
    log "Merged model already exists at ${target}."
    return 0
  fi
  log "Merging FSDP checkpoint ${checkpoint} into ${target}."
  if [[ -d "${target}" ]]; then
    find "${target}" -mindepth 1 -maxdepth 1 -type f -delete
  fi
  TARGET_DIR="${target}" bash scripts/merge_checkpoint.sh "${checkpoint}" \
    >> "${LOG_DIR}/merge_checkpoints_auto.log" 2>&1
}

wait_for_server() {
  local attempt
  for attempt in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

evaluate_checkpoint() {
  local checkpoint="$1"
  local served_name="$2"
  local server_log="$3"
  local eval_log="$4"
  local cuda_visible_devices="${5:-0}"
  local tensor_parallel_size="${6:-1}"
  local attempt=0

  while (( attempt < MAX_EVAL_RETRIES )); do
    attempt=$((attempt + 1))
    log "Starting ${served_name} evaluation attempt ${attempt}."
    CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" MODEL_PATH="${checkpoint}" SERVED_MODEL_NAME="${served_name}" \
      PORT="${PORT}" TENSOR_PARALLEL_SIZE="${tensor_parallel_size}" MAX_MODEL_LEN=32768 \
      GPU_MEMORY_UTILIZATION=0.85 bash scripts/serve_qwen_vllm.sh \
      >> "${server_log}" 2>&1 &
    local server_pid=$!

    if wait_for_server; then
      API_BASE="http://127.0.0.1:${PORT}/v1/" \
      OPENAI_MODEL_ID="${served_name}" \
      MODEL_NAME="${served_name}" \
      ENABLE_THINKING=False \
      BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
      BENCHMARK_DATA_DIR="${PREPARED_DIR}" \
      PREPARE_DATA=false \
      OUT_DIR="${BENCHMARK_DIR}/model_answer" \
      JUDGE_OUT_DIR="${BENCHMARK_DIR}/judge" \
      RESULTS_DIR="${BENCHMARK_DIR}/results" \
      MAX_TOKENS=1024 MAX_RETRIES=20 PARALLEL_WORKERS=16 \
      JUDGE_API_BASE="http://127.0.0.1:${PORT}/v1/" \
      JUDGE_MODEL="${served_name}" JUDGE_MAX_RETRIES=20 \
      JUDGE_PARALLEL_WORKERS=16 JUDGE_ENABLE_THINKING=False \
      bash eval/run_eval.sh >> "${eval_log}" 2>&1
      local eval_status=$?
      if (( eval_status == 0 )); then
        python scripts/verify_reproduction.py \
          --project-root "${PROJECT_ROOT}" --model-name "${served_name}" \
          >> "${eval_log}" 2>&1 || eval_status=1
      fi
    else
      log "${served_name} server did not become ready."
      local eval_status=1
    fi

    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    pkill -f "vllm serve ${checkpoint}" 2>/dev/null || true
    if (( eval_status == 0 )); then
      log "${served_name} completed all benchmarks."
      return 0
    fi
    log "${served_name} evaluation failed; resumable outputs are retained."
    sleep $(( attempt < 6 ? attempt * 10 : 60 ))
  done
  return 1
}

pipeline_status=0

run_training_until_complete \
  Qwen3.5-4B scripts/run_vision_opd_local_4b_resume.sh \
  "${FOUR_B_CKPT_DIR}" "${FOUR_B_EXPECTED_STEP}" \
  "${LOG_DIR}/vision_opd_4b_auto_resume.log" \
  vision_opd_4b_local_resume_noexpand_20260714 || pipeline_status=1

ray stop --force >> "${LOG_DIR}/auto_pipeline.log" 2>&1 || true

FOUR_B_FINAL_STEP="$(latest_step "${FOUR_B_CKPT_DIR}")"
FOUR_B_FINAL="${FOUR_B_CKPT_DIR}/global_step_${FOUR_B_FINAL_STEP}"
merge_checkpoint "${FOUR_B_FINAL}" "${FOUR_B_MERGED_DIR}" || pipeline_status=1

run_training_until_complete \
  Qwen3.5-9B scripts/run_vision_opd_local_9b.sh \
  "${NINE_B_CKPT_DIR}" "${NINE_B_EXPECTED_STEP}" \
  "${LOG_DIR}/vision_opd_9b_auto.log" || pipeline_status=1

wait_for_benchmarks || pipeline_status=1
ray stop --force >> "${LOG_DIR}/auto_pipeline.log" 2>&1 || true

NINE_B_FINAL_STEP="$(latest_step "${NINE_B_CKPT_DIR}")"
NINE_B_FINAL="${NINE_B_CKPT_DIR}/global_step_${NINE_B_FINAL_STEP}"
merge_checkpoint "${NINE_B_FINAL}" "${NINE_B_MERGED_DIR}" || pipeline_status=1
evaluate_checkpoint "${FOUR_B_MERGED_DIR}" Vision-OPD-Qwen3.5-4B \
  "${LOG_DIR}/vllm_4b_eval.log" "${LOG_DIR}/benchmark_4b_eval.log" \
  "0" "1" || pipeline_status=1
evaluate_checkpoint "${NINE_B_MERGED_DIR}" Vision-OPD-Qwen3.5-9B \
  "${LOG_DIR}/vllm_9b_eval.log" "${LOG_DIR}/benchmark_9b_eval.log" \
  "0,1" "2" || pipeline_status=1

python scripts/summarize_reproduction.py \
  --project-root "${PROJECT_ROOT}" \
  --output "${PROJECT_ROOT}/docs/reproduction_results.md"

python scripts/verify_reproduction.py \
  --project-root "${PROJECT_ROOT}" --full || pipeline_status=1

if (( pipeline_status == 0 )); then
  touch "${PROJECT_ROOT}/outputs/vision_opd_reproduction_complete"
  log "Vision-OPD 4B/9B reproduction pipeline completed."
else
  log "Pipeline ended with one or more failed stages; inspect the stage logs."
  exit 1
fi
