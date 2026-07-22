#!/usr/bin/env bash
set -euo pipefail

ROOT="/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab"
OPD_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
PYTHON="/data00/users/wanglikun/anaconda3/envs/vtc-opd-eval/bin/python"
VLLM="/data00/users/wanglikun/anaconda3/envs/vision-opd/bin/vllm"
PORT="${PORT:-8000}"
WAIT_MARKER="${OPD_ROOT}/outputs/vision_opd_4b_goal_audit_complete"
FINAL_MARKER="${ROOT}/runs/qwen35_vtc_base_sequence_complete"
GOAL_MARKER="${OPD_ROOT}/outputs/qwen35_vtc_base_goal_complete"
AUDIT_LOG="${OPD_ROOT}/logs/qwen35_vtc_base_goal_audit.log"
LOG_DIR="${ROOT}/logs"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

update_report() {
  "/data00/users/wanglikun/anaconda3/envs/vision-opd/bin/python" \
    "${OPD_ROOT}/scripts/summarize_4b_vtc_reproduction.py" \
    --project-root "${OPD_ROOT}" --vtc-root "${ROOT}" \
    --output "${OPD_ROOT}/docs/vision_opd_4b_vtc_reproduction.md"
}

"/data00/users/wanglikun/anaconda3/envs/vision-opd/bin/python" \
  "${OPD_ROOT}/scripts/validate_vtc_base_configs.py" --vtc-root "${ROOT}"

if [[ -f "${FINAL_MARKER}" ]]; then
  log "All three Qwen3.5 VTC Base evaluations are already complete."
  exit 0
fi

while [[ ! -f "${WAIT_MARKER}" ]]; do
  log "Waiting for the OPD-4B code/interface completion audit before Base evaluation."
  sleep 300
done

# The preceding pipeline creates its marker just before the vLLM trap exits.
for _ in $(seq 1 60); do
  if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done
if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  log "Port ${PORT} is still occupied after the preceding pipeline completed."
  exit 5
fi

export PYTHONPATH="${ROOT}/eval:${ROOT}/eval/eval/VLMEvalKit"
export QWEN_AGENT_IMAGE_MAX_SHORT_SIDE="1080"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"
unset QWEN_AGENT_REPEATED_NO_TOOL_LIMIT
unset QWEN_AGENT_MAX_LLM_CALL_PER_RUN
unset QWEN_AGENT_STOP_ON_FINAL_ANSWER
unset QWEN_AGENT_TEXT_TOOL_CALL_MODE

server_pid=""
stop_server() {
  if [[ -n "${server_pid}" ]]; then
    kill -TERM -- "-${server_pid}" 2>/dev/null || true
    sleep 5
    kill -KILL -- "-${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    server_pid=""
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_model() {
  local label="$1" model_path="$2" model_name="$3" config="$4"
  local results_dir="$5" tp="$6" dp="$7"
  local marker="${ROOT}/runs/${label}_complete"
  local score_file="${ROOT}/eval/VLMEvalKit/outputs/VTC_Bench/Qwen-Agent-Base-RawAPI-Instruct-${model_name}/${model_name}_VTC_Bench_score.csv"
  local server_log="${LOG_DIR}/vllm_${label}.log"

  if [[ -f "${marker}" ]]; then
    log "Skipping completed Base track: ${label}."
    return 0
  fi
  log "Starting ${label}: model=${model_path}, DP=${dp}, TP=${tp}."
  setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "${VLLM}" serve "${model_path}" \
      --served-model-name "${model_name}" \
      --host 127.0.0.1 --port "${PORT}" \
      --tensor-parallel-size "${tp}" --data-parallel-size "${dp}" \
      --max-model-len 65536 --gpu-memory-utilization 0.90 \
      --enable-prefix-caching \
      --default-chat-template-kwargs '{"enable_thinking":true}' \
      --reasoning-parser qwen3 --trust-remote-code > "${server_log}" 2>&1 &
  server_pid=$!

  for _ in $(seq 1 360); do
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${model_name}"; then
      break
    fi
    sleep 5
  done
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${model_name}"

  for attempt in $(seq 1 20); do
    log "${label} inference attempt ${attempt}."
    "${PYTHON}" eval/VTC_Bench_Eval.py -c "${config}" --eval-method heuristic || true
    if "${PYTHON}" "${OPD_ROOT}/integrations/vtc_bench/scripts/validate_vtc_base_track.py" \
      --results-dir "${results_dir}" --model "${model_name}" --score-file "${score_file}"; then
      touch "${marker}"
      stop_server
      update_report
      return 0
    fi
    sleep $((attempt < 5 ? attempt * 30 : 300))
  done
  stop_server
  return 1
}

run_model \
  "vision_opd_qwen35_4b_base" \
  "${OPD_ROOT}/merged_models/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4" \
  "Vision-OPD-Qwen3.5-4B-released-b96-r8-base" \
  "${ROOT}/eval/eval_config/vision_opd_qwen35_4b_base.yaml" \
  "${ROOT}/runs/vtc_vision_opd_4b_step65_base" 1 8

run_model \
  "qwen35_4b_base" \
  "/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B" \
  "Qwen3.5-4B-base-vtc" \
  "${ROOT}/eval/eval_config/qwen35_4b_base.yaml" \
  "${ROOT}/runs/vtc_qwen35_4b_base" 1 8

run_model \
  "qwen35_9b_base" \
  "/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-9b" \
  "Qwen3.5-9B-base-vtc" \
  "${ROOT}/eval/eval_config/qwen35_9b_base.yaml" \
  "${ROOT}/runs/vtc_qwen35_9b_base" 2 4

update_report
"/data00/users/wanglikun/anaconda3/envs/vision-opd/bin/python" \
  "${OPD_ROOT}/integrations/vtc_bench/scripts/audit_qwen35_vtc_base_sequence.py" \
  --project-root "${OPD_ROOT}" --vtc-root "${ROOT}" | tee "${AUDIT_LOG}"
touch "${FINAL_MARKER}" "${GOAL_MARKER}"
log "All three Qwen3.5 VTC-Bench Base evaluations completed."
