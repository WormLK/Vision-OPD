#!/usr/bin/env bash
set -euo pipefail

ROOT="/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab"
OPD_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
MODEL_PATH="${MODEL_PATH:-${OPD_ROOT}/merged_models/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4}"
MODEL_NAME="${MODEL_NAME:-Vision-OPD-Qwen3.5-4B-released-b96-r8-official}"
OPD_MARKER="${OPD_MARKER:-${OPD_ROOT}/outputs/vision_opd_4b_step65_official_complete}"
PYTHON="/data00/users/wanglikun/anaconda3/envs/vtc-opd-eval/bin/python"
VLLM="/data00/users/wanglikun/anaconda3/envs/vision-opd/bin/vllm"
PORT="${PORT:-8000}"
LOG_DIR="${ROOT}/logs"
VLLM_LOG="${LOG_DIR}/vllm_vision_opd_4b_vtc.log"
RUN_LOG="${LOG_DIR}/vision_opd_4b_vtc_bench.log"
CODE_CONFIG="${ROOT}/eval/eval_config/vision_opd_qwen35_4b_code.yaml"
INTERFACE_CONFIG="${ROOT}/eval/eval_config/vision_opd_qwen35_4b_interface.yaml"
MODEL_OUTPUT_NAME="${MODEL_NAME}"
EVAL_OUTPUT_ROOT="${ROOT}/eval/VLMEvalKit/outputs/VTC_Bench"
CODE_SCORE="${EVAL_OUTPUT_ROOT}/Qwen-Agent-Code-RawAPI-Instruct-${MODEL_NAME}/${MODEL_NAME}_VTC_Bench_score.csv"
INTERFACE_SCORE="${EVAL_OUTPUT_ROOT}/Qwen-Agent-Interface-RawAPI-Instruct-${MODEL_NAME}/${MODEL_NAME}_VTC_Bench_score.csv"

mkdir -p "${LOG_DIR}" "${ROOT}/runs/vtc_vision_opd_4b_workspace"
cd "${ROOT}"

exec > >(tee -a "${RUN_LOG}") 2>&1
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Checking the selected step-65 Vision-OPD-4B official gate."
if [[ ! -f "${OPD_MARKER}" ]]; then
  echo "The selected 4B evaluation ended without its completion marker: ${OPD_MARKER}" >&2
  exit 4
fi

"${PYTHON}" scripts/repair_vtc_bench_gt_mapping.py --data-dir data/vtc_bench
PYTHONPATH="${ROOT}/eval:${ROOT}/eval/eval/VLMEvalKit" \
  "${PYTHON}" scripts/smoke_vtc_code_interpreter.py
/data00/users/wanglikun/anaconda3/envs/vision-opd/bin/python \
  scripts/smoke_qwen35_tool_parser.py --model-path "${MODEL_PATH}"

server_pid=""
stop_server() {
  if [[ -n "${server_pid}" ]]; then
    kill -TERM -- "-${server_pid}" 2>/dev/null || true
    sleep 5
    kill -KILL -- "-${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "${VLLM}" serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --host 127.0.0.1 --port "${PORT}" \
    --tensor-parallel-size 1 --data-parallel-size 8 \
    --max-model-len 131072 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --default-chat-template-kwargs '{"enable_thinking":true}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --trust-remote-code > "${VLLM_LOG}" 2>&1 &
server_pid=$!

for _ in $(seq 1 360); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${MODEL_NAME}"; then
    break
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${MODEL_NAME}"

export PYTHONPATH="${ROOT}/eval:${ROOT}/eval/eval/VLMEvalKit"
export QWEN_AGENT_DEFAULT_WORKSPACE="${ROOT}/runs/vtc_vision_opd_4b_workspace"
export M6_CODE_INTERPRETER_WORK_DIR="${ROOT}/runs/vtc_vision_opd_4b_workspace/code_interpreter"
export QWEN_AGENT_IMAGE_MAX_SHORT_SIDE="1080"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

run_track() {
  local label="$1" config="$2" results_dir="$3" score_file="$4"
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting VTC ${label} 680-row evaluation."
  for attempt in $(seq 1 20); do
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] ${label} attempt ${attempt}."
    "${PYTHON}" eval/VTC_Bench_Eval.py -c "${config}" --eval-method heuristic || true
    if "${PYTHON}" scripts/validate_vision_opd_vtc_track.py \
      --results-dir "${results_dir}" --model "${MODEL_OUTPUT_NAME}" \
      --score-file "${score_file}"; then
      return 0
    fi
    sleep $((attempt < 5 ? attempt * 30 : 300))
  done
  "${PYTHON}" scripts/validate_vision_opd_vtc_track.py \
    --results-dir "${results_dir}" --model "${MODEL_OUTPUT_NAME}" \
    --score-file "${score_file}"
}

# Both tracks share the DP8 server. Each retains the reference 30-worker
# evaluator setting, so this changes scheduling only, not model generation.
run_track "code-driven" "${CODE_CONFIG}" \
  runs/vtc_vision_opd_4b_step65_code "${CODE_SCORE}" &
code_pid=$!
run_track "interface-driven" "${INTERFACE_CONFIG}" \
  runs/vtc_vision_opd_4b_step65_interface "${INTERFACE_SCORE}" &
interface_pid=$!

code_status=0
interface_status=0
wait "${code_pid}" || code_status=$?
wait "${interface_pid}" || interface_status=$?
if [[ "${code_status}" -ne 0 || "${interface_status}" -ne 0 ]]; then
  echo "VTC tracks failed: code=${code_status}, interface=${interface_status}" >&2
  exit 1
fi

touch "${ROOT}/runs/vision_opd_4b_vtc_bench_complete"
"/data00/users/wanglikun/anaconda3/envs/vision-opd/bin/python" \
  "${OPD_ROOT}/scripts/summarize_4b_vtc_reproduction.py" \
  --project-root "${OPD_ROOT}" --vtc-root "${ROOT}" \
  --output "${OPD_ROOT}/docs/vision_opd_4b_vtc_reproduction.md"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Vision-OPD-4B VTC-Bench tracks complete."
