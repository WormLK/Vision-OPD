#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
OFFICIAL_ROOT="${PROJECT_ROOT}/benchmark/official_reproduction_20260717"
EVAL_DIR="${OFFICIAL_ROOT}/source/eval"
ANSWER_DIR="${OFFICIAL_ROOT}/model_answer"
JUDGE_DIR="${OFFICIAL_ROOT}/judge"
RESULTS_DIR="${OFFICIAL_ROOT}/results"
LOG_DIR="${OFFICIAL_ROOT}/logs"
JUDGE_PATH="/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/OpenAI/gpt-oss-120b"
JUDGE_NAME="openai/gpt-oss-120b"
JQ="/data00/users/wanglikun/anaconda3/bin/jq"
MODEL_PORT="${MODEL_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
MAX_PIPELINE_ATTEMPTS="${MAX_PIPELINE_ATTEMPTS:-100}"
BENCHMARKS="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn"

declare -A BENCHMARK_FILES=(
  [vstar]="vstar.json"
  [zoombench]="zoombench.json"
  [hrbench-4k]="hr_bench_4k.json"
  [hrbench-8k]="hr_bench_8k.json"
  [mme-realworld]="MME_RealWorld.json"
  [mme-realworld-cn]="MME_RealWorld_CN.json"
)
declare -A EXPECTED_COUNTS=(
  [vstar]=191
  [zoombench]=845
  [hrbench-4k]=800
  [hrbench-8k]=800
  [mme-realworld]=23609
  [mme-realworld-cn]=5462
)

mkdir -p "${ANSWER_DIR}" "${JUDGE_DIR}" "${RESULTS_DIR}" "${LOG_DIR}" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

model_server_pid=""
judge_server_pid=""

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

stop_group() {
  local pid="${1:-}"
  [[ -z "${pid}" ]] && return 0
  kill -TERM -- "-${pid}" 2>/dev/null || true
  sleep 3
  kill -KILL -- "-${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  stop_group "${model_server_pid}"
  stop_group "${judge_server_pid}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
  local port="$1" name="$2"
  local attempt response=""
  for attempt in $(seq 1 240); do
    if response="$(curl -fsS "http://127.0.0.1:${port}/v1/models")"; then
      if "${JQ}" -e --arg name "${name}" 'any(.data[]; .id == $name)' \
        <<< "${response}" >/dev/null 2>&1; then
        return 0
      fi
    fi
    sleep 5
  done
  return 1
}

validate_judge_weights() {
  [[ -s "${JUDGE_PATH}/model.safetensors.index.json" ]] || return 1
  python - "${JUDGE_PATH}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
if len(shards) != 15:
    raise SystemExit(f"expected 15 shards, found {len(shards)}")
for shard in shards:
    path = root / shard
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"missing shard: {path}")
print(f"validated judge index and {len(shards)} shards")
PY
}

start_judge() {
  log "Starting official ${JUDGE_NAME} judge on GPUs 6,7."
  setsid env CUDA_VISIBLE_DEVICES=6,7 VLLM_WORKER_MULTIPROC_METHOD=spawn \
    vllm serve "${JUDGE_PATH}" \
      --served-model-name "${JUDGE_NAME}" \
      --host 127.0.0.1 --port "${JUDGE_PORT}" \
      --tensor-parallel-size 2 --max-model-len 8192 \
      --gpu-memory-utilization 0.92 --reasoning-parser openai_gptoss \
      --trust-remote-code \
      > "${LOG_DIR}/vllm_gpt_oss_120b_judge.log" 2>&1 &
  judge_server_pid=$!
  wait_for_server "${JUDGE_PORT}" "${JUDGE_NAME}"
}

start_model() {
  local path="$1" name="$2" tp="$3" dp="$4"
  log "Starting ${name} with TP=${tp}, DP=${dp} on GPUs 0-5."
  setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 VLLM_WORKER_MULTIPROC_METHOD=spawn \
    vllm serve "${path}" \
      --served-model-name "${name}" \
      --host 127.0.0.1 --port "${MODEL_PORT}" \
      --tensor-parallel-size "${tp}" --data-parallel-size "${dp}" \
      --max-model-len 65536 --gpu-memory-utilization 0.90 \
      --trust-remote-code \
      > "${LOG_DIR}/vllm_${name}.log" 2>&1 &
  model_server_pid=$!
  wait_for_server "${MODEL_PORT}" "${name}"
}

verify_model_outputs() {
  local name="$1" tag="${name}_seed42" benchmark answer judge count
  IFS=',' read -r -a benchmark_array <<< "${BENCHMARKS}"
  for benchmark in "${benchmark_array[@]}"; do
    answer="${ANSWER_DIR}/${benchmark}/${tag}_answer.jsonl"
    judge="${JUDGE_DIR}/${benchmark}/${tag}_answer.jsonl"
    [[ -s "${answer}" && -s "${judge}" ]] || return 1
    count="$(wc -l < "${answer}")"
    [[ "${count}" -eq "${EXPECTED_COUNTS[$benchmark]}" ]] || return 1
    if rg -q '"model_answer"\s*:\s*"\[(API|FUTURE)_ERROR\]' "${answer}"; then
      return 1
    fi
    "${JQ}" -e --argjson expected "${EXPECTED_COUNTS[$benchmark]}" \
      'length == $expected and all(.[]; ((.judge // "") | ascii_downcase) == "yes" or ((.judge // "") | ascii_downcase) == "no")' \
      "${judge}" >/dev/null || return 1
  done
}

save_scores() {
  local name="$1" tag="${name}_seed42" benchmark manifest
  IFS=',' read -r -a benchmark_array <<< "${BENCHMARKS}"
  for benchmark in "${benchmark_array[@]}"; do
    manifest="${EVAL_DIR}/${BENCHMARK_FILES[$benchmark]}"
    python "${EVAL_DIR}/cal_acc.py" \
      --benchmark "${benchmark}" \
      --judge_json "${JUDGE_DIR}/${benchmark}/${tag}_answer.jsonl" \
      --benchmark_json "${manifest}" \
      > "${RESULTS_DIR}/${tag}_${benchmark}.txt"
  done
}

evaluate_model() {
  local path="$1" name="$2" tp="$3" dp="$4" attempt status
  for attempt in $(seq 1 "${MAX_PIPELINE_ATTEMPTS}"); do
    status=0
    start_model "${path}" "${name}" "${tp}" "${dp}" || status=1
    if (( status == 0 )); then
      log "Running pristine official eval/run_eval.sh for ${name}, attempt ${attempt}."
      HF_HUB_OFFLINE=1 \
      API_BASE="http://127.0.0.1:${MODEL_PORT}/v1/" \
      OPENAI_MODEL_ID="${name}" MODEL_NAME="${name}" ENABLE_THINKING=False \
      BENCHMARK="${BENCHMARKS}" OUT_DIR="${ANSWER_DIR}" \
      SEED=42 MAX_TOKENS=32768 MAX_RETRIES=3 PARALLEL_WORKERS=256 \
      JUDGE_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1/" \
      JUDGE_MODEL="${JUDGE_NAME}" JUDGE_MAX_TOKENS=2048 \
      bash "${EVAL_DIR}/run_eval.sh" \
      > "${LOG_DIR}/official_eval_${name}_attempt${attempt}.log" 2>&1 || status=1
    fi
    if (( status == 0 )) && verify_model_outputs "${name}"; then
      save_scores "${name}"
      if python scripts/validate_official_model_outputs.py \
        --project-root "${PROJECT_ROOT}" --model "${name}"; then
        stop_group "${model_server_pid}"
        model_server_pid=""
        log "Verified all six official outputs for ${name}."
        return 0
      fi
    fi
    stop_group "${model_server_pid}"
    model_server_pid=""
    log "${name} attempt ${attempt} failed validation; retained resumable inference and retrying."
    sleep $((attempt < 10 ? attempt * 15 : 150))
  done
  return 1
}

if ! validate_judge_weights; then
  log "ERROR: official GPT-OSS-120B judge is not complete."
  exit 1
fi

start_judge

evaluate_model \
  "/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B" \
  "Qwen3.5-4B-baseline-official" 1 6
evaluate_model \
  "/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-9b" \
  "Qwen3.5-9B-baseline-official" 2 3
evaluate_model \
  "${PROJECT_ROOT}/merged_models/Vision-OPD-Qwen3.5-4B-paper-explicit" \
  "Vision-OPD-Qwen3.5-4B-official" 1 6
evaluate_model \
  "${PROJECT_ROOT}/merged_models/Vision-OPD-Qwen3.5-9B-paper-explicit" \
  "Vision-OPD-Qwen3.5-9B-official" 2 3

stop_group "${judge_server_pid}"
judge_server_pid=""
touch "${PROJECT_ROOT}/outputs/vision_opd_official_four_model_evaluation_complete"
log "All four models passed strict official evaluation validation."
