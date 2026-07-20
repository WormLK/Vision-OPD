#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
OFFICIAL_ROOT="${PROJECT_ROOT}/benchmark/official_reproduction_20260717"
EVAL_DIR="${OFFICIAL_ROOT}/source/eval"
JUDGE_PATH="/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/OpenAI/gpt-oss-120b"
JUDGE_NAME="openai/gpt-oss-120b"
JQ="/data00/users/wanglikun/anaconda3/bin/jq"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"
MODEL_TP="${MODEL_TP:-1}"
MODEL_DP="${MODEL_DP:-6}"
MODEL_PORT="${MODEL_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_CONTEXT_LEN="${JUDGE_CONTEXT_LEN:-65536}"
LOG_DIR="${OFFICIAL_ROOT}/logs"
BENCHMARKS="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn,mmstar,pope,cv-bench,mmvp"

declare -A MANIFESTS=(
  [vstar]="vstar.json"
  [zoombench]="zoombench.json"
  [hrbench-4k]="hr_bench_4k.json"
  [hrbench-8k]="hr_bench_8k.json"
  [mme-realworld]="MME_RealWorld.json"
  [mme-realworld-cn]="MME_RealWorld_CN.json"
  [mmstar]="mmstar.json"
  [pope]="POPE.json"
  [cv-bench]="cv_bench.json"
  [mmvp]="mmvp.json"
)
declare -A EXPECTED_COUNTS=(
  [vstar]=191
  [zoombench]=845
  [hrbench-4k]=800
  [hrbench-8k]=800
  [mme-realworld]=23609
  [mme-realworld-cn]=5462
  [mmstar]=1500
  [pope]=9000
  [cv-bench]=2638
  [mmvp]=300
)

mkdir -p "${LOG_DIR}" "${OFFICIAL_ROOT}/results"
cd "${PROJECT_ROOT}"

model_pid=""
judge_pid=""

stop_group() {
  local pid="${1:-}"
  [[ -z "${pid}" ]] && return 0
  kill -TERM -- "-${pid}" 2>/dev/null || true
  sleep 3
  kill -KILL -- "-${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
}

cleanup() {
  stop_group "${model_pid}"
  stop_group "${judge_pid}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_server() {
  local port="$1" name="$2"
  local response=""
  for _ in $(seq 1 360); do
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

judge_complete() {
  local benchmark="$1" tag="$2"
  local answer="${OFFICIAL_ROOT}/model_answer/${benchmark}/${tag}_answer.jsonl"
  local path="${OFFICIAL_ROOT}/judge/${benchmark}/${tag}_answer.jsonl"
  [[ -s "${answer}" ]] || return 1
  [[ "$(wc -l < "${answer}")" -eq "${EXPECTED_COUNTS[$benchmark]}" ]] || return 1
  ! rg -q '"model_answer"\s*:\s*"\[(API|FUTURE)_ERROR\]' "${answer}" || return 1
  [[ -s "${path}" ]] || return 1
  "${JQ}" -e --argjson expected "${EXPECTED_COUNTS[$benchmark]}" \
    'length == $expected and all(.[]; ((.judge // "") | ascii_downcase) == "yes" or ((.judge // "") | ascii_downcase) == "no")' \
    "${path}" >/dev/null
}

python "${PROJECT_ROOT}/scripts/validate_gpt_oss_judge.py" "${JUDGE_PATH}"

setsid env CUDA_VISIBLE_DEVICES=6,7 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  vllm serve "${JUDGE_PATH}" --served-model-name "${JUDGE_NAME}" \
    --host 127.0.0.1 --port "${JUDGE_PORT}" --tensor-parallel-size 2 \
    --max-model-len "${JUDGE_CONTEXT_LEN}" --gpu-memory-utilization 0.92 \
    --reasoning-parser openai_gptoss --trust-remote-code \
    > "${LOG_DIR}/vllm_${SERVED_MODEL_NAME}_judge.log" 2>&1 &
judge_pid=$!

setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  vllm serve "${MODEL_PATH}" --served-model-name "${SERVED_MODEL_NAME}" \
    --host 127.0.0.1 --port "${MODEL_PORT}" \
    --tensor-parallel-size "${MODEL_TP}" --data-parallel-size "${MODEL_DP}" \
    --max-model-len 65536 --gpu-memory-utilization 0.90 --trust-remote-code \
    > "${LOG_DIR}/vllm_${SERVED_MODEL_NAME}.log" 2>&1 &
model_pid=$!

wait_for_server "${JUDGE_PORT}" "${JUDGE_NAME}"
wait_for_server "${MODEL_PORT}" "${SERVED_MODEL_NAME}"

tag="${SERVED_MODEL_NAME}_seed42"
IFS=',' read -r -a benchmark_array <<< "${BENCHMARKS}"
for benchmark in "${benchmark_array[@]}"; do
  if judge_complete "${benchmark}" "${tag}"; then
    printf '[%s] Reusing complete judge output for %s/%s.\n' \
      "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${SERVED_MODEL_NAME}" "${benchmark}" \
      >> "${LOG_DIR}/official_eval_${SERVED_MODEL_NAME}.log"
  else
    HF_HUB_OFFLINE=1 \
    API_BASE="http://127.0.0.1:${MODEL_PORT}/v1/" \
    OPENAI_MODEL_ID="${SERVED_MODEL_NAME}" MODEL_NAME="${SERVED_MODEL_NAME}" \
    ENABLE_THINKING=False BENCHMARK="${benchmark}" \
    OUT_DIR="${OFFICIAL_ROOT}/model_answer" \
    SEED=42 MAX_TOKENS=32768 MAX_RETRIES=3 PARALLEL_WORKERS=256 \
    JUDGE_API_BASE="http://127.0.0.1:${JUDGE_PORT}/v1/" \
    JUDGE_MODEL="${JUDGE_NAME}" JUDGE_MAX_TOKENS=2048 \
      bash "${EVAL_DIR}/run_eval.sh" \
      >> "${LOG_DIR}/official_eval_${SERVED_MODEL_NAME}.log" 2>&1
    judge_complete "${benchmark}" "${tag}"
  fi
  answer="${OFFICIAL_ROOT}/model_answer/${benchmark}/${tag}_answer.jsonl"
  judge="${OFFICIAL_ROOT}/judge/${benchmark}/${tag}_answer.jsonl"
  [[ "$(wc -l < "${answer}")" -eq "${EXPECTED_COUNTS[$benchmark]}" ]]
  ! rg -q '"model_answer"\s*:\s*"\[(API|FUTURE)_ERROR\]' "${answer}"
  "${JQ}" -e --argjson expected "${EXPECTED_COUNTS[$benchmark]}" \
    'length == $expected and all(.[]; ((.judge // "") | ascii_downcase) == "yes" or ((.judge // "") | ascii_downcase) == "no")' \
    "${judge}" >/dev/null
  python "${EVAL_DIR}/cal_acc.py" --benchmark "${benchmark}" \
    --judge_json "${judge}" --benchmark_json "${EVAL_DIR}/${MANIFESTS[$benchmark]}" \
    > "${OFFICIAL_ROOT}/results/${tag}_${benchmark}.txt"
done

python "${PROJECT_ROOT}/scripts/validate_official_model_outputs.py" \
  --project-root "${PROJECT_ROOT}" --model "${SERVED_MODEL_NAME}"
touch "${PROJECT_ROOT}/outputs/${SERVED_MODEL_NAME}_official_evaluation_complete"
