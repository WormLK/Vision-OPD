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
JUDGE_PORT="${JUDGE_PORT:-8001}"
JUDGE_CONTEXT_LEN="${JUDGE_CONTEXT_LEN:-65536}"
LOG_DIR="${OFFICIAL_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${OFFICIAL_ROOT}/results" "${PROJECT_ROOT}/outputs"

declare -A MANIFESTS=(
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
BENCHMARKS=(vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn)
MODELS=(Qwen3.5-4B-baseline-official Qwen3.5-9B-baseline-official)

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    kill -TERM -- "-${server_pid}" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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

python "${PROJECT_ROOT}/scripts/verify_official_baseline_alignment.py" \
  --project-root "${PROJECT_ROOT}" --inference-only

# Training has completed before this prerequisite starts, but detached Ray
# workers can retain CUDA contexts after the trainer process exits.
ray stop --force >> "${LOG_DIR}/ray_stop_before_baseline_judge.log" 2>&1 || true

setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  vllm serve "${JUDGE_PATH}" --served-model-name "${JUDGE_NAME}" \
    --host 127.0.0.1 --port "${JUDGE_PORT}" --tensor-parallel-size 8 \
    --max-model-len "${JUDGE_CONTEXT_LEN}" --gpu-memory-utilization 0.90 \
    --reasoning-parser openai_gptoss --trust-remote-code \
    > "${LOG_DIR}/vllm_gpt_oss_120b_baseline_judge.log" 2>&1 &
server_pid=$!

ready=false
response=""
for _ in $(seq 1 240); do
  if response="$(curl -fsS "http://127.0.0.1:${JUDGE_PORT}/v1/models")"; then
    if "${JQ}" -e --arg name "${JUDGE_NAME}" 'any(.data[]; .id == $name)' \
      <<< "${response}" >/dev/null 2>&1; then
      ready=true
      break
    fi
  fi
  sleep 5
done
[[ "${ready}" == "true" ]]

cd "${EVAL_DIR}"
for model in "${MODELS[@]}"; do
  tag="${model}_seed42"
  for benchmark in "${BENCHMARKS[@]}"; do
    if judge_complete "${benchmark}" "${tag}"; then
      printf '[%s] Reusing complete judge output for %s/%s.\n' \
        "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${model}" "${benchmark}"
    else
      python judge_qwenlm.py --benchmark "${benchmark}" --model "${tag}" \
        --api_base "http://127.0.0.1:${JUDGE_PORT}/v1/" \
        --api_key EMPTY --judge_model "${JUDGE_NAME}" --judge_max_tokens 2048 \
        > "${LOG_DIR}/official_judge_${model}_${benchmark}.log" 2>&1
      judge_complete "${benchmark}" "${tag}"
    fi
    python cal_acc.py --benchmark "${benchmark}" \
      --judge_json "${OFFICIAL_ROOT}/judge/${benchmark}/${tag}_answer.jsonl" \
      --benchmark_json "${EVAL_DIR}/${MANIFESTS[$benchmark]}" \
      > "${OFFICIAL_ROOT}/results/${tag}_${benchmark}.txt"
  done
done

cd "${PROJECT_ROOT}"
python scripts/verify_official_baseline_alignment.py --project-root "${PROJECT_ROOT}"
python scripts/summarize_official_evaluation.py --project-root "${PROJECT_ROOT}"
python scripts/summarize_goal_reproduction.py --project-root "${PROJECT_ROOT}" \
  --output "${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md"
touch "${PROJECT_ROOT}/outputs/vision_opd_official_baselines_aligned"
