#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
ROOT="${PROJECT_ROOT}/benchmark/official_reproduction_20260717/diagnostic_qwen25_72b"
EVAL_DIR="${ROOT}/eval"
JUDGE_NAME="Qwen2.5-72B-Instruct-diagnostic-judge"
API_BASE="http://127.0.0.1:8002/v1/"

declare -A MANIFESTS=(
  [vstar]="vstar.json"
  [zoombench]="zoombench.json"
  [hrbench-4k]="hr_bench_4k.json"
  [hrbench-8k]="hr_bench_8k.json"
  [mme-realworld]="MME_RealWorld.json"
  [mme-realworld-cn]="MME_RealWorld_CN.json"
)
BENCHMARKS=(vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn)
MODELS=(Qwen3.5-4B-baseline-official Qwen3.5-9B-baseline-official)

mkdir -p "${ROOT}/logs" "${ROOT}/results"
cd "${EVAL_DIR}"

for model in "${MODELS[@]}"; do
  tag="${model}_seed42"
  for benchmark in "${BENCHMARKS[@]}"; do
    python judge_qwenlm.py --benchmark "${benchmark}" --model "${tag}" \
      --api_base "${API_BASE}" --api_key EMPTY --judge_model "${JUDGE_NAME}" \
      --judge_max_tokens 16 \
      > "${ROOT}/logs/${tag}_${benchmark}.log" 2>&1
    python cal_acc.py --benchmark "${benchmark}" \
      --judge_json "${ROOT}/judge/${benchmark}/${tag}_answer.jsonl" \
      --benchmark_json "${EVAL_DIR}/${MANIFESTS[$benchmark]}" \
      > "${ROOT}/results/${tag}_${benchmark}.txt"
  done
done

touch "${ROOT}/complete"
