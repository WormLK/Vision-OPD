#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
EVAL_DIR="${PROJECT_ROOT}/benchmark/official_reproduction_20260717/source/eval"
LOG_DIR="${PROJECT_ROOT}/benchmark/official_reproduction_20260717/logs"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-100}"

BENCHMARKS=(
  zoombench vstar hrbench-4k hrbench-8k
  mme-realworld mme-realworld-cn mme-realworld-lite
  mmstar pope pope_adv pope_pop pope_random cv-bench mmvp visualprobe
)

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN="${HF_HUB_DISABLE_IMPLICIT_TOKEN:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ ! -f "${EVAL_DIR}/MME_RealWorld_Lite.json" ]]; then
  bash "${PROJECT_ROOT}/scripts/prefetch_mme_realworld_lite.sh"
fi

for benchmark in "${BENCHMARKS[@]}"; do
  complete=false
  for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    printf '[%s] Preparing %s, attempt %s.\n' \
      "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${benchmark}" "${attempt}"
    if python "${EVAL_DIR}/prepare_data.py" --benchmark "${benchmark}" \
      --data_dir "${EVAL_DIR}" \
      >> "${LOG_DIR}/prepare_all_${benchmark}.log" 2>&1; then
      complete=true
      break
    fi
    sleep $((attempt < 10 ? attempt * 30 : 300))
  done
  [[ "${complete}" == "true" ]]
done

python scripts/validate_all_official_benchmarks.py \
  --eval-dir "${EVAL_DIR}" \
  --marker "${PROJECT_ROOT}/outputs/vision_opd_all_official_benchmarks_prepared"
