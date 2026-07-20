#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${PROJECT_ROOT}/benchmark}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval}"
DEEPEYES_DATASETS="${DEEPEYES_DATASETS:-/data00/users/wanglikun/ProjWormLK/DeepEyes/datasets}"
PREPARE_MAX_RETRIES="${PREPARE_MAX_RETRIES:-20}"

# Xet stalled with zero-byte incomplete files on this host. Plain HTTP is
# slower per connection but resumes reliably through the local proxy.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_SNAPSHOT_MAX_WORKERS="${HF_SNAPSHOT_MAX_WORKERS:-16}"

mkdir -p "${BENCHMARK_ROOT}/raw" "${BENCHMARK_ROOT}/prepared"

link_or_copy() {
  local src="$1"
  local dst="$2"
  if [[ -e "${src}" && ! -e "${dst}" ]]; then
    mkdir -p "$(dirname "${dst}")"
    cp -al "${src}" "${dst}" 2>/dev/null || cp -a "${src}" "${dst}"
  fi
}

# Keep local raw copies from DeepEyes when available. The evaluator still uses
# its own normalized JSON files generated below.
link_or_copy "${DEEPEYES_DATASETS}/vstar_bench" "${BENCHMARK_ROOT}/raw/vstar_bench"
link_or_copy "${DEEPEYES_DATASETS}/HR-Bench" "${BENCHMARK_ROOT}/raw/HR-Bench"
link_or_copy "${DEEPEYES_DATASETS}/HRBENCH" "${BENCHMARK_ROOT}/raw/HRBENCH"

cd "${PROJECT_ROOT}"

prepare_with_retries() {
  local bench="$1"
  local attempt=1
  while (( attempt <= PREPARE_MAX_RETRIES )); do
    if python eval/prepare_data.py --benchmark "${bench}" --data_dir "${BENCHMARK_ROOT}/prepared"; then
      return 0
    fi
    local delay=$(( attempt < 6 ? attempt * 10 : 60 ))
    echo "Preparation failed for ${bench} (attempt ${attempt}/${PREPARE_MAX_RETRIES}); retrying in ${delay}s."
    sleep "${delay}"
    attempt=$((attempt + 1))
  done
  echo "Preparation exhausted retries for ${bench}." >&2
  return 1
}

for bench in vstar zoombench hrbench-4k hrbench-8k mme-realworld mme-realworld-cn; do
  echo "Preparing ${bench} into ${BENCHMARK_ROOT}/prepared"
  prepare_with_retries "${bench}"
done

python scripts/augment_mme_metadata.py --prepared-dir "${BENCHMARK_ROOT}/prepared"
HR_RAW_DIR="${BENCHMARK_ROOT}/raw/HR-Bench"
if [[ ! -f "${HR_RAW_DIR}/hr_bench_4k.parquet" ]]; then
  HR_RAW_DIR="${BENCHMARK_ROOT}/raw/HRBENCH"
fi
python scripts/augment_hrbench_metadata.py \
  --prepared-dir "${BENCHMARK_ROOT}/prepared" \
  --raw-dir "${HR_RAW_DIR}"

echo "Prepared benchmark files:"
find "${BENCHMARK_ROOT}" -maxdepth 2 -type f | sort
