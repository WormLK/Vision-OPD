#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_BASE_DIR="${PROJECT_ROOT}/checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65/"
BASE_DIR="${BASE_DIR:-${1:-${DEFAULT_BASE_DIR}}}"
BASE_DIR="${BASE_DIR%/}"
ACTOR_DIR="${BASE_DIR}/actor"
TARGET_DIR="${TARGET_DIR:-${2:-${BASE_DIR}}}"
TARGET_DIR="${TARGET_DIR%/}"

if [ ! -d "${ACTOR_DIR}" ]; then
  echo "Actor checkpoint directory not found: ${ACTOR_DIR}" >&2
  exit 1
fi

echo "Merging ${ACTOR_DIR} -> ${TARGET_DIR}"

if [[ "${TARGET_DIR}" == "${BASE_DIR}" ]]; then
  # Preserve the historical in-place behavior for existing callers.
  find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type f -print -delete
else
  if [[ -d "${TARGET_DIR}" ]] && [[ -n "$(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Separate merge target is not empty: ${TARGET_DIR}" >&2
    exit 1
  fi
  mkdir -p "${TARGET_DIR}"
fi

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${TARGET_DIR}"

python3 "${PROJECT_ROOT}/scripts/validate_merged_model.py" "${TARGET_DIR}"

echo "Merge completed."
