#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4"
MERGED_ROOT="${PROJECT_ROOT}/merged_models/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4"
READY_MARKER="${PROJECT_ROOT}/outputs/vision_opd_official_existing_checkpoints_complete"
LOG="${PROJECT_ROOT}/logs/watch_post_strict_4b_prerequisites.log"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "${LOG}"
}

latest_step() {
  local marker="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
  [[ -f "${marker}" ]] && tr -cd '0-9' < "${marker}" || printf '0'
}

while (( $(latest_step) < 65 )); do
  log "Waiting for strict 4B checkpoint: $(latest_step)/65."
  sleep 300
done

while ! python scripts/validate_merged_model.py "${MERGED_ROOT}" >> "${LOG}" 2>&1; do
  log "Strict 4B reached 65; waiting for merge and merged-model validation."
  sleep 60
done

for attempt in $(seq 1 20); do
  [[ -f "${READY_MARKER}" ]] && break
  log "Running post-4B official prerequisite pipeline, attempt ${attempt}."
  if bash scripts/continue_official_eval_after_judge_download.sh >> "${LOG}" 2>&1; then
    break
  fi
  sleep $((attempt < 5 ? attempt * 60 : 300))
done

[[ -f "${READY_MARKER}" ]]
log "Post-4B official prerequisite marker is ready."
