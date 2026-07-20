#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/Vision-OPD-Qwen3.5-4B-full-repro-lowmem-20260714"
MARKER="${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
PIPELINE_SESSION="auto_reproduction_pipeline_20260714"
PIPELINE_LOG="${PROJECT_ROOT}/logs/auto_reproduction_pipeline_20260714.log"

cd "${PROJECT_ROOT}"
while true; do
  step=0
  if [[ -f "${MARKER}" ]]; then
    step="$(tr -cd '0-9' < "${MARKER}")"
  fi
  (( step >= 779 )) && break
  sleep 30
done

# The final marker is written during checkpoint save. Let the training driver
# finish cleanly and enter the benchmark wait before replacing the controller.
sleep 120
screen -S "${PIPELINE_SESSION}" -X quit 2>/dev/null || true
sleep 3
screen -L -Logfile "${PIPELINE_LOG}" -dmS "${PIPELINE_SESSION}" \
  bash scripts/auto_reproduction_pipeline.sh
