#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
SESSION="paper_explicit_evaluation_pipeline"
LOG="${PROJECT_ROOT}/logs/paper_explicit_evaluation_pipeline.log"
MARKER="${PROJECT_ROOT}/outputs/vision_opd_paper_explicit_reproduction_complete"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

while [[ ! -f "${MARKER}" ]]; do
  if ! screen -ls 2>/dev/null | rg -q "[.]${SESSION}[[:space:]]"; then
    printf '[%s] Restarting paper-explicit evaluation pipeline.\n' \
      "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    screen -L -Logfile "${LOG}" -dmS "${SESSION}" \
      bash scripts/paper_explicit_evaluation_pipeline.sh
  fi
  sleep 60
done

printf '[%s] Paper-explicit reproduction marker detected; watchdog exiting.\n' \
  "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
