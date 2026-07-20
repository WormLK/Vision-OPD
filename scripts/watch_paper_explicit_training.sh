#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
SESSION="paper_explicit_training_pipeline"
LOG="${PROJECT_ROOT}/logs/paper_explicit_training_pipeline.log"
TRAIN_LOG_4B="${PROJECT_ROOT}/logs/vision_opd_4b_paper_explicit.log"
TRAIN_LOG_9B="${PROJECT_ROOT}/logs/vision_opd_9b_paper_explicit.log"
MARKER="${PROJECT_ROOT}/outputs/vision_opd_paper_explicit_training_complete"
STALE_SECONDS="${STALE_SECONDS:-3600}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

while [[ ! -f "${MARKER}" ]]; do
  if ! screen -ls 2>/dev/null | rg -q "[.]${SESSION}[[:space:]]"; then
    printf '[%s] Restarting paper-explicit training pipeline.\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    screen -L -Logfile "${LOG}" -dmS "${SESSION}" \
      bash scripts/paper_explicit_training_pipeline.sh
  elif pgrep -f 'python3 -m verl[.]trainer[.]main_ppo' >/dev/null; then
    active_log="${TRAIN_LOG_4B}"
    if [[ -f "${TRAIN_LOG_9B}" ]] && \
      { [[ ! -f "${TRAIN_LOG_4B}" ]] || [[ "${TRAIN_LOG_9B}" -nt "${TRAIN_LOG_4B}" ]]; }; then
      active_log="${TRAIN_LOG_9B}"
    fi
    [[ -f "${active_log}" ]] || { sleep 60; continue; }
    now="$(date +%s)"
    modified="$(stat -c %Y "${active_log}")"
    if (( now - modified > STALE_SECONDS )); then
      printf '[%s] Training log is stale for more than %s seconds; restarting pipeline.\n' \
        "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${STALE_SECONDS}"
      screen -S "${SESSION}" -X quit || true
    fi
  fi
  sleep 60
done

printf '[%s] Paper-explicit training marker detected; watchdog exiting.\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
