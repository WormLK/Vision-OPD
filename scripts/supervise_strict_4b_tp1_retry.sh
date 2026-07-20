#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
EXPERIMENT="Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4-rollout-tp1-retry"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/${EXPERIMENT}"
TRAIN_SESSION="strict_4b_tp1_retry"
TRAIN_LOG="${PROJECT_ROOT}/logs/strict_4b_tp1_retry.screen.log"
SUPERVISOR_LOG="${PROJECT_ROOT}/logs/strict_4b_tp1_supervisor.log"
MAX_RESTARTS="${MAX_RESTARTS:-100}"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"
exec > >(tee -a "${SUPERVISOR_LOG}") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

latest_step() {
  local marker="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
  [[ -f "${marker}" ]] && tr -cd '0-9' < "${marker}" || printf '0'
}

session_running() {
  screen -ls 2>/dev/null | rg -q "[.]${TRAIN_SESSION}[[:space:]]"
}

restarts=0
while (( $(latest_step) < 65 )); do
  if session_running; then
    sleep 60
    continue
  fi

  step="$(latest_step)"
  if (( step > 0 )); then
    python scripts/validate_strict_checkpoint.py "${CHECKPOINT_ROOT}" --step "${step}"
  fi
  restarts=$((restarts + 1))
  if (( restarts > MAX_RESTARTS )); then
    log "Exceeded ${MAX_RESTARTS} automatic restart attempts at checkpoint ${step}."
    exit 1
  fi

  log "Training session is absent at checkpoint ${step}/65; launching resume attempt ${restarts}."
  ray stop --force || true
  screen -L -Logfile "${TRAIN_LOG}" -dmS "${TRAIN_SESSION}" \
    bash scripts/run_vision_opd_released_b96_r8_gradaccum_4b_tp1_retry.sh
  sleep 120
done

log "Strict TP1 4B reached checkpoint 65; supervisor complete."
