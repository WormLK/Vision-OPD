#!/usr/bin/env bash
set -uo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
LOG_DIR="${PROJECT_ROOT}/logs"
MERGED_ROOT="${PROJECT_ROOT}/merged_models"
FOUR_B_CKPT="${PROJECT_ROOT}/checkpoints/Vision-OPD-Qwen3.5-4B-paper-explicit-local"
NINE_B_CKPT="${PROJECT_ROOT}/checkpoints/Vision-OPD-Qwen3.5-9B-paper-explicit-local"
FOUR_B_MERGED="${MERGED_ROOT}/Vision-OPD-Qwen3.5-4B-paper-explicit"
NINE_B_MERGED="${MERGED_ROOT}/Vision-OPD-Qwen3.5-9B-paper-explicit"
EXPECTED_STEP="${EXPECTED_STEP:-780}"
MAX_RETRIES="${MAX_RETRIES:-100}"
COMPLETE_MARKER="${PROJECT_ROOT}/outputs/vision_opd_paper_explicit_training_complete"

mkdir -p "${LOG_DIR}" "${MERGED_ROOT}" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

latest_step() {
  local marker="$1/latest_checkpointed_iteration.txt"
  if [[ -f "${marker}" ]]; then
    tr -cd '0-9' < "${marker}"
  else
    printf '0'
  fi
}

reset_dataloader_state() {
  local checkpoint_root="$1"
  local step="$2"
  local state="${checkpoint_root}/global_step_${step}/data.pt"
  [[ -f "${state}" ]] || return 0
  local backup="${state}.resume_original"
  if [[ -e "${backup}" ]]; then
    backup="${state}.resume_original.$(date -u +'%Y%m%dT%H%M%SZ')"
  fi
  mv "${state}" "${backup}"
  log "Archived exhausted dataloader state ${state} -> ${backup}."
}

train_until_complete() {
  local label="$1"
  local run_script="$2"
  local checkpoint_root="$3"
  local train_log="$4"
  local attempt=0
  local step
  step="$(latest_step "${checkpoint_root}")"
  while (( step < EXPECTED_STEP )); do
    attempt=$((attempt + 1))
    if (( attempt > MAX_RETRIES )); then
      log "${label} exceeded ${MAX_RETRIES} retries at ${step}/${EXPECTED_STEP}."
      return 1
    fi
    log "Starting ${label} attempt ${attempt} from ${step}/${EXPECTED_STEP}."
    ray stop --force >> "${train_log}" 2>&1 || true
    local succeeded=0
    if bash "${run_script}" >> "${train_log}" 2>&1; then
      succeeded=1
      log "${label} process exited successfully."
    else
      log "${label} process failed; resuming from the latest checkpoint."
    fi
    step="$(latest_step "${checkpoint_root}")"
    if (( step >= EXPECTED_STEP )); then
      log "${label} reached checkpoint ${step}."
      return 0
    fi
    if (( succeeded == 1 )); then
      log "${label} exited at incomplete checkpoint ${step}/${EXPECTED_STEP}."
      reset_dataloader_state "${checkpoint_root}" "${step}"
    fi
    sleep $(( attempt < 6 ? attempt * 10 : 60 ))
  done
}

merge_checkpoint() {
  local checkpoint_root="$1"
  local target="$2"
  local step
  step="$(latest_step "${checkpoint_root}")"
  if python scripts/verify_reproduction.py --project-root "${PROJECT_ROOT}" \
      --merged-model-dir "${target}" >> "${LOG_DIR}/paper_explicit_merge.log" 2>&1; then
    log "Merged model already exists at ${target}."
    return 0
  fi
  if [[ -d "${target}" ]]; then
    find "${target}" -mindepth 1 -maxdepth 1 -type f -delete
  fi
  log "Merging ${checkpoint_root}/global_step_${step} -> ${target}."
  TARGET_DIR="${target}" bash scripts/merge_checkpoint.sh \
    "${checkpoint_root}/global_step_${step}" >> "${LOG_DIR}/paper_explicit_merge.log" 2>&1
  python scripts/verify_reproduction.py --project-root "${PROJECT_ROOT}" \
    --merged-model-dir "${target}" >> "${LOG_DIR}/paper_explicit_merge.log" 2>&1
}

status=0
train_until_complete Qwen3.5-4B-paper-explicit \
  scripts/run_vision_opd_paper_explicit_local_4b.sh "${FOUR_B_CKPT}" \
  "${LOG_DIR}/vision_opd_4b_paper_explicit.log" || status=1
ray stop --force >> "${LOG_DIR}/paper_explicit_pipeline.log" 2>&1 || true
merge_checkpoint "${FOUR_B_CKPT}" "${FOUR_B_MERGED}" || status=1

train_until_complete Qwen3.5-9B-paper-explicit \
  scripts/run_vision_opd_paper_explicit_local_9b.sh "${NINE_B_CKPT}" \
  "${LOG_DIR}/vision_opd_9b_paper_explicit.log" || status=1
ray stop --force >> "${LOG_DIR}/paper_explicit_pipeline.log" 2>&1 || true
merge_checkpoint "${NINE_B_CKPT}" "${NINE_B_MERGED}" || status=1

if (( status == 0 )); then
  touch "${COMPLETE_MARKER}"
  log "Paper-explicit 4B/9B training and merge completed."
else
  log "Paper-explicit training pipeline ended with failures."
  exit 1
fi
