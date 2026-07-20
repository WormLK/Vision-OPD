#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
VTC_ROOT="/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab"
BASELINE_PATH="/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B"
BASELINE_NAME="Qwen3.5-4B-baseline-official"
OPD_PATH="${PROJECT_ROOT}/merged_models/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4"
OPD_NAME="Vision-OPD-Qwen3.5-4B-released-b96-r8-official"
BASELINE_MARKER="${PROJECT_ROOT}/outputs/${BASELINE_NAME}_official_evaluation_complete"
OPD_MARKER="${PROJECT_ROOT}/outputs/${OPD_NAME}_official_evaluation_complete"
GOAL_MARKER="${PROJECT_ROOT}/outputs/vision_opd_4b_step65_official_complete"
LOG="${PROJECT_ROOT}/logs/selected_4b_goal_pipeline.log"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"
exec > >(tee -a "${LOG}") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

evaluate_until_complete() {
  local model_path="$1" model_name="$2" marker="$3"
  [[ -f "${marker}" ]] && return 0
  for attempt in $(seq 1 20); do
    log "Official evaluation attempt ${attempt} for ${model_name}."
    /data00/users/wanglikun/anaconda3/envs/vision-opd/bin/ray stop --force || true
    if MODEL_PATH="${model_path}" SERVED_MODEL_NAME="${model_name}" MODEL_TP=1 MODEL_DP=6 \
      bash scripts/evaluate_official_single_model.sh; then
      [[ -f "${marker}" ]] && return 0
    fi
    sleep $((attempt < 5 ? attempt * 30 : 300))
  done
  log "Exhausted retries for ${model_name}."
  return 1
}

while screen -ls 2>/dev/null | rg -q '[.]selected_4b_official_eval[[:space:]]'; do
  log "Waiting for the currently running baseline/trained official evaluation."
  sleep 120
done

evaluate_until_complete "${BASELINE_PATH}" "${BASELINE_NAME}" "${BASELINE_MARKER}"
evaluate_until_complete "${OPD_PATH}" "${OPD_NAME}" "${OPD_MARKER}"

python scripts/validate_official_model_outputs.py --project-root "${PROJECT_ROOT}" --model "${BASELINE_NAME}"
python scripts/validate_official_model_outputs.py --project-root "${PROJECT_ROOT}" --model "${OPD_NAME}"
touch "${GOAL_MARKER}"

python scripts/summarize_4b_vtc_reproduction.py \
  --project-root "${PROJECT_ROOT}" --vtc-root "${VTC_ROOT}" \
  --output "${PROJECT_ROOT}/docs/vision_opd_4b_vtc_reproduction.md"
python scripts/summarize_4b_vtc_reproduction.py \
  --project-root "${PROJECT_ROOT}" --vtc-root "${VTC_ROOT}" \
  --output "${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md"

log "Starting VTC-Bench code-driven and interface-driven tracks."
MODEL_PATH="${OPD_PATH}" MODEL_NAME="${OPD_NAME}" OPD_MARKER="${GOAL_MARKER}" \
  bash "${VTC_ROOT}/scripts/run_vision_opd_4b_vtc_bench.sh"

python scripts/summarize_4b_vtc_reproduction.py \
  --project-root "${PROJECT_ROOT}" --vtc-root "${VTC_ROOT}" \
  --output "${PROJECT_ROOT}/docs/vision_opd_4b_vtc_reproduction.md"
python scripts/summarize_4b_vtc_reproduction.py \
  --project-root "${PROJECT_ROOT}" --vtc-root "${VTC_ROOT}" \
  --output "${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md"
python scripts/audit_4b_goal_completion.py \
  --project-root "${PROJECT_ROOT}" --vtc-root "${VTC_ROOT}" \
  | tee "${PROJECT_ROOT}/logs/vision_opd_4b_goal_completion_audit.log"
touch "${PROJECT_ROOT}/outputs/vision_opd_4b_goal_audit_complete"
log "Selected step-65 Vision-OPD-4B official and VTC-Bench evaluation completed."
