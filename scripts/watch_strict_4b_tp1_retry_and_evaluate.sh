#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
EXP="Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4-rollout-tp1-retry"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/${EXP}"
ROLLOUT_ROOT="${PROJECT_ROOT}/rollouts/${EXP}"
TENSORBOARD_ROOT="${PROJECT_ROOT}/tensorboard_log/Vision-OPD/${EXP}"
MERGED_ROOT="${PROJECT_ROOT}/merged_models/${EXP}"
MODEL_NAME="Vision-OPD-Qwen3.5-4B-released-b96-r8-tp1-official"
BASELINE_PATH="/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B"
BASELINE_NAME="Qwen3.5-4B-baseline-official"
TRAIN_LOG="${PROJECT_ROOT}/logs/strict_4b_tp1_retry.screen.log"
PIPELINE_LOG="${PROJECT_ROOT}/logs/strict_4b_tp1_post_pipeline.log"
ALIGNMENT_LOG="${PROJECT_ROOT}/logs/strict_4b_tp1_alignment.log"
COMPLETE_MARKER="${PROJECT_ROOT}/outputs/vision_opd_strict_4b_tp1_complete"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs" "${PROJECT_ROOT}/merged_models"
cd "${PROJECT_ROOT}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

latest_step() {
  local marker="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
  [[ -f "${marker}" ]] && tr -cd '0-9' < "${marker}" || printf '0'
}

while (( $(latest_step) < 65 )); do
  log "Waiting for TP1 strict 4B checkpoint: $(latest_step)/65."
  sleep 120
done

while screen -ls 2>/dev/null | rg -q '[.]strict_4b_tp1_retry[[:space:]]'; do
  log "Checkpoint 65 is present; waiting for the training process to exit cleanly."
  sleep 30
done

log "Validating TP1 strict 4B runtime, trajectories, metrics, and checkpoint."
python scripts/validate_official_eval_source.py \
  --project-root "${PROJECT_ROOT}" \
  --official-root "${PROJECT_ROOT}/benchmark/official_reproduction_20260717"
python scripts/validate_goal_benchmark_contract.py --project-root "${PROJECT_ROOT}"
python scripts/validate_strict_runtime_config.py \
  --log "${TRAIN_LOG}" --backbone 4b --rollout-tp 1 --rollout-max-num-seqs 16
for step in $(seq 1 65); do
  python scripts/validate_strict_rollout.py "${ROLLOUT_ROOT}" --step "${step}"
done
python scripts/validate_strict_training_metrics.py \
  --tensorboard-dir "${TENSORBOARD_ROOT}" --expected-step 65
python scripts/validate_strict_checkpoint.py "${CHECKPOINT_ROOT}" --step 65

if [[ -d "${MERGED_ROOT}" ]] && [[ -n "$(find "${MERGED_ROOT}" -mindepth 1 -print -quit)" ]]; then
  if ! python scripts/validate_merged_model.py "${MERGED_ROOT}"; then
    mv "${MERGED_ROOT}" "${MERGED_ROOT}.incomplete.$(date -u +'%Y%m%dT%H%M%SZ')"
  fi
fi
if [[ ! -f "${MERGED_ROOT}/config.json" ]]; then
  log "Merging strict TP1 checkpoint 65."
  TARGET_DIR="${MERGED_ROOT}" bash scripts/merge_checkpoint.sh \
    "${CHECKPOINT_ROOT}/global_step_65"
fi
python scripts/validate_merged_model.py "${MERGED_ROOT}"

log "Completing pristine official 10-benchmark evaluation for ${BASELINE_NAME}."
ray stop --force || true
MODEL_PATH="${BASELINE_PATH}" SERVED_MODEL_NAME="${BASELINE_NAME}" MODEL_TP=1 MODEL_DP=6 \
  bash scripts/evaluate_official_single_model.sh
log "Running pristine official 10-benchmark evaluation for ${MODEL_NAME}."
MODEL_PATH="${MERGED_ROOT}" SERVED_MODEL_NAME="${MODEL_NAME}" MODEL_TP=1 MODEL_DP=6 \
  bash scripts/evaluate_official_single_model.sh
python scripts/verify_official_opd_alignment.py --project-root "${PROJECT_ROOT}" \
  --model "${MODEL_NAME}" --backbone 4b | tee "${ALIGNMENT_LOG}"

touch "${COMPLETE_MARKER}"
python scripts/summarize_official_evaluation.py --project-root "${PROJECT_ROOT}"
python scripts/summarize_goal_reproduction.py --project-root "${PROJECT_ROOT}" \
  --output "${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md"

log "Official 4B alignment gate passed; starting both VTC-Bench tracks."
MODEL_PATH="${MERGED_ROOT}" MODEL_NAME="${MODEL_NAME}" OPD_MARKER="${COMPLETE_MARKER}" \
  bash /data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/scripts/run_vision_opd_4b_vtc_bench.sh
python scripts/audit_4b_goal_completion.py \
  --project-root "${PROJECT_ROOT}" \
  --vtc-root /data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab \
  | tee "${PROJECT_ROOT}/logs/vision_opd_4b_goal_completion_audit.log"
touch "${PROJECT_ROOT}/outputs/vision_opd_4b_goal_audit_complete"
log "Strict TP1 4B official and VTC-Bench evaluation completed."
