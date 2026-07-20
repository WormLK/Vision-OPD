#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
PYTHON="/data00/users/wanglikun/anaconda3/envs/vision-opd/bin/python"
EXPERIMENT="${EXPERIMENT:-Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4}"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/${EXPERIMENT}"
ROLLOUT_ROOT="${PROJECT_ROOT}/rollouts/${EXPERIMENT}"
TENSORBOARD_ROOT="${PROJECT_ROOT}/tensorboard_log/Vision-OPD/${EXPERIMENT}"
TRACKER="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-30}"
HEALTH_PREFIX="${HEALTH_PREFIX:-strict_4b_training_health}"
HEALTH_LABEL="${HEALTH_LABEL:-Qwen3.5-4B}"

cd "${PROJECT_ROOT}"

latest_step() {
  [[ -f "${TRACKER}" ]] && tr -cd '0-9' < "${TRACKER}" || printf '0'
}

validate_step() {
  local step="$1"
  printf '[%s] validating strict 4B step %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${step}"
  "${PYTHON}" scripts/validate_strict_checkpoint.py "${CHECKPOINT_ROOT}" --step "${step}"
  "${PYTHON}" scripts/validate_strict_training_metrics.py \
    --tensorboard-dir "${TENSORBOARD_ROOT}" --expected-step "${step}"
  "${PYTHON}" scripts/validate_strict_rollout.py "${ROLLOUT_ROOT}" --step "${step}"
  "${PYTHON}" scripts/summarize_strict_training_health.py \
    --tensorboard-dir "${TENSORBOARD_ROOT}" --expected-step "${step}" --label "${HEALTH_LABEL}" \
    --output-md "${PROJECT_ROOT}/docs/${HEALTH_PREFIX}.md" \
    --output-json "${PROJECT_ROOT}/outputs/${HEALTH_PREFIX}.json" \
    --output-plot "${PROJECT_ROOT}/docs/${HEALTH_PREFIX}.png"
  "${PYTHON}" scripts/summarize_goal_reproduction.py \
    --project-root "${PROJECT_ROOT}" \
    --output "${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md"
  printf '[%s] strict 4B step %s semantic validation complete\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${step}"
}

last_verified="$(latest_step)"
if (( last_verified > 0 )); then
  validate_step "${last_verified}"
fi

while (( last_verified < 65 )); do
  sleep "${INTERVAL_SECONDS}"
  step="$(latest_step)"
  if (( step > last_verified )); then
    validate_step "${step}"
    last_verified="${step}"
  fi
done
