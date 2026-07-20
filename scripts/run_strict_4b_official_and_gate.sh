#!/usr/bin/env bash
set -euo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
MODEL_PATH="${PROJECT_ROOT}/merged_models/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4"
MODEL_NAME="Vision-OPD-Qwen3.5-4B-released-b96-r8-official"
LOG="${PROJECT_ROOT}/logs/strict_released_4b_evaluation.log"

cd "${PROJECT_ROOT}"
python scripts/validate_merged_model.py "${MODEL_PATH}"
ray stop --force >> "${LOG}" 2>&1 || true
MODEL_PATH="${MODEL_PATH}" SERVED_MODEL_NAME="${MODEL_NAME}" MODEL_TP=1 MODEL_DP=6 \
  bash scripts/evaluate_official_single_model.sh >> "${LOG}" 2>&1
python scripts/verify_official_opd_alignment.py --project-root "${PROJECT_ROOT}" \
  --model "${MODEL_NAME}" --backbone 4b \
  | tee "${PROJECT_ROOT}/logs/strict_released_4b_alignment.log"
touch "${PROJECT_ROOT}/outputs/vision_opd_strict_released_4b_complete"
python scripts/summarize_official_evaluation.py --project-root "${PROJECT_ROOT}"
python scripts/summarize_goal_reproduction.py --project-root "${PROJECT_ROOT}" \
  --output "${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md"
