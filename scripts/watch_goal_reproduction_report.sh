#!/usr/bin/env bash
set -uo pipefail

source /data00/users/wanglikun/anaconda3/etc/profile.d/conda.sh
conda activate vision-opd

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
OUTPUT="${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md"
MARKER="${PROJECT_ROOT}/outputs/vision_opd_paper_explicit_reproduction_complete"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"

cd "${PROJECT_ROOT}"
while [[ ! -f "${MARKER}" ]]; do
  python scripts/summarize_goal_reproduction.py \
    --project-root "${PROJECT_ROOT}" --output "${OUTPUT}" || true
  sleep "${INTERVAL_SECONDS}"
done

python scripts/summarize_goal_reproduction.py \
  --project-root "${PROJECT_ROOT}" --output "${OUTPUT}"
