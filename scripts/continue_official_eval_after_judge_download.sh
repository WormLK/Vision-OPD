#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
JUDGE_ROOT="/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/OpenAI/gpt-oss-120b"
MANIFEST="${JUDGE_ROOT}/.parallel_download/shards.tsv"
DOWNLOAD_SESSION="download_gpt_oss_mirror32_20260718"
LOG_DIR="${PROJECT_ROOT}/logs"
READY_MARKER="${PROJECT_ROOT}/outputs/vision_opd_official_existing_checkpoints_complete"
PIPELINE_LOCK="${PROJECT_ROOT}/outputs/.official_post_4b_prerequisites.lock"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

# The main controller and the recovery watcher can reach this entry point at
# nearly the same time. Serialize them so only one judge/eval stack owns GPUs.
exec 9>"${PIPELINE_LOCK}"
flock 9
if [[ -f "${READY_MARKER}" ]]; then
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

screen_is_running() {
  screen -ls 2>/dev/null | rg -q "[.]$1[[:space:]]"
}

verify_judge_weights() {
  local name expected expected_sha path actual_sha
  [[ -s "${MANIFEST}" ]] || return 1
  [[ "$(wc -l < "${MANIFEST}")" -eq 15 ]] || return 1
  while IFS=$'\t' read -r name expected expected_sha; do
    path="${JUDGE_ROOT}/${name}"
    [[ -f "${path}" && "$(stat -c '%s' "${path}")" -eq "${expected}" ]] || return 1
    actual_sha="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual_sha}" == "${expected_sha}" ]] || return 1
  done < "${MANIFEST}"
}

while screen_is_running "${DOWNLOAD_SESSION}"; do
  downloaded="$(find "${JUDGE_ROOT}/.parallel_download" -type f -name '*.part' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')"
  log "Waiting for judge download: ${downloaded}/65248893184 range bytes."
  sleep 120
done

for attempt in $(seq 1 20); do
  if verify_judge_weights; then
    log "All 15 judge shards passed official SHA-256 validation."
    break
  fi
  log "Judge weights are incomplete after download attempt ${attempt}; resuming missing ranges."
  DOWNLOAD_SOURCE=mirror WORKERS=32 MAX_ATTEMPTS=100 CHUNK_SIZE=4194304 \
    bash scripts/download_gpt_oss_parallel.sh \
    >> "${LOG_DIR}/download_gpt_oss_watcher_retry.log" 2>&1 || true
done
verify_judge_weights

for attempt in $(seq 1 5); do
  log "Running official baseline judge/alignment attempt ${attempt}."
  if bash scripts/judge_official_baselines.sh \
    >> "${LOG_DIR}/official_baseline_judge_pipeline.log" 2>&1; then
    break
  fi
  sleep $((attempt * 30))
done
[[ -f outputs/vision_opd_official_baselines_aligned ]]

for attempt in $(seq 1 20); do
  log "Running existing-checkpoint official evaluation attempt ${attempt}."
  if bash scripts/run_official_eval_reproduction.sh \
    >> "${LOG_DIR}/official_existing_checkpoints_pipeline.log" 2>&1; then
    break
  fi
  sleep $((attempt < 5 ? attempt * 60 : 300))
done
[[ -f outputs/vision_opd_official_four_model_evaluation_complete ]]

python scripts/summarize_official_evaluation.py --project-root "${PROJECT_ROOT}"
python scripts/summarize_goal_reproduction.py --project-root "${PROJECT_ROOT}" \
  --output "${PROJECT_ROOT}/docs/vision_opd_goal_reproduction_report.md"
touch "${READY_MARKER}"
log "Baseline alignment and existing 4B/9B checkpoint evaluation are complete."
