#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
PIPELINE_SESSION="auto_reproduction_pipeline_20260714"
PIPELINE_LOG="${PROJECT_ROOT}/logs/auto_reproduction_pipeline_20260714.log"
BENCHMARK_SESSION="prepare_benchmarks_local_hr_20260714"
BENCHMARK_LOG="${PROJECT_ROOT}/logs/prepare_benchmarks_local_hr_20260714.log"
COMPLETE_MARKER="${PROJECT_ROOT}/outputs/vision_opd_reproduction_complete"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs"
cd "${PROJECT_ROOT}"

while [[ ! -f "${COMPLETE_MARKER}" ]]; do
  if ! screen -ls 2>/dev/null | rg -q "[.]${PIPELINE_SESSION}[[:space:]]"; then
    printf '[%s] Restarting missing reproduction pipeline.\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    screen -L -Logfile "${PIPELINE_LOG}" -dmS "${PIPELINE_SESSION}" \
      bash scripts/auto_reproduction_pipeline.sh
  fi
  benchmark_ready=true
  for filename in vstar.json zoombench.json hr_bench_4k.json hr_bench_8k.json MME_RealWorld.json MME_RealWorld_CN.json; do
    if [[ ! -f "${PROJECT_ROOT}/benchmark/prepared/${filename}" ]]; then
      benchmark_ready=false
      break
    fi
  done
  if [[ "${benchmark_ready}" == false ]] && \
     ! screen -ls 2>/dev/null | rg -q "[.]${BENCHMARK_SESSION}[[:space:]]"; then
    printf '[%s] Restarting missing benchmark preparation.\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    screen -L -Logfile "${BENCHMARK_LOG}" -dmS "${BENCHMARK_SESSION}" \
      bash scripts/prepare_benchmarks.sh
  fi
  sleep 60
done

printf '[%s] Completion marker detected; watchdog exiting.\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
