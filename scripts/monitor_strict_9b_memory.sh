#!/usr/bin/env bash
set -u

PROJECT_ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD"
START_MARKER="${PROJECT_ROOT}/outputs/vision_opd_strict_released_4b_complete"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/Vision-OPD-Qwen3.5-9B-released-b96-r8-gradaccum-sp8"
TRACKER="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"

latest_step() {
  [[ -f "${TRACKER}" ]] && tr -cd '0-9' < "${TRACKER}" || printf '0'
}

sum_rss_kib() {
  local pattern="$1"
  ps -eo rss=,cmd= | awk -v pattern="${pattern}" '$2 ~ pattern { total += $1 } END { print total + 0 }'
}

while [[ ! -f "${START_MARKER}" ]]; do
  sleep "${INTERVAL_SECONDS}"
done

printf '%s\n' \
  'timestamp,checkpoint,mem_available_kib,actor_rss_kib,task_runner_rss_kib,agent_loop_rss_kib,vllm_rss_kib,object_store_bytes,gpu_process_mib'

while (( $(latest_step) < 65 )); do
  timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  checkpoint="$(latest_step)"
  mem_available_kib="$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)"
  actor_rss_kib="$(sum_rss_kib 'ray::WorkerDict')"
  task_runner_rss_kib="$(sum_rss_kib 'ray::TaskRunner')"
  agent_loop_rss_kib="$(sum_rss_kib 'ray::AgentLoopWorker')"
  vllm_rss_kib="$(sum_rss_kib 'ray::vLLMHttpServer|VLLM::EngineCore')"
  object_store_bytes="$(
    curl -fsS http://127.0.0.1:8265/api/cluster_status 2>/dev/null \
      | jq -r '.data.clusterStatus.loadMetricsReport.usage.objectStoreMemory[0] // 0' 2>/dev/null
  )"
  [[ "${object_store_bytes}" =~ ^[0-9]+([.][0-9]+)?$ ]] || object_store_bytes=0
  gpu_process_mib="$(
    nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk '{ total += $1 } END { print total + 0 }'
  )"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${timestamp}" "${checkpoint}" "${mem_available_kib}" "${actor_rss_kib}" \
    "${task_runner_rss_kib}" "${agent_loop_rss_kib}" "${vllm_rss_kib}" \
    "${object_store_bytes}" "${gpu_process_mib}"
  sleep "${INTERVAL_SECONDS}"
done
