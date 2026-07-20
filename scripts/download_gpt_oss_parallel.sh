#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-openai/gpt-oss-120b}"
TARGET_DIR="${TARGET_DIR:-/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/OpenAI/gpt-oss-120b}"
WORKERS="${WORKERS:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-4194304}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
API_URL="https://huggingface.co/api/models/${MODEL_ID}?blobs=true"
HF_URL="https://huggingface.co/${MODEL_ID}/resolve/main"
MIRROR_URL="https://hf-mirror.com/${MODEL_ID}/resolve/main"
MS_URL="https://www.modelscope.cn/models/openai-mirror/gpt-oss-120b/resolve/master"
STATE_DIR="${TARGET_DIR}/.parallel_download"
MODEL_INFO="${STATE_DIR}/model_info.json"
SHARD_MANIFEST="${STATE_DIR}/shards.tsv"
TASKS="${STATE_DIR}/tasks.tsv"
CURL_CONFIG="${STATE_DIR}/curl_parallel.conf"
PREPARE_ONLY="${PREPARE_ONLY:-false}"
ASSEMBLE_ONLY="${ASSEMBLE_ONLY:-false}"
ASSEMBLE_AVAILABLE_ONLY="${ASSEMBLE_AVAILABLE_ONLY:-false}"
DOWNLOAD_SOURCE="${DOWNLOAD_SOURCE:-rotate}"

mkdir -p "${TARGET_DIR}" "${STATE_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

file_size() {
  stat -c '%s' "$1" 2>/dev/null || printf '0\n'
}

if [[ ! -s "${MODEL_INFO}" ]]; then
  log "Fetching official Hugging Face model metadata."
  curl -L --fail --retry 20 --connect-timeout 30 --max-time 600 \
    -o "${MODEL_INFO}.tmp" "${API_URL}"
  mv "${MODEL_INFO}.tmp" "${MODEL_INFO}"
fi

jq -r '.siblings[] | select(.rfilename | test("^model-[0-9]+-of-[0-9]+[.]safetensors$")) | [.rfilename, (.size|tostring), .lfs.sha256] | @tsv' \
  "${MODEL_INFO}" | sort > "${SHARD_MANIFEST}"

if [[ "$(wc -l < "${SHARD_MANIFEST}")" -ne 15 ]]; then
  log "ERROR: expected 15 official safetensors shards in model metadata."
  exit 1
fi

: > "${TASKS}"
: > "${CURL_CONFIG}"
while IFS=$'\t' read -r name size sha256; do
  destination="${TARGET_DIR}/${name}"
  verified_marker="${destination}.verified.sha256"
  if [[ "$(file_size "${destination}")" -eq "${size}" ]] && \
    [[ "$(cat "${verified_marker}" 2>/dev/null || true)" == "${sha256}" ]]; then
    log "Already verified: ${name}"
    continue
  fi
  if [[ "$(file_size "${destination}")" -eq "${size}" ]] && \
    [[ "$(sha256sum "${destination}" | awk '{print $1}')" == "${sha256}" ]]; then
    printf '%s\n' "${sha256}" > "${verified_marker}"
    log "Already verified: ${name}"
    continue
  fi

  chunk_dir="${STATE_DIR}/${name}.chunks"
  mkdir -p "${chunk_dir}"
  start=0
  chunk_index=0
  while (( start < size )); do
    end=$((start + CHUNK_SIZE - 1))
    (( end >= size )) && end=$((size - 1))
    expected=$((end - start + 1))
    chunk_path="${chunk_dir}/$(printf '%012d' "${start}").part"
    if [[ "$(file_size "${chunk_path}")" -ne "${expected}" ]]; then
      case "${DOWNLOAD_SOURCE}" in
        mirror) url="${MIRROR_URL}/${name}" ;;
        huggingface) url="${HF_URL}/${name}" ;;
        modelscope) url="${MS_URL}/${name}" ;;
        rotate)
          case $((chunk_index % 3)) in
            0) url="${MIRROR_URL}/${name}" ;;
            1) url="${HF_URL}/${name}" ;;
            2) url="${MS_URL}/${name}" ;;
          esac
          ;;
        *)
          log "ERROR: unsupported DOWNLOAD_SOURCE=${DOWNLOAD_SOURCE}"
          exit 1
          ;;
      esac
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${url}" "${chunk_path}" "${start}" "${end}" "${expected}" >> "${TASKS}"
      printf 'url = "%s"\noutput = "%s"\nrange = "%s-%s"\nlocation\nfail\nsilent\nshow-error\nretry = 20\nretry-all-errors\nconnect-timeout = 60\nmax-time = 3600\nspeed-limit = 1024\nspeed-time = 300\nnext\n' \
        "${url}" "${chunk_path}" "${start}" "${end}" >> "${CURL_CONFIG}"
    fi
    start=$((end + 1))
    chunk_index=$((chunk_index + 1))
  done
done < "${SHARD_MANIFEST}"

download_chunk() {
  local url="$1" destination="$2" start="$3" end="$4" expected="$5"
  local attempt actual tmp
  tmp="${destination}.tmp.$$"
  for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    if curl -L --fail --silent --show-error --retry 0 \
      --connect-timeout 30 --max-time 600 --speed-limit 1024 --speed-time 90 \
      --range "${start}-${end}" -o "${tmp}" "${url}"; then
      actual="$(stat -c '%s' "${tmp}" 2>/dev/null || printf '0')"
      if [[ "${actual}" -eq "${expected}" ]]; then
        mv "${tmp}" "${destination}"
        return 0
      fi
    fi
    sleep $((attempt < 10 ? attempt * 2 : 20))
  done
  printf 'Failed chunk after %s attempts: %s bytes %s-%s\n' \
    "${MAX_ATTEMPTS}" "${url}" "${start}" "${end}" >&2
  return 1
}
export -f download_chunk
export MAX_ATTEMPTS

task_count="$(wc -l < "${TASKS}")"
if [[ "${PREPARE_ONLY}" == "true" ]]; then
  log "Prepared ${task_count} missing ranges in ${CURL_CONFIG}."
  log "Run curl -L --parallel --parallel-max ${WORKERS} --fail --retry 20 --retry-all-errors --connect-timeout 60 --max-time 3600 --speed-limit 1024 --speed-time 300 --config ${CURL_CONFIG}"
  exit 0
fi

if (( task_count > 0 )) && [[ "${ASSEMBLE_ONLY}" != "true" ]]; then
  log "Downloading ${task_count} missing ranges with ${WORKERS} workers."
  xargs -P "${WORKERS}" -n 5 bash -c 'download_chunk "$@"' _ < "${TASKS}"
fi

while IFS=$'\t' read -r name size sha256; do
  destination="${TARGET_DIR}/${name}"
  verified_marker="${destination}.verified.sha256"
  if [[ "$(file_size "${destination}")" -eq "${size}" ]] && \
    [[ "$(cat "${verified_marker}" 2>/dev/null || true)" == "${sha256}" ]]; then
    continue
  fi
  if [[ "$(file_size "${destination}")" -eq "${size}" ]] && \
    [[ "$(sha256sum "${destination}" | awk '{print $1}')" == "${sha256}" ]]; then
    printf '%s\n' "${sha256}" > "${verified_marker}"
    continue
  fi

  if [[ "${ASSEMBLE_AVAILABLE_ONLY}" == "true" ]]; then
    available_size="$(find "${STATE_DIR}/${name}.chunks" -maxdepth 1 -type f -name '*.part' -printf '%s\n' 2>/dev/null | awk '{total += $1} END {print total + 0}')"
    if [[ "${available_size}" -ne "${size}" ]]; then
      log "Skipping incomplete ${name}: ${available_size}/${size} bytes."
      continue
    fi
  fi

  log "Assembling and validating ${name}."
  assembled="${destination}.assembling"
  : > "${assembled}"
  find "${STATE_DIR}/${name}.chunks" -maxdepth 1 -type f -name '*.part' -print0 \
    | sort -z | xargs -0 cat >> "${assembled}"
  actual_size="$(file_size "${assembled}")"
  actual_sha256="$(sha256sum "${assembled}" | awk '{print $1}')"
  if [[ "${actual_size}" -ne "${size}" ]] || [[ "${actual_sha256}" != "${sha256}" ]]; then
    log "ERROR: validation failed for ${name}: size=${actual_size}/${size} sha256=${actual_sha256}/${sha256}"
    exit 1
  fi
  mv "${assembled}" "${destination}"
  printf '%s\n' "${sha256}" > "${verified_marker}"
  log "Verified ${name}."
done < "${SHARD_MANIFEST}"

if [[ "${ASSEMBLE_AVAILABLE_ONLY}" == "true" ]]; then
  log "All currently complete shard ranges were assembled and verified."
else
  log "All 15 GPT-OSS-120B weight shards are complete and SHA-256 verified."
fi
