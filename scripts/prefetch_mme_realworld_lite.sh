#!/usr/bin/env bash
set -euo pipefail

ROOT="/data00/users/wanglikun/ProjWormLK/Vision-OPD/benchmark/official_reproduction_20260717/source/eval/yifanzhang114_MME-RealWorld-lite-lmms-eval"
STATE_ROOT="${ROOT}/.parallel_download"
BASE_URL="https://hf-mirror.com/datasets/yifanzhang114/MME-RealWorld-lite-lmms-eval/resolve/main/data"
CURL_BIN="/data00/users/wanglikun/anaconda3/bin/curl"
CHUNK_SIZE="${CHUNK_SIZE:-4194304}"
WORKERS="${WORKERS:-32}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-100}"
MANIFEST="${STATE_ROOT}/manifest.tsv"
TASKS="${STATE_ROOT}/tasks.tsv"

mkdir -p "${ROOT}/data" "${STATE_ROOT}"

cat > "${MANIFEST}" <<'EOF'
train-00000-of-00004.parquet	498488644	0908f422107275374463a87cc7a195c77ae0030835996c3ab2d31e45cdc9d2de
train-00001-of-00004.parquet	176784125	c325a8d1c16a2ffb1b0d92649b8567dee3d69ff071583d59b05d43c4020edaf6
train-00002-of-00004.parquet	624400225	4ba6cbb22af7da0fb55ef51471344f5a579237be8c521b894ce78ae4730dfed5
train-00003-of-00004.parquet	581106081	d3a709d0a0a345c57eb05796f27651ea95834cb6a714dbad835d4dc7c75e37fb
EOF

file_size() {
  stat -c '%s' "$1" 2>/dev/null || printf '0\n'
}

: > "${TASKS}"
while IFS=$'\t' read -r name size sha256; do
  target="${ROOT}/data/${name}"
  if [[ "$(file_size "${target}")" -eq "${size}" ]] && \
    [[ "$(sha256sum "${target}" | awk '{print $1}')" == "${sha256}" ]]; then
    continue
  fi
  chunk_dir="${STATE_ROOT}/${name}.chunks"
  mkdir -p "${chunk_dir}"
  start=0
  while ((start < size)); do
    end=$((start + CHUNK_SIZE - 1))
    ((end >= size)) && end=$((size - 1))
    expected=$((end - start + 1))
    chunk="${chunk_dir}/$(printf '%012d' "${start}").part"
    if [[ "$(file_size "${chunk}")" -ne "${expected}" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${BASE_URL}/${name}" "${chunk}" "${start}" "${end}" "${expected}" >> "${TASKS}"
    fi
    start=$((end + 1))
  done
done < "${MANIFEST}"

sort -t $'\t' -k3,3n -k1,1 "${TASKS}" -o "${TASKS}"

download_chunk() {
  local url="$1" destination="$2" start="$3" end="$4" expected="$5"
  local attempt tmp actual
  tmp="${destination}.tmp.$$"
  for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    if "${CURL_BIN}" -L --fail --silent --show-error --retry 0 \
      --connect-timeout 30 --max-time 600 --speed-limit 1024 --speed-time 90 \
      --range "${start}-${end}" -o "${tmp}" "${url}"; then
      actual="$(file_size "${tmp}")"
      if [[ "${actual}" -eq "${expected}" ]]; then
        mv "${tmp}" "${destination}"
        return 0
      fi
    fi
    sleep $((attempt < 10 ? attempt * 2 : 20))
  done
  return 1
}
export -f download_chunk file_size
export CURL_BIN MAX_ATTEMPTS

if [[ -s "${TASKS}" ]]; then
  xargs -P "${WORKERS}" -n 5 bash -c 'download_chunk "$@"' _ < "${TASKS}"
fi

while IFS=$'\t' read -r name size sha256; do
  target="${ROOT}/data/${name}"
  if [[ "$(file_size "${target}")" -eq "${size}" ]] && \
    [[ "$(sha256sum "${target}" | awk '{print $1}')" == "${sha256}" ]]; then
    continue
  fi
  assembled="${target}.assembling"
  : > "${assembled}"
  find "${STATE_ROOT}/${name}.chunks" -maxdepth 1 -type f -name '*.part' -print0 \
    | sort -z | xargs -0 cat >> "${assembled}"
  [[ "$(file_size "${assembled}")" -eq "${size}" ]]
  [[ "$(sha256sum "${assembled}" | awk '{print $1}')" == "${sha256}" ]]
  mv "${assembled}" "${target}"
done < "${MANIFEST}"
