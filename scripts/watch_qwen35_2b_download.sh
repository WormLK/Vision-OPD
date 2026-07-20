#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="Qwen/Qwen3.5-2B"
MODEL_DIR="/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-2B"
MODELSCOPE="/data00/users/wanglikun/anaconda3/envs/deepeyes/bin/modelscope"
ACTIVE_SESSION="download_qwen35_2b"
READY_MARKER="${MODEL_DIR}/.download_validated"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-100}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

session_running() {
  screen -ls 2>/dev/null | grep -q "[.]${ACTIVE_SESSION}[[:space:]]"
}

validate_model() {
  /data00/users/wanglikun/anaconda3/envs/vision-opd/bin/python - "${MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

root = Path(sys.argv[1]).resolve()
index_path = root / "model.safetensors.index.json"
if not index_path.is_file():
    raise SystemExit("missing model.safetensors.index.json")
index = json.loads(index_path.read_text(encoding="utf-8"))
weight_map = index.get("weight_map") or {}
shards = sorted(set(weight_map.values()))
if len(weight_map) != 632 or shards != ["model.safetensors-00001-of-00001.safetensors"]:
    raise SystemExit(f"unexpected index: tensors={len(weight_map)} shards={shards}")
actual_keys = set()
for shard_name in shards:
    shard_path = root / shard_name
    if not shard_path.is_file() or shard_path.stat().st_size == 0:
        raise SystemExit(f"missing or empty shard: {shard_path}")
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        actual_keys.update(handle.keys())
if actual_keys != set(weight_map):
    raise SystemExit(
        f"index/header mismatch: missing={len(set(weight_map) - actual_keys)} "
        f"extra={len(actual_keys - set(weight_map))}"
    )
config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=True)
if config.model_type != "qwen3_5" or tokenizer is None or processor is None:
    raise SystemExit("offline config/tokenizer/processor validation failed")
print(
    f"PASS Qwen3.5-2B: tensors={len(actual_keys)} shards={len(shards)} "
    f"bytes={sum((root / name).stat().st_size for name in shards)}"
)
PY
}

mkdir -p "${MODEL_DIR}"
while session_running; do
  log "Waiting for active Qwen3.5-2B download session."
  sleep 60
done

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  if validate_model; then
    touch "${READY_MARKER}"
    log "Qwen3.5-2B download and offline validation complete."
    exit 0
  fi
  log "Qwen3.5-2B is incomplete; resume attempt ${attempt}/${MAX_ATTEMPTS}."
  "${MODELSCOPE}" download --model "${MODEL_ID}" --local_dir "${MODEL_DIR}" --max-workers 8 || true
  sleep $((attempt < 10 ? attempt * 15 : 180))
done

log "Qwen3.5-2B download exhausted retries."
exit 1
