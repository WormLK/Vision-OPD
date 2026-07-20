#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    root = args.model_dir.resolve()

    required = (
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for filename in required:
        path = root / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing required merged-model file: {path}")
    if not any((root / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        raise SystemExit(f"missing processor config in {root}")

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") not in {"qwen3_5", "qwen3_5_moe"}:
        raise SystemExit(f"unexpected model_type: {config.get('model_type')}")
    architectures = config.get("architectures") or []
    if not any(name in {"Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"} for name in architectures):
        raise SystemExit(f"unexpected multimodal architecture: {architectures}")
    if not isinstance(config.get("vision_config"), dict) or not config["vision_config"]:
        raise SystemExit(f"missing nonempty vision_config in {root / 'config.json'}")

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") or {}
        if not weight_map:
            raise SystemExit(f"empty weight_map: {index_path}")
        shard_names = sorted(set(weight_map.values()))
    else:
        weight_map = None
        shard_names = [path.name for path in sorted(root.glob("model*.safetensors"))]
    if not shard_names:
        raise SystemExit(f"no merged safetensors weights in {root}")

    keys_by_shard = {}
    all_actual_keys = set()
    for shard_name in shard_names:
        shard = root / shard_name
        if not shard.is_file() or shard.stat().st_size == 0:
            raise SystemExit(f"missing or empty merged shard: {shard}")
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                if metadata.get("format") != "pt":
                    raise ValueError(f"unexpected safetensors metadata: {metadata}")
                shard_keys = set(handle.keys())
                if not shard_keys:
                    raise ValueError("shard has no tensors")
                duplicate_keys = all_actual_keys & shard_keys
                if duplicate_keys:
                    raise ValueError(f"duplicate tensor keys across shards: {sorted(duplicate_keys)[:10]}")
                for key in shard_keys:
                    shape = handle.get_slice(key).get_shape()
                    if not shape or any(int(size) <= 0 for size in shape):
                        raise ValueError(f"invalid tensor shape for {key}: {shape}")
                keys_by_shard[shard_name] = shard_keys
                all_actual_keys.update(shard_keys)
        except Exception as exc:
            raise SystemExit(f"invalid merged safetensors shard {shard}: {exc}") from exc

    if weight_map is not None:
        indexed_keys = set(weight_map)
        missing_keys = indexed_keys - all_actual_keys
        extra_keys = all_actual_keys - indexed_keys
        wrong_shards = [
            name for name, shard_name in weight_map.items() if name not in keys_by_shard.get(shard_name, set())
        ]
        if missing_keys or extra_keys or wrong_shards:
            raise SystemExit(
                "safetensors index mismatch: "
                f"missing={sorted(missing_keys)[:10]} extra={sorted(extra_keys)[:10]} "
                f"wrong_shard={wrong_shards[:10]}"
            )
        tensor_count = len(weight_map)
    else:
        tensor_count = len(all_actual_keys)

    print(
        f"validated merged model: path={root} shards={len(shard_names)} "
        f"tensors={tensor_count} model_type={config['model_type']}"
    )


if __name__ == "__main__":
    main()
