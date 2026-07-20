#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Validate the local GPT-OSS-120B official judge")
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    root = args.model_dir.resolve()

    required = (
        "config.json",
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    for filename in required:
        path = root / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing or empty judge artifact: {path}")

    config_json = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config_json.get("model_type") != "gpt_oss":
        raise SystemExit(f"unexpected judge model_type: {config_json.get('model_type')}")
    if config_json.get("architectures") != ["GptOssForCausalLM"]:
        raise SystemExit(f"unexpected judge architecture: {config_json.get('architectures')}")
    if (config_json.get("quantization_config") or {}).get("quant_method") != "mxfp4":
        raise SystemExit("GPT-OSS judge is not the expected MXFP4 checkpoint")
    if int(config_json.get("max_position_embeddings", 0)) < 65536:
        raise SystemExit("GPT-OSS judge context is shorter than the configured 65536 tokens")

    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    shard_names = sorted(set(weight_map.values()))
    if len(weight_map) != 687 or len(shard_names) != 15:
        raise SystemExit(
            f"unexpected judge index: tensors={len(weight_map)}, shards={len(shard_names)}"
        )

    manifest_path = root / ".parallel_download" / "shards.tsv"
    if not manifest_path.is_file():
        raise SystemExit(f"missing verified shard manifest: {manifest_path}")
    manifest = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            name, size, digest = line.rstrip("\n").split("\t")
            manifest[name] = (int(size), digest)
    if set(manifest) != set(shard_names):
        raise SystemExit("verified shard manifest names differ from the safetensors index")

    actual_keys = set()
    keys_by_shard = {}
    for shard_name in shard_names:
        shard = root / shard_name
        expected_size, expected_digest = manifest[shard_name]
        if not shard.is_file() or shard.stat().st_size != expected_size:
            raise SystemExit(f"missing or size-mismatched judge shard: {shard}")
        marker = root / f"{shard_name}.verified.sha256"
        if not marker.is_file() or marker.read_text(encoding="ascii").strip() != expected_digest:
            raise SystemExit(f"missing or mismatched SHA-256 verification marker: {marker}")
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                shard_keys = set(handle.keys())
                if not shard_keys:
                    raise ValueError("no tensors")
                duplicates = actual_keys & shard_keys
                if duplicates:
                    raise ValueError(f"duplicate keys: {sorted(duplicates)[:10]}")
                for key in shard_keys:
                    shape = handle.get_slice(key).get_shape()
                    if not shape or any(int(size) <= 0 for size in shape):
                        raise ValueError(f"invalid tensor shape for {key}: {shape}")
                keys_by_shard[shard_name] = shard_keys
                actual_keys.update(shard_keys)
        except Exception as exc:
            raise SystemExit(f"invalid judge shard {shard}: {exc}") from exc

    indexed_keys = set(weight_map)
    wrong_shards = [
        key for key, shard_name in weight_map.items() if key not in keys_by_shard.get(shard_name, set())
    ]
    if actual_keys != indexed_keys or wrong_shards:
        raise SystemExit(
            "judge index/header mismatch: "
            f"missing={sorted(indexed_keys - actual_keys)[:10]} "
            f"extra={sorted(actual_keys - indexed_keys)[:10]} wrong_shard={wrong_shards[:10]}"
        )

    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Return Yes."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if config.model_type != "gpt_oss" or not rendered or "Return Yes." not in rendered:
        raise SystemExit("GPT-OSS config/tokenizer/chat template failed offline loading")

    from vllm.reasoning import ReasoningParserManager

    if "openai_gptoss" not in ReasoningParserManager.list_registered():
        raise SystemExit("vLLM does not register the openai_gptoss reasoning parser")
    parser_class = ReasoningParserManager.get_reasoning_parser("openai_gptoss")
    if parser_class.__name__ != "GptOssReasoningParser":
        raise SystemExit(f"unexpected GPT-OSS reasoning parser: {parser_class}")

    print(
        f"PASS GPT-OSS judge: tensors={len(actual_keys)} shards={len(shard_names)} "
        f"model_type={config.model_type} parser={parser_class.__name__}"
    )


if __name__ == "__main__":
    main()
