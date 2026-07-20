#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path


BENCHMARKS = {
    "vstar": "vstar.json",
    "zoombench": "zoombench.json",
    "hrbench-4k": "hr_bench_4k.json",
    "hrbench-8k": "hr_bench_8k.json",
    "mme-realworld": "MME_RealWorld.json",
    "mme-realworld-cn": "MME_RealWorld_CN.json",
}

EXPECTED_COUNTS = {
    "vstar": 191,
    "zoombench": 845,
    "hrbench-4k": 800,
    "hrbench-8k": 800,
    "mme-realworld": 23609,
    "mme-realworld-cn": 5462,
}

PROFILES = {
    "lowmem": {
        "models": {
            "Vision-OPD-Qwen3.5-4B": "Vision-OPD-Qwen3.5-4B-full-repro-lowmem-20260714",
            "Vision-OPD-Qwen3.5-9B": "Vision-OPD-Qwen3.5-9B-local-lowmem-20260714",
        },
        "expected_step": 779,
        "report": "reproduction_results.md",
    },
    "paper-explicit": {
        "models": {
            "Vision-OPD-Qwen3.5-4B-paper-explicit": "Vision-OPD-Qwen3.5-4B-paper-explicit-local",
            "Vision-OPD-Qwen3.5-9B-paper-explicit": "Vision-OPD-Qwen3.5-9B-paper-explicit-local",
        },
        "expected_step": 780,
        "report": "reproduction_results_paper_explicit.md",
        "training_logs": (
            "vision_opd_4b_paper_explicit.log",
            "vision_opd_9b_paper_explicit.log",
        ),
        "required_log_snippets": (
            "dataset len: 6241",
            "Size of train dataloader: 780",
            "Total training steps: 780",
            "'max_prompt_length': 8192",
            "'max_response_length': 1024",
            "'filter_overlong_prompts': False",
            "'truncation': 'error'",
            "'alpha': 0.5",
            "'distillation_topk': 100",
            "'teacher_always_on': True",
            "'teacher_model_source': 'legacy'",
            "'teacher_regularization': 'ema'",
            "'teacher_update_rate': 0.05",
            "'include_environment_feedback': False",
            "'total_epochs': 1",
        ),
        "chat_template": "chat_templates/perception_chat_template_qwen35.jinja",
    },
}
ERROR_PREFIXES = ("[API_ERROR]", "[FUTURE_ERROR]", "[JUDGE_API_ERROR]")


def load_json(path, errors):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"invalid JSON {path}: {exc}")
        return None


def sample_uid(row, benchmark):
    for key in ("sample_uid", "uid", "index", "question_id", "id"):
        value = row.get(key)
        if value is not None and str(value) != "":
            return f"{benchmark}:{key}:{value}"
    stable = {"benchmark": benchmark, "images": row.get("images") or [], "query": row.get("query", "")}
    raw = json.dumps(stable, ensure_ascii=False, sort_keys=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def verify_manifests(root, errors, decode_images=False):
    prepared = root / "benchmark" / "prepared"
    counts = {}
    expected_uids = {}
    all_images = set()
    for benchmark, filename in BENCHMARKS.items():
        path = prepared / filename
        if not path.is_file():
            errors.append(f"missing benchmark manifest: {path}")
            continue
        rows = load_json(path, errors)
        if not isinstance(rows, list) or not rows:
            errors.append(f"empty or invalid benchmark manifest: {path}")
            continue
        counts[benchmark] = len(rows)
        expected_uids[benchmark] = set()
        if len(rows) != EXPECTED_COUNTS[benchmark]:
            errors.append(
                f"unexpected benchmark count {path}: {len(rows)}/{EXPECTED_COUNTS[benchmark]}"
            )
        seen_uids = set()
        for index, row in enumerate(rows):
            images = row.get("images") or []
            if not images:
                errors.append(f"missing image list: {path}:{index}")
                continue
            for image in images:
                image_path = Path(image)
                all_images.add(image_path)
                if not image_path.is_file() or image_path.stat().st_size == 0:
                    errors.append(f"missing or empty image: {image_path}")
            for image in row.get("crop_images") or []:
                image_path = Path(image)
                all_images.add(image_path)
                if not image_path.is_file() or image_path.stat().st_size == 0:
                    errors.append(f"missing or empty crop image: {image_path}")
            if not str(row.get("query", "")).strip():
                errors.append(f"empty query: {path}:{index}")
            if not str(row.get("response", "")).strip():
                errors.append(f"empty response: {path}:{index}")
            if benchmark in {"vstar", "hrbench-4k", "hrbench-8k", "mme-realworld", "mme-realworld-cn"}:
                if not str(row.get("category", "")).strip() or row.get("category") == "unknown":
                    errors.append(f"missing category: {path}:{index}")
            if benchmark in {"mme-realworld", "mme-realworld-cn"}:
                if not str(row.get("l2_category", "")).strip() or row.get("l2_category") == "unknown":
                    errors.append(f"missing l2_category: {path}:{index}")
            if benchmark in {"hrbench-4k", "hrbench-8k"}:
                if row.get("cycle_category") not in {0, 1, 2, 3}:
                    errors.append(f"missing or invalid cycle_category: {path}:{index}")
            uid = sample_uid(row, benchmark)
            if uid in seen_uids:
                errors.append(f"duplicate sample UID: {path}:{index}:{uid}")
            seen_uids.add(uid)
            expected_uids[benchmark].add(uid)

    if decode_images:
        from PIL import Image

        for image_path in sorted(all_images):
            if not image_path.is_file() or image_path.stat().st_size == 0:
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                errors.append(f"invalid image {image_path}: {exc}")
    return counts, expected_uids


def verify_model_evaluation(root, model_name, counts, expected_uids, errors):
    model_tag = f"{model_name}_seed42"
    for benchmark, expected_count in counts.items():
        answer_path = root / "benchmark" / "model_answer" / benchmark / f"{model_tag}_answer.jsonl"
        if not answer_path.is_file():
            errors.append(f"missing inference output: {answer_path}")
            continue
        records = []
        try:
            with answer_path.open("r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        except Exception as exc:
            errors.append(f"invalid inference JSONL {answer_path}: {exc}")
            continue
        sample_uids = [record.get("sample_uid") for record in records]
        if len(records) != expected_count:
            errors.append(f"incomplete inference output {answer_path}: {len(records)}/{expected_count}")
        if any(not uid for uid in sample_uids) or len(set(sample_uids)) != len(sample_uids):
            errors.append(f"missing or duplicate sample_uid in {answer_path}")
        elif set(sample_uids) != expected_uids.get(benchmark, set()):
            errors.append(f"inference sample_uid set does not match manifest: {answer_path}")
        failed = [
            record for record in records
            if not str(record.get("model_answer", "")).strip()
            or str(record.get("model_answer", "")).strip().startswith(ERROR_PREFIXES)
        ]
        if failed:
            errors.append(f"failed inference records in {answer_path}: {len(failed)}")

        judge_path = root / "benchmark" / "judge" / benchmark / f"{model_tag}_answer.jsonl"
        if not judge_path.is_file():
            errors.append(f"missing judge output: {judge_path}")
            continue
        judged = load_json(judge_path, errors)
        if not isinstance(judged, list):
            continue
        if len(judged) != expected_count:
            errors.append(f"incomplete judge output {judge_path}: {len(judged)}/{expected_count}")
        judged_uids = [record.get("sample_uid") for record in judged]
        if any(not uid for uid in judged_uids) or len(set(judged_uids)) != len(judged_uids):
            errors.append(f"missing or duplicate sample_uid in {judge_path}")
        elif set(judged_uids) != set(sample_uids):
            errors.append(f"judge sample_uid set does not match inference output: {judge_path}")
        judge_failures = [
            record for record in judged
            if str(record.get("judge", "")).strip().lower() not in {"yes", "no"}
        ]
        if judge_failures:
            errors.append(f"invalid or failed judge records in {judge_path}: {len(judge_failures)}")
        missing_sources = [record for record in judged if not str(record.get("judge_source", "")).strip()]
        if missing_sources:
            errors.append(f"judge records missing judge_source in {judge_path}: {len(missing_sources)}")

        result_path = root / "benchmark" / "results" / f"{model_tag}_{benchmark}.txt"
        if not result_path.is_file() or result_path.stat().st_size == 0:
            errors.append(f"missing or empty score file: {result_path}")
        else:
            result_text = result_path.read_text(encoding="utf-8", errors="replace")
            percentages = re.findall(r"(\d+(?:\.\d+)?)%", result_text)
            if not percentages:
                errors.append(f"score percentage not found: {result_path}")
            elif isinstance(judged, list) and judged:
                expected_score = 100.0 * sum(
                    str(record.get("judge", "")).strip().lower() == "yes" for record in judged
                ) / len(judged)
                reported_score = float(percentages[-1])
                if abs(reported_score - expected_score) > 0.011:
                    errors.append(
                        f"score mismatch {result_path}: reported={reported_score:.2f}% "
                        f"computed={expected_score:.2f}%"
                    )


def verify_training_and_merge(root, models, expected_step, errors):
    for model_name, checkpoint_name in models.items():
        checkpoint_root = root / "checkpoints" / checkpoint_name
        marker = checkpoint_root / "latest_checkpointed_iteration.txt"
        if not marker.is_file():
            errors.append(f"missing checkpoint marker: {marker}")
            continue
        digits = re.sub(r"\D", "", marker.read_text(encoding="utf-8", errors="replace"))
        step = int(digits) if digits else -1
        if step < expected_step:
            errors.append(f"incomplete checkpoint {checkpoint_root}: {step}/{expected_step}")
        if not (checkpoint_root / f"global_step_{step}" / "actor").is_dir():
            errors.append(f"missing final actor checkpoint: {checkpoint_root}/global_step_{step}/actor")

        merged = root / "merged_models" / model_name
        verify_merged_model(merged, errors)


def verify_merged_model(merged, errors):
    from safetensors import safe_open
    from transformers import AutoConfig, AutoProcessor

    config_path = merged / "config.json"
    if not config_path.is_file():
        errors.append(f"missing merged config: {config_path}")
        return
    processor_configs = (merged / "processor_config.json", merged / "preprocessor_config.json")
    if not any(path.is_file() for path in processor_configs):
        errors.append(f"missing merged processor config: {merged}")

    index_path = merged / "model.safetensors.index.json"
    expected_keys = None
    if index_path.is_file():
        index = load_json(index_path, errors)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            errors.append(f"empty merged weight map: {index_path}")
            weight_files = []
        else:
            expected_keys = set(weight_map)
            weight_files = [merged / name for name in sorted(set(weight_map.values()))]
    else:
        weight_files = sorted(merged.glob("model*.safetensors"))
    if not weight_files:
        errors.append(f"missing merged weights: {merged}/model*.safetensors")
        return

    actual_keys = set()
    for weight_file in weight_files:
        if not weight_file.is_file() or weight_file.stat().st_size == 0:
            errors.append(f"missing or empty merged shard: {weight_file}")
            continue
        try:
            with safe_open(weight_file, framework="pt", device="cpu") as handle:
                actual_keys.update(handle.keys())
        except Exception as exc:
            errors.append(f"invalid merged safetensors shard {weight_file}: {exc}")
    if not actual_keys:
        errors.append(f"merged model contains no tensors: {merged}")
    if expected_keys is not None and actual_keys != expected_keys:
        errors.append(
            f"merged tensor/index mismatch {merged}: actual={len(actual_keys)} expected={len(expected_keys)}"
        )

    try:
        config = AutoConfig.from_pretrained(merged, trust_remote_code=True)
        if config.model_type != "qwen3_5":
            errors.append(f"unexpected merged model type {merged}: {config.model_type}")
    except Exception as exc:
        errors.append(f"failed to load merged config {merged}: {exc}")
    try:
        AutoProcessor.from_pretrained(merged, trust_remote_code=True)
    except Exception as exc:
        errors.append(f"failed to load merged processor {merged}: {exc}")


def verify_training_configuration(root, profile, errors):
    log_names = profile.get("training_logs") or ()
    snippets = profile.get("required_log_snippets") or ()
    expected_step = profile["expected_step"]
    for log_name in log_names:
        log_path = root / "logs" / log_name
        if not log_path.is_file():
            errors.append(f"missing training log: {log_path}")
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for snippet in snippets:
            if snippet not in text:
                errors.append(f"training configuration evidence missing from {log_path}: {snippet}")

        steps = {int(value) for value in re.findall(r"training/global_step:(\d+)", text)}
        if len(steps) < expected_step or max(steps, default=0) < expected_step:
            errors.append(
                f"incomplete training metrics in {log_path}: "
                f"unique_steps={len(steps)}, max_step={max(steps, default=0)}/{expected_step}"
            )
        prompt_clip_ratios = [
            float(value) for value in re.findall(r"prompt_length/clip_ratio:([0-9.]+)", text)
        ]
        if any(value > 0 for value in prompt_clip_ratios):
            errors.append(f"prompt truncation observed in paper-explicit log: {log_path}")
        response_maxima = [
            float(value) for value in re.findall(r"response_length/max:([0-9.]+)", text)
        ]
        if any(value > 1024 for value in response_maxima):
            errors.append(f"response exceeded configured 1024-token cap: {log_path}")

    template_name = profile.get("chat_template")
    if template_name:
        template_path = root / template_name
        if not template_path.is_file():
            errors.append(f"missing non-thinking chat template: {template_path}")
        else:
            template = template_path.read_text(encoding="utf-8", errors="replace")
            if r"<think>\n\n</think>\n\n" not in template:
                errors.append(f"chat template does not inject an empty thinking block: {template_path}")


def verify_training_dataset(root, errors):
    import pyarrow.parquet as pq

    path = root / "data" / "train.parquet"
    if not path.is_file():
        errors.append(f"missing training parquet: {path}")
        return
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != 6241:
        errors.append(f"unexpected training row count {path}: {parquet.metadata.num_rows}/6241")
    required_columns = {"prompt", "images", "bbox_images", "reward_model", "extra_info"}
    missing_columns = required_columns - set(parquet.schema_arrow.names)
    if missing_columns:
        errors.append(f"training parquet missing columns {path}: {sorted(missing_columns)}")
        return

    table = pq.read_table(path, columns=["images", "bbox_images"])
    for column_name in ("images", "bbox_images"):
        for index, image_list in enumerate(table.column(column_name).to_pylist()):
            if not image_list:
                errors.append(f"empty {column_name} at {path}:{index}")
                continue
            for image in image_list:
                image_path = Path(image.get("path", "")) if isinstance(image, dict) else Path("")
                if not image_path.is_file() or image_path.stat().st_size == 0:
                    errors.append(f"missing or empty training image: {path}:{index}:{image_path}")


def verify_source_models(root, errors):
    from safetensors import safe_open

    sources = {
        "Qwen3.5-4B": root.parent / "MODEL_ZOO" / "Qwen" / "Qwen3.5-4B",
        "Qwen3.5-9B": root.parent / "MODEL_ZOO" / "Qwen" / "Qwen3.5-9b",
    }
    for model_name, model_dir in sources.items():
        config_path = model_dir / "config.json"
        index_path = model_dir / "model.safetensors.index.json"
        if not config_path.is_file():
            errors.append(f"missing source model config: {config_path}")
            continue
        config = load_json(config_path, errors)
        if isinstance(config, dict) and config.get("model_type") != "qwen3_5":
            errors.append(f"unexpected source model type in {config_path}: {config.get('model_type')}")
        if not index_path.is_file():
            errors.append(f"missing source weight index: {index_path}")
            continue
        index = load_json(index_path, errors)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            errors.append(f"empty source weight map: {index_path}")
            continue
        shard_names = sorted(set(weight_map.values()))
        for shard_name in shard_names:
            shard = model_dir / shard_name
            if not shard.is_file() or shard.stat().st_size == 0:
                errors.append(f"missing or empty source shard: {shard}")
                continue
            try:
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    if not list(handle.keys()):
                        errors.append(f"source shard contains no tensors: {shard}")
            except Exception as exc:
                errors.append(f"invalid source safetensors shard {shard}: {exc}")
        for required in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
            path = model_dir / required
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"missing source model artifact: {path}")
        print(f"Validated {model_name} source model: {len(weight_map)} tensors across {len(shard_names)} shards.")


def main():
    parser = argparse.ArgumentParser(description="Verify Vision-OPD reproduction artifacts")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="lowmem")
    parser.add_argument("--model-name")
    parser.add_argument("--merged-model-dir")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    models = profile["models"]
    if args.merged_model_dir:
        errors = []
        verify_merged_model(Path(args.merged_model_dir).resolve(), errors)
        if errors:
            for error in errors:
                print(f"- {error}")
            raise SystemExit(1)
        print(f"Verified merged model: {Path(args.merged_model_dir).resolve()}")
        return
    if not args.model_name and not args.full:
        parser.error("set --model-name for one evaluation or --full for the final audit")
    if args.model_name and args.model_name not in models:
        parser.error(
            f"unknown model for profile {args.profile}: {args.model_name}; "
            f"choose one of {', '.join(models)}"
        )

    root = Path(args.project_root).resolve()
    errors = []
    counts, expected_uids = verify_manifests(root, errors, decode_images=args.full)
    selected_models = list(models) if args.full else [args.model_name]
    for model_name in selected_models:
        verify_model_evaluation(root, model_name, counts, expected_uids, errors)

    if args.full:
        verify_source_models(root, errors)
        verify_training_dataset(root, errors)
        verify_training_and_merge(root, models, profile["expected_step"], errors)
        verify_training_configuration(root, profile, errors)
        report = root / "docs" / profile["report"]
        if not report.is_file():
            errors.append(f"missing final report: {report}")
        elif "pending" in report.read_text(encoding="utf-8", errors="replace").lower():
            errors.append(f"final report still contains pending results: {report}")

    if errors:
        print("Reproduction verification failed:")
        for error in errors[:100]:
            print(f"- {error}")
        if len(errors) > 100:
            print(f"- ... {len(errors) - 100} additional errors")
        raise SystemExit(1)

    scope = f"full {args.profile} reproduction" if args.full else f"{args.model_name} evaluation"
    print(f"Verified {scope}: six benchmarks are complete and error-free.")


if __name__ == "__main__":
    main()
