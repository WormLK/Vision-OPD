#!/usr/bin/env python3
"""Generate the legacy combined 4B/9B reproduction history report.

The active selected-step-65 4B goal uses summarize_4b_vtc_reproduction.py.
This generator is retained to preserve the earlier 4B/9B experiment audit.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


BENCHMARKS = {
    "Vstar": ("vstar", "vstar.json", 191),
    "ZoomBench": ("zoombench", "zoombench.json", 845),
    "HR-Bench-4K": ("hrbench-4k", "hr_bench_4k.json", 800),
    "HR-Bench-8K": ("hrbench-8k", "hr_bench_8k.json", 800),
    "MME-RealWorld-EN": ("mme-realworld", "MME_RealWorld.json", 23609),
    "MME-RealWorld-CN": ("mme-realworld-cn", "MME_RealWorld_CN.json", 5462),
}

ALL_BENCHMARKS = {
    **BENCHMARKS,
    "MME-RealWorld-Lite": ("mme-realworld-lite", "MME_RealWorld_Lite.json", 1919),
    "MMStar": ("mmstar", "mmstar.json", 1500),
    "POPE-Test": ("pope", "POPE.json", 9000),
    "POPE-Adversarial": ("pope_adv", "POPE_adv.json", 3000),
    "POPE-Popular": ("pope_pop", "POPE_pop.json", 3000),
    "POPE-Random": ("pope_random", "POPE_random.json", 3000),
    "CV-Bench": ("cv-bench", "cv_bench.json", 2638),
    "MMVP": ("mmvp", "mmvp.json", 300),
    "VisualProbe": ("visualprobe", "visualprobe.json", 515),
}

EVALUATION_BENCHMARKS = {
    **BENCHMARKS,
    "MMStar": ALL_BENCHMARKS["MMStar"],
    "POPE-Test": ALL_BENCHMARKS["POPE-Test"],
    "CV-Bench": ALL_BENCHMARKS["CV-Bench"],
    "MMVP": ALL_BENCHMARKS["MMVP"],
}

EVAL_SOURCE_SHA256 = {
    "run_eval.sh": "46275049ec31d794c591238d49cd7683351f5af65c975a05d960cef1eb5554ea",
    "infer.py": "bb379999932658907196cdc98d22c60d63e3308cb5a867317481c4a85af70374",
    "judge_qwenlm.py": "abbe11dacf7fae19728ca16407a02c91d04a9bc8ea72edd3b4a91b6224f4b670",
    "cal_acc.py": "695dbddc3e63a1b9f8971c0d414d963a5da94776863d58589feaa4a1c6b0f025",
}

ACTIVE_FOUR_START_SHA256 = {
    "verl/experimental/agent_loop/agent_loop.py": "f3b9d6438a2c821c53533193ab9d5d97859ad6b5a31b1d2b7191e747ad841bad",
    "verl/workers/actor/dp_actor.py": "5cb2aaf6dcfd03ac41e426dae3fc50ea626d26daef66652bd55c0052a751d987",
    "verl/trainer/ppo/ray_trainer.py": "e34a00625f5b54f517aba6901a3559e53af8d76fbc4224b7c435c690699fac34",
    "verl/workers/fsdp_workers.py": "e2de73b6d162d02f7a8a01d53a4baed08e8275512ffd2fa25c7522192e1225dc",
    "scripts/run_vision_opd.sh": "fe59bcb61de0bdd03bce3acde295c5ad7eed11b2e77e06d381b2717ff65a7172",
    "scripts/run_vision_opd_released_b96_r8_gradaccum_4b.sh": "0ef15286e1fe2834656b1b13d04b8de0ca397dbb9bc68ad2b07d487af58f84bf",
    "scripts/run_strict_released_reproduction_pipeline.sh": "da141c6dbdba80862cfbcadc43cb8c81eb443532eab38d24e482eae724c01a45",
    "scripts/watch_post_strict_4b_prerequisites.sh": "9bd35f2f3b7002074f4ba8a2a4aa46e2a4ba2100d7e5181c4d53ba7f6a7f6c5d",
}

MODELS = {
    "Qwen3.5-4B": {
        "artifact": "Vision-OPD-Qwen3.5-4B-paper-explicit",
        "checkpoint": "Vision-OPD-Qwen3.5-4B-paper-explicit-local",
        "log": "vision_opd_4b_paper_explicit.log",
    },
    "Qwen3.5-9B": {
        "artifact": "Vision-OPD-Qwen3.5-9B-paper-explicit",
        "checkpoint": "Vision-OPD-Qwen3.5-9B-paper-explicit-local",
        "log": "vision_opd_9b_paper_explicit.log",
    },
}

PAPER_SCORES = {
    "Qwen3.5-4B baseline": [84.29, 47.69, 84.38, 80.13, 63.86, 63.70],
    "Vision-OPD 4B": [92.15, 59.76, 84.50, 80.38, 74.88, 70.76],
    "Qwen3.5-9B baseline": [82.72, 52.07, 85.75, 80.63, 71.40, 67.67],
    "Vision-OPD 9B": [94.76, 65.80, 88.13, 85.50, 73.40, 70.46],
}


def marker_step(path):
    marker = path / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        return 0
    digits = re.sub(r"\D", "", marker.read_text(encoding="utf-8", errors="replace"))
    return int(digits) if digits else 0


def log_metrics(path):
    metrics = {}
    if not path.is_file():
        return metrics
    patterns = {
        "step": re.compile(r"training/global_step:(\d+)"),
        "loss": re.compile(r"actor/vopd_loss:([0-9.eE+-]+)"),
        "grad": re.compile(r"actor/grad_norm:([0-9.eE+-]+)"),
        "response_max": re.compile(r"response_length/max:([0-9.]+)"),
        "response_clip": re.compile(r"response_length/clip_ratio:([0-9.]+)"),
        "prompt_clip": re.compile(r"prompt_length/clip_ratio:([0-9.]+)"),
        "memory": re.compile(r"perf/max_memory_allocated_gb:([0-9.]+)"),
        "cpu_memory": re.compile(r"perf/cpu_memory_used_gb:([0-9.]+)"),
    }
    by_step = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            step_match = patterns["step"].search(line)
            if not step_match:
                continue
            row = {}
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    row[key] = float(match.group(1))
            by_step[int(step_match.group(1))] = row
    if not by_step:
        return metrics
    rows = [by_step[step] for step in sorted(by_step)]
    window = min(50, len(rows))
    metrics["logged_steps"] = len(rows)
    metrics["max_step"] = max(by_step)
    metrics["response_max"] = max(row.get("response_max", 0) for row in rows)
    metrics["response_cap_steps"] = sum(row.get("response_clip", 0) > 0 for row in rows)
    metrics["prompt_cap_steps"] = sum(row.get("prompt_clip", 0) > 0 for row in rows)
    metrics["memory"] = max(row.get("memory", 0) for row in rows)
    metrics["cpu_memory"] = max(row.get("cpu_memory", 0) for row in rows)
    for key in ("loss", "grad"):
        first = [row[key] for row in rows[:window] if key in row]
        last = [row[key] for row in rows[-window:] if key in row]
        if first and last:
            metrics[f"{key}_first"] = sum(first) / len(first)
            metrics[f"{key}_last"] = sum(last) / len(last)
    return metrics


def tensorboard_metrics(path):
    if not path.is_dir():
        return {}
    event_files = list(path.glob("events.out.tfevents.*"))
    if not event_files:
        return {}
    # An experiment name can be reused after a failed run. Reading the whole
    # directory merges stale events and makes an old max step look current.
    event_path = max(event_files, key=lambda item: item.stat().st_mtime_ns)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(str(event_path))
        accumulator.Reload()
    except Exception:
        return {}

    scalar_tags = set(accumulator.Tags().get("scalars", []))
    tag_map = {
        "loss": "actor/vopd_loss",
        "grad": "actor/grad_norm",
        "response_max": "response_length/max",
        "response_clip": "response_length/clip_ratio",
        "prompt_clip": "prompt_length/clip_ratio",
        "memory": "perf/max_memory_allocated_gb",
        "cpu_memory": "perf/cpu_memory_used_gb",
    }
    by_step = {}
    for key, tag in tag_map.items():
        if tag not in scalar_tags:
            continue
        for event in accumulator.Scalars(tag):
            by_step.setdefault(event.step, {})[key] = float(event.value)
    if not by_step:
        return {}

    rows = [by_step[step] for step in sorted(by_step)]
    window = min(50, len(rows))
    metrics = {
        "logged_steps": len(rows),
        "max_step": max(by_step),
        "response_max": max(row.get("response_max", 0) for row in rows),
        "response_cap_steps": sum(row.get("response_clip", 0) > 0 for row in rows),
        "prompt_cap_steps": sum(row.get("prompt_clip", 0) > 0 for row in rows),
        "memory": max(row.get("memory", 0) for row in rows),
        "cpu_memory": max(row.get("cpu_memory", 0) for row in rows),
    }
    for key in ("loss", "grad"):
        first = [row[key] for row in rows[:window] if key in row]
        last = [row[key] for row in rows[-window:] if key in row]
        if first and last:
            metrics[f"{key}_first"] = sum(first) / len(first)
            metrics[f"{key}_last"] = sum(last) / len(last)
    return metrics


def merged_state(path):
    config = (path / "config.json").is_file()
    weights = any(path.glob("model*.safetensors"))
    processor = (path / "processor_config.json").is_file() or (path / "preprocessor_config.json").is_file()
    return "ready" if config and weights and processor else "pending"


def source_model_state(path):
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return "missing index"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map") or {}
    shards = set(weight_map.values())
    complete = all((path / shard).is_file() and (path / shard).stat().st_size > 0 for shard in shards)
    return f"{len(weight_map)} tensors / {len(shards)} shards / {'complete' if complete else 'incomplete'}"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_score(root, artifact, slug):
    path = root / "benchmark" / "results" / f"{artifact}_seed42_{slug}.txt"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if slug in {"pope", "pope_adv", "pope_pop", "pope_random"}:
        percentages = re.findall(r"accuracy=(\d+(?:\.\d+)?)%", text)
    else:
        percentages = re.findall(r"(\d+(?:\.\d+)?)%", text)
    return float(percentages[-1]) if percentages else None


def official_alignment_table(root):
    path = root / "docs" / "official_evaluation_reproduction.md"
    if not path.is_file():
        return "Strict official alignment report is pending."
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"## Table 1 Alignment\n\n(.*?)\n\n## (?:All 10|Baseline Deviation)", text, re.DOTALL)
    return match.group(1).strip() if match else "Strict official alignment table is pending."


def official_model_progress(root, model):
    official = root / "benchmark" / "official_reproduction_20260717"
    inference = judge = scores = 0
    for _, (slug, _, _) in EVALUATION_BENCHMARKS.items():
        answer = official / "model_answer" / slug / f"{model}_seed42_answer.jsonl"
        if answer.is_file():
            with answer.open(encoding="utf-8", errors="replace") as handle:
                inference += sum(bool(line.strip()) for line in handle)
        judged = official / "judge" / slug / f"{model}_seed42_answer.jsonl"
        if judged.is_file():
            try:
                judge += len(json.loads(judged.read_text(encoding="utf-8")))
            except Exception:
                pass
        if (official / "results" / f"{model}_seed42_{slug}.txt").is_file():
            scores += 1
    return inference, judge, scores


def evaluation_progress(root, artifact):
    inference = judge = scores = 0
    for _, (slug, _, _) in BENCHMARKS.items():
        answer_path = root / "benchmark" / "model_answer" / slug / f"{artifact}_seed42_answer.jsonl"
        if answer_path.is_file():
            with answer_path.open("r", encoding="utf-8", errors="replace") as handle:
                inference += sum(bool(line.strip()) for line in handle)
        judge_path = root / "benchmark" / "judge" / slug / f"{artifact}_seed42_answer.jsonl"
        if judge_path.is_file():
            try:
                judge += len(json.loads(judge_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        if result_score(root, artifact, slug) is not None:
            scores += 1
    return inference, judge, scores


def environment_table():
    rows = [f"| Python | {platform.python_version()} |"]
    for package in ("torch", "transformers", "vllm", "ray", "verl"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        rows.append(f"| {package} | {version} |")
    return "\n".join(rows)


def strict_memory_monitor_table(path):
    columns = (
        "mem_available_kib",
        "actor_rss_kib",
        "task_runner_rss_kib",
        "agent_loop_rss_kib",
        "vllm_rss_kib",
        "object_store_bytes",
    )
    rows = []
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    parsed = {key: float(row[key]) for key in columns}
                    parsed["checkpoint"] = int(row["checkpoint"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append(parsed)
    if not rows:
        return "| pending | pending | pending | pending | pending | pending | pending | pending |"

    gib = 1024**2
    object_gib = 1024**3
    return (
        f"| {len(rows)} | {min(row['checkpoint'] for row in rows)}-"
        f"{max(row['checkpoint'] for row in rows)} | "
        f"{min(row['mem_available_kib'] for row in rows) / gib:.2f} | "
        f"{max(row['actor_rss_kib'] for row in rows) / gib:.2f} | "
        f"{max(row['task_runner_rss_kib'] for row in rows) / gib:.2f} | "
        f"{max(row['agent_loop_rss_kib'] for row in rows) / gib:.2f} | "
        f"{max(row['vllm_rss_kib'] for row in rows) / gib:.2f} | "
        f"{max(row['object_store_bytes'] for row in rows) / object_gib:.2f} |"
    )


def strict_rollout_audit(path, checkpoint):
    if checkpoint <= 0:
        return "strict rollout JSONL 尚未生成。"

    errors = []
    audited_steps = []
    for step in range(1, checkpoint + 1):
        rollout_path = path / f"{step}.jsonl"
        if not rollout_path.is_file():
            errors.append(f"step {step} missing")
            continue
        counts = Counter()
        rows = 0
        forbidden = 0
        try:
            with rollout_path.open(encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    rows += 1
                    if any(
                        token in line
                        for token in ('"pixel_values"', '"deferred_images"', '"image_grid_thw"', '"bytes":')
                    ):
                        forbidden += 1
                    counts[json.loads(line)["input"]] += 1
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"step {step} invalid ({exc})")
            continue
        multiplicities = Counter(counts.values())
        if rows != 768 or len(counts) != 96 or multiplicities != Counter({8: 96}) or forbidden:
            errors.append(
                f"step {step}: rows={rows}, unique={len(counts)}, "
                f"multiplicities={dict(multiplicities)}, forbidden={forbidden}"
            )
            continue
        audited_steps.append(step)

    if errors:
        return "strict rollout 审计失败：" + "; ".join(errors)
    return (
        f"当前从零 strict run 的 step 1-{checkpoint} rollout JSONL 全部通过审计："
        "每步 768 行，由 96 个唯一 prompt 各生成 8 条轨迹，且不含 `pixel_values`、"
        "图片 bytes 或 deferred payload。这同时实证了 96x8 轨迹语义和 lazy image storage。"
    )


def fmt(value, digits=4):
    return f"{value:.{digits}f}" if value is not None else "pending"


def pct(value):
    return f"{value:.2f}%" if value is not None else "pending"


def main():
    parser = argparse.ArgumentParser(description="Generate the complete Vision-OPD goal retrospective")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    generated = datetime.now(timezone.utc).isoformat()

    training_complete = (root / "outputs" / "vision_opd_paper_explicit_training_complete").is_file()
    reproduction_complete = (root / "outputs" / "vision_opd_paper_explicit_reproduction_complete").is_file()
    four_b_stage_complete = (root / "outputs" / "vision_opd_paper_explicit_4b_evaluation_complete").is_file()
    official_existing_complete = (
        root / "outputs" / "vision_opd_official_existing_checkpoints_complete"
    ).is_file()
    strict_reproduction_complete = (
        root / "outputs" / "vision_opd_strict_released_reproduction_complete"
    ).is_file()
    overall_status = "complete" if strict_reproduction_complete else "active"

    training_rows = []
    optimization_rows = []
    evaluation_rows = []
    local_scores = {name: [] for name in MODELS}
    for model_name, info in MODELS.items():
        checkpoint_root = root / "checkpoints" / info["checkpoint"]
        metrics = log_metrics(root / "logs" / info["log"])
        checkpoint = marker_step(checkpoint_root)
        merged = merged_state(root / "merged_models" / info["artifact"])
        memory_cell = f"{metrics['memory']:.2f} GB" if "memory" in metrics else "pending"
        training_rows.append(
            f"| {model_name} | {checkpoint}/780 | {metrics.get('max_step', 0)}/780 | {merged} | "
            f"{int(metrics.get('response_max', 0)) or 'pending'} | "
            f"{metrics.get('response_cap_steps', 0)} | {metrics.get('prompt_cap_steps', 0)} | "
            f"{memory_cell} |"
        )
        optimization_rows.append(
            f"| {model_name} | {fmt(metrics.get('loss_first'), 6)} | {fmt(metrics.get('loss_last'), 6)} | "
            f"{fmt(metrics.get('grad_first'))} | {fmt(metrics.get('grad_last'))} |"
        )
        inference, judge, score_count = evaluation_progress(root, info["artifact"])
        evaluation_rows.append(
            f"| {model_name} | {inference}/31707 | {judge}/31707 | {score_count}/6 |"
        )
        for _, (slug, _, _) in BENCHMARKS.items():
            local_scores[model_name].append(result_score(root, info["artifact"], slug))

    benchmark_rows = []
    for name, (_, filename, expected) in ALL_BENCHMARKS.items():
        official_path = root / "benchmark" / "official_reproduction_20260717" / "source" / "eval" / filename
        prepared_path = root / "benchmark" / "prepared" / filename
        path = official_path if official_path.is_file() else prepared_path
        count = "missing"
        if path.is_file():
            count = len(json.loads(path.read_text(encoding="utf-8")))
        expected_cell = expected if expected is not None else "source manifest"
        ready = count == expected if expected is not None else isinstance(count, int) and count > 0
        benchmark_rows.append(
            f"| {name} | {count} | {expected_cell} | {'ready' if ready else 'incomplete'} |"
        )

    score_rows = []
    paper_names = list(BENCHMARKS)
    for index, benchmark in enumerate(paper_names):
        four = local_scores["Qwen3.5-4B"][index]
        nine = local_scores["Qwen3.5-9B"][index]
        score_rows.append(
            f"| {benchmark} | {pct(four)} | {PAPER_SCORES['Vision-OPD 4B'][index]:.2f}% | "
            f"{pct(nine)} | {PAPER_SCORES['Vision-OPD 9B'][index]:.2f}% |"
        )

    four_values = [value for value in local_scores["Qwen3.5-4B"] if value is not None]
    nine_values = [value for value in local_scores["Qwen3.5-9B"] if value is not None]
    four_macro = sum(four_values) / 6 if len(four_values) == 6 else None
    nine_macro = sum(nine_values) / 6 if len(nine_values) == 6 else None
    score_rows.append(
        f"| Macro average | {pct(four_macro)} | 77.07% | {pct(nine_macro)} | 79.68% |"
    )
    four_gap = (
        f"4B 本地 macro 为 {four_macro:.2f}%，相对论文 OPD 4B 的 77.07% 低 "
        f"{77.07 - four_macro:.2f} 个百分点。"
        if four_macro is not None
        else "4B 本地 macro 尚未完成。"
    )
    remaining_status = (
        "4B 阶段已通过单模型 verifier；paper-explicit 9B 按用户要求暂停在可恢复的 "
        "checkpoint 350，下一阶段将从该 checkpoint 恢复至 780，再完成同样六项评测。"
        if four_b_stage_complete and not reproduction_complete
        else "当前仍需完成的项目会随后台流水线自动推进。"
    )
    strict_alignment_table = official_alignment_table(root)
    strict_base4 = official_model_progress(root, "Qwen3.5-4B-baseline-official")
    strict_base9 = official_model_progress(root, "Qwen3.5-9B-baseline-official")
    strict_four_checkpoint = marker_step(
        root
        / "checkpoints"
        / "Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4-rollout-tp1-retry"
    )
    strict_nine_checkpoint = marker_step(
        root / "checkpoints" / "Vision-OPD-Qwen3.5-9B-released-b96-r8-gradaccum-sp8"
    )
    strict_four_eval = official_model_progress(
        root, "Vision-OPD-Qwen3.5-4B-released-b96-r8-tp1-official"
    )
    strict_nine_eval = official_model_progress(
        root, "Vision-OPD-Qwen3.5-9B-released-b96-r8-official"
    )
    strict_training_rows = []
    for label, checkpoint, log_name, experiment_name in (
        (
            "4B strict",
            strict_four_checkpoint,
            "strict_4b_tp1_retry.screen.log",
            "Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4-rollout-tp1-retry",
        ),
        (
            "9B strict",
            strict_nine_checkpoint,
            "strict_released_9b_training.log",
            "Vision-OPD-Qwen3.5-9B-released-b96-r8-gradaccum-sp8",
        ),
    ):
        metrics = log_metrics(root / "logs" / log_name)
        tensorboard = tensorboard_metrics(root / "tensorboard_log" / "Vision-OPD" / experiment_name)
        # Reusing an experiment name leaves prior-run TensorBoard events in place.
        # A live run can be at most one optimizer step ahead of its checkpoint.
        tensorboard_is_current = tensorboard.get("max_step", 0) <= checkpoint + 1
        if tensorboard_is_current and tensorboard.get("max_step", 0) > metrics.get("max_step", 0):
            metrics = tensorboard
        strict_training_rows.append(
            f"| {label} | {checkpoint}/65 | {metrics.get('max_step', 0)}/65 | "
            f"{fmt(metrics.get('loss_last'), 6)} | {fmt(metrics.get('grad_last'))} | "
            f"{fmt(metrics.get('memory'), 2)} GiB | {fmt(metrics.get('cpu_memory'), 2)} GiB |"
        )
    strict_memory_row = strict_memory_monitor_table(
        root / "logs" / "strict_4b_tp1_memory_monitor.csv"
    )
    strict_nine_memory_row = strict_memory_monitor_table(root / "logs" / "strict_9b_memory_monitor.csv")
    strict_rollout_summary = strict_rollout_audit(
        root
        / "rollouts"
        / "Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4-rollout-tp1-retry",
        strict_four_checkpoint,
    )

    eval_source = root / "benchmark" / "official_reproduction_20260717" / "source" / "eval"
    eval_hash_rows = []
    for filename, expected_hash in EVAL_SOURCE_SHA256.items():
        path = eval_source / filename
        actual_hash = sha256(path) if path.is_file() else None
        eval_hash_rows.append(
            f"| `{filename}` | `{actual_hash or 'missing'}` | "
            f"{'exact' if actual_hash == expected_hash else 'mismatch'} |"
        )
    runtime_hash_rows = []
    for relative_path, start_hash in ACTIVE_FOUR_START_SHA256.items():
        path = root / relative_path
        current_hash = sha256(path) if path.is_file() else None
        current_note = "current exact" if current_hash == start_hash else "current file changed after launch"
        if relative_path == "scripts/run_strict_released_reproduction_pipeline.sh":
            provenance_label = "active main controller startup"
        elif relative_path == "scripts/watch_post_strict_4b_prerequisites.sh":
            provenance_label = "active post-4B watcher startup"
        else:
            provenance_label = "active 4B attempt 1 startup"
        runtime_hash_rows.append(
            f"| `{relative_path}` | `{start_hash}` | {provenance_label}; {current_note} |"
        )
    for relative_path, label in (
        ("scripts/run_vision_opd_released_b96_r8_gradaccum_4b.sh", "current retry wrapper"),
        ("scripts/run_vision_opd_released_b96_r8_gradaccum_4b_tp1_retry.sh", "active TP1 launch wrapper"),
        ("scripts/watch_strict_4b_tp1_retry_and_evaluate.sh", "active TP1 post-training controller"),
        ("scripts/run_vision_opd_released_b96_r8_gradaccum_9b.sh", "pending 9B launch wrapper"),
        ("scripts/continue_official_eval_after_judge_download.sh", "pending post-4B prerequisite source"),
    ):
        path = root / relative_path
        runtime_hash_rows.append(
            f"| `{relative_path}` | `{sha256(path) if path.is_file() else 'missing'}` | {label} |"
        )

    report = f"""# Vision-OPD 4B/9B Goal 工作流与结果复盘

生成时间：{generated}

## 1. Goal 与当前结论

本次 goal 的目标是依据 Vision-OPD 论文 Experimental Settings，使用本地
Qwen3.5-4B 和 Qwen3.5-9B、完整 Vision-OPD-6K 数据完成训练，并在 Vstar、
ZoomBench、HR-Bench 4K/8K、MME-RealWorld EN/CN 上完成推理、判分、验证与
结果分析。

当前状态：**{overall_status}**。历史 `paper-explicit` 训练完成标记：**{training_complete}**；
历史 `paper-explicit` 训练、评测完成标记：**{reproduction_complete}**；已有 checkpoint 官方评测
完成标记：**{official_existing_complete}**；严格 released 复现完成标记：
**{strict_reproduction_complete}**。
历史 `paper-explicit` 4B 六项评测阶段标记：**{four_b_stage_complete}**。

历史 `paper-explicit` 标记不再代表整个 goal 完成。最终完成还要求本地 4B/9B
baseline、训练后模型均通过 pristine 官方评测，且 released `batch=96/rollout=8`
重训与复评完成。阶段报告中的 `pending` 不代表失败，只表示流水线仍在运行。

## 2. 论文设置与本地执行配置

论文明确给出的设置：Vision-OPD-6K、1 epoch、JSD beta/alpha 0.5、Top-K 100、
EMA teacher regularization、non-thinking、最大生成长度 1024。本次正式流程还
使用 8192 prompt 上限并禁止静默过滤超长样本。

论文正文没有明确规定 batch 和每 prompt rollout 数；官方发布脚本默认
`batch=96`、`rollout=8`。已完成的历史 checkpoint 使用 `batch=8/rollout=1`，
只能作为本地消融。严格重训将保留 96x8 global update，并用 dynamic micro-batch、
gradient accumulation、sequence parallel 和 offload 适配 8x NVIDIA L40S 46GB。

本地 `train.parquet` 有 6,241 条。上游 PPO trainer 固定 `drop_last=True`，因此
released batch 96 定义的一轮是 65 个完整 batch、共 6,240 prompts；shuffle 后的
尾部 1 条按官方 dataloader 行为丢弃。补做 batch 1 会额外产生一次非官方
optimizer/EMA update，故本次严格复现保持日志确认的 `Total training steps: 65`。

官方发布值与本地 strict wrapper 的逐项对照：

| 参数 | 官方发布 | strict 4B | strict 9B | 语义说明 |
| --- | ---: | ---: | ---: | --- |
| Train batch | 96 prompts | 96 | 96 | 不变 |
| Rollout `n` | 8 | 8 | 8 | 不变，每次更新 768 条轨迹 |
| PPO mini-batch | 96 prompts | 96 | 96 | 不变，累积完成后仅一次 optimizer/EMA update |
| Learning rate / warmup | 2e-6 / 10 steps | 相同 | 相同 | 不变 |
| Epoch / updates | 1 / 65 | 1 / 65 | 1 / 65 | 不变，`drop_last=True` |
| Prompt / response | 8192 / 1024 | 相同 | 相同 | 不截断、不过滤 |
| VOPD alpha / Top-K | 0.5 / 100 | 相同 | 相同 | 不变 |
| Teacher | EMA, rate 0.05 | 相同 | 相同 | 不变，另存 teacher checkpoint |
| Actor sequence parallel | 1 | 4 | 8 | 分解同一完整序列；collective/浮点归约顺序不同 |
| Per-rank dynamic token budget | 9216 | 2304 | 1152 | 乘以 SP 后均为完整 9216 tokens |
| Rollout tensor parallel | 1 | 1 | 8 | 4B 对齐官方 TP1；随机 rollout 仍不保证逐 token 相同 |
| Rollout GPU utilization | 0.70 | 0.30 | 0.25 | 为 actor/teacher forward 预留显存 |
| Rollout `max_num_seqs` | config default 1024 | 16 | 32 | 仅限制 vLLM 并发，不改变 768 条轨迹总数 |
| Layered summon | default off | on | on | rollout/actor 分阶段唤醒以降低共驻显存 |
| Param / optimizer / activation offload | on / on / off | on / on / on | on / on / on | 不改目标；activation 重计算路径不同，非 bitwise identical |
| Dataloader workers | 8 | 0 | 0 | 防止多进程预加载图片，shuffle/sampler 不变 |
| Multimodal storage | eager tensors | lazy paths | lazy paths | forward 时用同一 processor 重建 |
| Lazy tensor dtype | native eager tensor | bfloat16 cache | bfloat16 cache | 只用于工具生成图缓存；真实 forward processor 输出已逐 bit 验证 |
| Agent dispatch | 整批 | 8 waves x 96 | 8 waves x 96 | 汇总全部 768 条后才更新 |
| Checkpoint frequency | disabled | every step | every step | 仅容错，不增加训练更新 |
| Checkpoint retention | unlimited/default | latest 2 full | latest 2 full | 旧 step 仅留 dataloader stub，避免磁盘增长 |

这里的 `strict` 指核心训练语义严格，不表示 topology-identical 或 bitwise-identical。
本地节点是 8x46GB L40S；96x8 的官方 SP1 配置已经实测在 actor backward 阶段 GPU
OOM，同时 eager 保存 768 条多模态张量也曾触发 Ray 主机内存 OOM。因此各项偏差的
原因和风险需要分开理解：

| 偏差 | 直接原因 | 是否改变目标函数/更新次数 | 可能影响 | 风险判断 |
| --- | --- | --- | --- | --- |
| Actor SP1 -> SP4 | 单卡 GPU activation/attention 显存不足 | 否 | attention collective、浮点归约顺序和部分 kernel 路径改变，误差可随 65 次更新累积 | 中等，不能宣称逐参数复现 |
| Per-rank 9216 -> 2304 | 与 SP4 配套，每个 SP group 仍覆盖 9216 tokens | 否 | 动态 micro-batch/梯度累积布局改变；完整序列、loss normalization、单次 optimizer/EMA update 保持 | 低到中等，与 SP 共同评估 |
| Rollout TP1 -> TP1（当前 4B） | 无偏差；历史已完成模型曾用 TP4 | 否 | 当前模型该项已对齐；temperature 1 下异步调度本身仍不保证相同采样 | 当前低；历史 TP4 为中等 |
| `max_num_seqs` 1024 -> 16 | 限制 vLLM KV cache、GPU 峰值和 96x8 多模态主机并发 | 否 | 不减少 768 条轨迹，但会改变请求批形状、完成顺序和随机采样的实际轨迹 | 低到中等，分布目标相同但样本路径不同 |
| Layered summon off -> on | 降低 FSDP actor 与 vLLM rollout 共驻/同步时的 GPU 峰值 | 否 | 权重值应相同，主要改变逐层同步、sleep/wake 时序和吞吐；实现错误可能造成 stale weight | 低，需 rollout/training 一致性指标守护 |

以上 SP/offload 适配保持数据、完整序列长度、768 条轨迹的 global update、loss
归一化、optimizer 和 EMA 更新语义不变，因此属于算法等价的显存布局调整，而不是
topology-identical 或 bitwise reproduction。Activation offload 会替换 gradient
checkpoint 的重计算实现；Ulysses SP 会改变 attention collective、梯度归约顺序，且
VLM vision attention 在 SP 模式下使用 eager 路径；rollout TP 也可能因浮点差异改变
temperature 1 的随机采样轨迹。这些差异预期只产生数值/随机方差，不能先验保证最终
checkpoint 与官方逐参数一致。正式结论仍以 pristine 官方六项 benchmark 和论文指标
门禁为准；固定 batch 的单步 A/B 等价性测试用于隔离 offload 与 SP 的 loss、grad、
parameter delta 和 EMA delta 差异。

### 2.1 Vision-OPD 训练指标阅读指南

当前 `loss_mode=vopd`、`reward_model.enable=False`、`use_kl_loss=False`，因此本次
训练不是常规带 reward/critic 的 GRPO。正常情况下实际反向优化项为
`actor/vopd_loss`；`actor/pg_loss` 与它相同。以下指标应分组阅读：

| 类别 | 指标 | 本次预期与含义 | 需要警惕的变化 |
| --- | --- | --- | --- |
| 实际目标 | `actor/vopd_loss` | student 全图分布与 EMA teacher bbox 分布的 Top-K JSD；warmup 后看滑动均值下降 | NaN/Inf、持续上升；快速趋近 0 也需结合 probe，可能只是 teacher/student 同化 |
| 实际目标 | `self_distillation/raw_jsd_token_mean` | 未加权 token JSD，应与 VOPD loss 同方向且数值接近 | 与 VOPD loss 长期背离，提示 mask/权重或归一化异常 |
| 总策略项 | `actor/pg_loss` | 无 fallback 时等于 VOPD loss | 与 VOPD loss 不一致且 fallback 仍为 0 |
| 未启用项 | `actor/grpo_loss` | **应为 0**；只有缺失 teacher/bbox 的样本才进入 GRPO fallback | 非 0 表示数据缺少 teacher 输入或 fallback 被触发 |
| 未启用项 | `actor/kl_loss` | **应为 0**；配置关闭显式 reference-model KL 正则 | 非 0 表示训练目标配置偏离官方 VOPD 设置 |
| 更新强度 | `actor/grad_norm` | clip 前总梯度范数，阈值为 1.0；偶发大于 1 表示发生正常 clipping | NaN/Inf、长期远大于 10，或连续接近 0 表示爆炸/消失风险 |
| 学习率 | `actor/lr` | step 1 为 0，10-step 线性 warmup 后固定 2e-6 | 不符合该 schedule 表示 resume/scheduler 语义错误 |
| Teacher 覆盖 | `teacher_always_on_fraction`、`teacher_image_swap_fraction`、`self_distillation_mask.mean()` | 每一步必须严格为 1，表示所有轨迹都有 bbox teacher 输入 | 任一低于 1 都会改变训练样本和 loss 组成 |
| Fallback | `self_distillation/policy_fallback_fraction`、`actor/policy_fallback_fraction` | 每一步必须为 0；表示全部轨迹使用 VOPD，不需要 GRPO fallback | 大于 0 表示部分轨迹缺失 teacher/bbox；它不是一般意义的 rollout 生成失败率 |
| 有效监督 | `num_distill_tokens`、`empty_target_batch` | token 数必须大于 0，empty 必须为 0 | token 长期骤降或 empty 非 0 表示监督 mask 异常 |
| Policy 一致性 | `rollout_corr/kl`、`k3_kl` | **诊断项而非 loss**；衡量 vLLM rollout policy 与当前 FSDP actor 的差距，越小越稳定 | 持续上升或明显超过早期水平，提示 stale policy、并行数值差异或权重同步问题 |
| Policy 一致性 | `training_ppl`、`rollout_ppl`、`log_ppl_abs_diff`、`ppl_ratio` | PPL 受 batch 内容/短序列影响，优先比较两侧差值和 ratio，而非单看绝对 PPL | 两侧长期分离，或 ratio/绝对 log 差持续扩大 |
| 输出质量 | `response_length/clip_ratio` | 少量触及 1024 cap 可接受，重点看是否持续上升 | 高比例截断会丢失回答尾部并污染蒸馏目标 |
| 数据完整性 | `prompt_length/clip_ratio`、`response/aborted_ratio` | 本次必须严格为 0 | 非 0 表示输入被截断或 rollout 异常中止 |
| EMA | `teacher_ema_update` timing | 每个成功 optimizer step 必须大于 0 | 为 0 表示 teacher 未随 student 更新，checkpoint 也不可安全续训 |
| 资源/速度 | GPU/CPU memory、`timing_s/step` | 用于发现 OOM、泄漏或阶段性变慢，不直接衡量模型效果 | 同阶段内存单调增长或 step time 持续偏离历史范围 |

`actor/policy_fallback_fraction=0` 的精确定义是：当前 micro-batch 中
`self_distillation_mask <= 0.5` 的样本比例为 0，因此 `pg_loss=vopd_loss`，没有额外
GRPO fallback loss。它能证明 teacher/bbox 覆盖完整，但不能单独证明参数更新正常；
正常更新还需同时看到非零有限的 VOPD loss、有限 grad norm、正的 optimizer/EMA
timing、连续 checkpoint 和下降的早期健康趋势。当前自动诊断见
`docs/strict_4b_training_health.md`，最终效果仍以官方六项 benchmark 为准。

| 组件 | 版本 |
| --- | --- |
{environment_table()}

Active 4B attempt 1 于 `2026-07-19T01:57:15Z` 启动。下表区分启动时实际加载的
源码快照与启动后更新的 validator/wrapper；运行中的 Python worker 不会热加载后者。

| 文件 | SHA-256 | Provenance |
| --- | --- | --- |
{chr(10).join(runtime_hash_rows)}

Active attempt 1 的精确 4B wrapper 已归档到
`benchmark/official_reproduction_20260717/provenance/run_vision_opd_released_b96_r8_gradaccum_4b.active_attempt1.sh`；
其 SHA-256 重新计算为 `0ef15286e1fe2834656b1b13d04b8de0ca397dbb9bc68ad2b07d487af58f84bf`，
与启动快照一致。
9B wrapper 会在实际启动时自行复制到同一 provenance 目录并写入 `.sha256` sidecar，
因此最终报告将采用启动进程生成的快照，而非事后读取可变工作树。

## 3. 模型与训练数据审计

| 模型 | 本地源模型状态 |
| --- | --- |
| Qwen3.5-2B（后续小规模验证） | {source_model_state(root.parent / 'MODEL_ZOO' / 'Qwen' / 'Qwen3.5-2B')}；`.download_validated` present |
| Qwen3.5-4B | {source_model_state(root.parent / 'MODEL_ZOO' / 'Qwen' / 'Qwen3.5-4B')} |
| Qwen3.5-9B | {source_model_state(root.parent / 'MODEL_ZOO' / 'Qwen' / 'Qwen3.5-9b')} |

官方 judge `openai/gpt-oss-120b` 已通过 `scripts/validate_gpt_oss_judge.py`：
15 个 shard、687 个 index tensor，所有 SHA marker、shape 与 index/shard key 完全一致；
config/tokenizer/chat template 可离线加载，vLLM `openai_gptoss` 成功解析为
`GptOssReasoningParser`。持久化证据为 `logs/gpt_oss_judge_validation.log` 和
`outputs/vision_opd_gpt_oss_judge_validated` marker。

`data/train.parquet` 包含 6241 条数据，核心字段为 `prompt`、`images`、
`bbox_images`、`reward_model` 和 `extra_info`。当前已逐条验证 6,241 个原图引用
和 6,241 个 bbox context 图引用全部存在且非空；最终 verifier 会重复执行该检查。

## 4. 训练工作流

1. 核对论文 PDF 与 released config，区分论文明确参数和代码默认参数。
2. 验证本地 Qwen3.5-4B/9B 权重索引、所有 shard、tokenizer 和 processor。
3. 历史 `batch=8/rollout=1` 运行完成 780 step 并合并，仅作为本地消融。
4. 先用 pristine 官方配置完成未训练 4B/9B baseline 六项推理和 GPT-OSS 判分，
   通过相对论文 baseline 的偏差 gate 后再测试历史 merged checkpoint。
5. 严格重训恢复 released `batch=96/rollout=8`；4B 使用 SP4 与 2304-token
   per-GPU dynamic budget，9B 使用 SP8 与 1152-token budget。每个 9216-token
   完整序列不截断，768 条轨迹经 gradient accumulation 后执行一次 optimizer/EMA update。
6. 对严格重训 checkpoint 使用同一官方六项评测，分析本地 gain、论文偏差和
   holdout 风险；若未达到对齐 gate，依据训练/评测证据调整非语义性内存布局或继续训练审计。
   训练后 OPD 验收要求每项绝对偏差不超过 3pp、六项 macro 绝对偏差不超过 2pp；
   该门禁失败会在启动对应后续 backbone 前终止流水线。

以下表格仅描述历史 `batch=8/rollout=1` checkpoint：

| 历史模型 | Checkpoint | 日志 step | 合并模型 | 最大响应 | 触及 1024 cap 的 step | prompt 截断 step | 峰值 allocated GPU |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
{chr(10).join(training_rows)}

| 模型 | 前 50 步 VOPD loss | 后 50 步 VOPD loss | 前 50 步 grad norm | 后 50 步 grad norm |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(optimization_rows)}

严格 released `batch=96/rollout=8` 进度：

| 模型 | Checkpoint | 日志 step | 后段 VOPD loss | 后段 grad norm | 峰值 GPU allocated | 峰值 CPU used |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(strict_training_rows)}

4B 宿主内存定时采样：

| 样本数 | Checkpoint 范围 | 最低 MemAvailable GiB | 最大 actor RSS GiB | 最大 TaskRunner RSS GiB | 最大 AgentLoop RSS GiB | 最大 vLLM RSS GiB | 最大 Ray object store GiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{strict_memory_row}

9B 宿主内存定时采样（仅在 4B 对齐门禁通过后启动）：

| 样本数 | Checkpoint 范围 | 最低 MemAvailable GiB | 最大 actor RSS GiB | 最大 TaskRunner RSS GiB | 最大 AgentLoop RSS GiB | 最大 vLLM RSS GiB | 最大 Ray object store GiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{strict_nine_memory_row}

| 模型 | 官方推理记录 | GPT-OSS Judge | Score 文件 |
| --- | ---: | ---: | ---: |
| 4B strict | {strict_four_eval[0]}/45145 | {strict_four_eval[1]}/45145 | {strict_four_eval[2]}/10 |
| 9B strict | {strict_nine_eval[0]}/45145 | {strict_nine_eval[1]}/45145 | {strict_nine_eval[2]}/10 |

## 5. 失败尝试与决策复盘

- released batch 96 / rollout 8 的首次 SP1 运行在 8x46GB L40S 上 actor backward
  OOM；当时降为 batch 8 / rollout 1 完成了历史运行，但这改变了 optimizer/EMA
  update 数和 on-policy 分布，不能作为 released config 严格复现。
- 新严格 wrapper 保留 96x8，通过 SP4/SP8、每完整序列一个 dynamic micro-batch、
  activation/FSDP offload 和梯度累积降低峰值，而不是减少 rollout。
- 96x8 多模态轨迹最初在 Ray 返回/汇总阶段因重复持有 `pixel_values` 导致 CPU
  OOM。正式修复后轨迹只保存原图路径或工具生成图缓存路径及轻量 grid 元数据；
  actor/teacher forward 才重新打开图片并运行 processor。
- 首版 deferred forward 对 raw path 直接 `Image.open(...).convert("RGB")`，虽然
  `image_grid_thw` 一致，却跳过了 eager `qwen_vl_utils.fetch_image` 的 smart-resize；
  严格重开路径测试发现 `pixel_values` 最大绝对差 `0.0156862736`。该 run 在
  step 6 被立即停止，checkpoint/rollout/TensorBoard/日志均以
  `invalid_lazy_raw_reopen_20260719T013416Z` 完整归档，不进入合并或评测。
- 第二版 loader 将所有 path 都改为 `process_image`，但真实首批 768 条轨迹在 teacher
  forward 前发现 bbox eager 路径是原始 PIL 直接进入 processor，而非先经过
  `fetch_image`。因此部分 teacher 图的 `image_grid_thw` 改变；该次运行未执行
  backward、未生成 checkpoint，并以 `failed_mixed_preprocess_20260719T015300Z`
  归档。
- 当前 deferred 引用显式记录预处理来源：student 原图使用 `qwen_fetch`，保持 rollout
  侧 `fetch_image`/smart-resize；teacher bbox 与工具生成 PIL 使用 `processor_raw`，
  保持原始 PIL 直接进入 processor。4B/9B 各预检 16 条数据、student/teacher 共
  64 路真实输入的 `pixel_values` 和 `image_grid_thw` 均逐 bit 一致；严格 4B 于
  2026-07-19T01:57:15Z 从 step 0 干净重启。
- {strict_rollout_summary}
- 首次 lazy-loading 严格运行推进到 step 7 后，尝试在线补装 EMA teacher checkpoint
  时只有 3/8 rank 进入 FSDP snapshot collective，另外 5 个 rank 已进入 step 8，
  形成不可安全续跑的 collective 顺序错位。该 run 已按时间戳完整归档；正式 worker
  源码现从启动时保存/加载 student、optimizer、scheduler、RNG、dataloader 和 EMA
  teacher，本次严格 4B 从 step 0 重新训练，不沿用缺失 teacher 状态的 checkpoint。
- step 43 对 rank 0 的 student/EMA teacher checkpoint 做独立参数抽样：724 个参数键中
  横跨 vision、language 与 `lm_head` 的 5 个参数各取前 4,096 元素，均有
  4,093-4,096 个元素不同，最大绝对差为 `1.33e-5` 至 `3.29e-5`；这排除了
  teacher 文件只是 student 副本的情况，并与每步正的 EMA update timing 相互印证。
- 归档实验的 rollout JSON 不逐字相同来自官方 temperature 1 的异步随机采样；正式
  评测仍固定 temperature 0，因此该训练随机性不会改变推理协议。当前实验的 step、
  loss、grad norm 和资源峰值只从当前日志/TensorBoard 动态读取，不混用归档值。
- 宿主 RSS 采样会随 FSDP 参数/优化器在 CPU 与 GPU 间 offload 而变化，且多进程
  RSS 合计会重复计算共享页；泄漏判断以同训练阶段的 TensorBoard CPU used、系统
  MemAvailable 和 Ray object store 联合为准。已归档的首次 lazy-loading run 中，step 6
  CPU used 已低于 step 5，object store 也未随 checkpoint 累积；当前从零
  strict run 的独立定时采样见上表，不与归档 run 混合。
- 已归档的 raw-reopen 无效 run 在 step 4 保存后，Ray stdout 短时未刷出 step 5
  dispatch，同期 NVML 查询间歇返回
  driver communication error；内核无 Xid/NVRM 错误，GPU 持续计算。宿主 `ray list
  tasks` 最终确认 8/8 `actor_rollout_ref_update_actor` rank 均于
  2026-07-19T00:41:43Z 进入 RUNNING 且 `error_type=None`，因此未误判为死锁或重启
  已完成 step 4 的健康训练。
- 早期 low-memory 4B 虽完成 step 779，但 response cap 仅 48，日志存在大量
  clip，不能作为论文设置复现结果，只保留为诊断和备用 merged model。
- low-memory 9B 在 step 60 暂停保留，同样不作为最终结果。
- paper-explicit 4B/9B smoke 均成功，响应长度超过 48 且 smoke clip ratio 为 0，
  证明 1024 上限路径可运行。
- TorchInductor `CompiledFxGraph.__del__` cache traceback 是非致命缓存警告；
  正式训练继续推进且没有 OOM/NCCL/RuntimeError。
- 历史诊断判分审计修复了 `Answer: B` 被错误抽取为 A、错误 MCQ 被送入 LLM judge、
  ZoomBench 数值题缺少确定性判分等问题。
- 数据审计恢复了 HR-Bench `cycle_category` 和 MME-RealWorld `category` /
  `l2_category`，并与本地 VLMEvalKit 官方聚合实现核对。
- 单卡 MME-RealWorld 吞吐约 0.3 sample/s；利用可恢复 JSONL checkpoint 切换为
  TP1 x DP8、64 API workers 后完成 23609/5462 条 EN/CN 推理。
- 历史诊断流程中，训练后 4B 对少量 fallback judge 未稳定遵循 Yes/No 格式；对 238 条
  unparseable 样本使用本地基座 `Qwen3.5-4B` judge，另 3 条 HR-Bench fallback
  保留训练后 4B judge，所有记录均保存 `judge_model` provenance；这些结果不进入
  本次 pristine 官方对齐表。

## 6. Benchmark 下载与规范化结果

Benchmark 根目录：`benchmark/`。下表覆盖 pristine `prepare_data.py` 支持的全部
15 个 benchmark 名称；论文主表评测使用其中前 6 项。

| Benchmark | 本地记录数 | 预期记录数 | 状态 |
| --- | ---: | ---: | --- |
{chr(10).join(benchmark_rows)}

HR-Bench 4K/8K 各保留 4 个等长 cycle，每组 200 条。MME-RealWorld EN 保留
9 个 category / 39 个 l2 category，CN 保留 8 个 category / 33 个 l2 category。
全量复核日志为 `logs/all_official_benchmarks_validation.log`，对应 marker 为
`outputs/vision_opd_all_official_benchmarks_validated`；当前 15 项共 56,579 条记录、
56,579 个非空图片文件。

## 7. 推理与判分工作流

1. 从 git commit `c2e345f` 导出 pristine `eval/`，核心脚本逐文件保存 SHA-256。
2. 严格推理固定 seed 42、temperature 0、thinking=False、`max_tokens=32768`、
   3 次 request retry 和 256 workers；服务 context 为 65536 以容纳图像输入加最大输出。
   按官方原始 `infer.py`，seed 42 用于 artifact tag，但未作为 OpenAI request
   字段发送；请求级确定性来自显式 `temperature=0`。本地不额外添加官方未发送的 seed。
3. 判分使用官方未修改的 `judge_qwenlm.py` 和 `openai/gpt-oss-120b`；旧的扩展
   deterministic judge、本地 Qwen fallback 和 `max_tokens=1024` 分数只保留为诊断。
4. `scripts/verify_official_baseline_alignment.py --inference-only` 在加载 GPT-OSS 前核对
   六项记录数、UID 唯一性、UID 集与官方 manifest 完全相等、空答案及错误前缀；
   通过证据保存到 `logs/official_baseline_inference_validation.log` 并创建
   `outputs/vision_opd_official_baseline_inference_validated` marker；
   完成 judge 后再核对 Yes/No、分数重算，并执行逐项 5pp / macro 3pp baseline 偏差 gate。
5. 每个训练后模型必须再通过 `scripts/validate_official_model_outputs.py`：inference UID
   集等于 manifest、judge UID 集等于 inference、judge 全部为 Yes/No，且六个 score
   均能由 judge 记录重算；通过后才创建该模型的 official evaluation 完成标记。

冻结评测副本与仓库原始 Git 版本的逐文件校验：

工作树根目录 `eval/` 中的 `run_eval.sh`、`judge_qwenlm.py`、`cal_acc.py` 含早期
diagnostic 修改，正式结果明确不引用这些文件。正式入口固定为
`benchmark/official_reproduction_20260717/source/eval/`；下表哈希均与 Git HEAD
原始文件逐字节一致。

| 文件 | SHA-256 | 状态 |
| --- | --- | --- |
{chr(10).join(eval_hash_rows)}

以下第一张表是历史 `paper-explicit/max_tokens=1024` 诊断目录，不作为严格结果：

| 历史模型 | 推理记录 | Judge 记录 | 已完成 score 文件 |
| --- | ---: | ---: | ---: |
{chr(10).join(evaluation_rows)}

严格官方 baseline 进度：

| 模型 | 推理记录 | GPT-OSS Judge | 已完成 score 文件 |
| --- | ---: | ---: | ---: |
| Qwen3.5-4B baseline | {strict_base4[0]}/45145 | {strict_base4[1]}/45145 | {strict_base4[2]}/10 |
| Qwen3.5-9B baseline | {strict_base9[0]}/45145 | {strict_base9[1]}/45145 | {strict_base9[2]}/10 |

## 8. 本地结果与论文结果

{strict_alignment_table}

严格表始终同时保留论文/本地 baseline 与论文/本地 OPD 的 4B/9B 八个结果列。
完整评测配置、逐模型 artifact 数量和 baseline 偏差见
`docs/official_evaluation_reproduction.md`。旧的 `max_tokens=1024` 和本地 fallback
judge 结果仅作为诊断数据，不进入上表。

使用未修改官方 judge 脚本和本地 Qwen2.5-72B fallback 的 interim baseline
诊断已通过同一偏差 gate：4B macro 71.71%（论文 70.68%），9B macro 74.03%
（论文 73.37%）。完整 provenance 与逐项差值见
`docs/diagnostic_qwen25_72b_baseline_alignment.md`；最终正式列仍只接受 GPT-OSS。

## 9. 自动化与关键产物

- 严格训练入口：`scripts/run_vision_opd_released_b96_r8_gradaccum_4b.sh`、
  `scripts/run_vision_opd_released_b96_r8_gradaccum_9b.sh`
- 当前 4B TP1 内存拓扑入口：
  `scripts/run_vision_opd_released_b96_r8_gradaccum_4b_tp1_retry.sh`；完成后由
  `scripts/watch_strict_4b_tp1_retry_and_evaluate.sh` 校验、合并、执行目标 10 项官方评测和
  VTC-Bench code/interface 两条 track
- VTC-Bench 本地服务固定为 DP8/TP1、context 65536，并按 Qwen3.5 模型卡使用
  `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`；
  通过 `--default-chat-template-kwargs '{{"enable_thinking":true}}'` 显式对齐 Thinking
  track，且不使用与 Qwen3.5 XML tool-call 格式不匹配的 Hermes parser；官方 Vision-OPD
  10 项 benchmark 仍独立保持 `thinking=False`
- 严格 4B→9B controller：`scripts/run_strict_released_reproduction_pipeline.sh`；
  4B 未通过论文对齐门禁时不会启动 9B
- 历史消融/诊断入口：`scripts/run_vision_opd_paper_explicit_local_4b.sh`、
  `scripts/run_vision_opd_paper_explicit_local_9b.sh`；其产物不进入最终八列表
- Benchmark 准备：`scripts/prepare_benchmarks.sh`
- 全部 15 项 benchmark 验证：`scripts/validate_all_official_benchmarks.py`
- merged model 完整性验证：`scripts/validate_merged_model.py` 核对多模态 architecture、
  vision/processor/generation/chat-template 配置、safetensors shape/metadata 及 index 与实际 key 精确一致
- strict checkpoint 完整性验证：`scripts/validate_strict_checkpoint.py` 在合并前强制
  核对 student、optimizer、extra state、EMA teacher、dataloader state 与两套配置
- strict resolved config 验证：`scripts/validate_strict_runtime_config.py` 从实际训练日志
  解析 Hydra config，分别强制 4B/9B 的官方语义参数和内存拓扑
- strict 训练指标验证：`scripts/validate_strict_training_metrics.py` 强制 step 连续、
  指标有限、teacher/bbox 始终开启、无 fallback/空目标/prompt 截断/abort、每步完成 EMA，
  并逐 step 精确核对 10-step warmup 到 `2e-6` constant plateau 的学习率曲线
- strict rollout 验证：`scripts/validate_strict_rollout.py` 对单步 JSONL 强制检查
  `768 = 96 prompts x 8 trajectories`、JSON 可解析性，以及不存在 `pixel_values`、
  image grid、bytes 或 deferred payload
- 4B/9B 逐 step 语义守护：`scripts/watch_strict_4b_semantics.sh` 和
  `scripts/watch_strict_9b_semantics.sh` 在 checkpoint marker 更新后自动串联 checkpoint、
  metrics、rollout 三项验证并刷新本报告；任何验证失败都会令对应守护进程非零退出并
  保留最后一条失败日志
- lazy 图像数值等价验证：`scripts/validate_lazy_image_equivalence.py` 分别按
  `qwen_fetch` student 路径和 `processor_raw` teacher 路径比较 eager/deferred
  processor；4B/9B 各 16 条预检样本、共 64 路输入逐 bit 一致，正式 controller
  启动时还会再次执行 smoke gate
- path-only 数据启动 gate：`scripts/validate_training_data_paths.py` 强制训练 parquet
  为 6,241 行，`images`/`bbox_images` 均只能包含单个 `path: string`，并验证全部
  12,482 个本地图片文件存在且非空；4B/9B wrapper 在加载模型前执行，持久化证据为
  `logs/path_only_training_data_validation.log` 和
  `outputs/vision_opd_path_only_training_data_validated`
- 单模型官方 inference/judge/score 闭环验证：`scripts/validate_official_model_outputs.py`
- 冻结官方评测源：`benchmark/official_reproduction_20260717/source/eval/`
- strict 单模型评测入口：`scripts/evaluate_official_single_model.sh`；baseline
  GPT-OSS 判分入口：`scripts/judge_official_baselines.sh`
- GPT-OSS judge 启动前完整性验证：`scripts/validate_gpt_oss_judge.py`
- 最终对齐 verifier：`scripts/verify_official_opd_alignment.py`、
  `scripts/verify_official_baseline_alignment.py`
- 历史诊断分数报告：`docs/reproduction_results_paper_explicit.md`
- 严格八列评测报告：`docs/official_evaluation_reproduction.md`
- Baseline 临时对齐诊断：`docs/diagnostic_qwen25_72b_baseline_alignment.md`
- 正式下载后评测守护：`scripts/continue_official_eval_after_judge_download.sh`
- 4B 完成后 prerequisite 守护：`scripts/watch_post_strict_4b_prerequisites.sh`
- 主控制器与 prerequisite 守护共用文件锁，避免 merge 后重复启动 GPT-OSS judge/官方评测并争抢 GPU
- post-4B watcher 必须先看到 strict merged model 验证通过；baseline GPT-OSS wrapper
  启动前再执行 `ray stop --force`，清除训练退出后可能残留的 CUDA context，避免
  120B judge 与 detached Ray worker 争抢显存
- baseline 与 strict OPD 判分支持 benchmark 级断点续跑；仅当 inference 数量等于
  manifest、无 API/FUTURE error、judge 数量完整且全部为 Yes/No 时复用既有结果，
  否则仍调用 pristine `run_eval.sh`/`judge_qwenlm.py` 重新完成该 benchmark
- 4B 宿主内存监控：`scripts/monitor_strict_4b_memory.sh` 每 5 分钟记录 checkpoint、
  MemAvailable、actor/TaskRunner/AgentLoop/vLLM RSS、Ray object store 与 GPU 进程显存到
  `logs/strict_4b_tp1_memory_monitor.csv`
- 4B checkpoint 完整性守护：`scripts/watch_strict_4b_checkpoint_integrity.sh` 在每个
  marker 落盘后验证 8-rank student、optimizer、extra state、EMA teacher、dataloader
  state 及 HF 配置，并强制检查第 N 步恰好消费 N 个 batch / `N x 96` prompts、
  sampler generator state 非空；同时逐 rank 核对 scheduler epoch/step/LR 以及
  CPU、CUDA、NumPy、Python RNG state，并检查 rank 0 全部 Adam state 的 optimizer
  counter 必须恰好为 N（证明每个 global batch 只更新一次），结果写入
  `logs/strict_4b_checkpoint_integrity.log`
- step 44 另做一次 8-rank 全量 optimizer counter 审计：每个 rank 均有 57 个 Adam
  state，全部 counter 精确为 44、missing=0，排除了任一 rank 对 8 个 rollout wave
  分别执行 optimizer update 的可能
- 9B 门禁后内存监控：`scripts/monitor_strict_9b_memory.sh` 等待严格 4B 完成标记后，
  以同样字段写入 `logs/strict_9b_memory_monitor.csv`；门禁前不加载 9B、不产生采样
- 本复盘报告：`docs/vision_opd_goal_reproduction_report.md`
- strict 正式 merged model：
  `merged_models/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4-rollout-tp1-retry`、
  `merged_models/Vision-OPD-Qwen3.5-9B-released-b96-r8-gradaccum-sp8`

## 10. 监控与复现命令

```bash
cd {root}
screen -ls
tail -f logs/strict_4b_tp1_retry.screen.log
tail -f logs/strict_4b_tp1_post_pipeline.log

python scripts/validate_strict_checkpoint.py \\
  checkpoints/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4-rollout-tp1-retry
```

## 11. 完成标准与剩余工作

最终完成必须同时满足：历史 4B/9B ablation 完成官方测评；严格 released
`batch=96/rollout=8` 的 4B/9B checkpoint 均恰好完成 65 个一轮更新，不接受额外的第 66 次
optimizer/EMA update；两个 strict
merged model 通过 safetensors/config/processor 完整验证；六套 benchmark 对两个
strict 模型均有完整 inference、GPT-OSS judge 和 12 个 score 文件；八列表无
`pending`；最终审计通过并创建 strict reproduction completion marker。

{remaining_status} 本报告将在最终评测后自动重写，届时本节和结果表不再包含
`pending`。
"""

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
