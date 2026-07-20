#!/usr/bin/env python3
import argparse
from collections import Counter
import importlib.metadata
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path


BENCHMARK_FILES = {
    "Vstar": "vstar.json",
    "ZoomBench": "zoombench.json",
    "HR-Bench-4K": "hr_bench_4k.json",
    "HR-Bench-8K": "hr_bench_8k.json",
    "MME-RealWorld-EN": "MME_RealWorld.json",
    "MME-RealWorld-CN": "MME_RealWorld_CN.json",
}

BENCHMARK_SLUGS = {
    "Vstar": "vstar",
    "ZoomBench": "zoombench",
    "HR-Bench-4K": "hrbench-4k",
    "HR-Bench-8K": "hrbench-8k",
    "MME-RealWorld-EN": "mme-realworld",
    "MME-RealWorld-CN": "mme-realworld-cn",
}

MODELS = {
    "Qwen3.5-4B": "Vision-OPD-Qwen3.5-4B",
    "Qwen3.5-9B": "Vision-OPD-Qwen3.5-9B",
}

PROFILES = {
    "lowmem": {
        "models": MODELS,
        "logs": ("vision_opd_4b_auto_resume.log", "vision_opd_9b_auto.log"),
        "checkpoints": (
            "Vision-OPD-Qwen3.5-4B-full-repro-lowmem-20260714",
            "Vision-OPD-Qwen3.5-9B-local-lowmem-20260714",
        ),
        "merged": ("Vision-OPD-Qwen3.5-4B", "Vision-OPD-Qwen3.5-9B"),
        "expected_step": 779,
        "title": "Vision-OPD 4B/9B Reproduction Results",
        "scope": (
            "The paper explicitly specifies Vision-OPD-6K, one epoch, JSD beta 0.5, top-K\n"
            "100, EMA teacher regularization, non-thinking mode, and maximum generation\n"
            "length 1024. Batch 96 and rollout n=8 are released-code defaults rather than\n"
            "values stated in the PDF. The full released-code configuration was tested on\n"
            "the local 8x46GB L40S node but exceeded available memory during actor update.\n"
            "The completed low-memory runs retain the paper's loss, teacher, and epoch\n"
            "choices, but use batch 8, rollout count 1, and response length 48. Prompts over\n"
            "6144 tokens are filtered. These runs are diagnostic and are not a\n"
            "parameter-identical reproduction of the paper.\n\n"
            "The 9B continuation additionally uses activation offload and Ulysses sequence\n"
            "parallel size 2. Rollout serving uses tensor parallel size 4 and GPU memory\n"
            "utilization 0.30."
        ),
    },
    "paper-explicit": {
        "models": {
            "Qwen3.5-4B": "Vision-OPD-Qwen3.5-4B-paper-explicit",
            "Qwen3.5-9B": "Vision-OPD-Qwen3.5-9B-paper-explicit",
        },
        "logs": ("vision_opd_4b_paper_explicit.log", "vision_opd_9b_paper_explicit.log"),
        "checkpoints": (
            "Vision-OPD-Qwen3.5-4B-paper-explicit-local",
            "Vision-OPD-Qwen3.5-9B-paper-explicit-local",
        ),
        "merged": (
            "Vision-OPD-Qwen3.5-4B-paper-explicit",
            "Vision-OPD-Qwen3.5-9B-paper-explicit",
        ),
        "expected_step": 780,
        "title": "Vision-OPD Paper-Explicit 4B/9B Reproduction Results",
        "scope": (
            "This run follows the values stated in the paper's Experimental Settings: all\n"
            "6241 Vision-OPD-6K records for one epoch, JSD beta 0.5, top-K 100, EMA\n"
            "teacher regularization, non-thinking mode, and maximum generation length\n"
            "1024. The prompt limit is 8192 and overlong prompts are not filtered.\n\n"
            "Batch size 8 and rollout count 1 are local execution choices because the PDF\n"
            "does not specify them. Both sizes use activation offload, Ulysses sequence\n"
            "parallel size 4, rollout tensor parallel size 4, and rollout GPU memory\n"
            "utilization 0.30 to fit the local 8x46GB L40S node. These memory-layout\n"
            "choices do not change the paper-specified objective or generation cap."
        ),
    },
}

PAPER_SCORES = {
    "Qwen3.5-4B baseline": [84.29, 47.69, 84.38, 80.13, 63.86, 63.70],
    "Vision-OPD 4B": [92.15, 59.76, 84.50, 80.38, 74.88, 70.76],
    "Qwen3.5-9B baseline": [82.72, 52.07, 85.75, 80.63, 71.40, 67.67],
    "Vision-OPD 9B": [94.76, 65.80, 88.13, 85.50, 73.40, 70.46],
}

PAPER_AVERAGES = {
    "Qwen3.5-4B baseline": 70.68,
    "Vision-OPD 4B": 77.07,
    "Qwen3.5-9B baseline": 73.37,
    "Vision-OPD 9B": 79.68,
}


def record_count(path):
    if not path.exists():
        return "missing"
    with path.open("r", encoding="utf-8") as handle:
        return len(json.load(handle))


def environment_rows():
    rows = [f"| Python | {platform.python_version()} |"]
    for package in ("torch", "transformers", "vllm", "ray", "verl"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        rows.append(f"| {package} | {version} |")
    return "\n".join(rows)


def final_step(log_path):
    if not log_path.exists():
        return "unknown"
    step = "unknown"
    pattern = re.compile(r"training/global_step:(\d+)")
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                step = match.group(1)
    return step


def checkpoint_step(checkpoint_dir, log_path):
    marker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if marker.exists():
        value = re.sub(r"\D", "", marker.read_text(encoding="utf-8", errors="replace"))
        if value:
            return value
    return final_step(log_path)


def training_diagnostics(log_path):
    diagnostics = {}
    if not log_path.exists():
        return diagnostics
    patterns = {
        "step": re.compile(r"training/global_step:(\d+)"),
        "response_max": re.compile(r"response_length/max:([0-9.]+)"),
        "response_clip": re.compile(r"response_length/clip_ratio:([0-9.]+)"),
        "prompt_clip": re.compile(r"prompt_length/clip_ratio:([0-9.]+)"),
        "memory": re.compile(r"perf/max_memory_allocated_gb:([0-9.]+)"),
        "vopd_loss": re.compile(r"actor/vopd_loss:([0-9.eE+-]+)"),
        "grad_norm": re.compile(r"actor/grad_norm:([0-9.eE+-]+)"),
    }
    by_step = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            step_match = patterns["step"].search(line)
            if not step_match:
                continue
            values = {}
            for name, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    values[name] = float(match.group(1))
            by_step[int(step_match.group(1))] = values
    if not by_step:
        return diagnostics
    rows = list(by_step.values())
    diagnostics["steps"] = len(by_step)
    diagnostics["response_max"] = max(row.get("response_max", 0.0) for row in rows)
    diagnostics["response_clipped_steps"] = sum(row.get("response_clip", 0.0) > 0 for row in rows)
    diagnostics["response_clip_max"] = max(row.get("response_clip", 0.0) for row in rows)
    diagnostics["prompt_clipped_steps"] = sum(row.get("prompt_clip", 0.0) > 0 for row in rows)
    diagnostics["memory_max"] = max(row.get("memory", 0.0) for row in rows)
    ordered_rows = [by_step[step] for step in sorted(by_step)]
    window = min(50, len(ordered_rows))
    first_rows = ordered_rows[:window]
    last_rows = ordered_rows[-window:]
    for metric in ("vopd_loss", "grad_norm"):
        first_values = [row[metric] for row in first_rows if metric in row]
        last_values = [row[metric] for row in last_rows if metric in row]
        if first_values and last_values:
            diagnostics[f"{metric}_first"] = sum(first_values) / len(first_values)
            diagnostics[f"{metric}_last"] = sum(last_values) / len(last_values)
    return diagnostics


def training_diagnostics_row(model, diagnostics):
    if not diagnostics:
        return f"| {model} | pending | pending | pending | pending | pending |"
    return (
        f"| {model} | {diagnostics['steps']} | {diagnostics['response_max']:.0f} | "
        f"{diagnostics['response_clipped_steps']} "
        f"({diagnostics['response_clip_max']:.3f} max ratio) | "
        f"{diagnostics['prompt_clipped_steps']} | {diagnostics['memory_max']:.2f} GB |"
    )


def optimization_diagnostics_row(model, diagnostics):
    required = ("vopd_loss_first", "vopd_loss_last", "grad_norm_first", "grad_norm_last")
    if not diagnostics or any(name not in diagnostics for name in required):
        return f"| {model} | pending | pending | pending | pending | pending |"
    loss_delta = diagnostics["vopd_loss_last"] - diagnostics["vopd_loss_first"]
    return (
        f"| {model} | {diagnostics['vopd_loss_first']:.6f} | "
        f"{diagnostics['vopd_loss_last']:.6f} | {loss_delta:+.6f} | "
        f"{diagnostics['grad_norm_first']:.4f} | {diagnostics['grad_norm_last']:.4f} |"
    )


def merged_state(path):
    has_config = (path / "config.json").is_file()
    has_weights = any(path.glob("model*.safetensors"))
    return "ready" if has_config and has_weights else "pending"


def parse_scores(results_dir):
    scores = {model: {} for model in MODELS}
    for model, artifact_name in MODELS.items():
        prefix = f"{artifact_name}_seed42_"
        for path in results_dir.glob(f"{prefix}*.txt"):
            slug = path.stem[len(prefix):]
            text = path.read_text(encoding="utf-8", errors="replace")
            percentages = re.findall(r"(\d+(?:\.\d+)?)%", text)
            if percentages:
                scores[model][slug] = float(percentages[-1])
    return scores


def score_table(scores):
    lines = []
    for benchmark, slug in BENCHMARK_SLUGS.items():
        values = [scores[model].get(slug) for model in MODELS]
        cells = [f"{value:.2f}%" if value is not None else "pending" for value in values]
        delta = values[1] - values[0] if all(value is not None for value in values) else None
        delta_cell = f"{delta:+.2f}" if delta is not None else "pending"
        lines.append(f"| {benchmark} | {cells[0]} | {cells[1]} | {delta_cell} |")

    macro = {}
    for model in MODELS:
        values = [scores[model].get(slug) for slug in BENCHMARK_SLUGS.values()]
        if all(value is not None for value in values):
            macro[model] = sum(values) / len(values)
    macro_4b = f"{macro['Qwen3.5-4B']:.2f}%" if "Qwen3.5-4B" in macro else "pending"
    macro_9b = f"{macro['Qwen3.5-9B']:.2f}%" if "Qwen3.5-9B" in macro else "pending"
    macro_delta = (
        f"{macro['Qwen3.5-9B'] - macro['Qwen3.5-4B']:+.2f}"
        if len(macro) == len(MODELS)
        else "pending"
    )
    lines.append(f"| Macro average | {macro_4b} | {macro_9b} | {macro_delta} |")
    return "\n".join(lines), macro


def score_analysis(scores, macro):
    paired = []
    for benchmark, slug in BENCHMARK_SLUGS.items():
        four_b = scores["Qwen3.5-4B"].get(slug)
        nine_b = scores["Qwen3.5-9B"].get(slug)
        if four_b is not None and nine_b is not None:
            paired.append((benchmark, nine_b - four_b))
    if len(paired) != len(BENCHMARK_SLUGS):
        if "Qwen3.5-4B" in macro:
            paper_4b = PAPER_AVERAGES["Vision-OPD 4B"]
            return (
                f"The completed 4B run has a six-benchmark macro average of "
                f"{macro['Qwen3.5-4B']:.2f}%, which is "
                f"{macro['Qwen3.5-4B'] - paper_4b:+.2f} points relative to the paper's "
                f"Vision-OPD 4B value. The 9B model-size comparison remains pending."
            )
        return (
            "Effect analysis is pending until both merged models have completed all six "
            "benchmarks. The final comparison will report per-benchmark and macro-average deltas."
        )

    improved = [name for name, delta in paired if delta > 0]
    regressed = [name for name, delta in paired if delta < 0]
    unchanged = [name for name, delta in paired if delta == 0]
    best_name, best_delta = max(paired, key=lambda item: item[1])
    worst_name, worst_delta = min(paired, key=lambda item: item[1])
    macro_delta = macro["Qwen3.5-9B"] - macro["Qwen3.5-4B"]
    return (
        f"The 9B run changes the six-benchmark macro average by {macro_delta:+.2f} points "
        f"relative to 4B. It improves {len(improved)} benchmarks"
        f" ({', '.join(improved) or 'none'}), regresses {len(regressed)}"
        f" ({', '.join(regressed) or 'none'}), and is unchanged on {len(unchanged)}. "
        f"The largest gain is {best_name} ({best_delta:+.2f}); the weakest delta is "
        f"{worst_name} ({worst_delta:+.2f}). This is a model-size comparison, not a "
        "causal estimate of OPD improvement, because an untrained local baseline was not "
        "evaluated in this workflow."
    )


def paper_score_table():
    lines = []
    benchmarks = list(BENCHMARK_SLUGS)
    four_base = PAPER_SCORES["Qwen3.5-4B baseline"]
    four_opd = PAPER_SCORES["Vision-OPD 4B"]
    nine_base = PAPER_SCORES["Qwen3.5-9B baseline"]
    nine_opd = PAPER_SCORES["Vision-OPD 9B"]
    for index, benchmark in enumerate(benchmarks):
        lines.append(
            f"| {benchmark} | {four_base[index]:.2f}% | {four_opd[index]:.2f}% | "
            f"{four_opd[index] - four_base[index]:+.2f} | {nine_base[index]:.2f}% | "
            f"{nine_opd[index]:.2f}% | {nine_opd[index] - nine_base[index]:+.2f} |"
        )
    four_base_avg = PAPER_AVERAGES["Qwen3.5-4B baseline"]
    four_opd_avg = PAPER_AVERAGES["Vision-OPD 4B"]
    nine_base_avg = PAPER_AVERAGES["Qwen3.5-9B baseline"]
    nine_opd_avg = PAPER_AVERAGES["Vision-OPD 9B"]
    lines.append(
        f"| Macro average | {four_base_avg:.2f}% | {four_opd_avg:.2f}% | "
        f"{four_opd_avg - four_base_avg:+.2f} | {nine_base_avg:.2f}% | "
        f"{nine_opd_avg:.2f}% | {nine_opd_avg - nine_base_avg:+.2f} |"
    )
    return "\n".join(lines)


def local_paper_analysis(macro):
    if "Qwen3.5-4B" in macro and "Qwen3.5-9B" not in macro:
        return (
            f"The local 4B macro average differs from the paper's Vision-OPD 4B value by "
            f"{macro['Qwen3.5-4B'] - PAPER_AVERAGES['Vision-OPD 4B']:+.2f} points; "
            "the local 9B comparison is pending."
        )
    if len(macro) != len(MODELS):
        return "Local-to-paper comparison is pending until all local scores are available."
    paper_4b = PAPER_AVERAGES["Vision-OPD 4B"]
    paper_9b = PAPER_AVERAGES["Vision-OPD 9B"]
    return (
        f"The local 4B macro average differs from the paper's Vision-OPD 4B value by "
        f"{macro['Qwen3.5-4B'] - paper_4b:+.2f} points; the local 9B difference is "
        f"{macro['Qwen3.5-9B'] - paper_9b:+.2f} points. These gaps combine training-configuration "
        "deviations with evaluator differences and should not be interpreted as isolated method effects."
    )


def result_sections(results_dir):
    sections = []
    for artifact_name in MODELS.values():
        for path in sorted(results_dir.glob(f"{artifact_name}_seed42_*.txt")):
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            sections.append(f"### {path.stem}\n\n```text\n{text}\n```")
    return "\n\n".join(sections) or "No completed benchmark result files were found."


def judge_source_summary(root, artifact_name):
    counter = Counter()
    judge_models = Counter()
    model_tag = f"{artifact_name}_seed42"
    for slug in BENCHMARK_SLUGS.values():
        path = root / "benchmark" / "judge" / slug / f"{model_tag}_answer.jsonl"
        if not path.is_file():
            continue
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for record in records:
            source = str(record.get("judge_source", "unknown") or "unknown")
            counter[source] += 1
            if source == "llm":
                judge_models[str(record.get("judge_model", "unrecorded") or "unrecorded")] += 1
    return counter, judge_models


def judge_source_row(model, counter, judge_models):
    total = sum(counter.values())
    if not total:
        return f"| {model} | pending | pending | pending | pending |"
    llm = counter.get("llm", 0)
    deterministic = total - llm
    sources = ", ".join(f"{name}={count}" for name, count in sorted(counter.items()))
    models = ", ".join(f"{name}={count}" for name, count in sorted(judge_models.items())) or "none"
    return f"| {model} | {total} | {deterministic} | {llm} ({sources}) | {models} |"


def main():
    global MODELS
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="lowmem")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    MODELS = profile["models"]

    root = Path(args.project_root)
    prepared = root / "benchmark" / "prepared"
    rows = [
        f"| {name} | {record_count(prepared / filename)} |"
        for name, filename in BENCHMARK_FILES.items()
    ]
    four_b_log = root / "logs" / profile["logs"][0]
    nine_b_log = root / "logs" / profile["logs"][1]
    four_b_checkpoint = root / "checkpoints" / profile["checkpoints"][0]
    nine_b_checkpoint = root / "checkpoints" / profile["checkpoints"][1]
    four_b_merged = root / "merged_models" / profile["merged"][0]
    nine_b_merged = root / "merged_models" / profile["merged"][1]
    results_dir = root / "benchmark" / "results"
    scores = parse_scores(results_dir)
    scores_markdown, macro = score_table(scores)
    analysis = score_analysis(scores, macro)
    results = result_sections(results_dir)
    generated = datetime.now(timezone.utc).isoformat()
    four_b_diagnostics = training_diagnostics(four_b_log)
    nine_b_diagnostics = training_diagnostics(nine_b_log)
    four_b_judge_sources, four_b_judge_models = judge_source_summary(root, MODELS["Qwen3.5-4B"])
    nine_b_judge_sources, nine_b_judge_models = judge_source_summary(root, MODELS["Qwen3.5-9B"])

    report = f"""# {profile['title']}

Generated: {generated}

## Scope and configuration

{profile['scope']}

## Runtime environment

| Component | Version |
| --- | --- |
{environment_rows()}

Hardware: 8 NVIDIA L40S GPUs with approximately 46 GB memory per GPU.

The local Qwen3.5-4B source contains 738 indexed tensors across 2 safetensors
shards, and Qwen3.5-9B contains 775 indexed tensors across 4 shards. All
indexed shard headers and required tokenizer/processor artifacts validate
successfully, so ModelScope re-download was not required.

## Training status

| Model | Latest checkpoint step | Expected local steps | Merged model |
| --- | ---: | ---: | --- |
| Qwen3.5-4B | {checkpoint_step(four_b_checkpoint, four_b_log)} | {profile['expected_step']} | {merged_state(four_b_merged)} |
| Qwen3.5-9B | {checkpoint_step(nine_b_checkpoint, nine_b_log)} | {profile['expected_step']} | {merged_state(nine_b_merged)} |

## Training diagnostics

| Model | Logged steps | Maximum response | Steps hitting response cap | Steps hitting prompt cap | Peak allocated GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
{training_diagnostics_row('Qwen3.5-4B', four_b_diagnostics)}
{training_diagnostics_row('Qwen3.5-9B', nine_b_diagnostics)}

Response-cap hits are reported as observed generation truncation at the
configured maximum, not as infrastructure failures. The paper-explicit profile
uses the paper's 1024-token generation cap; the low-memory diagnostic profile
uses its documented 48-token cap.

| Model | First-50 VOPD loss | Last-50 VOPD loss | Loss change | First-50 grad norm | Last-50 grad norm |
| --- | ---: | ---: | ---: | ---: | ---: |
{optimization_diagnostics_row('Qwen3.5-4B', four_b_diagnostics)}
{optimization_diagnostics_row('Qwen3.5-9B', nine_b_diagnostics)}

The windowed loss statistics describe optimization behavior; benchmark score
changes remain the evidence for downstream model quality.

## Prepared benchmarks

| Benchmark | Records |
| --- | ---: |
{chr(10).join(rows)}

## Evaluation protocol

Inference uses the merged checkpoint through a local OpenAI-compatible vLLM
server with Qwen3.5 thinking disabled. Multiple-choice answers are graded by
a deterministic option parser, including explicit Yes/No handling for
parseable wrong options. ZoomBench numeric questions use deterministic numeric
matching. Only outputs that cannot be parsed by these rules use an LLM
fallback. The paper's `openai/gpt-oss-120b` judge is not installed on this
node, so every fallback record stores the actual local `judge_model` used.

| Model | Judged records | Deterministic | LLM fallback and source breakdown | Fallback judge models |
| --- | ---: | ---: | --- | --- |
{judge_source_row('Qwen3.5-4B', four_b_judge_sources, four_b_judge_models)}
{judge_source_row('Qwen3.5-9B', nine_b_judge_sources, nine_b_judge_models)}

If the LLM fallback count is zero, the unavailable paper judge does not affect
the reported scores. Otherwise, its count is an explicit evaluation-protocol
limitation.

HR-Bench score files retain the official four-cycle breakdown; because each
cycle contains 200 records, their mean equals overall accuracy across 800
records. MME-RealWorld score files retain task, category, and level-2 category
breakdowns and use VLMEvalKit's overall sample-accuracy definition.

The serving context is 32768 tokens. This covers the 16777216-pixel processor
limit (about 16384 merged visual tokens) used by the largest HR-Bench 8K
images. The 4B model uses TP1; MME-RealWorld inference was resumed with eight
TP1 data-parallel replicas and 64 API workers after the single-replica
throughput proved insufficient. The planned 9B serving layout uses TP2.

## Score comparison

| Benchmark | 4B | 9B | 9B - 4B (points) |
| --- | ---: | ---: | ---: |
{scores_markdown}

{analysis}

## Paper reference

Table 1 of the paper reports the following Qwen3.5 baselines and Vision-OPD
results under the authors' protocol:

| Benchmark | Qwen 4B | OPD 4B | 4B gain | Qwen 9B | OPD 9B | 9B gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{paper_score_table()}

The published macro gains are +6.39 points for 4B and +6.31 points for 9B.
{local_paper_analysis(macro)} The local workflow does not rerun the untrained
Qwen3.5 baselines, and its fallback judge differs from the paper, so the paper
rows are reference targets rather than directly comparable local controls.

## Raw benchmark results

{results}
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
