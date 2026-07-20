#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


BENCHMARKS = (
    ("vstar", "Vstar", 191),
    ("zoombench", "ZoomBench", 845),
    ("hrbench-4k", "HR-Bench-4K", 800),
    ("hrbench-8k", "HR-Bench-8K", 800),
    ("mme-realworld", "MME-RealWorld-EN", 23609),
    ("mme-realworld-cn", "MME-RealWorld-CN", 5462),
)
ALL_BENCHMARKS = BENCHMARKS + (
    ("mmstar", "MMStar", 1500),
    ("pope", "POPE-Test", 9000),
    ("cv-bench", "CV-Bench", 2638),
    ("mmvp", "MMVP", 300),
)

PAPER = {
    "vstar": (84.29, 92.15, 82.72, 94.76),
    "zoombench": (47.69, 59.76, 52.07, 65.80),
    "hrbench-4k": (84.38, 84.50, 85.75, 88.13),
    "hrbench-8k": (80.13, 80.38, 80.63, 85.50),
    "mme-realworld": (63.86, 74.88, 71.40, 73.40),
    "mme-realworld-cn": (63.70, 70.76, 67.67, 70.46),
}

MODELS = {
    "base4": "Qwen3.5-4B-baseline-official",
    "opd4": "Vision-OPD-Qwen3.5-4B-official",
    "base9": "Qwen3.5-9B-baseline-official",
    "opd9": "Vision-OPD-Qwen3.5-9B-official",
}

STRICT_MODELS = {
    "opd4": (
        "Vision-OPD-Qwen3.5-4B-released-b96-r8-tp1-official",
        "Vision-OPD-Qwen3.5-4B-released-b96-r8-official",
    ),
    "opd9": ("Vision-OPD-Qwen3.5-9B-released-b96-r8-official",),
}


def read_score(results_dir: Path, model: str, benchmark: str):
    path = results_dir / f"{model}_seed42_{benchmark}.txt"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if benchmark == "pope":
        matches = re.findall(r"accuracy=(\d+(?:\.\d+)?)%", text)
    else:
        matches = re.findall(r"(\d+(?:\.\d+)?)%", text)
    return float(matches[-1]) if matches else None


def answer_status(root: Path, model: str, benchmark: str, expected: int):
    path = root / "model_answer" / benchmark / f"{model}_seed42_answer.jsonl"
    if not path.is_file():
        return "0/{0}".format(expected)
    count = 0
    errors = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            try:
                answer = str(json.loads(line).get("model_answer", "")).strip()
            except Exception:
                errors += 1
                continue
            if not answer or answer.startswith(("[API_ERROR]", "[FUTURE_ERROR]")):
                errors += 1
    suffix = "" if errors == 0 else f", errors={errors}"
    return f"{count}/{expected}{suffix}"


def judge_status(root: Path, model: str, benchmark: str, expected: int):
    path = root / "judge" / benchmark / f"{model}_seed42_answer.jsonl"
    if not path.is_file():
        return f"0/{expected}"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "invalid"
    valid = sum(str(row.get("judge", "")).strip().lower() in {"yes", "no"} for row in rows)
    return f"{valid}/{expected}"


def fmt(value):
    return "pending" if value is None else f"{value:.2f}%"


def delta(local, paper):
    return "pending" if local is None else f"{local - paper:+.2f}"


def macro(values):
    return None if any(value is None for value in values) else sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project = args.project_root.resolve()
    root = project / "benchmark" / "official_reproduction_20260717"
    output = args.output or project / "docs" / "official_evaluation_reproduction.md"
    selected_models = dict(MODELS)
    for key, candidates in STRICT_MODELS.items():
        for model in candidates:
            if all(
                read_score(root / "results", model, benchmark) is not None
                for benchmark, _, _ in BENCHMARKS
            ):
                selected_models[key] = model
                break
    scores = {
        key: {benchmark: read_score(root / "results", model, benchmark) for benchmark, _, _ in BENCHMARKS}
        for key, model in selected_models.items()
    }
    ablation_scores = {
        key: {benchmark: read_score(root / "results", MODELS[key], benchmark) for benchmark, _, _ in BENCHMARKS}
        for key in ("opd4", "opd9")
    }

    lines = [
        "# Vision-OPD Official Evaluation Reproduction",
        "",
        "This report is reserved for the strict reproduction based on the pristine official",
        "`eval/run_eval.sh` implementation. Diagnostic runs with `max_tokens=1024` are excluded.",
        "",
        "## Table 1 Alignment",
        "",
        "| Benchmark | Paper Base 4B | Local Base 4B | Paper OPD 4B | Local OPD 4B | Paper Base 9B | Local Base 9B | Paper OPD 9B | Local OPD 9B |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for benchmark, label, _ in BENCHMARKS:
        paper_base4, paper_opd4, paper_base9, paper_opd9 = PAPER[benchmark]
        lines.append(
            f"| {label} | {paper_base4:.2f}% | {fmt(scores['base4'][benchmark])} | "
            f"{paper_opd4:.2f}% | {fmt(scores['opd4'][benchmark])} | "
            f"{paper_base9:.2f}% | {fmt(scores['base9'][benchmark])} | "
            f"{paper_opd9:.2f}% | {fmt(scores['opd9'][benchmark])} |"
        )
    paper_macros = (70.68, 77.07, 73.37, 79.68)
    local_macros = {key: macro([scores[key][b] for b, _, _ in BENCHMARKS]) for key in MODELS}
    lines.append(
        f"| Macro | {paper_macros[0]:.2f}% | {fmt(local_macros['base4'])} | "
        f"{paper_macros[1]:.2f}% | {fmt(local_macros['opd4'])} | "
        f"{paper_macros[2]:.2f}% | {fmt(local_macros['base9'])} | "
        f"{paper_macros[3]:.2f}% | {fmt(local_macros['opd9'])} |"
    )

    lines += [
        "",
        "## All 10 Goal Benchmarks: 4B",
        "",
        "The Vision-OPD paper main table reports values for the first six benchmarks. "
        "Paper cells for MMStar, POPE, CV-Bench, and MMVP are marked `N/R`.",
        "",
        "| Benchmark | Paper Baseline 4B | Local Baseline 4B | Paper OPD-4B | Local OPD-4B |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for benchmark, label, _ in ALL_BENCHMARKS:
        paper = PAPER.get(benchmark)
        paper_base = "N/R" if paper is None else f"{paper[0]:.2f}%"
        paper_opd = "N/R" if paper is None else f"{paper[1]:.2f}%"
        lines.append(
            f"| {label} | {paper_base} | {fmt(read_score(root / 'results', selected_models['base4'], benchmark))} | "
            f"{paper_opd} | {fmt(read_score(root / 'results', selected_models['opd4'], benchmark))} |"
        )

    lines += [
        "",
        "## Baseline Deviation",
        "",
        "| Benchmark | Local 4B - Paper 4B | Local 9B - Paper 9B |",
        "| --- | ---: | ---: |",
    ]
    for benchmark, label, _ in BENCHMARKS:
        lines.append(
            f"| {label} | {delta(scores['base4'][benchmark], PAPER[benchmark][0])} | "
            f"{delta(scores['base9'][benchmark], PAPER[benchmark][2])} |"
        )

    lines += [
        "",
        "## Artifact Completeness",
        "",
        "| Model | Benchmark | Inference | Official judge | Score |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for key, model in selected_models.items():
        for benchmark, label, expected in BENCHMARKS:
            lines.append(
                f"| {model} | {label} | {answer_status(root, model, benchmark, expected)} | "
                f"{judge_status(root, model, benchmark, expected)} | {fmt(scores[key][benchmark])} |"
            )

    lines += [
        "",
        "## Historical 8x1 Ablation",
        "",
        "These scores are retained separately after a complete strict 96x8 result replaces",
        "the corresponding Local OPD column in the main table.",
        "",
        "| Benchmark | Historical Local OPD 4B | Historical Local OPD 9B |",
        "| --- | ---: | ---: |",
    ]
    for benchmark, label, _ in BENCHMARKS:
        lines.append(
            f"| {label} | {fmt(ablation_scores['opd4'][benchmark])} | "
            f"{fmt(ablation_scores['opd9'][benchmark])} |"
        )

    lines += [
        "",
        "## Locked Evaluation Configuration",
        "",
        "- Official source commit: `c2e345fcab10c806ba83e2ec6e1e246d73e7aba2`.",
        "- Seed: `42`; temperature: `0`; thinking: disabled.",
        "- Inference `max_tokens=32768`, `max_retries=3`, `parallel_workers=256`.",
        "- Judge: `openai/gpt-oss-120b`, using the unmodified official `judge_qwenlm.py`.",
        "- Benchmarks: 10 goal targets: the paper core six plus MMStar, POPE, CV-Bench, and MMVP.",
        "",
        "## Training Status",
        "",
        "The existing paper-explicit checkpoints used local `train_batch_size=8` and `rollout.n=1`; "
        "they are retained as an ablation and are not a complete reproduction of the released "
        "`batch=96`, `rollout.n=8` defaults. A new memory-adapted run must preserve the released "
        "global batch and rollout semantics before it can be described as a strict released-code reproduction.",
        f"The main Local OPD 4B column currently reads `{selected_models['opd4']}`; the 9B column reads "
        f"`{selected_models['opd9']}`.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
