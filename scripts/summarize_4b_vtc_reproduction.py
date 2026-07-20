#!/usr/bin/env python3
"""Build the final four-column Vision-OPD-4B and VTC-Bench report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path


BENCHMARKS = (
    ("vstar", "Vstar", 84.29, 92.15),
    ("zoombench", "ZoomBench", 47.69, 59.76),
    ("hrbench-4k", "HR-Bench-4K", 84.38, 84.50),
    ("hrbench-8k", "HR-Bench-8K", 80.13, 80.38),
    ("mme-realworld", "MME-RealWorld-EN", 63.86, 74.88),
    ("mme-realworld-cn", "MME-RealWorld-CN", 63.70, 70.76),
    ("mmstar", "MMStar", 78.53, 79.60),
    ("pope", "POPE-Test", 88.28, 89.14),
    ("cv-bench", "CV-Bench", 87.13, 87.27),
    ("mmvp", "MMVP", 76.67, 79.67),
)
BASELINE_MODEL = "Qwen3.5-4B-baseline-official"
OPD_MODEL = "Vision-OPD-Qwen3.5-4B-released-b96-r8-official"


def read_official_score(results: Path, model: str, benchmark: str) -> float | None:
    path = results / f"{model}_seed42_{benchmark}.txt"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    if benchmark == "pope":
        values = re.findall(r"accuracy=(\d+(?:\.\d+)?)%", text)
    else:
        values = re.findall(r"(\d+(?:\.\d+)?)%", text)
    return float(values[-1]) if values else None


def format_score(value: float | None) -> str:
    return "pending" if value is None else f"{value:.2f}%"


def format_paper(value: float | None) -> str:
    return "N/R" if value is None else f"{value:.2f}%"


def macro(values: list[float | None]) -> float | None:
    return None if any(value is None for value in values) else sum(values) / len(values)  # type: ignore[arg-type]


def sha256(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_vtc_score(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) >= 2 and len(rows[0]) == len(rows[1]):
        parsed = {}
        for key, value in zip(rows[0], rows[1], strict=True):
            try:
                parsed[key.strip()] = float(value)
            except ValueError:
                continue
        if "Overall" in parsed:
            return parsed
    result: dict[str, float] = {}
    for row in rows:
        for index, cell in enumerate(row):
            key = cell.strip()
            if not key:
                continue
            for candidate in row[index + 1 :]:
                try:
                    result[key] = float(candidate)
                    break
                except ValueError:
                    continue
    return result


def result_jsonl_status(root: Path, model: str) -> str:
    model_dir = root / model
    files = sorted(model_dir.glob("results_*.jsonl")) if model_dir.is_dir() else []
    if not files:
        return "0/680"
    rows = 0
    errors = 0
    with files[-1].open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            try:
                item = json.loads(line)
                if item.get("error") or not str(item.get("agent_answer", "")).strip():
                    errors += 1
            except json.JSONDecodeError:
                errors += 1
    suffix = "" if errors == 0 else f", errors={errors}"
    return f"{rows}/680{suffix}"


def answer_length_stats(path: Path) -> tuple[int, int, int, int]:
    lengths = []
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    answer = json.loads(line).get("model_answer", "")
                except json.JSONDecodeError:
                    continue
                lengths.append(len(answer) if isinstance(answer, str) else 0)
    if not lengths:
        return 0, 0, 0, 0
    lengths.sort()
    p95 = lengths[int(0.95 * (len(lengths) - 1))]
    return len(lengths), p95, lengths[-1], sum(length > 10000 for length in lengths)


def interim_mme_rule_stats(answer_path: Path, judge_source: Path) -> dict[str, tuple[int, int]]:
    """Apply only the official judge's deterministic pre-LLM rules to a live snapshot."""
    groups = {"all": [0, 0], "<10k": [0, 0], ">=10k": [0, 0], ">=50k": [0, 0]}
    if not answer_path.is_file() or not judge_source.is_file():
        return {key: (0, 0) for key in groups}

    spec = importlib.util.spec_from_file_location("vision_opd_official_judge", judge_source)
    if spec is None or spec.loader is None:
        return {key: (0, 0) for key in groups}
    judge_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge_module)
    try:
        from mathruler.grader import grade_answer
    except ImportError:
        grade_answer = None

    with answer_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = str(item.get("model_answer", "") or "")
            extracted = judge_module.extract_answer(raw)
            ground_truth = str(item.get("response", "") or "")
            correct = False
            if grade_answer is not None:
                try:
                    correct = bool(grade_answer(ground_truth, extracted))
                except Exception:
                    correct = False
            if not correct:
                try:
                    correct = bool(judge_module.first_letter_match(ground_truth, extracted))
                except Exception:
                    correct = False
            predicates = {
                "all": True,
                "<10k": len(raw) < 10_000,
                ">=10k": len(raw) >= 10_000,
                ">=50k": len(raw) >= 50_000,
            }
            for name, included in predicates.items():
                if included:
                    groups[name][1] += 1
                    groups[name][0] += int(correct)
    return {key: (values[0], values[1]) for key, values in groups.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--vtc-root",
        type=Path,
        default=Path("/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    project = args.project_root.resolve()
    vtc = args.vtc_root.resolve()
    output = args.output or project / "docs" / "vision_opd_4b_vtc_reproduction.md"
    official = project / "benchmark" / "official_reproduction_20260717"
    results = official / "results"

    baseline_scores = [read_official_score(results, BASELINE_MODEL, item[0]) for item in BENCHMARKS]
    opd_scores = [read_official_score(results, OPD_MODEL, item[0]) for item in BENCHMARKS]
    model = OPD_MODEL
    code_folder = f"Qwen-Agent-Code-RawAPI-Instruct-{model}"
    interface_folder = f"Qwen-Agent-Interface-RawAPI-Instruct-{model}"
    eval_root = vtc / "eval" / "VLMEvalKit" / "outputs" / "VTC_Bench"
    code_score_path = eval_root / code_folder / f"{model}_VTC_Bench_score.csv"
    interface_score_path = eval_root / interface_folder / f"{model}_VTC_Bench_score.csv"
    code_scores = read_vtc_score(code_score_path)
    interface_scores = read_vtc_score(interface_score_path)
    code_config = vtc / "eval" / "eval_config" / "vision_opd_qwen35_4b_code.yaml"
    interface_config = vtc / "eval" / "eval_config" / "vision_opd_qwen35_4b_interface.yaml"
    vtc_gt = vtc / "data" / "vtc_bench" / "VTC-Bench_GTToolChain.tsv"
    benchmark_contract = official / "goal_4b_benchmarks.json"
    eval_provenance = official / "provenance" / "selected_4b_eval_config.json"
    source_commit_path = official / "source_commit.txt"
    source_commit = (
        source_commit_path.read_text(encoding="utf-8").strip()
        if source_commit_path.is_file()
        else "missing"
    )

    lines = [
        "# Vision-OPD-4B Official and VTC-Bench Reproduction",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Official Benchmark Alignment",
        "",
        "| Benchmark | Paper Baseline 4B | Local Baseline 4B | Paper OPD-4B | Local OPD-4B |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for index, (_, label, paper_base, paper_opd) in enumerate(BENCHMARKS):
        lines.append(
            f"| {label} | {format_paper(paper_base)} | {format_score(baseline_scores[index])} | "
            f"{format_paper(paper_opd)} | {format_score(opd_scores[index])} |"
        )
    core_baseline = baseline_scores[:6]
    core_opd = opd_scores[:6]
    lines.append(
        f"| Core-six Macro | 70.68% | {format_score(macro(core_baseline))} | "
        f"77.07% | {format_score(macro(core_opd))} |"
    )
    lines.append(
        f"| Local 10-benchmark unweighted mean | N/R | {format_score(macro(baseline_scores))} | "
        f"N/R | {format_score(macro(opd_scores))} |"
    )

    lines += [
        "",
        "### Paper Table 2 Hold-out Tasks",
        "",
        "The paper defines these four datasets as hold-out tasks that are unseen during "
        "Vision-OPD training. Values below are transcribed from Table 2 of "
        "[arXiv:2605.18740](https://arxiv.org/pdf/2605.18740).",
        "",
        "| Hold-out benchmark | Paper Vanilla 4B | Paper OPD-4B | Paper gain | "
        "Local Baseline 4B | Local OPD-4B |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index in (9, 8, 6, 7):
        _, label, paper_base, paper_opd = BENCHMARKS[index]
        assert paper_base is not None and paper_opd is not None
        lines.append(
            f"| {label} | {paper_base:.2f}% | {paper_opd:.2f}% | "
            f"{paper_opd - paper_base:+.2f} pp | {format_score(baseline_scores[index])} | "
            f"{format_score(opd_scores[index])} |"
        )

    local_baseline_macro = macro(core_baseline)
    local_opd_macro = macro(core_opd)
    lines += ["", "## Alignment Verdict", ""]
    if local_baseline_macro is None or local_opd_macro is None:
        lines.append(
            "The official core-six evaluation is still incomplete. The final alignment verdict "
            "will be generated after all inference and GPT-OSS judging artifacts validate."
        )
    else:
        paper_opd_core = [item[3] for item in BENCHMARKS[:6]]
        deviations = [
            (BENCHMARKS[index][1], local_opd - paper_opd)
            for index, (local_opd, paper_opd) in enumerate(
                zip(core_opd, paper_opd_core, strict=True)
            )
        ]
        outside_gate = [f"{name} ({delta:+.2f} pp)" for name, delta in deviations if abs(delta) > 3.0]
        macro_delta = local_opd_macro - 77.07
        aligned = abs(macro_delta) <= 2.0 and not outside_gate
        lines.append(
            f"Result: **{'PASS' if aligned else 'NOT ALIGNED'}** under the documented local "
            "gate of +/-2.0 pp for the core-six macro and +/-3.0 pp per benchmark."
        )
        lines.append(
            f"Local baseline macro is {local_baseline_macro:.2f}% ({local_baseline_macro - 70.68:+.2f} "
            f"pp versus paper baseline); local OPD macro is {local_opd_macro:.2f}% "
            f"({macro_delta:+.2f} pp versus paper OPD and "
            f"{local_opd_macro - local_baseline_macro:+.2f} pp versus the local baseline)."
        )
        lines.append(
            "Per-benchmark deviations outside the gate: "
            + (", ".join(outside_gate) if outside_gate else "none")
            + "."
        )

    lines += [
        "",
        "## Inference Behavior",
        "",
        "Character length is reported because the selected checkpoint produced unusually long "
        "non-thinking answers on some MME-RealWorld samples under the official 32,768-token cap.",
        "",
        "| Model | Benchmark | Rows | P95 characters | Max characters | Answers >10k chars |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model_name, model_label in (
        (BASELINE_MODEL, "Local baseline 4B"),
        (OPD_MODEL, "Local OPD-4B"),
    ):
        for benchmark, label in (
            ("mme-realworld", "MME-RealWorld-EN"),
            ("mme-realworld-cn", "MME-RealWorld-CN"),
        ):
            stats = answer_length_stats(
                official / "model_answer" / benchmark / f"{model_name}_seed42_answer.jsonl"
            )
            lines.append(
                f"| {model_label} | {label} | {stats[0]} | {stats[1]} | {stats[2]} | {stats[3]} |"
            )

    interim_answer = (
        official
        / "model_answer"
        / "mme-realworld"
        / f"{OPD_MODEL}_seed42_answer.jsonl"
    )
    interim_stats = interim_mme_rule_stats(
        interim_answer, official / "source" / "eval" / "judge_qwenlm.py"
    )
    lines += [
        "",
        "### Interim MME-RealWorld-EN Snapshot",
        "",
        "This is a moving partial snapshot, not the final benchmark score. `Rule-direct correct` "
        "uses only the deterministic MathRuler/first-option stages of the official judge. "
        "Unresolved rows still require GPT-OSS-120B, so the percentage is a conservative lower "
        "bound and the incomplete prefix need not be representative of the full dataset.",
        "",
        "| Response-length group | Snapshot rows | Rule-direct correct | Lower bound |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group in ("all", "<10k", ">=10k", ">=50k"):
        correct, total = interim_stats[group]
        percentage = 100.0 * correct / total if total else 0.0
        lines.append(f"| {group} characters | {total} | {correct} | {percentage:.2f}% |")
    lines += [
        "",
        "The long-response groups have a substantially lower rule-direct success rate. Responses "
        "that exceed the GPT-OSS judge's 8,192-token context can also fail judge requests and are "
        "then conservatively recorded as `No` after the official three retries. This affects final "
        "accuracy in addition to increasing inference latency.",
    ]

    lines += [
        "",
        "## Experiment Selection",
        "",
        "The final local column uses the user-selected one-epoch `released-b96-r8-gradaccum-sp4` "
        "checkpoint at global step 65. It preserves 96 prompts/update, 8 rollouts/prompt, 65 "
        "updates, and 6,240 prompts. Its rollout tensor parallelism is TP4 rather than the "
        "released TP1 setting; this is a documented runtime deviation and can change sampled "
        "trajectories even though the global optimization batch is preserved.",
    ]

    categories = sorted((set(code_scores) | set(interface_scores)) - {"Overall"})
    lines += [
        "",
        "## VTC-Bench",
        "",
        "| Track | Inference | Overall |",
        "| --- | ---: | ---: |",
        f"| Code-driven | {result_jsonl_status(vtc / 'runs' / 'vtc_vision_opd_4b_step65_code', model)} | "
        f"{format_score(code_scores.get('Overall'))} |",
        f"| Interface-driven | {result_jsonl_status(vtc / 'runs' / 'vtc_vision_opd_4b_step65_interface', model)} | "
        f"{format_score(interface_scores.get('Overall'))} |",
        "",
        "| Category | Code-driven | Interface-driven |",
        "| --- | ---: | ---: |",
    ]
    for category in categories:
        lines.append(
            f"| {category} | {format_score(code_scores.get(category))} | "
            f"{format_score(interface_scores.get(category))} |"
        )

    lines += [
        "",
        "## Locked Configuration",
        "",
        f"- Frozen official eval source commit: `{source_commit}`.",
        "- Official source gate: the Git blobs for `run_eval.sh`, `infer.py`, "
        "`judge_qwenlm.py`, `cal_acc.py`, and `prepare_data.py` must match the frozen commit.",
        "- Training: 6,240 prompts, one epoch, 65 global updates, 96 prompts/update, 8 rollouts/prompt.",
        "- Training objective: VOPD Top-K JSD, alpha 0.5, Top-K 100, EMA teacher rate 0.05.",
        "- Selected training topology: actor SP4, 2,304 tokens/rank (9,216 per SP group), "
        "rollout TP4, rollout GPU utilization 0.30, max_num_seqs 64, layered summon enabled.",
        "- Released topology deviation: rollout TP4 replaces the released TP1 default; the "
        "global batch and number of trajectories are unchanged, but sampled trajectories need "
        "not be bitwise identical.",
        "- Official inference: pristine `eval/run_eval.sh`, seed 42, temperature 0, thinking disabled, max tokens 32768, 256 workers.",
        "- Official scope: 10 benchmark names (core six plus MMStar, POPE, CV-Bench, and MMVP), "
        "45,145 inference rows per model; "
        "paper alignment gate remains the six benchmarks reported in the Vision-OPD main table.",
        f"- 10-benchmark contract SHA-256: `{sha256(benchmark_contract)}`.",
        f"- Selected evaluation provenance SHA-256: `{sha256(eval_provenance)}`.",
        "- Official judge: `openai/gpt-oss-120b` with the pristine `judge_qwenlm.py`.",
        "- VTC generation: temperature 0.6, top-p 0.95, top-k 20, seed 1234, max tokens 40960, 30 workers.",
        "- VTC serving: vLLM DP8/TP1, context 65536, thinking enabled, Qwen3 reasoning parser, and Qwen3-Coder native tool-call parser.",
        "- VTC code track: `code_interpreter`; interface track: all 35 OpenCV tools.",
        f"- VTC code YAML SHA-256: `{sha256(code_config)}`.",
        f"- VTC interface YAML SHA-256: `{sha256(interface_config)}`.",
        f"- Canonical repaired VTC GT SHA-256: `{sha256(vtc_gt)}`.",
        "",
        "## Artifacts",
        "",
        f"- Official results: `{results}`",
        f"- 10-benchmark contract: `{benchmark_contract}`",
        f"- Selected evaluation provenance: `{eval_provenance}`",
        f"- Selected checkpoint: `{project / 'checkpoints' / 'Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4' / 'global_step_65'}`",
        f"- Selected merged model: `{project / 'merged_models' / 'Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4'}`",
        f"- Final goal audit log: `{project / 'logs' / 'vision_opd_4b_goal_completion_audit.log'}`",
        f"- Final goal audit marker: `{project / 'outputs' / 'vision_opd_4b_goal_audit_complete'}`",
        f"- VTC code score: `{code_score_path}`",
        f"- VTC interface score: `{interface_score_path}`",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
