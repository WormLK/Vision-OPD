#!/usr/bin/env python3
"""Audit every artifact required by the selected step-65 Vision-OPD-4B goal."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import yaml


BENCHMARKS = (
    "vstar",
    "zoombench",
    "hrbench-4k",
    "hrbench-8k",
    "mme-realworld",
    "mme-realworld-cn",
    "mmstar",
    "pope",
    "cv-bench",
    "mmvp",
)
BENCHMARK_LABELS = (
    "Vstar",
    "ZoomBench",
    "HR-Bench-4K",
    "HR-Bench-8K",
    "MME-RealWorld-EN",
    "MME-RealWorld-CN",
    "MMStar",
    "POPE-Test",
    "CV-Bench",
    "MMVP",
)
PAPER_BASELINE = (84.29, 47.69, 84.38, 80.13, 63.86, 63.70, 78.53, 88.28, 87.13, 76.67)
PAPER_OPD = (92.15, 59.76, 84.50, 80.38, 74.88, 70.76, 79.60, 89.14, 87.27, 79.67)
BASELINE_MODEL = "Qwen3.5-4B-baseline-official"
OPD_MODEL = "Vision-OPD-Qwen3.5-4B-released-b96-r8-official"
EXPERIMENT = "Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4"
INTERFACE_TOOLS = (
    "opencv_resize",
    "opencv_rotate",
    "opencv_translate",
    "opencv_flip",
    "opencv_crop",
    "opencv_zoom_in",
    "opencv_colorspace_gray",
    "opencv_colorspace_hsv",
    "opencv_colorspace_lab",
    "opencv_inrange_color",
    "opencv_blur",
    "opencv_denoise",
    "opencv_canny",
    "opencv_gradients",
    "opencv_threshold",
    "opencv_morphology",
    "opencv_contours",
    "opencv_contour_area",
    "opencv_arc_length",
    "opencv_approx_poly",
    "opencv_watershed",
    "opencv_grabcut",
    "opencv_floodfill",
    "opencv_connected_components_with_stats",
    "opencv_histogram",
    "opencv_convertscaleabs",
    "opencv_features",
    "opencv_hough_lines",
    "opencv_hough_circles",
    "opencv_template_match",
    "opencv_dft",
    "opencv_pyramid",
    "opencv_inpaint",
    "opencv_draw_line",
    "opencv_draw_circle",
)


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def require_score(path: Path, benchmark: str) -> float:
    if not path.is_file():
        raise RuntimeError(f"missing score: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if benchmark == "pope":
        values = re.findall(r"accuracy=(\d+(?:\.\d+)?)%", text)
    else:
        values = re.findall(r"(\d+(?:\.\d+)?)%", text)
    if not values:
        raise RuntimeError(f"unparseable score: {path}")
    return float(values[-1])


def report_score(value: float | None) -> str:
    return "N/R" if value is None else f"{value:.2f}%"


def read_vtc_overall(path: Path) -> float:
    if not path.is_file():
        raise RuntimeError(f"missing VTC score: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "Overall" not in rows[0]:
        raise RuntimeError(f"missing Overall in VTC score: {path}")
    try:
        return float(rows[0]["Overall"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid Overall in VTC score: {path}") from exc


def validate_vtc_config(
    path: Path,
    *,
    expected_tsv: Path,
    expected_results: Path,
    expected_tools: tuple[str, ...],
) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing VTC config: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    generate = config["llm"]["generate_cfg"]
    expected = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 1234,
        "max_tokens": 40960,
    }
    for key, value in expected.items():
        if generate.get(key) != value:
            raise RuntimeError(f"VTC config mismatch {path}: {key}={generate.get(key)!r}")
    if generate.get("use_raw_api") is not True or generate.get("max_retries") != 10:
        raise RuntimeError(f"VTC config mismatch {path}: raw API/retry settings")
    if config["llm"].get("model_type") != "qwenvl_oai":
        raise RuntimeError(f"VTC config mismatch {path}: model_type")
    if config["llm"].get("model") != OPD_MODEL:
        raise RuntimeError(f"VTC config mismatch {path}: model")
    if config["llm"].get("model_server") != "http://127.0.0.1:8000/v1":
        raise RuntimeError(f"VTC config mismatch {path}: model_server")
    input_config = config["input"]
    if Path(input_config.get("tsv_path", "")).resolve() != expected_tsv.resolve():
        raise RuntimeError(f"VTC config mismatch {path}: tsv_path")
    if input_config.get("start_idx") != 0 or input_config.get("end_idx") is not None:
        raise RuntimeError(f"VTC config mismatch {path}: evaluation range is not all rows")
    if Path(config["output"].get("results_dir", "")).resolve() != expected_results.resolve():
        raise RuntimeError(f"VTC config mismatch {path}: results_dir")
    if config["processing"].get("num_workers") != 30:
        raise RuntimeError(f"VTC config mismatch {path}: num_workers")
    tools = tuple(config["tools"]["enabled"])
    if tools != expected_tools:
        raise RuntimeError(f"VTC config mismatch {path}: tool list differs from the locked track")


def validate_vtc_runner(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing VTC runner: {path}")
    text = path.read_text(encoding="utf-8")
    required = (
        "--reasoning-parser qwen3",
        "--enable-auto-tool-choice --tool-call-parser qwen3_coder",
        "--tensor-parallel-size 1 --data-parallel-size 8",
        "--max-model-len 65536",
        "--default-chat-template-kwargs '{\"enable_thinking\":true}'",
        "scripts/smoke_qwen35_tool_parser.py --model-path",
    )
    missing = [value for value in required if value not in text]
    if missing:
        raise RuntimeError(f"VTC runner is missing locked Qwen3.5 flags: {missing}")
    if "--tool-call-parser hermes" in text:
        raise RuntimeError("VTC runner still uses the incompatible Hermes tool parser")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--vtc-root", type=Path, required=True)
    parser.add_argument(
        "--vtc-python",
        default="/data00/users/wanglikun/anaconda3/envs/vtc-opd-eval/bin/python",
    )
    args = parser.parse_args()

    project = args.project_root.resolve()
    vtc = args.vtc_root.resolve()
    official = project / "benchmark" / "official_reproduction_20260717"
    results = official / "results"
    checkpoint = project / "checkpoints" / EXPERIMENT
    latest = checkpoint / "latest_checkpointed_iteration.txt"
    if not latest.is_file() or latest.read_text(encoding="utf-8").strip() != "65":
        raise RuntimeError(f"selected step-65 checkpoint is not complete: {latest}")

    run(
        [
            sys.executable,
            str(project / "scripts" / "validate_official_eval_source.py"),
            "--project-root",
            str(project),
            "--official-root",
            str(official),
        ],
        project,
    )
    run(
        [
            sys.executable,
            str(project / "scripts" / "validate_goal_benchmark_contract.py"),
            "--project-root",
            str(project),
        ],
        project,
    )
    run(
        [
            sys.executable,
            str(project / "scripts" / "validate_strict_checkpoint.py"),
            str(checkpoint),
            "--step",
            "65",
        ],
        project,
    )
    merged = project / "merged_models" / EXPERIMENT
    run([sys.executable, str(project / "scripts" / "validate_merged_model.py"), str(merged)], project)

    model_scores: dict[str, list[float]] = {}
    for model in (BASELINE_MODEL, OPD_MODEL):
        run(
            [
                sys.executable,
                str(project / "scripts" / "validate_official_model_outputs.py"),
                "--project-root",
                str(project),
                "--model",
                model,
            ],
            project,
        )
        scores = [
            require_score(results / f"{model}_seed42_{benchmark}.txt", benchmark)
            for benchmark in BENCHMARKS
        ]
        model_scores[model] = scores
        print(f"PASS 10 official scores {model}: unweighted_mean={sum(scores) / len(scores):.2f}")
    code_config = vtc / "eval/eval_config/vision_opd_qwen35_4b_code.yaml"
    interface_config = vtc / "eval/eval_config/vision_opd_qwen35_4b_interface.yaml"
    vtc_data = vtc / "data/vtc_bench"
    validate_vtc_config(
        code_config,
        expected_tsv=vtc_data / "VTC-Bench.absolute.tsv",
        expected_results=vtc / "runs/vtc_vision_opd_4b_step65_code",
        expected_tools=("code_interpreter",),
    )
    validate_vtc_config(
        interface_config,
        expected_tsv=vtc_data / "VTC-Bench_GTToolChain.absolute.tsv",
        expected_results=vtc / "runs/vtc_vision_opd_4b_step65_interface",
        expected_tools=INTERFACE_TOOLS,
    )
    validate_vtc_runner(vtc / "scripts/run_vision_opd_4b_vtc_bench.sh")
    eval_root = vtc / "eval/VLMEvalKit/outputs/VTC_Bench"
    tracks = (
        ("code", "Qwen-Agent-Code-RawAPI-Instruct", "runs/vtc_vision_opd_4b_step65_code"),
        ("interface", "Qwen-Agent-Interface-RawAPI-Instruct", "runs/vtc_vision_opd_4b_step65_interface"),
    )
    vtc_scores: dict[str, float] = {}
    for label, prefix, result_dir in tracks:
        score = eval_root / f"{prefix}-{OPD_MODEL}" / f"{OPD_MODEL}_VTC_Bench_score.csv"
        run(
            [
                args.vtc_python,
                str(vtc / "scripts/validate_vision_opd_vtc_track.py"),
                "--results-dir",
                str(vtc / result_dir),
                "--model",
                OPD_MODEL,
                "--score-file",
                str(score),
            ],
            vtc,
        )
        vtc_scores[label] = read_vtc_overall(score)
        print(f"PASS VTC {label} track")

    report = project / "docs/vision_opd_4b_vtc_reproduction.md"
    if not report.is_file():
        raise RuntimeError(f"missing final report: {report}")
    report_text = report.read_text(encoding="utf-8")
    expected_report_rows = zip(
        BENCHMARK_LABELS,
        PAPER_BASELINE,
        model_scores[BASELINE_MODEL],
        PAPER_OPD,
        model_scores[OPD_MODEL],
        strict=True,
    )
    for label, paper_base, local_base, paper_opd, local_opd in expected_report_rows:
        rows = [line for line in report_text.splitlines() if line.startswith(f"| {label} |")]
        if not rows:
            raise RuntimeError(f"missing 10-benchmark report row: {label}")
        cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
        expected_cells = [
            label,
            report_score(paper_base),
            report_score(local_base),
            report_score(paper_opd),
            report_score(local_opd),
        ]
        if cells != expected_cells:
            raise RuntimeError(f"incomplete four-column report row: {rows[0]}")

    holdout_indices = (9, 8, 6, 7)
    for index in holdout_indices:
        label = BENCHMARK_LABELS[index]
        rows = [line for line in report_text.splitlines() if line.startswith(f"| {label} |")]
        if len(rows) != 2:
            raise RuntimeError(f"missing or duplicate Table 2 hold-out row: {label}")
        expected_cells = [
            label,
            report_score(PAPER_BASELINE[index]),
            report_score(PAPER_OPD[index]),
            f"{PAPER_OPD[index] - PAPER_BASELINE[index]:+.2f} pp",  # type: ignore[operator]
            report_score(model_scores[BASELINE_MODEL][index]),
            report_score(model_scores[OPD_MODEL][index]),
        ]
        cells = [cell.strip() for cell in rows[1].strip("|").split("|")]
        if cells != expected_cells:
            raise RuntimeError(f"incorrect Table 2 hold-out row: {rows[1]}")
    core_macro_rows = [
        line for line in report_text.splitlines() if line.startswith("| Core-six Macro |")
    ]
    expected_macro = [
        "Core-six Macro",
        "70.68%",
        f"{sum(model_scores[BASELINE_MODEL][:6]) / 6:.2f}%",
        "77.07%",
        f"{sum(model_scores[OPD_MODEL][:6]) / 6:.2f}%",
    ]
    if len(core_macro_rows) != 1:
        raise RuntimeError("missing or duplicate Core-six Macro report row")
    macro_cells = [cell.strip() for cell in core_macro_rows[0].strip("|").split("|")]
    if macro_cells != expected_macro:
        raise RuntimeError(f"incorrect Core-six Macro report row: {core_macro_rows[0]}")

    expected_vtc_rows = {
        "Code-driven": vtc_scores["code"],
        "Interface-driven": vtc_scores["interface"],
    }
    for label, overall in expected_vtc_rows.items():
        rows = [line for line in report_text.splitlines() if line.startswith(f"| {label} |")]
        expected_cells = [label, "680/680", f"{overall:.2f}%"]
        if len(rows) != 1:
            raise RuntimeError(f"missing or duplicate VTC report row: {label}")
        cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
        if cells != expected_cells:
            raise RuntimeError(f"stale VTC report row: {rows[0]}")
    print("PASS selected step-65 Vision-OPD-4B artifact completion audit")


if __name__ == "__main__":
    main()
