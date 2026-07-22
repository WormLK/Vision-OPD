#!/usr/bin/env python3
"""Audit all three serial Qwen3.5 VTC-Bench Base evaluations and the report."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


TRACKS = (
    (
        "Local OPD-4B Base",
        "vision_opd_qwen35_4b_base",
        "Vision-OPD-Qwen3.5-4B-released-b96-r8-base",
        "vtc_vision_opd_4b_step65_base",
    ),
    (
        "Local Qwen3.5-4B Base",
        "qwen35_4b_base",
        "Qwen3.5-4B-base-vtc",
        "vtc_qwen35_4b_base",
    ),
    (
        "Local Qwen3.5-9B Base",
        "qwen35_9b_base",
        "Qwen3.5-9B-base-vtc",
        "vtc_qwen35_9b_base",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--vtc-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    vtc = args.vtc_root.resolve()
    validator = project / "integrations/vtc_bench/scripts/validate_vtc_base_track.py"
    report = project / "docs/vision_opd_4b_vtc_reproduction.md"
    report_text = report.read_text(encoding="utf-8")
    marker_times = []

    for label, marker_name, model, run_name in TRACKS:
        marker = vtc / "runs" / f"{marker_name}_complete"
        if not marker.is_file():
            raise RuntimeError(f"missing completion marker: {marker}")
        marker_times.append(marker.stat().st_mtime_ns)
        score = (
            vtc
            / "eval/VLMEvalKit/outputs/VTC_Bench"
            / f"Qwen-Agent-Base-RawAPI-Instruct-{model}"
            / f"{model}_VTC_Bench_score.csv"
        )
        subprocess.run(
            [
                sys.executable,
                str(validator),
                "--results-dir",
                str(vtc / "runs" / run_name),
                "--model",
                model,
                "--score-file",
                str(score),
            ],
            check=True,
        )
        with score.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        overall = float(rows[0]["Overall"])
        expected_row = rf"\| {re.escape(label)} \| 680/680 \| {overall:.2f}% \|"
        if not re.search(expected_row, report_text):
            raise RuntimeError(f"final report is missing the validated {label} row")
        print(f"PASS report row: {label} overall={overall:.2f}%")

    if marker_times != sorted(marker_times) or len(set(marker_times)) != len(marker_times):
        raise RuntimeError("Base completion markers do not prove the required serial model order")
    for required in (
        "enable_thinking=true",
        "functions=[]",
        "Exact Strong System Prompt",
        "Exact User Prompt template (without GT Toolchains)",
        "Model-native Qwen3.5 template",
    ):
        if required not in report_text:
            raise RuntimeError(f"final report is missing protocol record: {required}")
    print("PASS Qwen3.5 VTC Base sequence: OPD-4B -> baseline 4B -> baseline 9B")


if __name__ == "__main__":
    main()
