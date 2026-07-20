#!/usr/bin/env python3
"""Verify that every strict 4B component uses the same 10-benchmark contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    contract_path = (
        project / "benchmark/official_reproduction_20260717/goal_4b_benchmarks.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    slugs = [item["slug"] for item in contract]
    labels = [item["label"] for item in contract]
    counts = {item["slug"]: item["count"] for item in contract}
    manifests = {item["slug"]: item["manifest"] for item in contract}
    if len(slugs) != 10 or len(set(slugs)) != 10 or sum(counts.values()) != 45145:
        raise RuntimeError("invalid 10-benchmark contract cardinality or total")

    validator = load_module(project / "scripts/validate_official_model_outputs.py", "validator")
    validator_slugs = list(validator.BENCHMARKS)
    validator_counts = {key: value[1] for key, value in validator.BENCHMARKS.items()}
    validator_manifests = {key: value[0] for key, value in validator.BENCHMARKS.items()}
    if validator_slugs != slugs or validator_counts != counts or validator_manifests != manifests:
        raise RuntimeError("validate_official_model_outputs.py differs from contract")

    audit = load_module(project / "scripts/audit_4b_goal_completion.py", "audit")
    if list(audit.BENCHMARKS) != slugs or list(audit.BENCHMARK_LABELS) != labels:
        raise RuntimeError("audit_4b_goal_completion.py differs from contract")

    report = load_module(project / "scripts/summarize_4b_vtc_reproduction.py", "report")
    report_slugs = [item[0] for item in report.BENCHMARKS]
    report_labels = [item[1] for item in report.BENCHMARKS]
    if report_slugs != slugs or report_labels != labels or report.EXPECTED_COUNTS != counts:
        raise RuntimeError("summarize_4b_vtc_reproduction.py differs from contract")

    shell = (project / "scripts/evaluate_official_single_model.sh").read_text(encoding="utf-8")
    match = re.search(r'^BENCHMARKS="([^"]+)"$', shell, re.MULTILINE)
    if match is None or match.group(1).split(",") != slugs:
        raise RuntimeError("evaluate_official_single_model.sh differs from contract")

    print("PASS 4B goal benchmark contract: 10 benchmarks, 45,145 rows per model")


if __name__ == "__main__":
    main()
