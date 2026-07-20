#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


BENCHMARKS = (
    ("vstar", 92.15, 94.76),
    ("zoombench", 59.76, 65.80),
    ("hrbench-4k", 84.50, 88.13),
    ("hrbench-8k", 80.38, 85.50),
    ("mme-realworld", 74.88, 73.40),
    ("mme-realworld-cn", 70.76, 70.46),
)
PAPER_MACRO = {"4b": 77.07, "9b": 79.68}


def read_score(path: Path) -> float:
    values = re.findall(r"(\d+(?:\.\d+)?)%", path.read_text(encoding="utf-8", errors="replace"))
    if not values:
        raise ValueError(f"score not found: {path}")
    return float(values[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backbone", choices=("4b", "9b"), required=True)
    parser.add_argument("--max-benchmark-delta", type=float, default=3.0)
    parser.add_argument("--max-macro-delta", type=float, default=2.0)
    args = parser.parse_args()

    results = args.project_root.resolve() / "benchmark" / "official_reproduction_20260717" / "results"
    paper_index = 1 if args.backbone == "4b" else 2
    local_scores = []
    errors = []
    for benchmark, paper4, paper9 in BENCHMARKS:
        paper = (paper4, paper9)[paper_index - 1]
        path = results / f"{args.model}_seed42_{benchmark}.txt"
        if not path.is_file():
            errors.append(f"missing score: {path}")
            continue
        local = read_score(path)
        delta = local - paper
        local_scores.append(local)
        print(f"{benchmark}: local={local:.2f} paper={paper:.2f} delta={delta:+.2f}")
        if abs(delta) > args.max_benchmark_delta:
            errors.append(f"benchmark deviation exceeds gate: {benchmark} {delta:+.2f} pp")

    if len(local_scores) == len(BENCHMARKS):
        local_macro = sum(local_scores) / len(local_scores)
        paper_macro = PAPER_MACRO[args.backbone]
        macro_delta = local_macro - paper_macro
        print(f"macro: local={local_macro:.2f} paper={paper_macro:.2f} delta={macro_delta:+.2f}")
        if abs(macro_delta) > args.max_macro_delta:
            errors.append(f"macro deviation exceeds gate: {macro_delta:+.2f} pp")

    if errors:
        print("FAILED OPD alignment gate:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"PASS: {args.model} satisfies the paper OPD-{args.backbone.upper()} alignment gate")


if __name__ == "__main__":
    main()
