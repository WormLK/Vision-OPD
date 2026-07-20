#!/usr/bin/env python3
"""Verify that the frozen official evaluation scripts match their source commit."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


FILES = (
    "eval/run_eval.sh",
    "eval/infer.py",
    "eval/judge_qwenlm.py",
    "eval/cal_acc.py",
    "eval/prepare_data.py",
)


def git_output(project: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=project, text=True, stderr=subprocess.STDOUT
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    args = parser.parse_args()

    project = args.project_root.resolve()
    official = args.official_root.resolve()
    commit_file = official / "source_commit.txt"
    if not commit_file.is_file():
        raise SystemExit(f"missing source commit marker: {commit_file}")
    commit = commit_file.read_text(encoding="utf-8").strip()
    source = official / "source"

    failures: list[str] = []
    for relative in FILES:
        local_path = source / relative
        if not local_path.is_file():
            failures.append(f"missing: {local_path}")
            continue
        expected = git_output(project, "rev-parse", f"{commit}:{relative}")
        actual = git_output(project, "hash-object", str(local_path))
        status = "PASS" if actual == expected else "FAIL"
        print(f"{status} {relative}: expected={expected} actual={actual}")
        if actual != expected:
            failures.append(f"blob mismatch: {relative}")

    if failures:
        print("Official evaluation source validation failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"PASS official evaluation source matches commit {commit}")


if __name__ == "__main__":
    main()
