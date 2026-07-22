#!/usr/bin/env python3
"""Regression test for the strict VTC-Bench Base result validator."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    validator = project / "integrations/vtc_bench/scripts/validate_vtc_base_track.py"
    with tempfile.TemporaryDirectory(prefix="vtc-base-validator-") as tmp:
        root = Path(tmp)
        model = "test-base-model"
        model_dir = root / "results" / model
        model_dir.mkdir(parents=True)
        result = {
            "status": "success",
            "row_index": 0,
            "item_id": "item_0",
            "agent_answer": "A",
            "conversation": [
                {"role": "system", "content": "strong prompt"},
                {"role": "user", "content": [{"image": "image.png"}, {"text": "question"}]},
                {"role": "assistant", "content": "<answer>A</answer>"},
            ],
        }
        result_file = model_dir / "results_20260722_000000.jsonl"
        result_file.write_text(json.dumps(result) + "\n", encoding="utf-8")
        raw_file = model_dir / "response_list_item_0.json"
        valid_raw = {
            "response_list": [
                [{"role": "assistant", "content": "<answer>A</answer>"}]
            ]
        }
        write_json(raw_file, valid_raw)
        score_file = root / "score.csv"
        score_file.write_text("Overall,attention\n100.0,100.0\n", encoding="utf-8")
        command = [
            sys.executable,
            str(validator),
            "--results-dir",
            str(root / "results"),
            "--model",
            model,
            "--score-file",
            str(score_file),
            "--expected",
            "1",
        ]
        subprocess.run(command, check=True)

        invalid_raw = {
            "response_list": [
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": "crop", "arguments": "{}"}}],
                    }
                ]
            ]
        }
        write_json(raw_file, invalid_raw)
        rejected = subprocess.run(command, text=True, capture_output=True)
        if rejected.returncode == 0 or "raw Base protocol errors" not in rejected.stdout:
            raise RuntimeError(
                "validator did not reject a raw Base response containing tool_calls\n"
                + rejected.stdout
                + rejected.stderr
            )
    print("PASS VTC Base validator regression: valid accepted, tool_calls rejected")


if __name__ == "__main__":
    main()
