#!/usr/bin/env python3
"""Score completed VTC-Bench rows with the official heuristic evaluator.

This deliberately scores only resume-valid rows from the latest JSONL snapshot.
It never writes into the live VTC evaluator output directories and is therefore
safe to run while inference is still producing new rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


INVALID_ANSWER_MARKERS = (
    "unable",
    "your final answer here",
    "cannot",
    "indiscernible",
    "insufficient",
    "unreadable",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def resume_valid(row: dict) -> bool:
    answer = str(row.get("agent_answer") or "").strip().lower()
    return (
        row.get("status") == "success"
        and bool(answer)
        and not any(marker in answer for marker in INVALID_ANSWER_MARKERS)
    )


def load_snapshot(path: Path) -> tuple[list[dict], str, int]:
    raw = path.read_bytes()
    rows: list[dict] = []
    malformed = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows, sha256_bytes(raw), malformed


def latest_result_file(run_dir: Path, model: str) -> Path:
    files = sorted((run_dir / model).glob("results_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no results JSONL under {run_dir / model}")
    return files[-1]


def score_track(
    *,
    label: str,
    run_dir: Path,
    model: str,
    tsv_path: Path,
    dataset_cls: type,
) -> dict:
    result_path = latest_result_file(run_dir, model)
    rows, digest, malformed = load_snapshot(result_path)
    valid_rows = [row for row in rows if resume_valid(row)]
    predictions: dict[str, str] = {}
    duplicate_ids = 0
    for row in valid_rows:
        item_id = str(row.get("item_id") or "").strip()
        if not item_id:
            continue
        if item_id in predictions:
            duplicate_ids += 1
        predictions[item_id] = str(row.get("agent_answer") or "")

    source = pd.read_csv(tsv_path, sep="\t")
    source["id"] = source["id"].astype(str)
    subset = source[source["id"].isin(predictions)].copy()
    subset["prediction"] = subset["id"].map(predictions)
    subset = subset.sort_values("index").reset_index(drop=True)

    evaluator = dataset_cls.__new__(dataset_cls)
    scores = [
        float(evaluator._evaluate_single_item(row))
        for _, row in subset.iterrows()
    ]
    score_array = pd.Series(scores, dtype="float64")
    categories: dict[str, dict[str, float | int]] = {}
    for category, group in subset.groupby("category", sort=True):
        indices = group.index
        category_scores = score_array.loc[indices]
        categories[str(category)] = {
            "rows": len(category_scores),
            "correct": int((category_scores >= 0.5).sum()),
            "percent": float(category_scores.mean() * 100.0),
        }

    expected_overall = float(score_array.mean() * 100.0) if len(score_array) else None
    with tempfile.TemporaryDirectory(prefix=f"vtc_partial_{label}_") as tmp:
        eval_path = Path(tmp) / "partial.csv"
        subset.to_csv(eval_path, index=False)
        public_result = evaluator.evaluate(str(eval_path), model="exact_matching", nproc=1)
        public_overall = float(public_result.iloc[0]["Overall"])
    if expected_overall is None or abs(public_overall - expected_overall) > 1e-9:
        raise RuntimeError(
            f"public evaluator mismatch for {label}: {public_overall} != {expected_overall}"
        )

    return {
        "track": label,
        "result_file": str(result_path),
        "result_sha256": digest,
        "raw_rows": len(rows),
        "malformed_json_lines": malformed,
        "resume_valid_rows": len(valid_rows),
        "matched_rows_scored": len(subset),
        "duplicate_item_ids": duplicate_ids,
        "source_rows": len(source),
        "overall_percent": expected_overall,
        "correct_rows": int((score_array >= 0.5).sum()),
        "category_metrics": categories,
        "public_evaluator_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--vtc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Vision-OPD-Qwen3.5-4B-released-b96-r8-official")
    args = parser.parse_args()

    vtc_root = args.vtc_root.resolve()
    sys.path.insert(0, str(vtc_root / "eval"))
    sys.path.insert(0, str(vtc_root / "eval/eval/VLMEvalKit"))
    from vlmeval.dataset.image_vqa import VTCBenchDataset

    data_root = vtc_root / "data/vtc_bench"
    tracks = {
        "code-driven": (
            vtc_root / "runs/vtc_vision_opd_4b_step65_code",
            data_root / "VTC-Bench.absolute.tsv",
        ),
        "interface-driven": (
            vtc_root / "runs/vtc_vision_opd_4b_step65_interface",
            data_root / "VTC-Bench_GTToolChain.absolute.tsv",
        ),
    }
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator": "vlmeval.dataset.image_vqa.VTCBenchDataset.evaluate",
        "per_item_rule": "VTCBenchDataset._evaluate_single_item",
        "method": "official heuristic rule evaluation",
        "partial": True,
        "denominator_note": "Only resume-valid rows in the latest JSONL snapshot are scored; this is not a full 680-row result.",
        "tracks": {},
    }
    for label, (run_dir, tsv_path) in tracks.items():
        report["tracks"][label] = score_track(
            label=label,
            run_dir=run_dir,
            model=args.model,
            tsv_path=tsv_path,
            dataset_cls=VTCBenchDataset,
        )
        result = report["tracks"][label]
        print(
            f"{label}: {result['matched_rows_scored']}/{result['source_rows']} rows, "
            f"correct={result['correct_rows']}, overall={result['overall_percent']:.2f}%"
        )

    total_scored = sum(track["matched_rows_scored"] for track in report["tracks"].values())
    total_correct = sum(track["correct_rows"] for track in report["tracks"].values())
    report["combined_track_samples"] = {
        "rows": total_scored,
        "correct": total_correct,
        "micro_percent": 100.0 * total_correct / total_scored if total_scored else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
