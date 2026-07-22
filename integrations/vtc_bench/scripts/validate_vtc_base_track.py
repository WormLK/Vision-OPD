#!/usr/bin/env python3
"""Validate a complete no-tool, one-shot VTC-Bench Base track."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


INVALID_ANSWER_MARKERS = (
    "unable",
    "your final answer here",
    "cannot",
    "indiscernible",
    "insufficient",
    "unreadable",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=680)
    args = parser.parse_args()
    model_dir = args.results_dir.resolve() / args.model
    files = sorted(model_dir.glob("results_*.jsonl"))
    if not files:
        raise SystemExit(f"missing Base result JSONL under {model_dir}")
    rows = [json.loads(line) for line in files[-1].read_text(encoding="utf-8").splitlines() if line]
    indices = [row.get("row_index") for row in rows]
    item_ids = [str(row.get("item_id") or "").strip() for row in rows]
    invalid = []
    protocol_errors = []
    raw_response_errors = []

    def walk(value):
        if isinstance(value, dict):
            yield value
            for nested in value.values():
                yield from walk(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from walk(nested)

    for row in rows:
        answer = str(row.get("agent_answer") or "").strip().lower()
        if row.get("status") != "success" or not answer or any(x in answer for x in INVALID_ANSWER_MARKERS):
            invalid.append(row.get("row_index"))
        conversation = row.get("conversation") or []
        roles = [str(message.get("role") or "") for message in conversation]
        if any(role in {"tool", "function"} for role in roles):
            protocol_errors.append((row.get("row_index"), "tool/function message"))
        if roles.count("user") != 1:
            protocol_errors.append((row.get("row_index"), f"user turns={roles.count('user')}"))
        if any(message.get("function_call") for message in conversation):
            protocol_errors.append((row.get("row_index"), "function_call"))
        item_id = str(row.get("item_id") or "").strip()
        raw_path = model_dir / f"response_list_{item_id}.json"
        if not raw_path.is_file():
            raw_response_errors.append((row.get("row_index"), "missing raw response"))
            continue
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raw_response_errors.append((row.get("row_index"), f"invalid raw response: {exc}"))
            continue
        mappings = list(walk(raw.get("response_list")))
        assistant_messages = [item for item in mappings if item.get("role") == "assistant"]
        if len(assistant_messages) != 1:
            raw_response_errors.append(
                (row.get("row_index"), f"raw assistant messages={len(assistant_messages)}")
            )
        for item in mappings:
            if item.get("role") in {"tool", "function"}:
                raw_response_errors.append((row.get("row_index"), "raw tool/function role"))
                break
            if item.get("function_call") or item.get("tool_calls"):
                raw_response_errors.append((row.get("row_index"), "raw function/tool call"))
                break

    errors = []
    if len(rows) != args.expected:
        errors.append(f"rows={len(rows)}, expected={args.expected}")
    if len(set(indices)) != args.expected or None in indices:
        errors.append(f"unique row_index={len(set(indices))}, expected={args.expected}")
    if len(set(item_ids)) != args.expected or "" in item_ids:
        errors.append(f"unique item_id={len(set(item_ids))}, expected={args.expected}")
    if invalid:
        errors.append(f"invalid answers={len(invalid)} examples={invalid[:10]}")
    if protocol_errors:
        errors.append(f"Base protocol errors={len(protocol_errors)} examples={protocol_errors[:5]}")
    if raw_response_errors:
        errors.append(
            f"raw Base protocol errors={len(raw_response_errors)} "
            f"examples={raw_response_errors[:5]}"
        )

    overall = None
    if args.score_file.is_file():
        with args.score_file.open(encoding="utf-8-sig", newline="") as handle:
            score_rows = list(csv.DictReader(handle))
        if score_rows:
            try:
                overall = float(score_rows[0]["Overall"])
            except (KeyError, TypeError, ValueError):
                pass
    if overall is None:
        errors.append(f"missing or invalid Overall score: {args.score_file}")
    if errors:
        print(f"FAILED VTC Base validation: {files[-1]}")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(
        f"PASS VTC Base: rows={len(rows)} one_llm_response=true "
        f"no_tools=true one_user_turn=true overall={overall:.2f}"
    )


if __name__ == "__main__":
    main()
