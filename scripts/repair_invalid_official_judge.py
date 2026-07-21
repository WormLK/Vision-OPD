#!/usr/bin/env python3
"""Retry only malformed outputs produced by the frozen official LLM judge."""

import argparse
import importlib.util
import json
import os
from pathlib import Path


VALID_LABELS = {"yes": "Yes", "no": "No"}


def load_official_judge(path: Path):
    spec = importlib.util.spec_from_file_location("vision_opd_official_judge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import official judge from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invalid_indices(items):
    return [
        index
        for index, item in enumerate(items)
        if str(item.get("judge", "")).strip().lower() not in VALID_LABELS
    ]


def write_atomically(path: Path, items):
    temporary = path.with_suffix(path.suffix + ".repairing")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(items, output, ensure_ascii=False, indent=4)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-json", required=True, type=Path)
    parser.add_argument("--judge-script", required=True, type=Path)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--parallel-workers", type=int, default=32)
    parser.add_argument("--max-rounds", type=int, default=10)
    args = parser.parse_args()

    with args.judge_json.open("r", encoding="utf-8") as source:
        items = json.load(source)

    official = load_official_judge(args.judge_script)
    pending = invalid_indices(items)
    if not pending:
        print(f"Judge repair not needed: {args.judge_json}")
        return

    print(f"Repairing {len(pending)} invalid official judge outputs.")
    for round_number in range(1, args.max_rounds + 1):
        prompts = []
        for index in pending:
            item = items[index]
            question = item["query"].replace("<image>", "")
            extracted = item.get("extracted_answer")
            if extracted is None:
                extracted = official.extract_answer(item["model_answer"])
            prompts.append(
                official.PROMPT_TEMPLATE.format(
                    question=question,
                    gt=item["response"],
                    response=extracted,
                )
            )

        responses = official.judge_via_api(
            prompts,
            args.api_base,
            args.api_key,
            args.judge_model,
            args.judge_max_tokens,
            parallel_workers=args.parallel_workers,
        )
        for index, response in zip(pending, responses):
            normalized = str(response).strip().lower()
            items[index]["judge"] = VALID_LABELS.get(normalized, response)
            items[index]["judge_source"] = "llm"

        write_atomically(args.judge_json, items)
        pending = invalid_indices(items)
        print(
            f"Judge repair round {round_number}: "
            f"{len(pending)} invalid outputs remain."
        )
        if not pending:
            return

    raise SystemExit(
        f"Unable to repair {len(pending)} judge outputs after {args.max_rounds} rounds"
    )


if __name__ == "__main__":
    main()
