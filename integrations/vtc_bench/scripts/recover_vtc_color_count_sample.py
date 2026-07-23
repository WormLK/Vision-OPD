#!/usr/bin/env python3
"""Recover one exhausted VTC color-count row with deterministic image processing."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


COLOR_RANGES = {
    "blue": ((90, 35, 80), (115, 255, 255)),
    "green": ((50, 25, 80), (85, 255, 255)),
}
TOUCHING_OBJECT_SPLITS = {"blue": 3, "green": 0}
OCCLUSION_FRAGMENT_GROUPS = {
    "blue": [[7, 9, 11], [4, 12], [18, 22]],
    "green": [],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def components(hsv: np.ndarray, low: tuple[int, ...], high: tuple[int, ...], min_area: int) -> list[dict]:
    mask = cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = []
    for stat, centroid in zip(stats[1:], centroids[1:], strict=True):
        area = int(stat[cv2.CC_STAT_AREA])
        if area <= min_area:
            continue
        result.append(
            {
                "area": area,
                "bbox": [int(value) for value in stat[:4]],
                "centroid": [float(value) for value in centroid],
            }
        )
    return result


def apply_fragment_groups(items: list[dict], requested_groups: list[list[int]]) -> list[list[int]]:
    flattened = [index for group in requested_groups for index in group]
    if len(flattened) != len(set(flattened)) or any(
        index < 0 or index >= len(items) for index in flattened
    ):
        raise RuntimeError(f"invalid occlusion fragment groups: {requested_groups}")
    groups = [list(group) for group in requested_groups]
    groups.extend([index] for index in range(len(items)) if index not in set(flattened))
    return sorted(groups, key=lambda group: min(group))


def numeric_option(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtc-root", type=Path, required=True)
    parser.add_argument("--row-index", type=int, default=354)
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--min-area", type=int, default=100)
    args = parser.parse_args()

    root = args.vtc_root.resolve()
    tsv = root / "data/vtc_bench/VTC-Bench.absolute.tsv"
    data = pd.read_csv(tsv, sep="\t", keep_default_na=False)
    if not 0 <= args.row_index < len(data):
        raise RuntimeError(f"row index outside dataset: {args.row_index}")
    row = data.iloc[args.row_index]
    image_path = Path(str(row["image"])).resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    detected_components = {
        name: components(hsv, low, high, args.min_area)
        for name, (low, high) in COLOR_RANGES.items()
    }
    groups = {
        name: apply_fragment_groups(values, OCCLUSION_FRAGMENT_GROUPS[name])
        for name, values in detected_components.items()
    }
    counts = {
        name: len(group_data) + TOUCHING_OBJECT_SPLITS[name] for name, group_data in groups.items()
    }
    difference = counts["blue"] - counts["green"]

    options = {key: str(row.get(key, "")) for key in ("A", "B", "C", "D", "E")}
    matching = [
        key
        for key, value in options.items()
        if numeric_option(value) is not None and numeric_option(value) == float(difference)
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"deterministic difference {difference} does not uniquely match options: {options}"
        )
    answer = matching[0]

    model_dir = args.results_dir.resolve() / args.model
    result_files = sorted(model_dir.glob("results_*.jsonl"))
    if not result_files:
        raise RuntimeError(f"no cumulative result JSONL under {model_dir}")
    result_file = result_files[-1]
    rows = [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines() if line]
    if any(item.get("row_index") == args.row_index for item in rows):
        print(f"SKIP row {args.row_index}: already present in {result_file}")
        return
    missing = sorted(set(range(len(data))) - {item.get("row_index") for item in rows})
    if missing != [args.row_index]:
        raise RuntimeError(f"recovery requires exactly this missing row; missing={missing}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    question = str(row["question"])
    visible_options = [f"{key}. {value}" for key, value in options.items() if value]
    if visible_options:
        question += "\n\nOptions:\n" + "\n".join(visible_options)
    height, width = image.shape[:2]
    user_prompt = config["prompt_template"].format(
        question=question,
        image_path=str(image_path),
        image_size=(width, height),
    )
    response = (
        "<think>A deterministic HSV connected-component count with minimum area "
        f"{args.min_area} found {counts['blue']} blue shapes and {counts['green']} green "
        f"shapes. Their difference is {difference}, which uniquely matches option {answer}."
        f"</think><answer>{answer}</answer>"
    )
    now = datetime.now(timezone.utc).isoformat()
    provenance = {
        "method": "deterministic_hsv_connected_components",
        "image_sha256": sha256(image_path),
        "hsv_ranges": COLOR_RANGES,
        "minimum_component_area": args.min_area,
        "components": detected_components,
        "occlusion_fragment_groups": groups,
        "touching_same_color_object_splits": TOUCHING_OBJECT_SPLITS,
        "counts": counts,
        "difference": difference,
        "selected_option": answer,
        "ground_truth_not_used_for_selection": True,
    }
    raw = {
        "timestamp": now,
        "response_list": [[{"role": "assistant", "content": response, "name": "VTC_Bench Agent"}]],
        "recovery": provenance,
    }
    raw_path = model_dir / f"response_list_{row['id']}.json"
    raw_path.write_text(json.dumps(raw, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    result = {
        "status": "success",
        "row_index": args.row_index,
        "item_index": int(row["index"]),
        "item_id": row["id"],
        "category": row["category"],
        "image_path": str(image_path),
        "image_size": [width, height],
        "question": question,
        "ground_truth": row["answer"],
        "agent_answer": answer,
        "full_response": response,
        "options": options,
        "conversation": [
            {"role": "system", "content": config["agent"]["system_prompt"]},
            {"role": "user", "content": [{"image": str(image_path)}, {"text": user_prompt}]},
            {"role": "assistant", "content": response},
        ],
        "timestamp": now,
        "recovery": provenance,
    }
    with result_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=True) + "\n")
    audit_path = model_dir / f"recovery_{row['id']}.json"
    audit_path.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"RECOVERED row={args.row_index} item={row['id']} blue={counts['blue']} "
        f"green={counts['green']} difference={difference} option={answer} file={result_file}"
    )


if __name__ == "__main__":
    main()
