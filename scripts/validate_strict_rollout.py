#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


FORBIDDEN_TOKENS = ('"pixel_values"', '"deferred_images"', '"image_grid_thw"', '"bytes":')


def main():
    parser = argparse.ArgumentParser(description="Validate one strict 96x8 rollout artifact")
    parser.add_argument("rollout_dir", type=Path)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    path = args.rollout_dir / f"{args.step}.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing rollout artifact: {path}")

    counts = Counter()
    rows = 0
    forbidden = Counter()
    with path.open(encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            for token in FORBIDDEN_TOKENS:
                if token in line:
                    forbidden[token] += 1
            try:
                record = json.loads(line)
                counts[record["input"]] += 1
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise SystemExit(f"invalid rollout row {path}:{line_number}: {exc}") from exc

    multiplicities = Counter(counts.values())
    if rows != 768 or len(counts) != 96 or multiplicities != Counter({8: 96}) or forbidden:
        raise SystemExit(
            f"invalid strict rollout step={args.step}: rows={rows}, unique_prompts={len(counts)}, "
            f"multiplicities={dict(multiplicities)}, forbidden={dict(forbidden)}"
        )
    print(
        f"PASS strict rollout step={args.step}: 768 rows, 96 unique prompts x8, "
        "no image/deferred payload"
    )


if __name__ == "__main__":
    main()
