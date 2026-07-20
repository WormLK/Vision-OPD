#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


DATASETS = (
    ("hr_bench_4k.json", "hr_bench_4k.parquet"),
    ("hr_bench_8k.json", "hr_bench_8k.parquet"),
)


def augment(prepared_dir, raw_dir, json_name, parquet_name):
    json_path = prepared_dir / json_name
    parquet_path = raw_dir / parquet_name
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    rows = pq.read_table(parquet_path, columns=["index", "cycle_category"]).to_pylist()
    metadata = {int(row["index"]): int(row["cycle_category"]) for row in rows}
    records = json.loads(json_path.read_text(encoding="utf-8"))
    missing = []
    for record in records:
        index = int(record["index"])
        if index not in metadata:
            missing.append(index)
            continue
        record["cycle_category"] = metadata[index]
    if missing:
        raise ValueError(f"{json_name}: missing metadata for {len(missing)} indices")
    if len(metadata) != len(records):
        raise ValueError(f"{json_name}: metadata/JSON count mismatch {len(metadata)}/{len(records)}")

    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, json_path)
    print(f"Augmented {json_path}: {len(records)} records")


def main():
    parser = argparse.ArgumentParser(description="Restore HR-Bench cycle metadata")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    args = parser.parse_args()
    prepared_dir = Path(args.prepared_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    for json_name, parquet_name in DATASETS:
        augment(prepared_dir, raw_dir, json_name, parquet_name)


if __name__ == "__main__":
    main()
