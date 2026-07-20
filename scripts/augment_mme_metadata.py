#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import pyarrow.dataset as ds


DATASETS = (
    (
        "MME_RealWorld.json",
        "yifanzhang114_MME-RealWorld-Lmms-eval/data",
    ),
    (
        "MME_RealWorld_CN.json",
        "yifanzhang114_MME-RealWorld-CN-Lmms-eval/data",
    ),
)


def augment(prepared_dir, json_name, parquet_subdir):
    json_path = prepared_dir / json_name
    parquet_dir = prepared_dir / parquet_subdir
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    if not parquet_dir.is_dir():
        raise FileNotFoundError(parquet_dir)

    table = ds.dataset(str(parquet_dir), format="parquet").to_table(
        columns=["index", "category", "l2-category"]
    )
    metadata = {
        int(row["index"]): (
            str(row.get("category") or "unknown").strip(),
            str(row.get("l2-category") or "unknown").strip(),
        )
        for row in table.to_pylist()
    }
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    missing = []
    for row in rows:
        index = int(row["index"])
        values = metadata.get(index)
        if values is None:
            missing.append(index)
            continue
        row["category"], row["l2_category"] = values
    if missing:
        raise ValueError(f"{json_name}: missing metadata for {len(missing)} indices")
    if len(metadata) != len(rows):
        raise ValueError(f"{json_name}: metadata/JSON count mismatch {len(metadata)}/{len(rows)}")

    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, json_path)
    print(f"Augmented {json_path}: {len(rows)} records")


def main():
    parser = argparse.ArgumentParser(description="Restore MME-RealWorld category metadata")
    parser.add_argument("--prepared-dir", required=True)
    args = parser.parse_args()
    prepared_dir = Path(args.prepared_dir).resolve()
    for json_name, parquet_subdir in DATASETS:
        augment(prepared_dir, json_name, parquet_subdir)


if __name__ == "__main__":
    main()
