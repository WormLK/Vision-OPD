#!/usr/bin/env python3
import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate path-only Vision-OPD image columns")
    parser.add_argument("data", type=Path)
    parser.add_argument("--expected-rows", type=int, default=6241)
    args = parser.parse_args()

    data_path = args.data.resolve()
    parquet = pq.ParquetFile(data_path)
    schema = parquet.schema_arrow
    expected_type = "list<element: struct<path: string>>"
    errors = []
    for column in ("images", "bbox_images"):
        if column not in schema.names:
            errors.append(f"missing column: {column}")
        elif str(schema.field(column).type) != expected_type:
            errors.append(
                f"{column} must be path-only {expected_type}, got {schema.field(column).type}"
            )
    if parquet.metadata.num_rows != args.expected_rows:
        errors.append(f"rows={parquet.metadata.num_rows}, expected={args.expected_rows}")

    counts = {"images": 0, "bbox_images": 0}
    unique_paths = set()
    if not errors:
        rows = pq.read_table(data_path, columns=list(counts)).to_pylist()
        for row_index, row in enumerate(rows):
            for column in counts:
                refs = row[column] or []
                if len(refs) != 1:
                    errors.append(f"row {row_index} {column} refs={len(refs)}, expected=1")
                    continue
                ref = refs[0]
                if set(ref) != {"path"} or not isinstance(ref["path"], str) or not ref["path"]:
                    errors.append(f"row {row_index} {column} is not a single path reference")
                    continue
                path = Path(ref["path"])
                counts[column] += 1
                unique_paths.add(path)
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"row {row_index} {column} missing or empty: {path}")

    if errors:
        print(f"FAILED path-only training data validation: {data_path}")
        for error in errors[:50]:
            print(f"- {error}")
        if len(errors) > 50:
            print(f"- ... {len(errors) - 50} more errors")
        raise SystemExit(1)

    print(
        f"PASS path-only training data: rows={parquet.metadata.num_rows}, "
        f"images={counts['images']}, bbox_images={counts['bbox_images']}, "
        f"unique_paths={len(unique_paths)}"
    )


if __name__ == "__main__":
    main()
