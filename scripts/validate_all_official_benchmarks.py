#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


MANIFESTS = {
    "zoombench": ("zoombench.json", 845),
    "vstar": ("vstar.json", 191),
    "hrbench-4k": ("hr_bench_4k.json", 800),
    "hrbench-8k": ("hr_bench_8k.json", 800),
    "mme-realworld": ("MME_RealWorld.json", 23609),
    "mme-realworld-cn": ("MME_RealWorld_CN.json", 5462),
    "mme-realworld-lite": ("MME_RealWorld_Lite.json", 1919),
    "mmstar": ("mmstar.json", 1500),
    "pope": ("POPE.json", 9000),
    "pope_adv": ("POPE_adv.json", 3000),
    "pope_pop": ("POPE_pop.json", 3000),
    "pope_random": ("POPE_random.json", 3000),
    "cv-bench": ("cv_bench.json", 2638),
    "mmvp": ("mmvp.json", 300),
    "visualprobe": ("visualprobe.json", 515),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--marker", type=Path)
    args = parser.parse_args()

    total_records = 0
    total_images = 0
    for benchmark, (filename, expected_count) in MANIFESTS.items():
        path = args.eval_dir / filename
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != expected_count:
            raise SystemExit(
                f"record count mismatch for {benchmark}: {len(rows)} != {expected_count}"
            )
        image_count = 0
        for row_index, row in enumerate(rows):
            for image in row.get("images") or []:
                image_path = Path(image)
                if not image_path.is_file() or image_path.stat().st_size == 0:
                    raise SystemExit(
                        f"missing image for {benchmark} row {row_index}: {image_path}"
                    )
                image_count += 1
        total_records += len(rows)
        total_images += image_count
        print(f"{benchmark}: records={len(rows)} images={image_count}")

    print(f"total: benchmarks={len(MANIFESTS)} records={total_records} images={total_images}")
    if args.marker is not None:
        args.marker.parent.mkdir(parents=True, exist_ok=True)
        args.marker.touch()


if __name__ == "__main__":
    main()
