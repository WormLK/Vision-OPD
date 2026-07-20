#!/usr/bin/env python3
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Normalize empty official judge completions to the official API-error fallback value"
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    path = args.path.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != args.expected:
        raise SystemExit(f"unexpected judge payload size: {len(data) if isinstance(data, list) else 'non-list'}")

    invalid = [row for row in data if str(row.get("judge", "")).strip().lower() not in {"yes", "no"}]
    nonempty = [row for row in invalid if str(row.get("judge", "")).strip()]
    if nonempty:
        raise SystemExit(f"refusing to normalize {len(nonempty)} nonempty invalid judge values")
    if not invalid:
        print(f"PASS no empty judge completions: {path}")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.stem}.raw_empty_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    for row in invalid:
        row["judge_raw"] = row.get("judge", "")
        row["judge"] = "No"
        row["judge_source"] = "llm_empty_fallback_no"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print(f"NORMALIZED empty_judges={len(invalid)} backup={backup} output={path}")


if __name__ == "__main__":
    main()
