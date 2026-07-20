#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


BENCHMARKS = {
    "vstar": ("vstar.json", 191),
    "zoombench": ("zoombench.json", 845),
    "hrbench-4k": ("hr_bench_4k.json", 800),
    "hrbench-8k": ("hr_bench_8k.json", 800),
    "mme-realworld": ("MME_RealWorld.json", 23609),
    "mme-realworld-cn": ("MME_RealWorld_CN.json", 5462),
    "mmstar": ("mmstar.json", 1500),
    "pope": ("POPE.json", 9000),
    "cv-bench": ("cv_bench.json", 2638),
    "mmvp": ("mmvp.json", 300),
}
ERROR_PREFIXES = ("[API_ERROR]", "[FUTURE_ERROR]", "[JUDGE_API_ERROR]")


def sample_uid(item, benchmark):
    for key in ("sample_uid", "uid", "index", "question_id", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return f"{benchmark}:{key}:{value}"
    stable_obj = {
        "benchmark": benchmark,
        "images": item.get("images") or [],
        "query": item.get("query", ""),
    }
    raw = json.dumps(stable_obj, ensure_ascii=False, sort_keys=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_from_file(path, benchmark):
    text = path.read_text(encoding="utf-8", errors="replace")
    if benchmark == "pope":
        matches = re.findall(r"accuracy=(\d+(?:\.\d+)?)%", text)
    else:
        matches = re.findall(r"(\d+(?:\.\d+)?)%", text)
    if not matches:
        raise ValueError(f"score not found: {path}")
    return float(matches[-1])


def main():
    parser = argparse.ArgumentParser(description="Validate one model's pristine official evaluation artifacts")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve() / "benchmark" / "official_reproduction_20260717"
    eval_dir = root / "source" / "eval"
    tag = f"{args.model}_seed42"
    errors = []

    for benchmark, (manifest_name, expected_count) in BENCHMARKS.items():
        manifest_path = eval_dir / manifest_name
        answer_path = root / "model_answer" / benchmark / f"{tag}_answer.jsonl"
        judge_path = root / "judge" / benchmark / f"{tag}_answer.jsonl"
        result_path = root / "results" / f"{tag}_{benchmark}.txt"

        try:
            manifest = load_json(manifest_path)
            expected_uids = {sample_uid(row, benchmark) for row in manifest}
            if len(manifest) != expected_count or len(expected_uids) != expected_count:
                errors.append(
                    f"manifest count/UID mismatch {manifest_path}: "
                    f"rows={len(manifest)} uids={len(expected_uids)} expected={expected_count}"
                )
        except Exception as exc:
            errors.append(f"invalid manifest {manifest_path}: {exc}")
            continue

        try:
            answers = load_jsonl(answer_path)
        except Exception as exc:
            errors.append(f"invalid answers {answer_path}: {exc}")
            continue
        answer_uids = [row.get("sample_uid") for row in answers]
        if len(answers) != expected_count:
            errors.append(f"answer count {answer_path}: {len(answers)}/{expected_count}")
        if any(not uid for uid in answer_uids) or len(set(answer_uids)) != len(answer_uids):
            errors.append(f"missing or duplicate answer UID: {answer_path}")
        elif set(answer_uids) != expected_uids:
            errors.append(f"answer UID set differs from manifest: {answer_path}")
        failed_answers = [
            row
            for row in answers
            if not str(row.get("model_answer", "")).strip()
            or str(row.get("model_answer", "")).strip().startswith(ERROR_PREFIXES)
        ]
        if failed_answers:
            errors.append(f"failed answer records {answer_path}: {len(failed_answers)}")
        if args.inference_only:
            continue

        try:
            judged = load_json(judge_path)
        except Exception as exc:
            errors.append(f"invalid judge {judge_path}: {exc}")
            continue
        judged_uids = [row.get("sample_uid") for row in judged]
        if len(judged) != expected_count:
            errors.append(f"judge count {judge_path}: {len(judged)}/{expected_count}")
        if any(not uid for uid in judged_uids) or len(set(judged_uids)) != len(judged_uids):
            errors.append(f"missing or duplicate judge UID: {judge_path}")
        elif set(judged_uids) != set(answer_uids):
            errors.append(f"judge UID set differs from answers: {judge_path}")
        invalid_judges = [
            row for row in judged if str(row.get("judge", "")).strip().lower() not in {"yes", "no"}
        ]
        if invalid_judges:
            errors.append(f"non Yes/No judge records {judge_path}: {len(invalid_judges)}")

        try:
            score_from_file(result_path, benchmark)
            recomputed = subprocess.check_output(
                [
                    sys.executable,
                    str(eval_dir / "cal_acc.py"),
                    "--benchmark",
                    benchmark,
                    "--judge_json",
                    str(judge_path),
                    "--benchmark_json",
                    str(manifest_path),
                ],
                text=True,
            ).strip()
            reported = result_path.read_text(encoding="utf-8", errors="replace").strip()
            if reported != recomputed:
                errors.append(f"score output differs from pristine cal_acc.py: {result_path}")
        except Exception as exc:
            errors.append(f"invalid score {result_path}: {exc}")

    if errors:
        print(f"FAILED official artifact validation for {args.model}:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    scope = "inference" if args.inference_only else "inference, judge, and score"
    print(f"PASS: {args.model} has complete official {scope} artifacts for all 10 benchmarks")


if __name__ == "__main__":
    main()
