#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path


BENCHMARKS = {
    "vstar": (191, 84.29, 82.72),
    "zoombench": (845, 47.69, 52.07),
    "hrbench-4k": (800, 84.38, 85.75),
    "hrbench-8k": (800, 80.13, 80.63),
    "mme-realworld": (23609, 63.86, 71.40),
    "mme-realworld-cn": (5462, 63.70, 67.67),
}
MANIFESTS = {
    "vstar": "vstar.json",
    "zoombench": "zoombench.json",
    "hrbench-4k": "hr_bench_4k.json",
    "hrbench-8k": "hr_bench_8k.json",
    "mme-realworld": "MME_RealWorld.json",
    "mme-realworld-cn": "MME_RealWorld_CN.json",
}
MODELS = (
    ("Qwen3.5-4B-baseline-official", 1),
    ("Qwen3.5-9B-baseline-official", 2),
)


def score_from_file(path):
    matches = re.findall(r"(\d+(?:\.\d+)?)%", path.read_text(encoding="utf-8", errors="replace"))
    if not matches:
        raise ValueError(f"score not found: {path}")
    return float(matches[-1])


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--max-benchmark-delta", type=float, default=5.0)
    parser.add_argument("--max-macro-delta", type=float, default=3.0)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--backbone", choices=("4b", "9b", "all"), default="all")
    args = parser.parse_args()

    root = (
        args.evaluation_root.resolve()
        if args.evaluation_root
        else args.project_root.resolve() / "benchmark" / "official_reproduction_20260717"
    )
    errors = []
    summary = {}
    selected_models = MODELS
    if args.backbone != "all":
        selected_models = tuple(item for item in MODELS if item[0].lower().startswith(f"qwen3.5-{args.backbone}"))
    for model, paper_index in selected_models:
        scores = []
        rows = []
        for benchmark, (expected, paper4, paper9) in BENCHMARKS.items():
            paper = (paper4, paper9)[paper_index - 1]
            answer_path = root / "model_answer" / benchmark / f"{model}_seed42_answer.jsonl"
            judge_path = root / "judge" / benchmark / f"{model}_seed42_answer.jsonl"
            result_path = root / "results" / f"{model}_seed42_{benchmark}.txt"
            try:
                answers = [json.loads(line) for line in answer_path.open(encoding="utf-8") if line.strip()]
            except Exception as exc:
                errors.append(f"invalid answers {answer_path}: {exc}")
                continue
            if len(answers) != expected:
                errors.append(f"answer count {answer_path}: {len(answers)}/{expected}")
            uids = [row.get("sample_uid") for row in answers]
            if any(not uid for uid in uids) or len(set(uids)) != len(uids):
                errors.append(f"missing or duplicate answer UID: {answer_path}")
            manifest_path = root / "source" / "eval" / MANIFESTS[benchmark]
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                expected_uids = {sample_uid(row, benchmark) for row in manifest}
                if len(expected_uids) != expected:
                    errors.append(
                        f"manifest UID count {manifest_path}: {len(expected_uids)}/{expected}"
                    )
                if set(uids) != expected_uids:
                    errors.append(f"answer UID set differs from manifest: {answer_path}")
            except Exception as exc:
                errors.append(f"invalid manifest {manifest_path}: {exc}")
            failed = [
                row for row in answers
                if not str(row.get("model_answer", "")).strip()
                or str(row.get("model_answer", "")).startswith(("[API_ERROR]", "[FUTURE_ERROR]"))
            ]
            if failed:
                errors.append(f"failed answer records {answer_path}: {len(failed)}")
            if args.inference_only:
                continue
            try:
                judged = json.loads(judge_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"invalid judge {judge_path}: {exc}")
                continue
            if len(judged) != expected:
                errors.append(f"judge count {judge_path}: {len(judged)}/{expected}")
            invalid = [row for row in judged if str(row.get("judge", "")).strip().lower() not in {"yes", "no"}]
            if invalid:
                errors.append(f"non Yes/No judge records {judge_path}: {len(invalid)}")
            if not result_path.is_file():
                errors.append(f"missing score: {result_path}")
                continue
            score = score_from_file(result_path)
            recomputed = 100.0 * sum(
                str(row.get("judge", "")).strip().lower() == "yes" for row in judged
            ) / len(judged)
            if abs(score - recomputed) > 0.011:
                errors.append(f"score mismatch {result_path}: {score:.2f}/{recomputed:.2f}")
            deviation = score - paper
            if abs(deviation) > args.max_benchmark_delta:
                errors.append(
                    f"baseline deviation exceeds gate {model}/{benchmark}: {deviation:+.2f} pp"
                )
            scores.append(score)
            rows.append((benchmark, score, paper, deviation))
        if len(scores) == len(BENCHMARKS):
            local_macro = sum(scores) / len(scores)
            paper_macro = (70.68, 73.37)[paper_index - 1]
            macro_deviation = local_macro - paper_macro
            if abs(macro_deviation) > args.max_macro_delta:
                errors.append(f"macro deviation exceeds gate {model}: {macro_deviation:+.2f} pp")
            summary[model] = (rows, local_macro, paper_macro, macro_deviation)

    if args.inference_only:
        if errors:
            print("FAILED official baseline inference audit:")
            for error in errors:
                print(f"- {error}")
            raise SystemExit(1)
        print(f"PASS: {args.backbone} local baseline inference outputs exactly match all six manifests")
        return

    for model, (rows, local_macro, paper_macro, macro_deviation) in summary.items():
        print(model)
        for benchmark, score, paper, deviation in rows:
            print(f"  {benchmark}: local={score:.2f} paper={paper:.2f} delta={deviation:+.2f}")
        print(f"  macro: local={local_macro:.2f} paper={paper_macro:.2f} delta={macro_deviation:+.2f}")
    if errors:
        print("FAILED official baseline alignment audit:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"PASS: {args.backbone} local baselines satisfy the strict alignment gate")


if __name__ == "__main__":
    main()
