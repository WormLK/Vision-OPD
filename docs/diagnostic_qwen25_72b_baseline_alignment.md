# Baseline Alignment Diagnostic with Qwen2.5-72B

Date: 2026-07-18

## Scope

This is an interim diagnostic, not the final paper-aligned score table. The
model answers were produced by the pristine released Vision-OPD evaluation
pipeline. The only deliberate substitution is the fallback judge:
`Qwen2.5-72B-Instruct` was used while the released `openai/gpt-oss-120b`
judge weights were still downloading. Final report columns accept only the
GPT-OSS-120B results.

The diagnostic outputs are isolated under:

```text
benchmark/official_reproduction_20260717/diagnostic_qwen25_72b/
```

## Inference Configuration

Both untrained baselines used the released `eval/run_eval.sh` behavior:

```text
seed=42
temperature=0
ENABLE_THINKING=False
max_tokens=32768
max_retries=3
parallel_workers=256
```

All six datasets were prepared by the pristine released `prepare_data.py`.
Each model has 31,707 answers with no empty answers, API errors, missing UIDs,
or duplicate UIDs.

## Results

| Benchmark | Paper Base 4B | Local Base 4B (diagnostic) | Delta | Paper Base 9B | Local Base 9B (diagnostic) | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vstar | 84.29 | 82.72 | -1.57 | 82.72 | 85.34 | +2.62 |
| ZoomBench | 47.69 | 48.88 | +1.19 | 52.07 | 53.02 | +0.95 |
| HR-Bench-4K | 84.38 | 87.12 | +2.74 | 85.75 | 85.75 | +0.00 |
| HR-Bench-8K | 80.13 | 82.75 | +2.62 | 80.63 | 79.88 | -0.75 |
| MME-RealWorld-EN | 63.86 | 63.79 | -0.07 | 71.40 | 71.25 | -0.15 |
| MME-RealWorld-CN | 63.70 | 64.98 | +1.28 | 67.67 | 68.93 | +1.26 |
| **Macro** | **70.68** | **71.71** | **+1.03** | **73.37** | **74.03** | **+0.66** |

The diagnostic passes the predeclared alignment gate for both backbones:

- absolute per-benchmark deviation is at most 5 percentage points;
- absolute macro deviation is at most 3 percentage points.

The automated audit output is saved as
`benchmark/official_reproduction_20260717/diagnostic_qwen25_72b/alignment_audit.txt`
and ends with `PASS: both local baselines satisfy the strict alignment gate`.

This is strong evidence that the local baseline inference configuration and
prepared benchmark manifests are aligned with the released pipeline. It does
not remove the requirement to rerun the same answers with GPT-OSS-120B.

## Judge Provenance

The diagnostic used the unmodified official `judge_qwenlm.py` with:

```text
judge_model=Qwen2.5-72B-Instruct-diagnostic-judge
temperature=0
judge_max_tokens=16
tensor_parallel_size=8
max_model_len=8192
```

All twelve judge files contain the expected number of records and every
stored judge value is exactly `Yes` or `No` after case normalization. The
reproduction command is:

```bash
bash scripts/run_diagnostic_qwen25_72b_baseline_judge.sh
```

## Next Gate

After all 15 GPT-OSS-120B shards pass official SHA-256 validation,
`scripts/judge_official_baselines.sh` reruns the fallback judging and applies
the same alignment thresholds. Existing trained checkpoints are evaluated
only after that formal gate passes.
