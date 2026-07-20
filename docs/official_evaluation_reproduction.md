# Vision-OPD Official Evaluation Reproduction

This report is reserved for the strict reproduction based on the pristine official
`eval/run_eval.sh` implementation. Diagnostic runs with `max_tokens=1024` are excluded.

## Table 1 Alignment

| Benchmark | Paper Base 4B | Local Base 4B | Paper OPD 4B | Local OPD 4B | Paper Base 9B | Local Base 9B | Paper OPD 9B | Local OPD 9B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Vstar | 84.29% | 82.72% | 92.15% | pending | 82.72% | pending | 94.76% | pending |
| ZoomBench | 47.69% | 48.88% | 59.76% | pending | 52.07% | pending | 65.80% | pending |
| HR-Bench-4K | 84.38% | 87.25% | 84.50% | pending | 85.75% | pending | 88.13% | pending |
| HR-Bench-8K | 80.13% | 83.00% | 80.38% | pending | 80.63% | pending | 85.50% | pending |
| MME-RealWorld-EN | 63.86% | 63.85% | 74.88% | pending | 71.40% | pending | 73.40% | pending |
| MME-RealWorld-CN | 63.70% | 64.85% | 70.76% | pending | 67.67% | pending | 70.46% | pending |
| Macro | 70.68% | 71.76% | 77.07% | pending | 73.37% | pending | 79.68% | pending |

## All 10 Goal Benchmarks: 4B

The Vision-OPD paper main table reports values for the first six benchmarks. Paper cells for MMStar, POPE, CV-Bench, and MMVP are marked `N/R`.

| Benchmark | Paper Baseline 4B | Local Baseline 4B | Paper OPD-4B | Local OPD-4B |
| --- | ---: | ---: | ---: | ---: |
| Vstar | 84.29% | 82.72% | 92.15% | pending |
| ZoomBench | 47.69% | 48.88% | 59.76% | pending |
| HR-Bench-4K | 84.38% | 87.25% | 84.50% | pending |
| HR-Bench-8K | 80.13% | 83.00% | 80.38% | pending |
| MME-RealWorld-EN | 63.86% | 63.85% | 74.88% | pending |
| MME-RealWorld-CN | 63.70% | 64.85% | 70.76% | pending |
| MMStar | N/R | pending | N/R | pending |
| POPE-Test | N/R | pending | N/R | pending |
| CV-Bench | N/R | pending | N/R | pending |
| MMVP | N/R | pending | N/R | pending |

## Baseline Deviation

| Benchmark | Local 4B - Paper 4B | Local 9B - Paper 9B |
| --- | ---: | ---: |
| Vstar | -1.57 | pending |
| ZoomBench | +1.19 | pending |
| HR-Bench-4K | +2.87 | pending |
| HR-Bench-8K | +2.87 | pending |
| MME-RealWorld-EN | -0.01 | pending |
| MME-RealWorld-CN | +1.15 | pending |

## Artifact Completeness

| Model | Benchmark | Inference | Official judge | Score |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-4B-baseline-official | Vstar | 191/191 | 191/191 | 82.72% |
| Qwen3.5-4B-baseline-official | ZoomBench | 845/845 | 845/845 | 48.88% |
| Qwen3.5-4B-baseline-official | HR-Bench-4K | 800/800 | 800/800 | 87.25% |
| Qwen3.5-4B-baseline-official | HR-Bench-8K | 800/800 | 800/800 | 83.00% |
| Qwen3.5-4B-baseline-official | MME-RealWorld-EN | 23609/23609 | 23609/23609 | 63.85% |
| Qwen3.5-4B-baseline-official | MME-RealWorld-CN | 5462/5462 | 5462/5462 | 64.85% |
| Vision-OPD-Qwen3.5-4B-official | Vstar | 0/191 | 0/191 | pending |
| Vision-OPD-Qwen3.5-4B-official | ZoomBench | 0/845 | 0/845 | pending |
| Vision-OPD-Qwen3.5-4B-official | HR-Bench-4K | 0/800 | 0/800 | pending |
| Vision-OPD-Qwen3.5-4B-official | HR-Bench-8K | 0/800 | 0/800 | pending |
| Vision-OPD-Qwen3.5-4B-official | MME-RealWorld-EN | 0/23609 | 0/23609 | pending |
| Vision-OPD-Qwen3.5-4B-official | MME-RealWorld-CN | 0/5462 | 0/5462 | pending |
| Qwen3.5-9B-baseline-official | Vstar | 191/191 | 0/191 | pending |
| Qwen3.5-9B-baseline-official | ZoomBench | 845/845 | 0/845 | pending |
| Qwen3.5-9B-baseline-official | HR-Bench-4K | 800/800 | 0/800 | pending |
| Qwen3.5-9B-baseline-official | HR-Bench-8K | 800/800 | 0/800 | pending |
| Qwen3.5-9B-baseline-official | MME-RealWorld-EN | 23609/23609 | 0/23609 | pending |
| Qwen3.5-9B-baseline-official | MME-RealWorld-CN | 5462/5462 | 0/5462 | pending |
| Vision-OPD-Qwen3.5-9B-official | Vstar | 0/191 | 0/191 | pending |
| Vision-OPD-Qwen3.5-9B-official | ZoomBench | 0/845 | 0/845 | pending |
| Vision-OPD-Qwen3.5-9B-official | HR-Bench-4K | 0/800 | 0/800 | pending |
| Vision-OPD-Qwen3.5-9B-official | HR-Bench-8K | 0/800 | 0/800 | pending |
| Vision-OPD-Qwen3.5-9B-official | MME-RealWorld-EN | 0/23609 | 0/23609 | pending |
| Vision-OPD-Qwen3.5-9B-official | MME-RealWorld-CN | 0/5462 | 0/5462 | pending |

## Historical 8x1 Ablation

These scores are retained separately after a complete strict 96x8 result replaces
the corresponding Local OPD column in the main table.

| Benchmark | Historical Local OPD 4B | Historical Local OPD 9B |
| --- | ---: | ---: |
| Vstar | pending | pending |
| ZoomBench | pending | pending |
| HR-Bench-4K | pending | pending |
| HR-Bench-8K | pending | pending |
| MME-RealWorld-EN | pending | pending |
| MME-RealWorld-CN | pending | pending |

## Locked Evaluation Configuration

- Official source commit: `c2e345fcab10c806ba83e2ec6e1e246d73e7aba2`.
- Seed: `42`; temperature: `0`; thinking: disabled.
- Inference `max_tokens=32768`, `max_retries=3`, `parallel_workers=256`.
- Judge: `openai/gpt-oss-120b`, using the unmodified official `judge_qwenlm.py`.
- Benchmarks: 10 goal targets: the paper core six plus MMStar, POPE, CV-Bench, and MMVP.

## Training Status

The existing paper-explicit checkpoints used local `train_batch_size=8` and `rollout.n=1`; they are retained as an ablation and are not a complete reproduction of the released `batch=96`, `rollout.n=8` defaults. A new memory-adapted run must preserve the released global batch and rollout semantics before it can be described as a strict released-code reproduction.
The main Local OPD 4B column currently reads `Vision-OPD-Qwen3.5-4B-official`; the 9B column reads `Vision-OPD-Qwen3.5-9B-official`.
