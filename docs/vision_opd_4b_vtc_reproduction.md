# Vision-OPD-4B Official and VTC-Bench Reproduction

Generated: 2026-07-20T13:01:19.050117+00:00

## Official Benchmark Alignment

| Benchmark | Paper Baseline 4B | Local Baseline 4B | Paper OPD-4B | Local OPD-4B |
| --- | ---: | ---: | ---: | ---: |
| Vstar | 84.29% | 82.72% | 92.15% | 90.58% |
| ZoomBench | 47.69% | 48.88% | 59.76% | 56.57% |
| HR-Bench-4K | 84.38% | 87.25% | 84.50% | 80.25% |
| HR-Bench-8K | 80.13% | 83.00% | 80.38% | 77.75% |
| MME-RealWorld-EN | 63.86% | 63.85% | 74.88% | pending |
| MME-RealWorld-CN | 63.70% | 64.85% | 70.76% | pending |
| MMStar | N/R | 79.80% | N/R | pending |
| POPE-Test | N/R | 89.27% | N/R | pending |
| CV-Bench | N/R | 87.21% | N/R | pending |
| MMVP | N/R | 77.00% | N/R | pending |
| Core-six Macro | 70.68% | 71.76% | 77.07% | pending |
| Local 10-benchmark unweighted mean | N/R | 76.38% | N/R | pending |

## Alignment Verdict

The official core-six evaluation is still incomplete. The final alignment verdict will be generated after all inference and GPT-OSS judging artifacts validate.

## Inference Behavior

Character length is reported because the selected checkpoint produced unusually long non-thinking answers on some MME-RealWorld samples under the official 32,768-token cap.

| Model | Benchmark | Rows | P95 characters | Max characters | Answers >10k chars |
| --- | --- | ---: | ---: | ---: | ---: |
| Local baseline 4B | MME-RealWorld-EN | 23609 | 1934 | 153522 | 282 |
| Local baseline 4B | MME-RealWorld-CN | 5462 | 433 | 60847 | 37 |
| Local OPD-4B | MME-RealWorld-EN | 2919 | 1936 | 28018 | 21 |
| Local OPD-4B | MME-RealWorld-CN | 0 | 0 | 0 | 0 |

## Experiment Selection

The final local column uses the user-selected one-epoch `released-b96-r8-gradaccum-sp4` checkpoint at global step 65. It preserves 96 prompts/update, 8 rollouts/prompt, 65 updates, and 6,240 prompts. Its rollout tensor parallelism is TP4 rather than the released TP1 setting; this is a documented runtime deviation and can change sampled trajectories even though the global optimization batch is preserved.

## VTC-Bench

| Track | Inference | Overall |
| --- | ---: | ---: |
| Code-driven | 0/680 | pending |
| Interface-driven | 0/680 | pending |

| Category | Code-driven | Interface-driven |
| --- | ---: | ---: |

## Locked Configuration

- Frozen official eval source commit: `c2e345fcab10c806ba83e2ec6e1e246d73e7aba2`.
- Official source gate: the Git blobs for `run_eval.sh`, `infer.py`, `judge_qwenlm.py`, `cal_acc.py`, and `prepare_data.py` must match the frozen commit.
- Training: 6,240 prompts, one epoch, 65 global updates, 96 prompts/update, 8 rollouts/prompt.
- Training objective: VOPD Top-K JSD, alpha 0.5, Top-K 100, EMA teacher rate 0.05.
- Official inference: pristine `eval/run_eval.sh`, seed 42, temperature 0, thinking disabled, max tokens 32768, 256 workers.
- Official scope: 10 benchmark names (core six plus MMStar, POPE, CV-Bench, and MMVP), 45,145 inference rows per model; paper alignment gate remains the six benchmarks reported in the Vision-OPD main table.
- 10-benchmark contract SHA-256: `3ab0eadd256dcd333f9b5a2baf3aedb6c932b921a9c2d51097e5c8851147afcd`.
- Official judge: `openai/gpt-oss-120b` with the pristine `judge_qwenlm.py`.
- VTC generation: temperature 0.6, top-p 0.95, top-k 20, seed 1234, max tokens 40960, 30 workers.
- VTC serving: vLLM DP8/TP1, context 65536, thinking enabled, Qwen3 reasoning parser, and Qwen3-Coder native tool-call parser.
- VTC code track: `code_interpreter`; interface track: all 35 OpenCV tools.
- VTC code YAML SHA-256: `405d9d97bea4b6f150ace4e93731c244d6c19309175eaa348fbf2a7226d65ad9`.
- VTC interface YAML SHA-256: `c927b549a155be17ae590d3dd67cf6375086f0e9ffa62410726055d7f1038149`.
- Canonical repaired VTC GT SHA-256: `a17d9dd82dea023abafe1421027e6bc3f5ba5967c602013087093bed9da571a6`.

## Artifacts

- Official results: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/benchmark/official_reproduction_20260717/results`
- 10-benchmark contract: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/benchmark/official_reproduction_20260717/goal_4b_benchmarks.json`
- Selected checkpoint: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/checkpoints/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4/global_step_65`
- Selected merged model: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/merged_models/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4`
- Final goal audit log: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/logs/vision_opd_4b_goal_completion_audit.log`
- Final goal audit marker: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/outputs/vision_opd_4b_goal_audit_complete`
- VTC code score: `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/VLMEvalKit/outputs/VTC_Bench/Qwen-Agent-Code-RawAPI-Instruct-Vision-OPD-Qwen3.5-4B-released-b96-r8-official/Vision-OPD-Qwen3.5-4B-released-b96-r8-official_VTC_Bench_score.csv`
- VTC interface score: `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/VLMEvalKit/outputs/VTC_Bench/Qwen-Agent-Interface-RawAPI-Instruct-Vision-OPD-Qwen3.5-4B-released-b96-r8-official/Vision-OPD-Qwen3.5-4B-released-b96-r8-official_VTC_Bench_score.csv`
