# Vision-OPD-4B Official and VTC-Bench Reproduction

Generated: 2026-07-21T16:17:38.643405+00:00

## Progress Snapshot

| Stage | Completed | State |
| --- | ---: | --- |
| Official baseline 4B | 10/10 benchmarks | complete |
| Official OPD-4B | 10/10 benchmarks | complete |
| VTC code-driven | 397/680 | in progress |
| VTC interface-driven | 140/680 | in progress |

## Official Benchmark Alignment

| Benchmark | Paper Baseline 4B | Local Baseline 4B | Paper OPD-4B | Local OPD-4B |
| --- | ---: | ---: | ---: | ---: |
| Vstar | 84.29% | 82.72% | 92.15% | 90.58% |
| ZoomBench | 47.69% | 48.88% | 59.76% | 56.57% |
| HR-Bench-4K | 84.38% | 87.50% | 84.50% | 81.38% |
| HR-Bench-8K | 80.13% | 83.12% | 80.38% | 78.00% |
| MME-RealWorld-EN | 63.86% | 63.89% | 74.88% | 70.35% |
| MME-RealWorld-CN | 63.70% | 64.94% | 70.76% | 70.03% |
| MMStar | 78.53% | 80.00% | 79.60% | 75.07% |
| POPE-Test | 88.28% | 89.27% | 89.14% | 89.47% |
| CV-Bench | 87.13% | 87.21% | 87.27% | 85.68% |
| MMVP | 76.67% | 77.00% | 79.67% | 78.00% |
| Core-six Macro | 70.68% | 71.84% | 77.07% | 74.48% |
| Local 10-benchmark unweighted mean | N/R | 76.45% | N/R | 77.51% |

### Paper Table 2 Hold-out Tasks

The paper defines these four datasets as hold-out tasks that are unseen during Vision-OPD training. Values below are transcribed from Table 2 of [arXiv:2605.18740](https://arxiv.org/pdf/2605.18740).

| Hold-out benchmark | Paper Vanilla 4B | Paper OPD-4B | Paper gain | Local Baseline 4B | Local OPD-4B |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMVP | 76.67% | 79.67% | +3.00 pp | 77.00% | 78.00% |
| CV-Bench | 87.13% | 87.27% | +0.14 pp | 87.21% | 85.68% |
| MMStar | 78.53% | 79.60% | +1.07 pp | 80.00% | 75.07% |
| POPE-Test | 88.28% | 89.14% | +0.86 pp | 89.27% | 89.47% |

## Alignment Verdict

Result: **NOT ALIGNED** under the documented local gate of +/-2.0 pp for the core-six macro and +/-3.0 pp per benchmark.
Local baseline macro is 71.84% (+1.16 pp versus paper baseline); local OPD macro is 74.48% (-2.58 pp versus paper OPD and +2.64 pp versus the local baseline).
Per-benchmark deviations outside the gate: ZoomBench (-3.19 pp), HR-Bench-4K (-3.12 pp), MME-RealWorld-EN (-4.53 pp).

## Inference Behavior

Character length is reported because the selected checkpoint produced unusually long non-thinking answers on some MME-RealWorld samples under the official 32,768-token cap.

| Model | Benchmark | Rows | P95 characters | Max characters | Answers >10k chars |
| --- | --- | ---: | ---: | ---: | ---: |
| Local baseline 4B | MME-RealWorld-EN | 23609 | 1934 | 153522 | 282 |
| Local baseline 4B | MME-RealWorld-CN | 5462 | 433 | 60847 | 37 |
| Local OPD-4B | MME-RealWorld-EN | 23609 | 94037 | 186291 | 1467 |
| Local OPD-4B | MME-RealWorld-CN | 5462 | 421 | 65517 | 161 |

### Interim MME-RealWorld-EN Snapshot

This is a moving partial snapshot, not the final benchmark score. `Rule-direct correct` uses only the deterministic MathRuler/first-option stages of the official judge. Unresolved rows still require GPT-OSS-120B, so the percentage is a conservative lower bound and the incomplete prefix need not be representative of the full dataset.

| Response-length group | Snapshot rows | Rule-direct correct | Lower bound |
| --- | ---: | ---: | ---: |
| all characters | 23609 | 13987 | 59.24% |
| <10k characters | 22142 | 13724 | 61.98% |
| >=10k characters | 1467 | 263 | 17.93% |
| >=50k characters | 1388 | 239 | 17.22% |

The long-response groups have a substantially lower rule-direct success rate. An earlier local diagnostic judge service was limited to 8,192 tokens; those judge/score artifacts are archived and excluded because overlength requests could be recorded as `No` after the official three retries. Final baseline and OPD-4B results use a 65,536-token GPT-OSS context, while preserving the pristine official judge implementation.

## Experiment Selection

The final local column uses the user-selected one-epoch `released-b96-r8-gradaccum-sp4` checkpoint at global step 65. It preserves 96 prompts/update, 8 rollouts/prompt, 65 updates, and 6,240 prompts. Its rollout tensor parallelism is TP4 rather than the released TP1 setting; this is a documented runtime deviation and can change sampled trajectories even though the global optimization batch is preserved.

## VTC-Bench

| Track | Inference | Overall |
| --- | ---: | ---: |
| Code-driven | 397/680 | pending |
| Interface-driven | 140/680 | pending |

### Runtime Diagnostics

These counters are cumulative snapshots from the active strict-configuration run. They diagnose throughput and do not change generation or scoring parameters.

| Track | Completed rows | >10k chars | >100k chars | Max chars | Rows with tool messages |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code-driven | 397 | 6 | 1 | 199549 | 0 |
| Interface-driven | 140 | 0 | 0 | 9977 | 0 |

| Cumulative pipeline signal | Count |
| --- | ---: |
| Successful vLLM requests | 655 |
| HTTP 400 context-length rejections | 0 |
| Network/read timeout retry messages | 471 |
| Invalid-answer messages | 304 |
| Task-timeout messages | 31 |

The dominant runtime cost is retry amplification around long generations. The client and evaluator task timeouts are 3,600 seconds, and each row permits three evaluator attempts. The agent itself permits up to 20 LLM calls per run plus final-format retries. The earlier 65,536-context server rejected requests when the 40,960-token output allowance plus accumulated multimodal/tool context exceeded that limit; the resumed server uses 131,072 and its current HTTP 400 counter is shown above. Zero or few completed rows with tool messages indicates a model tool-use adherence issue rather than a missing tool registration; both parser and tool smoke tests pass.

| Category | Code-driven | Interface-driven |
| --- | ---: | ---: |

## Locked Configuration

- Frozen official eval source commit: `c2e345fcab10c806ba83e2ec6e1e246d73e7aba2`.
- Official source gate: the Git blobs for `run_eval.sh`, `infer.py`, `judge_qwenlm.py`, `cal_acc.py`, and `prepare_data.py` must match the frozen commit.
- Training: 6,240 prompts, one epoch, 65 global updates, 96 prompts/update, 8 rollouts/prompt.
- Training objective: VOPD Top-K JSD, alpha 0.5, Top-K 100, EMA teacher rate 0.05.
- Selected training topology: actor SP4, 2,304 tokens/rank (9,216 per SP group), rollout TP4, rollout GPU utilization 0.30, max_num_seqs 64, layered summon enabled.
- Released topology deviation: rollout TP4 replaces the released TP1 default; the global batch and number of trajectories are unchanged, but sampled trajectories need not be bitwise identical.
- Official inference: pristine `eval/run_eval.sh`, seed 42, temperature 0, thinking disabled, max tokens 32768, 256 workers.
- Official scope: 10 benchmark names (core six plus MMStar, POPE, CV-Bench, and MMVP), 45,145 inference rows per model; paper alignment gate remains the six benchmarks reported in the Vision-OPD main table.
- 10-benchmark contract SHA-256: `3ab0eadd256dcd333f9b5a2baf3aedb6c932b921a9c2d51097e5c8851147afcd`.
- Selected evaluation provenance SHA-256: `68368c8a4b887f2b19ff75ce0b81bf75c647610514aedbeba87b8199e175667b`.
- Official judge: `openai/gpt-oss-120b` with the pristine `judge_qwenlm.py`; judge context 65,536 tokens, sufficient for the official 32,768-token model response cap.
- VTC generation: temperature 0.6, top-p 0.95, top-k 20, seed 1234, max tokens 40960, 30 workers per track.
- VTC scheduling: code-driven and interface-driven run concurrently against one shared DP8 server (60 evaluator workers total); generation and scoring settings are unchanged.
- VTC serving: vLLM DP8/TP1, context 131072, prefix caching enabled, thinking enabled, Qwen3 reasoning parser, and Qwen3-Coder native tool-call parser. The merged model natively supports 262144 tokens; the larger serving limit prevents accumulated tool context plus the fixed output allowance from being rejected.
- VTC code track: `code_interpreter`; interface track: all 35 OpenCV tools.
- VTC code YAML SHA-256: `405d9d97bea4b6f150ace4e93731c244d6c19309175eaa348fbf2a7226d65ad9`.
- VTC interface YAML SHA-256: `c927b549a155be17ae590d3dd67cf6375086f0e9ffa62410726055d7f1038149`.
- Canonical repaired VTC GT SHA-256: `a17d9dd82dea023abafe1421027e6bc3f5ba5967c602013087093bed9da571a6`.

## Artifacts

- Official results: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/benchmark/official_reproduction_20260717/results`
- 10-benchmark contract: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/benchmark/official_reproduction_20260717/goal_4b_benchmarks.json`
- Selected evaluation provenance: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/benchmark/official_reproduction_20260717/provenance/selected_4b_eval_config.json`
- Selected checkpoint: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/checkpoints/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4/global_step_65`
- Selected merged model: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/merged_models/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4`
- Excluded 8,192-context judge diagnostics: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/benchmark/diagnostic_judge_ctx8192_20260720`
- Final goal audit log: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/logs/vision_opd_4b_goal_completion_audit.log`
- Final goal audit marker: `/data00/users/wanglikun/ProjWormLK/Vision-OPD/outputs/vision_opd_4b_goal_audit_complete`
- VTC code score: `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/VLMEvalKit/outputs/VTC_Bench/Qwen-Agent-Code-RawAPI-Instruct-Vision-OPD-Qwen3.5-4B-released-b96-r8-official/Vision-OPD-Qwen3.5-4B-released-b96-r8-official_VTC_Bench_score.csv`
- VTC interface score: `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/VLMEvalKit/outputs/VTC_Bench/Qwen-Agent-Interface-RawAPI-Instruct-Vision-OPD-Qwen3.5-4B-released-b96-r8-official/Vision-OPD-Qwen3.5-4B-released-b96-r8-official_VTC_Bench_score.csv`
