# Vision-OPD-4B Official and VTC-Bench Reproduction

Generated: 2026-07-22T16:22:40.722941+00:00

## Progress Snapshot

| Stage | Completed | State |
| --- | ---: | --- |
| Official baseline 4B | 10/10 benchmarks | complete |
| Official OPD-4B | 10/10 benchmarks | complete |
| VTC code-driven | 643/680 | in progress, scoring pending |
| VTC interface-driven | 670/680 | in progress, scoring pending |
| VTC combined | 1313/1360 (96.54%) | in progress, scoring pending |
| Local OPD-4B Base | 0/680 | pending |
| Local Qwen3.5-4B Base | 0/680 | pending |
| Local Qwen3.5-9B Base | 0/680 | pending |

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
| Code-driven | 643/680 | pending |
| Interface-driven | 670/680 | pending |

### Base (Direct, No Tool)

VTC-Bench Table 4 uses `Base` for direct visual question answering without tool calls. The paper does not report Qwen3.5-4B or Qwen3.5-9B, so the three local rows below are new backbone-matched measurements rather than claimed paper reproductions.

| Model | Inference | Overall | Serving topology |
| --- | ---: | ---: | --- |
| Local OPD-4B Base | 0/680 | pending | DP8 / TP1 |
| Local Qwen3.5-4B Base | 0/680 | pending | DP8 / TP1 |
| Local Qwen3.5-9B Base | 0/680 | pending | DP4 / TP2 |

### Base Protocol and Qwen3.5 Adaptation

The local Base implementation makes exactly one multimodal chat-completion request with the original image and `functions=[]`. It does not instantiate a tool, append a reference trajectory, or enter the multi-round function-calling loop. The result audit requires 680 unique rows, one user turn per row, no tool/function messages, no `function_call`, and a valid official heuristic score CSV.

The paper's Qwen3-VL Thinking recipe is used because all three local Qwen3.5 runs explicitly enable thinking. This is closer than the paper's Instruct recipe (temperature 0.7, top-p 0.8, presence penalty 1.5, max tokens 16,384, seed 3407). Qwen3.5 is not treated as numerically interchangeable with Qwen3-VL: it uses its own tokenizer, processor, native chat template and reasoning format, so model-family differences remain part of the measured result.

| Parameter | Locked value |
| --- | --- |
| System prompt | VTC-Bench Strong System Prompt |
| User prompt | Original image/question/path/size; no GT toolchain |
| Tools/functions | Empty (`tools.enabled=[]`, API `functions=[]`) |
| Reference trajectory | Forbidden and absent |
| Thinking | `enable_thinking=true`, fixed by vLLM default chat-template kwargs |
| Sampling | temperature 0.6, top-p 0.95, top-k 20 |
| Penalties | repetition 1.0, presence 0 |
| Output / seed | max tokens 40,960; seed 1234 |
| Evaluator | 30 workers, resume enabled, up to 20 full-run attempts |
| Server context | 65,536 tokens; sufficient for one image plus 40,960 output tokens |
| Processor | Qwen-Agent image base64 adapter; max short side 1,080; then each model's native Qwen3.5 processor |
| Chat template | Explicit original model-native Qwen3.5 Jinja file |
| vLLM | prefix caching, Qwen3 reasoning parser, trust remote code, GPU utilization 0.90 |

Exact Strong System Prompt:

```text
Your role is that of a helpful assistant specialized in solving real-world visual problems using image processing tools. Answer questions about images by combining your visual understanding with the precision of available tools.

Please follow this structured thinking process and show your work.

Start an iterative loop for each question:

- **First, look closely:** Begin with a detailed description of the image in the context of the user's real-life query. List what is immediately visible, and explicitly identify what specific visual details need to be clarified, adjusted, or measured using tools.
- **Next, apply tools:** Select and invoke just one appropriate function to process the image.
- **Then, review the findings:** Carefully analyze the tool's output (e.g., the processed image, detected features, or status) and decide on your next action (e.g., did the rotation fix the view? do you need further adjustments?).

Continue this loop until you have sufficient information.

To finish, bring everything together in a clear, synthesized answer that fully responds to the user's question.

Image Path **MUST** be Absolute Path. And only **ONE** tool can be used at a time.
```

Exact User Prompt template (without GT Toolchains):

```text
<image>

{question}

### User Image Path: "{image_path}"

### User Image Size: "{image_size}"

### **Output Format (strict adherence required):**

<think>Your detailed reasoning process, should go here.</think>

<answer>Your final answer to the user's question goes here.</answer>
```

Per-model reproducibility artifacts:

| Model | Model path | Processor config SHA-256 | Native chat template SHA-256 | Server |
| --- | --- | --- | --- | --- |
| Local OPD-4B Base | `/data00/users/wanglikun/ProjWormLK/Vision-OPD/merged_models/Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4` | `d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` | DP8 / TP1, context 65,536 |
| Local Qwen3.5-4B Base | `/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-4B` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` | DP8 / TP1, context 65,536 |
| Local Qwen3.5-9B Base | `/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-9b` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` | DP4 / TP2, context 65,536 |

Config and score paths:

- Local OPD-4B Base: config `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/eval_config/vision_opd_qwen35_4b_base.yaml` (SHA-256 `ded3a6392b14bb6c3dfb622748e8bb1df0844cb3b030ac3452145ff2e653bd8c`), score `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/VLMEvalKit/outputs/VTC_Bench/Qwen-Agent-Base-RawAPI-Instruct-Vision-OPD-Qwen3.5-4B-released-b96-r8-base/Vision-OPD-Qwen3.5-4B-released-b96-r8-base_VTC_Bench_score.csv`.
- Local Qwen3.5-4B Base: config `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/eval_config/qwen35_4b_base.yaml` (SHA-256 `e717110b3dd0e059a84bce5a21ce6185006026d0cfa988914541cc10d6e9fab9`), score `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/VLMEvalKit/outputs/VTC_Bench/Qwen-Agent-Base-RawAPI-Instruct-Qwen3.5-4B-base-vtc/Qwen3.5-4B-base-vtc_VTC_Bench_score.csv`.
- Local Qwen3.5-9B Base: config `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/eval_config/qwen35_9b_base.yaml` (SHA-256 `7cefed980018c7e04e4cca619b06501b33ef8ef99696f587f4001cab99179f54`), score `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/VLMEvalKit/outputs/VTC_Bench/Qwen-Agent-Base-RawAPI-Instruct-Qwen3.5-9B-base-vtc/Qwen3.5-9B-base-vtc_VTC_Bench_score.csv`.

### Partial Heuristic Snapshot

Snapshot generated at `2026-07-22T07:55:06.883924+00:00` from the latest cumulative JSONL files. Scoring reuses the public `VTCBenchDataset.evaluate(..., model="exact_matching")` path; both track results were independently checked against direct calls to the same official per-item rule.

Only resume-valid completed rows are included. Unresolved, malformed, empty, or explicitly invalid answers are excluded, so these values have tail-selection bias and must not be reported as final 680-row VTC-Bench scores.

| Track | Raw rows | Resume-valid/scored | Correct | Coverage | Partial overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code-driven | 593 | 590 | 96 | 86.76% | 16.27% |
| Interface-driven | 662 | 659 | 119 | 96.91% | 18.06% |
| Combined track-samples | 1255 | 1249 | 215 | 91.84% | 17.21% |

| Category | Code rows | Code correct | Code partial | Interface rows | Interface correct | Interface partial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention | 41 | 4 | 9.76% | 45 | 5 | 11.11% |
| chart | 84 | 20 | 23.81% | 96 | 23 | 23.96% |
| color | 87 | 12 | 13.79% | 89 | 22 | 24.72% |
| counting | 83 | 5 | 6.02% | 84 | 7 | 8.33% |
| math | 86 | 17 | 19.77% | 102 | 15 | 14.71% |
| measure | 77 | 14 | 18.18% | 98 | 16 | 16.33% |
| ocr | 47 | 7 | 14.89% | 50 | 6 | 12.00% |
| perceptual | 45 | 6 | 13.33% | 50 | 4 | 8.00% |
| spatial | 40 | 11 | 27.50% | 45 | 21 | 46.67% |

Machine-readable snapshot: `benchmark/vtc_partial_20260722/vision_opd_4b_partial_scores.json`. Code JSONL SHA-256: `471a6f0669cedf5dbd563963f5cb1e962a66159e93d40b2ada9d89fbb5297d90`; interface JSONL SHA-256: `fb7a2539d2045da09a023911a933821a912d695a515668ac89c1d6f33d64ef95`.

On these scored subsets, interface-driven is +1.79 pp above code-driven overall. Its largest observed advantages are spatial (+19.17 pp) and color (+10.93 pp); code-driven is higher on perceptual (+5.33 pp) and math (+5.06 pp). Because category coverage differs between tracks and unresolved tail rows are excluded, these deltas are descriptive rather than final track comparisons.

### Runtime Diagnostics

These counters are cumulative snapshots from the active documented run. They diagnose throughput and do not change generation or scoring parameters.

| Track | Completed rows | >10k chars | >100k chars | Max chars | Rows with tool messages |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code-driven | 643 | 13 | 4 | 199549 | 0 |
| Interface-driven | 670 | 4 | 1 | 118667 | 0 |

| Cumulative pipeline signal | Count |
| --- | ---: |
| Successful vLLM requests | 1910 |
| HTTP 400 context-length rejections | 0 |
| Network/read timeout retry messages | 887 |
| Invalid-answer messages | 646 |
| Task-timeout messages | 441 |

The dominant runtime cost is retry amplification around long generations. The client and evaluator task timeouts are 3,600 seconds, and each row permits three evaluator attempts. The base agent protocol permits up to 20 LLM calls per run plus final-format retries; the resumed tail deviation is recorded below. The earlier 65,536-context server rejected requests when the 40,960-token output allowance plus accumulated multimodal/tool context exceeded that limit; the resumed server uses 131,072 and its current HTTP 400 counter is shown above. Zero or few completed rows with tool messages indicates a model tool-use adherence issue rather than a missing tool registration; both parser and tool smoke tests pass.

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
- VTC repeated-no-tool guard: after two consecutive identical assistant responses with no native tool call and no final answer, the wrapper jumps to the agent's existing direct-answer fallback. The upstream default remains unchanged unless `QWEN_AGENT_REPEATED_NO_TOOL_LIMIT=2` is exported.
- VTC tail-call budget deviation: after 896 valid rows had produced zero recorded tool messages, resumed tail samples use `QWEN_AGENT_MAX_LLM_CALL_PER_RUN=4` instead of the upstream 20-call allowance. The first three calls still expose tools and the fourth uses the existing direct-answer fallback. This deadline-driven runtime deviation must be considered when comparing VTC scores to an unmodified 20-call agent protocol.
- VTC final-answer semantic stop: resumed tail samples set `QWEN_AGENT_STOP_ON_FINAL_ANSWER=1`. The configured `max_tokens=40960` remains unchanged; generation stops only after the model emits the required `</answer>` protocol delimiter, which is restored after the OpenAI-compatible API removes its matched stop string. This prevents post-answer repetition without truncating an unfinished answer.
- VTC serving: vLLM DP8/TP1, context 131072, prefix caching enabled, thinking enabled, Qwen3 reasoning parser, and Qwen3-Coder native tool-call parser. The merged model natively supports 262144 tokens; the larger serving limit prevents accumulated tool context plus the fixed output allowance from being rejected.
- VTC code track: `code_interpreter`; interface track: all 35 OpenCV tools.
- VTC Base tracks: direct one-shot original-image inference, no registered tools, no GT trajectory, and strict serial order OPD-4B then baseline 4B then baseline 9B.
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
- Local OPD-4B Base score: `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/VLMEvalKit/outputs/VTC_Bench/Qwen-Agent-Base-RawAPI-Instruct-Vision-OPD-Qwen3.5-4B-released-b96-r8-base/Vision-OPD-Qwen3.5-4B-released-b96-r8-base_VTC_Bench_score.csv`
- Local Qwen3.5-4B Base score: `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/VLMEvalKit/outputs/VTC_Bench/Qwen-Agent-Base-RawAPI-Instruct-Qwen3.5-4B-base-vtc/Qwen3.5-4B-base-vtc_VTC_Bench_score.csv`
- Local Qwen3.5-9B Base score: `/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/VLMEvalKit/outputs/VTC_Bench/Qwen-Agent-Base-RawAPI-Instruct-Qwen3.5-9B-base-vtc/Qwen3.5-9B-base-vtc_VTC_Bench_score.csv`
