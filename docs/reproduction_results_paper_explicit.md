# Vision-OPD Paper-Explicit 4B/9B Reproduction Results

Generated: 2026-07-16T20:35:57.122073+00:00

## Scope and configuration

This run follows the values stated in the paper's Experimental Settings: all
6241 Vision-OPD-6K records for one epoch, JSD beta 0.5, top-K 100, EMA
teacher regularization, non-thinking mode, and maximum generation length
1024. The prompt limit is 8192 and overlong prompts are not filtered.

Batch size 8 and rollout count 1 are local execution choices because the PDF
does not specify them. Both sizes use activation offload, Ulysses sequence
parallel size 4, rollout tensor parallel size 4, and rollout GPU memory
utilization 0.30 to fit the local 8x46GB L40S node. These memory-layout
choices do not change the paper-specified objective or generation cap.

## Runtime environment

| Component | Version |
| --- | --- |
| Python | 3.12.13 |
| torch | 2.10.0 |
| transformers | 5.5.0 |
| vllm | 0.18.0 |
| ray | 2.53.0 |
| verl | 0.7.0.dev0 |

Hardware: 8 NVIDIA L40S GPUs with approximately 46 GB memory per GPU.

The local Qwen3.5-4B source contains 738 indexed tensors across 2 safetensors
shards, and Qwen3.5-9B contains 775 indexed tensors across 4 shards. All
indexed shard headers and required tokenizer/processor artifacts validate
successfully, so ModelScope re-download was not required.

## Training status

| Model | Latest checkpoint step | Expected local steps | Merged model |
| --- | ---: | ---: | --- |
| Qwen3.5-4B | 780 | 780 | ready |
| Qwen3.5-9B | 350 | 780 | pending |

## Training diagnostics

| Model | Logged steps | Maximum response | Steps hitting response cap | Steps hitting prompt cap | Peak allocated GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 780 | 1024 | 16 (0.125 max ratio) | 0 | 37.36 GB |
| Qwen3.5-9B | 356 | 575 | 0 (0.000 max ratio) | 0 | 49.20 GB |

Response-cap hits are reported as observed generation truncation at the
configured maximum, not as infrastructure failures. The paper-explicit profile
uses the paper's 1024-token generation cap; the low-memory diagnostic profile
uses its documented 48-token cap.

| Model | First-50 VOPD loss | Last-50 VOPD loss | Loss change | First-50 grad norm | Last-50 grad norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 0.054567 | 0.002512 | -0.052056 | 3.9253 | 0.3528 |
| Qwen3.5-9B | 0.076621 | 0.006667 | -0.069954 | 17.9343 | 1.5814 |

The windowed loss statistics describe optimization behavior; benchmark score
changes remain the evidence for downstream model quality.

## Prepared benchmarks

| Benchmark | Records |
| --- | ---: |
| Vstar | 191 |
| ZoomBench | 845 |
| HR-Bench-4K | 800 |
| HR-Bench-8K | 800 |
| MME-RealWorld-EN | 23609 |
| MME-RealWorld-CN | 5462 |

## Evaluation protocol

Inference uses the merged checkpoint through a local OpenAI-compatible vLLM
server with Qwen3.5 thinking disabled. Multiple-choice answers are graded by
a deterministic option parser, including explicit Yes/No handling for
parseable wrong options. ZoomBench numeric questions use deterministic numeric
matching. Only outputs that cannot be parsed by these rules use an LLM
fallback. The paper's `openai/gpt-oss-120b` judge is not installed on this
node, so every fallback record stores the actual local `judge_model` used.

| Model | Judged records | Deterministic | LLM fallback and source breakdown | Fallback judge models |
| --- | ---: | ---: | --- | --- |
| Qwen3.5-4B | 31707 | 31466 | 241 (llm=241, mcq_option=31291, numeric_exact=175) | Qwen3.5-4B-base-judge=238, Vision-OPD-Qwen3.5-4B-paper-explicit=3 |
| Qwen3.5-9B | pending | pending | pending | pending |

If the LLM fallback count is zero, the unavailable paper judge does not affect
the reported scores. Otherwise, its count is an explicit evaluation-protocol
limitation.

HR-Bench score files retain the official four-cycle breakdown; because each
cycle contains 200 records, their mean equals overall accuracy across 800
records. MME-RealWorld score files retain task, category, and level-2 category
breakdowns and use VLMEvalKit's overall sample-accuracy definition.

The serving context is 32768 tokens. This covers the 16777216-pixel processor
limit (about 16384 merged visual tokens) used by the largest HR-Bench 8K
images. The 4B model uses TP1; MME-RealWorld inference was resumed with eight
TP1 data-parallel replicas and 64 API workers after the single-replica
throughput proved insufficient. The planned 9B serving layout uses TP2.

## Score comparison

| Benchmark | 4B | 9B | 9B - 4B (points) |
| --- | ---: | ---: | ---: |
| Vstar | 59.16% | pending | pending |
| ZoomBench | 44.38% | pending | pending |
| HR-Bench-4K | 63.62% | pending | pending |
| HR-Bench-8K | 59.38% | pending | pending |
| MME-RealWorld-EN | 55.57% | pending | pending |
| MME-RealWorld-CN | 60.31% | pending | pending |
| Macro average | 57.07% | pending | pending |

The completed 4B run has a six-benchmark macro average of 57.07%, which is -20.00 points relative to the paper's Vision-OPD 4B value. The 9B model-size comparison remains pending.

## Paper reference

Table 1 of the paper reports the following Qwen3.5 baselines and Vision-OPD
results under the authors' protocol:

| Benchmark | Qwen 4B | OPD 4B | 4B gain | Qwen 9B | OPD 9B | 9B gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vstar | 84.29% | 92.15% | +7.86 | 82.72% | 94.76% | +12.04 |
| ZoomBench | 47.69% | 59.76% | +12.07 | 52.07% | 65.80% | +13.73 |
| HR-Bench-4K | 84.38% | 84.50% | +0.12 | 85.75% | 88.13% | +2.38 |
| HR-Bench-8K | 80.13% | 80.38% | +0.25 | 80.63% | 85.50% | +4.87 |
| MME-RealWorld-EN | 63.86% | 74.88% | +11.02 | 71.40% | 73.40% | +2.00 |
| MME-RealWorld-CN | 63.70% | 70.76% | +7.06 | 67.67% | 70.46% | +2.79 |
| Macro average | 70.68% | 77.07% | +6.39 | 73.37% | 79.68% | +6.31 |

The published macro gains are +6.39 points for 4B and +6.31 points for 9B.
The local 4B macro average differs from the paper's Vision-OPD 4B value by -20.00 points; the local 9B comparison is pending. The local workflow does not rerun the untrained
Qwen3.5 baselines, and its fallback judge differs from the paper, so the paper
rows are reference targets rather than directly comparable local controls.

## Raw benchmark results

### Vision-OPD-Qwen3.5-4B-paper-explicit_seed42_hrbench-4k

```text
cycle/0: 134/200 = 67.00%
cycle/1: 127/200 = 63.50%
cycle/2: 120/200 = 60.00%
cycle/3: 128/200 = 64.00%
category/cross: 207/400 = 51.75%
category/single: 302/400 = 75.50%
hrbench-4k Acc: 509/800 = 63.62%
```

### Vision-OPD-Qwen3.5-4B-paper-explicit_seed42_hrbench-8k

```text
cycle/0: 125/200 = 62.50%
cycle/1: 122/200 = 61.00%
cycle/2: 112/200 = 56.00%
cycle/3: 116/200 = 58.00%
category/cross: 191/400 = 47.75%
category/single: 284/400 = 71.00%
hrbench-8k Acc: 475/800 = 59.38%
```

### Vision-OPD-Qwen3.5-4B-paper-explicit_seed42_mme-realworld-cn

```text
l2/Attention_TrafficSignal: 59/100 = 59.00%
l2/Attribute_Motion_MultiPedestrians: 28/100 = 28.00%
l2/Attribute_Motion_MultiVehicles: 34/100 = 34.00%
l2/Attribute_Motion_Pedestrian: 25/100 = 25.00%
l2/Attribute_Motion_Vehicle: 54/100 = 54.00%
l2/Attribute_Visual_TrafficSignal: 65/100 = 65.00%
l2/Character Identification: 51/100 = 51.00%
l2/Diagram: 557/602 = 92.52%
l2/Object_Count: 25/100 = 25.00%
l2/Objects_Identify: 39/100 = 39.00%
l2/Prediction_Intention_Ego: 13/100 = 13.00%
l2/Prediction_Intention_Pedestrian: 21/100 = 21.00%
l2/Prediction_Intention_Vehicle: 22/100 = 22.00%
l2/Relation_Interaction_Ego2Pedestrian: 28/100 = 28.00%
l2/Relation_Interaction_Ego2TrafficSignal: 29/100 = 29.00%
l2/Relation_Interaction_Ego2Vehicle: 25/100 = 25.00%
l2/Relation_Interaction_Other2Other: 14/100 = 14.00%
l2/Scene Understanding: 51/107 = 47.66%
l2/Table: 496/602 = 82.39%
l2/adver_and_product: 402/515 = 78.06%
l2/book_map_poster: 321/420 = 76.43%
l2/calculate: 26/100 = 26.00%
l2/intention: 20/43 = 46.51%
l2/license: 54/83 = 65.06%
l2/person/attribute/color: 20/88 = 22.73%
l2/person/attribute/orientation: 1/12 = 8.33%
l2/person/counting: 20/100 = 20.00%
l2/phone_and_address: 348/408 = 85.29%
l2/text_recog: 376/482 = 78.01%
l2/vehicle/attribute/color: 17/71 = 23.94%
l2/vehicle/attribute/orientation: 7/29 = 24.14%
l2/vehicle/counting: 18/100 = 18.00%
l2/vehicle/location: 28/100 = 28.00%
category/Perception/Autonomous_Driving: 270/700 = 38.57%
category/Perception/Diagram and Table: 570/602 = 94.68%
category/Perception/Monitoring: 111/500 = 22.20%
category/Perception/OCR with Complex Context: 1501/1908 = 78.67%
category/Reasoning/Autonomous_Driving: 211/800 = 26.38%
category/Reasoning/Diagram and Table: 483/602 = 80.23%
category/Reasoning/Monitoring: 46/143 = 32.17%
category/Reasoning/OCR with Complex Context: 102/207 = 49.28%
task/Perception: 2452/3710 = 66.09%
task/Reasoning: 842/1752 = 48.06%
mme-realworld-cn Acc: 3294/5462 = 60.31%
```

### Vision-OPD-Qwen3.5-4B-paper-explicit_seed42_mme-realworld

```text
l2/Attention_TrafficSignal: 112/217 = 51.61%
l2/Attribute_Motion_MultiPedestrians: 129/493 = 26.17%
l2/Attribute_Motion_MultiVehicles: 249/823 = 30.26%
l2/Attribute_Motion_Pedestrain: 55/164 = 33.54%
l2/Attribute_Motion_Vehicle: 87/158 = 55.06%
l2/Attribute_Visual_TrafficSignal: 88/201 = 43.78%
l2/Character Identification: 114/250 = 45.60%
l2/Object_Count: 211/720 = 29.31%
l2/Objects_Identify: 452/1101 = 41.05%
l2/Person/counting: 13/28 = 46.43%
l2/Prediction_Intention_Ego: 64/304 = 21.05%
l2/Prediction_Intention_Pedestrian: 23/103 = 22.33%
l2/Prediction_Intention_Vehicle: 57/207 = 27.54%
l2/Relation_Interaction_Ego2Pedestrain: 27/106 = 25.47%
l2/Relation_Interaction_Ego2TrafficSignal: 31/105 = 29.52%
l2/Relation_Interaction_Ego2Vehicle: 26/101 = 25.74%
l2/Relation_Interaction_Other2Other: 45/201 = 22.39%
l2/Scene Understanding: 137/250 = 54.80%
l2/Vehicle/counting: 239/255 = 93.73%
l2/adver_and_product: 1154/1558 = 74.07%
l2/book_map_poster: 1221/1555 = 78.52%
l2/calculate: 71/300 = 23.67%
l2/color: 527/1255 = 41.99%
l2/count: 258/1226 = 21.04%
l2/diagram: 1304/1589 = 82.06%
l2/intention: 30/98 = 30.61%
l2/license: 654/852 = 76.76%
l2/person/attribute/color: 21/89 = 23.60%
l2/person/attribute/orientation: 4/19 = 21.05%
l2/person/counting: 249/964 = 25.83%
l2/phone_and_address: 429/577 = 74.35%
l2/position: 666/1257 = 52.98%
l2/property: 27/100 = 27.00%
l2/table: 3242/4344 = 74.63%
l2/text_recog: 933/1198 = 77.88%
l2/vehicle/attribute/color: 33/197 = 16.75%
l2/vehicle/attribute/orientation: 47/155 = 30.32%
l2/vehicle/counting: 72/353 = 20.40%
l2/vehicle/location: 18/136 = 13.24%
category/Perception/Autonomous_Driving: 1271/3660 = 34.73%
category/Perception/Diagram and Table: 4240/5433 = 78.04%
category/Perception/Monitoring: 696/2196 = 31.69%
category/Perception/OCR with Complex Context: 4391/5740 = 76.50%
category/Perception/Remote Sensing: 1451/3738 = 38.82%
category/Reasoning/Autonomous_Driving: 385/1344 = 28.65%
category/Reasoning/Diagram and Table: 306/500 = 61.20%
category/Reasoning/Monitoring: 128/498 = 25.70%
category/Reasoning/OCR with Complex Context: 251/500 = 50.20%
task/Perception: 12049/20767 = 58.02%
task/Reasoning: 1070/2842 = 37.65%
mme-realworld Acc: 13119/23609 = 55.57%
```

### Vision-OPD-Qwen3.5-4B-paper-explicit_seed42_vstar

```text
direct_attributes: 55.65% (n=115)
relative_position: 64.47% (n=76)
vstar Acc: 59.16% (n=191)
```

### Vision-OPD-Qwen3.5-4B-paper-explicit_seed42_zoombench

```text
zoombench Acc: 375/845 = 44.38%
```
