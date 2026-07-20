# Vision-OPD 4B/9B Reproduction Results

Generated: 2026-07-15T23:06:19.200204+00:00

## Scope and configuration

The paper explicitly specifies Vision-OPD-6K, one epoch, JSD beta 0.5, top-K
100, EMA teacher regularization, non-thinking mode, and maximum generation
length 1024. Batch 96 and rollout n=8 are released-code defaults rather than
values stated in the PDF. The full released-code configuration was tested on
the local 8x46GB L40S node but exceeded available memory during actor update.
The completed local runs retain the paper's loss, teacher, and epoch choices,
but reduce batch to 8, rollout count to 1, and response length to 48. Prompts
longer than 6144 tokens are filtered with the multimodal processor after an
observed 7880-token sample; raising the full sequence budget to 8192 caused
actor-backward OOM. Results from this local-memory configuration must not be
presented as a parameter-identical reproduction of the paper.

The 9B continuation additionally uses activation offload, blocking optimizer
offload copies, and Ulysses sequence parallel size 2. This removed the repeated
step-31 actor-backward OOM on the 8x46GB L40S host. Rollout serving uses tensor
parallel size 4 and GPU memory utilization 0.30 for the 9B training run.

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

## Training status

| Model | Latest checkpoint step | Expected local steps | Merged model |
| --- | ---: | ---: | --- |
| Qwen3.5-4B | 779 | 779 | ready |
| Qwen3.5-9B | 60 | 779 | pending |

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
the evaluator's deterministic option parser. Remaining free-form answers are
judged by the same trained checkpoint through the local API because the
paper's `openai/gpt-oss-120b` judge is not available on this node. This judge
difference is a reproducibility limitation.

The serving context is 32768 tokens. This covers the 16777216-pixel processor
limit (about 16384 merged visual tokens) used by the largest HR-Bench 8K
images. The 4B model is served with tensor parallel size 1 and 9B with size 2.

## Score comparison

| Benchmark | 4B | 9B | 9B - 4B (points) |
| --- | ---: | ---: | ---: |
| Vstar | pending | pending | pending |
| ZoomBench | pending | pending | pending |
| HR-Bench-4K | pending | pending | pending |
| HR-Bench-8K | pending | pending | pending |
| MME-RealWorld-EN | pending | pending | pending |
| MME-RealWorld-CN | pending | pending | pending |
| Macro average | pending | pending | pending |

Effect analysis is pending until both merged models have completed all six benchmarks. The final comparison will report per-benchmark and macro-average deltas.

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
Local-to-paper comparison is pending until all local scores are available. The local workflow does not rerun the untrained
Qwen3.5 baselines, and its fallback judge differs from the paper, so the paper
rows are reference targets rather than directly comparable local controls.

## Raw benchmark results

No completed benchmark result files were found.
