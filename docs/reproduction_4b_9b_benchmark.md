# Vision-OPD 4B/9B Reproduction Plan

Date: 2026-07-14

## Paper Settings

The paper's Experimental Settings section states that Vision-OPD is trained on
6.2K synthetic samples and evaluated on V* Bench, ZoomBench, HR-Bench 4K,
HR-Bench 8K, MME-RealWorld-EN, and MME-RealWorld-CN. The reported backbone
models are Qwen3.5-4B and Qwen3.5-9B. The method uses JSD-based on-policy
self-distillation, top-K distillation with `K=100`, EMA teacher regularization,
maximum on-policy generation length `1024`, non-thinking mode, and one training
epoch. The paper also disables Qwen3.5 thinking mode for training and
evaluation. Batch size `96` and rollout count `8` are defaults in the released
`scripts/run_vision_opd.sh`; they are not stated in the PDF.

| Setting | Local key | Value | Source |
| --- | --- | --- | --- |
| train data | `data.train_files` | Vision-OPD-6K | paper |
| epochs | `trainer.total_epochs` | `1` | paper |
| train batch | `data.train_batch_size` | `96` | released code default |
| rollouts | `actor_rollout_ref.rollout.n` | `8` | released code default |
| max generation length | `data.max_response_length` | `1024` | paper |
| distillation top-K | `actor.self_distillation.distillation_topk` | `100` | paper |
| JSD alpha/beta | `actor.self_distillation.alpha` | `0.5` | paper |
| teacher regularization | `teacher_regularization` | `ema` | paper |
| teacher update rate | `teacher_update_rate` | `0.05` | paper ablation choice |
| reward model | `reward_model.enable` | `False` | method/released code |
| KL reward | `algorithm.use_kl_in_reward` | `False` | method/released code |
| thinking mode | chat template | non-thinking | paper |

`scripts/run_vision_opd_paper_qwen35_4b.sh` fixes the paper values plus the
released-code batch/rollout defaults for the 4B model. On this 8x46GB L40S
node, that full-scale configuration OOMed during actor update, so long local
runs use the low-memory continuation scripts below while preserving the OPD
loss configuration. The reduced generation cap remains a material difference
from the paper and is reported as such.

## Data Status

Training data is present:

```text
/data00/users/wanglikun/ProjWormLK/Vision-OPD/datasets/Vision-OPD-6K/train.jsonl
/data00/users/wanglikun/ProjWormLK/Vision-OPD/datasets/Vision-OPD-6K/train.parquet
```

`data/train.jsonl` and `data/train.parquet` are symlinked to the dataset. The
parquet file has 6241 rows.

## 4B Training

Released-code full-scale command:

```bash
cd /data00/users/wanglikun/ProjWormLK/Vision-OPD
screen -L -Logfile logs/vision_opd_4b_paper_20260714.log \
  -dmS vision_opd_4b_paper_20260714 \
  bash scripts/run_vision_opd_paper_qwen35_4b.sh
```

Local continuation command:

```bash
cd /data00/users/wanglikun/ProjWormLK/Vision-OPD
screen -L -Logfile logs/vision_opd_4b_local_resume_20260714.log \
  -dmS vision_opd_4b_local_resume_20260714 \
  bash scripts/run_vision_opd_local_4b_resume.sh
```

Final 4B FSDP checkpoint and merged Hugging Face model:

```text
checkpoints/Vision-OPD-Qwen3.5-4B-full-repro-lowmem-20260714/global_step_779
merged_models/Vision-OPD-Qwen3.5-4B
```

That run later OOMed near step 109 with the 8192/64 configuration. The current
`run_vision_opd_local_4b_resume.sh` reduces response length to 48 and max prompt
length to 6144. A later batch contained a 7880-token prompt. The local wrapper
therefore enables `FILTER_OVERLONG_PROMPTS=True`; the multimodal processor
filters those records before batching so token IDs and RoPE grids stay aligned.
Increasing the full sequence limit to 8192 had previously OOMed in actor
backward. The released-code full-scale wrapper keeps all prompts and
`truncation=error`.

The processor retained 6232 of 6241 records, producing 779 local steps. This
configuration completed the epoch. The resumed dataloader state at
`global_step_760` had already consumed its iterator and caused an early clean
exit; its `data.pt` was preserved as `data.pt.resume_original` and removed from
the active checkpoint before the final 19-step continuation.

## 9B Training

The local 9B model directory has been validated:

```text
/data00/users/wanglikun/ProjWormLK/MODEL_ZOO/Qwen/Qwen3.5-9b
```

`AutoConfig` and `Qwen3VLProcessor` load successfully. The safetensors index
contains 775 entries, all four referenced shards exist, and every shard header
opens successfully. A ModelScope download is therefore not needed unless a
later full weight load reveals corruption.

Local 9B command:

```bash
cd /data00/users/wanglikun/ProjWormLK/Vision-OPD
screen -L -Logfile logs/vision_opd_9b_local_20260714.log \
  -dmS vision_opd_9b_local_20260714 \
bash scripts/run_vision_opd_local_9b.sh
```

The 9B local run uses the same 6232 retained records and 779 expected steps.
It additionally enables activation offload, blocking optimizer offload copies,
and Ulysses sequence parallel size 2. Rollout uses TP4 with GPU memory
utilization 0.30. Ulysses SP2 removed the repeatable actor-backward OOM after
step 31 and has passed checkpoint save/resume. Read the authoritative current
step from:

```text
checkpoints/Vision-OPD-Qwen3.5-9B-local-lowmem-20260714/latest_checkpointed_iteration.txt
```

## Benchmark Preparation

Prepare benchmark data under the requested path:

```bash
cd /data00/users/wanglikun/ProjWormLK/Vision-OPD
bash scripts/prepare_benchmarks.sh
```

The script creates:

```text
benchmark/raw/
benchmark/prepared/
```

On this host, Hugging Face Xet stalled while fetching MME-RealWorld. The
preparation wrapper disables Xet, uses resumable HTTP with sixteen workers, and
retries each benchmark up to 20 times. The unattended watchdog restarts the
benchmark preparation screen until all six normalized JSON files exist.

It hardlinks or copies local DeepEyes raw datasets for V* and HR-Bench when
available, then uses `eval/prepare_data.py` to generate the normalized JSON
files used by `eval/run_eval.sh`:

```text
vstar.json
zoombench.json
hr_bench_4k.json
hr_bench_8k.json
MME_RealWorld.json
MME_RealWorld_CN.json
```

## Benchmark Inference

Serve a merged checkpoint:

```bash
MODEL_PATH=/path/to/merged/checkpoint \
SERVED_MODEL_NAME=Vision-OPD-4B \
PORT=8000 \
TENSOR_PARALLEL_SIZE=1 \
MAX_MODEL_LEN=32768 \
bash scripts/serve_qwen_vllm.sh
```

The 32K serving context accommodates the processor's 16,777,216-pixel upper
bound (roughly 16K merged visual tokens) on HR-Bench 8K. The unattended
pipeline serves 4B with tensor parallel size 1 and 9B with tensor parallel
size 2 on the L40S node.

The unattended pipeline writes merged models to `merged_models/` instead of
mixing Hugging Face files into the resumable FSDP checkpoint directories.

Run benchmark inference and judging:

```bash
cd /data00/users/wanglikun/ProjWormLK/Vision-OPD
API_BASE=http://127.0.0.1:8000/v1/ \
OPENAI_MODEL_ID=Vision-OPD-4B \
ENABLE_THINKING=False \
BENCHMARK=vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn \
OUT_DIR=benchmark/model_answer \
BENCHMARK_DATA_DIR=benchmark/prepared \
JUDGE_OUT_DIR=benchmark/judge \
RESULTS_DIR=benchmark/results \
bash eval/run_eval.sh
```

`BENCHMARK_DATA_DIR` lets the evaluator consume the normalized files directly;
it does not copy or redownload them under `eval/`. API inference and judge
calls default to 20 retries. `OUT_DIR`, `JUDGE_OUT_DIR`, and `RESULTS_DIR` are
independent paths, so all generated artifacts can stay under `benchmark/`.

## Unattended pipeline

The full local-memory workflow can be left running in a detached session. It
automatically retries failed training processes from the newest checkpoint,
merges the final 4B and 9B checkpoints, evaluates both models, and writes
`docs/reproduction_results.md`:

```bash
screen -L -Logfile logs/auto_reproduction_pipeline_20260714.log \
  -dmS auto_reproduction_pipeline_20260714 \
  bash scripts/auto_reproduction_pipeline.sh
```

For process-level supervision, a second detached session restarts the pipeline
if its screen exits before `outputs/vision_opd_reproduction_complete` exists:

```bash
screen -L -Logfile logs/auto_reproduction_watchdog_20260714.log \
  -dmS auto_reproduction_watchdog_20260714 \
bash scripts/watch_auto_reproduction.sh
```

The pipeline validates each model's six evaluations before stopping its vLLM
retry loop. Before creating `outputs/vision_opd_reproduction_complete`, it also
runs:

```bash
python scripts/verify_reproduction.py --project-root "$PWD" --full
```

This requires step 779 and merged model artifacts for both sizes, complete and
error-free inference/judge records, all twelve score files, valid benchmark
image references, and a generated report without pending entries.

The six target benchmarks use deterministic grading whenever the output can be
parsed: option matching for all multiple-choice records and numeric matching
for ZoomBench counting records. Parseable wrong answers are marked `No`
directly instead of being sent to an LLM. Only unparseable outputs use the
evaluated model as a fallback because the paper's `openai/gpt-oss-120b` judge
is not installed locally. The generated report records the exact fallback
count, so this limitation is measurable rather than assumed.

The normalized HR-Bench manifests retain the official four-way
`cycle_category` metadata, with 200 records in each cycle. MME-RealWorld EN/CN
retain `category` and `l2_category` from the source parquet files. Score files
include these official breakdowns and end with overall sample accuracy; this
matches VLMEvalKit's MME-RealWorld `Overall` definition and is numerically
equivalent to HR-Bench's mean over its four equal-size cycles.
