# Vision-OPD

**Vision-OPD: Learning to See Fine-Grained Details for Multimodal LLMs via On-Policy Self-Distillation**

<p align="center">
📃 <a href="https://arxiv.org/pdf/2605.18740" target="_blank">Paper</a> | 
🤗 <a href="https://huggingface.co/collections/yuanqianhao/vision-opd">Hugging Face</a>
</p>

## Overview

Vision-OPD is a regional-to-global on-policy self-distillation framework that transfers a model's own privileged regional perception to its full-image policy, enabling fine-grained visual understanding in a single forward pass — without external teachers, ground-truth labels, reward verifiers, or inference-time tool use.

<p align="center">
  <img src="figures/average_bar_chart.png" alt="Vision-OPD Average Scores" width="60%"/>
</p>

<p align="center"><i>Average scores across fine-grained visual understanding benchmarks, including V* Bench, ZoomBench,  HR-Bench 4K, HR-Bench 8K, MME-RealWorld-EN and MME-RealWorld-CN.</i></p>

## Environment Setup

```bash
conda create -n vision-opd python=3.12 -y
conda activate vision-opd
pip install --upgrade pip
pip install --no-deps -r requirements.txt
pip install -e . --no-deps
pip install flash-attn --no-build-isolation
pip install causal-conv1d==1.6.1 --no-build-isolation
```

## Local Reproduction

### Local Qwen3.5-4B/9B reproduction

The original 6241-row Vision-OPD-6K dataset is available under
`datasets/Vision-OPD-6K`; `data/train.parquet` and `data/train.jsonl` point to
that source. Local starting models are under `../MODEL_ZOO/Qwen/Qwen3.5-4B`
and `../MODEL_ZOO/Qwen/Qwen3.5-9b` relative to this repository.

The released batch-96, rollout-8 configuration exceeded the 8x46GB L40S node
during actor update. The completed local 4B run and the 9B continuation retain
the paper's VOPD/JSD loss, top-K 100 distillation, EMA teacher, non-thinking
mode, and one epoch, but use batch 8, rollout 1, response length 48, and filter
multimodal prompts above 6144 tokens. These results are therefore a
local-memory reproduction, not a parameter-identical paper reproduction. The
full configuration and deviations are documented in
`docs/reproduction_4b_9b_benchmark.md` and `docs/reproduction_results.md`.

The 4B run completed all 779 local steps and was merged to:

```text
merged_models/Vision-OPD-Qwen3.5-4B
```

The 9B wrapper enables activation offload and Ulysses sequence parallel size 2
to fit actor backward. Its authoritative resumable progress is stored in:

```text
checkpoints/Vision-OPD-Qwen3.5-9B-local-lowmem-20260714/latest_checkpointed_iteration.txt
```

The detached controller automatically resumes training, merges the final 9B
checkpoint, evaluates both models on the six paper benchmarks, and regenerates
the results report:

```bash
screen -ls
# auto_reproduction_pipeline_20260714
# auto_reproduction_watchdog_20260714

tail -f logs/auto_reproduction_pipeline_20260714.log
tail -f logs/vision_opd_9b_auto.log
```

The controller command is:

```bash
screen -L -Logfile logs/auto_reproduction_pipeline_20260714.log \
  -dmS auto_reproduction_pipeline_20260714 \
  bash scripts/auto_reproduction_pipeline.sh
```

`scripts/watch_auto_reproduction.sh` runs in a second detached screen and
restarts the controller if it exits before the completion marker is written.
Before writing that marker, the controller runs
`scripts/verify_reproduction.py --full`. The verifier requires both step-779
checkpoints and merged models, all six manifests and referenced images, exact
inference/judge sample counts for both models, no API-error records, all twelve
score files, and a final report with no pending results.

Benchmark downloads and normalized files are stored under `benchmark/`.
Completed inference, judge outputs, scores, and the final generated report are
written to `benchmark/model_answer`, `benchmark/judge`, `benchmark/results`,
and `docs/reproduction_results.md`, respectively. Benchmark serving uses a
32768-token context so the largest HR-Bench 8K images fit after Qwen3.5 image
processing; 4B uses TP1 and 9B uses TP2 for evaluation.

Run the same final audit manually with:

```bash
python scripts/verify_reproduction.py --project-root "$PWD" --full
```

### Paper-explicit local run

The completed `paper-explicit` run uses all 6241 records for one epoch (780
steps), an 8192-token prompt limit, a 1024-token response limit, JSD beta 0.5,
top-K 100, EMA teacher regularization, and non-thinking mode. It uses local
`batch=8` and `rollout.n=1`, however, whereas the released script defaults are
`batch=96` and `rollout.n=8`. These checkpoints are therefore a useful local
ablation, not a complete reproduction of the released training configuration.

Training and evaluation run in separately supervised detached sessions:

```bash
screen -ls
# paper_explicit_training_pipeline
# paper_explicit_training_watchdog
# paper_explicit_evaluation_pipeline
# paper_explicit_evaluation_watchdog

tail -f logs/vision_opd_4b_paper_explicit.log
tail -f logs/paper_explicit_evaluation_pipeline.log
```

The evaluation controller waits for both step-780 checkpoints and merged
models, then evaluates 4B and 9B on Vstar, ZoomBench, HR-Bench 4K/8K, and
MME-RealWorld EN/CN. It retries resumable inference and judge failures up to
100 times, writes `docs/reproduction_results_paper_explicit.md`, and only
creates `outputs/vision_opd_paper_explicit_reproduction_complete` after the
full audit passes.

These six evaluations use deterministic option grading plus deterministic
numeric grading for ZoomBench counting questions. Only unparseable model
outputs use the configured LLM fallback, and the final report records that
fallback count.

The complete workflow retrospective is continuously regenerated at
`docs/vision_opd_goal_reproduction_report.md`. It records the paper-parameter
audit, failed attempts, training diagnostics, benchmark preparation, scoring
fixes, artifact paths, current progress, and final local-to-paper comparison.
The lightweight `scripts/watch_goal_reproduction_report.sh` process refreshes
it every five minutes until the final completion marker exists.

```bash
python scripts/verify_reproduction.py --project-root "$PWD" \
  --profile paper-explicit --full
```

Paper-explicit merged models are written to:

```text
merged_models/Vision-OPD-Qwen3.5-4B-paper-explicit
merged_models/Vision-OPD-Qwen3.5-9B-paper-explicit
```

### Released batch-96/rollout-8 reproduction

The strict released-code rerun preserves 96 prompts and eight independently
sampled on-policy trajectories per prompt. It uses dynamic micro-batches and
gradient accumulation to fit the resulting 768-trajectory global update on
8x46GB L40S GPUs. Sequence parallelism and rollout tensor parallelism only
change the memory layout; they do not reduce the global batch or rollout count.

```bash
bash scripts/run_vision_opd_released_b96_r8_gradaccum_4b.sh
bash scripts/run_vision_opd_released_b96_r8_gradaccum_9b.sh
```

The 4B wrapper uses SP4 with a 2304-token per-GPU dynamic budget. The 9B
wrapper uses SP8 with a 1152-token per-GPU budget. In both cases, the sequence
parallel group retains the full 9216-token sequence limit and accumulates all
micro-batches before one optimizer and EMA update.

### Strict official evaluation

Strict evaluation uses a pristine export of the official `eval/` directory,
seed 42, temperature 0, thinking disabled, `max_tokens=32768`, three request
attempts, 256 inference workers, and the official `openai/gpt-oss-120b` judge.
The report always includes separate paper/local baseline and paper/local OPD
columns for both 4B and 9B:

```bash
python scripts/summarize_official_evaluation.py --project-root "$PWD"
# docs/official_evaluation_reproduction.md
```

The interim baseline alignment run with a local Qwen2.5-72B fallback judge is
documented in `docs/diagnostic_qwen25_72b_baseline_alignment.md`. It validates
the inference path but is intentionally excluded from the final GPT-OSS score
columns.

The older `max_tokens=1024` evaluations are diagnostic only and are archived
under `benchmark/diagnostic_max1024_20260717/`.

### Selected 4B step-65 reproduction

The local reproduction report evaluates the completed one-epoch checkpoint
`Vision-OPD-Qwen3.5-4B-released-b96-r8-gradaccum-sp4/global_step_65` and its
merged Hugging Face model. The run preserves the released global batch of 96,
eight rollouts per prompt, 65 optimizer/EMA updates, and 6,240 consumed
prompts. It uses rollout TP4 rather than the released TP1 setting; the report
records this topology difference because it can change sampled trajectories.

The end-to-end controller reuses validated benchmark artifacts, completes all
ten official benchmarks with the local `gpt-oss-120b` judge, generates the
four-column paper/local report, and then runs both 680-row VTC-Bench tracks:

```bash
screen -L -Logfile logs/selected_4b_goal_pipeline.screen.log \
  -dmS selected_4b_goal_pipeline \
  bash scripts/run_selected_4b_goal_pipeline.sh
```

The canonical reports are:

- `docs/vision_opd_4b_vtc_reproduction.md`
- `docs/vision_opd_goal_reproduction_report.md`

Large local artifacts are intentionally excluded from Git: model weights,
checkpoints, datasets, raw benchmark images, rollout trajectories, raw model
answers, judge outputs, TensorBoard events, and logs. The repository retains
the frozen official evaluator, provenance, compact score files, scripts,
configuration, and reports required to audit the reproduction.

### Checkpoint merge validation

After training, merge the FSDP-sharded checkpoint into a standard HuggingFace model:

```bash
bash scripts/merge_checkpoint.sh <path_to_checkpoint>
```

For example:

```bash
bash scripts/merge_checkpoint.sh ./checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65/
```

To preserve the resumable checkpoint and write a clean Hugging Face directory,
set a separate target:

```bash
TARGET_DIR=./merged_models/Vision-OPD-Qwen3.5-4B \
  bash scripts/merge_checkpoint.sh \
  ./checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65/
```

This merges the FSDP actor shards, saves the model weights, config, tokenizer, and processor into the specified directory. The merged checkpoint can then be loaded directly with `transformers` or served with vLLM.
## Quick Start

### 1. Deployment

Serve the merged checkpoint with vLLM, for example:

```bash
vllm serve yuanqianhao/Vision-OPD-9B \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 8 \
    --served-model-name Vision-OPD-9B \
    --trust-remote-code
```

The server listens on port 8000 by default. You can then query the model via the OpenAI-compatible API at `http://localhost:8000/v1/chat/completions`.

Please check out our [Hugging Face Collection](https://huggingface.co/collections/yuanqianhao/vision-opd) for all public Vision-OPD checkpoints.

### 2. Evaluation

Evaluate the deployed model on fine-grained visual benchmarks:

```bash
API_BASE="http://localhost:8000/v1/" \
OPENAI_MODEL_ID="Vision-OPD-9B" \
JUDGE_API_BASE="YOUR_JUDGE_API_BASE" \
JUDGE_MODEL="YOUR_JUDGE_MODEL_NAME" \
BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
bash eval/run_eval.sh
```

Supported benchmarks: `vstar`, `zoombench`, `hrbench-4k`, `hrbench-8k`, `mme-realworld`, `mme-realworld-cn`, `mme-realworld-lite`, `visualprobe`, `mmvp`, `cv-bench`, `mmstar`, `pope`.

The evaluation script runs inference via the OpenAI-compatible API. Judge configuration can be set via `JUDGE_API_BASE` / `JUDGE_MODEL_PATH` environment variables. We use `openai/gpt-oss-120b` as the judge model. Other powerful models like Qwen3.5 or closed-source models are also recommended.

To evaluate Qwen3.5 models as baselines, set `ENABLE_THINKING=False` to run in non-thinking mode, for example:

```bash
API_BASE="http://localhost:8000/v1/" \
OPENAI_MODEL_ID="Qwen3.5-9B" \
ENABLE_THINKING=False \
JUDGE_API_BASE="YOUR_JUDGE_API_BASE" \
JUDGE_MODEL="YOUR_JUDGE_MODEL_NAME" \
BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
bash eval/run_eval.sh
```

## Training

### 1. Prepare Training Data

Download and preprocess the [Vision-OPD-6K](https://huggingface.co/datasets/yuanqianhao/Vision-OPD-6K) dataset:

```bash
python scripts/prepare_data.py --data-dir ./data
```

This downloads images and metadata from HuggingFace, extracts archives, and converts `train.jsonl` to the parquet format expected by the training pipeline.

### 2. Training

Launch Vision-OPD training:

```bash
bash scripts/run_vision_opd.sh
```

Key hyperparameters can be edited at the top of the script. See the script for the full configuration.

### 3. Merge Checkpoints

After training, merge the FSDP-sharded checkpoint into a standard HuggingFace model:

```bash
bash scripts/merge_checkpoint.sh <path_to_checkpoint>
```

For example:

```bash
bash scripts/merge_checkpoint.sh ./checkpoints/Vision-OPD-Qwen3.5-4B/global_step_65/
```

This merges the FSDP actor shards, saves the model weights, config, tokenizer, and processor into the specified directory. The merged checkpoint can then be loaded directly with `transformers` or served with vLLM.

## Citation

If you find Vision-OPD useful for your research, please consider citing:

```bibtex
@article{yuan2026vision,
  title={Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation},
  author={Yuan, Qianhao and Lou, Jie and Yu, Xing and Lin, Hongyu and Sun, Le and Han, Xianpei and Lu, Yaojie},
  journal={arXiv preprint arXiv:2605.18740},
  year={2026}
}
```

## License

Apache-2.0 License
