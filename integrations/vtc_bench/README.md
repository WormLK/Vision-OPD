# VTC-Bench Integration

This directory snapshots the Vision-OPD-4B configuration used by the local
VTC-Bench reproduction. The executable workspace is expected at:

```text
/data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab
```

Before running the end-to-end controller, install or compare the snapshot with
the executable workspace:

```bash
cp integrations/vtc_bench/eval_config/vision_opd_qwen35_4b_code.yaml \
  /data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/eval_config/
cp integrations/vtc_bench/eval_config/vision_opd_qwen35_4b_interface.yaml \
  /data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/eval/eval_config/
cp integrations/vtc_bench/scripts/run_vision_opd_4b_vtc_bench.sh \
  /data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/scripts/
git -C /data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab \
  apply /data00/users/wanglikun/ProjWormLK/Vision-OPD/integrations/vtc_bench/patches/qwen_agent_vtc_compat.patch
```

Locked generation settings are temperature 0.6, top-p 0.95, top-k 20, seed
1234, `max_tokens=40960`, and 30 workers per track. Both tracks run concurrently
against the shared server. The code-driven track enables only
`code_interpreter`; the interface-driven track enables the 35 OpenCV tools
listed in its YAML. The vLLM launcher uses Qwen3 reasoning and Qwen3-Coder tool
parsers, thinking enabled, DP8/TP1, prefix caching, and a 131,072-token context.

The optional Qwen-Agent patch is activated only by the launcher export
`QWEN_AGENT_REPEATED_NO_TOOL_LIMIT=2`. It skips repeated identical no-tool
intent text and enters the agent's existing final-answer fallback. Normal tool
calls and non-repeating reasoning retain the upstream 20-call allowance.

For the deadline-driven resumed tail, the launcher initially exported
`QWEN_AGENT_MAX_LLM_CALL_PER_RUN=4`. After 1,326/1,360 valid track-samples, the
remaining 34 pathological MC rows use a two-call cap and
`VTC_FORCE_OPTION_LETTER=true`. The first call still exposes the track tools;
the second uses the existing direct-answer fallback, while the extra prompt
sentence only constrains the final `<answer>` to an option letter. Remove these
exports to reproduce the unmodified 20-call protocol; the final report records
both tail phases.

The resumed tail also sets `QWEN_AGENT_STOP_ON_FINAL_ANSWER=1`. This keeps the
configured `max_tokens=40960` unchanged and stops only after the required
`</answer>` delimiter has been generated. The compatibility patch restores the
delimiter omitted by OpenAI-compatible stop handling, so downstream extraction
sees the same completed answer while avoiding post-answer repetition.

Dataset images, TSV files, run directories, and VLMEvalKit outputs are not
vendored in this repository.

## Qwen3.5 Base tracks

The Base/no-tool reproduction is defined by the three YAML files whose names
end in `_base.yaml`. Apply `patches/vtc_base_direct_mode.patch` to add the
one-shot direct agent, then run:

```bash
screen -L -Logfile /data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab/logs/qwen35_vtc_base_sequence.log \
  -dmS qwen35_vtc_base_sequence \
  bash /data00/users/wanglikun/ProjWormLK/Vision-OPD/integrations/vtc_bench/scripts/run_qwen35_vtc_base_sequence.sh
```

The controller waits for the code/interface completion audit and then runs,
strictly serially, trained OPD-4B Base, Qwen3.5-4B Base, and Qwen3.5-9B Base.
Each track makes one multimodal call with `functions=[]`, uses the Strong System
Prompt and the user prompt without a GT toolchain, and fixes thinking through
vLLM's `--default-chat-template-kwargs '{"enable_thinking":true}'`. Generation
is locked to temperature 0.6, top-p 0.95, top-k 20, repetition penalty 1.0,
presence penalty 0, 40,960 output tokens, and seed 1234. The final audit checks
680 unique successful rows, the no-tool/one-user-turn protocol, heuristic score
files, serial order, and the generated report.

The launcher passes `--chat-template` explicitly. OPD-4B uses the original
Qwen3.5-4B `chat_template.jinja`; the two baselines use the template in their
own model directories. This is intentional because the merged OPD directory's
saved template differs from the original backbone template, while this Base
comparison requires the native Qwen3.5 prompting contract.

Validate the locked configuration and the positive/negative result-audit paths
without loading a model:

```bash
python scripts/validate_vtc_base_configs.py \
  --vtc-root /data00/users/wanglikun/ProjWormLK/visionReason/qwen_tool_calling_lab
python scripts/test_vtc_base_validator.py
```
