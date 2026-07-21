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

Dataset images, TSV files, run directories, and VLMEvalKit outputs are not
vendored in this repository.
