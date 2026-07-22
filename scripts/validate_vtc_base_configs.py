#!/usr/bin/env python3
"""Validate the three locked Qwen3.5 VTC-Bench Base configurations."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import yaml


GENERATE_CFG = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 0,
    "max_tokens": 40960,
    "seed": 1234,
}

CONFIGS = {
    "vision_opd_qwen35_4b_base.yaml": (
        "Vision-OPD-Qwen3.5-4B-released-b96-r8-base",
        "vtc_vision_opd_4b_step65_base",
    ),
    "qwen35_4b_base.yaml": ("Qwen3.5-4B-base-vtc", "vtc_qwen35_4b_base"),
    "qwen35_9b_base.yaml": ("Qwen3.5-9B-base-vtc", "vtc_qwen35_9b_base"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtc-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.vtc_root.resolve()
    config_root = root / "eval/eval_config"
    expected_tsv = (root / "data/vtc_bench/VTC-Bench.absolute.tsv").resolve()
    header = expected_tsv.open(encoding="utf-8").readline().rstrip("\n").split("\t")
    if "model_tools_gt" in header or "reference_trajectory" in header:
        raise RuntimeError("Base TSV must not expose a GT toolchain or reference trajectory")

    source_path = root / "eval/VTC_Bench_Eval.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    direct_classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DirectAnswerAgent"
    ]
    if len(direct_classes) != 1:
        raise RuntimeError("VTC evaluator must define exactly one DirectAnswerAgent")
    direct_calls = [
        node
        for node in ast.walk(direct_classes[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_call_llm"
    ]
    if len(direct_calls) != 1:
        raise RuntimeError("DirectAnswerAgent must make exactly one static _call_llm call")
    functions = next((kw.value for kw in direct_calls[0].keywords if kw.arg == "functions"), None)
    if not isinstance(functions, ast.List) or functions.elts:
        raise RuntimeError("DirectAnswerAgent must statically pass functions=[]")
    reference_system = yaml.safe_load(
        (config_root / "vision_opd_qwen35_4b_code.yaml").read_text(encoding="utf-8")
    )["agent"]["system_prompt"]

    for filename, (model, results_name) in CONFIGS.items():
        path = config_root / filename
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        generate = config["llm"]["generate_cfg"]
        for key, expected in GENERATE_CFG.items():
            if generate.get(key) != expected:
                raise RuntimeError(f"{filename}: {key}={generate.get(key)!r}, expected={expected!r}")
        if generate.get("use_raw_api") is not True or generate.get("max_retries") != 10:
            raise RuntimeError(f"{filename}: raw API/retry mismatch")
        if config["llm"].get("model") != model:
            raise RuntimeError(f"{filename}: model mismatch")
        if config["llm"].get("thought_in_content") is not True:
            raise RuntimeError(f"{filename}: thought_in_content must be true")
        if config["llm"].get("model_server") != "http://127.0.0.1:8000/v1":
            raise RuntimeError(f"{filename}: model_server mismatch")
        if config.get("tools", {}).get("enabled") != []:
            raise RuntimeError(f"{filename}: Base must register no tools")
        if config.get("agent", {}).get("mode") != "direct":
            raise RuntimeError(f"{filename}: Base must use one-shot direct mode")
        prompt = config.get("prompt_template", "")
        if "reference_trajectory" in prompt or "model_tools_gt" in prompt:
            raise RuntimeError(f"{filename}: Base prompt leaks a reference trajectory")
        for required in ("<image>", "{question}", "{image_path}", "{image_size}", "<think>", "<answer>"):
            if required not in prompt:
                raise RuntimeError(f"{filename}: missing prompt field {required}")
        system = config.get("agent", {}).get("system_prompt", "")
        if system != reference_system:
            raise RuntimeError(f"{filename}: system prompt is not the exact code-track Strong Prompt")
        for required in (
            "First, look closely",
            "Next, apply tools",
            "Then, review the findings",
            "Image Path **MUST** be Absolute Path. And only **ONE** tool can be used at a time.",
        ):
            if required not in system:
                raise RuntimeError(f"{filename}: Strong System Prompt is incomplete")
        if Path(config["input"]["tsv_path"]).resolve() != expected_tsv:
            raise RuntimeError(f"{filename}: must use the non-GT Base TSV")
        expected_results = (root / "runs" / results_name).resolve()
        if Path(config["output"]["results_dir"]).resolve() != expected_results:
            raise RuntimeError(f"{filename}: results_dir mismatch")
        if config["processing"].get("num_workers") != 30:
            raise RuntimeError(f"{filename}: num_workers must be 30")
        print(f"PASS {filename}: model={model}, direct one-shot, tools=[]")


if __name__ == "__main__":
    main()
