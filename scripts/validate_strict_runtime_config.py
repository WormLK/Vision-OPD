#!/usr/bin/env python3
import argparse
import ast
import math
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RAY_PREFIX_RE = re.compile(r"\((?:TaskRunner|WorkerDict) pid=\d+\) ?")


def load_resolved_config(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    text = ANSI_RE.sub("", text)
    text = RAY_PREFIX_RE.sub("", text)
    start = text.index("{'actor_rollout_ref'")
    end = text.index("\n[validate_config]", start)

    config_lines = []
    for line in text[start:end].splitlines():
        # Ray warnings can be appended to a pprint line before the next newline.
        line = re.split(r"(?<!['\"])/data00/users/", line, maxsplit=1)[0]
        stripped = line.strip()
        if stripped.startswith(("'", "{", "}", "[", "]")):
            config_lines.append(line.rstrip())

    try:
        config = ast.literal_eval("\n".join(config_lines))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"unable to parse resolved config from {log_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise TypeError(f"resolved config is not a dict: {type(config)}")
    return config


def nested(config, *keys):
    value = config
    for key in keys:
        value = value[key]
    return value


def assert_value(errors, label, actual, expected):
    if isinstance(expected, float):
        matches = isinstance(actual, (int, float)) and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        )
    else:
        matches = actual == expected
    if not matches:
        errors.append(f"{label}: actual={actual!r}, expected={expected!r}")


def main():
    parser = argparse.ArgumentParser(description="Validate a strict Vision-OPD resolved runtime config")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--backbone", choices=("4b", "9b"), required=True)
    parser.add_argument("--rollout-tp", type=int)
    parser.add_argument("--rollout-max-num-seqs", type=int)
    args = parser.parse_args()

    config = load_resolved_config(args.log.resolve())
    model = nested(config, "actor_rollout_ref", "model")
    actor = nested(config, "actor_rollout_ref", "actor")
    rollout = nested(config, "actor_rollout_ref", "rollout")
    ref = nested(config, "actor_rollout_ref", "ref")
    data = config["data"]
    trainer = config["trainer"]
    algorithm = config["algorithm"]
    self_distillation = actor["self_distillation"]
    optim = actor["optim"]
    agent = rollout["agent"]

    topology = {
        "4b": {"sp": 4, "tp": 4, "tokens": 2304, "gpu_util": 0.30, "max_num_seqs": 64},
        "9b": {"sp": 8, "tp": 8, "tokens": 1152, "gpu_util": 0.25, "max_num_seqs": 32},
    }[args.backbone]
    if args.rollout_tp is not None:
        topology["tp"] = args.rollout_tp
    if args.rollout_max_num_seqs is not None:
        topology["max_num_seqs"] = args.rollout_max_num_seqs

    checks = (
        ("data.train_batch_size", data["train_batch_size"], 96),
        ("data.max_prompt_length", data["max_prompt_length"], 8192),
        ("data.max_response_length", data["max_response_length"], 1024),
        ("data.filter_overlong_prompts", data["filter_overlong_prompts"], False),
        ("data.truncation", data["truncation"], "error"),
        ("data.shuffle", data["shuffle"], True),
        ("data.dataloader_num_workers", data["dataloader_num_workers"], 0),
        ("actor.ppo_mini_batch_size", actor["ppo_mini_batch_size"], 96),
        ("actor.rollout_n", actor.get("rollout_n", actor.get("n")), 8),
        ("actor.ppo_epochs", actor["ppo_epochs"], 1),
        ("actor.policy_loss.loss_mode", nested(actor, "policy_loss", "loss_mode"), "vopd"),
        ("actor.optim.lr", optim["lr"], 2e-6),
        ("actor.optim.lr_warmup_steps", optim["lr_warmup_steps"], 10),
        ("actor.clip_ratio_high", actor["clip_ratio_high"], 0.3),
        ("actor.clip_ratio_low", actor["clip_ratio_low"], 0.2),
        ("self_distillation.alpha", self_distillation["alpha"], 0.5),
        ("self_distillation.distillation_topk", self_distillation["distillation_topk"], 100),
        ("self_distillation.teacher_always_on", self_distillation["teacher_always_on"], True),
        ("self_distillation.teacher_model_source", self_distillation["teacher_model_source"], "legacy"),
        ("self_distillation.teacher_regularization", self_distillation["teacher_regularization"], "ema"),
        ("self_distillation.teacher_update_rate", self_distillation["teacher_update_rate"], 0.05),
        ("self_distillation.teacher_image_key", self_distillation["teacher_image_key"], "bbox_images"),
        ("algorithm.adv_estimator", algorithm["adv_estimator"], "grpo"),
        ("algorithm.norm_adv_by_std_in_grpo", algorithm["norm_adv_by_std_in_grpo"], False),
        ("algorithm.use_kl_in_reward", algorithm["use_kl_in_reward"], False),
        ("trainer.total_epochs", trainer["total_epochs"], 1),
        ("trainer.n_gpus_per_node", trainer["n_gpus_per_node"], 8),
        ("trainer.resume_mode", trainer["resume_mode"], "auto"),
        ("trainer.save_freq", trainer["save_freq"], 1),
        ("trainer.max_actor_ckpt_to_keep", trainer["max_actor_ckpt_to_keep"], 2),
        ("actor.ulysses_sequence_parallel_size", actor["ulysses_sequence_parallel_size"], topology["sp"]),
        ("actor.ppo_max_token_len_per_gpu", actor["ppo_max_token_len_per_gpu"], topology["tokens"]),
        ("rollout.tensor_model_parallel_size", rollout["tensor_model_parallel_size"], topology["tp"]),
        ("rollout.gpu_memory_utilization", rollout["gpu_memory_utilization"], topology["gpu_util"]),
        ("rollout.max_model_len", rollout["max_model_len"], 9216),
        ("rollout.max_num_seqs", rollout["max_num_seqs"], topology["max_num_seqs"]),
        ("agent.dispatch_batch_size", agent["dispatch_batch_size"], 96),
        ("agent.defer_multimodal_processing", agent["defer_multimodal_processing"], True),
        ("actor param offload", nested(actor, "fsdp_config", "param_offload"), True),
        ("actor optimizer offload", nested(actor, "fsdp_config", "optimizer_offload"), True),
        ("actor activation offload", model["enable_activation_offload"], True),
        ("ref param offload", nested(ref, "fsdp_config", "param_offload"), True),
    )

    errors = []
    for label, actual, expected in checks:
        assert_value(errors, label, actual, expected)

    effective_tokens = actor["ppo_max_token_len_per_gpu"] * actor["ulysses_sequence_parallel_size"]
    assert_value(errors, "effective full-sequence token budget", effective_tokens, 9216)

    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    for expected_line in ("dataset len: 6241", "Size of train dataloader: 65", "Total training steps: 65"):
        if expected_line not in log_text:
            errors.append(f"missing runtime evidence: {expected_line}")

    if errors:
        print(f"FAILED strict {args.backbone.upper()} runtime config audit:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(
        f"PASS strict {args.backbone.upper()} runtime config: official 96x8 semantics, "
        f"1 epoch/65 updates, SP{topology['sp']}/TP{topology['tp']}, "
        "effective 9216-token sequences"
    )


if __name__ == "__main__":
    main()
