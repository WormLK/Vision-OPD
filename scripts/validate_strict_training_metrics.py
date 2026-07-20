#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


EXACT_ONE_TAGS = (
    "self_distillation/teacher_always_on_fraction",
    "self_distillation/teacher_image_swap_fraction",
    "self_distillation/self_distillation_mask.mean()",
)
EXACT_ZERO_TAGS = (
    "self_distillation/policy_fallback_fraction",
    "self_distillation/grpo_fallback_count",
    "actor/policy_fallback_fraction",
    "self_distillation/empty_target_batch",
    "actor/kl_loss",
    "actor/grpo_loss",
    "prompt_length/clip_ratio",
    "response/aborted_ratio",
)
POSITIVE_TAGS = (
    "self_distillation/num_distill_tokens",
    "timing_s/update_actor/student_forward",
    "timing_s/update_actor/teacher_forward",
    "timing_s/update_actor/backward",
    "timing_s/update_actor/optimizer_step",
    "timing_s/update_actor/teacher_ema_update",
    "perf/total_num_tokens",
)
FINITE_TAGS = (
    "actor/vopd_loss",
    "actor/grad_norm",
    "rollout_corr/kl",
    "perf/max_memory_allocated_gb",
    "perf/cpu_memory_used_gb",
    "timing_s/step",
)


def main():
    parser = argparse.ArgumentParser(description="Validate strict Vision-OPD training metrics")
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()

    event_files = sorted(
        args.tensorboard_dir.resolve().glob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
    )
    if not event_files:
        raise SystemExit(f"no TensorBoard event files in {args.tensorboard_dir}")
    accumulators = []
    available = set()
    for event_path in event_files:
        accumulator = EventAccumulator(str(event_path))
        accumulator.Reload()
        accumulators.append(accumulator)
        available.update(accumulator.Tags().get("scalars", []))
    expected_steps = set(range(1, args.expected_step + 1))
    errors = []

    def values(tag):
        if tag not in available:
            errors.append(f"missing scalar tag: {tag}")
            return []
        events_by_step = {}
        for accumulator in accumulators:
            if tag in accumulator.Tags().get("scalars", []):
                for event in accumulator.Scalars(tag):
                    events_by_step[event.step] = event
        events = [events_by_step[step] for step in sorted(events_by_step)]
        actual_steps = {event.step for event in events if 1 <= event.step <= args.expected_step}
        if actual_steps != expected_steps:
            errors.append(
                f"non-contiguous steps for {tag}: actual={sorted(actual_steps)}, "
                f"expected=1-{args.expected_step}"
            )
        selected = [event.value for event in events if 1 <= event.step <= args.expected_step]
        if any(not math.isfinite(value) for value in selected):
            errors.append(f"non-finite values for {tag}")
        return selected

    for tag in EXACT_ONE_TAGS:
        selected = values(tag)
        if selected and any(not math.isclose(value, 1.0, abs_tol=1e-8) for value in selected):
            errors.append(f"{tag} is not exactly one: {selected}")
    for tag in EXACT_ZERO_TAGS:
        selected = values(tag)
        if selected and any(not math.isclose(value, 0.0, abs_tol=1e-8) for value in selected):
            errors.append(f"{tag} is not exactly zero: {selected}")
    for tag in POSITIVE_TAGS:
        selected = values(tag)
        if selected and any(value <= 0 for value in selected):
            errors.append(f"{tag} contains non-positive values: {selected}")
    for tag in FINITE_TAGS:
        values(tag)

    global_steps = values("training/global_step")
    if global_steps and [round(value) for value in global_steps] != list(range(1, args.expected_step + 1)):
        errors.append(f"training/global_step values do not match event steps: {global_steps}")
    epochs = values("training/epoch")
    if epochs and any(value != 0 for value in epochs):
        errors.append(f"unexpected epoch values before the one-epoch boundary: {epochs}")
    turns = values("num_turns/mean")
    if turns and any(not math.isclose(value, 2.0, abs_tol=1e-8) for value in turns):
        errors.append(f"unexpected mean turn count: {turns}")

    learning_rates = values("actor/lr")
    if learning_rates:
        expected_learning_rates = [2e-6 * min((step - 1) / 10, 1.0) for step in range(1, args.expected_step + 1)]
        mismatches = [
            (step, actual, expected)
            for step, (actual, expected) in enumerate(
                zip(learning_rates, expected_learning_rates, strict=True), start=1
            )
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        ]
        if mismatches:
            errors.append(f"learning-rate schedule differs from released 10-step warmup/2e-6 plateau: {mismatches}")

    if errors:
        print(f"FAILED strict metric audit through step {args.expected_step}:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(
        f"PASS strict metrics through step {args.expected_step}: contiguous, finite, "
        "teacher/bbox always on, no fallback/truncation/abort, EMA updated"
    )


if __name__ == "__main__":
    main()
