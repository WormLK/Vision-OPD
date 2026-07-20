#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import torch


def nonempty_rank_files(directory: Path, stem: str, world_size: int) -> set[int]:
    ranks = set()
    for rank in range(world_size):
        path = directory / f"{stem}_world_size_{world_size}_rank_{rank}.pt"
        if path.is_file() and path.stat().st_size > 0:
            ranks.add(rank)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an exact-resume Vision-OPD checkpoint")
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=96)
    parser.add_argument(
        "--max-step",
        type=int,
        default=65,
        help="Reject checkpoints with an extra optimizer/EMA update (default: 65).",
    )
    args = parser.parse_args()

    root = args.checkpoint_root.resolve()
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise SystemExit(f"missing checkpoint marker: {marker}")
    marker_step = int(marker.read_text(encoding="utf-8").strip())
    if marker_step > args.max_step:
        raise SystemExit(
            f"checkpoint marker {marker_step} exceeds strict one-epoch maximum {args.max_step}"
        )
    step = marker_step if args.step is None else args.step
    if step != marker_step:
        raise SystemExit(f"requested step {step} does not match latest marker {marker_step}")

    checkpoint = root / f"global_step_{step}"
    actor = checkpoint / "actor"
    teacher = actor / "teacher"
    expected = set(range(args.world_size))
    checks = {
        "actor model": nonempty_rank_files(actor, "model", args.world_size),
        "actor optimizer": nonempty_rank_files(actor, "optim", args.world_size),
        "actor extra state": nonempty_rank_files(actor, "extra_state", args.world_size),
        "EMA teacher model": nonempty_rank_files(teacher, "model", args.world_size),
    }
    errors = []
    for label, ranks in checks.items():
        if ranks != expected:
            errors.append(f"{label}: ranks={sorted(ranks)}, expected={sorted(expected)}")

    required = (
        checkpoint / "data.pt",
        actor / "huggingface" / "config.json",
        teacher / "huggingface" / "config.json",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty: {path}")

    # The checkpoint is written after scheduler.step(), while TensorBoard logs
    # the learning rate used by the just-completed optimizer update.
    expected_lr = min(step * 2e-7, 2e-6)
    for rank in range(args.world_size):
        extra_path = actor / f"extra_state_world_size_{args.world_size}_rank_{rank}.pt"
        if not extra_path.is_file() or extra_path.stat().st_size == 0:
            continue
        try:
            extra_state = torch.load(extra_path, map_location="cpu", weights_only=False)
            scheduler = extra_state.get("lr_scheduler") or {}
            if scheduler.get("last_epoch") != step:
                errors.append(
                    f"rank {rank} scheduler last_epoch: {scheduler.get('last_epoch')!r}, expected={step}"
                )
            if scheduler.get("_step_count") != step + 1:
                errors.append(
                    f"rank {rank} scheduler _step_count: {scheduler.get('_step_count')!r}, "
                    f"expected={step + 1}"
                )
            last_lr = scheduler.get("_last_lr")
            if (
                not isinstance(last_lr, list)
                or len(last_lr) != 1
                or not math.isclose(float(last_lr[0]), expected_lr, rel_tol=0.0, abs_tol=1e-12)
            ):
                errors.append(f"rank {rank} scheduler LR: {last_lr!r}, expected={[expected_lr]}")

            rng = extra_state.get("rng") or {}
            for key in ("cpu", "cuda"):
                value = rng.get(key)
                if not torch.is_tensor(value) or value.numel() == 0:
                    errors.append(f"rank {rank} RNG state {key!r} is missing or empty")
            for key in ("numpy", "random"):
                value = rng.get(key)
                if not isinstance(value, tuple) or not value:
                    errors.append(f"rank {rank} RNG state {key!r} is missing or invalid")
        except Exception as exc:
            errors.append(f"invalid actor extra state {extra_path}: {exc}")

    optimizer_path = actor / f"optim_world_size_{args.world_size}_rank_0.pt"
    if optimizer_path.is_file() and optimizer_path.stat().st_size > 0:
        try:
            optimizer_state = torch.load(
                optimizer_path, map_location="cpu", weights_only=False, mmap=True
            )
            state_entries = (optimizer_state or {}).get("state") or {}
            optimizer_steps = []
            missing_steps = 0
            for state in state_entries.values():
                value = state.get("step")
                if value is None:
                    missing_steps += 1
                else:
                    optimizer_steps.append(int(value.item() if torch.is_tensor(value) else value))
            if not state_entries:
                errors.append("rank 0 optimizer state is empty")
            if missing_steps:
                errors.append(f"rank 0 optimizer states missing step counter: {missing_steps}")
            unexpected_steps = sorted(set(optimizer_steps) - {step})
            if unexpected_steps:
                errors.append(
                    f"rank 0 optimizer step counters={unexpected_steps}, expected only {step}"
                )
        except Exception as exc:
            errors.append(f"invalid rank 0 optimizer state {optimizer_path}: {exc}")

    data_path = checkpoint / "data.pt"
    if data_path.is_file() and data_path.stat().st_size > 0:
        try:
            dataloader_state = torch.load(data_path, map_location="cpu", weights_only=False)
            sampler_state = dataloader_state.get("_sampler_iter_state") or {}
            nested_sampler_state = sampler_state.get("sampler_iter_state") or {}
            expected_samples = step * args.train_batch_size
            observed = {
                "_num_yielded": dataloader_state.get("_num_yielded"),
                "_sampler_iter_yielded": dataloader_state.get("_sampler_iter_yielded"),
                "samples_yielded": sampler_state.get("samples_yielded"),
                "sampler yielded": nested_sampler_state.get("yielded"),
            }
            for label in ("_num_yielded", "_sampler_iter_yielded"):
                if observed[label] != step:
                    errors.append(f"dataloader {label}: {observed[label]!r}, expected={step}")
            for label in ("samples_yielded", "sampler yielded"):
                if observed[label] != expected_samples:
                    errors.append(
                        f"dataloader {label}: {observed[label]!r}, expected={expected_samples}"
                    )
            generator = nested_sampler_state.get("generator")
            if not torch.is_tensor(generator) or generator.numel() == 0:
                errors.append("dataloader sampler generator state is missing or empty")
        except Exception as exc:
            errors.append(f"invalid dataloader state {data_path}: {exc}")

    if errors:
        print(f"FAILED strict checkpoint validation: {checkpoint}")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(
        f"PASS strict checkpoint step={step}: world_size={args.world_size}, "
        f"student/optimizer/EMA-teacher/config complete, scheduler/RNG valid, "
        f"optimizer_counter={step}, dataloader={step} batches/{step * args.train_batch_size} prompts"
    )


if __name__ == "__main__":
    main()
