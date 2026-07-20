#!/usr/bin/env python3
import argparse
import os
import types

import ray


def install_and_snapshot(worker_dict, checkpoint_path, expected_step, max_ckpt_to_keep):
    from omegaconf import OmegaConf

    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    worker = worker_dict.worker_dict["actor_rollout_ref"]
    actor_step = getattr(worker.actor, "_current_global_steps", None)
    if actor_step != expected_step:
        raise RuntimeError(
            f"Refusing mismatched teacher snapshot on rank {worker.rank}: "
            f"actor_step={actor_step}, expected_step={expected_step}"
        )

    manager = getattr(worker, "teacher_checkpoint_manager", None)
    if manager is None:
        config = OmegaConf.create({"load_contents": ["model"], "save_contents": ["model"]})
        manager = FSDPCheckpointManager(
            model=worker.actor.teacher_module,
            optimizer=None,
            lr_scheduler=None,
            processing_class=worker.processor if worker.processor is not None else worker.tokenizer,
            checkpoint_config=config,
        )
        worker.teacher_checkpoint_manager = manager

    if not getattr(worker, "_live_teacher_checkpoint_patch", False):
        original_save = worker.save_checkpoint
        original_load = worker.load_checkpoint

        def save_with_teacher(
            this, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None
        ):
            original_save(local_path, hdfs_path, global_step, max_ckpt_to_keep)
            teacher_local_path = os.path.join(local_path, "teacher")
            teacher_hdfs_path = os.path.join(hdfs_path, "teacher") if hdfs_path is not None else None
            this.teacher_checkpoint_manager.save_checkpoint(
                local_path=teacher_local_path,
                hdfs_path=teacher_hdfs_path,
                global_step=global_step,
                max_ckpt_to_keep=max_ckpt_to_keep,
            )

        def load_with_teacher(this, local_path, hdfs_path=None, del_local_after_load=False):
            original_load(local_path, hdfs_path, del_local_after_load)
            teacher_local_path = os.path.join(local_path, "teacher")
            if not os.path.isdir(teacher_local_path):
                raise RuntimeError(
                    f"EMA/progressive teacher checkpoint is required for exact resume: {teacher_local_path}"
                )
            teacher_hdfs_path = os.path.join(hdfs_path, "teacher") if hdfs_path is not None else None
            this.teacher_checkpoint_manager.load_checkpoint(
                local_path=teacher_local_path,
                hdfs_path=teacher_hdfs_path,
                del_local_after_load=del_local_after_load,
            )

        worker.save_checkpoint = types.MethodType(save_with_teacher, worker)
        worker.load_checkpoint = types.MethodType(load_with_teacher, worker)
        worker._live_teacher_checkpoint_patch = True

    teacher_path = os.path.join(checkpoint_path, "teacher")
    manager.save_checkpoint(
        local_path=teacher_path,
        hdfs_path=None,
        global_step=expected_step,
        max_ckpt_to_keep=max_ckpt_to_keep,
    )
    return {
        "rank": worker.rank,
        "actor_step": actor_step,
        "teacher_path": teacher_path,
        "patched": worker._live_teacher_checkpoint_patch,
    }


def main():
    parser = argparse.ArgumentParser(description="Install EMA teacher checkpointing in live Ray workers")
    parser.add_argument("--address", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--actor-prefix", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--max-ckpt-to-keep", type=int, default=2)
    args = parser.parse_args()

    ray.init(address=args.address, namespace=args.namespace)
    try:
        actors = [
            ray.get_actor(f"{args.actor_prefix}:{rank}", namespace=args.namespace)
            for rank in range(args.world_size)
        ]
        refs = [
            actor.execute_with_func_generator.remote(
                install_and_snapshot,
                args.checkpoint_path,
                args.expected_step,
                args.max_ckpt_to_keep,
            )
            for actor in actors
        ]
        results = sorted(ray.get(refs), key=lambda row: row["rank"])
        for result in results:
            print(result)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
