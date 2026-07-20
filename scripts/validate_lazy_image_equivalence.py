#!/usr/bin/env python3
import argparse
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from verl.utils.dataset.vision_utils import process_image


def build_messages(prompt, image_path):
    content = prompt[0]["content"]
    before, separator, after = content.partition("<image>")
    if not separator:
        raise ValueError("training prompt has no <image> placeholder")
    items = []
    if before:
        items.append({"type": "text", "text": before})
    items.append({"type": "image", "image": image_path, "path": image_path})
    if after:
        items.append({"type": "text", "text": after})
    return [{"role": "user", "content": items}]


def main():
    parser = argparse.ArgumentParser(description="Check eager/deferred Qwen image processor equivalence")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--chat-template", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3, help="Rows to check; 0 checks the full dataset")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model.resolve(), local_files_only=True, trust_remote_code=True
    )
    chat_template = args.chat_template.read_text(encoding="utf-8")
    table = pq.read_table(args.data.resolve())
    sample_count = table.num_rows if args.samples == 0 else args.samples
    rows = table.slice(0, sample_count).to_pylist()
    if len(rows) != sample_count:
        raise SystemExit(f"requested {args.samples} samples, found {len(rows)}")

    checked = 0
    for row_index, row in enumerate(rows):
        for role, image_ref in (("student", row["images"][0]), ("teacher", row["bbox_images"][0])):
            messages = build_messages(row["prompt"], image_ref["path"])
            if role == "student":
                images, _ = process_vision_info(
                    messages,
                    image_patch_size=processor.image_processor.patch_size,
                    return_video_metadata=True,
                )
                deferred_images = [
                    process_image(
                        {"path": image_ref["path"]},
                        image_patch_size=processor.image_processor.patch_size,
                    )
                ]
            else:
                with Image.open(image_ref["path"]) as pil_image:
                    images = [pil_image.convert("RGB")]
                with Image.open(image_ref["path"]) as pil_image:
                    deferred_images = [pil_image.convert("RGB")]
            raw_prompt = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                chat_template=chat_template,
            )
            eager = processor(
                text=[raw_prompt],
                images=images,
                videos=None,
                return_tensors="pt",
                do_sample_frames=False,
            )
            decoded_prompt = processor.tokenizer.decode(
                eager["input_ids"][0], skip_special_tokens=True
            )
            rebuilt = processor(
                text=[decoded_prompt],
                images=deferred_images,
                videos=None,
                return_tensors="pt",
                do_sample_frames=False,
            )

            pixel_equal = torch.equal(eager["pixel_values"], rebuilt["pixel_values"])
            grid_equal = torch.equal(eager["image_grid_thw"], rebuilt["image_grid_thw"])
            max_abs_diff = float(
                (eager["pixel_values"] - rebuilt["pixel_values"]).abs().max().item()
            )
            if not pixel_equal or not grid_equal or max_abs_diff != 0.0:
                raise SystemExit(
                    f"lazy processor mismatch row={row_index} role={role}: "
                    f"pixel_equal={pixel_equal} grid_equal={grid_equal} max_abs_diff={max_abs_diff}"
                )
            if not args.quiet:
                print(
                    f"row={row_index} role={role} shape={tuple(eager['pixel_values'].shape)} "
                    f"grid={eager['image_grid_thw'].tolist()} exact=True"
                )
            checked += 1

    print(f"PASS lazy image equivalence: {checked} student/teacher processor inputs are bit-identical")


if __name__ == "__main__":
    main()
