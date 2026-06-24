#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Inspect HY-VLA tokenizer/processor frontend constants.

This is a lightweight, dependency-free inventory helper for the C++ port.  It
reads the released checkpoint metadata and reports the constants needed to
reconstruct the prefix path before the routed VLM/action graph:

  * text/action dimensions from config.json;
  * image processor patch/merge/normalization settings;
  * multimodal special token ids from tokenizer.json/tokenizer_config.json;
  * the image placeholder sequence used by chat_template.jinja.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def added_token_map(tokenizer_json: dict[str, Any], tokenizer_config: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for tok in tokenizer_json.get("added_tokens", []):
        content = tok.get("content")
        idx = tok.get("id")
        if isinstance(content, str) and isinstance(idx, int):
            out[content] = idx
    dec = tokenizer_config.get("added_tokens_decoder", {})
    for idx, tok in dec.items():
        content = tok.get("content") if isinstance(tok, dict) else None
        if isinstance(content, str):
            out.setdefault(content, int(idx))
    return out


def placeholder_token(n: int) -> str:
    return f"<｜hy_place▁holder▁no▁{n}｜>"


def find_template_image_sequence(template: str) -> list[str]:
    m = re.search(r"(?:<｜hy_place▁holder▁no▁\d+｜>){4}", template)
    if not m:
        return []
    return re.findall(r"<｜hy_place▁holder▁no▁\d+｜>", m.group(0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/xuling/robotic_dataset/HY-VLA"))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    model_dir = args.model_dir
    cfg = load_json(model_dir / "config.json")
    prep = load_json(model_dir / "preprocessor_config.json")
    tok_cfg = load_json(model_dir / "tokenizer_config.json")
    tok_json = load_json(model_dir / "tokenizer.json")
    template = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    tok_map = added_token_map(tok_json, tok_cfg)

    text_cfg = cfg.get("vlm_config_dict", {}).get("text_config", {})
    placeholder_ids = {
        str(n): tok_map.get(placeholder_token(n))
        for n in [0, 1, 2, 3, 666, 667, 669, 670, 671, 672]
    }
    template_image_tokens = find_template_image_sequence(template)
    template_image_ids = [tok_map.get(t) for t in template_image_tokens]

    merge_size = int(prep.get("merge_size", 2))
    patch_size = int(prep.get("patch_size", 16))
    temporal_patch_size = int(prep.get("temporal_patch_size", 1))
    resize = cfg.get("resize_imgs_with_padding")
    if isinstance(resize, list) and len(resize) == 2:
        # For the released 224x224 config this is 14x14 patches and 7x7
        # merged visual tokens per image.
        grid_h = int(resize[0]) // patch_size
        grid_w = int(resize[1]) // patch_size
        merged_tokens = (grid_h // merge_size) * (grid_w // merge_size)
    else:
        grid_h = grid_w = merged_tokens = None

    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "hy_vla": {
            "chunk_size": cfg.get("chunk_size"),
            "n_action_steps": cfg.get("n_action_steps"),
            "num_steps": cfg.get("num_steps"),
            "max_state_dim": cfg.get("max_state_dim"),
            "max_action_dim": cfg.get("max_action_dim"),
            "tokenizer_max_length": cfg.get("tokenizer_max_length"),
            "use_video_encoder": cfg.get("use_video_encoder"),
            "spacetime_layer_stride": cfg.get("spacetime_layer_stride"),
            "image_features": cfg.get("image_features"),
        },
        "text_config": {
            "hidden_size": text_cfg.get("hidden_size"),
            "num_hidden_layers": text_cfg.get("num_hidden_layers"),
            "num_attention_heads": text_cfg.get("num_attention_heads"),
            "num_key_value_heads": text_cfg.get("num_key_value_heads"),
            "head_dim": text_cfg.get("head_dim"),
            "intermediate_size": text_cfg.get("intermediate_size"),
            "rms_norm_eps": text_cfg.get("rms_norm_eps"),
            "rope_theta": text_cfg.get("rope_theta"),
            "rope_scaling": text_cfg.get("rope_scaling"),
            "vocab_size": text_cfg.get("vocab_size"),
        },
        "processor": {
            "image_processor_type": prep.get("image_processor_type"),
            "patch_size": patch_size,
            "merge_size": merge_size,
            "temporal_patch_size": temporal_patch_size,
            "do_resize": prep.get("do_resize"),
            "do_rescale": prep.get("do_rescale"),
            "rescale_factor": prep.get("rescale_factor"),
            "do_normalize": prep.get("do_normalize"),
            "image_mean": prep.get("image_mean"),
            "image_std": prep.get("image_std"),
            "min_pixels": prep.get("min_pixels"),
            "max_pixels": prep.get("max_pixels"),
            "resize_imgs_with_padding": resize,
            "derived_grid_hw": [grid_h, grid_w] if grid_h is not None else None,
            "derived_merged_tokens_per_image": merged_tokens,
        },
        "special_tokens": {
            "bos_token_id": text_cfg.get("bos_token_id"),
            "eos_token_id": text_cfg.get("eos_token_id"),
            "pad_token_id": text_cfg.get("pad_token_id"),
            "user_token_id": tok_map.get("<｜hy_User｜>"),
            "assistant_token_id": tok_map.get("<｜hy_Assistant｜>"),
            "eot_token_id": tok_map.get("<｜hy_EOT｜>"),
            "placeholder_ids": placeholder_ids,
            "template_image_tokens": template_image_tokens,
            "template_image_ids": template_image_ids,
        },
        "cpp_port_notes": [
            "processor expands each image placeholder into rows of image tokens plus image_newline tokens",
            "model replaces HY_VL_MOT_IMAGE_TOKEN_ID embeddings with HyViT2+merger visual embeddings",
            "union image/video placeholder mask becomes the routed prefix modality mask",
            "vision_start/end and image_newline tokens remain text-branch tokens; image placeholder tokens become vision-branch tokens",
        ],
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
