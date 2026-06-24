#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build the HY-VLA prefix fixture from safetensors without loading the model.

This mirrors the C++ host-side prefix builder:

  <bos><hy_User>
  for each image:
      <vision_start>
      grid rows * (grid visual patch tokens + <vision_split>)
      <vision_end>
  language tokens

By default the visual patch tokens are the same deterministic synthetic values
used by ``scripts/smoke_hy_vla_server_precomputed.py``.  Alternatively,
``--visual-f32`` can provide real post-merger visual tokens, e.g. from
``run_hy_vla_vision_reference.py --mode vision_merger``.  The output can be
compared directly against ``VLA_HY_VLA_DUMP_PREFIX_F32`` and
``VLA_HY_VLA_DUMP_PREFIX_MASK_I32``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors import safe_open


DEFAULT_CKPT = Path("/home/xuling/robotic_dataset/HY-VLA/model.safetensors")
EMBED_NAME = "model.dual_tower.vlm.model.language_model.lm_head.weight"


def bf16_roundtrip(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32, copy=False).view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    rounded = (u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safetensors", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--vis-tokens-per-image", type=int, default=49)
    parser.add_argument("--visual-f32", type=Path, default=None,
                        help="Optional real visual tokens, raw f32 with shape [n_images*vis_tokens_per_image, hidden].")
    parser.add_argument("--lang-token", type=int, nargs="*", default=[120000, 120001, 120020, 7])
    parser.add_argument("--dump-f32", type=Path, required=True)
    parser.add_argument("--dump-mask-i32", type=Path, required=True)
    args = parser.parse_args()

    grid = int(round(args.vis_tokens_per_image ** 0.5))
    if grid * grid != args.vis_tokens_per_image:
        raise ValueError("--vis-tokens-per-image must be a square number")

    with safe_open(args.safetensors, framework="pt", device="cpu") as f:
        emb = f.get_tensor(EMBED_NAME).float().numpy()

    hidden = emb.shape[1]
    if args.visual_f32 is not None:
        flat = np.fromfile(args.visual_f32, dtype=np.float32)
        per_image = args.vis_tokens_per_image * hidden
        if flat.size % per_image != 0:
            raise SystemExit(
                f"{args.visual_f32} has {flat.size} floats, not divisible by "
                f"vis_tokens_per_image*hidden={args.vis_tokens_per_image}*{hidden}"
            )
        args.n_images = flat.size // per_image
        vis = flat.reshape(args.n_images, args.vis_tokens_per_image, hidden)
    else:
        n_vis = args.n_images * args.vis_tokens_per_image
        vis = np.sin(np.arange(n_vis * hidden, dtype=np.float32) * np.float32(0.013))
        vis = bf16_roundtrip(vis * np.float32(0.02)).reshape(args.n_images, args.vis_tokens_per_image, hidden)

    chunks: list[np.ndarray] = []
    mask: list[int] = []

    def add_text(ids: list[int]) -> None:
        chunks.append(emb[np.asarray(ids, dtype=np.int64)])
        mask.extend([0] * len(ids))

    def add_visual(x: np.ndarray) -> None:
        chunks.append(x.reshape(1, hidden).astype(np.float32, copy=False))
        mask.append(1)

    add_text([120000, 120006])
    for img_idx in range(args.n_images):
        add_text([120684])
        for r in range(grid):
            for c in range(grid):
                add_visual(vis[img_idx, r * grid + c])
            add_text([120689])
        add_text([120685])
    add_text(args.lang_token)

    prefix = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    mask_arr = np.asarray(mask, dtype=np.int32)

    args.dump_f32.parent.mkdir(parents=True, exist_ok=True)
    args.dump_mask_i32.parent.mkdir(parents=True, exist_ok=True)
    prefix.tofile(args.dump_f32)
    mask_arr.tofile(args.dump_mask_i32)

    print(
        f"hy_prefix: shape={prefix.shape} sum={float(prefix.sum()):.12f} "
        f"min={float(prefix.min()):.12f} max={float(prefix.max()):.12f}"
    )
    print(
        f"hy_prefix_mask: shape={mask_arr.shape} sum={int(mask_arr.sum())} "
        f"first20={mask_arr[:20].tolist()}"
    )
    print(f"dumped_f32={args.dump_f32}")
    print(f"dumped_mask_i32={args.dump_mask_i32}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
