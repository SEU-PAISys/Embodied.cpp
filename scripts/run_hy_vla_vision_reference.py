#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Dump HY-VLA Python visual-frontend tensors for C++ parity.

The script intentionally exercises one image at a time so its outputs match the
vla.cpp debug modes:

  vision_patch  -> patch embedding + resized position embedding, 196x1152
  vision_vit    -> vision_patch + first N HyViT2 blocks, 196x1152
  vision_merger -> vision_vit + HY merger/pooler, 49x2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


DEFAULT_HY_REPO = Path("/home/xuling/robotic_code/embodied.cpp/Hy-Embodied-0.5-VLA-main")
DEFAULT_CKPT = Path("/home/xuling/robotic_dataset/HY-VLA")


def make_image(kind: str, h: int, w: int) -> np.ndarray:
    if kind == "zero":
        return np.zeros((h, w, 3), dtype=np.uint8)
    if kind == "gradient":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        arr[..., 2] = 17
        return arr
    raise ValueError(f"unknown image kind: {kind}")


def dump_f32(path: Path, x: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = x.detach().float().cpu().numpy().astype(np.float32, copy=False)
    np.ascontiguousarray(arr).tofile(path)
    path.with_suffix(path.suffix + ".shape.txt").write_text(
        " ".join(str(v) for v in arr.shape) + "\n", encoding="utf-8"
    )


def summarize(name: str, x: torch.Tensor) -> None:
    y = x.detach().float().cpu().numpy()
    print(
        f"{name}: shape={tuple(x.shape)} sum={float(y.sum()):.12f} "
        f"mean={float(y.mean()):.12f} min={float(y.min()):.12f} max={float(y.max()):.12f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hy-repo", type=Path, default=DEFAULT_HY_REPO)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--mode", choices=["vision_pixels", "vision_patch", "vision_vit", "vision_merger"],
                        default="vision_merger")
    parser.add_argument("--layers", type=int, default=0, help="Number of HyViT2 blocks for vision_vit/vision_merger.")
    parser.add_argument("--image-kind", choices=["zero", "gradient"], default="zero")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--load-mode", choices=["visual-only", "full"], default="visual-only",
                        help="visual-only loads only model.dual_tower.vlm.model.visual.* tensors.")
    parser.add_argument("--dump-f32", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.hy_repo))
    from hy_vla import HyVLA, HyVLAConfig
    from hy_vla.modeling_hy_vla import resize_with_pad
    from hy_vla.hunyuan_vl_mot.modeling_hunyuan_vl_mot import HYViT2_400MAnyRes

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    config = HyVLAConfig.from_pretrained(args.ckpt)
    if args.load_mode == "full":
        policy = HyVLA.from_pretrained(str(args.ckpt), config=config, map_location="cpu")
        policy.enable_video_encoder_if_needed()
        policy.eval()
        policy = policy.to(device=args.device, dtype=dtype)
        visual = policy.model.dual_tower.vlm.model.visual
    else:
        visual = HYViT2_400MAnyRes(vision_tower="")
        tensor_path = args.ckpt / "model.safetensors"
        raw = load_file(str(tensor_path), device="cpu")
        prefix = "model.dual_tower.vlm.model.visual."
        state = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
        missing, unexpected = visual.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[hy-vla-vision-reference] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print("missing_first10=", list(missing)[:10])
            if unexpected:
                print("unexpected_first10=", list(unexpected)[:10])
        visual.eval()
        visual = visual.to(device=args.device, dtype=dtype)

    image = make_image(args.image_kind, args.height, args.width)
    x = torch.from_numpy(image).to(device=args.device, dtype=torch.float32)
    x = x.permute(2, 0, 1).unsqueeze(0) / 255.0
    x = resize_with_pad(x, 224, 224, pad_value=0, mode="bilinear")
    x = (x * 2.0 - 1.0).to(dtype=dtype)
    if args.mode == "vision_pixels":
        out = x.reshape(-1)
        summarize(args.mode, out)
        dump_f32(args.dump_f32, out)
        print(f"dumped_f32={args.dump_f32}")
        return 0

    vit = visual.vision_tower
    with torch.no_grad():
        _, _, h, w = x.shape
        h //= vit.patch_embed.patch_size[0]
        w //= vit.patch_embed.patch_size[1]
        out = vit.patch_embed(x)
        out = out + vit.rescale_positional_embedding(out_size=(h, w))
        out = vit.patch_drop(out)
        out = vit.norm_pre(out)
        if args.mode in {"vision_vit", "vision_merger"}:
            n_layers = max(0, min(args.layers, len(vit.blocks)))
            for i in range(n_layers):
                out = vit.blocks[i](out)
        if args.mode == "vision_merger":
            out = visual.merger(out, (h, w))
            out = out.reshape(-1, out.shape[-1])
        else:
            out = out.reshape(-1, out.shape[-1])

    summarize(args.mode, out)
    dump_f32(args.dump_f32, out)
    print(f"dumped_f32={args.dump_f32}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
