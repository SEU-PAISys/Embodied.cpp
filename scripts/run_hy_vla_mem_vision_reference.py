#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Dump HY-VLA MEM visual tokens for Python/C++ parity.

The output matches the C++ `VLA_HY_VLA_DUMP_VISUAL_TOKENS_F32` tensor:
`(num_cameras, 49, 2048)` float32 after HyViT2 MEM video encoding and merger.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


DEFAULT_HY_REPO = Path("/home/xuling/robotic_code/embodied.cpp/Hy-Embodied-0.5-VLA-main")
DEFAULT_CKPT = Path("/home/xuling/robotic_dataset/HY-VLA")


IMAGE_KEYS = [
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
]


def deterministic_image(cam: int, frame: int, h: int, w: int) -> np.ndarray:
    y, x, c = np.meshgrid(
        np.arange(h, dtype=np.uint16),
        np.arange(w, dtype=np.uint16),
        np.arange(3, dtype=np.uint16),
        indexing="ij",
    )
    arr = (x * (3 + cam * 7 + frame) + y * (5 + cam + frame * 3) + c * 37 + cam * 41 + frame * 19) % 256
    return np.ascontiguousarray(arr.astype(np.uint8))


def dump_f32(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    arr.tofile(path)
    path.with_suffix(path.suffix + ".shape.txt").write_text(
        " ".join(str(v) for v in arr.shape) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hy-repo", type=Path, default=DEFAULT_HY_REPO)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--layers", type=int, default=27,
                        help="Number of HyViT2 blocks to run before the merger.")
    parser.add_argument("--stage", choices=["vit", "merger"], default="merger")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dump-f32", type=Path, required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    sys.path.insert(0, str(args.hy_repo))
    from hy_vla import HyVLAConfig
    from hy_vla.hunyuan_vl_mot.modeling_hunyuan_vl_mot import HYViT2_400MAnyRes
    from hy_vla.modeling_hy_vla import resize_with_pad
    from hy_vla.space_time_attention import apply_video_encoder_patch

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    config = HyVLAConfig.from_pretrained(args.ckpt)
    visual = HYViT2_400MAnyRes(vision_tower="")
    prefix = "model.dual_tower.vlm.model.visual."
    state = {}
    with safe_open(str(args.ckpt / "model.safetensors"), framework="pt", device="cpu") as raw:
        for k in raw.keys():
            if k.startswith(prefix):
                state[k[len(prefix):]] = raw.get_tensor(k)
    missing, unexpected = visual.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[hy-vla-mem-reference] visual load missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("missing_first10=", list(missing)[:10])
        if unexpected:
            print("unexpected_first10=", list(unexpected)[:10])

    apply_video_encoder_patch(
        visual,
        spacetime_layer_stride=int(getattr(config, "spacetime_layer_stride", 4)),
        past_drop_layer=getattr(config, "past_drop_layer", None),
        max_num_frames=int(getattr(config, "max_num_frames", None) or args.history),
    )
    visual.eval()
    visual = visual.to(device=args.device, dtype=dtype)

    outs: list[np.ndarray] = []
    with torch.no_grad():
        for cam, key in enumerate(IMAGE_KEYS):
            frames = []
            for frame in range(args.history):
                img = deterministic_image(cam, frame, args.height, args.width)
                x = torch.from_numpy(img).to(device=args.device, dtype=torch.float32)
                x = x.permute(2, 0, 1) / 255.0
                frames.append(x)
            video = torch.stack(frames, dim=0).unsqueeze(0)
            b, k, c, h, w = video.shape
            video = video.reshape(b * k, c, h, w)
            video = resize_with_pad(video, 224, 224, pad_value=0, mode="bilinear")
            _, c2, h2, w2 = video.shape
            video = video.reshape(b, k, c2, h2, w2)
            video = (video * 2.0 - 1.0).to(device=args.device, dtype=dtype)
            vit = visual.vision_tower
            b, k, c, h_in, w_in = video.shape
            x = video.reshape(b * k, c, h_in, w_in)
            h = h_in // vit.patch_embed.patch_size[0]
            w = w_in // vit.patch_embed.patch_size[1]
            x = vit.patch_embed(x)
            x = x + vit.rescale_positional_embedding(out_size=(h, w))
            x = vit.patch_drop(x)
            x = vit.norm_pre(x)
            n_layers = max(0, min(args.layers, len(vit.blocks)))
            cur_k = k
            for i in range(n_layers):
                blk = vit.blocks[i]
                if hasattr(blk, "max_num_frames"):
                    x = blk(x, cu_slens=None, num_frames=cur_k)
                else:
                    x = blk(x)
            x = x.view(b, cur_k, -1, x.shape[-1])[:, -1]
            if args.stage == "vit":
                feat = x.reshape(1, -1, x.shape[-1])
            else:
                feat = visual.merger(x, (h, w)).reshape(1, -1, 2048)
            outs.append(feat[0].float().cpu().numpy())
            print(
                f"{key}: shape={tuple(feat.shape)} sum={float(feat.float().sum().cpu()):.12f} "
                f"mean={float(feat.float().mean().cpu()):.12f}"
            )

    arr = np.stack(outs, axis=0).astype(np.float32)
    dump_f32(args.dump_f32, arr)
    summary = {
        "ckpt": str(args.ckpt),
        "history": args.history,
        "layers": args.layers,
        "stage": args.stage,
        "shape": list(arr.shape),
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    args.dump_f32.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"dumped_f32={args.dump_f32}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
