#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Inspect HY-VLA HyViT2/Merger vision weights.

The full HY-VLA GGUF conversion already preserves all remaining checkpoint
tensors.  This helper gives the C++ port a compact inventory of the visual
frontend that still needs native GGML/CUDA implementation:

  * HyViT2 patch embedding and absolute position embedding;
  * 27 ViT blocks with qkv/proj/norm/mlp tensors;
  * MLP projector/merger and normalized depthwise pooler.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from safetensors import safe_open
except ImportError as exc:  # pragma: no cover
    raise SystemExit("safetensors is required; run with the HY-VLA Python env") from exc


DEFAULT_CKPT = Path("/home/xuling/robotic_dataset/HY-VLA/model.safetensors")
VIS_PREFIX = "model.dual_tower.vlm.model.visual."


def nparams(shape: list[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def tensor_info(sf, name: str) -> dict[str, Any]:
    sl = sf.get_slice(name)
    shape = list(sl.get_shape())
    return {
        "name": name,
        "shape": shape,
        "dtype": str(sl.get_dtype()),
        "params": nparams(shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safetensors", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    with safe_open(args.safetensors, framework="pt", device="cpu") as sf:
        keys = [k for k in sf.keys() if k.startswith(VIS_PREFIX)]
        if not keys:
            raise SystemExit(f"no visual tensors found under {VIS_PREFIX}")

        vision_keys = [k for k in keys if ".vision_tower." in k]
        merger_keys = [k for k in keys if ".merger." in k]

        block_re = re.compile(r"\.vision_tower\.blocks\.(\d+)\.")
        block_ids = sorted({int(m.group(1)) for k in vision_keys if (m := block_re.search(k))})
        per_block: dict[str, dict[str, Any]] = {}
        for bid in block_ids:
            prefix = f"{VIS_PREFIX}vision_tower.blocks.{bid}."
            bkeys = sorted(k for k in vision_keys if k.startswith(prefix))
            per_block[str(bid)] = {
                "tensor_count": len(bkeys),
                "params": sum(nparams(list(sf.get_slice(k).get_shape())) for k in bkeys),
                "tensors": [
                    {
                        "leaf": k[len(prefix):],
                        "shape": list(sf.get_slice(k).get_shape()),
                        "dtype": str(sf.get_slice(k).get_dtype()),
                    }
                    for k in bkeys
                ],
            }

        groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"tensor_count": 0, "params": 0})
        for k in keys:
            rel = k[len(VIS_PREFIX):]
            if rel.startswith("vision_tower.blocks."):
                group = "vision_tower.blocks"
            elif rel.startswith("vision_tower.patch_embed."):
                group = "vision_tower.patch_embed"
            elif rel.startswith("vision_tower.pos_embed"):
                group = "vision_tower.pos_embed"
            elif rel.startswith("merger.pooler."):
                group = "merger.pooler"
            elif rel.startswith("merger."):
                group = "merger.proj"
            else:
                group = rel.split(".", 1)[0]
            groups[group]["tensor_count"] += 1
            groups[group]["params"] += nparams(list(sf.get_slice(k).get_shape()))

        report = {
            "safetensors": str(args.safetensors),
            "visual_prefix": VIS_PREFIX,
            "total": {
                "tensor_count": len(keys),
                "params": sum(nparams(list(sf.get_slice(k).get_shape())) for k in keys),
            },
            "groups": dict(sorted(groups.items())),
            "vision_config_inferred": {
                "width": 1152,
                "layers": len(block_ids),
                "heads": 16,
                "head_dim": 72,
                "patch_size": 16,
                "input_image_size_for_current_config": [224, 224],
                "patch_grid": [14, 14],
                "merged_visual_tokens_per_image": 49,
                "merger_out_channels": 2048,
            },
            "key_tensors": {
                "patch_embed": tensor_info(sf, f"{VIS_PREFIX}vision_tower.patch_embed.proj.weight"),
                "pos_embed": tensor_info(sf, f"{VIS_PREFIX}vision_tower.pos_embed"),
                "merger_proj1": tensor_info(sf, f"{VIS_PREFIX}merger.proj1.weight"),
                "merger_proj2": tensor_info(sf, f"{VIS_PREFIX}merger.proj2.weight"),
                "pooler_fc0": tensor_info(sf, f"{VIS_PREFIX}merger.pooler.predictor.0.weight"),
                "pooler_fc2": tensor_info(sf, f"{VIS_PREFIX}merger.pooler.predictor.2.weight"),
            },
            "per_block": per_block,
            "cpp_port_order": [
                "image preprocess: RGB resize/pad to 224x224, rescale 1/255, normalize mean/std 0.5",
                "patch_embed conv: [1152,3,16,16] -> 14x14x1152",
                "pos_embed sample/rescale from [1,16384,1152] to 14x14",
                "27 ViT blocks: LayerNorm + fused qkv attention + MLP GELU",
                "merger: 14x14x1152 -> proj1 -> 2x2 normalized pool -> GELU -> proj2 -> 7x7x2048",
            ],
        }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if k != "per_block"}, indent=2, ensure_ascii=False))
    print("\nper_block_template:")
    first = report["per_block"].get("0")
    print(json.dumps(first, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
