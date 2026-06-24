#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Inspect a Hy-Embodied-0.5-VLA checkpoint without loading tensor payloads."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_CKPT = Path("/home/xuling/robotic_dataset/HY-VLA")


@dataclass
class TensorInfo:
    name: str
    group: str
    dtype: str
    shape: list[int]
    bytes: int | None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "unknown"
    value = float(n)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _shape_numel(shape: list[int]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return out


def _dtype_bytes(dtype: str) -> int | None:
    table = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }
    return table.get(dtype.upper())


def _classify_tensor(name: str) -> str:
    if name.startswith("model.dual_tower.vlm.model.visual."):
        return "vlm.vision"
    if name.startswith("model.dual_tower.vlm.model.language_model.model.embed_tokens."):
        return "vlm.text.embed_tokens"
    if name.startswith("model.dual_tower.vlm.model.language_model.model.layers."):
        rest = name.removeprefix("model.dual_tower.vlm.model.language_model.model.layers.")
        if ".self_attn." in rest:
            return "vlm.text.layers.attn"
        if ".mlp" in rest:
            return "vlm.text.layers.mlp"
        if "layernorm" in rest:
            return "vlm.text.layers.norm"
        return "vlm.text.layers.other"
    if name.startswith("model.dual_tower.vlm.model.language_model.model.norm."):
        return "vlm.text.norm"
    if name.startswith("model.dual_tower.vlm.model.language_model.lm_head."):
        return "vlm.text.lm_head"
    if name.startswith("model.dual_tower.vlm.language_model.model.embed_tokens."):
        return "vlm.text.embed_tokens"
    if name.startswith("model.dual_tower.vlm.language_model.model.layers."):
        rest = name.removeprefix("model.dual_tower.vlm.language_model.model.layers.")
        if ".self_attn." in rest:
            return "vlm.text.layers.attn"
        if ".mlp" in rest:
            return "vlm.text.layers.mlp"
        if "layernorm" in rest:
            return "vlm.text.layers.norm"
        return "vlm.text.layers.other"
    if name.startswith("model.dual_tower.vlm.language_model.model.norm."):
        return "vlm.text.norm"
    if name.startswith("model.dual_tower.vlm.language_model.lm_head."):
        return "vlm.text.lm_head"
    if name.startswith("model.dual_tower.vlm.model."):
        return "vlm.model_other"
    if name.startswith("model.dual_tower.vlm."):
        return "vlm.other"

    if name.startswith("model.dual_tower.expert.model.layers."):
        rest = name.removeprefix("model.dual_tower.expert.model.layers.")
        if ".self_attn." in rest:
            return "expert.layers.attn"
        if ".mlp" in rest:
            return "expert.layers.mlp"
        if "layernorm" in rest:
            return "expert.layers.norm"
        return "expert.layers.other"
    if name.startswith("model.dual_tower.expert.model.norm."):
        return "expert.norm"
    if name.startswith("model.dual_tower.expert.lm_head."):
        return "expert.lm_head"
    if name.startswith("model.dual_tower.expert."):
        return "expert.other"

    if name.startswith("model.action_in_proj."):
        return "action_in_proj"
    if name.startswith("model.action_out_proj."):
        return "action_out_proj"
    if name.startswith("model.state_proj."):
        return "state_proj"
    if name.startswith("model.action_time_mlp_in."):
        return "action_time_mlp_in"
    if name.startswith("model.action_time_mlp_out."):
        return "action_time_mlp_out"
    return "other"


def _prefix(name: str, depth: int) -> str:
    return ".".join(name.split(".")[:depth])


def _print_config(config: dict[str, Any]) -> None:
    print("\n## config snapshot")
    keys = [
        "chunk_size",
        "n_action_steps",
        "max_state_dim",
        "max_action_dim",
        "resize_imgs_with_padding",
        "tokenizer_max_length",
        "proj_width",
        "num_steps",
        "use_cache",
        "attention_implementation",
        "use_video_encoder",
        "spacetime_layer_stride",
        "past_drop_layer",
        "visual_segment_isolation",
        "architectures",
        "type",
    ]
    for key in keys:
        if key in config:
            print(f"{key}: {config[key]}")

    image_features = config.get("image_features", {})
    if image_features:
        print("image_features:")
        for key, value in image_features.items():
            print(f"- {key}: {value.get('shape')}")

    text_cfg = config.get("vlm_config_dict", {}).get("text_config", {})
    if text_cfg:
        print("vlm text_config:")
        for key in [
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "rope_theta",
            "rms_norm_eps",
            "use_qk_norm",
            "tie_word_embeddings",
        ]:
            if key in text_cfg:
                print(f"- {key}: {text_cfg[key]}")


def _print_norm_stats(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    print("\n## norm_stats.pkl")
    if not path.exists():
        print("missing")
        return out

    with path.open("rb") as fp:
        data = pickle.load(fp)

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            print(f"{prefix or '<root>'}: dict keys={list(value.keys())}")
            out[prefix or "<root>"] = {"type": "dict", "keys": list(value.keys())}
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                walk(child_prefix, child)
            return
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        try:
            checksum = float(value.sum())
        except Exception:
            checksum = None
        print(f"{prefix}: type={type(value).__name__}, shape={shape}, dtype={dtype}, sum={checksum}")
        out[prefix] = {
            "type": type(value).__name__,
            "shape": list(shape) if shape is not None else None,
            "dtype": str(dtype) if dtype is not None else None,
            "sum": checksum,
        }

    walk("", data)
    return out


def _read_safetensors(path: Path) -> list[TensorInfo]:
    try:
        from safetensors import safe_open  # type: ignore
    except Exception as exc:
        raise RuntimeError("safetensors is required for tensor header inspection") from exc

    infos: list[TensorInfo] = []
    with safe_open(path, framework="pt", device="cpu") as sf:
        for name in sorted(sf.keys()):
            meta = sf.get_slice(name)
            shape = list(meta.get_shape())
            dtype = str(meta.get_dtype())
            item_bytes = _dtype_bytes(dtype)
            nbytes = None if item_bytes is None else _shape_numel(shape) * item_bytes
            infos.append(
                TensorInfo(
                    name=name,
                    group=_classify_tensor(name),
                    dtype=dtype,
                    shape=shape,
                    bytes=nbytes,
                )
            )
    return infos


def _print_tensor_summary(infos: list[TensorInfo]) -> dict[str, Any]:
    print("\n## model.safetensors")
    print(f"tensors: {len(infos)}")
    print(f"estimated tensor bytes: {_fmt_bytes(sum(i.bytes or 0 for i in infos))}")

    dtype_counts = Counter(i.dtype for i in infos)
    print("dtype counts:")
    for dtype, count in dtype_counts.most_common():
        total = sum(i.bytes or 0 for i in infos if i.dtype == dtype)
        print(f"- {dtype}: tensors={count}, bytes={_fmt_bytes(total)}")

    by_group: dict[str, list[TensorInfo]] = defaultdict(list)
    for info in infos:
        by_group[info.group].append(info)

    print("\nmodule groups:")
    group_summary: dict[str, Any] = {}
    for group in sorted(by_group):
        group_infos = by_group[group]
        total = sum(i.bytes or 0 for i in group_infos)
        params = sum(_shape_numel(i.shape) for i in group_infos)
        print(f"- {group}: tensors={len(group_infos)}, params={params:,}, bytes={_fmt_bytes(total)}")
        group_summary[group] = {
            "tensors": len(group_infos),
            "params": params,
            "bytes": total,
            "dtypes": dict(Counter(i.dtype for i in group_infos)),
        }

    print("\ntop prefixes depth=4:")
    p4 = Counter(_prefix(i.name, 4) for i in infos)
    for prefix, count in p4.most_common(24):
        print(f"- {prefix}: {count}")

    print("\nlargest tensors:")
    for info in sorted(infos, key=lambda x: x.bytes or 0, reverse=True)[:24]:
        print(f"- {info.name}: shape={info.shape}, dtype={info.dtype}, bytes={_fmt_bytes(info.bytes)}")

    layer_counts: dict[str, set[int]] = defaultdict(set)
    for info in infos:
        parts = info.name.split(".")
        for idx, part in enumerate(parts[:-1]):
            if part == "layers" and idx + 1 < len(parts):
                try:
                    layer_idx = int(parts[idx + 1])
                except ValueError:
                    continue
                owner = ".".join(parts[:idx])
                layer_counts[owner].add(layer_idx)

    print("\nlayer index ranges:")
    layer_summary: dict[str, Any] = {}
    for owner, indices in sorted(layer_counts.items()):
        sorted_indices = sorted(indices)
        print(f"- {owner}.layers: count={len(sorted_indices)}, min={sorted_indices[0]}, max={sorted_indices[-1]}")
        layer_summary[owner] = {
            "count": len(sorted_indices),
            "min": sorted_indices[0],
            "max": sorted_indices[-1],
        }

    return {
        "tensor_count": len(infos),
        "estimated_bytes": sum(i.bytes or 0 for i in infos),
        "dtype_counts": dict(dtype_counts),
        "groups": group_summary,
        "layer_ranges": layer_summary,
    }


def inspect_checkpoint(root: Path, json_out: Path | None) -> int:
    root = root.expanduser().resolve()
    print("# Hy-Embodied-0.5-VLA checkpoint inspection")
    print(f"root: {root}")
    if not root.exists():
        print(f"error: checkpoint root does not exist: {root}")
        return 2

    config_path = root / "config.json"
    model_path = root / "model.safetensors"
    if not config_path.exists():
        print(f"error: missing config.json at {config_path}")
        return 2
    if not model_path.exists():
        print(f"error: missing model.safetensors at {model_path}")
        return 2

    config = _load_json(config_path)
    _print_config(config)
    norm_stats = _print_norm_stats(root / "norm_stats.pkl")
    infos = _read_safetensors(model_path)
    tensor_summary = _print_tensor_summary(infos)

    if json_out is not None:
        payload = {
            "root": str(root),
            "config": config,
            "norm_stats": norm_stats,
            "tensor_summary": tensor_summary,
            "tensors": [asdict(info) for info in infos],
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\njson_out: {json_out}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    return inspect_checkpoint(args.ckpt, args.json_out)


if __name__ == "__main__":
    raise SystemExit(main())
