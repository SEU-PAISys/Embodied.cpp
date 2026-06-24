#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Inspect a LingBot-VA checkpoint directory without loading full weights.

The script is intentionally useful before the full safetensors download is
complete: it reads config and index JSON files with the Python standard
library, reports expected shards, and marks which files are currently missing.
If safetensors is installed and shards are present, it can also summarize
actual tensor metadata from shard headers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_HF_SNAPSHOT = Path(
    "/home/xuling/.cache/huggingface/hub/"
    "models--robbyant--lingbot-va-posttrain-libero-long/"
    "snapshots/0e89d1e753019988aba484e8da2dc0810e264d9f"
)


@dataclass
class IndexedModule:
    name: str
    path: Path
    config_name: str
    index_name: str | None


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


def _summarize_config(name: str, config: dict[str, Any]) -> None:
    print(f"\n## {name} config")
    keys = [
        "_class_name",
        "architectures",
        "model_type",
        "num_layers",
        "num_attention_heads",
        "attention_head_dim",
        "hidden_size",
        "d_model",
        "ffn_dim",
        "in_channels",
        "out_channels",
        "action_dim",
        "text_dim",
        "patch_size",
        "attn_mode",
        "torch_dtype",
    ]
    for key in keys:
        if key in config:
            print(f"{key}: {config[key]}")


def _module_from_index(mod: IndexedModule) -> dict[str, Any] | None:
    if mod.index_name is None:
        return None
    index_path = mod.path / mod.index_name
    if not index_path.exists():
        return None
    return _load_json(index_path)


def _summarize_index(mod: IndexedModule, index: dict[str, Any]) -> None:
    weight_map = index.get("weight_map", {})
    metadata = index.get("metadata", {})
    total_size = metadata.get("total_size")
    print(f"\n## {mod.name} index")
    print(f"tensors: {len(weight_map)}")
    print(f"metadata.total_size: {_fmt_bytes(total_size)}")

    shard_to_tensors: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard_name in weight_map.items():
        shard_to_tensors[str(shard_name)].append(str(tensor_name))

    print(f"expected shards: {len(shard_to_tensors)}")
    total_present_size = 0
    missing = []
    for shard_name, tensors in sorted(shard_to_tensors.items()):
        shard_path = mod.path / shard_name
        if shard_path.exists():
            size = shard_path.stat().st_size
            total_present_size += size
            status = f"present, {_fmt_bytes(size)}"
        else:
            missing.append(shard_name)
            status = "missing"
        print(f"- {shard_name}: {status}, tensors={len(tensors)}")
    print(f"present shard bytes: {_fmt_bytes(total_present_size)}")
    if missing:
        print(f"missing shards: {len(missing)}")
    else:
        print("missing shards: 0")

    top_prefix = Counter(name.split(".", 1)[0] for name in weight_map)
    print("top tensor prefixes:")
    for prefix, count in top_prefix.most_common(16):
        print(f"- {prefix}: {count}")


def _summarize_safetensors_headers(mod: IndexedModule, index: dict[str, Any]) -> None:
    try:
        from safetensors import safe_open  # type: ignore
    except Exception:
        print(f"\n## {mod.name} safetensors headers")
        print("safetensors is not installed; skipping shard header inspection.")
        return

    weight_map = index.get("weight_map", {})
    shard_names = sorted(set(str(v) for v in weight_map.values()))
    dtype_counts: Counter[str] = Counter()
    tensor_count = 0
    estimated_bytes = 0
    unavailable = []

    for shard_name in shard_names:
        shard_path = mod.path / shard_name
        if not shard_path.exists():
            unavailable.append(shard_name)
            continue
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                info = sf.get_slice(key)
                shape = list(info.get_shape())
                dtype = str(info.get_dtype())
                tensor_count += 1
                dtype_counts[dtype] += 1
                nbytes = _dtype_bytes(dtype)
                if nbytes is not None:
                    estimated_bytes += _shape_numel(shape) * nbytes

    print(f"\n## {mod.name} safetensors headers")
    print(f"readable tensors: {tensor_count}")
    print(f"estimated tensor bytes from headers: {_fmt_bytes(estimated_bytes)}")
    if dtype_counts:
        print("dtype counts:")
        for dtype, count in dtype_counts.most_common():
            print(f"- {dtype}: {count}")
    if unavailable:
        print(f"unavailable shards skipped: {len(unavailable)}")


def inspect_checkpoint(root: Path) -> int:
    root = root.expanduser().resolve()
    print(f"# LingBot-VA checkpoint inspection")
    print(f"root: {root}")
    if not root.exists():
        print(f"error: root does not exist: {root}")
        return 2

    modules = [
        IndexedModule("text_encoder", root / "text_encoder", "config.json", "model.safetensors.index.json"),
        IndexedModule("transformer", root / "transformer", "config.json", "diffusion_pytorch_model.safetensors.index.json"),
        IndexedModule("vae", root / "vae", "config.json", None),
        IndexedModule("tokenizer", root / "tokenizer", "tokenizer_config.json", None),
    ]

    for mod in modules:
        print(f"\n# Module: {mod.name}")
        print(f"path: {mod.path}")
        if not mod.path.exists():
            print("status: missing directory")
            continue
        config_path = mod.path / mod.config_name
        if config_path.exists():
            _summarize_config(mod.name, _load_json(config_path))
        else:
            print(f"missing config: {config_path.name}")

        index = _module_from_index(mod)
        if index is not None:
            _summarize_index(mod, index)
            _summarize_safetensors_headers(mod, index)
        elif mod.index_name is not None:
            print(f"missing index: {mod.index_name}")
        else:
            safes = sorted(mod.path.glob("*.safetensors"))
            if safes:
                print("safetensors files:")
                for path in safes:
                    print(f"- {path.name}: {_fmt_bytes(path.stat().st_size)}")
            else:
                print("safetensors files: none found")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=DEFAULT_HF_SNAPSHOT,
        help="LingBot-VA checkpoint root containing text_encoder/, transformer/, vae/, tokenizer/.",
    )
    return inspect_checkpoint(parser.parse_args().checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
