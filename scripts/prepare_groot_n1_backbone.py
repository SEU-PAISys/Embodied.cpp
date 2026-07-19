#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepare GR00T N1.7's embedded Qwen3-VL backbone for GGUF conversion.

GR00T stores the already-truncated backbone under ``backbone.model.*`` inside
the policy safetensors. This script strips that prefix while copying raw tensor
payloads, then combines them with the public Qwen3-VL tokenizer and processor
metadata. Payloads are streamed, so preparing the multi-GiB checkpoint does not
require loading the model into RAM.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import struct
from typing import Any, BinaryIO

from huggingface_hub import hf_hub_download


DEFAULT_METADATA_REPO = "Qwen/Qwen3-VL-2B-Instruct"
SOURCE_PREFIX = "backbone.model."
COPY_BUFFER_SIZE = 16 * 1024 * 1024
METADATA_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header_raw = handle.read(header_length)
    if len(header_raw) != header_length:
        raise ValueError(f"truncated safetensors header in {path}")
    header = json.loads(header_raw.rstrip(b" ").decode("utf-8"))
    if not isinstance(header, dict):
        raise TypeError(f"invalid safetensors JSON header in {path}")
    return 8 + header_length, header


def _copy_range(
    source: BinaryIO,
    destination: BinaryIO,
    offset: int,
    size: int,
) -> None:
    source.seek(offset)
    remaining = size
    while remaining:
        chunk = source.read(min(COPY_BUFFER_SIZE, remaining))
        if not chunk:
            raise EOFError("unexpected end of safetensors payload")
        destination.write(chunk)
        remaining -= len(chunk)


def _language_layer(name: str) -> int | None:
    marker = "model.language_model.layers."
    if not name.startswith(marker):
        return None
    return int(name[len(marker):].split(".", 1)[0])


def _select_backbone_tensors(
    checkpoint_dir: Path,
    select_layer: int,
) -> dict[Path, list[tuple[str, str]]]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    index = _load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"missing weight_map in {index_path}")

    by_shard: dict[Path, list[tuple[str, str]]] = defaultdict(list)
    for source_name, shard_name in weight_map.items():
        if not source_name.startswith(SOURCE_PREFIX):
            continue
        destination_name = source_name.removeprefix(SOURCE_PREFIX)
        layer = _language_layer(destination_name)
        if layer is not None and layer >= select_layer:
            continue
        by_shard[checkpoint_dir / str(shard_name)].append(
            (source_name, destination_name)
        )
    if not by_shard:
        raise ValueError(f"no {SOURCE_PREFIX} tensors found in {index_path}")
    return dict(by_shard)


def _write_backbone_safetensors(
    checkpoint_dir: Path,
    output: Path,
    select_layer: int,
) -> tuple[int, int]:
    selected = _select_backbone_tensors(checkpoint_dir, select_layer)
    output_header: dict[str, Any] = {
        "__metadata__": {"format": "pt"},
    }
    copy_plan: list[tuple[Path, int, int]] = []
    payload_offset = 0

    for shard_path, names in sorted(selected.items()):
        data_start, source_header = _read_safetensors_header(shard_path)
        for source_name, destination_name in sorted(names):
            tensor = source_header.get(source_name)
            if not isinstance(tensor, dict):
                raise KeyError(f"{source_name} missing from {shard_path}")
            offsets = tensor.get("data_offsets")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"invalid data_offsets for {source_name}")
            source_begin, source_end = (int(offsets[0]), int(offsets[1]))
            size = source_end - source_begin
            if size < 0:
                raise ValueError(f"negative payload size for {source_name}")
            output_header[destination_name] = {
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "data_offsets": [payload_offset, payload_offset + size],
            }
            copy_plan.append((shard_path, data_start + source_begin, size))
            payload_offset += size

    encoded_header = json.dumps(
        output_header,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    padded_header_length = (len(encoded_header) + 7) // 8 * 8
    encoded_header += b" " * (padded_header_length - len(encoded_header))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as destination:
        destination.write(struct.pack("<Q", padded_header_length))
        destination.write(encoded_header)
        current_path: Path | None = None
        source: BinaryIO | None = None
        try:
            for shard_path, source_offset, size in copy_plan:
                if shard_path != current_path:
                    if source is not None:
                        source.close()
                    source = shard_path.open("rb")
                    current_path = shard_path
                assert source is not None
                _copy_range(source, destination, source_offset, size)
        finally:
            if source is not None:
                source.close()
    os.replace(temporary, output)
    return len(copy_plan), payload_offset


def _download_metadata(repo_id: str, output_dir: Path) -> dict[str, Any]:
    for name in METADATA_FILES:
        cached = Path(hf_hub_download(repo_id, name))
        shutil.copy2(cached, output_dir / name)
    return _load_json(output_dir / "config.json")


def _configure_backbone(
    config: dict[str, Any],
    output_path: Path,
    select_layer: int,
) -> None:
    architectures = config.get("architectures")
    if architectures != ["Qwen3VLForConditionalGeneration"]:
        raise ValueError(f"unexpected metadata architecture: {architectures}")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("metadata config is missing text_config")
    original_layers = int(text_config["num_hidden_layers"])
    if select_layer < 1 or select_layer > original_layers:
        raise ValueError(
            f"select_layer={select_layer} is outside metadata layer count {original_layers}"
        )
    text_config["num_hidden_layers"] = select_layer
    text_config["tie_word_embeddings"] = True
    config["tie_word_embeddings"] = True
    config["embedding_output_pre_norm"] = True
    config["name_or_path"] = "nvidia/Cosmos-Reason2-2B"
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _validate_output(output_dir: Path, select_layer: int) -> None:
    from safetensors import safe_open

    model_path = output_dir / "model.safetensors"
    with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
        names = set(checkpoint.keys())
        required = {
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "model.visual.patch_embed.proj.weight",
            f"model.language_model.layers.{select_layer - 1}.self_attn.q_proj.weight",
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"prepared backbone is missing {missing[0]}")
        excess_layers = sorted(
            name for name in names
            if (layer := _language_layer(name)) is not None and layer >= select_layer
        )
        if excess_layers:
            raise ValueError(f"prepared backbone contains excess layer: {excess_layers[0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-repo", default=DEFAULT_METADATA_REPO)
    parser.add_argument("--select-layer", type=int, default=None)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    policy_config = _load_json(checkpoint_dir / "config.json")
    select_layer = int(
        args.select_layer
        if args.select_layer is not None
        else policy_config["select_layer"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_config = _download_metadata(args.metadata_repo, output_dir)
    _configure_backbone(
        metadata_config,
        output_dir / "config.json",
        select_layer,
    )
    tensor_count, payload_bytes = _write_backbone_safetensors(
        checkpoint_dir,
        output_dir / "model.safetensors",
        select_layer,
    )
    _validate_output(output_dir, select_layer)
    print(
        f"prepared {tensor_count} Qwen3-VL tensors, "
        f"layers={select_layer}, payload={payload_bytes / (1024 ** 3):.2f} GiB at "
        f"{output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())