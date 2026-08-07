#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepare and convert GR00T N1.7's embedded Qwen3-VL backbone.

GR00T stores the already-truncated backbone under ``backbone.model.*`` inside
the policy safetensors. This script strips that prefix while copying raw tensor
payloads, combines them with local Qwen3-VL/Cosmos metadata, and can directly
write the text-backbone, multimodal-projector, and action-head GGUF files.
Payloads are streamed, so preparing the multi-GiB checkpoint does not require
loading the model into RAM.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any, BinaryIO

from convert_groot_n1_to_gguf import DEFAULT_EMBODIMENT, convert as convert_action_head
from gguf_quantize import add_outtype_args, quantize_gguf


DEFAULT_METADATA_REPO = "Qwen/Qwen3-VL-2B-Instruct"
SOURCE_PREFIX = "backbone.model."
COPY_BUFFER_SIZE = 16 * 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parents[1]
HF_CONVERTER = REPO_ROOT / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py"
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

QWEN3VL_TEXT_MATMUL_RE = re.compile(
    r"^(?:(?:token_embd|output)\.weight|blk\.\d+\."
    r"(?:attn_(?:q|k|v|output)|ffn_(?:gate|up|down))\.weight)$"
)
QWEN3VL_MMPROJ_MATMUL_RE = re.compile(
    r"^(?:v\.blk\.\d+\.(?:attn_(?:out|qkv)|ffn_(?:up|down))|"
    r"v\.deepstack\.\d+\.fc[12]|mm\.\d+)\.weight$"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


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
    single_path = checkpoint_dir / "model.safetensors"
    if index_path.is_file():
        index = _load_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"missing weight_map in {index_path}")
    elif single_path.is_file():
        _, header = _read_safetensors_header(single_path)
        weight_map = {
            name: single_path.name
            for name, metadata in header.items()
            if name != "__metadata__" and isinstance(metadata, dict)
        }
    else:
        raise FileNotFoundError(
            f"missing {index_path.name} or {single_path.name} under {checkpoint_dir}"
        )

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
    missing_shards = sorted(path for path in by_shard if not path.is_file())
    if missing_shards:
        raise FileNotFoundError(
            f"checkpoint index references missing shard {missing_shards[0]}"
        )
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


def _copy_local_metadata(
    qwen3vl_dir: Path | None,
    cosmos_dir: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    directories = [path for path in (qwen3vl_dir, cosmos_dir) if path is not None]
    if not directories:
        raise ValueError("at least one local Qwen3-VL or Cosmos directory is required")

    config: dict[str, Any] = {}
    for directory in directories:
        config_path = directory / "config.json"
        if config_path.is_file():
            config = _deep_merge(config, _load_json(config_path))
    if not config:
        raise FileNotFoundError("neither local model directory contains config.json")

    # Cosmos is the actual GR00T backbone model, so prefer its metadata files;
    # Qwen3-VL supplies any tokenizer/processor files Cosmos does not carry.
    preferred = [path for path in (cosmos_dir, qwen3vl_dir) if path is not None]
    for name in METADATA_FILES:
        if name == "config.json":
            continue
        source = next((directory / name for directory in preferred if (directory / name).is_file()), None)
        if source is None:
            raise FileNotFoundError(
                f"missing local metadata file {name}; searched "
                + ", ".join(str(path) for path in preferred)
            )
        destination = output_dir / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "metadata: local Cosmos overrides local Qwen3-VL; sources="
        + ", ".join(str(path) for path in directories)
    )
    return config


def _download_metadata(repo_id: str, output_dir: Path) -> dict[str, Any]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required only for explicit --metadata-repo use; "
            "pass --qwen3vl-dir/--cosmos-dir for offline conversion"
        ) from exc
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

    missing_metadata = [name for name in METADATA_FILES if not (output_dir / name).is_file()]
    if missing_metadata:
        raise ValueError(
            f"prepared backbone is missing metadata file {missing_metadata[0]}"
        )
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


def _run_hf_converter(
    prepared_dir: Path,
    output: Path,
    *,
    mmproj: bool,
) -> None:
    if not HF_CONVERTER.is_file():
        raise SystemExit(f"missing llama.cpp converter: {HF_CONVERTER}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(HF_CONVERTER),
        str(prepared_dir),
        "--outfile",
        str(output),
        "--outtype",
        "bf16",
        "--use-temp-file",
    ]
    if mmproj:
        command.append("--mmproj")
    label = "Qwen3-VL mmproj" if mmproj else "Qwen3-VL text backbone"
    print(f"converting {label} to temporary BF16 GGUF: {output}")
    environment = os.environ.copy()
    environment["TMPDIR"] = str(output.parent)
    subprocess.run(command, check=True, env=environment)
    if not output.is_file():
        raise RuntimeError(f"llama.cpp converter did not create {output}")


def _is_qwen3vl_text_matrix(tensor: Any) -> bool:
    return QWEN3VL_TEXT_MATMUL_RE.fullmatch(tensor.name) is not None


def _is_qwen3vl_mmproj_matrix(tensor: Any) -> bool:
    return QWEN3VL_MMPROJ_MATMUL_RE.fullmatch(tensor.name) is not None


def _write_backbone_ggufs(
    prepared_dir: Path,
    backbone_output: Path,
    mmproj_output: Path,
    outtype: str,
    ggml_lib: Path,
) -> None:
    if outtype == "bf16":
        _run_hf_converter(prepared_dir, backbone_output, mmproj=False)
        _run_hf_converter(prepared_dir, mmproj_output, mmproj=True)
        return

    backbone_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="groot-qwen3vl-bf16-",
        dir=str(backbone_output.parent),
    ) as temporary_dir:
        temporary = Path(temporary_dir)
        backbone_bf16 = temporary / "qwen3vl-backbone-bf16.gguf"
        mmproj_bf16 = temporary / "qwen3vl-mmproj-bf16.gguf"
        _run_hf_converter(prepared_dir, backbone_bf16, mmproj=False)
        _run_hf_converter(prepared_dir, mmproj_bf16, mmproj=True)

        print(f"quantizing Qwen3-VL text backbone to {outtype.upper()}: {backbone_output}")
        quantize_gguf(
            backbone_bf16,
            backbone_output,
            outtype,
            ggml_lib,
            arch="qwen3vl",
            should_quantize=_is_qwen3vl_text_matrix,
            metadata_key="groot_n1.backbone_quantization",
        )
        print(f"quantizing Qwen3-VL mmproj to {outtype.upper()}: {mmproj_output}")
        quantize_gguf(
            mmproj_bf16,
            mmproj_output,
            outtype,
            ggml_lib,
            arch="clip",
            should_quantize=_is_qwen3vl_mmproj_matrix,
            metadata_key="groot_n1.mmproj_quantization",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--qwen3vl-dir",
        "--qwen3vl",
        dest="qwen3vl_dir",
        type=Path,
        default=None,
        help="Local Qwen3-VL model directory containing tokenizer/processor metadata.",
    )
    parser.add_argument(
        "--cosmos-dir",
        "--cosmos",
        dest="cosmos_dir",
        type=Path,
        default=None,
        help="Local Cosmos-Reason2 model directory; its config/metadata takes priority.",
    )
    parser.add_argument(
        "--metadata-repo",
        default=None,
        help=(
            f"Explicit online metadata fallback (for example {DEFAULT_METADATA_REPO}); "
            "never used when local model directories are supplied."
        ),
    )
    parser.add_argument("--select-layer", type=int, default=None)
    parser.add_argument(
        "--gguf-dir",
        type=Path,
        default=None,
        help="Directory for all three GGUF files (default: parent of --output-dir).",
    )
    parser.add_argument("--backbone-out", type=Path, default=None)
    parser.add_argument("--mmproj-out", type=Path, default=None)
    parser.add_argument("--action-head-out", type=Path, default=None)
    parser.add_argument("--embodiment", default=DEFAULT_EMBODIMENT)
    parser.add_argument(
        "--reuse-prepared",
        action="store_true",
        help="Reuse and validate --output-dir instead of extracting safetensors again.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare the local Hugging Face directory without writing GGUF files.",
    )
    add_outtype_args(parser)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    qwen3vl_dir = args.qwen3vl_dir.expanduser().resolve() if args.qwen3vl_dir else None
    cosmos_dir = args.cosmos_dir.expanduser().resolve() if args.cosmos_dir else None
    for label, directory in (("Qwen3-VL", qwen3vl_dir), ("Cosmos", cosmos_dir)):
        if directory is not None and not directory.is_dir():
            raise SystemExit(f"local {label} directory does not exist: {directory}")
        if directory is not None and directory == output_dir:
            raise SystemExit(f"--output-dir must differ from the local {label} source directory")

    policy_config = _load_json(checkpoint_dir / "config.json")
    select_layer = int(
        args.select_layer
        if args.select_layer is not None
        else policy_config["select_layer"]
    )
    if args.reuse_prepared:
        _validate_output(output_dir, select_layer)
        tensor_count = 0
        payload_bytes = (output_dir / "model.safetensors").stat().st_size
        print(f"reusing prepared Qwen3-VL directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        if qwen3vl_dir is not None or cosmos_dir is not None:
            metadata_config = _copy_local_metadata(qwen3vl_dir, cosmos_dir, output_dir)
        elif args.metadata_repo:
            metadata_config = _download_metadata(args.metadata_repo, output_dir)
        else:
            raise SystemExit(
                "offline metadata is required: pass --qwen3vl-dir and/or --cosmos-dir; "
                "use --metadata-repo only when an explicit download is intended"
            )
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
    if args.prepare_only:
        return 0

    gguf_dir = (
        args.gguf_dir.expanduser().resolve()
        if args.gguf_dir
        else output_dir.parent
    )
    gguf_dir.mkdir(parents=True, exist_ok=True)
    backbone_output = (
        args.backbone_out.expanduser().resolve()
        if args.backbone_out
        else gguf_dir / f"qwen3vl-backbone-{args.outtype}.gguf"
    )
    mmproj_output = (
        args.mmproj_out.expanduser().resolve()
        if args.mmproj_out
        else gguf_dir / f"qwen3vl-mmproj-{args.outtype}.gguf"
    )
    action_head_output = (
        args.action_head_out.expanduser().resolve()
        if args.action_head_out
        else gguf_dir /
        f"groot-n1.7-libero-object-action-head-{args.outtype}.gguf"
    )

    _write_backbone_ggufs(
        output_dir,
        backbone_output,
        mmproj_output,
        args.outtype,
        args.ggml_lib,
    )
    convert_action_head(
        checkpoint_dir,
        action_head_output,
        args.embodiment,
        False,
        args.outtype,
        args.ggml_lib,
    )
    print("generated GR00T N1.7 GGUF artifacts:")
    print(f"  action head: {action_head_output}")
    print(f"  backbone:    {backbone_output}")
    print(f"  mmproj:      {mmproj_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())