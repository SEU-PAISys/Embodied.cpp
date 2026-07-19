#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Convert a GR00T N1.7 action head checkpoint to GGUF.

The Qwen3-VL backbone remains a normal llama.cpp text model plus mtmd projector.
This converter writes the GR00T-specific state/action projectors, VL adapter,
AlternateVLDiT, flow schedule, and embodiment normalization statistics.
"""

from __future__ import annotations

import argparse
import json
from math import prod
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if GGUF_PY.exists():
    sys.path.insert(0, str(GGUF_PY))

try:
    import gguf
except Exception as exc:
    raise SystemExit(
        "failed to import gguf; use the embodiedcpp environment and ensure "
        f"{GGUF_PY} exists. Original error: {exc}"
    ) from exc


ARCH = "groot_n1"
KV = lambda name: f"{ARCH}.{name}"
DEFAULT_EMBODIMENT = "libero_sim"


class ShardedCheckpoint:
    def __init__(self, checkpoint_dir: Path):
        index_path = checkpoint_dir / "model.safetensors.index.json"
        single_path = checkpoint_dir / "model.safetensors"
        if index_path.is_file():
            index = _load_json(index_path)
            self.weight_map = {
                str(name): checkpoint_dir / str(shard)
                for name, shard in index["weight_map"].items()
            }
        elif single_path.is_file():
            with safe_open(single_path, framework="pt", device="cpu") as sf:
                self.weight_map = {name: single_path for name in sf.keys()}
        else:
            raise SystemExit(
                f"missing {index_path.name} or {single_path.name} under {checkpoint_dir}"
            )
        self._readers: dict[Path, Any] = {}

    @property
    def keys(self) -> set[str]:
        return set(self.weight_map)

    def _reader(self, path: Path):
        if path not in self._readers:
            if not path.is_file():
                raise SystemExit(f"checkpoint index references missing shard {path}")
            self._readers[path] = safe_open(path, framework="pt", device="cpu")
        return self._readers[path]

    def get_tensor(self, name: str) -> torch.Tensor:
        try:
            path = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"checkpoint tensor not found: {name}") from exc
        return self._reader(path).get_tensor(name)

    def get_shape(self, name: str) -> list[int]:
        path = self.weight_map[name]
        return list(self._reader(path).get_slice(name).get_shape())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return value


def _tensor_rows(config: dict[str, Any]) -> list[tuple[str, str, bool]]:
    rows: list[tuple[str, str, bool]] = []

    def add(src: str, dst: str, category: bool = False) -> None:
        rows.append((f"action_head.{src}", dst, category))

    category_layers = {
        "state_encoder.layer1": "state_encoder.0",
        "state_encoder.layer2": "state_encoder.2",
        "action_encoder.W1": "action_encoder.w1",
        "action_encoder.W2": "action_encoder.w2",
        "action_encoder.W3": "action_encoder.w3",
        "action_decoder.layer1": "action_decoder.0",
        "action_decoder.layer2": "action_decoder.2",
    }
    for source, destination in category_layers.items():
        add(f"{source}.W", f"{destination}.weight", category=True)
        add(f"{source}.b", f"{destination}.bias", category=True)

    add("position_embedding.weight", "action_position.weight")
    add("vlln.weight", "vl.norm.weight")
    add("vlln.bias", "vl.norm.bias")

    vl_cfg = config.get("vl_self_attention_cfg") or {}
    for layer in range(int(vl_cfg.get("num_layers", 0))):
        source = f"vl_self_attention.transformer_blocks.{layer}"
        destination = f"vl.blk.{layer}"
        for suffix, mapped in (
            ("norm1.weight", "attn_norm.weight"),
            ("norm1.bias", "attn_norm.bias"),
            ("attn1.to_q.weight", "attn_q.weight"),
            ("attn1.to_q.bias", "attn_q.bias"),
            ("attn1.to_k.weight", "attn_k.weight"),
            ("attn1.to_k.bias", "attn_k.bias"),
            ("attn1.to_v.weight", "attn_v.weight"),
            ("attn1.to_v.bias", "attn_v.bias"),
            ("attn1.to_out.0.weight", "attn_out.weight"),
            ("attn1.to_out.0.bias", "attn_out.bias"),
            ("norm3.weight", "ffn_norm.weight"),
            ("norm3.bias", "ffn_norm.bias"),
            ("ff.net.0.proj.weight", "ffn_up.weight"),
            ("ff.net.0.proj.bias", "ffn_up.bias"),
            ("ff.net.2.weight", "ffn_down.weight"),
            ("ff.net.2.bias", "ffn_down.bias"),
        ):
            add(f"{source}.{suffix}", f"{destination}.{mapped}")

    add(
        "model.timestep_encoder.timestep_embedder.linear_1.weight",
        "dit.time.0.weight",
    )
    add(
        "model.timestep_encoder.timestep_embedder.linear_1.bias",
        "dit.time.0.bias",
    )
    add(
        "model.timestep_encoder.timestep_embedder.linear_2.weight",
        "dit.time.2.weight",
    )
    add(
        "model.timestep_encoder.timestep_embedder.linear_2.bias",
        "dit.time.2.bias",
    )

    dit_cfg = config["diffusion_model_cfg"]
    for layer in range(int(dit_cfg["num_layers"])):
        source = f"model.transformer_blocks.{layer}"
        destination = f"dit.blk.{layer}"
        for suffix, mapped in (
            ("norm1.linear.weight", "attn_norm.weight"),
            ("norm1.linear.bias", "attn_norm.bias"),
            ("attn1.to_q.weight", "attn_q.weight"),
            ("attn1.to_q.bias", "attn_q.bias"),
            ("attn1.to_k.weight", "attn_k.weight"),
            ("attn1.to_k.bias", "attn_k.bias"),
            ("attn1.to_v.weight", "attn_v.weight"),
            ("attn1.to_v.bias", "attn_v.bias"),
            ("attn1.to_out.0.weight", "attn_out.weight"),
            ("attn1.to_out.0.bias", "attn_out.bias"),
            ("ff.net.0.proj.weight", "ffn_up.weight"),
            ("ff.net.0.proj.bias", "ffn_up.bias"),
            ("ff.net.2.weight", "ffn_down.weight"),
            ("ff.net.2.bias", "ffn_down.bias"),
        ):
            add(f"{source}.{suffix}", f"{destination}.{mapped}")

    for suffix, mapped in (
        ("model.proj_out_1.weight", "dit.out_norm.weight"),
        ("model.proj_out_1.bias", "dit.out_norm.bias"),
        ("model.proj_out_2.weight", "dit.out_proj.weight"),
        ("model.proj_out_2.bias", "dit.out_proj.bias"),
    ):
        add(suffix, mapped)
    return rows


def _validate_tensor_map(
    checkpoint: ShardedCheckpoint,
    rows: list[tuple[str, str, bool]],
) -> None:
    sources = [source for source, _, _ in rows]
    destinations = [destination for _, destination, _ in rows]
    missing = sorted(set(sources) - checkpoint.keys)
    if missing:
        raise SystemExit(f"tensor map references {len(missing)} missing tensors; first: {missing[0]}")
    duplicate_sources = sorted({name for name in sources if sources.count(name) > 1})
    duplicate_destinations = sorted({name for name in destinations if destinations.count(name) > 1})
    if duplicate_sources or duplicate_destinations:
        raise SystemExit(
            "duplicate tensor mapping: "
            f"sources={duplicate_sources[:1]} destinations={duplicate_destinations[:1]}"
        )
    action_keys = {name for name in checkpoint.keys if name.startswith("action_head.")}
    unmapped = sorted(action_keys - set(sources))
    unexpected = sorted(set(sources) - action_keys)
    if unmapped or unexpected:
        detail = unmapped[0] if unmapped else unexpected[0]
        raise SystemExit(
            f"action-head tensor map is not exhaustive (unmapped={len(unmapped)}, "
            f"unexpected={len(unexpected)}); first: {detail}"
        )
    too_long = [name for name in destinations if len(name.encode("utf-8")) >= 64]
    if too_long:
        raise SystemExit(f"GGUF tensor name must be shorter than 64 bytes: {too_long[0]}")


def _add_tensor(writer, name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().contiguous().cpu()
    if tensor.dtype == torch.bfloat16:
        writer.add_tensor(
            name,
            tensor.view(torch.uint16).numpy(),
            raw_shape=list(tensor.shape),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )
    elif tensor.dtype == torch.float32:
        writer.add_tensor(
            name,
            tensor.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
    else:
        raise SystemExit(f"unsupported source dtype {tensor.dtype} for {name}")


def _processor_kwargs(checkpoint_dir: Path) -> dict[str, Any]:
    processor = _load_json(checkpoint_dir / "processor_config.json")
    kwargs = processor.get("processor_kwargs")
    if not isinstance(kwargs, dict):
        raise SystemExit("processor_config.json is missing processor_kwargs")
    return kwargs


def _modality_layout(
    processor: dict[str, Any],
    embodiment: str,
    section: str,
) -> tuple[list[str], int]:
    try:
        modality = processor["modality_configs"][embodiment][section]
        keys = [str(key) for key in modality["modality_keys"]]
        horizon = len(modality["delta_indices"])
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"missing {embodiment}.{section} modality configuration") from exc
    return keys, horizon


def _flatten_stat(
    statistics: dict[str, Any],
    embodiment: str,
    section: str,
    keys: list[str],
    stat: str,
) -> np.ndarray:
    values: list[float] = []
    try:
        group = statistics[embodiment][section]
        for key in keys:
            values.extend(float(value) for value in group[key][stat])
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"missing {embodiment}.{section}.{stat} statistics") from exc
    return np.asarray(values, dtype=np.float32)


def _add_metadata(
    writer,
    config: dict[str, Any],
    processor: dict[str, Any],
    embodiment: str,
    embodiment_id: int,
    state_dim: int,
    action_dim: int,
    action_horizon: int,
) -> None:
    dit_cfg = config["diffusion_model_cfg"]
    vl_cfg = config.get("vl_self_attention_cfg") or {}
    input_embedding_dim = int(
        config.get(
            "input_embedding_dim",
            int(dit_cfg["num_attention_heads"]) * int(dit_cfg["attention_head_dim"]),
        )
    )
    writer.add_string(KV("architecture"), ARCH)
    writer.add_string(KV("variant"), "n1.7")
    writer.add_string(KV("embodiment"), embodiment)
    writer.add_string(KV("backbone_architecture"), "qwen3vl")
    writer.add_string(KV("backbone_model"), str(config["model_name"]))
    writer.add_uint32(KV("embodiment_id"), embodiment_id)
    writer.add_uint32(KV("backbone_select_layer"), int(config["select_layer"]))
    writer.add_uint32(KV("backbone_embedding_dim"), int(config["backbone_embedding_dim"]))
    writer.add_uint32(KV("hidden_size"), int(config["hidden_size"]))
    writer.add_uint32(KV("input_embedding_dim"), input_embedding_dim)
    writer.add_uint32(KV("max_state_dim"), int(config["max_state_dim"]))
    writer.add_uint32(KV("max_action_dim"), int(config["max_action_dim"]))
    writer.add_uint32(KV("real_state_dim"), state_dim)
    writer.add_uint32(KV("real_action_dim"), action_dim)
    writer.add_uint32(KV("action_horizon"), int(config["action_horizon"]))
    writer.add_uint32(KV("real_action_horizon"), action_horizon)
    writer.add_uint32(KV("max_seq_len"), int(config["max_seq_len"]))
    writer.add_uint32(KV("vl_n_layers"), int(vl_cfg.get("num_layers", 0)))
    writer.add_uint32(KV("vl_n_heads"), int(vl_cfg.get("num_attention_heads", 0)))
    writer.add_uint32(KV("vl_head_dim"), int(vl_cfg.get("attention_head_dim", 0)))
    writer.add_uint32(KV("dit_n_layers"), int(dit_cfg["num_layers"]))
    writer.add_uint32(KV("dit_n_heads"), int(dit_cfg["num_attention_heads"]))
    writer.add_uint32(KV("dit_head_dim"), int(dit_cfg["attention_head_dim"]))
    writer.add_uint32(KV("num_inference_timesteps"), int(config["num_inference_timesteps"]))
    writer.add_uint32(KV("num_timestep_buckets"), int(config["num_timestep_buckets"]))
    writer.add_uint32(
        KV("attend_text_every_n_blocks"),
        int(config.get("attend_text_every_n_blocks", 2)),
    )
    writer.add_float32(KV("layer_norm_eps"), 1e-5)
    writer.add_float32(KV("output_norm_eps"), 1e-6)
    writer.add_bool(KV("add_pos_embed"), bool(config["add_pos_embed"]))
    writer.add_bool(KV("use_vlln"), bool(config["use_vlln"]))
    writer.add_bool(KV("use_alternate_vl_dit"), bool(config["use_alternate_vl_dit"]))
    writer.add_bool(KV("use_percentiles"), bool(processor.get("use_percentiles", True)))
    writer.add_bool(KV("clip_outliers"), bool(processor.get("clip_outliers", True)))
    writer.add_bool(KV("use_relative_action"), bool(processor.get("use_relative_action", False)))
    writer.add_uint32(KV("image_target_size"), int(processor.get("image_target_size", [256])[0]))
    writer.add_uint32(KV("image_crop_size"), int(processor.get("image_crop_size", [230])[0]))


def convert(
    checkpoint_dir: Path,
    output: Path,
    embodiment: str,
    dry_run: bool,
) -> None:
    config = _load_json(checkpoint_dir / "config.json")
    processor = _processor_kwargs(checkpoint_dir)
    embodiment_ids = _load_json(checkpoint_dir / "embodiment_id.json")
    statistics = _load_json(checkpoint_dir / "statistics.json")
    if config.get("model_type") != "Gr00tN1d7":
        raise SystemExit(
            f"unsupported model_type={config.get('model_type')!r}; expected 'Gr00tN1d7'"
        )
    if embodiment not in embodiment_ids:
        raise SystemExit(f"embodiment {embodiment!r} missing from embodiment_id.json")

    state_keys, state_horizon = _modality_layout(processor, embodiment, "state")
    action_keys, action_horizon = _modality_layout(processor, embodiment, "action")
    if state_horizon != int(config.get("state_history_length", 1)):
        raise SystemExit(
            f"state history mismatch: processor={state_horizon}, "
            f"model={config.get('state_history_length', 1)}"
        )
    state_q01 = _flatten_stat(statistics, embodiment, "state", state_keys, "q01")
    state_q99 = _flatten_stat(statistics, embodiment, "state", state_keys, "q99")
    action_q01 = _flatten_stat(statistics, embodiment, "action", action_keys, "q01")
    action_q99 = _flatten_stat(statistics, embodiment, "action", action_keys, "q99")
    embodiment_id = int(embodiment_ids[embodiment])

    checkpoint = ShardedCheckpoint(checkpoint_dir)
    rows = _tensor_rows(config)
    _validate_tensor_map(checkpoint, rows)

    tensor_bytes = 0
    for source, _, category in rows:
        shape = checkpoint.get_shape(source)
        if category:
            if shape[0] <= embodiment_id:
                raise SystemExit(
                    f"{source} has only {shape[0]} categories; requested id {embodiment_id}"
                )
            shape = shape[1:]
        tensor_bytes += prod(shape) * 2
    print(
        f"validated {len(rows)} action-head tensors for {embodiment} (id={embodiment_id}); "
        f"estimated BF16 size={tensor_bytes / (1024 ** 3):.2f} GiB"
    )
    print(
        f"state={state_q01.size} dims, action={action_q01.size} dims, "
        f"effective action horizon={action_horizon}"
    )
    if dry_run:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output), arch=ARCH)
    _add_metadata(
        writer,
        config,
        processor,
        embodiment,
        embodiment_id,
        int(state_q01.size),
        int(action_q01.size),
        action_horizon,
    )
    for name, value in (
        ("norm.state.q01", state_q01),
        ("norm.state.q99", state_q99),
        ("norm.action.q01", action_q01),
        ("norm.action.q99", action_q99),
    ):
        writer.add_tensor(name, value, raw_dtype=gguf.GGMLQuantizationType.F32)

    for source, destination, category in rows:
        tensor = checkpoint.get_tensor(source)
        if category:
            tensor = tensor[embodiment_id]
            if source.endswith(".W"):
                tensor = tensor.transpose(0, 1)
        print(f"  {source} -> {destination} {list(tensor.shape)}")
        _add_tensor(writer, destination, tensor)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {output} ({output.stat().st_size / (1024 ** 3):.2f} GiB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Local GR00T-N1.7 checkpoint subdirectory (for example libero_object).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GGUF (default: <ckpt>/groot-n1-action-head-bf16.gguf).",
    )
    parser.add_argument("--embodiment", default=DEFAULT_EMBODIMENT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = args.ckpt.expanduser().resolve()
    output = (
        args.out.expanduser().resolve()
        if args.out
        else checkpoint_dir / "groot-n1-action-head-bf16.gguf"
    )
    convert(checkpoint_dir, output, args.embodiment, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())