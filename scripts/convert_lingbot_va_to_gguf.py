#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Convert LingBot-VA checkpoints to GGUF.

This is currently a metadata-first skeleton. It can be used before the full
weights are downloaded to validate config parsing and GGUF metadata layout.
Full tensor streaming will be added after the safetensors shards are available.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if GGUF_PY.exists():
    sys.path.insert(0, str(GGUF_PY))

try:
    import gguf
except Exception as exc:  # pragma: no cover - user-facing startup error
    raise SystemExit(
        "failed to import gguf. Expected either an installed gguf package or "
        f"the vendored package at {GGUF_PY}. Original error: {exc}"
    )


ARCH = "lingbot_va"
KV = lambda name: f"{ARCH}.{name}"

DEFAULT_HF_SNAPSHOT = Path(
    "/home/xuling/.cache/huggingface/hub/"
    "models--robbyant--lingbot-va-posttrain-libero-long/"
    "snapshots/0e89d1e753019988aba484e8da2dc0810e264d9f"
)


def _bf16_to_u16(t: Any) -> Any:
    import torch

    if t.dtype != torch.bfloat16:
        print(f"  warn: casting non-BF16 tensor (dtype={t.dtype}) to BF16 for storage")
        t = t.to(torch.bfloat16)
    return t.contiguous().view(torch.uint16).cpu().numpy()


def _add_tensor(
    writer: "gguf.GGUFWriter",
    dst_name: str,
    t: Any,
    shape_records: dict[str, list[int]] | None = None,
) -> None:
    import torch

    original_shape = [int(x) for x in t.shape]
    if shape_records is not None:
        shape_records[dst_name] = original_shape

    if len(t.shape) > 4:
        t = t.reshape(int(t.shape[0]), -1)
        print(f"  note: flattening {dst_name} from {original_shape} to {list(t.shape)} for GGUF")

    if t.dtype == torch.float32:
        writer.add_tensor(
            dst_name,
            t.contiguous().cpu().numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
    elif t.dtype == torch.bfloat16:
        writer.add_tensor(
            dst_name,
            _bf16_to_u16(t),
            raw_shape=list(t.shape),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )
    else:
        raise NotImplementedError(f"unsupported dtype {t.dtype} for {dst_name}")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"missing required file or directory: {path}")
    return path


def _metadata_total_size(index_path: Path) -> int:
    if not index_path.exists():
        return 0
    blob = _load_json(index_path)
    return int(blob.get("metadata", {}).get("total_size", 0))


def _weight_map_count(index_path: Path) -> int:
    if not index_path.exists():
        return 0
    blob = _load_json(index_path)
    return len(blob.get("weight_map", {}))


def _all_index_shards_present(module_dir: Path, index_name: str) -> bool:
    index_path = module_dir / index_name
    if not index_path.exists():
        return False
    blob = _load_json(index_path)
    shards = sorted(set(str(v) for v in blob.get("weight_map", {}).values()))
    return bool(shards) and all((module_dir / shard).exists() for shard in shards)


def _transformer_tensor_map(num_layers: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("patch_embedding_mlp.weight", "wvm.patch_embd_mlp.weight"),
        ("patch_embedding_mlp.bias", "wvm.patch_embd_mlp.bias"),
        ("patch_embedding.weight", "wvm.patch_embd_legacy.weight"),
        ("patch_embedding.bias", "wvm.patch_embd_legacy.bias"),
        ("action_embedder.weight", "wvm.action_embd.weight"),
        ("action_embedder.bias", "wvm.action_embd.bias"),
        ("proj_out.weight", "wvm.output_proj.weight"),
        ("proj_out.bias", "wvm.output_proj.bias"),
        ("action_proj_out.weight", "wvm.action_out.weight"),
        ("action_proj_out.bias", "wvm.action_out.bias"),
        ("scale_shift_table", "wvm.output_scale_shift"),
    ]
    for src_prefix, dst_prefix in (
        ("condition_embedder", "wvm.cond"),
        ("condition_embedder_action", "wvm.action_cond"),
    ):
        rows.extend([
            (f"{src_prefix}.text_embedder.linear_1.weight", f"{dst_prefix}.text_l1.weight"),
            (f"{src_prefix}.text_embedder.linear_1.bias", f"{dst_prefix}.text_l1.bias"),
            (f"{src_prefix}.text_embedder.linear_2.weight", f"{dst_prefix}.text_l2.weight"),
            (f"{src_prefix}.text_embedder.linear_2.bias", f"{dst_prefix}.text_l2.bias"),
            (f"{src_prefix}.time_embedder.linear_1.weight", f"{dst_prefix}.time_l1.weight"),
            (f"{src_prefix}.time_embedder.linear_1.bias", f"{dst_prefix}.time_l1.bias"),
            (f"{src_prefix}.time_embedder.linear_2.weight", f"{dst_prefix}.time_l2.weight"),
            (f"{src_prefix}.time_embedder.linear_2.bias", f"{dst_prefix}.time_l2.bias"),
            (f"{src_prefix}.time_proj.weight", f"{dst_prefix}.time_proj.weight"),
            (f"{src_prefix}.time_proj.bias", f"{dst_prefix}.time_proj.bias"),
        ])
    for i in range(num_layers):
        p = f"blocks.{i}"
        d = f"wvm.blk.{i}"
        rows.extend([
            (f"{p}.scale_shift_table", f"{d}.scale_shift"),
            (f"{p}.norm2.weight", f"{d}.cross_norm.weight"),
            (f"{p}.norm2.bias", f"{d}.cross_norm.bias"),
        ])
        for attn in ("attn1", "attn2"):
            dst_attn = "self_attn" if attn == "attn1" else "cross_attn"
            rows.extend([
                (f"{p}.{attn}.to_q.weight", f"{d}.{dst_attn}.q.weight"),
                (f"{p}.{attn}.to_q.bias", f"{d}.{dst_attn}.q.bias"),
                (f"{p}.{attn}.to_k.weight", f"{d}.{dst_attn}.k.weight"),
                (f"{p}.{attn}.to_k.bias", f"{d}.{dst_attn}.k.bias"),
                (f"{p}.{attn}.to_v.weight", f"{d}.{dst_attn}.v.weight"),
                (f"{p}.{attn}.to_v.bias", f"{d}.{dst_attn}.v.bias"),
                (f"{p}.{attn}.to_out.0.weight", f"{d}.{dst_attn}.o.weight"),
                (f"{p}.{attn}.to_out.0.bias", f"{d}.{dst_attn}.o.bias"),
                (f"{p}.{attn}.norm_q.weight", f"{d}.{dst_attn}.q_norm.weight"),
                (f"{p}.{attn}.norm_k.weight", f"{d}.{dst_attn}.k_norm.weight"),
            ])
        rows.extend([
            (f"{p}.ffn.net.0.proj.weight", f"{d}.ffn_up.weight"),
            (f"{p}.ffn.net.0.proj.bias", f"{d}.ffn_up.bias"),
            (f"{p}.ffn.net.2.weight", f"{d}.ffn_down.weight"),
            (f"{p}.ffn.net.2.bias", f"{d}.ffn_down.bias"),
        ])
    return rows


def _validate_transformer_map(root: Path) -> tuple[list[tuple[str, str]], dict[str, str]]:
    transformer_dir = _require(root / "transformer")
    cfg = _load_json(_require(transformer_dir / "config.json"))
    index = _load_json(_require(transformer_dir / "diffusion_pytorch_model.safetensors.index.json"))
    weight_map = {str(k): str(v) for k, v in index.get("weight_map", {}).items()}
    rows = _transformer_tensor_map(int(cfg["num_layers"]))
    srcs = [src for src, _ in rows]
    dsts = [dst for _, dst in rows]
    missing = sorted(set(srcs) - set(weight_map))
    extra = sorted(set(weight_map) - set(srcs))
    dup_dst = sorted({name for name in dsts if dsts.count(name) > 1})
    if missing or extra or dup_dst:
        msg = [
            "transformer tensor map does not match index:",
            f"  missing mapped source tensors: {len(missing)}",
            f"  unmapped index tensors: {len(extra)}",
            f"  duplicate destination names: {len(dup_dst)}",
        ]
        for label, values in (("missing", missing), ("extra", extra), ("dup_dst", dup_dst)):
            for value in values[:32]:
                msg.append(f"  {label}: {value}")
        raise SystemExit("\n".join(msg))
    return rows, weight_map


def _text_encoder_tensor_map(num_layers: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("shared.weight", "text.token_embd.weight"),
        ("encoder.final_layer_norm.weight", "text.final_norm.weight"),
    ]
    for i in range(num_layers):
        p = f"encoder.block.{i}"
        d = f"text.blk.{i}"
        rows.extend([
            (f"{p}.layer.0.layer_norm.weight", f"{d}.attn_norm.weight"),
            (f"{p}.layer.0.SelfAttention.q.weight", f"{d}.attn.q.weight"),
            (f"{p}.layer.0.SelfAttention.k.weight", f"{d}.attn.k.weight"),
            (f"{p}.layer.0.SelfAttention.v.weight", f"{d}.attn.v.weight"),
            (f"{p}.layer.0.SelfAttention.o.weight", f"{d}.attn.o.weight"),
            (f"{p}.layer.0.SelfAttention.relative_attention_bias.weight", f"{d}.attn.rel_bias.weight"),
            (f"{p}.layer.1.layer_norm.weight", f"{d}.ffn_norm.weight"),
            (f"{p}.layer.1.DenseReluDense.wi_0.weight", f"{d}.ffn.wi_0.weight"),
            (f"{p}.layer.1.DenseReluDense.wi_1.weight", f"{d}.ffn.wi_1.weight"),
            (f"{p}.layer.1.DenseReluDense.wo.weight", f"{d}.ffn.wo.weight"),
        ])
    return rows


def _validate_text_encoder_map(root: Path) -> tuple[list[tuple[str, str]], dict[str, str], dict[str, Any]]:
    text_dir = _require(root / "text_encoder")
    cfg = _load_json(_require(text_dir / "config.json"))
    index = _load_json(_require(text_dir / "model.safetensors.index.json"))
    weight_map = {str(k): str(v) for k, v in index.get("weight_map", {}).items()}
    rows = _text_encoder_tensor_map(int(cfg["num_layers"]))
    srcs = [src for src, _ in rows]
    dsts = [dst for _, dst in rows]
    missing = sorted(set(srcs) - set(weight_map))
    extra = sorted(set(weight_map) - set(srcs))
    dup_dst = sorted({name for name in dsts if dsts.count(name) > 1})
    if missing or extra or dup_dst:
        msg = [
            "text_encoder tensor map does not match index:",
            f"  missing mapped source tensors: {len(missing)}",
            f"  unmapped index tensors: {len(extra)}",
            f"  duplicate destination names: {len(dup_dst)}",
        ]
        for label, values in (("missing", missing), ("extra", extra), ("dup_dst", dup_dst)):
            for value in values[:32]:
                msg.append(f"  {label}: {value}")
        raise SystemExit("\n".join(msg))
    return rows, weight_map, cfg


def _vae_tensor_map(root: Path, include_decoder: bool) -> list[tuple[str, str]]:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise SystemExit(
            "safetensors is required to inspect VAE tensors. "
            "Run with a Python environment that has safetensors installed. "
            f"Original error: {exc}"
        )

    vae_path = _require(root / "vae" / "diffusion_pytorch_model.safetensors")
    with safe_open(vae_path, framework="pt", device="cpu") as sf:
        keys = sorted(sf.keys())

    if include_decoder:
        selected = keys
    else:
        selected = [
            k for k in keys
            if k.startswith("encoder.") or k.startswith("quant_conv.")
        ]
    if not selected:
        raise SystemExit("VAE tensor map is empty")
    return [(src, f"vae.{src}") for src in selected]


def _stream_transformer_tensors(writer: "gguf.GGUFWriter", root: Path) -> None:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise SystemExit(
            "safetensors is required for full tensor conversion. "
            "Run with a Python environment that has safetensors installed. "
            f"Original error: {exc}"
        )

    rows, weight_map = _validate_transformer_map(root)
    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for src, dst in rows:
        by_shard[weight_map[src]].append((src, dst))

    transformer_dir = root / "transformer"
    print(f"streaming transformer tensors: {len(rows)} tensors from {len(by_shard)} shards")
    for shard, items in sorted(by_shard.items()):
        path = transformer_dir / shard
        print(f"  shard {shard}: {len(items)} tensors")
        with safe_open(path, framework="pt", device="cpu") as sf:
            for src, dst in items:
                _add_tensor(writer, dst, sf.get_tensor(src))


def _stream_text_encoder_tensors(writer: "gguf.GGUFWriter", root: Path) -> None:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise SystemExit(
            "safetensors is required for text_encoder tensor conversion. "
            "Run with a Python environment that has safetensors installed. "
            f"Original error: {exc}"
        )

    rows, weight_map, _ = _validate_text_encoder_map(root)
    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for src, dst in rows:
        by_shard[weight_map[src]].append((src, dst))

    text_dir = root / "text_encoder"
    shape_records: dict[str, list[int]] = {}
    print(f"streaming text_encoder tensors: {len(rows)} tensors from {len(by_shard)} shards")
    for shard, items in sorted(by_shard.items()):
        path = text_dir / shard
        print(f"  shard {shard}: {len(items)} tensors")
        with safe_open(path, framework="pt", device="cpu") as sf:
            for src, dst in items:
                _add_tensor(writer, dst, sf.get_tensor(src), shape_records)

    writer.add_array(KV("text_encoder.tensor_names"), [dst for _, dst in rows])
    writer.add_array(
        KV("text_encoder.tensor_shapes"),
        [
            name + ":" + ",".join(str(x) for x in shape_records[name])
            for _, name in rows
        ],
    )
    writer.add_uint32(KV("text_encoder.written_tensor_count"), len(rows))
    writer.add_string(KV("text_encoder.written_scope"), "encoder")


def _stream_vae_tensors(
    writer: "gguf.GGUFWriter",
    root: Path,
    include_decoder: bool,
) -> None:
    try:
        from safetensors import safe_open
    except Exception as exc:
        raise SystemExit(
            "safetensors is required for VAE tensor conversion. "
            "Run with a Python environment that has safetensors installed. "
            f"Original error: {exc}"
        )

    rows = _vae_tensor_map(root, include_decoder)
    shape_records: dict[str, list[int]] = {}
    vae_path = root / "vae" / "diffusion_pytorch_model.safetensors"
    print(f"streaming VAE tensors: {len(rows)} tensors from {vae_path.name}")
    with safe_open(vae_path, framework="pt", device="cpu") as sf:
        for src, dst in rows:
            _add_tensor(writer, dst, sf.get_tensor(src), shape_records)

    writer.add_array(KV("vae.tensor_names"), [dst for _, dst in rows])
    writer.add_array(
        KV("vae.tensor_shapes"),
        [
            name + ":" + ",".join(str(x) for x in shape_records[name])
            for _, name in rows
        ],
    )
    writer.add_uint32(KV("vae.written_tensor_count"), len(rows))
    writer.add_string(
        KV("vae.written_scope"),
        "full" if include_decoder else "encoder_quant_conv",
    )


def _add_common_metadata(writer: "gguf.GGUFWriter", root: Path) -> dict[str, Any]:
    text_dir = _require(root / "text_encoder")
    transformer_dir = _require(root / "transformer")
    vae_dir = _require(root / "vae")

    text_cfg = _load_json(_require(text_dir / "config.json"))
    transformer_cfg = _load_json(_require(transformer_dir / "config.json"))
    vae_cfg = _load_json(_require(vae_dir / "config.json"))

    text_index = text_dir / "model.safetensors.index.json"
    transformer_index = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"

    writer.add_string(KV("architecture"), ARCH)
    writer.add_string(KV("format_stage"), "metadata_skeleton")

    writer.add_uint32(KV("text_encoder.layers"), int(text_cfg["num_layers"]))
    writer.add_uint32(KV("text_encoder.d_model"), int(text_cfg["d_model"]))
    writer.add_uint32(KV("text_encoder.d_ff"), int(text_cfg["d_ff"]))
    writer.add_uint32(KV("text_encoder.d_kv"), int(text_cfg["d_kv"]))
    writer.add_uint32(KV("text_encoder.heads"), int(text_cfg["num_heads"]))
    writer.add_uint32(KV("text_encoder.vocab_size"), int(text_cfg["vocab_size"]))
    writer.add_uint32(KV("text_encoder.relative_attention_num_buckets"), int(text_cfg["relative_attention_num_buckets"]))
    writer.add_uint32(KV("text_encoder.relative_attention_max_distance"), int(text_cfg["relative_attention_max_distance"]))
    writer.add_float32(KV("text_encoder.layer_norm_epsilon"), float(text_cfg["layer_norm_epsilon"]))
    writer.add_string(KV("text_encoder.feed_forward_proj"), str(text_cfg.get("feed_forward_proj", "")))
    writer.add_string(KV("text_encoder.dense_act_fn"), str(text_cfg.get("dense_act_fn", "")))
    writer.add_uint32(KV("text_encoder.tensor_count"), _weight_map_count(text_index))
    writer.add_uint64(KV("text_encoder.total_size"), _metadata_total_size(text_index))

    writer.add_uint32(KV("transformer.layers"), int(transformer_cfg["num_layers"]))
    writer.add_uint32(KV("transformer.heads"), int(transformer_cfg["num_attention_heads"]))
    writer.add_uint32(KV("transformer.head_dim"), int(transformer_cfg["attention_head_dim"]))
    writer.add_uint32(KV("transformer.ffn_dim"), int(transformer_cfg["ffn_dim"]))
    writer.add_uint32(KV("transformer.in_channels"), int(transformer_cfg["in_channels"]))
    writer.add_uint32(KV("transformer.out_channels"), int(transformer_cfg["out_channels"]))
    writer.add_uint32(KV("transformer.text_dim"), int(transformer_cfg["text_dim"]))
    writer.add_uint32(KV("transformer.action_dim"), int(transformer_cfg.get("action_dim", 30)))
    patch_size = transformer_cfg.get("patch_size", [1, 2, 2])
    writer.add_uint32(KV("transformer.patch_t"), int(patch_size[0]))
    writer.add_uint32(KV("transformer.patch_h"), int(patch_size[1]))
    writer.add_uint32(KV("transformer.patch_w"), int(patch_size[2]))
    writer.add_string(KV("transformer.attn_mode_config"), str(transformer_cfg.get("attn_mode", "")))
    writer.add_uint32(KV("transformer.tensor_count"), _weight_map_count(transformer_index))
    writer.add_uint64(KV("transformer.total_size"), _metadata_total_size(transformer_index))

    writer.add_uint32(KV("vae.in_channels"), int(vae_cfg["in_channels"]))
    writer.add_uint32(KV("vae.out_channels"), int(vae_cfg["out_channels"]))
    writer.add_uint32(KV("vae.z_dim"), int(vae_cfg["z_dim"]))
    writer.add_uint32(KV("vae.base_dim"), int(vae_cfg["base_dim"]))
    writer.add_uint32(KV("vae.decoder_base_dim"), int(vae_cfg["decoder_base_dim"]))
    writer.add_uint32(KV("vae.patch_size"), int(vae_cfg.get("patch_size", 1)))
    writer.add_uint32(KV("vae.scale_factor_spatial"), int(vae_cfg["scale_factor_spatial"]))
    writer.add_uint32(KV("vae.scale_factor_temporal"), int(vae_cfg["scale_factor_temporal"]))
    writer.add_array(KV("vae.dim_mult"), [int(x) for x in vae_cfg["dim_mult"]])
    writer.add_array(KV("vae.temperal_downsample"), [bool(x) for x in vae_cfg["temperal_downsample"]])
    writer.add_array(KV("vae.latents_mean"), [float(x) for x in vae_cfg["latents_mean"]])
    writer.add_array(KV("vae.latents_std"), [float(x) for x in vae_cfg["latents_std"]])
    writer.add_string(KV("vae.class_name"), str(vae_cfg.get("_class_name", "")))

    return {
        "text_shards_present": _all_index_shards_present(text_dir, "model.safetensors.index.json"),
        "transformer_shards_present": _all_index_shards_present(
            transformer_dir,
            "diffusion_pytorch_model.safetensors.index.json",
        ),
        "vae_safetensors_present": any(vae_dir.glob("*.safetensors")),
        "tokenizer_present": (root / "tokenizer").exists(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_HF_SNAPSHOT,
        help="LingBot-VA checkpoint root containing text_encoder/, transformer/, vae/, tokenizer/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output GGUF path. Defaults to <ckpt>/lingbot_va.metadata.gguf for --metadata-only.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write only GGUF metadata. Useful while weights are still downloading.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and tensor maps without writing a GGUF file.",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=["text_encoder", "transformer", "vae_encoder", "vae"],
        default=[],
        help=(
            "'text_encoder' writes UMT5 encoder tensors; "
            "Modules to write. 'vae_encoder' writes encoder.* plus quant_conv.*; "
            "'vae' writes the full AutoencoderKLWan tensor set."
        ),
    )
    args = parser.parse_args()

    root = args.ckpt.expanduser().resolve()
    out = (args.out or (root / "lingbot_va.metadata.gguf")).expanduser().resolve()

    if args.dry_run:
        status = {
            "text_shards_present": _all_index_shards_present(root / "text_encoder", "model.safetensors.index.json"),
            "transformer_shards_present": _all_index_shards_present(
                root / "transformer",
                "diffusion_pytorch_model.safetensors.index.json",
            ),
            "vae_safetensors_present": any((root / "vae").glob("*.safetensors")),
            "tokenizer_present": (root / "tokenizer").exists(),
        }
        rows, _ = _validate_transformer_map(root)
        print(f"dry-run ok: transformer tensor map covers {len(rows)} tensors")
        if "text_encoder" in args.modules:
            text_rows, _, text_cfg = _validate_text_encoder_map(root)
            print(
                "dry-run ok: text_encoder tensor map covers "
                f"{len(text_rows)} tensors "
                f"(layers={text_cfg['num_layers']} d_model={text_cfg['d_model']})"
            )
        if "vae_encoder" in args.modules or "vae" in args.modules:
            vae_rows = _vae_tensor_map(root, include_decoder="vae" in args.modules)
            print(f"dry-run ok: VAE tensor map covers {len(vae_rows)} tensors")
        for name, present in status.items():
            print(f"{name}: {'yes' if present else 'no'}")
        return 0

    writer = gguf.GGUFWriter(str(out), arch=ARCH)
    status = _add_common_metadata(writer, root)

    if not args.metadata_only:
        missing = [name for name, present in status.items() if not present]
        if missing:
            raise SystemExit(
                "full tensor conversion is not implemented and the checkpoint "
                f"is incomplete for: {', '.join(missing)}. "
                "Use --metadata-only for the current planning stage."
            )
        if not args.modules:
            raise SystemExit(
                "no modules selected. Use --modules transformer, or "
                "--metadata-only for metadata skeleton generation."
            )
        if "transformer" in args.modules:
            _stream_transformer_tensors(writer, root)
        if "text_encoder" in args.modules:
            _stream_text_encoder_tensors(writer, root)
        if "vae_encoder" in args.modules:
            _stream_vae_tensors(writer, root, include_decoder=False)
        if "vae" in args.modules:
            _stream_vae_tensors(writer, root, include_decoder=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.metadata_only:
        print(f"writing metadata-only GGUF: {out}")
    else:
        print(f"writing GGUF: {out}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    if not args.metadata_only:
        writer.write_tensors_to_file()
    writer.close()
    for name, present in status.items():
        print(f"{name}: {'yes' if present else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
