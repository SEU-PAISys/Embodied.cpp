#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf

PFX_CANDIDATES = [
    "model.paligemma_with_expert.paligemma.model",
    "paligemma_with_expert.paligemma.model",
]


def _bf16_to_u16_bytes(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.view(torch.uint16).contiguous().cpu().numpy()


def _f32_np(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.float32).contiguous().cpu().numpy()


def _pick_prefix(keys: set[str], candidates: list[str], probe_suffix: str) -> str:
    for prefix in candidates:
        if f"{prefix}.{probe_suffix}" in keys:
            return prefix
    raise SystemExit(f"cannot resolve checkpoint prefix for {probe_suffix!r}")


def _add_tensor(writer: gguf.GGUFWriter, name: str, t: torch.Tensor, *,
                raw_shape: list[int] | None = None) -> None:
    if t.dtype == torch.bfloat16:
        writer.add_tensor(
            name,
            _bf16_to_u16_bytes(t),
            raw_shape=raw_shape or list(t.shape),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )
    else:
        writer.add_tensor(
            name,
            _f32_np(t),
            raw_shape=raw_shape,
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export the PaliGemma-224 vision tower + projector from a pi0.5 checkpoint as a llama.cpp mmproj GGUF."
    )
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="pi0.5 LeRobot checkpoint directory containing model.safetensors")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output mmproj GGUF path")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    sf_path = ckpt / "model.safetensors"
    if not sf_path.is_file():
        raise SystemExit(f"missing {sf_path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.out), arch="clip")
    writer.add_string("general.name", "_Workdir_Pi05_Libero_Finetuned_V044")
    writer.add_string("general.type", "mmproj")
    writer.add_string("general.finetune", "_workdir_pi05_libero_finetuned_v044")
    writer.add_string("general.size_label", "415M")
    writer.add_uint32("general.quantization_version", 2)
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_string("clip.projector_type", "paligemma")
    writer.add_bool("clip.use_gelu", True)
    writer.add_uint32("clip.vision.image_size", 224)
    writer.add_uint32("clip.vision.patch_size", 14)
    writer.add_uint32("clip.vision.embedding_length", 1152)
    writer.add_uint32("clip.vision.feed_forward_length", 4304)
    writer.add_uint32("clip.vision.block_count", 27)
    writer.add_uint32("clip.vision.attention.head_count", 16)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    writer.add_uint32("clip.vision.projection_dim", 2048)
    writer.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    writer.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)

    with safe_open(str(sf_path), framework="pt") as sf:
        keys = set(sf.keys())
        pfx = _pick_prefix(keys, PFX_CANDIDATES,
                           "vision_tower.vision_model.embeddings.patch_embedding.weight")
        ve = f"{pfx}.vision_tower.vision_model"
        mm = f"{pfx}.multi_modal_projector.linear"

        def get(name: str) -> torch.Tensor:
            return sf.get_tensor(name)

        # The local PaliGemma mtmd projector path applies a legacy
        # 1/sqrt(hidden_size) scale after the linear projection. Transformers'
        # PaliGemma projector is just Linear(weight, bias), so fold the inverse
        # scale into the exported tensors without changing third_party code.
        proj_scale = math.sqrt(2048.0)
        _add_tensor(writer, "mm.input_projection.bias",   get(f"{mm}.bias").to(torch.float32) * proj_scale)
        _add_tensor(writer, "mm.input_projection.weight", get(f"{mm}.weight") * proj_scale)

        _add_tensor(writer, "v.patch_embd.bias",      get(f"{ve}.embeddings.patch_embedding.bias"))
        _add_tensor(writer, "v.patch_embd.weight",    get(f"{ve}.embeddings.patch_embedding.weight"))
        _add_tensor(writer, "v.position_embd.weight", get(f"{ve}.embeddings.position_embedding.weight"))

        for i in range(27):
            src = f"{ve}.encoder.layers.{i}"
            dst = f"v.blk.{i}"
            _add_tensor(writer, f"{dst}.ln1.bias",         get(f"{src}.layer_norm1.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.ln1.weight",       get(f"{src}.layer_norm1.weight").to(torch.float32))
            _add_tensor(writer, f"{dst}.ln2.bias",         get(f"{src}.layer_norm2.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.ln2.weight",       get(f"{src}.layer_norm2.weight").to(torch.float32))
            _add_tensor(writer, f"{dst}.ffn_up.bias",      get(f"{src}.mlp.fc1.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.ffn_up.weight",    get(f"{src}.mlp.fc1.weight"))
            _add_tensor(writer, f"{dst}.ffn_down.bias",    get(f"{src}.mlp.fc2.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.ffn_down.weight",  get(f"{src}.mlp.fc2.weight"))
            _add_tensor(writer, f"{dst}.attn_k.bias",      get(f"{src}.self_attn.k_proj.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.attn_k.weight",    get(f"{src}.self_attn.k_proj.weight"))
            _add_tensor(writer, f"{dst}.attn_out.bias",    get(f"{src}.self_attn.out_proj.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.attn_out.weight",  get(f"{src}.self_attn.out_proj.weight"))
            _add_tensor(writer, f"{dst}.attn_q.bias",      get(f"{src}.self_attn.q_proj.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.attn_q.weight",    get(f"{src}.self_attn.q_proj.weight"))
            _add_tensor(writer, f"{dst}.attn_v.bias",      get(f"{src}.self_attn.v_proj.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.attn_v.weight",    get(f"{src}.self_attn.v_proj.weight"))

        _add_tensor(writer, "v.post_ln.bias",   get(f"{ve}.post_layernorm.bias").to(torch.float32))
        _add_tensor(writer, "v.post_ln.weight", get(f"{ve}.post_layernorm.weight").to(torch.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} ({args.out.stat().st_size / (1024 * 1024):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
