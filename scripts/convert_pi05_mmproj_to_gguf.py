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
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if GGUF_PY.exists():
    sys.path.insert(0, str(GGUF_PY))

try:
    import gguf
except Exception as exc:
    raise SystemExit(f"failed to import gguf from {GGUF_PY}: {exc}")

from gguf_quantize import TensorQuantizer, add_outtype_args

PFX_CANDIDATES = [
    "model.paligemma_with_expert.paligemma.model",
    "paligemma_with_expert.paligemma.model",
]

MMPROJ_MATMUL_RE = re.compile(
    r"^(?:mm\.input_projection|v\.blk\.\d+\."
    r"(?:ffn_(?:up|down)|attn_[qkv]|attn_out))\.weight$"
)


def _pick_prefix(keys: set[str], candidates: list[str], probe_suffix: str) -> str:
    for prefix in candidates:
        if f"{prefix}.{probe_suffix}" in keys:
            return prefix
    raise SystemExit(f"cannot resolve checkpoint prefix for {probe_suffix!r}")


def _add_tensor(writer: gguf.GGUFWriter, quantizer: TensorQuantizer,
                name: str, tensor: torch.Tensor) -> None:
    quantizer.add_tensor(
        writer,
        name,
        tensor,
        quantize=MMPROJ_MATMUL_RE.fullmatch(name) is not None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export the PaliGemma-224 vision tower + projector from a pi0.5 checkpoint as a llama.cpp mmproj GGUF."
    )
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="pi0.5 LeRobot checkpoint directory containing model.safetensors")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output mmproj GGUF path (default: <ckpt>/pi05-mmproj[-TYPE].gguf)")
    add_outtype_args(ap)
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    default_name = "pi05-mmproj.gguf" if args.outtype == "bf16" else f"pi05-mmproj-{args.outtype}.gguf"
    out = (args.out or ckpt / default_name).resolve()
    quantizer = TensorQuantizer(args.outtype, args.ggml_lib)
    sf_path = ckpt / "model.safetensors"
    if not sf_path.is_file():
        raise SystemExit(f"missing {sf_path}")

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out), arch="clip")
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
    writer.add_file_type(quantizer.file_type)
    writer.add_string("pi05.mmproj.quantization", args.outtype.upper())

    with safe_open(str(sf_path), framework="pt") as sf:
        keys = set(sf.keys())
        pfx = _pick_prefix(keys, PFX_CANDIDATES,
                           "vision_tower.vision_model.embeddings.patch_embedding.weight")
        ve = f"{pfx}.vision_tower.vision_model"
        mm = f"{pfx}.multi_modal_projector.linear"

        def get(name: str) -> torch.Tensor:
            return sf.get_tensor(name)

        def add(name: str, tensor: torch.Tensor) -> None:
            _add_tensor(writer, quantizer, name, tensor)

        # The local PaliGemma mtmd projector path applies a legacy
        # 1/sqrt(hidden_size) scale after the linear projection. Transformers'
        # PaliGemma projector is just Linear(weight, bias), so fold the inverse
        # scale into the exported tensors without changing third_party code.
        proj_scale = math.sqrt(2048.0)
        add("mm.input_projection.bias",   get(f"{mm}.bias").to(torch.float32) * proj_scale)
        add("mm.input_projection.weight", get(f"{mm}.weight") * proj_scale)

        add("v.patch_embd.bias",      get(f"{ve}.embeddings.patch_embedding.bias"))
        add("v.patch_embd.weight",    get(f"{ve}.embeddings.patch_embedding.weight"))
        add("v.position_embd.weight", get(f"{ve}.embeddings.position_embedding.weight"))

        for i in range(27):
            src = f"{ve}.encoder.layers.{i}"
            dst = f"v.blk.{i}"
            add(f"{dst}.ln1.bias",         get(f"{src}.layer_norm1.bias").to(torch.float32))
            add(f"{dst}.ln1.weight",       get(f"{src}.layer_norm1.weight").to(torch.float32))
            add(f"{dst}.ln2.bias",         get(f"{src}.layer_norm2.bias").to(torch.float32))
            add(f"{dst}.ln2.weight",       get(f"{src}.layer_norm2.weight").to(torch.float32))
            add(f"{dst}.ffn_up.bias",      get(f"{src}.mlp.fc1.bias").to(torch.float32))
            add(f"{dst}.ffn_up.weight",    get(f"{src}.mlp.fc1.weight"))
            add(f"{dst}.ffn_down.bias",    get(f"{src}.mlp.fc2.bias").to(torch.float32))
            add(f"{dst}.ffn_down.weight",  get(f"{src}.mlp.fc2.weight"))
            add(f"{dst}.attn_k.bias",      get(f"{src}.self_attn.k_proj.bias").to(torch.float32))
            add(f"{dst}.attn_k.weight",    get(f"{src}.self_attn.k_proj.weight"))
            add(f"{dst}.attn_out.bias",    get(f"{src}.self_attn.out_proj.bias").to(torch.float32))
            add(f"{dst}.attn_out.weight",  get(f"{src}.self_attn.out_proj.weight"))
            add(f"{dst}.attn_q.bias",      get(f"{src}.self_attn.q_proj.bias").to(torch.float32))
            add(f"{dst}.attn_q.weight",    get(f"{src}.self_attn.q_proj.weight"))
            add(f"{dst}.attn_v.bias",      get(f"{src}.self_attn.v_proj.bias").to(torch.float32))
            add(f"{dst}.attn_v.weight",    get(f"{src}.self_attn.v_proj.weight"))

        add("v.post_ln.bias",   get(f"{ve}.post_layernorm.bias").to(torch.float32))
        add("v.post_ln.weight", get(f"{ve}.post_layernorm.weight").to(torch.float32))

    quantizer.finish()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {out} ({out.stat().st_size / (1024 * 1024):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
