#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export the SigLIP vision tower of a LeRobot SmolVLA checkpoint as an
mtmd-compatible identity-proxy mmproj GGUF.

This reuses the same mtmd "clip" projection infrastructure as the pi0.5
mmproj converter (PROJECTOR_TYPE_PALIGEMMA path) implemented in
third_party/llama.cpp/tools/mtmd/models/siglip.cpp. Only the SigLIP
vision tower and an identity projection are exported here. The real
pixel-shuffle connector, text backbone, action expert, flow modules, and
normalization statistics live in the companion policy GGUF produced by
``convert_smolvla_to_gguf.py``.

Reference:
  * lerobot/policies/smolvla/smolvlm_with_expert.py  (upstream module)
  * scripts/convert_pi05_mmproj_to_gguf.py            (sibling template)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf

# --- SmolVLA vision tower layout (verified against the LeRobot checkpoint) ----
V_PREFIX = "model.vlm_with_expert.vlm.model.vision_model"
C_PREFIX = "model.vlm_with_expert.vlm.model.connector.modality_projection"

SIGLIP_IMAGE_SIZE      = 512          # resize_imgs_with_padding = (512, 512)
SIGLIP_PATCH_SIZE      = 16
SIGLIP_EMBED_DIM       = 768
SIGLIP_FFN_DIM         = 3072
SIGLIP_N_LAYERS        = 12            # encoder.layers.0..11
SIGLIP_N_HEADS         = 12            # q/k/v all 768 -> 12x64
SIGLIP_N_POSITIONS     = 1024          # (512/16)**2 = 32x32 = 1024
# Identity-proxy projection: mtmd's PALIGEMMA projector does mul_mat(W, cur)
# then add(b) then scale(1/sqrt(cur->ne[0])). We set W = identity (768x768)
# so the graph becomes a passthrough returning raw SigLIP encoder output
# (only distorted by the 1/sqrt(768) scale, which the Embodied.cpp SmolVLA
# runtime host code multiplies back by sqrt(768) to undo). The actual SmolVLA
# connector (pixel_shuffle + Linear(12288->960)) is then applied in
# models/smolvla.cpp::predict() using the resident connector.weight tensor
# stored in the main smolvla.gguf. This keeps the model-specific logic inside
# Embodied.cpp and avoids patching third_party/llama.cpp.
SIGLIP_PROJECTION_DIM  = SIGLIP_EMBED_DIM  # passthrough: input dim == output dim
SIGLIP_MEAN            = [0.5, 0.5, 0.5]
SIGLIP_STD             = [0.5, 0.5, 0.5]


def _bf16_to_u16_bytes(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.view(torch.uint16).contiguous().cpu().numpy()


def _f32_np(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.float32).contiguous().cpu().numpy()


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
        description="Export the SigLIP vision tower of a LeRobot SmolVLA "
                    "checkpoint as an mtmd-compatible identity-proxy mmproj "
                    "GGUF (paligemma projector type)."
    )
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="LeRobot SmolVLA checkpoint dir containing model.safetensors + config.json")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output SmolVLA mmproj GGUF path")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    sf_path = ckpt / "model.safetensors"
    if not sf_path.is_file():
        raise SystemExit(f"missing {sf_path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # SmolVLA is a "SigLIP + connector + SmolLM2 + flow action expert" stack.
    # The mtmd clip loader is reused with projector_type=paligemma so that
    # the existing SigLIP vision graph + linear projector path will pick up
    # these tensors unchanged.
    writer = gguf.GGUFWriter(str(args.out), arch="clip")
    writer.add_string("general.name", "lerobot_smolvla_libero")
    writer.add_string("general.type", "mmproj")
    writer.add_string("general.finetune", "lerobot_smolvla_libero")
    writer.add_string("general.size_label", "450M")
    writer.add_uint32("general.quantization_version", 2)
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_string("clip.projector_type", "paligemma")
    writer.add_bool("clip.use_gelu", True)
    writer.add_uint32("clip.vision.image_size",         SIGLIP_IMAGE_SIZE)
    writer.add_uint32("clip.vision.patch_size",         SIGLIP_PATCH_SIZE)
    writer.add_uint32("clip.vision.embedding_length",  SIGLIP_EMBED_DIM)
    writer.add_uint32("clip.vision.feed_forward_length", SIGLIP_FFN_DIM)
    writer.add_uint32("clip.vision.block_count",        SIGLIP_N_LAYERS)
    writer.add_uint32("clip.vision.attention.head_count", SIGLIP_N_HEADS)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    writer.add_uint32("clip.vision.projection_dim",    SIGLIP_PROJECTION_DIM)
    writer.add_array("clip.vision.image_mean", SIGLIP_MEAN)
    writer.add_array("clip.vision.image_std",  SIGLIP_STD)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)

    with safe_open(str(sf_path), framework="pt") as sf:
        keys = set(sf.keys())

        def need(name: str) -> None:
            if name not in keys:
                raise SystemExit(f"missing tensor in safetensors: {name}")

        def get(name: str) -> torch.Tensor:
            need(name)
            return sf.get_tensor(name)

        # --- patch / position embeddings -----------------------------------
        # Export the patch-embedding conv weight as F32 (instead of BF16) so the
        # ggml-cuda im2col path used during clip_encode_float_image picks a
        # supported F16/F32 dst type for patch size 16; this matches the
        # working pi0.5 PaliGemma-224 mmproj (which itself ships F32 patch
        # embeddings baked in upstream).
        patch_w = get(f"{V_PREFIX}.embeddings.patch_embedding.weight").to(torch.float32)
        _add_tensor(writer, "v.patch_embd.weight", patch_w)
        _add_tensor(writer, "v.patch_embd.bias",
                    get(f"{V_PREFIX}.embeddings.patch_embedding.bias").to(torch.float32))
        _add_tensor(writer, "v.position_embd.weight",
                    get(f"{V_PREFIX}.embeddings.position_embedding.weight").to(torch.float32))

        # --- encoder layers -------------------------------------------------
        for i in range(SIGLIP_N_LAYERS):
            src = f"{V_PREFIX}.encoder.layers.{i}"
            dst = f"v.blk.{i}"
            _add_tensor(writer, f"{dst}.ln1.bias",   get(f"{src}.layer_norm1.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.ln1.weight", get(f"{src}.layer_norm1.weight").to(torch.float32))
            _add_tensor(writer, f"{dst}.ln2.bias",   get(f"{src}.layer_norm2.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.ln2.weight", get(f"{src}.layer_norm2.weight").to(torch.float32))
            _add_tensor(writer, f"{dst}.ffn_up.bias",   get(f"{src}.mlp.fc1.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.ffn_up.weight", get(f"{src}.mlp.fc1.weight"))
            _add_tensor(writer, f"{dst}.ffn_down.bias", get(f"{src}.mlp.fc2.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.ffn_down.weight", get(f"{src}.mlp.fc2.weight"))
            _add_tensor(writer, f"{dst}.attn_k.bias",   get(f"{src}.self_attn.k_proj.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.attn_k.weight", get(f"{src}.self_attn.k_proj.weight"))
            _add_tensor(writer, f"{dst}.attn_out.bias",  get(f"{src}.self_attn.out_proj.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.attn_out.weight", get(f"{src}.self_attn.out_proj.weight"))
            _add_tensor(writer, f"{dst}.attn_q.bias",   get(f"{src}.self_attn.q_proj.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.attn_q.weight", get(f"{src}.self_attn.q_proj.weight"))
            _add_tensor(writer, f"{dst}.attn_v.bias",   get(f"{src}.self_attn.v_proj.bias").to(torch.float32))
            _add_tensor(writer, f"{dst}.attn_v.weight", get(f"{src}.self_attn.v_proj.weight"))

        # --- post layernorm -------------------------------------------------
        _add_tensor(writer, "v.post_ln.bias",
                    get(f"{V_PREFIX}.post_layernorm.bias").to(torch.float32))
        _add_tensor(writer, "v.post_ln.weight",
                    get(f"{V_PREFIX}.post_layernorm.weight").to(torch.float32))

        # --- identity-proxy connector (passthrough) ----------------------
        # Don't write the real SmolVLA connector here — its pixel_shuffle +
        # 12288->960 Linear doesn't fit mtmd's existing projector paths. The
        # real connector tensor lives in the main smolvla.gguf; here we ship
        # a 768x768 identity + zero bias so mtmd's PALIGEMMA projector becomes
        # a passive passthrough of the raw SigLIP encoder output (scaled by
        # 1/sqrt(768) internally, which the Embodied.cpp SmolVLA runtime
        # reverses on the host side).
        # Keep the proxy projector in F32. Both ggml-cuda and ggml-cpu require
        # the projector add operands to use a compatible non-BF16 type; BF16
        # here crashes when PALIGEMMA adds the F32 zero bias.
        ident_w = torch.eye(SIGLIP_EMBED_DIM, dtype=torch.float32)
        _add_tensor(writer, "mm.input_projection.weight", ident_w)
        zero_bias = torch.zeros(SIGLIP_EMBED_DIM, dtype=torch.float32)
        _add_tensor(writer, "mm.input_projection.bias", zero_bias)
        # The upstream connector tensor (C_PREFIX.proj.weight) is intentionally
        # NOT exported here — it goes into the main smolvla.gguf instead.

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.out} ({args.out.stat().st_size / (1024 * 1024):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
