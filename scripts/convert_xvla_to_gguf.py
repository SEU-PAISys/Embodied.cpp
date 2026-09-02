#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Convert an X-VLA checkpoint (Florence-2 backbone + SoftPromptedTransformer)
to GGUF.

X-VLA = DaViT vision tower (4 stages, window + channel attention, depthwise
convs) + BART-style text encoder (post-LN, learned positions with offset 2)
+ a domain-conditioned transformer action head driven by a 10-step linear
flow-matching denoise loop.

The converter reads tensors straight from model.safetensors (no transformers
model class is instantiated, so HF-version compat issues do not apply) and
writes every tensor plus the architecture metadata the C++ runtime needs.

Tensor naming contract consumed by models/xvla.cpp:
  vit.s{S}.conv.w/.b          stage patch-embed conv (ggml [KW,KH,IC,OC])
  vit.s{S}.norm.w/.b          pre-conv norm if patch_prenorm[S] else post-conv
  vit.s{S}.p{J}.{sp,ch}.*     per-pair spatial/channel block weights
  vproj.pos_row/.pos_col      learned 2D position embeddings [D/2 each]
  vproj.temporal              cosine temporal embedding row 0
  vproj.proj.w                image_projection transposed for mul_mat
  vproj.proj_norm.w/.b        image_proj_norm
  text.tok_emb                shared token embedding [vocab, D]
  text.pos_emb                learned positions [max_pos+2, D] (offset 2)
  text.emb_norm.w/.b          layernorm_embedding
  text.blk.{i}.*              BART encoder layers (post-LN)
  act.vlm_proj.w/.b           Linear(projection_dim -> hidden)
  act.aux_proj.w/.b           same for auxiliary views
  act.pos_emb                 [max_len_seq, hidden]
  act.out_norm.w/.b           final LayerNorm
  act.aenc.fc/.bias           DomainAwareLinear action encoder per-domain rows
  act.adec.fc/.bias           DomainAwareLinear action decoder per-domain rows
  act.prompts                 soft_prompt_hub [num_domains, len*hidden]
  act.blk.{i}.*               action transformer blocks (pre-LN, GELU-tanh)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if GGUF_PY.exists():
    sys.path.insert(0, str(GGUF_PY))

try:
    import gguf
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"failed to import gguf (vendored at {GGUF_PY}): {exc}")

try:
    from safetensors.numpy import load_file
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"failed to import safetensors: {exc}")

ARCH = "xvla"
KV = lambda name: f"{ARCH}.{name}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-dir", required=True, type=Path,
                   help="directory containing model.safetensors and config.json")
    p.add_argument("--output", required=True, type=Path, help="output GGUF path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    st_path = args.hf_dir / "model.safetensors"
    cfg_path = args.hf_dir / "config.json"
    if not st_path.is_file():
        raise SystemExit(f"missing {st_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    fcfg = cfg["florence_config"]
    tcfg = fcfg["text_config"]
    vcfg = fcfg["vision_config"]

    depths = [int(v) for v in vcfg["depths"]]
    dim_embed = [int(v) for v in vcfg["dim_embed"]]
    num_heads = [int(v) for v in vcfg["num_heads"]]
    num_groups = [int(v) for v in vcfg["num_groups"]]
    patch_size = [int(v) for v in vcfg["patch_size"]]
    patch_stride = [int(v) for v in vcfg["patch_stride"]]
    patch_padding = [int(v) for v in vcfg["patch_padding"]]
    patch_prenorm = [bool(v) for v in vcfg["patch_prenorm"]]
    window_size = int(vcfg["window_size"])
    image_size = 224  # CLIPImageProcessor resize target from preprocessor_config
    projection_dim = int(fcfg["projection_dim"])

    hidden = int(cfg["hidden_size"])
    depth = int(cfg["depth"])
    n_heads = int(cfg["num_heads"])
    mlp_ratio = float(cfg["mlp_ratio"])
    act_ffn = int(hidden * mlp_ratio)
    num_domains = int(cfg["num_domains"])
    len_soft_prompts = int(cfg["len_soft_prompts"])
    dim_time = int(cfg["dim_time"])
    max_len_seq = int(cfg["max_len_seq"])
    use_hetero_proj = bool(cfg.get("use_hetero_proj", False))
    if use_hetero_proj:
        raise SystemExit("use_hetero_proj=true checkpoints are not supported")
    num_actions = int(cfg["num_actions"])
    action_mode = str(cfg["action_mode"]).lower()
    if action_mode != "ee6d":
        raise SystemExit(f"unsupported action_mode {action_mode!r} (only ee6d)")
    dim_action = 20   # EE6DActionSpace.dim_action
    gripper_idx = (9, 19)
    dim_proprio = dim_action
    denoise_steps = 10  # generate_actions default
    text_len = 50       # XVLAProcessor.language_max_length

    d_model = int(tcfg["d_model"])
    enc_layers = int(tcfg.get("encoder_layers", 12))
    enc_heads = int(tcfg.get("encoder_attention_heads", 16))
    enc_ffn = int(tcfg.get("encoder_ffn_dim", 4096))
    vocab_size = int(tcfg["vocab_size"])
    max_pos = int(tcfg.get("max_position_embeddings", 4096))

    print(f"loading {st_path} ...")
    tensors = load_file(str(st_path))

    def get(name: str) -> np.ndarray:
        if name not in tensors:
            raise SystemExit(f"missing tensor {name}")
        return tensors[name]

    def require(shape: tuple, *dims: int) -> None:
        if tuple(shape) != tuple(dims):
            raise SystemExit(f"shape mismatch: got {shape}, expected {dims}")

    writer = gguf.GGUFWriter(args.output, arch=ARCH)
    writer.add_string(KV("architecture"), ARCH)

    def add_u32(key: str, val: int) -> None:
        writer.add_uint32(KV(key), int(val))

    def add_f32(key: str, val: float) -> None:
        writer.add_float32(KV(key), float(val))

    add_u32("num_actions", num_actions)
    add_u32("dim_action", dim_action)
    add_u32("dim_proprio", dim_proprio)
    add_u32("denoise_steps", denoise_steps)
    add_u32("gripper_idx_0", gripper_idx[0])
    add_u32("gripper_idx_1", gripper_idx[1])
    add_u32("hidden", hidden)
    add_u32("depth", depth)
    add_u32("heads", n_heads)
    add_u32("act_ffn", act_ffn)
    add_u32("num_domains", num_domains)
    add_u32("soft_prompts", len_soft_prompts)
    add_u32("dim_time", dim_time)
    add_u32("max_len_seq", max_len_seq)
    add_u32("text_len", text_len)

    add_u32("d_model", d_model)
    add_u32("enc_layers", enc_layers)
    add_u32("enc_heads", enc_heads)
    add_u32("enc_ffn", enc_ffn)
    add_u32("vocab", vocab_size)
    add_u32("max_pos", max_pos)

    add_u32("image_size", image_size)
    add_u32("window_size", window_size)
    add_u32("projection_dim", projection_dim)
    for i in range(4):
        add_u32(f"davit_depth_{i}", depths[i])
        add_u32(f"davit_dim_{i}", dim_embed[i])
        add_u32(f"davit_heads_{i}", num_heads[i])
        add_u32(f"davit_groups_{i}", num_groups[i])
        add_u32(f"davit_patch_{i}", patch_size[i])
        add_u32(f"davit_stride_{i}", patch_stride[i])
        add_u32(f"davit_padding_{i}", patch_padding[i])
        writer.add_uint8(KV(f"davit_prenorm_{i}"), 1 if patch_prenorm[i] else 0)

    def add_tensor(name: str, arr: np.ndarray, force_f32: bool = False) -> None:
        arr = np.ascontiguousarray(arr)
        if force_f32 or arr.dtype != np.float32:
            if arr.dtype == np.float16 or arr.dtype == np.float64:
                arr = arr.astype(np.float32)
        writer.add_tensor(name, arr)

    MAT = os.environ.get("VLA_XVLA_F32_WEIGHTS") != "1"

    def lin(name: str, arr: np.ndarray) -> None:
        """nn.Linear weight [out, in]: row-major bytes double as ggml [in, out]."""
        add_tensor(name, arr.astype(np.float16 if MAT else np.float32))

    def bias(name: str, arr: np.ndarray) -> None:
        add_tensor(name, arr.astype(np.float32))

    def norm_pair(stem: str, prefix: str) -> None:
        w = get(f"{prefix}.weight")
        b = get(f"{prefix}.bias")
        add_tensor(f"{stem}.w", w.astype(np.float32))
        add_tensor(f"{stem}.b", b.astype(np.float32))

    # ---------------- vision tower ----------------
    # gguf-py stores numpy shape reversed as ggml ne, so a PyTorch conv kernel
    # [OC, IC, KH, KW] lands as ggml ne [KW, KH, IC, OC] exactly as the
    # runtime's ggml_conv_2d expects. Same for depthwise [C, 1, 3, 3].
    for s in range(4):
        w = get(f"vlm.vision_tower.convs.{s}.proj.weight")
        require(w.shape, dim_embed[s], dim_embed[s - 1] if s else 3,
                patch_size[s], patch_size[s])
        add_tensor(f"vit.s{s}.conv.w", w.astype(np.float32))
        bias(f"vit.s{s}.conv.b", get(f"vlm.vision_tower.convs.{s}.proj.bias"))
        norm_pair(f"vit.s{s}.norm", f"vlm.vision_tower.convs.{s}.norm")

    def conv_block(stem: str, prefix: str, dim: int, ffn: int) -> None:
        attn = "window_attn" if stem.endswith(".sp") else "channel_attn"
        for kind in ("conv1", "conv2"):
            w = get(f"{prefix}.{kind}.fn.dw.weight")     # [C, 1, 3, 3]
            require(w.shape, dim, 1, 3, 3)
            add_tensor(f"{stem}.{kind}.w", w.astype(np.float32))
            bias(f"{stem}.{kind}.b", get(f"{prefix}.{kind}.fn.dw.bias"))
        norm_pair(f"{stem}.norm", f"{prefix}.{attn}.norm")
        lin(f"{stem}.qkv.w", get(f"{prefix}.{attn}.fn.qkv.weight"))
        bias(f"{stem}.qkv.b", get(f"{prefix}.{attn}.fn.qkv.bias"))
        lin(f"{stem}.proj.w", get(f"{prefix}.{attn}.fn.proj.weight"))
        bias(f"{stem}.proj.b", get(f"{prefix}.{attn}.fn.proj.bias"))
        norm_pair(f"{stem}.ffn_norm", f"{prefix}.ffn.norm")
        lin(f"{stem}.fc1.w", get(f"{prefix}.ffn.fn.net.fc1.weight"))
        bias(f"{stem}.fc1.b", get(f"{prefix}.ffn.fn.net.fc1.bias"))
        lin(f"{stem}.fc2.w", get(f"{prefix}.ffn.fn.net.fc2.weight"))
        bias(f"{stem}.fc2.b", get(f"{prefix}.ffn.fn.net.fc2.bias"))

    for s in range(4):
        for j in range(depths[s]):
            stem = f"vit.s{s}.p{j}"
            base = f"vlm.vision_tower.blocks.{s}.{j}"
            conv_block(f"{stem}.sp", f"{base}.spatial_block", dim_embed[s], act_ffn)
            conv_block(f"{stem}.ch", f"{base}.channel_block", dim_embed[s], act_ffn)

    # ---------------- image projection ----------------
    pos_rows = get("vlm.image_pos_embed.row_embeddings.weight")
    pos_cols = get("vlm.image_pos_embed.column_embeddings.weight")
    # LearnedAbsolutePositionEmbedding2D splits the tower output dim in half
    img_dim = dim_embed[-1]
    require(pos_rows.shape, 50, img_dim - img_dim // 2)
    require(pos_cols.shape, 50, img_dim // 2)
    add_tensor("vproj.pos_row", pos_rows.astype(np.float32))
    add_tensor("vproj.pos_col", pos_cols.astype(np.float32))
    temporal = get("vlm.visual_temporal_embed.pos_idx_to_embed")
    require(temporal.shape, 100, dim_embed[-1])
    add_tensor("vproj.temporal", temporal[0].astype(np.float32))
    proj = get("vlm.image_projection")              # [2048, 1024], used as x @ W
    require(proj.shape, dim_embed[-1], projection_dim)
    add_tensor("vproj.proj.w", np.transpose(proj, (1, 0)).astype(
        np.float16 if MAT else np.float32))
    norm_pair("vproj.proj_norm", "vlm.image_proj_norm")

    # ---------------- text encoder ----------------
    tok_emb = get("vlm.language_model.model.shared.weight")
    require(tok_emb.shape, vocab_size, d_model)
    lin("text.tok_emb", tok_emb)
    pos_emb = get("vlm.language_model.model.encoder.embed_positions.weight")
    require(pos_emb.shape, max_pos + 2, d_model)
    add_tensor("text.pos_emb", pos_emb.astype(np.float32))
    norm_pair("text.emb_norm", "vlm.language_model.model.encoder.layernorm_embedding")
    for i in range(enc_layers):
        base = f"vlm.language_model.model.encoder.layers.{i}"
        stem = f"text.blk.{i}"
        for kind, name in (("q", "q_proj"), ("k", "k_proj"),
                           ("v", "v_proj"), ("o", "out_proj")):
            lin(f"{stem}.attn_{kind}.w", get(f"{base}.self_attn.{name}.weight"))
            bias(f"{stem}.attn_{kind}.b", get(f"{base}.self_attn.{name}.bias"))
        norm_pair(f"{stem}.attn_norm", f"{base}.self_attn_layer_norm")
        lin(f"{stem}.fc1.w", get(f"{base}.fc1.weight"))
        bias(f"{stem}.fc1.b", get(f"{base}.fc1.bias"))
        lin(f"{stem}.fc2.w", get(f"{base}.fc2.weight"))
        bias(f"{stem}.fc2.b", get(f"{base}.fc2.bias"))
        norm_pair(f"{stem}.ffn_norm", f"{base}.final_layer_norm")

    # ---------------- action transformer ----------------
    lin("act.vlm_proj.w", get("transformer.vlm_proj.weight"))
    bias("act.vlm_proj.b", get("transformer.vlm_proj.bias"))
    lin("act.aux_proj.w", get("transformer.aux_visual_proj.weight"))
    bias("act.aux_proj.b", get("transformer.aux_visual_proj.bias"))
    pos = get("transformer.pos_emb")
    require(pos.shape, 1, max_len_seq, hidden)
    add_tensor("act.pos_emb", pos.reshape(max_len_seq, hidden).astype(np.float32))
    norm_pair("act.out_norm", "transformer.norm")

    aenc_fc = get("transformer.action_encoder.fc.weight")
    require(aenc_fc.shape, num_domains, (dim_action + dim_time + dim_proprio) * hidden)
    # Official DomainAwareLinear views the flat per-domain row directly as
    # [input_size, output_size] (in-major). The C++ graph consumes a
    # column-major [in, hidden] ggml tensor whose (i, o) offset is i + o*in,
    # so store the transposed-per-domain layout: new[i + o*in] = old[i*out + o].
    aenc_in = dim_action + dim_time + dim_proprio
    aenc_fc = aenc_fc.reshape(num_domains, aenc_in, hidden).transpose(0, 2, 1)
    add_tensor("act.aenc.fc",
               np.ascontiguousarray(aenc_fc).reshape(num_domains, -1).astype(np.float32))
    aenc_b = get("transformer.action_encoder.bias.weight")
    require(aenc_b.shape, num_domains, hidden)
    add_tensor("act.aenc.bias", aenc_b.astype(np.float32))
    adec_fc = get("transformer.action_decoder.fc.weight")
    require(adec_fc.shape, num_domains, hidden * dim_action)
    adec_fc = adec_fc.reshape(num_domains, hidden, dim_action).transpose(0, 2, 1)
    add_tensor("act.adec.fc",
               np.ascontiguousarray(adec_fc).reshape(num_domains, -1).astype(np.float32))
    adec_b = get("transformer.action_decoder.bias.weight")
    require(adec_b.shape, num_domains, dim_action)
    add_tensor("act.adec.bias", adec_b.astype(np.float32))

    prompts = get("transformer.soft_prompt_hub.weight")
    require(prompts.shape, num_domains, len_soft_prompts * hidden)
    add_tensor("act.prompts", prompts.astype(np.float32))

    for i in range(depth):
        base = f"transformer.blocks.{i}"
        stem = f"act.blk.{i}"
        norm_pair(f"{stem}.norm1", f"{base}.norm1")
        lin(f"{stem}.qkv.w", get(f"{base}.attn.qkv.weight"))
        bias(f"{stem}.qkv.b", get(f"{base}.attn.qkv.bias"))
        lin(f"{stem}.proj.w", get(f"{base}.attn.proj.weight"))
        bias(f"{stem}.proj.b", get(f"{base}.attn.proj.bias"))
        norm_pair(f"{stem}.norm2", f"{base}.norm2")
        lin(f"{stem}.fc1.w", get(f"{base}.mlp.fc1.weight"))
        bias(f"{stem}.fc1.b", get(f"{base}.mlp.fc1.bias"))
        lin(f"{stem}.fc2.w", get(f"{base}.mlp.fc2.weight"))
        bias(f"{stem}.fc2.b", get(f"{base}.mlp.fc2.bias"))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    total = sum(v.nbytes for v in tensors.values())
    print(f"wrote {args.output} ({total / 1e9:.2f} GiB source tensors)")


if __name__ == "__main__":
    main()
