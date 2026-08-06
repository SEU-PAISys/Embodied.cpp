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

"""Convert a LeRobot SmolVLA checkpoint (libero finetune) to an Embodied.cpp
GGUF that the unified ``vla-server`` can load when SmolVLA support is enabled.

This converter targets the HuggingFaceVLA/smolvla_libero finetune, whose
layout differs from the lerobot/smolvla_base "stock" checkpoint in three ways:
  * text_model uses all 32 layers (num_vlm_layers=0 means “full”)
  * lm_expert uses all 32 layers (num_expert_layers=-1 means “full”)
  * expert_width_multiplier=0.5 -> expert_hidden=480, expert_inter=1280

The vision tower is exported separately via
``convert_smolvla_mmproj_to_gguf.py``; this file produces the policy GGUF
containing the real pixel-shuffle connector, text backbone, action expert, the
five top-level action projection / time MLP modules, and MEAN_STD statistics.

The naming convention mirrors scripts/convert_pi05_to_gguf.py so that the
runtime loader (models/smolvla.cpp) reuses the same vlm.blk.* / aex.blk.*
tensor-name grammar and the same state_mean/state_std/action_mean/action_std
statistics-tensor convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf

ARCH = "smolvla"
KV = lambda name: f"{ARCH}.{name}"

# prefixes inside the LeRobot safetensors checkpoint
PFX_VLM_TXT  = "model.vlm_with_expert.vlm.model.text_model"
PFX_VLM_HEAD = "model.vlm_with_expert.vlm.lm_head.weight"
PFX_AEX      = "model.vlm_with_expert.lm_expert"
PFX_TOP      = "model"

# LeRobot's SmolVLA joint forward uses its local apply_rope() helper, whose
# max_wavelength default is 10_000. It does not use the base SmolVLM config's
# rope_theta (100_000) on this policy execution path.
ROPE_THETA   = 10000.0
RMS_NORM_EPS = 1e-5


def _bf16_to_u16_bytes(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.view(torch.uint16).contiguous().cpu().numpy()


def _f32_np(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.float32).contiguous().cpu().numpy()


def _add_one_tensor(writer: gguf.GGUFWriter, dst_name: str, t: torch.Tensor) -> None:
    if t.dtype == torch.float32:
        writer.add_tensor(dst_name, _f32_np(t),
                          raw_dtype=gguf.GGMLQuantizationType.F32)
    elif t.dtype == torch.bfloat16:
        writer.add_tensor(dst_name, _bf16_to_u16_bytes(t),
                          raw_shape=list(t.shape),
                          raw_dtype=gguf.GGMLQuantizationType.BF16)
    else:
        raise NotImplementedError(f"unsupported dtype {t.dtype} for {dst_name}")


def _stream_block(writer: gguf.GGUFWriter, sf, src_pfx: str, dst_pfx: str,
                  n_layers: int) -> None:
    """Stream a Gemma-style transformer block (RMSNorm + GQA + SwiGLU FFN).

    Both the SmolVLM2 text backbone and the SmolVLA action expert use plain
    RMSNorm (NOT AdaRMSNorm), so the simple suffix map is enough.
    """
    suffix_map = [
        ("input_layernorm.weight",          "attn_norm.weight"),
        ("self_attn.q_proj.weight",         "attn_q.weight"),
        ("self_attn.k_proj.weight",         "attn_k.weight"),
        ("self_attn.v_proj.weight",         "attn_v.weight"),
        ("self_attn.o_proj.weight",         "attn_o.weight"),
        ("post_attention_layernorm.weight", "ffn_norm.weight"),
        ("mlp.gate_proj.weight",            "ffn_gate.weight"),
        ("mlp.up_proj.weight",              "ffn_up.weight"),
        ("mlp.down_proj.weight",           "ffn_down.weight"),
    ]
    for i in range(n_layers):
        for src_suf, dst_suf in suffix_map:
            t = sf.get_tensor(f"{src_pfx}.layers.{i}.{src_suf}")
            _add_one_tensor(writer, f"{dst_pfx}.blk.{i}.{dst_suf}", t)


def _load_stats(ckpt: Path, real_state_dim: int, real_action_dim: int):
    out = {}
    norm_sf = ckpt / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    unnorm_sf = ckpt / "policy_postprocessor_step_1_unnormalizer_processor.safetensors"
    if not norm_sf.is_file() or not unnorm_sf.is_file():
        raise SystemExit(f"missing norm stats: {norm_sf} / {unnorm_sf}")
    for src, entries in [
        (norm_sf, [
            ("observation.state.mean", "state_mean", real_state_dim),
            ("observation.state.std",  "state_std",  real_state_dim),
        ]),
        (unnorm_sf, [
            ("action.mean", "action_mean", real_action_dim),
            ("action.std",  "action_std",  real_action_dim),
        ]),
    ]:
        with safe_open(str(src), framework="pt") as f:
            keys = set(f.keys())
            for name, dst, dim in entries:
                if name not in keys:
                    raise SystemExit(f"norm stats missing key {name} in {src.name}")
                arr = f.get_tensor(name).float().numpy().reshape(-1)
                if arr.size != dim:
                    raise SystemExit(f"{name}: dim {arr.size} != expected {dim}")
                out[dst] = arr.astype(np.float32, copy=False)
                print(f"  stats: loaded {dst} ({name}) shape={arr.shape} from {src.name}")
    return out


def _add_kv(writer: gguf.GGUFWriter, cfg: dict) -> None:
    writer.add_string(KV("architecture"),            ARCH)
    writer.add_uint32(KV("hidden"),                  cfg["hidden"])
    writer.add_uint32(KV("intermediate"),            cfg["intermediate"])
    writer.add_uint32(KV("n_q_heads"),               cfg["n_q_heads"])
    writer.add_uint32(KV("n_kv_heads"),               cfg["n_kv_heads"])
    writer.add_uint32(KV("head_dim"),                cfg["head_dim"])
    writer.add_uint32(KV("n_layers"),                cfg["n_layers"])
    writer.add_uint32(KV("vocab_size"),              cfg["vocab_size"])
    writer.add_uint32(KV("expert_h"),                cfg["expert_h"])
    writer.add_uint32(KV("expert_inter"),            cfg["expert_inter"])
    writer.add_uint32(KV("expert_n_q_heads"),        cfg["expert_n_q_heads"])
    writer.add_uint32(KV("expert_n_kv_heads"),       cfg["expert_n_kv_heads"])
    writer.add_uint32(KV("expert_head_dim"),         cfg["expert_head_dim"])
    writer.add_uint32(KV("expert_n_layers"),         cfg["expert_n_layers"])
    writer.add_uint32(KV("chunk_size"),              cfg["chunk_size"])
    writer.add_uint32(KV("num_steps"),               cfg["num_steps"])
    writer.add_uint32(KV("n_action_steps"),          cfg["n_action_steps"])
    writer.add_uint32(KV("max_state_dim"),           cfg["max_state_dim"])
    writer.add_uint32(KV("max_action_dim"),          cfg["max_action_dim"])
    writer.add_uint32(KV("real_state_dim"),          cfg["real_state_dim"])
    writer.add_uint32(KV("real_action_dim"),         cfg["real_action_dim"])
    writer.add_uint32(KV("tokenizer_max_length"),    cfg["tokenizer_max_length"])
    writer.add_uint32(KV("self_attn_every_n_layers"), cfg["self_attn_every_n_layers"])
    writer.add_uint32(KV("pixel_shuffle_scale"),      cfg["pixel_shuffle_scale"])
    writer.add_float64(KV("min_period"),             cfg["min_period"])
    writer.add_float64(KV("max_period"),             cfg["max_period"])
    writer.add_float64(KV("rope_theta"),              cfg["rope_theta"])
    writer.add_float32(KV("rms_norm_eps"),           cfg["rms_norm_eps"])
    writer.add_float32(KV("norm_eps"),               cfg["norm_eps"])
    writer.add_string(KV("state_norm_mode"),         cfg["state_norm_mode"])
    writer.add_string(KV("action_norm_mode"),        cfg["action_norm_mode"])
    writer.add_string(KV("attention_mode"),           cfg["attention_mode"])


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert a LeRobot SmolVLA (libero finetune) checkpoint "
                    "into an Embodied.cpp policy GGUF."
    )
    ap.add_argument("--ckpt", type=Path, required=True,
                    help="LeRobot smolvla_libero checkpoint dir "
                         "(model.safetensors + config.json + policy_*processor*.safetensors)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output GGUF path (default: <ckpt>/smolvla.gguf)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = (args.out or ckpt / "smolvla.gguf").resolve()
    sf_path  = ckpt / "model.safetensors"
    cfg_path = ckpt / "config.json"
    if not sf_path.exists(): raise SystemExit(f"missing {sf_path}")
    if not cfg_path.exists(): raise SystemExit(f"missing {cfg_path}")

    cfg_json = json.loads(cfg_path.read_text())
    if cfg_json.get("type") != "smolvla":
        raise SystemExit(f"config.json type={cfg_json.get('type')!r}, expected 'smolvla'")

    # fixed-by-architecture constants (SmolVLM2-500M text backbone + 32-layer expert)
    cfg = dict(
        hidden=960, intermediate=2560,
        n_q_heads=15, n_kv_heads=5, head_dim=64,
        expert_h=480, expert_inter=1280,
        expert_n_q_heads=15, expert_n_kv_heads=5, expert_head_dim=64,
    )
    cfg["chunk_size"]            = int(cfg_json["chunk_size"])
    cfg["num_steps"]             = int(cfg_json["num_steps"])
    cfg["n_action_steps"]        = int(cfg_json["n_action_steps"])
    cfg["max_state_dim"]         = int(cfg_json["max_state_dim"])
    cfg["max_action_dim"]        = int(cfg_json["max_action_dim"])
    cfg["real_state_dim"]        = int(cfg_json["input_features"]["observation.state"]["shape"][0])
    cfg["real_action_dim"]       = int(cfg_json["output_features"]["action"]["shape"][0])
    cfg["tokenizer_max_length"]  = int(cfg_json["tokenizer_max_length"])
    cfg["self_attn_every_n_layers"] = int(cfg_json["self_attn_every_n_layers"])
    # SmolVLA connector = pixel_shuffle(scale) + Linear(768*scale^2 -> text_hidden)
    # We hardcode scale=4 (SmolVLM2 default) but also export it as metadata so
    # the runtime can recompute the connector input dim correctly for other
    # checkpoints.
    cfg["pixel_shuffle_scale"] = 4
    cfg["min_period"]            = float(cfg_json["min_period"])
    cfg["max_period"]            = float(cfg_json["max_period"])
    cfg["rope_theta"]            = ROPE_THETA
    cfg["rms_norm_eps"]          = RMS_NORM_EPS
    cfg["norm_eps"]              = 1e-8
    nm = cfg_json.get("normalization_mapping", {})
    cfg["state_norm_mode"]  = str(nm.get("STATE",  "MEAN_STD")).upper()
    cfg["action_norm_mode"] = str(nm.get("ACTION", "MEAN_STD")).upper()
    cfg["attention_mode"]   = str(cfg_json.get("attention_mode", "cross_attn"))
    if "cross" not in cfg["attention_mode"]:
        raise SystemExit(
            f"unsupported attention_mode={cfg['attention_mode']!r}; "
            "this runtime implements SmolVLA cross-attention checkpoints"
        )
    if bool(cfg_json.get("add_image_special_tokens", False)):
        raise SystemExit(
            "add_image_special_tokens=true is not supported by the current "
            "Embodied.cpp SmolVLA prefix builder"
        )
    for k in ("state_norm_mode", "action_norm_mode"):
        if cfg[k] != "MEAN_STD":
            raise SystemExit(
                f"unsupported {k}={cfg[k]!r}; the SmolVLA runtime currently "
                "supports MEAN_STD checkpoints only"
            )

    print(f"opening {sf_path}")
    sf = safe_open(str(sf_path), framework="pt")
    keys = set(sf.keys())

    # infer actual layer counts from the checkpoint
    def _maxlayer(pfx: str) -> int:
        m = -1
        for k in keys:
            if k.startswith(pfx):
                try: m = max(m, int(k[len(pfx):].split(".", 1)[0]))
                except ValueError: pass
        return m + 1
    n_vlm = _maxlayer(f"{PFX_VLM_TXT}.layers.")
    n_aex = _maxlayer(f"{PFX_AEX}.layers.")
    if n_vlm <= 0: raise SystemExit("cannot find text_model layers")
    if n_aex <= 0: raise SystemExit("cannot find lm_expert layers")
    cfg["n_layers"]        = n_vlm
    cfg["expert_n_layers"] = n_aex

    # vocab from lm_head
    lm_head_w = sf.get_slice(PFX_VLM_HEAD).get_shape()
    if lm_head_w[1] != cfg["hidden"]:
        raise SystemExit(f"lm_head hidden mismatch: cfg={cfg['hidden']} ckpt={lm_head_w[1]}")
    cfg["vocab_size"] = int(lm_head_w[0])

    # sanity-check a few module dims
    q0  = sf.get_slice(f"{PFX_VLM_TXT}.layers.0.self_attn.q_proj.weight").get_shape()
    kv0 = sf.get_slice(f"{PFX_VLM_TXT}.layers.0.self_attn.k_proj.weight").get_shape()
    if q0[1] != cfg["hidden"]:
        raise SystemExit(f"VLM hidden mismatch: cfg={cfg['hidden']} ckpt={q0[1]}")
    if q0[0] != cfg["n_q_heads"] * cfg["head_dim"]:
        raise SystemExit(f"VLM q_proj rows {q0[0]} != n_q_heads*head_dim {cfg['n_q_heads']*cfg['head_dim']}")
    if kv0[0] != cfg["n_kv_heads"] * cfg["head_dim"]:
        raise SystemExit(f"VLM k_proj rows {kv0[0]} != n_kv_heads*head_dim")
    exq0 = sf.get_slice(f"{PFX_AEX}.layers.0.self_attn.q_proj.weight").get_shape()
    if exq0[1] != cfg["expert_h"]:
        raise SystemExit(f"expert_h mismatch: cfg={cfg['expert_h']} ckpt={exq0[1]}")
    kv_width = cfg["n_kv_heads"] * cfg["head_dim"]
    for i in range(n_aex):
        self_attn = cfg["self_attn_every_n_layers"] > 0 and i % cfg["self_attn_every_n_layers"] == 0
        expected_kv_input = cfg["expert_h"] if self_attn else kv_width
        for proj in ("k_proj", "v_proj"):
            shape = sf.get_slice(f"{PFX_AEX}.layers.{i}.self_attn.{proj}.weight").get_shape()
            if shape[1] != expected_kv_input:
                mode = "self-attention" if self_attn else "cross-attention"
                raise SystemExit(
                    f"expert layer {i} {proj} input={shape[1]} != "
                    f"expected {expected_kv_input} for {mode}"
                )
    print(f"resolved cfg: vlm hidden={cfg['hidden']} n_vlm={cfg['n_layers']} "
          f"heads={cfg['n_q_heads']}q/{cfg['n_kv_heads']}kv×{cfg['head_dim']} "
          f"expert_h={cfg['expert_h']} n_aex={cfg['expert_n_layers']} "
          f"expert_inter={cfg['expert_inter']} vocab={cfg['vocab_size']} "
          f"chunk={cfg['chunk_size']} steps={cfg['num_steps']} "
          f"real_state={cfg['real_state_dim']} real_action={cfg['real_action_dim']} "
          f"state_norm={cfg['state_norm_mode']} action_norm={cfg['action_norm_mode']} "
          f"attn_mode={cfg['attention_mode']} self_attn_every_n={cfg['self_attn_every_n_layers']}")

    print("loading normalizer stats...")
    stats = _load_stats(ckpt, cfg["real_state_dim"], cfg["real_action_dim"])

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out}")
    writer = gguf.GGUFWriter(str(out), arch=ARCH)
    _add_kv(writer, cfg)

    # VLM text backbone
    _add_one_tensor(writer, "token_embd.weight",       sf.get_tensor(f"{PFX_VLM_TXT}.embed_tokens.weight"))
    _add_one_tensor(writer, "vlm.output_norm.weight",  sf.get_tensor(f"{PFX_VLM_TXT}.norm.weight"))
    _add_one_tensor(writer, "vlm.lm_head.weight",      sf.get_tensor(PFX_VLM_HEAD))
    _stream_block(writer, sf, PFX_VLM_TXT, "vlm", cfg["n_layers"])

    # action expert (lm_expert)
    _add_one_tensor(writer, "aex.output_norm.weight", sf.get_tensor(f"{PFX_AEX}.norm.weight"))
    _stream_block(writer, sf, PFX_AEX, "aex", cfg["expert_n_layers"])

    # top-level action projection / time MLP modules
    for suf in ["state_proj.weight", "state_proj.bias",
                "action_in_proj.weight", "action_in_proj.bias",
                "action_out_proj.weight", "action_out_proj.bias",
                "action_time_mlp_in.weight", "action_time_mlp_in.bias",
                "action_time_mlp_out.weight", "action_time_mlp_out.bias"]:
        _add_one_tensor(writer, suf, sf.get_tensor(f"{PFX_TOP}.{suf}"))

    # SmolVLA vision-language connector (modality_projection.proj) shipped here
    # rather than in the mmproj GGUF because its pixel_shuffle + (768*16 -> 960)
    # projection doesn't match any existing mtmd projector type and is applied
    # on the Embodied.cpp host side instead.
    c_prefix = "model.vlm_with_expert.vlm.model.connector.modality_projection.proj"
    _add_one_tensor(writer, "connector.weight", sf.get_tensor(f"{c_prefix}.weight"))
    _add_one_tensor(writer, "connector.bias",
                    torch.zeros(cfg["hidden"], dtype=torch.float32))  # no bias in upstream

    # normalization statistics as F32 tensors (matches pi05 convention)
    for k, v in stats.items():
        writer.add_tensor(k, v, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"done. {out} ({out.stat().st_size / (1024*1024):.1f} MiB)")
    print("note: the SigLIP vision tower is in the separate mmproj GGUF; "
          "the real pixel-shuffle connector weights are stored in this policy GGUF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
