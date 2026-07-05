#!/usr/bin/env python3
# Copyright 2026 VinRobotics
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

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open

import gguf

ARCH = "pi05"
KV = lambda name: f"{ARCH}.{name}"

PFX_VLM_CANDIDATES = [
    "model.paligemma_with_expert.paligemma.model.language_model",
    "paligemma_with_expert.paligemma.model.language_model",
]
PFX_VLM_HEAD_CANDIDATES = [
    "model.paligemma_with_expert.paligemma.lm_head.weight",
    "paligemma_with_expert.paligemma.lm_head.weight",
]
PFX_AEX_CANDIDATES = [
    "model.paligemma_with_expert.gemma_expert.model",
    "paligemma_with_expert.gemma_expert.model",
]
PFX_PROJ_CANDIDATES = [
    "model",
    "",
]

GEMMA_2B   = dict(hidden=2048, n_q_heads=8, n_kv_heads=1, head_dim=256, intermediate=16384)
GEMMA_300M = dict(expert_h=1024, expert_inter=4096)

ROPE_THETA   = 10000.0
RMS_NORM_EPS = 1e-6

def _bf16_to_u16_bytes(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        print(f"  warn: casting non-BF16 tensor (dtype={t.dtype}) to BF16 for storage")
        t = t.to(torch.bfloat16)
    return t.view(torch.uint16).contiguous().cpu().numpy()

def _f32_np(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.float32, t.dtype
    return t.contiguous().cpu().numpy()

def _join_key(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix

def _pick_prefix(keys: set[str], candidates: list[str], probe_suffix: str) -> str:
    for prefix in candidates:
        probe = _join_key(prefix, probe_suffix) if probe_suffix else prefix
        if probe in keys:
            return prefix
    raise SystemExit(f"cannot resolve checkpoint prefix for {probe_suffix or candidates[0]!r}")

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

def _stream_block(writer: gguf.GGUFWriter, sf, src_pfx: str, dst_pfx: str, n_layers: int) -> None:

    suffix_map = [
        ("input_layernorm.weight",          "attn_norm.weight"),
        ("self_attn.q_proj.weight",         "attn_q.weight"),
        ("self_attn.k_proj.weight",         "attn_k.weight"),
        ("self_attn.v_proj.weight",         "attn_v.weight"),
        ("self_attn.o_proj.weight",         "attn_o.weight"),
        ("post_attention_layernorm.weight", "ffn_norm.weight"),
        ("mlp.gate_proj.weight",            "ffn_gate.weight"),
        ("mlp.up_proj.weight",              "ffn_up.weight"),
        ("mlp.down_proj.weight",            "ffn_down.weight"),
    ]
    for i in range(n_layers):
        for src_suf, dst_suf in suffix_map:
            t = sf.get_tensor(f"{src_pfx}.layers.{i}.{src_suf}")
            _add_one_tensor(writer, f"{dst_pfx}.blk.{i}.{dst_suf}", t)

def _stream_adarms_block(writer: gguf.GGUFWriter, sf, src_pfx: str, dst_pfx: str, n_layers: int) -> None:
    # pi0.5 uses adaptive RMSNorm in the action expert; export the dense
    # scale/shift/gate modulators instead of plain Gemma norm weights.
    suffix_map = [
        ("input_layernorm.dense.weight",          "attn_norm.dense.weight"),
        ("input_layernorm.dense.bias",            "attn_norm.dense.bias"),
        ("self_attn.q_proj.weight",               "attn_q.weight"),
        ("self_attn.k_proj.weight",               "attn_k.weight"),
        ("self_attn.v_proj.weight",               "attn_v.weight"),
        ("self_attn.o_proj.weight",               "attn_o.weight"),
        ("post_attention_layernorm.dense.weight", "ffn_norm.dense.weight"),
        ("post_attention_layernorm.dense.bias",   "ffn_norm.dense.bias"),
        ("mlp.gate_proj.weight",                  "ffn_gate.weight"),
        ("mlp.up_proj.weight",                    "ffn_up.weight"),
        ("mlp.down_proj.weight",                  "ffn_down.weight"),
    ]
    for i in range(n_layers):
        for src_suf, dst_suf in suffix_map:
            t = sf.get_tensor(f"{src_pfx}.layers.{i}.{src_suf}")
            _add_one_tensor(writer, f"{dst_pfx}.blk.{i}.{dst_suf}", t)

def _read_norm_eps(meta_path: Path) -> Optional[float]:

    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    for step in meta.get("steps", []):
        if step.get("registry_name") in ("normalizer_processor", "unnormalizer_processor"):
            cfg = step.get("config", {})
            if "eps" in cfg:
                return float(cfg["eps"])
    return None

def _identity_stats(mode: str, prefix: str, dim: int) -> dict[str, np.ndarray]:
    if mode == "MEAN_STD":
        return {
            f"{prefix}_mean": np.zeros(dim, dtype=np.float32),
            f"{prefix}_std":  np.ones (dim, dtype=np.float32),
        }
    if mode == "QUANTILES":
        return {
            f"{prefix}_q01": -np.ones(dim, dtype=np.float32),
            f"{prefix}_q99":  np.ones(dim, dtype=np.float32),
        }
    raise ValueError(f"unsupported pi05 normalization mode {mode!r} for {prefix}")

def _load_stats(sf, ckpt_dir: Path,
                real_state_dim: int, real_action_dim: int,
                state_norm_mode: str, action_norm_mode: str) -> dict[str, np.ndarray]:

    out = {}
    out.update(_identity_stats(state_norm_mode,  "state",  real_state_dim))
    out.update(_identity_stats(action_norm_mode, "action", real_action_dim))

    def _load_from_processor_sf(meta_json: str, registry: str, key_prefix: str,
                                mode: str, dst_prefix: str, dim: int) -> bool:
        meta_path = ckpt_dir / meta_json
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            print(f"  stats: {meta_json} parse failed ({e}) - trying legacy")
            return False
        state_file = None
        for step in meta.get("steps", []):
            if step.get("registry_name") == registry:
                state_file = step.get("state_file")
                break
        if not state_file:
            return False
        sf_path = ckpt_dir / state_file
        if not sf_path.is_file():
            print(f"  stats: {sf_path.name} referenced by {meta_json} but missing")
            return False
        if mode == "MEAN_STD":
            stat_a, stat_b = "mean", "std"
        elif mode == "QUANTILES":
            stat_a, stat_b = "q01", "q99"
        else:
            raise ValueError(f"unsupported pi05 normalization mode {mode!r} for {dst_prefix}")
        with safe_open(str(sf_path), framework="pt") as f:
            keys = set(f.keys())
            key_a, key_b = f"{key_prefix}.{stat_a}", f"{key_prefix}.{stat_b}"
            if key_a not in keys or key_b not in keys:
                return False
            arr_a = f.get_tensor(key_a).float().numpy().reshape(-1)
            arr_b = f.get_tensor(key_b).float().numpy().reshape(-1)
        if arr_a.size != dim or arr_b.size != dim:
            print(f"  stats: {key_a} dim mismatch ({arr_a.size} vs {dim}) in {sf_path.name}")
            return False
        out[f"{dst_prefix}_{stat_a}"] = arr_a.astype(np.float32, copy=False)
        out[f"{dst_prefix}_{stat_b}"] = arr_b.astype(np.float32, copy=False)
        print(f"  stats: loaded {dst_prefix} {mode} from {sf_path.name} ({key_a}/{key_b})")
        return True

    got_state  = _load_from_processor_sf("policy_preprocessor.json",  "normalizer_processor",
                                         "observation.state", state_norm_mode, "state",
                                         real_state_dim)
    got_action = _load_from_processor_sf("policy_postprocessor.json", "unnormalizer_processor",
                                         "action", action_norm_mode, "action",
                                         real_action_dim)

    keys = set(sf.keys())
    def _try_legacy(key_a: str, key_b: str, dst_prefix: str,
                    stat_a: str, stat_b: str, dim: int):
        if key_a not in keys or key_b not in keys:
            print(f"  stats: legacy {key_a} / {key_b} missing - using identity for {dst_prefix}")
            return
        arr_a = sf.get_tensor(key_a).float().numpy().reshape(-1)
        arr_b = sf.get_tensor(key_b).float().numpy().reshape(-1)
        if arr_a.size != dim or arr_b.size != dim:
            print(f"  stats: legacy {key_a} dim mismatch ({arr_a.size} vs {dim}) - using identity")
            return
        out[f"{dst_prefix}_{stat_a}"] = arr_a.astype(np.float32, copy=False)
        out[f"{dst_prefix}_{stat_b}"] = arr_b.astype(np.float32, copy=False)
        print(f"  stats: loaded {dst_prefix} from model.safetensors ({key_a}/{key_b}) [legacy]")

    if not got_state:
        if state_norm_mode == "MEAN_STD":
            _try_legacy("normalize_inputs.buffer_observation_state.mean",
                        "normalize_inputs.buffer_observation_state.std",
                        "state", "mean", "std", real_state_dim)
        else:
            _try_legacy("normalize_inputs.buffer_observation_state.q01",
                        "normalize_inputs.buffer_observation_state.q99",
                        "state", "q01", "q99", real_state_dim)
    if not got_action:
        if action_norm_mode == "MEAN_STD":
            if "unnormalize_outputs.buffer_action.mean" in keys:
                _try_legacy("unnormalize_outputs.buffer_action.mean",
                            "unnormalize_outputs.buffer_action.std",
                            "action", "mean", "std", real_action_dim)
            else:
                _try_legacy("normalize_targets.buffer_action.mean",
                            "normalize_targets.buffer_action.std",
                            "action", "mean", "std", real_action_dim)
        else:
            if "unnormalize_outputs.buffer_action.q01" in keys:
                _try_legacy("unnormalize_outputs.buffer_action.q01",
                            "unnormalize_outputs.buffer_action.q99",
                            "action", "q01", "q99", real_action_dim)
            else:
                _try_legacy("normalize_targets.buffer_action.q01",
                            "normalize_targets.buffer_action.q99",
                            "action", "q01", "q99", real_action_dim)
    return out

def _add_kv(writer: gguf.GGUFWriter, cfg: dict) -> None:
    writer.add_string  (KV("architecture"),             ARCH)
    writer.add_string  (KV("paligemma_variant"),        cfg["paligemma_variant"])
    writer.add_string  (KV("action_expert_variant"),    cfg["action_expert_variant"])
    writer.add_uint32  (KV("hidden"),                   cfg["hidden"])
    writer.add_uint32  (KV("intermediate"),             cfg["intermediate"])
    writer.add_uint32  (KV("n_q_heads"),                cfg["n_q_heads"])
    writer.add_uint32  (KV("n_kv_heads"),               cfg["n_kv_heads"])
    writer.add_uint32  (KV("head_dim"),                 cfg["head_dim"])
    writer.add_uint32  (KV("n_layers"),                 cfg["n_layers"])
    writer.add_uint32  (KV("vocab_size"),               cfg["vocab_size"])
    writer.add_uint32  (KV("expert_h"),                 cfg["expert_h"])
    writer.add_uint32  (KV("expert_inter"),             cfg["expert_inter"])
    writer.add_uint32  (KV("chunk_size"),               cfg["chunk_size"])
    writer.add_uint32  (KV("num_steps"),                cfg["num_steps"])
    writer.add_uint32  (KV("n_action_steps"),           cfg["n_action_steps"])
    writer.add_uint32  (KV("max_state_dim"),            cfg["max_state_dim"])
    writer.add_uint32  (KV("max_action_dim"),           cfg["max_action_dim"])
    writer.add_uint32  (KV("real_state_dim"),           cfg["real_state_dim"])
    writer.add_uint32  (KV("real_action_dim"),          cfg["real_action_dim"])
    writer.add_uint32  (KV("tokenizer_max_length"),     cfg["tokenizer_max_length"])
    writer.add_float64 (KV("min_period"),               cfg["min_period"])
    writer.add_float64 (KV("max_period"),               cfg["max_period"])
    writer.add_float64 (KV("rope_theta"),               cfg["rope_theta"])
    writer.add_float32 (KV("rms_norm_eps"),             cfg["rms_norm_eps"])
    writer.add_float32 (KV("norm_eps"),                 cfg["norm_eps"])
    writer.add_string  (KV("state_norm_mode"),          cfg["state_norm_mode"])
    writer.add_string  (KV("action_norm_mode"),         cfg["action_norm_mode"])

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True,
        help="lerobot pi0.5 checkpoint dir (model.safetensors + config.json + policy_*processor.json)")
    ap.add_argument("--out", type=Path, default=None,
        help="Output GGUF path (default: <ckpt>/pi05.gguf)")
    args = ap.parse_args()

    ckpt = args.ckpt.resolve()
    out  = (args.out or ckpt / "pi05.gguf").resolve()
    sf_path  = ckpt / "model.safetensors"
    cfg_path = ckpt / "config.json"
    if not sf_path.exists():
        raise SystemExit(f"missing {sf_path}")
    if not cfg_path.exists():
        raise SystemExit(f"missing {cfg_path}")

    cfg_json = json.loads(cfg_path.read_text())
    if cfg_json.get("type") != "pi05":
        raise SystemExit(f"config.json type is {cfg_json.get('type')!r}, expected 'pi05'")

    cfg = dict(GEMMA_2B, **GEMMA_300M)
    cfg["paligemma_variant"]     = str(cfg_json.get("paligemma_variant", "gemma_2b"))
    cfg["action_expert_variant"] = str(cfg_json.get("action_expert_variant", "gemma_300m"))
    cfg["chunk_size"]            = int(cfg_json["chunk_size"])
    cfg["num_steps"]             = int(cfg_json["num_inference_steps"])
    cfg["n_action_steps"]        = int(cfg_json["n_action_steps"])
    cfg["max_state_dim"]         = int(cfg_json["max_state_dim"])
    cfg["max_action_dim"]        = int(cfg_json["max_action_dim"])
    cfg["min_period"]            = float(cfg_json["min_period"])
    cfg["max_period"]            = float(cfg_json["max_period"])
    cfg["tokenizer_max_length"]  = int(cfg_json["tokenizer_max_length"])
    cfg["real_state_dim"]        = int(cfg_json["input_features"]["observation.state"]["shape"][0])
    cfg["real_action_dim"]       = int(cfg_json["output_features"]["action"]["shape"][0])
    cfg["rope_theta"]            = ROPE_THETA
    cfg["rms_norm_eps"]          = RMS_NORM_EPS
    norm_map = cfg_json.get("normalization_mapping", {})
    cfg["state_norm_mode"]       = str(norm_map.get("STATE",  "QUANTILES")).upper()
    cfg["action_norm_mode"]      = str(norm_map.get("ACTION", "QUANTILES")).upper()
    for key in ("state_norm_mode", "action_norm_mode"):
        if cfg[key] not in ("MEAN_STD", "QUANTILES"):
            raise SystemExit(f"unsupported pi05 {key}={cfg[key]!r}; expected MEAN_STD or QUANTILES")

    norm_eps = _read_norm_eps(ckpt / "policy_preprocessor.json")
    if norm_eps is None:
        norm_eps = _read_norm_eps(ckpt / "policy_postprocessor.json")
    cfg["norm_eps"] = float(norm_eps if norm_eps is not None else 1e-8)

    print(f"opening {sf_path}")
    sf = safe_open(sf_path, framework="pt")
    keys = set(sf.keys())

    def _maxlayer(pfx: str) -> int:
        m = -1
        for k in keys:
            if k.startswith(pfx):
                try:
                    m = max(m, int(k[len(pfx):].split(".", 1)[0]))
                except ValueError:
                    pass
        return m + 1

    pfx_vlm = _pick_prefix(keys, PFX_VLM_CANDIDATES, "layers.0.self_attn.q_proj.weight")
    pfx_vlm_head = _pick_prefix(keys, PFX_VLM_HEAD_CANDIDATES, "")
    pfx_aex = _pick_prefix(keys, PFX_AEX_CANDIDATES, "layers.0.self_attn.q_proj.weight")
    pfx_proj = _pick_prefix(keys, PFX_PROJ_CANDIDATES, "action_in_proj.weight")

    n_layers_vlm = _maxlayer(f"{pfx_vlm}.layers.")
    n_layers_aex = _maxlayer(f"{pfx_aex}.layers.")
    if n_layers_vlm <= 0:
        raise SystemExit("cannot find PaliGemma language-model layers in checkpoint")
    if n_layers_aex != n_layers_vlm:
        raise SystemExit(f"layer count mismatch: VLM={n_layers_vlm} expert={n_layers_aex} "
                         f"(pi0.5 expects them equal)")
    cfg["n_layers"] = n_layers_vlm

    q0  = sf.get_slice(f"{pfx_vlm}.layers.0.self_attn.q_proj.weight").get_shape()
    kv0 = sf.get_slice(f"{pfx_vlm}.layers.0.self_attn.k_proj.weight").get_shape()
    gate0 = sf.get_slice(f"{pfx_vlm}.layers.0.mlp.gate_proj.weight").get_shape()
    if q0[1] != cfg["hidden"]:
        raise SystemExit(f"hidden mismatch: cfg={cfg['hidden']} ckpt={q0[1]}")
    if q0[0] != cfg["n_q_heads"] * cfg["head_dim"]:
        raise SystemExit(f"q_proj rows {q0[0]} != n_q_heads*head_dim {cfg['n_q_heads']*cfg['head_dim']}")
    if kv0[0] != cfg["n_kv_heads"] * cfg["head_dim"]:
        raise SystemExit(f"k_proj rows {kv0[0]} != n_kv_heads*head_dim {cfg['n_kv_heads']*cfg['head_dim']}")
    if gate0[0] != cfg["intermediate"]:
        raise SystemExit(f"intermediate mismatch: cfg={cfg['intermediate']} ckpt={gate0[0]}")

    aex_gate0 = sf.get_slice(f"{pfx_aex}.layers.0.mlp.gate_proj.weight").get_shape()
    if aex_gate0[1] != cfg["expert_h"]:
        raise SystemExit(f"expert_h mismatch: cfg={cfg['expert_h']} ckpt={aex_gate0[1]}")
    if aex_gate0[0] != cfg["expert_inter"]:
        raise SystemExit(f"expert_inter mismatch: cfg={cfg['expert_inter']} ckpt={aex_gate0[0]}")
    aex_o0 = sf.get_slice(f"{pfx_aex}.layers.0.self_attn.o_proj.weight").get_shape()
    if aex_o0 != [cfg["expert_h"], cfg["n_q_heads"] * cfg["head_dim"]]:
        raise SystemExit(f"expert o_proj shape {aex_o0} != [expert_h, n_q*head_dim] "
                         f"{[cfg['expert_h'], cfg['n_q_heads']*cfg['head_dim']]}")

    # AdaRMSNorm dense modulators must project expert_h -> 3*expert_h.
    adarms0 = sf.get_slice(f"{pfx_aex}.layers.0.input_layernorm.dense.weight").get_shape()
    if adarms0 != [3 * cfg["expert_h"], cfg["expert_h"]]:
        raise SystemExit(f"expert AdaRMS dense shape {adarms0} != "
                         f"{[3 * cfg['expert_h'], cfg['expert_h']]}")

    head_w = sf.get_slice(pfx_vlm_head).get_shape()
    if head_w[1] != cfg["hidden"]:
        raise SystemExit(f"lm_head hidden mismatch: cfg={cfg['hidden']} ckpt={head_w[1]}")
    cfg["vocab_size"] = int(head_w[0])

    print(f"resolved cfg: hidden={cfg['hidden']} n_layers={cfg['n_layers']} "
          f"inter={cfg['intermediate']} heads={cfg['n_q_heads']}q/{cfg['n_kv_heads']}kv×{cfg['head_dim']} "
          f"expert_h={cfg['expert_h']} expert_inter={cfg['expert_inter']} vocab={cfg['vocab_size']} "
          f"chunk={cfg['chunk_size']} steps={cfg['num_steps']} "
          f"real_state={cfg['real_state_dim']} real_action={cfg['real_action_dim']} "
          f"state_norm={cfg['state_norm_mode']} action_norm={cfg['action_norm_mode']} "
          f"norm_eps={cfg['norm_eps']:g}")

    print("loading normalizer stats...")
    stats = _load_stats(sf, ckpt, cfg["real_state_dim"], cfg["real_action_dim"],
                        cfg["state_norm_mode"], cfg["action_norm_mode"])

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out}")
    writer = gguf.GGUFWriter(str(out), arch=ARCH)
    _add_kv(writer, cfg)

    _add_one_tensor(writer, "token_embd.weight",      sf.get_tensor(pfx_vlm_head))
    _add_one_tensor(writer, "vlm.output_norm.weight", sf.get_tensor(f"{pfx_vlm}.norm.weight"))
    _stream_block(writer, sf, pfx_vlm, "vlm", cfg["n_layers"])

    _add_one_tensor(writer, "aex.output_norm.dense.weight", sf.get_tensor(f"{pfx_aex}.norm.dense.weight"))
    _add_one_tensor(writer, "aex.output_norm.dense.bias",   sf.get_tensor(f"{pfx_aex}.norm.dense.bias"))
    _stream_adarms_block(writer, sf, pfx_aex, "aex", cfg["n_layers"])

    # pi0.5 embeds state in text, so there is no state_proj in the runtime path.
    for suf in ["action_in_proj.weight", "action_in_proj.bias",
                "action_out_proj.weight", "action_out_proj.bias"]:
        _add_one_tensor(writer, suf, sf.get_tensor(_join_key(pfx_proj, suf)))
    for src, dst in [
        ("time_mlp_in.weight",  "time_mlp_in.weight"),
        ("time_mlp_in.bias",    "time_mlp_in.bias"),
        ("time_mlp_out.weight", "time_mlp_out.weight"),
        ("time_mlp_out.bias",   "time_mlp_out.bias"),
    ]:
        # Some converted LeRobot checkpoints keep the older pi0 key prefix;
        # normalize it here so C++ only has one set of tensor names to load.
        src_key = _join_key(pfx_proj, src)
        if src_key not in keys:
            src_key = _join_key(pfx_proj, src.replace('time_mlp_', 'action_time_mlp_'))
        _add_one_tensor(writer, dst, sf.get_tensor(src_key))

    for k, v in stats.items():
        writer.add_tensor(k, v, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"done. {out} ({out.stat().st_size / (1024*1024):.1f} MiB)")
    print("note: the SigLIP vision tower + multi_modal_projector are NOT in this file - "
          "produce the mmproj GGUF separately with the same PaliGemma-224 vision tower.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
