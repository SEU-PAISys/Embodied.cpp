#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Convert Hy-Embodied-0.5-VLA checkpoints to GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
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
except Exception as exc:  # pragma: no cover - user-facing startup error
    raise SystemExit(
        "failed to import gguf. Expected either an installed gguf package or "
        f"the vendored package at {GGUF_PY}. Original error: {exc}"
    )


ARCH = "hy_vla"
KV = lambda name: f"{ARCH}.{name}"

DEFAULT_CKPT = Path("/home/xuling/robotic_dataset/HY-VLA")
PFX_EXPERT = "model.dual_tower.expert.model"
PFX_VLM_TEXT = "model.dual_tower.vlm.model.language_model.model"
PFX_VISUAL = "model.dual_tower.vlm.model.visual"
PFX_PROJ = "model"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bf16_to_u16(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        print(f"  warn: casting non-BF16 tensor (dtype={t.dtype}) to BF16 for storage")
        t = t.to(torch.bfloat16)
    return t.contiguous().view(torch.uint16).cpu().numpy()


def _add_tensor(writer: "gguf.GGUFWriter", dst_name: str, t: torch.Tensor) -> None:
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


def _add_array_tensor(writer: "gguf.GGUFWriter", dst_name: str, value: Any, dim: int) -> None:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    padded = np.zeros(dim, dtype=np.float32)
    padded[: min(dim, arr.size)] = arr[:dim]
    writer.add_tensor(dst_name, padded, raw_dtype=gguf.GGMLQuantizationType.F32)


def _text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    text_cfg = cfg.get("vlm_config_dict", {}).get("text_config")
    if not isinstance(text_cfg, dict):
        raise SystemExit("config.json is missing vlm_config_dict.text_config")
    return text_cfg


def _infer_cfg(ckpt: Path, sf) -> dict[str, Any]:
    cfg_json = _load_json(ckpt / "config.json")
    text_cfg = _text_config(cfg_json)

    n_layers = int(text_cfg["num_hidden_layers"])
    expert_gate = sf.get_slice(f"{PFX_EXPERT}.layers.0.mlp.gate_proj.weight")
    expert_intermediate, expert_hidden = expert_gate.get_shape()

    cfg: dict[str, Any] = {
        "chunk_size": int(cfg_json["chunk_size"]),
        "n_action_steps": int(cfg_json["n_action_steps"]),
        "max_state_dim": int(cfg_json["max_state_dim"]),
        "max_action_dim": int(cfg_json["max_action_dim"]),
        "tokenizer_max_length": int(cfg_json["tokenizer_max_length"]),
        "proj_width": int(cfg_json["proj_width"]),
        "num_steps": int(cfg_json["num_steps"]),
        "use_cache": bool(cfg_json.get("use_cache", True)),
        "use_video_encoder": bool(cfg_json.get("use_video_encoder", False)),
        "spacetime_layer_stride": int(cfg_json.get("spacetime_layer_stride", 0) or 0),
        "visual_segment_isolation": bool(cfg_json.get("visual_segment_isolation", False)),
        "vlm_hidden": int(text_cfg["hidden_size"]),
        "vlm_intermediate": int(text_cfg["intermediate_size"]),
        "expert_hidden": int(expert_hidden),
        "expert_intermediate": int(expert_intermediate),
        "n_layers": n_layers,
        "n_q_heads": int(text_cfg["num_attention_heads"]),
        "n_kv_heads": int(text_cfg["num_key_value_heads"]),
        "head_dim": int(text_cfg.get("head_dim", text_cfg.get("attention_head_dim", 128))),
        "vocab_size": int(text_cfg["vocab_size"]),
        "rope_theta": float(text_cfg["rope_theta"]),
        "rope_dynamic_alpha": float(text_cfg.get("rope_scaling", {}).get("alpha", 1000.0)),
        "rms_norm_eps": float(text_cfg["rms_norm_eps"]),
        "flow_min_period": 4.0e-3,
        "flow_max_period": 4.0,
    }
    if cfg["expert_hidden"] != cfg["proj_width"]:
        raise SystemExit(f"expert hidden mismatch: expert={cfg['expert_hidden']} proj_width={cfg['proj_width']}")
    return cfg


def _add_metadata(writer: "gguf.GGUFWriter", cfg: dict[str, Any], scope: str) -> None:
    writer.add_string(KV("architecture"), ARCH)
    writer.add_string(KV("format_stage"), scope)
    writer.add_uint32(KV("chunk_size"), cfg["chunk_size"])
    writer.add_uint32(KV("n_action_steps"), cfg["n_action_steps"])
    writer.add_uint32(KV("max_state_dim"), cfg["max_state_dim"])
    writer.add_uint32(KV("max_action_dim"), cfg["max_action_dim"])
    writer.add_uint32(KV("tokenizer_max_length"), cfg["tokenizer_max_length"])
    writer.add_uint32(KV("proj_width"), cfg["proj_width"])
    writer.add_uint32(KV("num_steps"), cfg["num_steps"])
    writer.add_bool(KV("use_cache"), cfg["use_cache"])
    writer.add_bool(KV("use_video_encoder"), cfg["use_video_encoder"])
    writer.add_uint32(KV("spacetime_layer_stride"), cfg["spacetime_layer_stride"])
    writer.add_bool(KV("visual_segment_isolation"), cfg["visual_segment_isolation"])
    writer.add_uint32(KV("vlm_hidden"), cfg["vlm_hidden"])
    writer.add_uint32(KV("vlm_intermediate"), cfg["vlm_intermediate"])
    writer.add_uint32(KV("expert_hidden"), cfg["expert_hidden"])
    writer.add_uint32(KV("expert_intermediate"), cfg["expert_intermediate"])
    writer.add_uint32(KV("n_layers"), cfg["n_layers"])
    writer.add_uint32(KV("n_q_heads"), cfg["n_q_heads"])
    writer.add_uint32(KV("n_kv_heads"), cfg["n_kv_heads"])
    writer.add_uint32(KV("head_dim"), cfg["head_dim"])
    writer.add_uint32(KV("vocab_size"), cfg["vocab_size"])
    writer.add_float32(KV("rope_theta"), cfg["rope_theta"])
    writer.add_float32(KV("rope_dynamic_alpha"), cfg["rope_dynamic_alpha"])
    writer.add_float32(KV("rms_norm_eps"), cfg["rms_norm_eps"])
    writer.add_float32(KV("flow_min_period"), cfg["flow_min_period"])
    writer.add_float32(KV("flow_max_period"), cfg["flow_max_period"])


def _expert_tensor_map(n_layers: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [(f"{PFX_EXPERT}.norm.weight", "expert.output_norm.weight")]
    suffix_map = [
        ("input_layernorm.weight", "attn_norm.weight"),
        ("input_layernorm_v.weight", "attn_norm_v.weight"),
        ("self_attn.q_proj.weight", "attn_q.weight"),
        ("self_attn.k_proj.weight", "attn_k.weight"),
        ("self_attn.v_proj.weight", "attn_v.weight"),
        ("self_attn.o_proj.weight", "attn_o.weight"),
        ("self_attn.q_proj_v.weight", "attn_q_v.weight"),
        ("self_attn.k_proj_v.weight", "attn_k_v.weight"),
        ("self_attn.v_proj_v.weight", "attn_v_v.weight"),
        ("self_attn.o_proj_v.weight", "attn_o_v.weight"),
        ("self_attn.query_layernorm.weight", "attn_q_norm.weight"),
        ("self_attn.key_layernorm.weight", "attn_k_norm.weight"),
        ("post_attention_layernorm.weight", "ffn_norm.weight"),
        ("post_attention_layernorm_v.weight", "ffn_norm_v.weight"),
        ("mlp.gate_proj.weight", "ffn_gate.weight"),
        ("mlp.up_proj.weight", "ffn_up.weight"),
        ("mlp.down_proj.weight", "ffn_down.weight"),
        ("mlp_v.gate_proj.weight", "ffn_gate_v.weight"),
        ("mlp_v.up_proj.weight", "ffn_up_v.weight"),
        ("mlp_v.down_proj.weight", "ffn_down_v.weight"),
    ]
    for i in range(n_layers):
        for src_suf, dst_suf in suffix_map:
            rows.append((f"{PFX_EXPERT}.layers.{i}.{src_suf}", f"expert.blk.{i}.{dst_suf}"))
    return rows


def _vlm_text_tensor_map(n_layers: int) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        (f"{PFX_VLM_TEXT}.norm.weight", "vlm.output_norm.weight"),
    ]
    suffix_map = [
        ("input_layernorm.weight", "attn_norm.weight"),
        ("input_layernorm_v.weight", "attn_norm_v.weight"),
        ("self_attn.q_proj.weight", "attn_q.weight"),
        ("self_attn.k_proj.weight", "attn_k.weight"),
        ("self_attn.v_proj.weight", "attn_v.weight"),
        ("self_attn.o_proj.weight", "attn_o.weight"),
        ("self_attn.q_proj_v.weight", "attn_q_v.weight"),
        ("self_attn.k_proj_v.weight", "attn_k_v.weight"),
        ("self_attn.v_proj_v.weight", "attn_v_v.weight"),
        ("self_attn.o_proj_v.weight", "attn_o_v.weight"),
        ("self_attn.query_layernorm.weight", "attn_q_norm.weight"),
        ("self_attn.key_layernorm.weight", "attn_k_norm.weight"),
        ("post_attention_layernorm.weight", "ffn_norm.weight"),
        ("post_attention_layernorm_v.weight", "ffn_norm_v.weight"),
        ("mlp.gate_proj.weight", "ffn_gate.weight"),
        ("mlp.up_proj.weight", "ffn_up.weight"),
        ("mlp.down_proj.weight", "ffn_down.weight"),
        ("mlp_v.gate_proj.weight", "ffn_gate_v.weight"),
        ("mlp_v.up_proj.weight", "ffn_up_v.weight"),
        ("mlp_v.down_proj.weight", "ffn_down_v.weight"),
    ]
    for i in range(n_layers):
        for src_suf, dst_suf in suffix_map:
            rows.append((f"{PFX_VLM_TEXT}.layers.{i}.{src_suf}", f"vlm.blk.{i}.{dst_suf}"))
    return rows


def _projection_tensor_map() -> list[tuple[str, str]]:
    return [
        (f"{PFX_PROJ}.state_proj.weight", "state_proj.weight"),
        (f"{PFX_PROJ}.state_proj.bias", "state_proj.bias"),
        (f"{PFX_PROJ}.action_in_proj.weight", "action_in_proj.weight"),
        (f"{PFX_PROJ}.action_in_proj.bias", "action_in_proj.bias"),
        (f"{PFX_PROJ}.action_time_mlp_in.weight", "action_time_mlp_in.weight"),
        (f"{PFX_PROJ}.action_time_mlp_in.bias", "action_time_mlp_in.bias"),
        (f"{PFX_PROJ}.action_time_mlp_out.weight", "action_time_mlp_out.weight"),
        (f"{PFX_PROJ}.action_time_mlp_out.bias", "action_time_mlp_out.bias"),
        (f"{PFX_PROJ}.action_out_proj.weight", "action_out_proj.weight"),
        (f"{PFX_PROJ}.action_out_proj.bias", "action_out_proj.bias"),
    ]


def _vision_tensor_map(n_layers: int = 27) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        (f"{PFX_VISUAL}.vision_tower.patch_embed.proj.weight", "vision.patch_embed.weight"),
        (f"{PFX_VISUAL}.vision_tower.patch_embed.proj.bias", "vision.patch_embed.bias"),
        (f"{PFX_VISUAL}.vision_tower.pos_embed", "vision.pos_embed"),
        (f"{PFX_VISUAL}.merger.proj1.weight", "vision.merger.proj1.weight"),
        (f"{PFX_VISUAL}.merger.proj1.bias", "vision.merger.proj1.bias"),
        (f"{PFX_VISUAL}.merger.proj2.weight", "vision.merger.proj2.weight"),
        (f"{PFX_VISUAL}.merger.proj2.bias", "vision.merger.proj2.bias"),
        (f"{PFX_VISUAL}.merger.pooler.predictor.0.weight", "vision.merger.pooler.fc0.weight"),
        (f"{PFX_VISUAL}.merger.pooler.predictor.0.bias", "vision.merger.pooler.fc0.bias"),
        (f"{PFX_VISUAL}.merger.pooler.predictor.2.weight", "vision.merger.pooler.fc2.weight"),
        (f"{PFX_VISUAL}.merger.pooler.predictor.2.bias", "vision.merger.pooler.fc2.bias"),
    ]
    suffix_map = [
        ("norm1.weight", "norm1.weight"),
        ("norm1.bias", "norm1.bias"),
        ("attn.qkv.weight", "attn_qkv.weight"),
        ("attn.qkv.bias", "attn_qkv.bias"),
        ("attn.proj.weight", "attn_proj.weight"),
        ("attn.proj.bias", "attn_proj.bias"),
        ("norm2.weight", "norm2.weight"),
        ("norm2.bias", "norm2.bias"),
        ("mlp.fc1.weight", "ffn_fc1.weight"),
        ("mlp.fc1.bias", "ffn_fc1.bias"),
        ("mlp.fc2.weight", "ffn_fc2.weight"),
        ("mlp.fc2.bias", "ffn_fc2.bias"),
    ]
    for i in range(n_layers):
        for src_suf, dst_suf in suffix_map:
            rows.append((f"{PFX_VISUAL}.vision_tower.blocks.{i}.{src_suf}", f"vision.blk.{i}.{dst_suf}"))
    return rows


def _stripped_tensor_name(src: str) -> str:
    name = src.removeprefix("model.")
    if len(name.encode("utf-8")) < 64:
        return name
    parts = name.split(".")
    if "language_model" in parts:
        parts[parts.index("language_model")] = "lm"
    if "dual_tower" in parts:
        parts[parts.index("dual_tower")] = "dt"
    if "vision_tower" in parts:
        parts[parts.index("vision_tower")] = "vit"
    if "self_attn" in parts:
        parts[parts.index("self_attn")] = "attn"
    if "post_attention_layernorm" in parts:
        parts[parts.index("post_attention_layernorm")] = "post_ln"
    name = ".".join(parts)
    if len(name.encode("utf-8")) < 64:
        return name
    digest = hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]
    leaf = ".".join(parts[-4:])
    name = f"hy.{digest}.{leaf}"
    if len(name.encode("utf-8")) < 64:
        return name
    return f"hy.{digest}.{parts[-2]}.{parts[-1]}"


def _full_tensor_map(sf, short_rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    short_srcs = {src for src, _ in short_rows}
    rows = list(short_rows)
    for src in sorted(sf.keys()):
        if src in short_srcs:
            continue
        rows.append((src, _stripped_tensor_name(src)))
    return rows


def _validate_map(sf, rows: list[tuple[str, str]]) -> None:
    keys = set(sf.keys())
    missing = [src for src, _ in rows if src not in keys]
    dsts = [dst for _, dst in rows]
    dup_dst = sorted({dst for dst in dsts if dsts.count(dst) > 1})
    if missing or dup_dst:
        msg = [
            "tensor map validation failed:",
            f"  missing source tensors: {len(missing)}",
            f"  duplicate destination names: {len(dup_dst)}",
        ]
        for value in missing[:48]:
            msg.append(f"  missing: {value}")
        for value in dup_dst[:48]:
            msg.append(f"  duplicate dst: {value}")
        raise SystemExit("\n".join(msg))


def _add_norm_stats(writer: "gguf.GGUFWriter", ckpt: Path, cfg: dict[str, Any]) -> None:
    path = ckpt / "norm_stats.pkl"
    if not path.exists():
        return
    with path.open("rb") as fp:
        stats = pickle.load(fp)
    mapping = [
        ("qpos_mean", "state_mean", cfg["max_state_dim"]),
        ("qpos_std", "state_std", cfg["max_state_dim"]),
        ("action_mean", "action_mean", cfg["max_action_dim"]),
        ("action_std", "action_std", cfg["max_action_dim"]),
        ("action_mean_abs", "action_mean_abs", cfg["max_action_dim"]),
        ("action_std_abs", "action_std_abs", cfg["max_action_dim"]),
    ]
    for src, dst, dim in mapping:
        if src not in stats or stats[src] is None:
            continue
        arr = stats[src]
        if getattr(arr, "ndim", 1) > 1:
            # Keep the table form for action chunk stats; C++ can choose row semantics later.
            writer.add_tensor(f"norm.{dst}", np.asarray(arr, dtype=np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            _add_array_tensor(writer, f"norm.{dst}", arr, dim)


def _write_gguf(ckpt: Path, out: Path, scope: str, dry_run: bool) -> None:
    sf_path = ckpt / "model.safetensors"
    with safe_open(sf_path, framework="pt", device="cpu") as sf:
        cfg = _infer_cfg(ckpt, sf)
        rows: list[tuple[str, str]] = []
        if scope in {"action-expert", "full"}:
            rows.extend(_expert_tensor_map(cfg["n_layers"]))
            rows.extend(_projection_tensor_map())
        if scope == "full":
            rows.extend(_vlm_text_tensor_map(cfg["n_layers"]))
            rows.extend(_vision_tensor_map())
            rows = _full_tensor_map(sf, rows)
        _validate_map(sf, rows)
        src_names = [src for src, _ in rows]
        dst_names = [dst for _, dst in rows]
        if any(len(name.encode("utf-8")) >= 64 for name in dst_names):
            too_long = [name for name in dst_names if len(name.encode("utf-8")) >= 64]
            raise SystemExit(f"GGUF tensor names must be <64 bytes; first too long: {too_long[0]}")
        total_bytes = 0
        for src, _ in rows:
            meta = sf.get_slice(src)
            shape = list(meta.get_shape())
            dtype = str(meta.get_dtype())
            item_size = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}.get(dtype.upper(), 0)
            numel = 1
            for dim in shape:
                numel *= int(dim)
            total_bytes += numel * item_size
        print(
            f"scope={scope} tensors={len(rows)} estimated_tensor_bytes="
            f"{total_bytes / (1024 * 1024):.1f} MiB"
        )
        if dry_run:
            print("dry-run: tensor map validated; no GGUF was written")
            return

        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"writing {out}")
        writer = gguf.GGUFWriter(str(out), arch=ARCH)
        _add_metadata(writer, cfg, scope)
        _add_norm_stats(writer, ckpt, cfg)
        writer.add_uint32(KV("written_tensor_count"), len(rows))
        writer.add_array(KV("written_tensor_names"), [dst for _, dst in rows])
        if scope == "full":
            writer.add_array(KV("source_tensor_names"), src_names)

        for src, dst in rows:
            print(f"  {src} -> {dst}")
            _add_tensor(writer, dst, sf.get_tensor(src))

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    print(f"done. {out} ({out.stat().st_size / (1024 * 1024):.1f} MiB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--scope",
        choices=["metadata", "action-expert", "full"],
        default="metadata",
        help="metadata writes config/stats; action-expert writes expert/projections; full writes every checkpoint tensor",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate tensor mapping without writing GGUF")
    args = parser.parse_args()

    ckpt = args.ckpt.expanduser().resolve()
    if not (ckpt / "config.json").exists():
        raise SystemExit(f"missing {ckpt / 'config.json'}")
    if not (ckpt / "model.safetensors").exists():
        raise SystemExit(f"missing {ckpt / 'model.safetensors'}")

    out = args.out
    if out is None:
        suffix = {"metadata": "metadata", "action-expert": "action_expert", "full": "full"}[args.scope]
        out = ckpt / f"hy_vla_{suffix}_bf16.gguf"
    _write_gguf(ckpt, out.expanduser().resolve(), args.scope, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
