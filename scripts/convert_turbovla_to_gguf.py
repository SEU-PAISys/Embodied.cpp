#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Convert TurboVLA checkpoints (LIBERO / RoboTwin suites) to GGUF.

TurboVLA = DINOv3 ViT vision tower (2D-RoPE, 4 register tokens)
+ BERT text encoder + 6-layer bidirectional cross-attention fusion
+ ACT-style transformer-decoder action head.

The GGUF bundles every tensor plus the LIBERO normalization statistics
(proprio mean/std, action min/max) so the C++ runtime is self-contained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if GGUF_PY.exists():
    sys.path.insert(0, str(GGUF_PY))

try:
    import gguf
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"failed to import gguf (vendored at {GGUF_PY}): {exc}")

ARCH = "turbovla"
KV = lambda name: f"{ARCH}.{name}"

# LIBERO evaluation constants from official turbovla/evaluation/policy.py.
PROPRIO_MEAN = [-0.04190646484494209, 0.03539437800645828, 0.8257066607475281,
                2.908315658569336, -0.5562158823013306, -0.16649103164672852,
                0.02831534668803215, -0.028561558574438095]
PROPRIO_STD = [0.10743443667888641, 0.14424759149551392, 0.25723373889923096,
               0.34413808584213257, 1.234430193901062, 0.35798805952072144,
               0.013308786787092686, 0.013174591585993767]
ACTION_MIN = [-0.9375, -0.9375, -0.9375, -0.23642857372760773,
              -0.3053571283817291, -0.3675000071525574, -1.0]
ACTION_MAX = [0.9375, 0.9375, 0.9375, 0.30000001192092896,
              0.29357144236564636, 0.375, 1.0]


def load_norm_stats(stats_json: Path, stats_key: str | None):
    import json

    payload = json.loads(stats_json.read_text(encoding="utf-8"))
    if stats_key is None:
        keys = [k for k in payload if k != "metadata"]
        if len(keys) != 1:
            raise SystemExit(f"--stats-key required; available keys: {keys}")
        stats_key = keys[0]
    if stats_key not in payload:
        raise SystemExit(f"stats key {stats_key!r} not found in {stats_json}")
    stats = payload[stats_key]
    state_section = "proprio" if "proprio" in stats else "state"
    return (
        [float(v) for v in stats[state_section]["mean"]],
        [float(v) for v in stats[state_section]["std"]],
        [float(v) for v in stats["action"]["min"]],
        [float(v) for v in stats["action"]["max"]],
        stats_key,
    )


def load_checkpoint(path: Path):
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"unexpected checkpoint payload type {type(ckpt)}")
    model_config = ckpt.get("model_config")
    if not isinstance(model_config, dict):
        model_config = None
    sd = ckpt.get("model_state_dict")
    if not isinstance(sd, dict) or not sd:
        for key in ("state_dict", "model", "module", "ema"):
            candidate = ckpt.get(key)
            if isinstance(candidate, dict) and candidate and torch.is_tensor(next(iter(candidate.values()))):
                sd = candidate
                break
    if sd is None:
        raise SystemExit(f"no model_state_dict found inside {path}")
    cleaned = {}
    for key, value in sd.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned, model_config, ckpt


def infer_cfg(sd, model_config):
    def count(prefix, index_at):
        best = -1
        for key in sd:
            if key.startswith(prefix):
                try:
                    best = max(best, int(key.split(".")[index_at]))
                except (ValueError, IndexError):
                    pass
        return best + 1

    cfg = {
        "hidden_dim": 256,
        "nheads": 8,
        "interaction_layers": count("vision_language_interaction.fusion_layers.", 2),
        "enhancer_inner_dim": int(sd["vision_language_interaction.text_layers.0.linear1.weight"].shape[0]),
        "act_ffn_dim": int(sd["action_head.decoder.decoder.layers.0.linear1.weight"].shape[0]),
        "action_dim": int(sd["action_head.decoder.action_projection.layers.2.weight"].shape[0]),
        "state_dim": int(sd["action_head.state_projection.net.1.weight"].shape[1]),
        "horizon": int(sd["action_head.decoder.action_queries.weight"].shape[0]),
        "num_state_tokens": int(sd["action_head.state_projection.position"].shape[1]),
        "act_layers": count("action_head.decoder.decoder.layers.", 4),
        "mlp_hidden_dim": int(sd["action_head.decoder.action_projection.layers.1.weight"].shape[0]),
        "state_hidden_dim": int(sd["action_head.state_projection.net.1.weight"].shape[0]),
        "num_views": int(sd["view_embedding"].shape[1]),
        "image_size": 256,
        "text_padding_length": 21,
        "max_text_len": 256,
        "bert_layers": count("text_encoder.bert.encoder.layer.", 4),
        "bert_hidden": int(sd["text_encoder.bert.embeddings.word_embeddings.weight"].shape[1]),
        "bert_intermediate": int(sd["text_encoder.bert.encoder.layer.0.intermediate.dense.weight"].shape[0]),
        "bert_vocab": int(sd["text_encoder.bert.embeddings.word_embeddings.weight"].shape[0]),
        "vit_layers": count("vision_encoder.backbone.layer.", 3),
        "vit_hidden": int(sd["vision_encoder.backbone.embeddings.cls_token"].shape[-1]),
        "vit_intermediate": int(sd["vision_encoder.backbone.layer.0.mlp.up_proj.weight"].shape[0]),
        "patch_size": int(sd["vision_encoder.backbone.embeddings.patch_embeddings.weight"].shape[-1]),
        "num_register_tokens": int(sd["vision_encoder.backbone.embeddings.register_tokens"].shape[1]),
        "rope_theta": 100.0,
    }
    if model_config is not None:
        text_cfg = model_config.get("text", {})
        if isinstance(text_cfg.get("padding_length"), int):
            cfg["text_padding_length"] = int(text_cfg["padding_length"])
        if isinstance(text_cfg.get("max_length"), int):
            cfg["max_text_len"] = int(text_cfg["max_length"])
        vision_cfg = model_config.get("vision", {})
        if isinstance(vision_cfg.get("image_size"), int):
            cfg["image_size"] = int(vision_cfg["image_size"])
    return cfg


# GGUF tensor map: (source state_dict key, destination GGUF name, squeeze leading 1s)
def build_tensor_map(cfg):
    rows = []  # list[tuple[str, str]]
    B = cfg["bert_layers"]
    V = cfg["vit_layers"]
    F = cfg["interaction_layers"]
    A = cfg["act_layers"]

    # --- BERT text encoder ---
    rows += [
        ("text_encoder.bert.embeddings.word_embeddings.weight", "text.token_emb"),
        ("text_encoder.bert.embeddings.position_embeddings.weight", "text.pos_emb"),
        ("text_encoder.bert.embeddings.token_type_embeddings.weight", "text.tok_type_emb"),
        ("text_encoder.bert.embeddings.LayerNorm.weight", "text.emb_norm.w"),
        ("text_encoder.bert.embeddings.LayerNorm.bias", "text.emb_norm.b"),
        ("text_encoder.text_projection.weight", "text.proj.w"),
        ("text_encoder.text_projection.bias", "text.proj.b"),
    ]
    for i in range(B):
        src = f"text_encoder.bert.encoder.layer.{i}"
        dst = f"text.blk.{i}"
        rows += [
            (f"{src}.attention.self.query.weight", f"{dst}.attn_q.w"),
            (f"{src}.attention.self.query.bias", f"{dst}.attn_q.b"),
            (f"{src}.attention.self.key.weight", f"{dst}.attn_k.w"),
            (f"{src}.attention.self.key.bias", f"{dst}.attn_k.b"),
            (f"{src}.attention.self.value.weight", f"{dst}.attn_v.w"),
            (f"{src}.attention.self.value.bias", f"{dst}.attn_v.b"),
            (f"{src}.attention.output.dense.weight", f"{dst}.attn_o.w"),
            (f"{src}.attention.output.dense.bias", f"{dst}.attn_o.b"),
            (f"{src}.attention.output.LayerNorm.weight", f"{dst}.attn_norm.w"),
            (f"{src}.attention.output.LayerNorm.bias", f"{dst}.attn_norm.b"),
            (f"{src}.intermediate.dense.weight", f"{dst}.ffn_up.w"),
            (f"{src}.intermediate.dense.bias", f"{dst}.ffn_up.b"),
            (f"{src}.output.dense.weight", f"{dst}.ffn_down.w"),
            (f"{src}.output.dense.bias", f"{dst}.ffn_down.b"),
            (f"{src}.output.LayerNorm.weight", f"{dst}.ffn_norm.w"),
            (f"{src}.output.LayerNorm.bias", f"{dst}.ffn_norm.b"),
        ]

    # --- DINOv3 ViT tower (2D RoPE, no absolute pos table) ---
    rows += [
        ("vision_encoder.backbone.embeddings.patch_embeddings.weight", "vit.patch_conv.w"),
        ("vision_encoder.backbone.embeddings.patch_embeddings.bias", "vit.patch_conv.b"),
        ("vision_encoder.backbone.embeddings.cls_token", "vit.cls_token"),
        ("vision_encoder.backbone.embeddings.register_tokens", "vit.reg_tokens"),
        ("vision_encoder.backbone.norm.weight", "vit.output_norm.w"),
        ("vision_encoder.backbone.norm.bias", "vit.output_norm.b"),
    ]
    for i in range(V):
        src = f"vision_encoder.backbone.layer.{i}"
        dst = f"vit.blk.{i}"
        rows += [
            (f"{src}.norm1.weight", f"{dst}.attn_norm.w"),
            (f"{src}.norm1.bias", f"{dst}.attn_norm.b"),
            (f"{src}.attention.q_proj.weight", f"{dst}.attn_q.w"),
            (f"{src}.attention.q_proj.bias", f"{dst}.attn_q.b"),
            (f"{src}.attention.k_proj.weight", f"{dst}.attn_k.w"),
            (f"{src}.attention.v_proj.weight", f"{dst}.attn_v.w"),
            (f"{src}.attention.v_proj.bias", f"{dst}.attn_v.b"),
            (f"{src}.attention.o_proj.weight", f"{dst}.attn_o.w"),
            (f"{src}.attention.o_proj.bias", f"{dst}.attn_o.b"),
            (f"{src}.layer_scale1.lambda1", f"{dst}.ls1"),
            (f"{src}.norm2.weight", f"{dst}.ffn_norm.w"),
            (f"{src}.norm2.bias", f"{dst}.ffn_norm.b"),
            (f"{src}.mlp.up_proj.weight", f"{dst}.ffn_up.w"),
            (f"{src}.mlp.up_proj.bias", f"{dst}.ffn_up.b"),
            (f"{src}.mlp.down_proj.weight", f"{dst}.ffn_down.w"),
            (f"{src}.mlp.down_proj.bias", f"{dst}.ffn_down.b"),
            (f"{src}.layer_scale2.lambda1", f"{dst}.ls2"),
        ]

    # --- vision projection + view embedding ---
    rows += [
        ("vision_projection.input_norm.weight", "vproj.input_norm.w"),
        ("vision_projection.input_norm.bias", "vproj.input_norm.b"),
        ("vision_projection.mlp.0.weight", "vproj.mlp_fc1.w"),
        ("vision_projection.mlp.0.bias", "vproj.mlp_fc1.b"),
        ("vision_projection.mlp.3.weight", "vproj.mlp_fc2.w"),
        ("vision_projection.mlp.3.bias", "vproj.mlp_fc2.b"),
        ("vision_projection.skip.weight", "vproj.skip_w"),
        ("vision_projection.output_norm.weight", "vproj.output_norm.w"),
        ("vision_projection.output_norm.bias", "vproj.output_norm.b"),
        ("view_embedding", "pos.view_emb"),
    ]

    # --- bidirectional fusion layers ---
    for i in range(F):
        src = f"vision_language_interaction.fusion_layers.{i}"
        dst = f"fuse.{i}"
        rows += [
            (f"{src}.layer_norm_v.weight", f"{dst}.ln_v.w"),
            (f"{src}.layer_norm_v.bias", f"{dst}.ln_v.b"),
            (f"{src}.layer_norm_l.weight", f"{dst}.ln_l.w"),
            (f"{src}.layer_norm_l.bias", f"{dst}.ln_l.b"),
            (f"{src}.attn.v_proj.weight", f"{dst}.q_from_v.w"),
            (f"{src}.attn.v_proj.bias", f"{dst}.q_from_v.b"),
            (f"{src}.attn.l_proj.weight", f"{dst}.k_from_l.w"),
            (f"{src}.attn.l_proj.bias", f"{dst}.k_from_l.b"),
            (f"{src}.attn.values_v_proj.weight", f"{dst}.val_from_v.w"),
            (f"{src}.attn.values_v_proj.bias", f"{dst}.val_from_v.b"),
            (f"{src}.attn.values_l_proj.weight", f"{dst}.val_from_l.w"),
            (f"{src}.attn.values_l_proj.bias", f"{dst}.val_from_l.b"),
            (f"{src}.attn.out_v_proj.weight", f"{dst}.out_v.w"),
            (f"{src}.attn.out_v_proj.bias", f"{dst}.out_v.b"),
            (f"{src}.attn.out_l_proj.weight", f"{dst}.out_l.w"),
            (f"{src}.attn.out_l_proj.bias", f"{dst}.out_l.b"),
            (f"{src}.gamma_v", f"{dst}.gamma_v"),
            (f"{src}.gamma_l", f"{dst}.gamma_l"),
        ]

    # --- interaction text self-attention layers (post-LN) ---
    for i in range(F):
        src = f"vision_language_interaction.text_layers.{i}"
        dst = f"tenh.{i}"
        rows += [
            (f"{src}.self_attn.in_proj_weight", f"{dst}.attn_qkv_w"),
            (f"{src}.self_attn.in_proj_bias", f"{dst}.attn_qkv_b"),
            (f"{src}.self_attn.out_proj.weight", f"{dst}.attn_o.w"),
            (f"{src}.self_attn.out_proj.bias", f"{dst}.attn_o.b"),
            (f"{src}.linear1.weight", f"{dst}.ffn_up.w"),
            (f"{src}.linear1.bias", f"{dst}.ffn_up.b"),
            (f"{src}.linear2.weight", f"{dst}.ffn_down.w"),
            (f"{src}.linear2.bias", f"{dst}.ffn_down.b"),
            (f"{src}.norm1.weight", f"{dst}.attn_norm.w"),
            (f"{src}.norm1.bias", f"{dst}.attn_norm.b"),
            (f"{src}.norm2.weight", f"{dst}.ffn_norm.w"),
            (f"{src}.norm2.bias", f"{dst}.ffn_norm.b"),
        ]

    # --- action head: state projection ---
    rows += [
        ("action_head.state_projection.net.0.weight", "act.state_ln.w"),
        ("action_head.state_projection.net.0.bias", "act.state_ln.b"),
        ("action_head.state_projection.net.1.weight", "act.state_fc1.w"),
        ("action_head.state_projection.net.1.bias", "act.state_fc1.b"),
        ("action_head.state_projection.net.4.weight", "act.state_fc2.w"),
        ("action_head.state_projection.net.4.bias", "act.state_fc2.b"),
        ("action_head.state_projection.position", "act.state_pos"),
        ("action_head.state_projection.output_norm.weight", "act.state_out_norm.w"),
        ("action_head.state_projection.output_norm.bias", "act.state_out_norm.b"),
        ("action_head.decoder.action_queries.weight", "act.queries"),
    ]
    for i in range(A):
        src = f"action_head.decoder.decoder.layers.{i}"
        dst = f"act.blk.{i}"
        rows += [
            (f"{src}.self_attn.in_proj_weight", f"{dst}.self_qkv_w"),
            (f"{src}.self_attn.in_proj_bias", f"{dst}.self_qkv_b"),
            (f"{src}.self_attn.out_proj.weight", f"{dst}.self_o.w"),
            (f"{src}.self_attn.out_proj.bias", f"{dst}.self_o.b"),
            (f"{src}.multihead_attn.in_proj_weight", f"{dst}.cross_qkv_w"),
            (f"{src}.multihead_attn.in_proj_bias", f"{dst}.cross_qkv_b"),
            (f"{src}.multihead_attn.out_proj.weight", f"{dst}.cross_o.w"),
            (f"{src}.multihead_attn.out_proj.bias", f"{dst}.cross_o.b"),
            (f"{src}.linear1.weight", f"{dst}.ffn_up.w"),
            (f"{src}.linear1.bias", f"{dst}.ffn_up.b"),
            (f"{src}.linear2.weight", f"{dst}.ffn_down.w"),
            (f"{src}.linear2.bias", f"{dst}.ffn_down.b"),
            (f"{src}.norm1.weight", f"{dst}.norm1.w"),
            (f"{src}.norm1.bias", f"{dst}.norm1.b"),
            (f"{src}.norm2.weight", f"{dst}.norm2.w"),
            (f"{src}.norm2.bias", f"{dst}.norm2.b"),
            (f"{src}.norm3.weight", f"{dst}.norm3.w"),
            (f"{src}.norm3.bias", f"{dst}.norm3.b"),
        ]
    rows += [
        ("action_head.decoder.action_projection.layers.0.weight", "act.head_fc1.w"),
        ("action_head.decoder.action_projection.layers.0.bias", "act.head_fc1.b"),
        ("action_head.decoder.action_projection.layers.1.weight", "act.head_fc2.w"),
        ("action_head.decoder.action_projection.layers.1.bias", "act.head_fc2.b"),
        ("action_head.decoder.action_projection.layers.2.weight", "act.head_fc3.w"),
        ("action_head.decoder.action_projection.layers.2.bias", "act.head_fc3.b"),
    ]
    return rows


# Tensors stored in F32 (small norms / gains / embeddings of token scale).
F32_FORCED = {"vit.cls_token", "vit.reg_tokens", "pos.view_emb",
              "act.state_pos", "act.queries"}
F32_PREFIXES = ("norm.", "text.emb_norm", "text.proj.b", "text.blk.",
                "vit.blk.", "vproj.", "fuse.", "tenh.", "act.state_ln",
                "act.state_out_norm", "act.blk.", "vit.output_norm")


def _force_f32(name: str, tensor: torch.Tensor) -> bool:
    if name in F32_FORCED:
        return True
    for prefix in F32_PREFIXES:
        if name.startswith(prefix):
            # weights under these prefixes stay BF16 unless tiny
            return False
    return tensor.numel() <= 2048


def add_metadata(writer, cfg, suite, extra):
    writer.add_string(KV("architecture"), ARCH)
    writer.add_string(KV("format_stage"), "full")
    if suite:
        writer.add_string(KV("suite"), suite)
    writer.add_uint32(KV("hidden_dim"), cfg["hidden_dim"])
    writer.add_uint32(KV("nheads"), cfg["nheads"])
    writer.add_uint32(KV("interaction_layers"), cfg["interaction_layers"])
    writer.add_uint32(KV("enhancer_inner_dim"), cfg["enhancer_inner_dim"])
    writer.add_uint32(KV("act_ffn_dim"), cfg["act_ffn_dim"])
    writer.add_uint32(KV("action_dim"), cfg["action_dim"])
    writer.add_uint32(KV("state_dim"), cfg["state_dim"])
    writer.add_uint32(KV("horizon"), cfg["horizon"])
    writer.add_uint32(KV("num_state_tokens"), cfg["num_state_tokens"])
    writer.add_uint32(KV("act_layers"), cfg["act_layers"])
    writer.add_uint32(KV("mlp_hidden_dim"), cfg["mlp_hidden_dim"])
    writer.add_uint32(KV("state_hidden_dim"), cfg["state_hidden_dim"])
    writer.add_uint32(KV("num_views"), cfg["num_views"])
    writer.add_uint32(KV("image_size"), cfg["image_size"])
    writer.add_uint32(KV("text_padding_length"), cfg["text_padding_length"])
    writer.add_uint32(KV("max_text_len"), cfg["max_text_len"])
    writer.add_uint32(KV("bert_layers"), cfg["bert_layers"])
    writer.add_uint32(KV("bert_hidden"), cfg["bert_hidden"])
    writer.add_uint32(KV("bert_intermediate"), cfg["bert_intermediate"])
    writer.add_uint32(KV("bert_vocab"), cfg["bert_vocab"])
    writer.add_uint32(KV("vit_layers"), cfg["vit_layers"])
    writer.add_uint32(KV("vit_hidden"), cfg["vit_hidden"])
    writer.add_uint32(KV("vit_intermediate"), cfg["vit_intermediate"])
    writer.add_uint32(KV("patch_size"), cfg["patch_size"])
    writer.add_uint32(KV("vit_heads"), max(1, cfg["vit_hidden"] // 64))
    writer.add_uint32(KV("num_register_tokens"), cfg["num_register_tokens"])
    writer.add_float32(KV("rope_theta"), cfg["rope_theta"])
    writer.add_float32(KV("state_norm_eps"), 1e-6)
    writer.add_array(KV("image_mean"), [0.485, 0.456, 0.406])
    writer.add_array(KV("image_std"), [0.229, 0.224, 0.225])
    for key, value in extra.items():
        writer.add_string(KV(key), str(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="TurboVLA .pth checkpoint")
    parser.add_argument("--out", type=Path, default=None, help="output GGUF path")
    parser.add_argument("--vocab", type=Path, default=None,
                        help="BERT vocab.txt (bundled into GGUF for the C++ WordPiece tokenizer). "
                             "Required unless --dry-run is used: the GGUF is useless without it "
                             "because the C++ runtime refuses to load a missing bert_vocab_list.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats-json", type=Path, default=None,
                        help="Normalization-stats JSON overriding the baked-in policy.py constants, "
                             "e.g. TurboVLA-official experiments/libero/configs/libero_all4_stats.json "
                             "(the statistics the released LIBERO checkpoints were trained with).")
    parser.add_argument("--stats-key", type=str, default=None,
                        help="Entry key inside --stats-json when it holds multiple suites, "
                             "e.g. libero_all4_no_noops.")
    args = parser.parse_args()

    ckpt_path = args.ckpt.expanduser().resolve()
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    sd, model_config, ckpt = load_checkpoint(ckpt_path)
    cfg = infer_cfg(sd, model_config)
    suite = ckpt.get("suite") if isinstance(ckpt, dict) else None
    print(f"suite={suite} tensors={len(sd)}")

    pad_layout = {}
    if isinstance(model_config, dict):
        text_cfg = model_config.get("text", {})
        raw_layout = text_cfg.get("padding_length_by_instruction")
        if isinstance(raw_layout, dict):
            pad_layout = {str(k): int(v) for k, v in raw_layout.items()}
            print(f"pad_layout entries={len(pad_layout)}")

    global PROPRIO_MEAN, PROPRIO_STD, ACTION_MIN, ACTION_MAX
    stats_source = "turbovla/evaluation/policy.py constants"
    if args.stats_json:
        stats_json = args.stats_json.expanduser().resolve()
        (PROPRIO_MEAN, PROPRIO_STD, ACTION_MIN, ACTION_MAX, stats_key) = load_norm_stats(
            stats_json, args.stats_key)
        stats_source = f"{stats_json}#{stats_key}"
        print(f"norm stats override: {stats_source}")
        print(f"  proprio_mean={PROPRIO_MEAN}")
        print(f"  proprio_std={PROPRIO_STD}")
        print(f"  action_min={ACTION_MIN}")
        print(f"  action_max={ACTION_MAX}")
    for key in sorted(cfg):
        print(f"  {key} = {cfg[key]}")

    rows = build_tensor_map(cfg)
    keys = set(sd.keys())
    missing = [src for src, _ in rows if src not in keys]
    unused = sorted(keys - {src for src, _ in rows} - {
        "text_encoder.bert.pooler.dense.weight", "text_encoder.bert.pooler.dense.bias",
        "vision_encoder.backbone.embeddings.mask_token",
    })
    dsts = [dst for _, dst in rows]
    dupes = sorted({d for d in dsts if dsts.count(d) > 1})
    if missing or dupes:
        raise SystemExit(f"tensor map broken: missing={missing[:8]} dupes={dupes[:8]}")
    if unused:
        print(f"note: {len(unused)} tensors left unmapped (skipped):")
        for name in unused[:12]:
            print(f"  skip {name}")

    total = sum(sd[src].numel() for src, _ in rows)
    print(f"mapped tensors={len(rows)} params={total/1e6:.1f}M")
    if args.dry_run:
        print("dry-run: map validated, no GGUF written")
        return 0

    out = args.out or ckpt_path.with_name(f"turbovla_{suite or 'model'}_bf16.gguf")
    print(f"writing {out}")
    writer = gguf.GGUFWriter(str(out), arch=ARCH)
    add_metadata(writer, cfg, suite, {"source": ckpt_path.name,
                                      "norm_stats_source": stats_source})

    writer.add_tensor("norm.proprio_mean", np.asarray(PROPRIO_MEAN, dtype=np.float32),
                      raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.add_tensor("norm.proprio_std", np.asarray(PROPRIO_STD, dtype=np.float32),
                      raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.add_tensor("norm.action_min", np.asarray(ACTION_MIN, dtype=np.float32),
                      raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.add_tensor("norm.action_max", np.asarray(ACTION_MAX, dtype=np.float32),
                      raw_dtype=gguf.GGMLQuantizationType.F32)
    if pad_layout:
        writer.add_array(KV("pad_layout_instr"), list(pad_layout.keys()))
        writer.add_array(KV("pad_layout_len"), [int(v) for v in pad_layout.values()])

    for src, dst in rows:
        tensor = sd[src].detach().cpu().contiguous()
        if tensor.ndim >= 2 and tensor.shape[0] == 1 and "conv" not in dst:
            tensor = tensor.squeeze(0)
        if _force_f32(dst, tensor):
            writer.add_tensor(dst, tensor.to(torch.float32).numpy(),
                              raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            bf16 = tensor.to(torch.bfloat16).contiguous()
            writer.add_tensor(dst, bf16.view(torch.uint16).numpy(),
                              raw_shape=list(tensor.shape),
                              raw_dtype=gguf.GGMLQuantizationType.BF16)

    if args.vocab is None:
        if args.dry_run:
            print("note: --vocab skipped in dry-run (no GGUF written)")
        else:
            raise SystemExit(
                "missing required --vocab <bert vocab.txt>. The GGUF is not loadable "
                "without the bundled WordPiece vocab (C++ requires turbovla.bert_vocab_list).")
    if args.vocab is not None:
        vocab_path = args.vocab.expanduser().resolve()
        vocab = [line.rstrip("\n") for line in vocab_path.read_text(encoding="utf-8").splitlines()]
        if len(vocab) != cfg["bert_vocab"]:
            raise SystemExit(f"vocab size mismatch: file={len(vocab)} expected={cfg['bert_vocab']}")
        writer.add_array(KV("bert_vocab_list"), vocab)
        print(f"bundled WordPiece vocab: {len(vocab)} entries")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"done. {out} ({out.stat().st_size / (1024*1024):.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
