#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Dump LingBot-VA config-derived shapes and tentative GGUF tensor names."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_HF_SNAPSHOT = Path(
    "/home/xuling/.cache/huggingface/hub/"
    "models--robbyant--lingbot-va-posttrain-libero-long/"
    "snapshots/0e89d1e753019988aba484e8da2dc0810e264d9f"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tensor_map(num_layers: int) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = [
        ("patch_embedding_mlp.weight", "wvm.patch_embd_mlp.weight", "latent patch projection; used by forward"),
        ("patch_embedding_mlp.bias", "wvm.patch_embd_mlp.bias", "latent patch projection bias"),
        ("patch_embedding.weight", "wvm.patch_embd_legacy.weight", "present in checkpoint; usage TBD"),
        ("patch_embedding.bias", "wvm.patch_embd_legacy.bias", "present in checkpoint; usage TBD"),
        ("action_embedder.weight", "wvm.action_embd.weight", "action latent projection"),
        ("action_embedder.bias", "wvm.action_embd.bias", "action latent projection bias"),
        ("proj_out.weight", "wvm.output_proj.weight", "latent output projection"),
        ("proj_out.bias", "wvm.output_proj.bias", "latent output projection bias"),
        ("action_proj_out.weight", "wvm.action_out.weight", "action output projection"),
        ("action_proj_out.bias", "wvm.action_out.bias", "action output projection bias"),
        ("scale_shift_table", "wvm.output_scale_shift", "final adaptive norm table"),
    ]
    for src_prefix, dst_prefix, note in (
        ("condition_embedder", "wvm.cond", "video/latent path timestep and text projection"),
        ("condition_embedder_action", "wvm.action_cond", "action path timestep and text projection"),
    ):
        rows.extend([
            (f"{src_prefix}.text_embedder.linear_1.weight", f"{dst_prefix}.text_l1.weight", note),
            (f"{src_prefix}.text_embedder.linear_1.bias", f"{dst_prefix}.text_l1.bias", ""),
            (f"{src_prefix}.text_embedder.linear_2.weight", f"{dst_prefix}.text_l2.weight", ""),
            (f"{src_prefix}.text_embedder.linear_2.bias", f"{dst_prefix}.text_l2.bias", ""),
            (f"{src_prefix}.time_embedder.linear_1.weight", f"{dst_prefix}.time_l1.weight", ""),
            (f"{src_prefix}.time_embedder.linear_1.bias", f"{dst_prefix}.time_l1.bias", ""),
            (f"{src_prefix}.time_embedder.linear_2.weight", f"{dst_prefix}.time_l2.weight", ""),
            (f"{src_prefix}.time_embedder.linear_2.bias", f"{dst_prefix}.time_l2.bias", ""),
            (f"{src_prefix}.time_proj.weight", f"{dst_prefix}.time_proj.weight", ""),
            (f"{src_prefix}.time_proj.bias", f"{dst_prefix}.time_proj.bias", ""),
        ])
    for i in range(num_layers):
        p = f"blocks.{i}"
        d = f"wvm.blk.{i}"
        rows.extend([
            (f"{p}.scale_shift_table", f"{d}.scale_shift", "block adaptive norm table"),
            (f"{p}.norm2.weight", f"{d}.cross_norm.weight", "cross-attn norm weight"),
            (f"{p}.norm2.bias", f"{d}.cross_norm.bias", "cross-attn norm bias"),
        ])
        for attn in ("attn1", "attn2"):
            dst_attn = "self_attn" if attn == "attn1" else "cross_attn"
            rows.extend([
                (f"{p}.{attn}.to_q.weight", f"{d}.{dst_attn}.q.weight", ""),
                (f"{p}.{attn}.to_q.bias", f"{d}.{dst_attn}.q.bias", ""),
                (f"{p}.{attn}.to_k.weight", f"{d}.{dst_attn}.k.weight", ""),
                (f"{p}.{attn}.to_k.bias", f"{d}.{dst_attn}.k.bias", ""),
                (f"{p}.{attn}.to_v.weight", f"{d}.{dst_attn}.v.weight", ""),
                (f"{p}.{attn}.to_v.bias", f"{d}.{dst_attn}.v.bias", ""),
                (f"{p}.{attn}.to_out.0.weight", f"{d}.{dst_attn}.o.weight", ""),
                (f"{p}.{attn}.to_out.0.bias", f"{d}.{dst_attn}.o.bias", ""),
                (f"{p}.{attn}.norm_q.weight", f"{d}.{dst_attn}.q_norm.weight", ""),
                (f"{p}.{attn}.norm_k.weight", f"{d}.{dst_attn}.k_norm.weight", ""),
            ])
        rows.extend([
            (f"{p}.ffn.net.0.proj.weight", f"{d}.ffn_up.weight", "FeedForward first projection"),
            (f"{p}.ffn.net.0.proj.bias", f"{d}.ffn_up.bias", ""),
            (f"{p}.ffn.net.2.weight", f"{d}.ffn_down.weight", "FeedForward second projection"),
            (f"{p}.ffn.net.2.bias", f"{d}.ffn_down.bias", ""),
        ])
    return rows


def dump_spec(root: Path) -> int:
    root = root.expanduser().resolve()
    cfg = _load_json(root / "transformer" / "config.json")
    idx = _load_json(root / "transformer" / "diffusion_pytorch_model.safetensors.index.json")
    weight_map = idx.get("weight_map", {})

    n_layers = int(cfg["num_layers"])
    n_heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg["attention_head_dim"])
    inner = n_heads * head_dim
    patch = list(cfg["patch_size"])
    in_ch = int(cfg["in_channels"])
    out_ch = int(cfg["out_channels"])
    action_dim = int(cfg.get("action_dim", 30))
    ffn_dim = int(cfg["ffn_dim"])
    text_dim = int(cfg["text_dim"])
    freq_dim = int(cfg.get("freq_dim", 256))

    print("# LingBot-VA Transformer Spec")
    print()
    print("## Config-Derived Shapes")
    print()
    print(f"- layers: {n_layers}")
    print(f"- heads: {n_heads}")
    print(f"- head_dim: {head_dim}")
    print(f"- inner_dim: {inner}")
    print(f"- ffn_dim: {ffn_dim}")
    print(f"- text_dim: {text_dim}")
    print(f"- freq_dim: {freq_dim}")
    print(f"- patch_size: {patch}")
    print(f"- latent patch input dim: {in_ch * patch[0] * patch[1] * patch[2]}")
    print(f"- latent patch output dim: {out_ch * patch[0] * patch[1] * patch[2]}")
    print(f"- action_dim: {action_dim}")
    print()

    print("## Expected Major Tensor Shapes")
    print()
    print("```text")
    print(f"patch_embedding_mlp.weight: [{inner}, {in_ch * patch[0] * patch[1] * patch[2]}]")
    print(f"action_embedder.weight:     [{inner}, {action_dim}]")
    print(f"condition text linear_1:    [{inner}, {text_dim}]")
    print(f"condition text linear_2:    [{inner}, {inner}]")
    print(f"time_embedder linear_1:     [{inner}, {freq_dim}]")
    print(f"time_embedder linear_2:     [{inner}, {inner}]")
    print(f"time_proj.weight:           [{inner * 6}, {inner}]")
    print(f"attn q/k/v/o weights:       [{inner}, {inner}]")
    print(f"cross-attn k/v weights:     [{inner}, {inner}] from text-projected states")
    print(f"ffn up weight:              [{ffn_dim}, {inner}]")
    print(f"ffn down weight:            [{inner}, {ffn_dim}]")
    print(f"proj_out.weight:            [{out_ch * patch[0] * patch[1] * patch[2]}, {inner}]")
    print(f"action_proj_out.weight:     [{action_dim}, {inner}]")
    print("```")
    print()

    rows = _tensor_map(n_layers)
    missing = [src for src, _, _ in rows if "*" not in src and src not in weight_map]
    extra = sorted(set(weight_map) - {src for src, _, _ in rows if "*" not in src})

    print("## Tentative GGUF Tensor Naming")
    print()
    print("| Source tensor | GGUF tensor | Note |")
    print("| --- | --- | --- |")
    for src, dst, note in rows[:80]:
        print(f"| `{src}` | `{dst}` | {note} |")
    print()
    print(f"Full generated mapping rows: {len(rows)}")
    print(f"Concrete mapped tensors missing from index: {len(missing)}")
    if missing:
        for name in missing[:32]:
            print(f"- missing: {name}")
    print(f"Index tensors not concretely listed by this draft: {len(extra)}")
    if extra:
        for name in extra[:32]:
            print(f"- extra/unexpanded: {name}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_HF_SNAPSHOT)
    return dump_spec(parser.parse_args().checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
