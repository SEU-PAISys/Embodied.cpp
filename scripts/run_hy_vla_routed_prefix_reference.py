#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Dump HY-VLA VLM routed-prefix tensors without instantiating the full policy.

This harness targets the C++ `VLA_HY_VLA_DEBUG_OUTPUT=routed_prefix` path.  It
loads only the HunYuanVL-MoT text decoder layers needed for a prefix run, then
feeds a prebuilt full-prefix embedding tensor plus the 0=text / 1=vision
modality mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


DEFAULT_HY_REPO = Path("/home/xuling/robotic_code/embodied.cpp/Hy-Embodied-0.5-VLA-main")
DEFAULT_CKPT = Path("/home/xuling/robotic_dataset/HY-VLA")


def summarize(name: str, x: torch.Tensor) -> None:
    y = x.detach().float().cpu().numpy()
    print(
        f"{name}: shape={tuple(x.shape)} sum={float(y.sum()):.12f} "
        f"mean={float(y.mean()):.12f} min={float(y.min()):.12f} max={float(y.max()):.12f}"
    )


def dump_f32(path: Path, x: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(x.detach().float().cpu().numpy().astype(np.float32, copy=False))
    arr.tofile(path)
    path.with_suffix(path.suffix + ".shape.txt").write_text(
        " ".join(str(v) for v in arr.shape) + "\n", encoding="utf-8"
    )
    print(f"dumped_f32={path}")


def load_text_decoder_state(ckpt: Path, layers: int) -> dict[str, torch.Tensor]:
    tensor_path = ckpt / "model.safetensors"
    prefix = "model.dual_tower.vlm.model.language_model."
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(tensor_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if name in {"lm_head.weight", "model.embed_tokens.weight"}:
                continue
            if name.startswith("model.layers."):
                parts = name.split(".")
                if len(parts) < 3 or not parts[2].isdigit():
                    continue
                if int(parts[2]) >= layers:
                    continue
            state[name] = f.get_tensor(key)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hy-repo", type=Path, default=DEFAULT_HY_REPO)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="f32")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--prefix-f32", type=Path, required=True,
                        help="Raw float32 full-prefix embeddings [prefix_tokens, hidden].")
    parser.add_argument("--mask-i32", type=Path, required=True,
                        help="Raw int32 modality mask [prefix_tokens], 0=text and 1=vision.")
    parser.add_argument("--dump-f32", type=Path, required=True)
    args = parser.parse_args()

    if args.layers < 1:
        raise SystemExit("--layers must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    sys.path.insert(0, str(args.hy_repo))
    from hy_vla import HyVLAConfig
    from hy_vla.modeling_hy_vla import _load_vlm_autoconfig
    from hy_vla.hunyuan_vl_mot.modeling_hunyuan_vl_mot import _HunYuanVLMoTTextForCausalLM

    hy_cfg = HyVLAConfig.from_pretrained(args.ckpt)
    vlm_cfg = _load_vlm_autoconfig(hy_cfg)
    hidden = int(vlm_cfg.hidden_size)
    if args.layers > int(vlm_cfg.num_hidden_layers):
        raise SystemExit(f"--layers {args.layers} exceeds config.num_hidden_layers={vlm_cfg.num_hidden_layers}")
    vlm_cfg.num_hidden_layers = args.layers

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    prefix_np = np.fromfile(args.prefix_f32, dtype=np.float32)
    if prefix_np.size % hidden != 0:
        raise SystemExit(f"{args.prefix_f32} size {prefix_np.size} is not divisible by hidden={hidden}")
    n_pref = prefix_np.size // hidden
    mask_np = np.fromfile(args.mask_i32, dtype=np.int32)
    if mask_np.size != n_pref:
        raise SystemExit(f"{args.mask_i32} length {mask_np.size} != prefix token count {n_pref}")

    print(f"loading text decoder: layers={args.layers} hidden={hidden} prefix_tokens={n_pref}")
    decoder = _HunYuanVLMoTTextForCausalLM._from_config(vlm_cfg)
    state = load_text_decoder_state(args.ckpt, args.layers)
    missing, unexpected = decoder.load_state_dict(state, strict=False)
    missing_allowed = {"lm_head.weight", "model.embed_tokens.weight"}
    bad_missing = [m for m in missing if m not in missing_allowed]
    if bad_missing or unexpected:
        print(f"missing={len(missing)} unexpected={len(unexpected)}")
        if bad_missing:
            print("bad_missing_first20=", bad_missing[:20])
        if unexpected:
            print("unexpected_first20=", list(unexpected)[:20])
        raise SystemExit("text decoder state load did not match expected keys")

    decoder.eval()
    decoder = decoder.to(device=args.device, dtype=dtype)

    prefix = torch.from_numpy(prefix_np.reshape(1, n_pref, hidden)).to(device=args.device, dtype=dtype)
    modality = torch.from_numpy(mask_np.reshape(1, n_pref).astype(np.bool_)).to(device=args.device)
    attention_mask = torch.ones((1, n_pref), dtype=torch.bool, device=args.device)
    position_ids = torch.arange(n_pref, dtype=torch.long, device=args.device).unsqueeze(0)

    with torch.no_grad():
        out = decoder.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=prefix,
            cache_position=None,
            use_cache=False,
            modality_mask=modality,
        ).last_hidden_state

    summarize(f"routed_prefix_{args.layers}blk", out)
    dump_f32(args.dump_f32, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
