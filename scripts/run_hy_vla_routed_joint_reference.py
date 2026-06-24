#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Dump HY-VLA routed-prefix + action-expert tensors without full policy load."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


DEFAULT_HY_REPO = Path("/home/xuling/robotic_code/embodied.cpp/Hy-Embodied-0.5-VLA-main")
DEFAULT_CKPT = Path("/home/xuling/robotic_dataset/HY-VLA")


class VLMTextWrapper(torch.nn.Module):
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model
        self.model = torch.nn.Module()
        self.model.visual = torch.nn.Identity()
        self.config = language_model.config


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


def load_prefixed_state(ckpt: Path, prefix: str, layers: int) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(ckpt / "model.safetensors"), framework="pt", device="cpu") as f:
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


def load_linear(raw, name: str, device: str, dtype: torch.dtype) -> torch.nn.Linear:
    w = raw.get_tensor(f"model.{name}.weight").float()
    b = raw.get_tensor(f"model.{name}.bias").float()
    out_features, in_features = w.shape
    layer = torch.nn.Linear(in_features, out_features)
    layer.weight.data.copy_(w)
    layer.bias.data.copy_(b)
    return layer.to(device=device, dtype=dtype)


def visual_patch_ranges_from_mask(mask_np: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_ranges: list[tuple[int, int]] = []
    full_ranges: list[tuple[int, int]] = []
    i = 0
    while i < int(mask_np.size):
        if mask_np[i] == 0:
            i += 1
            continue
        start = i
        while i < int(mask_np.size) and mask_np[i] != 0:
            i += 1
        end = i
        row_ranges.append((start, end))
    if row_ranges:
        full_ranges.append((row_ranges[0][0], row_ranges[-1][1]))
    return row_ranges, full_ranges


def apply_patch_only_visual_mask(att_2d: torch.Tensor, mask_np: np.ndarray) -> None:
    row_ranges, full_ranges = visual_patch_ranges_from_mask(mask_np)
    all_img_indices: list[int] = []
    for start, end in row_ranges:
        all_img_indices.extend(range(start, end))
    if all_img_indices:
        idx = torch.tensor(all_img_indices, dtype=torch.long, device=att_2d.device)
        att_2d[:, idx[:, None], idx[None, :]] = False
    for full_start, full_end in full_ranges:
        img_indices: list[int] = []
        for start, end in row_ranges:
            if start >= full_start and end <= full_end:
                img_indices.extend(range(start, end))
        if img_indices:
            idx = torch.tensor(img_indices, dtype=torch.long, device=att_2d.device)
            att_2d[:, idx[:, None], idx[None, :]] = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hy-repo", type=Path, default=DEFAULT_HY_REPO)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="f32")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--mode", choices=["routed_prefix", "routed_joint_vt", "routed_joint", "routed_sample"],
                        default="routed_joint_vt")
    parser.add_argument("--prefix-f32", type=Path, required=True)
    parser.add_argument("--mask-i32", type=Path, required=True)
    parser.add_argument("--dump-f32", type=Path, required=True)
    parser.add_argument("--x-f32", type=Path, default=None,
                        help="Optional raw float32 action/noise state [n_action_steps,max_action_dim].")
    parser.add_argument("--timestep", type=float, default=1.0,
                        help="Initial timestep for routed_joint_vt or routed_joint.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    sys.path.insert(0, str(args.hy_repo))
    from hy_vla import HyVLAConfig
    from hy_vla.modeling_dual_tower import HyDualTower
    from hy_vla.modeling_hy_vla import _load_vlm_autoconfig, create_sinusoidal_pos_embedding, make_att_2d_masks
    from hy_vla.hunyuan_vl_mot.modeling_hunyuan_vl_mot import _HunYuanVLMoTTextForCausalLM

    hy_cfg = HyVLAConfig.from_pretrained(args.ckpt)
    vlm_cfg = _load_vlm_autoconfig(hy_cfg)
    if args.layers > int(vlm_cfg.num_hidden_layers):
        raise SystemExit(f"--layers {args.layers} exceeds config.num_hidden_layers={vlm_cfg.num_hidden_layers}")
    vlm_cfg.num_hidden_layers = args.layers
    expert_cfg = copy.deepcopy(vlm_cfg)
    expert_cfg.hidden_size = int(hy_cfg.proj_width)
    expert_cfg.intermediate_size = 2048
    if hasattr(expert_cfg, "dense_list"):
        expert_cfg.dense_list = [int(hy_cfg.proj_width), 0]

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device_obj = torch.device(args.device)
    hidden = int(vlm_cfg.hidden_size)
    prefix_np = np.fromfile(args.prefix_f32, dtype=np.float32)
    if prefix_np.size % hidden != 0:
        raise SystemExit(f"{args.prefix_f32} size {prefix_np.size} is not divisible by hidden={hidden}")
    n_pref = prefix_np.size // hidden
    mask_np = np.fromfile(args.mask_i32, dtype=np.int32)
    if mask_np.size != n_pref:
        raise SystemExit(f"{args.mask_i32} length {mask_np.size} != prefix token count {n_pref}")

    vlm_lm = _HunYuanVLMoTTextForCausalLM._from_config(vlm_cfg)
    expert_lm = _HunYuanVLMoTTextForCausalLM._from_config(expert_cfg)
    miss, unexpected = vlm_lm.load_state_dict(
        load_prefixed_state(args.ckpt, "model.dual_tower.vlm.model.language_model.", args.layers),
        strict=False,
    )
    bad_missing = [m for m in miss if m not in {"lm_head.weight", "model.embed_tokens.weight"}]
    if bad_missing or unexpected:
        raise SystemExit(f"vlm load mismatch: bad_missing={bad_missing[:10]} unexpected={list(unexpected)[:10]}")
    miss, unexpected = expert_lm.load_state_dict(
        load_prefixed_state(args.ckpt, "model.dual_tower.expert.", args.layers),
        strict=False,
    )
    bad_missing = [m for m in miss if m not in {"lm_head.weight", "model.embed_tokens.weight"}]
    if bad_missing or unexpected:
        raise SystemExit(f"expert load mismatch: bad_missing={bad_missing[:10]} unexpected={list(unexpected)[:10]}")
    expert_lm.model.embed_tokens = None

    tower = HyDualTower.from_components(
        vlm=VLMTextWrapper(vlm_lm),
        expert=expert_lm,
        freeze_vision_encoder=False,
        train_expert_only=False,
        attention_implementation="eager",
        outer_config=hy_cfg,
    )
    tower.config.vlm_config.num_hidden_layers = args.layers
    tower.config.expert_config.num_hidden_layers = args.layers
    tower.eval()
    tower = tower.to(device=device_obj, dtype=dtype)

    with safe_open(str(args.ckpt / "model.safetensors"), framework="pt", device="cpu") as raw:
        state_proj = load_linear(raw, "state_proj", args.device, dtype)
        action_in_proj = load_linear(raw, "action_in_proj", args.device, dtype)
        action_time_mlp_in = load_linear(raw, "action_time_mlp_in", args.device, dtype)
        action_time_mlp_out = load_linear(raw, "action_time_mlp_out", args.device, dtype)
        action_out_proj = load_linear(raw, "action_out_proj", args.device, dtype)

    bsz = 1
    prefix = torch.from_numpy(prefix_np.reshape(1, n_pref, hidden)).to(device=device_obj, dtype=dtype)
    modality_prefix = torch.from_numpy(mask_np.reshape(1, n_pref).astype(np.bool_)).to(device=device_obj)
    prefix_pad = torch.ones((bsz, n_pref), dtype=torch.bool, device=device_obj)
    prefix_att = torch.ones((bsz, n_pref), dtype=torch.bool, device=device_obj)
    prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_att)
    apply_patch_only_visual_mask(prefix_att_2d, mask_np)
    prefix_pos = torch.arange(n_pref, dtype=torch.long, device=device_obj).unsqueeze(0)

    state = torch.zeros((bsz, int(hy_cfg.max_state_dim)), dtype=dtype, device=device_obj)
    if args.x_f32 is not None:
        x_np = np.fromfile(args.x_f32, dtype=np.float32)
        expected = int(hy_cfg.n_action_steps) * int(hy_cfg.max_action_dim)
        if x_np.size != expected:
            raise SystemExit(f"{args.x_f32} has {x_np.size} floats, expected {expected}")
        x_t = torch.from_numpy(x_np.reshape(bsz, int(hy_cfg.n_action_steps), int(hy_cfg.max_action_dim))).to(
            device=device_obj, dtype=torch.float32
        )
    else:
        x_t = torch.zeros((bsz, int(hy_cfg.n_action_steps), int(hy_cfg.max_action_dim)), dtype=torch.float32, device=device_obj)
    timestep = torch.full((bsz,), float(args.timestep), dtype=torch.float32, device=device_obj)

    with torch.no_grad():
        prefix_outputs, past_key_values, _, _ = tower.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
            fill_kv_cache=True,
            modality_masks=[modality_prefix, None],
        )
        if args.mode == "routed_prefix":
            prefix_out = prefix_outputs[0]
            summarize(f"{args.mode}_{args.layers}blk", prefix_out)
            dump_f32(args.dump_f32, prefix_out)
            return 0

        state_block = state_proj(state)[:, None, :].to(dtype=dtype)
        time_emb = create_sinusoidal_pos_embedding(
            timestep, int(hy_cfg.proj_width), min_period=4e-3, max_period=4.0, device=device_obj
        ).to(dtype=dtype)
        action_emb = action_in_proj(x_t.to(dtype=dtype))
        action_time = torch.cat([action_emb, time_emb[:, None, :].expand_as(action_emb)], dim=2)
        action_time = action_time_mlp_out(torch.nn.functional.silu(action_time_mlp_in(action_time)))
        suffix = torch.cat([state_block, action_time], dim=1)
        n_suf = suffix.shape[1]
        suffix_pad = torch.ones((bsz, n_suf), dtype=torch.bool, device=device_obj)
        suffix_att = torch.tensor([[1] + [1] + [0] * (int(hy_cfg.n_action_steps) - 1)],
                                  dtype=torch.bool, device=device_obj)
        suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
        prefix_pad_2d = prefix_pad[:, None, :].expand(bsz, n_suf, n_pref)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
        suffix_pos = n_pref + torch.arange(n_suf, dtype=torch.long, device=device_obj).unsqueeze(0)
        modality_suffix = torch.ones((bsz, n_suf), dtype=torch.bool, device=device_obj)

        def denoise_once(x_cur: torch.Tensor, timestep_cur: torch.Tensor) -> torch.Tensor:
            state_block_cur = state_proj(state)[:, None, :].to(dtype=dtype)
            time_emb_cur = create_sinusoidal_pos_embedding(
                timestep_cur, int(hy_cfg.proj_width), min_period=4e-3, max_period=4.0, device=device_obj
            ).to(dtype=dtype)
            action_emb_cur = action_in_proj(x_cur.to(dtype=dtype))
            action_time_cur = torch.cat(
                [action_emb_cur, time_emb_cur[:, None, :].expand_as(action_emb_cur)], dim=2
            )
            action_time_cur = action_time_mlp_out(torch.nn.functional.silu(action_time_mlp_in(action_time_cur)))
            suffix_cur = torch.cat([state_block_cur, action_time_cur], dim=1)
            outputs_embeds, _, _, _ = tower.forward(
                attention_mask=full_att_2d,
                position_ids=suffix_pos,
                past_key_values=copy.deepcopy(past_key_values),
                inputs_embeds=[None, suffix_cur],
                use_cache=True,
                fill_kv_cache=False,
                modality_masks=[None, modality_suffix],
            )
            suffix_out_cur = outputs_embeds[1][:, -int(hy_cfg.n_action_steps):]
            return action_out_proj(suffix_out_cur)

        v_t = denoise_once(x_t, timestep)
        if args.mode == "routed_sample":
            x_cur = x_t
            dt = -1.0 / int(hy_cfg.num_steps)
            for step in range(int(hy_cfg.num_steps)):
                t_cur = torch.full((bsz,), 1.0 + step * dt, dtype=torch.float32, device=device_obj)
                v_cur = denoise_once(x_cur, t_cur)
                x_cur = x_cur + dt * v_cur.float()
            out = x_cur
        else:
            x_1 = x_t + (-1.0 / int(hy_cfg.num_steps)) * v_t.float()
            out = x_1 if args.mode == "routed_joint" else v_t

    summarize(f"{args.mode}_{args.layers}blk", out)
    dump_f32(args.dump_f32, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
