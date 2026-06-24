#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Tiny Hy-VLA Python parity harness for suffix and prefix-KV action paths."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch


DEFAULT_HY_REPO = Path("/home/xuling/robotic_code/embodied.cpp/Hy-Embodied-0.5-VLA-main")
DEFAULT_CKPT = Path("/home/xuling/robotic_dataset/HY-VLA")


def summarize(name: str, x: torch.Tensor) -> None:
    y = x.detach().float().cpu().numpy()
    print(
        f"{name}: shape={tuple(x.shape)} "
        f"sum={float(y.sum()):.12f} min={float(y.min()):.12f} max={float(y.max()):.12f}"
    )


def dump_f32(path: Path | None, x: torch.Tensor) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    x.detach().float().cpu().numpy().astype(np.float32).tofile(path)
    print(f"dumped_f32={path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hy-repo", type=Path, default=DEFAULT_HY_REPO)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=["suffix", "joint", "mixed_prefix", "mixed_joint", "hy_prefix", "routed_prefix"],
        default="joint",
    )
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--dump-f32", type=Path, default=None)
    parser.add_argument("--dump-mask-i32", type=Path, default=None)
    parser.add_argument("--prefix-f32", type=Path, default=None,
                        help="Full prefix embeddings for routed_prefix mode, raw f32 [prefix_tokens, hidden].")
    parser.add_argument("--mask-i32", type=Path, default=None,
                        help="0=text / 1=vision modality mask for routed_prefix mode.")
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--vis-tokens-per-image", type=int, default=49)
    args = parser.parse_args()

    sys.path.insert(0, str(args.hy_repo))
    from hy_vla import HyVLA, HyVLAConfig
    from hy_vla.modeling_hy_vla import make_att_2d_masks

    config = HyVLAConfig.from_pretrained(args.ckpt)
    policy = HyVLA.from_pretrained(str(args.ckpt), config=config, map_location="cpu")
    policy.eval()
    policy.model.dual_tower.vlm.config.num_hidden_layers = args.layers
    policy.model.dual_tower.config.vlm_config.num_hidden_layers = args.layers

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    policy = policy.to(dtype=dtype)
    fm = policy.model

    bsz = 1
    prefix_tokens = torch.tensor([[120000, 120001, 120020, 7]], dtype=torch.long)
    prefix_len = prefix_tokens.shape[1]
    state = torch.zeros((bsz, config.max_state_dim), dtype=dtype)
    x_t = torch.zeros((bsz, config.n_action_steps, config.max_action_dim), dtype=torch.float32)
    timestep = torch.ones((bsz,), dtype=torch.float32)

    with torch.no_grad():
        prefix_embs = fm.dual_tower.embed_language_tokens(prefix_tokens)
        summarize("prefix_emb", prefix_embs)
        if args.mode == "hy_prefix":
            hidden = prefix_embs.shape[-1]
            grid = int(round(args.vis_tokens_per_image ** 0.5))
            if grid * grid != args.vis_tokens_per_image:
                raise ValueError("--vis-tokens-per-image must be a square number")
            n_vis = args.n_images * args.vis_tokens_per_image
            vis = torch.sin(torch.arange(n_vis * hidden, dtype=torch.float32) * 0.013)
            vis = (vis * 0.02).view(args.n_images, args.vis_tokens_per_image, hidden).to(dtype=dtype)

            def embed_ids(ids: list[int]) -> torch.Tensor:
                t = torch.tensor([ids], dtype=torch.long)
                return fm.dual_tower.embed_language_tokens(t).to(dtype=dtype)

            chunks: list[torch.Tensor] = []
            modality: list[int] = []

            def add_text(ids: list[int]) -> None:
                chunks.append(embed_ids(ids))
                modality.extend([0] * len(ids))

            def add_visual(x: torch.Tensor) -> None:
                chunks.append(x.reshape(1, 1, hidden).to(dtype=dtype))
                modality.append(1)

            add_text([120000, 120006])  # <bos><hy_User>
            for img_idx in range(args.n_images):
                add_text([120684])  # vision_start / placeholder no 666
                for r in range(grid):
                    for c in range(grid):
                        add_visual(vis[img_idx, r * grid + c])
                    add_text([120689])  # vision_split / placeholder no 671
                add_text([120685])  # vision_end / placeholder no 667
            add_text(prefix_tokens[0].tolist())

            hy_prefix = torch.cat(chunks, dim=1)
            mask = np.asarray(modality, dtype=np.int32)
            summarize("hy_prefix", hy_prefix)
            print(
                f"hy_prefix_mask: shape={(1, len(modality))} "
                f"sum={int(mask.sum())} first20={mask[:20].tolist()}"
            )
            dump_f32(args.dump_f32, hy_prefix)
            if args.dump_mask_i32 is not None:
                args.dump_mask_i32.parent.mkdir(parents=True, exist_ok=True)
                mask.tofile(args.dump_mask_i32)
                print(f"dumped_mask_i32={args.dump_mask_i32}")
            return 0

        if args.mode in {"mixed_prefix", "mixed_joint"}:
            n_vis = 3
            hidden = prefix_embs.shape[-1]
            vis = torch.sin(torch.arange(n_vis * hidden, dtype=torch.float32) * 0.013)
            vis = (vis * 0.02).view(bsz, n_vis, hidden).to(dtype=dtype)
            mixed = torch.cat([prefix_embs, vis], dim=1)
            mixed_len = mixed.shape[1]
            pad = torch.ones((bsz, mixed_len), dtype=torch.bool)
            att = torch.ones((bsz, mixed_len), dtype=torch.bool)
            att_2d = make_att_2d_masks(pad, att)
            pos = torch.cumsum(pad, dim=1) - 1
            modality = torch.cat(
                [torch.zeros((bsz, prefix_len), dtype=torch.bool),
                 torch.ones((bsz, n_vis), dtype=torch.bool)],
                dim=1,
            )
            outputs_embeds, past_key_values, _, _ = fm.dual_tower.forward(
                attention_mask=att_2d,
                position_ids=pos,
                past_key_values=None,
                inputs_embeds=[mixed, None],
                use_cache=(args.mode == "mixed_joint"),
                fill_kv_cache=(args.mode == "mixed_joint"),
                modality_masks=[modality, None],
            )
            if args.mode == "mixed_prefix":
                summarize(f"mixed_prefix_{args.layers}blk", outputs_embeds[0])
                dump_f32(args.dump_f32, outputs_embeds[0])
                return 0

            suffix_embs, suffix_pad_masks, suffix_att_masks, modality_mask_suffix = fm.embed_suffix(
                state, x_t, timestep
            )
            suffix_len = suffix_pad_masks.shape[1]
            prefix_pad_2d_masks = pad[:, None, :].expand(bsz, suffix_len, mixed_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(pad, dim=-1)[:, None]
            suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            outputs_embeds, _, _, _ = fm.dual_tower.forward(
                attention_mask=full_att_2d_masks,
                position_ids=suffix_position_ids,
                past_key_values=copy.deepcopy(past_key_values),
                inputs_embeds=[None, suffix_embs],
                use_cache=True,
                fill_kv_cache=False,
                modality_masks=[None, modality_mask_suffix],
            )
            suffix_out = outputs_embeds[1][:, -config.n_action_steps :]
            v_t = fm.action_out_proj(suffix_out)
            x_1 = x_t + (-1.0 / config.num_steps) * v_t.float()
            summarize(f"mixed_joint_v_t_{args.layers}blk", v_t)
            summarize(f"mixed_joint_x_1step_{args.layers}blk", x_1)
            dump_f32(args.dump_f32, v_t)
            return 0

        if args.mode == "routed_prefix":
            if args.prefix_f32 is None or args.mask_i32 is None:
                raise SystemExit("routed_prefix requires --prefix-f32 and --mask-i32")
            hidden = prefix_embs.shape[-1]
            prefix_np = np.fromfile(args.prefix_f32, dtype=np.float32)
            if prefix_np.size % hidden != 0:
                raise SystemExit(f"{args.prefix_f32} size {prefix_np.size} is not divisible by hidden={hidden}")
            n_pref = prefix_np.size // hidden
            mask_np = np.fromfile(args.mask_i32, dtype=np.int32)
            if mask_np.size != n_pref:
                raise SystemExit(f"{args.mask_i32} length {mask_np.size} != prefix tokens {n_pref}")
            routed = torch.from_numpy(prefix_np.reshape(1, n_pref, hidden)).to(dtype=dtype)
            modality = torch.from_numpy(mask_np.reshape(1, n_pref).astype(np.bool_))
            pad = torch.ones((bsz, n_pref), dtype=torch.bool)
            att = torch.ones((bsz, n_pref), dtype=torch.bool)
            att_2d = make_att_2d_masks(pad, att)
            pos = torch.cumsum(pad, dim=1) - 1
            outputs_embeds, _, _, _ = fm.dual_tower.forward(
                attention_mask=att_2d,
                position_ids=pos,
                past_key_values=None,
                inputs_embeds=[routed, None],
                use_cache=False,
                fill_kv_cache=False,
                modality_masks=[modality, None],
            )
            summarize(f"routed_prefix_{args.layers}blk", outputs_embeds[0])
            dump_f32(args.dump_f32, outputs_embeds[0])
            return 0

        prefix_pad_masks = torch.ones((bsz, prefix_len), dtype=torch.bool)
        prefix_att_masks = torch.ones((bsz, prefix_len), dtype=torch.bool)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        modality_mask_prefix = torch.zeros_like(prefix_pad_masks)

        (_, _), past_key_values, _, _ = fm.dual_tower.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
            modality_masks=[modality_mask_prefix, None],
        )

        suffix_embs, suffix_pad_masks, suffix_att_masks, modality_mask_suffix = fm.embed_suffix(
            state, x_t, timestep
        )

        if args.mode == "suffix":
            att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            position_ids = torch.cumsum(suffix_pad_masks, dim=1) - 1
            past = None
        else:
            suffix_len = suffix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsz, suffix_len, prefix_len)
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            past = copy.deepcopy(past_key_values)

        outputs_embeds, _, _, _ = fm.dual_tower.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=past,
            inputs_embeds=[None, suffix_embs],
            use_cache=(args.mode == "joint"),
            fill_kv_cache=False,
            modality_masks=[None, modality_mask_suffix],
        )

        suffix_out = outputs_embeds[1][:, -config.n_action_steps :]
        v_t = fm.action_out_proj(suffix_out)
        x_1 = x_t + (-1.0 / config.num_steps) * v_t.float()
        summarize(f"{args.mode}_v_t_{args.layers}blk", v_t)
        summarize(f"{args.mode}_x_1step_{args.layers}blk", x_1)
        dump_f32(args.dump_f32, x_1 if args.mode == "joint" else v_t)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
