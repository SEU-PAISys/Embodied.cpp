#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Check LingBot-VA / Wan 3D RoPE against ggml-style reuse candidates.

This script does not load model weights.  It compares the mathematical
frequency layout used by LingBot-VA's Python WanRotaryPosEmbed with candidate
layouts that are convenient to express through ggml RoPE primitives.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RopeDims:
    head_dim: int
    f_dim: int
    h_dim: int
    w_dim: int

    @classmethod
    def from_head_dim(cls, head_dim: int) -> "RopeDims":
        return cls(
            head_dim=head_dim,
            f_dim=head_dim - 2 * (head_dim // 3),
            h_dim=head_dim // 3,
            w_dim=head_dim // 3,
        )


def inv_freq(dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64)[: dim // 2] / dim))


def get_mesh_id(f: int, h: int, w: int, t: int, f_w: int = 1, f_shift: int = 0, action: bool = False) -> torch.Tensor:
    f_idx = torch.arange(f_shift, f + f_shift, dtype=torch.float64) * f_w
    h_idx = torch.arange(h, dtype=torch.float64)
    w_idx = torch.arange(w, dtype=torch.float64)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    if action:
        ff_offset = (torch.ones([h], dtype=torch.float64).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1
    grid_id = torch.cat([ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0).flatten(1)
    grid_id = torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)
    return grid_id


def wan_freqs(grid: torch.Tensor, dims: RopeDims, theta: float) -> torch.Tensor:
    f = grid[0, :, None] * inv_freq(dims.f_dim, theta).to(grid)
    h = grid[1, :, None] * inv_freq(dims.h_dim, theta).to(grid)
    w = grid[2, :, None] * inv_freq(dims.w_dim, theta).to(grid)
    return torch.cat([f, h, w], dim=-1)


def mrope_no_reset_freqs(grid: torch.Tensor, dims: RopeDims, theta: float) -> torch.Tensor:
    """Approximate ggml MROPE frequency indexing: no reset across sections."""
    pair_dims = [dims.f_dim // 2, dims.h_dim // 2, dims.w_dim // 2]
    inv_all = 1.0 / (theta ** (torch.arange(0, dims.head_dim, 2, dtype=torch.float64)[: sum(pair_dims)] / dims.head_dim))
    chunks = []
    offset = 0
    for axis, n_pairs in enumerate(pair_dims):
        chunks.append(grid[axis, :, None] * inv_all[offset : offset + n_pairs].to(grid))
        offset += n_pairs
    return torch.cat(chunks, dim=-1)


def normal_rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Wan/PyTorch ordering: adjacent pairs are interpreted as complex numbers."""
    xc = torch.view_as_complex(x.double().reshape(*x.shape[:-1], -1, 2))
    yc = xc * torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(yc).flatten(-2).to(x.dtype)


def neox_rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """NEOX-style ordering: first half and second half are paired."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def run(args: argparse.Namespace) -> int:
    dims = RopeDims.from_head_dim(args.head_dim)
    print("# LingBot-VA / Wan RoPE Check")
    print(f"head_dim: {dims.head_dim}")
    print(f"wan dims: f={dims.f_dim} h={dims.h_dim} w={dims.w_dim}")
    print(f"pair sections: f={dims.f_dim // 2} h={dims.h_dim // 2} w={dims.w_dim // 2}")
    print()

    for action in (False, True):
        grid = get_mesh_id(args.frames, args.height, args.width, t=0, action=action)
        seq = grid.shape[1]
        torch.manual_seed(0)
        x = torch.randn(seq, args.heads, args.head_dim, dtype=torch.float32)

        wf = wan_freqs(grid, dims, args.theta)
        mf = mrope_no_reset_freqs(grid, dims, args.theta)
        wan = normal_rope_apply(x, wf[:, None, :])
        same_math = normal_rope_apply(x, mf[:, None, :])
        neox = neox_rope_apply(x, mf[:, None, :])

        label = "action" if action else "latent"
        print(f"## {label}")
        print(f"grid shape: {tuple(grid.shape)} seq={seq}")
        print(f"grid sample columns: {grid[:, :min(4, seq)].tolist()}")
        print(f"max |Wan freq - MROPE-no-reset freq|: {max_abs(wf, mf):.8g}")
        print(f"max |Wan apply - normal(no-reset-freq) apply|: {max_abs(wan, same_math):.8g}")
        print(f"max |Wan apply - NEOX(no-reset-freq) apply|: {max_abs(wan, neox):.8g}")
        print()

    print("Interpretation:")
    print("- If both diffs are non-zero, direct ggml MROPE reuse is not mathematically identical.")
    print("- Action grids include fractional frame offsets, which also needs attention in a ggml reuse path.")
    print("- A Wan-specific cos/sin table path can still be expressed with ggml elementwise ops if needed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--height", type=int, default=4)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--theta", type=float, default=10000.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
