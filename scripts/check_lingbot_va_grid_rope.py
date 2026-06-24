#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Check LingBot-VA grid-id and Wan RoPE semantics against the Python reference."""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
LINGBOT_REPO = REPO.parent / "lingbot-va-main"
sys.path.insert(0, str(LINGBOT_REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_lingbot_va_oneblock_parity import GridSpec, lingbot_rope  # noqa: E402


def load_get_mesh_id():
    path = LINGBOT_REPO / "wan_va/utils/utils.py"
    spec = importlib.util.spec_from_file_location("lingbot_utils_direct", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_mesh_id


get_mesh_id = load_get_mesh_id()


def reference_rope(head_dim: int, grid: torch.Tensor) -> torch.Tensor:
    f_dim = head_dim - 2 * (head_dim // 3)
    h_dim = head_dim // 3
    w_dim = head_dim // 3
    theta = 10000.0

    def freqs_base(dim: int) -> torch.Tensor:
        return 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim))

    f_freqs = grid[None, 0, :].unsqueeze(-1) * freqs_base(f_dim)
    h_freqs = grid[None, 1, :].unsqueeze(-1) * freqs_base(h_dim)
    w_freqs = grid[None, 2, :].unsqueeze(-1) * freqs_base(w_dim)
    freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
    return torch.polar(torch.ones_like(freqs), freqs)[:, :, None]


def local_grid(spec: GridSpec) -> torch.Tensor:
    rows: list[list[float]] = [[], [], [], []]
    for fi in range(spec.f):
        for hi in range(spec.h):
            for wi in range(spec.w):
                f_id = float(spec.f_shift + fi) * float(spec.f_w)
                h_id = float(hi)
                w_id = float(wi)
                if spec.action:
                    f_id += float(hi + 1) / float(spec.h + 1)
                    h_id = -1.0
                    w_id = -1.0
                rows[0].append(f_id)
                rows[1].append(h_id)
                rows[2].append(w_id)
                rows[3].append(float(spec.t))
    return torch.tensor(rows, dtype=torch.float32)


def check_case(name: str, spec: GridSpec) -> None:
    ref_grid = get_mesh_id(
        spec.f,
        spec.h,
        spec.w,
        spec.t,
        spec.f_w,
        spec.f_shift,
        action=spec.action,
    ).float()
    cpp_like_grid = local_grid(spec)
    grid_diff = (ref_grid - cpp_like_grid).abs().max().item()

    ref = reference_rope(128, ref_grid)
    cos, sin = lingbot_rope(128, spec)
    local = torch.complex(cos.squeeze(0).squeeze(1).T, sin.squeeze(0).squeeze(1).T)[None, :, None]
    rope_diff = (ref - local).abs().max().item()

    print(f"{name}: seq={spec.seq} grid_max_diff={grid_diff:.8g} rope_max_diff={rope_diff:.8g}")


def main() -> int:
    cases = {
        "smoke_latent": GridSpec(f=2, h=1, w=1, t=0),
        "smoke_action": GridSpec(f=1, h=2, w=1, t=1, action=True),
        "latent_realish": GridSpec(f=4, h=4, w=8, t=0, f_shift=3),
        "action_realish": GridSpec(f=4, h=8, w=1, t=1, f_shift=3, action=True),
    }
    for name, spec in cases.items():
        check_case(name, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
