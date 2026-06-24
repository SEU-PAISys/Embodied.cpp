#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run the LingBot-VA one-block smoke path in PyTorch and compare C++ dumps.

This mirrors the C++ `VLA_LINGBOT_COMPUTE_SMOKE=1` debug path.  It intentionally
uses GGUF tensors as the source of truth, converts BF16 tensors to float32, and
uses the same deterministic synthetic inputs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party/llama.cpp/gguf-py"))
import gguf  # noqa: E402


def bf16_bytes_to_f32(data: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    u16 = np.asarray(data, dtype=np.uint8).reshape(-1, 2).view(np.uint16).reshape(-1)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape)


class GgufWeights:
    def __init__(self, path: Path):
        self.reader = gguf.GGUFReader(str(path))
        self.tensors = {t.name: t for t in self.reader.tensors}

    def f32(self, name: str) -> torch.Tensor:
        t = self.tensors[name]
        # gguf-py reports tensor.shape in ggml ne order.  The byte buffer is
        # laid out in the original PyTorch order, so reverse the shape for
        # PyTorch parity code.
        shape = tuple(reversed([int(x) for x in t.shape]))
        if t.tensor_type.name == "BF16":
            arr = bf16_bytes_to_f32(t.data, shape)
        elif t.tensor_type.name == "F32":
            arr = np.asarray(t.data, dtype=np.float32).reshape(shape)
        else:
            raise ValueError(f"unsupported tensor type for {name}: {t.tensor_type}")
        return torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))


def deterministic(shape: tuple[int, ...], scale: float) -> torch.Tensor:
    n = int(np.prod(shape))
    x = np.sin(np.arange(n, dtype=np.float32) * np.float32(0.013)) * np.float32(scale)
    if len(shape) == 2:
        # C++ fills ggml memory directly, where ne0 is contiguous.  Recreate
        # the same logical [ne0, ne1] tensor from that memory order.
        return torch.from_numpy(x.reshape((shape[1], shape[0])).T.copy())
    return torch.from_numpy(x.reshape(shape).copy())


def timestep_embedding(timestep: float, seq: int, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = torch.full((seq, 1), float(timestep), dtype=torch.float32) * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb.T.contiguous()


@dataclass(frozen=True)
class GridSpec:
    f: int = 1
    h: int = 1
    w: int = 1
    t: int = 0
    f_w: int = 1
    f_shift: int = 0
    action: bool = False

    @property
    def seq(self) -> int:
        return self.f * self.h * self.w


def smoke_grid_spec(seq: int, action: bool) -> GridSpec:
    if action:
        return GridSpec(f=1, h=seq, w=1, t=1, action=True)
    return GridSpec(f=seq, h=1, w=1, t=0, action=False)


def lingbot_rope(head_dim: int, spec: GridSpec) -> tuple[torch.Tensor, torch.Tensor]:
    h_dim = head_dim // 3
    w_dim = head_dim // 3
    f_dim = head_dim - 2 * h_dim
    dims = (f_dim, h_dim, w_dim)
    pairs = head_dim // 2
    seq = spec.seq
    cos = torch.ones((1, pairs, 1, seq), dtype=torch.float32)
    sin = torch.zeros((1, pairs, 1, seq), dtype=torch.float32)
    theta = 10000.0

    s = 0
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
                pair_index = 0
                for axis_dim, grid_id in zip(dims, (f_id, h_id, w_id)):
                    for p in range(axis_dim // 2):
                        base = 1.0 / (theta ** (float(2 * p) / float(axis_dim)))
                        freq = grid_id * base
                        cos[0, pair_index, 0, s] = np.cos(freq)
                        sin[0, pair_index, 0, s] = np.sin(freq)
                        pair_index += 1
                s += 1
    return cos, sin


def lin(w: GgufWeights, prefix: str, x: torch.Tensor) -> torch.Tensor:
    weight = w.f32(prefix + ".weight")
    bias = w.f32(prefix + ".bias")
    return weight @ x + bias[:, None]


def chunk_hidden(x: torch.Tensor, hidden: int, chunk_id: int) -> torch.Tensor:
    return x[chunk_id * hidden : (chunk_id + 1) * hidden]


def adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.layer_norm(x.T.float(), (x.shape[0],), eps=eps).T * (1.0 + scale) + shift


def apply_wan_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [head_dim, heads, seq], cos/sin: [1, pairs, 1, seq]
    hd, heads, seq = x.shape
    pairs = hd // 2
    xp = x.reshape(2, pairs, heads, seq)
    x0 = xp[0:1]
    x1 = xp[1:2]
    c = cos.expand(1, pairs, heads, seq)
    s = sin.expand(1, pairs, heads, seq)
    y0 = x0 * c - x1 * s
    y1 = x1 * c + x0 * s
    return torch.cat([y0, y1], dim=0).reshape(hd, heads, seq)


def attention(
    w: GgufWeights,
    prefix: str,
    q_in: torch.Tensor,
    kv_in: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    trace: dict[str, torch.Tensor] = {}
    q = lin(w, prefix + ".q", q_in)
    k = lin(w, prefix + ".k", kv_in)
    v = lin(w, prefix + ".v", kv_in)
    q_norm = w.f32(prefix + ".q_norm.weight")
    k_norm = w.f32(prefix + ".k_norm.weight")
    q = F.rms_norm(q.T.float(), (q.shape[0],), weight=q_norm, eps=1e-6).T
    k = F.rms_norm(k.T.float(), (k.shape[0],), weight=k_norm, eps=1e-6).T
    trace["q"] = q
    trace["k"] = k
    trace["v"] = v
    qh = q.reshape(head_dim, heads, q.shape[1])
    kh = k.reshape(head_dim, heads, k.shape[1])
    vh = v.reshape(head_dim, heads, v.shape[1])
    if cos is not None and sin is not None and q.shape[1] == k.shape[1]:
        qh = apply_wan_rope(qh, cos, sin)
        kh = apply_wan_rope(kh, cos, sin)
    # C++ smoke: Q/K shape [hd, seq, heads], V shape [heads, seq, hd] conceptually.
    q_t = qh.permute(2, 1, 0)  # [seq_q, heads, hd]
    k_t = kh.permute(2, 1, 0)  # [seq_k, heads, hd]
    v_t = vh.permute(2, 1, 0)  # [seq_k, heads, hd]
    scores = torch.einsum("qhd,khd->hqk", q_t, k_t) / (head_dim**0.5)
    probs = torch.softmax(scores, dim=-1)
    ctx = torch.einsum("hqk,khd->qhd", probs, v_t).reshape(q.shape[1], heads * head_dim).T
    return lin(w, prefix + ".o", ctx), trace


def block(
    w: GgufWeights,
    index: int,
    x: torch.Tensor,
    text: torch.Tensor,
    timestep_proj: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    trace: dict[str, torch.Tensor] = {}
    hidden = x.shape[0]
    p = f"wvm.blk.{index}"
    scale_shift = w.f32(p + ".scale_shift").reshape(-1)
    shift = chunk_hidden(timestep_proj, hidden, 0) + scale_shift[0 * hidden : 1 * hidden, None]
    scale = chunk_hidden(timestep_proj, hidden, 1) + scale_shift[1 * hidden : 2 * hidden, None]
    gate = chunk_hidden(timestep_proj, hidden, 2) + scale_shift[2 * hidden : 3 * hidden, None]
    c_shift = chunk_hidden(timestep_proj, hidden, 3) + scale_shift[3 * hidden : 4 * hidden, None]
    c_scale = chunk_hidden(timestep_proj, hidden, 4) + scale_shift[4 * hidden : 5 * hidden, None]
    c_gate = chunk_hidden(timestep_proj, hidden, 5) + scale_shift[5 * hidden : 6 * hidden, None]

    n1 = adaln(x, shift, scale)
    trace["n1"] = n1
    self_attn, self_trace = attention(w, p + ".self_attn", n1, n1, cos, sin, heads, head_dim)
    trace["self_q"] = self_trace["q"]
    trace["self_k"] = self_trace["k"]
    trace["self_v"] = self_trace["v"]
    trace["self_attn"] = self_attn
    x = x + self_attn * gate
    trace["post_self"] = x

    cn_w = w.f32(p + ".cross_norm.weight")
    cn_b = w.f32(p + ".cross_norm.bias")
    n2 = F.layer_norm(x.T.float(), (hidden,), weight=cn_w, bias=cn_b, eps=1e-6).T
    trace["n2"] = n2
    cross_attn, _ = attention(w, p + ".cross_attn", n2, text, None, None, heads, head_dim)
    trace["cross_attn"] = cross_attn
    x = x + cross_attn
    trace["post_cross"] = x

    n3 = adaln(x, c_shift, c_scale)
    trace["n3"] = n3
    ff = lin(w, p + ".ffn_down", F.gelu(lin(w, p + ".ffn_up", n3), approximate="tanh"))
    trace["ff"] = ff
    return x + ff * c_gate, trace


def run_one(w: GgufWeights, action: bool, blocks: int) -> dict[str, torch.Tensor]:
    hidden = 3072
    heads = 24
    head_dim = 128
    seq = 2
    text_seq = 2
    x_in = deterministic((30 if action else 192, seq), 0.03 if action else 0.02)
    text_raw = deterministic((4096, text_seq), 0.01)
    time_raw = timestep_embedding(0.0, seq)
    cos, sin = lingbot_rope(head_dim, smoke_grid_spec(seq, action))

    x = lin(w, "wvm.action_embd" if action else "wvm.patch_embd_mlp", x_in)
    text = lin(w, "wvm.cond.text_l2", F.gelu(lin(w, "wvm.cond.text_l1", text_raw), approximate="tanh"))
    cond = "wvm.action_cond" if action else "wvm.cond"
    t_hidden = lin(w, cond + ".time_l2", F.silu(lin(w, cond + ".time_l1", time_raw)))
    timestep_proj = lin(w, cond + ".time_proj", t_hidden)
    block_trace: dict[str, torch.Tensor] = {}
    for i in range(blocks):
        x, trace = block(w, i, x, text, timestep_proj, cos, sin, heads, head_dim)
        if i + 1 == blocks:
            block_trace = trace
    block_out = x
    ss = w.f32("wvm.output_scale_shift").reshape(-1)
    out_shift = t_hidden + ss[:hidden, None]
    out_scale = t_hidden + ss[hidden : 2 * hidden, None]
    out = adaln(x, out_shift, out_scale)
    final = lin(w, "wvm.action_out" if action else "wvm.output_proj", out)
    out = {
        "x_emb": (lin(w, "wvm.action_embd" if action else "wvm.patch_embd_mlp", x_in)),
        "text": text,
        "t_hidden": t_hidden,
        "timestep_proj": timestep_proj,
        "block_out": block_out,
        "": final,
    }
    out.update(block_trace)
    return out


def compare(label: str, py: torch.Tensor, dump_dir: Path) -> None:
    shape_path = dump_dir / f"lingbot_smoke_{label}.shape.txt"
    data_path = dump_dir / f"lingbot_smoke_{label}.f32"
    shape = tuple(int(x) for x in shape_path.read_text().split())
    # ggml stores ne0-contiguous tensors.  Interpret as [ne1, ne0] in C-order,
    # then transpose back to the logical [ne0, ne1] matrix used by PyTorch.
    cpp = np.fromfile(data_path, dtype=np.float32).reshape((shape[1], shape[0])).T
    pt = py.detach().cpu().numpy().astype(np.float32)
    diff = np.abs(pt - cpp)
    print(f"{label}: shape={pt.shape} cpp_shape={cpp.shape}")
    print(f"{label}: py_checksum={pt.sum():.8g} cpp_checksum={cpp.sum():.8g}")
    print(f"{label}: max_abs_diff={diff.max():.8g} mean_abs_diff={diff.mean():.8g}")


def compare_group(label: str, values: dict[str, torch.Tensor], dump_dir: Path) -> None:
    for suffix, tensor in values.items():
        dump_label = label if suffix == "" else f"{label}_{suffix}"
        compare(dump_label, tensor, dump_dir)


def compare_final(label: str, values: dict[str, torch.Tensor], dump_dir: Path) -> None:
    compare(label, values[""], dump_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, default=Path("/home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf"))
    parser.add_argument("--dump-dir", type=Path, default=Path("/tmp/lingbot_va_parity"))
    parser.add_argument("--blocks", type=int, default=1, help="Number of leading transformer blocks mirrored from the C++ smoke path")
    parser.add_argument("--final-only", action="store_true", help="Compare only final latent/action outputs")
    args = parser.parse_args()
    if args.blocks < 1:
        raise ValueError("--blocks must be >= 1")

    torch.set_num_threads(4)
    w = GgufWeights(args.gguf)
    latent = run_one(w, action=False, blocks=args.blocks)
    action = run_one(w, action=True, blocks=args.blocks)
    if args.final_only:
        compare_final("latent", latent, args.dump_dir)
        compare_final("action", action, args.dump_dir)
    else:
        compare_group("latent", latent, args.dump_dir)
        compare_group("action", action, args.dump_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
