#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Export small LingBot-VA ONNX skeletons and summarize operator portability.

The full LingBot-VA checkpoint is too large for a quick graph audit.  This
script exports representative subgraphs with the same operator families used by
the original model, then writes a bilingual Markdown report that classifies
which parts should be expressible with ggml graph composition and which parts
may deserve custom kernels later.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MODEL_ROOT = Path("/home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long")


class WanRopeSkeleton(nn.Module):
    def forward(self, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # q: [B, L, H, D], cos/sin: [B, L, 1, D/2]
        b, l, h, d = q.shape
        x = q.reshape(b, l, h, d // 2, 2)
        x0 = x[..., 0]
        x1 = x[..., 1]
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.stack((y0, y1), dim=-1).reshape(b, l, h, d)


class WanAttentionSkeleton(nn.Module):
    def __init__(self, dim: int = 96, heads: int = 4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.q_norm_weight = nn.Parameter(torch.ones(dim))
        self.k_norm_weight = nn.Parameter(torch.ones(dim))
        self.to_out = nn.Linear(dim, dim)

    @staticmethod
    def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * weight

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        q = self.rms_norm(self.to_q(x), self.q_norm_weight).reshape(b, l, self.heads, self.head_dim)
        k = self.rms_norm(self.to_k(x), self.k_norm_weight).reshape(b, l, self.heads, self.head_dim)
        v = self.to_v(x).reshape(b, l, self.heads, self.head_dim)
        q = WanRopeSkeleton()(q, rope_cos, rope_sin)
        k = WanRopeSkeleton()(k, rope_cos, rope_sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(b, l, self.dim)
        return self.to_out(y)


class WanBlockSkeleton(nn.Module):
    def __init__(self, dim: int = 96, heads: int = 4, text_dim: int = 128, ffn_dim: int = 192):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = WanAttentionSkeleton(dim, heads)
        self.cross_q = nn.Linear(dim, dim)
        self.cross_k = nn.Linear(dim, dim)
        self.cross_v = nn.Linear(dim, dim)
        self.cross_o = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(text_dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_up = nn.Linear(dim, ffn_dim)
        self.ff_down = nn.Linear(ffn_dim, dim)
        self.scale_shift = nn.Parameter(torch.zeros(1, 6, dim))

    def forward(self, x: torch.Tensor, text: torch.Tensor, temb6: torch.Tensor,
                rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        ss = self.scale_shift[:, None] + temb6
        shift, scale, gate, c_shift, c_scale, c_gate = ss.unbind(dim=2)
        n1 = self.norm1(x.float()) * (1.0 + scale) + shift
        x = x + self.self_attn(n1, rope_cos, rope_sin) * gate

        q = self.cross_q(self.norm2(x))
        t = self.text_proj(text)
        k = self.cross_k(t)
        v = self.cross_v(t)
        y = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None]).squeeze(1)
        x = x + self.cross_o(y)

        n3 = self.norm3(x.float()) * (1.0 + c_scale) + c_shift
        x = x + self.ff_down(F.gelu(self.ff_up(n3), approximate="tanh")) * c_gate
        return x


class FlowStepSkeleton(nn.Module):
    def forward(self, sample: torch.Tensor, model_output: torch.Tensor,
                sigma_now: torch.Tensor, sigma_next: torch.Tensor) -> torch.Tensor:
        return sample + model_output * (sigma_next - sigma_now)


class CausalConv3dSkeleton(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(12, 16, kernel_size=(3, 3, 3), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cache, x], dim=2)
        return F.silu(self.conv(x))


@dataclass
class ExportedGraph:
    name: str
    path: Path
    ops: Counter[str]


def export_graph(name: str, module: nn.Module, args: tuple[torch.Tensor, ...], out_dir: Path) -> ExportedGraph:
    path = out_dir / f"{name}.onnx"
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            args,
            str(path),
            input_names=[f"input_{i}" for i in range(len(args))],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
        )
    model = onnx.load(str(path))
    ops = Counter(node.op_type for node in model.graph.node)
    return ExportedGraph(name, path, ops)


def tensor_prefix_stats(index_path: Path) -> Counter[str]:
    if not index_path.exists():
        return Counter()
    data = json.loads(index_path.read_text())
    weight_map = data.get("weight_map", {})
    return Counter(name.split(".", 2)[0] if "." in name else name for name in weight_map)


def safetensor_name_stats(path: Path) -> Counter[str]:
    try:
        from safetensors import safe_open
    except Exception:
        return Counter()
    if not path.exists():
        return Counter()
    counts: Counter[str] = Counter()
    with safe_open(path, framework="pt", device="cpu") as sf:
        for key in sf.keys():
            counts[key.split(".", 2)[0]] += 1
    return counts


def report(out_dir: Path, model_root: Path, graphs: list[ExportedGraph]) -> None:
    transformer_cfg = json.loads((model_root / "transformer/config.json").read_text())
    vae_cfg = json.loads((model_root / "vae/config.json").read_text())
    text_cfg = json.loads((model_root / "text_encoder/config.json").read_text())
    transformer_prefix = tensor_prefix_stats(model_root / "transformer/diffusion_pytorch_model.safetensors.index.json")
    text_prefix = tensor_prefix_stats(model_root / "text_encoder/model.safetensors.index.json")
    vae_prefix = safetensor_name_stats(model_root / "vae/diffusion_pytorch_model.safetensors")

    all_ops: Counter[str] = Counter()
    for g in graphs:
        all_ops.update(g.ops)

    md = out_dir / "LINGBOT_VA_ONNX_AUDIT.md"
    lines: list[str] = []
    lines.append("# LingBot-VA ONNX Operator Audit / LingBot-VA ONNX 算子审计\n")
    lines.append("This report uses small representative ONNX skeletons plus checkpoint metadata. It is not a full 22GB model export.\n")
    lines.append("本文使用小型代表性 ONNX skeleton 和权重 metadata，不是完整 22GB 模型导出。\n")
    lines.append("## Exported ONNX Graphs / 已导出的 ONNX 图\n")
    for g in graphs:
        lines.append(f"- `{g.name}`: `{g.path}`")
        lines.append("  ops: " + ", ".join(f"{k}={v}" for k, v in sorted(g.ops.items())))
    lines.append("\n## Combined ONNX Ops / 汇总 ONNX 算子\n")
    for op, count in all_ops.most_common():
        lines.append(f"- `{op}`: {count}")
    lines.append("\n## Checkpoint Metadata / 权重元数据\n")
    lines.append(f"- Transformer: layers={transformer_cfg.get('num_layers')}, hidden={transformer_cfg.get('num_attention_heads') * transformer_cfg.get('attention_head_dim')}, ffn={transformer_cfg.get('ffn_dim')}, attn_mode={transformer_cfg.get('attn_mode')}")
    lines.append(f"- VAE: class={vae_cfg.get('_class_name')}, base_dim={vae_cfg.get('base_dim')}, z_dim={vae_cfg.get('z_dim')}, patch_size={vae_cfg.get('patch_size')}, temporal_downsample={vae_cfg.get('temperal_downsample')}")
    lines.append(f"- UMT5: layers={text_cfg.get('num_layers')}, d_model={text_cfg.get('d_model')}, d_ff={text_cfg.get('d_ff')}, heads={text_cfg.get('num_heads')}, buckets={text_cfg.get('relative_attention_num_buckets')}")
    lines.append(f"- Transformer tensor top prefixes: {dict(transformer_prefix.most_common(8))}")
    lines.append(f"- Text encoder tensor top prefixes: {dict(text_prefix.most_common(8))}")
    lines.append(f"- VAE tensor top prefixes: {dict(vae_prefix.most_common(8))}")
    lines.append("\n## Portability Classification / 可移植性分类\n")
    rows = [
        ("WanTransformer linear / norm / FFN", "ggml composition", "MatMul/Add/LayerNorm/RMSNorm/GELU/Silu are already available or easy to compose."),
        ("Wan self-attention and cross-attention", "ggml composition first", "Use ggml_flash_attn_ext + masks/cache. Custom windowed attention may be useful later."),
        ("Wan 3D RoPE", "ggml composition/custom helper", "We already implemented C++ real-op RoPE composition; no new backend op required for first version."),
        ("FlowMatchScheduler step", "host/C++ + ggml simple ops", "sample + velocity * delta_sigma; no custom op needed."),
        ("UMT5 encoder", "ggml composition, high memory risk", "Transformer encoder with relative position bias and gated GELU. Quantization/loading strategy matters more than new ops."),
        ("VAE conv/resnet/up/downsample", "ggml composition possible", "Conv3D/GroupNorm/SiLU/residual blocks likely map to ggml, but implementation is sizable."),
        ("WanCausalConv3d streaming cache", "possible custom kernel later", "Can be emulated by concat+conv first; high-performance streaming causal conv may deserve a specialized path."),
        ("Flex/block-sparse attention mask", "possible custom kernel later", "Training flex mask is complex; inference cache path may use dense attention first. Efficient long-window world-model attention is a strong custom-kernel candidate."),
    ]
    lines.append("| Module / 模块 | First port strategy / 第一版策略 | Notes / 说明 |")
    lines.append("|---|---|---|")
    for a, b, c in rows:
        lines.append(f"| {a} | {b} | {c} |")
    lines.append("\n## Netron / 可视化\n")
    lines.append("Open any `.onnx` file in this directory with Netron, for example:\n")
    lines.append("```bash\nnetron " + str((out_dir / "wan_block.onnx").resolve()) + "\n```\n")
    lines.append("可以用 Netron 打开本目录下任意 `.onnx` 文件查看算子图。\n")
    md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/lingbot_va_onnx"))
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = [
        export_graph(
            "wan_rope",
            WanRopeSkeleton(),
            (torch.randn(1, 8, 4, 24), torch.randn(1, 8, 1, 12), torch.randn(1, 8, 1, 12)),
            out_dir,
        ),
        export_graph(
            "wan_attention",
            WanAttentionSkeleton(),
            (torch.randn(1, 8, 96), torch.randn(1, 8, 1, 12), torch.randn(1, 8, 1, 12)),
            out_dir,
        ),
        export_graph(
            "wan_block",
            WanBlockSkeleton(),
            (
                torch.randn(1, 8, 96),
                torch.randn(1, 4, 128),
                torch.randn(1, 8, 6, 96),
                torch.randn(1, 8, 1, 12),
                torch.randn(1, 8, 1, 12),
            ),
            out_dir,
        ),
        export_graph(
            "flow_step",
            FlowStepSkeleton(),
            (torch.randn(1, 30, 4, 4, 1), torch.randn(1, 30, 4, 4, 1), torch.tensor(0.5), torch.tensor(0.45)),
            out_dir,
        ),
        export_graph(
            "causal_conv3d_skeleton",
            CausalConv3dSkeleton(),
            (torch.randn(1, 12, 2, 16, 16), torch.randn(1, 12, 2, 16, 16)),
            out_dir,
        ),
    ]
    report(out_dir, args.model_root.resolve(), graphs)
    print(f"wrote {out_dir}")
    for graph in graphs:
        print(f"{graph.name}: {graph.path}")
    print(f"report: {out_dir / 'LINGBOT_VA_ONNX_AUDIT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
