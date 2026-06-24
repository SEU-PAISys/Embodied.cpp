#!/usr/bin/env python3
"""Export an experimental pi0 full-inference ONNX graph.

This script is intentionally for graph visualization first.  The default
`structural` mode builds a tiny traceable PyTorch module that follows the
openpi pi0 inference data flow:

    images + prompt -> prefix tokens -> prefix KV/context
    state + noisy actions + timestep -> suffix tokens
    suffix expert -> velocity -> Euler denoising loop -> action chunk

It is not a weight-compatible openpi checkpoint export.  Use `--mode openpi`
after installing the openpi PyTorch dependencies if you want to experiment with
the repository's PI0Pytorch class directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MiniPi0Config:
    image_size: int = 32
    num_images: int = 3
    vocab_size: int = 256
    token_len: int = 8
    action_dim: int = 4
    action_horizon: int = 3
    width: int = 32
    depth: int = 2
    num_heads: int = 4
    num_steps: int = 2


class MiniTransformerBlock(nn.Module):
    def __init__(self, width: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class MiniPi0FullInference(nn.Module):
    """Small, traceable pi0-style full inference graph.

    The goal is to make a readable ONNX graph that contains the same major
    operator families as the original pi0 inference path, including the fixed
    Euler denoising loop.
    """

    def __init__(self, cfg: MiniPi0Config):
        super().__init__()
        self.cfg = cfg
        flat_image_dim = 3 * cfg.image_size * cfg.image_size
        self.image_proj = nn.Linear(flat_image_dim, cfg.width)
        self.text_embed = nn.Embedding(cfg.vocab_size, cfg.width)
        self.prefix_blocks = nn.ModuleList(
            [MiniTransformerBlock(cfg.width, cfg.num_heads) for _ in range(cfg.depth)]
        )

        self.state_proj = nn.Linear(cfg.action_dim, cfg.width)
        self.action_in_proj = nn.Linear(cfg.action_dim, cfg.width)
        self.action_time_mlp_in = nn.Linear(cfg.width * 2, cfg.width)
        self.action_time_mlp_out = nn.Linear(cfg.width, cfg.width)
        self.action_blocks = nn.ModuleList(
            [MiniTransformerBlock(cfg.width, cfg.num_heads) for _ in range(cfg.depth)]
        )
        self.action_out_proj = nn.Linear(cfg.width, cfg.action_dim)

    def _prefix(self, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        bsz = images.shape[0]
        img = images.reshape(bsz, self.cfg.num_images, -1)
        img_tokens = self.image_proj(img)
        text_tokens = self.text_embed(token_ids)
        prefix = torch.cat([img_tokens, text_tokens], dim=1)
        for block in self.prefix_blocks:
            prefix = block(prefix)
        return prefix

    def _denoise_step(
        self,
        prefix_context: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        state_token = self.state_proj(state)[:, None, :]
        action_tokens = self.action_in_proj(x_t)
        time_tokens = timestep[:, None, None].expand_as(action_tokens)
        time_tokens = torch.sin(time_tokens).expand_as(action_tokens)
        action_time = torch.cat([action_tokens, time_tokens], dim=-1)
        action_time = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time)))
        suffix = torch.cat([state_token, action_time], dim=1)

        # A compact stand-in for prefix KV attention: expose the prefix summary to
        # the suffix stream so ONNX contains the prefix -> action dependency.
        prefix_summary = prefix_context.mean(dim=1, keepdim=True)
        suffix = suffix + prefix_summary
        for block in self.action_blocks:
            suffix = block(suffix)
        return self.action_out_proj(suffix[:, -self.cfg.action_horizon :])

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        prefix_context = self._prefix(images, token_ids)
        x_t = noise
        dt = -1.0 / float(self.cfg.num_steps)
        for i in range(self.cfg.num_steps):
            t = torch.full((state.shape[0],), 1.0 + dt * i, dtype=x_t.dtype, device=x_t.device)
            v_t = self._denoise_step(prefix_context, state, x_t, t)
            x_t = x_t + dt * v_t
        return x_t


class MiniPi0PrefixWrapper(nn.Module):
    def __init__(self, model: MiniPi0FullInference):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model._prefix(images, token_ids)


class MiniPi0DenoiseStepWrapper(nn.Module):
    def __init__(self, model: MiniPi0FullInference):
        super().__init__()
        self.model = model

    def forward(
        self,
        prefix_context: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.model._denoise_step(prefix_context, state, x_t, timestep)


class MiniPi0EulerSamplerWrapper(nn.Module):
    def __init__(self, model: MiniPi0FullInference):
        super().__init__()
        self.model = model

    def forward(self, prefix_context: torch.Tensor, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        x_t = noise
        dt = -1.0 / float(self.model.cfg.num_steps)
        for i in range(self.model.cfg.num_steps):
            t = torch.full((state.shape[0],), 1.0 + dt * i, dtype=x_t.dtype, device=x_t.device)
            v_t = self.model._denoise_step(prefix_context, state, x_t, t)
            x_t = x_t + dt * v_t
        return x_t


def _install_openpi_export_monkeypatches(repo_root: Path) -> None:
    """Allow importing PI0Pytorch in a lean environment for experiments."""
    sys.path.insert(0, str(repo_root / "openpi-main" / "src"))

    import dataclasses
    import openpi

    models_pkg = types.ModuleType("openpi.models")
    models_pkg.__path__ = [str(repo_root / "openpi-main" / "src" / "openpi" / "models")]
    sys.modules["openpi.models"] = models_pkg
    setattr(openpi, "models", models_pkg)

    @dataclasses.dataclass
    class GemmaConfig:
        width: int
        depth: int
        mlp_dim: int
        num_heads: int
        num_kv_heads: int
        head_dim: int
        lora_configs: dict = dataclasses.field(default_factory=dict)

    def get_config(_variant: str) -> GemmaConfig:
        return GemmaConfig(width=64, depth=2, mlp_dim=128, num_heads=4, num_kv_heads=1, head_dim=16)

    gemma = types.ModuleType("openpi.models.gemma")
    gemma.Config = GemmaConfig
    gemma.get_config = get_config
    gemma.Variant = str
    sys.modules["openpi.models.gemma"] = gemma
    setattr(models_pkg, "gemma", gemma)

    prep = types.ModuleType("openpi.models_pytorch.preprocessing_pytorch")
    prep.preprocess_observation_pytorch = lambda obs, train=True: obs
    sys.modules["openpi.models_pytorch.preprocessing_pytorch"] = prep

    check = types.ModuleType("transformers.models.siglip.check")
    check.check_whether_transformers_replace_is_installed_correctly = lambda: True
    sys.modules["transformers.models.siglip.check"] = check


def _make_structural_model_and_inputs(args: argparse.Namespace):
    cfg = MiniPi0Config(
        image_size=args.image_size,
        token_len=args.token_len,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        width=args.width,
        depth=args.depth,
        num_heads=args.num_heads,
        num_steps=args.num_steps,
    )
    model = MiniPi0FullInference(cfg).eval()
    images = torch.randn(args.batch, cfg.num_images, 3, cfg.image_size, cfg.image_size)
    token_ids = torch.randint(0, cfg.vocab_size, (args.batch, cfg.token_len), dtype=torch.long)
    state = torch.randn(args.batch, cfg.action_dim)
    noise = torch.randn(args.batch, cfg.action_horizon, cfg.action_dim)
    timestep = torch.full((args.batch,), 1.0, dtype=torch.float32)
    prefix_context = model._prefix(images, token_ids).detach()
    return cfg, model, images, token_ids, state, noise, timestep, prefix_context


def _export_onnx(model, inputs, output, input_names, output_names, opset):
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        inputs,
        output,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamo=False,
    )
    print(f"wrote {output}")


def export_structural(args: argparse.Namespace) -> None:
    cfg, model, images, token_ids, state, noise, timestep, prefix_context = _make_structural_model_and_inputs(args)
    base = args.output_dir

    if args.scope in {"full", "all"}:
        _export_onnx(
            model,
            (images, token_ids, state, noise),
            args.output if args.scope == "full" else base / "full_inference_structural.onnx",
            ["images", "token_ids", "state", "noise"],
            ["action_chunk"],
            args.opset,
        )

    if args.scope in {"prefix", "all"}:
        _export_onnx(
            MiniPi0PrefixWrapper(model).eval(),
            (images, token_ids),
            base / "01_prefix_prefill_structural.onnx",
            ["images", "token_ids"],
            ["prefix_context"],
            args.opset,
        )

    if args.scope in {"denoise", "all"}:
        _export_onnx(
            MiniPi0DenoiseStepWrapper(model).eval(),
            (prefix_context, state, noise, timestep),
            base / "02_denoise_step_structural.onnx",
            ["prefix_context", "state", "x_t", "timestep"],
            ["velocity"],
            args.opset,
        )

    if args.scope in {"euler", "all"}:
        _export_onnx(
            MiniPi0EulerSamplerWrapper(model).eval(),
            (prefix_context, state, noise),
            base / "03_euler_sampler_structural.onnx",
            ["prefix_context", "state", "noise"],
            ["action_chunk"],
            args.opset,
        )


def export_openpi(args: argparse.Namespace) -> None:
    # This is intentionally experimental.  It keeps the openpi PI0Pytorch class,
    # but uses a tiny config and identity preprocessing.
    repo_root = Path(__file__).resolve().parents[2]
    _install_openpi_export_monkeypatches(repo_root)

    from types import SimpleNamespace
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    class Observation(SimpleNamespace):
        pass

    class Wrapper(nn.Module):
        def __init__(self, model: PI0Pytorch, num_steps: int):
            super().__init__()
            self.model = model
            self.num_steps = num_steps

        def forward(self, images: torch.Tensor, token_ids: torch.Tensor, state: torch.Tensor, noise: torch.Tensor):
            obs = Observation(
                images={f"image_{i}": images[:, i] for i in range(images.shape[1])},
                image_masks={f"image_{i}": torch.ones(images.shape[0], dtype=torch.bool, device=images.device) for i in range(images.shape[1])},
                tokenized_prompt=token_ids,
                tokenized_prompt_mask=torch.ones_like(token_ids, dtype=torch.bool),
                state=state,
            )
            return self.model.sample_actions(state.device, obs, noise=noise, num_steps=self.num_steps)

    cfg = SimpleNamespace(
        pi05=False,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_token_len=args.token_len,
        pytorch_compile_mode=None,
    )
    model = Wrapper(PI0Pytorch(cfg).eval(), args.num_steps).eval()
    images = torch.randn(args.batch, 3, 3, 224, 224)
    token_ids = torch.randint(0, 128, (args.batch, args.token_len), dtype=torch.long)
    state = torch.randn(args.batch, args.action_dim)
    noise = torch.randn(args.batch, args.action_horizon, args.action_dim)

    torch.onnx.export(
        model,
        (images, token_ids, state, noise),
        args.output,
        input_names=["images", "token_ids", "state", "noise"],
        output_names=["action_chunk"],
        opset_version=args.opset,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["structural", "openpi"], default="structural")
    parser.add_argument("--scope", choices=["full", "prefix", "denoise", "euler", "all"], default="full")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model_graphs/operator/pi0/full_inference_structural.onnx"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_graphs/operator/pi0"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--token-len", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "structural":
        export_structural(args)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_openpi(args)


if __name__ == "__main__":
    main()
