#!/usr/bin/env python3
"""Export original-source, random-weight pi0 module ONNX graphs.

These exports are for operator-graph inspection.  They instantiate modules that
match the openpi/PaliGemma pi0 source structure, but do not load checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
import types

import torch
from torch import nn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _install_openpi_monkeypatches() -> None:
    """Import PI0Pytorch without JAX/Flax and patch transformer API drift."""
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root / "openpi-main" / "src"))

    import openpi

    models_pkg = types.ModuleType("openpi.models")
    models_pkg.__path__ = [str(repo_root / "openpi-main" / "src" / "openpi" / "models")]
    sys.modules["openpi.models"] = models_pkg
    setattr(openpi, "models", models_pkg)

    @dataclass
    class GemmaConfig:
        width: int
        depth: int
        mlp_dim: int
        num_heads: int
        num_kv_heads: int
        head_dim: int
        lora_configs: dict = field(default_factory=dict)

    def get_config(variant: str) -> GemmaConfig:
        if "300m" in variant:
            return GemmaConfig(width=1024, depth=1, mlp_dim=2048, num_heads=8, num_kv_heads=1, head_dim=256)
        return GemmaConfig(width=2048, depth=1, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)

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

    from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel

    PaliGemmaWithExpertModel.embed_image = lambda self, image: self.paligemma.get_image_features(image)
    PaliGemmaWithExpertModel.embed_language_tokens = (
        lambda self, tokens: self.paligemma.get_input_embeddings()(tokens)
    )


def make_tiny_paligemma(vision_layers: int, text_layers: int, vocab_size: int):
    """Build random PaliGemma with pi0-compatible width and configurable depth."""
    from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration

    cfg = PaliGemmaConfig()
    cfg.vision_config.num_hidden_layers = vision_layers
    cfg.vision_config.projection_dim = 2048
    cfg.text_config.hidden_size = 2048
    cfg.text_config.intermediate_size = 4096
    cfg.text_config.num_hidden_layers = text_layers
    cfg.text_config.num_attention_heads = 8
    cfg.text_config.num_key_value_heads = 1
    cfg.text_config.head_dim = 256
    cfg.text_config.vocab_size = vocab_size
    cfg.vocab_size = vocab_size
    return PaliGemmaForConditionalGeneration(cfg).eval()


class Pi0ImageEncoderOriginalRandom(nn.Module):
    def __init__(self, vision_layers: int, text_layers: int, vocab_size: int):
        super().__init__()
        self.model = make_tiny_paligemma(vision_layers, text_layers, vocab_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values)


class Pi0LanguageEmbedOriginalRandom(nn.Module):
    def __init__(self, vision_layers: int, text_layers: int, vocab_size: int):
        super().__init__()
        self.model = make_tiny_paligemma(vision_layers, text_layers, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.model.get_input_embeddings()(input_ids)
        return emb * (emb.shape[-1] ** 0.5)


class Pi0PrefixPackOriginalRandom(nn.Module):
    def __init__(self, vision_layers: int, text_layers: int, vocab_size: int, num_images: int):
        super().__init__()
        self.num_images = num_images
        self.image_encoder = Pi0ImageEncoderOriginalRandom(vision_layers, text_layers, vocab_size)
        self.language_embed = Pi0LanguageEmbedOriginalRandom(vision_layers, text_layers, vocab_size)

    def forward(
        self,
        images: torch.Tensor,
        image_masks: torch.Tensor,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ):
        embs = []
        pad_masks = []
        att_masks = []
        for i in range(self.num_images):
            img_emb = self.image_encoder(images[:, i])
            embs.append(img_emb)
            pad_masks.append(image_masks[:, i : i + 1].expand(images.shape[0], img_emb.shape[1]))
            att_masks.append(torch.zeros((images.shape[0], img_emb.shape[1]), dtype=torch.bool, device=images.device))
        lang_emb = self.language_embed(input_ids)
        embs.append(lang_emb)
        pad_masks.append(token_mask)
        att_masks.append(torch.zeros_like(token_mask, dtype=torch.bool))
        return torch.cat(embs, dim=1), torch.cat(pad_masks, dim=1), torch.cat(att_masks, dim=1)


class Pi0PrefixPrefillOriginalRandom(nn.Module):
    def __init__(self, vision_layers: int, text_layers: int, vocab_size: int, num_images: int):
        super().__init__()
        self.pack = Pi0PrefixPackOriginalRandom(vision_layers, text_layers, vocab_size, num_images)
        self.lm = self.pack.language_embed.model.language_model.model

    @staticmethod
    def make_att_2d_masks(pad_masks, att_masks):
        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    def forward(
        self,
        images: torch.Tensor,
        image_masks: torch.Tensor,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        prefix_embs, pad_masks, att_masks = self.pack(images, image_masks, input_ids, token_mask)
        att_2d = self.make_att_2d_masks(pad_masks, att_masks)
        att_4d = torch.where(att_2d[:, None, :, :], 0.0, -2.3819763e38)
        pos = torch.cumsum(pad_masks, dim=1) - 1
        return self.lm(inputs_embeds=prefix_embs, attention_mask=att_4d, position_ids=pos).last_hidden_state


class Pi0SuffixEmbedOriginalRandom(nn.Module):
    def __init__(self):
        super().__init__()
        _install_openpi_monkeypatches()
        from types import SimpleNamespace
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        cfg = SimpleNamespace(
            pi05=False,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            dtype="float32",
            action_dim=4,
            action_horizon=3,
            max_token_len=5,
            pytorch_compile_mode=None,
        )
        self.model = PI0Pytorch(cfg).eval()

    def forward(self, state: torch.Tensor, x_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        suffix_embs, _, _, _ = self.model.embed_suffix(state, x_t, timestep)
        return suffix_embs


def export_onnx(model: nn.Module, inputs, output: Path, input_names, output_names, opset: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["image", "language", "prefix_pack", "prefix_prefill", "suffix_embed", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_graphs/operator/pi0"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--token-len", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--vision-layers", type=int, default=2)
    parser.add_argument("--text-layers", type=int, default=1)
    args = parser.parse_args()

    images = torch.randn(args.batch, args.num_images, 3, args.image_size, args.image_size)
    image = images[:, 0]
    image_masks = torch.ones(args.batch, args.num_images, dtype=torch.bool)
    input_ids = torch.randint(0, min(args.vocab_size, 128), (args.batch, args.token_len), dtype=torch.long)
    token_mask = torch.ones(args.batch, args.token_len, dtype=torch.bool)
    state = torch.randn(args.batch, 4)
    x_t = torch.randn(args.batch, 3, 4)
    timestep = torch.ones(args.batch)

    if args.scope in {"image", "all"}:
        export_onnx(
            Pi0ImageEncoderOriginalRandom(args.vision_layers, args.text_layers, args.vocab_size),
            (image,),
            args.output_dir / "01_image_encoder_original_random.onnx",
            ["image"],
            ["image_tokens"],
            args.opset,
        )
    if args.scope in {"language", "all"}:
        export_onnx(
            Pi0LanguageEmbedOriginalRandom(args.vision_layers, args.text_layers, args.vocab_size),
            (input_ids,),
            args.output_dir / "02_language_embed_original_random.onnx",
            ["input_ids"],
            ["language_embeddings"],
            args.opset,
        )
    if args.scope in {"prefix_pack", "all"}:
        export_onnx(
            Pi0PrefixPackOriginalRandom(args.vision_layers, args.text_layers, args.vocab_size, args.num_images),
            (images, image_masks, input_ids, token_mask),
            args.output_dir / "03_prefix_pack_original_random.onnx",
            ["images", "image_masks", "input_ids", "token_mask"],
            ["prefix_embeddings", "prefix_pad_mask", "prefix_attention_ar_mask"],
            args.opset,
        )
    if args.scope in {"prefix_prefill", "all"}:
        export_onnx(
            Pi0PrefixPrefillOriginalRandom(args.vision_layers, args.text_layers, args.vocab_size, args.num_images),
            (images, image_masks, input_ids, token_mask),
            args.output_dir / "04_prefix_prefill_original_random.onnx",
            ["images", "image_masks", "input_ids", "token_mask"],
            ["prefix_hidden_states"],
            args.opset,
        )
    if args.scope in {"suffix_embed", "all"}:
        export_onnx(
            Pi0SuffixEmbedOriginalRandom(),
            (state, x_t, timestep),
            args.output_dir / "05_suffix_embed_original_random.onnx",
            ["state", "x_t", "timestep"],
            ["suffix_embeddings"],
            args.opset,
        )


if __name__ == "__main__":
    main()
