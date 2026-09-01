#!/usr/bin/env python3
"""Generate a deterministic PyTorch reference for TurboVLA C++ parity."""

from __future__ import annotations

import argparse
import copy
import sys
import types
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertModel
from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel


def provide_eval_only_timm_shim() -> None:
    """TurboVLA only needs timm DropPath, which is identity in eval mode."""
    try:
        import timm.models.layers  # noqa: F401, PLC0415
        return
    except ImportError:
        pass

    class DropPath(torch.nn.Module):
        def __init__(self, drop_prob: float = 0.0) -> None:
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            if self.training and self.drop_prob:
                raise RuntimeError("eval-only timm shim cannot execute stochastic DropPath")
            return value

    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")
    layers.DropPath = DropPath
    models.layers = layers
    timm.models = models
    sys.modules.update({"timm": timm, "timm.models": models, "timm.models.layers": layers})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--bert-path", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--instruction",
        default="pick up the black bowl next to the cookie box and place it on the plate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.official_root))
    provide_eval_only_timm_shim()

    from turbovla.evaluation.policy import (  # noqa: PLC0415
        ACTION_MAX,
        ACTION_MIN,
        PROPRIO_MEAN,
        PROPRIO_STD,
    )
    from turbovla.models import text_encoder, vision_encoder  # noqa: PLC0415
    from turbovla.models.configuration import TurboVLAConfig  # noqa: PLC0415
    from turbovla.models.turbovla import build_turbovla  # noqa: PLC0415

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config = TurboVLAConfig.from_mapping(copy.deepcopy(checkpoint["model_config"]))
    config.text.model_name_or_path = str(args.bert_path)
    config.text.local_files_only = True
    config.vision.model_name_or_path = "offline-dinov3-vitb16"
    config.vision.local_files_only = True
    config.vision.compute_precision = "bf16" if args.precision == "bf16" else "bf16_autocast"

    bert_config = BertConfig.from_pretrained(args.bert_path, local_files_only=True)
    dino_config = DINOv3ViTConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        num_register_tokens=4,
        image_size=config.vision.image_size,
        patch_size=16,
        rope_theta=100.0,
        layerscale_value=1.0,
    )
    text_encoder._load_pretrained_model = lambda _: BertModel(bert_config)
    vision_encoder._load_pretrained_model = lambda _: DINOv3ViTModel(dino_config)

    model = build_turbovla(config)
    model.load_state_dict(state_dict, strict=True)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    device = torch.device(args.device)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    stages: dict[str, np.ndarray] = {}

    def capture(name: str, transform=lambda value: value):
        def hook(_module, _inputs, output):
            value = transform(output)
            stages[name] = value.detach().float().cpu().numpy()
        return hook

    model.vision_encoder.register_forward_hook(capture("dino"))
    model.vision_projection.register_forward_hook(capture("vision_projection"))
    model.text_encoder.bert.register_forward_hook(
        capture("bert", lambda output: output.last_hidden_state)
    )
    model.text_encoder.text_projection.register_forward_hook(capture("text_projected"))
    model.vision_language_interaction.register_forward_hook(
        capture("fused", lambda output: torch.cat(output, dim=1))
    )
    model.action_head.state_projection.register_forward_hook(capture("state_tokens"))

    rng = np.random.default_rng(20260819)
    images_u8 = rng.integers(0, 256, size=(2, 256, 256, 3), dtype=np.uint8)
    state_norm = np.linspace(-0.5, 0.5, 8, dtype=np.float32)
    state = np.asarray(PROPRIO_MEAN, dtype=np.float32) + (
        np.asarray(PROPRIO_STD, dtype=np.float32) * state_norm
    )
    images = images_u8.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    pixels = (images - mean) / std
    pixels = torch.from_numpy(pixels.transpose(0, 3, 1, 2)[None]).to(device=device, dtype=dtype)
    state_tensor = torch.from_numpy(state_norm[None]).to(device=device, dtype=dtype)

    tokenized, _, _ = model.text_encoder._tokenize_group(
        [args.instruction], device, config.text.padding_length
    )
    token_ids = tokenized.input_ids[0].cpu().numpy()
    token_valid = tokenized.attention_mask[0].cpu().numpy().astype(bool)
    with torch.inference_mode():
        normalized = model([args.instruction], {"dinov3": pixels}, state_tensor)
    normalized = normalized[0].float().cpu().numpy()
    env = normalized.copy()
    env[:, :6] = 0.5 * (env[:, :6] + 1.0) * (
        np.asarray(ACTION_MAX[:6]) - np.asarray(ACTION_MIN[:6])
    ) + np.asarray(ACTION_MIN[:6])
    env[:, 6] = np.where(env[:, 6] >= 0.0, 1.0, -1.0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "turbovla_parity_inputs.npz",
        images_chw=images.transpose(0, 3, 1, 2),
        state=state,
        instruction=np.asarray(args.instruction),
        token_ids=token_ids,
        token_valid=token_valid,
    )
    np.save(args.out_dir / "turbovla_parity_ref_normalized.npy", normalized)
    np.save(args.out_dir / "turbovla_parity_ref_env.npy", env)
    projected = stages.pop("vision_projection")
    view = model.view_embedding.detach().float().cpu().numpy()[:, :, None, :]
    stages["vision_projected"] = (projected + view).reshape(1, -1, projected.shape[-1])
    fused = stages.pop("fused")
    stages["vision_fused"] = fused[:, :512]
    stages["text_fused"] = fused[:, 512:]
    np.savez(args.out_dir / "turbovla_parity_ref_stages.npz", **stages)
    print(f"tokens={token_ids.tolist()}")
    print(f"valid={int(token_valid.sum())}/{len(token_valid)} pads={int((~token_valid).sum())}")
    print(f"normalized[0]={normalized[0].tolist()}")
    print(f"env[0]={env[0].tolist()}")


if __name__ == "__main__":
    main()
