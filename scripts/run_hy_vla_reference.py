#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Hy-VLA deterministic Python reference harness.

This script creates a fixed Hy-VLA fixture and can optionally run the released
Python model on it.  The default mode intentionally stops before loading the
9GB model weights, so environment/tokenizer/preprocess issues can be debugged
without stressing memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_HY_REPO = Path("/home/xuling/robotic_code/embodied.cpp/Hy-Embodied-0.5-VLA-main")
DEFAULT_CKPT = Path("/home/xuling/robotic_dataset/HY-VLA")


IMAGE_KEYS = [
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
]


def deterministic_image(cam: int, frame: int, h: int, w: int) -> np.ndarray:
    y, x, c = np.meshgrid(
        np.arange(h, dtype=np.uint16),
        np.arange(w, dtype=np.uint16),
        np.arange(3, dtype=np.uint16),
        indexing="ij",
    )
    arr = (x * (3 + cam) + y * (5 + frame) + c * 37 + cam * 41 + frame * 19) % 256
    return np.ascontiguousarray(arr.astype(np.uint8))


def make_fixture(config, *, prompt: str, history: int, image_size: int):
    images = {}
    for cam, key in enumerate(IMAGE_KEYS):
        frames = []
        for frame in range(history):
            hwc = deterministic_image(cam, frame, image_size, image_size)
            chw01 = np.asarray(hwc, dtype=np.float32).transpose(2, 0, 1) / 255.0
            frames.append(chw01)
        images[key] = np.ascontiguousarray(np.stack(frames, axis=0)[None, ...], dtype=np.float32)

    state = np.linspace(-0.25, 0.25, int(config.max_state_dim), dtype=np.float32)[None, :]
    n = int(config.n_action_steps) * int(config.max_action_dim)
    noise = np.sin(np.arange(n, dtype=np.float32) * np.float32(0.017)).reshape(
        1, int(config.n_action_steps), int(config.max_action_dim)
    )
    noise = np.ascontiguousarray(noise * np.float32(0.05), dtype=np.float32)
    return {
        "task": prompt,
        "images": images,
        "state": np.ascontiguousarray(state, dtype=np.float32),
        "noise": noise,
    }


def save_fixture(fixture, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "hy_vla_fixture.npz",
        task=np.array(fixture["task"]),
        state=fixture["state"],
        noise=fixture["noise"],
        top_head=fixture["images"]["observation.images.top_head"],
        hand_left=fixture["images"]["observation.images.hand_left"],
        hand_right=fixture["images"]["observation.images.hand_right"],
    )


def write_array(out_dir: Path, name: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    arr.tofile(out_dir / f"{name}.{arr.dtype.name}")
    (out_dir / f"{name}.shape.txt").write_text(" ".join(str(x) for x in arr.shape) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hy-repo", type=Path, default=DEFAULT_HY_REPO)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/hy_vla_reference"))
    parser.add_argument("--prompt", default="pick up the bottle")
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mode", choices=["fixture", "load", "forward"], default="fixture")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["f32", "bf16"], default="bf16")
    args = parser.parse_args()

    sys.path.insert(0, str(args.hy_repo))
    import torch
    from transformers import AutoTokenizer
    from hy_vla import HyVLA, HyVLAConfig

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = HyVLAConfig.from_pretrained(args.ckpt)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=False)
    fixture = make_fixture(config, prompt=args.prompt, history=args.history, image_size=args.image_size)
    save_fixture(fixture, args.out_dir)

    task = fixture["task"]
    task_for_tokenizer = task if task.endswith("<｜hy_Assistant｜>") else f"{task}<｜hy_Assistant｜>"
    tokenized = tokenizer(
        [task_for_tokenizer],
        padding="max_length",
        padding_side="right",
        truncation=True,
        max_length=int(config.tokenizer_max_length),
        return_tensors="np",
        add_special_tokens=False,
        return_token_type_ids=True,
    )
    write_array(args.out_dir, "lang_tokens", tokenized["input_ids"].astype(np.int32))
    write_array(args.out_dir, "lang_attention_mask", tokenized["attention_mask"].astype(np.int32))
    write_array(args.out_dir, "state", fixture["state"])
    write_array(args.out_dir, "noise", fixture["noise"])

    summary = {
        "ckpt": str(args.ckpt),
        "mode": args.mode,
        "prompt": task,
        "history": args.history,
        "image_size": args.image_size,
        "chunk_size": int(config.chunk_size),
        "n_action_steps": int(config.n_action_steps),
        "max_state_dim": int(config.max_state_dim),
        "max_action_dim": int(config.max_action_dim),
        "token_checksum": int(tokenized["input_ids"].sum()),
        "state_checksum": float(fixture["state"].sum()),
        "noise_checksum": float(fixture["noise"].sum()),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    if args.mode == "fixture":
        print(f"[hy-vla-reference] fixture written to {args.out_dir}")
        return 0

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print("[hy-vla-reference] loading HyVLA model; this may take a while...")
    policy = HyVLA.from_pretrained(str(args.ckpt), config=config, map_location="cpu")
    policy.enable_video_encoder_if_needed()
    policy.eval()
    if args.device == "cuda":
        policy = policy.to(device="cuda", dtype=dtype)
    else:
        policy = policy.to(dtype=dtype)
    print("[hy-vla-reference] model loaded")

    if args.mode == "load":
        return 0

    batch = {"task": [task]}
    for key, arr in fixture["images"].items():
        batch[key] = torch.from_numpy(arr).to(device=args.device, dtype=dtype)
    batch["observation.state"] = torch.from_numpy(fixture["state"]).to(device=args.device, dtype=dtype)
    noise = torch.from_numpy(fixture["noise"]).to(device=args.device, dtype=torch.float32)

    with torch.no_grad():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens, lang_masks, _ = policy.prepare_language(batch)
        actions = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
        actions = actions[..., : config.action_feature.shape[0]]

    actions_np = actions.float().cpu().numpy()
    write_array(args.out_dir, "pred_action_chunk", actions_np.astype(np.float32))
    print(
        "[hy-vla-reference] forward ok "
        f"shape={actions_np.shape} checksum={float(actions_np.sum()):.9g} max={float(np.abs(actions_np).max()):.9g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
