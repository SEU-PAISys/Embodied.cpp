#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""End-to-end-first LingBot-VA parity harness.

The harness keeps the parity contract at the final action chunk first.  It can
generate a deterministic LIBERO-like observation, send it to the C++
`lingbot-world-server`, and optionally run the Python reference implementation
from the original LingBot-VA repository.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import types
import importlib.machinery
import gc
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = Path("/home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long")
DEFAULT_LINGBOT_REPO = Path("/home/xuling/robotic_code/embodied.cpp/lingbot-va-main")
DEFAULT_TRANSFORMER_GGUF = Path("/home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf")
DEFAULT_TEXT_GGUF = Path("/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf")
DEFAULT_VAE_GGUF = Path("/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf")


@dataclass
class Fixture:
    index: int
    obs: dict[str, Any]
    action_noise: np.ndarray


PROMPTS = [
    "pick up the alphabet soup and place it in the basket",
    "pick up the tomato sauce and place it in the basket",
    "put the cream cheese in the basket",
    "pick up the butter and place it in the basket",
    "put the chocolate pudding in the basket",
    "pick up the orange juice and place it in the basket",
    "put the salad dressing in the basket",
    "pick up the ketchup and place it in the basket",
    "put the black bowl on the plate",
    "pick up the mug and place it on the tray",
]


def deterministic_image(seed: int, h: int = 128, w: int = 128) -> np.ndarray:
    y, x, c = np.meshgrid(
        np.arange(h, dtype=np.uint16),
        np.arange(w, dtype=np.uint16),
        np.arange(3, dtype=np.uint16),
        indexing="ij",
    )
    img = (x * 3 + y * 5 + c * 37 + seed * 17) % 256
    return np.ascontiguousarray(img.astype(np.uint8))


def deterministic_action_noise(
    sample_id: int,
    steps: int = 16,
    dim: int = 30,
    scale: float = 0.03,
) -> np.ndarray:
    n = steps * dim
    phase = np.float32(sample_id) * np.float32(0.071)
    x = np.sin(np.arange(n, dtype=np.float32) * np.float32(0.013) + phase) * np.float32(scale)
    return np.ascontiguousarray(x.reshape(steps, dim), dtype=np.float32)


def make_fixture(prompt: str, sample_id: int = 0) -> Fixture:
    pos = np.array([
        0.05 + 0.007 * sample_id,
        -0.10 + 0.003 * sample_id,
        0.20 - 0.002 * sample_id,
    ], dtype=np.float32)
    gripper = np.array([(-1.0) ** sample_id * 0.02], dtype=np.float32)
    obs = {
        "pixels": {
            "image": deterministic_image(1 + 2 * sample_id),
            "image2": deterministic_image(2 + 2 * sample_id),
        },
        "robot_state": {
            "eef": {
                "pos": pos,
                "quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            },
            "gripper": {"qpos": gripper},
        },
        "task_description": prompt,
    }
    return Fixture(index=sample_id, obs=obs, action_noise=deterministic_action_noise(sample_id))


def make_fixtures(prompt: str | None, num_fixtures: int) -> list[Fixture]:
    fixtures = []
    for i in range(num_fixtures):
        prompt_i = prompt if prompt is not None else PROMPTS[i % len(PROMPTS)]
        fixtures.append(make_fixture(prompt_i, i))
    return fixtures


def save_fixture(fixture: Fixture, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state = np.concatenate([
        fixture.obs["robot_state"]["eef"]["pos"],
        np.zeros(3, dtype=np.float32),
        fixture.obs["robot_state"]["gripper"]["qpos"],
    ]).astype(np.float32)
    np.savez(
        out_dir / f"fixture_{fixture.index:03d}.npz",
        image=fixture.obs["pixels"]["image"],
        image2=fixture.obs["pixels"]["image2"],
        action_noise=fixture.action_noise,
        state=state,
        prompt=np.array(fixture.obs["task_description"]),
    )


def wait_for_server(proc: subprocess.Popen[str], timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    captured: list[str] = []
    while time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout is not None else ""
        if line:
            captured.append(line.rstrip())
            print(line, end="")
            if "ready." in line:
                return
        if proc.poll() is not None:
            raise RuntimeError("lingbot-world-server exited before ready:\n" + "\n".join(captured[-80:]))
    raise TimeoutError("timed out waiting for lingbot-world-server")


def start_server_log_tee(proc: subprocess.Popen[str], log_path: Path) -> threading.Thread:
    """Keep draining server stdout after the readiness banner."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _reader() -> None:
        with log_path.open("a", encoding="utf-8") as f:
            while True:
                line = proc.stdout.readline() if proc.stdout is not None else ""
                if not line:
                    break
                print(line, end="")
                f.write(line)
                f.flush()

    thread = threading.Thread(target=_reader, name="lingbot-server-log", daemon=True)
    thread.start()
    return thread


def install_flash_attn_stubs() -> None:
    def _flash_attn_unavailable(*_args, **_kwargs):
        raise RuntimeError("flash_attn stub was called; parity harness expects non-flash attention")

    flash_stub = types.ModuleType("flash_attn")
    flash_stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash_stub.flash_attn_func = _flash_attn_unavailable
    flash_interface_stub = types.ModuleType("flash_attn_interface")
    flash_interface_stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn_interface", loader=None)
    flash_interface_stub.flash_attn_func = _flash_attn_unavailable
    sys.modules.setdefault("flash_attn", flash_stub)
    sys.modules.setdefault("flash_attn_interface", flash_interface_stub)


def resize_hwc_u8(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return np.ascontiguousarray(img, dtype=np.uint8)
    pil = Image.fromarray(np.asarray(img, dtype=np.uint8), mode="RGB")
    pil = pil.resize((size, size), resample=Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))


def _copy_cpp_final_dump(dump_dir: Path, sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    for src in dump_dir.glob("lingbot_predict_*"):
        dst = sample_dir / src.name
        dst.write_bytes(src.read_bytes())


def run_cpp(args: argparse.Namespace, fixtures: list[Fixture]) -> Path:
    from eval.client.lingbot_world_client import LingBotWorldClient

    dump_dir = args.out_dir / "cpp"
    dump_dir.mkdir(parents=True, exist_ok=True)
    server_cmd = [
        str(REPO / "build" / "lingbot-world-server"),
        "--bind",
        args.addr,
        str(args.transformer_gguf),
    ]
    env = os.environ.copy()
    env.update(
        {
            "VLA_LINGBOT_PREDICT_DUMP_DIR": str(dump_dir),
            "VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR": str(dump_dir),
            "VLA_LINGBOT_TEXT_GGUF": str(args.text_gguf),
            "VLA_LINGBOT_VAE_GGUF": str(args.vae_gguf),
            "VLA_LINGBOT_PREDICT_TEXT_ENCODER": "1",
            "VLA_LINGBOT_PREDICT_TEXT_BLOCKS": str(args.text_blocks),
            "VLA_LINGBOT_PREDICT_BLOCKS": str(args.blocks),
            "VLA_LINGBOT_PREDICT_VIDEO_STEPS": str(args.video_steps),
            "VLA_LINGBOT_PREDICT_ACTION_STEPS": str(args.action_steps),
            "VLA_LINGBOT_VIDEO_GUIDANCE_SCALE": str(args.guidance_scale),
            "VLA_LINGBOT_RESIDENT_BLOCK_CACHE": "1",
            "VLA_LINGBOT_RESIDENT_BLOCK_DTYPE": args.resident_dtype,
            "VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX": str(args.resident_blocks),
        }
    )
    if args.cuda_self_attn:
        env["VLA_LINGBOT_PREDICT_CUDA_SELF_ATTN"] = "1"
        env.setdefault("VLA_LINGBOT_PREDICT_MIXED", "1")

    proc = subprocess.Popen(
        server_cmd,
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        wait_for_server(proc)
        server_log_thread = start_server_log_tee(proc, dump_dir / "server.log")
        client = LingBotWorldClient(
            vla_addr=args.addr.replace("*", "127.0.0.1"),
            tokenizer_name=str(args.model_root / "tokenizer"),
            image_size=128,
            image_keys=("image", "image2"),
            max_length=512,
            recv_timeout_ms=args.recv_timeout_ms,
            session_id=args.session_id,
            max_cache_frames=4,
        )
        summary_lines = []
        for fixture in fixtures:
            client.session_id = args.session_id + fixture.index
            client.reset()
            action = client.predict_chunk(fixture.obs, action_noise=fixture.action_noise)
            sample_dir = dump_dir / f"sample_{fixture.index:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.asarray(action, dtype=np.float32).tofile(sample_dir / "client_action_chunk.f32")
            (sample_dir / "client_action_chunk.shape.txt").write_text(f"{action.shape[0]} {action.shape[1]}\n")
            _copy_cpp_final_dump(dump_dir, sample_dir)
            checksum = float(np.asarray(action, dtype=np.float32).sum())
            max_abs = float(np.abs(action).max())
            summary_lines.append(
                f"{fixture.index:03d} shape={action.shape[0]}x{action.shape[1]} "
                f"checksum={checksum:.9g} max={max_abs:.9g} prompt={fixture.obs['task_description']}"
            )
            print(f"[cpp] sample={fixture.index:03d} checksum={checksum:.9g} max={max_abs:.9g}")
        (dump_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        if "server_log_thread" in locals():
            server_log_thread.join(timeout=2)
    return dump_dir


def action_noise_to_python_tensor(noise: np.ndarray) -> "Any":
    import torch

    # C++ consumes [T,30] and maps t -> (frame=t//4, h=t%4).
    return torch.from_numpy(noise.T.reshape(1, 30, 4, 4, 1).copy())


def run_python_reference(args: argparse.Namespace, fixture: Fixture) -> Path:
    lingbot_root = args.lingbot_repo
    install_flash_attn_stubs()

    sys.path.insert(0, str(lingbot_root / "wan_va"))
    from configs import VA_CONFIGS
    from wan_va_server import VA_Server

    import torch


    out_dir = args.out_dir / "python"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = VA_CONFIGS["libero"]
    cfg.wan22_pretrained_model_name_or_path = str(args.model_root)
    cfg.local_rank = 0
    cfg.rank = 0
    cfg.world_size = 1
    cfg.save_root = str(out_dir / "save")
    cfg.enable_offload = args.python_offload
    cfg.guidance_scale = args.guidance_scale
    cfg.action_guidance_scale = args.action_guidance_scale
    cfg.num_inference_steps = args.video_steps
    cfg.action_num_inference_steps = args.action_steps
    cfg.video_exec_step = -1
    cfg.param_dtype = torch.bfloat16 if args.python_dtype == "bf16" else torch.float32

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.python_device == "cpu":
        from modules.utils import WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_transformer, load_vae
        from utils import FlowMatchScheduler

        server = object.__new__(VA_Server)
        server.cache_name = "pos"
        server.job_config = cfg
        server.save_root = cfg.save_root
        server.dtype = cfg.param_dtype
        server.device = torch.device("cpu")
        server.enable_offload = True
        server.scheduler = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)
        server.action_scheduler = FlowMatchScheduler(shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        server.scheduler.set_timesteps(1000, training=True)
        server.action_scheduler.set_timesteps(1000, training=True)
        server.vae = load_vae(str(args.model_root / "vae"), torch_dtype=server.dtype, torch_device="cpu")
        server.streaming_vae = WanVAEStreamingWrapper(server.vae)
        server.tokenizer = load_tokenizer(str(args.model_root / "tokenizer"))
        server.text_encoder = load_text_encoder(str(args.model_root / "text_encoder"),
                                                torch_dtype=server.dtype,
                                                torch_device="cpu")
        server.transformer = load_transformer(str(args.model_root / "transformer"),
                                              torch_dtype=server.dtype,
                                              torch_device="cpu",
                                              attn_mode="torch")
        server.transformer.eval().requires_grad_(False)
        server.env_type = cfg.env_type
        server.streaming_vae_half = None
    else:
        server = VA_Server(cfg)
    server._reset(prompt=fixture.obs["task_description"])

    py_obs = {
        "obs": [
            {
                "observation.images.agentview_rgb": fixture.obs["pixels"]["image"],
                "observation.images.eye_in_hand_rgb": fixture.obs["pixels"]["image2"],
            }
        ],
    }
    init_latent = server._encode_obs(py_obs)
    frame_chunk_size = cfg.frame_chunk_size
    latents = torch.zeros(
        1,
        48,
        frame_chunk_size,
        server.latent_height,
        server.latent_width,
        device=server.device,
        dtype=server.dtype,
    )
    actions = action_noise_to_python_tensor(fixture.action_noise).to(device=server.device, dtype=server.dtype)

    # Match the C++ current smoke contract: one non-CFG path and explicit action noise.
    server.use_cfg = False
    server.guidance_scale = 1
    server.action_guidance_scale = 1

    server.scheduler.set_timesteps(args.video_steps)
    server.action_scheduler.set_timesteps(args.action_steps)
    timesteps = torch.nn.functional.pad(server.scheduler.timesteps, (0, 1), mode="constant", value=0)
    action_timesteps = torch.nn.functional.pad(server.action_scheduler.timesteps, (0, 1), mode="constant", value=0)

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            last_step = i == len(timesteps) - 1
            latent_cond = init_latent[:, :, 0:1]
            input_dict = server._prepare_latent_input(latents, None, t, t, latent_cond, None, frame_st_id=0)
            pred = server.transformer(server._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                                      update_cache=1 if last_step else 0,
                                      cache_name=server.cache_name,
                                      action_mode=False)
            if not last_step:
                from wan_va.utils import data_seq_to_patch

                pred = data_seq_to_patch(cfg.patch_size, pred, frame_chunk_size,
                                         server.latent_height, server.latent_width, batch_size=1)
                latents = server.scheduler.step(pred, t, latents, return_dict=False)
            latents[:, :, 0:1] = latent_cond

        for i, t in enumerate(action_timesteps):
            last_step = i == len(action_timesteps) - 1
            action_cond = torch.zeros([1, cfg.action_dim, 1, cfg.action_per_frame, 1],
                                      device=server.device, dtype=server.dtype)
            input_dict = server._prepare_latent_input(None, actions, t, t, None, action_cond, frame_st_id=0)
            pred = server.transformer(server._repeat_input_for_cfg(input_dict["action_res_lst"]),
                                      update_cache=1 if last_step else 0,
                                      cache_name=server.cache_name,
                                      action_mode=True)
            if not last_step:
                from einops import rearrange

                pred = rearrange(pred, "b (f n) c -> b c f n 1", f=frame_chunk_size)
                actions = server.action_scheduler.step(pred, t, actions, return_dict=False)
            actions[:, :, 0:1] = action_cond

    actions[:, ~server.action_mask] *= 0
    py_action = server.postprocess_action(actions)
    py_action = np.asarray(py_action, dtype=np.float32).reshape(7, -1).T.copy()
    py_action.tofile(out_dir / "python_action_chunk.f32")
    (out_dir / "python_action_chunk.shape.txt").write_text(f"{py_action.shape[0]} {py_action.shape[1]}\n")
    return out_dir


def _torch_dtype(name: str) -> "Any":
    import torch

    return torch.bfloat16 if name == "bf16" else torch.float32


def _write_array_with_shape(path_prefix: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.tofile(path_prefix.with_suffix(".f32"))
    path_prefix.with_suffix(".shape.txt").write_text(" ".join(str(x) for x in arr.shape) + "\n")


def _truncate_umt5_blocks(text_encoder: "Any", n_blocks: int) -> None:
    if n_blocks < 0:
        return
    blocks = text_encoder.encoder.block
    if n_blocks > len(blocks):
        raise ValueError(f"--text-blocks {n_blocks} exceeds UMT5 layer count {len(blocks)}")
    if n_blocks < len(blocks):
        text_encoder.encoder.block = blocks[:n_blocks]


def run_python_modular_reference(args: argparse.Namespace, fixtures: list[Fixture]) -> Path:
    """Run lightweight Python reference modules for the 10-fixture suite.

    This intentionally avoids the full LingBot Python server path.  It validates
    the two expensive input-side modules that feed the C++ Wan/action bridge:
    UMT5 text encoding and Wan VAE image encoding.
    """
    import torch
    from diffusers import AutoencoderKLWan
    from transformers import T5TokenizerFast, UMT5EncoderModel

    out_dir = args.out_dir / "python_modular"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.python_device)
    dtype = _torch_dtype(args.python_dtype)

    tok = T5TokenizerFast.from_pretrained(args.model_root / "tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.model_root / "text_encoder",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    _truncate_umt5_blocks(text_encoder, args.text_blocks)
    text_encoder.eval().requires_grad_(False)
    with torch.no_grad():
        for fixture in fixtures:
            toks = tok(
                fixture.obs["task_description"],
                padding=False,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = toks.input_ids.to(device)
            mask = toks.attention_mask.to(device)
            hidden = text_encoder(input_ids, mask).last_hidden_state[0, : int(mask.sum().item())]
            sample_dir = out_dir / f"sample_{fixture.index:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _write_array_with_shape(sample_dir / "python_text_emb_raw", hidden.float().cpu().numpy())
            print(f"[python-modular:text] sample={fixture.index:03d} checksum={float(hidden.float().sum()):.9g}")
    del text_encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    vae = AutoencoderKLWan.from_pretrained(
        args.model_root / "vae",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    vae.eval().requires_grad_(False)
    enc_conv_num = getattr(vae, "_cached_conv_counts", {}).get("encoder")
    if enc_conv_num is None:
        enc_conv_num = sum(1 for m in vae.encoder.modules() if m.__class__.__name__ == "WanCausalConv3d")
    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)

    def patchify(x: "torch.Tensor", patch_size: int | None) -> "torch.Tensor":
        if patch_size is None or patch_size == 1:
            return x
        b, c, f, h, w = x.shape
        x = x.view(b, c, f, h // patch_size, patch_size, w // patch_size, patch_size)
        x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
        return x.view(b, c * patch_size * patch_size, f, h // patch_size, w // patch_size)

    with torch.no_grad():
        for fixture in fixtures:
            views = []
            for key in ("image", "image2"):
                img = np.asarray(fixture.obs["pixels"][key])
                img = np.ascontiguousarray(img[::-1, ::-1], dtype=np.uint8)
                img = resize_hwc_u8(img, 128)
                chw = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(1)
                views.append(chw)
            videos = torch.stack(views, dim=0).to(device=device, dtype=dtype) / 255.0 * 2.0 - 1.0
            videos = patchify(videos, getattr(vae.config, "patch_size", None))
            feat_cache = [None] * int(enc_conv_num)
            enc = vae.quant_conv(vae.encoder(videos, feat_cache=feat_cache, feat_idx=[0]))
            mu, _logvar = torch.chunk(enc, 2, dim=1)
            mu_norm = ((mu.float() - latents_mean) / latents_std).to(mu)
            video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
            sample_dir = out_dir / f"sample_{fixture.index:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            _write_array_with_shape(sample_dir / "python_vae_latent_raw", video_latent.float().cpu().numpy())
            print(f"[python-modular:vae] sample={fixture.index:03d} checksum={float(video_latent.float().sum()):.9g}")

    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return out_dir


def read_shape(path: Path) -> tuple[int, ...]:
    return tuple(int(x) for x in path.read_text().split())


def compare_outputs(args: argparse.Namespace) -> None:
    cpp_dir = args.out_dir / "cpp"
    py_dir = args.out_dir / "python"
    sample_dirs = sorted(cpp_dir.glob("sample_*"))
    if sample_dirs:
        rows = []
        for sample_dir in sample_dirs:
            py_sample_dir = py_dir / sample_dir.name
            cpp_path = sample_dir / "lingbot_predict_action_chunk.f32"
            py_path = py_sample_dir / "python_action_chunk.f32"
            if not cpp_path.exists() or not py_path.exists():
                rows.append(f"{sample_dir.name}: missing cpp={cpp_path.exists()} python={py_path.exists()}")
                continue
            cpp_shape = read_shape(sample_dir / "lingbot_predict_action_chunk.shape.txt")
            py_shape = read_shape(py_sample_dir / "python_action_chunk.shape.txt")
            cpp = np.fromfile(cpp_path, dtype=np.float32).reshape(cpp_shape)
            py = np.fromfile(py_path, dtype=np.float32).reshape(py_shape)
            if cpp.shape != py.shape:
                rows.append(f"{sample_dir.name}: shape mismatch cpp={cpp.shape} python={py.shape}")
                continue
            diff = np.abs(cpp - py)
            rows.append(
                f"{sample_dir.name}: max_abs_diff={diff.max():.9g} "
                f"mean_abs_diff={diff.mean():.9g} cpp_checksum={cpp.sum():.9g} py_checksum={py.sum():.9g}"
            )
        text = "\n".join(rows)
        print(text)
        (args.out_dir / "compare_summary.txt").write_text(text + "\n")
        return

    cpp_path = cpp_dir / "lingbot_predict_action_chunk.f32"
    if not cpp_path.exists():
        cpp_path = cpp_dir / "client_action_chunk.f32"
    py_path = py_dir / "python_action_chunk.f32"
    if not cpp_path.exists() or not py_path.exists():
        print(f"[compare] missing outputs: cpp={cpp_path.exists()} python={py_path.exists()}")
        return
    cpp_shape = read_shape(cpp_path.with_suffix(".shape.txt")) if cpp_path.name.startswith("client_") else read_shape(cpp_dir / "lingbot_predict_action_chunk.shape.txt")
    py_shape = read_shape(py_dir / "python_action_chunk.shape.txt")
    cpp = np.fromfile(cpp_path, dtype=np.float32).reshape(cpp_shape)
    py = np.fromfile(py_path, dtype=np.float32).reshape(py_shape)
    if cpp.shape != py.shape:
        print(f"[compare] shape mismatch cpp={cpp.shape} python={py.shape}")
        return
    diff = np.abs(cpp - py)
    print(f"[compare] shape={cpp.shape}")
    print(f"[compare] cpp_checksum={cpp.sum():.9g} py_checksum={py.sum():.9g}")
    print(f"[compare] max_abs_diff={diff.max():.9g} mean_abs_diff={diff.mean():.9g}")


def compare_modular_outputs(args: argparse.Namespace) -> None:
    cpp_dir = args.out_dir / "cpp"
    py_dir = args.out_dir / "python_modular"
    rows = []
    pairs = [
        ("text", "lingbot_predict_text_emb_raw", "python_text_emb_raw"),
        ("vae", "lingbot_predict_vae_latent_raw", "python_vae_latent_raw"),
    ]
    for sample_dir in sorted(cpp_dir.glob("sample_*")):
        py_sample_dir = py_dir / sample_dir.name
        for label, cpp_name, py_name in pairs:
            cpp_path = sample_dir / f"{cpp_name}.f32"
            py_path = py_sample_dir / f"{py_name}.f32"
            if not cpp_path.exists() or not py_path.exists():
                rows.append(f"{sample_dir.name} {label}: missing cpp={cpp_path.exists()} python={py_path.exists()}")
                continue
            cpp_shape = read_shape(sample_dir / f"{cpp_name}.shape.txt")
            py_shape = read_shape(py_sample_dir / f"{py_name}.shape.txt")
            cpp = np.fromfile(cpp_path, dtype=np.float32).reshape(cpp_shape)
            py = np.fromfile(py_path, dtype=np.float32).reshape(py_shape)
            if cpp.shape != py.shape:
                rows.append(f"{sample_dir.name} {label}: shape mismatch cpp={cpp.shape} python={py.shape}")
                continue
            diff = np.abs(cpp - py)
            rows.append(
                f"{sample_dir.name} {label}: max_abs_diff={diff.max():.9g} "
                f"mean_abs_diff={diff.mean():.9g} cpp_checksum={cpp.sum():.9g} py_checksum={py.sum():.9g}"
            )
    text = "\n".join(rows)
    print(text)
    (args.out_dir / "compare_modular_summary.txt").write_text(text + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[
        "fixtures", "cpp", "python", "python-modular", "both", "compare", "compare-modular"
    ], default="cpp")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/lingbot_e2e_parity"))
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--lingbot-repo", type=Path, default=DEFAULT_LINGBOT_REPO)
    parser.add_argument("--transformer-gguf", type=Path, default=DEFAULT_TRANSFORMER_GGUF)
    parser.add_argument("--text-gguf", type=Path, default=DEFAULT_TEXT_GGUF)
    parser.add_argument("--vae-gguf", type=Path, default=DEFAULT_VAE_GGUF)
    parser.add_argument("--addr", default="tcp://127.0.0.1:6055")
    parser.add_argument("--recv-timeout-ms", type=int, default=900_000)
    parser.add_argument("--session-id", type=int, default=101)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--num-fixtures", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--text-blocks", type=int, default=1)
    parser.add_argument("--video-steps", type=int, default=1)
    parser.add_argument("--action-steps", type=int, default=1)
    parser.add_argument("--resident-dtype", default="q8_0")
    parser.add_argument("--resident-blocks", type=int, default=1)
    parser.add_argument("--cuda-self-attn", action="store_true")
    parser.add_argument("--python-dtype", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--python-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--python-offload", action="store_true")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--action-guidance-scale", type=float, default=1.0)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    fixtures = make_fixtures(args.prompt, args.num_fixtures)
    for fixture in fixtures:
        save_fixture(fixture, args.out_dir / "fixtures")

    if args.mode == "fixtures":
        print(f"wrote {len(fixtures)} fixtures to {args.out_dir / 'fixtures'}")
        return 0
    if args.mode in ("cpp", "both"):
        run_cpp(args, fixtures)
    if args.mode in ("python", "both"):
        if len(fixtures) != 1:
            raise ValueError("python reference mode currently expects --num-fixtures 1")
        run_python_reference(args, fixtures[0])
    if args.mode == "python-modular":
        run_python_modular_reference(args, fixtures)
    if args.mode in ("compare", "both"):
        compare_outputs(args)
    if args.mode == "compare-modular":
        compare_modular_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
