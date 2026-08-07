# Copyright 2026 VinRobotics
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import builtins
from collections import deque
import locale
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import sentencepiece
from transformers import AutoTokenizer
import zmq

from client.reproducibility import generate_action_noise

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("LANG", "C.UTF-8")
os.environ.setdefault("LC_ALL", "C.UTF-8")

try:
    locale.setlocale(locale.LC_ALL, "C.UTF-8")
except locale.Error:
    pass


ARCH_PRESETS = {
    "pi05": {
        "image_size": 224,
        "tokenizer": None,
        "max_state_dim": 32,
        "max_length": 200,
        "n_action_steps": 10,
        "use_fast_tokenizer": True,
    },
    "smolvla": {
        "image_size": 512,
        "tokenizer": None,
        "max_state_dim": 32,
        "max_length": 48,
        "n_action_steps": 1,
        "use_fast_tokenizer": True,
        "noise_chunk_size": 50,
        "noise_action_dim": 32,
    },
    "hy_vla": {
        "image_size": 224,
        "tokenizer": None,
        "max_state_dim": 20,
        "max_length": 48,
        "n_action_steps": 1,
        "use_fast_tokenizer": True,
    },
    "groot_n1": {
        "image_size": 256,
        "image_crop_size": 230,
        "tokenizer": None,
        "max_state_dim": 132,
        "max_length": 1024,
        "n_action_steps": 8,
        "use_server_tokenizer": True,
    },
}


def _load_pb():
    if "vla_pb2" in sys.modules:
        return sys.modules["vla_pb2"]
    proto_file = Path(os.environ.get(
        "VLA_CPP_PROTO",
        Path(__file__).resolve().parents[2] / "serving" / "vla.proto",
    ))
    if not proto_file.exists():
        raise FileNotFoundError(
            f"vla.proto not found at {proto_file}. "
            "Set VLA_CPP_PROTO to override the expected serving/vla.proto path."
        )
    tmpdir = Path(tempfile.mkdtemp(prefix="vla-pb-"))
    protoc = os.environ.get("VLA_CPP_PROTOC")
    if protoc is None:
        protoc = "/usr/bin/protoc" if Path("/usr/bin/protoc").exists() else "protoc"
    subprocess.check_call([
        protoc,
        f"--proto_path={proto_file.parent}",
        f"--python_out={tmpdir}",
        str(proto_file),
    ])
    sys.path.insert(0, str(tmpdir))
    import vla_pb2
    return vla_pb2


def _load_paligemma_sentencepiece(tokenizer_name: str) -> sentencepiece.SentencePieceProcessor:
    candidates = []
    tok_path = Path(tokenizer_name)
    if tok_path.is_file():
        candidates.append(tok_path)
    else:
        candidates.extend([
            tok_path / "tokenizer.model",
            tok_path / "paligemma_tokenizer.model",
        ])
    candidates.extend([
        Path("/home/xuling/robotic_code/embodied.cpp/pi0.5实现/models/paligemma-3b-pt-224-tokenizer/tokenizer.model"),
        Path.home() / ".cache" / "openpi" / "paligemma_tokenizer.model",
    ])
    for path in candidates:
        if path.exists():
            print(f"vla-cpp-direct: using PaliGemma SentencePiece {path}", flush=True)
            with path.open("rb") as f:
                return sentencepiece.SentencePieceProcessor(model_proto=f.read())
    raise FileNotFoundError(
        "PaliGemma tokenizer.model not found. Pass --tokenizer as a tokenizer.model "
        "file or a directory containing it."
    )


def _tokenize_paligemma_prompt(
    tok: sentencepiece.SentencePieceProcessor,
    prompt: str,
    max_len: int,
) -> np.ndarray:
    cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
    tokens = tok.encode(cleaned_text, add_bos=True) + tok.encode("\n")
    if len(tokens) < max_len:
        tokens = tokens + [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    return np.asarray(tokens, dtype=np.int32)


def _resize_with_pad(
    img_chw: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    t = torch.from_numpy(img_chw).unsqueeze(0)
    cur_h, cur_w = t.shape[2:]
    ratio = max(cur_w / target_w, cur_h / target_h)
    rh = int(cur_h / ratio)
    rw = int(cur_w / ratio)
    t = F.interpolate(t, size=(rh, rw), mode="bilinear", align_corners=False)
    pad_h = max(0, target_h - rh)
    pad_w = max(0, target_w - rw)
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    t = F.pad(
        t,
        (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top),
        value=pad_value,
    )
    return t.squeeze(0).numpy()


def _resize_with_pad_f32_hwc(
    img_chw: np.ndarray,
    target_h: int,
    target_w: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    tensor = torch.from_numpy(img_chw).unsqueeze(0)
    current_h, current_w = tensor.shape[2:]
    ratio = max(current_w / target_w, current_h / target_h)
    resized_h = int(current_h / ratio)
    resized_w = int(current_w / ratio)
    tensor = F.interpolate(
        tensor,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )
    pad_h = max(0, target_h - resized_h)
    pad_w = max(0, target_w - resized_w)
    if pad_h or pad_w:
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        tensor = F.pad(
            tensor,
            (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top),
            value=pad_value,
        )
    return tensor.squeeze(0).permute(1, 2, 0).contiguous().numpy()


def _resize_center_crop(
    img_chw: np.ndarray,
    target_size: int,
    crop_size: int,
) -> np.ndarray:
    tensor = torch.from_numpy(img_chw).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    crop_top = (target_size - crop_size) // 2
    crop_left = (target_size - crop_size) // 2
    tensor = tensor[
        :,
        :,
        crop_top : crop_top + crop_size,
        crop_left : crop_left + crop_size,
    ]
    tensor = F.interpolate(
        tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return tensor.squeeze(0).numpy()


def _resize_center_crop_u8_hwc(
    img_chw: np.ndarray,
    target_size: int,
    crop_size: int,
) -> np.ndarray:
    tensor = torch.from_numpy(img_chw).unsqueeze(0)
    if tensor.shape[2:] != (target_size, target_size):
        tensor = F.interpolate(
            tensor,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    crop_top = (target_size - crop_size) // 2
    crop_left = (target_size - crop_size) // 2
    tensor = tensor[
        :,
        :,
        crop_top : crop_top + crop_size,
        crop_left : crop_left + crop_size,
    ]
    tensor = F.interpolate(
        tensor,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    tensor = tensor.clamp_(0.0, 1.0).mul_(255.0).add_(0.5).to(torch.uint8)
    return tensor.squeeze(0).permute(1, 2, 0).contiguous().numpy()


class VlaCppClient:
    DEFAULT_RECV_TIMEOUT_MS = 30_000

    def __init__(
        self,
        vla_addr: str = "tcp://localhost:5555",
        *,
        arch: str = "hy_vla",
        tokenizer_name: str | None = None,
        image_size: int | None = None,
        max_state_dim: int | None = None,
        real_action_dim: int = 7,
        image_keys: Sequence[str] = (
            "observation.images.image",
            "observation.images.image2",
        ),
        max_length: int | None = None,
        recv_timeout_ms: int = DEFAULT_RECV_TIMEOUT_MS,
        n_action_steps: int | None = None,
        noise_seed: int | None = None,
    ):
        if arch not in ARCH_PRESETS:
            raise ValueError(f"unknown arch {arch!r}; expected one of {sorted(ARCH_PRESETS)}")
        preset = ARCH_PRESETS[arch]
        tokenizer_name = tokenizer_name if tokenizer_name is not None else preset["tokenizer"]
        image_size = image_size if image_size is not None else preset["image_size"]
        max_state_dim = (
            max_state_dim if max_state_dim is not None else preset.get("max_state_dim", 32)
        )
        max_length = max_length if max_length is not None else preset.get("max_length", 48)
        n_action_steps = (
            n_action_steps if n_action_steps is not None else preset.get("n_action_steps", 1)
        )
        use_server_tokenizer = bool(preset.get("use_server_tokenizer", False))
        if tokenizer_name is None and not use_server_tokenizer:
            raise ValueError(f"arch={arch} has no default tokenizer; pass --tokenizer.")

        self.arch = arch
        self.pb = _load_pb()
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self.sock.connect(vla_addr)
        print(f"vla-cpp-direct[arch={arch}]: connected to {vla_addr}", flush=True)

        self.tok = None
        if not use_server_tokenizer:
            use_fast = bool(preset.get("use_fast_tokenizer", True))
            print(
                f"vla-cpp-direct: loading tokenizer {tokenizer_name}"
                f"{' (use_fast=False)' if not use_fast else ''}",
                flush=True,
            )
        if arch == "pi05":
            assert tokenizer_name is not None
            self.tok = _load_paligemma_sentencepiece(tokenizer_name)
        elif not use_server_tokenizer:
            assert tokenizer_name is not None
            tokenizer_kwargs = {}
            chat_template_path = Path(tokenizer_name) / "chat_template.jinja"
            if chat_template_path.exists():
                tokenizer_kwargs["chat_template"] = chat_template_path.read_text(encoding="utf-8")
            orig_open = builtins.open

            def open_utf8_chat_template(file, *args, **kwargs):
                if str(file).endswith("chat_template.jinja") and "encoding" not in kwargs:
                    kwargs["encoding"] = "utf-8"
                return orig_open(file, *args, **kwargs)

            try:
                builtins.open = open_utf8_chat_template
                self.tok = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    use_fast=use_fast,
                    **tokenizer_kwargs,
                )
            finally:
                builtins.open = orig_open

        self.image_size = image_size
        self.image_crop_size = int(preset.get("image_crop_size", image_size))
        self.max_state_dim = max_state_dim
        self.real_action_dim = real_action_dim
        self.image_keys = list(image_keys)
        self.max_length = max_length
        self.noise_chunk_size = int(preset.get("noise_chunk_size", 0))
        self.noise_action_dim = int(preset.get("noise_action_dim", 0))
        self._initial_noise_seed = noise_seed
        self._noise_rng: np.random.Generator | None = None
        self._step = 0
        self._last_response = None
        self._inference_sequence = 0
        self._last_inference_profile: dict[str, float | int] | None = None

        if n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")
        self.n_action_steps = n_action_steps
        self._action_queue: deque[np.ndarray] = deque(maxlen=n_action_steps)

    def get_arch(self) -> str:
        return self.arch

    def ping(self) -> bool:
        return True

    def reset(self, noise_seed: int | None = None) -> None:
        self._action_queue.clear()
        self._step = 0
        self._last_inference_profile = None
        resolved_seed = self._initial_noise_seed if noise_seed is None else noise_seed
        self._noise_rng = (
            np.random.default_rng(resolved_seed) if resolved_seed is not None else None
        )

    def get_last_inference_profile(self) -> dict[str, float | int] | None:
        if self._last_inference_profile is None:
            return None
        return dict(self._last_inference_profile)

    def get_action(self, observations: dict[str, Any]) -> np.ndarray:
        if not self._action_queue:
            chunk = self._predict_chunk(observations)
            for row in chunk[: self.n_action_steps, : self.real_action_dim]:
                self._action_queue.append(np.ascontiguousarray(row, dtype=np.float32))
        return self._action_queue.popleft()

    def get_action_from_queue(self) -> np.ndarray:
        if not self._action_queue:
            raise RuntimeError("action queue is empty; call get_action(obs) first")
        return self._action_queue.popleft()

    def has_queued_action(self) -> bool:
        return bool(self._action_queue)

    def _predict_chunk(self, observations: dict[str, Any]) -> np.ndarray:
        images: list[tuple[np.ndarray, int]] = []
        for key in self.image_keys:
            if key not in observations:
                raise KeyError(f"image key '{key}' missing; got {list(observations.keys())}")
            img = observations[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            img = np.asarray(img, dtype=np.float32)
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"{key}: expected CHW float32 [3, H, W], got {img.shape}")
            if self.arch == "groot_n1":
                img_u8 = _resize_center_crop_u8_hwc(
                    img,
                    self.image_size,
                    self.image_crop_size,
                )
                images.append((img_u8, self.pb.Image.RGB_U8))
            elif self.arch == "pi05":
                img_hwc = _resize_with_pad_f32_hwc(
                    img,
                    self.image_size,
                    self.image_size,
                    pad_value=0.0,
                )
                images.append((img_hwc, self.pb.Image.F32_RGB_01))
            else:
                img = _resize_with_pad(img, self.image_size, self.image_size, pad_value=0.0)
                img_hwc = np.transpose(img, (1, 2, 0))
                images.append((
                    np.ascontiguousarray(img_hwc, dtype=np.float32),
                    self.pb.Image.F32_RGB_01,
                ))

        state = observations["observation.state"]
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size > self.max_state_dim:
            raise ValueError(
                f"state has {state.size} dims, exceeds max_state_dim={self.max_state_dim}"
            )
        state_padded = np.zeros(self.max_state_dim, dtype=np.float32)
        state_padded[:state.shape[0]] = state

        task = observations.get("task", "")
        if isinstance(task, bytes):
            task = task.decode()
        task = str(task)
        if self.arch == "groot_n1":
            task = re.sub(r"[^\w\s]", "", task.lower()).strip()
            lang = None
            toks = None
        elif self.arch == "hy_vla":
            suffix = "<｜hy_Assistant｜>"
            task = task.strip().replace("_", " ").replace("\n", " ")
            task = task if task.endswith(suffix) else f"{task}{suffix}"
            toks = self.tok(
                task,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
                add_special_tokens=False,
            )
        elif self.arch == "smolvla":
            # SmolVLA uses the SmolVLM2 HuggingFace tokenizer (auto). LeRobot's
            # serialized pipeline first applies smolvla_new_line_processor.
            # Keep the fixed 48-token transport shape and carry its real
            # padding mask separately for the runtime attention mask.
            task = task.strip().replace("\n", " ") + "\n"
            toks = self.tok(
                task,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
                add_special_tokens=False,
            )
        else:
            assert self.tok is not None
            lang = _tokenize_paligemma_prompt(self.tok, task, self.max_length)
            toks = None
        if toks is not None:
            lang = toks["input_ids"][0].astype(np.int32)
            lang_attention_mask = toks.get("attention_mask")
            if lang_attention_mask is not None:
                lang_attention_mask = lang_attention_mask[0].astype(np.uint32)
        else:
            lang_attention_mask = None

        req = self.pb.PredictRequest()
        req.request_id = self._step
        self._step += 1
        for img, encoding in images:
            ip = req.images.add()
            ip.encoding = encoding
            ip.height = img.shape[0]
            ip.width = img.shape[1]
            ip.data = img.tobytes()
        if self.arch == "groot_n1":
            req.language_text = task
        else:
            assert lang is not None
            req.lang_tokens.extend(lang.tolist())
            if lang_attention_mask is not None:
                req.attention_mask.extend(lang_attention_mask.tolist())
        req.state.extend(state_padded.tolist())

        action_noise = observations.get("action_noise")
        if (
            action_noise is None
            and self._noise_rng is not None
            and self.noise_chunk_size
            and self.noise_action_dim
        ):
            action_noise = generate_action_noise(
                self._noise_rng, self.noise_chunk_size, self.noise_action_dim
            )
        if action_noise is not None:
            noise = np.ascontiguousarray(action_noise, dtype=np.float32).reshape(-1)
            req.noise.extend(noise.tolist())

        self.sock.send(req.SerializeToString())
        body = self.sock.recv()
        resp = self.pb.PredictResponse()
        resp.ParseFromString(body)
        if resp.error:
            raise RuntimeError(f"VLA server error: {resp.error}")
        self._last_response = resp
        self._inference_sequence += 1
        self._last_inference_profile = {
            "sequence": self._inference_sequence,
            "request_id": int(resp.request_id),
            "server_total_ms": float(resp.latency_ms_total),
            "server_vision_ms": float(resp.latency_ms_vision),
            "server_inference_ms": float(resp.latency_ms_inference),
            "server_prefill_ms": float(resp.latency_ms_prefill),
            "server_denoise_ms": float(resp.latency_ms_denoise),
            "model_chunk_size": int(resp.chunk_size),
            "model_action_dim": int(resp.action_dim),
            "replay_chunk_size": int(self.n_action_steps),
        }
        return np.array(resp.action_chunk, dtype=np.float32).reshape(
            resp.chunk_size,
            resp.action_dim,
        )

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
