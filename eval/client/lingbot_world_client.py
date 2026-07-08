# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import zmq
from PIL import Image
from transformers import AutoTokenizer


def _load_lingbot_pb():
    if "lingbot_pb2" in sys.modules:
        return sys.modules["lingbot_pb2"]
    proto_file = Path(os.environ.get(
        "LINGBOT_CPP_PROTO",
        Path(__file__).resolve().parents[2] / "serving" / "lingbot.proto",
    ))
    if not proto_file.exists():
        raise FileNotFoundError(
            f"lingbot.proto not found at {proto_file}. "
            f"Set LINGBOT_CPP_PROTO to override the expected "
            f"<vla.cpp>/serving/lingbot.proto location.")
    tmpdir = Path(tempfile.mkdtemp(prefix="lingbot-pb-"))
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
    import lingbot_pb2
    return lingbot_pb2


def _resize_hwc_u8(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return np.ascontiguousarray(img, dtype=np.uint8)
    pil = Image.fromarray(np.asarray(img, dtype=np.uint8), mode="RGB")
    pil = pil.resize((size, size), resample=Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))


class LingBotWorldClient:
    DEFAULT_RECV_TIMEOUT_MS = 900_000

    def __init__(
        self,
        vla_addr: str = "tcp://localhost:5557",
        *,
        tokenizer_name: str | None = None,
        image_size: int = 128,
        image_keys: Sequence[str] = ("image", "image2"),
        max_length: int = 512,
        recv_timeout_ms: int = DEFAULT_RECV_TIMEOUT_MS,
        n_action_steps: int = 1,
        session_id: int = 1,
        max_cache_frames: int = 4,
    ):
        if tokenizer_name is None:
            default_tok = Path("/home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long/tokenizer")
            tokenizer_name = str(default_tok) if default_tok.exists() else None
        if tokenizer_name is None:
            raise ValueError("LingBotWorldClient requires --tokenizer or the local LingBot tokenizer directory")
        if n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")

        self.pb = _load_lingbot_pb()
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self.sock.connect(vla_addr)
        print(f"lingbot-world-client: connected to {vla_addr}", flush=True)
        print(f"lingbot-world-client: loading tokenizer {tokenizer_name}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

        self.image_size = image_size
        self.image_keys = list(image_keys)
        self.max_length = max_length
        self.n_action_steps = n_action_steps
        self.session_id = int(session_id)
        self.max_cache_frames = int(max_cache_frames)
        self._request_id = 0
        self._action_queue: deque[np.ndarray] = deque(maxlen=n_action_steps)
        self._last_action_chunk: np.ndarray | None = None
        self._last_response = None

    def reset(self) -> None:
        self._action_queue.clear()
        self._last_action_chunk = None
        req = self.pb.LingBotRequest()
        req.request_id = self._request_id
        self._request_id += 1
        req.reset.session_id = self.session_id
        req.reset.clear_cache = True
        self.sock.send(req.SerializeToString())
        resp = self.pb.LingBotResponse()
        resp.ParseFromString(self.sock.recv())
        if resp.error:
            raise RuntimeError(f"wam-lingbot-server reset error: {resp.error}")

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        if not self._action_queue:
            chunk = self.predict_chunk(obs)
            self._last_action_chunk = np.ascontiguousarray(chunk[:, :7], dtype=np.float32)
            for row in chunk[: self.n_action_steps]:
                self._action_queue.append(np.ascontiguousarray(row[:7], dtype=np.float32))
        return self._action_queue.popleft()

    def predict_chunk(self, obs: dict[str, Any], action_noise: np.ndarray | None = None) -> np.ndarray:
        chunk = self._predict_chunk(obs, action_noise=action_noise)
        self._last_action_chunk = np.ascontiguousarray(chunk[:, :7], dtype=np.float32)
        return chunk

    def update_cache(
        self,
        obs_or_sequence: dict[str, Any] | Sequence[dict[str, Any]],
        action_chunk: np.ndarray | None = None,
        *,
        imagine: bool = False,
    ) -> str:
        req = self.pb.LingBotRequest()
        req.request_id = self._request_id
        self._request_id += 1
        req.step.session_id = self.session_id
        req.step.compute_kv_cache = True
        req.step.imagine = bool(imagine)
        obs_seq = self._as_obs_sequence(obs_or_sequence)
        if self.max_cache_frames > 0 and len(obs_seq) > self.max_cache_frames:
            obs_seq = obs_seq[-self.max_cache_frames :]
        self._add_images(req.step, obs_seq)
        if action_chunk is None:
            action_chunk = self._last_action_chunk
        if action_chunk is not None:
            self._add_action_condition(req.step, action_chunk)
        self._add_language(req.step, obs_seq[0])

        self.sock.send(req.SerializeToString())
        resp = self.pb.LingBotResponse()
        resp.ParseFromString(self.sock.recv())
        if resp.error:
            raise RuntimeError(f"wam-lingbot-server cache update error: {resp.error}")
        self._last_response = resp
        return resp.status

    def _extract_state(self, obs: dict[str, Any]) -> np.ndarray:
        robot_state = obs.get("robot_state", {})
        eef = robot_state.get("eef", {})
        gripper = robot_state.get("gripper", {})
        pos = np.asarray(eef.get("pos", np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
        quat = np.asarray(eef.get("quat", np.array([0, 0, 0, 1])), dtype=np.float64).reshape(-1)[:4]
        rpy = self.quat2axisangle(quat).astype(np.float32)
        qpos = np.asarray(gripper.get("qpos", np.zeros(1)), dtype=np.float32).reshape(-1)
        grip = qpos[:1] if qpos.size else np.zeros(1, dtype=np.float32)
        return np.concatenate([pos, rpy, grip]).astype(np.float32)

    def _predict_chunk(self, obs: dict[str, Any], action_noise: np.ndarray | None = None) -> np.ndarray:
        req = self.pb.LingBotRequest()
        req.request_id = self._request_id
        self._request_id += 1
        req.step.session_id = self.session_id

        self._add_images(req.step, obs)
        if action_noise is not None:
            self._add_action_noise(req.step, action_noise)
        state = self._extract_state(obs)
        req.step.state.extend(state.tolist())
        if self._last_action_chunk is not None and os.environ.get("VLA_LINGBOT_SEND_LAST_ACTION_COND"):
            self._add_action_condition(req.step, self._last_action_chunk)
        self._add_language(req.step, obs)

        self.sock.send(req.SerializeToString())
        resp = self.pb.LingBotResponse()
        resp.ParseFromString(self.sock.recv())
        if resp.error:
            raise RuntimeError(f"wam-lingbot-server error: {resp.error}")
        self._last_response = resp
        if resp.chunk_size <= 0 or resp.action_dim <= 0:
            raise RuntimeError(f"invalid LingBot response shape: chunk={resp.chunk_size} action_dim={resp.action_dim}")
        return np.asarray(resp.action_chunk, dtype=np.float32).reshape(resp.chunk_size, resp.action_dim)

    def _add_images(self, step, obs_or_sequence: dict[str, Any] | Sequence[dict[str, Any]]) -> None:
        obs_seq = self._as_obs_sequence(obs_or_sequence)
        first_pixels = obs_seq[0].get("pixels", {})
        for key in self.image_keys:
            if key not in first_pixels:
                raise KeyError(f"LingBot image key '{key}' missing; got pixels keys {list(first_pixels.keys())}")
            frames = []
            for obs_i in obs_seq:
                pixels = obs_i.get("pixels", {})
                if key not in pixels:
                    raise KeyError(f"LingBot image key '{key}' missing in sequence item; got pixels keys {list(pixels.keys())}")
                img = np.asarray(pixels[key])
                if img.ndim != 3 or img.shape[-1] != 3:
                    raise ValueError(f"pixels[{key!r}]: expected HWC RGB image, got {img.shape}")
                img = np.ascontiguousarray(img[::-1, ::-1], dtype=np.uint8)
                frames.append(_resize_hwc_u8(img, self.image_size))
            video = np.ascontiguousarray(np.stack(frames, axis=0), dtype=np.uint8)
            t = step.input_images.add()
            t.name = key
            t.dtype = self.pb.U8
            t.shape.extend([video.shape[0], video.shape[1], video.shape[2], 3])
            t.data = video.tobytes()

    def _add_action_condition(self, step, action_chunk: np.ndarray) -> None:
        action_cond = self._chunk_to_action_condition(action_chunk)
        t = step.action_condition
        t.name = "previous_action_condition"
        t.dtype = self.pb.F32
        t.shape.extend(action_cond.shape)
        t.data = action_cond.tobytes()

    def _add_action_noise(self, step, action_noise: np.ndarray) -> None:
        noise = np.ascontiguousarray(action_noise, dtype=np.float32)
        if noise.ndim != 2 or noise.shape[1] != 30:
            raise ValueError(f"LingBot action_noise must be [T,30], got {noise.shape}")
        t = step.action_noise
        t.name = "action_noise"
        t.dtype = self.pb.F32
        t.shape.extend(noise.shape)
        t.data = noise.tobytes()

    def _add_language(self, step, obs: dict[str, Any]) -> None:
        task = obs.get("task_description", "")
        if isinstance(task, bytes):
            task = task.decode()
        toks = self.tok(
            task,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="np",
        )
        lang = toks["input_ids"][0].astype(np.int32)
        step.lang_tokens.extend(lang.tolist())

    @staticmethod
    def _as_obs_sequence(obs_or_sequence: dict[str, Any] | Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(obs_or_sequence, dict):
            return [obs_or_sequence]
        seq = list(obs_or_sequence)
        if not seq:
            raise ValueError("observation sequence must not be empty")
        return seq

    @staticmethod
    def _first_obs(obs_or_sequence: dict[str, Any] | Sequence[dict[str, Any]]) -> dict[str, Any]:
        if isinstance(obs_or_sequence, dict):
            return obs_or_sequence
        seq = list(obs_or_sequence)
        if not seq:
            raise ValueError("observation sequence must not be empty")
        return seq[0]

    @staticmethod
    def _chunk_to_action_condition(chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] < 7:
            raise ValueError(f"expected action chunk [T,>=7], got {chunk.shape}")
        action_per_frame = 4
        frames = int(np.ceil(chunk.shape[0] / action_per_frame))
        out = np.zeros((7, frames, action_per_frame), dtype=np.float32)
        for t in range(chunk.shape[0]):
            f = t // action_per_frame
            h = t % action_per_frame
            out[:, f, h] = chunk[t, :7]
        return np.ascontiguousarray(out)

    @staticmethod
    def quat2axisangle(quat: np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=np.float64).copy()
        if q.size != 4:
            return np.zeros(3, dtype=np.float32)
        q[3] = np.clip(q[3], -1.0, 1.0)
        den = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
        if den < 1e-8:
            return np.zeros(3, dtype=np.float32)
        return (q[:3] * 2.0 * np.arccos(q[3]) / den).astype(np.float32)
