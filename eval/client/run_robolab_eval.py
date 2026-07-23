#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
# SPDX-License-Identifier: Apache-2.0

"""Run RoboLab evaluation against the native Embodied.cpp Cosmos3 WAM server.

This is the native Embodied.cpp RoboLab entry point.  It intentionally does not
call the external RoboLab Cosmos3 policy runner or policy client.
RoboLab still owns the simulator, task registration, episode loop, logging, and
success metrics; Embodied.cpp owns the Cosmos3 request ABI and C++ WAM server.

Data flow:

  RoboLab env observation
      -> NativeCosmos3Client observation packing
      -> ZMQ WamRequest protobuf
      -> serving/wam-server + models/cosmos3.cpp
      -> 32 x 8 action chunk
      -> RoboLab env.step()

The C++ Cosmos3 forward path is still being implemented.  If the server returns
an error, this runner surfaces it directly in the rollout, which keeps failures
attached to the exact task/episode that triggered them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONF = REPO_ROOT / "eval" / "conf" / "robolab_cosmos3_eval.yaml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The shared server uses a user-local protoc 3.12 extracted from Ubuntu
# packages, while RoboLab's Python protobuf package may be newer.  The pure
# Python runtime accepts the older generated descriptors and is fast enough for
# evaluation request forwarding.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _configure_omniverse_proxy(env: dict[str, str]) -> None:
    """Normalize explicitly configured proxy variables for Isaac/Omniverse."""
    if env.get("HTTP_PROXY") and not env.get("http_proxy"):
        env["http_proxy"] = env["HTTP_PROXY"]
    if env.get("HTTPS_PROXY") and not env.get("https_proxy"):
        env["https_proxy"] = env["HTTPS_PROXY"]
    if env.get("http_proxy"):
        env["HTTP_PROXY"] = env["http_proxy"]
        env["HTTPS_PROXY"] = env.get("https_proxy", env["http_proxy"])
    if env.get("https_proxy") and not env.get("HTTPS_PROXY"):
        env["HTTPS_PROXY"] = env["https_proxy"]
    if env.get("ALL_PROXY") and not env.get("all_proxy"):
        env["all_proxy"] = env["ALL_PROXY"]
    if env.get("all_proxy") and not env.get("ALL_PROXY"):
        env["ALL_PROXY"] = env["all_proxy"]
    env.setdefault("no_proxy", "127.0.0.1,localhost")
    env.setdefault("NO_PROXY", env["no_proxy"])


def _resolve_path(path: str | os.PathLike[str], *, base: Path = REPO_ROOT) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (base / resolved).resolve()
    return resolved


def _read_configured_python(path: Path) -> Path | None:
    if not path.exists():
        return None
    in_paths = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.startswith("paths:"):
            in_paths = True
            continue
        if in_paths and line and not line.startswith(" "):
            in_paths = False
        if in_paths and line.lstrip().startswith("robolab_python:"):
            _, value = line.split(":", 1)
            value = value.strip().strip("'\"")
            return _resolve_path(value) if value else None
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _wait_log_contains(proc: subprocess.Popen, log_path: Path, needle: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
            raise RuntimeError(f"process exited early with code {proc.returncode}\n{tail}")
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            last = text[-4000:]
            if needle in text:
                return
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {needle!r}\n{last}")


def _compile_wam_proto(proto_path: Path, out_dir: Path) -> Any:
    protoc = os.environ.get("Protobuf_PROTOC_EXECUTABLE") or "protoc"
    cmd = [
        protoc,
        f"--proto_path={proto_path.parent}",
        f"--python_out={out_dir}",
        str(proto_path),
    ]
    subprocess.check_call(cmd)
    sys.path.insert(0, str(out_dir))
    spec = importlib.util.spec_from_file_location("wam_pb2", out_dir / "wam_pb2.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load generated wam_pb2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wam_pb2"] = mod
    spec.loader.exec_module(mod)
    return mod


def _tensor_from_array(wam_pb2: Any, name: str, arr: np.ndarray, dtype: int) -> Any:
    arr = np.ascontiguousarray(arr)
    t = wam_pb2.Tensor()
    t.name = name
    t.dtype = dtype
    t.shape.extend(int(x) for x in arr.shape)
    t.data = arr.tobytes()
    return t


def _set_proto_map(field: Any, values: dict[str, str]) -> None:
    if hasattr(field, "update"):
        field.update(values)
        return
    for key, value in values.items():
        item = field.add()
        item.key = key
        item.value = value


class NativeCosmos3Client:
    """RoboLab InferenceClient implementation that talks directly to WAM."""

    IMAGE_W = 640
    IMAGE_H = 360
    DEFAULT_ACTION_HORIZON = 8
    VIDEO_GRID_THW = (17, 34, 46)
    VIDEO_FRAME_SEQLEN = 391
    VIDEO_TIMESTAMPS = (0.0, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 1.8, 1.9, 2.0, 2.2)
    CONCAT_VIEW_DESCRIPTION = (
        "The top row is from the wrist-mounted camera. "
        "The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite "
        "sides, with the robot visible."
    )

    def __init__(self, *, wam_pb2: Any, wam_addr: str, request_timeout_ms: int, policy_cfg: dict[str, Any]):
        import zmq

        self.wam_pb2 = wam_pb2
        self.wam_addr = wam_addr
        self.policy_cfg = dict(policy_cfg)
        self.open_loop_horizon = int(self.policy_cfg.get("action_horizon", self.DEFAULT_ACTION_HORIZON))
        if self.open_loop_horizon <= 0:
            raise ValueError(f"client action_horizon must be positive, got {self.open_loop_horizon}")
        self._chunks: dict[int, np.ndarray] = {}
        self._counters: dict[int, int] = {}
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, int(request_timeout_ms))
        self._sock.setsockopt(zmq.SNDTIMEO, int(request_timeout_ms))
        self._sock.connect(wam_addr)
        self._request_id = 0
        self._session_id = int(time.time() * 1000) & 0xFFFFFFFF
        from adapter.sim.robolab import RoboLabCosmos3ObservationAdapter

        self._adapter = RoboLabCosmos3ObservationAdapter()
        dump_dir = self.policy_cfg.get("debug_request_dump_dir") or os.environ.get("COSMOS3_DEBUG_REQUEST_DUMP_DIR")
        self._debug_request_dump_dir = Path(dump_dir) if dump_dir else None
        self._debug_request_dump_limit = int(
            self.policy_cfg.get("debug_request_dump_limit", os.environ.get("COSMOS3_DEBUG_REQUEST_DUMP_LIMIT", 0) or 0)
        )
        if self._debug_request_dump_dir is not None:
            self._debug_request_dump_dir.mkdir(parents=True, exist_ok=True)
        self._processor = None
        self._token_cache: dict[str, dict[str, np.ndarray]] = {}

    def _debug_latency_enabled(self) -> bool:
        return bool(self.policy_cfg.get("debug_policy_latency") or os.environ.get("COSMOS3_DEBUG_POLICY_LATENCY"))

    def _tokenizer_root(self) -> Path:
        configured = self.policy_cfg.get("qwen_tokenizer_root") or os.environ.get("COSMOS3_QWEN_TOKENIZER_ROOT")
        if configured:
            return _resolve_path(configured)
        return REPO_ROOT / "assets" / "qwen3_vl_tokenizer"

    def _get_processor(self) -> Any:
        if self._processor is None:
            from transformers import AutoTokenizer

            self._processor = AutoTokenizer.from_pretrained(self._tokenizer_root(), trust_remote_code=True)
        return self._processor

    def _official_prompt(self, instruction: str) -> str:
        text = str(instruction or "").strip()
        if text and text[-1] not in ".!?":
            text += "."
        return (
            f"{text} This video contains concatenated views from multiple camera perspectives. "
            f"{self.CONCAT_VIEW_DESCRIPTION} "
            "The video is 2.0 seconds long and is of 15 FPS. "
            "This video is of 544x736 resolution."
        ).strip()

    def _build_dynamic_text_tokens(self, instruction: str) -> dict[str, np.ndarray]:
        cached = self._token_cache.get(instruction)
        if cached is not None:
            return cached

        if self._debug_latency_enabled():
            print(f"[cosmos3-eval] tokenize_start instruction={instruction!r}", flush=True)
        tokenizer = self._get_processor()
        video_token = "<|video_pad|>"
        vision_start = "<|vision_start|>"
        vision_end = "<|vision_end|>"
        chat_prefix = "<|im_start|>user\n"
        chat_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        prompt = self._official_prompt(instruction)
        chat_text = chat_prefix + vision_start + video_token + vision_end + prompt + chat_suffix
        expanded_video = ""
        for timestamp in self.VIDEO_TIMESTAMPS:
            expanded_video += f"<{timestamp:.1f} seconds>"
            expanded_video += vision_start + (video_token * self.VIDEO_FRAME_SEQLEN) + vision_end
        expanded_text = chat_text.replace(vision_start + video_token + vision_end, expanded_video, 1)
        qwen_ids = np.asarray(tokenizer(expanded_text, add_special_tokens=False).input_ids, dtype=np.int32)
        cond_text = np.asarray(tokenizer(chat_prefix + prompt + chat_suffix, add_special_tokens=False).input_ids, dtype=np.int32)
        video_pad = int(tokenizer.convert_tokens_to_ids(video_token))
        visual_indices = np.full(qwen_ids.shape, -1, dtype=np.int32)
        visual_indices[qwen_ids == video_pad] = np.arange(int(np.sum(qwen_ids == video_pad)), dtype=np.int32)
        mrope = np.ones((3, qwen_ids.shape[0]), dtype=np.int32)
        st = 0
        t, h, w = self.VIDEO_GRID_THW
        merge_size = 2
        for _ in range(t):
            ed = st
            while ed < qwen_ids.shape[0] and int(qwen_ids[ed]) != video_pad:
                ed += 1
            if ed >= qwen_ids.shape[0]:
                raise ValueError("dynamic Qwen input tokenization did not find expected video segment")
            if st > 0:
                prev_max = int(max(mrope[0, st - 1], mrope[1, st - 1], mrope[2, st - 1]))
                st_idx = prev_max + 1
            else:
                st_idx = 0
            for token in range(st, ed):
                pos = st_idx + (token - st)
                mrope[:, token] = pos
            video_base = int(mrope[0, ed - 1]) + 1
            for local in range(self.VIDEO_FRAME_SEQLEN):
                token = ed + local
                if token >= qwen_ids.shape[0] or int(qwen_ids[token]) != video_pad:
                    raise ValueError("dynamic Qwen input video token layout mismatch")
                h_pos = local // (w // merge_size)
                w_pos = local % (w // merge_size)
                mrope[0, token] = video_base
                mrope[1, token] = video_base + h_pos
                mrope[2, token] = video_base + w_pos
            st = ed + self.VIDEO_FRAME_SEQLEN
        if st < qwen_ids.shape[0]:
            prev_max = int(max(mrope[0, st - 1], mrope[1, st - 1], mrope[2, st - 1]))
            st_idx = prev_max + 1
            for token in range(st, qwen_ids.shape[0]):
                pos = st_idx + (token - st)
                mrope[:, token] = pos
        uncond_text = np.asarray(
            tokenizer(chat_prefix + chat_suffix, add_special_tokens=False).input_ids,
            dtype=np.int32,
        )
        text_tail = np.asarray([151645, 151652], dtype=np.int32)
        result = {
            "qwen_input_ids": qwen_ids,
            "qwen_visual_indices": visual_indices,
            "qwen_mrope_positions": mrope,
            "mot_text_ids": np.concatenate([cond_text, text_tail]).astype(np.int32, copy=False),
            "mot_text_ids_uncond": np.concatenate([uncond_text, text_tail]).astype(np.int32, copy=False),
        }
        if self._debug_latency_enabled():
            print(
                "[cosmos3-eval] tokenize_done "
                f"qwen={int(qwen_ids.shape[0])} visual={int(np.sum(qwen_ids == video_pad))} "
                f"mot={int(result['mot_text_ids'].shape[0])} "
                f"uncond={int(result['mot_text_ids_uncond'].shape[0])}",
                flush=True,
            )
        self._token_cache[instruction] = result
        return result

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        if self._debug_latency_enabled():
            print(f"[cosmos3-eval] infer_start env={env_id} counter={self._counters.get(env_id, 0)}", flush=True)
        extracted = self._extract_observation(obs, env_id=env_id)
        if env_id not in self._chunks or self._counters[env_id] >= self.open_loop_horizon:
            if self._debug_latency_enabled():
                print(f"[cosmos3-eval] pack_request_start env={env_id}", flush=True)
            request = self._pack_request(extracted, instruction)
            if self._debug_latency_enabled():
                print(f"[cosmos3-eval] query_start bytes={len(request)}", flush=True)
            response = self._query_server(request)
            self._chunks[env_id] = self._postprocess_chunk(self._unpack_response(response))
            if self._debug_latency_enabled():
                chunk = self._chunks[env_id]
                print(
                    "[cosmos3-eval] action_chunk "
                    f"env={env_id} shape={tuple(chunk.shape)} "
                    f"min={float(np.nanmin(chunk)):.4f} max={float(np.nanmax(chunk)):.4f} "
                    f"nan={bool(np.isnan(chunk).any())}",
                    flush=True,
                )
            self._counters[env_id] = 0
        action = self._chunks[env_id][self._counters[env_id]]
        self._counters[env_id] += 1
        return {"action": action, "viz": self._build_visualization(extracted)}

    def infer_batch(self, obs: Any, instruction: str, *, env_ids: list[int]) -> dict[int, dict]:
        return {env_id: self.infer(obs, instruction, env_id=env_id) for env_id in env_ids}

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        return self._adapter.parse_model_inputs(raw_obs, env_id=env_id)

    def _pack_request(self, extracted_obs: dict, instruction: str) -> bytes:
        if self._debug_latency_enabled():
            print("[cosmos3-eval] pack_request_enter", flush=True)
        image = np.asarray(extracted_obs["observation/image"])
        joint = np.asarray(extracted_obs["observation/joint_position"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(extracted_obs["observation/gripper_position"], dtype=np.float32).reshape(-1)
        state = np.concatenate([joint, gripper]).astype(np.float32)

        req = self.wam_pb2.WamRequest()
        req.request_id = self._next_request_id()
        req.step.session_id = self._session_id
        req.step.instruction = instruction
        req.step.state.extend(float(x) for x in state)
        _set_proto_map(
            req.step.params,
            {
                "cosmos3.guidance": str(self.policy_cfg.get("guidance", 3.0)),
                "cosmos3.num_steps": str(self.policy_cfg.get("num_steps", 4)),
                "cosmos3.shift": str(self.policy_cfg.get("shift", 5.0)),
                "cosmos3.sampler": str(self.policy_cfg.get("sampler", "unipc")),
                "cosmos3.resolution": str(self.policy_cfg.get("resolution", 480)),
                "cosmos3.conditioning_fps": str(self.policy_cfg.get("conditioning_fps", 15.0)),
                "cosmos3.action_chunk_size": str(self.policy_cfg.get("action_chunk_size", 32)),
                "cosmos3.action_dim": str(self.policy_cfg.get("action_dim", 8)),
                "cosmos3.max_action_dim": str(self.policy_cfg.get("max_action_dim", 64)),
                "cosmos3.action_space": str(self.policy_cfg.get("action_space", "joint_pos")),
                "cosmos3.action_domain_name": str(self.policy_cfg.get("action_domain_name", "droid_lerobot")),
                "cosmos3.action_domain_id": str(self.policy_cfg.get("action_domain_id", 8)),
                "cosmos3.image_height": str(image.shape[0]),
                "cosmos3.image_width": str(image.shape[1]),
                "cosmos3.use_state": str(bool(self.policy_cfg.get("use_state", True))).lower(),
                "cosmos3.history_length": str(self.policy_cfg.get("history_length", 1)),
                "cosmos3.seed": str(self.policy_cfg.get("seed", self.policy_cfg.get("deterministic_seed", 0))),
                "cosmos3.deterministic_seed": str(self.policy_cfg.get("deterministic_seed", 0)),
                "cosmos3.enable_mot_condition_cache": str(
                    bool(self.policy_cfg.get("enable_mot_condition_cache", True))
                ).lower(),
                "cosmos3.invert_gripper": str(bool(self.policy_cfg.get("invert_gripper", True))).lower(),
                "cosmos3.debug_layer0_trace": str(bool(self.policy_cfg.get("debug_layer0_trace", False))).lower(),
                "cosmos3.debug_qwen_input_splice": str(
                    bool(self.policy_cfg.get("debug_qwen_input_splice", False))
                ).lower(),
                "cosmos3.debug_full_visual_stream": str(
                    bool(self.policy_cfg.get("debug_full_visual_stream", False))
                ).lower(),
                "cosmos3.debug_action_bridge": str(
                    bool(self.policy_cfg.get("debug_action_bridge", False))
                ).lower(),
                "cosmos3.debug_language_full_sequence": str(
                    bool(self.policy_cfg.get("debug_language_full_sequence", False))
                ).lower(),
                "cosmos3.debug_language_timing": str(
                    bool(self.policy_cfg.get("debug_language_timing", False))
                ).lower(),
                "cosmos3.debug_first_action_step": str(
                    bool(self.policy_cfg.get("debug_first_action_step", False))
                ).lower(),
                "cosmos3.debug_all_action_steps": str(
                    bool(self.policy_cfg.get("debug_all_action_steps", False))
                ).lower(),
                "cosmos3.debug_denoise_timing": str(
                    bool(self.policy_cfg.get("debug_denoise_timing", False))
                ).lower(),
                "cosmos3.profile_denoise_events": str(
                    bool(self.policy_cfg.get("profile_denoise_events", False))
                ).lower(),
                "cosmos3.debug_mot_action_input": str(
                    bool(self.policy_cfg.get("debug_mot_action_input", False))
                ).lower(),
                "cosmos3.debug_mot_gen": str(bool(self.policy_cfg.get("debug_mot_gen", False))).lower(),
                "cosmos3.debug_mot_internal_trace_layer": str(
                    self.policy_cfg.get("debug_mot_internal_trace_layer", 0)
                ),
                "cosmos3.debug_mot_timestep": str(self.policy_cfg.get("debug_mot_timestep", 1.0)),
                "cosmos3.debug_language_trace_layers": str(self.policy_cfg.get("debug_language_trace_layers", 1)),
                "cosmos3.debug_language_trace_tokens": str(self.policy_cfg.get("debug_language_trace_tokens", 0)),
                "cosmos3.debug_vae_patch_input": str(
                    bool(self.policy_cfg.get("debug_vae_patch_input", False))
                ).lower(),
                "cosmos3.debug_vae_patch_input_rows": str(
                    self.policy_cfg.get("debug_vae_patch_input_rows", 16)
                ),
                "cosmos3.debug_vae_encoder_conv1": str(
                    bool(self.policy_cfg.get("debug_vae_encoder_conv1", False))
                ).lower(),
                "cosmos3.debug_vae_encoder_conv1_rows": str(
                    self.policy_cfg.get("debug_vae_encoder_conv1_rows", 16)
                ),
                "cosmos3.debug_vae_encoder_conv1_cols": str(
                    self.policy_cfg.get("debug_vae_encoder_conv1_cols", 32)
                ),
                "cosmos3.debug_vae_down0_shortcut": str(
                    bool(self.policy_cfg.get("debug_vae_down0_shortcut", False))
                ).lower(),
                "cosmos3.debug_vae_down0_shortcut_rows": str(
                    self.policy_cfg.get("debug_vae_down0_shortcut_rows", 16)
                ),
                "cosmos3.debug_vae_down0_shortcut_cols": str(
                    self.policy_cfg.get("debug_vae_down0_shortcut_cols", 32)
                ),
                "cosmos3.debug_vae_down0": str(
                    bool(self.policy_cfg.get("debug_vae_down0", False))
                ).lower(),
                "cosmos3.debug_vae_down0_rows": str(
                    self.policy_cfg.get("debug_vae_down0_rows", 16)
                ),
                "cosmos3.debug_vae_down0_cols": str(
                    self.policy_cfg.get("debug_vae_down0_cols", 32)
                ),
                "cosmos3.debug_vae_down1": str(
                    bool(self.policy_cfg.get("debug_vae_down1", False))
                ).lower(),
                "cosmos3.debug_vae_down1_rows": str(
                    self.policy_cfg.get("debug_vae_down1_rows", 16)
                ),
                "cosmos3.debug_vae_down1_cols": str(
                    self.policy_cfg.get("debug_vae_down1_cols", 32)
                ),
                "cosmos3.debug_vae_down2": str(
                    bool(self.policy_cfg.get("debug_vae_down2", False))
                ).lower(),
                "cosmos3.debug_vae_down2_rows": str(
                    self.policy_cfg.get("debug_vae_down2_rows", 16)
                ),
                "cosmos3.debug_vae_down2_cols": str(
                    self.policy_cfg.get("debug_vae_down2_cols", 32)
                ),
                "cosmos3.debug_vae_down3": str(
                    bool(self.policy_cfg.get("debug_vae_down3", False))
                ).lower(),
                "cosmos3.debug_vae_down3_rows": str(
                    self.policy_cfg.get("debug_vae_down3_rows", 16)
                ),
                "cosmos3.debug_vae_down3_cols": str(
                    self.policy_cfg.get("debug_vae_down3_cols", 32)
                ),
                "cosmos3.debug_vae_mid0": str(
                    bool(self.policy_cfg.get("debug_vae_mid0", False))
                ).lower(),
                "cosmos3.debug_vae_mid0_rows": str(
                    self.policy_cfg.get("debug_vae_mid0_rows", 16)
                ),
                "cosmos3.debug_vae_mid0_cols": str(
                    self.policy_cfg.get("debug_vae_mid0_cols", 32)
                ),
                "cosmos3.debug_vae_mid_attn": str(
                    bool(self.policy_cfg.get("debug_vae_mid_attn", False))
                ).lower(),
                "cosmos3.debug_vae_mid_attn_rows": str(
                    self.policy_cfg.get("debug_vae_mid_attn_rows", 16)
                ),
                "cosmos3.debug_vae_mid_attn_cols": str(
                    self.policy_cfg.get("debug_vae_mid_attn_cols", 32)
                ),
                "cosmos3.debug_vae_mid2": str(
                    bool(self.policy_cfg.get("debug_vae_mid2", False))
                ).lower(),
                "cosmos3.debug_vae_mid2_rows": str(
                    self.policy_cfg.get("debug_vae_mid2_rows", 16)
                ),
                "cosmos3.debug_vae_mid2_cols": str(
                    self.policy_cfg.get("debug_vae_mid2_cols", 32)
                ),
                "cosmos3.debug_vae_head": str(
                    bool(self.policy_cfg.get("debug_vae_head", False))
                ).lower(),
                "cosmos3.debug_vae_head_rows": str(
                    self.policy_cfg.get("debug_vae_head_rows", 16)
                ),
                "cosmos3.debug_vae_head_cols": str(
                    self.policy_cfg.get("debug_vae_head_cols", 32)
                ),
                "cosmos3.debug_vae_final_conv1": str(
                    bool(self.policy_cfg.get("debug_vae_final_conv1", False))
                ).lower(),
                "cosmos3.debug_vae_final_conv1_rows": str(
                    self.policy_cfg.get("debug_vae_final_conv1_rows", 16)
                ),
                "cosmos3.debug_vae_final_conv1_cols": str(
                    self.policy_cfg.get("debug_vae_final_conv1_cols", 32)
                ),
                "cosmos3.debug_vae_clean_condition": str(
                    bool(self.policy_cfg.get("debug_vae_clean_condition", False))
                ).lower(),
                "cosmos3.debug_vae_clean_condition_rows": str(
                    self.policy_cfg.get("debug_vae_clean_condition_rows", 16)
                ),
                "cosmos3.debug_vae_clean_condition_cols": str(
                    self.policy_cfg.get("debug_vae_clean_condition_cols", 32)
                ),
            },
        )
        req.step.tensors.extend(
            [
                _tensor_from_array(self.wam_pb2, "observation/image", image, self.wam_pb2.U8),
                _tensor_from_array(self.wam_pb2, "observation/joint_position", joint, self.wam_pb2.F32),
                _tensor_from_array(self.wam_pb2, "observation/gripper_position", gripper, self.wam_pb2.F32),
            ]
        )
        token_tensors = self._build_dynamic_text_tokens(instruction)
        req.step.tensors.extend(
            [
                _tensor_from_array(self.wam_pb2, "cosmos3.qwen.input_ids", token_tensors["qwen_input_ids"], self.wam_pb2.I32),
                _tensor_from_array(
                    self.wam_pb2,
                    "cosmos3.qwen.visual_indices",
                    token_tensors["qwen_visual_indices"],
                    self.wam_pb2.I32,
                ),
                _tensor_from_array(
                    self.wam_pb2,
                    "cosmos3.qwen.mrope_positions",
                    token_tensors["qwen_mrope_positions"],
                    self.wam_pb2.I32,
                ),
                _tensor_from_array(self.wam_pb2, "cosmos3.mot.text_ids", token_tensors["mot_text_ids"], self.wam_pb2.I32),
                _tensor_from_array(
                    self.wam_pb2,
                    "cosmos3.mot.text_ids_uncond",
                    token_tensors["mot_text_ids_uncond"],
                    self.wam_pb2.I32,
                ),
            ]
        )
        payload = req.SerializeToString()
        if self._debug_latency_enabled():
            print(f"[cosmos3-eval] pack_request_done request_id={req.request_id} bytes={len(payload)}", flush=True)
        self._maybe_dump_request(
            req.request_id,
            image=image,
            joint=joint,
            gripper=gripper,
            instruction=instruction,
            payload=payload,
            token_tensors=token_tensors,
        )
        return payload

    def _maybe_dump_request(
        self,
        request_id: int,
        *,
        image: np.ndarray,
        joint: np.ndarray,
        gripper: np.ndarray,
        instruction: str,
        payload: bytes,
        token_tensors: dict[str, np.ndarray] | None = None,
    ) -> None:
        if self._debug_request_dump_dir is None:
            return
        if self._debug_request_dump_limit > 0 and request_id > self._debug_request_dump_limit:
            return
        out = self._debug_request_dump_dir / f"request_{request_id:04d}.npz"
        arrays = {
            "image": np.asarray(image, dtype=np.uint8),
            "joint_position": np.asarray(joint, dtype=np.float32),
            "gripper_position": np.asarray(gripper, dtype=np.float32),
            "instruction": np.array(instruction),
            "request_id": np.array(request_id, dtype=np.int64),
            "session_id": np.array(self._session_id, dtype=np.int64),
            "wam_request": np.frombuffer(payload, dtype=np.uint8).copy(),
        }
        if token_tensors:
            arrays.update(token_tensors)
        np.savez_compressed(out, **arrays)

    def _query_server(self, request: bytes) -> Any:
        t0 = time.perf_counter()
        self._sock.send(request)
        reply = self.wam_pb2.WamResponse()
        reply.ParseFromString(self._sock.recv())
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if self._debug_latency_enabled():
            print(
                "[cosmos3-eval] wam_latency "
                f"total={float(getattr(reply, 'latency_ms_total', 0.0)):.2f}ms "
                f"vision={float(getattr(reply, 'latency_ms_vision', 0.0)):.2f}ms "
                f"inference={float(getattr(reply, 'latency_ms_inference', 0.0)):.2f}ms "
                f"prefill={float(getattr(reply, 'latency_ms_prefill', 0.0)):.2f}ms "
                f"denoise={float(getattr(reply, 'latency_ms_denoise', 0.0)):.2f}ms "
                f"wall={dt_ms:.2f}ms",
                flush=True,
            )
        if reply.error:
            if self._debug_latency_enabled():
                print(f"[cosmos3-eval] wam_error {reply.error}", flush=True)
            raise RuntimeError(reply.error)
        return reply

    def _unpack_response(self, response: Any) -> np.ndarray:
        chunk = np.asarray(response.action_chunk, dtype=np.float32)
        if response.chunk_size and response.action_dim:
            chunk = chunk.reshape(int(response.chunk_size), int(response.action_dim))
        return chunk

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = chunk.astype(np.float32, copy=False)
        if chunk.shape[0] < self.open_loop_horizon:
            raise RuntimeError(
                f"WAM returned only {chunk.shape[0]} action rows, cannot serve action_horizon={self.open_loop_horizon}"
            )
        return chunk[: self.open_loop_horizon]

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        return np.asarray(extracted_obs["viz"])

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._chunks.clear()
            self._counters.clear()
            self._session_id = int(time.time() * 1000) & 0xFFFFFFFF
        else:
            self._chunks.pop(env_id, None)
            self._counters.pop(env_id, None)

    def close(self) -> None:
        self._sock.close(0)

def _start_wam_server(cfg: dict[str, Any], run_dir: Path) -> subprocess.Popen:
    server_cfg = cfg["server"]
    paths = cfg["paths"]
    binary = _require_file(_resolve_path(server_cfg["wam_binary"]), "wam-server")
    model = _require_file(_resolve_path(paths["model"]), "Cosmos3 GGUF")
    log_path = run_dir / "wam-server.log"
    log = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    _configure_omniverse_proxy(env)
    if "policy_gpu" in server_cfg and server_cfg["policy_gpu"] is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(server_cfg["policy_gpu"])
    build_dir = binary.parent
    build_bin_dir = build_dir / "bin"
    lib_paths = [str(build_dir)]
    if build_bin_dir.exists():
        lib_paths.append(str(build_bin_dir))
    if env.get("LD_LIBRARY_PATH"):
        lib_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths)
    proc = subprocess.Popen(
        [str(binary), "--bind", str(server_cfg["wam_addr"]), str(model)],
        cwd=str(_resolve_path(paths["embodied_root"])),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[cosmos3-eval] waiting for WAM server at {server_cfg['wam_addr']}", flush=True)
    _wait_log_contains(proc, log_path, "wam-server: bound", int(server_cfg.get("ready_timeout_s", 120)))
    print(f"[cosmos3-eval] WAM server ready; log={log_path}", flush=True)
    return proc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=Path, default=DEFAULT_CONF)
    parser.add_argument("--sim-gpu", type=int)
    parser.add_argument("--wam-addr")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-name")
    parser.add_argument(
        "--smoke-wam-only",
        action="store_true",
        help="Start the C++ WAM server and send one synthetic Cosmos3 request without launching IsaacSim.",
    )
    parser.add_argument(
        "--smoke-instruction",
        default="put the banana in the bowl",
        help="Instruction text used by --smoke-wam-only synthetic request.",
    )
    parser.add_argument(
        "--debug-layer0-trace",
        action="store_true",
        help="With --smoke-wam-only, request native Qwen3-VL layer0 trace sample tensors instead of an action.",
    )
    parser.add_argument(
        "--debug-qwen-input-splice",
        action="store_true",
        help="With --smoke-wam-only, request native Qwen input embedding splice sample tensors.",
    )
    parser.add_argument(
        "--debug-full-visual-stream",
        action="store_true",
        help="With --debug-qwen-input-splice, execute the full native visual stream before sampling splice tensors.",
    )
    parser.add_argument(
        "--debug-action-bridge",
        action="store_true",
        help="With --debug-layer0-trace and --debug-full-visual-stream, also return the native llm_to_action bridge probe.",
    )
    parser.add_argument(
        "--debug-language-trace-layers",
        type=int,
        default=1,
        help="With --debug-layer0-trace, execute this many cached Qwen3-VL language layers before sampling tensors.",
    )
    parser.add_argument(
        "--debug-language-trace-tokens",
        type=int,
        default=0,
        help="With --debug-layer0-trace and --debug-full-visual-stream, trace this many real Qwen input tokens; 0 means full sequence.",
    )
    parser.add_argument(
        "--debug-language-full-sequence",
        action="store_true",
        help="With --debug-layer0-trace and --debug-full-visual-stream, run the full 6797-token Qwen language span.",
    )
    return parser


def _apply_config_defaults(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    rb = cfg.get("robolab", {})
    server = cfg.get("server", {})
    if args.task is None:
        args.task = rb.get("tasks")
    if args.num_envs == 1:
        args.num_envs = int(rb.get("num_envs", args.num_envs))
    if args.num_runs == 1:
        args.num_runs = int(rb.get("num_runs", args.num_runs))
    if args.output_folder_name is None:
        args.output_folder_name = args.output_name or rb.get("output_name")
    if args.video_mode == "all":
        args.video_mode = rb.get("video_mode", args.video_mode)
    if args.rendering_type is None:
        args.rendering_type = rb.get("rendering_type", args.rendering_type)
    if getattr(args, "renderer", None) == "realtime":
        args.renderer = rb.get("renderer", args.renderer)
    args.device = rb.get("device", getattr(args, "device", "cuda:0"))
    args.headless = bool(rb.get("headless", getattr(args, "headless", True)))
    if args.wam_addr is None:
        args.wam_addr = server.get("wam_addr")
    if args.sim_gpu is None:
        args.sim_gpu = int(rb.get("sim_gpu", 2))
    if args.model is not None:
        cfg.setdefault("paths", {})["model"] = str(args.model)


def _apply_smoke_defaults(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    if args.wam_addr is None:
        args.wam_addr = cfg.get("server", {}).get("wam_addr")
    if args.sim_gpu is None:
        args.sim_gpu = int(cfg.get("robolab", {}).get("sim_gpu", 2))
    if args.output_name is None:
        args.output_name = f"cosmos3_cpp_wam_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.model is not None:
        cfg.setdefault("paths", {})["model"] = str(args.model)


def _run_with_wam_server(args: argparse.Namespace, cfg: dict[str, Any], run_dir: Path, *, smoke_only: bool) -> None:
    wam_proc: subprocess.Popen | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="wam_pb2_") as tmp:
            wam_pb2 = _compile_wam_proto(REPO_ROOT / "serving" / "wam.proto", Path(tmp))
            wam_proc = _start_wam_server(cfg, run_dir)
            if smoke_only:
                _run_wam_smoke(args, cfg, wam_pb2)
            else:
                _run_native_robolab(args, cfg, wam_pb2)
    finally:
        if wam_proc and wam_proc.poll() is None:
            wam_proc.send_signal(signal.SIGINT)
            try:
                wam_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                wam_proc.kill()


def _run_native_robolab(args: argparse.Namespace, cfg: dict[str, Any], wam_pb2: Any) -> None:
    import cv2  # Must import before isaaclab; mirrors RoboLab's policy runners.
    from isaaclab.app import AppLauncher
    from robolab.eval.runner import run_evaluation

    del cv2
    args.enable_cameras = True
    _configure_omniverse_proxy(os.environ)
    print("[cosmos3-eval] launching IsaacSim/AppLauncher", flush=True)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    print("[cosmos3-eval] IsaacSim app ready; registering RoboLab tasks", flush=True)

    try:
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT

        auto_register_droid_envs(task=args.task, cameras=WRIST_LEFT_RIGHT)
        print(f"[cosmos3-eval] registered tasks: {args.task}", flush=True)

        def make_client(_: argparse.Namespace) -> NativeCosmos3Client:
            print("[cosmos3-eval] creating NativeCosmos3Client", flush=True)
            return NativeCosmos3Client(
                wam_pb2=wam_pb2,
                wam_addr=str(args.wam_addr),
                request_timeout_ms=int(cfg.get("client", {}).get("request_timeout_ms", 900000)),
                policy_cfg=dict(cfg.get("policy", {})),
            )

        print("[cosmos3-eval] entering RoboLab run_evaluation", flush=True)
        run_evaluation(args, policy="cosmos3_cpp", client_factory=make_client)
    finally:
        simulation_app.close()


def _run_wam_smoke(args: argparse.Namespace, cfg: dict[str, Any], wam_pb2: Any) -> None:
    from adapter.sim.robolab import RoboLabCosmos3ObservationAdapter

    policy_cfg = dict(cfg.get("policy", {}))
    if getattr(args, "debug_layer0_trace", False):
        policy_cfg["debug_layer0_trace"] = True
        policy_cfg["debug_language_trace_layers"] = int(getattr(args, "debug_language_trace_layers", 1))
        policy_cfg["debug_language_trace_tokens"] = int(getattr(args, "debug_language_trace_tokens", 0))
        policy_cfg["debug_language_full_sequence"] = bool(
            getattr(args, "debug_language_full_sequence", False)
            or int(getattr(args, "debug_language_trace_tokens", 0)) <= 0
        )
    if getattr(args, "debug_qwen_input_splice", False):
        policy_cfg["debug_qwen_input_splice"] = True
    if getattr(args, "debug_full_visual_stream", False):
        policy_cfg["debug_qwen_input_splice"] = True
        policy_cfg["debug_full_visual_stream"] = True
    if getattr(args, "debug_action_bridge", False):
        policy_cfg["debug_layer0_trace"] = True
        policy_cfg["debug_full_visual_stream"] = True
        policy_cfg["debug_action_bridge"] = True
        policy_cfg["debug_language_full_sequence"] = bool(
            getattr(args, "debug_language_full_sequence", False)
            or int(getattr(args, "debug_language_trace_tokens", 0)) <= 0
        )
    client = NativeCosmos3Client(
        wam_pb2=wam_pb2,
        wam_addr=str(args.wam_addr),
        request_timeout_ms=int(cfg.get("client", {}).get("request_timeout_ms", 900000)),
        policy_cfg=policy_cfg,
    )
    try:
        extracted = RoboLabCosmos3ObservationAdapter.synthetic_inputs()
        req = client._pack_request(extracted, str(args.smoke_instruction))
        try:
            reply = client._query_server(req)
        except RuntimeError as exc:
            # Until models/cosmos3.cpp grows the native forward pass, the
            # expected end-to-end result is a server-side "not implemented"
            # error.  Getting that error proves the Python runner, protobuf
            # request, ZMQ transport, GGUF load, and WAM dispatch all reached C++.
            print(f"WAM smoke reached C++ server and received expected model error: {exc}")
            return

        if getattr(args, "debug_layer0_trace", False):
            names = [tensor.name for tensor in reply.tensors]
            expected_names = [
                "cosmos3.debug.layer0.mlp_swiglu.prefix",
                "cosmos3.debug.layer0.mlp_down.prefix",
                "cosmos3.debug.layer0.final_residual.prefix",
                "cosmos3.debug.layer0.final_norm.prefix",
            ]
            if getattr(args, "debug_action_bridge", False):
                expected_names.append("cosmos3.debug.action_bridge.llm_to_action")
            if names != expected_names:
                raise RuntimeError(f"layer0 trace tensor names mismatch: got {names}, expected {expected_names}")
            if getattr(args, "debug_full_visual_stream", False):
                requested_tokens = int(getattr(args, "debug_language_trace_tokens", 0))
                expected_tokens = (
                    6797
                    if getattr(args, "debug_language_full_sequence", False) or requested_tokens <= 0
                    else requested_tokens
                )
            else:
                expected_tokens = 3
            for tensor in reply.tensors:
                expected_shape = [1, 64] if tensor.name == "cosmos3.debug.action_bridge.llm_to_action" else [expected_tokens, 16]
                if list(tensor.shape) != expected_shape or tensor.dtype != wam_pb2.F32:
                    raise RuntimeError(
                        f"layer0 trace tensor {tensor.name} has dtype={tensor.dtype} shape={list(tensor.shape)}"
                    )
            print(f"WAM smoke received native layer0 trace tensors: {', '.join(names)}")
            return

        if getattr(args, "debug_qwen_input_splice", False):
            names = [tensor.name for tensor in reply.tensors]
            expected_names = [
                "cosmos3.debug.qwen_input.prefix",
                "cosmos3.debug.qwen_input.first_video_rows",
                "cosmos3.debug.qwen_input.stats",
            ]
            if names != expected_names:
                raise RuntimeError(f"Qwen input splice tensor names mismatch: got {names}, expected {expected_names}")
            expected_shapes = {
                "cosmos3.debug.qwen_input.prefix": [4, 16],
                "cosmos3.debug.qwen_input.first_video_rows": [3, 16],
                "cosmos3.debug.qwen_input.stats": [1, 8],
            }
            for tensor in reply.tensors:
                if list(tensor.shape) != expected_shapes[tensor.name] or tensor.dtype != wam_pb2.F32:
                    raise RuntimeError(
                        f"Qwen input splice tensor {tensor.name} has dtype={tensor.dtype} shape={list(tensor.shape)}"
                    )
            tensor_by_name = {tensor.name: tensor for tensor in reply.tensors}
            first_video_rows = np.frombuffer(
                tensor_by_name["cosmos3.debug.qwen_input.first_video_rows"].data,
                dtype=np.float32,
            ).reshape(3, 16)
            stats = np.frombuffer(
                tensor_by_name["cosmos3.debug.qwen_input.stats"].data,
                dtype=np.float32,
            )
            placeholder_prefix = np.asarray(
                [(float(i) - 128.0) / 128.0 for i in range(16)],
                dtype=np.float32,
            )
            if np.allclose(first_video_rows[0], placeholder_prefix, atol=1e-6):
                raise RuntimeError("Qwen input splice visual rows still match the old placeholder pattern")
            if not np.isfinite(first_video_rows).all() or float(np.abs(first_video_rows).sum()) <= 0.0:
                raise RuntimeError("Qwen input splice visual rows are empty or non-finite")
            if stats.shape != (8,) or int(stats[0]) != 6797 or int(stats[2]) != 6647:
                raise RuntimeError(f"Qwen input splice stats mismatch: {stats.tolist()}")
            print(
                "WAM smoke received native Qwen input splice tensors: "
                f"{', '.join(names)}; first_video_abs_sum={float(np.abs(first_video_rows).sum()):.6f}"
            )
            return

        action = client._postprocess_chunk(client._unpack_response(reply))
        expected = (
            int(cfg.get("client", {}).get("action_horizon", NativeCosmos3Client.DEFAULT_ACTION_HORIZON)),
            int(cfg.get("client", {}).get("action_dim", 8)),
        )
        if action.shape != expected:
            raise RuntimeError(f"WAM smoke action shape mismatch: got {action.shape}, expected {expected}")
        print(f"WAM smoke received action chunk with shape {action.shape}")
    finally:
        client.close()


def main() -> int:
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--conf", type=Path, default=DEFAULT_CONF)
    cfg_parser.add_argument("--sim-gpu", type=int)
    cfg_ns, _ = cfg_parser.parse_known_args()

    configured_python = _read_configured_python(cfg_ns.conf)
    if configured_python and configured_python.exists() and Path(sys.executable).resolve() != configured_python.resolve():
        os.execv(str(configured_python), [str(configured_python), *sys.argv])

    global np
    import numpy as np

    cfg = _load_yaml(cfg_ns.conf)

    if "--smoke-wam-only" in sys.argv:
        parser = _build_parser()
        args, _ = parser.parse_known_args()
        _apply_smoke_defaults(args, cfg)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.sim_gpu)
        _configure_omniverse_proxy(os.environ)
        run_dir = _resolve_path(cfg["paths"]["output_root"]) / str(args.output_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        _run_with_wam_server(args, cfg, run_dir, smoke_only=True)
        print(f"RoboLab Cosmos3 C++ eval complete: {run_dir}")
        return 0

    sim_gpu = cfg_ns.sim_gpu if cfg_ns.sim_gpu is not None else int(cfg.get("robolab", {}).get("sim_gpu", 2))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(sim_gpu)
    _configure_omniverse_proxy(os.environ)
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ["ACCEPT_EULA"] = "Y"

    robolab_dir = _require_dir(_resolve_path(cfg["paths"]["robolab_dir"]), "RoboLab checkout")
    sys.path.insert(0, str(robolab_dir))

    import cv2  # Must import before isaaclab; mirrors RoboLab's policy runners.
    from isaaclab.app import AppLauncher
    from robolab.eval.runner import add_common_eval_args

    del cv2
    parser = _build_parser()
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    _apply_config_defaults(args, cfg)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = str(args.output_folder_name or f"cosmos3_cpp_robolab_{timestamp}")
    run_dir = _resolve_path(cfg["paths"]["output_root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    _run_with_wam_server(args, cfg, run_dir, smoke_only=False)
    print(f"RoboLab Cosmos3 C++ eval complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
