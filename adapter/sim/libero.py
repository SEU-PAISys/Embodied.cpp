# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from typing import Any

import numpy as np

from adapter.pipeline import AdapterPipeline
from adapter.typed_io import EmbodiedObservation, ImageStream, TensorStream


class LingBotLIBEROParser:
    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        return obs

    def parse_embodied_observation(self, obs: dict[str, Any]) -> EmbodiedObservation:
        pixels = obs.get("pixels", {})
        images = []
        for key in ("image", "image2"):
            if key in pixels:
                images.append(ImageStream(key, np.asarray(pixels[key]), layout="HWC"))
        return EmbodiedObservation(
            instruction=str(obs.get("task_description", "")),
            images=images,
            proprioception=TensorStream("libero_state", _extract_libero_state(obs)),
            model_inputs=obs,
            raw=obs,
        )

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action[:7], dtype=np.float32)


class Pi05LIBEROParser:
    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        images = obs.get("pixels", {})
        proprio = _extract_pi05_libero_state(obs)
        task = str(obs.get("task_description", ""))
        model_inputs = {
            "observation.images.image": _to_chw_float01(images["image"]),
            "observation.state": proprio,
            "task": task,
        }
        if "image2" in images:
            model_inputs["observation.images.image2"] = _to_chw_float01(images["image2"])
        return model_inputs

    def parse_embodied_observation(self, obs: dict[str, Any]) -> EmbodiedObservation:
        pixels = obs.get("pixels", {})
        images = []
        for key in ("image", "image2"):
            if key in pixels:
                images.append(ImageStream(key, np.asarray(pixels[key]), layout="HWC"))
        return EmbodiedObservation(
            instruction=str(obs.get("task_description", "")),
            images=images,
            proprioception=TensorStream("libero_state", _extract_pi05_libero_state(obs)),
            model_inputs=self.parse_observation(obs),
            raw=obs,
        )

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action[:7], dtype=np.float32)


class GrootN1LIBEROParser:
    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        images = obs.get("pixels", {})
        model_inputs = {
            "observation.images.image": _to_chw_float01(images["image"]),
            "observation.state": _extract_pi05_libero_state(obs),
            "task": str(obs.get("task_description", "")),
        }
        if "image2" in images:
            model_inputs["observation.images.image2"] = _to_chw_float01(images["image2"])
        return model_inputs

    def parse_embodied_observation(self, obs: dict[str, Any]) -> EmbodiedObservation:
        pixels = obs.get("pixels", {})
        images = [
            ImageStream(key, np.asarray(pixels[key]), layout="HWC")
            for key in ("image", "image2")
            if key in pixels
        ]
        return EmbodiedObservation(
            instruction=str(obs.get("task_description", "")),
            images=images,
            proprioception=TensorStream("libero_state", _extract_pi05_libero_state(obs)),
            model_inputs=self.parse_observation(obs),
            raw=obs,
        )

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        result = np.asarray(action[:7], dtype=np.float32).copy()
        result[-1] = -np.sign(2.0 * result[-1] - 1.0)
        return result


LIBERO_PARSER_REGISTRY = {
    "pi05": Pi05LIBEROParser,
    "groot_n1": GrootN1LIBEROParser,
    "lingbot_va": LingBotLIBEROParser,
}


class LIBEROSimAdapter:
    def __init__(self, client: Any):
        arch = client.get_arch()
        parser_cls = LIBERO_PARSER_REGISTRY.get(arch)
        if parser_cls is None:
            raise ValueError(f"No parser found for architecture {arch}")
        self._client = client
        self._parser = parser_cls()
        self._adapter_pipeline = AdapterPipeline(self._parser)
        self.arch = arch

    def reset(self):
        return self._client.reset()

    def get_last_inference_profile(self) -> dict[str, float | int] | None:
        getter = getattr(self._client, "get_last_inference_profile", None)
        return getter() if getter is not None else None

    def parse_embodied_observation(self, obs: dict[str, Any]) -> EmbodiedObservation:
        return self._adapter_pipeline.parse_embodied_observation(obs)

    def get_action(self, obs: dict[str, Any]) -> Any:
        embodied_obs = self.parse_embodied_observation(obs)
        action = self._client.get_action(embodied_obs.model_inputs)
        return self._parser.parse_action(action)


def _extract_libero_state(obs: dict[str, Any]) -> np.ndarray:
    robot_state = obs.get("robot_state", {})
    eef = robot_state.get("eef", {})
    gripper = robot_state.get("gripper", {})
    pos = np.asarray(eef.get("pos", np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
    quat = np.asarray(eef.get("quat", np.array([0, 0, 0, 1])), dtype=np.float64).reshape(-1)[:4]
    qpos = np.asarray(gripper.get("qpos", np.zeros(1)), dtype=np.float32).reshape(-1)
    grip = qpos[:1] if qpos.size else np.zeros(1, dtype=np.float32)
    return np.concatenate([pos, _quat2axisangle(quat).astype(np.float32), grip]).astype(np.float32)


def _extract_pi05_libero_state(obs: dict[str, Any]) -> np.ndarray:
    robot_state = obs.get("robot_state", {})
    eef = robot_state.get("eef", {})
    gripper = robot_state.get("gripper", {})
    pos = np.asarray(eef.get("pos", np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
    quat = np.asarray(eef.get("quat", np.array([0, 0, 0, 1])), dtype=np.float64).reshape(-1)[:4]
    qpos = np.asarray(gripper.get("qpos", np.zeros(2)), dtype=np.float32).reshape(-1)
    grip = np.zeros(2, dtype=np.float32)
    grip[: min(2, qpos.size)] = qpos[:2]
    return np.concatenate([pos, _quat2axisangle(quat).astype(np.float32), grip]).astype(np.float32)


def _to_chw_float01(image: Any) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HWC image with 3 channels, got {arr.shape}")
    # openpi's LIBERO eval rotates both agentview and wrist images by 180 degrees
    # before resize/pad to match the training preprocessing.
    arr = np.ascontiguousarray(arr[::-1, ::-1])
    return np.transpose(arr / 255.0, (2, 0, 1)).astype(np.float32, copy=False)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if quat.size < 4:
        return np.zeros(3, dtype=np.float32)
    quat = quat[:4].copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if den <= 1e-12:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(quat[3]) / den).astype(np.float32)
