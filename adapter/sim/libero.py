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


class TurboVLALIBEROParser(Pi05LIBEROParser):
    """TurboVLA parser for LIBERO.

    TurboVLA adopts pi0.5's two-camera / 8-D state / task-text LIBERO layout, so
    observation parsing and CHW float01 image preprocessing (rotation included)
    are inherited. The C++ runtime already de-normalises the 7-DoF arm into
    world units and maps the gripper to a binary +1/-1, so no transform is
    applied to the returned action otherwise.
    """

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action[:7], dtype=np.float32)


class XiaomiRobotics0LIBEROParser(Pi05LIBEROParser):
    """Xiaomi-Robotics-0 parser for LIBERO.

    Xiaomi-Robotics-0 adopts the same two-camera / task-text LIBERO layout as pi0.5, so
    observation parsing and CHW float01 image preprocessing are inherited.
    The 8-D LIBERO state is zero-padded to the 32-D proprioceptive input in
    the client (VlaCppClient._predict_chunk_xr0); images stay at their native
    resolution (must be a multiple of 32) and the action output is the 7-DoF
    arm + gripper taken from the 30-step chunk.
    """

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action[:7], dtype=np.float32)


def _to_chw_float01_raw(image: Any) -> np.ndarray:
    """CHW float01 without the 180-degree flip (X-VLA wrist camera)."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HWC image with 3 channels, got {arr.shape}")
    return np.transpose(arr / 255.0, (2, 0, 1)).astype(np.float32, copy=False)


def _rotate6d_to_axisangle(r6d: np.ndarray) -> np.ndarray:
    import robosuite.utils.transform_utils as T

    a1 = np.asarray(r6d, dtype=np.float64).reshape(-1)[0:3]
    a2 = np.asarray(r6d, dtype=np.float64).reshape(-1)[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-6)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-6)
    b3 = np.cross(b1, b2)
    quat = T.mat2quat(np.stack([b1, b2, b3], axis=-1))
    return np.asarray(T.quat2axisangle(quat), dtype=np.float64)


def _extract_xvla_libero_state(obs: dict[str, Any]) -> np.ndarray:
    """Official X-VLA LIBERO proprioception: [pos(3), ori6d(6), grip(1)]
    followed by a zero legacy copy (20 dims); the gripper channel stays 0."""
    robot_state = obs.get("robot_state", {})
    eef = robot_state.get("eef", {})
    pos = np.asarray(eef.get("pos", np.zeros(3)), dtype=np.float64).reshape(-1)[:3]
    mat = np.asarray(eef.get("mat"), dtype=np.float64).reshape(3, 3)
    ori6d = np.concatenate([mat[:3, 0], mat[:3, 1]])
    cur = np.concatenate([pos, ori6d, [0.0]]).astype(np.float32)
    return np.concatenate([cur, np.zeros_like(cur)]).astype(np.float32)


class XVLALIBEROParser(Pi05LIBEROParser):
    """X-VLA parser for LIBERO (mirrors the official evaluation/libero client).

    - proprio: [pos3, ori6d6, grip1] + zero legacy copy -> 20 dims, gripper 0
    - agentview image flipped 180 degrees, wrist camera sent unflipped
      (matches the official client's _flip_agentview usage)
    - domain_id fixed to 3 (the embodiment row X-VLA-Libero is evaluated with)
    - actions: chunk rows are absolute ee6d targets [pos3, rot6d6, grip1];
      converted to [pos3, axis-angle3, grip(+/-1)] for the LIBERO env running
      in absolute control mode (robot.controller.use_delta=False).
    """

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        images = obs.get("pixels", {})
        model_inputs = {
            "observation.images.image": _to_chw_float01(images["image"]),
            "observation.state": _extract_xvla_libero_state(obs),
            "task": str(obs.get("task_description", "")),
            "domain_id": 3,
        }
        if "image2" in images:
            model_inputs["observation.images.image2"] = _to_chw_float01_raw(images["image2"])
        return model_inputs

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(-1)[:10]
        pos = a[0:3]
        ori = _rotate6d_to_axisangle(a[3:9])
        grip = 1.0 if a[9] > 0.5 else -1.0
        return np.concatenate([pos, ori, [grip]]).astype(np.float32)

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


LIBERO_PARSER_REGISTRY = {
    "pi05": Pi05LIBEROParser,
    # SmolVLA shares pi0.5's 8-D LIBERO state and two-camera layout.
    "smolvla": Pi05LIBEROParser,
    "groot_n1": GrootN1LIBEROParser,
    "lingbot_va": LingBotLIBEROParser,
    "turbovla": TurboVLALIBEROParser,
    "xr0": XiaomiRobotics0LIBEROParser,
    "xvla": XVLALIBEROParser,
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

    def reset(self, **kwargs):
        return self._client.reset(**kwargs)

    def get_last_inference_profile(self) -> dict[str, float | int] | None:
        getter = getattr(self._client, "get_last_inference_profile", None)
        return getter() if getter is not None else None

    def parse_embodied_observation(self, obs: dict[str, Any]) -> EmbodiedObservation:
        return self._adapter_pipeline.parse_embodied_observation(obs)

    def get_action(self, obs: dict[str, Any]) -> Any:
        has_queued_action = getattr(self._client, "has_queued_action", None)
        get_action_from_queue = getattr(self._client, "get_action_from_queue", None)
        if (
            callable(has_queued_action)
            and callable(get_action_from_queue)
            and has_queued_action()
        ):
            return self._parser.parse_action(get_action_from_queue())
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
