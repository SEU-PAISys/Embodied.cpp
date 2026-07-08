# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from adapter.typed_io import EmbodiedObservation, ImageStream, TensorStream


def _as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _lookup_nested(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def _first_existing(d: dict[str, Any], paths: Iterable[str]) -> Any:
    errors = []
    for path in paths:
        try:
            return _lookup_nested(d, path)
        except KeyError as e:
            errors.append(str(e))
    raise KeyError(f"none of {list(paths)} found; root keys={list(d)}")


def _image_to_chw_f32(img: Any) -> np.ndarray:
    arr = _as_numpy(img)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected image rank 3, got shape={arr.shape}")
    if arr.shape[0] == 3:
        chw = arr.astype(np.float32)
    elif arr.shape[-1] == 3:
        chw = np.transpose(arr, (2, 0, 1)).astype(np.float32)
    else:
        raise ValueError(f"expected RGB image with 3 channels, got shape={arr.shape}")
    if chw.max(initial=0.0) > 2.0:
        chw = chw / 255.0
    return np.ascontiguousarray(np.clip(chw, 0.0, 1.0), dtype=np.float32)


def _state_to_f32(state: Any, dim: int) -> np.ndarray:
    arr = _as_numpy(state).astype(np.float32).reshape(-1)
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, arr.size)
    out[:n] = arr[:n]
    return out


class RobotWinObservationAdapter:
    """Convert common RobotWin observations into embodied.cpp typed I/O."""

    DEFAULT_FRONT_KEYS = (
        "images.front",
        "images.image",
        "observation.images.front",
        "observation.images.image",
        "pixels.image",
        "front_image",
        "image",
    )
    DEFAULT_WRIST_KEYS = (
        "images.wrist",
        "images.image2",
        "observation.images.wrist",
        "observation.images.image2",
        "pixels.image2",
        "wrist_image",
        "image2",
    )
    DEFAULT_STATE_KEYS = (
        "observation.state",
        "state",
        "robot_state",
        "agent_pos",
        "qpos",
    )
    DEFAULT_TASK_KEYS = (
        "task_description",
        "instruction",
        "language_instruction",
        "task",
    )

    def __init__(
        self,
        *,
        front_key: str | None = None,
        wrist_key: str | None = None,
        state_key: str | None = None,
        task_key: str | None = None,
        state_dim: int = 20,
        default_task: str = "",
    ):
        self.front_keys = (front_key,) if front_key else self.DEFAULT_FRONT_KEYS
        self.wrist_keys = (wrist_key,) if wrist_key else self.DEFAULT_WRIST_KEYS
        self.state_keys = (state_key,) if state_key else self.DEFAULT_STATE_KEYS
        self.task_keys = (task_key,) if task_key else self.DEFAULT_TASK_KEYS
        self.state_dim = state_dim
        self.default_task = default_task

    def parse_model_inputs(self, obs: dict[str, Any]) -> dict[str, Any]:
        front = _image_to_chw_f32(_first_existing(obs, self.front_keys))
        try:
            wrist = _image_to_chw_f32(_first_existing(obs, self.wrist_keys))
        except KeyError:
            wrist = np.zeros_like(front)
        state = _state_to_f32(_first_existing(obs, self.state_keys), self.state_dim)
        try:
            task = _first_existing(obs, self.task_keys)
        except KeyError:
            task = self.default_task
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""
        if isinstance(task, bytes):
            task = task.decode()
        return {
            "observation.images.image": front,
            "observation.images.image2": wrist,
            "observation.state": state,
            "task": str(task),
        }

    def parse_embodied_observation(self, obs: dict[str, Any]) -> EmbodiedObservation:
        model_inputs = self.parse_model_inputs(obs)
        return EmbodiedObservation(
            instruction=str(model_inputs.get("task", "")),
            images=[
                ImageStream(
                    "observation.images.image",
                    model_inputs["observation.images.image"],
                    layout="CHW",
                ),
                ImageStream(
                    "observation.images.image2",
                    model_inputs["observation.images.image2"],
                    layout="CHW",
                ),
            ],
            proprioception=TensorStream(
                "observation.state",
                model_inputs["observation.state"],
            ),
            model_inputs=model_inputs,
            raw=obs,
        )


class RobotWinHyVLAAdapter:
    """Client-facing RobotWin policy adapter used by existing eval scripts."""

    def __init__(
        self,
        client: Any,
        *,
        front_key: str | None = None,
        wrist_key: str | None = None,
        state_key: str | None = None,
        task_key: str | None = None,
        state_dim: int = 20,
        action_dim: int = 20,
        default_task: str = "",
    ):
        self._client = client
        self._observation_adapter = RobotWinObservationAdapter(
            front_key=front_key,
            wrist_key=wrist_key,
            state_key=state_key,
            task_key=task_key,
            state_dim=state_dim,
            default_task=default_task,
        )
        self.action_dim = action_dim

    def reset(self) -> None:
        return self._client.reset()

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self._observation_adapter.parse_model_inputs(obs)

    def parse_embodied_observation(self, obs: dict[str, Any]) -> EmbodiedObservation:
        return self._observation_adapter.parse_embodied_observation(obs)

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32).reshape(-1)[: self.action_dim]

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        embodied_obs = self.parse_embodied_observation(obs)
        return self.parse_action(self._client.get_action(embodied_obs.model_inputs))
