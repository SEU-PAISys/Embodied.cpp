# Copyright 2026 SEU-PAISys
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np

from adapter.typed_io import EmbodiedObservation, ImageStream, TensorStream


def _as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu") and hasattr(x.cpu(), "numpy"):
        x = x.cpu().numpy()
    elif hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _resize_with_pad(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    import cv2

    image = np.asarray(image)
    h, w = image.shape[:2]
    scale = min(out_w / float(w), out_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((out_h, out_w, image.shape[2]), dtype=image.dtype)
    top = (out_h - new_h) // 2
    left = (out_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def _half_bilinear(image: np.ndarray) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    return cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR).astype(image.dtype)


class RoboLabCosmos3ObservationAdapter:
    """Convert RoboLab observations into the Cosmos3 WAM request tensors."""

    IMAGE_W = 640
    IMAGE_H = 360

    def parse_model_inputs(self, obs: dict[str, Any], *, env_id: int = 0) -> dict[str, Any]:
        image_obs = obs["image_obs"]
        proprio_obs = obs["proprio_obs"]
        left = _as_numpy(image_obs["over_shoulder_left_camera"][env_id])
        right = _as_numpy(image_obs["over_shoulder_right_camera"][env_id])
        wrist = _as_numpy(image_obs["wrist_cam"][env_id])

        left_image = _resize_with_pad(left, self.IMAGE_H, self.IMAGE_W)
        right_image = _resize_with_pad(right, self.IMAGE_H, self.IMAGE_W)
        wrist_image = _resize_with_pad(wrist, self.IMAGE_H, self.IMAGE_W)
        image = self.compose_cosmos3_image(
            left_image=left_image,
            right_image=right_image,
            wrist_image=wrist_image,
        )
        joint = _as_numpy(proprio_obs["arm_joint_pos"][env_id]).astype(np.float32).reshape(-1)
        gripper = _as_numpy(proprio_obs["gripper_pos"][env_id]).astype(np.float32).reshape(-1)

        return {
            "observation/image": image,
            "observation/joint_position": joint,
            "observation/gripper_position": gripper,
            "viz": self.build_visualization(
                left_image=left_image,
                right_image=right_image,
                wrist_image=wrist_image,
            ),
        }

    def parse_embodied_observation(
        self,
        obs: dict[str, Any],
        *,
        instruction: str = "",
        env_id: int = 0,
    ) -> EmbodiedObservation:
        model_inputs = self.parse_model_inputs(obs, env_id=env_id)
        return EmbodiedObservation(
            instruction=instruction,
            images=[
                ImageStream("observation/image", model_inputs["observation/image"], layout="HWC"),
            ],
            proprioception=TensorStream(
                "observation/state",
                np.concatenate(
                    [
                        model_inputs["observation/joint_position"],
                        model_inputs["observation/gripper_position"],
                    ]
                ).astype(np.float32),
            ),
            model_inputs=model_inputs,
            raw=obs,
        )

    @staticmethod
    def compose_cosmos3_image(
        *,
        left_image: np.ndarray,
        right_image: np.ndarray,
        wrist_image: np.ndarray,
    ) -> np.ndarray:
        left = _half_bilinear(np.asarray(left_image))
        right = _half_bilinear(np.asarray(right_image))
        image = np.concatenate((np.asarray(wrist_image), np.concatenate((left, right), axis=1)))
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    @staticmethod
    def build_visualization(
        *,
        left_image: np.ndarray,
        right_image: np.ndarray,
        wrist_image: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate((left_image, wrist_image, right_image), axis=1)

    @classmethod
    def synthetic_inputs(cls) -> dict[str, Any]:
        adapter = cls()
        zero = np.zeros((cls.IMAGE_H, cls.IMAGE_W, 3), dtype=np.uint8)
        image = adapter.compose_cosmos3_image(
            left_image=zero,
            right_image=zero,
            wrist_image=zero,
        )
        return {
            "observation/image": image,
            "observation/joint_position": np.zeros((7,), dtype=np.float32),
            "observation/gripper_position": np.zeros((1,), dtype=np.float32),
            "viz": adapter.build_visualization(left_image=zero, right_image=zero, wrist_image=zero),
        }
