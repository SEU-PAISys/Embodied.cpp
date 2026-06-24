# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Typed Python I/O objects for simulation and dataset adapters.

The C++ runtime has the authoritative low-level ABI. These dataclasses provide
the same boundary on the Python evaluation side without forcing every existing
client to change its model-specific input keys at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ImageStream:
    name: str
    data: np.ndarray
    layout: str = "CHW"
    color: str = "RGB"
    timestamp_ns: int | None = None


@dataclass
class TensorStream:
    name: str
    data: np.ndarray
    timestamp_ns: int | None = None


@dataclass
class EmbodiedObservation:
    """Observation at the embodied runtime boundary.

    `model_inputs` is a temporary bridge for the current vla.cpp-style clients.
    New adapters should populate typed fields first and then derive
    `model_inputs` as the compatibility view.
    """

    instruction: str = ""
    images: list[ImageStream] = field(default_factory=list)
    proprioception: TensorStream | None = None
    action_history: TensorStream | None = None
    extra_inputs: dict[str, Any] = field(default_factory=dict)
    model_inputs: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] | None = None
    timestamp_ns: int | None = None


@dataclass
class ActionChunk:
    values: np.ndarray
    steps: int
    action_dim: int
    timestamp_ns: int | None = None
    model_outputs: Any = None
