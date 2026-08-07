"""Deterministic seeds and action noise for evaluation clients."""

from __future__ import annotations

import hashlib

import numpy as np


def derive_episode_noise_seed(
    base_seed: int,
    suite: str,
    task_id: int,
    episode: int,
) -> int:
    """Return a stable uint64 seed unique to a benchmark episode."""
    if base_seed < 0 or task_id < 0 or episode < 0:
        raise ValueError("base_seed, task_id, and episode must be non-negative")
    payload = f"{base_seed}\0{suite}\0{task_id}\0{episode}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def generate_action_noise(
    rng: np.random.Generator,
    chunk_size: int,
    action_dim: int,
) -> np.ndarray:
    """Generate the exact contiguous float32 payload expected by the server."""
    if chunk_size <= 0 or action_dim <= 0:
        raise ValueError("chunk_size and action_dim must be positive")
    return np.ascontiguousarray(
        rng.standard_normal((chunk_size, action_dim), dtype=np.float32)
    )
