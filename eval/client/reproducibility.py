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


def noise_checksum(noise: np.ndarray) -> str:
    """Stable short fingerprint of the exact noise payload sent to a model."""
    return hashlib.sha256(
        np.ascontiguousarray(noise, dtype=np.float32).tobytes()
    ).hexdigest()[:16]


def generate_xr0_noise(seed: int, *, device, dtype) -> np.ndarray:
    """Mirror XR0's contiguous [1,30,32] randn_like(action_mask).

    RNG backend and dtype are part of the protocol, not inferred from the
    GGUF filename. An isolated generator avoids changing global RNG state.
    """
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((1, 30, 32), generator=generator, device=device,
                       dtype=dtype).float().cpu().numpy()
