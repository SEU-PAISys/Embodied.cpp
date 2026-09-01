"""Measure Xiaomi-Robotics-0 end-to-end latency against a running vla-server.

Loads the same synthetic LIBERO-like observation as smoke_xr0.py and issues
N repeat requests, printing per-phase latency read from the server response
(latency_ms_total / _inference / _prefill / _denoise / _vision) plus the
round-trip wall time. Reuses the tokenizer/arch from smoke_xr0.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval" / "client"))

from vla_cpp_client import VlaCppClient  # noqa: E402


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    addr = "tcp://127.0.0.1:5555"
    tokenizer = str(REPO_ROOT / "checkpoints" / "xr0" / "hf")

    client = VlaCppClient(
        addr,
        arch="xr0",
        tokenizer_name=tokenizer,
        real_action_dim=7,
        n_action_steps=30,
        image_keys=("__base__", "__wrist__"),
    )

    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(3, 256, 256), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(3, 256, 256), dtype=np.uint8)
    state = np.zeros(32, dtype=np.float32)
    state[0] = 0.1
    state[1] = -0.2
    obs = {
        "__base__": base,
        "__wrist__": wrist,
        "observation.state": state,
        "task": "pick up the black bowl and place it on the plate",
    }

    # warmup (first inference includes CUDA graph alloc / mmproj warm)
    client._predict_chunk_xr0(obs)

    rows = []
    for i in range(n):
        t0 = time.perf_counter()
        chunk = client._predict_chunk_xr0(obs)
        wall = (time.perf_counter() - t0) * 1000.0
        r = client._last_response
        rows.append((
            wall,
            float(getattr(r, "latency_ms_total", float("nan"))),
            float(getattr(r, "latency_ms_inference", float("nan"))),
            float(getattr(r, "latency_ms_prefill", float("nan"))),
            float(getattr(r, "latency_ms_denoise", float("nan"))),
            float(getattr(r, "latency_ms_vision", float("nan"))),
        ))
        assert np.isfinite(chunk).all() and chunk.shape == (30, 32)
        time.sleep(0.05)

    a = np.asarray(rows, dtype=np.float64)
    name = ["wall_rt", "total", "inference", "prefill", "denoise", "vision"]
    print(f"Xiaomi-Robotics-0  n_reqs={n}")
    for j, nm in enumerate(name):
        col = a[:, j]
        print(f"  {nm:<12} mean={col.mean():8.2f}ms  median={np.median(col):8.2f}ms  "
              f"p95={np.percentile(col, 95):8.2f}ms")
    client.close()


if __name__ == "__main__":
    main()