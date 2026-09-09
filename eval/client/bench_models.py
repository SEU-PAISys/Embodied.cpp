"""Measure per-chunk inference latency for pi05 / hy_vla / xr0 / turbovla / xvla.

Each model runs one predict round-trip per iteration using the same synthetic
LIBERO-style observation (2x 256x256 views + state + task). Warmup first, then
report wall-clock round-trip and server-reported latency phases from
latency_ms_total / _inference / _prefill / _denoise / _vision.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from client.vla_cpp_client import ARCH_PRESETS  # noqa: E402
from client.vla_cpp_client import VlaCppClient  # noqa: E402

ARCHES = {
    "pi05": dict(
        arch="pi05", tokenizer=None, image_size=224, max_state_dim=32,
        max_length=200, n_action_steps=5, image_keys=("observation.images.image",
                                                      "observation.images.image2"),
        state_dim=32, task="pick up the black bowl and place it on the plate",
    ),
    "hy_vla": dict(
        arch="hy_vla", tokenizer=None, image_size=224, max_state_dim=32,
        max_length=48, n_action_steps=1, image_keys=("observation.images.image",
                                                     "observation.images.image2"),
        state_dim=32, task="pick up the black bowl and place it on the plate",
    ),
    "xr0": dict(
        arch="xr0", tokenizer=str(REPO_ROOT / "checkpoints" / "xr0" / "hf"),
        image_size=None, max_state_dim=32, max_length=512, n_action_steps=10,
        image_keys=("__base__", "__wrist__"), state_dim=32,
        task="pick up the black bowl and place it on the plate",
    ),
    "turbovla": dict(
        arch="turbovla", tokenizer=None, image_size=256, max_state_dim=8,
        max_length=64, n_action_steps=12,
        image_keys=("observation.images.image", "observation.images.image2"),
        state_dim=8, task="pick up the black bowl and place it on the plate",
    ),
    "xvla": dict(
        arch="xvla", tokenizer=str(REPO_ROOT / "checkpoints" / "xvla" / "hf"),
        image_size=224, max_state_dim=20,
        max_length=50, n_action_steps=30,
        image_keys=("observation.images.image", "observation.images.image2"),
        state_dim=20, task="pick up the black bowl and place it on the plate",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch", choices=sorted(ARCHES))
    parser.add_argument("n", type=int, nargs="?", default=10)
    parser.add_argument("--vla-addr", default="tcp://127.0.0.1:5555")
    parser.add_argument("--tokenizer", help="Override the architecture's tokenizer path.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--domain-id", type=int, default=3,
                        help="X-VLA domain (3 for the LIBERO checkpoint).")
    args = parser.parse_args()
    if args.n < 1 or args.warmup < 1:
        parser.error("n and --warmup must be positive")
    if args.domain_id < 0:
        parser.error("--domain-id must be nonnegative")
    arch, n, addr = args.arch, args.n, args.vla_addr
    cfg = ARCHES[arch]

    client = VlaCppClient(
        addr,
        arch=cfg["arch"],
        tokenizer_name=args.tokenizer if args.tokenizer is not None else cfg["tokenizer"],
        image_size=cfg["image_size"],
        max_state_dim=cfg["max_state_dim"],
        max_length=cfg["max_length"],
        n_action_steps=cfg["n_action_steps"],
        image_keys=cfg["image_keys"],
        real_action_dim=7,
    )

    rng = np.random.default_rng(0)
    if arch == "xr0":
        base = rng.integers(0, 256, size=(3, 256, 256), dtype=np.uint8)
        wrist = rng.integers(0, 256, size=(3, 256, 256), dtype=np.uint8)
    else:
        base = rng.random((3, 256, 256), dtype=np.float32)
        wrist = rng.random((3, 256, 256), dtype=np.float32)
    state = np.zeros(cfg["state_dim"], dtype=np.float32)
    state[0] = 0.1
    state[1] = -0.2
    obs = {cfg["image_keys"][0]: base, cfg["image_keys"][1]: wrist,
           "observation.state": state, "task": cfg["task"]}
    if arch == "xvla":
        obs["domain_id"] = args.domain_id

    for _ in range(args.warmup):
        client._predict_chunk(obs)  # CUDA allocation / graph warmup

    rows = []
    for _ in range(n):
        t0 = time.perf_counter()
        chunk = client._predict_chunk(obs)
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
        assert np.isfinite(chunk).all() and chunk.ndim == 2
        time.sleep(0.05)

    a = np.asarray(rows, dtype=np.float64)
    name = ["wall_rt", "total", "inference", "prefill", "denoise", "vision"]
    # Phases the arch declares unmeasured (fused single-graph runtimes) are
    # reported as n/a instead of a misleading 0.00ms statistic.
    unmeasured = set(ARCH_PRESETS.get(arch, {}).get("unmeasured_phases", ()))
    for nm in unmeasured & set(name):
        a[:, name.index(nm)] = np.nan
    print(f"ARCH={arch}  n_reqs={n}  warmup={args.warmup}  "
          f"chunk={chunk.shape[0]}x{chunk.shape[1]}")
    print(f"  endpoint={addr}  tokenizer={args.tokenizer or cfg['tokenizer']}")
    if arch == "xvla":
        print(f"  domain_id={args.domain_id}")
    for j, nm in enumerate(name):
        col = a[:, j]
        if np.isnan(col).all():
            print(f"  {nm:<10} n/a (not measured separately for this architecture)")
            continue
        print(f"  {nm:<10} mean={col.mean():8.2f}ms  std={col.std():8.2f}ms  "
              f"p50={np.percentile(col, 50):8.2f}ms  p95={np.percentile(col, 95):8.2f}ms  "
              f"p99={np.percentile(col, 99):8.2f}ms")
    client.close()


if __name__ == "__main__":
    main()
