#!/usr/bin/env python3
"""Replay the TurboVLA PyTorch parity fixture through the C++ server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "eval" / "client"))

from vla_cpp_client import VlaCppClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-dir", required=True, type=Path)
    parser.add_argument("--address", default="tcp://127.0.0.1:5555")
    parser.add_argument("--atol", type=float, default=0.01)
    args = parser.parse_args()

    fixture = np.load(args.parity_dir / "turbovla_parity_inputs.npz")
    images = fixture["images_chw"]
    observation = {
        "observation.images.image": images[0],
        "observation.images.image2": images[1],
        "observation.state": fixture["state"],
        "task": str(fixture["instruction"]),
    }
    client = VlaCppClient(args.address, arch="turbovla")
    try:
        actual = client._predict_chunk(observation)
    finally:
        client.close()

    reference = np.load(args.parity_dir / "turbovla_parity_ref_env.npy")
    if actual.shape != reference.shape:
        raise SystemExit(f"shape mismatch: C++ {actual.shape}, PyTorch {reference.shape}")
    delta = np.abs(actual - reference)
    worst = np.unravel_index(int(delta.argmax()), delta.shape)
    np.save(args.parity_dir / "turbovla_parity_cpp_env.npy", actual)
    print(f"shape={actual.shape}")
    print(f"max_abs={delta.max():.8f} mean_abs={delta.mean():.8f} worst={worst}")
    print(f"pytorch[0]={reference[0].tolist()}")
    print(f"cpp[0]={actual[0].tolist()}")

    stage_reference = np.load(args.parity_dir / "turbovla_parity_ref_stages.npz")
    for name in (
        "dino",
        "vision_projected",
        "bert",
        "text_projected",
        "vision_fused",
        "text_fused",
        "state_tokens",
    ):
        expected = stage_reference[name].reshape(-1)
        observed = np.fromfile(args.parity_dir / f"turbovla_cpp_{name}.f32", dtype="<f4")
        if observed.size != expected.size:
            raise SystemExit(f"stage {name}: size mismatch {observed.size} != {expected.size}")
        stage_delta = np.abs(observed - expected)
        cosine = float(np.dot(observed, expected) / (np.linalg.norm(observed) * np.linalg.norm(expected)))
        print(
            f"stage={name:<18} max_abs={stage_delta.max():.8f} "
            f"mean_abs={stage_delta.mean():.8f} cosine={cosine:.8f}"
        )
        if name == "vision_projected":
            shaped = stage_reference[name].reshape(2, 256, 256)
            interleaved = shaped.transpose(1, 0, 2).reshape(-1)
            interleaved_cosine = float(
                np.dot(observed, interleaved)
                / (np.linalg.norm(observed) * np.linalg.norm(interleaved))
            )
            print(f"stage={'vision_interleaved':<18} cosine={interleaved_cosine:.8f}")
            print(f"vision_projected pytorch[:8]={expected[:8].tolist()}")
            print(f"vision_projected cpp[:8]={observed[:8].tolist()}")

    if not np.isfinite(actual).all() or float(delta.max()) > args.atol:
        raise SystemExit(f"FAIL: expected max_abs <= {args.atol}")
    print(f"PASS: max_abs <= {args.atol}")


if __name__ == "__main__":
    main()
