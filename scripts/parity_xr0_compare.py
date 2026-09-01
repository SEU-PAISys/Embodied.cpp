#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare the C++ Xiaomi-Robotics-0 output against the PyTorch reference.

  python scripts/parity_xr0_compare.py --parity-dir checkpoints/xr0 [--atol 5e-3]

Checks xr0_parity_out.bin (C++, world units) vs xr0_parity_ref.bin
(PyTorch float32 reference). Exit code 0 = within tolerance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity-dir", required=True, type=Path)
    ap.add_argument("--atol", type=float, default=5e-3,
                    help="absolute tolerance on world-unit actions (fp32 "
                         "reference vs fp32/bf16 C++ weights)")
    args = ap.parse_args()

    ref = np.fromfile(args.parity_dir / "xr0_parity_ref.bin", dtype="<f4")
    out = np.fromfile(args.parity_dir / "xr0_parity_out.bin", dtype="<f4")
    if ref.size % 32 or out.size % 32 or ref.size != out.size:
        raise SystemExit(f"size mismatch: ref={ref.size} out={out.size}")

    ref = ref.reshape(-1, 32)
    out = out.reshape(-1, 32)

    diff = np.abs(ref - out)
    rel = diff / (np.abs(ref) + 1e-6)
    print(f"max abs err : {diff.max():.6f}")
    print(f"mean abs err: {diff.mean():.6f}")
    print(f"max rel err : {rel.max():.4f}")
    print(f"ref  absmax : {np.abs(ref).max():.4f}")
    print(f"ref[0,:7]  :", np.round(ref[0, :7], 4))
    print(f"out[0,:7]  :", np.round(out[0, :7], 4))

    # location of the max abs error + per-dim / per-step breakdown
    amax = np.unravel_index(diff.argmax(), diff.shape)
    print(f"argmax err @ step={amax[0]} dim={amax[1]} "
          f"ref={ref[amax]:.6f} out={out[amax]:.6f}")
    per_dim = diff.max(axis=0)
    print("per-dim max err:", np.round(per_dim, 5))
    print("dim of max per-dim err:", int(per_dim.argmax()))
    per_step = diff.max(axis=1)
    print("per-step max err:", np.round(per_step, 5))
    print("step of max per-step err:", int(per_step.argmax()))
    # which elements exceed atol
    n_bad = int((diff > args.atol).sum())
    print(f"elements > atol({args.atol}): {n_bad} "
          f"({100.0*n_bad/diff.size:.2f}%)")
    if n_bad:
        bad = np.argwhere(diff > args.atol)
        print("first bad (step,dim):", bad[:10].tolist())

    ok = diff.max() <= args.atol
    print("PARITY: " + ("PASS" if ok else "FAIL") + f" (atol={args.atol})")
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
