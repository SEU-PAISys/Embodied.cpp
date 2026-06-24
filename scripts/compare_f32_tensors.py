#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Compare two raw float32 tensor dumps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lhs", type=Path)
    parser.add_argument("rhs", type=Path)
    parser.add_argument("--shape", nargs="+", type=int, default=None)
    parser.add_argument("--segments", nargs="*", type=int, default=None,
                        help="Optional sequence segment lengths for per-segment stats.")
    args = parser.parse_args()

    lhs = np.fromfile(args.lhs, dtype=np.float32)
    rhs = np.fromfile(args.rhs, dtype=np.float32)
    if lhs.size != rhs.size:
        raise SystemExit(f"size mismatch: {args.lhs} has {lhs.size}, {args.rhs} has {rhs.size}")
    if args.shape:
        lhs = lhs.reshape(args.shape)
        rhs = rhs.reshape(args.shape)

    diff = np.abs(lhs - rhs)
    print(f"lhs sum={float(lhs.sum()):.12f} min={float(lhs.min()):.12f} max={float(lhs.max()):.12f}")
    print(f"rhs sum={float(rhs.sum()):.12f} min={float(rhs.min()):.12f} max={float(rhs.max()):.12f}")
    print(
        f"all mean_abs={float(diff.mean()):.12f} "
        f"max_abs={float(diff.max()):.12f} "
        f"rmse={float(np.sqrt((diff * diff).mean())):.12f}"
    )

    if args.segments:
        if lhs.ndim < 2:
            raise SystemExit("--segments requires a shaped tensor with sequence dimension at axis 0")
        start = 0
        for idx, length in enumerate(args.segments):
            end = start + length
            seg = diff[start:end]
            print(
                f"segment{idx} [{start}:{end}] "
                f"mean_abs={float(seg.mean()):.12f} "
                f"max_abs={float(seg.max()):.12f} "
                f"lhs_sum={float(lhs[start:end].sum()):.12f} "
                f"rhs_sum={float(rhs[start:end].sum()):.12f}"
            )
            start = end
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
