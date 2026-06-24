#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Print LingBot-VA FlowMatchScheduler reference values for C++ parity."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
LINGBOT_REPO = REPO.parent / "lingbot-va-main"


def load_scheduler_cls():
    path = LINGBOT_REPO / "wan_va/utils/scheduler.py"
    spec = importlib.util.spec_from_file_location("lingbot_scheduler_direct", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlowMatchScheduler


def deterministic(n: int, scale: float) -> torch.Tensor:
    x = np.sin(np.arange(n, dtype=np.float32) * np.float32(0.013)) * np.float32(scale)
    return torch.from_numpy(x.copy())


def run_one(label: str, steps: int, shift: float) -> None:
    FlowMatchScheduler = load_scheduler_cls()
    sched = FlowMatchScheduler(shift=shift)
    sched.set_timesteps(steps)
    sample = deterministic(16, 0.2)
    model_output = deterministic(16, 0.03)
    for t in sched.timesteps:
        sample = sched.step(model_output, t, sample)
    print(
        f"{label} steps={steps} shift={shift:g} "
        f"sigma0={sched.sigmas[0].item():.9g} sigma_last={sched.sigmas[-1].item():.9g} "
        f"t0={sched.timesteps[0].item():.9g} t_last={sched.timesteps[-1].item():.9g} "
        f"checksum={sample.sum().item():.9g} max_abs={sample.abs().max().item():.9g}"
    )


def main() -> int:
    run_one("video", 20, 5.0)
    run_one("action", 50, 0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
