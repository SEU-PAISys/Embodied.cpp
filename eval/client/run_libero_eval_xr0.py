#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
# Licensed under the Apache License, Version 2.0.

"""Compatibility CLI for XR0; delegates rollouts to the shared LIBERO runner.

Prefer run_sim_client_direct.py --conf libero_xr0_eval.yaml for new runs.
This wrapper preserves the old task/host/replan flags, but writes the shared
<output-dir>/xr0/libero_<suite>/task_<id>/{result.json,summary.txt} layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def shared_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--task", default="libero_object/task_0")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="outputs/xr0_legacy")
    parser.add_argument("--profile-output", default=None)
    args = parser.parse_args(argv)
    suite, sep, task = args.task.partition("/task_")
    if not sep or not task.isdigit():
        parser.error("--task must be a suite/task_N, e.g. libero_object/task_0")
    config = Path(__file__).resolve().parents[1] / "conf" / "libero_xr0_eval.yaml"
    result = [
        "--conf", str(config), "--libero-suite", suite, "--task-id", task,
        "--vla-addr", f"tcp://{args.host}:{args.port}",
        "--tokenizer", args.tokenizer, "--n-episodes", str(args.n_episodes),
        "--n-action-steps", str(args.replan_steps),
        "--num-steps-wait", str(args.num_steps_wait), "--seed", str(args.seed),
        "--output-dir", args.output_dir,
        "--profile-output", args.profile_output or str(Path(args.output_dir) / "profile.json"),
    ]
    return result


def main(argv=None):
    forwarded = shared_arguments(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from client.run_sim_client_direct import main as run
    run(forwarded)


if __name__ == "__main__":
    main()
