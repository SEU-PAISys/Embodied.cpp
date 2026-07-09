# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(ROOT))

from adapter.sim.robotwin import RobotWinHyVLAAdapter


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        return repr(v)


def _is_success(info: dict) -> float:
    for key in ("is_success", "success", "task_success", "episode_success"):
        if key in info:
            return float(bool(info[key]))
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RobotWin evaluation against a remote or local vla-hy-vla-server.",
    )
    parser.add_argument("--env-id", required=True,
        help="RobotWin gymnasium env id, for example the id registered by the local RobotWin install.")
    parser.add_argument("--register-module", action="append", default=[],
        help="Python module imported before gym.make() to register RobotWin envs. Can be repeated.")
    parser.add_argument("--robotwin-pythonpath", action="append", default=[],
        help="Path inserted into sys.path before importing --register-module.")
    parser.add_argument("--vla-addr", default="tcp://localhost:5555")
    parser.add_argument("--tokenizer", required=True,
        help="HY-VLA tokenizer/checkpoint directory used by AutoTokenizer.")
    parser.add_argument("--output-dir", default="outputs/robotwin_hy_vla")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0,
        help="0 means use the environment termination/truncation only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=20)
    parser.add_argument("--action-dim", type=int, default=20)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--recv-timeout-ms", type=int, default=120_000)
    parser.add_argument("--front-key", default=None,
        help="Dot path for the front image in RobotWin observations.")
    parser.add_argument("--wrist-key", default=None,
        help="Dot path for the wrist image in RobotWin observations. Missing wrist image is zero-filled.")
    parser.add_argument("--state-key", default=None,
        help="Dot path for the state vector in RobotWin observations.")
    parser.add_argument("--task-key", default=None,
        help="Dot path for the language instruction in RobotWin observations.")
    parser.add_argument("--task", default="",
        help="Fallback language instruction if the env observation does not include one.")
    args = parser.parse_args()

    robotwin_root = None
    for p in args.robotwin_pythonpath:
        path = Path(p).resolve()
        sys.path.insert(0, str(path))
        if robotwin_root is None and (path / "envs").is_dir() and (path / "assets").is_dir():
            robotwin_root = path
    if robotwin_root is not None:
        os.chdir(robotwin_root)
    for mod in args.register_module:
        importlib.import_module(mod)
    try:
        import gymnasium as gym
    except ModuleNotFoundError as e:
        raise SystemExit(
            "gymnasium is not installed in this Python environment. Run this "
            "script from the RobotWin environment, or set ROBOTWIN_PYTHON in "
            "eval/run_robotwin_hy_vla_client.sh."
        ) from e
    from client.vla_cpp_client import VlaCppClient

    output_dir = Path(args.output_dir) / args.env_id.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    client = VlaCppClient(
        vla_addr=args.vla_addr,
        arch="hy_vla",
        tokenizer_name=args.tokenizer,
        image_size=args.image_size,
        max_state_dim=args.state_dim,
        real_action_dim=args.action_dim,
        image_keys=["observation.images.image", "observation.images.image2"],
        max_length=args.max_length,
        recv_timeout_ms=args.recv_timeout_ms,
        n_action_steps=args.n_action_steps,
    )
    policy = RobotWinHyVLAAdapter(
        client,
        front_key=args.front_key,
        wrist_key=args.wrist_key,
        state_key=args.state_key,
        task_key=args.task_key,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        default_task=args.task,
    )

    env = gym.make(args.env_id)
    successes = 0.0
    inference_times = []
    episodes = []

    for ep in range(args.n_episodes):
        policy.reset()
        reset_out = env.reset(seed=args.seed + ep)
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False
        truncated = False
        reward = 0.0
        info = {}
        step = 0
        ep_times = []
        while not (done or truncated):
            t0 = time.time()
            action = policy.get_action(obs)
            ep_times.append(time.time() - t0)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, done, truncated, info = step_out
            elif len(step_out) == 4:
                obs, reward, done, info = step_out
                truncated = False
            else:
                raise RuntimeError(f"RobotWin env.step returned {len(step_out)} values")
            step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                truncated = True

        success = _is_success(info)
        successes += success
        avg_ms = 1000.0 * sum(ep_times) / max(1, len(ep_times))
        inference_times.append(avg_ms)
        row = {
            "episode": ep,
            "steps": step,
            "reward": float(reward),
            "success": success,
            "avg_inference_ms": avg_ms,
            "info": {k: _jsonable(v) for k, v in dict(info).items()},
        }
        episodes.append(row)
        print(f"[robotwin] ep={ep} steps={step} reward={reward:.4f} success={success} avg_inf_ms={avg_ms:.2f}")

    env.close()
    summary = {
        "env_id": args.env_id,
        "episodes": args.n_episodes,
        "success_rate": successes / max(1, args.n_episodes),
        "avg_inference_ms": sum(inference_times) / max(1, len(inference_times)),
        "episode_results": episodes,
    }
    out = output_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[robotwin] wrote {out}")


if __name__ == "__main__":
    main()
