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

"""LIBERO evaluation for Xiaomi-Robotics-0 through the Embodied.cpp server.

Mirrors the official XiaomiRobotics/Xiaomi-Robotics-0 eval_libero/main.py
observation pipeline (image flip, axis-angle state, 24 zero padding,
capitalised instruction + '.', replan every 10 steps) but sends requests to
the vla-server over the Embodied.cpp ZMQ protocol.

  # terminal 1 (WSL)
  ./build/vla-server ./checkpoints/xr0/xr0-mmproj.gguf \
                         ./checkpoints/xr0/xr0.gguf --bind tcp://*:5555

  # terminal 2 (LIBERO env)
  python eval/client/run_libero_eval_xr0.py \
      --tokenizer checkpoints/xr0/hf --task libero_object/task_0 \
      --n-episodes 10
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval" / "client"))

from vla_cpp_client import VlaCppClient  # noqa: E402

import sim.libero  # noqa: F401,E402  side-effect: registers gymnasium envs

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--tokenizer", required=True,
                    help="Xiaomi-Robotics-0 HF dir (tokenizer.json + vocab etc.)")
    ap.add_argument("--task", default="libero_object/task_0",
                    help="e.g. libero_object/task_3, libero_10/task_0")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output-dir", default="outputs_xr0")
    args = ap.parse_args()

    import gymnasium as gym

    client = VlaCppClient(
        f"tcp://{args.host}:{args.port}",
        arch="xr0",
        tokenizer_name=args.tokenizer,
        real_action_dim=7,
        n_action_steps=args.replan_steps,
        image_keys=("__base__", "__wrist__"),  # filled manually below
    )

    suite = args.task.split("/")[0]
    max_steps = MAX_STEPS[suite]

    successes = 0
    for episode in range(args.n_episodes):
        env = gym.make(args.task, seed=args.seed)
        obs, _ = env.reset()
        for _ in range(args.num_steps_wait):
            obs, _, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        plan = deque()
        done = False
        steps = 0
        t_predict = 0.0
        while not done and steps < max_steps:
            if len(plan) == 0:
                base = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                state = np.concatenate([
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                    np.zeros(24),
                ]).astype(np.float32)
                language = str(env.task_description).capitalize()

                x = {
                    "__base__": np.transpose(base, (2, 0, 1)),
                    "__wrist__": np.transpose(wrist, (2, 0, 1)),
                    "observation.state": state,
                    "task": language,
                }
                t0 = time.time()
                chunk = client._predict_chunk(x)      # full [30, 32] world units
                t_predict += time.time() - t0
                for row in chunk[: args.replan_steps, :7]:
                    plan.append(row.astype(np.float32))

            action = plan.popleft().tolist()
            obs, _, done, truncated, info = env.step(action)
            steps += 1
            if truncated:
                break

        ok = bool(info.get("is_success", 0.0))
        successes += int(ok)
        logging.info(
            "episode %d/%d: %s after %d steps (avg predict %.0f ms)",
            episode + 1, args.n_episodes, "SUCCESS" if ok else "FAILURE",
            steps, 1000 * t_predict / max(1, steps))
        env.close()

    rate = successes / args.n_episodes
    logging.info("task=%s success rate: %.1f%% (%d/%d)",
                 args.task, 100 * rate, successes, args.n_episodes)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"result_{args.task.replace('/', '_')}.txt").write_text(
        f"task: {args.task}\nepisodes: {args.n_episodes}\n"
        f"success rate: {rate:.2%}\n")


if __name__ == "__main__":
    main()
