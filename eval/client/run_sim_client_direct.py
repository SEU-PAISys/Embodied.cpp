# Copyright 2026 VinRobotics
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

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym

import sim.libero  # noqa: F401  side-effect: registers gymnasium envs
from adapter.sim.libero import LIBEROSimAdapter
from client.lingbot_world_client import LingBotWorldClient
from client.vla_cpp_client import VlaCppClient

ARCH_CHOICES = ["pi05", "lingbot_va"]
LIBERO_SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}
LIBERO_SUITE_ALIASES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "10": "libero_10",
    "libero10": "libero_10",
    "long": "libero_90",
    "90": "libero_90",
    "libero90": "libero_90",
    "libero_long": "libero_90",
    "libero-long": "libero_90",
    "LIBERO_LONG": "libero_90",
}


def normalize_libero_suite(name: str) -> str:
    return LIBERO_SUITE_ALIASES.get(name, name)


def build_client(args):
    default_lerobot_image_keys = ["observation.images.image", "observation.images.image2"]
    lingbot_image_keys = (
        ["image", "image2"]
        if list(args.image_keys) == default_lerobot_image_keys
        else args.image_keys
    )
    if args.arch == "pi05":
        return LIBEROSimAdapter(
            client=VlaCppClient(
                vla_addr=args.vla_addr,
                arch=args.arch,
                tokenizer_name=args.tokenizer,
                image_size=args.image_size,
                max_state_dim=args.max_state_dim,
                real_action_dim=args.real_action_dim,
                image_keys=args.image_keys,
                max_length=args.max_length,
                recv_timeout_ms=args.recv_timeout_ms,
                n_action_steps=args.n_action_steps,
            )
        )
    return LingBotWorldClient(
        vla_addr=args.vla_addr,
        tokenizer_name=args.tokenizer,
        image_size=args.image_size or 128,
        image_keys=lingbot_image_keys,
        max_length=args.max_length if args.max_length != 48 else 512,
        recv_timeout_ms=args.recv_timeout_ms,
        n_action_steps=args.n_action_steps,
        session_id=args.lingbot_session_id,
        max_cache_frames=args.lingbot_max_cache_frames,
    )


def run_one_task(args, client, task: str, task_id: int) -> None:
    output_dir = Path(args.output_dir) / args.arch / task / f"task_{task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        f"{task}/task_{task_id}",
        seed=args.seed,
        video_fps=args.fps,
        output_video_dir=output_dir,
        video_view_mode=args.view_mode,
    )

    success_count, inference_times = 0.0, []
    skipped = 0
    for episode in range(args.n_episodes):
        print(f"*** {task}/task_{task_id} Episode {episode + 1}/{args.n_episodes}")

        client.reset()
        obs, info = env.reset()
        run_times, step_id = [], 0
        episode_aborted = False
        done = False
        truncated = False
        reward = 0.0

        while True:
            if args.arch == "lingbot_va":
                t0 = time.time()
                chunk = client.predict_chunk(obs)
                predict_dt = time.time() - t0
                chunk = chunk[:, :7]
                if chunk.ndim != 2 or chunk.shape[0] % 4 != 0:
                    raise RuntimeError(f"LingBot action chunk must be [4*K,7], got {chunk.shape}")
                action_per_frame = 4
                n_frames = chunk.shape[0] // action_per_frame
                chunk_fh = chunk.reshape(n_frames, action_per_frame, 7)
                key_frames = []
                start_frame = 1 if step_id == 0 else 0
                for frame_idx in range(start_frame, n_frames):
                    for sub_idx in range(action_per_frame):
                        action = chunk_fh[frame_idx, sub_idx]
                        try:
                            obs, reward, done, truncated, info = env.step(action)
                        except ValueError as e:
                            if "terminated episode" not in str(e):
                                raise
                            print(f"- Episode aborted (env reported terminated mid-step): {e}")
                            episode_aborted = True
                            break
                        step_id += 1
                        if done or truncated:
                            break
                        if args.max_steps > 0 and step_id >= args.max_steps:
                            truncated = True
                            break
                    key_frames.append(obs)
                    if done or truncated or episode_aborted:
                        break
                cache_dt = 0.0
                if not (done or truncated or episode_aborted) and key_frames and not args.lingbot_disable_cache_update:
                    t1 = time.time()
                    client.update_cache(key_frames, chunk, imagine=False)
                    cache_dt = time.time() - t1
                run_times.append(predict_dt + cache_dt)
            else:
                t0 = time.time()
                action = client.get_action(obs)
                run_times.append(time.time() - t0)

                try:
                    obs, reward, done, truncated, info = env.step(action)
                except ValueError as e:
                    if "terminated episode" not in str(e):
                        raise
                    print(f"- Episode aborted (env reported terminated mid-step): {e}")
                    episode_aborted = True
                    break
                step_id += 1
                if args.max_steps > 0 and step_id >= args.max_steps:
                    truncated = True

            if done or truncated or episode_aborted:
                avg_t = sum(run_times) / len(run_times)
                inference_times.append(avg_t)
                success_count += info.get("is_success", 0.0)

                print(f"- Episode finished after {step_id} steps.")
                print(f"- Final reward: {reward:.2f}")
                print(f"- Episode Information:\n{info}")
                print(f"- Average inference time per step: {round(1000 * avg_t, 2)} ms")
                break

        if episode_aborted:
            skipped += 1

    env.close()
    counted = max(1, args.n_episodes - skipped)
    avg_inf_ms = (round(1000 * sum(inference_times) / len(inference_times), 2)
                  if inference_times else 0.0)
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Arch: {args.arch}\n")
        f.write(f"Task: {task}/task_{task_id}\n")
        f.write(f"n_action_steps: {args.n_action_steps}\n")
        f.write(f"Success rate: {success_count / counted:.2%}  ({int(success_count)}/{counted})\n")
        f.write(f"Skipped (terminated mid-step): {skipped}/{args.n_episodes}\n")
        f.write(f"Average inference time per step: {avg_inf_ms} ms\n")

    print(f"*** {task}/task_{task_id} completed.")
    print(f"- Success rate: {success_count / counted:.2%}  ({int(success_count)}/{counted})")
    print(f"- Skipped (terminated mid-step): {skipped}/{args.n_episodes}")
    print(f"- Saved videos to: {output_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", type=str, default="libero_object",
        help="LIBERO suite: one of ['libero_10', 'libero_spatial', "
             "'libero_object', 'libero_goal', 'libero_90', 'libero_long']. "
             "libero_long is an alias for libero_90.",
    )
    parser.add_argument(
        "--libero-suite", type=str, default=None,
        help="Short LIBERO sub-dataset name: spatial | object | goal | 10 | long. "
             "Overrides --task when set. Full names like libero_object are also accepted.",
    )
    parser.add_argument("--task-id", type=int, default=0,
        help="Task variation id within the suite.")
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=0,
        help="Stop each episode after this many env steps for smoke tests. "
             "0 means run until done/truncated.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--view-mode",
        choices=["single-view", "multi-view"], default="multi-view",
        help="single-view: write one camera key; multi-view: side-by-side front+wrist.",
    )
    parser.add_argument("--seed", type=int, default=42,
        help="Seed for the LIBERO env reset/init-state rollout (default: 42).")

    parser.add_argument("--arch", choices=ARCH_CHOICES, default="lingbot_va",
        help="Model/client path. Also namespaces the output dir.")
    parser.add_argument("--vla-addr", type=str, default="tcp://localhost:5555",
        help="ZMQ address of the C++ inference daemon, for example vla-pi05-server or vla-hy-vla-server.")
    parser.add_argument("--tokenizer", type=str, default=None,
        help="Tokenizer directory or Hugging Face snapshot path for the selected model.")
    parser.add_argument("--image-size", type=int, default=None,
        help="Override the arch preset's vision input size.")
    parser.add_argument("--max-state-dim", type=int, default=None,
        help="Override the state vector length sent to the VLA server "
             "(default: arch preset).")
    parser.add_argument("--real-action-dim", type=int, default=7)
    parser.add_argument("--image-keys", nargs="+",
        default=["observation.images.image", "observation.images.image2"])
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--recv-timeout-ms", type=int, default=900_000,
        help="ZMQ receive timeout for the selected C++ inference server.")
    parser.add_argument("--lingbot-session-id", type=int, default=1,
        help="[lingbot_va] session id sent to wam-lingbot-server.")
    parser.add_argument("--lingbot-max-cache-frames", type=int, default=4,
        help="[lingbot_va] maximum observation frames sent to compute_kv_cache. "
             "Default 4 matches the original-style LIBERO key-frame update window.")
    parser.add_argument("--lingbot-disable-cache-update", action="store_true",
        help="[lingbot_va] skip post-chunk world/cache update. Useful for validating "
             "the dense no-cache C++ parity path before enabling cached evaluation.")
    parser.add_argument(
        "--n-action-steps", type=int, default=1,
        help="How many actions to replay from each predicted chunk before "
             "re-querying wam-lingbot-server.",
    )

    args = parser.parse_args()
    requested_suite = args.libero_suite or args.task
    args.task = normalize_libero_suite(requested_suite)
    if args.task not in LIBERO_SUITE_TASK_COUNTS:
        raise ValueError(
            f"unsupported LIBERO suite '{requested_suite}'. Use one of: "
            "spatial, object, goal, 10, long, libero_spatial, libero_object, "
            "libero_goal, libero_10, libero_90."
        )
    if args.task_id < 0 or args.task_id >= LIBERO_SUITE_TASK_COUNTS[args.task]:
        raise ValueError(
            f"task-id {args.task_id} out of range for {args.task}; "
            f"expected 0..{LIBERO_SUITE_TASK_COUNTS[args.task] - 1}."
        )

    client = build_client(args)
    run_one_task(args, client, args.task, args.task_id)
