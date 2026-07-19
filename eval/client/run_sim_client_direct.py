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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(ROOT))

import gymnasium as gym

import sim.libero  # noqa: F401  side-effect: registers gymnasium envs
from adapter.sim.libero import LIBEROSimAdapter
from client.libero_profile import LiberoSuiteProfiler
from client.lingbot_world_client import LingBotWorldClient
from client.vla_cpp_client import ARCH_PRESETS as VLA_ARCH_PRESETS
from client.vla_cpp_client import VlaCppClient

ARCH_CHOICES = ["pi05", "groot_n1", "lingbot_va"]
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


def _load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise SystemExit(
            "PyYAML is required for --conf. Install it in the eval environment "
            "with `pip install pyyaml` or run without --conf."
        ) from e

    conf_path = Path(path)
    if not conf_path.exists() and not conf_path.is_absolute():
        candidate = ROOT / "conf" / path
        if candidate.exists():
            conf_path = candidate
    if not conf_path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with conf_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{conf_path} must contain a YAML mapping at top level")
    return {str(k).replace("-", "_"): v for k, v in data.items()}


def normalize_libero_suite(name: str) -> str:
    return LIBERO_SUITE_ALIASES.get(name, name)


def resolve_task_ids(args) -> list[int]:
    n_tasks = LIBERO_SUITE_TASK_COUNTS[args.task]

    task_id = args.task_id
    if isinstance(task_id, str):
        if task_id.lower() != "all":
            raise ValueError(f"task_id must be an int or 'all', got {task_id!r}")
        return list(range(n_tasks))
    if task_id is not None:
        task_ids = [task_id]
    else:
        task_ids = args.task_ids

    if task_ids is None:
        task_ids = [0]
    elif isinstance(task_ids, str):
        if task_ids.lower() != "all":
            raise ValueError(f"task_ids must be 'all' or a list of ints, got {task_ids!r}")
        task_ids = list(range(n_tasks))
    else:
        task_ids = list(task_ids)

    for tid in task_ids:
        if not isinstance(tid, int):
            raise ValueError(f"task id must be int, got {tid!r}")
        if tid < 0 or tid >= n_tasks:
            raise ValueError(
                f"task-id {tid} out of range for {args.task}; expected 0..{n_tasks - 1}."
            )
    return task_ids


def build_client(args):
    if args.arch in VLA_ARCH_PRESETS:
        preset = VLA_ARCH_PRESETS[args.arch]
        args.max_length = (
            args.max_length if args.max_length is not None else preset.get("max_length", 48)
        )
        args.n_action_steps = (
            args.n_action_steps
            if args.n_action_steps is not None
            else preset.get("n_action_steps", 1)
        )
    else:
        args.max_length = args.max_length if args.max_length is not None else 512
        args.n_action_steps = args.n_action_steps if args.n_action_steps is not None else 1

    default_lerobot_image_keys = ["observation.images.image", "observation.images.image2"]
    lingbot_image_keys = (
        ["image", "image2"]
        if list(args.image_keys) == default_lerobot_image_keys
        else args.image_keys
    )
    if args.arch in VLA_ARCH_PRESETS:
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
        max_length=args.max_length,
        recv_timeout_ms=args.recv_timeout_ms,
        n_action_steps=args.n_action_steps,
        session_id=args.lingbot_session_id,
        max_cache_frames=args.lingbot_max_cache_frames,
    )


def run_one_task(
    args,
    client,
    task: str,
    task_id: int,
    profiler: LiberoSuiteProfiler | None = None,
) -> dict[str, Any]:
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
                step_before_chunk = step_id
                t0 = time.perf_counter()
                chunk = client.predict_chunk(obs)
                predict_dt = time.perf_counter() - t0
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
                    t1 = time.perf_counter()
                    client.update_cache(key_frames, chunk, imagine=False)
                    cache_dt = time.perf_counter() - t1
                run_times.append(predict_dt + cache_dt)
                replayed_steps = step_id - step_before_chunk
                if profiler is not None and replayed_steps > 0:
                    amortized_ms = 1000.0 * (predict_dt + cache_dt) / replayed_steps
                    for _ in range(replayed_steps):
                        profiler.record_step(amortized_ms)
                    profiler.capture_inference(client)
            else:
                t0 = time.perf_counter()
                action = client.get_action(obs)
                action_dt = time.perf_counter() - t0
                run_times.append(action_dt)
                if profiler is not None:
                    profiler.record_step(1000.0 * action_dt)
                    profiler.capture_inference(client)

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
                if profiler is not None:
                    profiler.record_episode(
                        task=task,
                        task_id=task_id,
                        episode=episode,
                        success=bool(info.get("is_success", 0.0)),
                        skipped=episode_aborted,
                        environment_steps=step_id,
                    )
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
    return {
        "task": task,
        "task_id": task_id,
        "episodes": args.n_episodes,
        "successes": int(success_count),
        "skipped": skipped,
        "average_step_ms": avg_inf_ms,
    }

if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--conf", "--config",
        dest="conf",
        default=None,
        help="YAML benchmark config. Relative names are resolved under eval/conf/.",
    )
    conf_args, _ = pre_parser.parse_known_args()
    conf_defaults = _load_yaml_config(conf_args.conf)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[pre_parser],
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
    parser.add_argument("--task-id", type=int, default=None,
        help="Task variation id within the suite.")
    parser.add_argument("--task-ids", nargs="+", type=int, default=None,
        help="Task variation ids to run. YAML configs may also set task_ids: all.")
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
    parser.add_argument("--max-length", type=int, default=None,
        help="Maximum language token count. Defaults to the selected arch preset "
             "(pi05=200, lingbot_va=512).")
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
        "--n-action-steps", type=int, default=None,
        help="How many actions to replay from each predicted chunk before "
             "re-querying the model. Defaults to the selected arch preset "
             "(pi05=5, groot_n1=8, lingbot_va=1).",
    )
    parser.add_argument("--profile-output", type=str, default=None,
        help="Write full-suite table metrics to this JSON path; CSV and Markdown "
             "rows are written beside it after a complete run.")
    parser.add_argument("--profile-model-label", type=str, default=None,
        help="Model label used in the generated table row.")
    parser.add_argument("--profile-backbone-label", type=str, default=None,
        help="Backbone label used in the generated table row.")
    parser.add_argument("--profile-server-pid", type=int, default=None,
        help="PID sampled for VRAM. By default it is derived from --vla-addr.")
    parser.add_argument("--profile-vram-interval-s", type=float, default=0.25,
        help="nvidia-smi VRAM sampling interval in seconds.")
    parser.add_argument("--profile-warmup-requests", type=int, default=5,
        help="Unique server requests excluded from the mean inf latency.")

    if conf_defaults:
        valid_dests = {action.dest for action in parser._actions}
        unknown = sorted(set(conf_defaults) - valid_dests)
        if unknown:
            raise ValueError(
                f"unknown keys in {conf_args.conf}: {', '.join(unknown)}. "
                f"Valid keys are: {', '.join(sorted(valid_dests))}"
            )
        parser.set_defaults(**conf_defaults)

    args = parser.parse_args()
    requested_suite = args.libero_suite or args.task
    args.task = normalize_libero_suite(requested_suite)
    if args.task not in LIBERO_SUITE_TASK_COUNTS:
        raise ValueError(
            f"unsupported LIBERO suite '{requested_suite}'. Use one of: "
            "spatial, object, goal, 10, long, libero_spatial, libero_object, "
            "libero_goal, libero_10, libero_90."
        )
    task_ids = resolve_task_ids(args)

    client = build_client(args)
    profiler = None
    if args.profile_output:
        default_labels = {
            "groot_n1": ("GR00T N1.7", "Qwen3-VL-16L"),
            "pi05": ("pi0.5", "PaliGemma"),
            "lingbot_va": ("LingBot-VA", "LingBot-VLM"),
        }
        model_default, backbone_default = default_labels[args.arch]
        profiler = LiberoSuiteProfiler(
            output_path=Path(args.profile_output),
            model_label=args.profile_model_label or model_default,
            backbone_label=args.profile_backbone_label or backbone_default,
            arch=args.arch,
            suite=args.task,
            replay_chunk_size=args.n_action_steps,
            expected_episodes=len(task_ids) * args.n_episodes,
            server_address=args.vla_addr,
            server_pid=args.profile_server_pid,
            vram_interval_s=args.profile_vram_interval_s,
            warmup_requests=args.profile_warmup_requests,
        )
        profiler.start()

    complete = False
    try:
        for task_id in task_ids:
            run_one_task(args, client, args.task, task_id, profiler)
        complete = True
    finally:
        if profiler is not None:
            profiler.stop()
            result = profiler.write(complete=complete)
            print(f"- Profile JSON: {profiler.output_path.resolve()}")
            if complete:
                print(f"- Profile table: {profiler.output_path.with_suffix('.md').resolve()}")
                print(f"- Table ready: {result['table_ready']}")
