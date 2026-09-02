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

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(ROOT))

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

from adapter.sim.libero import LIBEROSimAdapter
try:
    from client.libero_profile import LiberoSuiteProfiler
except ModuleNotFoundError:
    LiberoSuiteProfiler: Any = None
try:
    from client.inference_timeline import timeline_path_for_video, write_inference_timeline
except ImportError:
    # Upstream ships this import without the module; degrade gracefully so the
    # LIBERO direct client keeps working without inference timelines.
    def timeline_path_for_video(video_path):
        return None

    def write_inference_timeline(*args, **kwargs):
        return None
from client.reproducibility import derive_episode_noise_seed
from client.vla_cpp_client import ARCH_PRESETS as VLA_ARCH_PRESETS
from client.vla_cpp_client import VlaCppClient

ARCH_CHOICES = ["pi05", "lingbot_va", "xr0", "turbovla", "xvla", "groot_n1", "smolvla"]
PROFILE_LABELS = {
    "groot_n1": ("GR00T N1.7", "Qwen3-VL-16L"),
    "pi05": ("pi0.5", "PaliGemma"),
    "lingbot_va": ("LingBot-VA", "LingBot-VLM"),
    "smolvla": ("SmolVLA", "SmolVLM2-500M"),
    "xr0": ("Xiaomi-Robotics-0", "Qwen3-VL-4B"),
    "turbovla": ("TurboVLA", "DINOv3 + BERT"),
    "xvla": ("X-VLA", "Florence-2 DaViT + BART"),
}
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
    if args.control_mode is None:
        args.control_mode = "absolute" if args.arch == "xvla" else "relative"
    if args.real_action_dim is None:
        args.real_action_dim = 10 if args.arch == "xvla" else 7
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
    lingbot_image_sizes = None
    if args.arch == "lingbot_va" and args.lingbot_image_sizes:
        lingbot_image_sizes = []
        for item in args.lingbot_image_sizes.split(","):
            h_s, sep, w_s = item.lower().partition("x")
            if not sep:
                raise ValueError(f"invalid --lingbot-image-sizes item {item!r}; expected HxW")
            lingbot_image_sizes.append((int(h_s), int(w_s)))
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
                noise_seed=args.noise_seed,
            )
        )
    from client.lingbot_world_client import LingBotWorldClient

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
        action_per_frame=args.lingbot_action_per_frame,
        env_type=args.lingbot_env_type,
        image_sizes=lingbot_image_sizes,
    )


def run_one_task(
    args,
    client,
    task: str,
    task_id: int,
    profiler: Any = None,
) -> dict[str, Any]:
    import gymnasium as gym
    import sim.libero  # noqa: F401  registers gymnasium envs

    output_dir = Path(args.output_dir) / args.arch / task / f"task_{task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = {
        "seed": args.seed,
        "video_fps": args.fps,
        "output_video_dir": output_dir,
        "video_view_mode": args.view_mode,
        "control_mode": args.control_mode,
        "observation_width": args.observation_width,
        "observation_height": args.observation_height,
        "num_steps_wait": args.num_steps_wait,
    }
    if args.arch == "lingbot_va":
        # Match robbyant/lingbot-va's official LIBERO client: 128px cameras,
        # five zero-action settling steps, and an 800-step rollout cap.
        lingbot_image_size = args.image_size or 128
        env_kwargs.update(
            observation_width=lingbot_image_size,
            observation_height=lingbot_image_size,
            num_steps_wait=5,
            episode_length=args.max_steps if args.max_steps > 0 else 800,
        )
    env = gym.make(f"{task}/task_{task_id}", **env_kwargs)

    success_count, inference_times = 0.0, []
    lingbot_predict_wall_ms: list[float] = []
    lingbot_predict_server_ms: list[float] = []
    lingbot_cache_wall_ms: list[float] = []
    lingbot_cache_server_ms: list[float] = []
    episode_results: list[dict[str, Any]] = []
    skipped = 0
    lingbot_noise_gen = None

    def _finalize_episode(aborted: bool, info: dict[str, Any], reward: float, steps: int) -> None:
        """Record one episode exactly once, whether it completed or aborted.

        Aborted episodes are kept in the per-episode details and in the
        profiler (skipped=True) but never contribute to the success count,
        since they never reached a terminal observation.
        """
        nonlocal success_count
        avg_t = (sum(run_times) / len(run_times)) if run_times else 0.0
        inference_times.append(avg_t)
        success = (not aborted) and bool(info.get("is_success", 0.0))
        if not aborted:
            success_count += info.get("is_success", 0.0)
        if aborted:
            print(f"- Episode aborted after {steps} steps (terminated mid-step).")
        else:
            print(f"- Episode finished after {steps} steps.")
            print(f"- Final reward: {reward:.2f}")
            print(f"- Episode Information:\n{info}")
        print(f"- Average inference time per step: {round(1000 * avg_t, 2)} ms")
        if args.arch == "lingbot_va" and args.lingbot_print_timing:
            def _avg(values: list[float]) -> float:
                return sum(values) / len(values) if values else 0.0
            print(
                "- LingBot timing summary: "
                f"predict_wall_ms_avg={_avg(lingbot_predict_wall_ms):.2f} "
                f"predict_server_ms_avg={_avg(lingbot_predict_server_ms):.2f} "
                f"cache_wall_ms_avg={_avg(lingbot_cache_wall_ms):.2f} "
                f"cache_server_ms_avg={_avg(lingbot_cache_server_ms):.2f}",
                flush=True,
            )
        if profiler is not None:
            profiler.record_episode(
                task=task,
                task_id=task_id,
                episode=episode,
                success=success,
                skipped=aborted,
                environment_steps=steps,
            )
        video_path = output_dir / f"episode_{episode:06d}.mp4"
        _timeline_out = timeline_path_for_video(video_path)
        write_inference_timeline(
            _timeline_out,
            implementation="cpp",
            task=task,
            task_id=task_id,
            episode=episode,
            n_action_steps=args.n_action_steps,
            environment_steps=steps,
            video_has_initial_frame=True,
            requests=inference_requests,
        )
        episode_results.append({
            "episode": episode,
            "noise_seed": episode_noise_seed,
            "success": success,
            "skipped": aborted,
            "environment_steps": steps,
            "average_step_ms": round(1000 * avg_t, 2),
        })

    if args.arch == "lingbot_va" and args.lingbot_noise_mode == "torch_cuda_seed":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("--lingbot-noise-mode torch_cuda_seed requires CUDA torch")
        if 16 % args.lingbot_action_per_frame != 0:
            raise RuntimeError(
                f"--lingbot-action-per-frame must divide 16 for the current LingBot checkpoint, "
                f"got {args.lingbot_action_per_frame}"
            )
        lingbot_noise_gen = torch.Generator(device="cuda")
        lingbot_noise_gen.manual_seed(args.lingbot_noise_seed)
    for episode in range(args.n_episodes):
        print(f"*** {task}/task_{task_id} Episode {episode + 1}/{args.n_episodes}")

        if args.arch == "smolvla" and args.noise_seed is not None:
            episode_noise_seed = derive_episode_noise_seed(
                args.noise_seed, task, task_id, episode
            )
            client.reset(noise_seed=episode_noise_seed)
        else:
            episode_noise_seed = None
            client.reset()
        obs, info = env.reset()
        run_times, step_id = [], 0
        inference_requests: list[dict[str, float | int]] = []
        last_inference_sequence: int | None = None
        episode_aborted = False
        done = False
        truncated = False
        reward = 0.0

        while True:
            if args.arch == "lingbot_va":
                step_before_chunk = step_id
                action_noise = None
                latent_noise = None
                if args.lingbot_noise_mode == "zero":
                    action_noise = np.zeros((16, 30), dtype=np.float32)
                    latent_h = (args.image_size or 128) // 16
                    latent_w = latent_h * len(client.image_keys)
                    latent_noise = np.zeros((1, 48, 4, latent_h, latent_w), dtype=np.float32)
                elif args.lingbot_noise_mode == "torch_cuda_seed":
                    assert lingbot_noise_gen is not None
                    latent_h = (args.image_size or 128) // 16
                    latent_w = latent_h * len(client.image_keys)
                    action_frames = 16 // args.lingbot_action_per_frame
                    latent_t = torch.randn(
                        (1, 48, 4, latent_h, latent_w),
                        device="cuda",
                        dtype=torch.bfloat16,
                        generator=lingbot_noise_gen,
                    )
                    action_t = torch.randn(
                        (1, 30, action_frames, args.lingbot_action_per_frame, 1),
                        device="cuda",
                        dtype=torch.bfloat16,
                        generator=lingbot_noise_gen,
                    )
                    latent_noise = latent_t.float().cpu().numpy()
                    action_noise = (
                        action_t[0, :, :, :, 0]
                        .permute(1, 2, 0)
                        .reshape(16, 30)
                        .float()
                        .cpu()
                        .numpy()
                    )
                t0 = time.perf_counter()
                chunk = client.predict_chunk(obs, action_noise=action_noise, latent_noise=latent_noise)
                predict_dt = time.perf_counter() - t0
                predict_server_ms = float(getattr(client._last_response, "latency_ms_inference", 0.0) or 0.0)
                lingbot_predict_wall_ms.append(1000.0 * predict_dt)
                if predict_server_ms > 0.0:
                    lingbot_predict_server_ms.append(predict_server_ms)
                chunk = chunk[:, :7]
                action_per_frame = args.lingbot_action_per_frame
                if chunk.ndim != 2 or chunk.shape[0] % action_per_frame != 0:
                    raise RuntimeError(
                        f"LingBot action chunk must be [{action_per_frame}*K,7], got {chunk.shape}"
                    )
                n_frames = chunk.shape[0] // action_per_frame
                action_cfh = np.ascontiguousarray(
                    chunk.reshape(n_frames, action_per_frame, 7).transpose(2, 0, 1),
                    dtype=np.float32,
                )
                if args.lingbot_dump_first_action and episode == 0 and step_id == 0:
                    flat = action_cfh.reshape(-1)
                    first_values = ", ".join(f"{v:.10g}" for v in flat[:12])
                    print(
                        "- LingBot first action condition: "
                        f"shape={list(action_cfh.shape)} "
                        f"checksum={float(action_cfh.astype(np.float64).sum()):.12g} "
                        f"max_abs={float(np.max(np.abs(action_cfh))):.12g} "
                        f"first_values=[{first_values}]",
                        flush=True,
                    )
                key_frames = []
                start_frame = 1 if step_id == 0 else 0
                for frame_idx in range(start_frame, n_frames):
                    for sub_idx in range(action_per_frame):
                        action = action_cfh[:, frame_idx, sub_idx]
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
                cache_server_ms = 0.0
                if not (done or truncated or episode_aborted) and key_frames and not args.lingbot_disable_cache_update:
                    t1 = time.perf_counter()
                    client.update_cache(key_frames, action_cfh, imagine=False)
                    cache_dt = time.perf_counter() - t1
                    cache_server_ms = float(getattr(client._last_response, "latency_ms_inference", 0.0) or 0.0)
                    lingbot_cache_wall_ms.append(1000.0 * cache_dt)
                    if cache_server_ms > 0.0:
                        lingbot_cache_server_ms.append(cache_server_ms)
                replayed_steps = step_id - step_before_chunk
                run_times.append((predict_dt + cache_dt) / max(1, replayed_steps))
                if args.lingbot_print_timing:
                    print(
                        "- LingBot timing: "
                        f"step_start={step_before_chunk} "
                        f"replayed_steps={replayed_steps} "
                        f"predict_wall_ms={1000.0 * predict_dt:.2f} "
                        f"predict_server_ms={predict_server_ms:.2f} "
                        f"cache_wall_ms={1000.0 * cache_dt:.2f} "
                        f"cache_server_ms={cache_server_ms:.2f}",
                        flush=True,
                    )
                if profiler is not None and replayed_steps > 0:
                    amortized_ms = 1000.0 * (predict_dt + cache_dt) / replayed_steps
                    profiler.capture_inference(client)
                    for _ in range(replayed_steps):
                        profiler.record_step(amortized_ms)
            else:
                inference_step = step_id
                t0 = time.perf_counter()
                action = client.get_action(obs)
                action_dt = time.perf_counter() - t0
                run_times.append(action_dt)
                inference_profile = client.get_last_inference_profile()
                if inference_profile is not None:
                    sequence = int(inference_profile["sequence"])
                    if sequence != last_inference_sequence:
                        inference_requests.append(
                            {
                                "step_index": inference_step,
                                "inf_ms": float(inference_profile["server_total_ms"]),
                                "n_action_steps": args.n_action_steps,
                            }
                        )
                        last_inference_sequence = sequence
                if profiler is not None:
                    profiler.capture_inference(client)
                    profiler.record_step(1000.0 * action_dt)

                try:
                    obs, reward, done, truncated, info = env.step(action)
                    step_id += 1
                    if args.max_steps > 0 and step_id >= args.max_steps:
                        truncated = True
                except ValueError as e:
                    if "terminated episode" not in str(e):
                        raise
                    print(f"- Episode aborted (env reported terminated mid-step): {e}")
                    # Set the flag and fall through to the shared finalize
                    # point at the bottom of the loop; do not break here or
                    # the episode would be recorded zero times.
                    episode_aborted = True

            # Single finalize point for both branches (LingBot and generic):
            # an aborted episode sets the flag, lands here, and is recorded
            # exactly once with skipped=True before moving to the next episode.
            if done or truncated or episode_aborted:
                _finalize_episode(episode_aborted, info, reward, step_id)
                break

        if episode_aborted:
            skipped += 1

    env.close()
    # effective is the number of episodes that produced a result (completed
    # or aborted); counted guards against division by zero in the rates. The
    # summary/rate denominator must report the effective count so an
    # all-aborted task reads 0/0 (not 0/1) and stays consistent with the
    # episodes_counted field in result.json.
    effective = args.n_episodes - skipped
    counted = max(1, effective)
    avg_inf_ms = (round(1000 * sum(inference_times) / len(inference_times), 2)
                  if inference_times else 0.0)
    result = {
        "arch": args.arch,
        "suite": task,
        "task_id": task_id,
        "episodes_requested": args.n_episodes,
        "episodes_counted": effective,
        "successes": int(success_count),
        "skipped": skipped,
        "success_rate": success_count / counted,
        "average_step_ms": avg_inf_ms,
        "seed": args.seed,
        "noise_seed": args.noise_seed,
        "n_action_steps": args.n_action_steps,
        "episodes": episode_results,
        "observation_width": args.observation_width,
        "observation_height": args.observation_height,
        "image_size": args.image_size,
        "control_mode": args.control_mode,
        "num_steps_wait": args.num_steps_wait,
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"Arch: {args.arch}\n")
        f.write(f"Task: {task}/task_{task_id}\n")
        f.write(f"n_action_steps: {args.n_action_steps}\n")
        f.write(f"Success rate: {success_count / counted:.2%}  ({int(success_count)}/{effective})\n")
        f.write(f"Skipped (terminated mid-step): {skipped}/{args.n_episodes}\n")
        f.write(f"Average inference time per step: {avg_inf_ms} ms\n")
        if args.arch == "lingbot_va" and args.lingbot_print_timing:
            def _avg(values: list[float]) -> float:
                return sum(values) / len(values) if values else 0.0
            f.write(f"LingBot predict wall ms avg: {_avg(lingbot_predict_wall_ms):.2f}\n")
            f.write(f"LingBot predict server ms avg: {_avg(lingbot_predict_server_ms):.2f}\n")
            f.write(f"LingBot cache wall ms avg: {_avg(lingbot_cache_wall_ms):.2f}\n")
            f.write(f"LingBot cache server ms avg: {_avg(lingbot_cache_server_ms):.2f}\n")

    print(f"*** {task}/task_{task_id} completed.")
    print(f"- Success rate: {success_count / counted:.2%}  ({int(success_count)}/{effective})")
    print(f"- Skipped (terminated mid-step): {skipped}/{args.n_episodes}")
    print(f"- Saved videos to: {output_dir.resolve()}")
    return result

def parse_args(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--conf", "--config",
        dest="conf",
        default=None,
        help="YAML benchmark config. Relative names are resolved under eval/conf/.",
    )
    conf_args, _ = pre_parser.parse_known_args(argv)
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
    parser.add_argument("--num-steps-wait", type=int, default=10,
        help="Settling steps performed by the environment after reset.")
    parser.add_argument("--max-steps", type=int, default=0,
        help="Stop each episode after this many env steps for smoke tests. "
             "0 means run until done/truncated.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--observation-width", type=int, default=360,
        help="Raw LIBERO camera width before model preprocessing.")
    parser.add_argument("--observation-height", type=int, default=360,
        help="Raw LIBERO camera height before model preprocessing.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--view-mode",
        choices=["single-view", "multi-view"], default="multi-view",
        help="single-view: write one camera key; multi-view: side-by-side front+wrist.",
    )
    parser.add_argument("--seed", type=int, default=42,
        help="Seed for the LIBERO env reset/init-state rollout (default: 42).")
    parser.add_argument("--noise-seed", type=int, default=None,
        help="Deterministic SmolVLA action-noise seed. Each task/episode gets a stable derived seed.")

    parser.add_argument("--observation-size", type=int, default=None,
        help="Set both raw camera dimensions; cannot be combined with explicit "
             "--observation-width/height. Model resize is controlled by --image-size.")
    parser.add_argument("--control-mode", choices=["relative", "absolute"],
        default=None,
        help="LIBERO controller mode (default: absolute for xvla, relative "
             "otherwise). xvla emits absolute ee6d targets and needs 'absolute'.")

    parser.add_argument("--arch", choices=ARCH_CHOICES, default="lingbot_va",
        help="Model/client path. Also namespaces the output dir.")
    parser.add_argument("--vla-addr", type=str, default="tcp://localhost:5555",
        help="ZMQ address of the C++ inference daemon, for example vla-server or vla-server.")
    parser.add_argument("--tokenizer", type=str, default=None,
        help="Tokenizer directory or Hugging Face snapshot path for the selected model.")
    parser.add_argument("--image-size", type=int, default=None,
        help="Override the arch preset's vision input size.")
    parser.add_argument("--max-state-dim", type=int, default=None,
        help="Override the state vector length sent to the VLA server "
             "(default: arch preset).")
    parser.add_argument("--real-action-dim", type=int, default=None,
        help="Action dims consumed from each chunk row (default: 10 for xvla "
             "absolute ee6d rows, 7 otherwise).")
    parser.add_argument("--image-keys", nargs="+",
        default=["observation.images.image", "observation.images.image2"])
    parser.add_argument("--max-length", type=int, default=None,
        help="Maximum language token count. Defaults to the selected arch preset "
             "(pi05=200, lingbot_va=512, xr0=512, turbovla=64).")
    parser.add_argument("--recv-timeout-ms", type=int, default=900_000,
        help="ZMQ receive timeout for the selected C++ inference server.")
    parser.add_argument("--lingbot-session-id", type=int, default=1,
        help="[lingbot_va] session id sent to wam-lingbot-server.")
    parser.add_argument("--lingbot-max-cache-frames", type=int, default=0,
        help="[lingbot_va] maximum observation frames sent to compute_kv_cache. "
             "0 disables truncation and matches the upstream LIBERO client cadence.")
    parser.add_argument("--lingbot-disable-cache-update", action="store_true",
        help="[lingbot_va] skip post-chunk world/cache update. Useful for validating "
             "the dense no-cache C++ parity path before enabling cached evaluation.")
    parser.add_argument("--lingbot-dump-first-action", action="store_true",
        help="[lingbot_va] print shape/checksum/first values for the first predicted action.")
    parser.add_argument("--lingbot-print-timing", action="store_true",
        help="[lingbot_va] print per-chunk predict/cache wall time and server inference latency.")
    parser.add_argument("--lingbot-action-per-frame", type=int, default=4,
        help="[lingbot_va] actions represented per model frame; LIBERO uses 4, RobotWin uses 16.")
    parser.add_argument("--lingbot-env-type", type=str, default="none",
        help="[lingbot_va] LingBot job_config env_type, e.g. none or robotwin_tshape.")
    parser.add_argument("--lingbot-image-sizes", type=str, default=None,
        help="[lingbot_va] comma-separated per-view HxW sizes, e.g. 256x320,128x160,128x160.")
    parser.add_argument("--lingbot-noise-mode", choices=["server", "zero", "torch_cuda_seed"], default="torch_cuda_seed",
        help="[lingbot_va] source for initial video/action denoise noise. "
             "torch_cuda_seed mirrors PyTorch CUDA BF16 randn for parity probes.")
    parser.add_argument("--lingbot-noise-seed", type=int, default=42,
        help="[lingbot_va] seed used when --lingbot-noise-mode=torch_cuda_seed.")
    parser.add_argument(
        "--n-action-steps", type=int, default=None,
        help="How many actions to replay from each predicted chunk before "
             "re-querying the model. Defaults to the selected arch preset "
               "(pi05=10, groot_n1=8, lingbot_va=1).",
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

    args = parser.parse_args(argv)
    cli = sys.argv[1:] if argv is None else argv
    dimensions = ("--observation-width", "--observation-height")
    explicit_dimensions = {arg.split("=", 1)[0] for arg in cli} & set(dimensions)
    if args.observation_size is not None:
        if explicit_dimensions:
            if any(arg.split("=", 1)[0] == "--observation-size" for arg in cli):
                parser.error("use --observation-size or --observation-width/height, not both")
            # Explicit CLI dimensions override the YAML square-size default.
            if "--observation-width" not in explicit_dimensions:
                args.observation_width = args.observation_size
            if "--observation-height" not in explicit_dimensions:
                args.observation_height = args.observation_size
        else:
            args.observation_width = args.observation_height = args.observation_size
    if args.observation_width <= 0 or args.observation_height <= 0:
        parser.error("--observation-width and --observation-height must be positive")
    if args.arch == "xr0" and (
        args.observation_width % 32
        or args.observation_height % 32
        or args.observation_width != args.observation_height
    ):
        parser.error("xr0 camera must be square and divisible by 32; use --observation-size 256")
    if args.n_episodes <= 0 or args.num_steps_wait < 0:
        parser.error("--n-episodes must be positive and --num-steps-wait non-negative")
    # Suite precedence: an explicitly spelled CLI --libero-suite/--task wins
    # over a YAML default; with neither on the command line, the YAML
    # libero_suite (or the --task fallback default) applies.
    cli_flags = {arg.split("=", 1)[0] for arg in cli}
    if "--libero-suite" in cli_flags:
        requested_suite = args.libero_suite
    elif "--task" in cli_flags:
        requested_suite = args.task
    else:
        requested_suite = args.libero_suite or args.task
    args.task = normalize_libero_suite(requested_suite)
    if args.task not in LIBERO_SUITE_TASK_COUNTS:
        raise ValueError(
            f"unsupported LIBERO suite '{requested_suite}'. Use one of: "
            "spatial, object, goal, 10, long, libero_spatial, libero_object, "
            "libero_goal, libero_10, libero_90."
        )
    resolve_task_ids(args)
    return args


def main(argv=None):
    args = parse_args(argv)
    task_ids = resolve_task_ids(args)

    client = build_client(args)
    profiler = None
    if args.profile_output:
        if LiberoSuiteProfiler is None:
            raise RuntimeError(
                "client.libero_profile is required when --profile-output is set"
            )
        model_default, backbone_default = PROFILE_LABELS[args.arch]
        # TurboVLA runs vision + text + fusion + action head as one fused
        # graph, so its "inference" figure is the whole graph execution, not
        # an action-head-only phase; say so where the number is published.
        inference_definition = None
        if args.arch == "turbovla":
            inference_definition = (
                "single fused graph execution (vision tower + BERT + fusion + "
                "ACT head); per-phase breakdown unavailable for this architecture"
            )
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
            **({"inference_definition": inference_definition} if inference_definition else {}),
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


if __name__ == "__main__":
    main()
