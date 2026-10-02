#!/usr/bin/env python3
"""Run a matched official-Python SmolVLA LIBERO-Object baseline.

The parent mode runs one task per process so completed tasks are resumable.  A
hidden worker mode patches only the local LeRobot process to:

* select the requested LIBERO task with the installed ``gym_kwargs`` API;
* use the same init-state order and per-episode action-noise stream as the C++
  client;
* time ``SmolVLAPolicy._get_action_chunk`` with CUDA synchronization;
* record whole-device memory after every generated 50-action chunk; and
* disable video rendering during the measurement run.

No model or simulator assets are downloaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_IDS = tuple(range(10))
TIMING_DEFINITION = "CUDA-synchronized SmolVLAPolicy._get_action_chunk wall time"
VRAM_SOURCE = "torch.cuda.mem_get_info whole-device used memory after each request"
LEGACY_VRAM_SOURCE = "device_total_fallback"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _finite_samples(values: Any, label: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must contain raw samples")
    samples = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise ValueError(f"{label} contains an invalid sample")
    return samples


def distribution(values: list[float]) -> dict[str, float | int]:
    samples = np.asarray(_finite_samples(values, "distribution"), dtype=np.float64)
    return {
        "n": int(samples.size),
        "mean": float(samples.mean()),
        "std": float(samples.std()),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
        "min": float(samples.min()),
        "max": float(samples.max()),
    }


def wilson(successes: int, episodes: int) -> list[float]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    z = 1.959963984540054
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    half = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denominator
    return [100 * max(0.0, center - half), 100 * min(1.0, center + half)]


def _official_successes(value: Any) -> list[bool]:
    found: list[bool] = []
    if isinstance(value, dict):
        successes = value.get("successes")
        if isinstance(successes, list):
            return [bool(item) for item in successes]
        per_episode = value.get("per_episode")
        if isinstance(per_episode, list):
            found.extend(bool(row["success"]) for row in per_episode if "success" in row)
        else:
            for child in value.values():
                found.extend(_official_successes(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_official_successes(child))
    return found


def _normalize_final_info(info: dict[str, Any]) -> dict[str, Any]:
    """Adapt Gymnasium 0.29 final-info arrays to the Gymnasium 1.x shape."""
    final_info = info.get("final_info")
    if final_info is None or isinstance(final_info, dict):
        return info
    if not isinstance(final_info, (list, tuple, np.ndarray)):
        raise TypeError(f"unsupported final_info type: {type(final_info).__name__}")

    rows = list(final_info)
    if any(row is not None and not isinstance(row, dict) for row in rows):
        raise TypeError("legacy final_info must contain dictionaries or None")
    keys = sorted({key for row in rows if row is not None for key in row})
    normalized = dict(info)
    normalized["final_info"] = {
        key: np.asarray([None if row is None else row.get(key) for row in rows])
        for key in keys
    }
    return normalized


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _runtime_provenance(source_revision: str | None = None) -> dict[str, Any]:
    def command_output(command: list[str]) -> str | None:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            return None
        return completed.stdout.strip() or None

    packages: dict[str, str] = {}
    for name in ("torch", "gymnasium", "lerobot", "libero"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    cuda_version = None
    try:
        import torch

        cuda_version = torch.version.cuda
    except ImportError:
        pass
    git_prefix = ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT)]
    git_revision = command_output([*git_prefix, "rev-parse", "HEAD"])
    git_status = command_output([*git_prefix, "status", "--porcelain"])
    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    return {
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "python": sys.version.split()[0],
        "packages": packages,
        "torch_cuda": cuda_version,
        "gpu": gpu,
        "git_revision": source_revision or git_revision,
        "git_dirty": bool(git_status),
    }


def aggregate(
    output_dir: Path,
    *,
    task_ids: tuple[int, ...],
    episodes: int,
    generated_horizon: int,
    excluded_latency_samples: dict[tuple[int, int], str] | None = None,
    expected_protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_rows: list[dict[str, Any]] = []
    request_samples: list[float] = []
    generated_samples: list[float] = []
    vram_samples: list[float] = []
    latency_exclusions: list[dict[str, float | int | str]] = []
    unfiltered_request_samples: list[float] = []
    unfiltered_generated_samples: list[float] = []
    total_successes = 0
    excluded_latency_samples = excluded_latency_samples or {}

    for task_id in task_ids:
        task_dir = output_dir / f"task_{task_id}"
        eval_path = task_dir / "eval_info.json"
        profile_path = task_dir / "python_profile.json"
        if not eval_path.is_file() or not profile_path.is_file():
            raise ValueError(f"task {task_id} is incomplete")
        eval_info = json.loads(eval_path.read_text(encoding="utf-8"))
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        successes = _official_successes(eval_info)
        if len(successes) != episodes:
            raise ValueError(
                f"task {task_id} has {len(successes)} episodes, expected {episodes}"
            )
        if not profile.get("completed"):
            raise ValueError(f"task {task_id} profile is not marked complete")
        if profile.get("task_id") != task_id or profile.get("episodes") != episodes:
            raise ValueError(f"task {task_id} profile protocol mismatch")
        if profile.get("generated_action_horizon") != generated_horizon:
            raise ValueError(f"task {task_id} generated horizon mismatch")
        if expected_protocol is not None:
            for key, expected in expected_protocol.items():
                actual = profile.get(key)
                if key == "vram_source" and actual == LEGACY_VRAM_SOURCE:
                    actual = VRAM_SOURCE
                if actual != expected:
                    raise ValueError(
                        f"task {task_id} profile {key} mismatch: {actual!r} != {expected!r}"
                    )

        raw_request = _finite_samples(profile.get("request_latency_ms"), "request latency")
        raw_generated = _finite_samples(
            profile.get("generated_action_step_ms"), "generated action-step latency"
        )
        raw_vram = _finite_samples(profile.get("device_used_mib"), "device VRAM")
        if len(raw_request) != len(raw_generated) or len(raw_request) != len(raw_vram):
            raise ValueError(f"task {task_id} profile sample counts do not match")
        for index, (request_ms, generated_ms) in enumerate(zip(raw_request, raw_generated)):
            if not math.isclose(
                generated_ms,
                request_ms / generated_horizon,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise ValueError(f"task {task_id} generated sample {index} is inconsistent")
        warmup = int(profile.get("warmup_requests_excluded", -1))
        if warmup < 0 or len(raw_request) <= warmup:
            raise ValueError(f"task {task_id} has invalid warmup exclusion")
        for sample_index in range(warmup, len(raw_request)):
            request_ms = raw_request[sample_index]
            generated_ms = raw_generated[sample_index]
            unfiltered_request_samples.append(request_ms)
            unfiltered_generated_samples.append(generated_ms)
            exclusion_reason = excluded_latency_samples.get((task_id, sample_index))
            if exclusion_reason is not None:
                latency_exclusions.append(
                    {
                        "task_id": task_id,
                        "sample_index": sample_index,
                        "request_ms": request_ms,
                        "reason": exclusion_reason,
                    }
                )
                continue
            request_samples.append(request_ms)
            generated_samples.append(generated_ms)
        vram_samples.extend(raw_vram)

        task_successes = sum(successes)
        total_successes += task_successes
        task_rows.append(
            {
                "task_id": task_id,
                "successes": task_successes,
                "episodes": episodes,
                "success_rate_percent": 100.0 * task_successes / episodes,
            }
        )

    total_episodes = len(task_ids) * episodes
    result = {
        "schema": "smolvla_python_baseline.v1",
        "benchmark": "LIBERO-Object",
        "backend": "Python",
        "tasks": task_rows,
        "successes": total_successes,
        "episodes": total_episodes,
        "success_rate_percent": 100.0 * total_successes / total_episodes,
        "wilson_95_percent": wilson(total_successes, total_episodes),
        "request_latency_ms": distribution(request_samples),
        "generated_action_step_ms": distribution(generated_samples),
        "unfiltered_request_latency_ms": distribution(unfiltered_request_samples),
        "unfiltered_generated_action_step_ms": distribution(unfiltered_generated_samples),
        "vram_mib": distribution(vram_samples),
        "vram_source": VRAM_SOURCE,
        "generated_action_horizon": generated_horizon,
    }
    if excluded_latency_samples:
        missing = sorted(set(excluded_latency_samples) - {
            (int(item["task_id"]), int(item["sample_index"]))
            for item in latency_exclusions
        })
        if missing:
            raise ValueError(f"latency exclusions did not match samples: {missing}")
        result["latency_exclusions"] = {
            "reason": "host process suspension",
            "count": len(latency_exclusions),
            "samples": latency_exclusions,
        }
    return result


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    latency = result["generated_action_step_ms"]
    vram = result["vram_mib"]
    lines = [
        "# SmolVLA Python baseline: LIBERO-Object",
        "",
        "| Backend | Success rate | Generated action latency | Peak device VRAM |",
        "|---|---:|---:|---:|",
        (
            f"| Python | {result['success_rate_percent']:.2f}% "
            f"({result['successes']}/{result['episodes']}) | "
            f"{latency['mean']:.3f} ms/action | {vram['max']:.0f} MiB |"
        ),
        "",
        "The latency is CUDA-synchronized `_get_action_chunk` wall time divided by the "
        f"{result['generated_action_horizon']}-action model horizon. VRAM is whole-device "
        "usage from `torch.cuda.mem_get_info()` sampled after each inference request.",
    ]
    exclusions = result.get("latency_exclusions")
    if exclusions and exclusions["count"]:
        lines.extend(
            [
                "",
                f"Excluded {exclusions['count']} independently identified invalid request "
                "timing sample(s); each task/index/reason is recorded in JSON and all raw "
                "samples remain in the task profiles. Unfiltered summaries are retained too.",
            ]
        )
    lines.extend(
        [
            "",
            "| Task | Success / episodes | Success rate |",
            "|---:|---:|---:|",
        ]
    )
    for row in result["tasks"]:
        lines.append(
            f"| {row['task_id']} | {row['successes']} / {row['episodes']} | "
            f"{row['success_rate_percent']:.2f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _worker(args: argparse.Namespace) -> int:
    if not args.lerobot_args:
        raise ValueError("worker mode requires LeRobot arguments after --")
    forwarded = list(args.lerobot_args)
    if forwarded[0] == "--":
        forwarded = forwarded[1:]

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch
    from gymnasium.vector import SyncVectorEnv
    from transformers import AutoTokenizer
    from eval.client.reproducibility import derive_episode_noise_seed, generate_action_noise
    from lerobot.envs import configs as env_configs
    from lerobot.envs import libero as libero_env
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.scripts import lerobot_eval

    original_tokenizer_from_pretrained = AutoTokenizer.from_pretrained

    def local_tokenizer_from_pretrained(name_or_path: Any, *positional: Any, **keyword: Any):
        if str(name_or_path).startswith("HuggingFaceTB/SmolVLM2-500M"):
            name_or_path = args.base_vlm
        return original_tokenizer_from_pretrained(name_or_path, *positional, **keyword)

    AutoTokenizer.from_pretrained = staticmethod(local_tokenizer_from_pretrained)

    original_gym_kwargs = env_configs.LiberoEnv.gym_kwargs.fget
    assert original_gym_kwargs is not None

    def profiled_gym_kwargs(config: Any) -> dict[str, Any]:
        value = dict(original_gym_kwargs(config))
        value["task_ids"] = [args.task_id]
        return value

    env_configs.LiberoEnv.gym_kwargs = property(profiled_gym_kwargs)

    original_env_reset = libero_env.LiberoEnv.reset

    def matched_env_reset(self: Any, seed: int | None = None, **kwargs: Any):
        if seed is not None and self._init_states is not None:
            self._init_state_id = (int(seed) - args.seed) % len(self._init_states)
        return original_env_reset(self, seed=seed, **kwargs)

    libero_env.LiberoEnv.reset = matched_env_reset

    original_vector_step = SyncVectorEnv.step

    def compatible_vector_step(self: Any, actions: Any):
        observation, reward, terminated, truncated, info = original_vector_step(self, actions)
        return observation, reward, terminated, truncated, _normalize_final_info(info)

    SyncVectorEnv.step = compatible_vector_step

    original_policy_reset = SmolVLAPolicy.reset
    original_get_action_chunk = SmolVLAPolicy._get_action_chunk
    state: dict[str, Any] = {
        "constructed": False,
        "next_episode": 0,
        "episode": None,
        "rng": None,
    }
    request_latency_ms: list[float] = []
    generated_action_step_ms: list[float] = []
    device_used_mib: list[float] = []
    noise_checksums: list[str] = []

    def matched_policy_reset(self: Any) -> None:
        original_policy_reset(self)
        if not state["constructed"]:
            state["constructed"] = True
            return
        episode = int(state["next_episode"])
        state["next_episode"] = episode + 1
        state["episode"] = episode
        noise_seed = derive_episode_noise_seed(
            args.noise_seed, args.suite, args.task_id, episode
        )
        state["rng"] = np.random.default_rng(noise_seed)

    def profiled_get_action_chunk(
        self: Any,
        batch: dict[str, Any],
        noise: Any = None,
        **kwargs: Any,
    ):
        if noise is None:
            if state["rng"] is None:
                raise RuntimeError("matched action-noise RNG was not initialized")
            sample = generate_action_noise(
                state["rng"], args.generated_horizon, args.max_action_dim
            )
            noise_checksums.append(hashlib.sha256(sample.tobytes()).hexdigest()[:16])
            tensor_value = next(
                (value for value in batch.values() if isinstance(value, torch.Tensor)),
                None,
            )
            if tensor_value is None:
                raise RuntimeError("policy batch does not contain a tensor value")
            batch_size = int(tensor_value.shape[0])
            if batch_size != 1:
                raise RuntimeError("matched baseline requires batch size 1")
            noise = torch.from_numpy(sample).unsqueeze(0).to(
                device=next(self.parameters()).device,
                dtype=torch.float32,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        actions = original_get_action_chunk(self, batch, noise=noise, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        request_latency_ms.append(elapsed_ms)
        generated_action_step_ms.append(elapsed_ms / args.generated_horizon)
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            device_used_mib.append((total_bytes - free_bytes) / (1024 * 1024))
        else:
            device_used_mib.append(0.0)
        return actions

    SmolVLAPolicy.reset = matched_policy_reset
    SmolVLAPolicy._get_action_chunk = profiled_get_action_chunk

    original_eval_policy_all = lerobot_eval.eval_policy_all

    def no_video_eval_policy_all(*positional: Any, **keyword: Any):
        keyword["max_episodes_rendered"] = 0
        return original_eval_policy_all(*positional, **keyword)

    lerobot_eval.eval_policy_all = no_video_eval_policy_all

    completed = False
    error: str | None = None
    try:
        sys.argv = ["lerobot-eval", *forwarded]
        lerobot_eval.main()
        completed = True
        return 0
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write_json(
            args.profile_output,
            {
                "schema": "smolvla_python_task_profile.v1",
                "completed": completed,
                "error": error,
                "suite": args.suite,
                "task_id": args.task_id,
                "episodes": args.episodes,
                "seed": args.seed,
                "noise_seed": args.noise_seed,
                "generated_action_horizon": args.generated_horizon,
                "max_action_dim": args.max_action_dim,
                "warmup_requests_excluded": args.warmup_requests,
                "timing_definition": TIMING_DEFINITION,
                "request_latency_ms": request_latency_ms,
                "generated_action_step_ms": generated_action_step_ms,
                "device_used_mib": device_used_mib,
                "vram_source": VRAM_SOURCE,
                "policy_tree_sha256": args.policy_tree_sha256,
                "base_vlm_tree_sha256": args.base_vlm_tree_sha256,
                "noise_checksums": noise_checksums,
            },
        )


def _expected_profile_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "suite": args.suite,
        "seed": args.seed,
        "noise_seed": args.noise_seed,
        "warmup_requests_excluded": args.warmup_requests,
        "generated_action_horizon": args.generated_horizon,
        "max_action_dim": args.max_action_dim,
        "timing_definition": TIMING_DEFINITION,
        "vram_source": VRAM_SOURCE,
    }


def _task_is_complete(
    task_dir: Path,
    task_id: int,
    episodes: int,
    expected_protocol: dict[str, Any],
) -> bool:
    eval_path = task_dir / "eval_info.json"
    profile_path = task_dir / "python_profile.json"
    if not eval_path.is_file() or not profile_path.is_file():
        return False
    try:
        successes = _official_successes(json.loads(eval_path.read_text(encoding="utf-8")))
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not (
        len(successes) == episodes
        and profile.get("completed") is True
        and profile.get("task_id") == task_id
        and profile.get("episodes") == episodes
    ):
        return False
    for key, expected in expected_protocol.items():
        actual = profile.get(key)
        if key == "vram_source" and actual == LEGACY_VRAM_SOURCE:
            actual = VRAM_SOURCE
        if actual != expected:
            return False
    return True


def _parse_latency_exclusions(values: list[str]) -> dict[tuple[int, int], str]:
    exclusions: dict[tuple[int, int], str] = {}
    for value in values:
        parts = value.split(":", 2)
        if len(parts) != 3 or not parts[2].strip():
            raise ValueError(
                "latency exclusions must use TASK_ID:SAMPLE_INDEX:REASON"
            )
        key = (int(parts[0]), int(parts[1]))
        if key in exclusions:
            raise ValueError(f"duplicate latency exclusion: {key}")
        exclusions[key] = parts[2].strip()
    return exclusions


def _parent(args: argparse.Namespace) -> int:
    if args.suite != "libero_object":
        raise ValueError("this baseline is intentionally limited to libero_object")
    if not args.policy.is_dir():
        raise FileNotFoundError(f"policy directory not found: {args.policy}")
    if not args.base_vlm.is_dir():
        raise FileNotFoundError(f"local base VLM config/processor not found: {args.base_vlm}")
    if not (args.libero_root / "libero").is_dir():
        raise FileNotFoundError(f"LIBERO package not found: {args.libero_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_hash = _tree_hash(args.policy)
    base_vlm_hash = _tree_hash(args.base_vlm)
    expected_protocol = _expected_profile_protocol(args)
    prior_result_path = args.output_dir / "baseline_result.json"
    legacy_identity_verified = False
    if prior_result_path.is_file():
        prior_result = json.loads(prior_result_path.read_text(encoding="utf-8"))
        prior_policy = prior_result.get("policy", {})
        legacy_identity_verified = (
            prior_policy.get("tree_sha256") == policy_hash
            and prior_policy.get("base_vlm_config_tree_sha256") == base_vlm_hash
        )
    if not legacy_identity_verified:
        expected_protocol.update(
            {
                "policy_tree_sha256": policy_hash,
                "base_vlm_tree_sha256": base_vlm_hash,
            }
        )

    environment = os.environ.copy()
    prior_pythonpath = environment.get("PYTHONPATH")
    environment.update(
        {
            "PYTHONPATH": str(args.libero_root)
            + (os.pathsep + prior_pythonpath if prior_pythonpath else ""),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    for task_id in args.task_ids:
        task_dir = args.output_dir / f"task_{task_id}"
        if _task_is_complete(task_dir, task_id, args.episodes, expected_protocol):
            print(f"skip complete Python baseline task {task_id}", flush=True)
            continue
        if task_dir.exists() and any(task_dir.iterdir()):
            raise RuntimeError(
                f"incomplete task output already exists: {task_dir}; move it aside before rerunning"
            )
        task_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--profile-output",
            str(task_dir / "python_profile.json"),
            "--suite",
            args.suite,
            "--task-id",
            str(task_id),
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--noise-seed",
            str(args.noise_seed),
            "--warmup-requests",
            str(args.warmup_requests),
            "--generated-horizon",
            str(args.generated_horizon),
            "--max-action-dim",
            str(args.max_action_dim),
            "--base-vlm",
            str(args.base_vlm),
            "--policy-tree-sha256",
            policy_hash,
            "--base-vlm-tree-sha256",
            base_vlm_hash,
            "--",
            f"--output_dir={task_dir}",
            "--env.type=libero",
            f"--env.task={args.suite}",
            "--env.max_parallel_tasks=1",
            "--env.observation_width=360",
            "--env.observation_height=360",
            "--env.control_mode=relative",
            "--eval.batch_size=1",
            f"--eval.n_episodes={args.episodes}",
            f"--policy.path={args.policy}",
            f"--policy.vlm_model_name={args.base_vlm}",
            "--policy.load_vlm_weights=false",
            "--policy.n_action_steps=1",
            "--policy.num_steps=10",
            f"--seed={args.seed}",
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, env=environment)

    result = aggregate(
        args.output_dir,
        task_ids=tuple(args.task_ids),
        episodes=args.episodes,
        generated_horizon=args.generated_horizon,
        excluded_latency_samples=_parse_latency_exclusions(args.exclude_latency_sample),
        expected_protocol=expected_protocol,
    )
    result["protocol"] = {
        "suite": args.suite,
        "task_ids": list(args.task_ids),
        "episodes_per_task": args.episodes,
        "seed": args.seed,
        "noise_seed": args.noise_seed,
        "derive_episode_noise": True,
        "observation_resolution": [360, 360],
        "model_image_resolution": [512, 512],
        "relative_control": True,
        "reset_settling_steps": 10,
        "reset_settling_gripper_action": -1.0,
        "flow_steps": 10,
        "generated_action_horizon": args.generated_horizon,
    }
    result["policy"] = {
        "path": str(args.policy.resolve()),
        "tree_sha256": policy_hash,
        "base_vlm_config_path": str(args.base_vlm.resolve()),
        "base_vlm_config_tree_sha256": base_vlm_hash,
    }
    result["provenance"] = _runtime_provenance(args.source_revision)
    _write_json(args.output_dir / "baseline_result.json", result)
    _write_markdown(args.output_dir / "baseline_result.md", result)
    print(
        f"Python baseline: {result['successes']}/{result['episodes']} "
        f"({result['success_rate_percent']:.2f}%), "
        f"{result['generated_action_step_ms']['mean']:.3f} ms/action, "
        f"{result['vram_mib']['max']:.0f} MiB peak",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--policy", type=Path, default=Path("/root/checkpoints/smolvla_libero"))
    parser.add_argument(
        "--base-vlm",
        type=Path,
        default=Path("/root/checkpoints/smolvla_processor"),
        help="local SmolVLM2 config and processor directory; no weights are downloaded",
    )
    parser.add_argument(
        "--libero-root",
        type=Path,
        default=Path("/root/Embodied.cpp/eval/sim/libero/LIBERO"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(DEFAULT_TASK_IDS))
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--noise-seed", type=int, default=1000)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--generated-horizon", type=int, default=50)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument(
        "--exclude-latency-sample",
        action="append",
        default=[],
        metavar="TASK_ID:SAMPLE_INDEX:REASON",
        help="exclude one independently identified invalid timing sample; repeat as needed",
    )
    parser.add_argument("--profile-output", type=Path)
    parser.add_argument(
        "--source-revision",
        help="explicit repository revision when the worktree gitdir is not readable in WSL",
    )
    parser.add_argument("--policy-tree-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--base-vlm-tree-sha256", help=argparse.SUPPRESS)
    parser.add_argument("lerobot_args", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.episodes <= 0 or args.generated_horizon <= 0 or args.max_action_dim <= 0:
        raise ValueError("episodes, generated horizon, and action dimension must be positive")
    if args.warmup_requests < 0:
        raise ValueError("warmup requests must be non-negative")
    if args.worker:
        if args.task_id is None or args.profile_output is None:
            raise ValueError("worker mode requires --task-id and --profile-output")
        return _worker(args)
    if args.output_dir is None:
        raise ValueError("parent mode requires --output-dir")
    if tuple(args.task_ids) != DEFAULT_TASK_IDS:
        print("warning: non-standard task selection; result is not full 10-task evidence", file=sys.stderr)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
