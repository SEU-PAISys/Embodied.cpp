#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import math
import os
import locale
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

if (
    os.environ.get("VLA_CPP_UTF8_REEXEC") != "1"
    and "UTF" not in locale.getpreferredencoding(False).upper()
):
    env = os.environ.copy()
    env["VLA_CPP_UTF8_REEXEC"] = "1"
    env["LANG"] = "C.UTF-8"
    env["LC_ALL"] = "C.UTF-8"
    env["PYTHONUTF8"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R, Slerp

try:
    import torch
    if hasattr(torch.backends.cuda, "preferred_linalg_library"):
        torch.backends.cuda.preferred_linalg_library("magma")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval"))

from client.vla_cpp_client import VlaCppClient


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, text=True).strip()


def _nvidia_used_mib() -> int | None:
    try:
        out = _run([
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ])
    except Exception:
        return None
    vals = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            vals.append(int(line))
    return max(vals) if vals else None


def _cleanup_after_env(*, torch_cuda: bool = True) -> None:
    gc.collect()
    if torch_cuda and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


class VramSampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = _nvidia_used_mib()
            if used is not None:
                self.samples.append((time.time(), used))
            self._stop.wait(self.interval_s)

    @property
    def max_used(self) -> int | None:
        if not self.samples:
            return None
        return max(v for _, v in self.samples)


def _wait_server_ready(proc: subprocess.Popen, log_path: Path, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
            raise RuntimeError(f"vla-hy-vla-server exited early with code {proc.returncode}\n{tail}")
        if log_path.exists():
            text = log_path.read_text(errors="replace")
            last = text[-4000:]
            if "ready." in text:
                return
        time.sleep(1)
    raise TimeoutError(f"vla-hy-vla-server did not become ready within {timeout_s}s\n{last}")


def _image_chw(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"expected HWC/CHW image, got {arr.shape}")
    if arr.shape[0] == 3:
        chw = arr.astype(np.float32)
    else:
        chw = np.moveaxis(arr, -1, 0).astype(np.float32)
    if chw.max(initial=0) > 2:
        chw /= 255.0
    return np.ascontiguousarray(np.clip(chw, 0, 1), dtype=np.float32)


def _pos_quat_to_pos_rotation_matrix(pos: np.ndarray, quat_xyzw: np.ndarray, gripper: float) -> np.ndarray:
    out = np.ones(10, dtype=pos.dtype)
    mat = R.from_quat(quat_xyzw).as_matrix()
    out[0:3] = pos
    out[3:6] = mat[0, :]
    out[6:9] = mat[1, :]
    out[9] = gripper
    return out


def _robotwin_ee_state_16(observation: dict[str, Any]) -> np.ndarray:
    return np.array([
        *observation["endpose"]["left_endpose"],
        observation["endpose"]["left_gripper"],
        *observation["endpose"]["right_endpose"],
        observation["endpose"]["right_gripper"],
    ], dtype=np.float32)


def _hy_vla_raw_state_20(eepose16_wxyz: np.ndarray) -> np.ndarray:
    e = eepose16_wxyz.copy()
    e[3:7] = eepose16_wxyz[[4, 5, 6, 3]]
    e[11:15] = eepose16_wxyz[[12, 13, 14, 11]]
    left = _pos_quat_to_pos_rotation_matrix(e[:3], e[3:7], e[7])
    right = _pos_quat_to_pos_rotation_matrix(e[8:11], e[11:15], e[15])
    return np.concatenate([left, right]).astype(np.float32)


def _rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    a1 = d6[:, :3]
    a2 = d6[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    dot = np.sum(b1 * a2, axis=1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=1)


def _relative_matrices_to_poses(relative_matrices: np.ndarray, start_pose_xyzw: np.ndarray) -> np.ndarray:
    n = relative_matrices.shape[0]
    t0 = np.eye(4, dtype=np.float32)
    t0[:3, :3] = R.from_quat(start_pose_xyzw[3:]).as_matrix()
    t0[:3, 3] = start_pose_xyzw[:3]

    delta_t = np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(n, axis=0)
    delta_t[:, :3, :3] = _rotation_6d_to_matrix(relative_matrices[:, 3:])
    delta_t[:, :3, 3] = relative_matrices[:, :3]

    ti = t0 @ delta_t
    return np.concatenate([ti[:, :3, 3], R.from_matrix(ti[:, :3, :3]).as_quat()], axis=1)


def _relative_to_dual_arm_poses(relative_output: np.ndarray, start_dual_pose_xyzw: np.ndarray) -> np.ndarray:
    left = _relative_matrices_to_poses(relative_output[:, 0:9], start_dual_pose_xyzw[0:7])
    right = _relative_matrices_to_poses(relative_output[:, 10:19], start_dual_pose_xyzw[8:15])
    return np.concatenate([left, relative_output[:, 9:10], right, relative_output[:, 19:20]], axis=1)


def _cross_normalized(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    v3 = np.cross(v1n, v2n)
    return v3 / np.linalg.norm(v3)


def _pos_rotation_matrix_to_pos_quat(pos_rm: np.ndarray) -> np.ndarray:
    out = np.ones(8, dtype=pos_rm.dtype)
    c0 = pos_rm[3:6]
    c1 = pos_rm[6:9]
    c2 = _cross_normalized(c0, c1)
    mat = np.stack((c0, c1, c2), axis=0)
    out[0:3] = pos_rm[0:3]
    out[3:7] = R.from_matrix(mat).as_quat()
    out[7] = pos_rm[9]
    return out


def _abs_to_dual_arm_poses(abs_output: np.ndarray) -> np.ndarray:
    out = np.zeros((abs_output.shape[0], 16), dtype=abs_output.dtype)
    for i in range(abs_output.shape[0]):
        left = _pos_rotation_matrix_to_pos_quat(abs_output[i, :10])
        right = _pos_rotation_matrix_to_pos_quat(abs_output[i, 10:20])
        out[i] = np.concatenate([left, right])
    return out


def _slerp_quat_xyzw_half(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    out = np.empty_like(q1_xyzw)
    dots = np.einsum("ij,ij->i", q1_xyzw, q2_xyzw)
    q2_aligned = q2_xyzw.copy()
    q2_aligned[dots < 0.0] *= -1
    for i in range(q1_xyzw.shape[0]):
        rots = R.from_quat(np.stack([q1_xyzw[i], q2_aligned[i]], axis=0))
        out[i] = Slerp([0.0, 1.0], rots)([0.5]).as_quat()[0]
    return out


def _blend_dual_arm_pose_quat(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    out = np.empty_like(p1)
    out[:, 0:3] = 0.5 * (p1[:, 0:3] + p2[:, 0:3])
    out[:, 3:7] = _slerp_quat_xyzw_half(p1[:, 3:7], p2[:, 3:7])
    out[:, 7:8] = 0.5 * (p1[:, 7:8] + p2[:, 7:8])
    out[:, 8:11] = 0.5 * (p1[:, 8:11] + p2[:, 8:11])
    out[:, 11:15] = _slerp_quat_xyzw_half(p1[:, 11:15], p2[:, 11:15])
    out[:, 15:16] = 0.5 * (p1[:, 15:16] + p2[:, 15:16])
    return out


def _decode_actions_to_robotwin_wxyz(
    chunk: np.ndarray,
    eepose16_wxyz: np.ndarray,
    action_chunk_size: int,
    blend_mode: str,
) -> np.ndarray:
    start_xyzw = eepose16_wxyz.copy()
    start_xyzw[3:7] = eepose16_wxyz[[4, 5, 6, 3]]
    start_xyzw[11:15] = eepose16_wxyz[[12, 13, 14, 11]]

    rel = _relative_to_dual_arm_poses(chunk[:action_chunk_size, :20], start_xyzw)
    has_abs = chunk.shape[0] >= 2 * action_chunk_size
    if blend_mode in ("rel_abs", "abs_only") and has_abs:
        abs_pose = _abs_to_dual_arm_poses(chunk[action_chunk_size: 2 * action_chunk_size, :20])
    else:
        abs_pose = None

    if blend_mode == "rel_abs" and abs_pose is not None:
        actions_xyzw = _blend_dual_arm_pose_quat(rel, abs_pose)
    elif blend_mode == "abs_only" and abs_pose is not None:
        actions_xyzw = abs_pose
    else:
        actions_xyzw = rel

    actions_wxyz = actions_xyzw.copy()
    actions_wxyz[:, 3:7] = actions_xyzw[:, [6, 3, 4, 5]]
    actions_wxyz[:, 11:15] = actions_xyzw[:, [14, 11, 12, 13]]
    return actions_wxyz.astype(np.float32)


def _encode_robotwin_obs(observation: dict[str, Any], state_dim: int) -> tuple[dict[str, Any], np.ndarray]:
    obs = observation["observation"]
    eepose16 = _robotwin_ee_state_16(observation)
    raw_state = _hy_vla_raw_state_20(eepose16)
    state = np.zeros(state_dim, dtype=np.float32)
    state[: min(state_dim, raw_state.size)] = raw_state[: min(state_dim, raw_state.size)]
    batch = {
        "observation.images.top_head": _image_chw(obs["head_camera"]["rgb"]),
        "observation.images.hand_left": _image_chw(obs["left_camera"]["rgb"]),
        "observation.images.hand_right": _image_chw(obs["right_camera"]["rgb"]),
        "observation.state": state,
    }
    return batch, eepose16


def _history_image_keys(history_size: int) -> list[str]:
    keys: list[str] = []
    for cam in ("top_head", "hand_left", "hand_right"):
        for k in range(history_size):
            keys.append(f"observation.images.{cam}.{k}")
    return keys


def _history_indices(step_id: int, history_size: int, interval: int) -> tuple[list[int], list[bool]]:
    idxs: list[int] = []
    valid: list[bool] = []
    for k in range(history_size):
        raw = step_id - (history_size - 1 - k) * interval
        idxs.append(max(raw, 0))
        valid.append(raw >= 0)
    idxs[-1] = step_id
    valid[-1] = True
    return idxs, valid


def _append_history(history: dict[str, list[np.ndarray]], batch: dict[str, Any]) -> None:
    history["top_head"].append(batch["observation.images.top_head"])
    history["hand_left"].append(batch["observation.images.hand_left"])
    history["hand_right"].append(batch["observation.images.hand_right"])


def _inject_history_images(
    batch: dict[str, Any],
    history: dict[str, list[np.ndarray]],
    history_size: int,
    interval: int,
) -> None:
    step_id = len(history["top_head"]) - 1
    idxs, valid = _history_indices(step_id, history_size, interval)
    for cam in ("top_head", "hand_left", "hand_right"):
        zero = np.zeros_like(history[cam][-1])
        for k, (idx, ok) in enumerate(zip(idxs, valid)):
            batch[f"observation.images.{cam}.{k}"] = history[cam][idx] if ok else zero


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _load_task_config(robotwin_root: Path, task_config: str) -> dict[str, Any]:
    with (robotwin_root / "task_config" / f"{task_config}.yml").open("r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args["eval_video_log"] = False
    args["render_freq"] = 0
    args["collect_data"] = False
    return args


def _prepare_task(robotwin_root: Path, task_name: str, task_config: str, seed: int):
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(robotwin_root / "description" / "utils"))
    os.chdir(robotwin_root)

    from envs import CONFIGS_PATH

    task_mod = importlib.import_module(f"envs.{task_name}")
    task_cls = getattr(task_mod, task_name)
    task_env = task_cls()
    args = _load_task_config(robotwin_root, task_config)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = "hy_vla_cpp"
    args["policy_name"] = "hy_vla_cpp"
    args["eval_mode"] = True

    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", "r", encoding="utf-8") as f:
        emb_types = yaml.safe_load(f)

    embodiment_type = args["embodiment"]

    def emb_file(name: str) -> str:
        path = emb_types[name]["file_path"]
        if path is None:
            raise RuntimeError(f"no embodiment file path for {name}")
        return path

    if len(embodiment_type) == 1:
        args["left_robot_file"] = emb_file(embodiment_type[0])
        args["right_robot_file"] = emb_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = emb_file(embodiment_type[0])
        args["right_robot_file"] = emb_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError(f"bad embodiment config: {embodiment_type}")

    def emb_config(robot_file: str) -> dict[str, Any]:
        with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    args["left_embodiment_config"] = emb_config(args["left_robot_file"])
    args["right_embodiment_config"] = emb_config(args["right_robot_file"])
    args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    return task_env, args


def _find_valid_seed(
    robotwin_root: Path,
    task_name: str,
    task_config: str,
    seed: int,
    max_tries: int,
    episode_id: int,
    *,
    debug: bool = False,
    clear_cache: bool = False,
):
    sys.path.insert(0, str(robotwin_root))
    os.chdir(robotwin_root)
    from envs.utils.create_actor import UnStableError

    for i in range(max_tries):
        task_env, args = _prepare_task(robotwin_root, task_name, task_config, seed + i)
        cur_seed = seed + i
        try:
            task_env.setup_demo(now_ep_num=episode_id, seed=cur_seed, is_test=True, **args)
            episode_info = task_env.play_once()
            ok = task_env.plan_success and task_env.check_success()
            if debug:
                print(
                    f"[seed-search] seed={cur_seed} ok={ok} "
                    f"plan_success={task_env.plan_success} check_success={task_env.check_success()}",
                    flush=True,
                )
            task_env.close_env(clear_cache=clear_cache)
            _cleanup_after_env()
            if ok:
                return cur_seed, episode_info, args
        except UnStableError as e:
            if debug:
                print(f"[seed-search] seed={cur_seed} unstable: {e}", flush=True)
            task_env.close_env(clear_cache=clear_cache)
            _cleanup_after_env()
        except Exception as e:
            if debug:
                print(
                    f"[seed-search] seed={cur_seed} exception={type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}",
                    flush=True,
                )
            task_env.close_env(clear_cache=clear_cache)
            _cleanup_after_env()
    raise RuntimeError(f"could not find a stable expert seed after {max_tries} tries")


def _make_instruction(task_name: str, episode_info: dict[str, Any], instruction_type: str, n: int) -> str:
    from generate_episode_instructions import generate_episode_descriptions

    episode_info_list = [episode_info["info"]]
    results = generate_episode_descriptions(task_name, episode_info_list, max(1, n))
    choices = results[0][instruction_type]
    return str(np.random.choice(choices))


def _default_episode_info(task_name: str) -> dict[str, Any] | None:
    if task_name == "place_empty_cup":
        return {"info": {"{A}": "021_cup/base0", "{B}": "019_coaster/base0"}}
    return None


def _parse_seed_list(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seed-list did not contain any seeds")
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf")
    ap.add_argument("--tokenizer", default="/home/xuling/robotic_dataset/HY-VLA")
    ap.add_argument("--robotwin-root", default=str(REPO_ROOT / "eval/sim/robotwin/RoboTwin"))
    ap.add_argument("--server-bin", default=str(REPO_ROOT / "build/vla-hy-vla-server"))
    ap.add_argument("--addr", default="tcp://127.0.0.1:5555")
    ap.add_argument("--bind", default="tcp://*:5555")
    ap.add_argument("--task-name", default="place_empty_cup")
    ap.add_argument("--task-config", default="demo_clean")
    ap.add_argument("--instruction-type", default="unseen")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 means use RoboTwin task_env.step_lim, matching script/eval_policy.py.")
    ap.add_argument("--seed", type=int, default=100000,
                    help="RoboTwin eval_policy.py maps --seed 0 to the first test seed 100000.")
    ap.add_argument("--robotwin-eval-seed", type=int, default=None,
                    help="Use RoboTwin script/eval_policy.py seed mapping: actual start seed = 100000 * (1 + value).")
    ap.add_argument("--seed-step", type=int, default=1,
                    help="Seed increment between episodes when --skip-stable-seed-search is used.")
    ap.add_argument("--seed-list", default=None,
                    help="Comma-separated exact RoboTwin seeds to evaluate, e.g. 100000,100001,100002.")
    ap.add_argument("--stable-seed-tries", type=int, default=20)
    ap.add_argument("--skip-stable-seed-search", action="store_true",
                    help="Use --seed directly instead of first finding an expert-solvable seed.")
    ap.add_argument("--skip-expert-check", action="store_true",
                    help="Do not call play_once() for expert/stability check. Use with known-good seeds.")
    ap.add_argument("--instruction", default=None,
                    help="Override generated RoboTwin language instruction.")
    ap.add_argument("--state-dim", type=int, default=32)
    ap.add_argument("--model-action-dim", type=int, default=20)
    ap.add_argument("--action-chunk-size", type=int, default=20,
                    help="HY-VLA original RoboTwin action_chunk_size before rel/abs doubling.")
    ap.add_argument("--exec-action-size", type=int, default=7,
                    help="HY-VLA original RoboTwin exc_action_size: env steps replayed from one forward.")
    ap.add_argument("--blend-mode", choices=["rel_abs", "rel_only", "abs_only"], default="rel_abs")
    ap.add_argument("--img-history-size", type=int, default=6)
    ap.add_argument("--img-history-interval", type=int, default=5)
    ap.add_argument("--no-image-history", action="store_true",
                    help="Use only the current 3 camera frames instead of 3 x history_size images.")
    ap.add_argument("--warmup-env-steps", type=int, default=7)
    ap.add_argument("--warmup-forward-calls", type=int, default=1)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--model-name", default="HY-VLA-CPP")
    ap.add_argument("--backbone", default="Hunyuan-VL / HY-VLA")
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/robotwin_hy_vla_native"))
    ap.add_argument("--server-timeout-s", type=int, default=900)
    ap.add_argument("--reuse-server", action="store_true")
    ap.add_argument("--start-server-after-env", action="store_true",
                    help="Initialize RoboTwin/curobo before loading the HY-VLA server.")
    ap.add_argument("--hy-vla-text-layers", default="32",
                    help="Resident HY-VLA VLM text layers loaded by vla-hy-vla-server; use 32 for full HY-VLA.")
    ap.add_argument("--hy-vla-vision-layers", default="all",
                    help="HyViT2 visual frontend layers loaded by vla-hy-vla-server; use all for full HY-VLA.")
    ap.add_argument("--hy-vla-vision-cpu-sideload", action=argparse.BooleanOptionalAction, default=False,
                    help="Load the HyViT2 visual frontend on CPU. Default is CUDA vision; use this only when CUDA vision does not fit.")
    ap.add_argument("--hy-vla-cuda-oom-fallback-cpu", action=argparse.BooleanOptionalAction, default=True,
                    help="Retry resident weights on CPU if CUDA cannot allocate the full HY-VLA weights.")
    ap.add_argument("--action-noise-mode", choices=["server", "zero", "torch_cuda_seed"], default="torch_cuda_seed",
                    help="HY-VLA flow initial noise. torch_cuda_seed mirrors the official Python seed-controlled CUDA torch noise.")
    ap.add_argument("--noise-chunk-size", type=int, default=0,
                    help="Noise sequence length sent to vla-hy-vla-server. 0 uses 2 * action_chunk_size for rel_abs/abs_only, else action_chunk_size.")
    ap.add_argument("--noise-action-dim", type=int, default=32,
                    help="HY-VLA max_action_dim for flow noise.")
    ap.add_argument("--dump-action-dir", default=None,
                    help="Optional directory for per-forward raw C++ chunks and decoded RoboTwin actions.")
    ap.add_argument("--clear-cache-each-episode", action=argparse.BooleanOptionalAction, default=True,
                    help="Close each RoboTwin episode with SAPIEN cache cleanup and clear Python torch CUDA cache.")
    ap.add_argument("--stop-server-between-episodes", action=argparse.BooleanOptionalAction, default=False,
                    help="Free vla-hy-vla-server GPU memory between episodes before the next RoboTwin expert seed search.")
    ap.add_argument("--debug-seed-search", action="store_true",
                    help="Print per-candidate expert seed search status and exceptions.")
    args = ap.parse_args()

    if args.robotwin_eval_seed is not None:
        args.seed = 100000 * (1 + args.robotwin_eval_seed)
    seed_list = _parse_seed_list(args.seed_list)
    if seed_list is not None:
        args.skip_stable_seed_search = True
        args.episodes = min(args.episodes, len(seed_list)) if args.episodes > 0 else len(seed_list)

    robotwin_root = Path(args.robotwin_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir / "vla_server.log"
    dump_action_dir = Path(args.dump_action_dir).resolve() if args.dump_action_dir else None
    if dump_action_dir is not None:
        dump_action_dir.mkdir(parents=True, exist_ok=True)

    baseline_vram = _nvidia_used_mib()
    server_proc: subprocess.Popen | None = None

    def start_server_once() -> None:
        nonlocal server_proc
        if args.reuse_server or server_proc is not None:
            return
        server_env = os.environ.copy()
        server_env.setdefault("GGML_CUDA_DISABLE_GRAPHS", "1")
        server_env.setdefault("VLA_HY_VLA_TEXT_LAYERS", str(args.hy_vla_text_layers))
        server_env.setdefault("VLA_HY_VLA_VISION_LAYERS", str(args.hy_vla_vision_layers))
        if args.hy_vla_vision_cpu_sideload:
            server_env.setdefault("VLA_HY_VLA_VISION_CPU_SIDELOAD", "1")
        if args.hy_vla_cuda_oom_fallback_cpu:
            server_env.setdefault("VLA_HY_VLA_CUDA_OOM_FALLBACK_CPU", "1")
        if not args.no_image_history and args.img_history_size > 1:
            server_env.setdefault("VLA_HY_VLA_VIDEO_HISTORY", str(args.img_history_size))
        with server_log.open("w") as log:
            server_proc = subprocess.Popen(
                [args.server_bin, "--bind", args.bind, "--timing-detail", "phase", args.model],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=server_env,
                preexec_fn=os.setsid,
            )
        _wait_server_ready(server_proc, server_log, args.server_timeout_s)

    if not args.start_server_after_env:
        start_server_once()

    sampler = VramSampler()
    sampler.start()
    image_keys = (
        [
            "observation.images.top_head",
            "observation.images.hand_left",
            "observation.images.hand_right",
        ]
        if args.no_image_history
        else _history_image_keys(args.img_history_size)
    )
    client: VlaCppClient | None = None

    def make_client_once() -> VlaCppClient:
        nonlocal client
        if client is None:
            client = VlaCppClient(
                vla_addr=args.addr,
                arch="hy_vla",
                tokenizer_name=args.tokenizer,
                image_size=args.image_size,
                max_state_dim=args.state_dim,
                real_action_dim=args.model_action_dim,
                image_keys=image_keys,
                max_length=64,
                recv_timeout_ms=300_000,
                n_action_steps=1,
            )
        return client

    if not args.start_server_after_env:
        make_client_once()

    def stop_server_once() -> None:
        nonlocal client, server_proc
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
            client = None
        if server_proc is not None and server_proc.poll() is None:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
                server_proc.wait(timeout=10)
        server_proc = None
        _cleanup_after_env()

    successes = 0
    step_ms: list[float] = []
    total_ms: list[float] = []
    vision_ms: list[float] = []
    action_inf_ms: list[float] = []
    episodes: list[dict[str, Any]] = []
    observed_chunk_size: int | None = None
    noise_chunk_size = args.noise_chunk_size
    if noise_chunk_size <= 0:
        noise_chunk_size = args.action_chunk_size * 2 if args.blend_mode in ("rel_abs", "abs_only") else args.action_chunk_size
    next_search_seed = args.seed

    try:
        for ep in range(args.episodes):
            if args.skip_stable_seed_search:
                stable_seed = seed_list[ep] if seed_list is not None else args.seed + ep * args.seed_step
                episode_info = None
                rw_args = None
                if args.instruction is None and not args.skip_expert_check:
                    task_env_probe, rw_args = _prepare_task(
                        robotwin_root, args.task_name, args.task_config, stable_seed)
                    try:
                        task_env_probe.setup_demo(now_ep_num=ep, seed=stable_seed, is_test=True, **rw_args)
                        episode_info = task_env_probe.play_once()
                    finally:
                        task_env_probe.close_env(clear_cache=args.clear_cache_each_episode)
                        _cleanup_after_env()
                elif args.instruction is None:
                    episode_info = _default_episode_info(args.task_name)
                    if episode_info is None:
                        raise RuntimeError(
                            f"--skip-expert-check needs --instruction or a built-in episode_info for {args.task_name}")
            else:
                stable_seed, episode_info, rw_args = _find_valid_seed(
                    robotwin_root, args.task_name, args.task_config,
                    next_search_seed, args.stable_seed_tries, ep,
                    debug=args.debug_seed_search,
                    clear_cache=args.clear_cache_each_episode)
                next_search_seed = stable_seed + 1
            task_env, rw_args = _prepare_task(robotwin_root, args.task_name, args.task_config, stable_seed)
            task_env.setup_demo(now_ep_num=ep, seed=stable_seed, is_test=True, **rw_args)
            if args.instruction is not None:
                instruction = str(args.instruction)
            else:
                instruction = _make_instruction(
                    args.task_name, episode_info, args.instruction_type, args.episodes)
            task_env.set_instruction(instruction=instruction)
            if args.start_server_after_env:
                start_server_once()
            active_client = make_client_once()
            active_client.reset()
            noise_gen = None
            if args.action_noise_mode == "torch_cuda_seed":
                if not torch.cuda.is_available():
                    raise RuntimeError("--action-noise-mode torch_cuda_seed requires CUDA torch")
                noise_gen = torch.Generator(device="cuda")
                noise_gen.manual_seed(stable_seed)
            action_cache: list[np.ndarray] = []
            image_history = {"top_head": [], "hand_left": [], "hand_right": []}
            last_rid = None
            forward_calls = 0
            steps = 0
            succ = False
            stop_reason = "step_lim"
            while task_env.take_action_cnt < task_env.step_lim:
                if args.max_steps > 0 and steps >= args.max_steps:
                    stop_reason = "max_steps"
                    break
                t0 = time.time()
                observation = task_env.get_obs()
                obs, eepose16 = _encode_robotwin_obs(observation, args.state_dim)
                _append_history(image_history, obs)
                if not args.no_image_history:
                    _inject_history_images(
                        obs, image_history,
                        args.img_history_size, args.img_history_interval)
                obs["task"] = instruction

                if not action_cache:
                    if args.action_noise_mode == "zero":
                        obs["action_noise"] = np.zeros(
                            (noise_chunk_size, args.noise_action_dim), dtype=np.float32)
                    elif args.action_noise_mode == "torch_cuda_seed":
                        assert noise_gen is not None
                        noise_t = torch.normal(
                            mean=0.0,
                            std=1.0,
                            size=(1, noise_chunk_size, args.noise_action_dim),
                            dtype=torch.float32,
                            device="cuda",
                            generator=noise_gen,
                        )
                        obs["action_noise"] = noise_t.squeeze(0).cpu().numpy()
                    chunk = active_client._predict_chunk(obs)
                    resp = getattr(active_client, "_last_response", None)
                    if resp is not None and resp.request_id != last_rid:
                        forward_calls += 1
                        if forward_calls > args.warmup_forward_calls:
                            total_ms.append(float(resp.latency_ms_total))
                            vision_ms.append(float(resp.latency_ms_vision))
                            action_inf_ms.append(float(resp.latency_ms_inference))
                        observed_chunk_size = int(resp.chunk_size)
                        last_rid = resp.request_id
                    chunk = np.asarray(chunk, dtype=np.float32).reshape(chunk.shape[0], -1)
                    decoded = _decode_actions_to_robotwin_wxyz(
                        chunk, eepose16, args.action_chunk_size, args.blend_mode)
                    if dump_action_dir is not None:
                        stem = f"ep{ep:03d}_seed{stable_seed}_fwd{forward_calls:03d}"
                        np.save(dump_action_dir / f"{stem}_chunk.npy", chunk)
                        np.save(dump_action_dir / f"{stem}_decoded_wxyz.npy", decoded)
                        meta = {
                            "episode": ep,
                            "seed": stable_seed,
                            "forward_calls": forward_calls,
                            "steps_before_forward": steps,
                            "instruction": instruction,
                            "request_id": int(resp.request_id) if resp is not None else None,
                            "chunk_shape": list(chunk.shape),
                            "decoded_shape": list(decoded.shape),
                        }
                        (dump_action_dir / f"{stem}.json").write_text(
                            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                    action_cache.extend(decoded[: args.exec_action_size])

                action_exec = action_cache.pop(0)
                task_env.take_action(action_exec, action_type="ee")
                elapsed_ms = (time.time() - t0) * 1000.0
                if steps >= args.warmup_env_steps:
                    step_ms.append(elapsed_ms)
                steps += 1
                if task_env.eval_success:
                    succ = True
                    stop_reason = "success"
                    break
            if succ:
                successes += 1
            episodes.append({
                "episode": ep,
                "seed": stable_seed,
                "instruction": instruction,
                "steps": steps,
                "step_lim": int(task_env.step_lim),
                "forward_calls": int(forward_calls),
                "stop_reason": stop_reason,
                "success": bool(succ),
            })
            task_env.close_env(clear_cache=args.clear_cache_each_episode)
            _cleanup_after_env()
            if args.stop_server_between_episodes:
                stop_server_once()
    finally:
        sampler.stop()
        stop_server_once()

    lo, hi = _wilson(successes, args.episodes)
    sr = 100.0 * successes / max(1, args.episodes)
    avg_total = float(np.mean(total_ms)) if total_ms else 0.0
    avg_vision = float(np.mean(vision_ms)) if vision_ms else 0.0
    avg_action_inf = float(np.mean(action_inf_ms)) if action_inf_ms else 0.0
    na = observed_chunk_size if observed_chunk_size is not None else args.action_chunk_size
    avg_step = avg_total / max(1, int(na))
    max_used = sampler.max_used
    vram = None
    if max_used is not None:
        vram = max_used - baseline_vram if baseline_vram is not None else max_used
        vram = max(0, vram)

    row = {
        "Model": args.model_name,
        "Backbone": args.backbone,
        "na": na,
        "SR (%)": f"{sr:.1f} [{100*lo:.1f}, {100*hi:.1f}]",
        "step (ms)": f"{avg_step:.1f}",
        "inf (ms)": f"{avg_total:.1f}",
        "vision (ms)": f"{avg_vision:.1f}",
        "action inf (ms)": f"{avg_action_inf:.1f}",
        "VRAM (MiB)": str(vram) if vram is not None else "NA",
    }
    md = out_dir / "summary.md"
    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    headers = [
        "Model", "Backbone", "na", "SR (%)", "step (ms)",
        "inf (ms)", "vision (ms)", "action inf (ms)", "VRAM (MiB)",
    ]
    md.write_text(
        "| " + " | ".join(headers) + " |\n"
        + "| " + " | ".join(["---"] * len(headers)) + " |\n"
        + "| " + " | ".join(str(row[h]) for h in headers) + " |\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerow(row)
    json_path.write_text(json.dumps({
        "summary": row,
        "episodes": episodes,
        "raw": {
            "successes": successes,
            "episodes": args.episodes,
            "step_ms": step_ms,
            "total_ms": total_ms,
            "vision_ms": vision_ms,
            "action_inf_ms": action_inf_ms,
            "warmup_env_steps": args.warmup_env_steps,
            "warmup_forward_calls": args.warmup_forward_calls,
            "server_chunk_size": observed_chunk_size,
            "action_chunk_size": args.action_chunk_size,
            "exec_action_size": args.exec_action_size,
            "blend_mode": args.blend_mode,
            "image_history_size": 1 if args.no_image_history else args.img_history_size,
            "image_history_interval": 1 if args.no_image_history else args.img_history_interval,
            "image_count_per_forward": len(image_keys),
            "baseline_vram_mib": baseline_vram,
            "max_sampled_vram_mib": max_used,
            "server_log": str(server_log),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))
    print(f"[robotwin-native] wrote {md}")


if __name__ == "__main__":
    main()
