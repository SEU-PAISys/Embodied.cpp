"""Official-PyTorch baseline rollout for Xiaomi-Robotics-0 (XR0) on LIBERO.

Mirrors scripts/rollout_xvla_reference.py: same vendored LiberoEnv, seeds, init
states, per-suite step caps, queued-chunk replay and relative ee control, so
results pair directly with the C++ server numbers under the shared runner
(`run_sim_client_direct.py --implementation python --arch xr0`).

Noise protocol matches VlaCppClient's seeded XR0 path: reset-time episode
seed, then observation-hash fallback, with matching RNG device and dtype. Arbitrary
explicit observation noise is rejected because this upstream accepts only a seed. XR0's
action head draws its flow start from `torch.manual_seed(seed)` internally
(CUDA Philox), so the derived-seed protocol pairs the Python `seed=` channel
with the C++ explicit CUDA-generated noise of the same seed. A constructor
noise_seed never pins the RNG on its own.

Per-request noise metadata (mode/seed/checksum) is recorded in the inference
profile and per-episode runner records, same as the X-VLA reference client.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(EVAL))

import numpy as np
import torch

import sim.libero  # noqa: F401  side-effect: registers gymnasium envs
from client.reproducibility import generate_xr0_noise, noise_checksum
from client.vla_cpp_client import _xr0_hash_seed, _xr0_prompt

LIBERO_SUITE_ALIASES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "10": "libero_10",
}


class XR0ReferenceClient:
    """Official PyTorch XR0 behind the same queued-client API as vla-server."""

    def __init__(self, model_path: Path, *, vision_dtype: str = "f16",
                 n_action_steps: int = 10, noise_seed: int | None = None,
                 num_steps: int = 5, policy_precision: str = "bf16"):
        from transformers import AutoModel, AutoProcessor

        if vision_dtype not in ("bf16", "f16"):
            raise ValueError(f"unsupported XR0 vision dtype: {vision_dtype}")
        if policy_precision not in ("bf16", "f32"):
            raise ValueError(f"unsupported XR0 policy precision: {policy_precision}")
        if not 1 <= n_action_steps <= 30:
            raise ValueError("XR0 replay steps must be in [1, 30]")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy_dtype = (torch.float32 if policy_precision == "f32"
                              else torch.bfloat16)
        # F32-policy mirrors the C++ VLA_XR0_F32_WEIGHTS switch: policy-side
        # matmuls (LLM + DiT) in F32 while the vision tower keeps its own
        # dtype, matching the separately-stored F16 mmproj on the C++ side.
        if policy_precision == "f32":
            # Qwen3VL sub-configs carry bfloat16 entries that survive the
            # load-time cast, and the DiT reads its runtime dtype from the
            # config. C++ VLA_XR0_F32_WEIGHTS means every non-vision matmul
            # runs F32, so load F32, force the whole module tree to F32, and
            # pin the vision tower to the deployed F16 mmproj dtype.
            self.model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.float32,
            ).to(self.device).eval().to(dtype=torch.float32)
            # Upstream hardcodes runtime-cast attributes (e.g. TimestepEmbedder
            # dtype=torch.bfloat16 default) that .to() cannot touch; align them
            # with the F32 policy the way the C++ matmul-type switch does.
            for mod in self.model.modules():
                if getattr(mod, "dtype", None) == torch.bfloat16:
                    mod.dtype = torch.float32
        else:
            self.model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=self._policy_dtype,
            ).to(self.device).eval()
        self.model.vlm.visual.to(dtype=torch.float16 if vision_dtype == "f16" else torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        action_config = self.processor.action_config["libero_all"]
        for value in action_config.values():
            if not torch.equal(value, value[:, :1].expand_as(value)):
                raise ValueError("time-dependent XR0 normalization cannot extend to 30 actions")
        self._norm_mean = action_config["mean"][:, :1].to(self.device)
        self._norm_std = action_config["std"][:, :1].to(self.device)
        self._action_mask = (self._norm_std > 1e-5).to(self._policy_dtype).expand(1, 30, 32).contiguous()
        self.num_steps = num_steps
        self.n_action_steps = n_action_steps
        self._initial_noise_seed = noise_seed
        self._queue: deque[np.ndarray] = deque(maxlen=n_action_steps)
        self._last_inference_profile = None
        self._inference_sequence = 0
        self.reset()

    def get_arch(self):
        return "xr0"

    def reset(self, noise_seed: int | None = None):
        self._queue.clear()
        self._last_inference_profile = None
        self._episode_noise_seed = noise_seed

    def has_queued_action(self):
        return bool(self._queue)

    def get_action_from_queue(self):
        if not self._queue:
            raise RuntimeError("action queue is empty; call get_action(obs) first")
        return self._queue.popleft()

    def get_last_inference_profile(self):
        return dict(self._last_inference_profile) if self._last_inference_profile else None

    def get_action(self, observations):
        if not self._queue:
            chunk = self._predict_chunk(observations)
            for row in chunk[: self.n_action_steps, : 7]:
                self._queue.append(np.ascontiguousarray(row, dtype=np.float32))
        return self._queue.popleft()

    def _resolve_noise(self, observations, state, language_raw):
        """Return (seed, mode, checksum) for the model's internal flow start.

        XR0's head seeds torch's CUDA RNG and draws randn_like(action_mask)
        itself, so the client only decides WHICH seed is used and records the
        checksum of the equivalent explicit draw.
        """
        explicit_noise = observations.get("action_noise")
        if explicit_noise is not None:
            # XR0's head only accepts a seed; the flow start is drawn inside
            # the model via randn_like, so an arbitrary explicit noise cannot
            # be injected. Reject it loudly instead of silently ignoring it.
            raise ValueError(
                "XR0ReferenceClient cannot inject arbitrary action_noise: the "
                "XR0 head seeds its internal randn_like with --noise-seed. "
                "Use --noise-seed with --derive-episode-noise in the public runner.")
        if self._episode_noise_seed is not None:
            seed = int(self._episode_noise_seed)
            mode = "derived"
        else:
            images_u8 = getattr(self, "_last_images_u8", None)
            seed = _xr0_hash_seed("libero_all", state, images_u8 or [], language_raw)
            mode = "observation_hash"
        probe = generate_xr0_noise(seed, device=self.device, dtype=self._action_mask.dtype)
        return seed, mode, noise_checksum(probe)

    def _predict_chunk(self, observations):
        images_u8 = []
        for key in ("observation.images.image", "observation.images.image2"):
            img = observations[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            img = np.asarray(img)
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"{key}: expected CHW [3,H,W], got {img.shape}")
            h, w = img.shape[1], img.shape[2]
            if h % 32 != 0 or w % 32 != 0 or h != w:
                raise ValueError(f"{key}: XR0 needs square sides divisible by 32, got {w}x{h}")
            if np.issubdtype(img.dtype, np.floating):
                img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            images_u8.append(np.ascontiguousarray(np.transpose(img, (1, 2, 0))))
        self._last_images_u8 = images_u8

        state = np.asarray(observations.get("observation.state", np.zeros(0, np.float32)),
                           dtype=np.float32).reshape(-1)
        if state.size > 32:
            raise ValueError(f"state has {state.size} dims, exceeds 32")
        state_padded = np.zeros(32, dtype=np.float32)
        state_padded[: state.shape[0]] = state

        task = observations.get("task", "")
        if isinstance(task, bytes):
            task = task.decode()
        language_raw = str(task).capitalize()
        pads = 1  # HF processor expands the placeholder itself (the C++ mirror needs 64)
        prompt = _xr0_prompt(language_raw + ".", len(images_u8), pads)
        inputs = self.processor(text=[prompt], images=images_u8, videos=None,
                                padding=True, return_tensors="pt").to(self.device)

        seed, noise_mode, cksum = self._resolve_noise(observations, state, language_raw)

        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(**dict(inputs),
                                state=torch.from_numpy(state_padded).to(self.device, self._policy_dtype)[None, None],
                                action_mask=self._action_mask,
                                num_steps=self.num_steps,
                                seed=seed)
            actions = (output.actions.float() * self._norm_std + self._norm_mean)[0].cpu().numpy().copy()
        if actions.shape != (30, 32) or not np.isfinite(actions).all():
            raise RuntimeError(f"invalid XR0 output: shape={actions.shape}")
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._inference_sequence += 1
        self._last_inference_profile = {
            "sequence": self._inference_sequence,
            "server_total_ms": elapsed_ms, "server_inference_ms": elapsed_ms,
            "server_vision_ms": None, "server_prefill_ms": None,
            "server_denoise_ms": None, "model_chunk_size": 30,
            "model_action_dim": 32, "replay_chunk_size": self.n_action_steps,
            "noise_mode": noise_mode, "noise_seed": seed,
            "noise_checksum": cksum,
            "noise_device": self.device.type,
            "noise_dtype": "f32" if self._action_mask.dtype == torch.float32 else "bf16",
        }
        return actions


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", required=True, type=Path)
    parser.add_argument("--vision-dtype", choices=("bf16", "f16"), default="f16")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--libero-suite", default="spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    options = parser.parse_args(argv)
    if options.noise_seed is not None and options.noise_seed < 0:
        parser.error("--noise-seed must be non-negative")

    from client.run_sim_client_direct import LIBEROSimAdapter, parse_args, run_one_task

    client = XR0ReferenceClient(
        options.hf_dir, vision_dtype=options.vision_dtype,
        n_action_steps=10, noise_seed=options.noise_seed,
        num_steps=options.num_steps)
    runner_args = parse_args([
        "--arch", "xr0", "--libero-suite", options.libero_suite,
        "--task-ids", *map(str, options.task_ids), "--n-episodes", str(options.n_episodes),
        "--seed", str(options.seed), "--n-action-steps", "10", "--image-size", "256",
        "--observation-width", "256", "--observation-height", "256",
        "--output-dir", str(options.output_dir), "--no-video", "--control-mode", "relative",
    ] + (["--noise-seed", str(options.noise_seed), "--derive-episode-noise"]
         if options.noise_seed is not None else []))
    options.output_dir.mkdir(parents=True, exist_ok=True)
    # The simulator returns raw observations; wrap the client in the same
    # adapter the public --implementation python path uses, so image/state
    # conversion and action parsing are identical.
    adapter = LIBEROSimAdapter(client)
    for task_id in options.task_ids:
        run_one_task(runner_args, adapter, runner_args.task, task_id, implementation="pytorch")


if __name__ == "__main__":
    main()
