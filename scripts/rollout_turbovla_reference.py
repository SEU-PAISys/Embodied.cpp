#!/usr/bin/env python3
"""Run the official TurboVLA model through the shared LIBERO rollout loop.

Uses the selected raw/EMA checkpoint entry explicitly; this is a model
reference under Embodied.cpp's protocol, not the upstream release-policy CLI.
Requires the post-norm DINO output semantics checked by load_reference_model.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity_turbovla_reference import load_norm_arrays, load_reference_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from adapter.sim.libero import LIBEROSimAdapter
from client.run_sim_client_direct import parse_args, resolve_task_ids, run_one_task


class TurboReferenceClient:
    """CPU CHW float01 images + 8-D state -> 12 x 7 CPU actions."""
    def __init__(self, model, norm_arrays, precision="bf16", n_action_steps=12):
        if precision not in ("bf16", "fp32") or not 1 <= n_action_steps <= 12:
            raise ValueError("TurboVLA requires bf16/fp32 and replay steps in [1,12]")
        self.model = model
        self.mean, self.std, self.action_min, self.action_max = norm_arrays
        self.device = next(model.parameters()).device
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.queue = deque()
        self.n_action_steps = n_action_steps
        self._inference_sequence = 0
        self._last_inference_profile = None

    def get_arch(self):
        return "turbovla"

    def reset(self, **_kwargs):
        self.queue.clear()
        self._inference_sequence = 0
        self._last_inference_profile = None

    def get_last_inference_profile(self):
        return dict(self._last_inference_profile) if self._last_inference_profile else None

    def has_queued_action(self):
        return bool(self.queue)

    def get_action_from_queue(self):
        return self.queue.popleft()

    def _predict_chunk(self, obs):
        images = np.stack([obs[f"observation.images.{key}"] for key in ("image", "image2")])
        if images.shape != (2, 3, 256, 256):
            raise ValueError(f"reference requires native 256px observations, got {images.shape}")
        pixels = (images - np.array([.485, .456, .406], dtype=np.float32)[None, :, None, None])
        pixels /= np.array([.229, .224, .225], dtype=np.float32)[None, :, None, None]
        state = (np.asarray(obs["observation.state"], dtype=np.float32) - self.mean) / (self.std + 1e-6)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model(
                [str(obs["task"])],
                {"dinov3": torch.from_numpy(pixels[None]).to(self.device, self.dtype)},
                torch.from_numpy(state[None]).to(self.device, self.dtype),
            )
        chunk = output[0].float().cpu().numpy().copy()
        if chunk.shape != (12, 7) or not np.isfinite(chunk).all():
            raise ValueError("reference returned an invalid action chunk")
        chunk[:, :6] = .5 * (chunk[:, :6] + 1) * (self.action_max[:6] - self.action_min[:6]) + self.action_min[:6]
        chunk[:, 6] = np.where(chunk[:, 6] >= 0, 1, -1)
        elapsed_ms = (time.perf_counter() - started) * 1000
        self._inference_sequence += 1
        self._last_inference_profile = {
            "sequence": self._inference_sequence,
            "server_total_ms": elapsed_ms, "server_inference_ms": elapsed_ms,
            "server_vision_ms": None, "server_prefill_ms": None, "server_denoise_ms": None,
            "model_chunk_size": 12, "model_action_dim": 7,
            "replay_chunk_size": self.n_action_steps,
        }
        return chunk

    def get_action(self, obs):
        if not self.queue:
            self.queue.extend(self._predict_chunk(obs)[:self.n_action_steps])
        return self.get_action_from_queue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("checkpoint", "official-root", "bert-path", "norm-gguf", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--checkpoint-key", default="model_state_dict")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--libero-suite", default="spatial")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    if options.n_episodes < 1:
        parser.error("n-episodes must be positive")
    args = parse_args([
        "--arch", "turbovla", "--libero-suite", options.libero_suite,
        "--task-ids", *map(str, options.task_ids), "--n-episodes", str(options.n_episodes),
        "--seed", str(options.seed), "--n-action-steps", "12", "--image-size", "256",
        "--observation-width", "256", "--observation-height", "256",
        "--output-dir", str(options.output_dir), "--no-video", "--control-mode", "relative",
    ])
    model, _ = load_reference_model(options)
    client = LIBEROSimAdapter(TurboReferenceClient(model, load_norm_arrays(options.norm_gguf), options.precision))
    options.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}
    manifest.update(torch=torch.__version__, transformers=transformers.__version__,
                    implementation="pytorch-official-model/shared-libero-protocol")
    (options.output_dir / "reference_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for task_id in resolve_task_ids(args):
        run_one_task(args, client, args.task, task_id, implementation="pytorch")


if __name__ == "__main__":
    main()
