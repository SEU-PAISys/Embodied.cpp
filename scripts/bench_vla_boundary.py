#!/usr/bin/env python3
"""Fixed raw CPU observations -> full CPU action chunk deployment benchmark.

Includes preprocessing, host/device transfers and output readback on both sides.
C++ additionally includes its public ZMQ transport; this is NOT kernel latency.
No simulator, action-queue replay, model load or fixture construction is timed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "scripts")]
from client.libero_profile import VramSampler


def summarize(samples):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("expected nonempty finite nonnegative latency samples")
    return dict(mean=float(values.mean()), std=float(values.std()),
                **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)})


def xvla_predictor(model, processor, images, state, task, noise, domain_id, seed):
    """Official generate_actions, with a verified repeatable RNG input.

    Preprocessing and all transfers stay inside predict; model/processor loading
    and RNG validation stay outside timing. No upstream code is patched.
    """
    import torch
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    generator = torch.Generator(device=device).manual_seed(seed)
    rng_state = generator.get_state()
    expected = torch.randn((1, 30, 20), generator=generator, device=device, dtype=dtype)
    if not np.array_equal(noise, expected.float().cpu().numpy()):
        raise ValueError("X-VLA fixture noise differs from official seed/device/dtype noise")
    if model.num_actions != 30 or model.action_space.dim_action != 20:
        raise ValueError("X-VLA benchmark requires 30x20 model output")

    def predict():
        raw = [np.ascontiguousarray((im.transpose(1, 2, 0).clip(0, 1) * 255).round(),
                                   dtype=np.uint8) for im in images]
        inputs = processor(images=raw, language_instruction=task)
        proprio = np.zeros((1, 20), np.float32)
        proprio[0, :state.size] = state
        inputs.update(proprio=torch.from_numpy(proprio),
                      domain_id=torch.tensor([domain_id], dtype=torch.long))
        inputs = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                  for key, value in inputs.items()}
        devices = [device] if device.type == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            if device.type == "cuda":
                torch.cuda.set_rng_state(rng_state, device)
            else:
                torch.set_rng_state(rng_state)
            return model.generate_actions(**inputs, steps=10)[0].float().cpu().numpy().copy()

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("turbovla", "xr0", "xvla"), required=True)
    parser.add_argument("--backend", choices=("cpp", "python"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--memory-requests", type=int, default=20,
                        help="Separate untimed requests for process VRAM; 0 disables memory sampling.")
    parser.add_argument("--address", default="tcp://127.0.0.1:5555")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--hf-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--bert-path", type=Path)
    parser.add_argument("--norm-gguf", type=Path)
    parser.add_argument("--checkpoint-key", default="model_state_dict")
    parser.add_argument("--domain-id", type=int, default=3)
    parser.add_argument("--xvla-precision", choices=("bf16", "f32"), default="bf16",
                        help="Python X-VLA only; C++ precision is selected by its server environment")
    parser.add_argument("--xvla-noise-seed", type=int, default=42,
                        help="Python X-VLA requires a fixture matching this seed/device/dtype")
    parser.add_argument("--xr0-policy-precision", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--turbovla-precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--xr0-vision-dtype", choices=("bf16", "f16"), default="bf16",
                        help="Python XR0 only: select vision precision independently of the BF16 policy.")
    args = parser.parse_args()
    if args.n < 1 or args.warmup < 1 or args.memory_requests < 0:
        parser.error("n and warmup must be positive; memory-requests must be nonnegative")
    if args.output.exists() or args.output.with_suffix(".actions.npy").exists():
        parser.error("output already exists; choose a fresh run")
    if args.arch in ("xr0", "xvla") and args.hf_dir is None:
        parser.error("xr0/xvla requires --hf-dir (matching tokenizer assets)")
    if args.domain_id < 0:
        parser.error("domain-id must be nonnegative")
    if args.xvla_noise_seed < 0:
        parser.error("xvla-noise-seed must be nonnegative")
    if args.backend == "python" and args.arch == "turbovla" and any(
        getattr(args, k) is None for k in ("checkpoint", "official_root", "bert_path", "norm_gguf")
    ):
        parser.error("TurboVLA Python requires checkpoint, official-root, bert-path and norm-gguf")
    if args.backend == "cpp" and (args.server_pid is None or args.server_pid < 1):
        parser.error("C++ requires the exact --server-pid for process VRAM")
    with np.load(args.fixture, allow_pickle=False) as fixture:
        images = fixture["images_chw"].copy()
        state = fixture["state"].copy()
        task = str(fixture["instruction"])
        noise = fixture["action_noise"].copy() if "action_noise" in fixture else None
    if images.shape != (2, 3, 256, 256) or not np.isfinite(images).all():
        raise ValueError("fixture requires two finite 256px CHW images")
    if args.arch != "xr0" and (not np.issubdtype(images.dtype, np.floating) or images.min() < 0 or images.max() > 1):
        raise ValueError("TurboVLA/X-VLA fixtures must use float images in [0, 1]")
    state_shapes = ((8,), (20,)) if args.arch == "xvla" else ((8,),)
    if state.shape not in state_shapes or not np.isfinite(state).all():
        raise ValueError(f"{args.arch} requires finite state with shape in {state_shapes}")
    if args.arch == "xr0" and (noise is None or noise.shape != (1, 30, 32) or not np.isfinite(noise).all()):
        raise ValueError("XR0 requires fixed 30x32 noise generated with the reference's seed/dtype/device")
    if args.arch == "xvla" and (noise is None or noise.shape != (1, 30, 20) or not np.isfinite(noise).all()):
        raise ValueError("X-VLA requires fixed finite 30x20 noise")
    import torch
    import transformers
    obs = {"observation.images.image": images[0], "observation.images.image2": images[1],
           "observation.state": state, "task": task}
    if noise is not None:
        obs["action_noise"] = noise
    obs["domain_id"] = args.domain_id
    expected_shape = {"turbovla": (12, 7), "xr0": (30, 32), "xvla": (30, 20)}[args.arch]
    cleanup = lambda: None
    if args.backend == "cpp":
        from client.vla_cpp_client import VlaCppClient
        client = VlaCppClient(args.address, arch=args.arch,
            tokenizer_name=str(args.hf_dir) if args.arch != "turbovla" else None,
            image_keys=("observation.images.image", "observation.images.image2"),
            image_size={"turbovla":256, "xr0":None, "xvla":224}[args.arch],
            max_state_dim={"turbovla":8, "xr0":32, "xvla":20}[args.arch],
            max_length={"turbovla":64, "xr0":512, "xvla":50}[args.arch],
            n_action_steps=expected_shape[0], real_action_dim=7)
        predict = lambda: client._predict_chunk(obs)
        cleanup = client.close
    elif args.arch == "turbovla":
        from rollout_turbovla_reference import TurboReferenceClient, load_reference_model, load_norm_arrays
        args.precision, args.device = args.turbovla_precision, "cuda"
        model, _ = load_reference_model(args)
        client = TurboReferenceClient(model, load_norm_arrays(args.norm_gguf), args.precision)
        predict = lambda: client._predict_chunk(obs)
    elif args.arch == "xvla":
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor
        dtype = torch.bfloat16 if args.xvla_precision == "bf16" else torch.float32
        model = AutoModel.from_pretrained(args.hf_dir, trust_remote_code=True,
            torch_dtype=dtype, attn_implementation="eager").to(device="cuda", dtype=dtype).eval()
        processor = AutoProcessor.from_pretrained(args.hf_dir, trust_remote_code=True)
        # Pin PIL preprocessing without forcing a slow BART tokenizer: some
        # valid snapshots contain tokenizer.json but no separate merges.txt.
        processor.image_processor = AutoImageProcessor.from_pretrained(args.hf_dir, use_fast=False)
        predict = xvla_predictor(model, processor, images, state, task, noise,
                                 args.domain_id, args.xvla_noise_seed)
    else:
        from rollout_xr0_reference import XR0ReferenceClient
        from client.reproducibility import generate_xr0_noise
        reference = XR0ReferenceClient(args.hf_dir, vision_dtype=args.xr0_vision_dtype,
                                       policy_precision=args.xr0_policy_precision)
        expected_noise = generate_xr0_noise(42, device=reference.device,
                                            dtype=reference._action_mask.dtype)
        if not np.array_equal(noise, expected_noise):
            raise ValueError("fixture noise differs from XR0 seed=42/device/policy dtype")
        # The official model accepts a seed, not explicit noise. Verify the
        # supplied fixture once, then use the public reference implementation.
        reference.reset(noise_seed=42)
        reference_obs = {key: value for key, value in obs.items() if key != "action_noise"}
        predict = lambda: reference._predict_chunk(reference_obs)
    sampler = VramSampler(args.server_pid if args.backend == "cpp" else os.getpid(), .02)
    samples, server_samples = [], []
    try:
        for _ in range(args.warmup):
            actions = predict()
            if actions.shape != expected_shape or not np.isfinite(actions).all():
                raise ValueError("invalid warmup output")
        for _ in range(args.n):
            started = time.perf_counter()
            actions = predict()  # Both paths return CPU arrays, synchronizing GPU work.
            samples.append((time.perf_counter() - started) * 1000)
            if actions.shape != expected_shape or not np.isfinite(actions).all():
                raise ValueError("invalid measured output")
            if args.backend == "cpp":
                server_samples.append(client.get_last_inference_profile())
        # nvidia-smi and CUDA allocation can contend inside the driver. Keep
        # the instrument out of the latency window for both implementations.
        if args.memory_requests:
            sampler.start()
            for _ in range(args.memory_requests):
                memory_actions = predict()
                if memory_actions.shape != expected_shape or not np.isfinite(memory_actions).all():
                    raise ValueError("invalid memory-phase output")
    finally:
        if sampler.ident is not None:
            sampler.stop()
        cleanup()
    sources = sorted(set(sampler.sources))
    result = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        boundary=__doc__, shape=list(expected_shape), samples_ms=samples, latency_ms=summarize(samples),
        memory_measurement="separate untimed requests after latency sampling; no nvidia-smi during timing",
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
        fixture_state_dim=int(state.size),
        python_precision=({"policy":args.xvla_precision if args.arch == "xvla" else
                           args.xr0_policy_precision if args.arch == "xr0" else args.turbovla_precision,
                           "vision":args.xr0_vision_dtype if args.arch == "xr0" else
                           args.xvla_precision if args.arch == "xvla" else args.turbovla_precision}
                          if args.backend == "python" else None),
        server_samples=server_samples, vram_samples_mib=sampler.samples_mib, vram_sources=sources,
        gpu_uuids=sorted(set(sampler.gpu_uuids)),
        sampled_process_peak_mib=max(sampler.samples_mib) if sampler.samples_mib and sources == ["process_used_memory"] else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    with args.output.with_suffix(".actions.npy").open("xb") as stream:
        np.save(stream, actions, allow_pickle=False)
    print(json.dumps({"latency_ms": result["latency_ms"], "sampled_process_peak_mib": result["sampled_process_peak_mib"]}))


if __name__ == "__main__":
    main()
