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
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "scripts")]
from client.libero_profile import VramSampler

BOUNDARY_ID = "raw_cpu_observation_to_complete_cpu_action_chunk.v1"


def smolvla_processor_overrides(tokenizer_path):
    return {
        "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},
        "device_processor": {"device": "cuda"},
        "rename_observations_processor": {"rename_map": {}},
    }


def load_fixture(path, arch, instruction=None):
    """Load an immutable fixed-input fixture without doing timed work."""
    with np.load(path, allow_pickle=False) as fixture:
        if arch == "smolvla":
            images = np.stack((fixture["image"], fixture["image2"])).copy()
            state = fixture["state_raw"].copy()
            noise = fixture["noise"].copy()
            if "instruction" in fixture:
                task = str(fixture["instruction"].item())
            elif instruction:
                task = instruction
            else:
                raise ValueError("SmolVLA fixture requires --instruction when it does not embed one")
        else:
            images = fixture["images_chw"].copy()
            state = fixture["state"].copy()
            task = str(fixture["instruction"].item())
            noise = fixture["action_noise"].copy() if "action_noise" in fixture else None
    return images, state, task, noise


def validate_comparable_results(left, right):
    """Reject a Python/C++ pair that cannot support normalized table values."""
    if left.get("arch") != right.get("arch"):
        raise ValueError(f"non-comparable arch: {left.get('arch')!r} != {right.get('arch')!r}")
    if {left.get("backend"), right.get("backend")} != {"python", "cpp"}:
        raise ValueError("non-comparable backend pair: expected one python and one cpp result")
    if left.get("measurement_plan") != right.get("measurement_plan"):
        raise ValueError("non-comparable measurement_plan")
    for key in ("boundary_id", "script_sha256", "fixture_sha256", "instruction_sha256",
                "source_checkpoint_sha256", "tokenizer_sha256", "shape",
                "generated_action_horizon", "vram_sources", "gpu_uuids"):
        if left.get(key) != right.get(key):
            raise ValueError(f"non-comparable {key}: {left.get(key)!r} != {right.get(key)!r}")
    if left.get("boundary_id") != BOUNDARY_ID:
        raise ValueError(f"unsupported boundary_id: {left.get('boundary_id')!r}")
    if len(left.get("vram_sources", [])) != 1:
        raise ValueError("vram_sources must contain exactly one shared measurement source")
    if len(left.get("gpu_uuids", [])) != 1:
        raise ValueError("gpu_uuids must contain exactly one shared GPU UUID")


def validate_measurement_plan(n, warmup, memory_requests, formal=False):
    if n < 1 or warmup < 1 or memory_requests < 0:
        raise ValueError("n and warmup must be positive; memory-requests must be nonnegative")
    if formal and (n < 100 or warmup < 5 or memory_requests < 20):
        raise ValueError("formal evidence requires n>=100, warmup>=5, and memory-requests>=20")


def normalized_to_python(left, right):
    by_backend = {left.get("backend"): left, right.get("backend"): right}
    if set(by_backend) != {"python", "cpp"}:
        raise ValueError("normalization requires one python and one cpp result")
    python, cpp = by_backend["python"], by_backend["cpp"]

    def ratio(path):
        py_value, cpp_value = python, cpp
        for key in path:
            py_value, cpp_value = py_value[key], cpp_value[key]
        py_value, cpp_value = float(py_value), float(cpp_value)
        if not np.isfinite([py_value, cpp_value]).all() or py_value <= 0 or cpp_value < 0:
            raise ValueError(f"invalid normalization values for {'.'.join(path)}")
        return cpp_value / py_value

    return {
        "latency_per_request": ratio(("latency_ms", "mean")),
        "latency_per_generated_action": ratio(("generated_action_step_ms", "mean")),
        "vram_peak": ratio(("sampled_peak_mib",)),
    }


def sha256_path(path):
    """Hash one file or a directory tree using the repository's manifest scheme."""
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise ValueError(f"artifact is neither file nor directory: {path}")
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def runtime_provenance():
    def output(command):
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else None
    return {
        "command": list(sys.argv),
        "git_revision": output(["git", "rev-parse", "HEAD"]),
        "git_status": output(["git", "status", "--short"]),
        "platform": platform.platform(),
        "nvidia_smi": output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                              "--format=csv,noheader"]),
    }


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def summarize(samples):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("expected nonempty finite nonnegative latency samples")
    return dict(mean=float(values.mean()), std=float(values.std()),
                **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)})


def smolvla_predictor(policy, preprocessor, postprocessor, images, state, task, noise):
    """Raw CPU tensors -> official processors/model -> full CPU action chunk."""
    import torch
    device = next(policy.parameters()).device
    raw_images = [torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0) for image in images]
    raw_state = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0)
    raw_noise = torch.from_numpy(np.ascontiguousarray(noise)).unsqueeze(0)

    def predict():
        # Build a fresh CPU-side observation mapping for every request. Tensor
        # contents are immutable; official preprocessing owns all normalization,
        # tokenization, and host-to-device transfer inside the timed call.
        batch = {
            "observation.images.image": raw_images[0],
            "observation.images.image2": raw_images[1],
            "observation.state": raw_state,
            "task": [task],
        }
        batch = preprocessor(batch)
        request_noise = raw_noise.to(device=device)
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch, noise=request_noise)
            actions = postprocessor(actions)
        return np.ascontiguousarray(actions[0].float().cpu().numpy(), dtype=np.float32)

    return predict


def cpp_predictor(client, observation, environment_action_dim=None):
    """Include response conversion and optional environment-action slicing."""
    def predict():
        actions = client._predict_chunk(observation)
        if environment_action_dim is not None:
            actions = actions[:, :environment_action_dim]
        return np.ascontiguousarray(actions, dtype=np.float32)
    return predict


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
    parser.add_argument("--arch", choices=("turbovla", "xr0", "xvla", "smolvla"), required=True)
    parser.add_argument("--backend", choices=("cpp", "python"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--memory-requests", type=int, default=20,
                        help="Separate untimed requests for process VRAM; 0 disables memory sampling.")
    parser.add_argument("--formal", action="store_true",
                        help="Enforce report minimums: 5 warmups, 100 timed, 20 memory requests")
    parser.add_argument("--address", default="tcp://127.0.0.1:5555")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--hf-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--instruction",
                        help="Exact task text for fixtures that do not embed an instruction")
    parser.add_argument("--compare-with", type=Path,
                        help="Peer result JSON; reject output unless strict normalization fields match")
    parser.add_argument("--artifact", type=Path, action="append", default=[],
                        help="Model/server artifact to hash; repeat for every C++ artifact")
    parser.add_argument("--source-checkpoint-sha256",
                        help="Source policy tree identity used to produce a C++ GGUF")
    parser.add_argument("--source-storage-type")
    parser.add_argument("--runtime-weight-type")
    parser.add_argument("--runtime-compute-type")
    parser.add_argument("--build-flags")
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
    try:
        validate_measurement_plan(args.n, args.warmup, args.memory_requests, args.formal)
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists() or args.output.with_suffix(".actions.npy").exists():
        parser.error("output already exists; choose a fresh run")
    if args.arch in ("xr0", "xvla", "smolvla") and args.hf_dir is None:
        parser.error("xr0/xvla/smolvla requires --hf-dir (matching tokenizer assets)")
    if args.domain_id < 0:
        parser.error("domain-id must be nonnegative")
    if args.xvla_noise_seed < 0:
        parser.error("xvla-noise-seed must be nonnegative")
    if args.backend == "python" and args.arch == "turbovla" and any(
        getattr(args, k) is None for k in ("checkpoint", "official_root", "bert_path", "norm_gguf")
    ):
        parser.error("TurboVLA Python requires checkpoint, official-root, bert-path and norm-gguf")
    if args.backend == "python" and args.arch == "smolvla" and args.checkpoint is None:
        parser.error("SmolVLA Python requires --checkpoint")
    if args.backend == "cpp" and args.arch == "smolvla":
        required = ("source_checkpoint_sha256", "source_storage_type",
                    "runtime_weight_type", "runtime_compute_type", "build_flags")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error("SmolVLA C++ evidence requires " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
        if len(args.artifact) < 3:
            parser.error("SmolVLA C++ evidence requires repeated --artifact for server, policy GGUF, and mmproj GGUF")
    if args.backend == "cpp" and (args.server_pid is None or args.server_pid < 1):
        parser.error("C++ requires the exact --server-pid for process VRAM")
    images, state, task, noise = load_fixture(args.fixture, args.arch, args.instruction)
    expected_image_size = 512 if args.arch == "smolvla" else 256
    if images.shape != (2, 3, expected_image_size, expected_image_size) or not np.isfinite(images).all():
        raise ValueError(f"fixture requires two finite {expected_image_size}px CHW images")
    if args.arch != "xr0" and (not np.issubdtype(images.dtype, np.floating) or images.min() < 0 or images.max() > 1):
        raise ValueError("TurboVLA/X-VLA/SmolVLA fixtures must use float images in [0, 1]")
    state_shapes = ((8,), (20,)) if args.arch == "xvla" else ((8,),)
    if state.shape not in state_shapes or not np.isfinite(state).all():
        raise ValueError(f"{args.arch} requires finite state with shape in {state_shapes}")
    if args.arch == "xr0" and (noise is None or noise.shape != (1, 30, 32) or not np.isfinite(noise).all()):
        raise ValueError("XR0 requires fixed 30x32 noise generated with the reference's seed/dtype/device")
    if args.arch == "xvla" and (noise is None or noise.shape != (1, 30, 20) or not np.isfinite(noise).all()):
        raise ValueError("X-VLA requires fixed finite 30x20 noise")
    if args.arch == "smolvla" and (noise is None or noise.shape != (50, 32) or not np.isfinite(noise).all()):
        raise ValueError("SmolVLA requires fixed finite 50x32 noise")
    if args.arch == "smolvla":
        # Set these before importing Transformers/Hugging Face modules: their
        # offline flags are cached at import time in some supported versions.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    import transformers
    obs = {"observation.images.image": images[0], "observation.images.image2": images[1],
           "observation.state": state, "task": task}
    if noise is not None:
        obs["action_noise"] = noise
    obs["domain_id"] = args.domain_id
    expected_shape = {"turbovla": (12, 7), "xr0": (30, 32), "xvla": (30, 20),
                      "smolvla": (50, 7)}[args.arch]
    cleanup = lambda: None
    precision_manifest = None
    if args.backend == "cpp":
        from client.vla_cpp_client import VlaCppClient
        client = VlaCppClient(args.address, arch=args.arch,
            tokenizer_name=str(args.hf_dir) if args.arch != "turbovla" else None,
            image_keys=("observation.images.image", "observation.images.image2"),
            image_size={"turbovla":256, "xr0":None, "xvla":224, "smolvla":512}[args.arch],
            max_state_dim={"turbovla":8, "xr0":32, "xvla":20, "smolvla":32}[args.arch],
            max_length={"turbovla":64, "xr0":512, "xvla":50, "smolvla":48}[args.arch],
            n_action_steps=expected_shape[0], real_action_dim=7)
        predict = cpp_predictor(client, obs, 7 if args.arch == "smolvla" else None)
        cleanup = client.close
        if args.arch == "smolvla":
            precision_manifest = {
                "source_checkpoint_storage": args.source_storage_type,
                "gguf_resident_weight_storage": args.runtime_weight_type,
                "runtime_compute": args.runtime_compute_type,
                "build_flags": args.build_flags,
            }
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
    elif args.arch == "smolvla":
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        config = PreTrainedConfig.from_pretrained(str(args.checkpoint), local_files_only=True)
        config.device = "cuda"
        config.load_vlm_weights = False
        config.vlm_model_name = str(args.hf_dir)
        policy = SmolVLAPolicy.from_pretrained(str(args.checkpoint), config=config,
            local_files_only=True, strict=False).eval()
        parameter_dtypes = sorted({str(parameter.dtype) for parameter in policy.parameters()})
        precision_manifest = {
            "source_checkpoint_storage": "safetensors",
            "runtime_parameter_dtypes": parameter_dtypes,
            "autocast": False,
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config, pretrained_path=str(args.checkpoint),
            preprocessor_overrides=smolvla_processor_overrides(args.hf_dir))
        predict = smolvla_predictor(policy, preprocessor, postprocessor,
                                    images, state, task, noise)
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
    samples, server_samples, action_sha256_samples = [], [], []
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
            action_sha256_samples.append(hashlib.sha256(
                np.ascontiguousarray(actions).tobytes(order="C")).hexdigest())
            if args.backend == "cpp":
                server_samples.append(client.get_last_inference_profile())
        # nvidia-smi and CUDA allocation can contend inside the driver. Keep
        # the instrument out of the latency window for both implementations.
        if args.memory_requests:
            sampler.sample_once()
            sampler.start()
            for _ in range(args.memory_requests):
                memory_actions = predict()
                if memory_actions.shape != expected_shape or not np.isfinite(memory_actions).all():
                    raise ValueError("invalid memory-phase output")
            sampler.sample_once()
    finally:
        if sampler.ident is not None:
            sampler.stop()
        cleanup()
    sources = sorted(set(sampler.sources))
    tokenizer_sha256 = sha256_path(args.hf_dir) if args.arch == "smolvla" else None
    source_checkpoint_sha256 = (
        sha256_path(args.checkpoint) if args.arch == "smolvla" and args.backend == "python"
        else args.source_checkpoint_sha256
    )
    artifact_paths = list(args.artifact)
    if args.arch == "smolvla" and args.backend == "python":
        artifact_paths.extend((args.checkpoint, args.hf_dir))
    elif args.arch == "smolvla":
        artifact_paths.append(args.hf_dir)
    artifact_sha256 = {str(Path(path).resolve(strict=True)): sha256_path(path)
                       for path in artifact_paths}
    generated_samples = [sample / expected_shape[0] for sample in samples]
    result = dict(arch=args.arch, backend=args.backend,
        measurement_plan={"n":args.n, "warmup":args.warmup,
                          "memory_requests":args.memory_requests, "formal":args.formal},
        arguments=jsonable(vars(args)),
        boundary_id=BOUNDARY_ID, boundary=__doc__, shape=list(expected_shape),
        generated_action_horizon=expected_shape[0], samples_ms=samples,
        latency_ms=summarize(samples), generated_action_step_samples_ms=generated_samples,
        generated_action_step_ms=summarize(generated_samples),
        action_sha256_samples=action_sha256_samples,
        memory_measurement="separate untimed requests after latency sampling; no nvidia-smi during timing",
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        instruction_sha256=hashlib.sha256(task.encode("utf-8")).hexdigest(),
        source_checkpoint_sha256=source_checkpoint_sha256,
        tokenizer_sha256=tokenizer_sha256,
        artifact_sha256=artifact_sha256,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        provenance=runtime_provenance(), precision_manifest=precision_manifest,
        torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
        fixture_state_dim=int(state.size),
        python_precision=({"policy":args.xvla_precision if args.arch == "xvla" else
                           args.xr0_policy_precision if args.arch == "xr0" else args.turbovla_precision,
                           "vision":args.xr0_vision_dtype if args.arch == "xr0" else
                           args.xvla_precision if args.arch == "xvla" else args.turbovla_precision}
                          if args.backend == "python" else None),
        server_samples=server_samples, vram_samples_mib=sampler.samples_mib, vram_sources=sources,
        gpu_uuids=sorted(set(sampler.gpu_uuids)),
        sampled_peak_mib=max(sampler.samples_mib) if sampler.samples_mib else None,
        process_level_vram=sources == ["process_used_memory"],
        sampled_process_peak_mib=max(sampler.samples_mib) if sampler.samples_mib and sources == ["process_used_memory"] else None)
    if args.backend == "python" and args.arch == "smolvla":
        result["python_precision"] = {"policy":"checkpoint/default", "vision":"checkpoint/default"}
    if args.compare_with:
        peer = json.loads(args.compare_with.read_text(encoding="utf-8"))
        validate_comparable_results(result, peer)
        result["compared_with"] = str(args.compare_with)
        result["normalized_to_python"] = normalized_to_python(result, peer)
    payload = json.dumps(result, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    actions_output = args.output.with_suffix(".actions.npy")
    temporary_json = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary_actions = actions_output.with_name(f".{actions_output.name}.tmp-{os.getpid()}")
    try:
        with temporary_json.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.write("\n")
        with temporary_actions.open("xb") as stream:
            np.save(stream, actions, allow_pickle=False)
        temporary_actions.replace(actions_output)
        temporary_json.replace(args.output)
    except BaseException:
        temporary_json.unlink(missing_ok=True)
        temporary_actions.unlink(missing_ok=True)
        raise
    print(json.dumps({"latency_ms": result["latency_ms"], "sampled_process_peak_mib": result["sampled_process_peak_mib"]}))


if __name__ == "__main__":
    main()
