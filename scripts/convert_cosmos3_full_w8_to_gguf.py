#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Convert a Cosmos3 RoboLab full_w8 bundle to GGUF.

The official RoboLab quant bundle is not a pure int8 checkpoint.  It contains
BF16 safetensors shards plus packed W8 linear modules under tensors/*.pt.  This
converter preserves that layout in one GGUF file:

  * BF16 safetensors are stored as s.000000, s.000001, ...
  * Packed W8 modules are stored as w.000000.q / w.000000.s, ...
  * GGUF C readers in llama.cpp currently cap tensor names and dimensions, so
    long Cosmos3 source names are not used as tensor names.  Tensors with more
    than four dimensions are flattened for storage.
  * Metadata keeps the readable structure:
      cosmos3.tensor_name_map_json
          Full mapping from short GGUF tensor names to original safetensors
          names or packed W8 module fields.  Each record includes dtype,
          original shape, stored shape, and source file/module.
      cosmos3.packed_w8_records_json
          W8-only module records for loader convenience.  Each module lists the
          original .pt file, backend_class, qweight/scales short names, and
          non-tensor fields such as bias/input_scale when they are None.

This script intentionally does not requantize or dequantize the official W8
weights.  The C++ runtime should interpret the packed representation directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if GGUF_PY.exists():
    sys.path.insert(0, str(GGUF_PY))

try:
    import gguf
except Exception as exc:  # pragma: no cover - user-facing startup error
    raise SystemExit(
        "failed to import gguf. Expected either an installed gguf package or "
        f"the vendored package at {GGUF_PY}. Original error: {exc}"
    )


ARCH = "cosmos3"
KV = lambda name: f"{ARCH}.{name}"

MAX_GGUF_TENSOR_NAME = 63


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _bf16_to_u16(t: torch.Tensor) -> np.ndarray:
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    return t.contiguous().view(torch.uint16).cpu().numpy()


def _gguf_compatible_tensor(t: torch.Tensor) -> torch.Tensor:
    if t.dim() <= 4:
        return t
    return t.reshape(int(t.shape[0]), -1)


def _add_torch_tensor(writer: "gguf.GGUFWriter", name: str, t: torch.Tensor) -> str:
    t = _gguf_compatible_tensor(t)
    if t.dtype == torch.bfloat16:
        writer.add_tensor(
            name,
            _bf16_to_u16(t),
            raw_shape=list(t.shape),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )
        return "BF16"
    if t.dtype == torch.float32:
        writer.add_tensor(
            name,
            t.contiguous().cpu().numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F32,
        )
        return "F32"
    if t.dtype == torch.float16:
        writer.add_tensor(
            name,
            t.contiguous().cpu().numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F16,
        )
        return "F16"
    if t.dtype == torch.int32:
        writer.add_tensor(
            name,
            t.contiguous().cpu().numpy().astype(np.int32, copy=False),
            raw_dtype=gguf.GGMLQuantizationType.I32,
        )
        return "I32"
    if t.dtype == torch.int16:
        writer.add_tensor(
            name,
            t.contiguous().cpu().numpy().astype(np.int16, copy=False),
            raw_dtype=gguf.GGMLQuantizationType.I16,
        )
        return "I16"
    if t.dtype == torch.int8:
        writer.add_tensor(
            name,
            t.contiguous().cpu().numpy().astype(np.int8, copy=False),
            raw_dtype=gguf.GGMLQuantizationType.I8,
        )
        return "I8"
    if t.dtype == torch.uint8:
        writer.add_tensor(
            name,
            t.contiguous().cpu().numpy().astype(np.uint8, copy=False),
            raw_dtype=gguf.GGMLQuantizationType.I8,
        )
        return "U8_AS_I8"
    raise NotImplementedError(f"unsupported dtype {t.dtype} for {name}")


def _tensor_shape(t: torch.Tensor) -> list[int]:
    return [int(x) for x in t.shape]


def _stored_tensor_shape(t: torch.Tensor) -> list[int]:
    return _tensor_shape(_gguf_compatible_tensor(t))


def _assert_short_tensor_name(name: str) -> None:
    if len(name.encode("utf-8")) > MAX_GGUF_TENSOR_NAME:
        raise SystemExit(
            f"internal converter error: GGUF tensor name is too long "
            f"({len(name.encode('utf-8'))} > {MAX_GGUF_TENSOR_NAME}): {name}"
        )


def _validate_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not bundle.exists():
        raise SystemExit(f"bundle directory does not exist: {bundle}")
    cfg_path = bundle / "config.json"
    runtime_path = bundle / "runtime" / "config.json"
    index_path = bundle / "model.safetensors.index.json"
    tensors_dir = bundle / "tensors"
    for path in (cfg_path, runtime_path, index_path, tensors_dir):
        if not path.exists():
            raise SystemExit(f"missing required Cosmos3 full_w8 bundle path: {path}")

    cfg = _load_json(cfg_path)
    runtime_cfg = _load_json(runtime_path)
    index = _load_json(index_path)
    if cfg.get("artifact_type") != "cosmos3_robolab_quantized_policy":
        raise SystemExit(f"unexpected artifact_type in {cfg_path}: {cfg.get('artifact_type')!r}")
    if cfg.get("model_type") != "cosmos3_omni_quant_bundle":
        raise SystemExit(f"unexpected model_type in {cfg_path}: {cfg.get('model_type')!r}")

    shards = sorted(set(str(v) for v in index.get("weight_map", {}).values()))
    missing = [s for s in shards if not (bundle / s).exists()]
    if missing:
        raise SystemExit(f"missing safetensors shards referenced by index: {missing}")
    if not list(tensors_dir.glob("*.pt")):
        raise SystemExit(f"no packed W8 .pt tensors found under {tensors_dir}")
    return cfg, runtime_cfg, index


def _add_metadata(
    writer: "gguf.GGUFWriter",
    bundle: Path,
    cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    index: dict[str, Any],
    *,
    include_hashes: bool,
) -> None:
    weight_map = index.get("weight_map", {})
    shards = sorted(set(str(v) for v in weight_map.values()))
    tensor_files = sorted((bundle / "tensors").glob("*.pt"))

    writer.add_name("Cosmos3 RoboLab full_w8")
    writer.add_description("Cosmos3 RoboLab full_w8 mixed BF16 + packed W8 bundle converted for Embodied.cpp")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    writer.add_string(KV("artifact_type"), str(cfg.get("artifact_type", "")))
    writer.add_string(KV("model_type"), str(cfg.get("model_type", "")))
    writer.add_string(KV("quantization_strategy"), "full_w8")
    writer.add_string(KV("source_bundle"), str(bundle))
    writer.add_uint32(KV("schema_version"), 1)
    writer.add_uint32(KV("safetensors_tensor_count"), len(weight_map))
    writer.add_uint32(KV("safetensors_shard_count"), len(shards))
    writer.add_uint32(KV("packed_w8_module_count"), len(tensor_files))
    writer.add_uint64(KV("safetensors_total_size"), int(index.get("metadata", {}).get("total_size", 0)))
    writer.add_string(KV("config_json"), json.dumps(cfg, sort_keys=True, separators=(",", ":")))
    writer.add_string(KV("runtime_config_json"), json.dumps(runtime_cfg, sort_keys=True, separators=(",", ":")))
    writer.add_string(KV("safetensors_index_json"), json.dumps(index, sort_keys=True, separators=(",", ":")))
    writer.add_string(KV("safetensors_shards_json"), json.dumps(shards, separators=(",", ":")))
    writer.add_string(KV("packed_w8_modules_json"), json.dumps([p.stem for p in tensor_files], separators=(",", ":")))

    if include_hashes:
        hashes = {str(p.relative_to(bundle)): _sha256_file(p) for p in [bundle / s for s in shards]}
        hashes.update({str(p.relative_to(bundle)): _sha256_file(p) for p in tensor_files})
        writer.add_string(KV("source_sha256_json"), json.dumps(hashes, sort_keys=True, separators=(",", ":")))


def _stream_safetensors(
    writer: "gguf.GGUFWriter",
    bundle: Path,
    index: dict[str, Any],
    *,
    dry_run: bool,
) -> tuple[int, Counter[str], list[dict[str, Any]]]:
    weight_map = index.get("weight_map", {})
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(str(shard), []).append(str(key))

    count = 0
    dtypes: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for shard in sorted(by_shard):
        shard_path = bundle / shard
        print(f"streaming {shard_path.name} ({len(by_shard[shard])} tensors)")
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for key in sorted(by_shard[shard]):
                dst_name = f"s.{count:06d}"
                _assert_short_tensor_name(dst_name)
                t = sf.get_tensor(key)
                dtype_name = str(t.dtype).replace("torch.", "")
                dtypes[dtype_name] += 1
                records.append({
                    "gguf_name": dst_name,
                    "source_name": key,
                    "source_file": shard,
                    "dtype": dtype_name,
                    "shape": _tensor_shape(t),
                    "stored_shape": _stored_tensor_shape(t),
                })
                count += 1
                if dry_run:
                    continue
                _add_torch_tensor(writer, dst_name, t)
    return count, dtypes, records


def _load_pt(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _stream_w8_modules(
    writer: "gguf.GGUFWriter",
    bundle: Path,
    *,
    dry_run: bool,
) -> tuple[int, int, Counter[str], list[dict[str, Any]]]:
    tensor_dir = bundle / "tensors"
    module_files = sorted(tensor_dir.glob("*.pt"))
    manifest_path = bundle / "manifest.json"
    manifest_modules: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        for item in manifest.get("modules", []):
            name = str(item.get("name", ""))
            tensor_file = str(item.get("tensor_file", ""))
            if name:
                manifest_modules[name] = item
            if tensor_file:
                manifest_modules[Path(tensor_file).stem] = item
    tensor_count = 0
    dtype_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []

    for i, path in enumerate(module_files, 1):
        module_index = i - 1
        module_name = path.stem
        if i == 1 or i % 50 == 0 or i == len(module_files):
            print(f"streaming W8 module {i}/{len(module_files)}: {module_name}")
        obj = _load_pt(path)
        if not isinstance(obj, dict):
            raise SystemExit(f"expected dict in packed tensor file: {path}")
        manifest_rec = manifest_modules.get(module_name, {})

        rec: dict[str, Any] = {
            "module": module_name,
            "file": str(path.relative_to(bundle)),
            "backend_class": str(manifest_rec.get("backend_class", obj.get("backend_class", ""))),
            "tensors": {},
            "non_tensors": {},
        }
        for field in ("format", "size_k", "size_n", "group_size", "num_bits", "wtype_id"):
            if field in manifest_rec:
                rec[field] = manifest_rec[field]
        for field, value in sorted(obj.items()):
            if torch.is_tensor(value):
                if field == "qweight":
                    short_field = "q"
                elif field == "scales":
                    short_field = "s"
                elif field == "bias":
                    short_field = "b"
                elif field == "input_scale":
                    short_field = "is"
                else:
                    short_field = f"t{len(rec['tensors']):02d}"
                dst_name = f"w.{module_index:06d}.{short_field}"
                _assert_short_tensor_name(dst_name)
                rec["tensors"][field] = {
                    "gguf_name": dst_name,
                    "dtype": str(value.dtype).replace("torch.", ""),
                    "shape": _tensor_shape(value),
                    "stored_shape": _stored_tensor_shape(value),
                }
                dtype_counts[str(value.dtype).replace("torch.", "")] += 1
                tensor_count += 1
                if not dry_run:
                    _add_torch_tensor(writer, dst_name, value)
            else:
                rec["non_tensors"][field] = None if value is None else str(value)
        records.append(rec)

    if not dry_run:
        writer.add_string(KV("packed_w8_records_json"), json.dumps(records, sort_keys=True, separators=(",", ":")))
    return len(module_files), tensor_count, dtype_counts, records


def _stream_wan_vae_encoder(
    writer: "gguf.GGUFWriter",
    vae_pth: Path,
    *,
    dry_run: bool,
) -> tuple[int, Counter[str], list[dict[str, Any]]]:
    state = _load_pt(vae_pth)
    if not isinstance(state, dict):
        raise SystemExit(f"expected a state dict in {vae_pth}, got {type(state).__name__}")

    encoder_items: list[tuple[str, torch.Tensor]] = []
    for name, value in state.items():
        if name.startswith("encoder.") or name in {"conv1.weight", "conv1.bias"}:
            if not torch.is_tensor(value):
                continue
            encoder_items.append((name, value))

    if not encoder_items:
        raise SystemExit(f"no encoder tensors found in {vae_pth}")

    count = 0
    dtypes: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for name, value in encoder_items:
        gguf_name = f"vae.{count:06d}"
        _assert_short_tensor_name(gguf_name)
        t = value.to(torch.bfloat16)
        dtypes["bfloat16"] += 1
        records.append(
            {
                "gguf_name": gguf_name,
                "source_name": f"wan22_vae.{name}",
                "source_file": str(vae_pth.name),
                "dtype": "bfloat16",
                "shape": _tensor_shape(value),
                "stored_shape": _stored_tensor_shape(t),
            }
        )
        count += 1
        if dry_run:
            continue
        _add_torch_tensor(writer, gguf_name, t)

    scale_mean = torch.tensor([
        -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
        -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
        -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
        -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
        -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
        0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
    ], dtype=torch.bfloat16)
    scale_std = torch.tensor([
        0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
        0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
        0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
        0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
        0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
        0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
    ], dtype=torch.float32)
    scale_inv_std = (1.0 / scale_std).to(torch.bfloat16)
    for name, tensor in (("vae.scale.mean", scale_mean), ("vae.scale.inv_std", scale_inv_std)):
        gguf_name = f"vae.{count:06d}"
        _assert_short_tensor_name(gguf_name)
        dtypes["bfloat16"] += 1
        records.append(
            {
                "gguf_name": gguf_name,
                "source_name": f"wan22_vae.{name}",
                "source_file": str(vae_pth.name),
                "dtype": "bfloat16",
                "shape": _tensor_shape(tensor),
                "stored_shape": _stored_tensor_shape(tensor),
            }
        )
        count += 1
        if dry_run:
            continue
        _add_torch_tensor(writer, gguf_name, tensor)

    return count, dtypes, records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "bundle",
        type=Path,
        help="Cosmos3 RoboLab full_w8 bundle directory.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output GGUF path.",
    )
    ap.add_argument(
        "--include-source-hashes",
        action="store_true",
        help="Compute SHA256 for source shards and W8 .pt files and store them as metadata.",
    )
    ap.add_argument(
        "--include-vae-encoder",
        action="store_true",
        help="Also store Wan2.2 VAE encoder tensors and latent scale constants in the GGUF.",
    )
    ap.add_argument(
        "--vae-path",
        type=Path,
        help="Wan2.2_VAE.pth path used with --include-vae-encoder. Defaults to <bundle>/assets/Wan2.2_VAE.pth.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and scan tensors without writing a GGUF file.",
    )
    args = ap.parse_args()
    if args.include_vae_encoder and args.vae_path is None:
        args.vae_path = args.bundle / "assets" / "Wan2.2_VAE.pth"

    bundle = args.bundle.resolve()
    out = args.out.resolve()
    cfg, runtime_cfg, index = _validate_bundle(bundle)

    print(f"bundle: {bundle}")
    print(f"output: {out}")
    print(f"strategy: full_w8")

    writer = None
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = gguf.GGUFWriter(str(out), arch=ARCH, use_temp_file=True)
        _add_metadata(writer, bundle, cfg, runtime_cfg, index, include_hashes=args.include_source_hashes)

    assert writer is not None or args.dry_run
    safetensor_count, safetensor_dtypes, safetensor_records = _stream_safetensors(
        writer, bundle, index, dry_run=args.dry_run  # type: ignore[arg-type]
    )
    w8_modules, w8_tensor_count, w8_dtypes, w8_records = _stream_w8_modules(
        writer, bundle, dry_run=args.dry_run  # type: ignore[arg-type]
    )
    vae_tensor_count = 0
    vae_dtypes: Counter[str] = Counter()
    vae_records: list[dict[str, Any]] = []
    if args.include_vae_encoder:
        vae_path = args.vae_path.resolve()
        print(f"streaming Wan2.2 VAE encoder: {vae_path}")
        vae_tensor_count, vae_dtypes, vae_records = _stream_wan_vae_encoder(
            writer, vae_path, dry_run=args.dry_run  # type: ignore[arg-type]
        )

    if not args.dry_run:
        assert writer is not None
        writer.add_string(KV("safetensors_dtype_counts_json"), json.dumps(safetensor_dtypes, sort_keys=True))
        writer.add_string(KV("packed_w8_dtype_counts_json"), json.dumps(w8_dtypes, sort_keys=True))
        if args.include_vae_encoder:
            writer.add_string(KV("vae_encoder_dtype_counts_json"), json.dumps(vae_dtypes, sort_keys=True))
            writer.add_string(KV("vae_encoder_records_json"), json.dumps(vae_records, sort_keys=True, separators=(",", ":")))
        writer.add_string(
            KV("tensor_name_map_json"),
            json.dumps(
                {
                    "safetensors": safetensor_records,
                    "packed_w8": w8_records,
                    "vae_encoder": vae_records,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    print("summary:")
    print(f"  safetensors tensors: {safetensor_count} dtypes={dict(safetensor_dtypes)}")
    print(f"  packed W8 modules:   {w8_modules}")
    print(f"  packed W8 tensors:   {w8_tensor_count} dtypes={dict(w8_dtypes)}")
    if args.include_vae_encoder:
        print(f"  VAE encoder tensors: {vae_tensor_count} dtypes={dict(vae_dtypes)}")
    if not args.dry_run:
        print(f"done. {out} ({out.stat().st_size / (1024**3):.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
