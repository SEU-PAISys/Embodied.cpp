#!/usr/bin/env python3
"""Create a non-overwriting SmolVLA Q8_0 policy GGUF.

The policy is selectively quantized for native GGML matmul residency.  The
VLM/action-expert block matrices, connector, and action/time projection
matrices are eligible; embeddings, norms, biases, statistics, sensitive
state/output projections, small tensors, and incompatible layouts are copied
at their source dtype.  A deterministic JSON manifest accompanies the output.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if GGUF_PY.exists():
    sys.path.insert(0, str(GGUF_PY))

try:
    import gguf
except Exception as exc:  # pragma: no cover - environment-specific import failure
    raise SystemExit(f"failed to import gguf from {GGUF_PY}: {exc}")


QTYPE_BY_NAME = {
    "Q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}
CANONICAL_QTYPE = "Q8_0"
FILE_TYPE = gguf.LlamaFileType.MOSTLY_Q8_0
MANIFEST_SCHEMA = "smolvla_q8_0_quantization.v1"
QUANTIZATION_SCOPE = "main_model_full"

ELIGIBLE_PATTERNS = {
    "vlm": re.compile(
        r"^vlm\.blk\.\d+\.(?:attn_[qkvo]|ffn_(?:gate|up|down))\.weight$"
    ),
    "action_expert": re.compile(
        r"^aex\.blk\.\d+\.(?:attn_[qkvo]|ffn_(?:gate|up|down))\.weight$"
    ),
    "connector": re.compile(r"^connector\.weight$"),
    "action_projection": re.compile(
        r"^action_(?:in_proj|time_mlp_in|time_mlp_out)\.weight$"
    ),
}

SENSITIVE_PATTERNS = {
    "sensitive_projection": re.compile(r"^(?:state_proj|action_out_proj)\.(?:weight|bias)$"),
    "embedding": re.compile(
        r"(^|[._])(?:embed|embedding|embd|lm_head|vocab)([._]|$)", re.I
    ),
    "normalization": re.compile(r"(^|[._])(?:norm|layernorm|ln|rms)([._]|$)", re.I),
    "bias": re.compile(r"(?:^|[._])bias$", re.I),
    "statistics": re.compile(
        r"(^|[._])(?:stats?|mean|std|scale|zero|quant|cache|mask)([._]|$)", re.I
    ),
}


@dataclass(frozen=True)
class Decision:
    quantize: bool
    reason: str
    category: str


def _load_ggml(lib_path: Path) -> ctypes.CDLL:
    if not lib_path.is_file():
        raise FileNotFoundError(f"missing GGML quantizer library: {lib_path}")
    lib = ctypes.CDLL(str(lib_path))
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    return lib


def _bf16_bytes_to_f32(tensor: "gguf.ReaderTensor") -> np.ndarray:
    u16 = np.asarray(tensor.data, dtype=np.uint8).view(np.uint16)
    logical_shape = tuple(reversed([int(x) for x in tensor.shape.tolist()]))
    u16 = u16.reshape(logical_shape)
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return np.ascontiguousarray(u32.view(np.float32))


def _f32_tensor(tensor: "gguf.ReaderTensor") -> np.ndarray:
    if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
        return np.ascontiguousarray(tensor.data.astype(np.float32, copy=False))
    if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
        return np.ascontiguousarray(tensor.data.astype(np.float32, copy=False))
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        return _bf16_bytes_to_f32(tensor)
    raise ValueError(f"unsupported source dtype for {tensor.name}: {tensor.tensor_type.name}")


def _is_source_float(tensor: "gguf.ReaderTensor") -> bool:
    return tensor.tensor_type in (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    )


def _tensor_numel(tensor: "gguf.ReaderTensor") -> int:
    return int(np.prod([int(value) for value in tensor.shape.tolist()]))


def tensor_category(name: str) -> str:
    for category, pattern in ELIGIBLE_PATTERNS.items():
        if pattern.fullmatch(name):
            return category
    for category, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(name):
            return category
    return "other"


def _decide_tensor(
    tensor: "gguf.ReaderTensor",
    qtype: "gguf.GGMLQuantizationType",
    *,
    min_elements: int,
) -> Decision:
    if qtype != gguf.GGMLQuantizationType.Q8_0:
        raise ValueError("SmolVLA quantization supports only Q8_0")
    category = tensor_category(tensor.name)
    if category == "sensitive_projection":
        return Decision(False, "sensitive-projection", category)
    if category not in ELIGIBLE_PATTERNS:
        if category in SENSITIVE_PATTERNS:
            return Decision(False, "sensitive-category", category)
        return Decision(False, "not-eligible", category)
    if not _is_source_float(tensor):
        return Decision(False, f"source-{tensor.tensor_type.name}", category)
    shape = [int(value) for value in tensor.shape.tolist()]
    if len(shape) < 2:
        return Decision(False, "rank<2", category)
    if _tensor_numel(tensor) < min_elements:
        return Decision(False, "too-small", category)
    block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
    if shape[0] % block_size != 0:
        return Decision(False, f"ne0-not-divisible-by-{block_size}", category)
    return Decision(True, "eligible-matrix", category)


def _quantized_nbytes(
    tensor: "gguf.ReaderTensor", qtype: "gguf.GGMLQuantizationType"
) -> int:
    shape = [int(value) for value in tensor.shape.tolist()]
    n_per_row = shape[0]
    nrows = int(np.prod(shape) // n_per_row)
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    return nrows * (n_per_row // block_size) * type_size


def _quantize_tensor(
    lib: ctypes.CDLL | None,
    tensor: "gguf.ReaderTensor",
    qtype: "gguf.GGMLQuantizationType",
) -> np.ndarray:
    data = _f32_tensor(tensor)
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite source tensor: {tensor.name}")
    shape = [int(value) for value in tensor.shape.tolist()]
    n_per_row = shape[0]
    nrows = int(np.prod(shape) // n_per_row)
    qbytes = _quantized_nbytes(tensor, qtype)
    if lib is None:
        quantized = gguf.quantize(data, qtype)
        byte_shape = gguf.quant_shape_to_byte_shape(tuple(reversed(shape)), qtype)
        if quantized.nbytes != qbytes:
            raise RuntimeError(
                f"quantized byte mismatch for {tensor.name}: "
                f"{quantized.nbytes} vs {qbytes}"
            )
        return np.ascontiguousarray(quantized).reshape(byte_shape)
    out = np.empty(qbytes, dtype=np.uint8)
    written = lib.ggml_quantize_chunk(
        int(qtype),
        data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.c_void_p),
        0,
        nrows,
        n_per_row,
        None,
    )
    if written != qbytes:
        raise RuntimeError(f"quantized byte mismatch for {tensor.name}: {written} vs {qbytes}")
    byte_shape = gguf.quant_shape_to_byte_shape(tuple(reversed(shape)), qtype)
    return out.reshape(byte_shape)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close_reader(reader: "gguf.GGUFReader") -> None:
    """Release the Windows memmap backing a GGUFReader."""
    data = getattr(reader, "data", None)
    mmap = getattr(data, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _copy_metadata(
    reader: "gguf.GGUFReader",
    writer: "gguf.GGUFWriter",
    *,
    quantized_count: int,
    skipped_sensitive_count: int,
) -> None:
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key in ("general.architecture", "general.file_type"):
            continue
        subtype = field.types[1] if len(field.types) > 1 else None
        writer.add_key_value(key, field.contents(), field.types[0], subtype)
    writer.add_file_type(FILE_TYPE)
    writer.add_string("smolvla.quantized_by", "scripts/quantize_smolvla_full_gguf.py")
    writer.add_string("smolvla.quantization", CANONICAL_QTYPE)
    writer.add_string("smolvla.quantization_scope", QUANTIZATION_SCOPE)
    writer.add_uint32("smolvla.quantized_tensor_count", quantized_count)
    writer.add_uint32("smolvla.skipped_sensitive_tensor_count", skipped_sensitive_count)


def _plan(
    reader: "gguf.GGUFReader",
    qtype: "gguf.GGMLQuantizationType",
    *,
    min_elements: int,
) -> tuple[dict[str, Decision], int, int, int]:
    decisions: dict[str, Decision] = {}
    selected = 0
    source_bytes = 0
    output_bytes = 0
    for tensor in reader.tensors:
        decision = _decide_tensor(tensor, qtype, min_elements=min_elements)
        decisions[tensor.name] = decision
        source_bytes += int(tensor.n_bytes)
        if decision.quantize:
            selected += 1
            output_bytes += _quantized_nbytes(tensor, qtype)
        else:
            output_bytes += int(tensor.n_bytes)
    return decisions, selected, source_bytes, output_bytes


def _manifest(
    reader: "gguf.GGUFReader",
    decisions: dict[str, Decision],
    *,
    source: Path,
    output: Path,
    source_bytes: int,
    output_bytes: int,
    selected: int,
    output_sha256: str | None,
) -> dict[str, Any]:
    source_counts: dict[str, int] = {}
    output_counts: dict[str, int] = {}
    tensors: list[dict[str, Any]] = []
    skipped_sensitive = 0
    selected_quantized_bytes = 0
    for tensor in reader.tensors:
        decision = decisions[tensor.name]
        source_type = tensor.tensor_type.name
        output_type = CANONICAL_QTYPE if decision.quantize else source_type
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
        output_counts[output_type] = output_counts.get(output_type, 0) + 1
        if decision.category in SENSITIVE_PATTERNS:
            skipped_sensitive += 1
        tensor_output_bytes = (
            _quantized_nbytes(tensor, gguf.GGMLQuantizationType.Q8_0)
            if decision.quantize else int(tensor.n_bytes)
        )
        if decision.quantize:
            selected_quantized_bytes += tensor_output_bytes
        tensors.append({
            "name": tensor.name,
            "shape": [int(value) for value in tensor.shape.tolist()],
            "category": decision.category,
            "selected": decision.quantize,
            "reason": decision.reason,
            "source_type": source_type,
            "output_type": output_type,
            "source_bytes": int(tensor.n_bytes),
            "output_bytes": tensor_output_bytes,
        })
    metadata_identity = {
        key: _json_safe(field.contents())
        for key, field in sorted(reader.fields.items())
        if key.startswith("smolvla.")
        and key not in {
            "smolvla.quantized_by",
            "smolvla.quantization",
            "smolvla.quantization_scope",
            "smolvla.quantized_tensor_count",
            "smolvla.skipped_sensitive_tensor_count",
        }
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "model": "smolvla",
        "qtype": CANONICAL_QTYPE,
        "quantization_scope": QUANTIZATION_SCOPE,
        "source_sha256": _sha256(source),
        "output_sha256": output_sha256,
        "source_tensor_count": len(reader.tensors),
        "quantized_tensor_count": selected,
        "selected_quantized_bytes": selected_quantized_bytes,
        "skipped_sensitive_tensor_count": skipped_sensitive,
        "source_tensor_bytes": source_bytes,
        "output_tensor_bytes": output_bytes,
        "source_dtype_counts": dict(sorted(source_counts.items())),
        "output_dtype_counts": dict(sorted(output_counts.items())),
        "metadata_identity": metadata_identity,
        "tensors": tensors,
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_qtype(name: str) -> str:
    if name not in QTYPE_BY_NAME:
        raise ValueError("SmolVLA quantization supports only Q8_0")
    return CANONICAL_QTYPE


def quantize_file(
    source: Path,
    output: Path,
    qtype_name: str = "Q8_0",
    *,
    min_elements: int = 4096,
    ggml_lib: Path | None = None,
    manifest_path: Path | None = None,
) -> int:
    """Quantize a local SmolVLA policy and return the Q8_0 tensor count."""
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(f"source policy does not exist: {source}")
    if output.exists() or output.is_symlink() or source.resolve() == output.resolve():
        raise FileExistsError(f"refusing to overwrite {output}")
    if min_elements < 1:
        raise ValueError("min_elements must be positive")
    canonical = _canonical_qtype(qtype_name)
    manifest_path = Path(manifest_path) if manifest_path else output.with_suffix(
        output.suffix + ".manifest.json"
    )
    if manifest_path.resolve() == output.resolve():
        raise FileExistsError("manifest path must differ from output path")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")

    reader = gguf.GGUFReader(str(source))
    arch = reader.fields.get("general.architecture")
    if arch is None or arch.contents() != "smolvla":
        raise ValueError("source policy must have general.architecture=smolvla")
    qtype = QTYPE_BY_NAME[canonical]
    decisions, selected, source_bytes, output_bytes = _plan(
        reader, qtype, min_elements=min_elements
    )
    if selected == 0:
        raise ValueError("no eligible Q8_0 tensors selected")
    lib = _load_ggml(ggml_lib) if ggml_lib is not None else None

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".smolvla-q8-", dir=output.parent) as scratch:
        prepared = Path(scratch) / "prepared.gguf"
        writer = gguf.GGUFWriter(str(prepared), arch="smolvla", use_temp_file=True)
        try:
            skipped_sensitive = sum(
                decision.category in SENSITIVE_PATTERNS
                for decision in decisions.values()
            )
            _copy_metadata(
                reader,
                writer,
                quantized_count=selected,
                skipped_sensitive_count=skipped_sensitive,
            )
            for tensor in reader.tensors:
                decision = decisions[tensor.name]
                if decision.quantize:
                    writer.add_tensor(
                        tensor.name,
                        _quantize_tensor(lib, tensor, qtype),
                        raw_dtype=qtype,
                    )
                else:
                    writer.add_tensor(
                        tensor.name,
                        tensor.data,
                        raw_dtype=tensor.tensor_type,
                    )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
            if writer.temp_file is not None:
                writer.temp_file.close()
        os.link(prepared, output)

    try:
        manifest = _manifest(
            reader,
            decisions,
            source=source,
            output=output,
            source_bytes=source_bytes,
            output_bytes=output_bytes,
            selected=selected,
            output_sha256=_sha256(output),
        )
        _write_manifest(manifest_path, manifest)
        return selected
    finally:
        _close_reader(reader)


def _print_inventory(
    source: Path,
    output: Path,
    reader: "gguf.GGUFReader",
    decisions: dict[str, Decision],
    selected: int,
    source_bytes: int,
    output_bytes: int,
) -> None:
    print(f"input={source}")
    print(f"output={output}")
    print(f"qtype={CANONICAL_QTYPE}")
    print(f"quantization_scope={QUANTIZATION_SCOPE}")
    print(f"quantized_tensors={selected} / {len(reader.tensors)}")
    print(f"tensor_bytes: {source_bytes / 1024**3:.2f} GiB -> {output_bytes / 1024**3:.2f} GiB")
    print(f"ratio={output_bytes / max(1, source_bytes):.3f}")
    for tensor in reader.tensors:
        decision = decisions[tensor.name]
        if not decision.quantize:
            print(
                f"  skip {decision.reason}: {tensor.name} "
                f"category={decision.category} dtype={tensor.tensor_type.name} "
                f"shape={[int(value) for value in tensor.shape.tolist()]}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qtype", default="Q8_0", choices=sorted(QTYPE_BY_NAME))
    parser.add_argument(
        "--ggml-lib",
        type=Path,
        help="Optional local libggml-base for native quantization; bundled Q8_0 "
             "Python codecs are used when omitted.",
    )
    parser.add_argument("--min-elements", type=int, default=4096)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = args.input
    if not source.is_file():
        raise SystemExit(f"source policy does not exist: {source}")
    output = args.output
    canonical = _canonical_qtype(args.qtype)
    reader = gguf.GGUFReader(str(source))
    decisions, selected, source_bytes, output_bytes = _plan(
        reader,
        QTYPE_BY_NAME[canonical],
        min_elements=args.min_elements,
    )
    _print_inventory(source, output, reader, decisions, selected, source_bytes, output_bytes)
    if args.dry_run:
        if args.manifest:
            _write_manifest(
                args.manifest,
                _manifest(
                    reader,
                    decisions,
                    source=source,
                    output=output,
                    source_bytes=source_bytes,
                    output_bytes=output_bytes,
                    selected=selected,
                    output_sha256=None,
                ),
            )
            print(f"wrote manifest={args.manifest}")
        return 0
    count = quantize_file(
        source,
        output,
        canonical,
        min_elements=args.min_elements,
        ggml_lib=args.ggml_lib,
        manifest_path=args.manifest,
    )
    print(f"wrote {output} with {count} {CANONICAL_QTYPE} tensors")
    print("Native Q8_0 residency is required; startup evidence must report resident Q8_0 tensors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
