#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline-quantize the main HY-VLA GGUF model weights.

This is broader than quantize_hy_vla_gguf.py: it quantizes every large
matrix-like model tensor that ggml can store in the requested quantized format,
while keeping numerically sensitive or non-matmul tensors unchanged.  By
default it excludes normalization, bias, embeddings, positional/rope tensors,
statistics, and other small tensors.
"""

from __future__ import annotations

import argparse
import ctypes
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if GGUF_PY.exists():
    sys.path.insert(0, str(GGUF_PY))

try:
    import gguf
except Exception as exc:
    raise SystemExit(f"failed to import gguf from {GGUF_PY}: {exc}")


QTYPE_BY_NAME = {
    "q4_K": gguf.GGMLQuantizationType.Q4_K,
    "q4_k": gguf.GGMLQuantizationType.Q4_K,
    "Q4_K": gguf.GGMLQuantizationType.Q4_K,
    "q5_K": gguf.GGMLQuantizationType.Q5_K,
    "q5_k": gguf.GGMLQuantizationType.Q5_K,
    "Q5_K": gguf.GGMLQuantizationType.Q5_K,
    "q6_K": gguf.GGMLQuantizationType.Q6_K,
    "q6_k": gguf.GGMLQuantizationType.Q6_K,
    "Q6_K": gguf.GGMLQuantizationType.Q6_K,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "Q8_0": gguf.GGMLQuantizationType.Q8_0,
}

DEFAULT_EXCLUDE_RE = re.compile(
    r"(^|[._])("
    r"bias|norm|layernorm|ln|rms|"
    r"state_proj|action_out_proj|"
    r"embed|embedding|lm_head|pos|position|rope|rotary|inv_freq|"
    r"token|vocab|stats?|mean|std|scale|zero|quant|"
    r"cache|mask"
    r")([._]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Decision:
    quantize: bool
    reason: str


def _load_ggml(lib_path: Path) -> ctypes.CDLL:
    if not lib_path.exists():
        raise SystemExit(f"missing {lib_path}; build Embodied.cpp/llama.cpp first")
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
        gguf.GGMLQuantizationType.BF16,
        gguf.GGMLQuantizationType.F16,
    )


def _tensor_numel(tensor: "gguf.ReaderTensor") -> int:
    return int(np.prod([int(x) for x in tensor.shape.tolist()]))


def _matches_any(name: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(name) for pattern in patterns)


def _decide_tensor(
    tensor: "gguf.ReaderTensor",
    qtype: "gguf.GGMLQuantizationType",
    *,
    min_elements: int,
    include: list[re.Pattern[str]],
    exclude: list[re.Pattern[str]],
    force_include: list[re.Pattern[str]],
) -> Decision:
    name = tensor.name
    if force_include and _matches_any(name, force_include):
        forced = True
    else:
        forced = False

    if not forced and include and not _matches_any(name, include):
        return Decision(False, "not-in-include")
    if not forced and _matches_any(name, exclude):
        return Decision(False, "excluded-sensitive")
    if not _is_source_float(tensor):
        return Decision(False, f"source-{tensor.tensor_type.name}")

    ne = [int(x) for x in tensor.shape.tolist()]
    if len(ne) < 2:
        return Decision(False, "rank<2")
    if _tensor_numel(tensor) < min_elements and not forced:
        return Decision(False, "too-small")

    block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
    if ne[0] % block_size != 0:
        return Decision(False, f"ne0-not-divisible-by-{block_size}")
    return Decision(True, "forced" if forced else "main-matrix")


def _quantized_nbytes(tensor: "gguf.ReaderTensor", qtype: "gguf.GGMLQuantizationType") -> int:
    ne = [int(x) for x in tensor.shape.tolist()]
    n_per_row = ne[0]
    nrows = int(np.prod(ne) // n_per_row)
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    return nrows * (n_per_row // block_size) * type_size


def _quantize_tensor(
    lib: ctypes.CDLL,
    tensor: "gguf.ReaderTensor",
    qtype: "gguf.GGMLQuantizationType",
) -> np.ndarray:
    data = _f32_tensor(tensor)
    ne = [int(x) for x in tensor.shape.tolist()]
    n_per_row = ne[0]
    nrows = int(np.prod(ne) // n_per_row)
    qbytes = _quantized_nbytes(tensor, qtype)
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
    byte_shape = gguf.quant_shape_to_byte_shape(tuple(reversed(ne)), qtype)
    return out.reshape(byte_shape)


def _copy_metadata(
    reader: "gguf.GGUFReader",
    writer: "gguf.GGUFWriter",
    *,
    qtype: str,
    quantized_count: int,
    skipped_sensitive_count: int,
) -> None:
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        writer.add_key_value(key, field.contents(), field.types[0])
    writer.add_string("hy_vla.quantized_by", "scripts/quantize_hy_vla_full_gguf.py")
    writer.add_string("hy_vla.quantization", qtype)
    writer.add_string("hy_vla.quantization_scope", "main_model_full")
    writer.add_uint32("hy_vla.quantized_tensor_count", quantized_count)
    writer.add_uint32("hy_vla.skipped_sensitive_tensor_count", skipped_sensitive_count)


def _compile_patterns(values: list[str] | None) -> list[re.Pattern[str]]:
    return [re.compile(value) for value in (values or [])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qtype", default="q4_K", choices=sorted(QTYPE_BY_NAME))
    parser.add_argument("--ggml-lib", type=Path, default=REPO_ROOT / "build" / "bin" / "libggml-base.so")
    parser.add_argument("--min-elements", type=int, default=4096,
                        help="Do not quantize tensors smaller than this unless forced.")
    parser.add_argument("--include-regex", action="append",
                        help="Only consider tensor names matching this regex. Repeatable.")
    parser.add_argument("--exclude-regex", action="append",
                        help="Additional regex for tensors to keep unquantized. Repeatable.")
    parser.add_argument("--force-include-regex", action="append",
                        help="Quantize matching tensors even if they match the default exclusions. Repeatable.")
    parser.add_argument("--no-default-excludes", action="store_true",
                        help="Disable built-in norm/bias/embed/pos/stat exclusions.")
    parser.add_argument("--list-skipped", type=int, default=80,
                        help="Print at most this many skipped tensors in the dry-run/report.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qtype = QTYPE_BY_NAME[args.qtype]
    lib = _load_ggml(args.ggml_lib)
    reader = gguf.GGUFReader(str(args.input))
    include = _compile_patterns(args.include_regex)
    exclude = _compile_patterns(args.exclude_regex)
    if not args.no_default_excludes:
        exclude.insert(0, DEFAULT_EXCLUDE_RE)
    force_include = _compile_patterns(args.force_include_regex)

    decisions: dict[str, Decision] = {}
    old_total = sum(int(t.n_bytes) for t in reader.tensors)
    new_total = 0
    reason_counts: dict[str, int] = {}
    quant_names: set[str] = set()
    skipped_sensitive_count = 0

    for tensor in reader.tensors:
        decision = _decide_tensor(
            tensor,
            qtype,
            min_elements=args.min_elements,
            include=include,
            exclude=exclude,
            force_include=force_include,
        )
        decisions[tensor.name] = decision
        reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
        if decision.quantize:
            quant_names.add(tensor.name)
            new_total += _quantized_nbytes(tensor, qtype)
        else:
            new_total += int(tensor.n_bytes)
            if decision.reason == "excluded-sensitive":
                skipped_sensitive_count += 1

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"qtype={qtype.name}")
    print(f"min_elements={args.min_elements}")
    print(f"quantized_tensors={len(quant_names)} / {len(reader.tensors)}")
    print(f"reason_counts={dict(sorted(reason_counts.items()))}")
    print(f"tensor_bytes: {old_total / 1024**3:.2f} GiB -> {new_total / 1024**3:.2f} GiB")
    print(f"ratio={new_total / max(1, old_total):.3f}")

    if args.list_skipped > 0:
        shown = 0
        for tensor in reader.tensors:
            decision = decisions[tensor.name]
            if decision.quantize:
                continue
            if shown >= args.list_skipped:
                break
            print(
                f"  skip {decision.reason}: {tensor.name} "
                f"dtype={tensor.tensor_type.name} shape={[int(x) for x in tensor.shape.tolist()]}"
            )
            shown += 1

    if args.dry_run:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(args.output.parent)
    writer = gguf.GGUFWriter(str(args.output), arch="hy_vla", use_temp_file=True)
    _copy_metadata(
        reader,
        writer,
        qtype=args.qtype,
        quantized_count=len(quant_names),
        skipped_sensitive_count=skipped_sensitive_count,
    )

    q_count = 0
    for idx, tensor in enumerate(reader.tensors, start=1):
        if tensor.name in quant_names:
            arr = _quantize_tensor(lib, tensor, qtype)
            writer.add_tensor(tensor.name, arr, raw_dtype=qtype)
            q_count += 1
            if q_count <= 10 or q_count % 50 == 0:
                print(f"[{idx:04d}/{len(reader.tensors)}] quantized {tensor.name} -> {qtype.name}")
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            if idx % 250 == 0:
                print(f"[{idx:04d}/{len(reader.tensors)}] copied")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output} with {q_count} {qtype.name} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
