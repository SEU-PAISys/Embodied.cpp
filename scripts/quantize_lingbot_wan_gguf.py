#!/usr/bin/env python3
"""Offline-quantize LingBot-VA Wan transformer block weights in a GGUF file.

This keeps metadata and non-Wan-block tensors in their original dtype, and
quantizes only the matmul weights used by the resident Wan block path:

  wvm.blk.N.{self_attn,cross_attn}.{q,k,v,o}.weight
  wvm.blk.N.{ffn_up,ffn_down}.weight

The quantization itself calls ggml_quantize_chunk from the local llama.cpp
build, so the byte layout matches the C++ runtime quantization path.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if GGUF_PY.exists():
    sys.path.insert(0, str(GGUF_PY))

try:
    import gguf
except Exception as exc:
    raise SystemExit(
        "failed to import gguf. Expected vendored gguf-py under "
        f"{GGUF_PY}. Original error: {exc}"
    )


WAN_MATMUL_RE = re.compile(
    r"^wvm\.blk\.\d+\.(?:(?:self_attn|cross_attn)\.(?:q|k|v|o)|ffn_(?:up|down))\.weight$"
)

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


def _load_ggml(lib_path: Path) -> ctypes.CDLL:
    if not lib_path.exists():
        raise SystemExit(
            f"missing {lib_path}; build first, e.g. cmake --build build --target lingbot-world-server"
        )
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
    # GGUFReader exposes BF16 tensors as uint8 with byte-shaped rows.  Rebuild
    # the logical row-major [ne1, ne0] matrix before passing it to ggml.
    u16 = np.asarray(tensor.data, dtype=np.uint8).view(np.uint16)
    logical_shape = tuple(reversed([int(x) for x in tensor.shape.tolist()]))
    u16 = u16.reshape(logical_shape)
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return np.ascontiguousarray(u32.view(np.float32))


def _f32_tensor(tensor: "gguf.ReaderTensor") -> np.ndarray:
    if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
        return np.ascontiguousarray(tensor.data.astype(np.float32, copy=False))
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        return _bf16_bytes_to_f32(tensor)
    raise ValueError(f"cannot quantize {tensor.name}: unsupported source dtype {tensor.tensor_type.name}")


def _quantize_tensor(
    lib: ctypes.CDLL,
    tensor: "gguf.ReaderTensor",
    qtype: "gguf.GGMLQuantizationType",
) -> np.ndarray:
    data = _f32_tensor(tensor)
    ne = [int(x) for x in tensor.shape.tolist()]
    n_per_row = ne[0]
    nrows = int(np.prod(ne) // n_per_row)
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    if n_per_row % block_size != 0:
        raise ValueError(
            f"cannot quantize {tensor.name}: ne0={n_per_row} is not divisible by {block_size}"
        )
    qbytes = nrows * (n_per_row // block_size) * type_size
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


def _copy_metadata(reader: "gguf.GGUFReader", writer: "gguf.GGUFWriter") -> None:
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key == "general.architecture":
            continue  # GGUFWriter adds this from arch.
        writer.add_key_value(key, field.contents(), field.types[0])
    writer.add_string("lingbot_va.quantized_by", "scripts/quantize_lingbot_wan_gguf.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qtype", default="q4_K", choices=sorted(QTYPE_BY_NAME))
    parser.add_argument("--ggml-lib", type=Path, default=REPO_ROOT / "build" / "bin" / "libggml-base.so")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-quantized-tensors",
        type=int,
        default=0,
        help="debug limit; 0 means quantize all matching Wan matmul tensors",
    )
    args = parser.parse_args()

    qtype = QTYPE_BY_NAME[args.qtype]
    lib = _load_ggml(args.ggml_lib)
    reader = gguf.GGUFReader(str(args.input))

    quantizable = [
        t for t in reader.tensors
        if WAN_MATMUL_RE.match(t.name)
    ]
    if args.max_quantized_tensors > 0:
        quantizable = quantizable[: args.max_quantized_tensors]
    quantizable_names = {t.name for t in quantizable}

    old_total = sum(int(t.n_bytes) for t in reader.tensors)
    new_total = 0
    for t in reader.tensors:
        if t.name in quantizable_names:
            ne = [int(x) for x in t.shape.tolist()]
            block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
            new_total += int(np.prod(ne) * type_size // block_size)
        else:
            new_total += int(t.n_bytes)

    print(
        f"input={args.input}\n"
        f"output={args.output}\n"
        f"qtype={qtype.name} quantized_tensors={len(quantizable_names)} / {len(reader.tensors)}\n"
        f"tensor_bytes: {old_total / 1024**3:.2f} GiB -> {new_total / 1024**3:.2f} GiB"
    )
    if args.dry_run:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(args.output.parent)
    writer = gguf.GGUFWriter(str(args.output), arch="lingbot_va", use_temp_file=True)
    _copy_metadata(reader, writer)

    q_count = 0
    for idx, tensor in enumerate(reader.tensors, start=1):
        if tensor.name in quantizable_names:
            arr = _quantize_tensor(lib, tensor, qtype)
            writer.add_tensor(tensor.name, arr, raw_dtype=qtype)
            q_count += 1
            print(f"[{idx:03d}/{len(reader.tensors)}] quantized {tensor.name} -> {qtype.name}")
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            if idx % 100 == 0:
                print(f"[{idx:03d}/{len(reader.tensors)}] copied")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output} with {q_count} {qtype.name} Wan matmul tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
