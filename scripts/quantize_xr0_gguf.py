#!/usr/bin/env python3
"""Quantize the big matmul weights of a Xiaomi-Robotics-0 GGUF to a k-quant.

Quantizes vlm.blk / dit.blk attention and ffn weight matrices; keeps
token_embd (row-fetched at runtime), norms, biases and the small action
head MLPs at their original dtype for numerical fidelity.

Usage:
  python scripts/quantize_xr0_gguf.py \
    --input  checkpoints/xr0/xr0.gguf \
    --output checkpoints/xr0/xr0-q8_0.gguf \
    --outtype q8_0
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if GGUF_PY.exists():
    sys.path.insert(0, str(GGUF_PY))

try:
    import gguf
except Exception as exc:
    raise SystemExit(f"failed to import gguf from {GGUF_PY}: {exc}") from exc

QTYPES = {
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q6_k": gguf.GGMLQuantizationType.Q6_K,
    "q5_k": gguf.GGMLQuantizationType.Q5_K,
    "q4_k": gguf.GGMLQuantizationType.Q4_K,
}
# Keep in sync with scripts/gguf_quantize.py FILE_TYPE_BY_OUTTYPE; inlined
# here so this script does not import gguf_quantize (which pulls in torch).
FILE_TYPE_BY_QTYPE = {
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
    "q5_k": gguf.LlamaFileType.MOSTLY_Q5_K_M,
    "q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_M,
}

VLM_SUFFIXES = (
    ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_o.weight",
    ".ffn_gate.weight", ".ffn_up.weight", ".ffn_down.weight",
)
DIT_SUFFIXES = (
    ".attn_qkv.weight", ".attn_o.weight",
    ".ffn_gate.weight", ".ffn_up.weight", ".ffn_down.weight",
)


def should_quantize(name: str) -> bool:
    if name.startswith("vlm.blk."):
        return name.endswith(VLM_SUFFIXES)
    if name.startswith("dit.blk."):
        return name.endswith(DIT_SUFFIXES)
    return False


def bf16_bytes_to_f32(data: np.ndarray, gguf_shape: list[int]) -> np.ndarray:
    logical = tuple(reversed([int(v) for v in gguf_shape]))
    values = np.asarray(data, dtype=np.uint8).view(np.uint16).reshape(logical)
    return np.ascontiguousarray((values.astype(np.uint32) << np.uint32(16)).view(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--outtype", default="q8_0", choices=sorted(QTYPES))
    ap.add_argument("--ggml-lib", type=Path,
                    default=REPO_ROOT / "build" / "bin" / "libggml-base.so")
    args = ap.parse_args()

    qtype = QTYPES[args.outtype]
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]

    lib_path = args.ggml_lib.expanduser().resolve()
    if not lib_path.is_file():
        raise SystemExit(f"missing {lib_path}; build the runtime first or pass --ggml-lib")
    lib = ctypes.CDLL(str(lib_path))
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p,
    ]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t

    reader = gguf.GGUFReader(str(args.input))

    selected: set[str] = set()
    for tensor in reader.tensors:
        if not should_quantize(tensor.name):
            continue
        ne0 = int(tensor.shape[0])
        if len(tensor.shape) < 2 or ne0 % block_size != 0:
            print(f"  keep (shape unfit for {qtype.name}): {tensor.name}")
            continue
        if tensor.tensor_type not in (
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        ):
            print(f"  keep (source type {tensor.tensor_type.name}): {tensor.name}")
            continue
        selected.add(tensor.name)
    if not selected:
        raise SystemExit("no tensors matched the quantization policy")

    arch = "xr0"
    field = reader.fields.get("general.architecture")
    if field is not None:
        raw = field.contents() if callable(field.contents) else field.contents
        try:
            arch = bytes(raw).decode("utf-8", "replace") or "xr0"
        except Exception:
            arch = "xr0"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.output), arch=arch, use_temp_file=True)
    for key, f in reader.fields.items():
        if key.startswith("GGUF.") or key in {
            "general.architecture", "general.file_type", "xr0.weight_dtype",
        }:
            continue
        writer.add_key_value(key, f.contents(), f.types[0])
    writer.add_file_type(FILE_TYPE_BY_QTYPE[args.outtype])
    writer.add_string("xr0.weight_dtype", qtype.name)

    quantized_count = 0
    for index, tensor in enumerate(reader.tensors, start=1):
        if tensor.name in selected:
            if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
                f32 = bf16_bytes_to_f32(tensor.data, tensor.shape.tolist())
            elif tensor.tensor_type == gguf.GGMLQuantizationType.F32:
                f32 = np.ascontiguousarray(tensor.data.astype(np.float32, copy=False))
            elif tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                f32 = np.ascontiguousarray(tensor.data.astype(np.float32))
            else:
                raise SystemExit(f"cannot quantize {tensor.name}")
            f32 = np.ascontiguousarray(f32, dtype=np.float32)
            n_per_row = int(f32.shape[-1])
            nrows = int(f32.size // n_per_row)
            qbytes = nrows * (n_per_row // block_size) * type_size
            output = np.empty(qbytes, dtype=np.uint8)
            written = lib.ggml_quantize_chunk(
                int(qtype),
                f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output.ctypes.data_as(ctypes.c_void_p),
                0, nrows, n_per_row, None,
            )
            if written != qbytes:
                raise SystemExit(f"quantized byte mismatch for {tensor.name}")
            byte_shape = gguf.quant_shape_to_byte_shape(f32.shape, qtype)
            writer.add_tensor(tensor.name, output.reshape(byte_shape), raw_dtype=qtype)
            quantized_count += 1
            if quantized_count <= 3 or quantized_count % 25 == 0:
                print(f"[{index:04d}/{len(reader.tensors)}] {tensor.name} -> {qtype.name}")
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    in_gb = args.input.stat().st_size / 1e9
    out_gb = args.output.stat().st_size / 1e9
    print(f"quantized {quantized_count} tensors -> {qtype.name}")
    print(f"size: {in_gb:.2f} GB -> {out_gb:.2f} GB ({args.output})")


if __name__ == "__main__":
    main()
