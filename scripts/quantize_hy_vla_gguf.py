#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline-quantize HY-VLA GGUF matmul weights.

The HY-VLA runtime can use quantized ggml matmul tensors, but normalization,
biases, statistics, image position embeddings, and small projections should stay
in their original dtype.  This script quantizes only large Q4_K-compatible
matrices and copies everything else unchanged.
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

PATTERNS = {
    "expert": re.compile(
        r"^expert\.blk\.\d+\.(?:attn_[qkvo]_v|ffn_(?:gate|up|down)_v)\.weight$"
    ),
    "vlm": re.compile(
        r"^vlm\.blk\.\d+\.(?:attn_[qkvo](?:_v)?|ffn_(?:gate|up|down)(?:_v)?)\.weight$"
    ),
    "action": re.compile(
        r"^(?:action_in_proj|action_time_mlp_in|action_time_mlp_out)\.weight$"
    ),
    "vision": re.compile(
        r"^vision\.(?:merger\.(?:proj2|pooler\.fc0|pooler\.fc2))\.weight$"
    ),
}


def _load_ggml(lib_path: Path) -> ctypes.CDLL:
    if not lib_path.exists():
        raise SystemExit(f"missing {lib_path}; build first")
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
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        return _bf16_bytes_to_f32(tensor)
    raise ValueError(f"unsupported source dtype for {tensor.name}: {tensor.tensor_type.name}")


def _category(name: str) -> str | None:
    for cat, pattern in PATTERNS.items():
        if pattern.match(name):
            return cat
    return None


def _can_quantize(tensor: "gguf.ReaderTensor", qtype: "gguf.GGMLQuantizationType") -> bool:
    if tensor.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16):
        return False
    ne = [int(x) for x in tensor.shape.tolist()]
    if len(ne) < 2:
        return False
    block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
    return ne[0] % block_size == 0


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


def _copy_metadata(reader: "gguf.GGUFReader", writer: "gguf.GGUFWriter", qtype: str) -> None:
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        writer.add_key_value(key, field.contents(), field.types[0])
    writer.add_string("hy_vla.quantized_by", "scripts/quantize_hy_vla_gguf.py")
    writer.add_string("hy_vla.quantization", qtype)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qtype", default="q4_K", choices=sorted(QTYPE_BY_NAME))
    parser.add_argument("--ggml-lib", type=Path, default=REPO_ROOT / "build" / "bin" / "libggml-base.so")
    parser.add_argument("--include", choices=sorted(PATTERNS), action="append",
                        help="category to quantize; default: expert+vlm+action+vision")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    qtype = QTYPE_BY_NAME[args.qtype]
    include = set(args.include or PATTERNS.keys())
    lib = _load_ggml(args.ggml_lib)
    reader = gguf.GGUFReader(str(args.input))

    quant_names: set[str] = set()
    counts: dict[str, int] = {k: 0 for k in PATTERNS}
    old_total = sum(int(t.n_bytes) for t in reader.tensors)
    new_total = 0
    skipped = []
    for tensor in reader.tensors:
        cat = _category(tensor.name)
        if cat in include and _can_quantize(tensor, qtype):
            quant_names.add(tensor.name)
            counts[cat] += 1
            ne = [int(x) for x in tensor.shape.tolist()]
            block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
            new_total += int(np.prod(ne) * type_size // block_size)
        else:
            new_total += int(tensor.n_bytes)
            if cat in include:
                skipped.append((cat, tensor.name, [int(x) for x in tensor.shape.tolist()]))

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"qtype={qtype.name}")
    print(f"include={','.join(sorted(include))}")
    print(f"quantized_tensors={len(quant_names)} / {len(reader.tensors)} counts={counts}")
    print(f"tensor_bytes: {old_total / 1024**3:.2f} GiB -> {new_total / 1024**3:.2f} GiB")
    if skipped:
        print(f"skipped_matching_tensors={len(skipped)}")
        for cat, name, shape in skipped[:40]:
            print(f"  skip {cat}: {name} shape={shape}")
    if args.dry_run:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(args.output.parent)
    writer = gguf.GGUFWriter(str(args.output), arch="hy_vla", use_temp_file=True)
    _copy_metadata(reader, writer, args.qtype)

    q_count = 0
    for idx, tensor in enumerate(reader.tensors, start=1):
        if tensor.name in quant_names:
            arr = _quantize_tensor(lib, tensor, qtype)
            writer.add_tensor(tensor.name, arr, raw_dtype=qtype)
            q_count += 1
            if q_count % 25 == 0 or q_count <= 5:
                print(f"[{idx:04d}/{len(reader.tensors)}] quantized {tensor.name} -> {qtype.name}")
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            if idx % 200 == 0:
                print(f"[{idx:04d}/{len(reader.tensors)}] copied")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output} with {q_count} {qtype.name} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
