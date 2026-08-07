"""Shared helpers for quantizing safetensors weights while writing GGUF."""

from __future__ import annotations

import argparse
import ctypes
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"
if GGUF_PY.exists():
    sys.path.insert(0, str(GGUF_PY))

try:
    import gguf
except Exception as exc:
    raise SystemExit(f"failed to import gguf from {GGUF_PY}: {exc}") from exc


DEFAULT_GGML_LIB = REPO_ROOT / "build" / "bin" / "libggml-base.so"

QTYPE_BY_OUTTYPE = {
    "q2_k": gguf.GGMLQuantizationType.Q2_K,
    "q3_k": gguf.GGMLQuantizationType.Q3_K,
    "q4_0": gguf.GGMLQuantizationType.Q4_0,
    "q4_k": gguf.GGMLQuantizationType.Q4_K,
    "q5_0": gguf.GGMLQuantizationType.Q5_0,
    "q5_k": gguf.GGMLQuantizationType.Q5_K,
    "q6_k": gguf.GGMLQuantizationType.Q6_K,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
}
OUTTYPE_CHOICES = ("bf16", *QTYPE_BY_OUTTYPE)

FILE_TYPE_BY_OUTTYPE = {
    "bf16": gguf.LlamaFileType.MOSTLY_BF16,
    "q2_k": gguf.LlamaFileType.MOSTLY_Q2_K,
    "q3_k": gguf.LlamaFileType.MOSTLY_Q3_K_M,
    "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    "q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_M,
    "q5_0": gguf.LlamaFileType.MOSTLY_Q5_0,
    "q5_k": gguf.LlamaFileType.MOSTLY_Q5_K_M,
    "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
}


def parse_outtype(value: str) -> str:
    outtype = value.lower()
    if outtype not in OUTTYPE_CHOICES:
        raise argparse.ArgumentTypeError(
            f"unsupported output type {value!r}; choose from {', '.join(OUTTYPE_CHOICES)}"
        )
    return outtype


def add_outtype_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--outtype",
        "--qtype",
        dest="outtype",
        type=parse_outtype,
        default="bf16",
        metavar="TYPE",
        help=f"weight output type (case-insensitive): {', '.join(OUTTYPE_CHOICES)} (default: bf16)",
    )
    parser.add_argument(
        "--ggml-lib",
        type=Path,
        default=DEFAULT_GGML_LIB,
        help="libggml-base shared library used for quantized output",
    )


def _bf16_to_u16(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    return tensor.contiguous().view(torch.uint16).cpu().numpy()


class TensorQuantizer:
    def __init__(self, outtype: str, ggml_lib: Path) -> None:
        self.outtype = parse_outtype(outtype)
        self.qtype = QTYPE_BY_OUTTYPE.get(self.outtype)
        self.quantized_count = 0
        self.skipped: list[tuple[str, list[int], int]] = []
        self.lib: ctypes.CDLL | None = None

        if self.qtype is not None:
            lib_path = ggml_lib.expanduser().resolve()
            if not lib_path.is_file():
                raise SystemExit(
                    f"missing {lib_path}; build libggml-base first or pass --ggml-lib"
                )
            self.lib = ctypes.CDLL(str(lib_path))
            self.lib.ggml_quantize_chunk.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
            ]
            self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t

    @property
    def file_type(self) -> gguf.LlamaFileType:
        return FILE_TYPE_BY_OUTTYPE[self.outtype]

    def _quantize_array(self, name: str, data: np.ndarray) -> np.ndarray | None:
        assert self.qtype is not None
        assert self.lib is not None

        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim < 2:
            return None
        n_per_row = int(data.shape[-1])
        block_size, type_size = gguf.GGML_QUANT_SIZES[self.qtype]
        if n_per_row % block_size != 0:
            self.skipped.append((name, list(data.shape), block_size))
            return None

        nrows = int(data.size // n_per_row)
        qbytes = nrows * (n_per_row // block_size) * type_size
        output = np.empty(qbytes, dtype=np.uint8)
        written = self.lib.ggml_quantize_chunk(
            int(self.qtype),
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.c_void_p),
            0,
            nrows,
            n_per_row,
            None,
        )
        if written != qbytes:
            raise RuntimeError(f"quantized byte mismatch for {name}: {written} vs {qbytes}")
        byte_shape = gguf.quant_shape_to_byte_shape(data.shape, self.qtype)
        return output.reshape(byte_shape)

    def _quantize(self, name: str, tensor: torch.Tensor) -> np.ndarray | None:
        data = tensor.to(torch.float32).contiguous().cpu().numpy()
        return self._quantize_array(name, data)

    def add_tensor(
        self,
        writer: gguf.GGUFWriter,
        name: str,
        tensor: torch.Tensor,
        *,
        quantize: bool,
    ) -> None:
        if quantize and self.qtype is not None:
            quantized = self._quantize(name, tensor)
            if quantized is not None:
                writer.add_tensor(
                    name,
                    quantized,
                    raw_dtype=self.qtype,
                )
                self.quantized_count += 1
                return

        if tensor.dtype == torch.float32:
            writer.add_tensor(
                name,
                tensor.contiguous().cpu().numpy(),
                raw_dtype=gguf.GGMLQuantizationType.F32,
            )
        elif tensor.dtype == torch.float16:
            writer.add_tensor(
                name,
                tensor.contiguous().cpu().numpy(),
                raw_dtype=gguf.GGMLQuantizationType.F16,
            )
        elif tensor.dtype == torch.bfloat16:
            writer.add_tensor(
                name,
                _bf16_to_u16(tensor),
                raw_shape=list(tensor.shape),
                raw_dtype=gguf.GGMLQuantizationType.BF16,
            )
        else:
            raise NotImplementedError(f"unsupported dtype {tensor.dtype} for {name}")

    def finish(self) -> None:
        if self.qtype is None:
            print("quantization: disabled (bf16 output)")
            return
        if self.quantized_count == 0:
            raise SystemExit(
                f"no tensors were quantized to {self.qtype.name}; selected matrix dimensions "
                "are incompatible with its block size"
            )
        print(f"quantization: wrote {self.quantized_count} {self.qtype.name} tensors")
        for name, shape, block_size in self.skipped:
            print(
                f"  kept original dtype: {name} shape={shape} "
                f"(last dimension is not divisible by block size {block_size})"
            )


def _reader_tensor_to_f32(tensor: gguf.ReaderTensor) -> np.ndarray:
    if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
        return np.ascontiguousarray(tensor.data.astype(np.float32, copy=False))
    if tensor.tensor_type == gguf.GGMLQuantizationType.F16:
        return np.ascontiguousarray(tensor.data.astype(np.float32))
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        logical_shape = tuple(reversed([int(value) for value in tensor.shape.tolist()]))
        values = np.asarray(tensor.data, dtype=np.uint8).view(np.uint16)
        values = values.reshape(logical_shape)
        return np.ascontiguousarray((values.astype(np.uint32) << np.uint32(16)).view(np.float32))
    raise ValueError(
        f"cannot quantize {tensor.name}: unsupported source type {tensor.tensor_type.name}"
    )


def quantize_gguf(
    input_path: Path,
    output_path: Path,
    outtype: str,
    ggml_lib: Path,
    *,
    arch: str,
    should_quantize: Callable[[gguf.ReaderTensor], bool],
    metadata_key: str | None = None,
) -> int:
    """Rewrite selected F32/F16/BF16 GGUF tensors using the requested qtype."""
    quantizer = TensorQuantizer(outtype, ggml_lib)
    if quantizer.qtype is None:
        raise ValueError("quantize_gguf requires a quantized --outtype")

    reader = gguf.GGUFReader(str(input_path))
    selected: set[str] = set()
    for tensor in reader.tensors:
        if not should_quantize(tensor):
            continue
        ne0 = int(tensor.shape[0])
        block_size, _ = gguf.GGML_QUANT_SIZES[quantizer.qtype]
        if len(tensor.shape) < 2 or ne0 % block_size != 0:
            quantizer.skipped.append(
                (tensor.name, list(reversed(tensor.shape.tolist())), block_size)
            )
            continue
        if tensor.tensor_type not in (
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        ):
            raise ValueError(
                f"cannot quantize {tensor.name} from {tensor.tensor_type.name}"
            )
        selected.add(tensor.name)

    if not selected:
        raise ValueError(f"no tensors in {input_path} matched the quantization policy")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous_tempdir = tempfile.tempdir
    tempfile.tempdir = str(output_path.parent)
    try:
        writer = gguf.GGUFWriter(str(output_path), arch=arch, use_temp_file=True)
        for key, field in reader.fields.items():
            if key.startswith("GGUF.") or key in {
                "general.architecture",
                "general.file_type",
                metadata_key,
            }:
                continue
            writer.add_key_value(key, field.contents(), field.types[0])
        writer.add_file_type(quantizer.file_type)
        if metadata_key:
            writer.add_string(metadata_key, outtype.upper())

        for index, tensor in enumerate(reader.tensors, start=1):
            if tensor.name in selected:
                data = _reader_tensor_to_f32(tensor)
                quantized = quantizer._quantize_array(tensor.name, data)
                if quantized is None:
                    raise RuntimeError(f"failed to quantize selected tensor {tensor.name}")
                writer.add_tensor(tensor.name, quantized, raw_dtype=quantizer.qtype)
                quantizer.quantized_count += 1
                if quantizer.quantized_count <= 5 or quantizer.quantized_count % 25 == 0:
                    print(
                        f"[{index:04d}/{len(reader.tensors)}] "
                        f"{tensor.name} -> {quantizer.qtype.name}"
                    )
            else:
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)

        quantizer.finish()
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    finally:
        tempfile.tempdir = previous_tempdir
    return quantizer.quantized_count