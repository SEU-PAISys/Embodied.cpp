#!/usr/bin/env python3
"""Create Q8_0/Q4_0/Q6_K TurboVLA or X-VLA storage-quantized files.

Q8_0/Q4_0 use the NumPy codec; Q6_K uses the shared GGML quantizer.

This is storage quantization: both runtimes dequantize these files to their
configured resident type (BF16 by default). It is not native low-bit compute.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
sys.path.insert(0, str(ROOT / "scripts"))
import gguf

QTYPES = {"q8_0": gguf.GGMLQuantizationType.Q8_0, "q4_0": gguf.GGMLQuantizationType.Q4_0,
          "q6_k": gguf.GGMLQuantizationType.Q6_K}
FILE_TYPES = {"q8_0": gguf.LlamaFileType.MOSTLY_Q8_0, "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
              "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K}
EXCLUDE = ("token_emb", "pos_emb", "norm", ".bias", ".emb", "conv", "patch_embed",
           "lm_head", "head.", "action_head", "proj_in", "proj_out", "wte", "pe", "text.proj")


def quantize_file(source: Path, output: Path, outtype: str, min_rows: int = 128,
                  ggml_lib: Path | None = None) -> int:
    if min_rows < 1:
        raise ValueError("min_rows must be positive")
    source, output = Path(source), Path(output)
    if output.exists() or output.is_symlink() or source.resolve() == output.resolve():
        raise FileExistsError(f"refusing to overwrite {output}")
    qtype = QTYPES[outtype]
    block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
    reader = gguf.GGUFReader(str(source))
    arch = reader.fields["general.architecture"].contents()
    if arch not in ("turbovla", "xvla"):
        raise ValueError(f"unsupported architecture: {arch}")
    selected = set()
    for tensor in reader.tensors:
        if tensor.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.BF16):
            raise ValueError(f"expected original F32/BF16 input, got {tensor.tensor_type.name}: {tensor.name}")
        if (tensor.name.endswith((".w", ".weight")) and not any(x in tensor.name for x in EXCLUDE)
                and len(tensor.shape) == 2 and int(tensor.shape[0]) % block_size == 0
                and int(tensor.shape[1]) >= min_rows):
            selected.add(tensor.name)
    if not selected:
        raise ValueError("no tensors selected")
    quantizer = None
    if outtype == "q6_k":
        # Keep Q8/Q4 usable without torch or a compiled shared library.
        from gguf_quantize import TensorQuantizer
        quantizer = TensorQuantizer(outtype, ggml_lib or ROOT / "build/bin/libggml-base.so")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build beside the destination and publish with an exclusive hard link:
    # even another writer racing this conversion cannot be overwritten.
    with tempfile.TemporaryDirectory(prefix=".vla-quant-", dir=output.parent) as scratch:
        prepared = Path(scratch) / "prepared.gguf"
        writer = gguf.GGUFWriter(str(prepared), arch=arch, use_temp_file=True)
        try:
            for key, field in reader.fields.items():
                if key.startswith("GGUF.") or key in ("general.architecture", "general.file_type"):
                    continue
                subtype = field.types[1] if len(field.types) > 1 else None
                writer.add_key_value(key, field.contents(), field.types[0], subtype)
            writer.add_file_type(FILE_TYPES[outtype])
            for tensor in reader.tensors:
                if tensor.name in selected:
                    values = gguf.dequantize(tensor.data, tensor.tensor_type)
                    if not np.isfinite(values).all():
                        raise ValueError(f"non-finite source tensor: {tensor.name}")
                    quantized = (quantizer._quantize_array(tensor.name, values)
                                 if quantizer is not None else gguf.quantize(values, qtype))
                    if quantized is None:
                        raise ValueError(f"cannot quantize selected tensor: {tensor.name}")
                    writer.add_tensor(tensor.name, quantized, raw_dtype=qtype)
                else:
                    writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
            if writer.temp_file is not None:
                writer.temp_file.close()
        os.link(prepared, output)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outtype", choices=QTYPES, default="q8_0")
    parser.add_argument("--min-rows", type=int, default=128)
    parser.add_argument("--ggml-lib", type=Path, default=ROOT / "build/bin/libggml-base.so",
                        help="Q6_K only: shared library used by the existing GGML quantizer")
    args = parser.parse_args()
    count = quantize_file(args.input, args.output, args.outtype, args.min_rows, args.ggml_lib)
    print(f"Quantized {count} tensors: {args.output} ({args.output.stat().st_size} bytes)")
    print("Storage quantization only; default runtime residency remains BF16.")


if __name__ == "__main__":
    main()
