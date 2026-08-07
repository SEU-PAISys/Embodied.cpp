# Cosmos3 CUDA Kernels

This directory owns Cosmos3-specific CUDA kernels for the native Embodied.cpp
runtime.

The current full-w8 GGUF stores official RoboLab quantized layers as:

```text
w.000000.q  int32 packed qweight
w.000000.s  bf16 scales
```

The source module metadata is embedded in GGUF:

```text
cosmos3.tensor_name_map_json
cosmos3.packed_w8_records_json
```

These weights come from PyTorch modules whose `backend_class` is usually:

```text
VllmGptqMarlinW8A16Linear
```

That is not GGML `Q8_0`.  Treating the tensor as a standard GGML quantized
matrix will produce wrong results.

## Implementation Plan

1. Keep a stable model-facing CUDA entry point:
   `cosmos3_w8a16_linear_f32`.
   The current implementation dequantizes the proven vLLM byte-lane layout to
   a temporary FP32 column-major matrix, runs cuBLAS GEMM with TF32 enabled, and
   rounds output values to BF16 precision.  Model code must call the stable
   symbol so a fused tiled kernel can replace this path without changing
   `models/cosmos3.cpp`.
2. Verify the exact qweight packing, scale broadcasting, group size, and any
   implicit permutation against PyTorch tensor captures.
3. Replace the reference unpack path with a tiled W8A16 kernel once the layout
   is proven.
4. Keep temporary validation code out of the public inference path after the
   module is numerically aligned.

The dequant-plus-GEMM kernel is deliberately not the final fast path.  It exists
so the Cosmos3 kernel namespace, launch ABI, and packing semantics can be
validated against vLLM Marlin captures before the optimized implementation
lands.
`cosmos3_w8a16_kernel_status()` reports
`cublas_dequant_f32_gemm_tf32_bf16_io_value_round_with_direct_mapped_scalar_fallback`
for this stage.

## W8 Repack Index

For one logical weight byte at source coordinate `(k, n)` in the pre-repack
`[K/4, N]` int32 matrix, where each int32 stores four K lanes, the official vLLM
W8 `gptq_marlin_repack` places it at:

```text
row  = k / 16
col  = (n / 64) * 256 + (n % 8) * 32 + ((n / 8) % 8) + ((k % 8) / 2) * 8
lane = (k % 2) * 2 + ((k / 8) % 2)
dst_byte = ((row * (N * 4) + col) * 4) + lane
```

`scripts/check_cosmos3_w8_repack_formula.py` validates this closed form against
byte-lane maps captured from `torch.ops._C.gptq_marlin_repack`.
