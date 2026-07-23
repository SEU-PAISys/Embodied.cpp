// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Stable CUDA entry for Cosmos3 full_w8 packed linear layers.
//
// This ABI mirrors the GGUF representation produced by
// scripts/convert_cosmos3_full_w8_to_gguf.py:
//   qweight: int32 packed W8 payload from VllmGptqMarlinW8A16Linear
//   scales:  bf16 scale tensor
//
// models/cosmos3.cpp should call this symbol.  The implementation is currently
// a slow mapped-unpack correctness path using the proven vLLM byte-lane layout;
// once the tiled kernel lands, this symbol becomes the optimized path without
// changing model code.
int cosmos3_w8a16_linear_f32(
    const float * x,
    const int * qweight,
    const unsigned short * scales_bf16,
    float * y,
    int batch,
    int in_features,
    int out_features,
    int qweight_rows,
    int qweight_cols,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream);

// Same operation as cosmos3_w8a16_linear_f32, but uses caller-owned workspaces.
// This is intended for hot paths that call many W8 projections sequentially.
// Workspace sizes:
//   x_bf16_workspace: at least batch * in_features BF16 elements
//   y_bf16_workspace: at least batch * out_features BF16 elements
//   c_tmp_workspace:  at least sms * min(ceil(batch/16)*16, 64) * 256 float elements
//   int_workspace:    at least sms int elements
int cosmos3_w8a16_linear_f32_ws(
    const float * x,
    const int * qweight,
    const unsigned short * scales_bf16,
    float * y,
    unsigned short * x_bf16_workspace,
    unsigned short * y_bf16_workspace,
    float * c_tmp_workspace,
    int * int_workspace,
    int batch,
    int in_features,
    int out_features,
    int qweight_rows,
    int qweight_cols,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream);

int cosmos3_w8a16_f32_to_bf16_ws(
    const float * x,
    unsigned short * x_bf16_workspace,
    int count,
    cudaStream_t stream);

int cosmos3_w8a16_linear_bf16_ws(
    const unsigned short * x_bf16,
    const int * qweight,
    const unsigned short * scales_bf16,
    float * y,
    unsigned short * y_bf16_workspace,
    float * c_tmp_workspace,
    int * int_workspace,
    int batch,
    int in_features,
    int out_features,
    int qweight_rows,
    int qweight_cols,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream);

// Legacy reference CUDA entry.  This intentionally uses the old naive unpack
// assumption and is kept only for low-level transport diagnostics.
int cosmos3_w8a16_linear_f32_ref(
    const float * x,
    const int * qweight,
    const unsigned short * scales_bf16,
    float * y,
    int batch,
    int in_features,
    int out_features,
    int qweight_rows,
    int qweight_cols,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream);

// Benchmark/debug entry that bypasses the Marlin-first dispatch and forces the
// dequantize-to-F32 + cuBLAS fallback.  Do not use this in the model path.
int cosmos3_w8a16_linear_f32_cublas_debug(
    const float * x,
    const int * qweight,
    const unsigned short * scales_bf16,
    float * y,
    int batch,
    int in_features,
    int out_features,
    int qweight_rows,
    int qweight_cols,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream);

const char * cosmos3_w8a16_kernel_status(void);

int cosmos3_w8a16_cuda_smoke(cudaStream_t stream);

#ifdef __cplusplus
}
#endif
