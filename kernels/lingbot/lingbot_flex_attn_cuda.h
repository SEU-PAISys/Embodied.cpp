// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

int lingbot_flex_attn_f32(
    const float * q,
    const float * k,
    const float * v,
    const int * row_ptr,
    const int * col_idx,
    float * out,
    int seq_q,
    int seq_k,
    int n_heads,
    int head_dim,
    int block_size,
    float scale,
    cudaStream_t stream);

int lingbot_flex_attn_f32_masked(
    const float * q,
    const float * k,
    const float * v,
    const int * row_ptr,
    const int * col_idx,
    const unsigned char * token_mask,
    float * out,
    int seq_q,
    int seq_k,
    int n_heads,
    int head_dim,
    int block_size,
    float scale,
    cudaStream_t stream);

int lingbot_flex_attn_cuda_smoke(cudaStream_t stream);

int lingbot_causal_conv1d_cache_f32(
    const float * x,
    const float * past,
    const float * weight,
    const float * bias,
    float * out,
    float * next_past,
    int T,
    int C_in,
    int C_out,
    int K,
    cudaStream_t stream);

int lingbot_causal_conv1d_cache_f32_batched(
    const float * x,
    const float * past,
    const float * weight,
    const float * bias,
    float * out,
    float * next_past,
    int lanes,
    int T,
    int C_in,
    int C_out,
    int K,
    cudaStream_t stream);

int lingbot_causal_conv1d_cache_f32_batched_stride(
    const float * x,
    const float * past,
    const float * weight,
    const float * bias,
    float * out,
    float * next_past,
    int lanes,
    int T,
    int T_out,
    int C_in,
    int C_out,
    int K,
    int stride,
    cudaStream_t stream);

int lingbot_causal_conv1d_cache_cuda_smoke(cudaStream_t stream);

int lingbot_causal_conv1d_cache_vae_smoke(cudaStream_t stream);

int lingbot_vae_mid_attn_f32(
    const float * in_whdc,
    const float * norm_gamma,
    const float * qkv_weight,
    const float * qkv_bias,
    const float * proj_weight,
    const float * proj_bias,
    float * out_whdc,
    int W,
    int H,
    int T,
    int C,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
