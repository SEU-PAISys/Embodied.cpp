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

int lingbot_runtime_kv_update_f32(
    const float * k,
    const float * v,
    float * cache_k,
    float * cache_v,
    const int * slots,
    int seq,
    int n_heads,
    int head_dim,
    int capacity,
    cudaStream_t stream);

int lingbot_runtime_kv_attn_f32(
    const float * q,
    const float * cache_k,
    const float * cache_v,
    const int * valid_slots,
    float * out,
    int seq_q,
    int seq_k,
    int n_heads,
    int head_dim,
    float scale,
    cudaStream_t stream);

int lingbot_runtime_kv_attn_online_f32(
    const float * q,
    const float * cache_k,
    const float * cache_v,
    const int * valid_slots,
    float * out,
    int seq_q,
    int seq_k,
    int n_heads,
    int head_dim,
    float scale,
    cudaStream_t stream);

int lingbot_runtime_kv_attn_warp_f32(
    const float * q,
    const float * cache_k,
    const float * cache_v,
    const int * valid_slots,
    float * out,
    int seq_q,
    int seq_k,
    int n_heads,
    int head_dim,
    float scale,
    cudaStream_t stream);

int lingbot_runtime_kv_gather_f32(
    const float * cache_k,
    const float * cache_v,
    const int * valid_slots,
    float * compact_k,
    float * compact_v,
    int seq_k,
    int n_heads,
    int head_dim,
    int capacity,
    cudaStream_t stream);

int lingbot_scheduler_step_f32(
    float * sample,
    const float * model_output,
    int n,
    float delta,
    int bf16_round,
    cudaStream_t stream);

int lingbot_model_output_guidance_f32(
    float * cond_in_out,
    const float * uncond,
    int n,
    float scale,
    int use_guidance,
    int bf16_round,
    cudaStream_t stream);

int lingbot_action_finalize_f32(
    float * action_sample,
    const float * action_cond,
    int C,
    int F,
    int H,
    int W,
    int used_dim,
    int bf16_round,
    cudaStream_t stream);

int lingbot_latent_restore_round_f32(
    float * latent_sample,
    const float * latent_cond,
    int C,
    int F,
    int H,
    int W,
    int cond_F,
    int bf16_round,
    cudaStream_t stream);

int lingbot_patchify_latent_f32(
    const float * latent_bcfhw,
    float * tokens,
    int B,
    int C,
    int F,
    int H,
    int W,
    int pt,
    int ph,
    int pw,
    cudaStream_t stream);

int lingbot_projected_latent_to_tensor_f32(
    const float * projected_tokens,
    float * latent_bcfhw,
    int B,
    int C,
    int F,
    int H,
    int W,
    int pt,
    int ph,
    int pw,
    cudaStream_t stream);

int lingbot_action_tensor_to_tokens_f32(
    const float * action_bcfhw,
    float * tokens,
    int B,
    int C,
    int F,
    int H,
    int W,
    cudaStream_t stream);

int lingbot_action_tokens_to_tensor_f32(
    const float * tokens,
    float * action_bcfhw,
    int B,
    int C,
    int F,
    int H,
    int W,
    cudaStream_t stream);

int lingbot_action_sample_to_output_f32(
    const float * action_bcfhw,
    float * out,
    int B,
    int C,
    int F,
    int H,
    int W,
    int n_suffix,
    int output_dim,
    int postprocess_libero,
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
