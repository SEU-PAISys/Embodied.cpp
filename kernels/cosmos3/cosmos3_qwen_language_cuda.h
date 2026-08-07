// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

int cosmos3_qwen_rmsnorm_f32(
    const float * x,
    const unsigned short * gamma_bf16,
    float * y,
    int tokens,
    int hidden,
    float eps,
    cudaStream_t stream);

int cosmos3_qwen_head_rmsnorm_f32(
    const float * x,
    const unsigned short * gamma_bf16,
    float * y,
    int tokens,
    int width,
    int heads,
    int head_dim,
    float eps,
    cudaStream_t stream);

int cosmos3_qwen_rope_f32(
    const float * x,
    float * y,
    int tokens,
    int width,
    int heads,
    int head_dim,
    float rope_theta,
    int first_position,
    cudaStream_t stream);

int cosmos3_qwen_mrope_f32(
    const float * x,
    const int * position_ids,
    float * y,
    int tokens,
    int width,
    int heads,
    int head_dim,
    float rope_theta,
    cudaStream_t stream);

int cosmos3_qwen_mrope_pos_f32(
    const float * x,
    const float * position_ids,
    float * y,
    int tokens,
    int width,
    int heads,
    int head_dim,
    float rope_theta,
    cudaStream_t stream);

int cosmos3_qwen_head_rmsnorm_mrope_pos_f32(
    const float * x,
    const unsigned short * gamma_bf16,
    const float * position_ids,
    float * y,
    int tokens,
    int width,
    int heads,
    int head_dim,
    float eps,
    float rope_theta,
    cudaStream_t stream);

int cosmos3_qwen_causal_gqa_attention_f32(
    const float * q,
    const float * k,
    const float * v,
    float * y,
    int tokens,
    int hidden,
    int kv,
    int q_heads,
    int kv_heads,
    int head_dim,
    cudaStream_t stream);

int cosmos3_qwen_gen_full_gqa_attention_f32(
    const float * q_gen,
    const float * k_all,
    const float * v_all,
    float * y_gen,
    int gen_tokens,
    int all_tokens,
    int hidden,
    int kv,
    int q_heads,
    int kv_heads,
    int head_dim,
    cudaStream_t stream);

int cosmos3_qwen_gen_full_gqa_attention_workspace_f32(
    const float * q_gen,
    const float * k_all,
    const float * v_all,
    float * y_gen,
    float * scores_workspace,
    int gen_tokens,
    int all_tokens,
    int hidden,
    int kv,
    int q_heads,
    int kv_heads,
    int head_dim,
    cudaStream_t stream);

int cosmos3_qwen_gen_full_gqa_attention_workspace_bf16tc_f32(
    const float * q_gen,
    const float * k_all,
    const float * v_all,
    float * y_gen,
    float * scores_workspace,
    unsigned short * bf16_workspace,
    int gen_tokens,
    int all_tokens,
    int hidden,
    int kv,
    int q_heads,
    int kv_heads,
    int head_dim,
    cudaStream_t stream);

int cosmos3_qwen_gen_full_gqa_attention_f32_online_debug(
    const float * q_gen,
    const float * k_all,
    const float * v_all,
    float * y_gen,
    int gen_tokens,
    int all_tokens,
    int hidden,
    int kv,
    int q_heads,
    int kv_heads,
    int head_dim,
    cudaStream_t stream);

int cosmos3_qwen_gen_full_gqa_attention_f32_cublas_debug(
    const float * q_gen,
    const float * k_all,
    const float * v_all,
    float * y_gen,
    int gen_tokens,
    int all_tokens,
    int hidden,
    int kv,
    int q_heads,
    int kv_heads,
    int head_dim,
    cudaStream_t stream);

int cosmos3_qwen_swiglu_f32(
    const float * gate,
    const float * up,
    float * y,
    int count,
    cudaStream_t stream);

int cosmos3_qwen_residual_add_f32(
    const float * a,
    const float * b,
    float * y,
    int count,
    cudaStream_t stream);

int cosmos3_qwen_residual_add_rmsnorm_f32(
    const float * residual_input,
    const float * branch,
    const unsigned short * gamma_bf16,
    float * residual,
    float * norm,
    int tokens,
    int hidden,
    float eps,
    cudaStream_t stream);

int cosmos3_qwen_build_robolab_input_f32(
    const int * input_ids,
    const int * visual_indices,
    const unsigned short * embed_tokens_bf16,
    const float * visual_tokens,
    float * y,
    int tokens,
    int hidden,
    int vocab,
    int visual_token_count,
    cudaStream_t stream);

int cosmos3_qwen_embed_tokens_f32(
    const int * input_ids,
    const unsigned short * embed_tokens_bf16,
    float * y,
    int tokens,
    int hidden,
    int vocab,
    cudaStream_t stream);

int cosmos3_qwen_add_visual_deepstack_f32(
    float * hidden_states,
    const int * visual_indices,
    const float * deepstack_tokens,
    int tokens,
    int hidden,
    int visual_token_count,
    cudaStream_t stream);

const char * cosmos3_qwen_language_kernel_status(void);

#ifdef __cplusplus
}
#endif
