// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Cosmos3 DomainAwareLinear bridge used by the action pathway.
//
// The official tensors are stored as:
//   fc.weight   [domains, input_size * output_size] in PyTorch
//   bias.weight [domains, output_size]
//
// In GGUF/ggml metadata this appears as:
//   fc.weight   ne = [input_size * output_size, domains]
//   bias.weight ne = [output_size, domains]
//
// For a selected domain, the fc slice is reshaped as [input_size, output_size]
// and the operation is:
//   y = x @ weight + bias
int cosmos3_domain_aware_linear_bf16_f32(
    const float * x,
    const unsigned short * fc_weight_bf16,
    const unsigned short * bias_bf16,
    float * y,
    int batch,
    int input_size,
    int output_size,
    int domains,
    int domain_id,
    cudaStream_t stream);

// Same operation as cosmos3_domain_aware_linear_bf16_f32, but uses caller-owned
// BF16 workspace with at least batch * input_size elements.  This is the hot
// path for diffusion, where avoiding per-call cudaMallocAsync matters.
int cosmos3_domain_aware_linear_bf16_f32_ws(
    const float * x,
    const unsigned short * fc_weight_bf16,
    const unsigned short * bias_bf16,
    float * y,
    unsigned short * x_bf16_workspace,
    int batch,
    int input_size,
    int output_size,
    int domains,
    int domain_id,
    cudaStream_t stream);

int cosmos3_action2llm_plus_embed_f32(
    const float * action_latent,
    const unsigned short * action2llm_fc_bf16,
    const unsigned short * action2llm_bias_bf16,
    const unsigned short * action_modality_embed_bf16,
    float * llm_hidden,
    int batch,
    int domains,
    int domain_id,
    cudaStream_t stream);

int cosmos3_action2llm_plus_embed_f32_ws(
    const float * action_latent,
    const unsigned short * action2llm_fc_bf16,
    const unsigned short * action2llm_bias_bf16,
    const unsigned short * action_modality_embed_bf16,
    float * llm_hidden,
    unsigned short * action_bf16_workspace,
    int batch,
    int domains,
    int domain_id,
    cudaStream_t stream);

// Decode MoT action hidden states with the official llm2action
// DomainAwareLinear and keep only the real action dimensions used by RoboLab.
// The full 64-dim domain output is written to full_action_workspace; the
// returned action_out is row-major [batch, real_action_dim].
int cosmos3_llm2action_slice_f32_ws(
    const float * llm_hidden,
    const unsigned short * llm2action_fc_bf16,
    const unsigned short * llm2action_bias_bf16,
    float * full_action_workspace,
    float * action_out,
    unsigned short * llm_bf16_workspace,
    int batch,
    int hidden,
    int max_action_dim,
    int real_action_dim,
    int domains,
    int domain_id,
    cudaStream_t stream);

const char * cosmos3_action_bridge_kernel_status(void);

#ifdef __cplusplus
}
#endif
