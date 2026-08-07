// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Build the official Cosmos3 action input tokens for Unified MoT:
//
//   t = timestep * timestep_scale
//   t_freq = sinusoidal_embedding(t, 256)
//   t_emb = Linear(SiLU(Linear(t_freq)))
//   action_hidden = action2llm(action_latents, domain_id) + action_modality_embed
//   action_hidden[condition_action_tokens:] += t_emb
//
// This kernel group covers the timestep/action-token input path.  The actual
// transformer/MoT denoising loop consumes the returned [action_tokens, 4096]
// hidden states in the next stage.
int cosmos3_mot_build_action_input_f32(
    const float * action_latents,
    const unsigned short * action2llm_fc_bf16,
    const unsigned short * action2llm_bias_bf16,
    const unsigned short * action_modality_embed_bf16,
    const unsigned short * time_mlp0_weight_bf16,
    const unsigned short * time_mlp0_bias_bf16,
    const unsigned short * time_mlp2_weight_bf16,
    const unsigned short * time_mlp2_bias_bf16,
    float * action_hidden,
    float * timestep_hidden,
    float timestep,
    float timestep_scale,
    int action_tokens,
    int max_action_dim,
    int hidden,
    int time_freq_dim,
    int domains,
    int domain_id,
    int condition_action_tokens,
    cudaStream_t stream);

// Variant with caller-owned workspaces:
//   action_bf16_workspace: action_tokens * max_action_dim
//   time_freq_bf16_workspace: time_freq_dim
//   time_mlp0_workspace: hidden
//   time_mlp0_bf16_workspace: hidden
int cosmos3_mot_build_action_input_f32_ws(
    const float * action_latents,
    const unsigned short * action2llm_fc_bf16,
    const unsigned short * action2llm_bias_bf16,
    const unsigned short * action_modality_embed_bf16,
    const unsigned short * time_mlp0_weight_bf16,
    const unsigned short * time_mlp0_bias_bf16,
    const unsigned short * time_mlp2_weight_bf16,
    const unsigned short * time_mlp2_bias_bf16,
    float * action_hidden,
    float * timestep_hidden,
    unsigned short * action_bf16_workspace,
    unsigned short * time_freq_bf16_workspace,
    float * time_mlp0_workspace,
    unsigned short * time_mlp0_bf16_workspace,
    float timestep,
    float timestep_scale,
    int action_tokens,
    int max_action_dim,
    int hidden,
    int time_freq_dim,
    int domains,
    int domain_id,
    int condition_action_tokens,
    cudaStream_t stream);

int cosmos3_mot_pack_condition_action_f32(
    const float * condition_hidden,
    const float * action_hidden,
    float * packed_hidden,
    int condition_tokens,
    int action_tokens,
    int hidden,
    cudaStream_t stream);

int cosmos3_mot_vae2llm_f32_ws(
    const float * vision_latents,
    const unsigned short * vae2llm_weight_bf16,
    const unsigned short * vae2llm_bias_bf16,
    float * vision_hidden,
    unsigned short * vision_bf16_workspace,
    int vision_tokens,
    int patch_dim,
    int hidden,
    cudaStream_t stream);

int cosmos3_mot_add_timestep_to_noisy_tokens_f32(
    float * tokens,
    const float * timestep_hidden,
    int tokens_count,
    int condition_tokens,
    int hidden,
    cudaStream_t stream);

int cosmos3_mot_llm2vae_f32_ws(
    const float * vision_hidden,
    const unsigned short * llm2vae_weight_bf16,
    const unsigned short * llm2vae_bias_bf16,
    float * vision_velocity,
    unsigned short * vision_hidden_bf16_workspace,
    int vision_tokens,
    int hidden,
    int patch_dim,
    cudaStream_t stream);

int cosmos3_mot_pack_text_vision_action_f32(
    const float * text_hidden,
    const float * vision_hidden,
    const float * action_hidden,
    float * packed_hidden,
    int text_tokens,
    int vision_tokens,
    int action_tokens,
    int hidden,
    cudaStream_t stream);

int cosmos3_mot_update_action_latents_f32(
    float * action_latents,
    const float * velocity,
    int action_tokens,
    int max_action_dim,
    int velocity_dim,
    int condition_action_tokens,
    float step,
    cudaStream_t stream);

int cosmos3_mot_compute_x0_from_flow_f32(
    const float * sample,
    const float * velocity,
    float * x0,
    int count,
    float sigma,
    cudaStream_t stream);

int cosmos3_mot_mask_action_velocity_f32(
    float * velocity,
    int action_tokens,
    int max_action_dim,
    int real_action_dim,
    int condition_action_tokens,
    cudaStream_t stream);

int cosmos3_mot_copy_f32(
    const float * src,
    float * dst,
    int count,
    cudaStream_t stream);

int cosmos3_mot_linear_combo4_f32(
    const float * x0,
    const float * x1,
    const float * x2,
    const float * x3,
    float * y,
    int count,
    float c0,
    float c1,
    float c2,
    float c3,
    cudaStream_t stream);

const char * cosmos3_mot_action_kernel_status(void);

#ifdef __cplusplus
}
#endif
