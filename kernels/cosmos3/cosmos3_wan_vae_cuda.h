// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    COSMOS3_WAN_VAE_PATCH_W = 368,
    COSMOS3_WAN_VAE_PATCH_H = 272,
    COSMOS3_WAN_VAE_FRAMES = 33,
    COSMOS3_WAN_VAE_PATCH_CHANNELS = 12,
    COSMOS3_WAN_VAE_ENCODE_CHUNK_FRAMES = 8,
    COSMOS3_WAN_VAE_CAUSAL_CACHE_T = 2,
};

// Materialize the official RoboLab Wan2.2 VAE encoder input boundary.
//
// image_u8 layout:
//   [image_h, image_w, 3], interleaved RGB U8
//
// patch_whdc layout:
//   [C=12, T=33, H=272, W=368] in WHDC index order:
//   w + W * (h + H * (t + T * c))
//
// The input image is resized/reflect-padded to the official 544 x 736 canvas,
// normalized as value / 127.5 - 1, repeated into frame 0 and padded future
// frames with -1, then spatially patchified with patch_size=2:
//   [3,33,544,736] -> [12,33,272,368].
int cosmos3_wan_vae_robolab_image_to_patch_whdc_f32(
    const unsigned char * image_u8,
    int image_h,
    int image_w,
    float * patch_whdc,
    cudaStream_t stream);

// Debug helper: pack the first `rows` WHDC sites into a row-major [rows, 12]
// matrix, where row order is t-major over H/W:
//   row = (t * H + h) * W + w
int cosmos3_wan_vae_patch_whdc_prefix_f32(
    const float * patch_whdc,
    float * prefix,
    int rows,
    cudaStream_t stream);

enum {
    COSMOS3_WAN_VAE_CONV1_CHANNELS = 160,
    COSMOS3_WAN_VAE_DOWN0_W = 184,
    COSMOS3_WAN_VAE_DOWN0_H = 136,
    COSMOS3_WAN_VAE_DOWN0_T = 33,
    COSMOS3_WAN_VAE_DOWN0_CHANNELS = 160,
    COSMOS3_WAN_VAE_DOWN1_W = 92,
    COSMOS3_WAN_VAE_DOWN1_H = 68,
    COSMOS3_WAN_VAE_DOWN1_T = 17,
    COSMOS3_WAN_VAE_DOWN1_CHANNELS = 320,
    COSMOS3_WAN_VAE_DOWN2_W = 46,
    COSMOS3_WAN_VAE_DOWN2_H = 34,
    COSMOS3_WAN_VAE_DOWN2_T = 9,
    COSMOS3_WAN_VAE_DOWN2_CHANNELS = 640,
};

// Materialize official Wan2.2 VAE encoder.conv1:
//   input  [W=368,H=272,T=33,C=12] WHDC F32
//   output [W=368,H=272,T=33,C=160] WHDC F32
int cosmos3_wan_vae_encoder_conv1_whdc_f32(
    const float * patch_whdc,
    const unsigned short * weight_bf16,
    const unsigned short * bias_bf16,
    float * conv1_whdc,
    cudaStream_t stream);

int cosmos3_wan_vae_encoder_conv1_whdc_prefix_f32(
    const float * conv1_whdc,
    float * prefix,
    int rows,
    int cols,
    cudaStream_t stream);

// Official AvgDown3D shortcut used by encoder.downsamples.0:
//   in  [368,272,33,160] WHDC
//   out [184,136,33,160] WHDC
// factor_t=1, factor_s=2, group_size=4. This is a folded channel/spatial
// mean, not a plain average-pool.
int cosmos3_wan_vae_down0_avg_shortcut_whdc_f32(
    const float * conv1_whdc,
    float * down0_shortcut_whdc,
    cudaStream_t stream);

int cosmos3_wan_vae_down0_whdc_prefix_f32(
    const float * down0_whdc,
    float * prefix,
    int rows,
    int cols,
    cudaStream_t stream);

// Generic WHDC VAE primitives used to assemble Wan2.2 encoder stages without a
// monolithic ggml graph. All tensors are F32 activations in WHDC layout; weights
// are BF16 in PyTorch source order.
int cosmos3_wan_vae_norm_silu_whdc_f32(
    const float * input_whdc,
    const unsigned short * gamma_bf16,
    float * output_whdc,
    int W,
    int H,
    int T,
    int C,
    cudaStream_t stream);

int cosmos3_wan_vae_rms_norm_whdc_f32(
    const float * input_whdc,
    const unsigned short * gamma_bf16,
    float * output_whdc,
    int W,
    int H,
    int T,
    int C,
    cudaStream_t stream);

int cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(
    const float * input_whdc,
    const unsigned short * weight_bf16,
    const unsigned short * bias_bf16,
    float * output_whdc,
    int W,
    int H,
    int T,
    int in_C,
    int out_C,
    cudaStream_t stream);

int cosmos3_wan_vae_add_whdc_f32(
    const float * a,
    const float * b,
    float * out,
    size_t elems,
    cudaStream_t stream);

int cosmos3_wan_vae_spatial_downsample2d_whdc_f32(
    const float * input_whdc,
    const unsigned short * weight_bf16,
    const unsigned short * bias_bf16,
    float * output_whdc,
    int W,
    int H,
    int T,
    int C,
    cudaStream_t stream);

int cosmos3_wan_vae_conv1x1x1_whdc_f32(
    const float * input_whdc,
    const unsigned short * weight_bf16,
    const unsigned short * bias_bf16,
    float * output_whdc,
    int W,
    int H,
    int T,
    int in_C,
    int out_C,
    cudaStream_t stream);

int cosmos3_wan_vae_avg_down3d_whdc_f32(
    const float * input_whdc,
    float * output_whdc,
    int W,
    int H,
    int T,
    int in_C,
    int out_C,
    int factor_t,
    int factor_s,
    cudaStream_t stream);

int cosmos3_wan_vae_downsample3d_time_whdc_f32(
    const float * spatial_whdc,
    const unsigned short * weight_bf16,
    const unsigned short * bias_bf16,
    float * output_whdc,
    int W,
    int H,
    int T_total,
    int C,
    const int * chunks_host,
    int n_chunks,
    cudaStream_t stream);

int cosmos3_wan_vae_generic_whdc_prefix_f32(
    const float * input_whdc,
    float * prefix,
    int W,
    int H,
    int T,
    int C,
    int rows,
    int cols,
    cudaStream_t stream);

int cosmos3_wan_vae_mid_attention_f32(
    const float * qkv_whdc,
    float * scores,
    float * q_row,
    float * k_row,
    float * v_row,
    float * attn_row,
    float * attn_whdc,
    int W,
    int H,
    int T,
    int C,
    cudaStream_t stream);

int cosmos3_wan_vae_pack_clean_vision_condition_f32(
    const float * final_conv1_whdc,
    const unsigned short * scale_mean_bf16,
    const unsigned short * scale_inv_std_bf16,
    float * clean_condition,
    cudaStream_t stream);

// Debug/parity helper for official Wan2.2 VAE encoder.conv1:
//   CausalConv3d(12, 160, kernel=3, padding=1)
// input layout is WHDC:
//   input  [W=368,H=272,T=33,C=12]
// output prefix is row-major [rows, cols], row order is t-major over H/W.
int cosmos3_wan_vae_encoder_conv1_prefix_f32(
    const float * patch_whdc,
    const unsigned short * weight_bf16,
    const unsigned short * bias_bf16,
    float * prefix,
    int rows,
    int cols,
    cudaStream_t stream);

const char * cosmos3_wan_vae_kernel_status(void);

#ifdef __cplusplus
}
#endif
