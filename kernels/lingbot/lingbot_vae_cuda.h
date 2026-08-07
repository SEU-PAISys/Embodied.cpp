// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// LingBot-VA VAE CUDA primitives.
//
// These kernels are kept under kernels/lingbot so LingBot-specific VAE
// residency/streaming work does not depend on or mutate Cosmos3 Wan VAE code.
// Activations use WHDC layout:
//   w + W * (h + H * (t + T * c))

int lingbot_vae_norm_silu_whdc_f32w(
    const float * input_whdc,
    const float * gamma_f32,
    float * output_whdc,
    int W,
    int H,
    int T,
    int C,
    cudaStream_t stream);

int lingbot_vae_causal_conv3d_ks3_whdc_f32w(
    const float * input_whdc,
    const float * weight_f32,
    const float * bias_f32,
    float * output_whdc,
    int W,
    int H,
    int T,
    int in_C,
    int out_C,
    cudaStream_t stream);

int lingbot_vae_cached_causal_conv3d_ks3_whdc_f32w(
    const float * input_whdc,
    const float * cache_whdc,
    const float * weight_f32,
    const float * bias_f32,
    float * output_whdc,
    int W,
    int H,
    int T,
    int cache_T,
    int in_C,
    int out_C,
    cudaStream_t stream);

int lingbot_vae_add_whdc_f32(
    const float * a,
    const float * b,
    float * out,
    size_t elems,
    cudaStream_t stream);

int lingbot_vae_spatial_downsample2d_whdc_f32w(
    const float * input_whdc,
    const float * weight_f32,
    const float * bias_f32,
    float * output_whdc,
    int W,
    int H,
    int T,
    int C,
    cudaStream_t stream);

int lingbot_vae_conv1x1x1_whdc_f32w(
    const float * input_whdc,
    const float * weight_f32,
    const float * bias_f32,
    float * output_whdc,
    int W,
    int H,
    int T,
    int in_C,
    int out_C,
    cudaStream_t stream);

int lingbot_vae_avg_down3d_whdc_f32(
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

int lingbot_vae_downsample3d_time_stream_one_whdc_f32w(
    const float * spatial_whdc,
    const float * cache_whdc,
    const float * weight_f32,
    const float * bias_f32,
    float * output_whdc,
    int W,
    int H,
    int T,
    int cache_T,
    int in_C,
    int out_C,
    cudaStream_t stream);

int lingbot_vae_update_temporal_cache_last_whdc_f32(
    const float * input_whdc,
    float * cache_whdc,
    int W,
    int H,
    int T,
    int old_cache_T,
    int new_cache_T,
    int C,
    cudaStream_t stream);

int lingbot_vae_libero_scale_patchify_vcfhw_to_whdc_f32(
    const float * video_vcfhw,
    float * patch_whdc,
    int view,
    int views,
    int frames,
    int height,
    int width,
    cudaStream_t stream);

int lingbot_vae_normalize_cat_views_whdc_to_bcfhw_f32(
    const float * enc96_views_whdc,
    const float * latents_mean,
    const float * latents_inv_std,
    float * out_bcfhw,
    int views,
    int latent_W_single,
    int latent_H,
    int latent_T,
    int z_dim,
    cudaStream_t stream);

const char * lingbot_vae_kernel_status(void);

#ifdef __cplusplus
}
#endif
