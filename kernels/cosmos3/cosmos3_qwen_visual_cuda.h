// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cosmos3_visual_cuda_ctx cosmos3_visual_cuda_ctx;

typedef struct {
    int frames;
    int temporal_patch;
    int patch;
    int merge;
    int grid_t;
    int grid_h;
    int grid_w;
    int patch_rows;
    int patch_dim;
    int hidden;
    int heads;
    int head_dim;
    int blocks;
    int intermediate;
    int output_hidden;
} cosmos3_visual_cuda_config;

enum {
    COSMOS3_VISUAL_DEBUG_TOKEN_ENTRY = 0,
    COSMOS3_VISUAL_DEBUG_BLOCK0_NORM1,
    COSMOS3_VISUAL_DEBUG_BLOCK0_QKV,
    COSMOS3_VISUAL_DEBUG_BLOCK0_QKV_K,
    COSMOS3_VISUAL_DEBUG_BLOCK0_QKV_V,
    COSMOS3_VISUAL_DEBUG_BLOCK0_Q_ROPE,
    COSMOS3_VISUAL_DEBUG_BLOCK0_K_ROPE,
    COSMOS3_VISUAL_DEBUG_BLOCK0_ATTN,
    COSMOS3_VISUAL_DEBUG_BLOCK0_ATTN_PROJ,
    COSMOS3_VISUAL_DEBUG_BLOCK0_POST_ATTN,
    COSMOS3_VISUAL_DEBUG_BLOCK0_NORM2,
    COSMOS3_VISUAL_DEBUG_BLOCK0_MLP,
    COSMOS3_VISUAL_DEBUG_BLOCK0,
    COSMOS3_VISUAL_DEBUG_FINAL_HIDDEN,
    COSMOS3_VISUAL_DEBUG_MERGER_NORM,
    COSMOS3_VISUAL_DEBUG_MERGED,
    COSMOS3_VISUAL_DEBUG_MERGER_H,
    COSMOS3_VISUAL_DEBUG_COUNT,
};

typedef struct {
    const void * norm1_w;
    const void * norm1_b;
    const void * qkv_w;
    const void * qkv_b;
    const void * proj_w;
    const void * proj_b;
    const void * norm2_w;
    const void * norm2_b;
    const void * fc1_w;
    const void * fc1_b;
    const void * fc2_w;
    const void * fc2_b;
} cosmos3_visual_layer_cuda;

// BitVLA-style runtime object for the Qwen3-VL/Cosmos3 visual tower.
//
// The ctx owns persistent device buffers and lookup tables.  Model code binds
// weight pointers once during model creation and calls forward directly during
// inference; no ggml segment scheduler is involved in this API.
cosmos3_visual_cuda_ctx * cosmos3_visual_cuda_init(const cosmos3_visual_cuda_config * cfg);
void cosmos3_visual_cuda_free(cosmos3_visual_cuda_ctx * ctx);

void cosmos3_visual_cuda_set_embed(
    cosmos3_visual_cuda_ctx * ctx,
    const void * patch_w,
    const void * patch_b,
    const void * pos_embed);

void cosmos3_visual_cuda_set_layer(
    cosmos3_visual_cuda_ctx * ctx,
    int layer,
    const cosmos3_visual_layer_cuda * weights);

void cosmos3_visual_cuda_set_merger(
    cosmos3_visual_cuda_ctx * ctx,
    const void * norm_w,
    const void * norm_b,
    const void * fc1_w,
    const void * fc1_b,
    const void * fc2_w,
    const void * fc2_b);

void cosmos3_visual_cuda_set_deepstack_merger(
    cosmos3_visual_cuda_ctx * ctx,
    int index,
    const void * norm_w,
    const void * norm_b,
    const void * fc1_w,
    const void * fc1_b,
    const void * fc2_w,
    const void * fc2_b);

// Direct visual runtime entry point.
//
// Current contract:
//   image_u8 -> pixel_values is already CUDA-native and stored in ctx-owned
//   device memory.  The full visual block stack is intentionally exposed
//   through this ctx API so subsequent kernels can be registered without
//   reintroducing an intermediate ggml scheduler.
int cosmos3_visual_cuda_forward_robolab_image(
    cosmos3_visual_cuda_ctx * ctx,
    const unsigned char * image_u8_host,
    int image_h,
    int image_w,
    cudaStream_t stream);

const float * cosmos3_visual_cuda_pixel_values(const cosmos3_visual_cuda_ctx * ctx);
float * cosmos3_visual_cuda_tokens(cosmos3_visual_cuda_ctx * ctx);
float * cosmos3_visual_cuda_deepstack_tokens(cosmos3_visual_cuda_ctx * ctx, int index);
void cosmos3_visual_cuda_set_debug_prefix(cosmos3_visual_cuda_ctx * ctx, int enabled, int rows, int cols);
float * cosmos3_visual_cuda_debug_prefix(cosmos3_visual_cuda_ctx * ctx, int index);
void cosmos3_visual_cuda_set_timing(cosmos3_visual_cuda_ctx * ctx, int enabled);
int cosmos3_visual_cuda_copy_timing_ms(const cosmos3_visual_cuda_ctx * ctx, float * values, int count);

// Qwen3-VL visual attention over contiguous windows.
//
// qkv layout:
//   [tokens, 3 * heads * head_dim], token-major, F32
//
// rotary_coords_yx layout:
//   [tokens, 2], int32, storing patch-grid y/x coordinates per token.
//
// window_offsets layout:
//   [windows + 1], int32. Tokens in one window are the contiguous range
//   [window_offsets[w], window_offsets[w + 1]).
//
// token_to_window layout:
//   [tokens], int32. May be null only when windows == 1, in which case every
//   token attends to window 0.
//
// y layout:
//   [tokens, heads * head_dim], token-major, F32
int cosmos3_visual_rope_window_attention_f32(
    const float * qkv,
    const int * rotary_coords_yx,
    const int * window_offsets,
    const int * token_to_window,
    float * y,
    int tokens,
    int windows,
    int heads,
    int head_dim,
    float rope_theta,
    cudaStream_t stream);

// Reorder/concat Qwen3-VL visual patch tokens after merger LayerNorm.
//
// x layout:
//   [patch_rows, hidden], token-major, F32
//
// y layout:
//   [patch_rows / merge, hidden * merge], token-major, F32
//
// This matches the current Cosmos3 RoboLab layout where every 2 neighboring
// flattened rows are concatenated before the merger MLP.
int cosmos3_visual_merge2x2_reorder_f32(
    const float * x,
    float * y,
    int patch_rows,
    int hidden,
    int merge,
    cudaStream_t stream);

// Materialize the Qwen3-VL RoboLab visual input directly on GPU.
//
// image_u8 layout:
//   [image_h, image_w, 3], interleaved RGB U8
//
// pixel_values layout:
//   [patch_rows, patch_dim], token-major, F32
//
// This implements the official RoboLab/Cosmos3 image boundary:
//   padded canvas 544 x 736, reflect padding around the original image,
//   bilinear sampling, RGB normalization value / 127.5 - 1, and temporal
//   patch size 2 where missing future frames are filled with -1.
int cosmos3_visual_robolab_image_to_pixel_values_f32(
    const unsigned char * image_u8,
    int image_h,
    int image_w,
    float * pixel_values,
    int patch_rows,
    int patch_dim,
    cudaStream_t stream);

const char * cosmos3_qwen_visual_kernel_status(void);

#ifdef __cplusplus
}
#endif
