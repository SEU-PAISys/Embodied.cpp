// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#include "cosmos3_qwen_visual_cuda.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

enum VisualTimingSlot {
    VT_IMAGE = 0,
    VT_PATCH_EMBED,
    VT_POS_ADD,
    VT_NORM1,
    VT_QKV,
    VT_ATTENTION,
    VT_PROJ,
    VT_ATTN_RESIDUAL,
    VT_NORM2,
    VT_FC1,
    VT_GELU,
    VT_FC2,
    VT_MLP_RESIDUAL,
    VT_DEEPSTACK,
    VT_FINAL_MERGER,
    VT_COUNT,
};

__device__ __forceinline__ float visual_rope_angle(
        int d,
        int head_dim,
        int coord_y,
        int coord_x,
        float rope_theta) {
    const int rotary_dim = head_dim / 2;
    const int quarter = rotary_dim / 2;
    const int local = d % rotary_dim;
    const int local_pair = local < quarter ? local : local - quarter;
    const int dim_index = local_pair * 2;
    const float inv_freq = powf(rope_theta, -static_cast<float>(dim_index) / static_cast<float>(rotary_dim));
    const int coord = local < quarter ? coord_y : coord_x;
    return static_cast<float>(coord) * inv_freq;
}

__device__ __forceinline__ float visual_qk_rope_value(
        const float * src,
        int base,
        int d,
        int head_dim,
        int coord_y,
        int coord_x,
        float rope_theta) {
    const int half = head_dim / 2;
    const int rd = d < half ? d + half : d - half;
    const float rot_half = d < half ? -src[base + rd] : src[base + rd];
    float s;
    float c;
    sincosf(visual_rope_angle(d, head_dim, coord_y, coord_x, rope_theta), &s, &c);
    return src[base + d] * c + rot_half * s;
}

__device__ __forceinline__ float visual_qk_rope_value_cached(
        const float * src,
        const float * rope_cos,
        const float * rope_sin,
        int token,
        int base,
        int d,
        int head_dim) {
    const int half = head_dim / 2;
    const int rd = d < half ? d + half : d - half;
    const float rot_half = d < half ? -src[base + rd] : src[base + rd];
    const float c = rope_cos[token * head_dim + d];
    const float s = rope_sin[token * head_dim + d];
    return src[base + d] * c + rot_half * s;
}

__device__ __forceinline__ float visual_bf16_round(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

__global__ void visual_window_attention_block_kernel(
        const float * qkv,
        const float * rope_cos,
        const float * rope_sin,
        const int * coords,
        const int * window_offsets,
        const int * token_to_window,
        float * y,
        int tokens,
        int windows,
        int heads,
        int head_dim,
        int max_window,
        float rope_theta) {
    extern __shared__ float smem[];
    float * q_rot = smem;
    float * scores = q_rot + head_dim;
    float * reduce = scores + max_window;

    const int item = blockIdx.x;
    const int tid = threadIdx.x;
    const int token = item / heads;
    const int head = item - token * heads;
    if (token >= tokens) return;

    const int hidden = heads * head_dim;
    const int qkv_dim = 3 * hidden;
    const int window = token_to_window ? token_to_window[token] : 0;
    if (window < 0 || window >= windows) return;
    const int begin = window_offsets[window];
    const int end = window_offsets[window + 1];
    if (begin < 0 || end > tokens || begin >= end) return;
    const int length = end - begin;
    if (length > max_window) return;

    const int coord_y = coords[token * 2 + 0];
    const int coord_x = coords[token * 2 + 1];
    const int q_base = token * qkv_dim + head * head_dim;
    if (tid < head_dim) {
        q_rot[tid] = (rope_cos && rope_sin) ?
            visual_qk_rope_value_cached(qkv, rope_cos, rope_sin, token, q_base, tid, head_dim) :
            visual_qk_rope_value(qkv, q_base, tid, head_dim, coord_y, coord_x, rope_theta);
    }
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(head_dim));
    float max_score = -CUDART_INF_F;
    for (int rel = 0; rel < length; ++rel) {
        const int key = begin + rel;
        const int ky = coords[key * 2 + 0];
        const int kx = coords[key * 2 + 1];
        const int k_base = key * qkv_dim + hidden + head * head_dim;
        float part = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            const float ki = (rope_cos && rope_sin) ?
                visual_qk_rope_value_cached(qkv, rope_cos, rope_sin, key, k_base, i, head_dim) :
                visual_qk_rope_value(qkv, k_base, i, head_dim, ky, kx, rope_theta);
            part += q_rot[i] * ki;
        }
        reduce[tid] = part;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            const float score = reduce[0] * scale;
            scores[rel] = score;
            max_score = fmaxf(max_score, score);
        }
        __syncthreads();
    }

    __shared__ float max_score_shared;
    __shared__ float denom_shared;
    if (tid == 0) max_score_shared = max_score;
    __syncthreads();
    float denom_part = 0.0f;
    for (int rel = tid; rel < length; rel += blockDim.x) {
        denom_part += expf(scores[rel] - max_score_shared);
    }
    reduce[tid] = denom_part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) denom_shared = reduce[0];
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int rel = 0; rel < length; ++rel) {
            const int key = begin + rel;
            const float p = expf(scores[rel] - max_score_shared) / denom_shared;
            const int v_base = key * qkv_dim + 2 * hidden + head * head_dim;
            acc += p * qkv[v_base + d];
        }
        y[token * hidden + head * head_dim + d] = acc;
    }
}

__global__ void visual_qk_rope_cache_kernel(
        const float * qkv,
        const float * rope_cos,
        const float * rope_sin,
        float * q_rope,
        float * k_rope,
        int tokens,
        int heads,
        int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int hidden = heads * head_dim;
    const int total = tokens * hidden;
    if (idx >= total) return;
    const int token = idx / hidden;
    const int hidden_offset = idx - token * hidden;
    const int head = hidden_offset / head_dim;
    const int d = hidden_offset - head * head_dim;
    const int qkv_dim = 3 * hidden;
    const int q_base = token * qkv_dim + head * head_dim;
    const int k_base = token * qkv_dim + hidden + head * head_dim;
    q_rope[idx] = visual_bf16_round(
        visual_qk_rope_value_cached(qkv, rope_cos, rope_sin, token, q_base, d, head_dim));
    k_rope[idx] = visual_bf16_round(
        visual_qk_rope_value_cached(qkv, rope_cos, rope_sin, token, k_base, d, head_dim));
}

__global__ void visual_window_attention_qk_cached_kernel(
        const float * q_rope,
        const float * k_rope,
        const float * qkv,
        const int * window_offsets,
        const int * token_to_window,
        float * y,
        int tokens,
        int windows,
        int heads,
        int head_dim,
        int max_window) {
    extern __shared__ float smem[];
    float * q_cache = smem;
    float * scores = q_cache + head_dim;
    float * reduce = scores + max_window;

    const int item = blockIdx.x;
    const int tid = threadIdx.x;
    const int token = item / heads;
    const int head = item - token * heads;
    if (token >= tokens) return;

    const int hidden = heads * head_dim;
    const int qkv_dim = 3 * hidden;
    const int window = token_to_window ? token_to_window[token] : 0;
    if (window < 0 || window >= windows) return;
    const int begin = window_offsets[window];
    const int end = window_offsets[window + 1];
    if (begin < 0 || end > tokens || begin >= end) return;
    const int length = end - begin;
    if (length > max_window) return;

    const int q_offset = token * hidden + head * head_dim;
    if (tid < head_dim) q_cache[tid] = q_rope[q_offset + tid];
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(head_dim));
    float max_score = -CUDART_INF_F;
    for (int rel = 0; rel < length; ++rel) {
        const int key = begin + rel;
        const int k_offset = key * hidden + head * head_dim;
        float part = 0.0f;
        for (int i = tid; i < head_dim; i += blockDim.x) {
            part += q_cache[i] * k_rope[k_offset + i];
        }
        reduce[tid] = part;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            const float score = reduce[0] * scale;
            scores[rel] = score;
            max_score = fmaxf(max_score, score);
        }
        __syncthreads();
    }

    __shared__ float max_score_shared;
    __shared__ float denom_shared;
    if (tid == 0) max_score_shared = max_score;
    __syncthreads();
    float denom_part = 0.0f;
    for (int rel = tid; rel < length; rel += blockDim.x) {
        denom_part += expf(scores[rel] - max_score_shared);
    }
    reduce[tid] = denom_part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) denom_shared = reduce[0];
    __syncthreads();

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int rel = 0; rel < length; ++rel) {
            const int key = begin + rel;
            const float p = expf(scores[rel] - max_score_shared) / denom_shared;
            const int v_base = key * qkv_dim + 2 * hidden + head * head_dim;
            acc += p * qkv[v_base + d];
        }
        y[token * hidden + head * head_dim + d] = visual_bf16_round(acc);
    }
}

__global__ void visual_window_softmax_v_from_scores_kernel(
        const float * scores,
        const float * qkv,
        float * y,
        int begin,
        int length,
        int heads,
        int head_dim) {
    const int rel_token = blockIdx.x;
    const int head = blockIdx.y;
    const int d = threadIdx.x;
    if (rel_token >= length || head >= heads || d >= head_dim) return;
    const int hidden = heads * head_dim;
    const int qkv_dim = 3 * hidden;
    const float * head_scores = scores + static_cast<size_t>(head) * length * length;
    float max_score = -CUDART_INF_F;
    float denom = 0.0f;
    float acc = 0.0f;
    for (int rel_key = 0; rel_key < length; ++rel_key) {
        const float s = head_scores[rel_key + rel_token * length];
        const float next_max = fmaxf(max_score, s);
        const float old_scale = denom == 0.0f ? 0.0f : expf(max_score - next_max);
        const float new_scale = expf(s - next_max);
        const float next_denom = denom * old_scale + new_scale;
        const float old_weight = next_denom == 0.0f ? 0.0f : (denom * old_scale) / next_denom;
        const float new_weight = next_denom == 0.0f ? 0.0f : new_scale / next_denom;
        const int key = begin + rel_key;
        const float vv = qkv[key * qkv_dim + 2 * hidden + head * head_dim + d];
        acc = acc * old_weight + new_weight * vv;
        max_score = next_max;
        denom = next_denom;
    }
    const int token = begin + rel_token;
    y[token * hidden + head * head_dim + d] = visual_bf16_round(acc);
}

__global__ void merge2x2_reorder_kernel(
        const float * x,
        float * y,
        int patch_rows,
        int hidden,
        int merge) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int merged_rows = patch_rows / merge;
    const int merged_hidden = hidden * merge;
    const int total = merged_rows * merged_hidden;
    if (idx >= total) return;
    const int col = idx % merged_hidden;
    const int row = idx / merged_hidden;
    const int merge_i = col / hidden;
    const int hidden_i = col - merge_i * hidden;
    const int src_row = row * merge + merge_i;
    y[idx] = x[src_row * hidden + hidden_i];
}

__device__ __forceinline__ int reflect_to_content(
        int padded_coord,
        int pad_before,
        int content_size) {
    if (content_size <= 1) return 0;
    int rel = padded_coord - pad_before;
    while (rel < 0 || rel >= content_size) {
        if (rel < 0) {
            rel = -rel;
        } else {
            rel = 2 * content_size - 2 - rel;
        }
    }
    return rel;
}

__device__ __forceinline__ void decode_qwen3vl_flattened_index(
        int row,
        int col,
        int * frame,
        int * channel,
        int * y,
        int * x) {
    constexpr int patch = 16;
    constexpr int temporal = 2;
    constexpr int channels = 3;
    constexpr int merge = 2;
    constexpr int grid_h = 34;
    constexpr int grid_w = 46;
    constexpr int gh_merged = grid_h / merge;
    constexpr int gw_merged = grid_w / merge;

    int r = row;
    const int merge_w = r % merge; r /= merge;
    const int merge_h = r % merge; r /= merge;
    const int gw = r % gw_merged; r /= gw_merged;
    const int gh = r % gh_merged; r /= gh_merged;
    const int gt = r;

    int c = col;
    const int patch_x = c % patch; c /= patch;
    const int patch_y = c % patch; c /= patch;
    const int temporal_i = c % temporal; c /= temporal;
    *channel = c % channels;

    *frame = gt * temporal + temporal_i;
    *y = (gh * merge + merge_h) * patch + patch_y;
    *x = (gw * merge + merge_w) * patch + patch_x;
}

__device__ __forceinline__ float robolab_image_sample_norm(
        const unsigned char * image,
        int image_h,
        int image_w,
        int channel,
        int padded_y,
        int padded_x) {
    constexpr int padded_h = 544;
    constexpr int padded_w = 736;
    const int content_h = min(image_h, padded_h);
    const int content_w = min(image_w, padded_w);
    const int pad_top = (padded_h - content_h) / 2;
    const int pad_left = (padded_w - content_w) / 2;

    const int content_y = reflect_to_content(padded_y, pad_top, content_h);
    const int content_x = reflect_to_content(padded_x, pad_left, content_w);
    const float scale_y = static_cast<float>(content_h) / static_cast<float>(image_h);
    const float scale_x = static_cast<float>(content_w) / static_cast<float>(image_w);
    const float src_y = (static_cast<float>(content_y) + 0.5f) / scale_y - 0.5f;
    const float src_x = (static_cast<float>(content_x) + 0.5f) / scale_x - 0.5f;

    const int y0 = max(0, min(image_h - 1, static_cast<int>(floorf(src_y))));
    const int x0 = max(0, min(image_w - 1, static_cast<int>(floorf(src_x))));
    const int y1 = min(image_h - 1, y0 + 1);
    const int x1 = min(image_w - 1, x0 + 1);
    const float wy = min(1.0f, max(0.0f, src_y - static_cast<float>(y0)));
    const float wx = min(1.0f, max(0.0f, src_x - static_cast<float>(x0)));

    const float p00 = static_cast<float>(image[(y0 * image_w + x0) * 3 + channel]);
    const float p01 = static_cast<float>(image[(y0 * image_w + x1) * 3 + channel]);
    const float p10 = static_cast<float>(image[(y1 * image_w + x0) * 3 + channel]);
    const float p11 = static_cast<float>(image[(y1 * image_w + x1) * 3 + channel]);
    const float top = p00 * (1.0f - wx) + p01 * wx;
    const float bottom = p10 * (1.0f - wx) + p11 * wx;
    const float value = top * (1.0f - wy) + bottom * wy;
    return value / 127.5f - 1.0f;
}

__global__ void robolab_image_to_pixel_values_kernel(
        const unsigned char * image,
        int image_h,
        int image_w,
        float * pixel_values,
        int patch_rows,
        int patch_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = patch_rows * patch_dim;
    if (idx >= total) return;
    const int col = idx % patch_dim;
    const int row = idx / patch_dim;

    int frame = 0;
    int channel = 0;
    int y = 0;
    int x = 0;
    decode_qwen3vl_flattened_index(row, col, &frame, &channel, &y, &x);
    if (frame != 0) {
        pixel_values[idx] = -1.0f;
        return;
    }
    pixel_values[idx] = robolab_image_sample_norm(image, image_h, image_w, channel, y, x);
}

__global__ void visual_pos_add_kernel(
        float * x,
        const __nv_bfloat16 * pos,
        const int * coords_yx,
        int rows,
        int hidden,
        int grid_h,
        int grid_w,
        int grid_side) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * hidden;
    if (idx >= total) return;
    const int h = idx % hidden;
    const int row = idx / hidden;
    const int patch_y = coords_yx[row * 2 + 0];
    const int patch_x = coords_yx[row * 2 + 1];
    const float fy = grid_h > 1 ?
        static_cast<float>(patch_y) * static_cast<float>(grid_side - 1) / static_cast<float>(grid_h - 1) :
        0.0f;
    const float fx = grid_w > 1 ?
        static_cast<float>(patch_x) * static_cast<float>(grid_side - 1) / static_cast<float>(grid_w - 1) :
        0.0f;
    const int y0 = max(0, min(grid_side - 1, static_cast<int>(floorf(fy))));
    const int x0 = max(0, min(grid_side - 1, static_cast<int>(floorf(fx))));
    const int y1 = min(grid_side - 1, y0 + 1);
    const int x1 = min(grid_side - 1, x0 + 1);
    const float dy = fy - static_cast<float>(y0);
    const float dx = fx - static_cast<float>(x0);
    const float w00 = (1.0f - dy) * (1.0f - dx);
    const float w01 = (1.0f - dy) * dx;
    const float w10 = dy * (1.0f - dx);
    const float w11 = dy * dx;
    const int i00 = (y0 * grid_side + x0) * hidden + h;
    const int i01 = (y0 * grid_side + x1) * hidden + h;
    const int i10 = (y1 * grid_side + x0) * hidden + h;
    const int i11 = (y1 * grid_side + x1) * hidden + h;
    const float pe = w00 * __bfloat162float(pos[i00]) +
                     w01 * __bfloat162float(pos[i01]) +
                     w10 * __bfloat162float(pos[i10]) +
                     w11 * __bfloat162float(pos[i11]);
    x[idx] += pe;
}

__global__ void visual_layernorm_kernel(
        const float * x,
        const __nv_bfloat16 * w,
        const __nv_bfloat16 * b,
        float * y,
        float eps,
        int rows,
        int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float mean = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        mean += x[row * cols + c];
    }
    __shared__ float smem[256];
    smem[threadIdx.x] = mean;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    mean = smem[0] / static_cast<float>(cols);

    float var = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float d = x[row * cols + c] - mean;
        var += d * d;
    }
    smem[threadIdx.x] = var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / static_cast<float>(cols) + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        y[row * cols + c] = (x[row * cols + c] - mean) * inv *
                            __bfloat162float(w[c]) + __bfloat162float(b[c]);
    }
}

__global__ void visual_add_bias_kernel(
        float * y,
        const __nv_bfloat16 * b,
        int rows,
        int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    y[idx] += __bfloat162float(b[idx % cols]);
}

__global__ void visual_add_kernel(
        const float * a,
        const float * b,
        float * out,
        int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void visual_gelu_kernel(float * x, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = x[idx];
    x[idx] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

__global__ void visual_gelu_exact_kernel(float * x, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = x[idx];
    x[idx] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
}

__global__ void visual_round_to_bf16_kernel(float * x, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = visual_bf16_round(x[idx]);
}

__global__ void visual_f32_to_bf16_kernel(
        const float * x,
        __nv_bfloat16 * y,
        int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = __float2bfloat16(x[idx]);
}

} // namespace

struct cosmos3_visual_layer_cuda_dev {
    const __nv_bfloat16 * norm1_w = nullptr;
    const __nv_bfloat16 * norm1_b = nullptr;
    const __nv_bfloat16 * qkv_w = nullptr;
    const __nv_bfloat16 * qkv_b = nullptr;
    const __nv_bfloat16 * proj_w = nullptr;
    const __nv_bfloat16 * proj_b = nullptr;
    const __nv_bfloat16 * norm2_w = nullptr;
    const __nv_bfloat16 * norm2_b = nullptr;
    const __nv_bfloat16 * fc1_w = nullptr;
    const __nv_bfloat16 * fc1_b = nullptr;
    const __nv_bfloat16 * fc2_w = nullptr;
    const __nv_bfloat16 * fc2_b = nullptr;
};

struct cosmos3_visual_merger_cuda_dev {
    const __nv_bfloat16 * norm_w = nullptr;
    const __nv_bfloat16 * norm_b = nullptr;
    const __nv_bfloat16 * fc1_w = nullptr;
    const __nv_bfloat16 * fc1_b = nullptr;
    const __nv_bfloat16 * fc2_w = nullptr;
    const __nv_bfloat16 * fc2_b = nullptr;
};

struct cosmos3_visual_cuda_ctx {
    cosmos3_visual_cuda_config cfg{};
    cublasHandle_t blas = nullptr;
    unsigned char * d_image = nullptr;
    size_t d_image_bytes = 0;
    float * d_pixel_values = nullptr;
    float * d_tokens = nullptr;
    float * d_x0 = nullptr;
    float * d_x1 = nullptr;
    float * d_norm = nullptr;
    float * d_qkv = nullptr;
    float * d_q_rope = nullptr;
    float * d_k_rope = nullptr;
    float * d_attn = nullptr;
    float * d_mlp = nullptr;
    float * d_merged = nullptr;
    float * d_merger_h = nullptr;
    float * d_deepstack_tokens[3] = {nullptr, nullptr, nullptr};
    std::array<float *, COSMOS3_VISUAL_DEBUG_COUNT> d_debug_prefix{};
    int debug_rows = 0;
    int debug_cols = 0;
    bool debug_enabled = false;
    __nv_bfloat16 * d_linear_in_bf16 = nullptr;
    int * d_coords_yx = nullptr;
    int * d_window_offsets = nullptr;
    int * d_token_to_window = nullptr;
    int * d_pos_ids = nullptr;
    float * d_rope_cos = nullptr;
    float * d_rope_sin = nullptr;
    float * d_attn_scores = nullptr;
    int windows = 0;
    int max_window_tokens = 0;
    bool timing_enabled = false;
    float timing_ms[VT_COUNT] = {};

    const __nv_bfloat16 * patch_w = nullptr;
    const __nv_bfloat16 * patch_b = nullptr;
    const __nv_bfloat16 * pos_embed = nullptr;
    std::vector<cosmos3_visual_layer_cuda_dev> layers;
    const __nv_bfloat16 * merger_norm_w = nullptr;
    const __nv_bfloat16 * merger_norm_b = nullptr;
    const __nv_bfloat16 * merger_fc1_w = nullptr;
    const __nv_bfloat16 * merger_fc1_b = nullptr;
    const __nv_bfloat16 * merger_fc2_w = nullptr;
    const __nv_bfloat16 * merger_fc2_b = nullptr;
    cosmos3_visual_merger_cuda_dev deepstack[3];
};

struct VisualTimingGuard {
    cosmos3_visual_cuda_ctx * ctx = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    bool ok = true;

    VisualTimingGuard(cosmos3_visual_cuda_ctx * c, cudaStream_t s) : ctx(c), stream(s) {
        if (!ctx || !ctx->timing_enabled) return;
        ok = cudaEventCreate(&start) == cudaSuccess &&
             cudaEventCreate(&stop) == cudaSuccess;
        if (ok) std::fill(std::begin(ctx->timing_ms), std::end(ctx->timing_ms), 0.0f);
    }

    ~VisualTimingGuard() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }

    void begin() {
        if (ok && start) cudaEventRecord(start, stream);
    }

    bool end(int slot) {
        if (!ok || !start || !stop) return true;
        cudaEventRecord(stop, stream);
        if (cudaEventSynchronize(stop) != cudaSuccess) return false;
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        if (slot >= 0 && slot < VT_COUNT) ctx->timing_ms[slot] += ms;
        return true;
    }
};

static void visual_cuda_release_allocs(cosmos3_visual_cuda_ctx * ctx) {
    if (!ctx) return;
    if (ctx->d_image) cudaFree(ctx->d_image);
    if (ctx->d_pixel_values) cudaFree(ctx->d_pixel_values);
    if (ctx->d_tokens) cudaFree(ctx->d_tokens);
    if (ctx->d_x0) cudaFree(ctx->d_x0);
    if (ctx->d_x1) cudaFree(ctx->d_x1);
    if (ctx->d_norm) cudaFree(ctx->d_norm);
    if (ctx->d_qkv) cudaFree(ctx->d_qkv);
    if (ctx->d_q_rope) cudaFree(ctx->d_q_rope);
    if (ctx->d_k_rope) cudaFree(ctx->d_k_rope);
    if (ctx->d_attn) cudaFree(ctx->d_attn);
    if (ctx->d_mlp) cudaFree(ctx->d_mlp);
    if (ctx->d_merged) cudaFree(ctx->d_merged);
    if (ctx->d_merger_h) cudaFree(ctx->d_merger_h);
    for (float * ptr : ctx->d_deepstack_tokens) {
        if (ptr) cudaFree(ptr);
    }
    for (float * ptr : ctx->d_debug_prefix) {
        if (ptr) cudaFree(ptr);
    }
    if (ctx->d_linear_in_bf16) cudaFree(ctx->d_linear_in_bf16);
    if (ctx->d_coords_yx) cudaFree(ctx->d_coords_yx);
    if (ctx->d_window_offsets) cudaFree(ctx->d_window_offsets);
    if (ctx->d_token_to_window) cudaFree(ctx->d_token_to_window);
    if (ctx->d_pos_ids) cudaFree(ctx->d_pos_ids);
    if (ctx->d_rope_cos) cudaFree(ctx->d_rope_cos);
    if (ctx->d_rope_sin) cudaFree(ctx->d_rope_sin);
    if (ctx->d_attn_scores) cudaFree(ctx->d_attn_scores);
    ctx->d_image = nullptr;
    ctx->d_pixel_values = nullptr;
    ctx->d_tokens = nullptr;
    ctx->d_x0 = nullptr;
    ctx->d_x1 = nullptr;
    ctx->d_norm = nullptr;
    ctx->d_qkv = nullptr;
    ctx->d_q_rope = nullptr;
    ctx->d_k_rope = nullptr;
    ctx->d_attn = nullptr;
    ctx->d_mlp = nullptr;
    ctx->d_merged = nullptr;
    ctx->d_merger_h = nullptr;
    for (float *& ptr : ctx->d_deepstack_tokens) ptr = nullptr;
    for (float *& ptr : ctx->d_debug_prefix) ptr = nullptr;
    ctx->debug_enabled = false;
    ctx->debug_rows = 0;
    ctx->debug_cols = 0;
    ctx->d_linear_in_bf16 = nullptr;
    ctx->d_coords_yx = nullptr;
    ctx->d_window_offsets = nullptr;
    ctx->d_token_to_window = nullptr;
    ctx->d_pos_ids = nullptr;
    ctx->d_rope_cos = nullptr;
    ctx->d_rope_sin = nullptr;
    ctx->d_attn_scores = nullptr;
    ctx->max_window_tokens = 0;
}

static void visual_debug_release(cosmos3_visual_cuda_ctx * ctx) {
    if (!ctx) return;
    for (float *& ptr : ctx->d_debug_prefix) {
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
    }
    ctx->debug_enabled = false;
    ctx->debug_rows = 0;
    ctx->debug_cols = 0;
}

static bool visual_debug_capture_prefix(cosmos3_visual_cuda_ctx * ctx,
                                        int index,
                                        const float * src,
                                        int src_rows,
                                        int src_cols,
                                        int col_offset,
                                        cudaStream_t stream) {
    if (!ctx || !ctx->debug_enabled) return true;
    if (index < 0 || index >= COSMOS3_VISUAL_DEBUG_COUNT || !src) return false;
    float * dst = ctx->d_debug_prefix[static_cast<size_t>(index)];
    if (!dst || ctx->debug_rows <= 0 || ctx->debug_cols <= 0) return false;
    const int rows = std::min(ctx->debug_rows, src_rows);
    const int cols = std::min(ctx->debug_cols, std::max(0, src_cols - col_offset));
    if (rows <= 0 || cols <= 0) return false;
    const float * src0 = src + col_offset;
    return cudaMemcpy2DAsync(dst,
                             static_cast<size_t>(ctx->debug_cols) * sizeof(float),
                             src0,
                             static_cast<size_t>(src_cols) * sizeof(float),
                             static_cast<size_t>(cols) * sizeof(float),
                             static_cast<size_t>(rows),
                             cudaMemcpyDeviceToDevice,
                             stream) == cudaSuccess;
}

static bool visual_round_activation(cosmos3_visual_cuda_ctx * ctx,
                                    float * x,
                                    int n,
                                    cudaStream_t stream) {
    (void) ctx;
    (void) x;
    (void) n;
    (void) stream;
    return true;
}

static bool visual_linear_bf16_weight_f32_out(
        cosmos3_visual_cuda_ctx * ctx,
        const float * x,
        const __nv_bfloat16 * w,
        const __nv_bfloat16 * b,
        float * y,
        int rows,
        int in_dim,
        int out_dim,
        const char * label,
        cudaStream_t stream) {
    if (!ctx || !ctx->blas || !x || !w || !y) return false;
    constexpr int block = 256;
    const int input_elems = rows * in_dim;
    visual_f32_to_bf16_kernel<<<(input_elems + block - 1) / block, block, 0, stream>>>(
        x, ctx->d_linear_in_bf16, input_elems);
    if (cudaGetLastError() != cudaSuccess) return false;
    cublasSetStream(ctx->blas, stream);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major Y[rows,out] = X[rows,in] * W[out,in]^T.
    // Interpreted as column-major: Y^T[out,rows] = W[out,in] * X^T[in,rows].
    cublasStatus_t st = cublasGemmEx(
        ctx->blas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_dim,
        rows,
        in_dim,
        &alpha,
        w,
        CUDA_R_16BF,
        in_dim,
        ctx->d_linear_in_bf16,
        CUDA_R_16BF,
        in_dim,
        &beta,
        y,
        CUDA_R_32F,
        out_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "vla(cosmos3_visual_cuda): cublasGemmEx failed label=%s rows=%d in=%d out=%d status=%d\n",
                     label ? label : "<unnamed>", rows, in_dim, out_dim, static_cast<int>(st));
        return false;
    }
    if (b) {
        constexpr int block = 256;
        visual_add_bias_kernel<<<(rows * out_dim + block - 1) / block, block, 0, stream>>>(
            y, b, rows, out_dim);
    }
    visual_round_to_bf16_kernel<<<(rows * out_dim + block - 1) / block, block, 0, stream>>>(
        y, rows * out_dim);
    return cudaGetLastError() == cudaSuccess;
}

static int visual_rope_window_attention_qk_cached_f32(
    cosmos3_visual_cuda_ctx * ctx,
    const float * qkv,
    float * y,
    cudaStream_t stream) {
    if (!ctx || !qkv || !y || !ctx->d_rope_cos || !ctx->d_rope_sin ||
        !ctx->d_q_rope || !ctx->d_k_rope || !ctx->d_window_offsets ||
        !ctx->d_token_to_window) {
        return -1;
    }
    const auto & cfg = ctx->cfg;
    const int hidden = cfg.heads * cfg.head_dim;
    const int total = cfg.patch_rows * hidden;
    if (ctx->max_window_tokens <= 0) return -1;
    constexpr int block_threads = 256;
    visual_qk_rope_cache_kernel<<<(total + block_threads - 1) / block_threads,
                                  block_threads, 0, stream>>>(
        qkv, ctx->d_rope_cos, ctx->d_rope_sin,
        ctx->d_q_rope, ctx->d_k_rope,
        cfg.patch_rows, cfg.heads, cfg.head_dim);
    if (cudaGetLastError() != cudaSuccess) return -1;

    if (ctx->d_attn_scores && ctx->blas) {
        cublasSetStream(ctx->blas, stream);
        const float alpha = rsqrtf(static_cast<float>(cfg.head_dim));
        const float beta = 0.0f;
        bool cublas_ok = true;
        for (int window = 0; window < ctx->windows; ++window) {
            const int begin = window * ctx->max_window_tokens;
            const int end = min(begin + ctx->max_window_tokens, cfg.patch_rows);
            const int length = end - begin;
            if (length <= 0) continue;
            const float * q_head0 = ctx->d_q_rope + static_cast<size_t>(begin) * hidden;
            const float * k_head0 = ctx->d_k_rope + static_cast<size_t>(begin) * hidden;
            const long long stride_head = cfg.head_dim;
            const long long stride_scores = static_cast<long long>(length) * static_cast<long long>(length);
            const cublasStatus_t st = cublasGemmStridedBatchedEx(
                ctx->blas,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                length,
                length,
                cfg.head_dim,
                &alpha,
                k_head0,
                CUDA_R_32F,
                hidden,
                stride_head,
                q_head0,
                CUDA_R_32F,
                hidden,
                stride_head,
                &beta,
                ctx->d_attn_scores,
                CUDA_R_32F,
                length,
                stride_scores,
                cfg.heads,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if (st != CUBLAS_STATUS_SUCCESS) {
                cublas_ok = false;
                break;
            }
            const dim3 grid(length, cfg.heads);
            const dim3 block(cfg.head_dim);
            visual_window_softmax_v_from_scores_kernel<<<grid, block, 0, stream>>>(
                ctx->d_attn_scores, qkv, y, begin, length, cfg.heads, cfg.head_dim);
            if (cudaGetLastError() != cudaSuccess) {
                cublas_ok = false;
                break;
            }
        }
        if (cublas_ok) return 0;
    }

    const dim3 grid(cfg.patch_rows * cfg.heads);
    const dim3 block(128);
    const size_t smem = static_cast<size_t>(cfg.head_dim + ctx->max_window_tokens + 128) * sizeof(float);
    visual_window_attention_qk_cached_kernel<<<grid, block, smem, stream>>>(
        ctx->d_q_rope, ctx->d_k_rope, qkv,
        ctx->d_window_offsets, ctx->d_token_to_window, y,
        cfg.patch_rows, ctx->windows, cfg.heads, cfg.head_dim, ctx->max_window_tokens);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

static bool visual_run_patch_merger_postshuffle_norm(
        cosmos3_visual_cuda_ctx * ctx,
        const float * x,
        const cosmos3_visual_merger_cuda_dev & w,
        float * out,
        const char * label,
        cudaStream_t stream) {
    if (!ctx || !x || !out || !w.norm_w || !w.norm_b ||
        !w.fc1_w || !w.fc1_b || !w.fc2_w || !w.fc2_b) {
        return false;
    }
    const auto & cfg = ctx->cfg;
    constexpr int block = 256;
    const int visual_tokens = cfg.patch_rows / (cfg.merge * cfg.merge);
    const int merged_hidden = cfg.hidden * cfg.merge * cfg.merge;
    if (cosmos3_visual_merge2x2_reorder_f32(
            x, ctx->d_merged, cfg.patch_rows, cfg.hidden,
            cfg.merge * cfg.merge, stream) != 0) {
        return false;
    }
    visual_layernorm_kernel<<<visual_tokens, 256, 0, stream>>>(
        ctx->d_merged, w.norm_w, w.norm_b, ctx->d_merged, 1e-6f,
        visual_tokens, merged_hidden);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (!visual_round_activation(ctx, ctx->d_merged, visual_tokens * merged_hidden, stream)) {
        return false;
    }
    if (!visual_linear_bf16_weight_f32_out(
            ctx, ctx->d_merged, w.fc1_w, w.fc1_b,
            ctx->d_merger_h, visual_tokens, merged_hidden, merged_hidden,
            label, stream)) {
        return false;
    }
    visual_gelu_exact_kernel<<<(visual_tokens * merged_hidden + block - 1) / block, block, 0, stream>>>(
        ctx->d_merger_h, visual_tokens * merged_hidden);
    if (!visual_round_activation(ctx, ctx->d_merger_h, visual_tokens * merged_hidden, stream)) {
        return false;
    }
    return visual_linear_bf16_weight_f32_out(
        ctx, ctx->d_merger_h, w.fc2_w, w.fc2_b,
        out, visual_tokens, merged_hidden, cfg.output_hidden,
        label, stream);
}

extern "C" cosmos3_visual_cuda_ctx * cosmos3_visual_cuda_init(
        const cosmos3_visual_cuda_config * cfg) {
    if (!cfg || cfg->patch_rows <= 0 || cfg->patch_dim <= 0 ||
        cfg->output_hidden <= 0 || cfg->merge <= 0) {
        return nullptr;
    }
    auto * ctx = new cosmos3_visual_cuda_ctx();
    ctx->cfg = *cfg;
    if (cublasCreate(&ctx->blas) != CUBLAS_STATUS_SUCCESS) {
        delete ctx;
        return nullptr;
    }
    ctx->layers.resize(static_cast<size_t>(cfg->blocks));
    const size_t pixel_values_bytes =
        static_cast<size_t>(cfg->patch_rows) * static_cast<size_t>(cfg->patch_dim) * sizeof(float);
    const int visual_tokens = cfg->patch_rows / (cfg->merge * cfg->merge);
    const size_t tokens_bytes =
        static_cast<size_t>(visual_tokens) * static_cast<size_t>(cfg->output_hidden) * sizeof(float);
    const size_t hidden_bytes = static_cast<size_t>(cfg->patch_rows) * cfg->hidden * sizeof(float);
    const size_t qkv_bytes = static_cast<size_t>(cfg->patch_rows) * 3u * cfg->hidden * sizeof(float);
    const size_t mlp_bytes = static_cast<size_t>(cfg->patch_rows) * cfg->intermediate * sizeof(float);
    const size_t merged_bytes = static_cast<size_t>(visual_tokens) *
                                static_cast<size_t>(cfg->hidden * cfg->merge * cfg->merge) * sizeof(float);
    const size_t merger_h_bytes = static_cast<size_t>(visual_tokens) *
                                  static_cast<size_t>(cfg->hidden * cfg->merge * cfg->merge) * sizeof(float);
    const size_t deepstack_bytes =
        static_cast<size_t>(visual_tokens) * static_cast<size_t>(cfg->output_hidden) * sizeof(float);
    const int max_linear_in = std::max(std::max(cfg->patch_dim, cfg->hidden),
                                       std::max(cfg->intermediate, cfg->hidden * cfg->merge * cfg->merge));
    const size_t linear_in_bf16_bytes =
        static_cast<size_t>(cfg->patch_rows) * static_cast<size_t>(max_linear_in) * sizeof(__nv_bfloat16);
    if (cudaMalloc(reinterpret_cast<void **>(&ctx->d_pixel_values), pixel_values_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_tokens), tokens_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_x0), hidden_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_x1), hidden_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_norm), hidden_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_qkv), qkv_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_q_rope), hidden_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_k_rope), hidden_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_attn), hidden_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_mlp), mlp_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_merged), merged_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_merger_h), merger_h_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_deepstack_tokens[0]), deepstack_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_deepstack_tokens[1]), deepstack_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_deepstack_tokens[2]), deepstack_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_linear_in_bf16), linear_in_bf16_bytes) != cudaSuccess) {
        cosmos3_visual_cuda_free(ctx);
        return nullptr;
    }
    std::vector<int> coords(static_cast<size_t>(cfg->patch_rows) * 2u, 0);
    std::vector<int> pos_ids(cfg->patch_rows, 0);
    std::vector<float> rope_cos(static_cast<size_t>(cfg->patch_rows) * static_cast<size_t>(cfg->head_dim), 1.0f);
    std::vector<float> rope_sin(static_cast<size_t>(cfg->patch_rows) * static_cast<size_t>(cfg->head_dim), 0.0f);
    const int gw_merged = cfg->grid_w / cfg->merge;
    const int gh_merged = cfg->grid_h / cfg->merge;
    for (int row = 0; row < cfg->patch_rows; ++row) {
        int r = row;
        const int merge_w = r % cfg->merge; r /= cfg->merge;
        const int merge_h = r % cfg->merge; r /= cfg->merge;
        const int gw = r % gw_merged; r /= gw_merged;
        const int gh = r % gh_merged;
        const int patch_y = gh * cfg->merge + merge_h;
        const int patch_x = gw * cfg->merge + merge_w;
        coords[static_cast<size_t>(row) * 2u + 0u] = patch_y;
        coords[static_cast<size_t>(row) * 2u + 1u] = patch_x;
        pos_ids[static_cast<size_t>(row)] = patch_y * 48 + patch_x;
        const int rotary_dim = cfg->head_dim / 2;
        const int quarter = rotary_dim / 2;
        for (int d = 0; d < cfg->head_dim; ++d) {
            const int local = d % rotary_dim;
            const int local_pair = local < quarter ? local : local - quarter;
            const int dim_index = local_pair * 2;
            const float inv_freq = std::pow(10000.0f, -static_cast<float>(dim_index) / static_cast<float>(rotary_dim));
            const int coord = local < quarter ? patch_y : patch_x;
            const float angle = static_cast<float>(coord) * inv_freq;
            rope_cos[static_cast<size_t>(row) * static_cast<size_t>(cfg->head_dim) + static_cast<size_t>(d)] =
                std::cos(angle);
            rope_sin[static_cast<size_t>(row) * static_cast<size_t>(cfg->head_dim) + static_cast<size_t>(d)] =
                std::sin(angle);
        }
    }
    // Official Qwen3-VL visual attention uses cu_seqlens derived from
    // grid_thw.  For RoboLab this is grid_thw=[17,34,46], equivalent to
    // 17 contiguous per-frame segments of 34*46 patch tokens.
    const int window_size = cfg->grid_h * cfg->grid_w;
    ctx->windows = cfg->grid_t;
    ctx->max_window_tokens = window_size;
    const size_t attn_scores_bytes =
        static_cast<size_t>(cfg->heads) *
        static_cast<size_t>(ctx->max_window_tokens) *
        static_cast<size_t>(ctx->max_window_tokens) *
        sizeof(float);
    std::vector<int> offsets(static_cast<size_t>(ctx->windows) + 1u, 0);
    std::vector<int> token_to_window(cfg->patch_rows, 0);
    for (int w = 0; w <= ctx->windows; ++w) {
        offsets[static_cast<size_t>(w)] = min(w * window_size, cfg->patch_rows);
    }
    for (int w = 0; w < ctx->windows; ++w) {
        for (int token = offsets[static_cast<size_t>(w)];
             token < offsets[static_cast<size_t>(w + 1)]; ++token) {
            token_to_window[static_cast<size_t>(token)] = w;
        }
    }
    if (cudaMalloc(reinterpret_cast<void **>(&ctx->d_attn_scores), attn_scores_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_coords_yx), coords.size() * sizeof(int)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_window_offsets), offsets.size() * sizeof(int)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_token_to_window), token_to_window.size() * sizeof(int)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_pos_ids), pos_ids.size() * sizeof(int)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_rope_cos), rope_cos.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&ctx->d_rope_sin), rope_sin.size() * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(ctx->d_coords_yx, coords.data(), coords.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ctx->d_window_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ctx->d_token_to_window, token_to_window.data(), token_to_window.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ctx->d_pos_ids, pos_ids.data(), pos_ids.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ctx->d_rope_cos, rope_cos.data(), rope_cos.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ctx->d_rope_sin, rope_sin.data(), rope_sin.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cosmos3_visual_cuda_free(ctx);
        return nullptr;
    }
    cudaMemset(ctx->d_tokens, 0, tokens_bytes);
    cudaMemset(ctx->d_deepstack_tokens[0], 0, deepstack_bytes);
    cudaMemset(ctx->d_deepstack_tokens[1], 0, deepstack_bytes);
    cudaMemset(ctx->d_deepstack_tokens[2], 0, deepstack_bytes);
    return ctx;
}

extern "C" void cosmos3_visual_cuda_free(cosmos3_visual_cuda_ctx * ctx) {
    if (!ctx) return;
    visual_cuda_release_allocs(ctx);
    if (ctx->blas) cublasDestroy(ctx->blas);
    delete ctx;
}

extern "C" void cosmos3_visual_cuda_set_embed(
    cosmos3_visual_cuda_ctx * ctx,
    const void * patch_w,
    const void * patch_b,
    const void * pos_embed) {
    if (!ctx) return;
    ctx->patch_w = static_cast<const __nv_bfloat16 *>(patch_w);
    ctx->patch_b = static_cast<const __nv_bfloat16 *>(patch_b);
    ctx->pos_embed = static_cast<const __nv_bfloat16 *>(pos_embed);
}

extern "C" void cosmos3_visual_cuda_set_layer(
    cosmos3_visual_cuda_ctx * ctx,
    int layer,
    const cosmos3_visual_layer_cuda * weights) {
    if (!ctx || !weights || layer < 0 || layer >= static_cast<int>(ctx->layers.size())) return;
    auto & dst = ctx->layers[static_cast<size_t>(layer)];
    dst.norm1_w = static_cast<const __nv_bfloat16 *>(weights->norm1_w);
    dst.norm1_b = static_cast<const __nv_bfloat16 *>(weights->norm1_b);
    dst.qkv_w = static_cast<const __nv_bfloat16 *>(weights->qkv_w);
    dst.qkv_b = static_cast<const __nv_bfloat16 *>(weights->qkv_b);
    dst.proj_w = static_cast<const __nv_bfloat16 *>(weights->proj_w);
    dst.proj_b = static_cast<const __nv_bfloat16 *>(weights->proj_b);
    dst.norm2_w = static_cast<const __nv_bfloat16 *>(weights->norm2_w);
    dst.norm2_b = static_cast<const __nv_bfloat16 *>(weights->norm2_b);
    dst.fc1_w = static_cast<const __nv_bfloat16 *>(weights->fc1_w);
    dst.fc1_b = static_cast<const __nv_bfloat16 *>(weights->fc1_b);
    dst.fc2_w = static_cast<const __nv_bfloat16 *>(weights->fc2_w);
    dst.fc2_b = static_cast<const __nv_bfloat16 *>(weights->fc2_b);
}

extern "C" void cosmos3_visual_cuda_set_merger(
    cosmos3_visual_cuda_ctx * ctx,
    const void * norm_w,
    const void * norm_b,
    const void * fc1_w,
    const void * fc1_b,
    const void * fc2_w,
    const void * fc2_b) {
    if (!ctx) return;
    ctx->merger_norm_w = static_cast<const __nv_bfloat16 *>(norm_w);
    ctx->merger_norm_b = static_cast<const __nv_bfloat16 *>(norm_b);
    ctx->merger_fc1_w = static_cast<const __nv_bfloat16 *>(fc1_w);
    ctx->merger_fc1_b = static_cast<const __nv_bfloat16 *>(fc1_b);
    ctx->merger_fc2_w = static_cast<const __nv_bfloat16 *>(fc2_w);
    ctx->merger_fc2_b = static_cast<const __nv_bfloat16 *>(fc2_b);
}

extern "C" void cosmos3_visual_cuda_set_deepstack_merger(
    cosmos3_visual_cuda_ctx * ctx,
    int index,
    const void * norm_w,
    const void * norm_b,
    const void * fc1_w,
    const void * fc1_b,
    const void * fc2_w,
    const void * fc2_b) {
    if (!ctx || index < 0 || index >= 3) return;
    auto & dst = ctx->deepstack[index];
    dst.norm_w = static_cast<const __nv_bfloat16 *>(norm_w);
    dst.norm_b = static_cast<const __nv_bfloat16 *>(norm_b);
    dst.fc1_w = static_cast<const __nv_bfloat16 *>(fc1_w);
    dst.fc1_b = static_cast<const __nv_bfloat16 *>(fc1_b);
    dst.fc2_w = static_cast<const __nv_bfloat16 *>(fc2_w);
    dst.fc2_b = static_cast<const __nv_bfloat16 *>(fc2_b);
}

extern "C" int cosmos3_visual_cuda_forward_robolab_image(
        cosmos3_visual_cuda_ctx * ctx,
        const unsigned char * image_u8_host,
        int image_h,
        int image_w,
        cudaStream_t stream) {
    if (!ctx || !image_u8_host || image_h <= 0 || image_w <= 0 ||
        !ctx->d_pixel_values || !ctx->d_tokens) {
        return -1;
    }
    const size_t image_bytes = static_cast<size_t>(image_h) * static_cast<size_t>(image_w) * 3u;
    if (image_bytes > ctx->d_image_bytes) {
        if (ctx->d_image) {
            cudaFree(ctx->d_image);
            ctx->d_image = nullptr;
            ctx->d_image_bytes = 0;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&ctx->d_image), image_bytes) != cudaSuccess) {
            return -1;
        }
        ctx->d_image_bytes = image_bytes;
    }
    if (cudaMemcpyAsync(ctx->d_image, image_u8_host, image_bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return -1;
    }
    VisualTimingGuard timing(ctx, stream);
    timing.begin();
    const int rc = cosmos3_visual_robolab_image_to_pixel_values_f32(
        ctx->d_image,
        image_h,
        image_w,
        ctx->d_pixel_values,
        ctx->cfg.patch_rows,
        ctx->cfg.patch_dim,
        stream);
    if (rc != 0) return rc;
    if (!timing.end(VT_IMAGE)) return -1;
    const auto & cfg = ctx->cfg;
    constexpr int block = 256;
    if (!ctx->patch_w || !ctx->patch_b || !ctx->pos_embed ||
        !ctx->merger_norm_w || !ctx->merger_norm_b ||
        !ctx->merger_fc1_w || !ctx->merger_fc1_b ||
        !ctx->merger_fc2_w || !ctx->merger_fc2_b) {
        return -1;
    }
    timing.begin();
    if (!visual_linear_bf16_weight_f32_out(
            ctx, ctx->d_pixel_values, ctx->patch_w, ctx->patch_b,
            ctx->d_x0, cfg.patch_rows, cfg.patch_dim, cfg.hidden,
            "patch_embed", stream)) {
        return -1;
    }
    if (!timing.end(VT_PATCH_EMBED)) return -1;
    timing.begin();
    visual_pos_add_kernel<<<(cfg.patch_rows * cfg.hidden + block - 1) / block, block, 0, stream>>>(
        ctx->d_x0, ctx->pos_embed, ctx->d_coords_yx,
        cfg.patch_rows, cfg.hidden, cfg.grid_h, cfg.grid_w, 48);
    if (!visual_round_activation(ctx, ctx->d_x0, cfg.patch_rows * cfg.hidden, stream)) {
        return -1;
    }
    if (!visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_TOKEN_ENTRY,
                                     ctx->d_x0, cfg.patch_rows, cfg.hidden, 0, stream)) {
        return -1;
    }
    if (!timing.end(VT_POS_ADD)) return -1;

    for (int layer = 0; layer < cfg.blocks; ++layer) {
        const auto & w = ctx->layers[static_cast<size_t>(layer)];
        if (!w.norm1_w || !w.norm1_b || !w.qkv_w || !w.qkv_b ||
            !w.proj_w || !w.proj_b || !w.norm2_w || !w.norm2_b ||
            !w.fc1_w || !w.fc1_b || !w.fc2_w || !w.fc2_b) {
            return -1;
        }
        timing.begin();
        visual_layernorm_kernel<<<cfg.patch_rows, 256, 0, stream>>>(
            ctx->d_x0, w.norm1_w, w.norm1_b, ctx->d_norm, 1e-6f,
            cfg.patch_rows, cfg.hidden);
        if (!visual_round_activation(ctx, ctx->d_norm, cfg.patch_rows * cfg.hidden, stream)) {
            return -1;
        }
        if (layer == 0 &&
            !visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_BLOCK0_NORM1,
                                         ctx->d_norm, cfg.patch_rows, cfg.hidden, 0, stream)) {
            return -1;
        }
        if (!timing.end(VT_NORM1)) return -1;
        timing.begin();
        if (!visual_linear_bf16_weight_f32_out(
                ctx, ctx->d_norm, w.qkv_w, w.qkv_b,
                ctx->d_qkv, cfg.patch_rows, cfg.hidden, 3 * cfg.hidden,
                "block.qkv", stream)) {
            return -1;
        }
        if (layer == 0 &&
            (!visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_BLOCK0_QKV,
                                          ctx->d_qkv, cfg.patch_rows, 3 * cfg.hidden, 0, stream) ||
             !visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_BLOCK0_QKV_K,
                                          ctx->d_qkv, cfg.patch_rows, 3 * cfg.hidden, cfg.hidden, stream) ||
             !visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_BLOCK0_QKV_V,
                                          ctx->d_qkv, cfg.patch_rows, 3 * cfg.hidden, 2 * cfg.hidden, stream))) {
            return -1;
        }
        if (!timing.end(VT_QKV)) return -1;
        timing.begin();
        if (visual_rope_window_attention_qk_cached_f32(
                ctx, ctx->d_qkv, ctx->d_attn, stream) != 0) {
            return -1;
        }
        if (layer == 0 &&
            (!visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_BLOCK0_Q_ROPE,
                                          ctx->d_q_rope, cfg.patch_rows, cfg.hidden, 0, stream) ||
             !visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_BLOCK0_K_ROPE,
                                          ctx->d_k_rope, cfg.patch_rows, cfg.hidden, 0, stream) ||
             !visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_BLOCK0_ATTN,
                                          ctx->d_attn, cfg.patch_rows, cfg.hidden, 0, stream))) {
            return -1;
        }
        if (!timing.end(VT_ATTENTION)) return -1;
        timing.begin();
        if (!visual_linear_bf16_weight_f32_out(
                ctx, ctx->d_attn, w.proj_w, w.proj_b,
                ctx->d_x1, cfg.patch_rows, cfg.hidden, cfg.hidden,
                "block.proj", stream)) {
            return -1;
        }
        if (layer == 0 &&
            !visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_BLOCK0_ATTN_PROJ,
                                         ctx->d_x1, cfg.patch_rows, cfg.hidden, 0, stream)) {
            return -1;
        }
        if (!timing.end(VT_PROJ)) return -1;
        timing.begin();
        visual_add_kernel<<<(cfg.patch_rows * cfg.hidden + block - 1) / block, block, 0, stream>>>(
            ctx->d_x0, ctx->d_x1, ctx->d_x1, cfg.patch_rows * cfg.hidden);
        if (!visual_round_activation(ctx, ctx->d_x1, cfg.patch_rows * cfg.hidden, stream)) {
            return -1;
        }
        if (layer == 0 &&
            !visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_BLOCK0_POST_ATTN,
                                         ctx->d_x1, cfg.patch_rows, cfg.hidden, 0, stream)) {
            return -1;
        }
        if (!timing.end(VT_ATTN_RESIDUAL)) return -1;
        timing.begin();
        visual_layernorm_kernel<<<cfg.patch_rows, 256, 0, stream>>>(
            ctx->d_x1, w.norm2_w, w.norm2_b, ctx->d_norm, 1e-6f,
            cfg.patch_rows, cfg.hidden);
        if (!visual_round_activation(ctx, ctx->d_norm, cfg.patch_rows * cfg.hidden, stream)) {
            return -1;
        }
        if (layer == 0 &&
            !visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_BLOCK0_NORM2,
                                         ctx->d_norm, cfg.patch_rows, cfg.hidden, 0, stream)) {
            return -1;
        }
        if (!timing.end(VT_NORM2)) return -1;
        timing.begin();
        if (!visual_linear_bf16_weight_f32_out(
                ctx, ctx->d_norm, w.fc1_w, w.fc1_b,
                ctx->d_mlp, cfg.patch_rows, cfg.hidden, cfg.intermediate,
                "block.fc1", stream)) {
            return -1;
        }
        if (!timing.end(VT_FC1)) return -1;
        timing.begin();
        visual_gelu_kernel<<<(cfg.patch_rows * cfg.intermediate + block - 1) / block, block, 0, stream>>>(
            ctx->d_mlp, cfg.patch_rows * cfg.intermediate);
        if (!visual_round_activation(ctx, ctx->d_mlp, cfg.patch_rows * cfg.intermediate, stream)) {
            return -1;
        }
        if (layer == 0 &&
            !visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_BLOCK0_MLP,
                                         ctx->d_mlp, cfg.patch_rows, cfg.intermediate, 0, stream)) {
            return -1;
        }
        if (!timing.end(VT_GELU)) return -1;
        timing.begin();
        if (!visual_linear_bf16_weight_f32_out(
                ctx, ctx->d_mlp, w.fc2_w, w.fc2_b,
                ctx->d_x0, cfg.patch_rows, cfg.intermediate, cfg.hidden,
                "block.fc2", stream)) {
            return -1;
        }
        if (!timing.end(VT_FC2)) return -1;
        timing.begin();
        visual_add_kernel<<<(cfg.patch_rows * cfg.hidden + block - 1) / block, block, 0, stream>>>(
            ctx->d_x1, ctx->d_x0, ctx->d_x0, cfg.patch_rows * cfg.hidden);
        if (!visual_round_activation(ctx, ctx->d_x0, cfg.patch_rows * cfg.hidden, stream)) {
            return -1;
        }
        if (layer == 0 &&
            !visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_BLOCK0,
                                         ctx->d_x0, cfg.patch_rows, cfg.hidden, 0, stream)) {
            return -1;
        }
        if (!timing.end(VT_MLP_RESIDUAL)) return -1;
        int deepstack_index = -1;
        if (layer == 8) deepstack_index = 0;
        else if (layer == 16) deepstack_index = 1;
        else if (layer == 24) deepstack_index = 2;
        if (deepstack_index >= 0) {
            timing.begin();
            if (!visual_run_patch_merger_postshuffle_norm(
                    ctx, ctx->d_x0, ctx->deepstack[deepstack_index],
                    ctx->d_deepstack_tokens[deepstack_index],
                    "deepstack_merger", stream)) {
                return -1;
            }
            if (!timing.end(VT_DEEPSTACK)) return -1;
        }
    }

    const int visual_tokens = cfg.patch_rows / (cfg.merge * cfg.merge);
    const int merged_hidden = cfg.hidden * cfg.merge * cfg.merge;
    if (!visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_FINAL_HIDDEN,
                                     ctx->d_x0, cfg.patch_rows, cfg.hidden, 0, stream)) {
        return -1;
    }
    timing.begin();
    visual_layernorm_kernel<<<cfg.patch_rows, 256, 0, stream>>>(
        ctx->d_x0, ctx->merger_norm_w, ctx->merger_norm_b, ctx->d_norm, 1e-6f,
        cfg.patch_rows, cfg.hidden);
    if (!visual_round_activation(ctx, ctx->d_norm, cfg.patch_rows * cfg.hidden, stream)) {
        return -1;
    }
    if (!visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_MERGER_NORM,
                                     ctx->d_norm, cfg.patch_rows, cfg.hidden, 0, stream)) {
        return -1;
    }
    if (cosmos3_visual_merge2x2_reorder_f32(
            ctx->d_norm, ctx->d_merged, cfg.patch_rows, cfg.hidden,
            cfg.merge * cfg.merge, stream) != 0) {
        return -1;
    }
    if (!visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_MERGED,
                                     ctx->d_merged, visual_tokens, merged_hidden, 0, stream)) {
        return -1;
    }
    if (!visual_linear_bf16_weight_f32_out(
            ctx, ctx->d_merged, ctx->merger_fc1_w, ctx->merger_fc1_b,
            ctx->d_merger_h, visual_tokens, merged_hidden, merged_hidden,
            "merger.fc1", stream)) {
        return -1;
    }
    visual_gelu_exact_kernel<<<(visual_tokens * merged_hidden + block - 1) / block, block, 0, stream>>>(
        ctx->d_merger_h, visual_tokens * merged_hidden);
    if (!visual_round_activation(ctx, ctx->d_merger_h, visual_tokens * merged_hidden, stream)) {
        return -1;
    }
    if (!visual_debug_capture_prefix(ctx, COSMOS3_VISUAL_DEBUG_MERGER_H,
                                     ctx->d_merger_h, visual_tokens, merged_hidden, 0, stream)) {
        return -1;
    }
    if (!visual_linear_bf16_weight_f32_out(
            ctx, ctx->d_merger_h, ctx->merger_fc2_w, ctx->merger_fc2_b,
            ctx->d_tokens, visual_tokens, merged_hidden, cfg.output_hidden,
            "merger.fc2", stream)) {
        return -1;
    }
    if (!timing.end(VT_FINAL_MERGER)) return -1;
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" const float * cosmos3_visual_cuda_pixel_values(
        const cosmos3_visual_cuda_ctx * ctx) {
    return ctx ? ctx->d_pixel_values : nullptr;
}

extern "C" float * cosmos3_visual_cuda_tokens(cosmos3_visual_cuda_ctx * ctx) {
    return ctx ? ctx->d_tokens : nullptr;
}

extern "C" float * cosmos3_visual_cuda_deepstack_tokens(cosmos3_visual_cuda_ctx * ctx, int index) {
    if (!ctx || index < 0 || index >= 3) return nullptr;
    return ctx->d_deepstack_tokens[index];
}

extern "C" void cosmos3_visual_cuda_set_debug_prefix(
        cosmos3_visual_cuda_ctx * ctx,
        int enabled,
        int rows,
        int cols) {
    if (!ctx) return;
    if (!enabled || rows <= 0 || cols <= 0) {
        visual_debug_release(ctx);
        return;
    }
    rows = std::max(1, std::min(rows, ctx->cfg.patch_rows));
    cols = std::max(1, std::min(cols, std::max(ctx->cfg.intermediate,
                                               ctx->cfg.hidden * ctx->cfg.merge * ctx->cfg.merge)));
    if (ctx->debug_enabled && ctx->debug_rows == rows && ctx->debug_cols == cols) {
        return;
    }
    visual_debug_release(ctx);
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
    for (float *& ptr : ctx->d_debug_prefix) {
        if (cudaMalloc(reinterpret_cast<void **>(&ptr), bytes) != cudaSuccess) {
            visual_debug_release(ctx);
            return;
        }
        cudaMemset(ptr, 0, bytes);
    }
    ctx->debug_rows = rows;
    ctx->debug_cols = cols;
    ctx->debug_enabled = true;
}

extern "C" float * cosmos3_visual_cuda_debug_prefix(cosmos3_visual_cuda_ctx * ctx, int index) {
    if (!ctx || index < 0 || index >= COSMOS3_VISUAL_DEBUG_COUNT || !ctx->debug_enabled) return nullptr;
    return ctx->d_debug_prefix[static_cast<size_t>(index)];
}

extern "C" void cosmos3_visual_cuda_set_timing(cosmos3_visual_cuda_ctx * ctx, int enabled) {
    if (!ctx) return;
    ctx->timing_enabled = enabled != 0;
    if (!ctx->timing_enabled) {
        std::fill(std::begin(ctx->timing_ms), std::end(ctx->timing_ms), 0.0f);
    }
}

extern "C" int cosmos3_visual_cuda_copy_timing_ms(const cosmos3_visual_cuda_ctx * ctx, float * values, int count) {
    if (!ctx || !values || count < VT_COUNT) return -1;
    std::copy(ctx->timing_ms, ctx->timing_ms + VT_COUNT, values);
    return VT_COUNT;
}

extern "C" int cosmos3_visual_rope_window_attention_f32(
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
    cudaStream_t stream) {
    if (!qkv || !rotary_coords_yx || !window_offsets || !y ||
        tokens <= 0 || windows <= 0 || heads <= 0 || head_dim <= 0 ||
        head_dim % 4 != 0 || !(rope_theta > 1.0f)) {
        return -1;
    }
    if (!token_to_window && windows != 1) return -1;
    const dim3 grid(tokens * heads);
    const dim3 block(128);
    int max_window = 0;
    std::vector<int> h_offsets(static_cast<size_t>(windows) + 1u, 0);
    if (cudaMemcpyAsync(h_offsets.data(), window_offsets,
                        h_offsets.size() * sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return -1;
    }
    for (int w = 0; w < windows; ++w) {
        max_window = std::max(max_window, h_offsets[static_cast<size_t>(w + 1)] - h_offsets[static_cast<size_t>(w)]);
    }
    if (max_window <= 0) return -1;
    const size_t smem = static_cast<size_t>(head_dim + max_window + 128) * sizeof(float);
    visual_window_attention_block_kernel<<<grid, block, smem, stream>>>(
        qkv, nullptr, nullptr, rotary_coords_yx, window_offsets, token_to_window, y,
        tokens, windows, heads, head_dim, max_window, rope_theta);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_visual_merge2x2_reorder_f32(
    const float * x,
    float * y,
    int patch_rows,
    int hidden,
    int merge,
    cudaStream_t stream) {
    if (!x || !y || patch_rows <= 0 || hidden <= 0 || merge <= 0 || patch_rows % merge != 0) {
        return -1;
    }
    const int total = (patch_rows / merge) * hidden * merge;
    constexpr int block = 256;
    merge2x2_reorder_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        x, y, patch_rows, hidden, merge);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_visual_robolab_image_to_pixel_values_f32(
    const unsigned char * image_u8,
    int image_h,
    int image_w,
    float * pixel_values,
    int patch_rows,
    int patch_dim,
    cudaStream_t stream) {
    if (!image_u8 || !pixel_values || image_h <= 0 || image_w <= 0 ||
        patch_rows != 26588 || patch_dim != 1536) {
        return -1;
    }
    constexpr int block = 256;
    const int total = patch_rows * patch_dim;
    robolab_image_to_pixel_values_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        image_u8, image_h, image_w, pixel_values, patch_rows, patch_dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" const char * cosmos3_qwen_visual_kernel_status(void) {
    return "cosmos3-qwen-visual-cuda-ok ops=robolab_image_to_pixel_values,rope_window_attention,merge2x2_reorder";
}
