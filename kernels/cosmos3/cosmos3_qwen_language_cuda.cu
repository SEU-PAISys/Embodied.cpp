// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#include "cosmos3_qwen_language_cuda.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math_constants.h>

namespace {

__device__ float bf16_u16_to_f32(unsigned short bits) {
    __nv_bfloat16 v;
    *reinterpret_cast<unsigned short *>(&v) = bits;
    return __bfloat162float(v);
}

__device__ float round_to_bf16_value(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

__global__ void rmsnorm_kernel(const float * x,
                               const unsigned short * gamma,
                               float * y,
                               int tokens,
                               int hidden,
                               float eps) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    if (token >= tokens) return;
    extern __shared__ float scratch[];
    float sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        const float v = x[token * hidden + i];
        sum += v * v;
    }
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / static_cast<float>(hidden) + eps);
    for (int i = tid; i < hidden; i += blockDim.x) {
        const float normed = round_to_bf16_value(x[token * hidden + i] * inv);
        y[token * hidden + i] = round_to_bf16_value(normed * bf16_u16_to_f32(gamma[i]));
    }
}

__global__ void head_rmsnorm_kernel(const float * x,
                                    const unsigned short * gamma,
                                    float * y,
                                    int tokens,
                                    int width,
                                    int heads,
                                    int head_dim,
                                    float eps) {
    const int item = blockIdx.x;
    const int token = item / heads;
    const int head = item - token * heads;
    const int tid = threadIdx.x;
    if (token >= tokens || head >= heads) return;
    extern __shared__ float scratch[];
    const int base = token * width + head * head_dim;
    float sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        const float v = x[base + i];
        sum += v * v;
    }
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / static_cast<float>(head_dim) + eps);
    for (int i = tid; i < head_dim; i += blockDim.x) {
        const float normed = round_to_bf16_value(x[base + i] * inv);
        y[base + i] = round_to_bf16_value(normed * bf16_u16_to_f32(gamma[i]));
    }
}

__global__ void rope_kernel(const float * x,
                            float * y,
                            int tokens,
                            int width,
                            int heads,
                            int head_dim,
                            float rope_theta,
                            int first_position) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * width;
    if (idx >= total) return;
    const int d = idx % head_dim;
    const int head_offset = idx % width;
    const int token = idx / width;
    const int base = idx - d;
    const int half = head_dim / 2;
    const int pair_dim = d < half ? d : d - half;
    const int dim_index = pair_dim * 2;
    const float inv_freq = powf(rope_theta, -static_cast<float>(dim_index) / static_cast<float>(head_dim));
    const float angle = static_cast<float>(first_position + token) * inv_freq;
    float s;
    float c;
    sincosf(angle, &s, &c);
    s = round_to_bf16_value(s);
    c = round_to_bf16_value(c);
    const int rd = d < half ? d + half : d - half;
    const float rot_half = d < half ? -x[base + rd] : x[base + rd];
    (void) heads;
    (void) head_offset;
    y[idx] = round_to_bf16_value(x[idx] * c + rot_half * s);
}

__global__ void mrope_kernel(const float * x,
                             const int * position_ids,
                             float * y,
                             int tokens,
                             int width,
                             int heads,
                             int head_dim,
                             float rope_theta) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * width;
    if (idx >= total) return;
    const int d = idx % head_dim;
    const int token = idx / width;
    const int base = idx - d;
    const int half = head_dim / 2;
    const int pair_dim = d < half ? d : d - half;
    const int dim_index = pair_dim * 2;
    int pos_axis = 0;
    if (pair_dim < 60 && (pair_dim % 3) == 1) {
        pos_axis = 1;
    } else if (pair_dim < 60 && (pair_dim % 3) == 2) {
        pos_axis = 2;
    }
    const int position = position_ids[pos_axis * tokens + token];
    const float inv_freq = powf(rope_theta, -static_cast<float>(dim_index) / static_cast<float>(head_dim));
    const float angle = static_cast<float>(position) * inv_freq;
    float s;
    float c;
    sincosf(angle, &s, &c);
    s = round_to_bf16_value(s);
    c = round_to_bf16_value(c);
    const int rd = d < half ? d + half : d - half;
    const float rot_half = d < half ? -x[base + rd] : x[base + rd];
    (void) heads;
    y[idx] = round_to_bf16_value(x[idx] * c + rot_half * s);
}

__global__ void mrope_floatpos_kernel(const float * x,
                                      const float * position_ids,
                                      float * y,
                                      int tokens,
                                      int width,
                                      int heads,
                                      int head_dim,
                                      float rope_theta) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * width;
    if (idx >= total) return;
    const int d = idx % head_dim;
    const int token = idx / width;
    const int base = idx - d;
    const int half = head_dim / 2;
    const int pair_dim = d < half ? d : d - half;
    const int dim_index = pair_dim * 2;
    int pos_axis = 0;
    if (pair_dim < 60 && (pair_dim % 3) == 1) {
        pos_axis = 1;
    } else if (pair_dim < 60 && (pair_dim % 3) == 2) {
        pos_axis = 2;
    }
    const float position = position_ids[pos_axis * tokens + token];
    const float inv_freq = powf(rope_theta, -static_cast<float>(dim_index) / static_cast<float>(head_dim));
    const float angle = position * inv_freq;
    float s;
    float c;
    sincosf(angle, &s, &c);
    s = round_to_bf16_value(s);
    c = round_to_bf16_value(c);
    const int rd = d < half ? d + half : d - half;
    const float rot_half = d < half ? -x[base + rd] : x[base + rd];
    (void) heads;
    y[idx] = round_to_bf16_value(x[idx] * c + rot_half * s);
}

__global__ void head_rmsnorm_mrope_floatpos_kernel(const float * x,
                                                   const unsigned short * gamma,
                                                   const float * position_ids,
                                                   float * y,
                                                   int tokens,
                                                   int width,
                                                   int heads,
                                                   int head_dim,
                                                   float eps,
                                                   float rope_theta) {
    const int item = blockIdx.x;
    const int token = item / heads;
    const int head = item - token * heads;
    const int tid = threadIdx.x;
    if (token >= tokens || head >= heads || tid >= head_dim) return;
    extern __shared__ float scratch[];
    const int base = token * width + head * head_dim;
    const int half = head_dim / 2;
    float sum = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        const float v = x[base + i];
        sum += v * v;
    }
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / static_cast<float>(head_dim) + eps);
    const int rd = tid < half ? tid + half : tid - half;
    const float normed = round_to_bf16_value(x[base + tid] * inv);
    const float peer_normed = round_to_bf16_value(x[base + rd] * inv);
    const float value = round_to_bf16_value(normed * bf16_u16_to_f32(gamma[tid]));
    const float peer_value = round_to_bf16_value(peer_normed * bf16_u16_to_f32(gamma[rd]));
    const int pair_dim = tid < half ? tid : tid - half;
    const int dim_index = pair_dim * 2;
    int pos_axis = 0;
    if (pair_dim < 60 && (pair_dim % 3) == 1) {
        pos_axis = 1;
    } else if (pair_dim < 60 && (pair_dim % 3) == 2) {
        pos_axis = 2;
    }
    const float position = position_ids[pos_axis * tokens + token];
    const float inv_freq = powf(rope_theta, -static_cast<float>(dim_index) / static_cast<float>(head_dim));
    const float angle = position * inv_freq;
    float s;
    float c;
    sincosf(angle, &s, &c);
    s = round_to_bf16_value(s);
    c = round_to_bf16_value(c);
    const float rot_half = tid < half ? -peer_value : peer_value;
    y[base + tid] = round_to_bf16_value(value * c + rot_half * s);
}

__global__ void causal_gqa_attention_online_kernel(const float * q,
                                                   const float * k,
                                                   const float * v,
                                                   float * y,
                                                   int tokens,
                                                   int hidden,
                                                   int kv,
                                                   int q_heads,
                                                   int kv_heads,
                                                   int head_dim) {
    const int item = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    const int token = item / q_heads;
    const int qh = item - token * q_heads;
    if (token >= tokens) return;
    const int q_per_kv = q_heads / kv_heads;
    const int kvh = qh / q_per_kv;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float scratch[];
    __shared__ float max_score;
    __shared__ float denom;
    __shared__ float score;

    if (d == 0) max_score = -CUDART_INF_F;
    if (d == 0) denom = 0.0f;
    __syncthreads();
    float acc = 0.0f;
    for (int tk = 0; tk <= token; ++tk) {
        scratch[d] = q[token * hidden + qh * head_dim + d] *
                     k[tk * kv + kvh * head_dim + d];
        __syncthreads();
        for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
            if (d < stride) scratch[d] += scratch[d + stride];
            __syncthreads();
        }
        if (d == 0) {
            const float s = scratch[0] * scale;
            const float next_max = fmaxf(max_score, s);
            const float old_scale = denom == 0.0f ? 0.0f : expf(max_score - next_max);
            const float new_scale = expf(s - next_max);
            denom = denom * old_scale + new_scale;
            score = new_scale;
            max_score = next_max;
        }
        __syncthreads();
        const float old_scale = denom == 0.0f ? 0.0f : (denom - score) / denom;
        const float p = score / denom;
        acc = acc * old_scale + p * v[tk * kv + kvh * head_dim + d];
        __syncthreads();
    }
    y[token * hidden + qh * head_dim + d] = round_to_bf16_value(acc);
}

__global__ void causal_softmax_v_from_scores_kernel(const float * scores,
                                                    const float * v,
                                                    float * y,
                                                    int qh,
                                                    int kvh,
                                                    int tokens,
                                                    int hidden,
                                                    int kv,
                                                    int head_dim) {
    const int token = blockIdx.x;
    const int d = threadIdx.x;
    if (token >= tokens || d >= head_dim) return;
    float max_score = -CUDART_INF_F;
    float denom = 0.0f;
    float acc = 0.0f;
    for (int tk = 0; tk <= token; ++tk) {
        const float s = scores[tk + token * tokens];
        const float next_max = fmaxf(max_score, s);
        const float old_scale = denom == 0.0f ? 0.0f : expf(max_score - next_max);
        const float new_scale = expf(s - next_max);
        const float next_denom = denom * old_scale + new_scale;
        const float old_weight = next_denom == 0.0f ? 0.0f : (denom * old_scale) / next_denom;
        const float new_weight = next_denom == 0.0f ? 0.0f : new_scale / next_denom;
        acc = acc * old_weight + new_weight * v[tk * kv + kvh * head_dim + d];
        max_score = next_max;
        denom = next_denom;
    }
    y[token * hidden + qh * head_dim + d] = round_to_bf16_value(acc);
}

__global__ void causal_softmax_inplace_scores_kernel(float * scores,
                                                     int tokens) {
    const int token = blockIdx.x;
    const int d = threadIdx.x;
    if (token >= tokens) return;
    float * row = scores + static_cast<size_t>(token) * tokens;
    extern __shared__ float scratch[];

    float local_max = -CUDART_INF_F;
    for (int tk = d; tk <= token; tk += blockDim.x) {
        local_max = fmaxf(local_max, row[tk]);
    }
    scratch[d] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride) scratch[d] = fmaxf(scratch[d], scratch[d + stride]);
        __syncthreads();
    }
    const float max_score = scratch[0];

    float denom = 0.0f;
    for (int tk = d; tk <= token; tk += blockDim.x) {
        const float w = expf(row[tk] - max_score);
        row[tk] = w;
        denom += w;
    }
    scratch[d] = denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride) scratch[d] += scratch[d + stride];
        __syncthreads();
    }
    const float inv_denom = scratch[0] == 0.0f ? 0.0f : 1.0f / scratch[0];
    for (int tk = d; tk <= token; tk += blockDim.x) {
        row[tk] *= inv_denom;
    }
    for (int tk = token + 1 + d; tk < tokens; tk += blockDim.x) {
        row[tk] = 0.0f;
    }
}

__global__ void gen_full_gqa_attention_online_kernel(const float * q,
                                                     const float * k,
                                                     const float * v,
                                                     float * y,
                                                     int gen_tokens,
                                                     int all_tokens,
                                                     int hidden,
                                                     int kv,
                                                     int q_heads,
                                                     int kv_heads,
                                                     int head_dim) {
    const int item = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    const int token = item / q_heads;
    const int qh = item - token * q_heads;
    if (token >= gen_tokens) return;
    const int q_per_kv = q_heads / kv_heads;
    const int kvh = qh / q_per_kv;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    extern __shared__ float scratch[];
    __shared__ float max_score;
    __shared__ float denom;
    __shared__ float score;

    if (d == 0) max_score = -CUDART_INF_F;
    if (d == 0) denom = 0.0f;
    __syncthreads();
    float acc = 0.0f;
    for (int tk = 0; tk < all_tokens; ++tk) {
        scratch[d] = q[token * hidden + qh * head_dim + d] *
                     k[tk * kv + kvh * head_dim + d];
        __syncthreads();
        for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
            if (d < stride) scratch[d] += scratch[d + stride];
            __syncthreads();
        }
        if (d == 0) {
            const float s = scratch[0] * scale;
            const float next_max = fmaxf(max_score, s);
            const float old_scale = denom == 0.0f ? 0.0f : expf(max_score - next_max);
            const float new_scale = expf(s - next_max);
            denom = denom * old_scale + new_scale;
            score = new_scale;
            max_score = next_max;
        }
        __syncthreads();
        const float old_scale = denom == 0.0f ? 0.0f : (denom - score) / denom;
        const float p = score / denom;
        acc = acc * old_scale + p * v[tk * kv + kvh * head_dim + d];
        __syncthreads();
    }
    y[token * hidden + qh * head_dim + d] = round_to_bf16_value(acc);
}

__global__ void gen_full_softmax_inplace_scores_kernel(float * scores_group,
                                                       int q_per_kv,
                                                       int gen_tokens,
                                                       int all_tokens) {
    const int row = blockIdx.x;
    const int d = threadIdx.x;
    const int rows = q_per_kv * gen_tokens;
    if (row >= rows) return;
    const int local_q = row / gen_tokens;
    const int token = row - local_q * gen_tokens;
    const size_t score_elems = static_cast<size_t>(gen_tokens) * static_cast<size_t>(all_tokens);
    float * scores = scores_group + static_cast<size_t>(local_q) * score_elems +
                     static_cast<size_t>(token) * all_tokens;
    extern __shared__ float scratch[];

    float local_max = -CUDART_INF_F;
    for (int tk = d; tk < all_tokens; tk += blockDim.x) {
        local_max = fmaxf(local_max, scores[tk]);
    }
    scratch[d] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride) scratch[d] = fmaxf(scratch[d], scratch[d + stride]);
        __syncthreads();
    }
    const float max_score = scratch[0];

    float denom = 0.0f;
    for (int tk = d; tk < all_tokens; tk += blockDim.x) {
        const float w = expf(scores[tk] - max_score);
        scores[tk] = w;
        denom += w;
    }
    scratch[d] = denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (d < stride) scratch[d] += scratch[d + stride];
        __syncthreads();
    }
    const float inv_denom = scratch[0] == 0.0f ? 0.0f : 1.0f / scratch[0];
    for (int tk = d; tk < all_tokens; tk += blockDim.x) {
        scores[tk] *= inv_denom;
    }
}

__global__ void round_attention_q_heads_kernel(float * y,
                                               int qh0,
                                               int q_count,
                                               int gen_tokens,
                                               int hidden,
                                               int head_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gen_tokens * q_count * head_dim;
    if (idx >= total) return;
    const int d = idx % head_dim;
    const int q = (idx / head_dim) % q_count;
    const int token = idx / (q_count * head_dim);
    float * ptr = y + static_cast<size_t>(token) * hidden + static_cast<size_t>(qh0 + q) * head_dim + d;
    *ptr = round_to_bf16_value(*ptr);
}

__global__ void swiglu_kernel(const float * gate, const float * up, float * y, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const float g = gate[idx];
    y[idx] = round_to_bf16_value((g / (1.0f + expf(-g))) * up[idx]);
}

__global__ void residual_add_kernel(const float * a, const float * b, float * y, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    y[idx] = round_to_bf16_value(a[idx] + b[idx]);
}

__global__ void residual_add_rmsnorm_kernel(const float * residual_input,
                                            const float * branch,
                                            const unsigned short * gamma,
                                            float * residual,
                                            float * norm,
                                            int tokens,
                                            int hidden,
                                            float eps) {
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    if (token >= tokens) return;
    extern __shared__ float scratch[];
    const int base = token * hidden;
    float sum = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x) {
        const float v = round_to_bf16_value(residual_input[base + i] + branch[base + i]);
        residual[base + i] = v;
        sum += v * v;
    }
    scratch[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / static_cast<float>(hidden) + eps);
    for (int i = tid; i < hidden; i += blockDim.x) {
        const float add_value = residual[base + i];
        const float normed = round_to_bf16_value(add_value * inv);
        norm[base + i] = round_to_bf16_value(normed * bf16_u16_to_f32(gamma[i]));
    }
}

__global__ void build_robolab_input_kernel(const int * input_ids,
                                           const int * visual_indices,
                                           const unsigned short * embed_tokens,
                                           const float * visual_tokens,
                                           float * y,
                                           int tokens,
                                           int hidden,
                                           int vocab,
                                           int visual_token_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * hidden;
    if (idx >= total) return;
    const int token = idx / hidden;
    const int col = idx - token * hidden;
    const int visual_index = visual_indices[token];
    if (visual_index >= 0) {
        if (visual_index < visual_token_count) {
            y[idx] = visual_tokens[visual_index * hidden + col];
        } else {
            y[idx] = 0.0f;
        }
        return;
    }
    const int token_id = input_ids[token];
    if (token_id < 0 || token_id >= vocab) {
        y[idx] = 0.0f;
        return;
    }
    y[idx] = bf16_u16_to_f32(embed_tokens[token_id * hidden + col]);
}

__global__ void embed_tokens_kernel(const int * input_ids,
                                    const unsigned short * embed_tokens,
                                    float * y,
                                    int tokens,
                                    int hidden,
                                    int vocab) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * hidden;
    if (idx >= total) return;
    const int token = idx / hidden;
    const int col = idx - token * hidden;
    const int token_id = input_ids[token];
    if (token_id < 0 || token_id >= vocab) {
        y[idx] = 0.0f;
        return;
    }
    y[idx] = bf16_u16_to_f32(embed_tokens[token_id * hidden + col]);
}

__global__ void add_visual_deepstack_kernel(float * hidden_states,
                                            const int * visual_indices,
                                            const float * deepstack_tokens,
                                            int tokens,
                                            int hidden,
                                            int visual_token_count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * hidden;
    if (idx >= total) return;
    const int token = idx / hidden;
    const int col = idx - token * hidden;
    const int visual_index = visual_indices[token];
    if (visual_index >= 0 && visual_index < visual_token_count) {
        hidden_states[idx] += deepstack_tokens[visual_index * hidden + col];
    }
}

__global__ void f32_to_bf16_kernel(const float * x,
                                   unsigned short * y,
                                   int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const __nv_bfloat16 v = __float2bfloat16(x[idx]);
    y[idx] = *reinterpret_cast<const unsigned short *>(&v);
}

} // namespace

namespace {

cublasHandle_t get_language_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return handle;
}

int causal_gqa_attention_cublas_scores_f32(const float * q,
                                           const float * k,
                                           const float * v,
                                           float * y,
                                           int tokens,
                                           int hidden,
                                           int kv,
                                           int q_heads,
                                           int kv_heads,
                                           int head_dim,
                                           cudaStream_t stream) {
    float * scores = nullptr;
    const size_t score_elems = static_cast<size_t>(tokens) * static_cast<size_t>(tokens);
    if (cudaMallocAsync(reinterpret_cast<void **>(&scores), score_elems * sizeof(float), stream) != cudaSuccess) {
        return -1;
    }
    cublasHandle_t handle = get_language_cublas_handle();
    if (!handle || cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
        cudaFreeAsync(scores, stream);
        return -1;
    }
    const float alpha = rsqrtf(static_cast<float>(head_dim));
    const float beta = 0.0f;
    const int q_per_kv = q_heads / kv_heads;
    for (int qh = 0; qh < q_heads; ++qh) {
        const int kvh = qh / q_per_kv;
        const float * q_head = q + qh * head_dim;
        const float * k_head = k + kvh * head_dim;
        const cublasStatus_t st = cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            tokens,
            tokens,
            head_dim,
            &alpha,
            k_head,
            CUDA_R_32F,
            kv,
            q_head,
            CUDA_R_32F,
            hidden,
            &beta,
            scores,
            CUDA_R_32F,
            tokens,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            cudaFreeAsync(scores, stream);
            return -1;
        }
        constexpr int softmax_block = 256;
        causal_softmax_inplace_scores_kernel<<<tokens,
                                               softmax_block,
                                               softmax_block * sizeof(float),
                                               stream>>>(scores, tokens);
        if (cudaGetLastError() != cudaSuccess) {
            cudaFreeAsync(scores, stream);
            return -1;
        }
        const float value_alpha = 1.0f;
        const float value_beta = 0.0f;
        const float * v_head = v + kvh * head_dim;
        float * y_head = y + qh * head_dim;
        const cublasStatus_t vst = cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            head_dim,
            tokens,
            tokens,
            &value_alpha,
            v_head,
            CUDA_R_32F,
            kv,
            scores,
            CUDA_R_32F,
            tokens,
            &value_beta,
            y_head,
            CUDA_R_32F,
            hidden,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (vst != CUBLAS_STATUS_SUCCESS) {
            cudaFreeAsync(scores, stream);
            return -1;
        }
        round_attention_q_heads_kernel<<<(tokens * head_dim + 255) / 256,
                                         256,
                                         0,
                                         stream>>>(y, qh, 1, tokens, hidden, head_dim);
        if (cudaGetLastError() != cudaSuccess) {
            cudaFreeAsync(scores, stream);
            return -1;
        }
    }
    cudaFreeAsync(scores, stream);
    return 0;
}

int gen_full_gqa_attention_cublas_scores_workspace_f32(const float * q,
                                                       const float * k,
                                                       const float * v,
                                                       float * y,
                                                       float * scores_group,
                                                       int gen_tokens,
                                                       int all_tokens,
                                                       int hidden,
                                                       int kv,
                                                       int q_heads,
                                                       int kv_heads,
                                                       int head_dim,
                                                       cudaStream_t stream) {
    if (!scores_group) return -1;
    cudaGetLastError();
    const size_t score_elems = static_cast<size_t>(gen_tokens) * static_cast<size_t>(all_tokens);
    const int q_per_kv = q_heads / kv_heads;
    cublasHandle_t handle = get_language_cublas_handle();
    if (!handle || cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    const float score_alpha = rsqrtf(static_cast<float>(head_dim));
    const float gemm_alpha = 1.0f;
    const float beta = 0.0f;
    for (int kvh = 0; kvh < kv_heads; ++kvh) {
        const int qh0 = kvh * q_per_kv;
        const float * q_head0 = q + qh0 * head_dim;
        const float * k_head = k + kvh * head_dim;
        const cublasStatus_t score_st = cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            all_tokens,
            gen_tokens,
            head_dim,
            &score_alpha,
            k_head,
            CUDA_R_32F,
            kv,
            0,
            q_head0,
            CUDA_R_32F,
            hidden,
            head_dim,
            &beta,
            scores_group,
            CUDA_R_32F,
            all_tokens,
            static_cast<long long>(score_elems),
            q_per_kv,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (score_st != CUBLAS_STATUS_SUCCESS) {
            return -1;
        }
        constexpr int softmax_block = 256;
        gen_full_softmax_inplace_scores_kernel<<<q_per_kv * gen_tokens,
                                                 softmax_block,
                                                 softmax_block * sizeof(float),
                                                 stream>>>(
            scores_group, q_per_kv, gen_tokens, all_tokens);
        if (cudaGetLastError() != cudaSuccess) {
            return -1;
        }
        const float * v_head = v + static_cast<size_t>(kvh) * head_dim;
        float * y_head0 = y + static_cast<size_t>(qh0) * head_dim;
        const cublasStatus_t value_st = cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            head_dim,
            gen_tokens,
            all_tokens,
            &gemm_alpha,
            v_head,
            CUDA_R_32F,
            kv,
            0,
            scores_group,
            CUDA_R_32F,
            all_tokens,
            static_cast<long long>(score_elems),
            &beta,
            y_head0,
            CUDA_R_32F,
            hidden,
            head_dim,
            q_per_kv,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (value_st != CUBLAS_STATUS_SUCCESS) {
            return -1;
        }
        const int round_total = gen_tokens * q_per_kv * head_dim;
        round_attention_q_heads_kernel<<<(round_total + 255) / 256, 256, 0, stream>>>(
            y, qh0, q_per_kv, gen_tokens, hidden, head_dim);
        if (cudaGetLastError() != cudaSuccess) {
            return -1;
        }
    }
    return 0;
}

int gen_full_gqa_attention_bf16tc_workspace_f32(const float * q,
                                                const float * k,
                                                const float * v,
                                                float * y,
                                                float * scores_group,
                                                unsigned short * bf16_workspace,
                                                int gen_tokens,
                                                int all_tokens,
                                                int hidden,
                                                int kv,
                                                int q_heads,
                                                int kv_heads,
                                                int head_dim,
                                                cudaStream_t stream) {
    if (!scores_group || !bf16_workspace) return -1;
    cudaGetLastError();
    const size_t q_elems = static_cast<size_t>(gen_tokens) * hidden;
    const size_t k_elems = static_cast<size_t>(all_tokens) * kv;
    const size_t v_elems = static_cast<size_t>(all_tokens) * kv;
    unsigned short * q_bf16 = bf16_workspace;
    unsigned short * k_bf16 = q_bf16 + q_elems;
    unsigned short * v_bf16 = k_bf16 + k_elems;
    constexpr int block = 256;
    f32_to_bf16_kernel<<<static_cast<unsigned int>((q_elems + block - 1) / block), block, 0, stream>>>(
        q, q_bf16, static_cast<int>(q_elems));
    f32_to_bf16_kernel<<<static_cast<unsigned int>((k_elems + block - 1) / block), block, 0, stream>>>(
        k, k_bf16, static_cast<int>(k_elems));
    f32_to_bf16_kernel<<<static_cast<unsigned int>((v_elems + block - 1) / block), block, 0, stream>>>(
        v, v_bf16, static_cast<int>(v_elems));
    if (cudaGetLastError() != cudaSuccess) return -1;

    const size_t score_elems = static_cast<size_t>(gen_tokens) * static_cast<size_t>(all_tokens);
    const int q_per_kv = q_heads / kv_heads;
    cublasHandle_t handle = get_language_cublas_handle();
    if (!handle || cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    const float score_alpha = rsqrtf(static_cast<float>(head_dim));
    const float gemm_alpha = 1.0f;
    const float beta = 0.0f;
    for (int kvh = 0; kvh < kv_heads; ++kvh) {
        const int qh0 = kvh * q_per_kv;
        const unsigned short * q_head0 = q_bf16 + static_cast<size_t>(qh0) * head_dim;
        const unsigned short * k_head = k_bf16 + static_cast<size_t>(kvh) * head_dim;
        const cublasStatus_t score_st = cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            all_tokens,
            gen_tokens,
            head_dim,
            &score_alpha,
            k_head,
            CUDA_R_16BF,
            kv,
            0,
            q_head0,
            CUDA_R_16BF,
            hidden,
            head_dim,
            &beta,
            scores_group,
            CUDA_R_32F,
            all_tokens,
            static_cast<long long>(score_elems),
            q_per_kv,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (score_st != CUBLAS_STATUS_SUCCESS) return -1;
        constexpr int softmax_block = 256;
        gen_full_softmax_inplace_scores_kernel<<<q_per_kv * gen_tokens,
                                                 softmax_block,
                                                 softmax_block * sizeof(float),
                                                 stream>>>(
            scores_group, q_per_kv, gen_tokens, all_tokens);
        if (cudaGetLastError() != cudaSuccess) return -1;
        const unsigned short * v_head = v_bf16 + static_cast<size_t>(kvh) * head_dim;
        float * y_head0 = y + static_cast<size_t>(qh0) * head_dim;
        const cublasStatus_t value_st = cublasGemmStridedBatchedEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            head_dim,
            gen_tokens,
            all_tokens,
            &gemm_alpha,
            v_head,
            CUDA_R_16BF,
            kv,
            0,
            scores_group,
            CUDA_R_32F,
            all_tokens,
            static_cast<long long>(score_elems),
            &beta,
            y_head0,
            CUDA_R_32F,
            hidden,
            head_dim,
            q_per_kv,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (value_st != CUBLAS_STATUS_SUCCESS) return -1;
        const int round_total = gen_tokens * q_per_kv * head_dim;
        round_attention_q_heads_kernel<<<(round_total + 255) / 256, 256, 0, stream>>>(
            y, qh0, q_per_kv, gen_tokens, hidden, head_dim);
        if (cudaGetLastError() != cudaSuccess) return -1;
    }
    return 0;
}

int gen_full_gqa_attention_cublas_scores_f32(const float * q,
                                             const float * k,
                                             const float * v,
                                             float * y,
                                             int gen_tokens,
                                             int all_tokens,
                                             int hidden,
                                             int kv,
                                             int q_heads,
                                             int kv_heads,
                                             int head_dim,
                                             cudaStream_t stream) {
    float * scores_group = nullptr;
    const size_t score_elems = static_cast<size_t>(gen_tokens) * static_cast<size_t>(all_tokens);
    const int q_per_kv = q_heads / kv_heads;
    const size_t group_score_elems = score_elems * static_cast<size_t>(q_per_kv);
    if (cudaMallocAsync(reinterpret_cast<void **>(&scores_group),
                        group_score_elems * sizeof(float),
                        stream) != cudaSuccess) {
        return -1;
    }
    const int rc = gen_full_gqa_attention_cublas_scores_workspace_f32(
        q, k, v, y, scores_group,
        gen_tokens, all_tokens, hidden, kv, q_heads, kv_heads, head_dim,
        stream);
    cudaFreeAsync(scores_group, stream);
    return rc;
}

} // namespace

extern "C" int cosmos3_qwen_rmsnorm_f32(
    const float * x,
    const unsigned short * gamma_bf16,
    float * y,
    int tokens,
    int hidden,
    float eps,
    cudaStream_t stream) {
    if (!x || !gamma_bf16 || !y || tokens <= 0 || hidden <= 0) return -1;
    constexpr int block = 256;
    rmsnorm_kernel<<<tokens, block, block * sizeof(float), stream>>>(x, gamma_bf16, y, tokens, hidden, eps);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_head_rmsnorm_f32(
    const float * x,
    const unsigned short * gamma_bf16,
    float * y,
    int tokens,
    int width,
    int heads,
    int head_dim,
    float eps,
    cudaStream_t stream) {
    if (!x || !gamma_bf16 || !y || tokens <= 0 || width <= 0 || heads <= 0 || head_dim <= 0) return -1;
    constexpr int block = 128;
    head_rmsnorm_kernel<<<tokens * heads, block, block * sizeof(float), stream>>>(
        x, gamma_bf16, y, tokens, width, heads, head_dim, eps);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_rope_f32(
    const float * x,
    float * y,
    int tokens,
    int width,
    int heads,
    int head_dim,
    float rope_theta,
    int first_position,
    cudaStream_t stream) {
    if (!x || !y || tokens <= 0 || width <= 0 || heads <= 0 || head_dim <= 0) return -1;
    const int total = tokens * width;
    constexpr int block = 256;
    rope_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        x, y, tokens, width, heads, head_dim, rope_theta, first_position);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_mrope_f32(
    const float * x,
    const int * position_ids,
    float * y,
    int tokens,
    int width,
    int heads,
    int head_dim,
    float rope_theta,
    cudaStream_t stream) {
    if (!x || !position_ids || !y || tokens <= 0 || width <= 0 || heads <= 0 || head_dim <= 0) return -1;
    const int total = tokens * width;
    constexpr int block = 256;
    mrope_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        x, position_ids, y, tokens, width, heads, head_dim, rope_theta);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_mrope_pos_f32(
    const float * x,
    const float * position_ids,
    float * y,
    int tokens,
    int width,
    int heads,
    int head_dim,
    float rope_theta,
    cudaStream_t stream) {
    if (!x || !position_ids || !y || tokens <= 0 || width <= 0 || heads <= 0 || head_dim <= 0) return -1;
    const int total = tokens * width;
    constexpr int block = 256;
    mrope_floatpos_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        x, position_ids, y, tokens, width, heads, head_dim, rope_theta);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_head_rmsnorm_mrope_pos_f32(
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
    cudaStream_t stream) {
    if (!x || !gamma_bf16 || !position_ids || !y ||
        tokens <= 0 || width <= 0 || heads <= 0 || head_dim <= 0 ||
        width != heads * head_dim || head_dim > 1024) {
        return -1;
    }
    constexpr int block = 128;
    if (head_dim != block) return -1;
    head_rmsnorm_mrope_floatpos_kernel<<<tokens * heads, block, block * sizeof(float), stream>>>(
        x, gamma_bf16, position_ids, y, tokens, width, heads, head_dim, eps, rope_theta);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_causal_gqa_attention_f32(
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
    cudaStream_t stream) {
    if (!q || !k || !v || !y || tokens <= 0 || hidden <= 0 || kv <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || head_dim <= 0) return -1;
    if (causal_gqa_attention_cublas_scores_f32(q, k, v, y,
                                               tokens, hidden, kv,
                                               q_heads, kv_heads, head_dim,
                                               stream) == 0) {
        return 0;
    }
    causal_gqa_attention_online_kernel<<<tokens * q_heads, head_dim, head_dim * sizeof(float), stream>>>(
        q, k, v, y, tokens, hidden, kv, q_heads, kv_heads, head_dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_gen_full_gqa_attention_f32(
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
    cudaStream_t stream) {
    if (!q_gen || !k_all || !v_all || !y_gen ||
        gen_tokens <= 0 || all_tokens <= 0 || hidden <= 0 || kv <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        hidden != q_heads * head_dim || kv != kv_heads * head_dim ||
        q_heads % kv_heads != 0) {
        return -1;
    }
    if (gen_full_gqa_attention_cublas_scores_f32(q_gen, k_all, v_all, y_gen,
                                                gen_tokens, all_tokens, hidden, kv,
                                                q_heads, kv_heads, head_dim,
                                                stream) == 0) {
        return 0;
    }
    gen_full_gqa_attention_online_kernel<<<gen_tokens * q_heads,
                                           head_dim,
                                           head_dim * sizeof(float),
                                           stream>>>(
        q_gen, k_all, v_all, y_gen,
        gen_tokens, all_tokens, hidden, kv, q_heads, kv_heads, head_dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_gen_full_gqa_attention_workspace_f32(
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
    cudaStream_t stream) {
    if (!q_gen || !k_all || !v_all || !y_gen || !scores_workspace ||
        gen_tokens <= 0 || all_tokens <= 0 || hidden <= 0 || kv <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        hidden != q_heads * head_dim || kv != kv_heads * head_dim ||
        q_heads % kv_heads != 0) {
        return -1;
    }
    if (gen_full_gqa_attention_cublas_scores_workspace_f32(q_gen, k_all, v_all, y_gen,
                                                           scores_workspace,
                                                           gen_tokens, all_tokens, hidden, kv,
                                                           q_heads, kv_heads, head_dim,
                                                           stream) == 0) {
        return 0;
    }
    gen_full_gqa_attention_online_kernel<<<gen_tokens * q_heads,
                                           head_dim,
                                           head_dim * sizeof(float),
                                           stream>>>(
        q_gen, k_all, v_all, y_gen,
        gen_tokens, all_tokens, hidden, kv, q_heads, kv_heads, head_dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_gen_full_gqa_attention_workspace_bf16tc_f32(
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
    cudaStream_t stream) {
    if (!q_gen || !k_all || !v_all || !y_gen || !scores_workspace || !bf16_workspace ||
        gen_tokens <= 0 || all_tokens <= 0 || hidden <= 0 || kv <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        hidden != q_heads * head_dim || kv != kv_heads * head_dim ||
        q_heads % kv_heads != 0) {
        return -1;
    }
    if (gen_full_gqa_attention_bf16tc_workspace_f32(q_gen, k_all, v_all, y_gen,
                                                    scores_workspace,
                                                    bf16_workspace,
                                                    gen_tokens, all_tokens, hidden, kv,
                                                    q_heads, kv_heads, head_dim,
                                                    stream) == 0) {
        return 0;
    }
    return cosmos3_qwen_gen_full_gqa_attention_workspace_f32(q_gen,
                                                            k_all,
                                                            v_all,
                                                            y_gen,
                                                            scores_workspace,
                                                            gen_tokens,
                                                            all_tokens,
                                                            hidden,
                                                            kv,
                                                            q_heads,
                                                            kv_heads,
                                                            head_dim,
                                                            stream);
}

extern "C" int cosmos3_qwen_gen_full_gqa_attention_f32_online_debug(
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
    cudaStream_t stream) {
    if (!q_gen || !k_all || !v_all || !y_gen ||
        gen_tokens <= 0 || all_tokens <= 0 || hidden <= 0 || kv <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        hidden != q_heads * head_dim || kv != kv_heads * head_dim ||
        q_heads % kv_heads != 0) {
        return -1;
    }
    gen_full_gqa_attention_online_kernel<<<gen_tokens * q_heads,
                                           head_dim,
                                           head_dim * sizeof(float),
                                           stream>>>(
        q_gen, k_all, v_all, y_gen,
        gen_tokens, all_tokens, hidden, kv, q_heads, kv_heads, head_dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_gen_full_gqa_attention_f32_cublas_debug(
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
    cudaStream_t stream) {
    if (!q_gen || !k_all || !v_all || !y_gen ||
        gen_tokens <= 0 || all_tokens <= 0 || hidden <= 0 || kv <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        hidden != q_heads * head_dim || kv != kv_heads * head_dim ||
        q_heads % kv_heads != 0) {
        return -1;
    }
    return gen_full_gqa_attention_cublas_scores_f32(q_gen, k_all, v_all, y_gen,
                                                   gen_tokens, all_tokens, hidden, kv,
                                                   q_heads, kv_heads, head_dim,
                                                   stream);
}

extern "C" int cosmos3_qwen_swiglu_f32(
    const float * gate,
    const float * up,
    float * y,
    int count,
    cudaStream_t stream) {
    if (!gate || !up || !y || count <= 0) return -1;
    constexpr int block = 256;
    swiglu_kernel<<<(count + block - 1) / block, block, 0, stream>>>(gate, up, y, count);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_residual_add_f32(
    const float * a,
    const float * b,
    float * y,
    int count,
    cudaStream_t stream) {
    if (!a || !b || !y || count <= 0) return -1;
    constexpr int block = 256;
    residual_add_kernel<<<(count + block - 1) / block, block, 0, stream>>>(a, b, y, count);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_residual_add_rmsnorm_f32(
    const float * residual_input,
    const float * branch,
    const unsigned short * gamma_bf16,
    float * residual,
    float * norm,
    int tokens,
    int hidden,
    float eps,
    cudaStream_t stream) {
    if (!residual_input || !branch || !gamma_bf16 || !residual || !norm ||
        tokens <= 0 || hidden <= 0) return -1;
    constexpr int block = 256;
    residual_add_rmsnorm_kernel<<<tokens, block, block * sizeof(float), stream>>>(
        residual_input, branch, gamma_bf16, residual, norm, tokens, hidden, eps);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_build_robolab_input_f32(
    const int * input_ids,
    const int * visual_indices,
    const unsigned short * embed_tokens_bf16,
    const float * visual_tokens,
    float * y,
    int tokens,
    int hidden,
    int vocab,
    int visual_token_count,
    cudaStream_t stream) {
    if (!input_ids || !visual_indices || !embed_tokens_bf16 || !visual_tokens || !y ||
        tokens <= 0 || hidden <= 0 || vocab <= 0 || visual_token_count <= 0) {
        return -1;
    }
    constexpr int block = 256;
    const int total = tokens * hidden;
    build_robolab_input_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        input_ids, visual_indices, embed_tokens_bf16, visual_tokens, y,
        tokens, hidden, vocab, visual_token_count);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_embed_tokens_f32(
    const int * input_ids,
    const unsigned short * embed_tokens_bf16,
    float * y,
    int tokens,
    int hidden,
    int vocab,
    cudaStream_t stream) {
    if (!input_ids || !embed_tokens_bf16 || !y ||
        tokens <= 0 || hidden <= 0 || vocab <= 0) {
        return -1;
    }
    constexpr int block = 256;
    const int total = tokens * hidden;
    embed_tokens_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        input_ids, embed_tokens_bf16, y, tokens, hidden, vocab);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_qwen_add_visual_deepstack_f32(
    float * hidden_states,
    const int * visual_indices,
    const float * deepstack_tokens,
    int tokens,
    int hidden,
    int visual_token_count,
    cudaStream_t stream) {
    if (!hidden_states || !visual_indices || !deepstack_tokens ||
        tokens <= 0 || hidden <= 0 || visual_token_count <= 0) {
        return -1;
    }
    constexpr int block = 256;
    const int total = tokens * hidden;
    add_visual_deepstack_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        hidden_states, visual_indices, deepstack_tokens, tokens, hidden, visual_token_count);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" const char * cosmos3_qwen_language_kernel_status(void) {
    return "qwen_language_cuda_ops:gpu_splice_mrope_rmsnorm_cublas_causal_online_gen_gqa_swiglu_deepstack_residual";
}
