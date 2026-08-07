// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "lingbot_flex_attn_cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <vector>

#define CUDA_CHECK_RET(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "vla(lingbot_flex): CUDA error %s at %s:%d (%s)\n", cudaGetErrorString(_e), __FILE__, __LINE__, #x); \
    return -1; }} while (0)

__device__ __forceinline__ uint16_t lingbot_f32_to_bf16_bits_rne_device(float x) {
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;
    const uint32_t lsb = (v.u >> 16) & 1u;
    return (uint16_t) ((v.u + 0x7fffu + lsb) >> 16);
}

__device__ __forceinline__ float lingbot_bf16_bits_to_f32_device(uint16_t x) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t) x << 16;
    return v.f;
}

__device__ __forceinline__ float lingbot_bf16_round_device(float x) {
    return lingbot_bf16_bits_to_f32_device(lingbot_f32_to_bf16_bits_rne_device(x));
}

__global__ void scheduler_step_f32_kernel(
        float * __restrict__ sample,
        const float * __restrict__ model_output,
        int n,
        float delta,
        int bf16_round) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float prod = __fmul_rn(model_output[i], delta);
    float v = __fadd_rn(sample[i], prod);
    if (bf16_round) v = lingbot_bf16_round_device(v);
    sample[i] = v;
}

__global__ void model_output_guidance_f32_kernel(
        float * __restrict__ cond_in_out,
        const float * __restrict__ uncond,
        int n,
        float scale,
        int use_guidance,
        int bf16_round) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c = cond_in_out[i];
    if (bf16_round) c = lingbot_bf16_round_device(c);
    float v = c;
    if (use_guidance) {
        float u = uncond ? uncond[i] : 0.0f;
        if (bf16_round) u = lingbot_bf16_round_device(u);
        v = u + scale * (c - u);
        if (bf16_round) v = lingbot_bf16_round_device(v);
    }
    cond_in_out[i] = v;
}

__global__ void action_finalize_f32_kernel(
        float * __restrict__ action_sample,
        const float * __restrict__ action_cond,
        int C,
        int F,
        int H,
        int W,
        int used_dim,
        int bf16_round) {
    const int n = C * F * H * W;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int w = i % W;
    const int h = (i / W) % H;
    const int f = (i / (W * H)) % F;
    const int c = (i / (W * H * F)) % C;
    (void) h;
    (void) w;
    float v = action_sample[i];
    if (action_cond && f == 0) {
        v = action_cond[i];
    }
    if (C > used_dim && W == 1 && c >= used_dim) {
        v = 0.0f;
    }
    if (bf16_round) v = lingbot_bf16_round_device(v);
    action_sample[i] = v;
}

__global__ void latent_restore_round_f32_kernel(
        float * __restrict__ latent_sample,
        const float * __restrict__ latent_cond,
        int C,
        int F,
        int H,
        int W,
        int cond_F,
        int bf16_round) {
    const int n = C * F * H * W;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int w = i % W;
    const int h = (i / W) % H;
    const int f = (i / (W * H)) % F;
    const int c = (i / (W * H * F)) % C;
    float v = latent_sample[i];
    if (latent_cond && f < cond_F) {
        const int cond_idx = ((c * cond_F + f) * H + h) * W + w;
        v = latent_cond[cond_idx];
    }
    if (bf16_round) v = lingbot_bf16_round_device(v);
    latent_sample[i] = v;
}

__device__ __forceinline__ int idx5_bcfhw_device(int C, int F, int H, int W, int b, int c, int f, int h, int w) {
    return ((((b * C + c) * F + f) * H + h) * W + w);
}

__global__ void patchify_latent_f32_kernel(
        const float * __restrict__ latent,
        float * __restrict__ tokens,
        int B,
        int C,
        int F,
        int H,
        int W,
        int pt,
        int ph,
        int pw) {
    const int pf = F / pt;
    const int phn = H / ph;
    const int pwn = W / pw;
    const int feature_dim = C * pt * ph * pw;
    const int seq = B * pf * phn * pwn;
    const int total = feature_dim * seq;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const int feat = i % feature_dim;
    const int tok = i / feature_dim;
    int tmp_feat = feat;
    const int pk = tmp_feat % pw; tmp_feat /= pw;
    const int pj = tmp_feat % ph; tmp_feat /= ph;
    const int pi = tmp_feat % pt; tmp_feat /= pt;
    const int c = tmp_feat;

    int tmp_tok = tok;
    const int wo = tmp_tok % pwn; tmp_tok /= pwn;
    const int ho = tmp_tok % phn; tmp_tok /= phn;
    const int fo = tmp_tok % pf; tmp_tok /= pf;
    const int b = tmp_tok;
    tokens[i] = latent[idx5_bcfhw_device(C, F, H, W, b, c, fo * pt + pi, ho * ph + pj, wo * pw + pk)];
}

__global__ void projected_latent_to_tensor_f32_kernel(
        const float * __restrict__ projected,
        float * __restrict__ latent,
        int B,
        int C,
        int F,
        int H,
        int W,
        int pt,
        int ph,
        int pw) {
    const int pf = F / pt;
    const int phn = H / ph;
    const int pwn = W / pw;
    const int feature_dim = C * pt * ph * pw;
    const int total = B * C * F * H * W;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    int tmp = i;
    const int w = tmp % W; tmp /= W;
    const int h = tmp % H; tmp /= H;
    const int f = tmp % F; tmp /= F;
    const int c = tmp % C; tmp /= C;
    const int b = tmp;

    const int fo = f / pt;
    const int pi = f % pt;
    const int ho = h / ph;
    const int pj = h % ph;
    const int wo = w / pw;
    const int pk = w % pw;
    const int tok = (((b * pf + fo) * phn + ho) * pwn + wo);
    const int patch_index = ((pi * ph + pj) * pw + pk);
    const int feat = patch_index * C + c;
    latent[i] = projected[feat + feature_dim * tok];
}

__global__ void action_tensor_to_tokens_f32_kernel(
        const float * __restrict__ action,
        float * __restrict__ tokens,
        int B,
        int C,
        int F,
        int H,
        int W) {
    const int seq = B * F * H * W;
    const int total = C * seq;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = i % C;
    const int tok = i / C;
    int tmp_tok = tok;
    const int w = tmp_tok % W; tmp_tok /= W;
    const int h = tmp_tok % H; tmp_tok /= H;
    const int f = tmp_tok % F; tmp_tok /= F;
    const int b = tmp_tok;
    tokens[i] = action[idx5_bcfhw_device(C, F, H, W, b, c, f, h, w)];
}

__global__ void action_tokens_to_tensor_f32_kernel(
        const float * __restrict__ tokens,
        float * __restrict__ action,
        int B,
        int C,
        int F,
        int H,
        int W) {
    const int total = B * C * F * H * W;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int tmp = i;
    const int w = tmp % W; tmp /= W;
    const int h = tmp % H; tmp /= H;
    const int f = tmp % F; tmp /= F;
    const int c = tmp % C; tmp /= C;
    const int b = tmp;
    const int tok = (((b * F + f) * H + h) * W + w);
    action[i] = tokens[c + C * tok];
}

__global__ void action_sample_to_output_f32_kernel(
        const float * __restrict__ action,
        float * __restrict__ out,
        int B,
        int C,
        int F,
        int H,
        int W,
        int n_suffix,
        int output_dim,
        int postprocess_libero) {
    static constexpr int k_used_dim = 7;
    static constexpr float k_q01[k_used_dim] = {
        -0.6589285731315613f,
        -0.84375f,
        -0.9375f,
        -0.12107142806053162f,
        -0.15964286029338837f,
        -0.26571428775787354f,
        -1.0f,
    };
    static constexpr float k_q99[k_used_dim] = {
        0.8999999761581421f,
        0.8544642925262451f,
        0.9375f,
        0.17142857611179352f,
        0.1842857152223587f,
        0.34392857551574707f,
        1.0f,
    };
    const int total = n_suffix * output_dim;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int t = i / output_dim;
    const int c_out = i % output_dim;
    const int seq = B * F * H * W;
    float v = 0.0f;
    if (t < seq) {
        int tmp = t;
        const int w = tmp % W; tmp /= W;
        const int h = tmp % H; tmp /= H;
        const int f = tmp % F; tmp /= F;
        const int b = tmp;
        if (postprocess_libero) {
            if (c_out < k_used_dim && c_out < C) {
                const float x = action[idx5_bcfhw_device(C, F, H, W, b, c_out, f, h, w)];
                v = (x + 1.0f) * 0.5f * (k_q99[c_out] - k_q01[c_out] + 1.0e-6f) + k_q01[c_out];
            }
        } else if (c_out < C) {
            v = action[idx5_bcfhw_device(C, F, H, W, b, c_out, f, h, w)];
        }
    }
    out[i] = v;
}

extern "C" int lingbot_scheduler_step_f32(
    float * sample,
    const float * model_output,
    int n,
    float delta,
    int bf16_round,
    cudaStream_t stream) {
    if (!sample || !model_output || n < 0) return -1;
    if (n == 0) return 0;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    scheduler_step_f32_kernel<<<grid, block, 0, stream>>>(sample, model_output, n, delta, bf16_round);
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

extern "C" int lingbot_model_output_guidance_f32(
    float * cond_in_out,
    const float * uncond,
    int n,
    float scale,
    int use_guidance,
    int bf16_round,
    cudaStream_t stream) {
    if (!cond_in_out || n < 0 || (use_guidance && !uncond)) return -1;
    if (n == 0) return 0;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    model_output_guidance_f32_kernel<<<grid, block, 0, stream>>>(
        cond_in_out, uncond, n, scale, use_guidance, bf16_round);
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

extern "C" int lingbot_action_finalize_f32(
    float * action_sample,
    const float * action_cond,
    int C,
    int F,
    int H,
    int W,
    int used_dim,
    int bf16_round,
    cudaStream_t stream) {
    if (!action_sample || C < 0 || F < 0 || H < 0 || W < 0 || used_dim < 0) return -1;
    const int n = C * F * H * W;
    if (n == 0) return 0;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    action_finalize_f32_kernel<<<grid, block, 0, stream>>>(
        action_sample, action_cond, C, F, H, W, used_dim, bf16_round);
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

extern "C" int lingbot_latent_restore_round_f32(
    float * latent_sample,
    const float * latent_cond,
    int C,
    int F,
    int H,
    int W,
    int cond_F,
    int bf16_round,
    cudaStream_t stream) {
    if (!latent_sample || C < 0 || F < 0 || H < 0 || W < 0 || cond_F < 0 || cond_F > F) return -1;
    const int n = C * F * H * W;
    if (n == 0) return 0;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    latent_restore_round_f32_kernel<<<grid, block, 0, stream>>>(
        latent_sample, latent_cond, C, F, H, W, cond_F, bf16_round);
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

extern "C" int lingbot_patchify_latent_f32(
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
    cudaStream_t stream) {
    if (!latent_bcfhw || !tokens || B < 0 || C < 0 || F < 0 || H < 0 || W < 0 ||
        pt <= 0 || ph <= 0 || pw <= 0 || F % pt != 0 || H % ph != 0 || W % pw != 0) {
        return -1;
    }
    const int feature_dim = C * pt * ph * pw;
    const int seq = B * (F / pt) * (H / ph) * (W / pw);
    const int n = feature_dim * seq;
    if (n == 0) return 0;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    patchify_latent_f32_kernel<<<grid, block, 0, stream>>>(
        latent_bcfhw, tokens, B, C, F, H, W, pt, ph, pw);
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

extern "C" int lingbot_projected_latent_to_tensor_f32(
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
    cudaStream_t stream) {
    if (!projected_tokens || !latent_bcfhw || B < 0 || C < 0 || F < 0 || H < 0 || W < 0 ||
        pt <= 0 || ph <= 0 || pw <= 0 || F % pt != 0 || H % ph != 0 || W % pw != 0) {
        return -1;
    }
    const int n = B * C * F * H * W;
    if (n == 0) return 0;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    projected_latent_to_tensor_f32_kernel<<<grid, block, 0, stream>>>(
        projected_tokens, latent_bcfhw, B, C, F, H, W, pt, ph, pw);
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

extern "C" int lingbot_action_tensor_to_tokens_f32(
    const float * action_bcfhw,
    float * tokens,
    int B,
    int C,
    int F,
    int H,
    int W,
    cudaStream_t stream) {
    if (!action_bcfhw || !tokens || B < 0 || C < 0 || F < 0 || H < 0 || W < 0) return -1;
    const int n = B * C * F * H * W;
    if (n == 0) return 0;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    action_tensor_to_tokens_f32_kernel<<<grid, block, 0, stream>>>(
        action_bcfhw, tokens, B, C, F, H, W);
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

extern "C" int lingbot_action_tokens_to_tensor_f32(
    const float * tokens,
    float * action_bcfhw,
    int B,
    int C,
    int F,
    int H,
    int W,
    cudaStream_t stream) {
    if (!tokens || !action_bcfhw || B < 0 || C < 0 || F < 0 || H < 0 || W < 0) return -1;
    const int n = B * C * F * H * W;
    if (n == 0) return 0;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    action_tokens_to_tensor_f32_kernel<<<grid, block, 0, stream>>>(
        tokens, action_bcfhw, B, C, F, H, W);
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

extern "C" int lingbot_action_sample_to_output_f32(
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
    cudaStream_t stream) {
    if (!action_bcfhw || !out || B < 0 || C < 0 || F < 0 || H < 0 || W < 0 ||
        n_suffix < 0 || output_dim < 0) {
        return -1;
    }
    const int n = n_suffix * output_dim;
    if (n == 0) return 0;
    if (postprocess_libero && (output_dim != 7 || C < 7)) return -1;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    action_sample_to_output_f32_kernel<<<grid, block, 0, stream>>>(
        action_bcfhw, out, B, C, F, H, W, n_suffix, output_dim, postprocess_libero);
    CUDA_CHECK_RET(cudaGetLastError());
    return 0;
}

__global__ void flex_attn_f32_kernel(
        const float * __restrict__ q,
        const float * __restrict__ k,
        const float * __restrict__ v,
        const int * __restrict__ row_ptr,
        const int * __restrict__ col_idx,
        const uint8_t * __restrict__ token_mask,
        float * __restrict__ out,
        int seq_q,
        int seq_k,
        int n_heads,
        int head_dim,
        int block_size,
        float scale) {
    const int qid = blockIdx.x;
    const int h = blockIdx.y;
    if (qid >= seq_q || h >= n_heads) return;

    extern __shared__ float smem[];
    float * acc = smem;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) acc[d] = 0.0f;
    __shared__ float row_sum_s;
    if (threadIdx.x == 0) row_sum_s = 0.0f;
    __shared__ float row_max_s;
    if (threadIdx.x == 0) row_max_s = -INFINITY;
    __syncthreads();

    const int qb = qid / block_size;
    for (int p = row_ptr[qb]; p < row_ptr[qb + 1]; ++p) {
        const int kb = col_idx[p];
        const int k0 = kb * block_size;
        const int k1 = min(k0 + block_size, seq_k);
        for (int kid = k0; kid < k1; ++kid) {
            if (token_mask && !token_mask[qid * seq_k + kid]) continue;
            float dot = 0.0f;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                dot += q[(qid * n_heads + h) * head_dim + d] *
                       k[(kid * n_heads + h) * head_dim + d];
            }
            __shared__ float red[256];
            red[threadIdx.x] = dot;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
                __syncthreads();
            }
            if (threadIdx.x == 0) row_max_s = fmaxf(row_max_s, red[0] * scale);
            __syncthreads();
        }
    }

    for (int p = row_ptr[qb]; p < row_ptr[qb + 1]; ++p) {
        const int kb = col_idx[p];
        const int k0 = kb * block_size;
        const int k1 = min(k0 + block_size, seq_k);
        for (int kid = k0; kid < k1; ++kid) {
            if (token_mask && !token_mask[qid * seq_k + kid]) continue;
            float dot = 0.0f;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                dot += q[(qid * n_heads + h) * head_dim + d] *
                       k[(kid * n_heads + h) * head_dim + d];
            }
            __shared__ float red[256];
            red[threadIdx.x] = dot;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
                __syncthreads();
            }
            const float w = expf(red[0] * scale - row_max_s);
            if (threadIdx.x == 0) row_sum_s += w;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                acc[d] += w * v[(kid * n_heads + h) * head_dim + d];
            }
            __syncthreads();
        }
    }
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[(qid * n_heads + h) * head_dim + d] = row_sum_s > 0.0f ? acc[d] / row_sum_s : 0.0f;
    }
}

extern "C" int lingbot_flex_attn_f32(
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
    cudaStream_t stream) {
    // This public entry assumes the block table is exact at block granularity.
    flex_attn_f32_kernel<<<dim3(seq_q, n_heads, 1), dim3(128, 1, 1),
                           (size_t) head_dim * sizeof(float), stream>>>(
        q, k, v, row_ptr, col_idx, nullptr, out, seq_q, seq_k, n_heads, head_dim, block_size, scale);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_flex_attn_f32_masked(
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
    cudaStream_t stream) {
    flex_attn_f32_kernel<<<dim3(seq_q, n_heads, 1), dim3(128, 1, 1),
                           (size_t) head_dim * sizeof(float), stream>>>(
        q, k, v, row_ptr, col_idx, token_mask, out, seq_q, seq_k, n_heads, head_dim, block_size, scale);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ void runtime_kv_update_f32_kernel(
        const float * __restrict__ k,
        const float * __restrict__ v,
        float * __restrict__ cache_k,
        float * __restrict__ cache_v,
        const int * __restrict__ slots,
        int seq,
        int n_heads,
        int head_dim,
        int capacity) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq * n_heads * head_dim;
    if (idx >= total) return;
    const int hd = n_heads * head_dim;
    const int s = idx / hd;
    const int rem = idx - s * hd;
    const int h = rem / head_dim;
    const int d = rem - h * head_dim;
    const int slot = slots[s];
    if (slot < 0 || slot >= capacity) return;
    const int dst = (slot * n_heads + h) * head_dim + d;
    cache_k[dst] = k[idx];
    cache_v[dst] = v[idx];
}

__global__ void runtime_kv_attn_f32_kernel(
        const float * __restrict__ q,
        const float * __restrict__ cache_k,
        const float * __restrict__ cache_v,
        const int * __restrict__ valid_slots,
        float * __restrict__ out,
        int seq_q,
        int seq_k,
        int n_heads,
        int head_dim,
        float scale) {
    const int qid = blockIdx.x;
    const int h = blockIdx.y;
    if (qid >= seq_q || h >= n_heads) return;

    extern __shared__ float smem[];
    float * acc = smem;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) acc[d] = 0.0f;
    __shared__ float row_sum_s;
    __shared__ float row_max_s;
    if (threadIdx.x == 0) {
        row_sum_s = 0.0f;
        row_max_s = -INFINITY;
    }
    __syncthreads();

    for (int kid = 0; kid < seq_k; ++kid) {
        const int slot = valid_slots[kid];
        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += q[(qid * n_heads + h) * head_dim + d] *
                   cache_k[(slot * n_heads + h) * head_dim + d];
        }
        __shared__ float red[256];
        red[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) row_max_s = fmaxf(row_max_s, red[0] * scale);
        __syncthreads();
    }

    for (int kid = 0; kid < seq_k; ++kid) {
        const int slot = valid_slots[kid];
        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += q[(qid * n_heads + h) * head_dim + d] *
                   cache_k[(slot * n_heads + h) * head_dim + d];
        }
        __shared__ float red[256];
        red[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }
        const float w = expf(red[0] * scale - row_max_s);
        if (threadIdx.x == 0) row_sum_s += w;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            acc[d] += w * cache_v[(slot * n_heads + h) * head_dim + d];
        }
        __syncthreads();
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[(qid * n_heads + h) * head_dim + d] = row_sum_s > 0.0f ? acc[d] / row_sum_s : 0.0f;
    }
}

__global__ void runtime_kv_attn_online_f32_kernel(
        const float * __restrict__ q,
        const float * __restrict__ cache_k,
        const float * __restrict__ cache_v,
        const int * __restrict__ valid_slots,
        float * __restrict__ out,
        int seq_q,
        int seq_k,
        int n_heads,
        int head_dim,
        float scale) {
    const int qid = blockIdx.x;
    const int h = blockIdx.y;
    if (qid >= seq_q || h >= n_heads) return;

    extern __shared__ float smem[];
    float * acc = smem;
    float * red = smem + head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        acc[d] = 0.0f;
    }
    __shared__ float row_sum_s;
    __shared__ float row_max_s;
    __shared__ float alpha_s;
    __shared__ float beta_s;
    if (threadIdx.x == 0) {
        row_sum_s = 0.0f;
        row_max_s = -INFINITY;
        alpha_s = 0.0f;
        beta_s = 0.0f;
    }
    __syncthreads();

    const int q_base = (qid * n_heads + h) * head_dim;
    for (int kid = 0; kid < seq_k; ++kid) {
        const int slot = valid_slots[kid];
        float dot = 0.0f;
        if (slot >= 0) {
            const int k_base = (slot * n_heads + h) * head_dim;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                dot += q[q_base + d] * cache_k[k_base + d];
            }
        }
        red[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float score = red[0] * scale;
            const float new_max = fmaxf(row_max_s, score);
            alpha_s = (row_max_s == -INFINITY) ? 0.0f : expf(row_max_s - new_max);
            beta_s = expf(score - new_max);
            row_sum_s = row_sum_s * alpha_s + beta_s;
            row_max_s = new_max;
        }
        __syncthreads();
        if (slot >= 0) {
            const int v_base = (slot * n_heads + h) * head_dim;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                acc[d] = acc[d] * alpha_s + beta_s * cache_v[v_base + d];
            }
        }
        __syncthreads();
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[q_base + d] = row_sum_s > 0.0f ? acc[d] / row_sum_s : 0.0f;
    }
}

__device__ __forceinline__ float lingbot_warp_sum_f32(float v) {
    unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

__global__ void runtime_kv_attn_warp_f32_kernel(
        const float * __restrict__ q,
        const float * __restrict__ cache_k,
        const float * __restrict__ cache_v,
        const int * __restrict__ valid_slots,
        float * __restrict__ out,
        int seq_q,
        int seq_k,
        int n_heads,
        int head_dim,
        float scale) {
    const int qid = blockIdx.x;
    const int h = blockIdx.y;
    const int lane = threadIdx.x;
    if (qid >= seq_q || h >= n_heads || lane >= 32) return;

    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    float row_sum = 0.0f;
    float row_max = -INFINITY;
    const int q_base = (qid * n_heads + h) * head_dim;
    const int lane_count = (head_dim + 31) >> 5;

    for (int kid = 0; kid < seq_k; ++kid) {
        const int slot = valid_slots[kid];
        float dot_lane = 0.0f;
        if (slot >= 0) {
            const int k_base = (slot * n_heads + h) * head_dim;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < lane_count) {
                    const int d = lane + (i << 5);
                    if (d < head_dim) {
                        dot_lane += q[q_base + d] * cache_k[k_base + d];
                    }
                }
            }
        }
        const float dot = __shfl_sync(0xffffffffu, lingbot_warp_sum_f32(dot_lane), 0);
        const float score = dot * scale;
        const float new_max = fmaxf(row_max, score);
        const float alpha = (row_max == -INFINITY) ? 0.0f : expf(row_max - new_max);
        const float beta = expf(score - new_max);
        row_sum = row_sum * alpha + beta;
        row_max = new_max;
        if (slot >= 0) {
            const int v_base = (slot * n_heads + h) * head_dim;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (i < lane_count) {
                    const int d = lane + (i << 5);
                    if (d < head_dim) {
                        acc[i] = acc[i] * alpha + beta * cache_v[v_base + d];
                    }
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (i < lane_count) {
            const int d = lane + (i << 5);
            if (d < head_dim) {
                out[q_base + d] = row_sum > 0.0f ? acc[i] / row_sum : 0.0f;
            }
        }
    }
}

__global__ void runtime_kv_gather_f32_kernel(
        const float * __restrict__ cache_k,
        const float * __restrict__ cache_v,
        const int * __restrict__ valid_slots,
        float * __restrict__ compact_k,
        float * __restrict__ compact_v,
        int seq_k,
        int n_heads,
        int head_dim,
        int capacity) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = seq_k * n_heads * head_dim;
    if (idx >= total) return;
    const int hd = n_heads * head_dim;
    const int kid = idx / hd;
    const int rem = idx - kid * hd;
    const int h = rem / head_dim;
    const int d = rem - h * head_dim;
    const int slot = valid_slots[kid];
    if (slot < 0 || slot >= capacity) return;
    const int src = (slot * n_heads + h) * head_dim + d;
    compact_k[idx] = cache_k[src];
    compact_v[idx] = cache_v[src];
}

extern "C" int lingbot_runtime_kv_update_f32(
    const float * k,
    const float * v,
    float * cache_k,
    float * cache_v,
    const int * slots,
    int seq,
    int n_heads,
    int head_dim,
    int capacity,
    cudaStream_t stream) {
    if (!k || !v || !cache_k || !cache_v || !slots || seq <= 0 || n_heads <= 0 || head_dim <= 0 || capacity <= 0) {
        return -1;
    }
    const int total = seq * n_heads * head_dim;
    const int block = 256;
    runtime_kv_update_f32_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        k, v, cache_k, cache_v, slots, seq, n_heads, head_dim, capacity);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_runtime_kv_attn_f32(
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
    cudaStream_t stream) {
    if (!q || !cache_k || !cache_v || !valid_slots || !out ||
        seq_q <= 0 || seq_k <= 0 || n_heads <= 0 || head_dim <= 0) {
        return -1;
    }
    runtime_kv_attn_f32_kernel<<<dim3(seq_q, n_heads, 1), dim3(128, 1, 1),
                                 (size_t) head_dim * sizeof(float), stream>>>(
        q, cache_k, cache_v, valid_slots, out, seq_q, seq_k, n_heads, head_dim, scale);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_runtime_kv_attn_online_f32(
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
    cudaStream_t stream) {
    if (!q || !cache_k || !cache_v || !valid_slots || !out ||
        seq_q <= 0 || seq_k <= 0 || n_heads <= 0 || head_dim <= 0 || head_dim > 256) {
        return -1;
    }
    const int block = 256;
    runtime_kv_attn_online_f32_kernel<<<dim3(seq_q, n_heads, 1), dim3(block, 1, 1),
                                        (size_t) (head_dim + block) * sizeof(float), stream>>>(
        q, cache_k, cache_v, valid_slots, out, seq_q, seq_k, n_heads, head_dim, scale);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_runtime_kv_attn_warp_f32(
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
    cudaStream_t stream) {
    if (!q || !cache_k || !cache_v || !valid_slots || !out ||
        seq_q <= 0 || seq_k <= 0 || n_heads <= 0 || head_dim <= 0 || head_dim > 256) {
        return -1;
    }
    runtime_kv_attn_warp_f32_kernel<<<dim3(seq_q, n_heads, 1), dim3(32, 1, 1), 0, stream>>>(
        q, cache_k, cache_v, valid_slots, out, seq_q, seq_k, n_heads, head_dim, scale);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_runtime_kv_gather_f32(
    const float * cache_k,
    const float * cache_v,
    const int * valid_slots,
    float * compact_k,
    float * compact_v,
    int seq_k,
    int n_heads,
    int head_dim,
    int capacity,
    cudaStream_t stream) {
    if (!cache_k || !cache_v || !valid_slots || !compact_k || !compact_v ||
        seq_k <= 0 || n_heads <= 0 || head_dim <= 0 || capacity <= 0) {
        return -1;
    }
    const int total = seq_k * n_heads * head_dim;
    const int block = 256;
    runtime_kv_gather_f32_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        cache_k, cache_v, valid_slots, compact_k, compact_v,
        seq_k, n_heads, head_dim, capacity);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

static void dense_ref(const std::vector<float> & q, const std::vector<float> & k,
                      const std::vector<float> & v, const std::vector<uint8_t> & mask,
                      std::vector<float> & out, int S, int H, int D, float scale) {
    out.assign((size_t) S * H * D, 0.0f);
    for (int s = 0; s < S; ++s) for (int h = 0; h < H; ++h) {
        float mx = -INFINITY;
        for (int t = 0; t < S; ++t) if (mask[(size_t) s * S + t]) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += q[(s * H + h) * D + d] * k[(t * H + h) * D + d];
            mx = fmaxf(mx, dot * scale);
        }
        float sum = 0.0f;
        for (int t = 0; t < S; ++t) if (mask[(size_t) s * S + t]) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += q[(s * H + h) * D + d] * k[(t * H + h) * D + d];
            const float w = expf(dot * scale - mx);
            sum += w;
            for (int d = 0; d < D; ++d) out[(s * H + h) * D + d] += w * v[(t * H + h) * D + d];
        }
        for (int d = 0; d < D; ++d) out[(s * H + h) * D + d] = sum > 0.0f ? out[(s * H + h) * D + d] / sum : 0.0f;
    }
}

extern "C" int lingbot_flex_attn_cuda_smoke(cudaStream_t stream) {
    const int S = 8, H = 2, D = 8, BS = 4;
    const float scale = 1.0f / std::sqrt((float) D);
    std::vector<float> q((size_t) S * H * D), k(q.size()), v(q.size()), out(q.size()), ref;
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = std::sin((float) i * 0.017f) * 0.2f;
        k[i] = std::cos((float) i * 0.019f) * 0.15f;
        v[i] = std::sin((float) i * 0.023f) * 0.3f;
    }
    std::vector<uint8_t> mask((size_t) S * S, 0);
    for (int s = 0; s < S; ++s) for (int t = 0; t < S; ++t) {
        mask[(size_t) s * S + t] = (t <= s && (s - t) <= 4) ? 1 : 0;
    }
    std::vector<int> row_ptr{0, 1, 3};
    std::vector<int> col_idx{0, 0, 1};
    dense_ref(q, k, v, mask, ref, S, H, D, scale);

    float *dq=nullptr, *dk=nullptr, *dv=nullptr, *do_=nullptr;
    int *drp=nullptr, *dci=nullptr;
    uint8_t *dm=nullptr;
    CUDA_CHECK_RET(cudaMalloc(&dq, q.size()*sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dk, k.size()*sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dv, v.size()*sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&do_, out.size()*sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&drp, row_ptr.size()*sizeof(int)));
    CUDA_CHECK_RET(cudaMalloc(&dci, col_idx.size()*sizeof(int)));
    CUDA_CHECK_RET(cudaMalloc(&dm, mask.size()*sizeof(uint8_t)));
    CUDA_CHECK_RET(cudaMemcpyAsync(dq, q.data(), q.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(dk, k.data(), k.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(dv, v.data(), v.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(drp, row_ptr.data(), row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(dci, col_idx.data(), col_idx.size()*sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(dm, mask.data(), mask.size()*sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    flex_attn_f32_kernel<<<dim3(S, H, 1), dim3(128, 1, 1), (size_t) D*sizeof(float), stream>>>(
        dq, dk, dv, drp, dci, dm, do_, S, S, H, D, BS, scale);
    CUDA_CHECK_RET(cudaGetLastError());
    CUDA_CHECK_RET(cudaMemcpyAsync(out.data(), do_, out.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_RET(cudaStreamSynchronize(stream));
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(do_); cudaFree(drp); cudaFree(dci); cudaFree(dm);
    double max_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) max_diff = fmax(max_diff, fabs((double) out[i] - (double) ref[i]));
    std::printf("vla(lingbot_flex): cuda smoke max_diff=%.9g\n", max_diff);
    return max_diff < 1e-5 ? 0 : -1;
}

__global__ void causal_conv1d_cache_f32_kernel(
        const float * __restrict__ x,
        const float * __restrict__ past,
        const float * __restrict__ weight,
        const float * __restrict__ bias,
        float * __restrict__ out,
        int T,
        int C_in,
        int C_out,
        int K) {
    const int t = blockIdx.x;
    const int co = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || co >= C_out) return;

    float acc = bias ? bias[co] : 0.0f;
    for (int k = 0; k < K; ++k) {
        const int src_t = t + k - (K - 1);
        for (int ci = 0; ci < C_in; ++ci) {
            const float xv = src_t < 0
                ? past[(src_t + K - 1) * C_in + ci]
                : x[src_t * C_in + ci];
            acc += xv * weight[(co * C_in + ci) * K + k];
        }
    }
    out[t * C_out + co] = acc;
}

__global__ void causal_conv1d_next_past_f32_kernel(
        const float * __restrict__ x,
        const float * __restrict__ past,
        float * __restrict__ next_past,
        int T,
        int C_in,
        int K) {
    const int j = blockIdx.x;
    const int ci = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= K - 1 || ci >= C_in) return;
    const int src_t = T + j - (K - 1);
    next_past[j * C_in + ci] = src_t < 0
        ? past[(src_t + K - 1) * C_in + ci]
        : x[src_t * C_in + ci];
}

__global__ void causal_conv1d_cache_f32_batched_kernel(
        const float * __restrict__ x,
        const float * __restrict__ past,
        const float * __restrict__ weight,
        const float * __restrict__ bias,
        float * __restrict__ out,
        int lanes,
        int T,
        int C_in,
        int C_out,
        int K) {
    const int t = blockIdx.x;
    const int co = blockIdx.y * blockDim.x + threadIdx.x;
    const int lane = blockIdx.z;
    if (lane >= lanes || t >= T || co >= C_out) return;

    float acc = bias ? bias[co] : 0.0f;
    for (int k = 0; k < K; ++k) {
        const int src_t = t + k - (K - 1);
        for (int ci = 0; ci < C_in; ++ci) {
            const float xv = src_t < 0
                ? past[((lane * (K - 1) + (src_t + K - 1)) * C_in) + ci]
                : x[((lane * T + src_t) * C_in) + ci];
            acc += xv * weight[(co * C_in + ci) * K + k];
        }
    }
    out[((lane * T + t) * C_out) + co] = acc;
}

__global__ void causal_conv1d_cache_f32_batched_stride_kernel(
        const float * __restrict__ x,
        const float * __restrict__ past,
        const float * __restrict__ weight,
        const float * __restrict__ bias,
        float * __restrict__ out,
        int lanes,
        int T,
        int T_out,
        int C_in,
        int C_out,
        int K,
        int stride) {
    const int t = blockIdx.x;
    const int co = blockIdx.y * blockDim.x + threadIdx.x;
    const int lane = blockIdx.z;
    if (lane >= lanes || t >= T_out || co >= C_out) return;

    float acc = bias ? bias[co] : 0.0f;
    for (int k = 0; k < K; ++k) {
        const int src_t = t * stride + k - (K - 1);
        for (int ci = 0; ci < C_in; ++ci) {
            const float xv = src_t < 0
                ? past[((lane * (K - 1) + (src_t + K - 1)) * C_in) + ci]
                : (src_t < T ? x[((lane * T + src_t) * C_in) + ci] : 0.0f);
            acc += xv * weight[(co * C_in + ci) * K + k];
        }
    }
    out[((lane * T_out + t) * C_out) + co] = acc;
}

__global__ void causal_conv1d_next_past_f32_batched_kernel(
        const float * __restrict__ x,
        const float * __restrict__ past,
        float * __restrict__ next_past,
        int lanes,
        int T,
        int C_in,
        int K) {
    const int j = blockIdx.x;
    const int ci = blockIdx.y * blockDim.x + threadIdx.x;
    const int lane = blockIdx.z;
    if (lane >= lanes || j >= K - 1 || ci >= C_in) return;
    const int src_t = T + j - (K - 1);
    next_past[((lane * (K - 1) + j) * C_in) + ci] = src_t < 0
        ? past[((lane * (K - 1) + (src_t + K - 1)) * C_in) + ci]
        : x[((lane * T + src_t) * C_in) + ci];
}

extern "C" int lingbot_causal_conv1d_cache_f32(
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
    cudaStream_t stream) {
    if (T <= 0 || C_in <= 0 || C_out <= 0 || K <= 0) return -1;
    const int block = 128;
    const dim3 conv_grid(T, (C_out + block - 1) / block, 1);
    causal_conv1d_cache_f32_kernel<<<conv_grid, block, 0, stream>>>(
        x, past, weight, bias, out, T, C_in, C_out, K);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (next_past && K > 1) {
        const dim3 cache_grid(K - 1, (C_in + block - 1) / block, 1);
        causal_conv1d_next_past_f32_kernel<<<cache_grid, block, 0, stream>>>(
            x, past, next_past, T, C_in, K);
        if (cudaGetLastError() != cudaSuccess) return -1;
    }
    return 0;
}

extern "C" int lingbot_causal_conv1d_cache_f32_batched(
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
    cudaStream_t stream) {
    if (lanes <= 0 || T <= 0 || C_in <= 0 || C_out <= 0 || K <= 0) return -1;
    const int block = 128;
    const dim3 conv_grid(T, (C_out + block - 1) / block, lanes);
    causal_conv1d_cache_f32_batched_kernel<<<conv_grid, block, 0, stream>>>(
        x, past, weight, bias, out, lanes, T, C_in, C_out, K);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (next_past && K > 1) {
        const dim3 cache_grid(K - 1, (C_in + block - 1) / block, lanes);
        causal_conv1d_next_past_f32_batched_kernel<<<cache_grid, block, 0, stream>>>(
            x, past, next_past, lanes, T, C_in, K);
        if (cudaGetLastError() != cudaSuccess) return -1;
    }
    return 0;
}

extern "C" int lingbot_causal_conv1d_cache_f32_batched_stride(
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
    cudaStream_t stream) {
    if (lanes <= 0 || T <= 0 || T_out <= 0 || C_in <= 0 || C_out <= 0 || K <= 0 || stride <= 0) return -1;
    const int block = 128;
    const dim3 conv_grid(T_out, (C_out + block - 1) / block, lanes);
    causal_conv1d_cache_f32_batched_stride_kernel<<<conv_grid, block, 0, stream>>>(
        x, past, weight, bias, out, lanes, T, T_out, C_in, C_out, K, stride);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (next_past && K > 1) {
        const dim3 cache_grid(K - 1, (C_in + block - 1) / block, lanes);
        causal_conv1d_next_past_f32_batched_kernel<<<cache_grid, block, 0, stream>>>(
            x, past, next_past, lanes, T, C_in, K);
        if (cudaGetLastError() != cudaSuccess) return -1;
    }
    return 0;
}

static void causal_conv1d_ref(
        const std::vector<float> & x,
        const std::vector<float> & past,
        const std::vector<float> & weight,
        const std::vector<float> & bias,
        std::vector<float> & out,
        std::vector<float> & next,
        int T,
        int C_in,
        int C_out,
        int K) {
    out.assign((size_t) T * C_out, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int co = 0; co < C_out; ++co) {
            float acc = bias[co];
            for (int k = 0; k < K; ++k) {
                const int src_t = t + k - (K - 1);
                for (int ci = 0; ci < C_in; ++ci) {
                    const float xv = src_t < 0
                        ? past[(src_t + K - 1) * C_in + ci]
                        : x[src_t * C_in + ci];
                    acc += xv * weight[(co * C_in + ci) * K + k];
                }
            }
            out[t * C_out + co] = acc;
        }
    }
    next.assign((size_t) (K - 1) * C_in, 0.0f);
    for (int j = 0; j < K - 1; ++j) {
        const int src_t = T + j - (K - 1);
        for (int ci = 0; ci < C_in; ++ci) {
            next[j * C_in + ci] = src_t < 0
                ? past[(src_t + K - 1) * C_in + ci]
                : x[src_t * C_in + ci];
        }
    }
}

static void causal_conv1d_stride_ref(
        const std::vector<float> & x,
        const std::vector<float> & past,
        const std::vector<float> & weight,
        const std::vector<float> & bias,
        std::vector<float> & out,
        std::vector<float> & next,
        int T,
        int T_out,
        int C_in,
        int C_out,
        int K,
        int stride) {
    out.assign((size_t) T_out * C_out, 0.0f);
    for (int t = 0; t < T_out; ++t) {
        for (int co = 0; co < C_out; ++co) {
            float acc = bias[co];
            for (int k = 0; k < K; ++k) {
                const int src_t = t * stride + k - (K - 1);
                for (int ci = 0; ci < C_in; ++ci) {
                    const float xv = src_t < 0
                        ? past[(src_t + K - 1) * C_in + ci]
                        : (src_t < T ? x[src_t * C_in + ci] : 0.0f);
                    acc += xv * weight[(co * C_in + ci) * K + k];
                }
            }
            out[t * C_out + co] = acc;
        }
    }
    next.assign((size_t) (K - 1) * C_in, 0.0f);
    for (int j = 0; j < K - 1; ++j) {
        const int src_t = T + j - (K - 1);
        for (int ci = 0; ci < C_in; ++ci) {
            next[j * C_in + ci] = src_t < 0
                ? past[(src_t + K - 1) * C_in + ci]
                : x[src_t * C_in + ci];
        }
    }
}

extern "C" int lingbot_causal_conv1d_cache_cuda_smoke(cudaStream_t stream) {
    const int T = 5;
    const int C_in = 4;
    const int C_out = 3;
    const int K = 3;
    std::vector<float> x((size_t) T * C_in);
    std::vector<float> past((size_t) (K - 1) * C_in);
    std::vector<float> w((size_t) C_out * C_in * K);
    std::vector<float> b((size_t) C_out);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float) i * 0.13f) * 0.2f;
    for (size_t i = 0; i < past.size(); ++i) past[i] = std::cos((float) i * 0.17f) * 0.15f;
    for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin((float) i * 0.07f) * 0.1f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = 0.01f * (float) (i + 1);

    std::vector<float> ref_out, ref_next;
    causal_conv1d_ref(x, past, w, b, ref_out, ref_next, T, C_in, C_out, K);
    std::vector<float> out(ref_out.size(), 0.0f);
    std::vector<float> next(ref_next.size(), 0.0f);

    float * dx = nullptr;
    float * dp = nullptr;
    float * dw = nullptr;
    float * db = nullptr;
    float * dout = nullptr;
    float * dn = nullptr;
    CUDA_CHECK_RET(cudaMalloc(&dx, x.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dp, past.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dw, w.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&db, b.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dout, out.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dn, next.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMemcpyAsync(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(dp, past.data(), past.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    if (lingbot_causal_conv1d_cache_f32(dx, dp, dw, db, dout, dn, T, C_in, C_out, K, stream) != 0) {
        cudaFree(dx); cudaFree(dp); cudaFree(dw); cudaFree(db); cudaFree(dout); cudaFree(dn);
        return -1;
    }
    CUDA_CHECK_RET(cudaMemcpyAsync(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(next.data(), dn, next.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_RET(cudaStreamSynchronize(stream));
    cudaFree(dx); cudaFree(dp); cudaFree(dw); cudaFree(db); cudaFree(dout); cudaFree(dn);

    double max_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) max_diff = fmax(max_diff, fabs((double) out[i] - (double) ref_out[i]));
    double cache_diff = 0.0;
    for (size_t i = 0; i < next.size(); ++i) cache_diff = fmax(cache_diff, fabs((double) next[i] - (double) ref_next[i]));
    std::printf("vla(lingbot_causal_conv): cuda smoke max_diff=%.9g cache_diff=%.9g\n", max_diff, cache_diff);
    return max_diff < 1e-6 && cache_diff < 1e-6 ? 0 : -1;
}

extern "C" int lingbot_causal_conv1d_cache_vae_smoke(cudaStream_t stream) {
    const int T = 4;
    const int C_in = 32;
    const int C_out = 32;
    const int K = 3;
    std::vector<float> x((size_t) T * C_in);
    std::vector<float> past((size_t) (K - 1) * C_in);
    std::vector<float> w((size_t) C_out * C_in * K);
    std::vector<float> b((size_t) C_out);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin((float) i * 0.011f) * 0.08f;
    for (size_t i = 0; i < past.size(); ++i) past[i] = std::cos((float) i * 0.013f) * 0.06f;
    for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin((float) i * 0.017f) * 0.04f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = std::cos((float) i * 0.019f) * 0.01f;

    std::vector<float> ref_out, ref_next;
    causal_conv1d_ref(x, past, w, b, ref_out, ref_next, T, C_in, C_out, K);
    std::vector<float> out(ref_out.size(), 0.0f);
    std::vector<float> next(ref_next.size(), 0.0f);

    float * dx = nullptr;
    float * dp = nullptr;
    float * dw = nullptr;
    float * db = nullptr;
    float * dout = nullptr;
    float * dn = nullptr;
    CUDA_CHECK_RET(cudaMalloc(&dx, x.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dp, past.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dw, w.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&db, b.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dout, out.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dn, next.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMemcpyAsync(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(dp, past.data(), past.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    if (lingbot_causal_conv1d_cache_f32(dx, dp, dw, db, dout, dn, T, C_in, C_out, K, stream) != 0) {
        cudaFree(dx); cudaFree(dp); cudaFree(dw); cudaFree(db); cudaFree(dout); cudaFree(dn);
        return -1;
    }
    CUDA_CHECK_RET(cudaMemcpyAsync(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(next.data(), dn, next.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_RET(cudaStreamSynchronize(stream));
    cudaFree(dx); cudaFree(dp); cudaFree(dw); cudaFree(db); cudaFree(dout); cudaFree(dn);

    double max_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) max_diff = fmax(max_diff, fabs((double) out[i] - (double) ref_out[i]));
    double cache_diff = 0.0;
    for (size_t i = 0; i < next.size(); ++i) cache_diff = fmax(cache_diff, fabs((double) next[i] - (double) ref_next[i]));
    std::printf("vla(lingbot_causal_conv): VAE-like cuda smoke T=%d Cin=%d Cout=%d K=%d max_diff=%.9g cache_diff=%.9g\n",
                T, C_in, C_out, K, max_diff, cache_diff);
    if (!(max_diff < 1e-5 && cache_diff < 1e-6)) return -1;

    const int stride = 2;
    const int T_out = (T + stride - 1) / stride;
    causal_conv1d_stride_ref(x, past, w, b, ref_out, ref_next, T, T_out, C_in, C_out, K, stride);
    out.assign(ref_out.size(), 0.0f);
    next.assign(ref_next.size(), 0.0f);
    dx = nullptr; dp = nullptr; dw = nullptr; db = nullptr; dout = nullptr; dn = nullptr;
    CUDA_CHECK_RET(cudaMalloc(&dx, x.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dp, past.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dw, w.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&db, b.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dout, out.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&dn, next.size() * sizeof(float)));
    CUDA_CHECK_RET(cudaMemcpyAsync(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(dp, past.data(), past.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    if (lingbot_causal_conv1d_cache_f32_batched_stride(dx, dp, dw, db, dout, dn,
                                                       1, T, T_out, C_in, C_out, K, stride, stream) != 0) {
        cudaFree(dx); cudaFree(dp); cudaFree(dw); cudaFree(db); cudaFree(dout); cudaFree(dn);
        return -1;
    }
    CUDA_CHECK_RET(cudaMemcpyAsync(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_RET(cudaMemcpyAsync(next.data(), dn, next.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_RET(cudaStreamSynchronize(stream));
    cudaFree(dx); cudaFree(dp); cudaFree(dw); cudaFree(db); cudaFree(dout); cudaFree(dn);

    max_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) max_diff = fmax(max_diff, fabs((double) out[i] - (double) ref_out[i]));
    cache_diff = 0.0;
    for (size_t i = 0; i < next.size(); ++i) cache_diff = fmax(cache_diff, fabs((double) next[i] - (double) ref_next[i]));
    std::printf("vla(lingbot_causal_conv): VAE-like stride cuda smoke T=%d Tout=%d stride=%d max_diff=%.9g cache_diff=%.9g\n",
                T, T_out, stride, max_diff, cache_diff);
    return max_diff < 1e-5 && cache_diff < 1e-6 ? 0 : -1;
}

__device__ __forceinline__ size_t vae_whdc_idx_dev(int w, int h, int t, int c, int W, int H, int T) {
    return (size_t) w + (size_t) W * ((size_t) h + (size_t) H * ((size_t) t + (size_t) T * c));
}

__global__ void vae_mid_l2_norm_kernel(
        const float * __restrict__ in_whdc,
        const float * __restrict__ gamma,
        float * __restrict__ normed,
        int W,
        int H,
        int T,
        int C) {
    const int tok = blockIdx.x;
    const int t = tok % T;
    const int lane = tok / T;
    const int w = lane % W;
    const int h = lane / W;

    __shared__ double red[256];
    double ss = 0.0;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const double x = (double) in_whdc[vae_whdc_idx_dev(w, h, t, c, W, H, T)];
        ss += x * x;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const double l2 = sqrt(red[0]);
    const double norm_scale = l2 > 1e-12 ? sqrt((double) C) / l2 : 0.0;

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        const float x = in_whdc[vae_whdc_idx_dev(w, h, t, c, W, H, T)];
        normed[(size_t) tok * C + c] = (float) ((double) x * norm_scale) * gamma[c];
    }
}

__global__ void vae_linear_kernel(
        const float * __restrict__ x,
        const float * __restrict__ weight,
        const float * __restrict__ bias,
        float * __restrict__ out,
        int tokens,
        int in_dim,
        int out_dim) {
    const int o = blockIdx.x;
    const int tok = blockIdx.y;
    if (o >= out_dim || tok >= tokens) return;
    __shared__ float red[256];
    float acc = 0.0f;
    const float * x_row = x + (size_t) tok * in_dim;
    const float * w_row = weight + (size_t) o * in_dim;
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        acc += x_row[i] * w_row[i];
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[(size_t) tok * out_dim + o] = red[0] + (bias ? bias[o] : 0.0f);
    }
}

__global__ void vae_dense_attn_qkv_kernel(
        const float * __restrict__ qkv,
        float * __restrict__ ctx,
        int W,
        int H,
        int T,
        int C,
        float scale) {
    const int tokens = W * H * T;
    const int qi = blockIdx.x;
    if (qi >= tokens) return;
    const int q_t = qi % T;
    const int spatial_tokens = W * H;

    extern __shared__ float smem[];
    float * acc = smem;
    float * red = smem + C;
    for (int d = threadIdx.x; d < C; d += blockDim.x) acc[d] = 0.0f;
    __shared__ float row_max_s;
    __shared__ float row_sum_s;
    if (threadIdx.x == 0) {
        row_max_s = -INFINITY;
        row_sum_s = 0.0f;
    }
    __syncthreads();

    const float * q = qkv + (size_t) qi * (3 * C);
    for (int k_sp = 0; k_sp < spatial_tokens; ++k_sp) {
        const int kj = k_sp * T + q_t;
        const float * k = qkv + (size_t) kj * (3 * C) + C;
        float dot = 0.0f;
        for (int d = threadIdx.x; d < C; d += blockDim.x) {
            dot += q[d] * k[d];
        }
        red[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) row_max_s = fmaxf(row_max_s, red[0] * scale);
        __syncthreads();
    }

    for (int k_sp = 0; k_sp < spatial_tokens; ++k_sp) {
        const int kj = k_sp * T + q_t;
        const float * k = qkv + (size_t) kj * (3 * C) + C;
        const float * v = qkv + (size_t) kj * (3 * C) + 2 * C;
        float dot = 0.0f;
        for (int d = threadIdx.x; d < C; d += blockDim.x) {
            dot += q[d] * k[d];
        }
        red[threadIdx.x] = dot;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
            __syncthreads();
        }
        const float a = expf(red[0] * scale - row_max_s);
        if (threadIdx.x == 0) row_sum_s += a;
        for (int d = threadIdx.x; d < C; d += blockDim.x) {
            acc[d] += a * v[d];
        }
        __syncthreads();
    }

    for (int d = threadIdx.x; d < C; d += blockDim.x) {
        ctx[(size_t) qi * C + d] = row_sum_s > 0.0f ? acc[d] / row_sum_s : 0.0f;
    }
}

__global__ void vae_mid_residual_to_whdc_kernel(
        const float * __restrict__ in_whdc,
        const float * __restrict__ proj,
        float * __restrict__ out_whdc,
        int W,
        int H,
        int T,
        int C) {
    const size_t total = (size_t) W * H * T * C;
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int w = idx % W;
    const size_t q0 = idx / W;
    const int h = q0 % H;
    const size_t q1 = q0 / H;
    const int t = q1 % T;
    const int c = q1 / T;
    const int tok = (h * W + w) * T + t;
    out_whdc[idx] = in_whdc[idx] + proj[(size_t) tok * C + c];
}

struct LingBotVaeMidAttnScratch {
    float * normed = nullptr;
    float * qkv = nullptr;
    float * ctx = nullptr;
    float * proj = nullptr;
    size_t normed_elems = 0;
    size_t qkv_elems = 0;
    size_t ctx_elems = 0;
    size_t proj_elems = 0;

    ~LingBotVaeMidAttnScratch() {
        if (normed) cudaFree(normed);
        if (qkv) cudaFree(qkv);
        if (ctx) cudaFree(ctx);
        if (proj) cudaFree(proj);
    }

    bool ensure_one(float ** ptr, size_t * capacity, size_t elems) {
        if (*ptr && *capacity >= elems) return true;
        if (*ptr) {
            cudaFree(*ptr);
            *ptr = nullptr;
            *capacity = 0;
        }
        if (cudaMalloc(reinterpret_cast<void **>(ptr), elems * sizeof(float)) != cudaSuccess) {
            *ptr = nullptr;
            return false;
        }
        *capacity = elems;
        return true;
    }

    bool ensure(size_t tokens, int C) {
        const size_t tc = tokens * (size_t) C;
        return ensure_one(&normed, &normed_elems, tc) &&
               ensure_one(&qkv, &qkv_elems, tc * 3) &&
               ensure_one(&ctx, &ctx_elems, tc) &&
               ensure_one(&proj, &proj_elems, tc);
    }
};

extern "C" int lingbot_vae_mid_attn_f32(
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
    cudaStream_t stream) {
    if (!in_whdc || !norm_gamma || !qkv_weight || !qkv_bias || !proj_weight || !proj_bias || !out_whdc) return -1;
    if (W <= 0 || H <= 0 || T <= 0 || C <= 0 || C % 32 != 0) return -1;
    const int tokens = W * H * T;
    const int qkv_dim = 3 * C;
    static std::mutex scratch_mu;
    static LingBotVaeMidAttnScratch scratch;
    std::lock_guard<std::mutex> scratch_lock(scratch_mu);
    if (!scratch.ensure((size_t) tokens, C)) return -1;

    vae_mid_l2_norm_kernel<<<dim3(tokens, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        in_whdc, norm_gamma, scratch.normed, W, H, T, C);
    if (cudaGetLastError() != cudaSuccess) return -1;

    vae_linear_kernel<<<dim3(qkv_dim, tokens, 1), dim3(256, 1, 1), 0, stream>>>(
        scratch.normed, qkv_weight, qkv_bias, scratch.qkv, tokens, C, qkv_dim);
    if (cudaGetLastError() != cudaSuccess) return -1;

    vae_dense_attn_qkv_kernel<<<dim3(tokens, 1, 1), dim3(256, 1, 1),
                                (size_t) (C + 256) * sizeof(float), stream>>>(
        scratch.qkv, scratch.ctx, W, H, T, C, 1.0f / sqrtf((float) C));
    if (cudaGetLastError() != cudaSuccess) return -1;

    vae_linear_kernel<<<dim3(C, tokens, 1), dim3(256, 1, 1), 0, stream>>>(
        scratch.ctx, proj_weight, proj_bias, scratch.proj, tokens, C, C);
    if (cudaGetLastError() != cudaSuccess) return -1;

    const size_t total = (size_t) W * H * T * C;
    vae_mid_residual_to_whdc_kernel<<<(unsigned int) ((total + 255) / 256), 256, 0, stream>>>(
        in_whdc, scratch.proj, out_whdc, W, H, T, C);
    if (cudaGetLastError() != cudaSuccess) return -1;
    return 0;
}
