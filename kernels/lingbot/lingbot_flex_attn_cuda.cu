// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "lingbot_flex_attn_cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

#define CUDA_CHECK_RET(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "vla(lingbot_flex): CUDA error %s at %s:%d (%s)\n", cudaGetErrorString(_e), __FILE__, __LINE__, #x); \
    return -1; }} while (0)

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

__global__ void vae_mid_group_norm_kernel(
        const float * __restrict__ in_whdc,
        const float * __restrict__ gamma,
        float * __restrict__ normed,
        int W,
        int H,
        int T,
        int C,
        int groups) {
    const int tok = blockIdx.x;
    const int group = blockIdx.y;
    const int group_size = C / groups;
    const int t = tok % T;
    const int lane = tok / T;
    const int w = lane % W;
    const int h = lane / W;
    const int c0 = group * group_size;

    __shared__ float red[256];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < group_size; j += blockDim.x) {
        sum += in_whdc[vae_whdc_idx_dev(w, h, t, c0 + j, W, H, T)];
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean = red[0] / (float) group_size;

    float sq = 0.0f;
    for (int j = threadIdx.x; j < group_size; j += blockDim.x) {
        const float d = in_whdc[vae_whdc_idx_dev(w, h, t, c0 + j, W, H, T)] - mean;
        sq += d * d;
    }
    red[threadIdx.x] = sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv_std = rsqrtf(red[0] / (float) group_size + 1e-6f);

    for (int j = threadIdx.x; j < group_size; j += blockDim.x) {
        const int c = c0 + j;
        const float x = in_whdc[vae_whdc_idx_dev(w, h, t, c, W, H, T)];
        normed[(size_t) tok * C + c] = (x - mean) * inv_std * gamma[c];
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
        int tokens,
        int C,
        float scale) {
    const int qi = blockIdx.x;
    if (qi >= tokens) return;

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
    for (int kj = 0; kj < tokens; ++kj) {
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

    for (int kj = 0; kj < tokens; ++kj) {
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
    float * normed = nullptr;
    float * qkv = nullptr;
    float * ctx = nullptr;
    float * proj = nullptr;
    auto cleanup = [&]() {
        if (normed) cudaFree(normed);
        if (qkv) cudaFree(qkv);
        if (ctx) cudaFree(ctx);
        if (proj) cudaFree(proj);
    };
    if (cudaMalloc(&normed, (size_t) tokens * C * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&qkv, (size_t) tokens * qkv_dim * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&ctx, (size_t) tokens * C * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&proj, (size_t) tokens * C * sizeof(float)) != cudaSuccess) {
        cleanup();
        return -1;
    }

    vae_mid_group_norm_kernel<<<dim3(tokens, 32, 1), dim3(128, 1, 1), 0, stream>>>(
        in_whdc, norm_gamma, normed, W, H, T, C, 32);
    if (cudaGetLastError() != cudaSuccess) { cleanup(); return -1; }

    vae_linear_kernel<<<dim3(qkv_dim, tokens, 1), dim3(256, 1, 1), 0, stream>>>(
        normed, qkv_weight, qkv_bias, qkv, tokens, C, qkv_dim);
    if (cudaGetLastError() != cudaSuccess) { cleanup(); return -1; }

    vae_dense_attn_qkv_kernel<<<dim3(tokens, 1, 1), dim3(256, 1, 1),
                                (size_t) (C + 256) * sizeof(float), stream>>>(
        qkv, ctx, tokens, C, 1.0f / sqrtf((float) C));
    if (cudaGetLastError() != cudaSuccess) { cleanup(); return -1; }

    vae_linear_kernel<<<dim3(C, tokens, 1), dim3(256, 1, 1), 0, stream>>>(
        ctx, proj_weight, proj_bias, proj, tokens, C, C);
    if (cudaGetLastError() != cudaSuccess) { cleanup(); return -1; }

    const size_t total = (size_t) W * H * T * C;
    vae_mid_residual_to_whdc_kernel<<<(unsigned int) ((total + 255) / 256), 256, 0, stream>>>(
        in_whdc, proj, out_whdc, W, H, T, C);
    if (cudaGetLastError() != cudaSuccess) { cleanup(); return -1; }
    cleanup();
    return 0;
}
