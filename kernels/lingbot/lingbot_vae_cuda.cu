// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#include "lingbot_vae_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#ifdef VLA_LINGBOT_USE_CUDNN
#include <cudnn.h>
#endif
#include <cmath>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {

static bool lingbot_vae_env_enabled(const char * name) {
    const char * v = std::getenv(name);
    return v && *v && v[0] != '0';
}

static bool lingbot_vae_env_disabled(const char * name) {
    return lingbot_vae_env_enabled(name);
}

static bool lingbot_vae_official_fast_path_enabled() {
    return !lingbot_vae_env_enabled("VLA_LINGBOT_OFFICIAL_FAST_PATH_DISABLE");
}

__device__ __forceinline__ size_t whdc_index_dyn(
        int w,
        int h,
        int t,
        int c,
        int W,
        int H,
        int T) {
    return static_cast<size_t>(w) +
           static_cast<size_t>(W) *
               (static_cast<size_t>(h) +
                static_cast<size_t>(H) *
                    (static_cast<size_t>(t) +
                     static_cast<size_t>(T) * static_cast<size_t>(c)));
}

__global__ void norm_silu_whdc_f32w_kernel(
        const float * input,
        const float * gamma_f32,
        float * output,
        int W,
        int H,
        int T,
        int C) {
    const int site = blockIdx.x * blockDim.x + threadIdx.x;
    const int sites = W * H * T;
    if (site >= sites) return;
    int r = site;
    const int w = r % W; r /= W;
    const int h = r % H; r /= H;
    const int t = r;

    float ss = 0.0f;
    for (int c = 0; c < C; ++c) {
        const float v = input[whdc_index_dyn(w, h, t, c, W, H, T)];
        ss += v * v;
    }
    const float inv = rsqrtf(ss / static_cast<float>(C) + 1e-12f);
    for (int c = 0; c < C; ++c) {
        const float x = input[whdc_index_dyn(w, h, t, c, W, H, T)] *
                        inv * gamma_f32[c];
        output[whdc_index_dyn(w, h, t, c, W, H, T)] = x / (1.0f + expf(-x));
    }
}

__global__ void causal_conv3d_ks3_whdc_f32w_kernel(
        const float * input,
        const float * weight_f32,
        const float * bias_f32,
        float * output,
        int W,
        int H,
        int T,
        int in_C,
        int out_C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H * T * out_C;
    if (idx >= total) return;
    int r = idx;
    const int w = r % W; r /= W;
    const int h = r % H; r /= H;
    const int t = r % T; r /= T;
    const int co = r;

    float sum = bias_f32[co];
    for (int ci = 0; ci < in_C; ++ci) {
        for (int kt = 0; kt < 3; ++kt) {
            const int it = t + kt - 2;
            if (it < 0 || it >= T) continue;
            for (int kh = 0; kh < 3; ++kh) {
                const int ih = h + kh - 1;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < 3; ++kw) {
                    const int iw = w + kw - 1;
                    if (iw < 0 || iw >= W) continue;
                    const size_t wi =
                        (((static_cast<size_t>(co) * in_C + ci) * 3 + kt) * 3 + kh) * 3 + kw;
                    sum += input[whdc_index_dyn(iw, ih, it, ci, W, H, T)] * weight_f32[wi];
                }
            }
        }
    }
    output[whdc_index_dyn(w, h, t, co, W, H, T)] = sum;
}

__global__ void cached_causal_conv3d_ks3_whdc_f32w_kernel(
        const float * input,
        const float * cache,
        const float * weight_f32,
        const float * bias_f32,
        float * output,
        int W,
        int H,
        int T,
        int cache_T,
        int in_C,
        int out_C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H * T * out_C;
    if (idx >= total) return;
    int r = idx;
    const int w = r % W; r /= W;
    const int h = r % H; r /= H;
    const int t = r % T; r /= T;
    const int co = r;

    const int left_pad = cache_T < 2 ? 2 - cache_T : 0;
    float sum = bias_f32[co];
    for (int ci = 0; ci < in_C; ++ci) {
        for (int kt = 0; kt < 3; ++kt) {
            const int src_t = t + kt - left_pad;
            const bool from_cache = src_t >= 0 && src_t < cache_T;
            const int cur_t = src_t - cache_T;
            if (!from_cache && (cur_t < 0 || cur_t >= T)) continue;
            for (int kh = 0; kh < 3; ++kh) {
                const int ih = h + kh - 1;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < 3; ++kw) {
                    const int iw = w + kw - 1;
                    if (iw < 0 || iw >= W) continue;
                    const float x = from_cache
                        ? cache[whdc_index_dyn(iw, ih, src_t, ci, W, H, cache_T)]
                        : input[whdc_index_dyn(iw, ih, cur_t, ci, W, H, T)];
                    const size_t wi =
                        (((static_cast<size_t>(co) * in_C + ci) * 3 + kt) * 3 + kh) * 3 + kw;
                    sum += x * weight_f32[wi];
                }
            }
        }
    }
    output[whdc_index_dyn(w, h, t, co, W, H, T)] = sum;
}

__global__ void cached_causal_pad_time_whdc_f32_kernel(
        const float * input,
        const float * cache,
        float * padded,
        int W,
        int H,
        int T,
        int cache_T,
        int padded_T,
        int in_C) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(W) * H * padded_T * in_C;
    if (idx >= total) return;
    size_t r = idx;
    const int w = static_cast<int>(r % W); r /= W;
    const int h = static_cast<int>(r % H); r /= H;
    const int tp = static_cast<int>(r % padded_T); r /= padded_T;
    const int c = static_cast<int>(r);

    const int left_pad = cache_T < 2 ? 2 - cache_T : 0;
    const int src_t = tp - left_pad;
    if (src_t < 0) {
        padded[idx] = 0.0f;
    } else if (src_t < cache_T) {
        padded[idx] = cache[whdc_index_dyn(w, h, src_t, c, W, H, cache_T)];
    } else {
        padded[idx] = input[whdc_index_dyn(w, h, src_t - cache_T, c, W, H, T)];
    }
}

__global__ void add_bias_whdc_f32_kernel(
        float * output,
        const float * bias,
        int W,
        int H,
        int T,
        int C) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(W) * H * T * C;
    if (idx >= total) return;
    const int c = static_cast<int>(idx / (static_cast<size_t>(W) * H * T));
    output[idx] += bias[c];
}

#ifdef VLA_LINGBOT_USE_CUDNN
static bool lingbot_vae_cudnn_runtime_enabled() {
    if (lingbot_vae_env_disabled("VLA_LINGBOT_VAE_CUDNN_DISABLE")) return false;
    if (lingbot_vae_env_enabled("VLA_LINGBOT_VAE_CUDNN")) return true;
    return lingbot_vae_official_fast_path_enabled() ||
           lingbot_vae_env_enabled("VLA_LINGBOT_VAE_PERSISTENT_ENCODER");
}

struct LingBotVaeCudnnConvShape {
    int W = 0;
    int H = 0;
    int T = 0;
    int padded_T = 0;
    int in_C = 0;
    int out_C = 0;

    bool operator==(const LingBotVaeCudnnConvShape & other) const {
        return W == other.W &&
               H == other.H &&
               T == other.T &&
               padded_T == other.padded_T &&
               in_C == other.in_C &&
               out_C == other.out_C;
    }
};

struct LingBotVaeCudnnConvEntry {
    LingBotVaeCudnnConvShape shape;
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t workspace_bytes = 0;
    bool initialized = false;

    LingBotVaeCudnnConvEntry() = default;
    LingBotVaeCudnnConvEntry(const LingBotVaeCudnnConvEntry &) = delete;
    LingBotVaeCudnnConvEntry & operator=(const LingBotVaeCudnnConvEntry &) = delete;

    ~LingBotVaeCudnnConvEntry() {
        if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
        if (w_desc) cudnnDestroyFilterDescriptor(w_desc);
        if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
        if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
    }
};

struct LingBotVaeCudnnConvCache {
    cudnnHandle_t handle = nullptr;
    float * padded = nullptr;
    size_t padded_elems = 0;
    void * workspace = nullptr;
    size_t workspace_bytes = 0;
    std::vector<std::unique_ptr<LingBotVaeCudnnConvEntry>> entries;

    ~LingBotVaeCudnnConvCache() {
        entries.clear();
        if (workspace) cudaFree(workspace);
        if (padded) cudaFree(padded);
        if (handle) cudnnDestroy(handle);
    }
};

static LingBotVaeCudnnConvCache g_lingbot_vae_cudnn_conv_cache;

static bool lingbot_vae_cudnn_ensure_buffer(float ** ptr, size_t * capacity, size_t elems) {
    if (*ptr && *capacity >= elems) return true;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *capacity = 0;
    if (elems == 0) return true;
    if (cudaMalloc(reinterpret_cast<void **>(ptr), elems * sizeof(float)) != cudaSuccess) {
        return false;
    }
    *capacity = elems;
    return true;
}

static bool lingbot_vae_cudnn_ensure_workspace(void ** ptr, size_t * capacity, size_t bytes) {
    if (*ptr && *capacity >= bytes) return true;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *capacity = 0;
    if (bytes == 0) return true;
    if (cudaMalloc(ptr, bytes) != cudaSuccess) return false;
    *capacity = bytes;
    return true;
}

static LingBotVaeCudnnConvEntry * lingbot_vae_cudnn_get_conv_entry(
        LingBotVaeCudnnConvCache & cache,
        const LingBotVaeCudnnConvShape & shape) {
    for (const auto & entry : cache.entries) {
        if (entry->initialized && entry->shape == shape) {
            return entry.get();
        }
    }

    auto entry = std::make_unique<LingBotVaeCudnnConvEntry>();
    entry->shape = shape;
    if (cudnnCreateTensorDescriptor(&entry->x_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateTensorDescriptor(&entry->y_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateFilterDescriptor(&entry->w_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateConvolutionDescriptor(&entry->conv_desc) != CUDNN_STATUS_SUCCESS) {
        return nullptr;
    }

    const int x_dim[5] = {1, shape.in_C, shape.padded_T, shape.H, shape.W};
    const int x_stride[5] = {shape.in_C * shape.padded_T * shape.H * shape.W,
                             shape.padded_T * shape.H * shape.W,
                             shape.H * shape.W,
                             shape.W,
                             1};
    const int y_dim[5] = {1, shape.out_C, shape.T, shape.H, shape.W};
    const int y_stride[5] = {shape.out_C * shape.T * shape.H * shape.W,
                             shape.T * shape.H * shape.W,
                             shape.H * shape.W,
                             shape.W,
                             1};
    const int filt_dim[5] = {shape.out_C, shape.in_C, 3, 3, 3};
    const int pad[3] = {0, 1, 1};
    const int stride[3] = {1, 1, 1};
    const int dilation[3] = {1, 1, 1};
    if (cudnnSetTensorNdDescriptor(entry->x_desc, CUDNN_DATA_FLOAT, 5, x_dim, x_stride) != CUDNN_STATUS_SUCCESS ||
        cudnnSetTensorNdDescriptor(entry->y_desc, CUDNN_DATA_FLOAT, 5, y_dim, y_stride) != CUDNN_STATUS_SUCCESS ||
        cudnnSetFilterNdDescriptor(entry->w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filt_dim) != CUDNN_STATUS_SUCCESS ||
        cudnnSetConvolutionNdDescriptor(entry->conv_desc, 3, pad, stride, dilation,
                                        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS) {
        return nullptr;
    }

    entry->algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    entry->workspace_bytes = 0;
    if (cudnnGetConvolutionForwardWorkspaceSize(cache.handle,
                                                entry->x_desc,
                                                entry->w_desc,
                                                entry->conv_desc,
                                                entry->y_desc,
                                                entry->algo,
                                                &entry->workspace_bytes) != CUDNN_STATUS_SUCCESS) {
        entry->algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        entry->workspace_bytes = 0;
    }
    entry->initialized = true;
    LingBotVaeCudnnConvEntry * result = entry.get();
    cache.entries.emplace_back(std::move(entry));
    return result;
}

static int lingbot_vae_cached_causal_conv3d_ks3_whdc_cudnn(
        const float * input,
        const float * cache,
        const float * weight_f32,
        const float * bias_f32,
        float * output,
        int W,
        int H,
        int T,
        int cache_T,
        int in_C,
        int out_C,
        cudaStream_t stream) {
    auto & cudnn_cache = g_lingbot_vae_cudnn_conv_cache;
    if (!cudnn_cache.handle &&
        cudnnCreate(&cudnn_cache.handle) != CUDNN_STATUS_SUCCESS) {
        return -1;
    }
    if (cudnnSetStream(cudnn_cache.handle, stream) != CUDNN_STATUS_SUCCESS) return -1;

    const int left_pad = cache_T < 2 ? 2 - cache_T : 0;
    const int padded_T = left_pad + cache_T + T;
    if (padded_T < 3) return -1;
    const size_t padded_elems = static_cast<size_t>(W) * H * padded_T * in_C;
    if (!lingbot_vae_cudnn_ensure_buffer(&cudnn_cache.padded,
                                         &cudnn_cache.padded_elems,
                                         padded_elems)) {
        return -1;
    }

    constexpr int block = 256;
    cached_causal_pad_time_whdc_f32_kernel<<<static_cast<unsigned int>((padded_elems + block - 1) / block),
                                             block, 0, stream>>>(
        input, cache, cudnn_cache.padded, W, H, T, cache_T, padded_T, in_C);
    if (cudaGetLastError() != cudaSuccess) return -1;

    LingBotVaeCudnnConvShape shape;
    shape.W = W;
    shape.H = H;
    shape.T = T;
    shape.padded_T = padded_T;
    shape.in_C = in_C;
    shape.out_C = out_C;
    LingBotVaeCudnnConvEntry * entry = lingbot_vae_cudnn_get_conv_entry(cudnn_cache, shape);
    int ok = entry ? 0 : -1;
    if (ok == 0 && !lingbot_vae_cudnn_ensure_workspace(&cudnn_cache.workspace,
                                                       &cudnn_cache.workspace_bytes,
                                                       entry->workspace_bytes)) {
        ok = -1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (ok == 0 &&
        cudnnConvolutionForward(cudnn_cache.handle,
                                &alpha,
                                entry->x_desc,
                                cudnn_cache.padded,
                                entry->w_desc,
                                weight_f32,
                                entry->conv_desc,
                                entry->algo,
                                cudnn_cache.workspace,
                                entry->workspace_bytes,
                                &beta,
                                entry->y_desc,
                                output) != CUDNN_STATUS_SUCCESS) {
        ok = -1;
    }
    if (ok == 0) {
        const size_t total = static_cast<size_t>(W) * H * T * out_C;
        add_bias_whdc_f32_kernel<<<static_cast<unsigned int>((total + block - 1) / block),
                                   block, 0, stream>>>(output, bias_f32, W, H, T, out_C);
        if (cudaGetLastError() != cudaSuccess) ok = -1;
    }

    return ok;
}
#endif

struct LingBotVaeCublasCache {
    cublasHandle_t handle = nullptr;

    ~LingBotVaeCublasCache() {
        if (handle) cublasDestroy(handle);
    }
};

static LingBotVaeCublasCache g_lingbot_vae_cublas_cache;

static bool lingbot_vae_cublas_runtime_enabled() {
    if (lingbot_vae_env_disabled("VLA_LINGBOT_VAE_CUBLAS_DISABLE")) return false;
    if (lingbot_vae_env_enabled("VLA_LINGBOT_VAE_CUBLAS")) return true;
    return lingbot_vae_official_fast_path_enabled() ||
           lingbot_vae_env_enabled("VLA_LINGBOT_VAE_PERSISTENT_ENCODER");
}

static int lingbot_vae_conv1x1x1_whdc_cublas(
        const float * input,
        const float * weight_f32,
        const float * bias_f32,
        float * output,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        cudaStream_t stream) {
    auto & cache = g_lingbot_vae_cublas_cache;
    if (!cache.handle && cublasCreate(&cache.handle) != CUBLAS_STATUS_SUCCESS) return -1;
    if (cublasSetStream(cache.handle, stream) != CUBLAS_STATUS_SUCCESS) return -1;

    const int sites = W * H * T;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t st = cublasSgemm(
        cache.handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        sites,
        out_C,
        in_C,
        &alpha,
        input,
        sites,
        weight_f32,
        in_C,
        &beta,
        output,
        sites);
    if (st != CUBLAS_STATUS_SUCCESS) return -1;

    constexpr int block = 256;
    const size_t total = static_cast<size_t>(sites) * out_C;
    add_bias_whdc_f32_kernel<<<static_cast<unsigned int>((total + block - 1) / block),
                               block, 0, stream>>>(output, bias_f32, W, H, T, out_C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

__global__ void add_whdc_kernel(
        const float * a,
        const float * b,
        float * out,
        size_t elems) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < elems) out[idx] = a[idx] + b[idx];
}

__global__ void spatial_downsample2d_whdc_f32w_kernel(
        const float * input,
        const float * weight_f32,
        const float * bias_f32,
        float * output,
        int W,
        int H,
        int T,
        int C) {
    const int Wo = W / 2;
    const int Ho = H / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Wo * Ho * T * C;
    if (idx >= total) return;
    int r = idx;
    const int ow = r % Wo; r /= Wo;
    const int oh = r % Ho; r /= Ho;
    const int t = r % T; r /= T;
    const int co = r;

    float sum = bias_f32[co];
    for (int ci = 0; ci < C; ++ci) {
        for (int kh = 0; kh < 3; ++kh) {
            const int ih = oh * 2 + kh;
            if (ih >= H + 1) continue;
            for (int kw = 0; kw < 3; ++kw) {
                const int iw = ow * 2 + kw;
                if (iw >= W + 1) continue;
                if (iw >= W || ih >= H) continue;
                const size_t wi = (((static_cast<size_t>(co) * C + ci) * 3 + kh) * 3 + kw);
                sum += input[whdc_index_dyn(iw, ih, t, ci, W, H, T)] * weight_f32[wi];
            }
        }
    }
    output[whdc_index_dyn(ow, oh, t, co, Wo, Ho, T)] = sum;
}

__global__ void spatial_pad_right_bottom_whdc_f32_kernel(
        const float * input,
        float * padded,
        int W,
        int H,
        int T,
        int C) {
    const int Wp = W + 1;
    const int Hp = H + 1;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(Wp) * Hp * T * C;
    if (idx >= total) return;
    size_t r = idx;
    const int w = static_cast<int>(r % Wp); r /= Wp;
    const int h = static_cast<int>(r % Hp); r /= Hp;
    const int t = static_cast<int>(r % T); r /= T;
    const int c = static_cast<int>(r);
    padded[idx] = (w < W && h < H) ? input[whdc_index_dyn(w, h, t, c, W, H, T)] : 0.0f;
}

#ifdef VLA_LINGBOT_USE_CUDNN
struct LingBotVaeCudnnSpatialShape {
    int W = 0;
    int H = 0;
    int T = 0;
    int C = 0;

    bool operator==(const LingBotVaeCudnnSpatialShape & other) const {
        return W == other.W && H == other.H && T == other.T && C == other.C;
    }
};

struct LingBotVaeCudnnSpatialEntry {
    LingBotVaeCudnnSpatialShape shape;
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t workspace_bytes = 0;
    bool initialized = false;

    LingBotVaeCudnnSpatialEntry() = default;
    LingBotVaeCudnnSpatialEntry(const LingBotVaeCudnnSpatialEntry &) = delete;
    LingBotVaeCudnnSpatialEntry & operator=(const LingBotVaeCudnnSpatialEntry &) = delete;

    ~LingBotVaeCudnnSpatialEntry() {
        if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
        if (w_desc) cudnnDestroyFilterDescriptor(w_desc);
        if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
        if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
    }
};

struct LingBotVaeCudnnSpatialCache {
    cudnnHandle_t handle = nullptr;
    float * padded = nullptr;
    size_t padded_elems = 0;
    void * workspace = nullptr;
    size_t workspace_bytes = 0;
    std::vector<std::unique_ptr<LingBotVaeCudnnSpatialEntry>> entries;

    ~LingBotVaeCudnnSpatialCache() {
        entries.clear();
        if (workspace) cudaFree(workspace);
        if (padded) cudaFree(padded);
        if (handle) cudnnDestroy(handle);
    }
};

static LingBotVaeCudnnSpatialCache g_lingbot_vae_cudnn_spatial_cache;

static LingBotVaeCudnnSpatialEntry * lingbot_vae_cudnn_get_spatial_entry(
        LingBotVaeCudnnSpatialCache & cache,
        const LingBotVaeCudnnSpatialShape & shape) {
    for (const auto & entry : cache.entries) {
        if (entry->initialized && entry->shape == shape) return entry.get();
    }
    auto entry = std::make_unique<LingBotVaeCudnnSpatialEntry>();
    entry->shape = shape;
    if (cudnnCreateTensorDescriptor(&entry->x_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateTensorDescriptor(&entry->y_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateFilterDescriptor(&entry->w_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateConvolutionDescriptor(&entry->conv_desc) != CUDNN_STATUS_SUCCESS) {
        return nullptr;
    }
    const int Wp = shape.W + 1;
    const int Hp = shape.H + 1;
    const int Wo = shape.W / 2;
    const int Ho = shape.H / 2;
    const int x_dim[5] = {1, shape.C, shape.T, Hp, Wp};
    const int x_stride[5] = {shape.C * shape.T * Hp * Wp,
                             shape.T * Hp * Wp,
                             Hp * Wp,
                             Wp,
                             1};
    const int y_dim[5] = {1, shape.C, shape.T, Ho, Wo};
    const int y_stride[5] = {shape.C * shape.T * Ho * Wo,
                             shape.T * Ho * Wo,
                             Ho * Wo,
                             Wo,
                             1};
    const int filt_dim[5] = {shape.C, shape.C, 1, 3, 3};
    const int pad[3] = {0, 0, 0};
    const int stride[3] = {1, 2, 2};
    const int dilation[3] = {1, 1, 1};
    if (cudnnSetTensorNdDescriptor(entry->x_desc, CUDNN_DATA_FLOAT, 5, x_dim, x_stride) != CUDNN_STATUS_SUCCESS ||
        cudnnSetTensorNdDescriptor(entry->y_desc, CUDNN_DATA_FLOAT, 5, y_dim, y_stride) != CUDNN_STATUS_SUCCESS ||
        cudnnSetFilterNdDescriptor(entry->w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filt_dim) != CUDNN_STATUS_SUCCESS ||
        cudnnSetConvolutionNdDescriptor(entry->conv_desc, 3, pad, stride, dilation,
                                        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS) {
        return nullptr;
    }
    entry->algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    entry->workspace_bytes = 0;
    if (cudnnGetConvolutionForwardWorkspaceSize(cache.handle,
                                                entry->x_desc,
                                                entry->w_desc,
                                                entry->conv_desc,
                                                entry->y_desc,
                                                entry->algo,
                                                &entry->workspace_bytes) != CUDNN_STATUS_SUCCESS) {
        entry->algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        entry->workspace_bytes = 0;
    }
    entry->initialized = true;
    LingBotVaeCudnnSpatialEntry * result = entry.get();
    cache.entries.emplace_back(std::move(entry));
    return result;
}

static int lingbot_vae_spatial_downsample2d_whdc_cudnn(
        const float * input,
        const float * weight_f32,
        const float * bias_f32,
        float * output,
        int W,
        int H,
        int T,
        int C,
        cudaStream_t stream) {
    auto & cache = g_lingbot_vae_cudnn_spatial_cache;
    if (!cache.handle && cudnnCreate(&cache.handle) != CUDNN_STATUS_SUCCESS) return -1;
    if (cudnnSetStream(cache.handle, stream) != CUDNN_STATUS_SUCCESS) return -1;
    const int Wp = W + 1;
    const int Hp = H + 1;
    const size_t padded_elems = static_cast<size_t>(Wp) * Hp * T * C;
    if (!lingbot_vae_cudnn_ensure_buffer(&cache.padded, &cache.padded_elems, padded_elems)) {
        return -1;
    }
    constexpr int block = 256;
    spatial_pad_right_bottom_whdc_f32_kernel<<<static_cast<unsigned int>((padded_elems + block - 1) / block),
                                               block, 0, stream>>>(input, cache.padded, W, H, T, C);
    if (cudaGetLastError() != cudaSuccess) return -1;

    LingBotVaeCudnnSpatialShape shape;
    shape.W = W;
    shape.H = H;
    shape.T = T;
    shape.C = C;
    LingBotVaeCudnnSpatialEntry * entry = lingbot_vae_cudnn_get_spatial_entry(cache, shape);
    if (!entry) return -1;
    if (!lingbot_vae_cudnn_ensure_workspace(&cache.workspace,
                                            &cache.workspace_bytes,
                                            entry->workspace_bytes)) {
        return -1;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (cudnnConvolutionForward(cache.handle,
                                &alpha,
                                entry->x_desc,
                                cache.padded,
                                entry->w_desc,
                                weight_f32,
                                entry->conv_desc,
                                entry->algo,
                                cache.workspace,
                                entry->workspace_bytes,
                                &beta,
                                entry->y_desc,
                                output) != CUDNN_STATUS_SUCCESS) {
        return -1;
    }
    const size_t total = static_cast<size_t>(W / 2) * (H / 2) * T * C;
    add_bias_whdc_f32_kernel<<<static_cast<unsigned int>((total + block - 1) / block),
                               block, 0, stream>>>(output, bias_f32, W / 2, H / 2, T, C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
#endif

__global__ void conv1x1x1_whdc_f32w_kernel(
        const float * input,
        const float * weight_f32,
        const float * bias_f32,
        float * output,
        int W,
        int H,
        int T,
        int in_C,
        int out_C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H * T * out_C;
    if (idx >= total) return;
    int r = idx;
    const int w = r % W; r /= W;
    const int h = r % H; r /= H;
    const int t = r % T; r /= T;
    const int co = r;
    float sum = bias_f32[co];
    for (int ci = 0; ci < in_C; ++ci) {
        sum += input[whdc_index_dyn(w, h, t, ci, W, H, T)] *
               weight_f32[static_cast<size_t>(co) * in_C + ci];
    }
    output[whdc_index_dyn(w, h, t, co, W, H, T)] = sum;
}

__global__ void avg_down3d_whdc_kernel(
        const float * input,
        float * output,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        int factor_t,
        int factor_s) {
    const int Wo = W / factor_s;
    const int Ho = H / factor_s;
    const int pad_t = (factor_t - (T % factor_t)) % factor_t;
    const int To = (T + pad_t) / factor_t;
    const int factor = factor_t * factor_s * factor_s;
    const int group_size = (in_C * factor) / out_C;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Wo * Ho * To * out_C;
    if (idx >= total) return;
    int r = idx;
    const int wo = r % Wo; r /= Wo;
    const int ho = r % Ho; r /= Ho;
    const int to = r % To; r /= To;
    const int co = r;

    float acc = 0.0f;
    for (int gidx = 0; gidx < group_size; ++gidx) {
        const int flat = co * group_size + gidx;
        const int ci = flat / factor;
        const int rem0 = flat - ci * factor;
        const int ft = rem0 / (factor_s * factor_s);
        const int rem1 = rem0 - ft * factor_s * factor_s;
        const int fs_h = rem1 / factor_s;
        const int fs_w = rem1 - fs_h * factor_s;
        const int src_t = to * factor_t + ft - pad_t;
        if (src_t >= 0 && src_t < T) {
            acc += input[whdc_index_dyn(wo * factor_s + fs_w,
                                        ho * factor_s + fs_h,
                                        src_t,
                                        ci,
                                        W,
                                        H,
                                        T)];
        }
    }
    output[whdc_index_dyn(wo, ho, to, co, Wo, Ho, To)] =
        acc / static_cast<float>(group_size);
}

__global__ void downsample3d_time_stream_one_whdc_f32w_kernel(
        const float * spatial,
        const float * cache,
        const float * weight_f32,
        const float * bias_f32,
        float * output,
        int W,
        int H,
        int T,
        int cache_T,
        int in_C,
        int out_C,
        int To) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H * To * out_C;
    if (idx >= total) return;
    int r = idx;
    const int w = r % W; r /= W;
    const int h = r % H; r /= H;
    const int to = r % To; r /= To;
    const int co = r;

    float sum = bias_f32[co];
    for (int kt = 0; kt < 3; ++kt) {
        const int src = to * 2 + kt;
        for (int ci = 0; ci < in_C; ++ci) {
            const float xv = src == 0
                ? cache[whdc_index_dyn(w, h, cache_T - 1, ci, W, H, cache_T)]
                : spatial[whdc_index_dyn(w, h, src - 1, ci, W, H, T)];
            sum += xv * weight_f32[(static_cast<size_t>(co) * in_C + ci) * 3 + kt];
        }
    }
    output[whdc_index_dyn(w, h, to, co, W, H, To)] = sum;
}

__global__ void update_temporal_cache_current_whdc_f32_kernel(
        const float * input,
        float * cache,
        int W,
        int H,
        int T,
        int new_cache_T,
        int C,
        int dst_t0,
        int keep_cur) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H * keep_cur * C;
    if (idx >= total) return;
    int r = idx;
    const int w = r % W; r /= W;
    const int h = r % H; r /= H;
    const int t = r % keep_cur; r /= keep_cur;
    const int c = r;
    const int src_t = T - keep_cur + t;
    cache[whdc_index_dyn(w, h, dst_t0 + t, c, W, H, new_cache_T)] =
        input[whdc_index_dyn(w, h, src_t, c, W, H, T)];
}

__global__ void update_temporal_cache_old_last_whdc_f32_kernel(
        float * cache,
        int W,
        int H,
        int old_cache_T,
        int new_cache_T,
        int C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H * C;
    if (idx >= total) return;
    int r = idx;
    const int w = r % W; r /= W;
    const int h = r % H; r /= H;
    const int c = r;
    cache[whdc_index_dyn(w, h, 0, c, W, H, new_cache_T)] =
        cache[whdc_index_dyn(w, h, old_cache_T - 1, c, W, H, old_cache_T)];
}

__global__ void libero_patchify_scaled_vcfhw_to_whdc_f32_kernel(
        const float * video_vcfhw,
        float * patch_whdc,
        int view,
        int views,
        int frames,
        int height,
        int width) {
    const int patch_W = width / 2;
    const int patch_H = height / 2;
    const int out_C = 12;
    const size_t total = static_cast<size_t>(patch_W) * patch_H * frames * out_C;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t r = idx;
    const int w = static_cast<int>(r % patch_W); r /= patch_W;
    const int h = static_cast<int>(r % patch_H); r /= patch_H;
    const int f = static_cast<int>(r % frames); r /= frames;
    const int out_c = static_cast<int>(r);

    const int rgb_c = out_c / 4;
    const int patch = out_c % 4;
    const int pw = patch / 2;
    const int ph = patch % 2;
    const int src_h = h * 2 + ph;
    const int src_w = w * 2 + pw;
    const size_t src_idx =
        (((static_cast<size_t>(view) * 3 + rgb_c) * frames + f) * height + src_h) * width + src_w;
    (void) views;
    patch_whdc[whdc_index_dyn(w, h, f, out_c, patch_W, patch_H, frames)] =
        video_vcfhw[src_idx];
}

__global__ void normalize_cat_views_whdc_to_bcfhw_f32_kernel(
        const float * enc96_views_whdc,
        const float * latents_mean,
        const float * latents_inv_std,
        float * out_bcfhw,
        int views,
        int latent_W_single,
        int latent_H,
        int latent_T,
        int z_dim) {
    const int out_W = latent_W_single * views;
    const size_t total = static_cast<size_t>(z_dim) * latent_T * latent_H * out_W;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t r = idx;
    const int w_cat = static_cast<int>(r % out_W); r /= out_W;
    const int h = static_cast<int>(r % latent_H); r /= latent_H;
    const int t = static_cast<int>(r % latent_T); r /= latent_T;
    const int c = static_cast<int>(r);
    const int view = w_cat / latent_W_single;
    const int w = w_cat - view * latent_W_single;
    const size_t view_elems = static_cast<size_t>(latent_W_single) * latent_H * latent_T * (2 * z_dim);
    const float * enc = enc96_views_whdc + static_cast<size_t>(view) * view_elems;
    const float mu = enc[whdc_index_dyn(w, h, t, c, latent_W_single, latent_H, latent_T)];
    out_bcfhw[(((static_cast<size_t>(c) * latent_T + t) * latent_H + h) * out_W + w_cat)] =
        (mu - latents_mean[c]) * latents_inv_std[c];
}

} // namespace

extern "C" int lingbot_vae_norm_silu_whdc_f32w(
    const float * input_whdc,
    const float * gamma_f32,
    float * output_whdc,
    int W,
    int H,
    int T,
    int C,
    cudaStream_t stream) {
    if (!input_whdc || !gamma_f32 || !output_whdc ||
        W <= 0 || H <= 0 || T <= 0 || C <= 0) return -1;
    constexpr int block = 128;
    const int sites = W * H * T;
    norm_silu_whdc_f32w_kernel<<<(sites + block - 1) / block, block, 0, stream>>>(
        input_whdc, gamma_f32, output_whdc, W, H, T, C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_vae_causal_conv3d_ks3_whdc_f32w(
    const float * input_whdc,
    const float * weight_f32,
    const float * bias_f32,
    float * output_whdc,
    int W,
    int H,
    int T,
    int in_C,
    int out_C,
    cudaStream_t stream) {
    if (!input_whdc || !weight_f32 || !bias_f32 || !output_whdc ||
        W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0) return -1;
#ifdef VLA_LINGBOT_USE_CUDNN
    if (lingbot_vae_cudnn_runtime_enabled() &&
        lingbot_vae_cached_causal_conv3d_ks3_whdc_cudnn(
            input_whdc, nullptr, weight_f32, bias_f32, output_whdc,
            W, H, T, 0, in_C, out_C, stream) == 0) {
        return 0;
    }
#endif
    constexpr int block = 128;
    const int total = W * H * T * out_C;
    causal_conv3d_ks3_whdc_f32w_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        input_whdc, weight_f32, bias_f32, output_whdc, W, H, T, in_C, out_C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_vae_cached_causal_conv3d_ks3_whdc_f32w(
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
    cudaStream_t stream) {
    if (!input_whdc || !weight_f32 || !bias_f32 || !output_whdc ||
        W <= 0 || H <= 0 || T <= 0 || cache_T < 0 || cache_T > 2 ||
        in_C <= 0 || out_C <= 0) {
        return -1;
    }
    if (cache_T > 0 && !cache_whdc) return -1;
#ifdef VLA_LINGBOT_USE_CUDNN
    if (lingbot_vae_cudnn_runtime_enabled() &&
        lingbot_vae_cached_causal_conv3d_ks3_whdc_cudnn(
            input_whdc, cache_whdc, weight_f32, bias_f32, output_whdc,
            W, H, T, cache_T, in_C, out_C, stream) == 0) {
        return 0;
    }
#endif
    constexpr int block = 128;
    const int total = W * H * T * out_C;
    cached_causal_conv3d_ks3_whdc_f32w_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        input_whdc, cache_whdc, weight_f32, bias_f32, output_whdc,
        W, H, T, cache_T, in_C, out_C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_vae_add_whdc_f32(
    const float * a,
    const float * b,
    float * out,
    size_t elems,
    cudaStream_t stream) {
    if (!a || !b || !out || elems == 0) return -1;
    constexpr int block = 256;
    add_whdc_kernel<<<static_cast<unsigned int>((elems + block - 1) / block),
                      block, 0, stream>>>(a, b, out, elems);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_vae_spatial_downsample2d_whdc_f32w(
    const float * input_whdc,
    const float * weight_f32,
    const float * bias_f32,
    float * output_whdc,
    int W,
    int H,
    int T,
    int C,
    cudaStream_t stream) {
    if (!input_whdc || !weight_f32 || !bias_f32 || !output_whdc ||
        W <= 0 || H <= 0 || T <= 0 || C <= 0 || (W % 2) != 0 || (H % 2) != 0) {
        return -1;
    }
#ifdef VLA_LINGBOT_USE_CUDNN
    if (lingbot_vae_cudnn_runtime_enabled() &&
        !lingbot_vae_env_disabled("VLA_LINGBOT_VAE_SPATIAL_CUDNN_DISABLE") &&
        lingbot_vae_spatial_downsample2d_whdc_cudnn(
            input_whdc, weight_f32, bias_f32, output_whdc,
            W, H, T, C, stream) == 0) {
        return 0;
    }
#endif
    constexpr int block = 128;
    const int total = (W / 2) * (H / 2) * T * C;
    spatial_downsample2d_whdc_f32w_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        input_whdc, weight_f32, bias_f32, output_whdc, W, H, T, C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_vae_conv1x1x1_whdc_f32w(
    const float * input_whdc,
    const float * weight_f32,
    const float * bias_f32,
    float * output_whdc,
    int W,
    int H,
    int T,
    int in_C,
    int out_C,
    cudaStream_t stream) {
    if (!input_whdc || !weight_f32 || !bias_f32 || !output_whdc ||
        W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0) return -1;
    if (lingbot_vae_cublas_runtime_enabled() &&
        lingbot_vae_conv1x1x1_whdc_cublas(input_whdc, weight_f32, bias_f32, output_whdc,
                                          W, H, T, in_C, out_C, stream) == 0) {
        return 0;
    }
    constexpr int block = 128;
    const int total = W * H * T * out_C;
    conv1x1x1_whdc_f32w_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        input_whdc, weight_f32, bias_f32, output_whdc, W, H, T, in_C, out_C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_vae_avg_down3d_whdc_f32(
    const float * input_whdc,
    float * output_whdc,
    int W,
    int H,
    int T,
    int in_C,
    int out_C,
    int factor_t,
    int factor_s,
    cudaStream_t stream) {
    if (!input_whdc || !output_whdc || W <= 0 || H <= 0 || T <= 0 ||
        in_C <= 0 || out_C <= 0 || factor_t <= 0 || factor_s <= 0 ||
        (W % factor_s) != 0 || (H % factor_s) != 0) {
        return -1;
    }
    const int factor = factor_t * factor_s * factor_s;
    if ((in_C * factor) % out_C != 0) return -1;
    const int To = (T + ((factor_t - (T % factor_t)) % factor_t)) / factor_t;
    const int total = (W / factor_s) * (H / factor_s) * To * out_C;
    constexpr int block = 256;
    avg_down3d_whdc_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        input_whdc, output_whdc, W, H, T, in_C, out_C, factor_t, factor_s);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_vae_downsample3d_time_stream_one_whdc_f32w(
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
    cudaStream_t stream) {
    if (!spatial_whdc || !cache_whdc || !weight_f32 || !bias_f32 || !output_whdc ||
        W <= 0 || H <= 0 || T <= 0 || cache_T <= 0 || in_C <= 0 || out_C <= 0) {
        return -1;
    }
    const int To = (T + 1) >= 3 ? ((T + 1 - 3) / 2 + 1) : 0;
    if (To <= 0) return -1;
    constexpr int block = 128;
    const int total = W * H * To * out_C;
    downsample3d_time_stream_one_whdc_f32w_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        spatial_whdc, cache_whdc, weight_f32, bias_f32, output_whdc,
        W, H, T, cache_T, in_C, out_C, To);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_vae_update_temporal_cache_last_whdc_f32(
    const float * input_whdc,
    float * cache_whdc,
    int W,
    int H,
    int T,
    int old_cache_T,
    int new_cache_T,
    int C,
    cudaStream_t stream) {
    if (!input_whdc || !cache_whdc || W <= 0 || H <= 0 || T <= 0 ||
        old_cache_T < 0 || old_cache_T > 2 ||
        new_cache_T <= 0 || new_cache_T > 2 || C <= 0) {
        return -1;
    }
    const bool one_frame_cache = new_cache_T == 1 && old_cache_T == 0;
    if (!one_frame_cache) {
        if (T < 2 && old_cache_T <= 0 && new_cache_T != T) return -1;
        if (T >= 2 && new_cache_T != 2) return -1;
    }
    constexpr int block = 256;
    const bool prepend_prev = T < 2 && old_cache_T > 0;
    const int dst_t0 = prepend_prev ? 1 : 0;
    const int keep_cur = new_cache_T - dst_t0;
    if (prepend_prev) {
        const int old_total = W * H * C;
        update_temporal_cache_old_last_whdc_f32_kernel<<<(old_total + block - 1) / block,
                                                         block, 0, stream>>>(
            cache_whdc, W, H, old_cache_T, new_cache_T, C);
        if (cudaGetLastError() != cudaSuccess) return -1;
    }
    if (keep_cur > 0) {
        const int cur_total = W * H * keep_cur * C;
        update_temporal_cache_current_whdc_f32_kernel<<<(cur_total + block - 1) / block,
                                                        block, 0, stream>>>(
            input_whdc, cache_whdc, W, H, T, new_cache_T, C, dst_t0, keep_cur);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_vae_libero_scale_patchify_vcfhw_to_whdc_f32(
    const float * video_vcfhw,
    float * patch_whdc,
    int view,
    int views,
    int frames,
    int height,
    int width,
    cudaStream_t stream) {
    if (!video_vcfhw || !patch_whdc || view < 0 || view >= views || views <= 0 ||
        frames <= 0 || height <= 0 || width <= 0 ||
        (height % 2) != 0 || (width % 2) != 0) {
        return -1;
    }
    const size_t total = static_cast<size_t>(width / 2) * (height / 2) * frames * 12;
    constexpr int block = 256;
    libero_patchify_scaled_vcfhw_to_whdc_f32_kernel<<<static_cast<unsigned int>((total + block - 1) / block),
                                                      block, 0, stream>>>(
        video_vcfhw, patch_whdc, view, views, frames, height, width);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int lingbot_vae_normalize_cat_views_whdc_to_bcfhw_f32(
    const float * enc96_views_whdc,
    const float * latents_mean,
    const float * latents_inv_std,
    float * out_bcfhw,
    int views,
    int latent_W_single,
    int latent_H,
    int latent_T,
    int z_dim,
    cudaStream_t stream) {
    if (!enc96_views_whdc || !latents_mean || !latents_inv_std || !out_bcfhw ||
        views <= 0 || latent_W_single <= 0 || latent_H <= 0 || latent_T <= 0 || z_dim <= 0) {
        return -1;
    }
    const size_t total = static_cast<size_t>(z_dim) * latent_T * latent_H * latent_W_single * views;
    constexpr int block = 256;
    normalize_cat_views_whdc_to_bcfhw_f32_kernel<<<static_cast<unsigned int>((total + block - 1) / block),
                                                   block, 0, stream>>>(
        enc96_views_whdc, latents_mean, latents_inv_std, out_bcfhw,
        views, latent_W_single, latent_H, latent_T, z_dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" const char * lingbot_vae_kernel_status(void) {
    return "lingbot_vae_cuda: ok";
}
