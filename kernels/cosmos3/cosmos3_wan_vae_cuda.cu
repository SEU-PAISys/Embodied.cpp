// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#include "cosmos3_wan_vae_cuda.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#ifdef VLA_COSMOS3_USE_CUDNN
#include "/usr/local/cuda-11.6/include/cudnn.h"
#endif
#include <cmath>
#include <unordered_map>

namespace {

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

__device__ __forceinline__ float robolab_image_sample_norm(
        const unsigned char * image,
        int image_h,
        int image_w,
        int channel,
        int padded_y,
        int padded_x) {
    constexpr int padded_h = 544;
    constexpr int padded_w = 736;
    // Official ActionTransformPipeline keeps the original frame at top-left
    // and reflection-pads only the right/bottom edges for 540x640 RoboLab.
    const int content_h = min(image_h, padded_h);
    const int content_w = min(image_w, padded_w);

    const int content_y = reflect_to_content(padded_y, 0, content_h);
    const int content_x = reflect_to_content(padded_x, 0, content_w);
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

__device__ __forceinline__ size_t whdc_index(
        int w,
        int h,
        int t,
        int c) {
    constexpr int W = COSMOS3_WAN_VAE_PATCH_W;
    constexpr int H = COSMOS3_WAN_VAE_PATCH_H;
    constexpr int T = COSMOS3_WAN_VAE_FRAMES;
    return static_cast<size_t>(w) +
           static_cast<size_t>(W) *
               (static_cast<size_t>(h) +
                static_cast<size_t>(H) *
                    (static_cast<size_t>(t) +
                     static_cast<size_t>(T) * static_cast<size_t>(c)));
}

__device__ __forceinline__ size_t conv1_whdc_index(
        int w,
        int h,
        int t,
        int c) {
    constexpr int W = COSMOS3_WAN_VAE_PATCH_W;
    constexpr int H = COSMOS3_WAN_VAE_PATCH_H;
    constexpr int T = COSMOS3_WAN_VAE_FRAMES;
    return static_cast<size_t>(w) +
           static_cast<size_t>(W) *
               (static_cast<size_t>(h) +
                static_cast<size_t>(H) *
                    (static_cast<size_t>(t) +
                     static_cast<size_t>(T) * static_cast<size_t>(c)));
}

__device__ __forceinline__ size_t down0_whdc_index(
        int w,
        int h,
        int t,
        int c) {
    constexpr int W = COSMOS3_WAN_VAE_DOWN0_W;
    constexpr int H = COSMOS3_WAN_VAE_DOWN0_H;
    constexpr int T = COSMOS3_WAN_VAE_DOWN0_T;
    return static_cast<size_t>(w) +
           static_cast<size_t>(W) *
               (static_cast<size_t>(h) +
                static_cast<size_t>(H) *
                    (static_cast<size_t>(t) +
                     static_cast<size_t>(T) * static_cast<size_t>(c)));
}

__device__ __forceinline__ float bf16_to_float(const unsigned short v) {
    __nv_bfloat16 b;
    *reinterpret_cast<unsigned short *>(&b) = v;
    return __bfloat162float(b);
}

__global__ void bf16_to_f32_kernel(
        const unsigned short * src,
        float * dst,
        size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = bf16_to_float(src[idx]);
}

__global__ void causal_pad_time_whdc_kernel(
        const float * input,
        float * padded,
        int W,
        int H,
        int T,
        int C) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(W) * H * (T + 2) * C;
    if (idx >= total) return;
    int r = static_cast<int>(idx);
    const int w = r % W; r /= W;
    const int h = r % H; r /= H;
    const int tp = r % (T + 2); r /= (T + 2);
    const int c = r;
    if (tp < 2) {
        padded[idx] = 0.0f;
    } else {
        padded[idx] = input[static_cast<size_t>(w) +
                            static_cast<size_t>(W) *
                                (static_cast<size_t>(h) +
                                 static_cast<size_t>(H) *
                                     (static_cast<size_t>(tp - 2) +
                                      static_cast<size_t>(T) * static_cast<size_t>(c)))];
    }
}

__global__ void add_bias_whdc_kernel(
        float * output,
        const unsigned short * bias_bf16,
        int W,
        int H,
        int T,
        int C) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(W) * H * T * C;
    if (idx >= total) return;
    const int c = static_cast<int>(idx / (static_cast<size_t>(W) * H * T));
    output[idx] += bf16_to_float(bias_bf16[c]);
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

__global__ void robolab_image_to_vae_patch_whdc_kernel(
        const unsigned char * image,
        int image_h,
        int image_w,
        float * patch_whdc) {
    constexpr int W = COSMOS3_WAN_VAE_PATCH_W;
    constexpr int H = COSMOS3_WAN_VAE_PATCH_H;
    constexpr int T = COSMOS3_WAN_VAE_FRAMES;
    constexpr int C = COSMOS3_WAN_VAE_PATCH_CHANNELS;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H * T * C;
    if (idx >= total) return;

    int r = idx;
    const int w = r % W; r /= W;
    const int h = r % H; r /= H;
    const int t = r % T; r /= T;
    const int c_patch = r;

    const int channel = c_patch / 4;
    const int rem = c_patch - channel * 4;
    const int pw = rem / 2;
    const int ph = rem - pw * 2;
    const int y = h * 2 + ph;
    const int x = w * 2 + pw;

    patch_whdc[whdc_index(w, h, t, c_patch)] =
        (t == 0) ? robolab_image_sample_norm(image, image_h, image_w, channel, y, x) : -1.0f;
}

__global__ void patch_whdc_prefix_kernel(
        const float * patch_whdc,
        float * prefix,
        int rows) {
    constexpr int W = COSMOS3_WAN_VAE_PATCH_W;
    constexpr int H = COSMOS3_WAN_VAE_PATCH_H;
    constexpr int T = COSMOS3_WAN_VAE_FRAMES;
    constexpr int C = COSMOS3_WAN_VAE_PATCH_CHANNELS;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * C;
    if (idx >= total) return;
    const int c = idx % C;
    int row = idx / C;
    const int w = row % W; row /= W;
    const int h = row % H; row /= H;
    const int t = row % T;
    prefix[idx] = patch_whdc[whdc_index(w, h, t, c)];
}

__global__ void encoder_conv1_prefix_kernel(
        const float * patch_whdc,
        const unsigned short * weight_bf16,
        const unsigned short * bias_bf16,
        float * prefix,
        int rows,
        int cols) {
    constexpr int W = COSMOS3_WAN_VAE_PATCH_W;
    constexpr int H = COSMOS3_WAN_VAE_PATCH_H;
    constexpr int T = COSMOS3_WAN_VAE_FRAMES;
    constexpr int Cin = COSMOS3_WAN_VAE_PATCH_CHANNELS;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;

    const int co = idx % cols;
    int row = idx / cols;
    const int w = row % W; row /= W;
    const int h = row % H; row /= H;
    const int t = row % T;

    float sum = bf16_to_float(bias_bf16[co]);
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kt = 0; kt < 3; ++kt) {
            const int it = t + kt - 2; // F.pad temporal=(2,0), Conv3d kt=3.
            if (it < 0 || it >= T) continue;
            for (int kh = 0; kh < 3; ++kh) {
                const int ih = h + kh - 1;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < 3; ++kw) {
                    const int iw = w + kw - 1;
                    if (iw < 0 || iw >= W) continue;
                    const size_t wi =
                        (((static_cast<size_t>(co) * Cin + ci) * 3 + kt) * 3 + kh) * 3 + kw;
                    sum += patch_whdc[whdc_index(iw, ih, it, ci)] *
                           bf16_to_float(weight_bf16[wi]);
                }
            }
        }
    }
    prefix[idx] = sum;
}

__global__ void encoder_conv1_whdc_kernel(
        const float * patch_whdc,
        const unsigned short * weight_bf16,
        const unsigned short * bias_bf16,
        float * conv1_whdc) {
    constexpr int W = COSMOS3_WAN_VAE_PATCH_W;
    constexpr int H = COSMOS3_WAN_VAE_PATCH_H;
    constexpr int T = COSMOS3_WAN_VAE_FRAMES;
    constexpr int Cin = COSMOS3_WAN_VAE_PATCH_CHANNELS;
    constexpr int Cout = COSMOS3_WAN_VAE_CONV1_CHANNELS;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H * T * Cout;
    if (idx >= total) return;

    int r = idx;
    const int w = r % W; r /= W;
    const int h = r % H; r /= H;
    const int t = r % T; r /= T;
    const int co = r;

    float sum = bf16_to_float(bias_bf16[co]);
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kt = 0; kt < 3; ++kt) {
            const int it = t + kt - 2; // F.pad temporal=(2,0), Conv3d kt=3.
            if (it < 0 || it >= T) continue;
            for (int kh = 0; kh < 3; ++kh) {
                const int ih = h + kh - 1;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < 3; ++kw) {
                    const int iw = w + kw - 1;
                    if (iw < 0 || iw >= W) continue;
                    const size_t wi =
                        (((static_cast<size_t>(co) * Cin + ci) * 3 + kt) * 3 + kh) * 3 + kw;
                    sum += patch_whdc[whdc_index(iw, ih, it, ci)] *
                           bf16_to_float(weight_bf16[wi]);
                }
            }
        }
    }
    conv1_whdc[conv1_whdc_index(w, h, t, co)] = sum;
}

__global__ void encoder_conv1_whdc_prefix_kernel(
        const float * conv1_whdc,
        float * prefix,
        int rows,
        int cols) {
    constexpr int W = COSMOS3_WAN_VAE_PATCH_W;
    constexpr int H = COSMOS3_WAN_VAE_PATCH_H;
    constexpr int T = COSMOS3_WAN_VAE_FRAMES;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    const int c = idx % cols;
    int row = idx / cols;
    const int w = row % W; row /= W;
    const int h = row % H; row /= H;
    const int t = row % T;
    prefix[idx] = conv1_whdc[conv1_whdc_index(w, h, t, c)];
}

__global__ void down0_avg_shortcut_kernel(
        const float * conv1_whdc,
        float * down0_shortcut_whdc) {
    constexpr int Wo = COSMOS3_WAN_VAE_DOWN0_W;
    constexpr int Ho = COSMOS3_WAN_VAE_DOWN0_H;
    constexpr int T = COSMOS3_WAN_VAE_DOWN0_T;
    constexpr int Co = COSMOS3_WAN_VAE_DOWN0_CHANNELS;
    constexpr int factor_s = 2;
    constexpr int group_size = 4;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Wo * Ho * T * Co;
    if (idx >= total) return;

    int r = idx;
    const int w = r % Wo; r /= Wo;
    const int h = r % Ho; r /= Ho;
    const int t = r % T; r /= T;
    const int co = r;

    float acc = 0.0f;
    #pragma unroll
    for (int gidx = 0; gidx < group_size; ++gidx) {
        const int flat = co * group_size + gidx;
        const int ci = flat / (factor_s * factor_s);
        const int rem = flat - ci * factor_s * factor_s;
        const int fs_h = rem / factor_s;
        const int fs_w = rem - fs_h * factor_s;
        acc += conv1_whdc[conv1_whdc_index(w * factor_s + fs_w,
                                           h * factor_s + fs_h,
                                           t,
                                           ci)];
    }
    down0_shortcut_whdc[down0_whdc_index(w, h, t, co)] = acc * 0.25f;
}

__global__ void down0_whdc_prefix_kernel(
        const float * down0_whdc,
        float * prefix,
        int rows,
        int cols) {
    constexpr int W = COSMOS3_WAN_VAE_DOWN0_W;
    constexpr int H = COSMOS3_WAN_VAE_DOWN0_H;
    constexpr int T = COSMOS3_WAN_VAE_DOWN0_T;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    const int c = idx % cols;
    int row = idx / cols;
    const int w = row % W; row /= W;
    const int h = row % H; row /= H;
    const int t = row % T;
    prefix[idx] = down0_whdc[down0_whdc_index(w, h, t, c)];
}

__global__ void norm_silu_whdc_kernel(
        const float * input,
        const unsigned short * gamma_bf16,
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
                        inv * bf16_to_float(gamma_bf16[c]);
        output[whdc_index_dyn(w, h, t, c, W, H, T)] = x / (1.0f + expf(-x));
    }
}

__global__ void rms_norm_whdc_kernel(
        const float * input,
        const unsigned short * gamma_bf16,
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
        output[whdc_index_dyn(w, h, t, c, W, H, T)] =
            input[whdc_index_dyn(w, h, t, c, W, H, T)] *
            inv * bf16_to_float(gamma_bf16[c]);
    }
}

__global__ void causal_conv3d_ks3_whdc_kernel(
        const float * input,
        const unsigned short * weight_bf16,
        const unsigned short * bias_bf16,
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

    float sum = bf16_to_float(bias_bf16[co]);
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
                    sum += input[whdc_index_dyn(iw, ih, it, ci, W, H, T)] *
                           bf16_to_float(weight_bf16[wi]);
                }
            }
        }
    }
    output[whdc_index_dyn(w, h, t, co, W, H, T)] = sum;
}

static cublasHandle_t g_wan_vae_cublas_handle = nullptr;

#ifdef VLA_COSMOS3_USE_CUDNN
struct CudnnCausalConv3dCache {
    cudnnHandle_t handle = nullptr;
    float * padded = nullptr;
    size_t padded_elems = 0;
    void * workspace = nullptr;
    size_t workspace_bytes = 0;
    struct WeightCacheEntry {
        float * data = nullptr;
        size_t elems = 0;
    };
    std::unordered_map<const unsigned short *, WeightCacheEntry> weights_f32;
};

static CudnnCausalConv3dCache g_cudnn_conv3d_cache;

static bool ensure_cudnn_buffer(float ** ptr, size_t * capacity, size_t elems) {
    if (*capacity >= elems && *ptr) return true;
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

static bool ensure_cudnn_workspace(void ** ptr, size_t * capacity, size_t bytes) {
    if (*capacity >= bytes && *ptr) return true;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *capacity = 0;
    if (bytes == 0) return true;
    if (cudaMalloc(ptr, bytes) != cudaSuccess) return false;
    *capacity = bytes;
    return true;
}

static const float * ensure_cudnn_weight_f32(
        CudnnCausalConv3dCache & cache,
        const unsigned short * weight_bf16,
        size_t elems,
        cudaStream_t stream) {
    auto it = cache.weights_f32.find(weight_bf16);
    if (it != cache.weights_f32.end() && it->second.data && it->second.elems == elems) {
        return it->second.data;
    }
    if (it != cache.weights_f32.end()) {
        if (it->second.data) cudaFree(it->second.data);
        cache.weights_f32.erase(it);
    }
    CudnnCausalConv3dCache::WeightCacheEntry entry;
    if (cudaMalloc(reinterpret_cast<void **>(&entry.data), elems * sizeof(float)) != cudaSuccess) {
        return nullptr;
    }
    entry.elems = elems;
    constexpr int block = 256;
    bf16_to_f32_kernel<<<static_cast<unsigned int>((elems + block - 1) / block),
                         block, 0, stream>>>(weight_bf16, entry.data, elems);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(entry.data);
        return nullptr;
    }
    float * out = entry.data;
    cache.weights_f32.emplace(weight_bf16, entry);
    return out;
}

static int causal_conv3d_ks3_whdc_cudnn(
        const float * input,
        const unsigned short * weight_bf16,
        const unsigned short * bias_bf16,
        float * output,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        cudaStream_t stream) {
    auto & cache = g_cudnn_conv3d_cache;
    if (!cache.handle) {
        if (cudnnCreate(&cache.handle) != CUDNN_STATUS_SUCCESS) return -1;
    }
    if (cudnnSetStream(cache.handle, stream) != CUDNN_STATUS_SUCCESS) return -1;

    const size_t padded_elems = static_cast<size_t>(W) * H * (T + 2) * in_C;
    const size_t weight_elems = static_cast<size_t>(out_C) * in_C * 3 * 3 * 3;
    if (!ensure_cudnn_buffer(&cache.padded, &cache.padded_elems, padded_elems)) {
        return -1;
    }

    constexpr int block = 256;
    causal_pad_time_whdc_kernel<<<static_cast<unsigned int>((padded_elems + block - 1) / block),
                                  block, 0, stream>>>(input, cache.padded, W, H, T, in_C);
    if (cudaGetLastError() != cudaSuccess) return -1;
    const float * weight_f32 = ensure_cudnn_weight_f32(cache, weight_bf16, weight_elems, stream);
    if (!weight_f32) return -1;

    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    if (cudnnCreateTensorDescriptor(&x_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateTensorDescriptor(&y_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateFilterDescriptor(&w_desc) != CUDNN_STATUS_SUCCESS ||
        cudnnCreateConvolutionDescriptor(&conv_desc) != CUDNN_STATUS_SUCCESS) {
        return -1;
    }

    const int x_dim[5] = {1, in_C, T + 2, H, W};
    const int x_stride[5] = {in_C * (T + 2) * H * W,
                             (T + 2) * H * W,
                             H * W,
                             W,
                             1};
    const int y_dim[5] = {1, out_C, T, H, W};
    const int y_stride[5] = {out_C * T * H * W,
                             T * H * W,
                             H * W,
                             W,
                             1};
    const int filt_dim[5] = {out_C, in_C, 3, 3, 3};
    const int pad[3] = {0, 1, 1};
    const int stride[3] = {1, 1, 1};
    const int dilation[3] = {1, 1, 1};

    int ok = 0;
    if (cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 5, x_dim, x_stride) != CUDNN_STATUS_SUCCESS ||
        cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 5, y_dim, y_stride) != CUDNN_STATUS_SUCCESS ||
        cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filt_dim) != CUDNN_STATUS_SUCCESS ||
        cudnnSetConvolutionNdDescriptor(conv_desc, 3, pad, stride, dilation,
                                        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) != CUDNN_STATUS_SUCCESS) {
        ok = -1;
    }

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t workspace_bytes = 0;
    if (ok == 0 &&
        cudnnGetConvolutionForwardWorkspaceSize(cache.handle, x_desc, w_desc, conv_desc, y_desc,
                                                algo, &workspace_bytes) != CUDNN_STATUS_SUCCESS) {
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        workspace_bytes = 0;
    }
    if (ok == 0 && !ensure_cudnn_workspace(&cache.workspace, &cache.workspace_bytes, workspace_bytes)) {
        ok = -1;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (ok == 0 &&
        cudnnConvolutionForward(cache.handle,
                                &alpha,
                                x_desc,
                                cache.padded,
                                w_desc,
                                weight_f32,
                                conv_desc,
                                algo,
                                cache.workspace,
                                workspace_bytes,
                                &beta,
                                y_desc,
                                output) != CUDNN_STATUS_SUCCESS) {
        ok = -1;
    }
    if (ok == 0) {
        const size_t total = static_cast<size_t>(W) * H * T * out_C;
        add_bias_whdc_kernel<<<static_cast<unsigned int>((total + block - 1) / block),
                               block, 0, stream>>>(output, bias_bf16, W, H, T, out_C);
        if (cudaGetLastError() != cudaSuccess) ok = -1;
    }

    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(x_desc);
    return ok;
}

static int conv1x1x1_whdc_cublas(
        const float * input,
        const unsigned short * weight_bf16,
        const unsigned short * bias_bf16,
        float * output,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        cudaStream_t stream) {
    auto & cache = g_cudnn_conv3d_cache;
    const size_t weight_elems = static_cast<size_t>(out_C) * in_C;
    const float * weight_f32 = ensure_cudnn_weight_f32(cache, weight_bf16, weight_elems, stream);
    if (!weight_f32) return -1;
    if (!g_wan_vae_cublas_handle &&
        cublasCreate(&g_wan_vae_cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    if (cublasSetStream(g_wan_vae_cublas_handle, stream) != CUBLAS_STATUS_SUCCESS) return -1;
    const int sites = W * H * T;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t st = cublasSgemm(
        g_wan_vae_cublas_handle,
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
    add_bias_whdc_kernel<<<static_cast<unsigned int>((total + block - 1) / block),
                           block, 0, stream>>>(output, bias_bf16, W, H, T, out_C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
#endif

__global__ void add_whdc_kernel(
        const float * a,
        const float * b,
        float * out,
        size_t elems) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < elems) out[idx] = a[idx] + b[idx];
}

__global__ void spatial_downsample2d_whdc_kernel(
        const float * input,
        const unsigned short * weight_bf16,
        const unsigned short * bias_bf16,
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

    float sum = bf16_to_float(bias_bf16[co]);
    for (int ci = 0; ci < C; ++ci) {
        for (int kh = 0; kh < 3; ++kh) {
            const int ih = oh * 2 + kh;
            if (ih >= H + 1) continue;
            for (int kw = 0; kw < 3; ++kw) {
                const int iw = ow * 2 + kw;
                if (iw >= W + 1) continue;
                if (iw >= W || ih >= H) continue; // right/bottom zero pad
                const size_t wi = (((static_cast<size_t>(co) * C + ci) * 3 + kh) * 3 + kw);
                sum += input[whdc_index_dyn(iw, ih, t, ci, W, H, T)] *
                       bf16_to_float(weight_bf16[wi]);
            }
        }
    }
    output[whdc_index_dyn(ow, oh, t, co, Wo, Ho, T)] = sum;
}

__global__ void conv1x1x1_whdc_kernel(
        const float * input,
        const unsigned short * weight_bf16,
        const unsigned short * bias_bf16,
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
    float sum = bf16_to_float(bias_bf16[co]);
    for (int ci = 0; ci < in_C; ++ci) {
        sum += input[whdc_index_dyn(w, h, t, ci, W, H, T)] *
               bf16_to_float(weight_bf16[static_cast<size_t>(co) * in_C + ci]);
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

__global__ void downsample3d_time_whdc_kernel(
        const float * spatial,
        const unsigned short * weight_bf16,
        const unsigned short * bias_bf16,
        float * output,
        int W,
        int H,
        int T_total,
        int C,
        int total_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = W * H * total_out * C;
    if (idx >= total) return;
    int r = idx;
    const int w = r % W; r /= W;
    const int h = r % H; r /= H;
    const int to_global = r % total_out; r /= total_out;
    const int co = r;

    // Official streaming semantics: the first one-frame chunk is a passthrough.
    // Each later output j uses the previous cached frame plus the current pair,
    // which is equivalent to convolving input frames [2j-2, 2j-1, 2j].
    if (to_global == 0) {
        output[whdc_index_dyn(w, h, 0, co, W, H, total_out)] =
            spatial[whdc_index_dyn(w, h, 0, co, W, H, T_total)];
        return;
    }
    float sum = bf16_to_float(bias_bf16[co]);
    for (int k = 0; k < 3; ++k) {
        const int in_t = 2 * to_global - 2 + k;
        for (int ci = 0; ci < C; ++ci) {
            const size_t wi = (static_cast<size_t>(co) * C + ci) * 3 + k;
            sum += spatial[whdc_index_dyn(w, h, in_t, ci, W, H, T_total)] *
                   bf16_to_float(weight_bf16[wi]);
        }
    }
    output[whdc_index_dyn(w, h, to_global, co, W, H, total_out)] = sum;
}

__global__ void generic_whdc_prefix_kernel(
        const float * input,
        float * prefix,
        int W,
        int H,
        int T,
        int C,
        int rows,
        int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;
    const int c = idx % cols;
    int row = idx / cols;
    const int w = row % W; row /= W;
    const int h = row % H; row /= H;
    const int t = row % T;
    prefix[idx] = input[whdc_index_dyn(w, h, t, c, W, H, T)];
}

__global__ void mid_qkv_whdc_to_row_kernel(
        const float * qkv_whdc,
        float * q_row,
        float * k_row,
        float * v_row,
        int W,
        int H,
        int T,
        int C) {
    const int N = W * H;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = T * N * C;
    if (idx >= total) return;
    int r = idx;
    const int c = r % C; r /= C;
    const int token = r % N; r /= N;
    const int t = r;
    const int w = token % W;
    const int h = token / W;
    const size_t row_idx = (static_cast<size_t>(t) * N + token) * C + c;
    q_row[row_idx] = qkv_whdc[whdc_index_dyn(w, h, t, c, W, H, T)];
    k_row[row_idx] = qkv_whdc[whdc_index_dyn(w, h, t, C + c, W, H, T)];
    v_row[row_idx] = qkv_whdc[whdc_index_dyn(w, h, t, 2 * C + c, W, H, T)];
}

__global__ void softmax_rows_kernel(float * scores, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float * p = scores + static_cast<size_t>(row) * cols;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        m = fmaxf(m, p[i]);
    }
    __shared__ float shared[256];
    shared[threadIdx.x] = m;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        __syncthreads();
    }
    m = shared[0];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float e = expf(p[i] - m);
        p[i] = e;
        sum += e;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = 1.0f / shared[0];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        p[i] *= inv;
    }
}

__global__ void mid_attn_row_to_whdc_kernel(
        const float * row,
        float * whdc,
        int W,
        int H,
        int T,
        int C) {
    const int N = W * H;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = T * N * C;
    if (idx >= total) return;
    int r = idx;
    const int c = r % C; r /= C;
    const int token = r % N; r /= N;
    const int t = r;
    const int w = token % W;
    const int h = token / W;
    whdc[whdc_index_dyn(w, h, t, c, W, H, T)] =
        row[(static_cast<size_t>(t) * N + token) * C + c];
}

__global__ void pack_clean_vision_condition_kernel(
        const float * final_conv1_whdc,
        const unsigned short * scale_mean_bf16,
        const unsigned short * scale_inv_std_bf16,
        float * clean_condition) {
    constexpr int latent_W = COSMOS3_WAN_VAE_DOWN2_W;
    constexpr int latent_H_full = COSMOS3_WAN_VAE_DOWN2_H;
    constexpr int latent_H = 33;
    constexpr int latent_W_packed = 40;
    constexpr int latent_T = COSMOS3_WAN_VAE_DOWN2_T;
    constexpr int z_dim = 48;
    constexpr int patch_dim = 192;
    constexpr int rows = 17 * 20;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * patch_dim;
    if (idx >= total) return;

    const int col = idx % patch_dim;
    int row = idx / patch_dim;
    const int token_w = row % 20;
    row /= 20;
    const int token_h = row % 17;
    const int patch_offset = col / z_dim;
    const int c = col - patch_offset * z_dim;
    const int ph = patch_offset / 2;
    const int pw = patch_offset - ph * 2;
    const int w = token_w * 2 + pw;
    const int h = token_h * 2 + ph;
    float value = 0.0f;
    if (c < z_dim && h < latent_H && w < latent_W_packed && w < latent_W) {
        const size_t src = whdc_index_dyn(w, h, 0, c, latent_W, latent_H_full, latent_T);
        value = (final_conv1_whdc[src] - bf16_to_float(scale_mean_bf16[c])) *
                bf16_to_float(scale_inv_std_bf16[c]);
    }
    clean_condition[idx] = value;
}

} // namespace

extern "C" int cosmos3_wan_vae_robolab_image_to_patch_whdc_f32(
    const unsigned char * image_u8,
    int image_h,
    int image_w,
    float * patch_whdc,
    cudaStream_t stream) {
    if (!image_u8 || !patch_whdc || image_h <= 0 || image_w <= 0) return -1;
    constexpr int block = 256;
    constexpr int total = COSMOS3_WAN_VAE_PATCH_W *
                          COSMOS3_WAN_VAE_PATCH_H *
                          COSMOS3_WAN_VAE_FRAMES *
                          COSMOS3_WAN_VAE_PATCH_CHANNELS;
    robolab_image_to_vae_patch_whdc_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        image_u8, image_h, image_w, patch_whdc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_patch_whdc_prefix_f32(
    const float * patch_whdc,
    float * prefix,
    int rows,
    cudaStream_t stream) {
    constexpr int max_rows = COSMOS3_WAN_VAE_PATCH_W *
                             COSMOS3_WAN_VAE_PATCH_H *
                             COSMOS3_WAN_VAE_FRAMES;
    if (!patch_whdc || !prefix || rows <= 0 || rows > max_rows) return -1;
    constexpr int block = 256;
    const int total = rows * COSMOS3_WAN_VAE_PATCH_CHANNELS;
    patch_whdc_prefix_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        patch_whdc, prefix, rows);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_encoder_conv1_whdc_f32(
    const float * patch_whdc,
    const unsigned short * weight_bf16,
    const unsigned short * bias_bf16,
    float * conv1_whdc,
    cudaStream_t stream) {
    if (!patch_whdc || !weight_bf16 || !bias_bf16 || !conv1_whdc) return -1;
    constexpr int block = 256;
    constexpr int total = COSMOS3_WAN_VAE_PATCH_W *
                          COSMOS3_WAN_VAE_PATCH_H *
                          COSMOS3_WAN_VAE_FRAMES *
                          COSMOS3_WAN_VAE_CONV1_CHANNELS;
    encoder_conv1_whdc_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        patch_whdc, weight_bf16, bias_bf16, conv1_whdc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_encoder_conv1_whdc_prefix_f32(
    const float * conv1_whdc,
    float * prefix,
    int rows,
    int cols,
    cudaStream_t stream) {
    constexpr int max_rows = COSMOS3_WAN_VAE_PATCH_W *
                             COSMOS3_WAN_VAE_PATCH_H *
                             COSMOS3_WAN_VAE_FRAMES;
    if (!conv1_whdc || !prefix || rows <= 0 || rows > max_rows ||
        cols <= 0 || cols > COSMOS3_WAN_VAE_CONV1_CHANNELS) {
        return -1;
    }
    constexpr int block = 256;
    const int total = rows * cols;
    encoder_conv1_whdc_prefix_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        conv1_whdc, prefix, rows, cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_down0_avg_shortcut_whdc_f32(
    const float * conv1_whdc,
    float * down0_shortcut_whdc,
    cudaStream_t stream) {
    if (!conv1_whdc || !down0_shortcut_whdc) return -1;
    constexpr int block = 256;
    constexpr int total = COSMOS3_WAN_VAE_DOWN0_W *
                          COSMOS3_WAN_VAE_DOWN0_H *
                          COSMOS3_WAN_VAE_DOWN0_T *
                          COSMOS3_WAN_VAE_DOWN0_CHANNELS;
    down0_avg_shortcut_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        conv1_whdc, down0_shortcut_whdc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_down0_whdc_prefix_f32(
    const float * down0_whdc,
    float * prefix,
    int rows,
    int cols,
    cudaStream_t stream) {
    constexpr int max_rows = COSMOS3_WAN_VAE_DOWN0_W *
                             COSMOS3_WAN_VAE_DOWN0_H *
                             COSMOS3_WAN_VAE_DOWN0_T;
    if (!down0_whdc || !prefix || rows <= 0 || rows > max_rows ||
        cols <= 0 || cols > COSMOS3_WAN_VAE_DOWN0_CHANNELS) {
        return -1;
    }
    constexpr int block = 256;
    const int total = rows * cols;
    down0_whdc_prefix_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        down0_whdc, prefix, rows, cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_norm_silu_whdc_f32(
    const float * input_whdc,
    const unsigned short * gamma_bf16,
    float * output_whdc,
    int W,
    int H,
    int T,
    int C,
    cudaStream_t stream) {
    if (!input_whdc || !gamma_bf16 || !output_whdc ||
        W <= 0 || H <= 0 || T <= 0 || C <= 0) return -1;
    constexpr int block = 128;
    const int sites = W * H * T;
    norm_silu_whdc_kernel<<<(sites + block - 1) / block, block, 0, stream>>>(
        input_whdc, gamma_bf16, output_whdc, W, H, T, C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(
    const float * input_whdc,
    const unsigned short * weight_bf16,
    const unsigned short * bias_bf16,
    float * output_whdc,
    int W,
    int H,
    int T,
    int in_C,
    int out_C,
    cudaStream_t stream) {
    if (!input_whdc || !weight_bf16 || !bias_bf16 || !output_whdc ||
        W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0) return -1;
#ifdef VLA_COSMOS3_USE_CUDNN
    if (causal_conv3d_ks3_whdc_cudnn(input_whdc,
                                      weight_bf16,
                                      bias_bf16,
                                      output_whdc,
                                      W,
                                      H,
                                      T,
                                      in_C,
                                      out_C,
                                      stream) == 0) {
        return 0;
    }
#endif
    constexpr int block = 128;
    const int total = W * H * T * out_C;
    causal_conv3d_ks3_whdc_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        input_whdc, weight_bf16, bias_bf16, output_whdc, W, H, T, in_C, out_C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_rms_norm_whdc_f32(
    const float * input_whdc,
    const unsigned short * gamma_bf16,
    float * output_whdc,
    int W,
    int H,
    int T,
    int C,
    cudaStream_t stream) {
    if (!input_whdc || !gamma_bf16 || !output_whdc ||
        W <= 0 || H <= 0 || T <= 0 || C <= 0) return -1;
    constexpr int block = 128;
    const int sites = W * H * T;
    rms_norm_whdc_kernel<<<(sites + block - 1) / block, block, 0, stream>>>(
        input_whdc, gamma_bf16, output_whdc, W, H, T, C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_add_whdc_f32(
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

extern "C" int cosmos3_wan_vae_spatial_downsample2d_whdc_f32(
    const float * input_whdc,
    const unsigned short * weight_bf16,
    const unsigned short * bias_bf16,
    float * output_whdc,
    int W,
    int H,
    int T,
    int C,
    cudaStream_t stream) {
    if (!input_whdc || !weight_bf16 || !bias_bf16 || !output_whdc ||
        W <= 0 || H <= 0 || T <= 0 || C <= 0 || (W % 2) != 0 || (H % 2) != 0) {
        return -1;
    }
    constexpr int block = 128;
    const int total = (W / 2) * (H / 2) * T * C;
    spatial_downsample2d_whdc_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        input_whdc, weight_bf16, bias_bf16, output_whdc, W, H, T, C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_conv1x1x1_whdc_f32(
    const float * input_whdc,
    const unsigned short * weight_bf16,
    const unsigned short * bias_bf16,
    float * output_whdc,
    int W,
    int H,
    int T,
    int in_C,
    int out_C,
    cudaStream_t stream) {
    if (!input_whdc || !weight_bf16 || !bias_bf16 || !output_whdc ||
        W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0) return -1;
#ifdef VLA_COSMOS3_USE_CUDNN
    if (conv1x1x1_whdc_cublas(input_whdc,
                              weight_bf16,
                              bias_bf16,
                              output_whdc,
                              W,
                              H,
                              T,
                              in_C,
                              out_C,
                              stream) == 0) {
        return 0;
    }
#endif
    constexpr int block = 128;
    const int total = W * H * T * out_C;
    conv1x1x1_whdc_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        input_whdc, weight_bf16, bias_bf16, output_whdc, W, H, T, in_C, out_C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_avg_down3d_whdc_f32(
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

extern "C" int cosmos3_wan_vae_downsample3d_time_whdc_f32(
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
    cudaStream_t stream) {
    if (!spatial_whdc || !weight_bf16 || !bias_bf16 || !output_whdc ||
        W <= 0 || H <= 0 || T_total <= 0 || C <= 0 ||
        !chunks_host || n_chunks <= 0 || chunks_host[0] != 1) {
        return -1;
    }
    int chunk_sum = 0;
    for (int i = 0; i < n_chunks; ++i) {
        if (chunks_host[i] <= 0) return -1;
        chunk_sum += chunks_host[i];
    }
    if (chunk_sum != T_total) return -1;
    const int total_out = (T_total + 1) / 2;
    constexpr int block = 128;
    const int total = W * H * total_out * C;
    downsample3d_time_whdc_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        spatial_whdc, weight_bf16, bias_bf16, output_whdc, W, H, T_total, C, total_out);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_generic_whdc_prefix_f32(
    const float * input_whdc,
    float * prefix,
    int W,
    int H,
    int T,
    int C,
    int rows,
    int cols,
    cudaStream_t stream) {
    if (!input_whdc || !prefix || W <= 0 || H <= 0 || T <= 0 || C <= 0 ||
        rows <= 0 || rows > W * H * T || cols <= 0 || cols > C) {
        return -1;
    }
    constexpr int block = 256;
    const int total = rows * cols;
    generic_whdc_prefix_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        input_whdc, prefix, W, H, T, C, rows, cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_mid_attention_f32(
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
    cudaStream_t stream) {
    if (!qkv_whdc || !scores || !q_row || !k_row || !v_row || !attn_row || !attn_whdc ||
        W <= 0 || H <= 0 || T <= 0 || C <= 0) return -1;
    const int N = W * H;
    constexpr int block = 256;
    const int qkv_total = T * N * C;
    mid_qkv_whdc_to_row_kernel<<<(qkv_total + block - 1) / block, block, 0, stream>>>(
        qkv_whdc, q_row, k_row, v_row, W, H, T, C);
    if (cudaGetLastError() != cudaSuccess) return -1;

    if (!g_wan_vae_cublas_handle &&
        cublasCreate(&g_wan_vae_cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    cublasHandle_t handle = g_wan_vae_cublas_handle;
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    const float alpha = 1.0f / sqrtf(static_cast<float>(C));
    const float beta = 0.0f;
    const long long stride_qkv = static_cast<long long>(N) * C;
    const long long stride_scores = static_cast<long long>(N) * N;
    cublasStatus_t st = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        N,
        C,
        &alpha,
        k_row,
        C,
        stride_qkv,
        q_row,
        C,
        stride_qkv,
        &beta,
        scores,
        N,
        stride_scores,
        T);
    if (st != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    softmax_rows_kernel<<<T * N, 256, 0, stream>>>(scores, T * N, N);
    if (cudaGetLastError() != cudaSuccess) {
        return -1;
    }
    const float alpha_pv = 1.0f;
    st = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        C,
        N,
        N,
        &alpha_pv,
        v_row,
        C,
        stride_qkv,
        scores,
        N,
        stride_scores,
        &beta,
        attn_row,
        C,
        stride_qkv,
        T);
    if (st != CUBLAS_STATUS_SUCCESS) return -1;
    mid_attn_row_to_whdc_kernel<<<(qkv_total + block - 1) / block, block, 0, stream>>>(
        attn_row, attn_whdc, W, H, T, C);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_pack_clean_vision_condition_f32(
    const float * final_conv1_whdc,
    const unsigned short * scale_mean_bf16,
    const unsigned short * scale_inv_std_bf16,
    float * clean_condition,
    cudaStream_t stream) {
    if (!final_conv1_whdc || !scale_mean_bf16 || !scale_inv_std_bf16 || !clean_condition) {
        return -1;
    }
    constexpr int block = 256;
    constexpr int total = 17 * 20 * 192;
    pack_clean_vision_condition_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        final_conv1_whdc, scale_mean_bf16, scale_inv_std_bf16, clean_condition);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_wan_vae_encoder_conv1_prefix_f32(
    const float * patch_whdc,
    const unsigned short * weight_bf16,
    const unsigned short * bias_bf16,
    float * prefix,
    int rows,
    int cols,
    cudaStream_t stream) {
    constexpr int max_rows = COSMOS3_WAN_VAE_PATCH_W *
                             COSMOS3_WAN_VAE_PATCH_H *
                             COSMOS3_WAN_VAE_FRAMES;
    if (!patch_whdc || !weight_bf16 || !bias_bf16 || !prefix ||
        rows <= 0 || rows > max_rows ||
        cols <= 0 || cols > COSMOS3_WAN_VAE_CONV1_CHANNELS) {
        return -1;
    }
    constexpr int block = 128;
    const int total = rows * cols;
    encoder_conv1_prefix_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        patch_whdc, weight_bf16, bias_bf16, prefix, rows, cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" const char * cosmos3_wan_vae_kernel_status(void) {
    return "cosmos3-wan-vae-cuda-ok ops=robolab_image_to_patch_whdc,encoder_conv1,down_primitives,mid_attention,tail,clean_pack";
}
