// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "cosmos3_w8_marlin_cuda.h"
#include "third_party/vllm_marlin/core/scalar_type.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK_RET(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "vla(cosmos3_w8): CUDA error %s at %s:%d (%s)\n", cudaGetErrorString(_e), __FILE__, __LINE__, #x); \
    return -1; }} while (0)

namespace marlin {
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, int prob_m, int prob_n, int prob_k,
               int lda, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float);
}

namespace {

__device__ float bf16_u16_to_f32(unsigned short bits) {
    __nv_bfloat16 v;
    *reinterpret_cast<unsigned short *>(&v) = bits;
    return __bfloat162float(v);
}

// Temporary validation unpack:
// Treat each int32 qweight word as four signed int8 values in little-endian
// order.  This is deliberately isolated behind a "ref" symbol because the real
// VllmGptqMarlinW8A16Linear layout may require tile/permutation handling.
__device__ int unpack_i8_lane(int packed, int lane) {
    const int shift = lane * 8;
    return static_cast<int>(static_cast<signed char>((packed >> shift) & 0xff));
}

__device__ int marlin_w8_dst_byte(int k, int n, int out_features) {
    const int row = k / 16;
    const int col = (n / 64) * 256
        + (n % 8) * 32
        + ((n / 8) % 8)
        + ((k % 8) / 2) * 8;
    const int lane = (k % 2) * 2 + ((k / 8) % 2);
    return ((row * (out_features * 4) + col) * 4) + lane;
}

__device__ int unpack_marlin_uint8b128(const int * qweight,
                                       int k,
                                       int out_col,
                                       int in_features,
                                       int out_features,
                                       int qweight_rows,
                                       int qweight_cols) {
    if (k < 0 || k >= in_features || out_col < 0 || out_col >= out_features) return 0;
    const int dst_byte = marlin_w8_dst_byte(k, out_col, out_features);
    const int word = dst_byte / 4;
    const int lane = dst_byte & 3;
    const int row = word / qweight_cols;
    const int col = word - row * qweight_cols;
    if (row < 0 || row >= qweight_rows || col < 0 || col >= qweight_cols) return 0;
    const unsigned int packed = static_cast<unsigned int>(qweight[row * qweight_cols + col]);
    const int u8 = static_cast<int>((packed >> (lane * 8)) & 0xff);
    return u8 - 128;
}

__device__ int marlin_scale_index(int out_col, int out_features) {
    (void) out_features;
    constexpr int inv_scale_perm_single[32] = {
        0, 1, 8, 9, 16, 17, 24, 25,
        2, 3, 10, 11, 18, 19, 26, 27,
        4, 5, 12, 13, 20, 21, 28, 29,
        6, 7, 14, 15, 22, 23, 30, 31,
    };
    return (out_col / 32) * 32 + inv_scale_perm_single[out_col & 31];
}

__global__ void w8a16_linear_f32_ref_kernel(
        const float * __restrict__ x,
        const int * __restrict__ qweight,
        const unsigned short * __restrict__ scales_bf16,
        float * __restrict__ y,
        int batch,
        int in_features,
        int out_features,
        int qweight_rows,
        int qweight_cols,
        int scale_rows,
        int scale_cols) {
    const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (b >= batch || out_col >= out_features) return;

    float acc = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        const int packed_index = k / 4;
        const int lane = k & 3;
        if (packed_index >= qweight_rows || out_col >= qweight_cols) continue;

        const int q = unpack_i8_lane(qweight[packed_index * qweight_cols + out_col], lane);
        const int scale_index = (scale_rows == out_features && scale_cols == 1)
            ? out_col
            : (out_col < scale_rows ? out_col * scale_cols : 0);
        const float s = scales_bf16 ? bf16_u16_to_f32(scales_bf16[scale_index]) : 1.0f;
        acc += x[b * in_features + k] * (static_cast<float>(q) * s);
    }
    y[b * out_features + out_col] = acc;
}

__global__ void w8a16_linear_f32_tiled_mapped_kernel(
        const float * __restrict__ x,
        const int * __restrict__ qweight,
        const unsigned short * __restrict__ scales_bf16,
        float * __restrict__ y,
        int batch,
        int in_features,
        int out_features,
        int qweight_rows,
        int qweight_cols,
        int scale_rows,
        int scale_cols) {
    constexpr int kRows = 8;
    constexpr int kCols = 16;
    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;
    const int b = blockIdx.y * kRows + local_row;
    const int out_col = blockIdx.x * kCols + local_col;
    if (b >= batch || out_col >= out_features) return;

    const int mapped_scale = marlin_scale_index(out_col, out_features);
    const int scale_index = (scale_rows == 1 && scale_cols == out_features)
        ? mapped_scale
        : ((scale_rows == out_features && scale_cols == 1) ? mapped_scale : 0);
    const float s = scales_bf16 ? bf16_u16_to_f32(scales_bf16[scale_index]) : 1.0f;

    float acc = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        const int q = unpack_marlin_uint8b128(qweight, k, out_col, in_features, out_features,
                                              qweight_rows, qweight_cols);
        acc = fmaf(x[b * in_features + k], static_cast<float>(q) * s, acc);
    }
    y[b * out_features + out_col] = acc;
}

__global__ void dequant_marlin_w8_to_f32_colmajor_kernel(
        const int * __restrict__ qweight,
        const unsigned short * __restrict__ scales_bf16,
        float * __restrict__ w_colmajor,
        int in_features,
        int out_features,
        int qweight_rows,
        int qweight_cols,
        int scale_rows,
        int scale_cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = in_features * out_features;
    if (idx >= total) return;
    const int out_col = idx % out_features;
    const int k = idx / out_features;
    const int mapped_scale = marlin_scale_index(out_col, out_features);
    const int scale_index = (scale_rows == 1 && scale_cols == out_features)
        ? mapped_scale
        : ((scale_rows == out_features && scale_cols == 1) ? mapped_scale : 0);
    const float s = scales_bf16 ? bf16_u16_to_f32(scales_bf16[scale_index]) : 1.0f;
    const int q = unpack_marlin_uint8b128(qweight, k, out_col, in_features, out_features,
                                          qweight_rows, qweight_cols);
    w_colmajor[out_col + k * out_features] = static_cast<float>(q) * s;
}

__global__ void round_f32_to_bf16_value_kernel(float * y, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    y[idx] = __bfloat162float(__float2bfloat16(y[idx]));
}

__global__ void f32_to_bf16_kernel(const float * x, __nv_bfloat16 * y, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    y[idx] = __float2bfloat16(x[idx]);
}

__global__ void bf16_to_f32_kernel(const __nv_bfloat16 * x, float * y, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    y[idx] = __bfloat162float(x[idx]);
}

__global__ void smoke_kernel(float * out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = 1.0f;
}

bool marlin_scale_layout_compatible(int scale_rows, int scale_cols, int out_features) {
    return (scale_rows == 1 && scale_cols == out_features) ||
           (scale_rows == out_features && scale_cols == 1);
}

} // namespace

namespace {

cublasHandle_t get_w8_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }
    return handle;
}

int cosmos3_w8a16_linear_f32_cublas_dequant(
    const float * x,
    const int * qweight,
    const unsigned short * scales_bf16,
    float * y,
    int batch,
    int in_features,
    int out_features,
    int qweight_rows,
    int qweight_cols,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream) {
    float * w_colmajor = nullptr;
    const size_t weight_elems = static_cast<size_t>(in_features) * static_cast<size_t>(out_features);
    if (cudaMallocAsync(reinterpret_cast<void **>(&w_colmajor), weight_elems * sizeof(float), stream) != cudaSuccess) {
        return -1;
    }

    constexpr int block = 256;
    dequant_marlin_w8_to_f32_colmajor_kernel<<<(static_cast<int>(weight_elems) + block - 1) / block, block, 0, stream>>>(
        qweight,
        scales_bf16,
        w_colmajor,
        in_features,
        out_features,
        qweight_rows,
        qweight_cols,
        scale_rows,
        scale_cols);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFreeAsync(w_colmajor, stream);
        return -1;
    }

    cublasHandle_t handle = get_w8_cublas_handle();
    if (!handle) {
        cudaFreeAsync(w_colmajor, stream);
        return -1;
    }
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
        cudaFreeAsync(w_colmajor, stream);
        return -1;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // y(row-major BxN) is the same memory as column-major C^T(NxB):
    // C^T = W_colmajor(NxK) * X^T(KxB).
    const cublasStatus_t st = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        out_features,
        batch,
        in_features,
        &alpha,
        w_colmajor,
        CUDA_R_32F,
        out_features,
        x,
        CUDA_R_32F,
        in_features,
        &beta,
        y,
        CUDA_R_32F,
        out_features,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) {
        const int y_elems = batch * out_features;
        round_f32_to_bf16_value_kernel<<<(y_elems + block - 1) / block, block, 0, stream>>>(y, y_elems);
        if (cudaGetLastError() != cudaSuccess) {
            cudaFreeAsync(w_colmajor, stream);
            return -1;
        }
    }
    cudaFreeAsync(w_colmajor, stream);
    return st == CUBLAS_STATUS_SUCCESS ? 0 : -1;
}

int cosmos3_w8a16_linear_f32_marlin(const float * x,
                                    const int * qweight,
                                    const unsigned short * scales_bf16,
                                    float * y,
                                    int batch,
                                    int in_features,
                                    int out_features,
                                    int scale_rows,
                                    int scale_cols,
                                    cudaStream_t stream) {
    if (!x || !qweight || !scales_bf16 || !y) return -1;
    if (!marlin_scale_layout_compatible(scale_rows, scale_cols, out_features)) return -1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return -1;
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) return -1;

    constexpr int block = 256;
    const int x_elems = batch * in_features;
    const int y_elems = batch * out_features;
    __nv_bfloat16 * x_bf16 = nullptr;
    __nv_bfloat16 * y_bf16 = nullptr;
    float * c_tmp = nullptr;
    int * workspace = nullptr;

    const int max_m_block_size = std::min(((batch + 15) / 16) * 16, 64);
    const size_t c_tmp_elems = static_cast<size_t>(sms) * static_cast<size_t>(max_m_block_size) * 256u;

    if (cudaMallocAsync(reinterpret_cast<void **>(&x_bf16),
                        static_cast<size_t>(x_elems) * sizeof(__nv_bfloat16),
                        stream) != cudaSuccess) return -1;
    if (cudaMallocAsync(reinterpret_cast<void **>(&y_bf16),
                        static_cast<size_t>(y_elems) * sizeof(__nv_bfloat16),
                        stream) != cudaSuccess) {
        cudaFreeAsync(x_bf16, stream);
        return -1;
    }
    if (cudaMallocAsync(reinterpret_cast<void **>(&c_tmp),
                        c_tmp_elems * sizeof(float),
                        stream) != cudaSuccess) {
        cudaFreeAsync(y_bf16, stream);
        cudaFreeAsync(x_bf16, stream);
        return -1;
    }
    if (cudaMallocAsync(reinterpret_cast<void **>(&workspace),
                        static_cast<size_t>(sms) * sizeof(int),
                        stream) != cudaSuccess) {
        cudaFreeAsync(c_tmp, stream);
        cudaFreeAsync(y_bf16, stream);
        cudaFreeAsync(x_bf16, stream);
        return -1;
    }
    if (cudaMemsetAsync(workspace, 0, static_cast<size_t>(sms) * sizeof(int), stream) != cudaSuccess) {
        cudaFreeAsync(workspace, stream);
        cudaFreeAsync(c_tmp, stream);
        cudaFreeAsync(y_bf16, stream);
        cudaFreeAsync(x_bf16, stream);
        return -1;
    }

    f32_to_bf16_kernel<<<(x_elems + block - 1) / block, block, 0, stream>>>(x, x_bf16, x_elems);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFreeAsync(workspace, stream);
        cudaFreeAsync(c_tmp, stream);
        cudaFreeAsync(y_bf16, stream);
        cudaFreeAsync(x_bf16, stream);
        return -1;
    }

    marlin::marlin_mm(
        x_bf16,
        qweight,
        y_bf16,
        c_tmp,
        nullptr,
        nullptr,
        const_cast<unsigned short *>(scales_bf16),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        batch,
        out_features,
        in_features,
        in_features,
        workspace,
        vllm::kBFloat16,
        vllm::kU8B128,
        vllm::kBFloat16,
        vllm::kBFloat16,
        false,
        false,
        true,
        false,
        1,
        -1,
        dev,
        stream,
        -1,
        -1,
        sms,
        false,
        true,
        false);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFreeAsync(workspace, stream);
        cudaFreeAsync(c_tmp, stream);
        cudaFreeAsync(y_bf16, stream);
        cudaFreeAsync(x_bf16, stream);
        return -1;
    }
    bf16_to_f32_kernel<<<(y_elems + block - 1) / block, block, 0, stream>>>(y_bf16, y, y_elems);
    const bool ok = cudaGetLastError() == cudaSuccess;
    cudaFreeAsync(workspace, stream);
    cudaFreeAsync(c_tmp, stream);
    cudaFreeAsync(y_bf16, stream);
    cudaFreeAsync(x_bf16, stream);
    return ok ? 0 : -1;
}

int cosmos3_w8a16_linear_f32_marlin_ws(const float * x,
                                       const int * qweight,
                                       const unsigned short * scales_bf16,
                                       float * y,
                                       unsigned short * x_bf16_workspace,
                                       unsigned short * y_bf16_workspace,
                                       float * c_tmp_workspace,
                                       int * int_workspace,
                                       int batch,
                                       int in_features,
                                       int out_features,
                                       int scale_rows,
                                       int scale_cols,
                                       cudaStream_t stream) {
    if (!x || !qweight || !scales_bf16 || !y ||
        !x_bf16_workspace || !y_bf16_workspace || !c_tmp_workspace || !int_workspace) return -1;
    if (!marlin_scale_layout_compatible(scale_rows, scale_cols, out_features)) return -1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return -1;
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) return -1;

    constexpr int block = 256;
    const int x_elems = batch * in_features;
    const int y_elems = batch * out_features;
    __nv_bfloat16 * x_bf16 = reinterpret_cast<__nv_bfloat16 *>(x_bf16_workspace);
    __nv_bfloat16 * y_bf16 = reinterpret_cast<__nv_bfloat16 *>(y_bf16_workspace);

    if (cudaMemsetAsync(int_workspace, 0, static_cast<size_t>(sms) * sizeof(int), stream) != cudaSuccess) {
        return -1;
    }

    f32_to_bf16_kernel<<<(x_elems + block - 1) / block, block, 0, stream>>>(x, x_bf16, x_elems);
    if (cudaGetLastError() != cudaSuccess) {
        return -1;
    }

    marlin::marlin_mm(
        x_bf16,
        qweight,
        y_bf16,
        c_tmp_workspace,
        nullptr,
        nullptr,
        const_cast<unsigned short *>(scales_bf16),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        batch,
        out_features,
        in_features,
        in_features,
        int_workspace,
        vllm::kBFloat16,
        vllm::kU8B128,
        vllm::kBFloat16,
        vllm::kBFloat16,
        false,
        false,
        true,
        false,
        1,
        -1,
        dev,
        stream,
        -1,
        -1,
        sms,
        false,
        true,
        false);
    if (cudaGetLastError() != cudaSuccess) {
        return -1;
    }
    bf16_to_f32_kernel<<<(y_elems + block - 1) / block, block, 0, stream>>>(y_bf16, y, y_elems);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cosmos3_w8a16_f32_to_bf16_ws_impl(const float * x,
                                      unsigned short * x_bf16_workspace,
                                      int count,
                                      cudaStream_t stream) {
    if (!x || !x_bf16_workspace || count <= 0) return -1;
    constexpr int block = 256;
    f32_to_bf16_kernel<<<(count + block - 1) / block, block, 0, stream>>>(
        x, reinterpret_cast<__nv_bfloat16 *>(x_bf16_workspace), count);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cosmos3_w8a16_linear_bf16_ws_impl(const unsigned short * x_bf16,
                                      const int * qweight,
                                      const unsigned short * scales_bf16,
                                      float * y,
                                      unsigned short * y_bf16_workspace,
                                      float * c_tmp_workspace,
                                      int * int_workspace,
                                      int batch,
                                      int in_features,
                                      int out_features,
                                      int scale_rows,
                                      int scale_cols,
                                      cudaStream_t stream) {
    if (!x_bf16 || !qweight || !scales_bf16 || !y ||
        !y_bf16_workspace || !c_tmp_workspace || !int_workspace) return -1;
    if (!marlin_scale_layout_compatible(scale_rows, scale_cols, out_features)) return -1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return -1;
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) return -1;

    constexpr int block = 256;
    const int y_elems = batch * out_features;
    __nv_bfloat16 * y_bf16 = reinterpret_cast<__nv_bfloat16 *>(y_bf16_workspace);

    if (cudaMemsetAsync(int_workspace, 0, static_cast<size_t>(sms) * sizeof(int), stream) != cudaSuccess) {
        return -1;
    }

    marlin::marlin_mm(
        x_bf16,
        qweight,
        y_bf16,
        c_tmp_workspace,
        nullptr,
        nullptr,
        const_cast<unsigned short *>(scales_bf16),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        batch,
        out_features,
        in_features,
        in_features,
        int_workspace,
        vllm::kBFloat16,
        vllm::kU8B128,
        vllm::kBFloat16,
        vllm::kBFloat16,
        false,
        false,
        true,
        false,
        1,
        -1,
        dev,
        stream,
        -1,
        -1,
        sms,
        false,
        true,
        false);
    if (cudaGetLastError() != cudaSuccess) {
        return -1;
    }
    bf16_to_f32_kernel<<<(y_elems + block - 1) / block, block, 0, stream>>>(y_bf16, y, y_elems);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // namespace

extern "C" int cosmos3_w8a16_f32_to_bf16_ws(
    const float * x,
    unsigned short * x_bf16_workspace,
    int count,
    cudaStream_t stream) {
    return cosmos3_w8a16_f32_to_bf16_ws_impl(x, x_bf16_workspace, count, stream);
}

extern "C" int cosmos3_w8a16_linear_bf16_ws(
    const unsigned short * x_bf16,
    const int * qweight,
    const unsigned short * scales_bf16,
    float * y,
    unsigned short * y_bf16_workspace,
    float * c_tmp_workspace,
    int * int_workspace,
    int batch,
    int in_features,
    int out_features,
    int qweight_rows,
    int qweight_cols,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream) {
    (void) qweight_rows;
    (void) qweight_cols;
    return cosmos3_w8a16_linear_bf16_ws_impl(x_bf16,
                                             qweight,
                                             scales_bf16,
                                             y,
                                             y_bf16_workspace,
                                             c_tmp_workspace,
                                             int_workspace,
                                             batch,
                                             in_features,
                                             out_features,
                                             scale_rows,
                                             scale_cols,
                                             stream);
}

extern "C" int cosmos3_w8a16_linear_f32_ref(
    const float * x,
    const int * qweight,
    const unsigned short * scales_bf16,
    float * y,
    int batch,
    int in_features,
    int out_features,
    int qweight_rows,
    int qweight_cols,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream) {
    if (!x || !qweight || !y || batch <= 0 || in_features <= 0 || out_features <= 0) return -1;
    dim3 block(128, 1, 1);
    dim3 grid((out_features + block.x - 1) / block.x, batch, 1);
    w8a16_linear_f32_ref_kernel<<<grid, block, 0, stream>>>(
        x, qweight, scales_bf16, y, batch, in_features, out_features,
        qweight_rows, qweight_cols, scale_rows, scale_cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_w8a16_linear_f32(
    const float * x,
    const int * qweight,
    const unsigned short * scales_bf16,
    float * y,
    int batch,
    int in_features,
    int out_features,
    int qweight_rows,
    int qweight_cols,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream) {
    if (!x || !qweight || !y || batch <= 0 || in_features <= 0 || out_features <= 0) return -1;
    if ((in_features % 16) != 0 || (out_features % 64) != 0) return -1;
    if (marlin_scale_layout_compatible(scale_rows, scale_cols, out_features) &&
        (in_features % 128) == 0 && (out_features % 64) == 0 &&
        cosmos3_w8a16_linear_f32_marlin(x,
                                        qweight,
                                        scales_bf16,
                                        y,
                                        batch,
                                        in_features,
                                        out_features,
                                        scale_rows,
                                        scale_cols,
                                        stream) == 0) {
        return 0;
    }
    if (cosmos3_w8a16_linear_f32_cublas_dequant(x,
                                                qweight,
                                                scales_bf16,
                                                y,
                                                batch,
                                                in_features,
                                                out_features,
                                                qweight_rows,
                                                qweight_cols,
                                                scale_rows,
                                                scale_cols,
                                                stream) == 0) {
        return 0;
    }
    dim3 block(16, 8, 1);
    dim3 grid((out_features + block.x - 1) / block.x,
              (batch + block.y - 1) / block.y,
              1);
    w8a16_linear_f32_tiled_mapped_kernel<<<grid, block, 0, stream>>>(
        x, qweight, scales_bf16, y, batch, in_features, out_features,
        qweight_rows, qweight_cols, scale_rows, scale_cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_w8a16_linear_f32_ws(
    const float * x,
    const int * qweight,
    const unsigned short * scales_bf16,
    float * y,
    unsigned short * x_bf16_workspace,
    unsigned short * y_bf16_workspace,
    float * c_tmp_workspace,
    int * int_workspace,
    int batch,
    int in_features,
    int out_features,
    int qweight_rows,
    int qweight_cols,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream) {
    if (!x || !qweight || !y || !x_bf16_workspace || !y_bf16_workspace ||
        !c_tmp_workspace || !int_workspace ||
        batch <= 0 || in_features <= 0 || out_features <= 0) return -1;
    if ((in_features % 16) != 0 || (out_features % 64) != 0) return -1;
    if (marlin_scale_layout_compatible(scale_rows, scale_cols, out_features) &&
        (in_features % 128) == 0 && (out_features % 64) == 0 &&
        cosmos3_w8a16_linear_f32_marlin_ws(x,
                                           qweight,
                                           scales_bf16,
                                           y,
                                           x_bf16_workspace,
                                           y_bf16_workspace,
                                           c_tmp_workspace,
                                           int_workspace,
                                           batch,
                                           in_features,
                                           out_features,
                                           scale_rows,
                                           scale_cols,
                                           stream) == 0) {
        return 0;
    }
    if (cosmos3_w8a16_linear_f32_cublas_dequant(x,
                                                qweight,
                                                scales_bf16,
                                                y,
                                                batch,
                                                in_features,
                                                out_features,
                                                qweight_rows,
                                                qweight_cols,
                                                scale_rows,
                                                scale_cols,
                                                stream) == 0) {
        return 0;
    }
    dim3 block(16, 8, 1);
    dim3 grid((out_features + block.x - 1) / block.x,
              (batch + block.y - 1) / block.y,
              1);
    w8a16_linear_f32_tiled_mapped_kernel<<<grid, block, 0, stream>>>(
        x, qweight, scales_bf16, y, batch, in_features, out_features,
        qweight_rows, qweight_cols, scale_rows, scale_cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_w8a16_linear_f32_cublas_debug(
    const float * x,
    const int * qweight,
    const unsigned short * scales_bf16,
    float * y,
    int batch,
    int in_features,
    int out_features,
    int qweight_rows,
    int qweight_cols,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream) {
    return cosmos3_w8a16_linear_f32_cublas_dequant(x,
                                                   qweight,
                                                   scales_bf16,
                                                   y,
                                                   batch,
                                                   in_features,
                                                   out_features,
                                                   qweight_rows,
                                                   qweight_cols,
                                                   scale_rows,
                                                   scale_cols,
                                                   stream);
}

extern "C" const char * cosmos3_w8a16_kernel_status(void) {
    return "marlin_bf16_u8b128_primary_with_cublas_dequant_f32_gemm_debug_fallback";
}

extern "C" int cosmos3_w8a16_cuda_smoke(cudaStream_t stream) {
    float * d = nullptr;
    CUDA_CHECK_RET(cudaMalloc(&d, sizeof(float)));
    smoke_kernel<<<1, 32, 0, stream>>>(d);
    CUDA_CHECK_RET(cudaGetLastError());
    CUDA_CHECK_RET(cudaFree(d));
    return 0;
}
