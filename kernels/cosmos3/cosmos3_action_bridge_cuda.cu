// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#include "cosmos3_action_bridge_cuda.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kActionBridgeMaxActionDim = 64;
constexpr int kActionBridgeHidden = 4096;

cublasHandle_t get_action_bridge_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return handle;
}

__device__ __forceinline__ float bridge_bf16_to_f32(unsigned short bits) {
    __nv_bfloat16 v;
    *reinterpret_cast<unsigned short *>(&v) = bits;
    return __bfloat162float(v);
}

__device__ __forceinline__ float bridge_round_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

__global__ void add_domain_bias_kernel(
        float * y,
        const unsigned short * bias,
        int batch,
        int output_size,
        int domains,
        int domain_id) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * output_size;
    if (idx >= total) return;
    const int col = idx % output_size;
    const float linear = bridge_round_bf16(y[idx]);
    y[idx] = bridge_round_bf16(linear + bridge_bf16_to_f32(bias[domain_id * output_size + col]));
    (void) domains;
}

__global__ void f32_to_bf16_kernel(
        const float * x,
        __nv_bfloat16 * y,
        int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = __float2bfloat16(x[idx]);
}

__global__ void add_action_modality_embed_kernel(
        float * y,
        const unsigned short * embed,
        int batch,
        int hidden,
        int round_linear_to_bf16) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * hidden;
    if (idx >= total) return;
    const int col = idx % hidden;
    float v = y[idx];
    if (round_linear_to_bf16) v = bridge_round_bf16(v);
    y[idx] = round_linear_to_bf16 ?
        bridge_round_bf16(v + bridge_bf16_to_f32(embed[col])) :
        v + bridge_bf16_to_f32(embed[col]);
}

__global__ void slice_real_action_kernel(
        const float * full_action,
        float * action,
        int batch,
        int max_action_dim,
        int real_action_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * real_action_dim;
    if (idx >= total) return;
    const int row = idx / real_action_dim;
    const int col = idx - row * real_action_dim;
    action[idx] = full_action[row * max_action_dim + col];
}

} // namespace

extern "C" int cosmos3_domain_aware_linear_bf16_f32(
    const float * x,
    const unsigned short * fc_weight_bf16,
    const unsigned short * bias_bf16,
    float * y,
    int batch,
    int input_size,
    int output_size,
    int domains,
    int domain_id,
    cudaStream_t stream) {
    if (!x || !fc_weight_bf16 || !bias_bf16 || !y ||
        batch <= 0 || input_size <= 0 || output_size <= 0 ||
        domains <= 0 || domain_id < 0 || domain_id >= domains) {
        return -1;
    }
    __nv_bfloat16 * x_bf16 = nullptr;
    const size_t x_elems = static_cast<size_t>(batch) * static_cast<size_t>(input_size);
    if (cudaMallocAsync(reinterpret_cast<void **>(&x_bf16), x_elems * sizeof(__nv_bfloat16), stream) != cudaSuccess) {
        return -1;
    }
    const int rc = cosmos3_domain_aware_linear_bf16_f32_ws(
        x,
        fc_weight_bf16,
        bias_bf16,
        y,
        reinterpret_cast<unsigned short *>(x_bf16),
        batch,
        input_size,
        output_size,
        domains,
        domain_id,
        stream);
    cudaFreeAsync(x_bf16, stream);
    return rc;
}

extern "C" int cosmos3_domain_aware_linear_bf16_f32_ws(
    const float * x,
    const unsigned short * fc_weight_bf16,
    const unsigned short * bias_bf16,
    float * y,
    unsigned short * x_bf16_workspace,
    int batch,
    int input_size,
    int output_size,
    int domains,
    int domain_id,
    cudaStream_t stream) {
    if (!x || !fc_weight_bf16 || !bias_bf16 || !y || !x_bf16_workspace ||
        batch <= 0 || input_size <= 0 || output_size <= 0 ||
        domains <= 0 || domain_id < 0 || domain_id >= domains) {
        return -1;
    }
    cublasHandle_t handle = get_action_bridge_cublas_handle();
    if (!handle || cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    __nv_bfloat16 * x_bf16 = reinterpret_cast<__nv_bfloat16 *>(x_bf16_workspace);
    const size_t x_elems = static_cast<size_t>(batch) * static_cast<size_t>(input_size);
    constexpr int block = 256;
    f32_to_bf16_kernel<<<(static_cast<int>(x_elems) + block - 1) / block, block, 0, stream>>>(
        x, x_bf16, static_cast<int>(x_elems));
    if (cudaGetLastError() != cudaSuccess) {
        return -1;
    }
    const unsigned short * w_domain =
        fc_weight_bf16 + static_cast<size_t>(domain_id) *
                         static_cast<size_t>(input_size) *
                         static_cast<size_t>(output_size);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major y[batch,out] = x[batch,in] @ w[in,out].
    // Interpreted by cuBLAS as column-major:
    //   y^T[out,batch] = w^T[out,in] @ x^T[in,batch]
    // The row-major [in,out] weight slice is exactly column-major [out,in].
    const cublasStatus_t st = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        output_size,
        batch,
        input_size,
        &alpha,
        w_domain,
        CUDA_R_16BF,
        output_size,
        x_bf16,
        CUDA_R_16BF,
        input_size,
        &beta,
        y,
        CUDA_R_32F,
        output_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }
    add_domain_bias_kernel<<<(batch * output_size + block - 1) / block, block, 0, stream>>>(
        y, bias_bf16, batch, output_size, domains, domain_id);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_action2llm_plus_embed_f32(
    const float * action_latent,
    const unsigned short * action2llm_fc_bf16,
    const unsigned short * action2llm_bias_bf16,
    const unsigned short * action_modality_embed_bf16,
    float * llm_hidden,
    int batch,
    int domains,
    int domain_id,
    cudaStream_t stream) {
    if (!action_modality_embed_bf16) return -1;
    if (cosmos3_domain_aware_linear_bf16_f32(
            action_latent,
            action2llm_fc_bf16,
            action2llm_bias_bf16,
            llm_hidden,
            batch,
            kActionBridgeMaxActionDim,
            kActionBridgeHidden,
            domains,
            domain_id,
            stream) != 0) {
        return -1;
    }
    constexpr int block = 256;
    add_action_modality_embed_kernel<<<(batch * kActionBridgeHidden + block - 1) / block,
                                       block, 0, stream>>>(
        llm_hidden, action_modality_embed_bf16, batch, kActionBridgeHidden, 1);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_action2llm_plus_embed_f32_ws(
    const float * action_latent,
    const unsigned short * action2llm_fc_bf16,
    const unsigned short * action2llm_bias_bf16,
    const unsigned short * action_modality_embed_bf16,
    float * llm_hidden,
    unsigned short * action_bf16_workspace,
    int batch,
    int domains,
    int domain_id,
    cudaStream_t stream) {
    if (!action_modality_embed_bf16 || !action_bf16_workspace) return -1;
    if (cosmos3_domain_aware_linear_bf16_f32_ws(
            action_latent,
            action2llm_fc_bf16,
            action2llm_bias_bf16,
            llm_hidden,
            action_bf16_workspace,
            batch,
            kActionBridgeMaxActionDim,
            kActionBridgeHidden,
            domains,
            domain_id,
            stream) != 0) {
        return -1;
    }
    constexpr int block = 256;
    add_action_modality_embed_kernel<<<(batch * kActionBridgeHidden + block - 1) / block,
                                       block, 0, stream>>>(
        llm_hidden, action_modality_embed_bf16, batch, kActionBridgeHidden, 1);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_llm2action_slice_f32_ws(
    const float * llm_hidden,
    const unsigned short * llm2action_fc_bf16,
    const unsigned short * llm2action_bias_bf16,
    float * full_action_workspace,
    float * action_out,
    unsigned short * llm_bf16_workspace,
    int batch,
    int hidden,
    int max_action_dim,
    int real_action_dim,
    int domains,
    int domain_id,
    cudaStream_t stream) {
    if (!llm_hidden || !llm2action_fc_bf16 || !llm2action_bias_bf16 ||
        !full_action_workspace || !action_out || !llm_bf16_workspace ||
        batch <= 0 || hidden <= 0 || max_action_dim <= 0 ||
        real_action_dim <= 0 || real_action_dim > max_action_dim ||
        domains <= 0 || domain_id < 0 || domain_id >= domains) {
        return -1;
    }
    if (cosmos3_domain_aware_linear_bf16_f32_ws(
            llm_hidden,
            llm2action_fc_bf16,
            llm2action_bias_bf16,
            full_action_workspace,
            llm_bf16_workspace,
            batch,
            hidden,
            max_action_dim,
            domains,
            domain_id,
            stream) != 0) {
        return -1;
    }
    constexpr int block = 256;
    slice_real_action_kernel<<<(batch * real_action_dim + block - 1) / block,
                               block, 0, stream>>>(
        full_action_workspace,
        action_out,
        batch,
        max_action_dim,
        real_action_dim);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" const char * cosmos3_action_bridge_kernel_status(void) {
    return "cosmos3-action-bridge-cuda-ok ops=domain_aware_linear_bf16_cublas,action2llm_plus_embed,llm2action_slice";
}
