// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#include "cosmos3_mot_action_cuda.h"
#include "cosmos3_action_bridge_cuda.h"

#include <cmath>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

cublasHandle_t get_mot_cublas_handle() {
    static thread_local cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
        cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    return handle;
}

__device__ __forceinline__ float mot_bf16_to_f32(unsigned short bits) {
    __nv_bfloat16 v;
    *reinterpret_cast<unsigned short *>(&v) = bits;
    return __bfloat162float(v);
}

__device__ __forceinline__ float mot_round_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

__global__ void timestep_freq_bf16_kernel(
        unsigned short * out,
        float timestep,
        int dim,
        float max_period) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    const int half = dim / 2;
    float v = 0.0f;
    if (idx < half) {
        const float freq = expf(-logf(max_period) * static_cast<float>(idx) / static_cast<float>(half));
        v = cosf(timestep * freq);
    } else if (idx < 2 * half) {
        const int j = idx - half;
        const float freq = expf(-logf(max_period) * static_cast<float>(j) / static_cast<float>(half));
        v = sinf(timestep * freq);
    }
    __nv_bfloat16 b = __float2bfloat16(v);
    out[idx] = *reinterpret_cast<unsigned short *>(&b);
}

__global__ void add_bias_round_bf16_kernel(
        float * y,
        const unsigned short * bias,
        int elems,
        int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elems) return;
    const int col = idx % cols;
    y[idx] = mot_round_bf16(y[idx] + mot_bf16_to_f32(bias[col]));
}

__global__ void silu_to_bf16_kernel(
        const float * x,
        unsigned short * y,
        int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = x[idx];
    const float s = v / (1.0f + expf(-v));
    __nv_bfloat16 b = __float2bfloat16(s);
    y[idx] = *reinterpret_cast<unsigned short *>(&b);
}

__global__ void add_timestep_to_action_tokens_kernel(
        float * action_hidden,
        const float * timestep_hidden,
        int action_tokens,
        int condition_action_tokens,
        int hidden) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = action_tokens * hidden;
    if (idx >= total) return;
    const int token = idx / hidden;
    if (token < condition_action_tokens) return;
    const int col = idx % hidden;
    action_hidden[idx] = mot_round_bf16(action_hidden[idx] + timestep_hidden[col]);
}

__global__ void pack_condition_action_kernel(
        const float * condition_hidden,
        const float * action_hidden,
        float * packed_hidden,
        int condition_tokens,
        int action_tokens,
        int hidden) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = (condition_tokens + action_tokens) * hidden;
    if (idx >= total) return;
    const int token = idx / hidden;
    const int col = idx - token * hidden;
    if (token < condition_tokens) {
        packed_hidden[idx] = condition_hidden[token * hidden + col];
    } else {
        const int action_token = token - condition_tokens;
        packed_hidden[idx] = action_hidden[action_token * hidden + col];
    }
}

__global__ void f32_to_bf16_kernel(
        const float * x,
        unsigned short * y,
        int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    __nv_bfloat16 b = __float2bfloat16(x[idx]);
    y[idx] = *reinterpret_cast<unsigned short *>(&b);
}

__global__ void pack_text_vision_action_kernel(
        const float * text_hidden,
        const float * vision_hidden,
        const float * action_hidden,
        float * packed_hidden,
        int text_tokens,
        int vision_tokens,
        int action_tokens,
        int hidden) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = (text_tokens + vision_tokens + action_tokens) * hidden;
    if (idx >= total) return;
    const int token = idx / hidden;
    const int col = idx - token * hidden;
    if (token < text_tokens) {
        packed_hidden[idx] = text_hidden[token * hidden + col];
    } else if (token < text_tokens + vision_tokens) {
        const int vision_token = token - text_tokens;
        packed_hidden[idx] = vision_hidden[vision_token * hidden + col];
    } else {
        const int action_token = token - text_tokens - vision_tokens;
        packed_hidden[idx] = action_hidden[action_token * hidden + col];
    }
}

__global__ void update_action_latents_kernel(
        float * action_latents,
        const float * velocity,
        int action_tokens,
        int max_action_dim,
        int velocity_dim,
        int condition_action_tokens,
        float step) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = action_tokens * velocity_dim;
    if (idx >= total) return;
    const int token = idx / velocity_dim;
    if (token < condition_action_tokens) return;
    const int col = idx - token * velocity_dim;
    action_latents[token * max_action_dim + col] += step * velocity[idx];
}

__global__ void compute_x0_from_flow_kernel(
        const float * sample,
        const float * velocity,
        float * x0,
        int count,
        float sigma) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    x0[idx] = sample[idx] - sigma * velocity[idx];
}

__global__ void mask_action_velocity_kernel(
        float * velocity,
        int action_tokens,
        int max_action_dim,
        int real_action_dim,
        int condition_action_tokens) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = action_tokens * max_action_dim;
    if (idx >= total) return;
    const int token = idx / max_action_dim;
    const int col = idx - token * max_action_dim;
    if (token < condition_action_tokens || col >= real_action_dim) {
        velocity[idx] = 0.0f;
    }
}

__global__ void copy_f32_kernel(
        const float * src,
        float * dst,
        int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = src[idx];
}

__global__ void linear_combo4_kernel(
        const float * x0,
        const float * x1,
        const float * x2,
        const float * x3,
        float * y,
        int count,
        float c0,
        float c1,
        float c2,
        float c3) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    float v = 0.0f;
    if (x0) v += c0 * x0[idx];
    if (x1) v += c1 * x1[idx];
    if (x2) v += c2 * x2[idx];
    if (x3) v += c3 * x3[idx];
    y[idx] = v;
}

int linear_bf16_rowmajor_weight_f32(
        cublasHandle_t handle,
        const unsigned short * x_bf16,
        const unsigned short * weight_bf16,
        const unsigned short * bias_bf16,
        float * y,
        int batch,
        int in_features,
        int out_features,
        cudaStream_t stream) {
    if (!handle || !x_bf16 || !weight_bf16 || !bias_bf16 || !y ||
        batch <= 0 || in_features <= 0 || out_features <= 0) {
        return -1;
    }
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) return -1;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // PyTorch Linear stores weight as row-major [out,in].  Treat that memory as
    // column-major [in,out] and multiply weight^T @ x^T to obtain y^T.
    const cublasStatus_t st = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_features,
        batch,
        in_features,
        &alpha,
        weight_bf16,
        CUDA_R_16BF,
        in_features,
        x_bf16,
        CUDA_R_16BF,
        in_features,
        &beta,
        y,
        CUDA_R_32F,
        out_features,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) return -1;
    constexpr int block = 256;
    add_bias_round_bf16_kernel<<<(batch * out_features + block - 1) / block,
                                 block, 0, stream>>>(
        y, bias_bf16, batch * out_features, out_features);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // namespace

extern "C" int cosmos3_mot_build_action_input_f32(
    const float * action_latents,
    const unsigned short * action2llm_fc_bf16,
    const unsigned short * action2llm_bias_bf16,
    const unsigned short * action_modality_embed_bf16,
    const unsigned short * time_mlp0_weight_bf16,
    const unsigned short * time_mlp0_bias_bf16,
    const unsigned short * time_mlp2_weight_bf16,
    const unsigned short * time_mlp2_bias_bf16,
    float * action_hidden,
    float * timestep_hidden,
    float timestep,
    float timestep_scale,
    int action_tokens,
    int max_action_dim,
    int hidden,
    int time_freq_dim,
    int domains,
    int domain_id,
    int condition_action_tokens,
    cudaStream_t stream) {
    unsigned short * action_ws = nullptr;
    unsigned short * time_freq_ws = nullptr;
    float * time_mlp0_ws = nullptr;
    unsigned short * time_mlp0_bf16_ws = nullptr;
    if (cudaMallocAsync(reinterpret_cast<void **>(&action_ws),
                        static_cast<size_t>(action_tokens) * max_action_dim * sizeof(unsigned short),
                        stream) != cudaSuccess ||
        cudaMallocAsync(reinterpret_cast<void **>(&time_freq_ws),
                        static_cast<size_t>(time_freq_dim) * sizeof(unsigned short),
                        stream) != cudaSuccess ||
        cudaMallocAsync(reinterpret_cast<void **>(&time_mlp0_ws),
                        static_cast<size_t>(hidden) * sizeof(float),
                        stream) != cudaSuccess ||
        cudaMallocAsync(reinterpret_cast<void **>(&time_mlp0_bf16_ws),
                        static_cast<size_t>(hidden) * sizeof(unsigned short),
                        stream) != cudaSuccess) {
        return -1;
    }
    const int rc = cosmos3_mot_build_action_input_f32_ws(
        action_latents,
        action2llm_fc_bf16,
        action2llm_bias_bf16,
        action_modality_embed_bf16,
        time_mlp0_weight_bf16,
        time_mlp0_bias_bf16,
        time_mlp2_weight_bf16,
        time_mlp2_bias_bf16,
        action_hidden,
        timestep_hidden,
        action_ws,
        time_freq_ws,
        time_mlp0_ws,
        time_mlp0_bf16_ws,
        timestep,
        timestep_scale,
        action_tokens,
        max_action_dim,
        hidden,
        time_freq_dim,
        domains,
        domain_id,
        condition_action_tokens,
        stream);
    cudaFreeAsync(action_ws, stream);
    cudaFreeAsync(time_freq_ws, stream);
    cudaFreeAsync(time_mlp0_ws, stream);
    cudaFreeAsync(time_mlp0_bf16_ws, stream);
    return rc;
}

extern "C" int cosmos3_mot_build_action_input_f32_ws(
    const float * action_latents,
    const unsigned short * action2llm_fc_bf16,
    const unsigned short * action2llm_bias_bf16,
    const unsigned short * action_modality_embed_bf16,
    const unsigned short * time_mlp0_weight_bf16,
    const unsigned short * time_mlp0_bias_bf16,
    const unsigned short * time_mlp2_weight_bf16,
    const unsigned short * time_mlp2_bias_bf16,
    float * action_hidden,
    float * timestep_hidden,
    unsigned short * action_bf16_workspace,
    unsigned short * time_freq_bf16_workspace,
    float * time_mlp0_workspace,
    unsigned short * time_mlp0_bf16_workspace,
    float timestep,
    float timestep_scale,
    int action_tokens,
    int max_action_dim,
    int hidden,
    int time_freq_dim,
    int domains,
    int domain_id,
    int condition_action_tokens,
    cudaStream_t stream) {
    if (!action_latents || !action2llm_fc_bf16 || !action2llm_bias_bf16 ||
        !action_modality_embed_bf16 || !time_mlp0_weight_bf16 ||
        !time_mlp0_bias_bf16 || !time_mlp2_weight_bf16 ||
        !time_mlp2_bias_bf16 || !action_hidden || !timestep_hidden ||
        !action_bf16_workspace || !time_freq_bf16_workspace ||
        !time_mlp0_workspace || !time_mlp0_bf16_workspace ||
        action_tokens <= 0 || max_action_dim <= 0 || hidden <= 0 ||
        time_freq_dim <= 0 || domains <= 0 ||
        domain_id < 0 || domain_id >= domains ||
        condition_action_tokens < 0 || condition_action_tokens > action_tokens) {
        return -1;
    }
    if (cosmos3_action2llm_plus_embed_f32_ws(
            action_latents,
            action2llm_fc_bf16,
            action2llm_bias_bf16,
            action_modality_embed_bf16,
            action_hidden,
            action_bf16_workspace,
            action_tokens,
            domains,
            domain_id,
            stream) != 0) {
        return -1;
    }
    cublasHandle_t handle = get_mot_cublas_handle();
    if (!handle) return -1;
    constexpr int block = 256;
    timestep_freq_bf16_kernel<<<(time_freq_dim + block - 1) / block,
                                block, 0, stream>>>(
        time_freq_bf16_workspace,
        timestep * timestep_scale,
        time_freq_dim,
        10000.0f);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (linear_bf16_rowmajor_weight_f32(
            handle,
            time_freq_bf16_workspace,
            time_mlp0_weight_bf16,
            time_mlp0_bias_bf16,
            time_mlp0_workspace,
            1,
            time_freq_dim,
            hidden,
            stream) != 0) {
        return -1;
    }
    silu_to_bf16_kernel<<<(hidden + block - 1) / block, block, 0, stream>>>(
        time_mlp0_workspace,
        time_mlp0_bf16_workspace,
        hidden);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (linear_bf16_rowmajor_weight_f32(
            handle,
            time_mlp0_bf16_workspace,
            time_mlp2_weight_bf16,
            time_mlp2_bias_bf16,
            timestep_hidden,
            1,
            hidden,
            hidden,
            stream) != 0) {
        return -1;
    }
    add_timestep_to_action_tokens_kernel<<<(action_tokens * hidden + block - 1) / block,
                                           block, 0, stream>>>(
        action_hidden,
        timestep_hidden,
        action_tokens,
        condition_action_tokens,
        hidden);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" const char * cosmos3_mot_action_kernel_status(void) {
    return "cosmos3-mot-action-cuda-ok ops=action2llm,time_embedder_mlp,timestep_scatter_add,condition_action_pack";
}

extern "C" int cosmos3_mot_pack_condition_action_f32(
    const float * condition_hidden,
    const float * action_hidden,
    float * packed_hidden,
    int condition_tokens,
    int action_tokens,
    int hidden,
    cudaStream_t stream) {
    if (!condition_hidden || !action_hidden || !packed_hidden ||
        condition_tokens <= 0 || action_tokens <= 0 || hidden <= 0) {
        return -1;
    }
    constexpr int block = 256;
    const int total = (condition_tokens + action_tokens) * hidden;
    pack_condition_action_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        condition_hidden,
        action_hidden,
        packed_hidden,
        condition_tokens,
        action_tokens,
        hidden);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_mot_vae2llm_f32_ws(
    const float * vision_latents,
    const unsigned short * vae2llm_weight_bf16,
    const unsigned short * vae2llm_bias_bf16,
    float * vision_hidden,
    unsigned short * vision_bf16_workspace,
    int vision_tokens,
    int patch_dim,
    int hidden,
    cudaStream_t stream) {
    if (!vision_latents || !vae2llm_weight_bf16 || !vae2llm_bias_bf16 ||
        !vision_hidden || !vision_bf16_workspace ||
        vision_tokens <= 0 || patch_dim <= 0 || hidden <= 0) {
        return -1;
    }
    constexpr int block = 256;
    const int elems = vision_tokens * patch_dim;
    f32_to_bf16_kernel<<<(elems + block - 1) / block, block, 0, stream>>>(
        vision_latents,
        vision_bf16_workspace,
        elems);
    if (cudaGetLastError() != cudaSuccess) return -1;
    cublasHandle_t handle = get_mot_cublas_handle();
    if (!handle) return -1;
    return linear_bf16_rowmajor_weight_f32(
        handle,
        vision_bf16_workspace,
        vae2llm_weight_bf16,
        vae2llm_bias_bf16,
        vision_hidden,
        vision_tokens,
        patch_dim,
        hidden,
        stream);
}

extern "C" int cosmos3_mot_add_timestep_to_noisy_tokens_f32(
    float * tokens,
    const float * timestep_hidden,
    int tokens_count,
    int condition_tokens,
    int hidden,
    cudaStream_t stream) {
    if (!tokens || !timestep_hidden || tokens_count <= 0 || hidden <= 0 ||
        condition_tokens < 0 || condition_tokens > tokens_count) {
        return -1;
    }
    constexpr int block = 256;
    add_timestep_to_action_tokens_kernel<<<(tokens_count * hidden + block - 1) / block,
                                           block, 0, stream>>>(
        tokens,
        timestep_hidden,
        tokens_count,
        condition_tokens,
        hidden);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_mot_llm2vae_f32_ws(
    const float * vision_hidden,
    const unsigned short * llm2vae_weight_bf16,
    const unsigned short * llm2vae_bias_bf16,
    float * vision_velocity,
    unsigned short * vision_hidden_bf16_workspace,
    int vision_tokens,
    int hidden,
    int patch_dim,
    cudaStream_t stream) {
    if (!vision_hidden || !llm2vae_weight_bf16 || !llm2vae_bias_bf16 ||
        !vision_velocity || !vision_hidden_bf16_workspace ||
        vision_tokens <= 0 || hidden <= 0 || patch_dim <= 0) {
        return -1;
    }
    constexpr int block = 256;
    const int elems = vision_tokens * hidden;
    f32_to_bf16_kernel<<<(elems + block - 1) / block, block, 0, stream>>>(
        vision_hidden,
        vision_hidden_bf16_workspace,
        elems);
    if (cudaGetLastError() != cudaSuccess) return -1;
    cublasHandle_t handle = get_mot_cublas_handle();
    if (!handle) return -1;
    return linear_bf16_rowmajor_weight_f32(
        handle,
        vision_hidden_bf16_workspace,
        llm2vae_weight_bf16,
        llm2vae_bias_bf16,
        vision_velocity,
        vision_tokens,
        hidden,
        patch_dim,
        stream);
}

extern "C" int cosmos3_mot_pack_text_vision_action_f32(
    const float * text_hidden,
    const float * vision_hidden,
    const float * action_hidden,
    float * packed_hidden,
    int text_tokens,
    int vision_tokens,
    int action_tokens,
    int hidden,
    cudaStream_t stream) {
    if (!text_hidden || !vision_hidden || !action_hidden || !packed_hidden ||
        text_tokens <= 0 || vision_tokens <= 0 || action_tokens <= 0 || hidden <= 0) {
        return -1;
    }
    constexpr int block = 256;
    const int total = (text_tokens + vision_tokens + action_tokens) * hidden;
    pack_text_vision_action_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        text_hidden,
        vision_hidden,
        action_hidden,
        packed_hidden,
        text_tokens,
        vision_tokens,
        action_tokens,
        hidden);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_mot_update_action_latents_f32(
    float * action_latents,
    const float * velocity,
    int action_tokens,
    int max_action_dim,
    int velocity_dim,
    int condition_action_tokens,
    float step,
    cudaStream_t stream) {
    if (!action_latents || !velocity || action_tokens <= 0 ||
        max_action_dim <= 0 || velocity_dim <= 0 || velocity_dim > max_action_dim ||
        condition_action_tokens < 0 || condition_action_tokens > action_tokens) {
        return -1;
    }
    constexpr int block = 256;
    update_action_latents_kernel<<<(action_tokens * velocity_dim + block - 1) / block,
                                   block, 0, stream>>>(
        action_latents,
        velocity,
        action_tokens,
        max_action_dim,
        velocity_dim,
        condition_action_tokens,
        step);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_mot_compute_x0_from_flow_f32(
    const float * sample,
    const float * velocity,
    float * x0,
    int count,
    float sigma,
    cudaStream_t stream) {
    if (!sample || !velocity || !x0 || count <= 0) return -1;
    constexpr int block = 256;
    compute_x0_from_flow_kernel<<<(count + block - 1) / block, block, 0, stream>>>(
        sample, velocity, x0, count, sigma);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_mot_mask_action_velocity_f32(
    float * velocity,
    int action_tokens,
    int max_action_dim,
    int real_action_dim,
    int condition_action_tokens,
    cudaStream_t stream) {
    if (!velocity || action_tokens <= 0 || max_action_dim <= 0 ||
        real_action_dim <= 0 || real_action_dim > max_action_dim ||
        condition_action_tokens < 0 || condition_action_tokens > action_tokens) {
        return -1;
    }
    constexpr int block = 256;
    const int total = action_tokens * max_action_dim;
    mask_action_velocity_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
        velocity,
        action_tokens,
        max_action_dim,
        real_action_dim,
        condition_action_tokens);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_mot_copy_f32(
    const float * src,
    float * dst,
    int count,
    cudaStream_t stream) {
    if (!src || !dst || count <= 0) return -1;
    constexpr int block = 256;
    copy_f32_kernel<<<(count + block - 1) / block, block, 0, stream>>>(src, dst, count);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int cosmos3_mot_linear_combo4_f32(
    const float * x0,
    const float * x1,
    const float * x2,
    const float * x3,
    float * y,
    int count,
    float c0,
    float c1,
    float c2,
    float c3,
    cudaStream_t stream) {
    if (!y || count <= 0) return -1;
    constexpr int block = 256;
    linear_combo4_kernel<<<(count + block - 1) / block, block, 0, stream>>>(
        x0, x1, x2, x3, y, count, c0, c1, c2, c3);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
