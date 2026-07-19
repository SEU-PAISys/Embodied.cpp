// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "arch.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vla {
namespace {

struct GgufReader {
    gguf_context * gctx = nullptr;
    ggml_context * meta_ctx = nullptr;
    FILE * fp = nullptr;
    size_t data_offset = 0;

    bool open(const std::string & path) {
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = &meta_ctx;
        gctx = gguf_init_from_file(path.c_str(), params);
        if (!gctx) {
            std::fprintf(stderr, "vla(groot_n1): failed to open GGUF metadata: %s\n", path.c_str());
            return false;
        }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "vla(groot_n1): failed to open GGUF data: %s\n", path.c_str());
            return false;
        }
        data_offset = gguf_get_data_offset(gctx);
        return true;
    }

    ~GgufReader() {
        if (fp) std::fclose(fp);
        if (gctx) gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
    }

    bool has_key(const char * key) const { return gguf_find_key(gctx, key) >= 0; }
    uint32_t u32(const char * key) const {
        return gguf_get_val_u32(gctx, gguf_find_key(gctx, key));
    }
    float f32(const char * key) const {
        return gguf_get_val_f32(gctx, gguf_find_key(gctx, key));
    }
    bool boolean(const char * key) const {
        return gguf_get_val_bool(gctx, gguf_find_key(gctx, key));
    }
    std::string str(const char * key) const {
        return gguf_get_val_str(gctx, gguf_find_key(gctx, key));
    }
    const ggml_tensor * meta(const char * name) const {
        return ggml_get_tensor(meta_ctx, name);
    }

    bool read_raw(const char * name, void * destination) {
        const int64_t tensor_id = gguf_find_tensor(gctx, name);
        if (tensor_id < 0) {
            std::fprintf(stderr, "vla(groot_n1): missing tensor %s\n", name);
            return false;
        }
        const size_t offset = data_offset + gguf_get_tensor_offset(gctx, tensor_id);
        const size_t bytes = gguf_get_tensor_size(gctx, tensor_id);
        if (std::fseek(fp, static_cast<long>(offset), SEEK_SET) != 0) return false;
        return std::fread(destination, 1, bytes, fp) == bytes;
    }

    std::vector<uint8_t> read_convert(const char * name, ggml_type target) {
        const ggml_tensor * source = meta(name);
        if (!source) {
            std::fprintf(stderr, "vla(groot_n1): missing tensor %s\n", name);
            return {};
        }
        if (source->type == target) {
            std::vector<uint8_t> result(ggml_nbytes(source));
            if (!read_raw(name, result.data())) return {};
            return result;
        }
        if (ggml_is_quantized(source->type)) {
            std::fprintf(stderr,
                         "vla(groot_n1): cannot requantize %s from %s to %s\n",
                         name, ggml_type_name(source->type), ggml_type_name(target));
            return {};
        }

        const int64_t count = ggml_nelements(source);
        std::vector<float> values(static_cast<size_t>(count));
        if (source->type == GGML_TYPE_F32) {
            if (!read_raw(name, values.data())) return {};
        } else if (source->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> bf16(static_cast<size_t>(count));
            if (!read_raw(name, bf16.data())) return {};
            ggml_bf16_to_fp32_row(bf16.data(), values.data(), count);
        } else if (source->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> fp16(static_cast<size_t>(count));
            if (!read_raw(name, fp16.data())) return {};
            ggml_fp16_to_fp32_row(fp16.data(), values.data(), count);
        } else {
            std::fprintf(stderr, "vla(groot_n1): unsupported source type %s for %s\n",
                         ggml_type_name(source->type), name);
            return {};
        }

        if (target == GGML_TYPE_F32) {
            std::vector<uint8_t> result(static_cast<size_t>(count) * sizeof(float));
            std::memcpy(result.data(), values.data(), result.size());
            return result;
        }
        if (target == GGML_TYPE_BF16) {
            std::vector<uint8_t> result(static_cast<size_t>(count) * sizeof(ggml_bf16_t));
            ggml_fp32_to_bf16_row(values.data(),
                                  reinterpret_cast<ggml_bf16_t *>(result.data()), count);
            return result;
        }
        if (target == GGML_TYPE_F16) {
            std::vector<uint8_t> result(static_cast<size_t>(count) * sizeof(ggml_fp16_t));
            ggml_fp32_to_fp16_row(values.data(),
                                  reinterpret_cast<ggml_fp16_t *>(result.data()), count);
            return result;
        }
        if (ggml_is_quantized(target)) {
            const int64_t row_size = source->ne[0];
            const int64_t rows = count / row_size;
            const int64_t block_size = ggml_blck_size(target);
            if (row_size % block_size != 0) {
                std::fprintf(stderr,
                             "vla(groot_n1): %s ne0=%lld is not divisible by %s block=%lld\n",
                             name, static_cast<long long>(row_size), ggml_type_name(target),
                             static_cast<long long>(block_size));
                return {};
            }
            const size_t expected = static_cast<size_t>(rows) *
                static_cast<size_t>(row_size / block_size) * ggml_type_size(target);
            std::vector<uint8_t> result(expected);
            const size_t written = ggml_quantize_chunk(
                target, values.data(), result.data(), 0, rows, row_size, nullptr);
            if (written != expected) {
                std::fprintf(stderr,
                             "vla(groot_n1): quantized size mismatch for %s (%zu != %zu)\n",
                             name, written, expected);
                return {};
            }
            return result;
        }
        std::fprintf(stderr, "vla(groot_n1): unsupported target type %s for %s\n",
                     ggml_type_name(target), name);
        return {};
    }
};

bool ends_with(const std::string & value, const char * suffix) {
    const size_t suffix_size = std::strlen(suffix);
    return value.size() >= suffix_size &&
           value.compare(value.size() - suffix_size, suffix_size, suffix) == 0;
}

ggml_type matmul_type_from_env() {
    const char * value = std::getenv("VLA_GROOT_WEIGHT_DTYPE");
    if (!value || !*value) return GGML_TYPE_Q4_K;
    if (std::strcmp(value, "f32") == 0 || std::strcmp(value, "F32") == 0) return GGML_TYPE_F32;
    if (std::strcmp(value, "bf16") == 0 || std::strcmp(value, "BF16") == 0) return GGML_TYPE_BF16;
    if (std::strcmp(value, "f16") == 0 || std::strcmp(value, "F16") == 0) return GGML_TYPE_F16;
    if (std::strcmp(value, "q8_0") == 0 || std::strcmp(value, "Q8_0") == 0) return GGML_TYPE_Q8_0;
    if (std::strcmp(value, "q6_K") == 0 || std::strcmp(value, "q6_k") == 0) return GGML_TYPE_Q6_K;
    if (std::strcmp(value, "q5_K") == 0 || std::strcmp(value, "q5_k") == 0) return GGML_TYPE_Q5_K;
    if (std::strcmp(value, "q4_K") == 0 || std::strcmp(value, "q4_k") == 0) return GGML_TYPE_Q4_K;
    if (std::strcmp(value, "q4_0") == 0 || std::strcmp(value, "Q4_0") == 0) return GGML_TYPE_Q4_0;
    std::fprintf(stderr,
                 "vla(groot_n1): unknown VLA_GROOT_WEIGHT_DTYPE=%s; using q4_K\n", value);
    return GGML_TYPE_Q4_K;
}

struct LinearWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;
};

struct AttentionWeights {
    LinearWeights q;
    LinearWeights k;
    LinearWeights v;
    LinearWeights out;
};

struct VlBlockWeights {
    ggml_tensor * attn_norm_weight = nullptr;
    ggml_tensor * attn_norm_bias = nullptr;
    AttentionWeights attention;
    ggml_tensor * ffn_norm_weight = nullptr;
    ggml_tensor * ffn_norm_bias = nullptr;
    LinearWeights ffn_up;
    LinearWeights ffn_down;
};

struct DitBlockWeights {
    LinearWeights ada_norm;
    AttentionWeights attention;
    LinearWeights ffn_up;
    LinearWeights ffn_down;
};

struct TraceTensor {
    std::string name;
    ggml_tensor * tensor = nullptr;
};

struct VisionTrace {
    std::string directory;
    std::unordered_map<std::string, int> counts;
};

struct GrootN1ModelArch final : ModelArchBase {
    GrootN1ModelArch() : ModelArchBase(Arch::GROOT_N1) {}
    ~GrootN1ModelArch() override;

    std::vector<float> predict(const Inputs & input) override;

    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t weight_buffer = nullptr;
    ggml_context * weight_context = nullptr;
    bool is_cuda = false;
    int n_threads = 4;
    ggml_type matmul_type = GGML_TYPE_Q4_K;

    llama_model * backbone_model = nullptr;
    llama_context * backbone_context = nullptr;
    mtmd_context * mmproj_context = nullptr;
    int backbone_n_batch = 0;
    VisionTrace vision_trace;

    int64_t backbone_dim = 0;
    int64_t input_dim = 0;
    int64_t head_hidden = 0;
    int64_t max_seq_len = 0;
    int64_t real_action_horizon = 0;
    int vl_layers_count = 0;
    int vl_heads = 0;
    int vl_head_dim = 0;
    int dit_heads = 0;
    int dit_head_dim = 0;
    int attend_text_every_n_blocks = 2;
    int num_timestep_buckets = 1000;
    float layer_norm_eps = 1e-5f;
    float output_norm_eps = 1e-6f;
    bool clip_outliers = true;

    ggml_tensor * vl_norm_weight = nullptr;
    ggml_tensor * vl_norm_bias = nullptr;
    std::vector<VlBlockWeights> vl_blocks;

    LinearWeights state_encoder_0;
    LinearWeights state_encoder_2;
    LinearWeights action_encoder_w1;
    LinearWeights action_encoder_w2;
    LinearWeights action_encoder_w3;
    LinearWeights action_decoder_0;
    LinearWeights action_decoder_2;
    ggml_tensor * action_position = nullptr;

    LinearWeights time_0;
    LinearWeights time_2;
    std::vector<DitBlockWeights> dit_blocks;
    LinearWeights output_norm;
    LinearWeights output_projection;

    std::vector<float> state_q01;
    std::vector<float> state_q99;
    std::vector<float> action_q01;
    std::vector<float> action_q99;
    std::mt19937 rng{std::random_device{}()};
};

ggml_tensor * matmul(ggml_context * context, ggml_tensor * weight, ggml_tensor * input) {
    ggml_tensor * output = ggml_mul_mat(context, weight, input);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    return output;
}

ggml_tensor * linear(ggml_context * context, const LinearWeights & weights,
                     ggml_tensor * input) {
    ggml_tensor * output = matmul(context, weights.weight, input);
    return weights.bias ? ggml_add(context, output, weights.bias) : output;
}

ggml_tensor * layer_norm(ggml_context * context, ggml_tensor * input,
                         float epsilon, ggml_tensor * weight = nullptr,
                         ggml_tensor * bias = nullptr) {
    ggml_tensor * output = ggml_norm(context, input, epsilon);
    if (weight) output = ggml_mul(context, output, weight);
    if (bias) output = ggml_add(context, output, bias);
    return output;
}

ggml_tensor * build_attention(
        ggml_context * context,
        const AttentionWeights & weights,
        ggml_tensor * query_input,
        ggml_tensor * key_value_input,
        ggml_tensor * mask,
        int64_t query_sequence,
        int64_t key_value_sequence,
        int heads,
        int head_dim) {
    const int64_t width = static_cast<int64_t>(heads) * head_dim;
    ggml_tensor * query = linear(context, weights.q, query_input);
    ggml_tensor * key = linear(context, weights.k, key_value_input);
    ggml_tensor * value = linear(context, weights.v, key_value_input);

    ggml_tensor * query_heads = ggml_reshape_3d(context, query, head_dim, heads, query_sequence);
    ggml_tensor * key_heads = ggml_reshape_3d(context, key, head_dim, heads, key_value_sequence);
    ggml_tensor * value_heads = ggml_reshape_3d(context, value, head_dim, heads, key_value_sequence);
    ggml_tensor * query_permuted = ggml_cont(
        context, ggml_permute(context, query_heads, 0, 2, 1, 3));
    ggml_tensor * key_permuted = ggml_cont(
        context, ggml_permute(context, key_heads, 0, 2, 1, 3));
    ggml_tensor * value_permuted = ggml_cont(
        context, ggml_permute(context, value_heads, 1, 2, 0, 3));

    ggml_tensor * scores = ggml_mul_mat(context, key_permuted, query_permuted);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    ggml_tensor * probabilities = ggml_soft_max_ext(
        context, scores, mask, 1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f);
    ggml_tensor * attended = ggml_mul_mat(context, value_permuted, probabilities);
    ggml_mul_mat_set_prec(attended, GGML_PREC_F32);
    ggml_tensor * merged = ggml_reshape_2d(
        context,
        ggml_cont(context, ggml_permute(context, attended, 0, 2, 1, 3)),
        width, query_sequence);
    return linear(context, weights.out, merged);
}

ggml_tensor * build_vl_block(
        ggml_context * context,
        const VlBlockWeights & weights,
        ggml_tensor * input,
        int64_t sequence,
        int heads,
        int head_dim,
        float epsilon) {
    ggml_tensor * normalized = layer_norm(
        context, input, epsilon, weights.attn_norm_weight, weights.attn_norm_bias);
    ggml_tensor * attention = build_attention(
        context, weights.attention, normalized, normalized, nullptr,
        sequence, sequence, heads, head_dim);
    ggml_tensor * residual = ggml_add(context, input, attention);
    ggml_tensor * ffn_normalized = layer_norm(
        context, residual, epsilon, weights.ffn_norm_weight, weights.ffn_norm_bias);
    ggml_tensor * hidden = ggml_gelu(context, linear(context, weights.ffn_up, ffn_normalized));
    return ggml_add(context, residual, linear(context, weights.ffn_down, hidden));
}

ggml_tensor * build_ada_norm(
        ggml_context * context,
        ggml_tensor * input,
        ggml_tensor * time_embedding,
        const LinearWeights & modulation,
        int64_t width,
        float epsilon) {
    ggml_tensor * values = linear(context, modulation, ggml_silu(context, time_embedding));
    const size_t row_bytes = static_cast<size_t>(width) * sizeof(float);
    ggml_tensor * scale = ggml_view_1d(context, values, width, 0);
    ggml_tensor * shift = ggml_view_1d(context, values, width, row_bytes);
    ggml_tensor * normalized = layer_norm(context, input, epsilon);
    return ggml_add(
        context,
        ggml_mul(context, normalized, ggml_scale_bias(context, scale, 1.0f, 1.0f)),
        shift);
}

ggml_tensor * build_dit_block(
        ggml_context * context,
        const DitBlockWeights & weights,
        ggml_tensor * input,
        ggml_tensor * backbone,
        ggml_tensor * time_embedding,
        ggml_tensor * cross_mask,
        int64_t sequence,
        int64_t backbone_sequence,
        int heads,
        int head_dim,
        float epsilon,
        bool self_attention) {
    const int64_t width = static_cast<int64_t>(heads) * head_dim;
    ggml_tensor * normalized = build_ada_norm(
        context, input, time_embedding, weights.ada_norm, width, epsilon);
    ggml_tensor * attention = build_attention(
        context, weights.attention, normalized,
        self_attention ? normalized : backbone,
        self_attention ? nullptr : cross_mask,
        sequence, self_attention ? sequence : backbone_sequence,
        heads, head_dim);
    ggml_tensor * residual = ggml_add(context, input, attention);
    ggml_tensor * ffn_normalized = layer_norm(context, residual, epsilon);
    ggml_tensor * hidden = ggml_gelu(context, linear(context, weights.ffn_up, ffn_normalized));
    return ggml_add(context, residual, linear(context, weights.ffn_down, hidden));
}

void mark_trace(std::vector<TraceTensor> & trace, const std::string & name,
                ggml_tensor * tensor, bool enabled) {
    if (!enabled) return;
    ggml_set_output(tensor);
    trace.push_back({name, tensor});
}

bool dump_host_tensor(const std::string & directory, const std::string & name,
                      const std::vector<float> & values,
                      const std::vector<int64_t> & shape) {
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        std::fprintf(stderr, "vla(groot_n1): cannot create dump directory %s: %s\n",
                     directory.c_str(), error.message().c_str());
        return false;
    }
    const std::string base = directory + "/" + name;
    std::ofstream data(base + ".f32", std::ios::binary);
    if (!data) return false;
    data.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    std::ofstream shape_file(base + ".shape.txt");
    if (!shape_file) return false;
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index) shape_file << ' ';
        shape_file << shape[index];
    }
    shape_file << '\n';
    return data.good() && shape_file.good();
}

bool dump_graph_tensor(const std::string & directory, const std::string & name,
                       ggml_tensor * tensor) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    std::vector<int64_t> shape;
    if (tensor->ne[3] > 1) shape.push_back(tensor->ne[3]);
    if (tensor->ne[2] > 1) shape.push_back(tensor->ne[2]);
    if (tensor->ne[1] > 1) shape.push_back(tensor->ne[1]);
    shape.push_back(tensor->ne[0]);
    return dump_host_tensor(directory, name, values, shape);
}

bool trace_vision_tensor(ggml_tensor * tensor, bool ask, void * user_data) {
    if (!tensor || !user_data || tensor->type != GGML_TYPE_F32) return false;
    const std::string name = tensor->name;
    const bool selected = name == "inp_pos_emb" || name.rfind("layer_out-", 0) == 0;
    if (!selected) return false;
    if (ask) return true;
    VisionTrace & trace = *static_cast<VisionTrace *>(user_data);
    const int index = trace.counts[name]++;
    char output_name[128];
    std::snprintf(output_name, sizeof(output_name), "vision_%s_image_%02d",
                  name.c_str(), index);
    dump_graph_tensor(trace.directory, output_name, tensor);
    return true;
}

std::vector<float> timestep_embedding(int timestep) {
    constexpr int dimension = 256;
    constexpr int half = dimension / 2;
    std::vector<float> result(dimension);
    for (int index = 0; index < half; ++index) {
        const double exponent = -std::log(10000.0) * index / static_cast<double>(half - 1);
        const double value = timestep * std::exp(exponent);
        result[index] = static_cast<float>(std::cos(value));
        result[half + index] = static_cast<float>(std::sin(value));
    }
    return result;
}

std::vector<float> action_time_encoding(int timestep, int64_t horizon, int64_t width) {
    const int64_t half = width / 2;
    std::vector<float> result(static_cast<size_t>(horizon * width));
    for (int64_t token = 0; token < horizon; ++token) {
        float * row = result.data() + static_cast<size_t>(token * width);
        for (int64_t index = 0; index < half; ++index) {
            const double exponent = -index * std::log(10000.0) / static_cast<double>(half);
            const double value = timestep * std::exp(exponent);
            row[index] = static_cast<float>(std::sin(value));
            row[half + index] = static_cast<float>(std::cos(value));
        }
    }
    return result;
}

bool run_preprocess(
        GrootN1ModelArch & model,
        const Inputs & input,
        const std::vector<float> & normalized_state,
        std::vector<float> & backbone_output,
        std::vector<float> & state_output,
        const std::string & dump_directory) {
    const int64_t sequence = input.backbone_seq_len;
    ggml_init_params params{64u * 1024u * 1024u, nullptr, true};
    ggml_context * context = ggml_init(params);
    if (!context) return false;

    ggml_tensor * backbone_input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, model.backbone_dim, sequence);
    ggml_tensor * state_input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, model.cfg.max_state_dim, 1);
    ggml_set_input(backbone_input);
    ggml_set_input(state_input);

    const bool dump_enabled = !dump_directory.empty();
    std::vector<TraceTensor> trace;
    mark_trace(trace, "backbone_input", backbone_input, dump_enabled);
    ggml_tensor * backbone = layer_norm(
        context, backbone_input, model.layer_norm_eps,
        model.vl_norm_weight, model.vl_norm_bias);
    mark_trace(trace, "backbone_vlln", backbone, dump_enabled);
    for (int layer = 0; layer < model.vl_layers_count; ++layer) {
        backbone = build_vl_block(
            context, model.vl_blocks[static_cast<size_t>(layer)], backbone,
            sequence, model.vl_heads, model.vl_head_dim, model.layer_norm_eps);
        char name[64];
        std::snprintf(name, sizeof(name), "backbone_vl_block_%02d", layer);
        mark_trace(trace, name, backbone, dump_enabled);
    }

    ggml_tensor * state_hidden = ggml_relu(
        context, linear(context, model.state_encoder_0, state_input));
    ggml_tensor * state_features = linear(context, model.state_encoder_2, state_hidden);
    mark_trace(trace, "state_features", state_features, dump_enabled);
    ggml_set_output(backbone);
    ggml_set_output(state_features);

    ggml_cgraph * graph = ggml_new_graph_custom(context, 32768, false);
    ggml_build_forward_expand(graph, backbone);
    ggml_build_forward_expand(graph, state_features);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(model.backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        std::fprintf(stderr, "vla(groot_n1): preprocess graph allocation failed\n");
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(context);
        return false;
    }
    ggml_backend_tensor_set(
        backbone_input, input.precomputed_backbone_features, 0, ggml_nbytes(backbone_input));
    ggml_backend_tensor_set(state_input, normalized_state.data(), 0, ggml_nbytes(state_input));
    const ggml_status status = ggml_backend_graph_compute(model.backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(groot_n1): preprocess graph failed (%d)\n", status);
        ggml_gallocr_free(allocator);
        ggml_free(context);
        return false;
    }

    backbone_output.resize(static_cast<size_t>(sequence * model.backbone_dim));
    state_output.resize(static_cast<size_t>(model.input_dim));
    ggml_backend_tensor_get(backbone, backbone_output.data(), 0,
                            backbone_output.size() * sizeof(float));
    ggml_backend_tensor_get(state_features, state_output.data(), 0,
                            state_output.size() * sizeof(float));
    for (const TraceTensor & item : trace) {
        if (!dump_graph_tensor(dump_directory, item.name, item.tensor)) {
            std::fprintf(stderr, "vla(groot_n1): failed to dump %s\n", item.name.c_str());
        }
    }
    ggml_gallocr_free(allocator);
    ggml_free(context);
    return true;
}

bool run_denoise(
        GrootN1ModelArch & model,
        const Inputs & input,
        const std::vector<float> & backbone,
        const std::vector<float> & state_features,
        std::vector<float> & actions,
        const std::string & dump_directory) {
    const int64_t backbone_sequence = input.backbone_seq_len;
    const int64_t horizon = model.cfg.n_suffix;
    const int64_t action_dim = model.cfg.max_action_dim;
    const int64_t sequence = horizon + 1;

    ggml_init_params params{96u * 1024u * 1024u, nullptr, true};
    ggml_context * context = ggml_init(params);
    if (!context) return false;

    ggml_tensor * backbone_input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, model.backbone_dim, backbone_sequence);
    ggml_tensor * state_input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, model.input_dim, 1);
    ggml_tensor * action_input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, action_dim, horizon);
    ggml_tensor * action_time_input = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, model.input_dim, horizon);
    ggml_tensor * timestep_input = ggml_new_tensor_2d(context, GGML_TYPE_F32, 256, 1);
    ggml_tensor * text_mask = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, backbone_sequence, sequence);
    ggml_tensor * image_mask = ggml_new_tensor_2d(
        context, GGML_TYPE_F32, backbone_sequence, sequence);
    for (ggml_tensor * tensor : {
             backbone_input, state_input, action_input, action_time_input,
             timestep_input, text_mask, image_mask}) {
        ggml_set_input(tensor);
    }

    const bool dump_enabled = !dump_directory.empty();
    std::vector<TraceTensor> trace;
    ggml_tensor * timestep_hidden = ggml_silu(
        context, linear(context, model.time_0, timestep_input));
    ggml_tensor * timestep_features = linear(context, model.time_2, timestep_hidden);
    mark_trace(trace, "timestep_features", timestep_features, dump_enabled);

    ggml_tensor * action_w1 = linear(context, model.action_encoder_w1, action_input);
    mark_trace(trace, "action_w1", action_w1, dump_enabled);
    ggml_tensor * action_joined = ggml_concat(context, action_w1, action_time_input, 0);
    ggml_tensor * action_w2 = linear(context, model.action_encoder_w2, action_joined);
    mark_trace(trace, "action_w2", action_w2, dump_enabled);
    ggml_tensor * action_features = linear(
        context, model.action_encoder_w3, ggml_silu(context, action_w2));
    ggml_tensor * position = ggml_view_2d(
        context, model.action_position, model.input_dim, horizon,
        model.action_position->nb[1], 0);
    action_features = ggml_add(context, action_features, position);
    mark_trace(trace, "action_features", action_features, dump_enabled);

    ggml_tensor * hidden = ggml_concat(context, state_input, action_features, 1);
    mark_trace(trace, "state_action_features", hidden, dump_enabled);
    for (int layer = 0; layer < model.cfg.n_layers; ++layer) {
        const bool self_attention = layer % 2 == 1;
        ggml_tensor * mask = nullptr;
        if (!self_attention) {
            const int period = 2 * model.attend_text_every_n_blocks;
            mask = layer % period == 0 ? text_mask : image_mask;
        }
        hidden = build_dit_block(
            context, model.dit_blocks[static_cast<size_t>(layer)], hidden,
            backbone_input, timestep_features, mask, sequence, backbone_sequence,
            model.dit_heads, model.dit_head_dim, model.layer_norm_eps, self_attention);
        char name[64];
        std::snprintf(name, sizeof(name), "dit_block_%02d", layer);
        mark_trace(trace, name, hidden, dump_enabled);
    }

    ggml_tensor * output_modulation = linear(
        context, model.output_norm, ggml_silu(context, timestep_features));
    const size_t output_row_bytes = static_cast<size_t>(model.input_dim) * sizeof(float);
    ggml_tensor * output_shift = ggml_view_1d(
        context, output_modulation, model.input_dim, 0);
    ggml_tensor * output_scale = ggml_view_1d(
        context, output_modulation, model.input_dim, output_row_bytes);
    ggml_tensor * normalized_output = layer_norm(
        context, hidden, model.output_norm_eps);
    normalized_output = ggml_add(
        context,
        ggml_mul(context, normalized_output,
                 ggml_scale_bias(context, output_scale, 1.0f, 1.0f)),
        output_shift);
    ggml_tensor * model_output = linear(context, model.output_projection, normalized_output);
    mark_trace(trace, "dit_model_output", model_output, dump_enabled);

    ggml_tensor * decoded = linear(
        context, model.action_decoder_2,
        ggml_relu(context, linear(context, model.action_decoder_0, model_output)));
    ggml_tensor * velocity = ggml_view_2d(
        context, decoded, action_dim, horizon, decoded->nb[1], decoded->nb[1]);
    ggml_set_output(velocity);
    mark_trace(trace, "pred_velocity", velocity, dump_enabled);

    ggml_cgraph * graph = ggml_new_graph_custom(context, 65536, false);
    ggml_build_forward_expand(graph, velocity);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(model.backend));
    if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
        std::fprintf(stderr, "vla(groot_n1): denoise graph allocation failed\n");
        if (allocator) ggml_gallocr_free(allocator);
        ggml_free(context);
        return false;
    }

    std::vector<float> text_mask_values(static_cast<size_t>(backbone_sequence * sequence));
    std::vector<float> image_mask_values(static_cast<size_t>(backbone_sequence * sequence));
    for (int64_t query = 0; query < sequence; ++query) {
        for (int64_t key = 0; key < backbone_sequence; ++key) {
            const bool valid = input.backbone_attention_mask[key] != 0;
            const bool is_image = input.backbone_image_mask[key] != 0;
            const size_t index = static_cast<size_t>(query * backbone_sequence + key);
            text_mask_values[index] = valid && !is_image ? 0.0f : -INFINITY;
            image_mask_values[index] = valid && is_image ? 0.0f : -INFINITY;
        }
    }
    std::vector<float> velocity_values(static_cast<size_t>(horizon * action_dim));
    const float dt = 1.0f / static_cast<float>(model.cfg.num_steps);
    for (int step = 0; step < model.cfg.num_steps; ++step) {
        const int discrete_timestep = static_cast<int>(
            step / static_cast<float>(model.cfg.num_steps) *
            static_cast<float>(model.num_timestep_buckets));
        const std::vector<float> time_values = timestep_embedding(discrete_timestep);
        const std::vector<float> action_time_values = action_time_encoding(
            discrete_timestep, horizon, model.input_dim);
        ggml_backend_tensor_set(
            backbone_input, backbone.data(), 0, ggml_nbytes(backbone_input));
        ggml_backend_tensor_set(
            state_input, state_features.data(), 0, ggml_nbytes(state_input));
        ggml_backend_tensor_set(
            text_mask, text_mask_values.data(), 0, ggml_nbytes(text_mask));
        ggml_backend_tensor_set(
            image_mask, image_mask_values.data(), 0, ggml_nbytes(image_mask));
        ggml_backend_tensor_set(action_input, actions.data(), 0, ggml_nbytes(action_input));
        ggml_backend_tensor_set(action_time_input, action_time_values.data(), 0,
                                ggml_nbytes(action_time_input));
        ggml_backend_tensor_set(timestep_input, time_values.data(), 0, ggml_nbytes(timestep_input));
        const ggml_status status = ggml_backend_graph_compute(model.backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(groot_n1): denoise step %d failed (%d)\n", step, status);
            ggml_gallocr_free(allocator);
            ggml_free(context);
            return false;
        }
        ggml_backend_tensor_get(velocity, velocity_values.data(), 0,
                                velocity_values.size() * sizeof(float));
        if (dump_enabled) {
            char prefix[32];
            std::snprintf(prefix, sizeof(prefix), "step_%02d_", step);
            dump_host_tensor(dump_directory, std::string(prefix) + "action_input", actions,
                             {horizon, action_dim});
            dump_host_tensor(dump_directory, std::string(prefix) + "action_time_encoding",
                             action_time_values, {horizon, model.input_dim});
            for (const TraceTensor & item : trace) {
                dump_graph_tensor(dump_directory, std::string(prefix) + item.name, item.tensor);
            }
        }
        for (size_t index = 0; index < actions.size(); ++index) {
            actions[index] += dt * velocity_values[index];
        }
        if (dump_enabled) {
            char name[64];
            std::snprintf(name, sizeof(name), "step_%02d_action_output", step);
            dump_host_tensor(dump_directory, name, actions, {horizon, action_dim});
        }
    }

    ggml_gallocr_free(allocator);
    ggml_free(context);
    return true;
}

bool load_config(GgufReader & reader, GrootN1ModelArch & model) {
    const char * required[] = {
        "groot_n1.backbone_embedding_dim", "groot_n1.hidden_size",
        "groot_n1.input_embedding_dim", "groot_n1.max_state_dim",
        "groot_n1.max_action_dim", "groot_n1.real_state_dim",
        "groot_n1.real_action_dim", "groot_n1.action_horizon",
        "groot_n1.real_action_horizon", "groot_n1.max_seq_len",
        "groot_n1.vl_n_layers", "groot_n1.vl_n_heads", "groot_n1.vl_head_dim",
        "groot_n1.dit_n_layers", "groot_n1.dit_n_heads", "groot_n1.dit_head_dim",
        "groot_n1.num_inference_timesteps", "groot_n1.num_timestep_buckets",
        "groot_n1.attend_text_every_n_blocks",
    };
    for (const char * key : required) {
        if (!reader.has_key(key)) {
            std::fprintf(stderr, "vla(groot_n1): missing GGUF key %s\n", key);
            return false;
        }
    }

    model.backbone_dim = reader.u32("groot_n1.backbone_embedding_dim");
    model.head_hidden = reader.u32("groot_n1.hidden_size");
    model.input_dim = reader.u32("groot_n1.input_embedding_dim");
    model.max_seq_len = reader.u32("groot_n1.max_seq_len");
    model.real_action_horizon = reader.u32("groot_n1.real_action_horizon");
    model.vl_layers_count = reader.u32("groot_n1.vl_n_layers");
    model.vl_heads = reader.u32("groot_n1.vl_n_heads");
    model.vl_head_dim = reader.u32("groot_n1.vl_head_dim");
    model.dit_heads = reader.u32("groot_n1.dit_n_heads");
    model.dit_head_dim = reader.u32("groot_n1.dit_head_dim");
    model.num_timestep_buckets = reader.u32("groot_n1.num_timestep_buckets");
    model.attend_text_every_n_blocks = reader.u32("groot_n1.attend_text_every_n_blocks");
    model.layer_norm_eps = reader.has_key("groot_n1.layer_norm_eps")
        ? reader.f32("groot_n1.layer_norm_eps") : 1e-5f;
    model.output_norm_eps = reader.has_key("groot_n1.output_norm_eps")
        ? reader.f32("groot_n1.output_norm_eps") : 1e-6f;
    model.clip_outliers = reader.has_key("groot_n1.clip_outliers")
        ? reader.boolean("groot_n1.clip_outliers") : true;

    Config & config = model.cfg;
    config = Config{};
    config.hidden = model.backbone_dim;
    config.expert_h = model.input_dim;
    config.n_q_heads = model.dit_heads;
    config.n_kv_heads = model.dit_heads;
    config.head_dim = model.dit_head_dim;
    config.q_full_dim = model.input_dim;
    config.kv_full_dim = model.input_dim;
    config.n_layers = reader.u32("groot_n1.dit_n_layers");
    config.max_state_dim = reader.u32("groot_n1.max_state_dim");
    config.max_action_dim = reader.u32("groot_n1.max_action_dim");
    config.real_state_dim = reader.u32("groot_n1.real_state_dim");
    config.real_action_dim = reader.u32("groot_n1.real_action_dim");
    config.n_suffix = reader.u32("groot_n1.action_horizon");
    config.num_steps = static_cast<int>(reader.u32("groot_n1.num_inference_timesteps"));
    config.n_lang = model.max_seq_len;
    config.n_img = 0;
    config.n_state = 1;
    config.n_prefix = 0;
    config.n_full = config.n_suffix + 1;
    config.norm_eps = 1e-8f;
    config.rms_eps = model.layer_norm_eps;
    return model.backbone_dim == static_cast<int64_t>(model.vl_heads) * model.vl_head_dim &&
           model.input_dim == static_cast<int64_t>(model.dit_heads) * model.dit_head_dim;
}

bool load_statistics(GgufReader & reader, GrootN1ModelArch & model) {
    auto load = [&](const char * name, int64_t expected, std::vector<float> & destination) {
        const ggml_tensor * tensor = reader.meta(name);
        if (!tensor || tensor->type != GGML_TYPE_F32 || tensor->ne[0] != expected) {
            std::fprintf(stderr, "vla(groot_n1): invalid normalization tensor %s\n", name);
            return false;
        }
        destination.resize(static_cast<size_t>(expected));
        return reader.read_raw(name, destination.data());
    };
    return load("norm.state.q01", model.cfg.real_state_dim, model.state_q01) &&
           load("norm.state.q99", model.cfg.real_state_dim, model.state_q99) &&
           load("norm.action.q01", model.cfg.real_action_dim, model.action_q01) &&
           load("norm.action.q99", model.cfg.real_action_dim, model.action_q99);
}

bool load_backbone(
        GrootN1ModelArch & model,
        const std::string & backbone_path,
        const std::string & mmproj_path) {
    if (backbone_path.empty() && mmproj_path.empty()) return true;
    if (backbone_path.empty() || mmproj_path.empty()) {
        std::fprintf(stderr,
                     "vla(groot_n1): integrated backbone requires both text GGUF and mmproj\n");
        return false;
    }

    static std::once_flag backend_once;
    std::call_once(backend_once, llama_backend_init);
    const char * flash_attention_env = std::getenv("VLA_GROOT_FLASH_ATTN");
    const bool use_flash_attention = flash_attention_env &&
        std::strcmp(flash_attention_env, "0") != 0 &&
        std::strcmp(flash_attention_env, "false") != 0 &&
        std::strcmp(flash_attention_env, "FALSE") != 0;
    const llama_flash_attn_type flash_attention_type = use_flash_attention
        ? LLAMA_FLASH_ATTN_TYPE_AUTO
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = std::getenv("VLA_GROOT_CPU") ? 0 : -1;
    model.backbone_model = llama_model_load_from_file(backbone_path.c_str(), model_params);
    if (!model.backbone_model) {
        std::fprintf(stderr, "vla(groot_n1): failed to load backbone %s\n",
                     backbone_path.c_str());
        return false;
    }
    if (llama_model_n_layer(model.backbone_model) != 16 ||
        llama_model_n_embd_out(model.backbone_model) != model.backbone_dim) {
        std::fprintf(stderr,
                     "vla(groot_n1): backbone must expose 16 layers at width %lld "
                     "(got layers=%d width=%d)\n",
                     static_cast<long long>(model.backbone_dim),
                     llama_model_n_layer(model.backbone_model),
                     llama_model_n_embd_out(model.backbone_model));
        return false;
    }

    model.backbone_n_batch = static_cast<int>(model.max_seq_len);
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = static_cast<uint32_t>(model.max_seq_len);
    context_params.n_batch = static_cast<uint32_t>(model.backbone_n_batch);
    context_params.n_ubatch = static_cast<uint32_t>(model.backbone_n_batch);
    context_params.n_seq_max = 1;
    context_params.n_threads = model.n_threads;
    context_params.n_threads_batch = model.n_threads;
    context_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    context_params.flash_attn_type = flash_attention_type;
    context_params.embeddings = true;
    model.backbone_context = llama_init_from_model(model.backbone_model, context_params);
    if (!model.backbone_context) {
        std::fprintf(stderr, "vla(groot_n1): failed to initialize backbone context\n");
        return false;
    }

    mtmd_context_params mmproj_params = mtmd_context_params_default();
    mmproj_params.use_gpu = !std::getenv("VLA_GROOT_CPU");
    mmproj_params.n_threads = model.n_threads;
    mmproj_params.flash_attn_type = flash_attention_type;
    mmproj_params.warmup = false;
    if (const char * dump_env = std::getenv("VLA_GROOT_DUMP_DIR")) {
        if (*dump_env) {
            model.vision_trace.directory = dump_env;
            mmproj_params.cb_eval = trace_vision_tensor;
            mmproj_params.cb_eval_user_data = &model.vision_trace;
        }
    }
    model.mmproj_context = mtmd_init_from_file(
        mmproj_path.c_str(), model.backbone_model, mmproj_params);
    if (!model.mmproj_context || !mtmd_support_vision(model.mmproj_context)) {
        std::fprintf(stderr, "vla(groot_n1): failed to load Qwen3-VL mmproj %s\n",
                     mmproj_path.c_str());
        return false;
    }
    std::printf(
        "vla(groot_n1): integrated Qwen3-VL backbone loaded "
        "(layers=%d, input=%d, output=%d, batch=%d, flash_attn=%s)\n",
        llama_model_n_layer(model.backbone_model),
        llama_model_n_embd_inp(model.backbone_model),
        llama_model_n_embd_out(model.backbone_model),
        model.backbone_n_batch,
        use_flash_attention ? "auto" : "disabled");
    return true;
}

std::string format_backbone_prompt(
        const GrootN1ModelArch & model,
        const Inputs & input) {
    std::string content;
    for (int image = 0; image < input.n_images; ++image) {
        content += mtmd_default_marker();
    }
    content += input.language_text;
    const llama_chat_message message{"user", content.c_str()};
    const char * chat_template = llama_model_chat_template(model.backbone_model, nullptr);
    const int32_t required = llama_chat_apply_template(
        chat_template, &message, 1, false, nullptr, 0);
    if (required < 0) return {};
    std::vector<char> buffer(static_cast<size_t>(required) + 1);
    const int32_t written = llama_chat_apply_template(
        chat_template, &message, 1, false, buffer.data(), buffer.size());
    if (written < 0 || written > required) return {};
    return std::string(buffer.data(), static_cast<size_t>(written));
}

bool run_backbone(
        GrootN1ModelArch & model,
        const Inputs & input,
        std::vector<float> & features,
        std::vector<int32_t> & attention_mask,
        std::vector<int32_t> & image_mask) {
    if (!model.backbone_context || !model.mmproj_context ||
        !input.language_text || !*input.language_text ||
        !input.images || input.n_images < 1) {
        std::fprintf(stderr,
                     "vla(groot_n1): raw backbone inference requires images and language_text\n");
        return false;
    }

    std::vector<std::vector<uint8_t>> image_bytes(static_cast<size_t>(input.n_images));
    std::vector<mtmd_bitmap *> owned_bitmaps;
    std::vector<const mtmd_bitmap *> bitmaps;
    owned_bitmaps.reserve(static_cast<size_t>(input.n_images));
    bitmaps.reserve(static_cast<size_t>(input.n_images));
    auto free_bitmaps = [&]() {
        for (mtmd_bitmap * bitmap : owned_bitmaps) mtmd_bitmap_free(bitmap);
    };
    for (int index = 0; index < input.n_images; ++index) {
        const ImageView & image = input.images[index];
        if (!image.data || image.w < 1 || image.h < 1) {
            free_bitmaps();
            return false;
        }
        const size_t count = static_cast<size_t>(image.w) * image.h * 3;
        const uint8_t * bytes = nullptr;
        if (image.format == PixelFormat::U8) {
            bytes = static_cast<const uint8_t *>(image.data);
        } else {
            std::vector<uint8_t> & converted = image_bytes[static_cast<size_t>(index)];
            converted.resize(count);
            const float * source = static_cast<const float *>(image.data);
            for (size_t element = 0; element < count; ++element) {
                converted[element] = static_cast<uint8_t>(std::lround(
                    std::max(0.0f, std::min(1.0f, source[element])) * 255.0f));
            }
            bytes = converted.data();
        }
        mtmd_bitmap * bitmap = mtmd_bitmap_init(
            static_cast<uint32_t>(image.w), static_cast<uint32_t>(image.h), bytes);
        if (!bitmap) {
            free_bitmaps();
            return false;
        }
        owned_bitmaps.push_back(bitmap);
        bitmaps.push_back(bitmap);
    }

    const std::string prompt = format_backbone_prompt(model, input);
    if (prompt.empty()) {
        free_bitmaps();
        std::fprintf(stderr, "vla(groot_n1): failed to format Qwen3-VL prompt\n");
        return false;
    }
    mtmd_input_text text{prompt.c_str(), false, true};
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const int32_t tokenize_status = mtmd_tokenize(
        model.mmproj_context, chunks, &text, bitmaps.data(), bitmaps.size());
    free_bitmaps();
    if (tokenize_status != 0) {
        std::fprintf(stderr, "vla(groot_n1): mtmd tokenization failed (%d)\n",
                     tokenize_status);
        mtmd_input_chunks_free(chunks);
        return false;
    }

    const size_t total_tokens = mtmd_helper_get_n_tokens(chunks);
    if (total_tokens < 1 || total_tokens > static_cast<size_t>(model.max_seq_len)) {
        std::fprintf(stderr, "vla(groot_n1): backbone sequence %zu exceeds limit %lld\n",
                     total_tokens, static_cast<long long>(model.max_seq_len));
        mtmd_input_chunks_free(chunks);
        return false;
    }
    features.clear();
    attention_mask.clear();
    image_mask.clear();
    features.reserve(total_tokens * static_cast<size_t>(model.backbone_dim));
    attention_mask.reserve(total_tokens);
    image_mask.reserve(total_tokens);
    llama_memory_clear(llama_get_memory(model.backbone_context), true);

    llama_pos n_past = 0;
    int vision_index = 0;
    for (size_t index = 0; index < mtmd_input_chunks_size(chunks); ++index) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, index);
        const size_t chunk_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        llama_pos new_n_past = n_past;
        const int32_t status = mtmd_helper_eval_chunk_single(
            model.mmproj_context, model.backbone_context, chunk,
            n_past, 0, model.backbone_n_batch, false, &new_n_past);
        if (status != 0) {
            std::fprintf(stderr, "vla(groot_n1): backbone chunk %zu failed (%d)\n",
                         index, status);
            mtmd_input_chunks_free(chunks);
            return false;
        }
        llama_synchronize(model.backbone_context);
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const char * dump_env = std::getenv("VLA_GROOT_DUMP_DIR");
            if (dump_env && *dump_env) {
                const int64_t vision_width = llama_model_n_embd_inp(model.backbone_model);
                const float * vision_embeddings = mtmd_get_output_embd(model.mmproj_context);
                std::vector<float> vision_values(
                    vision_embeddings,
                    vision_embeddings + chunk_tokens * static_cast<size_t>(vision_width));
                char name[64];
                std::snprintf(name, sizeof(name), "backbone_vision_%02d", vision_index++);
                dump_host_tensor(
                    dump_env, name, vision_values,
                    {static_cast<int64_t>(chunk_tokens), vision_width});
            }
        }
        const float * embeddings = llama_get_embeddings(model.backbone_context);
        if (!embeddings) {
            std::fprintf(stderr, "vla(groot_n1): backbone chunk %zu has no embeddings\n",
                         index);
            mtmd_input_chunks_free(chunks);
            return false;
        }
        features.insert(
            features.end(), embeddings,
            embeddings + chunk_tokens * static_cast<size_t>(model.backbone_dim));
        const bool is_image =
            mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE;
        attention_mask.insert(attention_mask.end(), chunk_tokens, 1);
        image_mask.insert(image_mask.end(), chunk_tokens, is_image ? 1 : 0);
        n_past = new_n_past;
    }
    mtmd_input_chunks_free(chunks);
    return features.size() == total_tokens * static_cast<size_t>(model.backbone_dim) &&
           attention_mask.size() == total_tokens && image_mask.size() == total_tokens;
}

} // namespace

GrootN1ModelArch::~GrootN1ModelArch() {
    if (mmproj_context) mtmd_free(mmproj_context);
    if (backbone_context) llama_free(backbone_context);
    if (backbone_model) llama_model_free(backbone_model);
    if (weight_buffer) ggml_backend_buffer_free(weight_buffer);
    if (weight_context) ggml_free(weight_context);
    if (backend) ggml_backend_free(backend);
}

std::unique_ptr<ModelArchBase> groot_n1_create(
        const std::string & mmproj_path,
        const std::string & checkpoint_path,
        const std::string & config_path) {
    if (!ends_with(checkpoint_path, ".gguf")) {
        std::fprintf(stderr,
                     "vla(groot_n1): checkpoint must be produced by "
                     "scripts/convert_groot_n1_to_gguf.py\n");
        return nullptr;
    }

    auto model = std::make_unique<GrootN1ModelArch>();
    model->matmul_type = matmul_type_from_env();
    GgufReader reader;
    if (!reader.open(checkpoint_path)) return nullptr;
    if (!reader.has_key("groot_n1.architecture") ||
        reader.str("groot_n1.architecture") != "groot_n1") {
        std::fprintf(stderr, "vla(groot_n1): checkpoint architecture metadata is invalid\n");
        return nullptr;
    }
    if (!load_config(reader, *model) || !load_statistics(reader, *model)) return nullptr;

#ifdef GGML_USE_CUDA
    if (!std::getenv("VLA_GROOT_CPU")) {
        model->backend = ggml_backend_cuda_init(0);
        if (model->backend) {
            model->is_cuda = true;
            std::printf("vla(groot_n1): backend = CUDA device 0\n");
        }
    }
#endif
    const unsigned hardware_threads = std::thread::hardware_concurrency();
    model->n_threads = hardware_threads == 0 ? 4 : static_cast<int>(std::min(8u, hardware_threads));
    if (!model->backend) {
        model->backend = ggml_backend_cpu_init();
        if (!model->backend) return nullptr;
        ggml_backend_cpu_set_n_threads(model->backend, model->n_threads);
        std::printf("vla(groot_n1): backend = CPU (%d threads)\n", model->n_threads);
    }
    if (!load_backbone(*model, config_path, mmproj_path)) return nullptr;

    ggml_init_params weight_params{64u * 1024u * 1024u, nullptr, true};
    model->weight_context = ggml_init(weight_params);
    if (!model->weight_context) return nullptr;
    std::vector<ggml_tensor *> weights;

    auto make = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * metadata = reader.meta(name);
        if (!metadata) {
            std::fprintf(stderr, "vla(groot_n1): missing tensor %s\n", name);
            return nullptr;
        }
        ggml_tensor * tensor = ggml_new_tensor(
            model->weight_context, type, GGML_MAX_DIMS, metadata->ne);
        ggml_set_name(tensor, name);
        weights.push_back(tensor);
        return tensor;
    };
    auto make_matrix = [&](const char * name) -> ggml_tensor * {
        const ggml_tensor * metadata = reader.meta(name);
        if (!metadata) return make(name, model->matmul_type);
        ggml_type type = ggml_is_quantized(metadata->type) ? metadata->type : model->matmul_type;
        if (ggml_is_quantized(type) && metadata->ne[0] % ggml_blck_size(type) != 0) {
            type = GGML_TYPE_BF16;
        }
        return make(name, type);
    };
    auto make_f32 = [&](const char * name) { return make(name, GGML_TYPE_F32); };
    auto load_linear = [&](const std::string & prefix, LinearWeights & destination) {
        destination.weight = make_matrix((prefix + ".weight").c_str());
        destination.bias = make_f32((prefix + ".bias").c_str());
        return destination.weight && destination.bias;
    };

    model->vl_norm_weight = make_f32("vl.norm.weight");
    model->vl_norm_bias = make_f32("vl.norm.bias");
    model->vl_blocks.resize(static_cast<size_t>(model->vl_layers_count));
    for (int layer = 0; layer < model->vl_layers_count; ++layer) {
        const std::string prefix = "vl.blk." + std::to_string(layer);
        VlBlockWeights & block = model->vl_blocks[static_cast<size_t>(layer)];
        block.attn_norm_weight = make_f32((prefix + ".attn_norm.weight").c_str());
        block.attn_norm_bias = make_f32((prefix + ".attn_norm.bias").c_str());
        block.ffn_norm_weight = make_f32((prefix + ".ffn_norm.weight").c_str());
        block.ffn_norm_bias = make_f32((prefix + ".ffn_norm.bias").c_str());
        load_linear(prefix + ".attn_q", block.attention.q);
        load_linear(prefix + ".attn_k", block.attention.k);
        load_linear(prefix + ".attn_v", block.attention.v);
        load_linear(prefix + ".attn_out", block.attention.out);
        load_linear(prefix + ".ffn_up", block.ffn_up);
        load_linear(prefix + ".ffn_down", block.ffn_down);
    }

    load_linear("state_encoder.0", model->state_encoder_0);
    load_linear("state_encoder.2", model->state_encoder_2);
    load_linear("action_encoder.w1", model->action_encoder_w1);
    load_linear("action_encoder.w2", model->action_encoder_w2);
    load_linear("action_encoder.w3", model->action_encoder_w3);
    load_linear("action_decoder.0", model->action_decoder_0);
    load_linear("action_decoder.2", model->action_decoder_2);
    model->action_position = make_f32("action_position.weight");
    load_linear("dit.time.0", model->time_0);
    load_linear("dit.time.2", model->time_2);
    load_linear("dit.out_norm", model->output_norm);
    load_linear("dit.out_proj", model->output_projection);

    model->dit_blocks.resize(static_cast<size_t>(model->cfg.n_layers));
    for (int layer = 0; layer < model->cfg.n_layers; ++layer) {
        const std::string prefix = "dit.blk." + std::to_string(layer);
        DitBlockWeights & block = model->dit_blocks[static_cast<size_t>(layer)];
        load_linear(prefix + ".attn_norm", block.ada_norm);
        load_linear(prefix + ".attn_q", block.attention.q);
        load_linear(prefix + ".attn_k", block.attention.k);
        load_linear(prefix + ".attn_v", block.attention.v);
        load_linear(prefix + ".attn_out", block.attention.out);
        load_linear(prefix + ".ffn_up", block.ffn_up);
        load_linear(prefix + ".ffn_down", block.ffn_down);
    }
    for (ggml_tensor * tensor : weights) {
        if (!tensor) return nullptr;
    }

    model->weight_buffer = ggml_backend_alloc_ctx_tensors(
        model->weight_context, model->backend);
    if (!model->weight_buffer) {
        std::fprintf(stderr, "vla(groot_n1): resident weight allocation failed\n");
        return nullptr;
    }
    for (ggml_tensor * tensor : weights) {
        std::vector<uint8_t> bytes = reader.read_convert(tensor->name, tensor->type);
        if (bytes.size() != ggml_nbytes(tensor)) {
            std::fprintf(stderr, "vla(groot_n1): upload size mismatch for %s\n", tensor->name);
            return nullptr;
        }
        ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
    }

    std::printf(
        "vla(groot_n1): loaded %.2f GiB resident weights (%s), VL=%d layers, "
        "DiT=%lld layers, horizon=%lld/%lld, action=%lld/%lld\n",
        ggml_backend_buffer_get_size(model->weight_buffer) / (1024.0 * 1024.0 * 1024.0),
        ggml_type_name(model->matmul_type), model->vl_layers_count,
        static_cast<long long>(model->cfg.n_layers),
        static_cast<long long>(model->real_action_horizon),
        static_cast<long long>(model->cfg.n_suffix),
        static_cast<long long>(model->cfg.real_action_dim),
        static_cast<long long>(model->cfg.max_action_dim));
    return model;
}

std::vector<float> GrootN1ModelArch::predict(const Inputs & input) {
    using Clock = std::chrono::high_resolution_clock;
    const auto started = Clock::now();
    stats = Stats{};
    Inputs resolved_input = input;
    std::vector<float> raw_backbone_features;
    std::vector<int32_t> raw_backbone_attention_mask;
    std::vector<int32_t> raw_backbone_image_mask;
    if (!resolved_input.precomputed_backbone_features) {
        const auto vision_started = Clock::now();
        if (!run_backbone(*this, input, raw_backbone_features,
                          raw_backbone_attention_mask, raw_backbone_image_mask)) {
            return {};
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(
            Clock::now() - vision_started).count();
        resolved_input.precomputed_backbone_features = raw_backbone_features.data();
        resolved_input.backbone_seq_len = static_cast<int>(raw_backbone_attention_mask.size());
        resolved_input.backbone_attention_mask = raw_backbone_attention_mask.data();
        resolved_input.backbone_attention_mask_n = resolved_input.backbone_seq_len;
        resolved_input.backbone_image_mask = raw_backbone_image_mask.data();
        resolved_input.backbone_image_mask_n = resolved_input.backbone_seq_len;
        const char * dump_env = std::getenv("VLA_GROOT_DUMP_DIR");
        if (dump_env && *dump_env) {
            std::vector<float> attention_mask_dump(
                raw_backbone_attention_mask.begin(), raw_backbone_attention_mask.end());
            std::vector<float> image_mask_dump(
                raw_backbone_image_mask.begin(), raw_backbone_image_mask.end());
            dump_host_tensor(
                dump_env, "backbone_attention_mask", attention_mask_dump,
                {resolved_input.backbone_seq_len});
            dump_host_tensor(
                dump_env, "backbone_image_mask", image_mask_dump,
                {resolved_input.backbone_seq_len});
        }
    }
    if (resolved_input.backbone_seq_len < 1 ||
        resolved_input.backbone_seq_len > max_seq_len ||
        resolved_input.backbone_attention_mask_n != resolved_input.backbone_seq_len ||
        resolved_input.backbone_image_mask_n != resolved_input.backbone_seq_len ||
        !resolved_input.backbone_attention_mask || !resolved_input.backbone_image_mask) {
        std::fprintf(stderr, "vla(groot_n1): invalid backbone sequence or masks\n");
        return {};
    }
    if (!input.state) {
        std::fprintf(stderr, "vla(groot_n1): state is required\n");
        return {};
    }

    std::vector<float> normalized_state(static_cast<size_t>(cfg.max_state_dim), 0.0f);
    for (int64_t index = 0; index < cfg.real_state_dim; ++index) {
        const float denominator = state_q99[static_cast<size_t>(index)] -
                                  state_q01[static_cast<size_t>(index)];
        float value = 2.0f * (input.state[index] - state_q01[static_cast<size_t>(index)]) /
                      denominator - 1.0f;
        if (clip_outliers) value = std::max(-1.0f, std::min(1.0f, value));
        normalized_state[static_cast<size_t>(index)] = value;
    }

    const char * dump_env = std::getenv("VLA_GROOT_DUMP_DIR");
    const std::string dump_directory = dump_env ? dump_env : "";
    std::vector<float> backbone;
    std::vector<float> state_features;
    const auto prefill_started = Clock::now();
    if (!run_preprocess(*this, resolved_input, normalized_state, backbone, state_features,
                        dump_directory)) {
        return {};
    }
    stats.ms_prefill = std::chrono::duration<float, std::milli>(
        Clock::now() - prefill_started).count();

    std::vector<float> actions(static_cast<size_t>(cfg.n_suffix * cfg.max_action_dim));
    if (input.noise) {
        std::memcpy(actions.data(), input.noise, actions.size() * sizeof(float));
    } else {
        std::normal_distribution<float> distribution(0.0f, 1.0f);
        for (float & value : actions) value = distribution(rng);
    }
    const auto denoise_started = Clock::now();
    if (!run_denoise(*this, resolved_input, backbone, state_features, actions, dump_directory)) {
        return {};
    }
    stats.ms_denoise = std::chrono::duration<float, std::milli>(
        Clock::now() - denoise_started).count();
    stats.ms_inference = stats.ms_prefill + stats.ms_denoise;

    for (int64_t step = 0; step < cfg.n_suffix; ++step) {
        float * row = actions.data() + static_cast<size_t>(step * cfg.max_action_dim);
        for (int64_t index = 0; index < cfg.real_action_dim; ++index) {
            const float low = action_q01[static_cast<size_t>(index)];
            const float high = action_q99[static_cast<size_t>(index)];
            row[index] = (row[index] + 1.0f) * 0.5f * (high - low) + low;
        }
    }
    if (!dump_directory.empty()) {
        dump_host_tensor(dump_directory, "action_unnormalized", actions,
                         {cfg.n_suffix, cfg.max_action_dim});
    }
    stats.ms_total = std::chrono::duration<float, std::milli>(Clock::now() - started).count();
    return actions;
}

} // namespace vla