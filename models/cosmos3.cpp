// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

// Cosmos3 native CUDA/ggml graph implementation, rebuilt from the PyTorch
// model structure.  The old reference implementation lives in
// models/cosmos3_legacy.cpp for consultation only; this file is the new main
// implementation surface.

#include "arch.h"
#include "model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#ifdef VLA_COSMOS3_CUDA_KERNELS
#include "cosmos3_action_bridge_cuda.h"
#include "cosmos3_mot_action_cuda.h"
#include "cosmos3_qwen_language_cuda.h"
#include "cosmos3_qwen_visual_cuda.h"
#include "cosmos3_wan_vae_cuda.h"
#include "cosmos3_w8_marlin_cuda.h"
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
    cudaStream_t stream);
extern "C" int cosmos3_w8a16_f32_to_bf16_ws(
    const float * x,
    unsigned short * x_bf16_workspace,
    int count,
    cudaStream_t stream);
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
    cudaStream_t stream);
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vla {
namespace {

constexpr int kVisualFrames = 33;
constexpr int kVisualTemporalPatch = 2;
constexpr int kVisualPatch = 16;
constexpr int kVisualMerge = 2;
constexpr int kVisualMergeGroup = kVisualMerge * kVisualMerge;
constexpr int kVisualGridT = 17;
constexpr int kVisualGridH = 34;
constexpr int kVisualGridW = 46;
constexpr int kVisualPatchRows = kVisualGridT * kVisualGridH * kVisualGridW; // 26588
constexpr int kVisualPatchDim = 3 * kVisualTemporalPatch * kVisualPatch * kVisualPatch; // 1536
constexpr int kVisualHidden = 1152;
constexpr int kVisualHeads = 16;
constexpr int kVisualHeadDim = 72;
constexpr int kVisualQkv = 3 * kVisualHidden;
constexpr int kVisualBlocks = 27;
constexpr int kVisualIntermediate = 4304;
constexpr int kVisualMergedHidden = kVisualHidden * kVisualMergeGroup;
constexpr int kVisualTokens = kVisualPatchRows / kVisualMergeGroup; // 6647
constexpr int kLanguageHidden = 4096;
constexpr int kLanguageTokens = 6797;
constexpr int kLanguageUncondTokens = 6791;
constexpr int kLanguageMaxTokens = 8192;
constexpr int kLanguageLayers = 36;
constexpr int kLanguageIntermediate = 12288;
constexpr int kLanguageQHeads = 32;
constexpr int kLanguageKvHeads = 8;
constexpr int kLanguageHeadDim = 128;
constexpr int kLanguageKv = kLanguageKvHeads * kLanguageHeadDim;
constexpr int kLanguageVideoPadToken = 151656;
constexpr int kLanguageVisualFrameSeqLen = 391;
constexpr float kLanguageRmsEps = 1e-6f;
constexpr float kLanguageRopeTheta = 5000000.0f;
constexpr int kActionSteps = 32;
constexpr int kActionConditionTokens = 1;
constexpr int kActionTokens = kActionConditionTokens + kActionSteps;
constexpr int kActionDim = 8;
constexpr int kActionMaxDim = 64;
constexpr int kActionDomains = 32;
constexpr int kActionDefaultDomainId = 8;
constexpr int kMotTextCondTokens = 91;
constexpr int kMotTextUncondTokens = 10;
constexpr int kMotTextMaxTokens = 256;
constexpr int kMotVisionT = 9;
constexpr int kMotVisionH = 17;
constexpr int kMotVisionW = 20;
constexpr int kMotVisionTokens = kMotVisionT * kMotVisionH * kMotVisionW; // 3060
constexpr int kMotVisionConditionTokens = kMotVisionH * kMotVisionW; // first frame is clean conditioning
constexpr int kMotVisionPatchDim = 192;
constexpr int kMotVisionVaeChannels = 48;
constexpr int kMotVisionPatchSize = 2;
constexpr int kMotVisionLatentH = kMotVisionH * kMotVisionPatchSize - 1; // 33
constexpr int kMotVisionLatentW = kMotVisionW * kMotVisionPatchSize; // 40
constexpr int kMotFullTokens = kMotVisionTokens + kActionTokens; // 3093
constexpr int kMotMaxPackedTokens = kMotTextMaxTokens + kMotFullTokens; // 3184
constexpr int kTimeFreqDim = 256;
constexpr float kTimestepScale = 0.001f;

constexpr const char * kTensorMapKey = "cosmos3.tensor_name_map_json";
constexpr const char * kPackedW8Key = "cosmos3.packed_w8_records_json";

constexpr std::array<int, kMotTextCondTokens> kRobolabMotCondTextIds = {
    151644, 872, 198, 19103, 279, 43096, 304, 279, 19212, 13, 1096, 2766,
    5610, 97534, 6194, 504, 5248, 6249, 38455, 13, 576, 1909, 2802, 374,
    504, 279, 32171, 77730, 6249, 13, 576, 5622, 2802, 5610, 1378, 58888,
    97534, 4843, 28045, 13057, 6194, 315, 279, 6109, 504, 14002, 11067, 11,
    448, 279, 12305, 9434, 13, 576, 2766, 374, 220, 17, 13, 15, 6486, 1293,
    323, 374, 315, 220, 16, 20, 43628, 13, 1096, 2766, 374, 315, 220, 20,
    19, 19, 87, 22, 18, 21, 10935, 13, 151645, 198, 151644, 77091, 198,
    151645, 151652,
};

constexpr std::array<int, kMotTextUncondTokens> kRobolabMotUncondTextIds = {
    151644, 872, 198, 151645, 198, 151644, 77091, 198, 151645, 151652,
};

float cosmos3_shift_sigma(float sigma, float shift) {
    return shift * sigma / (1.0f + (shift - 1.0f) * sigma);
}

std::vector<float> cosmos3_unipc_like_sigmas(int num_steps, float shift) {
    num_steps = std::max(1, num_steps);
    std::vector<float> sigmas(static_cast<size_t>(num_steps) + 1, 0.0f);
    constexpr float sigma_max = 1.0f - (1.0f / 1000.0f);
    constexpr float sigma_min = 0.0f;
    for (int i = 0; i < num_steps; ++i) {
        const float base = sigma_max + (sigma_min - sigma_max) *
                                       (static_cast<float>(i) / static_cast<float>(num_steps));
        sigmas[static_cast<size_t>(i)] = cosmos3_shift_sigma(base, shift);
    }
    sigmas[static_cast<size_t>(num_steps)] = 0.0f;
    return sigmas;
}

float cosmos3_lambda_from_sigma(float sigma) {
    if (sigma <= 0.0f) return 80.0f;
    const float alpha = 1.0f - sigma;
    return std::log(std::max(alpha, 1e-20f)) - std::log(sigma);
}

float cosmos3_unipc_h_phi1(float h) {
    return std::expm1(-h);
}

std::string visual_block_name(int block, const char * suffix) {
    return "net.language_model.visual.blocks." + std::to_string(block) + "." + suffix;
}

bool extract_json_string_after(const std::string & text,
                               size_t start,
                               const char * key,
                               std::string * value,
                               size_t * end_pos = nullptr) {
    const std::string needle = std::string("\"") + key + "\":";
    const size_t key_pos = text.find(needle, start);
    if (key_pos == std::string::npos) return false;
    size_t pos = text.find('"', key_pos + needle.size());
    if (pos == std::string::npos) return false;
    ++pos;
    std::string out;
    bool escaped = false;
    for (; pos < text.size(); ++pos) {
        const char c = text[pos];
        if (escaped) {
            out.push_back(c);
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            *value = std::move(out);
            if (end_pos) *end_pos = pos + 1;
            return true;
        }
        out.push_back(c);
    }
    return false;
}

bool extract_json_i64_after(const std::string & text,
                            size_t start,
                            const char * key,
                            int64_t * value) {
    const std::string needle = std::string("\"") + key + "\":";
    const size_t key_pos = text.find(needle, start);
    if (key_pos == std::string::npos) return false;
    size_t pos = key_pos + needle.size();
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) ++pos;
    bool neg = false;
    if (pos < text.size() && text[pos] == '-') {
        neg = true;
        ++pos;
    }
    if (pos >= text.size() || text[pos] < '0' || text[pos] > '9') return false;
    int64_t out = 0;
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
        out = out * 10 + static_cast<int64_t>(text[pos] - '0');
        ++pos;
    }
    *value = neg ? -out : out;
    return true;
}

bool extract_json_i64_array_after(const std::string & text,
                                  size_t start,
                                  const char * key,
                                  std::vector<int64_t> * values) {
    const std::string needle = std::string("\"") + key + "\":";
    const size_t key_pos = text.find(needle, start);
    if (key_pos == std::string::npos) return false;
    size_t pos = key_pos + needle.size();
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) ++pos;
    if (pos >= text.size() || text[pos] != '[') return false;
    ++pos;
    std::vector<int64_t> out;
    while (pos < text.size()) {
        while (pos < text.size() &&
               (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' ||
                text[pos] == '\r' || text[pos] == ',')) {
            ++pos;
        }
        if (pos < text.size() && text[pos] == ']') {
            *values = std::move(out);
            return true;
        }
        bool neg = false;
        if (pos < text.size() && text[pos] == '-') {
            neg = true;
            ++pos;
        }
        if (pos >= text.size() || text[pos] < '0' || text[pos] > '9') return false;
        int64_t value = 0;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
            value = value * 10 + static_cast<int64_t>(text[pos] - '0');
            ++pos;
        }
        out.push_back(neg ? -value : value);
    }
    return false;
}

std::string language_layer_name(int layer, const char * suffix) {
    return "net.language_model.model.layers." + std::to_string(layer) + "." + suffix;
}

bool wam_param_bool(const WamInputs & inputs, const char * key, bool fallback = false) {
    auto it = inputs.params.find(key);
    if (it == inputs.params.end()) return fallback;
    const std::string & v = it->second;
    return v == "1" || v == "true" || v == "True" || v == "yes" || v == "on";
}

int wam_param_int(const WamInputs & inputs, const char * key, int fallback) {
    auto it = inputs.params.find(key);
    if (it == inputs.params.end()) return fallback;
    char * end = nullptr;
    const long value = std::strtol(it->second.c_str(), &end, 10);
    if (!end || *end != '\0') return fallback;
    if (value < 0 || value > 1000000) return fallback;
    return static_cast<int>(value);
}

float wam_param_float(const WamInputs & inputs, const char * key, float fallback) {
    auto it = inputs.params.find(key);
    if (it == inputs.params.end()) return fallback;
    char * end = nullptr;
    const float value = std::strtof(it->second.c_str(), &end);
    if (!end || *end != '\0') return fallback;
    return value;
}

uint64_t cosmos3_hash_bytes(const void * data, size_t bytes, uint64_t seed = 1469598103934665603ULL) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

uint64_t cosmos3_hash_u64(uint64_t h, uint64_t value) {
    return cosmos3_hash_bytes(&value, sizeof(value), h);
}

uint64_t cosmos3_mot_condition_cache_key(const WamTensorView * token_ids,
                                         const int * default_token_ids,
                                         size_t default_token_count,
                                         const WamTensorView * hidden_override,
                                         bool uncond) {
    uint64_t h = cosmos3_hash_u64(1469598103934665603ULL, uncond ? 0x756e636f6e64ULL : 0x636f6e64ULL);
    if (token_ids && token_ids->data && token_ids->bytes > 0) {
        h = cosmos3_hash_u64(h, static_cast<uint64_t>(token_ids->shape.size()));
        for (int64_t dim : token_ids->shape) {
            h = cosmos3_hash_u64(h, static_cast<uint64_t>(dim));
        }
        h = cosmos3_hash_bytes(token_ids->data, token_ids->bytes, h);
    } else if (default_token_ids && default_token_count > 0) {
        h = cosmos3_hash_u64(h, static_cast<uint64_t>(default_token_count));
        h = cosmos3_hash_bytes(default_token_ids, default_token_count * sizeof(int), h);
    }
    if (hidden_override && hidden_override->data && hidden_override->bytes > 0) {
        h = cosmos3_hash_u64(h, 0x68696464656eULL);
        h = cosmos3_hash_u64(h, static_cast<uint64_t>(hidden_override->dtype));
        h = cosmos3_hash_u64(h, static_cast<uint64_t>(hidden_override->shape.size()));
        for (int64_t dim : hidden_override->shape) {
            h = cosmos3_hash_u64(h, static_cast<uint64_t>(dim));
        }
        h = cosmos3_hash_bytes(hidden_override->data, hidden_override->bytes, h);
    }
    return h == 0 ? 1 : h;
}

struct GgufReader {
    gguf_context * gctx = nullptr;
    ggml_context * meta_ctx = nullptr;
    FILE * fp = nullptr;
    size_t data_off = 0;

    ~GgufReader() {
        if (fp) std::fclose(fp);
        if (gctx) gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
    }

    GgufReader() = default;
    GgufReader(const GgufReader &) = delete;
    GgufReader & operator=(const GgufReader &) = delete;

    bool open(const std::string & path) {
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = &meta_ctx;
        gctx = gguf_init_from_file(path.c_str(), params);
        if (!gctx) {
            std::fprintf(stderr, "vla(cosmos3): gguf_init_from_file failed for %s\n", path.c_str());
            return false;
        }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "vla(cosmos3): fopen failed for %s\n", path.c_str());
            return false;
        }
        data_off = gguf_get_data_offset(gctx);
        return true;
    }

    std::string str(const char * key) const {
        const int64_t id = gguf_find_key(gctx, key);
        if (id < 0) return {};
        const char * value = gguf_get_val_str(gctx, id);
        return value ? std::string(value) : std::string();
    }

    bool has_key(const char * key) const {
        return gguf_find_key(gctx, key) >= 0;
    }

    const ggml_tensor * meta(const char * name) const {
        return meta_ctx ? ggml_get_tensor(meta_ctx, name) : nullptr;
    }

    bool read_raw(const char * name, std::vector<uint8_t> * out) const {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) {
            std::fprintf(stderr, "vla(cosmos3): missing GGUF tensor %s\n", name);
            return false;
        }
        const size_t bytes = gguf_get_tensor_size(gctx, id);
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        out->resize(bytes);
        if (std::fseek(fp, static_cast<long>(off), SEEK_SET) != 0) return false;
        return std::fread(out->data(), 1, bytes, fp) == bytes;
    }
};

struct TensorNameIndex {
    std::unordered_map<std::string, std::string> source_to_gguf;
    std::unordered_map<std::string, std::string> gguf_to_source;
    std::unordered_map<std::string, std::vector<int64_t>> source_to_shape;

    bool parse_from_gguf(const GgufReader & g) {
        const std::string json = g.str(kTensorMapKey);
        if (json.empty()) {
            std::fprintf(stderr, "vla(cosmos3): missing %s\n", kTensorMapKey);
            return false;
        }
        size_t pos = 0;
        while (true) {
            const size_t source_pos = json.find("\"source_name\":", pos);
            if (source_pos == std::string::npos) break;
            const size_t rec_begin = json.rfind('{', source_pos);
            const size_t rec_end = json.find('}', source_pos);
            if (rec_begin == std::string::npos || rec_end == std::string::npos) return false;
            const std::string rec = json.substr(rec_begin, rec_end - rec_begin + 1);
            std::string source;
            std::string gguf;
            if (!extract_json_string_after(rec, 0, "source_name", &source)) return false;
            if (!extract_json_string_after(rec, 0, "gguf_name", &gguf)) return false;
            source_to_gguf.emplace(source, gguf);
            gguf_to_source.emplace(gguf, source);
            std::vector<int64_t> shape;
            if (extract_json_i64_array_after(rec, 0, "shape", &shape)) {
                source_to_shape.emplace(source, std::move(shape));
            }
            pos = rec_end + 1;
        }
        if (source_to_gguf.empty()) {
            std::fprintf(stderr, "vla(cosmos3): parsed empty tensor-name map\n");
            return false;
        }
        return true;
    }

    std::string gguf_name_for_source(const std::string & source) const {
        const auto it = source_to_gguf.find(source);
        return it == source_to_gguf.end() ? std::string() : it->second;
    }

    const std::vector<int64_t> * shape_for_source(const std::string & source) const {
        const auto it = source_to_shape.find(source);
        return it == source_to_shape.end() ? nullptr : &it->second;
    }
};

struct PackedW8Index {
    struct Record {
        std::string module;
        std::string qweight_gguf;
        std::string scales_gguf;
        int64_t size_k = 0;
        int64_t size_n = 0;
    };

    std::vector<Record> records;
    std::unordered_map<std::string, size_t> by_module;

    bool parse_from_gguf(const GgufReader & g) {
        const std::string json = g.str(kPackedW8Key);
        if (json.empty()) {
            std::fprintf(stderr, "vla(cosmos3): missing %s\n", kPackedW8Key);
            return false;
        }
        records.clear();
        by_module.clear();
        size_t pos = 0;
        while (true) {
            Record rec;
            size_t after_module = 0;
            if (!extract_json_string_after(json, pos, "module", &rec.module, &after_module)) break;
            const size_t module_pos = json.rfind("\"module\":", after_module);
            const size_t rec_begin = json.rfind('{', module_pos);
            const size_t next_rec = json.find("},{", after_module);
            const size_t rec_end = next_rec == std::string::npos ? json.rfind('}') : next_rec;
            if (module_pos == std::string::npos || rec_begin == std::string::npos ||
                rec_end == std::string::npos || rec_begin > module_pos || rec_end <= rec_begin) {
                return false;
            }
            const std::string rec_json = json.substr(rec_begin, rec_end - rec_begin + 1);
            const size_t q_pos = rec_json.find("\"qweight\":");
            const size_t s_pos = rec_json.find("\"scales\":");
            if (q_pos == std::string::npos || s_pos == std::string::npos) return false;
            if (!extract_json_i64_after(rec_json, 0, "size_k", &rec.size_k)) return false;
            if (!extract_json_i64_after(rec_json, 0, "size_n", &rec.size_n)) return false;
            if (!extract_json_string_after(rec_json, q_pos, "gguf_name", &rec.qweight_gguf)) return false;
            if (!extract_json_string_after(rec_json, s_pos, "gguf_name", &rec.scales_gguf)) return false;
            by_module.emplace(rec.module, records.size());
            records.push_back(std::move(rec));
            pos = after_module;
        }
        if (records.empty()) {
            std::fprintf(stderr, "vla(cosmos3): parsed empty packed W8 index\n");
            return false;
        }
        return true;
    }

    const Record * find(const std::string & module) const {
        const auto it = by_module.find(module);
        return it == by_module.end() ? nullptr : &records[it->second];
    }
};

struct GgufGpuResidentModel {
    std::unique_ptr<GgufReader> reader;
    TensorNameIndex names;
    PackedW8Index w8;
    ggml_backend_t backend = nullptr;
    ggml_context * ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> by_gguf_name;
    std::unordered_map<std::string, ggml_tensor *> by_source_name;

    ~GgufGpuResidentModel() {
        if (weight_buf) ggml_backend_buffer_free(weight_buf);
        if (ctx_weights) ggml_free(ctx_weights);
        if (backend) ggml_backend_free(backend);
    }

    GgufGpuResidentModel() = default;
    GgufGpuResidentModel(const GgufGpuResidentModel &) = delete;
    GgufGpuResidentModel & operator=(const GgufGpuResidentModel &) = delete;

    ggml_tensor * tensor_source(const std::string & source_name) const {
        const auto it = by_source_name.find(source_name);
        return it == by_source_name.end() ? nullptr : it->second;
    }

    bool init_backend() {
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
        if (!backend) {
            std::fprintf(stderr, "vla(cosmos3): ggml_backend_cuda_init failed; Cosmos3 new path requires CUDA\n");
            return false;
        }
        std::fprintf(stderr, "vla(cosmos3): backend = CUDA\n");
        return true;
#else
        std::fprintf(stderr, "vla(cosmos3): Cosmos3 new path requires GGML_CUDA=ON\n");
        return false;
#endif
    }

    bool load(const std::string & path) {
        reader = std::make_unique<GgufReader>();
        if (!reader->open(path)) return false;
        const std::string arch = reader->str("general.architecture");
        if (arch != "cosmos3") {
            std::fprintf(stderr, "vla(cosmos3): %s is not a Cosmos3 GGUF\n", path.c_str());
            return false;
        }
        if (!reader->has_key(kPackedW8Key) ||
            !names.parse_from_gguf(*reader) ||
            !w8.parse_from_gguf(*reader)) {
            return false;
        }
        if (!init_backend()) return false;

        ggml_init_params params{};
        params.mem_size = 64ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ctx_weights = ggml_init(params);
        if (!ctx_weights) {
            std::fprintf(stderr, "vla(cosmos3): ggml_init weight context failed\n");
            return false;
        }

        const int64_t n_tensors = gguf_get_n_tensors(reader->gctx);
        by_gguf_name.reserve(static_cast<size_t>(n_tensors));
        by_source_name.reserve(names.source_to_gguf.size());
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(reader->gctx, i);
            const ggml_tensor * meta = name ? reader->meta(name) : nullptr;
            if (!name || !meta) {
                std::fprintf(stderr, "vla(cosmos3): missing metadata for GGUF tensor index %lld\n",
                             static_cast<long long>(i));
                return false;
            }
            ggml_tensor * t = ggml_new_tensor(ctx_weights, meta->type, GGML_MAX_DIMS, meta->ne);
            if (!t) {
                std::fprintf(stderr, "vla(cosmos3): failed to create resident tensor %s\n", name);
                return false;
            }
            ggml_set_name(t, name);
            by_gguf_name.emplace(name, t);
            const auto source_it = names.gguf_to_source.find(name);
            if (source_it != names.gguf_to_source.end()) {
                by_source_name.emplace(source_it->second, t);
            }
        }

        weight_buf = ggml_backend_alloc_ctx_tensors(ctx_weights, backend);
        if (!weight_buf) {
            std::fprintf(stderr, "vla(cosmos3): failed to allocate full GGUF resident GPU weights\n");
            return false;
        }

        std::vector<uint8_t> bytes;
        for (const auto & item : by_gguf_name) {
            ggml_tensor * t = item.second;
            if (!reader->read_raw(item.first.c_str(), &bytes)) return false;
            if (bytes.size() != ggml_nbytes(t)) {
                std::fprintf(stderr,
                             "vla(cosmos3): resident tensor size mismatch for %s (%zu vs %zu)\n",
                             item.first.c_str(), bytes.size(), ggml_nbytes(t));
                return false;
            }
            ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        }

        std::fprintf(stderr,
                     "vla(cosmos3): cosmos3-new-full-gguf-gpu-resident-ok tensors=%lld bytes=%zu gib=%.2f source_map=%zu\n",
                     static_cast<long long>(n_tensors),
                     ggml_backend_buffer_get_size(weight_buf),
                     ggml_backend_buffer_get_size(weight_buf) / (1024.0 * 1024.0 * 1024.0),
                     names.source_to_gguf.size());
        return true;
    }
};

float bf16_bits_to_f32(uint16_t bits) {
    uint32_t wide = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

bool read_source_tensor_to_f32(const GgufGpuResidentModel & weights,
                               const std::string & source_name,
                               float * dst,
                               int64_t expected_nelements,
                               std::string * error) {
    if (!dst || expected_nelements < 0 || !weights.reader) {
        if (error) *error = "invalid Cosmos3 host tensor read request";
        return false;
    }
    const std::string gguf_name = weights.names.gguf_name_for_source(source_name);
    if (gguf_name.empty()) {
        if (error) *error = "missing Cosmos3 source tensor mapping: " + source_name;
        return false;
    }
    const ggml_tensor * meta = weights.reader->meta(gguf_name.c_str());
    if (!meta) {
        if (error) *error = "missing Cosmos3 GGUF tensor metadata: " + gguf_name;
        return false;
    }
    if (ggml_nelements(meta) != expected_nelements) {
        if (error) {
            *error = "Cosmos3 source tensor element mismatch: " + source_name +
                     " expected=" + std::to_string(static_cast<long long>(expected_nelements)) +
                     " got=" + std::to_string(static_cast<long long>(ggml_nelements(meta)));
        }
        return false;
    }
    std::vector<uint8_t> raw;
    if (!weights.reader->read_raw(gguf_name.c_str(), &raw)) {
        if (error) *error = "failed to read Cosmos3 GGUF tensor: " + gguf_name;
        return false;
    }
    if (meta->type == GGML_TYPE_F32) {
        const size_t bytes = static_cast<size_t>(expected_nelements) * sizeof(float);
        if (raw.size() != bytes) {
            if (error) *error = "Cosmos3 F32 tensor byte mismatch: " + source_name;
            return false;
        }
        std::memcpy(dst, raw.data(), bytes);
        return true;
    }
    if (meta->type == GGML_TYPE_BF16) {
        const size_t bytes = static_cast<size_t>(expected_nelements) * sizeof(uint16_t);
        if (raw.size() != bytes) {
            if (error) *error = "Cosmos3 BF16 tensor byte mismatch: " + source_name;
            return false;
        }
        const uint16_t * src = reinterpret_cast<const uint16_t *>(raw.data());
        for (int64_t i = 0; i < expected_nelements; ++i) {
            dst[i] = bf16_bits_to_f32(src[i]);
        }
        return true;
    }
    if (error) {
        *error = "unsupported Cosmos3 host tensor dtype for " + source_name +
                 ": " + std::to_string(static_cast<int>(meta->type));
    }
    return false;
}

bool read_source_tensor_to_f32(const GgufGpuResidentModel & weights,
                               const std::string & source_name,
                               std::vector<float> * out,
                               int64_t expected_nelements,
                               std::string * error) {
    if (!out || expected_nelements < 0) {
        if (error) *error = "invalid Cosmos3 host vector read request";
        return false;
    }
    out->assign(static_cast<size_t>(expected_nelements), 0.0f);
    return read_source_tensor_to_f32(weights, source_name, out->data(), expected_nelements, error);
}

bool set_tensor_from_source_f32(const GgufGpuResidentModel & weights,
                                ggml_tensor * t,
                                const std::string & source_name,
                                std::string * error) {
    if (!t) {
        if (error) *error = "null ggml tensor for source read: " + source_name;
        return false;
    }
    std::vector<float> tmp;
    if (!read_source_tensor_to_f32(weights, source_name, &tmp, ggml_nelements(t), error)) {
        return false;
    }
    ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size() * sizeof(float));
    return true;
}

struct VisualTowerKernelGaps {
    bool needs_pixel_values_kernel = true;
    bool needs_2d_rope_window_attention_kernel = true;
    bool needs_merge_reorder_kernel = true;
};

struct VisualGraphTensors {
    ggml_tensor * pixel_values = nullptr;
    ggml_tensor * pos_ids = nullptr;
    ggml_tensor * output_tokens = nullptr;
};

enum class VisualSidecarKind {
    ROPE_WINDOW_ATTENTION,
    MERGE2X2_REORDER,
};

struct VisualSidecarNode {
    VisualSidecarKind kind = VisualSidecarKind::ROPE_WINDOW_ATTENTION;
    int block = -1;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
};

struct VisualGraphPlan {
    VisualGraphTensors tensors;
    std::vector<VisualSidecarNode> sidecars;
};

struct VisualCudaTables {
    int * coords_yx = nullptr;
    int * window_offsets = nullptr;
    int * token_to_window = nullptr;
    std::vector<int32_t> pos_ids_host;
    int windows = 0;
    int window_size = 128;

    ~VisualCudaTables() { release(); }

    VisualCudaTables() = default;
    VisualCudaTables(const VisualCudaTables &) = delete;
    VisualCudaTables & operator=(const VisualCudaTables &) = delete;

    void release() {
#ifdef VLA_COSMOS3_CUDA_KERNELS
        if (coords_yx) cudaFree(coords_yx);
        if (window_offsets) cudaFree(window_offsets);
        if (token_to_window) cudaFree(token_to_window);
#endif
        coords_yx = nullptr;
        window_offsets = nullptr;
        token_to_window = nullptr;
        pos_ids_host.clear();
        windows = 0;
    }

    bool init(std::string * error) {
#ifndef VLA_COSMOS3_CUDA_KERNELS
        *error = "Cosmos3 visual CUDA tables require VLA_COSMOS3_CUDA_KERNELS";
        return false;
#else
        release();
        std::vector<int> coords(static_cast<size_t>(kVisualPatchRows) * 2u, 0);
        pos_ids_host.assign(kVisualPatchRows, 0);
        for (int row = 0; row < kVisualPatchRows; ++row) {
            int r = row;
            const int merge_w = r % kVisualMerge; r /= kVisualMerge;
            const int merge_h = r % kVisualMerge; r /= kVisualMerge;
            const int gw_merged = kVisualGridW / kVisualMerge;
            const int gh_merged = kVisualGridH / kVisualMerge;
            const int gw = r % gw_merged; r /= gw_merged;
            const int gh = r % gh_merged; r /= gh_merged;
            (void) gh_merged;
            const int patch_y = gh * kVisualMerge + merge_h;
            const int patch_x = gw * kVisualMerge + merge_w;
            coords[static_cast<size_t>(row) * 2u + 0u] = patch_y;
            coords[static_cast<size_t>(row) * 2u + 1u] = patch_x;
            pos_ids_host[static_cast<size_t>(row)] = patch_y * 48 + patch_x;
        }

        windows = (kVisualPatchRows + window_size - 1) / window_size;
        std::vector<int> offsets(static_cast<size_t>(windows) + 1u, 0);
        std::vector<int> token_window(kVisualPatchRows, 0);
        for (int w = 0; w <= windows; ++w) {
            offsets[static_cast<size_t>(w)] = std::min(w * window_size, kVisualPatchRows);
        }
        for (int w = 0; w < windows; ++w) {
            for (int token = offsets[static_cast<size_t>(w)];
                 token < offsets[static_cast<size_t>(w + 1)]; ++token) {
                token_window[static_cast<size_t>(token)] = w;
            }
        }

        auto upload = [&](const std::vector<int> & src, int ** dst, const char * label) -> bool {
            const size_t bytes = src.size() * sizeof(int);
            cudaError_t status = cudaMalloc(reinterpret_cast<void **>(dst), bytes);
            if (status != cudaSuccess) {
                *error = std::string("cudaMalloc failed for ") + label + ": " + cudaGetErrorString(status);
                release();
                return false;
            }
            status = cudaMemcpy(*dst, src.data(), bytes, cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                *error = std::string("cudaMemcpy failed for ") + label + ": " + cudaGetErrorString(status);
                release();
                return false;
            }
            return true;
        };
        if (!upload(coords, &coords_yx, "visual.coords_yx")) return false;
        if (!upload(offsets, &window_offsets, "visual.window_offsets")) return false;
        if (!upload(token_window, &token_to_window, "visual.token_to_window")) return false;
        return true;
#endif
    }
};

struct VisualTowerScheduler {
    ggml_backend_t backend = nullptr;
    VisualCudaTables * tables = nullptr;

    bool build_master_graph(ggml_context * ctx,
                            const VisualGraphPlan & plan,
                            ggml_cgraph ** graph,
                            std::string * error) const {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
        if (!gf) {
            *error = "failed to allocate Cosmos3 visual scheduler master graph";
            return false;
        }
        for (const VisualSidecarNode & node : plan.sidecars) {
            if (node.input) ggml_build_forward_expand(gf, node.input);
            if (node.output) ggml_build_forward_expand(gf, node.output);
        }
        if (plan.tensors.output_tokens) {
            ggml_build_forward_expand(gf, plan.tensors.output_tokens);
        }
        *graph = gf;
        return true;
    }

    bool compute_to(ggml_context * ctx,
                    ggml_tensor * target,
                    std::string * error) const {
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, false);
        if (!gf) {
            *error = "failed to allocate Cosmos3 visual scheduler segment graph";
            return false;
        }
        ggml_build_forward_expand(gf, target);
        const ggml_status status = ggml_backend_graph_compute(backend, gf);
        if (status != GGML_STATUS_SUCCESS) {
            *error = std::string("Cosmos3 visual scheduler ggml segment compute failed at ") +
                     (target && target->name[0] ? target->name : "<unnamed>");
            return false;
        }
#ifdef VLA_COSMOS3_CUDA_KERNELS
        cudaError_t sync_status = cudaDeviceSynchronize();
        if (sync_status != cudaSuccess) {
            *error = std::string("Cosmos3 visual scheduler ggml segment CUDA sync failed at ") +
                     (target && target->name[0] ? target->name : "<unnamed>") +
                     ": " + cudaGetErrorString(sync_status);
            return false;
        }
#endif
        return true;
    }

    bool launch_sidecar(const VisualSidecarNode & node, std::string * error) const {
#ifndef VLA_COSMOS3_CUDA_KERNELS
        *error = "Cosmos3 visual scheduler requires VLA_COSMOS3_CUDA_KERNELS";
        return false;
#else
        if (!node.input || !node.output || !node.input->data || !node.output->data ||
            !tables || !tables->coords_yx || !tables->window_offsets || !tables->token_to_window) {
            *error = "Cosmos3 visual scheduler sidecar received uninitialized tensors/tables";
            return false;
        }
        cudaGetLastError();
        int rc = -1;
        if (node.kind == VisualSidecarKind::ROPE_WINDOW_ATTENTION) {
            rc = cosmos3_visual_rope_window_attention_f32(
                static_cast<const float *>(node.input->data),
                tables->coords_yx,
                tables->window_offsets,
                tables->token_to_window,
                static_cast<float *>(node.output->data),
                kVisualPatchRows,
                tables->windows,
                kVisualHeads,
                kVisualHeadDim,
                10000.0f,
                nullptr);
        } else {
            rc = cosmos3_visual_merge2x2_reorder_f32(
                static_cast<const float *>(node.input->data),
                static_cast<float *>(node.output->data),
                kVisualPatchRows,
                kVisualHidden,
                kVisualMergeGroup,
                nullptr);
        }
        if (rc != 0) {
            *error = "Cosmos3 visual scheduler CUDA sidecar launch failed at block=" +
                     std::to_string(node.block);
            return false;
        }
        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            *error = "Cosmos3 visual scheduler CUDA sidecar sync failed at block=" +
                     std::to_string(node.block) + ": " +
                     cudaGetErrorString(status);
            return false;
        }
        return true;
#endif
    }

    bool prepare(ggml_context * ctx,
                 const VisualGraphPlan & plan,
                 ggml_gallocr_t * allocator,
                 std::string * error) const {
        ggml_cgraph * master = nullptr;
        if (!build_master_graph(ctx, plan, &master, error)) return false;
        *allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!*allocator || !ggml_gallocr_alloc_graph(*allocator, master)) {
            if (*allocator) {
                ggml_gallocr_free(*allocator);
                *allocator = nullptr;
            }
            *error = "Cosmos3 visual scheduler failed to allocate graph tensors";
            return false;
        }
        return true;
    }

    bool execute(ggml_context * ctx,
                 const VisualGraphPlan & plan,
                 std::string * error) const {
        // Segment scheduler: compute each sidecar input with ggml, launch the
        // matching CUDA kernel into its sidecar output, then continue.  This is
        // intentionally explicit so later Cosmos3 kernels can be added to the
        // same scheduling surface before we collapse stable kernels into ggml
        // backend custom ops.
        for (const VisualSidecarNode & node : plan.sidecars) {
            if (!compute_to(ctx, node.input, error)) return false;
            if (!launch_sidecar(node, error)) return false;
        }
        if (!compute_to(ctx, plan.tensors.output_tokens, error)) return false;
        return true;
    }
};

struct VisualSchedulerPlanSummary {
    size_t sidecars = 0;
    size_t attention_sidecars = 0;
    size_t merge_sidecars = 0;
};

VisualSchedulerPlanSummary summarize_visual_plan(const VisualGraphPlan & plan) {
    VisualSchedulerPlanSummary summary;
    summary.sidecars = plan.sidecars.size();
    for (const VisualSidecarNode & node : plan.sidecars) {
        if (node.kind == VisualSidecarKind::ROPE_WINDOW_ATTENTION) {
            ++summary.attention_sidecars;
        } else if (node.kind == VisualSidecarKind::MERGE2X2_REORDER) {
            ++summary.merge_sidecars;
        }
    }
    return summary;
}

const WamTensorView * find_wam_tensor(const WamInputs & inputs, const char * name) {
    for (const WamTensorView & tensor : inputs.tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

bool is_shape(const WamTensorView & tensor, std::initializer_list<int64_t> dims) {
    if (tensor.shape.size() != dims.size()) return false;
    size_t i = 0;
    for (int64_t dim : dims) {
        if (tensor.shape[i++] != dim) return false;
    }
    return true;
}

#ifdef VLA_COSMOS3_CUDA_KERNELS
bool append_cuda_prefix_tensor(std::vector<WamTensor> * tensors,
                               const char * name,
                               const float * src,
                               int rows,
                               int cols,
                               int src_cols,
                               std::string * error) {
    if (!tensors || !name || !src || rows <= 0 || cols <= 0 || src_cols <= 0) return true;
    WamTensor tensor;
    tensor.name = name;
    tensor.dtype = WamDType::F32;
    tensor.shape = {rows, cols};
    tensor.data.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float));
    cudaError_t st = cudaMemcpy2D(tensor.data.data(),
                                  static_cast<size_t>(cols) * sizeof(float),
                                  src,
                                  static_cast<size_t>(src_cols) * sizeof(float),
                                  static_cast<size_t>(cols) * sizeof(float),
                                  static_cast<size_t>(rows),
                                  cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) {
        *error = std::string("Cosmos3 CUDA trace copy failed for ") + name + ": " +
                 cudaGetErrorString(st);
        return false;
    }
    tensors->push_back(std::move(tensor));
    return true;
}

constexpr std::array<int, 14> kMotVisionTraceRows = {{
    0, 1, 16, 339, 340, 341, 680, 1020, 1360, 1700, 2040, 2380, 2720, 3059,
}};

constexpr std::array<int, 23> kMotLayerTraceLayers = {{
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    15, 16, 17, 18, 19, 20, 21, 22, 23,
    31, 32, 33, 34, 35,
}};

void append_visual_timing_tensor(std::vector<WamTensor> * tensors,
                                 const float * values,
                                 int count) {
    if (!tensors || !values || count <= 0) return;
    WamTensor tensor;
    tensor.name = "cosmos3.debug.visual_timing_ms";
    tensor.dtype = WamDType::F32;
    tensor.shape = {1, count};
    tensor.data.resize(static_cast<size_t>(count) * sizeof(float));
    std::memcpy(tensor.data.data(), values, static_cast<size_t>(count) * sizeof(float));
    tensors->push_back(std::move(tensor));
}

void append_scalar_timing_tensor(std::vector<WamTensor> * tensors,
                                 const char * name,
                                 float value) {
    if (!tensors || !name) return;
    WamTensor tensor;
    tensor.name = name;
    tensor.dtype = WamDType::F32;
    tensor.shape = {1, 1};
    tensor.data.resize(sizeof(float));
    std::memcpy(tensor.data.data(), &value, sizeof(float));
    tensors->push_back(std::move(tensor));
}

void append_host_f32_tensor(std::vector<WamTensor> * tensors,
                            const char * name,
                            const float * values,
                            int rows,
                            int cols) {
    if (!tensors || !name || !values || rows <= 0 || cols <= 0) return;
    WamTensor tensor;
    tensor.name = name;
    tensor.dtype = WamDType::F32;
    tensor.shape = {rows, cols};
    tensor.data.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float));
    std::memcpy(tensor.data.data(), values, tensor.data.size());
    tensors->push_back(std::move(tensor));
}
#endif

class NumpyRandomStateNormal {
public:
    explicit NumpyRandomStateNormal(uint32_t seed) {
        state_[0] = seed;
        for (int i = 1; i < kStateSize; ++i) {
            state_[i] = 1812433253U * (state_[i - 1] ^ (state_[i - 1] >> 30)) +
                        static_cast<uint32_t>(i);
        }
        pos_ = kStateSize;
    }

    float standard_normal_f32() {
        if (has_gauss_) {
            has_gauss_ = false;
            return static_cast<float>(gauss_);
        }
        double x1;
        double x2;
        double r2;
        do {
            x1 = 2.0 * uniform_double() - 1.0;
            x2 = 2.0 * uniform_double() - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);
        const double f = std::sqrt(-2.0 * std::log(r2) / r2);
        gauss_ = f * x1;
        has_gauss_ = true;
        return static_cast<float>(f * x2);
    }

private:
    static constexpr int kStateSize = 624;
    static constexpr int kPeriod = 397;

    void twist() {
        static constexpr uint32_t kUpperMask = 0x80000000U;
        static constexpr uint32_t kLowerMask = 0x7fffffffU;
        static constexpr uint32_t kMatrixA = 0x9908b0dfU;
        for (int i = 0; i < kStateSize; ++i) {
            const uint32_t y = (state_[i] & kUpperMask) |
                               (state_[(i + 1) % kStateSize] & kLowerMask);
            state_[i] = state_[(i + kPeriod) % kStateSize] ^ (y >> 1) ^
                        ((y & 1U) ? kMatrixA : 0U);
        }
        pos_ = 0;
    }

    uint32_t random_u32() {
        if (pos_ >= kStateSize) {
            twist();
        }
        uint32_t y = state_[pos_++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }

    double uniform_double() {
        const uint32_t a = random_u32() >> 5;
        const uint32_t b = random_u32() >> 6;
        return (static_cast<double>(a) * 67108864.0 + static_cast<double>(b)) /
               9007199254740992.0;
    }

    std::array<uint32_t, kStateSize> state_{};
    int pos_ = kStateSize;
    bool has_gauss_ = false;
    double gauss_ = 0.0;
};

float round_to_bf16_value(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1U;
    bits += 0x7fffU + lsb;
    bits &= 0xffff0000U;
    float rounded;
    std::memcpy(&rounded, &bits, sizeof(rounded));
    return rounded;
}

bool bind_visual_fixed_inputs(const VisualCudaTables & tables,
                              const VisualGraphPlan & plan,
                              std::string * error) {
    if (!plan.tensors.pos_ids || tables.pos_ids_host.size() != static_cast<size_t>(kVisualPatchRows)) {
        *error = "Cosmos3 visual scheduler has no valid pos_ids table";
        return false;
    }
    ggml_backend_tensor_set(plan.tensors.pos_ids,
                            tables.pos_ids_host.data(),
                            0,
                            tables.pos_ids_host.size() * sizeof(int32_t));
    return true;
}

bool bind_visual_pixel_values_if_present(const WamInputs & inputs,
                                         const VisualGraphPlan & plan,
                                         bool * bound,
                                         std::string * error) {
    *bound = false;
    const WamTensorView * pixel = find_wam_tensor(inputs, "cosmos3.visual.pixel_values");
    if (!pixel) return true;
    if (pixel->dtype != WamDType::F32 ||
        !is_shape(*pixel, {kVisualPatchRows, kVisualPatchDim}) ||
        pixel->bytes != static_cast<size_t>(kVisualPatchRows) *
                        static_cast<size_t>(kVisualPatchDim) * sizeof(float)) {
        *error = "cosmos3.visual.pixel_values must be F32 [26588,1536]";
        return false;
    }
    if (!plan.tensors.pixel_values) {
        *error = "Cosmos3 visual scheduler has no pixel_values graph input";
        return false;
    }
    ggml_backend_tensor_set(plan.tensors.pixel_values, pixel->data, 0, pixel->bytes);
    *bound = true;
    return true;
}

bool bind_visual_pixel_values_from_robolab_image_if_present(const WamInputs & inputs,
                                                            const VisualGraphPlan & plan,
                                                            bool * bound,
                                                            std::string * error) {
    *bound = false;
    const WamTensorView * image = find_wam_tensor(inputs, "observation/image");
    if (!image) return true;
    if (image->dtype != WamDType::U8 ||
        image->shape.size() != 3 ||
        image->shape[0] <= 0 ||
        image->shape[1] <= 0 ||
        image->shape[2] != 3 ||
        image->bytes != static_cast<size_t>(image->shape[0]) *
                        static_cast<size_t>(image->shape[1]) * 3u) {
        *error = "observation/image must be U8 [H,W,3]";
        return false;
    }
    if (!plan.tensors.pixel_values || !plan.tensors.pixel_values->data) {
        *error = "Cosmos3 visual scheduler has no allocated pixel_values graph input";
        return false;
    }
#ifndef VLA_COSMOS3_CUDA_KERNELS
    *error = "Cosmos3 RoboLab image preprocessing requires VLA_COSMOS3_CUDA_KERNELS";
    return false;
#else
    unsigned char * image_dev = nullptr;
    cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&image_dev), image->bytes);
    if (status != cudaSuccess) {
        *error = std::string("cudaMalloc failed for observation/image: ") + cudaGetErrorString(status);
        return false;
    }
    status = cudaMemcpy(image_dev, image->data, image->bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(image_dev);
        *error = std::string("cudaMemcpy failed for observation/image: ") + cudaGetErrorString(status);
        return false;
    }
    const int rc = cosmos3_visual_robolab_image_to_pixel_values_f32(
        image_dev,
        static_cast<int>(image->shape[0]),
        static_cast<int>(image->shape[1]),
        static_cast<float *>(plan.tensors.pixel_values->data),
        kVisualPatchRows,
        kVisualPatchDim,
        nullptr);
    if (rc != 0) {
        cudaFree(image_dev);
        *error = "Cosmos3 visual RoboLab image-to-pixel_values CUDA launch failed";
        return false;
    }
    status = cudaDeviceSynchronize();
    cudaFree(image_dev);
    if (status != cudaSuccess) {
        *error = std::string("Cosmos3 visual RoboLab image preprocessing CUDA sync failed: ") +
                 cudaGetErrorString(status);
        return false;
    }
    *bound = true;
    return true;
#endif
}

ggml_tensor * wan_vae_norm_silu_to_whdc(ggml_context * C,
                                        ggml_tensor * x_whdc,
                                        ggml_tensor * gamma) {
    ggml_tensor * x_cwhd = ggml_cont(C, ggml_permute(C, x_whdc, 1, 2, 3, 0));
    ggml_tensor * n = ggml_rms_norm(C, x_cwhd, 1e-12f);
    ggml_tensor * y = ggml_silu(C, ggml_mul(C, n, gamma));
    return ggml_cont(C, ggml_permute(C, y, 3, 0, 1, 2));
}

ggml_tensor * wan_vae_causal_conv3d_ks3_pad1(ggml_context * C,
                                             ggml_tensor * w,
                                             ggml_tensor * x_whdc,
                                             int in_channels) {
    ggml_tensor * padded = ggml_pad_ext(C, x_whdc,
                                        1, 1,
                                        1, 1,
                                        2, 0,
                                        0, 0);
    return ggml_conv_3d(C, w, padded, in_channels,
                        1, 1, 1,
                        0, 0, 0,
                        1, 1, 1);
}

ggml_tensor * wan_vae_spatial_downsample2d(ggml_context * C,
                                           ggml_tensor * w,
                                           ggml_tensor * x_whdc) {
    ggml_tensor * x_whct = ggml_cont(C, ggml_permute(C, x_whdc, 0, 1, 3, 2));
    ggml_tensor * padded = ggml_pad_ext(C, x_whct,
                                        0, 1,
                                        0, 1,
                                        0, 0,
                                        0, 0);
    ggml_tensor * down = ggml_conv_2d(C, w, padded, 2, 2, 0, 0, 1, 1);
    return ggml_cont(C, ggml_permute(C, down, 0, 1, 3, 2));
}

ggml_tensor * need_source(const GgufGpuResidentModel & weights,
                          const char * source_name,
                          std::string * error) {
    ggml_tensor * t = weights.tensor_source(source_name);
    if (!t) {
        *error = std::string("missing Cosmos3 source tensor in resident GGUF pool: ") + source_name;
    }
    return t;
}

ggml_tensor * need_gpu_bf16_source(const GgufGpuResidentModel & weights,
                                   const char * source_name,
                                   std::string * error) {
    ggml_tensor * t = weights.tensor_source(source_name);
    if (!t) {
        *error = std::string("missing BF16 tensor: ") + source_name;
        return nullptr;
    }
    if (t->type != GGML_TYPE_BF16 || !t->data) {
        *error = std::string("tensor is not GPU-resident BF16: ") + source_name;
        return nullptr;
    }
    return t;
}

ggml_tensor * graph_linear(ggml_context * ctx,
                           ggml_tensor * weight,
                           ggml_tensor * bias,
                           ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(ctx, weight, x);
    if (!bias) return y;
    ggml_tensor * bias_f32 = bias->type == GGML_TYPE_F32 ? bias : ggml_cast(ctx, bias, GGML_TYPE_F32);
    return ggml_add(ctx, y, bias_f32);
}

ggml_tensor * graph_layer_norm(ggml_context * ctx,
                               ggml_tensor * x,
                               ggml_tensor * weight,
                               ggml_tensor * bias,
                               float eps = 1e-6f) {
    ggml_tensor * weight_f32 = weight->type == GGML_TYPE_F32 ? weight : ggml_cast(ctx, weight, GGML_TYPE_F32);
    ggml_tensor * bias_f32 = bias->type == GGML_TYPE_F32 ? bias : ggml_cast(ctx, bias, GGML_TYPE_F32);
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), weight_f32), bias_f32);
}

bool build_visual_block_graph(ggml_context * ctx,
                              const GgufGpuResidentModel & weights,
                              int block,
                              ggml_tensor * x,
                              std::vector<VisualSidecarNode> * sidecars,
                              ggml_tensor ** out,
                              std::string * error) {
    ggml_tensor * norm1_w = need_source(weights, visual_block_name(block, "norm1.weight").c_str(), error);
    ggml_tensor * norm1_b = need_source(weights, visual_block_name(block, "norm1.bias").c_str(), error);
    ggml_tensor * qkv_w = need_source(weights, visual_block_name(block, "attn.qkv.weight").c_str(), error);
    ggml_tensor * qkv_b = need_source(weights, visual_block_name(block, "attn.qkv.bias").c_str(), error);
    ggml_tensor * proj_w = need_source(weights, visual_block_name(block, "attn.proj.weight").c_str(), error);
    ggml_tensor * proj_b = need_source(weights, visual_block_name(block, "attn.proj.bias").c_str(), error);
    ggml_tensor * norm2_w = need_source(weights, visual_block_name(block, "norm2.weight").c_str(), error);
    ggml_tensor * norm2_b = need_source(weights, visual_block_name(block, "norm2.bias").c_str(), error);
    ggml_tensor * fc1_w = need_source(weights, visual_block_name(block, "mlp.linear_fc1.weight").c_str(), error);
    ggml_tensor * fc1_b = need_source(weights, visual_block_name(block, "mlp.linear_fc1.bias").c_str(), error);
    ggml_tensor * fc2_w = need_source(weights, visual_block_name(block, "mlp.linear_fc2.weight").c_str(), error);
    ggml_tensor * fc2_b = need_source(weights, visual_block_name(block, "mlp.linear_fc2.bias").c_str(), error);
    if (!norm1_w || !norm1_b || !qkv_w || !qkv_b || !proj_w || !proj_b ||
        !norm2_w || !norm2_b || !fc1_w || !fc1_b || !fc2_w || !fc2_b) {
        return false;
    }

    ggml_tensor * x_norm = graph_layer_norm(ctx, x, norm1_w, norm1_b);
    ggml_tensor * qkv = graph_linear(ctx, qkv_w, qkv_b, x_norm);
    ggml_set_name(qkv, ("cosmos3.visual.block." + std::to_string(block) + ".qkv.sidecar_input").c_str());
    ggml_set_output(qkv);

    // This tensor is produced by cosmos3_visual_rope_window_attention_f32.
    // It is intentionally represented in the ggml graph as an input-like
    // sidecar tensor so downstream standard ops remain graph-native.
    ggml_tensor * attn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kVisualHidden, kVisualPatchRows);
    if (!attn) {
        *error = "failed to allocate Cosmos3 visual sidecar attention tensor";
        return false;
    }
    ggml_set_name(attn, ("cosmos3.visual.block." + std::to_string(block) + ".attention.sidecar").c_str());
    ggml_set_input(attn);
    ggml_set_output(attn);
    if (sidecars) {
        sidecars->push_back(VisualSidecarNode{
            VisualSidecarKind::ROPE_WINDOW_ATTENTION,
            block,
            qkv,
            attn,
        });
    }

    ggml_tensor * projected = graph_linear(ctx, proj_w, proj_b, attn);
    ggml_tensor * post_attn = ggml_add(ctx, x, projected);
    ggml_tensor * norm2 = graph_layer_norm(ctx, post_attn, norm2_w, norm2_b);
    ggml_tensor * mlp_h = ggml_gelu(ctx, graph_linear(ctx, fc1_w, fc1_b, norm2));
    ggml_tensor * mlp_y = graph_linear(ctx, fc2_w, fc2_b, mlp_h);
    *out = ggml_add(ctx, post_attn, mlp_y);
    return true;
}

bool build_visual_tower_graph(ggml_context * ctx,
                              const GgufGpuResidentModel & weights,
                              VisualGraphPlan * plan,
                              VisualTowerKernelGaps * gaps,
                              std::string * error) {
    ggml_tensor * patch_w = need_source(weights, "net.language_model.visual.patch_embed.proj.weight", error);
    ggml_tensor * patch_b = need_source(weights, "net.language_model.visual.patch_embed.proj.bias", error);
    ggml_tensor * pos_w = need_source(weights, "net.language_model.visual.pos_embed.weight", error);
    if (!patch_w || !patch_b || !pos_w) return false;

    VisualGraphTensors & tensors = plan->tensors;
    tensors.pixel_values = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kVisualPatchDim, kVisualPatchRows);
    tensors.pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kVisualPatchRows);
    if (!tensors.pixel_values || !tensors.pos_ids) {
        *error = "failed to allocate Cosmos3 visual graph inputs";
        return false;
    }
    ggml_set_name(tensors.pixel_values, "cosmos3.visual.pixel_values");
    ggml_set_name(tensors.pos_ids, "cosmos3.visual.pos_ids");
    ggml_set_input(tensors.pixel_values);
    ggml_set_input(tensors.pos_ids);

    ggml_tensor * x = graph_linear(ctx, patch_w, patch_b, tensors.pixel_values);
    ggml_tensor * pos = ggml_get_rows(ctx, pos_w, tensors.pos_ids);
    pos = pos->type == GGML_TYPE_F32 ? pos : ggml_cast(ctx, pos, GGML_TYPE_F32);
    x = ggml_add(ctx, x, pos);

    for (int block = 0; block < kVisualBlocks; ++block) {
        ggml_tensor * next = nullptr;
        if (!build_visual_block_graph(ctx, weights, block, x, &plan->sidecars, &next, error)) {
            return false;
        }
        x = next;
    }

    ggml_tensor * merger_norm_w = need_source(weights, "net.language_model.visual.merger.norm.weight", error);
    ggml_tensor * merger_norm_b = need_source(weights, "net.language_model.visual.merger.norm.bias", error);
    ggml_tensor * merger_fc1_w = need_source(weights, "net.language_model.visual.merger.linear_fc1.weight", error);
    ggml_tensor * merger_fc1_b = need_source(weights, "net.language_model.visual.merger.linear_fc1.bias", error);
    ggml_tensor * merger_fc2_w = need_source(weights, "net.language_model.visual.merger.linear_fc2.weight", error);
    ggml_tensor * merger_fc2_b = need_source(weights, "net.language_model.visual.merger.linear_fc2.bias", error);
    if (!merger_norm_w || !merger_norm_b || !merger_fc1_w || !merger_fc1_b || !merger_fc2_w || !merger_fc2_b) {
        return false;
    }

    ggml_tensor * normed = graph_layer_norm(ctx, x, merger_norm_w, merger_norm_b);
    ggml_set_name(normed, "cosmos3.visual.merger.normed.sidecar_input");
    ggml_set_output(normed);
    ggml_tensor * merged = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kVisualMergedHidden, kVisualTokens);
    if (!merged) {
        *error = "failed to allocate Cosmos3 visual merge sidecar tensor";
        return false;
    }
    ggml_set_name(merged, "cosmos3.visual.merger.merge2x2.sidecar");
    ggml_set_input(merged);
    ggml_set_output(merged);
    plan->sidecars.push_back(VisualSidecarNode{
        VisualSidecarKind::MERGE2X2_REORDER,
        -1,
        normed,
        merged,
    });

    ggml_tensor * merger_h = ggml_gelu(ctx, graph_linear(ctx, merger_fc1_w, merger_fc1_b, merged));
    tensors.output_tokens = graph_linear(ctx, merger_fc2_w, merger_fc2_b, merger_h);
    ggml_set_name(tensors.output_tokens, "cosmos3.visual.tokens");
    ggml_set_output(tensors.output_tokens);
    gaps->needs_2d_rope_window_attention_kernel = false;
    gaps->needs_merge_reorder_kernel = false;
    return true;
}

#ifdef VLA_COSMOS3_CUDA_KERNELS
bool bind_visual_cuda_runtime_weights(cosmos3_visual_cuda_ctx * visual,
                                      const GgufGpuResidentModel & weights,
                                      std::string * error) {
    auto need = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = weights.tensor_source(name);
        if (!t) {
            *error = "missing visual CUDA weight: " + name;
        } else if (!t->data) {
            *error = "visual CUDA weight has no device data: " + name;
            return nullptr;
        }
        return t;
    };
    ggml_tensor * patch_w = need("net.language_model.visual.patch_embed.proj.weight");
    ggml_tensor * patch_b = need("net.language_model.visual.patch_embed.proj.bias");
    ggml_tensor * pos = need("net.language_model.visual.pos_embed.weight");
    if (!patch_w || !patch_b || !pos) return false;
    cosmos3_visual_cuda_set_embed(visual, patch_w->data, patch_b->data, pos->data);

    for (int block = 0; block < kVisualBlocks; ++block) {
        cosmos3_visual_layer_cuda layer{};
        auto n = [&](const char * suffix) { return visual_block_name(block, suffix); };
        ggml_tensor * norm1_w = need(n("norm1.weight"));
        ggml_tensor * norm1_b = need(n("norm1.bias"));
        ggml_tensor * qkv_w = need(n("attn.qkv.weight"));
        ggml_tensor * qkv_b = need(n("attn.qkv.bias"));
        ggml_tensor * proj_w = need(n("attn.proj.weight"));
        ggml_tensor * proj_b = need(n("attn.proj.bias"));
        ggml_tensor * norm2_w = need(n("norm2.weight"));
        ggml_tensor * norm2_b = need(n("norm2.bias"));
        ggml_tensor * fc1_w = need(n("mlp.linear_fc1.weight"));
        ggml_tensor * fc1_b = need(n("mlp.linear_fc1.bias"));
        ggml_tensor * fc2_w = need(n("mlp.linear_fc2.weight"));
        ggml_tensor * fc2_b = need(n("mlp.linear_fc2.bias"));
        if (!norm1_w || !norm1_b || !qkv_w || !qkv_b || !proj_w || !proj_b ||
            !norm2_w || !norm2_b || !fc1_w || !fc1_b || !fc2_w || !fc2_b) {
            return false;
        }
        layer.norm1_w = norm1_w->data;
        layer.norm1_b = norm1_b->data;
        layer.qkv_w = qkv_w->data;
        layer.qkv_b = qkv_b->data;
        layer.proj_w = proj_w->data;
        layer.proj_b = proj_b->data;
        layer.norm2_w = norm2_w->data;
        layer.norm2_b = norm2_b->data;
        layer.fc1_w = fc1_w->data;
        layer.fc1_b = fc1_b->data;
        layer.fc2_w = fc2_w->data;
        layer.fc2_b = fc2_b->data;
        cosmos3_visual_cuda_set_layer(visual, block, &layer);
    }

    ggml_tensor * merger_norm_w = need("net.language_model.visual.merger.norm.weight");
    ggml_tensor * merger_norm_b = need("net.language_model.visual.merger.norm.bias");
    ggml_tensor * merger_fc1_w = need("net.language_model.visual.merger.linear_fc1.weight");
    ggml_tensor * merger_fc1_b = need("net.language_model.visual.merger.linear_fc1.bias");
    ggml_tensor * merger_fc2_w = need("net.language_model.visual.merger.linear_fc2.weight");
    ggml_tensor * merger_fc2_b = need("net.language_model.visual.merger.linear_fc2.bias");
    if (!merger_norm_w || !merger_norm_b || !merger_fc1_w || !merger_fc1_b ||
        !merger_fc2_w || !merger_fc2_b) {
        return false;
    }
    cosmos3_visual_cuda_set_merger(visual,
                                   merger_norm_w->data,
                                   merger_norm_b->data,
                                   merger_fc1_w->data,
                                   merger_fc1_b->data,
                                   merger_fc2_w->data,
                                   merger_fc2_b->data);
    for (int i = 0; i < 3; ++i) {
        const std::string prefix = "net.language_model.visual.deepstack_merger_list." +
                                   std::to_string(i) + ".";
        ggml_tensor * norm_w = need(prefix + "norm.weight");
        ggml_tensor * norm_b = need(prefix + "norm.bias");
        ggml_tensor * fc1_w = need(prefix + "linear_fc1.weight");
        ggml_tensor * fc1_b = need(prefix + "linear_fc1.bias");
        ggml_tensor * fc2_w = need(prefix + "linear_fc2.weight");
        ggml_tensor * fc2_b = need(prefix + "linear_fc2.bias");
        if (!norm_w || !norm_b || !fc1_w || !fc1_b || !fc2_w || !fc2_b) {
            return false;
        }
        cosmos3_visual_cuda_set_deepstack_merger(visual,
                                                 i,
                                                 norm_w->data,
                                                 norm_b->data,
                                                 fc1_w->data,
                                                 fc1_b->data,
                                                 fc2_w->data,
                                                 fc2_b->data);
    }
    return true;
}

struct W8LinearGpu {
    const int * qweight = nullptr;
    const unsigned short * scales = nullptr;
    int in_features = 0;
    int out_features = 0;
    int qweight_rows = 0;
    int qweight_cols = 0;
    int scale_rows = 0;
    int scale_cols = 0;
};

struct LanguageLayerGpu {
    const unsigned short * input_norm = nullptr;
    const unsigned short * post_norm = nullptr;
    const unsigned short * q_norm = nullptr;
    const unsigned short * k_norm = nullptr;
    W8LinearGpu q_proj;
    W8LinearGpu k_proj;
    W8LinearGpu v_proj;
    W8LinearGpu o_proj;
    W8LinearGpu gate_proj;
    W8LinearGpu up_proj;
    W8LinearGpu down_proj;
};

struct LanguageWeightsGpu {
    const unsigned short * embed_tokens = nullptr;
    int vocab = 0;
    const unsigned short * final_norm = nullptr;
    std::array<LanguageLayerGpu, kLanguageLayers> layers{};
};

struct MotGenLayerGpu {
    const unsigned short * und_input_norm = nullptr;
    const unsigned short * und_post_norm = nullptr;
    const unsigned short * und_q_norm = nullptr;
    const unsigned short * und_k_norm = nullptr;
    const unsigned short * gen_input_norm = nullptr;
    const unsigned short * gen_post_norm = nullptr;
    const unsigned short * gen_q_norm = nullptr;
    const unsigned short * gen_k_norm = nullptr;
    W8LinearGpu und_q_proj;
    W8LinearGpu und_k_proj;
    W8LinearGpu und_v_proj;
    W8LinearGpu und_o_proj;
    W8LinearGpu und_gate_proj;
    W8LinearGpu und_up_proj;
    W8LinearGpu und_down_proj;
    W8LinearGpu gen_q_proj;
    W8LinearGpu gen_k_proj;
    W8LinearGpu gen_v_proj;
    W8LinearGpu gen_o_proj;
    W8LinearGpu gen_gate_proj;
    W8LinearGpu gen_up_proj;
    W8LinearGpu gen_down_proj;
};

struct MotGenWeightsGpu {
    const unsigned short * final_norm = nullptr;
    std::array<MotGenLayerGpu, kLanguageLayers> layers{};
};

struct ActionBridgeWeightsGpu {
    const unsigned short * action2llm_fc = nullptr;
    const unsigned short * action2llm_bias = nullptr;
    const unsigned short * llm2action_fc = nullptr;
    const unsigned short * llm2action_bias = nullptr;
    const unsigned short * action_modality_embed = nullptr;
};

struct MotActionWeightsGpu {
    ActionBridgeWeightsGpu bridge;
    const unsigned short * vae2llm_weight = nullptr;
    const unsigned short * vae2llm_bias = nullptr;
    const unsigned short * llm2vae_weight = nullptr;
    const unsigned short * llm2vae_bias = nullptr;
    const unsigned short * time_mlp0_weight = nullptr;
    const unsigned short * time_mlp0_bias = nullptr;
    const unsigned short * time_mlp2_weight = nullptr;
    const unsigned short * time_mlp2_bias = nullptr;
};

struct WanVaeEncoderWeightsGpu {
    bool available = false;
    int encoder_tensor_count = 0;
    const unsigned short * scale_mean = nullptr;
    const unsigned short * scale_inv_std = nullptr;
    const unsigned short * encoder_conv1_weight = nullptr;
    const unsigned short * encoder_conv1_bias = nullptr;
};

struct WanVaeCudaRuntime {
    enum TimingSlot {
        VAE_T_PATCH = 0,
        VAE_T_CONV1,
        VAE_T_DOWN0_SHORTCUT,
        VAE_T_DOWN0,
        VAE_T_DOWN1_SHORTCUT,
        VAE_T_DOWN1,
        VAE_T_DOWN2_SHORTCUT,
        VAE_T_DOWN2,
        VAE_T_DOWN3,
        VAE_T_MID0,
        VAE_T_MID_ATTN,
        VAE_T_MID2,
        VAE_T_HEAD,
        VAE_T_FINAL_CONV1,
        VAE_T_CLEAN_PACK,
        VAE_T_COUNT,
    };
    bool available = false;
    const GgufGpuResidentModel * model = nullptr;
    WanVaeEncoderWeightsGpu weights;
    unsigned char * image_dev = nullptr;
    size_t image_bytes = 0;
    float * patch_whdc = nullptr;
    float * encoder_conv1_whdc = nullptr;
    float * down0_shortcut_whdc = nullptr;
    float * down0_whdc = nullptr;
    float * down0_tmp_a_whdc = nullptr;
    float * down0_tmp_b_whdc = nullptr;
    float * down1_whdc = nullptr;
    float * down1_shortcut_whdc = nullptr;
    float * down1_spatial_whdc = nullptr;
    float * down1_tmp_a_whdc = nullptr;
    float * down1_tmp_b_whdc = nullptr;
    float * down2_whdc = nullptr;
    float * down2_shortcut_whdc = nullptr;
    float * down2_spatial_whdc = nullptr;
    float * down2_tmp_a_whdc = nullptr;
    float * down2_tmp_b_whdc = nullptr;
    float * down3_whdc = nullptr;
    float * mid0_whdc = nullptr;
    float * mid_attn_whdc = nullptr;
    float * mid_qkv_whdc = nullptr;
    float * mid_scores = nullptr;
    float * mid_q_row = nullptr;
    float * mid_k_row = nullptr;
    float * mid_v_row = nullptr;
    float * mid_attn_row = nullptr;
    float * mid2_whdc = nullptr;
    float * encoder_head_whdc = nullptr;
    float * final_conv1_whdc = nullptr;
    float * clean_vision_condition = nullptr;
    float * debug_prefix = nullptr;
    int debug_prefix_rows = 0;
    int debug_prefix_cols = 0;
    bool patch_ready = false;
    bool conv1_ready = false;
    bool down0_shortcut_ready = false;
    bool down0_ready = false;
    bool down1_shortcut_ready = false;
    bool down1_ready = false;
    bool down2_shortcut_ready = false;
    bool down2_ready = false;
    bool down3_ready = false;
    bool mid0_ready = false;
    bool mid_attn_ready = false;
    bool mid2_ready = false;
    bool head_ready = false;
    bool final_conv1_ready = false;
    bool clean_vision_ready = false;
    bool timing_enabled = false;
    float timing_ms[VAE_T_COUNT] = {};

    static std::vector<int> encode_chunks_for_robolab() {
        std::vector<int> chunks;
        chunks.push_back(1);
        int remain = COSMOS3_WAN_VAE_FRAMES - 1;
        while (remain > 0) {
            const int chunk = std::min(static_cast<int>(COSMOS3_WAN_VAE_ENCODE_CHUNK_FRAMES), remain);
            chunks.push_back(chunk);
            remain -= chunk;
        }
        return chunks;
    }

    static std::vector<int> temporal_down_chunks(int frames) {
        std::vector<int> chunks;
        if (frames <= 0) return chunks;
        chunks.push_back(1);
        int remain = frames - 1;
        while (remain > 0) {
            const int chunk = std::min(static_cast<int>(COSMOS3_WAN_VAE_ENCODE_CHUNK_FRAMES), remain);
            chunks.push_back(chunk);
            remain -= chunk;
        }
        return chunks;
    }

    ~WanVaeCudaRuntime() { release(); }
    WanVaeCudaRuntime() = default;
    WanVaeCudaRuntime(const WanVaeCudaRuntime &) = delete;
    WanVaeCudaRuntime & operator=(const WanVaeCudaRuntime &) = delete;

    void set_timing(bool enabled) { timing_enabled = enabled; }

    void reset_timing() {
        for (float & v : timing_ms) v = 0.0f;
    }

    float timing_value(TimingSlot slot) const {
        return slot >= 0 && slot < VAE_T_COUNT ? timing_ms[slot] : 0.0f;
    }

    void add_timing(TimingSlot slot,
                    const std::chrono::steady_clock::time_point & start) {
        if (!timing_enabled || slot < 0 || slot >= VAE_T_COUNT) return;
        timing_ms[slot] +=
            std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - start).count();
    }

    void release() {
        if (image_dev) cudaFree(image_dev);
        if (patch_whdc) cudaFree(patch_whdc);
        if (encoder_conv1_whdc) cudaFree(encoder_conv1_whdc);
        if (down0_shortcut_whdc) cudaFree(down0_shortcut_whdc);
        if (down0_whdc) cudaFree(down0_whdc);
        if (down0_tmp_a_whdc) cudaFree(down0_tmp_a_whdc);
        if (down0_tmp_b_whdc) cudaFree(down0_tmp_b_whdc);
        if (down1_whdc) cudaFree(down1_whdc);
        if (down1_shortcut_whdc) cudaFree(down1_shortcut_whdc);
        if (down1_spatial_whdc) cudaFree(down1_spatial_whdc);
        if (down1_tmp_a_whdc) cudaFree(down1_tmp_a_whdc);
        if (down1_tmp_b_whdc) cudaFree(down1_tmp_b_whdc);
        if (down2_whdc) cudaFree(down2_whdc);
        if (down2_shortcut_whdc) cudaFree(down2_shortcut_whdc);
        if (down2_spatial_whdc) cudaFree(down2_spatial_whdc);
        if (down2_tmp_a_whdc) cudaFree(down2_tmp_a_whdc);
        if (down2_tmp_b_whdc) cudaFree(down2_tmp_b_whdc);
        if (down3_whdc) cudaFree(down3_whdc);
        if (mid0_whdc) cudaFree(mid0_whdc);
        if (mid_attn_whdc) cudaFree(mid_attn_whdc);
        if (mid_qkv_whdc) cudaFree(mid_qkv_whdc);
        if (mid_scores) cudaFree(mid_scores);
        if (mid_q_row) cudaFree(mid_q_row);
        if (mid_k_row) cudaFree(mid_k_row);
        if (mid_v_row) cudaFree(mid_v_row);
        if (mid_attn_row) cudaFree(mid_attn_row);
        if (mid2_whdc) cudaFree(mid2_whdc);
        if (encoder_head_whdc) cudaFree(encoder_head_whdc);
        if (final_conv1_whdc) cudaFree(final_conv1_whdc);
        if (clean_vision_condition) cudaFree(clean_vision_condition);
        if (debug_prefix) cudaFree(debug_prefix);
        image_dev = nullptr;
        image_bytes = 0;
        patch_whdc = nullptr;
        encoder_conv1_whdc = nullptr;
        down0_shortcut_whdc = nullptr;
        down0_whdc = nullptr;
        down0_tmp_a_whdc = nullptr;
        down0_tmp_b_whdc = nullptr;
        down1_whdc = nullptr;
        down1_shortcut_whdc = nullptr;
        down1_spatial_whdc = nullptr;
        down1_tmp_a_whdc = nullptr;
        down1_tmp_b_whdc = nullptr;
        down2_whdc = nullptr;
        down2_shortcut_whdc = nullptr;
        down2_spatial_whdc = nullptr;
        down2_tmp_a_whdc = nullptr;
        down2_tmp_b_whdc = nullptr;
        down3_whdc = nullptr;
        mid0_whdc = nullptr;
        mid_attn_whdc = nullptr;
        mid_qkv_whdc = nullptr;
        mid_scores = nullptr;
        mid_q_row = nullptr;
        mid_k_row = nullptr;
        mid_v_row = nullptr;
        mid_attn_row = nullptr;
        mid2_whdc = nullptr;
        encoder_head_whdc = nullptr;
        final_conv1_whdc = nullptr;
        clean_vision_condition = nullptr;
        debug_prefix = nullptr;
        debug_prefix_rows = 0;
        debug_prefix_cols = 0;
        patch_ready = false;
        conv1_ready = false;
        down0_shortcut_ready = false;
        down0_ready = false;
        down1_shortcut_ready = false;
        down1_ready = false;
        down2_shortcut_ready = false;
        down2_ready = false;
        down3_ready = false;
        mid0_ready = false;
        mid_attn_ready = false;
        mid2_ready = false;
        head_ready = false;
        final_conv1_ready = false;
        clean_vision_ready = false;
        available = false;
        model = nullptr;
    }

    bool init(const GgufGpuResidentModel & resident_model,
              const WanVaeEncoderWeightsGpu & bound,
              std::string * error) {
        release();
        model = &resident_model;
        weights = bound;
        available = bound.available;
        if (!available) return true;
        const size_t patch_elems =
            static_cast<size_t>(COSMOS3_WAN_VAE_PATCH_W) *
            COSMOS3_WAN_VAE_PATCH_H *
            COSMOS3_WAN_VAE_FRAMES *
            COSMOS3_WAN_VAE_PATCH_CHANNELS;
        if (cudaMalloc(reinterpret_cast<void **>(&patch_whdc),
                       patch_elems * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE patch buffer allocation failed: ") +
                     cudaGetErrorString(st);
            release();
            return false;
        }
        const size_t conv1_elems =
            static_cast<size_t>(COSMOS3_WAN_VAE_PATCH_W) *
            COSMOS3_WAN_VAE_PATCH_H *
            COSMOS3_WAN_VAE_FRAMES *
            COSMOS3_WAN_VAE_CONV1_CHANNELS;
        if (cudaMalloc(reinterpret_cast<void **>(&encoder_conv1_whdc),
                       conv1_elems * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE encoder.conv1 buffer allocation failed: ") +
                     cudaGetErrorString(st);
            release();
            return false;
        }
        const size_t down0_elems =
            static_cast<size_t>(COSMOS3_WAN_VAE_DOWN0_W) *
            COSMOS3_WAN_VAE_DOWN0_H *
            COSMOS3_WAN_VAE_DOWN0_T *
            COSMOS3_WAN_VAE_DOWN0_CHANNELS;
        if (cudaMalloc(reinterpret_cast<void **>(&down0_shortcut_whdc),
                       down0_elems * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE down0 shortcut buffer allocation failed: ") +
                     cudaGetErrorString(st);
            release();
            return false;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&down0_whdc),
                       down0_elems * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE down0 buffer allocation failed: ") +
                     cudaGetErrorString(st);
            release();
            return false;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&down0_tmp_a_whdc),
                       conv1_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&down0_tmp_b_whdc),
                       conv1_elems * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE down0 temp buffer allocation failed: ") +
                     cudaGetErrorString(st);
            release();
            return false;
        }
        const size_t down1_pre_elems =
            static_cast<size_t>(COSMOS3_WAN_VAE_DOWN0_W) *
            COSMOS3_WAN_VAE_DOWN0_H *
            COSMOS3_WAN_VAE_DOWN0_T *
            COSMOS3_WAN_VAE_DOWN1_CHANNELS;
        const size_t down1_spatial_elems =
            static_cast<size_t>(COSMOS3_WAN_VAE_DOWN1_W) *
            COSMOS3_WAN_VAE_DOWN1_H *
            COSMOS3_WAN_VAE_DOWN0_T *
            COSMOS3_WAN_VAE_DOWN1_CHANNELS;
        const size_t down1_elems =
            static_cast<size_t>(COSMOS3_WAN_VAE_DOWN1_W) *
            COSMOS3_WAN_VAE_DOWN1_H *
            COSMOS3_WAN_VAE_DOWN1_T *
            COSMOS3_WAN_VAE_DOWN1_CHANNELS;
        if (cudaMalloc(reinterpret_cast<void **>(&down1_tmp_a_whdc),
                       down1_pre_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&down1_tmp_b_whdc),
                       down1_pre_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&down1_spatial_whdc),
                       down1_spatial_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&down1_shortcut_whdc),
                       down1_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&down1_whdc),
                       down1_elems * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE down1 buffer allocation failed: ") +
                     cudaGetErrorString(st);
            release();
            return false;
        }
        const size_t down2_pre_elems =
            static_cast<size_t>(COSMOS3_WAN_VAE_DOWN1_W) *
            COSMOS3_WAN_VAE_DOWN1_H *
            COSMOS3_WAN_VAE_DOWN1_T *
            COSMOS3_WAN_VAE_DOWN2_CHANNELS;
        const size_t down2_spatial_elems =
            static_cast<size_t>(COSMOS3_WAN_VAE_DOWN2_W) *
            COSMOS3_WAN_VAE_DOWN2_H *
            COSMOS3_WAN_VAE_DOWN1_T *
            COSMOS3_WAN_VAE_DOWN2_CHANNELS;
        const size_t down2_elems =
            static_cast<size_t>(COSMOS3_WAN_VAE_DOWN2_W) *
            COSMOS3_WAN_VAE_DOWN2_H *
            COSMOS3_WAN_VAE_DOWN2_T *
            COSMOS3_WAN_VAE_DOWN2_CHANNELS;
        if (cudaMalloc(reinterpret_cast<void **>(&down2_tmp_a_whdc),
                       down2_pre_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&down2_tmp_b_whdc),
                       down2_pre_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&down2_spatial_whdc),
                       down2_spatial_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&down2_shortcut_whdc),
                       down2_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&down2_whdc),
                       down2_elems * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE down2 buffer allocation failed: ") +
                     cudaGetErrorString(st);
            release();
            return false;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&down3_whdc),
                       down2_elems * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE down3 buffer allocation failed: ") +
                     cudaGetErrorString(st);
            release();
            return false;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&mid0_whdc),
                       down2_elems * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE mid0 buffer allocation failed: ") +
                     cudaGetErrorString(st);
            release();
            return false;
        }
        const size_t mid_elems = down2_elems;
        const size_t mid_qkv_elems = mid_elems * 3;
        const size_t mid_tokens = static_cast<size_t>(COSMOS3_WAN_VAE_DOWN2_W) *
                                  COSMOS3_WAN_VAE_DOWN2_H;
        const size_t mid_scores_elems = static_cast<size_t>(COSMOS3_WAN_VAE_DOWN2_T) *
                                        mid_tokens * mid_tokens;
        if (cudaMalloc(reinterpret_cast<void **>(&mid_attn_whdc),
                       mid_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&mid_qkv_whdc),
                       mid_qkv_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&mid_scores),
                       mid_scores_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&mid_q_row),
                       mid_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&mid_k_row),
                       mid_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&mid_v_row),
                       mid_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&mid_attn_row),
                       mid_elems * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE middle attention buffer allocation failed: ") +
                     cudaGetErrorString(st);
            release();
            return false;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&mid2_whdc),
                       mid_elems * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE mid2 buffer allocation failed: ") +
                     cudaGetErrorString(st);
            release();
            return false;
        }
        const size_t tail_elems = static_cast<size_t>(COSMOS3_WAN_VAE_DOWN2_W) *
                                  COSMOS3_WAN_VAE_DOWN2_H *
                                  COSMOS3_WAN_VAE_DOWN2_T *
                                  (2 * kMotVisionVaeChannels);
        if (cudaMalloc(reinterpret_cast<void **>(&encoder_head_whdc),
                       tail_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&final_conv1_whdc),
                       tail_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&clean_vision_condition),
                       static_cast<size_t>(kMotVisionConditionTokens) *
                           kMotVisionPatchDim * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE tail buffer allocation failed: ") +
                     cudaGetErrorString(st);
            release();
            return false;
        }
        const std::vector<int> chunks = encode_chunks_for_robolab();
        std::string chunk_msg;
        for (size_t i = 0; i < chunks.size(); ++i) {
            if (i) chunk_msg += ",";
            chunk_msg += std::to_string(chunks[i]);
        }
        std::fprintf(stderr,
                     "vla(cosmos3): Wan2.2 VAE encoder chunk plan=[%s]\n",
                     chunk_msg.c_str());
        return true;
    }

    bool forward_robolab_image(const WamTensorView & image,
                               cudaStream_t stream,
                               std::string * error) {
        if (!available) return true;
        reset_timing();
        const auto t_stage = std::chrono::steady_clock::now();
        if (image.dtype != WamDType::U8 || image.shape.size() != 3 ||
            image.shape[0] <= 0 || image.shape[1] <= 0 || image.shape[2] != 3) {
            *error = "Cosmos3 Wan VAE image input must be U8 [H,W,3]";
            return false;
        }
        const size_t bytes = image.bytes;
        if (bytes != static_cast<size_t>(image.shape[0]) *
                         static_cast<size_t>(image.shape[1]) * 3u) {
            *error = "Cosmos3 Wan VAE image byte size mismatch";
            return false;
        }
        if (bytes > image_bytes) {
            if (image_dev) cudaFree(image_dev);
            image_dev = nullptr;
            image_bytes = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&image_dev), bytes) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE image buffer allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            image_bytes = bytes;
        }
        if (cudaMemcpyAsync(image_dev, image.data, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 Wan VAE image upload failed: ") +
                     cudaGetErrorString(st);
            return false;
        }
        if (cosmos3_wan_vae_robolab_image_to_patch_whdc_f32(
                image_dev,
                static_cast<int>(image.shape[0]),
                static_cast<int>(image.shape[1]),
                patch_whdc,
                stream) != 0) {
            *error = "Cosmos3 Wan VAE patchify CUDA launch failed";
            return false;
        }
        patch_ready = true;
        conv1_ready = false;
        down0_shortcut_ready = false;
        down0_ready = false;
        down1_shortcut_ready = false;
        down1_ready = false;
        down2_shortcut_ready = false;
        down2_ready = false;
        down3_ready = false;
        mid0_ready = false;
        mid_attn_ready = false;
        mid2_ready = false;
        head_ready = false;
        final_conv1_ready = false;
        clean_vision_ready = false;
        add_timing(VAE_T_PATCH, t_stage);
        return true;
    }

    bool ensure_encoder_conv1(cudaStream_t stream,
                              std::string * error) {
        if (!available) return true;
        if (conv1_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!patch_ready || !patch_whdc || !encoder_conv1_whdc) {
            *error = "Cosmos3 Wan VAE encoder.conv1 requested before patch input is ready";
            return false;
        }
        if (cosmos3_wan_vae_encoder_conv1_whdc_f32(patch_whdc,
                                                   weights.encoder_conv1_weight,
                                                   weights.encoder_conv1_bias,
                                                   encoder_conv1_whdc,
                                                   stream) != 0) {
            *error = "Cosmos3 Wan VAE encoder.conv1 CUDA launch failed";
            return false;
        }
        conv1_ready = true;
        add_timing(VAE_T_CONV1, t_stage);
        return true;
    }

    bool ensure_down0_shortcut(cudaStream_t stream,
                               std::string * error) {
        if (!available) return true;
        if (down0_shortcut_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!ensure_encoder_conv1(stream, error)) return false;
        if (cosmos3_wan_vae_down0_avg_shortcut_whdc_f32(encoder_conv1_whdc,
                                                        down0_shortcut_whdc,
                                                        stream) != 0) {
            *error = "Cosmos3 Wan VAE down0 AvgDown3D shortcut CUDA launch failed";
            return false;
        }
        down0_shortcut_ready = true;
        add_timing(VAE_T_DOWN0_SHORTCUT, t_stage);
        return true;
    }

    bool ensure_down0(cudaStream_t stream,
                      std::string * error) {
        if (!available) return true;
        if (down0_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!model) {
            *error = "Cosmos3 Wan VAE down0 has no resident model";
            return false;
        }
        if (!ensure_down0_shortcut(stream, error)) return false;

        auto need_bf16 = [&](const char * name) -> const unsigned short * {
            ggml_tensor * t = model->tensor_source(name);
            if (!t || t->type != GGML_TYPE_BF16 || !t->data) {
                *error = std::string("missing BF16 Wan VAE down0 tensor: ") + name;
                return nullptr;
            }
            return static_cast<const unsigned short *>(t->data);
        };
        struct ResPtrs {
            const unsigned short * n1 = nullptr;
            const unsigned short * c1w = nullptr;
            const unsigned short * c1b = nullptr;
            const unsigned short * n2 = nullptr;
            const unsigned short * c2w = nullptr;
            const unsigned short * c2b = nullptr;
        };
        auto make_res = [&](const char * p) -> ResPtrs {
            return {
                need_bf16((std::string(p) + ".residual.0.gamma").c_str()),
                need_bf16((std::string(p) + ".residual.2.weight").c_str()),
                need_bf16((std::string(p) + ".residual.2.bias").c_str()),
                need_bf16((std::string(p) + ".residual.3.gamma").c_str()),
                need_bf16((std::string(p) + ".residual.6.weight").c_str()),
                need_bf16((std::string(p) + ".residual.6.bias").c_str()),
            };
        };
        const char * base = "wan22_vae.encoder.downsamples.0.downsamples.";
        ResPtrs r0 = make_res("wan22_vae.encoder.downsamples.0.downsamples.0");
        if (!error->empty()) return false;
        ResPtrs r1 = make_res("wan22_vae.encoder.downsamples.0.downsamples.1");
        if (!error->empty()) return false;
        const unsigned short * down_w = need_bf16("wan22_vae.encoder.downsamples.0.downsamples.2.resample.1.weight");
        const unsigned short * down_b = need_bf16("wan22_vae.encoder.downsamples.0.downsamples.2.resample.1.bias");
        if (!down_w || !down_b) return false;
        (void) base;

        constexpr int W = COSMOS3_WAN_VAE_PATCH_W;
        constexpr int H = COSMOS3_WAN_VAE_PATCH_H;
        constexpr int T = COSMOS3_WAN_VAE_FRAMES;
        constexpr int Cc = COSMOS3_WAN_VAE_CONV1_CHANNELS;
        const size_t full_elems = static_cast<size_t>(W) * H * T * Cc;
        const size_t down_elems = static_cast<size_t>(COSMOS3_WAN_VAE_DOWN0_W) *
                                  COSMOS3_WAN_VAE_DOWN0_H *
                                  COSMOS3_WAN_VAE_DOWN0_T *
                                  COSMOS3_WAN_VAE_DOWN0_CHANNELS;
        auto apply_res = [&](float * x, const ResPtrs & r) -> bool {
            if (cosmos3_wan_vae_norm_silu_whdc_f32(x, r.n1, down0_tmp_a_whdc, W, H, T, Cc, stream) != 0 ||
                cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down0_tmp_a_whdc, r.c1w, r.c1b,
                                                           down0_tmp_b_whdc, W, H, T, Cc, Cc, stream) != 0 ||
                cosmos3_wan_vae_norm_silu_whdc_f32(down0_tmp_b_whdc, r.n2, down0_tmp_a_whdc, W, H, T, Cc, stream) != 0 ||
                cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down0_tmp_a_whdc, r.c2w, r.c2b,
                                                           down0_tmp_b_whdc, W, H, T, Cc, Cc, stream) != 0 ||
                cosmos3_wan_vae_add_whdc_f32(down0_tmp_b_whdc, x, x, full_elems, stream) != 0) {
                return false;
            }
            return true;
        };
        if (!apply_res(encoder_conv1_whdc, r0) ||
            !apply_res(encoder_conv1_whdc, r1) ||
            cosmos3_wan_vae_spatial_downsample2d_whdc_f32(encoder_conv1_whdc, down_w, down_b,
                                                          down0_whdc, W, H, T, Cc, stream) != 0 ||
            cosmos3_wan_vae_add_whdc_f32(down0_whdc, down0_shortcut_whdc,
                                         down0_whdc, down_elems, stream) != 0) {
            *error = "Cosmos3 Wan VAE down0 CUDA primitive execution failed";
            return false;
        }
        down0_ready = true;
        add_timing(VAE_T_DOWN0, t_stage);
        return true;
    }

    bool ensure_down1_shortcut(cudaStream_t stream,
                               std::string * error) {
        if (!available) return true;
        if (down1_shortcut_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!ensure_down0(stream, error)) return false;
        if (cosmos3_wan_vae_avg_down3d_whdc_f32(down0_whdc,
                                                down1_shortcut_whdc,
                                                COSMOS3_WAN_VAE_DOWN0_W,
                                                COSMOS3_WAN_VAE_DOWN0_H,
                                                COSMOS3_WAN_VAE_DOWN0_T,
                                                COSMOS3_WAN_VAE_DOWN0_CHANNELS,
                                                COSMOS3_WAN_VAE_DOWN1_CHANNELS,
                                                2,
                                                2,
                                                stream) != 0) {
            *error = "Cosmos3 Wan VAE down1 AvgDown3D shortcut CUDA launch failed";
            return false;
        }
        down1_shortcut_ready = true;
        add_timing(VAE_T_DOWN1_SHORTCUT, t_stage);
        return true;
    }

    bool ensure_down1(cudaStream_t stream,
                      std::string * error) {
        if (!available) return true;
        if (down1_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!model) {
            *error = "Cosmos3 Wan VAE down1 has no resident model";
            return false;
        }
        if (!ensure_down1_shortcut(stream, error)) return false;

        auto need_bf16 = [&](const char * name) -> const unsigned short * {
            ggml_tensor * t = model->tensor_source(name);
            if (!t || t->type != GGML_TYPE_BF16 || !t->data) {
                *error = std::string("missing BF16 Wan VAE down1 tensor: ") + name;
                return nullptr;
            }
            return static_cast<const unsigned short *>(t->data);
        };
        struct ResPtrs {
            const unsigned short * n1 = nullptr;
            const unsigned short * c1w = nullptr;
            const unsigned short * c1b = nullptr;
            const unsigned short * n2 = nullptr;
            const unsigned short * c2w = nullptr;
            const unsigned short * c2b = nullptr;
            const unsigned short * sw = nullptr;
            const unsigned short * sb = nullptr;
        };
        auto make_res = [&](const char * p, bool shortcut) -> ResPtrs {
            ResPtrs r{
                need_bf16((std::string(p) + ".residual.0.gamma").c_str()),
                need_bf16((std::string(p) + ".residual.2.weight").c_str()),
                need_bf16((std::string(p) + ".residual.2.bias").c_str()),
                need_bf16((std::string(p) + ".residual.3.gamma").c_str()),
                need_bf16((std::string(p) + ".residual.6.weight").c_str()),
                need_bf16((std::string(p) + ".residual.6.bias").c_str()),
                nullptr,
                nullptr,
            };
            if (shortcut) {
                r.sw = need_bf16((std::string(p) + ".shortcut.weight").c_str());
                r.sb = need_bf16((std::string(p) + ".shortcut.bias").c_str());
            }
            return r;
        };
        ResPtrs r0 = make_res("wan22_vae.encoder.downsamples.1.downsamples.0", true);
        if (!error->empty()) return false;
        ResPtrs r1 = make_res("wan22_vae.encoder.downsamples.1.downsamples.1", false);
        if (!error->empty()) return false;
        const unsigned short * down_w =
            need_bf16("wan22_vae.encoder.downsamples.1.downsamples.2.resample.1.weight");
        const unsigned short * down_b =
            need_bf16("wan22_vae.encoder.downsamples.1.downsamples.2.resample.1.bias");
        const unsigned short * time_w =
            need_bf16("wan22_vae.encoder.downsamples.1.downsamples.2.time_conv.weight");
        const unsigned short * time_b =
            need_bf16("wan22_vae.encoder.downsamples.1.downsamples.2.time_conv.bias");
        if (!down_w || !down_b || !time_w || !time_b) return false;

        constexpr int W = COSMOS3_WAN_VAE_DOWN0_W;
        constexpr int H = COSMOS3_WAN_VAE_DOWN0_H;
        constexpr int T = COSMOS3_WAN_VAE_DOWN0_T;
        constexpr int Cin0 = COSMOS3_WAN_VAE_DOWN0_CHANNELS;
        constexpr int Cout = COSMOS3_WAN_VAE_DOWN1_CHANNELS;
        const size_t down1_pre_elems = static_cast<size_t>(W) * H * T * Cout;
        auto apply_res_160_to_320 = [&]() -> bool {
            if (cosmos3_wan_vae_norm_silu_whdc_f32(down0_whdc, r0.n1,
                                                   down0_tmp_a_whdc, W, H, T, Cin0, stream) != 0 ||
                cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down0_tmp_a_whdc, r0.c1w, r0.c1b,
                                                           down1_tmp_a_whdc, W, H, T, Cin0, Cout, stream) != 0 ||
                cosmos3_wan_vae_norm_silu_whdc_f32(down1_tmp_a_whdc, r0.n2,
                                                   down1_tmp_b_whdc, W, H, T, Cout, stream) != 0 ||
                cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down1_tmp_b_whdc, r0.c2w, r0.c2b,
                                                           down1_tmp_a_whdc, W, H, T, Cout, Cout, stream) != 0 ||
                cosmos3_wan_vae_conv1x1x1_whdc_f32(down0_whdc, r0.sw, r0.sb,
                                                   down1_tmp_b_whdc, W, H, T, Cin0, Cout, stream) != 0 ||
                cosmos3_wan_vae_add_whdc_f32(down1_tmp_a_whdc, down1_tmp_b_whdc,
                                             down1_tmp_a_whdc, down1_pre_elems, stream) != 0) {
                return false;
            }
            return true;
        };
        auto apply_res_320 = [&]() -> bool {
            if (cosmos3_wan_vae_norm_silu_whdc_f32(down1_tmp_a_whdc, r1.n1,
                                                   down1_tmp_b_whdc, W, H, T, Cout, stream) != 0 ||
                cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down1_tmp_b_whdc, r1.c1w, r1.c1b,
                                                           down0_tmp_a_whdc, W, H, T, Cout, Cout, stream) != 0 ||
                cosmos3_wan_vae_norm_silu_whdc_f32(down0_tmp_a_whdc, r1.n2,
                                                   down1_tmp_b_whdc, W, H, T, Cout, stream) != 0 ||
                cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down1_tmp_b_whdc, r1.c2w, r1.c2b,
                                                           down0_tmp_a_whdc, W, H, T, Cout, Cout, stream) != 0 ||
                cosmos3_wan_vae_add_whdc_f32(down0_tmp_a_whdc, down1_tmp_a_whdc,
                                             down1_tmp_a_whdc, down1_pre_elems, stream) != 0) {
                return false;
            }
            return true;
        };
        const std::vector<int> chunks = temporal_down_chunks(T);
        if (!apply_res_160_to_320() ||
            !apply_res_320() ||
            cosmos3_wan_vae_spatial_downsample2d_whdc_f32(down1_tmp_a_whdc, down_w, down_b,
                                                          down1_spatial_whdc, W, H, T, Cout, stream) != 0 ||
            cosmos3_wan_vae_downsample3d_time_whdc_f32(down1_spatial_whdc, time_w, time_b,
                                                       down1_whdc,
                                                       COSMOS3_WAN_VAE_DOWN1_W,
                                                       COSMOS3_WAN_VAE_DOWN1_H,
                                                       COSMOS3_WAN_VAE_DOWN0_T,
                                                       Cout,
                                                       chunks.data(),
                                                       static_cast<int>(chunks.size()),
                                                       stream) != 0 ||
            cosmos3_wan_vae_add_whdc_f32(down1_whdc, down1_shortcut_whdc, down1_whdc,
                                         static_cast<size_t>(COSMOS3_WAN_VAE_DOWN1_W) *
                                             COSMOS3_WAN_VAE_DOWN1_H *
                                             COSMOS3_WAN_VAE_DOWN1_T *
                                             COSMOS3_WAN_VAE_DOWN1_CHANNELS,
                                         stream) != 0) {
            *error = "Cosmos3 Wan VAE down1 CUDA primitive execution failed";
            return false;
        }
        down1_ready = true;
        add_timing(VAE_T_DOWN1, t_stage);
        return true;
    }

    bool ensure_down2_shortcut(cudaStream_t stream,
                               std::string * error) {
        if (!available) return true;
        if (down2_shortcut_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!ensure_down1(stream, error)) return false;
        if (cosmos3_wan_vae_avg_down3d_whdc_f32(down1_whdc,
                                                down2_shortcut_whdc,
                                                COSMOS3_WAN_VAE_DOWN1_W,
                                                COSMOS3_WAN_VAE_DOWN1_H,
                                                COSMOS3_WAN_VAE_DOWN1_T,
                                                COSMOS3_WAN_VAE_DOWN1_CHANNELS,
                                                COSMOS3_WAN_VAE_DOWN2_CHANNELS,
                                                2,
                                                2,
                                                stream) != 0) {
            *error = "Cosmos3 Wan VAE down2 AvgDown3D shortcut CUDA launch failed";
            return false;
        }
        down2_shortcut_ready = true;
        add_timing(VAE_T_DOWN2_SHORTCUT, t_stage);
        return true;
    }

    bool ensure_down2(cudaStream_t stream,
                      std::string * error) {
        if (!available) return true;
        if (down2_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!model) {
            *error = "Cosmos3 Wan VAE down2 has no resident model";
            return false;
        }
        if (!ensure_down2_shortcut(stream, error)) return false;

        auto need_bf16 = [&](const char * name) -> const unsigned short * {
            ggml_tensor * t = model->tensor_source(name);
            if (!t || t->type != GGML_TYPE_BF16 || !t->data) {
                *error = std::string("missing BF16 Wan VAE down2 tensor: ") + name;
                return nullptr;
            }
            return static_cast<const unsigned short *>(t->data);
        };
        struct ResPtrs {
            const unsigned short * n1 = nullptr;
            const unsigned short * c1w = nullptr;
            const unsigned short * c1b = nullptr;
            const unsigned short * n2 = nullptr;
            const unsigned short * c2w = nullptr;
            const unsigned short * c2b = nullptr;
            const unsigned short * sw = nullptr;
            const unsigned short * sb = nullptr;
        };
        auto make_res = [&](const char * p, bool shortcut) -> ResPtrs {
            ResPtrs r{
                need_bf16((std::string(p) + ".residual.0.gamma").c_str()),
                need_bf16((std::string(p) + ".residual.2.weight").c_str()),
                need_bf16((std::string(p) + ".residual.2.bias").c_str()),
                need_bf16((std::string(p) + ".residual.3.gamma").c_str()),
                need_bf16((std::string(p) + ".residual.6.weight").c_str()),
                need_bf16((std::string(p) + ".residual.6.bias").c_str()),
                nullptr,
                nullptr,
            };
            if (shortcut) {
                r.sw = need_bf16((std::string(p) + ".shortcut.weight").c_str());
                r.sb = need_bf16((std::string(p) + ".shortcut.bias").c_str());
            }
            return r;
        };
        ResPtrs r0 = make_res("wan22_vae.encoder.downsamples.2.downsamples.0", true);
        if (!error->empty()) return false;
        ResPtrs r1 = make_res("wan22_vae.encoder.downsamples.2.downsamples.1", false);
        if (!error->empty()) return false;
        const unsigned short * down_w =
            need_bf16("wan22_vae.encoder.downsamples.2.downsamples.2.resample.1.weight");
        const unsigned short * down_b =
            need_bf16("wan22_vae.encoder.downsamples.2.downsamples.2.resample.1.bias");
        const unsigned short * time_w =
            need_bf16("wan22_vae.encoder.downsamples.2.downsamples.2.time_conv.weight");
        const unsigned short * time_b =
            need_bf16("wan22_vae.encoder.downsamples.2.downsamples.2.time_conv.bias");
        if (!down_w || !down_b || !time_w || !time_b) return false;

        constexpr int W = COSMOS3_WAN_VAE_DOWN1_W;
        constexpr int H = COSMOS3_WAN_VAE_DOWN1_H;
        constexpr int T = COSMOS3_WAN_VAE_DOWN1_T;
        constexpr int Cin = COSMOS3_WAN_VAE_DOWN1_CHANNELS;
        constexpr int Cout = COSMOS3_WAN_VAE_DOWN2_CHANNELS;
        const size_t pre_elems = static_cast<size_t>(W) * H * T * Cout;
        auto apply_res_320_to_640 = [&]() -> bool {
            if (cosmos3_wan_vae_norm_silu_whdc_f32(down1_whdc, r0.n1,
                                                   down1_tmp_a_whdc, W, H, T, Cin, stream) != 0 ||
                cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down1_tmp_a_whdc, r0.c1w, r0.c1b,
                                                           down2_tmp_a_whdc, W, H, T, Cin, Cout, stream) != 0 ||
                cosmos3_wan_vae_norm_silu_whdc_f32(down2_tmp_a_whdc, r0.n2,
                                                   down2_tmp_b_whdc, W, H, T, Cout, stream) != 0 ||
                cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down2_tmp_b_whdc, r0.c2w, r0.c2b,
                                                           down2_tmp_a_whdc, W, H, T, Cout, Cout, stream) != 0 ||
                cosmos3_wan_vae_conv1x1x1_whdc_f32(down1_whdc, r0.sw, r0.sb,
                                                   down2_tmp_b_whdc, W, H, T, Cin, Cout, stream) != 0 ||
                cosmos3_wan_vae_add_whdc_f32(down2_tmp_a_whdc, down2_tmp_b_whdc,
                                             down2_tmp_a_whdc, pre_elems, stream) != 0) {
                return false;
            }
            return true;
        };
        auto apply_res_640 = [&]() -> bool {
            if (cosmos3_wan_vae_norm_silu_whdc_f32(down2_tmp_a_whdc, r1.n1,
                                                   down2_tmp_b_whdc, W, H, T, Cout, stream) != 0 ||
                cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down2_tmp_b_whdc, r1.c1w, r1.c1b,
                                                           down1_tmp_a_whdc, W, H, T, Cout, Cout, stream) != 0 ||
                cosmos3_wan_vae_norm_silu_whdc_f32(down1_tmp_a_whdc, r1.n2,
                                                   down2_tmp_b_whdc, W, H, T, Cout, stream) != 0 ||
                cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down2_tmp_b_whdc, r1.c2w, r1.c2b,
                                                           down1_tmp_a_whdc, W, H, T, Cout, Cout, stream) != 0 ||
                cosmos3_wan_vae_add_whdc_f32(down1_tmp_a_whdc, down2_tmp_a_whdc,
                                             down2_tmp_a_whdc, pre_elems, stream) != 0) {
                return false;
            }
            return true;
        };
        const std::vector<int> chunks = temporal_down_chunks(T);
        if (!apply_res_320_to_640() ||
            !apply_res_640() ||
            cosmos3_wan_vae_spatial_downsample2d_whdc_f32(down2_tmp_a_whdc, down_w, down_b,
                                                          down2_spatial_whdc, W, H, T, Cout, stream) != 0 ||
            cosmos3_wan_vae_downsample3d_time_whdc_f32(down2_spatial_whdc, time_w, time_b,
                                                       down2_whdc,
                                                       COSMOS3_WAN_VAE_DOWN2_W,
                                                       COSMOS3_WAN_VAE_DOWN2_H,
                                                       T,
                                                       Cout,
                                                       chunks.data(),
                                                       static_cast<int>(chunks.size()),
                                                       stream) != 0 ||
            cosmos3_wan_vae_add_whdc_f32(down2_whdc, down2_shortcut_whdc, down2_whdc,
                                         static_cast<size_t>(COSMOS3_WAN_VAE_DOWN2_W) *
                                             COSMOS3_WAN_VAE_DOWN2_H *
                                             COSMOS3_WAN_VAE_DOWN2_T *
                                             COSMOS3_WAN_VAE_DOWN2_CHANNELS,
                                         stream) != 0) {
            *error = "Cosmos3 Wan VAE down2 CUDA primitive execution failed";
            return false;
        }
        down2_ready = true;
        add_timing(VAE_T_DOWN2, t_stage);
        return true;
    }

    bool ensure_down3(cudaStream_t stream,
                      std::string * error) {
        if (!available) return true;
        if (down3_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!model) {
            *error = "Cosmos3 Wan VAE down3 has no resident model";
            return false;
        }
        if (!ensure_down2(stream, error)) return false;

        auto need_bf16 = [&](const char * name) -> const unsigned short * {
            ggml_tensor * t = model->tensor_source(name);
            if (!t || t->type != GGML_TYPE_BF16 || !t->data) {
                *error = std::string("missing BF16 Wan VAE down3 tensor: ") + name;
                return nullptr;
            }
            return static_cast<const unsigned short *>(t->data);
        };
        struct ResPtrs {
            const unsigned short * n1 = nullptr;
            const unsigned short * c1w = nullptr;
            const unsigned short * c1b = nullptr;
            const unsigned short * n2 = nullptr;
            const unsigned short * c2w = nullptr;
            const unsigned short * c2b = nullptr;
        };
        auto make_res = [&](const char * p) -> ResPtrs {
            return {
                need_bf16((std::string(p) + ".residual.0.gamma").c_str()),
                need_bf16((std::string(p) + ".residual.2.weight").c_str()),
                need_bf16((std::string(p) + ".residual.2.bias").c_str()),
                need_bf16((std::string(p) + ".residual.3.gamma").c_str()),
                need_bf16((std::string(p) + ".residual.6.weight").c_str()),
                need_bf16((std::string(p) + ".residual.6.bias").c_str()),
            };
        };
        ResPtrs r0 = make_res("wan22_vae.encoder.downsamples.3.downsamples.0");
        if (!error->empty()) return false;
        ResPtrs r1 = make_res("wan22_vae.encoder.downsamples.3.downsamples.1");
        if (!error->empty()) return false;

        constexpr int W = COSMOS3_WAN_VAE_DOWN2_W;
        constexpr int H = COSMOS3_WAN_VAE_DOWN2_H;
        constexpr int T = COSMOS3_WAN_VAE_DOWN2_T;
        constexpr int Cc = COSMOS3_WAN_VAE_DOWN2_CHANNELS;
        const size_t elems = static_cast<size_t>(W) * H * T * Cc;
        auto apply_res = [&](const float * input, float * output, const ResPtrs & r) -> bool {
            if (cosmos3_wan_vae_norm_silu_whdc_f32(input, r.n1,
                                                   down2_tmp_b_whdc, W, H, T, Cc, stream) != 0 ||
                cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down2_tmp_b_whdc, r.c1w, r.c1b,
                                                           down2_tmp_a_whdc, W, H, T, Cc, Cc, stream) != 0 ||
                cosmos3_wan_vae_norm_silu_whdc_f32(down2_tmp_a_whdc, r.n2,
                                                   down2_tmp_b_whdc, W, H, T, Cc, stream) != 0 ||
                cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down2_tmp_b_whdc, r.c2w, r.c2b,
                                                           down2_tmp_a_whdc, W, H, T, Cc, Cc, stream) != 0 ||
                cosmos3_wan_vae_add_whdc_f32(down2_tmp_a_whdc, input, output, elems, stream) != 0) {
                return false;
            }
            return true;
        };
        if (!apply_res(down2_whdc, down3_whdc, r0) ||
            !apply_res(down3_whdc, down3_whdc, r1)) {
            *error = "Cosmos3 Wan VAE down3 CUDA primitive execution failed";
            return false;
        }
        if (cosmos3_wan_vae_add_whdc_f32(down3_whdc, down2_whdc, down3_whdc,
                                         elems, stream) != 0) {
            *error = "Cosmos3 Wan VAE down3 outer shortcut CUDA execution failed";
            return false;
        }
        down3_ready = true;
        add_timing(VAE_T_DOWN3, t_stage);
        return true;
    }

    bool ensure_mid0(cudaStream_t stream,
                     std::string * error) {
        if (!available) return true;
        if (mid0_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!model) {
            *error = "Cosmos3 Wan VAE mid0 has no resident model";
            return false;
        }
        if (!ensure_down3(stream, error)) return false;

        auto need_bf16 = [&](const char * name) -> const unsigned short * {
            ggml_tensor * t = model->tensor_source(name);
            if (!t || t->type != GGML_TYPE_BF16 || !t->data) {
                *error = std::string("missing BF16 Wan VAE mid0 tensor: ") + name;
                return nullptr;
            }
            return static_cast<const unsigned short *>(t->data);
        };
        const char * p = "wan22_vae.encoder.middle.0";
        const unsigned short * n1 = need_bf16((std::string(p) + ".residual.0.gamma").c_str());
        const unsigned short * c1w = need_bf16((std::string(p) + ".residual.2.weight").c_str());
        const unsigned short * c1b = need_bf16((std::string(p) + ".residual.2.bias").c_str());
        const unsigned short * n2 = need_bf16((std::string(p) + ".residual.3.gamma").c_str());
        const unsigned short * c2w = need_bf16((std::string(p) + ".residual.6.weight").c_str());
        const unsigned short * c2b = need_bf16((std::string(p) + ".residual.6.bias").c_str());
        if (!n1 || !c1w || !c1b || !n2 || !c2w || !c2b) return false;

        constexpr int W = COSMOS3_WAN_VAE_DOWN2_W;
        constexpr int H = COSMOS3_WAN_VAE_DOWN2_H;
        constexpr int T = COSMOS3_WAN_VAE_DOWN2_T;
        constexpr int Cc = COSMOS3_WAN_VAE_DOWN2_CHANNELS;
        const size_t elems = static_cast<size_t>(W) * H * T * Cc;
        if (cosmos3_wan_vae_norm_silu_whdc_f32(down3_whdc, n1,
                                               down2_tmp_b_whdc, W, H, T, Cc, stream) != 0 ||
            cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down2_tmp_b_whdc, c1w, c1b,
                                                       down2_tmp_a_whdc, W, H, T, Cc, Cc, stream) != 0 ||
            cosmos3_wan_vae_norm_silu_whdc_f32(down2_tmp_a_whdc, n2,
                                               down2_tmp_b_whdc, W, H, T, Cc, stream) != 0 ||
            cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down2_tmp_b_whdc, c2w, c2b,
                                                       down2_tmp_a_whdc, W, H, T, Cc, Cc, stream) != 0 ||
            cosmos3_wan_vae_add_whdc_f32(down2_tmp_a_whdc, down3_whdc,
                                         mid0_whdc, elems, stream) != 0) {
            *error = "Cosmos3 Wan VAE mid0 CUDA primitive execution failed";
            return false;
        }
        mid0_ready = true;
        add_timing(VAE_T_MID0, t_stage);
        return true;
    }

    bool ensure_mid_attn(cudaStream_t stream,
                         std::string * error) {
        if (!available) return true;
        if (mid_attn_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!model) {
            *error = "Cosmos3 Wan VAE mid attention has no resident model";
            return false;
        }
        if (!ensure_mid0(stream, error)) return false;

        auto need_bf16 = [&](const char * name) -> const unsigned short * {
            ggml_tensor * t = model->tensor_source(name);
            if (!t || t->type != GGML_TYPE_BF16 || !t->data) {
                *error = std::string("missing BF16 Wan VAE middle attention tensor: ") + name;
                return nullptr;
            }
            return static_cast<const unsigned short *>(t->data);
        };
        const unsigned short * norm = need_bf16("wan22_vae.encoder.middle.1.norm.gamma");
        const unsigned short * qkv_w = need_bf16("wan22_vae.encoder.middle.1.to_qkv.weight");
        const unsigned short * qkv_b = need_bf16("wan22_vae.encoder.middle.1.to_qkv.bias");
        const unsigned short * proj_w = need_bf16("wan22_vae.encoder.middle.1.proj.weight");
        const unsigned short * proj_b = need_bf16("wan22_vae.encoder.middle.1.proj.bias");
        if (!norm || !qkv_w || !qkv_b || !proj_w || !proj_b) return false;

        constexpr int W = COSMOS3_WAN_VAE_DOWN2_W;
        constexpr int H = COSMOS3_WAN_VAE_DOWN2_H;
        constexpr int T = COSMOS3_WAN_VAE_DOWN2_T;
        constexpr int Cc = COSMOS3_WAN_VAE_DOWN2_CHANNELS;
        const size_t elems = static_cast<size_t>(W) * H * T * Cc;
        if (cosmos3_wan_vae_rms_norm_whdc_f32(mid0_whdc, norm,
                                              down2_tmp_a_whdc, W, H, T, Cc, stream) != 0 ||
            cosmos3_wan_vae_conv1x1x1_whdc_f32(down2_tmp_a_whdc, qkv_w, qkv_b,
                                               mid_qkv_whdc, W, H, T, Cc, 3 * Cc, stream) != 0 ||
            cosmos3_wan_vae_mid_attention_f32(mid_qkv_whdc,
                                              mid_scores,
                                              mid_q_row,
                                              mid_k_row,
                                              mid_v_row,
                                              mid_attn_row,
                                              down2_tmp_a_whdc,
                                              W,
                                              H,
                                              T,
                                              Cc,
                                              stream) != 0 ||
            cosmos3_wan_vae_conv1x1x1_whdc_f32(down2_tmp_a_whdc, proj_w, proj_b,
                                               down2_tmp_b_whdc, W, H, T, Cc, Cc, stream) != 0 ||
            cosmos3_wan_vae_add_whdc_f32(down2_tmp_b_whdc, mid0_whdc,
                                         mid_attn_whdc, elems, stream) != 0) {
            *error = "Cosmos3 Wan VAE middle attention CUDA primitive execution failed";
            return false;
        }
        mid_attn_ready = true;
        add_timing(VAE_T_MID_ATTN, t_stage);
        return true;
    }

    bool ensure_mid2(cudaStream_t stream,
                     std::string * error) {
        if (!available) return true;
        if (mid2_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!model) {
            *error = "Cosmos3 Wan VAE mid2 has no resident model";
            return false;
        }
        if (!ensure_mid_attn(stream, error)) return false;

        auto need_bf16 = [&](const char * name) -> const unsigned short * {
            ggml_tensor * t = model->tensor_source(name);
            if (!t || t->type != GGML_TYPE_BF16 || !t->data) {
                *error = std::string("missing BF16 Wan VAE mid2 tensor: ") + name;
                return nullptr;
            }
            return static_cast<const unsigned short *>(t->data);
        };
        const char * p = "wan22_vae.encoder.middle.2";
        const unsigned short * n1 = need_bf16((std::string(p) + ".residual.0.gamma").c_str());
        const unsigned short * c1w = need_bf16((std::string(p) + ".residual.2.weight").c_str());
        const unsigned short * c1b = need_bf16((std::string(p) + ".residual.2.bias").c_str());
        const unsigned short * n2 = need_bf16((std::string(p) + ".residual.3.gamma").c_str());
        const unsigned short * c2w = need_bf16((std::string(p) + ".residual.6.weight").c_str());
        const unsigned short * c2b = need_bf16((std::string(p) + ".residual.6.bias").c_str());
        if (!n1 || !c1w || !c1b || !n2 || !c2w || !c2b) return false;

        constexpr int W = COSMOS3_WAN_VAE_DOWN2_W;
        constexpr int H = COSMOS3_WAN_VAE_DOWN2_H;
        constexpr int T = COSMOS3_WAN_VAE_DOWN2_T;
        constexpr int Cc = COSMOS3_WAN_VAE_DOWN2_CHANNELS;
        const size_t elems = static_cast<size_t>(W) * H * T * Cc;
        if (cosmos3_wan_vae_norm_silu_whdc_f32(mid_attn_whdc, n1,
                                               down2_tmp_b_whdc, W, H, T, Cc, stream) != 0 ||
            cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down2_tmp_b_whdc, c1w, c1b,
                                                       down2_tmp_a_whdc, W, H, T, Cc, Cc, stream) != 0 ||
            cosmos3_wan_vae_norm_silu_whdc_f32(down2_tmp_a_whdc, n2,
                                               down2_tmp_b_whdc, W, H, T, Cc, stream) != 0 ||
            cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down2_tmp_b_whdc, c2w, c2b,
                                                       down2_tmp_a_whdc, W, H, T, Cc, Cc, stream) != 0 ||
            cosmos3_wan_vae_add_whdc_f32(down2_tmp_a_whdc, mid_attn_whdc,
                                         mid2_whdc, elems, stream) != 0) {
            *error = "Cosmos3 Wan VAE mid2 CUDA primitive execution failed";
            return false;
        }
        mid2_ready = true;
        add_timing(VAE_T_MID2, t_stage);
        return true;
    }

    bool ensure_encoder_head(cudaStream_t stream,
                             std::string * error) {
        if (!available) return true;
        if (head_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!model) {
            *error = "Cosmos3 Wan VAE encoder head has no resident model";
            return false;
        }
        if (!ensure_mid2(stream, error)) return false;

        auto need_bf16 = [&](const char * name) -> const unsigned short * {
            ggml_tensor * t = model->tensor_source(name);
            if (!t || t->type != GGML_TYPE_BF16 || !t->data) {
                *error = std::string("missing BF16 Wan VAE encoder head tensor: ") + name;
                return nullptr;
            }
            return static_cast<const unsigned short *>(t->data);
        };
        const unsigned short * norm = need_bf16("wan22_vae.encoder.head.0.gamma");
        const unsigned short * conv_w = need_bf16("wan22_vae.encoder.head.2.weight");
        const unsigned short * conv_b = need_bf16("wan22_vae.encoder.head.2.bias");
        if (!norm || !conv_w || !conv_b) return false;

        constexpr int W = COSMOS3_WAN_VAE_DOWN2_W;
        constexpr int H = COSMOS3_WAN_VAE_DOWN2_H;
        constexpr int T = COSMOS3_WAN_VAE_DOWN2_T;
        constexpr int Cc = COSMOS3_WAN_VAE_DOWN2_CHANNELS;
        constexpr int OutC = 2 * kMotVisionVaeChannels;
        if (cosmos3_wan_vae_norm_silu_whdc_f32(mid2_whdc, norm,
                                               down2_tmp_b_whdc, W, H, T, Cc, stream) != 0 ||
            cosmos3_wan_vae_causal_conv3d_ks3_whdc_f32(down2_tmp_b_whdc, conv_w, conv_b,
                                                       encoder_head_whdc, W, H, T, Cc, OutC, stream) != 0) {
            *error = "Cosmos3 Wan VAE encoder head CUDA primitive execution failed";
            return false;
        }
        head_ready = true;
        add_timing(VAE_T_HEAD, t_stage);
        return true;
    }

    bool ensure_final_conv1(cudaStream_t stream,
                            std::string * error) {
        if (!available) return true;
        if (final_conv1_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!model) {
            *error = "Cosmos3 Wan VAE final conv1 has no resident model";
            return false;
        }
        if (!ensure_encoder_head(stream, error)) return false;

        auto need_bf16 = [&](const char * name) -> const unsigned short * {
            ggml_tensor * t = model->tensor_source(name);
            if (!t || t->type != GGML_TYPE_BF16 || !t->data) {
                *error = std::string("missing BF16 Wan VAE final conv1 tensor: ") + name;
                return nullptr;
            }
            return static_cast<const unsigned short *>(t->data);
        };
        const unsigned short * conv_w = need_bf16("wan22_vae.conv1.weight");
        const unsigned short * conv_b = need_bf16("wan22_vae.conv1.bias");
        if (!conv_w || !conv_b) return false;

        constexpr int W = COSMOS3_WAN_VAE_DOWN2_W;
        constexpr int H = COSMOS3_WAN_VAE_DOWN2_H;
        constexpr int T = COSMOS3_WAN_VAE_DOWN2_T;
        constexpr int Cc = 2 * kMotVisionVaeChannels;
        if (cosmos3_wan_vae_conv1x1x1_whdc_f32(encoder_head_whdc, conv_w, conv_b,
                                               final_conv1_whdc, W, H, T, Cc, Cc, stream) != 0) {
            *error = "Cosmos3 Wan VAE final conv1 CUDA primitive execution failed";
            return false;
        }
        final_conv1_ready = true;
        add_timing(VAE_T_FINAL_CONV1, t_stage);
        return true;
    }

    bool ensure_clean_vision_condition(cudaStream_t stream,
                                       std::string * error) {
        if (!available) return true;
        if (clean_vision_ready) return true;
        const auto t_stage = std::chrono::steady_clock::now();
        if (!ensure_final_conv1(stream, error)) return false;
        if (cosmos3_wan_vae_pack_clean_vision_condition_f32(final_conv1_whdc,
                                                            weights.scale_mean,
                                                            weights.scale_inv_std,
                                                            clean_vision_condition,
                                                            stream) != 0) {
            *error = "Cosmos3 Wan VAE clean vision condition pack CUDA launch failed";
            return false;
        }
        clean_vision_ready = true;
        add_timing(VAE_T_CLEAN_PACK, t_stage);
        return true;
    }

    const float * clean_vision_condition_ptr(cudaStream_t stream,
                                             std::string * error) {
        if (!available) return nullptr;
        if (!ensure_clean_vision_condition(stream, error)) return nullptr;
        return clean_vision_condition;
    }

    bool append_patch_prefix_tensor(std::vector<WamTensor> * tensors,
                                    int rows,
                                    cudaStream_t stream,
                                    std::string * error) {
        if (!available || !patch_whdc || rows <= 0) return true;
        const int max_rows = COSMOS3_WAN_VAE_PATCH_W *
                             COSMOS3_WAN_VAE_PATCH_H *
                             COSMOS3_WAN_VAE_FRAMES;
        rows = std::max(1, std::min(rows, max_rows));
        if (rows > debug_prefix_rows) {
            if (debug_prefix) cudaFree(debug_prefix);
            debug_prefix = nullptr;
            debug_prefix_rows = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&debug_prefix),
                           static_cast<size_t>(rows) *
                               COSMOS3_WAN_VAE_PATCH_CHANNELS * sizeof(float)) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE debug prefix allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            debug_prefix_rows = rows;
            debug_prefix_cols = COSMOS3_WAN_VAE_PATCH_CHANNELS;
        }
        if (cosmos3_wan_vae_patch_whdc_prefix_f32(patch_whdc, debug_prefix, rows, stream) != 0) {
            *error = "Cosmos3 Wan VAE debug prefix CUDA launch failed";
            return false;
        }
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.patch_input.prefix",
                                         debug_prefix,
                                         rows,
                                         COSMOS3_WAN_VAE_PATCH_CHANNELS,
                                         COSMOS3_WAN_VAE_PATCH_CHANNELS,
                                         error);
    }

    bool append_conv1_prefix_tensor(std::vector<WamTensor> * tensors,
                                    int rows,
                                    int cols,
                                    cudaStream_t stream,
                                    std::string * error) {
        if (!available || !patch_whdc || rows <= 0 || cols <= 0) return true;
        if (!ensure_encoder_conv1(stream, error)) return false;
        const int max_rows = COSMOS3_WAN_VAE_PATCH_W *
                             COSMOS3_WAN_VAE_PATCH_H *
                             COSMOS3_WAN_VAE_FRAMES;
        rows = std::max(1, std::min(rows, max_rows));
        cols = std::max(1, std::min(cols, int(COSMOS3_WAN_VAE_CONV1_CHANNELS)));
        if (rows * cols > debug_prefix_rows * debug_prefix_cols) {
            if (debug_prefix) cudaFree(debug_prefix);
            debug_prefix = nullptr;
            debug_prefix_rows = 0;
            debug_prefix_cols = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&debug_prefix),
                           static_cast<size_t>(rows) * cols * sizeof(float)) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE conv1 debug prefix allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            debug_prefix_rows = rows;
            debug_prefix_cols = cols;
        }
        if (cosmos3_wan_vae_encoder_conv1_whdc_prefix_f32(encoder_conv1_whdc,
                                                          debug_prefix,
                                                          rows,
                                                          cols,
                                                          stream) != 0) {
            *error = "Cosmos3 Wan VAE encoder.conv1 WHDC prefix CUDA launch failed";
            return false;
        }
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.encoder_conv1.prefix",
                                         debug_prefix,
                                         rows,
                                         cols,
                                         cols,
                                         error);
    }

    bool append_down0_shortcut_prefix_tensor(std::vector<WamTensor> * tensors,
                                             int rows,
                                             int cols,
                                             cudaStream_t stream,
                                             std::string * error) {
        if (!available || rows <= 0 || cols <= 0) return true;
        if (!ensure_down0_shortcut(stream, error)) return false;
        const int max_rows = COSMOS3_WAN_VAE_DOWN0_W *
                             COSMOS3_WAN_VAE_DOWN0_H *
                             COSMOS3_WAN_VAE_DOWN0_T;
        rows = std::max(1, std::min(rows, max_rows));
        cols = std::max(1, std::min(cols, int(COSMOS3_WAN_VAE_DOWN0_CHANNELS)));
        if (rows * cols > debug_prefix_rows * debug_prefix_cols) {
            if (debug_prefix) cudaFree(debug_prefix);
            debug_prefix = nullptr;
            debug_prefix_rows = 0;
            debug_prefix_cols = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&debug_prefix),
                           static_cast<size_t>(rows) * cols * sizeof(float)) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE down0 shortcut debug prefix allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            debug_prefix_rows = rows;
            debug_prefix_cols = cols;
        }
        if (cosmos3_wan_vae_down0_whdc_prefix_f32(down0_shortcut_whdc,
                                                  debug_prefix,
                                                  rows,
                                                  cols,
                                                  stream) != 0) {
            *error = "Cosmos3 Wan VAE down0 shortcut prefix CUDA launch failed";
            return false;
        }
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.encoder_down0_shortcut.prefix",
                                         debug_prefix,
                                         rows,
                                         cols,
                                         cols,
                                         error);
    }

    bool append_down0_prefix_tensor(std::vector<WamTensor> * tensors,
                                    int rows,
                                    int cols,
                                    cudaStream_t stream,
                                    std::string * error) {
        if (!available || rows <= 0 || cols <= 0) return true;
        if (!ensure_down0(stream, error)) return false;
        const int max_rows = COSMOS3_WAN_VAE_DOWN0_W *
                             COSMOS3_WAN_VAE_DOWN0_H *
                             COSMOS3_WAN_VAE_DOWN0_T;
        rows = std::max(1, std::min(rows, max_rows));
        cols = std::max(1, std::min(cols, int(COSMOS3_WAN_VAE_DOWN0_CHANNELS)));
        if (rows * cols > debug_prefix_rows * debug_prefix_cols) {
            if (debug_prefix) cudaFree(debug_prefix);
            debug_prefix = nullptr;
            debug_prefix_rows = 0;
            debug_prefix_cols = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&debug_prefix),
                           static_cast<size_t>(rows) * cols * sizeof(float)) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE down0 debug prefix allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            debug_prefix_rows = rows;
            debug_prefix_cols = cols;
        }
        if (cosmos3_wan_vae_down0_whdc_prefix_f32(down0_whdc,
                                                  debug_prefix,
                                                  rows,
                                                  cols,
                                                  stream) != 0) {
            *error = "Cosmos3 Wan VAE down0 prefix CUDA launch failed";
            return false;
        }
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.encoder_down0.prefix",
                                         debug_prefix,
                                         rows,
                                         cols,
                                         cols,
                                         error);
    }

    bool append_down1_prefix_tensor(std::vector<WamTensor> * tensors,
                                    int rows,
                                    int cols,
                                    cudaStream_t stream,
                                    std::string * error) {
        if (!available || rows <= 0 || cols <= 0) return true;
        if (!ensure_down1(stream, error)) return false;
        const int max_rows = COSMOS3_WAN_VAE_DOWN1_W *
                             COSMOS3_WAN_VAE_DOWN1_H *
                             COSMOS3_WAN_VAE_DOWN1_T;
        rows = std::max(1, std::min(rows, max_rows));
        cols = std::max(1, std::min(cols, int(COSMOS3_WAN_VAE_DOWN1_CHANNELS)));
        if (rows * cols > debug_prefix_rows * debug_prefix_cols) {
            if (debug_prefix) cudaFree(debug_prefix);
            debug_prefix = nullptr;
            debug_prefix_rows = 0;
            debug_prefix_cols = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&debug_prefix),
                           static_cast<size_t>(rows) * cols * sizeof(float)) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE down1 debug prefix allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            debug_prefix_rows = rows;
            debug_prefix_cols = cols;
        }
        if (cosmos3_wan_vae_generic_whdc_prefix_f32(down1_whdc,
                                                    debug_prefix,
                                                    COSMOS3_WAN_VAE_DOWN1_W,
                                                    COSMOS3_WAN_VAE_DOWN1_H,
                                                    COSMOS3_WAN_VAE_DOWN1_T,
                                                    COSMOS3_WAN_VAE_DOWN1_CHANNELS,
                                                    rows,
                                                    cols,
                                                    stream) != 0) {
            *error = "Cosmos3 Wan VAE down1 prefix CUDA launch failed";
            return false;
        }
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.encoder_down1.prefix",
                                         debug_prefix,
                                         rows,
                                         cols,
                                         cols,
                                         error);
    }

    bool append_down2_prefix_tensor(std::vector<WamTensor> * tensors,
                                    int rows,
                                    int cols,
                                    cudaStream_t stream,
                                    std::string * error) {
        if (!available || rows <= 0 || cols <= 0) return true;
        if (!ensure_down2(stream, error)) return false;
        const int max_rows = COSMOS3_WAN_VAE_DOWN2_W *
                             COSMOS3_WAN_VAE_DOWN2_H *
                             COSMOS3_WAN_VAE_DOWN2_T;
        rows = std::max(1, std::min(rows, max_rows));
        cols = std::max(1, std::min(cols, int(COSMOS3_WAN_VAE_DOWN2_CHANNELS)));
        if (rows * cols > debug_prefix_rows * debug_prefix_cols) {
            if (debug_prefix) cudaFree(debug_prefix);
            debug_prefix = nullptr;
            debug_prefix_rows = 0;
            debug_prefix_cols = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&debug_prefix),
                           static_cast<size_t>(rows) * cols * sizeof(float)) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE down2 debug prefix allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            debug_prefix_rows = rows;
            debug_prefix_cols = cols;
        }
        if (cosmos3_wan_vae_generic_whdc_prefix_f32(down2_whdc,
                                                    debug_prefix,
                                                    COSMOS3_WAN_VAE_DOWN2_W,
                                                    COSMOS3_WAN_VAE_DOWN2_H,
                                                    COSMOS3_WAN_VAE_DOWN2_T,
                                                    COSMOS3_WAN_VAE_DOWN2_CHANNELS,
                                                    rows,
                                                    cols,
                                                    stream) != 0) {
            *error = "Cosmos3 Wan VAE down2 prefix CUDA launch failed";
            return false;
        }
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.encoder_down2.prefix",
                                         debug_prefix,
                                         rows,
                                         cols,
                                         cols,
                                         error);
    }

    bool append_down3_prefix_tensor(std::vector<WamTensor> * tensors,
                                    int rows,
                                    int cols,
                                    cudaStream_t stream,
                                    std::string * error) {
        if (!available || rows <= 0 || cols <= 0) return true;
        if (!ensure_down3(stream, error)) return false;
        const int max_rows = COSMOS3_WAN_VAE_DOWN2_W *
                             COSMOS3_WAN_VAE_DOWN2_H *
                             COSMOS3_WAN_VAE_DOWN2_T;
        rows = std::max(1, std::min(rows, max_rows));
        cols = std::max(1, std::min(cols, int(COSMOS3_WAN_VAE_DOWN2_CHANNELS)));
        if (rows * cols > debug_prefix_rows * debug_prefix_cols) {
            if (debug_prefix) cudaFree(debug_prefix);
            debug_prefix = nullptr;
            debug_prefix_rows = 0;
            debug_prefix_cols = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&debug_prefix),
                           static_cast<size_t>(rows) * cols * sizeof(float)) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE down3 debug prefix allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            debug_prefix_rows = rows;
            debug_prefix_cols = cols;
        }
        if (cosmos3_wan_vae_generic_whdc_prefix_f32(down3_whdc,
                                                    debug_prefix,
                                                    COSMOS3_WAN_VAE_DOWN2_W,
                                                    COSMOS3_WAN_VAE_DOWN2_H,
                                                    COSMOS3_WAN_VAE_DOWN2_T,
                                                    COSMOS3_WAN_VAE_DOWN2_CHANNELS,
                                                    rows,
                                                    cols,
                                                    stream) != 0) {
            *error = "Cosmos3 Wan VAE down3 prefix CUDA launch failed";
            return false;
        }
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.encoder_down3.prefix",
                                         debug_prefix,
                                         rows,
                                         cols,
                                         cols,
                                         error);
    }

    bool append_mid0_prefix_tensor(std::vector<WamTensor> * tensors,
                                   int rows,
                                   int cols,
                                   cudaStream_t stream,
                                   std::string * error) {
        if (!available || rows <= 0 || cols <= 0) return true;
        if (!ensure_mid0(stream, error)) return false;
        const int max_rows = COSMOS3_WAN_VAE_DOWN2_W *
                             COSMOS3_WAN_VAE_DOWN2_H *
                             COSMOS3_WAN_VAE_DOWN2_T;
        rows = std::max(1, std::min(rows, max_rows));
        cols = std::max(1, std::min(cols, int(COSMOS3_WAN_VAE_DOWN2_CHANNELS)));
        if (rows * cols > debug_prefix_rows * debug_prefix_cols) {
            if (debug_prefix) cudaFree(debug_prefix);
            debug_prefix = nullptr;
            debug_prefix_rows = 0;
            debug_prefix_cols = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&debug_prefix),
                           static_cast<size_t>(rows) * cols * sizeof(float)) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE mid0 debug prefix allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            debug_prefix_rows = rows;
            debug_prefix_cols = cols;
        }
        if (cosmos3_wan_vae_generic_whdc_prefix_f32(mid0_whdc,
                                                    debug_prefix,
                                                    COSMOS3_WAN_VAE_DOWN2_W,
                                                    COSMOS3_WAN_VAE_DOWN2_H,
                                                    COSMOS3_WAN_VAE_DOWN2_T,
                                                    COSMOS3_WAN_VAE_DOWN2_CHANNELS,
                                                    rows,
                                                    cols,
                                                    stream) != 0) {
            *error = "Cosmos3 Wan VAE mid0 prefix CUDA launch failed";
            return false;
        }
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.encoder_mid0.prefix",
                                         debug_prefix,
                                         rows,
                                         cols,
                                         cols,
                                         error);
    }

    bool append_mid_attn_prefix_tensor(std::vector<WamTensor> * tensors,
                                       int rows,
                                       int cols,
                                       cudaStream_t stream,
                                       std::string * error) {
        if (!available || rows <= 0 || cols <= 0) return true;
        if (!ensure_mid_attn(stream, error)) return false;
        const int max_rows = COSMOS3_WAN_VAE_DOWN2_W *
                             COSMOS3_WAN_VAE_DOWN2_H *
                             COSMOS3_WAN_VAE_DOWN2_T;
        rows = std::max(1, std::min(rows, max_rows));
        cols = std::max(1, std::min(cols, int(COSMOS3_WAN_VAE_DOWN2_CHANNELS)));
        if (rows * cols > debug_prefix_rows * debug_prefix_cols) {
            if (debug_prefix) cudaFree(debug_prefix);
            debug_prefix = nullptr;
            debug_prefix_rows = 0;
            debug_prefix_cols = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&debug_prefix),
                           static_cast<size_t>(rows) * cols * sizeof(float)) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE middle attention debug prefix allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            debug_prefix_rows = rows;
            debug_prefix_cols = cols;
        }
        if (cosmos3_wan_vae_generic_whdc_prefix_f32(mid_attn_whdc,
                                                    debug_prefix,
                                                    COSMOS3_WAN_VAE_DOWN2_W,
                                                    COSMOS3_WAN_VAE_DOWN2_H,
                                                    COSMOS3_WAN_VAE_DOWN2_T,
                                                    COSMOS3_WAN_VAE_DOWN2_CHANNELS,
                                                    rows,
                                                    cols,
                                                    stream) != 0) {
            *error = "Cosmos3 Wan VAE middle attention prefix CUDA launch failed";
            return false;
        }
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.encoder_mid_attn.prefix",
                                         debug_prefix,
                                         rows,
                                         cols,
                                         cols,
                                         error);
    }

    bool append_mid2_prefix_tensor(std::vector<WamTensor> * tensors,
                                   int rows,
                                   int cols,
                                   cudaStream_t stream,
                                   std::string * error) {
        if (!available || rows <= 0 || cols <= 0) return true;
        if (!ensure_mid2(stream, error)) return false;
        const int max_rows = COSMOS3_WAN_VAE_DOWN2_W *
                             COSMOS3_WAN_VAE_DOWN2_H *
                             COSMOS3_WAN_VAE_DOWN2_T;
        rows = std::max(1, std::min(rows, max_rows));
        cols = std::max(1, std::min(cols, int(COSMOS3_WAN_VAE_DOWN2_CHANNELS)));
        if (rows * cols > debug_prefix_rows * debug_prefix_cols) {
            if (debug_prefix) cudaFree(debug_prefix);
            debug_prefix = nullptr;
            debug_prefix_rows = 0;
            debug_prefix_cols = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&debug_prefix),
                           static_cast<size_t>(rows) * cols * sizeof(float)) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE mid2 debug prefix allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            debug_prefix_rows = rows;
            debug_prefix_cols = cols;
        }
        if (cosmos3_wan_vae_generic_whdc_prefix_f32(mid2_whdc,
                                                    debug_prefix,
                                                    COSMOS3_WAN_VAE_DOWN2_W,
                                                    COSMOS3_WAN_VAE_DOWN2_H,
                                                    COSMOS3_WAN_VAE_DOWN2_T,
                                                    COSMOS3_WAN_VAE_DOWN2_CHANNELS,
                                                    rows,
                                                    cols,
                                                    stream) != 0) {
            *error = "Cosmos3 Wan VAE mid2 prefix CUDA launch failed";
            return false;
        }
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.encoder_mid2.prefix",
                                         debug_prefix,
                                         rows,
                                         cols,
                                         cols,
                                         error);
    }

    bool append_head_prefix_tensor(std::vector<WamTensor> * tensors,
                                   int rows,
                                   int cols,
                                   cudaStream_t stream,
                                   std::string * error) {
        if (!available || rows <= 0 || cols <= 0) return true;
        if (!ensure_encoder_head(stream, error)) return false;
        const int max_rows = COSMOS3_WAN_VAE_DOWN2_W *
                             COSMOS3_WAN_VAE_DOWN2_H *
                             COSMOS3_WAN_VAE_DOWN2_T;
        rows = std::max(1, std::min(rows, max_rows));
        cols = std::max(1, std::min(cols, 2 * kMotVisionVaeChannels));
        if (rows * cols > debug_prefix_rows * debug_prefix_cols) {
            if (debug_prefix) cudaFree(debug_prefix);
            debug_prefix = nullptr;
            debug_prefix_rows = 0;
            debug_prefix_cols = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&debug_prefix),
                           static_cast<size_t>(rows) * cols * sizeof(float)) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE encoder head debug prefix allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            debug_prefix_rows = rows;
            debug_prefix_cols = cols;
        }
        if (cosmos3_wan_vae_generic_whdc_prefix_f32(encoder_head_whdc,
                                                    debug_prefix,
                                                    COSMOS3_WAN_VAE_DOWN2_W,
                                                    COSMOS3_WAN_VAE_DOWN2_H,
                                                    COSMOS3_WAN_VAE_DOWN2_T,
                                                    2 * kMotVisionVaeChannels,
                                                    rows,
                                                    cols,
                                                    stream) != 0) {
            *error = "Cosmos3 Wan VAE encoder head prefix CUDA launch failed";
            return false;
        }
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.encoder_head.prefix",
                                         debug_prefix,
                                         rows,
                                         cols,
                                         cols,
                                         error);
    }

    bool append_final_conv1_prefix_tensor(std::vector<WamTensor> * tensors,
                                          int rows,
                                          int cols,
                                          cudaStream_t stream,
                                          std::string * error) {
        if (!available || rows <= 0 || cols <= 0) return true;
        if (!ensure_final_conv1(stream, error)) return false;
        const int max_rows = COSMOS3_WAN_VAE_DOWN2_W *
                             COSMOS3_WAN_VAE_DOWN2_H *
                             COSMOS3_WAN_VAE_DOWN2_T;
        rows = std::max(1, std::min(rows, max_rows));
        cols = std::max(1, std::min(cols, 2 * kMotVisionVaeChannels));
        if (rows * cols > debug_prefix_rows * debug_prefix_cols) {
            if (debug_prefix) cudaFree(debug_prefix);
            debug_prefix = nullptr;
            debug_prefix_rows = 0;
            debug_prefix_cols = 0;
            if (cudaMalloc(reinterpret_cast<void **>(&debug_prefix),
                           static_cast<size_t>(rows) * cols * sizeof(float)) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 Wan VAE final conv1 debug prefix allocation failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            debug_prefix_rows = rows;
            debug_prefix_cols = cols;
        }
        if (cosmos3_wan_vae_generic_whdc_prefix_f32(final_conv1_whdc,
                                                    debug_prefix,
                                                    COSMOS3_WAN_VAE_DOWN2_W,
                                                    COSMOS3_WAN_VAE_DOWN2_H,
                                                    COSMOS3_WAN_VAE_DOWN2_T,
                                                    2 * kMotVisionVaeChannels,
                                                    rows,
                                                    cols,
                                                    stream) != 0) {
            *error = "Cosmos3 Wan VAE final conv1 prefix CUDA launch failed";
            return false;
        }
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.final_conv1.prefix",
                                         debug_prefix,
                                         rows,
                                         cols,
                                         cols,
                                         error);
    }

    bool append_clean_vision_condition_prefix_tensor(std::vector<WamTensor> * tensors,
                                                     int rows,
                                                     int cols,
                                                     cudaStream_t stream,
                                                     std::string * error) {
        if (!available || rows <= 0 || cols <= 0) return true;
        if (!ensure_clean_vision_condition(stream, error)) return false;
        rows = std::max(1, std::min(rows, kMotVisionConditionTokens));
        cols = std::max(1, std::min(cols, kMotVisionPatchDim));
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.wan_vae.clean_vision_condition.prefix",
                                         clean_vision_condition,
                                         rows,
                                         cols,
                                         kMotVisionPatchDim,
                                         error);
    }
};

bool bind_action_bridge_cuda_weights(const GgufGpuResidentModel & weights,
                                     ActionBridgeWeightsGpu * bridge,
                                     std::string * error) {
    auto need = [&](const char * name) -> ggml_tensor * {
        return need_gpu_bf16_source(weights, name, error);
    };
    ggml_tensor * a2l_fc = need("net.action2llm.fc.weight");
    ggml_tensor * a2l_bias = need("net.action2llm.bias.weight");
    ggml_tensor * l2a_fc = need("net.llm2action.fc.weight");
    ggml_tensor * l2a_bias = need("net.llm2action.bias.weight");
    ggml_tensor * modality = need("net.action_modality_embed");
    if (!a2l_fc || !a2l_bias || !l2a_fc || !l2a_bias || !modality) return false;
    if (a2l_fc->ne[0] != kActionMaxDim * kLanguageHidden || a2l_fc->ne[1] != kActionDomains ||
        a2l_bias->ne[0] != kLanguageHidden || a2l_bias->ne[1] != kActionDomains ||
        l2a_fc->ne[0] != kLanguageHidden * kActionMaxDim || l2a_fc->ne[1] != kActionDomains ||
        l2a_bias->ne[0] != kActionMaxDim || l2a_bias->ne[1] != kActionDomains ||
        modality->ne[0] != kLanguageHidden) {
        *error = "Cosmos3 action bridge tensor shape mismatch";
        return false;
    }
    bridge->action2llm_fc = static_cast<const unsigned short *>(a2l_fc->data);
    bridge->action2llm_bias = static_cast<const unsigned short *>(a2l_bias->data);
    bridge->llm2action_fc = static_cast<const unsigned short *>(l2a_fc->data);
    bridge->llm2action_bias = static_cast<const unsigned short *>(l2a_bias->data);
    bridge->action_modality_embed = static_cast<const unsigned short *>(modality->data);
    return true;
}

bool bind_mot_action_cuda_weights(const GgufGpuResidentModel & weights,
                                  MotActionWeightsGpu * mot,
                                  std::string * error) {
    if (!bind_action_bridge_cuda_weights(weights, &mot->bridge, error)) return false;
    auto need = [&](const char * name) -> ggml_tensor * {
        return need_gpu_bf16_source(weights, name, error);
    };
    ggml_tensor * mlp0_w = need("net.time_embedder.mlp.0.weight");
    ggml_tensor * mlp0_b = need("net.time_embedder.mlp.0.bias");
    ggml_tensor * mlp2_w = need("net.time_embedder.mlp.2.weight");
    ggml_tensor * mlp2_b = need("net.time_embedder.mlp.2.bias");
    ggml_tensor * vae2llm_w = need("net.vae2llm.weight");
    ggml_tensor * vae2llm_b = need("net.vae2llm.bias");
    ggml_tensor * llm2vae_w = need("net.llm2vae.weight");
    ggml_tensor * llm2vae_b = need("net.llm2vae.bias");
    if (!mlp0_w || !mlp0_b || !mlp2_w || !mlp2_b ||
        !vae2llm_w || !vae2llm_b || !llm2vae_w || !llm2vae_b) return false;
    if (vae2llm_w->ne[0] != kMotVisionPatchDim || vae2llm_w->ne[1] != kLanguageHidden ||
        vae2llm_b->ne[0] != kLanguageHidden ||
        llm2vae_w->ne[0] != kLanguageHidden || llm2vae_w->ne[1] != kMotVisionPatchDim ||
        llm2vae_b->ne[0] != kMotVisionPatchDim ||
        mlp0_w->ne[0] != kTimeFreqDim || mlp0_w->ne[1] != kLanguageHidden ||
        mlp0_b->ne[0] != kLanguageHidden ||
        mlp2_w->ne[0] != kLanguageHidden || mlp2_w->ne[1] != kLanguageHidden ||
        mlp2_b->ne[0] != kLanguageHidden) {
        *error = "Cosmos3 MoT action/vision bridge tensor shape mismatch";
        return false;
    }
    mot->vae2llm_weight = static_cast<const unsigned short *>(vae2llm_w->data);
    mot->vae2llm_bias = static_cast<const unsigned short *>(vae2llm_b->data);
    mot->llm2vae_weight = static_cast<const unsigned short *>(llm2vae_w->data);
    mot->llm2vae_bias = static_cast<const unsigned short *>(llm2vae_b->data);
    mot->time_mlp0_weight = static_cast<const unsigned short *>(mlp0_w->data);
    mot->time_mlp0_bias = static_cast<const unsigned short *>(mlp0_b->data);
    mot->time_mlp2_weight = static_cast<const unsigned short *>(mlp2_w->data);
    mot->time_mlp2_bias = static_cast<const unsigned short *>(mlp2_b->data);
    return true;
}

bool bind_wan22_vae_encoder_cuda_weights(const GgufGpuResidentModel & weights,
                                         WanVaeEncoderWeightsGpu * vae,
                                         std::string * error) {
    auto need_bf16 = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = weights.tensor_source(name);
        if (!t) {
            *error = "missing Wan2.2 VAE tensor: " + name;
            return nullptr;
        }
        if (t->type != GGML_TYPE_BF16 || !t->data) {
            *error = "Wan2.2 VAE tensor is not GPU-resident BF16: " + name;
            return nullptr;
        }
        return t;
    };

    const bool has_any_vae_source =
        weights.tensor_source("wan22_vae.conv1.weight") != nullptr ||
        weights.tensor_source("wan22_vae.conv1.bias") != nullptr ||
        weights.tensor_source("wan22_vae.vae.scale.mean") != nullptr ||
        weights.tensor_source("wan22_vae.vae.scale.inv_std") != nullptr;
    if (!has_any_vae_source) {
        vae->available = false;
        vae->encoder_tensor_count = 0;
        vae->scale_mean = nullptr;
        vae->scale_inv_std = nullptr;
        vae->encoder_conv1_weight = nullptr;
        vae->encoder_conv1_bias = nullptr;
        return true;
    }

    ggml_tensor * scale_mean = need_bf16("wan22_vae.vae.scale.mean");
    ggml_tensor * scale_inv_std = need_bf16("wan22_vae.vae.scale.inv_std");
    ggml_tensor * conv1_w = need_bf16("wan22_vae.conv1.weight");
    ggml_tensor * conv1_b = need_bf16("wan22_vae.conv1.bias");
    ggml_tensor * enc_conv1_w = need_bf16("wan22_vae.encoder.conv1.weight");
    ggml_tensor * enc_conv1_b = need_bf16("wan22_vae.encoder.conv1.bias");
    if (!scale_mean || !scale_inv_std || !conv1_w || !conv1_b ||
        !enc_conv1_w || !enc_conv1_b) return false;
    auto shape_eq = [&](const std::string & name, std::initializer_list<int64_t> expected) -> bool {
        const std::vector<int64_t> * shape = weights.names.shape_for_source(name);
        if (!shape) {
            *error = "missing original shape metadata for Wan2.2 VAE tensor: " + name;
            return false;
        }
        if (shape->size() != expected.size()) {
            *error = "Wan2.2 VAE tensor rank mismatch: " + name;
            return false;
        }
        size_t i = 0;
        for (const int64_t dim : expected) {
            if ((*shape)[i++] != dim) {
                *error = "Wan2.2 VAE tensor shape mismatch: " + name;
                return false;
            }
        }
        return true;
    };
    if (scale_mean->ne[0] != kMotVisionVaeChannels || scale_inv_std->ne[0] != kMotVisionVaeChannels ||
        conv1_b->ne[0] != 2 * kMotVisionVaeChannels ||
        !shape_eq("wan22_vae.conv1.weight",
                  {2 * kMotVisionVaeChannels, 2 * kMotVisionVaeChannels, 1, 1, 1}) ||
        !shape_eq("wan22_vae.conv1.bias", {2 * kMotVisionVaeChannels}) ||
        !shape_eq("wan22_vae.vae.scale.mean", {kMotVisionVaeChannels}) ||
        !shape_eq("wan22_vae.vae.scale.inv_std", {kMotVisionVaeChannels})) {
        if (error->empty()) {
            *error = "Wan2.2 VAE encoder scale/conv1 shape mismatch";
        }
        return false;
    }
    if (!shape_eq("wan22_vae.encoder.conv1.weight", {160, 12, 3, 3, 3}) ||
        !shape_eq("wan22_vae.encoder.conv1.bias", {160}) ||
        !shape_eq("wan22_vae.encoder.head.2.weight", {2 * kMotVisionVaeChannels, 640, 3, 3, 3}) ||
        !shape_eq("wan22_vae.encoder.head.2.bias", {2 * kMotVisionVaeChannels})) {
        if (error->empty()) {
            *error = "Wan2.2 VAE encoder boundary tensor shape mismatch";
        }
        return false;
    }

    int encoder_tensor_count = 0;
    for (const auto & kv : weights.by_source_name) {
        if (kv.first.rfind("wan22_vae.encoder.", 0) == 0) {
            ++encoder_tensor_count;
        }
    }
    if (encoder_tensor_count == 0) {
        *error = "Wan2.2 VAE encoder tensors were requested but none were found";
        return false;
    }

    vae->available = true;
    vae->encoder_tensor_count = encoder_tensor_count;
    vae->scale_mean = static_cast<const unsigned short *>(scale_mean->data);
    vae->scale_inv_std = static_cast<const unsigned short *>(scale_inv_std->data);
    vae->encoder_conv1_weight = static_cast<const unsigned short *>(enc_conv1_w->data);
    vae->encoder_conv1_bias = static_cast<const unsigned short *>(enc_conv1_b->data);
    return true;
}

bool cosmos3_wan_vae_reference_readiness_check(const GgufGpuResidentModel & weights,
                                               std::string * error) {
    struct Need {
        std::string name;
        int64_t numel;
    };
    std::vector<Need> needs = {
        {"wan22_vae.encoder.conv1.weight", 160ll * 12 * 3 * 3 * 3},
        {"wan22_vae.encoder.conv1.bias", 160},
        {"wan22_vae.encoder.head.0.gamma", 640},
        {"wan22_vae.encoder.head.2.weight", 96ll * 640 * 3 * 3 * 3},
        {"wan22_vae.encoder.head.2.bias", 96},
        {"wan22_vae.conv1.weight", 96ll * 96},
        {"wan22_vae.conv1.bias", 96},
        {"wan22_vae.vae.scale.mean", 48},
        {"wan22_vae.vae.scale.inv_std", 48},
    };

    auto add_residual = [&](const std::string & prefix, int in_c, int out_c, bool shortcut) {
        needs.push_back({prefix + ".residual.0.gamma", in_c});
        needs.push_back({prefix + ".residual.2.weight",
                         static_cast<int64_t>(out_c) * in_c * 3 * 3 * 3});
        needs.push_back({prefix + ".residual.2.bias", out_c});
        needs.push_back({prefix + ".residual.3.gamma", out_c});
        needs.push_back({prefix + ".residual.6.weight",
                         static_cast<int64_t>(out_c) * out_c * 3 * 3 * 3});
        needs.push_back({prefix + ".residual.6.bias", out_c});
        if (shortcut) {
            needs.push_back({prefix + ".shortcut.weight",
                             static_cast<int64_t>(out_c) * in_c});
            needs.push_back({prefix + ".shortcut.bias", out_c});
        }
    };

    add_residual("wan22_vae.encoder.downsamples.0.downsamples.0", 160, 160, false);
    add_residual("wan22_vae.encoder.downsamples.0.downsamples.1", 160, 160, false);
    needs.push_back({"wan22_vae.encoder.downsamples.0.downsamples.2.resample.1.weight", 160ll * 160 * 3 * 3});
    needs.push_back({"wan22_vae.encoder.downsamples.0.downsamples.2.resample.1.bias", 160});

    add_residual("wan22_vae.encoder.downsamples.1.downsamples.0", 160, 320, true);
    add_residual("wan22_vae.encoder.downsamples.1.downsamples.1", 320, 320, false);
    needs.push_back({"wan22_vae.encoder.downsamples.1.downsamples.2.resample.1.weight", 320ll * 320 * 3 * 3});
    needs.push_back({"wan22_vae.encoder.downsamples.1.downsamples.2.resample.1.bias", 320});
    needs.push_back({"wan22_vae.encoder.downsamples.1.downsamples.2.time_conv.weight", 320ll * 320 * 3});
    needs.push_back({"wan22_vae.encoder.downsamples.1.downsamples.2.time_conv.bias", 320});

    add_residual("wan22_vae.encoder.downsamples.2.downsamples.0", 320, 640, true);
    add_residual("wan22_vae.encoder.downsamples.2.downsamples.1", 640, 640, false);
    needs.push_back({"wan22_vae.encoder.downsamples.2.downsamples.2.resample.1.weight", 640ll * 640 * 3 * 3});
    needs.push_back({"wan22_vae.encoder.downsamples.2.downsamples.2.resample.1.bias", 640});
    needs.push_back({"wan22_vae.encoder.downsamples.2.downsamples.2.time_conv.weight", 640ll * 640 * 3});
    needs.push_back({"wan22_vae.encoder.downsamples.2.downsamples.2.time_conv.bias", 640});

    add_residual("wan22_vae.encoder.downsamples.3.downsamples.0", 640, 640, false);
    add_residual("wan22_vae.encoder.downsamples.3.downsamples.1", 640, 640, false);
    add_residual("wan22_vae.encoder.middle.0", 640, 640, false);
    add_residual("wan22_vae.encoder.middle.2", 640, 640, false);
    needs.push_back({"wan22_vae.encoder.middle.1.norm.gamma", 640});
    needs.push_back({"wan22_vae.encoder.middle.1.to_qkv.weight", 1920ll * 640});
    needs.push_back({"wan22_vae.encoder.middle.1.to_qkv.bias", 1920});
    needs.push_back({"wan22_vae.encoder.middle.1.proj.weight", 640ll * 640});
    needs.push_back({"wan22_vae.encoder.middle.1.proj.bias", 640});

    std::vector<float> scratch;
    for (const Need & need : needs) {
        if (!read_source_tensor_to_f32(weights, need.name, &scratch, need.numel, error)) {
            if (error && error->empty()) *error = std::string("Wan VAE readiness failed at ") + need.name;
            return false;
        }
    }
    return true;
}

bool bind_w8_linear_gpu(const GgufGpuResidentModel & weights,
                        const std::string & module,
                        int expected_in,
                        int expected_out,
                        W8LinearGpu * out,
                        std::string * error) {
    const PackedW8Index::Record * rec = weights.w8.find(module);
    if (!rec) {
        *error = "missing Cosmos3 W8 module in GGUF metadata: " + module;
        return false;
    }
    ggml_tensor * q = nullptr;
    ggml_tensor * s = nullptr;
    const auto q_it = weights.by_gguf_name.find(rec->qweight_gguf);
    const auto s_it = weights.by_gguf_name.find(rec->scales_gguf);
    if (q_it != weights.by_gguf_name.end()) q = q_it->second;
    if (s_it != weights.by_gguf_name.end()) s = s_it->second;
    if (!q || !s || !q->data || !s->data) {
        *error = "missing GPU-resident W8 tensors for module: " + module;
        return false;
    }
    if (q->type != GGML_TYPE_I32 || s->type != GGML_TYPE_BF16) {
        *error = "unexpected W8 tensor type for module: " + module;
        return false;
    }
    if (rec->size_k != expected_in || rec->size_n != expected_out) {
        *error = "unexpected W8 logical shape for " + module +
                 ": got [" + std::to_string(static_cast<long long>(rec->size_k)) +
                 "x" + std::to_string(static_cast<long long>(rec->size_n)) +
                 "], expected [" + std::to_string(expected_in) +
                 "x" + std::to_string(expected_out) + "]";
        return false;
    }
    out->qweight = static_cast<const int *>(q->data);
    out->scales = static_cast<const unsigned short *>(s->data);
    out->in_features = expected_in;
    out->out_features = expected_out;
    out->qweight_rows = static_cast<int>(q->ne[0]);
    out->qweight_cols = static_cast<int>(q->ne[1]);
    out->scale_rows = static_cast<int>(s->ne[0]);
    out->scale_cols = s->ne[1] > 0 ? static_cast<int>(s->ne[1]) : 1;
    return true;
}

bool bind_language_cuda_weights(const GgufGpuResidentModel & weights,
                                LanguageWeightsGpu * lang,
                                std::string * error) {
    auto need_bf16 = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = weights.tensor_source(name);
        if (!t) {
            *error = "missing language BF16 tensor: " + name;
            return nullptr;
        }
        if (t->type != GGML_TYPE_BF16 || !t->data) {
            *error = "language tensor is not GPU-resident BF16: " + name;
            return nullptr;
        }
        return t;
    };
    ggml_tensor * embed = need_bf16("net.language_model.model.embed_tokens.weight");
    ggml_tensor * final_norm = need_bf16("net.language_model.model.norm.weight");
    if (!embed || !final_norm) return false;
    if (embed->ne[0] != kLanguageHidden || embed->ne[1] <= kLanguageVideoPadToken ||
        final_norm->ne[0] != kLanguageHidden) {
        *error = "Qwen3 language embed/final_norm shape mismatch";
        return false;
    }
    lang->embed_tokens = static_cast<const unsigned short *>(embed->data);
    lang->vocab = static_cast<int>(embed->ne[1]);
    lang->final_norm = static_cast<const unsigned short *>(final_norm->data);

    for (int layer = 0; layer < kLanguageLayers; ++layer) {
        LanguageLayerGpu & l = lang->layers[static_cast<size_t>(layer)];
        ggml_tensor * in_norm = need_bf16(language_layer_name(layer, "input_layernorm.weight"));
        ggml_tensor * post_norm = need_bf16(language_layer_name(layer, "post_attention_layernorm.weight"));
        ggml_tensor * q_norm = need_bf16(language_layer_name(layer, "self_attn.q_norm.weight"));
        ggml_tensor * k_norm = need_bf16(language_layer_name(layer, "self_attn.k_norm.weight"));
        if (!in_norm || !post_norm || !q_norm || !k_norm) return false;
        if (in_norm->ne[0] != kLanguageHidden || post_norm->ne[0] != kLanguageHidden ||
            q_norm->ne[0] != kLanguageHeadDim || k_norm->ne[0] != kLanguageHeadDim) {
            *error = "Qwen3 language norm shape mismatch at layer " + std::to_string(layer);
            return false;
        }
        l.input_norm = static_cast<const unsigned short *>(in_norm->data);
        l.post_norm = static_cast<const unsigned short *>(post_norm->data);
        l.q_norm = static_cast<const unsigned short *>(q_norm->data);
        l.k_norm = static_cast<const unsigned short *>(k_norm->data);
        if (!bind_w8_linear_gpu(weights, language_layer_name(layer, "self_attn.q_proj"),
                                kLanguageHidden, kLanguageHidden, &l.q_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "self_attn.k_proj"),
                                kLanguageHidden, kLanguageKv, &l.k_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "self_attn.v_proj"),
                                kLanguageHidden, kLanguageKv, &l.v_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "self_attn.o_proj"),
                                kLanguageHidden, kLanguageHidden, &l.o_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "mlp.gate_proj"),
                                kLanguageHidden, kLanguageIntermediate, &l.gate_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "mlp.up_proj"),
                                kLanguageHidden, kLanguageIntermediate, &l.up_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "mlp.down_proj"),
                                kLanguageIntermediate, kLanguageHidden, &l.down_proj, error)) {
            return false;
        }
    }
    return true;
}

bool bind_mot_gen_cuda_weights(const GgufGpuResidentModel & weights,
                               MotGenWeightsGpu * mot,
                               std::string * error) {
    auto need_bf16 = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = weights.tensor_source(name);
        if (!t) {
            *error = "missing MoT BF16 tensor: " + name;
            return nullptr;
        }
        if (t->type != GGML_TYPE_BF16 || !t->data) {
            *error = "MoT tensor is not GPU-resident BF16: " + name;
            return nullptr;
        }
        return t;
    };
    ggml_tensor * final_norm = need_bf16("net.language_model.model.norm_moe_gen.weight");
    if (!final_norm || final_norm->ne[0] != kLanguageHidden) {
        *error = "MoT generation final norm shape mismatch";
        return false;
    }
    mot->final_norm = static_cast<const unsigned short *>(final_norm->data);
    for (int layer = 0; layer < kLanguageLayers; ++layer) {
        MotGenLayerGpu & l = mot->layers[static_cast<size_t>(layer)];
        ggml_tensor * und_in_norm = need_bf16(language_layer_name(layer, "input_layernorm.weight"));
        ggml_tensor * und_post_norm = need_bf16(language_layer_name(layer, "post_attention_layernorm.weight"));
        ggml_tensor * und_q_norm = need_bf16(language_layer_name(layer, "self_attn.q_norm.weight"));
        ggml_tensor * und_k_norm = need_bf16(language_layer_name(layer, "self_attn.k_norm.weight"));
        ggml_tensor * gen_in_norm = need_bf16(language_layer_name(layer, "input_layernorm_moe_gen.weight"));
        ggml_tensor * gen_post_norm = need_bf16(language_layer_name(layer, "post_attention_layernorm_moe_gen.weight"));
        ggml_tensor * gen_q_norm = need_bf16(language_layer_name(layer, "self_attn.q_norm_moe_gen.weight"));
        ggml_tensor * gen_k_norm = need_bf16(language_layer_name(layer, "self_attn.k_norm_moe_gen.weight"));
        if (!und_in_norm || !und_post_norm || !und_q_norm || !und_k_norm ||
            !gen_in_norm || !gen_post_norm || !gen_q_norm || !gen_k_norm) return false;
        if (und_in_norm->ne[0] != kLanguageHidden ||
            und_post_norm->ne[0] != kLanguageHidden ||
            und_q_norm->ne[0] != kLanguageHeadDim ||
            und_k_norm->ne[0] != kLanguageHeadDim ||
            gen_in_norm->ne[0] != kLanguageHidden ||
            gen_post_norm->ne[0] != kLanguageHidden ||
            gen_q_norm->ne[0] != kLanguageHeadDim ||
            gen_k_norm->ne[0] != kLanguageHeadDim) {
            *error = "MoT generation norm shape mismatch at layer " + std::to_string(layer);
            return false;
        }
        l.und_input_norm = static_cast<const unsigned short *>(und_in_norm->data);
        l.und_post_norm = static_cast<const unsigned short *>(und_post_norm->data);
        l.und_q_norm = static_cast<const unsigned short *>(und_q_norm->data);
        l.und_k_norm = static_cast<const unsigned short *>(und_k_norm->data);
        l.gen_input_norm = static_cast<const unsigned short *>(gen_in_norm->data);
        l.gen_post_norm = static_cast<const unsigned short *>(gen_post_norm->data);
        l.gen_q_norm = static_cast<const unsigned short *>(gen_q_norm->data);
        l.gen_k_norm = static_cast<const unsigned short *>(gen_k_norm->data);
        if (!bind_w8_linear_gpu(weights, language_layer_name(layer, "self_attn.q_proj"),
                                kLanguageHidden, kLanguageHidden, &l.und_q_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "self_attn.k_proj"),
                                kLanguageHidden, kLanguageKv, &l.und_k_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "self_attn.v_proj"),
                                kLanguageHidden, kLanguageKv, &l.und_v_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "self_attn.o_proj"),
                                kLanguageHidden, kLanguageHidden, &l.und_o_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "mlp.gate_proj"),
                                kLanguageHidden, kLanguageIntermediate, &l.und_gate_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "mlp.up_proj"),
                                kLanguageHidden, kLanguageIntermediate, &l.und_up_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "mlp.down_proj"),
                                kLanguageIntermediate, kLanguageHidden, &l.und_down_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "self_attn.q_proj_moe_gen"),
                                kLanguageHidden, kLanguageHidden, &l.gen_q_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "self_attn.k_proj_moe_gen"),
                                kLanguageHidden, kLanguageKv, &l.gen_k_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "self_attn.v_proj_moe_gen"),
                                kLanguageHidden, kLanguageKv, &l.gen_v_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "self_attn.o_proj_moe_gen"),
                                kLanguageHidden, kLanguageHidden, &l.gen_o_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "mlp_moe_gen.gate_proj"),
                                kLanguageHidden, kLanguageIntermediate, &l.gen_gate_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "mlp_moe_gen.up_proj"),
                                kLanguageHidden, kLanguageIntermediate, &l.gen_up_proj, error) ||
            !bind_w8_linear_gpu(weights, language_layer_name(layer, "mlp_moe_gen.down_proj"),
                                kLanguageIntermediate, kLanguageHidden, &l.gen_down_proj, error)) {
            return false;
        }
    }
    return true;
}

bool build_robolab_qwen_input_layout(std::vector<int> * input_ids,
                                     std::vector<int> * visual_indices,
                                     std::vector<int> * mrope_positions,
                                     bool include_prompt,
                                     std::string * error) {
    static constexpr int kSegmentHeader[17][7] = {
        {27, 15, 13, 15, 6486, 29, 151652},
        {27, 15, 13, 17, 6486, 29, 151652},
        {27, 15, 13, 18, 6486, 29, 151652},
        {27, 15, 13, 19, 6486, 29, 151652},
        {27, 15, 13, 21, 6486, 29, 151652},
        {27, 15, 13, 22, 6486, 29, 151652},
        {27, 15, 13, 23, 6486, 29, 151652},
        {27, 16, 13, 15, 6486, 29, 151652},
        {27, 16, 13, 16, 6486, 29, 151652},
        {27, 16, 13, 17, 6486, 29, 151652},
        {27, 16, 13, 19, 6486, 29, 151652},
        {27, 16, 13, 20, 6486, 29, 151652},
        {27, 16, 13, 21, 6486, 29, 151652},
        {27, 16, 13, 23, 6486, 29, 151652},
        {27, 16, 13, 24, 6486, 29, 151652},
        {27, 17, 13, 15, 6486, 29, 151652},
        {27, 17, 13, 17, 6486, 29, 151652},
    };
    static constexpr int kChatPrefix[] = {151644, 872, 198};
    static constexpr int kPrompt[] = {628, 279, 43096, 304, 279, 19212};
    static constexpr int kChatSuffix[] = {151645, 198, 151644, 77091, 198};
    const int expected_tokens = include_prompt ? kLanguageTokens : kLanguageUncondTokens;

    input_ids->clear();
    input_ids->reserve(static_cast<size_t>(expected_tokens));
    input_ids->insert(input_ids->end(), std::begin(kChatPrefix), std::end(kChatPrefix));
    for (int segment = 0; segment < 17; ++segment) {
        for (int i = 0; i < 7; ++i) input_ids->push_back(kSegmentHeader[segment][i]);
        input_ids->insert(input_ids->end(), kLanguageVisualFrameSeqLen, kLanguageVideoPadToken);
        input_ids->push_back(151653);
    }
    if (include_prompt) {
        input_ids->insert(input_ids->end(), std::begin(kPrompt), std::end(kPrompt));
    }
    input_ids->insert(input_ids->end(), std::begin(kChatSuffix), std::end(kChatSuffix));
    if (input_ids->size() != static_cast<size_t>(expected_tokens)) {
        *error = "Qwen3 RoboLab input id layout length mismatch";
        return false;
    }
    visual_indices->assign(static_cast<size_t>(expected_tokens), -1);
    int visual = 0;
    for (int i = 0; i < expected_tokens; ++i) {
        if ((*input_ids)[static_cast<size_t>(i)] == kLanguageVideoPadToken) {
            (*visual_indices)[static_cast<size_t>(i)] = visual++;
        }
    }
    if (visual != kVisualTokens) {
        *error = "Qwen3 RoboLab input id layout video token count mismatch";
        return false;
    }
    mrope_positions->assign(static_cast<size_t>(3) * expected_tokens, 1);
    int st = 0;
    for (int segment = 0; segment < 17; ++segment) {
        int ed = st;
        while (ed < expected_tokens && (*input_ids)[static_cast<size_t>(ed)] != kLanguageVideoPadToken) {
            ++ed;
        }
        if (ed >= expected_tokens) {
            *error = "Qwen3 MRoPE layout could not find expected video pad segment";
            return false;
        }
        int st_idx = 0;
        if (st > 0) {
            int prev_max = (*mrope_positions)[static_cast<size_t>(st - 1)];
            prev_max = std::max(prev_max, (*mrope_positions)[static_cast<size_t>(expected_tokens + st - 1)]);
            prev_max = std::max(prev_max, (*mrope_positions)[static_cast<size_t>(2 * expected_tokens + st - 1)]);
            st_idx = prev_max + 1;
        }
        const int text_len = ed - st;
        for (int i = 0; i < text_len; ++i) {
            const int pos = st_idx + i;
            const int token = st + i;
            (*mrope_positions)[static_cast<size_t>(token)] = pos;
            (*mrope_positions)[static_cast<size_t>(expected_tokens + token)] = pos;
            (*mrope_positions)[static_cast<size_t>(2 * expected_tokens + token)] = pos;
        }
        const int video_base = st_idx + text_len;
        for (int h = 0; h < kVisualGridH / kVisualMerge; ++h) {
            for (int w = 0; w < kVisualGridW / kVisualMerge; ++w) {
                const int local = h * (kVisualGridW / kVisualMerge) + w;
                const int token = ed + local;
                if (token >= expected_tokens ||
                    (*input_ids)[static_cast<size_t>(token)] != kLanguageVideoPadToken) {
                    *error = "Qwen3 MRoPE visual grid token layout mismatch";
                    return false;
                }
                (*mrope_positions)[static_cast<size_t>(token)] = video_base;
                (*mrope_positions)[static_cast<size_t>(expected_tokens + token)] = video_base + h;
                (*mrope_positions)[static_cast<size_t>(2 * expected_tokens + token)] = video_base + w;
            }
        }
        st = ed + kLanguageVisualFrameSeqLen;
    }
    if (st < expected_tokens) {
        int prev_max = (*mrope_positions)[static_cast<size_t>(st - 1)];
        prev_max = std::max(prev_max, (*mrope_positions)[static_cast<size_t>(expected_tokens + st - 1)]);
        prev_max = std::max(prev_max, (*mrope_positions)[static_cast<size_t>(2 * expected_tokens + st - 1)]);
        const int st_idx = prev_max + 1;
        for (int token = st; token < expected_tokens; ++token) {
            const int pos = st_idx + (token - st);
            (*mrope_positions)[static_cast<size_t>(token)] = pos;
            (*mrope_positions)[static_cast<size_t>(expected_tokens + token)] = pos;
            (*mrope_positions)[static_cast<size_t>(2 * expected_tokens + token)] = pos;
        }
    }
    return true;
}

std::vector<int> build_robolab_action_mrope_positions() {
    std::vector<int> positions(static_cast<size_t>(3) * kActionTokens, 0);
    // Official action packing calls get_3d_mrope_ids_vae_tokens with a 1x1
    // spatial grid, temporal_compression_factor=1, and start_frame_offset=1.
    // In this fixed RoboLab prompt layout the first visual grid starts at
    // temporal position 10, so action[0] aligns to temporal position 11.
    constexpr int kRobolabVisionStartTemporalOffset = 10;
    for (int i = 0; i < kActionTokens; ++i) {
        positions[static_cast<size_t>(i)] = kRobolabVisionStartTemporalOffset + 1 + i;
        positions[static_cast<size_t>(kActionTokens + i)] = 0;
        positions[static_cast<size_t>(2 * kActionTokens + i)] = 0;
    }
    return positions;
}

std::vector<float> build_robolab_mot_full_mrope_positions(int text_tokens) {
    std::vector<float> positions(static_cast<size_t>(3) * kMotFullTokens, 0.0f);
    const float base = 15000.0f + static_cast<float>(text_tokens);
    constexpr float kBaseFps = 24.0f;
    constexpr float kRobolabFps = 15.0f;
    constexpr float kVisionTemporalCompression = 4.0f;
    constexpr float kVisionTemporalStep = (kBaseFps / kVisionTemporalCompression) /
                                          (kRobolabFps / kVisionTemporalCompression);
    constexpr float kActionTemporalStep = (kBaseFps / kVisionTemporalCompression) / kRobolabFps;
    for (int t = 0; t < kMotVisionT; ++t) {
        for (int h = 0; h < kMotVisionH; ++h) {
            for (int w = 0; w < kMotVisionW; ++w) {
                const int token = (t * kMotVisionH + h) * kMotVisionW + w;
                positions[static_cast<size_t>(token)] =
                    base + static_cast<float>(t) * kVisionTemporalStep;
                positions[static_cast<size_t>(kMotFullTokens + token)] = static_cast<float>(h);
                positions[static_cast<size_t>(2 * kMotFullTokens + token)] = static_cast<float>(w);
            }
        }
    }
    for (int a = 0; a < kActionTokens; ++a) {
        const int token = kMotVisionTokens + a;
        positions[static_cast<size_t>(token)] =
            base + static_cast<float>(a) * kActionTemporalStep;
        positions[static_cast<size_t>(kMotFullTokens + token)] = 0.0f;
        positions[static_cast<size_t>(2 * kMotFullTokens + token)] = 0.0f;
    }
    return positions;
}

std::vector<float> build_robolab_mot_condition_mrope_positions(int text_tokens) {
    std::vector<float> positions(static_cast<size_t>(3) * text_tokens, 0.0f);
    for (int token = 0; token < text_tokens; ++token) {
        const float pos = static_cast<float>(token);
        positions[static_cast<size_t>(token)] = pos;
        positions[static_cast<size_t>(text_tokens + token)] = pos;
        positions[static_cast<size_t>(2 * text_tokens + token)] = pos;
    }
    return positions;
}

struct LanguageCudaRuntime {
    LanguageWeightsGpu weights;
    cudaStream_t stream = nullptr;
    int * input_ids_dev = nullptr;
    int * visual_indices_dev = nullptr;
    int * mrope_positions_dev = nullptr;
    int * input_ids_uncond_dev = nullptr;
    int * visual_indices_uncond_dev = nullptr;
    int * mrope_positions_uncond_dev = nullptr;
    int * mot_text_ids_dev = nullptr;
    int * mot_text_ids_uncond_dev = nullptr;
    float * x = nullptr;
    float * norm = nullptr;
    float * q = nullptr;
    float * k = nullptr;
    float * v = nullptr;
    float * q_norm = nullptr;
    float * k_norm = nullptr;
    float * q_rot = nullptr;
    float * k_rot = nullptr;
    float * attn = nullptr;
    float * o = nullptr;
    float * post_residual = nullptr;
    float * post_norm = nullptr;
    float * gate = nullptr;
    float * up = nullptr;
    float * swiglu = nullptr;
    float * down = nullptr;
    float * final_norm = nullptr;
    float * condition_input = nullptr;
    float * condition_input_uncond = nullptr;
    unsigned short * w8_x_bf16_workspace = nullptr;
    unsigned short * w8_y_bf16_workspace = nullptr;
    float * w8_c_tmp_workspace = nullptr;
    int * w8_int_workspace = nullptr;
    std::vector<int> default_input_ids;
    std::vector<int> default_visual_indices;
    std::vector<int> default_mrope_positions;
    std::vector<int> default_input_ids_uncond;
    std::vector<int> default_visual_indices_uncond;
    std::vector<int> default_mrope_positions_uncond;
    int last_run_tokens = 0;
    int last_layers_run = 0;
    int active_language_tokens = kLanguageTokens;
    int active_mot_text_tokens = kMotTextCondTokens;
    int active_mot_text_tokens_uncond = kMotTextUncondTokens;

    struct TraceConfig {
        bool enabled = false;
        bool simple_rope = false;
        bool timing = false;
        int layers = kLanguageLayers;
        int tokens = 0;
        int layer_index = 0;
        int cols = 16;
    };

    ~LanguageCudaRuntime() { release(); }
    LanguageCudaRuntime() = default;
    LanguageCudaRuntime(const LanguageCudaRuntime &) = delete;
    LanguageCudaRuntime & operator=(const LanguageCudaRuntime &) = delete;

    void release() {
        if (input_ids_dev) cudaFree(input_ids_dev);
        if (visual_indices_dev) cudaFree(visual_indices_dev);
        if (mrope_positions_dev) cudaFree(mrope_positions_dev);
        if (input_ids_uncond_dev) cudaFree(input_ids_uncond_dev);
        if (visual_indices_uncond_dev) cudaFree(visual_indices_uncond_dev);
        if (mrope_positions_uncond_dev) cudaFree(mrope_positions_uncond_dev);
        if (mot_text_ids_dev) cudaFree(mot_text_ids_dev);
        if (mot_text_ids_uncond_dev) cudaFree(mot_text_ids_uncond_dev);
        if (x) cudaFree(x);
        if (norm) cudaFree(norm);
        if (q) cudaFree(q);
        if (k) cudaFree(k);
        if (v) cudaFree(v);
        if (q_norm) cudaFree(q_norm);
        if (k_norm) cudaFree(k_norm);
        if (q_rot) cudaFree(q_rot);
        if (k_rot) cudaFree(k_rot);
        if (attn) cudaFree(attn);
        if (o) cudaFree(o);
        if (post_residual) cudaFree(post_residual);
        if (post_norm) cudaFree(post_norm);
        if (gate) cudaFree(gate);
        if (up) cudaFree(up);
        if (swiglu) cudaFree(swiglu);
        if (down) cudaFree(down);
        if (final_norm) cudaFree(final_norm);
        if (condition_input) cudaFree(condition_input);
        if (condition_input_uncond) cudaFree(condition_input_uncond);
        if (w8_x_bf16_workspace) cudaFree(w8_x_bf16_workspace);
        if (w8_y_bf16_workspace) cudaFree(w8_y_bf16_workspace);
        if (w8_c_tmp_workspace) cudaFree(w8_c_tmp_workspace);
        if (w8_int_workspace) cudaFree(w8_int_workspace);
        if (stream) cudaStreamDestroy(stream);
        input_ids_dev = nullptr;
        visual_indices_dev = nullptr;
        mrope_positions_dev = nullptr;
        input_ids_uncond_dev = nullptr;
        visual_indices_uncond_dev = nullptr;
        mrope_positions_uncond_dev = nullptr;
        mot_text_ids_dev = nullptr;
        mot_text_ids_uncond_dev = nullptr;
        x = norm = q = k = v = q_norm = k_norm = q_rot = k_rot = attn = o = nullptr;
        post_residual = post_norm = gate = up = swiglu = down = final_norm = nullptr;
        condition_input = nullptr;
        condition_input_uncond = nullptr;
        w8_x_bf16_workspace = nullptr;
        w8_y_bf16_workspace = nullptr;
        w8_c_tmp_workspace = nullptr;
        w8_int_workspace = nullptr;
        default_input_ids.clear();
        default_visual_indices.clear();
        default_mrope_positions.clear();
        default_input_ids_uncond.clear();
        default_visual_indices_uncond.clear();
        default_mrope_positions_uncond.clear();
        last_run_tokens = 0;
        last_layers_run = 0;
        active_language_tokens = kLanguageTokens;
        active_mot_text_tokens = kMotTextCondTokens;
        active_mot_text_tokens_uncond = kMotTextUncondTokens;
        stream = nullptr;
    }

    bool init(const GgufGpuResidentModel & model, std::string * error) {
        release();
        if (!bind_language_cuda_weights(model, &weights, error)) return false;
        if (!build_robolab_qwen_input_layout(&default_input_ids,
                                             &default_visual_indices,
                                             &default_mrope_positions,
                                             true,
                                             error) ||
            !build_robolab_qwen_input_layout(&default_input_ids_uncond,
                                             &default_visual_indices_uncond,
                                             &default_mrope_positions_uncond,
                                             false,
                                             error)) return false;
        auto fail = [&](const char * what) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 language CUDA init failed at ") + what + ": " +
                     cudaGetErrorString(st);
            release();
            return false;
        };
        if (cudaStreamCreate(&stream) != cudaSuccess) return fail("cudaStreamCreate");
        const size_t id_bytes = static_cast<size_t>(kLanguageMaxTokens) * sizeof(int);
        const size_t id_uncond_bytes = static_cast<size_t>(kLanguageMaxTokens) * sizeof(int);
        if (cudaMalloc(reinterpret_cast<void **>(&input_ids_dev), id_bytes) != cudaSuccess) return fail("input_ids");
        if (cudaMalloc(reinterpret_cast<void **>(&visual_indices_dev), id_bytes) != cudaSuccess) return fail("visual_indices");
        if (cudaMalloc(reinterpret_cast<void **>(&mrope_positions_dev), 3u * id_bytes) != cudaSuccess) return fail("mrope_positions");
        if (cudaMalloc(reinterpret_cast<void **>(&input_ids_uncond_dev), id_uncond_bytes) != cudaSuccess) return fail("input_ids_uncond");
        if (cudaMalloc(reinterpret_cast<void **>(&visual_indices_uncond_dev), id_uncond_bytes) != cudaSuccess) return fail("visual_indices_uncond");
        if (cudaMalloc(reinterpret_cast<void **>(&mrope_positions_uncond_dev), 3u * id_uncond_bytes) != cudaSuccess) return fail("mrope_positions_uncond");
        if (cudaMalloc(reinterpret_cast<void **>(&mot_text_ids_dev),
                       static_cast<size_t>(kMotTextMaxTokens) * sizeof(int)) != cudaSuccess) return fail("mot_text_ids");
        if (cudaMalloc(reinterpret_cast<void **>(&mot_text_ids_uncond_dev),
                       static_cast<size_t>(kMotTextMaxTokens) * sizeof(int)) != cudaSuccess) return fail("mot_text_ids_uncond");
        if (cudaMemcpyAsync(input_ids_dev, default_input_ids.data(), default_input_ids.size() * sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail("copy input_ids");
        if (cudaMemcpyAsync(visual_indices_dev, default_visual_indices.data(), default_visual_indices.size() * sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail("copy visual_indices");
        if (cudaMemcpyAsync(mrope_positions_dev, default_mrope_positions.data(), default_mrope_positions.size() * sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail("copy mrope_positions");
        if (cudaMemcpyAsync(input_ids_uncond_dev, default_input_ids_uncond.data(), default_input_ids_uncond.size() * sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail("copy input_ids_uncond");
        if (cudaMemcpyAsync(visual_indices_uncond_dev, default_visual_indices_uncond.data(), default_visual_indices_uncond.size() * sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail("copy visual_indices_uncond");
        if (cudaMemcpyAsync(mrope_positions_uncond_dev, default_mrope_positions_uncond.data(), default_mrope_positions_uncond.size() * sizeof(int), cudaMemcpyHostToDevice, stream) != cudaSuccess) return fail("copy mrope_positions_uncond");
        if (cudaMemcpyAsync(mot_text_ids_dev,
                            kRobolabMotCondTextIds.data(),
                            kRobolabMotCondTextIds.size() * sizeof(int),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) return fail("copy mot_text_ids");
        if (cudaMemcpyAsync(mot_text_ids_uncond_dev,
                            kRobolabMotUncondTextIds.data(),
                            kRobolabMotUncondTextIds.size() * sizeof(int),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) return fail("copy mot_text_ids_uncond");

        auto alloc_f32 = [&](float ** ptr, size_t elems, const char * label) -> bool {
            if (cudaMalloc(reinterpret_cast<void **>(ptr), elems * sizeof(float)) != cudaSuccess) {
                return fail(label);
            }
            return true;
        };
        auto alloc_u16 = [&](unsigned short ** ptr, size_t elems, const char * label) -> bool {
            if (cudaMalloc(reinterpret_cast<void **>(ptr), elems * sizeof(unsigned short)) != cudaSuccess) {
                return fail(label);
            }
            return true;
        };
        auto alloc_i32 = [&](int ** ptr, size_t elems, const char * label) -> bool {
            if (cudaMalloc(reinterpret_cast<void **>(ptr), elems * sizeof(int)) != cudaSuccess) {
                return fail(label);
            }
            return true;
        };
        const size_t h = static_cast<size_t>(kLanguageMaxTokens) * kLanguageHidden;
        const size_t kv = static_cast<size_t>(kLanguageMaxTokens) * kLanguageKv;
        const size_t inter = static_cast<size_t>(kLanguageMaxTokens) * kLanguageIntermediate;
        int dev = 0;
        int sms = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            sms <= 0) {
            return fail("language_w8_workspace_device");
        }
        const size_t w8_workspace_elems =
            static_cast<size_t>(kLanguageMaxTokens) *
            static_cast<size_t>(std::max(kLanguageHidden, kLanguageIntermediate));
        const size_t w8_c_tmp_elems = static_cast<size_t>(sms) * 64u * 256u;
        if (!alloc_f32(&x, h, "x") ||
            !alloc_f32(&norm, h, "norm") ||
            !alloc_f32(&q, h, "q") ||
            !alloc_f32(&k, kv, "k") ||
            !alloc_f32(&v, kv, "v") ||
            !alloc_f32(&q_norm, h, "q_norm") ||
            !alloc_f32(&k_norm, kv, "k_norm") ||
            !alloc_f32(&q_rot, h, "q_rot") ||
            !alloc_f32(&k_rot, kv, "k_rot") ||
            !alloc_f32(&attn, h, "attn") ||
            !alloc_f32(&o, h, "o") ||
            !alloc_f32(&post_residual, h, "post_residual") ||
            !alloc_f32(&post_norm, h, "post_norm") ||
            !alloc_f32(&gate, inter, "gate") ||
            !alloc_f32(&up, inter, "up") ||
            !alloc_f32(&swiglu, inter, "swiglu") ||
            !alloc_f32(&down, h, "down") ||
            !alloc_f32(&final_norm, h, "final_norm") ||
            !alloc_f32(&condition_input, h, "condition_input") ||
            !alloc_f32(&condition_input_uncond, h, "condition_input_uncond") ||
            !alloc_u16(&w8_x_bf16_workspace, w8_workspace_elems, "language_w8_x_bf16_workspace") ||
            !alloc_u16(&w8_y_bf16_workspace, w8_workspace_elems, "language_w8_y_bf16_workspace") ||
            !alloc_f32(&w8_c_tmp_workspace, w8_c_tmp_elems, "language_w8_c_tmp_workspace") ||
            !alloc_i32(&w8_int_workspace, static_cast<size_t>(sms), "language_w8_int_workspace")) {
            return false;
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) return fail("cudaStreamSynchronize");
        return true;
    }

    bool validate_i32_1d(const WamTensorView * tensor,
                         int max_tokens,
                         const char * name,
                         std::string * error) const {
        if (!tensor) return true;
        if (tensor->dtype != WamDType::I32 ||
            tensor->shape.size() != 1 ||
            tensor->shape[0] <= 0 ||
            tensor->shape[0] > static_cast<uint64_t>(max_tokens) ||
            tensor->bytes != static_cast<size_t>(tensor->shape[0]) * sizeof(int)) {
            *error = std::string(name) + " must be I32 [N] with 0 < N <= " +
                     std::to_string(max_tokens);
            return false;
        }
        return true;
    }

    bool validate_i32_2d_3xn(const WamTensorView * tensor,
                             int tokens,
                             const char * name,
                             std::string * error) const {
        if (!tensor) return true;
        if (tensor->dtype != WamDType::I32 ||
            tensor->shape.size() != 2 ||
            tensor->shape[0] != 3 ||
            tensor->shape[1] != static_cast<uint64_t>(tokens) ||
            tensor->bytes != static_cast<size_t>(3) * static_cast<size_t>(tokens) * sizeof(int)) {
            *error = std::string(name) + " must be I32 [3," + std::to_string(tokens) + "]";
            return false;
        }
        return true;
    }

    bool copy_i32_to_device(const int * src,
                            size_t count,
                            int * dst,
                            const char * name,
                            std::string * error) {
        if (!src || !dst || count == 0) {
            *error = std::string("Cosmos3 token upload received empty ") + name;
            return false;
        }
        if (cudaMemcpyAsync(dst,
                            src,
                            count * sizeof(int),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 token upload failed for ") + name + ": " +
                     cudaGetErrorString(st);
            return false;
        }
        return true;
    }

    bool configure_request_tokens(const WamTensorView * qwen_input_ids,
                                  const WamTensorView * qwen_visual_indices,
                                  const WamTensorView * qwen_mrope_positions,
                                  const WamTensorView * mot_text_ids,
                                  const WamTensorView * mot_text_ids_uncond,
                                  std::string * error) {
        const bool has_qwen = qwen_input_ids || qwen_visual_indices || qwen_mrope_positions;
        if (has_qwen && (!qwen_input_ids || !qwen_visual_indices || !qwen_mrope_positions)) {
            *error = "Cosmos3 dynamic Qwen tokens require input_ids, visual_indices, and mrope_positions";
            return false;
        }
        if (!validate_i32_1d(qwen_input_ids, kLanguageMaxTokens, "cosmos3.qwen.input_ids", error) ||
            !validate_i32_1d(qwen_visual_indices, kLanguageMaxTokens, "cosmos3.qwen.visual_indices", error)) {
            return false;
        }
        int qwen_tokens = kLanguageTokens;
        if (qwen_input_ids) {
            qwen_tokens = static_cast<int>(qwen_input_ids->shape[0]);
            if (qwen_visual_indices->shape[0] != static_cast<uint64_t>(qwen_tokens)) {
                *error = "Cosmos3 dynamic Qwen input_ids and visual_indices length mismatch";
                return false;
            }
        }
        if (!validate_i32_2d_3xn(qwen_mrope_positions, qwen_tokens, "cosmos3.qwen.mrope_positions", error) ||
            !validate_i32_1d(mot_text_ids, kMotTextMaxTokens, "cosmos3.mot.text_ids", error) ||
            !validate_i32_1d(mot_text_ids_uncond, kMotTextMaxTokens, "cosmos3.mot.text_ids_uncond", error)) {
            return false;
        }

        if (has_qwen) {
            if (!copy_i32_to_device(static_cast<const int *>(qwen_input_ids->data),
                                    static_cast<size_t>(qwen_tokens),
                                    input_ids_dev,
                                    "cosmos3.qwen.input_ids",
                                    error) ||
                !copy_i32_to_device(static_cast<const int *>(qwen_visual_indices->data),
                                    static_cast<size_t>(qwen_tokens),
                                    visual_indices_dev,
                                    "cosmos3.qwen.visual_indices",
                                    error) ||
                !copy_i32_to_device(static_cast<const int *>(qwen_mrope_positions->data),
                                    static_cast<size_t>(3) * static_cast<size_t>(qwen_tokens),
                                    mrope_positions_dev,
                                    "cosmos3.qwen.mrope_positions",
                                    error)) {
                return false;
            }
            active_language_tokens = qwen_tokens;
        } else {
            if (!copy_i32_to_device(default_input_ids.data(),
                                    default_input_ids.size(),
                                    input_ids_dev,
                                    "default qwen input_ids",
                                    error) ||
                !copy_i32_to_device(default_visual_indices.data(),
                                    default_visual_indices.size(),
                                    visual_indices_dev,
                                    "default qwen visual_indices",
                                    error) ||
                !copy_i32_to_device(default_mrope_positions.data(),
                                    default_mrope_positions.size(),
                                    mrope_positions_dev,
                                    "default qwen mrope_positions",
                                    error)) {
                return false;
            }
            active_language_tokens = static_cast<int>(default_input_ids.size());
        }

        if (mot_text_ids) {
            active_mot_text_tokens = static_cast<int>(mot_text_ids->shape[0]);
            if (!copy_i32_to_device(static_cast<const int *>(mot_text_ids->data),
                                    static_cast<size_t>(active_mot_text_tokens),
                                    mot_text_ids_dev,
                                    "cosmos3.mot.text_ids",
                                    error)) return false;
        } else {
            active_mot_text_tokens = kMotTextCondTokens;
            if (!copy_i32_to_device(kRobolabMotCondTextIds.data(),
                                    kRobolabMotCondTextIds.size(),
                                    mot_text_ids_dev,
                                    "default mot text_ids",
                                    error)) return false;
        }
        if (mot_text_ids_uncond) {
            active_mot_text_tokens_uncond = static_cast<int>(mot_text_ids_uncond->shape[0]);
            if (!copy_i32_to_device(static_cast<const int *>(mot_text_ids_uncond->data),
                                    static_cast<size_t>(active_mot_text_tokens_uncond),
                                    mot_text_ids_uncond_dev,
                                    "cosmos3.mot.text_ids_uncond",
                                    error)) return false;
        } else {
            active_mot_text_tokens_uncond = kMotTextUncondTokens;
            if (!copy_i32_to_device(kRobolabMotUncondTextIds.data(),
                                    kRobolabMotUncondTextIds.size(),
                                    mot_text_ids_uncond_dev,
                                    "default mot text_ids_uncond",
                                    error)) return false;
        }
        return true;
    }

    bool w8(const W8LinearGpu & linear,
            const float * in,
            float * out,
            int tokens,
            const char * label,
            int layer,
            std::string * error) {
        cudaGetLastError();
        if (cosmos3_w8a16_linear_f32_ws(in,
                                        linear.qweight,
                                        linear.scales,
                                        out,
                                        w8_x_bf16_workspace,
                                        w8_y_bf16_workspace,
                                        w8_c_tmp_workspace,
                                        w8_int_workspace,
                                        tokens,
                                        linear.in_features,
                                        linear.out_features,
                                        linear.qweight_rows,
                                        linear.qweight_cols,
                                        linear.scale_rows,
                                        linear.scale_cols,
                                        stream) != 0) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 language W8 linear launch failed at layer=") +
                     std::to_string(layer) + " proj=" + (label ? label : "?") +
                     " tokens=" + std::to_string(tokens) +
                     " in=" + std::to_string(linear.in_features) +
                     " out=" + std::to_string(linear.out_features) +
                     " qshape=[" + std::to_string(linear.qweight_rows) + "," +
                     std::to_string(linear.qweight_cols) + "]" +
                     " sshape=[" + std::to_string(linear.scale_rows) + "," +
                     std::to_string(linear.scale_cols) + "]" +
                     " cuda=" + cudaGetErrorString(st);
            return false;
        }
        return true;
    }

    bool append_prefix_tensor(std::vector<WamTensor> * tensors,
                              const char * name,
                              const float * src,
                              int rows,
                              int cols,
                              int src_cols,
                              std::string * error) {
        if (!tensors || !name || !src || rows <= 0 || cols <= 0 || src_cols <= 0) return true;
        WamTensor tensor;
        tensor.name = name;
        tensor.dtype = WamDType::F32;
        tensor.shape = {rows, cols};
        tensor.data.resize(static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float));
        cudaError_t st = cudaMemcpy2DAsync(tensor.data.data(),
                                           static_cast<size_t>(cols) * sizeof(float),
                                           src,
                                           static_cast<size_t>(src_cols) * sizeof(float),
                                           static_cast<size_t>(cols) * sizeof(float),
                                           static_cast<size_t>(rows),
                                           cudaMemcpyDeviceToHost,
                                           stream);
        if (st != cudaSuccess) {
            *error = std::string("Cosmos3 language trace copy failed for ") + name + ": " +
                     cudaGetErrorString(st);
            return false;
        }
        tensors->push_back(std::move(tensor));
        return true;
    }

    enum TimingSlot {
        T_INPUT = 0,
        T_INPUT_NORM,
        T_QKV_W8,
        T_QK_NORM,
        T_ROPE,
        T_ATTENTION,
        T_O_W8,
        T_POST_NORM,
        T_MLP_GATE_UP_W8,
        T_SWIGLU,
        T_DOWN_W8,
        T_MLP_RESIDUAL,
        T_DEEPSTACK,
        T_FINAL_NORM,
        T_COUNT,
    };

    bool append_timing_tensor(std::vector<WamTensor> * tensors,
                              const float * values,
                              int count) {
        if (!tensors || !values || count <= 0) return true;
        WamTensor tensor;
        tensor.name = "cosmos3.debug.language_timing_ms";
        tensor.dtype = WamDType::F32;
        tensor.shape = {1, count};
        tensor.data.resize(static_cast<size_t>(count) * sizeof(float));
        std::memcpy(tensor.data.data(), values, static_cast<size_t>(count) * sizeof(float));
        tensors->push_back(std::move(tensor));
        return true;
    }

    void append_timing_scalars(std::vector<WamTensor> * tensors,
                               const float * timing,
                               int count) {
        if (!tensors || !timing || count < T_COUNT) return;
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.input_ms", timing[T_INPUT]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.input_norm_ms", timing[T_INPUT_NORM]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.qkv_w8_ms", timing[T_QKV_W8]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.qk_norm_ms", timing[T_QK_NORM]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.mrope_ms", timing[T_ROPE]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.attention_ms", timing[T_ATTENTION]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.o_w8_ms", timing[T_O_W8]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.post_norm_ms", timing[T_POST_NORM]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.gate_up_w8_ms", timing[T_MLP_GATE_UP_W8]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.swiglu_ms", timing[T_SWIGLU]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.down_w8_ms", timing[T_DOWN_W8]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.residual_ms", timing[T_MLP_RESIDUAL]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.deepstack_ms", timing[T_DEEPSTACK]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.prefill.language.final_norm_ms", timing[T_FINAL_NORM]);
    }

    bool forward(const float * visual_tokens,
                 const float * deepstack0,
                 const float * deepstack1,
                 const float * deepstack2,
                 const WamTensorView * direct_qwen_input,
                 const TraceConfig & trace,
                 std::vector<WamTensor> * trace_tensors,
                 std::string * error) {
        if (!visual_tokens && !direct_qwen_input) {
            *error = "Cosmos3 language forward received null visual tokens";
            return false;
        }
        const float * deepstack[3] = {deepstack0, deepstack1, deepstack2};
        float timing[T_COUNT] = {};
        cudaEvent_t timing_start = nullptr;
        cudaEvent_t timing_end = nullptr;
        if (trace.timing) {
            cudaEventCreate(&timing_start);
            cudaEventCreate(&timing_end);
        }
        auto time_begin = [&]() {
            if (trace.timing) cudaEventRecord(timing_start, stream);
        };
        auto time_end = [&](int slot) {
            if (!trace.timing) return true;
            cudaEventRecord(timing_end, stream);
            cudaError_t st = cudaEventSynchronize(timing_end);
            if (st != cudaSuccess) {
                *error = std::string("Cosmos3 language timing failed: ") + cudaGetErrorString(st);
                return false;
            }
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, timing_start, timing_end);
            if (slot >= 0 && slot < T_COUNT) timing[slot] += ms;
            return true;
        };
        auto fail = [&](const char * what) {
            if (timing_start) cudaEventDestroy(timing_start);
            if (timing_end) cudaEventDestroy(timing_end);
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 language CUDA forward failed at ") + what + ": " +
                     cudaGetErrorString(st);
            return false;
        };
        time_begin();
        int run_tokens = active_language_tokens;
        if (direct_qwen_input) {
            if (direct_qwen_input->dtype != WamDType::F32 ||
                direct_qwen_input->shape.size() != 2 ||
                direct_qwen_input->shape[0] <= 0 ||
                direct_qwen_input->shape[0] > kLanguageMaxTokens ||
                direct_qwen_input->shape[1] != kLanguageHidden ||
                direct_qwen_input->bytes != static_cast<size_t>(direct_qwen_input->shape[0]) *
                                            static_cast<size_t>(kLanguageHidden) * sizeof(float)) {
                *error = "cosmos3.debug.qwen_input must be F32 [N,4096] with 0 < N <= 8192";
                return false;
            }
            run_tokens = static_cast<int>(direct_qwen_input->shape[0]);
            if (cudaMemcpyAsync(x,
                                direct_qwen_input->data,
                                direct_qwen_input->bytes,
                                cudaMemcpyHostToDevice,
                                stream) != cudaSuccess) {
                return fail("copy_direct_qwen_input");
            }
        } else if (cosmos3_qwen_build_robolab_input_f32(input_ids_dev,
                                                       visual_indices_dev,
                                                       weights.embed_tokens,
                                                       visual_tokens,
                                                       x,
                                                       active_language_tokens,
                                                       kLanguageHidden,
                                                       weights.vocab,
                                                       kVisualTokens,
                                                       stream) != 0) {
            return fail("input_splice");
        }
        if (!time_end(T_INPUT)) return false;
        if (cosmos3_qwen_embed_tokens_f32(mot_text_ids_dev,
                                          weights.embed_tokens,
                                          condition_input,
                                          active_mot_text_tokens,
                                          kLanguageHidden,
                                          weights.vocab,
                                          stream) != 0) {
            return fail("mot_text_embed_cond");
        }
        if (cosmos3_qwen_embed_tokens_f32(mot_text_ids_uncond_dev,
                                          weights.embed_tokens,
                                          condition_input_uncond,
                                          active_mot_text_tokens_uncond,
                                          kLanguageHidden,
                                          weights.vocab,
                                          stream) != 0) {
            return fail("mot_text_embed_uncond");
        }

        const int trace_rows = std::max(1, std::min(trace.tokens > 0 ? trace.tokens : 3, run_tokens));
        const int trace_cols = std::max(1, std::min(trace.cols, kLanguageIntermediate));
        if (trace.enabled && direct_qwen_input) run_tokens = trace_rows;
        const int layers_to_run = std::max(1, std::min(trace.enabled ? trace.layers : kLanguageLayers, kLanguageLayers));
        last_run_tokens = run_tokens;
        last_layers_run = layers_to_run;
        for (int layer = 0; layer < layers_to_run; ++layer) {
            const LanguageLayerGpu & l = weights.layers[static_cast<size_t>(layer)];
            if (trace.enabled && layer == trace.layer_index) {
                if (!append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.layer_input.prefix",
                                          x,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageHidden),
                                          kLanguageHidden,
                                          error)) return false;
            }
            time_begin();
            if (cosmos3_qwen_rmsnorm_f32(x, l.input_norm, norm,
                                         run_tokens, kLanguageHidden,
                                         kLanguageRmsEps, stream) != 0) return fail("input_rmsnorm");
            if (!time_end(T_INPUT_NORM)) return false;
            if (trace.enabled && layer == trace.layer_index) {
                if (!append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.attn_input.prefix",
                                          norm,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageHidden),
                                          kLanguageHidden,
                                          error)) return false;
            }
            time_begin();
            if (!w8(l.q_proj, norm, q, run_tokens, "q_proj", layer, error) ||
                !w8(l.k_proj, norm, k, run_tokens, "k_proj", layer, error) ||
                !w8(l.v_proj, norm, v, run_tokens, "v_proj", layer, error)) return false;
            if (!time_end(T_QKV_W8)) return false;
            if (trace.enabled && layer == trace.layer_index) {
                if (!append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.q.prefix",
                                          q,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageHidden),
                                          kLanguageHidden,
                                          error) ||
                    !append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.k.prefix",
                                          k,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageKv),
                                          kLanguageKv,
                                          error) ||
                    !append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.v.prefix",
                                          v,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageKv),
                                          kLanguageKv,
                                          error)) return false;
            }
            time_begin();
            if (cosmos3_qwen_head_rmsnorm_f32(q, l.q_norm, q_norm,
                                              run_tokens, kLanguageHidden,
                                              kLanguageQHeads, kLanguageHeadDim,
                                              kLanguageRmsEps, stream) != 0) return fail("q_head_rmsnorm");
            if (cosmos3_qwen_head_rmsnorm_f32(k, l.k_norm, k_norm,
                                              run_tokens, kLanguageKv,
                                              kLanguageKvHeads, kLanguageHeadDim,
                                              kLanguageRmsEps, stream) != 0) return fail("k_head_rmsnorm");
            if (!time_end(T_QK_NORM)) return false;
            if (trace.enabled && layer == trace.layer_index) {
                if (!append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.q_norm.prefix",
                                          q_norm,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageHidden),
                                          kLanguageHidden,
                                          error) ||
                    !append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.k_norm.prefix",
                                          k_norm,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageKv),
                                          kLanguageKv,
                                          error)) return false;
            }
            time_begin();
            if (trace.enabled && trace.simple_rope) {
                if (cosmos3_qwen_rope_f32(q_norm, q_rot,
                                          run_tokens, kLanguageHidden,
                                          kLanguageQHeads, kLanguageHeadDim,
                                          kLanguageRopeTheta, 1, stream) != 0) return fail("q_rope");
                if (cosmos3_qwen_rope_f32(k_norm, k_rot,
                                          run_tokens, kLanguageKv,
                                          kLanguageKvHeads, kLanguageHeadDim,
                                          kLanguageRopeTheta, 1, stream) != 0) return fail("k_rope");
            } else {
                if (cosmos3_qwen_mrope_f32(q_norm, mrope_positions_dev, q_rot,
                                           run_tokens, kLanguageHidden,
                                           kLanguageQHeads, kLanguageHeadDim,
                                           kLanguageRopeTheta, stream) != 0) return fail("q_mrope");
                if (cosmos3_qwen_mrope_f32(k_norm, mrope_positions_dev, k_rot,
                                           run_tokens, kLanguageKv,
                                           kLanguageKvHeads, kLanguageHeadDim,
                                           kLanguageRopeTheta, stream) != 0) return fail("k_mrope");
            }
            if (!time_end(T_ROPE)) return false;
            if (trace.enabled && layer == trace.layer_index) {
                if (!append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.q_rot.prefix",
                                          q_rot,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageHidden),
                                          kLanguageHidden,
                                          error) ||
                    !append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.k_rot.prefix",
                                          k_rot,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageKv),
                                          kLanguageKv,
                                          error)) return false;
            }
            time_begin();
            if (cosmos3_qwen_causal_gqa_attention_f32(q_rot, k_rot, v, attn,
                                                       run_tokens,
                                                       kLanguageHidden,
                                                       kLanguageKv,
                                                       kLanguageQHeads,
                                                       kLanguageKvHeads,
                                                       kLanguageHeadDim,
                                                       stream) != 0) return fail("causal_gqa_attention");
            if (!time_end(T_ATTENTION)) return false;
            if (trace.enabled && layer == trace.layer_index) {
                if (!append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.attn.prefix",
                                          attn,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageHidden),
                                          kLanguageHidden,
                                          error)) return false;
            }
            time_begin();
            if (!w8(l.o_proj, attn, o, run_tokens, "o_proj", layer, error)) return false;
            if (!time_end(T_O_W8)) return false;
            if (trace.enabled && layer == trace.layer_index) {
                if (!append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.o_proj.prefix",
                                          o,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageHidden),
                                          kLanguageHidden,
                                          error)) return false;
            }
            time_begin();
            if (cosmos3_qwen_residual_add_rmsnorm_f32(x, o, l.post_norm,
                                                      post_residual, post_norm,
                                                      run_tokens, kLanguageHidden,
                                                      kLanguageRmsEps, stream) != 0) return fail("post_attention_residual_rmsnorm");
            if (!time_end(T_POST_NORM)) return false;
            if (trace.enabled && layer == trace.layer_index) {
                if (!append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.post_residual.prefix",
                                          post_residual,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageHidden),
                                          kLanguageHidden,
                                          error) ||
                    !append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.post_norm.prefix",
                                          post_norm,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageHidden),
                                          kLanguageHidden,
                                          error)) return false;
            }
            time_begin();
            if (!w8(l.gate_proj, post_norm, gate, run_tokens, "gate_proj", layer, error) ||
                !w8(l.up_proj, post_norm, up, run_tokens, "up_proj", layer, error)) return false;
            if (!time_end(T_MLP_GATE_UP_W8)) return false;
            if (trace.enabled && layer == trace.layer_index) {
                if (!append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.gate.prefix",
                                          gate,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageIntermediate),
                                          kLanguageIntermediate,
                                          error) ||
                    !append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.up.prefix",
                                          up,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageIntermediate),
                                          kLanguageIntermediate,
                                          error)) return false;
            }
            time_begin();
            if (cosmos3_qwen_swiglu_f32(gate, up, swiglu,
                                        run_tokens * kLanguageIntermediate,
                                        stream) != 0) return fail("swiglu");
            if (!time_end(T_SWIGLU)) return false;
            if (trace.enabled && layer == trace.layer_index) {
                if (!append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.mlp_swiglu.prefix",
                                          swiglu,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageIntermediate),
                                          kLanguageIntermediate,
                                          error)) return false;
            }
            time_begin();
            if (!w8(l.down_proj, swiglu, down, run_tokens, "down_proj", layer, error)) return false;
            if (!time_end(T_DOWN_W8)) return false;
            if (trace.enabled && layer == trace.layer_index) {
                if (!append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.mlp_down.prefix",
                                          down,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageHidden),
                                          kLanguageHidden,
                                          error)) return false;
            }
            time_begin();
            if (cosmos3_qwen_residual_add_f32(post_residual, down, x,
                                              run_tokens * kLanguageHidden,
                                              stream) != 0) return fail("mlp_residual");
            if (!time_end(T_MLP_RESIDUAL)) return false;
            if (trace.enabled && layer == trace.layer_index) {
                if (!append_prefix_tensor(trace_tensors,
                                          "cosmos3.debug.layer0.final_residual.prefix",
                                          x,
                                          trace_rows,
                                          std::min(trace_cols, kLanguageHidden),
                                          kLanguageHidden,
                                          error)) return false;
            }
            if (layer < 3 && !direct_qwen_input) {
                time_begin();
                if (!deepstack[layer]) {
                    *error = "Cosmos3 language forward missing visual deepstack tensor";
                    return false;
                }
                if (cosmos3_qwen_add_visual_deepstack_f32(x,
                                                          visual_indices_dev,
                                                          deepstack[layer],
                                                          run_tokens,
                                                          kLanguageHidden,
                                                          kVisualTokens,
                                                          stream) != 0) {
                    return fail("deepstack_add");
                }
                if (!time_end(T_DEEPSTACK)) return false;
            }
        }
        time_begin();
        if (cosmos3_qwen_rmsnorm_f32(x, weights.final_norm, final_norm,
                                     run_tokens, kLanguageHidden,
                                     kLanguageRmsEps, stream) != 0) return fail("final_norm");
        if (!time_end(T_FINAL_NORM)) return false;
        if (trace.enabled) {
            if (!append_prefix_tensor(trace_tensors,
                                      "cosmos3.debug.layer0.final_norm.prefix",
                                      final_norm,
                                      trace_rows,
                                      std::min(trace_cols, kLanguageHidden),
                                      kLanguageHidden,
                                      error)) return false;
            if (trace.timing) {
                append_timing_tensor(trace_tensors, timing, T_COUNT);
            }
        }
        if (trace.timing) {
            append_timing_scalars(trace_tensors, timing, T_COUNT);
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) return fail("cudaStreamSynchronize");
        if (timing_start) cudaEventDestroy(timing_start);
        if (timing_end) cudaEventDestroy(timing_end);
        return true;
    }

    const float * final_norm_last_token() const {
        if (!final_norm || last_run_tokens <= 0 || last_layers_run != kLanguageLayers) return nullptr;
        return final_norm + static_cast<size_t>(last_run_tokens - 1) * kLanguageHidden;
    }

    const float * final_norm_tokens() const {
        if (!final_norm || last_run_tokens != active_language_tokens || last_layers_run != kLanguageLayers) return nullptr;
        return final_norm;
    }

    const float * condition_input_tokens() const {
        if (!condition_input) return nullptr;
        return condition_input;
    }

    const float * condition_input_tokens(bool uncond) const {
        return uncond ? condition_input_uncond : condition_input;
    }

    bool override_mot_condition_inputs(const WamTensorView * cond,
                                       const WamTensorView * uncond,
                                       std::string * error) {
        auto copy_one = [&](const WamTensorView * tensor,
                            float * dst,
                            int tokens,
                            const char * name) -> bool {
            if (!tensor) return true;
            if (tensor->dtype != WamDType::F32 ||
                tensor->shape.size() != 2 ||
                tensor->shape[0] != tokens ||
                tensor->shape[1] != kLanguageHidden ||
                tensor->bytes != static_cast<size_t>(tokens) *
                                 static_cast<size_t>(kLanguageHidden) * sizeof(float)) {
                *error = std::string(name) + " must be F32 [" +
                         std::to_string(tokens) + ",4096]";
                return false;
            }
            if (cudaMemcpyAsync(dst,
                                tensor->data,
                                tensor->bytes,
                                cudaMemcpyHostToDevice,
                                stream) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 MoT condition override copy failed for ") +
                         name + ": " + cudaGetErrorString(st);
                return false;
            }
            return true;
        };
        return copy_one(cond,
                        condition_input,
                        active_mot_text_tokens,
                        "cosmos3.debug.mot_text_hidden") &&
               copy_one(uncond,
                        condition_input_uncond,
                        active_mot_text_tokens_uncond,
                        "cosmos3.debug.mot_text_hidden_uncond");
    }

    int condition_token_count(bool uncond) const {
        return uncond ? active_mot_text_tokens_uncond : active_mot_text_tokens;
    }

    const int * mrope_positions() const { return mrope_positions_dev; }

    const int * mrope_positions(bool uncond) const {
        return uncond ? mrope_positions_uncond_dev : mrope_positions_dev;
    }

    cudaStream_t cuda_stream() const { return stream; }
};

struct ActionBridgeCudaRuntime {
    ActionBridgeWeightsGpu weights;
    float * llm2action_out = nullptr;
    float * cfg_cond_action_out = nullptr;
    float * cfg_uncond_action_out = nullptr;
    float * action2llm_out = nullptr;
    float * debug_llm_hidden = nullptr;
    unsigned short * llm_bf16_workspace = nullptr;
    unsigned short * action_bf16_workspace = nullptr;

    ~ActionBridgeCudaRuntime() { release(); }
    ActionBridgeCudaRuntime() = default;
    ActionBridgeCudaRuntime(const ActionBridgeCudaRuntime &) = delete;
    ActionBridgeCudaRuntime & operator=(const ActionBridgeCudaRuntime &) = delete;

    void release() {
        if (llm2action_out) cudaFree(llm2action_out);
        if (cfg_cond_action_out) cudaFree(cfg_cond_action_out);
        if (cfg_uncond_action_out) cudaFree(cfg_uncond_action_out);
        if (action2llm_out) cudaFree(action2llm_out);
        if (debug_llm_hidden) cudaFree(debug_llm_hidden);
        if (llm_bf16_workspace) cudaFree(llm_bf16_workspace);
        if (action_bf16_workspace) cudaFree(action_bf16_workspace);
        llm2action_out = nullptr;
        cfg_cond_action_out = nullptr;
        cfg_uncond_action_out = nullptr;
        action2llm_out = nullptr;
        debug_llm_hidden = nullptr;
        llm_bf16_workspace = nullptr;
        action_bf16_workspace = nullptr;
    }

    bool init(const GgufGpuResidentModel & model, std::string * error) {
        release();
        if (!bind_action_bridge_cuda_weights(model, &weights, error)) return false;
        if (cudaMalloc(reinterpret_cast<void **>(&llm2action_out),
                       static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&cfg_cond_action_out),
                       static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&cfg_uncond_action_out),
                       static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&action2llm_out),
                       static_cast<size_t>(kActionTokens) * kLanguageHidden * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&debug_llm_hidden),
                       static_cast<size_t>(kActionSteps) * kLanguageHidden * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&llm_bf16_workspace),
                       static_cast<size_t>(kActionTokens) * kLanguageHidden * sizeof(unsigned short)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&action_bf16_workspace),
                       static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(unsigned short)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 action bridge CUDA init failed: ") + cudaGetErrorString(st);
            release();
            return false;
        }
        return true;
    }

    bool llm_to_action_probe(const float * llm_hidden,
                             int domain_id,
                             cudaStream_t stream,
                             std::string * error) {
        // Diagnostic bridge only.  In the official action pathway llm2action is
        // applied to last_hidden_state[action.mse_loss_indexes] from the
        // denoise/MoT sequence, not to the final text token used by this probe.
        if (!llm_hidden) {
            *error = "Cosmos3 action bridge received null LLM hidden state";
            return false;
        }
        if (domain_id < 0 || domain_id >= kActionDomains) {
            *error = "Cosmos3 action bridge domain id out of range";
            return false;
        }
        if (cosmos3_domain_aware_linear_bf16_f32_ws(
                llm_hidden,
                weights.llm2action_fc,
                weights.llm2action_bias,
                llm2action_out,
                llm_bf16_workspace,
                1,
                kLanguageHidden,
                kActionMaxDim,
                kActionDomains,
                domain_id,
                stream) != 0) {
            *error = "Cosmos3 llm2action bridge CUDA launch failed";
            return false;
        }
        return true;
    }

    bool llm_to_noisy_action_tokens_from_tensor(const WamTensorView * hidden,
                                                int domain_id,
                                                cudaStream_t stream,
                                                std::string * error) {
        if (!hidden) return true;
        if (hidden->dtype != WamDType::F32 ||
            hidden->shape.size() != 2 ||
            hidden->shape[0] != kActionSteps ||
            hidden->shape[1] != kLanguageHidden ||
            hidden->bytes != static_cast<size_t>(kActionSteps) *
                             kLanguageHidden * sizeof(float)) {
            *error = "cosmos3.debug.torch_action_hidden must be F32 [32,4096]";
            return false;
        }
        if (cudaMemcpyAsync(debug_llm_hidden,
                            hidden->data,
                            hidden->bytes,
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 torch action hidden debug copy failed: ") +
                     cudaGetErrorString(st);
            return false;
        }
        return llm_to_noisy_action_tokens(debug_llm_hidden, domain_id, stream, error);
    }

    bool llm_to_action_tokens(const float * llm_hidden,
                              int tokens,
                              int domain_id,
                              cudaStream_t stream,
                              std::string * error) {
        if (!llm_hidden) {
            *error = "Cosmos3 action bridge received null MoT action hidden states";
            return false;
        }
        if (tokens <= 0 || tokens > kActionTokens) {
            *error = "Cosmos3 action bridge decode token count out of range";
            return false;
        }
        if (domain_id < 0 || domain_id >= kActionDomains) {
            *error = "Cosmos3 action bridge domain id out of range";
            return false;
        }
        if (cosmos3_llm2action_slice_f32_ws(
                llm_hidden,
                weights.llm2action_fc,
                weights.llm2action_bias,
                llm2action_out,
                action2llm_out,
                llm_bf16_workspace,
                tokens,
                kLanguageHidden,
                kActionMaxDim,
                kActionDim,
                kActionDomains,
                domain_id,
                stream) != 0) {
            *error = "Cosmos3 batched llm2action bridge CUDA launch failed";
            return false;
        }
        return true;
    }

    bool llm_to_noisy_action_tokens(const float * llm_hidden,
                                    int domain_id,
                                    cudaStream_t stream,
                                    std::string * error) {
        if (!llm_hidden) {
            *error = "Cosmos3 action bridge received null noisy MoT action hidden states";
            return false;
        }
        if (domain_id < 0 || domain_id >= kActionDomains) {
            *error = "Cosmos3 action bridge domain id out of range";
            return false;
        }
        if (cudaMemsetAsync(llm2action_out,
                            0,
                            static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(float),
                            stream) != cudaSuccess) {
            *error = "Cosmos3 action bridge velocity clear failed";
            return false;
        }
        if (cosmos3_llm2action_slice_f32_ws(
                llm_hidden,
                weights.llm2action_fc,
                weights.llm2action_bias,
                llm2action_out + static_cast<size_t>(kActionConditionTokens) * kActionMaxDim,
                action2llm_out,
                llm_bf16_workspace,
                kActionSteps,
                kLanguageHidden,
                kActionMaxDim,
                kActionDim,
                kActionDomains,
                domain_id,
                stream) != 0) {
            *error = "Cosmos3 noisy-action llm2action bridge CUDA launch failed";
            return false;
        }
        return true;
    }

    bool copy_action_output(std::vector<float> * action,
                            int tokens,
                            cudaStream_t stream,
                            std::string * error) const {
        if (!action || !action2llm_out) return true;
        action->assign(static_cast<size_t>(tokens) * kActionDim, 0.0f);
        if (cudaMemcpyAsync(action->data(),
                            action2llm_out,
                            action->size() * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 action output copy failed: ") +
                     cudaGetErrorString(st);
            return false;
        }
        return true;
    }

    const float * full_action_output() const { return llm2action_out; }
    float * mutable_full_action_output() { return llm2action_out; }

    bool save_cfg_cond_velocity(cudaStream_t stream, std::string * error) {
        if (cosmos3_mot_copy_f32(llm2action_out,
                                 cfg_cond_action_out,
                                 kActionTokens * kActionMaxDim,
                                 stream) != 0) {
            *error = "Cosmos3 CFG cond velocity save failed";
            return false;
        }
        return true;
    }

    bool save_cfg_uncond_velocity(cudaStream_t stream, std::string * error) {
        if (cosmos3_mot_copy_f32(llm2action_out,
                                 cfg_uncond_action_out,
                                 kActionTokens * kActionMaxDim,
                                 stream) != 0) {
            *error = "Cosmos3 CFG uncond velocity save failed";
            return false;
        }
        return true;
    }

    bool blend_cfg_velocity(float guidance, cudaStream_t stream, std::string * error) {
        if (cosmos3_mot_linear_combo4_f32(
                cfg_cond_action_out,
                llm2action_out,
                nullptr,
                nullptr,
                llm2action_out,
                kActionTokens * kActionMaxDim,
                guidance,
                1.0f - guidance,
                0.0f,
                0.0f,
                stream) != 0) {
            *error = "Cosmos3 CFG velocity blend failed";
            return false;
        }
        return true;
    }

    bool action_to_llm_token(const float * action_latent,
                             int domain_id,
                             cudaStream_t stream,
                             std::string * error) {
        if (!action_latent) {
            *error = "Cosmos3 action bridge received null action latent";
            return false;
        }
        if (domain_id < 0 || domain_id >= kActionDomains) {
            *error = "Cosmos3 action bridge domain id out of range";
            return false;
        }
        if (cosmos3_action2llm_plus_embed_f32_ws(
                action_latent,
                weights.action2llm_fc,
                weights.action2llm_bias,
                weights.action_modality_embed,
                action2llm_out,
                action_bf16_workspace,
                1,
                kActionDomains,
                domain_id,
                stream) != 0) {
            *error = "Cosmos3 action2llm bridge CUDA launch failed";
            return false;
        }
        return true;
    }

    bool action_to_llm_tokens(const float * action_latents,
                              int tokens,
                              int domain_id,
                              cudaStream_t stream,
                              std::string * error) {
        if (!action_latents) {
            *error = "Cosmos3 action bridge received null action latents";
            return false;
        }
        if (tokens <= 0 || tokens > kActionTokens) {
            *error = "Cosmos3 action bridge token count out of range";
            return false;
        }
        if (domain_id < 0 || domain_id >= kActionDomains) {
            *error = "Cosmos3 action bridge domain id out of range";
            return false;
        }
        if (cosmos3_action2llm_plus_embed_f32_ws(
                action_latents,
                weights.action2llm_fc,
                weights.action2llm_bias,
                weights.action_modality_embed,
                action2llm_out,
                action_bf16_workspace,
                tokens,
                kActionDomains,
                domain_id,
                stream) != 0) {
            *error = "Cosmos3 batched action2llm bridge CUDA launch failed";
            return false;
        }
        return true;
    }

    bool append_llm_to_action_tensor(std::vector<WamTensor> * tensors,
                                     cudaStream_t stream,
                                     std::string * error) const {
        if (!tensors || !llm2action_out) return true;
        WamTensor tensor;
        tensor.name = "cosmos3.debug.action_bridge.llm_to_action";
        tensor.dtype = WamDType::F32;
        tensor.shape = {1, kActionMaxDim};
        tensor.data.resize(static_cast<size_t>(kActionMaxDim) * sizeof(float));
        if (cudaMemcpyAsync(tensor.data.data(),
                            llm2action_out,
                            tensor.data.size(),
                            cudaMemcpyDeviceToHost,
                            stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 action bridge tensor copy failed: ") +
                     cudaGetErrorString(st);
            return false;
        }
        tensors->push_back(std::move(tensor));
        return true;
    }

    bool append_action_tensor(std::vector<WamTensor> * tensors,
                              const char * name,
                              const float * src,
                              cudaStream_t stream,
                              std::string * error) const {
        if (!tensors || !name || !src) return true;
        WamTensor tensor;
        tensor.name = name;
        tensor.dtype = WamDType::F32;
        tensor.shape = {kActionTokens, kActionMaxDim};
        tensor.data.resize(static_cast<size_t>(kActionTokens) *
                           static_cast<size_t>(kActionMaxDim) * sizeof(float));
        if (cudaMemcpyAsync(tensor.data.data(),
                            src,
                            tensor.data.size(),
                            cudaMemcpyDeviceToHost,
                            stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 action debug tensor copy failed: ") +
                     cudaGetErrorString(st);
            return false;
        }
        tensors->push_back(std::move(tensor));
        return true;
    }

    bool append_first_step_debug_tensors(std::vector<WamTensor> * tensors,
                                         cudaStream_t stream,
                                         std::string * error) const {
        return append_action_tensor(tensors,
                                    "cosmos3.debug.action.first_cond_velocity",
                                    cfg_cond_action_out,
                                    stream,
                                    error) &&
               append_action_tensor(tensors,
                                    "cosmos3.debug.action.first_uncond_velocity",
                                    cfg_uncond_action_out,
                                    stream,
                                    error) &&
               append_action_tensor(tensors,
                                    "cosmos3.debug.action.first_cfg_velocity",
                                    llm2action_out,
                                    stream,
                                    error);
    }

    bool append_step_debug_tensors(std::vector<WamTensor> * tensors,
                                   int step,
                                   cudaStream_t stream,
                                   std::string * error) const {
        const std::string base =
            "cosmos3.debug.action.all_steps.step" + std::to_string(step);
        return append_action_tensor(tensors,
                                    (base + ".cond_velocity").c_str(),
                                    cfg_cond_action_out,
                                    stream,
                                    error) &&
               append_action_tensor(tensors,
                                    (base + ".uncond_velocity").c_str(),
                                    cfg_uncond_action_out,
                                    stream,
                                    error) &&
               append_action_tensor(tensors,
                                    (base + ".cfg_velocity").c_str(),
                                    llm2action_out,
                                    stream,
                                    error);
    }
};

struct MotActionCudaRuntime {
    MotActionWeightsGpu weights;
    float * action_latents = nullptr;
    float * x0_current = nullptr;
    float * x0_prev = nullptr;
    float * x0_prev2 = nullptr;
    float * last_sample = nullptr;
    float * vision_latents = nullptr;
    float * vision_velocity = nullptr;
    float * vision_velocity_cond = nullptr;
    float * vision_velocity_uncond = nullptr;
    float * vision_x0_current = nullptr;
    float * vision_x0_prev = nullptr;
    float * vision_x0_prev2 = nullptr;
    float * vision_last_sample = nullptr;
    float * vision_hidden = nullptr;
    float * action_hidden = nullptr;
    float * timestep_hidden = nullptr;
    float * packed_hidden = nullptr;
    unsigned short * vision_bf16_workspace = nullptr;
    unsigned short * vision_hidden_bf16_workspace = nullptr;
    unsigned short * action_bf16_workspace = nullptr;
    unsigned short * time_freq_bf16_workspace = nullptr;
    float * time_mlp0_workspace = nullptr;
    unsigned short * time_mlp0_bf16_workspace = nullptr;
    int last_condition_tokens = kMotTextCondTokens;

    ~MotActionCudaRuntime() { release(); }
    MotActionCudaRuntime() = default;
    MotActionCudaRuntime(const MotActionCudaRuntime &) = delete;
    MotActionCudaRuntime & operator=(const MotActionCudaRuntime &) = delete;

    void release() {
        if (action_latents) cudaFree(action_latents);
        if (x0_current) cudaFree(x0_current);
        if (x0_prev) cudaFree(x0_prev);
        if (x0_prev2) cudaFree(x0_prev2);
        if (last_sample) cudaFree(last_sample);
        if (vision_latents) cudaFree(vision_latents);
        if (vision_velocity) cudaFree(vision_velocity);
        if (vision_velocity_cond) cudaFree(vision_velocity_cond);
        if (vision_velocity_uncond) cudaFree(vision_velocity_uncond);
        if (vision_x0_current) cudaFree(vision_x0_current);
        if (vision_x0_prev) cudaFree(vision_x0_prev);
        if (vision_x0_prev2) cudaFree(vision_x0_prev2);
        if (vision_last_sample) cudaFree(vision_last_sample);
        if (vision_hidden) cudaFree(vision_hidden);
        if (action_hidden) cudaFree(action_hidden);
        if (timestep_hidden) cudaFree(timestep_hidden);
        if (packed_hidden) cudaFree(packed_hidden);
        if (vision_bf16_workspace) cudaFree(vision_bf16_workspace);
        if (vision_hidden_bf16_workspace) cudaFree(vision_hidden_bf16_workspace);
        if (action_bf16_workspace) cudaFree(action_bf16_workspace);
        if (time_freq_bf16_workspace) cudaFree(time_freq_bf16_workspace);
        if (time_mlp0_workspace) cudaFree(time_mlp0_workspace);
        if (time_mlp0_bf16_workspace) cudaFree(time_mlp0_bf16_workspace);
        action_latents = nullptr;
        x0_current = nullptr;
        x0_prev = nullptr;
        x0_prev2 = nullptr;
        last_sample = nullptr;
        vision_latents = nullptr;
        vision_velocity = nullptr;
        vision_velocity_cond = nullptr;
        vision_velocity_uncond = nullptr;
        vision_x0_current = nullptr;
        vision_x0_prev = nullptr;
        vision_x0_prev2 = nullptr;
        vision_last_sample = nullptr;
        vision_hidden = nullptr;
        action_hidden = nullptr;
        timestep_hidden = nullptr;
        packed_hidden = nullptr;
        vision_bf16_workspace = nullptr;
        vision_hidden_bf16_workspace = nullptr;
        action_bf16_workspace = nullptr;
        time_freq_bf16_workspace = nullptr;
        time_mlp0_workspace = nullptr;
        time_mlp0_bf16_workspace = nullptr;
        last_condition_tokens = kMotTextCondTokens;
    }

    bool init(const GgufGpuResidentModel & model, std::string * error) {
        release();
        if (!bind_mot_action_cuda_weights(model, &weights, error)) return false;
        if (cudaMalloc(reinterpret_cast<void **>(&action_latents),
                       static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&x0_current),
                       static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&x0_prev),
                       static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&x0_prev2),
                       static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&last_sample),
                       static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&vision_latents),
                       static_cast<size_t>(kMotVisionTokens) * kMotVisionPatchDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&vision_velocity),
                       static_cast<size_t>(kMotVisionTokens) * kMotVisionPatchDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&vision_velocity_cond),
                       static_cast<size_t>(kMotVisionTokens) * kMotVisionPatchDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&vision_velocity_uncond),
                       static_cast<size_t>(kMotVisionTokens) * kMotVisionPatchDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&vision_x0_current),
                       static_cast<size_t>(kMotVisionTokens) * kMotVisionPatchDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&vision_x0_prev),
                       static_cast<size_t>(kMotVisionTokens) * kMotVisionPatchDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&vision_x0_prev2),
                       static_cast<size_t>(kMotVisionTokens) * kMotVisionPatchDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&vision_last_sample),
                       static_cast<size_t>(kMotVisionTokens) * kMotVisionPatchDim * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&vision_hidden),
                       static_cast<size_t>(kMotVisionTokens) * kLanguageHidden * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&action_hidden),
                       static_cast<size_t>(kActionTokens) * kLanguageHidden * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&timestep_hidden),
                       static_cast<size_t>(kLanguageHidden) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&packed_hidden),
                       static_cast<size_t>(kMotMaxPackedTokens) * kLanguageHidden * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&vision_bf16_workspace),
                       static_cast<size_t>(kMotVisionTokens) * kMotVisionPatchDim * sizeof(unsigned short)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&vision_hidden_bf16_workspace),
                       static_cast<size_t>(kMotVisionTokens) * kLanguageHidden * sizeof(unsigned short)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&action_bf16_workspace),
                       static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(unsigned short)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&time_freq_bf16_workspace),
                       static_cast<size_t>(kTimeFreqDim) * sizeof(unsigned short)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&time_mlp0_workspace),
                       static_cast<size_t>(kLanguageHidden) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&time_mlp0_bf16_workspace),
                       static_cast<size_t>(kLanguageHidden) * sizeof(unsigned short)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT action CUDA init failed: ") + cudaGetErrorString(st);
            release();
            return false;
        }
        if (cudaMemset(action_latents, 0,
                       static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(float)) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT action latent init failed: ") + cudaGetErrorString(st);
            release();
            return false;
        }
        return true;
    }

    bool forward(const WamTensorView * action_latents_input,
                 const float * condition_hidden,
                 int condition_tokens,
                 float timestep,
                 int domain_id,
                 cudaStream_t stream,
                 std::string * error) {
        if (!condition_hidden) {
            *error = "Cosmos3 MoT action packing received null condition hidden states";
            return false;
        }
        if (condition_tokens <= 0 || condition_tokens > kMotTextMaxTokens) {
            *error = "Cosmos3 MoT action packing received invalid condition token count";
            return false;
        }
        last_condition_tokens = condition_tokens;
        if (domain_id < 0 || domain_id >= kActionDomains) {
            *error = "Cosmos3 MoT action domain id out of range";
            return false;
        }
        if (action_latents_input) {
            if (action_latents_input->dtype != WamDType::F32 ||
                action_latents_input->shape.size() != 2 ||
                action_latents_input->shape[0] != kActionSteps ||
                (action_latents_input->shape[1] != kActionDim &&
                 action_latents_input->shape[1] != kActionMaxDim) ||
                action_latents_input->bytes != static_cast<size_t>(kActionSteps) *
                                               action_latents_input->shape[1] * sizeof(float)) {
                *error = "cosmos3.debug.action_latents must be F32 [32,8] or padded F32 [32,64]";
                return false;
            }
            if (action_latents_input->shape[1] == kActionDim) {
                std::vector<float> padded(static_cast<size_t>(kActionTokens) * kActionMaxDim, 0.0f);
                const float * src = static_cast<const float *>(action_latents_input->data);
                for (int t = 0; t < kActionSteps; ++t) {
                    std::copy(src + static_cast<size_t>(t) * kActionDim,
                              src + static_cast<size_t>(t + 1) * kActionDim,
                              padded.data() + static_cast<size_t>(t + kActionConditionTokens) * kActionMaxDim);
                }
                if (cudaMemcpyAsync(action_latents,
                                    padded.data(),
                                    padded.size() * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    stream) != cudaSuccess) {
                    cudaError_t st = cudaGetLastError();
                    *error = std::string("Cosmos3 MoT action latent copy failed: ") +
                             cudaGetErrorString(st);
                    return false;
                }
                return true;
            }
            if (cudaMemcpyAsync(action_latents,
                                action_latents_input->data,
                                action_latents_input->bytes,
                                cudaMemcpyHostToDevice,
                                stream) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 MoT action latent copy failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
        } else if (cudaMemsetAsync(action_latents, 0,
                                  static_cast<size_t>(kActionTokens) * kActionMaxDim * sizeof(float),
                                  stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT action latent zero failed: ") +
                     cudaGetErrorString(st);
            return false;
        }
        if (cosmos3_mot_build_action_input_f32_ws(
                action_latents,
                weights.bridge.action2llm_fc,
                weights.bridge.action2llm_bias,
                weights.bridge.action_modality_embed,
                weights.time_mlp0_weight,
                weights.time_mlp0_bias,
                weights.time_mlp2_weight,
                weights.time_mlp2_bias,
                action_hidden,
                timestep_hidden,
                action_bf16_workspace,
                time_freq_bf16_workspace,
                time_mlp0_workspace,
                time_mlp0_bf16_workspace,
                timestep,
                kTimestepScale,
                kActionTokens,
                kActionMaxDim,
                kLanguageHidden,
                kTimeFreqDim,
                kActionDomains,
                domain_id,
                kActionConditionTokens,
                stream) != 0) {
            *error = "Cosmos3 MoT action input CUDA launch failed";
            return false;
        }
        if (cosmos3_mot_pack_condition_action_f32(
                condition_hidden,
                action_hidden,
                packed_hidden,
                condition_tokens,
                kActionTokens,
                kLanguageHidden,
                stream) != 0) {
            *error = "Cosmos3 MoT condition/action packing CUDA launch failed";
            return false;
        }
        return true;
    }

    bool reset_latents(const WamTensorView * action_latents_input,
                       const WamTensorView * vision_latents_input,
                       const WamTensorView * clean_vision_condition_input,
                       const float * native_clean_vision_condition,
                       const std::vector<float> & state,
                       int seed,
                       cudaStream_t stream,
                       std::string * error) {
        if (vision_latents_input) {
            if (vision_latents_input->dtype != WamDType::F32 ||
                vision_latents_input->shape.size() != 2 ||
                vision_latents_input->shape[0] != kMotVisionTokens ||
                vision_latents_input->shape[1] != kMotVisionPatchDim ||
                vision_latents_input->bytes != static_cast<size_t>(kMotVisionTokens) *
                                               kMotVisionPatchDim * sizeof(float)) {
                *error = "cosmos3.debug.vision_latents must be F32 [3060,192]";
                return false;
            }
            if (cudaMemcpyAsync(vision_latents,
                                vision_latents_input->data,
                                vision_latents_input->bytes,
                                cudaMemcpyHostToDevice,
                                stream) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 MoT vision latent copy failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            if (clean_vision_condition_input) {
                if (clean_vision_condition_input->dtype != WamDType::F32 ||
                    clean_vision_condition_input->shape.size() != 2 ||
                    (clean_vision_condition_input->shape[0] != kMotVisionConditionTokens &&
                     clean_vision_condition_input->shape[0] != kMotVisionTokens) ||
                    clean_vision_condition_input->shape[1] != kMotVisionPatchDim ||
                    clean_vision_condition_input->bytes !=
                        static_cast<size_t>(clean_vision_condition_input->shape[0]) *
                        kMotVisionPatchDim * sizeof(float)) {
                    *error = "cosmos3.debug.clean_vision_condition must be F32 [340,192] or [3060,192]";
                    return false;
                }
                if (cudaMemcpyAsync(vision_latents,
                                    clean_vision_condition_input->data,
                                    static_cast<size_t>(kMotVisionConditionTokens) *
                                        kMotVisionPatchDim * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    stream) != cudaSuccess) {
                    cudaError_t st = cudaGetLastError();
                    *error = std::string("Cosmos3 MoT clean vision condition copy failed: ") +
                             cudaGetErrorString(st);
                    return false;
                }
            } else if (native_clean_vision_condition) {
                if (cudaMemcpyAsync(vision_latents,
                                    native_clean_vision_condition,
                                    static_cast<size_t>(kMotVisionConditionTokens) *
                                        kMotVisionPatchDim * sizeof(float),
                                    cudaMemcpyDeviceToDevice,
                                    stream) != cudaSuccess) {
                    cudaError_t st = cudaGetLastError();
                    *error = std::string("Cosmos3 MoT native clean vision condition copy failed: ") +
                             cudaGetErrorString(st);
                    return false;
                }
            }
        } else {
            NumpyRandomStateNormal rng(static_cast<uint32_t>(seed));
            const int padded_h = kMotVisionLatentH + (kMotVisionLatentH % 2);
            const int padded_w = kMotVisionLatentW + (kMotVisionLatentW % 2);
            std::vector<float> vision_source(
                static_cast<size_t>(kMotVisionVaeChannels) *
                    static_cast<size_t>(kMotVisionT) *
                    static_cast<size_t>(padded_h) *
                    static_cast<size_t>(padded_w),
                0.0f);
            std::vector<float> vision_patched(
                static_cast<size_t>(kMotVisionTokens) *
                    static_cast<size_t>(kMotVisionPatchDim),
                0.0f);
            for (int c = 0; c < kMotVisionVaeChannels; ++c) {
                for (int t = 0; t < kMotVisionT; ++t) {
                    for (int h = 0; h < kMotVisionLatentH; ++h) {
                        for (int w = 0; w < kMotVisionLatentW; ++w) {
                            const size_t src_idx =
                                (((static_cast<size_t>(c) * kMotVisionT + t) *
                                      padded_h + h) *
                                      padded_w + w);
                            vision_source[src_idx] =
                                round_to_bf16_value(rng.standard_normal_f32());
                        }
                    }
                }
            }
            for (int t = 0; t < kMotVisionT; ++t) {
                for (int h = 0; h < padded_h; ++h) {
                    for (int w = 0; w < padded_w; ++w) {
                        const int token = (t * kMotVisionH + (h / kMotVisionPatchSize)) *
                                          kMotVisionW + (w / kMotVisionPatchSize);
                        const int ph = h % kMotVisionPatchSize;
                        const int pw = w % kMotVisionPatchSize;
                        for (int c = 0; c < kMotVisionVaeChannels; ++c) {
                            const size_t src_idx =
                                (((static_cast<size_t>(c) * kMotVisionT + t) *
                                      padded_h + h) *
                                      padded_w + w);
                            const int dim =
                                ((ph * kMotVisionPatchSize + pw) *
                                 kMotVisionVaeChannels) + c;
                            vision_patched[static_cast<size_t>(token) *
                                               kMotVisionPatchDim + dim] =
                                (t == 0) ? 0.0f : vision_source[src_idx];
                        }
                    }
                }
            }
            if (clean_vision_condition_input) {
                if (clean_vision_condition_input->dtype != WamDType::F32 ||
                    clean_vision_condition_input->shape.size() != 2 ||
                    (clean_vision_condition_input->shape[0] != kMotVisionConditionTokens &&
                     clean_vision_condition_input->shape[0] != kMotVisionTokens) ||
                    clean_vision_condition_input->shape[1] != kMotVisionPatchDim ||
                    clean_vision_condition_input->bytes !=
                        static_cast<size_t>(clean_vision_condition_input->shape[0]) *
                        kMotVisionPatchDim * sizeof(float)) {
                    *error = "cosmos3.debug.clean_vision_condition must be F32 [340,192] or [3060,192]";
                    return false;
                }
                const float * clean_src = static_cast<const float *>(clean_vision_condition_input->data);
                for (int token = 0; token < kMotVisionConditionTokens; ++token) {
                    std::copy(clean_src + static_cast<size_t>(token) * kMotVisionPatchDim,
                              clean_src + static_cast<size_t>(token + 1) * kMotVisionPatchDim,
                              vision_patched.data() + static_cast<size_t>(token) * kMotVisionPatchDim);
                }
            }
            if (cudaMemcpyAsync(vision_latents,
                                vision_patched.data(),
                                static_cast<size_t>(kMotVisionTokens) *
                                    kMotVisionPatchDim * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 MoT vision latent noise init failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            if (!clean_vision_condition_input && native_clean_vision_condition) {
                if (cudaMemcpyAsync(vision_latents,
                                    native_clean_vision_condition,
                                    static_cast<size_t>(kMotVisionConditionTokens) *
                                        kMotVisionPatchDim * sizeof(float),
                                    cudaMemcpyDeviceToDevice,
                                    stream) != cudaSuccess) {
                    cudaError_t st = cudaGetLastError();
                    *error = std::string("Cosmos3 MoT native clean vision condition copy failed: ") +
                             cudaGetErrorString(st);
                    return false;
                }
            }
        }
        auto fill_condition_state = [&](std::vector<float> & padded) {
            if (state.size() >= static_cast<size_t>(kActionDim)) {
                for (int d = 0; d < kActionDim - 1; ++d) {
                    padded[static_cast<size_t>(d)] =
                        round_to_bf16_value(state[static_cast<size_t>(d)]);
                }
                padded[static_cast<size_t>(kActionDim - 1)] =
                    round_to_bf16_value(1.0f - state[static_cast<size_t>(kActionDim - 1)]);
            }
        };
        if (action_latents_input) {
            if (action_latents_input->dtype != WamDType::F32 ||
                action_latents_input->shape.size() != 2 ||
                action_latents_input->shape[0] != kActionSteps ||
                (action_latents_input->shape[1] != kActionDim &&
                 action_latents_input->shape[1] != kActionMaxDim) ||
                action_latents_input->bytes != static_cast<size_t>(kActionSteps) *
                                               action_latents_input->shape[1] * sizeof(float)) {
                *error = "cosmos3.debug.action_latents must be F32 [32,8] or padded F32 [32,64]";
                return false;
            }
            std::vector<float> padded(static_cast<size_t>(kActionTokens) * kActionMaxDim, 0.0f);
            fill_condition_state(padded);
            const float * src = static_cast<const float *>(action_latents_input->data);
            const int src_dim = static_cast<int>(action_latents_input->shape[1]);
            for (int t = 0; t < kActionSteps; ++t) {
                const int copy_dim = std::min(src_dim, kActionMaxDim);
                std::copy(src + static_cast<size_t>(t) * src_dim,
                          src + static_cast<size_t>(t) * src_dim + copy_dim,
                          padded.data() + static_cast<size_t>(t + kActionConditionTokens) * kActionMaxDim);
            }
            if (cudaMemcpyAsync(action_latents,
                                padded.data(),
                                padded.size() * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 MoT action latent copy failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            return true;
        }
        std::vector<float> noise(static_cast<size_t>(kActionTokens) * kActionMaxDim, 0.0f);
        NumpyRandomStateNormal rng(static_cast<uint32_t>(seed));
        for (int t = 0; t < kActionTokens; ++t) {
            for (int d = 0; d < kActionMaxDim; ++d) {
                noise[static_cast<size_t>(t) * kActionMaxDim + d] =
                    round_to_bf16_value(rng.standard_normal_f32());
            }
            for (int d = kActionDim; d < kActionMaxDim; ++d) {
                noise[static_cast<size_t>(t) * kActionMaxDim + d] = 0.0f;
            }
        }
        fill_condition_state(noise);
        if (cudaMemcpyAsync(action_latents,
                            noise.data(),
                            noise.size() * sizeof(float),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT action latent noise init failed: ") +
                     cudaGetErrorString(st);
            return false;
        }
        return true;
    }

    bool forward_current_latents(const float * condition_hidden,
                                 int condition_tokens,
                                 float timestep,
                                 int domain_id,
                                 cudaStream_t stream,
                                 std::string * error) {
        if (!condition_hidden) {
            *error = "Cosmos3 MoT action packing received null condition hidden states";
            return false;
        }
        if (condition_tokens <= 0 || condition_tokens > kMotTextMaxTokens) {
            *error = "Cosmos3 MoT action packing received invalid condition token count";
            return false;
        }
        last_condition_tokens = condition_tokens;
        if (domain_id < 0 || domain_id >= kActionDomains) {
            *error = "Cosmos3 MoT action domain id out of range";
            return false;
        }
        if (cosmos3_mot_vae2llm_f32_ws(
                vision_latents,
                weights.vae2llm_weight,
                weights.vae2llm_bias,
                vision_hidden,
                vision_bf16_workspace,
                kMotVisionTokens,
                kMotVisionPatchDim,
                kLanguageHidden,
                stream) != 0) {
            *error = "Cosmos3 MoT vae2llm vision input CUDA launch failed";
            return false;
        }
        if (cosmos3_mot_build_action_input_f32_ws(
                action_latents,
                weights.bridge.action2llm_fc,
                weights.bridge.action2llm_bias,
                weights.bridge.action_modality_embed,
                weights.time_mlp0_weight,
                weights.time_mlp0_bias,
                weights.time_mlp2_weight,
                weights.time_mlp2_bias,
                action_hidden,
                timestep_hidden,
                action_bf16_workspace,
                time_freq_bf16_workspace,
                time_mlp0_workspace,
                time_mlp0_bf16_workspace,
                timestep,
                kTimestepScale,
                kActionTokens,
                kActionMaxDim,
                kLanguageHidden,
                kTimeFreqDim,
                kActionDomains,
                domain_id,
                kActionConditionTokens,
                stream) != 0) {
            *error = "Cosmos3 MoT action input CUDA launch failed";
            return false;
        }
        if (cosmos3_mot_add_timestep_to_noisy_tokens_f32(
                vision_hidden,
                timestep_hidden,
                kMotVisionTokens,
                kMotVisionConditionTokens,
                kLanguageHidden,
                stream) != 0) {
            *error = "Cosmos3 MoT vision timestep embedding CUDA launch failed";
            return false;
        }
        if (cosmos3_mot_pack_text_vision_action_f32(
                condition_hidden,
                vision_hidden,
                action_hidden,
                packed_hidden,
                condition_tokens,
                kMotVisionTokens,
                kActionTokens,
                kLanguageHidden,
                stream) != 0) {
            *error = "Cosmos3 MoT text/vision/action packing CUDA launch failed";
            return false;
        }
        return true;
    }

    bool update_latents(const float * velocity,
                        int velocity_dim,
                        float step,
                        cudaStream_t stream,
                        std::string * error) {
        if (cosmos3_mot_update_action_latents_f32(
                action_latents,
                velocity,
                kActionTokens,
                kActionMaxDim,
                velocity_dim,
                kActionConditionTokens,
                step,
                stream) != 0) {
            *error = "Cosmos3 MoT action latent update CUDA launch failed";
            return false;
        }
        return true;
    }

    bool decode_vision_velocity(const float * vision_hidden_tokens,
                                cudaStream_t stream,
                                std::string * error) {
        if (!vision_hidden_tokens) {
            *error = "Cosmos3 MoT vision decode received null hidden states";
            return false;
        }
        if (cosmos3_mot_llm2vae_f32_ws(
                vision_hidden_tokens,
                weights.llm2vae_weight,
                weights.llm2vae_bias,
                vision_velocity,
                vision_hidden_bf16_workspace,
                kMotVisionTokens,
                kLanguageHidden,
                kMotVisionPatchDim,
                stream) != 0) {
            *error = "Cosmos3 MoT llm2vae vision velocity CUDA launch failed";
            return false;
        }
        if (cosmos3_mot_mask_action_velocity_f32(
                vision_velocity,
                kMotVisionTokens,
                kMotVisionPatchDim,
                kMotVisionPatchDim,
                kMotVisionConditionTokens,
                stream) != 0) {
            *error = "Cosmos3 MoT vision velocity mask CUDA launch failed";
            return false;
        }
        return true;
    }

    bool save_cfg_cond_vision_velocity(cudaStream_t stream, std::string * error) {
        if (cosmos3_mot_copy_f32(vision_velocity,
                                 vision_velocity_cond,
                                 kMotVisionTokens * kMotVisionPatchDim,
                                 stream) != 0) {
            *error = "Cosmos3 CFG cond vision velocity save failed";
            return false;
        }
        return true;
    }

    bool save_cfg_uncond_vision_velocity(cudaStream_t stream, std::string * error) {
        if (cosmos3_mot_copy_f32(vision_velocity,
                                 vision_velocity_uncond,
                                 kMotVisionTokens * kMotVisionPatchDim,
                                 stream) != 0) {
            *error = "Cosmos3 CFG uncond vision velocity save failed";
            return false;
        }
        return true;
    }

    bool blend_cfg_vision_velocity(float guidance, cudaStream_t stream, std::string * error) {
        if (cosmos3_mot_linear_combo4_f32(
                vision_velocity_cond,
                vision_velocity_uncond,
                nullptr,
                nullptr,
                vision_velocity,
                kMotVisionTokens * kMotVisionPatchDim,
                guidance,
                1.0f - guidance,
                0.0f,
                0.0f,
                stream) != 0) {
            *error = "Cosmos3 CFG vision velocity blend failed";
            return false;
        }
        return true;
    }

    bool compute_x0_from_velocity(const float * velocity,
                                  float sigma,
                                  cudaStream_t stream,
                                  std::string * error) {
        if (cosmos3_mot_compute_x0_from_flow_f32(
                action_latents,
                velocity,
                x0_current,
                kActionTokens * kActionMaxDim,
                sigma,
                stream) != 0) {
            *error = "Cosmos3 UniPC x0 conversion CUDA launch failed";
            return false;
        }
        if (cosmos3_mot_copy_f32(action_latents,
                                 x0_current,
                                 kActionConditionTokens * kActionMaxDim,
                                 stream) != 0) {
            *error = "Cosmos3 UniPC condition row restore failed";
            return false;
        }
        return true;
    }

    bool compute_vision_x0_from_velocity(float sigma,
                                         cudaStream_t stream,
                                         std::string * error) {
        if (cosmos3_mot_compute_x0_from_flow_f32(
                vision_latents,
                vision_velocity,
                vision_x0_current,
                kMotVisionTokens * kMotVisionPatchDim,
                sigma,
                stream) != 0) {
            *error = "Cosmos3 UniPC vision x0 conversion CUDA launch failed";
            return false;
        }
        return true;
    }

    bool mask_action_velocity(float * velocity,
                              cudaStream_t stream,
                              std::string * error) {
        if (cosmos3_mot_mask_action_velocity_f32(
                velocity,
                kActionTokens,
                kActionMaxDim,
                kActionDim,
                kActionConditionTokens,
                stream) != 0) {
            *error = "Cosmos3 action velocity mask CUDA launch failed";
            return false;
        }
        return true;
    }

    bool restore_condition_latents(cudaStream_t stream, std::string * error) {
        if (cosmos3_mot_copy_f32(x0_current,
                                 action_latents,
                                 kActionConditionTokens * kActionMaxDim,
                                 stream) != 0) {
            *error = "Cosmos3 UniPC condition latent restore CUDA launch failed";
            return false;
        }
        if (cosmos3_mot_copy_f32(vision_x0_current,
                                 vision_latents,
                                 kMotVisionConditionTokens * kMotVisionPatchDim,
                                 stream) != 0) {
            *error = "Cosmos3 UniPC vision condition latent restore CUDA launch failed";
            return false;
        }
        return true;
    }

    bool shift_unipc_history(cudaStream_t stream, std::string * error) {
        const int n = kActionTokens * kActionMaxDim;
        if (cosmos3_mot_copy_f32(x0_prev, x0_prev2, n, stream) != 0 ||
            cosmos3_mot_copy_f32(x0_current, x0_prev, n, stream) != 0) {
            *error = "Cosmos3 UniPC history shift CUDA launch failed";
            return false;
        }
        const int vn = kMotVisionTokens * kMotVisionPatchDim;
        if (cosmos3_mot_copy_f32(vision_x0_prev, vision_x0_prev2, vn, stream) != 0 ||
            cosmos3_mot_copy_f32(vision_x0_current, vision_x0_prev, vn, stream) != 0) {
            *error = "Cosmos3 UniPC vision history shift CUDA launch failed";
            return false;
        }
        return true;
    }

    bool save_last_sample(cudaStream_t stream, std::string * error) {
        if (cosmos3_mot_copy_f32(action_latents,
                                 last_sample,
                                 kActionTokens * kActionMaxDim,
                                 stream) != 0) {
            *error = "Cosmos3 UniPC last sample copy CUDA launch failed";
            return false;
        }
        if (cosmos3_mot_copy_f32(vision_latents,
                                 vision_last_sample,
                                 kMotVisionTokens * kMotVisionPatchDim,
                                 stream) != 0) {
            *error = "Cosmos3 UniPC vision last sample copy CUDA launch failed";
            return false;
        }
        return true;
    }

    bool unipc_corrector(int step_index,
                         int order,
                         const std::vector<float> & sigmas,
                         cudaStream_t stream,
                         std::string * error) {
        if (step_index <= 0 || order <= 0) return true;
        const float sigma_t = sigmas[static_cast<size_t>(step_index)];
        const float sigma_s0 = sigmas[static_cast<size_t>(step_index - 1)];
        const float alpha_t = 1.0f - sigma_t;
        const float lambda_t = cosmos3_lambda_from_sigma(sigma_t);
        const float lambda_s0 = cosmos3_lambda_from_sigma(sigma_s0);
        const float h = lambda_t - lambda_s0;
        const float h_phi_1 = cosmos3_unipc_h_phi1(h);
        const float b_h = h_phi_1;
        const float c_last = sigma_t / sigma_s0;
        float c_prev = -alpha_t * h_phi_1;
        float c_prev2 = 0.0f;
        float c_cur = 0.0f;
        if (order <= 1) {
            c_prev += alpha_t * b_h * 0.5f;
            c_cur = -alpha_t * b_h * 0.5f;
        } else {
            const float lambda_si = cosmos3_lambda_from_sigma(sigmas[static_cast<size_t>(step_index - 2)]);
            const float rk = (lambda_si - lambda_s0) / h;
            const float hh = -h;
            float h_phi_k = h_phi_1 / hh - 1.0f;
            const float b0 = h_phi_k / b_h;
            h_phi_k = h_phi_k / hh - 0.5f;
            const float b1 = h_phi_k * 2.0f / b_h;
            const float det = 1.0f - rk;
            const float rho0 = (b0 - b1) / det;
            const float rho1 = (b1 - rk * b0) / det;
            c_prev2 = -alpha_t * b_h * (rho0 / rk);
            c_prev += alpha_t * b_h * (rho0 / rk + rho1);
            c_cur = -alpha_t * b_h * rho1;
        }
        if (cosmos3_mot_linear_combo4_f32(
                last_sample,
                x0_prev,
                x0_prev2,
                x0_current,
                action_latents,
                kActionTokens * kActionMaxDim,
                c_last,
                c_prev,
                c_prev2,
                c_cur,
                stream) != 0) {
            *error = "Cosmos3 UniPC corrector CUDA launch failed";
            return false;
        }
        if (cosmos3_mot_linear_combo4_f32(
                vision_last_sample,
                vision_x0_prev,
                vision_x0_prev2,
                vision_x0_current,
                vision_latents,
                kMotVisionTokens * kMotVisionPatchDim,
                c_last,
                c_prev,
                c_prev2,
                c_cur,
                stream) != 0) {
            *error = "Cosmos3 UniPC vision corrector CUDA launch failed";
            return false;
        }
        return true;
    }

    bool unipc_predictor(int step_index,
                         int order,
                         const std::vector<float> & sigmas,
                         cudaStream_t stream,
                         std::string * error) {
        const float sigma_t = sigmas[static_cast<size_t>(step_index + 1)];
        const float sigma_s0 = sigmas[static_cast<size_t>(step_index)];
        const float alpha_t = 1.0f - sigma_t;
        const float lambda_t = cosmos3_lambda_from_sigma(sigma_t);
        const float lambda_s0 = cosmos3_lambda_from_sigma(sigma_s0);
        const float h = lambda_t - lambda_s0;
        const float h_phi_1 = cosmos3_unipc_h_phi1(h);
        const float b_h = h_phi_1;
        const float c_sample = sigma_t / sigma_s0;
        float c_cur = -alpha_t * h_phi_1;
        float c_prev = 0.0f;
        if (order >= 2) {
            const float lambda_si = cosmos3_lambda_from_sigma(sigmas[static_cast<size_t>(step_index - 1)]);
            const float rk = (lambda_si - lambda_s0) / h;
            c_prev = -alpha_t * b_h * (0.5f / rk);
            c_cur += alpha_t * b_h * (0.5f / rk);
        }
        if (cosmos3_mot_linear_combo4_f32(
                last_sample,
                x0_prev,
                x0_prev2,
                nullptr,
                action_latents,
                kActionTokens * kActionMaxDim,
                c_sample,
                c_cur,
                c_prev,
                0.0f,
                stream) != 0) {
            *error = "Cosmos3 UniPC predictor CUDA launch failed";
            return false;
        }
        if (cosmos3_mot_linear_combo4_f32(
                vision_last_sample,
                vision_x0_prev,
                vision_x0_prev2,
                nullptr,
                vision_latents,
                kMotVisionTokens * kMotVisionPatchDim,
                c_sample,
                c_cur,
                c_prev,
                0.0f,
                stream) != 0) {
            *error = "Cosmos3 UniPC vision predictor CUDA launch failed";
            return false;
        }
        return true;
    }

    bool copy_latent_action_output(std::vector<float> * action,
                                   cudaStream_t stream,
                                   std::string * error) const {
        if (!action || !action_latents) return true;
        std::vector<float> full(static_cast<size_t>(kActionTokens) * kActionMaxDim, 0.0f);
        if (cudaMemcpyAsync(full.data(),
                            action_latents,
                            full.size() * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 final action latent copy failed: ") +
                     cudaGetErrorString(st);
            return false;
        }
        action->assign(static_cast<size_t>(kActionSteps) * kActionDim, 0.0f);
        for (int t = 0; t < kActionSteps; ++t) {
            for (int d = 0; d < kActionDim; ++d) {
                (*action)[static_cast<size_t>(t) * kActionDim + d] =
                    full[static_cast<size_t>(t + kActionConditionTokens) * kActionMaxDim + d];
            }
        }
        return true;
    }

    bool append_action_state_tensor(std::vector<WamTensor> * tensors,
                                    const char * name,
                                    const float * src,
                                    cudaStream_t stream,
                                    std::string * error) const {
        if (!tensors || !name || !src) return true;
        WamTensor tensor;
        tensor.name = name;
        tensor.dtype = WamDType::F32;
        tensor.shape = {kActionTokens, kActionMaxDim};
        tensor.data.resize(static_cast<size_t>(kActionTokens) *
                           static_cast<size_t>(kActionMaxDim) * sizeof(float));
        if (cudaMemcpyAsync(tensor.data.data(),
                            src,
                            tensor.data.size(),
                            cudaMemcpyDeviceToHost,
                            stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 action state debug tensor copy failed: ") +
                     cudaGetErrorString(st);
            return false;
        }
        tensors->push_back(std::move(tensor));
        return true;
    }

    bool append_x0_debug_tensor(std::vector<WamTensor> * tensors,
                                cudaStream_t stream,
                                std::string * error) const {
        return append_action_state_tensor(tensors,
                                          "cosmos3.debug.action.first_x0",
                                          x0_current,
                                          stream,
                                          error);
    }

    bool append_step_x0_debug_tensor(std::vector<WamTensor> * tensors,
                                     int step,
                                     cudaStream_t stream,
                                     std::string * error) const {
        const std::string name =
            "cosmos3.debug.action.all_steps.step" + std::to_string(step) + ".x0";
        return append_action_state_tensor(tensors, name.c_str(), x0_current, stream, error);
    }

    bool append_step_latent_debug_tensor(std::vector<WamTensor> * tensors,
                                         int step,
                                         const char * stage,
                                         cudaStream_t stream,
                                         std::string * error) const {
        const std::string name =
            "cosmos3.debug.action.all_steps.step" + std::to_string(step) + "." + stage;
        return append_action_state_tensor(tensors, name.c_str(), action_latents, stream, error);
    }

    bool append_debug_tensors(std::vector<WamTensor> * tensors,
                              cudaStream_t stream,
                              std::string * error) const {
        return append_debug_tensors_with_prefix(tensors, "cosmos3.debug.mot", stream, error);
    }

    bool append_first_step_input_tensors(std::vector<WamTensor> * tensors,
                                         cudaStream_t stream,
                                         std::string * error) const {
        return append_debug_tensors_with_prefix(tensors, "cosmos3.debug.mot.first_step", stream, error);
    }

    bool append_debug_tensors_with_prefix(std::vector<WamTensor> * tensors,
                                          const char * prefix,
                                          cudaStream_t stream,
                                          std::string * error) const {
        if (!prefix) {
            *error = "Cosmos3 MoT debug tensor prefix is null";
            return false;
        }
        const std::string base(prefix);
        const std::string action_input_name = base + ".action_input.prefix";
        const std::string timestep_embed_name = base + ".timestep_embed.prefix";
        const std::string packed_condition_name = base + ".packed_condition.prefix";
        const std::string packed_vision_name = base + ".packed_vision.prefix";
        const std::string packed_action_name = base + ".packed_action.prefix";
        if (!append_cuda_prefix_tensor(tensors,
                                       action_input_name.c_str(),
                                       action_hidden,
                                       kActionTokens,
                                       kLanguageHidden,
                                       kLanguageHidden,
                                       error)) return false;
        if (!append_cuda_prefix_tensor(tensors,
                                       timestep_embed_name.c_str(),
                                       timestep_hidden,
                                       1,
                                       kLanguageHidden,
                                       kLanguageHidden,
                                       error)) return false;
        if (!append_cuda_prefix_tensor(tensors,
                                       packed_condition_name.c_str(),
                                       packed_hidden,
                                       3,
                                       16,
                                       kLanguageHidden,
                                       error)) return false;
        if (!append_cuda_prefix_tensor(tensors,
                                       packed_vision_name.c_str(),
                                       packed_hidden + static_cast<size_t>(last_condition_tokens) * kLanguageHidden,
                                       16,
                                       kLanguageHidden,
                                       kLanguageHidden,
                                       error)) return false;
        if (!append_cuda_prefix_tensor(tensors,
                                       packed_action_name.c_str(),
                                       packed_hidden + static_cast<size_t>(last_condition_tokens + kMotVisionTokens) *
                                                       kLanguageHidden,
                                       kActionTokens,
                                       kLanguageHidden,
                                       kLanguageHidden,
                                       error)) return false;
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT action debug sync failed: ") +
                     cudaGetErrorString(st);
            return false;
        }
        return true;
    }

    const float * action_input_tokens() const { return action_hidden; }
    const float * timestep_embedding() const { return timestep_hidden; }
    const float * packed_sequence_tokens() const { return packed_hidden; }
};

struct MotGenCudaRuntime {
    MotGenWeightsGpu weights;
    std::array<float *, 2> condition_mrope_positions_dev{{nullptr, nullptr}};
    std::array<float *, 2> full_mrope_positions_dev{{nullptr, nullptr}};
    float * und_x = nullptr;
    float * gen_x = nullptr;
    float * und_norm = nullptr;
    float * gen_norm = nullptr;
    float * q_und = nullptr;
    float * q_gen = nullptr;
    float * k_und = nullptr;
    float * v_und = nullptr;
    float * k_gen = nullptr;
    float * v_gen = nullptr;
    float * q_gen_norm = nullptr;
    float * q_und_norm = nullptr;
    float * k_und_norm = nullptr;
    float * k_gen_norm = nullptr;
    float * q_und_rot = nullptr;
    float * q_gen_rot = nullptr;
    float * k_und_rot = nullptr;
    float * k_gen_rot = nullptr;
    float * k_all = nullptr;
    float * v_all = nullptr;
    float * gen_attention_scores = nullptr;
    unsigned short * gen_attention_bf16_workspace = nullptr;
    unsigned short * w8_x_bf16_workspace = nullptr;
    unsigned short * w8_y_bf16_workspace = nullptr;
    float * w8_c_tmp_workspace = nullptr;
    int * w8_int_workspace = nullptr;
    float * attn_und = nullptr;
    float * attn_gen = nullptr;
    float * o_und = nullptr;
    float * o_gen = nullptr;
    float * residual_und = nullptr;
    float * residual_gen = nullptr;
    float * post_norm_und = nullptr;
    float * post_norm_gen = nullptr;
    float * gate_und = nullptr;
    float * up_und = nullptr;
    float * swiglu_und = nullptr;
    float * down_und = nullptr;
    float * gate = nullptr;
    float * up = nullptr;
    float * swiglu = nullptr;
    float * down = nullptr;
    float * layer0_gen_x = nullptr;
    float * layer0_trace_gen_norm = nullptr;
    float * layer0_trace_q_gen = nullptr;
    float * layer0_trace_k_gen = nullptr;
    float * layer0_trace_v_gen = nullptr;
    float * layer0_trace_q_gen_norm = nullptr;
    float * layer0_trace_k_gen_norm = nullptr;
    float * layer0_trace_q_gen_rot = nullptr;
    float * layer0_trace_k_gen_rot = nullptr;
    float * layer0_trace_attn_gen = nullptr;
    float * layer0_trace_o_gen = nullptr;
    float * layer0_trace_post_norm_gen = nullptr;
    float * layer0_trace_gate = nullptr;
    float * layer0_trace_up = nullptr;
    float * layer0_trace_swiglu = nullptr;
    float * layer0_trace_down = nullptr;
    float * layer0_trace_und_norm = nullptr;
    float * layer0_trace_q_und = nullptr;
    float * layer0_trace_k_und = nullptr;
    float * layer0_trace_v_und = nullptr;
    float * layer0_trace_q_und_norm = nullptr;
    float * layer0_trace_k_und_norm = nullptr;
    float * layer0_trace_q_und_rot = nullptr;
    float * layer0_trace_k_und_rot = nullptr;
    float * layer0_trace_attn_und = nullptr;
    float * layer0_trace_o_und = nullptr;
    float * layer0_trace_residual_und = nullptr;
    float * layer0_trace_post_norm_und = nullptr;
    float * layer0_trace_gate_und = nullptr;
    float * layer0_trace_up_und = nullptr;
    float * layer0_trace_swiglu_und = nullptr;
    float * layer0_trace_down_und = nullptr;
    float * layer0_trace_vision_gen_norm = nullptr;
    float * layer0_trace_vision_k_gen = nullptr;
    float * layer0_trace_vision_v_gen = nullptr;
    float * layer0_trace_vision_k_gen_norm = nullptr;
    float * layer0_trace_vision_k_gen_rot = nullptr;
    float * layer0_trace_k_gen_rot_full = nullptr;
    float * layer0_trace_v_gen_full = nullptr;
    std::array<float *, kMotLayerTraceLayers.size()> layer_trace_action_hidden{};
    std::array<float *, kMotLayerTraceLayers.size()> layer_trace_condition_hidden{};
    float * final_norm_gen = nullptr;
    std::array<float *, 2> cache_k_und_rot{{nullptr, nullptr}};
    std::array<float *, 2> cache_v_und{{nullptr, nullptr}};
    std::array<int, 2> cache_condition_tokens{{0, 0}};
    std::array<uint64_t, 2> cache_condition_keys{{0, 0}};
    std::array<bool, 2> cache_valid{{false, false}};
    std::array<int, 2> active_position_tokens{{0, 0}};
    int internal_trace_layer = 0;
    enum TimingSlot {
        MT_GEN_NORM = 0,
        MT_GEN_QKV_W8,
        MT_GEN_QK_NORM,
        MT_GEN_MROPE,
        MT_GEN_PACK_KV,
        MT_GEN_ATTENTION,
        MT_GEN_O_W8,
        MT_GEN_POST_NORM,
        MT_GEN_GATE_UP_W8,
        MT_GEN_SWIGLU,
        MT_GEN_DOWN_W8,
        MT_GEN_RESIDUAL,
        MT_GEN_FINAL_NORM,
        MT_COUNT,
    };
    bool timing_enabled = false;
    bool profile_enabled = false;
    bool internal_trace_enabled = false;
    bool layer_trace_enabled = false;
    int condition_cache_hits = 0;
    int condition_cache_builds = 0;
    std::array<float, MT_COUNT> timing_ms{};
    struct ProfileEvent {
        TimingSlot slot;
        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
    };
    std::vector<ProfileEvent> profile_events;

    ~MotGenCudaRuntime() { release(); }
    MotGenCudaRuntime() = default;
    MotGenCudaRuntime(const MotGenCudaRuntime &) = delete;
    MotGenCudaRuntime & operator=(const MotGenCudaRuntime &) = delete;

    void set_internal_trace_layer(int layer) {
        internal_trace_layer = std::max(0, std::min(layer, kLanguageLayers - 1));
    }

    void set_trace_enabled(bool internal_enabled, bool layer_enabled) {
        internal_trace_enabled = internal_enabled;
        layer_trace_enabled = layer_enabled;
    }

    void set_timing(bool enabled) {
        timing_enabled = enabled;
        timing_ms.fill(0.0f);
        condition_cache_hits = 0;
        condition_cache_builds = 0;
    }

    void clear_profile_events() {
        for (ProfileEvent & ev : profile_events) {
            if (ev.start) cudaEventDestroy(ev.start);
            if (ev.end) cudaEventDestroy(ev.end);
        }
        profile_events.clear();
    }

    void set_profile(bool enabled) {
        profile_enabled = enabled;
        clear_profile_events();
    }

    bool should_trace_internal_layer(int layer) const {
        return internal_trace_enabled && layer == internal_trace_layer;
    }

    void invalidate_condition_cache() {
        cache_condition_tokens = {{0, 0}};
        cache_condition_keys = {{0, 0}};
        cache_valid = {{false, false}};
    }

    void release() {
        for (float * ptr : condition_mrope_positions_dev) {
            if (ptr) cudaFree(ptr);
        }
        for (float * ptr : full_mrope_positions_dev) {
            if (ptr) cudaFree(ptr);
        }
        if (und_x) cudaFree(und_x);
        if (gen_x) cudaFree(gen_x);
        if (und_norm) cudaFree(und_norm);
        if (gen_norm) cudaFree(gen_norm);
        if (q_und) cudaFree(q_und);
        if (q_gen) cudaFree(q_gen);
        if (k_und) cudaFree(k_und);
        if (v_und) cudaFree(v_und);
        if (k_gen) cudaFree(k_gen);
        if (v_gen) cudaFree(v_gen);
        if (q_gen_norm) cudaFree(q_gen_norm);
        if (q_und_norm) cudaFree(q_und_norm);
        if (k_und_norm) cudaFree(k_und_norm);
        if (k_gen_norm) cudaFree(k_gen_norm);
        if (q_und_rot) cudaFree(q_und_rot);
        if (q_gen_rot) cudaFree(q_gen_rot);
        if (k_und_rot) cudaFree(k_und_rot);
        if (k_gen_rot) cudaFree(k_gen_rot);
        if (k_all) cudaFree(k_all);
        if (v_all) cudaFree(v_all);
        if (gen_attention_scores) cudaFree(gen_attention_scores);
        if (gen_attention_bf16_workspace) cudaFree(gen_attention_bf16_workspace);
        if (w8_x_bf16_workspace) cudaFree(w8_x_bf16_workspace);
        if (w8_y_bf16_workspace) cudaFree(w8_y_bf16_workspace);
        if (w8_c_tmp_workspace) cudaFree(w8_c_tmp_workspace);
        if (w8_int_workspace) cudaFree(w8_int_workspace);
        clear_profile_events();
        if (attn_und) cudaFree(attn_und);
        if (attn_gen) cudaFree(attn_gen);
        if (o_und) cudaFree(o_und);
        if (o_gen) cudaFree(o_gen);
        if (residual_und) cudaFree(residual_und);
        if (residual_gen) cudaFree(residual_gen);
        if (post_norm_und) cudaFree(post_norm_und);
        if (post_norm_gen) cudaFree(post_norm_gen);
        if (gate_und) cudaFree(gate_und);
        if (up_und) cudaFree(up_und);
        if (swiglu_und) cudaFree(swiglu_und);
        if (down_und) cudaFree(down_und);
        if (gate) cudaFree(gate);
        if (up) cudaFree(up);
        if (swiglu) cudaFree(swiglu);
        if (down) cudaFree(down);
        if (layer0_gen_x) cudaFree(layer0_gen_x);
        if (layer0_trace_gen_norm) cudaFree(layer0_trace_gen_norm);
        if (layer0_trace_q_gen) cudaFree(layer0_trace_q_gen);
        if (layer0_trace_k_gen) cudaFree(layer0_trace_k_gen);
        if (layer0_trace_v_gen) cudaFree(layer0_trace_v_gen);
        if (layer0_trace_q_gen_norm) cudaFree(layer0_trace_q_gen_norm);
        if (layer0_trace_k_gen_norm) cudaFree(layer0_trace_k_gen_norm);
        if (layer0_trace_q_gen_rot) cudaFree(layer0_trace_q_gen_rot);
        if (layer0_trace_k_gen_rot) cudaFree(layer0_trace_k_gen_rot);
        if (layer0_trace_attn_gen) cudaFree(layer0_trace_attn_gen);
        if (layer0_trace_o_gen) cudaFree(layer0_trace_o_gen);
        if (layer0_trace_post_norm_gen) cudaFree(layer0_trace_post_norm_gen);
        if (layer0_trace_gate) cudaFree(layer0_trace_gate);
        if (layer0_trace_up) cudaFree(layer0_trace_up);
        if (layer0_trace_swiglu) cudaFree(layer0_trace_swiglu);
        if (layer0_trace_down) cudaFree(layer0_trace_down);
        if (layer0_trace_und_norm) cudaFree(layer0_trace_und_norm);
        if (layer0_trace_q_und) cudaFree(layer0_trace_q_und);
        if (layer0_trace_k_und) cudaFree(layer0_trace_k_und);
        if (layer0_trace_v_und) cudaFree(layer0_trace_v_und);
        if (layer0_trace_q_und_norm) cudaFree(layer0_trace_q_und_norm);
        if (layer0_trace_k_und_norm) cudaFree(layer0_trace_k_und_norm);
        if (layer0_trace_q_und_rot) cudaFree(layer0_trace_q_und_rot);
        if (layer0_trace_k_und_rot) cudaFree(layer0_trace_k_und_rot);
        if (layer0_trace_attn_und) cudaFree(layer0_trace_attn_und);
        if (layer0_trace_o_und) cudaFree(layer0_trace_o_und);
        if (layer0_trace_residual_und) cudaFree(layer0_trace_residual_und);
        if (layer0_trace_post_norm_und) cudaFree(layer0_trace_post_norm_und);
        if (layer0_trace_gate_und) cudaFree(layer0_trace_gate_und);
        if (layer0_trace_up_und) cudaFree(layer0_trace_up_und);
        if (layer0_trace_swiglu_und) cudaFree(layer0_trace_swiglu_und);
        if (layer0_trace_down_und) cudaFree(layer0_trace_down_und);
        if (layer0_trace_vision_gen_norm) cudaFree(layer0_trace_vision_gen_norm);
        if (layer0_trace_vision_k_gen) cudaFree(layer0_trace_vision_k_gen);
        if (layer0_trace_vision_v_gen) cudaFree(layer0_trace_vision_v_gen);
        if (layer0_trace_vision_k_gen_norm) cudaFree(layer0_trace_vision_k_gen_norm);
        if (layer0_trace_vision_k_gen_rot) cudaFree(layer0_trace_vision_k_gen_rot);
        if (layer0_trace_k_gen_rot_full) cudaFree(layer0_trace_k_gen_rot_full);
        if (layer0_trace_v_gen_full) cudaFree(layer0_trace_v_gen_full);
        for (float * ptr : layer_trace_action_hidden) {
            if (ptr) cudaFree(ptr);
        }
        for (float * ptr : layer_trace_condition_hidden) {
            if (ptr) cudaFree(ptr);
        }
        if (final_norm_gen) cudaFree(final_norm_gen);
        for (float * ptr : cache_k_und_rot) {
            if (ptr) cudaFree(ptr);
        }
        for (float * ptr : cache_v_und) {
            if (ptr) cudaFree(ptr);
        }
        condition_mrope_positions_dev = {{nullptr, nullptr}};
        full_mrope_positions_dev = {{nullptr, nullptr}};
        und_x = gen_x = und_norm = gen_norm = q_und = q_gen = k_und = v_und = k_gen = v_gen = nullptr;
        q_gen_norm = q_und_norm = k_und_norm = k_gen_norm = q_und_rot = q_gen_rot = k_und_rot = k_gen_rot = nullptr;
        k_all = v_all = gen_attention_scores = nullptr;
        gen_attention_bf16_workspace = nullptr;
        w8_x_bf16_workspace = w8_y_bf16_workspace = nullptr;
        w8_c_tmp_workspace = nullptr;
        w8_int_workspace = nullptr;
        profile_enabled = false;
        attn_und = attn_gen = o_und = o_gen = residual_und = residual_gen = nullptr;
        post_norm_und = post_norm_gen = gate_und = up_und = swiglu_und = down_und = nullptr;
        gate = up = swiglu = down = layer0_gen_x = nullptr;
        layer0_trace_gen_norm = layer0_trace_q_gen = layer0_trace_k_gen = layer0_trace_v_gen = nullptr;
        layer0_trace_q_gen_norm = layer0_trace_k_gen_norm = nullptr;
        layer0_trace_q_gen_rot = layer0_trace_k_gen_rot = layer0_trace_attn_gen = layer0_trace_o_gen = nullptr;
        layer0_trace_post_norm_gen = layer0_trace_gate = layer0_trace_up = layer0_trace_swiglu = layer0_trace_down = nullptr;
        layer0_trace_und_norm = layer0_trace_q_und = layer0_trace_k_und = layer0_trace_v_und = nullptr;
        layer0_trace_q_und_norm = layer0_trace_k_und_norm = layer0_trace_q_und_rot = layer0_trace_k_und_rot = nullptr;
        layer0_trace_attn_und = layer0_trace_o_und = layer0_trace_residual_und = layer0_trace_post_norm_und = nullptr;
        layer0_trace_gate_und = layer0_trace_up_und = layer0_trace_swiglu_und = layer0_trace_down_und = nullptr;
        layer0_trace_vision_gen_norm = layer0_trace_vision_k_gen = layer0_trace_vision_v_gen = nullptr;
        layer0_trace_vision_k_gen_norm = layer0_trace_vision_k_gen_rot = nullptr;
        layer0_trace_k_gen_rot_full = layer0_trace_v_gen_full = final_norm_gen = nullptr;
        layer_trace_action_hidden.fill(nullptr);
        layer_trace_condition_hidden.fill(nullptr);
        cache_k_und_rot = {{nullptr, nullptr}};
        cache_v_und = {{nullptr, nullptr}};
        cache_condition_tokens = {{0, 0}};
        cache_condition_keys = {{0, 0}};
        cache_valid = {{false, false}};
        active_position_tokens = {{0, 0}};
        internal_trace_enabled = false;
        layer_trace_enabled = false;
    }

    bool init(const GgufGpuResidentModel & model, std::string * error) {
        release();
        if (!bind_mot_gen_cuda_weights(model, &weights, error)) return false;
        auto fail = [&](const char * what) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT gen CUDA init failed at ") + what + ": " +
                     cudaGetErrorString(st);
            release();
            return false;
        };
        auto alloc = [&](float ** ptr, size_t elems, const char * label) -> bool {
            if (cudaMalloc(reinterpret_cast<void **>(ptr), elems * sizeof(float)) != cudaSuccess) {
                return fail(label);
            }
            return true;
        };
        auto alloc_u16 = [&](unsigned short ** ptr, size_t elems, const char * label) -> bool {
            if (cudaMalloc(reinterpret_cast<void **>(ptr), elems * sizeof(unsigned short)) != cudaSuccess) {
                return fail(label);
            }
            return true;
        };
        auto alloc_i32 = [&](int ** ptr, size_t elems, const char * label) -> bool {
            if (cudaMalloc(reinterpret_cast<void **>(ptr), elems * sizeof(int)) != cudaSuccess) {
                return fail(label);
            }
            return true;
        };
        if (cudaMalloc(reinterpret_cast<void **>(&condition_mrope_positions_dev[0]),
                       static_cast<size_t>(3) * kMotTextMaxTokens * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&condition_mrope_positions_dev[1]),
                       static_cast<size_t>(3) * kMotTextMaxTokens * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&full_mrope_positions_dev[0]),
                       static_cast<size_t>(3) * kMotFullTokens * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void **>(&full_mrope_positions_dev[1]),
                       static_cast<size_t>(3) * kMotFullTokens * sizeof(float)) != cudaSuccess) {
            return fail("mot_mrope_positions");
        }
        if (!configure_position_tables(kMotTextCondTokens, kMotTextUncondTokens, nullptr, error)) {
            return false;
        }
        const size_t und_h = static_cast<size_t>(kMotTextMaxTokens) * kLanguageHidden;
        const size_t gen_h = static_cast<size_t>(kMotFullTokens) * kLanguageHidden;
        const size_t und_kv = static_cast<size_t>(kMotTextMaxTokens) * kLanguageKv;
        const size_t gen_kv = static_cast<size_t>(kMotFullTokens) * kLanguageKv;
        const size_t all_kv = static_cast<size_t>(kMotMaxPackedTokens) * kLanguageKv;
        const size_t gen_attention_score_elems =
            static_cast<size_t>(kMotFullTokens) *
            static_cast<size_t>(kMotMaxPackedTokens) *
            static_cast<size_t>(kLanguageQHeads / kLanguageKvHeads);
        const size_t gen_attention_bf16_workspace_elems =
            static_cast<size_t>(kMotFullTokens) * kLanguageHidden +
            static_cast<size_t>(kMotMaxPackedTokens) * kLanguageKv * 2u;
        int dev = 0;
        int sms = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
            sms <= 0) {
            return fail("mot_w8_workspace_device");
        }
        const size_t w8_workspace_elems =
            static_cast<size_t>(kMotFullTokens) *
            static_cast<size_t>(std::max(kLanguageHidden, kLanguageIntermediate));
        const size_t w8_c_tmp_elems =
            static_cast<size_t>(sms) * 64u * 256u;
        const size_t layer_und_kv = static_cast<size_t>(kLanguageLayers) * und_kv;
        const size_t gen_inter = static_cast<size_t>(kMotFullTokens) * kLanguageIntermediate;
        const size_t und_inter = static_cast<size_t>(kMotTextMaxTokens) * kLanguageIntermediate;
        const size_t action_h = static_cast<size_t>(kActionTokens) * kLanguageHidden;
        const size_t action_steps_h = static_cast<size_t>(kActionSteps) * kLanguageHidden;
        const size_t vision_trace_h = kMotVisionTraceRows.size() * static_cast<size_t>(kLanguageHidden);
        const size_t vision_trace_kv = kMotVisionTraceRows.size() * static_cast<size_t>(kLanguageKv);
        if (!alloc(&und_x, und_h, "und_x") ||
            !alloc(&gen_x, gen_h, "gen_x") ||
            !alloc(&und_norm, und_h, "und_norm") ||
            !alloc(&gen_norm, gen_h, "gen_norm") ||
            !alloc(&q_und, und_h, "q_und") ||
            !alloc(&q_gen, gen_h, "q_gen") ||
            !alloc(&k_und, und_kv, "k_und") ||
            !alloc(&v_und, und_kv, "v_und") ||
            !alloc(&k_gen, gen_kv, "k_gen") ||
            !alloc(&v_gen, gen_kv, "v_gen") ||
            !alloc(&q_gen_norm, gen_h, "q_gen_norm") ||
            !alloc(&q_und_norm, und_h, "q_und_norm") ||
            !alloc(&k_und_norm, und_kv, "k_und_norm") ||
            !alloc(&k_gen_norm, gen_kv, "k_gen_norm") ||
            !alloc(&q_und_rot, und_h, "q_und_rot") ||
            !alloc(&q_gen_rot, gen_h, "q_gen_rot") ||
            !alloc(&k_und_rot, und_kv, "k_und_rot") ||
            !alloc(&k_gen_rot, gen_kv, "k_gen_rot") ||
            !alloc(&k_all, all_kv, "k_all") ||
            !alloc(&v_all, all_kv, "v_all") ||
            !alloc(&gen_attention_scores, gen_attention_score_elems, "gen_attention_scores") ||
            !alloc_u16(&gen_attention_bf16_workspace,
                       gen_attention_bf16_workspace_elems,
                       "gen_attention_bf16_workspace") ||
            !alloc_u16(&w8_x_bf16_workspace, w8_workspace_elems, "w8_x_bf16_workspace") ||
            !alloc_u16(&w8_y_bf16_workspace, w8_workspace_elems, "w8_y_bf16_workspace") ||
            !alloc(&w8_c_tmp_workspace, w8_c_tmp_elems, "w8_c_tmp_workspace") ||
            !alloc_i32(&w8_int_workspace, static_cast<size_t>(sms), "w8_int_workspace") ||
            !alloc(&attn_und, und_h, "attn_und") ||
            !alloc(&attn_gen, gen_h, "attn_gen") ||
            !alloc(&o_und, und_h, "o_und") ||
            !alloc(&o_gen, gen_h, "o_gen") ||
            !alloc(&residual_und, und_h, "residual_und") ||
            !alloc(&residual_gen, gen_h, "residual_gen") ||
            !alloc(&post_norm_und, und_h, "post_norm_und") ||
            !alloc(&post_norm_gen, gen_h, "post_norm_gen") ||
            !alloc(&gate_und, und_inter, "gate_und") ||
            !alloc(&up_und, und_inter, "up_und") ||
            !alloc(&swiglu_und, und_inter, "swiglu_und") ||
            !alloc(&down_und, und_h, "down_und") ||
            !alloc(&gate, gen_inter, "gate") ||
            !alloc(&up, gen_inter, "up") ||
            !alloc(&swiglu, gen_inter, "swiglu") ||
            !alloc(&down, gen_h, "down") ||
            !alloc(&layer0_gen_x, gen_h, "layer0_gen_x") ||
            !alloc(&layer0_trace_gen_norm, action_h, "layer0_trace_gen_norm") ||
            !alloc(&layer0_trace_q_gen, action_h, "layer0_trace_q_gen") ||
            !alloc(&layer0_trace_k_gen, static_cast<size_t>(kActionTokens) * kLanguageKv, "layer0_trace_k_gen") ||
            !alloc(&layer0_trace_v_gen, static_cast<size_t>(kActionTokens) * kLanguageKv, "layer0_trace_v_gen") ||
            !alloc(&layer0_trace_q_gen_norm, action_h, "layer0_trace_q_gen_norm") ||
            !alloc(&layer0_trace_k_gen_norm, static_cast<size_t>(kActionTokens) * kLanguageKv, "layer0_trace_k_gen_norm") ||
            !alloc(&layer0_trace_q_gen_rot, action_h, "layer0_trace_q_gen_rot") ||
            !alloc(&layer0_trace_k_gen_rot, static_cast<size_t>(kActionTokens) * kLanguageKv, "layer0_trace_k_gen_rot") ||
            !alloc(&layer0_trace_attn_gen, action_h, "layer0_trace_attn_gen") ||
            !alloc(&layer0_trace_o_gen, action_h, "layer0_trace_o_gen") ||
            !alloc(&layer0_trace_post_norm_gen, action_h, "layer0_trace_post_norm_gen") ||
            !alloc(&layer0_trace_gate, static_cast<size_t>(kActionSteps) * kLanguageIntermediate, "layer0_trace_gate") ||
            !alloc(&layer0_trace_up, static_cast<size_t>(kActionSteps) * kLanguageIntermediate, "layer0_trace_up") ||
            !alloc(&layer0_trace_swiglu, static_cast<size_t>(kActionSteps) * kLanguageIntermediate, "layer0_trace_swiglu") ||
            !alloc(&layer0_trace_down, action_h, "layer0_trace_down") ||
            !alloc(&layer0_trace_und_norm, und_h, "layer0_trace_und_norm") ||
            !alloc(&layer0_trace_q_und, und_h, "layer0_trace_q_und") ||
            !alloc(&layer0_trace_k_und, und_kv, "layer0_trace_k_und") ||
            !alloc(&layer0_trace_v_und, und_kv, "layer0_trace_v_und") ||
            !alloc(&layer0_trace_q_und_norm, und_h, "layer0_trace_q_und_norm") ||
            !alloc(&layer0_trace_k_und_norm, und_kv, "layer0_trace_k_und_norm") ||
            !alloc(&layer0_trace_q_und_rot, und_h, "layer0_trace_q_und_rot") ||
            !alloc(&layer0_trace_k_und_rot, und_kv, "layer0_trace_k_und_rot") ||
            !alloc(&layer0_trace_attn_und, und_h, "layer0_trace_attn_und") ||
            !alloc(&layer0_trace_o_und, und_h, "layer0_trace_o_und") ||
            !alloc(&layer0_trace_residual_und, und_h, "layer0_trace_residual_und") ||
            !alloc(&layer0_trace_post_norm_und, und_h, "layer0_trace_post_norm_und") ||
            !alloc(&layer0_trace_gate_und, und_inter, "layer0_trace_gate_und") ||
            !alloc(&layer0_trace_up_und, und_inter, "layer0_trace_up_und") ||
            !alloc(&layer0_trace_swiglu_und, und_inter, "layer0_trace_swiglu_und") ||
            !alloc(&layer0_trace_down_und, und_h, "layer0_trace_down_und") ||
            !alloc(&layer0_trace_vision_gen_norm, vision_trace_h, "layer0_trace_vision_gen_norm") ||
            !alloc(&layer0_trace_vision_k_gen, vision_trace_kv, "layer0_trace_vision_k_gen") ||
            !alloc(&layer0_trace_vision_v_gen, vision_trace_kv, "layer0_trace_vision_v_gen") ||
            !alloc(&layer0_trace_vision_k_gen_norm, vision_trace_kv, "layer0_trace_vision_k_gen_norm") ||
            !alloc(&layer0_trace_vision_k_gen_rot, vision_trace_kv, "layer0_trace_vision_k_gen_rot") ||
            !alloc(&layer0_trace_k_gen_rot_full, gen_kv, "layer0_trace_k_gen_rot_full") ||
            !alloc(&layer0_trace_v_gen_full, gen_kv, "layer0_trace_v_gen_full") ||
            !alloc(&final_norm_gen, gen_h, "final_norm_gen") ||
            !alloc(&cache_k_und_rot[0], layer_und_kv, "cache_k_und_rot_cond") ||
            !alloc(&cache_v_und[0], layer_und_kv, "cache_v_und_cond") ||
            !alloc(&cache_k_und_rot[1], layer_und_kv, "cache_k_und_rot_uncond") ||
            !alloc(&cache_v_und[1], layer_und_kv, "cache_v_und_uncond")) {
            return false;
        }
        for (size_t i = 0; i < kMotLayerTraceLayers.size(); ++i) {
            if (!alloc(&layer_trace_action_hidden[i],
                       action_steps_h,
                       "layer_trace_action_hidden")) {
                return false;
            }
            if (!alloc(&layer_trace_condition_hidden[i],
                       und_h,
                       "layer_trace_condition_hidden")) {
                return false;
            }
        }
        return true;
    }

    bool configure_position_tables(int cond_tokens,
                                   int uncond_tokens,
                                   cudaStream_t stream,
                                   std::string * error) {
        if (cond_tokens <= 0 || cond_tokens > kMotTextMaxTokens ||
            uncond_tokens <= 0 || uncond_tokens > kMotTextMaxTokens) {
            *error = "Cosmos3 MoT position table token count out of range";
            return false;
        }
        const std::array<int, 2> requested{{cond_tokens, uncond_tokens}};
        for (int slot = 0; slot < 2; ++slot) {
            if (active_position_tokens[static_cast<size_t>(slot)] == requested[static_cast<size_t>(slot)]) {
                continue;
            }
            const int tokens = requested[static_cast<size_t>(slot)];
            const std::vector<float> condition_mrope = build_robolab_mot_condition_mrope_positions(tokens);
            const std::vector<float> full_mrope = build_robolab_mot_full_mrope_positions(tokens);
            if (cudaMemcpyAsync(condition_mrope_positions_dev[static_cast<size_t>(slot)],
                                condition_mrope.data(),
                                condition_mrope.size() * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream) != cudaSuccess ||
                cudaMemcpyAsync(full_mrope_positions_dev[static_cast<size_t>(slot)],
                                full_mrope.data(),
                                full_mrope.size() * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 MoT position table copy failed: ") +
                         cudaGetErrorString(st);
                return false;
            }
            active_position_tokens[static_cast<size_t>(slot)] = tokens;
            cache_valid[static_cast<size_t>(slot)] = false;
            cache_condition_tokens[static_cast<size_t>(slot)] = 0;
            cache_condition_keys[static_cast<size_t>(slot)] = 0;
        }
        return true;
    }

    bool copy_layer0_action_slice(float * dst,
                                  const float * src,
                                  int width,
                                  cudaStream_t stream,
                                  std::string * error,
                                  const char * label) {
        if (cudaMemcpyAsync(dst,
                            src + static_cast<size_t>(kMotVisionTokens + kActionConditionTokens) * width,
                            static_cast<size_t>(kActionSteps) * width * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT layer0 trace copy failed at ") +
                     label + ": " + cudaGetErrorString(st);
            return false;
        }
        return true;
    }

    bool copy_layer0_condition_slice(float * dst,
                                     const float * src,
                                     int rows,
                                     int width,
                                     cudaStream_t stream,
                                     std::string * error,
                                     const char * label) {
        if (rows <= 0 || rows > kMotTextMaxTokens) {
            *error = "Cosmos3 MoT layer0 condition trace received invalid row count";
            return false;
        }
        if (cudaMemcpyAsync(dst,
                            src,
                            static_cast<size_t>(rows) * width * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT layer0 condition trace copy failed at ") +
                     label + ": " + cudaGetErrorString(st);
            return false;
        }
        return true;
    }

    bool copy_layer0_vision_sample_rows(float * dst,
                                        const float * src,
                                        int width,
                                        cudaStream_t stream,
                                        std::string * error,
                                        const char * label) {
        for (size_t i = 0; i < kMotVisionTraceRows.size(); ++i) {
            const int row = kMotVisionTraceRows[i];
            if (cudaMemcpyAsync(dst + i * static_cast<size_t>(width),
                                src + static_cast<size_t>(row) * width,
                                static_cast<size_t>(width) * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess) {
                cudaError_t st = cudaGetLastError();
                *error = std::string("Cosmos3 MoT layer0 sampled vision trace copy failed at ") +
                         label + ": " + cudaGetErrorString(st);
                return false;
            }
        }
        return true;
    }

    int layer_trace_slot(int layer) const {
        for (size_t i = 0; i < kMotLayerTraceLayers.size(); ++i) {
            if (kMotLayerTraceLayers[i] == layer) return static_cast<int>(i);
        }
        return -1;
    }

    bool copy_layer_action_hidden_trace(int layer,
                                        const float * src,
                                        cudaStream_t stream,
                                        std::string * error,
                                        const char * label) {
        const int slot = layer_trace_slot(layer);
        if (!layer_trace_enabled) return true;
        if (slot < 0) return true;
        if (cudaMemcpyAsync(layer_trace_action_hidden[static_cast<size_t>(slot)],
                            src + static_cast<size_t>(kMotVisionTokens + kActionConditionTokens) * kLanguageHidden,
                            static_cast<size_t>(kActionSteps) * kLanguageHidden * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT layer action hidden trace copy failed at ") +
                     label + ": " + cudaGetErrorString(st);
            return false;
        }
        return true;
    }

    bool copy_layer_condition_hidden_trace(int layer,
                                           const float * src,
                                           int condition_tokens,
                                           cudaStream_t stream,
                                           std::string * error,
                                           const char * label) {
        const int slot = layer_trace_slot(layer);
        if (!layer_trace_enabled) return true;
        if (slot < 0) return true;
        if (condition_tokens <= 0 || condition_tokens > kMotTextMaxTokens) {
            *error = "Cosmos3 MoT layer condition trace received invalid token count";
            return false;
        }
        if (cudaMemcpyAsync(layer_trace_condition_hidden[static_cast<size_t>(slot)],
                            src,
                            static_cast<size_t>(condition_tokens) * kLanguageHidden * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT layer condition hidden trace copy failed at ") +
                     label + ": " + cudaGetErrorString(st);
            return false;
        }
        return true;
    }

    bool w8(const W8LinearGpu & linear,
            const float * in,
            float * out,
            int tokens,
            cudaStream_t stream,
            std::string * error) {
        if (cosmos3_w8a16_linear_f32_ws(in,
                                        linear.qweight,
                                        linear.scales,
                                        out,
                                        w8_x_bf16_workspace,
                                        w8_y_bf16_workspace,
                                        w8_c_tmp_workspace,
                                        w8_int_workspace,
                                        tokens,
                                        linear.in_features,
                                        linear.out_features,
                                        linear.qweight_rows,
                                        linear.qweight_cols,
                                        linear.scale_rows,
                                        linear.scale_cols,
                                        stream) != 0) {
            *error = "Cosmos3 MoT gen W8 linear launch failed";
            return false;
        }
        return true;
    }

    bool w8_prepare_input_bf16(const float * in,
                               int tokens,
                               int in_features,
                               cudaStream_t stream,
                               std::string * error) {
        if (cosmos3_w8a16_f32_to_bf16_ws(in,
                                         w8_x_bf16_workspace,
                                         tokens * in_features,
                                         stream) != 0) {
            *error = "Cosmos3 MoT gen W8 input bf16 convert failed";
            return false;
        }
        return true;
    }

    bool w8_bf16(const W8LinearGpu & linear,
                 const unsigned short * in_bf16,
                 float * out,
                 int tokens,
                 cudaStream_t stream,
                 std::string * error) {
        if (cosmos3_w8a16_linear_bf16_ws(in_bf16,
                                         linear.qweight,
                                         linear.scales,
                                         out,
                                         w8_y_bf16_workspace,
                                         w8_c_tmp_workspace,
                                         w8_int_workspace,
                                         tokens,
                                         linear.in_features,
                                         linear.out_features,
                                         linear.qweight_rows,
                                         linear.qweight_cols,
                                         linear.scale_rows,
                                         linear.scale_cols,
                                         stream) != 0) {
            char buf[256];
            cudaError_t st = cudaGetLastError();
            std::snprintf(buf,
                          sizeof(buf),
                          "Cosmos3 MoT gen W8 linear bf16 launch failed "
                          "tokens=%d in=%d out=%d qweight=%dx%d scale=%dx%d cuda=%s",
                          tokens,
                          linear.in_features,
                          linear.out_features,
                          linear.qweight_rows,
                          linear.qweight_cols,
                          linear.scale_rows,
                          linear.scale_cols,
                          cudaGetErrorString(st));
            *error = buf;
            return false;
        }
        return true;
    }

    bool w8_bf16_compatible(const W8LinearGpu & linear) const {
        const bool scale_ok =
            (linear.scale_rows == 1 && linear.scale_cols == linear.out_features) ||
            (linear.scale_rows == linear.out_features && linear.scale_cols == 1);
        return scale_ok &&
               (linear.in_features % 128) == 0 &&
               (linear.out_features % 64) == 0;
    }

    bool w8_shared_input_pair(const W8LinearGpu & a,
                              const W8LinearGpu & b,
                              const float * in,
                              float * out_a,
                              float * out_b,
                              int tokens,
                              cudaStream_t stream,
                              std::string * error) {
        if (!w8_bf16_compatible(a) || !w8_bf16_compatible(b)) {
            return w8(a, in, out_a, tokens, stream, error) &&
                   w8(b, in, out_b, tokens, stream, error);
        }
        if (!w8_prepare_input_bf16(in, tokens, a.in_features, stream, error)) return false;
        const unsigned short * in_bf16 = w8_x_bf16_workspace;
        return w8_bf16(a, in_bf16, out_a, tokens, stream, error) &&
               w8_bf16(b, in_bf16, out_b, tokens, stream, error);
    }

    bool w8_shared_input_triple(const W8LinearGpu & a,
                                const W8LinearGpu & b,
                                const W8LinearGpu & c,
                                const float * in,
                                float * out_a,
                                float * out_b,
                                float * out_c,
                                int tokens,
                                cudaStream_t stream,
                                std::string * error) {
        if (!w8_bf16_compatible(a) ||
            !w8_bf16_compatible(b) ||
            !w8_bf16_compatible(c)) {
            return w8(a, in, out_a, tokens, stream, error) &&
                   w8(b, in, out_b, tokens, stream, error) &&
                   w8(c, in, out_c, tokens, stream, error);
        }
        if (!w8_prepare_input_bf16(in, tokens, a.in_features, stream, error)) return false;
        const unsigned short * in_bf16 = w8_x_bf16_workspace;
        return w8_bf16(a, in_bf16, out_a, tokens, stream, error) &&
               w8_bf16(b, in_bf16, out_b, tokens, stream, error) &&
               w8_bf16(c, in_bf16, out_c, tokens, stream, error);
    }

    float * cache_k_layer(int slot, int layer) const {
        return cache_k_und_rot[static_cast<size_t>(slot)] +
               static_cast<size_t>(layer) * kMotTextMaxTokens * kLanguageKv;
    }

    float * cache_v_layer(int slot, int layer) const {
        return cache_v_und[static_cast<size_t>(slot)] +
               static_cast<size_t>(layer) * kMotTextMaxTokens * kLanguageKv;
    }

    bool prepare_condition_cache(int slot,
                                 const float * condition_hidden,
                                 const float * condition_mrope_positions,
                                 int condition_tokens,
                                 uint64_t condition_key,
                                 cudaStream_t stream,
                                 std::string * error) {
        if (slot < 0 || slot >= 2 || !condition_hidden || !condition_mrope_positions) {
            *error = "Cosmos3 MoT condition cache received invalid input";
            return false;
        }
        if (cache_valid[static_cast<size_t>(slot)] &&
            cache_condition_tokens[static_cast<size_t>(slot)] == condition_tokens &&
            cache_condition_keys[static_cast<size_t>(slot)] == condition_key) {
            ++condition_cache_hits;
            return true;
        }
        if (condition_tokens <= 0 || condition_tokens > kMotTextMaxTokens) {
            *error = "Cosmos3 MoT condition cache received invalid token count";
            return false;
        }
        ++condition_cache_builds;
        if (cudaMemcpyAsync(und_x,
                            condition_hidden,
                            static_cast<size_t>(condition_tokens) * kLanguageHidden * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            *error = "Cosmos3 MoT condition cache input copy failed";
            return false;
        }
        auto fail = [&](const char * what) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT condition cache failed at ") + what + ": " +
                     cudaGetErrorString(st);
            return false;
        };
        for (int layer = 0; layer < kLanguageLayers; ++layer) {
            const MotGenLayerGpu & l = weights.layers[static_cast<size_t>(layer)];
            if (cosmos3_qwen_rmsnorm_f32(und_x, l.und_input_norm, und_norm,
                                         condition_tokens, kLanguageHidden,
                                         kLanguageRmsEps, stream) != 0) return fail("und_input_norm");
            if (!w8_shared_input_triple(l.und_q_proj, l.und_k_proj, l.und_v_proj,
                                        und_norm, q_und, k_und, v_und,
                                        condition_tokens, stream, error)) return false;
            if (cosmos3_qwen_head_rmsnorm_f32(q_und, l.und_q_norm, q_und_norm,
                                              condition_tokens, kLanguageHidden,
                                              kLanguageQHeads, kLanguageHeadDim,
                                              kLanguageRmsEps, stream) != 0) return fail("und_q_norm");
            if (cosmos3_qwen_head_rmsnorm_f32(k_und, l.und_k_norm, k_und_norm,
                                              condition_tokens, kLanguageKv,
                                              kLanguageKvHeads, kLanguageHeadDim,
                                              kLanguageRmsEps, stream) != 0) return fail("und_k_norm");
            if (cosmos3_qwen_mrope_pos_f32(q_und_norm, condition_mrope_positions, q_und_rot,
                                       condition_tokens, kLanguageHidden,
                                       kLanguageQHeads, kLanguageHeadDim,
                                       kLanguageRopeTheta, stream) != 0) return fail("und_q_mrope");
            if (cosmos3_qwen_mrope_pos_f32(k_und_norm, condition_mrope_positions, k_und_rot,
                                       condition_tokens, kLanguageKv,
                                       kLanguageKvHeads, kLanguageHeadDim,
                                       kLanguageRopeTheta, stream) != 0) return fail("und_k_mrope");
            if (cudaMemcpyAsync(cache_k_layer(slot, layer),
                                k_und_rot,
                                static_cast<size_t>(condition_tokens) * kLanguageKv * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess ||
                cudaMemcpyAsync(cache_v_layer(slot, layer),
                                v_und,
                                static_cast<size_t>(condition_tokens) * kLanguageKv * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess) {
                *error = "Cosmos3 MoT condition cache K/V copy failed";
                return false;
            }
            if (cosmos3_qwen_causal_gqa_attention_f32(q_und_rot, k_und_rot, v_und, attn_und,
                                                      condition_tokens, kLanguageHidden,
                                                      kLanguageKv, kLanguageQHeads,
                                                      kLanguageKvHeads, kLanguageHeadDim,
                                                      stream) != 0) return fail("und_causal_attention");
            if (!w8(l.und_o_proj, attn_und, o_und, condition_tokens, stream, error)) return false;
            if (cosmos3_qwen_residual_add_rmsnorm_f32(und_x, o_und, l.und_post_norm,
                                                      residual_und, post_norm_und,
                                                      condition_tokens, kLanguageHidden,
                                                      kLanguageRmsEps, stream) != 0) return fail("und_post_norm");
            if (!w8_shared_input_pair(l.und_gate_proj, l.und_up_proj,
                                      post_norm_und, gate_und, up_und,
                                      condition_tokens, stream, error)) return false;
            if (cosmos3_qwen_swiglu_f32(gate_und, up_und, swiglu_und,
                                        condition_tokens * kLanguageIntermediate,
                                        stream) != 0) return fail("und_swiglu");
            if (!w8(l.und_down_proj, swiglu_und, down_und, condition_tokens, stream, error)) return false;
            if (cosmos3_qwen_residual_add_f32(residual_und, down_und, und_x,
                                              condition_tokens * kLanguageHidden,
                                              stream) != 0) return fail("und_mlp_residual");
            if (!copy_layer_condition_hidden_trace(layer,
                                                   und_x,
                                                   condition_tokens,
                                                   stream,
                                                   error,
                                                   "condition_cache")) return false;
        }
        cache_condition_tokens[static_cast<size_t>(slot)] = condition_tokens;
        cache_condition_keys[static_cast<size_t>(slot)] = condition_key;
        cache_valid[static_cast<size_t>(slot)] = true;
        return true;
    }

    bool forward(const float * packed_hidden,
                 const float * condition_mrope_positions,
                 const float * full_mrope_positions,
                 int condition_tokens,
                 int full_tokens,
                 cudaStream_t stream,
                 std::string * error) {
        if (!packed_hidden || !condition_mrope_positions || !full_mrope_positions) {
            *error = "Cosmos3 MoT gen forward received null input";
            return false;
        }
        if (condition_tokens <= 0 || condition_tokens > kMotTextMaxTokens) {
            *error = "Cosmos3 MoT gen forward received invalid condition token count";
            return false;
        }
        if (full_tokens != kMotFullTokens) {
            *error = "Cosmos3 MoT gen forward received invalid full token count";
            return false;
        }
        const float * condition_x = packed_hidden;
        const float * action_x = packed_hidden + static_cast<size_t>(condition_tokens) * kLanguageHidden;
        if (cudaMemcpyAsync(und_x,
                            condition_x,
                            static_cast<size_t>(condition_tokens) * kLanguageHidden * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            *error = "Cosmos3 MoT gen condition input copy failed";
            return false;
        }
        if (cudaMemcpyAsync(gen_x,
                            action_x,
                            static_cast<size_t>(full_tokens) * kLanguageHidden * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            *error = "Cosmos3 MoT gen action input copy failed";
            return false;
        }
        auto fail = [&](const char * what) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT gen CUDA forward failed at ") + what + ": " +
                     cudaGetErrorString(st);
            return false;
        };
        for (int layer = 0; layer < kLanguageLayers; ++layer) {
            const MotGenLayerGpu & l = weights.layers[static_cast<size_t>(layer)];
            if (cosmos3_qwen_rmsnorm_f32(und_x, l.und_input_norm, und_norm,
                                         condition_tokens, kLanguageHidden,
                                         kLanguageRmsEps, stream) != 0) return fail("und_input_norm");
            if (cosmos3_qwen_rmsnorm_f32(gen_x, l.gen_input_norm, gen_norm,
                                         full_tokens, kLanguageHidden,
                                         kLanguageRmsEps, stream) != 0) return fail("gen_input_norm");
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_condition_slice(layer0_trace_und_norm, und_norm, condition_tokens,
                                             kLanguageHidden, stream, error, "und_norm")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_gen_norm, gen_norm, kLanguageHidden,
                                          stream, error, "gen_norm")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_vision_sample_rows(layer0_trace_vision_gen_norm,
                                                gen_norm,
                                                kLanguageHidden,
                                                stream,
                                                error,
                                                "vision_gen_norm")) return false;
            if (!w8(l.und_q_proj, und_norm, q_und, condition_tokens, stream, error) ||
                !w8(l.und_k_proj, und_norm, k_und, condition_tokens, stream, error) ||
                !w8(l.und_v_proj, und_norm, v_und, condition_tokens, stream, error) ||
                !w8(l.gen_q_proj, gen_norm, q_gen, full_tokens, stream, error) ||
                !w8(l.gen_k_proj, gen_norm, k_gen, full_tokens, stream, error) ||
                !w8(l.gen_v_proj, gen_norm, v_gen, full_tokens, stream, error)) return false;
            if (should_trace_internal_layer(layer) &&
                (!copy_layer0_condition_slice(layer0_trace_q_und, q_und, condition_tokens,
                                              kLanguageHidden, stream, error, "q_und") ||
                 !copy_layer0_condition_slice(layer0_trace_k_und, k_und, condition_tokens,
                                              kLanguageKv, stream, error, "k_und") ||
                 !copy_layer0_condition_slice(layer0_trace_v_und, v_und, condition_tokens,
                                              kLanguageKv, stream, error, "v_und"))) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_q_gen, q_gen, kLanguageHidden,
                                          stream, error, "q_gen")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_k_gen, k_gen, kLanguageKv,
                                          stream, error, "k_gen")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_v_gen, v_gen, kLanguageKv,
                                          stream, error, "v_gen")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_vision_sample_rows(layer0_trace_vision_k_gen,
                                                k_gen,
                                                kLanguageKv,
                                                stream,
                                                error,
                                                "vision_k_gen")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_vision_sample_rows(layer0_trace_vision_v_gen,
                                                v_gen,
                                                kLanguageKv,
                                                stream,
                                                error,
                                                "vision_v_gen")) return false;
            if (should_trace_internal_layer(layer) &&
                cudaMemcpyAsync(layer0_trace_v_gen_full,
                                v_gen,
                                static_cast<size_t>(full_tokens) * kLanguageKv * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess) {
                return fail("v_gen_full_copy");
            }
            if (cosmos3_qwen_head_rmsnorm_f32(q_und, l.und_q_norm, q_und_norm,
                                              condition_tokens, kLanguageHidden,
                                              kLanguageQHeads, kLanguageHeadDim,
                                              kLanguageRmsEps, stream) != 0) return fail("und_q_norm");
            if (cosmos3_qwen_head_rmsnorm_f32(k_und, l.und_k_norm, k_und_norm,
                                              condition_tokens, kLanguageKv,
                                              kLanguageKvHeads, kLanguageHeadDim,
                                              kLanguageRmsEps, stream) != 0) return fail("und_k_norm");
            if (should_trace_internal_layer(layer) &&
                (!copy_layer0_condition_slice(layer0_trace_q_und_norm, q_und_norm, condition_tokens,
                                              kLanguageHidden, stream, error, "q_und_norm") ||
                 !copy_layer0_condition_slice(layer0_trace_k_und_norm, k_und_norm, condition_tokens,
                                              kLanguageKv, stream, error, "k_und_norm"))) return false;
            if (cosmos3_qwen_head_rmsnorm_f32(q_gen, l.gen_q_norm, q_gen_norm,
                                              full_tokens, kLanguageHidden,
                                              kLanguageQHeads, kLanguageHeadDim,
                                              kLanguageRmsEps, stream) != 0) return fail("gen_q_norm");
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_q_gen_norm, q_gen_norm, kLanguageHidden,
                                          stream, error, "q_gen_norm")) return false;
            if (cosmos3_qwen_head_rmsnorm_f32(k_gen, l.gen_k_norm, k_gen_norm,
                                              full_tokens, kLanguageKv,
                                              kLanguageKvHeads, kLanguageHeadDim,
                                              kLanguageRmsEps, stream) != 0) return fail("gen_k_norm");
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_k_gen_norm, k_gen_norm, kLanguageKv,
                                          stream, error, "k_gen_norm")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_vision_sample_rows(layer0_trace_vision_k_gen_norm,
                                                k_gen_norm,
                                                kLanguageKv,
                                                stream,
                                                error,
                                                "vision_k_gen_norm")) return false;
            if (cosmos3_qwen_mrope_pos_f32(q_und_norm, condition_mrope_positions, q_und_rot,
                                       condition_tokens, kLanguageHidden,
                                       kLanguageQHeads, kLanguageHeadDim,
                                       kLanguageRopeTheta, stream) != 0) return fail("und_q_mrope");
            if (cosmos3_qwen_mrope_pos_f32(k_und_norm, condition_mrope_positions, k_und_rot,
                                       condition_tokens, kLanguageKv,
                                       kLanguageKvHeads, kLanguageHeadDim,
                                       kLanguageRopeTheta, stream) != 0) return fail("und_k_mrope");
            if (should_trace_internal_layer(layer) &&
                (!copy_layer0_condition_slice(layer0_trace_q_und_rot, q_und_rot, condition_tokens,
                                              kLanguageHidden, stream, error, "q_und_rot") ||
                 !copy_layer0_condition_slice(layer0_trace_k_und_rot, k_und_rot, condition_tokens,
                                              kLanguageKv, stream, error, "k_und_rot"))) return false;
            if (cosmos3_qwen_mrope_pos_f32(q_gen_norm, full_mrope_positions, q_gen_rot,
                                       full_tokens, kLanguageHidden,
                                       kLanguageQHeads, kLanguageHeadDim,
                                       kLanguageRopeTheta, stream) != 0) return fail("gen_q_mrope");
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_q_gen_rot, q_gen_rot, kLanguageHidden,
                                          stream, error, "q_gen_rot")) return false;
            if (cosmos3_qwen_mrope_pos_f32(k_gen_norm, full_mrope_positions, k_gen_rot,
                                       full_tokens, kLanguageKv,
                                       kLanguageKvHeads, kLanguageHeadDim,
                                       kLanguageRopeTheta, stream) != 0) return fail("gen_k_mrope");
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_k_gen_rot, k_gen_rot, kLanguageKv,
                                          stream, error, "k_gen_rot")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_vision_sample_rows(layer0_trace_vision_k_gen_rot,
                                                k_gen_rot,
                                                kLanguageKv,
                                                stream,
                                                error,
                                                "vision_k_gen_rot")) return false;
            if (should_trace_internal_layer(layer) &&
                cudaMemcpyAsync(layer0_trace_k_gen_rot_full,
                                k_gen_rot,
                                static_cast<size_t>(full_tokens) * kLanguageKv * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess) {
                return fail("k_gen_rot_full_copy");
            }
            if (cosmos3_mot_pack_condition_action_f32(k_und_rot, k_gen_rot, k_all,
                                                      condition_tokens, full_tokens,
                                                      kLanguageKv, stream) != 0) return fail("pack_k");
            if (cosmos3_mot_pack_condition_action_f32(v_und, v_gen, v_all,
                                                      condition_tokens, full_tokens,
                                                      kLanguageKv, stream) != 0) return fail("pack_v");
            if (cosmos3_qwen_causal_gqa_attention_f32(q_und_rot, k_und_rot, v_und, attn_und,
                                                      condition_tokens, kLanguageHidden,
                                                      kLanguageKv, kLanguageQHeads,
                                                      kLanguageKvHeads, kLanguageHeadDim,
                                                      stream) != 0) return fail("und_causal_attention");
            if (cosmos3_qwen_gen_full_gqa_attention_workspace_f32(q_gen_rot, k_all, v_all, attn_gen,
                                                                  gen_attention_scores,
                                                                  full_tokens, condition_tokens + full_tokens,
                                                                  kLanguageHidden, kLanguageKv,
                                                                  kLanguageQHeads, kLanguageKvHeads,
                                                                  kLanguageHeadDim, stream) != 0) return fail("gen_full_attention");
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_condition_slice(layer0_trace_attn_und, attn_und, condition_tokens,
                                             kLanguageHidden, stream, error, "attn_und")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_attn_gen, attn_gen, kLanguageHidden,
                                          stream, error, "attn_gen")) return false;
            if (!w8(l.und_o_proj, attn_und, o_und, condition_tokens, stream, error)) return false;
            if (!w8(l.gen_o_proj, attn_gen, o_gen, full_tokens, stream, error)) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_condition_slice(layer0_trace_o_und, o_und, condition_tokens,
                                             kLanguageHidden, stream, error, "o_und")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_o_gen, o_gen, kLanguageHidden,
                                          stream, error, "o_gen")) return false;
            if (cosmos3_qwen_residual_add_rmsnorm_f32(und_x, o_und, l.und_post_norm,
                                                      residual_und, post_norm_und,
                                                      condition_tokens, kLanguageHidden,
                                                      kLanguageRmsEps, stream) != 0) return fail("und_post_norm");
            if (cosmos3_qwen_residual_add_rmsnorm_f32(gen_x, o_gen, l.gen_post_norm,
                                                      residual_gen, post_norm_gen,
                                                      full_tokens, kLanguageHidden,
                                                      kLanguageRmsEps, stream) != 0) return fail("gen_post_norm");
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_condition_slice(layer0_trace_residual_und, residual_und, condition_tokens,
                                             kLanguageHidden, stream, error, "residual_und")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_condition_slice(layer0_trace_post_norm_und, post_norm_und, condition_tokens,
                                             kLanguageHidden, stream, error, "post_norm_und")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_post_norm_gen, post_norm_gen, kLanguageHidden,
                                          stream, error, "post_norm_gen")) return false;
            if (!w8(l.und_gate_proj, post_norm_und, gate_und, condition_tokens, stream, error) ||
                !w8(l.und_up_proj, post_norm_und, up_und, condition_tokens, stream, error) ||
                !w8(l.gen_gate_proj, post_norm_gen, gate, full_tokens, stream, error) ||
                !w8(l.gen_up_proj, post_norm_gen, up, full_tokens, stream, error)) return false;
            if (should_trace_internal_layer(layer) &&
                (!copy_layer0_action_slice(layer0_trace_gate, gate, kLanguageIntermediate,
                                           stream, error, "gate") ||
                 !copy_layer0_action_slice(layer0_trace_up, up, kLanguageIntermediate,
                                           stream, error, "up"))) return false;
            if (should_trace_internal_layer(layer) &&
                (!copy_layer0_condition_slice(layer0_trace_gate_und, gate_und, condition_tokens,
                                             kLanguageIntermediate, stream, error, "gate_und") ||
                 !copy_layer0_condition_slice(layer0_trace_up_und, up_und, condition_tokens,
                                             kLanguageIntermediate, stream, error, "up_und"))) return false;
            if (cosmos3_qwen_swiglu_f32(gate_und, up_und, swiglu_und,
                                        condition_tokens * kLanguageIntermediate,
                                        stream) != 0) return fail("und_swiglu");
            if (cosmos3_qwen_swiglu_f32(gate, up, swiglu,
                                        full_tokens * kLanguageIntermediate,
                                        stream) != 0) return fail("gen_swiglu");
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_swiglu, swiglu, kLanguageIntermediate,
                                          stream, error, "swiglu")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_condition_slice(layer0_trace_swiglu_und, swiglu_und, condition_tokens,
                                             kLanguageIntermediate, stream, error, "swiglu_und")) return false;
            if (!w8(l.und_down_proj, swiglu_und, down_und, condition_tokens, stream, error)) return false;
            if (!w8(l.gen_down_proj, swiglu, down, full_tokens, stream, error)) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_condition_slice(layer0_trace_down_und, down_und, condition_tokens,
                                             kLanguageHidden, stream, error, "down_und")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_down, down, kLanguageHidden,
                                          stream, error, "down")) return false;
            if (cosmos3_qwen_residual_add_f32(residual_und, down_und, und_x,
                                              condition_tokens * kLanguageHidden,
                                              stream) != 0) return fail("und_mlp_residual");
            if (cosmos3_qwen_residual_add_f32(residual_gen, down, gen_x,
                                              full_tokens * kLanguageHidden,
                                              stream) != 0) return fail("gen_mlp_residual");
            if (!copy_layer_condition_hidden_trace(layer,
                                                   und_x,
                                                   condition_tokens,
                                                   stream,
                                                   error,
                                                   "full_forward")) return false;
            if (!copy_layer_action_hidden_trace(layer,
                                                gen_x,
                                                stream,
                                                error,
                                                "full_forward")) return false;
            if (should_trace_internal_layer(layer) &&
                cudaMemcpyAsync(layer0_gen_x,
                                gen_x,
                                static_cast<size_t>(full_tokens) * kLanguageHidden * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess) {
                return fail("layer0_gen_copy");
            }
        }
        if (cosmos3_qwen_rmsnorm_f32(gen_x, weights.final_norm, final_norm_gen,
                                     full_tokens, kLanguageHidden,
                                     kLanguageRmsEps, stream) != 0) return fail("gen_final_norm");
        return true;
    }

    bool forward_with_condition_cache(int slot,
                                      const float * packed_hidden,
                                      int condition_tokens,
                                      cudaStream_t stream,
                                      std::string * error) {
        if (slot < 0 || slot >= 2 || !packed_hidden ||
            !cache_valid[static_cast<size_t>(slot)] ||
            cache_condition_tokens[static_cast<size_t>(slot)] != condition_tokens) {
            *error = "Cosmos3 MoT gen cached forward called without a valid condition cache";
            return false;
        }
        const float * action_x = packed_hidden + static_cast<size_t>(condition_tokens) * kLanguageHidden;
        if (cudaMemcpyAsync(gen_x,
                            action_x,
                            static_cast<size_t>(kMotFullTokens) * kLanguageHidden * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
            *error = "Cosmos3 MoT cached gen action input copy failed";
            return false;
        }
        auto fail = [&](const char * what) {
            cudaError_t st = cudaGetLastError();
            *error = std::string("Cosmos3 MoT cached gen CUDA forward failed at ") + what + ": " +
                     cudaGetErrorString(st);
            return false;
        };
        auto time_block = [&](TimingSlot timing_slot, auto && fn) -> bool {
            if (profile_enabled && !timing_enabled) {
                ProfileEvent ev;
                ev.slot = timing_slot;
                if (cudaEventCreateWithFlags(&ev.start, cudaEventDefault) != cudaSuccess ||
                    cudaEventCreateWithFlags(&ev.end, cudaEventDefault) != cudaSuccess ||
                    cudaEventRecord(ev.start, stream) != cudaSuccess) {
                    if (ev.start) cudaEventDestroy(ev.start);
                    if (ev.end) cudaEventDestroy(ev.end);
                    *error = "Cosmos3 MoT CUDA event profile setup failed";
                    return false;
                }
                const bool ok = fn();
                if (cudaEventRecord(ev.end, stream) != cudaSuccess) {
                    if (ev.start) cudaEventDestroy(ev.start);
                    if (ev.end) cudaEventDestroy(ev.end);
                    *error = "Cosmos3 MoT CUDA event profile record failed";
                    return false;
                }
                profile_events.push_back(ev);
                return ok;
            }
            if (!timing_enabled) return fn();
            const auto tb = std::chrono::steady_clock::now();
            const bool ok = fn();
            cudaStreamSynchronize(stream);
            timing_ms[timing_slot] +=
                std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tb).count();
            return ok;
        };
        for (int layer = 0; layer < kLanguageLayers; ++layer) {
            const MotGenLayerGpu & l = weights.layers[static_cast<size_t>(layer)];
            if (!time_block(MT_GEN_NORM, [&]() {
                    return cosmos3_qwen_rmsnorm_f32(gen_x, l.gen_input_norm, gen_norm,
                                                    kMotFullTokens, kLanguageHidden,
                                                    kLanguageRmsEps, stream) == 0;
                })) return fail("gen_input_norm");
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_gen_norm, gen_norm, kLanguageHidden,
                                          stream, error, "cached_gen_norm")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_vision_sample_rows(layer0_trace_vision_gen_norm,
                                                gen_norm,
                                                kLanguageHidden,
                                                stream,
                                                error,
                                                "cached_vision_gen_norm")) return false;
            if (!time_block(MT_GEN_QKV_W8, [&]() {
                    return w8_shared_input_triple(l.gen_q_proj, l.gen_k_proj, l.gen_v_proj,
                                                  gen_norm, q_gen, k_gen, v_gen,
                                                  kMotFullTokens, stream, error);
                })) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_q_gen, q_gen, kLanguageHidden,
                                          stream, error, "cached_q_gen")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_k_gen, k_gen, kLanguageKv,
                                          stream, error, "cached_k_gen")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_v_gen, v_gen, kLanguageKv,
                                          stream, error, "cached_v_gen")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_vision_sample_rows(layer0_trace_vision_k_gen,
                                                k_gen,
                                                kLanguageKv,
                                                stream,
                                                error,
                                                "cached_vision_k_gen")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_vision_sample_rows(layer0_trace_vision_v_gen,
                                                v_gen,
                                                kLanguageKv,
                                                stream,
                                                error,
                                                "cached_vision_v_gen")) return false;
            if (should_trace_internal_layer(layer) &&
                cudaMemcpyAsync(layer0_trace_v_gen_full,
                                v_gen,
                                static_cast<size_t>(kMotFullTokens) * kLanguageKv * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess) {
                return fail("cached_v_gen_full_copy");
            }
            const bool trace_this_layer = should_trace_internal_layer(layer);
            if (trace_this_layer) {
                if (!time_block(MT_GEN_QK_NORM, [&]() {
                        return cosmos3_qwen_head_rmsnorm_f32(q_gen, l.gen_q_norm, q_gen_norm,
                                                             kMotFullTokens, kLanguageHidden,
                                                             kLanguageQHeads, kLanguageHeadDim,
                                                             kLanguageRmsEps, stream) == 0;
                    })) return fail("gen_q_norm");
                if (!copy_layer0_action_slice(layer0_trace_q_gen_norm, q_gen_norm, kLanguageHidden,
                                              stream, error, "cached_q_gen_norm")) return false;
                if (!time_block(MT_GEN_QK_NORM, [&]() {
                        return cosmos3_qwen_head_rmsnorm_f32(k_gen, l.gen_k_norm, k_gen_norm,
                                                             kMotFullTokens, kLanguageKv,
                                                             kLanguageKvHeads, kLanguageHeadDim,
                                                             kLanguageRmsEps, stream) == 0;
                    })) return fail("gen_k_norm");
                if (!copy_layer0_action_slice(layer0_trace_k_gen_norm, k_gen_norm, kLanguageKv,
                                              stream, error, "cached_k_gen_norm")) return false;
                if (!copy_layer0_vision_sample_rows(layer0_trace_vision_k_gen_norm,
                                                    k_gen_norm,
                                                    kLanguageKv,
                                                    stream,
                                                    error,
                                                    "cached_vision_k_gen_norm")) return false;
                if (!time_block(MT_GEN_MROPE, [&]() {
                        return cosmos3_qwen_mrope_pos_f32(q_gen_norm, full_mrope_positions_dev[static_cast<size_t>(slot)], q_gen_rot,
                                                      kMotFullTokens, kLanguageHidden,
                                                      kLanguageQHeads, kLanguageHeadDim,
                                                      kLanguageRopeTheta, stream) == 0;
                    })) return fail("gen_q_mrope");
                if (!copy_layer0_action_slice(layer0_trace_q_gen_rot, q_gen_rot, kLanguageHidden,
                                              stream, error, "cached_q_gen_rot")) return false;
                if (!time_block(MT_GEN_MROPE, [&]() {
                        return cosmos3_qwen_mrope_pos_f32(k_gen_norm, full_mrope_positions_dev[static_cast<size_t>(slot)], k_gen_rot,
                                                      kMotFullTokens, kLanguageKv,
                                                      kLanguageKvHeads, kLanguageHeadDim,
                                                      kLanguageRopeTheta, stream) == 0;
                    })) return fail("gen_k_mrope");
            } else {
                if (!time_block(MT_GEN_QK_NORM, [&]() {
                        return cosmos3_qwen_head_rmsnorm_mrope_pos_f32(q_gen, l.gen_q_norm,
                                                                       full_mrope_positions_dev[static_cast<size_t>(slot)],
                                                                       q_gen_rot,
                                                                       kMotFullTokens, kLanguageHidden,
                                                                       kLanguageQHeads, kLanguageHeadDim,
                                                                       kLanguageRmsEps, kLanguageRopeTheta,
                                                                       stream) == 0;
                    })) return fail("gen_q_norm_mrope");
                if (!time_block(MT_GEN_QK_NORM, [&]() {
                        return cosmos3_qwen_head_rmsnorm_mrope_pos_f32(k_gen, l.gen_k_norm,
                                                                       full_mrope_positions_dev[static_cast<size_t>(slot)],
                                                                       k_gen_rot,
                                                                       kMotFullTokens, kLanguageKv,
                                                                       kLanguageKvHeads, kLanguageHeadDim,
                                                                       kLanguageRmsEps, kLanguageRopeTheta,
                                                                       stream) == 0;
                    })) return fail("gen_k_norm_mrope");
            }
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_k_gen_rot, k_gen_rot, kLanguageKv,
                                          stream, error, "cached_k_gen_rot")) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_vision_sample_rows(layer0_trace_vision_k_gen_rot,
                                                k_gen_rot,
                                                kLanguageKv,
                                                stream,
                                                error,
                                                "cached_vision_k_gen_rot")) return false;
            if (should_trace_internal_layer(layer) &&
                cudaMemcpyAsync(layer0_trace_k_gen_rot_full,
                                k_gen_rot,
                                static_cast<size_t>(kMotFullTokens) * kLanguageKv * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess) {
                return fail("cached_k_gen_rot_full_copy");
            }
            if (!time_block(MT_GEN_PACK_KV, [&]() {
                    return cosmos3_mot_pack_condition_action_f32(cache_k_layer(slot, layer), k_gen_rot, k_all,
                                                                 condition_tokens, kMotFullTokens,
                                                                 kLanguageKv, stream) == 0 &&
                           cosmos3_mot_pack_condition_action_f32(cache_v_layer(slot, layer), v_gen, v_all,
                                                                 condition_tokens, kMotFullTokens,
                                                                 kLanguageKv, stream) == 0;
                })) return fail("pack_kv");
            if (!time_block(MT_GEN_ATTENTION, [&]() {
                    return cosmos3_qwen_gen_full_gqa_attention_workspace_bf16tc_f32(
                               q_gen_rot, k_all, v_all, attn_gen,
                               gen_attention_scores,
                               gen_attention_bf16_workspace,
                               kMotFullTokens, condition_tokens + kMotFullTokens,
                               kLanguageHidden, kLanguageKv,
                               kLanguageQHeads, kLanguageKvHeads,
                               kLanguageHeadDim, stream) == 0;
                })) return fail("gen_full_attention");
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_attn_gen, attn_gen, kLanguageHidden,
                                          stream, error, "cached_attn_gen")) return false;
            if (!time_block(MT_GEN_O_W8, [&]() {
                    return w8(l.gen_o_proj, attn_gen, o_gen, kMotFullTokens, stream, error);
                })) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_o_gen, o_gen, kLanguageHidden,
                                          stream, error, "cached_o_gen")) return false;
            if (!time_block(MT_GEN_POST_NORM, [&]() {
                    return cosmos3_qwen_residual_add_rmsnorm_f32(gen_x, o_gen, l.gen_post_norm,
                                                                 residual_gen, post_norm_gen,
                                                                 kMotFullTokens, kLanguageHidden,
                                                                 kLanguageRmsEps, stream) == 0;
                })) return fail("gen_post_norm");
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_post_norm_gen, post_norm_gen, kLanguageHidden,
                                          stream, error, "cached_post_norm_gen")) return false;
            if (!time_block(MT_GEN_GATE_UP_W8, [&]() {
                    return w8_shared_input_pair(l.gen_gate_proj, l.gen_up_proj,
                                                post_norm_gen, gate, up,
                                                kMotFullTokens, stream, error);
                })) return false;
            if (should_trace_internal_layer(layer) &&
                (!copy_layer0_action_slice(layer0_trace_gate, gate, kLanguageIntermediate,
                                           stream, error, "gate") ||
                 !copy_layer0_action_slice(layer0_trace_up, up, kLanguageIntermediate,
                                           stream, error, "up"))) return false;
            if (!time_block(MT_GEN_SWIGLU, [&]() {
                    return cosmos3_qwen_swiglu_f32(gate, up, swiglu,
                                                   kMotFullTokens * kLanguageIntermediate,
                                                   stream) == 0;
                })) return fail("gen_swiglu");
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_swiglu, swiglu, kLanguageIntermediate,
                                          stream, error, "swiglu")) return false;
            if (!time_block(MT_GEN_DOWN_W8, [&]() {
                    return w8(l.gen_down_proj, swiglu, down, kMotFullTokens, stream, error);
                })) return false;
            if (should_trace_internal_layer(layer) &&
                !copy_layer0_action_slice(layer0_trace_down, down, kLanguageHidden,
                                          stream, error, "cached_down")) return false;
            if (!time_block(MT_GEN_RESIDUAL, [&]() {
                    return cosmos3_qwen_residual_add_f32(residual_gen, down, gen_x,
                                                         kMotFullTokens * kLanguageHidden,
                                                         stream) == 0;
                })) return fail("gen_mlp_residual");
            if (!copy_layer_action_hidden_trace(layer,
                                                gen_x,
                                                stream,
                                                error,
                                                "cached_forward")) return false;
            if (should_trace_internal_layer(layer) &&
                cudaMemcpyAsync(layer0_gen_x,
                                gen_x,
                                static_cast<size_t>(kMotFullTokens) * kLanguageHidden * sizeof(float),
                                cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess) {
                return fail("layer0_gen_copy");
            }
        }
        if (!time_block(MT_GEN_FINAL_NORM, [&]() {
                return cosmos3_qwen_rmsnorm_f32(gen_x, weights.final_norm, final_norm_gen,
                                                kMotFullTokens, kLanguageHidden,
                                                kLanguageRmsEps, stream) == 0;
            })) return fail("gen_final_norm");
        return true;
    }

    void append_timing_tensors(std::vector<WamTensor> * tensors) {
        if (profile_enabled) {
            std::array<float, MT_COUNT> profile_ms{};
            for (ProfileEvent & ev : profile_events) {
                if (!ev.start || !ev.end) continue;
                if (cudaEventSynchronize(ev.end) != cudaSuccess) continue;
                float ms = 0.0f;
                if (cudaEventElapsedTime(&ms, ev.start, ev.end) == cudaSuccess &&
                    ev.slot >= 0 && ev.slot < MT_COUNT) {
                    profile_ms[ev.slot] += ms;
                }
            }
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.gen_norm_ms", profile_ms[MT_GEN_NORM]);
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.qkv_w8_ms", profile_ms[MT_GEN_QKV_W8]);
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.qk_norm_ms", profile_ms[MT_GEN_QK_NORM]);
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.mrope_ms", profile_ms[MT_GEN_MROPE]);
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.pack_kv_ms", profile_ms[MT_GEN_PACK_KV]);
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.attention_ms", profile_ms[MT_GEN_ATTENTION]);
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.o_w8_ms", profile_ms[MT_GEN_O_W8]);
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.post_norm_ms", profile_ms[MT_GEN_POST_NORM]);
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.gate_up_w8_ms", profile_ms[MT_GEN_GATE_UP_W8]);
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.swiglu_ms", profile_ms[MT_GEN_SWIGLU]);
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.down_w8_ms", profile_ms[MT_GEN_DOWN_W8]);
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.residual_ms", profile_ms[MT_GEN_RESIDUAL]);
            append_scalar_timing_tensor(tensors, "cosmos3.profile.mot_gen.final_norm_ms", profile_ms[MT_GEN_FINAL_NORM]);
            clear_profile_events();
        }
        if (!timing_enabled) return;
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.gen_norm_ms", timing_ms[MT_GEN_NORM]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.qkv_w8_ms", timing_ms[MT_GEN_QKV_W8]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.qk_norm_ms", timing_ms[MT_GEN_QK_NORM]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.mrope_ms", timing_ms[MT_GEN_MROPE]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.pack_kv_ms", timing_ms[MT_GEN_PACK_KV]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.attention_ms", timing_ms[MT_GEN_ATTENTION]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.o_w8_ms", timing_ms[MT_GEN_O_W8]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.post_norm_ms", timing_ms[MT_GEN_POST_NORM]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.gate_up_w8_ms", timing_ms[MT_GEN_GATE_UP_W8]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.swiglu_ms", timing_ms[MT_GEN_SWIGLU]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.down_w8_ms", timing_ms[MT_GEN_DOWN_W8]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.residual_ms", timing_ms[MT_GEN_RESIDUAL]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.final_norm_ms", timing_ms[MT_GEN_FINAL_NORM]);
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.condition_cache_hits", static_cast<float>(condition_cache_hits));
        append_scalar_timing_tensor(tensors, "cosmos3.debug.timing.mot_gen.condition_cache_builds", static_cast<float>(condition_cache_builds));
    }

    bool append_debug_tensors(std::vector<WamTensor> * tensors,
                              std::string * error) const {
        return append_cuda_prefix_tensor(tensors,
                                         "cosmos3.debug.mot.gen_action_hidden.prefix",
                                         gen_action_hidden(),
                                         kActionSteps,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error);
    }

    bool append_action_hidden_tensor(std::vector<WamTensor> * tensors,
                                     const char * name,
                                     std::string * error) const {
        return append_cuda_prefix_tensor(tensors,
                                         name,
                                         gen_action_hidden(),
                                         kActionSteps,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error);
    }

    bool append_layer0_action_hidden_tensor(std::vector<WamTensor> * tensors,
                                            const char * name,
                                            std::string * error) const {
        return append_cuda_prefix_tensor(tensors,
                                         name,
                                         layer0_gen_x + static_cast<size_t>(kMotVisionTokens + kActionConditionTokens) * kLanguageHidden,
                                         kActionSteps,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error);
    }

    bool append_layer0_trace_tensors(std::vector<WamTensor> * tensors,
                                     const char * suffix,
                                     int condition_tokens,
                                     std::string * error) const {
        const std::string base = std::string("cosmos3.debug.mot.layer") +
                                 std::to_string(internal_trace_layer) + ".";
        return append_cuda_prefix_tensor(tensors,
                                         (base + "gen_norm." + suffix).c_str(),
                                         layer0_trace_gen_norm,
                                         kActionSteps,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "q_gen." + suffix).c_str(),
                                         layer0_trace_q_gen,
                                         kActionSteps,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "k_gen." + suffix).c_str(),
                                         layer0_trace_k_gen,
                                         kActionSteps,
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "v_gen." + suffix).c_str(),
                                         layer0_trace_v_gen,
                                         kActionSteps,
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "q_gen_norm." + suffix).c_str(),
                                         layer0_trace_q_gen_norm,
                                         kActionSteps,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "k_gen_norm." + suffix).c_str(),
                                         layer0_trace_k_gen_norm,
                                         kActionSteps,
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "q_gen_rot." + suffix).c_str(),
                                         layer0_trace_q_gen_rot,
                                         kActionSteps,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "k_gen_rot." + suffix).c_str(),
                                         layer0_trace_k_gen_rot,
                                         kActionSteps,
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "v_gen_full." + suffix).c_str(),
                                         layer0_trace_v_gen_full,
                                         kMotFullTokens,
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "k_gen_rot_full." + suffix).c_str(),
                                         layer0_trace_k_gen_rot_full,
                                         kMotFullTokens,
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "attn_gen." + suffix).c_str(),
                                         layer0_trace_attn_gen,
                                         kActionSteps,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "o_gen." + suffix).c_str(),
                                         layer0_trace_o_gen,
                                         kActionSteps,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "post_norm_gen." + suffix).c_str(),
                                         layer0_trace_post_norm_gen,
                                         kActionSteps,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "gate." + suffix).c_str(),
                                         layer0_trace_gate,
                                         kActionSteps,
                                         kLanguageIntermediate,
                                         kLanguageIntermediate,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "up." + suffix).c_str(),
                                         layer0_trace_up,
                                         kActionSteps,
                                         kLanguageIntermediate,
                                         kLanguageIntermediate,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "swiglu." + suffix).c_str(),
                                         layer0_trace_swiglu,
                                         kActionSteps,
                                         kLanguageIntermediate,
                                         kLanguageIntermediate,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "down." + suffix).c_str(),
                                         layer0_trace_down,
                                         kActionSteps,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "und_norm." + suffix).c_str(),
                                         layer0_trace_und_norm,
                                         condition_tokens,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "q_und." + suffix).c_str(),
                                         layer0_trace_q_und,
                                         condition_tokens,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "k_und." + suffix).c_str(),
                                         layer0_trace_k_und,
                                         condition_tokens,
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "v_und." + suffix).c_str(),
                                         layer0_trace_v_und,
                                         condition_tokens,
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "q_und_norm." + suffix).c_str(),
                                         layer0_trace_q_und_norm,
                                         condition_tokens,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "k_und_norm." + suffix).c_str(),
                                         layer0_trace_k_und_norm,
                                         condition_tokens,
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "q_und_rot." + suffix).c_str(),
                                         layer0_trace_q_und_rot,
                                         condition_tokens,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "k_und_rot." + suffix).c_str(),
                                         layer0_trace_k_und_rot,
                                         condition_tokens,
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "attn_und." + suffix).c_str(),
                                         layer0_trace_attn_und,
                                         condition_tokens,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "o_und." + suffix).c_str(),
                                         layer0_trace_o_und,
                                         condition_tokens,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "residual_und." + suffix).c_str(),
                                         layer0_trace_residual_und,
                                         condition_tokens,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "post_norm_und." + suffix).c_str(),
                                         layer0_trace_post_norm_und,
                                         condition_tokens,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "gate_und." + suffix).c_str(),
                                         layer0_trace_gate_und,
                                         condition_tokens,
                                         kLanguageIntermediate,
                                         kLanguageIntermediate,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "up_und." + suffix).c_str(),
                                         layer0_trace_up_und,
                                         condition_tokens,
                                         kLanguageIntermediate,
                                         kLanguageIntermediate,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "swiglu_und." + suffix).c_str(),
                                         layer0_trace_swiglu_und,
                                         condition_tokens,
                                         kLanguageIntermediate,
                                         kLanguageIntermediate,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "down_und." + suffix).c_str(),
                                         layer0_trace_down_und,
                                         condition_tokens,
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "vision_gen_norm." + suffix).c_str(),
                                         layer0_trace_vision_gen_norm,
                                         static_cast<int>(kMotVisionTraceRows.size()),
                                         kLanguageHidden,
                                         kLanguageHidden,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "vision_k_gen." + suffix).c_str(),
                                         layer0_trace_vision_k_gen,
                                         static_cast<int>(kMotVisionTraceRows.size()),
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "vision_v_gen." + suffix).c_str(),
                                         layer0_trace_vision_v_gen,
                                         static_cast<int>(kMotVisionTraceRows.size()),
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "vision_k_gen_norm." + suffix).c_str(),
                                         layer0_trace_vision_k_gen_norm,
                                         static_cast<int>(kMotVisionTraceRows.size()),
                                         kLanguageKv,
                                         kLanguageKv,
                                         error) &&
               append_cuda_prefix_tensor(tensors,
                                         (base + "vision_k_gen_rot." + suffix).c_str(),
                                         layer0_trace_vision_k_gen_rot,
                                         static_cast<int>(kMotVisionTraceRows.size()),
                                         kLanguageKv,
                                         kLanguageKv,
                                         error);
    }

    bool append_layer_trace_tensors(std::vector<WamTensor> * tensors,
                                    const char * suffix,
                                    int condition_tokens,
                                    std::string * error) const {
        const std::string base = std::string("cosmos3.debug.mot.layer_trace.");
        for (size_t i = 0; i < kMotLayerTraceLayers.size(); ++i) {
            if (!append_cuda_prefix_tensor(
                    tensors,
                    (base + "layer" + std::to_string(kMotLayerTraceLayers[i]) +
                     "_action_hidden." + suffix).c_str(),
                    layer_trace_action_hidden[i],
                    kActionSteps,
                    kLanguageHidden,
                    kLanguageHidden,
                    error)) {
                return false;
            }
            if (!append_cuda_prefix_tensor(
                    tensors,
                    (base + "layer" + std::to_string(kMotLayerTraceLayers[i]) +
                     "_condition_hidden." + suffix).c_str(),
                    layer_trace_condition_hidden[i],
                    condition_tokens,
                    kLanguageHidden,
                    kLanguageHidden,
                    error)) {
                return false;
            }
        }
        return true;
    }

    const float * gen_action_hidden() const {
        return final_norm_gen + static_cast<size_t>(kMotVisionTokens + kActionConditionTokens) * kLanguageHidden;
    }

    const float * gen_vision_hidden() const {
        return final_norm_gen;
    }

    const float * full_mrope_positions(bool uncond) const {
        return full_mrope_positions_dev[uncond ? 1 : 0];
    }

    const float * condition_mrope_positions(bool uncond) const {
        return condition_mrope_positions_dev[uncond ? 1 : 0];
    }
};
#else
struct LanguageCudaRuntime {};
struct ActionBridgeCudaRuntime {};
struct MotActionCudaRuntime {};
struct MotGenCudaRuntime {};
struct WanVaeCudaRuntime {};
#endif

struct Cosmos3ModelArch : public ModelArchBase {
    Cosmos3ModelArch(std::unique_ptr<GgufGpuResidentModel> weights,
                     cosmos3_visual_cuda_ctx * visual_cuda,
                     LanguageCudaRuntime * language_cuda,
                     ActionBridgeCudaRuntime * action_bridge_cuda,
                     MotActionCudaRuntime * mot_action_cuda,
                     MotGenCudaRuntime * mot_gen_cuda,
                     WanVaeCudaRuntime * wan_vae_cuda)
        : ModelArchBase(Arch::COSMOS3),
          weights_(std::move(weights)),
          visual_cuda_(visual_cuda),
          language_cuda_(language_cuda),
          action_bridge_cuda_(action_bridge_cuda),
          mot_action_cuda_(mot_action_cuda),
          mot_gen_cuda_(mot_gen_cuda),
          wan_vae_cuda_(wan_vae_cuda) {
        cfg.n_img = kVisualTokens;
        cfg.n_lang = 150;
        cfg.n_state = 1;
        cfg.n_prefix = kLanguageTokens;
        cfg.n_suffix = kActionSteps;
        cfg.n_full = cfg.n_prefix + cfg.n_suffix;
        cfg.hidden = kLanguageHidden;
        cfg.expert_h = kLanguageHidden;
        cfg.intermediate = kLanguageIntermediate;
        cfg.expert_inter = kLanguageIntermediate;
        cfg.n_q_heads = kLanguageQHeads;
        cfg.n_kv_heads = kLanguageKvHeads;
        cfg.head_dim = kLanguageHeadDim;
        cfg.q_full_dim = kLanguageHidden;
        cfg.kv_full_dim = kLanguageKv;
        cfg.n_layers = kLanguageLayers;
        cfg.max_state_dim = kActionMaxDim;
        cfg.max_action_dim = kActionMaxDim;
        cfg.real_state_dim = kActionDim;
        cfg.real_action_dim = kActionDim;
        cfg.num_steps = 4;
        cfg.rms_eps = kLanguageRmsEps;
        cfg.rope_freq_base = kLanguageRopeTheta;
    }

    std::vector<float> predict(const Inputs &) override {
        stats = Stats{};
        std::fprintf(stderr, "vla(cosmos3): predict() is not enabled on the new WAM-first graph path\n");
        return {};
    }

    bool supports_wam() const override { return true; }

    WamOutput predict_wam(const WamInputs & inputs) override {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        stats = Stats{};

        WamOutput out;
        out.action_steps = kActionSteps;
        out.action_dim = kActionDim;

        std::string error;
        const WamTensorView * image = find_wam_tensor(inputs, "observation/image");
        const WamTensorView * direct_qwen_input = find_wam_tensor(inputs, "cosmos3.debug.qwen_input");
        const WamTensorView * debug_action_latents = find_wam_tensor(inputs, "cosmos3.debug.action_latents");
        const WamTensorView * debug_vision_latents = find_wam_tensor(inputs, "cosmos3.debug.vision_latents");
        const WamTensorView * debug_clean_vision_condition =
            find_wam_tensor(inputs, "cosmos3.debug.clean_vision_condition");
        const WamTensorView * debug_mot_text_hidden = find_wam_tensor(inputs, "cosmos3.debug.mot_text_hidden");
        const WamTensorView * debug_mot_text_hidden_uncond =
            find_wam_tensor(inputs, "cosmos3.debug.mot_text_hidden_uncond");
        const WamTensorView * debug_torch_action_hidden_cond =
            find_wam_tensor(inputs, "cosmos3.debug.torch_action_hidden_cond");
        const WamTensorView * debug_torch_action_hidden_uncond =
            find_wam_tensor(inputs, "cosmos3.debug.torch_action_hidden_uncond");
        const WamTensorView * qwen_input_ids = find_wam_tensor(inputs, "cosmos3.qwen.input_ids");
        const WamTensorView * qwen_visual_indices = find_wam_tensor(inputs, "cosmos3.qwen.visual_indices");
        const WamTensorView * qwen_mrope_positions = find_wam_tensor(inputs, "cosmos3.qwen.mrope_positions");
        const WamTensorView * mot_text_ids = find_wam_tensor(inputs, "cosmos3.mot.text_ids");
        const WamTensorView * mot_text_ids_uncond = find_wam_tensor(inputs, "cosmos3.mot.text_ids_uncond");
        LanguageCudaRuntime::TraceConfig language_trace;
        language_trace.enabled = wam_param_bool(inputs, "cosmos3.debug_layer0_trace", false);
        language_trace.simple_rope = wam_param_bool(inputs, "cosmos3.debug_language_simple_rope", false);
        language_trace.timing = wam_param_bool(inputs, "cosmos3.debug_language_timing", false);
        language_trace.layers = wam_param_int(inputs, "cosmos3.debug_language_trace_layers", kLanguageLayers);
        language_trace.tokens = wam_param_int(inputs, "cosmos3.debug_language_trace_tokens", 0);
        language_trace.layer_index = wam_param_int(inputs, "cosmos3.debug_language_trace_layer", 0);
        language_trace.cols = wam_param_int(inputs, "cosmos3.debug_language_trace_cols", 16);
        if (wam_param_bool(inputs, "cosmos3.debug_language_full_sequence", false)) {
            language_trace.tokens = kLanguageTokens;
        }
        const bool visual_timing = wam_param_bool(inputs, "cosmos3.debug_visual_timing", false);
        const bool visual_tokens_debug = wam_param_bool(inputs, "cosmos3.debug_visual_tokens", false);
        const bool wan_vae_patch_debug = wam_param_bool(inputs, "cosmos3.debug_vae_patch_input", false);
        const bool wan_vae_conv1_debug = wam_param_bool(inputs, "cosmos3.debug_vae_encoder_conv1", false);
        const bool wan_vae_down0_shortcut_debug =
            wam_param_bool(inputs, "cosmos3.debug_vae_down0_shortcut", false);
        const bool wan_vae_down0_debug =
            wam_param_bool(inputs, "cosmos3.debug_vae_down0", false);
        const bool wan_vae_down1_debug =
            wam_param_bool(inputs, "cosmos3.debug_vae_down1", false);
        const bool wan_vae_down2_debug =
            wam_param_bool(inputs, "cosmos3.debug_vae_down2", false);
        const bool wan_vae_down3_debug =
            wam_param_bool(inputs, "cosmos3.debug_vae_down3", false);
        const bool wan_vae_mid0_debug =
            wam_param_bool(inputs, "cosmos3.debug_vae_mid0", false);
        const bool wan_vae_mid_attn_debug =
            wam_param_bool(inputs, "cosmos3.debug_vae_mid_attn", false);
        const bool wan_vae_mid2_debug =
            wam_param_bool(inputs, "cosmos3.debug_vae_mid2", false);
        const bool wan_vae_head_debug =
            wam_param_bool(inputs, "cosmos3.debug_vae_head", false);
        const bool wan_vae_final_conv1_debug =
            wam_param_bool(inputs, "cosmos3.debug_vae_final_conv1", false);
        const bool wan_vae_clean_condition_debug =
            wam_param_bool(inputs, "cosmos3.debug_vae_clean_condition", false);
        const int wan_vae_patch_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_patch_input_rows", 16),
                                 COSMOS3_WAN_VAE_PATCH_W *
                                     COSMOS3_WAN_VAE_PATCH_H *
                                     COSMOS3_WAN_VAE_FRAMES));
        const int wan_vae_conv1_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_encoder_conv1_rows", 16),
                                 COSMOS3_WAN_VAE_PATCH_W *
                                     COSMOS3_WAN_VAE_PATCH_H *
                                     COSMOS3_WAN_VAE_FRAMES));
        const int wan_vae_conv1_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_encoder_conv1_cols", 32),
                                 int(COSMOS3_WAN_VAE_CONV1_CHANNELS)));
        const int wan_vae_down0_shortcut_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_down0_shortcut_rows", 16),
                                 COSMOS3_WAN_VAE_DOWN0_W *
                                     COSMOS3_WAN_VAE_DOWN0_H *
                                     COSMOS3_WAN_VAE_DOWN0_T));
        const int wan_vae_down0_shortcut_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_down0_shortcut_cols", 32),
                                 int(COSMOS3_WAN_VAE_DOWN0_CHANNELS)));
        const int wan_vae_down0_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_down0_rows", 16),
                                 COSMOS3_WAN_VAE_DOWN0_W *
                                     COSMOS3_WAN_VAE_DOWN0_H *
                                     COSMOS3_WAN_VAE_DOWN0_T));
        const int wan_vae_down0_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_down0_cols", 32),
                                 int(COSMOS3_WAN_VAE_DOWN0_CHANNELS)));
        const int wan_vae_down1_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_down1_rows", 16),
                                 COSMOS3_WAN_VAE_DOWN1_W *
                                     COSMOS3_WAN_VAE_DOWN1_H *
                                     COSMOS3_WAN_VAE_DOWN1_T));
        const int wan_vae_down1_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_down1_cols", 32),
                                 int(COSMOS3_WAN_VAE_DOWN1_CHANNELS)));
        const int wan_vae_down2_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_down2_rows", 16),
                                 COSMOS3_WAN_VAE_DOWN2_W *
                                     COSMOS3_WAN_VAE_DOWN2_H *
                                     COSMOS3_WAN_VAE_DOWN2_T));
        const int wan_vae_down2_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_down2_cols", 32),
                                 int(COSMOS3_WAN_VAE_DOWN2_CHANNELS)));
        const int wan_vae_down3_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_down3_rows", 16),
                                 COSMOS3_WAN_VAE_DOWN2_W *
                                     COSMOS3_WAN_VAE_DOWN2_H *
                                     COSMOS3_WAN_VAE_DOWN2_T));
        const int wan_vae_down3_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_down3_cols", 32),
                                 int(COSMOS3_WAN_VAE_DOWN2_CHANNELS)));
        const int wan_vae_mid0_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_mid0_rows", 16),
                                 COSMOS3_WAN_VAE_DOWN2_W *
                                     COSMOS3_WAN_VAE_DOWN2_H *
                                     COSMOS3_WAN_VAE_DOWN2_T));
        const int wan_vae_mid0_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_mid0_cols", 32),
                                 int(COSMOS3_WAN_VAE_DOWN2_CHANNELS)));
        const int wan_vae_mid_attn_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_mid_attn_rows", 16),
                                 COSMOS3_WAN_VAE_DOWN2_W *
                                     COSMOS3_WAN_VAE_DOWN2_H *
                                     COSMOS3_WAN_VAE_DOWN2_T));
        const int wan_vae_mid_attn_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_mid_attn_cols", 32),
                                 int(COSMOS3_WAN_VAE_DOWN2_CHANNELS)));
        const int wan_vae_mid2_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_mid2_rows", 16),
                                 COSMOS3_WAN_VAE_DOWN2_W *
                                     COSMOS3_WAN_VAE_DOWN2_H *
                                     COSMOS3_WAN_VAE_DOWN2_T));
        const int wan_vae_mid2_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_mid2_cols", 32),
                                 int(COSMOS3_WAN_VAE_DOWN2_CHANNELS)));
        const int wan_vae_head_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_head_rows", 16),
                                 COSMOS3_WAN_VAE_DOWN2_W *
                                     COSMOS3_WAN_VAE_DOWN2_H *
                                     COSMOS3_WAN_VAE_DOWN2_T));
        const int wan_vae_head_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_head_cols", 32),
                                 2 * kMotVisionVaeChannels));
        const int wan_vae_final_conv1_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_final_conv1_rows", 16),
                                 COSMOS3_WAN_VAE_DOWN2_W *
                                     COSMOS3_WAN_VAE_DOWN2_H *
                                     COSMOS3_WAN_VAE_DOWN2_T));
        const int wan_vae_final_conv1_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_final_conv1_cols", 32),
                                 2 * kMotVisionVaeChannels));
        const int wan_vae_clean_condition_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_clean_condition_rows", 16),
                                 kMotVisionConditionTokens));
        const int wan_vae_clean_condition_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_vae_clean_condition_cols", 32),
                                 kMotVisionPatchDim));
        const int visual_tokens_debug_rows =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_visual_tokens_rows", 16),
                                 kVisualPatchRows));
        const int visual_tokens_debug_cols =
            std::max(1, std::min(wam_param_int(inputs, "cosmos3.debug_visual_tokens_cols", 32),
                                 kVisualIntermediate));
        const bool action_bridge_debug = wam_param_bool(inputs, "cosmos3.debug_action_bridge", false);
        const bool mot_action_debug = wam_param_bool(inputs, "cosmos3.debug_mot_action_input", false);
        const bool mot_gen_debug = wam_param_bool(inputs, "cosmos3.debug_mot_gen", false);
        const bool first_action_step_debug = wam_param_bool(inputs, "cosmos3.debug_first_action_step", false);
        const bool all_action_steps_debug = wam_param_bool(inputs, "cosmos3.debug_all_action_steps", false);
        const bool denoise_timing = wam_param_bool(inputs, "cosmos3.debug_denoise_timing", false);
        const bool denoise_profile = wam_param_bool(inputs, "cosmos3.profile_denoise_events", false);
        const int mot_internal_trace_layer =
            std::max(0, std::min(wam_param_int(inputs, "cosmos3.debug_mot_internal_trace_layer", 0),
                                 kLanguageLayers - 1));
        const float mot_action_timestep = wam_param_float(inputs, "cosmos3.debug_mot_timestep", 1.0f);
        const int num_denoise_steps = std::max(1, wam_param_int(inputs, "cosmos3.num_steps", cfg.num_steps));
        const float guidance = wam_param_float(inputs, "cosmos3.guidance", 3.0f);
        const float sampler_shift = wam_param_float(inputs, "cosmos3.shift", 5.0f);
        const int sampler_seed = wam_param_int(inputs, "cosmos3.seed", 0);
        const int action_domain_id = wam_param_int(inputs, "cosmos3.action_domain_id", kActionDefaultDomainId);
        const bool invert_gripper = wam_param_bool(inputs, "cosmos3.invert_gripper", true);
        const bool enable_mot_condition_cache =
            wam_param_bool(inputs, "cosmos3.enable_mot_condition_cache", false);
        if (mot_gen_cuda_) {
            mot_gen_cuda_->set_internal_trace_layer(mot_internal_trace_layer);
            mot_gen_cuda_->set_trace_enabled(first_action_step_debug || mot_gen_debug,
                                             first_action_step_debug);
            if (first_action_step_debug || mot_gen_debug) {
                mot_gen_cuda_->invalidate_condition_cache();
            }
            mot_gen_cuda_->set_timing(denoise_timing);
            mot_gen_cuda_->set_profile(denoise_profile);
        }
        bool visual_executed = false;
        bool language_executed = false;
        if (!visual_cuda_) {
            error = "Cosmos3 direct visual CUDA runtime is not initialized";
        } else if (!language_cuda_) {
            error = "Cosmos3 direct language CUDA runtime is not initialized";
        } else if (!action_bridge_cuda_) {
            error = "Cosmos3 direct action bridge CUDA runtime is not initialized";
        } else if (!mot_action_cuda_) {
            error = "Cosmos3 direct MoT action CUDA runtime is not initialized";
        } else if (!mot_gen_cuda_) {
            error = "Cosmos3 direct MoT gen CUDA runtime is not initialized";
        } else if (!wan_vae_cuda_) {
            error = "Cosmos3 Wan VAE CUDA runtime is not initialized";
        } else if (!image) {
            error = "Cosmos3 direct visual CUDA runtime requires observation/image";
        } else if (image->dtype != WamDType::U8 ||
                   image->shape.size() != 3 ||
                   image->shape[0] <= 0 ||
                   image->shape[1] <= 0 ||
                   image->shape[2] != 3 ||
                   image->bytes != static_cast<size_t>(image->shape[0]) *
                                   static_cast<size_t>(image->shape[1]) * 3u) {
            error = "observation/image must be U8 [H,W,3]";
        } else {
#ifdef VLA_COSMOS3_CUDA_KERNELS
            cosmos3_visual_cuda_set_timing(visual_cuda_.get(), visual_timing ? 1 : 0);
            cosmos3_visual_cuda_set_debug_prefix(visual_cuda_.get(),
                                                 visual_tokens_debug ? 1 : 0,
                                                 visual_tokens_debug_rows,
                                                 visual_tokens_debug_cols);
            const auto t_vision0 = clock::now();
            const int rc = cosmos3_visual_cuda_forward_robolab_image(
                visual_cuda_.get(),
                static_cast<const unsigned char *>(image->data),
                static_cast<int>(image->shape[0]),
                static_cast<int>(image->shape[1]),
                nullptr);
            cudaError_t status = cudaDeviceSynchronize();
            if (rc != 0 || status != cudaSuccess) {
                error = std::string("Cosmos3 direct visual CUDA runtime failed: ") +
                        cudaGetErrorString(status);
            } else {
                visual_executed = true;
                stats.ms_vision = std::chrono::duration<float, std::milli>(clock::now() - t_vision0).count();
                wan_vae_cuda_->set_timing(language_trace.timing);
                float timing_wan_vae_ms = 0.0f;
                const auto t_wan_vae0 = clock::now();
                if (!wan_vae_cuda_->forward_robolab_image(*image, nullptr, &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE patchify failed";
                } else if (language_trace.timing) {
                    cudaError_t vae_status = cudaDeviceSynchronize();
                    timing_wan_vae_ms =
                        std::chrono::duration<float, std::milli>(clock::now() - t_wan_vae0).count();
                    if (vae_status != cudaSuccess) {
                        error = std::string("Cosmos3 Wan VAE timing sync failed: ") +
                                cudaGetErrorString(vae_status);
                    }
                }
                if (error.empty() && wan_vae_patch_debug &&
                    !wan_vae_cuda_->append_patch_prefix_tensor(&out.tensors,
                                                               wan_vae_patch_debug_rows,
                                                               nullptr,
                                                               &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE patch debug copy failed";
                }
                if (error.empty() && wan_vae_conv1_debug &&
                    !wan_vae_cuda_->append_conv1_prefix_tensor(&out.tensors,
                                                               wan_vae_conv1_debug_rows,
                                                               wan_vae_conv1_debug_cols,
                                                               nullptr,
                                                               &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE encoder.conv1 debug copy failed";
                }
                if (error.empty() && wan_vae_down0_shortcut_debug &&
                    !wan_vae_cuda_->append_down0_shortcut_prefix_tensor(
                        &out.tensors,
                        wan_vae_down0_shortcut_debug_rows,
                        wan_vae_down0_shortcut_debug_cols,
                        nullptr,
                        &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE down0 shortcut debug copy failed";
                }
                if (error.empty() && wan_vae_down0_debug &&
                    !wan_vae_cuda_->append_down0_prefix_tensor(&out.tensors,
                                                               wan_vae_down0_debug_rows,
                                                               wan_vae_down0_debug_cols,
                                                               nullptr,
                                                               &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE down0 debug copy failed";
                }
                if (error.empty() && wan_vae_down1_debug &&
                    !wan_vae_cuda_->append_down1_prefix_tensor(&out.tensors,
                                                               wan_vae_down1_debug_rows,
                                                               wan_vae_down1_debug_cols,
                                                               nullptr,
                                                               &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE down1 debug copy failed";
                }
                if (error.empty() && wan_vae_down2_debug &&
                    !wan_vae_cuda_->append_down2_prefix_tensor(&out.tensors,
                                                               wan_vae_down2_debug_rows,
                                                               wan_vae_down2_debug_cols,
                                                               nullptr,
                                                               &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE down2 debug copy failed";
                }
                if (error.empty() && wan_vae_down3_debug &&
                    !wan_vae_cuda_->append_down3_prefix_tensor(&out.tensors,
                                                               wan_vae_down3_debug_rows,
                                                               wan_vae_down3_debug_cols,
                                                               nullptr,
                                                               &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE down3 debug copy failed";
                }
                if (error.empty() && wan_vae_mid0_debug &&
                    !wan_vae_cuda_->append_mid0_prefix_tensor(&out.tensors,
                                                              wan_vae_mid0_debug_rows,
                                                              wan_vae_mid0_debug_cols,
                                                              nullptr,
                                                              &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE mid0 debug copy failed";
                }
                if (error.empty() && wan_vae_mid_attn_debug &&
                    !wan_vae_cuda_->append_mid_attn_prefix_tensor(
                        &out.tensors,
                        wan_vae_mid_attn_debug_rows,
                        wan_vae_mid_attn_debug_cols,
                        nullptr,
                        &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE middle attention debug copy failed";
                }
                if (error.empty() && wan_vae_mid2_debug &&
                    !wan_vae_cuda_->append_mid2_prefix_tensor(&out.tensors,
                                                              wan_vae_mid2_debug_rows,
                                                              wan_vae_mid2_debug_cols,
                                                              nullptr,
                                                              &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE mid2 debug copy failed";
                }
                if (error.empty() && wan_vae_head_debug &&
                    !wan_vae_cuda_->append_head_prefix_tensor(&out.tensors,
                                                              wan_vae_head_debug_rows,
                                                              wan_vae_head_debug_cols,
                                                              nullptr,
                                                              &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE encoder head debug copy failed";
                }
                if (error.empty() && wan_vae_final_conv1_debug &&
                    !wan_vae_cuda_->append_final_conv1_prefix_tensor(
                        &out.tensors,
                        wan_vae_final_conv1_debug_rows,
                        wan_vae_final_conv1_debug_cols,
                        nullptr,
                        &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE final conv1 debug copy failed";
                }
                if (error.empty() && wan_vae_clean_condition_debug &&
                    !wan_vae_cuda_->append_clean_vision_condition_prefix_tensor(
                        &out.tensors,
                        wan_vae_clean_condition_debug_rows,
                        wan_vae_clean_condition_debug_cols,
                        nullptr,
                        &error)) {
                    if (error.empty()) error = "Cosmos3 Wan VAE clean condition debug copy failed";
                }
                const auto t_prefill0 = clock::now();
                if (visual_timing) {
                    float visual_timing_ms[15] = {};
                    const int count = cosmos3_visual_cuda_copy_timing_ms(
                        visual_cuda_.get(), visual_timing_ms, 15);
                    if (count > 0) {
                        append_visual_timing_tensor(&out.tensors, visual_timing_ms, count);
                    }
                }
                if (visual_tokens_debug) {
                    auto append_visual_debug = [&](const char * name, int index) -> bool {
                        return append_cuda_prefix_tensor(&out.tensors,
                                                         name,
                                                         cosmos3_visual_cuda_debug_prefix(visual_cuda_.get(), index),
                                                         visual_tokens_debug_rows,
                                                         visual_tokens_debug_cols,
                                                         visual_tokens_debug_cols,
                                                         &error);
                    };
                    if (!append_visual_debug("cosmos3.debug.visual.token_entry.prefix",
                                             COSMOS3_VISUAL_DEBUG_TOKEN_ENTRY) ||
                        !append_visual_debug("cosmos3.debug.visual.block0.norm1.prefix",
                                             COSMOS3_VISUAL_DEBUG_BLOCK0_NORM1) ||
                        !append_visual_debug("cosmos3.debug.visual.block0.qkv.prefix",
                                             COSMOS3_VISUAL_DEBUG_BLOCK0_QKV) ||
                        !append_visual_debug("cosmos3.debug.visual.block0.qkv_k.prefix",
                                             COSMOS3_VISUAL_DEBUG_BLOCK0_QKV_K) ||
                        !append_visual_debug("cosmos3.debug.visual.block0.qkv_v.prefix",
                                             COSMOS3_VISUAL_DEBUG_BLOCK0_QKV_V) ||
                        !append_visual_debug("cosmos3.debug.visual.block0.q_rope.prefix",
                                             COSMOS3_VISUAL_DEBUG_BLOCK0_Q_ROPE) ||
                        !append_visual_debug("cosmos3.debug.visual.block0.k_rope.prefix",
                                             COSMOS3_VISUAL_DEBUG_BLOCK0_K_ROPE) ||
                        !append_visual_debug("cosmos3.debug.visual.block0.attn.prefix",
                                             COSMOS3_VISUAL_DEBUG_BLOCK0_ATTN) ||
                        !append_visual_debug("cosmos3.debug.visual.block0.attn_proj.prefix",
                                             COSMOS3_VISUAL_DEBUG_BLOCK0_ATTN_PROJ) ||
                        !append_visual_debug("cosmos3.debug.visual.block0.post_attn.prefix",
                                             COSMOS3_VISUAL_DEBUG_BLOCK0_POST_ATTN) ||
                        !append_visual_debug("cosmos3.debug.visual.block0.norm2.prefix",
                                             COSMOS3_VISUAL_DEBUG_BLOCK0_NORM2) ||
                        !append_visual_debug("cosmos3.debug.visual.block0.mlp.prefix",
                                             COSMOS3_VISUAL_DEBUG_BLOCK0_MLP) ||
                        !append_visual_debug("cosmos3.debug.visual.block0.prefix",
                                             COSMOS3_VISUAL_DEBUG_BLOCK0) ||
                        !append_visual_debug("cosmos3.debug.visual.final_hidden.prefix",
                                             COSMOS3_VISUAL_DEBUG_FINAL_HIDDEN) ||
                        !append_visual_debug("cosmos3.debug.visual.merger_norm.prefix",
                                             COSMOS3_VISUAL_DEBUG_MERGER_NORM) ||
                        !append_visual_debug("cosmos3.debug.visual.merged.prefix",
                                             COSMOS3_VISUAL_DEBUG_MERGED) ||
                        !append_visual_debug("cosmos3.debug.visual.merger_h.prefix",
                                             COSMOS3_VISUAL_DEBUG_MERGER_H) ||
                        !append_cuda_prefix_tensor(&out.tensors,
                                                   "cosmos3.debug.visual.tokens.prefix",
                                                   cosmos3_visual_cuda_tokens(visual_cuda_.get()),
                                                   visual_tokens_debug_rows,
                                                   visual_tokens_debug_cols,
                                                   kLanguageHidden,
                                                   &error)) {
                        if (error.empty()) error = "Cosmos3 visual debug prefix copy failed";
                    }
                }
                if (language_trace.enabled) {
                    const int trace_rows = std::max(1, std::min(language_trace.tokens > 0 ? language_trace.tokens : 3, kLanguageTokens));
                    const int visual_rows = std::max(1, std::min(trace_rows, kVisualTokens));
                    if (!append_cuda_prefix_tensor(&out.tensors,
                                                   "cosmos3.debug.visual.tokens.prefix",
                                                   cosmos3_visual_cuda_tokens(visual_cuda_.get()),
                                                   visual_rows,
                                                   16,
                                                   kLanguageHidden,
                                                   &error) ||
                        !append_cuda_prefix_tensor(&out.tensors,
                                                   "cosmos3.debug.visual.deepstack0.prefix",
                                                   cosmos3_visual_cuda_deepstack_tokens(visual_cuda_.get(), 0),
                                                   visual_rows,
                                                   16,
                                                   kLanguageHidden,
                                                   &error) ||
                        !append_cuda_prefix_tensor(&out.tensors,
                                                   "cosmos3.debug.visual.deepstack1.prefix",
                                                   cosmos3_visual_cuda_deepstack_tokens(visual_cuda_.get(), 1),
                                                   visual_rows,
                                                   16,
                                                   kLanguageHidden,
                                                   &error) ||
                        !append_cuda_prefix_tensor(&out.tensors,
                                                   "cosmos3.debug.visual.deepstack2.prefix",
                                                   cosmos3_visual_cuda_deepstack_tokens(visual_cuda_.get(), 2),
                                                   visual_rows,
                                                   16,
                                                   kLanguageHidden,
                                                   &error)) {
                        if (error.empty()) error = "Cosmos3 visual trace copy failed";
                    }
                }
                float timing_prefill_language_total_ms = 0.0f;
                float timing_prefill_mot_setup_ms = 0.0f;
                const auto t_language_total0 = clock::now();
                if (!error.empty()) {
                    // Return the trace-copy error below.
                } else if (!language_cuda_->configure_request_tokens(qwen_input_ids,
                                                                     qwen_visual_indices,
                                                                     qwen_mrope_positions,
                                                                     mot_text_ids,
                                                                     mot_text_ids_uncond,
                                                                     &error)) {
                    if (error.empty()) error = "Cosmos3 dynamic text token upload failed";
                } else if (!language_cuda_->forward(cosmos3_visual_cuda_tokens(visual_cuda_.get()),
                                             cosmos3_visual_cuda_deepstack_tokens(visual_cuda_.get(), 0),
                                             cosmos3_visual_cuda_deepstack_tokens(visual_cuda_.get(), 1),
                                             cosmos3_visual_cuda_deepstack_tokens(visual_cuda_.get(), 2),
                                             direct_qwen_input,
                                             language_trace,
	                                             &out.tensors,
	                                             &error)) {
	                    if (error.empty()) error = "Cosmos3 direct language CUDA runtime failed";
	                } else {
	                    if (language_trace.timing) {
	                        cudaError_t language_status = cudaDeviceSynchronize();
	                        timing_prefill_language_total_ms =
	                            std::chrono::duration<float, std::milli>(clock::now() - t_language_total0).count();
	                        if (language_status != cudaSuccess) {
	                            error = std::string("Cosmos3 language timing sync failed: ") +
	                                    cudaGetErrorString(language_status);
	                        }
	                    }
	                    language_executed = true;
	                    const auto t_mot_setup0 = clock::now();
	                    float timing_prefill_clean_ptr_ms = 0.0f;
	                    float timing_prefill_reset_latents_ms = 0.0f;
	                    float timing_prefill_cond_cache_ms = 0.0f;
	                    float timing_prefill_uncond_cache_ms = 0.0f;
	                    if (action_bridge_debug) {
	                        const float * last_hidden = language_cuda_->final_norm_last_token();
                        if (!action_bridge_cuda_->llm_to_action_probe(
                                last_hidden,
                                action_domain_id,
                                language_cuda_->cuda_stream(),
                                &error) ||
                            !action_bridge_cuda_->append_llm_to_action_tensor(
                                &out.tensors,
                                language_cuda_->cuda_stream(),
                                &error)) {
                            if (error.empty()) error = "Cosmos3 action bridge debug probe failed";
                        }
                    }
                    if (error.empty()) {
                        if (!language_cuda_->override_mot_condition_inputs(debug_mot_text_hidden,
                                                                           debug_mot_text_hidden_uncond,
                                                                           &error)) {
                            if (error.empty()) error = "Cosmos3 MoT condition override failed";
                        }
                        const float * condition_hidden = language_cuda_->condition_input_tokens(false);
                        const float * uncond_condition_hidden = language_cuda_->condition_input_tokens(true);
                        const int condition_tokens = language_cuda_->condition_token_count(false);
                        const int uncond_condition_tokens = language_cuda_->condition_token_count(true);
                        const uint64_t condition_cache_key =
                            cosmos3_mot_condition_cache_key(mot_text_ids,
                                                            kRobolabMotCondTextIds.data(),
                                                            kRobolabMotCondTextIds.size(),
                                                            debug_mot_text_hidden,
                                                            false);
                        const uint64_t uncond_condition_cache_key =
                            cosmos3_mot_condition_cache_key(mot_text_ids_uncond,
                                                            kRobolabMotUncondTextIds.data(),
                                                            kRobolabMotUncondTextIds.size(),
                                                            debug_mot_text_hidden_uncond,
                                                            true);
                        if (!mot_gen_cuda_->configure_position_tables(condition_tokens,
                                                                      uncond_condition_tokens,
                                                                      language_cuda_->cuda_stream(),
                                                                      &error)) {
                            if (error.empty()) error = "Cosmos3 MoT dynamic position table setup failed";
                        }
                        const bool single_step_debug = mot_action_debug || mot_gen_debug;
                        if (error.empty() && debug_torch_action_hidden_cond) {
                            if (!action_bridge_cuda_->llm_to_noisy_action_tokens_from_tensor(
                                    debug_torch_action_hidden_cond,
                                    action_domain_id,
                                    language_cuda_->cuda_stream(),
                                    &error) ||
                                !mot_action_cuda_->mask_action_velocity(
                                    action_bridge_cuda_->mutable_full_action_output(),
                                    language_cuda_->cuda_stream(),
                                    &error) ||
                                !action_bridge_cuda_->append_action_tensor(
                                    &out.tensors,
                                    "cosmos3.debug.action_bridge.torch_hidden_cond_velocity",
                                    action_bridge_cuda_->full_action_output(),
                                    language_cuda_->cuda_stream(),
                                    &error)) {
                                if (error.empty()) error = "Cosmos3 torch-hidden cond llm2action debug failed";
                            }
                        }
                        if (error.empty() && debug_torch_action_hidden_uncond) {
                            if (!action_bridge_cuda_->llm_to_noisy_action_tokens_from_tensor(
                                    debug_torch_action_hidden_uncond,
                                    action_domain_id,
                                    language_cuda_->cuda_stream(),
                                    &error) ||
                                !mot_action_cuda_->mask_action_velocity(
                                    action_bridge_cuda_->mutable_full_action_output(),
                                    language_cuda_->cuda_stream(),
                                    &error) ||
                                !action_bridge_cuda_->append_action_tensor(
                                    &out.tensors,
                                    "cosmos3.debug.action_bridge.torch_hidden_uncond_velocity",
                                    action_bridge_cuda_->full_action_output(),
                                    language_cuda_->cuda_stream(),
                                    &error)) {
                                if (error.empty()) error = "Cosmos3 torch-hidden uncond llm2action debug failed";
                            }
                        }
	                        const float * native_clean_vision_condition = nullptr;
	                        if (error.empty() && !debug_clean_vision_condition) {
	                            const auto t_clean_ptr0 = clock::now();
	                            native_clean_vision_condition =
	                                wan_vae_cuda_->clean_vision_condition_ptr(language_cuda_->cuda_stream(), &error);
	                            if (language_trace.timing) {
	                                timing_prefill_clean_ptr_ms =
	                                    std::chrono::duration<float, std::milli>(clock::now() - t_clean_ptr0).count();
	                            }
	                            if (error.empty() && !native_clean_vision_condition) {
	                                error = "Cosmos3 Wan VAE clean vision condition was not produced";
	                            }
	                        }
	                        if (error.empty() &&
	                            ![&]() {
	                                const auto t_reset0 = clock::now();
	                                const bool ok = mot_action_cuda_->reset_latents(debug_action_latents,
	                                                                               debug_vision_latents,
	                                                                               debug_clean_vision_condition,
	                                                                               native_clean_vision_condition,
	                                                                               inputs.state,
	                                                                               sampler_seed,
	                                                                               language_cuda_->cuda_stream(),
	                                                                               &error);
	                                if (language_trace.timing) {
	                                    cudaError_t reset_status = cudaDeviceSynchronize();
	                                    timing_prefill_reset_latents_ms =
	                                        std::chrono::duration<float, std::milli>(clock::now() - t_reset0).count();
	                                    if (reset_status != cudaSuccess) {
	                                        error = std::string("Cosmos3 MoT reset_latents timing sync failed: ") +
	                                                cudaGetErrorString(reset_status);
	                                        return false;
	                                    }
	                                }
	                                return ok;
	                            }()) {
	                            if (error.empty()) error = "Cosmos3 MoT action latent initialization failed";
	                        }
                        if (error.empty() && single_step_debug) {
                            if (!mot_action_cuda_->forward_current_latents(condition_hidden,
                                                                          condition_tokens,
                                                                          mot_action_timestep,
                                                                          action_domain_id,
                                                                          language_cuda_->cuda_stream(),
                                                                          &error)) {
                                if (error.empty()) error = "Cosmos3 MoT action input debug path failed";
                            }
                            if (error.empty() && mot_action_debug &&
                                !mot_action_cuda_->append_debug_tensors(&out.tensors,
                                                                        language_cuda_->cuda_stream(),
                                                                        &error)) {
                                if (error.empty()) error = "Cosmos3 MoT action input debug tensor copy failed";
                            }
                            if (error.empty() &&
                                !mot_gen_cuda_->forward(mot_action_cuda_->packed_sequence_tokens(),
                                                        mot_gen_cuda_->condition_mrope_positions(false),
                                                        mot_gen_cuda_->full_mrope_positions(false),
                                                        condition_tokens,
                                                        kMotFullTokens,
                                                        language_cuda_->cuda_stream(),
                                                        &error)) {
                                if (error.empty()) error = "Cosmos3 MoT generation debug path failed";
                            }
                            if (error.empty() && mot_gen_debug &&
                                !mot_gen_cuda_->append_debug_tensors(&out.tensors, &error)) {
                                if (error.empty()) error = "Cosmos3 MoT generation debug tensor copy failed";
                            }
                        }
	                        if (error.empty() && enable_mot_condition_cache) {
	                            const auto t_cond_cache0 = clock::now();
	                            if (!mot_gen_cuda_->prepare_condition_cache(0,
	                                                                         condition_hidden,
	                                                                         mot_gen_cuda_->condition_mrope_positions(false),
	                                                                         condition_tokens,
	                                                                         condition_cache_key,
	                                                                         language_cuda_->cuda_stream(),
	                                                                         &error)) {
	                                if (error.empty()) error = "Cosmos3 MoT condition cache preparation failed";
	                            } else if (language_trace.timing) {
	                                cudaError_t cache_status = cudaDeviceSynchronize();
	                                timing_prefill_cond_cache_ms =
	                                    std::chrono::duration<float, std::milli>(clock::now() - t_cond_cache0).count();
	                                if (cache_status != cudaSuccess) {
	                                    error = std::string("Cosmos3 MoT condition cache timing sync failed: ") +
	                                            cudaGetErrorString(cache_status);
	                                }
	                            }
	                        }
	                        if (error.empty() && enable_mot_condition_cache && guidance != 1.0f) {
	                            const auto t_uncond_cache0 = clock::now();
	                            if (!mot_gen_cuda_->prepare_condition_cache(1,
	                                                                         uncond_condition_hidden,
	                                                                         mot_gen_cuda_->condition_mrope_positions(true),
	                                                                         uncond_condition_tokens,
	                                                                         uncond_condition_cache_key,
	                                                                         language_cuda_->cuda_stream(),
	                                                                         &error)) {
	                                if (error.empty()) error = "Cosmos3 MoT uncond condition cache preparation failed";
	                            } else if (language_trace.timing) {
	                                cudaError_t cache_status = cudaDeviceSynchronize();
	                                timing_prefill_uncond_cache_ms =
	                                    std::chrono::duration<float, std::milli>(clock::now() - t_uncond_cache0).count();
	                                if (cache_status != cudaSuccess) {
	                                    error = std::string("Cosmos3 MoT uncond condition cache timing sync failed: ") +
	                                            cudaGetErrorString(cache_status);
	                                }
	                            }
	                        }
	                        if (language_trace.timing) {
	                            cudaError_t mot_setup_status = cudaDeviceSynchronize();
	                            timing_prefill_mot_setup_ms =
	                                std::chrono::duration<float, std::milli>(clock::now() - t_mot_setup0).count();
	                            if (mot_setup_status != cudaSuccess) {
	                                error = std::string("Cosmos3 MoT setup timing sync failed: ") +
	                                        cudaGetErrorString(mot_setup_status);
	                            }
	                        }
	                        stats.ms_prefill = std::chrono::duration<float, std::milli>(clock::now() - t_prefill0).count();
	                        if (language_trace.timing) {
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.prefill_wan_vae_ms",
	                                                        timing_wan_vae_ms);
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.prefill_language_total_ms",
	                                                        timing_prefill_language_total_ms);
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.prefill_mot_setup_ms",
	                                                        timing_prefill_mot_setup_ms);
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.prefill_clean_ptr_ms",
	                                                        timing_prefill_clean_ptr_ms);
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.prefill_reset_latents_ms",
	                                                        timing_prefill_reset_latents_ms);
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.prefill_cond_cache_ms",
	                                                        timing_prefill_cond_cache_ms);
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.prefill_uncond_cache_ms",
	                                                        timing_prefill_uncond_cache_ms);
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.patch_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_PATCH));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.conv1_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_CONV1));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.down0_shortcut_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_DOWN0_SHORTCUT));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.down0_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_DOWN0));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.down1_shortcut_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_DOWN1_SHORTCUT));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.down1_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_DOWN1));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.down2_shortcut_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_DOWN2_SHORTCUT));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.down2_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_DOWN2));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.down3_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_DOWN3));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.mid0_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_MID0));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.mid_attn_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_MID_ATTN));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.mid2_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_MID2));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.head_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_HEAD));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.final_conv1_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_FINAL_CONV1));
	                            append_scalar_timing_tensor(&out.tensors,
	                                                        "cosmos3.debug.timing.vae.clean_pack_ms",
	                                                        wan_vae_cuda_->timing_value(WanVaeCudaRuntime::VAE_T_CLEAN_PACK));
	                        }
	                        if (error.empty()) {
                            const auto t_denoise0 = clock::now();
                            enum DenoiseTimingSlot {
                                DT_COND_PACK = 0,
                                DT_COND_GEN,
                                DT_COND_DECODE,
                                DT_UNCOND_PACK,
                                DT_UNCOND_GEN,
                                DT_UNCOND_DECODE,
                                DT_CFG,
                                DT_X0,
                                DT_UNIPC,
                                DT_FINAL_COPY,
                                DT_COUNT,
                            };
                            std::array<float, DT_COUNT> denoise_detail_ms{};
                            auto time_part = [&](DenoiseTimingSlot slot, auto && fn) -> bool {
                                if (!denoise_timing) return fn();
                                const auto tb = clock::now();
                                const bool ok = fn();
                                cudaStreamSynchronize(language_cuda_->cuda_stream());
                                denoise_detail_ms[slot] +=
                                    std::chrono::duration<float, std::milli>(clock::now() - tb).count();
                                return ok;
                            };
                            const std::vector<float> sigmas =
                                cosmos3_unipc_like_sigmas(num_denoise_steps, sampler_shift);
                            const std::string mot_internal_action_hidden_base =
                                "cosmos3.debug.mot.layer" +
                                std::to_string(mot_internal_trace_layer) +
                                "_action_hidden.";
                            int lower_order_nums = 0;
                            int previous_order = 1;
                            for (int step = 0; step < num_denoise_steps && error.empty(); ++step) {
                                const float sigma = sigmas[static_cast<size_t>(step)];
                                const float timestep = sigma * 1000.0f;
                                if (all_action_steps_debug &&
                                    !mot_action_cuda_->append_step_latent_debug_tensor(
                                        &out.tensors,
                                        step,
                                        "pre_latent",
                                        language_cuda_->cuda_stream(),
                                        &error)) {
                                    if (error.empty()) error = "Cosmos3 action pre-latent debug tensor copy failed";
                                    break;
                                }
                                if (!time_part(DT_COND_PACK, [&]() {
                                        return mot_action_cuda_->forward_current_latents(condition_hidden,
                                                                                         condition_tokens,
                                                                                         timestep,
                                                                                         action_domain_id,
                                                                                         language_cuda_->cuda_stream(),
                                                                                         &error) &&
                                               (!first_action_step_debug || step != 0 ||
                                                mot_action_cuda_->append_first_step_input_tensors(
                                                    &out.tensors,
                                                    language_cuda_->cuda_stream(),
                                                    &error));
                                    }) ||
                                    !time_part(DT_COND_GEN, [&]() {
                                        const bool ok = enable_mot_condition_cache
                                            ? mot_gen_cuda_->forward_with_condition_cache(
                                                  0,
                                                  mot_action_cuda_->packed_sequence_tokens(),
                                                  condition_tokens,
                                                  language_cuda_->cuda_stream(),
                                                  &error)
                                            : mot_gen_cuda_->forward(
                                                  mot_action_cuda_->packed_sequence_tokens(),
                                                  mot_gen_cuda_->condition_mrope_positions(false),
                                                  mot_gen_cuda_->full_mrope_positions(false),
                                                  condition_tokens,
                                                  kMotFullTokens,
                                                  language_cuda_->cuda_stream(),
                                                  &error);
                                        if (!ok) return false;
                                        return (!first_action_step_debug || step != 0 ||
                                                (mot_gen_cuda_->append_action_hidden_tensor(
                                                     &out.tensors,
                                                     "cosmos3.debug.mot.gen_action_hidden.cond",
                                                     &error) &&
                                                 mot_gen_cuda_->append_layer0_action_hidden_tensor(
                                                     &out.tensors,
                                                     (mot_internal_action_hidden_base + "cond").c_str(),
                                                     &error) &&
                                                 mot_gen_cuda_->append_layer0_trace_tensors(
                                                     &out.tensors,
                                                     "cond",
                                                     condition_tokens,
                                                     &error) &&
                                                 mot_gen_cuda_->append_layer_trace_tensors(
                                                     &out.tensors,
                                                     "cond",
                                                     condition_tokens,
                                                     &error)));
                                    }) ||
                                    !time_part(DT_COND_DECODE, [&]() {
                                        return action_bridge_cuda_->llm_to_noisy_action_tokens(
                                                   mot_gen_cuda_->gen_action_hidden(),
                                                   action_domain_id,
                                                   language_cuda_->cuda_stream(),
                                                   &error) &&
                                               mot_action_cuda_->mask_action_velocity(
                                                   action_bridge_cuda_->mutable_full_action_output(),
                                                   language_cuda_->cuda_stream(),
                                                   &error) &&
                                               mot_action_cuda_->decode_vision_velocity(
                                                   mot_gen_cuda_->gen_vision_hidden(),
                                                   language_cuda_->cuda_stream(),
                                                   &error);
                                    })) {
                                    if (error.empty()) error = "Cosmos3 MoT denoise step failed";
                                    break;
                                }
                                if (guidance != 1.0f) {
                                    if (!time_part(DT_CFG, [&]() {
                                            return action_bridge_cuda_->save_cfg_cond_velocity(
                                                       language_cuda_->cuda_stream(),
                                                       &error) &&
                                                   mot_action_cuda_->save_cfg_cond_vision_velocity(
                                                       language_cuda_->cuda_stream(),
                                                       &error);
                                        }) ||
                                        !time_part(DT_UNCOND_PACK, [&]() {
                                            return mot_action_cuda_->forward_current_latents(uncond_condition_hidden,
                                                                                             uncond_condition_tokens,
                                                                                             timestep,
                                                                                             action_domain_id,
                                                                                             language_cuda_->cuda_stream(),
                                                                                             &error);
                                        }) ||
                                        !time_part(DT_UNCOND_GEN, [&]() {
                                            const bool ok = enable_mot_condition_cache
                                                ? mot_gen_cuda_->forward_with_condition_cache(
                                                      1,
                                                      mot_action_cuda_->packed_sequence_tokens(),
                                                      uncond_condition_tokens,
                                                      language_cuda_->cuda_stream(),
                                                      &error)
                                                : mot_gen_cuda_->forward(
                                                      mot_action_cuda_->packed_sequence_tokens(),
                                                      mot_gen_cuda_->condition_mrope_positions(true),
                                                      mot_gen_cuda_->full_mrope_positions(true),
                                                      uncond_condition_tokens,
                                                      kMotFullTokens,
                                                      language_cuda_->cuda_stream(),
                                                      &error);
                                            if (!ok) return false;
                                            return (!first_action_step_debug || step != 0 ||
                                                    (mot_gen_cuda_->append_action_hidden_tensor(
                                                         &out.tensors,
                                                         "cosmos3.debug.mot.gen_action_hidden.uncond",
                                                         &error) &&
                                                     mot_gen_cuda_->append_layer0_action_hidden_tensor(
                                                         &out.tensors,
                                                         (mot_internal_action_hidden_base + "uncond").c_str(),
                                                         &error) &&
                                                     mot_gen_cuda_->append_layer0_trace_tensors(
                                                         &out.tensors,
                                                         "uncond",
                                                         uncond_condition_tokens,
                                                         &error) &&
                                                     mot_gen_cuda_->append_layer_trace_tensors(
                                                         &out.tensors,
                                                         "uncond",
                                                         uncond_condition_tokens,
                                                         &error)));
                                        }) ||
                                        !time_part(DT_UNCOND_DECODE, [&]() {
                                            return action_bridge_cuda_->llm_to_noisy_action_tokens(
                                                       mot_gen_cuda_->gen_action_hidden(),
                                                       action_domain_id,
                                                       language_cuda_->cuda_stream(),
                                                       &error) &&
                                                   mot_action_cuda_->mask_action_velocity(
                                                       action_bridge_cuda_->mutable_full_action_output(),
                                                       language_cuda_->cuda_stream(),
                                                       &error) &&
                                                   mot_action_cuda_->decode_vision_velocity(
                                                       mot_gen_cuda_->gen_vision_hidden(),
                                                       language_cuda_->cuda_stream(),
                                                       &error) &&
                                                   action_bridge_cuda_->save_cfg_uncond_velocity(
                                                       language_cuda_->cuda_stream(),
                                                       &error) &&
                                                   mot_action_cuda_->save_cfg_uncond_vision_velocity(
                                                       language_cuda_->cuda_stream(),
                                                       &error);
                                        }) ||
                                        !time_part(DT_CFG, [&]() {
                                            return action_bridge_cuda_->blend_cfg_velocity(
                                                       guidance,
                                                       language_cuda_->cuda_stream(),
                                                       &error) &&
                                                   mot_action_cuda_->blend_cfg_vision_velocity(
                                                       guidance,
                                                       language_cuda_->cuda_stream(),
                                                       &error);
                                        })) {
                                        if (error.empty()) error = "Cosmos3 CFG/uncond denoise branch failed";
                                        break;
                                    }
                                }
                                if (!time_part(DT_X0, [&]() {
                                        return mot_action_cuda_->mask_action_velocity(
                                                   action_bridge_cuda_->mutable_full_action_output(),
                                                   language_cuda_->cuda_stream(),
                                                   &error);
                                    })) {
                                    if (error.empty()) error = "Cosmos3 action velocity mask failed";
                                    break;
                                }
                                if (!time_part(DT_X0, [&]() {
                                        return mot_action_cuda_->compute_x0_from_velocity(
                                                   action_bridge_cuda_->full_action_output(),
                                                   sigma,
                                                   language_cuda_->cuda_stream(),
                                                   &error);
                                    })) {
                                    if (error.empty()) error = "Cosmos3 UniPC x0 conversion failed";
                                    break;
                                }
                                if (!time_part(DT_X0, [&]() {
                                        return mot_action_cuda_->compute_vision_x0_from_velocity(
                                                   sigma,
                                                   language_cuda_->cuda_stream(),
                                                   &error);
                                    })) {
                                    if (error.empty()) error = "Cosmos3 UniPC vision x0 conversion failed";
                                    break;
                                }
                                if (first_action_step_debug && step == 0) {
                                    if (!action_bridge_cuda_->append_first_step_debug_tensors(
                                            &out.tensors,
                                            language_cuda_->cuda_stream(),
                                            &error) ||
                                        !mot_action_cuda_->append_x0_debug_tensor(
                                            &out.tensors,
                                            language_cuda_->cuda_stream(),
                                            &error)) {
                                        if (error.empty()) error = "Cosmos3 first action step debug tensor copy failed";
                                        break;
                                    }
                                }
                                if (all_action_steps_debug) {
                                    if (!action_bridge_cuda_->append_step_debug_tensors(
                                            &out.tensors,
                                            step,
                                            language_cuda_->cuda_stream(),
                                            &error) ||
                                        !mot_action_cuda_->append_step_x0_debug_tensor(
                                            &out.tensors,
                                            step,
                                            language_cuda_->cuda_stream(),
                                            &error)) {
                                        if (error.empty()) error = "Cosmos3 all action steps debug tensor copy failed";
                                        break;
                                    }
                                }
                                if (step > 0 &&
                                    !time_part(DT_UNIPC, [&]() {
                                        return mot_action_cuda_->unipc_corrector(
                                                   step,
                                                   previous_order,
                                                   sigmas,
                                                   language_cuda_->cuda_stream(),
                                                   &error);
                                    })) {
                                    if (error.empty()) error = "Cosmos3 UniPC corrector failed";
                                    break;
                                }
                                if (all_action_steps_debug && step > 0 &&
                                    !mot_action_cuda_->append_step_latent_debug_tensor(
                                        &out.tensors,
                                        step,
                                        "after_corrector",
                                        language_cuda_->cuda_stream(),
                                        &error)) {
                                    if (error.empty()) error = "Cosmos3 action after-corrector debug tensor copy failed";
                                    break;
                                }
                                if (step > 0 &&
                                    !time_part(DT_UNIPC, [&]() {
                                        return mot_action_cuda_->restore_condition_latents(
                                                   language_cuda_->cuda_stream(),
                                                   &error);
                                    })) {
                                    if (error.empty()) error = "Cosmos3 UniPC condition restore after corrector failed";
                                    break;
                                }
                                if (!time_part(DT_UNIPC, [&]() {
                                        return mot_action_cuda_->shift_unipc_history(
                                                   language_cuda_->cuda_stream(),
                                                   &error);
                                    })) {
                                    if (error.empty()) error = "Cosmos3 UniPC history shift failed";
                                    break;
                                }
                                int this_order = std::min(2, num_denoise_steps - step);
                                this_order = std::min(this_order, lower_order_nums + 1);
                                previous_order = this_order;
                                if (!time_part(DT_UNIPC, [&]() {
                                        return mot_action_cuda_->save_last_sample(
                                                   language_cuda_->cuda_stream(),
                                                   &error);
                                    })) {
                                    if (error.empty()) error = "Cosmos3 UniPC sample save failed";
                                    break;
                                }
                                if (!time_part(DT_UNIPC, [&]() {
                                        return mot_action_cuda_->unipc_predictor(
                                                   step,
                                                   this_order,
                                                   sigmas,
                                                   language_cuda_->cuda_stream(),
                                                   &error);
                                    })) {
                                    if (error.empty()) error = "Cosmos3 UniPC predictor failed";
                                    break;
                                }
                                if (all_action_steps_debug &&
                                    !mot_action_cuda_->append_step_latent_debug_tensor(
                                        &out.tensors,
                                        step,
                                        "post_latent",
                                        language_cuda_->cuda_stream(),
                                        &error)) {
                                    if (error.empty()) error = "Cosmos3 action post-latent debug tensor copy failed";
                                    break;
                                }
                                if (!time_part(DT_UNIPC, [&]() {
                                        return mot_action_cuda_->restore_condition_latents(
                                                   language_cuda_->cuda_stream(),
                                                   &error);
                                    })) {
                                    if (error.empty()) error = "Cosmos3 UniPC condition restore after predictor failed";
                                    break;
                                }
                                if (lower_order_nums < 2) ++lower_order_nums;
                            }
                            if (error.empty() &&
                                !time_part(DT_FINAL_COPY, [&]() {
                                    return mot_action_cuda_->copy_latent_action_output(
                                               &out.action,
                                               language_cuda_->cuda_stream(),
                                               &error);
                                })) {
                                if (error.empty()) error = "Cosmos3 MoT final action latent copy failed";
                            }
                            if (error.empty() && all_action_steps_debug) {
                                append_host_f32_tensor(&out.tensors,
                                                       "cosmos3.debug.action.pre_invert",
                                                       out.action.data(),
                                                       kActionSteps,
                                                       kActionDim);
                            }
                            if (error.empty() && invert_gripper) {
                                for (int t = 0; t < kActionSteps; ++t) {
                                    const size_t idx = static_cast<size_t>(t) * kActionDim + (kActionDim - 1);
                                    if (idx < out.action.size()) out.action[idx] = 1.0f - out.action[idx];
                                }
                            }
                            stats.ms_denoise = std::chrono::duration<float, std::milli>(clock::now() - t_denoise0).count();
                            if (denoise_timing) {
                                append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.denoise.cond_pack_ms", denoise_detail_ms[DT_COND_PACK]);
                                append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.denoise.cond_gen_ms", denoise_detail_ms[DT_COND_GEN]);
                                append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.denoise.cond_decode_ms", denoise_detail_ms[DT_COND_DECODE]);
                                append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.denoise.uncond_pack_ms", denoise_detail_ms[DT_UNCOND_PACK]);
                                append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.denoise.uncond_gen_ms", denoise_detail_ms[DT_UNCOND_GEN]);
                                append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.denoise.uncond_decode_ms", denoise_detail_ms[DT_UNCOND_DECODE]);
                                append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.denoise.cfg_ms", denoise_detail_ms[DT_CFG]);
                                append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.denoise.x0_ms", denoise_detail_ms[DT_X0]);
                                append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.denoise.unipc_ms", denoise_detail_ms[DT_UNIPC]);
                                append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.denoise.final_copy_ms", denoise_detail_ms[DT_FINAL_COPY]);
                            }
                            if (denoise_timing || denoise_profile) {
                                mot_gen_cuda_->append_timing_tensors(&out.tensors);
                            }
                        }
                    }
                }
            }
#else
            error = "Cosmos3 direct visual CUDA runtime requires VLA_COSMOS3_CUDA_KERNELS";
#endif
        }

        const auto t1 = clock::now();
        stats.ms_total = std::chrono::duration<float, std::milli>(t1 - t0).count();
        stats.ms_inference = stats.ms_prefill + stats.ms_denoise;
        if (!error.empty()) {
            out.error = error;
            return out;
        }
        if (!visual_executed) {
            out.error = "Cosmos3 direct visual CUDA runtime did not execute";
        } else {
            append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.total_ms", stats.ms_total);
            append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.vision_ms", stats.ms_vision);
            append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.prefill_ms", stats.ms_prefill);
            append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.denoise_ms", stats.ms_denoise);
            append_scalar_timing_tensor(&out.tensors, "cosmos3.debug.timing.inference_ms", stats.ms_inference);
        }
        return out;
    }

    std::unique_ptr<GgufGpuResidentModel> weights_;
    struct VisualCudaDeleter {
        void operator()(cosmos3_visual_cuda_ctx * ctx) const {
#ifdef VLA_COSMOS3_CUDA_KERNELS
            cosmos3_visual_cuda_free(ctx);
#else
            (void) ctx;
#endif
        }
    };
    std::unique_ptr<cosmos3_visual_cuda_ctx, VisualCudaDeleter> visual_cuda_;
    struct LanguageCudaDeleter {
        void operator()(LanguageCudaRuntime * ctx) const {
            delete ctx;
        }
    };
    std::unique_ptr<LanguageCudaRuntime, LanguageCudaDeleter> language_cuda_;
    struct ActionBridgeCudaDeleter {
        void operator()(ActionBridgeCudaRuntime * ctx) const {
            delete ctx;
        }
    };
    std::unique_ptr<ActionBridgeCudaRuntime, ActionBridgeCudaDeleter> action_bridge_cuda_;
    struct MotActionCudaDeleter {
        void operator()(MotActionCudaRuntime * ctx) const {
            delete ctx;
        }
    };
    std::unique_ptr<MotActionCudaRuntime, MotActionCudaDeleter> mot_action_cuda_;
    struct MotGenCudaDeleter {
        void operator()(MotGenCudaRuntime * ctx) const {
            delete ctx;
        }
    };
    std::unique_ptr<MotGenCudaRuntime, MotGenCudaDeleter> mot_gen_cuda_;
    struct WanVaeCudaDeleter {
        void operator()(WanVaeCudaRuntime * ctx) const {
            delete ctx;
        }
    };
    std::unique_ptr<WanVaeCudaRuntime, WanVaeCudaDeleter> wan_vae_cuda_;
};

} // namespace

std::unique_ptr<ModelArchBase> cosmos3_create(const std::string & mmproj_path,
                                              const std::string & ckpt_path,
                                              const std::string &) {
    if (!mmproj_path.empty()) {
        std::fprintf(stderr, "vla(cosmos3): new path expects one self-contained Cosmos3 GGUF; mmproj is unsupported\n");
        return nullptr;
    }

    auto weights = std::make_unique<GgufGpuResidentModel>();
    if (!weights->load(ckpt_path)) return nullptr;
#ifdef VLA_COSMOS3_CUDA_KERNELS
    WanVaeEncoderWeightsGpu vae_encoder_weights;
    std::string vae_encoder_bind_error;
    if (!bind_wan22_vae_encoder_cuda_weights(*weights, &vae_encoder_weights, &vae_encoder_bind_error)) {
        std::fprintf(stderr, "vla(cosmos3): Wan2.2 VAE encoder weight binding failed: %s\n",
                     vae_encoder_bind_error.c_str());
        return nullptr;
    }
    if (vae_encoder_weights.available) {
        std::string vae_readiness_error;
        if (!cosmos3_wan_vae_reference_readiness_check(*weights, &vae_readiness_error)) {
            std::fprintf(stderr, "vla(cosmos3): Wan2.2 VAE encoder readiness check failed: %s\n",
                         vae_readiness_error.c_str());
            return nullptr;
        }
        std::fprintf(stderr,
                     "vla(cosmos3): Wan2.2 VAE encoder tensors detected=%d readiness=ok (condition-latent kernel not wired yet)\n",
                     vae_encoder_weights.encoder_tensor_count);
    } else {
        std::fprintf(stderr,
                     "vla(cosmos3): Wan2.2 VAE encoder tensors not present; using current MoT vision condition fallback\n");
    }
#endif
    cosmos3_visual_cuda_ctx * visual_cuda = nullptr;
    LanguageCudaRuntime * language_cuda = nullptr;
    ActionBridgeCudaRuntime * action_bridge_cuda = nullptr;
    MotActionCudaRuntime * mot_action_cuda = nullptr;
    MotGenCudaRuntime * mot_gen_cuda = nullptr;
    WanVaeCudaRuntime * wan_vae_cuda = nullptr;
#ifdef VLA_COSMOS3_CUDA_KERNELS
    cosmos3_visual_cuda_config visual_cfg{};
    visual_cfg.frames = kVisualFrames;
    visual_cfg.temporal_patch = kVisualTemporalPatch;
    visual_cfg.patch = kVisualPatch;
    visual_cfg.merge = kVisualMerge;
    visual_cfg.grid_t = kVisualGridT;
    visual_cfg.grid_h = kVisualGridH;
    visual_cfg.grid_w = kVisualGridW;
    visual_cfg.patch_rows = kVisualPatchRows;
    visual_cfg.patch_dim = kVisualPatchDim;
    visual_cfg.hidden = kVisualHidden;
    visual_cfg.heads = kVisualHeads;
    visual_cfg.head_dim = kVisualHeadDim;
    visual_cfg.blocks = kVisualBlocks;
    visual_cfg.intermediate = kVisualIntermediate;
    visual_cfg.output_hidden = kLanguageHidden;
    visual_cuda = cosmos3_visual_cuda_init(&visual_cfg);
    if (!visual_cuda) {
        std::fprintf(stderr, "vla(cosmos3): cosmos3_visual_cuda_init failed\n");
        return nullptr;
    }
    std::string visual_bind_error;
    if (!bind_visual_cuda_runtime_weights(visual_cuda, *weights, &visual_bind_error)) {
        std::fprintf(stderr, "vla(cosmos3): visual CUDA weight binding failed: %s\n",
                     visual_bind_error.c_str());
        cosmos3_visual_cuda_free(visual_cuda);
        return nullptr;
    }
    language_cuda = new LanguageCudaRuntime();
    std::string language_bind_error;
    if (!language_cuda->init(*weights, &language_bind_error)) {
        std::fprintf(stderr, "vla(cosmos3): language CUDA runtime init failed: %s\n",
                     language_bind_error.c_str());
        delete language_cuda;
        cosmos3_visual_cuda_free(visual_cuda);
        return nullptr;
    }
    action_bridge_cuda = new ActionBridgeCudaRuntime();
    std::string action_bridge_bind_error;
    if (!action_bridge_cuda->init(*weights, &action_bridge_bind_error)) {
        std::fprintf(stderr, "vla(cosmos3): action bridge CUDA runtime init failed: %s\n",
                     action_bridge_bind_error.c_str());
        delete action_bridge_cuda;
        delete language_cuda;
        cosmos3_visual_cuda_free(visual_cuda);
        return nullptr;
    }
    mot_action_cuda = new MotActionCudaRuntime();
    std::string mot_action_bind_error;
    if (!mot_action_cuda->init(*weights, &mot_action_bind_error)) {
        std::fprintf(stderr, "vla(cosmos3): MoT action CUDA runtime init failed: %s\n",
                     mot_action_bind_error.c_str());
        delete mot_action_cuda;
        delete action_bridge_cuda;
        delete language_cuda;
        cosmos3_visual_cuda_free(visual_cuda);
        return nullptr;
    }
    mot_gen_cuda = new MotGenCudaRuntime();
    std::string mot_gen_bind_error;
    if (!mot_gen_cuda->init(*weights, &mot_gen_bind_error)) {
        std::fprintf(stderr, "vla(cosmos3): MoT gen CUDA runtime init failed: %s\n",
                     mot_gen_bind_error.c_str());
        delete mot_gen_cuda;
        delete mot_action_cuda;
        delete action_bridge_cuda;
        delete language_cuda;
        cosmos3_visual_cuda_free(visual_cuda);
        return nullptr;
    }
    wan_vae_cuda = new WanVaeCudaRuntime();
    std::string wan_vae_bind_error;
    if (!wan_vae_cuda->init(*weights, vae_encoder_weights, &wan_vae_bind_error)) {
        std::fprintf(stderr, "vla(cosmos3): Wan VAE CUDA runtime init failed: %s\n",
                     wan_vae_bind_error.c_str());
        delete wan_vae_cuda;
        delete mot_gen_cuda;
        delete mot_action_cuda;
        delete action_bridge_cuda;
        delete language_cuda;
        cosmos3_visual_cuda_free(visual_cuda);
        return nullptr;
    }
#endif
#ifdef VLA_COSMOS3_CUDA_KERNELS
    std::fprintf(stderr, "vla(cosmos3): %s\n", cosmos3_qwen_visual_kernel_status());
    std::fprintf(stderr, "vla(cosmos3): %s\n", cosmos3_qwen_language_kernel_status());
    std::fprintf(stderr, "vla(cosmos3): %s\n", cosmos3_w8a16_kernel_status());
    std::fprintf(stderr, "vla(cosmos3): %s\n", cosmos3_action_bridge_kernel_status());
    std::fprintf(stderr, "vla(cosmos3): %s\n", cosmos3_mot_action_kernel_status());
    std::fprintf(stderr, "vla(cosmos3): %s\n", cosmos3_wan_vae_kernel_status());
#endif
    std::fprintf(stderr,
                 "vla(cosmos3): cosmos3-new-create-ok full_gguf_gpu_resident=yes visual_runtime=direct_cuda_ctx language_runtime=direct_cuda_ctx action_bridge_runtime=direct_cuda_ctx mot_action_runtime=direct_cuda_ctx mot_gen_runtime=direct_cuda_ctx wan_vae_runtime=direct_cuda_ctx visual_scheduler=disabled\n");
    return std::make_unique<Cosmos3ModelArch>(std::move(weights),
                                              visual_cuda,
                                              language_cuda,
                                              action_bridge_cuda,
                                              mot_action_cuda,
                                              mot_gen_cuda,
                                              wan_vae_cuda);
}

} // namespace vla
