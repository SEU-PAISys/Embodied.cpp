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
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "gguf.h"

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
#include "kernels/lingbot/lingbot_flex_attn_cuda.h"
#include "kernels/lingbot/lingbot_vae_cuda.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <random>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <vector>
#include <numeric>
#include <map>
#include <sys/stat.h>
#include <sys/types.h>

namespace vla {

namespace {

struct LingBotScopedTimer {
    const char * name = nullptr;
    std::chrono::steady_clock::time_point t0;

    explicit LingBotScopedTimer(const char * name_) : name(name_), t0(std::chrono::steady_clock::now()) {}

    ~LingBotScopedTimer() {
        if (!std::getenv("VLA_LINGBOT_TIMING")) return;
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("vla(lingbot_va): timing %s %.3fms\n", name ? name : "unknown", ms);
    }
};

using LingBotClock = std::chrono::steady_clock;

bool lingbot_timing_enabled() {
    return std::getenv("VLA_LINGBOT_TIMING") != nullptr ||
           std::getenv("VLA_LINGBOT_CUDA_SELF_ATTN_TIMING") != nullptr;
}

double lingbot_elapsed_ms(LingBotClock::time_point t0, LingBotClock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int lingbot_cpu_threads() {
    int threads = 4;
    if (const char * env = std::getenv("VLA_LINGBOT_CPU_THREADS")) {
        char * end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end && *end == '\0' && v > 0) {
            threads = (int) std::min<long>(v, 256);
        } else {
            std::fprintf(stderr, "vla(lingbot_va): ignoring invalid VLA_LINGBOT_CPU_THREADS='%s'\n", env);
        }
    }
    return threads;
}

bool lingbot_env_enabled(const char * name) {
    const char * v = std::getenv(name);
    return v && std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0 && std::strcmp(v, "FALSE") != 0;
}

bool lingbot_env_disabled(const char * name) {
    return lingbot_env_enabled(name);
}

uint64_t lingbot_fnv1a_update(uint64_t h, const void * data, size_t bytes) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t lingbot_fnv1a_string(uint64_t h, const std::string & s) {
    return lingbot_fnv1a_update(h, s.data(), s.size());
}

bool lingbot_ensure_dir_recursive(const std::string & path) {
    if (path.empty()) return false;
    for (size_t pos = 0; pos < path.size();) {
        const size_t slash = path.find('/', pos);
        const size_t end = slash == std::string::npos ? path.size() : slash;
        std::string cur = path.substr(0, end);
        while (cur.size() > 1 && cur.back() == '/') cur.pop_back();
        if (!cur.empty() && cur != "/") {
            if (::mkdir(cur.c_str(), 0775) != 0 && errno != EEXIST) {
                std::fprintf(stderr, "vla(lingbot_va): failed to create dir %s: errno=%d\n",
                             cur.c_str(), errno);
                return false;
            }
        }
        if (slash == std::string::npos) break;
        pos = slash + 1;
    }
    if (!path.empty() && path.back() == '/') {
        std::string cur = path;
        while (cur.size() > 1 && cur.back() == '/') cur.pop_back();
        if (::mkdir(cur.c_str(), 0775) != 0 && errno != EEXIST) {
            std::fprintf(stderr, "vla(lingbot_va): failed to create dir %s: errno=%d\n",
                         cur.c_str(), errno);
            return false;
        }
    }
    return true;
}

bool lingbot_predict_cuda_self_attn_enabled() {
    return true;
}

bool lingbot_runtime_kv_device_enabled() {
    return true;
}

bool lingbot_runtime_kv_legacy_clear_pred_enabled() {
    return lingbot_env_enabled("VLA_LINGBOT_RUNTIME_KV_LEGACY_CLEAR_PRED");
}

bool lingbot_official_fast_path_enabled() {
    return true;
}

std::string lingbot_text_disk_cache_dir() {
    if (lingbot_env_enabled("VLA_LINGBOT_TEXT_DISK_CACHE_DISABLE")) return {};
    if (const char * env = std::getenv("VLA_LINGBOT_TEXT_DISK_CACHE_DIR")) {
        return *env ? std::string(env) : std::string();
    }
    return lingbot_official_fast_path_enabled() ? std::string("lingbot_text_cache/umt5") : std::string();
}

bool lingbot_vae_time_downsample_legacy_cache_enabled() {
    return (lingbot_official_fast_path_enabled() ||
            lingbot_env_enabled("VLA_LINGBOT_VAE_TIME_DOWNSAMPLE_LEGACY_CACHE")) &&
           !lingbot_env_enabled("VLA_LINGBOT_VAE_TIME_DOWNSAMPLE_LEGACY_CACHE_DISABLE");
}

int64_t lingbot_env_i64(const char * name, int64_t fallback) {
    const char * v = std::getenv(name);
    if (!v || !*v) return fallback;
    char * end = nullptr;
    const long long parsed = std::strtoll(v, &end, 10);
    return (end && *end == '\0') ? (int64_t) parsed : fallback;
}

void lingbot_dump_f32_stage(const char * prefix, const std::string & label,
                            const std::vector<float> & data,
                            int W, int H, int T, int C) {
    const char * dump_dir = std::getenv("VLA_LINGBOT_VAE_DUMP_DIR");
    if (!dump_dir) return;
    const std::string base = std::string(dump_dir) + "/" + prefix + "_" + label;
    std::ofstream f32(base + ".f32", std::ios::binary);
    if (f32) {
        f32.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * sizeof(float)));
    }
    std::ofstream shape(base + ".shape.txt");
    if (shape) {
        shape << W << " " << H << " " << T << " " << C << "\n";
    }
}

void lingbot_dump_embed_stage(const char * mode,
                              int64_t seq,
                              const std::vector<float> & raw_input,
                              const std::vector<float> & time_raw,
                              const std::vector<float> & x,
                              const std::vector<float> & text,
                              const std::vector<float> & t_hidden,
                              const std::vector<float> & timestep_proj) {
    const char * dump_dir = std::getenv("VLA_LINGBOT_EMBED_DUMP_DIR");
    if (!dump_dir || !*dump_dir) return;
    static std::atomic<int> counter{0};
    const int idx = counter.fetch_add(1);
    auto env_int = [](const char * name, int fallback) {
        const char * v = std::getenv(name);
        if (!v || !*v) return fallback;
        char * end = nullptr;
        const long parsed = std::strtol(v, &end, 10);
        return (end && *end == '\0') ? (int) parsed : fallback;
    };
    const int min_idx = env_int("VLA_LINGBOT_EMBED_DUMP_MIN", -1);
    const int max_idx = env_int("VLA_LINGBOT_EMBED_DUMP_MAX", INT_MAX);
    if (idx < min_idx || idx > max_idx) return;
    const std::string base = std::string(dump_dir) + "/" + std::string(mode) + "_" + std::to_string(idx);
    auto write_shape = [&](const std::string & path, size_t total, int64_t d0) {
        std::ofstream shape(path);
        if (shape) shape << d0 << " " << (long long) (d0 > 0 ? (long long) total / d0 : 0) << "\n";
    };
    std::ofstream f_raw(base + "_raw_input.f32", std::ios::binary);
    if (f_raw) f_raw.write(reinterpret_cast<const char *>(raw_input.data()), (std::streamsize) (raw_input.size() * sizeof(float)));
    write_shape(base + "_raw_input.shape.txt", raw_input.size(), seq);
    std::ofstream f_time_raw(base + "_time_raw.f32", std::ios::binary);
    if (f_time_raw) f_time_raw.write(reinterpret_cast<const char *>(time_raw.data()), (std::streamsize) (time_raw.size() * sizeof(float)));
    write_shape(base + "_time_raw.shape.txt", time_raw.size(), seq);
    std::ofstream f_x(base + "_x.f32", std::ios::binary);
    if (f_x) f_x.write(reinterpret_cast<const char *>(x.data()), (std::streamsize) (x.size() * sizeof(float)));
    write_shape(base + "_x.shape.txt", x.size(), seq);
    std::ofstream f_text(base + "_text.f32", std::ios::binary);
    if (f_text) f_text.write(reinterpret_cast<const char *>(text.data()), (std::streamsize) (text.size() * sizeof(float)));
    write_shape(base + "_text.shape.txt", text.size(), seq);
    std::ofstream f_t_hidden(base + "_t_hidden.f32", std::ios::binary);
    if (f_t_hidden) f_t_hidden.write(reinterpret_cast<const char *>(t_hidden.data()), (std::streamsize) (t_hidden.size() * sizeof(float)));
    write_shape(base + "_t_hidden.shape.txt", t_hidden.size(), seq);
    std::ofstream f_timestep_proj(base + "_timestep_proj.f32", std::ios::binary);
    if (f_timestep_proj) f_timestep_proj.write(reinterpret_cast<const char *>(timestep_proj.data()), (std::streamsize) (timestep_proj.size() * sizeof(float)));
    write_shape(base + "_timestep_proj.shape.txt", timestep_proj.size(), seq);
}

struct gguf_reader {
    gguf_context * gctx     = nullptr;
    ggml_context * meta_ctx = nullptr;
    FILE *         fp       = nullptr;
    size_t         data_off = 0;

    bool open(const std::string & path) {
        gguf_init_params p{};
        p.no_alloc = true;
        p.ctx      = &meta_ctx;
        gctx = gguf_init_from_file(path.c_str(), p);
        if (!gctx) {
            std::fprintf(stderr, "vla(lingbot_va): gguf_init_from_file failed for %s\n", path.c_str());
            return false;
        }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "vla(lingbot_va): fopen failed for %s\n", path.c_str());
            gguf_free(gctx); gctx = nullptr;
            ggml_free(meta_ctx); meta_ctx = nullptr;
            return false;
        }
        data_off = gguf_get_data_offset(gctx);
        return true;
    }

    ~gguf_reader() {
        if (fp)       std::fclose(fp);
        if (gctx)     gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
    }

    gguf_reader() = default;
    gguf_reader(const gguf_reader &) = delete;
    gguf_reader & operator=(const gguf_reader &) = delete;

    bool has(const char * key) const {
        return gguf_find_key(gctx, key) >= 0;
    }

    uint32_t u32(const char * key) const {
        return gguf_get_val_u32(gctx, gguf_find_key(gctx, key));
    }

    std::string str(const char * key) const {
        const int64_t id = gguf_find_key(gctx, key);
        return id < 0 ? std::string() : std::string(gguf_get_val_str(gctx, id));
    }

    bool f32_array(const char * key, std::vector<float> & out) const {
        const int64_t id = gguf_find_key(gctx, key);
        if (id < 0) {
            std::fprintf(stderr, "vla(lingbot_va): missing GGUF array %s\n", key);
            return false;
        }
        if (gguf_get_arr_type(gctx, id) != GGUF_TYPE_FLOAT32) {
            std::fprintf(stderr, "vla(lingbot_va): GGUF array %s is not F32\n", key);
            return false;
        }
        const size_t n = gguf_get_arr_n(gctx, id);
        const float * p = static_cast<const float *>(gguf_get_arr_data(gctx, id));
        out.assign(p, p + n);
        return true;
    }

    const ggml_tensor * meta(const char * name) const {
        return ggml_get_tensor(meta_ctx, name);
    }

    ggml_type tensor_type(const char * name) const {
        const ggml_tensor * gt = meta(name);
        return gt ? gt->type : GGML_TYPE_COUNT;
    }

    bool read_raw(const char * name, void * buf, size_t expected_bytes) {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) {
            std::fprintf(stderr, "vla(lingbot_va): missing tensor %s\n", name);
            return false;
        }
        const size_t bytes = gguf_get_tensor_size(gctx, id);
        if (bytes != expected_bytes) {
            std::fprintf(stderr, "vla(lingbot_va): size mismatch for %s (%zu vs %zu)\n",
                         name, bytes, expected_bytes);
            return false;
        }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        if (std::fseek(fp, (long) off, SEEK_SET) != 0) {
            std::fprintf(stderr, "vla(lingbot_va): fseek failed for %s\n", name);
            return false;
        }
        return std::fread(buf, 1, bytes, fp) == bytes;
    }

    bool read_bf16_rows_to_f32(
            const char * name,
            const std::vector<int32_t> & row_ids,
            int64_t row_size,
            int64_t row_count,
            float * dst) {
        const int64_t id = gguf_find_tensor(gctx, name);
        const ggml_tensor * gt = meta(name);
        if (id < 0 || !gt) {
            std::fprintf(stderr, "vla(lingbot_va): missing tensor %s\n", name);
            return false;
        }
        if (gt->type != GGML_TYPE_BF16) {
            std::fprintf(stderr, "vla(lingbot_va): row read currently expects BF16 tensor: %s\n", name);
            return false;
        }
        const int64_t expected = row_size * row_count;
        if (ggml_nelements(gt) != expected) {
            std::fprintf(stderr, "vla(lingbot_va): row read element mismatch for %s (%lld vs %lld)\n",
                         name, (long long) ggml_nelements(gt), (long long) expected);
            return false;
        }
        const size_t tensor_off = data_off + gguf_get_tensor_offset(gctx, id);
        std::vector<ggml_bf16_t> row((size_t) row_size);
        for (size_t i = 0; i < row_ids.size(); ++i) {
            const int64_t rid = (int64_t) row_ids[i];
            if (rid < 0 || rid >= row_count) {
                std::fprintf(stderr, "vla(lingbot_va): token id out of range for %s: %lld / %lld\n",
                             name, (long long) rid, (long long) row_count);
                return false;
            }
            const size_t off = tensor_off + (size_t) rid * (size_t) row_size * sizeof(ggml_bf16_t);
            if (std::fseek(fp, (long) off, SEEK_SET) != 0) {
                std::fprintf(stderr, "vla(lingbot_va): fseek row failed for %s row=%lld\n",
                             name, (long long) rid);
                return false;
            }
            const size_t bytes = row.size() * sizeof(ggml_bf16_t);
            if (std::fread(row.data(), 1, bytes, fp) != bytes) {
                std::fprintf(stderr, "vla(lingbot_va): fread row failed for %s row=%lld\n",
                             name, (long long) rid);
                return false;
            }
            ggml_bf16_to_fp32_row(row.data(), dst + i * (size_t) row_size, row_size);
        }
        return true;
    }

    bool read_to_f32(const char * name, float * dst, int64_t expected_nelements) {
        const ggml_tensor * gt = meta(name);
        if (!gt) {
            std::fprintf(stderr, "vla(lingbot_va): missing tensor %s\n", name);
            return false;
        }
        const int64_t n = ggml_nelements(gt);
        if (n != expected_nelements) {
            std::fprintf(stderr, "vla(lingbot_va): f32 read element mismatch for %s (%lld vs %lld)\n",
                         name, (long long) n, (long long) expected_nelements);
            return false;
        }
        if (gt->type == GGML_TYPE_F32) {
            return read_raw(name, dst, (size_t) n * sizeof(float));
        }
        if (gt->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp((size_t) n);
            if (!read_raw(name, tmp.data(), tmp.size() * sizeof(ggml_bf16_t))) return false;
            ggml_bf16_to_fp32_row(tmp.data(), dst, n);
            return true;
        }
        std::fprintf(stderr, "vla(lingbot_va): unsupported tensor dtype %d for f32 read: %s\n",
                     (int) gt->type, name);
        return false;
    }
};

std::vector<std::string> transformer_tensor_names(int64_t n_layers) {
    std::vector<std::string> out = {
        "wvm.patch_embd_mlp.weight",
        "wvm.patch_embd_mlp.bias",
        "wvm.patch_embd_legacy.weight",
        "wvm.patch_embd_legacy.bias",
        "wvm.action_embd.weight",
        "wvm.action_embd.bias",
        "wvm.output_proj.weight",
        "wvm.output_proj.bias",
        "wvm.action_out.weight",
        "wvm.action_out.bias",
        "wvm.output_scale_shift",
    };
    for (const auto & pair : {
             std::pair<std::string, std::string>{"wvm.cond", "condition"},
             std::pair<std::string, std::string>{"wvm.action_cond", "action condition"},
         }) {
        const std::string & p = pair.first;
        (void) pair;
        out.push_back(p + ".text_l1.weight");
        out.push_back(p + ".text_l1.bias");
        out.push_back(p + ".text_l2.weight");
        out.push_back(p + ".text_l2.bias");
        out.push_back(p + ".time_l1.weight");
        out.push_back(p + ".time_l1.bias");
        out.push_back(p + ".time_l2.weight");
        out.push_back(p + ".time_l2.bias");
        out.push_back(p + ".time_proj.weight");
        out.push_back(p + ".time_proj.bias");
    }
    for (int64_t i = 0; i < n_layers; ++i) {
        const std::string p = "wvm.blk." + std::to_string(i);
        out.push_back(p + ".scale_shift");
        out.push_back(p + ".cross_norm.weight");
        out.push_back(p + ".cross_norm.bias");
        for (const auto & attn : {"self_attn", "cross_attn"}) {
            const std::string a = p + "." + attn;
            out.push_back(a + ".q.weight");
            out.push_back(a + ".q.bias");
            out.push_back(a + ".k.weight");
            out.push_back(a + ".k.bias");
            out.push_back(a + ".v.weight");
            out.push_back(a + ".v.bias");
            out.push_back(a + ".o.weight");
            out.push_back(a + ".o.bias");
            out.push_back(a + ".q_norm.weight");
            out.push_back(a + ".k_norm.weight");
        }
        out.push_back(p + ".ffn_up.weight");
        out.push_back(p + ".ffn_up.bias");
        out.push_back(p + ".ffn_down.weight");
        out.push_back(p + ".ffn_down.bias");
    }
    return out;
}

bool validate_transformer_tensors(const gguf_reader & g, int64_t n_layers) {
    const auto names = transformer_tensor_names(n_layers);
    size_t missing = 0;
    size_t bf16 = 0;
    size_t f32 = 0;
    size_t other = 0;
    uint64_t nbytes = 0;

    for (const std::string & name : names) {
        const ggml_tensor * t = g.meta(name.c_str());
        if (!t) {
            if (missing < 16) {
                std::fprintf(stderr, "vla(lingbot_va): missing tensor %s\n", name.c_str());
            }
            ++missing;
            continue;
        }
        nbytes += (uint64_t) ggml_nbytes(t);
        if (t->type == GGML_TYPE_BF16) ++bf16;
        else if (t->type == GGML_TYPE_F32) ++f32;
        else ++other;
    }

    if (missing) {
        std::fprintf(stderr, "vla(lingbot_va): transformer tensor validation failed, missing=%zu/%zu\n",
                     missing, names.size());
        return false;
    }

    std::printf("vla(lingbot_va): transformer tensor metadata ok: %zu tensors, %.2f GiB, BF16=%zu F32=%zu other=%zu\n",
                names.size(), nbytes / (1024.0 * 1024.0 * 1024.0), bf16, f32, other);
    return true;
}

std::vector<std::string> text_encoder_tensor_names(int64_t n_layers) {
    std::vector<std::string> out = {
        "text.token_embd.weight",
        "text.final_norm.weight",
    };
    for (int64_t i = 0; i < n_layers; ++i) {
        const std::string p = "text.blk." + std::to_string(i);
        out.push_back(p + ".attn_norm.weight");
        out.push_back(p + ".attn.q.weight");
        out.push_back(p + ".attn.k.weight");
        out.push_back(p + ".attn.v.weight");
        out.push_back(p + ".attn.o.weight");
        out.push_back(p + ".attn.rel_bias.weight");
        out.push_back(p + ".ffn_norm.weight");
        out.push_back(p + ".ffn.wi_0.weight");
        out.push_back(p + ".ffn.wi_1.weight");
        out.push_back(p + ".ffn.wo.weight");
    }
    return out;
}

bool validate_text_encoder_tensors(const gguf_reader & g) {
    if (!g.has("lingbot_va.text_encoder.layers") ||
        !g.has("lingbot_va.text_encoder.d_model") ||
        !g.has("lingbot_va.text_encoder.d_ff") ||
        !g.has("lingbot_va.text_encoder.d_kv") ||
        !g.has("lingbot_va.text_encoder.heads") ||
        !g.has("lingbot_va.text_encoder.vocab_size") ||
        !g.has("lingbot_va.text_encoder.written_tensor_count") ||
        !g.has("lingbot_va.text_encoder.written_scope") ||
        !g.has("lingbot_va.text_encoder.tensor_shapes")) {
        std::fprintf(stderr,
                     "vla(lingbot_va): text_encoder GGUF is missing required metadata\n");
        return false;
    }
    const uint32_t layers = g.u32("lingbot_va.text_encoder.layers");
    const uint32_t d_model = g.u32("lingbot_va.text_encoder.d_model");
    const uint32_t d_ff = g.u32("lingbot_va.text_encoder.d_ff");
    const uint32_t d_kv = g.u32("lingbot_va.text_encoder.d_kv");
    const uint32_t heads = g.u32("lingbot_va.text_encoder.heads");
    const uint32_t vocab = g.u32("lingbot_va.text_encoder.vocab_size");
    const uint32_t expected = g.u32("lingbot_va.text_encoder.written_tensor_count");
    const std::string scope = g.str("lingbot_va.text_encoder.written_scope");
    if (scope != "encoder") {
        std::fprintf(stderr, "vla(lingbot_va): unsupported text_encoder written_scope='%s'\n", scope.c_str());
        return false;
    }
    const auto names = text_encoder_tensor_names(layers);
    if (expected != names.size()) {
        std::fprintf(stderr,
                     "vla(lingbot_va): text_encoder written tensor count mismatch (%u vs %zu)\n",
                     expected, names.size());
        return false;
    }
    const int64_t shape_key = gguf_find_key(g.gctx, "lingbot_va.text_encoder.tensor_shapes");
    const size_t shape_count = shape_key >= 0 ? gguf_get_arr_n(g.gctx, shape_key) : 0;
    if (shape_count != expected) {
        std::fprintf(stderr,
                     "vla(lingbot_va): text_encoder shape metadata count mismatch (%zu vs %u)\n",
                     shape_count, expected);
        return false;
    }

    size_t missing = 0;
    size_t bf16 = 0;
    size_t f32 = 0;
    size_t other = 0;
    uint64_t nbytes = 0;
    for (const std::string & name : names) {
        const ggml_tensor * t = g.meta(name.c_str());
        if (!t) {
            if (missing < 16) {
                std::fprintf(stderr, "vla(lingbot_va): missing text_encoder tensor %s\n", name.c_str());
            }
            ++missing;
            continue;
        }
        nbytes += (uint64_t) ggml_nbytes(t);
        if (t->type == GGML_TYPE_BF16) ++bf16;
        else if (t->type == GGML_TYPE_F32) ++f32;
        else ++other;
    }
    if (missing || other) {
        std::fprintf(stderr,
                     "vla(lingbot_va): text_encoder tensor validation failed missing=%zu other_dtype=%zu\n",
                     missing, other);
        return false;
    }
    const char * first = shape_count > 0 ? gguf_get_arr_str(g.gctx, shape_key, 0) : "";
    const char * last = shape_count > 0 ? gguf_get_arr_str(g.gctx, shape_key, shape_count - 1) : "";
    std::printf("vla(lingbot_va): text_encoder GGUF metadata ok: layers=%u d_model=%u d_ff=%u "
                "heads=%u d_kv=%u vocab=%u tensors=%zu %.2f GiB BF16=%zu F32=%zu\n",
                layers, d_model, d_ff, heads, d_kv, vocab, names.size(),
                nbytes / (1024.0 * 1024.0 * 1024.0), bf16, f32);
    std::printf("vla(lingbot_va): text_encoder shape sample: first=%s last=%s\n",
                first ? first : "", last ? last : "");
    return true;
}

bool validate_vae_encoder_tensors(const gguf_reader & g) {
    if (!g.has("lingbot_va.vae.written_tensor_count") ||
        !g.has("lingbot_va.vae.written_scope") ||
        !g.has("lingbot_va.vae.tensor_shapes")) {
        std::fprintf(stderr,
                     "vla(lingbot_va): VAE GGUF is missing required metadata "
                     "(written_tensor_count / written_scope / tensor_shapes)\n");
        return false;
    }
    const uint32_t expected = g.u32("lingbot_va.vae.written_tensor_count");
    const std::string scope = g.str("lingbot_va.vae.written_scope");
    if (scope != "encoder_quant_conv" && scope != "full") {
        std::fprintf(stderr, "vla(lingbot_va): unsupported VAE written_scope='%s'\n", scope.c_str());
        return false;
    }

    const int64_t shape_key = gguf_find_key(g.gctx, "lingbot_va.vae.tensor_shapes");
    const size_t shape_count = shape_key >= 0 ? gguf_get_arr_n(g.gctx, shape_key) : 0;
    if (shape_count != expected) {
        std::fprintf(stderr,
                     "vla(lingbot_va): VAE shape metadata count mismatch (%zu vs %u)\n",
                     shape_count, expected);
        return false;
    }

    uint32_t count = 0;
    uint32_t f32 = 0;
    uint32_t other = 0;
    uint64_t nbytes = 0;
    const int64_t n_tensors = gguf_get_n_tensors(g.gctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(g.gctx, i);
        if (!name || std::strncmp(name, "vae.", 4) != 0) continue;
        ++count;
        nbytes += (uint64_t) gguf_get_tensor_size(g.gctx, i);
        const ggml_type type = gguf_get_tensor_type(g.gctx, i);
        if (type == GGML_TYPE_F32) ++f32;
        else ++other;
    }
    if (count != expected) {
        std::fprintf(stderr,
                     "vla(lingbot_va): VAE tensor count mismatch (%u vs %u)\n",
                     count, expected);
        return false;
    }
    if (scope == "encoder_quant_conv" && expected != 86) {
        std::fprintf(stderr,
                     "vla(lingbot_va): VAE encoder_quant_conv expected 86 tensors, got %u\n",
                     expected);
        return false;
    }
    if (scope == "full" && expected != 196) {
        std::fprintf(stderr,
                     "vla(lingbot_va): VAE full scope expected 196 tensors, got %u\n",
                     expected);
        return false;
    }
    if (other != 0) {
        std::fprintf(stderr, "vla(lingbot_va): VAE GGUF has unsupported non-F32 tensors: %u\n", other);
        return false;
    }

    std::printf("vla(lingbot_va): VAE GGUF metadata ok: scope=%s tensors=%u shapes=%zu %.2f MiB F32=%u other=%u\n",
                scope.c_str(), count, shape_count, nbytes / (1024.0 * 1024.0), f32, other);
    if (shape_count > 0) {
        const char * first = gguf_get_arr_str(g.gctx, shape_key, 0);
        const char * last  = gguf_get_arr_str(g.gctx, shape_key, shape_count - 1);
        std::printf("vla(lingbot_va): VAE shape sample: first=%s last=%s\n",
                    first ? first : "", last ? last : "");
    }
    return true;
}

struct LingBotVaeEncoderContext {
    gguf_reader reader;
    int z_dim = 0;
    std::vector<float> latents_mean;
    std::vector<float> latents_std;
    std::mutex mu;
};

LingBotVaeEncoderContext * lingbot_get_vae_encoder_context(const std::string & path) {
    static std::mutex s_mu;
    static std::unordered_map<std::string, std::unique_ptr<LingBotVaeEncoderContext>> s_cache;
    std::lock_guard<std::mutex> lock(s_mu);
    auto it = s_cache.find(path);
    if (it != s_cache.end()) return it->second.get();

    auto ctx = std::make_unique<LingBotVaeEncoderContext>();
    if (!ctx->reader.open(path)) return nullptr;
    if (!validate_vae_encoder_tensors(ctx->reader)) return nullptr;
    ctx->z_dim = (int) ctx->reader.u32("lingbot_va.vae.z_dim");
    if (ctx->z_dim != 48) {
        std::fprintf(stderr, "vla(lingbot_va): expected VAE z_dim=48, got %d\n", ctx->z_dim);
        return nullptr;
    }
    if (!ctx->reader.f32_array("lingbot_va.vae.latents_mean", ctx->latents_mean) ||
        !ctx->reader.f32_array("lingbot_va.vae.latents_std", ctx->latents_std)) {
        return nullptr;
    }
    if ((int) ctx->latents_mean.size() != ctx->z_dim || (int) ctx->latents_std.size() != ctx->z_dim) {
        std::fprintf(stderr,
                     "vla(lingbot_va): VAE latent stat size mismatch mean=%zu std=%zu z_dim=%d\n",
                     ctx->latents_mean.size(), ctx->latents_std.size(), ctx->z_dim);
        return nullptr;
    }

    LingBotVaeEncoderContext * out = ctx.get();
    s_cache.emplace(path, std::move(ctx));
    std::printf("vla(lingbot_va): VAE encoder context cache store path=%s z_dim=%d\n",
                path.c_str(), out->z_dim);
    return out;
}

struct LingBotLinearW {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias   = nullptr;
};

struct LingBotConditionW {
    LingBotLinearW text_l1;
    LingBotLinearW text_l2;
    LingBotLinearW time_l1;
    LingBotLinearW time_l2;
    LingBotLinearW time_proj;
};

struct LingBotAttentionW {
    LingBotLinearW q;
    LingBotLinearW k;
    LingBotLinearW v;
    LingBotLinearW o;
    ggml_tensor * q_norm_weight = nullptr;
    ggml_tensor * k_norm_weight = nullptr;
};

struct LingBotBlockW {
    ggml_tensor * scale_shift       = nullptr;
    ggml_tensor * cross_norm_weight = nullptr;
    ggml_tensor * cross_norm_bias   = nullptr;
    LingBotAttentionW self_attn;
    LingBotAttentionW cross_attn;
    LingBotLinearW ffn_up;
    LingBotLinearW ffn_down;
};

struct LingBotRuntimeWeights;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
struct LingBotCudaSelfAttnQkvGraph;
struct LingBotCudaSelfAttnPostGraph;
struct LingBotCudaSelfAttnPostQkvGraph;
struct LingBotCudaSelfAttnBlockRunner;
struct LingBotCudaSelfAttnForwardRunner;
struct LingBotOfficialRuntimeKvDenoiseRunner;
struct LingBotOutputOneGraph;
struct LingBotEmbeddingStageGraph;
#endif

class LingBotVAModelArch final : public ModelArchBase {
public:
    LingBotVAModelArch(const std::string& ckpt_path, const gguf_reader & g,
                       const LingBotComponentPaths& components)
        : ModelArchBase(Arch::LINGBOT_VA),
          ckpt_path(ckpt_path),
          text_encoder_gguf(components.text_encoder_gguf),
          vae_encoder_gguf(components.vae_encoder_gguf) {
        n_layers     = g.u32("lingbot_va.transformer.layers");
        n_heads      = g.u32("lingbot_va.transformer.heads");
        head_dim     = g.u32("lingbot_va.transformer.head_dim");
        ffn_dim      = g.u32("lingbot_va.transformer.ffn_dim");
        in_channels  = g.u32("lingbot_va.transformer.in_channels");
        out_channels = g.u32("lingbot_va.transformer.out_channels");
        text_dim     = g.u32("lingbot_va.transformer.text_dim");
        action_dim   = g.u32("lingbot_va.transformer.action_dim");
        patch_t      = g.u32("lingbot_va.transformer.patch_t");
        patch_h      = g.u32("lingbot_va.transformer.patch_h");
        patch_w      = g.u32("lingbot_va.transformer.patch_w");
        attn_mode_config = g.str("lingbot_va.transformer.attn_mode_config");

        cfg.n_img           = 0;
        cfg.n_lang          = 512;
        cfg.n_state         = 0;
        cfg.n_prefix        = 0;
        cfg.n_suffix        = 16;
        cfg.n_full          = cfg.n_suffix;
        cfg.hidden          = n_heads * head_dim;
        cfg.expert_h        = 0;
        cfg.intermediate    = ffn_dim;
        cfg.expert_inter    = 0;
        cfg.n_q_heads       = n_heads;
        cfg.n_kv_heads      = n_heads;
        cfg.head_dim        = head_dim;
        cfg.q_full_dim      = cfg.n_q_heads * cfg.head_dim;
        cfg.kv_full_dim     = cfg.n_kv_heads * cfg.head_dim;
        cfg.n_layers        = n_layers;
        cfg.max_state_dim   = 7;
        cfg.max_action_dim  = action_dim;
        cfg.real_state_dim  = 7;
        cfg.real_action_dim = 7;
        cfg.num_steps       = (int) cfg.n_suffix;
        cfg.rms_eps         = 1e-6f;
        cfg.norm_eps        = 1e-6f;
        cfg.rope_n_dims     = (int) head_dim;
        cfg.rope_freq_base  = 10000.0f;
    }

    ~LingBotVAModelArch() override;

    std::vector<float> predict(const Inputs& in) override;

    bool text_cache_lookup(
            const std::string & path,
            int blocks,
            const std::vector<int32_t> & ids,
            std::vector<float> & out,
            int64_t & seq,
            int64_t & dim) {
        const std::string key = text_cache_key(path, blocks, ids);
        const auto it = text_cache.find(key);
        if (it == text_cache.end() || it->second.hidden.empty()) {
            if (!text_cache_disk_load(path, blocks, ids, out, seq, dim)) {
                return false;
            }
            text_cache_insert_memory(key, out, seq, dim);
            return true;
        }
        out = it->second.hidden;
        seq = it->second.seq;
        dim = it->second.dim;
        return true;
    }

    void text_cache_store(
            const std::string & path,
            int blocks,
            const std::vector<int32_t> & ids,
            const std::vector<float> & hidden,
            int64_t seq,
            int64_t dim) {
        const std::string key = text_cache_key(path, blocks, ids);
        text_cache_insert_memory(key, hidden, seq, dim);
        text_cache_disk_store(path, blocks, ids, hidden, seq, dim);
    }

    void text_cache_insert_memory(
            const std::string & key,
            const std::vector<float> & hidden,
            int64_t seq,
            int64_t dim) {
        constexpr size_t kMaxTextCacheEntries = 16;
        if (text_cache.size() >= kMaxTextCacheEntries) {
            text_cache.erase(text_cache.begin());
        }
        text_cache[key] = TextCacheEntry{hidden, seq, dim};
    }

    static uint64_t text_cache_hash(
            const std::string & path,
            int blocks,
            const std::vector<int32_t> & ids) {
        uint64_t h = 1469598103934665603ull;
        h = lingbot_fnv1a_string(h, path);
        h = lingbot_fnv1a_update(h, &blocks, sizeof(blocks));
        const int64_t n = (int64_t) ids.size();
        h = lingbot_fnv1a_update(h, &n, sizeof(n));
        if (!ids.empty()) h = lingbot_fnv1a_update(h, ids.data(), ids.size() * sizeof(ids[0]));
        return h;
    }

    static std::string text_cache_disk_path(
            const std::string & path,
            int blocks,
            const std::vector<int32_t> & ids) {
        const std::string dir = lingbot_text_disk_cache_dir();
        if (dir.empty()) return {};
        const uint64_t h = text_cache_hash(path, blocks, ids);
        char name[64];
        std::snprintf(name, sizeof(name), "%016llx.b%d.n%zu.lbut5",
                      (unsigned long long) h, blocks, ids.size());
        return dir + "/" + name;
    }

    struct TextDiskCacheHeader {
        char magic[8];
        uint32_t version;
        uint32_t id_count;
        int32_t blocks;
        int32_t reserved;
        int64_t seq;
        int64_t dim;
        uint64_t key_hash;
        uint64_t value_count;
    };

    static bool text_cache_disk_load(
            const std::string & path,
            int blocks,
            const std::vector<int32_t> & ids,
            std::vector<float> & out,
            int64_t & seq,
            int64_t & dim) {
        const std::string file = text_cache_disk_path(path, blocks, ids);
        if (file.empty()) return false;
        std::ifstream in(file, std::ios::binary | std::ios::ate);
        if (!in) return false;
        const std::streamoff end = in.tellg();
        if (end < (std::streamoff) sizeof(TextDiskCacheHeader)) return false;
        in.seekg(0, std::ios::beg);
        TextDiskCacheHeader hdr{};
        if (!in.read(reinterpret_cast<char *>(&hdr), sizeof(hdr))) return false;
        const char expected_magic[8] = {'L','B','U','T','5','C','1','\0'};
        if (std::memcmp(hdr.magic, expected_magic, sizeof(expected_magic)) != 0 ||
            hdr.version != 1 ||
            hdr.blocks != blocks ||
            hdr.id_count != ids.size() ||
            hdr.seq <= 0 ||
            hdr.dim <= 0 ||
            hdr.value_count != (uint64_t) hdr.seq * (uint64_t) hdr.dim ||
            hdr.key_hash != text_cache_hash(path, blocks, ids)) {
            return false;
        }
        const int64_t ids_bytes = (int64_t) hdr.id_count * (int64_t) sizeof(int32_t);
        const int64_t value_bytes = (int64_t) hdr.value_count * (int64_t) sizeof(float);
        const int64_t expected_bytes = (int64_t) sizeof(TextDiskCacheHeader) + ids_bytes + value_bytes;
        if ((int64_t) end != expected_bytes) return false;
        std::vector<int32_t> stored_ids((size_t) hdr.id_count);
        if (!stored_ids.empty() &&
            !in.read(reinterpret_cast<char *>(stored_ids.data()), ids_bytes)) {
            return false;
        }
        if (stored_ids != ids) return false;
        out.resize((size_t) hdr.value_count);
        if (!in.read(reinterpret_cast<char *>(out.data()), value_bytes)) {
            out.clear();
            return false;
        }
        seq = hdr.seq;
        dim = hdr.dim;
        std::printf("vla(lingbot_va): UMT5 text disk cache hit path=%s seq=%lld dim=%lld\n",
                    file.c_str(), (long long) seq, (long long) dim);
        return true;
    }

    static void text_cache_disk_store(
            const std::string & path,
            int blocks,
            const std::vector<int32_t> & ids,
            const std::vector<float> & hidden,
            int64_t seq,
            int64_t dim) {
        const std::string file = text_cache_disk_path(path, blocks, ids);
        const std::string dir = lingbot_text_disk_cache_dir();
        if (file.empty() || dir.empty() || hidden.empty() || seq <= 0 || dim <= 0 ||
            hidden.size() != (size_t) seq * (size_t) dim) {
            return;
        }
        if (!lingbot_ensure_dir_recursive(dir)) return;
        TextDiskCacheHeader hdr{};
        const char magic[8] = {'L','B','U','T','5','C','1','\0'};
        std::memcpy(hdr.magic, magic, sizeof(magic));
        hdr.version = 1;
        hdr.id_count = (uint32_t) ids.size();
        hdr.blocks = blocks;
        hdr.seq = seq;
        hdr.dim = dim;
        hdr.key_hash = text_cache_hash(path, blocks, ids);
        hdr.value_count = (uint64_t) hidden.size();
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::fprintf(stderr, "vla(lingbot_va): failed to open UMT5 text disk cache for write %s\n",
                         file.c_str());
            return;
        }
        out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
        if (!ids.empty()) {
            out.write(reinterpret_cast<const char *>(ids.data()),
                      (std::streamsize) ids.size() * (std::streamsize) sizeof(int32_t));
        }
        out.write(reinterpret_cast<const char *>(hidden.data()),
                  (std::streamsize) hidden.size() * (std::streamsize) sizeof(float));
        if (!out) {
            std::fprintf(stderr, "vla(lingbot_va): failed while writing UMT5 text disk cache %s\n",
                         file.c_str());
            return;
        }
        std::printf("vla(lingbot_va): UMT5 text disk cache store path=%s bytes=%.2f MiB\n",
                    file.c_str(),
                    (sizeof(hdr) + ids.size() * sizeof(int32_t) + hidden.size() * sizeof(float)) /
                    (1024.0 * 1024.0));
    }

    static std::string text_cache_key(
            const std::string & path,
            int blocks,
            const std::vector<int32_t> & ids) {
        std::string key = path;
        key += "|";
        key += std::to_string(blocks);
        key += "|";
        for (int32_t id : ids) {
            key += std::to_string(id);
            key += ",";
        }
        return key;
    }

    std::string ckpt_path;
    std::string text_encoder_gguf;
    std::string vae_encoder_gguf;
    int64_t n_layers = 0;
    int64_t n_heads = 0;
    int64_t head_dim = 0;
    int64_t ffn_dim = 0;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t text_dim = 0;
    int64_t action_dim = 0;
    int64_t patch_t = 0;
    int64_t patch_h = 0;
    int64_t patch_w = 0;
    std::string attn_mode_config;

    ggml_context *        ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf  = nullptr;
    std::vector<ggml_tensor *> weights;
    LingBotRuntimeWeights * full_transformer_weights = nullptr;

    LingBotLinearW patch_embd_mlp;
    LingBotLinearW patch_embd_legacy;
    LingBotLinearW action_embd;
    LingBotLinearW output_proj;
    LingBotLinearW action_out;
    ggml_tensor * output_scale_shift = nullptr;
    LingBotConditionW cond;
    LingBotConditionW action_cond;
    std::vector<LingBotBlockW> blocks;

    struct TextCacheEntry {
        std::vector<float> hidden;
        int64_t seq = 0;
        int64_t dim = 0;
    };
    std::unordered_map<std::string, TextCacheEntry> text_cache;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    mutable std::vector<std::unique_ptr<LingBotCudaSelfAttnQkvGraph>> cuda_self_attn_qkv_graphs;
    mutable std::vector<std::unique_ptr<LingBotCudaSelfAttnPostGraph>> cuda_self_attn_post_graphs;
    mutable std::vector<std::unique_ptr<LingBotCudaSelfAttnPostQkvGraph>> cuda_self_attn_post_qkv_graphs;
    mutable std::vector<std::unique_ptr<LingBotCudaSelfAttnBlockRunner>> cuda_self_attn_block_runners;
    mutable std::vector<std::unique_ptr<LingBotCudaSelfAttnForwardRunner>> cuda_self_attn_forward_runners;
    mutable std::vector<std::unique_ptr<LingBotOfficialRuntimeKvDenoiseRunner>> official_runtime_kv_denoise_runners;
    mutable std::vector<std::unique_ptr<LingBotOutputOneGraph>> output_one_graphs;
    mutable std::vector<std::unique_ptr<LingBotEmbeddingStageGraph>> embedding_stage_graphs;
#endif
};

ggml_tensor * make_weight_tensor(gguf_reader & g, LingBotVAModelArch & m, const std::string & name) {
    const ggml_tensor * gt = g.meta(name.c_str());
    if (!gt) {
        std::fprintf(stderr, "vla(lingbot_va): missing tensor %s\n", name.c_str());
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor(m.ctx_weights, gt->type, GGML_MAX_DIMS, gt->ne);
    ggml_set_name(t, name.c_str());
    m.weights.push_back(t);
    return t;
}

bool make_linear(gguf_reader & g, LingBotVAModelArch & m, const std::string & prefix, LingBotLinearW & out) {
    out.weight = make_weight_tensor(g, m, prefix + ".weight");
    out.bias   = make_weight_tensor(g, m, prefix + ".bias");
    return out.weight && out.bias;
}

bool make_condition(gguf_reader & g, LingBotVAModelArch & m, const std::string & prefix, LingBotConditionW & out) {
    return make_linear(g, m, prefix + ".text_l1",  out.text_l1)
        && make_linear(g, m, prefix + ".text_l2",  out.text_l2)
        && make_linear(g, m, prefix + ".time_l1",  out.time_l1)
        && make_linear(g, m, prefix + ".time_l2",  out.time_l2)
        && make_linear(g, m, prefix + ".time_proj", out.time_proj);
}

bool make_attention(gguf_reader & g, LingBotVAModelArch & m, const std::string & prefix, LingBotAttentionW & out) {
    out.q_norm_weight = make_weight_tensor(g, m, prefix + ".q_norm.weight");
    out.k_norm_weight = make_weight_tensor(g, m, prefix + ".k_norm.weight");
    return make_linear(g, m, prefix + ".q", out.q)
        && make_linear(g, m, prefix + ".k", out.k)
        && make_linear(g, m, prefix + ".v", out.v)
        && make_linear(g, m, prefix + ".o", out.o)
        && out.q_norm_weight
        && out.k_norm_weight;
}

bool make_transformer_weight_layout(gguf_reader & g, LingBotVAModelArch & m) {
    if (m.ctx_weights) return true;

    ggml_init_params wp = {
        size_t(16) * 1024 * 1024,
        nullptr,
        true,
    };
    m.ctx_weights = ggml_init(wp);
    if (!m.ctx_weights) {
        std::fprintf(stderr, "vla(lingbot_va): ggml_init(ctx_weights) failed\n");
        return false;
    }

    if (!make_linear(g, m, "wvm.patch_embd_mlp",    m.patch_embd_mlp))    return false;
    if (!make_linear(g, m, "wvm.patch_embd_legacy", m.patch_embd_legacy)) return false;
    if (!make_linear(g, m, "wvm.action_embd",       m.action_embd))       return false;
    if (!make_linear(g, m, "wvm.output_proj",       m.output_proj))       return false;
    if (!make_linear(g, m, "wvm.action_out",        m.action_out))        return false;
    m.output_scale_shift = make_weight_tensor(g, m, "wvm.output_scale_shift");
    if (!m.output_scale_shift) return false;

    if (!make_condition(g, m, "wvm.cond",        m.cond))        return false;
    if (!make_condition(g, m, "wvm.action_cond", m.action_cond)) return false;

    m.blocks.resize((size_t) m.n_layers);
    for (int64_t i = 0; i < m.n_layers; ++i) {
        const std::string prefix = "wvm.blk." + std::to_string(i);
        LingBotBlockW & b = m.blocks[(size_t) i];
        b.scale_shift       = make_weight_tensor(g, m, prefix + ".scale_shift");
        b.cross_norm_weight = make_weight_tensor(g, m, prefix + ".cross_norm.weight");
        b.cross_norm_bias   = make_weight_tensor(g, m, prefix + ".cross_norm.bias");
        if (!b.scale_shift || !b.cross_norm_weight || !b.cross_norm_bias) return false;
        if (!make_attention(g, m, prefix + ".self_attn",  b.self_attn))  return false;
        if (!make_attention(g, m, prefix + ".cross_attn", b.cross_attn)) return false;
        if (!make_linear(g, m, prefix + ".ffn_up",   b.ffn_up))          return false;
        if (!make_linear(g, m, prefix + ".ffn_down", b.ffn_down))        return false;
    }

    const auto expected = transformer_tensor_names(m.n_layers);
    if (m.weights.size() != expected.size()) {
        std::fprintf(stderr, "vla(lingbot_va): internal weight layout count mismatch (%zu vs %zu)\n",
                     m.weights.size(), expected.size());
        return false;
    }

    return true;
}

bool load_transformer_weights_cpu(gguf_reader & g, LingBotVAModelArch & m) {
    if (!make_transformer_weight_layout(g, m)) return false;

    uint64_t requested_bytes = 0;
    for (const ggml_tensor * t : m.weights) {
        requested_bytes += (uint64_t) ggml_nbytes(t);
    }

    m.weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(m.ctx_weights, ggml_backend_cpu_buffer_type());
    if (!m.weight_buf) {
        std::fprintf(stderr, "vla(lingbot_va): CPU weight buffer allocation failed (%.2f GiB requested)\n",
                     requested_bytes / (1024.0 * 1024.0 * 1024.0));
        return false;
    }

    uint64_t loaded_bytes = 0;
    std::vector<uint8_t> tmp;
    for (ggml_tensor * t : m.weights) {
        const size_t nbytes = ggml_nbytes(t);
        tmp.resize(nbytes);
        if (!g.read_raw(ggml_get_name(t), tmp.data(), nbytes)) {
            return false;
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
        loaded_bytes += (uint64_t) nbytes;
    }

    std::printf("vla(lingbot_va): transformer weights loaded to CPU buffer: %.2f GiB (%zu tensors)\n",
                loaded_bytes / (1024.0 * 1024.0 * 1024.0), m.weights.size());
    return true;
}

ggml_tensor * lin(ggml_context * C, const LingBotLinearW & w, ggml_tensor * x) {
    return ggml_add(C, ggml_mul_mat(C, w.weight, x), w.bias);
}

ggml_tensor * chunk_hidden(ggml_context * C, ggml_tensor * x, int64_t hidden, int64_t seq, int64_t chunk_id) {
    return ggml_view_2d(C, x, hidden, seq, x->nb[1], (size_t) chunk_id * (size_t) hidden * ggml_element_size(x));
}

ggml_tensor * adaln(ggml_context * C, ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale, float eps) {
    return ggml_add(C, ggml_mul(C, ggml_norm(C, x, eps), ggml_scale_bias(C, scale, 1.0f, 1.0f)), shift);
}

ggml_tensor * apply_wan_rope_shape(
        ggml_context * C,
        ggml_tensor * x,
        ggml_tensor * cos,
        ggml_tensor * sin,
        int64_t heads,
        int64_t seq) {
    const int64_t hd = x->ne[0];
    const int64_t pairs = hd / 2;
    ggml_tensor * xp = ggml_reshape_4d(C, x, 2, pairs, heads, seq);
    ggml_tensor * x0 = ggml_view_4d(C, xp, 1, pairs, heads, seq, xp->nb[1], xp->nb[2], xp->nb[3], 0);
    ggml_tensor * x1 = ggml_view_4d(C, xp, 1, pairs, heads, seq, xp->nb[1], xp->nb[2], xp->nb[3], xp->nb[0]);
    ggml_tensor * c = ggml_repeat_4d(C, cos, 1, pairs, heads, seq);
    ggml_tensor * s = ggml_repeat_4d(C, sin, 1, pairs, heads, seq);
    ggml_tensor * y0 = ggml_sub(C, ggml_mul(C, x0, c), ggml_mul(C, x1, s));
    ggml_tensor * y1 = ggml_add(C, ggml_mul(C, x1, c), ggml_mul(C, x0, s));
    return ggml_reshape_3d(C, ggml_concat(C, y0, y1, 0), hd, heads, seq);
}

struct LingBotAttentionTrace {
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * qh = nullptr;
    ggml_tensor * kh = nullptr;
    ggml_tensor * vh = nullptr;
    ggml_tensor * merged = nullptr;
};

ggml_tensor * build_attention_shape(
        ggml_context * C,
        const LingBotAttentionW & w,
        ggml_tensor * q_in,
        ggml_tensor * kv_in,
        ggml_tensor * rope_cos,
        ggml_tensor * rope_sin,
        const LingBotVAModelArch & m,
        int64_t seq_q,
        int64_t seq_k,
        LingBotAttentionTrace * trace = nullptr) {
    const int64_t hidden = m.cfg.hidden;
    const int64_t heads  = m.n_heads;
    const int64_t hd     = m.head_dim;

    ggml_tensor * q = ggml_mul(C, ggml_rms_norm(C, lin(C, w.q, q_in), 1e-6f), w.q_norm_weight);
    ggml_tensor * k = ggml_mul(C, ggml_rms_norm(C, lin(C, w.k, kv_in), 1e-6f), w.k_norm_weight);
    ggml_tensor * v = lin(C, w.v, kv_in);
    if (trace) {
        trace->q = q;
        trace->k = k;
        trace->v = v;
    }

    ggml_tensor * qh = ggml_reshape_3d(C, q, hd, heads, seq_q);
    ggml_tensor * kh = ggml_reshape_3d(C, k, hd, heads, seq_k);
    ggml_tensor * vh = ggml_reshape_3d(C, v, hd, heads, seq_k);

    if (rope_cos && rope_sin && seq_q == seq_k) {
        qh = apply_wan_rope_shape(C, qh, rope_cos, rope_sin, heads, seq_q);
        kh = apply_wan_rope_shape(C, kh, rope_cos, rope_sin, heads, seq_k);
    }
    qh = ggml_cont(C, qh);
    kh = ggml_cont(C, kh);
    if (trace) {
        trace->qh = qh;
        trace->kh = kh;
        trace->vh = vh;
    }

    ggml_tensor * Q = ggml_permute(C, qh, 0, 2, 1, 3);
    ggml_tensor * K = ggml_permute(C, kh, 0, 2, 1, 3);
    ggml_tensor * V = ggml_permute(C, vh, 0, 2, 1, 3);

    ggml_tensor * fa = ggml_flash_attn_ext(C, Q, K, V, nullptr,
                                           1.0f / std::sqrt((float) hd),
                                           0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_tensor * merged = ggml_reshape_2d(C, fa, hidden, seq_q);
    if (trace) {
        trace->merged = merged;
    }
    return lin(C, w.o, merged);
}

struct LingBotBlockTrace {
    ggml_tensor * n1 = nullptr;
    ggml_tensor * self_q = nullptr;
    ggml_tensor * self_k = nullptr;
    ggml_tensor * self_v = nullptr;
    ggml_tensor * self_qh = nullptr;
    ggml_tensor * self_kh = nullptr;
    ggml_tensor * self_merged = nullptr;
    ggml_tensor * self_attn = nullptr;
    ggml_tensor * post_self = nullptr;
    ggml_tensor * n2 = nullptr;
    ggml_tensor * cross_q = nullptr;
    ggml_tensor * cross_k = nullptr;
    ggml_tensor * cross_v = nullptr;
    ggml_tensor * cross_merged = nullptr;
    ggml_tensor * cross_attn = nullptr;
    ggml_tensor * post_cross = nullptr;
    ggml_tensor * n3 = nullptr;
    ggml_tensor * ff = nullptr;
};

ggml_tensor * build_block_shape(
        ggml_context * C,
        const LingBotBlockW & b,
        ggml_tensor * x,
        ggml_tensor * text,
        ggml_tensor * timestep_proj,
        ggml_tensor * rope_cos,
        ggml_tensor * rope_sin,
        const LingBotVAModelArch & m,
        int64_t seq,
        int64_t text_seq,
        LingBotBlockTrace * trace = nullptr) {
    const int64_t hidden = m.cfg.hidden;
    ggml_tensor * shift_msa   = ggml_add(C, chunk_hidden(C, timestep_proj, hidden, seq, 0),
                                         ggml_view_1d(C, b.scale_shift, hidden, 0));
    ggml_tensor * scale_msa   = ggml_add(C, chunk_hidden(C, timestep_proj, hidden, seq, 1),
                                         ggml_view_1d(C, b.scale_shift, hidden, (size_t) hidden * ggml_element_size(b.scale_shift)));
    ggml_tensor * gate_msa    = ggml_add(C, chunk_hidden(C, timestep_proj, hidden, seq, 2),
                                         ggml_view_1d(C, b.scale_shift, hidden, (size_t) 2 * (size_t) hidden * ggml_element_size(b.scale_shift)));
    ggml_tensor * c_shift_msa = ggml_add(C, chunk_hidden(C, timestep_proj, hidden, seq, 3),
                                         ggml_view_1d(C, b.scale_shift, hidden, (size_t) 3 * (size_t) hidden * ggml_element_size(b.scale_shift)));
    ggml_tensor * c_scale_msa = ggml_add(C, chunk_hidden(C, timestep_proj, hidden, seq, 4),
                                         ggml_view_1d(C, b.scale_shift, hidden, (size_t) 4 * (size_t) hidden * ggml_element_size(b.scale_shift)));
    ggml_tensor * c_gate_msa  = ggml_add(C, chunk_hidden(C, timestep_proj, hidden, seq, 5),
                                         ggml_view_1d(C, b.scale_shift, hidden, (size_t) 5 * (size_t) hidden * ggml_element_size(b.scale_shift)));

    ggml_tensor * n1 = adaln(C, x, shift_msa, scale_msa, 1e-6f);
    LingBotAttentionTrace self_trace;
    ggml_tensor * a1 = build_attention_shape(C, b.self_attn, n1, n1, rope_cos, rope_sin, m, seq, seq,
                                             trace ? &self_trace : nullptr);
    x = ggml_add(C, x, ggml_mul(C, a1, gate_msa));
    if (trace) {
        trace->n1 = n1;
        trace->self_q = self_trace.q;
        trace->self_k = self_trace.k;
        trace->self_v = self_trace.v;
        trace->self_qh = self_trace.qh;
        trace->self_kh = self_trace.kh;
        trace->self_merged = self_trace.merged;
        trace->self_attn = a1;
        trace->post_self = x;
    }

    ggml_tensor * n2 = ggml_add(C, ggml_mul(C, ggml_norm(C, x, 1e-6f), b.cross_norm_weight), b.cross_norm_bias);
    LingBotAttentionTrace cross_trace;
    ggml_tensor * a2 = build_attention_shape(C, b.cross_attn, n2, text, nullptr, nullptr, m, seq, text_seq,
                                             trace ? &cross_trace : nullptr);
    x = ggml_add(C, x, a2);
    if (trace) {
        trace->n2 = n2;
        trace->cross_q = cross_trace.q;
        trace->cross_k = cross_trace.k;
        trace->cross_v = cross_trace.v;
        trace->cross_merged = cross_trace.merged;
        trace->cross_attn = a2;
        trace->post_cross = x;
    }

    ggml_tensor * n3 = adaln(C, x, c_shift_msa, c_scale_msa, 1e-6f);
    ggml_tensor * ff = lin(C, b.ffn_down, ggml_gelu(C, lin(C, b.ffn_up, n3)));
    ggml_tensor * out = ggml_add(C, x, ggml_mul(C, ff, c_gate_msa));
    if (trace) {
        trace->n3 = n3;
        trace->ff = ff;
    }
    return out;
}



struct LingBotRuntimeWeights {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    bool is_cuda = false;
    bool owns_backend = true;
    std::vector<ggml_tensor *> tensors;

    LingBotLinearW patch_embd_mlp;
    LingBotLinearW action_embd;
    LingBotLinearW output_proj;
    LingBotLinearW action_out;
    ggml_tensor * output_scale_shift = nullptr;
    LingBotConditionW cond;
    LingBotConditionW action_cond;
    std::vector<LingBotBlockW> blocks;

    ~LingBotRuntimeWeights() {
        if (buf)     ggml_backend_buffer_free(buf);
        if (backend && owns_backend) ggml_backend_free(backend);
        if (ctx)     ggml_free(ctx);
    }
};

LingBotVAModelArch::~LingBotVAModelArch() {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    embedding_stage_graphs.clear();
    output_one_graphs.clear();
    official_runtime_kv_denoise_runners.clear();
    cuda_self_attn_forward_runners.clear();
    cuda_self_attn_block_runners.clear();
    cuda_self_attn_qkv_graphs.clear();
    cuda_self_attn_post_graphs.clear();
    cuda_self_attn_post_qkv_graphs.clear();
#endif
    delete full_transformer_weights;
    full_transformer_weights = nullptr;
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
}

bool lingbot_runtime_cuda_requested() {
    const char * backend = std::getenv("VLA_LINGBOT_RUNTIME_BACKEND");
    if (backend && (std::strcmp(backend, "cuda") == 0 ||
                    std::strcmp(backend, "CUDA") == 0 ||
                    std::strcmp(backend, "gpu") == 0 ||
                    std::strcmp(backend, "GPU") == 0)) {
        return true;
    }
    return std::getenv("VLA_LINGBOT_RUNTIME_CUDA") != nullptr ||
           std::getenv("VLA_LINGBOT_GPU_WEIGHTS") != nullptr;
}

std::string lingbot_runtime_backend_key() {
    return lingbot_runtime_cuda_requested() ? "cuda" : "cpu";
}

bool lingbot_runtime_backend_init(LingBotRuntimeWeights & sw, const char * label) {
    if (sw.backend) return true;
    const bool want_cuda = lingbot_runtime_cuda_requested();
#ifdef GGML_USE_CUDA
    if (want_cuda) {
        setenv("GGML_CUDA_GRAPH_EVICT_AFTER_MS", "300000", 0);
        static ggml_backend_t s_cuda_backend = nullptr;
        if (!s_cuda_backend) {
            s_cuda_backend = ggml_backend_cuda_init(0);
        }
        sw.backend = s_cuda_backend;
        if (sw.backend) {
            sw.is_cuda = true;
            sw.owns_backend = false;
            std::printf("vla(lingbot_va): runtime backend = CUDA (device 0, %s)\n", label);
            return true;
        }
        std::fprintf(stderr,
                     "vla(lingbot_va): runtime CUDA backend init failed for %s; falling back to CPU\n",
                     label);
        if (std::getenv("VLA_LINGBOT_REQUIRE_CUDA")) return false;
    }
#else
    if (want_cuda) {
        std::fprintf(stderr,
                     "vla(lingbot_va): runtime CUDA requested for %s but binary was not built with GGML_USE_CUDA\n",
                     label);
        if (std::getenv("VLA_LINGBOT_REQUIRE_CUDA")) return false;
    }
#endif
    sw.backend = ggml_backend_cpu_init();
    if (!sw.backend) {
        std::fprintf(stderr, "vla(lingbot_va): runtime CPU backend init failed for %s\n", label);
        return false;
    }
    ggml_backend_cpu_set_n_threads(sw.backend, 4);
    sw.is_cuda = false;
    std::printf("vla(lingbot_va): runtime backend = CPU (4 threads, %s)\n", label);
    return true;
}

ggml_backend_buffer_type_t lingbot_runtime_buffer_type(ggml_backend_t backend) {
    return backend ? ggml_backend_get_default_buffer_type(backend)
                   : ggml_backend_cpu_buffer_type();
}

ggml_gallocr_t lingbot_runtime_gallocr(ggml_backend_t backend) {
    return ggml_gallocr_new(lingbot_runtime_buffer_type(backend));
}

struct LingBotAuxGraphBackend {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    bool owns_backend = false;
    bool is_cuda = false;
};

bool lingbot_aux_ggml_graph_cuda_requested() {
    if (lingbot_env_disabled("VLA_LINGBOT_GGML_GRAPHS_CUDA_DISABLE")) return false;
    return lingbot_runtime_cuda_requested() ||
           lingbot_env_enabled("VLA_LINGBOT_GGML_GRAPHS_CUDA") ||
           lingbot_env_enabled("VLA_LINGBOT_ALL_GGML_GRAPHS_CUDA");
}

bool lingbot_vae_cuda_requested() {
    if (lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_DISABLE")) return false;
    return lingbot_env_enabled("VLA_LINGBOT_VAE_CUDA") ||
           lingbot_aux_ggml_graph_cuda_requested();
}

bool lingbot_vae_mid_attn_cuda_requested() {
    if (lingbot_env_disabled("VLA_LINGBOT_VAE_MID_ATTN_CUDA_DISABLE")) return false;
    return lingbot_env_enabled("VLA_LINGBOT_VAE_MID_ATTN_CUDA") ||
           lingbot_vae_cuda_requested();
}

bool lingbot_aux_graph_backend_init(LingBotAuxGraphBackend & gb, const char * label) {
    if (gb.backend) return true;
    const bool want_cuda = lingbot_aux_ggml_graph_cuda_requested();
#ifdef GGML_USE_CUDA
    if (want_cuda) {
        static ggml_backend_t s_aux_cuda_backend = nullptr;
        if (!s_aux_cuda_backend) {
            s_aux_cuda_backend = ggml_backend_cuda_init(0);
        }
        if (s_aux_cuda_backend) {
            gb.backend = s_aux_cuda_backend;
            gb.buft = ggml_backend_get_default_buffer_type(gb.backend);
            gb.owns_backend = false;
            gb.is_cuda = true;
            if (std::getenv("VLA_LINGBOT_TIMING")) {
                std::printf("vla(lingbot_va): aux ggml graph backend = CUDA (%s)\n", label ? label : "graph");
            }
            return true;
        }
        std::fprintf(stderr,
                     "vla(lingbot_va): aux CUDA backend init failed for %s; falling back to CPU\n",
                     label ? label : "graph");
        if (std::getenv("VLA_LINGBOT_REQUIRE_CUDA")) return false;
    }
#else
    if (want_cuda) {
        std::fprintf(stderr,
                     "vla(lingbot_va): aux CUDA graph requested for %s but binary has no GGML CUDA\n",
                     label ? label : "graph");
        if (std::getenv("VLA_LINGBOT_REQUIRE_CUDA")) return false;
    }
#endif
    gb.backend = ggml_backend_cpu_init();
    if (!gb.backend) {
        std::fprintf(stderr, "vla(lingbot_va): aux CPU backend init failed for %s\n",
                     label ? label : "graph");
        return false;
    }
    ggml_backend_cpu_set_n_threads(gb.backend, lingbot_cpu_threads());
    gb.buft = ggml_backend_cpu_buffer_type();
    gb.owns_backend = true;
    gb.is_cuda = false;
    return true;
}

void lingbot_aux_graph_backend_free(LingBotAuxGraphBackend & gb) {
    if (gb.owns_backend && gb.backend) {
        ggml_backend_free(gb.backend);
    }
    gb.backend = nullptr;
    gb.buft = nullptr;
    gb.owns_backend = false;
    gb.is_cuda = false;
}

ggml_tensor * runtime_tensor(gguf_reader & g, LingBotRuntimeWeights & sw, const std::string & name) {
    const ggml_tensor * gt = g.meta(name.c_str());
    if (!gt) {
        std::fprintf(stderr, "vla(lingbot_va): runtime missing tensor %s\n", name.c_str());
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor(sw.ctx, GGML_TYPE_F32, GGML_MAX_DIMS, gt->ne);
    ggml_set_name(t, name.c_str());
    sw.tensors.push_back(t);
    return t;
}

ggml_tensor * runtime_tensor_typed(
        gguf_reader & g,
        LingBotRuntimeWeights & sw,
        const std::string & name,
        ggml_type type) {
    const ggml_tensor * gt = g.meta(name.c_str());
    if (!gt) {
        std::fprintf(stderr, "vla(lingbot_va): runtime missing tensor %s\n", name.c_str());
        return nullptr;
    }
    if (ggml_is_quantized(type)) {
        const ggml_type_traits * traits = ggml_get_type_traits(type);
        if (!traits || gt->ne[0] % traits->blck_size != 0) {
            std::fprintf(stderr,
                         "vla(lingbot_va): tensor %s ne0=%lld is not compatible with quant type %s\n",
                         name.c_str(), (long long) gt->ne[0],
                         traits ? traits->type_name : "<unknown>");
            return nullptr;
        }
    }
    ggml_tensor * t = ggml_new_tensor(sw.ctx, type, GGML_MAX_DIMS, gt->ne);
    ggml_set_name(t, name.c_str());
    sw.tensors.push_back(t);
    return t;
}

bool runtime_linear(gguf_reader & g, LingBotRuntimeWeights & sw, const std::string & prefix, LingBotLinearW & out) {
    out.weight = runtime_tensor(g, sw, prefix + ".weight");
    out.bias   = runtime_tensor(g, sw, prefix + ".bias");
    return out.weight && out.bias;
}

bool runtime_linear_typed(
        gguf_reader & g,
        LingBotRuntimeWeights & sw,
        const std::string & prefix,
        LingBotLinearW & out,
        ggml_type weight_type) {
    out.weight = runtime_tensor_typed(g, sw, prefix + ".weight", weight_type);
    out.bias   = runtime_tensor(g, sw, prefix + ".bias");
    return out.weight && out.bias;
}

bool runtime_condition(gguf_reader & g, LingBotRuntimeWeights & sw, const std::string & prefix, LingBotConditionW & out) {
    return runtime_linear(g, sw, prefix + ".text_l1", out.text_l1)
        && runtime_linear(g, sw, prefix + ".text_l2", out.text_l2)
        && runtime_linear(g, sw, prefix + ".time_l1", out.time_l1)
        && runtime_linear(g, sw, prefix + ".time_l2", out.time_l2)
        && runtime_linear(g, sw, prefix + ".time_proj", out.time_proj);
}

bool runtime_attention(gguf_reader & g, LingBotRuntimeWeights & sw, const std::string & prefix, LingBotAttentionW & out) {
    out.q_norm_weight = runtime_tensor(g, sw, prefix + ".q_norm.weight");
    out.k_norm_weight = runtime_tensor(g, sw, prefix + ".k_norm.weight");
    return runtime_linear(g, sw, prefix + ".q", out.q)
        && runtime_linear(g, sw, prefix + ".k", out.k)
        && runtime_linear(g, sw, prefix + ".v", out.v)
        && runtime_linear(g, sw, prefix + ".o", out.o)
        && out.q_norm_weight
        && out.k_norm_weight;
}

bool runtime_attention_typed(
        gguf_reader & g,
        LingBotRuntimeWeights & sw,
        const std::string & prefix,
        LingBotAttentionW & out,
        ggml_type weight_type) {
    out.q_norm_weight = runtime_tensor(g, sw, prefix + ".q_norm.weight");
    out.k_norm_weight = runtime_tensor(g, sw, prefix + ".k_norm.weight");
    return runtime_linear_typed(g, sw, prefix + ".q", out.q, weight_type)
        && runtime_linear_typed(g, sw, prefix + ".k", out.k, weight_type)
        && runtime_linear_typed(g, sw, prefix + ".v", out.v, weight_type)
        && runtime_linear_typed(g, sw, prefix + ".o", out.o, weight_type)
        && out.q_norm_weight
        && out.k_norm_weight;
}



bool runtime_block(gguf_reader & g, LingBotRuntimeWeights & sw, int64_t index, LingBotBlockW & out) {
    const std::string p = "wvm.blk." + std::to_string(index);
    out.scale_shift       = runtime_tensor(g, sw, p + ".scale_shift");
    out.cross_norm_weight = runtime_tensor(g, sw, p + ".cross_norm.weight");
    out.cross_norm_bias   = runtime_tensor(g, sw, p + ".cross_norm.bias");
    if (!out.scale_shift || !out.cross_norm_weight || !out.cross_norm_bias) return false;
    if (!runtime_attention(g, sw, p + ".self_attn",  out.self_attn)) return false;
    if (!runtime_attention(g, sw, p + ".cross_attn", out.cross_attn)) return false;
    if (!runtime_linear(g, sw, p + ".ffn_up",   out.ffn_up)) return false;
    if (!runtime_linear(g, sw, p + ".ffn_down", out.ffn_down)) return false;
    return true;
}

bool runtime_block_typed(
        gguf_reader & g,
        LingBotRuntimeWeights & sw,
        int64_t index,
        LingBotBlockW & out,
        ggml_type weight_type) {
    const std::string p = "wvm.blk." + std::to_string(index);
    out.scale_shift       = runtime_tensor(g, sw, p + ".scale_shift");
    out.cross_norm_weight = runtime_tensor(g, sw, p + ".cross_norm.weight");
    out.cross_norm_bias   = runtime_tensor(g, sw, p + ".cross_norm.bias");
    if (!out.scale_shift || !out.cross_norm_weight || !out.cross_norm_bias) return false;
    if (!runtime_attention_typed(g, sw, p + ".self_attn",  out.self_attn,  weight_type)) return false;
    if (!runtime_attention_typed(g, sw, p + ".cross_attn", out.cross_attn, weight_type)) return false;
    if (!runtime_linear_typed(g, sw, p + ".ffn_up",   out.ffn_up,   weight_type)) return false;
    if (!runtime_linear_typed(g, sw, p + ".ffn_down", out.ffn_down, weight_type)) return false;
    return true;
}

uint64_t estimate_runtime_block_f32_bytes(gguf_reader & g, int64_t index) {
    const std::string p = "wvm.blk." + std::to_string(index);
    std::vector<std::string> names = {
        p + ".scale_shift",
        p + ".cross_norm.weight",
        p + ".cross_norm.bias",
        p + ".self_attn.q_norm.weight",
        p + ".self_attn.k_norm.weight",
        p + ".self_attn.q.weight",
        p + ".self_attn.q.bias",
        p + ".self_attn.k.weight",
        p + ".self_attn.k.bias",
        p + ".self_attn.v.weight",
        p + ".self_attn.v.bias",
        p + ".self_attn.o.weight",
        p + ".self_attn.o.bias",
        p + ".cross_attn.q_norm.weight",
        p + ".cross_attn.k_norm.weight",
        p + ".cross_attn.q.weight",
        p + ".cross_attn.q.bias",
        p + ".cross_attn.k.weight",
        p + ".cross_attn.k.bias",
        p + ".cross_attn.v.weight",
        p + ".cross_attn.v.bias",
        p + ".cross_attn.o.weight",
        p + ".cross_attn.o.bias",
        p + ".ffn_up.weight",
        p + ".ffn_up.bias",
        p + ".ffn_down.weight",
        p + ".ffn_down.bias",
    };
    uint64_t bytes = 0;
    for (const std::string & name : names) {
        const ggml_tensor * t = g.meta(name.c_str());
        if (!t) return 0;
        bytes += (uint64_t) ggml_nelements(t) * sizeof(float);
    }
    return bytes;
}

int64_t stream_window_size(gguf_reader & g, const LingBotVAModelArch & m, int64_t total_blocks) {
    const char * env = std::getenv("VLA_LINGBOT_BLOCK_WINDOW");
    if (!env || std::strcmp(env, "1") == 0) return 1;
    if (std::strcmp(env, "auto") != 0) {
        char * end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end && *end == '\0' && v > 0) {
            return std::min<int64_t>((int64_t) v, total_blocks);
        }
        std::fprintf(stderr,
                     "vla(lingbot_va): ignoring invalid VLA_LINGBOT_BLOCK_WINDOW='%s'; using 1\n",
                     env);
        return 1;
    }

    int64_t budget_mb = 4096;
    if (const char * b = std::getenv("VLA_LINGBOT_BLOCK_BUDGET_MB")) {
        char * end = nullptr;
        const long v = std::strtol(b, &end, 10);
        if (end && *end == '\0' && v > 0) budget_mb = (int64_t) v;
    }
    const uint64_t block_bytes = estimate_runtime_block_f32_bytes(g, 0);
    if (block_bytes == 0) return 1;
    const uint64_t reserve_bytes = 768ull * 1024ull * 1024ull;
    const uint64_t budget_bytes = (uint64_t) budget_mb * 1024ull * 1024ull;
    if (budget_bytes <= reserve_bytes + block_bytes) return 1;
    const uint64_t usable = budget_bytes - reserve_bytes;
    int64_t window = (int64_t) (usable / block_bytes);
    window = std::max<int64_t>(1, window);
    window = std::min<int64_t>(window, total_blocks);
    std::printf("vla(lingbot_va): auto block window=%lld (budget=%lld MiB, reserve=768 MiB, block_f32=%.2f MiB)\n",
                (long long) window, (long long) budget_mb, block_bytes / (1024.0 * 1024.0));
    return window;
}

bool allocate_and_load_runtime_weights(gguf_reader & g, LingBotRuntimeWeights & sw, const char * label) {
    if (!lingbot_runtime_backend_init(sw, label)) {
        return false;
    }

    sw.buf = ggml_backend_alloc_ctx_tensors_from_buft(sw.ctx, lingbot_runtime_buffer_type(sw.backend));
    if (!sw.buf) {
        std::fprintf(stderr, "vla(lingbot_va): runtime weight buffer allocation failed\n");
        return false;
    }

    uint64_t loaded = 0;
    std::vector<float> tmp;
    std::vector<uint8_t> qtmp;
    for (ggml_tensor * t : sw.tensors) {
        const char * name = ggml_get_name(t);
        const int64_t n = ggml_nelements(t);
        const ggml_type src_type = g.tensor_type(name);
        if (src_type == GGML_TYPE_COUNT) {
            std::fprintf(stderr, "vla(lingbot_va): missing tensor %s\n", name);
            return false;
        }

        if (src_type == t->type) {
            qtmp.assign(ggml_nbytes(t), 0);
            if (!g.read_raw(name, qtmp.data(), qtmp.size())) return false;
            ggml_backend_tensor_set(t, qtmp.data(), 0, qtmp.size());
        } else if (t->type == GGML_TYPE_F32) {
            tmp.resize((size_t) n);
            if (!g.read_to_f32(name, tmp.data(), n)) return false;
            ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size() * sizeof(float));
        } else if (ggml_is_quantized(t->type)) {
            tmp.resize((size_t) n);
            if (!g.read_to_f32(name, tmp.data(), n)) return false;
            qtmp.assign(ggml_nbytes(t), 0);
            const int64_t n_per_row = t->ne[0];
            const int64_t nrows = n / n_per_row;
            const size_t qbytes = ggml_quantize_chunk(t->type, tmp.data(), qtmp.data(), 0, nrows, n_per_row, nullptr);
            if (qbytes != qtmp.size()) {
                std::fprintf(stderr,
                             "vla(lingbot_va): quantized byte mismatch for %s (%zu vs %zu)\n",
                             name, qbytes, qtmp.size());
                return false;
            }
            ggml_backend_tensor_set(t, qtmp.data(), 0, qtmp.size());
        } else {
            std::fprintf(stderr, "vla(lingbot_va): unsupported runtime tensor dtype %d for %s\n",
                         (int) t->type, name);
            return false;
        }
        loaded += (uint64_t) ggml_nbytes(t);
    }
    std::printf("vla(lingbot_va): runtime weights loaded %.2f MiB into %s buffer (%zu tensors, %s)\n",
                loaded / (1024.0 * 1024.0), sw.is_cuda ? "CUDA" : "CPU", sw.tensors.size(), label);
    return true;
}

ggml_type resident_block_weight_type() {
    const char * env = std::getenv("VLA_LINGBOT_RESIDENT_BLOCK_DTYPE");
    if (!env || std::strlen(env) == 0 || std::strcmp(env, "f32") == 0 || std::strcmp(env, "F32") == 0) {
        return GGML_TYPE_F32;
    }
    if (std::strcmp(env, "bf16") == 0 || std::strcmp(env, "BF16") == 0) {
        return GGML_TYPE_BF16;
    }
    if (std::strcmp(env, "q4_0") == 0 || std::strcmp(env, "Q4_0") == 0) {
        return GGML_TYPE_Q4_0;
    }
    if (std::strcmp(env, "q4_1") == 0 || std::strcmp(env, "Q4_1") == 0) {
        return GGML_TYPE_Q4_1;
    }
    if (std::strcmp(env, "q5_0") == 0 || std::strcmp(env, "Q5_0") == 0) {
        return GGML_TYPE_Q5_0;
    }
    if (std::strcmp(env, "q5_1") == 0 || std::strcmp(env, "Q5_1") == 0) {
        return GGML_TYPE_Q5_1;
    }
    if (std::strcmp(env, "q8_0") == 0 || std::strcmp(env, "Q8_0") == 0) {
        return GGML_TYPE_Q8_0;
    }
    if (std::strcmp(env, "q2_K") == 0 || std::strcmp(env, "Q2_K") == 0 || std::strcmp(env, "q2_k") == 0) {
        return GGML_TYPE_Q2_K;
    }
    if (std::strcmp(env, "q3_K") == 0 || std::strcmp(env, "Q3_K") == 0 || std::strcmp(env, "q3_k") == 0) {
        return GGML_TYPE_Q3_K;
    }
    if (std::strcmp(env, "q4_K") == 0 || std::strcmp(env, "Q4_K") == 0 || std::strcmp(env, "q4_k") == 0) {
        return GGML_TYPE_Q4_K;
    }
    if (std::strcmp(env, "q5_K") == 0 || std::strcmp(env, "Q5_K") == 0 || std::strcmp(env, "q5_k") == 0) {
        return GGML_TYPE_Q5_K;
    }
    if (std::strcmp(env, "q6_K") == 0 || std::strcmp(env, "Q6_K") == 0 || std::strcmp(env, "q6_k") == 0) {
        return GGML_TYPE_Q6_K;
    }
    std::fprintf(stderr,
                 "vla(lingbot_va): unsupported VLA_LINGBOT_RESIDENT_BLOCK_DTYPE='%s'; "
                 "using f32. Supported: f32, bf16, q8_0, q6_K, q5_K, q4_K, q5_1, q5_0, q4_1, q4_0, q3_K, q2_K\n",
                 env);
    return GGML_TYPE_F32;
}

LingBotRuntimeWeights * get_resident_block_weights(
        gguf_reader & g,
        const std::string & ckpt_path,
        int64_t block_index) {
    struct ResidentBlockCache {
        std::string path;
        std::string backend_key;
        ggml_type type = GGML_TYPE_F32;
        std::vector<std::unique_ptr<LingBotRuntimeWeights>> blocks;
    };
    static ResidentBlockCache cache;
    const std::string backend_key = lingbot_runtime_backend_key();
    const ggml_type block_weight_type = resident_block_weight_type();
    const char * block_weight_type_name = ggml_get_type_traits(block_weight_type)->type_name;
    if (cache.path != ckpt_path || cache.backend_key != backend_key || cache.type != block_weight_type) {
        cache.blocks.clear();
        cache.path = ckpt_path;
        cache.backend_key = backend_key;
        cache.type = block_weight_type;
    }
    if (block_index < 0) return nullptr;
    int64_t max_blocks = 2;
    if (const char * env = std::getenv("VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX")) {
        char * end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end && *end == '\0' && v > 0) {
            max_blocks = (int64_t) v;
        } else {
            std::fprintf(stderr,
                         "vla(lingbot_va): ignoring invalid VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX='%s'\n",
                         env);
        }
    }
    if (block_index >= max_blocks) {
        auto bw = std::make_unique<LingBotRuntimeWeights>();
        ggml_init_params wp = { size_t(8) * 1024 * 1024, nullptr, true };
        bw->ctx = ggml_init(wp);
        if (!bw->ctx) return nullptr;
        bw->blocks.resize(1);
        if (!runtime_block_typed(g, *bw, block_index, bw->blocks[0], block_weight_type)) return nullptr;
        const std::string label = "uncached Wan block " + std::to_string(block_index) +
                                  " (" + block_weight_type_name + ")";
        if (!allocate_and_load_runtime_weights(g, *bw, label.c_str())) return nullptr;
        static std::unique_ptr<LingBotRuntimeWeights> overflow;
        overflow = std::move(bw);
        std::printf("vla(lingbot_va): resident block cache bypass: block=%lld max=%lld\n",
                    (long long) block_index, (long long) max_blocks);
        return overflow.get();
    }
    if ((size_t) block_index >= cache.blocks.size()) {
        cache.blocks.resize((size_t) block_index + 1);
    }
    if (cache.blocks[(size_t) block_index]) {
        std::printf("vla(lingbot_va): resident block cache hit: block=%lld dtype=%s\n",
                    (long long) block_index, block_weight_type_name);
        return cache.blocks[(size_t) block_index].get();
    }

    auto bw = std::make_unique<LingBotRuntimeWeights>();
    ggml_init_params wp = { size_t(8) * 1024 * 1024, nullptr, true };
    bw->ctx = ggml_init(wp);
    if (!bw->ctx) {
        std::fprintf(stderr, "vla(lingbot_va): resident block cache ggml_init failed for block %lld\n",
                     (long long) block_index);
        return nullptr;
    }
    bw->blocks.resize(1);
    if (!runtime_block_typed(g, *bw, block_index, bw->blocks[0], block_weight_type)) return nullptr;
    const std::string label = "resident Wan block " + std::to_string(block_index) +
                              " (" + block_weight_type_name + ")";
    if (!allocate_and_load_runtime_weights(g, *bw, label.c_str())) return nullptr;
    uint64_t resident_bytes = 0;
    for (const ggml_tensor * t : bw->tensors) resident_bytes += (uint64_t) ggml_nbytes(t);
    std::printf("vla(lingbot_va): resident block cache store: block=%lld dtype=%s tensors=%zu bytes=%.2f MiB\n",
                (long long) block_index, block_weight_type_name, bw->tensors.size(),
                resident_bytes / (1024.0 * 1024.0));
    cache.blocks[(size_t) block_index] = std::move(bw);
    return cache.blocks[(size_t) block_index].get();
}

bool make_runtime_common_weights(gguf_reader & g, LingBotRuntimeWeights & sw) {
    ggml_init_params wp = {
        size_t(8) * 1024 * 1024,
        nullptr,
        true,
    };
    sw.ctx = ggml_init(wp);
    if (!sw.ctx) {
        std::fprintf(stderr, "vla(lingbot_va): runtime ggml_init(weights) failed\n");
        return false;
    }

    if (!runtime_linear(g, sw, "wvm.patch_embd_mlp", sw.patch_embd_mlp)) return false;
    if (!runtime_linear(g, sw, "wvm.action_embd", sw.action_embd)) return false;
    if (!runtime_linear(g, sw, "wvm.output_proj", sw.output_proj)) return false;
    if (!runtime_linear(g, sw, "wvm.action_out", sw.action_out)) return false;
    sw.output_scale_shift = runtime_tensor(g, sw, "wvm.output_scale_shift");
    if (!sw.output_scale_shift) return false;
    if (!runtime_condition(g, sw, "wvm.cond", sw.cond)) return false;
    if (!runtime_condition(g, sw, "wvm.action_cond", sw.action_cond)) return false;
    return allocate_and_load_runtime_weights(g, sw, "common");
}

bool runtime_condition_typed(
        gguf_reader & g,
        LingBotRuntimeWeights & sw,
        const std::string & prefix,
        LingBotConditionW & out,
        ggml_type weight_type) {
    return runtime_linear_typed(g, sw, prefix + ".text_l1",  out.text_l1,  weight_type)
        && runtime_linear_typed(g, sw, prefix + ".text_l2",  out.text_l2,  weight_type)
        && runtime_linear_typed(g, sw, prefix + ".time_l1",  out.time_l1,  weight_type)
        && runtime_linear_typed(g, sw, prefix + ".time_l2",  out.time_l2,  weight_type)
        && runtime_linear_typed(g, sw, prefix + ".time_proj", out.time_proj, weight_type);
}

bool make_runtime_full_transformer_weights(
        gguf_reader & g,
        LingBotRuntimeWeights & sw,
        const LingBotVAModelArch & m,
        ggml_type weight_type) {
    ggml_init_params wp = { size_t(64) * 1024 * 1024, nullptr, true };
    sw.ctx = ggml_init(wp);
    if (!sw.ctx) {
        std::fprintf(stderr, "vla(lingbot_va): runtime ggml_init(full transformer weights) failed\n");
        return false;
    }

    if (!runtime_linear_typed(g, sw, "wvm.patch_embd_mlp", sw.patch_embd_mlp, weight_type)) return false;
    if (!runtime_linear_typed(g, sw, "wvm.action_embd",    sw.action_embd,    weight_type)) return false;
    if (!runtime_linear_typed(g, sw, "wvm.output_proj",    sw.output_proj,    weight_type)) return false;
    if (!runtime_linear_typed(g, sw, "wvm.action_out",     sw.action_out,     weight_type)) return false;
    sw.output_scale_shift = runtime_tensor(g, sw, "wvm.output_scale_shift");
    if (!sw.output_scale_shift) return false;
    if (!runtime_condition_typed(g, sw, "wvm.cond",        sw.cond,        weight_type)) return false;
    if (!runtime_condition_typed(g, sw, "wvm.action_cond", sw.action_cond, weight_type)) return false;

    sw.blocks.resize((size_t) m.n_layers);
    for (int64_t i = 0; i < m.n_layers; ++i) {
        if (!runtime_block_typed(g, sw, i, sw.blocks[(size_t) i], weight_type)) return false;
    }

    const std::string label = std::string("full Wan transformer (") +
        ggml_get_type_traits(weight_type)->type_name + ")";
    return allocate_and_load_runtime_weights(g, sw, label.c_str());
}

bool load_full_transformer_weights_if_requested(gguf_reader & g, LingBotVAModelArch & m) {
    if (std::getenv("VLA_LINGBOT_FULL_WEIGHTS_DISABLE") != nullptr) {
        std::printf("vla(lingbot_va): full transformer weight preload disabled by env\n");
        return true;
    }
    if (m.full_transformer_weights) return true;
    auto full = std::make_unique<LingBotRuntimeWeights>();
    ggml_type weight_type = GGML_TYPE_BF16;
    if (const char * env = std::getenv("VLA_LINGBOT_FULL_WEIGHT_DTYPE")) {
        if (std::strcmp(env, "f32") == 0 || std::strcmp(env, "F32") == 0) {
            weight_type = GGML_TYPE_F32;
        } else if (std::strcmp(env, "bf16") == 0 || std::strcmp(env, "BF16") == 0) {
            weight_type = GGML_TYPE_BF16;
        } else {
            std::fprintf(stderr,
                         "vla(lingbot_va): unsupported VLA_LINGBOT_FULL_WEIGHT_DTYPE='%s'; "
                         "using bf16\n", env);
        }
    }
    if (!make_runtime_full_transformer_weights(g, *full, m, weight_type)) return false;
    m.full_transformer_weights = full.release();
    std::printf("vla(lingbot_va): full transformer weights are resident for predict path\n");
    return true;
}



void fill_deterministic(std::vector<float> & v, float scale) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = std::sin((float) i * 0.013f) * scale;
    }
}

uint64_t lingbot_env_u64(const char * name, uint64_t fallback) {
    const char * env = std::getenv(name);
    if (!env || std::strlen(env) == 0) return fallback;
    char * end = nullptr;
    const unsigned long long v = std::strtoull(env, &end, 10);
    if (end && *end == '\0') return (uint64_t) v;
    std::fprintf(stderr, "vla(lingbot_va): ignoring invalid %s='%s'\n", name, env);
    return fallback;
}

uint64_t lingbot_mix_seed(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::mt19937_64 make_lingbot_noise_rng(uint64_t session_id, uint64_t predict_index) {
    if (const char * env = std::getenv("VLA_LINGBOT_NOISE_SEED")) {
        const uint64_t base = lingbot_env_u64("VLA_LINGBOT_NOISE_SEED", 0);
        const uint64_t mixed = lingbot_mix_seed(base) ^
            lingbot_mix_seed(session_id + 0x100000001b3ULL) ^
            lingbot_mix_seed(predict_index + 0xcbf29ce484222325ULL);
        return std::mt19937_64(mixed);
    }

    std::random_device rd;
    const auto now = (uint64_t) std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const uint64_t seed = lingbot_mix_seed(now) ^
        lingbot_mix_seed(((uint64_t) rd() << 32) ^ (uint64_t) rd()) ^
        lingbot_mix_seed(session_id + 0x100000001b3ULL) ^
        lingbot_mix_seed(predict_index + 0xcbf29ce484222325ULL);
    return std::mt19937_64(seed);
}

void fill_standard_normal(std::vector<float> & v, std::mt19937_64 & rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float & x : v) {
        x = dist(rng);
    }
}

uint16_t lingbot_f32_to_bf16_bits_rne(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    return (uint16_t) ((u + 0x7fffu + lsb) >> 16);
}

float lingbot_bf16_bits_to_f32(uint16_t x) {
    uint32_t u = (uint32_t) x << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

void lingbot_bf16_roundtrip_inplace(std::vector<float> & v) {
    for (float & x : v) {
        x = lingbot_bf16_bits_to_f32(lingbot_f32_to_bf16_bits_rne(x));
    }
}

void fill_timestep_embedding(std::vector<float> & out, int64_t dim, int64_t seq, double timestep) {
    if ((int64_t) out.size() != dim * seq || dim <= 0) {
        return;
    }
    const int64_t half = dim / 2;
    const double denom = (double) half; // diffusers Timesteps(..., downscale_freq_shift=0)
    for (int64_t s = 0; s < seq; ++s) {
        for (int64_t i = 0; i < half; ++i) {
            const double exponent = -std::log(10000.0) * (double) i / denom;
            const double v = timestep * std::exp(exponent);
            const size_t cos_off = (size_t) s * (size_t) dim + (size_t) i;
            const size_t sin_off = cos_off + (size_t) half;
            // LingBot uses diffusers Timesteps(..., flip_sin_to_cos=True).
            out[cos_off] = (float) std::cos(v);
            out[sin_off] = (float) std::sin(v);
        }
        if (dim % 2 != 0) {
            out[(size_t) s * (size_t) dim + (size_t) dim - 1] = 0.0f;
        }
    }
}

struct LingBotTensor5DShape {
    int64_t b = 1;
    int64_t c = 1;
    int64_t f = 1;
    int64_t h = 1;
    int64_t w = 1;
};

size_t idx5(const LingBotTensor5DShape & s, int64_t b, int64_t c, int64_t f, int64_t h, int64_t w) {
    return (((((size_t) b * (size_t) s.c + (size_t) c) * (size_t) s.f + (size_t) f) *
             (size_t) s.h + (size_t) h) * (size_t) s.w + (size_t) w);
}

bool shape_valid(const LingBotTensor5DShape & s) {
    return s.b > 0 && s.c > 0 && s.f > 0 && s.h > 0 && s.w > 0;
}

double checksum(const std::vector<float> & v);

struct LingBotCachedLatent {
    LingBotTensor5DShape shape{};
    std::vector<float> data;
};

struct LingBotCachedVideoHistory {
    int64_t views = 0;
    int64_t channels = 0;
    int64_t height = 0;
    int64_t width = 0;
    std::vector<int> chunks;
    std::vector<float> data_vcfhw;
};

std::mutex & lingbot_init_latent_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<uint64_t, LingBotCachedLatent> & lingbot_init_latent_cache() {
    static std::unordered_map<uint64_t, LingBotCachedLatent> cache;
    return cache;
}

std::unordered_map<uint64_t, LingBotCachedVideoHistory> & lingbot_video_history_cache() {
    static std::unordered_map<uint64_t, LingBotCachedVideoHistory> cache;
    return cache;
}

void lingbot_store_init_latent(uint64_t session_id,
                               const std::vector<float> & latent,
                               const LingBotTensor5DShape & shape) {
    if (!shape_valid(shape) || shape.f != 1 || latent.empty()) return;
    std::lock_guard<std::mutex> lock(lingbot_init_latent_mutex());
    lingbot_init_latent_cache()[session_id] = LingBotCachedLatent{shape, latent};
    std::printf("vla(lingbot_va): stored init latent session=%llu shape=[%lld,%lld,%lld,%lld,%lld] checksum=%.9g\n",
                (unsigned long long) session_id,
                (long long) shape.b, (long long) shape.c, (long long) shape.f,
                (long long) shape.h, (long long) shape.w, checksum(latent));
}

bool lingbot_prepend_init_latent(uint64_t session_id,
                                 std::vector<float> & latent,
                                 LingBotTensor5DShape & shape) {
    LingBotCachedLatent cached;
    {
        std::lock_guard<std::mutex> lock(lingbot_init_latent_mutex());
        auto it = lingbot_init_latent_cache().find(session_id);
        if (it == lingbot_init_latent_cache().end()) return false;
        cached = it->second;
    }
    if (!shape_valid(shape) || !shape_valid(cached.shape) ||
        shape.b != cached.shape.b || shape.c != cached.shape.c ||
        shape.h != cached.shape.h || shape.w != cached.shape.w ||
        cached.shape.f != 1 || latent.empty() || cached.data.empty()) {
        return false;
    }
    LingBotTensor5DShape out_shape = shape;
    out_shape.f = cached.shape.f + shape.f;
    std::vector<float> out((size_t) out_shape.b * (size_t) out_shape.c *
                           (size_t) out_shape.f * (size_t) out_shape.h *
                           (size_t) out_shape.w);
    for (int64_t b = 0; b < out_shape.b; ++b) {
        for (int64_t c = 0; c < out_shape.c; ++c) {
            for (int64_t f = 0; f < cached.shape.f; ++f) {
                for (int64_t h = 0; h < out_shape.h; ++h) {
                    for (int64_t w = 0; w < out_shape.w; ++w) {
                        out[idx5(out_shape, b, c, f, h, w)] =
                            cached.data[idx5(cached.shape, b, c, f, h, w)];
                    }
                }
            }
            for (int64_t f = 0; f < shape.f; ++f) {
                for (int64_t h = 0; h < out_shape.h; ++h) {
                    for (int64_t w = 0; w < out_shape.w; ++w) {
                        out[idx5(out_shape, b, c, f + cached.shape.f, h, w)] =
                            latent[idx5(shape, b, c, f, h, w)];
                    }
                }
            }
        }
    }
    latent = std::move(out);
    shape = out_shape;
    std::printf("vla(lingbot_va): prepended init latent for first cache update session=%llu "
                "shape=[%lld,%lld,%lld,%lld,%lld] checksum=%.9g\n",
                (unsigned long long) session_id,
                (long long) shape.b, (long long) shape.c, (long long) shape.f,
                (long long) shape.h, (long long) shape.w, checksum(latent));
    return true;
}

bool lingbot_slice_latent_time(const std::vector<float> & latent,
                               const LingBotTensor5DShape & shape,
                               int64_t start_f,
                               int64_t count_f,
                               std::vector<float> & out,
                               LingBotTensor5DShape & out_shape) {
    if (!shape_valid(shape) || start_f < 0 || count_f <= 0 || start_f + count_f > shape.f ||
        latent.size() != (size_t) shape.b * (size_t) shape.c * (size_t) shape.f *
                         (size_t) shape.h * (size_t) shape.w) {
        return false;
    }
    out_shape = shape;
    out_shape.f = count_f;
    out.assign((size_t) out_shape.b * (size_t) out_shape.c * (size_t) out_shape.f *
               (size_t) out_shape.h * (size_t) out_shape.w, 0.0f);
    for (int64_t b = 0; b < out_shape.b; ++b) {
        for (int64_t c = 0; c < out_shape.c; ++c) {
            for (int64_t f = 0; f < count_f; ++f) {
                for (int64_t h = 0; h < out_shape.h; ++h) {
                    for (int64_t w = 0; w < out_shape.w; ++w) {
                        out[idx5(out_shape, b, c, f, h, w)] =
                            latent[idx5(shape, b, c, start_f + f, h, w)];
                    }
                }
            }
        }
    }
    return true;
}

void lingbot_store_video_history(uint64_t session_id,
                                 const float * video_vcfhw,
                                 int64_t views,
                                 int64_t channels,
                                 int64_t frames,
                                 int64_t height,
                                 int64_t width) {
    if (session_id == 0 || !video_vcfhw || views <= 0 || channels <= 0 || frames <= 0 || height <= 0 || width <= 0) return;
    const size_t elems = (size_t) views * (size_t) channels * (size_t) frames *
                         (size_t) height * (size_t) width;
    LingBotCachedVideoHistory hist;
    hist.views = views;
    hist.channels = channels;
    hist.height = height;
    hist.width = width;
    hist.chunks.push_back((int) frames);
    hist.data_vcfhw.assign(video_vcfhw, video_vcfhw + elems);
    std::lock_guard<std::mutex> lock(lingbot_init_latent_mutex());
    lingbot_video_history_cache()[session_id] = std::move(hist);
}

bool lingbot_build_video_history_with_current(uint64_t session_id,
                                              const float * video_vcfhw,
                                              int64_t views,
                                              int64_t channels,
                                              int64_t frames,
                                              int64_t height,
                                              int64_t width,
                                              std::vector<float> & out_vcfhw,
                                              std::vector<int> & out_chunks,
                                              int64_t * out_frames) {
    if (session_id == 0 || !video_vcfhw || views <= 0 || channels <= 0 || frames <= 0 || height <= 0 || width <= 0) return false;
    LingBotCachedVideoHistory hist;
    {
        std::lock_guard<std::mutex> lock(lingbot_init_latent_mutex());
        auto it = lingbot_video_history_cache().find(session_id);
        if (it == lingbot_video_history_cache().end()) return false;
        hist = it->second;
    }
    if (hist.views != views || hist.channels != channels || hist.height != height || hist.width != width ||
        hist.chunks.empty() || hist.data_vcfhw.empty()) {
        return false;
    }
    int64_t hist_frames = 0;
    for (int c : hist.chunks) {
        if (c <= 0) return false;
        hist_frames += c;
    }
    const size_t hist_expected = (size_t) views * (size_t) channels * (size_t) hist_frames *
                                 (size_t) height * (size_t) width;
    if (hist.data_vcfhw.size() != hist_expected) return false;

    const int64_t total_frames = hist_frames + frames;
    out_vcfhw.assign((size_t) views * (size_t) channels * (size_t) total_frames *
                     (size_t) height * (size_t) width, 0.0f);
    for (int64_t v = 0; v < views; ++v) {
        for (int64_t c = 0; c < channels; ++c) {
            for (int64_t f = 0; f < hist_frames; ++f) {
                for (int64_t h = 0; h < height; ++h) {
                    for (int64_t w = 0; w < width; ++w) {
                        const size_t dst = (((((size_t) v * (size_t) channels + (size_t) c) * (size_t) total_frames + (size_t) f) *
                                             (size_t) height + (size_t) h) * (size_t) width + (size_t) w);
                        const size_t src = (((((size_t) v * (size_t) channels + (size_t) c) * (size_t) hist_frames + (size_t) f) *
                                             (size_t) height + (size_t) h) * (size_t) width + (size_t) w);
                        out_vcfhw[dst] = hist.data_vcfhw[src];
                    }
                }
            }
            for (int64_t f = 0; f < frames; ++f) {
                for (int64_t h = 0; h < height; ++h) {
                    for (int64_t w = 0; w < width; ++w) {
                        const size_t dst = (((((size_t) v * (size_t) channels + (size_t) c) * (size_t) total_frames + (size_t) (hist_frames + f)) *
                                             (size_t) height + (size_t) h) * (size_t) width + (size_t) w);
                        const size_t src = (((((size_t) v * (size_t) channels + (size_t) c) * (size_t) frames + (size_t) f) *
                                             (size_t) height + (size_t) h) * (size_t) width + (size_t) w);
                        out_vcfhw[dst] = video_vcfhw[src];
                    }
                }
            }
        }
    }
    out_chunks = hist.chunks;
    out_chunks.push_back((int) frames);
    if (out_frames) *out_frames = total_frames;
    return true;
}

void lingbot_append_video_history(uint64_t session_id,
                                  const float * video_vcfhw,
                                  int64_t views,
                                  int64_t channels,
                                  int64_t frames,
                                  int64_t height,
                                  int64_t width) {
    std::vector<float> combined;
    std::vector<int> chunks;
    int64_t total_frames = 0;
    if (!lingbot_build_video_history_with_current(session_id, video_vcfhw, views, channels, frames, height, width,
                                                  combined, chunks, &total_frames)) {
        lingbot_store_video_history(session_id, video_vcfhw, views, channels, frames, height, width);
        return;
    }
    LingBotCachedVideoHistory hist;
    hist.views = views;
    hist.channels = channels;
    hist.height = height;
    hist.width = width;
    hist.chunks = std::move(chunks);
    hist.data_vcfhw = std::move(combined);
    std::lock_guard<std::mutex> lock(lingbot_init_latent_mutex());
    lingbot_video_history_cache()[session_id] = std::move(hist);
}

bool patchify_latent_tokens(
        const std::vector<float> & latent,
        const LingBotTensor5DShape & s,
        int64_t pt,
        int64_t ph,
        int64_t pw,
        std::vector<float> & tokens,
        int64_t & feature_dim,
        int64_t & seq) {
    if (!shape_valid(s) || pt <= 0 || ph <= 0 || pw <= 0 ||
        s.f % pt != 0 || s.h % ph != 0 || s.w % pw != 0 ||
        latent.size() != (size_t) s.b * (size_t) s.c * (size_t) s.f * (size_t) s.h * (size_t) s.w) {
        return false;
    }
    const int64_t pf = s.f / pt;
    const int64_t phn = s.h / ph;
    const int64_t pwn = s.w / pw;
    feature_dim = s.c * pt * ph * pw;
    seq = s.b * pf * phn * pwn;
    tokens.assign((size_t) feature_dim * (size_t) seq, 0.0f);

    int64_t tok = 0;
    for (int64_t b = 0; b < s.b; ++b) {
        for (int64_t fo = 0; fo < pf; ++fo) {
            for (int64_t ho = 0; ho < phn; ++ho) {
                for (int64_t wo = 0; wo < pwn; ++wo, ++tok) {
                    for (int64_t c = 0; c < s.c; ++c) {
                        for (int64_t pi = 0; pi < pt; ++pi) {
                            for (int64_t pj = 0; pj < ph; ++pj) {
                                for (int64_t pk = 0; pk < pw; ++pk) {
                                    const int64_t feat = (((c * pt + pi) * ph + pj) * pw + pk);
                                    tokens[(size_t) feat + (size_t) feature_dim * (size_t) tok] =
                                        latent[idx5(s, b, c, fo * pt + pi, ho * ph + pj, wo * pw + pk)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool unpatchify_latent_tokens(
        const std::vector<float> & tokens,
        const LingBotTensor5DShape & s,
        int64_t pt,
        int64_t ph,
        int64_t pw,
        std::vector<float> & latent) {
    int64_t feature_dim = 0;
    int64_t seq = 0;
    std::vector<float> unused;
    if (!patchify_latent_tokens(std::vector<float>((size_t) s.b * (size_t) s.c * (size_t) s.f * (size_t) s.h * (size_t) s.w),
                                s, pt, ph, pw, unused, feature_dim, seq) ||
        tokens.size() != (size_t) feature_dim * (size_t) seq) {
        return false;
    }
    latent.assign((size_t) s.b * (size_t) s.c * (size_t) s.f * (size_t) s.h * (size_t) s.w, 0.0f);
    const int64_t pf = s.f / pt;
    const int64_t phn = s.h / ph;
    const int64_t pwn = s.w / pw;
    int64_t tok = 0;
    for (int64_t b = 0; b < s.b; ++b) {
        for (int64_t fo = 0; fo < pf; ++fo) {
            for (int64_t ho = 0; ho < phn; ++ho) {
                for (int64_t wo = 0; wo < pwn; ++wo, ++tok) {
                    for (int64_t c = 0; c < s.c; ++c) {
                        for (int64_t pi = 0; pi < pt; ++pi) {
                            for (int64_t pj = 0; pj < ph; ++pj) {
                                for (int64_t pk = 0; pk < pw; ++pk) {
                                    const int64_t feat = (((c * pt + pi) * ph + pj) * pw + pk);
                                    latent[idx5(s, b, c, fo * pt + pi, ho * ph + pj, wo * pw + pk)] =
                                        tokens[(size_t) feat + (size_t) feature_dim * (size_t) tok];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool projected_latent_tokens_to_tensor(
        const std::vector<float> & projected,
        const LingBotTensor5DShape & s,
        int64_t pt,
        int64_t ph,
        int64_t pw,
        std::vector<float> & latent) {
    if (!shape_valid(s) || pt <= 0 || ph <= 0 || pw <= 0 ||
        s.f % pt != 0 || s.h % ph != 0 || s.w % pw != 0) {
        return false;
    }
    const int64_t pf = s.f / pt;
    const int64_t phn = s.h / ph;
    const int64_t pwn = s.w / pw;
    const int64_t n = pt * ph * pw;
    const int64_t feature_dim = n * s.c;
    const int64_t seq = s.b * pf * phn * pwn;
    if (projected.size() != (size_t) feature_dim * (size_t) seq) {
        return false;
    }
    latent.assign((size_t) s.b * (size_t) s.c * (size_t) s.f * (size_t) s.h * (size_t) s.w, 0.0f);
    int64_t tok = 0;
    for (int64_t b = 0; b < s.b; ++b) {
        for (int64_t fo = 0; fo < pf; ++fo) {
            for (int64_t ho = 0; ho < phn; ++ho) {
                for (int64_t wo = 0; wo < pwn; ++wo, ++tok) {
                    for (int64_t pi = 0; pi < pt; ++pi) {
                        for (int64_t pj = 0; pj < ph; ++pj) {
                            for (int64_t pk = 0; pk < pw; ++pk) {
                                const int64_t patch_index = ((pi * ph + pj) * pw + pk);
                                for (int64_t c = 0; c < s.c; ++c) {
                                    const int64_t feat = patch_index * s.c + c;
                                    latent[idx5(s, b, c, fo * pt + pi, ho * ph + pj, wo * pw + pk)] =
                                        projected[(size_t) feat + (size_t) feature_dim * (size_t) tok];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool action_tensor_to_tokens(
        const std::vector<float> & action,
        const LingBotTensor5DShape & s,
        std::vector<float> & tokens,
        int64_t & feature_dim,
        int64_t & seq) {
    if (!shape_valid(s) || action.size() != (size_t) s.b * (size_t) s.c * (size_t) s.f * (size_t) s.h * (size_t) s.w) {
        return false;
    }
    feature_dim = s.c;
    seq = s.b * s.f * s.h * s.w;
    tokens.assign((size_t) feature_dim * (size_t) seq, 0.0f);
    if (seq == 1) {
        std::copy(action.begin(), action.end(), tokens.begin());
        return true;
    }
    int64_t tok = 0;
    for (int64_t b = 0; b < s.b; ++b) {
        for (int64_t f = 0; f < s.f; ++f) {
            for (int64_t h = 0; h < s.h; ++h) {
                for (int64_t w = 0; w < s.w; ++w, ++tok) {
                    for (int64_t c = 0; c < s.c; ++c) {
                        tokens[(size_t) c + (size_t) feature_dim * (size_t) tok] = action[idx5(s, b, c, f, h, w)];
                    }
                }
            }
        }
    }
    return true;
}

bool action_tokens_to_tensor(
        const std::vector<float> & tokens,
        const LingBotTensor5DShape & s,
        std::vector<float> & action) {
    int64_t feature_dim = 0;
    int64_t seq = 0;
    std::vector<float> unused;
    if (!action_tensor_to_tokens(std::vector<float>((size_t) s.b * (size_t) s.c * (size_t) s.f * (size_t) s.h * (size_t) s.w),
                                 s, unused, feature_dim, seq) ||
        tokens.size() != (size_t) feature_dim * (size_t) seq) {
        return false;
    }
    action.assign((size_t) s.b * (size_t) s.c * (size_t) s.f * (size_t) s.h * (size_t) s.w, 0.0f);
    if (seq == 1) {
        std::copy(tokens.begin(), tokens.end(), action.begin());
        return true;
    }
    int64_t tok = 0;
    for (int64_t b = 0; b < s.b; ++b) {
        for (int64_t f = 0; f < s.f; ++f) {
            for (int64_t h = 0; h < s.h; ++h) {
                for (int64_t w = 0; w < s.w; ++w, ++tok) {
                    for (int64_t c = 0; c < s.c; ++c) {
                        action[idx5(s, b, c, f, h, w)] = tokens[(size_t) c + (size_t) feature_dim * (size_t) tok];
                    }
                }
            }
        }
    }
    return true;
}

double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double out = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        out = std::max(out, std::abs((double) a[i] - (double) b[i]));
    }
    return out;
}

double checksum(const std::vector<float> & v) {
    return std::accumulate(v.begin(), v.end(), 0.0);
}

uint64_t lingbot_hash_f32_bytes(const std::vector<float> & v) {
    const unsigned char * bytes = reinterpret_cast<const unsigned char *>(v.data());
    const size_t n = v.size() * sizeof(float);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t) bytes[i];
        h *= 1099511628211ULL;
    }
    h ^= (uint64_t) v.size();
    h *= 1099511628211ULL;
    return h ? h : 1ULL;
}

double max_abs_value(const std::vector<float> & v) {
    double out = 0.0;
    for (float x : v) out = std::max(out, std::abs((double) x));
    return out;
}

ggml_tensor * vae_causal_conv3d_ks3_pad1(
        ggml_context * C,
        ggml_tensor * w,
        ggml_tensor * x_w_h_t_c,
        int in_C);

bool vae_patchify_spatial_host(
        const std::vector<float> & input,
        const LingBotTensor5DShape & in_shape,
        int64_t patch,
        std::vector<float> & output,
        LingBotTensor5DShape & out_shape) {
    if (!shape_valid(in_shape) || patch <= 0 ||
        in_shape.h % patch != 0 || in_shape.w % patch != 0) {
        return false;
    }
    out_shape = {in_shape.b, in_shape.c * patch * patch,
                 in_shape.f, in_shape.h / patch, in_shape.w / patch};
    output.assign((size_t) out_shape.b * (size_t) out_shape.c *
                  (size_t) out_shape.f * (size_t) out_shape.h * (size_t) out_shape.w, 0.0f);

    for (int64_t b = 0; b < in_shape.b; ++b) {
        for (int64_t c = 0; c < in_shape.c; ++c) {
            for (int64_t pw = 0; pw < patch; ++pw) {
                for (int64_t ph = 0; ph < patch; ++ph) {
                    const int64_t out_c = ((c * patch + pw) * patch + ph);
                    for (int64_t f = 0; f < in_shape.f; ++f) {
                        for (int64_t h = 0; h < out_shape.h; ++h) {
                            for (int64_t w = 0; w < out_shape.w; ++w) {
                                output[idx5(out_shape, b, out_c, f, h, w)] =
                                    input[idx5(in_shape, b, c, f, h * patch + ph, w * patch + pw)];
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool vae_patchify_rgb_bcfhw_to_whdc(
        const std::vector<float> & raw_bcfhw,
        int H_raw,
        int W_raw,
        int T,
        std::vector<float> & patch_whdc,
        int * out_W,
        int * out_H) {
    const LingBotTensor5DShape raw_shape{1, 3, T, H_raw, W_raw};
    LingBotTensor5DShape patch_shape;
    std::vector<float> patch_bcfhw;
    if (!vae_patchify_spatial_host(raw_bcfhw, raw_shape, 2, patch_bcfhw, patch_shape)) {
        return false;
    }
    if (patch_shape.b != 1 || patch_shape.c != 12 || patch_shape.f != T) return false;
    patch_whdc.assign((size_t) patch_shape.w * patch_shape.h * patch_shape.f * patch_shape.c, 0.0f);
    for (int64_t c = 0; c < patch_shape.c; ++c) {
        for (int64_t f = 0; f < patch_shape.f; ++f) {
            for (int64_t h = 0; h < patch_shape.h; ++h) {
                for (int64_t w = 0; w < patch_shape.w; ++w) {
                    const size_t whdc_idx = (size_t) w + (size_t) patch_shape.w *
                        ((size_t) h + (size_t) patch_shape.h *
                        ((size_t) f + (size_t) patch_shape.f * (size_t) c));
                    patch_whdc[whdc_idx] =
                        patch_bcfhw[idx5(patch_shape, 0, c, f, h, w)];
                }
            }
        }
    }
    if (out_W) *out_W = (int) patch_shape.w;
    if (out_H) *out_H = (int) patch_shape.h;
    return true;
}







ggml_tensor * vae_norm_silu_to_conv_layout(
        ggml_context * C,
        ggml_tensor * x_w_h_d_c,
        ggml_tensor * gamma_c_1_1_1) {
    ggml_tensor * x_c_w_h_d = ggml_cont(C, ggml_permute(C, x_w_h_d_c, 1, 2, 3, 0));
    ggml_tensor * n = ggml_rms_norm(C, x_c_w_h_d, 1e-12f);
    if (!ggml_can_repeat(gamma_c_1_1_1, n)) {
        std::fprintf(stderr,
                     "vla(lingbot_va): VAE RMSNorm gamma shape mismatch: x=[%lld,%lld,%lld,%lld] "
                     "norm=[%lld,%lld,%lld,%lld] gamma=[%lld,%lld,%lld,%lld] name=%s\n",
                     (long long) x_w_h_d_c->ne[0], (long long) x_w_h_d_c->ne[1],
                     (long long) x_w_h_d_c->ne[2], (long long) x_w_h_d_c->ne[3],
                     (long long) n->ne[0], (long long) n->ne[1],
                     (long long) n->ne[2], (long long) n->ne[3],
                     (long long) gamma_c_1_1_1->ne[0], (long long) gamma_c_1_1_1->ne[1],
                     (long long) gamma_c_1_1_1->ne[2], (long long) gamma_c_1_1_1->ne[3],
                     ggml_get_name(gamma_c_1_1_1));
        return x_w_h_d_c;
    }
    ggml_tensor * a = ggml_silu(C, ggml_mul(C, n, gamma_c_1_1_1));
    return ggml_cont(C, ggml_permute(C, a, 3, 0, 1, 2));
}

ggml_tensor * vae_causal_conv3d_ks3_pad1(
        ggml_context * C,
        ggml_tensor * w,
        ggml_tensor * x_w_h_t_c,
        int in_C) {
    ggml_tensor * padded = ggml_pad_ext(C, x_w_h_t_c,
                                        1, 1,  // W
                                        1, 1,  // H
                                        2, 0,  // T, causal left padding
                                        0, 0); // C
    return ggml_conv_3d(C, w, padded, in_C,
                        1, 1, 1,
                        0, 0, 0,
                        1, 1, 1);
}

ggml_tensor * vae_spatial_downsample_conv2d(
        ggml_context * C,
        ggml_tensor * w,
        ggml_tensor * x_w_h_c_t) {
    ggml_tensor * padded = ggml_pad_ext(C, x_w_h_c_t,
                                        0, 1,  // W, right pad only
                                        0, 1,  // H, bottom pad only
                                        0, 0,
                                        0, 0);
    return ggml_conv_2d(C, w, padded, 2, 2, 0, 0, 1, 1);
}

bool lingbot_cached_f32_tensor(gguf_reader & g,
                               const std::string & name,
                               int64_t expected_nelements,
                               std::vector<float> & out) {
    if (expected_nelements <= 0) return false;
    struct Entry {
        int64_t nelements = 0;
        std::vector<float> data;
    };
    static std::mutex mu;
    static std::unordered_map<std::string, Entry> cache;

    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(name);
        if (it != cache.end() && it->second.nelements == expected_nelements) {
            out = it->second.data;
            return true;
        }
    }

    std::vector<float> tmp((size_t) expected_nelements);
    if (!g.read_to_f32(name.c_str(), tmp.data(), expected_nelements)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mu);
        auto & entry = cache[name];
        entry.nelements = expected_nelements;
        entry.data = tmp;
        out = entry.data;
    }
    return true;
}

bool set_tensor_from_gguf_f32(gguf_reader & g, ggml_tensor * t) {
    std::vector<float> tmp;
    if (!lingbot_cached_f32_tensor(g, ggml_get_name(t), ggml_nelements(t), tmp)) {
        return false;
    }
    ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size() * sizeof(float));
    return true;
}

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
uint16_t lingbot_f32_to_bf16_bits(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    const uint32_t rounding_bias = 0x7fffu + lsb;
    return (uint16_t) ((u + rounding_bias) >> 16);
}

bool lingbot_cuda_upload_f32_as_bf16(
        gguf_reader & g,
        const char * name,
        int64_t expected_nelements,
        unsigned short ** dst) {
    if (!dst || expected_nelements <= 0) return false;
    if (*dst) return true;
    std::vector<float> f32;
    if (!lingbot_cached_f32_tensor(g, name, expected_nelements, f32)) return false;
    std::vector<uint16_t> bf16((size_t) expected_nelements);
    for (size_t i = 0; i < f32.size(); ++i) bf16[i] = lingbot_f32_to_bf16_bits(f32[i]);
    if (cudaMalloc(reinterpret_cast<void **>(dst), bf16.size() * sizeof(uint16_t)) != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA malloc failed for VAE tensor %s\n", name);
        *dst = nullptr;
        return false;
    }
    if (cudaMemcpy(*dst, bf16.data(), bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA upload failed for VAE tensor %s\n", name);
        cudaFree(*dst);
        *dst = nullptr;
        return false;
    }
    return true;
}

const unsigned short * lingbot_cuda_cached_bf16_tensor(
        gguf_reader & g,
        const std::string & name,
        int64_t expected_nelements) {
    static std::mutex mu;
    static std::unordered_map<std::string, unsigned short *> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    unsigned short * ptr = nullptr;
    if (!lingbot_cuda_upload_f32_as_bf16(g, name.c_str(), expected_nelements, &ptr)) {
        return nullptr;
    }
    cache.emplace(name, ptr);
    return ptr;
}

bool lingbot_cuda_upload_f32(gguf_reader & g, const char * name, int64_t expected_nelements, float ** dst) {
    if (!dst || expected_nelements <= 0) return false;
    std::vector<float> f32;
    if (!lingbot_cached_f32_tensor(g, name, expected_nelements, f32)) return false;
    if (cudaMalloc(reinterpret_cast<void **>(dst), f32.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA malloc failed for VAE tensor %s\n", name);
        *dst = nullptr;
        return false;
    }
    if (cudaMemcpy(*dst, f32.data(), f32.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA upload failed for VAE tensor %s\n", name);
        cudaFree(*dst);
        *dst = nullptr;
        return false;
    }
    return true;
}

const float * lingbot_cuda_cached_f32_tensor(
        gguf_reader & g,
        const std::string & name,
        int64_t expected_nelements) {
    static std::mutex mu;
    static std::unordered_map<std::string, float *> cache;
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    float * ptr = nullptr;
    if (!lingbot_cuda_upload_f32(g, name.c_str(), expected_nelements, &ptr)) {
        return nullptr;
    }
    cache.emplace(name, ptr);
    return ptr;
}

struct LingBotCudaF32Scratch {
    float * ptr = nullptr;
    size_t elems = 0;

    ~LingBotCudaF32Scratch() {
        if (ptr) cudaFree(ptr);
    }

    bool ensure(size_t need_elems) {
        if (need_elems == 0) return false;
        if (ptr && elems >= need_elems) return true;
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            elems = 0;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&ptr), need_elems * sizeof(float)) != cudaSuccess) {
            ptr = nullptr;
            return false;
        }
        elems = need_elems;
        return true;
    }
};

struct LingBotVaeConvInCudaCache {
    float * weight = nullptr;
    float * bias = nullptr;

    ~LingBotVaeConvInCudaCache() {
        if (weight) cudaFree(weight);
        if (bias) cudaFree(bias);
    }
};
#endif

bool vae_encoder_conv_in_ggml_execute(
        gguf_reader & g,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int in_C,
        std::vector<float> & out_whdc) {
    const int out_C = 160;
    if (W <= 0 || H <= 0 || T <= 0 || in_C != 12) return false;
    if (in_whdc.size() != (size_t) W * H * T * in_C) return false;

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_cuda_requested() &&
        !lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_CONV_IN_DISABLE")) {
        static LingBotVaeConvInCudaCache cache;
        static std::mutex cache_mu;
        std::lock_guard<std::mutex> lock(cache_mu);
        const int64_t weight_elems = (int64_t) out_C * in_C * 3 * 3 * 3;
        const bool cache_ok =
            (cache.weight || lingbot_cuda_upload_f32(g, "vae.encoder.conv_in.weight", weight_elems, &cache.weight)) &&
            (cache.bias   || lingbot_cuda_upload_f32(g, "vae.encoder.conv_in.bias", out_C, &cache.bias));
        if (cache_ok) {
            float * d_in = nullptr;
            float * d_out = nullptr;
            const size_t in_bytes = in_whdc.size() * sizeof(float);
            const size_t out_elems = (size_t) W * (size_t) H * (size_t) T * (size_t) out_C;
            const size_t out_bytes = out_elems * sizeof(float);
            bool ok = cudaMalloc(reinterpret_cast<void **>(&d_in), in_bytes) == cudaSuccess &&
                      cudaMalloc(reinterpret_cast<void **>(&d_out), out_bytes) == cudaSuccess &&
                      cudaMemcpy(d_in, in_whdc.data(), in_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                      lingbot_vae_causal_conv3d_ks3_whdc_f32w(d_in, cache.weight, cache.bias,
                                                                  d_out, W, H, T, in_C, out_C, nullptr) == 0 &&
                      cudaDeviceSynchronize() == cudaSuccess;
            if (ok) {
                out_whdc.assign(out_elems, 0.0f);
                ok = cudaMemcpy(out_whdc.data(), d_out, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
            }
            if (d_in) cudaFree(d_in);
            if (d_out) cudaFree(d_out);
            if (ok) {
                if (std::getenv("VLA_LINGBOT_TIMING")) {
                    std::printf("vla(lingbot_va): VAE conv_in CUDA path ok shape=[%d,%d,%d,%d]\n",
                                W, H, T, in_C);
                }
                return true;
            }
            std::fprintf(stderr, "vla(lingbot_va): VAE conv_in CUDA path failed; falling back to ggml CPU\n");
        }
    }
#endif

    ggml_init_params params = { size_t(192) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(params);
    if (!C) return false;

    ggml_tensor * w = ggml_new_tensor_4d(C, GGML_TYPE_F32, 3, 3, 3, out_C * in_C);
    ggml_set_name(w, "vae.encoder.conv_in.weight");
    ggml_tensor * b = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, 1, out_C);
    ggml_set_name(b, "vae.encoder.conv_in.bias");
    ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, W, H, T, in_C);
    ggml_set_name(x, "vae.encoder.conv_in.exec.input");
    ggml_tensor * y = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, w, x, in_C), b);
    ggml_set_name(y, "vae.encoder.conv_in.exec.output");
    ggml_set_output(y);

    LingBotAuxGraphBackend graph_backend;
    if (!lingbot_aux_graph_backend_init(graph_backend, "vae.encoder.conv_in")) {
        ggml_free(C);
        return false;
    }
    ggml_backend_t backend = graph_backend.backend;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, graph_backend.buft);
    if (!buf) {
        lingbot_aux_graph_backend_free(graph_backend);
        ggml_free(C);
        return false;
    }

    bool ok = set_tensor_from_gguf_f32(g, w) && set_tensor_from_gguf_f32(g, b);
    if (ok) {
        ggml_backend_tensor_set(x, in_whdc.data(), 0, in_whdc.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(gf, y);
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        ok = st == GGML_STATUS_SUCCESS;
        if (!ok) {
            std::fprintf(stderr, "vla(lingbot_va): VAE encoder conv_in graph failed (%d)\n", (int) st);
        }
    }
    if (ok) {
        out_whdc.assign((size_t) ggml_nelements(y), 0.0f);
        ggml_backend_tensor_get(y, out_whdc.data(), 0, out_whdc.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    lingbot_aux_graph_backend_free(graph_backend);
    ggml_free(C);
    return ok;
}





bool vae_down_block0_ggml_execute(
        gguf_reader & g,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int in_C,
        std::vector<float> & out_whdc,
        int * out_W,
        int * out_H) {
    const int Cc = 160;
    if (W <= 0 || H <= 0 || T <= 0 || in_C != 12) return false;
    if (in_whdc.size() != (size_t) W * H * T * in_C) return false;

    ggml_init_params params = { size_t(384) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(params);
    if (!C) return false;

    auto new_conv3_w = [&](const std::string & name, int64_t out_ch, int64_t in_ch,
                           int64_t kt = 3, int64_t kh = 3, int64_t kw = 3) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, kw, kh, kt, out_ch * in_ch);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_conv2_w = [&](const std::string & name, int64_t out_ch, int64_t in_ch,
                           int64_t kh = 3, int64_t kw = 3) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, kw, kh, in_ch, out_ch);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_bias3 = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, 1, channels);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_bias2 = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, channels, 1);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_gamma = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, channels, 1, 1, 1);
        ggml_set_name(t, name.c_str());
        return t;
    };

    struct ResW {
        ggml_tensor * n1 = nullptr;
        ggml_tensor * c1w = nullptr;
        ggml_tensor * c1b = nullptr;
        ggml_tensor * n2 = nullptr;
        ggml_tensor * c2w = nullptr;
        ggml_tensor * c2b = nullptr;
    };
    auto make_res = [&](const std::string & prefix) {
        ResW r;
        r.n1  = new_gamma(prefix + ".norm1.gamma", Cc);
        r.c1w = new_conv3_w(prefix + ".conv1.weight", Cc, Cc);
        r.c1b = new_bias3(prefix + ".conv1.bias", Cc);
        r.n2  = new_gamma(prefix + ".norm2.gamma", Cc);
        r.c2w = new_conv3_w(prefix + ".conv2.weight", Cc, Cc);
        r.c2b = new_bias3(prefix + ".conv2.bias", Cc);
        return r;
    };

    ggml_tensor * conv_in_w = new_conv3_w("vae.encoder.conv_in.weight", Cc, in_C);
    ggml_tensor * conv_in_b = new_bias3("vae.encoder.conv_in.bias", Cc);
    ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, W, H, T, in_C);
    ggml_set_name(x, "vae.down_block0.exec.input");
    ResW r0 = make_res("vae.encoder.down_blocks.0.resnets.0");
    ResW r1 = make_res("vae.encoder.down_blocks.0.resnets.1");
    ggml_tensor * down_w = new_conv2_w("vae.encoder.down_blocks.0.downsampler.resample.1.weight", Cc, Cc);
    ggml_tensor * down_b = new_bias2("vae.encoder.down_blocks.0.downsampler.resample.1.bias", Cc);

    auto apply_res = [&](ggml_tensor * h, const ResW & r) {
        ggml_tensor * residual = h;
        h = vae_norm_silu_to_conv_layout(C, h, r.n1);
        h = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, r.c1w, h, Cc), r.c1b);
        h = vae_norm_silu_to_conv_layout(C, h, r.n2);
        h = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, r.c2w, h, Cc), r.c2b);
        return ggml_add(C, h, residual);
    };

    ggml_tensor * h = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, conv_in_w, x, in_C), conv_in_b);
    h = apply_res(h, r0);
    h = apply_res(h, r1);
    ggml_tensor * h_2d = ggml_cont(C, ggml_permute(C, h, 0, 1, 3, 2));
    ggml_tensor * down = vae_spatial_downsample_conv2d(C, down_w, h_2d);
    down = ggml_add(C, down, down_b);
    ggml_tensor * out = ggml_cont(C, ggml_permute(C, down, 0, 1, 3, 2));
    ggml_set_name(out, "vae.down_block0.exec.output");
    ggml_set_output(out);

    LingBotAuxGraphBackend graph_backend;
    if (!lingbot_aux_graph_backend_init(graph_backend, "vae.ggml")) {
        ggml_free(C);
        return false;
    }
    ggml_backend_t backend = graph_backend.backend;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, graph_backend.buft);
    if (!buf) {
        lingbot_aux_graph_backend_free(graph_backend);
        ggml_free(C);
        return false;
    }

    std::vector<ggml_tensor *> weights = {
        conv_in_w, conv_in_b,
        r0.n1, r0.c1w, r0.c1b, r0.n2, r0.c2w, r0.c2b,
        r1.n1, r1.c1w, r1.c1b, r1.n2, r1.c2w, r1.c2b,
        down_w, down_b,
    };
    bool ok = true;
    for (ggml_tensor * t : weights) {
        if (!set_tensor_from_gguf_f32(g, t)) {
            ok = false;
            break;
        }
    }
    if (ok) {
        ggml_backend_tensor_set(x, in_whdc.data(), 0, in_whdc.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(C, 65536, false);
        ggml_build_forward_expand(gf, out);
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        ok = st == GGML_STATUS_SUCCESS;
        if (!ok) {
            std::fprintf(stderr, "vla(lingbot_va): VAE down block0 executor graph failed (%d)\n", (int) st);
        }
    }
    if (ok) {
        out_whdc.assign((size_t) ggml_nelements(out), 0.0f);
        ggml_backend_tensor_get(out, out_whdc.data(), 0, out_whdc.size() * sizeof(float));
        if (out_W) *out_W = (int) out->ne[0];
        if (out_H) *out_H = (int) out->ne[1];
    }

    ggml_backend_buffer_free(buf);
    lingbot_aux_graph_backend_free(graph_backend);
    ggml_free(C);
    return ok;
}

bool vae_spatial_downsample_ggml_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        std::vector<float> & out_whdc,
        int * out_W,
        int * out_H) {
    if (W <= 0 || H <= 0 || T <= 0 || Cc <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * Cc) return false;

    ggml_init_params params = { size_t(96) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(params);
    if (!C) return false;

    const std::string w_name = std::string(prefix) + ".resample.1.weight";
    const std::string b_name = std::string(prefix) + ".resample.1.bias";
    ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, W, H, T, Cc);
    ggml_set_name(x, "vae.downsampler.spatial.input");
    ggml_tensor * w = ggml_new_tensor_4d(C, GGML_TYPE_F32, 3, 3, Cc, Cc);
    ggml_set_name(w, w_name.c_str());
    ggml_tensor * b = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, Cc, 1);
    ggml_set_name(b, b_name.c_str());

    ggml_tensor * x_w_h_c_t = ggml_cont(C, ggml_permute(C, x, 0, 1, 3, 2));
    ggml_tensor * down = vae_spatial_downsample_conv2d(C, w, x_w_h_c_t);
    down = ggml_add(C, down, b);
    ggml_tensor * out = ggml_cont(C, ggml_permute(C, down, 0, 1, 3, 2));
    ggml_set_name(out, "vae.downsampler.spatial.output");
    ggml_set_output(out);

    LingBotAuxGraphBackend graph_backend;
    if (!lingbot_aux_graph_backend_init(graph_backend, "vae.ggml")) {
        ggml_free(C);
        return false;
    }
    ggml_backend_t backend = graph_backend.backend;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, graph_backend.buft);
    if (!buf) {
        lingbot_aux_graph_backend_free(graph_backend);
        ggml_free(C);
        return false;
    }

    bool ok = set_tensor_from_gguf_f32(g, w) && set_tensor_from_gguf_f32(g, b);
    if (ok) {
        ggml_backend_tensor_set(x, in_whdc.data(), 0, in_whdc.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(gf, out);
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        ok = st == GGML_STATUS_SUCCESS;
        if (!ok) {
            std::fprintf(stderr, "vla(lingbot_va): VAE spatial downsampler graph failed (%d) for %s\n",
                         (int) st, prefix);
        }
    }
    if (ok) {
        out_whdc.assign((size_t) ggml_nelements(out), 0.0f);
        ggml_backend_tensor_get(out, out_whdc.data(), 0, out_whdc.size() * sizeof(float));
        if (out_W) *out_W = (int) out->ne[0];
        if (out_H) *out_H = (int) out->ne[1];
    }

    ggml_backend_buffer_free(buf);
    lingbot_aux_graph_backend_free(graph_backend);
    ggml_free(C);
    return ok;
}

bool vae_down_block_resnets_spatial_ggml_execute(
        gguf_reader & g,
        const char * block_prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        std::vector<float> & out_whdc,
        int * out_W,
        int * out_H) {
    if (W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * in_C) return false;

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_cuda_requested() &&
        !lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_SPATIAL_DISABLE")) {
        const std::string bp(block_prefix);
        auto need = [&](const std::string & name, int64_t elems) -> const float * {
            return lingbot_cuda_cached_f32_tensor(g, name, elems);
        };
        struct ResPtrs {
            const float * n1 = nullptr;
            const float * c1w = nullptr;
            const float * c1b = nullptr;
            const float * n2 = nullptr;
            const float * c2w = nullptr;
            const float * c2b = nullptr;
            const float * scw = nullptr;
            const float * scb = nullptr;
            int in_ch = 0;
            int out_ch = 0;
        };
        auto make_res = [&](const std::string & prefix, int rin, int rout, bool shortcut) {
            ResPtrs r;
            r.in_ch = rin;
            r.out_ch = rout;
            r.n1 = need(prefix + ".norm1.gamma", rin);
            r.c1w = need(prefix + ".conv1.weight", (int64_t) rout * rin * 3 * 3 * 3);
            r.c1b = need(prefix + ".conv1.bias", rout);
            r.n2 = need(prefix + ".norm2.gamma", rout);
            r.c2w = need(prefix + ".conv2.weight", (int64_t) rout * rout * 3 * 3 * 3);
            r.c2b = need(prefix + ".conv2.bias", rout);
            if (shortcut) {
                r.scw = need(prefix + ".conv_shortcut.weight", (int64_t) rout * rin);
                r.scb = need(prefix + ".conv_shortcut.bias", rout);
            }
            return r;
        };
        ResPtrs r0 = make_res(bp + ".resnets.0", in_C, out_C, in_C != out_C);
        ResPtrs r1 = make_res(bp + ".resnets.1", out_C, out_C, false);
        const float * down_w = need(bp + ".downsampler.resample.1.weight",
                                    (int64_t) out_C * out_C * 3 * 3);
        const float * down_b = need(bp + ".downsampler.resample.1.bias", out_C);
        const bool have = r0.n1 && r0.c1w && r0.c1b && r0.n2 && r0.c2w && r0.c2b &&
                          ((r0.scw != nullptr) == (r0.scb != nullptr)) &&
                          r1.n1 && r1.c1w && r1.c1b && r1.n2 && r1.c2w && r1.c2b &&
                          down_w && down_b;
        if (have) {
            float * d_in = nullptr;
            float * d_a = nullptr;
            float * d_b = nullptr;
            float * d_sc = nullptr;
            float * d_down = nullptr;
            const size_t in_elems = in_whdc.size();
            const size_t pre_elems = (size_t) W * (size_t) H * (size_t) T * (size_t) out_C;
            const int out_W_cuda = (W + 1) / 2;
            const int out_H_cuda = (H + 1) / 2;
            const size_t down_elems = (size_t) out_W_cuda * (size_t) out_H_cuda * (size_t) T * (size_t) out_C;
            bool ok = cudaMalloc(reinterpret_cast<void **>(&d_in), in_elems * sizeof(float)) == cudaSuccess &&
                      cudaMalloc(reinterpret_cast<void **>(&d_a), std::max(in_elems, pre_elems) * sizeof(float)) == cudaSuccess &&
                      cudaMalloc(reinterpret_cast<void **>(&d_b), pre_elems * sizeof(float)) == cudaSuccess &&
                      cudaMalloc(reinterpret_cast<void **>(&d_sc), pre_elems * sizeof(float)) == cudaSuccess &&
                      cudaMalloc(reinterpret_cast<void **>(&d_down), down_elems * sizeof(float)) == cudaSuccess &&
                      cudaMemcpy(d_in, in_whdc.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
            std::string dump_tag(block_prefix);
            const std::string dump_needle = "vae.encoder.down_blocks.";
            const size_t dump_pos = dump_tag.find(dump_needle);
            if (dump_pos != std::string::npos) dump_tag = "block" + dump_tag.substr(dump_pos + dump_needle.size());
            auto dump_dev = [&](const std::string & suffix, const float * ptr, size_t elems, int dW, int dH, int dT, int dC) {
                if (!std::getenv("VLA_LINGBOT_VAE_DUMP_DIR") || !ptr || elems == 0) return;
                std::vector<float> tmp(elems, 0.0f);
                if (cudaDeviceSynchronize() != cudaSuccess ||
                    cudaMemcpy(tmp.data(), ptr, elems * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                    std::fprintf(stderr, "vla(lingbot_va): VAE CUDA dump failed for %s_%s\n",
                                 dump_tag.c_str(), suffix.c_str());
                    return;
                }
                lingbot_dump_f32_stage("vae_encoder", dump_tag + "_" + suffix, tmp, dW, dH, dT, dC);
            };
            auto apply_res = [&](const float * input, float * output, const ResPtrs & r,
                                 int cur_C, size_t elems, const char * res_label) {
                const bool preserve_residual = (input == d_a || input == d_b);
                const float * residual_input = input;
                if (preserve_residual) {
                    if (r.scw) {
                        return false;
                    }
                    if (cudaMemcpy(d_sc, input, elems * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess) {
                        return false;
                    }
                    residual_input = d_sc;
                }
                if (lingbot_vae_norm_silu_whdc_f32w(input, r.n1, d_a, W, H, T, cur_C, nullptr) != 0) {
                    return false;
                }
                dump_dev(std::string(res_label) + "_norm1_silu", d_a, (size_t) W * H * T * cur_C, W, H, T, cur_C);
                if (lingbot_vae_causal_conv3d_ks3_whdc_f32w(d_a, r.c1w, r.c1b, d_b,
                                                                W, H, T, cur_C, r.out_ch, nullptr) != 0) {
                    return false;
                }
                dump_dev(std::string(res_label) + "_conv1", d_b, elems, W, H, T, r.out_ch);
                if (lingbot_vae_norm_silu_whdc_f32w(d_b, r.n2, d_a, W, H, T, r.out_ch, nullptr) != 0 ||
                    lingbot_vae_causal_conv3d_ks3_whdc_f32w(d_a, r.c2w, r.c2b, d_b,
                                                                W, H, T, r.out_ch, r.out_ch, nullptr) != 0) {
                    return false;
                }
                const float * residual = residual_input;
                if (r.scw) {
                    if (lingbot_vae_conv1x1x1_whdc_f32w(input, r.scw, r.scb, d_sc,
                                                            W, H, T, r.in_ch, r.out_ch, nullptr) != 0) {
                        return false;
                    }
                    residual = d_sc;
                }
                const bool added = lingbot_vae_add_whdc_f32(d_b, residual, output, elems, nullptr) == 0;
                if (added) dump_dev(res_label, output, elems, W, H, T, r.out_ch);
                return added;
            };
            if (ok) {
                ok = apply_res(d_in, d_a, r0, in_C, pre_elems, "res0") &&
                     apply_res(d_a, d_b, r1, out_C, pre_elems, "res1") &&
                     lingbot_vae_spatial_downsample2d_whdc_f32w(d_b, down_w, down_b, d_down,
                                                                    W, H, T, out_C, nullptr) == 0 &&
                     cudaDeviceSynchronize() == cudaSuccess;
            }
            if (ok) {
                out_whdc.assign(down_elems, 0.0f);
                ok = cudaMemcpy(out_whdc.data(), d_down, down_elems * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
            }
            if (ok && std::getenv("VLA_LINGBOT_VAE_DUMP_DIR")) {
                std::string tag(block_prefix);
                const std::string needle = "vae.encoder.down_blocks.";
                const size_t pos = tag.find(needle);
                if (pos != std::string::npos) tag = "block" + tag.substr(pos + needle.size());
                lingbot_dump_f32_stage("vae_encoder", tag + "_main_down", out_whdc,
                                       out_W_cuda, out_H_cuda, T, out_C);
            }
            if (d_in) cudaFree(d_in);
            if (d_a) cudaFree(d_a);
            if (d_b) cudaFree(d_b);
            if (d_sc) cudaFree(d_sc);
            if (d_down) cudaFree(d_down);
            if (ok) {
                if (out_W) *out_W = out_W_cuda;
                if (out_H) *out_H = out_H_cuda;
                if (std::getenv("VLA_LINGBOT_TIMING")) {
                    std::printf("vla(lingbot_va): VAE %s CUDA spatial path ok in=[%d,%d,%d,%d] out=[%d,%d,%d,%d]\n",
                                block_prefix, W, H, T, in_C, out_W_cuda, out_H_cuda, T, out_C);
                }
                return true;
            }
            std::fprintf(stderr, "vla(lingbot_va): VAE %s CUDA spatial path failed; falling back to ggml CPU\n",
                         block_prefix);
        }
    }
#endif

    ggml_init_params params = { size_t(768) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(params);
    if (!C) return false;

    auto new_conv3_w = [&](const std::string & name, int64_t out_ch, int64_t in_ch,
                           int64_t kt = 3, int64_t kh = 3, int64_t kw = 3) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, kw, kh, kt, out_ch * in_ch);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_conv2_w = [&](const std::string & name, int64_t out_ch, int64_t in_ch,
                           int64_t kh = 3, int64_t kw = 3) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, kw, kh, in_ch, out_ch);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_bias3 = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, 1, channels);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_bias2 = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, channels, 1);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_gamma = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, channels, 1, 1, 1);
        ggml_set_name(t, name.c_str());
        return t;
    };

    struct ResW {
        ggml_tensor * n1 = nullptr;
        ggml_tensor * c1w = nullptr;
        ggml_tensor * c1b = nullptr;
        ggml_tensor * n2 = nullptr;
        ggml_tensor * c2w = nullptr;
        ggml_tensor * c2b = nullptr;
        ggml_tensor * scw = nullptr;
        ggml_tensor * scb = nullptr;
        int64_t in_ch = 0;
        int64_t out_ch = 0;
    };
    auto make_res = [&](const std::string & prefix, int64_t rin, int64_t rout, bool shortcut) {
        ResW r;
        r.in_ch = rin;
        r.out_ch = rout;
        r.n1  = new_gamma(prefix + ".norm1.gamma", rin);
        r.c1w = new_conv3_w(prefix + ".conv1.weight", rout, rin);
        r.c1b = new_bias3(prefix + ".conv1.bias", rout);
        r.n2  = new_gamma(prefix + ".norm2.gamma", rout);
        r.c2w = new_conv3_w(prefix + ".conv2.weight", rout, rout);
        r.c2b = new_bias3(prefix + ".conv2.bias", rout);
        if (shortcut) {
            r.scw = new_conv3_w(prefix + ".conv_shortcut.weight", rout, rin, 1, 1, 1);
            r.scb = new_bias3(prefix + ".conv_shortcut.bias", rout);
        }
        return r;
    };

    const std::string bp(block_prefix);
    ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, W, H, T, in_C);
    ggml_set_name(x, "vae.down_block.full.input");
    ResW r0 = make_res(bp + ".resnets.0", in_C, out_C, in_C != out_C);
    ResW r1 = make_res(bp + ".resnets.1", out_C, out_C, false);
    ggml_tensor * down_w = new_conv2_w(bp + ".downsampler.resample.1.weight", out_C, out_C);
    ggml_tensor * down_b = new_bias2(bp + ".downsampler.resample.1.bias", out_C);

    auto apply_res = [&](ggml_tensor * h, const ResW & r, ggml_tensor ** out_norm1, ggml_tensor ** out_conv1) {
        ggml_tensor * residual = h;
        h = vae_norm_silu_to_conv_layout(C, h, r.n1);
        if (out_norm1) *out_norm1 = h;
        h = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, r.c1w, h, r.in_ch), r.c1b);
        if (out_conv1) *out_conv1 = h;
        h = vae_norm_silu_to_conv_layout(C, h, r.n2);
        h = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, r.c2w, h, r.out_ch), r.c2b);
        if (r.scw) {
            residual = ggml_add(C, ggml_conv_3d(C, r.scw, residual, r.in_ch, 1, 1, 1, 0, 0, 0, 1, 1, 1), r.scb);
        }
        return ggml_add(C, h, residual);
    };

    ggml_tensor * h0_norm1 = nullptr;
    ggml_tensor * h0_conv1 = nullptr;
    ggml_tensor * h1_norm1 = nullptr;
    ggml_tensor * h1_conv1 = nullptr;
    ggml_tensor * h0 = apply_res(x, r0, &h0_norm1, &h0_conv1);
    ggml_tensor * h1 = apply_res(h0, r1, &h1_norm1, &h1_conv1);
    ggml_tensor * h_2d = ggml_cont(C, ggml_permute(C, h1, 0, 1, 3, 2));
    ggml_tensor * down = vae_spatial_downsample_conv2d(C, down_w, h_2d);
    down = ggml_add(C, down, down_b);
    ggml_tensor * out = ggml_cont(C, ggml_permute(C, down, 0, 1, 3, 2));
    ggml_set_name(out, "vae.down_block.full.spatial_output");
    ggml_set_output(h0_norm1);
    ggml_set_output(h0_conv1);
    ggml_set_output(h0);
    ggml_set_output(h1_norm1);
    ggml_set_output(h1_conv1);
    ggml_set_output(h1);
    ggml_set_output(out);

    LingBotAuxGraphBackend graph_backend;
    if (!lingbot_aux_graph_backend_init(graph_backend, "vae.ggml")) {
        ggml_free(C);
        return false;
    }
    ggml_backend_t backend = graph_backend.backend;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, graph_backend.buft);
    if (!buf) {
        lingbot_aux_graph_backend_free(graph_backend);
        ggml_free(C);
        return false;
    }

    std::vector<ggml_tensor *> weights = {
        r0.n1, r0.c1w, r0.c1b, r0.n2, r0.c2w, r0.c2b,
        r1.n1, r1.c1w, r1.c1b, r1.n2, r1.c2w, r1.c2b,
        down_w, down_b,
    };
    if (r0.scw) {
        weights.push_back(r0.scw);
        weights.push_back(r0.scb);
    }
    bool ok = true;
    for (ggml_tensor * t : weights) {
        if (!set_tensor_from_gguf_f32(g, t)) {
            ok = false;
            break;
        }
    }
    if (ok) {
        ggml_backend_tensor_set(x, in_whdc.data(), 0, in_whdc.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(C, 65536, false);
        ggml_build_forward_expand(gf, out);
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        ok = st == GGML_STATUS_SUCCESS;
        if (!ok) {
            std::fprintf(stderr, "vla(lingbot_va): VAE down block graph failed (%d) for %s\n",
                         (int) st, block_prefix);
        }
    }
    if (ok) {
        out_whdc.assign((size_t) ggml_nelements(out), 0.0f);
        ggml_backend_tensor_get(out, out_whdc.data(), 0, out_whdc.size() * sizeof(float));
        if (const char * dump_dir = std::getenv("VLA_LINGBOT_VAE_DUMP_DIR")) {
            std::string tag(block_prefix);
            const std::string needle = "vae.encoder.down_blocks.";
            const size_t pos = tag.find(needle);
            if (pos != std::string::npos) tag = "block" + tag.substr(pos + needle.size());
            auto dump_tensor = [&](const std::string & suffix, ggml_tensor * t, int dW, int dH, int dT, int dC) {
                std::vector<float> tmp((size_t) ggml_nelements(t), 0.0f);
                ggml_backend_tensor_get(t, tmp.data(), 0, tmp.size() * sizeof(float));
                const std::string base = std::string(dump_dir) + "/vae_encoder_" + tag + "_" + suffix;
                std::ofstream f32(base + ".f32", std::ios::binary);
                if (f32) f32.write(reinterpret_cast<const char *>(tmp.data()), (std::streamsize) (tmp.size() * sizeof(float)));
                std::ofstream shape(base + ".shape.txt");
                if (shape) shape << dW << " " << dH << " " << dT << " " << dC << "\n";
            };
            dump_tensor("res0_norm1_silu", h0_norm1, W, H, T, in_C);
            dump_tensor("res0_conv1", h0_conv1, W, H, T, out_C);
            dump_tensor("res0", h0, W, H, T, out_C);
            dump_tensor("res1_norm1_silu", h1_norm1, W, H, T, out_C);
            dump_tensor("res1_conv1", h1_conv1, W, H, T, out_C);
            dump_tensor("res1", h1, W, H, T, out_C);
            dump_tensor("main_down", out, (int) out->ne[0], (int) out->ne[1], (int) out->ne[2], out_C);
        }
        if (out_W) *out_W = (int) out->ne[0];
        if (out_H) *out_H = (int) out->ne[1];
    }

    ggml_backend_buffer_free(buf);
    lingbot_aux_graph_backend_free(graph_backend);
    ggml_free(C);
    return ok;
}

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS




struct LingBotVaeCudaTemporalConvCache {
    int C = 0;
    int K = 0;
    float * past = nullptr;
    float * next = nullptr;

    bool init(int channels, int kernel) {
        C = channels;
        K = kernel;
        const size_t bytes = (size_t) (K - 1) * C * sizeof(float);
        if (cudaMalloc(&past, bytes) != cudaSuccess || cudaMalloc(&next, bytes) != cudaSuccess) {
            release();
            return false;
        }
        return cudaMemset(past, 0, bytes) == cudaSuccess &&
               cudaMemset(next, 0, bytes) == cudaSuccess;
    }

    bool step(
            const float * x,
            const float * w,
            const float * b,
            float * out,
            int T) {
        if (!past || !next || T <= 0) return false;
        if (lingbot_causal_conv1d_cache_f32(x, past, w, b, out, next, T, C, C, K, 0) != 0) {
            return false;
        }
        std::swap(past, next);
        return true;
    }

    bool read_cache(std::vector<float> & out) const {
        out.assign((size_t) (K - 1) * C, 0.0f);
        return cudaMemcpy(out.data(), past, out.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    void release() {
        if (past) cudaFree(past);
        if (next) cudaFree(next);
        past = nullptr;
        next = nullptr;
        C = 0;
        K = 0;
    }

    ~LingBotVaeCudaTemporalConvCache() {
        release();
    }
};

void vae_time_conv_ref(
        const std::vector<float> & x,
        const std::vector<float> & past,
        const std::vector<float> & w,
        const std::vector<float> & b,
        std::vector<float> & out,
        std::vector<float> & next,
        int T,
        int C,
        int K) {
    out.assign((size_t) T * C, 0.0f);
    for (int t = 0; t < T; ++t) {
        for (int co = 0; co < C; ++co) {
            float acc = b[(size_t) co];
            for (int k = 0; k < K; ++k) {
                const int src_t = t + k - (K - 1);
                for (int ci = 0; ci < C; ++ci) {
                    const float xv = src_t < 0
                        ? past[(size_t) (src_t + K - 1) * C + ci]
                        : x[(size_t) src_t * C + ci];
                    acc += xv * w[((size_t) co * C + ci) * K + k];
                }
            }
            out[(size_t) t * C + co] = acc;
        }
    }
    next.assign((size_t) (K - 1) * C, 0.0f);
    for (int j = 0; j < K - 1; ++j) {
        const int src_t = T + j - (K - 1);
        for (int ci = 0; ci < C; ++ci) {
            next[(size_t) j * C + ci] = src_t < 0
                ? past[(size_t) (src_t + K - 1) * C + ci]
                : x[(size_t) src_t * C + ci];
        }
    }
}





struct LingBotVaeCudaTemporalConvBatchedCache {
    int lanes = 0;
    int C = 0;
    int K = 0;
    float * past = nullptr;
    float * next = nullptr;

    bool init(int lanes_, int channels, int kernel) {
        lanes = lanes_;
        C = channels;
        K = kernel;
        const size_t bytes = (size_t) lanes * (K - 1) * C * sizeof(float);
        if (cudaMalloc(&past, bytes) != cudaSuccess || cudaMalloc(&next, bytes) != cudaSuccess) {
            release();
            return false;
        }
        return cudaMemset(past, 0, bytes) == cudaSuccess &&
               cudaMemset(next, 0, bytes) == cudaSuccess;
    }

    bool step(
            const float * x,
            const float * w,
            const float * b,
            float * out,
            int T) {
        if (!past || !next || lanes <= 0 || T <= 0) return false;
        if (lingbot_causal_conv1d_cache_f32_batched(x, past, w, b, out, next, lanes, T, C, C, K, 0) != 0) {
            return false;
        }
        std::swap(past, next);
        return true;
    }

    bool read_cache(std::vector<float> & out) const {
        out.assign((size_t) lanes * (K - 1) * C, 0.0f);
        return cudaMemcpy(out.data(), past, out.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    void release() {
        if (past) cudaFree(past);
        if (next) cudaFree(next);
        past = nullptr;
        next = nullptr;
        lanes = 0;
        C = 0;
        K = 0;
    }

    ~LingBotVaeCudaTemporalConvBatchedCache() {
        release();
    }
};

void vae_time_conv_batched_ref(
        const std::vector<float> & x,
        const std::vector<float> & past,
        const std::vector<float> & w,
        const std::vector<float> & b,
        std::vector<float> & out,
        std::vector<float> & next,
        int lanes,
        int T,
        int C,
        int K) {
    out.assign((size_t) lanes * T * C, 0.0f);
    for (int lane = 0; lane < lanes; ++lane) {
        for (int t = 0; t < T; ++t) {
            for (int co = 0; co < C; ++co) {
                float acc = b[(size_t) co];
                for (int k = 0; k < K; ++k) {
                    const int src_t = t + k - (K - 1);
                    for (int ci = 0; ci < C; ++ci) {
                        const float xv = src_t < 0
                            ? past[((size_t) lane * (K - 1) + (src_t + K - 1)) * C + ci]
                            : x[((size_t) lane * T + src_t) * C + ci];
                        acc += xv * w[((size_t) co * C + ci) * K + k];
                    }
                }
                out[((size_t) lane * T + t) * C + co] = acc;
            }
        }
    }
    next.assign((size_t) lanes * (K - 1) * C, 0.0f);
    for (int lane = 0; lane < lanes; ++lane) {
        for (int j = 0; j < K - 1; ++j) {
            const int src_t = T + j - (K - 1);
            for (int ci = 0; ci < C; ++ci) {
                next[((size_t) lane * (K - 1) + j) * C + ci] = src_t < 0
                    ? past[((size_t) lane * (K - 1) + (src_t + K - 1)) * C + ci]
                    : x[((size_t) lane * T + src_t) * C + ci];
            }
        }
    }
}





size_t vae_ggml_whdc_index(int w, int h, int t, int c, int W, int H, int T) {
    return (size_t) w + (size_t) W * ((size_t) h + (size_t) H * ((size_t) t + (size_t) T * c));
}

bool vae_slice_time_chunk(
        const std::vector<float> & in_whdc,
        std::vector<float> & chunk_whdc,
        int W,
        int H,
        int T_total,
        int C,
        int offset,
        int chunk);

void vae_append_time_chunk(
        std::vector<float> & out_whdc,
        const std::vector<float> & chunk_whdc,
        int W,
        int H,
        int T_total,
        int C,
        int offset,
        int chunk);

bool vae_avg_down3d_host(
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        int factor_t,
        int factor_s,
        int * out_W,
        int * out_H,
        int * out_T) {
    if (W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0 || factor_t <= 0 || factor_s <= 0) return false;
    if (W % factor_s != 0 || H % factor_s != 0) return false;
    const int factor = factor_t * factor_s * factor_s;
    if ((in_C * factor) % out_C != 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * in_C) return false;
    const int pad_t = (factor_t - (T % factor_t)) % factor_t;
    const int Tp = T + pad_t;
    const int Wo = W / factor_s;
    const int Ho = H / factor_s;
    const int To = Tp / factor_t;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_cuda_requested() &&
        !lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_AVG_DOWN_DISABLE")) {
        const size_t in_bytes = in_whdc.size() * sizeof(float);
        const size_t out_elems = (size_t) Wo * Ho * To * out_C;
        static std::mutex scratch_mu;
        static LingBotCudaF32Scratch scratch_in;
        static LingBotCudaF32Scratch scratch_out;
        std::lock_guard<std::mutex> scratch_lock(scratch_mu);
        bool ok = scratch_in.ensure(in_whdc.size()) &&
                  scratch_out.ensure(out_elems) &&
                  cudaMemcpy(scratch_in.ptr, in_whdc.data(), in_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                  lingbot_vae_avg_down3d_whdc_f32(
                      scratch_in.ptr, scratch_out.ptr, W, H, T, in_C, out_C, factor_t, factor_s, nullptr) == 0 &&
                  cudaDeviceSynchronize() == cudaSuccess;
        if (ok) {
            out_whdc.assign(out_elems, 0.0f);
            ok = cudaMemcpy(out_whdc.data(), scratch_out.ptr, out_elems * sizeof(float),
                            cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (ok) {
            if (out_W) *out_W = Wo;
            if (out_H) *out_H = Ho;
            if (out_T) *out_T = To;
            return true;
        }
        std::fprintf(stderr, "vla(lingbot_va): VAE AvgDown3D CUDA path failed; falling back to host\n");
    }
#endif
    const int group_size = (in_C * factor) / out_C;
    out_whdc.assign((size_t) Wo * Ho * To * out_C, 0.0f);
    for (int wo = 0; wo < Wo; ++wo) {
        for (int ho = 0; ho < Ho; ++ho) {
            for (int to = 0; to < To; ++to) {
                for (int co = 0; co < out_C; ++co) {
                    double acc = 0.0;
                    for (int gidx = 0; gidx < group_size; ++gidx) {
                        const int flat = co * group_size + gidx;
                        const int c = flat / factor;
                        const int rem0 = flat % factor;
                        const int ft = rem0 / (factor_s * factor_s);
                        const int rem1 = rem0 % (factor_s * factor_s);
                        const int fs_h = rem1 / factor_s;
                        const int fs_w = rem1 % factor_s;
                        const int src_t_padded = to * factor_t + ft;
                        const int src_t = src_t_padded - pad_t;
                        if (src_t >= 0 && src_t < T) {
                            const int src_w = wo * factor_s + fs_w;
                            const int src_h = ho * factor_s + fs_h;
                            acc += in_whdc[vae_ggml_whdc_index(src_w, src_h, src_t, c, W, H, T)];
                        }
                    }
                    out_whdc[vae_ggml_whdc_index(wo, ho, to, co, Wo, Ho, To)] = (float) (acc / (double) group_size);
                }
            }
        }
    }
    if (out_W) *out_W = Wo;
    if (out_H) *out_H = Ho;
    if (out_T) *out_T = To;
    return true;
}

bool vae_avg_down3d_stream_host(
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        int factor_t,
        int factor_s,
        const std::vector<int> & chunks,
        std::vector<int> * out_chunks,
        int * out_W,
        int * out_H,
        int * out_T) {
    int chunk_sum = 0;
    for (int c : chunks) {
        if (c <= 0) return false;
        chunk_sum += c;
    }
    if (chunk_sum != T || chunks.empty()) return false;

    std::vector<std::vector<float>> parts;
    parts.reserve(chunks.size());
    std::vector<int> local_chunks;
    local_chunks.reserve(chunks.size());
    int total_T = 0;
    int Wo = 0, Ho = 0, ref_W = 0, ref_H = 0;
    int offset = 0;
    for (int chunk : chunks) {
        std::vector<float> x_chunk;
        if (!vae_slice_time_chunk(in_whdc, x_chunk, W, H, T, in_C, offset, chunk)) return false;
        std::vector<float> y_chunk;
        int cw = 0, ch = 0, ct = 0;
        if (!vae_avg_down3d_host(x_chunk, y_chunk, W, H, chunk, in_C, out_C,
                                 factor_t, factor_s, &cw, &ch, &ct)) return false;
        if (parts.empty()) {
            ref_W = cw;
            ref_H = ch;
        } else if (cw != ref_W || ch != ref_H) {
            return false;
        }
        local_chunks.push_back(ct);
        total_T += ct;
        parts.push_back(std::move(y_chunk));
        offset += chunk;
    }
    Wo = ref_W;
    Ho = ref_H;
    out_whdc.assign((size_t) Wo * Ho * total_T * out_C, 0.0f);
    int dst = 0;
    for (size_t i = 0; i < parts.size(); ++i) {
        vae_append_time_chunk(out_whdc, parts[i], Wo, Ho, total_T, out_C, dst, local_chunks[i]);
        dst += local_chunks[i];
    }
    if (out_chunks) *out_chunks = std::move(local_chunks);
    if (out_W) *out_W = Wo;
    if (out_H) *out_H = Ho;
    if (out_T) *out_T = total_T;
    return true;
}

bool vae_dup_up3d_host(
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        int factor_t,
        int factor_s,
        bool first_chunk,
        int * out_W,
        int * out_H,
        int * out_T) {
    if (W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0 || factor_t <= 0 || factor_s <= 0) return false;
    const int factor = factor_t * factor_s * factor_s;
    if ((out_C * factor) % in_C != 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * in_C) return false;
    const int repeats = (out_C * factor) / in_C;
    const int Wo = W * factor_s;
    const int Ho = H * factor_s;
    const int Tfull = T * factor_t;
    const int trim = first_chunk ? factor_t - 1 : 0;
    const int To = Tfull - trim;
    out_whdc.assign((size_t) Wo * Ho * To * out_C, 0.0f);
    for (int w = 0; w < W; ++w) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                for (int ci = 0; ci < in_C; ++ci) {
                    const float v = in_whdc[vae_ggml_whdc_index(w, h, t, ci, W, H, T)];
                    for (int rep = 0; rep < repeats; ++rep) {
                        const int flat = ci * repeats + rep;
                        const int co = flat / factor;
                        const int rem0 = flat % factor;
                        const int ft = rem0 / (factor_s * factor_s);
                        const int rem1 = rem0 % (factor_s * factor_s);
                        const int fs_h = rem1 / factor_s;
                        const int fs_w = rem1 % factor_s;
                        const int ot_full = t * factor_t + ft;
                        const int ot = ot_full - trim;
                        if (ot >= 0 && ot < To) {
                            const int ow = w * factor_s + fs_w;
                            const int oh = h * factor_s + fs_h;
                            out_whdc[vae_ggml_whdc_index(ow, oh, ot, co, Wo, Ho, To)] = v;
                        }
                    }
                }
            }
        }
    }
    if (out_W) *out_W = Wo;
    if (out_H) *out_H = Ho;
    if (out_T) *out_T = To;
    return true;
}

bool vae_add_same_shape(std::vector<float> & dst, const std::vector<float> & src) {
    if (dst.size() != src.size()) return false;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_cuda_requested() &&
        !lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_ADD_DISABLE") &&
        dst.size() >= 4096) {
        const size_t bytes = dst.size() * sizeof(float);
        float * d_a = nullptr;
        float * d_b = nullptr;
        float * d_out = nullptr;
        bool ok = cudaMalloc(reinterpret_cast<void **>(&d_a), bytes) == cudaSuccess &&
                  cudaMalloc(reinterpret_cast<void **>(&d_b), bytes) == cudaSuccess &&
                  cudaMalloc(reinterpret_cast<void **>(&d_out), bytes) == cudaSuccess &&
                  cudaMemcpy(d_a, dst.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                  cudaMemcpy(d_b, src.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                  lingbot_vae_add_whdc_f32(d_a, d_b, d_out, dst.size(), nullptr) == 0 &&
                  cudaDeviceSynchronize() == cudaSuccess;
        if (ok) {
            ok = cudaMemcpy(dst.data(), d_out, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_out) cudaFree(d_out);
        if (ok) return true;
        std::fprintf(stderr, "vla(lingbot_va): VAE add CUDA path failed; falling back to host\n");
    }
#endif
    for (size_t i = 0; i < dst.size(); ++i) dst[i] += src[i];
    return true;
}

void vae_norm_silu_host(
        const std::vector<float> & in_whdc,
        const std::vector<float> & gamma,
        int W,
        int H,
        int T,
        int C,
        std::vector<float> & out_whdc) {
    if (W > 0 && H > 0 && T > 0 && C > 0 &&
        in_whdc.size() == (size_t) W * H * T * C &&
        gamma.size() == (size_t) C) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
        if (lingbot_vae_cuda_requested() &&
            !lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_NORM_SILU_DISABLE")) {
            const size_t bytes = in_whdc.size() * sizeof(float);
            float * d_in = nullptr;
            float * d_gamma = nullptr;
            float * d_out = nullptr;
            bool ok = cudaMalloc(reinterpret_cast<void **>(&d_in), bytes) == cudaSuccess &&
                      cudaMalloc(reinterpret_cast<void **>(&d_out), bytes) == cudaSuccess &&
                      cudaMalloc(reinterpret_cast<void **>(&d_gamma), gamma.size() * sizeof(float)) == cudaSuccess &&
                      cudaMemcpy(d_in, in_whdc.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                      cudaMemcpy(d_gamma, gamma.data(), gamma.size() * sizeof(float),
                                 cudaMemcpyHostToDevice) == cudaSuccess &&
                      lingbot_vae_norm_silu_whdc_f32w(
                          d_in, d_gamma, d_out, W, H, T, C, nullptr) == 0 &&
                      cudaDeviceSynchronize() == cudaSuccess;
            if (ok) {
                out_whdc.assign(in_whdc.size(), 0.0f);
                ok = cudaMemcpy(out_whdc.data(), d_out, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
            }
            if (d_in) cudaFree(d_in);
            if (d_gamma) cudaFree(d_gamma);
            if (d_out) cudaFree(d_out);
            if (ok) return;
            std::fprintf(stderr, "vla(lingbot_va): VAE norm+silu CUDA path failed; falling back to host\n");
        }
#endif
    }
    const bool aliased = &in_whdc == &out_whdc;
    const std::vector<float> input_copy = aliased ? in_whdc : std::vector<float>();
    const std::vector<float> & src_in = aliased ? input_copy : in_whdc;
    out_whdc.assign(in_whdc.size(), 0.0f);
    for (int w = 0; w < W; ++w) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                double ss = 0.0;
                for (int c = 0; c < C; ++c) {
                    const float v = src_in[vae_ggml_whdc_index(w, h, t, c, W, H, T)];
                    ss += (double) v * (double) v;
                }
                const double l2 = std::sqrt(ss);
                const double norm_scale = l2 > 1e-12 ? std::sqrt((double) C) / l2 : 0.0;
                for (int c = 0; c < C; ++c) {
                    const float v = src_in[vae_ggml_whdc_index(w, h, t, c, W, H, T)];
                    const double x = (double) v * norm_scale * (double) gamma[(size_t) c];
                    out_whdc[vae_ggml_whdc_index(w, h, t, c, W, H, T)] =
                        (float) (x / (1.0 + std::exp(-x)));
                }
            }
        }
    }
}

struct VaeTemporalCacheWHDC {
    int W = 0;
    int H = 0;
    int C = 0;
    int T = 0;
    std::vector<float> data;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    float * d_data = nullptr;
    size_t d_capacity = 0;
    int d_W = 0;
    int d_H = 0;
    int d_C = 0;
    int d_T = 0;
    bool d_valid = false;
#endif

    VaeTemporalCacheWHDC() = default;
    VaeTemporalCacheWHDC(const VaeTemporalCacheWHDC &) = delete;
    VaeTemporalCacheWHDC & operator=(const VaeTemporalCacheWHDC &) = delete;

    VaeTemporalCacheWHDC(VaeTemporalCacheWHDC && other) noexcept {
        move_from(std::move(other));
    }

    VaeTemporalCacheWHDC & operator=(VaeTemporalCacheWHDC && other) noexcept {
        if (this != &other) {
            release_device();
            W = other.W;
            H = other.H;
            C = other.C;
            T = other.T;
            data = std::move(other.data);
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            d_data = other.d_data;
            d_capacity = other.d_capacity;
            d_W = other.d_W;
            d_H = other.d_H;
            d_C = other.d_C;
            d_T = other.d_T;
            d_valid = other.d_valid;
            other.d_data = nullptr;
            other.d_capacity = 0;
            other.d_valid = false;
            other.d_W = other.d_H = other.d_C = other.d_T = 0;
#endif
        }
        return *this;
    }

    ~VaeTemporalCacheWHDC() {
        release_device();
    }

    void move_from(VaeTemporalCacheWHDC && other) noexcept {
        W = other.W;
        H = other.H;
        C = other.C;
        T = other.T;
        data = std::move(other.data);
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
        d_data = other.d_data;
        d_capacity = other.d_capacity;
        d_W = other.d_W;
        d_H = other.d_H;
        d_C = other.d_C;
        d_T = other.d_T;
        d_valid = other.d_valid;
        other.d_data = nullptr;
        other.d_capacity = 0;
        other.d_valid = false;
        other.d_W = other.d_H = other.d_C = other.d_T = 0;
#endif
    }

    void release_device() {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
        if (d_data) cudaFree(d_data);
        d_data = nullptr;
        d_capacity = 0;
        d_W = d_H = d_C = d_T = 0;
        d_valid = false;
#endif
    }

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    bool ensure_device(size_t elems) {
        if (elems == 0) return false;
        if (d_data && d_capacity >= elems) return true;
        if (d_data) {
            cudaFree(d_data);
            d_data = nullptr;
            d_capacity = 0;
            d_valid = false;
        }
        if (cudaMalloc(reinterpret_cast<void **>(&d_data), elems * sizeof(float)) != cudaSuccess) {
            d_data = nullptr;
            return false;
        }
        d_capacity = elems;
        d_valid = false;
        return true;
    }

    bool upload_device_from_host() {
        if (data.empty() || W <= 0 || H <= 0 || T <= 0 || C <= 0) {
            d_valid = false;
            return false;
        }
        if (!ensure_device(data.size())) return false;
        if (cudaMemcpy(d_data, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            d_valid = false;
            return false;
        }
        d_W = W;
        d_H = H;
        d_C = C;
        d_T = T;
        d_valid = true;
        return true;
    }

    const float * device_ptr_or_upload() {
        if (T <= 0) return nullptr;
        if (d_valid && d_data && d_W == W && d_H == H && d_C == C && d_T == T) {
            return d_data;
        }
        return upload_device_from_host() ? d_data : nullptr;
    }

    bool device_valid_for_shape(int w, int h, int t, int c) const {
        return d_valid && d_data && d_W == w && d_H == h && d_T == t && d_C == c;
    }

    bool update_device_from_chunk(const float * d_chunk,
                                  int prev_W,
                                  int prev_H,
                                  int prev_T,
                                  int prev_C,
                                  int w,
                                  int h,
                                  int t,
                                  int c) {
        if (!d_chunk || W <= 0 || H <= 0 || T <= 0 || C <= 0 || data.empty()) {
            d_valid = false;
            return false;
        }
        const bool prepend_prev = t < 2 && prev_T > 0 &&
                                  prev_W == w && prev_H == h && prev_C == c;
        if (prepend_prev &&
            !(d_valid && d_data && d_W == prev_W && d_H == prev_H &&
              d_C == prev_C && d_T == prev_T)) {
            return upload_device_from_host();
        }
        if (!ensure_device(data.size())) return false;
        if (lingbot_vae_update_temporal_cache_last_whdc_f32(
                d_chunk, d_data, w, h, t, prepend_prev ? prev_T : 0, T, c, nullptr) != 0) {
            d_valid = false;
            return false;
        }
        d_W = W;
        d_H = H;
        d_C = C;
        d_T = T;
        d_valid = true;
        return true;
    }

    bool update_from_device_chunk(const float * d_chunk, int w, int h, int t, int c) {
        const int prev_T = T;
        const int prev_W = W;
        const int prev_H = H;
        const int prev_C = C;
        if (!d_chunk || w <= 0 || h <= 0 || t <= 0 || c <= 0) return false;
        const bool prepend_prev = t < 2 && prev_T > 0 &&
                                  prev_W == w && prev_H == h && prev_C == c;
        const int new_T = std::min(2, t + (prepend_prev ? 1 : 0));
        if (prepend_prev && prev_T != new_T) {
            return false;
        }
        W = w;
        H = h;
        C = c;
        T = new_T;
        data.assign((size_t) W * H * T * C, 0.0f);
        const size_t reserve_elems = (size_t) W * H * std::max(2, T) * C;
        if (!ensure_device(reserve_elems)) {
            d_valid = false;
            return false;
        }
        if (lingbot_vae_update_temporal_cache_last_whdc_f32(
                d_chunk, d_data, w, h, t, prepend_prev ? prev_T : 0, T, c, nullptr) != 0 ||
            cudaMemcpy(data.data(), d_data, data.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            d_valid = false;
            return false;
        }
        d_W = W;
        d_H = H;
        d_C = C;
        d_T = T;
        d_valid = true;
        return true;
    }

    bool update_from_device_chunk_device_only(const float * d_chunk, int w, int h, int t, int c) {
        const int prev_T = T;
        const int prev_W = W;
        const int prev_H = H;
        const int prev_C = C;
        if (!d_chunk || w <= 0 || h <= 0 || t <= 0 || c <= 0) return false;
        const bool prepend_prev = t < 2 && prev_T > 0 &&
                                  prev_W == w && prev_H == h && prev_C == c;
        const int new_T = std::min(2, t + (prepend_prev ? 1 : 0));
        if (prepend_prev &&
            !(d_valid && d_data && d_W == prev_W && d_H == prev_H &&
              d_C == prev_C && d_T == prev_T)) {
            return false;
        }
        W = w;
        H = h;
        C = c;
        T = new_T;
        data.clear();
        const size_t reserve_elems = (size_t) W * H * std::max(2, T) * C;
        if (!ensure_device(reserve_elems)) {
            d_valid = false;
            return false;
        }
        if (lingbot_vae_update_temporal_cache_last_whdc_f32(
                d_chunk, d_data, w, h, t, prepend_prev ? prev_T : 0, T, c, nullptr) != 0) {
            d_valid = false;
            return false;
        }
        d_W = W;
        d_H = H;
        d_C = C;
        d_T = T;
        d_valid = true;
        return true;
    }

    bool update_from_device_chunk_last_frame(const float * d_chunk, int w, int h, int t, int c) {
        if (!d_chunk || w <= 0 || h <= 0 || t <= 0 || c <= 0) return false;
        W = w;
        H = h;
        C = c;
        T = 1;
        data.assign((size_t) W * H * C, 0.0f);
        if (!ensure_device(data.size())) {
            d_valid = false;
            return false;
        }
        if (lingbot_vae_update_temporal_cache_last_whdc_f32(
                d_chunk, d_data, w, h, t, 0, T, c, nullptr) != 0 ||
            cudaMemcpy(data.data(), d_data, data.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            d_valid = false;
            return false;
        }
        d_W = W;
        d_H = H;
        d_C = C;
        d_T = T;
        d_valid = true;
        return true;
    }

    bool update_from_device_chunk_last_frame_device_only(const float * d_chunk, int w, int h, int t, int c) {
        if (!d_chunk || w <= 0 || h <= 0 || t <= 0 || c <= 0) return false;
        W = w;
        H = h;
        C = c;
        T = 1;
        data.clear();
        const size_t elems = (size_t) W * H * C;
        if (!ensure_device(elems)) {
            d_valid = false;
            return false;
        }
        if (lingbot_vae_update_temporal_cache_last_whdc_f32(
                d_chunk, d_data, w, h, t, 0, T, c, nullptr) != 0) {
            d_valid = false;
            return false;
        }
        d_W = W;
        d_H = H;
        d_C = C;
        d_T = T;
        d_valid = true;
        return true;
    }
#endif

    void update_from_chunk_last_frame(const std::vector<float> & chunk, int w, int h, int t, int c) {
        W = w;
        H = h;
        C = c;
        if (t <= 0 || chunk.size() != (size_t) w * h * t * c) {
            T = 0;
            data.clear();
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            d_valid = false;
#endif
            return;
        }
        T = 1;
        data.assign((size_t) W * H * C, 0.0f);
        const int src_t = t - 1;
        for (int iw = 0; iw < W; ++iw) {
            for (int ih = 0; ih < H; ++ih) {
                for (int ic = 0; ic < C; ++ic) {
                    data[vae_ggml_whdc_index(iw, ih, 0, ic, W, H, T)] =
                        chunk[vae_ggml_whdc_index(iw, ih, src_t, ic, W, H, t)];
                }
            }
        }
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
        d_valid = false;
#endif
    }

    void update_from_chunk(const std::vector<float> & chunk, int w, int h, int t, int c) {
        const int prev_T = T;
        const int prev_W = W;
        const int prev_H = H;
        const int prev_C = C;
        std::vector<float> prev = data;
        W = w;
        H = h;
        C = c;
        if (t <= 0 || chunk.size() != (size_t) w * h * t * c) {
            T = 0;
            data.clear();
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            d_valid = false;
#endif
            return;
        }
        const bool prepend_prev = t < 2 && prev_T > 0 && prev_W == w && prev_H == h && prev_C == c;
        T = std::min(2, t + (prepend_prev ? 1 : 0));
        data.assign((size_t) W * H * T * C, 0.0f);
        int dst_t = 0;
        if (prepend_prev) {
            for (int iw = 0; iw < W; ++iw) {
                for (int ih = 0; ih < H; ++ih) {
                    for (int ic = 0; ic < C; ++ic) {
                        data[vae_ggml_whdc_index(iw, ih, dst_t, ic, W, H, T)] =
                            prev[vae_ggml_whdc_index(iw, ih, prev_T - 1, ic, W, H, prev_T)];
                    }
                }
            }
            ++dst_t;
        }
        const int keep_cur = std::min(t, 2 - dst_t);
        const int src_t0 = t - keep_cur;
        for (int iw = 0; iw < W; ++iw) {
            for (int ih = 0; ih < H; ++ih) {
                for (int it = 0; it < keep_cur; ++it) {
                    for (int ic = 0; ic < C; ++ic) {
                        data[vae_ggml_whdc_index(iw, ih, dst_t + it, ic, W, H, T)] =
                            chunk[vae_ggml_whdc_index(iw, ih, src_t0 + it, ic, W, H, t)];
                    }
                }
            }
        }
    }
};

struct LingBotVaeStreamingViewCache {
    std::vector<VaeTemporalCacheWHDC> convs;
    size_t next = 0;

    LingBotVaeStreamingViewCache() = default;
    LingBotVaeStreamingViewCache(const LingBotVaeStreamingViewCache &) = delete;
    LingBotVaeStreamingViewCache & operator=(const LingBotVaeStreamingViewCache &) = delete;
    LingBotVaeStreamingViewCache(LingBotVaeStreamingViewCache &&) noexcept = default;
    LingBotVaeStreamingViewCache & operator=(LingBotVaeStreamingViewCache &&) noexcept = default;

    void begin() {
        next = 0;
    }

    VaeTemporalCacheWHDC & take() {
        if (next >= convs.size()) convs.emplace_back();
        return convs[next++];
    }
};

struct LingBotVaeStreamingSessionCache {
    std::vector<LingBotVaeStreamingViewCache> views;

    LingBotVaeStreamingSessionCache() = default;
    LingBotVaeStreamingSessionCache(const LingBotVaeStreamingSessionCache &) = delete;
    LingBotVaeStreamingSessionCache & operator=(const LingBotVaeStreamingSessionCache &) = delete;
    LingBotVaeStreamingSessionCache(LingBotVaeStreamingSessionCache &&) noexcept = default;
    LingBotVaeStreamingSessionCache & operator=(LingBotVaeStreamingSessionCache &&) noexcept = default;
};

std::unordered_map<uint64_t, LingBotVaeStreamingSessionCache> & lingbot_vae_stream_cache() {
    static std::unordered_map<uint64_t, LingBotVaeStreamingSessionCache> cache;
    return cache;
}

LingBotVaeStreamingSessionCache * lingbot_get_vae_stream_cache(uint64_t session_id, int64_t views) {
    if (session_id == 0 || views <= 0) return nullptr;
    std::lock_guard<std::mutex> lock(lingbot_init_latent_mutex());
    LingBotVaeStreamingSessionCache & cache = lingbot_vae_stream_cache()[session_id];
    if ((int64_t) cache.views.size() != views) {
        cache.views.clear();
        cache.views.resize((size_t) views);
    }
    return &cache;
}

bool vae_causal_conv3d_stream_host(
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        VaeTemporalCacheWHDC & cache,
        const std::vector<float> & weight,
        const std::vector<float> & bias,
        int W,
        int H,
        int T,
        int in_C,
        int out_C) {
    if (W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * in_C) return false;
    if (weight.size() != (size_t) 3 * 3 * 3 * out_C * in_C || bias.size() != (size_t) out_C) return false;
    if (cache.T != 0 && (cache.W != W || cache.H != H || cache.C != in_C)) return false;

    const bool aliased = &in_whdc == &out_whdc;
    const std::vector<float> input_copy = aliased ? in_whdc : std::vector<float>();
    const std::vector<float> & src_in = aliased ? input_copy : in_whdc;
    const int cache_T = cache.T;
    const int left_pad = std::max(0, 2 - cache_T);
    out_whdc.assign((size_t) W * H * T * out_C, 0.0f);

    auto weight_at = [&](int kw, int kh, int kt, int co, int ci) -> float {
        return weight[(size_t) kw + 3ull * ((size_t) kh + 3ull * ((size_t) kt + 3ull * ((size_t) co * in_C + ci)))];
    };
    auto input_at = [&](int w, int h, int concat_t, int c) -> float {
        if (w < 0 || w >= W || h < 0 || h >= H) return 0.0f;
        if (concat_t < 0) return 0.0f;
        if (concat_t < cache_T) {
            return cache.data[vae_ggml_whdc_index(w, h, concat_t, c, W, H, cache_T)];
        }
        const int cur_t = concat_t - cache_T;
        if (cur_t < 0 || cur_t >= T) return 0.0f;
        return src_in[vae_ggml_whdc_index(w, h, cur_t, c, W, H, T)];
    };

    for (int ow = 0; ow < W; ++ow) {
        for (int oh = 0; oh < H; ++oh) {
            for (int ot = 0; ot < T; ++ot) {
                for (int co = 0; co < out_C; ++co) {
                    double acc = bias[(size_t) co];
                    for (int kw = 0; kw < 3; ++kw) {
                        const int iw = ow + kw - 1;
                        for (int kh = 0; kh < 3; ++kh) {
                            const int ih = oh + kh - 1;
                            for (int kt = 0; kt < 3; ++kt) {
                                const int src_t = ot + kt - left_pad;
                                for (int ci = 0; ci < in_C; ++ci) {
                                    acc += (double) input_at(iw, ih, src_t, ci) *
                                           (double) weight_at(kw, kh, kt, co, ci);
                                }
                            }
                        }
                    }
                    out_whdc[vae_ggml_whdc_index(ow, oh, ot, co, W, H, T)] = (float) acc;
                }
            }
        }
    }
    cache.update_from_chunk(src_in, W, H, T, in_C);
    return true;
}

bool vae_conv1x1x1_host(
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        const std::vector<float> & weight,
        const std::vector<float> & bias,
        int W,
        int H,
        int T,
        int in_C,
        int out_C) {
    if (W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * in_C) return false;
    if (weight.size() != (size_t) out_C * in_C || bias.size() != (size_t) out_C) return false;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_cuda_requested() &&
        !lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_CONV1X1_DISABLE")) {
        const size_t in_bytes = in_whdc.size() * sizeof(float);
        const size_t out_elems = (size_t) W * H * T * out_C;
        float * d_in = nullptr;
        float * d_w = nullptr;
        float * d_b = nullptr;
        float * d_out = nullptr;
        bool ok = cudaMalloc(reinterpret_cast<void **>(&d_in), in_bytes) == cudaSuccess &&
                  cudaMalloc(reinterpret_cast<void **>(&d_out), out_elems * sizeof(float)) == cudaSuccess &&
                  cudaMalloc(reinterpret_cast<void **>(&d_w), weight.size() * sizeof(float)) == cudaSuccess &&
                  cudaMalloc(reinterpret_cast<void **>(&d_b), bias.size() * sizeof(float)) == cudaSuccess &&
                  cudaMemcpy(d_in, in_whdc.data(), in_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                  cudaMemcpy(d_w, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess &&
                  cudaMemcpy(d_b, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess &&
                  lingbot_vae_conv1x1x1_whdc_f32w(
                      d_in, d_w, d_b, d_out, W, H, T, in_C, out_C, nullptr) == 0 &&
                  cudaDeviceSynchronize() == cudaSuccess;
        if (ok) {
            out_whdc.assign(out_elems, 0.0f);
            ok = cudaMemcpy(out_whdc.data(), d_out, out_elems * sizeof(float),
                            cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (d_in) cudaFree(d_in);
        if (d_w) cudaFree(d_w);
        if (d_b) cudaFree(d_b);
        if (d_out) cudaFree(d_out);
        if (ok) return true;
        std::fprintf(stderr, "vla(lingbot_va): VAE conv1x1x1 CUDA path failed; falling back to host\n");
    }
#endif
    out_whdc.assign((size_t) W * H * T * out_C, 0.0f);
    for (int w = 0; w < W; ++w) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T; ++t) {
                for (int co = 0; co < out_C; ++co) {
                    double acc = bias[(size_t) co];
                    for (int ci = 0; ci < in_C; ++ci) {
                        acc += (double) in_whdc[vae_ggml_whdc_index(w, h, t, ci, W, H, T)] *
                               (double) weight[(size_t) co * in_C + ci];
                    }
                    out_whdc[vae_ggml_whdc_index(w, h, t, co, W, H, T)] = (float) acc;
                }
            }
        }
    }
    return true;
}

bool vae_conv1x1x1_cached_cuda_execute(
        gguf_reader & g,
        const std::string & weight_name,
        const std::string & bias_name,
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (!lingbot_vae_cuda_requested() ||
        lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_CONV1X1_DISABLE")) {
        return false;
    }
    if (W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0 ||
        in_whdc.size() != (size_t) W * H * T * in_C) {
        return false;
    }
    const float * d_w = lingbot_cuda_cached_f32_tensor(g, weight_name, (int64_t) out_C * in_C);
    const float * d_b = lingbot_cuda_cached_f32_tensor(g, bias_name, out_C);
    if (!d_w || !d_b) return false;
    const size_t in_bytes = in_whdc.size() * sizeof(float);
    const size_t out_elems = (size_t) W * (size_t) H * (size_t) T * (size_t) out_C;
    static std::mutex scratch_mu;
    static LingBotCudaF32Scratch scratch_in;
    static LingBotCudaF32Scratch scratch_out;
    std::lock_guard<std::mutex> scratch_lock(scratch_mu);
    bool ok = scratch_in.ensure(in_whdc.size()) &&
              scratch_out.ensure(out_elems) &&
              cudaMemcpy(scratch_in.ptr, in_whdc.data(), in_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
              lingbot_vae_conv1x1x1_whdc_f32w(
                  scratch_in.ptr, d_w, d_b, scratch_out.ptr, W, H, T, in_C, out_C, nullptr) == 0 &&
              cudaDeviceSynchronize() == cudaSuccess;
    if (ok) {
        out_whdc.assign(out_elems, 0.0f);
        ok = cudaMemcpy(out_whdc.data(), scratch_out.ptr, out_elems * sizeof(float),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    return ok;
#else
    (void) g;
    (void) weight_name;
    (void) bias_name;
    (void) in_whdc;
    (void) out_whdc;
    (void) W;
    (void) H;
    (void) T;
    (void) in_C;
    (void) out_C;
    return false;
#endif
}

bool vae_slice_time_chunk(
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        int W,
        int H,
        int T_total,
        int C,
        int t_offset,
        int T_chunk) {
    if (t_offset < 0 || T_chunk <= 0 || t_offset + T_chunk > T_total) return false;
    if (in_whdc.size() != (size_t) W * H * T_total * C) return false;
    out_whdc.assign((size_t) W * H * T_chunk * C, 0.0f);
    for (int w = 0; w < W; ++w) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T_chunk; ++t) {
                for (int c = 0; c < C; ++c) {
                    out_whdc[vae_ggml_whdc_index(w, h, t, c, W, H, T_chunk)] =
                        in_whdc[vae_ggml_whdc_index(w, h, t_offset + t, c, W, H, T_total)];
                }
            }
        }
    }
    return true;
}

void vae_append_time_chunk(
        std::vector<float> & dst_whdc,
        const std::vector<float> & chunk_whdc,
        int W,
        int H,
        int T_total,
        int C,
        int t_offset,
        int T_chunk) {
    if (dst_whdc.empty()) dst_whdc.assign((size_t) W * H * T_total * C, 0.0f);
    for (int w = 0; w < W; ++w) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < T_chunk; ++t) {
                for (int c = 0; c < C; ++c) {
                    dst_whdc[vae_ggml_whdc_index(w, h, t_offset + t, c, W, H, T_total)] =
                        chunk_whdc[vae_ggml_whdc_index(w, h, t, c, W, H, T_chunk)];
                }
            }
        }
    }
}

void vae_pack_whdc_to_lanes(
        const std::vector<float> & whdc,
        std::vector<float> & lanes,
        int W,
        int H,
        int T_total,
        int C,
        int t_offset,
        int T_chunk) {
    const int n_lanes = W * H;
    lanes.assign((size_t) n_lanes * T_chunk * C, 0.0f);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const int lane = h * W + w;
            for (int t = 0; t < T_chunk; ++t) {
                for (int c = 0; c < C; ++c) {
                    lanes[((size_t) lane * T_chunk + t) * C + c] =
                        whdc[vae_ggml_whdc_index(w, h, t_offset + t, c, W, H, T_total)];
                }
            }
        }
    }
}

void vae_unpack_lanes_to_whdc(
        const std::vector<float> & lanes,
        std::vector<float> & whdc,
        int W,
        int H,
        int T_total,
        int C,
        int t_offset,
        int T_chunk) {
    whdc.assign((size_t) W * H * T_total * C, 0.0f);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const int lane = h * W + w;
            for (int t = 0; t < T_chunk; ++t) {
                for (int c = 0; c < C; ++c) {
                    whdc[vae_ggml_whdc_index(w, h, t_offset + t, c, W, H, T_total)] =
                        lanes[((size_t) lane * T_chunk + t) * C + c];
                }
            }
        }
    }
}

void vae_write_lanes_chunk_to_whdc(
        const std::vector<float> & lanes,
        std::vector<float> & whdc,
        int W,
        int H,
        int T_total,
        int C,
        int t_offset,
        int T_chunk) {
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            const int lane = h * W + w;
            for (int t = 0; t < T_chunk; ++t) {
                for (int c = 0; c < C; ++c) {
                    whdc[vae_ggml_whdc_index(w, h, t_offset + t, c, W, H, T_total)] =
                        lanes[((size_t) lane * T_chunk + t) * C + c];
                }
            }
        }
    }
}

struct LingBotVaeCudaTimeConvWeights {
    int C = 0;
    int K = 0;
    std::vector<float> host_w;
    std::vector<float> host_b;
    float * d_w = nullptr;
    float * d_b = nullptr;
    bool borrowed_device = false;

    bool load(gguf_reader & g, const char * prefix, int channels, int kernel) {
        release();
        C = channels;
        K = kernel;
        const std::string w_name = std::string(prefix) + ".weight";
        const std::string b_name = std::string(prefix) + ".bias";
        const int64_t w_ne = (int64_t) C * (int64_t) C * (int64_t) K;
        const int64_t b_ne = (int64_t) C;
        if (!lingbot_cached_f32_tensor(g, w_name, w_ne, host_w)) return false;
        if (!lingbot_cached_f32_tensor(g, b_name, b_ne, host_b)) return false;
        const float * cached_w = lingbot_cuda_cached_f32_tensor(g, w_name, w_ne);
        const float * cached_b = lingbot_cuda_cached_f32_tensor(g, b_name, b_ne);
        if (cached_w && cached_b) {
            d_w = const_cast<float *>(cached_w);
            d_b = const_cast<float *>(cached_b);
            borrowed_device = true;
            return true;
        }
        if (cudaMalloc(&d_w, host_w.size() * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_b, host_b.size() * sizeof(float)) != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): CUDA malloc failed for VAE time_conv weights %s\n", prefix);
            release();
            return false;
        }
        if (cudaMemcpy(d_w, host_w.data(), host_w.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): CUDA upload failed for VAE time_conv weights %s\n", prefix);
            release();
            return false;
        }
        return true;
    }

    void release() {
        if (!borrowed_device) {
            if (d_w) cudaFree(d_w);
            if (d_b) cudaFree(d_b);
        }
        d_w = nullptr;
        d_b = nullptr;
        borrowed_device = false;
        C = 0;
        K = 0;
        host_w.clear();
        host_b.clear();
    }

    ~LingBotVaeCudaTimeConvWeights() {
        release();
    }
};

struct LingBotVaeCudaTimeConvWeightsIO {
    int in_C = 0;
    int out_C = 0;
    int K = 0;
    std::vector<float> host_w;
    std::vector<float> host_b;
    float * d_w = nullptr;
    float * d_b = nullptr;
    bool borrowed_device = false;

    bool load(gguf_reader & g, const char * prefix, int in_channels, int out_channels, int kernel) {
        release();
        in_C = in_channels;
        out_C = out_channels;
        K = kernel;
        const std::string w_name = std::string(prefix) + ".weight";
        const std::string b_name = std::string(prefix) + ".bias";
        const int64_t w_ne = (int64_t) out_C * (int64_t) in_C * (int64_t) K;
        const int64_t b_ne = (int64_t) out_C;
        if (!lingbot_cached_f32_tensor(g, w_name, w_ne, host_w)) return false;
        if (!lingbot_cached_f32_tensor(g, b_name, b_ne, host_b)) return false;
        const float * cached_w = lingbot_cuda_cached_f32_tensor(g, w_name, w_ne);
        const float * cached_b = lingbot_cuda_cached_f32_tensor(g, b_name, b_ne);
        if (cached_w && cached_b) {
            d_w = const_cast<float *>(cached_w);
            d_b = const_cast<float *>(cached_b);
            borrowed_device = true;
            return true;
        }
        if (cudaMalloc(&d_w, host_w.size() * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&d_b, host_b.size() * sizeof(float)) != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): CUDA malloc failed for VAE time_conv IO weights %s\n", prefix);
            release();
            return false;
        }
        if (cudaMemcpy(d_w, host_w.data(), host_w.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): CUDA upload failed for VAE time_conv IO weights %s\n", prefix);
            release();
            return false;
        }
        return true;
    }

    void release() {
        if (!borrowed_device) {
            if (d_w) cudaFree(d_w);
            if (d_b) cudaFree(d_b);
        }
        d_w = nullptr;
        d_b = nullptr;
        borrowed_device = false;
        in_C = 0;
        out_C = 0;
        K = 0;
        host_w.clear();
        host_b.clear();
    }

    ~LingBotVaeCudaTimeConvWeightsIO() {
        release();
    }
};

bool vae_downsample3d_time_stream_host(
        const std::vector<float> & spatial_whdc,
        std::vector<float> & out_whdc,
        const LingBotVaeCudaTimeConvWeightsIO & weights,
        int W,
        int H,
        int T_total,
        const std::vector<int> & chunks,
        std::vector<int> * out_chunks,
        int * out_T_total) {
    if (W <= 0 || H <= 0 || T_total <= 0 || weights.in_C <= 0 || weights.out_C <= 0 || weights.K != 3) return false;
    if (weights.in_C != weights.out_C) return false;
    if (spatial_whdc.size() != (size_t) W * H * T_total * weights.in_C) return false;
    int chunk_sum = 0;
    int total_out = 0;
    std::vector<int> local_out_chunks;
    local_out_chunks.reserve(chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
        const int chunk = chunks[i];
        if (chunk <= 0) return false;
        chunk_sum += chunk;
        const int out_chunk = i == 0 ? chunk : ((chunk + 1) >= weights.K ? ((chunk + 1 - weights.K) / 2 + 1) : 0);
        local_out_chunks.push_back(out_chunk);
        total_out += out_chunk;
    }
    if (chunk_sum != T_total || total_out <= 0) return false;

    out_whdc.assign((size_t) W * H * total_out * weights.out_C, 0.0f);
    std::vector<float> cache_last((size_t) W * H * weights.in_C, 0.0f);
    int in_offset = 0;
    int out_offset = 0;
    for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
        const int chunk = chunks[chunk_idx];
        if (chunk_idx == 0) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int t = 0; t < chunk; ++t) {
                        for (int c = 0; c < weights.out_C; ++c) {
                            out_whdc[vae_ggml_whdc_index(w, h, out_offset + t, c, W, H, total_out)] =
                                spatial_whdc[vae_ggml_whdc_index(w, h, in_offset + t, c, W, H, T_total)];
                        }
                    }
                    for (int c = 0; c < weights.in_C; ++c) {
                        cache_last[((size_t) h * W + w) * weights.in_C + c] =
                            spatial_whdc[vae_ggml_whdc_index(w, h, in_offset + chunk - 1, c, W, H, T_total)];
                    }
                }
            }
            in_offset += chunk;
            out_offset += chunk;
            continue;
        }

        const int concat_T = chunk + 1;
        const int chunk_out = concat_T >= weights.K ? ((concat_T - weights.K) / 2 + 1) : 0;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                const int lane = h * W + w;
                for (int to = 0; to < chunk_out; ++to) {
                    for (int co = 0; co < weights.out_C; ++co) {
                        double acc = weights.host_b[(size_t) co];
                        for (int k = 0; k < weights.K; ++k) {
                            const int src = to * 2 + k;
                            for (int ci = 0; ci < weights.in_C; ++ci) {
                                const float xv = src == 0
                                    ? cache_last[(size_t) lane * weights.in_C + ci]
                                    : spatial_whdc[vae_ggml_whdc_index(w, h, in_offset + src - 1, ci, W, H, T_total)];
                                acc += (double) xv * (double) weights.host_w[((size_t) co * weights.in_C + ci) * weights.K + k];
                            }
                        }
                        out_whdc[vae_ggml_whdc_index(w, h, out_offset + to, co, W, H, total_out)] = (float) acc;
                    }
                }
                for (int c = 0; c < weights.in_C; ++c) {
                    cache_last[(size_t) lane * weights.in_C + c] =
                        spatial_whdc[vae_ggml_whdc_index(w, h, in_offset + chunk - 1, c, W, H, T_total)];
                }
            }
        }
        in_offset += chunk;
        out_offset += chunk_out;
    }
    if (out_chunks) *out_chunks = std::move(local_out_chunks);
    if (out_T_total) *out_T_total = total_out;
    return true;
}

bool vae_time_conv_cuda_whdc_execute(
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        LingBotVaeCudaTimeConvWeights & weights,
        LingBotVaeCudaTemporalConvBatchedCache & cache,
        int W,
        int H,
        int T_total,
        int C,
        const std::vector<int> & chunks) {
    int chunk_sum = 0;
    for (int t : chunks) chunk_sum += t;
    if (W <= 0 || H <= 0 || C <= 0 || T_total <= 0 || chunk_sum != T_total) return false;
    if (weights.C != C || weights.K <= 0 || !weights.d_w || !weights.d_b) return false;
    const int lanes = W * H;
    if (in_whdc.size() != (size_t) W * H * T_total * C) return false;
    if (cache.lanes != lanes || cache.C != C || cache.K != weights.K || !cache.past) {
        cache.release();
        if (!cache.init(lanes, C, weights.K)) {
            std::fprintf(stderr, "vla(lingbot_va): CUDA VAE time_conv cache init failed WHDC=[%d,%d,%d,%d]\n",
                         W, H, T_total, C);
            return false;
        }
    }

    float * d_x = nullptr;
    float * d_out = nullptr;
    auto cleanup = [&]() {
        if (d_x) cudaFree(d_x);
        if (d_out) cudaFree(d_out);
    };
    const int max_chunk = *std::max_element(chunks.begin(), chunks.end());
    if (cudaMalloc(&d_x, (size_t) lanes * max_chunk * C * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out, (size_t) lanes * max_chunk * C * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA scratch malloc failed for VAE WHDC time_conv executor\n");
        cleanup();
        return false;
    }

    out_whdc.assign((size_t) W * H * T_total * C, 0.0f);
    std::vector<float> chunk_lanes;
    int offset = 0;
    for (int chunk : chunks) {
        vae_pack_whdc_to_lanes(in_whdc, chunk_lanes, W, H, T_total, C, offset, chunk);
        const size_t elems = (size_t) lanes * chunk * C;
        if (cudaMemcpy(d_x, chunk_lanes.data(), elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            !cache.step(d_x, weights.d_w, weights.d_b, d_out, chunk) ||
            cudaMemcpy(chunk_lanes.data(), d_out, elems * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): CUDA WHDC time_conv executor failed at offset=%d chunk=%d\n",
                         offset, chunk);
            cleanup();
            return false;
        }
        vae_write_lanes_chunk_to_whdc(chunk_lanes, out_whdc, W, H, T_total, C, offset, chunk);
        offset += chunk;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA WHDC time_conv executor sync failed\n");
        cleanup();
        return false;
    }
    cleanup();
    return true;
}













bool vae_time_conv_cuda_whdc_execute_io_stride(
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        LingBotVaeCudaTimeConvWeightsIO & weights,
        LingBotVaeCudaTemporalConvBatchedCache & cache,
        int W,
        int H,
        int T_total,
        int stride,
        const std::vector<int> & chunks,
        int * out_T_total);

bool vae_down_block_with_time_execute(
        gguf_reader & g,
        const char * block_prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        const std::vector<int> & chunks,
        std::vector<float> & out_whdc,
        std::vector<int> * out_chunks,
        int * out_W,
        int * out_H,
        int * out_T) {
    std::vector<float> spatial;
    int sw = 0;
    int sh = 0;
    if (!vae_down_block_resnets_spatial_ggml_execute(g, block_prefix, in_whdc, W, H, T, in_C, out_C,
                                                     spatial, &sw, &sh)) {
        return false;
    }
    const std::string time_prefix = std::string(block_prefix) + ".downsampler.time_conv";
    LingBotVaeCudaTimeConvWeightsIO weights;
    if (!weights.load(g, time_prefix.c_str(), out_C, out_C, 3)) return false;
    int time_out = 0;
    std::vector<int> time_chunks;
    if (!vae_downsample3d_time_stream_host(spatial, out_whdc, weights, sw, sh, T, chunks, &time_chunks, &time_out)) {
        return false;
    }
    auto dump_stage = [&](const std::string & suffix, const std::vector<float> & data, int dW, int dH, int dT, int dC) {
        const char * dump_dir = std::getenv("VLA_LINGBOT_VAE_DUMP_DIR");
        if (!dump_dir) return;
        std::string tag(block_prefix);
        const std::string needle = "vae.encoder.down_blocks.";
        const size_t pos = tag.find(needle);
        if (pos != std::string::npos) tag = "block" + tag.substr(pos + needle.size());
        const std::string base = std::string(dump_dir) + "/vae_encoder_" + tag + "_" + suffix;
        std::ofstream f32(base + ".f32", std::ios::binary);
        if (f32) f32.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * sizeof(float)));
        std::ofstream shape(base + ".shape.txt");
        if (shape) shape << dW << " " << dH << " " << dT << " " << dC << "\n";
    };
    dump_stage("time", out_whdc, sw, sh, time_out, out_C);
    std::vector<float> shortcut;
    std::vector<int> shortcut_chunks;
    int sc_w = 0, sc_h = 0, sc_t = 0;
    if (!vae_avg_down3d_stream_host(in_whdc, shortcut, W, H, T, in_C, out_C, 2, 2,
                                    chunks, &shortcut_chunks, &sc_w, &sc_h, &sc_t) ||
        sc_w != sw || sc_h != sh || sc_t != time_out || shortcut_chunks != time_chunks ||
        !vae_add_same_shape(out_whdc, shortcut)) {
        std::fprintf(stderr, "vla(lingbot_va): VAE down block %s AvgDown3D shortcut failed\n", block_prefix);
        return false;
    }
    dump_stage("shortcut", shortcut, sc_w, sc_h, sc_t, out_C);
    if (out_chunks) *out_chunks = std::move(time_chunks);
    if (out_W) *out_W = sw;
    if (out_H) *out_H = sh;
    if (out_T) *out_T = time_out;
    return true;
}

bool vae_mid_resnet_stream_host_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        const std::vector<int> & chunks,
        std::vector<float> & out_whdc);

bool vae_causal_conv3d_cached_ggml_execute(
        gguf_reader & g,
        const char * weight_name,
        const char * bias_name,
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        VaeTemporalCacheWHDC & cache,
        int W,
        int H,
        int T,
        int in_C,
        int out_C);

bool vae_down_block_resnets_spatial_stream_execute(
        gguf_reader & g,
        const char * block_prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        LingBotVaeStreamingViewCache & stream,
        std::vector<float> & out_whdc,
        int * out_W,
        int * out_H);

bool vae_down_block_with_time_stream_execute(
        gguf_reader & g,
        const char * block_prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        LingBotVaeStreamingViewCache & stream,
        std::vector<float> & out_whdc,
        int * out_W,
        int * out_H,
        int * out_T);

bool vae_mid_resnet_stream_one_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        LingBotVaeStreamingViewCache & stream,
        std::vector<float> & out_whdc);

bool vae_encoder_down_path_execute(
        gguf_reader & g,
        const std::vector<float> & x,
        int W,
        int H,
        int T,
        const std::vector<int> * input_chunks,
        std::vector<float> & b0,
        int * out_w0,
        int * out_h0,
        std::vector<float> & b1,
        int * out_w1,
        int * out_h1,
        int * out_t1,
        std::vector<float> & b2,
        int * out_w2,
        int * out_h2,
        int * out_t2,
        std::vector<int> * out_chunks_t2) {
    auto dump_stage = [](const char * label, const std::vector<float> & data, int W, int H, int T, int C) {
        const char * dump_dir = std::getenv("VLA_LINGBOT_VAE_DUMP_DIR");
        if (!dump_dir) return;
        const std::string base = std::string(dump_dir) + "/vae_encoder_" + label;
        std::ofstream f32(base + ".f32", std::ios::binary);
        if (f32) {
            f32.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * sizeof(float)));
        }
        std::ofstream shape(base + ".shape.txt");
        if (shape) {
            shape << W << " " << H << " " << T << " " << C << "\n";
        }
    };
    dump_stage("patch_input", x, W, H, T, 12);

    int w0 = 0;
    int h0 = 0;
    std::vector<float> conv_in;
    if (!vae_encoder_conv_in_ggml_execute(g, x, W, H, T, 12, conv_in)) {
        std::fprintf(stderr, "vla(lingbot_va): VAE encoder down path failed at conv_in\n");
        return false;
    }
    dump_stage("conv_in", conv_in, W, H, T, 160);
    if (!vae_down_block_resnets_spatial_ggml_execute(g, "vae.encoder.down_blocks.0",
                                                     conv_in, W, H, T, 160, 160, b0, &w0, &h0)) {
        std::fprintf(stderr, "vla(lingbot_va): VAE encoder down path failed at block0 main path\n");
        return false;
    }
    std::vector<float> b0_shortcut;
    int sc_w0 = 0, sc_h0 = 0, sc_t0 = 0;
    if (!vae_avg_down3d_host(conv_in, b0_shortcut, W, H, T, 160, 160, 1, 2, &sc_w0, &sc_h0, &sc_t0) ||
        sc_w0 != w0 || sc_h0 != h0 || sc_t0 != T || !vae_add_same_shape(b0, b0_shortcut)) {
        std::fprintf(stderr, "vla(lingbot_va): VAE encoder down path failed at block0 AvgDown3D shortcut\n");
        return false;
    }
    dump_stage("block0", b0, w0, h0, T, 160);
    std::vector<int> chunks0;
    if (input_chunks && !input_chunks->empty()) {
        chunks0 = *input_chunks;
    } else {
        chunks0.push_back(T);
    }
    int chunks0_sum = 0;
    for (int c : chunks0) {
        if (c <= 0) return false;
        chunks0_sum += c;
    }
    if (chunks0_sum != T) return false;

    int w1 = 0;
    int h1 = 0;
    int t1 = 0;
    std::vector<int> chunks1;
    if (!vae_down_block_with_time_execute(g, "vae.encoder.down_blocks.1", b0, w0, h0, T, 160, 320,
                                          chunks0, b1, &chunks1, &w1, &h1, &t1)) {
        std::fprintf(stderr, "vla(lingbot_va): VAE encoder down path failed at block1\n");
        return false;
    }
    dump_stage("block1", b1, w1, h1, t1, 320);

    int w2 = 0;
    int h2 = 0;
    int t2 = 0;
    std::vector<int> chunks2;
    if (!vae_down_block_with_time_execute(g, "vae.encoder.down_blocks.2", b1, w1, h1, t1, 320, 640,
                                          chunks1, b2, &chunks2, &w2, &h2, &t2)) {
        std::fprintf(stderr, "vla(lingbot_va): VAE encoder down path failed at block2\n");
        return false;
    }
    dump_stage("block2", b2, w2, h2, t2, 640);

    const std::vector<float> b2_shortcut = b2;
    std::vector<float> b3;
    if (!vae_mid_resnet_stream_host_execute(g, "vae.encoder.down_blocks.3.resnets.0",
                                            b2, w2, h2, t2, 640, chunks2, b3) ||
        !vae_mid_resnet_stream_host_execute(g, "vae.encoder.down_blocks.3.resnets.1",
                                            b3, w2, h2, t2, 640, chunks2, b2)) {
        std::fprintf(stderr, "vla(lingbot_va): VAE encoder down path failed at block3\n");
        return false;
    }
    if (!vae_add_same_shape(b2, b2_shortcut)) {
        std::fprintf(stderr, "vla(lingbot_va): VAE encoder down path failed at block3 outer shortcut\n");
        return false;
    }
    dump_stage("block3", b2, w2, h2, t2, 640);

    if (out_w0) *out_w0 = w0;
    if (out_h0) *out_h0 = h0;
    if (out_w1) *out_w1 = w1;
    if (out_h1) *out_h1 = h1;
    if (out_t1) *out_t1 = t1;
    if (out_w2) *out_w2 = w2;
    if (out_h2) *out_h2 = h2;
    if (out_t2) *out_t2 = t2;
    if (out_chunks_t2) *out_chunks_t2 = std::move(chunks2);
    return true;
}

bool vae_encoder_down_path_stream_execute(
        gguf_reader & g,
        const std::vector<float> & x,
        int W,
        int H,
        int T,
        LingBotVaeStreamingViewCache & stream,
        std::vector<float> & b2,
        int * out_w2,
        int * out_h2,
        int * out_t2) {
    std::vector<float> conv_in;
    if (!vae_causal_conv3d_cached_ggml_execute(g, "vae.encoder.conv_in.weight", "vae.encoder.conv_in.bias",
                                               x, conv_in, stream.take(), W, H, T, 12, 160)) {
        return false;
    }

    int w0 = 0;
    int h0 = 0;
    std::vector<float> b0;
    if (!vae_down_block_resnets_spatial_stream_execute(g, "vae.encoder.down_blocks.0",
                                                       conv_in, W, H, T, 160, 160,
                                                       stream, b0, &w0, &h0)) {
        return false;
    }
    std::vector<float> b0_shortcut;
    int sc_w0 = 0, sc_h0 = 0, sc_t0 = 0;
    if (!vae_avg_down3d_host(conv_in, b0_shortcut, W, H, T, 160, 160, 1, 2,
                             &sc_w0, &sc_h0, &sc_t0) ||
        sc_w0 != w0 || sc_h0 != h0 || sc_t0 != T || !vae_add_same_shape(b0, b0_shortcut)) {
        return false;
    }

    int w1 = 0, h1 = 0, t1 = 0;
    std::vector<float> b1;
    if (!vae_down_block_with_time_stream_execute(g, "vae.encoder.down_blocks.1",
                                                 b0, w0, h0, T, 160, 320,
                                                 stream, b1, &w1, &h1, &t1)) {
        return false;
    }

    int w2 = 0, h2 = 0, t2 = 0;
    if (!vae_down_block_with_time_stream_execute(g, "vae.encoder.down_blocks.2",
                                                 b1, w1, h1, t1, 320, 640,
                                                 stream, b2, &w2, &h2, &t2)) {
        return false;
    }

    const std::vector<float> b2_shortcut = b2;
    std::vector<float> b3;
    if (!vae_mid_resnet_stream_one_execute(g, "vae.encoder.down_blocks.3.resnets.0",
                                           b2, w2, h2, t2, 640, stream, b3) ||
        !vae_mid_resnet_stream_one_execute(g, "vae.encoder.down_blocks.3.resnets.1",
                                           b3, w2, h2, t2, 640, stream, b2) ||
        !vae_add_same_shape(b2, b2_shortcut)) {
        return false;
    }

    if (out_w2) *out_w2 = w2;
    if (out_h2) *out_h2 = h2;
    if (out_t2) *out_t2 = t2;
    return true;
}



bool vae_mid_resnet_ggml_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        std::vector<float> & out_whdc) {
    if (W <= 0 || H <= 0 || T <= 0 || Cc <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * Cc) return false;

    ggml_init_params params = { size_t(768) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(params);
    if (!C) return false;

    auto new_conv3_w = [&](const std::string & name, int64_t out_ch, int64_t in_ch) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, 3, 3, 3, out_ch * in_ch);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_bias3 = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, 1, channels);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_gamma = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, channels, 1, 1, 1);
        ggml_set_name(t, name.c_str());
        return t;
    };

    const std::string p(prefix);
    ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, W, H, T, Cc);
    ggml_set_name(x, "vae.mid_resnet.input");
    ggml_tensor * n1  = new_gamma(p + ".norm1.gamma", Cc);
    ggml_tensor * c1w = new_conv3_w(p + ".conv1.weight", Cc, Cc);
    ggml_tensor * c1b = new_bias3(p + ".conv1.bias", Cc);
    ggml_tensor * n2  = new_gamma(p + ".norm2.gamma", Cc);
    ggml_tensor * c2w = new_conv3_w(p + ".conv2.weight", Cc, Cc);
    ggml_tensor * c2b = new_bias3(p + ".conv2.bias", Cc);

    ggml_tensor * h = vae_norm_silu_to_conv_layout(C, x, n1);
    h = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, c1w, h, Cc), c1b);
    h = vae_norm_silu_to_conv_layout(C, h, n2);
    h = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, c2w, h, Cc), c2b);
    ggml_tensor * out = ggml_add(C, h, x);
    ggml_set_name(out, "vae.mid_resnet.output");
    ggml_set_output(out);

    LingBotAuxGraphBackend graph_backend;
    if (!lingbot_aux_graph_backend_init(graph_backend, "vae.ggml")) {
        ggml_free(C);
        return false;
    }
    ggml_backend_t backend = graph_backend.backend;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, graph_backend.buft);
    if (!buf) {
        lingbot_aux_graph_backend_free(graph_backend);
        ggml_free(C);
        return false;
    }

    std::vector<ggml_tensor *> weights = { n1, c1w, c1b, n2, c2w, c2b };
    bool ok = true;
    for (ggml_tensor * t : weights) {
        if (!set_tensor_from_gguf_f32(g, t)) {
            ok = false;
            break;
        }
    }
    if (ok) {
        ggml_backend_tensor_set(x, in_whdc.data(), 0, in_whdc.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(C, 65536, false);
        ggml_build_forward_expand(gf, out);
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        ok = st == GGML_STATUS_SUCCESS;
        if (!ok) {
            std::fprintf(stderr, "vla(lingbot_va): VAE mid ResNet graph failed (%d) for %s\n",
                         (int) st, prefix);
        }
    }
    if (ok) {
        out_whdc.assign((size_t) ggml_nelements(out), 0.0f);
        ggml_backend_tensor_get(out, out_whdc.data(), 0, out_whdc.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    lingbot_aux_graph_backend_free(graph_backend);
    ggml_free(C);
    return ok;
}

bool vae_resnet_ggml_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        std::vector<float> & out_whdc) {
    if (W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * in_C) return false;

    ggml_init_params params = { size_t(1024) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(params);
    if (!C) return false;

    auto new_conv3_w = [&](const std::string & name, int64_t out_ch, int64_t in_ch,
                           int64_t kt = 3, int64_t kh = 3, int64_t kw = 3) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, kw, kh, kt, out_ch * in_ch);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_bias3 = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, 1, channels);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_gamma = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, channels, 1, 1, 1);
        ggml_set_name(t, name.c_str());
        return t;
    };

    const std::string p(prefix);
    ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, W, H, T, in_C);
    ggml_set_name(x, "vae.resnet.input");
    ggml_tensor * n1  = new_gamma(p + ".norm1.gamma", in_C);
    ggml_tensor * c1w = new_conv3_w(p + ".conv1.weight", out_C, in_C);
    ggml_tensor * c1b = new_bias3(p + ".conv1.bias", out_C);
    ggml_tensor * n2  = new_gamma(p + ".norm2.gamma", out_C);
    ggml_tensor * c2w = new_conv3_w(p + ".conv2.weight", out_C, out_C);
    ggml_tensor * c2b = new_bias3(p + ".conv2.bias", out_C);
    ggml_tensor * scw = nullptr;
    ggml_tensor * scb = nullptr;
    if (in_C != out_C) {
        scw = new_conv3_w(p + ".conv_shortcut.weight", out_C, in_C, 1, 1, 1);
        scb = new_bias3(p + ".conv_shortcut.bias", out_C);
    }

    ggml_tensor * h = vae_norm_silu_to_conv_layout(C, x, n1);
    h = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, c1w, h, in_C), c1b);
    h = vae_norm_silu_to_conv_layout(C, h, n2);
    h = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, c2w, h, out_C), c2b);
    ggml_tensor * residual = x;
    if (scw) {
        residual = ggml_add(C, ggml_conv_3d(C, scw, x, in_C, 1, 1, 1, 0, 0, 0, 1, 1, 1), scb);
    }
    ggml_tensor * out = ggml_add(C, h, residual);
    ggml_set_name(out, "vae.resnet.output");
    ggml_set_output(out);

    LingBotAuxGraphBackend graph_backend;
    if (!lingbot_aux_graph_backend_init(graph_backend, "vae.ggml")) {
        ggml_free(C);
        return false;
    }
    ggml_backend_t backend = graph_backend.backend;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, graph_backend.buft);
    if (!buf) {
        lingbot_aux_graph_backend_free(graph_backend);
        ggml_free(C);
        return false;
    }

    std::vector<ggml_tensor *> weights = { n1, c1w, c1b, n2, c2w, c2b };
    if (scw) {
        weights.push_back(scw);
        weights.push_back(scb);
    }
    bool ok = true;
    for (ggml_tensor * t : weights) {
        if (!set_tensor_from_gguf_f32(g, t)) {
            ok = false;
            break;
        }
    }
    if (ok) {
        ggml_backend_tensor_set(x, in_whdc.data(), 0, in_whdc.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(C, 65536, false);
        ggml_build_forward_expand(gf, out);
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        ok = st == GGML_STATUS_SUCCESS;
        if (!ok) {
            std::fprintf(stderr, "vla(lingbot_va): VAE ResNet graph failed (%d) for %s\n",
                         (int) st, prefix);
        }
    }
    if (ok) {
        out_whdc.assign((size_t) ggml_nelements(out), 0.0f);
        ggml_backend_tensor_get(out, out_whdc.data(), 0, out_whdc.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    lingbot_aux_graph_backend_free(graph_backend);
    ggml_free(C);
    return ok;
}

bool vae_mid_attention_host_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        std::vector<float> & out_whdc) {
    if (W <= 0 || H <= 0 || T <= 0 || Cc <= 0 || Cc % 32 != 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * Cc) return false;
    const int tokens = W * H * T;
    const int spatial_tokens = W * H;
    const int qkv_C = 3 * Cc;

    const std::string p(prefix);
    std::vector<float> gamma;
    std::vector<float> qkv_w;
    std::vector<float> qkv_b;
    std::vector<float> proj_w;
    std::vector<float> proj_b;
    if (!lingbot_cached_f32_tensor(g, p + ".norm.gamma", Cc, gamma) ||
        !lingbot_cached_f32_tensor(g, p + ".to_qkv.weight", (int64_t) qkv_C * Cc, qkv_w) ||
        !lingbot_cached_f32_tensor(g, p + ".to_qkv.bias", qkv_C, qkv_b) ||
        !lingbot_cached_f32_tensor(g, p + ".proj.weight", (int64_t) Cc * Cc, proj_w) ||
        !lingbot_cached_f32_tensor(g, p + ".proj.bias", Cc, proj_b)) {
        return false;
    }

    std::vector<float> normed((size_t) tokens * Cc, 0.0f);
    for (int tok = 0; tok < tokens; ++tok) {
        const int t = tok % T;
        const int lane = tok / T;
        const int w = lane % W;
        const int h = lane / W;
        double ss = 0.0;
        for (int c = 0; c < Cc; ++c) {
            const float xv = in_whdc[vae_ggml_whdc_index(w, h, t, c, W, H, T)];
            ss += (double) xv * (double) xv;
        }
        const double l2 = std::sqrt(ss);
        const double norm_scale = l2 > 1e-12 ? std::sqrt((double) Cc) / l2 : 0.0;
        for (int c = 0; c < Cc; ++c) {
            const float xv = in_whdc[vae_ggml_whdc_index(w, h, t, c, W, H, T)];
            normed[(size_t) tok * Cc + c] = (float) ((double) xv * norm_scale) * gamma[(size_t) c];
        }
    }

    std::vector<float> qkv((size_t) tokens * qkv_C, 0.0f);
    for (int tok = 0; tok < tokens; ++tok) {
        const float * x = normed.data() + (size_t) tok * Cc;
        for (int o = 0; o < qkv_C; ++o) {
            double acc = qkv_b[(size_t) o];
            const float * w_row = qkv_w.data() + (size_t) o * Cc;
            for (int i = 0; i < Cc; ++i) {
                acc += (double) x[i] * (double) w_row[i];
            }
            qkv[(size_t) tok * qkv_C + o] = (float) acc;
        }
    }

    std::vector<float> ctx((size_t) tokens * Cc, 0.0f);
    std::vector<float> scores((size_t) spatial_tokens, 0.0f);
    const float scale = 1.0f / std::sqrt((float) Cc);
    for (int t = 0; t < T; ++t) {
        for (int q_sp = 0; q_sp < spatial_tokens; ++q_sp) {
            const int qi = q_sp * T + t;
            const float * q = qkv.data() + (size_t) qi * qkv_C;
            float max_score = -std::numeric_limits<float>::infinity();
            for (int k_sp = 0; k_sp < spatial_tokens; ++k_sp) {
                const int kj = k_sp * T + t;
                const float * k = qkv.data() + (size_t) kj * qkv_C + Cc;
                double dot = 0.0;
                for (int c = 0; c < Cc; ++c) {
                    dot += (double) q[c] * (double) k[c];
                }
                const float s = (float) dot * scale;
                scores[(size_t) k_sp] = s;
                max_score = std::max(max_score, s);
            }
            double denom = 0.0;
            for (int k_sp = 0; k_sp < spatial_tokens; ++k_sp) {
                const double e = std::exp((double) scores[(size_t) k_sp] - (double) max_score);
                scores[(size_t) k_sp] = (float) e;
                denom += e;
            }
            const double inv_denom = denom > 0.0 ? 1.0 / denom : 0.0;
            float * dst = ctx.data() + (size_t) qi * Cc;
            for (int k_sp = 0; k_sp < spatial_tokens; ++k_sp) {
                const int kj = k_sp * T + t;
                const float a = (float) ((double) scores[(size_t) k_sp] * inv_denom);
                const float * v = qkv.data() + (size_t) kj * qkv_C + 2 * Cc;
                for (int c = 0; c < Cc; ++c) {
                    dst[c] += a * v[c];
                }
            }
        }
    }

    std::vector<float> proj((size_t) tokens * Cc, 0.0f);
    for (int tok = 0; tok < tokens; ++tok) {
        const float * x = ctx.data() + (size_t) tok * Cc;
        for (int o = 0; o < Cc; ++o) {
            double acc = proj_b[(size_t) o];
            const float * w_row = proj_w.data() + (size_t) o * Cc;
            for (int i = 0; i < Cc; ++i) {
                acc += (double) x[i] * (double) w_row[i];
            }
            proj[(size_t) tok * Cc + o] = (float) acc;
        }
    }

    out_whdc = in_whdc;
    for (int tok = 0; tok < tokens; ++tok) {
        const int t = tok % T;
        const int lane = tok / T;
        const int w = lane % W;
        const int h = lane / W;
        for (int c = 0; c < Cc; ++c) {
            out_whdc[vae_ggml_whdc_index(w, h, t, c, W, H, T)] += proj[(size_t) tok * Cc + c];
        }
    }
    return true;
}

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
bool vae_mid_attention_cuda_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        std::vector<float> & out_whdc) {
    if (W <= 0 || H <= 0 || T <= 0 || Cc <= 0 || Cc % 32 != 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * Cc) return false;

    const std::string p(prefix);
    const int qkv_C = 3 * Cc;
    const float * d_gamma = lingbot_cuda_cached_f32_tensor(g, p + ".norm.gamma", Cc);
    const float * d_qkv_w = lingbot_cuda_cached_f32_tensor(g, p + ".to_qkv.weight", (int64_t) qkv_C * Cc);
    const float * d_qkv_b = lingbot_cuda_cached_f32_tensor(g, p + ".to_qkv.bias", qkv_C);
    const float * d_proj_w = lingbot_cuda_cached_f32_tensor(g, p + ".proj.weight", (int64_t) Cc * Cc);
    const float * d_proj_b = lingbot_cuda_cached_f32_tensor(g, p + ".proj.bias", Cc);
    if (!d_gamma || !d_qkv_w || !d_qkv_b || !d_proj_w || !d_proj_b) {
        return false;
    }

    out_whdc.assign(in_whdc.size(), 0.0f);
    float * d_in = nullptr;
    float * d_out = nullptr;
    auto cleanup = [&]() {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
    };
    const size_t in_bytes = in_whdc.size() * sizeof(float);
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess ||
        cudaMalloc(&d_out, in_bytes) != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA malloc failed for VAE mid attention\n");
        cleanup();
        return false;
    }
    if (cudaMemcpy(d_in, in_whdc.data(), in_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA upload failed for VAE mid attention\n");
        cleanup();
        return false;
    }
    if (lingbot_vae_mid_attn_f32(d_in, d_gamma, d_qkv_w, d_qkv_b, d_proj_w, d_proj_b,
                                 d_out, W, H, T, Cc, 0) != 0) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA VAE mid attention kernel failed\n");
        cleanup();
        return false;
    }
    if (cudaMemcpy(out_whdc.data(), d_out, in_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA download/sync failed for VAE mid attention\n");
        cleanup();
        return false;
    }
    cleanup();
    return true;
}

bool vae_time_conv_cuda_whdc_execute_io(
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        LingBotVaeCudaTimeConvWeightsIO & weights,
        LingBotVaeCudaTemporalConvBatchedCache & cache,
        int W,
        int H,
        int T_total,
        const std::vector<int> & chunks) {
    int chunk_sum = 0;
    for (int t : chunks) chunk_sum += t;
    if (W <= 0 || H <= 0 || weights.in_C <= 0 || weights.out_C <= 0 || T_total <= 0 || chunk_sum != T_total) return false;
    if (!weights.d_w || !weights.d_b || weights.K <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T_total * weights.in_C) return false;
    const int lanes = W * H;
    if (cache.lanes != lanes || cache.C != weights.in_C || cache.K != weights.K || !cache.past) {
        cache.release();
        if (!cache.init(lanes, weights.in_C, weights.K)) {
            std::fprintf(stderr, "vla(lingbot_va): CUDA VAE time_conv IO cache init failed WHDC=[%d,%d,%d,%d]\n",
                         W, H, T_total, weights.in_C);
            return false;
        }
    }

    float * d_x = nullptr;
    float * d_out = nullptr;
    auto cleanup = [&]() {
        if (d_x) cudaFree(d_x);
        if (d_out) cudaFree(d_out);
    };
    const int max_chunk = *std::max_element(chunks.begin(), chunks.end());
    if (cudaMalloc(&d_x, (size_t) lanes * max_chunk * weights.in_C * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out, (size_t) lanes * max_chunk * weights.out_C * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA scratch malloc failed for VAE WHDC time_conv IO executor\n");
        cleanup();
        return false;
    }

    out_whdc.assign((size_t) W * H * T_total * weights.out_C, 0.0f);
    int offset = 0;
    std::vector<float> chunk_lanes;
    for (int chunk : chunks) {
        vae_pack_whdc_to_lanes(in_whdc, chunk_lanes, W, H, T_total, weights.in_C, offset, chunk);
        const size_t in_elems = (size_t) lanes * chunk * weights.in_C;
        const size_t out_elems = (size_t) lanes * chunk * weights.out_C;
        std::vector<float> out_lanes(out_elems, 0.0f);
        if (cudaMemcpy(d_x, chunk_lanes.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            lingbot_causal_conv1d_cache_f32_batched(d_x, cache.past, weights.d_w, weights.d_b, d_out, cache.next,
                                                    lanes, chunk, weights.in_C, weights.out_C, weights.K, 0) != 0 ||
            cudaMemcpy(out_lanes.data(), d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): CUDA WHDC time_conv IO executor failed at offset=%d chunk=%d\n",
                         offset, chunk);
            cleanup();
            return false;
        }
        std::swap(cache.past, cache.next);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                const int lane = h * W + w;
                for (int t = 0; t < chunk; ++t) {
                    for (int c = 0; c < weights.out_C; ++c) {
                        out_whdc[vae_ggml_whdc_index(w, h, offset + t, c, W, H, T_total)] =
                            out_lanes[((size_t) lane * chunk + t) * weights.out_C + c];
                    }
                }
            }
        }
        offset += chunk;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA WHDC time_conv IO executor sync failed\n");
        cleanup();
        return false;
    }
    cleanup();
    return true;
}

bool vae_time_conv_cuda_whdc_execute_io_stride(
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        LingBotVaeCudaTimeConvWeightsIO & weights,
        LingBotVaeCudaTemporalConvBatchedCache & cache,
        int W,
        int H,
        int T_total,
        int stride,
        const std::vector<int> & chunks,
        int * out_T_total) {
    int chunk_sum = 0;
    int total_out = 0;
    for (int t : chunks) {
        chunk_sum += t;
        total_out += (t + stride - 1) / stride;
    }
    if (W <= 0 || H <= 0 || weights.in_C <= 0 || weights.out_C <= 0 || T_total <= 0 ||
        stride <= 0 || chunk_sum != T_total || total_out <= 0) return false;
    if (!weights.d_w || !weights.d_b || weights.K <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T_total * weights.in_C) return false;
    const int lanes = W * H;
    if (cache.lanes != lanes || cache.C != weights.in_C || cache.K != weights.K || !cache.past) {
        cache.release();
        if (!cache.init(lanes, weights.in_C, weights.K)) {
            std::fprintf(stderr, "vla(lingbot_va): CUDA VAE stride time_conv cache init failed WHDC=[%d,%d,%d,%d]\n",
                         W, H, T_total, weights.in_C);
            return false;
        }
    }

    float * d_x = nullptr;
    float * d_out = nullptr;
    auto cleanup = [&]() {
        if (d_x) cudaFree(d_x);
        if (d_out) cudaFree(d_out);
    };
    const int max_chunk = *std::max_element(chunks.begin(), chunks.end());
    const int max_chunk_out = (max_chunk + stride - 1) / stride;
    if (cudaMalloc(&d_x, (size_t) lanes * max_chunk * weights.in_C * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out, (size_t) lanes * max_chunk_out * weights.out_C * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA scratch malloc failed for VAE stride WHDC time_conv executor\n");
        cleanup();
        return false;
    }

    out_whdc.assign((size_t) W * H * total_out * weights.out_C, 0.0f);
    int offset_in = 0;
    int offset_out = 0;
    std::vector<float> chunk_lanes;
    for (int chunk : chunks) {
        const int chunk_out = (chunk + stride - 1) / stride;
        vae_pack_whdc_to_lanes(in_whdc, chunk_lanes, W, H, T_total, weights.in_C, offset_in, chunk);
        const size_t in_elems = (size_t) lanes * chunk * weights.in_C;
        const size_t out_elems = (size_t) lanes * chunk_out * weights.out_C;
        std::vector<float> out_lanes(out_elems, 0.0f);
        if (cudaMemcpy(d_x, chunk_lanes.data(), in_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            lingbot_causal_conv1d_cache_f32_batched_stride(d_x, cache.past, weights.d_w, weights.d_b,
                                                           d_out, cache.next, lanes, chunk, chunk_out,
                                                           weights.in_C, weights.out_C, weights.K, stride, 0) != 0 ||
            cudaMemcpy(out_lanes.data(), d_out, out_elems * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): CUDA stride WHDC time_conv failed at offset=%d chunk=%d stride=%d\n",
                         offset_in, chunk, stride);
            cleanup();
            return false;
        }
        std::swap(cache.past, cache.next);
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                const int lane = h * W + w;
                for (int t = 0; t < chunk_out; ++t) {
                    for (int c = 0; c < weights.out_C; ++c) {
                        out_whdc[vae_ggml_whdc_index(w, h, offset_out + t, c, W, H, total_out)] =
                            out_lanes[((size_t) lane * chunk_out + t) * weights.out_C + c];
                    }
                }
            }
        }
        offset_in += chunk;
        offset_out += chunk_out;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA stride WHDC time_conv executor sync failed\n");
        cleanup();
        return false;
    }
    cleanup();
    if (out_T_total) *out_T_total = total_out;
    return true;
}
#endif

bool vae_mid_attention_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        std::vector<float> & out_whdc) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_mid_attn_cuda_requested()) {
        if (vae_mid_attention_cuda_execute(g, prefix, in_whdc, W, H, T, Cc, out_whdc)) {
            return true;
        }
        std::fprintf(stderr, "vla(lingbot_va): falling back to host VAE mid attention\n");
    }
#endif
    return vae_mid_attention_host_execute(g, prefix, in_whdc, W, H, T, Cc, out_whdc);
}

bool vae_encoder_tail_ggml_execute(
        gguf_reader & g,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        std::vector<float> & out_whdc) {
    if (W <= 0 || H <= 0 || T <= 0 || Cc != 640) return false;
    if (in_whdc.size() != (size_t) W * H * T * Cc) return false;

    const int out_C = 96;
    ggml_init_params params = { size_t(512) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(params);
    if (!C) return false;

    auto new_conv3_w = [&](const std::string & name, int64_t out_ch, int64_t in_ch,
                           int64_t kt, int64_t kh, int64_t kw) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, kw, kh, kt, out_ch * in_ch);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_bias3 = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, 1, channels);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_gamma = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, channels, 1, 1, 1);
        ggml_set_name(t, name.c_str());
        return t;
    };

    ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, W, H, T, Cc);
    ggml_set_name(x, "vae.encoder_tail.input");
    ggml_tensor * norm = new_gamma("vae.encoder.norm_out.gamma", Cc);
    ggml_tensor * conv_w = new_conv3_w("vae.encoder.conv_out.weight", out_C, Cc, 3, 3, 3);
    ggml_tensor * conv_b = new_bias3("vae.encoder.conv_out.bias", out_C);
    ggml_tensor * q_w = new_conv3_w("vae.quant_conv.weight", out_C, out_C, 1, 1, 1);
    ggml_tensor * q_b = new_bias3("vae.quant_conv.bias", out_C);

    ggml_tensor * h = vae_norm_silu_to_conv_layout(C, x, norm);
    h = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, conv_w, h, Cc), conv_b);
    h = ggml_add(C, ggml_conv_3d(C, q_w, h, out_C, 1, 1, 1, 0, 0, 0, 1, 1, 1), q_b);
    ggml_set_name(h, "vae.encoder_tail.output");
    ggml_set_output(h);

    LingBotAuxGraphBackend graph_backend;
    if (!lingbot_aux_graph_backend_init(graph_backend, "vae.ggml")) {
        ggml_free(C);
        return false;
    }
    ggml_backend_t backend = graph_backend.backend;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, graph_backend.buft);
    if (!buf) {
        lingbot_aux_graph_backend_free(graph_backend);
        ggml_free(C);
        return false;
    }

    std::vector<ggml_tensor *> weights = { norm, conv_w, conv_b, q_w, q_b };
    bool ok = true;
    for (ggml_tensor * t : weights) {
        if (!set_tensor_from_gguf_f32(g, t)) {
            ok = false;
            break;
        }
    }
    if (ok) {
        ggml_backend_tensor_set(x, in_whdc.data(), 0, in_whdc.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(C, 65536, false);
        ggml_build_forward_expand(gf, h);
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        ok = st == GGML_STATUS_SUCCESS;
        if (!ok) {
            std::fprintf(stderr, "vla(lingbot_va): VAE encoder tail graph failed (%d)\n", (int) st);
        }
    }
    if (ok) {
        out_whdc.assign((size_t) ggml_nelements(h), 0.0f);
        ggml_backend_tensor_get(h, out_whdc.data(), 0, out_whdc.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    lingbot_aux_graph_backend_free(graph_backend);
    ggml_free(C);
    return ok;
}

bool vae_causal_conv3d_cached_ggml_execute(
        gguf_reader & g,
        const char * weight_name,
        const char * bias_name,
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        VaeTemporalCacheWHDC & cache,
        int W,
        int H,
        int T,
        int in_C,
        int out_C) {
    if (W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * in_C) return false;
    if (cache.T != 0 && (cache.W != W || cache.H != H || cache.C != in_C)) return false;

    const bool aliased = &in_whdc == &out_whdc;
    const std::vector<float> input_copy = aliased ? in_whdc : std::vector<float>();
    const std::vector<float> & src_in = aliased ? input_copy : in_whdc;
    const int cache_T = cache.T;
    const int prev_cache_W = cache.W;
    const int prev_cache_H = cache.H;
    const int prev_cache_T = cache.T;
    const int prev_cache_C = cache.C;

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_cuda_requested() &&
        !lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_CACHED_CONV_DISABLE")) {
        const int64_t weight_elems = (int64_t) out_C * in_C * 3 * 3 * 3;
        const float * d_weight = lingbot_cuda_cached_f32_tensor(g, weight_name, weight_elems);
        const float * d_bias = lingbot_cuda_cached_f32_tensor(g, bias_name, out_C);
        if (d_weight && d_bias) {
            const size_t in_bytes = src_in.size() * sizeof(float);
            const size_t cache_bytes = cache.data.size() * sizeof(float);
            const size_t out_elems = (size_t) W * (size_t) H * (size_t) T * (size_t) out_C;
            static std::mutex scratch_mu;
            static LingBotCudaF32Scratch scratch_in;
            static LingBotCudaF32Scratch scratch_cache;
            static LingBotCudaF32Scratch scratch_out;
            std::lock_guard<std::mutex> scratch_lock(scratch_mu);
            const float * d_cache = cache_T > 0 ? cache.device_ptr_or_upload() : nullptr;
            bool ok = scratch_in.ensure(src_in.size()) &&
                      scratch_out.ensure(out_elems) &&
                      cudaMemcpy(scratch_in.ptr, src_in.data(), in_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
            if (ok && cache_T > 0 && !d_cache) {
                ok = scratch_cache.ensure(cache.data.size()) &&
                     cudaMemcpy(scratch_cache.ptr, cache.data.data(), cache_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
                d_cache = ok ? scratch_cache.ptr : nullptr;
            }
            if (ok) {
                ok = lingbot_vae_cached_causal_conv3d_ks3_whdc_f32w(
                         scratch_in.ptr, cache_T > 0 ? d_cache : nullptr,
                         d_weight, d_bias, scratch_out.ptr,
                         W, H, T, cache_T, in_C, out_C, nullptr) == 0 &&
                     cudaDeviceSynchronize() == cudaSuccess;
            }
            if (ok) {
                out_whdc.assign(out_elems, 0.0f);
                ok = cudaMemcpy(out_whdc.data(), scratch_out.ptr, out_elems * sizeof(float),
                                cudaMemcpyDeviceToHost) == cudaSuccess;
            }
            if (ok) {
                cache.update_from_chunk(src_in, W, H, T, in_C);
                cache.update_device_from_chunk(scratch_in.ptr,
                                               prev_cache_W, prev_cache_H, prev_cache_T, prev_cache_C,
                                               W, H, T, in_C);
                return true;
            }
            std::fprintf(stderr,
                         "vla(lingbot_va): VAE cached causal conv CUDA path failed for %s; falling back to ggml\n",
                         weight_name);
        }
    }
#endif

    const int eff_T = cache_T + T;
    std::vector<float> eff((size_t) W * H * eff_T * in_C, 0.0f);
    for (int w = 0; w < W; ++w) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < cache_T; ++t) {
                for (int c = 0; c < in_C; ++c) {
                    eff[vae_ggml_whdc_index(w, h, t, c, W, H, eff_T)] =
                        cache.data[vae_ggml_whdc_index(w, h, t, c, W, H, cache_T)];
                }
            }
            for (int t = 0; t < T; ++t) {
                for (int c = 0; c < in_C; ++c) {
                    eff[vae_ggml_whdc_index(w, h, cache_T + t, c, W, H, eff_T)] =
                        src_in[vae_ggml_whdc_index(w, h, t, c, W, H, T)];
                }
            }
        }
    }

    ggml_init_params params = { size_t(384) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(params);
    if (!C) return false;

    ggml_tensor * w = ggml_new_tensor_4d(C, GGML_TYPE_F32, 3, 3, 3, out_C * in_C);
    ggml_set_name(w, weight_name);
    ggml_tensor * b = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, 1, out_C);
    ggml_set_name(b, bias_name);
    ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, W, H, eff_T, in_C);
    ggml_set_name(x, "vae.cached_conv.input");
    const int pad_left_t = std::max(0, 2 - cache_T);
    ggml_tensor * padded = ggml_pad_ext(C, x,
                                        1, 1,
                                        1, 1,
                                        pad_left_t, 0,
                                        0, 0);
    ggml_tensor * y = ggml_add(C,
                               ggml_conv_3d(C, w, padded, in_C,
                                            1, 1, 1,
                                            0, 0, 0,
                                            1, 1, 1),
                               b);
    ggml_set_name(y, "vae.cached_conv.output");
    ggml_set_output(y);

    LingBotAuxGraphBackend graph_backend;
    if (!lingbot_aux_graph_backend_init(graph_backend, "vae.ggml")) {
        ggml_free(C);
        return false;
    }
    ggml_backend_t backend = graph_backend.backend;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, graph_backend.buft);
    if (!buf) {
        lingbot_aux_graph_backend_free(graph_backend);
        ggml_free(C);
        return false;
    }

    bool ok = set_tensor_from_gguf_f32(g, w) && set_tensor_from_gguf_f32(g, b);
    if (ok) {
        ggml_backend_tensor_set(x, eff.data(), 0, eff.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(gf, y);
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        ok = st == GGML_STATUS_SUCCESS;
        if (!ok) {
            std::fprintf(stderr, "vla(lingbot_va): cached VAE conv graph failed (%d) for %s\n",
                         (int) st, weight_name);
        }
    }
    if (ok) {
        if (y->ne[2] != T) {
            std::fprintf(stderr,
                         "vla(lingbot_va): cached VAE conv temporal mismatch for %s: got %lld expected %d\n",
                         weight_name, (long long) y->ne[2], T);
            ok = false;
        } else {
            out_whdc.assign((size_t) ggml_nelements(y), 0.0f);
            ggml_backend_tensor_get(y, out_whdc.data(), 0, out_whdc.size() * sizeof(float));
        }
    }

    ggml_backend_buffer_free(buf);
    lingbot_aux_graph_backend_free(graph_backend);
    ggml_free(C);
    if (ok) cache.update_from_chunk(src_in, W, H, T, in_C);
    return ok;
}

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
bool vae_resnet_stream_cuda_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        LingBotVaeStreamingViewCache & stream,
        std::vector<float> & out_whdc) {
    if (!lingbot_vae_cuda_requested() ||
        !lingbot_env_enabled("VLA_LINGBOT_VAE_STREAM_RESNET_CUDA")) {
        return false;
    }
    if (W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0 ||
        in_whdc.size() != (size_t) W * H * T * in_C) {
        return false;
    }
    const std::string p(prefix);
    const float * n1 = lingbot_cuda_cached_f32_tensor(g, p + ".norm1.gamma", in_C);
    const float * c1w = lingbot_cuda_cached_f32_tensor(g, p + ".conv1.weight",
                                                       (int64_t) out_C * in_C * 3 * 3 * 3);
    const float * c1b = lingbot_cuda_cached_f32_tensor(g, p + ".conv1.bias", out_C);
    const float * n2 = lingbot_cuda_cached_f32_tensor(g, p + ".norm2.gamma", out_C);
    const float * c2w = lingbot_cuda_cached_f32_tensor(g, p + ".conv2.weight",
                                                       (int64_t) out_C * out_C * 3 * 3 * 3);
    const float * c2b = lingbot_cuda_cached_f32_tensor(g, p + ".conv2.bias", out_C);
    const float * scw = nullptr;
    const float * scb = nullptr;
    if (in_C != out_C) {
        scw = lingbot_cuda_cached_f32_tensor(g, p + ".conv_shortcut.weight", (int64_t) out_C * in_C);
        scb = lingbot_cuda_cached_f32_tensor(g, p + ".conv_shortcut.bias", out_C);
    }
    if (!n1 || !c1w || !c1b || !n2 || !c2w || !c2b || ((in_C != out_C) && (!scw || !scb))) {
        return false;
    }

    if (stream.next + 2 > stream.convs.size()) {
        stream.convs.resize(stream.next + 2);
    }
    VaeTemporalCacheWHDC & conv1_cache = stream.convs[stream.next++];
    VaeTemporalCacheWHDC & conv2_cache = stream.convs[stream.next++];
    if (conv1_cache.T != 0 && (conv1_cache.W != W || conv1_cache.H != H || conv1_cache.C != in_C)) return false;
    if (conv2_cache.T != 0 && (conv2_cache.W != W || conv2_cache.H != H || conv2_cache.C != out_C)) return false;

    const size_t in_elems = in_whdc.size();
    const size_t out_elems = (size_t) W * H * T * out_C;
    static std::mutex scratch_mu;
    static LingBotCudaF32Scratch scratch_in;
    static LingBotCudaF32Scratch scratch_norm;
    static LingBotCudaF32Scratch scratch_conv;
    static LingBotCudaF32Scratch scratch_residual;
    static LingBotCudaF32Scratch scratch_out;
    static LingBotCudaF32Scratch scratch_cache;
    std::lock_guard<std::mutex> scratch_lock(scratch_mu);
    bool ok = scratch_in.ensure(in_elems) &&
              scratch_norm.ensure(std::max(in_elems, out_elems)) &&
              scratch_conv.ensure(out_elems) &&
              scratch_residual.ensure(out_elems) &&
              scratch_out.ensure(out_elems) &&
              cudaMemcpy(scratch_in.ptr, in_whdc.data(), in_elems * sizeof(float),
                         cudaMemcpyHostToDevice) == cudaSuccess;
    const int c1_T = conv1_cache.T;
    const int c2_T = conv2_cache.T;
    const float * d_c1 = c1_T > 0 ? conv1_cache.device_ptr_or_upload() : nullptr;
    const float * d_c2 = c2_T > 0 ? conv2_cache.device_ptr_or_upload() : nullptr;
    if ((c1_T > 0 && !d_c1) || (c2_T > 0 && !d_c2)) ok = false;

    auto update_cache_from_device = [&](VaeTemporalCacheWHDC & cache,
                                        const float * d_chunk,
                                        int cW,
                                        int cH,
                                        int cT,
                                        int cC) -> bool {
        if (cache.update_from_device_chunk_device_only(d_chunk, cW, cH, cT, cC)) return true;
        if (cache.update_from_device_chunk(d_chunk, cW, cH, cT, cC)) return true;
        std::vector<float> chunk_host((size_t) cW * cH * cT * cC, 0.0f);
        if (cudaMemcpy(chunk_host.data(), d_chunk, chunk_host.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        cache.update_from_chunk(chunk_host, cW, cH, cT, cC);
        cache.upload_device_from_host();
        return true;
    };

    const float * residual = scratch_in.ptr;
    if (ok && in_C != out_C) {
        ok = lingbot_vae_conv1x1x1_whdc_f32w(
                 scratch_in.ptr, scw, scb, scratch_residual.ptr,
                 W, H, T, in_C, out_C, nullptr) == 0;
        residual = scratch_residual.ptr;
    }
    if (ok) {
        ok = lingbot_vae_norm_silu_whdc_f32w(scratch_in.ptr, n1, scratch_norm.ptr,
                                            W, H, T, in_C, nullptr) == 0 &&
             lingbot_vae_cached_causal_conv3d_ks3_whdc_f32w(
                 scratch_norm.ptr, c1_T > 0 ? d_c1 : nullptr,
                 c1w, c1b, scratch_conv.ptr,
                 W, H, T, c1_T, in_C, out_C, nullptr) == 0 &&
             update_cache_from_device(conv1_cache, scratch_norm.ptr, W, H, T, in_C) &&
             lingbot_vae_norm_silu_whdc_f32w(scratch_conv.ptr, n2, scratch_norm.ptr,
                                            W, H, T, out_C, nullptr) == 0 &&
             lingbot_vae_cached_causal_conv3d_ks3_whdc_f32w(
                 scratch_norm.ptr, c2_T > 0 ? d_c2 : nullptr,
                 c2w, c2b, scratch_conv.ptr,
                 W, H, T, c2_T, out_C, out_C, nullptr) == 0 &&
             update_cache_from_device(conv2_cache, scratch_norm.ptr, W, H, T, out_C) &&
             lingbot_vae_add_whdc_f32(scratch_conv.ptr, residual, scratch_out.ptr,
                                      out_elems, nullptr) == 0 &&
             cudaDeviceSynchronize() == cudaSuccess;
    }
    if (ok) {
        out_whdc.assign(out_elems, 0.0f);
        ok = cudaMemcpy(out_whdc.data(), scratch_out.ptr, out_elems * sizeof(float),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    return ok;
}

bool vae_resnet_stream_cuda_device_execute(
        gguf_reader & g,
        const char * prefix,
        const float * d_input,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        LingBotVaeStreamingViewCache & stream,
        LingBotCudaF32Scratch & scratch_norm,
        LingBotCudaF32Scratch & scratch_conv,
        LingBotCudaF32Scratch & scratch_residual,
        float * d_output) {
    if (!d_input || !d_output || W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0) return false;
    const std::string p(prefix);
    const float * n1 = lingbot_cuda_cached_f32_tensor(g, p + ".norm1.gamma", in_C);
    const float * c1w = lingbot_cuda_cached_f32_tensor(g, p + ".conv1.weight",
                                                       (int64_t) out_C * in_C * 3 * 3 * 3);
    const float * c1b = lingbot_cuda_cached_f32_tensor(g, p + ".conv1.bias", out_C);
    const float * n2 = lingbot_cuda_cached_f32_tensor(g, p + ".norm2.gamma", out_C);
    const float * c2w = lingbot_cuda_cached_f32_tensor(g, p + ".conv2.weight",
                                                       (int64_t) out_C * out_C * 3 * 3 * 3);
    const float * c2b = lingbot_cuda_cached_f32_tensor(g, p + ".conv2.bias", out_C);
    const float * scw = nullptr;
    const float * scb = nullptr;
    if (in_C != out_C) {
        scw = lingbot_cuda_cached_f32_tensor(g, p + ".conv_shortcut.weight", (int64_t) out_C * in_C);
        scb = lingbot_cuda_cached_f32_tensor(g, p + ".conv_shortcut.bias", out_C);
    }
    if (!n1 || !c1w || !c1b || !n2 || !c2w || !c2b || ((in_C != out_C) && (!scw || !scb))) {
        return false;
    }
    if (stream.next + 2 > stream.convs.size()) {
        stream.convs.resize(stream.next + 2);
    }
    VaeTemporalCacheWHDC & conv1_cache = stream.convs[stream.next++];
    VaeTemporalCacheWHDC & conv2_cache = stream.convs[stream.next++];
    if (conv1_cache.T != 0 && (conv1_cache.W != W || conv1_cache.H != H || conv1_cache.C != in_C)) return false;
    if (conv2_cache.T != 0 && (conv2_cache.W != W || conv2_cache.H != H || conv2_cache.C != out_C)) return false;

    const size_t in_elems = (size_t) W * H * T * in_C;
    const size_t out_elems = (size_t) W * H * T * out_C;
    if (!scratch_norm.ensure(std::max(in_elems, out_elems)) ||
        !scratch_conv.ensure(out_elems) ||
        !scratch_residual.ensure(out_elems)) {
        return false;
    }

    const int c1_T = conv1_cache.T;
    const int c2_T = conv2_cache.T;
    const float * d_c1 = c1_T > 0 ? conv1_cache.device_ptr_or_upload() : nullptr;
    const float * d_c2 = c2_T > 0 ? conv2_cache.device_ptr_or_upload() : nullptr;
    if ((c1_T > 0 && !d_c1) || (c2_T > 0 && !d_c2)) return false;

    const float * residual = d_input;
    if (in_C != out_C) {
        if (lingbot_vae_conv1x1x1_whdc_f32w(d_input, scw, scb, scratch_residual.ptr,
                                            W, H, T, in_C, out_C, nullptr) != 0) {
            return false;
        }
        residual = scratch_residual.ptr;
    } else if (d_output == d_input) {
        if (cudaMemcpy(scratch_residual.ptr, d_input, out_elems * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            return false;
        }
        residual = scratch_residual.ptr;
    }

    return lingbot_vae_norm_silu_whdc_f32w(d_input, n1, scratch_norm.ptr,
                                           W, H, T, in_C, nullptr) == 0 &&
           lingbot_vae_cached_causal_conv3d_ks3_whdc_f32w(
               scratch_norm.ptr, c1_T > 0 ? d_c1 : nullptr,
               c1w, c1b, scratch_conv.ptr,
               W, H, T, c1_T, in_C, out_C, nullptr) == 0 &&
           conv1_cache.update_from_device_chunk_device_only(scratch_norm.ptr, W, H, T, in_C) &&
           lingbot_vae_norm_silu_whdc_f32w(scratch_conv.ptr, n2, scratch_norm.ptr,
                                           W, H, T, out_C, nullptr) == 0 &&
           lingbot_vae_cached_causal_conv3d_ks3_whdc_f32w(
               scratch_norm.ptr, c2_T > 0 ? d_c2 : nullptr,
               c2w, c2b, scratch_conv.ptr,
               W, H, T, c2_T, out_C, out_C, nullptr) == 0 &&
           conv2_cache.update_from_device_chunk_device_only(scratch_norm.ptr, W, H, T, out_C) &&
           lingbot_vae_add_whdc_f32(scratch_conv.ptr, residual, d_output, out_elems, nullptr) == 0;
}

bool vae_causal_conv3d_cached_cuda_device_execute(
        gguf_reader & g,
        const char * weight_name,
        const char * bias_name,
        const float * d_input,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        VaeTemporalCacheWHDC & cache,
        float * d_output) {
    if (!d_input || !d_output || W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0) return false;
    const float * d_w = lingbot_cuda_cached_f32_tensor(g, weight_name, (int64_t) out_C * in_C * 3 * 3 * 3);
    const float * d_b = lingbot_cuda_cached_f32_tensor(g, bias_name, out_C);
    if (!d_w || !d_b) return false;
    if (cache.T != 0 && (cache.W != W || cache.H != H || cache.C != in_C)) return false;
    const int cache_T = cache.T;
    const float * d_cache = cache_T > 0 ? cache.device_ptr_or_upload() : nullptr;
    if (cache_T > 0 && !d_cache) return false;
    return lingbot_vae_cached_causal_conv3d_ks3_whdc_f32w(
               d_input, cache_T > 0 ? d_cache : nullptr,
               d_w, d_b, d_output,
               W, H, T, cache_T, in_C, out_C, nullptr) == 0 &&
           cache.update_from_device_chunk_device_only(d_input, W, H, T, in_C);
}

bool vae_mid_attention_cuda_device_execute(
        gguf_reader & g,
        const char * prefix,
        const float * d_input,
        int W,
        int H,
        int T,
        int Cc,
        float * d_output) {
    if (!d_input || !d_output || W <= 0 || H <= 0 || T <= 0 || Cc <= 0 || Cc % 32 != 0) return false;
    const std::string p(prefix);
    const int qkv_C = 3 * Cc;
    const float * d_gamma = lingbot_cuda_cached_f32_tensor(g, p + ".norm.gamma", Cc);
    const float * d_qkv_w = lingbot_cuda_cached_f32_tensor(g, p + ".to_qkv.weight", (int64_t) qkv_C * Cc);
    const float * d_qkv_b = lingbot_cuda_cached_f32_tensor(g, p + ".to_qkv.bias", qkv_C);
    const float * d_proj_w = lingbot_cuda_cached_f32_tensor(g, p + ".proj.weight", (int64_t) Cc * Cc);
    const float * d_proj_b = lingbot_cuda_cached_f32_tensor(g, p + ".proj.bias", Cc);
    if (!d_gamma || !d_qkv_w || !d_qkv_b || !d_proj_w || !d_proj_b) return false;
    return lingbot_vae_mid_attn_f32(d_input, d_gamma, d_qkv_w, d_qkv_b, d_proj_w, d_proj_b,
                                    d_output, W, H, T, Cc, nullptr) == 0;
}

bool vae_encoder_tail_stream_cuda_device_execute(
        gguf_reader & g,
        const float * d_input,
        int W,
        int H,
        int T,
        int Cc,
        LingBotVaeStreamingViewCache & stream,
        LingBotCudaF32Scratch & scratch_norm,
        LingBotCudaF32Scratch & scratch_conv,
        float * d_output) {
    if (!d_input || !d_output || W <= 0 || H <= 0 || T <= 0 || Cc != 640) return false;
    const int out_C = 96;
    const float * d_norm = lingbot_cuda_cached_f32_tensor(g, "vae.encoder.norm_out.gamma", Cc);
    const float * d_conv_w = lingbot_cuda_cached_f32_tensor(g, "vae.encoder.conv_out.weight",
                                                            (int64_t) out_C * Cc * 3 * 3 * 3);
    const float * d_conv_b = lingbot_cuda_cached_f32_tensor(g, "vae.encoder.conv_out.bias", out_C);
    const float * d_q_w = lingbot_cuda_cached_f32_tensor(g, "vae.quant_conv.weight",
                                                        (int64_t) out_C * out_C);
    const float * d_q_b = lingbot_cuda_cached_f32_tensor(g, "vae.quant_conv.bias", out_C);
    if (!d_norm || !d_conv_w || !d_conv_b || !d_q_w || !d_q_b) return false;
    VaeTemporalCacheWHDC & conv_cache = stream.take();
    if (conv_cache.T != 0 &&
        (conv_cache.W != W || conv_cache.H != H || conv_cache.C != Cc)) {
        return false;
    }
    const int cache_T = conv_cache.T;
    const float * d_cache = cache_T > 0 ? conv_cache.device_ptr_or_upload() : nullptr;
    if (cache_T > 0 && !d_cache) return false;
    const size_t in_elems = (size_t) W * H * T * Cc;
    const size_t conv_elems = (size_t) W * H * T * out_C;
    if (!scratch_norm.ensure(in_elems) || !scratch_conv.ensure(conv_elems)) return false;
    return lingbot_vae_norm_silu_whdc_f32w(d_input, d_norm, scratch_norm.ptr, W, H, T, Cc, nullptr) == 0 &&
           lingbot_vae_cached_causal_conv3d_ks3_whdc_f32w(
               scratch_norm.ptr, cache_T > 0 ? d_cache : nullptr,
               d_conv_w, d_conv_b, scratch_conv.ptr,
               W, H, T, cache_T, Cc, out_C, nullptr) == 0 &&
           conv_cache.update_from_device_chunk_device_only(scratch_norm.ptr, W, H, T, Cc) &&
           lingbot_vae_conv1x1x1_whdc_f32w(
               scratch_conv.ptr, d_q_w, d_q_b, d_output,
               W, H, T, out_C, out_C, nullptr) == 0;
}

bool vae_down_block_resnets_spatial_stream_cuda_device_execute(
        gguf_reader & g,
        const char * block_prefix,
        const float * d_input,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        LingBotVaeStreamingViewCache & stream,
        LingBotCudaF32Scratch & scratch_norm,
        LingBotCudaF32Scratch & scratch_conv,
        LingBotCudaF32Scratch & scratch_residual,
        LingBotCudaF32Scratch & scratch_a,
        LingBotCudaF32Scratch & scratch_b,
        float * d_output,
        int * out_W,
        int * out_H) {
    if (!d_input || !d_output || W <= 0 || H <= 0 || T <= 0 ||
        in_C <= 0 || out_C <= 0 || (W % 2) != 0 || (H % 2) != 0) {
        return false;
    }
    const std::string bp(block_prefix);
    if (!vae_resnet_stream_cuda_device_execute(
            g, (bp + ".resnets.0").c_str(),
            d_input, W, H, T, in_C, out_C, stream,
            scratch_norm, scratch_conv, scratch_residual, scratch_a.ptr)) {
        return false;
    }
    if (!vae_resnet_stream_cuda_device_execute(
            g, (bp + ".resnets.1").c_str(),
            scratch_a.ptr, W, H, T, out_C, out_C, stream,
            scratch_norm, scratch_conv, scratch_residual, scratch_b.ptr)) {
        return false;
    }
    const float * down_w = lingbot_cuda_cached_f32_tensor(
        g, bp + ".downsampler.resample.1.weight", (int64_t) out_C * out_C * 3 * 3);
    const float * down_b = lingbot_cuda_cached_f32_tensor(
        g, bp + ".downsampler.resample.1.bias", out_C);
    if (!down_w || !down_b) return false;
    if (lingbot_vae_spatial_downsample2d_whdc_f32w(
            scratch_b.ptr, down_w, down_b, d_output,
            W, H, T, out_C, nullptr) != 0) {
        return false;
    }
    if (out_W) *out_W = W / 2;
    if (out_H) *out_H = H / 2;
    return true;
}

bool vae_down_block_with_time_stream_cuda_device_execute(
        gguf_reader & g,
        const char * block_prefix,
        const float * d_input,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        LingBotVaeStreamingViewCache & stream,
        LingBotCudaF32Scratch & scratch_norm,
        LingBotCudaF32Scratch & scratch_conv,
        LingBotCudaF32Scratch & scratch_residual,
        LingBotCudaF32Scratch & scratch_a,
        LingBotCudaF32Scratch & scratch_b,
        LingBotCudaF32Scratch & scratch_spatial,
        LingBotCudaF32Scratch & scratch_time,
        LingBotCudaF32Scratch & scratch_shortcut,
        float * d_output,
        int * out_W,
        int * out_H,
        int * out_T) {
    if (!d_input || !d_output || W <= 0 || H <= 0 || T <= 0 ||
        in_C <= 0 || out_C <= 0) {
        return false;
    }
    int sw = 0;
    int sh = 0;
    if (!vae_down_block_resnets_spatial_stream_cuda_device_execute(
            g, block_prefix, d_input, W, H, T, in_C, out_C, stream,
            scratch_norm, scratch_conv, scratch_residual,
            scratch_a, scratch_b, scratch_spatial.ptr, &sw, &sh)) {
        return false;
    }
    LingBotVaeCudaTimeConvWeightsIO weights;
    if (!weights.load(g, (std::string(block_prefix) + ".downsampler.time_conv").c_str(), out_C, out_C, 3) ||
        !weights.d_w || !weights.d_b) {
        return false;
    }
    VaeTemporalCacheWHDC & time_cache = stream.take();
    if (time_cache.T != 0 &&
        (time_cache.W != sw || time_cache.H != sh || time_cache.C != out_C)) {
        return false;
    }
    const int cache_T = time_cache.T;
    const int time_out = cache_T == 0 ? T : ((T + 1) >= weights.K ? ((T + 1 - weights.K) / 2 + 1) : 0);
    if (time_out <= 0) return false;
    const int sc_w = W / 2;
    const int sc_h = H / 2;
    const int sc_t = (T + ((2 - (T % 2)) % 2)) / 2;
    if (sc_w != sw || sc_h != sh || sc_t != time_out) return false;
    const size_t out_elems = (size_t) sw * sh * time_out * out_C;
    const float * d_cache = cache_T > 0 ? time_cache.device_ptr_or_upload() : nullptr;
    if (cache_T > 0 && !d_cache) return false;
    bool ok = true;
    if (cache_T == 0) {
        ok = cudaMemcpy(scratch_time.ptr, scratch_spatial.ptr,
                        out_elems * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
    } else {
        ok = lingbot_vae_downsample3d_time_stream_one_whdc_f32w(
                 scratch_spatial.ptr, d_cache, weights.d_w, weights.d_b,
                 scratch_time.ptr, sw, sh, T, cache_T, out_C, out_C, nullptr) == 0;
    }
    ok = ok &&
         time_cache.update_from_device_chunk_last_frame_device_only(scratch_spatial.ptr, sw, sh, T, out_C) &&
         lingbot_vae_avg_down3d_whdc_f32(
             d_input, scratch_shortcut.ptr, W, H, T, in_C, out_C, 2, 2, nullptr) == 0 &&
         lingbot_vae_add_whdc_f32(
             scratch_time.ptr, scratch_shortcut.ptr, d_output, out_elems, nullptr) == 0;
    if (!ok) return false;
    if (out_W) *out_W = sw;
    if (out_H) *out_H = sh;
    if (out_T) *out_T = time_out;
    return true;
}
#endif

bool vae_resnet_stream_ggml_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        LingBotVaeStreamingViewCache & stream,
        std::vector<float> & out_whdc) {
    if (W <= 0 || H <= 0 || T <= 0 || in_C <= 0 || out_C <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * in_C) return false;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (vae_resnet_stream_cuda_execute(g, prefix, in_whdc, W, H, T, in_C, out_C, stream, out_whdc)) {
        return true;
    }
    if (lingbot_env_enabled("VLA_LINGBOT_VAE_STREAM_RESNET_CUDA")) {
        std::fprintf(stderr, "vla(lingbot_va): VAE stream resnet CUDA path failed for %s\n", prefix);
        return false;
    }
#endif
    const std::string p(prefix);
    std::vector<float> n1;
    std::vector<float> n2;
    if (!lingbot_cached_f32_tensor(g, p + ".norm1.gamma", in_C, n1) ||
        !lingbot_cached_f32_tensor(g, p + ".norm2.gamma", out_C, n2)) {
        return false;
    }
    std::vector<float> h;
    vae_norm_silu_host(in_whdc, n1, W, H, T, in_C, h);
    if (!vae_causal_conv3d_cached_ggml_execute(g, (p + ".conv1.weight").c_str(), (p + ".conv1.bias").c_str(),
                                               h, h, stream.take(), W, H, T, in_C, out_C)) {
        return false;
    }
    vae_norm_silu_host(h, n2, W, H, T, out_C, h);
    if (!vae_causal_conv3d_cached_ggml_execute(g, (p + ".conv2.weight").c_str(), (p + ".conv2.bias").c_str(),
                                               h, h, stream.take(), W, H, T, out_C, out_C)) {
        return false;
    }
    std::vector<float> residual = in_whdc;
    if (in_C != out_C) {
        if (vae_conv1x1x1_cached_cuda_execute(g,
                                              p + ".conv_shortcut.weight",
                                              p + ".conv_shortcut.bias",
                                              in_whdc,
                                              residual,
                                              W, H, T, in_C, out_C)) {
            // Fast path preserves the same shortcut projection tensor.
        } else {
            std::vector<float> sc_w;
            std::vector<float> sc_b;
            if (!lingbot_cached_f32_tensor(g, p + ".conv_shortcut.weight", (int64_t) out_C * in_C, sc_w) ||
                !lingbot_cached_f32_tensor(g, p + ".conv_shortcut.bias", out_C, sc_b) ||
                !vae_conv1x1x1_host(in_whdc, residual, sc_w, sc_b, W, H, T, in_C, out_C)) {
                return false;
            }
        }
    }
    if (!vae_add_same_shape(h, residual)) return false;
    out_whdc = std::move(h);
    return true;
}

bool vae_down_block_resnets_spatial_stream_execute(
        gguf_reader & g,
        const char * block_prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        LingBotVaeStreamingViewCache & stream,
        std::vector<float> & out_whdc,
        int * out_W,
        int * out_H) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_cuda_requested() &&
        lingbot_env_enabled("VLA_LINGBOT_VAE_STREAM_SPATIAL_BLOCK_CUDA")) {
        const std::string bp(block_prefix);
        auto need = [&](const std::string & name, int64_t elems) -> const float * {
            return lingbot_cuda_cached_f32_tensor(g, name, elems);
        };
        struct ResPtrs {
            const float * n1 = nullptr;
            const float * c1w = nullptr;
            const float * c1b = nullptr;
            const float * n2 = nullptr;
            const float * c2w = nullptr;
            const float * c2b = nullptr;
            const float * scw = nullptr;
            const float * scb = nullptr;
            int in_ch = 0;
            int out_ch = 0;
        };
        auto make_res = [&](const std::string & prefix, int rin, int rout, bool shortcut) {
            ResPtrs r;
            r.in_ch = rin;
            r.out_ch = rout;
            r.n1 = need(prefix + ".norm1.gamma", rin);
            r.c1w = need(prefix + ".conv1.weight", (int64_t) rout * rin * 3 * 3 * 3);
            r.c1b = need(prefix + ".conv1.bias", rout);
            r.n2 = need(prefix + ".norm2.gamma", rout);
            r.c2w = need(prefix + ".conv2.weight", (int64_t) rout * rout * 3 * 3 * 3);
            r.c2b = need(prefix + ".conv2.bias", rout);
            if (shortcut) {
                r.scw = need(prefix + ".conv_shortcut.weight", (int64_t) rout * rin);
                r.scb = need(prefix + ".conv_shortcut.bias", rout);
            }
            return r;
        };
        ResPtrs r0 = make_res(bp + ".resnets.0", in_C, out_C, in_C != out_C);
        ResPtrs r1 = make_res(bp + ".resnets.1", out_C, out_C, false);
        const float * down_w = need(bp + ".downsampler.resample.1.weight",
                                    (int64_t) out_C * out_C * 3 * 3);
        const float * down_b = need(bp + ".downsampler.resample.1.bias", out_C);
        const bool have = r0.n1 && r0.c1w && r0.c1b && r0.n2 && r0.c2w && r0.c2b &&
                          ((r0.scw != nullptr) == (r0.scb != nullptr)) &&
                          r1.n1 && r1.c1w && r1.c1b && r1.n2 && r1.c2w && r1.c2b &&
                          down_w && down_b;
        if (!have) {
            std::fprintf(stderr,
                         "vla(lingbot_va): VAE stream spatial CUDA missing weights for %s\n",
                         block_prefix);
            return false;
        }

        if (stream.next + 4 > stream.convs.size()) {
            stream.convs.resize(stream.next + 4);
        }
        VaeTemporalCacheWHDC & r0_c1 = stream.convs[stream.next++];
        VaeTemporalCacheWHDC & r0_c2 = stream.convs[stream.next++];
        VaeTemporalCacheWHDC & r1_c1 = stream.convs[stream.next++];
        VaeTemporalCacheWHDC & r1_c2 = stream.convs[stream.next++];
        const size_t in_elems = in_whdc.size();
        const size_t out_elems = (size_t) W * H * T * out_C;
        const int out_W_cuda = W / 2;
        const int out_H_cuda = H / 2;
        const size_t down_elems = (size_t) out_W_cuda * out_H_cuda * T * out_C;
        static std::mutex scratch_mu;
        static LingBotCudaF32Scratch scratch_in;
        static LingBotCudaF32Scratch scratch_a;
        static LingBotCudaF32Scratch scratch_b;
        static LingBotCudaF32Scratch scratch_sc;
        static LingBotCudaF32Scratch scratch_cache;
        static LingBotCudaF32Scratch scratch_down;
        std::lock_guard<std::mutex> scratch_lock(scratch_mu);
        bool ok = W > 0 && H > 0 && (W % 2) == 0 && (H % 2) == 0 &&
                  scratch_in.ensure(in_elems) &&
                  scratch_a.ensure(std::max(in_elems, out_elems)) &&
                  scratch_b.ensure(out_elems) &&
                  scratch_sc.ensure(out_elems) &&
                  scratch_down.ensure(down_elems) &&
                  cudaMemcpy(scratch_in.ptr, in_whdc.data(), in_elems * sizeof(float),
                             cudaMemcpyHostToDevice) == cudaSuccess;

        auto update_cache_from_device = [&](VaeTemporalCacheWHDC & cache,
                                            const float * d_chunk,
                                            int cW,
                                            int cH,
                                            int cT,
                                            int cC) -> bool {
            if (cache.update_from_device_chunk_device_only(d_chunk, cW, cH, cT, cC)) return true;
            if (cache.update_from_device_chunk(d_chunk, cW, cH, cT, cC)) return true;
            std::vector<float> chunk_host((size_t) cW * cH * cT * cC, 0.0f);
            if (cudaMemcpy(chunk_host.data(), d_chunk, chunk_host.size() * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                return false;
            }
            cache.update_from_chunk(chunk_host, cW, cH, cT, cC);
            cache.upload_device_from_host();
            return true;
        };

        auto apply_res = [&](const float * input,
                             float * output,
                             const ResPtrs & r,
                             VaeTemporalCacheWHDC & c1,
                             VaeTemporalCacheWHDC & c2,
                             int cur_C) -> bool {
            const int c1_T = c1.T;
            const int c2_T = c2.T;
            const float * d_c1 = c1_T > 0 ? c1.device_ptr_or_upload() : nullptr;
            const float * d_c2 = c2_T > 0 ? c2.device_ptr_or_upload() : nullptr;
            if ((c1_T > 0 && !d_c1) || (c2_T > 0 && !d_c2)) return false;
            const float * residual = input;
            if (r.scw) {
                if (lingbot_vae_conv1x1x1_whdc_f32w(input, r.scw, r.scb, scratch_sc.ptr,
                                                    W, H, T, r.in_ch, r.out_ch, nullptr) != 0) {
                    return false;
                }
                residual = scratch_sc.ptr;
            } else if (input == scratch_a.ptr || input == scratch_b.ptr) {
                if (cudaMemcpy(scratch_sc.ptr, input, out_elems * sizeof(float),
                               cudaMemcpyDeviceToDevice) != cudaSuccess) {
                    return false;
                }
                residual = scratch_sc.ptr;
            }
            if (lingbot_vae_norm_silu_whdc_f32w(input, r.n1, scratch_a.ptr,
                                                W, H, T, cur_C, nullptr) != 0 ||
                lingbot_vae_cached_causal_conv3d_ks3_whdc_f32w(
                    scratch_a.ptr, c1_T > 0 ? d_c1 : nullptr,
                    r.c1w, r.c1b, scratch_b.ptr,
                    W, H, T, c1_T, cur_C, r.out_ch, nullptr) != 0 ||
                !update_cache_from_device(c1, scratch_a.ptr, W, H, T, cur_C)) {
                return false;
            }
            if (lingbot_vae_norm_silu_whdc_f32w(scratch_b.ptr, r.n2, scratch_a.ptr,
                                                W, H, T, r.out_ch, nullptr) != 0 ||
                lingbot_vae_cached_causal_conv3d_ks3_whdc_f32w(
                    scratch_a.ptr, c2_T > 0 ? d_c2 : nullptr,
                    r.c2w, r.c2b, scratch_b.ptr,
                    W, H, T, c2_T, r.out_ch, r.out_ch, nullptr) != 0 ||
                !update_cache_from_device(c2, scratch_a.ptr, W, H, T, r.out_ch)) {
                return false;
            }
            return lingbot_vae_add_whdc_f32(scratch_b.ptr, residual, output, out_elems, nullptr) == 0;
        };

        if (ok) {
            ok = apply_res(scratch_in.ptr, scratch_a.ptr, r0, r0_c1, r0_c2, in_C) &&
                 apply_res(scratch_a.ptr, scratch_b.ptr, r1, r1_c1, r1_c2, out_C) &&
                 lingbot_vae_spatial_downsample2d_whdc_f32w(
                     scratch_b.ptr, down_w, down_b, scratch_down.ptr,
                     W, H, T, out_C, nullptr) == 0 &&
                 cudaDeviceSynchronize() == cudaSuccess;
        }
        if (ok) {
            out_whdc.assign(down_elems, 0.0f);
            ok = cudaMemcpy(out_whdc.data(), scratch_down.ptr, down_elems * sizeof(float),
                            cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (!ok) {
            std::fprintf(stderr,
                         "vla(lingbot_va): VAE stream spatial CUDA path failed for %s\n",
                         block_prefix);
            return false;
        }
        if (out_W) *out_W = out_W_cuda;
        if (out_H) *out_H = out_H_cuda;
        return true;
    }
#endif

    std::vector<float> h;
    if (!vae_resnet_stream_ggml_execute(g, (std::string(block_prefix) + ".resnets.0").c_str(),
                                        in_whdc, W, H, T, in_C, out_C, stream, h) ||
        !vae_resnet_stream_ggml_execute(g, (std::string(block_prefix) + ".resnets.1").c_str(),
                                        h, W, H, T, out_C, out_C, stream, h)) {
        return false;
    }
    return vae_spatial_downsample_ggml_execute(g, (std::string(block_prefix) + ".downsampler").c_str(),
                                               h, W, H, T, out_C, out_whdc, out_W, out_H);
}

bool vae_downsample3d_time_stream_one_host(
        const std::vector<float> & spatial_whdc,
        std::vector<float> & out_whdc,
        const LingBotVaeCudaTimeConvWeightsIO & weights,
        VaeTemporalCacheWHDC & cache,
        int W,
        int H,
        int T,
        int * out_T) {
    if (W <= 0 || H <= 0 || T <= 0 || weights.in_C <= 0 || weights.out_C <= 0 || weights.K != 3) return false;
    if (weights.in_C != weights.out_C) return false;
    if (spatial_whdc.size() != (size_t) W * H * T * weights.in_C) return false;
    const bool legacy_cache = lingbot_vae_time_downsample_legacy_cache_enabled();
    if (cache.T == 0) {
        out_whdc = spatial_whdc;
        if (legacy_cache) {
            cache.update_from_chunk(spatial_whdc, W, H, T, weights.in_C);
        } else {
            cache.update_from_chunk_last_frame(spatial_whdc, W, H, T, weights.in_C);
        }
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
        cache.upload_device_from_host();
#endif
        if (out_T) *out_T = T;
        return true;
    }
    if (cache.W != W || cache.H != H || cache.C != weights.in_C || cache.data.empty()) return false;
    const int concat_T = T + 1;
    const int To = concat_T >= weights.K ? ((concat_T - weights.K) / 2 + 1) : 0;
    if (To <= 0) return false;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_cuda_requested() &&
        !lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_TIME_DOWNSAMPLE_DISABLE") &&
        weights.d_w && weights.d_b) {
        const size_t spatial_bytes = spatial_whdc.size() * sizeof(float);
        const size_t out_elems = (size_t) W * H * To * weights.out_C;
        static std::mutex scratch_mu;
        static LingBotCudaF32Scratch scratch_spatial;
        static LingBotCudaF32Scratch scratch_cache;
        static LingBotCudaF32Scratch scratch_out;
        std::lock_guard<std::mutex> scratch_lock(scratch_mu);
        const float * d_cache = cache.device_ptr_or_upload();
        bool ok = scratch_spatial.ensure(spatial_whdc.size()) &&
                  scratch_out.ensure(out_elems) &&
                  cudaMemcpy(scratch_spatial.ptr, spatial_whdc.data(), spatial_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok && !d_cache) {
            ok = scratch_cache.ensure(cache.data.size()) &&
                 cudaMemcpy(scratch_cache.ptr, cache.data.data(), cache.data.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
            d_cache = ok ? scratch_cache.ptr : nullptr;
        }
        ok = ok &&
                  lingbot_vae_downsample3d_time_stream_one_whdc_f32w(
                      scratch_spatial.ptr, d_cache, weights.d_w, weights.d_b, scratch_out.ptr,
                      W, H, T, cache.T, weights.in_C, weights.out_C, nullptr) == 0 &&
                  cudaDeviceSynchronize() == cudaSuccess;
        if (ok) {
            out_whdc.assign(out_elems, 0.0f);
            ok = cudaMemcpy(out_whdc.data(), scratch_out.ptr, out_elems * sizeof(float),
                            cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (ok) {
            ok = legacy_cache
                ? cache.update_from_device_chunk(scratch_spatial.ptr, W, H, T, weights.in_C)
                : cache.update_from_device_chunk_last_frame(scratch_spatial.ptr, W, H, T, weights.in_C);
        }
        if (ok) {
            if (out_T) *out_T = To;
            return true;
        }
        std::fprintf(stderr, "vla(lingbot_va): VAE time downsample CUDA path failed; falling back to host\n");
    }
#endif
    out_whdc.assign((size_t) W * H * To * weights.out_C, 0.0f);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int to = 0; to < To; ++to) {
                for (int co = 0; co < weights.out_C; ++co) {
                    double acc = weights.host_b[(size_t) co];
                    for (int kt = 0; kt < weights.K; ++kt) {
                        const int src = to * 2 + kt;
                        for (int ci = 0; ci < weights.in_C; ++ci) {
                            const float xv = src == 0
                                ? cache.data[vae_ggml_whdc_index(w, h, cache.T - 1, ci, W, H, cache.T)]
                                : spatial_whdc[vae_ggml_whdc_index(w, h, src - 1, ci, W, H, T)];
                            acc += (double) xv * (double) weights.host_w[((size_t) co * weights.in_C + ci) * weights.K + kt];
                        }
                    }
                    out_whdc[vae_ggml_whdc_index(w, h, to, co, W, H, To)] = (float) acc;
                }
            }
        }
    }
    if (legacy_cache) {
        cache.update_from_chunk(spatial_whdc, W, H, T, weights.in_C);
    } else {
        cache.update_from_chunk_last_frame(spatial_whdc, W, H, T, weights.in_C);
    }
    if (out_T) *out_T = To;
    return true;
}

bool vae_down_block_with_time_stream_execute(
        gguf_reader & g,
        const char * block_prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        LingBotVaeStreamingViewCache & stream,
        std::vector<float> & out_whdc,
        int * out_W,
        int * out_H,
        int * out_T) {
    std::vector<float> spatial;
    int sw = 0;
    int sh = 0;
    if (!vae_down_block_resnets_spatial_stream_execute(g, block_prefix, in_whdc, W, H, T, in_C, out_C,
                                                       stream, spatial, &sw, &sh)) {
        return false;
    }
    LingBotVaeCudaTimeConvWeightsIO weights;
    if (!weights.load(g, (std::string(block_prefix) + ".downsampler.time_conv").c_str(), out_C, out_C, 3)) {
        return false;
    }
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_cuda_requested() &&
        lingbot_env_enabled("VLA_LINGBOT_VAE_STREAM_TIME_BLOCK_CUDA") &&
        !lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_TIME_DOWNSAMPLE_DISABLE") &&
        !lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_AVG_DOWN_DISABLE") &&
        weights.d_w && weights.d_b) {
        VaeTemporalCacheWHDC & time_cache = stream.take();
        if (time_cache.T != 0 &&
            (time_cache.W != sw || time_cache.H != sh || time_cache.C != out_C ||
             (time_cache.data.empty() && !time_cache.device_valid_for_shape(sw, sh, time_cache.T, out_C)))) {
            return false;
        }
        const int cache_T = time_cache.T;
        const int time_out = cache_T == 0 ? T : ((T + 1) >= weights.K ? ((T + 1 - weights.K) / 2 + 1) : 0);
        if (time_out <= 0) return false;
        const int sc_w = W / 2;
        const int sc_h = H / 2;
        const int sc_t = (T + ((2 - (T % 2)) % 2)) / 2;
        if (sc_w != sw || sc_h != sh || sc_t != time_out) {
            return false;
        }
        const size_t spatial_elems = spatial.size();
        const size_t input_elems = in_whdc.size();
        const size_t out_elems = (size_t) sw * sh * time_out * out_C;
        static std::mutex scratch_mu;
        static LingBotCudaF32Scratch scratch_spatial;
        static LingBotCudaF32Scratch scratch_input;
        static LingBotCudaF32Scratch scratch_time;
        static LingBotCudaF32Scratch scratch_shortcut;
        static LingBotCudaF32Scratch scratch_out;
        static LingBotCudaF32Scratch scratch_cache;
        std::lock_guard<std::mutex> scratch_lock(scratch_mu);
        const float * d_cache = cache_T > 0 ? time_cache.device_ptr_or_upload() : nullptr;
        bool ok = scratch_spatial.ensure(spatial_elems) &&
                  scratch_input.ensure(input_elems) &&
                  scratch_time.ensure(out_elems) &&
                  scratch_shortcut.ensure(out_elems) &&
                  scratch_out.ensure(out_elems) &&
                  cudaMemcpy(scratch_spatial.ptr, spatial.data(), spatial_elems * sizeof(float),
                             cudaMemcpyHostToDevice) == cudaSuccess &&
                  cudaMemcpy(scratch_input.ptr, in_whdc.data(), input_elems * sizeof(float),
                             cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok && cache_T > 0 && !d_cache) {
            ok = scratch_cache.ensure(time_cache.data.size()) &&
                 cudaMemcpy(scratch_cache.ptr, time_cache.data.data(),
                            time_cache.data.size() * sizeof(float),
                            cudaMemcpyHostToDevice) == cudaSuccess;
            d_cache = ok ? scratch_cache.ptr : nullptr;
        }
        if (ok) {
            if (cache_T == 0) {
                ok = cudaMemcpy(scratch_time.ptr, scratch_spatial.ptr,
                                out_elems * sizeof(float), cudaMemcpyDeviceToDevice) == cudaSuccess;
            } else {
                ok = lingbot_vae_downsample3d_time_stream_one_whdc_f32w(
                         scratch_spatial.ptr, d_cache, weights.d_w, weights.d_b, scratch_time.ptr,
                         sw, sh, T, cache_T, out_C, out_C, nullptr) == 0;
            }
        }
        if (ok) {
            if (!time_cache.update_from_device_chunk_last_frame_device_only(scratch_spatial.ptr, sw, sh, T, out_C) &&
                !time_cache.update_from_device_chunk_last_frame(scratch_spatial.ptr, sw, sh, T, out_C)) {
                time_cache.update_from_chunk_last_frame(spatial, sw, sh, T, out_C);
                time_cache.upload_device_from_host();
            }
            ok = lingbot_vae_avg_down3d_whdc_f32(
                     scratch_input.ptr, scratch_shortcut.ptr,
                     W, H, T, in_C, out_C, 2, 2, nullptr) == 0 &&
                 lingbot_vae_add_whdc_f32(
                     scratch_time.ptr, scratch_shortcut.ptr, scratch_out.ptr,
                     out_elems, nullptr) == 0 &&
                 cudaDeviceSynchronize() == cudaSuccess;
        }
        if (ok) {
            out_whdc.assign(out_elems, 0.0f);
            ok = cudaMemcpy(out_whdc.data(), scratch_out.ptr,
                            out_elems * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (!ok) {
            std::fprintf(stderr,
                         "vla(lingbot_va): VAE stream time CUDA path failed for %s\n",
                         block_prefix);
            return false;
        }
        if (out_W) *out_W = sw;
        if (out_H) *out_H = sh;
        if (out_T) *out_T = time_out;
        return true;
    }
#endif
    int time_out = 0;
    if (!vae_downsample3d_time_stream_one_host(spatial, out_whdc, weights, stream.take(), sw, sh, T, &time_out)) {
        return false;
    }
    std::vector<float> shortcut;
    int sc_w = 0, sc_h = 0, sc_t = 0;
    if (!vae_avg_down3d_host(in_whdc, shortcut, W, H, T, in_C, out_C, 2, 2,
                             &sc_w, &sc_h, &sc_t) ||
        sc_w != sw || sc_h != sh || sc_t != time_out ||
        !vae_add_same_shape(out_whdc, shortcut)) {
        std::fprintf(stderr,
                     "vla(lingbot_va): streaming VAE down block %s shortcut failed "
                     "main=[%d,%d,%d,%d] shortcut=[%d,%d,%d,%d] sizes=%zu/%zu input=[%d,%d,%d,%d]\n",
                     block_prefix, sw, sh, time_out, out_C, sc_w, sc_h, sc_t, out_C,
                     out_whdc.size(), shortcut.size(), W, H, T, in_C);
        return false;
    }
    if (out_W) *out_W = sw;
    if (out_H) *out_H = sh;
    if (out_T) *out_T = time_out;
    return true;
}

bool vae_mid_resnet_stream_one_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        LingBotVaeStreamingViewCache & stream,
        std::vector<float> & out_whdc) {
    return vae_resnet_stream_ggml_execute(g, prefix, in_whdc, W, H, T, Cc, Cc, stream, out_whdc);
}

bool vae_encoder_tail_stream_one_execute(
        gguf_reader & g,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        LingBotVaeStreamingViewCache & stream,
        std::vector<float> & out_whdc) {
    if (W <= 0 || H <= 0 || T <= 0 || Cc != 640) return false;
    if (in_whdc.size() != (size_t) W * H * T * Cc) return false;

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_cuda_requested() &&
        !lingbot_env_disabled("VLA_LINGBOT_VAE_CUDA_TAIL_STREAM_DISABLE")) {
        const int out_C = 96;
        const float * d_norm = lingbot_cuda_cached_f32_tensor(g, "vae.encoder.norm_out.gamma", Cc);
        const float * d_conv_w = lingbot_cuda_cached_f32_tensor(g, "vae.encoder.conv_out.weight",
                                                                (int64_t) out_C * Cc * 3 * 3 * 3);
        const float * d_conv_b = lingbot_cuda_cached_f32_tensor(g, "vae.encoder.conv_out.bias", out_C);
        const float * d_q_w = lingbot_cuda_cached_f32_tensor(g, "vae.quant_conv.weight",
                                                            (int64_t) out_C * out_C);
        const float * d_q_b = lingbot_cuda_cached_f32_tensor(g, "vae.quant_conv.bias", out_C);
        if (d_norm && d_conv_w && d_conv_b && d_q_w && d_q_b) {
            VaeTemporalCacheWHDC & conv_cache = stream.take();
            if (conv_cache.T != 0 &&
                (conv_cache.W != W || conv_cache.H != H || conv_cache.C != Cc)) {
                return false;
            }
            const int prev_cache_W = conv_cache.W;
            const int prev_cache_H = conv_cache.H;
            const int prev_cache_T = conv_cache.T;
            const int prev_cache_C = conv_cache.C;
            const int cache_T = conv_cache.T;
            const float * d_cache = cache_T > 0 ? conv_cache.device_ptr_or_upload() : nullptr;

            const size_t in_elems = in_whdc.size();
            const size_t conv_elems = (size_t) W * H * T * out_C;
            static std::mutex scratch_mu;
            static LingBotCudaF32Scratch scratch_in;
            static LingBotCudaF32Scratch scratch_norm;
            static LingBotCudaF32Scratch scratch_cache;
            static LingBotCudaF32Scratch scratch_conv;
            static LingBotCudaF32Scratch scratch_out;
            std::lock_guard<std::mutex> scratch_lock(scratch_mu);
            bool ok = scratch_in.ensure(in_elems) &&
                      scratch_norm.ensure(in_elems) &&
                      scratch_conv.ensure(conv_elems) &&
                      scratch_out.ensure(conv_elems) &&
                      cudaMemcpy(scratch_in.ptr, in_whdc.data(), in_elems * sizeof(float),
                                 cudaMemcpyHostToDevice) == cudaSuccess;
            if (ok && cache_T > 0 && !d_cache) {
                ok = scratch_cache.ensure(conv_cache.data.size()) &&
                     cudaMemcpy(scratch_cache.ptr, conv_cache.data.data(),
                                conv_cache.data.size() * sizeof(float),
                                cudaMemcpyHostToDevice) == cudaSuccess;
                d_cache = ok ? scratch_cache.ptr : nullptr;
            }
            if (ok) {
                ok = lingbot_vae_norm_silu_whdc_f32w(
                         scratch_in.ptr, d_norm, scratch_norm.ptr, W, H, T, Cc, nullptr) == 0 &&
                     lingbot_vae_cached_causal_conv3d_ks3_whdc_f32w(
                         scratch_norm.ptr, cache_T > 0 ? d_cache : nullptr,
                         d_conv_w, d_conv_b, scratch_conv.ptr,
                         W, H, T, cache_T, Cc, out_C, nullptr) == 0 &&
                     lingbot_vae_conv1x1x1_whdc_f32w(
                         scratch_conv.ptr, d_q_w, d_q_b, scratch_out.ptr,
                         W, H, T, out_C, out_C, nullptr) == 0 &&
                     cudaDeviceSynchronize() == cudaSuccess;
            }
            if (ok) {
                out_whdc.assign(conv_elems, 0.0f);
                ok = cudaMemcpy(out_whdc.data(), scratch_out.ptr,
                                conv_elems * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
            }
            if (ok) {
                if (!conv_cache.update_from_device_chunk_device_only(scratch_norm.ptr, W, H, T, Cc)) {
                    std::vector<float> norm_host(in_elems, 0.0f);
                    ok = cudaMemcpy(norm_host.data(), scratch_norm.ptr,
                                    in_elems * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
                    if (ok) {
                        conv_cache.update_from_chunk(norm_host, W, H, T, Cc);
                        conv_cache.update_device_from_chunk(scratch_norm.ptr,
                                                            prev_cache_W, prev_cache_H, prev_cache_T, prev_cache_C,
                                                            W, H, T, Cc);
                    }
                }
            }
            if (ok) {
                return true;
            }
            std::fprintf(stderr, "vla(lingbot_va): VAE streaming tail CUDA path failed; falling back to host\n");
        }
    }
#endif

    std::vector<float> norm;
    if (!lingbot_cached_f32_tensor(g, "vae.encoder.norm_out.gamma", Cc, norm)) {
        return false;
    }
    std::vector<float> h;
    vae_norm_silu_host(in_whdc, norm, W, H, T, Cc, h);
    if (!vae_causal_conv3d_cached_ggml_execute(g, "vae.encoder.conv_out.weight", "vae.encoder.conv_out.bias",
                                               h, h, stream.take(), W, H, T, Cc, 96)) {
        return false;
    }
    if (vae_conv1x1x1_cached_cuda_execute(g,
                                          "vae.quant_conv.weight",
                                          "vae.quant_conv.bias",
                                          h,
                                          out_whdc,
                                          W, H, T, 96, 96)) {
        return true;
    }
    std::vector<float> q_w;
    std::vector<float> q_b;
    if (!lingbot_cached_f32_tensor(g, "vae.quant_conv.weight", 96 * 96, q_w) ||
        !lingbot_cached_f32_tensor(g, "vae.quant_conv.bias", 96, q_b)) {
        return false;
    }
    return vae_conv1x1x1_host(h, out_whdc, q_w, q_b, W, H, T, 96, 96);
}

bool vae_mid_resnet_stream_host_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        const std::vector<int> & chunks,
        std::vector<float> & out_whdc) {
    if (W <= 0 || H <= 0 || T <= 0 || Cc <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * Cc) return false;
    int chunk_sum = 0;
    for (int c : chunks) {
        if (c <= 0) return false;
        chunk_sum += c;
    }
    if (chunk_sum != T) return false;

    const std::string p(prefix);
    std::vector<float> n1;
    std::vector<float> n2;
    if (!lingbot_cached_f32_tensor(g, p + ".norm1.gamma", Cc, n1) ||
        !lingbot_cached_f32_tensor(g, p + ".norm2.gamma", Cc, n2)) {
        return false;
    }

    VaeTemporalCacheWHDC conv1_cache;
    VaeTemporalCacheWHDC conv2_cache;
    out_whdc.assign((size_t) W * H * T * Cc, 0.0f);
    auto dump_debug = [&](const char * suffix, const std::vector<float> & data, int offset, int chunk) {
        const char * dump_dir = std::getenv("VLA_LINGBOT_VAE_DUMP_DIR");
        if (!dump_dir) return;
        std::string tag(prefix);
        std::replace(tag.begin(), tag.end(), '.', '_');
        const std::string base = std::string(dump_dir) + "/" + tag + "_" + suffix + "_t" + std::to_string(offset);
        std::ofstream f32(base + ".f32", std::ios::binary);
        if (f32) f32.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * sizeof(float)));
        std::ofstream shape(base + ".shape.txt");
        if (shape) shape << W << " " << H << " " << chunk << " " << Cc << "\n";
    };
    int offset = 0;
    for (int chunk : chunks) {
        std::vector<float> x_chunk;
        if (!vae_slice_time_chunk(in_whdc, x_chunk, W, H, T, Cc, offset, chunk)) return false;

        std::vector<float> h;
        vae_norm_silu_host(x_chunk, n1, W, H, chunk, Cc, h);
        dump_debug("norm1_silu", h, offset, chunk);
        if (!vae_causal_conv3d_cached_ggml_execute(g, (p + ".conv1.weight").c_str(), (p + ".conv1.bias").c_str(),
                                                   h, h, conv1_cache, W, H, chunk, Cc, Cc)) return false;
        dump_debug("conv1", h, offset, chunk);
        vae_norm_silu_host(h, n2, W, H, chunk, Cc, h);
        dump_debug("norm2_silu", h, offset, chunk);
        if (!vae_causal_conv3d_cached_ggml_execute(g, (p + ".conv2.weight").c_str(), (p + ".conv2.bias").c_str(),
                                                   h, h, conv2_cache, W, H, chunk, Cc, Cc)) return false;
        dump_debug("conv2", h, offset, chunk);
        if (!vae_add_same_shape(h, x_chunk)) return false;
        vae_append_time_chunk(out_whdc, h, W, H, T, Cc, offset, chunk);
        offset += chunk;
    }
    return true;
}

bool vae_encoder_tail_stream_host_execute(
        gguf_reader & g,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        const std::vector<int> & chunks,
        std::vector<float> & out_whdc) {
    if (W <= 0 || H <= 0 || T <= 0 || Cc != 640) return false;
    if (in_whdc.size() != (size_t) W * H * T * Cc) return false;
    int chunk_sum = 0;
    for (int c : chunks) {
        if (c <= 0) return false;
        chunk_sum += c;
    }
    if (chunk_sum != T) return false;

    const int out_C = 96;
    std::vector<float> norm;
    if (!lingbot_cached_f32_tensor(g, "vae.encoder.norm_out.gamma", Cc, norm)) {
        return false;
    }

    VaeTemporalCacheWHDC conv_cache;
    std::vector<float> conv_out((size_t) W * H * T * out_C, 0.0f);
    int offset = 0;
    for (int chunk : chunks) {
        std::vector<float> x_chunk;
        if (!vae_slice_time_chunk(in_whdc, x_chunk, W, H, T, Cc, offset, chunk)) return false;
        std::vector<float> h;
        vae_norm_silu_host(x_chunk, norm, W, H, chunk, Cc, h);
        if (!vae_causal_conv3d_cached_ggml_execute(g, "vae.encoder.conv_out.weight", "vae.encoder.conv_out.bias",
                                                   h, h, conv_cache, W, H, chunk, Cc, out_C)) return false;
        vae_append_time_chunk(conv_out, h, W, H, T, out_C, offset, chunk);
        offset += chunk;
    }

    if (vae_conv1x1x1_cached_cuda_execute(g,
                                          "vae.quant_conv.weight",
                                          "vae.quant_conv.bias",
                                          conv_out,
                                          out_whdc,
                                          W, H, T, out_C, out_C)) {
        return true;
    }
    std::vector<float> q_w;
    std::vector<float> q_b;
    if (!lingbot_cached_f32_tensor(g, "vae.quant_conv.weight", (int64_t) out_C * out_C, q_w) ||
        !lingbot_cached_f32_tensor(g, "vae.quant_conv.bias", out_C, q_b)) {
        return false;
    }
    return vae_conv1x1x1_host(conv_out, out_whdc, q_w, q_b, W, H, T, out_C, out_C);
}

bool vae_encoder_mid_tail_execute(
        gguf_reader & g,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        const std::vector<int> * input_chunks,
        std::vector<float> & out_whdc,
        double * res0_checksum,
        double * attn_checksum,
        double * res1_checksum) {
    auto dump_stage = [&](const char * label, const std::vector<float> & data, int dW, int dH, int dT, int dC) {
        const char * dump_dir = std::getenv("VLA_LINGBOT_VAE_DUMP_DIR");
        if (!dump_dir) return;
        const std::string base = std::string(dump_dir) + "/vae_encoder_" + label;
        std::ofstream f32(base + ".f32", std::ios::binary);
        if (f32) f32.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * sizeof(float)));
        std::ofstream shape(base + ".shape.txt");
        if (shape) shape << dW << " " << dH << " " << dT << " " << dC << "\n";
    };
    std::vector<int> chunks;
    if (input_chunks && !input_chunks->empty()) {
        chunks = *input_chunks;
    } else {
        chunks.push_back(T);
    }
    int chunk_sum = 0;
    for (int c : chunks) {
        if (c <= 0) return false;
        chunk_sum += c;
    }
    if (chunk_sum != T) return false;

    std::vector<float> r0;
    if (!vae_mid_resnet_stream_host_execute(g, "vae.encoder.mid_block.resnets.0", in_whdc, W, H, T, Cc, chunks, r0)) {
        return false;
    }
    dump_stage("mid_res0", r0, W, H, T, Cc);
    std::vector<float> attn;
    if (!vae_mid_attention_execute(g, "vae.encoder.mid_block.attentions.0", r0, W, H, T, Cc, attn)) {
        return false;
    }
    dump_stage("mid_attn", attn, W, H, T, Cc);
    std::vector<float> r1;
    if (!vae_mid_resnet_stream_host_execute(g, "vae.encoder.mid_block.resnets.1", attn, W, H, T, Cc, chunks, r1)) {
        return false;
    }
    dump_stage("mid_res1", r1, W, H, T, Cc);
    if (!vae_encoder_tail_stream_host_execute(g, r1, W, H, T, Cc, chunks, out_whdc)) {
        return false;
    }
    if (res0_checksum) *res0_checksum = checksum(r0);
    if (attn_checksum) *attn_checksum = checksum(attn);
    if (res1_checksum) *res1_checksum = checksum(r1);
    return true;
}

bool vae_encoder_mid_tail_stream_execute(
        gguf_reader & g,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        LingBotVaeStreamingViewCache & stream,
        std::vector<float> & out_whdc) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_cuda_requested() &&
        lingbot_env_enabled("VLA_LINGBOT_VAE_MID_TAIL_DEVICE") &&
        !lingbot_env_disabled("VLA_LINGBOT_VAE_MID_TAIL_DEVICE_DISABLE")) {
        const size_t in_elems = in_whdc.size();
        const size_t mid_elems = (size_t) W * H * T * Cc;
        const size_t out_elems = (size_t) W * H * T * 96;
        static std::mutex scratch_mu;
        static LingBotCudaF32Scratch scratch_in;
        static LingBotCudaF32Scratch scratch_a;
        static LingBotCudaF32Scratch scratch_b;
        static LingBotCudaF32Scratch scratch_out;
        static LingBotCudaF32Scratch scratch_norm;
        static LingBotCudaF32Scratch scratch_conv;
        static LingBotCudaF32Scratch scratch_residual;
        std::lock_guard<std::mutex> scratch_lock(scratch_mu);
        bool ok = W > 0 && H > 0 && T > 0 && Cc == 640 &&
                  in_elems == mid_elems &&
                  scratch_in.ensure(mid_elems) &&
                  scratch_a.ensure(mid_elems) &&
                  scratch_b.ensure(mid_elems) &&
                  scratch_out.ensure(out_elems) &&
                  cudaMemcpy(scratch_in.ptr, in_whdc.data(), in_elems * sizeof(float),
                             cudaMemcpyHostToDevice) == cudaSuccess;
        if (ok) {
            ok = vae_resnet_stream_cuda_device_execute(
                     g, "vae.encoder.mid_block.resnets.0",
                     scratch_in.ptr, W, H, T, Cc, Cc, stream,
                     scratch_norm, scratch_conv, scratch_residual, scratch_a.ptr) &&
                 vae_mid_attention_cuda_device_execute(
                     g, "vae.encoder.mid_block.attentions.0",
                     scratch_a.ptr, W, H, T, Cc, scratch_b.ptr) &&
                 vae_resnet_stream_cuda_device_execute(
                     g, "vae.encoder.mid_block.resnets.1",
                     scratch_b.ptr, W, H, T, Cc, Cc, stream,
                     scratch_norm, scratch_conv, scratch_residual, scratch_a.ptr) &&
                 vae_encoder_tail_stream_cuda_device_execute(
                     g, scratch_a.ptr, W, H, T, Cc, stream,
                     scratch_norm, scratch_conv, scratch_out.ptr) &&
                 cudaDeviceSynchronize() == cudaSuccess;
        }
        if (ok) {
            out_whdc.assign(out_elems, 0.0f);
            ok = cudaMemcpy(out_whdc.data(), scratch_out.ptr,
                            out_elems * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
        }
        if (ok) {
            return true;
        }
        if (lingbot_env_enabled("VLA_LINGBOT_VAE_MID_TAIL_DEVICE_REQUIRED")) {
            std::fprintf(stderr, "vla(lingbot_va): VAE mid-tail device runner failed\n");
            return false;
        }
    }
#endif
    std::vector<float> r0;
    if (!vae_mid_resnet_stream_one_execute(g, "vae.encoder.mid_block.resnets.0",
                                           in_whdc, W, H, T, Cc, stream, r0)) {
        return false;
    }
    std::vector<float> attn;
    if (!vae_mid_attention_execute(g, "vae.encoder.mid_block.attentions.0", r0, W, H, T, Cc, attn)) {
        return false;
    }
    std::vector<float> r1;
    if (!vae_mid_resnet_stream_one_execute(g, "vae.encoder.mid_block.resnets.1",
                                           attn, W, H, T, Cc, stream, r1)) {
        return false;
    }
    return vae_encoder_tail_stream_one_execute(g, r1, W, H, T, Cc, stream, out_whdc);
}

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
bool lingbot_vae_persistent_encoder_requested() {
    return lingbot_vae_cuda_requested();
}

struct LingBotVaePersistentEncoderRunner {
    std::mutex mu;
    LingBotCudaF32Scratch video;
    LingBotCudaF32Scratch patch;
    LingBotCudaF32Scratch conv_in;
    LingBotCudaF32Scratch b0;
    LingBotCudaF32Scratch b1;
    LingBotCudaF32Scratch b2;
    LingBotCudaF32Scratch a;
    LingBotCudaF32Scratch b;
    LingBotCudaF32Scratch residual;
    LingBotCudaF32Scratch norm;
    LingBotCudaF32Scratch conv;
    LingBotCudaF32Scratch spatial;
    LingBotCudaF32Scratch time;
    LingBotCudaF32Scratch shortcut;
    LingBotCudaF32Scratch enc_views;
    LingBotCudaF32Scratch out_latent;
    LingBotCudaF32Scratch latent_mean;
    LingBotCudaF32Scratch latent_inv_std;
    std::string stats_path;
    int stats_z_dim = 0;

    bool ensure_stats(const std::string & path,
                      const std::vector<float> & mean,
                      const std::vector<float> & stdv,
                      int z_dim) {
        if (z_dim <= 0 || (int) mean.size() != z_dim || (int) stdv.size() != z_dim) return false;
        if (!latent_mean.ensure((size_t) z_dim) || !latent_inv_std.ensure((size_t) z_dim)) return false;
        if (stats_path == path && stats_z_dim == z_dim) return true;
        std::vector<float> inv((size_t) z_dim, 0.0f);
        for (int i = 0; i < z_dim; ++i) {
            if (stdv[(size_t) i] == 0.0f) return false;
            inv[(size_t) i] = 1.0f / stdv[(size_t) i];
        }
        if (cudaMemcpy(latent_mean.ptr, mean.data(), (size_t) z_dim * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(latent_inv_std.ptr, inv.data(), (size_t) z_dim * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            stats_path.clear();
            stats_z_dim = 0;
            return false;
        }
        stats_path = path;
        stats_z_dim = z_dim;
        return true;
    }

    bool ensure_shape(int views, int frames, int height, int width, int z_dim) {
        if (views <= 0 || frames <= 0 || height <= 0 || width <= 0 || z_dim != 48) return false;
        const int patch_W = width / 2;
        const int patch_H = height / 2;
        if ((patch_W % 8) != 0 || (patch_H % 8) != 0) return false;
        const size_t video_elems = (size_t) views * 3 * frames * height * width;
        const size_t patch_elems = (size_t) patch_W * patch_H * frames * 12;
        const size_t conv_in_elems = (size_t) patch_W * patch_H * frames * 160;
        const size_t b0_elems = (size_t) (patch_W / 2) * (patch_H / 2) * frames * 160;
        const size_t b1_elems = (size_t) (patch_W / 4) * (patch_H / 4) * frames * 320;
        const size_t b2_elems = (size_t) (patch_W / 8) * (patch_H / 8) * frames * 640;
        const size_t max_mid = std::max(std::max(conv_in_elems, b0_elems), std::max(b1_elems, b2_elems));
        const size_t enc_view_elems = (size_t) (patch_W / 8) * (patch_H / 8) * frames * 96;
        const size_t out_elems = (size_t) z_dim * frames * (patch_H / 8) * ((patch_W / 8) * views);
        return video.ensure(video_elems) &&
               patch.ensure(patch_elems) &&
               conv_in.ensure(conv_in_elems) &&
               b0.ensure(b0_elems) &&
               b1.ensure(b1_elems) &&
               b2.ensure(b2_elems) &&
               a.ensure(max_mid) &&
               b.ensure(max_mid) &&
               residual.ensure(max_mid) &&
               norm.ensure(max_mid) &&
               conv.ensure(max_mid) &&
               spatial.ensure(max_mid) &&
               time.ensure(max_mid) &&
               shortcut.ensure(max_mid) &&
               enc_views.ensure(enc_view_elems * (size_t) views) &&
               out_latent.ensure(out_elems);
    }
};

bool encode_lingbot_video_to_latent_persistent_libero(
        gguf_reader & vg,
        const std::string & vae_path,
        const float * video_vcfhw,
        int64_t views64,
        int64_t channels64,
        int64_t frames64,
        int64_t height64,
        int64_t width64,
        LingBotVaeStreamingSessionCache * stream_cache,
        const std::vector<float> & latents_mean,
        const std::vector<float> & latents_std,
        int z_dim,
        std::vector<int> * output_chunks,
        std::vector<float> & out_bcfhw,
        LingBotTensor5DShape & out_shape) {
    if (!lingbot_vae_persistent_encoder_requested()) return false;
    if (!video_vcfhw || !stream_cache) return false;
    if (views64 != 2 || channels64 != 3 || height64 != 128 || width64 != 128 ||
        frames64 <= 0 || z_dim != 48) {
        return false;
    }
    const int views = (int) views64;
    const int frames = (int) frames64;
    const int height = (int) height64;
    const int width = (int) width64;
    if ((int) stream_cache->views.size() < views) return false;

    static LingBotVaePersistentEncoderRunner runner;
    const bool breakdown = lingbot_env_enabled("VLA_LINGBOT_VAE_BREAKDOWN");
    LingBotClock::time_point stage_t = LingBotClock::now();
    auto log_stage = [&](const char * stage, int view = -1) {
        if (!breakdown) return;
        cudaDeviceSynchronize();
        const auto now = LingBotClock::now();
        if (view >= 0) {
            std::printf("vla(lingbot_va): timing vae_persistent_stage view=%d stage=%s %.3fms\n",
                        view, stage, lingbot_elapsed_ms(stage_t, now));
        } else {
            std::printf("vla(lingbot_va): timing vae_persistent_stage stage=%s %.3fms\n",
                        stage, lingbot_elapsed_ms(stage_t, now));
        }
        stage_t = now;
    };
    std::lock_guard<std::mutex> runner_lock(runner.mu);
    if (!runner.ensure_shape(views, frames, height, width, z_dim) ||
        !runner.ensure_stats(vae_path, latents_mean, latents_std, z_dim)) {
        return false;
    }
    log_stage("ensure_shape_stats");

    const int patch_W = width / 2;
    const int patch_H = height / 2;
    const size_t video_elems = (size_t) views * 3 * frames * height * width;
    if (cudaMemcpy(runner.video.ptr, video_vcfhw, video_elems * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    log_stage("input_h2d");

    int latent_W_single = 0;
    int latent_H = 0;
    int latent_T = 0;
    for (int v = 0; v < views; ++v) {
        LingBotVaeStreamingViewCache & view_stream = stream_cache->views[(size_t) v];
        view_stream.begin();
        if (lingbot_vae_libero_scale_patchify_vcfhw_to_whdc_f32(
                runner.video.ptr, runner.patch.ptr,
                v, views, frames, height, width, nullptr) != 0) {
            return false;
        }
        log_stage("scale_patchify", v);

        int w0 = 0, h0 = 0, w1 = 0, h1 = 0, t1 = 0, w2 = 0, h2 = 0, t2 = 0;
        if (!vae_causal_conv3d_cached_cuda_device_execute(
                vg, "vae.encoder.conv_in.weight", "vae.encoder.conv_in.bias",
                runner.patch.ptr, patch_W, patch_H, frames, 12, 160,
                view_stream.take(), runner.conv_in.ptr)) {
            return false;
        }
        log_stage("conv_in", v);
        if (!vae_down_block_resnets_spatial_stream_cuda_device_execute(
                vg, "vae.encoder.down_blocks.0",
                runner.conv_in.ptr, patch_W, patch_H, frames, 160, 160,
                view_stream, runner.norm, runner.conv, runner.residual,
                runner.a, runner.b, runner.b0.ptr, &w0, &h0)) {
            return false;
        }
        log_stage("down0_resnets_spatial", v);
        const size_t b0_elems = (size_t) w0 * h0 * frames * 160;
        if (lingbot_vae_avg_down3d_whdc_f32(
                runner.conv_in.ptr, runner.shortcut.ptr,
                patch_W, patch_H, frames, 160, 160, 1, 2, nullptr) != 0 ||
            lingbot_vae_add_whdc_f32(
                runner.b0.ptr, runner.shortcut.ptr, runner.b0.ptr, b0_elems, nullptr) != 0) {
            return false;
        }
        log_stage("down0_shortcut_add", v);
        if (!vae_down_block_with_time_stream_cuda_device_execute(
                vg, "vae.encoder.down_blocks.1",
                runner.b0.ptr, w0, h0, frames, 160, 320,
                view_stream, runner.norm, runner.conv, runner.residual,
                runner.a, runner.b, runner.spatial, runner.time, runner.shortcut,
                runner.b1.ptr, &w1, &h1, &t1)) {
            return false;
        }
        log_stage("down1_resnets_spatial_time_shortcut", v);
        if (!vae_down_block_with_time_stream_cuda_device_execute(
                vg, "vae.encoder.down_blocks.2",
                runner.b1.ptr, w1, h1, t1, 320, 640,
                view_stream, runner.norm, runner.conv, runner.residual,
                runner.a, runner.b, runner.spatial, runner.time, runner.shortcut,
                runner.b2.ptr, &w2, &h2, &t2)) {
            return false;
        }
        log_stage("down2_resnets_spatial_time_shortcut", v);
        const size_t b2_elems = (size_t) w2 * h2 * t2 * 640;
        if (cudaMemcpy(runner.shortcut.ptr, runner.b2.ptr,
                       b2_elems * sizeof(float), cudaMemcpyDeviceToDevice) != cudaSuccess ||
            !vae_resnet_stream_cuda_device_execute(
                vg, "vae.encoder.down_blocks.3.resnets.0",
                runner.b2.ptr, w2, h2, t2, 640, 640, view_stream,
                runner.norm, runner.conv, runner.residual, runner.a.ptr) ||
            !vae_resnet_stream_cuda_device_execute(
                vg, "vae.encoder.down_blocks.3.resnets.1",
                runner.a.ptr, w2, h2, t2, 640, 640, view_stream,
                runner.norm, runner.conv, runner.residual, runner.b2.ptr) ||
            lingbot_vae_add_whdc_f32(
                runner.b2.ptr, runner.shortcut.ptr, runner.b2.ptr, b2_elems, nullptr) != 0) {
            return false;
        }
        log_stage("down3_resnets_add", v);
        if (v > 0 && (latent_W_single != w2 || latent_H != h2 || latent_T != t2)) {
            return false;
        }
        const size_t enc_view_stride = (size_t) w2 * h2 * t2 * 96;
        if (!vae_resnet_stream_cuda_device_execute(
                vg, "vae.encoder.mid_block.resnets.0",
                runner.b2.ptr, w2, h2, t2, 640, 640, view_stream,
                runner.norm, runner.conv, runner.residual, runner.a.ptr)) {
            return false;
        }
        log_stage("mid_res0", v);
        if (!vae_mid_attention_cuda_device_execute(
                vg, "vae.encoder.mid_block.attentions.0",
                runner.a.ptr, w2, h2, t2, 640, runner.b.ptr)) {
            return false;
        }
        log_stage("mid_attention", v);
        if (!vae_resnet_stream_cuda_device_execute(
                vg, "vae.encoder.mid_block.resnets.1",
                runner.b.ptr, w2, h2, t2, 640, 640, view_stream,
                runner.norm, runner.conv, runner.residual, runner.a.ptr)) {
            return false;
        }
        log_stage("mid_res1", v);
        if (!vae_encoder_tail_stream_cuda_device_execute(
                vg, runner.a.ptr, w2, h2, t2, 640, view_stream,
                runner.norm, runner.conv,
                runner.enc_views.ptr + (size_t) v * enc_view_stride)) {
            return false;
        }
        log_stage("tail_quant", v);
        if (v == 0) {
            latent_W_single = w2;
            latent_H = h2;
            latent_T = t2;
        }
    }

    out_shape = {1, z_dim, latent_T, latent_H, latent_W_single * views};
    const size_t out_elems = (size_t) out_shape.c * out_shape.f * out_shape.h * out_shape.w;
    out_bcfhw.assign(out_elems, 0.0f);
    if (lingbot_vae_normalize_cat_views_whdc_to_bcfhw_f32(
            runner.enc_views.ptr, runner.latent_mean.ptr, runner.latent_inv_std.ptr,
            runner.out_latent.ptr, views, latent_W_single, latent_H, latent_T, z_dim,
            nullptr) != 0 ||
        cudaMemcpy(out_bcfhw.data(), runner.out_latent.ptr,
                   out_elems * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        return false;
    }
    log_stage("normalize_cat_d2h");
    if (output_chunks) {
        output_chunks->clear();
        output_chunks->push_back(latent_T);
    }
    std::printf("vla(lingbot_va): VAE persistent LIBERO bridge ok views=%d input=[3,%d,%d,%d] latent=[1,%d,%d,%d,%d] checksum=%.9g\n",
                views, frames, height, width,
                (int) out_shape.c, (int) out_shape.f, (int) out_shape.h, (int) out_shape.w,
                checksum(out_bcfhw));
    return true;
}
#endif

bool encode_lingbot_video_to_latent(
        const std::string & vae_path,
        const float * video_vcfhw,
        int64_t views,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        const std::vector<int> * input_chunks,
        std::vector<int> * output_chunks,
        LingBotVaeStreamingSessionCache * stream_cache,
        std::vector<float> & out_bcfhw,
        LingBotTensor5DShape & out_shape) {
    if (!video_vcfhw || views <= 0 || channels != 3 || frames <= 0 || height <= 0 || width <= 0) {
        std::fprintf(stderr,
                     "vla(lingbot_va): invalid LingBot video shape views=%lld c=%lld f=%lld h=%lld w=%lld\n",
                     (long long) views, (long long) channels, (long long) frames,
                     (long long) height, (long long) width);
        return false;
    }
    if (height % 2 != 0 || width % 2 != 0) {
        std::fprintf(stderr,
                     "vla(lingbot_va): VAE video height/width must be divisible by patch_size=2, got %lldx%lld\n",
                     (long long) height, (long long) width);
        return false;
    }
    LingBotVaeEncoderContext * vae_ctx = lingbot_get_vae_encoder_context(vae_path);
    if (!vae_ctx) return false;
    std::lock_guard<std::mutex> vae_lock(vae_ctx->mu);
    gguf_reader & vg = vae_ctx->reader;
    const int z_dim = vae_ctx->z_dim;
    const std::vector<float> & latents_mean = vae_ctx->latents_mean;
    const std::vector<float> & latents_std = vae_ctx->latents_std;

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (lingbot_vae_persistent_encoder_requested()) {
        if (encode_lingbot_video_to_latent_persistent_libero(
                vg, vae_path, video_vcfhw, views, channels, frames, height, width,
                stream_cache, latents_mean, latents_std, z_dim,
                output_chunks, out_bcfhw, out_shape)) {
            return true;
        }
        if (lingbot_env_enabled("VLA_LINGBOT_VAE_PERSISTENT_ENCODER_REQUIRED")) {
            std::fprintf(stderr, "vla(lingbot_va): VAE persistent LIBERO encoder failed\n");
            return false;
        }
        std::fprintf(stderr, "vla(lingbot_va): VAE persistent LIBERO encoder unavailable; falling back\n");
    }
#endif

    int latent_W_single = 0;
    int latent_H = 0;
    int latent_T = 0;
    std::vector<int> latent_chunks;
    std::vector<std::vector<float>> view_latents;
    view_latents.reserve((size_t) views);
    const size_t view_elems = (size_t) channels * (size_t) frames * (size_t) height * (size_t) width;
    for (int64_t v = 0; v < views; ++v) {
        const float * src = video_vcfhw + (size_t) v * view_elems;
        std::vector<float> raw(src, src + view_elems);
        std::vector<float> patch;
        int patch_W = 0;
        int patch_H = 0;
        if (!vae_patchify_rgb_bcfhw_to_whdc(raw, (int) height, (int) width, (int) frames,
                                            patch, &patch_W, &patch_H)) {
            return false;
        }

        std::vector<float> b0;
        std::vector<float> b1;
        std::vector<float> b2;
        int w0 = 0, h0 = 0, w1 = 0, h1 = 0, t1 = 0, w2 = 0, h2 = 0, t2 = 0;
        std::vector<int> chunks2;
        LingBotVaeStreamingViewCache * view_stream = nullptr;
        if (stream_cache && v >= 0 && (size_t) v < stream_cache->views.size()) {
            view_stream = &stream_cache->views[(size_t) v];
            view_stream->begin();
        }
        if (view_stream) {
            if (!vae_encoder_down_path_stream_execute(vg, patch, patch_W, patch_H, (int) frames,
                                                      *view_stream, b2, &w2, &h2, &t2)) {
                return false;
            }
            chunks2.push_back(t2);
        } else {
            if (!vae_encoder_down_path_execute(vg, patch, patch_W, patch_H, (int) frames,
                                               input_chunks,
                                               b0, &w0, &h0, b1, &w1, &h1, &t1,
                                               b2, &w2, &h2, &t2, &chunks2)) {
                return false;
            }
        }

        std::vector<float> enc96;
        if (view_stream) {
            if (!vae_encoder_mid_tail_stream_execute(vg, b2, w2, h2, t2, 640, *view_stream, enc96)) {
                return false;
            }
        } else {
            if (!vae_encoder_mid_tail_execute(vg, b2, w2, h2, t2, 640, &chunks2,
                                              enc96, nullptr, nullptr, nullptr)) {
                return false;
            }
        }
        if ((int64_t) enc96.size() != (int64_t) w2 * h2 * t2 * (2 * z_dim)) {
            std::fprintf(stderr,
                         "vla(lingbot_va): VAE encoder output size mismatch for view %lld\n",
                         (long long) v);
            return false;
        }
        if (v == 0) {
            latent_W_single = w2;
            latent_H = h2;
            latent_T = t2;
            latent_chunks = chunks2;
        } else if (latent_W_single != w2 || latent_H != h2 || latent_T != t2) {
            std::fprintf(stderr, "vla(lingbot_va): VAE view latent shapes differ\n");
            return false;
        }

        std::vector<float> mu_norm((size_t) z_dim * (size_t) latent_T *
                                   (size_t) latent_H * (size_t) latent_W_single, 0.0f);
        for (int c = 0; c < z_dim; ++c) {
            const float inv_std = 1.0f / latents_std[(size_t) c];
            for (int t = 0; t < latent_T; ++t) {
                for (int h = 0; h < latent_H; ++h) {
                    for (int w = 0; w < latent_W_single; ++w) {
                        const float mu = enc96[vae_ggml_whdc_index(w, h, t, c,
                                                                    latent_W_single,
                                                                    latent_H,
                                                                    latent_T)];
                        mu_norm[idx5(LingBotTensor5DShape{1, z_dim, latent_T, latent_H, latent_W_single},
                                     0, c, t, h, w)] =
                            (mu - latents_mean[(size_t) c]) * inv_std;
                    }
                }
            }
        }
        view_latents.push_back(std::move(mu_norm));
    }

    out_shape = {1, z_dim, latent_T, latent_H, latent_W_single * views};
    out_bcfhw.assign((size_t) out_shape.c * (size_t) out_shape.f *
                     (size_t) out_shape.h * (size_t) out_shape.w, 0.0f);
    const LingBotTensor5DShape single_shape{1, z_dim, latent_T, latent_H, latent_W_single};
    for (int64_t v = 0; v < views; ++v) {
        for (int c = 0; c < z_dim; ++c) {
            for (int t = 0; t < latent_T; ++t) {
                for (int h = 0; h < latent_H; ++h) {
                    for (int w = 0; w < latent_W_single; ++w) {
                        out_bcfhw[idx5(out_shape, 0, c, t, h, v * latent_W_single + w)] =
                            view_latents[(size_t) v][idx5(single_shape, 0, c, t, h, w)];
                    }
                }
            }
        }
    }
    std::printf("vla(lingbot_va): VAE image bridge ok views=%lld input=[3,%lld,%lld,%lld] latent=[1,%lld,%lld,%lld,%lld] checksum=%.9g\n",
                (long long) views, (long long) frames, (long long) height, (long long) width,
                (long long) out_shape.c, (long long) out_shape.f, (long long) out_shape.h,
                (long long) out_shape.w, checksum(out_bcfhw));
    if (output_chunks) *output_chunks = std::move(latent_chunks);
    return true;
}

bool encode_lingbot_robotwin_tshape_video_to_latent(
        const std::string & vae_path,
        const float * video_views_cfhw,
        int64_t views,
        int64_t channels,
        const int64_t * view_frames,
        const int64_t * view_heights,
        const int64_t * view_widths,
        const std::vector<int> * input_chunks,
        std::vector<int> * output_chunks,
        LingBotVaeStreamingSessionCache * stream_cache,
        std::vector<float> & out_bcfhw,
        LingBotTensor5DShape & out_shape) {
    if (!video_views_cfhw || views != 3 || channels != 3 ||
        !view_frames || !view_heights || !view_widths) {
        std::fprintf(stderr,
                     "vla(lingbot_va): robotwin_tshape VAE expects 3 RGB views with per-view shapes\n");
        return false;
    }
    const int64_t frames = view_frames[0];
    if (frames <= 0 || view_frames[1] != frames || view_frames[2] != frames) {
        std::fprintf(stderr, "vla(lingbot_va): robotwin_tshape views must share frame count\n");
        return false;
    }
    if (view_heights[1] != view_heights[2] || view_widths[1] != view_widths[2]) {
        std::fprintf(stderr, "vla(lingbot_va): robotwin_tshape wrist views must share H,W\n");
        return false;
    }

    size_t offsets[3] = {0, 0, 0};
    size_t cursor = 0;
    for (int64_t v = 0; v < views; ++v) {
        if (view_heights[v] <= 0 || view_widths[v] <= 0) return false;
        offsets[v] = cursor;
        cursor += (size_t) channels * (size_t) view_frames[v] *
                  (size_t) view_heights[v] * (size_t) view_widths[v];
    }

    auto encode_one = [&](int64_t view_index,
                          std::vector<float> & latent,
                          LingBotTensor5DShape & shape,
                          std::vector<int> & chunks) -> bool {
        LingBotVaeStreamingSessionCache one_cache;
        LingBotVaeStreamingSessionCache * one_cache_ptr = nullptr;
        if (stream_cache) {
            if (view_index < 0 || (size_t) view_index >= stream_cache->views.size()) {
                return false;
            }
            one_cache.views.resize(1);
            one_cache.views[0] = std::move(stream_cache->views[(size_t) view_index]);
            one_cache_ptr = &one_cache;
        }
        std::vector<int> local_chunks;
        const bool ok = encode_lingbot_video_to_latent(
            vae_path,
            video_views_cfhw + offsets[(size_t) view_index],
            1,
            channels,
            view_frames[view_index],
            view_heights[view_index],
            view_widths[view_index],
            input_chunks,
            &local_chunks,
            one_cache_ptr,
            latent,
            shape);
        if (stream_cache && one_cache_ptr) {
            stream_cache->views[(size_t) view_index] = std::move(one_cache.views[0]);
        }
        if (!ok) return false;
        chunks = std::move(local_chunks);
        return true;
    };

    std::vector<float> high;
    std::vector<float> left;
    std::vector<float> right;
    LingBotTensor5DShape high_shape{};
    LingBotTensor5DShape left_shape{};
    LingBotTensor5DShape right_shape{};
    std::vector<int> high_chunks;
    std::vector<int> left_chunks;
    std::vector<int> right_chunks;
    if (!encode_one(0, high, high_shape, high_chunks) ||
        !encode_one(1, left, left_shape, left_chunks) ||
        !encode_one(2, right, right_shape, right_chunks)) {
        return false;
    }
    if (high_shape.b != 1 || left_shape.b != 1 || right_shape.b != 1 ||
        high_shape.c != left_shape.c || high_shape.c != right_shape.c ||
        high_shape.f != left_shape.f || high_shape.f != right_shape.f ||
        left_shape.h != right_shape.h || left_shape.w != right_shape.w) {
        std::fprintf(stderr, "vla(lingbot_va): robotwin_tshape latent branch shape mismatch\n");
        return false;
    }
    if (left_shape.w + right_shape.w != high_shape.w) {
        std::fprintf(stderr,
                     "vla(lingbot_va): robotwin_tshape wrist latent width %lld+%lld != high width %lld\n",
                     (long long) left_shape.w, (long long) right_shape.w, (long long) high_shape.w);
        return false;
    }

    out_shape = {1, high_shape.c, high_shape.f,
                 left_shape.h + high_shape.h, high_shape.w};
    out_bcfhw.assign((size_t) out_shape.c * (size_t) out_shape.f *
                     (size_t) out_shape.h * (size_t) out_shape.w, 0.0f);

    for (int64_t c = 0; c < out_shape.c; ++c) {
        for (int64_t f = 0; f < out_shape.f; ++f) {
            for (int64_t h = 0; h < left_shape.h; ++h) {
                for (int64_t w = 0; w < left_shape.w; ++w) {
                    out_bcfhw[idx5(out_shape, 0, c, f, h, w)] =
                        left[idx5(left_shape, 0, c, f, h, w)];
                    out_bcfhw[idx5(out_shape, 0, c, f, h, left_shape.w + w)] =
                        right[idx5(right_shape, 0, c, f, h, w)];
                }
            }
            for (int64_t h = 0; h < high_shape.h; ++h) {
                for (int64_t w = 0; w < high_shape.w; ++w) {
                    out_bcfhw[idx5(out_shape, 0, c, f, left_shape.h + h, w)] =
                        high[idx5(high_shape, 0, c, f, h, w)];
                }
            }
        }
    }
    if (output_chunks) *output_chunks = std::move(high_chunks);
    std::printf("vla(lingbot_va): robotwin_tshape VAE image bridge ok "
                "high=[3,%lld,%lld,%lld] wrist=[3,%lld,%lld,%lld] "
                "latent=[1,%lld,%lld,%lld,%lld] checksum=%.9g\n",
                (long long) view_frames[0], (long long) view_heights[0], (long long) view_widths[0],
                (long long) view_frames[1], (long long) view_heights[1], (long long) view_widths[1],
                (long long) out_shape.c, (long long) out_shape.f,
                (long long) out_shape.h, (long long) out_shape.w,
                checksum(out_bcfhw));
    return true;
}

bool vae_decoder_post_quant_conv_in_ggml_execute(
        gguf_reader & g,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        std::vector<float> & out_whdc) {
    const int in_C = 48;
    const int hid_C = 1024;
    if (W <= 0 || H <= 0 || T <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * in_C) return false;

    ggml_init_params params = { size_t(512) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(params);
    if (!C) return false;

    auto new_conv3_w = [&](const std::string & name, int64_t out_ch, int64_t in_ch,
                           int64_t kt, int64_t kh, int64_t kw) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, kw, kh, kt, out_ch * in_ch);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_bias3 = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, 1, channels);
        ggml_set_name(t, name.c_str());
        return t;
    };

    ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, W, H, T, in_C);
    ggml_set_name(x, "vae.decoder.post_quant.input");
    ggml_tensor * pq_w = new_conv3_w("vae.post_quant_conv.weight", in_C, in_C, 1, 1, 1);
    ggml_tensor * pq_b = new_bias3("vae.post_quant_conv.bias", in_C);
    ggml_tensor * ci_w = new_conv3_w("vae.decoder.conv_in.weight", hid_C, in_C, 3, 3, 3);
    ggml_tensor * ci_b = new_bias3("vae.decoder.conv_in.bias", hid_C);

    ggml_tensor * h = ggml_add(C, ggml_conv_3d(C, pq_w, x, in_C, 1, 1, 1, 0, 0, 0, 1, 1, 1), pq_b);
    h = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, ci_w, h, in_C), ci_b);
    ggml_set_name(h, "vae.decoder.conv_in.output");
    ggml_set_output(h);

    LingBotAuxGraphBackend graph_backend;
    if (!lingbot_aux_graph_backend_init(graph_backend, "vae.ggml")) {
        ggml_free(C);
        return false;
    }
    ggml_backend_t backend = graph_backend.backend;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, graph_backend.buft);
    if (!buf) {
        lingbot_aux_graph_backend_free(graph_backend);
        ggml_free(C);
        return false;
    }

    std::vector<ggml_tensor *> weights = { pq_w, pq_b, ci_w, ci_b };
    bool ok = true;
    for (ggml_tensor * t : weights) {
        if (!set_tensor_from_gguf_f32(g, t)) {
            ok = false;
            break;
        }
    }
    if (ok) {
        ggml_backend_tensor_set(x, in_whdc.data(), 0, in_whdc.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(gf, h);
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        ok = st == GGML_STATUS_SUCCESS;
        if (!ok) {
            std::fprintf(stderr, "vla(lingbot_va): VAE decoder post_quant/conv_in graph failed (%d)\n", (int) st);
        }
    }
    if (ok) {
        out_whdc.assign((size_t) ggml_nelements(h), 0.0f);
        ggml_backend_tensor_get(h, out_whdc.data(), 0, out_whdc.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    lingbot_aux_graph_backend_free(graph_backend);
    ggml_free(C);
    return ok;
}



bool vae_decoder_mid_execute(
        gguf_reader & g,
        const std::vector<float> & z_whdc,
        int W,
        int H,
        int T,
        std::vector<float> & out_whdc) {
    std::vector<float> h;
    if (!vae_decoder_post_quant_conv_in_ggml_execute(g, z_whdc, W, H, T, h)) return false;
    std::vector<float> r0;
    if (!vae_mid_resnet_ggml_execute(g, "vae.decoder.mid_block.resnets.0", h, W, H, T, 1024, r0)) return false;
    std::vector<float> attn;
    if (!vae_mid_attention_execute(g, "vae.decoder.mid_block.attentions.0", r0, W, H, T, 1024, attn)) return false;
    return vae_mid_resnet_ggml_execute(g, "vae.decoder.mid_block.resnets.1", attn, W, H, T, 1024, out_whdc);
}

bool vae_decoder_up_block_resnets_execute(
        gguf_reader & g,
        int block,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        std::vector<float> & out_whdc) {
    std::vector<float> h = in_whdc;
    int cur_C = in_C;
    for (int r = 0; r < 3; ++r) {
        const int next_C = r == 0 ? out_C : cur_C;
        const std::string prefix = "vae.decoder.up_blocks." + std::to_string(block) + ".resnets." + std::to_string(r);
        std::vector<float> next;
        if (!vae_resnet_ggml_execute(g, prefix.c_str(), h, W, H, T, cur_C, next_C, next)) {
            std::fprintf(stderr, "vla(lingbot_va): VAE decoder up block %d failed at resnet %d\n", block, r);
            return false;
        }
        h.swap(next);
        cur_C = next_C;
    }
    out_whdc.swap(h);
    return true;
}

bool vae_decoder_spatial_upsample_ggml_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        std::vector<float> & out_whdc,
        int * out_W,
        int * out_H) {
    if (W <= 0 || H <= 0 || T <= 0 || Cc <= 0) return false;
    if (in_whdc.size() != (size_t) W * H * T * Cc) return false;

    ggml_init_params params = { size_t(512) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(params);
    if (!C) return false;

    const std::string w_name = std::string(prefix) + ".resample.1.weight";
    const std::string b_name = std::string(prefix) + ".resample.1.bias";
    ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, W, H, T, Cc);
    ggml_set_name(x, "vae.decoder.upsample.input");
    ggml_tensor * w = ggml_new_tensor_4d(C, GGML_TYPE_F32, 3, 3, Cc, Cc);
    ggml_set_name(w, w_name.c_str());
    ggml_tensor * b = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, Cc, 1);
    ggml_set_name(b, b_name.c_str());

    ggml_tensor * x_w_h_c_t = ggml_cont(C, ggml_permute(C, x, 0, 1, 3, 2));
    ggml_tensor * up = ggml_interpolate(C, x_w_h_c_t, W * 2, H * 2, Cc, T, GGML_SCALE_MODE_NEAREST);
    up = ggml_conv_2d(C, w, up, 1, 1, 1, 1, 1, 1);
    up = ggml_add(C, up, b);
    ggml_tensor * out = ggml_cont(C, ggml_permute(C, up, 0, 1, 3, 2));
    ggml_set_name(out, "vae.decoder.upsample.output");
    ggml_set_output(out);

    LingBotAuxGraphBackend graph_backend;
    if (!lingbot_aux_graph_backend_init(graph_backend, "vae.ggml")) {
        ggml_free(C);
        return false;
    }
    ggml_backend_t backend = graph_backend.backend;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, graph_backend.buft);
    if (!buf) {
        lingbot_aux_graph_backend_free(graph_backend);
        ggml_free(C);
        return false;
    }

    bool ok = set_tensor_from_gguf_f32(g, w) && set_tensor_from_gguf_f32(g, b);
    if (ok) {
        ggml_backend_tensor_set(x, in_whdc.data(), 0, in_whdc.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(gf, out);
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        ok = st == GGML_STATUS_SUCCESS;
        if (!ok) {
            std::fprintf(stderr, "vla(lingbot_va): VAE decoder spatial upsample graph failed (%d) for %s\n",
                         (int) st, prefix);
        }
    }
    if (ok) {
        out_whdc.assign((size_t) ggml_nelements(out), 0.0f);
        ggml_backend_tensor_get(out, out_whdc.data(), 0, out_whdc.size() * sizeof(float));
        if (out_W) *out_W = (int) out->ne[0];
        if (out_H) *out_H = (int) out->ne[1];
    }

    ggml_backend_buffer_free(buf);
    lingbot_aux_graph_backend_free(graph_backend);
    ggml_free(C);
    return ok;
}

void vae_decoder_temporal_x2_unpack(
        const std::vector<float> & in_whd_2c,
        std::vector<float> & out_whdc,
        int W,
        int H,
        int T,
        int Cc) {
    out_whdc.assign((size_t) W * H * T * 2 * Cc, 0.0f);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int t = 0; t < T; ++t) {
                for (int split = 0; split < 2; ++split) {
                    for (int c = 0; c < Cc; ++c) {
                        out_whdc[vae_ggml_whdc_index(w, h, 2 * t + split, c, W, H, 2 * T)] =
                            in_whd_2c[vae_ggml_whdc_index(w, h, t, split * Cc + c, W, H, T)];
                    }
                }
            }
        }
    }
}

bool vae_decoder_temporal_upsample_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        std::vector<float> & out_whdc,
        int * out_T) {
    const std::string time_prefix = std::string(prefix) + ".time_conv";
    LingBotVaeCudaTimeConvWeightsIO weights;
    if (!weights.load(g, time_prefix.c_str(), Cc, 2 * Cc, 3)) return false;
    LingBotVaeCudaTemporalConvBatchedCache cache;
    std::vector<float> tmp;
    if (!vae_time_conv_cuda_whdc_execute_io(in_whdc, tmp, weights, cache, W, H, T, {T})) {
        return false;
    }
    vae_decoder_temporal_x2_unpack(tmp, out_whdc, W, H, T, Cc);
    if (out_T) *out_T = 2 * T;
    return true;
}

bool vae_decoder_temporal_upsample_stream_execute(
        gguf_reader & g,
        const char * prefix,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        const std::vector<int> & in_chunks,
        std::vector<float> & out_whdc,
        std::vector<int> & out_chunks,
        int * out_T) {
    int chunk_sum = 0;
    for (int c : in_chunks) {
        if (c <= 0) return false;
        chunk_sum += c;
    }
    if (chunk_sum != T || in_chunks.empty()) return false;

    out_chunks.clear();
    out_chunks.reserve(in_chunks.size());
    int total_out = 0;
    for (size_t i = 0; i < in_chunks.size(); ++i) {
        const int oc = i == 0 ? in_chunks[i] : 2 * in_chunks[i];
        out_chunks.push_back(oc);
        total_out += oc;
    }
    out_whdc.assign((size_t) W * H * total_out * Cc, 0.0f);

    int in_off = 0;
    int out_off = 0;
    std::vector<float> first;
    if (!vae_slice_time_chunk(in_whdc, first, W, H, T, Cc, 0, in_chunks[0])) return false;
    vae_append_time_chunk(out_whdc, first, W, H, total_out, Cc, 0, in_chunks[0]);
    in_off += in_chunks[0];
    out_off += in_chunks[0];

    if (in_chunks.size() > 1) {
        std::vector<int> rest_chunks(in_chunks.begin() + 1, in_chunks.end());
        int rest_T = 0;
        for (int c : rest_chunks) rest_T += c;
        std::vector<float> rest;
        if (!vae_slice_time_chunk(in_whdc, rest, W, H, T, Cc, in_off, rest_T)) return false;

        const std::string time_prefix = std::string(prefix) + ".time_conv";
        LingBotVaeCudaTimeConvWeightsIO weights;
        if (!weights.load(g, time_prefix.c_str(), Cc, 2 * Cc, 3)) return false;
        LingBotVaeCudaTemporalConvBatchedCache cache;
        std::vector<float> tmp;
        if (!vae_time_conv_cuda_whdc_execute_io(rest, tmp, weights, cache, W, H, rest_T, rest_chunks)) {
            return false;
        }
        std::vector<float> unpacked;
        vae_decoder_temporal_x2_unpack(tmp, unpacked, W, H, rest_T, Cc);
        vae_append_time_chunk(out_whdc, unpacked, W, H, total_out, Cc, out_off, 2 * rest_T);
    }

    if (out_T) *out_T = total_out;
    return true;
}

bool vae_dup_up3d_stream_shortcut(
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        int W,
        int H,
        int T,
        int in_C,
        int out_C,
        int factor_t,
        int factor_s,
        const std::vector<int> & in_chunks,
        std::vector<int> & out_chunks,
        int * out_W,
        int * out_H,
        int * out_T) {
    int chunk_sum = 0;
    for (int c : in_chunks) {
        if (c <= 0) return false;
        chunk_sum += c;
    }
    if (chunk_sum != T || in_chunks.empty()) return false;
    out_chunks.clear();
    out_chunks.reserve(in_chunks.size());
    int total_out = 0;
    for (size_t i = 0; i < in_chunks.size(); ++i) {
        const int out_chunk = in_chunks[i] * factor_t - (i == 0 ? factor_t - 1 : 0);
        out_chunks.push_back(out_chunk);
        total_out += out_chunk;
    }
    const int Wo = W * factor_s;
    const int Ho = H * factor_s;
    out_whdc.assign((size_t) Wo * Ho * total_out * out_C, 0.0f);
    int src_off = 0;
    int dst_off = 0;
    for (size_t i = 0; i < in_chunks.size(); ++i) {
        std::vector<float> chunk;
        if (!vae_slice_time_chunk(in_whdc, chunk, W, H, T, in_C, src_off, in_chunks[i])) return false;
        std::vector<float> up;
        int uw = 0, uh = 0, ut = 0;
        if (!vae_dup_up3d_host(chunk, up, W, H, in_chunks[i], in_C, out_C,
                               factor_t, factor_s, i == 0, &uw, &uh, &ut) ||
            uw != Wo || uh != Ho || ut != out_chunks[i]) {
            return false;
        }
        vae_append_time_chunk(out_whdc, up, Wo, Ho, total_out, out_C, dst_off, ut);
        src_off += in_chunks[i];
        dst_off += ut;
    }
    if (out_W) *out_W = Wo;
    if (out_H) *out_H = Ho;
    if (out_T) *out_T = total_out;
    return true;
}

bool vae_decoder_tail_ggml_execute(
        gguf_reader & g,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
        std::vector<float> & out_whdc) {
    if (W <= 0 || H <= 0 || T <= 0 || Cc != 256) return false;
    if (in_whdc.size() != (size_t) W * H * T * Cc) return false;

    const int out_C = 12;
    ggml_init_params params = { size_t(512) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(params);
    if (!C) return false;

    auto new_conv3_w = [&](const std::string & name, int64_t out_ch, int64_t in_ch) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, 3, 3, 3, out_ch * in_ch);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_bias3 = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, 1, 1, channels);
        ggml_set_name(t, name.c_str());
        return t;
    };
    auto new_gamma = [&](const std::string & name, int64_t channels) {
        ggml_tensor * t = ggml_new_tensor_4d(C, GGML_TYPE_F32, channels, 1, 1, 1);
        ggml_set_name(t, name.c_str());
        return t;
    };

    ggml_tensor * x = ggml_new_tensor_4d(C, GGML_TYPE_F32, W, H, T, Cc);
    ggml_set_name(x, "vae.decoder.tail.input");
    ggml_tensor * norm = new_gamma("vae.decoder.norm_out.gamma", Cc);
    ggml_tensor * conv_w = new_conv3_w("vae.decoder.conv_out.weight", out_C, Cc);
    ggml_tensor * conv_b = new_bias3("vae.decoder.conv_out.bias", out_C);
    ggml_tensor * h = vae_norm_silu_to_conv_layout(C, x, norm);
    h = ggml_add(C, vae_causal_conv3d_ks3_pad1(C, conv_w, h, Cc), conv_b);
    ggml_set_name(h, "vae.decoder.tail.output");
    ggml_set_output(h);

    LingBotAuxGraphBackend graph_backend;
    if (!lingbot_aux_graph_backend_init(graph_backend, "vae.ggml")) {
        ggml_free(C);
        return false;
    }
    ggml_backend_t backend = graph_backend.backend;
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, graph_backend.buft);
    if (!buf) {
        lingbot_aux_graph_backend_free(graph_backend);
        ggml_free(C);
        return false;
    }

    bool ok = set_tensor_from_gguf_f32(g, norm) && set_tensor_from_gguf_f32(g, conv_w) && set_tensor_from_gguf_f32(g, conv_b);
    if (ok) {
        ggml_backend_tensor_set(x, in_whdc.data(), 0, in_whdc.size() * sizeof(float));
        ggml_cgraph * gf = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(gf, h);
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        ok = st == GGML_STATUS_SUCCESS;
        if (!ok) {
            std::fprintf(stderr, "vla(lingbot_va): VAE decoder tail graph failed (%d)\n", (int) st);
        }
    }
    if (ok) {
        out_whdc.assign((size_t) ggml_nelements(h), 0.0f);
        ggml_backend_tensor_get(h, out_whdc.data(), 0, out_whdc.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    lingbot_aux_graph_backend_free(graph_backend);
    ggml_free(C);
    return ok;
}

void vae_unpatchify_ps2_rgb(
        const std::vector<float> & in_whdc,
        std::vector<float> & out_whdc,
        int W,
        int H,
        int T) {
    const int ps = 2;
    const int C_img = 3;
    out_whdc.assign((size_t) (W * ps) * (H * ps) * T * C_img, 0.0f);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int t = 0; t < T; ++t) {
                for (int c = 0; c < C_img; ++c) {
                    for (int pw = 0; pw < ps; ++pw) {
                        for (int ph = 0; ph < ps; ++ph) {
                            const int pc = ((c * ps + pw) * ps + ph);
                            const int ow = w * ps + pw;
                            const int oh = h * ps + ph;
                            out_whdc[vae_ggml_whdc_index(ow, oh, t, c, W * ps, H * ps, T)] =
                                in_whdc[vae_ggml_whdc_index(w, h, t, pc, W, H, T)];
                        }
                    }
                }
            }
        }
    }
}

bool vae_decoder_full_execute(
        gguf_reader & g,
        const std::vector<float> & z_whdc,
        int W,
        int H,
        int T,
        std::vector<float> & patch_whdc,
        std::vector<float> * image_whdc,
        int * out_W,
        int * out_H,
        int * out_T) {
    std::vector<float> h;
    if (!vae_decoder_mid_execute(g, z_whdc, W, H, T, h)) return false;
    int cur_W = W;
    int cur_H = H;
    int cur_T = T;
    std::vector<int> chunks;
    chunks.reserve((size_t) T);
    for (int i = 0; i < T; ++i) chunks.push_back(1);
    struct Case { int block; int in_C; int out_C; bool temporal; bool spatial; };
    const Case cases[] = {
        {0, 1024, 1024, true,  true},
        {1, 1024, 1024, true,  true},
        {2, 1024, 512,  false, true},
        {3, 512,  256,  false, false},
    };
    for (const Case & c : cases) {
        std::vector<float> shortcut_input = h;
        std::vector<int> shortcut_chunks_in = chunks;
        const int shortcut_W = cur_W;
        const int shortcut_H = cur_H;
        const int shortcut_T = cur_T;
        std::vector<float> r;
        if (!vae_decoder_up_block_resnets_execute(g, c.block, h, cur_W, cur_H, cur_T, c.in_C, c.out_C, r)) return false;
        h.swap(r);
        if (c.temporal) {
            std::vector<float> tu;
            std::vector<int> next_chunks;
            int next_T = 0;
            const std::string prefix = "vae.decoder.up_blocks." + std::to_string(c.block) + ".upsampler";
            if (!vae_decoder_temporal_upsample_stream_execute(g, prefix.c_str(), h, cur_W, cur_H, cur_T, c.out_C,
                                                              chunks, tu, next_chunks, &next_T)) return false;
            h.swap(tu);
            chunks.swap(next_chunks);
            cur_T = next_T;
        }
        if (c.spatial) {
            std::vector<float> su;
            int next_W = 0;
            int next_H = 0;
            const std::string prefix = "vae.decoder.up_blocks." + std::to_string(c.block) + ".upsampler";
            if (!vae_decoder_spatial_upsample_ggml_execute(g, prefix.c_str(), h, cur_W, cur_H, cur_T, c.out_C, su, &next_W, &next_H)) return false;
            h.swap(su);
            cur_W = next_W;
            cur_H = next_H;
        }
        if (c.spatial) {
            std::vector<float> shortcut;
            std::vector<int> shortcut_chunks;
            int sc_W = 0, sc_H = 0, sc_T = 0;
            const int factor_t = c.temporal ? 2 : 1;
            if (!vae_dup_up3d_stream_shortcut(shortcut_input, shortcut,
                                              shortcut_W, shortcut_H, shortcut_T,
                                              c.in_C, c.out_C, factor_t, 2,
                                              shortcut_chunks_in, shortcut_chunks,
                                              &sc_W, &sc_H, &sc_T) ||
                sc_W != cur_W || sc_H != cur_H || sc_T != cur_T ||
                !vae_add_same_shape(h, shortcut)) {
                std::fprintf(stderr, "vla(lingbot_va): VAE decoder up block %d DupUp3D shortcut failed\n", c.block);
                return false;
            }
        }
    }
    if (!vae_decoder_tail_ggml_execute(g, h, cur_W, cur_H, cur_T, 256, patch_whdc)) return false;
    if (image_whdc) {
        vae_unpatchify_ps2_rgb(patch_whdc, *image_whdc, cur_W, cur_H, cur_T);
    }
    if (out_W) *out_W = cur_W;
    if (out_H) *out_H = cur_H;
    if (out_T) *out_T = cur_T;
    return true;
}




#endif

#ifndef VLA_LINGBOT_FLEX_CUDA_KERNELS

struct LingBotVaeStreamingSessionCache {};

LingBotVaeStreamingSessionCache * lingbot_get_vae_stream_cache(uint64_t, int64_t) {
    return nullptr;
}

bool encode_lingbot_video_to_latent(
        const std::string &,
        const float *,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        const std::vector<int> *,
        std::vector<int> *,
        LingBotVaeStreamingSessionCache *,
        std::vector<float> &,
        LingBotTensor5DShape &) {
    std::fprintf(stderr,
                 "vla(lingbot_va): LingBot video VAE encode requires "
                 "MODEL_BUILD_WAM_LINGBOT_VA=ON with GGML_CUDA=ON\n");
    return false;
}

bool encode_lingbot_robotwin_tshape_video_to_latent(
        const std::string &,
        const float *,
        int64_t,
        int64_t,
        const int64_t *,
        const int64_t *,
        const int64_t *,
        const std::vector<int> *,
        std::vector<int> *,
        LingBotVaeStreamingSessionCache *,
        std::vector<float> &,
        LingBotTensor5DShape &) {
    std::fprintf(stderr,
                 "vla(lingbot_va): LingBot robotwin_tshape video VAE encode requires "
                 "MODEL_BUILD_WAM_LINGBOT_VA=ON with GGML_CUDA=ON\n");
    return false;
}

#endif

void dump_f32_file(const std::string & path, const std::vector<float> & data);
void dump_text_file(const std::string & path, const std::string & text);
void dump_i64_file(const std::string & path, const std::vector<int64_t> & data);
void dump_u8_file(const std::string & path, const std::vector<uint8_t> & data);

struct LingBotKVCache {
    int64_t batch = 0;
    int64_t total_tokens = 0;
    int64_t heads = 0;
    int64_t head_dim = 0;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<int64_t> id;
    std::vector<uint8_t> mask;
    std::vector<uint8_t> is_pred;

    void init(int64_t batch_, int64_t total_tokens_, int64_t heads_, int64_t head_dim_) {
        batch = batch_;
        total_tokens = total_tokens_;
        heads = heads_;
        head_dim = head_dim_;
        const size_t n = (size_t) batch * (size_t) total_tokens * (size_t) heads * (size_t) head_dim;
        k.assign(n, 0.0f);
        v.assign(n, 0.0f);
        id.assign((size_t) total_tokens, -1);
        mask.assign((size_t) total_tokens, 0);
        is_pred.assign((size_t) total_tokens, 0);
    }

    void clear() {
        k.clear();
        v.clear();
        id.clear();
        mask.clear();
        is_pred.clear();
        batch = total_tokens = heads = head_dim = 0;
    }

    int64_t used_count() const {
        return (int64_t) std::count(mask.begin(), mask.end(), (uint8_t) 1);
    }

    int64_t pred_count() const {
        int64_t n = 0;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] && is_pred[i]) ++n;
        }
        return n;
    }

    int64_t next_id() const {
        int64_t out = -1;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i]) out = std::max(out, id[i]);
        }
        return out + 1;
    }

    bool allocate_slots_into(
            int64_t key_size,
            std::vector<int64_t> & slots,
            std::vector<int64_t> & free,
            std::vector<int64_t> & used) {
        slots.clear();
        free.clear();
        used.clear();
        if (key_size <= 0) {
            return false;
        }
        for (int64_t i = 0; i < total_tokens; ++i) {
            if (!mask[(size_t) i]) free.push_back(i);
        }
        if ((int64_t) free.size() < key_size) {
            for (int64_t i = 0; i < total_tokens; ++i) {
                if (mask[(size_t) i]) used.push_back(i);
            }
            std::sort(used.begin(), used.end(), [&](int64_t a, int64_t b) {
                return id[(size_t) a] < id[(size_t) b];
            });
            const int64_t need = key_size - (int64_t) free.size();
            for (int64_t i = 0; i < need && i < (int64_t) used.size(); ++i) {
                const int64_t slot = used[(size_t) i];
                mask[(size_t) slot] = 0;
                id[(size_t) slot] = -1;
                if (lingbot_runtime_kv_legacy_clear_pred_enabled()) {
                    is_pred[(size_t) slot] = 0;
                }
                free.push_back(slot);
            }
            std::sort(free.begin(), free.end());
        }
        if ((int64_t) free.size() < key_size) {
            return false;
        }
        slots.assign(free.begin(), free.begin() + key_size);
        return true;
    }

    std::vector<int64_t> allocate_slots(int64_t key_size) {
        std::vector<int64_t> slots;
        std::vector<int64_t> free;
        std::vector<int64_t> used;
        (void) allocate_slots_into(key_size, slots, free, used);
        return slots;
    }

    size_t kv_offset(int64_t b, int64_t slot, int64_t h, int64_t d) const {
        return (((size_t) b * (size_t) total_tokens + (size_t) slot) *
                (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d;
    }

    bool update(const std::vector<float> & key, const std::vector<float> & value,
                int64_t key_size, bool pred, std::vector<int64_t> & slots) {
        std::vector<int64_t> free;
        std::vector<int64_t> used;
        return update_with_workspace(key, value, key_size, pred, slots, free, used);
    }

    bool update_with_workspace(const std::vector<float> & key,
                               const std::vector<float> & value,
                               int64_t key_size,
                               bool pred,
                               std::vector<int64_t> & slots,
                               std::vector<int64_t> & free,
                               std::vector<int64_t> & used) {
        const size_t expected = (size_t) batch * (size_t) key_size * (size_t) heads * (size_t) head_dim;
        if (key.size() != expected || value.size() != expected || key_size <= 0) {
            return false;
        }
        if (!allocate_slots_into(key_size, slots, free, used)) {
            return false;
        }
        if ((int64_t) slots.size() != key_size) {
            return false;
        }
        const int64_t new_id = next_id();
        const size_t token_floats = (size_t) heads * (size_t) head_dim;
        const size_t token_bytes = token_floats * sizeof(float);
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t i = 0; i < key_size; ++i) {
                const int64_t slot = slots[(size_t) i];
                const size_t src = ((size_t) b * (size_t) key_size + (size_t) i) * token_floats;
                const size_t dst = kv_offset(b, slot, 0, 0);
                std::memcpy(k.data() + dst, key.data() + src, token_bytes);
                std::memcpy(v.data() + dst, value.data() + src, token_bytes);
                mask[(size_t) slot] = 1;
                id[(size_t) slot] = new_id;
                is_pred[(size_t) slot] = pred ? 1 : 0;
            }
        }
        return true;
    }

    bool update_meta_only(int64_t key_size, bool pred, std::vector<int64_t> & slots) {
        if (batch != 1 || key_size <= 0) return false;
        slots = allocate_slots(key_size);
        if ((int64_t) slots.size() != key_size) {
            return false;
        }
        const int64_t new_id = next_id();
        for (int64_t i = 0; i < key_size; ++i) {
            const int64_t slot = slots[(size_t) i];
            mask[(size_t) slot] = 1;
            id[(size_t) slot] = new_id;
            is_pred[(size_t) slot] = pred ? 1 : 0;
        }
        return true;
    }

    bool prepare_mode0_temp_slots_no_insert(
            int64_t key_size,
            std::vector<int64_t> & slots,
            std::vector<int64_t> & valid_slots) {
        if (batch != 1 || key_size <= 0) return false;
        std::vector<int64_t> free;
        std::vector<int64_t> used;
        if (!allocate_slots_into(key_size, slots, free, used) ||
            (int64_t) slots.size() != key_size) {
            return false;
        }
        // Match Python's update_cache(..., update_cache=0) side effects without
        // making the temporary current K/V visible in the long-lived cache mask.
        if (!std::is_sorted(slots.begin(), slots.end()) ||
            std::adjacent_find(slots.begin(), slots.end()) != slots.end()) {
            return false;
        }
        for (int64_t slot : slots) {
            if (slot < 0 || slot >= total_tokens || mask[(size_t) slot]) {
                return false;
            }
        }

        const int64_t history_used = used_count();
        const int64_t new_id = next_id();
        for (int64_t slot : slots) {
            id[(size_t) slot] = new_id;
            is_pred[(size_t) slot] = 0;
        }

        valid_slots.clear();
        valid_slots.reserve(history_used + (int64_t) slots.size());
        for (int64_t i = 0; i < total_tokens; ++i) {
            if (mask[(size_t) i] ||
                std::binary_search(slots.begin(), slots.end(), i)) {
                valid_slots.push_back(i);
            }
        }
        return std::is_sorted(valid_slots.begin(), valid_slots.end()) &&
               (int64_t) valid_slots.size() == history_used + key_size;
    }

    void restore(const std::vector<int64_t> & slots) {
        for (int64_t slot : slots) {
            if (slot >= 0 && slot < total_tokens) {
                mask[(size_t) slot] = 0;
            }
        }
    }

    void clear_pred() {
        const bool legacy_clear_pred = lingbot_runtime_kv_legacy_clear_pred_enabled();
        for (size_t i = 0; i < mask.size(); ++i) {
            if (is_pred[i]) {
                mask[i] = 0;
                if (legacy_clear_pred) {
                    is_pred[i] = 0;
                    id[i] = -1;
                }
            }
        }
    }

    void collect_valid_slots(std::vector<int64_t> & valid_slots) const {
        valid_slots.clear();
        for (int64_t i = 0; i < total_tokens; ++i) {
            if (mask[(size_t) i]) valid_slots.push_back(i);
        }
    }

    bool compact_valid(std::vector<float> & out_k,
                       std::vector<float> & out_v,
                       std::vector<int64_t> & valid_slots) const {
        valid_slots.clear();
        for (int64_t i = 0; i < total_tokens; ++i) {
            if (mask[(size_t) i]) valid_slots.push_back(i);
        }
        const size_t n = (size_t) batch * valid_slots.size() * (size_t) heads * (size_t) head_dim;
        out_k.resize(n);
        out_v.resize(n);
        const size_t token_floats = (size_t) heads * (size_t) head_dim;
        const size_t token_bytes = token_floats * sizeof(float);
        for (int64_t b = 0; b < batch; ++b) {
            for (size_t vi = 0; vi < valid_slots.size(); ++vi) {
                const int64_t slot = valid_slots[vi];
                const size_t dst = ((size_t) b * valid_slots.size() + vi) * token_floats;
                const size_t src = kv_offset(b, slot, 0, 0);
                std::memcpy(out_k.data() + dst, k.data() + src, token_bytes);
                std::memcpy(out_v.data() + dst, v.data() + src, token_bytes);
            }
        }
        return true;
    }
};

void lingbot_dump_runtime_kv_state(uint64_t session,
                                   int64_t block,
                                   int mode,
                                   int64_t seq,
                                   const LingBotKVCache & cache,
                                   const std::vector<int64_t> & slots,
                                   const std::vector<int64_t> & valid_slots,
                                   const std::vector<float> & compact_k,
                                   const std::vector<float> & compact_v) {
    const char * dump_dir = std::getenv("VLA_LINGBOT_KV_DUMP_DIR");
    if (!dump_dir || !*dump_dir) return;
    const int64_t filter_block = lingbot_env_i64("VLA_LINGBOT_KV_DUMP_BLOCK", -1);
    if (filter_block >= 0 && filter_block != block) return;
    const int64_t filter_mode = lingbot_env_i64("VLA_LINGBOT_KV_DUMP_MODE", -1);
    if (filter_mode >= 0 && filter_mode != mode) return;
    static std::atomic<int64_t> counter{0};
    const int64_t idx = counter.fetch_add(1);
    const int64_t min_idx = lingbot_env_i64("VLA_LINGBOT_KV_DUMP_MIN", -1);
    const int64_t max_idx = lingbot_env_i64("VLA_LINGBOT_KV_DUMP_MAX", INT64_MAX);
    if (idx < min_idx || idx > max_idx) return;
    const std::string base = std::string(dump_dir) + "/cpp_kv_call_" +
        std::to_string(idx) + "_session_" + std::to_string((unsigned long long) session) +
        "_block_" + std::to_string(block) + "_mode_" + std::to_string(mode);

    dump_i64_file(base + "_slots.i64", slots);
    dump_text_file(base + "_slots.shape.txt", std::to_string((long long) slots.size()) + "\n");
    dump_i64_file(base + "_valid_slots.i64", valid_slots);
    dump_text_file(base + "_valid_slots.shape.txt", std::to_string((long long) valid_slots.size()) + "\n");
    dump_i64_file(base + "_id.i64", cache.id);
    dump_text_file(base + "_id.shape.txt", std::to_string((long long) cache.id.size()) + "\n");
    dump_u8_file(base + "_mask.u8", cache.mask);
    dump_text_file(base + "_mask.shape.txt", std::to_string((long long) cache.mask.size()) + "\n");
    dump_u8_file(base + "_is_pred.u8", cache.is_pred);
    dump_text_file(base + "_is_pred.shape.txt", std::to_string((long long) cache.is_pred.size()) + "\n");
    dump_f32_file(base + "_compact_k.f32", compact_k);
    dump_text_file(base + "_compact_k.shape.txt",
                   "1 " + std::to_string((long long) valid_slots.size()) + " " +
                   std::to_string((long long) cache.heads) + " " +
                   std::to_string((long long) cache.head_dim) + "\n");
    dump_f32_file(base + "_compact_v.f32", compact_v);
    dump_text_file(base + "_compact_v.shape.txt",
                   "1 " + std::to_string((long long) valid_slots.size()) + " " +
                   std::to_string((long long) cache.heads) + " " +
                   std::to_string((long long) cache.head_dim) + "\n");
    const double k_sum = checksum(compact_k);
    const double v_sum = checksum(compact_v);
    dump_text_file(base + "_summary.txt",
                   "session " + std::to_string((unsigned long long) session) + "\n" +
                   "block " + std::to_string((long long) block) + "\n" +
                   "mode " + std::to_string(mode) + "\n" +
                   "seq_q " + std::to_string((long long) seq) + "\n" +
                   "seq_k " + std::to_string((long long) valid_slots.size()) + "\n" +
                   "used " + std::to_string((long long) cache.used_count()) + "\n" +
                   "pred " + std::to_string((long long) cache.pred_count()) + "\n" +
                   "k_checksum " + std::to_string(k_sum) + "\n" +
                   "v_checksum " + std::to_string(v_sum) + "\n");
    std::printf("vla(lingbot_va): dumped runtime KV call=%lld session=%llu block=%lld mode=%d q=%lld k=%lld ksum=%.12g vsum=%.12g\n",
                (long long) idx,
                (unsigned long long) session,
                (long long) block,
                mode,
                (long long) seq,
                (long long) valid_slots.size(),
                k_sum,
                v_sum);
}

struct LingBotRuntimeKVKey {
    const void * model = nullptr;
    uint64_t session = 0;
    int64_t block = 0;
    int branch = 0;

    bool operator==(const LingBotRuntimeKVKey & other) const {
        return model == other.model && session == other.session &&
               block == other.block && branch == other.branch;
    }
};

struct LingBotRuntimeKVKeyHash {
    size_t operator()(const LingBotRuntimeKVKey & k) const {
        size_t h = std::hash<const void *>{}(k.model);
        h ^= std::hash<uint64_t>{}(k.session + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        h ^= std::hash<int64_t>{}(k.block + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        h ^= std::hash<int>{}(k.branch + 0x9e3779b9 + (h << 6) + (h >> 2));
        return h;
    }
};

std::mutex g_lingbot_runtime_kv_mu;
std::unordered_map<LingBotRuntimeKVKey, LingBotKVCache, LingBotRuntimeKVKeyHash> g_lingbot_runtime_kv;

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
struct LingBotDeviceKVCache {
    int64_t capacity = 0;
    int64_t heads = 0;
    int64_t head_dim = 0;
    float * k = nullptr;
    float * v = nullptr;
    int * slots = nullptr;
    int * valid_slots = nullptr;
    float * out = nullptr;
    cudaEvent_t out_ready = nullptr;
    size_t k_bytes = 0;
    size_t v_bytes = 0;
    size_t slots_bytes = 0;
    size_t valid_slots_bytes = 0;
    size_t out_bytes = 0;

    LingBotDeviceKVCache() = default;
    LingBotDeviceKVCache(const LingBotDeviceKVCache &) = delete;
    LingBotDeviceKVCache & operator=(const LingBotDeviceKVCache &) = delete;

    LingBotDeviceKVCache(LingBotDeviceKVCache && other) noexcept {
        *this = std::move(other);
    }

    LingBotDeviceKVCache & operator=(LingBotDeviceKVCache && other) noexcept {
        if (this != &other) {
            release();
            capacity = other.capacity;
            heads = other.heads;
            head_dim = other.head_dim;
            k = other.k;
            v = other.v;
            slots = other.slots;
            valid_slots = other.valid_slots;
            out = other.out;
            out_ready = other.out_ready;
            k_bytes = other.k_bytes;
            v_bytes = other.v_bytes;
            slots_bytes = other.slots_bytes;
            valid_slots_bytes = other.valid_slots_bytes;
            out_bytes = other.out_bytes;
            other.capacity = other.heads = other.head_dim = 0;
            other.k = other.v = other.out = nullptr;
            other.slots = other.valid_slots = nullptr;
            other.out_ready = nullptr;
            other.k_bytes = other.v_bytes = other.slots_bytes = other.valid_slots_bytes = other.out_bytes = 0;
        }
        return *this;
    }

    ~LingBotDeviceKVCache() {
        release();
    }

    void release() {
        if (out_ready) cudaEventDestroy(out_ready);
        if (k) cudaFree(k);
        if (v) cudaFree(v);
        if (slots) cudaFree(slots);
        if (valid_slots) cudaFree(valid_slots);
        if (out) cudaFree(out);
        k = v = out = nullptr;
        slots = valid_slots = nullptr;
        out_ready = nullptr;
        capacity = heads = head_dim = 0;
        k_bytes = v_bytes = slots_bytes = valid_slots_bytes = out_bytes = 0;
    }

    template <typename T>
    bool ensure_ptr(T *& ptr, size_t & cap, size_t bytes, const char * name) {
        if (bytes == 0) return true;
        if (ptr && cap >= bytes) return true;
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            cap = 0;
        }
        const cudaError_t e = cudaMalloc(&ptr, bytes);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): cudaMalloc(%s, %zu) failed: %s\n",
                         name, bytes, cudaGetErrorString(e));
            return false;
        }
        cap = bytes;
        return true;
    }

    bool ensure(int64_t capacity_, int64_t heads_, int64_t head_dim_, int64_t seq_q) {
        if (capacity != capacity_ || heads != heads_ || head_dim != head_dim_) {
            release();
            capacity = capacity_;
            heads = heads_;
            head_dim = head_dim_;
        }
        if (!out_ready) {
            const cudaError_t e = cudaEventCreateWithFlags(&out_ready, cudaEventDisableTiming);
            if (e != cudaSuccess) {
                std::fprintf(stderr, "vla(lingbot_va): cudaEventCreate(runtime_kv.out_ready) failed: %s\n",
                             cudaGetErrorString(e));
                return false;
            }
        }
        const size_t kv_elems = (size_t) capacity * (size_t) heads * (size_t) head_dim;
        const size_t out_elems = (size_t) seq_q * (size_t) heads * (size_t) head_dim;
        return ensure_ptr(k, k_bytes, kv_elems * sizeof(float), "runtime_kv.k") &&
               ensure_ptr(v, v_bytes, kv_elems * sizeof(float), "runtime_kv.v") &&
               ensure_ptr(slots, slots_bytes, (size_t) seq_q * sizeof(int), "runtime_kv.slots") &&
               ensure_ptr(valid_slots, valid_slots_bytes, (size_t) capacity * sizeof(int), "runtime_kv.valid_slots") &&
               ensure_ptr(out, out_bytes, out_elems * sizeof(float), "runtime_kv.out");
    }
};

struct LingBotRuntimeKVDeviceResult {
    float * data = nullptr;
    size_t bytes = 0;
    cudaEvent_t ready_event = nullptr;
    int64_t seq = 0;
    int64_t heads = 0;
    int64_t head_dim = 0;
    bool valid = false;
};

std::unordered_map<LingBotRuntimeKVKey, LingBotDeviceKVCache, LingBotRuntimeKVKeyHash> g_lingbot_runtime_device_kv;
#endif

int64_t lingbot_runtime_kv_capacity(const LingBotVAModelArch & m, int64_t seq) {
    if (const char * env = std::getenv("VLA_LINGBOT_KV_CACHE_TOKENS")) {
        char * end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end && *end == '\0' && v > 0) {
            return std::max<int64_t>((int64_t) v, seq);
        } else {
            std::fprintf(stderr,
                         "vla(lingbot_va): ignoring invalid VLA_LINGBOT_KV_CACHE_TOKENS='%s'\n",
                         env);
        }
    }

    const char * env_type_c = std::getenv("VLA_LINGBOT_ENV_TYPE");
    const bool robotwin_tshape = env_type_c && std::strcmp(env_type_c, "robotwin_tshape") == 0;
    const int64_t frame_chunk_size =
        lingbot_env_i64("VLA_LINGBOT_FRAME_CHUNK_SIZE", robotwin_tshape ? 2 : 4);
    const int64_t action_per_frame =
        lingbot_env_i64("VLA_LINGBOT_ACTION_PER_FRAME", robotwin_tshape ? 16 : 4);
    const int64_t n_suffix =
        lingbot_env_i64("VLA_LINGBOT_N_SUFFIX", frame_chunk_size * action_per_frame);
    const int64_t attn_window =
        lingbot_env_i64("VLA_LINGBOT_ATTN_WINDOW", robotwin_tshape ? 72 : 30);
    const int64_t latent_h = lingbot_env_i64("VLA_LINGBOT_LATENT_H", robotwin_tshape ? 24 : 8);
    const int64_t latent_w = lingbot_env_i64("VLA_LINGBOT_LATENT_W", robotwin_tshape ? 20 : 16);
    const int64_t latent_tokens = (m.patch_t > 0 && m.patch_h > 0 && m.patch_w > 0)
        ? (frame_chunk_size * latent_h * latent_w) / (m.patch_t * m.patch_h * m.patch_w)
        : 128;
    const int64_t action_tokens = n_suffix > 0 ? n_suffix : frame_chunk_size * action_per_frame;
    const int64_t capacity = (attn_window / 2) * latent_tokens +
                             (attn_window / 2) * action_tokens;
    return std::max<int64_t>(capacity, seq);
}

void runtime_kv_clear_pred_for_session(const void * model, uint64_t session) {
    if (!model || session == 0) return;
    std::lock_guard<std::mutex> lock(g_lingbot_runtime_kv_mu);
    int64_t touched = 0;
    int64_t pred_before = 0;
    int64_t pred_after = 0;
    for (auto & kv : g_lingbot_runtime_kv) {
        if (kv.first.model == model && kv.first.session == session) {
            pred_before += kv.second.pred_count();
            kv.second.clear_pred();
            pred_after += kv.second.pred_count();
            ++touched;
        }
    }
    std::printf("vla(lingbot_va): runtime KV clear_pred session=%llu blocks=%lld pred_before=%lld pred_after=%lld\n",
                (unsigned long long) session,
                (long long) touched,
                (long long) pred_before,
                (long long) pred_after);
}



struct LingBotFlexMaskMeta {
    std::vector<int64_t> seq_ids;
    std::vector<int64_t> frame_ids;
    std::vector<int64_t> noise_ids;
    std::vector<int64_t> text_seq_ids;
    int64_t window_size = 0;
    int64_t block_size = 64;
};

LingBotFlexMaskMeta build_flex_mask_meta(
        const LingBotTensor5DShape & latent,
        const LingBotTensor5DShape & action,
        int64_t padded_length,
        int64_t chunk_size,
        int64_t window_size,
        int64_t pt,
        int64_t ph,
        int64_t pw,
        int64_t text_seq,
        int64_t block_size) {
    LingBotFlexMaskMeta meta;
    meta.window_size = window_size;
    meta.block_size = block_size;
    std::vector<int64_t> latent_seq_id;
    std::vector<int64_t> latent_frame_id;
    std::vector<int64_t> action_seq_id;
    std::vector<int64_t> action_frame_id;
    for (int64_t b = 0; b < latent.b; ++b) {
        for (int64_t f = 0; f < latent.f / pt; ++f) {
            for (int64_t h = 0; h < latent.h / ph; ++h) {
                for (int64_t w = 0; w < latent.w / pw; ++w) {
                    latent_seq_id.push_back(b);
                    latent_frame_id.push_back((f * pt) / chunk_size * 2);
                }
            }
        }
        for (int64_t f = 0; f < action.f; ++f) {
            for (int64_t h = 0; h < action.h; ++h) {
                for (int64_t w = 0; w < action.w; ++w) {
                    action_seq_id.push_back(b);
                    action_frame_id.push_back(f / chunk_size * 2 + 1);
                }
            }
        }
    }
    auto append = [](std::vector<int64_t> & dst, const std::vector<int64_t> & src) {
        dst.insert(dst.end(), src.begin(), src.end());
    };
    auto fill = [](std::vector<int64_t> & dst, int64_t v, int64_t n) {
        for (int64_t i = 0; i < n; ++i) dst.push_back(v);
    };
    append(meta.seq_ids, latent_seq_id); append(meta.frame_ids, latent_frame_id); fill(meta.noise_ids, 0, (int64_t) latent_seq_id.size());
    append(meta.seq_ids, latent_seq_id); append(meta.frame_ids, latent_frame_id); fill(meta.noise_ids, 1, (int64_t) latent_seq_id.size());
    append(meta.seq_ids, action_seq_id); append(meta.frame_ids, action_frame_id); fill(meta.noise_ids, 0, (int64_t) action_seq_id.size());
    append(meta.seq_ids, action_seq_id); append(meta.frame_ids, action_frame_id); fill(meta.noise_ids, 1, (int64_t) action_seq_id.size());
    fill(meta.seq_ids, -1, padded_length);
    fill(meta.frame_ids, -1, padded_length);
    fill(meta.noise_ids, -1, padded_length);
    for (int64_t b = 0; b < latent.b; ++b) fill(meta.text_seq_ids, b, text_seq);
    return meta;
}

bool flex_self_allowed(const LingBotFlexMaskMeta & meta, int64_t q, int64_t k) {
    if (meta.seq_ids[(size_t) q] < 0 || meta.seq_ids[(size_t) q] != meta.seq_ids[(size_t) k]) return false;
    const int64_t qf = meta.frame_ids[(size_t) q], kf = meta.frame_ids[(size_t) k];
    const int64_t qn = meta.noise_ids[(size_t) q], kn = meta.noise_ids[(size_t) k];
    const bool clean2clean = qn == 1 && kn == 1 && kf <= qf;
    const bool noise2clean = qn == 0 && kn == 1 && kf < qf;
    const bool noise2noise = qn == 0 && kn == 0 && kf == qf;
    return (clean2clean || noise2clean || noise2noise) && std::abs(qf - kf) <= meta.window_size;
}

bool flex_cross_allowed(const LingBotFlexMaskMeta & meta, int64_t q, int64_t k) {
    return meta.seq_ids[(size_t) q] >= 0 && meta.seq_ids[(size_t) q] == meta.text_seq_ids[(size_t) k];
}

struct LingBotBlockSparseTable {
    int64_t q_blocks = 0;
    int64_t kv_blocks = 0;
    int64_t block_size = 0;
    std::vector<int64_t> row_ptr;
    std::vector<int64_t> col_idx;
};

LingBotBlockSparseTable build_dense_block_table(int64_t seq_q, int64_t seq_k, int64_t block_size) {
    LingBotBlockSparseTable table;
    table.block_size = block_size;
    table.q_blocks = (seq_q + block_size - 1) / block_size;
    table.kv_blocks = (seq_k + block_size - 1) / block_size;
    table.row_ptr.reserve((size_t) table.q_blocks + 1);
    table.row_ptr.push_back(0);
    for (int64_t qb = 0; qb < table.q_blocks; ++qb) {
        for (int64_t kb = 0; kb < table.kv_blocks; ++kb) {
            table.col_idx.push_back(kb);
        }
        table.row_ptr.push_back((int64_t) table.col_idx.size());
    }
    return table;
}

LingBotBlockSparseTable build_self_block_table(const LingBotFlexMaskMeta & meta) {
    LingBotBlockSparseTable table;
    const int64_t n = (int64_t) meta.seq_ids.size();
    table.block_size = meta.block_size;
    table.q_blocks = (n + table.block_size - 1) / table.block_size;
    table.kv_blocks = table.q_blocks;
    table.row_ptr.reserve((size_t) table.q_blocks + 1);
    table.row_ptr.push_back(0);
    for (int64_t qb = 0; qb < table.q_blocks; ++qb) {
        const int64_t q0 = qb * table.block_size;
        const int64_t q1 = std::min(q0 + table.block_size, n);
        for (int64_t kb = 0; kb < table.kv_blocks; ++kb) {
            const int64_t k0 = kb * table.block_size;
            const int64_t k1 = std::min(k0 + table.block_size, n);
            bool any = false;
            for (int64_t q = q0; q < q1 && !any; ++q) {
                for (int64_t k = k0; k < k1; ++k) {
                    if (flex_self_allowed(meta, q, k)) {
                        any = true;
                        break;
                    }
                }
            }
            if (any) table.col_idx.push_back(kb);
        }
        table.row_ptr.push_back((int64_t) table.col_idx.size());
    }
    return table;
}

LingBotBlockSparseTable build_cross_block_table(const LingBotFlexMaskMeta & meta) {
    LingBotBlockSparseTable table;
    const int64_t qn = (int64_t) meta.seq_ids.size();
    const int64_t kn = (int64_t) meta.text_seq_ids.size();
    table.block_size = meta.block_size;
    table.q_blocks = (qn + table.block_size - 1) / table.block_size;
    table.kv_blocks = (kn + table.block_size - 1) / table.block_size;
    table.row_ptr.reserve((size_t) table.q_blocks + 1);
    table.row_ptr.push_back(0);
    for (int64_t qb = 0; qb < table.q_blocks; ++qb) {
        const int64_t q0 = qb * table.block_size;
        const int64_t q1 = std::min(q0 + table.block_size, qn);
        for (int64_t kb = 0; kb < table.kv_blocks; ++kb) {
            const int64_t k0 = kb * table.block_size;
            const int64_t k1 = std::min(k0 + table.block_size, kn);
            bool any = false;
            for (int64_t q = q0; q < q1 && !any; ++q) {
                for (int64_t k = k0; k < k1; ++k) {
                    if (flex_cross_allowed(meta, q, k)) {
                        any = true;
                        break;
                    }
                }
            }
            if (any) table.col_idx.push_back(kb);
        }
        table.row_ptr.push_back((int64_t) table.col_idx.size());
    }
    return table;
}

LingBotFlexMaskMeta build_runtime_flex_meta(const LingBotVAModelArch & m) {
    const LingBotTensor5DShape latent{1, m.in_channels, 2 * m.patch_t, 2 * m.patch_h, 2 * m.patch_w};
    const LingBotTensor5DShape action{1, m.action_dim, 2, 2, 1};
    return build_flex_mask_meta(latent, action, 3, 1, 4,
                                m.patch_t, m.patch_h, m.patch_w, 512, 4);
}

int64_t max_row_nnz(const LingBotBlockSparseTable & table) {
    int64_t out = 0;
    for (int64_t r = 0; r < table.q_blocks; ++r) {
        out = std::max(out, table.row_ptr[(size_t) r + 1] - table.row_ptr[(size_t) r]);
    }
    return out;
}

std::vector<int> i64_to_i32(const std::vector<int64_t> & in) {
    std::vector<int> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) out[i] = (int) in[i];
    return out;
}

std::vector<unsigned char> build_self_token_mask(const LingBotFlexMaskMeta & meta) {
    const int64_t n = (int64_t) meta.seq_ids.size();
    std::vector<unsigned char> mask((size_t) n * (size_t) n, 0);
    for (int64_t q = 0; q < n; ++q) {
        for (int64_t k = 0; k < n; ++k) {
            mask[(size_t) q * (size_t) n + (size_t) k] = flex_self_allowed(meta, q, k) ? 1 : 0;
        }
    }
    return mask;
}

void dense_masked_attention_ref(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<unsigned char> & mask,
        int64_t seq,
        int64_t heads,
        int64_t head_dim,
        std::vector<float> & out) {
    const float scale = 1.0f / std::sqrt((float) head_dim);
    out.assign((size_t) seq * (size_t) heads * (size_t) head_dim, 0.0f);
    for (int64_t s = 0; s < seq; ++s) {
        for (int64_t h = 0; h < heads; ++h) {
            float mx = -INFINITY;
            for (int64_t t = 0; t < seq; ++t) {
                if (!mask[(size_t) s * (size_t) seq + (size_t) t]) continue;
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += q[((size_t) s * (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d] *
                           k[((size_t) t * (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d];
                }
                mx = std::max(mx, dot * scale);
            }
            float sum = 0.0f;
            for (int64_t t = 0; t < seq; ++t) {
                if (!mask[(size_t) s * (size_t) seq + (size_t) t]) continue;
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += q[((size_t) s * (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d] *
                           k[((size_t) t * (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d];
                }
                const float w = std::exp(dot * scale - mx);
                sum += w;
                for (int64_t d = 0; d < head_dim; ++d) {
                    out[((size_t) s * (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d] +=
                        w * v[((size_t) t * (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d];
                }
            }
            for (int64_t d = 0; d < head_dim; ++d) {
                float & y = out[((size_t) s * (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d];
                y = sum > 0.0f ? y / sum : 0.0f;
            }
        }
    }
}

bool tensor_to_f32_vector(ggml_tensor * t, std::vector<float> & out) {
    if (!t || t->type != GGML_TYPE_F32) return false;
    out.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
    return true;
}

bool hds_tensor_to_shd(
        ggml_tensor * t,
        int64_t seq,
        int64_t heads,
        int64_t head_dim,
        std::vector<float> & out) {
    const size_t expected = (size_t) seq * (size_t) heads * (size_t) head_dim;
    if (!tensor_to_f32_vector(t, out)) return false;
    return out.size() == expected;
}

std::vector<float> shd_to_hidden_seq(
        const std::vector<float> & shd,
        int64_t seq,
        int64_t heads,
        int64_t head_dim) {
    const size_t expected = (size_t) seq * (size_t) heads * (size_t) head_dim;
    return shd.size() == expected ? shd : std::vector<float>{};
}

std::vector<float> hidden_seq_to_shd(
        const std::vector<float> & hidden_seq,
        int64_t seq,
        int64_t heads,
        int64_t head_dim) {
    const size_t expected = (size_t) seq * (size_t) heads * (size_t) head_dim;
    return hidden_seq.size() == expected ? hidden_seq : std::vector<float>{};
}

std::vector<float> shd_to_hds_raw(
        const std::vector<float> & shd,
        int64_t seq,
        int64_t heads,
        int64_t head_dim) {
    std::vector<float> out((size_t) seq * (size_t) heads * (size_t) head_dim, 0.0f);
    for (int64_t s = 0; s < seq; ++s) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const size_t src = ((size_t) s * (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d;
                const size_t dst = (size_t) d + (size_t) head_dim * ((size_t) h + (size_t) heads * (size_t) s);
                out[dst] = shd[src];
            }
        }
    }
    return out;
}

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
bool run_lingbot_runtime_kv_device_attention(
        ggml_backend_t backend,
        LingBotDeviceKVCache & device_cache,
        ggml_tensor * qh,
        ggml_tensor * kh,
        ggml_tensor * vh,
        const std::vector<int64_t> & slots_i64,
        const std::vector<int64_t> & valid_slots_i64,
        int64_t seq_q,
        int64_t seq_k,
        int64_t capacity,
        int64_t heads,
        int64_t head_dim,
        std::vector<float> & out_shd,
        LingBotRuntimeKVDeviceResult * device_out = nullptr,
        const float * q_device = nullptr,
        const float * k_device = nullptr,
        const float * v_device = nullptr) {
    if (device_out) {
        *device_out = LingBotRuntimeKVDeviceResult{};
    }
    if (!qh || !kh || !vh || !qh->data || !kh->data || !vh->data ||
        qh->type != GGML_TYPE_F32 || kh->type != GGML_TYPE_F32 || vh->type != GGML_TYPE_F32) {
        return false;
    }
    if ((int64_t) slots_i64.size() != seq_q || (int64_t) valid_slots_i64.size() != seq_k) {
        return false;
    }
    if (!device_cache.ensure(capacity, heads, head_dim, seq_q)) {
        return false;
    }

    std::vector<int> slots = i64_to_i32(slots_i64);
    std::vector<int> valid_slots = i64_to_i32(valid_slots_i64);
    auto ok = [](cudaError_t e, const char * what) {
        if (e != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): %s failed: %s\n", what, cudaGetErrorString(e));
        }
        return e == cudaSuccess;
    };
    if (!ok(cudaMemcpy(device_cache.slots, slots.data(), slots.size() * sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy(runtime_kv.slots)") ||
        !ok(cudaMemcpy(device_cache.valid_slots, valid_slots.data(), valid_slots.size() * sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy(runtime_kv.valid_slots)")) {
        return false;
    }
    int rc = lingbot_runtime_kv_update_f32(
        (const float *) kh->data, (const float *) vh->data,
        device_cache.k, device_cache.v, device_cache.slots,
        (int) seq_q, (int) heads, (int) head_dim, (int) capacity, 0);
    if (rc != 0) {
        std::fprintf(stderr, "vla(lingbot_va): runtime device KV update failed\n");
        return false;
    }
    rc = lingbot_runtime_kv_attn_cublas_f32(
        (const float *) qh->data, device_cache.k, device_cache.v,
        device_cache.valid_slots, device_cache.out,
        (int) seq_q, (int) seq_k, (int) heads, (int) head_dim,
        (int) capacity, 1.0f / std::sqrt((float) head_dim), 0);
    if (rc != 0) {
        std::fprintf(stderr, "vla(lingbot_va): runtime device KV cuBLAS attention failed\n");
        return false;
    }
    if (device_cache.out_ready) {
        const cudaError_t event_err = cudaEventRecord(device_cache.out_ready, 0);
        if (event_err != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): cudaEventRecord(runtime_kv.out_ready) failed: %s\n",
                         cudaGetErrorString(event_err));
            return false;
        }
    }
    if (device_out) {
        device_out->data = device_cache.out;
        device_out->bytes = (size_t) seq_q * (size_t) heads * (size_t) head_dim * sizeof(float);
        device_out->ready_event = device_cache.out_ready;
        device_out->seq = seq_q;
        device_out->heads = heads;
        device_out->head_dim = head_dim;
        device_out->valid = true;
        return true;
    }
    out_shd.assign((size_t) seq_q * (size_t) heads * (size_t) head_dim, 0.0f);
    if (!ok(cudaMemcpy(out_shd.data(), device_cache.out, out_shd.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy(runtime_kv.out)") ||
        !ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize(runtime_kv)")) {
        return false;
    }
    return true;
}


bool real_qkv_to_cuda_context(
        ggml_backend_t backend,
        ggml_tensor * qh,
        ggml_tensor * kh,
        ggml_tensor * vh,
        const LingBotBlockSparseTable & self_table,
        const std::vector<unsigned char> & token_mask,
        const LingBotVAModelArch & m,
        int64_t seq,
        std::vector<float> & context_shd,
        LingBotKVCache * cache_out = nullptr,
        uint64_t cache_session_id = 0,
        int64_t cache_block_index = -1,
        int cache_mode = 0,
        int cache_branch = 0,
        std::vector<float> * attn_k_out = nullptr,
        std::vector<float> * attn_v_out = nullptr,
        int64_t * attn_seq_k_out = nullptr,
        LingBotRuntimeKVDeviceResult * device_out = nullptr) {
    if (device_out) {
        *device_out = LingBotRuntimeKVDeviceResult{};
    }
    const bool timing = lingbot_timing_enabled();
    const auto t_total0 = LingBotClock::now();
    auto t_last = t_total0;
    auto log_part = [&](const char * name) {
        if (!timing) return;
        const auto t_now = LingBotClock::now();
        std::printf("vla(lingbot_va): timing runtime-kv block=%lld mode=%d branch=%d seq=%lld part=%s %.3fms\n",
                    (long long) cache_block_index + 1,
                    cache_mode,
                    cache_branch,
                    (long long) seq,
                    name,
                    lingbot_elapsed_ms(t_last, t_now));
        t_last = t_now;
    };

    const bool can_use_device_kv =
        cache_session_id != 0 &&
        cache_block_index >= 0 &&
        backend != nullptr &&
        ggml_backend_is_cuda(backend) &&
        qh && kh && vh && qh->data && kh->data && vh->data &&
        qh->type == GGML_TYPE_F32 && kh->type == GGML_TYPE_F32 && vh->type == GGML_TYPE_F32 &&
        attn_k_out == nullptr && attn_v_out == nullptr && attn_seq_k_out == nullptr &&
        lingbot_runtime_kv_device_enabled();
    if (can_use_device_kv) {
        std::lock_guard<std::mutex> lock(g_lingbot_runtime_kv_mu);
        const LingBotRuntimeKVKey key{&m, cache_session_id, cache_block_index, cache_branch};
        LingBotKVCache & cache = g_lingbot_runtime_kv[key];
        const int64_t min_capacity = lingbot_runtime_kv_capacity(m, seq);
        if (cache.batch != 1 || cache.heads != m.n_heads || cache.head_dim != m.head_dim ||
            cache.total_tokens < min_capacity) {
            cache.init(1, min_capacity, m.n_heads, m.head_dim);
        }
        LingBotDeviceKVCache & device_cache = g_lingbot_runtime_device_kv[key];
        std::vector<int64_t> slots;
        std::vector<int64_t> valid_slots;
        const bool pred = cache_mode == 1;
        const bool mode0_no_temp_cache_insert =
            cache_mode == 0 &&
            (lingbot_official_fast_path_enabled() ||
             lingbot_env_enabled("VLA_LINGBOT_RUNTIME_KV_MODE0_NO_TEMP_CACHE_INSERT")) &&
            !lingbot_env_enabled("VLA_LINGBOT_RUNTIME_KV_MODE0_NO_TEMP_CACHE_INSERT_DISABLE");
        const bool meta_ok = mode0_no_temp_cache_insert
            ? cache.prepare_mode0_temp_slots_no_insert(seq, slots, valid_slots)
            : cache.update_meta_only(seq, pred, slots);
        if (!meta_ok || (int64_t) slots.size() != seq) {
            std::fprintf(stderr, "vla(lingbot_va): runtime device KV metadata update failed session=%llu block=%lld mode=%d branch=%d\n",
                         (unsigned long long) cache_session_id,
                         (long long) cache_block_index, cache_mode, cache_branch);
            return false;
        }
        log_part("device_cache_meta_update");
        if (!mode0_no_temp_cache_insert) {
            cache.collect_valid_slots(valid_slots);
        }
        const int64_t seq_k = (int64_t) valid_slots.size();
        log_part("device_valid_slots");
        const bool ok = run_lingbot_runtime_kv_device_attention(
            backend, device_cache, qh, kh, vh, slots, valid_slots,
            seq, seq_k, cache.total_tokens, m.n_heads, m.head_dim, context_shd, device_out);
        log_part(device_out ? "device_update_attention" : "device_update_attention_d2h");
        if (cache_mode == 0 && !mode0_no_temp_cache_insert) {
            cache.restore(slots);
        }
        log_part("device_cache_restore");
        if (cache_out) *cache_out = cache;
        if (ok && lingbot_env_enabled("VLA_LINGBOT_KV_VERBOSE")) {
            std::printf("vla(lingbot_va): runtime device KV self-attn session=%llu block=%lld mode=%d branch=%d q=%lld k=%lld used=%lld pred=%lld mode0_no_temp_insert=%d\n",
                        (unsigned long long) cache_session_id,
                        (long long) cache_block_index, cache_mode, cache_branch,
                        (long long) seq, (long long) seq_k,
                        (long long) cache.used_count(), (long long) cache.pred_count(),
                        mode0_no_temp_cache_insert ? 1 : 0);
        }
        if (timing) {
            const auto t_total1 = LingBotClock::now();
            std::printf("vla(lingbot_va): timing runtime-kv-device block=%lld mode=%d branch=%d seq=%lld k=%lld total=%.3fms\n",
                        (long long) cache_block_index + 1,
                        cache_mode,
                        cache_branch,
                        (long long) seq,
                        (long long) seq_k,
                        lingbot_elapsed_ms(t_total0, t_total1));
        }
        return ok;
    }

    std::fprintf(stderr, "vla(lingbot_va): official runtime-KV requires CUDA device-KV warp path; fallback paths were removed\n");
    return false;
}



#endif





struct LingBotFlowScheduler {
    int num_train_timesteps = 1000;
    double shift = 3.0;
    double sigma_max = 1.0;
    double sigma_min = 0.003 / 1.002;
    bool inverse_timesteps = false;
    bool extra_one_step = false;
    bool reverse_sigmas = false;
    std::vector<double> sigmas;
    std::vector<double> timesteps;

    void set_timesteps(int num_inference_steps, double denoising_strength = 1.0) {
        sigmas.clear();
        timesteps.clear();
        const double sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength;
        const int n = extra_one_step ? num_inference_steps + 1 : num_inference_steps;
        if (n <= 1) {
            sigmas.push_back(sigma_start);
        } else {
            for (int i = 0; i < n; ++i) {
                const double a = (double) i / (double) (n - 1);
                sigmas.push_back(sigma_start + (sigma_min - sigma_start) * a);
            }
            if (extra_one_step && !sigmas.empty()) {
                sigmas.pop_back();
            }
        }
        if (inverse_timesteps) {
            std::reverse(sigmas.begin(), sigmas.end());
        }
        for (double & s : sigmas) {
            s = shift * s / (1.0 + (shift - 1.0) * s);
            if (reverse_sigmas) {
                s = 1.0 - s;
            }
        }
        timesteps.reserve(sigmas.size());
        for (double s : sigmas) {
            timesteps.push_back(s * (double) num_train_timesteps);
        }
    }

    int timestep_index(double timestep) const {
        int best = 0;
        double best_dist = std::numeric_limits<double>::infinity();
        for (int i = 0; i < (int) timesteps.size(); ++i) {
            const double d = std::abs(timesteps[(size_t) i] - timestep);
            if (d < best_dist) {
                best_dist = d;
                best = i;
            }
        }
        return best;
    }

    double next_sigma(double timestep, bool to_final = false) const {
        const int i = timestep_index(timestep);
        if (to_final || i + 1 >= (int) sigmas.size()) {
            return (inverse_timesteps || reverse_sigmas) ? 1.0 : 0.0;
        }
        return sigmas[(size_t) i + 1];
    }

    float step_delta(double timestep, bool to_final = false) const {
        const int i = timestep_index(timestep);
        const double sigma = sigmas[(size_t) i];
        const double sigma_next = next_sigma(timestep, to_final);
        return (float) (sigma_next - sigma);
    }

    void step_inplace(std::vector<float> & sample, const std::vector<float> & model_output,
                      double timestep, bool to_final = false) const {
        const float delta = step_delta(timestep, to_final);
        for (size_t j = 0; j < sample.size(); ++j) {
            sample[j] += model_output[j] * delta;
        }
    }

    void step_inplace_round(
            std::vector<float> & sample,
            const std::vector<float> & model_output,
            double timestep,
            bool bf16_state,
            bool to_final = false) const {
        const float delta = step_delta(timestep, to_final);
        if (bf16_state) {
            for (size_t j = 0; j < sample.size(); ++j) {
                const float v = sample[j] + model_output[j] * delta;
                sample[j] = lingbot_bf16_bits_to_f32(lingbot_f32_to_bf16_bits_rne(v));
            }
        } else {
            for (size_t j = 0; j < sample.size(); ++j) {
                sample[j] += model_output[j] * delta;
            }
        }
    }
};

struct LingBotGridSpec {
    int64_t f = 1;
    int64_t h = 1;
    int64_t w = 1;
    int64_t t = 0;
    int64_t f_w = 1;
    int64_t f_shift = 0;
    bool action = false;

    int64_t seq() const {
        return f * h * w;
    }
};

LingBotGridSpec lingbot_default_grid_spec(int64_t seq, bool action_mode) {
    LingBotGridSpec spec;
    if (action_mode) {
        // Minimal shape that preserves get_mesh_id(..., action=True) behavior:
        // one frame, seq action rows, one width token.
        spec.f = 1;
        spec.h = seq;
        spec.w = 1;
        spec.t = 1;
        spec.action = true;
    } else {
        // Minimal latent shape with seq temporal tokens after patching.
        spec.f = seq;
        spec.h = 1;
        spec.w = 1;
        spec.t = 0;
        spec.action = false;
    }
    return spec;
}

void build_lingbot_rope(
        int64_t head_dim,
        const LingBotGridSpec & spec,
        std::vector<float> & cos,
        std::vector<float> & sin) {
    const int64_t h_dim = head_dim / 3;
    const int64_t w_dim = head_dim / 3;
    const int64_t f_dim = head_dim - 2 * h_dim;
    const int64_t f_pairs = f_dim / 2;
    const int64_t h_pairs = h_dim / 2;
    const int64_t w_pairs = w_dim / 2;
    const int64_t pairs = head_dim / 2;
    const int64_t seq = spec.seq();
    const double theta = 10000.0;

    cos.assign((size_t) pairs * (size_t) seq, 1.0f);
    sin.assign((size_t) pairs * (size_t) seq, 0.0f);

    int64_t s = 0;
    for (int64_t fi = 0; fi < spec.f; ++fi) {
        for (int64_t hi = 0; hi < spec.h; ++hi) {
            for (int64_t wi = 0; wi < spec.w; ++wi, ++s) {
                double f_id = (double) (spec.f_shift + fi) * (double) spec.f_w;
                double h_id = (double) hi;
                double w_id = (double) wi;
                if (spec.action) {
                    // Matches LingBot get_mesh_id(..., action=True): add a
                    // per-action-row fractional offset to f and mark h/w as
                    // non-visual sentinel coordinates.
                    f_id += (double) (hi + 1) / (double) (spec.h + 1);
                    h_id = -1.0;
                    w_id = -1.0;
                }

                int64_t pair_index = 0;
                auto write_axis = [&](int64_t axis_pairs, int64_t axis_dim, double grid_id) {
                    for (int64_t p = 0; p < axis_pairs; ++p, ++pair_index) {
                        const double base = 1.0 / std::pow(theta, (double) (2 * p) / (double) axis_dim);
                        const double freq = grid_id * base;
                        const size_t off = (size_t) pair_index + (size_t) pairs * (size_t) s;
                        cos[off] = (float) std::cos(freq);
                        sin[off] = (float) std::sin(freq);
                    }
                };
                write_axis(f_pairs, f_dim, f_id);
                write_axis(h_pairs, h_dim, h_id);
                write_axis(w_pairs, w_dim, w_id);
            }
        }
    }
}

using LingBotGridId = std::array<double, 3>;

std::vector<LingBotGridId> build_grid_ids_from_spec(const LingBotGridSpec & spec) {
    std::vector<LingBotGridId> out;
    out.reserve((size_t) spec.seq());
    for (int64_t fi = 0; fi < spec.f; ++fi) {
        for (int64_t hi = 0; hi < spec.h; ++hi) {
            for (int64_t wi = 0; wi < spec.w; ++wi) {
                double f_id = (double) (spec.f_shift + fi) * (double) spec.f_w;
                double h_id = (double) hi;
                double w_id = (double) wi;
                if (spec.action) {
                    f_id += (double) (hi + 1) / (double) (spec.h + 1);
                    h_id = -1.0;
                    w_id = -1.0;
                }
                out.push_back({f_id, h_id, w_id});
            }
        }
    }
    return out;
}

void build_lingbot_rope_from_grid_ids(
        int64_t head_dim,
        const std::vector<LingBotGridId> & grid_ids,
        std::vector<float> & cos,
        std::vector<float> & sin) {
    const int64_t h_dim = head_dim / 3;
    const int64_t w_dim = head_dim / 3;
    const int64_t f_dim = head_dim - 2 * h_dim;
    const int64_t f_pairs = f_dim / 2;
    const int64_t h_pairs = h_dim / 2;
    const int64_t w_pairs = w_dim / 2;
    const int64_t pairs = head_dim / 2;
    const double theta = 10000.0;

    cos.assign((size_t) pairs * grid_ids.size(), 1.0f);
    sin.assign((size_t) pairs * grid_ids.size(), 0.0f);
    for (size_t s = 0; s < grid_ids.size(); ++s) {
        int64_t pair_index = 0;
        auto write_axis = [&](int64_t axis_pairs, int64_t axis_dim, double grid_id) {
            for (int64_t p = 0; p < axis_pairs; ++p, ++pair_index) {
                const double base = 1.0 / std::pow(theta, (double) (2 * p) / (double) axis_dim);
                const double freq = grid_id * base;
                const size_t off = (size_t) pair_index + (size_t) pairs * s;
                cos[off] = (float) std::cos(freq);
                sin[off] = (float) std::sin(freq);
            }
        };
        write_axis(f_pairs, f_dim, grid_ids[s][0]);
        write_axis(h_pairs, h_dim, grid_ids[s][1]);
        write_axis(w_pairs, w_dim, grid_ids[s][2]);
    }
}

bool build_runtime_mixed_rope(
        const LingBotVAModelArch & m,
        const LingBotFlexMaskMeta & meta,
        std::vector<float> & cos,
        std::vector<float> & sin) {
    const int64_t latent_tokens = 8;
    const int64_t action_tokens = 4;
    const int64_t padded_tokens = 3;
    const int64_t expected = 2 * latent_tokens + 2 * action_tokens + padded_tokens;
    if ((int64_t) meta.seq_ids.size() != expected) {
        std::fprintf(stderr, "vla(lingbot_va): mixed rope meta mismatch, got tokens=%zu expected=%lld\n",
                     meta.seq_ids.size(), (long long) expected);
        return false;
    }

    LingBotGridSpec latent;
    latent.f = 2;
    latent.h = 2;
    latent.w = 2;
    latent.t = 0;
    latent.action = false;
    LingBotGridSpec action;
    action.f = 2;
    action.h = 2;
    action.w = 1;
    action.t = 1;
    action.action = true;

    const std::vector<LingBotGridId> latent_ids = build_grid_ids_from_spec(latent);
    const std::vector<LingBotGridId> action_ids = build_grid_ids_from_spec(action);
    if ((int64_t) latent_ids.size() != latent_tokens || (int64_t) action_ids.size() != action_tokens) {
        return false;
    }

    std::vector<LingBotGridId> full;
    full.reserve((size_t) expected);
    full.insert(full.end(), latent_ids.begin(), latent_ids.end());
    full.insert(full.end(), latent_ids.begin(), latent_ids.end());
    full.insert(full.end(), action_ids.begin(), action_ids.end());
    full.insert(full.end(), action_ids.begin(), action_ids.end());
    for (int64_t i = 0; i < padded_tokens; ++i) {
        full.push_back({0.0, 0.0, 0.0});
    }
    build_lingbot_rope_from_grid_ids(m.head_dim, full, cos, sin);
    return true;
}



#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS

#endif

void dump_f32_file(const std::string & path, const std::vector<float> & data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "vla(lingbot_va): failed to open dump file %s\n", path.c_str());
        return;
    }
    out.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * sizeof(float)));
}

void dump_text_file(const std::string & path, const std::string & text) {
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "vla(lingbot_va): failed to open dump file %s\n", path.c_str());
        return;
    }
    out << text;
}

void dump_i64_file(const std::string & path, const std::vector<int64_t> & data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "vla(lingbot_va): failed to open dump file %s\n", path.c_str());
        return;
    }
    out.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * sizeof(int64_t)));
}

void dump_u8_file(const std::string & path, const std::vector<uint8_t> & data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "vla(lingbot_va): failed to open dump file %s\n", path.c_str());
        return;
    }
    out.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) data.size());
}

bool read_text_embedding_f32_file(const char * path,
                                  int64_t expected_dim,
                                  std::vector<float> & out,
                                  int64_t & out_seq) {
    out.clear();
    out_seq = 0;
    if (!path || !*path || expected_dim <= 0) return false;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::fprintf(stderr, "vla(lingbot_va): failed to open text embedding override %s\n", path);
        return false;
    }
    const std::streamoff end = in.tellg();
    if (end <= 0) {
        std::fprintf(stderr, "vla(lingbot_va): empty text embedding override %s\n", path);
        return false;
    }
    const int64_t bytes = (int64_t) end;
    const int64_t row_bytes = expected_dim * (int64_t) sizeof(float);
    if (row_bytes <= 0 || bytes % row_bytes != 0) {
        std::fprintf(stderr,
                     "vla(lingbot_va): text embedding override size mismatch for %s "
                     "(bytes=%lld dim=%lld)\n",
                     path, (long long) bytes, (long long) expected_dim);
        return false;
    }
    out_seq = bytes / row_bytes;
    out.resize((size_t) (out_seq * expected_dim));
    in.seekg(0, std::ios::beg);
    if (!in.read(reinterpret_cast<char *>(out.data()), end)) {
        std::fprintf(stderr, "vla(lingbot_va): failed to read text embedding override %s\n", path);
        out.clear();
        out_seq = 0;
        return false;
    }
    return true;
}

bool pad_text_embedding_to_sequence(std::vector<float> & text,
                                    int64_t text_dim,
                                    int64_t & text_seq,
                                    int64_t target_seq,
                                    const char * label) {
    if (text_dim <= 0 || text_seq <= 0 || target_seq <= 0 ||
        text.size() != (size_t) text_dim * (size_t) text_seq) {
        std::fprintf(stderr,
                     "vla(lingbot_va): invalid text embedding before pad (%s) "
                     "size=%zu dim=%lld seq=%lld target=%lld\n",
                     label ? label : "unknown",
                     text.size(),
                     (long long) text_dim,
                     (long long) text_seq,
                     (long long) target_seq);
        return false;
    }
    if (text_seq > target_seq) {
        std::fprintf(stderr,
                     "vla(lingbot_va): text embedding seq exceeds max_sequence_length (%s) "
                     "seq=%lld max=%lld\n",
                     label ? label : "unknown",
                     (long long) text_seq,
                     (long long) target_seq);
        return false;
    }
    if (text_seq == target_seq) return true;

    std::vector<float> padded((size_t) target_seq * (size_t) text_dim, 0.0f);
    std::memcpy(padded.data(), text.data(), text.size() * sizeof(float));
    text.swap(padded);
    std::printf("vla(lingbot_va): padded UMT5 text embedding (%s) seq=%lld -> %lld\n",
                label ? label : "unknown",
                (long long) text_seq,
                (long long) target_seq);
    text_seq = target_seq;
    return true;
}

int64_t umt5_relative_position_bucket(
        int64_t relative_position,
        bool bidirectional,
        int64_t num_buckets,
        int64_t max_distance) {
    int64_t relative_buckets = 0;
    if (bidirectional) {
        num_buckets /= 2;
        if (relative_position > 0) relative_buckets += num_buckets;
        relative_position = std::llabs(relative_position);
    } else {
        relative_position = -std::min<int64_t>(relative_position, 0);
    }
    const int64_t max_exact = num_buckets / 2;
    const bool is_small = relative_position < max_exact;
    int64_t relative_position_if_large = max_exact;
    if (relative_position > 0 && max_exact > 0) {
        const double log_ratio = std::log((double) relative_position / (double) max_exact) /
                                 std::log((double) max_distance / (double) max_exact);
        relative_position_if_large = max_exact + (int64_t) (log_ratio * (double) (num_buckets - max_exact));
        relative_position_if_large = std::min<int64_t>(relative_position_if_large, num_buckets - 1);
    }
    relative_buckets += is_small ? relative_position : relative_position_if_large;
    return relative_buckets;
}

void umt5_rms_norm_host(
        const std::vector<float> & x,
        const std::vector<float> & weight,
        int seq,
        int dim,
        float eps,
        std::vector<float> & out) {
    out.assign((size_t) seq * dim, 0.0f);
    for (int s = 0; s < seq; ++s) {
        double ss = 0.0;
        const float * row = x.data() + (size_t) s * dim;
        for (int i = 0; i < dim; ++i) ss += (double) row[i] * row[i];
        const double inv = 1.0 / std::sqrt(ss / (double) dim + (double) eps);
        float * dst = out.data() + (size_t) s * dim;
        for (int i = 0; i < dim; ++i) dst[i] = (float) ((double) row[i] * inv) * weight[(size_t) i];
    }
}

void umt5_linear_host(
        const std::vector<float> & x,
        const std::vector<float> & w,
        int seq,
        int in_dim,
        int out_dim,
        std::vector<float> & out) {
    out.assign((size_t) seq * out_dim, 0.0f);
    for (int s = 0; s < seq; ++s) {
        const float * row = x.data() + (size_t) s * in_dim;
        float * dst = out.data() + (size_t) s * out_dim;
        for (int o = 0; o < out_dim; ++o) {
            const float * wr = w.data() + (size_t) o * in_dim;
            double acc = 0.0;
            for (int i = 0; i < in_dim; ++i) acc += (double) row[i] * wr[i];
            dst[o] = (float) acc;
        }
    }
}

float umt5_gelu_new(float x) {
    const double xd = (double) x;
    const double c = std::sqrt(2.0 / M_PI);
    return (float) (0.5 * xd * (1.0 + std::tanh(c * (xd + 0.044715 * xd * xd * xd))));
}



bool run_umt5_block_host(
        gguf_reader & g,
        int block,
        const std::vector<float> & x,
        int seq,
        int dim,
        int heads,
        int d_kv,
        int d_ff,
        int buckets,
        int max_distance,
        std::vector<float> & block_out,
        const std::string & dump_dir,
        bool dump_first_block) {
    const std::string p = "text.blk." + std::to_string(block);

    std::vector<float> norm_w((size_t) dim);
    std::vector<float> qw((size_t) dim * dim);
    std::vector<float> kw((size_t) dim * dim);
    std::vector<float> vw((size_t) dim * dim);
    std::vector<float> ow((size_t) dim * dim);
    std::vector<float> rb((size_t) buckets * heads);
    if (!g.read_to_f32((p + ".attn_norm.weight").c_str(), norm_w.data(), (int64_t) norm_w.size()) ||
        !g.read_to_f32((p + ".attn.q.weight").c_str(), qw.data(), (int64_t) qw.size()) ||
        !g.read_to_f32((p + ".attn.k.weight").c_str(), kw.data(), (int64_t) kw.size()) ||
        !g.read_to_f32((p + ".attn.v.weight").c_str(), vw.data(), (int64_t) vw.size()) ||
        !g.read_to_f32((p + ".attn.o.weight").c_str(), ow.data(), (int64_t) ow.size()) ||
        !g.read_to_f32((p + ".attn.rel_bias.weight").c_str(), rb.data(), (int64_t) rb.size())) {
        return false;
    }

    std::vector<float> norm;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    umt5_rms_norm_host(x, norm_w, seq, dim, 1e-6f, norm);
    umt5_linear_host(norm, qw, seq, dim, dim, q);
    umt5_linear_host(norm, kw, seq, dim, dim, k);
    umt5_linear_host(norm, vw, seq, dim, dim, v);

    std::vector<float> pos_bias((size_t) heads * seq * seq, 0.0f);
    for (int h = 0; h < heads; ++h) {
        for (int qi = 0; qi < seq; ++qi) {
            for (int kj = 0; kj < seq; ++kj) {
                const int64_t rel = (int64_t) kj - (int64_t) qi;
                const int64_t bucket = umt5_relative_position_bucket(rel, true, buckets, max_distance);
                pos_bias[((size_t) h * seq + qi) * seq + kj] = rb[(size_t) bucket * heads + h];
            }
        }
    }

    std::vector<float> ctx((size_t) seq * dim, 0.0f);
    std::vector<float> scores((size_t) seq, 0.0f);
    for (int h = 0; h < heads; ++h) {
        for (int qi = 0; qi < seq; ++qi) {
            float max_score = -std::numeric_limits<float>::infinity();
            for (int kj = 0; kj < seq; ++kj) {
                double dot = 0.0;
                const float * qv = q.data() + (size_t) qi * dim + (size_t) h * d_kv;
                const float * kv = k.data() + (size_t) kj * dim + (size_t) h * d_kv;
                for (int d = 0; d < d_kv; ++d) dot += (double) qv[d] * kv[d];
                const float s = (float) dot + pos_bias[((size_t) h * seq + qi) * seq + kj];
                scores[(size_t) kj] = s;
                max_score = std::max(max_score, s);
            }
            double denom = 0.0;
            for (int kj = 0; kj < seq; ++kj) {
                const double e = std::exp((double) scores[(size_t) kj] - (double) max_score);
                scores[(size_t) kj] = (float) e;
                denom += e;
            }
            const double inv = denom > 0.0 ? 1.0 / denom : 0.0;
            float * dst = ctx.data() + (size_t) qi * dim + (size_t) h * d_kv;
            for (int kj = 0; kj < seq; ++kj) {
                const float a = (float) ((double) scores[(size_t) kj] * inv);
                const float * vv = v.data() + (size_t) kj * dim + (size_t) h * d_kv;
                for (int d = 0; d < d_kv; ++d) dst[d] += a * vv[d];
            }
        }
    }

    std::vector<float> attn_out;
    std::vector<float> residual((size_t) seq * dim, 0.0f);
    umt5_linear_host(ctx, ow, seq, dim, dim, attn_out);
    for (size_t i = 0; i < residual.size(); ++i) residual[i] = x[i] + attn_out[i];

    std::vector<float> ffn_norm_w((size_t) dim);
    std::vector<float> wi0((size_t) d_ff * dim);
    std::vector<float> wi1((size_t) d_ff * dim);
    std::vector<float> wo_ffn((size_t) dim * d_ff);
    if (!g.read_to_f32((p + ".ffn_norm.weight").c_str(), ffn_norm_w.data(), (int64_t) ffn_norm_w.size()) ||
        !g.read_to_f32((p + ".ffn.wi_0.weight").c_str(), wi0.data(), (int64_t) wi0.size()) ||
        !g.read_to_f32((p + ".ffn.wi_1.weight").c_str(), wi1.data(), (int64_t) wi1.size()) ||
        !g.read_to_f32((p + ".ffn.wo.weight").c_str(), wo_ffn.data(), (int64_t) wo_ffn.size())) {
        return false;
    }

    std::vector<float> ffn_norm;
    std::vector<float> ffn_gate;
    std::vector<float> ffn_linear;
    std::vector<float> ffn_hidden((size_t) seq * d_ff, 0.0f);
    std::vector<float> ffn_out;
    umt5_rms_norm_host(residual, ffn_norm_w, seq, dim, 1e-6f, ffn_norm);
    umt5_linear_host(ffn_norm, wi0, seq, dim, d_ff, ffn_gate);
    umt5_linear_host(ffn_norm, wi1, seq, dim, d_ff, ffn_linear);
    for (size_t i = 0; i < ffn_hidden.size(); ++i) {
        ffn_hidden[i] = umt5_gelu_new(ffn_gate[i]) * ffn_linear[i];
    }
    umt5_linear_host(ffn_hidden, wo_ffn, seq, d_ff, dim, ffn_out);

    block_out.assign((size_t) seq * dim, 0.0f);
    for (size_t i = 0; i < block_out.size(); ++i) block_out[i] = residual[i] + ffn_out[i];

    if (!dump_dir.empty()) {
        dump_f32_file(dump_dir + "/umt5_block" + std::to_string(block) + "_out.f32", block_out);
        dump_text_file(dump_dir + "/umt5_block" + std::to_string(block) + "_out.shape.txt",
                       std::to_string(seq) + " " + std::to_string(dim) + "\n");
        if (dump_first_block && block == 0) {
            dump_f32_file(dump_dir + "/umt5_blocks_input.f32", x);
            dump_text_file(dump_dir + "/umt5_blocks_input.shape.txt",
                           std::to_string(seq) + " " + std::to_string(dim) + "\n");
            dump_f32_file(dump_dir + "/umt5_blocks_block0_norm.f32", norm);
            dump_f32_file(dump_dir + "/umt5_blocks_block0_q.f32", q);
            dump_f32_file(dump_dir + "/umt5_blocks_block0_position_bias.f32", pos_bias);
            dump_f32_file(dump_dir + "/umt5_blocks_block0_context.f32", ctx);
            dump_f32_file(dump_dir + "/umt5_blocks_block0_attn_out.f32", attn_out);
            dump_f32_file(dump_dir + "/umt5_blocks_block0_residual.f32", residual);
        }
    }

    std::printf("vla(lingbot_va): UMT5 block %d host ok: checksum=%.9g max=%.9g\n",
                block, checksum(block_out), max_abs_value(block_out));
    return true;
}

int text_blocks_runtime_count(int layers) {
    int blocks = 2;
    if (const char * env = std::getenv("VLA_LINGBOT_TEXT_BLOCKS")) {
        char * end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end && *end == '\0' && v > 0) {
            blocks = (int) v;
        } else {
            std::fprintf(stderr,
                         "vla(lingbot_va): ignoring invalid VLA_LINGBOT_TEXT_BLOCKS='%s'\n",
                         env);
        }
    }
    return std::max(1, std::min(blocks, layers));
}



std::vector<int32_t> text_runtime_token_ids(int vocab) {
    std::vector<int32_t> ids = { 1, 42, 1234, 32000 };
    if (const char * env = std::getenv("VLA_LINGBOT_TEXT_TOKEN_IDS")) {
        ids.clear();
        const char * p = env;
        while (*p) {
            char * end = nullptr;
            const long v = std::strtol(p, &end, 10);
            if (end == p) {
                std::fprintf(stderr,
                             "vla(lingbot_va): invalid VLA_LINGBOT_TEXT_TOKEN_IDS near '%s'\n",
                             p);
                ids = { 1, 42, 1234, 32000 };
                break;
            }
            ids.push_back((int32_t) v);
            p = end;
            while (*p == ',' || *p == ' ' || *p == '\t') ++p;
        }
    }
    if (ids.empty()) ids = { 1 };
    for (int32_t id : ids) {
        if (id < 0 || id >= vocab) {
            std::fprintf(stderr,
                         "vla(lingbot_va): token id %d is outside vocab=%d\n",
                         id, vocab);
            return {};
        }
    }
    return ids;
}



bool encode_umt5_tokens_host(
        gguf_reader & g,
        const std::vector<int32_t> & token_ids,
        int requested_blocks,
        std::vector<float> & final_out,
        int64_t & out_seq,
        int64_t & out_dim) {
    LingBotScopedTimer timer("umt5_encode_host");
    if (token_ids.empty()) {
        std::fprintf(stderr, "vla(lingbot_va): UMT5 encode requested with empty token ids\n");
        return false;
    }
    if (!validate_text_encoder_tensors(g)) return false;
    const int layers = (int) g.u32("lingbot_va.text_encoder.layers");
    const int dim = (int) g.u32("lingbot_va.text_encoder.d_model");
    const int d_ff = (int) g.u32("lingbot_va.text_encoder.d_ff");
    const int heads = (int) g.u32("lingbot_va.text_encoder.heads");
    const int d_kv = (int) g.u32("lingbot_va.text_encoder.d_kv");
    const int vocab = (int) g.u32("lingbot_va.text_encoder.vocab_size");
    const int buckets = (int) g.u32("lingbot_va.text_encoder.relative_attention_num_buckets");
    const int max_distance = (int) g.u32("lingbot_va.text_encoder.relative_attention_max_distance");
    if (dim != 4096 || heads != 64 || d_kv != 64 || heads * d_kv != dim) {
        std::fprintf(stderr, "vla(lingbot_va): unexpected UMT5 encode dims dim=%d heads=%d d_kv=%d\n",
                     dim, heads, d_kv);
        return false;
    }
    for (int32_t id : token_ids) {
        if (id < 0 || id >= vocab) {
            std::fprintf(stderr, "vla(lingbot_va): UMT5 token id %d outside vocab=%d\n", id, vocab);
            return false;
        }
    }

    const int seq = (int) token_ids.size();
    std::vector<float> x((size_t) seq * dim);
    if (!g.read_bf16_rows_to_f32("text.token_embd.weight", token_ids, dim, vocab, x.data())) {
        return false;
    }
    const int blocks = std::max(0, std::min(requested_blocks <= 0 ? layers : requested_blocks, layers));
    std::vector<float> tmp;
    for (int i = 0; i < blocks; ++i) {
        if (!run_umt5_block_host(g, i, x, seq, dim, heads, d_kv, d_ff, buckets, max_distance,
                                 tmp, std::string(), false)) {
            return false;
        }
        x.swap(tmp);
    }
    std::vector<float> final_norm_w((size_t) dim);
    if (!g.read_to_f32("text.final_norm.weight", final_norm_w.data(), (int64_t) final_norm_w.size())) {
        return false;
    }
    umt5_rms_norm_host(x, final_norm_w, seq, dim, 1e-6f, final_out);
    if (!lingbot_env_enabled("VLA_LINGBOT_TEXT_F32_OUTPUT")) {
        lingbot_bf16_roundtrip_inplace(final_out);
    }
    out_seq = seq;
    out_dim = dim;
    std::printf("vla(lingbot_va): UMT5 encode host ok: tokens=%d blocks=%d checksum=%.9g max=%.9g\n",
                seq, blocks, checksum(final_out), max_abs_value(final_out));
    return true;
}

struct LingBotExecState {
    int64_t seq = 0;
    int64_t text_seq = 0;
    uint64_t cache_session_id = 0;
    int cache_mode = 0;
    int cache_branch = 0;
    int64_t cache_block_index = -1;
    std::vector<float> raw_input;
    std::vector<float> time_raw;
    std::vector<float> x;
    std::vector<float> text;
    std::vector<float> t_hidden;
    std::vector<float> timestep_proj;
    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
    uint64_t text_signature = 0;
    uint64_t rope_signature = 0;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    bool allow_device_x_handoff = false;
    struct CudaDeviceBuffer {
        void * data = nullptr;
        size_t bytes = 0;
        ~CudaDeviceBuffer() {
            if (data) cudaFree(data);
        }
        bool ensure(size_t requested) {
            if (data && bytes == requested) return true;
            if (data) {
                cudaFree(data);
                data = nullptr;
                bytes = 0;
            }
            if (requested == 0) return true;
            const cudaError_t err = cudaMalloc(&data, requested);
            if (err != cudaSuccess) {
                std::fprintf(stderr,
                             "vla(lingbot_va): cudaMalloc(device-x staging, %zu) failed: %s\n",
                             requested, cudaGetErrorString(err));
                return false;
            }
            bytes = requested;
            return true;
        }
    };
    std::shared_ptr<CudaDeviceBuffer> device_x;
    ggml_backend_t device_x_backend = nullptr;
    std::shared_ptr<CudaDeviceBuffer> device_out;
    ggml_backend_t device_out_backend = nullptr;
    std::shared_ptr<CudaDeviceBuffer> device_text;
    ggml_backend_t device_text_backend = nullptr;
    std::shared_ptr<CudaDeviceBuffer> device_timestep_proj;
    ggml_backend_t device_timestep_proj_backend = nullptr;
    bool allow_device_self_ctx_handoff = false;
    bool prefer_device_out_only = false;
#endif
};

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
bool lingbot_device_x_valid(
        const LingBotExecState & state,
        int64_t hidden,
        ggml_backend_t backend) {
    return state.device_x != nullptr &&
           state.device_x->data != nullptr &&
           state.device_x_backend == backend &&
           state.device_x->bytes == (size_t) hidden * (size_t) state.seq * sizeof(float);
}

void lingbot_clear_device_x(LingBotExecState & state) {
    state.device_x.reset();
    state.device_x_backend = nullptr;
}

bool lingbot_device_out_valid(
        const LingBotExecState & state,
        size_t bytes,
        ggml_backend_t backend) {
    return state.device_out != nullptr &&
           state.device_out->data != nullptr &&
           state.device_out_backend == backend &&
           state.device_out->bytes == bytes;
}

void lingbot_clear_device_out(LingBotExecState & state) {
    state.device_out.reset();
    state.device_out_backend = nullptr;
}

bool lingbot_device_text_valid(
        const LingBotExecState & state,
        size_t bytes,
        ggml_backend_t backend) {
    return state.device_text != nullptr &&
           state.device_text->data != nullptr &&
           state.device_text_backend == backend &&
           state.device_text->bytes == bytes;
}

void lingbot_clear_device_text(LingBotExecState & state) {
    state.device_text.reset();
    state.device_text_backend = nullptr;
}

bool lingbot_device_timestep_proj_valid(
        const LingBotExecState & state,
        size_t bytes,
        ggml_backend_t backend) {
    return state.device_timestep_proj != nullptr &&
           state.device_timestep_proj->data != nullptr &&
           state.device_timestep_proj_backend == backend &&
           state.device_timestep_proj->bytes == bytes;
}

void lingbot_clear_device_timestep_proj(LingBotExecState & state) {
    state.device_timestep_proj.reset();
    state.device_timestep_proj_backend = nullptr;
}

bool lingbot_copy_tensor_to_device_x(
        LingBotExecState & state,
        ggml_backend_t backend,
        ggml_tensor * src,
        const char * label) {
    if (!src || !src->data) return false;
    if (!state.device_x) {
        state.device_x = std::make_shared<LingBotExecState::CudaDeviceBuffer>();
    }
    const size_t bytes = ggml_nbytes(src);
    if (!state.device_x->ensure(bytes)) return false;
    const cudaError_t err = cudaMemcpy(state.device_x->data, src->data, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): cudaMemcpy(%s -> device-x staging) failed: %s\n",
                     label ? label : "tensor", cudaGetErrorString(err));
        return false;
    }
    state.device_x_backend = backend;
    return true;
}

bool lingbot_copy_tensor_to_device_out(
        LingBotExecState & state,
        ggml_backend_t backend,
        ggml_tensor * src,
        const char * label) {
    if (!src || !src->data) return false;
    if (!state.device_out) {
        state.device_out = std::make_shared<LingBotExecState::CudaDeviceBuffer>();
    }
    const size_t bytes = ggml_nbytes(src);
    if (!state.device_out->ensure(bytes)) return false;
    const cudaError_t err = cudaMemcpy(state.device_out->data, src->data, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): cudaMemcpy(%s -> device-output staging) failed: %s\n",
                     label ? label : "tensor", cudaGetErrorString(err));
        return false;
    }
    state.device_out_backend = backend;
    return true;
}

bool lingbot_copy_tensor_to_device_text(
        LingBotExecState & state,
        ggml_backend_t backend,
        ggml_tensor * src,
        const char * label) {
    if (!src || !src->data) return false;
    if (!state.device_text) {
        state.device_text = std::make_shared<LingBotExecState::CudaDeviceBuffer>();
    }
    const size_t bytes = ggml_nbytes(src);
    if (!state.device_text->ensure(bytes)) return false;
    const cudaError_t err = cudaMemcpy(state.device_text->data, src->data, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): cudaMemcpy(%s -> device-text staging) failed: %s\n",
                     label ? label : "tensor", cudaGetErrorString(err));
        return false;
    }
    state.device_text_backend = backend;
    return true;
}

bool lingbot_copy_tensor_to_device_timestep_proj(
        LingBotExecState & state,
        ggml_backend_t backend,
        ggml_tensor * src,
        const char * label) {
    if (!src || !src->data) return false;
    if (!state.device_timestep_proj) {
        state.device_timestep_proj = std::make_shared<LingBotExecState::CudaDeviceBuffer>();
    }
    const size_t bytes = ggml_nbytes(src);
    if (!state.device_timestep_proj->ensure(bytes)) return false;
    const cudaError_t err = cudaMemcpy(state.device_timestep_proj->data, src->data, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): cudaMemcpy(%s -> device-timestep-proj staging) failed: %s\n",
                     label ? label : "tensor", cudaGetErrorString(err));
        return false;
    }
    state.device_timestep_proj_backend = backend;
    return true;
}

bool lingbot_copy_device_text_to_tensor(
        const LingBotExecState & state,
        ggml_tensor * dst,
        const char * label) {
    if (!dst || !dst->data || !state.device_text || !state.device_text->data) return false;
    const size_t bytes = ggml_nbytes(dst);
    if (state.device_text->bytes != bytes) {
        std::fprintf(stderr,
                     "vla(lingbot_va): device-text staging size mismatch for %s: have=%zu want=%zu\n",
                     label ? label : "tensor", state.device_text->bytes, bytes);
        return false;
    }
    const cudaError_t err = cudaMemcpy(dst->data, state.device_text->data, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): cudaMemcpy(device-text staging -> %s) failed: %s\n",
                     label ? label : "tensor", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool lingbot_copy_device_x_to_tensor(
        const LingBotExecState & state,
        ggml_tensor * dst,
        const char * label) {
    if (!dst || !dst->data || !state.device_x || !state.device_x->data) return false;
    const size_t bytes = ggml_nbytes(dst);
    if (state.device_x->bytes != bytes) {
        std::fprintf(stderr,
                     "vla(lingbot_va): device-x staging size mismatch for %s: have=%zu want=%zu\n",
                     label ? label : "tensor", state.device_x->bytes, bytes);
        return false;
    }
    const cudaError_t err = cudaMemcpy(dst->data, state.device_x->data, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): cudaMemcpy(device-x staging -> %s) failed: %s\n",
                     label ? label : "tensor", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool lingbot_copy_device_timestep_proj_to_tensor(
        const LingBotExecState & state,
        ggml_tensor * dst,
        const char * label) {
    if (!dst || !dst->data || !state.device_timestep_proj || !state.device_timestep_proj->data) return false;
    const size_t bytes = ggml_nbytes(dst);
    if (state.device_timestep_proj->bytes != bytes) {
        std::fprintf(stderr,
                     "vla(lingbot_va): device-timestep-proj staging size mismatch for %s: have=%zu want=%zu\n",
                     label ? label : "tensor", state.device_timestep_proj->bytes, bytes);
        return false;
    }
    const cudaError_t err = cudaMemcpy(dst->data, state.device_timestep_proj->data, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): cudaMemcpy(device-timestep-proj staging -> %s) failed: %s\n",
                     label ? label : "tensor", cudaGetErrorString(err));
        return false;
    }
    return true;
}

struct LingBotDeviceTensor {
    std::shared_ptr<LingBotExecState::CudaDeviceBuffer> buffer;
    ggml_backend_t backend = nullptr;
    size_t count = 0;
};

size_t lingbot_tensor5d_count(const LingBotTensor5DShape & shape) {
    if (!shape_valid(shape)) return 0;
    return (size_t) shape.b * (size_t) shape.c * (size_t) shape.f *
           (size_t) shape.h * (size_t) shape.w;
}

bool lingbot_device_tensor_valid(
        const LingBotDeviceTensor & tensor,
        size_t count,
        ggml_backend_t backend) {
    return tensor.buffer != nullptr &&
           tensor.buffer->data != nullptr &&
           tensor.backend == backend &&
           tensor.count == count &&
           tensor.buffer->bytes == count * sizeof(float);
}

bool lingbot_upload_host_tensor(
        LingBotDeviceTensor & dst,
        ggml_backend_t backend,
        const std::vector<float> & src,
        const char * label) {
    if (!backend || !ggml_backend_is_cuda(backend)) return false;
    if (!dst.buffer) {
        dst.buffer = std::make_shared<LingBotExecState::CudaDeviceBuffer>();
    }
    const size_t bytes = src.size() * sizeof(float);
    if (!dst.buffer->ensure(bytes)) return false;
    if (bytes > 0) {
        const cudaError_t err = cudaMemcpy(dst.buffer->data, src.data(), bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): cudaMemcpy(%s H2D) failed: %s\n",
                         label ? label : "tensor", cudaGetErrorString(err));
            return false;
        }
    }
    dst.backend = backend;
    dst.count = src.size();
    return true;
}

bool lingbot_download_device_tensor(
        const LingBotDeviceTensor & src,
        ggml_backend_t backend,
        std::vector<float> & dst,
        const char * label) {
    if (!lingbot_device_tensor_valid(src, dst.size(), backend)) return false;
    const size_t bytes = dst.size() * sizeof(float);
    if (bytes > 0) {
        const cudaError_t err = cudaMemcpy(dst.data(), src.buffer->data, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): cudaMemcpy(%s D2H) failed: %s\n",
                         label ? label : "tensor", cudaGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool lingbot_detokenize_device_out_to_tensor(
        const LingBotExecState & state,
        ggml_backend_t backend,
        bool action_mode,
        const LingBotTensor5DShape & shape,
        int64_t pt,
        int64_t ph,
        int64_t pw,
        LingBotDeviceTensor & out_tensor) {
    if (!backend || !ggml_backend_is_cuda(backend) || !shape_valid(shape)) return false;
    const size_t tensor_count = lingbot_tensor5d_count(shape);
    if (tensor_count == 0) return false;
    size_t token_count = tensor_count;
    if (!action_mode) {
        if (pt <= 0 || ph <= 0 || pw <= 0 ||
            shape.f % pt != 0 || shape.h % ph != 0 || shape.w % pw != 0) {
            return false;
        }
        const int64_t pf = shape.f / pt;
        const int64_t phn = shape.h / ph;
        const int64_t pwn = shape.w / pw;
        token_count = (size_t) shape.b * (size_t) pf * (size_t) phn * (size_t) pwn *
                      (size_t) shape.c * (size_t) pt * (size_t) ph * (size_t) pw;
    }
    if (!lingbot_device_out_valid(state, token_count * sizeof(float), backend)) {
        return false;
    }
    if (shape.b > INT_MAX || shape.c > INT_MAX || shape.f > INT_MAX ||
        shape.h > INT_MAX || shape.w > INT_MAX ||
        pt > INT_MAX || ph > INT_MAX || pw > INT_MAX) {
        return false;
    }
    if (!out_tensor.buffer) {
        out_tensor.buffer = std::make_shared<LingBotExecState::CudaDeviceBuffer>();
    }
    if (!out_tensor.buffer->ensure(tensor_count * sizeof(float))) return false;
    int rc = 0;
    if (action_mode) {
        rc = lingbot_action_tokens_to_tensor_f32(
            static_cast<const float *>(state.device_out->data),
            static_cast<float *>(out_tensor.buffer->data),
            (int) shape.b, (int) shape.c, (int) shape.f, (int) shape.h, (int) shape.w, nullptr);
    } else {
        rc = lingbot_projected_latent_to_tensor_f32(
            static_cast<const float *>(state.device_out->data),
            static_cast<float *>(out_tensor.buffer->data),
            (int) shape.b, (int) shape.c, (int) shape.f, (int) shape.h, (int) shape.w,
            (int) pt, (int) ph, (int) pw, nullptr);
    }
    if (rc != 0) {
        std::fprintf(stderr, "vla(lingbot_va): device detokenize kernel failed action=%d\n",
                     action_mode ? 1 : 0);
        return false;
    }
    out_tensor.backend = backend;
    out_tensor.count = tensor_count;
    return true;
}

bool lingbot_tokenize_device_sample_to_tokens(
        const LingBotDeviceTensor & sample,
        ggml_backend_t backend,
        bool action_mode,
        const LingBotTensor5DShape & shape,
        int64_t pt,
        int64_t ph,
        int64_t pw,
        LingBotDeviceTensor & out_tokens,
        int64_t & feature,
        int64_t & seq) {
    feature = 0;
    seq = 0;
    if (!backend || !ggml_backend_is_cuda(backend) || !shape_valid(shape)) return false;
    const size_t tensor_count = lingbot_tensor5d_count(shape);
    if (!lingbot_device_tensor_valid(sample, tensor_count, backend)) return false;
    size_t token_count = tensor_count;
    if (action_mode) {
        feature = shape.c;
        seq = shape.b * shape.f * shape.h * shape.w;
    } else {
        if (pt <= 0 || ph <= 0 || pw <= 0 ||
            shape.f % pt != 0 || shape.h % ph != 0 || shape.w % pw != 0) {
            return false;
        }
        feature = shape.c * pt * ph * pw;
        seq = shape.b * (shape.f / pt) * (shape.h / ph) * (shape.w / pw);
        token_count = (size_t) feature * (size_t) seq;
    }
    if (feature <= 0 || seq <= 0) return false;
    if (shape.b > INT_MAX || shape.c > INT_MAX || shape.f > INT_MAX ||
        shape.h > INT_MAX || shape.w > INT_MAX ||
        pt > INT_MAX || ph > INT_MAX || pw > INT_MAX) {
        return false;
    }
    if (!out_tokens.buffer) {
        out_tokens.buffer = std::make_shared<LingBotExecState::CudaDeviceBuffer>();
    }
    if (!out_tokens.buffer->ensure(token_count * sizeof(float))) return false;
    int rc = 0;
    if (action_mode) {
        rc = lingbot_action_tensor_to_tokens_f32(
            static_cast<const float *>(sample.buffer->data),
            static_cast<float *>(out_tokens.buffer->data),
            (int) shape.b, (int) shape.c, (int) shape.f, (int) shape.h, (int) shape.w, nullptr);
    } else {
        rc = lingbot_patchify_latent_f32(
            static_cast<const float *>(sample.buffer->data),
            static_cast<float *>(out_tokens.buffer->data),
            (int) shape.b, (int) shape.c, (int) shape.f, (int) shape.h, (int) shape.w,
            (int) pt, (int) ph, (int) pw, nullptr);
    }
    if (rc != 0) {
        std::fprintf(stderr, "vla(lingbot_va): device tokenize kernel failed action=%d\n",
                     action_mode ? 1 : 0);
        return false;
    }
    out_tokens.backend = backend;
    out_tokens.count = token_count;
    return true;
}

bool lingbot_device_scheduler_step_bridge(
        ggml_backend_t backend,
        LingBotDeviceTensor & sample_device,
        LingBotDeviceTensor & pred_device,
        std::vector<float> & sample_host,
        const std::vector<float> & pred_host,
        float delta,
        bool bf16_state,
        const char * label,
        bool download_host) {
    if (sample_host.size() != pred_host.size()) return false;
    if (!lingbot_device_tensor_valid(sample_device, sample_host.size(), backend) &&
        !lingbot_upload_host_tensor(sample_device, backend, sample_host, label)) {
        return false;
    }
    if (!lingbot_upload_host_tensor(pred_device, backend, pred_host, label)) {
        return false;
    }
    const int rc = lingbot_scheduler_step_f32(
        static_cast<float *>(sample_device.buffer->data),
        static_cast<const float *>(pred_device.buffer->data),
        (int) sample_host.size(),
        delta,
        bf16_state ? 1 : 0,
        nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "vla(lingbot_va): device scheduler bridge failed for %s\n",
                     label ? label : "sample");
        return false;
    }
    return !download_host || lingbot_download_device_tensor(sample_device, backend, sample_host, label);
}

bool lingbot_device_scheduler_step_device_pred(
        ggml_backend_t backend,
        LingBotDeviceTensor & sample_device,
        const LingBotDeviceTensor & pred_device,
        size_t count,
        float delta,
        bool bf16_state,
        const char * label) {
    if (!lingbot_device_tensor_valid(sample_device, count, backend) ||
        !lingbot_device_tensor_valid(pred_device, count, backend) ||
        count > (size_t) INT_MAX) {
        return false;
    }
    const int rc = lingbot_scheduler_step_f32(
        static_cast<float *>(sample_device.buffer->data),
        static_cast<const float *>(pred_device.buffer->data),
        (int) count,
        delta,
        bf16_state ? 1 : 0,
        nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "vla(lingbot_va): device scheduler pred bridge failed for %s\n",
                     label ? label : "sample");
        return false;
    }
    return true;
}

bool lingbot_device_model_output_guidance_bridge(
        ggml_backend_t backend,
        LingBotDeviceTensor & cond,
        const LingBotDeviceTensor * uncond,
        size_t count,
        float scale,
        bool use_guidance,
        bool bf16_state,
        const char * label) {
    if (!lingbot_device_tensor_valid(cond, count, backend) || count > (size_t) INT_MAX) {
        return false;
    }
    const float * uncond_ptr = nullptr;
    if (use_guidance) {
        if (!uncond || !lingbot_device_tensor_valid(*uncond, count, backend)) {
            return false;
        }
        uncond_ptr = static_cast<const float *>(uncond->buffer->data);
    }
    const int rc = lingbot_model_output_guidance_f32(
        static_cast<float *>(cond.buffer->data),
        uncond_ptr,
        (int) count,
        scale,
        use_guidance ? 1 : 0,
        bf16_state ? 1 : 0,
        nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "vla(lingbot_va): device model-output guidance failed for %s\n",
                     label ? label : "model_output");
        return false;
    }
    return true;
}

bool lingbot_device_action_finalize_bridge(
        ggml_backend_t backend,
        LingBotDeviceTensor & action_sample,
        const LingBotDeviceTensor * action_cond,
        const LingBotTensor5DShape & action_shape,
        bool bf16_state) {
    if (!shape_valid(action_shape) || action_shape.b != 1 ||
        action_shape.c > INT_MAX || action_shape.f > INT_MAX ||
        action_shape.h > INT_MAX || action_shape.w > INT_MAX) {
        return false;
    }
    const size_t count = lingbot_tensor5d_count(action_shape);
    if (!lingbot_device_tensor_valid(action_sample, count, backend)) return false;
    const float * cond_ptr = nullptr;
    if (action_cond) {
        if (!lingbot_device_tensor_valid(*action_cond, count, backend)) return false;
        cond_ptr = static_cast<const float *>(action_cond->buffer->data);
    }
    const int rc = lingbot_action_finalize_f32(
        static_cast<float *>(action_sample.buffer->data),
        cond_ptr,
        (int) action_shape.c, (int) action_shape.f, (int) action_shape.h, (int) action_shape.w,
        7,
        bf16_state ? 1 : 0,
        nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "vla(lingbot_va): device action finalize bridge failed\n");
        return false;
    }
    return true;
}

bool lingbot_device_latent_restore_round_bridge(
        ggml_backend_t backend,
        LingBotDeviceTensor & latent_sample,
        const LingBotDeviceTensor * latent_cond,
        const LingBotTensor5DShape & latent_shape,
        const LingBotTensor5DShape & latent_cond_shape,
        bool bf16_state) {
    if (!shape_valid(latent_shape) || latent_shape.b != 1 ||
        latent_shape.c > INT_MAX || latent_shape.f > INT_MAX ||
        latent_shape.h > INT_MAX || latent_shape.w > INT_MAX) {
        return false;
    }
    const size_t count = lingbot_tensor5d_count(latent_shape);
    if (!lingbot_device_tensor_valid(latent_sample, count, backend)) return false;
    const float * cond_ptr = nullptr;
    int64_t cond_f = 0;
    if (latent_cond) {
        if (!shape_valid(latent_cond_shape) ||
            latent_cond_shape.b != 1 ||
            latent_cond_shape.c != latent_shape.c ||
            latent_cond_shape.h != latent_shape.h ||
            latent_cond_shape.w != latent_shape.w ||
            latent_cond_shape.f < 0 ||
            latent_cond_shape.f > latent_shape.f ||
            !lingbot_device_tensor_valid(*latent_cond, lingbot_tensor5d_count(latent_cond_shape), backend)) {
            return false;
        }
        cond_ptr = static_cast<const float *>(latent_cond->buffer->data);
        cond_f = latent_cond_shape.f;
    }
    const int rc = lingbot_latent_restore_round_f32(
        static_cast<float *>(latent_sample.buffer->data),
        cond_ptr,
        (int) latent_shape.c, (int) latent_shape.f, (int) latent_shape.h, (int) latent_shape.w,
        (int) cond_f,
        bf16_state ? 1 : 0,
        nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "vla(lingbot_va): device latent restore bridge failed\n");
        return false;
    }
    return true;
}

bool lingbot_device_action_sample_to_output_bridge(
        ggml_backend_t backend,
        const LingBotDeviceTensor & action_sample,
        const LingBotTensor5DShape & action_shape,
        int64_t n_suffix,
        int64_t output_dim,
        bool postprocess_libero,
        std::vector<float> & out) {
    if (!shape_valid(action_shape) || action_shape.b > INT_MAX ||
        action_shape.c > INT_MAX || action_shape.f > INT_MAX ||
        action_shape.h > INT_MAX || action_shape.w > INT_MAX ||
        n_suffix < 0 || n_suffix > INT_MAX ||
        output_dim < 0 || output_dim > INT_MAX) {
        return false;
    }
    const size_t sample_count = lingbot_tensor5d_count(action_shape);
    if (!lingbot_device_tensor_valid(action_sample, sample_count, backend)) return false;
    const size_t out_count = (size_t) n_suffix * (size_t) output_dim;
    LingBotDeviceTensor out_device;
    out_device.buffer = std::make_shared<LingBotExecState::CudaDeviceBuffer>();
    if (!out_device.buffer->ensure(out_count * sizeof(float))) return false;
    out_device.backend = backend;
    out_device.count = out_count;
    const int rc = lingbot_action_sample_to_output_f32(
        static_cast<const float *>(action_sample.buffer->data),
        static_cast<float *>(out_device.buffer->data),
        (int) action_shape.b, (int) action_shape.c, (int) action_shape.f,
        (int) action_shape.h, (int) action_shape.w,
        (int) n_suffix, (int) output_dim,
        postprocess_libero ? 1 : 0,
        nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "vla(lingbot_va): device action final output bridge failed\n");
        return false;
    }
    out.assign(out_count, 0.0f);
    return lingbot_download_device_tensor(out_device, backend, out, "action_output_final");
}

#else
struct LingBotDeviceTensor {};
#endif

struct LingBotBlockSequenceGraph {
    LingBotRuntimeWeights * weights = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * x_in = nullptr;
    ggml_tensor * text = nullptr;
    ggml_tensor * timestep_proj = nullptr;
    ggml_tensor * rope_cos = nullptr;
    ggml_tensor * rope_sin = nullptr;
    ggml_tensor * out = nullptr;
    int64_t first_block = -1;
    int64_t n_blocks = 0;
    int64_t seq = 0;
    int64_t text_seq = 0;

    bool matches(
            LingBotRuntimeWeights & bw,
            int64_t first,
            int64_t count,
            int64_t seq_,
            int64_t text_seq_) const {
        return weights == &bw &&
               first_block == first &&
               n_blocks == count &&
               seq == seq_ &&
               text_seq == text_seq_;
    }

    void reset() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
        weights = nullptr;
        ctx = nullptr;
        graph = nullptr;
        galloc = nullptr;
        x_in = nullptr;
        text = nullptr;
        timestep_proj = nullptr;
        rope_cos = nullptr;
        rope_sin = nullptr;
        out = nullptr;
        first_block = -1;
        n_blocks = 0;
        seq = 0;
        text_seq = 0;
    }

    ~LingBotBlockSequenceGraph() {
        reset();
    }
};

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
struct LingBotCudaSelfAttnQkvGraph {
    LingBotRuntimeWeights * weights = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * x_in = nullptr;
    ggml_tensor * timestep_proj = nullptr;
    ggml_tensor * rope_cos = nullptr;
    ggml_tensor * rope_sin = nullptr;
    ggml_tensor * n1 = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * qh = nullptr;
    ggml_tensor * kh = nullptr;
    ggml_tensor * vh = nullptr;
    size_t block_index = 0;
    int64_t seq = 0;
    uint64_t uploaded_rope_signature = 0;

    bool matches(LingBotRuntimeWeights & bw, size_t block_index_, int64_t seq_) const {
        return weights == &bw && block_index == block_index_ && seq == seq_;
    }

    void reset() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
        weights = nullptr;
        ctx = nullptr;
        graph = nullptr;
        galloc = nullptr;
        x_in = nullptr;
        timestep_proj = nullptr;
        rope_cos = nullptr;
        rope_sin = nullptr;
        n1 = nullptr;
        q = nullptr;
        k = nullptr;
        v = nullptr;
        qh = nullptr;
        kh = nullptr;
        vh = nullptr;
        block_index = 0;
        seq = 0;
        uploaded_rope_signature = 0;
    }

    ~LingBotCudaSelfAttnQkvGraph() {
        reset();
    }
};

struct LingBotCudaSelfAttnPostGraph {
    LingBotRuntimeWeights * weights = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * x_in = nullptr;
    ggml_tensor * text = nullptr;
    ggml_tensor * timestep_proj = nullptr;
    ggml_tensor * self_ctx = nullptr;
    ggml_tensor * gate_msa = nullptr;
    ggml_tensor * self_attn_out = nullptr;
    ggml_tensor * gated_self = nullptr;
    ggml_tensor * post_self = nullptr;
    ggml_tensor * n2 = nullptr;
    ggml_tensor * cross_q = nullptr;
    ggml_tensor * cross_k = nullptr;
    ggml_tensor * cross_v = nullptr;
    ggml_tensor * cross_ctx = nullptr;
    ggml_tensor * cross_attn = nullptr;
    ggml_tensor * post_cross = nullptr;
    ggml_tensor * n3 = nullptr;
    ggml_tensor * ff = nullptr;
    ggml_tensor * out = nullptr;
    size_t block_index = 0;
    int64_t seq = 0;
    int64_t text_seq = 0;
    int text_slot = -1;
    bool bf16_boundary = false;
    uint64_t uploaded_text_signature = 0;

    bool matches(
            LingBotRuntimeWeights & bw,
            size_t block_index_,
            int64_t seq_,
            int64_t text_seq_,
            bool bf16_boundary_,
            int text_slot_) const {
        return weights == &bw &&
               block_index == block_index_ &&
               seq == seq_ &&
               text_seq == text_seq_ &&
               bf16_boundary == bf16_boundary_ &&
               text_slot == text_slot_;
    }

    void reset() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
        weights = nullptr;
        ctx = nullptr;
        graph = nullptr;
        galloc = nullptr;
        x_in = nullptr;
        text = nullptr;
        timestep_proj = nullptr;
        self_ctx = nullptr;
        gate_msa = nullptr;
        self_attn_out = nullptr;
        gated_self = nullptr;
        post_self = nullptr;
        n2 = nullptr;
        cross_q = nullptr;
        cross_k = nullptr;
        cross_v = nullptr;
        cross_ctx = nullptr;
        cross_attn = nullptr;
        post_cross = nullptr;
        n3 = nullptr;
        ff = nullptr;
        out = nullptr;
        block_index = 0;
        seq = 0;
        text_seq = 0;
        text_slot = -1;
        bf16_boundary = false;
        uploaded_text_signature = 0;
    }

    ~LingBotCudaSelfAttnPostGraph() {
        reset();
    }
};

struct LingBotCudaSelfAttnPostQkvGraph {
    LingBotRuntimeWeights * weights = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * x_in = nullptr;
    ggml_tensor * text = nullptr;
    ggml_tensor * timestep_proj = nullptr;
    ggml_tensor * self_ctx = nullptr;
    ggml_tensor * rope_cos = nullptr;
    ggml_tensor * rope_sin = nullptr;
    ggml_tensor * post_out = nullptr;
    ggml_tensor * next_n1 = nullptr;
    ggml_tensor * next_q = nullptr;
    ggml_tensor * next_k = nullptr;
    ggml_tensor * next_v = nullptr;
    ggml_tensor * next_qh = nullptr;
    ggml_tensor * next_kh = nullptr;
    ggml_tensor * next_vh = nullptr;
    ggml_tensor * output_t_hidden = nullptr;
    ggml_tensor * output_out = nullptr;
    size_t block_index = 0;
    int64_t seq = 0;
    int64_t text_seq = 0;
    int text_slot = -1;
    bool bf16_boundary = false;
    bool include_next_qkv = true;
    bool include_output = false;
    bool action_mode = false;
    uint64_t uploaded_text_signature = 0;
    uint64_t uploaded_rope_signature = 0;

    bool matches(
            LingBotRuntimeWeights & bw,
            size_t block_index_,
            int64_t seq_,
            int64_t text_seq_,
            bool bf16_boundary_,
            int text_slot_,
            bool include_next_qkv_,
            bool include_output_,
            bool action_mode_) const {
        return weights == &bw &&
               block_index == block_index_ &&
               seq == seq_ &&
               text_seq == text_seq_ &&
               bf16_boundary == bf16_boundary_ &&
               text_slot == text_slot_ &&
               include_next_qkv == include_next_qkv_ &&
               include_output == include_output_ &&
               action_mode == action_mode_;
    }

    void reset() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
        weights = nullptr;
        ctx = nullptr;
        graph = nullptr;
        galloc = nullptr;
        x_in = nullptr;
        text = nullptr;
        timestep_proj = nullptr;
        self_ctx = nullptr;
        rope_cos = nullptr;
        rope_sin = nullptr;
        post_out = nullptr;
        next_n1 = nullptr;
        next_q = nullptr;
        next_k = nullptr;
        next_v = nullptr;
        next_qh = nullptr;
        next_kh = nullptr;
        next_vh = nullptr;
        output_t_hidden = nullptr;
        output_out = nullptr;
        block_index = 0;
        seq = 0;
        text_seq = 0;
        text_slot = -1;
        bf16_boundary = false;
        include_next_qkv = true;
        include_output = false;
        action_mode = false;
        uploaded_text_signature = 0;
        uploaded_rope_signature = 0;
    }

    ~LingBotCudaSelfAttnPostQkvGraph() {
        reset();
    }
};

struct LingBotCudaSelfAttnBlockRunner {
    LingBotRuntimeWeights * weights = nullptr;
    size_t block_index = 0;
    int64_t seq = 0;
    int64_t text_seq = 0;
    bool bf16_boundary = false;
    LingBotCudaSelfAttnQkvGraph * qkv = nullptr;
    LingBotCudaSelfAttnPostGraph * post = nullptr;
    LingBotCudaSelfAttnPostGraph * post_by_branch[2] = {nullptr, nullptr};
    LingBotCudaSelfAttnPostQkvGraph * post_qkv_next = nullptr;
    LingBotCudaSelfAttnPostQkvGraph * post_qkv_next_by_branch[2] = {nullptr, nullptr};

    bool matches(
            LingBotRuntimeWeights & bw,
            size_t block_index_,
            int64_t seq_,
            int64_t text_seq_,
            bool bf16_boundary_) const {
        return weights == &bw &&
               block_index == block_index_ &&
               seq == seq_ &&
               text_seq == text_seq_ &&
               bf16_boundary == bf16_boundary_;
    }
};

struct LingBotCudaSelfAttnForwardRunner {
    LingBotRuntimeWeights * weights = nullptr;
    int blocks = 0;
    int64_t seq = 0;
    int64_t text_seq = 0;
    bool bf16_boundary = false;
    std::vector<LingBotCudaSelfAttnBlockRunner *> block_runners;
    LingBotFlexMaskMeta cuda_meta;
    LingBotBlockSparseTable cuda_self_table;
    std::vector<unsigned char> cuda_token_mask;

    bool matches(
            LingBotRuntimeWeights & bw,
            int blocks_,
            int64_t seq_,
            int64_t text_seq_,
            bool bf16_boundary_) const {
        return weights == &bw &&
               blocks == blocks_ &&
               seq == seq_ &&
               text_seq == text_seq_ &&
               bf16_boundary == bf16_boundary_;
    }
};

struct LingBotOfficialRuntimeKvDenoiseRunner {
    LingBotRuntimeWeights * weights = nullptr;
    int blocks = 0;
    int64_t text_seq = 0;
    int64_t latent_seq = 0;
    int64_t action_seq = 0;
    bool bf16_boundary = false;
    LingBotCudaSelfAttnForwardRunner * latent_forward = nullptr;
    LingBotCudaSelfAttnForwardRunner * action_forward = nullptr;
    LingBotCudaSelfAttnForwardRunner * latent_forward_by_mode[3] = {nullptr, nullptr, nullptr};
    LingBotCudaSelfAttnForwardRunner * action_forward_by_mode[3] = {nullptr, nullptr, nullptr};

    bool matches(
            LingBotRuntimeWeights & bw,
            int blocks_,
            int64_t text_seq_,
            int64_t latent_seq_,
            int64_t action_seq_,
            bool bf16_boundary_) const {
        return weights == &bw &&
               blocks == blocks_ &&
               text_seq == text_seq_ &&
               latent_seq == latent_seq_ &&
               action_seq == action_seq_ &&
               bf16_boundary == bf16_boundary_;
    }
};

int lingbot_cache_mode_slot(int cache_mode) {
    return (cache_mode >= 0 && cache_mode <= 2) ? cache_mode : 0;
}

LingBotCudaSelfAttnForwardRunner * lingbot_select_forward_runner_for_mode(
        LingBotCudaSelfAttnForwardRunner * fallback,
        LingBotCudaSelfAttnForwardRunner * const by_mode[3],
        int cache_mode) {
    const int slot = lingbot_cache_mode_slot(cache_mode);
    return by_mode[slot] ? by_mode[slot] : fallback;
}

LingBotCudaSelfAttnBlockRunner * lingbot_get_cuda_self_attn_block_runner(
        const LingBotVAModelArch & model,
        LingBotRuntimeWeights & bw,
        size_t block_index,
        int64_t seq,
        int64_t text_seq,
        bool bf16_boundary) {
    for (const auto & entry : model.cuda_self_attn_block_runners) {
        if (entry && entry->matches(bw, block_index, seq, text_seq, bf16_boundary)) {
            return entry.get();
        }
    }
    auto built = std::make_unique<LingBotCudaSelfAttnBlockRunner>();
    built->weights = &bw;
    built->block_index = block_index;
    built->seq = seq;
    built->text_seq = text_seq;
    built->bf16_boundary = bf16_boundary;
    LingBotCudaSelfAttnBlockRunner * runner = built.get();
    model.cuda_self_attn_block_runners.push_back(std::move(built));
    std::printf("vla(lingbot_va): cached official CUDA self-attn block runner block=%zu seq=%lld text_seq=%lld bf16_boundary=%d\n",
                block_index + 1, (long long) seq, (long long) text_seq, bf16_boundary ? 1 : 0);
    return runner;
}

LingBotCudaSelfAttnForwardRunner * lingbot_get_cuda_self_attn_forward_runner(
        const LingBotVAModelArch & model,
        LingBotRuntimeWeights & bw,
        int blocks,
        int64_t seq,
        int64_t text_seq,
        bool bf16_boundary) {
    for (const auto & entry : model.cuda_self_attn_forward_runners) {
        if (entry && entry->matches(bw, blocks, seq, text_seq, bf16_boundary)) {
            return entry.get();
        }
    }
    auto built = std::make_unique<LingBotCudaSelfAttnForwardRunner>();
    built->weights = &bw;
    built->blocks = blocks;
    built->seq = seq;
    built->text_seq = text_seq;
    built->bf16_boundary = bf16_boundary;
    built->cuda_meta.seq_ids.assign((size_t) seq, 0);
    built->cuda_self_table = build_dense_block_table(seq, seq, 64);
    built->cuda_token_mask.assign((size_t) seq * (size_t) seq, (unsigned char) 1);
    built->block_runners.reserve((size_t) blocks);
    for (int b = 0; b < blocks; ++b) {
        built->block_runners.push_back(lingbot_get_cuda_self_attn_block_runner(
            model, bw, (size_t) b, seq, text_seq, bf16_boundary));
    }
    LingBotCudaSelfAttnForwardRunner * runner = built.get();
    model.cuda_self_attn_forward_runners.push_back(std::move(built));
    std::printf("vla(lingbot_va): cached official CUDA self-attn forward runner blocks=%d seq=%lld text_seq=%lld bf16_boundary=%d\n",
                blocks, (long long) seq, (long long) text_seq, bf16_boundary ? 1 : 0);
    return runner;
}

LingBotCudaSelfAttnPostQkvGraph * lingbot_get_cuda_self_attn_post_qkv_graph(
        const LingBotVAModelArch & model,
        LingBotRuntimeWeights & bw,
        size_t block_index,
        int64_t seq,
        int64_t text_seq,
        bool bf16_boundary,
        int text_slot,
        bool include_next_qkv = true,
        bool include_output = false,
        bool action_mode = false,
        LingBotCudaSelfAttnBlockRunner * block_runner = nullptr) {
    if (include_output) include_next_qkv = false;
    if (include_next_qkv && block_index + 1 >= bw.blocks.size()) return nullptr;
    if (block_index >= bw.blocks.size()) return nullptr;
    if (block_runner) {
        LingBotCudaSelfAttnPostQkvGraph * cached =
            (text_slot >= 0) ? block_runner->post_qkv_next_by_branch[text_slot] : block_runner->post_qkv_next;
        if (cached && cached->matches(bw, block_index, seq, text_seq, bf16_boundary,
                                      text_slot, include_next_qkv, include_output, action_mode)) {
            return cached;
        }
    }
    for (const auto & entry : model.cuda_self_attn_post_qkv_graphs) {
        if (entry && entry->matches(bw, block_index, seq, text_seq, bf16_boundary,
                                    text_slot, include_next_qkv, include_output, action_mode)) {
            if (block_runner) {
                if (text_slot >= 0) block_runner->post_qkv_next_by_branch[text_slot] = entry.get();
                else block_runner->post_qkv_next = entry.get();
            }
            return entry.get();
        }
    }

    const LingBotVAModelArch & m = model;
    const int64_t hidden = m.cfg.hidden;
    const LingBotBlockW & b = bw.blocks[block_index];
    const LingBotBlockW * nb = include_next_qkv ? &bw.blocks[block_index + 1] : nullptr;
    auto built = std::make_unique<LingBotCudaSelfAttnPostQkvGraph>();
    ggml_init_params cp = { size_t(192) * 1024 * 1024, nullptr, true };
    built->ctx = ggml_init(cp);
    if (!built->ctx) return nullptr;
    ggml_context * C = built->ctx;

    built->weights = &bw;
    built->block_index = block_index;
    built->seq = seq;
    built->text_seq = text_seq;
    built->text_slot = text_slot;
    built->bf16_boundary = bf16_boundary;
    built->include_next_qkv = include_next_qkv;
    built->include_output = include_output;
    built->action_mode = action_mode;
    built->x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, seq);
    built->text = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, text_seq);
    built->timestep_proj = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden * 6, seq);
    built->self_ctx = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, seq);
    built->rope_cos = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, m.head_dim / 2, 1, seq);
    built->rope_sin = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, m.head_dim / 2, 1, seq);
    if (include_output) {
        built->output_t_hidden = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, seq);
        ggml_set_input(built->output_t_hidden);
    }
    ggml_set_input(built->x_in);
    ggml_set_input(built->text);
    ggml_set_input(built->timestep_proj);
    ggml_set_input(built->self_ctx);
    ggml_set_input(built->rope_cos);
    ggml_set_input(built->rope_sin);

    auto bf16_boundary_fn = [&](ggml_tensor * t) -> ggml_tensor * {
        return bf16_boundary ? ggml_cast(C, ggml_cast(C, t, GGML_TYPE_BF16), GGML_TYPE_F32) : t;
    };

    ggml_tensor * gate_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, seq, 2),
                                      ggml_view_1d(C, b.scale_shift, hidden,
                                                   (size_t) 2 * (size_t) hidden * ggml_element_size(b.scale_shift)));
    ggml_tensor * c_shift_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, seq, 3),
                                         ggml_view_1d(C, b.scale_shift, hidden,
                                                      (size_t) 3 * (size_t) hidden * ggml_element_size(b.scale_shift)));
    ggml_tensor * c_scale_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, seq, 4),
                                         ggml_view_1d(C, b.scale_shift, hidden,
                                                      (size_t) 4 * (size_t) hidden * ggml_element_size(b.scale_shift)));
    ggml_tensor * c_gate_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, seq, 5),
                                        ggml_view_1d(C, b.scale_shift, hidden,
                                                     (size_t) 5 * (size_t) hidden * ggml_element_size(b.scale_shift)));

    ggml_tensor * a1 = lin(C, b.self_attn.o, built->self_ctx);
    ggml_tensor * x = ggml_add(C, built->x_in, ggml_mul(C, a1, gate_msa));
    x = bf16_boundary_fn(x);
    ggml_tensor * n2 = ggml_add(C, ggml_mul(C, ggml_norm(C, x, 1e-6f), b.cross_norm_weight), b.cross_norm_bias);
    n2 = bf16_boundary_fn(n2);
    ggml_tensor * a2 = build_attention_shape(C, b.cross_attn, n2, built->text, nullptr, nullptr,
                                             m, seq, text_seq, nullptr);
    a2 = bf16_boundary_fn(a2);
    x = ggml_add(C, x, a2);
    x = bf16_boundary_fn(x);
    ggml_tensor * n3 = adaln(C, x, c_shift_msa, c_scale_msa, 1e-6f);
    n3 = bf16_boundary_fn(n3);
    ggml_tensor * ff = lin(C, b.ffn_down, ggml_gelu(C, lin(C, b.ffn_up, n3)));
    ff = bf16_boundary_fn(ff);
    built->post_out = ggml_add(C, x, ggml_mul(C, ff, c_gate_msa));
    built->post_out = bf16_boundary_fn(built->post_out);
    // Match the old graph boundary: post graph output is materialized, then
    // copied into the next block's QKV input tensor.
    built->post_out = ggml_cont(C, built->post_out);

    const bool include_next =
        include_next_qkv &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FUSED_POST_QKV_POST_ONLY_DEBUG");
    if (include_next) {
    ggml_tensor * next_shift_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, seq, 0),
                                            ggml_view_1d(C, nb->scale_shift, hidden, 0));
    ggml_tensor * next_scale_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, seq, 1),
                                            ggml_view_1d(C, nb->scale_shift, hidden,
                                                         (size_t) hidden * ggml_element_size(nb->scale_shift)));
    built->next_n1 = adaln(C, built->post_out, next_shift_msa, next_scale_msa, 1e-6f);
    LingBotAttentionTrace next_trace;
    (void) build_attention_shape(C, nb->self_attn, built->next_n1, built->next_n1,
                                 built->rope_cos, built->rope_sin, m, seq, seq, &next_trace);
    built->next_q = next_trace.q;
    built->next_k = next_trace.k;
    built->next_v = next_trace.v;
    built->next_qh = next_trace.qh;
    built->next_kh = next_trace.kh;
    built->next_vh = next_trace.vh;
    }
    if (include_output) {
        ggml_tensor * out_shift = ggml_add(C, built->output_t_hidden,
                                           ggml_view_1d(C, bw.output_scale_shift, hidden, 0));
        ggml_tensor * out_scale = ggml_add(C, built->output_t_hidden,
                                           ggml_view_1d(C, bw.output_scale_shift, hidden,
                                                        (size_t) hidden * ggml_element_size(bw.output_scale_shift)));
        built->output_out = action_mode
            ? lin(C, bw.action_out, adaln(C, built->post_out, out_shift, out_scale, 1e-6f))
            : lin(C, bw.output_proj, adaln(C, built->post_out, out_shift, out_scale, 1e-6f));
    }
    ggml_set_output(built->post_out);
    if (include_next) {
        ggml_set_output(built->next_n1);
        ggml_set_output(built->next_q);
        ggml_set_output(built->next_k);
        ggml_set_output(built->next_v);
        ggml_set_output(built->next_qh);
        ggml_set_output(built->next_kh);
        ggml_set_output(built->next_vh);
    }
    if (include_output) {
        ggml_set_output(built->output_out);
    }

    built->graph = ggml_new_graph_custom(C, 65536, false);
    ggml_build_forward_expand(built->graph, built->post_out);
    if (include_next) {
        ggml_build_forward_expand(built->graph, built->next_n1);
        ggml_build_forward_expand(built->graph, built->next_q);
        ggml_build_forward_expand(built->graph, built->next_k);
        ggml_build_forward_expand(built->graph, built->next_v);
        ggml_build_forward_expand(built->graph, built->next_qh);
        ggml_build_forward_expand(built->graph, built->next_kh);
        ggml_build_forward_expand(built->graph, built->next_vh);
    }
    if (include_output) {
        ggml_build_forward_expand(built->graph, built->output_out);
    }
    built->galloc = lingbot_runtime_gallocr(bw.backend);
    if (!built->galloc || !ggml_gallocr_alloc_graph(built->galloc, built->graph)) {
        return nullptr;
    }

    LingBotCudaSelfAttnPostQkvGraph * graph = built.get();
    model.cuda_self_attn_post_qkv_graphs.push_back(std::move(built));
    if (block_runner) {
        if (text_slot >= 0) block_runner->post_qkv_next_by_branch[text_slot] = graph;
        else block_runner->post_qkv_next = graph;
    }
    std::printf("vla(lingbot_va): cached official CUDA self-attn %s graph "
                "block=%zu next=%zu seq=%lld text_seq=%lld text_slot=%d bf16_boundary=%d action=%d\n",
                include_output ? "post+output" : (include_next ? "post+next-QKV" : "post-only"),
                block_index + 1,
                include_next ? block_index + 2 : 0,
                (long long) seq,
                (long long) text_seq,
                text_slot,
                bf16_boundary ? 1 : 0,
                action_mode ? 1 : 0);
    return graph;
}

LingBotCudaSelfAttnQkvGraph * lingbot_get_or_build_cuda_self_attn_qkv_graph(
        const LingBotVAModelArch & model,
        LingBotRuntimeWeights & bw,
        size_t block_index,
        int64_t seq,
        LingBotCudaSelfAttnBlockRunner * block_runner = nullptr) {
    if (block_runner && block_runner->qkv &&
        block_runner->qkv->matches(bw, block_index, seq)) {
        return block_runner->qkv;
    }
    for (const auto & entry : model.cuda_self_attn_qkv_graphs) {
        if (entry && entry->matches(bw, block_index, seq)) {
            if (block_runner) block_runner->qkv = entry.get();
            return entry.get();
        }
    }

    const LingBotVAModelArch & m = model;
    const int64_t hidden = m.cfg.hidden;
    const LingBotBlockW & b = bw.blocks[block_index];
    auto built = std::make_unique<LingBotCudaSelfAttnQkvGraph>();
    ggml_init_params cp = { size_t(96) * 1024 * 1024, nullptr, true };
    built->ctx = ggml_init(cp);
    if (!built->ctx) return nullptr;
    ggml_context * C = built->ctx;

    built->weights = &bw;
    built->block_index = block_index;
    built->seq = seq;
    built->x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, seq);
    built->timestep_proj = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden * 6, seq);
    built->rope_cos = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, m.head_dim / 2, 1, seq);
    built->rope_sin = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, m.head_dim / 2, 1, seq);
    ggml_set_input(built->x_in);
    ggml_set_input(built->timestep_proj);
    ggml_set_input(built->rope_cos);
    ggml_set_input(built->rope_sin);

    ggml_tensor * shift_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, seq, 0),
                                       ggml_view_1d(C, b.scale_shift, hidden, 0));
    ggml_tensor * scale_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, seq, 1),
                                       ggml_view_1d(C, b.scale_shift, hidden,
                                                    (size_t) hidden * ggml_element_size(b.scale_shift)));
    built->n1 = adaln(C, built->x_in, shift_msa, scale_msa, 1e-6f);
    LingBotAttentionTrace trace;
    (void) build_attention_shape(C, b.self_attn, built->n1, built->n1,
                                 built->rope_cos, built->rope_sin, m, seq, seq, &trace);
    built->q = trace.q;
    built->k = trace.k;
    built->v = trace.v;
    built->qh = trace.qh;
    built->kh = trace.kh;
    built->vh = trace.vh;
    ggml_set_output(built->n1);
    ggml_set_output(built->q);
    ggml_set_output(built->k);
    ggml_set_output(built->v);
    ggml_set_output(built->qh);
    ggml_set_output(built->kh);
    ggml_set_output(built->vh);

    built->graph = ggml_new_graph_custom(C, 32768, false);
    ggml_build_forward_expand(built->graph, built->n1);
    ggml_build_forward_expand(built->graph, built->q);
    ggml_build_forward_expand(built->graph, built->k);
    ggml_build_forward_expand(built->graph, built->v);
    ggml_build_forward_expand(built->graph, built->qh);
    ggml_build_forward_expand(built->graph, built->kh);
    ggml_build_forward_expand(built->graph, built->vh);
    built->galloc = lingbot_runtime_gallocr(bw.backend);
    if (!built->galloc || !ggml_gallocr_alloc_graph(built->galloc, built->graph)) {
        return nullptr;
    }

    LingBotCudaSelfAttnQkvGraph * graph = built.get();
    model.cuda_self_attn_qkv_graphs.push_back(std::move(built));
    if (block_runner) block_runner->qkv = graph;
    std::printf("vla(lingbot_va): cached official CUDA self-attn QKV graph block=%zu seq=%lld\n",
                block_index + 1, (long long) seq);
    return graph;
}

bool lingbot_copy_tensor_d2d_checked(
        ggml_tensor * dst,
        const ggml_tensor * src,
        const char * label) {
    if (!dst || !dst->data || !src || !src->data) return false;
    const size_t dst_bytes = ggml_nbytes(dst);
    const size_t src_bytes = ggml_nbytes(src);
    if (dst_bytes != src_bytes) {
        std::fprintf(stderr,
                     "vla(lingbot_va): D2D tensor size mismatch for %s: src=%zu dst=%zu\n",
                     label ? label : "tensor", src_bytes, dst_bytes);
        return false;
    }
    const cudaError_t err = cudaMemcpy(dst->data, src->data, dst_bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "vla(lingbot_va): cudaMemcpy(%s D2D) failed: %s\n",
                     label ? label : "tensor", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool lingbot_compute_cuda_self_attn_qkv_graph(
        LingBotRuntimeWeights & bw,
        LingBotExecState & state,
        LingBotCudaSelfAttnQkvGraph * qkv_graph,
        const ggml_tensor * x_device_source = nullptr) {
    if (!qkv_graph) return false;
    const int64_t hidden = qkv_graph->x_in ? qkv_graph->x_in->ne[0] : 0;
    if (x_device_source) {
        if (!lingbot_copy_tensor_d2d_checked(qkv_graph->x_in, x_device_source, "qkv.x_in")) {
            return false;
        }
    } else if (state.allow_device_x_handoff &&
               hidden > 0 &&
               lingbot_device_x_valid(state, hidden, bw.backend)) {
        if (!lingbot_copy_device_x_to_tensor(state, qkv_graph->x_in, "qkv.x_in")) {
            return false;
        }
    } else {
        ggml_backend_tensor_set(qkv_graph->x_in, state.x.data(), 0, state.x.size() * sizeof(float));
        if (state.allow_device_x_handoff) {
            ggml_backend_synchronize(bw.backend);
            if (!lingbot_copy_tensor_to_device_x(state, bw.backend, qkv_graph->x_in, "qkv.x_in")) {
                return false;
            }
        }
    }

    bool timestep_copied_d2d = false;
    if (!lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_QKV_TIMESTEP_D2D_DISABLE") &&
        lingbot_device_timestep_proj_valid(state, ggml_nbytes(qkv_graph->timestep_proj), bw.backend)) {
        if (!lingbot_copy_device_timestep_proj_to_tensor(state, qkv_graph->timestep_proj,
                                                         "qkv.timestep_proj")) {
            return false;
        }
        timestep_copied_d2d = true;
    }
    if (!timestep_copied_d2d) {
        ggml_backend_tensor_set(qkv_graph->timestep_proj, state.timestep_proj.data(), 0,
                                state.timestep_proj.size() * sizeof(float));
    }
    const bool rope_cache_enabled =
        lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_ROPE_CACHE") &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_ROPE_CACHE_DISABLE") &&
        state.rope_signature != 0;
    if (!rope_cache_enabled || qkv_graph->uploaded_rope_signature != state.rope_signature) {
        ggml_backend_tensor_set(qkv_graph->rope_cos, state.rope_cos.data(), 0, state.rope_cos.size() * sizeof(float));
        ggml_backend_tensor_set(qkv_graph->rope_sin, state.rope_sin.data(), 0, state.rope_sin.size() * sizeof(float));
        qkv_graph->uploaded_rope_signature = rope_cache_enabled ? state.rope_signature : 0;
    }
    const ggml_status st = ggml_backend_graph_compute(bw.backend, qkv_graph->graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA self-attn QKV graph failed (%d)\n", (int) st);
        return false;
    }
    return true;
}

bool lingbot_official_post_qkv_fusion_enabled() {
    return (lingbot_official_fast_path_enabled() ||
            lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FUSED_POST_QKV")) &&
           !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FUSED_POST_QKV_DISABLE");
}

bool lingbot_official_final_post_output_fusion_enabled() {
    return (lingbot_official_fast_path_enabled() ||
            lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FUSED_FINAL_POST_OUTPUT")) &&
           !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FUSED_FINAL_POST_OUTPUT_DISABLE");
}

bool lingbot_official_post_text_d2d_enabled() {
    return (lingbot_official_fast_path_enabled() ||
            lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_POST_TEXT_D2D")) &&
           !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_POST_TEXT_D2D_DISABLE");
}

bool lingbot_runtime_kv_self_ctx_event_handoff_enabled() {
    return (lingbot_official_fast_path_enabled() ||
            lingbot_env_enabled("VLA_LINGBOT_RUNTIME_KV_SELF_CTX_EVENT_HANDOFF")) &&
           !lingbot_env_enabled("VLA_LINGBOT_RUNTIME_KV_SELF_CTX_EVENT_HANDOFF_DISABLE");
}

bool lingbot_stage5_workspace_enabled() {
    return (lingbot_official_fast_path_enabled() ||
            lingbot_env_enabled("VLA_LINGBOT_STAGE5_WORKSPACE")) &&
           !lingbot_env_enabled("VLA_LINGBOT_STAGE5_WORKSPACE_DISABLE");
}

bool lingbot_runtime_kv_self_ctx_from_qkv_tensors(
        LingBotRuntimeWeights & bw,
        const LingBotVAModelArch & m,
        LingBotExecState & state,
        const LingBotBlockSparseTable & self_table,
        const std::vector<unsigned char> & token_mask,
        ggml_tensor * qh,
        ggml_tensor * kh,
        ggml_tensor * vh,
        LingBotRuntimeKVDeviceResult & self_ctx_device) {
    std::vector<float> unused_context;
    ggml_backend_synchronize(bw.backend);
    if (!real_qkv_to_cuda_context(bw.backend, qh, kh, vh, self_table, token_mask,
                                  m, state.seq, unused_context, nullptr,
                                  state.cache_session_id,
                                  state.cache_block_index,
                                  state.cache_mode,
                                  state.cache_branch,
                                  nullptr, nullptr, nullptr,
                                  &self_ctx_device)) {
        return false;
    }
    if (!self_ctx_device.valid ||
        self_ctx_device.seq != state.seq ||
        self_ctx_device.heads != m.n_heads ||
        self_ctx_device.head_dim != m.head_dim) {
        std::fprintf(stderr,
                     "vla(lingbot_va): fused post+QKV requires device self_ctx block=%lld seq=%lld\n",
                     (long long) state.cache_block_index + 1,
                     (long long) state.seq);
        return false;
    }
    return true;
}

bool lingbot_copy_runtime_self_ctx_to_tensor(
        ggml_tensor * dst,
        const LingBotRuntimeKVDeviceResult & src,
        const char * label) {
    if (!dst || !dst->data || !src.valid || !src.data || src.bytes != ggml_nbytes(dst)) {
        std::fprintf(stderr, "vla(lingbot_va): %s missing compatible device self_ctx\n",
                     label ? label : "self_ctx");
        return false;
    }
    const bool event_handoff =
        lingbot_runtime_kv_self_ctx_event_handoff_enabled() &&
        src.ready_event != nullptr;
    if (event_handoff) {
        const cudaError_t wait_err = cudaStreamWaitEvent(cudaStreamPerThread, src.ready_event, 0);
        if (wait_err != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): cudaStreamWaitEvent(%s) failed: %s\n",
                         label ? label : "self_ctx", cudaGetErrorString(wait_err));
            return false;
        }
        const cudaError_t copy_err = cudaMemcpyAsync(dst->data, src.data, src.bytes,
                                                     cudaMemcpyDeviceToDevice,
                                                     cudaStreamPerThread);
        if (copy_err != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): cudaMemcpyAsync(%s D2D) failed: %s\n",
                         label ? label : "self_ctx", cudaGetErrorString(copy_err));
            return false;
        }
        const cudaError_t sync_err = cudaStreamSynchronize(cudaStreamPerThread);
        if (sync_err != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): cudaStreamSynchronize(%s D2D) failed: %s\n",
                         label ? label : "self_ctx", cudaGetErrorString(sync_err));
            return false;
        }
        return true;
    }

    const cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): cudaDeviceSynchronize(%s source) failed: %s\n",
                     label ? label : "self_ctx", cudaGetErrorString(sync_err));
        return false;
    }
    const cudaError_t copy_err = cudaMemcpy(dst->data, src.data, src.bytes, cudaMemcpyDeviceToDevice);
    if (copy_err != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): cudaMemcpy(%s D2D) failed: %s\n",
                     label ? label : "self_ctx", cudaGetErrorString(copy_err));
        return false;
    }
    const bool warp_self_ctx_handoff =
        lingbot_env_enabled("VLA_LINGBOT_RUNTIME_KV_WARP_ATTN") &&
        !lingbot_env_enabled("VLA_LINGBOT_RUNTIME_KV_WARP_ATTN_DISABLE");
    if (warp_self_ctx_handoff) {
        const cudaError_t dst_sync_err = cudaStreamSynchronize(cudaStreamPerThread);
        if (dst_sync_err != cudaSuccess) {
            std::fprintf(stderr, "vla(lingbot_va): cudaStreamSynchronize(%s destination) failed: %s\n",
                         label ? label : "self_ctx", cudaGetErrorString(dst_sync_err));
            return false;
        }
    }
    return true;
}

bool lingbot_compute_cuda_self_attn_post_qkv_graph(
        LingBotRuntimeWeights & bw,
        LingBotExecState & state,
        LingBotCudaSelfAttnPostQkvGraph * graph,
        const ggml_tensor * x_device_source,
        const LingBotRuntimeKVDeviceResult & self_ctx_device) {
    if (!graph) return false;
    if (x_device_source) {
        if (!lingbot_copy_tensor_d2d_checked(graph->x_in, x_device_source, "post_qkv.x_in")) {
            return false;
        }
    } else {
        const int64_t hidden = graph->x_in ? graph->x_in->ne[0] : 0;
        if (hidden > 0 && lingbot_device_x_valid(state, hidden, bw.backend)) {
            if (!lingbot_copy_device_x_to_tensor(state, graph->x_in, "post_qkv.x_in")) {
                return false;
            }
        } else {
            ggml_backend_tensor_set(graph->x_in, state.x.data(), 0, state.x.size() * sizeof(float));
        }
    }

    const bool static_text_cache_enabled =
        lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_TEXT_CACHE") &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_TEXT_CACHE_DISABLE") &&
        state.text_signature != 0 &&
        (state.cache_branch == 0 || state.cache_branch == 1);
    bool text_copied_d2d = false;
    if (lingbot_official_post_text_d2d_enabled() &&
        lingbot_device_text_valid(state, ggml_nbytes(graph->text), bw.backend)) {
        if (!lingbot_copy_device_text_to_tensor(state, graph->text, "post_qkv.text")) {
            return false;
        }
        text_copied_d2d = true;
        graph->uploaded_text_signature = 0;
    }
    if (!text_copied_d2d &&
        (!static_text_cache_enabled || graph->uploaded_text_signature != state.text_signature)) {
        ggml_backend_tensor_set(graph->text, state.text.data(), 0, state.text.size() * sizeof(float));
        graph->uploaded_text_signature = static_text_cache_enabled ? state.text_signature : 0;
    }

    bool timestep_copied_d2d = false;
    if (!lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_POST_TIMESTEP_D2D_DISABLE") &&
        lingbot_device_timestep_proj_valid(state, ggml_nbytes(graph->timestep_proj), bw.backend)) {
        if (!lingbot_copy_device_timestep_proj_to_tensor(state, graph->timestep_proj,
                                                         "post_qkv.timestep_proj")) {
            return false;
        }
        timestep_copied_d2d = true;
    }
    if (!timestep_copied_d2d) {
        ggml_backend_tensor_set(graph->timestep_proj, state.timestep_proj.data(), 0,
                                state.timestep_proj.size() * sizeof(float));
    }
    if (graph->include_output && graph->output_t_hidden) {
        ggml_backend_tensor_set(graph->output_t_hidden, state.t_hidden.data(), 0,
                                state.t_hidden.size() * sizeof(float));
    }

    const bool rope_cache_enabled =
        lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_ROPE_CACHE") &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_ROPE_CACHE_DISABLE") &&
        state.rope_signature != 0;
    if (graph->rope_cos && graph->rope_cos->data &&
        graph->rope_sin && graph->rope_sin->data &&
        (!rope_cache_enabled || graph->uploaded_rope_signature != state.rope_signature)) {
        ggml_backend_tensor_set(graph->rope_cos, state.rope_cos.data(), 0, state.rope_cos.size() * sizeof(float));
        ggml_backend_tensor_set(graph->rope_sin, state.rope_sin.data(), 0, state.rope_sin.size() * sizeof(float));
        graph->uploaded_rope_signature = rope_cache_enabled ? state.rope_signature : 0;
    }

    if (!self_ctx_device.valid ||
        !self_ctx_device.data ||
        self_ctx_device.bytes != ggml_nbytes(graph->self_ctx)) {
        std::fprintf(stderr,
                     "vla(lingbot_va): fused post+QKV missing compatible device self_ctx\n");
        return false;
    }
    if (!lingbot_copy_runtime_self_ctx_to_tensor(graph->self_ctx, self_ctx_device,
                                                 "fused post+QKV self_ctx")) {
        return false;
    }

    const ggml_status st = ggml_backend_graph_compute(bw.backend, graph->graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA self-attn fused post+QKV graph failed (%d)\n", (int) st);
        return false;
    }
    return true;
}

LingBotOfficialRuntimeKvDenoiseRunner * lingbot_get_official_runtime_kv_denoise_runner(
        const LingBotVAModelArch & model,
        LingBotRuntimeWeights & bw,
        int blocks,
        int64_t text_seq,
        int64_t latent_seq,
        int64_t action_seq,
        bool bf16_boundary) {
    for (const auto & entry : model.official_runtime_kv_denoise_runners) {
        if (entry && entry->matches(bw, blocks, text_seq, latent_seq, action_seq, bf16_boundary)) {
            return entry.get();
        }
    }
    auto built = std::make_unique<LingBotOfficialRuntimeKvDenoiseRunner>();
    built->weights = &bw;
    built->blocks = blocks;
    built->text_seq = text_seq;
    built->latent_seq = latent_seq;
    built->action_seq = action_seq;
    built->bf16_boundary = bf16_boundary;
    built->latent_forward = lingbot_get_cuda_self_attn_forward_runner(
        model, bw, blocks, latent_seq, text_seq, bf16_boundary);
    built->action_forward = lingbot_get_cuda_self_attn_forward_runner(
        model, bw, blocks, action_seq, text_seq, bf16_boundary);
    if (!built->latent_forward || !built->action_forward) {
        return nullptr;
    }
    for (int mode = 0; mode < 3; ++mode) {
        built->latent_forward_by_mode[mode] = built->latent_forward;
        built->action_forward_by_mode[mode] = built->action_forward;
    }
    LingBotOfficialRuntimeKvDenoiseRunner * runner = built.get();
    model.official_runtime_kv_denoise_runners.push_back(std::move(built));
    std::printf("vla(lingbot_va): cached official runtime-KV denoise runner "
                "blocks=%d text_seq=%lld latent_seq=%lld action_seq=%lld bf16_boundary=%d cache_modes=0/1/2\n",
                blocks,
                (long long) text_seq,
                (long long) latent_seq,
                (long long) action_seq,
                bf16_boundary ? 1 : 0);
    return runner;
}

struct LingBotOutputOneGraph {
    LingBotRuntimeWeights * weights = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * x_in = nullptr;
    ggml_tensor * t_hidden = nullptr;
    ggml_tensor * out = nullptr;
    int64_t seq = 0;
    bool action_mode = false;

    bool matches(
            LingBotRuntimeWeights & bw,
            bool action_mode_,
            int64_t seq_) const {
        return weights == &bw &&
               action_mode == action_mode_ &&
               seq == seq_;
    }

    void reset() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
        weights = nullptr;
        ctx = nullptr;
        graph = nullptr;
        galloc = nullptr;
        x_in = nullptr;
        t_hidden = nullptr;
        out = nullptr;
        seq = 0;
        action_mode = false;
    }

    ~LingBotOutputOneGraph() {
        reset();
    }
};

struct LingBotEmbeddingStageGraph {
    LingBotRuntimeWeights * weights = nullptr;
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_tensor * x_in = nullptr;
    ggml_tensor * text_raw = nullptr;
    ggml_tensor * time_raw = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * text = nullptr;
    ggml_tensor * t_hidden = nullptr;
    ggml_tensor * timestep_proj = nullptr;
    int64_t input_dim = 0;
    int64_t seq = 0;
    int64_t text_seq = 0;
    bool action_mode = false;

    bool matches(
            LingBotRuntimeWeights & bw,
            bool action_mode_,
            int64_t input_dim_,
            int64_t seq_,
            int64_t text_seq_) const {
        return weights == &bw &&
               action_mode == action_mode_ &&
               input_dim == input_dim_ &&
               seq == seq_ &&
               text_seq == text_seq_;
    }

    void reset() {
        if (galloc) ggml_gallocr_free(galloc);
        if (ctx) ggml_free(ctx);
        weights = nullptr;
        ctx = nullptr;
        graph = nullptr;
        galloc = nullptr;
        x_in = nullptr;
        text_raw = nullptr;
        time_raw = nullptr;
        x = nullptr;
        text = nullptr;
        t_hidden = nullptr;
        timestep_proj = nullptr;
        input_dim = 0;
        seq = 0;
        text_seq = 0;
        action_mode = false;
    }

    ~LingBotEmbeddingStageGraph() {
        reset();
    }
};
#endif

struct LingBotTransformerExecutor {
    const LingBotVAModelArch & m;
    LingBotRuntimeWeights & common;
    std::string dump_dir;
    int cache_branch = 0;
    std::vector<float> text_raw_override;
    int64_t text_raw_seq = 0;
    std::unordered_map<std::string, std::vector<float>> timestep_embedding_cache;
    std::vector<std::unique_ptr<LingBotBlockSequenceGraph>> block_sequence_graphs;
};

// Transformer execution skeleton.  The execution path can feed synthetic
// inputs and F32 debug weights, but the stages are split to match the future
// predict() flow: embeddings -> adaptive block windows -> output heads.
bool compute_graph(ggml_backend_t backend, ggml_context * C, ggml_cgraph * gf) {
    ggml_gallocr_t galloc = lingbot_runtime_gallocr(backend);
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "vla(lingbot_va): streaming graph allocation failed\n");
        if (galloc) ggml_gallocr_free(galloc);
        return false;
    }
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    ggml_gallocr_free(galloc);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(lingbot_va): streaming graph compute failed (%d)\n", (int) st);
        return false;
    }
    (void) C;
    return true;
}

bool exec_embedding_stage(
        LingBotTransformerExecutor & ex,
        bool action_mode,
        LingBotExecState & out,
        const std::vector<float> * raw_input = nullptr,
        double timestep = 0.0,
        const LingBotGridSpec * grid_spec = nullptr,
        const std::vector<double> * token_timesteps = nullptr,
        const LingBotDeviceTensor * device_raw_input = nullptr) {
    const LingBotVAModelArch & m = ex.m;
    LingBotRuntimeWeights & sw = ex.common;
    const int64_t input_dim = action_mode
        ? m.action_dim
        : m.in_channels * m.patch_t * m.patch_h * m.patch_w;
    int64_t seq = 2;
    if (raw_input) {
        if (input_dim <= 0 || raw_input->size() % (size_t) input_dim != 0) {
            std::fprintf(stderr, "vla(lingbot_va): invalid %s raw input size=%zu input_dim=%lld\n",
                         action_mode ? "action" : "latent",
                         raw_input->size(), (long long) input_dim);
            return false;
        }
        seq = (int64_t) (raw_input->size() / (size_t) input_dim);
        if (seq <= 0) {
            std::fprintf(stderr, "vla(lingbot_va): invalid %s raw input seq=%lld\n",
                         action_mode ? "action" : "latent", (long long) seq);
            return false;
        }
    }
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    else if (device_raw_input) {
        if (input_dim <= 0 || device_raw_input->count % (size_t) input_dim != 0) {
            std::fprintf(stderr, "vla(lingbot_va): invalid %s device raw input count=%zu input_dim=%lld\n",
                         action_mode ? "action" : "latent",
                         device_raw_input->count, (long long) input_dim);
            return false;
        }
        seq = (int64_t) (device_raw_input->count / (size_t) input_dim);
        if (seq <= 0) {
            std::fprintf(stderr, "vla(lingbot_va): invalid %s device raw input seq=%lld\n",
                         action_mode ? "action" : "latent", (long long) seq);
            return false;
        }
    }
#endif
    const LingBotGridSpec resolved_grid = grid_spec ? *grid_spec : lingbot_default_grid_spec(seq, action_mode);
    if (resolved_grid.seq() != seq) {
        std::fprintf(stderr, "vla(lingbot_va): %s grid seq mismatch (%lld vs %lld)\n",
                     action_mode ? "action" : "latent",
                     (long long) resolved_grid.seq(), (long long) seq);
        return false;
    }
    int64_t text_seq = ex.text_raw_seq > 0 ? ex.text_raw_seq : 2;
    if (text_seq <= 0 ||
        (!ex.text_raw_override.empty() &&
         ex.text_raw_override.size() != (size_t) m.text_dim * (size_t) text_seq)) {
        std::fprintf(stderr, "vla(lingbot_va): invalid text override size=%zu text_dim=%lld text_seq=%lld\n",
                     ex.text_raw_override.size(), (long long) m.text_dim, (long long) text_seq);
        return false;
    }
    out.seq = seq;
    out.text_seq = text_seq;
    auto fill_time_raw_cached = [&]() -> std::vector<float> {
        const bool state_f32 = lingbot_env_enabled("VLA_LINGBOT_STATE_F32");
        std::string key = state_f32 ? "f32|" : "bf16|";
        key += std::to_string((long long) seq);
        key += "|";
        char buf[64];
        if (token_timesteps && token_timesteps->size() == (size_t) seq) {
            key += "tok";
            for (double tv : *token_timesteps) {
                std::snprintf(buf, sizeof(buf), "|%.17g", tv);
                key += buf;
            }
        } else {
            key += "one";
            std::snprintf(buf, sizeof(buf), "|%.17g", timestep);
            key += buf;
        }
        const auto it = ex.timestep_embedding_cache.find(key);
        if (it != ex.timestep_embedding_cache.end()) {
            return it->second;
        }
        std::vector<float> timh((size_t) 256 * (size_t) seq);
        if (token_timesteps && token_timesteps->size() == (size_t) seq) {
            std::vector<float> one_time((size_t) 256);
            for (int64_t i = 0; i < seq; ++i) {
                fill_timestep_embedding(one_time, 256, 1, (*token_timesteps)[(size_t) i]);
                std::copy(one_time.begin(), one_time.end(), timh.begin() + (size_t) i * 256);
            }
        } else {
            if (token_timesteps) {
                std::fprintf(stderr,
                             "vla(lingbot_va): ignoring token timestep vector size=%zu for seq=%lld\n",
                             token_timesteps->size(), (long long) seq);
            }
            fill_timestep_embedding(timh, 256, seq, timestep);
        }
        if (!state_f32) {
            lingbot_bf16_roundtrip_inplace(timh);
        }
        if (ex.timestep_embedding_cache.size() >= 256) {
            ex.timestep_embedding_cache.erase(ex.timestep_embedding_cache.begin());
        }
        ex.timestep_embedding_cache.emplace(std::move(key), timh);
        return timh;
    };

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (!lingbot_env_enabled("VLA_LINGBOT_EMBEDDING_STAGE_GRAPH_CACHE_DISABLE")) {
        LingBotEmbeddingStageGraph * cached = nullptr;
        for (const auto & entry : ex.m.embedding_stage_graphs) {
            if (entry && entry->matches(sw, action_mode, input_dim, seq, text_seq)) {
                cached = entry.get();
                break;
            }
        }
        if (!cached) {
            auto built = std::make_unique<LingBotEmbeddingStageGraph>();
            ggml_init_params cp = { size_t(32) * 1024 * 1024, nullptr, true };
            built->ctx = ggml_init(cp);
            if (!built->ctx) return false;
            ggml_context * C = built->ctx;

            built->weights = &sw;
            built->input_dim = input_dim;
            built->seq = seq;
            built->text_seq = text_seq;
            built->action_mode = action_mode;
            built->x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, input_dim, seq);
            built->text_raw = ggml_new_tensor_2d(C, GGML_TYPE_F32, m.text_dim, text_seq);
            built->time_raw = ggml_new_tensor_2d(C, GGML_TYPE_F32, 256, seq);
            ggml_set_input(built->x_in);
            ggml_set_input(built->text_raw);
            ggml_set_input(built->time_raw);

            built->x = action_mode ? lin(C, sw.action_embd, built->x_in)
                                   : lin(C, sw.patch_embd_mlp, built->x_in);
            built->text = lin(C, sw.cond.text_l2,
                              ggml_gelu(C, lin(C, sw.cond.text_l1, built->text_raw)));
            const LingBotConditionW & cond = action_mode ? sw.action_cond : sw.cond;
            built->t_hidden = lin(C, cond.time_l2,
                                  ggml_silu(C, lin(C, cond.time_l1, built->time_raw)));
            built->timestep_proj = lin(C, cond.time_proj, ggml_silu(C, built->t_hidden));
            ggml_set_output(built->x);
            ggml_set_output(built->text);
            ggml_set_output(built->t_hidden);
            ggml_set_output(built->timestep_proj);

            built->graph = ggml_new_graph_custom(C, 8192, false);
            ggml_build_forward_expand(built->graph, built->x);
            ggml_build_forward_expand(built->graph, built->text);
            ggml_build_forward_expand(built->graph, built->t_hidden);
            ggml_build_forward_expand(built->graph, built->timestep_proj);
            built->galloc = lingbot_runtime_gallocr(sw.backend);
            if (!built->galloc || !ggml_gallocr_alloc_graph(built->galloc, built->graph)) {
                return false;
            }
            cached = built.get();
            ex.m.embedding_stage_graphs.push_back(std::move(built));
            std::printf("vla(lingbot_va): cached embedding-stage graph action=%d seq=%lld text_seq=%lld input_dim=%lld\n",
                        action_mode ? 1 : 0, (long long) seq, (long long) text_seq, (long long) input_dim);
        }

        std::vector<float> xh((size_t) cached->x_in->ne[0] * cached->x_in->ne[1]);
        std::vector<float> th((size_t) cached->text_raw->ne[0] * cached->text_raw->ne[1]);
        std::vector<float> timh;
        const bool use_device_raw =
            device_raw_input &&
            lingbot_device_tensor_valid(*device_raw_input, xh.size(), sw.backend);
        if (raw_input && raw_input->size() == xh.size()) {
            xh = *raw_input;
        } else if (use_device_raw) {
            xh.clear();
        } else {
            fill_deterministic(xh, action_mode ? 0.03f : 0.02f);
        }
        if (!ex.text_raw_override.empty()) {
            th = ex.text_raw_override;
        } else {
            fill_deterministic(th, 0.01f);
        }
        timh = fill_time_raw_cached();
        out.raw_input = xh;
        out.time_raw = timh;

        if (use_device_raw) {
            const cudaError_t err = cudaMemcpy(cached->x_in->data, device_raw_input->buffer->data,
                                               ggml_nbytes(cached->x_in), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "vla(lingbot_va): cudaMemcpy(device raw -> embed.x_in) failed: %s\n",
                             cudaGetErrorString(err));
                return false;
            }
        } else {
            ggml_backend_tensor_set(cached->x_in, xh.data(), 0, ggml_nbytes(cached->x_in));
        }
        ggml_backend_tensor_set(cached->text_raw, th.data(), 0, ggml_nbytes(cached->text_raw));
        ggml_backend_tensor_set(cached->time_raw, timh.data(), 0, ggml_nbytes(cached->time_raw));
        const ggml_status st = ggml_backend_graph_compute(sw.backend, cached->graph);
        if (st != GGML_STATUS_SUCCESS) {
            return false;
        }

        out.x.resize((size_t) cached->x->ne[0] * cached->x->ne[1]);
        out.text.resize((size_t) cached->text->ne[0] * cached->text->ne[1]);
        out.t_hidden.resize((size_t) cached->t_hidden->ne[0] * cached->t_hidden->ne[1]);
        out.timestep_proj.resize((size_t) cached->timestep_proj->ne[0] * cached->timestep_proj->ne[1]);
        ggml_backend_tensor_get(cached->x, out.x.data(), 0, out.x.size() * sizeof(float));
        ggml_backend_tensor_get(cached->text, out.text.data(), 0, out.text.size() * sizeof(float));
        ggml_backend_tensor_get(cached->t_hidden, out.t_hidden.data(), 0, out.t_hidden.size() * sizeof(float));
        ggml_backend_tensor_get(cached->timestep_proj, out.timestep_proj.data(), 0, out.timestep_proj.size() * sizeof(float));
        if (!lingbot_env_enabled("VLA_LINGBOT_EMBEDDING_STAGE_DEVICE_X_DISABLE")) {
            if (!lingbot_copy_tensor_to_device_x(out, sw.backend, cached->x, "embed.x")) {
                return false;
            }
        }
        if (!lingbot_env_enabled("VLA_LINGBOT_EMBEDDING_STAGE_DEVICE_TEXT_DISABLE")) {
            if (!lingbot_copy_tensor_to_device_text(out, sw.backend, cached->text, "embed.text")) {
                return false;
            }
        } else {
            lingbot_clear_device_text(out);
        }
        if (!lingbot_env_enabled("VLA_LINGBOT_EMBEDDING_STAGE_DEVICE_TIMESTEP_DISABLE")) {
            if (!lingbot_copy_tensor_to_device_timestep_proj(out, sw.backend,
                                                             cached->timestep_proj,
                                                             "embed.timestep_proj")) {
                return false;
            }
        } else {
            lingbot_clear_device_timestep_proj(out);
        }
        lingbot_dump_embed_stage(action_mode ? "action" : "latent", seq, out.raw_input, out.time_raw,
                                 out.x, out.text, out.t_hidden, out.timestep_proj);
        build_lingbot_rope(m.head_dim, resolved_grid, out.rope_cos, out.rope_sin);
        out.text_signature = lingbot_hash_f32_bytes(out.text);
        std::vector<float> rope_sig_data;
        rope_sig_data.reserve(out.rope_cos.size() + out.rope_sin.size());
        rope_sig_data.insert(rope_sig_data.end(), out.rope_cos.begin(), out.rope_cos.end());
        rope_sig_data.insert(rope_sig_data.end(), out.rope_sin.begin(), out.rope_sin.end());
        out.rope_signature = lingbot_hash_f32_bytes(rope_sig_data);
        return true;
    }
#endif

    ggml_init_params cp = { size_t(32) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) return false;

    ggml_tensor * x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, input_dim, seq);
    ggml_tensor * text_raw = ggml_new_tensor_2d(C, GGML_TYPE_F32, m.text_dim, text_seq);
    ggml_tensor * time_raw = ggml_new_tensor_2d(C, GGML_TYPE_F32, 256, seq);
    ggml_set_input(x_in);
    ggml_set_input(text_raw);
    ggml_set_input(time_raw);

    ggml_tensor * x = action_mode ? lin(C, sw.action_embd, x_in)
                                  : lin(C, sw.patch_embd_mlp, x_in);
    ggml_tensor * text = lin(C, sw.cond.text_l2,
                             ggml_gelu(C, lin(C, sw.cond.text_l1, text_raw)));
    const LingBotConditionW & cond = action_mode ? sw.action_cond : sw.cond;
    ggml_tensor * t_hidden = lin(C, cond.time_l2,
                                 ggml_silu(C, lin(C, cond.time_l1, time_raw)));
    ggml_tensor * timestep_proj = lin(C, cond.time_proj, ggml_silu(C, t_hidden));
    ggml_set_output(x);
    ggml_set_output(text);
    ggml_set_output(t_hidden);
    ggml_set_output(timestep_proj);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 8192, false);
    ggml_build_forward_expand(gf, x);
    ggml_build_forward_expand(gf, text);
    ggml_build_forward_expand(gf, t_hidden);
    ggml_build_forward_expand(gf, timestep_proj);

    std::vector<float> xh((size_t) x_in->ne[0] * x_in->ne[1]);
    std::vector<float> th((size_t) text_raw->ne[0] * text_raw->ne[1]);
    std::vector<float> timh;
    if (raw_input && raw_input->size() == xh.size()) {
        xh = *raw_input;
    } else {
        fill_deterministic(xh, action_mode ? 0.03f : 0.02f);
    }
    if (!ex.text_raw_override.empty()) {
        th = ex.text_raw_override;
    } else {
        fill_deterministic(th, 0.01f);
    }
    timh = fill_time_raw_cached();
    out.raw_input = xh;
    out.time_raw = timh;

    ggml_gallocr_t galloc = lingbot_runtime_gallocr(sw.backend);
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }
    ggml_backend_tensor_set(x_in, xh.data(), 0, ggml_nbytes(x_in));
    ggml_backend_tensor_set(text_raw, th.data(), 0, ggml_nbytes(text_raw));
    ggml_backend_tensor_set(time_raw, timh.data(), 0, ggml_nbytes(time_raw));
    const ggml_status st = ggml_backend_graph_compute(sw.backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }

    out.x.resize((size_t) x->ne[0] * x->ne[1]);
    out.text.resize((size_t) text->ne[0] * text->ne[1]);
    out.t_hidden.resize((size_t) t_hidden->ne[0] * t_hidden->ne[1]);
    out.timestep_proj.resize((size_t) timestep_proj->ne[0] * timestep_proj->ne[1]);
    ggml_backend_tensor_get(x, out.x.data(), 0, out.x.size() * sizeof(float));
    ggml_backend_tensor_get(text, out.text.data(), 0, out.text.size() * sizeof(float));
    ggml_backend_tensor_get(t_hidden, out.t_hidden.data(), 0, out.t_hidden.size() * sizeof(float));
    ggml_backend_tensor_get(timestep_proj, out.timestep_proj.data(), 0, out.timestep_proj.size() * sizeof(float));
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (!lingbot_env_enabled("VLA_LINGBOT_EMBEDDING_STAGE_DEVICE_TEXT_DISABLE")) {
        if (!lingbot_copy_tensor_to_device_text(out, sw.backend, text, "embed.text")) {
            ggml_gallocr_free(galloc);
            ggml_free(C);
            return false;
        }
    } else {
        lingbot_clear_device_text(out);
    }
    if (!lingbot_env_enabled("VLA_LINGBOT_EMBEDDING_STAGE_DEVICE_TIMESTEP_DISABLE")) {
        if (!lingbot_copy_tensor_to_device_timestep_proj(out, sw.backend,
                                                         timestep_proj,
                                                         "embed.timestep_proj")) {
            ggml_gallocr_free(galloc);
            ggml_free(C);
            return false;
        }
    } else {
        lingbot_clear_device_timestep_proj(out);
    }
#endif
    lingbot_dump_embed_stage(action_mode ? "action" : "latent", seq, out.raw_input, out.time_raw,
                             out.x, out.text, out.t_hidden, out.timestep_proj);
    build_lingbot_rope(m.head_dim, resolved_grid, out.rope_cos, out.rope_sin);
    out.text_signature = lingbot_hash_f32_bytes(out.text);
    std::vector<float> rope_sig_data;
    rope_sig_data.reserve(out.rope_cos.size() + out.rope_sin.size());
    rope_sig_data.insert(rope_sig_data.end(), out.rope_cos.begin(), out.rope_cos.end());
    rope_sig_data.insert(rope_sig_data.end(), out.rope_sin.begin(), out.rope_sin.end());
    out.rope_signature = lingbot_hash_f32_bytes(rope_sig_data);
    ggml_gallocr_free(galloc);
    ggml_free(C);
    return true;
}

bool exec_block_one(
        LingBotTransformerExecutor & ex,
        LingBotRuntimeWeights & bw,
        size_t block_index,
        LingBotExecState & state,
        const char * branch_label = nullptr) {
    const LingBotVAModelArch & m = ex.m;
    const int64_t hidden = m.cfg.hidden;
    const int64_t text_seq = state.text_seq > 0 ? state.text_seq : 2;

    ggml_init_params cp = { size_t(64) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) return false;

    ggml_tensor * x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
    ggml_tensor * text = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, text_seq);
    ggml_tensor * timestep_proj = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden * 6, state.seq);
    ggml_tensor * rope_cos = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, m.head_dim / 2, 1, state.seq);
    ggml_tensor * rope_sin = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, m.head_dim / 2, 1, state.seq);
    ggml_set_input(x_in);
    ggml_set_input(text);
    ggml_set_input(timestep_proj);
    ggml_set_input(rope_cos);
    ggml_set_input(rope_sin);

    LingBotBlockTrace trace;
    const bool dump_trace = std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR") != nullptr && branch_label != nullptr;
    ggml_tensor * out = build_block_shape(C, bw.blocks[block_index], x_in, text, timestep_proj,
                                          rope_cos, rope_sin, m, state.seq, text_seq,
                                          dump_trace ? &trace : nullptr);
    ggml_set_output(out);
    if (dump_trace) {
        ggml_set_output(trace.n1);
        ggml_set_output(trace.self_q);
        ggml_set_output(trace.self_k);
        ggml_set_output(trace.self_v);
        ggml_set_output(trace.self_qh);
        ggml_set_output(trace.self_kh);
        ggml_set_output(trace.self_merged);
        ggml_set_output(trace.self_attn);
        ggml_set_output(trace.post_self);
        ggml_set_output(trace.n2);
        ggml_set_output(trace.cross_q);
        ggml_set_output(trace.cross_k);
        ggml_set_output(trace.cross_v);
        ggml_set_output(trace.cross_merged);
        ggml_set_output(trace.cross_attn);
        ggml_set_output(trace.post_cross);
        ggml_set_output(trace.n3);
        ggml_set_output(trace.ff);
    }
    ggml_cgraph * gf = ggml_new_graph_custom(C, 32768, false);
    if (dump_trace) {
        ggml_build_forward_expand(gf, trace.n1);
        ggml_build_forward_expand(gf, trace.self_q);
        ggml_build_forward_expand(gf, trace.self_k);
        ggml_build_forward_expand(gf, trace.self_v);
        ggml_build_forward_expand(gf, trace.self_qh);
        ggml_build_forward_expand(gf, trace.self_kh);
        ggml_build_forward_expand(gf, trace.self_merged);
        ggml_build_forward_expand(gf, trace.self_attn);
        ggml_build_forward_expand(gf, trace.post_self);
        ggml_build_forward_expand(gf, trace.n2);
        ggml_build_forward_expand(gf, trace.cross_q);
        ggml_build_forward_expand(gf, trace.cross_k);
        ggml_build_forward_expand(gf, trace.cross_v);
        ggml_build_forward_expand(gf, trace.cross_merged);
        ggml_build_forward_expand(gf, trace.cross_attn);
        ggml_build_forward_expand(gf, trace.post_cross);
        ggml_build_forward_expand(gf, trace.n3);
        ggml_build_forward_expand(gf, trace.ff);
    }
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = lingbot_runtime_gallocr(bw.backend);
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }
    ggml_backend_tensor_set(x_in, state.x.data(), 0, state.x.size() * sizeof(float));
    ggml_backend_tensor_set(text, state.text.data(), 0, state.text.size() * sizeof(float));
    ggml_backend_tensor_set(timestep_proj, state.timestep_proj.data(), 0, state.timestep_proj.size() * sizeof(float));
    ggml_backend_tensor_set(rope_cos, state.rope_cos.data(), 0, state.rope_cos.size() * sizeof(float));
    ggml_backend_tensor_set(rope_sin, state.rope_sin.data(), 0, state.rope_sin.size() * sizeof(float));

    const ggml_status st = ggml_backend_graph_compute(bw.backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }
    state.x.resize((size_t) out->ne[0] * out->ne[1]);
    ggml_backend_tensor_get(out, state.x.data(), 0, state.x.size() * sizeof(float));
    if (dump_trace) {
        const std::string dump_dir(std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR"));
        std::string tag;
        if (const char * tag_c = std::getenv("VLA_LINGBOT_PREDICT_TRACE_TAG")) {
            tag = tag_c;
        }
        auto dump_tensor_2d = [&](const char * name, ggml_tensor * t) {
            std::vector<float> data((size_t) t->ne[0] * (size_t) t->ne[1]);
            ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
            const std::string stem = std::string("lingbot_predict_trace_") + branch_label +
                                     "_block_" + name + (tag.empty() ? std::string() : "_" + tag);
            dump_f32_file(dump_dir + "/" + stem + ".f32", data);
            dump_text_file(dump_dir + "/" + stem + ".shape.txt",
                           std::to_string((long long) t->ne[1]) + " " +
                           std::to_string((long long) t->ne[0]) + "\n");
        };
        dump_tensor_2d("n1", trace.n1);
        dump_tensor_2d("self_q", trace.self_q);
        dump_tensor_2d("self_k", trace.self_k);
        dump_tensor_2d("self_v", trace.self_v);
        auto dump_tensor_3d = [&](const char * name, ggml_tensor * t) {
            std::vector<float> data((size_t) t->ne[0] * (size_t) t->ne[1] * (size_t) t->ne[2]);
            ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
            const std::string stem = std::string("lingbot_predict_trace_") + branch_label +
                                     "_block_" + name + (tag.empty() ? std::string() : "_" + tag);
            dump_f32_file(dump_dir + "/" + stem + ".f32", data);
            dump_text_file(dump_dir + "/" + stem + ".shape.txt",
                           std::to_string((long long) t->ne[2]) + " " +
                           std::to_string((long long) t->ne[1]) + " " +
                           std::to_string((long long) t->ne[0]) + "\n");
        };
        dump_tensor_3d("self_q_rope", trace.self_qh);
        dump_tensor_3d("self_k_rope", trace.self_kh);
        dump_tensor_2d("self_ctx", trace.self_merged);
        dump_tensor_2d("self_attn", trace.self_attn);
        dump_tensor_2d("post_self", trace.post_self);
        dump_tensor_2d("n2", trace.n2);
        dump_tensor_2d("cross_q", trace.cross_q);
        dump_tensor_2d("cross_k", trace.cross_k);
        dump_tensor_2d("cross_v", trace.cross_v);
        dump_tensor_2d("cross_ctx", trace.cross_merged);
        dump_tensor_2d("cross_attn", trace.cross_attn);
        dump_tensor_2d("post_cross", trace.post_cross);
        dump_tensor_2d("n3", trace.n3);
        dump_tensor_2d("ff", trace.ff);
    }
    ggml_gallocr_free(galloc);
    ggml_free(C);
    return true;
}

bool exec_block_sequence(
        LingBotTransformerExecutor & ex,
        LingBotRuntimeWeights & bw,
        int64_t first_block,
        int64_t n_blocks,
        LingBotExecState & state) {
    const LingBotVAModelArch & m = ex.m;
    const int64_t hidden = m.cfg.hidden;
    const int64_t text_seq = state.text_seq > 0 ? state.text_seq : 2;
    if (first_block < 0 || n_blocks <= 0 ||
        first_block + n_blocks > (int64_t) bw.blocks.size()) {
        std::fprintf(stderr,
                     "vla(lingbot_va): invalid block sequence first=%lld count=%lld available=%zu\n",
                     (long long) first_block, (long long) n_blocks, bw.blocks.size());
        return false;
    }

    LingBotBlockSequenceGraph * cached = nullptr;
    for (const auto & entry : ex.block_sequence_graphs) {
        if (entry && entry->matches(bw, first_block, n_blocks, state.seq, text_seq)) {
            cached = entry.get();
            break;
        }
    }
    if (!cached) {
        auto built = std::make_unique<LingBotBlockSequenceGraph>();
        built->weights = &bw;
        built->first_block = first_block;
        built->n_blocks = n_blocks;
        built->seq = state.seq;
        built->text_seq = text_seq;

        ggml_init_params cp = { size_t(512) * 1024 * 1024, nullptr, true };
        built->ctx = ggml_init(cp);
        if (!built->ctx) return false;

        ggml_context * C = built->ctx;
        built->x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
        built->text = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, text_seq);
        built->timestep_proj = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden * 6, state.seq);
        built->rope_cos = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, m.head_dim / 2, 1, state.seq);
        built->rope_sin = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, m.head_dim / 2, 1, state.seq);
        ggml_set_input(built->x_in);
        ggml_set_input(built->text);
        ggml_set_input(built->timestep_proj);
        ggml_set_input(built->rope_cos);
        ggml_set_input(built->rope_sin);

        ggml_tensor * h = built->x_in;
        for (int64_t i = 0; i < n_blocks; ++i) {
            h = build_block_shape(C, bw.blocks[(size_t) (first_block + i)], h, built->text,
                                  built->timestep_proj, built->rope_cos, built->rope_sin, m,
                                  state.seq, text_seq);
        }
        built->out = h;
        ggml_set_output(built->out);

        built->graph = ggml_new_graph_custom(C, 262144, false);
        ggml_build_forward_expand(built->graph, built->out);
        built->galloc = lingbot_runtime_gallocr(bw.backend);
        if (!built->galloc || !ggml_gallocr_alloc_graph(built->galloc, built->graph)) {
            std::fprintf(stderr,
                         "vla(lingbot_va): block sequence graph allocation failed first=%lld count=%lld\n",
                         (long long) first_block, (long long) n_blocks);
            return false;
        }
        cached = built.get();
        ex.block_sequence_graphs.push_back(std::move(built));
        std::printf("vla(lingbot_va): cached block sequence graph first=%lld count=%lld seq=%lld text_seq=%lld\n",
                    (long long) first_block + 1,
                    (long long) n_blocks,
                    (long long) state.seq,
                    (long long) text_seq);
    }

    ggml_backend_tensor_set(cached->x_in, state.x.data(), 0, state.x.size() * sizeof(float));
    ggml_backend_tensor_set(cached->text, state.text.data(), 0, state.text.size() * sizeof(float));
    ggml_backend_tensor_set(cached->timestep_proj, state.timestep_proj.data(), 0, state.timestep_proj.size() * sizeof(float));
    ggml_backend_tensor_set(cached->rope_cos, state.rope_cos.data(), 0, state.rope_cos.size() * sizeof(float));
    ggml_backend_tensor_set(cached->rope_sin, state.rope_sin.data(), 0, state.rope_sin.size() * sizeof(float));

    const ggml_status st = ggml_backend_graph_compute(bw.backend, cached->graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(lingbot_va): block sequence graph failed (%d) first=%lld count=%lld\n",
                     (int) st, (long long) first_block, (long long) n_blocks);
        return false;
    }
    state.x.resize((size_t) cached->out->ne[0] * cached->out->ne[1]);
    ggml_backend_tensor_get(cached->out, state.x.data(), 0, state.x.size() * sizeof(float));
    return true;
}

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
bool exec_block_one_cuda_self_attn(
        LingBotTransformerExecutor & ex,
        LingBotRuntimeWeights & bw,
        size_t block_index,
        LingBotExecState & state,
        const LingBotFlexMaskMeta & meta,
        const LingBotBlockSparseTable & self_table,
        const std::vector<unsigned char> & token_mask,
        LingBotCudaSelfAttnBlockRunner * block_runner_hint = nullptr) {
    const LingBotVAModelArch & m = ex.m;
    const int64_t hidden = m.cfg.hidden;
    const int64_t text_seq = state.text_seq > 0 ? state.text_seq : 2;
    const bool timing = lingbot_timing_enabled();
    const auto t_block0 = LingBotClock::now();
    auto t_last = t_block0;
    auto log_part = [&](const char * name) {
        if (!timing) return;
        const auto t_now = LingBotClock::now();
        std::printf("vla(lingbot_va): timing cuda-self-attn block=%zu seq=%lld mode=%d branch=%d part=%s %.3fms\n",
                    block_index + 1,
                    (long long) state.seq,
                    state.cache_mode,
                    state.cache_branch,
                    name,
                    lingbot_elapsed_ms(t_last, t_now));
        t_last = t_now;
    };
    if (state.seq != (int64_t) meta.seq_ids.size()) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA self-attn block seq mismatch state=%lld meta=%zu\n",
                     (long long) state.seq, meta.seq_ids.size());
        return false;
    }

    const LingBotBlockW & b = bw.blocks[block_index];
    std::vector<float> self_ctx_hseq;
    LingBotRuntimeKVDeviceResult self_ctx_device;
    const bool disable_graph_cache =
        lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_CUDA_SELF_ATTN_GRAPH_CACHE_DISABLE");
    const bool trace_dump_enabled = std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR") != nullptr;
    const bool device_x_handoff =
        state.allow_device_x_handoff &&
        !disable_graph_cache &&
        !trace_dump_enabled &&
        !lingbot_env_enabled("VLA_LINGBOT_STREAM_BLOCK_TRACE") &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_CUDA_SELF_ATTN_DEVICE_X_DISABLE");
    const bool use_bf16_boundary =
        lingbot_env_enabled("VLA_LINGBOT_BLOCK_BF16_BOUNDARY") &&
        !lingbot_env_enabled("VLA_LINGBOT_STATE_F32");
    LingBotCudaSelfAttnBlockRunner * block_runner = nullptr;
    if (!disable_graph_cache && block_runner_hint &&
        block_runner_hint->matches(bw, block_index, state.seq, text_seq, use_bf16_boundary)) {
        block_runner = block_runner_hint;
    }
    if (!disable_graph_cache && !block_runner) {
        block_runner = lingbot_get_cuda_self_attn_block_runner(
            ex.m, bw, block_index, state.seq, text_seq, use_bf16_boundary);
    }
    std::unique_ptr<LingBotCudaSelfAttnQkvGraph> temp_qkv_graph;
    LingBotCudaSelfAttnQkvGraph * qkv_graph = nullptr;
    if (block_runner && block_runner->qkv) {
        qkv_graph = block_runner->qkv;
    }
    if (!disable_graph_cache) {
        if (!qkv_graph) {
            for (const auto & entry : ex.m.cuda_self_attn_qkv_graphs) {
                if (entry && entry->matches(bw, block_index, state.seq)) {
                    qkv_graph = entry.get();
                    if (block_runner) block_runner->qkv = qkv_graph;
                    break;
                }
            }
        }
    }
    if (!qkv_graph) {
        auto built = std::make_unique<LingBotCudaSelfAttnQkvGraph>();
        ggml_init_params cp = { size_t(96) * 1024 * 1024, nullptr, true };
        built->ctx = ggml_init(cp);
        if (!built->ctx) return false;
        ggml_context * C = built->ctx;

        built->weights = &bw;
        built->block_index = block_index;
        built->seq = state.seq;
        built->x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
        built->timestep_proj = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden * 6, state.seq);
        built->rope_cos = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, m.head_dim / 2, 1, state.seq);
        built->rope_sin = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, m.head_dim / 2, 1, state.seq);
        ggml_set_input(built->x_in);
        ggml_set_input(built->timestep_proj);
        ggml_set_input(built->rope_cos);
        ggml_set_input(built->rope_sin);

        ggml_tensor * shift_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, state.seq, 0),
                                           ggml_view_1d(C, b.scale_shift, hidden, 0));
        ggml_tensor * scale_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, state.seq, 1),
                                           ggml_view_1d(C, b.scale_shift, hidden,
                                                        (size_t) hidden * ggml_element_size(b.scale_shift)));
        built->n1 = adaln(C, built->x_in, shift_msa, scale_msa, 1e-6f);
        LingBotAttentionTrace trace;
        (void) build_attention_shape(C, b.self_attn, built->n1, built->n1, built->rope_cos, built->rope_sin,
                                     m, state.seq, state.seq, &trace);
        built->q = trace.q;
        built->k = trace.k;
        built->v = trace.v;
        built->qh = trace.qh;
        built->kh = trace.kh;
        built->vh = trace.vh;
        ggml_set_output(built->n1);
        ggml_set_output(built->q);
        ggml_set_output(built->k);
        ggml_set_output(built->v);
        ggml_set_output(built->qh);
        ggml_set_output(built->kh);
        ggml_set_output(built->vh);

        built->graph = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(built->graph, built->n1);
        ggml_build_forward_expand(built->graph, built->q);
        ggml_build_forward_expand(built->graph, built->k);
        ggml_build_forward_expand(built->graph, built->v);
        ggml_build_forward_expand(built->graph, built->qh);
        ggml_build_forward_expand(built->graph, built->kh);
        ggml_build_forward_expand(built->graph, built->vh);
        log_part("qkv_graph_build");

        built->galloc = lingbot_runtime_gallocr(bw.backend);
        if (!built->galloc || !ggml_gallocr_alloc_graph(built->galloc, built->graph)) {
            return false;
        }
        log_part("qkv_graph_alloc");
        qkv_graph = built.get();
        if (disable_graph_cache) {
            temp_qkv_graph = std::move(built);
        } else {
            ex.m.cuda_self_attn_qkv_graphs.push_back(std::move(built));
            if (block_runner) block_runner->qkv = qkv_graph;
            std::printf("vla(lingbot_va): cached official CUDA self-attn QKV graph block=%zu seq=%lld\n",
                        block_index + 1, (long long) state.seq);
        }
    }
    {
        if (device_x_handoff && block_index > 0 && !lingbot_device_x_valid(state, hidden, bw.backend)) {
            std::fprintf(stderr,
                         "vla(lingbot_va): CUDA self-attn device-x handoff missing at block=%zu seq=%lld\n",
                         block_index + 1, (long long) state.seq);
            return false;
        }
        if (device_x_handoff && lingbot_device_x_valid(state, hidden, bw.backend)) {
            if (!lingbot_copy_device_x_to_tensor(state, qkv_graph->x_in, "qkv.x_in")) {
                return false;
            }
            log_part("qkv_input_d2d");
        } else {
            ggml_backend_tensor_set(qkv_graph->x_in, state.x.data(), 0, state.x.size() * sizeof(float));
            log_part("qkv_input_h2d_x");
            if (device_x_handoff) {
                ggml_backend_synchronize(bw.backend);
                if (!lingbot_copy_tensor_to_device_x(state, bw.backend, qkv_graph->x_in, "qkv.x_in")) {
                    return false;
                }
                log_part("qkv_input_stage_d2d");
            }
        }
        bool qkv_timestep_copied_d2d = false;
        if (!lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_QKV_TIMESTEP_D2D_DISABLE") &&
            lingbot_device_timestep_proj_valid(state, ggml_nbytes(qkv_graph->timestep_proj), bw.backend)) {
            if (!lingbot_copy_device_timestep_proj_to_tensor(state, qkv_graph->timestep_proj,
                                                             "qkv.timestep_proj")) {
                return false;
            }
            qkv_timestep_copied_d2d = true;
        }
        if (!qkv_timestep_copied_d2d) {
            ggml_backend_tensor_set(qkv_graph->timestep_proj, state.timestep_proj.data(), 0,
                                    state.timestep_proj.size() * sizeof(float));
        }
        const bool rope_cache_enabled =
            lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_ROPE_CACHE") &&
            !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_ROPE_CACHE_DISABLE") &&
            state.rope_signature != 0;
        if (!rope_cache_enabled || qkv_graph->uploaded_rope_signature != state.rope_signature) {
            ggml_backend_tensor_set(qkv_graph->rope_cos, state.rope_cos.data(), 0, state.rope_cos.size() * sizeof(float));
            ggml_backend_tensor_set(qkv_graph->rope_sin, state.rope_sin.data(), 0, state.rope_sin.size() * sizeof(float));
            qkv_graph->uploaded_rope_signature = rope_cache_enabled ? state.rope_signature : 0;
        }
        log_part("qkv_input_h2d_aux");

        const ggml_status st = ggml_backend_graph_compute(bw.backend, qkv_graph->graph);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(lingbot_va): CUDA self-attn QKV graph failed (%d)\n", (int) st);
            return false;
        }
        log_part("qkv_graph_compute");
        std::vector<float> context_shd;
        std::vector<float> attn_k_shd;
        std::vector<float> attn_v_shd;
        int64_t attn_seq_k = 0;
        const char * stream_dump_dir_c =
            (block_index == 0 && lingbot_env_enabled("VLA_LINGBOT_STREAM_BLOCK_TRACE"))
                ? std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR")
                : nullptr;
        const bool need_stream_attn_dump = stream_dump_dir_c && *stream_dump_dir_c;
        const bool use_device_self_ctx =
            !need_stream_attn_dump &&
            (state.allow_device_self_ctx_handoff ||
             lingbot_env_enabled("VLA_LINGBOT_RUNTIME_KV_DEVICE_SELF_CTX_D2D")) &&
            !lingbot_env_enabled("VLA_LINGBOT_RUNTIME_KV_DEVICE_SELF_CTX_D2D_DISABLE");
        const bool warp_runtime_kv =
            lingbot_env_enabled("VLA_LINGBOT_RUNTIME_KV_WARP_ATTN") &&
            !lingbot_env_enabled("VLA_LINGBOT_RUNTIME_KV_WARP_ATTN_DISABLE");
        if (use_device_self_ctx && warp_runtime_kv) {
            // qh/kh/vh are produced by the ggml QKV graph. The custom warp
            // runtime-KV kernels launch on stream 0, so wait for the ggml
            // backend stream before those kernels read Q/K/V device pointers.
            ggml_backend_synchronize(bw.backend);
            log_part("qkv_graph_sync_for_warp_runtime_kv");
        }
        if (!real_qkv_to_cuda_context(bw.backend, qkv_graph->qh, qkv_graph->kh, qkv_graph->vh, self_table, token_mask,
                                      m, state.seq, context_shd, nullptr,
                                      state.cache_session_id,
                                      state.cache_block_index,
                                      state.cache_mode,
                                      state.cache_branch,
                                      need_stream_attn_dump ? &attn_k_shd : nullptr,
                                      need_stream_attn_dump ? &attn_v_shd : nullptr,
                                      need_stream_attn_dump ? &attn_seq_k : nullptr,
                                      use_device_self_ctx ? &self_ctx_device : nullptr)) {
            return false;
        }
        log_part("runtime_kv_context");
        if (self_ctx_device.valid) {
            goto lingbot_cuda_self_attn_done;
        }
        if (need_stream_attn_dump) {
                static std::atomic<int> cuda_trace_counter{0};
                const int trace_idx = cuda_trace_counter.fetch_add(1);
                auto trace_env_i64 = [](const char * name, int64_t fallback) {
                    const char * env = std::getenv(name);
                    if (!env || !*env) return fallback;
                    char * end = nullptr;
                    const long long v = std::strtoll(env, &end, 10);
                    return (end && *end == '\0') ? (int64_t) v : fallback;
                };
                const int64_t filter_mode = trace_env_i64("VLA_LINGBOT_STREAM_TRACE_MODE", INT64_MIN);
                const int64_t filter_seq = trace_env_i64("VLA_LINGBOT_STREAM_TRACE_SEQ", INT64_MIN);
                const int64_t filter_call_min = trace_env_i64("VLA_LINGBOT_STREAM_TRACE_CALL_MIN", INT64_MIN);
                const int64_t filter_call_max = trace_env_i64("VLA_LINGBOT_STREAM_TRACE_CALL_MAX", INT64_MAX);
                const bool dump_this =
                    (filter_mode == INT64_MIN || filter_mode == state.cache_mode) &&
                    (filter_seq == INT64_MIN || filter_seq == state.seq) &&
                    (trace_idx >= filter_call_min && trace_idx <= filter_call_max);
                if (!dump_this) {
                    self_ctx_hseq = shd_to_hidden_seq(context_shd, state.seq, m.n_heads, m.head_dim);
                    log_part("context_shd_to_hseq");
                    goto lingbot_cuda_self_attn_done;
                }
                const std::string prefix = std::string(stream_dump_dir_c) +
                    "/lingbot_predict_trace_cuda_block0_call_" + std::to_string(trace_idx) +
                    "_seq" + std::to_string((long long) state.seq) +
                    "_mode" + std::to_string(state.cache_mode);
                std::vector<float> q_shd;
                std::vector<float> k_shd;
                std::vector<float> v_shd;
                hds_tensor_to_shd(qkv_graph->qh, state.seq, m.n_heads, m.head_dim, q_shd);
                hds_tensor_to_shd(qkv_graph->kh, state.seq, m.n_heads, m.head_dim, k_shd);
                hds_tensor_to_shd(qkv_graph->vh, state.seq, m.n_heads, m.head_dim, v_shd);
                auto dump_tensor_2d = [&](const char * name, ggml_tensor * t) {
                    std::vector<float> data((size_t) t->ne[0] * (size_t) t->ne[1]);
                    ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
                    dump_f32_file(prefix + "_" + name + ".f32", data);
                    dump_text_file(prefix + "_" + name + ".shape.txt",
                                   std::to_string((long long) t->ne[1]) + " " +
                                   std::to_string((long long) t->ne[0]) + "\n");
                };
                dump_f32_file(prefix + "_x.f32", state.x);
                dump_text_file(prefix + "_x.shape.txt",
                               std::to_string((long long) state.seq) + " " +
                               std::to_string((long long) hidden) + "\n");
                dump_f32_file(prefix + "_timestep_proj.f32", state.timestep_proj);
                dump_text_file(prefix + "_timestep_proj.shape.txt",
                               std::to_string((long long) state.seq) + " " +
                               std::to_string((long long) hidden * 6) + "\n");
                dump_tensor_2d("n1", qkv_graph->n1);
                dump_tensor_2d("self_q", qkv_graph->q);
                dump_tensor_2d("self_k", qkv_graph->k);
                dump_tensor_2d("self_v", qkv_graph->v);
                dump_f32_file(prefix + "_self_q_rope.f32", q_shd);
                dump_f32_file(prefix + "_self_k_rope.f32", k_shd);
                dump_f32_file(prefix + "_self_v_heads.f32", v_shd);
                dump_f32_file(prefix + "_self_ctx.f32", context_shd);
                const std::string hshape = std::to_string((long long) state.seq) + " " +
                    std::to_string((long long) m.n_heads) + " " +
                    std::to_string((long long) m.head_dim) + "\n";
                const std::string kshape = std::to_string((long long) attn_seq_k) + " " +
                    std::to_string((long long) m.n_heads) + " " +
                    std::to_string((long long) m.head_dim) + "\n";
                dump_text_file(prefix + "_self_q_rope.shape.txt", hshape);
                dump_text_file(prefix + "_self_k_rope.shape.txt", hshape);
                dump_text_file(prefix + "_self_v_heads.shape.txt", hshape);
                dump_text_file(prefix + "_self_ctx.shape.txt", hshape);
                dump_f32_file(prefix + "_self_k_attn.f32", attn_k_shd);
                dump_f32_file(prefix + "_self_v_attn.f32", attn_v_shd);
                dump_text_file(prefix + "_self_k_attn.shape.txt", kshape);
                dump_text_file(prefix + "_self_v_attn.shape.txt", kshape);
        }
        self_ctx_hseq = shd_to_hidden_seq(context_shd, state.seq, m.n_heads, m.head_dim);
        log_part("context_shd_to_hseq");
    }
lingbot_cuda_self_attn_done:

    std::unique_ptr<LingBotCudaSelfAttnPostGraph> temp_post_graph;
    LingBotCudaSelfAttnPostGraph * post_graph = nullptr;
    const bool static_text_cache_enabled =
        lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_TEXT_CACHE") &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_TEXT_CACHE_DISABLE") &&
        state.text_signature != 0 &&
        (state.cache_branch == 0 || state.cache_branch == 1);
    const int post_text_slot = static_text_cache_enabled ? state.cache_branch : -1;
    if (block_runner && post_text_slot >= 0 && block_runner->post_by_branch[post_text_slot]) {
        post_graph = block_runner->post_by_branch[post_text_slot];
    } else if (block_runner && block_runner->post) {
        post_graph = block_runner->post;
    }
    if (!disable_graph_cache) {
        if (!post_graph) {
            for (const auto & entry : ex.m.cuda_self_attn_post_graphs) {
                if (entry && entry->matches(bw, block_index, state.seq, text_seq,
                                            use_bf16_boundary, post_text_slot)) {
                    post_graph = entry.get();
                    if (block_runner) {
                        if (post_text_slot >= 0) block_runner->post_by_branch[post_text_slot] = post_graph;
                        else block_runner->post = post_graph;
                    }
                    break;
                }
            }
        }
    }
    if (!post_graph) {
        auto built = std::make_unique<LingBotCudaSelfAttnPostGraph>();
        ggml_init_params cp = { size_t(96) * 1024 * 1024, nullptr, true };
        built->ctx = ggml_init(cp);
        if (!built->ctx) return false;
        ggml_context * C = built->ctx;

        built->weights = &bw;
        built->block_index = block_index;
        built->seq = state.seq;
        built->text_seq = text_seq;
        built->text_slot = post_text_slot;
        built->bf16_boundary = use_bf16_boundary;
        built->x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
        built->text = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, text_seq);
        built->timestep_proj = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden * 6, state.seq);
        built->self_ctx = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
        ggml_set_input(built->x_in);
        ggml_set_input(built->text);
        ggml_set_input(built->timestep_proj);
        ggml_set_input(built->self_ctx);

        ggml_tensor * gate_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, state.seq, 2),
                                          ggml_view_1d(C, b.scale_shift, hidden,
                                                       (size_t) 2 * (size_t) hidden * ggml_element_size(b.scale_shift)));
        built->gate_msa = gate_msa;
        ggml_tensor * c_shift_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, state.seq, 3),
                                             ggml_view_1d(C, b.scale_shift, hidden,
                                                          (size_t) 3 * (size_t) hidden * ggml_element_size(b.scale_shift)));
        ggml_tensor * c_scale_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, state.seq, 4),
                                             ggml_view_1d(C, b.scale_shift, hidden,
                                                          (size_t) 4 * (size_t) hidden * ggml_element_size(b.scale_shift)));
        ggml_tensor * c_gate_msa = ggml_add(C, chunk_hidden(C, built->timestep_proj, hidden, state.seq, 5),
                                            ggml_view_1d(C, b.scale_shift, hidden,
                                                         (size_t) 5 * (size_t) hidden * ggml_element_size(b.scale_shift)));
        auto bf16_boundary = [&](ggml_tensor * t) -> ggml_tensor * {
            return use_bf16_boundary ? ggml_cast(C, ggml_cast(C, t, GGML_TYPE_BF16), GGML_TYPE_F32) : t;
        };

        ggml_tensor * a1 = lin(C, b.self_attn.o, built->self_ctx);
        built->self_attn_out = a1;
        ggml_tensor * gated_self = ggml_mul(C, a1, gate_msa);
        built->gated_self = gated_self;
        ggml_tensor * x = ggml_add(C, built->x_in, gated_self);
        x = bf16_boundary(x);
        built->post_self = x;
        ggml_tensor * n2 = ggml_add(C, ggml_mul(C, ggml_norm(C, x, 1e-6f), b.cross_norm_weight), b.cross_norm_bias);
        n2 = bf16_boundary(n2);
        built->n2 = n2;
        LingBotAttentionTrace cross_trace;
        const bool post_trace_enabled =
            lingbot_env_enabled("VLA_LINGBOT_STREAM_BLOCK_TRACE") &&
            std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR") != nullptr;
        ggml_tensor * a2 = build_attention_shape(C, b.cross_attn, n2, built->text, nullptr, nullptr,
                                                 m, state.seq, text_seq,
                                                 post_trace_enabled ? &cross_trace : nullptr);
        a2 = bf16_boundary(a2);
        built->cross_q = cross_trace.q;
        built->cross_k = cross_trace.k;
        built->cross_v = cross_trace.v;
        built->cross_ctx = cross_trace.merged;
        built->cross_attn = a2;
        x = ggml_add(C, x, a2);
        x = bf16_boundary(x);
        built->post_cross = x;
        ggml_tensor * n3 = adaln(C, x, c_shift_msa, c_scale_msa, 1e-6f);
        n3 = bf16_boundary(n3);
        built->n3 = n3;
        ggml_tensor * ff = lin(C, b.ffn_down, ggml_gelu(C, lin(C, b.ffn_up, n3)));
        ff = bf16_boundary(ff);
        built->ff = ff;
        built->out = ggml_add(C, x, ggml_mul(C, ff, c_gate_msa));
        built->out = bf16_boundary(built->out);
        if (lingbot_env_enabled("VLA_LINGBOT_STREAM_BLOCK_TRACE") &&
            std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR") != nullptr) {
            ggml_set_output(built->self_attn_out);
            ggml_set_output(built->gate_msa);
            ggml_set_output(built->gated_self);
            ggml_set_output(built->post_self);
            ggml_set_output(built->n2);
            ggml_set_output(built->cross_q);
            ggml_set_output(built->cross_k);
            ggml_set_output(built->cross_v);
            ggml_set_output(built->cross_ctx);
            ggml_set_output(built->cross_attn);
            ggml_set_output(built->post_cross);
            ggml_set_output(built->n3);
            ggml_set_output(built->ff);
        }
        ggml_set_output(built->out);

        built->graph = ggml_new_graph_custom(C, 32768, false);
        if (post_trace_enabled) {
            ggml_build_forward_expand(built->graph, built->self_attn_out);
            ggml_build_forward_expand(built->graph, built->gate_msa);
            ggml_build_forward_expand(built->graph, built->gated_self);
            ggml_build_forward_expand(built->graph, built->post_self);
            ggml_build_forward_expand(built->graph, built->n2);
            ggml_build_forward_expand(built->graph, built->cross_q);
            ggml_build_forward_expand(built->graph, built->cross_k);
            ggml_build_forward_expand(built->graph, built->cross_v);
            ggml_build_forward_expand(built->graph, built->cross_ctx);
            ggml_build_forward_expand(built->graph, built->cross_attn);
            ggml_build_forward_expand(built->graph, built->post_cross);
            ggml_build_forward_expand(built->graph, built->n3);
            ggml_build_forward_expand(built->graph, built->ff);
        }
        ggml_build_forward_expand(built->graph, built->out);
        log_part("post_graph_build");
        built->galloc = lingbot_runtime_gallocr(bw.backend);
        if (!built->galloc || !ggml_gallocr_alloc_graph(built->galloc, built->graph)) {
            return false;
        }
        log_part("post_graph_alloc");
        post_graph = built.get();
        if (disable_graph_cache) {
            temp_post_graph = std::move(built);
        } else {
            ex.m.cuda_self_attn_post_graphs.push_back(std::move(built));
            if (block_runner) {
                if (post_text_slot >= 0) block_runner->post_by_branch[post_text_slot] = post_graph;
                else block_runner->post = post_graph;
            }
            std::printf("vla(lingbot_va): cached official CUDA self-attn post graph block=%zu seq=%lld text_seq=%lld text_slot=%d bf16_boundary=%d\n",
                        block_index + 1, (long long) state.seq, (long long) text_seq,
                        post_text_slot, use_bf16_boundary ? 1 : 0);
        }
    }
    if (device_x_handoff && lingbot_device_x_valid(state, hidden, bw.backend)) {
        if (!lingbot_copy_device_x_to_tensor(state, post_graph->x_in, "post.x_in")) {
            return false;
        }
        log_part("post_x_d2d");
    } else {
        ggml_backend_tensor_set(post_graph->x_in, state.x.data(), 0, state.x.size() * sizeof(float));
        log_part("post_x_h2d");
    }
    bool post_text_copied_d2d = false;
    if (lingbot_official_post_text_d2d_enabled() &&
        lingbot_device_text_valid(state, ggml_nbytes(post_graph->text), bw.backend)) {
        if (lingbot_copy_device_text_to_tensor(state, post_graph->text, "post.text")) {
            post_text_copied_d2d = true;
            post_graph->uploaded_text_signature = 0;
        } else {
            std::fprintf(stderr, "vla(lingbot_va): post text D2D copy failed, falling back to host\n");
        }
    }
    if (!post_text_copied_d2d &&
        (!static_text_cache_enabled || post_graph->uploaded_text_signature != state.text_signature)) {
        ggml_backend_tensor_set(post_graph->text, state.text.data(), 0, state.text.size() * sizeof(float));
        post_graph->uploaded_text_signature = static_text_cache_enabled ? state.text_signature : 0;
    }
    if (post_text_copied_d2d) {
        log_part("post_text_d2d");
    }
    bool post_timestep_copied_d2d = false;
    if (!lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_POST_TIMESTEP_D2D_DISABLE") &&
        lingbot_device_timestep_proj_valid(state, ggml_nbytes(post_graph->timestep_proj), bw.backend)) {
        if (lingbot_copy_device_timestep_proj_to_tensor(state, post_graph->timestep_proj,
                                                        "post.timestep_proj")) {
            post_timestep_copied_d2d = true;
        } else {
            std::fprintf(stderr, "vla(lingbot_va): post timestep_proj D2D copy failed, falling back to host\n");
        }
    }
    if (!post_timestep_copied_d2d) {
        ggml_backend_tensor_set(post_graph->timestep_proj, state.timestep_proj.data(), 0, state.timestep_proj.size() * sizeof(float));
    }
    log_part("post_input_h2d_aux");
    bool self_ctx_copied_d2d = false;
    if (self_ctx_device.valid &&
        self_ctx_device.data &&
        self_ctx_device.seq == state.seq &&
        self_ctx_device.heads == m.n_heads &&
        self_ctx_device.head_dim == m.head_dim &&
        post_graph->self_ctx && post_graph->self_ctx->data &&
        self_ctx_device.bytes == ggml_nbytes(post_graph->self_ctx)) {
        if (lingbot_copy_runtime_self_ctx_to_tensor(post_graph->self_ctx, self_ctx_device,
                                                    "post self_ctx")) {
            self_ctx_copied_d2d = true;
        } else {
            std::fprintf(stderr, "vla(lingbot_va): post self_ctx event/D2D handoff failed\n");
        }
    }
    if (self_ctx_copied_d2d) {
        log_part("post_self_ctx_d2d");
    } else {
        const size_t expected_self_ctx = (size_t) state.seq * (size_t) m.n_heads * (size_t) m.head_dim;
        if (self_ctx_hseq.size() != expected_self_ctx) {
            std::fprintf(stderr,
                         "vla(lingbot_va): missing CUDA self-attn context block=%zu seq=%lld got=%zu expected=%zu\n",
                         block_index + 1, (long long) state.seq,
                         self_ctx_hseq.size(), expected_self_ctx);
            return false;
        }
        ggml_backend_tensor_set(post_graph->self_ctx, self_ctx_hseq.data(), 0, self_ctx_hseq.size() * sizeof(float));
        log_part("post_self_ctx_h2d");
    }
    const ggml_status st = ggml_backend_graph_compute(bw.backend, post_graph->graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA self-attn post graph failed (%d)\n", (int) st);
        return false;
    }
    log_part("post_graph_compute");
    if (block_index == 0 &&
        lingbot_env_enabled("VLA_LINGBOT_STREAM_BLOCK_TRACE") &&
        std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR") != nullptr) {
        static std::atomic<int> post_trace_counter{0};
        const int trace_idx = post_trace_counter.fetch_add(1);
        auto trace_env_i64 = [](const char * name, int64_t fallback) {
            const char * env = std::getenv(name);
            if (!env || !*env) return fallback;
            char * end = nullptr;
            const long long v = std::strtoll(env, &end, 10);
            return (end && *end == '\0') ? (int64_t) v : fallback;
        };
        const int64_t filter_mode = trace_env_i64("VLA_LINGBOT_STREAM_TRACE_MODE", INT64_MIN);
        const int64_t filter_seq = trace_env_i64("VLA_LINGBOT_STREAM_TRACE_SEQ", INT64_MIN);
        const int64_t filter_call_min = trace_env_i64("VLA_LINGBOT_STREAM_TRACE_CALL_MIN", INT64_MIN);
        const int64_t filter_call_max = trace_env_i64("VLA_LINGBOT_STREAM_TRACE_CALL_MAX", INT64_MAX);
        const bool dump_this =
            (filter_mode == INT64_MIN || filter_mode == state.cache_mode) &&
            (filter_seq == INT64_MIN || filter_seq == state.seq) &&
            (trace_idx >= filter_call_min && trace_idx <= filter_call_max);
        if (dump_this) {
            const std::string prefix = std::string(std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR")) +
                "/lingbot_predict_trace_cuda_block0_call_" + std::to_string(trace_idx) +
                "_seq" + std::to_string((long long) state.seq) +
                "_mode" + std::to_string(state.cache_mode);
            auto dump_tensor_2d = [&](const char * name, ggml_tensor * t) {
                if (!t) return;
                std::vector<float> data((size_t) t->ne[0] * (size_t) t->ne[1]);
                ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
                dump_f32_file(prefix + "_" + name + ".f32", data);
                dump_text_file(prefix + "_" + name + ".shape.txt",
                               std::to_string((long long) t->ne[1]) + " " +
                               std::to_string((long long) t->ne[0]) + "\n");
            };
            dump_tensor_2d("self_attn_out", post_graph->self_attn_out);
            dump_tensor_2d("gate_msa", post_graph->gate_msa);
            dump_tensor_2d("gated_self", post_graph->gated_self);
            dump_tensor_2d("post_self", post_graph->post_self);
            dump_tensor_2d("n2", post_graph->n2);
            dump_tensor_2d("cross_q", post_graph->cross_q);
            dump_tensor_2d("cross_k", post_graph->cross_k);
            dump_tensor_2d("cross_v", post_graph->cross_v);
            dump_tensor_2d("cross_ctx", post_graph->cross_ctx);
            dump_tensor_2d("cross_attn", post_graph->cross_attn);
            dump_tensor_2d("post_cross", post_graph->post_cross);
            dump_tensor_2d("n3", post_graph->n3);
            dump_tensor_2d("ff", post_graph->ff);
            dump_tensor_2d("post_out", post_graph->out);
        }
    }
    state.x.resize((size_t) post_graph->out->ne[0] * post_graph->out->ne[1]);
    if (device_x_handoff) {
        ggml_backend_synchronize(bw.backend);
        if (!lingbot_copy_tensor_to_device_x(state, bw.backend, post_graph->out, "post.out")) {
            return false;
        }
        log_part("post_output_device");
    } else {
        ggml_backend_tensor_get(post_graph->out, state.x.data(), 0, state.x.size() * sizeof(float));
        lingbot_clear_device_x(state);
        log_part("post_output_d2h");
    }
    if (timing) {
        const auto t_block1 = LingBotClock::now();
        std::printf("vla(lingbot_va): timing cuda-self-attn block=%zu seq=%lld mode=%d branch=%d total=%.3fms\n",
                    block_index + 1,
                    (long long) state.seq,
                    state.cache_mode,
                    state.cache_branch,
                    lingbot_elapsed_ms(t_block0, t_block1));
    }
    return true;
}
#endif

bool exec_block_window(
        LingBotTransformerExecutor & ex,
        LingBotRuntimeWeights & bw,
        size_t block_index,
        LingBotExecState & latent,
        LingBotExecState & action) {
    return exec_block_one(ex, bw, block_index, latent) &&
           exec_block_one(ex, bw, block_index, action);
}

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS

#endif

bool exec_output_stage(
        LingBotTransformerExecutor & ex,
        LingBotExecState & latent,
        LingBotExecState & action) {
    const LingBotVAModelArch & m = ex.m;
    LingBotRuntimeWeights & sw = ex.common;
    const std::string & dump_dir = ex.dump_dir;
    const int64_t hidden = m.cfg.hidden;
    ggml_init_params cp = { size_t(32) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) return false;

    struct OutputGraphIO {
        ggml_tensor * out = nullptr;
        ggml_tensor * x_in = nullptr;
        ggml_tensor * t_hidden = nullptr;
    };
    auto build_one = [&](bool action_mode, LingBotExecState & state) -> OutputGraphIO {
        OutputGraphIO io;
        io.x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
        io.t_hidden = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
        ggml_set_input(io.x_in);
        ggml_set_input(io.t_hidden);
        ggml_tensor * out_shift = ggml_add(C, io.t_hidden, ggml_view_1d(C, sw.output_scale_shift, hidden, 0));
        ggml_tensor * out_scale = ggml_add(C, io.t_hidden, ggml_view_1d(C, sw.output_scale_shift, hidden,
                                                                     (size_t) hidden * ggml_element_size(sw.output_scale_shift)));
        io.out = action_mode ? lin(C, sw.action_out, adaln(C, io.x_in, out_shift, out_scale, 1e-6f))
                             : lin(C, sw.output_proj, adaln(C, io.x_in, out_shift, out_scale, 1e-6f));
        ggml_set_output(io.out);
        return io;
    };
    auto latent_io = build_one(false, latent);
    auto action_io = build_one(true, action);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 8192, false);
    ggml_build_forward_expand(gf, latent_io.out);
    ggml_build_forward_expand(gf, action_io.out);

    ggml_gallocr_t galloc = lingbot_runtime_gallocr(sw.backend);
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }
    ggml_backend_tensor_set(latent_io.x_in, latent.x.data(), 0, latent.x.size() * sizeof(float));
    ggml_backend_tensor_set(latent_io.t_hidden, latent.t_hidden.data(), 0, latent.t_hidden.size() * sizeof(float));
    ggml_backend_tensor_set(action_io.x_in, action.x.data(), 0, action.x.size() * sizeof(float));
    ggml_backend_tensor_set(action_io.t_hidden, action.t_hidden.data(), 0, action.t_hidden.size() * sizeof(float));
    const ggml_status st = ggml_backend_graph_compute(sw.backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }

    auto report = [&](const char * label, ggml_tensor * out) {
        std::vector<float> out_h((size_t) out->ne[0] * out->ne[1]);
        ggml_backend_tensor_get(out, out_h.data(), 0, out_h.size() * sizeof(float));
        double checksum = 0.0;
        double max_abs = 0.0;
        for (float v : out_h) {
            checksum += (double) v;
            max_abs = std::max(max_abs, std::abs((double) v));
        }
        std::printf("vla(lingbot_va): streaming forward %s ok: out=[%lld,%lld] checksum=%g max_abs=%g\n",
                    label, (long long) out->ne[0], (long long) out->ne[1], checksum, max_abs);
        if (!dump_dir.empty()) {
            const std::string base = dump_dir + "/lingbot_runtime_" + label;
            dump_f32_file(base + ".f32", out_h);
            dump_text_file(base + ".shape.txt",
                           std::to_string((long long) out->ne[0]) + " " +
                           std::to_string((long long) out->ne[1]) + "\n");
        }
    };
    report("latent", latent_io.out);
    report("action", action_io.out);
    ggml_gallocr_free(galloc);
    ggml_free(C);
    return true;
}

bool exec_output_one(
        LingBotTransformerExecutor & ex,
        bool action_mode,
        LingBotExecState & state,
        std::vector<float> & out_h) {
    const LingBotVAModelArch & m = ex.m;
    LingBotRuntimeWeights & sw = ex.common;
    const int64_t hidden = m.cfg.hidden;

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    const bool disable_graph_cache = lingbot_env_enabled("VLA_LINGBOT_OUTPUT_ONE_GRAPH_CACHE_DISABLE");
    std::unique_ptr<LingBotOutputOneGraph> temp_graph;
    LingBotOutputOneGraph * cached = nullptr;
    if (!disable_graph_cache) {
        for (const auto & entry : ex.m.output_one_graphs) {
            if (entry && entry->matches(sw, action_mode, state.seq)) {
                cached = entry.get();
                break;
            }
        }
    }
    if (!cached) {
        auto built = std::make_unique<LingBotOutputOneGraph>();
        ggml_init_params cp = { size_t(16) * 1024 * 1024, nullptr, true };
        built->ctx = ggml_init(cp);
        if (!built->ctx) return false;
        ggml_context * C = built->ctx;

        built->weights = &sw;
        built->seq = state.seq;
        built->action_mode = action_mode;
        built->x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
        built->t_hidden = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
        ggml_set_input(built->x_in);
        ggml_set_input(built->t_hidden);
        ggml_tensor * out_shift = ggml_add(C, built->t_hidden, ggml_view_1d(C, sw.output_scale_shift, hidden, 0));
        ggml_tensor * out_scale = ggml_add(C, built->t_hidden, ggml_view_1d(C, sw.output_scale_shift, hidden,
                                                                             (size_t) hidden * ggml_element_size(sw.output_scale_shift)));
        built->out = action_mode ? lin(C, sw.action_out, adaln(C, built->x_in, out_shift, out_scale, 1e-6f))
                                 : lin(C, sw.output_proj, adaln(C, built->x_in, out_shift, out_scale, 1e-6f));
        ggml_set_output(built->out);

        built->graph = ggml_new_graph_custom(C, 8192, false);
        ggml_build_forward_expand(built->graph, built->out);
        built->galloc = lingbot_runtime_gallocr(sw.backend);
        if (!built->galloc || !ggml_gallocr_alloc_graph(built->galloc, built->graph)) {
            return false;
        }
        cached = built.get();
        if (disable_graph_cache) {
            temp_graph = std::move(built);
        } else {
            ex.m.output_one_graphs.push_back(std::move(built));
            std::printf("vla(lingbot_va): cached output-one graph action=%d seq=%lld\n",
                        action_mode ? 1 : 0, (long long) state.seq);
        }
    }

    if (state.allow_device_x_handoff) {
        if (!lingbot_device_x_valid(state, hidden, sw.backend)) {
            std::fprintf(stderr,
                         "vla(lingbot_va): output head requested device-x handoff but no valid device tensor is available seq=%lld\n",
                         (long long) state.seq);
            return false;
        }
        if (!lingbot_copy_device_x_to_tensor(state, cached->x_in, "output.x_in")) {
            return false;
        }
    } else {
        ggml_backend_tensor_set(cached->x_in, state.x.data(), 0, state.x.size() * sizeof(float));
    }
    ggml_backend_tensor_set(cached->t_hidden, state.t_hidden.data(), 0, state.t_hidden.size() * sizeof(float));
    const ggml_status st = ggml_backend_graph_compute(sw.backend, cached->graph);
    if (st != GGML_STATUS_SUCCESS) {
        return false;
    }
    bool device_out_ready = false;
    if (!lingbot_env_enabled("VLA_LINGBOT_OUTPUT_DEVICE_OUT_DISABLE")) {
        if (!lingbot_copy_tensor_to_device_out(state, sw.backend, cached->out, "output.out")) {
            return false;
        }
        device_out_ready = true;
    } else {
        lingbot_clear_device_out(state);
    }
    if (state.prefer_device_out_only && device_out_ready) {
        out_h.clear();
        return true;
    }
    out_h.resize((size_t) cached->out->ne[0] * cached->out->ne[1]);
    ggml_backend_tensor_get(cached->out, out_h.data(), 0, out_h.size() * sizeof(float));
    return true;
#else
    ggml_init_params cp = { size_t(16) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) return false;

    ggml_tensor * x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
    ggml_tensor * t_hidden = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
    ggml_set_input(x_in);
    ggml_set_input(t_hidden);
    ggml_tensor * out_shift = ggml_add(C, t_hidden, ggml_view_1d(C, sw.output_scale_shift, hidden, 0));
    ggml_tensor * out_scale = ggml_add(C, t_hidden, ggml_view_1d(C, sw.output_scale_shift, hidden,
                                                                 (size_t) hidden * ggml_element_size(sw.output_scale_shift)));
    ggml_tensor * out = action_mode ? lin(C, sw.action_out, adaln(C, x_in, out_shift, out_scale, 1e-6f))
                                    : lin(C, sw.output_proj, adaln(C, x_in, out_shift, out_scale, 1e-6f));
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 8192, false);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t galloc = lingbot_runtime_gallocr(sw.backend);
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    if (state.allow_device_x_handoff) {
        if (!lingbot_device_x_valid(state, hidden, sw.backend)) {
            std::fprintf(stderr,
                         "vla(lingbot_va): output head requested device-x handoff but no valid device tensor is available seq=%lld\n",
                         (long long) state.seq);
            ggml_gallocr_free(galloc);
            ggml_free(C);
            return false;
        }
        if (!lingbot_copy_device_x_to_tensor(state, x_in, "output.x_in")) {
            ggml_gallocr_free(galloc);
            ggml_free(C);
            return false;
        }
    } else
#endif
    {
        ggml_backend_tensor_set(x_in, state.x.data(), 0, state.x.size() * sizeof(float));
    }
    ggml_backend_tensor_set(t_hidden, state.t_hidden.data(), 0, state.t_hidden.size() * sizeof(float));
    const ggml_status st = ggml_backend_graph_compute(sw.backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }
    out_h.resize((size_t) out->ne[0] * out->ne[1]);
    ggml_backend_tensor_get(out, out_h.data(), 0, out_h.size() * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(C);
    return true;
#endif
}

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
bool exec_official_runtime_kv_full_forward_fused_post_qkv(
        LingBotTransformerExecutor & ex,
        LingBotRuntimeWeights & bw,
        bool action_mode,
        LingBotExecState & state,
        int blocks,
        std::vector<float> & out_h,
        uint64_t cache_session_id,
        int cache_mode,
        LingBotCudaSelfAttnForwardRunner * forward_runner) {
    const LingBotVAModelArch & m = ex.m;
    const int64_t hidden = m.cfg.hidden;
    const int64_t text_seq = state.text_seq > 0 ? state.text_seq : 2;
    const bool use_bf16_boundary =
        lingbot_env_enabled("VLA_LINGBOT_BLOCK_BF16_BOUNDARY") &&
        !lingbot_env_enabled("VLA_LINGBOT_STATE_F32");
    if (!forward_runner || blocks < 2 || !state.allow_device_x_handoff || !state.allow_device_self_ctx_handoff) {
        return false;
    }
    if ((int) forward_runner->block_runners.size() < blocks) {
        return false;
    }
    if (lingbot_env_enabled("VLA_LINGBOT_STREAM_BLOCK_TRACE") ||
        std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR") != nullptr ||
        lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_CUDA_SELF_ATTN_GRAPH_CACHE_DISABLE")) {
        return false;
    }

    LingBotCudaSelfAttnQkvGraph * qkv_graph = lingbot_get_or_build_cuda_self_attn_qkv_graph(
        ex.m, bw, 0, state.seq, forward_runner->block_runners[0]);
    if (!qkv_graph ||
        !lingbot_compute_cuda_self_attn_qkv_graph(bw, state, qkv_graph, nullptr)) {
        return false;
    }

    ggml_tensor * qh = qkv_graph->qh;
    ggml_tensor * kh = qkv_graph->kh;
    ggml_tensor * vh = qkv_graph->vh;
    // Block0 post must read the pre-QKV device_x staged by the old successful path,
    // not qkv_graph->x_in after the QKV graph has executed.
    const ggml_tensor * current_x = nullptr;
    int max_fused_pairs = blocks - 1;
    if (const char * env = std::getenv("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FUSED_POST_QKV_PAIRS")) {
        char * end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end && *end == '\0') {
            max_fused_pairs = (int) std::max<long>(0, std::min<long>(v, blocks - 1));
        } else {
            std::fprintf(stderr,
                         "vla(lingbot_va): ignoring invalid "
                         "VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FUSED_POST_QKV_PAIRS='%s'\n",
                         env);
        }
    }

    int fused_pairs_done = 0;
    for (int b = 0; b + 1 < blocks && b < max_fused_pairs; ++b) {
        state.cache_session_id = cache_session_id;
        state.cache_mode = cache_mode;
        state.cache_branch = ex.cache_branch;
        state.cache_block_index = b;

        LingBotRuntimeKVDeviceResult self_ctx_device;
        if (!lingbot_runtime_kv_self_ctx_from_qkv_tensors(
                bw, m, state,
                forward_runner->cuda_self_table,
                forward_runner->cuda_token_mask,
                qh, kh, vh,
                self_ctx_device)) {
            return false;
        }

        const bool static_text_cache_enabled =
            lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_TEXT_CACHE") &&
            !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_TEXT_CACHE_DISABLE") &&
            state.text_signature != 0 &&
            (state.cache_branch == 0 || state.cache_branch == 1);
        const int post_text_slot = static_text_cache_enabled ? state.cache_branch : -1;
        LingBotCudaSelfAttnPostQkvGraph * post_qkv_graph =
            lingbot_get_cuda_self_attn_post_qkv_graph(
                ex.m, bw, (size_t) b, state.seq, text_seq, use_bf16_boundary,
                post_text_slot, true, false, false, forward_runner->block_runners[(size_t) b]);
        if (!post_qkv_graph ||
            !lingbot_compute_cuda_self_attn_post_qkv_graph(
                bw, state, post_qkv_graph, current_x, self_ctx_device)) {
            return false;
        }
        current_x = post_qkv_graph->post_out;
        if (lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FUSED_POST_QKV_POST_ONLY_DEBUG")) {
            fused_pairs_done = b + 1;
            break;
        }
        qh = post_qkv_graph->next_qh;
        kh = post_qkv_graph->next_kh;
        vh = post_qkv_graph->next_vh;
        fused_pairs_done = b + 1;
    }

    const bool fused_final_post_output = lingbot_official_final_post_output_fusion_enabled();
    if ((fused_final_post_output ||
         lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FUSED_FINAL_POST")) &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FUSED_FINAL_POST_DISABLE") &&
        fused_pairs_done == blocks - 1 &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FUSED_POST_QKV_POST_ONLY_DEBUG")) {
        const int b = blocks - 1;
        state.cache_session_id = cache_session_id;
        state.cache_mode = cache_mode;
        state.cache_branch = ex.cache_branch;
        state.cache_block_index = b;

        LingBotRuntimeKVDeviceResult self_ctx_device;
        if (!lingbot_runtime_kv_self_ctx_from_qkv_tensors(
                bw, m, state,
                forward_runner->cuda_self_table,
                forward_runner->cuda_token_mask,
                qh, kh, vh,
                self_ctx_device)) {
            return false;
        }

        const bool static_text_cache_enabled =
            lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_TEXT_CACHE") &&
            !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_STATIC_TEXT_CACHE_DISABLE") &&
            state.text_signature != 0 &&
            (state.cache_branch == 0 || state.cache_branch == 1);
        const int post_text_slot = static_text_cache_enabled ? state.cache_branch : -1;
        LingBotCudaSelfAttnPostQkvGraph * final_post_graph =
            lingbot_get_cuda_self_attn_post_qkv_graph(
                ex.m, bw, (size_t) b, state.seq, text_seq, use_bf16_boundary,
                post_text_slot, false, fused_final_post_output, action_mode,
                forward_runner->block_runners[(size_t) b]);
        if (!final_post_graph ||
            !lingbot_compute_cuda_self_attn_post_qkv_graph(
                bw, state, final_post_graph, current_x, self_ctx_device)) {
            return false;
        }
        if (fused_final_post_output) {
            ggml_tensor * out = final_post_graph->output_out;
            if (!out) return false;
            bool device_out_ready = false;
            if (!lingbot_env_enabled("VLA_LINGBOT_OUTPUT_DEVICE_OUT_DISABLE")) {
                if (!lingbot_copy_tensor_to_device_out(state, bw.backend, out, "fused_final_post_output.out")) {
                    return false;
                }
                device_out_ready = true;
            } else {
                lingbot_clear_device_out(state);
            }
            if (state.prefer_device_out_only && device_out_ready) {
                out_h.clear();
                return true;
            }
            out_h.resize((size_t) out->ne[0] * out->ne[1]);
            ggml_backend_tensor_get(out, out_h.data(), 0, out_h.size() * sizeof(float));
            return true;
        }
        current_x = final_post_graph->post_out;
        fused_pairs_done = blocks;
    }

    if (fused_pairs_done > 0) {
        ggml_backend_synchronize(bw.backend);
        state.device_x = std::make_shared<LingBotExecState::CudaDeviceBuffer>();
        if (!lingbot_copy_tensor_to_device_x(state, bw.backend, const_cast<ggml_tensor *>(current_x),
                                             "fused_post_qkv.current_x")) {
            return false;
        }
    }
    for (int b = fused_pairs_done; b < blocks; ++b) {
        state.cache_session_id = cache_session_id;
        state.cache_mode = cache_mode;
        state.cache_branch = ex.cache_branch;
        state.cache_block_index = b;
        if (!exec_block_one_cuda_self_attn(ex, bw, (size_t) b,
                                           state,
                                           forward_runner->cuda_meta,
                                           forward_runner->cuda_self_table,
                                           forward_runner->cuda_token_mask,
                                           forward_runner->block_runners[(size_t) b])) {
            return false;
        }
    }
    if (state.allow_device_x_handoff && !lingbot_device_x_valid(state, hidden, bw.backend)) {
        std::fprintf(stderr,
                     "vla(lingbot_va): fused post+QKV forward lost final device-x seq=%lld\n",
                     (long long) state.seq);
        return false;
    }
    return exec_output_one(ex, action_mode, state, out_h);
}

bool exec_official_runtime_kv_full_forward(
        LingBotTransformerExecutor & ex,
        LingBotRuntimeWeights & bw,
        bool action_mode,
        LingBotExecState & state,
        int blocks,
        std::vector<float> & out_h,
        uint64_t cache_session_id,
        int cache_mode,
        LingBotCudaSelfAttnForwardRunner * forward_runner_hint = nullptr) {
    const LingBotVAModelArch & m = ex.m;
    const int64_t hidden = m.cfg.hidden;
    if (cache_session_id == 0) {
        std::fprintf(stderr,
                     "vla(lingbot_va): official runtime-KV full forward requires a nonzero cache session\n");
        return false;
    }
    if (blocks != (int) m.n_layers && !lingbot_env_enabled("VLA_LINGBOT_ALLOW_PARTIAL_BLOCKS")) {
        std::fprintf(stderr,
                     "vla(lingbot_va): official runtime-KV full forward requires the complete model "
                     "(requested=%d model_layers=%lld); set VLA_LINGBOT_ALLOW_PARTIAL_BLOCKS=1 only for debug\n",
                     blocks, (long long) m.n_layers);
        return false;
    }
    if ((int64_t) bw.blocks.size() < blocks) {
        std::fprintf(stderr,
                     "vla(lingbot_va): official runtime-KV full forward requires resident full weights "
                     "(have=%zu need=%d)\n",
                     bw.blocks.size(), blocks);
        return false;
    }

    const bool trace_enabled =
        lingbot_env_enabled("VLA_LINGBOT_STREAM_BLOCK_TRACE") ||
        std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR") != nullptr;
    const bool graph_cache_disabled =
        lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_CUDA_SELF_ATTN_GRAPH_CACHE_DISABLE");
    const bool host_boundary =
        lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FULL_FORWARD_HOST_BOUNDARY");
    const bool use_bf16_boundary =
        lingbot_env_enabled("VLA_LINGBOT_BLOCK_BF16_BOUNDARY") &&
        !lingbot_env_enabled("VLA_LINGBOT_STATE_F32");
    state.allow_device_x_handoff =
        !trace_enabled &&
        !graph_cache_disabled &&
        !host_boundary &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FULL_FORWARD_DISABLE_DEVICE_X");
    state.allow_device_self_ctx_handoff =
        !trace_enabled &&
        !host_boundary &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FULL_FORWARD_DISABLE_SELF_CTX_D2D");
    if (!state.allow_device_x_handoff || !lingbot_device_x_valid(state, hidden, bw.backend)) {
        lingbot_clear_device_x(state);
    }

    LingBotCudaSelfAttnForwardRunner * forward_runner = nullptr;
    const int64_t text_seq = state.text_seq > 0 ? state.text_seq : 2;
    LingBotFlexMaskMeta local_cuda_meta;
    LingBotBlockSparseTable local_cuda_self_table;
    std::vector<unsigned char> local_cuda_token_mask;
    const LingBotFlexMaskMeta * cuda_meta = nullptr;
    const LingBotBlockSparseTable * cuda_self_table = nullptr;
    const std::vector<unsigned char> * cuda_token_mask = nullptr;
    if (!graph_cache_disabled) {
        if (forward_runner_hint &&
            forward_runner_hint->matches(bw, blocks, state.seq, text_seq, use_bf16_boundary)) {
            forward_runner = forward_runner_hint;
        } else {
            forward_runner = lingbot_get_cuda_self_attn_forward_runner(
                ex.m, bw, blocks, state.seq, text_seq, use_bf16_boundary);
        }
        if (!forward_runner || (int) forward_runner->block_runners.size() < blocks) {
            std::fprintf(stderr,
                         "vla(lingbot_va): official runtime-KV forward runner bind failed blocks=%d seq=%lld\n",
                         blocks, (long long) state.seq);
            return false;
        }
        cuda_meta = &forward_runner->cuda_meta;
        cuda_self_table = &forward_runner->cuda_self_table;
        cuda_token_mask = &forward_runner->cuda_token_mask;
    } else {
        local_cuda_meta.seq_ids.assign((size_t) state.seq, 0);
        local_cuda_self_table = build_dense_block_table(state.seq, state.seq, 64);
        local_cuda_token_mask.assign((size_t) state.seq * (size_t) state.seq, (unsigned char) 1);
        cuda_meta = &local_cuda_meta;
        cuda_self_table = &local_cuda_self_table;
        cuda_token_mask = &local_cuda_token_mask;
    }

    if (lingbot_env_enabled("VLA_LINGBOT_KV_VERBOSE")) {
        std::printf("vla(lingbot_va): official runtime-KV full forward seq=%lld blocks=%d branch=%d "
                    "mode=%d device_x=%d action=%d forward_runner=%d\n",
                    (long long) state.seq,
                    blocks,
                    ex.cache_branch,
                    cache_mode,
                    state.allow_device_x_handoff ? 1 : 0,
                    action_mode ? 1 : 0,
                    forward_runner ? 1 : 0);
    }

    if (lingbot_official_post_qkv_fusion_enabled()) {
        std::printf("vla(lingbot_va): official runtime-KV fused post+next-QKV forward enabled "
                    "seq=%lld blocks=%d mode=%d branch=%d action=%d\n",
                    (long long) state.seq,
                    blocks,
                    cache_mode,
                    ex.cache_branch,
                    action_mode ? 1 : 0);
        return exec_official_runtime_kv_full_forward_fused_post_qkv(
            ex, bw, action_mode, state, blocks, out_h,
            cache_session_id, cache_mode, forward_runner);
    }

    for (int b = 0; b < blocks; ++b) {
        state.cache_session_id = cache_session_id;
        state.cache_mode = cache_mode;
        state.cache_branch = ex.cache_branch;
        state.cache_block_index = b;
        LingBotCudaSelfAttnBlockRunner * block_runner =
            forward_runner ? forward_runner->block_runners[(size_t) b] : nullptr;
        if (!exec_block_one_cuda_self_attn(ex, bw, (size_t) b,
                                           state, *cuda_meta, *cuda_self_table, *cuda_token_mask,
                                           block_runner)) {
            return false;
        }
        if (state.allow_device_x_handoff && b + 1 < blocks &&
            !lingbot_device_x_valid(state, hidden, bw.backend)) {
            std::fprintf(stderr,
                         "vla(lingbot_va): official runtime-KV full forward lost device-x handoff "
                         "after block=%d seq=%lld\n",
                         b + 1, (long long) state.seq);
            return false;
        }
    }
    return exec_output_one(ex, action_mode, state, out_h);
}
#endif

bool exec_forward_one_streaming(
        LingBotTransformerExecutor & ex,
        LingBotRuntimeWeights & bw,
        bool action_mode,
        const std::vector<float> * raw_input,
        double timestep,
        int blocks,
        std::vector<float> & out_h,
        const LingBotGridSpec * grid_spec = nullptr,
        const std::vector<double> * token_timesteps = nullptr,
        uint64_t cache_session_id = 0,
        int cache_mode = 0,
        bool require_runtime_kv_self_attn = false,
        bool prefer_device_out_only = false,
        const LingBotDeviceTensor * device_raw_input = nullptr,
        LingBotExecState * out_state = nullptr
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
        ,
        LingBotCudaSelfAttnForwardRunner * official_forward_runner_hint = nullptr
#endif
        ) {
    LingBotExecState state;
    if (!exec_embedding_stage(ex, action_mode, state, raw_input, timestep, grid_spec, token_timesteps,
                              device_raw_input)) {
        return false;
    }
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    state.prefer_device_out_only = prefer_device_out_only;
    const bool use_cuda_self_attn =
        lingbot_predict_cuda_self_attn_enabled() || require_runtime_kv_self_attn;
    state.allow_device_x_handoff =
        use_cuda_self_attn &&
        lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_CUDA_SELF_ATTN_DEVICE_X") &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_CUDA_SELF_ATTN_DEVICE_X_DISABLE") &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_CUDA_SELF_ATTN_GRAPH_CACHE_DISABLE") &&
        !lingbot_env_enabled("VLA_LINGBOT_STREAM_BLOCK_TRACE") &&
        std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR") == nullptr;
    lingbot_clear_device_x(state);
    LingBotFlexMaskMeta cuda_meta;
    LingBotBlockSparseTable cuda_self_table;
    std::vector<unsigned char> cuda_token_mask;
    if (use_cuda_self_attn) {
        if (cache_session_id == 0) {
            std::fprintf(stderr,
                         "vla(lingbot_va): VLA_LINGBOT_CUDA_SELF_ATTN_STREAM separate path "
                         "requires a nonzero cache session\n");
            return false;
        }
        cuda_meta.seq_ids.assign((size_t) state.seq, 0);
        cuda_self_table = build_dense_block_table(state.seq, state.seq, 64);
        cuda_token_mask.assign((size_t) state.seq * (size_t) state.seq, (unsigned char) 1);
        if (lingbot_env_enabled("VLA_LINGBOT_KV_VERBOSE")) {
            std::printf("vla(lingbot_va): CUDA self-attn streaming uses dense Python-style KV cache seq=%lld mode=%d\n",
                        (long long) state.seq, cache_mode);
        }
        if (require_runtime_kv_self_attn &&
            !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FULL_FORWARD_DISABLE")) {
            const bool ok = exec_official_runtime_kv_full_forward(ex, bw, action_mode, state, blocks,
                                                                  out_h, cache_session_id, cache_mode,
                                                                  official_forward_runner_hint);
            if (ok && out_state) *out_state = state;
            return ok;
        }
    }
#else
    const bool use_cuda_self_attn = false;
    if (require_runtime_kv_self_attn) {
        std::fprintf(stderr,
                     "vla(lingbot_va): official-order runtime KV self-attn requires "
                     "LingBot CUDA flex kernels; refusing dense no-cache fallback\n");
        return false;
    }
    if (lingbot_predict_cuda_self_attn_enabled()) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA self-attn requested but LingBot CUDA kernels were not built\n");
        return false;
    }
#endif
    const bool trace_enabled = lingbot_env_enabled("VLA_LINGBOT_STREAM_BLOCK_TRACE");
    const bool can_use_block_sequence =
        !use_cuda_self_attn &&
        !trace_enabled &&
        lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_BLOCK_SEQUENCE") &&
        !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_BLOCK_SEQUENCE_DISABLE") &&
        (int64_t) ex.common.blocks.size() >= blocks;
    if (can_use_block_sequence) {
        int64_t block_window = blocks;
        if (const char * env = std::getenv("VLA_LINGBOT_OFFICIAL_BLOCK_SEQUENCE_BLOCKS")) {
            char * end = nullptr;
            const long v = std::strtol(env, &end, 10);
            if (end && *end == '\0' && v > 0) {
                block_window = std::min<int64_t>((int64_t) v, blocks);
            } else {
                std::fprintf(stderr,
                             "vla(lingbot_va): ignoring invalid "
                             "VLA_LINGBOT_OFFICIAL_BLOCK_SEQUENCE_BLOCKS='%s'\n",
                             env);
            }
        }
        for (int64_t start = 0; start < blocks; start += block_window) {
            const int64_t count = std::min<int64_t>(block_window, blocks - start);
            state.cache_session_id = cache_session_id;
            state.cache_mode = cache_mode;
            state.cache_branch = ex.cache_branch;
            state.cache_block_index = start;
            if (!exec_block_sequence(ex, ex.common, start, count, state)) {
                return false;
            }
        }
    } else for (int b = 0; b < blocks; ++b) {
        state.cache_session_id = cache_session_id;
        state.cache_mode = cache_mode;
        state.cache_branch = ex.cache_branch;
        state.cache_block_index = b;
        std::string trace_label;
        const char * trace_label_c = nullptr;
        if (b == 0 && lingbot_env_enabled("VLA_LINGBOT_STREAM_BLOCK_TRACE")) {
            static std::atomic<int> stream_trace_counter{0};
            const int trace_idx = stream_trace_counter.fetch_add(1);
            trace_label = std::string(action_mode ? "action" : "latent") + "_" + std::to_string(trace_idx);
            trace_label_c = trace_label.c_str();
        }
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
        const bool ok = use_cuda_self_attn
            ? exec_block_one_cuda_self_attn(ex, bw, (size_t) b, state, cuda_meta, cuda_self_table, cuda_token_mask)
            : exec_block_one(ex, bw, (size_t) b, state, trace_label_c);
#else
        const bool ok = exec_block_one(ex, bw, (size_t) b, state, trace_label_c);
#endif
        if (!ok) {
            return false;
        }
    }
    const bool ok = exec_output_one(ex, action_mode, state, out_h);
    if (ok && out_state) *out_state = state;
    return ok;
}





int env_i32(const char * name, int fallback);
float env_f32(const char * name, float fallback);
std::vector<int32_t> env_i32_list(const char * name, std::vector<int32_t> fallback);



int env_i32(const char * name, int fallback) {
    if (const char * v = std::getenv(name)) {
        char * end = nullptr;
        const long x = std::strtol(v, &end, 10);
        if (end && *end == '\0' && x > 0) {
            return (int) x;
        }
        std::fprintf(stderr, "vla(lingbot_va): ignoring invalid %s='%s'\n", name, v);
    }
    return fallback;
}

float env_f32(const char * name, float fallback) {
    if (const char * v = std::getenv(name)) {
        char * end = nullptr;
        const float x = std::strtof(v, &end);
        if (end && *end == '\0' && std::isfinite(x) && x > 0.0f) {
            return x;
        }
        std::fprintf(stderr, "vla(lingbot_va): ignoring invalid %s='%s'\n", name, v);
    }
    return fallback;
}

std::vector<int32_t> env_i32_list(const char * name, std::vector<int32_t> fallback) {
    const char * v = std::getenv(name);
    if (!v || !*v) return fallback;
    std::vector<int32_t> out;
    const char * p = v;
    while (*p) {
        char * end = nullptr;
        const long x = std::strtol(p, &end, 10);
        if (end == p || x < 0 || x > std::numeric_limits<int32_t>::max()) {
            std::fprintf(stderr, "vla(lingbot_va): ignoring invalid %s='%s'\n", name, v);
            return fallback;
        }
        out.push_back((int32_t) x);
        p = end;
        if (*p == ',') {
            ++p;
        } else if (*p != '\0') {
            std::fprintf(stderr, "vla(lingbot_va): ignoring invalid %s='%s'\n", name, v);
            return fallback;
        }
    }
    return out.empty() ? fallback : out;
}

std::vector<int> lingbot_unit_time_chunks(int T) {
    std::vector<int> chunks;
    chunks.reserve((size_t) std::max(T, 0));
    for (int i = 0; i < T; ++i) chunks.push_back(1);
    return chunks;
}

bool lingbot_action_postprocess_libero_enabled() {
    const char * mode = std::getenv("VLA_LINGBOT_ACTION_POSTPROCESS");
    return !(mode && (std::strcmp(mode, "raw") == 0 ||
                      std::strcmp(mode, "none") == 0 ||
                      std::strcmp(mode, "0") == 0));
}

bool lingbot_predict_official_order_enabled() {
    return true;
}

int64_t lingbot_runtime_action_dim(const Config & cfg) {
    return lingbot_action_postprocess_libero_enabled() ? cfg.real_action_dim : cfg.max_action_dim;
}

bool postprocess_libero_action_tokens(
        const std::vector<float> & action_tokens,
        int64_t action_feature,
        int64_t action_seq,
        int64_t n_suffix,
        std::vector<float> & out) {
    static constexpr int k_used_dim = 7;
    static constexpr int k_used_ids[k_used_dim] = {0, 1, 2, 3, 4, 5, 6};
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

    if (action_feature <= 0 || action_seq <= 0 || n_suffix <= 0) return false;
    for (int i = 0; i < k_used_dim; ++i) {
        if (k_used_ids[i] >= action_feature) {
            std::fprintf(stderr,
                         "vla(lingbot_va): LIBERO action channel %d exceeds action_feature=%lld\n",
                         k_used_ids[i], (long long) action_feature);
            return false;
        }
    }

    const int64_t steps = std::min<int64_t>(n_suffix, action_seq);
    out.assign((size_t) n_suffix * (size_t) k_used_dim, 0.0f);
    for (int64_t t = 0; t < steps; ++t) {
        for (int c = 0; c < k_used_dim; ++c) {
            const float x = action_tokens[(size_t) k_used_ids[c] +
                                          (size_t) action_feature * (size_t) t];
            out[(size_t) t * (size_t) k_used_dim + (size_t) c] =
                (x + 1.0f) * 0.5f * (k_q99[c] - k_q01[c] + 1.0e-6f) + k_q01[c];
        }
    }
    return true;
}

bool preprocess_libero_state_to_action_condition(
        const float * state,
        int64_t state_dim,
        const LingBotTensor5DShape & action_shape,
        std::vector<float> & action_cond) {
    static constexpr int k_used_dim = 7;
    static constexpr int k_action_dim = 30;
    static constexpr int k_used_ids[k_used_dim] = {0, 1, 2, 3, 4, 5, 6};
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
    if (!state || state_dim < k_used_dim || action_shape.c != k_action_dim ||
        action_shape.b != 1 || action_shape.f <= 0 || action_shape.h <= 0 ||
        action_shape.w != 1) {
        return false;
    }
    action_cond.assign((size_t) action_shape.b * (size_t) action_shape.c *
                       (size_t) action_shape.f * (size_t) action_shape.h *
                       (size_t) action_shape.w, 0.0f);
    for (int i = 0; i < k_used_dim; ++i) {
        const int c = k_used_ids[i];
        const float denom = k_q99[i] - k_q01[i] + 1.0e-6f;
        const float norm = (state[i] - k_q01[i]) / denom * 2.0f - 1.0f;
        for (int64_t h = 0; h < action_shape.h; ++h) {
            action_cond[idx5(action_shape, 0, c, 0, h, 0)] = norm;
        }
    }
    return true;
}

bool preprocess_libero_action_history_to_action_condition(
        const float * action_history,
        int64_t c_in,
        int64_t f_in,
        int64_t h_in,
        const LingBotTensor5DShape & action_shape,
        std::vector<float> & action_cond) {
    static constexpr int k_used_dim = 7;
    static constexpr int k_action_dim = 30;
    static constexpr int k_inverse_ids[k_action_dim] = {
        0, 1, 2, 3, 4, 5, 6,
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7, 7,
        7, 7, 7, 7, 7, 7, 7,
    };
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
    if (!action_history || c_in != k_used_dim || f_in <= 0 || h_in <= 0 ||
        action_shape.b != 1 || action_shape.c != k_action_dim ||
        action_shape.f <= 0 || action_shape.h <= 0 || action_shape.w != 1) {
        return false;
    }
    action_cond.assign((size_t) action_shape.b * (size_t) action_shape.c *
                       (size_t) action_shape.f * (size_t) action_shape.h *
                       (size_t) action_shape.w, 0.0f);
    const int64_t copy_f = std::min<int64_t>(f_in, action_shape.f);
    const int64_t copy_h = std::min<int64_t>(h_in, action_shape.h);
    for (int64_t c = 0; c < action_shape.c; ++c) {
        const int src_c = k_inverse_ids[c];
        const bool padded_channel = src_c >= k_used_dim;
        const float denom = padded_channel ? 1.0e-6f : (k_q99[src_c] - k_q01[src_c] + 1.0e-6f);
        for (int64_t f = 0; f < copy_f; ++f) {
            for (int64_t h = 0; h < copy_h; ++h) {
                float norm = -1.0f;
                if (!padded_channel) {
                    const size_t src = (size_t) src_c * (size_t) f_in * (size_t) h_in +
                                       (size_t) f * (size_t) h_in + (size_t) h;
                    norm = (action_history[src] - k_q01[src_c]) / denom * 2.0f - 1.0f;
                }
                action_cond[idx5(action_shape, 0, c, f, h, 0)] = norm;
            }
        }
    }
    return true;
}

void zero_unused_libero_action_channels(
        std::vector<float> & action_sample,
        const LingBotTensor5DShape & action_shape) {
    static constexpr int k_used_dim = 7;
    if (action_shape.b != 1 || action_shape.c <= k_used_dim || action_shape.w != 1) return;
    for (int64_t c = k_used_dim; c < action_shape.c; ++c) {
        for (int64_t f = 0; f < action_shape.f; ++f) {
            for (int64_t h = 0; h < action_shape.h; ++h) {
                action_sample[idx5(action_shape, 0, c, f, h, 0)] = 0.0f;
            }
        }
    }
}

void apply_action_condition_frame0(
        std::vector<float> & action_sample,
        const std::vector<float> & action_cond,
        const LingBotTensor5DShape & action_shape) {
    if (action_cond.size() != action_sample.size() || action_shape.f <= 0) return;
    for (int64_t c = 0; c < action_shape.c; ++c) {
        for (int64_t h = 0; h < action_shape.h; ++h) {
            action_sample[idx5(action_shape, 0, c, 0, h, 0)] =
                action_cond[idx5(action_shape, 0, c, 0, h, 0)];
        }
    }
}

void finalize_libero_action_sample(
        std::vector<float> & action_sample,
        const std::vector<float> * action_cond,
        const LingBotTensor5DShape & action_shape,
        bool bf16_state) {
    static constexpr int k_used_dim = 7;
    const bool has_cond =
        action_cond &&
        action_cond->size() == action_sample.size() &&
        action_shape.f > 0;
    for (int64_t c = 0; c < action_shape.c; ++c) {
        for (int64_t f = 0; f < action_shape.f; ++f) {
            for (int64_t h = 0; h < action_shape.h; ++h) {
                for (int64_t w = 0; w < action_shape.w; ++w) {
                    const size_t idx = idx5(action_shape, 0, c, f, h, w);
                    float v = action_sample[idx];
                    if (has_cond && f == 0) {
                        v = (*action_cond)[idx];
                    }
                    if (action_shape.b == 1 && action_shape.c > k_used_dim &&
                        action_shape.w == 1 && c >= k_used_dim) {
                        v = 0.0f;
                    }
                    if (bf16_state) {
                        v = lingbot_bf16_bits_to_f32(lingbot_f32_to_bf16_bits_rne(v));
                    }
                    action_sample[idx] = v;
                }
            }
        }
    }
}

void restore_latent_condition_round(
        std::vector<float> & latent_sample,
        const LingBotTensor5DShape & latent_shape,
        const std::vector<float> & latent_condition,
        const LingBotTensor5DShape & latent_condition_shape,
        bool bf16_state) {
    const bool has_condition =
        !latent_condition.empty() &&
        latent_condition_shape.f > 0 &&
        latent_condition_shape.c == latent_shape.c &&
        latent_condition_shape.h == latent_shape.h &&
        latent_condition_shape.w == latent_shape.w;
    for (int64_t c = 0; c < latent_shape.c; ++c) {
        for (int64_t f = 0; f < latent_shape.f; ++f) {
            for (int64_t h = 0; h < latent_shape.h; ++h) {
                for (int64_t w = 0; w < latent_shape.w; ++w) {
                    const size_t idx = idx5(latent_shape, 0, c, f, h, w);
                    float v = latent_sample[idx];
                    if (has_condition && f < latent_condition_shape.f) {
                        v = latent_condition[idx5(latent_condition_shape, 0, c, f, h, w)];
                    }
                    if (bf16_state) {
                        v = lingbot_bf16_bits_to_f32(lingbot_f32_to_bf16_bits_rne(v));
                    }
                    latent_sample[idx] = v;
                }
            }
        }
    }
}



struct LingBotFlowBranchConfig {
    const char * label = nullptr;
    bool action_mode = false;
    int steps = 0;
    double shift = 1.0;
    int64_t input_dim = 0;
    int64_t seq = 0;
    LingBotGridSpec grid;
    LingBotTensor5DShape tensor_shape;
};







std::vector<float> LingBotVAModelArch::predict(const Inputs& in) {
    LingBotScopedTimer predict_timer("predict_total");
    const auto t_total0 = std::chrono::steady_clock::now();
    stats = {};
    double text_encode_ms = 0.0;
    double vae_encode_ms = 0.0;

    int blocks = (int) n_layers;
    if (const char * env = std::getenv("VLA_LINGBOT_PREDICT_BLOCKS")) {
        char * end = nullptr;
        const long v = std::strtol(env, &end, 10);
        if (end && *end == '\0' && v > 0) {
            blocks = (int) std::min<int64_t>(v, n_layers);
        } else {
            std::fprintf(stderr, "vla(lingbot_va): ignoring invalid VLA_LINGBOT_PREDICT_BLOCKS='%s'\n", env);
        }
    }
    const char * env_type_c = (in.lingbot_env_type && *in.lingbot_env_type)
        ? in.lingbot_env_type
        : std::getenv("VLA_LINGBOT_ENV_TYPE");
    const std::string lingbot_env_type = env_type_c && *env_type_c ? std::string(env_type_c) : std::string("none");
    const bool robotwin_tshape = lingbot_env_type == "robotwin_tshape";
    const int64_t default_frame_chunk_size = robotwin_tshape ? 2 : 4;
    const int64_t default_action_per_frame = robotwin_tshape ? 16 : 4;
    const int64_t frame_chunk_size = env_i32("VLA_LINGBOT_FRAME_CHUNK_SIZE", (int) default_frame_chunk_size);
    const int64_t n_suffix = lingbot_env_i64("VLA_LINGBOT_N_SUFFIX",
                                             frame_chunk_size * default_action_per_frame);
    const int video_steps = env_i32("VLA_LINGBOT_PREDICT_VIDEO_STEPS", robotwin_tshape ? 25 : 20);
    const int action_steps = env_i32("VLA_LINGBOT_PREDICT_ACTION_STEPS", 50);
    const bool text_encoder_requested = !text_encoder_gguf.empty();
    const float default_video_guidance_scale = text_encoder_requested ? 5.0f : 1.0f;
    const float video_guidance_scale = env_f32("VLA_LINGBOT_VIDEO_GUIDANCE_SCALE",
                                               default_video_guidance_scale);
    const float action_guidance_scale = env_f32("VLA_LINGBOT_ACTION_GUIDANCE_SCALE", 1.0f);
    const bool video_cfg_enabled = video_guidance_scale > 1.0f;
    const bool action_cfg_enabled = action_guidance_scale > 1.0f;
    const bool use_cfg = video_cfg_enabled || action_cfg_enabled;
    const bool use_official_order = lingbot_predict_official_order_enabled();
    const bool cache_prefill_update = in.lingbot_cache_mode == 2;
    const int64_t frame_start_id = std::max<int64_t>(0, in.lingbot_frame_start_id);
    const bool bf16_state = !lingbot_env_enabled("VLA_LINGBOT_STATE_F32");
    auto round_model_state = [&](std::vector<float> & v) {
        if (bf16_state) lingbot_bf16_roundtrip_inplace(v);
    };
    const bool use_cuda_self_attn =
        lingbot_predict_cuda_self_attn_enabled();
    const bool official_runtime_kv_self_attn = use_official_order && in.lingbot_session_id != 0;
    if (official_runtime_kv_self_attn && !use_cuda_self_attn) {
        std::printf("vla(lingbot_va): official-order enables runtime KV self-attn by default "
                    "for Python cache semantics\n");
    }
    if (in.lingbot_clear_pred_cache) {
        runtime_kv_clear_pred_for_session(this, in.lingbot_session_id);
    }

    if (n_suffix <= 0 || cfg.max_action_dim <= 0) {
        std::fprintf(stderr, "vla(lingbot_va): invalid action output shape suffix=%lld action_dim=%lld\n",
                     (long long) n_suffix, (long long) cfg.max_action_dim);
        return {};
    }

    gguf_reader g;
    if (!g.open(ckpt_path)) return {};
    std::unique_ptr<LingBotRuntimeWeights> local_common;
    LingBotRuntimeWeights * common_ptr = nullptr;
    if (full_transformer_weights) {
        common_ptr = full_transformer_weights;
        std::printf("vla(lingbot_va): full transformer weight cache hit\n");
    } else {
        static std::unique_ptr<LingBotRuntimeWeights> s_common_cache;
        static std::string s_common_cache_key;
        const bool common_cache_disabled = std::getenv("VLA_LINGBOT_COMMON_CACHE_DISABLE") != nullptr;
        const std::string common_cache_key = ckpt_path + "|" + lingbot_runtime_backend_key();
        if (!common_cache_disabled && s_common_cache && s_common_cache_key == common_cache_key) {
            common_ptr = s_common_cache.get();
            std::printf("vla(lingbot_va): common weight cache hit\n");
        } else {
            local_common = std::make_unique<LingBotRuntimeWeights>();
            if (!make_runtime_common_weights(g, *local_common)) return {};
            if (!common_cache_disabled) {
                s_common_cache = std::move(local_common);
                s_common_cache_key = common_cache_key;
                common_ptr = s_common_cache.get();
                std::printf("vla(lingbot_va): common weight cache store\n");
            } else {
                common_ptr = local_common.get();
            }
        }
    }
    LingBotTransformerExecutor ex{*this, *common_ptr, std::string()};
    LingBotTransformerExecutor ex_uncond{*this, *common_ptr, std::string()};
    ex.cache_branch = 0;
    ex_uncond.cache_branch = 1;
    const bool use_text_encoder = text_encoder_requested;
    const auto t_text0 = std::chrono::steady_clock::now();
    if (use_text_encoder) {
        if (!in.lang_tokens || in.n_lang <= 0) {
            std::fprintf(stderr, "vla(lingbot_va): LingBot text encoder path requires lang_tokens\n");
            return {};
        }
        if (in.n_lang > (int) cfg.n_lang) {
            std::fprintf(stderr, "vla(lingbot_va): lang_tokens length %d exceeds cfg.n_lang=%lld\n",
                         in.n_lang, (long long) cfg.n_lang);
            return {};
        }
        int text_blocks = 24;
        if (const char * env = std::getenv("VLA_LINGBOT_PREDICT_TEXT_BLOCKS")) {
            char * end = nullptr;
            const long v = std::strtol(env, &end, 10);
            if (end && *end == '\0' && v >= 0) {
                text_blocks = (int) v;
            } else {
                std::fprintf(stderr,
                             "vla(lingbot_va): ignoring invalid VLA_LINGBOT_PREDICT_TEXT_BLOCKS='%s'\n",
                             env);
            }
        }
        const std::string& text_path = text_encoder_gguf;
        const bool cache_disabled = std::getenv("VLA_LINGBOT_TEXT_CACHE_DISABLE") != nullptr;
        auto encode_text_for_executor = [&](const std::vector<int32_t> & ids,
                                            LingBotTransformerExecutor & target,
                                            const char * label) -> bool {
            int64_t text_seq = 0;
            int64_t text_dim = 0;
            bool store_text_cache = false;
            const char * override_env = nullptr;
            if (std::strcmp(label, "cond") == 0) {
                override_env = std::getenv("VLA_LINGBOT_TEXT_EMB_COND_F32");
            } else if (std::strcmp(label, "uncond") == 0) {
                override_env = std::getenv("VLA_LINGBOT_TEXT_EMB_UNCOND_F32");
            }
            if (override_env && *override_env) {
                text_dim = this->text_dim;
                if (!read_text_embedding_f32_file(override_env, text_dim, target.text_raw_override, text_seq)) {
                    return false;
                }
                std::printf("vla(lingbot_va): UMT5 text override (%s): path=%s tokens=%lld "
                            "checksum=%.9g max=%.9g\n",
                            label, override_env, (long long) text_seq,
                            checksum(target.text_raw_override), max_abs_value(target.text_raw_override));
            } else if (!cache_disabled &&
                text_cache_lookup(text_path, text_blocks, ids, target.text_raw_override, text_seq, text_dim)) {
                std::printf("vla(lingbot_va): UMT5 text cache hit (%s): tokens=%zu blocks=%d checksum=%.9g\n",
                            label, ids.size(), text_blocks, checksum(target.text_raw_override));
            } else {
                gguf_reader tg;
                if (!tg.open(text_path)) return false;
                if (!encode_umt5_tokens_host(tg, ids, text_blocks,
                                             target.text_raw_override, text_seq, text_dim)) {
                    return false;
                }
                store_text_cache = !cache_disabled;
            }
            if (text_dim != this->text_dim) {
                std::fprintf(stderr, "vla(lingbot_va): UMT5 output dim %lld != transformer text_dim=%lld\n",
                             (long long) text_dim, (long long) this->text_dim);
                return false;
            }
            const int64_t target_text_seq = this->cfg.n_lang > 0 ? this->cfg.n_lang : 512;
            if (!pad_text_embedding_to_sequence(target.text_raw_override,
                                                text_dim,
                                                text_seq,
                                                target_text_seq,
                                                label)) {
                return false;
            }
            if (store_text_cache) {
                text_cache_store(text_path, text_blocks, ids, target.text_raw_override, text_seq, text_dim);
                std::printf("vla(lingbot_va): UMT5 text cache store (%s): tokens=%zu blocks=%d bytes=%.2f MiB\n",
                            label, ids.size(), text_blocks,
                            target.text_raw_override.size() * sizeof(float) / (1024.0 * 1024.0));
            }
            target.text_raw_seq = text_seq;
            return true;
        };

        std::vector<int32_t> ids(in.lang_tokens, in.lang_tokens + in.n_lang);
        if (!encode_text_for_executor(ids, ex, "cond")) return {};
        if (use_cfg) {
            // Python negative_prompt=None becomes the empty string. With the
            // official T5 tokenizer its only active token is EOS (id 1); the
            // encoder output is then padded to cfg.n_lang by encode_text_for_executor().
            std::vector<int32_t> negative_ids = env_i32_list("VLA_LINGBOT_NEGATIVE_TOKEN_IDS", {1});
            if (!encode_text_for_executor(negative_ids, ex_uncond, "uncond")) return {};
        }
    } else if (use_cfg) {
        std::fprintf(stderr,
                     "vla(lingbot_va): classifier-free guidance requires "
                     "a text encoder GGUF passed at model load\n");
        return {};
    }
    if (use_text_encoder) {
        text_encode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_text0).count();
    }
    if (const char * dump_dir_c = std::getenv("VLA_LINGBOT_PREDICT_DUMP_DIR")) {
        if (!ex.text_raw_override.empty()) {
            const std::string dump_dir(dump_dir_c);
            dump_f32_file(dump_dir + "/lingbot_predict_text_emb_raw.f32", ex.text_raw_override);
            dump_text_file(dump_dir + "/lingbot_predict_text_emb_raw.shape.txt",
                           std::to_string((long long) ex.text_raw_seq) + " " +
                           std::to_string((long long) this->text_dim) + "\n");
        }
    }
    const int64_t window_size = stream_window_size(g, *this, blocks);
    const bool force_latent_condition = std::getenv("VLA_LINGBOT_FORCE_LATENT_COND") != nullptr;
    const bool condition_current_frame =
        force_latent_condition || in.lingbot_first_chunk || in.lingbot_cache_mode == 2 ||
        !in.lingbot_has_history_cache;
    std::printf("vla(lingbot_va): session=%llu predict_index=%llu cache_updates=%llu "
                "first_chunk=%s history_cache=%s cache_mode=%d latent_condition=%s\n",
                (unsigned long long) in.lingbot_session_id,
                (unsigned long long) in.lingbot_predict_index,
                (unsigned long long) in.lingbot_cache_update_index,
                in.lingbot_first_chunk ? "true" : "false",
                in.lingbot_has_history_cache ? "true" : "false",
                in.lingbot_cache_mode,
                condition_current_frame ? "true" : "false");

    std::vector<float> encoded_video_latent;
    LingBotTensor5DShape encoded_video_shape{};
    if (!in.lingbot_latent && in.lingbot_video && condition_current_frame) {
        if (vae_encoder_gguf.empty()) {
            std::fprintf(stderr,
                         "vla(lingbot_va): LingBot video input requires a VAE encoder GGUF "
                         "passed at model load\n");
            return {};
        }
        {
            LingBotScopedTimer vae_timer("vae_encode_video_to_latent");
            const auto t_vae0 = std::chrono::steady_clock::now();
            const float * video_for_encode = in.lingbot_video;
            int64_t frames_for_encode = in.lingbot_video_f;
            LingBotVaeStreamingSessionCache * vae_stream_cache =
                lingbot_env_disabled("VLA_LINGBOT_VAE_STREAM_CACHE_DISABLE")
                    ? nullptr
                    : lingbot_get_vae_stream_cache(in.lingbot_session_id, in.lingbot_video_views);
            std::vector<float> replay_video;
            std::vector<int> input_chunks;
            std::vector<int> latent_chunks;
            bool replayed_stream_history = false;
            if (!robotwin_tshape && !vae_stream_cache && cache_prefill_update &&
                lingbot_build_video_history_with_current(in.lingbot_session_id,
                                                         in.lingbot_video,
                                                         in.lingbot_video_views,
                                                         in.lingbot_video_c,
                                                         in.lingbot_video_f,
                                                         in.lingbot_video_h,
                                                         in.lingbot_video_w,
                                                         replay_video,
                                                         input_chunks,
                                                         &frames_for_encode)) {
                video_for_encode = replay_video.data();
                replayed_stream_history = true;
                std::printf("vla(lingbot_va): replaying VAE streaming history session=%llu chunks=%zu total_frames=%lld\n",
                            (unsigned long long) in.lingbot_session_id,
                            input_chunks.size(),
                            (long long) frames_for_encode);
            } else {
                input_chunks.push_back((int) in.lingbot_video_f);
            }
            if (robotwin_tshape) {
                if (in.lingbot_video_view_shape_count != in.lingbot_video_views ||
                    !in.lingbot_video_view_f || !in.lingbot_video_view_h || !in.lingbot_video_view_w) {
                    std::fprintf(stderr,
                                 "vla(lingbot_va): robotwin_tshape requires per-view video shapes\n");
                    return {};
                }
                if (!encode_lingbot_robotwin_tshape_video_to_latent(
                        vae_encoder_gguf.c_str(),
                        in.lingbot_video,
                        in.lingbot_video_views,
                        in.lingbot_video_c,
                        in.lingbot_video_view_f,
                        in.lingbot_video_view_h,
                        in.lingbot_video_view_w,
                        &input_chunks,
                        &latent_chunks,
                        vae_stream_cache,
                        encoded_video_latent,
                        encoded_video_shape)) {
                    return {};
                }
            } else {
                if (!encode_lingbot_video_to_latent(vae_encoder_gguf.c_str(),
                                                    video_for_encode,
                                                    in.lingbot_video_views,
                                                    in.lingbot_video_c,
                                                    frames_for_encode,
                                                    in.lingbot_video_h,
                                                    in.lingbot_video_w,
                                                    &input_chunks,
                                                    &latent_chunks,
                                                    vae_stream_cache,
                                                    encoded_video_latent,
                                                    encoded_video_shape)) {
                    return {};
                }
            }
            if (replayed_stream_history) {
                int64_t prior_latent_frames = 0;
                for (size_t i = 0; i + 1 < latent_chunks.size(); ++i) prior_latent_frames += latent_chunks[i];
                const int64_t current_latent_frames = latent_chunks.empty() ? 0 : latent_chunks.back();
                std::vector<float> current_latent;
                LingBotTensor5DShape current_shape{};
                if (!lingbot_slice_latent_time(encoded_video_latent,
                                               encoded_video_shape,
                                               prior_latent_frames,
                                               current_latent_frames,
                                               current_latent,
                                               current_shape)) {
                    std::fprintf(stderr, "vla(lingbot_va): failed to slice replayed VAE latent current chunk\n");
                    return {};
                }
                encoded_video_latent = std::move(current_latent);
                encoded_video_shape = current_shape;
                std::printf("vla(lingbot_va): sliced replayed VAE latent current chunk shape=[%lld,%lld,%lld,%lld,%lld] checksum=%.9g\n",
                            (long long) encoded_video_shape.b,
                            (long long) encoded_video_shape.c,
                            (long long) encoded_video_shape.f,
                            (long long) encoded_video_shape.h,
                            (long long) encoded_video_shape.w,
                            checksum(encoded_video_latent));
            }
            round_model_state(encoded_video_latent);
            vae_encode_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_vae0).count();
        }
        if (const char * dump_dir_c = std::getenv("VLA_LINGBOT_PREDICT_DUMP_DIR")) {
            const std::string dump_dir(dump_dir_c);
            dump_f32_file(dump_dir + "/lingbot_predict_vae_latent_raw.f32", encoded_video_latent);
            dump_text_file(dump_dir + "/lingbot_predict_vae_latent_raw.shape.txt",
                           std::to_string((long long) encoded_video_shape.b) + " " +
                           std::to_string((long long) encoded_video_shape.c) + " " +
                           std::to_string((long long) encoded_video_shape.f) + " " +
                           std::to_string((long long) encoded_video_shape.h) + " " +
                           std::to_string((long long) encoded_video_shape.w) + "\n");
        }
        if (!robotwin_tshape && in.lingbot_first_chunk && encoded_video_shape.f == 1) {
            lingbot_store_init_latent(in.lingbot_session_id,
                                      encoded_video_latent,
                                      encoded_video_shape);
            lingbot_store_video_history(in.lingbot_session_id,
                                        in.lingbot_video,
                                        in.lingbot_video_views,
                                        in.lingbot_video_c,
                                        in.lingbot_video_f,
                                        in.lingbot_video_h,
                                        in.lingbot_video_w);
        } else if (!robotwin_tshape && cache_prefill_update && in.lingbot_cache_update_index == 0) {
            lingbot_prepend_init_latent(in.lingbot_session_id,
                                        encoded_video_latent,
                                        encoded_video_shape);
            lingbot_append_video_history(in.lingbot_session_id,
                                         in.lingbot_video,
                                         in.lingbot_video_views,
                                         in.lingbot_video_c,
                                         in.lingbot_video_f,
                                         in.lingbot_video_h,
                                         in.lingbot_video_w);
        } else if (!robotwin_tshape && cache_prefill_update) {
            lingbot_append_video_history(in.lingbot_session_id,
                                         in.lingbot_video,
                                         in.lingbot_video_views,
                                         in.lingbot_video_c,
                                         in.lingbot_video_f,
                                         in.lingbot_video_h,
                                         in.lingbot_video_w);
        }
    }

    LingBotTensor5DShape latent_shape{1, in_channels,
                                      frame_chunk_size * patch_t, 2 * patch_h, 2 * patch_w};
    if (in.lingbot_latent) {
        latent_shape = {
            in.lingbot_latent_b,
            in.lingbot_latent_c,
            in.lingbot_latent_f,
            in.lingbot_latent_h,
            in.lingbot_latent_w,
        };
        if (!shape_valid(latent_shape) || latent_shape.c != in_channels ||
            latent_shape.f % patch_t != 0 || latent_shape.h % patch_h != 0 ||
            latent_shape.w % patch_w != 0) {
            std::fprintf(stderr,
                         "vla(lingbot_va): invalid LingBot latent shape [%lld,%lld,%lld,%lld,%lld]; "
                         "expected C=%lld and divisibility by patch=[%lld,%lld,%lld]\n",
                         (long long) latent_shape.b, (long long) latent_shape.c,
                         (long long) latent_shape.f, (long long) latent_shape.h,
                         (long long) latent_shape.w, (long long) in_channels,
                         (long long) patch_t, (long long) patch_h, (long long) patch_w);
            return {};
        }
    } else if (in.lingbot_latent_noise) {
        latent_shape = {
            in.lingbot_latent_noise_b,
            in.lingbot_latent_noise_c,
            in.lingbot_latent_noise_f,
            in.lingbot_latent_noise_h,
            in.lingbot_latent_noise_w,
        };
        if (!shape_valid(latent_shape) || latent_shape.c != in_channels ||
            latent_shape.f % patch_t != 0 || latent_shape.h % patch_h != 0 ||
            latent_shape.w % patch_w != 0) {
            std::fprintf(stderr,
                         "vla(lingbot_va): invalid LingBot latent noise shape [%lld,%lld,%lld,%lld,%lld]; "
                         "expected C=%lld and divisibility by patch=[%lld,%lld,%lld]\n",
                         (long long) latent_shape.b, (long long) latent_shape.c,
                         (long long) latent_shape.f, (long long) latent_shape.h,
                         (long long) latent_shape.w, (long long) in_channels,
                         (long long) patch_t, (long long) patch_h, (long long) patch_w);
            return {};
        }
    } else if (!encoded_video_latent.empty() && condition_current_frame) {
        latent_shape = encoded_video_shape;
        if (!shape_valid(latent_shape) || latent_shape.c != in_channels ||
            latent_shape.f % patch_t != 0 || latent_shape.h % patch_h != 0 ||
            latent_shape.w % patch_w != 0) {
            std::fprintf(stderr,
                         "vla(lingbot_va): VAE encoded latent shape [%lld,%lld,%lld,%lld,%lld] "
                         "is incompatible with transformer C=%lld patch=[%lld,%lld,%lld]\n",
                         (long long) latent_shape.b, (long long) latent_shape.c,
                         (long long) latent_shape.f, (long long) latent_shape.h,
                         (long long) latent_shape.w, (long long) in_channels,
                         (long long) patch_t, (long long) patch_h, (long long) patch_w);
            return {};
        }
    }
    const int64_t action_per_frame = in.lingbot_action_per_frame > 0
        ? in.lingbot_action_per_frame
        : (int64_t) env_i32("VLA_LINGBOT_ACTION_PER_FRAME", (int) default_action_per_frame);
    if (n_suffix % action_per_frame != 0) {
        std::fprintf(stderr,
                     "vla(lingbot_va): n_suffix=%lld is not divisible by action_per_frame=%lld\n",
                     (long long) n_suffix, (long long) action_per_frame);
        return {};
    }
    const LingBotTensor5DShape action_shape{1, action_dim, n_suffix / action_per_frame, action_per_frame, 1};
    std::vector<float> latent_condition;
    LingBotTensor5DShape latent_condition_shape{};
    bool has_latent_condition = false;
    if (!in.lingbot_latent && condition_current_frame && !encoded_video_latent.empty() &&
        encoded_video_shape.b == 1 && encoded_video_shape.c == in_channels &&
        encoded_video_shape.f > 0 && encoded_video_shape.f <= latent_shape.f &&
        encoded_video_shape.h == latent_shape.h && encoded_video_shape.w == latent_shape.w) {
        latent_condition = encoded_video_latent;
        latent_condition_shape = encoded_video_shape;
        round_model_state(latent_condition);
        has_latent_condition = true;
        latent_shape = {1, in_channels, frame_chunk_size * patch_t,
                        encoded_video_shape.h, encoded_video_shape.w};
    }

    std::vector<float> latent_sample((size_t) latent_shape.b * (size_t) latent_shape.c *
                                     (size_t) latent_shape.f * (size_t) latent_shape.h *
                                     (size_t) latent_shape.w);
    std::vector<float> action_sample((size_t) action_shape.b * (size_t) action_shape.c *
                                     (size_t) action_shape.f * (size_t) action_shape.h *
                                     (size_t) action_shape.w);
    const bool deterministic_noise = lingbot_env_enabled("VLA_LINGBOT_DETERMINISTIC_NOISE");
    std::mt19937_64 noise_rng = make_lingbot_noise_rng(in.lingbot_session_id,
                                                       in.lingbot_predict_index);
    if (in.lingbot_latent) {
        std::memcpy(latent_sample.data(), in.lingbot_latent, latent_sample.size() * sizeof(float));
        round_model_state(latent_sample);
        std::printf("vla(lingbot_va): using caller LingBot latent shape=[%lld,%lld,%lld,%lld,%lld] checksum=%.9g\n",
                    (long long) latent_shape.b, (long long) latent_shape.c,
                    (long long) latent_shape.f, (long long) latent_shape.h,
                    (long long) latent_shape.w, checksum(latent_sample));
    } else if (in.lingbot_latent_noise) {
        const LingBotTensor5DShape noise_shape{
            in.lingbot_latent_noise_b,
            in.lingbot_latent_noise_c,
            in.lingbot_latent_noise_f,
            in.lingbot_latent_noise_h,
            in.lingbot_latent_noise_w,
        };
        if (noise_shape.b != latent_shape.b || noise_shape.c != latent_shape.c ||
            noise_shape.f != latent_shape.f || noise_shape.h != latent_shape.h ||
            noise_shape.w != latent_shape.w) {
            std::fprintf(stderr,
                         "vla(lingbot_va): LingBot latent noise shape [%lld,%lld,%lld,%lld,%lld] "
                         "!= expected latent shape [%lld,%lld,%lld,%lld,%lld]\n",
                         (long long) noise_shape.b, (long long) noise_shape.c,
                         (long long) noise_shape.f, (long long) noise_shape.h,
                         (long long) noise_shape.w,
                         (long long) latent_shape.b, (long long) latent_shape.c,
                         (long long) latent_shape.f, (long long) latent_shape.h,
                         (long long) latent_shape.w);
            return {};
        }
        std::memcpy(latent_sample.data(), in.lingbot_latent_noise,
                    latent_sample.size() * sizeof(float));
        round_model_state(latent_sample);
        std::printf("vla(lingbot_va): using caller LingBot latent noise shape=[%lld,%lld,%lld,%lld,%lld] "
                    "checksum=%.9g max=%.9g\n",
                    (long long) latent_shape.b, (long long) latent_shape.c,
                    (long long) latent_shape.f, (long long) latent_shape.h,
                    (long long) latent_shape.w, checksum(latent_sample), max_abs_value(latent_sample));
    } else if (deterministic_noise) {
        fill_deterministic(latent_sample, 0.02f);
        round_model_state(latent_sample);
    } else {
        fill_standard_normal(latent_sample, noise_rng);
        round_model_state(latent_sample);
    }
    if (has_latent_condition) {
        if (latent_sample.empty()) {
            if (deterministic_noise) {
                fill_deterministic(latent_sample, 0.02f);
            } else {
                fill_standard_normal(latent_sample, noise_rng);
            }
        }
        for (int64_t c = 0; c < latent_shape.c; ++c) {
            for (int64_t f = 0; f < latent_condition_shape.f; ++f) {
                for (int64_t h = 0; h < latent_shape.h; ++h) {
                    for (int64_t w = 0; w < latent_shape.w; ++w) {
                        latent_sample[idx5(latent_shape, 0, c, f, h, w)] =
                            latent_condition[idx5(latent_condition_shape, 0, c, f, h, w)];
                    }
                }
            }
        }
        std::printf("vla(lingbot_va): using VAE latent as frame condition shape=[%lld,%lld,%lld,%lld,%lld] "
                    "sample_shape=[%lld,%lld,%lld,%lld,%lld] checksum=%.9g\n",
                    (long long) latent_condition_shape.b, (long long) latent_condition_shape.c,
                    (long long) latent_condition_shape.f, (long long) latent_condition_shape.h,
                    (long long) latent_condition_shape.w,
                    (long long) latent_shape.b, (long long) latent_shape.c,
                    (long long) latent_shape.f, (long long) latent_shape.h,
                    (long long) latent_shape.w, checksum(latent_condition));
    }
    if (in.noise) {
        for (int64_t t = 0; t < n_suffix; ++t) {
            const int64_t f = t / action_per_frame;
            const int64_t h = t % action_per_frame;
            for (int64_t c = 0; c < action_dim; ++c) {
                action_sample[idx5(action_shape, 0, c, f, h, 0)] =
                    in.noise[(size_t) t * (size_t) action_dim + (size_t) c];
            }
        }
        round_model_state(action_sample);
    } else if (deterministic_noise) {
        fill_deterministic(action_sample, 0.03f);
        round_model_state(action_sample);
    } else {
        fill_standard_normal(action_sample, noise_rng);
        round_model_state(action_sample);
    }
    std::vector<float> action_condition;
    if (in.lingbot_action_condition != nullptr) {
        if (!preprocess_libero_action_history_to_action_condition(
                    in.lingbot_action_condition,
                    in.lingbot_action_condition_c,
                    in.lingbot_action_condition_f,
                    in.lingbot_action_condition_h,
                    action_shape,
                    action_condition)) {
            std::fprintf(stderr, "vla(lingbot_va): failed to preprocess action history condition\n");
            return {};
        }
        round_model_state(action_condition);
        apply_action_condition_frame0(action_sample, action_condition, action_shape);
        std::printf("vla(lingbot_va): using LingBot action history condition shape=[%lld,%lld,%lld] checksum=%.9g max=%.9g\n",
                    (long long) in.lingbot_action_condition_c,
                    (long long) in.lingbot_action_condition_f,
                    (long long) in.lingbot_action_condition_h,
                    checksum(action_condition), max_abs_value(action_condition));
    } else if (!std::getenv("VLA_LINGBOT_DISABLE_ZERO_ACTION_COND") &&
               (in.lingbot_first_chunk ||
                std::getenv("VLA_LINGBOT_FORCE_ZERO_ACTION_COND") != nullptr)) {
        action_condition.assign(action_sample.size(), 0.0f);
        apply_action_condition_frame0(action_sample, action_condition, action_shape);
        std::printf("vla(lingbot_va): using default zero action condition for frame 0\n");
    } else if (in.state != nullptr &&
               std::getenv("VLA_LINGBOT_STATE_COND") != nullptr) {
        if (!preprocess_libero_state_to_action_condition(in.state, cfg.real_state_dim,
                                                         action_shape, action_condition)) {
            std::fprintf(stderr, "vla(lingbot_va): failed to preprocess state into action condition\n");
            return {};
        }
        round_model_state(action_condition);
        apply_action_condition_frame0(action_sample, action_condition, action_shape);
        std::printf("vla(lingbot_va): using LIBERO state action condition checksum=%.9g max=%.9g\n",
                    checksum(action_condition), max_abs_value(action_condition));
    }

    if (cache_prefill_update) {
        if (!encoded_video_latent.empty()) {
            latent_shape = encoded_video_shape;
            latent_sample = encoded_video_latent;
            has_latent_condition = false;
            std::printf("vla(lingbot_va): cache update uses encoded observation latent as input "
                        "shape=[%lld,%lld,%lld,%lld,%lld] checksum=%.9g\n",
                        (long long) latent_shape.b, (long long) latent_shape.c,
                        (long long) latent_shape.f, (long long) latent_shape.h,
                        (long long) latent_shape.w, checksum(latent_sample));
        }
        if (!action_condition.empty()) {
            action_sample = action_condition;
            std::printf("vla(lingbot_va): cache update uses executed action condition as input "
                        "shape=[%lld,%lld,%lld,%lld,%lld] checksum=%.9g max=%.9g\n",
                        (long long) action_shape.b, (long long) action_shape.c,
                        (long long) action_shape.f, (long long) action_shape.h,
                        (long long) action_shape.w, checksum(action_sample),
                        max_abs_value(action_sample));
        } else {
            std::printf("vla(lingbot_va): cache update has no action condition; using fallback action sample\n");
        }

        if (lingbot_env_enabled("VLA_LINGBOT_VAE_ONLY")) {
            const auto t_total1 = std::chrono::steady_clock::now();
            stats.ms_vision = (float) vae_encode_ms;
            stats.ms_prefill = (float) text_encode_ms;
            stats.ms_denoise = 0.0f;
            stats.ms_total = std::chrono::duration<float, std::milli>(t_total1 - t_total0).count();
            stats.ms_inference = stats.ms_total;
            std::printf("vla(lingbot_va): VAE-only cache update parity return "
                        "latent=[%lld,%lld,%lld,%lld,%lld] checksum=%.9g action_checksum=%.9g total=%.1fms\n",
                        (long long) latent_shape.b, (long long) latent_shape.c,
                        (long long) latent_shape.f, (long long) latent_shape.h,
                        (long long) latent_shape.w, checksum(latent_sample),
                        checksum(action_sample), stats.ms_total);
            return std::vector<float>((size_t) n_suffix * (size_t) cfg.real_action_dim, 0.0f);
        }

        zero_unused_libero_action_channels(action_sample, action_shape);
        round_model_state(action_sample);

        if (use_official_order) {
            auto run_cache_forward = [&](LingBotTransformerExecutor & fex,
                                         bool action_mode,
                                         const std::vector<float> & tensor,
                                         const LingBotTensor5DShape & shape) -> bool {
                std::vector<float> tokens;
                int64_t feature = 0;
                int64_t seq = 0;
                LingBotGridSpec grid;
                if (action_mode) {
                    if (!action_tensor_to_tokens(tensor, shape, tokens, feature, seq)) {
                        std::fprintf(stderr, "vla(lingbot_va): official-order cache action tokenization failed\n");
                        return false;
                    }
                    grid.f = shape.f;
                    grid.h = shape.h;
                    grid.w = shape.w;
                    grid.t = 1;
                    grid.f_shift = frame_start_id;
                    grid.action = true;
                } else {
                    if (!patchify_latent_tokens(tensor, shape, patch_t, patch_h, patch_w,
                                                tokens, feature, seq)) {
                        std::fprintf(stderr, "vla(lingbot_va): official-order cache latent tokenization failed\n");
                        return false;
                    }
                    grid.f = shape.f / patch_t;
                    grid.h = shape.h / patch_h;
                    grid.w = shape.w / patch_w;
                    grid.t = 0;
                    grid.f_shift = frame_start_id;
                    grid.action = false;
                }
                std::vector<float> ignored;
                return exec_forward_one_streaming(fex, *common_ptr, action_mode, &tokens,
                                                  0.0, blocks, ignored, &grid, nullptr,
                                                  in.lingbot_session_id, 2,
                                                  official_runtime_kv_self_attn);
            };

            const auto t_inf0 = std::chrono::steady_clock::now();
            if (use_cfg &&
                (!run_cache_forward(ex_uncond, false, latent_sample, latent_shape) ||
                 !run_cache_forward(ex_uncond, true, action_sample, action_shape))) {
                return {};
            }
            if (!run_cache_forward(ex, false, latent_sample, latent_shape) ||
                !run_cache_forward(ex, true, action_sample, action_shape)) {
                return {};
            }
            const auto t_inf1 = std::chrono::steady_clock::now();
            const auto t_total1 = std::chrono::steady_clock::now();
            stats.ms_vision = (float) vae_encode_ms;
            stats.ms_prefill = (float) text_encode_ms;
            stats.ms_denoise = std::chrono::duration<float, std::milli>(t_inf1 - t_inf0).count();
            stats.ms_total = std::chrono::duration<float, std::milli>(t_total1 - t_total0).count();
            stats.ms_inference = stats.ms_total;
            const double other_ms = std::max<double>(
                0.0, (double) stats.ms_total - vae_encode_ms - text_encode_ms - (double) stats.ms_denoise);
            std::printf("vla(lingbot_va): official-order kv cache update bridge ok blocks=%d "
                        "latent=[%lld,%lld,%lld,%lld,%lld] action=[%lld,%lld,%lld,%lld,%lld] "
                        "inf=%.1fms\n",
                        blocks,
                        (long long) latent_shape.b, (long long) latent_shape.c,
                        (long long) latent_shape.f, (long long) latent_shape.h,
                        (long long) latent_shape.w,
                        (long long) action_shape.b, (long long) action_shape.c,
                        (long long) action_shape.f, (long long) action_shape.h,
                        (long long) action_shape.w,
                        stats.ms_inference);
            std::printf("vla(lingbot_va): timing model kind=cache_update total=%.3fms text=%.3fms "
                        "vae=%.3fms denoise=%.3fms output=0.000ms other=%.3fms\n",
                        stats.ms_total, text_encode_ms, vae_encode_ms,
                        stats.ms_denoise, other_ms);
            return std::vector<float>((size_t) n_suffix * (size_t) cfg.real_action_dim, 0.0f);
        }

        std::fprintf(stderr, "vla(lingbot_va): non-official cache_update path was removed; official-order is required\n");
        return {};
    }

    if (lingbot_env_enabled("VLA_LINGBOT_VAE_ONLY")) {
        const auto t_total1 = std::chrono::steady_clock::now();
        stats.ms_vision = (float) vae_encode_ms;
        stats.ms_prefill = (float) text_encode_ms;
        stats.ms_denoise = 0.0f;
        stats.ms_total = std::chrono::duration<float, std::milli>(t_total1 - t_total0).count();
        stats.ms_inference = stats.ms_total;
        std::printf("vla(lingbot_va): VAE-only predict parity return "
                    "latent_condition=%s shape=[%lld,%lld,%lld,%lld,%lld] checksum=%.9g total=%.1fms\n",
                    has_latent_condition ? "true" : "false",
                    (long long) latent_condition_shape.b, (long long) latent_condition_shape.c,
                    (long long) latent_condition_shape.f, (long long) latent_condition_shape.h,
                    (long long) latent_condition_shape.w, checksum(latent_condition), stats.ms_total);
        return std::vector<float>((size_t) n_suffix * (size_t) cfg.real_action_dim, 0.0f);
    }

    LingBotFlowScheduler video_sched;
    video_sched.shift = env_f32("VLA_LINGBOT_VIDEO_SNR_SHIFT", 5.0f);
    video_sched.sigma_min = 0.0;
    video_sched.extra_one_step = true;
    video_sched.set_timesteps(video_steps);
    LingBotFlowScheduler action_sched;
    action_sched.shift = env_f32("VLA_LINGBOT_ACTION_SNR_SHIFT",
                                 robotwin_tshape ? 1.0f : 0.05f);
    action_sched.sigma_min = 0.0;
    action_sched.extra_one_step = true;
    action_sched.set_timesteps(action_steps);

    std::vector<double> video_loop_timesteps = video_sched.timesteps;
    std::vector<double> action_loop_timesteps = action_sched.timesteps;
    // Python LingBot pads both scheduler timestep arrays with a terminal zero.
    // The final t=0 forward is not integrated into the sample; it produces the
    // returned prediction and writes the pred KV cache.
    video_loop_timesteps.push_back(0.0);
    action_loop_timesteps.push_back(0.0);

    const auto t_inf0 = std::chrono::steady_clock::now();
    const int total_steps = std::max((int) video_loop_timesteps.size(),
                                     (int) action_loop_timesteps.size());
    double denoise_forward_ms = 0.0;
    int denoise_forward_calls = 0;
    double latent_forward_ms = 0.0;
    double action_forward_ms = 0.0;
    int latent_forward_calls = 0;
    int action_forward_calls = 0;
    double video_restore_ms = 0.0;
    double video_guidance_ms = 0.0;
    double video_scheduler_ms = 0.0;
    double action_finalize_ms = 0.0;
    double action_guidance_ms = 0.0;
    double action_scheduler_ms = 0.0;
    double final_output_ms = 0.0;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    bool device_denoise_bridge = false;
    LingBotDeviceTensor latent_sample_device;
    LingBotDeviceTensor latent_pred_device;
    LingBotDeviceTensor latent_condition_device;
    LingBotDeviceTensor action_sample_device;
    LingBotDeviceTensor action_pred_device;
    LingBotDeviceTensor action_condition_device;
#endif
    if (use_official_order) {
        auto restore_latent_condition = [&]() {
            if (!has_latent_condition) return;
            for (int64_t c = 0; c < latent_shape.c; ++c) {
                for (int64_t f = 0; f < latent_condition_shape.f; ++f) {
                    for (int64_t h = 0; h < latent_shape.h; ++h) {
                        for (int64_t w = 0; w < latent_shape.w; ++w) {
                            latent_sample[idx5(latent_shape, 0, c, f, h, w)] =
                                latent_condition[idx5(latent_condition_shape, 0, c, f, h, w)];
                        }
                    }
                }
            }
        };
        const LingBotGridSpec latent_grid_spec{
            latent_shape.f / patch_t,
            latent_shape.h / patch_h,
            latent_shape.w / patch_w,
            0,
            1,
            frame_start_id,
            false
        };
        const int64_t latent_forward_seq = latent_shape.b * latent_grid_spec.seq();
        std::vector<unsigned char> latent_zero_timestep_mask;
        if (has_latent_condition) {
            latent_zero_timestep_mask.assign((size_t) latent_forward_seq, (unsigned char) 0);
            const int64_t cond_patch_f = latent_condition_shape.f / patch_t;
            int64_t tok = 0;
            for (int64_t b = 0; b < latent_shape.b; ++b) {
                for (int64_t f = 0; f < latent_grid_spec.f; ++f) {
                    for (int64_t h = 0; h < latent_grid_spec.h; ++h) {
                        for (int64_t w = 0; w < latent_grid_spec.w; ++w, ++tok) {
                            if (f < cond_patch_f) {
                                latent_zero_timestep_mask[(size_t) tok] = 1;
                            }
                        }
                    }
                }
            }
        }
        const LingBotGridSpec action_grid_spec{
            action_shape.f,
            action_shape.h,
            action_shape.w,
            1,
            1,
            frame_start_id,
            true
        };
        const int64_t action_forward_seq = action_shape.b * action_grid_spec.seq();
        std::vector<unsigned char> action_zero_timestep_mask;
        if (!action_condition.empty()) {
            action_zero_timestep_mask.assign((size_t) action_forward_seq, (unsigned char) 0);
            int64_t tok = 0;
            for (int64_t b = 0; b < action_shape.b; ++b) {
                for (int64_t f = 0; f < action_shape.f; ++f) {
                    for (int64_t h = 0; h < action_shape.h; ++h) {
                        for (int64_t w = 0; w < action_shape.w; ++w, ++tok) {
                            if (f == 0) {
                                action_zero_timestep_mask[(size_t) tok] = 1;
                            }
                        }
                    }
                }
            }
        }
        struct OfficialForwardTensor {
            std::vector<float> host;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            LingBotDeviceTensor device;
            bool has_device = false;
#endif
            void reset() {
                host.clear();
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
                has_device = false;
#endif
            }
        };
        struct OfficialDenoiseForwardWorkspace {
            std::vector<float> latent_tokens;
            std::vector<float> latent_out_tokens;
            std::vector<float> action_tokens;
            std::vector<float> action_out_tokens;
            OfficialForwardTensor latent_pred;
            OfficialForwardTensor latent_uncond;
            OfficialForwardTensor action_pred;
            OfficialForwardTensor action_uncond;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            LingBotDeviceTensor latent_device_tokens;
            LingBotDeviceTensor action_device_tokens;
#endif
        };
        OfficialDenoiseForwardWorkspace denoise_ws;
        const bool stage5_workspace_enabled = lingbot_stage5_workspace_enabled();
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
        device_denoise_bridge =
            common_ptr->backend != nullptr &&
            ggml_backend_is_cuda(common_ptr->backend) &&
            !lingbot_env_enabled("VLA_LINGBOT_DEVICE_DENOISE_BRIDGE_DISABLE") &&
            !lingbot_env_enabled("VLA_LINGBOT_OUTPUT_DEVICE_OUT_DISABLE") &&
            !lingbot_env_enabled("VLA_LINGBOT_DENOISE_SAMPLE_DUMP") &&
            std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR") == nullptr &&
            !lingbot_env_enabled("VLA_LINGBOT_STREAM_BLOCK_TRACE");
        if (device_denoise_bridge) {
            if (!lingbot_upload_host_tensor(latent_sample_device, common_ptr->backend,
                                            latent_sample, "latent_sample") ||
                !lingbot_upload_host_tensor(action_sample_device, common_ptr->backend,
                                            action_sample, "action_sample")) {
                return {};
            }
            if (has_latent_condition &&
                !lingbot_upload_host_tensor(latent_condition_device, common_ptr->backend,
                                            latent_condition, "latent_condition")) {
                return {};
            }
            if (!action_condition.empty() &&
                !lingbot_upload_host_tensor(action_condition_device, common_ptr->backend,
                                            action_condition, "action_condition")) {
                return {};
            }
        }
#else
        const bool device_denoise_bridge = false;
#endif
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
        LingBotOfficialRuntimeKvDenoiseRunner * official_denoise_runner = nullptr;
        if (official_runtime_kv_self_attn &&
            !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_RUNTIME_KV_FULL_FORWARD_DISABLE") &&
            !lingbot_env_enabled("VLA_LINGBOT_OFFICIAL_CUDA_SELF_ATTN_GRAPH_CACHE_DISABLE") &&
            common_ptr->backend != nullptr &&
            ggml_backend_is_cuda(common_ptr->backend) &&
            (int64_t) common_ptr->blocks.size() >= blocks) {
            const bool use_bf16_boundary =
                lingbot_env_enabled("VLA_LINGBOT_BLOCK_BF16_BOUNDARY") &&
                !lingbot_env_enabled("VLA_LINGBOT_STATE_F32");
            const int64_t text_seq = ex.text_raw_seq > 0 ? ex.text_raw_seq : 2;
            official_denoise_runner = lingbot_get_official_runtime_kv_denoise_runner(
                *this, *common_ptr, blocks, text_seq,
                latent_forward_seq, action_forward_seq, use_bf16_boundary);
            if (!official_denoise_runner) {
                std::fprintf(stderr,
                             "vla(lingbot_va): failed to bind official runtime-KV denoise runner "
                             "latent_seq=%lld action_seq=%lld blocks=%d\n",
                             (long long) latent_forward_seq,
                             (long long) action_forward_seq,
                             blocks);
                return {};
            }
        }
#endif
        std::vector<double> latent_token_timesteps;
        std::vector<double> action_token_timesteps;
        auto fill_token_timesteps = [](
                std::vector<double> & buffer,
                const std::vector<unsigned char> & zero_mask,
                int64_t seq,
                double timestep,
                const char * label,
                const std::vector<double> *& out) -> bool {
            out = nullptr;
            if (zero_mask.empty()) return true;
            if ((int64_t) zero_mask.size() != seq) {
                std::fprintf(stderr, "vla(lingbot_va): %s timestep mask seq mismatch\n",
                             label ? label : "token");
                return false;
            }
            buffer.assign((size_t) seq, timestep);
            for (int64_t tok = 0; tok < seq; ++tok) {
                if (zero_mask[(size_t) tok]) {
                    buffer[(size_t) tok] = 0.0;
                }
            }
            out = &buffer;
            return true;
        };
        const char * denoise_dump_dir_c = std::getenv("VLA_LINGBOT_PREDICT_DUMP_DIR");
        const bool denoise_sample_dump =
            denoise_dump_dir_c != nullptr &&
            lingbot_env_enabled("VLA_LINGBOT_DENOISE_SAMPLE_DUMP");
        auto dump_tensor5d = [&](const std::string & stem,
                                 const std::vector<float> & data,
                                 const LingBotTensor5DShape & shape) {
            if (!denoise_sample_dump) return;
            const std::string base = std::string(denoise_dump_dir_c) + "/" + stem;
            dump_f32_file(base + ".f32", data);
            dump_text_file(base + ".shape.txt",
                           std::to_string((long long) shape.b) + " " +
                           std::to_string((long long) shape.c) + " " +
                           std::to_string((long long) shape.f) + " " +
                           std::to_string((long long) shape.h) + " " +
                           std::to_string((long long) shape.w) + "\n");
        };
        auto denoise_step_stem = [](const char * phase, int step, const char * suffix) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                          "lingbot_predict_denoise_%s_step%03d_%s",
                          phase, step, suffix);
            return std::string(buf);
        };
        dump_tensor5d("lingbot_predict_denoise_initial_latent_sample",
                      latent_sample, latent_shape);
        dump_tensor5d("lingbot_predict_denoise_initial_action_sample",
                      action_sample, action_shape);
        if (!latent_condition.empty()) {
            dump_tensor5d("lingbot_predict_denoise_latent_condition",
                          latent_condition, latent_condition_shape);
        }
        if (!action_condition.empty()) {
            dump_tensor5d("lingbot_predict_denoise_action_condition",
                          action_condition, action_shape);
        }
        auto run_latent_forward = [&](LingBotTransformerExecutor & fex,
                                      double timestep,
                                      OfficialForwardTensor & out_tensor,
                                      int cache_mode) -> bool {
            out_tensor.reset();
            std::vector<float> local_tokens;
            std::vector<float> local_out_tokens;
            std::vector<float> & tokens =
                stage5_workspace_enabled ? denoise_ws.latent_tokens : local_tokens;
            std::vector<float> & out_tokens =
                stage5_workspace_enabled ? denoise_ws.latent_out_tokens : local_out_tokens;
            if (stage5_workspace_enabled) {
                tokens.clear();
                out_tokens.clear();
            }
            int64_t feature = 0;
            int64_t seq = 0;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            LingBotDeviceTensor local_device_tokens;
            LingBotDeviceTensor & device_tokens =
                stage5_workspace_enabled ? denoise_ws.latent_device_tokens : local_device_tokens;
            bool use_device_tokens = false;
            if (device_denoise_bridge) {
                use_device_tokens = lingbot_tokenize_device_sample_to_tokens(
                        latent_sample_device, common_ptr->backend, false, latent_shape,
                        patch_t, patch_h, patch_w, device_tokens, feature, seq);
                if (!use_device_tokens) {
                    std::fprintf(stderr, "vla(lingbot_va): official-order latent device tokenization failed\n");
                    return false;
                }
            }
            if (!use_device_tokens)
#else
            const bool use_device_tokens = false;
#endif
            {
                if (!patchify_latent_tokens(latent_sample, latent_shape,
                                            patch_t, patch_h, patch_w,
                                            tokens, feature, seq)) {
                    std::fprintf(stderr, "vla(lingbot_va): official-order latent tokenization failed\n");
                    return false;
                }
            }
            LingBotGridSpec grid = latent_grid_spec;
            const std::vector<double> * token_timesteps = nullptr;
            if (!fill_token_timesteps(latent_token_timesteps, latent_zero_timestep_mask,
                                      seq, timestep, "latent", token_timesteps)) {
                return false;
            }
            const auto f0 = std::chrono::steady_clock::now();
            LingBotExecState forward_state;
            if (!exec_forward_one_streaming(fex, *common_ptr, false,
                                            use_device_tokens ? nullptr : &tokens,
                                            timestep, blocks, out_tokens, &grid,
                                            token_timesteps,
                                            in.lingbot_session_id,
                                            cache_mode,
                                            official_runtime_kv_self_attn,
                                            device_denoise_bridge,
                                            use_device_tokens ? &device_tokens : nullptr,
                                            &forward_state
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
                                            ,
                                            official_denoise_runner
                                                ? lingbot_select_forward_runner_for_mode(
                                                      official_denoise_runner->latent_forward,
                                                      official_denoise_runner->latent_forward_by_mode,
                                                      cache_mode)
                                                : nullptr
#endif
                                            )) {
                return false;
            }
            if (lingbot_env_enabled("VLA_LINGBOT_DEVICE_RESIDENCY_VERBOSE")) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
                const size_t expected_bytes =
                    (size_t) out_channels * (size_t) patch_t * (size_t) patch_h * (size_t) patch_w *
                    (size_t) seq * sizeof(float);
                std::printf("vla(lingbot_va): device residency latent output_tokens device=%d bytes=%zu expected=%zu\n",
                            lingbot_device_out_valid(forward_state, expected_bytes, common_ptr->backend) ? 1 : 0,
                            forward_state.device_out ? forward_state.device_out->bytes : 0,
                            expected_bytes);
#endif
            }
            const auto f1 = std::chrono::steady_clock::now();
            const double forward_ms = std::chrono::duration<double, std::milli>(f1 - f0).count();
            denoise_forward_ms += forward_ms;
            denoise_forward_calls += 1;
            latent_forward_ms += forward_ms;
            latent_forward_calls += 1;
            const LingBotTensor5DShape out_shape{1, out_channels,
                                                 latent_shape.f, latent_shape.h, latent_shape.w};
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            if (device_denoise_bridge) {
                if (!lingbot_detokenize_device_out_to_tensor(
                        forward_state, common_ptr->backend, false, out_shape,
                        patch_t, patch_h, patch_w, out_tensor.device)) {
                    std::fprintf(stderr, "vla(lingbot_va): official-order latent device detokenize failed\n");
                    return false;
                }
                out_tensor.has_device = true;
            } else
#endif
            {
                if (!projected_latent_tokens_to_tensor(out_tokens, out_shape,
                                                       patch_t, patch_h, patch_w,
                                                       out_tensor.host)) {
                    std::fprintf(stderr, "vla(lingbot_va): official-order latent unpatch failed\n");
                    return false;
                }
            }
            return true;
        };
        auto run_action_forward = [&](LingBotTransformerExecutor & fex,
                                      double timestep,
                                      OfficialForwardTensor & out_tensor,
                                      int cache_mode) -> bool {
            out_tensor.reset();
            std::vector<float> local_tokens;
            std::vector<float> local_out_tokens;
            std::vector<float> & tokens =
                stage5_workspace_enabled ? denoise_ws.action_tokens : local_tokens;
            std::vector<float> & out_tokens =
                stage5_workspace_enabled ? denoise_ws.action_out_tokens : local_out_tokens;
            if (stage5_workspace_enabled) {
                tokens.clear();
                out_tokens.clear();
            }
            int64_t feature = 0;
            int64_t seq = 0;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            LingBotDeviceTensor local_device_tokens;
            LingBotDeviceTensor & device_tokens =
                stage5_workspace_enabled ? denoise_ws.action_device_tokens : local_device_tokens;
            bool use_device_tokens = false;
            if (device_denoise_bridge) {
                use_device_tokens = lingbot_tokenize_device_sample_to_tokens(
                        action_sample_device, common_ptr->backend, true, action_shape,
                        1, 1, 1, device_tokens, feature, seq);
                if (!use_device_tokens) {
                    std::fprintf(stderr, "vla(lingbot_va): official-order action device tokenization failed\n");
                    return false;
                }
            }
            if (!use_device_tokens)
#else
            const bool use_device_tokens = false;
#endif
            {
                if (!action_tensor_to_tokens(action_sample, action_shape,
                                             tokens, feature, seq)) {
                    std::fprintf(stderr, "vla(lingbot_va): official-order action tokenization failed\n");
                    return false;
                }
            }
            LingBotGridSpec grid = action_grid_spec;
            const std::vector<double> * token_timesteps = nullptr;
            if (!fill_token_timesteps(action_token_timesteps, action_zero_timestep_mask,
                                      seq, timestep, "action", token_timesteps)) {
                return false;
            }
            const auto f0 = std::chrono::steady_clock::now();
            LingBotExecState forward_state;
            if (!exec_forward_one_streaming(fex, *common_ptr, true,
                                            use_device_tokens ? nullptr : &tokens,
                                            timestep, blocks, out_tokens, &grid,
                                            token_timesteps,
                                            in.lingbot_session_id,
                                            cache_mode,
                                            official_runtime_kv_self_attn,
                                            device_denoise_bridge,
                                            use_device_tokens ? &device_tokens : nullptr,
                                            &forward_state
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
                                            ,
                                            official_denoise_runner
                                                ? lingbot_select_forward_runner_for_mode(
                                                      official_denoise_runner->action_forward,
                                                      official_denoise_runner->action_forward_by_mode,
                                                      cache_mode)
                                                : nullptr
#endif
                                            )) {
                return false;
            }
            if (lingbot_env_enabled("VLA_LINGBOT_DEVICE_RESIDENCY_VERBOSE")) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
                const size_t expected_bytes = lingbot_tensor5d_count(action_shape) * sizeof(float);
                std::printf("vla(lingbot_va): device residency action output_tokens device=%d bytes=%zu expected=%zu\n",
                            lingbot_device_out_valid(forward_state, expected_bytes, common_ptr->backend) ? 1 : 0,
                            forward_state.device_out ? forward_state.device_out->bytes : 0,
                            expected_bytes);
#endif
            }
            const auto f1 = std::chrono::steady_clock::now();
            const double forward_ms = std::chrono::duration<double, std::milli>(f1 - f0).count();
            denoise_forward_ms += forward_ms;
            denoise_forward_calls += 1;
            action_forward_ms += forward_ms;
            action_forward_calls += 1;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            if (device_denoise_bridge) {
                if (!lingbot_detokenize_device_out_to_tensor(
                        forward_state, common_ptr->backend, true, action_shape,
                        1, 1, 1, out_tensor.device)) {
                    std::fprintf(stderr, "vla(lingbot_va): official-order action device detokenize failed\n");
                    return false;
                }
                out_tensor.has_device = true;
            } else
#endif
            {
                if (!action_tokens_to_tensor(out_tokens, action_shape, out_tensor.host)) {
                    std::fprintf(stderr, "vla(lingbot_va): official-order action detokenize failed\n");
                    return false;
                }
            }
            return true;
        };

        std::printf("vla(lingbot_va): predict uses official PyTorch denoise order "
                    "(video loop then action loop)\n");
        for (int step = 0; step < (int) video_loop_timesteps.size(); ++step) {
            const double t = video_loop_timesteps[(size_t) step];
            const bool last_step = step == (int) video_loop_timesteps.size() - 1;
            const auto video_restore0 = std::chrono::steady_clock::now();
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            if (device_denoise_bridge) {
                if (has_latent_condition &&
                    !lingbot_device_latent_restore_round_bridge(
                        common_ptr->backend, latent_sample_device, &latent_condition_device,
                        latent_shape, latent_condition_shape, false)) {
                    return {};
                }
            } else
#endif
            {
                restore_latent_condition();
            }
            video_restore_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - video_restore0).count();
            dump_tensor5d(denoise_step_stem("video", step, "pre_sample"),
                          latent_sample, latent_shape);
            OfficialForwardTensor local_pred_latent;
            OfficialForwardTensor & pred_latent =
                stage5_workspace_enabled ? denoise_ws.latent_pred : local_pred_latent;
            if (stage5_workspace_enabled) pred_latent.reset();
            if (use_cfg) {
                OfficialForwardTensor local_uncond_latent;
                OfficialForwardTensor & uncond_latent =
                    stage5_workspace_enabled ? denoise_ws.latent_uncond : local_uncond_latent;
                if (stage5_workspace_enabled) uncond_latent.reset();
                if (!run_latent_forward(ex_uncond, t, uncond_latent, last_step ? 1 : 0) ||
                    !run_latent_forward(ex, t, pred_latent, last_step ? 1 : 0)) {
                    return {};
                }
                const auto video_guidance0 = std::chrono::steady_clock::now();
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
                if (device_denoise_bridge) {
                    const size_t count = lingbot_tensor5d_count(
                        LingBotTensor5DShape{1, out_channels, latent_shape.f, latent_shape.h, latent_shape.w});
                    if (!uncond_latent.has_device || !pred_latent.has_device ||
                        !lingbot_device_model_output_guidance_bridge(
                            common_ptr->backend, pred_latent.device, &uncond_latent.device,
                            count, video_guidance_scale, video_cfg_enabled, true,
                            "latent_model_output")) {
                        return {};
                    }
                } else
#endif
                {
                if (uncond_latent.host.size() != pred_latent.host.size()) {
                    std::fprintf(stderr, "vla(lingbot_va): official-order CFG latent size mismatch\n");
                    return {};
                }
                round_model_state(uncond_latent.host);
                round_model_state(pred_latent.host);
                dump_tensor5d(denoise_step_stem("video", step, "uncond_pred"),
                              uncond_latent.host,
                              LingBotTensor5DShape{1, out_channels,
                                                    latent_shape.f, latent_shape.h, latent_shape.w});
                dump_tensor5d(denoise_step_stem("video", step, "cond_pred"),
                              pred_latent.host,
                              LingBotTensor5DShape{1, out_channels,
                                                    latent_shape.f, latent_shape.h, latent_shape.w});
                if (video_cfg_enabled) {
                    for (size_t i = 0; i < pred_latent.host.size(); ++i) {
                        pred_latent.host[i] = uncond_latent.host[i] +
                            video_guidance_scale * (pred_latent.host[i] - uncond_latent.host[i]);
                    }
                }
                round_model_state(pred_latent.host);
                }
                video_guidance_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - video_guidance0).count();
            } else if (!run_latent_forward(ex, t, pred_latent, last_step ? 1 : 0)) {
                return {};
            }
            const auto video_guidance0 = std::chrono::steady_clock::now();
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            if (device_denoise_bridge) {
                const size_t count = lingbot_tensor5d_count(
                    LingBotTensor5DShape{1, out_channels, latent_shape.f, latent_shape.h, latent_shape.w});
                if (!pred_latent.has_device ||
                    (!use_cfg &&
                     !lingbot_device_model_output_guidance_bridge(
                         common_ptr->backend, pred_latent.device, nullptr,
                         count, 1.0f, false, true, "latent_model_output"))) {
                    return {};
                }
            } else
#endif
            {
                round_model_state(pred_latent.host);
            }
            dump_tensor5d(denoise_step_stem("video", step, "pred"),
                          pred_latent.host,
                          LingBotTensor5DShape{1, out_channels,
                                                latent_shape.f, latent_shape.h, latent_shape.w});
            video_guidance_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - video_guidance0).count();
            const auto video_sched0 = std::chrono::steady_clock::now();
            if (!last_step && step < (int) video_sched.timesteps.size()) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
                if (device_denoise_bridge) {
                    (void) latent_pred_device;
                    const size_t count = lingbot_tensor5d_count(latent_shape);
                    if (!lingbot_device_scheduler_step_device_pred(
                            common_ptr->backend, latent_sample_device, pred_latent.device,
                            count, video_sched.step_delta(t), bf16_state, "latent_sample")) {
                        return {};
                    }
                } else
#endif
                {
                    video_sched.step_inplace_round(latent_sample, pred_latent.host, t, bf16_state);
                }
            }
            if (has_latent_condition) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
                if (device_denoise_bridge) {
                    if (!lingbot_device_latent_restore_round_bridge(
                            common_ptr->backend, latent_sample_device, &latent_condition_device,
                            latent_shape, latent_condition_shape, bf16_state)) {
                        return {};
                    }
                } else
#endif
                {
                    restore_latent_condition_round(latent_sample, latent_shape,
                                                   latent_condition, latent_condition_shape,
                                                   bf16_state);
                }
            }
            video_scheduler_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - video_sched0).count();
            dump_tensor5d(denoise_step_stem("video", step, "post_sample"),
                          latent_sample, latent_shape);
        }

        for (int step = 0; step < (int) action_loop_timesteps.size(); ++step) {
            const double t = action_loop_timesteps[(size_t) step];
            const bool last_step = step == (int) action_loop_timesteps.size() - 1;
            const auto action_finalize0 = std::chrono::steady_clock::now();
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            if (device_denoise_bridge) {
                const LingBotDeviceTensor * cond_device =
                    action_condition.empty() ? nullptr : &action_condition_device;
                if (!lingbot_device_action_finalize_bridge(
                        common_ptr->backend, action_sample_device, cond_device,
                        action_shape, bf16_state)) {
                    return {};
                }
            } else
#endif
            {
                finalize_libero_action_sample(action_sample,
                                              action_condition.empty() ? nullptr : &action_condition,
                                              action_shape,
                                              bf16_state);
            }
            action_finalize_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - action_finalize0).count();
            dump_tensor5d(denoise_step_stem("action", step, "pre_sample"),
                          action_sample, action_shape);
            OfficialForwardTensor local_uncond_action;
            OfficialForwardTensor & uncond_action =
                stage5_workspace_enabled ? denoise_ws.action_uncond : local_uncond_action;
            if (stage5_workspace_enabled) uncond_action.reset();
            if (use_cfg) {
                if (!run_action_forward(ex_uncond, t, uncond_action, last_step ? 1 : 0)) {
                    return {};
                }
            }
            OfficialForwardTensor local_pred_action;
            OfficialForwardTensor & pred_action =
                stage5_workspace_enabled ? denoise_ws.action_pred : local_pred_action;
            if (stage5_workspace_enabled) pred_action.reset();
            if (!run_action_forward(ex, t, pred_action, last_step ? 1 : 0)) {
                return {};
            }
            const auto action_guidance0 = std::chrono::steady_clock::now();
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            if (device_denoise_bridge) {
                const size_t count = lingbot_tensor5d_count(action_shape);
                if (!pred_action.has_device ||
                    !lingbot_device_model_output_guidance_bridge(
                        common_ptr->backend, pred_action.device, nullptr,
                        count, 1.0f, false, true, "action_model_output")) {
                    return {};
                }
            } else
#endif
            {
                round_model_state(pred_action.host);
            }
            dump_tensor5d(denoise_step_stem("action", step, "pred"),
                          pred_action.host, action_shape);
            if (action_cfg_enabled) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
                if (device_denoise_bridge) {
                    const size_t count = lingbot_tensor5d_count(action_shape);
                    if (!uncond_action.has_device || !pred_action.has_device ||
                        !lingbot_device_model_output_guidance_bridge(
                            common_ptr->backend, pred_action.device, &uncond_action.device,
                            count, action_guidance_scale, true, true,
                            "action_model_output")) {
                        return {};
                    }
                } else
#endif
                {
                if (uncond_action.host.size() != pred_action.host.size()) {
                    std::fprintf(stderr, "vla(lingbot_va): official-order CFG action size mismatch\n");
                    return {};
                }
                round_model_state(uncond_action.host);
                dump_tensor5d(denoise_step_stem("action", step, "uncond_pred"),
                              uncond_action.host, action_shape);
                dump_tensor5d(denoise_step_stem("action", step, "cond_pred"),
                              pred_action.host, action_shape);
                for (size_t i = 0; i < pred_action.host.size(); ++i) {
                    pred_action.host[i] = uncond_action.host[i] +
                        action_guidance_scale * (pred_action.host[i] - uncond_action.host[i]);
                }
                round_model_state(pred_action.host);
                }
            }
            action_guidance_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - action_guidance0).count();
            const auto action_sched0 = std::chrono::steady_clock::now();
            if (!last_step && step < (int) action_sched.timesteps.size()) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
                if (device_denoise_bridge) {
                    (void) action_pred_device;
                    const size_t count = lingbot_tensor5d_count(action_shape);
                    if (!lingbot_device_scheduler_step_device_pred(
                            common_ptr->backend, action_sample_device, pred_action.device,
                            count, action_sched.step_delta(t), bf16_state, "action_sample")) {
                        return {};
                    }
                } else
#endif
                {
                    action_sched.step_inplace_round(action_sample, pred_action.host, t, bf16_state);
                }
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
                if (device_denoise_bridge) {
                    const LingBotDeviceTensor * cond_device =
                        action_condition.empty() ? nullptr : &action_condition_device;
                    if (!lingbot_device_action_finalize_bridge(
                            common_ptr->backend, action_sample_device, cond_device,
                            action_shape, bf16_state)) {
                        return {};
                    }
                } else
#endif
                {
                    if (!action_condition.empty()) {
                        apply_action_condition_frame0(action_sample, action_condition, action_shape);
                    }
                    round_model_state(action_sample);
                }
            }
            action_scheduler_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - action_sched0).count();
            dump_tensor5d(denoise_step_stem("action", step, "post_sample"),
                          action_sample, action_shape);
        }
    }
    const auto t_inf1 = std::chrono::steady_clock::now();
    if (std::getenv("VLA_LINGBOT_TIMING")) {
        std::printf("vla(lingbot_va): timing denoise_forwards calls=%d total=%.3fms avg=%.3fms\n",
                    denoise_forward_calls,
                    denoise_forward_ms,
                    denoise_forward_calls > 0 ? denoise_forward_ms / (double) denoise_forward_calls : 0.0);
    }

    const auto final_output0 = std::chrono::steady_clock::now();
    std::vector<float> out;
    const bool postprocess_libero = lingbot_action_postprocess_libero_enabled();
    int64_t output_action_dim = postprocess_libero ? cfg.real_action_dim : cfg.max_action_dim;
    std::vector<float> action_tokens;
    int64_t action_feature = 0;
    int64_t action_seq = 0;
    const char * dump_dir_c = std::getenv("VLA_LINGBOT_PREDICT_DUMP_DIR");
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    bool action_sample_downloaded = false;
    if (use_official_order && device_denoise_bridge) {
        const LingBotDeviceTensor * cond_device =
            action_condition.empty() ? nullptr : &action_condition_device;
        if (!lingbot_device_action_finalize_bridge(
                common_ptr->backend, action_sample_device, cond_device,
                action_shape, bf16_state)) {
            return {};
        }
        if (!dump_dir_c) {
            if (!lingbot_device_action_sample_to_output_bridge(
                    common_ptr->backend, action_sample_device, action_shape,
                    n_suffix, output_action_dim, postprocess_libero, out)) {
                return {};
            }
        } else {
            if (!lingbot_download_device_tensor(action_sample_device, common_ptr->backend,
                                                action_sample, "action_sample_final") ||
                !lingbot_download_device_tensor(latent_sample_device, common_ptr->backend,
                                                latent_sample, "latent_sample_final")) {
                return {};
            }
            action_sample_downloaded = true;
        }
    }
#endif
    if (out.empty()) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
        if (use_official_order && device_denoise_bridge && !action_sample_downloaded &&
            !lingbot_download_device_tensor(action_sample_device, common_ptr->backend,
                                            action_sample, "action_sample_final")) {
            return {};
        }
#endif
        zero_unused_libero_action_channels(action_sample, action_shape);
        round_model_state(action_sample);
        if (!action_tensor_to_tokens(action_sample, action_shape, action_tokens, action_feature, action_seq)) {
            std::fprintf(stderr, "vla(lingbot_va): predict action flatten failed\n");
            return {};
        }
        if (postprocess_libero) {
            if (!postprocess_libero_action_tokens(action_tokens, action_feature, action_seq,
                                                  n_suffix, out)) {
                return {};
            }
        } else {
            out.assign((size_t) n_suffix * (size_t) cfg.max_action_dim, 0.0f);
            const int64_t copy_dim = std::min<int64_t>(cfg.max_action_dim, action_feature);
            const int64_t copy_steps = std::min<int64_t>(n_suffix, action_seq);
            for (int64_t t = 0; t < copy_steps; ++t) {
                for (int64_t c = 0; c < copy_dim; ++c) {
                    out[(size_t) t * (size_t) cfg.max_action_dim + (size_t) c] =
                        action_tokens[(size_t) c + (size_t) action_feature * (size_t) t];
                }
            }
        }
    }

    if (dump_dir_c) {
        const std::string dump_dir(dump_dir_c);
        dump_f32_file(dump_dir + "/lingbot_predict_action_chunk.f32", out);
        dump_text_file(dump_dir + "/lingbot_predict_action_chunk.shape.txt",
                       std::to_string((long long) n_suffix) + " " +
                       std::to_string((long long) output_action_dim) + "\n");
        if (action_tokens.empty()) {
            (void) action_tensor_to_tokens(action_sample, action_shape,
                                           action_tokens, action_feature, action_seq);
        }
        if (!action_tokens.empty()) {
            dump_f32_file(dump_dir + "/lingbot_predict_action_tokens_raw.f32", action_tokens);
            dump_text_file(dump_dir + "/lingbot_predict_action_tokens_raw.shape.txt",
                           std::to_string((long long) action_feature) + " " +
                           std::to_string((long long) action_seq) + "\n");
        }
        dump_f32_file(dump_dir + "/lingbot_predict_action_sample_final.f32", action_sample);
        dump_text_file(dump_dir + "/lingbot_predict_action_sample_final.shape.txt",
                       std::to_string((long long) action_shape.b) + " " +
                       std::to_string((long long) action_shape.c) + " " +
                       std::to_string((long long) action_shape.f) + " " +
                       std::to_string((long long) action_shape.h) + " " +
                       std::to_string((long long) action_shape.w) + "\n");
        dump_f32_file(dump_dir + "/lingbot_predict_latent_sample_final.f32", latent_sample);
        dump_text_file(dump_dir + "/lingbot_predict_latent_sample_final.shape.txt",
                       std::to_string((long long) latent_shape.b) + " " +
                       std::to_string((long long) latent_shape.c) + " " +
                       std::to_string((long long) latent_shape.f) + " " +
                       std::to_string((long long) latent_shape.h) + " " +
                       std::to_string((long long) latent_shape.w) + "\n");
    }
    final_output_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - final_output0).count();
    if (std::getenv("VLA_LINGBOT_TIMING")) {
        std::printf("vla(lingbot_va): timing official-order coarse "
                    "latent_forward=%d/%.3fms/%.3fms action_forward=%d/%.3fms/%.3fms "
                    "video_restore=%.3fms video_guidance=%.3fms video_scheduler=%.3fms "
                    "action_finalize=%.3fms action_guidance=%.3fms action_scheduler=%.3fms "
                    "final_output=%.3fms\n",
                    latent_forward_calls,
                    latent_forward_ms,
                    latent_forward_calls > 0 ? latent_forward_ms / (double) latent_forward_calls : 0.0,
                    action_forward_calls,
                    action_forward_ms,
                    action_forward_calls > 0 ? action_forward_ms / (double) action_forward_calls : 0.0,
                    video_restore_ms,
                    video_guidance_ms,
                    video_scheduler_ms,
                    action_finalize_ms,
                    action_guidance_ms,
                    action_scheduler_ms,
                    final_output_ms);
    }

    const auto t_total1 = std::chrono::steady_clock::now();
    stats.ms_vision = (float) vae_encode_ms;
    stats.ms_prefill = (float) text_encode_ms;
    stats.ms_denoise = std::chrono::duration<float, std::milli>(t_inf1 - t_inf0).count();
    stats.ms_total = std::chrono::duration<float, std::milli>(t_total1 - t_total0).count();
    stats.ms_inference = stats.ms_total;
    const double other_ms = std::max<double>(
        0.0,
        (double) stats.ms_total - vae_encode_ms - text_encode_ms -
            (double) stats.ms_denoise - final_output_ms);

    std::printf("vla(lingbot_va): predict bridge ok blocks=%d mode=%s window=%lld "
                "video_steps=%d action_steps=%d action_postprocess=%s chunk=[%lld,%lld] checksum=%.9g max=%.9g\n",
                blocks, use_cuda_self_attn ? "official-runtime-kv-device-cublas" : "ggml-dense",
                (long long) window_size, video_steps, action_steps,
                postprocess_libero ? "libero_quantiles" : "raw",
                (long long) n_suffix, (long long) output_action_dim,
                checksum(out), max_abs_value(out));
    std::printf("vla(lingbot_va): timing model kind=predict total=%.3fms text=%.3fms "
                "vae=%.3fms denoise=%.3fms output=%.3fms other=%.3fms\n",
                stats.ms_total, text_encode_ms, vae_encode_ms,
                stats.ms_denoise, final_output_ms, other_ms);
    return out;
}

}

std::unique_ptr<ModelArchBase> lingbot_va_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string&,
                                                 const LingBotComponentPaths& components) {
    if (!mmproj_path.empty()) {
        std::printf("vla(lingbot_va): note - mmproj '%s' is ignored; "
                    "LingBot-VA transformer is bundled in the GGUF\n",
                    mmproj_path.c_str());
    }

    gguf_reader g;
    if (!g.open(ckpt_path)) return nullptr;
    if (!g.has("lingbot_va.architecture") || g.str("lingbot_va.architecture") != "lingbot_va") {
        std::fprintf(stderr, "vla(lingbot_va): %s is not a LingBot-VA GGUF\n", ckpt_path.c_str());
        return nullptr;
    }

    auto m = std::make_unique<LingBotVAModelArch>(ckpt_path, g, components);
    if (!validate_transformer_tensors(g, m->n_layers)) return nullptr;
    if (!load_full_transformer_weights_if_requested(g, *m)) return nullptr;

    std::printf("vla(lingbot_va): metadata loaded from %s\n", ckpt_path.c_str());
    std::printf("vla(lingbot_va): transformer=%lld layers, hidden=%lld, heads=%lld, head_dim=%lld, "
                "ffn=%lld, patch=[%lld,%lld,%lld], action_dim=%lld, attn_mode_config=%s\n",
                (long long) m->n_layers,
                (long long) m->cfg.hidden,
                (long long) m->n_heads,
                (long long) m->head_dim,
                (long long) m->ffn_dim,
                (long long) m->patch_t,
                (long long) m->patch_h,
                (long long) m->patch_w,
                (long long) m->action_dim,
                m->attn_mode_config.c_str());
    std::printf("vla(lingbot_va): runtime WanTransformer, UMT5, VAE, and LIBERO postprocess paths are available\n");
    return m;
}


}
