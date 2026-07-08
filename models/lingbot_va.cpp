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
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
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

namespace vla {

namespace {

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

class LingBotVAModelArch final : public ModelArchBase {
public:
    LingBotVAModelArch(const std::string& ckpt_path, const gguf_reader & g)
        : ModelArchBase(Arch::LINGBOT_VA), ckpt_path(ckpt_path) {
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

    ~LingBotVAModelArch() override {
        if (weight_buf)  ggml_backend_buffer_free(weight_buf);
        if (ctx_weights) ggml_free(ctx_weights);
    }

    std::vector<float> predict(const Inputs& in) override;

    bool text_cache_lookup(
            const std::string & path,
            int blocks,
            const std::vector<int32_t> & ids,
            std::vector<float> & out,
            int64_t & seq,
            int64_t & dim) const {
        const auto it = text_cache.find(text_cache_key(path, blocks, ids));
        if (it == text_cache.end() || it->second.hidden.empty()) {
            return false;
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
        constexpr size_t kMaxTextCacheEntries = 16;
        if (text_cache.size() >= kMaxTextCacheEntries) {
            text_cache.erase(text_cache.begin());
        }
        text_cache[text_cache_key(path, blocks, ids)] = TextCacheEntry{hidden, seq, dim};
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
    ggml_tensor * a2 = build_attention_shape(C, b.cross_attn, n2, text, nullptr, nullptr, m, seq, text_seq);
    x = ggml_add(C, x, a2);
    if (trace) {
        trace->n2 = n2;
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
                 "using f32. Supported: f32, q8_0, q6_K, q5_K, q4_K, q5_1, q5_0, q4_1, q4_0, q3_K, q2_K\n",
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



void fill_deterministic(std::vector<float> & v, float scale) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = std::sin((float) i * 0.013f) * scale;
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

bool set_tensor_from_gguf_f32(gguf_reader & g, ggml_tensor * t) {
    std::vector<float> tmp((size_t) ggml_nelements(t));
    if (!g.read_to_f32(ggml_get_name(t), tmp.data(), ggml_nelements(t))) {
        return false;
    }
    ggml_backend_tensor_set(t, tmp.data(), 0, tmp.size() * sizeof(float));
    return true;
}

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

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(C);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_backend_free(backend);
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
    ggml_backend_free(backend);
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

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(C);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_backend_free(backend);
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
    ggml_backend_free(backend);
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

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(C);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_backend_free(backend);
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
    ggml_backend_free(backend);
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

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(C);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_backend_free(backend);
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
    ggml_backend_free(backend);
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
    const int group_size = (in_C * factor) / out_C;
    const int pad_t = (factor_t - (T % factor_t)) % factor_t;
    const int Tp = T + pad_t;
    const int Wo = W / factor_s;
    const int Ho = H / factor_s;
    const int To = Tp / factor_t;
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

    void update_from_chunk(const std::vector<float> & chunk, int w, int h, int t, int c) {
        W = w;
        H = h;
        C = c;
        T = std::min(2, t);
        if (t <= 0 || chunk.size() != (size_t) w * h * t * c) {
            T = 0;
            data.clear();
            return;
        }
        data.assign((size_t) W * H * T * C, 0.0f);
        const int src_t0 = t - T;
        for (int iw = 0; iw < W; ++iw) {
            for (int ih = 0; ih < H; ++ih) {
                for (int it = 0; it < T; ++it) {
                    for (int ic = 0; ic < C; ++ic) {
                        data[vae_ggml_whdc_index(iw, ih, it, ic, W, H, T)] =
                            chunk[vae_ggml_whdc_index(iw, ih, src_t0 + it, ic, W, H, t)];
                    }
                }
            }
        }
    }
};

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

    bool load(gguf_reader & g, const char * prefix, int channels, int kernel) {
        release();
        C = channels;
        K = kernel;
        const std::string w_name = std::string(prefix) + ".weight";
        const std::string b_name = std::string(prefix) + ".bias";
        host_w.assign((size_t) C * C * K, 0.0f);
        host_b.assign((size_t) C, 0.0f);
        if (!g.read_to_f32(w_name.c_str(), host_w.data(), (int64_t) host_w.size())) return false;
        if (!g.read_to_f32(b_name.c_str(), host_b.data(), (int64_t) host_b.size())) return false;
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
        if (d_w) cudaFree(d_w);
        if (d_b) cudaFree(d_b);
        d_w = nullptr;
        d_b = nullptr;
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

    bool load(gguf_reader & g, const char * prefix, int in_channels, int out_channels, int kernel) {
        release();
        in_C = in_channels;
        out_C = out_channels;
        K = kernel;
        const std::string w_name = std::string(prefix) + ".weight";
        const std::string b_name = std::string(prefix) + ".bias";
        host_w.assign((size_t) out_C * in_C * K, 0.0f);
        host_b.assign((size_t) out_C, 0.0f);
        if (!g.read_to_f32(w_name.c_str(), host_w.data(), (int64_t) host_w.size())) return false;
        if (!g.read_to_f32(b_name.c_str(), host_b.data(), (int64_t) host_b.size())) return false;
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
        if (d_w) cudaFree(d_w);
        if (d_b) cudaFree(d_b);
        d_w = nullptr;
        d_b = nullptr;
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
        int * out_T_total) {
    if (W <= 0 || H <= 0 || T_total <= 0 || weights.in_C <= 0 || weights.out_C <= 0 || weights.K != 3) return false;
    if (weights.in_C != weights.out_C) return false;
    if (spatial_whdc.size() != (size_t) W * H * T_total * weights.in_C) return false;
    int chunk_sum = 0;
    int total_out = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
        const int chunk = chunks[i];
        if (chunk <= 0) return false;
        chunk_sum += chunk;
        total_out += i == 0 ? chunk : ((chunk + 1) >= weights.K ? ((chunk + 1 - weights.K) / 2 + 1) : 0);
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
    if (!vae_downsample3d_time_stream_host(spatial, out_whdc, weights, sw, sh, T, chunks, &time_out)) {
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
    int sc_w = 0, sc_h = 0, sc_t = 0;
    if (!vae_avg_down3d_host(in_whdc, shortcut, W, H, T, in_C, out_C, 2, 2, &sc_w, &sc_h, &sc_t) ||
        sc_w != sw || sc_h != sh || sc_t != time_out || !vae_add_same_shape(out_whdc, shortcut)) {
        std::fprintf(stderr, "vla(lingbot_va): VAE down block %s AvgDown3D shortcut failed\n", block_prefix);
        return false;
    }
    dump_stage("shortcut", shortcut, sc_w, sc_h, sc_t, out_C);
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

bool vae_encoder_down_path_execute(
        gguf_reader & g,
        const std::vector<float> & x,
        int W,
        int H,
        int T,
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
        int * out_t2) {
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
    auto unit_chunks = [](int n) {
        std::vector<int> chunks;
        chunks.reserve((size_t) std::max(n, 0));
        for (int i = 0; i < n; ++i) chunks.push_back(1);
        return chunks;
    };
    auto temporal_down_chunks = [](int n) {
        std::vector<int> chunks;
        if (n <= 0) return chunks;
        chunks.push_back(1);
        int remain = n - 1;
        while (remain > 0) {
            const int chunk = std::min(2, remain);
            chunks.push_back(chunk);
            remain -= chunk;
        }
        return chunks;
    };

    int w1 = 0;
    int h1 = 0;
    int t1 = 0;
    if (!vae_down_block_with_time_execute(g, "vae.encoder.down_blocks.1", b0, w0, h0, T, 160, 320, temporal_down_chunks(T), b1, &w1, &h1, &t1)) {
        std::fprintf(stderr, "vla(lingbot_va): VAE encoder down path failed at block1\n");
        return false;
    }
    dump_stage("block1", b1, w1, h1, t1, 320);

    int w2 = 0;
    int h2 = 0;
    int t2 = 0;
    if (!vae_down_block_with_time_execute(g, "vae.encoder.down_blocks.2", b1, w1, h1, t1, 320, 640, temporal_down_chunks(t1), b2, &w2, &h2, &t2)) {
        std::fprintf(stderr, "vla(lingbot_va): VAE encoder down path failed at block2\n");
        return false;
    }
    dump_stage("block2", b2, w2, h2, t2, 640);

    std::vector<int> block3_chunks;
    block3_chunks.reserve((size_t) t2);
    for (int i = 0; i < t2; ++i) block3_chunks.push_back(1);
    const std::vector<float> b2_shortcut = b2;
    std::vector<float> b3;
    if (!vae_mid_resnet_stream_host_execute(g, "vae.encoder.down_blocks.3.resnets.0",
                                            b2, w2, h2, t2, 640, block3_chunks, b3) ||
        !vae_mid_resnet_stream_host_execute(g, "vae.encoder.down_blocks.3.resnets.1",
                                            b3, w2, h2, t2, 640, block3_chunks, b2)) {
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

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(C);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_backend_free(backend);
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
    ggml_backend_free(backend);
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

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(C);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_backend_free(backend);
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
    ggml_backend_free(backend);
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

    std::vector<float> gamma((size_t) Cc);
    std::vector<float> qkv_w((size_t) qkv_C * Cc);
    std::vector<float> qkv_b((size_t) qkv_C);
    std::vector<float> proj_w((size_t) Cc * Cc);
    std::vector<float> proj_b((size_t) Cc);
    const std::string p(prefix);
    if (!g.read_to_f32((p + ".norm.gamma").c_str(), gamma.data(), (int64_t) gamma.size()) ||
        !g.read_to_f32((p + ".to_qkv.weight").c_str(), qkv_w.data(), (int64_t) qkv_w.size()) ||
        !g.read_to_f32((p + ".to_qkv.bias").c_str(), qkv_b.data(), (int64_t) qkv_b.size()) ||
        !g.read_to_f32((p + ".proj.weight").c_str(), proj_w.data(), (int64_t) proj_w.size()) ||
        !g.read_to_f32((p + ".proj.bias").c_str(), proj_b.data(), (int64_t) proj_b.size())) {
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

    std::vector<float> gamma((size_t) Cc);
    std::vector<float> qkv_w((size_t) 3 * Cc * Cc);
    std::vector<float> qkv_b((size_t) 3 * Cc);
    std::vector<float> proj_w((size_t) Cc * Cc);
    std::vector<float> proj_b((size_t) Cc);
    const std::string p(prefix);
    if (!g.read_to_f32((p + ".norm.gamma").c_str(), gamma.data(), (int64_t) gamma.size()) ||
        !g.read_to_f32((p + ".to_qkv.weight").c_str(), qkv_w.data(), (int64_t) qkv_w.size()) ||
        !g.read_to_f32((p + ".to_qkv.bias").c_str(), qkv_b.data(), (int64_t) qkv_b.size()) ||
        !g.read_to_f32((p + ".proj.weight").c_str(), proj_w.data(), (int64_t) proj_w.size()) ||
        !g.read_to_f32((p + ".proj.bias").c_str(), proj_b.data(), (int64_t) proj_b.size())) {
        return false;
    }

    out_whdc.assign(in_whdc.size(), 0.0f);
    float * d_in = nullptr;
    float * d_gamma = nullptr;
    float * d_qkv_w = nullptr;
    float * d_qkv_b = nullptr;
    float * d_proj_w = nullptr;
    float * d_proj_b = nullptr;
    float * d_out = nullptr;
    auto cleanup = [&]() {
        if (d_in) cudaFree(d_in);
        if (d_gamma) cudaFree(d_gamma);
        if (d_qkv_w) cudaFree(d_qkv_w);
        if (d_qkv_b) cudaFree(d_qkv_b);
        if (d_proj_w) cudaFree(d_proj_w);
        if (d_proj_b) cudaFree(d_proj_b);
        if (d_out) cudaFree(d_out);
    };
    const size_t in_bytes = in_whdc.size() * sizeof(float);
    if (cudaMalloc(&d_in, in_bytes) != cudaSuccess ||
        cudaMalloc(&d_gamma, gamma.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_qkv_w, qkv_w.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_qkv_b, qkv_b.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_proj_w, proj_w.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_proj_b, proj_b.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out, in_bytes) != cudaSuccess) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA malloc failed for VAE mid attention\n");
        cleanup();
        return false;
    }
    if (cudaMemcpy(d_in, in_whdc.data(), in_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_gamma, gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_qkv_w, qkv_w.data(), qkv_w.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_qkv_b, qkv_b.data(), qkv_b.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_proj_w, proj_w.data(), proj_w.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_proj_b, proj_b.data(), proj_b.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
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
    if (std::getenv("VLA_LINGBOT_VAE_MID_ATTN_CUDA") != nullptr) {
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

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(C);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_backend_free(backend);
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
    ggml_backend_free(backend);
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

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(C);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_backend_free(backend);
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
    ggml_backend_free(backend);
    ggml_free(C);
    if (ok) cache.update_from_chunk(src_in, W, H, T, in_C);
    return ok;
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
    std::vector<float> n1((size_t) Cc);
    std::vector<float> n2((size_t) Cc);
    if (!g.read_to_f32((p + ".norm1.gamma").c_str(), n1.data(), (int64_t) n1.size()) ||
        !g.read_to_f32((p + ".norm2.gamma").c_str(), n2.data(), (int64_t) n2.size())) {
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
    std::vector<float> norm((size_t) Cc);
    std::vector<float> q_w((size_t) out_C * out_C);
    std::vector<float> q_b((size_t) out_C);
    if (!g.read_to_f32("vae.encoder.norm_out.gamma", norm.data(), (int64_t) norm.size()) ||
        !g.read_to_f32("vae.quant_conv.weight", q_w.data(), (int64_t) q_w.size()) ||
        !g.read_to_f32("vae.quant_conv.bias", q_b.data(), (int64_t) q_b.size())) {
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

    return vae_conv1x1x1_host(conv_out, out_whdc, q_w, q_b, W, H, T, out_C, out_C);
}

bool vae_encoder_mid_tail_execute(
        gguf_reader & g,
        const std::vector<float> & in_whdc,
        int W,
        int H,
        int T,
        int Cc,
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
    chunks.reserve((size_t) T);
    for (int i = 0; i < T; ++i) chunks.push_back(1);

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





bool encode_lingbot_video_to_latent(
        const std::string & vae_path,
        const float * video_vcfhw,
        int64_t views,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
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
    gguf_reader vg;
    if (!vg.open(vae_path)) return false;
    if (!validate_vae_encoder_tensors(vg)) return false;

    const int z_dim = (int) vg.u32("lingbot_va.vae.z_dim");
    if (z_dim != 48) {
        std::fprintf(stderr, "vla(lingbot_va): expected VAE z_dim=48, got %d\n", z_dim);
        return false;
    }
    std::vector<float> latents_mean;
    std::vector<float> latents_std;
    if (!vg.f32_array("lingbot_va.vae.latents_mean", latents_mean) ||
        !vg.f32_array("lingbot_va.vae.latents_std", latents_std)) {
        return false;
    }
    if ((int) latents_mean.size() != z_dim || (int) latents_std.size() != z_dim) {
        std::fprintf(stderr,
                     "vla(lingbot_va): VAE latent stat size mismatch mean=%zu std=%zu z_dim=%d\n",
                     latents_mean.size(), latents_std.size(), z_dim);
        return false;
    }

    int latent_W_single = 0;
    int latent_H = 0;
    int latent_T = 0;
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
        if (!vae_encoder_down_path_execute(vg, patch, patch_W, patch_H, (int) frames,
                                           b0, &w0, &h0, b1, &w1, &h1, &t1,
                                           b2, &w2, &h2, &t2)) {
            return false;
        }

        std::vector<float> enc96;
        if (!vae_encoder_mid_tail_execute(vg, b2, w2, h2, t2, 640,
                                          enc96, nullptr, nullptr, nullptr)) {
            return false;
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

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(C);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_backend_free(backend);
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
    ggml_backend_free(backend);
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

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(C);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_backend_free(backend);
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
    ggml_backend_free(backend);
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

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(C);
        return false;
    }
    ggml_backend_cpu_set_n_threads(backend, 4);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(C, ggml_backend_cpu_buffer_type());
    if (!buf) {
        ggml_backend_free(backend);
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
    ggml_backend_free(backend);
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

    std::vector<int64_t> allocate_slots(int64_t key_size) {
        std::vector<int64_t> free;
        for (int64_t i = 0; i < total_tokens; ++i) {
            if (!mask[(size_t) i]) free.push_back(i);
        }
        if ((int64_t) free.size() < key_size) {
            std::vector<int64_t> used;
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
                is_pred[(size_t) slot] = 0;
                free.push_back(slot);
            }
            std::sort(free.begin(), free.end());
        }
        if ((int64_t) free.size() < key_size) {
            return {};
        }
        free.resize((size_t) key_size);
        return free;
    }

    size_t kv_offset(int64_t b, int64_t slot, int64_t h, int64_t d) const {
        return (((size_t) b * (size_t) total_tokens + (size_t) slot) *
                (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d;
    }

    bool update(const std::vector<float> & key, const std::vector<float> & value,
                int64_t key_size, bool pred, std::vector<int64_t> & slots) {
        const size_t expected = (size_t) batch * (size_t) key_size * (size_t) heads * (size_t) head_dim;
        if (key.size() != expected || value.size() != expected || key_size <= 0) {
            return false;
        }
        slots = allocate_slots(key_size);
        if ((int64_t) slots.size() != key_size) {
            return false;
        }
        const int64_t new_id = next_id();
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t i = 0; i < key_size; ++i) {
                const int64_t slot = slots[(size_t) i];
                for (int64_t h = 0; h < heads; ++h) {
                    for (int64_t d = 0; d < head_dim; ++d) {
                        const size_t src = (((size_t) b * (size_t) key_size + (size_t) i) *
                                            (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d;
                        const size_t dst = kv_offset(b, slot, h, d);
                        k[dst] = key[src];
                        v[dst] = value[src];
                    }
                }
                mask[(size_t) slot] = 1;
                id[(size_t) slot] = new_id;
                is_pred[(size_t) slot] = pred ? 1 : 0;
            }
        }
        return true;
    }

    void restore(const std::vector<int64_t> & slots) {
        for (int64_t slot : slots) {
            if (slot >= 0 && slot < total_tokens) {
                mask[(size_t) slot] = 0;
            }
        }
    }

    void clear_pred() {
        for (size_t i = 0; i < mask.size(); ++i) {
            if (is_pred[i]) {
                mask[i] = 0;
                is_pred[i] = 0;
                id[i] = -1;
            }
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
        out_k.assign(n, 0.0f);
        out_v.assign(n, 0.0f);
        for (int64_t b = 0; b < batch; ++b) {
            for (size_t vi = 0; vi < valid_slots.size(); ++vi) {
                const int64_t slot = valid_slots[vi];
                for (int64_t h = 0; h < heads; ++h) {
                    for (int64_t d = 0; d < head_dim; ++d) {
                        const size_t dst = (((size_t) b * valid_slots.size() + vi) *
                                            (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d;
                        out_k[dst] = k[kv_offset(b, slot, h, d)];
                        out_v[dst] = v[kv_offset(b, slot, h, d)];
                    }
                }
            }
        }
        return true;
    }
};

struct LingBotRuntimeKVKey {
    const void * model = nullptr;
    uint64_t session = 0;
    int64_t block = 0;

    bool operator==(const LingBotRuntimeKVKey & other) const {
        return model == other.model && session == other.session && block == other.block;
    }
};

struct LingBotRuntimeKVKeyHash {
    size_t operator()(const LingBotRuntimeKVKey & k) const {
        size_t h = std::hash<const void *>{}(k.model);
        h ^= std::hash<uint64_t>{}(k.session + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        h ^= std::hash<int64_t>{}(k.block + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        return h;
    }
};

std::mutex g_lingbot_runtime_kv_mu;
std::unordered_map<LingBotRuntimeKVKey, LingBotKVCache, LingBotRuntimeKVKeyHash> g_lingbot_runtime_kv;

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
    std::vector<float> raw;
    if (!tensor_to_f32_vector(t, raw)) return false;
    const size_t expected = (size_t) seq * (size_t) heads * (size_t) head_dim;
    if (raw.size() != expected) return false;
    out.assign(expected, 0.0f);
    for (int64_t s = 0; s < seq; ++s) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const size_t src = (size_t) d + (size_t) head_dim * ((size_t) h + (size_t) heads * (size_t) s);
                const size_t dst = ((size_t) s * (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d;
                out[dst] = raw[src];
            }
        }
    }
    return true;
}

std::vector<float> shd_to_hidden_seq(
        const std::vector<float> & shd,
        int64_t seq,
        int64_t heads,
        int64_t head_dim) {
    std::vector<float> out((size_t) seq * (size_t) heads * (size_t) head_dim, 0.0f);
    const int64_t hidden = heads * head_dim;
    for (int64_t s = 0; s < seq; ++s) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const size_t src = ((size_t) s * (size_t) heads + (size_t) h) * (size_t) head_dim + (size_t) d;
                const size_t dst = (size_t) (h * head_dim + d) + (size_t) hidden * (size_t) s;
                out[dst] = shd[src];
            }
        }
    }
    return out;
}

#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
bool run_lingbot_cuda_attention_qk(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const LingBotBlockSparseTable & table,
        const std::vector<unsigned char> & token_mask,
        int64_t seq_q,
        int64_t seq_k,
        int64_t heads,
        int64_t head_dim,
        std::vector<float> & out) {
    const size_t nq = (size_t) seq_q * (size_t) heads * (size_t) head_dim;
    const size_t nk = (size_t) seq_k * (size_t) heads * (size_t) head_dim;
    if (q.size() != nq || k.size() != nk || v.size() != nk ||
        token_mask.size() != (size_t) seq_q * (size_t) seq_k) {
        std::fprintf(stderr, "vla(lingbot_va): invalid CUDA attention input sizes\n");
        return false;
    }

    std::vector<int> row_ptr = i64_to_i32(table.row_ptr);
    std::vector<int> col_idx = i64_to_i32(table.col_idx);
    out.assign(nq, 0.0f);
    float * dq = nullptr; float * dk = nullptr; float * dv = nullptr; float * dout = nullptr;
    int * drp = nullptr; int * dci = nullptr; unsigned char * dm = nullptr;
    auto ok = [](cudaError_t e, const char * what) {
        if (e != cudaSuccess) std::fprintf(stderr, "vla(lingbot_va): %s failed: %s\n", what, cudaGetErrorString(e));
        return e == cudaSuccess;
    };
    if (!ok(cudaMalloc(&dq, nq * sizeof(float)), "cudaMalloc(q)") ||
        !ok(cudaMalloc(&dk, nk * sizeof(float)), "cudaMalloc(k)") ||
        !ok(cudaMalloc(&dv, nk * sizeof(float)), "cudaMalloc(v)") ||
        !ok(cudaMalloc(&dout, nq * sizeof(float)), "cudaMalloc(out)") ||
        !ok(cudaMalloc(&drp, row_ptr.size() * sizeof(int)), "cudaMalloc(row_ptr)") ||
        !ok(cudaMalloc(&dci, col_idx.size() * sizeof(int)), "cudaMalloc(col_idx)") ||
        !ok(cudaMalloc(&dm, token_mask.size() * sizeof(unsigned char)), "cudaMalloc(mask)")) {
        if (dq) cudaFree(dq); if (dk) cudaFree(dk); if (dv) cudaFree(dv); if (dout) cudaFree(dout);
        if (drp) cudaFree(drp); if (dci) cudaFree(dci); if (dm) cudaFree(dm);
        return false;
    }
    cudaMemcpy(dq, q.data(), nq * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), nk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), nk * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(drp, row_ptr.data(), row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dci, col_idx.data(), col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dm, token_mask.data(), token_mask.size() * sizeof(unsigned char), cudaMemcpyHostToDevice);
    const int rc = lingbot_flex_attn_f32_masked(dq, dk, dv, drp, dci, dm, dout,
                                                (int) seq_q, (int) seq_k, (int) heads, (int) head_dim,
                                                (int) table.block_size,
                                                1.0f / std::sqrt((float) head_dim), 0);
    cudaMemcpy(out.data(), dout, nq * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(drp); cudaFree(dci); cudaFree(dm);
    if (rc != 0) return false;
    return true;
}

bool run_lingbot_cuda_attention(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const LingBotBlockSparseTable & table,
        const std::vector<unsigned char> & token_mask,
        int64_t seq,
        int64_t heads,
        int64_t head_dim,
        std::vector<float> & out) {
    return run_lingbot_cuda_attention_qk(q, k, v, table, token_mask,
                                         seq, seq, heads, head_dim, out);
}



bool real_qkv_to_cuda_context(
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
        int cache_mode = 0) {
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    if (!hds_tensor_to_shd(qh, seq, m.n_heads, m.head_dim, q) ||
        !hds_tensor_to_shd(kh, seq, m.n_heads, m.head_dim, k) ||
        !hds_tensor_to_shd(vh, seq, m.n_heads, m.head_dim, v)) {
        std::fprintf(stderr, "vla(lingbot_va): failed to read real Q/K/V tensors for CUDA context\n");
        return false;
    }

    if (cache_session_id != 0 && cache_block_index >= 0) {
        std::lock_guard<std::mutex> lock(g_lingbot_runtime_kv_mu);
        const LingBotRuntimeKVKey key{&m, cache_session_id, cache_block_index};
        LingBotKVCache & cache = g_lingbot_runtime_kv[key];
        const int64_t min_capacity = std::max<int64_t>(seq * 8, seq + 128);
        if (cache.batch != 1 || cache.heads != m.n_heads || cache.head_dim != m.head_dim ||
            cache.total_tokens < min_capacity) {
            cache.init(1, min_capacity, m.n_heads, m.head_dim);
        }
        std::vector<int64_t> slots;
        const bool pred = cache_mode == 1;
        if (!cache.update(k, v, seq, pred, slots) || (int64_t) slots.size() != seq) {
            std::fprintf(stderr, "vla(lingbot_va): runtime KV update failed session=%llu block=%lld mode=%d\n",
                         (unsigned long long) cache_session_id,
                         (long long) cache_block_index, cache_mode);
            return false;
        }
        std::vector<float> compact_k;
        std::vector<float> compact_v;
        std::vector<int64_t> valid_slots;
        cache.compact_valid(compact_k, compact_v, valid_slots);
        const int64_t seq_k = (int64_t) valid_slots.size();
        LingBotBlockSparseTable dense_table = build_dense_block_table(seq, seq_k, self_table.block_size);
        std::vector<unsigned char> dense_mask((size_t) seq * (size_t) seq_k, (unsigned char) 1);
        const bool ok = run_lingbot_cuda_attention_qk(q, compact_k, compact_v, dense_table, dense_mask,
                                                      seq, seq_k, m.n_heads, m.head_dim, context_shd);
        if (cache_mode == 0) {
            cache.restore(slots);
        }
        if (cache_out) *cache_out = cache;
        if (ok) {
            std::printf("vla(lingbot_va): runtime KV self-attn session=%llu block=%lld mode=%d q=%lld k=%lld used=%lld pred=%lld\n",
                        (unsigned long long) cache_session_id,
                        (long long) cache_block_index, cache_mode,
                        (long long) seq, (long long) seq_k,
                        (long long) cache.used_count(), (long long) cache.pred_count());
        }
        return ok;
    }

    LingBotKVCache cache;
    cache.init(1, seq, m.n_heads, m.head_dim);
    std::vector<int64_t> slots;
    if (!cache.update(k, v, seq, false, slots) || (int64_t) slots.size() != seq) {
        std::fprintf(stderr, "vla(lingbot_va): KV cache update failed while preparing CUDA context\n");
        return false;
    }
    if (!run_lingbot_cuda_attention(q, cache.k, cache.v, self_table, token_mask,
                                    seq, m.n_heads, m.head_dim, context_shd)) {
        return false;
    }
    if (cache_out) {
        *cache_out = std::move(cache);
    }
    return true;
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

    void step_inplace(std::vector<float> & sample, const std::vector<float> & model_output,
                      double timestep, bool to_final = false) const {
        const int i = timestep_index(timestep);
        const double sigma = sigmas[(size_t) i];
        const double sigma_next = next_sigma(timestep, to_final);
        const float delta = (float) (sigma_next - sigma);
        for (size_t j = 0; j < sample.size(); ++j) {
            sample[j] += model_output[j] * delta;
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
    int64_t cache_block_index = -1;
    std::vector<float> raw_input;
    std::vector<float> time_raw;
    std::vector<float> x;
    std::vector<float> text;
    std::vector<float> t_hidden;
    std::vector<float> timestep_proj;
    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
};

struct LingBotMixedForwardState {
    int64_t latent_seq = 0;
    int64_t action_seq = 0;
    int64_t text_seq = 0;
    int64_t padded_seq = 0;
    int64_t total_seq = 0;
    uint64_t cache_session_id = 0;
    int cache_mode = 0;
    std::vector<float> x;
    std::vector<float> text;
    std::vector<float> t_hidden;
    std::vector<float> timestep_proj;
    std::vector<float> rope_cos;
    std::vector<float> rope_sin;
    LingBotFlexMaskMeta flex_meta;
    LingBotBlockSparseTable self_table;
    std::vector<unsigned char> token_mask;
};

struct LingBotTransformerExecutor {
    const LingBotVAModelArch & m;
    LingBotRuntimeWeights & common;
    std::string dump_dir;
    std::vector<float> text_raw_override;
    int64_t text_raw_seq = 0;
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
        const LingBotGridSpec * grid_spec = nullptr) {
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
    std::vector<float> timh((size_t) time_raw->ne[0] * time_raw->ne[1]);
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
    fill_timestep_embedding(timh, 256, seq, timestep);
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
    build_lingbot_rope(m.head_dim, resolved_grid, out.rope_cos, out.rope_sin);
    ggml_gallocr_free(galloc);
    ggml_free(C);
    return true;
}

void append_hidden_seq(
        std::vector<float> & dst,
        const std::vector<float> & src,
        int64_t hidden,
        int64_t & dst_seq,
        int64_t src_seq) {
    const size_t old_seq = (size_t) dst_seq;
    dst.resize((size_t) hidden * (old_seq + (size_t) src_seq), 0.0f);
    for (int64_t s = 0; s < src_seq; ++s) {
        for (int64_t h = 0; h < hidden; ++h) {
            dst[(size_t) h + (size_t) hidden * (old_seq + (size_t) s)] =
                src[(size_t) h + (size_t) hidden * (size_t) s];
        }
    }
    dst_seq += src_seq;
}

void append_zero_hidden_seq(
        std::vector<float> & dst,
        int64_t hidden,
        int64_t & dst_seq,
        int64_t add_seq) {
    dst.resize((size_t) hidden * (size_t) (dst_seq + add_seq), 0.0f);
    dst_seq += add_seq;
}

bool build_mixed_forward_state(
        LingBotTransformerExecutor & ex,
        const std::vector<float> & latent_tokens,
        const std::vector<float> & action_tokens,
        double latent_timestep,
        double action_timestep,
        const LingBotGridSpec & latent_grid,
        const LingBotGridSpec & action_grid,
        int64_t padded_seq,
        int64_t chunk_size,
        int64_t window_size,
        LingBotMixedForwardState & out) {
    const LingBotVAModelArch & m = ex.m;
    const int64_t hidden = m.cfg.hidden;
    LingBotExecState latent_noisy;
    LingBotExecState latent_clean;
    LingBotExecState action_noisy;
    LingBotExecState action_clean;
    if (!exec_embedding_stage(ex, false, latent_noisy, &latent_tokens, latent_timestep, &latent_grid) ||
        !exec_embedding_stage(ex, false, latent_clean, &latent_tokens, 0.0, &latent_grid) ||
        !exec_embedding_stage(ex, true,  action_noisy, &action_tokens, action_timestep, &action_grid) ||
        !exec_embedding_stage(ex, true,  action_clean, &action_tokens, 0.0, &action_grid)) {
        return false;
    }

    out.latent_seq = latent_noisy.seq;
    out.action_seq = action_noisy.seq;
    out.text_seq = latent_noisy.text_seq;
    out.padded_seq = padded_seq;
    out.total_seq = 0;
    if (latent_clean.seq != out.latent_seq || action_clean.seq != out.action_seq ||
        latent_clean.text_seq != out.text_seq || action_noisy.text_seq != out.text_seq ||
        action_clean.text_seq != out.text_seq ||
        latent_grid.seq() != out.latent_seq || action_grid.seq() != out.action_seq) {
        std::fprintf(stderr, "vla(lingbot_va): mixed forward embedding seq mismatch\n");
        return false;
    }

    append_hidden_seq(out.x, latent_noisy.x, hidden, out.total_seq, out.latent_seq);
    append_hidden_seq(out.x, latent_clean.x, hidden, out.total_seq, out.latent_seq);
    append_hidden_seq(out.x, action_noisy.x, hidden, out.total_seq, out.action_seq);
    append_hidden_seq(out.x, action_clean.x, hidden, out.total_seq, out.action_seq);
    append_zero_hidden_seq(out.x, hidden, out.total_seq, padded_seq);

    int64_t th_seq = 0;
    append_hidden_seq(out.t_hidden, latent_noisy.t_hidden, hidden, th_seq, out.latent_seq);
    append_hidden_seq(out.t_hidden, latent_clean.t_hidden, hidden, th_seq, out.latent_seq);
    append_hidden_seq(out.t_hidden, action_noisy.t_hidden, hidden, th_seq, out.action_seq);
    append_hidden_seq(out.t_hidden, action_clean.t_hidden, hidden, th_seq, out.action_seq);
    append_zero_hidden_seq(out.t_hidden, hidden, th_seq, padded_seq);

    int64_t tp_seq = 0;
    append_hidden_seq(out.timestep_proj, latent_noisy.timestep_proj, hidden * 6, tp_seq, out.latent_seq);
    append_hidden_seq(out.timestep_proj, latent_clean.timestep_proj, hidden * 6, tp_seq, out.latent_seq);
    append_hidden_seq(out.timestep_proj, action_noisy.timestep_proj, hidden * 6, tp_seq, out.action_seq);
    append_hidden_seq(out.timestep_proj, action_clean.timestep_proj, hidden * 6, tp_seq, out.action_seq);
    append_zero_hidden_seq(out.timestep_proj, hidden * 6, tp_seq, padded_seq);

    out.text = latent_noisy.text;
    LingBotTensor5DShape latent_shape{1, m.in_channels,
                                      latent_grid.f * m.patch_t,
                                      latent_grid.h * m.patch_h,
                                      latent_grid.w * m.patch_w};
    LingBotTensor5DShape action_shape{1, m.action_dim, action_grid.f, action_grid.h, action_grid.w};
    out.flex_meta = build_flex_mask_meta(latent_shape, action_shape, padded_seq, chunk_size, window_size,
                                         m.patch_t, m.patch_h, m.patch_w, 512, 4);
    if ((int64_t) out.flex_meta.seq_ids.size() != out.total_seq) {
        std::fprintf(stderr, "vla(lingbot_va): mixed forward flex meta seq mismatch meta=%zu total=%lld\n",
                     out.flex_meta.seq_ids.size(), (long long) out.total_seq);
        return false;
    }
    out.self_table = build_self_block_table(out.flex_meta);
    out.token_mask = build_self_token_mask(out.flex_meta);

    std::vector<LingBotGridId> latent_ids = build_grid_ids_from_spec(latent_grid);
    std::vector<LingBotGridId> action_ids = build_grid_ids_from_spec(action_grid);
    std::vector<LingBotGridId> full;
    full.reserve((size_t) out.total_seq);
    full.insert(full.end(), latent_ids.begin(), latent_ids.end());
    full.insert(full.end(), latent_ids.begin(), latent_ids.end());
    full.insert(full.end(), action_ids.begin(), action_ids.end());
    full.insert(full.end(), action_ids.begin(), action_ids.end());
    for (int64_t i = 0; i < padded_seq; ++i) full.push_back({0.0, 0.0, 0.0});
    build_lingbot_rope_from_grid_ids(m.head_dim, full, out.rope_cos, out.rope_sin);
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
        dump_tensor_2d("cross_attn", trace.cross_attn);
        dump_tensor_2d("post_cross", trace.post_cross);
        dump_tensor_2d("n3", trace.n3);
        dump_tensor_2d("ff", trace.ff);
    }
    ggml_gallocr_free(galloc);
    ggml_free(C);
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
        const std::vector<unsigned char> & token_mask) {
    const LingBotVAModelArch & m = ex.m;
    const int64_t hidden = m.cfg.hidden;
    const int64_t text_seq = state.text_seq > 0 ? state.text_seq : 2;
    if (state.seq != (int64_t) meta.seq_ids.size()) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA self-attn block seq mismatch state=%lld meta=%zu\n",
                     (long long) state.seq, meta.seq_ids.size());
        return false;
    }

    const LingBotBlockW & b = bw.blocks[block_index];
    std::vector<float> self_ctx_hseq;
    {
        ggml_init_params cp = { size_t(96) * 1024 * 1024, nullptr, true };
        ggml_context * C = ggml_init(cp);
        if (!C) return false;

        ggml_tensor * x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
        ggml_tensor * timestep_proj = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden * 6, state.seq);
        ggml_tensor * rope_cos = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, m.head_dim / 2, 1, state.seq);
        ggml_tensor * rope_sin = ggml_new_tensor_4d(C, GGML_TYPE_F32, 1, m.head_dim / 2, 1, state.seq);
        ggml_set_input(x_in);
        ggml_set_input(timestep_proj);
        ggml_set_input(rope_cos);
        ggml_set_input(rope_sin);

        ggml_tensor * shift_msa = ggml_add(C, chunk_hidden(C, timestep_proj, hidden, state.seq, 0),
                                           ggml_view_1d(C, b.scale_shift, hidden, 0));
        ggml_tensor * scale_msa = ggml_add(C, chunk_hidden(C, timestep_proj, hidden, state.seq, 1),
                                           ggml_view_1d(C, b.scale_shift, hidden,
                                                        (size_t) hidden * ggml_element_size(b.scale_shift)));
        ggml_tensor * n1 = adaln(C, x_in, shift_msa, scale_msa, 1e-6f);
        LingBotAttentionTrace trace;
        ggml_tensor * attn_out = build_attention_shape(C, b.self_attn, n1, n1, rope_cos, rope_sin, m,
                                                       state.seq, state.seq, &trace);
        ggml_set_output(trace.qh);
        ggml_set_output(trace.kh);
        ggml_set_output(trace.vh);
        ggml_set_output(attn_out);

        ggml_cgraph * gf = ggml_new_graph_custom(C, 32768, false);
        ggml_build_forward_expand(gf, trace.qh);
        ggml_build_forward_expand(gf, trace.kh);
        ggml_build_forward_expand(gf, trace.vh);

        ggml_gallocr_t galloc = lingbot_runtime_gallocr(bw.backend);
        if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
            if (galloc) ggml_gallocr_free(galloc);
            ggml_free(C);
            return false;
        }
        ggml_backend_tensor_set(x_in, state.x.data(), 0, state.x.size() * sizeof(float));
        ggml_backend_tensor_set(timestep_proj, state.timestep_proj.data(), 0, state.timestep_proj.size() * sizeof(float));
        ggml_backend_tensor_set(rope_cos, state.rope_cos.data(), 0, state.rope_cos.size() * sizeof(float));
        ggml_backend_tensor_set(rope_sin, state.rope_sin.data(), 0, state.rope_sin.size() * sizeof(float));

        const ggml_status st = ggml_backend_graph_compute(bw.backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(lingbot_va): CUDA self-attn QKV graph failed (%d)\n", (int) st);
            ggml_gallocr_free(galloc);
            ggml_free(C);
            return false;
        }

        std::vector<float> context_shd;
        LingBotKVCache cache;
        if (!real_qkv_to_cuda_context(trace.qh, trace.kh, trace.vh, self_table, token_mask,
                                      m, state.seq, context_shd, &cache,
                                      state.cache_session_id,
                                      state.cache_block_index,
                                      state.cache_mode)) {
            ggml_gallocr_free(galloc);
            ggml_free(C);
            return false;
        }
        self_ctx_hseq = shd_to_hidden_seq(context_shd, state.seq, m.n_heads, m.head_dim);
        ggml_gallocr_free(galloc);
        ggml_free(C);
    }

    ggml_init_params cp = { size_t(96) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) return false;

    ggml_tensor * x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
    ggml_tensor * text = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, text_seq);
    ggml_tensor * timestep_proj = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden * 6, state.seq);
    ggml_tensor * self_ctx = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.seq);
    ggml_set_input(x_in);
    ggml_set_input(text);
    ggml_set_input(timestep_proj);
    ggml_set_input(self_ctx);

    ggml_tensor * gate_msa = ggml_add(C, chunk_hidden(C, timestep_proj, hidden, state.seq, 2),
                                      ggml_view_1d(C, b.scale_shift, hidden,
                                                   (size_t) 2 * (size_t) hidden * ggml_element_size(b.scale_shift)));
    ggml_tensor * c_shift_msa = ggml_add(C, chunk_hidden(C, timestep_proj, hidden, state.seq, 3),
                                         ggml_view_1d(C, b.scale_shift, hidden,
                                                      (size_t) 3 * (size_t) hidden * ggml_element_size(b.scale_shift)));
    ggml_tensor * c_scale_msa = ggml_add(C, chunk_hidden(C, timestep_proj, hidden, state.seq, 4),
                                         ggml_view_1d(C, b.scale_shift, hidden,
                                                      (size_t) 4 * (size_t) hidden * ggml_element_size(b.scale_shift)));
    ggml_tensor * c_gate_msa = ggml_add(C, chunk_hidden(C, timestep_proj, hidden, state.seq, 5),
                                        ggml_view_1d(C, b.scale_shift, hidden,
                                                     (size_t) 5 * (size_t) hidden * ggml_element_size(b.scale_shift)));

    ggml_tensor * a1 = lin(C, b.self_attn.o, self_ctx);
    ggml_tensor * x = ggml_add(C, x_in, ggml_mul(C, a1, gate_msa));
    ggml_tensor * n2 = ggml_add(C, ggml_mul(C, ggml_norm(C, x, 1e-6f), b.cross_norm_weight), b.cross_norm_bias);
    ggml_tensor * a2 = build_attention_shape(C, b.cross_attn, n2, text, nullptr, nullptr, m, state.seq, text_seq);
    x = ggml_add(C, x, a2);
    ggml_tensor * n3 = adaln(C, x, c_shift_msa, c_scale_msa, 1e-6f);
    ggml_tensor * ff = lin(C, b.ffn_down, ggml_gelu(C, lin(C, b.ffn_up, n3)));
    ggml_tensor * out = ggml_add(C, x, ggml_mul(C, ff, c_gate_msa));
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 32768, false);
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
    ggml_backend_tensor_set(self_ctx, self_ctx_hseq.data(), 0, self_ctx_hseq.size() * sizeof(float));

    const ggml_status st = ggml_backend_graph_compute(bw.backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(lingbot_va): CUDA self-attn post graph failed (%d)\n", (int) st);
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }
    state.x.resize((size_t) out->ne[0] * out->ne[1]);
    ggml_backend_tensor_get(out, state.x.data(), 0, state.x.size() * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(C);
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
    ggml_backend_tensor_set(x_in, state.x.data(), 0, state.x.size() * sizeof(float));
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
}

bool exec_mixed_output_heads(
        LingBotTransformerExecutor & ex,
        const LingBotMixedForwardState & state,
        std::vector<float> & latent_out,
        std::vector<float> & action_out) {
    const LingBotVAModelArch & m = ex.m;
    LingBotRuntimeWeights & sw = ex.common;
    const int64_t hidden = m.cfg.hidden;
    const int64_t latent_offset = 0;
    const int64_t action_offset = 2 * state.latent_seq;
    ggml_init_params cp = { size_t(32) * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) return false;

    ggml_tensor * x_in = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.total_seq);
    ggml_tensor * t_hidden = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, state.total_seq);
    ggml_set_input(x_in);
    ggml_set_input(t_hidden);

    auto build_head = [&](bool action_mode, int64_t offset_seq, int64_t seq) -> ggml_tensor * {
        ggml_tensor * x_view = ggml_view_2d(C, x_in, hidden, seq, x_in->nb[1],
                                            (size_t) offset_seq * (size_t) hidden * ggml_element_size(x_in));
        ggml_tensor * t_view = ggml_view_2d(C, t_hidden, hidden, seq, t_hidden->nb[1],
                                            (size_t) offset_seq * (size_t) hidden * ggml_element_size(t_hidden));
        ggml_tensor * out_shift = ggml_add(C, t_view, ggml_view_1d(C, sw.output_scale_shift, hidden, 0));
        ggml_tensor * out_scale = ggml_add(C, t_view, ggml_view_1d(C, sw.output_scale_shift, hidden,
                                                                   (size_t) hidden * ggml_element_size(sw.output_scale_shift)));
        return action_mode ? lin(C, sw.action_out, adaln(C, x_view, out_shift, out_scale, 1e-6f))
                           : lin(C, sw.output_proj, adaln(C, x_view, out_shift, out_scale, 1e-6f));
    };

    ggml_tensor * latent = build_head(false, latent_offset, state.latent_seq);
    ggml_tensor * action = build_head(true,  action_offset, state.action_seq);
    ggml_set_output(latent);
    ggml_set_output(action);
    ggml_cgraph * gf = ggml_new_graph_custom(C, 16384, false);
    ggml_build_forward_expand(gf, latent);
    ggml_build_forward_expand(gf, action);

    ggml_gallocr_t galloc = lingbot_runtime_gallocr(sw.backend);
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }
    ggml_backend_tensor_set(x_in, state.x.data(), 0, state.x.size() * sizeof(float));
    ggml_backend_tensor_set(t_hidden, state.t_hidden.data(), 0, state.t_hidden.size() * sizeof(float));
    const ggml_status st = ggml_backend_graph_compute(sw.backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(lingbot_va): mixed output heads graph failed (%d)\n", (int) st);
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return false;
    }
    latent_out.resize((size_t) latent->ne[0] * latent->ne[1]);
    action_out.resize((size_t) action->ne[0] * action->ne[1]);
    ggml_backend_tensor_get(latent, latent_out.data(), 0, latent_out.size() * sizeof(float));
    ggml_backend_tensor_get(action, action_out.data(), 0, action_out.size() * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(C);
    return true;
}

struct LingBotWanMixedForwardResult {
    std::vector<float> latent_tokens;
    std::vector<float> action_tokens;
    std::vector<float> latent_tensor;
    std::vector<float> action_tensor;
    LingBotTensor5DShape latent_tensor_shape;
    LingBotTensor5DShape action_tensor_shape;
    int64_t latent_feature = 0;
    int64_t latent_seq = 0;
    int64_t action_feature = 0;
    int64_t action_seq = 0;
    int64_t total_seq = 0;
    int64_t padded_seq = 0;
    int64_t window_size = 1;
};

bool exec_wan_mixed_forward_tensors(
        gguf_reader & g,
        LingBotTransformerExecutor & ex,
        const std::vector<float> & latent_tensor,
        const LingBotTensor5DShape & latent_shape,
        const std::vector<float> & action_tensor,
        const LingBotTensor5DShape & action_shape,
        double latent_timestep,
        double action_timestep,
        int blocks,
        bool use_cuda_self_attn,
        int64_t chunk_size,
        int64_t window_size,
        uint64_t cache_session_id,
        int cache_mode,
        LingBotWanMixedForwardResult & result) {
    const LingBotVAModelArch & m = ex.m;
    std::vector<float> latent_tokens;
    std::vector<float> action_tokens;
    int64_t latent_feature = 0, latent_seq = 0;
    int64_t action_feature = 0, action_seq = 0;
    if (!patchify_latent_tokens(latent_tensor, latent_shape, m.patch_t, m.patch_h, m.patch_w,
                                latent_tokens, latent_feature, latent_seq) ||
        !action_tensor_to_tokens(action_tensor, action_shape, action_tokens, action_feature, action_seq)) {
        std::fprintf(stderr, "vla(lingbot_va): Wan mixed forward tokenization failed\n");
        return false;
    }

    LingBotGridSpec latent_grid;
    latent_grid.f = latent_shape.f / m.patch_t;
    latent_grid.h = latent_shape.h / m.patch_h;
    latent_grid.w = latent_shape.w / m.patch_w;
    latent_grid.t = 0;
    latent_grid.action = false;
    LingBotGridSpec action_grid;
    action_grid.f = action_shape.f;
    action_grid.h = action_shape.h;
    action_grid.w = action_shape.w;
    action_grid.t = 1;
    action_grid.action = true;

    const int64_t total_no_pad = 2 * latent_seq + 2 * action_seq;
    const int64_t padded_seq = (128 - total_no_pad % 128) % 128;
    LingBotMixedForwardState state;
    if (!build_mixed_forward_state(ex, latent_tokens, action_tokens,
                                   latent_timestep, action_timestep,
                                   latent_grid, action_grid,
                                   padded_seq, chunk_size, 4, state)) {
        return false;
    }
    state.cache_session_id = cache_session_id;
    state.cache_mode = cache_mode;

    const bool use_resident_block_cache =
        std::getenv("VLA_LINGBOT_RESIDENT_BLOCK_CACHE") != nullptr &&
        std::getenv("VLA_LINGBOT_RESIDENT_BLOCK_CACHE_DISABLE") == nullptr;
    if (use_resident_block_cache) {
        for (int64_t global_block = 0; global_block < blocks; ++global_block) {
            LingBotRuntimeWeights * bw = get_resident_block_weights(g, m.ckpt_path, global_block);
            if (!bw) return false;
            LingBotExecState block_state;
            block_state.seq = state.total_seq;
            block_state.text_seq = state.text_seq;
            block_state.x = state.x;
            block_state.text = state.text;
            block_state.t_hidden = state.t_hidden;
            block_state.timestep_proj = state.timestep_proj;
            block_state.rope_cos = state.rope_cos;
            block_state.rope_sin = state.rope_sin;
            block_state.cache_session_id = state.cache_session_id;
            block_state.cache_mode = state.cache_mode;
            block_state.cache_block_index = global_block;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            const bool ok = use_cuda_self_attn
                ? exec_block_one_cuda_self_attn(ex, *bw, 0, block_state, state.flex_meta, state.self_table, state.token_mask)
                : exec_block_one(ex, *bw, 0, block_state);
#else
            if (use_cuda_self_attn) {
                std::fprintf(stderr, "vla(lingbot_va): CUDA mixed self-attn requested but CUDA kernels are not built\n");
                return false;
            }
            const bool ok = exec_block_one(ex, *bw, 0, block_state);
#endif
            if (!ok) return false;
            state.x = std::move(block_state.x);
            std::printf("vla(lingbot_va): Wan mixed forward block %lld/%d complete (resident %s self-attn)\n",
                        (long long) global_block + 1, blocks,
                        use_cuda_self_attn ? "CUDA flex" : "ggml dense");
        }
    } else {
    for (int64_t start = 0; start < blocks; start += window_size) {
        const int64_t count = std::min<int64_t>(window_size, blocks - start);
        LingBotRuntimeWeights bw;
        ggml_init_params wp = { size_t(8) * 1024 * 1024, nullptr, true };
        bw.ctx = ggml_init(wp);
        if (!bw.ctx) return false;
        bw.blocks.resize((size_t) count);
        for (int64_t j = 0; j < count; ++j) {
            if (!runtime_block(g, bw, start + j, bw.blocks[(size_t) j])) return false;
        }
        const std::string label = count == 1
            ? "wan mixed forward block " + std::to_string(start)
            : "wan mixed forward blocks " + std::to_string(start) + "-" + std::to_string(start + count - 1);
        if (!allocate_and_load_runtime_weights(g, bw, label.c_str())) return false;

        for (int64_t j = 0; j < count; ++j) {
            const int64_t global_block = start + j;
            LingBotExecState block_state;
            block_state.seq = state.total_seq;
            block_state.text_seq = state.text_seq;
            block_state.x = state.x;
            block_state.text = state.text;
            block_state.t_hidden = state.t_hidden;
            block_state.timestep_proj = state.timestep_proj;
            block_state.rope_cos = state.rope_cos;
            block_state.rope_sin = state.rope_sin;
            block_state.cache_session_id = state.cache_session_id;
            block_state.cache_mode = state.cache_mode;
            block_state.cache_block_index = global_block;
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
            const bool ok = use_cuda_self_attn
                ? exec_block_one_cuda_self_attn(ex, bw, (size_t) j, block_state, state.flex_meta, state.self_table, state.token_mask)
                : exec_block_one(ex, bw, (size_t) j, block_state);
#else
            if (use_cuda_self_attn) {
                std::fprintf(stderr, "vla(lingbot_va): CUDA mixed self-attn requested but CUDA kernels are not built\n");
                return false;
            }
            const bool ok = exec_block_one(ex, bw, (size_t) j, block_state);
#endif
            if (!ok) return false;
            state.x = std::move(block_state.x);
            std::printf("vla(lingbot_va): Wan mixed forward block %lld/%d complete (%s self-attn, window=%lld)\n",
                        (long long) global_block + 1, blocks,
                        use_cuda_self_attn ? "CUDA flex" : "ggml dense",
                        (long long) count);
        }
    }
    }

    std::vector<float> latent_out;
    std::vector<float> action_out;
    if (!exec_mixed_output_heads(ex, state, latent_out, action_out)) return false;
    const LingBotTensor5DShape latent_out_shape{1, m.out_channels, latent_shape.f, latent_shape.h, latent_shape.w};
    std::vector<float> latent_out_tensor;
    std::vector<float> action_out_tensor;
    if (!projected_latent_tokens_to_tensor(latent_out, latent_out_shape, m.patch_t, m.patch_h, m.patch_w, latent_out_tensor)) {
        std::fprintf(stderr, "vla(lingbot_va): Wan mixed forward latent output unpatch failed\n");
        return false;
    }
    if (!action_tokens_to_tensor(action_out, action_shape, action_out_tensor)) {
        std::fprintf(stderr, "vla(lingbot_va): Wan mixed forward action output detokenize failed\n");
        return false;
    }

    result.latent_tokens = std::move(latent_out);
    result.action_tokens = std::move(action_out);
    result.latent_tensor = std::move(latent_out_tensor);
    result.action_tensor = std::move(action_out_tensor);
    result.latent_tensor_shape = latent_out_shape;
    result.action_tensor_shape = action_shape;
    result.latent_feature = latent_feature;
    result.latent_seq = latent_seq;
    result.action_feature = action_feature;
    result.action_seq = action_seq;
    result.total_seq = state.total_seq;
    result.padded_seq = state.padded_seq;
    result.window_size = window_size;
    return true;
}

bool exec_wan_separate_forward_tensors(
        gguf_reader & g,
        LingBotTransformerExecutor & ex,
        const std::vector<float> & latent_tensor,
        const LingBotTensor5DShape & latent_shape,
        const std::vector<float> & action_tensor,
        const LingBotTensor5DShape & action_shape,
        double latent_timestep,
        double action_timestep,
        int blocks,
        int64_t window_size,
        LingBotWanMixedForwardResult & result) {
    const LingBotVAModelArch & m = ex.m;
    std::vector<float> latent_tokens;
    std::vector<float> action_tokens;
    int64_t latent_feature = 0, latent_seq = 0;
    int64_t action_feature = 0, action_seq = 0;
    if (!patchify_latent_tokens(latent_tensor, latent_shape, m.patch_t, m.patch_h, m.patch_w,
                                latent_tokens, latent_feature, latent_seq) ||
        !action_tensor_to_tokens(action_tensor, action_shape, action_tokens, action_feature, action_seq)) {
        std::fprintf(stderr, "vla(lingbot_va): Wan separate forward tokenization failed\n");
        return false;
    }

    LingBotGridSpec latent_grid;
    latent_grid.f = latent_shape.f / m.patch_t;
    latent_grid.h = latent_shape.h / m.patch_h;
    latent_grid.w = latent_shape.w / m.patch_w;
    latent_grid.t = 0;
    latent_grid.action = false;
    LingBotGridSpec action_grid;
    action_grid.f = action_shape.f;
    action_grid.h = action_shape.h;
    action_grid.w = action_shape.w;
    action_grid.t = 1;
    action_grid.action = true;

    LingBotExecState latent_state;
    LingBotExecState action_state;
    if (!exec_embedding_stage(ex, false, latent_state, &latent_tokens, latent_timestep, &latent_grid) ||
        !exec_embedding_stage(ex, true,  action_state, &action_tokens, action_timestep, &action_grid)) {
        return false;
    }

    if (const char * dump_dir_c = std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR")) {
        const std::string dump_dir(dump_dir_c);
        std::string tag;
        if (const char * tag_c = std::getenv("VLA_LINGBOT_PREDICT_TRACE_TAG")) {
            tag = tag_c;
        }
        auto trace_path = [&](const char * stem, const char * ext) {
            return dump_dir + "/" + stem + (tag.empty() ? std::string() : "_" + tag) + ext;
        };
        dump_f32_file(trace_path("lingbot_predict_trace_latent_raw_tokens", ".f32"), latent_tokens);
        dump_text_file(trace_path("lingbot_predict_trace_latent_raw_tokens", ".shape.txt"),
                       std::to_string((long long) latent_seq) + " " +
                       std::to_string((long long) latent_feature) + "\n");
        dump_f32_file(trace_path("lingbot_predict_trace_action_raw_tokens", ".f32"), action_tokens);
        dump_text_file(trace_path("lingbot_predict_trace_action_raw_tokens", ".shape.txt"),
                       std::to_string((long long) action_seq) + " " +
                       std::to_string((long long) action_feature) + "\n");
        dump_f32_file(trace_path("lingbot_predict_trace_latent_time_raw", ".f32"), latent_state.time_raw);
        dump_text_file(trace_path("lingbot_predict_trace_latent_time_raw", ".shape.txt"),
                       std::to_string((long long) latent_state.seq) + " 256\n");
        dump_f32_file(trace_path("lingbot_predict_trace_action_time_raw", ".f32"), action_state.time_raw);
        dump_text_file(trace_path("lingbot_predict_trace_action_time_raw", ".shape.txt"),
                       std::to_string((long long) action_state.seq) + " 256\n");
        dump_f32_file(trace_path("lingbot_predict_trace_latent_x_emb", ".f32"), latent_state.x);
        dump_text_file(trace_path("lingbot_predict_trace_latent_x_emb", ".shape.txt"),
                       std::to_string((long long) latent_state.seq) + " " +
                       std::to_string((long long) m.cfg.hidden) + "\n");
        dump_f32_file(trace_path("lingbot_predict_trace_action_x_emb", ".f32"), action_state.x);
        dump_text_file(trace_path("lingbot_predict_trace_action_x_emb", ".shape.txt"),
                       std::to_string((long long) action_state.seq) + " " +
                       std::to_string((long long) m.cfg.hidden) + "\n");
        dump_f32_file(trace_path("lingbot_predict_trace_latent_t_hidden", ".f32"), latent_state.t_hidden);
        dump_text_file(trace_path("lingbot_predict_trace_latent_t_hidden", ".shape.txt"),
                       std::to_string((long long) latent_state.seq) + " " +
                       std::to_string((long long) m.cfg.hidden) + "\n");
        dump_f32_file(trace_path("lingbot_predict_trace_action_t_hidden", ".f32"), action_state.t_hidden);
        dump_text_file(trace_path("lingbot_predict_trace_action_t_hidden", ".shape.txt"),
                       std::to_string((long long) action_state.seq) + " " +
                       std::to_string((long long) m.cfg.hidden) + "\n");
        dump_f32_file(trace_path("lingbot_predict_trace_latent_timestep_proj", ".f32"), latent_state.timestep_proj);
        dump_text_file(trace_path("lingbot_predict_trace_latent_timestep_proj", ".shape.txt"),
                       std::to_string((long long) latent_state.seq) + " " +
                       std::to_string((long long) m.cfg.hidden * 6) + "\n");
        dump_f32_file(trace_path("lingbot_predict_trace_action_timestep_proj", ".f32"), action_state.timestep_proj);
        dump_text_file(trace_path("lingbot_predict_trace_action_timestep_proj", ".shape.txt"),
                       std::to_string((long long) action_state.seq) + " " +
                       std::to_string((long long) m.cfg.hidden * 6) + "\n");
    }

    const bool use_resident_block_cache =
        std::getenv("VLA_LINGBOT_RESIDENT_BLOCK_CACHE") != nullptr &&
        std::getenv("VLA_LINGBOT_RESIDENT_BLOCK_CACHE_DISABLE") == nullptr;
    if (use_resident_block_cache) {
        for (int64_t global_block = 0; global_block < blocks; ++global_block) {
            LingBotRuntimeWeights * bw = get_resident_block_weights(g, m.ckpt_path, global_block);
            if (!bw) return false;
            if (!exec_block_one(ex, *bw, 0, latent_state, "latent") ||
                !exec_block_one(ex, *bw, 0, action_state, "action")) {
                return false;
            }
            std::printf("vla(lingbot_va): Wan separate forward block %lld/%d complete (resident ggml dense)\n",
                        (long long) global_block + 1, blocks);
        }
    } else {
        for (int64_t start = 0; start < blocks; start += window_size) {
            const int64_t count = std::min<int64_t>(window_size, blocks - start);
            LingBotRuntimeWeights bw;
            ggml_init_params wp = { size_t(8) * 1024 * 1024, nullptr, true };
            bw.ctx = ggml_init(wp);
            if (!bw.ctx) return false;
            bw.blocks.resize((size_t) count);
            for (int64_t j = 0; j < count; ++j) {
                if (!runtime_block(g, bw, start + j, bw.blocks[(size_t) j])) return false;
            }
            const std::string label = count == 1
                ? "wan separate forward block " + std::to_string(start)
                : "wan separate forward blocks " + std::to_string(start) + "-" + std::to_string(start + count - 1);
            if (!allocate_and_load_runtime_weights(g, bw, label.c_str())) return false;
            for (int64_t j = 0; j < count; ++j) {
                if (!exec_block_one(ex, bw, (size_t) j, latent_state, "latent") ||
                    !exec_block_one(ex, bw, (size_t) j, action_state, "action")) {
                    return false;
                }
                std::printf("vla(lingbot_va): Wan separate forward block %lld/%d complete (ggml dense, window=%lld)\n",
                            (long long) start + j + 1, blocks, (long long) count);
            }
        }
    }

    if (const char * dump_dir_c = std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR")) {
        const std::string dump_dir(dump_dir_c);
        std::string tag;
        if (const char * tag_c = std::getenv("VLA_LINGBOT_PREDICT_TRACE_TAG")) {
            tag = tag_c;
        }
        auto trace_path = [&](const char * stem, const char * ext) {
            return dump_dir + "/" + stem + (tag.empty() ? std::string() : "_" + tag) + ext;
        };
        dump_f32_file(trace_path("lingbot_predict_trace_latent_block_out", ".f32"), latent_state.x);
        dump_text_file(trace_path("lingbot_predict_trace_latent_block_out", ".shape.txt"),
                       std::to_string((long long) latent_state.seq) + " " +
                       std::to_string((long long) m.cfg.hidden) + "\n");
        dump_f32_file(trace_path("lingbot_predict_trace_action_block_out", ".f32"), action_state.x);
        dump_text_file(trace_path("lingbot_predict_trace_action_block_out", ".shape.txt"),
                       std::to_string((long long) action_state.seq) + " " +
                       std::to_string((long long) m.cfg.hidden) + "\n");
    }

    std::vector<float> latent_out;
    std::vector<float> action_out;
    if (!exec_output_one(ex, false, latent_state, latent_out) ||
        !exec_output_one(ex, true,  action_state, action_out)) {
        return false;
    }
    const LingBotTensor5DShape latent_out_shape{1, m.out_channels, latent_shape.f, latent_shape.h, latent_shape.w};
    std::vector<float> latent_out_tensor;
    std::vector<float> action_out_tensor;
    if (!projected_latent_tokens_to_tensor(latent_out, latent_out_shape, m.patch_t, m.patch_h, m.patch_w, latent_out_tensor)) {
        std::fprintf(stderr, "vla(lingbot_va): Wan separate forward latent output unpatch failed\n");
        return false;
    }
    if (!action_tokens_to_tensor(action_out, action_shape, action_out_tensor)) {
        std::fprintf(stderr, "vla(lingbot_va): Wan separate forward action output detokenize failed\n");
        return false;
    }
    result.latent_tokens = std::move(latent_out);
    result.action_tokens = std::move(action_out);
    result.latent_tensor = std::move(latent_out_tensor);
    result.action_tensor = std::move(action_out_tensor);
    result.latent_tensor_shape = latent_out_shape;
    result.action_tensor_shape = action_shape;
    result.latent_feature = latent_feature;
    result.latent_seq = latent_seq;
    result.action_feature = action_feature;
    result.action_seq = action_seq;
    result.total_seq = latent_state.seq + action_state.seq;
    result.padded_seq = 0;
    result.window_size = window_size;
    return true;
}

bool exec_forward_one_streaming(
        LingBotTransformerExecutor & ex,
        LingBotRuntimeWeights & bw,
        bool action_mode,
        const std::vector<float> * raw_input,
        double timestep,
        int blocks,
        std::vector<float> & out_h,
        const LingBotGridSpec * grid_spec = nullptr) {
    LingBotExecState state;
    if (!exec_embedding_stage(ex, action_mode, state, raw_input, timestep, grid_spec)) {
        return false;
    }
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
    const bool use_cuda_self_attn = std::getenv("VLA_LINGBOT_CUDA_SELF_ATTN_STREAM") != nullptr;
    LingBotFlexMaskMeta cuda_meta;
    LingBotBlockSparseTable cuda_self_table;
    std::vector<unsigned char> cuda_token_mask;
    if (use_cuda_self_attn) {
        cuda_meta = build_runtime_flex_meta(ex.m);
        if (state.seq != (int64_t) cuda_meta.seq_ids.size()) {
            std::fprintf(stderr,
                         "vla(lingbot_va): VLA_LINGBOT_CUDA_SELF_ATTN_STREAM requires seq=%zu, got seq=%lld. "
                         "For current runtime use VLA_LINGBOT_FLOW_LOOP_LATENT_SEQ=27 or ACTION_SEQ=27.\n",
                         cuda_meta.seq_ids.size(), (long long) state.seq);
            return false;
        }
        cuda_self_table = build_self_block_table(cuda_meta);
        cuda_token_mask = build_self_token_mask(cuda_meta);
        if (!build_runtime_mixed_rope(ex.m, cuda_meta, state.rope_cos, state.rope_sin)) {
            return false;
        }
        std::printf("vla(lingbot_va): CUDA self-attn streaming uses mixed LingBot RoPE order "
                    "[latent, latent, action, action, pad], seq=%lld\n",
                    (long long) state.seq);
    }
#else
    const bool use_cuda_self_attn = false;
    if (std::getenv("VLA_LINGBOT_CUDA_SELF_ATTN_STREAM")) {
        std::fprintf(stderr, "vla(lingbot_va): VLA_LINGBOT_CUDA_SELF_ATTN_STREAM requested but LingBot CUDA kernels were not built\n");
        return false;
    }
#endif
    for (int b = 0; b < blocks; ++b) {
#ifdef VLA_LINGBOT_FLEX_CUDA_KERNELS
        const bool ok = use_cuda_self_attn
            ? exec_block_one_cuda_self_attn(ex, bw, (size_t) b, state, cuda_meta, cuda_self_table, cuda_token_mask)
            : exec_block_one(ex, bw, (size_t) b, state);
#else
        const bool ok = exec_block_one(ex, bw, (size_t) b, state);
#endif
        if (!ok) {
            return false;
        }
    }
    return exec_output_one(ex, action_mode, state, out_h);
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
        if (src_c >= k_used_dim) continue;
        const float denom = k_q99[src_c] - k_q01[src_c] + 1.0e-6f;
        for (int64_t f = 0; f < copy_f; ++f) {
            for (int64_t h = 0; h < copy_h; ++h) {
                const size_t src = (size_t) src_c * (size_t) f_in * (size_t) h_in +
                                   (size_t) f * (size_t) h_in + (size_t) h;
                const float norm = (action_history[src] - k_q01[src_c]) / denom * 2.0f - 1.0f;
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
    const auto t_total0 = std::chrono::steady_clock::now();
    stats = {};

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
    const int video_steps = env_i32("VLA_LINGBOT_PREDICT_VIDEO_STEPS", 20);
    const int action_steps = env_i32("VLA_LINGBOT_PREDICT_ACTION_STEPS", 50);
    const float video_guidance_scale = env_f32("VLA_LINGBOT_VIDEO_GUIDANCE_SCALE", 1.0f);
    const bool use_video_cfg = video_guidance_scale != 1.0f;
    const bool cache_prefill_update = in.lingbot_cache_mode == 2;
    const bool use_cuda_self_attn =
        std::getenv("VLA_LINGBOT_PREDICT_CUDA_SELF_ATTN") != nullptr ||
        std::getenv("VLA_LINGBOT_MIXED_CUDA_SELF_ATTN") != nullptr;
    if (in.lingbot_clear_pred_cache) {
        runtime_kv_clear_pred_for_session(this, in.lingbot_session_id);
    }

    if (cfg.n_suffix <= 0 || cfg.max_action_dim <= 0) {
        std::fprintf(stderr, "vla(lingbot_va): invalid action output shape suffix=%lld action_dim=%lld\n",
                     (long long) cfg.n_suffix, (long long) cfg.max_action_dim);
        return {};
    }

    gguf_reader g;
    if (!g.open(ckpt_path)) return {};
    static std::unique_ptr<LingBotRuntimeWeights> s_common_cache;
    static std::string s_common_cache_key;
    std::unique_ptr<LingBotRuntimeWeights> local_common;
    LingBotRuntimeWeights * common_ptr = nullptr;
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
    LingBotTransformerExecutor ex{*this, *common_ptr, std::string()};
    LingBotTransformerExecutor ex_uncond{*this, *common_ptr, std::string()};
    const bool use_text_encoder = std::getenv("VLA_LINGBOT_PREDICT_TEXT_ENCODER") != nullptr;
    if (use_text_encoder) {
        if (!in.lang_tokens || in.n_lang <= 0) {
            std::fprintf(stderr, "vla(lingbot_va): VLA_LINGBOT_PREDICT_TEXT_ENCODER requires lang_tokens\n");
            return {};
        }
        if (in.n_lang > (int) cfg.n_lang) {
            std::fprintf(stderr, "vla(lingbot_va): lang_tokens length %d exceeds cfg.n_lang=%lld\n",
                         in.n_lang, (long long) cfg.n_lang);
            return {};
        }
        const char * text_path_c = std::getenv("VLA_LINGBOT_TEXT_GGUF");
        if (!text_path_c || std::strlen(text_path_c) == 0) {
            std::fprintf(stderr, "vla(lingbot_va): VLA_LINGBOT_PREDICT_TEXT_ENCODER requires VLA_LINGBOT_TEXT_GGUF\n");
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
        const std::string text_path(text_path_c);
        const bool cache_disabled = std::getenv("VLA_LINGBOT_TEXT_CACHE_DISABLE") != nullptr;
        auto encode_text_for_executor = [&](const std::vector<int32_t> & ids,
                                            LingBotTransformerExecutor & target,
                                            const char * label) -> bool {
            int64_t text_seq = 0;
            int64_t text_dim = 0;
            if (!cache_disabled &&
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
                if (!cache_disabled) {
                    text_cache_store(text_path, text_blocks, ids, target.text_raw_override, text_seq, text_dim);
                    std::printf("vla(lingbot_va): UMT5 text cache store (%s): tokens=%zu blocks=%d bytes=%.2f MiB\n",
                                label, ids.size(), text_blocks,
                                target.text_raw_override.size() * sizeof(float) / (1024.0 * 1024.0));
                }
            }
            if (text_dim != this->text_dim) {
                std::fprintf(stderr, "vla(lingbot_va): UMT5 output dim %lld != transformer text_dim=%lld\n",
                             (long long) text_dim, (long long) this->text_dim);
                return false;
            }
            target.text_raw_seq = text_seq;
            return true;
        };

        std::vector<int32_t> ids(in.lang_tokens, in.lang_tokens + in.n_lang);
        if (!encode_text_for_executor(ids, ex, "cond")) return {};
        if (use_video_cfg) {
            std::vector<int32_t> negative_ids = env_i32_list("VLA_LINGBOT_NEGATIVE_TOKEN_IDS", {1});
            if (!encode_text_for_executor(negative_ids, ex_uncond, "uncond")) return {};
        }
    } else if (use_video_cfg) {
        std::fprintf(stderr,
                     "vla(lingbot_va): VLA_LINGBOT_VIDEO_GUIDANCE_SCALE requires "
                     "VLA_LINGBOT_PREDICT_TEXT_ENCODER=1 for the unconditional text branch\n");
        return {};
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
        const char * vae_path_c = std::getenv("VLA_LINGBOT_VAE_GGUF");
        if (!vae_path_c || std::strlen(vae_path_c) == 0) {
            std::fprintf(stderr,
                         "vla(lingbot_va): LingBot video input requires VLA_LINGBOT_VAE_GGUF\n");
            return {};
        }
        if (!encode_lingbot_video_to_latent(vae_path_c,
                                            in.lingbot_video,
                                            in.lingbot_video_views,
                                            in.lingbot_video_c,
                                            in.lingbot_video_f,
                                            in.lingbot_video_h,
                                            in.lingbot_video_w,
                                            encoded_video_latent,
                                            encoded_video_shape)) {
            return {};
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
    }

    const int64_t frame_chunk_size = env_i32("VLA_LINGBOT_FRAME_CHUNK_SIZE", 4);
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
    const int64_t action_per_frame = 4;
    if (cfg.n_suffix % action_per_frame != 0) {
        std::fprintf(stderr,
                     "vla(lingbot_va): n_suffix=%lld is not divisible by action_per_frame=%lld\n",
                     (long long) cfg.n_suffix, (long long) action_per_frame);
        return {};
    }
    const LingBotTensor5DShape action_shape{1, action_dim, cfg.n_suffix / action_per_frame, action_per_frame, 1};
    std::vector<float> latent_condition;
    LingBotTensor5DShape latent_condition_shape{};
    bool has_latent_condition = false;
    if (!in.lingbot_latent && condition_current_frame && !encoded_video_latent.empty() &&
        encoded_video_shape.b == 1 && encoded_video_shape.c == in_channels &&
        encoded_video_shape.f > 0 && encoded_video_shape.f <= latent_shape.f &&
        encoded_video_shape.h == latent_shape.h && encoded_video_shape.w == latent_shape.w) {
        latent_condition = encoded_video_latent;
        latent_condition_shape = encoded_video_shape;
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
    if (in.lingbot_latent) {
        std::memcpy(latent_sample.data(), in.lingbot_latent, latent_sample.size() * sizeof(float));
        std::printf("vla(lingbot_va): using caller LingBot latent shape=[%lld,%lld,%lld,%lld,%lld] checksum=%.9g\n",
                    (long long) latent_shape.b, (long long) latent_shape.c,
                    (long long) latent_shape.f, (long long) latent_shape.h,
                    (long long) latent_shape.w, checksum(latent_sample));
    } else {
        fill_deterministic(latent_sample, 0.02f);
    }
    if (has_latent_condition) {
        if (latent_sample.empty()) fill_deterministic(latent_sample, 0.02f);
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
        for (int64_t t = 0; t < cfg.n_suffix; ++t) {
            const int64_t f = t / action_per_frame;
            const int64_t h = t % action_per_frame;
            for (int64_t c = 0; c < action_dim; ++c) {
                action_sample[idx5(action_shape, 0, c, f, h, 0)] =
                    in.noise[(size_t) t * (size_t) action_dim + (size_t) c];
            }
        }
    } else {
        fill_deterministic(action_sample, 0.03f);
    }
    zero_unused_libero_action_channels(action_sample, action_shape);
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
    } else if (in.state != nullptr && std::getenv("VLA_LINGBOT_STATE_COND") != nullptr) {
        if (!preprocess_libero_state_to_action_condition(in.state, cfg.real_state_dim,
                                                         action_shape, action_condition)) {
            std::fprintf(stderr, "vla(lingbot_va): failed to preprocess state into action condition\n");
            return {};
        }
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
            zero_unused_libero_action_channels(action_sample, action_shape);
            std::printf("vla(lingbot_va): cache update uses executed action condition as input "
                        "shape=[%lld,%lld,%lld,%lld,%lld] checksum=%.9g max=%.9g\n",
                        (long long) action_shape.b, (long long) action_shape.c,
                        (long long) action_shape.f, (long long) action_shape.h,
                        (long long) action_shape.w, checksum(action_sample),
                        max_abs_value(action_sample));
        } else {
            std::printf("vla(lingbot_va): cache update has no action condition; using fallback action sample\n");
        }

        const auto t_inf0 = std::chrono::steady_clock::now();
        LingBotWanMixedForwardResult cache_result;
        if (!exec_wan_mixed_forward_tensors(g, ex,
                                            latent_sample, latent_shape,
                                            action_sample, action_shape,
                                            0.0, 0.0,
                                            blocks, use_cuda_self_attn,
                                            1, window_size,
                                            in.lingbot_session_id,
                                            2,
                                            cache_result)) {
            return {};
        }
        const auto t_inf1 = std::chrono::steady_clock::now();
        const auto t_total1 = std::chrono::steady_clock::now();
        stats.ms_vision = 0.0f;
        stats.ms_prefill = 0.0f;
        stats.ms_denoise = 0.0f;
        stats.ms_inference = std::chrono::duration<float, std::milli>(t_inf1 - t_inf0).count();
        stats.ms_total = std::chrono::duration<float, std::milli>(t_total1 - t_total0).count();
        std::printf("vla(lingbot_va): kv cache update bridge ok blocks=%d mode=%s window=%lld "
                    "latent=[%lld,%lld,%lld,%lld,%lld] action=[%lld,%lld,%lld,%lld,%lld] "
                    "inf=%.1fms\n",
                    blocks, use_cuda_self_attn ? "cuda-flex-self-attn" : "ggml-dense",
                    (long long) window_size,
                    (long long) latent_shape.b, (long long) latent_shape.c,
                    (long long) latent_shape.f, (long long) latent_shape.h,
                    (long long) latent_shape.w,
                    (long long) action_shape.b, (long long) action_shape.c,
                    (long long) action_shape.f, (long long) action_shape.h,
                    (long long) action_shape.w,
                    stats.ms_inference);
        return std::vector<float>((size_t) cfg.n_suffix * (size_t) cfg.real_action_dim, 0.0f);
    }

    LingBotFlowScheduler video_sched;
    video_sched.shift = 5.0;
    video_sched.sigma_min = 0.0;
    video_sched.extra_one_step = true;
    video_sched.set_timesteps(video_steps);
    LingBotFlowScheduler action_sched;
    action_sched.shift = 0.05;
    action_sched.sigma_min = 0.0;
    action_sched.extra_one_step = true;
    action_sched.set_timesteps(action_steps);

    const auto t_inf0 = std::chrono::steady_clock::now();
    const int total_steps = std::max((int) video_sched.timesteps.size(),
                                     (int) action_sched.timesteps.size());
    LingBotWanMixedForwardResult last_result;
    for (int step = 0; step <= total_steps; ++step) {
        const double video_t = step < (int) video_sched.timesteps.size()
            ? video_sched.timesteps[(size_t) step] : 0.0;
        const double action_t = step < (int) action_sched.timesteps.size()
            ? action_sched.timesteps[(size_t) step] : 0.0;
        if (!action_condition.empty()) {
            apply_action_condition_frame0(action_sample, action_condition, action_shape);
        }
        if (has_latent_condition) {
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
        }
        zero_unused_libero_action_channels(action_sample, action_shape);
        LingBotWanMixedForwardResult pred;
        int current_cache_mode = in.lingbot_cache_mode;
        if (current_cache_mode == 0 &&
            in.lingbot_session_id != 0 &&
            use_cuda_self_attn &&
            step == total_steps &&
            std::getenv("VLA_LINGBOT_DISABLE_PRED_CACHE") == nullptr) {
            current_cache_mode = 1;
        }
        const bool use_mixed_predict = std::getenv("VLA_LINGBOT_PREDICT_MIXED") != nullptr;
        char trace_tag_buf[32];
        std::snprintf(trace_tag_buf, sizeof(trace_tag_buf), "step%03d", step);
        const bool trace_tag_enabled = std::getenv("VLA_LINGBOT_PREDICT_TRACE_DUMP_DIR") != nullptr;
        if (trace_tag_enabled) {
            setenv("VLA_LINGBOT_PREDICT_TRACE_TAG", trace_tag_buf, 1);
        }
        if (use_mixed_predict) {
            LingBotWanMixedForwardResult uncond_pred;
            if (use_video_cfg) {
                if (!exec_wan_mixed_forward_tensors(g, ex_uncond,
                                                    latent_sample, latent_shape,
                                                    action_sample, action_shape,
                                                    video_t, action_t,
                                                    blocks, use_cuda_self_attn,
                                                    1, window_size,
                                                    in.lingbot_session_id,
                                                    0,
                                                    uncond_pred)) {
                    return {};
                }
            }
            if (!exec_wan_mixed_forward_tensors(g, ex,
                                                latent_sample, latent_shape,
                                                action_sample, action_shape,
                                                video_t, action_t,
                                                blocks, use_cuda_self_attn,
                                                1, window_size,
                                                in.lingbot_session_id,
                                                current_cache_mode,
                                                pred)) {
                return {};
            }
            if (use_video_cfg) {
                if (uncond_pred.latent_tensor.size() != pred.latent_tensor.size()) {
                    std::fprintf(stderr, "vla(lingbot_va): CFG latent tensor size mismatch (%zu vs %zu)\n",
                                 uncond_pred.latent_tensor.size(), pred.latent_tensor.size());
                    return {};
                }
                for (size_t i = 0; i < pred.latent_tensor.size(); ++i) {
                    const float u = uncond_pred.latent_tensor[i];
                    const float c = pred.latent_tensor[i];
                    pred.latent_tensor[i] = u + video_guidance_scale * (c - u);
                }
            }
        } else {
            if (use_video_cfg) {
                std::fprintf(stderr,
                             "vla(lingbot_va): video CFG requires VLA_LINGBOT_PREDICT_MIXED=1; "
                             "separate predict path cannot run conditional/unconditional branches together\n");
                return {};
            }
            if (use_cuda_self_attn || current_cache_mode != 0) {
                std::fprintf(stderr,
                             "vla(lingbot_va): separate predict path currently supports dense no-cache mode only; "
                             "set VLA_LINGBOT_PREDICT_MIXED=1 for mixed cached/CUDA path\n");
                return {};
            }
            if (!exec_wan_separate_forward_tensors(g, ex,
                                                   latent_sample, latent_shape,
                                                   action_sample, action_shape,
                                                   video_t, action_t,
                                                   blocks, window_size,
                                                   pred)) {
                return {};
            }
        }
        if (trace_tag_enabled) {
            unsetenv("VLA_LINGBOT_PREDICT_TRACE_TAG");
        }
        last_result = std::move(pred);
        if (step < (int) video_sched.timesteps.size()) {
            if (last_result.latent_tensor.size() != latent_sample.size()) {
                std::fprintf(stderr, "vla(lingbot_va): predict latent tensor size mismatch (%zu vs %zu)\n",
                             last_result.latent_tensor.size(), latent_sample.size());
                return {};
            }
            video_sched.step_inplace(latent_sample, last_result.latent_tensor,
                                     video_sched.timesteps[(size_t) step]);
            if (has_latent_condition) {
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
            }
        }
        if (step < (int) action_sched.timesteps.size()) {
            if (last_result.action_tensor.size() != action_sample.size()) {
                std::fprintf(stderr, "vla(lingbot_va): predict action tensor size mismatch (%zu vs %zu)\n",
                             last_result.action_tensor.size(), action_sample.size());
                return {};
            }
            action_sched.step_inplace(action_sample, last_result.action_tensor,
                                      action_sched.timesteps[(size_t) step]);
            if (!action_condition.empty()) {
                apply_action_condition_frame0(action_sample, action_condition, action_shape);
            }
            zero_unused_libero_action_channels(action_sample, action_shape);
        }
    }
    const auto t_inf1 = std::chrono::steady_clock::now();

    std::vector<float> action_tokens;
    int64_t action_feature = 0;
    int64_t action_seq = 0;
    if (!action_tensor_to_tokens(action_sample, action_shape, action_tokens, action_feature, action_seq)) {
        std::fprintf(stderr, "vla(lingbot_va): predict action flatten failed\n");
        return {};
    }
    std::vector<float> out;
    int64_t output_action_dim = cfg.max_action_dim;
    const bool postprocess_libero = lingbot_action_postprocess_libero_enabled();
    if (postprocess_libero) {
        if (!postprocess_libero_action_tokens(action_tokens, action_feature, action_seq,
                                              cfg.n_suffix, out)) {
            return {};
        }
        output_action_dim = cfg.real_action_dim;
    } else {
        out.assign((size_t) cfg.n_suffix * (size_t) cfg.max_action_dim, 0.0f);
        const int64_t copy_dim = std::min<int64_t>(cfg.max_action_dim, action_feature);
        const int64_t copy_steps = std::min<int64_t>(cfg.n_suffix, action_seq);
        for (int64_t t = 0; t < copy_steps; ++t) {
            for (int64_t c = 0; c < copy_dim; ++c) {
                out[(size_t) t * (size_t) cfg.max_action_dim + (size_t) c] =
                    action_tokens[(size_t) c + (size_t) action_feature * (size_t) t];
            }
        }
    }

    if (const char * dump_dir_c = std::getenv("VLA_LINGBOT_PREDICT_DUMP_DIR")) {
        const std::string dump_dir(dump_dir_c);
        dump_f32_file(dump_dir + "/lingbot_predict_action_chunk.f32", out);
        dump_text_file(dump_dir + "/lingbot_predict_action_chunk.shape.txt",
                       std::to_string((long long) cfg.n_suffix) + " " +
                       std::to_string((long long) output_action_dim) + "\n");
        dump_f32_file(dump_dir + "/lingbot_predict_action_tokens_raw.f32", action_tokens);
        dump_text_file(dump_dir + "/lingbot_predict_action_tokens_raw.shape.txt",
                       std::to_string((long long) action_feature) + " " +
                       std::to_string((long long) action_seq) + "\n");
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
        if (!last_result.action_tensor.empty()) {
            dump_f32_file(dump_dir + "/lingbot_predict_last_action_pred.f32", last_result.action_tensor);
            dump_text_file(dump_dir + "/lingbot_predict_last_action_pred.shape.txt",
                           std::to_string((long long) action_shape.b) + " " +
                           std::to_string((long long) action_shape.c) + " " +
                           std::to_string((long long) action_shape.f) + " " +
                           std::to_string((long long) action_shape.h) + " " +
                           std::to_string((long long) action_shape.w) + "\n");
        }
        if (!last_result.latent_tensor.empty()) {
            dump_f32_file(dump_dir + "/lingbot_predict_last_latent_pred.f32", last_result.latent_tensor);
            dump_text_file(dump_dir + "/lingbot_predict_last_latent_pred.shape.txt",
                           std::to_string((long long) latent_shape.b) + " " +
                           std::to_string((long long) latent_shape.c) + " " +
                           std::to_string((long long) latent_shape.f) + " " +
                           std::to_string((long long) latent_shape.h) + " " +
                           std::to_string((long long) latent_shape.w) + "\n");
        }
    }

    const auto t_total1 = std::chrono::steady_clock::now();
    stats.ms_vision = 0.0f;
    stats.ms_prefill = 0.0f;
    stats.ms_denoise = std::chrono::duration<float, std::milli>(t_inf1 - t_inf0).count();
    stats.ms_inference = stats.ms_denoise;
    stats.ms_total = std::chrono::duration<float, std::milli>(t_total1 - t_total0).count();

    std::printf("vla(lingbot_va): predict bridge ok blocks=%d mode=%s window=%lld "
                "video_steps=%d action_steps=%d action_postprocess=%s chunk=[%lld,%lld] checksum=%.9g max=%.9g\n",
                blocks, use_cuda_self_attn ? "cuda-flex-self-attn" : "ggml-dense",
                (long long) window_size, video_steps, action_steps,
                postprocess_libero ? "libero_quantiles" : "raw",
                (long long) cfg.n_suffix, (long long) output_action_dim,
                checksum(out), max_abs_value(out));
    return out;
}

}

std::unique_ptr<ModelArchBase> lingbot_va_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string&) {
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

    auto m = std::make_unique<LingBotVAModelArch>(ckpt_path, g);
    if (!validate_transformer_tensors(g, m->n_layers)) return nullptr;

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
