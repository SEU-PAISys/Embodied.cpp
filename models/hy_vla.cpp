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
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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
            std::fprintf(stderr, "vla(hy_vla): gguf_init_from_file failed for %s\n", path.c_str());
            return false;
        }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "vla(hy_vla): fopen failed for %s\n", path.c_str());
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

    bool has_key(const char * k) const { return gguf_find_key(gctx, k) >= 0; }
    uint32_t u32(const char * k) const { return gguf_get_val_u32(gctx, gguf_find_key(gctx, k)); }
    float    f32(const char * k) const { return gguf_get_val_f32(gctx, gguf_find_key(gctx, k)); }
    std::string str(const char * k) const { return gguf_get_val_str(gctx, gguf_find_key(gctx, k)); }
    const ggml_tensor * meta(const char * name) const { return ggml_get_tensor(meta_ctx, name); }
    ggml_type tensor_type(const char * name) const {
        const ggml_tensor * t = meta(name);
        return t ? t->type : GGML_TYPE_COUNT;
    }
    const char * first_existing(const char * a, const char * b) const {
        if (meta(a)) return a;
        if (meta(b)) return b;
        return a;
    }

    bool read_raw(const char * name, void * buf) {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) {
            std::fprintf(stderr, "vla(hy_vla): missing tensor %s\n", name);
            return false;
        }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t nb  = gguf_get_tensor_size(gctx, id);
        if (std::fseek(fp, (long) off, SEEK_SET) != 0) return false;
        return std::fread(buf, 1, nb, fp) == nb;
    }

    std::vector<uint8_t> read_convert(const char * name, ggml_type target) {
        const ggml_tensor * t = meta(name);
        if (!t) {
            std::fprintf(stderr, "vla(hy_vla): missing tensor %s\n", name);
            return {};
        }
        if (t->type == target) {
            const int64_t id = gguf_find_tensor(gctx, name);
            if (id < 0) return {};
            const size_t nb = gguf_get_tensor_size(gctx, id);
            std::vector<uint8_t> out(nb);
            if (!read_raw(name, out.data())) return {};
            return out;
        }
        const int64_t n = ggml_nelements(t);
        std::vector<float> f32((size_t) n);
        if (t->type == GGML_TYPE_F32) {
            if (!read_raw(name, f32.data())) return {};
        } else if (t->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp((size_t) n);
            if (!read_raw(name, tmp.data())) return {};
            ggml_bf16_to_fp32_row(tmp.data(), f32.data(), n);
        } else {
            std::fprintf(stderr, "vla(hy_vla): unsupported tensor dtype %d for %s\n", (int) t->type, name);
            return {};
        }
        if (target == GGML_TYPE_F32) {
            std::vector<uint8_t> out((size_t) n * sizeof(float));
            std::memcpy(out.data(), f32.data(), out.size());
            return out;
        }
        if (target == GGML_TYPE_BF16) {
            std::vector<uint8_t> out((size_t) n * sizeof(ggml_bf16_t));
            ggml_fp32_to_bf16_row(f32.data(), reinterpret_cast<ggml_bf16_t *>(out.data()), n);
            return out;
        }
        if (ggml_is_quantized(target)) {
            const int64_t n_per_row = t->ne[0];
            const int64_t nrows = n / n_per_row;
            const int64_t blck = ggml_blck_size(target);
            if (n_per_row % blck != 0) {
                std::fprintf(stderr,
                             "vla(hy_vla): cannot quantize %s to %s: ne0=%lld not divisible by block=%lld\n",
                             name, ggml_type_name(target), (long long) n_per_row, (long long) blck);
                return {};
            }
            const size_t qbytes = (size_t) nrows * (size_t) (n_per_row / blck) * ggml_type_size(target);
            std::vector<uint8_t> out(qbytes);
            const size_t written = ggml_quantize_chunk(target, f32.data(), out.data(), 0, nrows, n_per_row, nullptr);
            if (written != qbytes) {
                std::fprintf(stderr,
                             "vla(hy_vla): quantized byte mismatch for %s (%zu vs %zu)\n",
                             name, written, qbytes);
                return {};
            }
            return out;
        }
        std::fprintf(stderr, "vla(hy_vla): unsupported resident dtype %d for %s\n", (int) target, name);
        return {};
    }

    bool read_to_f32(const char * name, std::vector<float> & out) {
        const ggml_tensor * t = meta(name);
        if (!t) return false;
        out.resize((size_t) ggml_nelements(t));
        if (t->type == GGML_TYPE_F32) return read_raw(name, out.data());
        if (t->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp((size_t) ggml_nelements(t));
            if (!read_raw(name, tmp.data())) return false;
            ggml_bf16_to_fp32_row(tmp.data(), out.data(), ggml_nelements(t));
            return true;
        }
        return false;
    }

    bool fetch_rows_f32(const char * name, const std::vector<int32_t> & row_ids,
                        float * dst, int64_t cols) {
        const ggml_tensor * t = meta(name);
        if (!t) {
            std::fprintf(stderr, "vla(hy_vla): missing tensor %s\n", name);
            return false;
        }
        if (t->ne[0] != cols || t->ne[2] != 1 || t->ne[3] != 1) {
            std::fprintf(stderr, "vla(hy_vla): %s shape unfit for row-fetch\n", name);
            return false;
        }
        const int64_t rows = t->ne[1];
        const int64_t id = gguf_find_tensor(gctx, name);
        const size_t base = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t elsz = (t->type == GGML_TYPE_F32) ? 4u : 2u;
        const size_t rb = (size_t) cols * elsz;
        std::vector<uint8_t> row(rb);
        for (size_t k = 0; k < row_ids.size(); ++k) {
            const int32_t r = row_ids[k];
            if (r < 0 || r >= rows) {
                std::fprintf(stderr, "vla(hy_vla): token id %d out of range for %s rows=%lld\n",
                             r, name, (long long) rows);
                return false;
            }
            if (std::fseek(fp, (long) (base + (size_t) r * rb), SEEK_SET) != 0) return false;
            if (std::fread(row.data(), 1, rb, fp) != rb) return false;
            if (elsz == 4) std::memcpy(dst + k * cols, row.data(), rb);
            else ggml_bf16_to_fp32_row(reinterpret_cast<ggml_bf16_t *>(row.data()), dst + k * cols, cols);
        }
        return true;
    }
};

ggml_type hy_vla_matmul_type_from_env() {
    if (std::getenv("VLA_HY_VLA_F32_WEIGHTS")) return GGML_TYPE_F32;
    const char * env = std::getenv("VLA_HY_VLA_WEIGHT_DTYPE");
    if (!env || std::strlen(env) == 0) return GGML_TYPE_BF16;
    if (std::strcmp(env, "f32") == 0 || std::strcmp(env, "F32") == 0) return GGML_TYPE_F32;
    if (std::strcmp(env, "bf16") == 0 || std::strcmp(env, "BF16") == 0) return GGML_TYPE_BF16;
    if (std::strcmp(env, "q8_0") == 0 || std::strcmp(env, "Q8_0") == 0) return GGML_TYPE_Q8_0;
    if (std::strcmp(env, "q6_K") == 0 || std::strcmp(env, "Q6_K") == 0 || std::strcmp(env, "q6_k") == 0) return GGML_TYPE_Q6_K;
    if (std::strcmp(env, "q5_K") == 0 || std::strcmp(env, "Q5_K") == 0 || std::strcmp(env, "q5_k") == 0) return GGML_TYPE_Q5_K;
    if (std::strcmp(env, "q4_K") == 0 || std::strcmp(env, "Q4_K") == 0 || std::strcmp(env, "q4_k") == 0) return GGML_TYPE_Q4_K;
    if (std::strcmp(env, "q4_0") == 0 || std::strcmp(env, "Q4_0") == 0) return GGML_TYPE_Q4_0;
    std::fprintf(stderr,
                 "vla(hy_vla): unknown VLA_HY_VLA_WEIGHT_DTYPE='%s', using bf16. "
                 "Supported: f32, bf16, q8_0, q6_K, q5_K, q4_K, q4_0\n",
                 env);
    return GGML_TYPE_BF16;
}

struct HyLayerW {
    ggml_tensor * ln_in = nullptr;
    ggml_tensor * Wq = nullptr;
    ggml_tensor * Wk = nullptr;
    ggml_tensor * Wv = nullptr;
    ggml_tensor * Wo = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
    ggml_tensor * ln_post = nullptr;
    ggml_tensor * Wgate = nullptr;
    ggml_tensor * Wup = nullptr;
    ggml_tensor * Wdown = nullptr;
};

struct HyVisionBlockW {
    ggml_tensor * norm1_w = nullptr;
    ggml_tensor * norm1_b = nullptr;
    ggml_tensor * qkv_w = nullptr;
    ggml_tensor * qkv_b = nullptr;
    ggml_tensor * proj_w = nullptr;
    ggml_tensor * proj_b = nullptr;
    ggml_tensor * norm2_w = nullptr;
    ggml_tensor * norm2_b = nullptr;
    ggml_tensor * fc1_w = nullptr;
    ggml_tensor * fc1_b = nullptr;
    ggml_tensor * fc2_w = nullptr;
    ggml_tensor * fc2_b = nullptr;
};

struct HyVisionW {
    ggml_tensor * patch_w = nullptr;
    ggml_tensor * patch_b = nullptr;
    ggml_tensor * pos_embed = nullptr;
    ggml_tensor * pos_embed_14 = nullptr;
    ggml_tensor * merger_proj1_w = nullptr;
    ggml_tensor * merger_proj1_b = nullptr;
    ggml_tensor * merger_proj2_w = nullptr;
    ggml_tensor * merger_proj2_b = nullptr;
    ggml_tensor * pooler_fc0_w = nullptr;
    ggml_tensor * pooler_fc0_b = nullptr;
    ggml_tensor * pooler_fc2_w = nullptr;
    ggml_tensor * pooler_fc2_b = nullptr;
    std::vector<HyVisionBlockW> blocks;
    int layers_loaded = 0;
    bool base_loaded = false;
};

struct HyPrefixRun {
    int64_t start = 0;
    int64_t len = 0;
    bool vision = false;
};

struct HyPrefixRange {
    int64_t start = 0;
    int64_t end = 0;
};

struct HyVLAModelArch final : ModelArchBase {
    HyVLAModelArch() : ModelArchBase(Arch::HY_VLA) {}
    ~HyVLAModelArch() override {
        if (vision_weight_buf)  ggml_backend_buffer_free(vision_weight_buf);
        if (vision_galloc)      ggml_gallocr_free(vision_galloc);
        if (routed_galloc)      ggml_gallocr_free(routed_galloc);
        if (ctx_vision_weights) ggml_free(ctx_vision_weights);
        if (vision_backend)     ggml_backend_free(vision_backend);
        if (weight_buf)  ggml_backend_buffer_free(weight_buf);
        if (ctx_weights) ggml_free(ctx_weights);
        if (backend)     ggml_backend_free(backend);
    }

    std::vector<float> predict(const Inputs & in) override;

    std::string ckpt_path;
    ggml_backend_t backend = nullptr;
    ggml_context * ctx_weights = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_backend_t vision_backend = nullptr;
    ggml_context * ctx_vision_weights = nullptr;
    ggml_backend_buffer_t vision_weight_buf = nullptr;
    mutable ggml_gallocr_t vision_galloc = nullptr;
    mutable ggml_gallocr_t routed_galloc = nullptr;
    ggml_type matmul_type = GGML_TYPE_BF16;
    bool is_cuda = false;
    bool vlm_only = false;
    int32_t eos_token_id = 120020;
    int n_threads = 4;
    int text_layers_loaded = 0;
    bool prefix_vision_loaded = false;
    int vision_layers_loaded = 0;
    bool vision_frontend_loaded = false;

    std::vector<HyLayerW> layers;
    std::vector<HyLayerW> text_layers;
    std::vector<HyLayerW> vision_layers;
    HyVisionW vision_frontend;
    HyVisionW vision_cpu_frontend;
    bool vision_cpu_sideload = false;
    ggml_tensor * Wnorm = nullptr;
    ggml_tensor * Wtext_norm = nullptr;
    ggml_tensor * W_vocab = nullptr;
    ggml_tensor * W_sp = nullptr;
    ggml_tensor * b_sp = nullptr;
    ggml_tensor * W_ain = nullptr;
    ggml_tensor * b_ain = nullptr;
    ggml_tensor * W_at1 = nullptr;
    ggml_tensor * b_at1 = nullptr;
    ggml_tensor * W_at2 = nullptr;
    ggml_tensor * b_at2 = nullptr;
    ggml_tensor * W_aout = nullptr;
    ggml_tensor * b_aout = nullptr;

    std::vector<float> state_mean, state_std, action_mean, action_std;
    std::vector<float> action_mean_abs, action_std_abs;
    std::mt19937 rng{std::random_device{}()};
};

static inline ggml_tensor * mm_w(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    ggml_tensor * r = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_prec(r, GGML_PREC_F32);
    return r;
}

ggml_tensor * layer_norm_affine(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * w, ggml_tensor * b,
                                float eps = 1e-6f) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), w), b);
}

ggml_tensor * concat_seq(ggml_context * ctx, const std::vector<ggml_tensor *> & xs);
bool write_binary_file(const char * path, const void * data, size_t bytes);

size_t hy_vla_graph_size(size_t fallback) {
    if (const char * e = std::getenv("VLA_HY_VLA_GRAPH_SIZE")) {
        const long long v = std::atoll(e);
        if (v > 0) return (size_t) v;
    }
    return fallback;
}

ggml_tensor * build_hy_vision_block(ggml_context * ctx, const HyVisionBlockW & w,
                                    ggml_tensor * x_in, int64_t seq,
                                    int64_t width = 1152, int64_t n_heads = 16) {
    const int64_t head_dim = width / n_heads;
    ggml_tensor * x_norm = layer_norm_affine(ctx, x_in, w.norm1_w, w.norm1_b);
    ggml_tensor * qkv = ggml_add(ctx, mm_w(ctx, w.qkv_w, x_norm), w.qkv_b);
    const size_t row_stride = (size_t) qkv->nb[1];
    const size_t elem_size = ggml_element_size(qkv);
    ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, width, seq, row_stride, 0));
    ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, width, seq, row_stride, (size_t) width * elem_size));
    ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, width, seq, row_stride, (size_t) width * 2u * elem_size));
    ggml_tensor * qh = ggml_reshape_3d(ctx, q, head_dim, n_heads, seq);
    ggml_tensor * kh = ggml_reshape_3d(ctx, k, head_dim, n_heads, seq);
    ggml_tensor * vh = ggml_reshape_3d(ctx, v, head_dim, n_heads, seq);
    ggml_tensor * Q = ggml_permute(ctx, qh, 0, 2, 1, 3);
    ggml_tensor * K = ggml_permute(ctx, kh, 0, 2, 1, 3);
    ggml_tensor * V = ggml_permute(ctx, vh, 0, 2, 1, 3);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, K, V, nullptr,
                                           1.f / std::sqrt((float) head_dim), 0.f, 0.f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_tensor * att = ggml_reshape_2d(ctx, fa, width, seq);
    ggml_tensor * h1 = ggml_add(ctx, x_in, ggml_add(ctx, mm_w(ctx, w.proj_w, att), w.proj_b));

    ggml_tensor * x_mlp = layer_norm_affine(ctx, h1, w.norm2_w, w.norm2_b);
    ggml_tensor * hidden = ggml_gelu_erf(ctx, ggml_add(ctx, mm_w(ctx, w.fc1_w, x_mlp), w.fc1_b));
    return ggml_add(ctx, h1, ggml_add(ctx, mm_w(ctx, w.fc2_w, hidden), w.fc2_b));
}

ggml_tensor * build_hy_vision_block_per_frame(ggml_context * ctx, const HyVisionBlockW & w,
                                              ggml_tensor * x_in, int64_t n_patches,
                                              int64_t n_frames,
                                              int64_t width = 1152,
                                              int64_t n_heads = 16) {
    if (n_frames <= 1) {
        return build_hy_vision_block(ctx, w, x_in, n_patches, width, n_heads);
    }
    const size_t rb = (size_t) width * sizeof(float);
    std::vector<ggml_tensor *> frames;
    frames.reserve((size_t) n_frames);
    for (int64_t f = 0; f < n_frames; ++f) {
        ggml_tensor * frame = ggml_view_2d(ctx, x_in, width, n_patches, rb,
                                           rb * (size_t) f * (size_t) n_patches);
        frames.push_back(build_hy_vision_block(ctx, w, frame, n_patches, width, n_heads));
    }
    return concat_seq(ctx, frames);
}

ggml_tensor * build_hy_vision_block_mem(ggml_context * ctx, const HyVisionBlockW & w,
                                        ggml_tensor * x_in, int64_t n_patches,
                                        int64_t n_frames, ggml_tensor * time_pe,
                                        ggml_tensor * time_mask,
                                        int64_t width = 1152, int64_t n_heads = 16) {
    if (n_frames <= 1) {
        return build_hy_vision_block(ctx, w, x_in, n_patches, width, n_heads);
    }
    const int64_t head_dim = width / n_heads;
    const int64_t seq = n_patches * n_frames;
    ggml_tensor * x_norm = layer_norm_affine(ctx, ggml_add(ctx, x_in, time_pe),
                                             w.norm1_w, w.norm1_b);
    ggml_tensor * qkv = ggml_add(ctx, mm_w(ctx, w.qkv_w, x_norm), w.qkv_b);
    const size_t row_stride = (size_t) qkv->nb[1];
    const size_t elem_size = ggml_element_size(qkv);
    ggml_tensor * q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, width, seq, row_stride, 0));
    ggml_tensor * k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, width, seq, row_stride, (size_t) width * elem_size));
    ggml_tensor * v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, width, seq, row_stride, (size_t) width * 2u * elem_size));

    const float scale = 1.f / std::sqrt((float) head_dim);

    ggml_tensor * v_mixed = nullptr;
    ggml_tensor * q4 = ggml_reshape_4d(ctx, q, head_dim, n_heads, n_patches, n_frames);
    ggml_tensor * k4 = ggml_reshape_4d(ctx, k, head_dim, n_heads, n_patches, n_frames);
    ggml_tensor * v4 = ggml_reshape_4d(ctx, v, head_dim, n_heads, n_patches, n_frames);

    if (std::getenv("VLA_HY_VLA_BATCHED_MEM_ATTN")) {
        // Vectorized temporal attention:
        // old path launches one tiny flash-attn per patch; this batches patches
        // in ne[3], preserving independent causal attention for each patch/head.
        ggml_tensor * Qtime = ggml_cont(ctx, ggml_permute(ctx, q4, 0, 3, 1, 2)); // [D, T, H, P]
        ggml_tensor * Ktime = ggml_cont(ctx, ggml_permute(ctx, k4, 0, 3, 1, 2));
        ggml_tensor * Vtime = ggml_cont(ctx, ggml_permute(ctx, v4, 0, 3, 1, 2));
        ggml_tensor * time_att = ggml_flash_attn_ext(ctx, Qtime, Ktime, Vtime,
                                                     time_mask, scale, 0.f, 0.f);
        ggml_flash_attn_ext_set_prec(time_att, GGML_PREC_F32);
        ggml_tensor * time_back = ggml_permute(ctx, time_att, 0, 2, 3, 1); // [D, H, P, T]
        v_mixed = ggml_reshape_2d(ctx, ggml_cont(ctx, time_back), width, seq);
    } else {
        std::vector<std::vector<ggml_tensor *>> frame_patch_cols((size_t) n_frames);
        for (int64_t f = 0; f < n_frames; ++f) {
            frame_patch_cols[(size_t) f].reserve((size_t) n_patches);
        }
        for (int64_t p = 0; p < n_patches; ++p) {
            const size_t patch_off = q4->nb[2] * (size_t) p;
            ggml_tensor * qp = ggml_view_3d(ctx, q4, head_dim, n_heads, n_frames,
                                            q4->nb[1], q4->nb[3], patch_off);
            ggml_tensor * kp = ggml_view_3d(ctx, k4, head_dim, n_heads, n_frames,
                                            k4->nb[1], k4->nb[3], patch_off);
            ggml_tensor * vp = ggml_view_3d(ctx, v4, head_dim, n_heads, n_frames,
                                            v4->nb[1], v4->nb[3], patch_off);
            ggml_tensor * Qp = ggml_permute(ctx, qp, 0, 2, 1, 3);
            ggml_tensor * Kp = ggml_permute(ctx, kp, 0, 2, 1, 3);
            ggml_tensor * Vp = ggml_permute(ctx, vp, 0, 2, 1, 3);
            ggml_tensor * time_att = ggml_flash_attn_ext(ctx, Qp, Kp, Vp, time_mask,
                                                         scale, 0.f, 0.f);
            ggml_flash_attn_ext_set_prec(time_att, GGML_PREC_F32);
            ggml_tensor * mixed = ggml_reshape_2d(ctx, time_att, width, n_frames);
            const size_t mixed_rb = (size_t) width * sizeof(float);
            for (int64_t f = 0; f < n_frames; ++f) {
                frame_patch_cols[(size_t) f].push_back(
                    ggml_view_2d(ctx, mixed, width, 1, mixed_rb, mixed_rb * (size_t) f));
            }
        }
        std::vector<ggml_tensor *> frame_tokens;
        frame_tokens.reserve((size_t) n_frames);
        for (int64_t f = 0; f < n_frames; ++f) {
            frame_tokens.push_back(concat_seq(ctx, frame_patch_cols[(size_t) f]));
        }
        v_mixed = concat_seq(ctx, frame_tokens);
    }

    std::vector<ggml_tensor *> space_att_frames;
    space_att_frames.reserve((size_t) n_frames);
    const size_t rb = (size_t) width * sizeof(float);
    for (int64_t f = 0; f < n_frames; ++f) {
        const size_t off = rb * (size_t) f * (size_t) n_patches;
        ggml_tensor * qf = ggml_view_2d(ctx, q, width, n_patches, rb, off);
        ggml_tensor * kf = ggml_view_2d(ctx, k, width, n_patches, rb, off);
        ggml_tensor * vf = ggml_view_2d(ctx, v_mixed, width, n_patches, rb, off);
        ggml_tensor * qh = ggml_reshape_3d(ctx, qf, head_dim, n_heads, n_patches);
        ggml_tensor * kh = ggml_reshape_3d(ctx, kf, head_dim, n_heads, n_patches);
        ggml_tensor * vh = ggml_reshape_3d(ctx, vf, head_dim, n_heads, n_patches);
        ggml_tensor * Q = ggml_permute(ctx, qh, 0, 2, 1, 3);
        ggml_tensor * K = ggml_permute(ctx, kh, 0, 2, 1, 3);
        ggml_tensor * V = ggml_permute(ctx, vh, 0, 2, 1, 3);
        ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, K, V, nullptr, scale, 0.f, 0.f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        space_att_frames.push_back(ggml_reshape_2d(ctx, fa, width, n_patches));
    }
    ggml_tensor * att = concat_seq(ctx, space_att_frames);
    ggml_tensor * h1 = ggml_add(ctx, x_in, ggml_add(ctx, mm_w(ctx, w.proj_w, att), w.proj_b));

    ggml_tensor * x_mlp = layer_norm_affine(ctx, h1, w.norm2_w, w.norm2_b);
    ggml_tensor * hidden = ggml_gelu_erf(ctx, ggml_add(ctx, mm_w(ctx, w.fc1_w, x_mlp), w.fc1_b));
    return ggml_add(ctx, h1, ggml_add(ctx, mm_w(ctx, w.fc2_w, hidden), w.fc2_b));
}

ggml_tensor * build_hy_vision_merger_14(ggml_context * ctx, const HyVLAModelArch & m,
                                        ggml_tensor * vit_tokens) {
    constexpr int64_t in_width = 1152;
    constexpr int64_t out_width = 2048;
    constexpr int64_t src_grid = 14;
    constexpr int64_t dst_grid = 7;

    ggml_tensor * x = ggml_add(ctx, mm_w(ctx, m.vision_frontend.merger_proj1_w, vit_tokens),
                               m.vision_frontend.merger_proj1_b);
    std::vector<ggml_tensor *> pooled_cells;
    pooled_cells.reserve((size_t) dst_grid * (size_t) dst_grid);
    const size_t rb = (size_t) out_width * sizeof(float);
    for (int64_t gy = 0; gy < dst_grid; ++gy) {
        for (int64_t gx = 0; gx < dst_grid; ++gx) {
            const int64_t p0 = (2 * gy) * src_grid + (2 * gx);
            const int64_t ids[4] = {p0, p0 + 1, p0 + src_grid, p0 + src_grid + 1};
            std::vector<ggml_tensor *> patches;
            patches.reserve(4);
            for (int k = 0; k < 4; ++k) {
                patches.push_back(ggml_view_2d(ctx, x, out_width, 1, rb, rb * (size_t) ids[k]));
            }
            ggml_tensor * cell = concat_seq(ctx, patches); // [C, 4]
            ggml_tensor * cell_t = ggml_cont(ctx, ggml_transpose(ctx, cell)); // [4, C]
            ggml_tensor * mean_t = ggml_mean(ctx, cell_t); // [1, C]
            ggml_tensor * pooled_t = ggml_repeat(ctx, mean_t, cell_t); // [4, C]
            ggml_tensor * pooled = ggml_cont(ctx, ggml_transpose(ctx, pooled_t)); // [C, 4]
            ggml_tensor * fused = ggml_concat(ctx, cell, pooled, 0); // [2C, 4]
            ggml_tensor * score = ggml_add(ctx, mm_w(ctx, m.vision_frontend.pooler_fc0_w, fused),
                                           m.vision_frontend.pooler_fc0_b);
            score = ggml_gelu_erf(ctx, score);
            score = ggml_add(ctx, mm_w(ctx, m.vision_frontend.pooler_fc2_w, score),
                             m.vision_frontend.pooler_fc2_b); // [C, 4]
            ggml_tensor * weights_t = ggml_soft_max(ctx, ggml_cont(ctx, ggml_transpose(ctx, score))); // [4, C]
            ggml_tensor * weights = ggml_cont(ctx, ggml_transpose(ctx, weights_t)); // [C, 4]
            ggml_tensor * weighted_t = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul(ctx, cell, weights))); // [4, C]
            ggml_tensor * pooled_cell_t = ggml_sum_rows(ctx, weighted_t); // [1, C]
            pooled_cells.push_back(ggml_cont(ctx, ggml_transpose(ctx, pooled_cell_t))); // [C, 1]
        }
    }
    ggml_tensor * pooled = concat_seq(ctx, pooled_cells); // [C, 49]
    ggml_tensor * activated = ggml_gelu_erf(ctx, pooled);
    return ggml_add(ctx, mm_w(ctx, m.vision_frontend.merger_proj2_w, activated),
                    m.vision_frontend.merger_proj2_b);
}

ggml_tensor * build_hy_vision_patch_embed_224(ggml_context * ctx, const HyVLAModelArch & m,
                                              ggml_tensor * pixels_chw) {
    constexpr int64_t image_size = 224;
    constexpr int64_t patch_size = 16;
    constexpr int64_t width = 1152;
    constexpr int64_t grid = image_size / patch_size;
    constexpr int64_t n_patches = grid * grid;
    ggml_tensor * conv = ggml_conv_2d(ctx, m.vision_frontend.patch_w, pixels_chw,
                                      patch_size, patch_size, 0, 0, 1, 1);
    ggml_tensor * patches = ggml_transpose(ctx, ggml_reshape_2d(ctx, conv, n_patches, width));
    patches = ggml_add(ctx, ggml_cont(ctx, patches), m.vision_frontend.patch_b);
    return ggml_add(ctx, patches, m.vision_frontend.pos_embed_14);
}

bool preprocess_hy_image_chw_224(const ImageView & v, std::vector<float> & out) {
    constexpr int64_t side = 224;
    if (v.w < 1 || v.h < 1 || !v.data) {
        std::fprintf(stderr, "vla(hy_vla): invalid image view %dx%d\n", v.w, v.h);
        return false;
    }
    out.assign((size_t) 3 * side * side, -1.0f);
    const double ratio = std::max((double) v.w / (double) side, (double) v.h / (double) side);
    const int resized_h = std::max(1, std::min<int>((int) side, (int) ((double) v.h / ratio)));
    const int resized_w = std::max(1, std::min<int>((int) side, (int) ((double) v.w / ratio)));
    const int pad_top = ((int) side - resized_h) / 2;
    const int pad_left = ((int) side - resized_w) / 2;
    auto pixel01 = [&](int y, int x, int c) -> float {
        y = std::max(0, std::min(v.h - 1, y));
        x = std::max(0, std::min(v.w - 1, x));
        if (v.format == PixelFormat::U8) {
            return ((const uint8_t *) v.data)[((size_t) y * (size_t) v.w + (size_t) x) * 3u + (size_t) c] / 255.0f;
        }
        return ((const float *) v.data)[((size_t) y * (size_t) v.w + (size_t) x) * 3u + (size_t) c];
    };
    for (int ry = 0; ry < resized_h; ++ry) {
        const float sy = (float) (((double) ry + 0.5) * (double) v.h / (double) resized_h - 0.5);
        const int y0 = std::max(0, std::min(v.h - 1, (int) std::floor(sy)));
        const int y1 = std::max(0, std::min(v.h - 1, y0 + 1));
        const float wy = sy - std::floor(sy);
        for (int rx = 0; rx < resized_w; ++rx) {
            const float sx = (float) (((double) rx + 0.5) * (double) v.w / (double) resized_w - 0.5);
            const int x0 = std::max(0, std::min(v.w - 1, (int) std::floor(sx)));
            const int x1 = std::max(0, std::min(v.w - 1, x0 + 1));
            const float wx = sx - std::floor(sx);
            const int oy = pad_top + ry;
            const int ox = pad_left + rx;
            for (int64_t c = 0; c < 3; ++c) {
                const float top = pixel01(y0, x0, (int) c) * (1.f - wx) + pixel01(y0, x1, (int) c) * wx;
                const float bot = pixel01(y1, x0, (int) c) * (1.f - wx) + pixel01(y1, x1, (int) c) * wx;
                const float px = top * (1.f - wy) + bot * wy;
                out[(size_t) c * (size_t) side * (size_t) side + (size_t) oy * (size_t) side + (size_t) ox] =
                    px * 2.0f - 1.0f;
            }
        }
    }
    return true;
}

std::vector<float> hy_mem_time_pe_2d(int64_t width, int64_t n_patches,
                                     int64_t n_frames, float base = 100.f) {
    std::vector<float> out((size_t) width * (size_t) n_patches * (size_t) n_frames, 0.f);
    for (int64_t f = 0; f < n_frames; ++f) {
        for (int64_t d = 0; d < width; d += 2) {
            const float inv = std::exp(-(std::log(base) * (float) d) / (float) width);
            const float s = std::sin((float) f * inv);
            const float c = std::cos((float) f * inv) - 1.f;
            for (int64_t p = 0; p < n_patches; ++p) {
                const size_t col = (size_t) f * (size_t) n_patches + (size_t) p;
                out[col * (size_t) width + (size_t) d] = s;
                if (d + 1 < width) {
                    out[col * (size_t) width + (size_t) d + 1] = c;
                }
            }
        }
    }
    return out;
}

std::vector<ggml_fp16_t> hy_mem_causal_mask(int64_t n_frames) {
    std::vector<ggml_fp16_t> out((size_t) n_frames * (size_t) n_frames);
    for (int64_t r = 0; r < n_frames; ++r) {
        for (int64_t c = 0; c < n_frames; ++c) {
            out[(size_t) r * (size_t) n_frames + (size_t) c] =
                ggml_fp32_to_fp16(c <= r ? 0.f : -INFINITY);
        }
    }
    return out;
}

int hy_vla_video_history_from_env(int n_images) {
    int history = 1;
    if (const char * e = std::getenv("VLA_HY_VLA_VIDEO_HISTORY")) {
        history = std::max(1, std::atoi(e));
    }
    if (history > 1 && (n_images < history || n_images % history != 0)) {
        std::fprintf(stderr,
                     "vla(hy_vla): ignoring VLA_HY_VLA_VIDEO_HISTORY=%d because n_images=%d is not divisible by it\n",
                     history, n_images);
        history = 1;
    }
    return history;
}

bool run_hy_vision_frontend_group_224(const HyVLAModelArch & m, const HyVisionW & vf,
                                      ggml_backend_t vision_backend,
                                      int vision_layers_loaded,
                                      const ImageView * images,
                                      int video_history,
                                      float * out_group_tokens,
                                      float * out_group_vit_tokens = nullptr) {
    constexpr int64_t image_size = 224;
    constexpr int64_t n_patches = 196;
    constexpr int64_t vit_width = 1152;

    ggml_init_params vp{(size_t) 3072 * 1024 * 1024, nullptr, true};
    ggml_context * V = ggml_init(vp);
    if (!V) return false;

    HyVLAModelArch mv;
    mv.vision_frontend = vf;

    std::vector<ggml_tensor *> t_px((size_t) video_history);
    std::vector<ggml_tensor *> frame_patches;
    frame_patches.reserve((size_t) video_history);
    for (int f = 0; f < video_history; ++f) {
        t_px[(size_t) f] = ggml_new_tensor_3d(V, GGML_TYPE_F32, image_size, image_size, 3);
        ggml_set_input(t_px[(size_t) f]);
        frame_patches.push_back(build_hy_vision_patch_embed_224(V, mv, t_px[(size_t) f]));
    }

    ggml_tensor * h = concat_seq(V, frame_patches);
    ggml_tensor * t_time_pe = nullptr;
    ggml_tensor * t_time_mask = nullptr;
    const bool use_mem_layers = video_history > 1 && vision_layers_loaded >= 4;
    if (use_mem_layers) {
        t_time_pe = ggml_new_tensor_2d(V, GGML_TYPE_F32, vit_width, n_patches * video_history);
        ggml_set_input(t_time_pe);
        t_time_mask = ggml_new_tensor_2d(V, GGML_TYPE_F16, video_history, video_history);
        ggml_set_input(t_time_mask);
    }
    for (int i = 0; i < vision_layers_loaded; ++i) {
        const bool mem_layer = use_mem_layers && ((i + 1) % 4 == 0);
        h = mem_layer
            ? build_hy_vision_block_mem(V, vf.blocks[(size_t) i], h, n_patches,
                                        video_history, t_time_pe, t_time_mask)
            : build_hy_vision_block_per_frame(V, vf.blocks[(size_t) i], h, n_patches,
                                              video_history);
    }
    if (video_history > 1) {
        const size_t rb = (size_t) vit_width * sizeof(float);
        h = ggml_view_2d(V, h, vit_width, n_patches, rb,
                         rb * (size_t) (video_history - 1) * (size_t) n_patches);
    }
    ggml_tensor * vit_out = h;
    h = build_hy_vision_merger_14(V, mv, h);
    if (out_group_vit_tokens) {
        ggml_set_output(vit_out);
    }
    ggml_set_output(h);

    ggml_cgraph * vg = ggml_new_graph_custom(V, 65536, false);
    ggml_build_forward_expand(vg, h);
    const bool persist_vision_galloc =
        !m.vision_cpu_sideload && std::getenv("VLA_HY_VLA_PERSIST_VISION_GALLOC");
    ggml_gallocr_t vga = persist_vision_galloc ? m.vision_galloc : nullptr;
    if (!vga) {
        vga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(vision_backend));
        if (persist_vision_galloc) {
            m.vision_galloc = vga;
        }
    }
    if (!vga || !ggml_gallocr_alloc_graph(vga, vg)) {
        std::fprintf(stderr, "vla(hy_vla): vision frontend graph allocation failed\n");
        if (!persist_vision_galloc && vga) ggml_gallocr_free(vga);
        ggml_free(V);
        return false;
    }

    std::vector<float> chw;
    for (int f = 0; f < video_history; ++f) {
        if (!preprocess_hy_image_chw_224(images[f], chw)) {
            if (!persist_vision_galloc) ggml_gallocr_free(vga);
            ggml_free(V);
            return false;
        }
        ggml_backend_tensor_set(t_px[(size_t) f], chw.data(), 0,
                                ggml_nbytes(t_px[(size_t) f]));
    }
    if (use_mem_layers) {
        const std::vector<float> time_pe = hy_mem_time_pe_2d(vit_width, n_patches, video_history);
        const std::vector<ggml_fp16_t> time_mask = hy_mem_causal_mask(video_history);
        ggml_backend_tensor_set(t_time_pe, time_pe.data(), 0, ggml_nbytes(t_time_pe));
        ggml_backend_tensor_set(t_time_mask, time_mask.data(), 0, ggml_nbytes(t_time_mask));
    }

    if (ggml_backend_graph_compute(vision_backend, vg) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(hy_vla): vision frontend graph compute failed (history=%d)\n",
                     video_history);
        if (!persist_vision_galloc) ggml_gallocr_free(vga);
        ggml_free(V);
        return false;
    }
    if (out_group_vit_tokens) {
        ggml_backend_tensor_get(vit_out, out_group_vit_tokens, 0, ggml_nbytes(vit_out));
    }
    ggml_backend_tensor_get(h, out_group_tokens, 0, ggml_nbytes(h));
    if (!persist_vision_galloc) ggml_gallocr_free(vga);
    ggml_free(V);
    return true;
}

bool run_hy_vision_frontend_224(const HyVLAModelArch & m, const Inputs & in,
                                std::vector<float> & out_tokens, int & out_views) {
    const HyVisionW & vf = m.vision_cpu_sideload ? m.vision_cpu_frontend : m.vision_frontend;
    ggml_backend_t vision_backend = m.vision_cpu_sideload ? m.vision_backend : m.backend;
    const int vision_layers_loaded = m.vision_cpu_sideload ? m.vision_cpu_frontend.layers_loaded : m.vision_layers_loaded;
    if (!m.vision_frontend_loaded || !vf.base_loaded || !vision_backend) {
        std::fprintf(stderr, "vla(hy_vla): image path requires VLA_HY_VLA_VISION_LAYERS=0 or higher\n");
        return false;
    }
    if (!in.images || in.n_images < 1) {
        std::fprintf(stderr, "vla(hy_vla): image path requires at least one 224x224 image\n");
        return false;
    }
    constexpr int64_t image_size = 224;
    constexpr int64_t n_visual = 49;
    constexpr int64_t hidden = 2048;
    constexpr int64_t n_patches = 196;
    constexpr int64_t vit_width = 1152;
    const int video_history = hy_vla_video_history_from_env(in.n_images);
    const int n_groups = in.n_images / video_history;
    out_views = n_groups;
    out_tokens.assign((size_t) n_groups * (size_t) n_visual * (size_t) hidden, 0.f);
    std::vector<float> vit_tokens;
    const char * dump_vit_path = std::getenv("VLA_HY_VLA_DUMP_VIT_TOKENS_F32");
    if (dump_vit_path) {
        vit_tokens.assign((size_t) n_groups * (size_t) n_patches * (size_t) vit_width, 0.f);
    }
    for (int gidx = 0; gidx < n_groups; ++gidx) {
        if (!run_hy_vision_frontend_group_224(
                m, vf, vision_backend, vision_layers_loaded,
                in.images + (size_t) gidx * (size_t) video_history,
                video_history,
                out_tokens.data() + (size_t) gidx * (size_t) n_visual * (size_t) hidden,
                dump_vit_path ? vit_tokens.data() + (size_t) gidx * (size_t) n_patches * (size_t) vit_width : nullptr)) {
            return false;
        }
    }
    if (dump_vit_path) {
        write_binary_file(dump_vit_path, vit_tokens.data(), vit_tokens.size() * sizeof(float));
        std::fprintf(stderr, "vla(hy_vla): dumped vit tokens f32 [%d,196,1152] to %s\n",
                     n_groups, dump_vit_path);
    }
    return true;
}

std::vector<float> sinusoidal_time_emb(double timestep, int64_t dim, double min_period, double max_period) {
    const int64_t half = dim / 2;
    std::vector<float> out((size_t) dim);
    for (int64_t i = 0; i < half; ++i) {
        const double frac = (half == 1) ? 0.0 : double(i) / double(half - 1);
        const double period = min_period * std::pow(max_period / min_period, frac);
        const double s = (2.0 * M_PI / period) * timestep;
        out[(size_t) i] = (float) std::sin(s);
        out[(size_t) half + (size_t) i] = (float) std::cos(s);
    }
    return out;
}

double hy_vla_debug_timestep() {
    if (const char * e = std::getenv("VLA_HY_VLA_DEBUG_TIMESTEP")) {
        return std::atof(e);
    }
    return 1.0;
}

ggml_tensor * rope_qk(ggml_context * ctx, ggml_tensor * x, ggml_tensor * positions, const Config & cfg) {
    return ggml_rope_ext(ctx, x, positions, nullptr,
                         cfg.rope_n_dims, cfg.rope_mode, 0,
                         cfg.rope_freq_base, 1.f, 0.f, 1.f, 32.f, 1.f);
}

ggml_tensor * build_suffix_embed(ggml_context * ctx, const HyVLAModelArch & m,
                                 ggml_tensor * state, ggml_tensor * x, ggml_tensor * time_bcast) {
    ggml_tensor * state_emb = ggml_add(ctx, mm_w(ctx, m.W_sp, state), m.b_sp);
    ggml_tensor * action_emb = ggml_add(ctx, mm_w(ctx, m.W_ain, x), m.b_ain);
    ggml_tensor * action_time_in = ggml_concat(ctx, action_emb, time_bcast, 0);
    ggml_tensor * mlp1 = ggml_add(ctx, mm_w(ctx, m.W_at1, action_time_in), m.b_at1);
    ggml_tensor * action_time_emb = ggml_add(ctx, mm_w(ctx, m.W_at2, ggml_silu(ctx, mlp1)), m.b_at2);
    ggml_tensor * state_2d = ggml_reshape_2d(ctx, state_emb, state_emb->ne[0], 1);
    return ggml_concat(ctx, state_2d, action_time_emb, 1);
}

ggml_tensor * build_hy_expert_layer(ggml_context * ctx, const HyLayerW & w,
                                    ggml_tensor * x_in, ggml_tensor * pos,
                                    ggml_tensor * mask, const Config & cfg) {
    const int64_t seq = cfg.n_suffix + 1;
    ggml_tensor * x_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_in, cfg.rms_eps), w.ln_in);
    ggml_tensor * q = mm_w(ctx, w.Wq, x_norm);
    ggml_tensor * k = mm_w(ctx, w.Wk, x_norm);
    ggml_tensor * v = mm_w(ctx, w.Wv, x_norm);
    ggml_tensor * qh = ggml_reshape_3d(ctx, q, cfg.head_dim, cfg.n_q_heads, seq);
    ggml_tensor * kh = ggml_reshape_3d(ctx, k, cfg.head_dim, cfg.n_kv_heads, seq);
    ggml_tensor * vh = ggml_reshape_3d(ctx, v, cfg.head_dim, cfg.n_kv_heads, seq);
    ggml_tensor * qr = ggml_mul(ctx, ggml_rms_norm(ctx, rope_qk(ctx, qh, pos, cfg), cfg.rms_eps), w.q_norm);
    ggml_tensor * kr = ggml_mul(ctx, ggml_rms_norm(ctx, rope_qk(ctx, kh, pos, cfg), cfg.rms_eps), w.k_norm);
    ggml_tensor * Q = ggml_permute(ctx, qr, 0, 2, 1, 3);
    ggml_tensor * K = ggml_permute(ctx, kr, 0, 2, 1, 3);
    ggml_tensor * V = ggml_permute(ctx, vh, 0, 2, 1, 3);
    const float scale = 1.f / std::sqrt((float) cfg.head_dim);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.f, 0.f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_tensor * att_pre = ggml_reshape_2d(ctx, fa, cfg.q_full_dim, seq);
    ggml_tensor * h1 = ggml_add(ctx, x_in, mm_w(ctx, w.Wo, att_pre));
    ggml_tensor * x_mlp = ggml_mul(ctx, ggml_rms_norm(ctx, h1, cfg.rms_eps), w.ln_post);
    ggml_tensor * inter = ggml_mul(ctx, ggml_silu(ctx, mm_w(ctx, w.Wgate, x_mlp)), mm_w(ctx, w.Wup, x_mlp));
    return ggml_add(ctx, h1, mm_w(ctx, w.Wdown, inter));
}

ggml_tensor * build_hy_text_layer(ggml_context * ctx, const HyLayerW & w,
                                  ggml_tensor * x_in, ggml_tensor * pos,
                                  ggml_tensor * mask, const Config & cfg, int64_t seq,
                                  ggml_tensor ** k_out = nullptr,
                                  ggml_tensor ** v_out = nullptr) {
    ggml_tensor * x_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_in, cfg.rms_eps), w.ln_in);
    ggml_tensor * q = mm_w(ctx, w.Wq, x_norm);
    ggml_tensor * k = mm_w(ctx, w.Wk, x_norm);
    ggml_tensor * v = mm_w(ctx, w.Wv, x_norm);
    ggml_tensor * qh = ggml_reshape_3d(ctx, q, cfg.head_dim, cfg.n_q_heads, seq);
    ggml_tensor * kh = ggml_reshape_3d(ctx, k, cfg.head_dim, cfg.n_kv_heads, seq);
    ggml_tensor * vh = ggml_reshape_3d(ctx, v, cfg.head_dim, cfg.n_kv_heads, seq);
    ggml_tensor * qr = ggml_mul(ctx, ggml_rms_norm(ctx, rope_qk(ctx, qh, pos, cfg), cfg.rms_eps), w.q_norm);
    ggml_tensor * kr = ggml_mul(ctx, ggml_rms_norm(ctx, rope_qk(ctx, kh, pos, cfg), cfg.rms_eps), w.k_norm);
    if (k_out) *k_out = kr;
    if (v_out) *v_out = vh;
    ggml_tensor * Q = ggml_permute(ctx, qr, 0, 2, 1, 3);
    ggml_tensor * K = ggml_permute(ctx, kr, 0, 2, 1, 3);
    ggml_tensor * V = ggml_permute(ctx, vh, 0, 2, 1, 3);
    const float scale = 1.f / std::sqrt((float) cfg.head_dim);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.f, 0.f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_tensor * att_pre = ggml_reshape_2d(ctx, fa, cfg.q_full_dim, seq);
    ggml_tensor * h1 = ggml_add(ctx, x_in, mm_w(ctx, w.Wo, att_pre));
    ggml_tensor * x_mlp = ggml_mul(ctx, ggml_rms_norm(ctx, h1, cfg.rms_eps), w.ln_post);
    ggml_tensor * inter = ggml_mul(ctx, ggml_silu(ctx, mm_w(ctx, w.Wgate, x_mlp)), mm_w(ctx, w.Wup, x_mlp));
    return ggml_add(ctx, h1, mm_w(ctx, w.Wdown, inter));
}

ggml_tensor * build_hy_prefix_two_segment_layer(
        ggml_context * ctx, const HyLayerW & wt, const HyLayerW & wv,
        ggml_tensor * x_text, ggml_tensor * x_vis, ggml_tensor * pos,
        ggml_tensor * mask, const Config & cfg, int64_t n_text, int64_t n_vis,
        ggml_tensor ** k_out = nullptr,
        ggml_tensor ** v_out = nullptr) {
    const int64_t seq = n_text + n_vis;
    ggml_tensor * xt_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_text, cfg.rms_eps), wt.ln_in);
    ggml_tensor * xv_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_vis, cfg.rms_eps), wv.ln_in);

    ggml_tensor * qt = mm_w(ctx, wt.Wq, xt_norm);
    ggml_tensor * kt = mm_w(ctx, wt.Wk, xt_norm);
    ggml_tensor * vt = mm_w(ctx, wt.Wv, xt_norm);
    ggml_tensor * qv = mm_w(ctx, wv.Wq, xv_norm);
    ggml_tensor * kv = mm_w(ctx, wv.Wk, xv_norm);
    ggml_tensor * vv = mm_w(ctx, wv.Wv, xv_norm);

    ggml_tensor * q = ggml_concat(ctx, qt, qv, 1);
    ggml_tensor * k = ggml_concat(ctx, kt, kv, 1);
    ggml_tensor * v = ggml_concat(ctx, vt, vv, 1);
    ggml_tensor * qh = ggml_reshape_3d(ctx, q, cfg.head_dim, cfg.n_q_heads, seq);
    ggml_tensor * kh = ggml_reshape_3d(ctx, k, cfg.head_dim, cfg.n_kv_heads, seq);
    ggml_tensor * vh = ggml_reshape_3d(ctx, v, cfg.head_dim, cfg.n_kv_heads, seq);
    ggml_tensor * qr = ggml_mul(ctx, ggml_rms_norm(ctx, rope_qk(ctx, qh, pos, cfg), cfg.rms_eps), wt.q_norm);
    ggml_tensor * kr = ggml_mul(ctx, ggml_rms_norm(ctx, rope_qk(ctx, kh, pos, cfg), cfg.rms_eps), wt.k_norm);
    if (k_out) *k_out = kr;
    if (v_out) *v_out = vh;
    ggml_tensor * Q = ggml_permute(ctx, qr, 0, 2, 1, 3);
    ggml_tensor * K = ggml_permute(ctx, kr, 0, 2, 1, 3);
    ggml_tensor * V = ggml_permute(ctx, vh, 0, 2, 1, 3);
    const float scale = 1.f / std::sqrt((float) cfg.head_dim);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.f, 0.f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_tensor * att_pre = ggml_reshape_2d(ctx, fa, cfg.q_full_dim, seq);

    const size_t rb = (size_t) cfg.q_full_dim * sizeof(float);
    ggml_tensor * att_text = ggml_view_2d(ctx, att_pre, cfg.q_full_dim, n_text, rb, 0);
    ggml_tensor * att_vis = ggml_view_2d(ctx, att_pre, cfg.q_full_dim, n_vis, rb, rb * (size_t) n_text);

    ggml_tensor * ht1 = ggml_add(ctx, x_text, mm_w(ctx, wt.Wo, att_text));
    ggml_tensor * hv1 = ggml_add(ctx, x_vis, mm_w(ctx, wv.Wo, att_vis));

    ggml_tensor * xt_mlp = ggml_mul(ctx, ggml_rms_norm(ctx, ht1, cfg.rms_eps), wt.ln_post);
    ggml_tensor * xv_mlp = ggml_mul(ctx, ggml_rms_norm(ctx, hv1, cfg.rms_eps), wv.ln_post);
    ggml_tensor * it = ggml_mul(ctx, ggml_silu(ctx, mm_w(ctx, wt.Wgate, xt_mlp)), mm_w(ctx, wt.Wup, xt_mlp));
    ggml_tensor * iv = ggml_mul(ctx, ggml_silu(ctx, mm_w(ctx, wv.Wgate, xv_mlp)), mm_w(ctx, wv.Wup, xv_mlp));
    ggml_tensor * out_text = ggml_add(ctx, ht1, mm_w(ctx, wt.Wdown, it));
    ggml_tensor * out_vis = ggml_add(ctx, hv1, mm_w(ctx, wv.Wdown, iv));
    return ggml_concat(ctx, out_text, out_vis, 1);
}

ggml_tensor * concat_seq(ggml_context * ctx, const std::vector<ggml_tensor *> & xs) {
    if (xs.empty()) return nullptr;
    auto rec = [&](auto && self, size_t lo, size_t hi) -> ggml_tensor * {
        if (hi - lo == 1) return xs[lo];
        const size_t mid = lo + (hi - lo) / 2;
        return ggml_concat(ctx, self(self, lo, mid), self(self, mid, hi), 1);
    };
    for (ggml_tensor * x : xs) {
        if (!x) return nullptr;
    }
    return rec(rec, 0, xs.size());
}

ggml_tensor * bf16_roundtrip(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cast(ctx, ggml_cast(ctx, x, GGML_TYPE_BF16), GGML_TYPE_F32);
}

ggml_tensor * build_hy_prefix_routed_layer(
        ggml_context * ctx, const HyLayerW & wt, const HyLayerW & wv,
        ggml_tensor * x_in, ggml_tensor * pos, ggml_tensor * mask,
        const Config & cfg, const std::vector<HyPrefixRun> & runs,
        const std::vector<HyPrefixRange> & visual_segments,
        bool cast_attn_bf16 = false,
        ggml_tensor ** k_out = nullptr,
        ggml_tensor ** v_out = nullptr) {
    const int64_t seq = x_in->ne[1];
    const size_t x_rb = (size_t) cfg.hidden * sizeof(float);
    const size_t q_rb = (size_t) cfg.q_full_dim * sizeof(float);
    std::vector<ggml_tensor *> qs, ks, vs;
    qs.reserve(runs.size());
    ks.reserve(runs.size());
    vs.reserve(runs.size());
    for (const HyPrefixRun & r : runs) {
        const HyLayerW & w = r.vision ? wv : wt;
        ggml_tensor * x = ggml_view_2d(ctx, x_in, cfg.hidden, r.len, x_rb, x_rb * (size_t) r.start);
        ggml_tensor * xn = ggml_mul(ctx, ggml_rms_norm(ctx, x, cfg.rms_eps), w.ln_in);
        qs.push_back(mm_w(ctx, w.Wq, xn));
        ks.push_back(mm_w(ctx, w.Wk, xn));
        vs.push_back(mm_w(ctx, w.Wv, xn));
    }
    ggml_tensor * q = concat_seq(ctx, qs);
    ggml_tensor * k = concat_seq(ctx, ks);
    ggml_tensor * v = concat_seq(ctx, vs);
    ggml_tensor * qh = ggml_reshape_3d(ctx, q, cfg.head_dim, cfg.n_q_heads, seq);
    ggml_tensor * kh = ggml_reshape_3d(ctx, k, cfg.head_dim, cfg.n_kv_heads, seq);
    ggml_tensor * vh = ggml_reshape_3d(ctx, v, cfg.head_dim, cfg.n_kv_heads, seq);
    ggml_tensor * qr = ggml_mul(ctx, ggml_rms_norm(ctx, rope_qk(ctx, qh, pos, cfg), cfg.rms_eps), wt.q_norm);
    ggml_tensor * kr = ggml_mul(ctx, ggml_rms_norm(ctx, rope_qk(ctx, kh, pos, cfg), cfg.rms_eps), wt.k_norm);
    if (k_out) *k_out = kr;
    if (v_out) *v_out = vh;
    ggml_tensor * Q = ggml_permute(ctx, qr, 0, 2, 1, 3);
    ggml_tensor * K = ggml_permute(ctx, kr, 0, 2, 1, 3);
    ggml_tensor * V = ggml_permute(ctx, vh, 0, 2, 1, 3);
    const float scale = 1.f / std::sqrt((float) cfg.head_dim);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.f, 0.f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_tensor * att_pre = ggml_reshape_2d(ctx, fa, cfg.q_full_dim, seq);
    if (!visual_segments.empty()) {
        std::vector<ggml_tensor *> pieces;
        pieces.reserve(visual_segments.size() * 2 + 1);
        int64_t cur = 0;
        for (const HyPrefixRange & seg : visual_segments) {
            const int64_t s = std::max<int64_t>(0, std::min<int64_t>(seq, seg.start));
            const int64_t e = std::max<int64_t>(s, std::min<int64_t>(seq, seg.end));
            if (cur < s) {
                pieces.push_back(ggml_view_2d(ctx, att_pre, cfg.q_full_dim, s - cur,
                                              q_rb, q_rb * (size_t) cur));
            }
            if (e > s) {
                ggml_tensor * Qs = ggml_view_3d(ctx, Q, cfg.head_dim, e - s, cfg.n_q_heads,
                                                Q->nb[1], Q->nb[2], Q->nb[1] * (size_t) s);
                ggml_tensor * Ks = ggml_view_3d(ctx, K, cfg.head_dim, e - s, cfg.n_kv_heads,
                                                K->nb[1], K->nb[2], K->nb[1] * (size_t) s);
                ggml_tensor * Vs = ggml_view_3d(ctx, V, cfg.head_dim, e - s, cfg.n_kv_heads,
                                                V->nb[1], V->nb[2], V->nb[1] * (size_t) s);
                ggml_tensor * vf = ggml_flash_attn_ext(ctx, Qs, Ks, Vs, nullptr, scale, 0.f, 0.f);
                ggml_flash_attn_ext_set_prec(vf, GGML_PREC_F32);
                pieces.push_back(ggml_reshape_2d(ctx, vf, cfg.q_full_dim, e - s));
            }
            cur = e;
        }
        if (cur < seq) {
            pieces.push_back(ggml_view_2d(ctx, att_pre, cfg.q_full_dim, seq - cur,
                                          q_rb, q_rb * (size_t) cur));
        }
        att_pre = concat_seq(ctx, pieces);
    }
    if (cast_attn_bf16) {
        att_pre = bf16_roundtrip(ctx, att_pre);
    }

    std::vector<ggml_tensor *> outs;
    outs.reserve(runs.size());
    for (const HyPrefixRun & r : runs) {
        const HyLayerW & w = r.vision ? wv : wt;
        ggml_tensor * x = ggml_view_2d(ctx, x_in, cfg.hidden, r.len, x_rb, x_rb * (size_t) r.start);
        ggml_tensor * a = ggml_view_2d(ctx, att_pre, cfg.q_full_dim, r.len, q_rb, q_rb * (size_t) r.start);
        ggml_tensor * h1 = ggml_add(ctx, x, mm_w(ctx, w.Wo, a));
        ggml_tensor * xm = ggml_mul(ctx, ggml_rms_norm(ctx, h1, cfg.rms_eps), w.ln_post);
        ggml_tensor * inter = ggml_mul(ctx, ggml_silu(ctx, mm_w(ctx, w.Wgate, xm)), mm_w(ctx, w.Wup, xm));
        outs.push_back(ggml_add(ctx, h1, mm_w(ctx, w.Wdown, inter)));
    }
    return concat_seq(ctx, outs);
}

ggml_tensor * build_hy_expert_layer_with_prefix(
        ggml_context * ctx, const HyLayerW & w,
        ggml_tensor * x_in, ggml_tensor * pos,
        ggml_tensor * prefix_K, ggml_tensor * prefix_V,
        ggml_tensor * mask_full, const Config & cfg,
        bool cast_attn_bf16 = false) {
    const int64_t seq = cfg.n_suffix + 1;
    ggml_tensor * x_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_in, cfg.rms_eps), w.ln_in);
    ggml_tensor * q = mm_w(ctx, w.Wq, x_norm);
    ggml_tensor * k = mm_w(ctx, w.Wk, x_norm);
    ggml_tensor * v = mm_w(ctx, w.Wv, x_norm);
    ggml_tensor * qh = ggml_reshape_3d(ctx, q, cfg.head_dim, cfg.n_q_heads, seq);
    ggml_tensor * kh = ggml_reshape_3d(ctx, k, cfg.head_dim, cfg.n_kv_heads, seq);
    ggml_tensor * vh = ggml_reshape_3d(ctx, v, cfg.head_dim, cfg.n_kv_heads, seq);
    ggml_tensor * qr = ggml_mul(ctx, ggml_rms_norm(ctx, rope_qk(ctx, qh, pos, cfg), cfg.rms_eps), w.q_norm);
    ggml_tensor * kr = ggml_mul(ctx, ggml_rms_norm(ctx, rope_qk(ctx, kh, pos, cfg), cfg.rms_eps), w.k_norm);
    ggml_tensor * K_full = ggml_concat(ctx, prefix_K, kr, 2);
    ggml_tensor * V_full = ggml_concat(ctx, prefix_V, vh, 2);
    ggml_tensor * Q = ggml_permute(ctx, qr, 0, 2, 1, 3);
    ggml_tensor * K = ggml_permute(ctx, K_full, 0, 2, 1, 3);
    ggml_tensor * V = ggml_permute(ctx, V_full, 0, 2, 1, 3);
    const float scale = 1.f / std::sqrt((float) cfg.head_dim);
    ggml_tensor * fa = ggml_flash_attn_ext(ctx, Q, K, V, mask_full, scale, 0.f, 0.f);
    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
    ggml_tensor * att_pre = ggml_reshape_2d(ctx, fa, cfg.q_full_dim, seq);
    if (cast_attn_bf16) {
        att_pre = bf16_roundtrip(ctx, att_pre);
    }
    ggml_tensor * h1 = ggml_add(ctx, x_in, mm_w(ctx, w.Wo, att_pre));
    ggml_tensor * x_mlp = ggml_mul(ctx, ggml_rms_norm(ctx, h1, cfg.rms_eps), w.ln_post);
    ggml_tensor * inter = ggml_mul(ctx, ggml_silu(ctx, mm_w(ctx, w.Wgate, x_mlp)), mm_w(ctx, w.Wup, x_mlp));
    return ggml_add(ctx, h1, mm_w(ctx, w.Wdown, inter));
}

bool load_config(const gguf_reader & g, Config & cfg, bool vlm_only) {
    for (const char * k : {"hy_vla.architecture",
                           "hy_vla.n_q_heads", "hy_vla.n_kv_heads", "hy_vla.head_dim",
                           "hy_vla.n_layers",
                           "hy_vla.rms_norm_eps", "hy_vla.rope_theta"}) {
        if (!g.has_key(k)) {
            std::fprintf(stderr, "vla(hy_vla): GGUF missing key %s\n", k);
            return false;
        }
    }
    if (!vlm_only) {
        for (const char * k : {"hy_vla.expert_hidden", "hy_vla.expert_intermediate",
                               "hy_vla.n_action_steps", "hy_vla.num_steps",
                               "hy_vla.max_state_dim", "hy_vla.max_action_dim",
                               "hy_vla.flow_min_period", "hy_vla.flow_max_period"}) {
            if (!g.has_key(k)) {
                std::fprintf(stderr, "vla(hy_vla): GGUF missing key %s\n", k);
                return false;
            }
        }
    }
    cfg = Config{};
    cfg.hidden = g.u32("hy_vla.vlm_hidden");
    cfg.intermediate = g.u32("hy_vla.vlm_intermediate");
    cfg.expert_h = vlm_only ? 0 : g.u32("hy_vla.expert_hidden");
    cfg.expert_inter = vlm_only ? 0 : g.u32("hy_vla.expert_intermediate");
    cfg.n_q_heads = g.u32("hy_vla.n_q_heads");
    cfg.n_kv_heads = g.u32("hy_vla.n_kv_heads");
    cfg.head_dim = g.u32("hy_vla.head_dim");
    cfg.q_full_dim = cfg.n_q_heads * cfg.head_dim;
    cfg.kv_full_dim = cfg.n_kv_heads * cfg.head_dim;
    cfg.n_layers = g.u32("hy_vla.n_layers");
    cfg.n_lang = vlm_only ? 0 : g.u32("hy_vla.tokenizer_max_length");
    cfg.n_img = 0;
    cfg.n_suffix = vlm_only ? 0 : g.u32("hy_vla.n_action_steps");
    cfg.num_steps = vlm_only ? 1 : (int) g.u32("hy_vla.num_steps");
    cfg.max_state_dim = vlm_only ? 0 : g.u32("hy_vla.max_state_dim");
    cfg.max_action_dim = vlm_only ? 0 : g.u32("hy_vla.max_action_dim");
    cfg.real_state_dim = 20;
    cfg.real_action_dim = 20;
    cfg.n_state = 1;
    cfg.n_full = cfg.n_suffix + 1;
    cfg.min_period = vlm_only ? 1.0 : g.f32("hy_vla.flow_min_period");
    cfg.max_period = vlm_only ? 1.0 : g.f32("hy_vla.flow_max_period");
    cfg.rms_eps = g.f32("hy_vla.rms_norm_eps");
    cfg.norm_eps = 1e-8f;
    cfg.rope_mode = GGML_ROPE_TYPE_NEOX;
    cfg.rope_n_dims = (int) cfg.head_dim;
    {
        const float theta = g.f32("hy_vla.rope_theta");
        const float alpha = g.has_key("hy_vla.rope_dynamic_alpha") ? g.f32("hy_vla.rope_dynamic_alpha") : 1000.0f;
        cfg.rope_freq_base = theta * std::pow(alpha, (float) cfg.head_dim / (float) (cfg.head_dim - 2));
    }
    return true;
}

bool load_stats(gguf_reader & g, HyVLAModelArch & m) {
    auto read = [&](const char * name, std::vector<float> & dst, int64_t n, float fill) {
        dst.assign((size_t) n, fill);
        std::vector<float> tmp;
        if (!g.read_to_f32(name, tmp)) return;
        for (size_t i = 0; i < dst.size() && i < tmp.size(); ++i) dst[i] = tmp[i];
    };
    auto read_action_stat = [&](const char * name, std::vector<float> & dst, float fill) {
        const int64_t steps = m.cfg.n_suffix;
        const int64_t max_ad = m.cfg.max_action_dim;
        const int64_t real_ad = m.cfg.real_action_dim;
        dst.assign((size_t) steps * (size_t) max_ad, fill);
        const ggml_tensor * gt = g.meta(name);
        std::vector<float> tmp;
        if (!gt || !g.read_to_f32(name, tmp)) return;
        const int64_t src_cols = gt->ne[0];
        const int64_t src_rows = ggml_nelements(gt) / std::max<int64_t>(1, src_cols);
        const int64_t copy_rows = std::min<int64_t>(steps, src_rows);
        const int64_t copy_cols = std::min<int64_t>(std::min(max_ad, real_ad), src_cols);
        for (int64_t r = 0; r < copy_rows; ++r) {
            for (int64_t c = 0; c < copy_cols; ++c) {
                dst[(size_t) r * (size_t) max_ad + (size_t) c] =
                    tmp[(size_t) r * (size_t) src_cols + (size_t) c];
            }
        }
    };
    read("norm.state_mean", m.state_mean, m.cfg.max_state_dim, 0.f);
    read("norm.state_std", m.state_std, m.cfg.max_state_dim, 1.f);
    read_action_stat("norm.action_mean", m.action_mean, 0.f);
    read_action_stat("norm.action_std", m.action_std, 1.f);
    read_action_stat("norm.action_mean_abs", m.action_mean_abs, 0.f);
    read_action_stat("norm.action_std_abs", m.action_std_abs, 1.f);
    return true;
}

std::vector<HyPrefixRun> make_prefix_runs(const int32_t * mask, int n) {
    std::vector<HyPrefixRun> runs;
    if (!mask || n <= 0) return runs;
    int start = 0;
    bool vision = mask[0] != 0;
    for (int i = 1; i <= n; ++i) {
        const bool end = (i == n) || ((mask[i] != 0) != vision);
        if (end) {
            runs.push_back(HyPrefixRun{start, i - start, vision});
            if (i < n) {
                start = i;
                vision = mask[i] != 0;
            }
        }
    }
    return runs;
}

static inline void denorm_hy_vla_action_row(HyVLAModelArch & m, float * row, int64_t t) {
    const int64_t max_ad = m.cfg.max_action_dim;
    const int64_t half = m.cfg.n_suffix / 2;
    const bool use_abs_stats = (m.cfg.n_suffix % 2 == 0) && (t >= half);
    const int64_t stat_t = use_abs_stats ? (t - half) : t;
    const std::vector<float> & mean = use_abs_stats ? m.action_mean_abs : m.action_mean;
    const std::vector<float> & stdv = use_abs_stats ? m.action_std_abs : m.action_std;
    for (int64_t j = 0; j < m.cfg.real_action_dim && j < max_ad; ++j) {
        const size_t stat_i = (size_t) stat_t * (size_t) max_ad + (size_t) j;
        row[j] = row[j] * (stdv[stat_i] + m.cfg.norm_eps) + mean[stat_i];
    }
}

std::vector<HyPrefixRange> make_visual_override_segments(const int32_t * mask, int n) {
    std::vector<HyPrefixRange> separators;
    std::vector<HyPrefixRange> segments;
    if (!mask || n <= 1) return segments;

    int i = 0;
    while (i < n) {
        if (mask[i] != 0) {
            ++i;
            continue;
        }
        const int start = i;
        while (i < n && mask[i] == 0) ++i;
        const int end = i;
        if (end - start >= 2) {
            separators.push_back(HyPrefixRange{start, end});
        }
    }

    int64_t seg_start = 0;
    for (const HyPrefixRange & sep : separators) {
        const int64_t seg_end_inclusive = sep.start - 1;
        if (seg_end_inclusive >= seg_start) {
            segments.push_back(HyPrefixRange{
                seg_start,
                std::min<int64_t>(n, seg_end_inclusive + 2),
            });
        }
        seg_start = sep.end;
    }
    if (seg_start < n) {
        segments.push_back(HyPrefixRange{seg_start, n});
    }

    std::vector<HyPrefixRange> out;
    out.reserve(segments.size());
    for (const HyPrefixRange & r : segments) {
        if (r.end > r.start) out.push_back(r);
    }
    return out;
}

std::vector<std::vector<int64_t>> make_visual_patch_groups(const int32_t * mask, int n) {
    std::vector<std::vector<int64_t>> groups;
    if (!mask || n <= 0) return groups;
    std::vector<int64_t> cur;
    int i = 0;
    int zero_run = 2;
    while (i < n) {
        if (mask[i] == 0) {
            const int z0 = i;
            while (i < n && mask[i] == 0) ++i;
            zero_run = i - z0;
            if (zero_run >= 2 && !cur.empty()) {
                groups.push_back(std::move(cur));
                cur = {};
            }
            continue;
        }
        while (i < n && mask[i] != 0) {
            cur.push_back(i);
            ++i;
        }
        zero_run = 0;
    }
    (void) zero_run;
    if (!cur.empty()) groups.push_back(std::move(cur));
    return groups;
}

std::vector<ggml_fp16_t> make_prefix_mask(int64_t n, const std::vector<std::vector<int64_t>> * patch_groups) {
    std::vector<ggml_fp16_t> mask((size_t) n * (size_t) n);
    for (int64_t r = 0; r < n; ++r) {
        for (int64_t c = 0; c < n; ++c) {
            mask[(size_t) r * (size_t) n + (size_t) c] =
                ggml_fp32_to_fp16((c <= r) ? 0.f : -INFINITY);
        }
    }
    if (patch_groups) {
        std::vector<int64_t> all;
        for (const auto & g : *patch_groups) {
            all.insert(all.end(), g.begin(), g.end());
        }
        for (int64_t r : all) {
            for (int64_t c : all) {
                mask[(size_t) r * (size_t) n + (size_t) c] = ggml_fp32_to_fp16(-INFINITY);
            }
        }
        for (const auto & g : *patch_groups) {
            for (int64_t r : g) {
                for (int64_t c : g) {
                    mask[(size_t) r * (size_t) n + (size_t) c] = ggml_fp32_to_fp16(0.f);
                }
            }
        }
    }
    return mask;
}

// The vocab matrix doubles as the input embedding: the VLA GGUF materializes
// it as lm_head, a VLM-only GGUF keeps the tied embed_tokens tensor instead.
const char * hy_vla_embedding_name(const gguf_reader & g) {
    return g.first_existing("dual_tower.vlm.model.language_model.lm_head.weight",
                            "language_model.model.embed_tokens.weight");
}

bool append_token_embeddings(gguf_reader & g, const std::vector<int32_t> & ids,
                             int64_t hidden, std::vector<float> & out) {
    if (ids.empty()) return true;
    const size_t old = out.size();
    out.resize(old + (size_t) ids.size() * (size_t) hidden);
    return g.fetch_rows_f32(hy_vla_embedding_name(g), ids, out.data() + old, hidden);
}

bool write_binary_file(const char * path, const void * data, size_t bytes) {
    FILE * fp = std::fopen(path, "wb");
    if (!fp) {
        std::fprintf(stderr, "vla(hy_vla): failed to open dump file %s\n", path);
        return false;
    }
    const bool ok = std::fwrite(data, 1, bytes, fp) == bytes;
    std::fclose(fp);
    if (!ok) {
        std::fprintf(stderr, "vla(hy_vla): failed to write dump file %s\n", path);
    }
    return ok;
}

bool build_hy_vla_prefix_from_visual_tokens(
        const HyVLAModelArch & m,
        const Inputs & in,
        int visual_tokens_per_image,
        std::vector<float> & prefix,
        std::vector<int32_t> & modality_mask) {
    if (!in.precomputed_img_emb || in.n_img_views < 1 || !in.lang_tokens || in.n_lang < 1) {
        std::fprintf(stderr, "vla(hy_vla): prefix builder requires precomputed visual tokens and lang_tokens\n");
        return false;
    }
    if (visual_tokens_per_image <= 0 || in.n_img_views % visual_tokens_per_image != 0) {
        std::fprintf(stderr,
                     "vla(hy_vla): visual token count %d is not divisible by visual_tokens_per_image=%d\n",
                     in.n_img_views, visual_tokens_per_image);
        return false;
    }
    const int grid = (int) std::lround(std::sqrt((double) visual_tokens_per_image));
    if (grid * grid != visual_tokens_per_image) {
        std::fprintf(stderr, "vla(hy_vla): visual_tokens_per_image=%d is not a square grid\n",
                     visual_tokens_per_image);
        return false;
    }

    constexpr int32_t HY_BOS_ID          = 120000;
    constexpr int32_t HY_USER_ID         = 120006;
    constexpr int32_t HY_VISION_START_ID = 120684; // placeholder no 666
    constexpr int32_t HY_VISION_END_ID   = 120685; // placeholder no 667
    constexpr int32_t HY_VISION_SPLIT_ID = 120689; // placeholder no 671

    gguf_reader g;
    if (!g.open(m.ckpt_path)) return false;
    prefix.clear();
    modality_mask.clear();
    const int n_images = in.n_img_views / visual_tokens_per_image;
    const int64_t hidden = m.cfg.hidden;
    const size_t expected_tokens =
        2u + (size_t) n_images * (size_t) (1 + grid * (grid + 1) + 1) + (size_t) in.n_lang;
    prefix.reserve(expected_tokens * (size_t) hidden);
    modality_mask.reserve(expected_tokens);

    auto add_ids = [&](std::initializer_list<int32_t> ids) -> bool {
        const size_t n0 = modality_mask.size();
        if (!append_token_embeddings(g, std::vector<int32_t>(ids), hidden, prefix)) return false;
        modality_mask.resize(n0 + ids.size(), 0);
        return true;
    };
    auto add_lang_ids = [&]() -> bool {
        std::vector<int32_t> ids(in.lang_tokens, in.lang_tokens + in.n_lang);
        const size_t n0 = modality_mask.size();
        if (!append_token_embeddings(g, ids, hidden, prefix)) return false;
        modality_mask.resize(n0 + ids.size(), 0);
        return true;
    };
    auto add_visual_patch = [&](int image_idx, int row, int col) {
        const int patch_idx = image_idx * visual_tokens_per_image + row * grid + col;
        const float * src = in.precomputed_img_emb + (size_t) patch_idx * (size_t) hidden;
        prefix.insert(prefix.end(), src, src + hidden);
        modality_mask.push_back(1);
    };

    if (!add_ids({HY_BOS_ID, HY_USER_ID})) return false;
    for (int img = 0; img < n_images; ++img) {
        if (!add_ids({HY_VISION_START_ID})) return false;
        for (int r = 0; r < grid; ++r) {
            for (int c = 0; c < grid; ++c) {
                add_visual_patch(img, r, c);
            }
            if (!add_ids({HY_VISION_SPLIT_ID})) return false;
        }
        if (!add_ids({HY_VISION_END_ID})) return false;
    }
    if (!add_lang_ids()) return false;

    if (prefix.size() != modality_mask.size() * (size_t) hidden) {
        std::fprintf(stderr, "vla(hy_vla): internal prefix size mismatch\n");
        return false;
    }
    return true;
}

bool ends_with(const std::string & s, const char * suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && std::strcmp(s.c_str() + s.size() - n, suf) == 0;
}

bool env_enabled(const char * name) {
    const char * v = std::getenv(name);
    return v && std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "FALSE") != 0;
}

int parse_layer_count_env(const char * value, int max_layers) {
    if (!value) return -1;
    if (std::strcmp(value, "all") == 0 || std::strcmp(value, "ALL") == 0) return max_layers;
    return std::max(0, std::min(max_layers, std::atoi(value)));
}

std::vector<float> resize_abs_pos_embed_2d(const std::vector<float> & src,
                                           int src_hw, int dst_hw, int width) {
    std::vector<float> out((size_t) dst_hw * (size_t) dst_hw * (size_t) width, 0.f);
    const float scale = (float) src_hw / (float) dst_hw;
    for (int oy = 0; oy < dst_hw; ++oy) {
        const float fy = ((float) oy + 0.5f) * scale - 0.5f;
        const int y0 = std::max(0, std::min(src_hw - 1, (int) std::floor(fy)));
        const int y1 = std::max(0, std::min(src_hw - 1, y0 + 1));
        const float wy = fy - std::floor(fy);
        for (int ox = 0; ox < dst_hw; ++ox) {
            const float fx = ((float) ox + 0.5f) * scale - 0.5f;
            const int x0 = std::max(0, std::min(src_hw - 1, (int) std::floor(fx)));
            const int x1 = std::max(0, std::min(src_hw - 1, x0 + 1));
            const float wx = fx - std::floor(fx);
            const size_t o_base = ((size_t) oy * (size_t) dst_hw + (size_t) ox) * (size_t) width;
            const size_t s00 = ((size_t) y0 * (size_t) src_hw + (size_t) x0) * (size_t) width;
            const size_t s01 = ((size_t) y0 * (size_t) src_hw + (size_t) x1) * (size_t) width;
            const size_t s10 = ((size_t) y1 * (size_t) src_hw + (size_t) x0) * (size_t) width;
            const size_t s11 = ((size_t) y1 * (size_t) src_hw + (size_t) x1) * (size_t) width;
            for (int c = 0; c < width; ++c) {
                const float top = src[s00 + (size_t) c] * (1.f - wx) + src[s01 + (size_t) c] * wx;
                const float bot = src[s10 + (size_t) c] * (1.f - wx) + src[s11 + (size_t) c] * wx;
                out[o_base + (size_t) c] = top * (1.f - wy) + bot * wy;
            }
        }
    }
    return out;
}

} // namespace

std::unique_ptr<ModelArchBase> hy_vla_create(const std::string & mmproj_path,
                                             const std::string & ckpt_path,
                                             const std::string & config_path) {
    (void) mmproj_path;
    (void) config_path;
    if (!ends_with(ckpt_path, ".gguf")) {
        std::fprintf(stderr, "vla(hy_vla): ckpt must be a GGUF\n");
        return nullptr;
    }
    gguf_reader g;
    if (!g.open(ckpt_path)) return nullptr;
    if (!g.has_key("hy_vla.architecture") || g.str("hy_vla.architecture") != "hy_vla") {
        std::fprintf(stderr, "vla(hy_vla): architecture key missing/wrong\n");
        return nullptr;
    }

    auto m = std::make_unique<HyVLAModelArch>();
    m->ckpt_path = ckpt_path;
    m->matmul_type = hy_vla_matmul_type_from_env();
    // A VLM-only GGUF (tied-embedding language/vision towers, no action
    // expert) carries none of the expert/action keys or tensors.
    m->vlm_only = !g.has_key("hy_vla.expert_hidden");
    if (!load_config(g, m->cfg, m->vlm_only)) return nullptr;
    if (g.has_key("hy_vla.eos_token_id")) {
        m->eos_token_id = (int32_t) g.u32("hy_vla.eos_token_id");
    }

#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) {
        m->is_cuda = true;
        std::printf("vla(hy_vla): backend = CUDA (device 0)\n");
    }
#endif
    const unsigned hw = std::thread::hardware_concurrency();
    m->n_threads = (hw == 0) ? 4 : (int) std::min(hw, 8u);
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) return nullptr;
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(hy_vla): backend = CPU (%d threads)\n", m->n_threads);
    }

    ggml_init_params wp{(size_t) 16 * 1024 * 1024, nullptr, true};
    m->ctx_weights = ggml_init(wp);
    if (!m->ctx_weights) return nullptr;
    std::vector<ggml_tensor *> weights;
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> generated_weights;
    std::vector<ggml_tensor *> vision_side_weights;
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> vision_side_generated_weights;

    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) {
            std::fprintf(stderr, "vla(hy_vla): missing tensor %s\n", name);
            return nullptr;
        }
        ggml_tensor * t = ggml_new_tensor(m->ctx_weights, type, GGML_MAX_DIMS, gt->ne);
        ggml_set_name(t, name);
        weights.push_back(t);
        return t;
    };
    auto mk_mm = [&](const char * name) {
        const ggml_type src_type = g.tensor_type(name);
        if (ggml_is_quantized(src_type)) {
            return mk(name, src_type);
        }
        return mk(name, m->matmul_type);
    };
    auto mk_f32 = [&](const char * name) { return mk(name, GGML_TYPE_F32); };
    auto mk_generated_f32 = [&](const char * name, int64_t ne0, int64_t ne1,
                                const std::vector<float> & data) -> ggml_tensor * {
        int64_t ne[GGML_MAX_DIMS] = {ne0, ne1, 1, 1};
        ggml_tensor * t = ggml_new_tensor(m->ctx_weights, GGML_TYPE_F32, GGML_MAX_DIMS, ne);
        ggml_set_name(t, name);
        std::vector<uint8_t> bytes(data.size() * sizeof(float));
        std::memcpy(bytes.data(), data.data(), bytes.size());
        generated_weights.emplace_back(t, std::move(bytes));
        return t;
    };

    if (!m->vlm_only) {
        m->layers.resize((size_t) m->cfg.n_layers);
        for (int64_t i = 0; i < m->cfg.n_layers; ++i) {
            char b[256];
            auto nm = [&](const char * s) {
                std::snprintf(b, sizeof(b), "expert.blk.%lld.%s", (long long) i, s);
                return b;
            };
            HyLayerW & w = m->layers[(size_t) i];
            w.ln_in = mk_f32(nm("attn_norm_v.weight"));
            w.Wq = mk_mm(nm("attn_q_v.weight"));
            w.Wk = mk_mm(nm("attn_k_v.weight"));
            w.Wv = mk_mm(nm("attn_v_v.weight"));
            w.Wo = mk_mm(nm("attn_o_v.weight"));
            {
                char qn[256];
                char kn[256];
                char qn_old[256];
                char kn_old[256];
                std::snprintf(qn, sizeof(qn), "vlm.blk.%lld.attn_q_norm.weight", (long long) i);
                std::snprintf(kn, sizeof(kn), "vlm.blk.%lld.attn_k_norm.weight", (long long) i);
                std::snprintf(qn_old, sizeof(qn_old), "dt.vlm.model.lm.model.layers.%lld.attn.query_layernorm.weight", (long long) i);
                std::snprintf(kn_old, sizeof(kn_old), "dt.vlm.model.lm.model.layers.%lld.attn.key_layernorm.weight", (long long) i);
                const char * q_name = g.first_existing(qn, qn_old);
                const char * k_name = g.first_existing(kn, kn_old);
                w.q_norm = mk_f32(q_name);
                w.k_norm = mk_f32(k_name);
            }
            w.ln_post = mk_f32(nm("ffn_norm_v.weight"));
            w.Wgate = mk_mm(nm("ffn_gate_v.weight"));
            w.Wup = mk_mm(nm("ffn_up_v.weight"));
            w.Wdown = mk_mm(nm("ffn_down_v.weight"));
        }
        m->Wnorm = mk_f32("expert.output_norm.weight");
        m->W_sp = mk_f32("state_proj.weight");              m->b_sp = mk_f32("state_proj.bias");
        m->W_ain = mk_mm("action_in_proj.weight");          m->b_ain = mk_f32("action_in_proj.bias");
        m->W_at1 = mk_mm("action_time_mlp_in.weight");      m->b_at1 = mk_f32("action_time_mlp_in.bias");
        m->W_at2 = mk_mm("action_time_mlp_out.weight");     m->b_at2 = mk_f32("action_time_mlp_out.bias");
        m->W_aout = mk_f32("action_out_proj.weight");       m->b_aout = mk_f32("action_out_proj.bias");
    }

    if (const char * e = std::getenv("VLA_HY_VLA_TEXT_LAYERS")) {
        m->text_layers_loaded = std::max(0, std::min<int>((int) m->cfg.n_layers, std::atoi(e)));
    }
    if (m->text_layers_loaded > 0) {
        m->text_layers.resize((size_t) m->text_layers_loaded);
        m->prefix_vision_loaded = g.meta("vlm.blk.0.attn_norm_v.weight") != nullptr;
        if (m->prefix_vision_loaded) {
            m->vision_layers.resize((size_t) m->text_layers_loaded);
        }
        for (int i = 0; i < m->text_layers_loaded; ++i) {
            char b_new[256];
            char b_old[256];
            auto nm = [&](const char * s_new, const char * s_old) {
                std::snprintf(b_new, sizeof(b_new), "vlm.blk.%d.%s", i, s_new);
                std::snprintf(b_old, sizeof(b_old), "dt.vlm.model.lm.model.layers.%d.%s", i, s_old);
                return g.first_existing(b_new, b_old);
            };
            HyLayerW & w = m->text_layers[(size_t) i];
            w.ln_in = mk_f32(nm("attn_norm.weight", "input_layernorm.weight"));
            w.Wq = mk_mm(nm("attn_q.weight", "attn.q_proj.weight"));
            w.Wk = mk_mm(nm("attn_k.weight", "attn.k_proj.weight"));
            w.Wv = mk_mm(nm("attn_v.weight", "attn.v_proj.weight"));
            w.Wo = mk_mm(nm("attn_o.weight", "attn.o_proj.weight"));
            w.q_norm = mk_f32(nm("attn_q_norm.weight", "attn.query_layernorm.weight"));
            w.k_norm = mk_f32(nm("attn_k_norm.weight", "attn.key_layernorm.weight"));
            w.ln_post = mk_f32(nm("ffn_norm.weight", "post_ln.weight"));
            w.Wgate = mk_mm(nm("ffn_gate.weight", "mlp.gate_proj.weight"));
            w.Wup = mk_mm(nm("ffn_up.weight", "mlp.up_proj.weight"));
            w.Wdown = mk_mm(nm("ffn_down.weight", "mlp.down_proj.weight"));
            if (m->prefix_vision_loaded) {
                auto nv = [&](const char * s) {
                    std::snprintf(b_new, sizeof(b_new), "vlm.blk.%d.%s", i, s);
                    return b_new;
                };
                HyLayerW & v = m->vision_layers[(size_t) i];
                v.ln_in = mk_f32(nv("attn_norm_v.weight"));
                v.Wq = mk_mm(nv("attn_q_v.weight"));
                v.Wk = mk_mm(nv("attn_k_v.weight"));
                v.Wv = mk_mm(nv("attn_v_v.weight"));
                v.Wo = mk_mm(nv("attn_o_v.weight"));
                v.q_norm = w.q_norm;
                v.k_norm = w.k_norm;
                v.ln_post = mk_f32(nv("ffn_norm_v.weight"));
                v.Wgate = mk_mm(nv("ffn_gate_v.weight"));
                v.Wup = mk_mm(nv("ffn_up_v.weight"));
                v.Wdown = mk_mm(nv("ffn_down_v.weight"));
            }
        }
        m->Wtext_norm = mk_f32(g.first_existing("vlm.output_norm.weight",
                                                "dual_tower.vlm.model.language_model.model.norm.weight"));
    }
    // Text generation multiplies hidden states by the tied vocab matrix.
    // Only make it resident when the generate mode is requested so action
    // inference does not pay for it.
    if (const char * e = std::getenv("VLA_HY_VLA_DEBUG_OUTPUT")) {
        if (std::strcmp(e, "generate") == 0) {
            m->W_vocab = mk_mm(hy_vla_embedding_name(g));
        }
    }

    if (const char * e = std::getenv("VLA_HY_VLA_VISION_LAYERS")) {
        const int max_vision_layers = 27;
        const int requested_layers = parse_layer_count_env(e, max_vision_layers);
        m->vision_cpu_sideload = env_enabled("VLA_HY_VLA_VISION_CPU_SIDELOAD");
        ggml_context * vision_weight_ctx = m->ctx_weights;
        HyVisionW * vision_frontend = &m->vision_frontend;
        if (m->vision_cpu_sideload) {
            m->vision_backend = ggml_backend_cpu_init();
            if (!m->vision_backend) return nullptr;
            ggml_backend_cpu_set_n_threads(m->vision_backend, m->n_threads);
            ggml_init_params vwp{(size_t) 16 * 1024 * 1024, nullptr, true};
            m->ctx_vision_weights = ggml_init(vwp);
            if (!m->ctx_vision_weights) return nullptr;
            vision_weight_ctx = m->ctx_vision_weights;
            vision_frontend = &m->vision_cpu_frontend;
        }
        bool vision_ok = true;
        auto mk_required = [&](const char * name, ggml_type type) -> ggml_tensor * {
            ggml_tensor * t = nullptr;
            if (m->vision_cpu_sideload) {
                const ggml_tensor * gt = g.meta(name);
                if (!gt) {
                    std::fprintf(stderr, "vla(hy_vla): missing tensor %s\n", name);
                    vision_ok = false;
                    return nullptr;
                }
                t = ggml_new_tensor(vision_weight_ctx, type, GGML_MAX_DIMS, gt->ne);
                ggml_set_name(t, name);
                vision_side_weights.push_back(t);
            } else {
                t = mk(name, type);
            }
            if (!t) vision_ok = false;
            return t;
        };
        auto mk_vmm = [&](const char * name) {
            const ggml_type src_type = g.tensor_type(name);
            if (ggml_is_quantized(src_type)) {
                return mk_required(name, src_type);
            }
            return mk_required(name, m->matmul_type);
        };
        auto mk_vf32 = [&](const char * name) { return mk_required(name, GGML_TYPE_F32); };
        auto mk_vision_generated_f32 = [&](const char * name, int64_t ne0, int64_t ne1,
                                           const std::vector<float> & data) -> ggml_tensor * {
            if (!m->vision_cpu_sideload) {
                return mk_generated_f32(name, ne0, ne1, data);
            }
            int64_t ne[GGML_MAX_DIMS] = {ne0, ne1, 1, 1};
            ggml_tensor * t = ggml_new_tensor(vision_weight_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, ne);
            ggml_set_name(t, name);
            std::vector<uint8_t> bytes(data.size() * sizeof(float));
            std::memcpy(bytes.data(), data.data(), bytes.size());
            vision_side_generated_weights.emplace_back(t, std::move(bytes));
            return t;
        };

        vision_frontend->patch_w = mk_vf32("vision.patch_embed.weight");
        vision_frontend->patch_b = mk_vf32("vision.patch_embed.bias");
        vision_frontend->pos_embed = mk_vf32("vision.pos_embed");
        {
            std::vector<float> pos_full;
            if (!g.read_to_f32("vision.pos_embed", pos_full)) {
                vision_ok = false;
            } else {
                const int pos_width = 1152;
                const int pos_src_hw = 128;
                const int pos_dst_hw = 14;
                const size_t expected = (size_t) pos_src_hw * (size_t) pos_src_hw * (size_t) pos_width;
                if (pos_full.size() != expected) {
                    std::fprintf(stderr, "vla(hy_vla): vision.pos_embed size %zu != expected %zu\n",
                                 pos_full.size(), expected);
                    vision_ok = false;
                } else {
                    vision_frontend->pos_embed_14 = mk_vision_generated_f32(
                        "vision.pos_embed_14x14.f32", pos_width, pos_dst_hw * pos_dst_hw,
                        resize_abs_pos_embed_2d(pos_full, pos_src_hw, pos_dst_hw, pos_width));
                }
            }
        }
        vision_frontend->merger_proj1_w = mk_vmm("vision.merger.proj1.weight");
        vision_frontend->merger_proj1_b = mk_vf32("vision.merger.proj1.bias");
        vision_frontend->merger_proj2_w = mk_vmm("vision.merger.proj2.weight");
        vision_frontend->merger_proj2_b = mk_vf32("vision.merger.proj2.bias");
        vision_frontend->pooler_fc0_w = mk_vmm("vision.merger.pooler.fc0.weight");
        vision_frontend->pooler_fc0_b = mk_vf32("vision.merger.pooler.fc0.bias");
        vision_frontend->pooler_fc2_w = mk_vmm("vision.merger.pooler.fc2.weight");
        vision_frontend->pooler_fc2_b = mk_vf32("vision.merger.pooler.fc2.bias");
        vision_frontend->base_loaded = vision_ok;

        if (vision_ok && requested_layers > 0) {
            vision_frontend->blocks.resize((size_t) requested_layers);
            for (int i = 0; i < requested_layers; ++i) {
                char b[256];
                auto vn = [&](const char * s) {
                    std::snprintf(b, sizeof(b), "vision.blk.%d.%s", i, s);
                    return b;
                };
                HyVisionBlockW & w = vision_frontend->blocks[(size_t) i];
                w.norm1_w = mk_vf32(vn("norm1.weight"));
                w.norm1_b = mk_vf32(vn("norm1.bias"));
                w.qkv_w = mk_vmm(vn("attn_qkv.weight"));
                w.qkv_b = mk_vf32(vn("attn_qkv.bias"));
                w.proj_w = mk_vmm(vn("attn_proj.weight"));
                w.proj_b = mk_vf32(vn("attn_proj.bias"));
                w.norm2_w = mk_vf32(vn("norm2.weight"));
                w.norm2_b = mk_vf32(vn("norm2.bias"));
                w.fc1_w = mk_vmm(vn("ffn_fc1.weight"));
                w.fc1_b = mk_vf32(vn("ffn_fc1.bias"));
                w.fc2_w = mk_vmm(vn("ffn_fc2.weight"));
                w.fc2_b = mk_vf32(vn("ffn_fc2.bias"));
            }
        }
        if (!vision_ok) {
            std::fprintf(stderr,
                         "vla(hy_vla): VLA_HY_VLA_VISION_LAYERS requested, but stable vision frontend tensors are missing; "
                         "use a GGUF converted with stable vision names such as hy_vla_full_bf16_vlmvisionstable.gguf\n");
            return nullptr;
        }
        vision_frontend->layers_loaded = requested_layers;
        m->vision_layers_loaded = requested_layers;
        m->vision_frontend_loaded = true;
    }
    for (ggml_tensor * t : weights) if (!t) return nullptr;
    for (ggml_tensor * t : vision_side_weights) if (!t) return nullptr;

    if (m->vision_cpu_sideload) {
        m->vision_weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_vision_weights, m->vision_backend);
        if (!m->vision_weight_buf) {
            std::fprintf(stderr, "vla(hy_vla): failed to allocate CPU vision sidecar weights\n");
            return nullptr;
        }
        for (ggml_tensor * t : vision_side_weights) {
            std::vector<uint8_t> bytes = g.read_convert(t->name, t->type);
            if (bytes.size() != ggml_nbytes(t)) {
                std::fprintf(stderr, "vla(hy_vla): vision sidecar upload size mismatch for %s\n", t->name);
                return nullptr;
            }
            ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        }
        for (const auto & generated : vision_side_generated_weights) {
            ggml_tensor * t = generated.first;
            const std::vector<uint8_t> & bytes = generated.second;
            if (bytes.size() != ggml_nbytes(t)) {
                std::fprintf(stderr, "vla(hy_vla): vision sidecar generated upload size mismatch for %s\n", t->name);
                return nullptr;
            }
            ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
        }
        std::printf("vla(hy_vla): loaded CPU vision sidecar weights %.2f GiB (vision_layers=%d)\n",
                    ggml_backend_buffer_get_size(m->vision_weight_buf) / (1024.0 * 1024.0 * 1024.0),
                    m->vision_cpu_frontend.layers_loaded);
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) {
        std::fprintf(stderr, "vla(hy_vla): failed to allocate resident weights\n");
#ifdef GGML_USE_CUDA
        if (m->is_cuda && env_enabled("VLA_HY_VLA_CUDA_OOM_FALLBACK_CPU")) {
            std::fprintf(stderr, "vla(hy_vla): retrying resident weights on CPU because VLA_HY_VLA_CUDA_OOM_FALLBACK_CPU is set\n");
            ggml_backend_free(m->backend);
            m->backend = ggml_backend_cpu_init();
            m->is_cuda = false;
            if (!m->backend) return nullptr;
            ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
            std::printf("vla(hy_vla): backend = CPU (%d threads)\n", m->n_threads);
            m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
            if (!m->weight_buf) {
                std::fprintf(stderr, "vla(hy_vla): failed to allocate resident weights on CPU fallback\n");
                return nullptr;
            }
        } else
#endif
        return nullptr;
    }
    for (ggml_tensor * t : weights) {
        std::vector<uint8_t> bytes = g.read_convert(t->name, t->type);
        if (bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(hy_vla): upload size mismatch for %s\n", t->name);
            return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    for (const auto & generated : generated_weights) {
        ggml_tensor * t = generated.first;
        const std::vector<uint8_t> & bytes = generated.second;
        if (bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(hy_vla): generated upload size mismatch for %s\n", t->name);
            return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    if (!load_stats(g, *m)) return nullptr;
    std::printf("vla(hy_vla): loaded resident weights %.2f GiB (suffix expert + text_layers=%d prefix_vision=%s vision_frontend=%s vision_layers=%d vision_sidecar=%s; full GGUF may contain unloaded VLM/vision too)\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0),
                m->text_layers_loaded,
                m->prefix_vision_loaded ? "yes" : "no",
                m->vision_frontend_loaded ? "yes" : "no",
                m->vision_layers_loaded,
                m->vision_cpu_sideload ? "cpu" : "none");
    return m;
}

std::vector<float> HyVLAModelArch::predict(const Inputs & in) {
    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();
    stats = Stats{};
    const int64_t chunk = cfg.n_suffix;
    const int64_t max_sd = cfg.max_state_dim;
    const int64_t max_ad = cfg.max_action_dim;
    const int64_t seq = chunk + 1;
    int run_steps = cfg.num_steps;
    int64_t run_layers = cfg.n_layers;
    if (const char * e = std::getenv("VLA_HY_VLA_DEBUG_STEPS")) {
        run_steps = std::max(1, std::min(cfg.num_steps, std::atoi(e)));
    }
    if (const char * e = std::getenv("VLA_HY_VLA_DEBUG_LAYERS")) {
        run_layers = std::max<int64_t>(1, std::min<int64_t>(cfg.n_layers, std::atoll(e)));
    }
    const char * debug_output_env = std::getenv("VLA_HY_VLA_DEBUG_OUTPUT");
    const std::string debug_output = debug_output_env ? std::string(debug_output_env) : std::string();
    const float dt = -1.0f / (float) cfg.num_steps;

    if (vlm_only && debug_output != "generate") {
        std::fprintf(stderr, "vla(hy_vla): VLM-only GGUF has no action expert; only VLA_HY_VLA_DEBUG_OUTPUT=generate is supported\n");
        return {};
    }

    if (debug_output == "lang") {
        if (!in.lang_tokens || in.n_lang < 1) {
            std::fprintf(stderr, "vla(hy_vla): lang debug requires lang_tokens\n");
            return {};
        }
        std::vector<int32_t> ids(in.lang_tokens, in.lang_tokens + in.n_lang);
        std::vector<float> rows((size_t) in.n_lang * (size_t) cfg.hidden);
        gguf_reader g;
        if (!g.open(ckpt_path)) return {};
        if (!g.fetch_rows_f32("dual_tower.vlm.model.language_model.lm_head.weight",
                              ids, rows.data(), cfg.hidden)) {
            return {};
        }
        return rows;
    }

    if (debug_output == "vision_pixels" || debug_output == "vision_patch" ||
        debug_output == "vision_vit" || debug_output == "vision_merger") {
        if (!vision_frontend_loaded || !vision_frontend.base_loaded) {
            std::fprintf(stderr, "vla(hy_vla): %s debug requires VLA_HY_VLA_VISION_LAYERS=0 or higher\n",
                         debug_output.c_str());
            return {};
        }
        if (!in.images || in.n_images < 1) {
            std::fprintf(stderr, "vla(hy_vla): %s debug requires at least one 224x224 image\n",
                         debug_output.c_str());
            return {};
        }
        constexpr int64_t image_size = 224;
        constexpr int64_t vision_width = 1152;
        constexpr int64_t n_patches = 196;
        if (debug_output == "vision_pixels") {
            std::vector<float> chw;
            if (!preprocess_hy_image_chw_224(in.images[0], chw)) return {};
            return chw;
        }
        ggml_init_params vp{(size_t) 2048 * 1024 * 1024, nullptr, true};
        ggml_context * V = ggml_init(vp);
        if (!V) return {};
        ggml_tensor * t_px = ggml_new_tensor_3d(V, GGML_TYPE_F32, image_size, image_size, 3);
        ggml_set_input(t_px);
        ggml_tensor * h = build_hy_vision_patch_embed_224(V, *this, t_px);
        if (debug_output == "vision_vit" || debug_output == "vision_merger") {
            const int n_vit = std::max<int>(0, std::min<int>(vision_layers_loaded, (int) run_layers));
            if (debug_output == "vision_vit" && n_vit < 1) {
                std::fprintf(stderr, "vla(hy_vla): vision_vit debug requires VLA_HY_VLA_VISION_LAYERS >= 1\n");
                ggml_free(V);
                return {};
            }
            for (int i = 0; i < n_vit; ++i) {
                h = build_hy_vision_block(V, vision_frontend.blocks[(size_t) i], h, n_patches);
            }
        }
        if (debug_output == "vision_merger") {
            h = build_hy_vision_merger_14(V, *this, h);
        }
        ggml_set_output(h);
        ggml_cgraph * vg = ggml_new_graph_custom(V, 65536, false);
        ggml_build_forward_expand(vg, h);
        ggml_gallocr_t vga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!vga || !ggml_gallocr_alloc_graph(vga, vg)) {
            std::fprintf(stderr, "vla(hy_vla): %s graph allocation failed\n", debug_output.c_str());
            if (vga) ggml_gallocr_free(vga);
            ggml_free(V);
            return {};
        }
        std::vector<float> chw;
        if (!preprocess_hy_image_chw_224(in.images[0], chw)) {
            ggml_gallocr_free(vga);
            ggml_free(V);
            return {};
        }
        ggml_backend_tensor_set(t_px, chw.data(), 0, ggml_nbytes(t_px));
        if (ggml_backend_graph_compute(backend, vg) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(hy_vla): %s graph compute failed\n", debug_output.c_str());
            ggml_gallocr_free(vga);
            ggml_free(V);
            return {};
        }
        std::vector<float> out((size_t) ggml_nelements(h));
        ggml_backend_tensor_get(h, out.data(), 0, out.size() * sizeof(float));
        ggml_gallocr_free(vga);
        ggml_free(V);
        return out;
    }

    if (debug_output == "prefix") {
        if (!in.lang_tokens || in.n_lang < 1) {
            std::fprintf(stderr, "vla(hy_vla): prefix debug requires lang_tokens\n");
            return {};
        }
        const int64_t n_tok = in.n_lang;
        std::vector<int32_t> ids(in.lang_tokens, in.lang_tokens + in.n_lang);
        std::vector<float> lang_rows((size_t) n_tok * (size_t) cfg.hidden);
        {
            gguf_reader g;
            if (!g.open(ckpt_path)) return {};
            if (!g.fetch_rows_f32("dual_tower.vlm.model.language_model.lm_head.weight",
                                  ids, lang_rows.data(), cfg.hidden)) {
                return {};
            }
        }

        ggml_init_params pp{(size_t) 256 * 1024 * 1024, nullptr, true};
        ggml_context * P = ggml_init(pp);
        if (!P) return {};
        ggml_tensor * t_lang = ggml_new_tensor_2d(P, GGML_TYPE_F32, cfg.hidden, n_tok); ggml_set_input(t_lang);
        ggml_tensor * t_pos = ggml_new_tensor_1d(P, GGML_TYPE_I32, n_tok); ggml_set_input(t_pos);
        ggml_tensor * t_mask = ggml_new_tensor_2d(P, GGML_TYPE_F16, n_tok, n_tok); ggml_set_input(t_mask);
        ggml_tensor * h = t_lang;
        for (int i = 0; i < text_layers_loaded; ++i) {
            h = build_hy_text_layer(P, text_layers[(size_t) i], h, t_pos, t_mask, cfg, n_tok);
        }
        if (Wtext_norm) {
            h = ggml_mul(P, ggml_rms_norm(P, h, cfg.rms_eps), Wtext_norm);
        }
        ggml_set_output(h);
        ggml_cgraph * gf = ggml_new_graph_custom(P, 32768, false);
        ggml_build_forward_expand(gf, h);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
            std::fprintf(stderr, "vla(hy_vla): prefix graph allocation failed\n");
            if (galloc) ggml_gallocr_free(galloc);
            ggml_free(P);
            return {};
        }
        ggml_backend_tensor_set(t_lang, lang_rows.data(), 0, ggml_nbytes(t_lang));
        std::vector<int32_t> pos((size_t) n_tok);
        for (int64_t i = 0; i < n_tok; ++i) pos[(size_t) i] = (int32_t) i;
        ggml_backend_tensor_set(t_pos, pos.data(), 0, ggml_nbytes(t_pos));
        std::vector<ggml_fp16_t> mask((size_t) n_tok * (size_t) n_tok);
        for (int64_t r = 0; r < n_tok; ++r) {
            for (int64_t c = 0; c < n_tok; ++c) {
                const float v = (c <= r) ? 0.f : -INFINITY;
                mask[(size_t) r * (size_t) n_tok + (size_t) c] = ggml_fp32_to_fp16(v);
            }
        }
        ggml_backend_tensor_set(t_mask, mask.data(), 0, ggml_nbytes(t_mask));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(hy_vla): prefix graph compute failed\n");
            ggml_gallocr_free(galloc);
            ggml_free(P);
            return {};
        }
        std::vector<float> out((size_t) ggml_nelements(h));
        ggml_backend_tensor_get(h, out.data(), 0, out.size() * sizeof(float));
        ggml_gallocr_free(galloc);
        ggml_free(P);
        return out;
    }

    // Greedy text generation over the causal prefill graph. Inputs stay
    // ids-in/ids-out: lang_tokens carries the prompt ids (plus optionally
    // images through the standard vision frontend + prefix builder), the
    // return vector carries the generated ids. Each step re-runs the
    // prefill over the grown prefix; fine for short demo generations.
    if (debug_output == "generate") {
        if (!W_vocab) {
            std::fprintf(stderr, "vla(hy_vla): generate requires VLA_HY_VLA_DEBUG_OUTPUT=generate at load time\n");
            return {};
        }
        if (text_layers_loaded < 1) {
            std::fprintf(stderr, "vla(hy_vla): generate requires VLA_HY_VLA_TEXT_LAYERS >= 1\n");
            return {};
        }
        if (!in.lang_tokens || in.n_lang < 1) {
            std::fprintf(stderr, "vla(hy_vla): generate requires lang_tokens\n");
            return {};
        }
        int n_predict = 64;
        if (const char * e = std::getenv("VLA_HY_VLA_N_PREDICT")) {
            n_predict = std::max(1, std::atoi(e));
        }

        gguf_reader g;
        if (!g.open(ckpt_path)) return {};

        std::vector<float> prefix;
        std::vector<int32_t> modality_mask;
        const bool with_images = in.images && in.n_images > 0;
        if (with_images) {
            if (!prefix_vision_loaded) {
                std::fprintf(stderr, "vla(hy_vla): generate with images requires stable GGUF VLM text/vision branches\n");
                return {};
            }
            std::vector<float> visual_tokens;
            int visual_views = 0;
            if (!run_hy_vision_frontend_224(*this, in, visual_tokens, visual_views)) {
                return {};
            }
            Inputs prefix_inputs = in;
            prefix_inputs.precomputed_img_emb = visual_tokens.data();
            prefix_inputs.n_img_views = visual_views * 49;
            prefix_inputs.images = nullptr;
            prefix_inputs.n_images = 0;
            if (!build_hy_vla_prefix_from_visual_tokens(*this, prefix_inputs, 49,
                                                        prefix, modality_mask)) {
                return {};
            }
        } else {
            const std::vector<int32_t> ids(in.lang_tokens, in.lang_tokens + in.n_lang);
            if (!append_token_embeddings(g, ids, cfg.hidden, prefix)) return {};
            modality_mask.assign(ids.size(), 0);
        }
        // Generated tokens are all text, so the vision patch groups are
        // fixed for the whole loop.
        const std::vector<std::vector<int64_t>> patch_groups =
            make_visual_patch_groups(modality_mask.data(), (int) modality_mask.size());
        const int gen_layers = std::max<int>(1, std::min<int>(text_layers_loaded, (int) run_layers));

        std::vector<float> gen_ids;
        gen_ids.reserve((size_t) n_predict);
        for (int step = 0; step < n_predict; ++step) {
            const int64_t n_tok = (int64_t) modality_mask.size();
            ggml_init_params gp{(size_t) 512 * 1024 * 1024, nullptr, true};
            ggml_context * G = ggml_init(gp);
            if (!G) return {};
            ggml_tensor * t_pref = ggml_new_tensor_2d(G, GGML_TYPE_F32, cfg.hidden, n_tok); ggml_set_input(t_pref);
            ggml_tensor * t_pos = ggml_new_tensor_1d(G, GGML_TYPE_I32, n_tok); ggml_set_input(t_pos);
            ggml_tensor * t_mask = ggml_new_tensor_2d(G, GGML_TYPE_F16, n_tok, n_tok); ggml_set_input(t_mask);
            ggml_tensor * h = t_pref;
            if (with_images) {
                const std::vector<HyPrefixRun> runs =
                    make_prefix_runs(modality_mask.data(), (int) modality_mask.size());
                const std::vector<HyPrefixRange> visual_segments;
                for (int i = 0; i < gen_layers; ++i) {
                    h = build_hy_prefix_routed_layer(G, text_layers[(size_t) i], vision_layers[(size_t) i],
                                                     h, t_pos, t_mask, cfg, runs, visual_segments, true);
                }
            } else {
                for (int i = 0; i < gen_layers; ++i) {
                    h = build_hy_text_layer(G, text_layers[(size_t) i], h, t_pos, t_mask, cfg, n_tok);
                }
            }
            if (Wtext_norm) {
                h = ggml_mul(G, ggml_rms_norm(G, h, cfg.rms_eps), Wtext_norm);
            }
            const size_t rb = (size_t) cfg.hidden * sizeof(float);
            ggml_tensor * h_last = ggml_view_2d(G, h, cfg.hidden, 1, rb, rb * (size_t) (n_tok - 1));
            ggml_tensor * logits = mm_w(G, W_vocab, h_last);
            ggml_set_output(logits);
            ggml_cgraph * gf = ggml_new_graph_custom(G, hy_vla_graph_size(1048576), false);
            ggml_build_forward_expand(gf, logits);
            ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
                std::fprintf(stderr, "vla(hy_vla): generate graph allocation failed\n");
                if (galloc) ggml_gallocr_free(galloc);
                ggml_free(G);
                return {};
            }
            ggml_backend_tensor_set(t_pref, prefix.data(), 0, ggml_nbytes(t_pref));
            std::vector<int32_t> pos((size_t) n_tok);
            for (int64_t i = 0; i < n_tok; ++i) pos[(size_t) i] = (int32_t) i;
            ggml_backend_tensor_set(t_pos, pos.data(), 0, ggml_nbytes(t_pos));
            const std::vector<ggml_fp16_t> mask =
                make_prefix_mask(n_tok, with_images ? &patch_groups : nullptr);
            ggml_backend_tensor_set(t_mask, mask.data(), 0, ggml_nbytes(t_mask));
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "vla(hy_vla): generate graph compute failed\n");
                ggml_gallocr_free(galloc);
                ggml_free(G);
                return {};
            }
            std::vector<float> logits_row((size_t) ggml_nelements(logits));
            ggml_backend_tensor_get(logits, logits_row.data(), 0, logits_row.size() * sizeof(float));
            ggml_gallocr_free(galloc);
            ggml_free(G);

            int32_t next = 0;
            for (size_t i = 1; i < logits_row.size(); ++i) {
                if (logits_row[i] > logits_row[(size_t) next]) next = (int32_t) i;
            }
            gen_ids.push_back((float) next);
            if (next == eos_token_id) break;
            if (!append_token_embeddings(g, {next}, cfg.hidden, prefix)) return {};
            modality_mask.push_back(0);
        }
        stats.ms_total = std::chrono::duration<float, std::milli>(clk::now() - t0).count();
        return gen_ids;
    }

    if (debug_output == "mixed_prefix") {
        if (text_layers_loaded < 1 || !prefix_vision_loaded) {
            std::fprintf(stderr, "vla(hy_vla): mixed_prefix debug requires stable GGUF VLM text/vision branches and VLA_HY_VLA_TEXT_LAYERS >= 1\n");
            return {};
        }
        if (!in.lang_tokens || in.n_lang < 1) {
            std::fprintf(stderr, "vla(hy_vla): mixed_prefix debug requires lang_tokens\n");
            return {};
        }
        if (!in.precomputed_img_emb || in.n_img_views < 1) {
            std::fprintf(stderr, "vla(hy_vla): mixed_prefix debug requires precomputed_img_emb; n_img_views is interpreted as visual-token count\n");
            return {};
        }
        const int64_t n_text = in.n_lang;
        const int64_t n_vis = in.n_img_views;
        const int64_t n_pref = n_text + n_vis;
        const int mixed_layers = std::max<int>(1, std::min<int>(text_layers_loaded, (int) run_layers));

        std::vector<int32_t> ids(in.lang_tokens, in.lang_tokens + in.n_lang);
        std::vector<float> lang_rows((size_t) n_text * (size_t) cfg.hidden);
        {
            gguf_reader g;
            if (!g.open(ckpt_path)) return {};
            if (!g.fetch_rows_f32("dual_tower.vlm.model.language_model.lm_head.weight",
                                  ids, lang_rows.data(), cfg.hidden)) return {};
        }
        std::vector<float> vis_rows(in.precomputed_img_emb,
                                    in.precomputed_img_emb + (size_t) n_vis * (size_t) cfg.hidden);

        ggml_init_params mp{(size_t) 512 * 1024 * 1024, nullptr, true};
        ggml_context * M = ggml_init(mp);
        if (!M) return {};
        ggml_tensor * t_text = ggml_new_tensor_2d(M, GGML_TYPE_F32, cfg.hidden, n_text); ggml_set_input(t_text);
        ggml_tensor * t_vis = ggml_new_tensor_2d(M, GGML_TYPE_F32, cfg.hidden, n_vis); ggml_set_input(t_vis);
        ggml_tensor * t_pos = ggml_new_tensor_1d(M, GGML_TYPE_I32, n_pref); ggml_set_input(t_pos);
        ggml_tensor * t_mask = ggml_new_tensor_2d(M, GGML_TYPE_F16, n_pref, n_pref); ggml_set_input(t_mask);

        ggml_tensor * ht = t_text;
        ggml_tensor * hv = t_vis;
        ggml_tensor * hcat = nullptr;
        for (int i = 0; i < mixed_layers; ++i) {
            hcat = build_hy_prefix_two_segment_layer(M, text_layers[(size_t) i], vision_layers[(size_t) i],
                                                     ht, hv, t_pos, t_mask, cfg, n_text, n_vis);
            const size_t rb = (size_t) cfg.hidden * sizeof(float);
            ht = ggml_view_2d(M, hcat, cfg.hidden, n_text, rb, 0);
            hv = ggml_view_2d(M, hcat, cfg.hidden, n_vis, rb, rb * (size_t) n_text);
        }
        if (Wtext_norm) {
            hcat = ggml_mul(M, ggml_rms_norm(M, hcat, cfg.rms_eps), Wtext_norm);
        }
        ggml_set_output(hcat);
        ggml_cgraph * gf = ggml_new_graph_custom(M, 32768, false);
        ggml_build_forward_expand(gf, hcat);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
            std::fprintf(stderr, "vla(hy_vla): mixed_prefix graph allocation failed\n");
            if (galloc) ggml_gallocr_free(galloc);
            ggml_free(M);
            return {};
        }
        ggml_backend_tensor_set(t_text, lang_rows.data(), 0, ggml_nbytes(t_text));
        ggml_backend_tensor_set(t_vis, vis_rows.data(), 0, ggml_nbytes(t_vis));
        std::vector<int32_t> pos((size_t) n_pref);
        for (int64_t i = 0; i < n_pref; ++i) pos[(size_t) i] = (int32_t) i;
        ggml_backend_tensor_set(t_pos, pos.data(), 0, ggml_nbytes(t_pos));
        std::vector<ggml_fp16_t> mask((size_t) n_pref * (size_t) n_pref);
        for (int64_t r = 0; r < n_pref; ++r) {
            for (int64_t c = 0; c < n_pref; ++c) {
                mask[(size_t) r * (size_t) n_pref + (size_t) c] =
                    ggml_fp32_to_fp16((c <= r) ? 0.f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(t_mask, mask.data(), 0, ggml_nbytes(t_mask));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(hy_vla): mixed_prefix graph compute failed\n");
            ggml_gallocr_free(galloc);
            ggml_free(M);
            return {};
        }
        std::vector<float> out((size_t) ggml_nelements(hcat));
        ggml_backend_tensor_get(hcat, out.data(), 0, out.size() * sizeof(float));
        ggml_gallocr_free(galloc);
        ggml_free(M);
        return out;
    }

    if (debug_output == "mixed_joint_vt" || debug_output == "mixed_joint") {
        if (text_layers_loaded < 1 || !prefix_vision_loaded) {
            std::fprintf(stderr, "vla(hy_vla): mixed_joint debug requires stable GGUF VLM text/vision branches and VLA_HY_VLA_TEXT_LAYERS >= 1\n");
            return {};
        }
        if (!in.lang_tokens || in.n_lang < 1) {
            std::fprintf(stderr, "vla(hy_vla): mixed_joint debug requires lang_tokens\n");
            return {};
        }
        if (!in.precomputed_img_emb || in.n_img_views < 1) {
            std::fprintf(stderr, "vla(hy_vla): mixed_joint debug requires precomputed_img_emb; n_img_views is interpreted as visual-token count\n");
            return {};
        }
        const int64_t n_text = in.n_lang;
        const int64_t n_vis = in.n_img_views;
        const int64_t n_pref = n_text + n_vis;
        const int64_t n_suf = cfg.n_suffix + 1;
        const int64_t n_total = n_pref + n_suf;
        const int mixed_layers = std::max<int>(1, std::min<int>(text_layers_loaded, (int) run_layers));

        std::vector<int32_t> ids(in.lang_tokens, in.lang_tokens + in.n_lang);
        std::vector<float> lang_rows((size_t) n_text * (size_t) cfg.hidden);
        {
            gguf_reader g;
            if (!g.open(ckpt_path)) return {};
            if (!g.fetch_rows_f32("dual_tower.vlm.model.language_model.lm_head.weight",
                                  ids, lang_rows.data(), cfg.hidden)) return {};
        }
        std::vector<float> vis_rows(in.precomputed_img_emb,
                                    in.precomputed_img_emb + (size_t) n_vis * (size_t) cfg.hidden);

        ggml_init_params mp{(size_t) 768 * 1024 * 1024, nullptr, true};
        ggml_context * M = ggml_init(mp);
        if (!M) return {};
        ggml_tensor * t_text = ggml_new_tensor_2d(M, GGML_TYPE_F32, cfg.hidden, n_text); ggml_set_input(t_text);
        ggml_tensor * t_vis = ggml_new_tensor_2d(M, GGML_TYPE_F32, cfg.hidden, n_vis); ggml_set_input(t_vis);
        ggml_tensor * t_pref_pos = ggml_new_tensor_1d(M, GGML_TYPE_I32, n_pref); ggml_set_input(t_pref_pos);
        ggml_tensor * t_pref_mask = ggml_new_tensor_2d(M, GGML_TYPE_F16, n_pref, n_pref); ggml_set_input(t_pref_mask);
        ggml_tensor * t_state = ggml_new_tensor_1d(M, GGML_TYPE_F32, max_sd); ggml_set_input(t_state);
        ggml_tensor * t_x0 = ggml_new_tensor_2d(M, GGML_TYPE_F32, max_ad, chunk); ggml_set_input(t_x0);
        ggml_tensor * t_time = ggml_new_tensor_2d(M, GGML_TYPE_F32, cfg.expert_h, chunk); ggml_set_input(t_time);
        ggml_tensor * t_suf_pos = ggml_new_tensor_1d(M, GGML_TYPE_I32, n_suf); ggml_set_input(t_suf_pos);
        ggml_tensor * t_full_mask = ggml_new_tensor_2d(M, GGML_TYPE_F16, n_total, n_suf); ggml_set_input(t_full_mask);

        std::vector<ggml_tensor *> k_cache((size_t) mixed_layers);
        std::vector<ggml_tensor *> v_cache((size_t) mixed_layers);
        ggml_tensor * ht = t_text;
        ggml_tensor * hv = t_vis;
        ggml_tensor * hcat = nullptr;
        for (int i = 0; i < mixed_layers; ++i) {
            hcat = build_hy_prefix_two_segment_layer(M, text_layers[(size_t) i], vision_layers[(size_t) i],
                                                     ht, hv, t_pref_pos, t_pref_mask, cfg, n_text, n_vis,
                                                     &k_cache[(size_t) i], &v_cache[(size_t) i]);
            const size_t rb = (size_t) cfg.hidden * sizeof(float);
            ht = ggml_view_2d(M, hcat, cfg.hidden, n_text, rb, 0);
            hv = ggml_view_2d(M, hcat, cfg.hidden, n_vis, rb, rb * (size_t) n_text);
        }

        ggml_tensor * hs = build_suffix_embed(M, *this, t_state, t_x0, t_time);
        for (int i = 0; i < mixed_layers; ++i) {
            hs = build_hy_expert_layer_with_prefix(M, layers[(size_t) i], hs, t_suf_pos,
                                                   k_cache[(size_t) i], v_cache[(size_t) i],
                                                   t_full_mask, cfg, true);
        }
        ggml_tensor * h_final = ggml_mul(M, ggml_rms_norm(M, hs, cfg.rms_eps), Wnorm);
        const size_t rb = (size_t) cfg.expert_h * sizeof(float);
        ggml_tensor * h_actions = ggml_view_2d(M, h_final, cfg.expert_h, chunk, rb, rb);
        ggml_tensor * v_t = ggml_add(M, mm_w(M, W_aout, h_actions), b_aout);
        ggml_tensor * out_t = v_t;
        if (debug_output == "mixed_joint") {
            out_t = ggml_add(M, t_x0, ggml_scale(M, v_t, dt));
        }
        ggml_set_output(out_t);
        ggml_cgraph * gf = ggml_new_graph_custom(M, 32768, false);
        ggml_build_forward_expand(gf, out_t);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
            std::fprintf(stderr, "vla(hy_vla): mixed_joint graph allocation failed\n");
            if (galloc) ggml_gallocr_free(galloc);
            ggml_free(M);
            return {};
        }
        auto set_if_alloc = [](ggml_tensor * t, const void * data) {
            ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
            if (buf) ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        };
        set_if_alloc(t_text, lang_rows.data());
        set_if_alloc(t_vis, vis_rows.data());
        std::vector<int32_t> pref_pos((size_t) n_pref);
        for (int64_t i = 0; i < n_pref; ++i) pref_pos[(size_t) i] = (int32_t) i;
        set_if_alloc(t_pref_pos, pref_pos.data());
        std::vector<ggml_fp16_t> pref_mask((size_t) n_pref * (size_t) n_pref);
        for (int64_t r = 0; r < n_pref; ++r) {
            for (int64_t c = 0; c < n_pref; ++c) {
                pref_mask[(size_t) r * (size_t) n_pref + (size_t) c] =
                    ggml_fp32_to_fp16((c <= r) ? 0.f : -INFINITY);
            }
        }
        set_if_alloc(t_pref_mask, pref_mask.data());
        std::vector<float> sh((size_t) max_sd, 0.f);
        if (in.state) std::memcpy(sh.data(), in.state, (size_t) max_sd * sizeof(float));
        set_if_alloc(t_state, sh.data());
        std::vector<float> x0((size_t) max_ad * (size_t) chunk);
        if (in.noise) std::memcpy(x0.data(), in.noise, x0.size() * sizeof(float));
        else std::fill(x0.begin(), x0.end(), 0.f);
        set_if_alloc(t_x0, x0.data());
        const std::vector<float> tv = sinusoidal_time_emb(hy_vla_debug_timestep(), cfg.expert_h, cfg.min_period, cfg.max_period);
        std::vector<float> tile((size_t) cfg.expert_h * (size_t) chunk);
        for (int64_t c = 0; c < chunk; ++c) {
            std::memcpy(tile.data() + (size_t) c * (size_t) cfg.expert_h, tv.data(), tv.size() * sizeof(float));
        }
        set_if_alloc(t_time, tile.data());
        std::vector<int32_t> suf_pos((size_t) n_suf);
        for (int64_t i = 0; i < n_suf; ++i) suf_pos[(size_t) i] = (int32_t) (n_pref + i);
        set_if_alloc(t_suf_pos, suf_pos.data());
        std::vector<ggml_fp16_t> full_mask((size_t) n_total * (size_t) n_suf);
        for (int64_t r = 0; r < n_suf; ++r) {
            for (int64_t c = 0; c < n_total; ++c) {
                bool allowed = c < n_pref;
                if (!allowed) {
                    const int64_t s = c - n_pref;
                    allowed = (r == 0) ? (s == 0) : true;
                }
                full_mask[(size_t) r * (size_t) n_total + (size_t) c] =
                    ggml_fp32_to_fp16(allowed ? 0.f : -INFINITY);
            }
        }
        set_if_alloc(t_full_mask, full_mask.data());
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(hy_vla): mixed_joint graph compute failed\n");
            ggml_gallocr_free(galloc);
            ggml_free(M);
            return {};
        }
        std::vector<float> out((size_t) ggml_nelements(out_t));
        ggml_backend_tensor_get(out_t, out.data(), 0, out.size() * sizeof(float));
        ggml_gallocr_free(galloc);
        ggml_free(M);
        return out;
    }

    if (debug_output == "routed_prefix" || debug_output == "routed_joint_vt" || debug_output == "routed_joint") {
        if (text_layers_loaded < 1 || !prefix_vision_loaded) {
            std::fprintf(stderr, "vla(hy_vla): routed debug requires stable GGUF VLM text/vision branches and VLA_HY_VLA_TEXT_LAYERS >= 1\n");
            return {};
        }
        if (!in.precomputed_img_emb || in.n_img_views < 1) {
            std::fprintf(stderr, "vla(hy_vla): routed debug requires full prefix embeddings in precomputed_img_emb; n_img_views is prefix token count\n");
            return {};
        }
        if (!in.attention_mask || in.attention_mask_n != in.n_img_views) {
            std::fprintf(stderr, "vla(hy_vla): routed debug requires attention_mask length == prefix token count, using 0=text 1=vision\n");
            return {};
        }
        const int64_t n_pref = in.n_img_views;
        const bool want_joint = debug_output == "routed_joint_vt" || debug_output == "routed_joint";
        const std::vector<HyPrefixRun> runs = make_prefix_runs(in.attention_mask, in.attention_mask_n);
        const std::vector<HyPrefixRange> visual_segments =
            want_joint ? std::vector<HyPrefixRange>{} :
                         make_visual_override_segments(in.attention_mask, in.attention_mask_n);
        const std::vector<std::vector<int64_t>> patch_groups =
            make_visual_patch_groups(in.attention_mask, in.attention_mask_n);
        if (runs.empty()) return {};
        const int routed_layers = std::max<int>(1, std::min<int>(text_layers_loaded, (int) run_layers));
        std::vector<float> pref_rows(in.precomputed_img_emb,
                                     in.precomputed_img_emb + (size_t) n_pref * (size_t) cfg.hidden);

        const int64_t n_suf = cfg.n_suffix + 1;
        const int64_t n_total = n_pref + n_suf;
        ggml_init_params rp{(size_t) (want_joint ? 768 : 512) * 1024 * 1024, nullptr, true};
        ggml_context * R = ggml_init(rp);
        if (!R) return {};
        ggml_tensor * t_pref = ggml_new_tensor_2d(R, GGML_TYPE_F32, cfg.hidden, n_pref); ggml_set_input(t_pref);
        ggml_tensor * t_pref_pos = ggml_new_tensor_1d(R, GGML_TYPE_I32, n_pref); ggml_set_input(t_pref_pos);
        ggml_tensor * t_pref_mask = ggml_new_tensor_2d(R, GGML_TYPE_F16, n_pref, n_pref); ggml_set_input(t_pref_mask);
        ggml_tensor * h = t_pref;
        std::vector<ggml_tensor *> k_cache((size_t) routed_layers);
        std::vector<ggml_tensor *> v_cache((size_t) routed_layers);
        for (int i = 0; i < routed_layers; ++i) {
            h = build_hy_prefix_routed_layer(R, text_layers[(size_t) i], vision_layers[(size_t) i],
                                             h, t_pref_pos, t_pref_mask, cfg, runs, visual_segments,
                                             want_joint,
                                             want_joint ? &k_cache[(size_t) i] : nullptr,
                                             want_joint ? &v_cache[(size_t) i] : nullptr);
        }
        ggml_tensor * out_t = h;
        ggml_tensor * t_state = nullptr;
        ggml_tensor * t_x0 = nullptr;
        ggml_tensor * t_time = nullptr;
        ggml_tensor * t_suf_pos = nullptr;
        ggml_tensor * t_full_mask = nullptr;
        if (want_joint) {
            t_state = ggml_new_tensor_1d(R, GGML_TYPE_F32, max_sd); ggml_set_input(t_state);
            t_x0 = ggml_new_tensor_2d(R, GGML_TYPE_F32, max_ad, chunk); ggml_set_input(t_x0);
            t_time = ggml_new_tensor_2d(R, GGML_TYPE_F32, cfg.expert_h, chunk); ggml_set_input(t_time);
            t_suf_pos = ggml_new_tensor_1d(R, GGML_TYPE_I32, n_suf); ggml_set_input(t_suf_pos);
            t_full_mask = ggml_new_tensor_2d(R, GGML_TYPE_F16, n_total, n_suf); ggml_set_input(t_full_mask);
            ggml_tensor * hs = build_suffix_embed(R, *this, t_state, t_x0, t_time);
            for (int i = 0; i < routed_layers; ++i) {
                hs = build_hy_expert_layer_with_prefix(R, layers[(size_t) i], hs, t_suf_pos,
                                                       k_cache[(size_t) i], v_cache[(size_t) i],
                                                       t_full_mask, cfg, true);
            }
            ggml_tensor * h_final = ggml_mul(R, ggml_rms_norm(R, hs, cfg.rms_eps), Wnorm);
            const size_t rb = (size_t) cfg.expert_h * sizeof(float);
            ggml_tensor * h_actions = ggml_view_2d(R, h_final, cfg.expert_h, chunk, rb, rb);
            ggml_tensor * v_t = ggml_add(R, mm_w(R, W_aout, h_actions), b_aout);
            out_t = (debug_output == "routed_joint") ? ggml_add(R, t_x0, ggml_scale(R, v_t, dt)) : v_t;
        } else if (Wtext_norm) {
            out_t = ggml_mul(R, ggml_rms_norm(R, h, cfg.rms_eps), Wtext_norm);
        }
        ggml_set_output(out_t);
        ggml_cgraph * gf = ggml_new_graph_custom(R, hy_vla_graph_size(1048576), false);
        ggml_build_forward_expand(gf, out_t);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
            std::fprintf(stderr, "vla(hy_vla): routed graph allocation failed\n");
            if (galloc) ggml_gallocr_free(galloc);
            ggml_free(R);
            return {};
        }
        auto set_if_alloc = [](ggml_tensor * t, const void * data) {
            if (!t) return;
            ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
            if (buf) ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        };
        set_if_alloc(t_pref, pref_rows.data());
        std::vector<int32_t> pref_pos((size_t) n_pref);
        for (int64_t i = 0; i < n_pref; ++i) pref_pos[(size_t) i] = (int32_t) i;
        set_if_alloc(t_pref_pos, pref_pos.data());
        std::vector<ggml_fp16_t> pref_mask =
            make_prefix_mask(n_pref, want_joint ? &patch_groups : nullptr);
        set_if_alloc(t_pref_mask, pref_mask.data());
        if (want_joint) {
            std::vector<float> sh((size_t) max_sd, 0.f);
            if (in.state) std::memcpy(sh.data(), in.state, (size_t) max_sd * sizeof(float));
            set_if_alloc(t_state, sh.data());
            std::vector<float> x0((size_t) max_ad * (size_t) chunk);
            if (in.noise) std::memcpy(x0.data(), in.noise, x0.size() * sizeof(float));
            else std::fill(x0.begin(), x0.end(), 0.f);
            set_if_alloc(t_x0, x0.data());
            const std::vector<float> tv = sinusoidal_time_emb(hy_vla_debug_timestep(), cfg.expert_h, cfg.min_period, cfg.max_period);
            std::vector<float> tile((size_t) cfg.expert_h * (size_t) chunk);
            for (int64_t c = 0; c < chunk; ++c) {
                std::memcpy(tile.data() + (size_t) c * (size_t) cfg.expert_h, tv.data(), tv.size() * sizeof(float));
            }
            set_if_alloc(t_time, tile.data());
            std::vector<int32_t> suf_pos((size_t) n_suf);
            for (int64_t i = 0; i < n_suf; ++i) suf_pos[(size_t) i] = (int32_t) (n_pref + i);
            set_if_alloc(t_suf_pos, suf_pos.data());
            std::vector<ggml_fp16_t> full_mask((size_t) n_total * (size_t) n_suf);
            for (int64_t r = 0; r < n_suf; ++r) {
                for (int64_t c = 0; c < n_total; ++c) {
                    bool allowed = c < n_pref;
                    if (!allowed) {
                        const int64_t s = c - n_pref;
                        allowed = (r == 0) ? (s == 0) : true;
                    }
                    full_mask[(size_t) r * (size_t) n_total + (size_t) c] =
                        ggml_fp32_to_fp16(allowed ? 0.f : -INFINITY);
                }
            }
            set_if_alloc(t_full_mask, full_mask.data());
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(hy_vla): routed graph compute failed\n");
            ggml_gallocr_free(galloc);
            ggml_free(R);
            return {};
        }
        std::vector<float> out((size_t) ggml_nelements(out_t));
        ggml_backend_tensor_get(out_t, out.data(), 0, out.size() * sizeof(float));
        ggml_gallocr_free(galloc);
        ggml_free(R);
        return out;
    }

    if (debug_output == "joint_vt" || debug_output == "joint") {
        if (text_layers_loaded < 1) {
            std::fprintf(stderr, "vla(hy_vla): joint debug requires VLA_HY_VLA_TEXT_LAYERS >= 1\n");
            return {};
        }
        if (!in.lang_tokens || in.n_lang < 1) {
            std::fprintf(stderr, "vla(hy_vla): joint debug requires lang_tokens\n");
            return {};
        }
        const int64_t n_pref = in.n_lang;
        const int64_t n_suf = cfg.n_suffix + 1;
        const int64_t n_total = n_pref + n_suf;
        const int joint_layers = std::max<int>(1, std::min<int>(text_layers_loaded, (int) run_layers));

        std::vector<int32_t> ids(in.lang_tokens, in.lang_tokens + in.n_lang);
        std::vector<float> lang_rows((size_t) n_pref * (size_t) cfg.hidden);
        {
            gguf_reader g;
            if (!g.open(ckpt_path)) return {};
            if (!g.fetch_rows_f32("dual_tower.vlm.model.language_model.lm_head.weight",
                                  ids, lang_rows.data(), cfg.hidden)) return {};
        }

        ggml_init_params jp{(size_t) 384 * 1024 * 1024, nullptr, true};
        ggml_context * J = ggml_init(jp);
        if (!J) return {};
        ggml_tensor * t_lang = ggml_new_tensor_2d(J, GGML_TYPE_F32, cfg.hidden, n_pref); ggml_set_input(t_lang);
        ggml_tensor * t_pref_pos = ggml_new_tensor_1d(J, GGML_TYPE_I32, n_pref); ggml_set_input(t_pref_pos);
        ggml_tensor * t_pref_mask = ggml_new_tensor_2d(J, GGML_TYPE_F16, n_pref, n_pref); ggml_set_input(t_pref_mask);
        ggml_tensor * t_state = ggml_new_tensor_1d(J, GGML_TYPE_F32, max_sd); ggml_set_input(t_state);
        ggml_tensor * t_x0 = ggml_new_tensor_2d(J, GGML_TYPE_F32, max_ad, chunk); ggml_set_input(t_x0);
        ggml_tensor * t_time = ggml_new_tensor_2d(J, GGML_TYPE_F32, cfg.expert_h, chunk); ggml_set_input(t_time);
        ggml_tensor * t_suf_pos = ggml_new_tensor_1d(J, GGML_TYPE_I32, n_suf); ggml_set_input(t_suf_pos);
        ggml_tensor * t_full_mask = ggml_new_tensor_2d(J, GGML_TYPE_F16, n_total, n_suf); ggml_set_input(t_full_mask);

        std::vector<ggml_tensor *> k_cache((size_t) joint_layers);
        std::vector<ggml_tensor *> v_cache((size_t) joint_layers);
        ggml_tensor * hp = t_lang;
        for (int i = 0; i < joint_layers; ++i) {
            hp = build_hy_text_layer(J, text_layers[(size_t) i], hp, t_pref_pos, t_pref_mask, cfg, n_pref,
                                     &k_cache[(size_t) i], &v_cache[(size_t) i]);
        }
        ggml_tensor * hs = build_suffix_embed(J, *this, t_state, t_x0, t_time);
        for (int i = 0; i < joint_layers; ++i) {
            hs = build_hy_expert_layer_with_prefix(J, layers[(size_t) i], hs, t_suf_pos,
                                                   k_cache[(size_t) i], v_cache[(size_t) i],
                                                   t_full_mask, cfg, true);
        }
        ggml_tensor * h_final = ggml_mul(J, ggml_rms_norm(J, hs, cfg.rms_eps), Wnorm);
        const size_t rb = (size_t) cfg.expert_h * sizeof(float);
        ggml_tensor * h_actions = ggml_view_2d(J, h_final, cfg.expert_h, chunk, rb, rb);
        ggml_tensor * v_t = ggml_add(J, mm_w(J, W_aout, h_actions), b_aout);
        ggml_tensor * out_t = v_t;
        if (debug_output == "joint") {
            out_t = ggml_add(J, t_x0, ggml_scale(J, v_t, dt));
        }
        ggml_set_output(out_t);
        ggml_cgraph * gf = ggml_new_graph_custom(J, 32768, false);
        ggml_build_forward_expand(gf, out_t);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
            std::fprintf(stderr, "vla(hy_vla): joint graph allocation failed\n");
            if (galloc) ggml_gallocr_free(galloc);
            ggml_free(J);
            return {};
        }
        auto set_if_alloc = [](ggml_tensor * t, const void * data) {
            ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
            if (buf) {
                ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
            }
        };

        set_if_alloc(t_lang, lang_rows.data());
        std::vector<int32_t> pref_pos((size_t) n_pref);
        for (int64_t i = 0; i < n_pref; ++i) pref_pos[(size_t) i] = (int32_t) i;
        set_if_alloc(t_pref_pos, pref_pos.data());
        std::vector<ggml_fp16_t> pref_mask((size_t) n_pref * (size_t) n_pref);
        for (int64_t r = 0; r < n_pref; ++r) {
            for (int64_t c = 0; c < n_pref; ++c) {
                pref_mask[(size_t) r * (size_t) n_pref + (size_t) c] =
                    ggml_fp32_to_fp16((c <= r) ? 0.f : -INFINITY);
            }
        }
        set_if_alloc(t_pref_mask, pref_mask.data());
        std::vector<float> sh((size_t) max_sd, 0.f);
        if (in.state) std::memcpy(sh.data(), in.state, (size_t) max_sd * sizeof(float));
        set_if_alloc(t_state, sh.data());
        std::vector<float> x0((size_t) max_ad * (size_t) chunk);
        if (in.noise) std::memcpy(x0.data(), in.noise, x0.size() * sizeof(float));
        else std::fill(x0.begin(), x0.end(), 0.f);
        set_if_alloc(t_x0, x0.data());
        const std::vector<float> tv = sinusoidal_time_emb(hy_vla_debug_timestep(), cfg.expert_h, cfg.min_period, cfg.max_period);
        std::vector<float> tile((size_t) cfg.expert_h * (size_t) chunk);
        for (int64_t c = 0; c < chunk; ++c) {
            std::memcpy(tile.data() + (size_t) c * (size_t) cfg.expert_h, tv.data(), tv.size() * sizeof(float));
        }
        set_if_alloc(t_time, tile.data());
        std::vector<int32_t> suf_pos((size_t) n_suf);
        for (int64_t i = 0; i < n_suf; ++i) suf_pos[(size_t) i] = (int32_t) (n_pref + i);
        set_if_alloc(t_suf_pos, suf_pos.data());
        std::vector<ggml_fp16_t> full_mask((size_t) n_total * (size_t) n_suf);
        for (int64_t r = 0; r < n_suf; ++r) {
            for (int64_t c = 0; c < n_total; ++c) {
                bool allowed = c < n_pref;
                if (!allowed) {
                    const int64_t s = c - n_pref;
                    allowed = (r == 0) ? (s == 0) : true;
                }
                full_mask[(size_t) r * (size_t) n_total + (size_t) c] =
                    ggml_fp32_to_fp16(allowed ? 0.f : -INFINITY);
            }
        }
        set_if_alloc(t_full_mask, full_mask.data());
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(hy_vla): joint graph compute failed\n");
            ggml_gallocr_free(galloc);
            ggml_free(J);
            return {};
        }
        std::vector<float> out((size_t) ggml_nelements(out_t));
        ggml_backend_tensor_get(out_t, out.data(), 0, out.size() * sizeof(float));
        ggml_gallocr_free(galloc);
        ggml_free(J);
        return out;
    }

    std::vector<float> built_prefix;
    std::vector<int32_t> built_modality_mask;
    std::vector<float> built_visual_tokens;
    const float * routed_prefix_ptr = in.precomputed_img_emb;
    int routed_prefix_n = in.n_img_views;
    const int32_t * routed_mask_ptr = in.attention_mask;
    int routed_mask_n = in.attention_mask_n;
    Inputs visual_prefix_inputs = in;
    if (debug_output.empty() && !in.precomputed_img_emb && in.images && in.n_images > 0) {
        const auto tv0 = clk::now();
        int built_visual_views = 0;
        if (!run_hy_vision_frontend_224(*this, in, built_visual_tokens, built_visual_views)) {
            return {};
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(clk::now() - tv0).count();
        if (const char * path = std::getenv("VLA_HY_VLA_DUMP_VISUAL_TOKENS_F32")) {
            write_binary_file(path, built_visual_tokens.data(),
                              built_visual_tokens.size() * sizeof(float));
            std::fprintf(stderr, "vla(hy_vla): dumped visual tokens f32 [%d,49,%lld] to %s\n",
                         built_visual_views, (long long) cfg.hidden, path);
        }
        visual_prefix_inputs.precomputed_img_emb = built_visual_tokens.data();
        visual_prefix_inputs.n_img_views = built_visual_views * 49;
        visual_prefix_inputs.images = nullptr;
        visual_prefix_inputs.n_images = 0;
        if (!build_hy_vla_prefix_from_visual_tokens(*this, visual_prefix_inputs, 49,
                                                    built_prefix, built_modality_mask)) {
            return {};
        }
        routed_prefix_ptr = built_prefix.data();
        routed_prefix_n = (int) built_modality_mask.size();
        routed_mask_ptr = built_modality_mask.data();
        routed_mask_n = (int) built_modality_mask.size();
    }
    if (debug_output.empty() && in.precomputed_img_emb && in.n_img_views > 0 &&
        (!in.attention_mask || in.attention_mask_n == 0)) {
        int vis_tokens_per_image = 49;
        if (const char * e = std::getenv("VLA_HY_VLA_VIS_TOKENS_PER_IMAGE")) {
            vis_tokens_per_image = std::max(1, std::atoi(e));
        }
        if (!build_hy_vla_prefix_from_visual_tokens(*this, in, vis_tokens_per_image,
                                                    built_prefix, built_modality_mask)) {
            return {};
        }
        routed_prefix_ptr = built_prefix.data();
        routed_prefix_n = (int) built_modality_mask.size();
        routed_mask_ptr = built_modality_mask.data();
        routed_mask_n = (int) built_modality_mask.size();
    }

    if (debug_output.empty() && routed_prefix_ptr && routed_prefix_n > 0 &&
        routed_mask_ptr && routed_mask_n == routed_prefix_n) {
        if (text_layers_loaded < 1 || !prefix_vision_loaded) {
            std::fprintf(stderr, "vla(hy_vla): routed default path requires stable GGUF VLM text/vision branches and VLA_HY_VLA_TEXT_LAYERS >= 1\n");
            return {};
        }
        const int64_t n_pref = routed_prefix_n;
        const int64_t n_suf = cfg.n_suffix + 1;
        const int64_t n_total = n_pref + n_suf;
        const std::vector<HyPrefixRun> runs = make_prefix_runs(routed_mask_ptr, routed_mask_n);
        const std::vector<HyPrefixRange> visual_segments;
        const std::vector<std::vector<int64_t>> patch_groups =
            make_visual_patch_groups(routed_mask_ptr, routed_mask_n);
        if (runs.empty()) return {};
        const int routed_layers = std::max<int>(1, std::min<int>(text_layers_loaded, (int) run_layers));
        std::vector<float> pref_rows(routed_prefix_ptr,
                                     routed_prefix_ptr + (size_t) n_pref * (size_t) cfg.hidden);
        if (const char * path = std::getenv("VLA_HY_VLA_DUMP_PREFIX_F32")) {
            write_binary_file(path, pref_rows.data(), pref_rows.size() * sizeof(float));
            std::fprintf(stderr, "vla(hy_vla): dumped routed prefix f32 [%lld,%lld] to %s\n",
                         (long long) n_pref, (long long) cfg.hidden, path);
        }
        if (const char * path = std::getenv("VLA_HY_VLA_DUMP_PREFIX_MASK_I32")) {
            write_binary_file(path, routed_mask_ptr, (size_t) routed_mask_n * sizeof(int32_t));
            std::fprintf(stderr, "vla(hy_vla): dumped routed prefix mask i32 [%d] to %s\n",
                         routed_mask_n, path);
        }

        ggml_init_params rp{(size_t) 1536 * 1024 * 1024, nullptr, true};
        ggml_context * R = ggml_init(rp);
        if (!R) return {};
        ggml_tensor * t_pref = ggml_new_tensor_2d(R, GGML_TYPE_F32, cfg.hidden, n_pref); ggml_set_input(t_pref);
        ggml_tensor * t_pref_pos = ggml_new_tensor_1d(R, GGML_TYPE_I32, n_pref); ggml_set_input(t_pref_pos);
        ggml_tensor * t_pref_mask = ggml_new_tensor_2d(R, GGML_TYPE_F16, n_pref, n_pref); ggml_set_input(t_pref_mask);
        ggml_tensor * t_state = ggml_new_tensor_1d(R, GGML_TYPE_F32, max_sd); ggml_set_input(t_state);
        ggml_tensor * t_x0 = ggml_new_tensor_2d(R, GGML_TYPE_F32, max_ad, chunk); ggml_set_input(t_x0);
        std::vector<ggml_tensor *> t_time((size_t) run_steps);
        for (int i = 0; i < run_steps; ++i) {
            t_time[(size_t) i] = ggml_new_tensor_2d(R, GGML_TYPE_F32, cfg.expert_h, chunk);
            ggml_set_input(t_time[(size_t) i]);
        }
        ggml_tensor * t_suf_pos = ggml_new_tensor_1d(R, GGML_TYPE_I32, n_suf); ggml_set_input(t_suf_pos);
        ggml_tensor * t_full_mask = ggml_new_tensor_2d(R, GGML_TYPE_F16, n_total, n_suf); ggml_set_input(t_full_mask);

        ggml_tensor * hp = t_pref;
        std::vector<ggml_tensor *> k_cache((size_t) routed_layers);
        std::vector<ggml_tensor *> v_cache((size_t) routed_layers);
        for (int i = 0; i < routed_layers; ++i) {
            hp = build_hy_prefix_routed_layer(R, text_layers[(size_t) i], vision_layers[(size_t) i],
                                              hp, t_pref_pos, t_pref_mask, cfg, runs, visual_segments,
                                              true,
                                              &k_cache[(size_t) i], &v_cache[(size_t) i]);
        }

        ggml_tensor * x_t = t_x0;
        std::vector<ggml_tensor *> denoise_steps;
        denoise_steps.reserve((size_t) run_steps);
        for (int step = 0; step < run_steps; ++step) {
            ggml_tensor * hs = build_suffix_embed(R, *this, t_state, x_t, t_time[(size_t) step]);
            for (int i = 0; i < routed_layers; ++i) {
                hs = build_hy_expert_layer_with_prefix(R, layers[(size_t) i], hs, t_suf_pos,
                                                       k_cache[(size_t) i], v_cache[(size_t) i],
                                                       t_full_mask, cfg, true);
            }
            ggml_tensor * h_final = ggml_mul(R, ggml_rms_norm(R, hs, cfg.rms_eps), Wnorm);
            const size_t rb = (size_t) cfg.expert_h * sizeof(float);
            ggml_tensor * h_actions = ggml_view_2d(R, h_final, cfg.expert_h, chunk, rb, rb);
            ggml_tensor * v_t = ggml_add(R, mm_w(R, W_aout, h_actions), b_aout);
            x_t = ggml_add(R, x_t, ggml_scale(R, v_t, dt));
            if (std::getenv("VLA_HY_VLA_DUMP_DENOISE_STEPS_F32")) {
                ggml_set_output(x_t);
            }
            denoise_steps.push_back(x_t);
        }
        ggml_set_output(x_t);
        ggml_cgraph * gf = ggml_new_graph_custom(R, hy_vla_graph_size(1048576), false);
        ggml_build_forward_expand(gf, x_t);
        // On 8 GiB cards RoboTwin/curobo leaves very little headroom. Keeping the
        // routed/action scratch buffer resident can make the next vision graph
        // allocation fail even when it only needs a small CUDA block. Keep this
        // opt-in until we have a shared allocator or tighter memory planning.
        const bool persist_routed_galloc = is_cuda && env_enabled("VLA_HY_VLA_PERSIST_ROUTED_GALLOC");
        ggml_gallocr_t galloc = persist_routed_galloc ? routed_galloc : nullptr;
        if (!galloc) {
            galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (persist_routed_galloc) {
                routed_galloc = galloc;
            }
        }
        if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
            std::fprintf(stderr, "vla(hy_vla): routed default graph allocation failed\n");
            if (!persist_routed_galloc && galloc) ggml_gallocr_free(galloc);
            ggml_free(R);
            return {};
        }
        auto set_if_alloc = [](ggml_tensor * t, const void * data) {
            if (!t) return;
            ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
            if (buf) ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
        };
        set_if_alloc(t_pref, pref_rows.data());
        std::vector<int32_t> pref_pos((size_t) n_pref);
        for (int64_t i = 0; i < n_pref; ++i) pref_pos[(size_t) i] = (int32_t) i;
        set_if_alloc(t_pref_pos, pref_pos.data());
        std::vector<ggml_fp16_t> pref_mask = make_prefix_mask(n_pref, &patch_groups);
        set_if_alloc(t_pref_mask, pref_mask.data());
        std::vector<float> sh((size_t) max_sd, 0.f);
        if (in.state) std::memcpy(sh.data(), in.state, (size_t) max_sd * sizeof(float));
        if (!std::getenv("VLA_HY_VLA_RAW_STATE")) {
            for (int64_t i = 0; i < cfg.real_state_dim && i < max_sd; ++i) {
                sh[(size_t) i] = (sh[(size_t) i] - state_mean[(size_t) i]) /
                                 (state_std[(size_t) i] + cfg.norm_eps);
            }
        }
        set_if_alloc(t_state, sh.data());
        std::vector<float> x0((size_t) max_ad * (size_t) chunk);
        if (in.noise) std::memcpy(x0.data(), in.noise, x0.size() * sizeof(float));
        else {
            std::normal_distribution<float> nd(0.f, 1.f);
            for (float & v : x0) v = nd(rng);
        }
        set_if_alloc(t_x0, x0.data());
        for (int step = 0; step < run_steps; ++step) {
            const float timestep = 1.0f + (float) step * dt;
            const std::vector<float> tv = sinusoidal_time_emb(timestep, cfg.expert_h, cfg.min_period, cfg.max_period);
            std::vector<float> tile((size_t) cfg.expert_h * (size_t) chunk);
            for (int64_t c = 0; c < chunk; ++c) {
                std::memcpy(tile.data() + (size_t) c * (size_t) cfg.expert_h, tv.data(), tv.size() * sizeof(float));
            }
            set_if_alloc(t_time[(size_t) step], tile.data());
        }
        std::vector<int32_t> suf_pos((size_t) n_suf);
        for (int64_t i = 0; i < n_suf; ++i) suf_pos[(size_t) i] = (int32_t) (n_pref + i);
        set_if_alloc(t_suf_pos, suf_pos.data());
        std::vector<ggml_fp16_t> full_mask((size_t) n_total * (size_t) n_suf);
        for (int64_t r = 0; r < n_suf; ++r) {
            for (int64_t c = 0; c < n_total; ++c) {
                bool allowed = c < n_pref;
                if (!allowed) {
                    const int64_t s = c - n_pref;
                    allowed = (r == 0) ? (s == 0) : true;
                }
                full_mask[(size_t) r * (size_t) n_total + (size_t) c] =
                    ggml_fp32_to_fp16(allowed ? 0.f : -INFINITY);
            }
        }
        set_if_alloc(t_full_mask, full_mask.data());

        const auto ti0 = clk::now();
        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        stats.ms_inference = std::chrono::duration<float, std::milli>(clk::now() - ti0).count();
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "vla(hy_vla): routed default graph compute failed (%d)\n", (int) st);
            if (!persist_routed_galloc) ggml_gallocr_free(galloc);
            ggml_free(R);
            return {};
        }
        std::vector<float> out((size_t) ggml_nelements(x_t));
        ggml_backend_tensor_get(x_t, out.data(), 0, out.size() * sizeof(float));
        if (const char * path = std::getenv("VLA_HY_VLA_DUMP_DENOISE_STEPS_F32")) {
            std::vector<float> steps((size_t) run_steps * (size_t) ggml_nelements(x_t));
            const size_t step_bytes = (size_t) ggml_nelements(x_t) * sizeof(float);
            for (int step = 0; step < run_steps; ++step) {
                ggml_backend_tensor_get(denoise_steps[(size_t) step],
                                        steps.data() + (size_t) step * (size_t) ggml_nelements(x_t),
                                        0, step_bytes);
            }
            write_binary_file(path, steps.data(), steps.size() * sizeof(float));
            std::fprintf(stderr, "vla(hy_vla): dumped denoise steps f32 [%d,%lld,%lld] to %s\n",
                         run_steps, (long long) chunk, (long long) max_ad, path);
        }
        if (!persist_routed_galloc) ggml_gallocr_free(galloc);
        ggml_free(R);
        if (!std::getenv("VLA_HY_VLA_RAW_ACTION")) {
            for (int64_t t = 0; t < chunk; ++t) {
                float * row = out.data() + (size_t) t * (size_t) max_ad;
                denorm_hy_vla_action_row(*this, row, t);
            }
        }
        stats.ms_total = std::chrono::duration<float, std::milli>(clk::now() - t0).count();
        return out;
    }

    ggml_init_params cp{(size_t) 96 * 1024 * 1024, nullptr, true};
    ggml_context * C = ggml_init(cp);
    if (!C) return {};
    ggml_tensor * t_state = ggml_new_tensor_1d(C, GGML_TYPE_F32, max_sd); ggml_set_input(t_state);
    ggml_tensor * t_x0 = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk); ggml_set_input(t_x0);
    ggml_tensor * t_pos = ggml_new_tensor_1d(C, GGML_TYPE_I32, seq); ggml_set_input(t_pos);
    ggml_tensor * t_mask = ggml_new_tensor_2d(C, GGML_TYPE_F16, seq, seq); ggml_set_input(t_mask);
    std::vector<ggml_tensor *> t_time((size_t) run_steps);
    for (int i = 0; i < run_steps; ++i) {
        t_time[(size_t) i] = ggml_new_tensor_2d(C, GGML_TYPE_F32, cfg.expert_h, chunk);
        ggml_set_input(t_time[(size_t) i]);
    }

    ggml_tensor * x_t = t_x0;
    ggml_tensor * final_tensor = nullptr;
    for (int step = 0; step < run_steps; ++step) {
        ggml_tensor * h = build_suffix_embed(C, *this, t_state, x_t, t_time[(size_t) step]);
        if (debug_output == "suffix") {
            final_tensor = h;
            break;
        }
        for (int64_t li = 0; li < run_layers; ++li) h = build_hy_expert_layer(C, layers[(size_t) li], h, t_pos, t_mask, cfg);
        if (debug_output == "block") {
            final_tensor = h;
            break;
        }
        ggml_tensor * h_final = ggml_mul(C, ggml_rms_norm(C, h, cfg.rms_eps), Wnorm);
        const size_t rb = (size_t) cfg.expert_h * sizeof(float);
        ggml_tensor * h_actions = ggml_view_2d(C, h_final, cfg.expert_h, chunk, rb, rb);
        ggml_tensor * v_t = ggml_add(C, mm_w(C, W_aout, h_actions), b_aout);
        if (debug_output == "vt") {
            final_tensor = v_t;
            break;
        }
        x_t = ggml_add(C, x_t, ggml_scale(C, v_t, dt));
    }
    if (!final_tensor) final_tensor = x_t;
    ggml_set_output(final_tensor);
    ggml_cgraph * gf = ggml_new_graph_custom(C, 32768, false);
    ggml_build_forward_expand(gf, final_tensor);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "vla(hy_vla): graph allocation failed\n");
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }

    std::vector<float> sh((size_t) max_sd, 0.f);
    if (in.state) std::memcpy(sh.data(), in.state, (size_t) max_sd * sizeof(float));
    if (!std::getenv("VLA_HY_VLA_RAW_STATE")) {
        for (int64_t i = 0; i < cfg.real_state_dim && i < max_sd; ++i) {
            sh[(size_t) i] = (sh[(size_t) i] - state_mean[(size_t) i]) /
                             (state_std[(size_t) i] + cfg.norm_eps);
        }
    }
    ggml_backend_tensor_set(t_state, sh.data(), 0, ggml_nbytes(t_state));

    std::vector<float> x0((size_t) max_ad * (size_t) chunk);
    if (in.noise) std::memcpy(x0.data(), in.noise, x0.size() * sizeof(float));
    else {
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float & v : x0) v = nd(rng);
    }
    ggml_backend_tensor_set(t_x0, x0.data(), 0, ggml_nbytes(t_x0));

    std::vector<int32_t> pos((size_t) seq);
    for (int64_t i = 0; i < seq; ++i) pos[(size_t) i] = (int32_t) i;
    if (debug_output != "suffix") {
        ggml_backend_tensor_set(t_pos, pos.data(), 0, ggml_nbytes(t_pos));
    }

    std::vector<ggml_fp16_t> mask((size_t) seq * (size_t) seq);
    for (int64_t r = 0; r < seq; ++r) {
        for (int64_t c = 0; c < seq; ++c) {
            const bool allowed = (r == 0) ? (c == 0) : true;
            const float v = allowed ? 0.f : -INFINITY;
            mask[(size_t) r * (size_t) seq + (size_t) c] = ggml_fp32_to_fp16(v);
        }
    }
    if (debug_output != "suffix") {
        ggml_backend_tensor_set(t_mask, mask.data(), 0, ggml_nbytes(t_mask));
    }

    for (int step = 0; step < run_steps; ++step) {
        const float timestep = 1.0f + (float) step * dt;
        const std::vector<float> tv = sinusoidal_time_emb(timestep, cfg.expert_h, cfg.min_period, cfg.max_period);
        std::vector<float> tile((size_t) cfg.expert_h * (size_t) chunk);
        for (int64_t c = 0; c < chunk; ++c) {
            std::memcpy(tile.data() + (size_t) c * (size_t) cfg.expert_h, tv.data(), tv.size() * sizeof(float));
        }
        ggml_backend_tensor_set(t_time[(size_t) step], tile.data(), 0, ggml_nbytes(t_time[(size_t) step]));
    }

    const auto ti0 = clk::now();
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    stats.ms_inference = std::chrono::duration<float, std::milli>(clk::now() - ti0).count();
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(hy_vla): graph compute failed (%d)\n", (int) st);
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }
    std::vector<float> out((size_t) ggml_nelements(final_tensor));
    ggml_backend_tensor_get(final_tensor, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(C);
    if (!std::getenv("VLA_HY_VLA_RAW_ACTION")) {
        for (int64_t t = 0; t < chunk; ++t) {
            float * row = out.data() + (size_t) t * (size_t) max_ad;
            denorm_hy_vla_action_row(*this, row, t);
        }
    }
    stats.ms_total = std::chrono::duration<float, std::milli>(clk::now() - t0).count();
    return out;
}

} // namespace vla
