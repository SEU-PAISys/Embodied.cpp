// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file smolvla.cpp
 * @brief LeRobot SmolVLA runtime (SigLIP + SmolLM2 + flow-matching action expert).
 *
 * Modeled on models/pi05.cpp but adapted to the SmolVLA checkpoint layout produced by
 * scripts/convert_smolvla_*_to_gguf.py. Key differences from pi0.5:
 *  * Both VLM text backbone and action expert use plain Gemma-style RMSNorm
 *    (NOT AdaRMSNorm), so one builder and one set of weights suffice.
 *  * State is embedded via a dedicated state_proj Linear and appended as one
 *    extra prefix token rather than being baked into the language tokens.
 *  * The action expert shares the head_dim / q/kv head geometry of the VLM
 *    backbone but runs at expert_h=480 (expert_width_multiplier=0.5 on the
 *    libero finetune) with its own 32-layer stack.
 *  * Time embedding is fused into the suffix via action_time_mlp_in/out (SiLU)
 *    rather than being fed as an AdaRMSNorm condition.
 */

#include "arch.h"
#include "model.h"

#include "clip.h"
#include "mtmd.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#ifndef _WIN32
#include <sys/types.h>
#endif
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace vla {

namespace {

// GGUF offsets can exceed 2 GiB; `long` is only 32-bit on Windows.
static bool seek_absolute(FILE * fp, uint64_t offset) {
#ifdef _WIN32
    return _fseeki64(fp, static_cast<__int64>(offset), SEEK_SET) == 0;
#else
    return fseeko(fp, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
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
        if (!gctx) { std::fprintf(stderr, "vla(smolvla): gguf_init_from_file failed for %s\n", path.c_str()); return false; }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp)       { std::fprintf(stderr, "vla(smolvla): fopen failed for %s\n", path.c_str()); return false; }
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

    bool has_key(const char * k) const { return gguf_find_key(gctx, k) >= 0; }
    uint32_t    u32(const char * k) const { return gguf_get_val_u32(gctx, gguf_find_key(gctx, k)); }
    float       f32(const char * k) const { return gguf_get_val_f32(gctx, gguf_find_key(gctx, k)); }
    double      f64(const char * k) const { return gguf_get_val_f64(gctx, gguf_find_key(gctx, k)); }
    std::string str(const char * k) const { return gguf_get_val_str(gctx, gguf_find_key(gctx, k)); }

    const ggml_tensor * meta(const char * name) const { return ggml_get_tensor(meta_ctx, name); }

    bool read_raw(const char * name, void * buf) const {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) { std::fprintf(stderr, "vla(smolvla): missing tensor %s\n", name); return false; }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t nb  = gguf_get_tensor_size(gctx, id);
        if (!seek_absolute(fp, off)) return false;
        return std::fread(buf, 1, nb, fp) == nb;
    }

    std::vector<uint8_t> read_convert(const char * name, ggml_type target) {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(smolvla): missing tensor %s\n", name); return {}; }
        const int64_t n = ggml_nelements(t);
        std::vector<float> f32(n);
        if (t->type == GGML_TYPE_F32) {
            if (!read_raw(name, f32.data())) return {};
        } else if (t->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp(n);
            if (!read_raw(name, tmp.data())) return {};
            ggml_bf16_to_fp32_row(tmp.data(), f32.data(), n);
        } else {
            std::fprintf(stderr, "vla(smolvla): tensor %s has unsupported type %d\n", name, (int) t->type);
            return {};
        }
        if (target == GGML_TYPE_F32) {
            std::vector<uint8_t> out(n * sizeof(float));
            std::memcpy(out.data(), f32.data(), out.size());
            return out;
        }
        if (target == GGML_TYPE_BF16) {
            std::vector<uint8_t> out(n * sizeof(ggml_bf16_t));
            ggml_fp32_to_bf16_row(f32.data(), reinterpret_cast<ggml_bf16_t *>(out.data()), n);
            return out;
        }
        std::fprintf(stderr, "vla(smolvla): unsupported resident type %d for %s\n", (int) target, name);
        return {};
    }

    bool fetch_rows_f32(const char * name, const std::vector<int32_t> & row_ids,
                        float * dst, int64_t cols) {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(smolvla): missing tensor %s\n", name); return false; }
        if (t->ne[0] != cols || t->ne[2] != 1 || t->ne[3] != 1) {
            std::fprintf(stderr, "vla(smolvla): %s shape unfit for row-fetch\n", name); return false;
        }
        const int64_t rows = t->ne[1];
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_BF16) {
            std::fprintf(stderr, "vla(smolvla): tensor %s has unsupported row type %d\n",
                         name, (int) t->type);
            return false;
        }
        const int64_t id   = gguf_find_tensor(gctx, name);
        const size_t  base = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t  elsz = (t->type == GGML_TYPE_F32) ? 4u : 2u;
        const size_t  rb   = (size_t) cols * elsz;
        std::vector<uint8_t> row(rb);
        for (size_t k = 0; k < row_ids.size(); ++k) {
            const int32_t r = row_ids[k];
            if (r < 0 || r >= rows) { std::fprintf(stderr, "vla(smolvla): row %d out of range for %s\n", r, name); return false; }
            if (!seek_absolute(fp, base + (size_t) r * rb)) return false;
            if (std::fread(row.data(), 1, rb, fp) != rb) return false;
            if (elsz == 4) std::memcpy(dst + k * cols, row.data(), rb);
            else ggml_bf16_to_fp32_row(reinterpret_cast<ggml_bf16_t *>(row.data()), dst + k * cols, cols);
        }
        return true;
    }
};

struct GemmaLayerW {
    ggml_tensor * ln_in   = nullptr;
    ggml_tensor * Wq      = nullptr;
    ggml_tensor * Wk      = nullptr;
    ggml_tensor * Wv      = nullptr;
    ggml_tensor * Wo      = nullptr;
    ggml_tensor * ln_post = nullptr;
    ggml_tensor * Wgate   = nullptr;
    ggml_tensor * Wup     = nullptr;
    ggml_tensor * Wdown   = nullptr;
};

std::vector<float> sinusoidal_time_emb(double t, int64_t dim, double min_p, double max_p) {
    const int64_t half = dim / 2;
    std::vector<float> out(dim);
    for (int64_t i = 0; i < half; ++i) {
        const double frac   = (half == 1) ? 0.0 : double(i) / double(half - 1);
        const double period = min_p * std::pow(max_p / min_p, frac);
        const double s      = (2.0 * M_PI / period) * t;
        out[i]        = (float) std::sin(s);
        out[half + i] = (float) std::cos(s);
    }
    return out;
}

// CUDA elementwise kernels do not accept BF16 operands in this ggml revision.
// Keep tensors operationally F32 while reproducing a BF16 storage round-trip.
ggml_tensor * bf16_round(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cast(ctx, ggml_cast(ctx, x, GGML_TYPE_BF16), GGML_TYPE_F32);
}

ggml_tensor * build_self_attn_layer(
        ggml_context * ctx, const GemmaLayerW & w,
        ggml_tensor * x_in, ggml_tensor * positions,
        int64_t n_q_heads, int64_t n_kv_heads, int64_t head_dim,
        int64_t seq, float rope_base, float rms_eps,
        ggml_tensor * cached_K, ggml_tensor * cached_V, ggml_tensor * mask,
        ggml_tensor ** k_out, ggml_tensor ** v_out) {
    const int64_t hd  = head_dim;
    const int64_t nq  = n_q_heads;
    const int64_t nkv = n_kv_heads;
    const int64_t qf  = nq * hd;

    // The reference model's transformer weights are BF16. PyTorch therefore
    // rounds normalized activations, projections, and residual results at each
    // layer boundary; retaining FP32 activations here causes sizeable drift
    // after 32 layers even when every matmul uses the same BF16 weights.
    ggml_tensor * x_norm = bf16_round(ctx,
        ggml_mul(ctx, ggml_rms_norm(ctx, x_in, rms_eps), w.ln_in));

    ggml_tensor * q = bf16_round(ctx, ggml_mul_mat(ctx, w.Wq, x_norm));
    ggml_tensor * k = bf16_round(ctx, ggml_mul_mat(ctx, w.Wk, x_norm));
    ggml_tensor * v = bf16_round(ctx, ggml_mul_mat(ctx, w.Wv, x_norm));

    ggml_tensor * q_h = ggml_reshape_3d(ctx, q, hd, nq,  seq);
    ggml_tensor * k_h = ggml_reshape_3d(ctx, k, hd, nkv, seq);
    ggml_tensor * v_h = ggml_reshape_3d(ctx, v, hd, nkv, seq);

    auto rope_call = [&](ggml_tensor * t) {
        ggml_tensor * rotated = ggml_rope_ext(ctx,
                             ggml_cast(ctx, t, GGML_TYPE_F32), positions, nullptr,
                             (int) hd, GGML_ROPE_TYPE_NEOX, 0,
                             rope_base, 1.f, 0.f, 1.f, 32.f, 1.f);
        return bf16_round(ctx, rotated);
    };
    ggml_tensor * q_rope = rope_call(q_h);
    ggml_tensor * k_rope = rope_call(k_h);

    if (k_out) *k_out = k_rope;
    if (v_out) *v_out = v_h;

    ggml_tensor * K_full = k_rope;
    ggml_tensor * V_full = v_h;
    if (cached_K && cached_V) {
        K_full = ggml_concat(ctx, cached_K, k_rope, 2);
        V_full = ggml_concat(ctx, cached_V, v_h,    2);
    }

    ggml_tensor * Q = ggml_cast(ctx,
        ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3)), GGML_TYPE_F32);
    ggml_tensor * K = ggml_cast(ctx,
        ggml_cont(ctx, ggml_permute(ctx, K_full, 0, 2, 1, 3)), GGML_TYPE_F32);
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, V_full, 1, 2, 0, 3));

    ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    const float scale = 1.f / std::sqrt((float) hd);
    ggml_tensor * attn = bf16_round(ctx,
        ggml_soft_max_ext(ctx, kq, mask, scale, 0.f));
    ggml_tensor * kqv  = bf16_round(ctx, ggml_mul_mat(ctx, V, attn));

    ggml_tensor * att_pre = bf16_round(ctx, ggml_reshape_2d(ctx,
        ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), qf, seq));
    ggml_tensor * o_out = bf16_round(ctx, ggml_mul_mat(ctx, w.Wo, att_pre));
    ggml_tensor * h1    = bf16_round(ctx, ggml_add(ctx, bf16_round(ctx, x_in), o_out));

    ggml_tensor * x_norm_mlp = bf16_round(ctx,
        ggml_mul(ctx, ggml_rms_norm(ctx, h1, rms_eps), w.ln_post));
    ggml_tensor * gate    = bf16_round(ctx, ggml_mul_mat(ctx, w.Wgate, x_norm_mlp));
    ggml_tensor * up      = bf16_round(ctx, ggml_mul_mat(ctx, w.Wup,   x_norm_mlp));
    ggml_tensor * inter_t = bf16_round(ctx, ggml_mul(ctx, ggml_silu(ctx, gate), up));
    ggml_tensor * mlp_out = bf16_round(ctx, ggml_mul_mat(ctx, w.Wdown, inter_t));
    return bf16_round(ctx, ggml_add(ctx, h1, mlp_out));
}

ggml_tensor * build_cross_attn_layer(
        ggml_context * ctx, const GemmaLayerW & w,
        ggml_tensor * x_in, ggml_tensor * positions,
        int64_t n_q_heads, int64_t n_kv_heads, int64_t head_dim,
        int64_t seq, int64_t prefix_seq, float rope_base, float rms_eps,
        ggml_tensor * cached_K, ggml_tensor * cached_V, ggml_tensor * mask) {
    const int64_t hd  = head_dim;
    const int64_t nq  = n_q_heads;
    const int64_t nkv = n_kv_heads;
    const int64_t qf  = nq * hd;
    const int64_t kvf = nkv * hd;

    ggml_tensor * x_norm = bf16_round(ctx,
        ggml_mul(ctx, ggml_rms_norm(ctx, x_in, rms_eps), w.ln_in));
    ggml_tensor * q = bf16_round(ctx, ggml_mul_mat(ctx, w.Wq, x_norm));

    // Cross-attention k_proj/v_proj consume flattened VLM prefix K/V.
    ggml_tensor * prefix_k = ggml_reshape_2d(ctx, cached_K, kvf, prefix_seq);
    ggml_tensor * prefix_v = ggml_reshape_2d(ctx, cached_V, kvf, prefix_seq);
    ggml_tensor * k = bf16_round(ctx, ggml_mul_mat(ctx, w.Wk, prefix_k));
    ggml_tensor * v = bf16_round(ctx, ggml_mul_mat(ctx, w.Wv, prefix_v));

    ggml_tensor * q_h = ggml_reshape_3d(ctx, q, hd, nq, seq);
    ggml_tensor * k_h = ggml_reshape_3d(ctx, k, hd, nkv, prefix_seq);
    ggml_tensor * v_h = ggml_reshape_3d(ctx, v, hd, nkv, prefix_seq);
    ggml_tensor * q_rope = bf16_round(ctx, ggml_rope_ext(ctx,
                                         ggml_cast(ctx, q_h, GGML_TYPE_F32), positions, nullptr,
                                         (int) hd, GGML_ROPE_TYPE_NEOX, 0,
                                         rope_base, 1.f, 0.f, 1.f, 32.f, 1.f));

    ggml_tensor * Q = ggml_cast(ctx,
        ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3)), GGML_TYPE_F32);
    ggml_tensor * K = ggml_cast(ctx,
        ggml_cont(ctx, ggml_permute(ctx, k_h, 0, 2, 1, 3)), GGML_TYPE_F32);
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, v_h,    1, 2, 0, 3));
    ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    ggml_tensor * attn = bf16_round(ctx,
        ggml_soft_max_ext(ctx, kq, mask, 1.f / std::sqrt((float) hd), 0.f));
    ggml_tensor * kqv  = bf16_round(ctx, ggml_mul_mat(ctx, V, attn));

    ggml_tensor * att_pre = bf16_round(ctx, ggml_reshape_2d(ctx,
        ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), qf, seq));
    ggml_tensor * o_out = bf16_round(ctx, ggml_mul_mat(ctx, w.Wo, att_pre));
    ggml_tensor * h1 = bf16_round(ctx, ggml_add(ctx, bf16_round(ctx, x_in), o_out));
    ggml_tensor * x_norm_mlp = bf16_round(ctx,
        ggml_mul(ctx, ggml_rms_norm(ctx, h1, rms_eps), w.ln_post));
    ggml_tensor * gate = bf16_round(ctx, ggml_mul_mat(ctx, w.Wgate, x_norm_mlp));
    ggml_tensor * up   = bf16_round(ctx, ggml_mul_mat(ctx, w.Wup, x_norm_mlp));
    ggml_tensor * inter_t = bf16_round(ctx,
        ggml_mul(ctx, ggml_silu(ctx, gate), up));
    ggml_tensor * mlp_out = bf16_round(ctx,
        ggml_mul_mat(ctx, w.Wdown, inter_t));
    return bf16_round(ctx, ggml_add(ctx, h1, mlp_out));
}

} // namespace

struct SmolVLAModelArch : public ModelArchBase {
    SmolVLAModelArch() : ModelArchBase(Arch::SMOLVLA) {}
    ~SmolVLAModelArch() override;

    std::vector<float> predict(const Inputs& in) override;

    clip_ctx *            cctx        = nullptr;
    ggml_backend_t        backend     = nullptr;
    bool                  is_cuda     = false;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_context *        ctx_weights = nullptr;
    std::string           ckpt_path_;
    ggml_type             matmul_type = GGML_TYPE_BF16;

    std::vector<GemmaLayerW> vlm_layers;
    std::vector<GemmaLayerW> ex_layers;

    ggml_tensor * vlm_final_norm = nullptr;
    ggml_tensor * ex_final_norm   = nullptr;

    ggml_tensor * W_state_proj = nullptr, * b_state_proj = nullptr;
    ggml_tensor * W_ain  = nullptr, * b_ain  = nullptr;
    ggml_tensor * W_aout = nullptr, * b_aout = nullptr;
    ggml_tensor * W_tmlp_in  = nullptr, * b_tmlp_in  = nullptr;
    ggml_tensor * W_tmlp_out = nullptr, * b_tmlp_out = nullptr;

    // SmolVLA vision-language connector: pixel_shuffle(scale) + Linear(768*scale^2 -> text_hidden).
    // The cheap pixel shuffle remains a host-side memory rearrangement; the large
    // projection is part of the main backend graph.
    int64_t       pixel_shuffle_scale = 4;
    ggml_tensor * W_connector = nullptr;   // (text_hidden, 768*scale^2)
    ggml_tensor * b_connector = nullptr;   // (text_hidden,) -- always zeros in upstream

    // SmolVLA's expert differs from pi0.5: keep its hyperparams separate from CFG's
    // pi0.5-oriented expert_h/expert_inter slots (which we leave at pi0.5 defaults).
    int64_t expert_n_layers    = 32;
    int64_t expert_h            = 480;
    int64_t expert_inter        = 1280;
    int64_t expert_n_q_heads    = 3;
    int64_t expert_n_kv_heads   = 1;
    int64_t expert_head_dim     = 320;
    int64_t smolvla_chunk_size  = 50;
    int64_t self_attn_every_n_layers = 2;
    int64_t smolvla_n_lang       = 48;

    std::string state_norm_mode  = "MEAN_STD";
    std::string action_norm_mode = "MEAN_STD";
    std::vector<float> state_mean, state_std, action_mean, action_std;

    // Protocol clients should send explicit noise. Keep the fallback stable so
    // identical request sequences remain reproducible for simpler clients too.
    std::mt19937 rng{0};
    int n_threads = 4;
};

SmolVLAModelArch::~SmolVLAModelArch() {
    if (cctx)        clip_free(cctx);
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend)     ggml_backend_free(backend);
}

std::vector<float> SmolVLAModelArch::predict(const Inputs& in) {
    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();
    stats = Stats{};

    const Config & cfg = this->cfg;
    const int64_t hidden_pl = cfg.hidden;
    const int64_t hidden_ex = expert_h;
    const int64_t chunk     = smolvla_chunk_size;
    const int64_t n_suf     = chunk;
    const int64_t n_vlm     = cfg.n_layers;
    const int64_t n_aex     = expert_n_layers;
    const int64_t max_ad    = cfg.max_action_dim;
    const int     num_steps = cfg.num_steps;
    const int     graph_steps = num_steps;
    const float   dt        = -1.0f / (float) num_steps;
    const float   rope_base = cfg.rope_freq_base;
    const float   rms_eps   = cfg.rms_eps;

    // 1) vision tower (SigLIP via mtmd, identity-proxy mmproj -> raw SigLIP output)
//    The mmproj ships a 768x768 identity projector so mtmd's PALIGEMMA graph
//    returns raw SigLIP features (scaled by 1/sqrt(768)). We undo that scale,
//    arrange the scale=4 pixel-shuffle input on the host, then run the large
//    Linear(12288->960) connector in the same backend graph as the language
//    backbone and action expert.
    std::vector<float> img_emb_host;       // used only for precomputed text-space embeddings
    std::vector<float> shuffled_img_host;  // connector input: (conn_in_dim, total_tokens)
    int64_t n_img_tokens = 0;
    const int64_t vision_hidden = 768;                                  // SigLIP embedding_length
    const int64_t scale         = pixel_shuffle_scale;
    const int64_t conn_in_dim   = vision_hidden * scale * scale;       // 768 * 16 = 12288
    const int64_t tokens_per_img_out = 0;                              // filled after we know npatches
    (void) tokens_per_img_out;

    if (in.precomputed_img_emb) {
        // pre-computed embeddings are already in the text-hidden space (hidden_pl),
        // so the caller is responsible for producing them; we just absorb them.
        n_img_tokens = (int64_t) in.n_img_views * cfg.n_img;
        img_emb_host.assign(in.precomputed_img_emb,
                            in.precomputed_img_emb + (size_t) n_img_tokens * hidden_pl);
    } else {
        if (in.n_images < 1 || !in.images) {
            std::fprintf(stderr, "vla(smolvla): predict: no images and no precomputed_img_emb\n");
            return {};
        }
        const int    img_sz  = clip_get_image_size(cctx);
        const size_t per_pix  = (size_t) 3 * img_sz * img_sz;
        const size_t per_out_bytes = clip_embd_nbytes_by_img(cctx, img_sz, img_sz);
        const size_t per_out_f32  = per_out_bytes / sizeof(float);
        // SigLIP output layout per image: (vision_hidden, n_patches) stored
        // row-major in our buffer (clip_encode_float_image writes a flat F32
        // vector of length vision_hidden * n_patches).
        const int64_t n_patches = (int64_t) (per_out_f32 / (size_t) vision_hidden);
        const int64_t patches_per_side = (int64_t) std::lround(std::sqrt((double) n_patches));
        if (patches_per_side * patches_per_side != n_patches) {
            std::fprintf(stderr, "vla(smolvla): SigLIP n_patches=%lld not a square\n", (long long) n_patches);
            return {};
        }
        if (patches_per_side % scale != 0) {
            std::fprintf(stderr, "vla(smolvla): SigLIP side %lld not divisible by pixel_shuffle_scale %lld\n",
                         (long long) patches_per_side, (long long) scale);
            return {};
        }
        const int64_t out_tokens_per_img = n_patches / (scale * scale);  // 64 for 1024 patches
        const float  un_scale_factor = std::sqrt((float) vision_hidden);  // undo PALIGEMMA 1/sqrt(768)

        std::vector<float> raw_vision(per_out_f32);
        std::vector<float> hwc(per_pix);
        const auto tv0 = clk::now();
        shuffled_img_host.resize((size_t) conn_in_dim * out_tokens_per_img * in.n_images);
        for (int v = 0; v < in.n_images; ++v) {
            const ImageView & view = in.images[v];
            if (view.w != img_sz || view.h != img_sz) {
                std::fprintf(stderr, "vla(smolvla): image[%d] is %dx%d; SmolVLA requires %dx%d\n",
                             v, view.w, view.h, img_sz, img_sz);
                return {};
            }
            if (view.format == PixelFormat::U8) {
                const uint8_t * src = static_cast<const uint8_t *>(view.data);
                for (size_t i = 0; i < per_pix; ++i) hwc[i] = (float) src[i] / 127.5f - 1.0f;
            } else {
                const float * src = static_cast<const float *>(view.data);
                for (size_t i = 0; i < per_pix; ++i) hwc[i] = src[i] * 2.0f - 1.0f;
            }
            if (!clip_encode_float_image(cctx, n_threads, hwc.data(), img_sz, img_sz,
                                         raw_vision.data())) {
                std::fprintf(stderr, "vla(smolvla): clip_encode_float_image failed (view %d)\n", v);
                return {};
            }

            // raw_vision is (vision_hidden, n_patches) in row-major (col=vision_hidden, row=patch).
            // Undo PALIGEMMA's 1/sqrt(vision_hidden) scale baked into the identity mmproj.
            for (float & x : raw_vision) x *= un_scale_factor;

            // Host pixel_shuffle scale x scale: for each (i, j) in the output
            // side (patches_per_side / scale), gather an scalexscale block of
            // input patches and concatenate their vision_hidden vectors into a
            // conn_in_dim-wide row.
            float * shuffled = shuffled_img_host.data() +
                (size_t) v * conn_in_dim * out_tokens_per_img;
            const int64_t in_side  = patches_per_side;
            const int64_t out_side = patches_per_side / scale;
            for (int64_t oi = 0; oi < out_side; ++oi) {
                for (int64_t oj = 0; oj < out_side; ++oj) {
                    const int64_t out_tok = oi * out_side + oj;
                    float * dst = shuffled + (size_t) out_tok * conn_in_dim;
                    int64_t slot = 0;
                    for (int64_t bi = 0; bi < scale; ++bi) {
                        for (int64_t bj = 0; bj < scale; ++bj) {
                            const int64_t in_tok = (oi * scale + bi) * in_side + (oj * scale + bj);
                            const float * src = raw_vision.data() + (size_t) in_tok * vision_hidden;
                            std::memcpy(dst + slot, src, (size_t) vision_hidden * sizeof(float));
                            slot += vision_hidden;
                        }
                    }
                }
            }

        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(clk::now() - tv0).count();
        n_img_tokens = out_tokens_per_img * (int64_t) in.n_images;
    }

    if (in.n_lang < 1 || !in.lang_tokens) {
        std::fprintf(stderr, "vla(smolvla): predict: empty lang_tokens\n");
        return {};
    }
    const int64_t n_lang = in.n_lang;
    if (in.attention_mask && in.attention_mask_n != n_lang) {
        std::fprintf(stderr,
                     "vla(smolvla): attention_mask length %d != n_lang %lld\n",
                     in.attention_mask_n, (long long) n_lang);
        return {};
    }
    if (!in.state) {
        std::fprintf(stderr, "vla(smolvla): predict: missing state (SmolVLA uses state_proj prefix token)\n");
        return {};
    }
    const int64_t n_prefix = n_img_tokens + n_lang + 1;  // img + lang + state token
    const int64_t n_total  = n_prefix + n_suf;
    std::vector<uint8_t> prefix_valid((size_t) n_prefix, 1);
    if (in.attention_mask) {
        for (int64_t i = 0; i < n_lang; ++i) {
            prefix_valid[(size_t) n_img_tokens + i] = in.attention_mask[i] != 0;
        }
    }

    // 2) language embedding rows fetched from token_embd (PI05-style row fetch)
    std::vector<int32_t> lang_ids(in.lang_tokens, in.lang_tokens + n_lang);
    std::vector<float> lang_rows((size_t) n_lang * hidden_pl);
    {
        gguf_reader g;
        if (!g.open(ckpt_path_)) return {};
        if (!g.fetch_rows_f32("token_embd.weight", lang_ids, lang_rows.data(), hidden_pl)) return {};
    }

    // 3) Normalize state on the host; state_proj itself is part of the backend graph.
    std::vector<float> state_padded((size_t) cfg.max_state_dim, 0.f);
    for (int64_t i = 0; i < cfg.real_state_dim && i < cfg.max_state_dim; ++i) {
        // normalize state before state_proj (LeRobot normalizer_processor MEAN_STD)
        if (state_norm_mode == "MEAN_STD") {
            state_padded[i] = (in.state[i] - state_mean[i]) / (state_std[i] + cfg.norm_eps);
        } else {
            state_padded[i] = in.state[i];
        }
    }
    ggml_init_params cp = { (size_t) 192 * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) { std::fprintf(stderr, "vla(smolvla): ggml_init(ctx_compute) failed\n"); return {}; }

    ggml_tensor * t_image_input = nullptr;
    if (in.precomputed_img_emb) {
        t_image_input = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_img_tokens);
    } else {
        t_image_input = ggml_new_tensor_2d(C, GGML_TYPE_F32, conn_in_dim, n_img_tokens);
    }
    ggml_set_input(t_image_input);
    ggml_tensor * t_lang_emb   = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_pl, n_lang);       ggml_set_input(t_lang_emb);
    ggml_tensor * t_state      = ggml_new_tensor_2d(C, GGML_TYPE_F32, cfg.max_state_dim, 1);    ggml_set_input(t_state);
    ggml_tensor * t_prefix_pos = ggml_new_tensor_1d(C, GGML_TYPE_I32, n_prefix);                ggml_set_input(t_prefix_pos);
    ggml_tensor * t_x0         = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);           ggml_set_input(t_x0);
    ggml_tensor * t_suffix_pos = ggml_new_tensor_1d(C, GGML_TYPE_I32, n_suf);                   ggml_set_input(t_suffix_pos);
    ggml_tensor * t_cross_pos  = ggml_new_tensor_1d(C, GGML_TYPE_I32, n_suf);                   ggml_set_input(t_cross_pos);
    ggml_tensor * t_prefix_mask= ggml_new_tensor_2d(C, GGML_TYPE_F32, n_prefix, n_prefix);      ggml_set_input(t_prefix_mask);
    ggml_tensor * t_full_mask  = ggml_new_tensor_2d(C, GGML_TYPE_F32, n_total, n_suf);          ggml_set_input(t_full_mask);
    ggml_tensor * t_cross_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, n_prefix, n_suf);         ggml_set_input(t_cross_mask);
    std::vector<ggml_tensor *> t_time(graph_steps);
    for (int s = 0; s < graph_steps; ++s) {
        t_time[s] = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden_ex, 1);
        ggml_set_input(t_time[s]);
    }

    ggml_tensor * image_embs = t_image_input;
    if (!in.precomputed_img_emb) {
        image_embs = ggml_add(C, ggml_mul_mat(C, W_connector, t_image_input), b_connector);
        image_embs = ggml_scale(C, image_embs, (float) std::sqrt((double) hidden_pl));
    }
    ggml_tensor * state_emb = ggml_add(C, ggml_mul_mat(C, W_state_proj, t_state), b_state_proj);

    // language embedding normalized by sqrt(hidden) (matches LeRobot embed_prefix)
    const float lang_scale = (float) std::sqrt((double) hidden_pl);
    ggml_tensor * img_lang    = ggml_concat(C, image_embs, ggml_scale(C, t_lang_emb, lang_scale), 1);
    ggml_tensor * prefix_embs = ggml_concat(C, img_lang, state_emb, 1);

    // VLM prefix pass builds KV cache for the action expert to cross-attend into.
    std::vector<ggml_tensor *> cK(n_vlm), cV(n_vlm);
    {
        ggml_tensor * h = prefix_embs;
        for (int64_t i = 0; i < n_vlm; ++i) {
            h = build_self_attn_layer(C, vlm_layers[i], h, t_prefix_pos,
                                      cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim,
                                      n_prefix, rope_base, rms_eps,
                                      nullptr, nullptr, t_prefix_mask,
                                      &cK[i], &cV[i]);
        }
    }

    // Flow-matching denoising loop. Each step: action_in_proj(x_t) fused with
    // sinusoidal time emb via action_time_mlp_in/out(SiLU), then cross-attention
    // through the action expert into the cached VLM prefix, final RMSNorm +
    // action_out_proj, and x_t += dt*v.
    ggml_tensor * x_t = t_x0;
    for (int step = 0; step < graph_steps; ++step) {
        ggml_tensor * a_emb = ggml_add(C, ggml_mul_mat(C, W_ain, x_t), b_ain);
        // broadcast time emb across the chunk dimension
        ggml_tensor * t_emb_b = ggml_repeat(C, t_time[step], a_emb);
        ggml_tensor * at_cat  = ggml_concat(C, a_emb, t_emb_b, 0);   // (2*expert_h, chunk)
        ggml_tensor * tmlp    = ggml_add(C, ggml_mul_mat(C, W_tmlp_in, at_cat), b_tmlp_in);
        tmlp = ggml_silu(C, tmlp);
        tmlp = ggml_add(C, ggml_mul_mat(C, W_tmlp_out, tmlp), b_tmlp_out);
        ggml_tensor * h = tmlp;
        for (int64_t i = 0; i < n_aex; ++i) {
            const bool self_attn =
                self_attn_every_n_layers > 0 && i % self_attn_every_n_layers == 0;
            if (self_attn) {
                h = build_self_attn_layer(C, ex_layers[i], h, t_suffix_pos,
                                          expert_n_q_heads, expert_n_kv_heads, expert_head_dim,
                                          n_suf, rope_base, rms_eps,
                                          cK[i], cV[i], t_full_mask,
                                          nullptr, nullptr);
            } else {
                h = build_cross_attn_layer(C, ex_layers[i], h, t_cross_pos,
                                           expert_n_q_heads, expert_n_kv_heads, expert_head_dim,
                                           n_suf, n_prefix, rope_base, rms_eps,
                                           cK[i], cV[i], t_cross_mask);
            }
        }
        // The expert final norm is BF16 in the reference model, then explicitly
        // upcast before the F32 action projection.
        ggml_tensor * h_final = bf16_round(C,
            ggml_mul(C, ggml_rms_norm(C, h, rms_eps), ex_final_norm));
        ggml_tensor * v_t = ggml_add(C, ggml_mul_mat(C, W_aout, h_final), b_aout);
        x_t = ggml_add(C, x_t, ggml_scale(C, v_t, dt));
    }
    ggml_set_output(x_t);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 1 << 16, false);
    ggml_build_forward_expand(gf, x_t);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "vla(smolvla): ggml_gallocr_alloc_graph failed (out of memory?)\n");
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }

    const std::vector<float> & image_input_host =
        in.precomputed_img_emb ? img_emb_host : shuffled_img_host;
    ggml_backend_tensor_set(t_image_input, image_input_host.data(), 0, ggml_nbytes(t_image_input));
    ggml_backend_tensor_set(t_lang_emb,  lang_rows.data(),    0, ggml_nbytes(t_lang_emb));
    ggml_backend_tensor_set(t_state, state_padded.data(), 0, ggml_nbytes(t_state));
    {
        std::vector<int32_t> pp(n_prefix);
        int32_t pos = -1;
        for (int64_t i = 0; i < n_prefix; ++i) {
            if (prefix_valid[(size_t) i]) ++pos;
            pp[i] = std::max(pos, 0);
        }
        ggml_backend_tensor_set(t_prefix_pos, pp.data(), 0, ggml_nbytes(t_prefix_pos));
        const int32_t prefix_offset = pos + 1;
        std::vector<int32_t> sp(n_suf);     for (int64_t i = 0; i < n_suf; ++i)    sp[i] = prefix_offset + (int32_t) i;
        ggml_backend_tensor_set(t_suffix_pos, sp.data(), 0, ggml_nbytes(t_suffix_pos));
        std::vector<int32_t> cp(n_suf);     for (int64_t i = 0; i < n_suf; ++i)    cp[i] = (int32_t) i;
        ggml_backend_tensor_set(t_cross_pos, cp.data(), 0, ggml_nbytes(t_cross_pos));
    }
    {
        std::vector<float> x0h((size_t) max_ad * chunk);
        if (in.noise) std::memcpy(x0h.data(), in.noise, x0h.size() * sizeof(float));
        else { std::normal_distribution<float> nd(0.f, 1.f); for (auto & v : x0h) v = nd(rng); }
        ggml_backend_tensor_set(t_x0, x0h.data(), 0, ggml_nbytes(t_x0));
    }
    {
        const float neg_inf = -INFINITY;
        // Prefix blocks: image/language queries cannot see the later state
        // token; state can see all valid image/language/state tokens.
        std::vector<float> pm((size_t) n_prefix * n_prefix, 0.f);
        for (int64_t q = 0; q < n_prefix; ++q) {
            for (int64_t k = 0; k < n_prefix; ++k) {
                if (!prefix_valid[(size_t) k] ||
                    (q < n_prefix - 1 && k == n_prefix - 1)) {
                    pm[(size_t) q * n_prefix + k] = neg_inf;
                }
            }
        }
        ggml_backend_tensor_set(t_prefix_mask, pm.data(), 0, ggml_nbytes(t_prefix_mask));

        std::vector<float> fm((size_t) n_total * n_suf, 0.f);
        std::vector<float> cm((size_t) n_prefix * n_suf, 0.f);
        for (int64_t q = 0; q < n_suf; ++q) {
            for (int64_t k = 0; k < n_prefix; ++k) {
                if (!prefix_valid[(size_t) k]) {
                    fm[(size_t) q * n_total + k] = neg_inf;
                    cm[(size_t) q * n_prefix + k] = neg_inf;
                }
            }
            // SmolVLA suffix att_masks is all ones. make_att_2d_masks()
            // therefore gives causal attention within the action chunk.
            for (int64_t k = q + 1; k < n_suf; ++k) {
                fm[(size_t) q * n_total + n_prefix + k] = neg_inf;
            }
        }
        ggml_backend_tensor_set(t_full_mask,  fm.data(), 0, ggml_nbytes(t_full_mask));
        ggml_backend_tensor_set(t_cross_mask, cm.data(), 0, ggml_nbytes(t_cross_mask));
    }
    for (int s = 0; s < graph_steps; ++s) {
        const float timestep = 1.0f + (float) s * dt;
        const std::vector<float> tv = sinusoidal_time_emb(timestep, hidden_ex, cfg.min_period, cfg.max_period);
        ggml_backend_tensor_set(t_time[s], tv.data(), 0, ggml_nbytes(t_time[s]));
    }

    const auto ti0 = clk::now();
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    stats.ms_inference = std::chrono::duration<float, std::milli>(clk::now() - ti0).count();
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(smolvla): ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }

    std::vector<float> out((size_t) chunk * max_ad);
    ggml_backend_tensor_get(x_t, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(C);
    for (int64_t t = 0; t < chunk; ++t) {
        float * row = out.data() + (size_t) t * max_ad;
        for (int64_t j = 0; j < cfg.real_action_dim && j < max_ad; ++j) {
            if (action_norm_mode == "MEAN_STD") {
                row[j] = row[j] * (action_std[j] + cfg.norm_eps) + action_mean[j];
            }
            // SmolVLA exports MEAN_STD only; no QUANTILES branch.
        }
    }

    stats.ms_total = std::chrono::duration<float, std::milli>(clk::now() - t0).count();
    return out;
}

namespace {

bool load_stats(const gguf_reader & g, SmolVLAModelArch & m) {
    auto fetch = [&](const char * name, std::vector<float> & dst, int64_t dim) -> bool {
        const ggml_tensor * t = g.meta(name);
        if (!t) return false;
        if (t->type != GGML_TYPE_F32) { std::fprintf(stderr, "vla(smolvla): stats tensor %s not F32\n", name); return false; }
        if (ggml_nelements(t) != dim) {
            std::fprintf(stderr, "vla(smolvla): stats %s dim %lld != expected %lld\n",
                         name, (long long) ggml_nelements(t), (long long) dim);
            return false;
        }
        dst.resize(dim);
        return g.read_raw(name, dst.data());
    };
    const int64_t rsd = m.cfg.real_state_dim;
    const int64_t rad = m.cfg.real_action_dim;
    if (!fetch("state_mean",  m.state_mean,  rsd)) { std::fprintf(stderr, "vla(smolvla): missing state_mean\n");  return false; }
    if (!fetch("state_std",   m.state_std,   rsd)) { std::fprintf(stderr, "vla(smolvla): missing state_std\n");   return false; }
    if (!fetch("action_mean", m.action_mean, rad)) { std::fprintf(stderr, "vla(smolvla): missing action_mean\n"); return false; }
    if (!fetch("action_std",  m.action_std,  rad)) { std::fprintf(stderr, "vla(smolvla): missing action_std\n");   return false; }
    return true;
}

} // namespace

std::unique_ptr<ModelArchBase> smolvla_create(const std::string & mmproj_path,
                                              const std::string & ckpt_path,
                                              const std::string & config_path) {
    if (!config_path.empty()) {
        std::fprintf(stderr,
                     "vla(smolvla): external config overrides are not supported; "
                     "all runtime metadata must come from the GGUF\n");
        return nullptr;
    }
    if (ckpt_path.size() < 5 || ckpt_path.compare(ckpt_path.size() - 5, 5, ".gguf") != 0) {
        std::fprintf(stderr, "vla(smolvla): ckpt must be a GGUF produced by "
                              "scripts/convert_smolvla_to_gguf.py (got '%s')\n", ckpt_path.c_str());
        return nullptr;
    }

    auto m = std::make_unique<SmolVLAModelArch>();
    m->ckpt_path_  = ckpt_path;
    m->matmul_type = std::getenv("VLA_SMOLVLA_F32_WEIGHTS") ? GGML_TYPE_F32 : GGML_TYPE_BF16;

    gguf_reader g;
    if (!g.open(ckpt_path)) return nullptr;
    if (!g.has_key("smolvla.architecture") || g.str("smolvla.architecture") != "smolvla") {
        std::fprintf(stderr, "vla(smolvla): '%s' is not a SmolVLA GGUF (smolvla.architecture missing/wrong)\n",
                     ckpt_path.c_str());
        return nullptr;
    }
    static const char * required_keys[] = {
        "smolvla.hidden", "smolvla.intermediate", "smolvla.n_q_heads",
        "smolvla.n_kv_heads", "smolvla.head_dim", "smolvla.n_layers",
        "smolvla.vocab_size", "smolvla.expert_h", "smolvla.expert_inter",
        "smolvla.expert_n_q_heads", "smolvla.expert_n_kv_heads",
        "smolvla.expert_head_dim", "smolvla.expert_n_layers",
        "smolvla.chunk_size", "smolvla.num_steps", "smolvla.max_state_dim",
        "smolvla.max_action_dim", "smolvla.real_state_dim",
        "smolvla.real_action_dim", "smolvla.tokenizer_max_length",
        "smolvla.self_attn_every_n_layers", "smolvla.pixel_shuffle_scale",
        "smolvla.min_period", "smolvla.max_period", "smolvla.rope_theta",
        "smolvla.state_norm_mode", "smolvla.action_norm_mode",
    };
    for (const char * key : required_keys) {
        if (!g.has_key(key)) {
            std::fprintf(stderr, "vla(smolvla): required GGUF metadata key missing: %s\n", key);
            return nullptr;
        }
    }

    Config & c = m->cfg;
    c.hidden          = (int64_t) g.u32("smolvla.hidden");
    c.intermediate    = (int64_t) g.u32("smolvla.intermediate");
    c.n_q_heads       = (int64_t) g.u32("smolvla.n_q_heads");
    c.n_kv_heads      = (int64_t) g.u32("smolvla.n_kv_heads");
    c.head_dim        = (int64_t) g.u32("smolvla.head_dim");
    c.n_layers        = (int64_t) g.u32("smolvla.n_layers");
    const int64_t smolvla_vocab_size = (int64_t) g.u32("smolvla.vocab_size");
    (void) smolvla_vocab_size;  // kept for future lm_head use; action expert does not need it
    m->expert_h            = (int64_t) g.u32("smolvla.expert_h");
    m->expert_inter        = (int64_t) g.u32("smolvla.expert_inter");
    m->expert_n_q_heads    = (int64_t) g.u32("smolvla.expert_n_q_heads");
    m->expert_n_kv_heads   = (int64_t) g.u32("smolvla.expert_n_kv_heads");
    m->expert_head_dim     = (int64_t) g.u32("smolvla.expert_head_dim");
    m->expert_n_layers     = (int64_t) g.u32("smolvla.expert_n_layers");
    m->smolvla_chunk_size  = (int64_t) g.u32("smolvla.chunk_size");
    c.n_suffix              = m->smolvla_chunk_size;
    c.num_steps             = (int)  g.u32("smolvla.num_steps");
    c.max_state_dim         = (int64_t) g.u32("smolvla.max_state_dim");
    c.max_action_dim        = (int64_t) g.u32("smolvla.max_action_dim");
    c.real_state_dim        = (int64_t) g.u32("smolvla.real_state_dim");
    c.real_action_dim       = (int64_t) g.u32("smolvla.real_action_dim");
    m->smolvla_n_lang       = (int64_t) g.u32("smolvla.tokenizer_max_length");
    m->self_attn_every_n_layers = (int64_t) g.u32("smolvla.self_attn_every_n_layers");
    m->pixel_shuffle_scale     = (int64_t) g.u32("smolvla.pixel_shuffle_scale");
    c.min_period            = g.f64("smolvla.min_period");
    c.max_period            = g.f64("smolvla.max_period");
    c.rope_freq_base        = (float) g.f64("smolvla.rope_theta");
    c.rms_eps               = g.has_key("smolvla.rms_norm_eps") ? g.f32("smolvla.rms_norm_eps") : 1e-6f;
    c.norm_eps              = g.has_key("smolvla.norm_eps")     ? g.f32("smolvla.norm_eps")     : 1e-8f;
    m->state_norm_mode  = g.str("smolvla.state_norm_mode");
    m->action_norm_mode = g.str("smolvla.action_norm_mode");

    if (c.hidden <= 0 || c.intermediate <= 0 || c.n_layers <= 0 ||
        c.n_q_heads <= 0 || c.n_kv_heads <= 0 || c.head_dim <= 0 ||
        m->expert_h <= 0 || m->expert_inter <= 0 || m->expert_n_layers <= 0 ||
        m->expert_n_q_heads <= 0 || m->expert_n_kv_heads <= 0 ||
        m->expert_head_dim <= 0 || m->smolvla_chunk_size <= 0 ||
        c.num_steps <= 0 || c.max_state_dim <= 0 || c.max_action_dim <= 0 ||
        c.real_state_dim <= 0 || c.real_state_dim > c.max_state_dim ||
        c.real_action_dim <= 0 || c.real_action_dim > c.max_action_dim ||
        m->smolvla_n_lang <= 0 || m->pixel_shuffle_scale <= 0 ||
        c.n_q_heads % c.n_kv_heads != 0 ||
        m->expert_n_q_heads % m->expert_n_kv_heads != 0 ||
        c.min_period <= 0.0 || c.max_period <= c.min_period ||
        c.rope_freq_base <= 0.0f) {
        std::fprintf(stderr, "vla(smolvla): invalid or inconsistent GGUF dimensions/metadata\n");
        return nullptr;
    }
    // The action expert indexes the VLM prefix KV cache by layer.  The graph
    // therefore requires a one-to-one layer mapping; accepting mismatched
    // metadata would make predict() index cK/cV out of bounds.
    if (c.n_layers != m->expert_n_layers) {
        std::fprintf(stderr,
                     "vla(smolvla): VLM layer count (%lld) must equal action-expert "
                     "layer count (%lld) for prefix KV mapping\n",
                     (long long) c.n_layers, (long long) m->expert_n_layers);
        return nullptr;
    }
    if (m->state_norm_mode != "MEAN_STD" || m->action_norm_mode != "MEAN_STD") {
        std::fprintf(stderr,
                     "vla(smolvla): unsupported normalization modes state=%s action=%s\n",
                     m->state_norm_mode.c_str(), m->action_norm_mode.c_str());
        return nullptr;
    }

    // populate pi0.5-era Config slots used by the ggml graph builder
    // 512 / 16 = 32 patches per side, then pixel_shuffle(scale=4)
    // produces 8 * 8 = 64 connector tokens per image.
    c.n_img           = 64;
    c.n_lang          = (int) m->smolvla_n_lang;
    c.q_full_dim      = c.n_q_heads  * c.head_dim;
    c.kv_full_dim     = c.n_kv_heads * c.head_dim;
    c.self_attn_every_n = 0;
    c.rope_mode       = GGML_ROPE_TYPE_NEOX;
    c.rope_n_dims     = (int) c.head_dim;
    c.n_prefix        = 0;
    c.n_full          = 0;
    c.expert_h        = m->expert_h;       // share so stats logging is consistent
    c.expert_inter    = m->expert_inter;

    std::printf("vla(smolvla): hidden=%lld inter=%lld heads=%lldq/%lldkv x%lld n_vlm=%lld "
                "expert_h=%lld expert_inter=%lld n_aex=%lld "
                "chunk=%lld steps=%d real_state=%lld real_action=%lld "
                "state_norm=%s action_norm=%s matmul_weights=%s\n",
                (long long) c.hidden, (long long) c.intermediate,
                (long long) c.n_q_heads, (long long) c.n_kv_heads, (long long) c.head_dim,
                (long long) c.n_layers,
                (long long) m->expert_h, (long long) m->expert_inter, (long long) m->expert_n_layers,
                (long long) m->smolvla_chunk_size, c.num_steps,
                (long long) c.real_state_dim, (long long) c.real_action_dim,
                m->state_norm_mode.c_str(), m->action_norm_mode.c_str(),
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

    // backend
    {
        const unsigned hw = std::thread::hardware_concurrency();
        m->n_threads = (hw == 0) ? 4 : (int) std::min(hw, 8u);
    }
    m->is_cuda = false;
#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) { m->is_cuda = true; std::printf("vla(smolvla): backend = CUDA (device 0)\n"); }
    else            { std::fprintf(stderr, "vla(smolvla): ggml_backend_cuda_init failed; falling back to CPU\n"); }
#endif
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) { std::fprintf(stderr, "vla(smolvla): ggml_backend_cpu_init failed\n"); return nullptr; }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(smolvla): backend = CPU (%d threads)\n", m->n_threads);
    }

    // vision tower (SigLIP via mtmd)
    {
        clip_context_params cp = {};
        cp.use_gpu           = m->is_cuda;
        cp.flash_attn_type   = m->is_cuda ? CLIP_FLASH_ATTN_TYPE_AUTO : CLIP_FLASH_ATTN_TYPE_DISABLED;
        cp.image_min_tokens  = -1;
        cp.image_max_tokens  = -1;
        cp.warmup            = m->is_cuda;
        cp.cb_eval           = nullptr;
        cp.cb_eval_user_data = nullptr;
        clip_init_result r = clip_init(mmproj_path.c_str(), cp);
        if (!r.ctx_v) {
            std::fprintf(stderr, "vla(smolvla): clip_init failed for %s\n", mmproj_path.c_str());
            return nullptr;
        }
        m->cctx = r.ctx_v;
        const int img_sz  = clip_get_image_size(m->cctx);
        const int mm_embd = clip_n_mmproj_embd(m->cctx);
        std::printf("vla(smolvla): mmproj image_size=%d mmproj_embd=%d (model hidden=%lld, "
                    "identity-proxy passthrough -> connector matmul in policy graph)\n",
                    img_sz, mm_embd, (long long) c.hidden);
        // SmolVLA uses an identity-proxy mmproj (mmproj_embd == SigLIP vision hidden = 768)
        // because the real connector (pixel_shuffle + 768*16->960 Linear) is applied by the
        // policy path inside predict(). So we only require mmproj_embd to match the SigLIP vision
        // embedding length, not the text-backbone hidden.
        if (img_sz != 512) {
            std::fprintf(stderr, "vla(smolvla): mmproj image size %d != expected 512\n", img_sz);
            return nullptr;
        }
        if (mm_embd != 768) {
            std::fprintf(stderr, "vla(smolvla): mmproj embd %d != expected SigLIP hidden 768 "
                                 "(the identity-proxy mmproj should passthrough raw SigLIP)\n", mm_embd);
            return nullptr;
        }
    }

    // weight context
    {
        ggml_init_params wp = { (size_t) 32 * 1024 * 1024, nullptr, true };
        m->ctx_weights = ggml_init(wp);
        if (!m->ctx_weights) { std::fprintf(stderr, "vla(smolvla): ggml_init(ctx_weights) failed\n"); return nullptr; }
    }
    ggml_context * W = m->ctx_weights;
    std::vector<ggml_tensor *> weights;

    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) { std::fprintf(stderr, "vla(smolvla): missing tensor %s\n", name); return nullptr; }
        ggml_tensor * t = ggml_new_tensor(W, type, GGML_MAX_DIMS, gt->ne);
        ggml_set_name(t, name);
        weights.push_back(t);
        return t;
    };
    auto mk_mm  = [&](const char * name) -> ggml_tensor * { return mk(name, m->matmul_type); };
    auto mk_f32 = [&](const char * name) -> ggml_tensor * { return mk(name, GGML_TYPE_F32); };

    auto load_layer = [&](const char * tower, int64_t i, GemmaLayerW & lw) -> bool {
        char b[256];
        auto suf = [&](const char * s) { std::snprintf(b, sizeof(b), "%s.blk.%lld.%s", tower, (long long) i, s); return b; };
        lw.ln_in   = mk_f32(suf("attn_norm.weight"));
        lw.Wq      = mk_mm (suf("attn_q.weight"));
        lw.Wk      = mk_mm (suf("attn_k.weight"));
        lw.Wv      = mk_mm (suf("attn_v.weight"));
        lw.Wo      = mk_mm (suf("attn_o.weight"));
        lw.ln_post = mk_f32(suf("ffn_norm.weight"));
        lw.Wgate   = mk_mm (suf("ffn_gate.weight"));
        lw.Wup     = mk_mm (suf("ffn_up.weight"));
        lw.Wdown   = mk_mm (suf("ffn_down.weight"));
        return lw.ln_in && lw.Wq && lw.Wk && lw.Wv && lw.Wo && lw.ln_post && lw.Wgate && lw.Wup && lw.Wdown;
    };
    m->vlm_layers.resize(c.n_layers);
    m->ex_layers.resize(m->expert_n_layers);
    for (int64_t i = 0; i < c.n_layers; ++i) {
        if (!load_layer("vlm", i, m->vlm_layers[i])) return nullptr;
    }
    for (int64_t i = 0; i < m->expert_n_layers; ++i) {
        if (!load_layer("aex", i, m->ex_layers[i])) return nullptr;
    }
    m->vlm_final_norm = mk_f32("vlm.output_norm.weight");
    m->ex_final_norm  = mk_f32("aex.output_norm.weight");
    m->W_ain  = mk_f32("action_in_proj.weight");       m->b_ain  = mk_f32("action_in_proj.bias");
    m->W_aout = mk_f32("action_out_proj.weight");      m->b_aout = mk_f32("action_out_proj.bias");
    m->W_tmlp_in  = mk_f32("action_time_mlp_in.weight");  m->b_tmlp_in  = mk_f32("action_time_mlp_in.bias");
    m->W_tmlp_out = mk_f32("action_time_mlp_out.weight"); m->b_tmlp_out = mk_f32("action_time_mlp_out.bias");
    m->W_connector = mk_f32("connector.weight");
    m->b_connector = mk_f32("connector.bias");
    m->W_state_proj = mk_f32("state_proj.weight");
    m->b_state_proj = mk_f32("state_proj.bias");
    for (ggml_tensor * t : weights) if (!t) { std::fprintf(stderr, "vla(smolvla): weight tensor creation failed\n"); return nullptr; }
    if (!m->vlm_final_norm || !m->ex_final_norm ||
        !m->W_ain || !m->b_ain || !m->W_aout || !m->b_aout ||
        !m->W_tmlp_in || !m->b_tmlp_in || !m->W_tmlp_out || !m->b_tmlp_out ||
        !m->W_connector || !m->b_connector || !m->W_state_proj || !m->b_state_proj) {
        std::fprintf(stderr, "vla(smolvla): failed to wire projection / norm tensors\n"); return nullptr;
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) { std::fprintf(stderr, "vla(smolvla): ggml_backend_alloc_ctx_tensors failed (out of memory?)\n"); return nullptr; }
    for (ggml_tensor * t : weights) {
        std::vector<uint8_t> bytes = g.read_convert(t->name, t->type);
        if (bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(smolvla): upload size mismatch for %s (%zu vs %zu)\n",
                         t->name, bytes.size(), ggml_nbytes(t));
            return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }

    // Validate connector dimensions used by the backend graph.
    {
        const ggml_tensor * wt = g.meta("connector.weight");
        const ggml_tensor * bt = g.meta("connector.bias");
        if (!wt || !bt) {
            std::fprintf(stderr, "vla(smolvla): connector.weight/bias missing in GGUF\n"); return nullptr;
        }
        const int64_t out_dim = (int64_t) wt->ne[1];   // ggml tensor ne=[cols, rows,...]; PyTorch (out,in) -> ne[0]=in, ne[1]=out
        const int64_t in_dim  = (int64_t) wt->ne[0];
        const int64_t expect_in = 768 * m->pixel_shuffle_scale * m->pixel_shuffle_scale;
        if (out_dim != c.hidden || in_dim != expect_in || wt->ne[2] != 1 || wt->ne[3] != 1 ||
            ggml_nelements(bt) != out_dim || bt->ne[1] != 1 || bt->ne[2] != 1 || bt->ne[3] != 1) {
            std::fprintf(stderr,
                         "vla(smolvla): connector shape weight=(%lld,%lld) bias=%lld; "
                         "expected=(%lld,%lld) bias=%lld\n",
                         (long long) out_dim, (long long) in_dim,
                         (long long) ggml_nelements(bt), (long long) c.hidden,
                         (long long) expect_in, (long long) c.hidden);
            return nullptr;
        }
        std::printf("vla(smolvla): connector resident backend: out_dim=%lld in_dim=%lld pixel_shuffle_scale=%lld\n",
                    (long long) out_dim, (long long) in_dim, (long long) m->pixel_shuffle_scale);
    }

    std::printf("vla(smolvla): resident weights = %.2f GiB\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0));

    if (!load_stats(g, *m)) return nullptr;
    std::printf("vla(smolvla): model loaded (n_threads=%d)\n", m->n_threads);
    return m;
}

} // namespace vla
