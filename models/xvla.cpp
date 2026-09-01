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
 * @file xvla.cpp
 * @brief X-VLA runtime (Florence-2 DaViT vision tower + BART text encoder +
 *        SoftPromptedTransformer flow-matching action head).
 *
 * Layout follows scripts/convert_xvla_to_gguf.py; the whole model lives in
 * one GGUF. Reference numerical contract (2toINF/X-VLA, WidowX checkpoint):
 *  * Vision: 224x224 RGB, ImageNet mean/std. DaViT (depths 1,1,9,1; dims
 *    256/512/1024/2048; window size 12) with depthwise 3x3 convs around every
 *    window/channel attention and FFN. Stage convs: k7s4p3 then k3s2p1 x3,
 *    with a LayerNorm before the conv when patch_prenorm[s] else after it.
 *    Per view the tower emits 7x7=49 patch tokens (dim 2048); after adding
 *    learned 2D positions (col ids first half, row ids second) and the cosine
 *    temporal row 0, features are [spatial_avg_pool (1 global token);
 *    temporal_avg_pool (identity at T=1, i.e. the 49 patches)] -> 50 tokens
 *    per view, projected to 1024 by image_projection + image_proj_norm.
 *  * Text: BartTokenizer ids padded to 50 with pad id 1. Embeddings unscaled
 *    (scale_embedding=false), learned positions at id+2, layernorm_embedding,
 *    then the first view's 50 image tokens are prepended -> 100 encoder
 *    tokens through 12 post-LN BART layers (exact-erf GELU; the reference
 *    merge path attends with an all-ones mask, so no mask here either).
 *  * Action: SoftPromptedTransformer, hidden 1024, 24 pre-LN blocks
 *    (qkv_bias, tanh-approx GELU). Tokens = [30 noisy-action tokens encoded
 *    by a DomainAwareLinear(20 act + 20 proprio + 32 time), projected VLM
 *    features, projected auxiliary-view features, 32 domain soft prompts] +
 *    learned positions. The decoder reads the 30 action positions through a
 *    second DomainAwareLinear. Denoise: x_t = noise*t + prev*(1-t), t from
 *    steps..1/steps (linear schedule, sinusoidal time embedding with
 *    max_period=100); gripper channels 9/19 are zeroed in x_t/proprio before
 *    encoding and sigmoid-ed on the final output (ee6d space).
 *
 * All views run through the tower as one ggml batch dimension; window
 * attention treats (view, window) pairs as independent batch heads.
 */

#include "arch.h"
#include "model.h"

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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace vla {

namespace {

constexpr float kLnEps = 1e-5f;   // torch nn.LayerNorm default

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
        if (!gctx) { std::fprintf(stderr, "vla(xvla): gguf_init_from_file failed for %s\n", path.c_str()); return false; }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp)       { std::fprintf(stderr, "vla(xvla): fopen failed for %s\n", path.c_str()); return false; }
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
    uint8_t     u8 (const char * k) const { return gguf_get_val_u8 (gctx, gguf_find_key(gctx, k)); }
    std::string str(const char * k) const { return gguf_get_val_str(gctx, gguf_find_key(gctx, k)); }

    const ggml_tensor * meta(const char * name) const { return ggml_get_tensor(meta_ctx, name); }

    bool read_raw(const char * name, void * buf) const {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) { std::fprintf(stderr, "vla(xvla): missing tensor %s\n", name); return false; }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t nb  = gguf_get_tensor_size(gctx, id);
        if (!seek_absolute(fp, off)) return false;
        return std::fread(buf, 1, nb, fp) == nb;
    }

    std::vector<uint8_t> read_convert(const char * name, ggml_type target) {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(xvla): missing tensor %s\n", name); return {}; }
        const int64_t n = ggml_nelements(t);
        std::vector<float> f32(n);
        if (t->type == GGML_TYPE_F32) {
            if (!read_raw(name, f32.data())) return {};
        } else if (t->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp(n);
            if (!read_raw(name, tmp.data())) return {};
            ggml_bf16_to_fp32_row(tmp.data(), f32.data(), n);
        } else {
            std::fprintf(stderr, "vla(xvla): tensor %s has unsupported type %d\n", name, (int) t->type);
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
        if (target == GGML_TYPE_Q8_0 || target == GGML_TYPE_Q4_K) {
            // Load-time weight quantization: the GGUF stays F32, residency is
            // quantized. Q4_K rows need n_per_row % QK_K; smaller rows (early
            // DaViT stages) fall back to Q8_0 for that tensor.
            ggml_type eff = target;
            const int64_t n_per_row = t->ne[0];
            if (eff == GGML_TYPE_Q4_K && n_per_row % ggml_blck_size(GGML_TYPE_Q4_K) != 0) {
                std::fprintf(stderr, "vla(xvla): %s ne[0]=%lld not Q4_K-aligned, using Q8_0\n",
                             name, (long long) n_per_row);
                eff = GGML_TYPE_Q8_0;
            }
            const int64_t nrows = t->ne[1] * t->ne[2] * t->ne[3];
            std::vector<uint8_t> out(ggml_row_size(eff, n_per_row) * (size_t) nrows);
            ggml_quantize_chunk(eff, f32.data(), out.data(), 0, nrows, n_per_row, nullptr);
            return out;
        }
        std::fprintf(stderr, "vla(xvla): unsupported resident type %d for %s\n", (int) target, name);
        return {};
    }

    bool fetch_rows_f32(const char * name, const std::vector<int32_t> & row_ids,
                        float * dst, int64_t cols) const {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(xvla): missing tensor %s\n", name); return false; }
        if (t->ne[0] != cols || t->ne[2] != 1 || t->ne[3] != 1) {
            std::fprintf(stderr, "vla(xvla): %s shape unfit for row-fetch\n", name); return false;
        }
        const int64_t rows = t->ne[1];
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_BF16) {
            std::fprintf(stderr, "vla(xvla): tensor %s has unsupported row type %d\n",
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
            if (r < 0 || r >= rows) { std::fprintf(stderr, "vla(xvla): row %d out of range for %s\n", r, name); return false; }
            if (!seek_absolute(fp, base + (size_t) r * rb)) return false;
            if (std::fread(row.data(), 1, rb, fp) != rb) return false;
            if (elsz == 4) std::memcpy(dst + k * cols, row.data(), rb);
            else ggml_bf16_to_fp32_row(reinterpret_cast<ggml_bf16_t *>(row.data()), dst + k * cols, cols);
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// Weight structs (GGUF tensor names in parentheses).
// ---------------------------------------------------------------------------
struct WB { ggml_tensor * w = nullptr; ggml_tensor * b = nullptr; };

struct DavitBlockW {         // vit.s{S}.p{J}.{sp,ch}.*
    WB conv1, norm, qkv, proj, conv2, ffn_norm, fc1, fc2;
};

struct EncLayerW {           // text.blk.N.* (post-LN BART)
    WB attn_q, attn_k, attn_v, attn_o;
    WB attn_norm, ffn_up, ffn_down, ffn_norm;
};

struct ActBlockW {           // act.blk.N.* (pre-LN)
    WB norm1, qkv, proj, norm2, fc1, fc2;
};

// ---------------------------------------------------------------------------
// ggml graph helpers. Activations are F32; big matmul weights are BF16.
// ---------------------------------------------------------------------------
ggml_tensor * ln_f32(ggml_context * ctx, ggml_tensor * x, const WB & w, float eps) {
    // CUDA's element-wise mul only accepts F32/F16 as src1 (binbcast.cu) —
    // LayerNorm scales must therefore be resident in F32, never BF16.
    GGML_ASSERT(w.w->type == GGML_TYPE_F32 && "ln_f32: scale must be F32");
    if (w.b) GGML_ASSERT(w.b->type == GGML_TYPE_F32 && "ln_f32: bias must be F32");
    ggml_tensor * n = ggml_norm(ctx, x, eps);
    n = ggml_mul(ctx, n, w.w);
    return ggml_add(ctx, n, w.b);
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const WB & w) {
    ggml_tensor * y = ggml_mul_mat(ctx, w.w, x);
    if (w.b) y = ggml_add(ctx, y, w.b);
    return y;
}

// Feature rows [off, off+n) of a 2D [in, seq] tensor along the innermost
// (feature) axis, as a contiguous copy (used to split fused qkv).
ggml_tensor * row_slice(ggml_context * ctx, ggml_tensor * t, int64_t off, int64_t n) {
    ggml_tensor * v = ggml_view_2d(ctx, t, n, t->ne[1], t->nb[1], (size_t) off * t->nb[0]);
    return ggml_cont(ctx, v);
}

// Batched softmax multi-head attention.
//   q: [head_dim, H, Nq, B]   k: [head_dim, H, Nk, B]   v: [head_dim, H, Nk, B]
//   mask: additive [Nk, Nq] or NULL. Returns [head_dim, H, Nq, B].
ggml_tensor * mha(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                  ggml_tensor * mask, float scale) {
    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));  // [hd, Nq, H, B]
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));  // [hd, Nk, H, B]
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [Nk, hd, H, B]
    ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);                          // [Nk, Nq, H, B]
    kq = ggml_scale(ctx, kq, scale);
    if (mask) {
        ggml_tensor * m4 = ggml_reshape_4d(ctx, mask, mask->ne[0], mask->ne[1], 1, 1);
        kq = ggml_add(ctx, kq, m4);
    }
    ggml_tensor * p = ggml_soft_max(ctx, kq);
    ggml_tensor * o = ggml_mul_mat(ctx, V, p);                           // [hd, Nq, H, B]
    return ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));             // [hd, H, Nq, B]
}

// [C, W*H*V] token matrix -> [W, H, C, V] image batch. Token memory has the
// channel dim fastest while the conv wants x fastest, so this is a real
// data permutation (reshape to [C,W,H,V], then move channels last).
ggml_tensor * img_from_tokens(ggml_context * ctx, ggml_tensor * tok,
                              int64_t W, int64_t H) {
    const int64_t V = tok->ne[1] / (W * H);
    ggml_tensor * r = ggml_reshape_4d(ctx, tok, tok->ne[0], W, H, V);  // [C,W,H,V]
    ggml_tensor * p = ggml_permute(ctx, r, 2, 0, 1, 3);                // -> [W,H,C,V]
    return ggml_cont(ctx, p);
}

// [W, H, C, V] image batch -> [C, W*H*V] tokens (view-major, x-fastest).
// NOTE: ggml_permute scatters axes (ne[axis_i] = a->ne[i]).
ggml_tensor * tokens_from_img(ggml_context * ctx, ggml_tensor * img) {
    ggml_tensor * p = ggml_permute(ctx, img, 1, 2, 0, 3);                // [C, W, H, V]
    ggml_tensor * c = ggml_cont(ctx, p);
    return ggml_reshape_2d(ctx, c, c->ne[0], c->ne[1] * c->ne[2] * c->ne[3]);
}

// Depthwise 3x3 conv (stride/pad/dilation 1) on [C, W*H*V] tokens, with bias.
ggml_tensor * dwconv_tokens(ggml_context * ctx, ggml_tensor * tok,
                            const WB & w, int64_t W, int64_t H) {
    ggml_tensor * img = img_from_tokens(ctx, tok, W, H);
    // direct variant keeps everything F32 (the im2col-based ggml_conv_2d_dw
    // hardcodes an F16 im2col, which costs visible accuracy across the 24
    // depthwise convs of the tower).
    ggml_tensor * o = ggml_conv_2d_dw_direct(ctx, w.w, img, 1, 1, 1, 1, 1, 1);
    if (w.b) {
        o = ggml_add(ctx, o,
            ggml_reshape_4d(ctx, w.b, 1, 1, w.b->ne[0], 1));
    }
    return tokens_from_img(ctx, o);
}

// Sinusoidal time embedding used by SoftPromptedTransformer (max_period=100):
// emb = [cos(t*freqs) | sin(t*freqs)], freqs[f] = exp(-ln(100)*f/half).
static void time_embedding(float t, int64_t dim, float * dst) {
    const int64_t half = dim / 2;
    for (int64_t f = 0; f < half; ++f) {
        const double freq = std::exp(-std::log(100.0) * (double) f / (double) half);
        const double arg  = (double) t * freq;
        dst[f]        = (float) std::cos(arg);
        dst[half + f] = (float) std::sin(arg);
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
struct XVLAModelArch : public ModelArchBase {
    XVLAModelArch() : ModelArchBase(Arch::XVLA) {}
    ~XVLAModelArch() override;

    std::vector<float> predict(const Inputs& in) override;

    ggml_backend_t        backend     = nullptr;
    bool                  is_cuda     = false;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_context *        ctx_weights = nullptr;
    ggml_type             matmul_type = GGML_TYPE_BF16;

    // geometry
    int64_t hidden        = 1024;  // action-transformer width
    int64_t depth         = 24;
    int64_t heads         = 16;
    int64_t num_domains   = 30;
    int64_t soft_prompts  = 32;
    int64_t dim_time      = 32;
    int64_t max_len_seq   = 512;
    int64_t num_actions   = 30;
    int64_t dim_action    = 20;
    int64_t dim_proprio   = 20;
    int64_t grip0         = 9;
    int64_t grip1         = 19;
    int64_t denoise_steps = 10;
    int64_t text_len      = 50;

    int64_t d_model       = 1024;  // BART encoder width
    int64_t enc_layers    = 12;
    int64_t enc_heads     = 16;
    int64_t vocab         = 51289;
    int64_t max_pos       = 4096;

    int64_t image_size    = 224;
    int64_t window_size   = 12;
    int64_t proj_dim      = 1024;
    int64_t davit_depths[4]  = {1, 1, 9, 1};
    int64_t davit_dims[4]    = {256, 512, 1024, 2048};
    int64_t davit_heads[4]   = {8, 16, 32, 64};
    int64_t davit_groups[4]  = {8, 16, 32, 64};
    int64_t davit_patch[4]   = {7, 3, 3, 3};
    int64_t davit_stride[4]  = {4, 2, 2, 2};
    int64_t davit_padding[4] = {3, 1, 1, 1};
    bool    davit_prenorm[4] = {false, true, true, true};

    int64_t grid          = 7;     // final feature map side (image_size/16)
    int64_t n_patches     = 49;    // grid*grid
    int64_t img_tokens    = 50;    // 1 global + n_patches

    // vision tower
    std::vector<std::vector<DavitBlockW>> davit;  // [stage][pair*2+kind]
    WB conv_k[4], conv_nb[4];                     // stage conv kernel/bias, norm
    ggml_tensor * vproj_pos_row = nullptr;        // [proj-half, 50]
    ggml_tensor * vproj_pos_col = nullptr;        // [proj-half, 50]
    ggml_tensor * vproj_temporal = nullptr;       // [2048]
    WB vproj_proj, vproj_proj_norm;

    // text encoder
    ggml_tensor * tok_emb = nullptr;              // [d_model, vocab]
    ggml_tensor * pos_emb = nullptr;              // [d_model, max_pos+2]
    WB emb_norm;
    std::vector<EncLayerW> enc;

    // action transformer
    WB act_vlm_proj, act_aux_proj;
    ggml_tensor * act_pos = nullptr;              // [hidden, max_len_seq]
    WB act_out_norm;
    ggml_tensor * aenc_fc = nullptr;              // [domains, in*hidden]
    ggml_tensor * aenc_b  = nullptr;              // [domains, hidden]
    ggml_tensor * adec_fc = nullptr;              // [domains, hidden*dim_action]
    ggml_tensor * adec_b  = nullptr;              // [domains, dim_action]
    ggml_tensor * prompts = nullptr;              // [domains, soft_prompts*hidden]
    std::vector<ActBlockW> act_blk;

    std::unique_ptr<gguf_reader> reader;          // kept open: embedding lookups
    int n_threads = 4;
};

XVLAModelArch::~XVLAModelArch() {
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend)     ggml_backend_free(backend);
}

std::vector<float> XVLAModelArch::predict(const Inputs& in) {
    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();
    stats = Stats{};

    const int64_t V = in.n_images;

    // ---------------- 0) input validation ----------------
    if (V < 1 || V > 3 || !in.images) {
        std::fprintf(stderr, "vla(xvla): predict needs 1..3 images (got %lld)\n", (long long) V);
        return {};
    }
    if (!in.lang_tokens || in.n_lang <= 0) {
        std::fprintf(stderr, "vla(xvla): predict needs lang_tokens "
                             "(tokenize client-side with BartTokenizer)\n");
        return {};
    }
    if (!in.noise) {
        std::fprintf(stderr, "vla(xvla): predict needs noise (%lld floats) "
                             "for the flow-matching loop\n",
                     (long long) (num_actions * dim_action));
        return {};
    }

    // ---------------- 1) host-side preprocessing ----------------
    // 1a) pixels -> normalized planes laid out as [W, H, C*V] (the graph input
    // is reshaped to a [W, H, 3, V] image batch inside the graph).
    const int64_t S = image_size;
    std::vector<float> pixels((size_t) S * S * 3 * V);
    {
        static const float mean3[3] = {0.485f, 0.456f, 0.406f};
        static const float std3[3]  = {0.229f, 0.224f, 0.225f};
        for (int64_t v = 0; v < V; ++v) {
            const ImageView & view = in.images[v];
            if (view.w != (int) S || view.h != (int) S) {
                std::fprintf(stderr, "vla(xvla): image[%lld] is %dx%d, expected %lldx%lld "
                                     "(resize client-side, bicubic)\n",
                             (long long) v, view.w, view.h, (long long) S, (long long) S);
                return {};
            }
            float * dst = pixels.data() + (size_t) v * S * S * 3;
            for (int64_t c = 0; c < 3; ++c) {
                float * plane = dst + (size_t) c * S * S;
                for (int64_t y = 0; y < S; ++y) {
                    for (int64_t x = 0; x < S; ++x) {
                        float px;
                        if (view.format == PixelFormat::U8) {
                            const uint8_t * src = static_cast<const uint8_t *>(view.data);
                            px = (float) src[(y * S + x) * 3 + c] / 255.0f;
                        } else {
                            const float * src = static_cast<const float *>(view.data);
                            px = src[(y * S + x) * 3 + c];
                        }
                        plane[y * S + x] = (px - mean3[c]) / std3[c];
                    }
                }
            }
        }
    }

    // 1b) BART encoder input rows: shared token embedding + learned position
    // (id + 2). Padding keeps whatever ids the client supplied.
    std::vector<int32_t> ids(in.lang_tokens, in.lang_tokens + in.n_lang);
    if ((int64_t) ids.size() > text_len) ids.resize((size_t) text_len);
    while ((int64_t) ids.size() < text_len) ids.push_back(1);   // pad_token_id
    std::vector<float> bert_emb((size_t) d_model * text_len);
    if (!reader->fetch_rows_f32("text.tok_emb", ids, bert_emb.data(), d_model)) {
        std::fprintf(stderr, "vla(xvla): token embedding lookup failed\n");
        return {};
    }
    {
        // Official flow: merge image+text embeds FIRST, then add learned
        // positions over the whole merged sequence (pos id = index + 2), so
        // the leading img_tokens rows take pos[2 .. 2+img_tokens) and the
        // text rows take pos[2+img_tokens .. 2+img_tokens+text_len).
        std::vector<int32_t> pos_ids((size_t) text_len);
        for (int64_t i = 0; i < text_len; ++i)
            pos_ids[(size_t) i] = (int32_t) (img_tokens + i + 2);
        std::vector<float> pos_rows((size_t) d_model * text_len);
        if (!reader->fetch_rows_f32("text.pos_emb", pos_ids, pos_rows.data(), d_model)) {
            std::fprintf(stderr, "vla(xvla): position embedding lookup failed\n");
            return {};
        }
        for (size_t i = 0; i < bert_emb.size(); ++i) bert_emb[i] += pos_rows[i];
    }
    // Position rows consumed by the image-token block of the merged sequence.
    std::vector<float> bert_img_pos((size_t) d_model * img_tokens);
    {
        std::vector<int32_t> pos_ids((size_t) img_tokens);
        for (int64_t i = 0; i < img_tokens; ++i) pos_ids[(size_t) i] = (int32_t) (i + 2);
        if (!reader->fetch_rows_f32("text.pos_emb", pos_ids,
                                    bert_img_pos.data(), d_model)) {
            std::fprintf(stderr, "vla(xvla): image position embedding lookup failed\n");
            return {};
        }
    }

    // 1c) flow-matching constants: sinusoidal time embeddings for every step,
    // initial noise, proprio with gripper channels zeroed, channel mask.
    std::vector<float> time_embs((size_t) denoise_steps * dim_time);
    for (int64_t s = 0; s < denoise_steps; ++s) {
        time_embedding((float) (denoise_steps - s) / (float) denoise_steps,
                       dim_time, time_embs.data() + (size_t) s * dim_time);
    }
    const int64_t nch = num_actions * dim_action;
    std::vector<float> noise((const float *) in.noise,
                             (const float *) in.noise + nch);
    std::vector<float> proprio((size_t) dim_proprio, 0.f);
    for (int64_t i = 0; i < dim_proprio && in.state; ++i) proprio[(size_t) i] = in.state[i];
    proprio[(size_t) grip0] = 0.f;
    proprio[(size_t) grip1] = 0.f;
    std::vector<float> ch_mask((size_t) dim_action, 1.f);
    ch_mask[(size_t) grip0] = 0.f;
    ch_mask[(size_t) grip1] = 0.f;

    // 1d) fixed index tables: 2D position ids for the patch grid and the
    // window partition permutations for every stage.
    std::vector<int32_t> pos_x((size_t) n_patches), pos_y((size_t) n_patches);
    for (int64_t y = 0; y < grid; ++y)
        for (int64_t x = 0; x < grid; ++x) {
            pos_x[(size_t) (y * grid + x)] = (int32_t) x;
            pos_y[(size_t) (y * grid + x)] = (int32_t) y;
        }
    struct WinPerm { std::vector<int32_t> fwd, inv; };
    std::vector<WinPerm> win_perms(4);
    for (int64_t s = 0; s < 4; ++s) {
        int64_t hw = S;
        for (int64_t q = 0; q <= s; ++q) {
            hw = (hw + 2 * davit_padding[q] - davit_patch[q]) / davit_stride[q] + 1;
        }
        const int64_t ws   = window_size;
        const int64_t nwx  = (hw + ws - 1) / ws;
        const int64_t hp   = nwx * ws;
        WinPerm & wp = win_perms[(size_t) s];
        wp.fwd.assign((size_t) hp * hp, 0);
        wp.inv.assign((size_t) hw * hw, -1);
        int64_t out = 0;
        for (int64_t wy = 0; wy < nwx; ++wy)
            for (int64_t wx = 0; wx < nwx; ++wx)
                for (int64_t iy = 0; iy < ws; ++iy)
                    for (int64_t ix = 0; ix < ws; ++ix) {
                        const int64_t x = wx * ws + ix, y = wy * ws + iy;
                        const int64_t src = (y < hw && x < hw) ? (x + y * hw) : -1;
                        wp.fwd[(size_t) out] = (int32_t) (src >= 0 ? src : 0);
                        if (src >= 0) wp.inv[(size_t) src] = (int32_t) out;
                        ++out;
                    }
    }

    // ---------------- 2) build the graph ----------------
    ggml_init_params cp = { (size_t) 512 * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) { std::fprintf(stderr, "vla(xvla): ggml_init(ctx_compute) failed\n"); return {}; }

    const int64_t ahd = hidden / heads;        // 64
    const int64_t ehd = d_model / enc_heads;   // 64

    // Host->device uploads are deferred until after ggml_gallocr_alloc_graph:
    // input tensors have no storage before the graph allocator runs. The
    // upload owns a copy of the host bytes because several source vectors are
    // block-scoped and die before the deferred upload executes.
    struct Upload { ggml_tensor * t; std::vector<uint8_t> bytes; };
    std::vector<Upload> uploads;
    auto inp_f32 = [&](const std::vector<float> & host, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ggml_new_tensor_2d(C, GGML_TYPE_F32, ne0, ne1);
        ggml_set_input(t);
        const uint8_t * p = reinterpret_cast<const uint8_t *>(host.data());
        uploads.push_back({t, std::vector<uint8_t>(p, p + ggml_nbytes(t))});
        return t;
    };
    auto inp_i32 = [&](const std::vector<int32_t> & host, int64_t n) {
        ggml_tensor * t = ggml_new_tensor_1d(C, GGML_TYPE_I32, n);
        ggml_set_input(t);
        const uint8_t * p = reinterpret_cast<const uint8_t *>(host.data());
        uploads.push_back({t, std::vector<uint8_t>(p, p + ggml_nbytes(t))});
        return t;
    };
    // ggml_get_rows requires the ids' batch dims to match src0 exactly, so
    // per-view index tables are uploaded as [n, V] matrices.
    auto inp_i32_2d = [&](const std::vector<int32_t> & host, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ggml_new_tensor_2d(C, GGML_TYPE_I32, ne0, ne1);
        ggml_set_input(t);
        const uint8_t * p = reinterpret_cast<const uint8_t *>(host.data());
        uploads.push_back({t, std::vector<uint8_t>(p, p + ggml_nbytes(t))});
        return t;
    };

    ggml_tensor * t_pixels = inp_f32(pixels, S, S * 3 * V);  // reshaped in-graph
    ggml_tensor * t_bert   = inp_f32(bert_emb, d_model, text_len);
    ggml_tensor * t_imgpos = inp_f32(bert_img_pos, d_model, img_tokens);
    ggml_tensor * t_noise  = inp_f32(noise, dim_action, num_actions);
    ggml_tensor * t_prop   = inp_f32(proprio, dim_proprio, 1);
    ggml_tensor * t_mask   = inp_f32(ch_mask, dim_action, 1);
    ggml_tensor * t_times  = inp_f32(time_embs, dim_time, denoise_steps);

    ggml_tensor * t_domain = inp_i32({in.domain_id}, 1);

    // ---- 2a) DaViT vision tower (all views as one batch) ----
    ggml_tensor * vis = nullptr;
    {
        ggml_tensor * imgs = ggml_reshape_4d(C, t_pixels, S, S, 3, V);
        int64_t hw = S, cin = 3;
        for (int64_t s = 0; s < 4; ++s) {
            ggml_tensor * cur;
            if (s == 0) {
                cur = imgs;
                if (davit_prenorm[s]) {
                    // stage 0 pre-norm would sit on raw RGB; not used by this
                    // checkpoint but kept for config fidelity.
                    cur = ln_f32(C, cur, conv_nb[s], kLnEps);
                }
            } else {
                ggml_tensor * toks = ggml_reshape_3d(C, vis, cin, vis->ne[1] / V, V);
                if (davit_prenorm[s]) toks = ln_f32(C, toks, conv_nb[s], kLnEps);
                // [cin, L, V] -> [L, cin, V] planes -> [hw, hw, cin, V] images
                ggml_tensor * planes = ggml_cont(C, ggml_permute(C, toks, 1, 0, 2, 3));
                cur = ggml_reshape_4d(C, planes, hw, hw, cin, V);
            }
            ggml_tensor * o = ggml_conv_2d(C, conv_k[s].w, cur,
                                           (int) davit_stride[s], (int) davit_stride[s],
                                           (int) davit_padding[s], (int) davit_padding[s], 1, 1);
            if (conv_k[s].b) {
                // image layout [W,H,C,V]: bias must broadcast from dim C
                o = ggml_add(C, o,
                    ggml_reshape_4d(C, conv_k[s].b, 1, 1, conv_k[s].b->ne[0], 1));
            }
            vis = tokens_from_img(C, o);
            if (!davit_prenorm[s]) vis = ln_f32(C, vis, conv_nb[s], kLnEps);
            hw  = (hw + 2 * davit_padding[s] - davit_patch[s]) / davit_stride[s] + 1;
            cin = davit_dims[s];

            const int64_t L    = hw * hw;
            const int64_t ws   = window_size;
            const int64_t nwin = ((hw + ws - 1) / ws) * ((hw + ws - 1) / ws);
            const int64_t lpad = nwin * ws * ws;
            const WinPerm & wp = win_perms[(size_t) s];

            std::vector<int32_t> fwd_v((size_t) lpad * V), inv_v((size_t) L * V);
            for (int64_t v = 0; v < V; ++v) {
                std::memcpy(fwd_v.data() + (size_t) v * lpad, wp.fwd.data(), (size_t) lpad * 4);
                std::memcpy(inv_v.data() + (size_t) v * L, wp.inv.data(), (size_t) L * 4);
            }
            ggml_tensor * t_fwd = inp_i32_2d(fwd_v, lpad, V);
            ggml_tensor * t_inv = inp_i32_2d(inv_v, L, V);

            for (int64_t j = 0; j < davit_depths[s]; ++j) {
                for (int kind = 0; kind < 2; ++kind) {
                    const DavitBlockW & w = davit[(size_t) s][(size_t) (j * 2 + kind)];
                    // PreNorm(None, DepthWiseConv2d): residual dwconv.
                    vis = ggml_add(C, vis, dwconv_tokens(C, vis, w.conv1, hw, hw));
                    ggml_tensor * h = ln_f32(C, vis, w.norm, kLnEps);
                    ggml_tensor * attn_out;
                    if (kind == 0) {
                        // ---- window attention ----
                        ggml_tensor * xp = ggml_reshape_3d(C, h, cin, L, V);
                        if (lpad != L) xp = ggml_pad(C, xp, 0, lpad - L, 0, 0);
                        xp = ggml_get_rows(C, xp, t_fwd);                // window order
                        // Padded window slots were gathered from row 0; the
                        // reference keeps them at zero before the qkv projection,
                        // so restore that here (bias-only rows downstream).
                        if (lpad != L) {
                            const int64_t nwx = (hw + ws - 1) / ws;
                            std::vector<float> rmask((size_t) lpad * V, 0.f);
                            int64_t out = 0;
                            for (int64_t wy = 0; wy < nwx; ++wy)
                                for (int64_t wx = 0; wx < nwx; ++wx)
                                    for (int64_t iy = 0; iy < ws; ++iy)
                                        for (int64_t ix = 0; ix < ws; ++ix, ++out) {
                                            const int64_t x = wx * ws + ix, y = wy * ws + iy;
                                            if (y < hw && x < hw)
                                                for (int64_t v = 0; v < V; ++v)
                                                    rmask[(size_t) v * lpad + out] = 1.f;
                                        }
                            xp = ggml_mul(C, xp,
                                ggml_reshape_3d(C, inp_f32(rmask, lpad, V), 1, lpad, V));
                        }
                        ggml_tensor * qkv = linear(C, xp, w.qkv);        // [3C, lpad, V]
                        qkv = ggml_reshape_2d(C, qkv, 3 * cin, lpad * V);
                        const int64_t dh = cin / davit_heads[s];
                        ggml_tensor * q  = row_slice(C, qkv, 0, cin);
                        ggml_tensor * k  = row_slice(C, qkv, cin, cin);
                        ggml_tensor * vv = row_slice(C, qkv, 2 * cin, cin);
                        ggml_tensor * o = mha(C,
                            ggml_reshape_4d(C, q,  dh, davit_heads[s], ws * ws, nwin * V),
                            ggml_reshape_4d(C, k,  dh, davit_heads[s], ws * ws, nwin * V),
                            ggml_reshape_4d(C, vv, dh, davit_heads[s], ws * ws, nwin * V),
                            nullptr, 1.0f / std::sqrt((float) dh));
                        o = ggml_reshape_2d(C, o, cin, lpad * V);
                        o = linear(C, o, w.proj);
                        o = ggml_reshape_3d(C, o, cin, lpad, V);
                        o = ggml_get_rows(C, o, t_inv);                  // back to hw order
                        attn_out = ggml_reshape_2d(C, o, cin, L * V);
                    } else {
                        // ---- channel attention (grouped over channels) ----
                        const int64_t G  = davit_groups[s];
                        const int64_t dg = cin / G;
                        ggml_tensor * qkv = linear(C, h, w.qkv);         // [3C, L*V]
                        ggml_tensor * q  = row_slice(C, qkv, 0, cin);
                        ggml_tensor * k  = row_slice(C, qkv, cin, cin);
                        ggml_tensor * vv = row_slice(C, qkv, 2 * cin, cin);
                        // [C, L*V] -> [dg, L, G, V] with channels grouped
                        auto group4 = [&](ggml_tensor * t) {
                            ggml_tensor * g4 = ggml_reshape_4d(C, t, dg, G, L, V);
                            return ggml_cont(C, ggml_permute(C, g4, 0, 2, 1, 3)); // [dg, L, G, V]
                        };
                        ggml_tensor * q4 = group4(q), * k4 = group4(k), * v4 = group4(vv);
                        // Channel attention: the reduction axis is TOKENS (L),
                        // producing a dg x dg attention per (group, view).
                        // Token-major copies put L on the mul_mat K axis.
                        auto tok_major = [&](ggml_tensor * t) {
                            return ggml_cont(C, ggml_permute(C, t, 1, 0, 2, 3)); // [L, dg, G, V]
                        };
                        ggml_tensor * qT = tok_major(q4);
                        ggml_tensor * kT = tok_major(k4);
                        // scores[dg_k, dg_q, G, V] = k . q / sqrt(L);
                        // ne[0] is the key axis so softmax normalises over it.
                        ggml_tensor * sc = ggml_mul_mat(C, kT, qT);      // [dg, dg, G, V]
                        sc = ggml_scale(C, sc, 1.0f / std::sqrt((float) L));
                        sc = ggml_soft_max(C, sc);
                        // out[dg_q, L, G, V] = attn . v
                        ggml_tensor * o = ggml_mul_mat(C, sc, v4);       // [dg, L, G, V]
                        // regroup to [dg, G, L, V]; its memory layout already equals
                        // channel-major [C, L*V] with c = g*dg + d, so a plain
                        // reshape merges the groups before the output projection.
                        o = ggml_cont(C, ggml_permute(C, o, 0, 2, 1, 3));
                        o = ggml_reshape_2d(C, o, cin, L * V);
                        o = linear(C, o, w.proj);
                        attn_out = o;
                    }
                    vis = ggml_add(C, vis, attn_out);
                    // PreNorm(None, DepthWiseConv2d): residual dwconv.
                    vis = ggml_add(C, vis, dwconv_tokens(C, vis, w.conv2, hw, hw));
                    h = ln_f32(C, vis, w.ffn_norm, kLnEps);
                    ggml_tensor * f = linear(C, h, w.fc1);
                    f = ggml_gelu_erf(C, f);
                    f = linear(C, f, w.fc2);
                    vis = ggml_add(C, vis, f);
                }
            }
        }
        // vis: [2048, n_patches*V]
    }

    // ---- 2b) per-view feature assembly + image projection ----
    ggml_tensor * img_feats = nullptr;   // [proj_dim, img_tokens*V]
    {
        ggml_tensor * toks = ggml_reshape_3d(C, vis, davit_dims[3], n_patches, V);
        // learned 2D positions: col ids occupy the first half of the dim
        ggml_tensor * px = inp_i32(pos_x, n_patches);
        ggml_tensor * py = inp_i32(pos_y, n_patches);
        ggml_tensor * pe = ggml_concat(C,
            ggml_get_rows(C, vproj_pos_col, px),
            ggml_get_rows(C, vproj_pos_row, py), 0);                         // [2048, P]
        pe = ggml_repeat(C, ggml_reshape_3d(C, pe, davit_dims[3], n_patches, 1),
                         ggml_new_tensor_3d(C, GGML_TYPE_F32, davit_dims[3], n_patches, V));
        toks = ggml_add(C, toks, pe);
        toks = ggml_add(C, toks, vproj_temporal);                            // temporal row 0
        // spatial average over patches per view: mean = ones/P . patches
        std::vector<float> onesv((size_t) n_patches, 1.0f / (float) n_patches);
        ggml_tensor * t_ones = inp_f32(onesv, n_patches, 1);
        ggml_tensor * bt = ggml_cont(C, ggml_permute(C, toks, 1, 0, 2, 3));  // [P, 2048, V]
        ggml_tensor * mean = ggml_mul_mat(C, t_ones, bt);                    // [1, 2048, V]
        mean = ggml_reshape_3d(C, mean, davit_dims[3], 1, V);
        // view-major concat: [global | patches] per view
        ggml_tensor * g_t = ggml_cont(C, ggml_permute(C, mean, 0, 2, 1, 3)); // [2048, V, 1]
        ggml_tensor * p_t = ggml_cont(C, ggml_permute(C, toks, 0, 2, 1, 3)); // [2048, V, P]
        ggml_tensor * feats = ggml_concat(C, g_t, p_t, 2);                   // [2048, V, 1+P]
        // reorder to (view, position)-major so a plain flatten yields
        // [2048, V*(1+P)] with each view's [global | patches] block contiguous
        feats = ggml_cont(C, ggml_permute(C, feats, 0, 2, 1, 3));            // [2048, 1+P, V]
        feats = ggml_reshape_2d(C, feats, davit_dims[3], img_tokens * V);
        ggml_tensor * pr = ggml_mul_mat(C, vproj_proj.w, feats);             // [proj_dim, V*T]
        img_feats = ln_f32(C, pr, vproj_proj_norm, kLnEps);
    }

    // ---- 2c) BART text encoder ----
    ggml_tensor * enc_out;
    {
        ggml_tensor * img0 = ggml_view_2d(C, img_feats, proj_dim, img_tokens,
                                          img_feats->nb[1], 0);              // first view
        // BART applies layernorm_embedding to the whole merged sequence
        // (image + text embeds) AFTER adding learned positions over all
        // tokens (pos id = sequence index + 2).
        ggml_tensor * x = ln_f32(C,
            ggml_concat(C, ggml_add(C, img0, t_imgpos), t_bert, 1),
                                 emb_norm, kLnEps);                  // [d_model, 100]
        for (int64_t i = 0; i < enc_layers; ++i) {
            const EncLayerW & w = enc[(size_t) i];
            ggml_tensor * q = linear(C, x, w.attn_q);
            ggml_tensor * k = linear(C, x, w.attn_k);
            ggml_tensor * v = linear(C, x, w.attn_v);
            ggml_tensor * o = mha(C,
                ggml_reshape_4d(C, q, ehd, enc_heads, x->ne[1], 1),
                ggml_reshape_4d(C, k, ehd, enc_heads, x->ne[1], 1),
                ggml_reshape_4d(C, v, ehd, enc_heads, x->ne[1], 1),
                nullptr, 1.0f / std::sqrt((float) ehd));
            o = ggml_reshape_2d(C, o, d_model, x->ne[1]);
            o = linear(C, o, w.attn_o);
            x = ln_f32(C, ggml_add(C, x, o), w.attn_norm, kLnEps);           // post-LN
            ggml_tensor * f = linear(C, x, w.ffn_up);
            f = ggml_gelu_erf(C, f);
            f = linear(C, f, w.ffn_down);
            x = ln_f32(C, ggml_add(C, x, f), w.ffn_norm, kLnEps);
        }
        enc_out = x;
    }

    // ---- 2d) project VLM streams into the action-transformer width ----
    ggml_tensor * vlm_tok = linear(C, enc_out, act_vlm_proj);                // [hidden, 100]
    ggml_tensor * aux_tok = nullptr;
    if (V > 1) {
        ggml_tensor * aux = ggml_view_2d(C, img_feats, proj_dim, img_tokens * (V - 1),
                                         img_feats->nb[1],
                                         (size_t) img_tokens * img_feats->nb[1]);
        aux_tok = linear(C, aux, act_aux_proj);                              // [hidden, T*(V-1)]
    }
    // The official processor pads views to num_views=3; masked-out view slots
    // keep a zero feature block that becomes an aux_proj(bias-only) token
    // block after projection and still occupies attention key/value slots.
    {
        const int64_t aux_slots = 2;                        // num_views(3) - first view
        const int64_t aux_valid = V > 1 ? V - 1 : 0;
        if (aux_valid < aux_slots) {
            ggml_tensor * b = ggml_reshape_2d(C, act_aux_proj.b, hidden, 1);
            ggml_tensor * blk = ggml_repeat(C, b,
                ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, img_tokens));
            ggml_tensor * z = blk;
            for (int64_t k = 1; k < aux_slots - aux_valid; ++k)
                z = ggml_concat(C, z, blk, 1);
            aux_tok = aux_tok ? ggml_concat(C, aux_tok, z, 1) : z;
        }
    }

    // ---- 2e) flow-matching denoise loop (unrolled in-graph) ----
    ggml_tensor * prev = ggml_scale(C, t_noise, 0.0f);                       // zeros
    for (int64_t s = 0; s < denoise_steps; ++s) {
        const float t = (float) (denoise_steps - s) / (float) denoise_steps;
        ggml_tensor * x_t = ggml_add(C,
            ggml_scale(C, t_noise, t),
            ggml_scale(C, prev, 1.0f - t));
        x_t = ggml_mul(C, x_t, t_mask);                                      // zero gripper
        ggml_tensor * prop = ggml_repeat(C, t_prop,
            ggml_new_tensor_2d(C, GGML_TYPE_F32, dim_proprio, num_actions));
        ggml_tensor * tim = ggml_view_2d(C, t_times, dim_time, 1, t_times->nb[1],
                                         (size_t) s * t_times->nb[1]);
        tim = ggml_repeat(C, tim,
            ggml_new_tensor_2d(C, GGML_TYPE_F32, dim_time, num_actions));
        ggml_tensor * toks = ggml_concat(C, ggml_concat(C, x_t, prop, 0), tim, 0);  // [72, 30]

        // DomainAwareLinear action encoder: gather this domain's row.
        ggml_tensor * dec_in = ggml_get_rows(C, aenc_fc, t_domain);          // [in*h, 1]
        dec_in = ggml_reshape_2d(C, dec_in, dim_action + dim_proprio + dim_time, hidden);
        ggml_tensor * a = ggml_mul_mat(C, dec_in, toks);                     // [hidden, 30]
        a = ggml_add(C, a, ggml_get_rows(C, aenc_b, t_domain));

        // assemble the full sequence -- positions cover only the real tokens:
        // official SoftPromptedTransformer appends soft prompts AFTER the
        // pos_emb add, so prompt slots receive no positional embedding.
        ggml_tensor * seq = ggml_concat(C, a, vlm_tok, 1);
        if (aux_tok) seq = ggml_concat(C, seq, aux_tok, 1);
        const int64_t slen_main = seq->ne[1];
        ggml_tensor * pe = ggml_view_2d(C, act_pos, hidden, slen_main,
                                        act_pos->nb[1], 0);
        seq = ggml_add(C, seq, ggml_cont(C, pe));
        ggml_tensor * spr = ggml_reshape_2d(C, ggml_get_rows(C, prompts, t_domain),
                                            hidden, soft_prompts);
        seq = ggml_concat(C, seq, spr, 1);
        const int64_t slen = seq->ne[1];
        if (slen > max_len_seq) {
            std::fprintf(stderr, "vla(xvla): sequence %lld exceeds max_len_seq %lld\n",
                         (long long) slen, (long long) max_len_seq);
            ggml_free(C);
            return {};
        }

        for (int64_t i = 0; i < depth; ++i) {
            const ActBlockW & w = act_blk[(size_t) i];
            ggml_tensor * hh = ln_f32(C, seq, w.norm1, kLnEps);
            ggml_tensor * qkv = linear(C, hh, w.qkv);
            ggml_tensor * q  = row_slice(C, qkv, 0, hidden);
            ggml_tensor * k  = row_slice(C, qkv, hidden, hidden);
            ggml_tensor * v  = row_slice(C, qkv, 2 * hidden, hidden);
            ggml_tensor * o = mha(C,
                ggml_reshape_4d(C, q, ahd, heads, slen, 1),
                ggml_reshape_4d(C, k, ahd, heads, slen, 1),
                ggml_reshape_4d(C, v, ahd, heads, slen, 1),
                nullptr, 1.0f / std::sqrt((float) ahd));
            o = ggml_reshape_2d(C, o, hidden, slen);
            seq = ggml_add(C, seq, linear(C, o, w.proj));
            hh = ln_f32(C, seq, w.norm2, kLnEps);
            ggml_tensor * f = linear(C, hh, w.fc1);
            // Exact tanh-approximation GELU (transformer.py Mlp):
            //   x * sigmoid(2*sqrt(2/pi) * (x + 0.044715 x^3))
            // (ggml_gelu_quick is the different 1.702x sigmoid form.)
            {
                ggml_tensor * xc = ggml_mul(C, f, f);
                xc = ggml_mul(C, xc, f);                                 // x^3
                xc = ggml_scale(C, ggml_add(C, f, ggml_scale(C, xc, 0.044715f)),
                                1.5957691216f);                          // 2*sqrt(2/pi)
                f = ggml_mul(C, f, ggml_sigmoid(C, xc));
            }
            f = linear(C, f, w.fc2);
            seq = ggml_add(C, seq, f);
        }

        // decode the 30 action positions
        ggml_tensor * head = ggml_view_2d(C, seq, hidden, num_actions, seq->nb[1], 0);
        head = ln_f32(C, ggml_cont(C, head), act_out_norm, kLnEps);
        ggml_tensor * dw = ggml_get_rows(C, adec_fc, t_domain);
        dw = ggml_reshape_2d(C, dw, hidden, dim_action);
        ggml_tensor * pred = ggml_mul_mat(C, dw, head);                      // [20, 30]
        prev = ggml_add(C, pred, ggml_get_rows(C, adec_b, t_domain));
    }
    ggml_set_output(prev);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 1 << 17, false);
    ggml_build_forward_expand(gf, prev);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "vla(xvla): ggml_gallocr_alloc_graph failed (out of memory?)\n");
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }

    for (const Upload & u : uploads) {
        ggml_backend_tensor_set(u.t, u.bytes.data(), 0, u.bytes.size());
    }

    const auto ti0 = clk::now();
    const ggml_status stt = ggml_backend_graph_compute(backend, gf);
    if (stt != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(xvla): ggml_backend_graph_compute failed (%d)\n", (int) stt);
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }
    stats.ms_inference = std::chrono::duration<float, std::milli>(clk::now() - ti0).count();

    // [dim_action, num_actions] column-major get == [step, dim] flatten
    std::vector<float> out((size_t) num_actions * dim_action);
    ggml_backend_tensor_get(prev, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(C);

    // ee6d postprocess: sigmoid on the two gripper logits.
    for (int64_t s = 0; s < num_actions; ++s) {
        float * row = out.data() + (size_t) s * dim_action;
        row[grip0] = 1.0f / (1.0f + std::exp(-row[grip0]));
        row[grip1] = 1.0f / (1.0f + std::exp(-row[grip1]));
    }

    stats.ms_total = std::chrono::duration<float, std::milli>(clk::now() - t0).count();
    return out;
}



} // namespace (close anonymous; the factory is the public vla:: symbol)

// ---------------------------------------------------------------------------
// Factory: metadata + weights -> backend-resident model
// ---------------------------------------------------------------------------
std::unique_ptr<ModelArchBase> xvla_create(const std::string & mmproj_path,
                                           const std::string & ckpt_path,
                                           const std::string & config_path) {
    if (!mmproj_path.empty()) {
        std::fprintf(stderr, "vla(xvla): mmproj is not used (vision tower is baked into the ckpt)\n");
    }
    if (!config_path.empty()) {
        std::fprintf(stderr,
                     "vla(xvla): external config overrides are not supported; "
                     "all runtime metadata must come from the GGUF\n");
        return nullptr;
    }
    if (ckpt_path.size() < 5 || ckpt_path.compare(ckpt_path.size() - 5, 5, ".gguf") != 0) {
        std::fprintf(stderr, "vla(xvla): ckpt must be a GGUF produced by "
                              "scripts/convert_xvla_to_gguf.py (got '%s')\n", ckpt_path.c_str());
        return nullptr;
    }

    auto m = std::make_unique<XVLAModelArch>();
    m->reader = std::make_unique<gguf_reader>();
    if (!m->reader->open(ckpt_path)) return nullptr;
    gguf_reader & g = *m->reader;
    if (!g.has_key("xvla.architecture") || g.str("xvla.architecture") != "xvla") {
        std::fprintf(stderr, "vla(xvla): '%s' is not an X-VLA GGUF (xvla.architecture missing/wrong)\n",
                     ckpt_path.c_str());
        return nullptr;
    }
    m->matmul_type = std::getenv("VLA_XVLA_F32_WEIGHTS") ? GGML_TYPE_F32 : GGML_TYPE_BF16;
    if (const char * q = std::getenv("VLA_XVLA_QUANT")) {
        if (!std::strcmp(q, "Q8_0"))      m->matmul_type = GGML_TYPE_Q8_0;
        else if (!std::strcmp(q, "Q4_K")) m->matmul_type = GGML_TYPE_Q4_K;
        else std::fprintf(stderr, "vla(xvla): unknown VLA_XVLA_QUANT '%s' (use Q8_0|Q4_K)\n", q);
    }

    static const char * required_keys[] = {
        "xvla.num_actions", "xvla.dim_action", "xvla.dim_proprio", "xvla.denoise_steps",
        "xvla.gripper_idx_0", "xvla.gripper_idx_1", "xvla.hidden", "xvla.depth",
        "xvla.heads", "xvla.num_domains", "xvla.soft_prompts",
        "xvla.dim_time", "xvla.max_len_seq", "xvla.text_len", "xvla.d_model",
        "xvla.enc_layers", "xvla.enc_heads", "xvla.vocab",
        "xvla.max_pos", "xvla.image_size", "xvla.window_size", "xvla.projection_dim",
    };
    for (const char * key : required_keys) {
        if (!g.has_key(key)) {
            std::fprintf(stderr, "vla(xvla): required GGUF metadata key missing: %s\n", key);
            return nullptr;
        }
    }

    m->num_actions   = (int64_t) g.u32("xvla.num_actions");
    m->dim_action    = (int64_t) g.u32("xvla.dim_action");
    m->dim_proprio   = (int64_t) g.u32("xvla.dim_proprio");
    m->denoise_steps = (int64_t) g.u32("xvla.denoise_steps");
    m->grip0         = (int64_t) g.u32("xvla.gripper_idx_0");
    m->grip1         = (int64_t) g.u32("xvla.gripper_idx_1");
    m->hidden        = (int64_t) g.u32("xvla.hidden");
    m->depth         = (int64_t) g.u32("xvla.depth");
    m->heads         = (int64_t) g.u32("xvla.heads");
    m->num_domains   = (int64_t) g.u32("xvla.num_domains");
    m->soft_prompts  = (int64_t) g.u32("xvla.soft_prompts");
    m->dim_time      = (int64_t) g.u32("xvla.dim_time");
    m->max_len_seq   = (int64_t) g.u32("xvla.max_len_seq");
    m->text_len      = (int64_t) g.u32("xvla.text_len");
    m->d_model       = (int64_t) g.u32("xvla.d_model");
    m->enc_layers    = (int64_t) g.u32("xvla.enc_layers");
    m->enc_heads     = (int64_t) g.u32("xvla.enc_heads");
    m->vocab         = (int64_t) g.u32("xvla.vocab");
    m->max_pos       = (int64_t) g.u32("xvla.max_pos");
    m->image_size    = (int64_t) g.u32("xvla.image_size");
    m->window_size   = (int64_t) g.u32("xvla.window_size");
    m->proj_dim      = (int64_t) g.u32("xvla.projection_dim");
    for (int i = 0; i < 4; ++i) {
        char key[64];
        std::snprintf(key, sizeof(key), "xvla.davit_depth_%d", i);
        m->davit_depths[i] = (int64_t) g.u32(key);
        std::snprintf(key, sizeof(key), "xvla.davit_dim_%d", i);
        m->davit_dims[i] = (int64_t) g.u32(key);
        std::snprintf(key, sizeof(key), "xvla.davit_heads_%d", i);
        m->davit_heads[i] = (int64_t) g.u32(key);
        std::snprintf(key, sizeof(key), "xvla.davit_groups_%d", i);
        m->davit_groups[i] = (int64_t) g.u32(key);
        std::snprintf(key, sizeof(key), "xvla.davit_patch_%d", i);
        m->davit_patch[i] = (int64_t) g.u32(key);
        std::snprintf(key, sizeof(key), "xvla.davit_stride_%d", i);
        m->davit_stride[i] = (int64_t) g.u32(key);
        std::snprintf(key, sizeof(key), "xvla.davit_padding_%d", i);
        m->davit_padding[i] = (int64_t) g.u32(key);
        std::snprintf(key, sizeof(key), "xvla.davit_prenorm_%d", i);
        m->davit_prenorm[i] = g.u8(key) != 0;
    }

    // derived geometry
    int64_t hw = m->image_size;
    for (int s = 0; s < 4; ++s) {
        hw = (hw + 2 * m->davit_padding[s] - m->davit_patch[s]) / m->davit_stride[s] + 1;
    }
    m->grid       = hw;
    m->n_patches  = hw * hw;
    m->img_tokens = 1 + m->n_patches;

    if (m->num_actions <= 0 || m->dim_action <= 0 || m->denoise_steps <= 0 ||
        m->hidden <= 0 || m->depth <= 0 || m->heads <= 0 || m->num_domains <= 0 ||
        m->soft_prompts <= 0 || m->dim_time <= 0 || m->max_len_seq <= 0 ||
        m->text_len <= 0 || m->d_model <= 0 || m->enc_layers <= 0 || m->enc_heads <= 0 ||
        m->image_size <= 0 || m->window_size <= 0 || m->proj_dim <= 0 ||
        m->hidden % m->heads != 0 || m->d_model % m->enc_heads != 0 ||
        m->grip0 >= m->dim_action || m->grip1 >= m->dim_action) {
        std::fprintf(stderr, "vla(xvla): invalid or inconsistent GGUF dimensions/metadata\n");
        return nullptr;
    }

    Config & c = m->cfg;
    c.n_img           = m->img_tokens * 3;
    c.n_lang          = m->text_len;
    c.n_state         = 1;
    c.n_prefix        = m->img_tokens + m->text_len;
    c.n_suffix        = m->num_actions;
    c.n_full          = c.n_prefix + c.n_suffix;
    c.hidden          = m->hidden;
    c.expert_h        = m->hidden;
    c.intermediate    = 4 * m->hidden;
    c.expert_inter    = 4 * m->hidden;
    c.n_q_heads       = m->heads;
    c.n_kv_heads      = m->heads;
    c.head_dim        = m->hidden / m->heads;
    c.q_full_dim      = m->hidden;
    c.kv_full_dim     = m->hidden;
    c.n_layers        = m->depth;
    c.self_attn_every_n = 1;
    c.max_state_dim   = m->dim_proprio;
    c.max_action_dim  = m->dim_action;
    c.real_state_dim  = m->dim_proprio;
    c.real_action_dim = m->dim_action;
    c.norm_eps        = kLnEps;
    c.min_period      = 0.0;
    c.max_period      = 100.0;
    c.num_steps       = (int) m->num_actions;
    c.rms_eps         = 0.0f;
    c.rope_n_dims     = 0;
    c.rope_mode       = 0;
    c.rope_freq_base  = 0.0f;

    std::printf("vla(xvla): hidden=%lld depth=%lld heads=%lld domains=%lld prompts=%lld "
                "actions=%lldx%lld steps=%lld enc=%lldx%lld davit=%lld/%lld/%lld/%lld "
                "grid=%lld img_tokens=%lld matmul=%s\n",
                (long long) m->hidden, (long long) m->depth, (long long) m->heads,
                (long long) m->num_domains, (long long) m->soft_prompts,
                (long long) m->num_actions, (long long) m->dim_action,
                (long long) m->denoise_steps,
                (long long) m->enc_layers, (long long) m->d_model,
                (long long) m->davit_depths[0], (long long) m->davit_depths[1],
                (long long) m->davit_depths[2], (long long) m->davit_depths[3],
                (long long) m->grid, (long long) m->img_tokens,
                m->matmul_type == GGML_TYPE_F32 ? "F32" :
                m->matmul_type == GGML_TYPE_Q8_0 ? "Q8_0" :
                m->matmul_type == GGML_TYPE_Q4_K ? "Q4_K" : "BF16");

    // backend
    {
        const unsigned hc = std::thread::hardware_concurrency();
        m->n_threads = (hc == 0) ? 4 : (int) std::min(hc, 8u);
    }
    m->is_cuda = false;
#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) { m->is_cuda = true; std::printf("vla(xvla): backend = CUDA (device 0)\n"); }
    else            { std::fprintf(stderr, "vla(xvla): ggml_backend_cuda_init failed; falling back to CPU\n"); }
#endif
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) { std::fprintf(stderr, "vla(xvla): ggml_backend_cpu_init failed\n"); return nullptr; }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(xvla): backend = CPU (%d threads)\n", m->n_threads);
    }

    // weight context
    {
        ggml_init_params wp = { (size_t) 64 * 1024 * 1024, nullptr, true };
        m->ctx_weights = ggml_init(wp);
        if (!m->ctx_weights) { std::fprintf(stderr, "vla(xvla): ggml_init(ctx_weights) failed\n"); return nullptr; }
    }
    ggml_context * W = m->ctx_weights;
    std::vector<ggml_tensor *> weights;

    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) { std::fprintf(stderr, "vla(xvla): missing tensor %s\n", name); return nullptr; }
        ggml_tensor * t = ggml_new_tensor(W, type, GGML_MAX_DIMS, gt->ne);
        ggml_set_name(t, name);
        weights.push_back(t);
        return t;
    };
    auto mk_mm  = [&](const char * name) -> ggml_tensor * {
        ggml_type ty = m->matmul_type;
        if (ty == GGML_TYPE_Q4_K) {
            // Keep the fallback decision identical to read_convert's so the
            // resident tensor type matches the produced byte stream.
            const ggml_tensor * gt = g.meta(name);
            if (gt && gt->ne[0] % ggml_blck_size(GGML_TYPE_Q4_K) != 0) ty = GGML_TYPE_Q8_0;
        }
        return mk(name, ty);
    };
    auto mk_f32 = [&](const char * name) -> ggml_tensor * { return mk(name, GGML_TYPE_F32); };
    auto tensor_exists = [&](const char * name) -> bool {
        return gguf_find_tensor(g.gctx, name) >= 0;
    };
    auto mk_wb = [&](const char * stem) -> WB {
        char b[128];
        std::snprintf(b, sizeof(b), "%s.w", stem);
        ggml_tensor * w = mk_mm(b);
        std::snprintf(b, sizeof(b), "%s.b", stem);
        ggml_tensor * bb = tensor_exists(b) ? mk_f32(b) : nullptr;
        return {w, bb};
    };
    auto mk_wb_f32 = [&](const char * stem) -> WB {
        char b[128];
        std::snprintf(b, sizeof(b), "%s.w", stem);
        ggml_tensor * w = mk_f32(b);
        std::snprintf(b, sizeof(b), "%s.b", stem);
        ggml_tensor * bb = tensor_exists(b) ? mk_f32(b) : nullptr;
        return {w, bb};
    };
    // Conv kernels stay F32: ggml_conv_2d im2col inherits the kernel type and
    // ggml_conv_2d_dw's im2col is F16 (BF16 kernel x F16 src1 is unsupported).
    auto mk_conv_kernel = [&](const char * name) -> ggml_tensor * {
        return mk(name, GGML_TYPE_F32);
    };

    // ---- vision tower ----
    m->davit.resize(4);
    for (int s = 0; s < 4; ++s) {
        char stem[128], kb[160];
        std::snprintf(kb, sizeof(kb), "vit.s%d.conv.w", s);
        m->conv_k[s].w = mk_conv_kernel(kb);
        std::snprintf(kb, sizeof(kb), "vit.s%d.conv.b", s);
        m->conv_k[s].b = tensor_exists(kb) ? mk_f32(kb) : nullptr;
        std::snprintf(stem, sizeof(stem), "vit.s%d.norm", s);
        m->conv_nb[s] = mk_wb_f32(stem);
        m->davit[(size_t) s].resize((size_t) m->davit_depths[s] * 2);
        for (int64_t j = 0; j < m->davit_depths[s]; ++j) {
            for (int kind = 0; kind < 2; ++kind) {
                const char * kk = (kind == 0) ? "sp" : "ch";
                DavitBlockW & w = m->davit[(size_t) s][(size_t) (j * 2 + kind)];
                auto pick_conv = [&](const char * suffix, WB & dst) {
                    char cs[128];
                    std::snprintf(cs, sizeof(cs), "vit.s%d.p%lld.%s.%s.w",
                                  s, (long long) j, kk, suffix);
                    dst.w = mk_conv_kernel(cs);
                    std::snprintf(cs, sizeof(cs), "vit.s%d.p%lld.%s.%s.b",
                                  s, (long long) j, kk, suffix);
                    dst.b = tensor_exists(cs) ? mk_f32(cs) : nullptr;
                };
                auto pick_lin = [&](const char * suffix, WB & dst) {
                    std::snprintf(stem, sizeof(stem), "vit.s%d.p%lld.%s.%s",
                                  s, (long long) j, kk, suffix);
                    dst = mk_wb(stem);
                };
                auto pick_norm = [&](const char * suffix, WB & dst) {
                    std::snprintf(stem, sizeof(stem), "vit.s%d.p%lld.%s.%s",
                                  s, (long long) j, kk, suffix);
                    dst = mk_wb_f32(stem);
                };
                pick_conv("conv1", w.conv1);
                pick_norm("norm", w.norm);
                pick_lin("qkv", w.qkv);
                pick_lin("proj", w.proj);
                pick_conv("conv2", w.conv2);
                pick_norm("ffn_norm", w.ffn_norm);
                pick_lin("fc1", w.fc1);
                pick_lin("fc2", w.fc2);
            }
        }
    }

    // ---- image projection ----
    m->vproj_pos_row  = mk_f32("vproj.pos_row");
    m->vproj_pos_col  = mk_f32("vproj.pos_col");
    m->vproj_temporal = mk_f32("vproj.temporal");
    m->vproj_proj.w   = mk_mm("vproj.proj.w");
    m->vproj_proj.b   = nullptr;
    m->vproj_proj_norm = mk_wb_f32("vproj.proj_norm");

    // ---- text encoder ----
    m->tok_emb  = mk_mm("text.tok_emb");
    m->pos_emb  = mk_f32("text.pos_emb");
    m->emb_norm = mk_wb_f32("text.emb_norm");
    m->enc.resize((size_t) m->enc_layers);
    for (int64_t i = 0; i < m->enc_layers; ++i) {
        char stem[96];
        EncLayerW & w = m->enc[(size_t) i];
        auto pick = [&](const char * suffix, WB & dst) {
            std::snprintf(stem, sizeof(stem), "text.blk.%lld.%s", (long long) i, suffix);
            dst = mk_wb(stem);
        };
        auto pick_norm = [&](const char * suffix, WB & dst) {
            std::snprintf(stem, sizeof(stem), "text.blk.%lld.%s", (long long) i, suffix);
            dst = mk_wb_f32(stem);
        };
        pick("attn_q", w.attn_q);
        pick("attn_k", w.attn_k);
        pick("attn_v", w.attn_v);
        pick("attn_o", w.attn_o);
        pick_norm("attn_norm", w.attn_norm);
        pick("fc1", w.ffn_up);
        pick("fc2", w.ffn_down);
        pick_norm("ffn_norm", w.ffn_norm);
    }

    // ---- action transformer ----
    m->act_vlm_proj = mk_wb("act.vlm_proj");
    m->act_aux_proj = mk_wb("act.aux_proj");
    m->act_pos      = mk_f32("act.pos_emb");
    m->act_out_norm = mk_wb_f32("act.out_norm");
    m->aenc_fc      = mk_f32("act.aenc.fc");
    m->aenc_b       = mk_f32("act.aenc.bias");
    m->adec_fc      = mk_f32("act.adec.fc");
    m->adec_b       = mk_f32("act.adec.bias");
    m->prompts      = mk_f32("act.prompts");
    m->act_blk.resize((size_t) m->depth);
    for (int64_t i = 0; i < m->depth; ++i) {
        char stem[96];
        ActBlockW & w = m->act_blk[(size_t) i];
        auto pick = [&](const char * suffix, WB & dst) {
            std::snprintf(stem, sizeof(stem), "act.blk.%lld.%s", (long long) i, suffix);
            dst = mk_wb(stem);
        };
        auto pick_norm = [&](const char * suffix, WB & dst) {
            std::snprintf(stem, sizeof(stem), "act.blk.%lld.%s", (long long) i, suffix);
            dst = mk_wb_f32(stem);
        };
        pick_norm("norm1", w.norm1);
        pick("qkv", w.qkv);
        pick("proj", w.proj);
        pick_norm("norm2", w.norm2);
        pick("fc1", w.fc1);
        pick("fc2", w.fc2);
    }

    for (ggml_tensor * t : weights) if (!t) {
        std::fprintf(stderr, "vla(xvla): weight tensor creation failed\n");
        return nullptr;
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) { std::fprintf(stderr, "vla(xvla): ggml_backend_alloc_ctx_tensors failed (out of memory?)\n"); return nullptr; }
    for (ggml_tensor * t : weights) {
        std::vector<uint8_t> bytes = g.read_convert(t->name, t->type);
        if (bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(xvla): upload size mismatch for %s (%zu vs %zu)\n",
                         t->name, bytes.size(), ggml_nbytes(t));
            return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::printf("vla(xvla): resident weights = %.2f GiB\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0));

    return m;
}

}
