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
 * @file turbovla.cpp
 * @brief TurboVLA runtime (DINOv3 ViT + BERT + bidirectional cross-attn fusion + ACT decoder).
 *
 * Layout follows scripts/convert_turbovla_to_gguf.py. The whole model lives in
 * one GGUF: BERT text encoder, DINOv3 vision tower, vision projection, view
 * embeddings, 6 bidirectional fusion blocks, 6 text-enhancer blocks (post-LN),
 * state projection, and the 3-layer ACT-style transformer-decoder action head.
 *
 * Reference numerical contract (turbovla-official):
 *  * Text: instructions are tokenized with the bundled WordPiece vocab,
 *    padded to turbovla.text_padding_length (LIBERO object: L=21, including
 *    any trailing pads). BERT runs with the GroundingDINO sub-sentence mask (block
 *    diagonal between special tokens; for LIBERO prompts a pure diagonal),
 *    position_ids = 0, token_type_ids = 0, then Linear(768 -> hidden).
 *  * Vision: per-view DINOv3 (2D RoPE on patch tokens only, cls + 4 register
 *    prefix tokens un-rotated), LayerNorm, patch tokens only, VisionProjection
 *    (input LN + GELU MLP + skip + output LN) plus a per-view embedding,
 *    flattened (view-major) to 2*256 = 512 tokens.
 *  * Fusion: pre-LN BiAttentionBlock (v->l and l->v share one score matrix,
 *    scale folded into the v-query, text pads masked for the v-attention) with
 *    per-dim gamma residuals, followed by a post-LN self-attn text enhancer
 *    (sub-sentence mask; padded rows keep their diagonal self-attention).
 *  * Action: state -> LayerNorm -> GELU MLP -> 2 tokens (+ learned position),
 *    condition = [visual 512, text L, state 2]; ACT decoder (pre-LN
 *    TransformerDecoder: self-attn, cross-attn, ReLU FFN) over learned
 *    queries; ReLU MLP head; tanh. Denormalization follows the released
 *    LIBERO policy: arm = 0.5*(a+1)*(max-min)+min, gripper = sign(a).
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
#ifndef _WIN32
#include <sys/types.h>
#endif
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
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
        if (!gctx) { std::fprintf(stderr, "vla(turbovla): gguf_init_from_file failed for %s\n", path.c_str()); return false; }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp)       { std::fprintf(stderr, "vla(turbovla): fopen failed for %s\n", path.c_str()); return false; }
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
    std::string str(const char * k) const { return gguf_get_val_str(gctx, gguf_find_key(gctx, k)); }

    const ggml_tensor * meta(const char * name) const { return ggml_get_tensor(meta_ctx, name); }

    bool read_raw(const char * name, void * buf) const {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) { std::fprintf(stderr, "vla(turbovla): missing tensor %s\n", name); return false; }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t nb  = gguf_get_tensor_size(gctx, id);
        if (!seek_absolute(fp, off)) return false;
        return std::fread(buf, 1, nb, fp) == nb;
    }

    std::vector<uint8_t> read_convert(const char * name, ggml_type target) {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(turbovla): missing tensor %s\n", name); return {}; }
        const int64_t n = ggml_nelements(t);
        std::vector<float> f32(n);
        if (t->type == GGML_TYPE_F32) {
            if (!read_raw(name, f32.data())) return {};
        } else if (t->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp(n);
            if (!read_raw(name, tmp.data())) return {};
            ggml_bf16_to_fp32_row(tmp.data(), f32.data(), n);
        } else {
            std::fprintf(stderr, "vla(turbovla): tensor %s has unsupported type %d\n", name, (int) t->type);
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
        std::fprintf(stderr, "vla(turbovla): unsupported resident type %d for %s\n", (int) target, name);
        return {};
    }

    bool fetch_rows_f32(const char * name, const std::vector<int32_t> & row_ids,
                        float * dst, int64_t cols) const {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(turbovla): missing tensor %s\n", name); return false; }
        if (t->ne[0] != cols || t->ne[2] != 1 || t->ne[3] != 1) {
            std::fprintf(stderr, "vla(turbovla): %s shape unfit for row-fetch\n", name); return false;
        }
        const int64_t rows = t->ne[1];
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_BF16) {
            std::fprintf(stderr, "vla(turbovla): tensor %s has unsupported row type %d\n",
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
            if (r < 0 || r >= rows) { std::fprintf(stderr, "vla(turbovla): row %d out of range for %s\n", r, name); return false; }
            if (!seek_absolute(fp, base + (size_t) r * rb)) return false;
            if (std::fread(row.data(), 1, rb, fp) != rb) return false;
            if (elsz == 4) std::memcpy(dst + k * cols, row.data(), rb);
            else ggml_bf16_to_fp32_row(reinterpret_cast<ggml_bf16_t *>(row.data()), dst + k * cols, cols);
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// WordPiece tokenizer (bert-base-uncased rules; vocab bundled in the GGUF).
// ---------------------------------------------------------------------------
struct WordPieceTokenizer {
    std::unordered_map<std::string, int32_t> vocab;
    int32_t id_cls = 101;
    int32_t id_sep = 102;
    int32_t id_unk = 100;
    int32_t id_pad = 0;

    bool load(const gguf_reader & g, const char * key, int64_t expected_size) {
        const int kid = gguf_find_key(g.gctx, key);
        if (kid < 0) return false;
        const size_t n = gguf_get_arr_n(g.gctx, kid);
        if ((int64_t) n != expected_size) {
            std::fprintf(stderr, "vla(turbovla): vocab size %zu != expected %lld\n",
                         n, (long long) expected_size);
            return false;
        }
        vocab.reserve(n * 2);
        for (size_t i = 0; i < n; ++i) {
            const char * s = gguf_get_arr_str(g.gctx, kid, (int) i);
            if (!s) return false;
            vocab.emplace(std::string(s), (int32_t) i);
        }
        return true;
    }

    bool empty() const { return vocab.empty(); }

    // BERT basic tokenization (lowercase + punctuation split) then greedy
    // longest-match WordPiece with "##" continuation markers.
    std::vector<int32_t> encode(const std::string & text) const {
        std::vector<int32_t> ids;
        std::string lowered;
        lowered.reserve(text.size());
        for (unsigned char c : text) {
            if (c == '\t' || c == '\n' || c == '\r') c = ' ';
            lowered.push_back((char) std::tolower(c));
        }
        auto emit_word = [&](const std::string & word) {
            if (word.empty()) return;
            if (word.size() > 100) { ids.push_back(id_unk); return; }
            size_t start = 0;
            bool any = false;
            while (start < word.size()) {
                size_t end = word.size();
                int32_t found = -1;
                while (start < end) {
                    std::string sub = word.substr(start, end - start);
                    if (start > 0) sub = "##" + sub;
                    auto it = vocab.find(sub);
                    if (it != vocab.end()) { found = it->second; break; }
                    --end;
                }
                if (found < 0) { ids.push_back(id_unk); any = true; break; }
                ids.push_back(found);
                any = true;
                start = end;
            }
            (void) any;
        };
        std::string cur;
        for (char c : lowered) {
            const bool alnum = std::isalnum((unsigned char) c) != 0;
            if (c == ' ') { emit_word(cur); cur.clear(); }
            else if (!alnum) {
                emit_word(cur); cur.clear();
                std::string p(1, c);
                auto it = vocab.find(p);
                ids.push_back(it != vocab.end() ? it->second : id_unk);
            } else cur.push_back(c);
        }
        emit_word(cur);
        return ids;
    }
};

// ---------------------------------------------------------------------------
// Weight structs (GGUF tensor names in parentheses).
// ---------------------------------------------------------------------------
struct WB { ggml_tensor * w = nullptr; ggml_tensor * b = nullptr; };

struct BertLayerW {          // text.blk.N.*
    WB attn_q, attn_k, attn_v, attn_o;
    WB attn_norm, ffn_up, ffn_down, ffn_norm;
};

struct VitLayerW {           // vit.blk.N.*
    WB attn_norm, attn_q, attn_k, attn_v, attn_o;   // k has no bias
    ggml_tensor * ls1 = nullptr;
    WB ffn_norm, ffn_up, ffn_down;
    ggml_tensor * ls2 = nullptr;
};

struct FuseLayerW {          // fuse.N.*
    WB ln_v, ln_l;
    WB q_from_v, k_from_l, val_from_v, val_from_l, out_v, out_l;
    ggml_tensor * gamma_v = nullptr, * gamma_l = nullptr;
};

struct TenhLayerW {          // tenh.N.*
    WB attn_qkv, attn_o, ffn_up, ffn_down, attn_norm, ffn_norm;
};

struct ActLayerW {           // act.blk.N.*
    WB self_qkv, self_o, cross_qkv, cross_o;
    WB ffn_up, ffn_down, norm1, norm2, norm3;
};

// ---------------------------------------------------------------------------
// ggml graph helpers. All activations are F32; matmul weights are BF16.
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

// Flatten a [n] / [1,n] / [n,1] tensor to a plain column of n elements so it
// broadcasts over trailing sequence/batch dims in element-wise ops.
ggml_tensor * reshape_col(ggml_context * ctx, ggml_tensor * t, int64_t rows) {
    return ggml_reshape_1d(ctx, t, rows);
}

// Feature rows [off, off+n) of a 2D [in, seq] tensor along the innermost
// (feature) axis, as a contiguous copy. This is how fused qkv/proj heads are
// split: each source stores all features per token contiguously (col-major),
// so the per-token stride is t->nb[1] while the feature offset is off*nb[0].
ggml_tensor * row_slice(ggml_context * ctx, ggml_tensor * t,
                        int64_t /*total_feats*/, int64_t off, int64_t n) {
    ggml_tensor * v = ggml_view_2d(ctx, t, n, t->ne[1], t->nb[1], (size_t) off * t->nb[0]);
    return ggml_cont(ctx, v);
}

// Batched softmax multi-head attention.
//   q: [head_dim, H, Nq, B]   k: [head_dim, H, Nk, B]   v: [head_dim, H, Nk, B]
//   mask: additive [Nk, Nq] (already -inf where masked) or NULL.
// Returns [head_dim, H, Nq, B].
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

// Rotate-half style 2D RoPE application to the patch-token tail of a
// [head_dim, H, N_patch, B] tensor. cos/sin are [head_dim, N_patch] inputs,
// broadcast over heads and batch.
ggml_tensor * rope_2d_patches(ggml_context * ctx, ggml_tensor * x,
                              ggml_tensor * cos, ggml_tensor * sin) {
    const int64_t hd = x->ne[0], H = x->ne[1], N = x->ne[2], B = x->ne[3];
    // cos/sin are [hd, N]; reshape to [hd, 1, N, 1] and let ggml broadcast
    // over heads and batch in the element-wise mul (avoids materialising
    // full-size cos_b/sin_b copies per head/batch).
    ggml_tensor * cos4 = ggml_reshape_4d(ctx, cos, hd, 1, N, 1);
    ggml_tensor * sin4 = ggml_reshape_4d(ctx, sin, hd, 1, N, 1);
    ggml_tensor * x1 = ggml_view_4d(ctx, x, hd / 2, H, N, B, x->nb[1], x->nb[2], x->nb[3], 0);
    ggml_tensor * x2 = ggml_view_4d(ctx, x, hd / 2, H, N, B, x->nb[1], x->nb[2], x->nb[3], (hd / 2) * x->nb[0]);
    ggml_tensor * rot = ggml_concat(ctx,
        ggml_cont(ctx, ggml_neg(ctx, ggml_cont(ctx, x2))),
        ggml_cont(ctx, x1), 0);
    return ggml_add(ctx, ggml_mul(ctx, x, cos4), ggml_mul(ctx, rot, sin4));
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
struct TurboVLAModelArch : public ModelArchBase {
    TurboVLAModelArch() : ModelArchBase(Arch::TURBOVLA) {}
    ~TurboVLAModelArch() override;

    std::vector<float> predict(const Inputs& in) override;

    ggml_backend_t        backend     = nullptr;
    bool                  is_cuda     = false;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_context *        ctx_weights = nullptr;
    ggml_type             matmul_type = GGML_TYPE_BF16;

    // geometry
    int64_t hidden        = 256;   // interaction hidden_dim
    int64_t nheads        = 8;     // action-head heads (interaction config)
    int64_t fuse_heads    = 4;     // max(1, nheads / 2)
    int64_t fuse_embed    = 1024;  // enhancer_inner_dim
    int64_t n_fuse        = 6;
    int64_t act_ffn       = 2048;
    int64_t act_layers    = 3;
    int64_t horizon       = 12;
    int64_t action_dim    = 7;
    int64_t state_dim     = 8;
    int64_t n_state_tok   = 2;
    int64_t mlp_hidden    = 512;
    int64_t state_hidden  = 256;
    int64_t num_views     = 2;
    int64_t image_size    = 256;
    int64_t patch_size    = 16;
    int64_t grid          = 16;    // image_size / patch_size
    int64_t n_patches     = 256;   // grid*grid
    int64_t vit_hidden    = 768;
    int64_t vit_heads     = 12;
    int64_t vit_layers    = 12;
    int64_t n_prefix_tok  = 5;     // cls + register tokens
    double  rope_theta    = 100.0;
    int64_t bert_hidden   = 768;
    int64_t bert_heads    = 12;
    int64_t bert_layers   = 12;
    int64_t text_len      = 21;    // text_padding_length
    int64_t visual_tokens = 512;   // num_views * n_patches

    // text encoder
    WB text_emb_norm;                    // embeddings.LayerNorm
    WB text_proj;
    std::vector<BertLayerW> bert;

    // vision tower
    WB vit_patch;                        // conv weights reshaped [vit_hidden, 3*p*p]
    ggml_tensor * vit_cls = nullptr;     // [vit_hidden]
    ggml_tensor * vit_reg = nullptr;     // [vit_hidden, n_reg]
    WB vit_out_norm;
    std::vector<VitLayerW> vit;

    // vision projection + view embedding
    WB vp_input_norm, vp_fc1, vp_fc2, vp_skip, vp_out_norm;
    ggml_tensor * view_emb = nullptr;    // [hidden, num_views]

    // fusion + enhancer
    std::vector<FuseLayerW> fuse;
    std::vector<TenhLayerW> tenh;

    // action head
    WB st_ln, st_fc1, st_fc2, st_out_norm;
    ggml_tensor * st_pos = nullptr;      // [hidden, n_state_tok]
    ggml_tensor * act_queries = nullptr; // [hidden, horizon]
    std::vector<ActLayerW> act;
    WB head_fc1, head_fc2, head_fc3;

    // normalization statistics (host)
    std::vector<float> proprio_mean, proprio_std, action_min, action_max;

    // 2D RoPE tables (fixed once the geometry is known; computed lazily on the
    // first predict to avoid recomputing cos/sin on every call).
    std::vector<float> rope_cos, rope_sin;
    bool               rope_ready = false;

    WordPieceTokenizer tokenizer;
    std::unique_ptr<gguf_reader> reader;   // kept open: embedding lookups in predict
    int n_threads = 4;
};

TurboVLAModelArch::~TurboVLAModelArch() {
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend)     ggml_backend_free(backend);
}

std::vector<float> TurboVLAModelArch::predict(const Inputs& in) {
    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();
    stats = Stats{};

    const int64_t L = text_len;
    const float ln_eps_layer = 1e-5f;   // torch nn.LayerNorm default

    // ---------------- 0) language tokens ----------------
    std::vector<int32_t> ids;           // raw wordpiece ids (no specials)
    if (in.lang_tokens && in.n_lang > 0) {
        // Client-side tokenization: strip [CLS]/[SEP] framing if present so we
        // can re-frame deterministically to our fixed padding layout. Padding
        // (if any) is assumed to be trailing, as standard: trim trailing pads
        // first, then a trailing [SEP], then a leading [CLS]. Interleaved pads
        // are not supported by this runtime and are treated as ordinary tokens.
        ids.assign(in.lang_tokens, in.lang_tokens + in.n_lang);
        if (!ids.empty() && ids.front() == tokenizer.id_cls) ids.erase(ids.begin());
        while (!ids.empty() && ids.back() == tokenizer.id_pad) ids.pop_back();
        if (!ids.empty() && ids.back() == tokenizer.id_sep)  ids.pop_back();
    } else if (in.language_text && in.language_text[0] != '\0') {
        ids = tokenizer.encode(in.language_text);
    } else {
        std::fprintf(stderr, "vla(turbovla): predict needs language_text or lang_tokens\n");
        return {};
    }
    if ((int64_t) ids.size() > L - 2) ids.resize((size_t) (L - 2));
    std::vector<int32_t> framed;                 // [CLS] ... [SEP] [PAD]...
    framed.reserve((size_t) L);
    framed.push_back(tokenizer.id_cls);
    framed.insert(framed.end(), ids.begin(), ids.end());
    framed.push_back(tokenizer.id_sep);
    while ((int64_t) framed.size() < L) framed.push_back(tokenizer.id_pad);
    std::vector<int32_t> token_valid(L);         // 1 = real token, 0 = pad
    for (int64_t i = 0; i < L; ++i) token_valid[i] = (i < (int64_t) (ids.size() + 2)) ? 1 : 0;

    // ---------------- 1) host-side preprocessing ----------------
    // 1a) vision: 2 views -> normalized patch columns [3*p*p, n_patches, V]
    if (in.n_images != (int) num_views || !in.images) {
        std::fprintf(stderr, "vla(turbovla): predict needs exactly %lld images (got %d)\n",
                     (long long) num_views, in.n_images);
        return {};
    }
    const int64_t p = patch_size;
    const int64_t chwp = 3 * p * p;
    std::vector<float> patch_cols((size_t) chwp * n_patches * num_views);
    {
        const float mean3[3] = {0.485f, 0.456f, 0.406f};
        const float std3[3]  = {0.229f, 0.224f, 0.225f};
        for (int64_t v = 0; v < num_views; ++v) {
            const ImageView & view = in.images[v];
            if (view.w != (int) image_size || view.h != (int) image_size) {
                std::fprintf(stderr, "vla(turbovla): image[%lld] is %dx%d, expected %lldx%lld\n",
                             (long long) v, view.w, view.h,
                             (long long) image_size, (long long) image_size);
                return {};
            }
            float * dst = patch_cols.data() + (size_t) v * chwp * n_patches;
            for (int64_t ph = 0; ph < grid; ++ph) {
                for (int64_t pw = 0; pw < grid; ++pw) {
                    float * col = dst + (size_t) (ph * grid + pw) * chwp;
                    for (int64_t c = 0; c < 3; ++c) {
                        float * seg = col + c * p * p;
                        for (int64_t kh = 0; kh < p; ++kh) {
                            const int64_t y = ph * p + kh;
                            for (int64_t kw = 0; kw < p; ++kw) {
                                const int64_t x = pw * p + kw;
                                float px;
                                if (view.format == PixelFormat::U8) {
                                    const uint8_t * src = static_cast<const uint8_t *>(view.data);
                                    px = (float) src[(y * image_size + x) * 3 + c] / 255.0f;
                                } else {
                                    const float * src = static_cast<const float *>(view.data);
                                    px = src[(y * image_size + x) * 3 + c];
                                }
                                *seg++ = (px - mean3[c]) / std3[c];
                            }
                        }
                    }
                }
            }
        }
    }

    // 1b) BERT input rows: word emb + pos emb (sub-sentence ids) + tok_type[0].
    // Special-token mask first: [CLS]/[SEP]/'.'/'?' delimit sub-sentence blocks.
    std::vector<char> is_special((size_t) L, 0);
    {
        static const int32_t specials[] = {101, 102, 1012, 1029};
        for (int64_t i = 0; i < L; ++i)
            for (int32_t s : specials)
                if (framed[(size_t) i] == s) is_special[(size_t) i] = 1;
    }
    std::vector<float> bert_emb((size_t) bert_hidden * L);
    {
        // generate_masks_with_special_tokens position ids: each block counts
        // arange(0, block_len) starting at the token after the previous
        // special token; tokens outside any mid-sequence block stay at 0.
        std::vector<int32_t> pos_ids((size_t) L, 0);
        {
            int64_t prev = -1;
            for (int64_t i = 0; i < L; ++i) {
                if (!is_special[(size_t) i]) continue;
                if (i != 0 && i != L - 1)
                    for (int64_t q = prev + 1; q <= i; ++q)
                        pos_ids[(size_t) q] = (int32_t) (q - prev - 1);
                prev = i;
            }
        }
        if (!reader->fetch_rows_f32("text.token_emb", framed, bert_emb.data(), bert_hidden)) {
            std::fprintf(stderr, "vla(turbovla): token embedding lookup failed\n");
            return {};
        }
        std::vector<float> pos_rows((size_t) bert_hidden * L);
        std::vector<float> type0((size_t) bert_hidden);
        const bool have_pos = reader->fetch_rows_f32("text.pos_emb", pos_ids, pos_rows.data(), bert_hidden);
        const bool have_typ = reader->fetch_rows_f32("text.tok_type_emb", {0}, type0.data(), bert_hidden);
        if (have_pos || have_typ) {
            for (int64_t i = 0; i < L; ++i) {
                float * row = bert_emb.data() + (size_t) i * bert_hidden;
                const float * p = have_pos ? pos_rows.data() + (size_t) i * bert_hidden : nullptr;
                for (int64_t j = 0; j < bert_hidden; ++j)
                    row[j] += (p ? p[(size_t) j] : 0.f) + (have_typ ? type0[(size_t) j] : 0.f);
            }
        }
    }

    // 1c) DINOv3 2D RoPE tables (fp32, exact). Fixed geometry -> cache once.
    if (!rope_ready) {
        rope_cos.resize((size_t) (vit_hidden / vit_heads) * n_patches);
        rope_sin.resize((size_t) (vit_hidden / vit_heads) * n_patches);
        const int64_t hd = vit_hidden / vit_heads;      // 64
        const int64_t quarter = hd / 4;                 // 16
        std::vector<double> inv_freq((size_t) quarter);
        for (int64_t i = 0; i < quarter; ++i)
            inv_freq[(size_t) i] = 1.0 / std::pow(rope_theta, (double) (4 * i) / (double) hd);
        std::vector<float> angles((size_t) hd / 2);     // per-patch scratch
        for (int64_t ph = 0; ph < grid; ++ph) {
            for (int64_t pw = 0; pw < grid; ++pw) {
                const double cy = 2.0 * ((double) ph + 0.5) / (double) grid - 1.0;
                const double cx = 2.0 * ((double) pw + 0.5) / (double) grid - 1.0;
                // HF DINOv3 layout: angles = [y_freqs | x_freqs], then tile(2)
                // so the rotate-half pair (d, d + hd/2) shares one frequency.
                for (int64_t f = 0; f < quarter; ++f) {
                    angles[(size_t) f]             = (float) (2.0 * M_PI * cy * inv_freq[(size_t) f]);
                    angles[(size_t) (quarter + f)] = (float) (2.0 * M_PI * cx * inv_freq[(size_t) f]);
                }
                float * c = rope_cos.data() + (size_t) (ph * grid + pw) * hd;
                float * s = rope_sin.data() + (size_t) (ph * grid + pw) * hd;
                for (int64_t d = 0; d < hd / 2; ++d) {
                    c[d] = std::cos(angles[(size_t) d]);
                    s[d] = std::sin(angles[(size_t) d]);
                    c[hd / 2 + d] = c[d];
                    s[hd / 2 + d] = s[d];
                }
            }
        }
        rope_ready = true;
    }

    // 1d) attention masks (additive, 0 / -inf). Sub-sentence mask from
    // generate_masks_with_special_tokens (is_special computed in 1b): blocks
    // between special tokens attend internally; the tail stays diagonal.
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<float> bert_mask((size_t) L * L, neg_inf);      // [k, q] row-major
    std::vector<float> enh_mask((size_t) L * L, neg_inf);
    {
        int64_t prev = -1;
        for (int64_t i = 0; i < L; ++i) {
            if (!is_special[(size_t) i]) continue;
            if (i == 0 || i == L - 1) {
                bert_mask[(size_t) (i * L + i)] = 0.f;   // self only
            } else {
                for (int64_t q = prev + 1; q <= i; ++q)
                    for (int64_t k = prev + 1; k <= i; ++k)
                        bert_mask[(size_t) (q * L + k)] = 0.f;
            }
            prev = (int64_t) i;
        }
        // enhancer: same sub-sentence mask, but padded KEYS are banned while
        // padded ROWS keep a diagonal escape so no all--inf row produces NaN.
        enh_mask = bert_mask;
        for (int64_t q = 0; q < L; ++q)
            for (int64_t k = 0; k < L; ++k)
                if (!token_valid[(size_t) k]) enh_mask[(size_t) (q * L + k)] = neg_inf;
        for (int64_t q = 0; q < L; ++q)
            if (!token_valid[(size_t) q]) enh_mask[(size_t) (q * L + q)] = 0.f;
        // BERT pad rows (after [SEP] with no trailing special token) end up
        // all --inf, whose softmax is NaN. Give every row a diagonal escape so
        // the mask never produces NaN (the padded rows are discarded downstream).
        for (int64_t q = 0; q < L; ++q) {
            bool any_open = false;
            for (int64_t k = 0; k < L; ++k)
                if (bert_mask[(size_t) (q * L + k)] == 0.f) { any_open = true; break; }
            if (!any_open) bert_mask[(size_t) (q * L + q)] = 0.f;
        }
    }
    std::vector<float> fuse_mask((size_t) L * visual_tokens, 0.f);  // [k=L, q=visual]
    for (int64_t q = 0; q < visual_tokens; ++q)
        for (int64_t k = 0; k < L; ++k)
            if (!token_valid[(size_t) k]) fuse_mask[(size_t) (q * L + k)] = neg_inf;

    // 1e) state normalization
    std::vector<float> state_norm((size_t) state_dim, 0.f);
    for (int64_t i = 0; i < state_dim; ++i) {
        const float raw = (in.state && i < cfg.real_state_dim) ? in.state[i] : 0.f;
        state_norm[(size_t) i] = (raw - proprio_mean[(size_t) i]) /
                                 (proprio_std[(size_t) i] + 1e-6f);
    }

    // ---------------- 2) build the graph ----------------
    ggml_init_params cp = { (size_t) 256 * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) { std::fprintf(stderr, "vla(turbovla): ggml_init(ctx_compute) failed\n"); return {}; }

    const int64_t vhd = vit_hidden / vit_heads;      // 64
    const int64_t bhd = bert_hidden / bert_heads;    // 64
    const int64_t ehd = hidden / fuse_heads;         // 64  (enhancer head dim)
    const int64_t fhd = fuse_embed / fuse_heads;     // 256 (fusion head dim)
    const int64_t ahd = hidden / nheads;             // 32  (act decoder head dim)

    ggml_tensor * t_patch = ggml_new_tensor_3d(C, GGML_TYPE_F32, chwp, n_patches, num_views);
    ggml_set_input(t_patch);
    ggml_tensor * t_bert_emb = ggml_new_tensor_2d(C, GGML_TYPE_F32, bert_hidden, L);
    ggml_set_input(t_bert_emb);
    ggml_tensor * t_state = ggml_new_tensor_1d(C, GGML_TYPE_F32, state_dim);
    ggml_set_input(t_state);
    ggml_tensor * t_cos = ggml_new_tensor_2d(C, GGML_TYPE_F32, vhd, n_patches);
    ggml_set_input(t_cos);
    ggml_tensor * t_sin = ggml_new_tensor_2d(C, GGML_TYPE_F32, vhd, n_patches);
    ggml_set_input(t_sin);
    ggml_tensor * t_bert_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, L, L);
    ggml_set_input(t_bert_mask);
    ggml_tensor * t_enh_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, L, L);
    ggml_set_input(t_enh_mask);
    ggml_tensor * t_fuse_mask = ggml_new_tensor_2d(C, GGML_TYPE_F32, L, visual_tokens);
    ggml_set_input(t_fuse_mask);
    // ---- 2a) vision tower (batch = views) ----
    ggml_tensor * vis = nullptr;
    ggml_tensor * vis_dino = nullptr;
    ggml_tensor * vis_projected = nullptr;
    ggml_tensor * txt_bert = nullptr;
    ggml_tensor * txt_projected = nullptr;
    {
        ggml_tensor * patches = linear(C, t_patch, vit_patch);          // [vit_hidden, N, V]
        ggml_tensor * cls2 = ggml_reshape_3d(C, vit_cls, vit_hidden, 1, 1);
        cls2 = ggml_repeat(C, cls2, ggml_new_tensor_3d(C, GGML_TYPE_F32, vit_hidden, 1, num_views));
        ggml_tensor * reg2 = ggml_reshape_3d(C, vit_reg, vit_hidden, n_prefix_tok - 1, 1);
        reg2 = ggml_repeat(C, reg2, ggml_new_tensor_3d(C, GGML_TYPE_F32, vit_hidden, n_prefix_tok - 1, num_views));
        ggml_tensor * prefix = ggml_concat(C, cls2, reg2, 1);           // [vit_hidden, n_prefix, V]
        ggml_tensor * x = ggml_concat(C, prefix, patches, 1);           // [vit_hidden, n_prefix + N, V]
        const int64_t seq = n_prefix_tok + n_patches;
        for (int64_t i = 0; i < vit_layers; ++i) {
            const VitLayerW & w = vit[(size_t) i];
            ggml_tensor * h = ln_f32(C, x, w.attn_norm, 1e-6f);
            ggml_tensor * q = linear(C, h, w.attn_q);
            ggml_tensor * k = linear(C, h, w.attn_k);
            ggml_tensor * v = linear(C, h, w.attn_v);
            ggml_tensor * q4 = ggml_reshape_4d(C, q, vhd, vit_heads, seq, num_views);
            ggml_tensor * k4 = ggml_reshape_4d(C, k, vhd, vit_heads, seq, num_views);
            ggml_tensor * v4 = ggml_reshape_4d(C, v, vhd, vit_heads, seq, num_views);
            // RoPE on the patch tail only; prefix tokens stay un-rotated.
            auto rope_tail = [&](ggml_tensor * t4) {
                ggml_tensor * pre = ggml_view_4d(C, t4, vhd, vit_heads, n_prefix_tok, num_views,
                                                 t4->nb[1], t4->nb[2], t4->nb[3], 0);
                ggml_tensor * pat = ggml_cont(C, ggml_view_4d(C, t4, vhd, vit_heads, n_patches, num_views,
                                                 t4->nb[1], t4->nb[2], t4->nb[3], n_prefix_tok * t4->nb[2]));
                ggml_tensor * rot = rope_2d_patches(C, pat, t_cos, t_sin);
                return ggml_concat(C, ggml_cont(C, pre), rot, 2);
            };
            q4 = rope_tail(q4);
            k4 = rope_tail(k4);
            ggml_tensor * o = mha(C, q4, k4, v4, nullptr, 1.0f / std::sqrt((float) vhd));
            o = ggml_reshape_3d(C, o, vit_hidden, seq, num_views);
            o = linear(C, o, w.attn_o);
            x = ggml_add(C, x, ggml_mul(C, o, reshape_col(C, w.ls1, vit_hidden)));
            h = ln_f32(C, x, w.ffn_norm, 1e-6f);
            ggml_tensor * f = linear(C, h, w.ffn_up);
            f = ggml_gelu_erf(C, f);
            f = linear(C, f, w.ffn_down);
            x = ggml_add(C, x, ggml_mul(C, f, reshape_col(C, w.ls2, vit_hidden)));
        }
        x = ln_f32(C, x, vit_out_norm, 1e-6f);
        // strip prefix tokens -> [vit_hidden, n_patches, V]
        vis = ggml_cont(C, ggml_view_3d(C, x, vit_hidden, n_patches, num_views,
                                        x->nb[1], x->nb[2], n_prefix_tok * x->nb[1]));
        vis_dino = vis;
    }
    const auto tv0 = clk::now();

    // ---- 2b) vision projection + view embedding ----
    {
        ggml_tensor * h = ln_f32(C, vis, vp_input_norm, ln_eps_layer);
        ggml_tensor * m = linear(C, h, vp_fc1);
        m = ggml_gelu_erf(C, m);
        m = linear(C, m, vp_fc2);
        ggml_tensor * skip = ggml_mul_mat(C, vp_skip.w, vis);
        ggml_tensor * proj = ggml_add(C, skip, m);
        proj = ln_f32(C, proj, vp_out_norm, ln_eps_layer);            // [hidden, N, V]
        ggml_tensor * ve = ggml_reshape_3d(C, view_emb, hidden, 1, num_views);
        proj = ggml_add(C, proj, ve);                                  // broadcast patch dim
        // proj is [hidden, N, V]. Its contiguous storage is already view-major:
        // all N patches from view 0, then all N patches from view 1, matching
        // PyTorch's [B,V,N,C].flatten(1, 2). Interleaving V before N here
        // silently changes the token order seen by every fusion layer.
        proj = ggml_cont(C, proj);
        vis_projected = proj;
        vis = ggml_reshape_2d(C, proj, hidden, visual_tokens);         // [hidden, V*N]
    }

    // ---- 2c) BERT text encoder ----
    ggml_tensor * txt = ln_f32(C, t_bert_emb, text_emb_norm, 1e-12f);  // [bert_hidden, L]
    for (int64_t i = 0; i < bert_layers; ++i) {
        const BertLayerW & w = bert[(size_t) i];
        ggml_tensor * q = linear(C, txt, w.attn_q);
        ggml_tensor * k = linear(C, txt, w.attn_k);
        ggml_tensor * v = linear(C, txt, w.attn_v);
        ggml_tensor * q4 = ggml_reshape_4d(C, q, bhd, bert_heads, L, 1);
        ggml_tensor * k4 = ggml_reshape_4d(C, k, bhd, bert_heads, L, 1);
        ggml_tensor * v4 = ggml_reshape_4d(C, v, bhd, bert_heads, L, 1);
        ggml_tensor * o = mha(C, q4, k4, v4, t_bert_mask, 1.0f / std::sqrt((float) bhd));
        o = ggml_reshape_2d(C, o, bert_hidden, L);
        o = linear(C, o, w.attn_o);
        txt = ln_f32(C, ggml_add(C, txt, o), w.attn_norm, 1e-12f);    // post-LN
        ggml_tensor * f = linear(C, txt, w.ffn_up);
        f = ggml_gelu_erf(C, f);
        f = linear(C, f, w.ffn_down);
        txt = ln_f32(C, ggml_add(C, txt, f), w.ffn_norm, 1e-12f);
    }
    txt_bert = txt;
    txt = linear(C, txt, text_proj);                                   // [hidden, L]
    txt_projected = txt;

    // ---- 2d) bidirectional fusion + text enhancer ----
    for (int64_t i = 0; i < n_fuse; ++i) {
        const FuseLayerW & fw = fuse[(size_t) i];
        ggml_tensor * vn = ln_f32(C, vis, fw.ln_v, ln_eps_layer);
        ggml_tensor * ln = ln_f32(C, txt, fw.ln_l, ln_eps_layer);
        ggml_tensor * qv = linear(C, vn, fw.q_from_v);                 // [fuse_embed, Nv]
        qv = ggml_scale(C, qv, 1.0f / std::sqrt((float) fhd));
        ggml_tensor * kl = linear(C, ln, fw.k_from_l);                 // [fuse_embed, L]
        ggml_tensor * vv = linear(C, vn, fw.val_from_v);
        ggml_tensor * vl = linear(C, ln, fw.val_from_l);
        ggml_tensor * qv4 = ggml_reshape_4d(C, qv, fhd, fuse_heads, visual_tokens, 1);
        ggml_tensor * kl4 = ggml_reshape_4d(C, kl, fhd, fuse_heads, L, 1);
        ggml_tensor * vv4 = ggml_reshape_4d(C, vv, fhd, fuse_heads, visual_tokens, 1);
        ggml_tensor * vl4 = ggml_reshape_4d(C, vl, fhd, fuse_heads, L, 1);
        ggml_tensor * ov = mha(C, qv4, kl4, vl4, t_fuse_mask, 1.0f);   // v-attend-text
        ov = ggml_reshape_2d(C, ov, fuse_embed, visual_tokens);
        ov = linear(C, ov, fw.out_v);
        ggml_tensor * ol = mha(C, kl4, qv4, vv4, nullptr, 1.0f);       // l-attend-vision
        ol = ggml_reshape_2d(C, ol, fuse_embed, L);
        ol = linear(C, ol, fw.out_l);
        // The released checkpoint uses residual_style="normalized": the
        // normalized inputs, not the pre-LN inputs, are the residual bases.
        vis = ggml_add(C, vn, ggml_mul(C, ov, reshape_col(C, fw.gamma_v, hidden)));
        txt = ggml_add(C, ln, ggml_mul(C, ol, reshape_col(C, fw.gamma_l, hidden)));

        const TenhLayerW & tw = tenh[(size_t) i];
        ggml_tensor * qkv = linear(C, txt, tw.attn_qkv);               // [3*hidden, L]
        const int64_t qkv_rows = 3 * hidden;
        ggml_tensor * q = row_slice(C, qkv, qkv_rows, 0, hidden);
        ggml_tensor * k = row_slice(C, qkv, qkv_rows, hidden, hidden);
        ggml_tensor * v = row_slice(C, qkv, qkv_rows, 2 * hidden, hidden);
        ggml_tensor * q4 = ggml_reshape_4d(C, q, ehd, fuse_heads, L, 1);
        ggml_tensor * k4 = ggml_reshape_4d(C, k, ehd, fuse_heads, L, 1);
        ggml_tensor * v4 = ggml_reshape_4d(C, v, ehd, fuse_heads, L, 1);
        ggml_tensor * o = mha(C, q4, k4, v4, t_enh_mask, 1.0f / std::sqrt((float) ehd));
        o = ggml_reshape_2d(C, o, hidden, L);
        o = linear(C, o, tw.attn_o);
        txt = ggml_add(C, txt, o);                                      // post-LN: residual
        txt = ln_f32(C, txt, tw.attn_norm, ln_eps_layer);
        ggml_tensor * f = linear(C, txt, tw.ffn_up);
        f = ggml_relu(C, f);
        f = linear(C, f, tw.ffn_down);
        txt = ln_f32(C, ggml_add(C, txt, f), tw.ffn_norm, ln_eps_layer);
    }

    // ---- 2e) condition + state tokens ----
    ggml_tensor * st = ln_f32(C, t_state, st_ln, ln_eps_layer);
    st = linear(C, st, st_fc1);                                         // [state_hidden]
    st = ggml_gelu_erf(C, st);
    st = linear(C, st, st_fc2);                                         // [n_state_tok*hidden]
    st = ggml_reshape_2d(C, st, hidden, n_state_tok);
    st = ggml_add(C, st, st_pos);
    st = ln_f32(C, st, st_out_norm, ln_eps_layer);
    ggml_tensor * memory = ggml_concat(C, ggml_concat(C, vis, txt, 1), st, 1);  // [hidden, V*N + L + 2]

    // ---- 2f) ACT decoder ----
    ggml_tensor * q = act_queries;                                      // [hidden, horizon]
    const int64_t mem_len = memory->ne[1];
    for (int64_t i = 0; i < act_layers; ++i) {
        const ActLayerW & w = act[(size_t) i];
        // self-attention over queries (no mask)
        ggml_tensor * sqkv = linear(C, ln_f32(C, q, w.norm1, ln_eps_layer), w.self_qkv);
        ggml_tensor * sq = row_slice(C, sqkv, 3 * hidden, 0, hidden);
        ggml_tensor * sk = row_slice(C, sqkv, 3 * hidden, hidden, hidden);
        ggml_tensor * sv = row_slice(C, sqkv, 3 * hidden, 2 * hidden, hidden);
        ggml_tensor * o = mha(C,
            ggml_reshape_4d(C, sq, ahd, nheads, horizon, 1),
            ggml_reshape_4d(C, sk, ahd, nheads, horizon, 1),
            ggml_reshape_4d(C, sv, ahd, nheads, horizon, 1),
            nullptr, 1.0f / std::sqrt((float) ahd));
        o = ggml_reshape_2d(C, o, hidden, horizon);
        q = ggml_add(C, q, linear(C, o, w.self_o));
        // cross-attention into memory
        ggml_tensor * cq = ln_f32(C, q, w.norm2, ln_eps_layer);
        ggml_tensor * cqkv = linear(C, cq, w.cross_qkv);
        cq = row_slice(C, cqkv, 3 * hidden, 0, hidden);
        ggml_tensor * ckv = linear(C, memory, w.cross_qkv);             // [3*hidden, mem]
        ggml_tensor * ck = row_slice(C, ckv, 3 * hidden, hidden, hidden);
        ggml_tensor * cv = row_slice(C, ckv, 3 * hidden, 2 * hidden, hidden);
        o = mha(C,
            ggml_reshape_4d(C, cq, ahd, nheads, horizon, 1),
            ggml_reshape_4d(C, ck, ahd, nheads, mem_len, 1),
            ggml_reshape_4d(C, cv, ahd, nheads, mem_len, 1),
            nullptr, 1.0f / std::sqrt((float) ahd));
        o = ggml_reshape_2d(C, o, hidden, horizon);
        q = ggml_add(C, q, linear(C, o, w.cross_o));
        // FFN
        ggml_tensor * f = linear(C, ln_f32(C, q, w.norm3, ln_eps_layer), w.ffn_up);
        f = ggml_relu(C, f);
        f = linear(C, f, w.ffn_down);
        q = ggml_add(C, q, f);
    }

    // ---- 2g) action head: ReLU MLP + tanh ----
    ggml_tensor * a = linear(C, q, head_fc1);
    a = ggml_relu(C, a);
    a = linear(C, a, head_fc2);
    a = ggml_relu(C, a);
    a = linear(C, a, head_fc3);                                          // [action_dim, horizon]
    a = ggml_tanh(C, a);
    ggml_set_output(a);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 1 << 16, false);
    ggml_build_forward_expand(gf, a);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "vla(turbovla): ggml_gallocr_alloc_graph failed (out of memory?)\n");
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }

    ggml_backend_tensor_set(t_patch, patch_cols.data(), 0, ggml_nbytes(t_patch));
    ggml_backend_tensor_set(t_bert_emb, bert_emb.data(), 0, ggml_nbytes(t_bert_emb));
    ggml_backend_tensor_set(t_state, state_norm.data(), 0, ggml_nbytes(t_state));
    ggml_backend_tensor_set(t_cos, rope_cos.data(), 0, ggml_nbytes(t_cos));
    ggml_backend_tensor_set(t_sin, rope_sin.data(), 0, ggml_nbytes(t_sin));
    ggml_backend_tensor_set(t_bert_mask, bert_mask.data(), 0, ggml_nbytes(t_bert_mask));
    ggml_backend_tensor_set(t_enh_mask, enh_mask.data(), 0, ggml_nbytes(t_enh_mask));
    ggml_backend_tensor_set(t_fuse_mask, fuse_mask.data(), 0, ggml_nbytes(t_fuse_mask));

    // The vision tower, BERT and the ACT head share one graph, so their
    // phases cannot be timed independently. ms_vision is therefore the
    // host-side vision preprocessing up to the graph launch, and the graph
    // execution itself is reported once as ms_inference; do not read either
    // as a per-phase breakdown.
    const auto ti0 = clk::now();
    stats.ms_vision = std::chrono::duration<float, std::milli>(ti0 - tv0).count();
    const ggml_status stt = ggml_backend_graph_compute(backend, gf);
    if (stt != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(turbovla): ggml_backend_graph_compute failed (%d)\n", (int) stt);
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }

    // [action_dim, horizon] row-major get == [step, dim] flatten
    std::vector<float> out((size_t) horizon * action_dim);
    ggml_backend_tensor_get(a, out.data(), 0, out.size() * sizeof(float));
    ggml_gallocr_free(galloc);
    ggml_free(C);
    stats.ms_inference = std::chrono::duration<float, std::milli>(clk::now() - ti0).count();

    // Released LIBERO denormalization: arm in world units, gripper sign.
    for (int64_t t = 0; t < horizon; ++t) {
        float * row = out.data() + (size_t) t * action_dim;
        for (int64_t j = 0; j < 6; ++j) {
            row[j] = 0.5f * (row[j] + 1.0f) * (action_max[(size_t) j] - action_min[(size_t) j]) + action_min[(size_t) j];
        }
        row[6] = (row[6] > 0.f) ? 1.f : ((row[6] < 0.f) ? -1.f : 1.f);
    }

    stats.ms_total = std::chrono::duration<float, std::milli>(clk::now() - t0).count();
    return out;
}



} // namespace (close anonymous; the factory is the public vla:: symbol)

// ---------------------------------------------------------------------------
// Factory: metadata + weights -> backend-resident model
// ---------------------------------------------------------------------------
std::unique_ptr<ModelArchBase> turbovla_create(const std::string & mmproj_path,
                                               const std::string & ckpt_path,
                                               const std::string & config_path) {
    if (!mmproj_path.empty()) {
        std::fprintf(stderr, "vla(turbovla): mmproj is not used (vision tower is baked into the ckpt)\n");
    }
    if (!config_path.empty()) {
        std::fprintf(stderr,
                     "vla(turbovla): external config overrides are not supported; "
                     "all runtime metadata must come from the GGUF\n");
        return nullptr;
    }
    if (ckpt_path.size() < 5 || ckpt_path.compare(ckpt_path.size() - 5, 5, ".gguf") != 0) {
        std::fprintf(stderr, "vla(turbovla): ckpt must be a GGUF produced by "
                              "scripts/convert_turbovla_to_gguf.py (got '%s')\n", ckpt_path.c_str());
        return nullptr;
    }

    auto m = std::make_unique<TurboVLAModelArch>();
    m->reader = std::make_unique<gguf_reader>();
    if (!m->reader->open(ckpt_path)) return nullptr;
    gguf_reader & g = *m->reader;
    if (!g.has_key("turbovla.architecture") || g.str("turbovla.architecture") != "turbovla") {
        std::fprintf(stderr, "vla(turbovla): '%s' is not a TurboVLA GGUF (turbovla.architecture missing/wrong)\n",
                     ckpt_path.c_str());
        return nullptr;
    }
    m->matmul_type = std::getenv("VLA_TURBOVLA_F32_WEIGHTS") ? GGML_TYPE_F32 : GGML_TYPE_BF16;

    static const char * required_keys[] = {
        "turbovla.hidden_dim", "turbovla.nheads", "turbovla.interaction_layers",
        "turbovla.enhancer_inner_dim", "turbovla.act_ffn_dim", "turbovla.action_dim",
        "turbovla.state_dim", "turbovla.horizon", "turbovla.num_state_tokens",
        "turbovla.act_layers", "turbovla.mlp_hidden_dim", "turbovla.state_hidden_dim",
        "turbovla.num_views", "turbovla.image_size", "turbovla.text_padding_length",
        "turbovla.bert_layers", "turbovla.bert_hidden", "turbovla.bert_vocab",
        "turbovla.vit_layers", "turbovla.vit_hidden", "turbovla.patch_size",
        "turbovla.vit_heads", "turbovla.num_register_tokens", "turbovla.rope_theta",
    };
    for (const char * key : required_keys) {
        if (!g.has_key(key)) {
            std::fprintf(stderr, "vla(turbovla): required GGUF metadata key missing: %s\n", key);
            return nullptr;
        }
    }

    m->hidden      = (int64_t) g.u32("turbovla.hidden_dim");
    m->nheads      = (int64_t) g.u32("turbovla.nheads");
    m->n_fuse      = (int64_t) g.u32("turbovla.interaction_layers");
    m->fuse_embed  = (int64_t) g.u32("turbovla.enhancer_inner_dim");
    m->act_ffn     = (int64_t) g.u32("turbovla.act_ffn_dim");
    m->action_dim  = (int64_t) g.u32("turbovla.action_dim");
    m->state_dim   = (int64_t) g.u32("turbovla.state_dim");
    m->horizon     = (int64_t) g.u32("turbovla.horizon");
    m->n_state_tok = (int64_t) g.u32("turbovla.num_state_tokens");
    m->act_layers  = (int64_t) g.u32("turbovla.act_layers");
    m->mlp_hidden  = (int64_t) g.u32("turbovla.mlp_hidden_dim");
    m->state_hidden= (int64_t) g.u32("turbovla.state_hidden_dim");
    m->num_views   = (int64_t) g.u32("turbovla.num_views");
    m->image_size  = (int64_t) g.u32("turbovla.image_size");
    m->text_len    = (int64_t) g.u32("turbovla.text_padding_length");
    m->bert_layers = (int64_t) g.u32("turbovla.bert_layers");
    m->bert_hidden = (int64_t) g.u32("turbovla.bert_hidden");
    const int64_t bert_vocab = (int64_t) g.u32("turbovla.bert_vocab");
    m->vit_layers  = (int64_t) g.u32("turbovla.vit_layers");
    m->vit_hidden  = (int64_t) g.u32("turbovla.vit_hidden");
    m->patch_size  = (int64_t) g.u32("turbovla.patch_size");
    m->vit_heads   = (int64_t) g.u32("turbovla.vit_heads");
    const int64_t n_reg = (int64_t) g.u32("turbovla.num_register_tokens");
    m->rope_theta  = (double) g.f32("turbovla.rope_theta");

    // derived geometry
    m->fuse_heads    = std::max<int64_t>(1, m->nheads / 2);
    m->grid          = m->image_size / m->patch_size;
    m->n_patches     = m->grid * m->grid;
    m->n_prefix_tok  = 1 + n_reg;
    m->visual_tokens = m->num_views * m->n_patches;
    m->bert_heads    = m->bert_hidden / 64;   // bert-base head_dim = 64

    if (m->hidden <= 0 || m->nheads <= 0 || m->n_fuse <= 0 || m->horizon <= 0 ||
        m->action_dim <= 0 || m->state_dim <= 0 || m->n_state_tok <= 0 ||
        m->act_layers <= 0 || m->num_views <= 0 || m->image_size <= 0 ||
        m->patch_size <= 0 || m->grid <= 0 || m->bert_layers <= 0 ||
        m->bert_hidden <= 0 || m->vit_layers <= 0 || m->vit_hidden <= 0 ||
        m->vit_heads <= 0 || m->n_prefix_tok <= 0 || m->text_len <= 2 ||
        m->hidden % m->nheads != 0 || m->fuse_embed % m->fuse_heads != 0 ||
        m->hidden % m->fuse_heads != 0 || m->bert_hidden % m->bert_heads != 0 ||
        m->vit_hidden % m->vit_heads != 0 || m->rope_theta <= 0.0) {
        std::fprintf(stderr, "vla(turbovla): invalid or inconsistent GGUF dimensions/metadata\n");
        return nullptr;
    }

    Config & c = m->cfg;
    c.hidden          = m->hidden;
    c.n_img           = m->visual_tokens;
    c.n_lang          = m->text_len;
    c.n_state         = m->n_state_tok;
    c.n_prefix        = m->visual_tokens + m->text_len + m->n_state_tok;
    c.n_suffix        = m->horizon;
    c.n_full          = c.n_prefix + c.n_suffix;
    c.n_layers        = m->n_fuse;
    c.max_state_dim   = m->state_dim;
    c.max_action_dim  = m->action_dim;
    c.real_state_dim  = m->state_dim;
    c.real_action_dim = m->action_dim;
    c.num_steps       = (int) m->horizon;

    std::printf("vla(turbovla): hidden=%lld nheads=%lld fuse=%lldx%lld(embed=%lld) "
                "bert=%lldx%lld vit=%lldx%lld(p=%lld,heads=%lld,reg=%lld) views=%lld img=%lld "
                "text_len=%lld act=%lldx%lld ffn=%lld horizon=%lld action=%lld state=%lld "
                "matmul_weights=%s\n",
                (long long) m->hidden, (long long) m->nheads,
                (long long) m->n_fuse, (long long) m->fuse_heads, (long long) m->fuse_embed,
                (long long) m->bert_layers, (long long) m->bert_hidden,
                (long long) m->vit_layers, (long long) m->vit_hidden,
                (long long) m->patch_size, (long long) m->vit_heads, (long long) n_reg,
                (long long) m->num_views, (long long) m->image_size,
                (long long) m->text_len, (long long) m->act_layers, (long long) m->horizon,
                (long long) m->act_ffn, (long long) m->horizon, (long long) m->action_dim,
                (long long) m->state_dim,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

    // WordPiece vocab (bundled)
    if (!m->tokenizer.load(g, "turbovla.bert_vocab_list", bert_vocab)) {
        std::fprintf(stderr, "vla(turbovla): bundled vocab missing or size mismatch "
                             "(turbovla.bert_vocab_list)\n");
        return nullptr;
    }

    // backend
    {
        const unsigned hw = std::thread::hardware_concurrency();
        m->n_threads = (hw == 0) ? 4 : (int) std::min(hw, 8u);
    }
    m->is_cuda = false;
#ifdef GGML_USE_CUDA
    m->backend = ggml_backend_cuda_init(0);
    if (m->backend) { m->is_cuda = true; std::printf("vla(turbovla): backend = CUDA (device 0)\n"); }
    else            { std::fprintf(stderr, "vla(turbovla): ggml_backend_cuda_init failed; falling back to CPU\n"); }
#endif
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) { std::fprintf(stderr, "vla(turbovla): ggml_backend_cpu_init failed\n"); return nullptr; }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(turbovla): backend = CPU (%d threads)\n", m->n_threads);
    }

    // weight context
    {
        ggml_init_params wp = { (size_t) 32 * 1024 * 1024, nullptr, true };
        m->ctx_weights = ggml_init(wp);
        if (!m->ctx_weights) { std::fprintf(stderr, "vla(turbovla): ggml_init(ctx_weights) failed\n"); return nullptr; }
    }
    ggml_context * W = m->ctx_weights;
    std::vector<ggml_tensor *> weights;
    std::vector<uint8_t> vit_patch_upload;   // rearranged conv kernel (vit.patch_conv.w)

    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) { std::fprintf(stderr, "vla(turbovla): missing tensor %s\n", name); return nullptr; }
        ggml_tensor * t = ggml_new_tensor(W, type, GGML_MAX_DIMS, gt->ne);
        ggml_set_name(t, name);
        weights.push_back(t);
        return t;
    };
    auto mk_mm  = [&](const char * name) -> ggml_tensor * { return mk(name, m->matmul_type); };
    auto mk_f32 = [&](const char * name) -> ggml_tensor * { return mk(name, GGML_TYPE_F32); };
    auto tensor_exists = [&](const char * name) -> bool {
        return gguf_find_tensor(g.gctx, name) >= 0;
    };
    // pair helper: matmul weight + f32 bias ("dst.w"/"dst.b" naming)
    auto mk_wb  = [&](const char * stem) -> WB {
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

    // ---- text encoder ----
    m->text_emb_norm = mk_wb_f32("text.emb_norm");
    m->text_proj     = mk_wb("text.proj");
    m->bert.resize((size_t) m->bert_layers);
    for (int64_t i = 0; i < m->bert_layers; ++i) {
        char stem[96];
        BertLayerW & w = m->bert[(size_t) i];
        auto pick = [&](const char * suffix, WB & dst) {
            std::snprintf(stem, sizeof(stem), "text.blk.%lld.%s", (long long) i, suffix);
            dst = mk_wb(stem);
        };
        // LayerNorm scale enters an element-wise ggml_mul in ln_f32(); the CUDA
        // backend requires F32 there (binbcast.cu: src1 must be F32/F16), so
        // norms must never be stored in the (BF16) matmul type.
        auto pick_norm = [&](const char * suffix, WB & dst) {
            std::snprintf(stem, sizeof(stem), "text.blk.%lld.%s", (long long) i, suffix);
            dst = mk_wb_f32(stem);
        };
        pick("attn_q",     w.attn_q);
        pick("attn_k",     w.attn_k);
        pick("attn_v",     w.attn_v);
        pick("attn_o",     w.attn_o);
        pick_norm("attn_norm",  w.attn_norm);
        pick("ffn_up",     w.ffn_up);
        pick("ffn_down",   w.ffn_down);
        pick_norm("ffn_norm",   w.ffn_norm);
    }

    // ---- DINOv3 vision tower ----
    {
        // PyTorch conv weight is [vit_hidden, 3, p, p].  GGUF stores it as
        // [p, p, 3, vit_hidden] (numpy row-major → ggml ne[0]=p, ne[3]=vit_hidden).
        // The 2D matmul layout [3*p*p, vit_hidden] is NOT byte-identical to
        // the 4D storage, so we must transpose the data on load.
        const ggml_tensor * cw = g.meta("vit.patch_conv.w");
        if (!cw) { std::fprintf(stderr, "vla(turbovla): missing tensor vit.patch_conv.w\n"); return nullptr; }
        const int64_t P = m->patch_size;
        const int64_t chwp = 3 * P * P;
        if (cw->ne[3] != m->vit_hidden || chwp != 3 * P * P) {
            std::fprintf(stderr, "vla(turbovla): vit.patch_conv.w shape mismatch "
                                 "(ne=[%lld,%lld,%lld,%lld], expected 3*%lld*%lld cols=%lld)\n",
                         (long long) cw->ne[0], (long long) cw->ne[1],
                         (long long) cw->ne[2], (long long) cw->ne[3],
                         (long long) P, (long long) P, (long long) chwp);
            return nullptr;
        }
        // Read raw 4D data, rearrange to 2D [chwp, vit_hidden].
        const int64_t nd = m->vit_hidden * chwp;
        const ggml_type src_type = cw->type;
        std::vector<uint8_t> raw4d((size_t) ggml_nbytes(cw));
        if (!g.read_raw("vit.patch_conv.w", raw4d.data())) return nullptr;
        const size_t elsz = (src_type == GGML_TYPE_F32) ? 4u : 2u;
        // BF16 → F32 for the rearrange, then convert back
        std::vector<float> f32_rearr((size_t) nd);
        const size_t nb0 = (size_t) elsz;           // stride kw
        const size_t nb1 = (size_t) P * elsz;       // stride kh
        const size_t nb2 = (size_t) P * P * elsz;   // stride c
        const size_t nb3 = (size_t) chwp * elsz;    // stride o
        for (int64_t o = 0; o < m->vit_hidden; ++o) {
            for (int64_t c = 0; c < 3; ++c) {
                for (int64_t kh = 0; kh < P; ++kh) {
                    for (int64_t kw = 0; kw < P; ++kw) {
                        const size_t src_off = (size_t) kw * nb0 + (size_t) kh * nb1 +
                                               (size_t) c * nb2 + (size_t) o * nb3;
                        float val;
                        if (src_type == GGML_TYPE_F32) {
                            std::memcpy(&val, &raw4d[src_off], 4);
                        } else {
                            ggml_bf16_t b; std::memcpy(&b, &raw4d[src_off], 2);
                            val = ggml_bf16_to_fp32(b);
                        }
                        const int64_t flat = c * P * P + kh * P + kw;
                        f32_rearr[(size_t) (o * chwp + flat)] = val;
                    }
                }
            }
        }
        m->vit_patch.w = ggml_new_tensor_2d(W, m->matmul_type, chwp, m->vit_hidden);
        ggml_set_name(m->vit_patch.w, "vit.patch_conv.w");
        weights.push_back(m->vit_patch.w);
        // Convert the rearranged data to the target type and keep it for the
        // upload loop below. The original 4D GGUF storage is NOT byte-identical
        // to this 2D column-major layout, so we must NOT use read_convert() here.
        vit_patch_upload.resize((size_t) ggml_nbytes(m->vit_patch.w));
        if (m->matmul_type == GGML_TYPE_F32) {
            std::memcpy(vit_patch_upload.data(), f32_rearr.data(), vit_patch_upload.size());
        } else {
            ggml_fp32_to_bf16_row(f32_rearr.data(),
                                  reinterpret_cast<ggml_bf16_t *>(vit_patch_upload.data()), nd);
        }
    }
    m->vit_patch.b = mk_f32("vit.patch_conv.b");
    m->vit_cls     = mk_f32("vit.cls_token");
    m->vit_reg     = mk_f32("vit.reg_tokens");
    m->vit_out_norm= mk_wb_f32("vit.output_norm");
    m->vit.resize((size_t) m->vit_layers);
    for (int64_t i = 0; i < m->vit_layers; ++i) {
        char stem[96];
        VitLayerW & w = m->vit[(size_t) i];
        auto mm_pair = [&](const char * suffix, bool has_bias) {
            std::snprintf(stem, sizeof(stem), "vit.blk.%lld.%s", (long long) i, suffix);
            WB dst = mk_wb(stem);
            if (!has_bias) dst.b = nullptr;   // k_proj has no bias in DINOv3
            return dst;
        };
        w.attn_norm = mk_wb_f32((std::string("vit.blk.") + std::to_string(i) + ".attn_norm").c_str());
        w.attn_q    = mm_pair("attn_q", true);
        w.attn_k    = mm_pair("attn_k", false);
        w.attn_v    = mm_pair("attn_v", true);
        w.attn_o    = mm_pair("attn_o", true);
        std::snprintf(stem, sizeof(stem), "vit.blk.%lld.ls1", (long long) i);
        w.ls1 = mk_f32(stem);
        w.ffn_norm = mk_wb_f32((std::string("vit.blk.") + std::to_string(i) + ".ffn_norm").c_str());
        w.ffn_up   = mm_pair("ffn_up", true);
        w.ffn_down = mm_pair("ffn_down", true);
        std::snprintf(stem, sizeof(stem), "vit.blk.%lld.ls2", (long long) i);
        w.ls2 = mk_f32(stem);
    }

    // ---- vision projection + view embedding ----
    m->vp_input_norm = mk_wb_f32("vproj.input_norm");
    m->vp_fc1        = mk_wb("vproj.mlp_fc1");
    m->vp_fc2        = mk_wb("vproj.mlp_fc2");
    {
        const ggml_tensor * sw = g.meta("vproj.skip_w");
        if (!sw) { std::fprintf(stderr, "vla(turbovla): missing tensor vproj.skip_w\n"); return nullptr; }
        m->vp_skip.w = mk_mm("vproj.skip_w");
        m->vp_skip.b = nullptr;
    }
    m->vp_out_norm = mk_wb_f32("vproj.output_norm");
    m->view_emb    = mk_f32("pos.view_emb");

    // ---- fusion + text enhancer ----
    m->fuse.resize((size_t) m->n_fuse);
    m->tenh.resize((size_t) m->n_fuse);
    for (int64_t i = 0; i < m->n_fuse; ++i) {
        char stem[96];
        FuseLayerW & fw = m->fuse[(size_t) i];
        std::snprintf(stem, sizeof(stem), "fuse.%lld.ln_v", (long long) i);
        fw.ln_v = mk_wb_f32(stem);
        std::snprintf(stem, sizeof(stem), "fuse.%lld.ln_l", (long long) i);
        fw.ln_l = mk_wb_f32(stem);
        std::snprintf(stem, sizeof(stem), "fuse.%lld.q_from_v", (long long) i);
        fw.q_from_v = mk_wb(stem);
        std::snprintf(stem, sizeof(stem), "fuse.%lld.k_from_l", (long long) i);
        fw.k_from_l = mk_wb(stem);
        std::snprintf(stem, sizeof(stem), "fuse.%lld.val_from_v", (long long) i);
        fw.val_from_v = mk_wb(stem);
        std::snprintf(stem, sizeof(stem), "fuse.%lld.val_from_l", (long long) i);
        fw.val_from_l = mk_wb(stem);
        std::snprintf(stem, sizeof(stem), "fuse.%lld.out_v", (long long) i);
        fw.out_v = mk_wb(stem);
        std::snprintf(stem, sizeof(stem), "fuse.%lld.out_l", (long long) i);
        fw.out_l = mk_wb(stem);
        std::snprintf(stem, sizeof(stem), "fuse.%lld.gamma_v", (long long) i);
        fw.gamma_v = mk_f32(stem);
        std::snprintf(stem, sizeof(stem), "fuse.%lld.gamma_l", (long long) i);
        fw.gamma_l = mk_f32(stem);

        TenhLayerW & tw = m->tenh[(size_t) i];
        auto fused = [&](const char * suffix) {
            char ws[128];
            std::snprintf(ws, sizeof(ws), "tenh.%lld.%s_w", (long long) i, suffix);
            ggml_tensor * w = mk_mm(ws);
            std::snprintf(ws, sizeof(ws), "tenh.%lld.%s_b", (long long) i, suffix);
            ggml_tensor * b = mk_f32(ws);
            return WB{w, b};
        };
        tw.attn_qkv = fused("attn_qkv");
        tw.attn_o   = mk_wb((std::string("tenh.") + std::to_string(i) + ".attn_o").c_str());
        tw.ffn_up   = mk_wb((std::string("tenh.") + std::to_string(i) + ".ffn_up").c_str());
        tw.ffn_down = mk_wb((std::string("tenh.") + std::to_string(i) + ".ffn_down").c_str());
        tw.attn_norm= mk_wb_f32((std::string("tenh.") + std::to_string(i) + ".attn_norm").c_str());
        tw.ffn_norm = mk_wb_f32((std::string("tenh.") + std::to_string(i) + ".ffn_norm").c_str());
    }

    // ---- action head ----
    m->st_ln       = mk_wb_f32("act.state_ln");
    m->st_fc1      = mk_wb("act.state_fc1");
    m->st_fc2      = mk_wb("act.state_fc2");
    m->st_pos      = mk_f32("act.state_pos");
    m->st_out_norm = mk_wb_f32("act.state_out_norm");
    m->act_queries = mk_f32("act.queries");
    m->act.resize((size_t) m->act_layers);
    for (int64_t i = 0; i < m->act_layers; ++i) {
        char stem[96];
        ActLayerW & w = m->act[(size_t) i];
        auto fused = [&](const char * suffix) {
            char ws[128];
            std::snprintf(ws, sizeof(ws), "act.blk.%lld.%s_w", (long long) i, suffix);
            ggml_tensor * wt = mk_mm(ws);
            std::snprintf(ws, sizeof(ws), "act.blk.%lld.%s_b", (long long) i, suffix);
            ggml_tensor * b = mk_f32(ws);
            return WB{wt, b};
        };
        w.self_qkv  = fused("self_qkv");
        w.self_o    = mk_wb((std::string("act.blk.") + std::to_string(i) + ".self_o").c_str());
        w.cross_qkv = fused("cross_qkv");
        w.cross_o   = mk_wb((std::string("act.blk.") + std::to_string(i) + ".cross_o").c_str());
        w.ffn_up    = mk_wb((std::string("act.blk.") + std::to_string(i) + ".ffn_up").c_str());
        w.ffn_down  = mk_wb((std::string("act.blk.") + std::to_string(i) + ".ffn_down").c_str());
        w.norm1     = mk_wb_f32((std::string("act.blk.") + std::to_string(i) + ".norm1").c_str());
        w.norm2     = mk_wb_f32((std::string("act.blk.") + std::to_string(i) + ".norm2").c_str());
        w.norm3     = mk_wb_f32((std::string("act.blk.") + std::to_string(i) + ".norm3").c_str());
        (void) stem;
    }
    m->head_fc1 = mk_wb("act.head_fc1");
    m->head_fc2 = mk_wb("act.head_fc2");
    m->head_fc3 = mk_wb("act.head_fc3");

    for (ggml_tensor * t : weights) if (!t) {
        std::fprintf(stderr, "vla(turbovla): weight tensor creation failed\n");
        return nullptr;
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) { std::fprintf(stderr, "vla(turbovla): ggml_backend_alloc_ctx_tensors failed (out of memory?)\n"); return nullptr; }
    for (ggml_tensor * t : weights) {
        if (t == m->vit_patch.w) {
            ggml_backend_tensor_set(t, vit_patch_upload.data(), 0, vit_patch_upload.size());
            continue;
        }
        std::vector<uint8_t> bytes = g.read_convert(t->name, t->type);
        if (bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(turbovla): upload size mismatch for %s (%zu vs %zu)\n",
                         t->name, bytes.size(), ggml_nbytes(t));
            return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::printf("vla(turbovla): resident weights = %.2f GiB\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0));

    // normalization statistics (host, F32)
    auto read_stats = [&](const char * name, std::vector<float> & dst, int64_t n) {
        dst.resize((size_t) n);
        const ggml_tensor * t = g.meta(name);
        if (!t || t->type != GGML_TYPE_F32 || ggml_nelements(t) != n) {
            std::fprintf(stderr, "vla(turbovla): stats tensor %s missing or malformed\n", name);
            return false;
        }
        return g.read_raw(name, dst.data());
    };
    if (!read_stats("norm.proprio_mean", m->proprio_mean, m->state_dim) ||
        !read_stats("norm.proprio_std",  m->proprio_std,  m->state_dim) ||
        !read_stats("norm.action_min",   m->action_min,   m->action_dim) ||
        !read_stats("norm.action_max",   m->action_max,   m->action_dim)) {
        return nullptr;
    }

    std::printf("vla(turbovla): model loaded (n_threads=%d)\n", m->n_threads);
    return m;
}

} // namespace vla
