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

// Xiaomi-Robotics-0 (MiBoT) runtime: Qwen3-VL-4B backbone + 16-layer DiT
// flow-matching action head.
//
// Pipeline (mirrors modeling_mibot.MiBoTForActionGeneration.forward):
//
//   1. vision   : llama.cpp clip (qwen3vl mmproj) encodes each view; the
//                 per-token output is [main(hidden) | ds0 | ds1 | ds2]; the
//                 main features scatter into image-token slots and the
//                 deepstack features are injected after text layers 0-2.
//   2. backbone : 36 Qwen3-VL decoder layers with interleaved MRoPE
//                 (GGML_ROPE_TYPE_IMROPE, sections 24/20/20, theta 5e6);
//                 q/k RMSNorm on head_dim before rope; GQA 32q/8kv comes
//                 free with ggml_mul_mat's group-query head broadcast.
//   3. action   : DiT queries [sink, state, chunk x noisy_action] cross-
//                 attend to the backbone K/V of layers kv_start..35
//                 (36-16) and self-attend causally; adaLN modulation comes
//                 from adaln_table + t_projector(TimestepEmbedder(t*1000)).
//                 Flow matching integrates t = 0 -> 1 in num_steps Euler
//                 steps (dt = +1/num_steps), the reverse of pi0.5.
//
// The caller pre-tokenizes the Qwen chat prompt (with one <|image_pad|>
// per merged 2x2 patch) and passes it through Inputs::lang_tokens; images
// arrive as raw RGB views whose sides must be multiples of 32 (LIBERO
// 256x256 -> 8x8=64 tokens per view, matching the HF processor's
// smart_resize no-op). Inputs::noise seeds the initial flow sample
// (chunk * action_dim floats, row-major per action step).

#include "arch.h"
#include "model.h"

#include "clip.h"
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
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
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
            std::fprintf(stderr, "vla(xr0): gguf_init_from_file failed for %s\n", path.c_str());
            return false;
        }
        fp = std::fopen(path.c_str(), "rb");
        if (!fp) {
            std::fprintf(stderr, "vla(xr0): fopen failed for %s\n", path.c_str());
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

    bool has_key(const char * k) const { return gguf_find_key(gctx, k) >= 0; }
    uint32_t    u32(const char * k) const { return gguf_get_val_u32(gctx, gguf_find_key(gctx, k)); }
    float       f32(const char * k) const { return gguf_get_val_f32(gctx, gguf_find_key(gctx, k)); }
    std::string str(const char * k) const { return gguf_get_val_str(gctx, gguf_find_key(gctx, k)); }

    const ggml_tensor * meta(const char * name) const { return ggml_get_tensor(meta_ctx, name); }

    bool read_raw(const char * name, void * buf) {
        const int64_t id = gguf_find_tensor(gctx, name);
        if (id < 0) { std::fprintf(stderr, "vla(xr0): missing tensor %s\n", name); return false; }
        const size_t off = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t nb  = gguf_get_tensor_size(gctx, id);
        if (std::fseek(fp, (long) off, SEEK_SET) != 0) return false;
        return std::fread(buf, 1, nb, fp) == nb;
    }

    std::vector<uint8_t> read_convert(const char * name, ggml_type target) {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(xr0): missing tensor %s\n", name); return {}; }
        const int64_t n = ggml_nelements(t);

        // quantized payload passes through untouched when types match
        if (t->type == target && ggml_is_quantized(t->type)) {
            std::vector<uint8_t> out(ggml_nbytes(t));
            if (!read_raw(name, out.data())) return {};
            return out;
        }

        std::vector<float> f32(n);
        if (t->type == GGML_TYPE_F32) {
            if (!read_raw(name, f32.data())) return {};
        } else if (t->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> tmp(n);
            if (!read_raw(name, tmp.data())) return {};
            ggml_bf16_to_fp32_row(tmp.data(), f32.data(), n);
        } else {
            std::fprintf(stderr, "vla(xr0): tensor %s has unsupported type %d\n", name, (int) t->type);
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
        std::fprintf(stderr, "vla(xr0): unsupported resident type %d for %s\n", (int) target, name);
        return {};
    }

    bool fetch_rows_f32(const char * name, const std::vector<int32_t> & row_ids,
                        float * dst, int64_t cols) {
        const ggml_tensor * t = meta(name);
        if (!t) { std::fprintf(stderr, "vla(xr0): missing tensor %s\n", name); return false; }
        if (t->ne[0] != cols || t->ne[2] != 1 || t->ne[3] != 1) {
            std::fprintf(stderr, "vla(xr0): %s shape unfit for row-fetch\n", name); return false;
        }
        const int64_t rows = t->ne[1];
        const int64_t id   = gguf_find_tensor(gctx, name);
        const size_t  base = data_off + gguf_get_tensor_offset(gctx, id);
        const size_t  elsz = (t->type == GGML_TYPE_F32) ? 4u : 2u;
        const size_t  rb   = (size_t) cols * elsz;
        std::vector<uint8_t> row(rb);
        for (size_t k = 0; k < row_ids.size(); ++k) {
            const int32_t r = row_ids[k];
            if (r < 0 || r >= rows) {
                std::fprintf(stderr, "vla(xr0): row %d out of range for %s\n", r, name); return false;
            }
            if (std::fseek(fp, (long) (base + (size_t) r * rb), SEEK_SET) != 0) return false;
            if (std::fread(row.data(), 1, rb, fp) != rb) return false;
            if (elsz == 4) std::memcpy(dst + k * cols, row.data(), rb);
            else ggml_bf16_to_fp32_row(reinterpret_cast<ggml_bf16_t *>(row.data()), dst + k * cols, cols);
        }
        return true;
    }
};

struct VlmLayerW {
    ggml_tensor * ln_in   = nullptr;
    ggml_tensor * Wq      = nullptr;
    ggml_tensor * Wk      = nullptr;
    ggml_tensor * Wv      = nullptr;
    ggml_tensor * Wo      = nullptr;
    ggml_tensor * q_norm  = nullptr;
    ggml_tensor * k_norm  = nullptr;
    ggml_tensor * ln_post = nullptr;
    ggml_tensor * Wgate   = nullptr;
    ggml_tensor * Wup     = nullptr;
    ggml_tensor * Wdown   = nullptr;
};

struct DitLayerW {
    ggml_tensor * ln_in   = nullptr;
    ggml_tensor * ln_mid  = nullptr;
    ggml_tensor * ln_post = nullptr;
    ggml_tensor * ln_fin  = nullptr;
    ggml_tensor * adaln   = nullptr;   // [dit_h, 6]
    ggml_tensor * Wqkv    = nullptr;
    ggml_tensor * bqkv    = nullptr;
    ggml_tensor * q_norm  = nullptr;
    ggml_tensor * k_norm  = nullptr;
    ggml_tensor * Wo      = nullptr;
    ggml_tensor * Wgate   = nullptr;
    ggml_tensor * Wup     = nullptr;
    ggml_tensor * Wdown   = nullptr;
};

// HF TimestepEmbedder.timestep_embedding: [cos(w t), sin(w t)] halves.
std::vector<float> timestep_freq_emb(double t, int64_t dim, double max_period) {
    const int64_t half = dim / 2;
    std::vector<float> out(dim);
    for (int64_t i = 0; i < half; ++i) {
        const double freq = std::exp(-std::log(max_period) * double(i) / double(half));
        const double arg  = t * freq;
        out[i]        = (float) std::cos(arg);
        out[half + i] = (float) std::sin(arg);
    }
    return out;
}

// One Qwen3-VL decoder layer. Exports post-rope K and raw V per layer so the
// DiT head can cross-attend to the backbone KV cache.
ggml_tensor * build_vlm_layer(
        ggml_context * ctx, const VlmLayerW & w,
        ggml_tensor * x_in, ggml_tensor * pos4,
        const Config & cfg, const int rope_sections[4], float rope_base,
        ggml_tensor * mask, ggml_tensor ** k_out, ggml_tensor ** v_out) {
    const int64_t hd  = cfg.head_dim;
    const int64_t nq  = cfg.n_q_heads;
    const int64_t nkv = cfg.n_kv_heads;
    const int64_t seq = x_in->ne[1];
    const int64_t qf  = nq * hd;

    ggml_tensor * x_norm = ggml_mul(ctx, ggml_rms_norm(ctx, x_in, cfg.rms_eps), w.ln_in);

    ggml_tensor * q = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w.Wq, x_norm), hd, nq,  seq);
    ggml_tensor * k = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w.Wk, x_norm), hd, nkv, seq);
    ggml_tensor * v = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, w.Wv, x_norm), hd, nkv, seq);

    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, cfg.rms_eps), w.q_norm);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, cfg.rms_eps), w.k_norm);

    auto rope_call = [&](ggml_tensor * t) {
        return ggml_rope_multi(ctx, t, pos4, nullptr,
                               (int) hd, const_cast<int *>(rope_sections),
                               GGML_ROPE_TYPE_IMROPE, 32768,
                               rope_base, 1.f, 0.f, 1.f, 32.f, 1.f);
    };
    ggml_tensor * q_rope = rope_call(q);
    ggml_tensor * k_rope = rope_call(k);

    if (k_out) *k_out = k_rope;
    if (v_out) *v_out = v;

    ggml_tensor * Q = ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3));
    ggml_tensor * K = ggml_cont(ctx, ggml_permute(ctx, k_rope, 0, 2, 1, 3));
    ggml_tensor * V = ggml_cont(ctx, ggml_permute(ctx, v,       1, 2, 0, 3));

    ggml_tensor * kq = ggml_mul_mat(ctx, K, Q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    const float scale = 1.f / std::sqrt((float) hd);
    ggml_tensor * attn = ggml_soft_max_ext(ctx, kq, mask, scale, 0.f);
    ggml_tensor * kqv  = ggml_mul_mat(ctx, V, attn);

    ggml_tensor * att_pre = ggml_reshape_2d(ctx,
        ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), qf, seq);
    ggml_tensor * h1 = ggml_add(ctx, x_in, ggml_mul_mat(ctx, w.Wo, att_pre));

    ggml_tensor * x_norm_mlp = ggml_mul(ctx, ggml_rms_norm(ctx, h1, cfg.rms_eps), w.ln_post);
    ggml_tensor * gate    = ggml_mul_mat(ctx, w.Wgate, x_norm_mlp);
    ggml_tensor * up      = ggml_mul_mat(ctx, w.Wup,   x_norm_mlp);
    ggml_tensor * mlp_out = ggml_mul_mat(ctx, w.Wdown,
                                ggml_mul(ctx, ggml_silu(ctx, gate), up));
    return ggml_add(ctx, h1, mlp_out);
}

} // namespace

struct XiaomiRobotics0ModelArch : public ModelArchBase {
    XiaomiRobotics0ModelArch() : ModelArchBase(Arch::XR0) {}
    ~XiaomiRobotics0ModelArch() override;

    std::vector<float> predict(const Inputs& in) override;

    clip_ctx *            cctx        = nullptr;
    ggml_backend_t        backend     = nullptr;
    bool                  is_cuda     = false;
    ggml_backend_buffer_t weight_buf  = nullptr;
    ggml_context *        ctx_weights = nullptr;
    std::string           ckpt_path_;
    ggml_type             matmul_type = GGML_TYPE_BF16;

    std::vector<VlmLayerW> vlm_layers;
    std::vector<DitLayerW> dit_layers;

    ggml_tensor * sink_w   = nullptr;   // [dit_h, 1]
    ggml_tensor * W_state0 = nullptr, * W_state2 = nullptr;
    ggml_tensor * W_ain0   = nullptr, * W_ain2   = nullptr;
    ggml_tensor * W_aout0  = nullptr, * W_aout2  = nullptr;
    ggml_tensor * W_temb0  = nullptr, * W_temb2  = nullptr;
    ggml_tensor * W_tproj  = nullptr, * b_tproj  = nullptr;

    // dims beyond vla::Config
    int64_t dit_h        = 1024;
    int64_t dit_layers_n = 16;
    int64_t dit_kv_start = 20;
    int     rope_sections[4] = {24, 20, 20, 0};
    int32_t image_token_id    = 151655;

    std::vector<float> action_mean, action_std;  // [max_action_dim]
    std::vector<float> action_mask;              // std > 1e-5

    std::mt19937 rng{std::random_device{}()};
    int n_threads = 4;
};

XiaomiRobotics0ModelArch::~XiaomiRobotics0ModelArch() {
    if (weight_buf)  ggml_backend_buffer_free(weight_buf);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend)     ggml_backend_free(backend);
    if (cctx)        clip_free(cctx);
}

std::unique_ptr<ModelArchBase> xr0_create(const std::string& mmproj_path,
                                          const std::string& ckpt_path,
                                          const std::string& config_path) {
    (void) config_path;

    auto m = std::make_unique<XiaomiRobotics0ModelArch>();
    m->ckpt_path_  = ckpt_path;
    m->matmul_type = std::getenv("VLA_XR0_F32_WEIGHTS") ? GGML_TYPE_F32 : GGML_TYPE_BF16;

    gguf_reader g;
    if (!g.open(ckpt_path)) return nullptr;
    if (!g.has_key("xr0.architecture") || g.str("xr0.architecture") != "xr0") {
        std::fprintf(stderr, "vla(xr0): '%s' is not a Xiaomi-Robotics-0 GGUF (xr0.architecture missing/wrong)\n",
                     ckpt_path.c_str());
        return nullptr;
    }

    Config & cfg = m->cfg;
    cfg = Config{};
    cfg.hidden          = g.u32("xr0.vlm.hidden");
    cfg.intermediate    = g.u32("xr0.vlm.intermediate");
    cfg.n_q_heads       = g.u32("xr0.vlm.n_q_heads");
    cfg.n_kv_heads      = g.u32("xr0.vlm.n_kv_heads");
    cfg.head_dim        = g.u32("xr0.vlm.head_dim");
    cfg.n_layers        = g.u32("xr0.vlm.n_layers");
    cfg.rms_eps         = g.has_key("xr0.vlm.rms_norm_eps") ? g.f32("xr0.vlm.rms_norm_eps") : 1e-6f;
    cfg.rope_freq_base  = g.has_key("xr0.vlm.rope_theta")   ? g.f32("xr0.vlm.rope_theta")   : 5e6f;
    cfg.expert_h        = g.u32("xr0.dit.hidden");
    cfg.expert_inter    = cfg.expert_h * 4;
    m->dit_layers_n     = g.u32("xr0.dit.n_layers");
    m->dit_kv_start     = g.u32("xr0.dit.kv_start_layer");
    cfg.n_suffix        = g.u32("xr0.action_length");
    cfg.max_action_dim  = g.u32("xr0.action_dim");
    cfg.real_action_dim = cfg.max_action_dim;
    cfg.max_state_dim   = g.u32("xr0.state_dim");
    cfg.real_state_dim  = cfg.max_state_dim;
    cfg.num_steps       = g.has_key("xr0.num_steps") ? (int) g.u32("xr0.num_steps") : 5;
    // The checkpoint's action_length (30) is the training horizon; the
    // official LIBERO deployment builds action_mask from the per-timestep
    // stats (10 steps). Allow overriding at load time to match.
    if (const char * chunk_env = std::getenv("VLA_XR0_CHUNK")) {
        const int c = std::atoi(chunk_env);
        if (c > 0 && c <= 64) cfg.n_suffix = c;
    }
    m->image_token_id   = g.has_key("xr0.vlm.image_token_id")
                            ? (int32_t) g.u32("xr0.vlm.image_token_id") : 151655;
    // rope sections are fixed to the Qwen3-VL-4B config (24/20/20) and are
    // asserted by the converter; see also the IMROPE layout note at the top.
    m->dit_h            = cfg.expert_h;
    cfg.q_full_dim      = cfg.n_q_heads  * cfg.head_dim;
    cfg.kv_full_dim     = cfg.n_kv_heads * cfg.head_dim;
    cfg.n_img           = 0;   // dynamic: token count depends on image size
    cfg.n_lang          = 512; // capacity hint used by the server-side check
    cfg.n_state         = 0;
    cfg.n_prefix        = 0;
    cfg.n_full          = 0;
    cfg.self_attn_every_n = 0;
    cfg.norm_eps        = cfg.rms_eps;
    cfg.rope_mode       = GGML_ROPE_TYPE_IMROPE;
    cfg.rope_n_dims     = (int) cfg.head_dim;
    cfg.min_period      = 10000.0;
    cfg.max_period      = 10000.0;

    {
        const int64_t km = gguf_find_key(g.gctx, "xr0.action_mean");
        const int64_t ks = gguf_find_key(g.gctx, "xr0.action_std");
        if (km < 0 || ks < 0) {
            std::fprintf(stderr, "vla(xr0): gguf missing action stats\n");
            return nullptr;
        }
        const uint32_t n = gguf_get_arr_n(g.gctx, km);
        if (n != (uint32_t) cfg.max_action_dim ||
            gguf_get_arr_n(g.gctx, ks) != n) {
            std::fprintf(stderr, "vla(xr0): action stats length mismatch\n");
            return nullptr;
        }
        const float * mean = (const float *) gguf_get_arr_data(g.gctx, km);
        const float * std_ = (const float *) gguf_get_arr_data(g.gctx, ks);
        m->action_mean.assign(mean, mean + n);
        m->action_std .assign(std_, std_ + n);
        m->action_mask.resize(n);
        for (uint32_t i = 0; i < n; ++i)
            m->action_mask[i] = (m->action_std[i] > 1e-5f) ? 1.f : 0.f;
    }

    std::printf("vla(xr0): vlm %lldL hidden=%lld heads=%lldq/%lldkv x%lld theta=%.0f | "
                "dit %lldL hidden=%lld kv_start=%lld | chunk=%lld action_dim=%lld steps=%d "
                "matmul=%s\n",
                (long long) cfg.n_layers, (long long) cfg.hidden,
                (long long) cfg.n_q_heads, (long long) cfg.n_kv_heads,
                (long long) cfg.head_dim, (double) cfg.rope_freq_base,
                (long long) m->dit_layers_n, (long long) m->dit_h,
                (long long) m->dit_kv_start,
                (long long) cfg.n_suffix, (long long) cfg.max_action_dim, cfg.num_steps,
                m->matmul_type == GGML_TYPE_F32 ? "F32" : "BF16");

#ifdef GGML_USE_CUDA
    if (!std::getenv("VLA_XR0_FORCE_CPU")) {
        m->backend = ggml_backend_cuda_init(0);
        if (m->backend) { m->is_cuda = true; std::printf("vla(xr0): backend = CUDA (device 0)\n"); }
        else            { std::fprintf(stderr, "vla(xr0): ggml_backend_cuda_init failed; falling back to CPU\n"); }
    } else {
        std::printf("vla(xr0): VLA_XR0_FORCE_CPU set, using CPU backend\n");
    }
#endif
    {
        const unsigned hw = std::thread::hardware_concurrency();
        m->n_threads = (hw == 0) ? 4 : (int) std::min(hw, 8u);
    }
    if (!m->backend) {
        m->backend = ggml_backend_cpu_init();
        if (!m->backend) { std::fprintf(stderr, "vla(xr0): ggml_backend_cpu_init failed\n"); return nullptr; }
        ggml_backend_cpu_set_n_threads(m->backend, m->n_threads);
        std::printf("vla(xr0): backend = CPU (%d threads)\n", m->n_threads);
    }

    {
        clip_context_params cp = {};
        // The CLIP vision tower defaults to CPU for numerical stability; opt
        // into GPU with VLA_XR0_CLIP_GPU=1 (large-VRAM cards).
        cp.use_gpu           = std::getenv("VLA_XR0_CLIP_GPU") != nullptr;
        cp.flash_attn_type   = CLIP_FLASH_ATTN_TYPE_DISABLED;  // no flash-attn either way
        cp.image_min_tokens  = -1;
        cp.image_max_tokens  = -1;
        cp.warmup            = false;  // CUDA warmup crashes in some llama.cpp builds; skip it
        cp.max_nodes         = 65536;  // deepstack qwen3vl graph exceeds the default 8192-node meta budget
        clip_init_result r = clip_init(mmproj_path.c_str(), cp);
        if (!r.ctx_v) {
            std::fprintf(stderr, "vla(xr0): clip_init failed for %s\n", mmproj_path.c_str());
            return nullptr;
        }
        m->cctx = r.ctx_v;
        // qwen3vl mmproj emits [main | ds0 | ds1 | ds2] per merged token
        if (clip_n_mmproj_embd(m->cctx) != (int) (4 * cfg.hidden)) {
            std::fprintf(stderr,
                "vla(xr0): mmproj embd=%d, want %lld (qwen3vl with deepstack)\n",
                clip_n_mmproj_embd(m->cctx), (long long) (4 * cfg.hidden));
            return nullptr;
        }
    }

    {
        ggml_init_params wp = { (size_t) 32 * 1024 * 1024, nullptr, true };
        m->ctx_weights = ggml_init(wp);
        if (!m->ctx_weights) { std::fprintf(stderr, "vla(xr0): ggml_init(ctx_weights) failed\n"); return nullptr; }
    }
    ggml_context * W = m->ctx_weights;
    std::vector<ggml_tensor *> weights;

    auto mk = [&](const char * name, ggml_type type) -> ggml_tensor * {
        const ggml_tensor * gt = g.meta(name);
        if (!gt) { std::fprintf(stderr, "vla(xr0): missing tensor %s\n", name); return nullptr; }
        // quantized on-disk tensors stay quantized in VRAM (ggml mul_mat dequantizes)
        if (ggml_is_quantized(gt->type)) type = gt->type;
        ggml_tensor * t = ggml_new_tensor(W, type, GGML_MAX_DIMS, gt->ne);
        ggml_set_name(t, name);
        weights.push_back(t);
        return t;
    };
    auto mk_mm  = [&](const char * name) { return mk(name, m->matmul_type); };
    auto mk_f32 = [&](const char * name) { return mk(name, GGML_TYPE_F32); };

    m->vlm_layers.resize(cfg.n_layers);
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        char b[160];
        auto suf = [&](const char * s) { std::snprintf(b, sizeof(b), "vlm.blk.%lld.%s", (long long) i, s); return b; };
        VlmLayerW & lw = m->vlm_layers[i];
        lw.ln_in  = mk_f32(suf("attn_norm.weight"));
        lw.Wq     = mk_mm (suf("attn_q.weight"));
        lw.Wk     = mk_mm (suf("attn_k.weight"));
        lw.Wv     = mk_mm (suf("attn_v.weight"));
        lw.Wo     = mk_mm (suf("attn_o.weight"));
        lw.q_norm = mk_f32(suf("attn_q_norm.weight"));
        lw.k_norm = mk_f32(suf("attn_k_norm.weight"));
        lw.ln_post= mk_f32(suf("ffn_norm.weight"));
        lw.Wgate  = mk_mm (suf("ffn_gate.weight"));
        lw.Wup    = mk_mm (suf("ffn_up.weight"));
        lw.Wdown  = mk_mm (suf("ffn_down.weight"));
        if (!lw.ln_in || !lw.Wq || !lw.Wk || !lw.Wv || !lw.Wo || !lw.q_norm || !lw.k_norm ||
            !lw.ln_post || !lw.Wgate || !lw.Wup || !lw.Wdown)
            return nullptr;
    }

    m->dit_layers.resize(m->dit_layers_n);
    for (int64_t i = 0; i < m->dit_layers_n; ++i) {
        char b[160];
        auto suf = [&](const char * s) { std::snprintf(b, sizeof(b), "dit.blk.%lld.%s", (long long) i, s); return b; };
        DitLayerW & lw = m->dit_layers[i];
        lw.ln_in   = mk_f32(suf("input_layernorm.weight"));
        lw.ln_mid  = mk_f32(suf("middle_layernorm.weight"));
        lw.ln_post = mk_f32(suf("post_layernorm.weight"));
        lw.ln_fin  = mk_f32(suf("final_layernorm.weight"));
        lw.adaln   = mk_f32(suf("adaln_table"));
        lw.Wqkv    = mk_mm (suf("attn_qkv.weight"));
        lw.bqkv    = mk_f32(suf("attn_qkv.bias"));
        lw.q_norm  = mk_f32(suf("attn_q_norm.weight"));
        lw.k_norm  = mk_f32(suf("attn_k_norm.weight"));
        lw.Wo      = mk_mm (suf("attn_o.weight"));
        lw.Wgate   = mk_mm (suf("ffn_gate.weight"));
        lw.Wup     = mk_mm (suf("ffn_up.weight"));
        lw.Wdown   = mk_mm (suf("ffn_down.weight"));
        if (!lw.ln_in || !lw.ln_mid || !lw.ln_post || !lw.ln_fin || !lw.adaln || !lw.Wqkv ||
            !lw.bqkv || !lw.q_norm || !lw.k_norm || !lw.Wo || !lw.Wgate || !lw.Wup || !lw.Wdown)
            return nullptr;
    }

    m->sink_w   = mk_f32("dit.sink.weight");
    m->W_state0 = mk_mm ("dit.state_proj.0.weight");
    m->W_state2 = mk_mm ("dit.state_proj.2.weight");
    m->W_ain0   = mk_mm ("dit.action_proj.0.weight");
    m->W_ain2   = mk_mm ("dit.action_proj.2.weight");
    m->W_aout0  = mk_mm ("dit.action_out.0.weight");
    m->W_aout2  = mk_mm ("dit.action_out.2.weight");
    m->W_temb0  = mk_mm ("dit.t_embedder.0.weight");
    m->W_temb2  = mk_mm ("dit.t_embedder.2.weight");
    m->W_tproj  = mk_mm ("dit.t_proj.weight");
    m->b_tproj  = mk_f32("dit.t_proj.bias");
    if (!m->sink_w || !m->W_state0 || !m->W_state2 || !m->W_ain0 || !m->W_ain2 ||
        !m->W_aout0 || !m->W_aout2 || !m->W_temb0 || !m->W_temb2 || !m->W_tproj || !m->b_tproj)
        return nullptr;

    // token embedding stays in the file; rows are fetched per predict (pi0.5
    // does the same, keeps ~1.5 GiB of VRAM free).
    if (!g.meta("token_embd.weight")) {
        std::fprintf(stderr, "vla(xr0): missing tensor token_embd.weight\n");
        return nullptr;
    }

    m->weight_buf = ggml_backend_alloc_ctx_tensors(m->ctx_weights, m->backend);
    if (!m->weight_buf) { std::fprintf(stderr, "vla(xr0): ggml_backend_alloc_ctx_tensors failed (out of memory?)\n"); return nullptr; }
    for (ggml_tensor * t : weights) {
        std::vector<uint8_t> bytes = g.read_convert(t->name, t->type);
        if (bytes.size() != ggml_nbytes(t)) {
            std::fprintf(stderr, "vla(xr0): upload size mismatch for %s (%zu vs %zu)\n",
                         t->name, bytes.size(), ggml_nbytes(t));
            return nullptr;
        }
        ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
    }
    std::printf("vla(xr0): resident weights = %.2f GiB\n",
                ggml_backend_buffer_get_size(m->weight_buf) / (1024.0 * 1024.0 * 1024.0));

    std::printf("vla(xr0): model loaded (n_threads=%d)\n", m->n_threads);
    return m;
}

std::vector<float> XiaomiRobotics0ModelArch::predict(const Inputs& in) {
    using clk = std::chrono::high_resolution_clock;
    const auto t0 = clk::now();
    stats = Stats{};

    const Config & cfg = this->cfg;
    const int64_t hidden    = cfg.hidden;
    const int64_t dit_h     = this->dit_h;
    const int64_t chunk     = cfg.n_suffix;
    const int64_t max_ad    = cfg.max_action_dim;
    const int64_t n_suf_q   = chunk + 2;                  // sink + state + actions
    const int64_t dit_hd    = cfg.head_dim;               // 128, shared with the backbone
    const int64_t dit_heads = dit_h / dit_hd;
    const int     num_steps = cfg.num_steps;
    const float   dt        = 1.0f / (float) num_steps;   // flow: t 0 -> 1
    const float   rope_base = cfg.rope_freq_base;

    // ---------------------------------------------------------------- vision
    if (in.n_images < 1 || !in.images) {
        std::fprintf(stderr, "vla(xr0): predict requires raw images\n");
        return {};
    }
    const int n_views = in.n_images;
    const int img_w = in.images[0].w, img_h = in.images[0].h;
    if (img_w % 32 != 0 || img_h % 32 != 0) {
        std::fprintf(stderr, "vla(xr0): image %dx%d is not a multiple of 32\n", img_w, img_h);
        return {};
    }
    for (int v = 1; v < n_views; ++v) {
        if (in.images[v].w != img_w || in.images[v].h != img_h) {
            std::fprintf(stderr, "vla(xr0): all views must share one resolution\n");
            return {};
        }
    }
    const int64_t grid      = img_w / 16 / 2;             // merged grid side
    const int64_t n_tok_view = grid * grid;
    // per-token output width: main + 3 deepstack slices
    const int64_t feat_w = 4 * hidden;

    const size_t per_pix  = (size_t) 3 * img_w * img_h;
    const size_t per_outf = (size_t) n_tok_view * (size_t) feat_w;
    std::vector<float> img_feats(per_outf * n_views);
    std::vector<float> hwc(per_pix);
    {
        const auto tv0 = clk::now();
        for (int v = 0; v < n_views; ++v) {
            const ImageView & view = in.images[v];
            if (view.format == PixelFormat::U8) {
                const uint8_t * src = static_cast<const uint8_t *>(view.data);
                for (size_t i = 0; i < per_pix; ++i) hwc[i] = (float) src[i] / 127.5f - 1.0f;
            } else {
                const float * src = static_cast<const float *>(view.data);
                for (size_t i = 0; i < per_pix; ++i) hwc[i] = src[i] * 2.0f - 1.0f;
            }
            if (!clip_encode_float_image(cctx, n_threads, hwc.data(), img_w, img_h,
                                         img_feats.data() + (size_t) v * per_outf)) {
                std::fprintf(stderr, "vla(xr0): clip_encode_float_image failed (view %d)\n", v);
                return {};
            }
        }
        stats.ms_vision = std::chrono::duration<float, std::milli>(clk::now() - tv0).count();
    }

    // ------------------------------------------------------- tokens & embeds
    if (in.n_lang < 1 || !in.lang_tokens) {
        std::fprintf(stderr, "vla(xr0): predict: empty lang_tokens\n");
        return {};
    }
    const int64_t n_seq = in.n_lang;

    std::vector<int32_t> toks(in.lang_tokens, in.lang_tokens + n_seq);
    {
        const int64_t n_img_total = n_tok_view * n_views;
        int64_t seen = 0;
        for (int32_t t : toks) if (t == image_token_id) ++seen;
        if (seen != n_img_total) {
            std::fprintf(stderr,
                "vla(xr0): image_pad tokens %lld != image tokens %lld (views=%d "
                "grid=%lld); the prompt must carry one <|image_pad|> per "
                "merged patch\n",
                (long long) seen, (long long) n_img_total, n_views, (long long) grid);
            return {};
        }
    }

    std::vector<float> seq_emb((size_t) n_seq * hidden);
    {
        std::vector<int32_t> text_ids;
        text_ids.reserve(n_seq);
        for (int32_t t : toks) if (t != image_token_id) text_ids.push_back(t);
        std::vector<float> text_rows((size_t) text_ids.size() * hidden);
        {
            gguf_reader g;
            if (!g.open(ckpt_path_)) return {};
            if (!g.fetch_rows_f32("token_embd.weight", text_ids, text_rows.data(), hidden))
                return {};
        }
        size_t tr = 0, ir = 0;
        for (int64_t i = 0; i < n_seq; ++i) {
            if (toks[i] == image_token_id) {
                std::memcpy(seq_emb.data() + (size_t) i * hidden,
                            img_feats.data() + ir * (size_t) feat_w,
                            (size_t) hidden * sizeof(float));
                ++ir;
            } else {
                std::memcpy(seq_emb.data() + (size_t) i * hidden,
                            text_rows.data() + tr * (size_t) hidden,
                            (size_t) hidden * sizeof(float));
                ++tr;
            }
        }
    }

    // deepstack features padded to the full sequence (zero on text slots)
    std::vector<std::vector<float>> ds_full(3, std::vector<float>((size_t) n_seq * hidden, 0.f));
    {
        size_t ir = 0;
        for (int64_t i = 0; i < n_seq; ++i) {
            if (toks[i] != image_token_id) continue;
            for (int k = 0; k < 3; ++k) {
                const float * src = img_feats.data() + ir * (size_t) feat_w
                                  + (size_t) (1 + k) * hidden;
                std::memcpy(ds_full[k].data() + (size_t) i * hidden, src,
                            (size_t) hidden * sizeof(float));
            }
            ++ir;
        }
    }

    // MRoPE position ids: 4 blocks of n_seq (t, h, w, e=t) per the ggml rope
    // op; image blocks get t=0, h=row, w=col over the merged grid (HF
    // get_rope_index with llm_grid_t = 1).
    std::vector<int32_t> pos4((size_t) 4 * n_seq);
    {
        int64_t counter = 0;
        int64_t i = 0;
        while (i < n_seq) {
            if (toks[i] == image_token_id) {
                const int64_t base = counter;
                for (int64_t r = 0; r < grid; ++r) {
                    for (int64_t c = 0; c < grid; ++c) {
                        const int64_t t_idx = i + r * grid + c;
                        pos4[              t_idx] = (int32_t) base;       // t
                        pos4[    n_seq + t_idx] = (int32_t) (base + r);  // h
                        pos4[2 * n_seq + t_idx] = (int32_t) (base + c);  // w
                    }
                }
                counter = base + grid;   // max(h) + 1 == max(w) + 1
                i += grid * grid;
            } else {
                const int64_t base = counter;
                int64_t j = i;
                while (j < n_seq && toks[j] != image_token_id) ++j;
                for (int64_t p = 0; p < j - i; ++p) {
                    pos4[              i + p] = (int32_t) (base + p);
                    pos4[    n_seq + i + p] = (int32_t) (base + p);
                    pos4[2 * n_seq + i + p] = (int32_t) (base + p);
                }
                counter = base + (j - i);
                i = j;
            }
        }
        // 4th row mirrors the t row so the IMROPE tail dims (61, 62) match
        // HF, where those dims keep the temporal position.
        std::memcpy(pos4.data() + 3 * n_seq, pos4.data(), (size_t) n_seq * sizeof(int32_t));
    }
    const int64_t vlm_max_pos = *std::max_element(pos4.begin(),
                                                   pos4.begin() + 3 * n_seq);

    // ------------------------------------------------------------- ggml graph
    ggml_init_params cp = { (size_t) 96 * 1024 * 1024, nullptr, true };
    ggml_context * C = ggml_init(cp);
    if (!C) { std::fprintf(stderr, "vla(xr0): ggml_init(ctx_compute) failed\n"); return {}; }

    ggml_tensor * t_emb      = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, n_seq);   ggml_set_input(t_emb);
    ggml_tensor * t_ds0      = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, n_seq);   ggml_set_input(t_ds0);
    ggml_tensor * t_ds1      = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, n_seq);   ggml_set_input(t_ds1);
    ggml_tensor * t_ds2      = ggml_new_tensor_2d(C, GGML_TYPE_F32, hidden, n_seq);   ggml_set_input(t_ds2);
    ggml_tensor * t_pos      = ggml_new_tensor_1d(C, GGML_TYPE_I32, 4 * n_seq);       ggml_set_input(t_pos);
    ggml_tensor * t_mask_vlm = ggml_new_tensor_2d(C, GGML_TYPE_F32, n_seq, n_seq);    ggml_set_input(t_mask_vlm);
    ggml_tensor * t_state    = ggml_new_tensor_1d(C, GGML_TYPE_F32, cfg.max_state_dim); ggml_set_input(t_state);
    ggml_tensor * t_x0       = ggml_new_tensor_2d(C, GGML_TYPE_F32, max_ad, chunk);   ggml_set_input(t_x0);
    ggml_tensor * t_amask    = ggml_new_tensor_1d(C, GGML_TYPE_F32, max_ad);          ggml_set_input(t_amask);
    ggml_tensor * t_one      = ggml_new_tensor_1d(C, GGML_TYPE_F32, 1);               ggml_set_input(t_one);
    ggml_tensor * t_dpos     = ggml_new_tensor_1d(C, GGML_TYPE_I32, 4 * n_suf_q);     ggml_set_input(t_dpos);
    ggml_tensor * t_mask_dit = ggml_new_tensor_2d(C, GGML_TYPE_F32, n_seq + n_suf_q, n_suf_q); ggml_set_input(t_mask_dit);
    std::vector<ggml_tensor *> t_time(num_steps);
    for (int s = 0; s < num_steps; ++s) {
        t_time[s] = ggml_new_tensor_1d(C, GGML_TYPE_F32, 256);   // freq embedding
        ggml_set_input(t_time[s]);
    }

    // ---------------------------------------------------------- backbone pass
    std::vector<ggml_tensor *> cK(cfg.n_layers), cV(cfg.n_layers);
    {
        ggml_tensor * h = t_emb;
        for (int64_t i = 0; i < cfg.n_layers; ++i) {
            h = build_vlm_layer(C, vlm_layers[i], h, t_pos, cfg, rope_sections, rope_base,
                                t_mask_vlm, &cK[i], &cV[i]);
            if      (i == 0) h = ggml_add(C, h, t_ds0);
            else if (i == 1) h = ggml_add(C, h, t_ds1);
            else if (i == 2) h = ggml_add(C, h, t_ds2);
        }
        (void) h; // last hidden state is unused by the action head
    }

    // ------------------------------------------------------------- DiT head
    // HF MLPProjector uses nn.GELU(approximate="tanh"), i.e. the tanh
    // approximation 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) -- NOT the
    // erf approximation that ggml_gelu() computes.  Build it from primitives
    // so the projector activations match the reference.
    auto gelu_tanh = [&](ggml_tensor * x) {
        ggml_tensor * x2    = ggml_sqr(C, x);
        ggml_tensor * x3    = ggml_mul(C, x2, x);
        ggml_tensor * inner = ggml_add(C, x, ggml_scale(C, x3, 0.044715f));
        ggml_tensor * th    = ggml_tanh(C, ggml_scale(C, inner, 0.7978845608028654f));  // sqrt(2/pi)
        return ggml_scale(C, ggml_mul(C, x, ggml_add(C, th, t_one)), 0.5f);
    };

    ggml_tensor * state_1  = ggml_reshape_2d(C, t_state, cfg.max_state_dim, 1);
    ggml_tensor * state_emb = ggml_mul_mat(C, W_state2,
            gelu_tanh(ggml_mul_mat(C, W_state0, state_1)));

    auto modulate = [&](ggml_tensor * x, ggml_tensor * sh, ggml_tensor * sc) {
        // x * (1 + scale) + shift; scale/shift are [dit_h, 1] broadcasts
        return ggml_add(C, ggml_mul(C, x, ggml_add(C, sc, t_one)), sh);
    };

    ggml_tensor * x_t = t_x0;
    for (int step = 0; step < num_steps; ++step) {
        // TimestepEmbedder: Linear(256->h) -> SiLU -> Linear(h->h); then
        // t_projector Linear(h -> 6h, bias) shared across layers' adaLN.
        ggml_tensor * t1 = ggml_silu(C, ggml_mul_mat(C, W_temb0, t_time[step]));
        ggml_tensor * t2 = ggml_mul_mat(C, W_temb2, t1);
        ggml_tensor * t6 = ggml_add(C, ggml_mul_mat(C, W_tproj, t2), b_tproj);
        t6 = ggml_reshape_2d(C, t6, dit_h, 6);

        ggml_tensor * noisy = ggml_mul(C, x_t, t_amask);
        ggml_tensor * ain0_mm = ggml_mul_mat(C, W_ain0, noisy);
        ggml_mul_mat_set_prec(ain0_mm, GGML_PREC_F32); // F32 accumulation
        ggml_tensor * ain2_mm = ggml_mul_mat(C, W_ain2, gelu_tanh(ain0_mm));
        ggml_mul_mat_set_prec(ain2_mm, GGML_PREC_F32); // F32 accumulation
        ggml_tensor * a_emb = ain2_mm;

        ggml_tensor * h = ggml_concat(C, sink_w,
                            ggml_concat(C, state_emb, a_emb, 1), 1);

        for (int64_t j = 0; j < dit_layers_n; ++j) {
            const DitLayerW & w = dit_layers[j];

            // adaLN rows of (adaln_table + t6): 6 x [dit_h, 1]
            ggml_tensor * mod = ggml_add(C, w.adaln, t6);
            const size_t rb = (size_t) dit_h * sizeof(float);
            ggml_tensor * sh_msa = ggml_view_2d(C, mod, dit_h, 1, rb, 0 * rb);
            ggml_tensor * sc_msa = ggml_view_2d(C, mod, dit_h, 1, rb, 1 * rb);
            ggml_tensor * ga_msa = ggml_view_2d(C, mod, dit_h, 1, rb, 2 * rb);
            ggml_tensor * sh_mlp = ggml_view_2d(C, mod, dit_h, 1, rb, 3 * rb);
            ggml_tensor * sc_mlp = ggml_view_2d(C, mod, dit_h, 1, rb, 4 * rb);
            ggml_tensor * ga_mlp = ggml_view_2d(C, mod, dit_h, 1, rb, 5 * rb);

            // ---- attention block
            ggml_tensor * xn = ggml_mul(C, ggml_rms_norm(C, h, cfg.rms_eps), w.ln_in);
            xn = modulate(xn, sh_msa, sc_msa);

            const int64_t hd  = dit_hd;
            const int64_t nq  = dit_heads;
            const int64_t seq = n_suf_q;
            ggml_tensor * qkv_mm = ggml_mul_mat(C, w.Wqkv, xn);
            ggml_mul_mat_set_prec(qkv_mm, GGML_PREC_F32);   // F32 accumulation
            ggml_tensor * qkv = ggml_add(C, qkv_mm, w.bqkv);
            // qkv is [3*dit_h, seq] with per-seq block ordered as HF's
            // (which, head, head_dim) row-major: q(0..dit_h), k, v each span
            // dit_h floats and are head-major (head stride = head_dim floats).
            // Head stride must be head_dim (not the whole dit_h row), else
            // heads bleed into the k/v blocks and attention diverges.
            const size_t head_b = (size_t) hd * sizeof(float);     // 128 floats
            const size_t row_b  = (size_t) dit_h * sizeof(float);  // 1024 floats (qkv block span)
            ggml_tensor * q = ggml_view_3d(C, qkv, hd, nq, seq, head_b, 3 * row_b, 0);
            ggml_tensor * k = ggml_view_3d(C, qkv, hd, nq, seq, head_b, 3 * row_b, row_b);
            ggml_tensor * v = ggml_view_3d(C, qkv, hd, nq, seq, head_b, 3 * row_b, 2 * row_b);

            q = ggml_mul(C, ggml_rms_norm(C, q, cfg.rms_eps), w.q_norm);
            k = ggml_mul(C, ggml_rms_norm(C, k, cfg.rms_eps), w.k_norm);

            auto rope_call = [&](ggml_tensor * t) {
                return ggml_rope_multi(C, t, t_dpos, nullptr,
                                       (int) hd, const_cast<int *>(rope_sections),
                                       GGML_ROPE_TYPE_IMROPE, 32768,
                                       rope_base, 1.f, 0.f, 1.f, 32.f, 1.f);
            };
            ggml_tensor * q_rope = rope_call(q);
            ggml_tensor * k_rope = rope_call(k);

            ggml_tensor * K_full = ggml_concat(C, cK[dit_kv_start + j], k_rope, 2);
            ggml_tensor * V_full = ggml_concat(C, cV[dit_kv_start + j], v,       2);

            ggml_tensor * Q = ggml_cont(C, ggml_permute(C, q_rope, 0, 2, 1, 3));
            ggml_tensor * K = ggml_cont(C, ggml_permute(C, K_full,  0, 2, 1, 3));
            ggml_tensor * V = ggml_cont(C, ggml_permute(C, V_full,  1, 2, 0, 3));

            ggml_tensor * kq = ggml_mul_mat(C, K, Q);
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            ggml_tensor * attn = ggml_soft_max_ext(C, kq, t_mask_dit,
                                                    1.f / std::sqrt((float) hd), 0.f);
            ggml_tensor * kqv = ggml_mul_mat(C, V, attn);
            ggml_tensor * att_pre = ggml_reshape_2d(C,
                ggml_cont(C, ggml_permute(C, kqv, 0, 2, 1, 3)), dit_h, seq);

            ggml_tensor * attn_out = ggml_mul_mat(C, w.Wo, att_pre);
            ggml_mul_mat_set_prec(attn_out, GGML_PREC_F32); // F32 accumulation
            ggml_tensor * h1 = ggml_add(C, h, ggml_mul(C, attn_out, ga_msa));
            h1 = ggml_mul(C, ggml_rms_norm(C, h1, cfg.rms_eps), w.ln_mid);

            // ---- MLP block
            ggml_tensor * xm = ggml_mul(C, ggml_rms_norm(C, h1, cfg.rms_eps), w.ln_post);
            xm = modulate(xm, sh_mlp, sc_mlp);
            ggml_tensor * gate = ggml_mul_mat(C, w.Wgate, xm);
            ggml_tensor * up   = ggml_mul_mat(C, w.Wup,   xm);
            ggml_tensor * mlp  = ggml_mul_mat(C, w.Wdown,
                                   ggml_mul(C, ggml_silu(C, gate), up));
            ggml_mul_mat_set_prec(gate, GGML_PREC_F32); // F32 accumulation
            ggml_mul_mat_set_prec(up,   GGML_PREC_F32); // F32 accumulation
            ggml_mul_mat_set_prec(mlp,  GGML_PREC_F32); // F32 accumulation
            ggml_tensor * h2 = ggml_add(C, h1, ggml_mul(C, mlp, ga_mlp));
            h = ggml_mul(C, ggml_rms_norm(C, h2, cfg.rms_eps), w.ln_fin);
        }

        // the trailing `chunk` rows are the action tokens
        ggml_tensor * rows = ggml_view_2d(C, h, dit_h, chunk,
                                          (size_t) dit_h * sizeof(float),
                                          (size_t) 2 * dit_h * sizeof(float));
        ggml_tensor * aout0_mm = ggml_mul_mat(C, W_aout0, rows);
        ggml_mul_mat_set_prec(aout0_mm, GGML_PREC_F32); // F32 accumulation
        ggml_tensor * aout2_mm = ggml_mul_mat(C, W_aout2, gelu_tanh(aout0_mm));
        ggml_mul_mat_set_prec(aout2_mm, GGML_PREC_F32); // F32 accumulation
        x_t = ggml_add(C, x_t, ggml_scale(C, aout2_mm, dt));
    }
    ggml_tensor * x_final = x_t;
    ggml_set_output(x_final);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 32768, false);
    ggml_build_forward_expand(gf, x_final);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!galloc || !ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "vla(xr0): ggml_gallocr_alloc_graph failed (out of memory?)\n");
        if (galloc) ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }

    // ------------------------------------------------------------- set inputs
    ggml_backend_tensor_set(t_emb,  seq_emb.data(),   0, ggml_nbytes(t_emb));
    ggml_backend_tensor_set(t_ds0,  ds_full[0].data(), 0, ggml_nbytes(t_ds0));
    ggml_backend_tensor_set(t_ds1,  ds_full[1].data(), 0, ggml_nbytes(t_ds1));
    ggml_backend_tensor_set(t_ds2,  ds_full[2].data(), 0, ggml_nbytes(t_ds2));
    ggml_backend_tensor_set(t_pos,  pos4.data(),      0, ggml_nbytes(t_pos));
    {
        const float one = 1.0f;
        ggml_backend_tensor_set(t_one, &one, 0, sizeof(one));
    }
    {
        // causal mask: 0 attends, -inf masked (ggml soft_max convention)
        std::vector<float> mk((size_t) n_seq * n_seq);
        for (int64_t q = 0; q < n_seq; ++q)
            for (int64_t k = 0; k < n_seq; ++k)
                mk[(size_t) q * n_seq + k] = (k <= q) ? 0.f : -INFINITY;
        ggml_backend_tensor_set(t_mask_vlm, mk.data(), 0, ggml_nbytes(t_mask_vlm));
    }
    {
        // DiT mask: full attention over the backbone prefix, causal over the
        // 32 query tokens
        std::vector<float> mk((size_t) (n_seq + n_suf_q) * n_suf_q, 0.f);
        for (int64_t q = 0; q < n_suf_q; ++q)
            for (int64_t k = n_seq; k < n_seq + n_suf_q; ++k)
                if (k - n_seq > q)
                    mk[(size_t) q * (n_seq + n_suf_q) + k] = -INFINITY;
        ggml_backend_tensor_set(t_mask_dit, mk.data(), 0, ggml_nbytes(t_mask_dit));
    }
    {
        std::vector<int32_t> dp((size_t) 4 * n_suf_q);
        for (int64_t i = 0; i < n_suf_q; ++i) {
            const int32_t p = (int32_t) (vlm_max_pos + 1 + i);
            dp[i] = dp[n_suf_q + i] = dp[2 * n_suf_q + i] = dp[3 * n_suf_q + i] = p;
        }
        ggml_backend_tensor_set(t_dpos, dp.data(), 0, ggml_nbytes(t_dpos));
    }
    {
        std::vector<float> st(cfg.max_state_dim, 0.f);
        if (in.state) std::memcpy(st.data(), in.state, (size_t) cfg.max_state_dim * sizeof(float));
        ggml_backend_tensor_set(t_state, st.data(), 0, ggml_nbytes(t_state));
        ggml_backend_tensor_set(t_amask, action_mask.data(), 0, ggml_nbytes(t_amask));
    }
    {
        std::vector<float> x0h((size_t) max_ad * chunk);
        if (in.noise) std::memcpy(x0h.data(), in.noise, x0h.size() * sizeof(float));
        else { std::normal_distribution<float> nd(0.f, 1.f); for (auto & v : x0h) v = nd(rng); }
        ggml_backend_tensor_set(t_x0, x0h.data(), 0, ggml_nbytes(t_x0));
    }
    for (int s = 0; s < num_steps; ++s) {
        const float timestep = (float) s * dt;                       // 0,.2,.4,.6,.8
        const std::vector<float> tv = timestep_freq_emb((double) timestep * 1000.0, 256, 10000.0);
        ggml_backend_tensor_set(t_time[s], tv.data(), 0, ggml_nbytes(t_time[s]));
    }

    const auto ti0 = clk::now();
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    stats.ms_inference = std::chrono::duration<float, std::milli>(clk::now() - ti0).count();
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "vla(xr0): ggml_backend_graph_compute failed (%d)\n", (int) st);
        ggml_gallocr_free(galloc);
        ggml_free(C);
        return {};
    }

    std::vector<float> out((size_t) chunk * max_ad);
    ggml_backend_tensor_get(x_final, out.data(), 0, out.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(C);

    // decode to world units: a = a_norm * std + mean (HF decode_action)
    for (int64_t t = 0; t < chunk; ++t) {
        float * row = out.data() + (size_t) t * max_ad;
        for (int64_t j = 0; j < max_ad; ++j) {
            row[j] = row[j] * action_std[j] + action_mean[j];
        }
    }

    stats.ms_total = std::chrono::duration<float, std::milli>(clk::now() - t0).count();
    return out;
}

} // namespace vla
