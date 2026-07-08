// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "model.h"
#include "serving/lingbot.pb.h"

#include <zmq.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::atomic<bool> g_shutdown{false};

void on_signal(int) { g_shutdown.store(true, std::memory_order_relaxed); }

size_t dtype_size(lingbot::TensorDType dtype) {
    switch (dtype) {
        case lingbot::F32:  return 4;
        case lingbot::BF16: return 2;
        case lingbot::F16:  return 2;
        case lingbot::U8:   return 1;
        default:            return 0;
    }
}

bool tensor_element_count(const lingbot::Tensor & t, uint64_t & out) {
    out = 1;
    if (t.shape_size() == 0) return false;
    for (uint64_t d : t.shape()) {
        if (d == 0) return false;
        if (out > UINT64_MAX / d) return false;
        out *= d;
    }
    return true;
}

bool validate_tensor_bytes(const lingbot::Tensor & t, std::string & err) {
    const size_t elem_size = dtype_size(t.dtype());
    if (elem_size == 0) {
        err = "unsupported tensor dtype";
        return false;
    }
    uint64_t n = 0;
    if (!tensor_element_count(t, n)) {
        err = "invalid tensor shape";
        return false;
    }
    if (n > UINT64_MAX / elem_size) {
        err = "tensor byte size overflow";
        return false;
    }
    const uint64_t expected = n * elem_size;
    if (t.data().size() != expected) {
        char buf[192];
        std::snprintf(buf, sizeof(buf), "tensor '%s' byte size %zu != expected %llu",
                      t.name().c_str(), t.data().size(), (unsigned long long) expected);
        err = buf;
        return false;
    }
    return true;
}

std::string make_error_response(uint64_t request_id, const std::string & msg) {
    lingbot::LingBotResponse resp;
    resp.set_request_id(request_id);
    resp.set_error(msg);
    resp.set_status("error");
    return resp.SerializeAsString();
}

bool tensor_f32_to_vector(const lingbot::Tensor & t, std::vector<float> & out, std::string & err) {
    if (t.dtype() != lingbot::F32) {
        err = "only F32 tensors are accepted for this bridge path";
        return false;
    }
    if (!validate_tensor_bytes(t, err)) return false;
    const size_t n = t.data().size() / sizeof(float);
    out.resize(n);
    std::memcpy(out.data(), t.data().data(), t.data().size());
    return true;
}

const lingbot::Tensor * find_lingbot_latent_tensor(const lingbot::StepRequest & step) {
    const lingbot::Tensor * first_5d_f32 = nullptr;
    for (const auto & t : step.input_latents()) {
        if (t.dtype() != lingbot::F32 || t.shape_size() != 5) continue;
        if (!first_5d_f32) first_5d_f32 = &t;
        const std::string & name = t.name();
        if (name == "video_latent" || name == "world_latent" || name == "latent" ||
            name == "vae_latent") {
            return &t;
        }
    }
    return first_5d_f32;
}

bool append_image_tensor_as_view(const lingbot::Tensor & t,
                                 std::vector<float> & video_vcfhw,
                                 uint64_t & frames,
                                 uint64_t & height,
                                 uint64_t & width,
                                 std::string & err) {
    if (t.dtype() != lingbot::U8 && t.dtype() != lingbot::F32) {
        err = "input_images accepts only U8 or F32 tensors";
        return false;
    }
    if (!validate_tensor_bytes(t, err)) return false;
    if (t.shape_size() != 4) {
        err = "input image tensor must have shape [F,H,W,3] or [3,F,H,W]";
        return false;
    }

    const bool nhwc = t.shape(3) == 3;
    const bool cfhw = t.shape(0) == 3;
    if (!nhwc && !cfhw) {
        err = "input image tensor must have RGB channel dimension of size 3";
        return false;
    }

    const uint64_t f = nhwc ? t.shape(0) : t.shape(1);
    const uint64_t h = nhwc ? t.shape(1) : t.shape(2);
    const uint64_t w = nhwc ? t.shape(2) : t.shape(3);
    if (f == 0 || h == 0 || w == 0) {
        err = "input image tensor has an empty dimension";
        return false;
    }
    if (frames == 0) {
        frames = f;
        height = h;
        width = w;
    } else if (frames != f || height != h || width != w) {
        err = "all input image views must share F,H,W";
        return false;
    }

    const size_t view_elems = (size_t) 3 * (size_t) f * (size_t) h * (size_t) w;
    const size_t base = video_vcfhw.size();
    video_vcfhw.resize(base + view_elems);
    auto dst_idx = [&](uint64_t c, uint64_t tf, uint64_t ih, uint64_t iw) {
        return base + (size_t) c * (size_t) f * (size_t) h * (size_t) w +
               (size_t) tf * (size_t) h * (size_t) w +
               (size_t) ih * (size_t) w + (size_t) iw;
    };

    if (t.dtype() == lingbot::U8) {
        const uint8_t * p = reinterpret_cast<const uint8_t *>(t.data().data());
        for (uint64_t tf = 0; tf < f; ++tf) {
            for (uint64_t ih = 0; ih < h; ++ih) {
                for (uint64_t iw = 0; iw < w; ++iw) {
                    for (uint64_t c = 0; c < 3; ++c) {
                        const size_t src = nhwc
                            ? ((size_t) tf * (size_t) h * (size_t) w * 3 + (size_t) ih * (size_t) w * 3 + (size_t) iw * 3 + (size_t) c)
                            : ((size_t) c * (size_t) f * (size_t) h * (size_t) w + (size_t) tf * (size_t) h * (size_t) w + (size_t) ih * (size_t) w + (size_t) iw);
                        video_vcfhw[dst_idx(c, tf, ih, iw)] = (float) p[src] / 255.0f * 2.0f - 1.0f;
                    }
                }
            }
        }
    } else {
        const float * p = reinterpret_cast<const float *>(t.data().data());
        uint64_t n = 0;
        if (!tensor_element_count(t, n)) {
            err = "invalid input image tensor shape";
            return false;
        }
        float mn = p[0], mx = p[0];
        for (uint64_t i = 1; i < n; ++i) {
            mn = std::min(mn, p[i]);
            mx = std::max(mx, p[i]);
        }
        const bool looks_u8 = mx > 2.0f;
        const bool looks_01 = !looks_u8 && mn >= 0.0f && mx <= 1.0f;
        for (uint64_t tf = 0; tf < f; ++tf) {
            for (uint64_t ih = 0; ih < h; ++ih) {
                for (uint64_t iw = 0; iw < w; ++iw) {
                    for (uint64_t c = 0; c < 3; ++c) {
                        const size_t src = nhwc
                            ? ((size_t) tf * (size_t) h * (size_t) w * 3 + (size_t) ih * (size_t) w * 3 + (size_t) iw * 3 + (size_t) c)
                            : ((size_t) c * (size_t) f * (size_t) h * (size_t) w + (size_t) tf * (size_t) h * (size_t) w + (size_t) ih * (size_t) w + (size_t) iw);
                        float v = p[src];
                        if (looks_u8) v = v / 255.0f * 2.0f - 1.0f;
                        else if (looks_01) v = v * 2.0f - 1.0f;
                        video_vcfhw[dst_idx(c, tf, ih, iw)] = v;
                    }
                }
            }
        }
    }
    return true;
}

struct SessionCache {
    uint64_t runtime_id = 0;
    uint64_t steps = 0;
    uint64_t predict_count = 0;
    uint64_t cache_update_count = 0;
    bool has_history_cache = false;
    uint64_t last_latent_bytes = 0;
    int last_latent_count = 0;
    uint64_t last_image_bytes = 0;
    int last_image_count = 0;

    void record_input_metadata(const lingbot::StepRequest & step) {
        last_latent_bytes = 0;
        last_latent_count = step.input_latents_size();
        for (const auto & t : step.input_latents()) {
            last_latent_bytes += (uint64_t) t.data().size();
        }
        last_image_bytes = 0;
        last_image_count = step.input_images_size();
        for (const auto & t : step.input_images()) {
            last_image_bytes += (uint64_t) t.data().size();
        }
    }

    void mark_predict_success() {
        ++steps;
        ++predict_count;
    }

    void mark_cache_update_success() {
        ++steps;
        ++cache_update_count;
        has_history_cache = true;
    }
};

void usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s [--bind ADDR] <lingbot-transformer.gguf>\n"
        "  --bind ADDR   ZMQ bind address (default: tcp://*:5557)\n",
        prog);
}

}

int main(int argc, char ** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string bind_addr = "tcp://*:5557";
    std::vector<std::string> positionals;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bind" && i + 1 < argc) {
            bind_addr = argv[++i];
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            positionals.push_back(std::move(a));
        }
    }
    if (positionals.size() != 1) {
        usage(argv[0]);
        return 1;
    }

    const std::string ckpt_path = positionals[0];
    std::printf("lingbot-world-server: loading model ...\n  ckpt: %s\n", ckpt_path.c_str());
    vla::Model * model = vla::model_load("", ckpt_path);
    if (!model) {
        std::fprintf(stderr, "lingbot-world-server: model_load failed\n");
        return 1;
    }
    const vla::Config & cfg = vla::model_config(model);
    std::printf("lingbot-world-server: loaded. chunk_size=%lld max_action_dim=%lld real_action_dim=%lld state_dim=%lld\n",
                (long long) cfg.n_suffix,
                (long long) cfg.max_action_dim,
                (long long) cfg.real_action_dim,
                (long long) cfg.max_state_dim);

    zmq::context_t zctx(1);
    zmq::socket_t sock(zctx, zmq::socket_type::rep);
    int linger = 0;
    sock.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
    sock.bind(bind_addr);
    std::printf("lingbot-world-server: bound to %s. ready.\n", bind_addr.c_str());

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    zmq::pollitem_t poll[] = {{ static_cast<void*>(sock), 0, ZMQ_POLLIN, 0 }};
    uint64_t served = 0;
    std::unordered_map<uint64_t, SessionCache> sessions;
    std::unordered_map<uint64_t, uint64_t> session_generations;
    uint64_t next_session_generation = 1;

    while (!g_shutdown.load(std::memory_order_relaxed)) {
        try {
            zmq::poll(poll, 1, std::chrono::milliseconds(200));
        } catch (const zmq::error_t & e) {
            if (e.num() == EINTR) continue;
            throw;
        }
        if (!(poll[0].revents & ZMQ_POLLIN)) continue;

        zmq::message_t req_msg;
        try {
            auto rr = sock.recv(req_msg, zmq::recv_flags::none);
            if (!rr) continue;
        } catch (const zmq::error_t & e) {
            if (e.num() == EINTR) continue;
            throw;
        }

        lingbot::LingBotRequest req;
        if (!req.ParseFromArray(req_msg.data(), static_cast<int>(req_msg.size()))) {
            sock.send(zmq::buffer(make_error_response(0, "LingBotRequest parse failed")),
                      zmq::send_flags::none);
            continue;
        }
        const uint64_t rid = req.request_id();

        if (req.has_ping()) {
            lingbot::LingBotResponse resp;
            resp.set_request_id(rid);
            resp.set_status(req.ping().text().empty() ? "pong" : req.ping().text());
            const std::string body = resp.SerializeAsString();
            sock.send(zmq::buffer(body), zmq::send_flags::none);
            continue;
        }

        if (req.has_reset()) {
            session_generations[req.reset().session_id()] = next_session_generation++;
            if (req.reset().clear_cache()) {
                sessions.erase(req.reset().session_id());
            } else {
                SessionCache fresh{};
                fresh.runtime_id = session_generations[req.reset().session_id()];
                sessions[req.reset().session_id()] = fresh;
            }
            lingbot::LingBotResponse resp;
            resp.set_request_id(rid);
            resp.set_status(req.reset().clear_cache() ? "reset_clear_cache" : "reset");
            const std::string body = resp.SerializeAsString();
            sock.send(zmq::buffer(body), zmq::send_flags::none);
            continue;
        }

        if (!req.has_step()) {
            sock.send(zmq::buffer(make_error_response(rid, "request must contain ping, reset, or step")),
                      zmq::send_flags::none);
            continue;
        }

        const lingbot::StepRequest & step = req.step();
        std::string err;
        for (const auto & t : step.input_latents()) {
            if (!validate_tensor_bytes(t, err)) {
                sock.send(zmq::buffer(make_error_response(rid, err)), zmq::send_flags::none);
                err.clear();
                goto next_request;
            }
        }
        for (const auto & t : step.input_images()) {
            if (!validate_tensor_bytes(t, err)) {
                sock.send(zmq::buffer(make_error_response(rid, err)), zmq::send_flags::none);
                err.clear();
                goto next_request;
            }
        }
        {
            SessionCache & session = sessions[step.session_id()];
            if (session.runtime_id == 0) {
                auto it_gen = session_generations.find(step.session_id());
                if (it_gen == session_generations.end()) {
                    it_gen = session_generations.emplace(step.session_id(), next_session_generation++).first;
                }
                session.runtime_id = it_gen->second;
            }
            session.record_input_metadata(step);
            const bool is_cache_update = step.compute_kv_cache();
            const bool is_first_predict_chunk =
                !is_cache_update && session.predict_count == 0 && !session.has_history_cache;

            if (step.state_size() != 0 && step.state_size() != (int) cfg.max_state_dim) {
                char buf[128];
                std::snprintf(buf, sizeof(buf), "state length %d != 0 or %lld",
                              step.state_size(), (long long) cfg.max_state_dim);
                sock.send(zmq::buffer(make_error_response(rid, buf)), zmq::send_flags::none);
                goto next_request;
            }
            if (step.lang_tokens_size() > (int) cfg.n_lang) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "lang_tokens length %d exceeds %lld",
                              step.lang_tokens_size(), (long long) cfg.n_lang);
                sock.send(zmq::buffer(make_error_response(rid, buf)), zmq::send_flags::none);
                goto next_request;
            }
            for (int i = 0; i < step.lang_tokens_size(); ++i) {
                if (step.lang_tokens(i) < 0) {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), "lang_tokens[%d] is negative: %d",
                                  i, step.lang_tokens(i));
                    sock.send(zmq::buffer(make_error_response(rid, buf)), zmq::send_flags::none);
                    goto next_request;
                }
            }

            std::vector<float> noise;
            if (step.has_action_noise() && step.action_noise().data().size() > 0) {
                if (!tensor_f32_to_vector(step.action_noise(), noise, err)) {
                    sock.send(zmq::buffer(make_error_response(rid, err)), zmq::send_flags::none);
                    goto next_request;
                }
                const size_t expected = (size_t) cfg.n_suffix * (size_t) cfg.max_action_dim;
                if (noise.size() != expected) {
                    char buf[128];
                    std::snprintf(buf, sizeof(buf), "action_noise elements %zu != expected %zu",
                                  noise.size(), expected);
                    sock.send(zmq::buffer(make_error_response(rid, buf)), zmq::send_flags::none);
                    goto next_request;
                }
            }
            std::vector<float> action_condition;
            uint64_t action_condition_shape[3] = {0, 0, 0};
            if (step.has_action_condition() && step.action_condition().data().size() > 0) {
                const lingbot::Tensor & t = step.action_condition();
                if (t.dtype() != lingbot::F32 || t.shape_size() != 3) {
                    sock.send(zmq::buffer(make_error_response(rid, "action_condition must be F32 with shape [C,F,H]")),
                              zmq::send_flags::none);
                    goto next_request;
                }
                if (!tensor_f32_to_vector(t, action_condition, err)) {
                    sock.send(zmq::buffer(make_error_response(rid, err)), zmq::send_flags::none);
                    goto next_request;
                }
                for (int i = 0; i < 3; ++i) action_condition_shape[i] = t.shape(i);
                if (action_condition_shape[0] != (uint64_t) cfg.real_action_dim) {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), "action_condition C=%llu != real_action_dim=%lld",
                                  (unsigned long long) action_condition_shape[0],
                                  (long long) cfg.real_action_dim);
                    sock.send(zmq::buffer(make_error_response(rid, buf)), zmq::send_flags::none);
                    goto next_request;
                }
            }

            std::vector<float> state((size_t) cfg.max_state_dim, 0.0f);
            if (step.state_size() == (int) cfg.max_state_dim) {
                std::copy(step.state().begin(), step.state().end(), state.begin());
            }
            std::vector<int32_t> lang_tokens(step.lang_tokens().begin(), step.lang_tokens().end());
            std::vector<float> lingbot_latent;
            uint64_t latent_shape[5] = {0, 0, 0, 0, 0};
            const lingbot::Tensor * latent_tensor = find_lingbot_latent_tensor(step);
            if (latent_tensor) {
                if (!tensor_f32_to_vector(*latent_tensor, lingbot_latent, err)) {
                    sock.send(zmq::buffer(make_error_response(rid, err)), zmq::send_flags::none);
                    goto next_request;
                }
                for (int i = 0; i < 5; ++i) latent_shape[i] = latent_tensor->shape(i);
            }
            std::vector<float> lingbot_video;
            uint64_t video_frames = 0;
            uint64_t video_height = 0;
            uint64_t video_width = 0;
            if (!latent_tensor && step.input_images_size() > 0) {
                for (const auto & t : step.input_images()) {
                    if (!append_image_tensor_as_view(t, lingbot_video, video_frames,
                                                     video_height, video_width, err)) {
                        sock.send(zmq::buffer(make_error_response(rid, err)), zmq::send_flags::none);
                        goto next_request;
                    }
                }
            }

            vla::Inputs in{};
            in.lingbot_session_id = session.runtime_id;
            in.lingbot_cache_mode = is_cache_update ? 2 : 0;
            in.lingbot_clear_pred_cache = is_cache_update;
            in.lingbot_first_chunk = is_first_predict_chunk;
            in.lingbot_has_history_cache = session.has_history_cache;
            in.lingbot_predict_index = session.predict_count;
            in.lingbot_cache_update_index = session.cache_update_count;
            in.state = state.data();
            in.noise = noise.empty() ? nullptr : noise.data();
            if (!action_condition.empty()) {
                in.lingbot_action_condition = action_condition.data();
                in.lingbot_action_condition_c = (int64_t) action_condition_shape[0];
                in.lingbot_action_condition_f = (int64_t) action_condition_shape[1];
                in.lingbot_action_condition_h = (int64_t) action_condition_shape[2];
            }
            in.lang_tokens = lang_tokens.empty() ? nullptr : lang_tokens.data();
            in.n_lang = static_cast<int>(lang_tokens.size());
            if (!lingbot_latent.empty()) {
                in.lingbot_latent = lingbot_latent.data();
                in.lingbot_latent_b = (int64_t) latent_shape[0];
                in.lingbot_latent_c = (int64_t) latent_shape[1];
                in.lingbot_latent_f = (int64_t) latent_shape[2];
                in.lingbot_latent_h = (int64_t) latent_shape[3];
                in.lingbot_latent_w = (int64_t) latent_shape[4];
            } else if (!lingbot_video.empty()) {
                in.lingbot_video = lingbot_video.data();
                in.lingbot_video_views = step.input_images_size();
                in.lingbot_video_c = 3;
                in.lingbot_video_f = (int64_t) video_frames;
                in.lingbot_video_h = (int64_t) video_height;
                in.lingbot_video_w = (int64_t) video_width;
            }

            const auto t0 = std::chrono::steady_clock::now();
            std::vector<float> actions = vla::predict(model, in);
            const auto t1 = std::chrono::steady_clock::now();
            if (actions.empty()) {
                sock.send(zmq::buffer(make_error_response(rid, "predict failed")),
                          zmq::send_flags::none);
                goto next_request;
            }

            const vla::Stats & st = vla::last_stats(model);
            lingbot::LingBotResponse resp;
            resp.set_request_id(rid);
            if (is_cache_update) {
                session.mark_cache_update_success();
                char status[384];
                std::snprintf(status, sizeof(status),
                              "kv_cache_updated session=%llu step=%llu cache_updates=%llu predicts=%llu image_count=%d image_bytes=%llu action_cond=%s imagine=%s",
                              (unsigned long long) step.session_id(),
                              (unsigned long long) session.steps,
                              (unsigned long long) session.cache_update_count,
                              (unsigned long long) session.predict_count,
                              session.last_image_count,
                              (unsigned long long) session.last_image_bytes,
                              action_condition.empty() ? "none" : "c_f_h",
                              step.imagine() ? "true" : "false");
                resp.set_status(status);
                resp.set_latency_ms_inference(st.ms_inference);
                resp.set_latency_ms_total(
                    std::chrono::duration<float, std::milli>(t1 - t0).count());
                const std::string body = resp.SerializeAsString();
                sock.send(zmq::buffer(body), zmq::send_flags::none);
                ++served;
                if (served % 10 == 1) {
                    std::printf("lingbot-world-server: rid=%llu served=%llu cache_update total=%.1fms input_images=%d\n",
                                (unsigned long long) rid,
                                (unsigned long long) served,
                                resp.latency_ms_total(),
                                step.input_images_size());
                }
                goto next_request;
            }
            session.mark_predict_success();
            uint32_t response_action_dim = static_cast<uint32_t>(cfg.max_action_dim);
            if (cfg.n_suffix > 0 && actions.size() % (size_t) cfg.n_suffix == 0) {
                response_action_dim = static_cast<uint32_t>(actions.size() / (size_t) cfg.n_suffix);
            }

            char status[512];
            std::snprintf(status, sizeof(status), "%s session=%llu step=%llu predicts=%llu cache_updates=%llu first_chunk=%s history_cache=%s latent_count=%d latent_bytes=%llu image_count=%d image_bytes=%llu lang_tokens=%d action_cond=%s used_latent=%s action_dim=%u",
                          latent_tensor ? "step_with_latents_bridge" :
                          (!lingbot_video.empty() ? "step_with_images_vae_bridge" : "step_bridge"),
                          (unsigned long long) step.session_id(),
                          (unsigned long long) session.steps,
                          (unsigned long long) session.predict_count,
                          (unsigned long long) session.cache_update_count,
                          is_first_predict_chunk ? "true" : "false",
                          session.has_history_cache ? "true" : "false",
                          session.last_latent_count,
                          (unsigned long long) session.last_latent_bytes,
                          session.last_image_count,
                          (unsigned long long) session.last_image_bytes,
                          step.lang_tokens_size(),
                          action_condition.empty() ? "none" : "c_f_h",
                          latent_tensor ? latent_tensor->name().c_str() : "none",
                          response_action_dim);
            resp.set_status(status);
            resp.mutable_action_chunk()->Reserve(static_cast<int>(actions.size()));
            for (float v : actions) resp.add_action_chunk(v);
            resp.set_chunk_size(static_cast<uint32_t>(cfg.n_suffix));
            resp.set_action_dim(response_action_dim);
            if (step.return_world_latents()) {
                for (const auto & t : step.input_latents()) {
                    *resp.add_output_latents() = t;
                }
            }
            resp.set_latency_ms_inference(st.ms_inference);
            resp.set_latency_ms_total(
                std::chrono::duration<float, std::milli>(t1 - t0).count());
            const std::string body = resp.SerializeAsString();
            sock.send(zmq::buffer(body), zmq::send_flags::none);
            ++served;
            if (served % 10 == 1) {
                std::printf("lingbot-world-server: rid=%llu served=%llu total=%.1fms input_latents=%d\n",
                            (unsigned long long) rid,
                            (unsigned long long) served,
                            resp.latency_ms_total(),
                            step.input_latents_size());
            }
        }

next_request:
        continue;
    }

    std::printf("lingbot-world-server: shutting down (served %llu requests)\n",
                (unsigned long long) served);
    vla::model_free(model);
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
