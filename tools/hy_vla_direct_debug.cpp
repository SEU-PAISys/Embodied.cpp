// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include "model.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

template <typename T>
bool read_raw(const std::string & path, std::vector<T> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        return false;
    }
    const std::streamsize bytes = f.tellg();
    if (bytes < 0 || bytes % (std::streamsize) sizeof(T) != 0) {
        std::fprintf(stderr, "bad raw file size for %s: %lld\n",
                     path.c_str(), (long long) bytes);
        return false;
    }
    f.seekg(0, std::ios::beg);
    out.resize((size_t) bytes / sizeof(T));
    if (!out.empty() && !f.read(reinterpret_cast<char *>(out.data()), bytes)) {
        std::fprintf(stderr, "failed to read %s\n", path.c_str());
        return false;
    }
    return true;
}

template <typename T>
bool write_raw(const std::string & path, const std::vector<T> & data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "failed to open %s for write\n", path.c_str());
        return false;
    }
    if (!data.empty()) {
        f.write(reinterpret_cast<const char *>(data.data()),
                (std::streamsize) (data.size() * sizeof(T)));
    }
    return (bool) f;
}

void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s --model MODEL.gguf --out-f32 OUT.f32 "
                 "[--prefix-f32 PREFIX.f32 --mask-i32 MASK.i32 | "
                 "--image-kind zero|gradient --height H --width W] "
                 "[--state-f32 STATE.f32] [--noise-f32 NOISE.f32]\n",
                 argv0);
}

bool parse_csv_i32(const std::string & s, std::vector<int32_t> & out) {
    out.clear();
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        char * end = nullptr;
        const long v = std::strtol(item.c_str(), &end, 10);
        if (!end || *end != '\0') return false;
        out.push_back((int32_t) v);
    }
    return !out.empty();
}

std::vector<uint8_t> make_image(const std::string & kind, int h, int w) {
    std::vector<uint8_t> img((size_t) h * (size_t) w * 3u, 0);
    if (kind == "zero") {
        return img;
    }
    if (kind == "gradient") {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t off = ((size_t) y * (size_t) w + (size_t) x) * 3u;
                img[off + 0] = (uint8_t) ((w <= 1) ? 0 : (x * 255 / (w - 1)));
                img[off + 1] = (uint8_t) ((h <= 1) ? 0 : (y * 255 / (h - 1)));
                img[off + 2] = 17;
            }
        }
        return img;
    }
    return {};
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prefix_path;
    std::string mask_path;
    std::string out_path;
    std::string image_kind;
    std::string state_path;
    std::string noise_path;
    std::vector<int32_t> lang_tokens = {120000, 120001, 120020, 7};
    int image_h = 224;
    int image_w = 224;

    for (int i = 1; i < argc; ++i) {
        auto require_value = [&](std::string & dst) -> bool {
            if (i + 1 >= argc) return false;
            dst = argv[++i];
            return true;
        };
        if (std::strcmp(argv[i], "--model") == 0) {
            if (!require_value(model_path)) { usage(argv[0]); return 2; }
        } else if (std::strcmp(argv[i], "--prefix-f32") == 0) {
            if (!require_value(prefix_path)) { usage(argv[0]); return 2; }
        } else if (std::strcmp(argv[i], "--mask-i32") == 0) {
            if (!require_value(mask_path)) { usage(argv[0]); return 2; }
        } else if (std::strcmp(argv[i], "--out-f32") == 0) {
            if (!require_value(out_path)) { usage(argv[0]); return 2; }
        } else if (std::strcmp(argv[i], "--state-f32") == 0) {
            if (!require_value(state_path)) { usage(argv[0]); return 2; }
        } else if (std::strcmp(argv[i], "--noise-f32") == 0) {
            if (!require_value(noise_path)) { usage(argv[0]); return 2; }
        } else if (std::strcmp(argv[i], "--image-kind") == 0) {
            if (!require_value(image_kind)) { usage(argv[0]); return 2; }
        } else if (std::strcmp(argv[i], "--height") == 0) {
            std::string v;
            if (!require_value(v)) { usage(argv[0]); return 2; }
            image_h = std::atoi(v.c_str());
        } else if (std::strcmp(argv[i], "--width") == 0) {
            std::string v;
            if (!require_value(v)) { usage(argv[0]); return 2; }
            image_w = std::atoi(v.c_str());
        } else if (std::strcmp(argv[i], "--lang-tokens") == 0) {
            std::string v;
            if (!require_value(v) || !parse_csv_i32(v, lang_tokens)) {
                std::fprintf(stderr, "bad --lang-tokens, expected comma-separated int32 ids\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    if (model_path.empty() || out_path.empty()) {
        usage(argv[0]);
        return 2;
    }
    const bool use_prefix = !prefix_path.empty() || !mask_path.empty();
    const bool use_image = !image_kind.empty();
    if (use_prefix == use_image) {
        std::fprintf(stderr, "choose exactly one input mode: prefix/mask or image-kind\n");
        usage(argv[0]);
        return 2;
    }
    if (use_prefix && (prefix_path.empty() || mask_path.empty())) {
        std::fprintf(stderr, "--prefix-f32 and --mask-i32 must be provided together\n");
        return 2;
    }

    vla::Model * model = vla::model_load("", model_path, "");
    if (!model) {
        std::fprintf(stderr, "failed to load model %s\n", model_path.c_str());
        return 1;
    }

    const vla::Config & cfg = vla::model_config(model);
    std::vector<float> prefix;
    std::vector<int32_t> mask;
    int n_pref = 0;
    if (use_prefix) {
        bool ok = read_raw(prefix_path, prefix) && read_raw(mask_path, mask);
        if (!ok) {
            vla::model_free(model);
            return 1;
        }
        if (cfg.hidden <= 0 || prefix.size() % (size_t) cfg.hidden != 0) {
            std::fprintf(stderr, "prefix size %zu is not divisible by hidden=%lld\n",
                         prefix.size(), (long long) cfg.hidden);
            vla::model_free(model);
            return 1;
        }
        n_pref = (int) (prefix.size() / (size_t) cfg.hidden);
        if ((int) mask.size() != n_pref) {
            std::fprintf(stderr, "mask length %zu != prefix tokens %d\n", mask.size(), n_pref);
            vla::model_free(model);
            return 1;
        }
    }

    std::vector<float> state((size_t) std::max<int64_t>(1, cfg.max_state_dim), 0.0f);
    std::vector<float> noise((size_t) std::max<int64_t>(1, cfg.max_action_dim * cfg.n_suffix), 0.0f);
    if (!state_path.empty()) {
        std::vector<float> tmp;
        if (!read_raw(state_path, tmp)) {
            vla::model_free(model);
            return 1;
        }
        if ((int64_t) tmp.size() != cfg.max_state_dim) {
            std::fprintf(stderr, "state length %zu != max_state_dim=%lld\n",
                         tmp.size(), (long long) cfg.max_state_dim);
            vla::model_free(model);
            return 1;
        }
        state = std::move(tmp);
    }
    if (!noise_path.empty()) {
        std::vector<float> tmp;
        if (!read_raw(noise_path, tmp)) {
            vla::model_free(model);
            return 1;
        }
        const int64_t expected = cfg.max_action_dim * cfg.n_suffix;
        if ((int64_t) tmp.size() != expected) {
            std::fprintf(stderr, "noise length %zu != n_suffix*max_action_dim=%lld\n",
                         tmp.size(), (long long) expected);
            vla::model_free(model);
            return 1;
        }
        noise = std::move(tmp);
    }
    std::vector<uint8_t> image_bytes;
    vla::ImageView image{};
    if (use_image) {
        if (image_h <= 0 || image_w <= 0) {
            std::fprintf(stderr, "invalid image size %dx%d\n", image_w, image_h);
            vla::model_free(model);
            return 2;
        }
        image_bytes = make_image(image_kind, image_h, image_w);
        if (image_bytes.empty()) {
            std::fprintf(stderr, "unknown --image-kind %s\n", image_kind.c_str());
            vla::model_free(model);
            return 2;
        }
        image.data = image_bytes.data();
        image.w = image_w;
        image.h = image_h;
        image.format = vla::PixelFormat::U8;
    }

    vla::Inputs in{};
    if (use_prefix) {
        in.precomputed_img_emb = prefix.data();
        in.n_img_views = n_pref;
        in.attention_mask = mask.data();
        in.attention_mask_n = n_pref;
    } else {
        in.images = &image;
        in.n_images = 1;
        in.lang_tokens = lang_tokens.data();
        in.n_lang = (int) lang_tokens.size();
    }
    in.state = state.data();
    in.noise = noise.data();

    std::vector<float> out = vla::predict(model, in);
    if (out.empty()) {
        std::fprintf(stderr, "predict returned an empty tensor\n");
        vla::model_free(model);
        return 1;
    }
    if (!write_raw(out_path, out)) {
        vla::model_free(model);
        return 1;
    }

    double sum = 0.0;
    float mn = out[0];
    float mx = out[0];
    for (float v : out) {
        sum += v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    std::printf("hy-vla-direct-debug: n=%zu sum=%.12f min=%.12f max=%.12f\n",
                out.size(), sum, mn, mx);
    std::printf("dumped_f32=%s\n", out_path.c_str());

    vla::model_free(model);
    return 0;
}
