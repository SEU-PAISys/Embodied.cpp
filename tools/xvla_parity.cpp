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

// X-VLA numerical-parity harness: replays the fixed inputs produced by
// scripts/parity_xvla_reference.py through the C++ runtime and writes the
// resulting action chunk to a flat binary file for the Python comparator.
//
//   xvla-parity <ckpt.gguf> <parity_dir>
//
// reads  : <parity_dir>/xvla_parity_inputs.bin
// writes : <parity_dir>/xvla_parity_out.bin       (float32 [30, 20])

#include "model.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "xvla-parity: cannot open %s\n", path.c_str()); return {}; }
    f.seekg(0, std::ios::end);
    const size_t n = (size_t) f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(n);
    f.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) n);
    return buf;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <xvla.gguf> <parity_dir>\n", argv[0]);
        return 1;
    }
    const std::string ckpt = argv[1];
    const std::string dir  = argv[2];

    auto input_bytes = read_file(dir + "/xvla_parity_inputs.bin");
    if (input_bytes.empty()) return 1;
    const uint8_t * p = input_bytes.data();
    size_t left = input_bytes.size();

    auto take = [&](void * dst, size_t n) -> bool {
        if (left < n) return false;
        std::memcpy(dst, p, n);
        p += n; left -= n;
        return true;
    };

    constexpr int64_t kImg = 224;
    constexpr int64_t kActions = 30;
    constexpr int64_t kDim = 20;

    int32_t n_views = 0, domain_id = 0, n_lang = 0;
    if (!take(&n_views, 4) || n_views < 1 || n_views > 3 ||
        !take(&domain_id, 4) ||
        !take(&n_lang, 4) || n_lang < 1) {
        std::fprintf(stderr, "bad inputs header\n"); return 1;
    }
    std::vector<int32_t> lang((size_t) n_lang);
    if (!take(lang.data(), (size_t) n_lang * 4)) { std::fprintf(stderr, "bad tokens\n"); return 1; }

    const size_t per_view = (size_t) 3 * kImg * kImg;
    std::vector<uint8_t> pixels(per_view * (size_t) n_views);
    if (!take(pixels.data(), pixels.size())) { std::fprintf(stderr, "bad pixels\n"); return 1; }

    std::vector<float> state((size_t) kDim);
    std::vector<float> noise((size_t) (kActions * kDim));
    if (!take(state.data(), state.size() * 4) || !take(noise.data(), noise.size() * 4)) {
        std::fprintf(stderr, "bad state/noise\n"); return 1;
    }

    vla::Model * model = vla::model_load("", ckpt, "");
    if (!model) { std::fprintf(stderr, "xvla-parity: model_load failed\n"); return 1; }

    std::vector<vla::ImageView> image_views((size_t) n_views);
    for (int v = 0; v < n_views; ++v) {
        image_views[(size_t) v].data   = pixels.data() + (size_t) v * per_view;
        image_views[(size_t) v].w      = (int) kImg;
        image_views[(size_t) v].h      = (int) kImg;
        image_views[(size_t) v].format = vla::PixelFormat::U8;
    }

    vla::Inputs in{};
    in.images     = image_views.data();
    in.n_images   = n_views;
    in.lang_tokens = lang.data();
    in.n_lang      = n_lang;
    in.state       = state.data();
    in.noise       = noise.data();
    in.domain_id   = domain_id;

    std::vector<float> actions = vla::predict(model, in);
    const auto & cfg = vla::model_config(model);
    const auto & st  = vla::last_stats(model);
    std::printf("xvla-parity: chunk=%lld dim=%lld total=%.1f ms inference=%.1f ms\n",
                (long long) cfg.n_suffix, (long long) cfg.real_action_dim,
                st.ms_total, st.ms_inference);
    if (actions.size() != (size_t) (kActions * kDim)) {
        std::fprintf(stderr, "bad action size %zu\n", actions.size());
        return 1;
    }

    const std::string out_path = dir + "/xvla_parity_out.bin";
    FILE * fo = std::fopen(out_path.c_str(), "wb");
    if (!fo || std::fwrite(actions.data(), 4, actions.size(), fo) != actions.size()) {
        std::fprintf(stderr, "xvla-parity: cannot write %s\n", out_path.c_str());
        if (fo) std::fclose(fo);
        return 1;
    }
    std::fclose(fo);
    std::printf("xvla-parity: wrote %s (%zu floats)\n", out_path.c_str(), actions.size());
    vla::model_free(model);
    return 0;
}
