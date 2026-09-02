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

// Xiaomi-Robotics-0 numerical-parity harness: replays the fixed inputs produced by
// scripts/parity_xr0_reference.py through the C++ runtime and writes the
// resulting action chunk (and intermediate vision features when asked) to
// flat binary files for the Python comparator.
//
//   xr0-parity <mmproj.gguf> <ckpt.gguf> <parity_dir> [--dump-vision]
//
// reads  : <parity_dir>/xr0_parity_inputs.bin
// writes : <parity_dir>/xr0_parity_out.bin        (float32 [30, 32])
//          <parity_dir>/xr0_out_vision.bin        (optional, clip layout)

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
    if (!f) { std::fprintf(stderr, "xr0-parity: cannot open %s\n", path.c_str()); return {}; }
    f.seekg(0, std::ios::end);
    const size_t n = (size_t) f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(n);
    f.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) n);
    return buf;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <xr0-mmproj.gguf> <xr0.gguf> <parity_dir> [--dump-vision]\n",
            argv[0]);
        return 1;
    }
    const std::string mmproj = argv[1];
    const std::string ckpt   = argv[2];
    const std::string dir    = argv[3];
    const bool dump_vision   = argc > 4 && std::strcmp(argv[4], "--dump-vision") == 0;

    const char * img_env = std::getenv("XR0_PARITY_IMG");
    const int img = img_env ? std::atoi(img_env) : 256;
    const std::string sfx = (img == 256) ? "" : ("_" + std::to_string(img));

    auto input_bytes = read_file(dir + "/xr0_parity_inputs" + sfx + ".bin");
    if (input_bytes.empty()) return 1;
    const uint8_t * p = input_bytes.data();
    size_t left = input_bytes.size();

    auto take = [&](void * dst, size_t n) -> bool {
        if (left < n) return false;
        std::memcpy(dst, p, n);
        p += n; left -= n;
        return true;
    };

    int32_t n_lang = 0;
    if (!take(&n_lang, 4) || n_lang < 1) { std::fprintf(stderr, "bad inputs\n"); return 1; }
    std::vector<int32_t> lang((size_t) n_lang);
    if (!take(lang.data(), (size_t) n_lang * 4)) { std::fprintf(stderr, "bad tokens\n"); return 1; }

    const int views = 2;
    const size_t per_pix = (size_t) 3 * img * img;
    std::vector<uint8_t> pixels(per_pix * views);
    if (!take(pixels.data(), pixels.size())) { std::fprintf(stderr, "bad pixels\n"); return 1; }

    float state[32] = {};
    float noise[30 * 32] = {};
    if (!take(state, sizeof(state)) || !take(noise, sizeof(noise))) {
        std::fprintf(stderr, "bad state/noise\n"); return 1;
    }

    vla::Model * model = vla::model_load(mmproj, ckpt, "");
    if (!model) { std::fprintf(stderr, "xr0-parity: model_load failed\n"); return 1; }

    std::vector<vla::ImageView> image_views(views);
    for (int v = 0; v < views; ++v) {
        image_views[v].data   = pixels.data() + (size_t) v * per_pix;
        image_views[v].w      = img;
        image_views[v].h      = img;
        image_views[v].format = vla::PixelFormat::U8;
    }

    vla::Inputs in{};
    in.images    = image_views.data();
    in.n_images  = views;
    in.lang_tokens = lang.data();
    in.n_lang      = n_lang;
    in.state       = state;
    in.noise       = noise;

    std::vector<float> actions = vla::predict(model, in);
    const auto & cfg = vla::model_config(model);
    const auto & st = vla::last_stats(model);
    std::printf("xr0-parity: chunk=%lld total=%.1f ms vision=%.1f ms inference=%.1f ms\n",
                (long long) cfg.n_suffix, st.ms_total, st.ms_vision, st.ms_inference);
    if (actions.size() != (size_t) (cfg.n_suffix * cfg.max_action_dim)) {
        std::fprintf(stderr, "bad action size %zu\n", actions.size());
        return 1;
    }

    {
        std::ofstream f(dir + "/xr0_parity_out" + sfx + ".bin", std::ios::binary);
        f.write(reinterpret_cast<const char *>(actions.data()), actions.size() * 4);
    }
    std::printf("xr0-parity: wrote %zu actions; first row:", actions.size());
    for (int j = 0; j < 7; ++j) std::printf(" %+.5f", actions[j]);
    std::printf("\n");

    (void) dump_vision; // reserved for vision-only debug dumps
    return 0;
}
