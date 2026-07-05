// Copyright 2026 VinRobotics
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

#include "arch.h"
#include "model.h"

#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

namespace vla {

struct Model {
    std::unique_ptr<ModelArchBase> impl;
};

namespace {

bool ends_with_gguf(const std::string& p) {
    if (p.size() < 5) return false;
    return std::strcmp(p.c_str() + p.size() - 5, ".gguf") == 0;
}

bool detect_arch_gguf(const std::string& path, Arch* out) {
    gguf_init_params p{};
    p.no_alloc = true;
    p.ctx      = nullptr;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), p);
    if (!gctx) return false;

    auto try_str = [&](const char * key, std::string& val) -> bool {
        const int64_t kid = gguf_find_key(gctx, key);
        if (kid < 0) return false;
        const char * s = gguf_get_val_str(gctx, kid);
        if (!s) return false;
        val = s;
        return true;
    };

    bool ok = false;
    std::string arch_str;
    if (try_str("general.architecture",  arch_str) ||
        try_str("pi05.architecture",     arch_str) ||
        try_str("lingbot_va.architecture", arch_str) ||
        try_str("hy_vla.architecture",     arch_str)) {
        if      (arch_str == "pi05")       { *out = Arch::PI05;       ok = true; }
        else if (arch_str == "lingbot_va") { *out = Arch::LINGBOT_VA; ok = true; }
        else if (arch_str == "hy_vla")     { *out = Arch::HY_VLA;     ok = true; }
    }

    gguf_free(gctx);
    return ok;
}

bool detect_arch_safetensors(const std::string& path, Arch* out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint64_t header_size = 0;
    f.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
    if (!f || header_size == 0 || header_size > (1u << 28)) return false;
    std::string header(header_size, '\0');
    f.read(header.data(), header_size);
    if (!f) return false;

    if (header.find("paligemma_with_expert.paligemma.") != std::string::npos) {
        if (header.find("time_mlp_in") != std::string::npos) {
            *out = Arch::PI05;
        } else {
            return false;
        }
        return true;
    }
    return false;
}

}

bool detect_arch_from_ckpt(const std::string& ckpt_path, Arch* out) {
    if (!out) return false;
    if (ends_with_gguf(ckpt_path)) return detect_arch_gguf(ckpt_path, out);
    return detect_arch_safetensors(ckpt_path, out);
}

Model* model_load(const std::string& mmproj_path, const std::string& ckpt_path,
                  const std::string& config_path) {
    Arch arch;
    if (!detect_arch_from_ckpt(ckpt_path, &arch)) {
        std::fprintf(stderr,
                     "vla: cannot detect architecture from %s "
                     "(unrecognised GGUF KV / safetensors namespace)\n",
                     ckpt_path.c_str());
        return nullptr;
    }

    std::unique_ptr<ModelArchBase> impl;
    switch (arch) {
        case Arch::PI05:
            std::printf("vla: arch = pi05\n");
#if MODEL_BUILD_VLA_PI05
            impl = pi05_create(mmproj_path, ckpt_path, config_path);
#else
            std::fprintf(stderr,
                         "vla: pi0.5 support was not built into this binary "
                         "(reconfigure with -DMODEL_BUILD_VLA_PI05=ON)\n");
#endif
            break;
        case Arch::LINGBOT_VA:
            std::printf("vla: arch = lingbot_va\n");
#if MODEL_BUILD_WAM_LINGBOT_VA
            impl = lingbot_va_create(mmproj_path, ckpt_path, config_path);
#else
            std::fprintf(stderr,
                         "vla: LingBot-VA support was not built into this binary "
                         "(reconfigure with -DMODEL_BUILD_WAM_LINGBOT_VA=ON)\n");
#endif
            break;
        case Arch::HY_VLA:
            std::printf("vla: arch = hy_vla\n");
#if MODEL_BUILD_VLA_HY_VLA
            impl = hy_vla_create(mmproj_path, ckpt_path, config_path);
#else
            std::fprintf(stderr,
                         "vla: HY-VLA support was not built into this binary "
                         "(reconfigure with -DMODEL_BUILD_VLA_HY_VLA=ON)\n");
#endif
            break;
    }
    if (!impl) return nullptr;

    auto* m = new Model();
    m->impl = std::move(impl);
    return m;
}

void model_free(Model* m) {
    delete m;
}

const Config& model_config(const Model* m) {
    return m->impl->cfg;
}

const Stats& last_stats(const Model* m) {
    return m->impl->stats;
}

std::vector<float> predict(Model* m, const Inputs& in) {
    return m->impl->predict(in);
}

}
