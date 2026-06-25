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

/**
 * @file arch.h
 * @brief Per-architecture model interface and factory declarations.
 *
 * Every supported VLA architecture implements @ref vla::ModelArchBase and is
 * created through a matching @c *_create factory in this header. Adding a new
 * architecture means: extend the @ref vla::Arch enum, declare a new factory
 * here, implement it under @c models/, and wire detection/dispatch in
 * @c runtime/model.cpp.
 */

#pragma once

#include "model.h"

#include <memory>
#include <string>

namespace vla {

/**
 * @brief Identifier for every VLA architecture the engine can serve.
 *
 * The value is detected from the GGUF checkpoint at load time
 * (@ref detect_arch_from_ckpt) and routed to the corresponding factory.
 */
enum class Arch {
    PI05,       ///< Physical Intelligence pi0.5 policy.
    LINGBOT_VA, ///< Robbyant LingBot-VA video-action world model.
    HY_VLA,     ///< Tencent Hy-Embodied-0.5-VLA dual-tower flow policy.
};

/**
 * @brief Common base for every concrete architecture implementation.
 *
 * Each subclass owns its llama.cpp contexts, vision tower, and any custom
 * CUDA state. Construction is performed through the per-arch
 * @c *_create factories declared below; the engine never instantiates
 * @ref ModelArchBase directly.
 */
class ModelArchBase {
public:
    Arch   arch;       ///< The architecture this instance implements.
    Config cfg{};      ///< Resolved model hyper-parameters (see @ref Config).
    Stats  stats{};    ///< Phase timings of the most recent @ref predict call.

    /**
     * @brief Construct a base for the given architecture.
     * @param a Arch tag the subclass implements.
     */
    explicit ModelArchBase(Arch a) : arch(a) {}
    virtual ~ModelArchBase() = default;

    /**
     * @brief Run a full forward pass and return one chunk of normalised actions.
     * @param in Vision + language + state inputs (see @ref Inputs).
     * @return Flattened action chunk of length
     *         @c cfg.num_steps * cfg.real_action_dim.
     */
    virtual std::vector<float> predict(const Inputs& in) = 0;
};

/**
 * @brief Build a pi0.5 model from its mmproj and checkpoint GGUFs.
 * @param mmproj_path Path to the vision-tower GGUF.
 * @param ckpt_path   Path to the LM + action-expert GGUF.
 * @param config_path Optional JSON override; pass empty to use bundled config.
 * @return Owning pointer to the constructed model.
 */
std::unique_ptr<ModelArchBase> pi05_create(const std::string& mmproj_path,
                                           const std::string& ckpt_path,
                                           const std::string& config_path);

/**
 * @brief Build a LingBot-VA model. Vision/text/world-model weights are
 *        planned to be bundled in @p ckpt_path.
 * @param mmproj_path Ignored for LingBot-VA.
 * @param ckpt_path   Path to the LingBot-VA transformer GGUF.
 * @param config_path Optional JSON override; pass empty to use bundled config.
 * @return Owning pointer to the constructed model.
 */
std::unique_ptr<ModelArchBase> lingbot_va_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string& config_path);

/**
 * @brief Build a HY-VLA model. Full model weights are bundled in @p ckpt_path.
 * @param mmproj_path Ignored for HY-VLA.
 * @param ckpt_path   Path to the HY-VLA GGUF.
 * @param config_path Optional JSON override; pass empty to use bundled config.
 * @return Owning pointer to the constructed model.
 */
std::unique_ptr<ModelArchBase> hy_vla_create(const std::string& mmproj_path,
                                             const std::string& ckpt_path,
                                             const std::string& config_path);

/**
 * @brief Inspect a GGUF and identify the architecture tag.
 *
 * Reads the GGUF metadata (without loading weights) and matches it against
 * the per-arch fingerprints declared in @c runtime/model.cpp.
 *
 * @param ckpt_path Path to the candidate GGUF.
 * @param[out] out  Receives the detected @ref Arch on success.
 * @return @c true if the architecture was recognised; @c false otherwise
 *         (and @p out is left untouched).
 */
bool detect_arch_from_ckpt(const std::string& ckpt_path, Arch* out);

}
