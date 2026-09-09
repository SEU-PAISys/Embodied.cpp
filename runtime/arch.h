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
    XR0,        ///< Xiaomi-Robotics-0 (Qwen3-VL backbone + DiT flow-matching head).
    TURBOVLA,   ///< TurboVLA direct V+L->A policy (DINOv3 + BERT + bidirectional cross-attn fusion).
    XVLA,       ///< X-VLA (Florence-2 DaViT + BART encoder + domain-conditioned flow head).
    COSMOS3,    ///< NVIDIA Cosmos3 video-action world model.
    GROOT_N1,   ///< NVIDIA GR00T N1 vision-language-action policy.
    SMOLVLA,    ///< LeRobot SmolVLA (SigLIP + SmolLM2 + flow-matching action expert).
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

    virtual bool supports_wam() const { return false; }
    virtual WamOutput predict_wam(const WamInputs&) {
        WamOutput out;
        out.error = "architecture does not implement the typed WAM interface";
        return out;
    }
    virtual std::string reset_wam(uint64_t) { return {}; }
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
 * @brief Build a LingBot-VA model from transformer and component GGUFs.
 * @param mmproj_path Ignored for LingBot-VA.
 * @param ckpt_path   Path to the LingBot-VA transformer GGUF.
 * @param config_path Optional JSON override; pass empty to use bundled config.
 * @param components  UMT5 text encoder and Wan VAE encoder GGUF paths.
 * @return Owning pointer to the constructed model.
 */
std::unique_ptr<ModelArchBase> lingbot_va_create(const std::string& mmproj_path,
                                                 const std::string& ckpt_path,
                                                 const std::string& config_path,
                                                 const LingBotComponentPaths& components);

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

std::unique_ptr<ModelArchBase> cosmos3_create(const std::string& mmproj_path,
                                              const std::string& ckpt_path,
                                             const std::string& config_path);

std::unique_ptr<ModelArchBase> groot_n1_create(const std::string& mmproj_path,
                                               const std::string& ckpt_path,
                                                const std::string& config_path);

/**
 * @brief Build a SmolVLA model from its mmproj (SigLIP vision tower + connector)
 *        and policy GGUF (SmolLM2 text backbone + flow-matching action expert +
 *        action projections + normalization statistics).
 * @param mmproj_path Path to the SmolVLA mmproj GGUF (SigLIP + connector).
 * @param ckpt_path   Path to the SmolVLA policy GGUF.
 * @param config_path Optional JSON override; pass empty to use bundled config.
 * @return Owning pointer to the constructed model.
 */
std::unique_ptr<ModelArchBase> smolvla_create(const std::string& mmproj_path,
                                               const std::string& ckpt_path,
                                               const std::string& config_path);

/**
 * @brief Build a Xiaomi-Robotics-0 model (Qwen3-VL-4B backbone + 16-layer
 *        DiT flow-matching action head cross-attending to backbone KV).
 *
 * Weights are split across two GGUFs:
 *   - @p mmproj_path: Qwen3-VL vision tower in llama.cpp mmproj format
 *     (produced by llama.cpp's convert_hf_to_gguf.py --mmproj), including
 *     the three deepstack mergers.
 *   - @p ckpt_path: text backbone + DiT head + per-robot action statistics
 *     (produced by scripts/convert_xr0_to_gguf.py).
 *
 * @param mmproj_path Path to the Xiaomi-Robotics-0 mmproj GGUF (vision tower).
 * @param ckpt_path   Path to the Xiaomi-Robotics-0 main GGUF.
 * @param config_path Optional JSON override; pass empty to use bundled config.
 * @return Owning pointer to the constructed model.
 */
std::unique_ptr<ModelArchBase> xr0_create(const std::string& mmproj_path,
                                          const std::string& ckpt_path,
                                          const std::string& config_path);

/**
 * @brief Build a TurboVLA model (DINOv3 ViT + BERT + bidirectional
 *        cross-attn fusion + ACT-style transformer-decoder action head).
 *
 * Unlike the LLM-based policies above, TurboVLA has no language backbone:
 * it is a direct vision+language-to-action (V+L->A) policy that produces a
 * fixed 12-step non-autoregressive action chunk. The whole model (text
 * encoder, vision tower, fusion blocks, action head) lives in a single
 * GGUF produced by scripts/convert_turbovla_to_gguf.py, so @p mmproj_path
 * is ignored. The runtime tokenizes the raw instruction with a bundled
 * WordPiece vocab via @ref Inputs::language_text.
 *
 * @param mmproj_path Ignored for TurboVLA (vision tower is baked into the ckpt).
 * @param ckpt_path   Path to the TurboVLA GGUF.
 * @param config_path Optional JSON override; pass empty to use bundled config.
 * @return Owning pointer to the constructed model.
 */
std::unique_ptr<ModelArchBase> turbovla_create(const std::string& mmproj_path,
                                               const std::string& ckpt_path,
                                               const std::string& config_path);

/**
 * @brief Build an X-VLA model (Florence-2 DaViT vision tower + BART text
 *        encoder + SoftPromptedTransformer domain-conditioned flow head).
 *
 * The whole model lives in a single GGUF produced by
 * scripts/convert_xvla_to_gguf.py, so @p mmproj_path is ignored. The runtime
 * expects BartTokenizer ids via @ref Inputs::lang_tokens (padded to 50 with
 * pad id 1 client-side), bicubic-resized 224x224 RGB views, proprioception
 * via @ref Inputs::state, initial noise via @ref Inputs::noise, and the
 * embodiment's soft-prompt domain via @ref Inputs::domain_id.
 *
 * @param mmproj_path Ignored for X-VLA (vision tower is baked into the ckpt).
 * @param ckpt_path   Path to the X-VLA GGUF.
 * @param config_path Optional JSON override; pass empty to use bundled config.
 * @return Owning pointer to the constructed model.
 */
std::unique_ptr<ModelArchBase> xvla_create(const std::string& mmproj_path,
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
