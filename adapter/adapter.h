// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

/**
 * @file adapter.h
 * @brief Adapter interfaces between embodied I/O and the current model ABI.
 */

#pragma once

#include "adapter/typed_io.h"
#include "runtime/model.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace embodied::adapter {

struct AdapterStatus {
    bool ok = true;
    std::string message;

    static AdapterStatus success();
    static AdapterStatus error(std::string msg);
};

struct AdapterConfig {
    int64_t state_dim = 0;
    int64_t action_dim = 0;
    int64_t action_steps = 0;
    bool pad_state = true;
    bool allow_missing_noise = true;
};

struct ModelInputStorage {
    std::vector<vla::ImageView> image_views;
    std::vector<int32_t> language_tokens;
    std::string instruction;
    std::vector<float> state;
    std::vector<float> noise;

    vla::Inputs inputs{};
};

class InputAdapter {
public:
    virtual ~InputAdapter() = default;
    virtual AdapterStatus reset();
    virtual AdapterStatus build(const Observation& observation,
                                ModelInputStorage* out) const = 0;
};

class OutputAdapter {
public:
    virtual ~OutputAdapter() = default;
    virtual AdapterStatus reset();
    virtual AdapterStatus build_command(const ActionChunk& action,
                                        DeploymentCommand* out) const = 0;
};

/**
 * @brief Default bridge from typed embodied observations to the existing
 *        vla::Inputs struct.
 *
 * This keeps the first refactor conservative: existing model implementations
 * still consume vla::Inputs, while callers can start speaking in terms of
 * images, proprioception, language tokens, and action chunks at the adapter
 * boundary.
 */
class VlaModelInputAdapter final : public InputAdapter {
public:
    explicit VlaModelInputAdapter(AdapterConfig config);

    AdapterStatus build(const Observation& observation,
                        ModelInputStorage* out) const override;

private:
    AdapterConfig config_;
};

class DirectActionOutputAdapter final : public OutputAdapter {
public:
    AdapterStatus build_command(const ActionChunk& action,
                                DeploymentCommand* out) const override;
};

ActionChunk make_action_chunk(const std::vector<float>& values,
                              int64_t steps,
                              int64_t action_dim,
                              uint64_t timestamp_ns = 0);

} // namespace embodied::adapter
