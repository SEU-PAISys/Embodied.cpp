// Copyright 2026 VinRobotics
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "adapter/adapter.h"

#include <algorithm>
#include <utility>

namespace embodied::adapter {

AdapterStatus AdapterStatus::success() {
    return {};
}

AdapterStatus AdapterStatus::error(std::string msg) {
    AdapterStatus st;
    st.ok = false;
    st.message = std::move(msg);
    return st;
}

AdapterStatus InputAdapter::reset() {
    return AdapterStatus::success();
}

AdapterStatus OutputAdapter::reset() {
    return AdapterStatus::success();
}

namespace {

bool valid_raw_rgb_image(const ImageView& image) {
    return image.data != nullptr &&
           image.width > 0 &&
           image.height > 0 &&
           image.encoding == ImageEncoding::RAW &&
           image.color == ImageColor::RGB &&
           (image.layout == ImageLayout::HWC || image.layout == ImageLayout::CHW) &&
           (image.type == ElementType::U8 || image.type == ElementType::F32);
}

vla::PixelFormat to_vla_pixel_format(ElementType type) {
    return type == ElementType::F32 ? vla::PixelFormat::F32_RGB_01
                                    : vla::PixelFormat::U8;
}

} // namespace

VlaModelInputAdapter::VlaModelInputAdapter(AdapterConfig config)
    : config_(config) {}

AdapterStatus VlaModelInputAdapter::build(const Observation& observation,
                                          ModelInputStorage* out) const {
    if (!out) {
        return AdapterStatus::error("ModelInputStorage must not be null");
    }
    if (observation.language_tokens.empty()) {
        return AdapterStatus::error("observation.language_tokens must not be empty");
    }
    if (config_.state_dim < 0 || config_.action_dim < 0 || config_.action_steps < 0) {
        return AdapterStatus::error("adapter dimensions must be non-negative");
    }

    out->image_views.clear();
    out->language_tokens = observation.language_tokens;
    out->state.clear();
    out->noise.clear();
    out->inputs = {};

    out->image_views.reserve(observation.images.size());
    for (const ImageView& image : observation.images) {
        if (!valid_raw_rgb_image(image)) {
            return AdapterStatus::error(
                "only borrowed raw RGB U8/F32 images are supported by VlaModelInputAdapter");
        }
        if (image.layout != ImageLayout::HWC) {
            return AdapterStatus::error(
                "VlaModelInputAdapter currently expects HWC images for vla::Inputs");
        }
        out->image_views.push_back(
            vla::ImageView{image.data, image.width, image.height,
                           to_vla_pixel_format(image.type)});
    }

    const size_t state_dim = static_cast<size_t>(std::max<int64_t>(0, config_.state_dim));
    if (state_dim > 0) {
        if (observation.proprioception.size() > state_dim) {
            return AdapterStatus::error("observation.proprioception exceeds configured state_dim");
        }
        if (!config_.pad_state && observation.proprioception.size() != state_dim) {
            return AdapterStatus::error("observation.proprioception does not match state_dim");
        }
        out->state.assign(state_dim, 0.0f);
        std::copy(observation.proprioception.begin(),
                  observation.proprioception.end(),
                  out->state.begin());
    } else {
        out->state = observation.proprioception;
    }

    const int64_t expected_noise =
        config_.action_steps > 0 && config_.action_dim > 0
            ? config_.action_steps * config_.action_dim
            : 0;
    if (!observation.noise.empty()) {
        if (expected_noise > 0 &&
            observation.noise.size() != static_cast<size_t>(expected_noise)) {
            return AdapterStatus::error("observation.noise does not match action_steps * action_dim");
        }
        out->noise = observation.noise;
    } else if (!config_.allow_missing_noise && expected_noise > 0) {
        return AdapterStatus::error("observation.noise is required by adapter config");
    }

    out->inputs.images = out->image_views.empty() ? nullptr : out->image_views.data();
    out->inputs.n_images = static_cast<int>(out->image_views.size());
    out->inputs.lang_tokens = out->language_tokens.data();
    out->inputs.n_lang = static_cast<int>(out->language_tokens.size());
    out->inputs.state = out->state.empty() ? nullptr : out->state.data();
    out->inputs.noise = out->noise.empty() ? nullptr : out->noise.data();

    return AdapterStatus::success();
}

AdapterStatus DirectActionOutputAdapter::build_command(const ActionChunk& action,
                                                       DeploymentCommand* out) const {
    if (!out) {
        return AdapterStatus::error("DeploymentCommand must not be null");
    }
    if (action.steps < 0 || action.action_dim < 0) {
        return AdapterStatus::error("action shape must be non-negative");
    }
    const int64_t expected = action.steps * action.action_dim;
    if (expected > 0 && action.values.size() != static_cast<size_t>(expected)) {
        return AdapterStatus::error("action.values does not match steps * action_dim");
    }
    out->action = action;
    out->extra_outputs.clear();
    return AdapterStatus::success();
}

ActionChunk make_action_chunk(const std::vector<float>& values,
                              int64_t steps,
                              int64_t action_dim,
                              uint64_t timestamp_ns) {
    ActionChunk chunk;
    chunk.values = values;
    chunk.steps = steps;
    chunk.action_dim = action_dim;
    chunk.timestamp_ns = timestamp_ns;
    return chunk;
}

} // namespace embodied::adapter
