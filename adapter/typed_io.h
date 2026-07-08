// Copyright 2026 SEU-PAISys
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

/**
 * @file typed_io.h
 * @brief Embodiment-facing typed I/O structures.
 *
 * This layer describes what crosses the runtime boundary: sensor streams,
 * dataset/simulator observations, action chunks, and deployment commands. It is
 * intentionally separate from the model-specific vla::Inputs ABI so adapters
 * can translate between different robots, simulators, and model heads.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace embodied::adapter {

enum class ElementType {
    U8,
    I32,
    F32,
};

enum class ImageLayout {
    HWC,
    CHW,
};

enum class ImageColor {
    RGB,
};

enum class ImageEncoding {
    RAW,
    JPEG,
};

enum class PortKind {
    SENSOR,
    DATASET,
    SIMULATOR,
    ROBOT,
};

struct PortDescriptor {
    std::string name;
    PortKind    kind = PortKind::SENSOR;
    double      nominal_hz = 0.0;
};

struct TensorView {
    const void* data = nullptr;
    ElementType type = ElementType::F32;
    std::vector<int64_t> shape;
};

struct NamedTensorView {
    std::string name;
    TensorView  tensor;
};

struct ImageView {
    const void*   data = nullptr;
    int           width = 0;
    int           height = 0;
    ElementType   type = ElementType::U8;
    ImageLayout   layout = ImageLayout::HWC;
    ImageColor    color = ImageColor::RGB;
    ImageEncoding encoding = ImageEncoding::RAW;
    std::string   frame_id;
    uint64_t      timestamp_ns = 0;
};

struct Observation {
    uint64_t timestamp_ns = 0;
    std::string instruction;

    std::vector<ImageView> images;
    std::vector<int32_t>   language_tokens;
    std::vector<float>     proprioception;
    std::vector<float>     action_history;
    std::vector<float>     noise;

    std::vector<NamedTensorView> extra_inputs;
};

struct ActionChunk {
    std::vector<float> values;
    int64_t steps = 0;
    int64_t action_dim = 0;
    uint64_t timestamp_ns = 0;
    std::string frame_id;
};

struct DeploymentCommand {
    ActionChunk action;
    std::vector<NamedTensorView> extra_outputs;
};

} // namespace embodied::adapter
