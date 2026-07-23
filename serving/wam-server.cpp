// Copyright 2026 SEU-PAISys
// SPDX-License-Identifier: Apache-2.0

#include "model.h"
#include "serving/wam.pb.h"

#include <zmq.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
std::atomic<bool> g_shutdown{false};
void on_signal(int) { g_shutdown.store(true, std::memory_order_relaxed); }

vla::WamDType from_proto(wam::TensorDType type) {
    switch (type) {
        case wam::F32: return vla::WamDType::F32;
        case wam::BF16: return vla::WamDType::BF16;
        case wam::F16: return vla::WamDType::F16;
        case wam::U8: return vla::WamDType::U8;
        case wam::I32: return vla::WamDType::I32;
        default: return vla::WamDType::F32;
    }
}

wam::TensorDType to_proto(vla::WamDType type) {
    switch (type) {
        case vla::WamDType::F32: return wam::F32;
        case vla::WamDType::BF16: return wam::BF16;
        case vla::WamDType::F16: return wam::F16;
        case vla::WamDType::U8: return wam::U8;
        case vla::WamDType::I32: return wam::I32;
    }
    return wam::F32;
}

size_t element_size(wam::TensorDType type) {
    switch (type) {
        case wam::F32: case wam::I32: return 4;
        case wam::BF16: case wam::F16: return 2;
        case wam::U8: return 1;
        default: return 0;
    }
}

bool validate_tensor(const wam::Tensor & src, std::string * error) {
    if (src.name().empty()) { *error = "tensor name must not be empty"; return false; }
    size_t elements = 1;
    for (uint64_t dim : src.shape()) {
        if (dim == 0 || dim > std::numeric_limits<size_t>::max() / elements) {
            *error = "tensor shape is invalid or overflows";
            return false;
        }
        elements *= static_cast<size_t>(dim);
    }
    const size_t width = element_size(src.dtype());
    if (width == 0 || elements > std::numeric_limits<size_t>::max() / width ||
        src.data().size() != elements * width) {
        *error = "tensor byte count does not match dtype and shape";
        return false;
    }
    return true;
}

std::string serialize_error(uint64_t request_id, const std::string & error) {
    wam::WamResponse response;
    response.set_request_id(request_id);
    response.set_error(error);
    return response.SerializeAsString();
}

void usage(const char * prog) {
    std::fprintf(stderr, "usage: %s [--bind ADDR] <wam-model.gguf>\n", prog);
}
} // namespace

int main(int argc, char ** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::string bind_addr = "tcp://*:5557";
    std::vector<std::string> positionals;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--bind" && i + 1 < argc) bind_addr = argv[++i];
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else positionals.push_back(arg);
    }
    if (positionals.size() != 1) { usage(argv[0]); return 1; }

    vla::Model * model = vla::model_load("", positionals[0]);
    if (!model) return 1;
    if (!vla::model_supports_wam(model)) {
        std::fprintf(stderr, "wam-server: loaded model does not implement WAM inputs\n");
        vla::model_free(model);
        return 1;
    }

    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::rep);
    int linger = 0;
    socket.setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
    socket.bind(bind_addr);
    std::printf("wam-server: bound to %s. ready.\n", bind_addr.c_str());
    std::fflush(stdout);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    while (!g_shutdown.load(std::memory_order_relaxed)) {
        zmq::message_t wire;
        try {
            if (!socket.recv(wire, zmq::recv_flags::none)) continue;
        } catch (const zmq::error_t & e) {
            if (g_shutdown.load(std::memory_order_relaxed) || e.num() == EINTR) break;
            throw;
        }
        wam::WamRequest request;
        if (!request.ParseFromArray(wire.data(), static_cast<int>(wire.size()))) {
            socket.send(zmq::buffer(serialize_error(0, "WamRequest parse failed")), zmq::send_flags::none);
            continue;
        }
        const uint64_t request_id = request.request_id();
        wam::WamResponse response;
        response.set_request_id(request_id);
        if (request.has_ping()) {
            response.set_status(request.ping().text().empty() ? "pong" : request.ping().text());
            socket.send(zmq::buffer(response.SerializeAsString()), zmq::send_flags::none);
            continue;
        }
        if (request.has_reset()) {
            response.set_status("reset");
            socket.send(zmq::buffer(response.SerializeAsString()), zmq::send_flags::none);
            continue;
        }
        if (!request.has_step()) {
            socket.send(zmq::buffer(serialize_error(request_id, "request must contain ping, reset, or step")),
                        zmq::send_flags::none);
            continue;
        }

        const auto & step = request.step();
        std::vector<std::vector<uint8_t>> storage;
        std::vector<vla::WamTensorView> tensors;
        storage.reserve(step.tensors_size());
        tensors.reserve(step.tensors_size());
        std::string error;
        bool valid = true;
        for (const auto & tensor : step.tensors()) {
            if (!validate_tensor(tensor, &error)) { valid = false; break; }
            storage.emplace_back(tensor.data().begin(), tensor.data().end());
            vla::WamTensorView view;
            view.name = tensor.name();
            view.dtype = from_proto(tensor.dtype());
            view.data = storage.back().data();
            view.bytes = storage.back().size();
            view.shape.assign(tensor.shape().begin(), tensor.shape().end());
            tensors.push_back(std::move(view));
        }
        if (!valid) {
            socket.send(zmq::buffer(serialize_error(request_id, error)), zmq::send_flags::none);
            continue;
        }

        vla::WamInputs inputs;
        inputs.session_id = step.session_id();
        inputs.instruction = step.instruction();
        inputs.state.assign(step.state().begin(), step.state().end());
        inputs.tensors = std::move(tensors);
        for (const auto & item : step.params()) {
            inputs.params.emplace(item.first, item.second);
        }
        const auto begin = std::chrono::steady_clock::now();
        vla::WamOutput out = vla::wam_predict(model, inputs);
        const auto end = std::chrono::steady_clock::now();
        if (!out.error.empty()) {
            response.set_error(out.error);
        } else {
            for (float value : out.action) response.add_action_chunk(value);
            response.set_chunk_size(static_cast<uint32_t>(out.action_steps));
            response.set_action_dim(static_cast<uint32_t>(out.action_dim));
            for (const auto & tensor : out.tensors) {
                wam::Tensor * dst = response.add_tensors();
                dst->set_name(tensor.name);
                dst->set_dtype(to_proto(tensor.dtype));
                for (int64_t dim : tensor.shape) dst->add_shape(static_cast<uint64_t>(dim));
                dst->set_data(tensor.data.data(), tensor.data.size());
            }
        }
        const auto elapsed = std::chrono::duration<float, std::milli>(end - begin).count();
        response.set_latency_ms_total(elapsed);
        response.set_latency_ms_inference(vla::last_stats(model).ms_inference);
        socket.send(zmq::buffer(response.SerializeAsString()), zmq::send_flags::none);
    }
    vla::model_free(model);
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
