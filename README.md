# embodied.cpp

<p align="center">
  <img src="assets/20260622-145312.png" alt="embodied.cpp overview" width="100%">
</p>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
<!-- Reserved for future badges:
[![GitHub stars](https://img.shields.io/github/stars/SEU-PAISys/Embodied.cpp?style=social)](#)
[![GitHub forks](https://img.shields.io/github/forks/SEU-PAISys/Embodied.cpp?style=social)](#)
[![GitHub issues](https://img.shields.io/github/issues/SEU-PAISys/Embodied.cpp)](#)
[![Last commit](https://img.shields.io/github/last-commit/SEU-PAISys/Embodied.cpp)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-coming_soon-yellow)](#)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b)](#)
[![Trending #1](https://img.shields.io/badge/Trending-%231-lightgrey)](#)
-->

`Embodied.cpp` is an inference runtime for **embodied AI models** — Vision-Language-Action (VLA) and World-Action Models (WAM) that let robots perceive and act in the real world. It runs these models efficiently on heterogeneous hardware (CPU / CUDA GPU / NPU) using GGUF weights, and ships ready-to-use servers and evaluation clients.

---

## Table of Contents

- [embodied.cpp](#embodiedcpp)
  - [Table of Contents](#table-of-contents)
  - [Supported Models and Roadmap](#supported-models-and-roadmap)
  - [Quick Start](#quick-start)
    - [1. Prepare dependencies](#1-prepare-dependencies)
    - [2. Build](#2-build)
    - [3. Start a server](#3-start-a-server)
    - [4. Evaluate in simulation (example: LIBERO with LingBot-VA)](#4-evaluate-in-simulation-example-libero-with-lingbot-va)
  - [Run a Server](#run-a-server)
  - [Evaluate in Simulation](#evaluate-in-simulation)
    - [LIBERO](#libero)
    - [RoboTwin](#robotwin)
  - [Convert Your Own Model](#convert-your-own-model)
  - [Project Structure](#project-structure)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

---

## Supported Models and Roadmap

The table below summarizes the models that `Embodied.cpp` already supports and the model families we plan to support next. For a more detailed taxonomy and the original architectural categorization, please refer to our technical report.

| Family | Subtype | Implemented | Planned |
|---|---|---|---|
| VLA | AR-Token VLA | - | [OpenVLA](https://github.com/openvla/openvla), [RT-2](https://arxiv.org/abs/2307.15818)$\dagger$ |
| VLA | VLM-Backboned VLA | [pi0.5](https://github.com/Physical-Intelligence/openpi), [HY-VLA](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA) | [Octo](https://github.com/octo-models/octo), [MuseVLA](https://arxiv.org/abs/2606.17598)$\dagger$ |
| VLA | Hierarchical VLA | - | [Hi Robot](https://arxiv.org/abs/2502.19417), [GeneralVLA](https://github.com/AIGeeksGroup/GeneralVLA-2), [RT-H](https://arxiv.org/abs/2403.01823)$\dagger$, [Gemini Robotics 1.5](https://arxiv.org/abs/2510.03342)$\dagger$ |
| VLA | Asynchronous VLA | - | [GR00T N1](https://developer.nvidia.com/isaac/gr00t), [Fast-in-Slow](https://github.com/CHEN-H01/Fast-in-Slow), [DAM-VLA](https://arxiv.org/abs/2606.12105)$\dagger$ |
| WAM | Predict-then-Act WAM | - | [UniPi](https://github.com/flow-diffusion/AVDC_experiments/) |
| WAM | Unified AR-Modeling WAM | [LingBot-VA](https://github.com/robbyant/lingbot-va) | [WorldVLA](https://github.com/alibaba-damo-academy/RynnVLA-002) |
| WAM | Shared-Backbone WAM | - | [DreamZero](https://github.com/dreamzero0/dreamzero), [FastWAM](https://github.com/yuantianyuan01/FastWAM), [Cosmos Policy](https://github.com/nvlabs/cosmos-policy), [UWM](https://github.com/ShuangLI59/unified_video_action) |
| WAM | Latent-space WAM | - | [LaWAM](https://arxiv.org/abs/2606.15768)$\dagger$, [Being-H0.7](https://github.com/BeingBeyond/Being-H) |

---

$\dagger$ We plan to support this model once it is open sourcece :)
## Quick Start

### 1. Prepare dependencies

```bash
# Clone the repo and fetch third-party code
git clone <repo-url> && cd embodied.cpp
./patches/init_third_party.sh
```

### 2. Build

**CPU-only:**
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vla-server lingbot-world-server -j$(nproc)
```

**CUDA GPU:**
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=<your-arch> \
  -DProtobuf_PROTOC_EXECUTABLE=/usr/bin/protoc
cmake --build build --target lingbot-world-server -j$(nproc)
```

### 3. Start a server

```bash
# VLA server (pi0.5, HY-VLA)
./build/vla-server --model <path-to-gguf> (<path-to-mmproj>)

# LingBot world-action server
./build/lingbot-world-server --model <path-to-gguf>
```

### 4. Evaluate in simulation (example: LIBERO with LingBot-VA)

```bash
# Install the LIBERO runtime once
bash eval/sim/libero/setup_libero.sh

# Run a test episode
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --arch lingbot_va \
  --libero-suite object \
  --task-id 0 \
  --n-episodes 1 \
  --tokenizer /path/to/lingbot-va-tokenizer \
  --vla-addr tcp://localhost:5555
```

---

## Run a Server

| Executable | What it serves |
|---|---|
| `./build/vla-server` | VLA models — takes observations + text, outputs robot action chunks |
| `./build/lingbot-world-server` | LingBot-VA world-action model — video-conditioned future-aware planning |
| `./build/hy-vla-direct-debug` | Debug HY-VLA in-process (no server) |

Run with `--help` to see all model, checkpoint, and quantization options.

---

## Evaluate in Simulation

### LIBERO

LIBERO tests robotic manipulation skills on four task suites: `spatial`, `object`, `goal`, and `10`. A fifth suite `long` (90 tasks) is also available.

```bash
--libero-suite spatial  → libero_spatial
--libero-suite object   → libero_object
--libero-suite goal     → libero_goal
--libero-suite 10       → libero_10
--libero-suite long     → libero_90
```

Use `--task-id 0..9` (or `0..89` for `long`) to pick individual tasks.

### RoboTwin

RoboTWIN is a dual-arm robot benchmark with real-world-style manipulation tasks. Run HY-VLA natively in C++:

```bash
bash eval/sim/robotwin/setup_robotwin.sh   # one-time setup

GGML_CUDA_DISABLE_GRAPHS=1 \
eval/sim/robotwin/robotwin_uv/.venv/bin/python \
  eval/client/run_robotwin_native_hy_vla.py \
  --model <path-to-gguf> \
  --task-name place_empty_cup \
  --episodes 1
```

See [`eval/sim/robotwin/README.md`](eval/sim/robotwin/README.md) for detailed setup modes and troubleshooting.

---

## Convert Your Own Model

GGUF conversion scripts are in [`scripts/`](scripts/):

| Script | Converts |
|---|---|
| `convert_pi05_to_gguf.py` | pi0.5 model weights |
| `convert_pi05_mmproj_to_gguf.py` | pi0.5 multimodal projector |
| `convert_hy_vla_to_gguf.py` | HY-VLA combined vision+action |
| `convert_lingbot_va_to_gguf.py` | LingBot-VA transformer + companion GGUFs |

Quantization helpers:

| Script | Quantizes |
|---|---|
| `quantize_hy_vla_gguf.py` | HY-VLA models |
| `quantize_lingbot_wan_gguf.py` | LingBot-VA models |

---

## Project Structure

What lives where, in plain language:

| Directory | What it contains |
|---|---|
| `models/` | C++ model implementations (pi0.5, HY-VLA, LingBot-VA) |
| `runtime/` | Model registry, architecture detection, shared utilities |
| `adapter/` | I/O boundary — translates sensor/simulator data into typed inputs the models understand |
| `serving/` | Server code (ZeroMQ/Protobuf) for VLA and LingBot APIs |
| `kernels/` | Custom CUDA kernels (used when building with GPU support) |
| `scripts/` | GGUF conversion, quantization, and evaluation helpers |
| `tools/` | Local debug utilities |
| `patches/` | Third-party code patches applied during setup |
| `eval/` | Evaluation clients and simulation setups (LIBERO, RoboTwin) |

---

## License

This project is released under the [Apache License 2.0](LICENSE.md). Third-party dependencies, model checkpoints, datasets, and upstream reference implementations are distributed under their own licenses.

---

## Acknowledgements

**Supported models:**
- [pi0.5 / OpenPI](https://github.com/Physical-Intelligence/openpi)
- [HY-VLA](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA)
- [LingBot-VA](https://github.com/robbyant/lingbot-vla)

**Foundational projects this build depends on:**
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (LLM inference engine)
- [vla.cpp](https://github.com/VinRobotics/vla.cpp) (unified VLA runtime)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (manipulation benchmark)
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) (dual-arm robot benchmark)
