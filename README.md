# Embodied.cpp 🤖

<p align="center">
  <img src="assets/embodied-cpp-icon.png" alt="embodied.cpp overview" width="100%">
</p>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.md)
[![arXiv](https://img.shields.io/badge/arXiv-2607.02501-b31b1b.svg)](https://arxiv.org/abs/2607.02501)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-SEU--PAISys%2FEmbodied.cpp-yellow)](https://huggingface.co/SEU-PAISys/Embodied.cpp)
<!-- Reserved for future badges:
[![GitHub stars](https://img.shields.io/github/stars/SEU-PAISys/Embodied.cpp?style=social)](#)
[![GitHub forks](https://img.shields.io/github/forks/SEU-PAISys/Embodied.cpp?style=social)](#)
[![GitHub issues](https://img.shields.io/github/issues/SEU-PAISys/Embodied.cpp)](#)
[![Last commit](https://img.shields.io/github/last-commit/SEU-PAISys/Embodied.cpp)](#)
[![Trending #1](https://img.shields.io/badge/Trending-%231-lightgrey)](#)
-->

`Embodied.cpp` is an inference runtime for **embodied AI models**: Vision-Language-Action (VLA) models and World-Action Models (WAMs) for robotic perception and control. It runs these models efficiently on heterogeneous hardware (CPU / CUDA GPU / NPU) using GGUF weights, and ships with ready-to-use servers and evaluation clients.

---
## NEWS

- **[2026.09]** Added support for **Xiaomi-Robotics-0**, **TurboVLA**, and **X-VLA** runtimes with full LIBERO evaluations (`docs/results/`).
- **[2026.08]** 🔥🔥 Released Embodied.cpp v1.0.
- **[2026.07]** Added support for **Cosmos3-Nano** and **GR00T N1.7**, the **RoboLab** benchmark, and Isaac Sim.
- **[2026.06]** Released the initial version of Embodied.cpp with support for **pi0.5**, **HY-VLA**, and **LingBot-VA**, plus the **LIBERO** and **RoboTwin** benchmarks.

---

## 🎬 Demos

### GR00T N1.7

https://github.com/user-attachments/assets/d9cfa5fb-2145-4500-9257-6bd99a9406b3

- Successful execution ✅
- Cumulative inference time 2.5 s → 1.7 s (⬇32%).
- End-to-end execution time 14.6 s → 13.3 s. 

### HY-VLA

https://github.com/user-attachments/assets/586fc8fe-b87e-4d04-b896-88756cbbc0f4

- Successful execution ✅
- Cumulative inference time 9.4 s → 6.6 s (⬇30%).
- End-to-end execution time 14.8 s → 12.0 s. 
  
---

## Table of Contents

- [Embodied.cpp 🤖](#embodiedcpp-)
  - [NEWS](#news)
  - [🎬 Demos](#-demos)
    - [GR00T N1.7](#gr00t-n17)
    - [HY-VLA](#hy-vla)
  - [Table of Contents](#table-of-contents)
  - [1. 🧭 Current Support and Roadmap](#1--current-support-and-roadmap)
    - [1.1 Supported Models](#11-supported-models)
      - [Vision-Language-Action Models](#vision-language-action-models)
      - [World Models](#world-models)
    - [1.2 Performance Acceleration](#12-performance-acceleration)
    - [1.3 Runtime Roadmap](#13-runtime-roadmap)
  - [2. 🚀 Quick Start](#2--quick-start)
    - [2.1 Clone the Repo](#21-clone-the-repo)
    - [2.2 Get GGUF Weights](#22-get-gguf-weights)
    - [2.3 Install System Dependencies](#23-install-system-dependencies)
    - [2.4 Build by Model and Backend](#24-build-by-model-and-backend)
    - [2.5 Start a Server](#25-start-a-server)
  - [3. 🧪 Evaluation](#3--evaluation)
    - [3.1 LIBERO](#31-libero)
    - [3.2 RoboTwin](#32-robotwin)
    - [3.3 RoboLab (Cosmos3-Nano)](#33-robolab-cosmos3-nano)
  - [4. 🔧 Convert and Quantize Models](#4--convert-and-quantize-models)
  - [5. 🗂️ Project Structure](#5-️-project-structure)
  - [6. 🚧 Known Limitations & Future Work](#6--known-limitations--future-work)
  - [7. 📄 Citation](#7--citation)
  - [8. ⚖️ License](#8-️-license)
  - [9. 🙏 Acknowledgements](#9--acknowledgements)

---

## 1. 🧭 Current Support and Roadmap

### 1.1 Supported Models

#### Vision-Language-Action Models

<table>
  <tr>
    <td align="center" width="25%">
      <a href="https://github.com/Physical-Intelligence/openpi">
        <img src="https://github.com/Physical-Intelligence.png?size=160" alt="Physical Intelligence" height="72"><br>
        <strong>pi0.5</strong>
      </a><br>
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/huggingface/lerobot">
        <img src="https://github.com/huggingface.png?size=160" alt="Hugging Face" height="72"><br>
        <strong>SmolVLA</strong>
      </a><br>
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA">
        <img src="https://github.com/Tencent-Hunyuan.png?size=160" alt="Tencent Hunyuan" height="72"><br>
        <strong>HY-VLA</strong>
      </a><br>
    </td>
    <td align="center" width="25%">
      <a href="https://developer.nvidia.com/isaac/gr00t">
        <img src="https://github.com/NVIDIA.png?size=160" alt="NVIDIA" height="72"><br>
        <strong>GR00T N1.7</strong>
      </a><br>
    </td>
  </tr>
  <tr>
    <td align="center" width="25%">
      <a href="https://github.com/XiaomiRobotics/Xiaomi-Robotics-0">
        <img src="https://github.com/XiaomiRobotics.png?size=160" alt="Xiaomi Robotics" height="72"><br>
        <strong>Xiaomi-Robotics-0</strong>
      </a><br>
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/H-EmbodVis/TurboVLA">
        <img src="https://github.com/H-EmbodVis.png?size=160" alt="H-EmbodVis" height="72"><br>
        <strong>TurboVLA</strong>
      </a><br>
    </td>
    <td align="center" width="25%">
      <a href="https://github.com/2toinf/X-VLA">
        <img src="https://github.com/2toinf.png?size=160" alt="X-VLA" height="72"><br>
        <strong>X-VLA</strong>
      </a><br>
    </td>
    <td align="center" width="25%"></td>
  </tr>
</table>

#### World Models

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://github.com/robbyant/lingbot-va">
        <img src="https://github.com/robbyant.png?size=160" alt="LingBot" height="72"><br>
        <strong>LingBot-VA</strong>
      </a><br>
    </td>
    <td align="center" width="50%">
      <a href="https://github.com/NVIDIA/Cosmos">
        <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" alt="NVIDIA Cosmos" height="72"><br>
        <strong>Cosmos3-Nano</strong>
      </a><br>
    </td>
  </tr>
</table>

We continuously track advances in embodied AI and adapt `Embodied.cpp` to the latest open models. Pull requests that add support for new models are always welcome.

### 1.2 Performance Acceleration

VLA results are normalized to each model's Python baseline (`1.00`), and each cell reports **Python → C++**. Lower inference latency and VRAM are better; higher success rate is better. `C++` denotes the BF16 implementation.

| Model | Inference Latency ↓ | VRAM ↓ |
|---|---:|---:|
| **pi0.5** | 1.00 → 0.90 (**10% lower**) |  1.00 → 0.60 (**40% lower**) |
| **GR00T N1.7** | 1.00 → 0.72 (**28% lower**) |  1.00 → 0.93 (**7% lower**) |
| **HY-VLA** | 1.00 → 0.48 (**52% lower**) | 1.00 → 0.68 (**32% lower**) |
| **Xiaomi-Robotics-0** | Python baseline TBD · C++ ≈ 2.17 s / 30-step chunk | 2.06 GiB (q4_k) |
| **TurboVLA** | Python baseline TBD · C++ ≈ 11.9 ms / episode | 0.36 GiB |
| **X-VLA** | Python baseline TBD · C++ ≈ 0.89 s / 30-step chunk | 1.70 GiB |

The three integrated models were measured on an RTX 4090 (CUDA 12.8,
BF16 implementation) from the committed parity harnesses and server load
outputs; their Python baselines still require a PyTorch environment to
benchmark side-by-side, so the `Python → C++` ratio is reported once
measured.

For World Models, C++ substantially reduces VRAM while keeping the success rate close to the Python baseline.

| Model | VRAM ↓ |
|---|---:|
| **Cosmos3** | 21.84 GB → 19.49 GB (**10.8% lower**) |
| **LingBot-VA** | 24.75 GB → 16.44 GB (**33.6% lower**) |

> **Highlights:** Compared with Python, C++ BF16 reduces VLA inference latency by up to **52%** and VRAM by up to **40%**. For World Models, it reduces VRAM by up to **33.6%**, with success-rate changes limited to **2 percentage points**.

### 1.3 Runtime Roadmap
- This project is still under active construction 🚧
- [ ] A more modular and maintainable runtime architecture for `Embodied.cpp`
- [ ] Additional inference optimizations, such as real-time chunking and VLA caching
---

## 2. 🚀 Quick Start

### 2.1 Clone the Repo

```bash
git clone <repo-url> && cd embodied.cpp
./patches/init_third_party.sh
```

By default, the setup script prepares a combined `llama.cpp` source tree for all
supported runtimes. For smaller model-specific setups or custom patch profiles,
see [`patches/PATCH.md`](patches/PATCH.md).

### 2.2 Get GGUF Weights

Pre-converted GGUF releases for the original models are available on
Hugging Face:

- https://huggingface.co/SEU-PAISys/Embodied.cpp

The repository currently hosts GGUF artifacts for the original runtime
models:

- `pi0.5`: main policy GGUF plus multimodal projector GGUF
- `GR00T N1.7`: truncated Qwen3-VL text GGUF, vision projector GGUF, and action-head GGUF
- `HY-VLA-0.5`: combined VLA GGUF for RoboTwin and related runtime paths
- `LingBot-VA`: transformer GGUF and companion artifacts used by the LingBot path

The remaining models are **converted locally** from their upstream
checkpoints with the scripts in [`scripts/`](scripts/README.md):

- `Cosmos3-Nano`: [RoboLab WAM GGUF with the Wan VAE encoder](scripts/README.md#cosmos3-nano)
- `SmolVLA`: LeRobot policy GGUF plus SigLIP mmproj GGUF
- `Xiaomi-Robotics-0`: Qwen3-VL-4B backbone + DiT flow-matching action head, converted with `scripts/convert_xr0_to_gguf.py`; quantize with `scripts/quantize_xr0_gguf.py` (q8_0/q6_k/q5_k/q4_k)
- `TurboVLA`: DINOv3 ViT + BERT + bidirectional cross-attn fusion + ACT decoder, converted with `scripts/convert_turbovla_to_gguf.py`
- `X-VLA`: Florence-2 DaViT + BART encoder + domain-conditioned flow head, converted with `scripts/convert_xvla_to_gguf.py`

Recommended local layout:

```text
checkpoints/
  pi05/
    pi05.gguf
    pi05-mmproj.gguf
  groot-n1/
    qwen3vl-backbone-bf16.gguf
    qwen3vl-mmproj-bf16.gguf
    groot-n1.7-libero-object-action-head-bf16.gguf
  Hy-Embodied-0.5-VLA-RoboTwin/
    Hy-Embodied-0.5-VLA-RoboTwin_bf16.gguf
    Hy-Embodied-0.5-VLA-RoboTwin_q4_K.gguf
  lingbot_va/
    lingbot_transformer.gguf
    ...
  cosmos3/
    cosmos3_robolab_full_w8_with_vae_encoder.gguf
  smolvla/
    smolvla.gguf
    mmproj-smolvla.gguf
  xr0/
    xr0.gguf              # convert locally via scripts/convert_xr0_to_gguf.py
    xr0-mmproj.gguf
  turbovla/
    turbovla.gguf         # convert locally via scripts/convert_turbovla_to_gguf.py
  xvla/
    xvla-libero.gguf      # convert locally via scripts/convert_xvla_to_gguf.py
```

You can also convert upstream checkpoints yourself with the scripts in
[`scripts/`](scripts/), but for most users the Hugging Face GGUF releases are
the fastest way to get started.

### 2.3 Install System Dependencies

Install the required system packages for your platform before building.

Minimum build requirements:

- CMake >= 3.22
- A C++17 compiler, such as GCC 11+ or Clang 14+
- CUDA 12.x, optional and required only for GPU builds

**Linux:**
Make sure `cmake`, `protobuf=3.20.3`, `zeromq`, `cppzmq`, `pkg-config` and `uv` are
available before building. A typical Ubuntu/Debian native-Linux installation is:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config protobuf-compiler libprotobuf-dev \
  libzmq3-dev cppzmq-dev libegl1-mesa-dev libglu1-mesa-dev \
  libgl1-mesa-dev ffmpeg iproute2
```

CUDA builds additionally require a compatible NVIDIA driver and CUDA toolkit.
Desktop systems may use `nvidia-smi` for device monitoring; Jetson systems
provide `tegrastats` for resource monitoring. Install `uv` separately if your
distribution does not package it.


### 2.4 Build by Model and Backend

Model switches default to `OFF`. Enable only the runtimes you need.

**CUDA GPU template:**

```bash
CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
CUDA_ARCH=${CUDA_ARCH:-native}

cmake -S . -B <BUILD_DIR> \
  -DCMAKE_BUILD_TYPE=Release \
  -D<MODEL_BUILD_FLAG>=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
cmake --build <BUILD_DIR> --target <SERVER_TARGET> -j$(nproc)
```

Use a separate `<BUILD_DIR>` for each model or CMake configuration, such as
`build-groot-cuda` or `build-lingbot-cuda`; a build directory stores one CMake
configuration and its generated artifacts.

Replace the placeholders with the model you want to build:

| Model | `<MODEL_BUILD_FLAG>` | `<SERVER_TARGET>` |
|---|---|---|
| pi0.5 | `MODEL_BUILD_VLA_PI05` | `vla-server` |
| SmolVLA | `MODEL_BUILD_VLA_SMOLVLA` | `vla-server` |
| HY-VLA | `MODEL_BUILD_VLA_HY_VLA` | `vla-server` |
| GR00T N1.7 | `MODEL_BUILD_VLA_GROOT_N1` | `vla-server` |
| LingBot-VA | `MODEL_BUILD_WAM_LINGBOT_VA` | `wam-lingbot-server` |
| Cosmos3-Nano | `MODEL_BUILD_WAM_COSMOS3` | `wam-server` |
| Xiaomi-Robotics-0 | `MODEL_BUILD_VLA_XR0` | `vla-server` |
| TurboVLA | `MODEL_BUILD_VLA_TURBOVLA` | `vla-server` |
| X-VLA | `MODEL_BUILD_VLA_XVLA` | `vla-server` |

`CUDA_ARCH` defaults to `native`, so CMake detects the GPU installed on the
build machine. Override it with an explicit architecture when cross-compiling or
when using CMake older than 3.24. Common explicit values include `75` (Turing),
`80` or `86` (Ampere), `87` (Ampere, Jetson AGX Orin), `89` (Ada), `90`
(Hopper), and `120` (Blackwell). The selected CUDA toolkit must support that
architecture; for example, Blackwell `sm_120` requires CUDA 12.8 or newer.

**Jetson AGX Orin with JetPack 6:**
JetPack 6 commonly provides CMake 3.22, so select Orin's compute capability
explicitly instead of using the `native` value introduced in CMake 3.24:

```bash
CUDA_HOME=/usr/local/cuda
cmake -S . -B build-pi05-jetson \
  -DCMAKE_BUILD_TYPE=Release \
  -DMODEL_BUILD_VLA_PI05=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build-pi05-jetson --target vla-server -j4
```

cuDNN is detected automatically and enables accelerated convolution paths for
the LingBot-VA and Cosmos3 WAM CUDA builds. When cuDNN is unavailable, the
existing CUDA fallback paths remain enabled. Discovery searches normal system
and multiarch locations as well as the CUDA toolkit. Set `CUDNN_ROOT` for a
non-system installation.

### 2.5 Start a Server

```bash
./<BUILD_DIR>/<SERVER_TARGET> <MODEL_ARGUMENTS>
```

Use the `<BUILD_DIR>` and `<SERVER_TARGET>` selected in section 2.4. Replace
`<MODEL_ARGUMENTS>` with the arguments for the selected model:

| Model | `<SERVER_TARGET>` | `<MODEL_ARGUMENTS>` |
|---|---|---|
| pi0.5 | `vla-server` | `<MMPROJ_GGUF> <MODEL_GGUF>` |
| SmolVLA | `vla-server` | `<MMPROJ_GGUF> <MODEL_GGUF>` |
| HY-VLA | `vla-server` | `<MODEL_GGUF>` |
| GR00T N1.7 | `vla-server` | `--backbone <BACKBONE_GGUF> <MMPROJ_GGUF> <ACTION_HEAD_GGUF>` |
| LingBot-VA | `wam-lingbot-server` | `<TRANSFORMER_GGUF> <TEXT_ENCODER_GGUF> <VAE_ENCODER_GGUF>` |
| Cosmos3-Nano | `wam-server` | `<MODEL_GGUF>` |
| Xiaomi-Robotics-0 | `vla-server` | `<MMPROJ_GGUF> <MODEL_GGUF>` |
| TurboVLA | `vla-server` | `<MODEL_GGUF>` |
| X-VLA | `vla-server` | `<MODEL_GGUF>` |

VLA servers bind to `tcp://*:5555` by default. LingBot-VA and Cosmos3-Nano
bind to `tcp://*:5557` by default. Pass `--bind <ADDR>` to override the
listening address or port.

## 3. 🧪 Evaluation

Start the required server as described in section 2.5, then select the
configuration and runner for the model and benchmark you want to evaluate.

| Model | Benchmark | Configuration | Results | Server |
|---|---|---|---|---|
| pi0.5 | LIBERO | [pi0.5](eval/conf/libero_pi05_eval.yaml) | - | Manual |
| SmolVLA | LIBERO | [SmolVLA](eval/conf/libero_smolvla_eval.yaml) | - | Manual |
| GR00T N1.7 | LIBERO | [GR00T](eval/conf/libero_groot_n1_eval.yaml) | - | Manual |
| LingBot-VA | LIBERO | [LingBot](eval/conf/libero_lingbot_va_eval.yaml) | - | Manual |
| HY-VLA | RoboTwin | [HY-VLA](eval/conf/robotwin_hy_vla_eval.yaml) | - | Managed |
| Cosmos3-Nano | RoboLab | [Cosmos3](eval/conf/robolab_cosmos3_eval.yaml) | - | Managed |
| Xiaomi-Robotics-0 | LIBERO | [Xiaomi-Robotics-0](eval/conf/libero_xr0_eval.yaml) | [report](docs/results/xr0_libero.md) | Manual |
| TurboVLA | LIBERO | [TurboVLA](eval/conf/libero_turbovla_eval.yaml) | [report](docs/results/turbovla_libero.md) | Manual |
| X-VLA | LIBERO | [X-VLA](eval/conf/libero_xvla_eval.yaml) | [report](docs/results/xvla_libero.md) | Manual |

LIBERO uses `eval/client/run_sim_client_direct.py`; start its matching server
separately. RoboTwin and RoboLab runners start their servers from the selected
configuration and stop them when the evaluation finishes.

**LIBERO with a manual server** (all VLA models follow the same two-step flow):

```bash
# 1. Start the server for the model you built (see section 2.5).
./build/vla-server checkpoints/xr0/xr0-mmproj.gguf \
                   checkpoints/xr0/xr0.gguf --bind tcp://*:5555

# 2. Run the matching configuration in a second shell.
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_xr0_eval.yaml
```

Swap the checkpoint and configuration for the model under test; TurboVLA
(`libero_turbovla_eval.yaml`) and X-VLA (`libero_xvla_eval.yaml`) take a single
GGUF, while Xiaomi-Robotics-0 takes the mmproj GGUF first.

**SmolVLA on LIBERO:**

Convert a LeRobot LIBERO checkpoint into the policy GGUF and the SigLIP
identity-proxy mmproj. The pixel-shuffle connector is stored in the policy
GGUF and executed by `models/smolvla.cpp`.

```bash
python scripts/convert_smolvla_to_gguf.py \
  --ckpt checkpoints/smolvla_libero \
  --out checkpoints/smolvla/smolvla.gguf
python scripts/convert_smolvla_mmproj_to_gguf.py \
  --ckpt checkpoints/smolvla_libero \
  --out checkpoints/smolvla/mmproj-smolvla.gguf

# Start `vla-server` with MODEL_BUILD_VLA_SMOLVLA=ON, then run the smoke test.
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_smolvla_eval.yaml
```

The serialized SmolVLA processor requires a trailing newline in each task
prompt; the direct client applies it automatically. The checked-in configuration
uses one replayed action per model request and the full LIBERO episode horizon.

See [eval/SMOLVLA_VALIDATION.md](eval/SMOLVLA_VALIDATION.md) for the acceptance
matrix, smoke protocol, build matrix, and parity methodology. The implementation
and validation report is available at
[eval/SMOLVLA_TECHNICAL_REPORT_ZH.md](eval/SMOLVLA_TECHNICAL_REPORT_ZH.md).

**HY-VLA on RoboTwin:**

Each checked-in YAML is a baseline evaluation configuration. Adjust its task
selection, episode count, model paths, output location, and other
benchmark-specific settings for your run; the exact field names are documented
in the corresponding configuration and simulator README.

### 3.1 LIBERO

LIBERO tests robotic manipulation skills on the `spatial`, `object`, `goal`,
`short`, and `long` suites. Install the simulator once:

```bash
bash eval/sim/libero/setup_libero.sh
```

After starting the matching server in another terminal, run a checked-in
configuration:

```bash
# pi0.5
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_pi05_eval.yaml

# GR00T N1.7
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_groot_n1_eval.yaml

# LingBot-VA
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_lingbot_va_eval.yaml

# Xiaomi-Robotics-0
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_xr0_eval.yaml

# TurboVLA
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_turbovla_eval.yaml

# X-VLA
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_xvla_eval.yaml
```

See [`eval/sim/libero/README.md`](eval/sim/libero/README.md) for LIBERO suite
selection, headless EGL execution, and configuration details.

### 3.2 RoboTwin

RoboTwin is a dual-arm manipulation benchmark. Install it once:

```bash
bash eval/sim/robotwin/setup_robotwin.sh
```

Run HY-VLA with the standard configuration:

```bash
eval/sim/robotwin/robotwin_uv/.venv/bin/python \
  eval/client/run_robotwin_eval.py \
  --conf eval/conf/robotwin_hy_vla_eval.yaml
```

See [`eval/sim/robotwin/README.md`](eval/sim/robotwin/README.md) for detailed setup modes and troubleshooting.

### 3.3 RoboLab (Cosmos3-Nano)

RoboLab evaluates the native C++ Cosmos3 WAM path. Install RoboLab once
without launching its optional Isaac Sim smoke test:

```bash
RUN_SMOKE_TEST=0 bash eval/sim/robolab/setup_robolab.sh
```

Run Cosmos3-Nano with the standard configuration:

```bash
python3 eval/client/run_robolab_eval.py \
  --conf eval/conf/robolab_cosmos3_eval.yaml
```

The runner starts the C++ `wam-server` automatically. See
[`eval/sim/robolab/README.md`](eval/sim/robolab/README.md) for configuration
details, the transport-only smoke test, and the PyTorch-reference path.

## 4. 🔧 Convert and Quantize Models

Pre-converted GGUF releases for the original runtime models are available on
[Hugging Face](https://huggingface.co/SEU-PAISys/Embodied.cpp); the newer
models are converted locally from their upstream checkpoints (see section
2.2). Use the conversion tools when preparing a compatible upstream
checkpoint or a custom quantization.

| Model | Workflow |
|---|---|
| pi0.5 | [Policy and vision projector](scripts/README.md#pi05) |
| GR00T N1.7 | [Action head and Qwen3-VL backbone](scripts/README.md#groot-n17) |
| HY-VLA | [Combined GGUF and quantization](scripts/README.md#hy-vla) |
| LingBot-VA | [Model artifacts and Wan quantization](scripts/README.md#lingbot-va) |
| Cosmos3-Nano | [RoboLab full_w8 GGUF](scripts/README.md#cosmos3-nano) |
| Xiaomi-Robotics-0 | [GGUF conversion and k-quantization](scripts/README.md#xiaomi-robotics-0) |
| TurboVLA | [Self-contained GGUF](scripts/README.md#turbovla) |
| X-VLA | [Policy GGUF](scripts/README.md#x-vla) |

See [`scripts/README.md`](scripts/README.md) for prerequisites, commands,
expected outputs, and post-conversion checks.

## 5. 🗂️ Project Structure

What lives where, in plain language:

| Directory | What it contains |
|---|---|
| `models/` | C++ implementations of supported models |
| `runtime/` | Model registry, architecture detection, shared utilities |
| `adapter/` | Typed I/O boundary between observations and model inputs |
| `serving/` | ZeroMQ/Protobuf inference servers and API definitions |
| `kernels/` | Custom CUDA kernels for GPU builds |
| `scripts/` | GGUF conversion and quantization tools |
| `patches/` | Third-party setup patches |
| `eval/` | Evaluation clients, configurations, and simulator integrations |

## 6. 🚧 Known Limitations & Future Work

**Project-wide:**
- No unit test framework; coverage relies on smoke tests and LIBERO/RoboTwin evaluation scripts.
- No batched inference (batch_size=1 only).

## 7. 📄 Citation

If you find `Embodied.cpp` useful in your research, please consider citing:

```bibtex
@article{xu2026embodiedcpp,
  title={Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots},
  author={Xu, Ling and Han, Chuyu and Li, Borui and Wu, Hao and Jiang, Shiqi and Cao, Ting and Li, Chuanyou and Zhong, Sheng and Wang, Shuai},
  journal={arXiv preprint arXiv:2607.02501},
  year={2026},
  doi={10.48550/arXiv.2607.02501},
  url={https://arxiv.org/abs/2607.02501}
}
```

## 8. ⚖️ License

This project is released under the [Apache License 2.0](LICENSE.md). Third-party dependencies, model checkpoints, datasets, and upstream reference implementations are distributed under their own licenses.

## 9. 🙏 Acknowledgements

**Supported models:**
- [pi0.5 / OpenPI](https://github.com/Physical-Intelligence/openpi)
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)
- [HY-VLA](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA)
- [LingBot-VA](https://github.com/robbyant/lingbot-vla)
- [NVIDIA Cosmos / Cosmos3-Nano](https://github.com/nvidia/cosmos)
- [SmolVLA](https://github.com/huggingface/lerobot)
- [Xiaomi-Robotics-0](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0)
- [TurboVLA](https://github.com/H-EmbodVis/TurboVLA)
- [X-VLA](https://github.com/2toinf/X-VLA)

**Foundational projects this build depends on:**
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (LLM inference engine)
- [vla.cpp](https://github.com/VinRobotics/vla.cpp) (unified VLA runtime)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (manipulation benchmark)
- [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) (dual-arm robot benchmark)
- [RoboLab](https://github.com/NVlabs/RoboLab) (Isaac Sim DROID evaluation benchmark)
