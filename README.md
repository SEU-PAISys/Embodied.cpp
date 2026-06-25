# embodied.cpp

<p align="center">
  <img src="assets/20260622-145312.png" alt="embodied.cpp overview" width="100%">
</p>

`embodied.cpp` is a C++ inference project for embodied VLA models. The current
tree keeps the project-owned implementation in the main source tree and keeps
external upstream code under `third_party/`.


## Directory Layout

```text
models/              C++ implementations of supported VLA models
runtime/             common model API, architecture detection, model registry
adapter/             typed embodied I/O boundary and simulator/deployment adapters
serving/             ZeroMQ/protobuf servers for VLA and LingBot world-action APIs
kernels/             custom CUDA kernels used by LingBot-VA when CUDA is enabled
scripts/             GGUF conversion, quantization, and full-evaluation helpers
tools/               local debug tools
patches/             patches applied to third-party code
third_party/         external code populated outside Git
artifacts/           model graph notes and ONNX audit artifacts
eval/                lightweight client/evaluation helpers kept for this project
```

## Models

`embodied.cpp` currently keeps the project-maintained model implementations in
the main tree and treats upstream reference code as third-party material. Each
supported model is loaded from GGUF metadata and routed to the matching C++
implementation automatically.

| Model | Runtime arch | Backbone / core | GGUF layout | Main server/client path |
|---|---|---|---|---|
| pi0.5 | `pi05` | PaliGemma + flow action expert | action/model GGUF plus matching mmproj GGUF | `vla-server` + generic VLA client |
| HY-VLA | `hy_vla` | Hunyuan-VL / HY-VLA dual-tower policy | combined HY-VLA GGUF with vision/action weights | `vla-server` + `eval/client/run_robotwin_native_hy_vla.py` |
| LingBot-VA | `lingbot_va` | video-action world model with VAE bridge | LingBot transformer GGUF plus text/VAE companion GGUFs | `lingbot-world-server` + LIBERO client |

Conversion scripts are in [`scripts/`](scripts/). The retained converters are:

```text
scripts/convert_pi05_to_gguf.py
scripts/convert_pi05_mmproj_to_gguf.py
scripts/convert_hy_vla_to_gguf.py
scripts/convert_lingbot_va_to_gguf.py
```

The retained GGUF quantizers are:

```text
scripts/quantize_hy_vla_gguf.py
scripts/quantize_lingbot_wan_gguf.py
```

## Initialize Third-party Code

Run this once after cloning the repository:

```bash
./patches/init_third_party.sh
```

The script creates and populates:

```text
third_party/llama.cpp/
```

It also applies:

```text
patches/llama.cpp-vla.patch
```

to `third_party/llama.cpp`. The default llama.cpp ref can be overridden:

```bash
LLAMA_REF=b9016 ./patches/init_third_party.sh
```

## Build

Generic CPU build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vla-server lingbot-world-server hy-vla-direct-debug -j
```


The working configuration is:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DProtobuf_PROTOC_EXECUTABLE=/usr/bin/protoc

cmake --build build --target lingbot-world-server -j"$(nproc)"
```

Notes:

Identify your machine CUDA architecture and set `CMAKE_CUDA_ARCHITECTURES`
accordingly:

| GPU family | Example cards | `CUDA_ARCHITECTURE` |
|---|---|---|
| Ampere (Jetson) | Orin Nano, Orin NX | `87` |
| Ampere (consumer) | RTX 30-series, A40 | `86` |
| Ada Lovelace | RTX 40-series, L40 | `89` |
| Hopper | H100, H200 | `90` |
| Blackwell (consumer) | RTX 50-series | `120` |
| Blackwell (datacenter) | B100, B200, GB200 | `100` |

For example, pass `-DCMAKE_CUDA_ARCHITECTURES=86` for an RTX 30-series GPU.

- Explicitly setting `CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` avoids
  accidentally picking the older `/usr/bin/nvcc`.
- Explicitly setting `Protobuf_PROTOC_EXECUTABLE=/usr/bin/protoc` avoids using
  Conda's newer `protoc`, which can generate C++ headers incompatible with the
  system protobuf library.
- When `GGML_CUDA=ON`, the build enables `VLA_LINGBOT_FLEX_CUDA_KERNELS` and
  compiles `kernels/lingbot/lingbot_flex_attn_cuda.cu`.

The CMake file prefers `/usr/bin/protoc` when available, because Conda-provided
`protoc` versions may generate protobuf files that do not match the system
protobuf headers.

After a successful build, check the server:

```bash
./build/lingbot-world-server --help
ldd ./build/lingbot-world-server | grep "not found" || true
```

## Servers

The main VLA server is:

```bash
./build/vla-server <args>
```

LingBot world-action serving is:

```bash
./build/lingbot-world-server <args>
```

HY-VLA direct debugging is:

```bash
./build/hy-vla-direct-debug <args>
```

Use the executable help output for exact model/checkpoint arguments.

## Model Conversion

Conversion scripts live in `scripts/`; the core converters are listed in
[Models](#models). Quantization helpers for HY-VLA and LingBot-VA are retained
in the same directory.

## LIBERO Evaluation

LIBERO environments are set up under:

```text
eval/sim/libero/
```

Install the LIBERO runtime once:

```bash
bash eval/sim/libero/setup_libero.sh
```

The underlying LIBERO package supports `libero_90`. In this repository,
`libero_long` and `libero-long` are accepted as aliases for `libero_90`, which
is the suite to use for LIBERO-LONG style evaluation.

The direct LIBERO client also accepts a short sub-dataset selector:

```text
--libero-suite spatial -> libero_spatial
--libero-suite object  -> libero_object
--libero-suite goal    -> libero_goal
--libero-suite 10      -> libero_10
--libero-suite long    -> libero_90
```

Use `--task-id` to choose the task inside that suite. The valid task range is
`0..9` for `spatial`, `object`, `goal`, and `10`; it is `0..89` for `long`.
The older full-suite form such as `--task libero_object` still works.

For a direct LingBot-VA LIBERO-LONG smoke test, start `lingbot-world-server`
with the matching checkpoint first, then run:

```bash
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --arch lingbot_va \
  --libero-suite long \
  --task-id 0 \
  --n-episodes 1 \
  --tokenizer /path/to/lingbot-va-tokenizer \
  --vla-addr tcp://localhost:5555
```

For example, to test the LIBERO object suite instead:

```bash
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --arch lingbot_va \
  --libero-suite object \
  --task-id 0 \
  --n-episodes 1 \
  --tokenizer /path/to/lingbot-va-tokenizer \
  --vla-addr tcp://localhost:5555
```

## RobotWin Evaluation For HY-VLA

The recommended RoboTwin path for HY-VLA is the native C++ runner. First clone and configure RoboTwin under `eval/sim/robotwin/`:

```bash
bash eval/sim/robotwin/setup_robotwin.sh
```

If downloads need a local proxy:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --proxy http://127.0.0.1:7890
```

Build `vla-server`:

```bash
cmake --build build --target vla-server -j"$(nproc)"
```

Run HY-VLA on RoboTwin with the native client:

```bash
GGML_CUDA_DISABLE_GRAPHS=1 \
eval/sim/robotwin/robotwin_uv/.venv/bin/python \
  eval/client/run_robotwin_native_hy_vla.py \
  --model /path/to/hy_vla_full_q4_K_vlmvisionstable.gguf \
  --tokenizer /path/to/HY-VLA \
  --task-name place_empty_cup \
  --task-config demo_clean \
  --episodes 1 \
  --max-steps 0 \
  --hy-vla-text-layers 32 \
  --hy-vla-vision-layers all \
  --no-hy-vla-cuda-oom-fallback-cpu
```

For reproducible multi-episode evaluation, pass a fixed seed list and `--skip-expert-check`. Detailed setup modes, troubleshooting notes, timing definitions, and the validated fixed-seed command are documented in [`eval/sim/robotwin/README.md`](eval/sim/robotwin/README.md).

## Notes

- Large model/checkpoint/data files are ignored by `.gitignore`.
- CUDA `.run` installers are not part of the source project and should not be
  committed.
- The VLM-only chat example/server path has been removed; the remaining serving
  code is focused on embodied VLA models.
