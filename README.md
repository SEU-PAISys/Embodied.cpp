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
serving/             ZeroMQ/protobuf servers for VLA and LingBot world-action APIs
kernels/             custom CUDA kernels used by LingBot-VA when CUDA is enabled
scripts/             conversion, inspection, parity, smoke-test, and setup scripts
tools/               local debug tools
patches/             patches applied to third-party code
third_party/         external code populated by scripts/init_third_party.sh
artifacts/           model graph notes and ONNX audit artifacts
eval/                lightweight client/evaluation helpers kept for this project
```

Supported model entries in the C++ runtime are:

- `pi05`
- `hy_vla`
- `lingbot_va`

## Initialize Third-party Code

Run this once after cloning the repository:

```bash
./scripts/init_third_party.sh
```

The script creates and populates `third_party/`:

- `third_party/llama.cpp`: required by the build
- `third_party/vla.cpp`: original upstream reference repository

It also applies:

```text
patches/llama.cpp-vla.patch
```

to `third_party/llama.cpp`.

The default refs can be overridden:

```bash
LLAMA_REF=846262d7875dcabf502a150fa3d7b9c770dde7eb ./scripts/init_third_party.sh
VLA_REF=main ./scripts/init_third_party.sh
```

`third_party/llama.cpp/` and `third_party/vla.cpp/` are ignored by Git, so they
can be deleted and recreated by the init script.

## Build

Generic CPU build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vla-server lingbot-world-server hy-vla-direct-debug -j
```

For LingBot-VA LIBERO evaluation, use a CUDA build. LingBot image input uses the
video-to-latent VAE bridge, which depends on the LingBot CUDA kernel path being
compiled into `vla_core`.

On this machine the working configuration is:

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

- `CMAKE_CUDA_ARCHITECTURES=86` matches the RTX 3070 Ti Laptop GPU. Use the
  matching architecture for other GPUs.
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

## Model Conversion And Inspection

Conversion scripts live in `scripts/`:

- `convert_pi05_to_gguf.py`
- `convert_pi05_mmproj_to_gguf.py`
- `convert_hy_vla_to_gguf.py`
- `convert_lingbot_va_to_gguf.py`

Inspection and parity scripts are also under `scripts/`; they are intended for
debugging conversion correctness and matching Python reference behavior.

## LIBERO / LIBERO-LONG Evaluation

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

The sweep helpers also accept `-s libero_long` and optional task ranges:

```bash
bash eval/run_libero_client.sh -s libero_long -T 0-2 -m pi0 -a tcp://<server-host>:5555
python eval/collect_libero_results.py --suite libero_long
```

## RobotWin Evaluation For HY-VLA

The RobotWin evaluation code in this repository is an adapter/client layer. It
does not include RobotWin itself.

You cannot run RobotWin evaluation unless RobotWin is installed or otherwise
available in the Python environment. The evaluation flow is:

```text
RobotWin environment observation
  -> eval/utils/sim_adapters/robotwin.py
  -> eval/client/run_robotwin_eval.py
  -> vla-server loaded with HY-VLA
  -> action
  -> RobotWin env.step(action)
```

This differs from the LIBERO setup in this backup tree. LIBERO has project-side
setup scripts and an environment directory under:

```text
eval/sim/libero/
```

RobotWin is not vendored here, so the minimum requirements are:

- RobotWin Python package or source tree is importable.
- RobotWin gymnasium environments are registered.
- The target RobotWin env id is known.
- HY-VLA tokenizer/checkpoint path is available to the client.
- `vla-server` is already running with the RobotWin-trained HY-VLA GGUF.

To clone and configure RobotWin under `eval/sim/robotwin/`, run:

```bash
bash eval/sim/robotwin/setup_robotwin.sh
```

This clones the upstream RobotWin/RoboTwin repository, creates a local
`eval/sim/robotwin/robotwin_uv/.venv` environment with `uv`, installs the
evaluation runtime requirements, applies the small upstream sapien/mplib runtime
patches, and downloads simulation assets. It does not run the
training/planning-heavy full upstream installer by default.

The default RobotWin setup now installs CUDA PyTorch and RoboTwin's curobo
planner dependency, which are required by the RoboTwin environment import path.
It still skips training-only extras such as pytorch3d. If you explicitly need
the complete upstream install, use:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --mode full
```

If you need to use an existing conda environment instead of the local uv venv:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --env-backend conda --conda-env embodied-cpp
```

If dependency downloads are slow, use the local network proxy explicitly:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --proxy http://127.0.0.1:7890
```

The setup script also auto-detects common local proxy ports such as `7890` when
no proxy environment variable is already set. This matters because the eval
requirements include large wheels such as `torch`, `open3d`, `sapien`, and CUDA
runtime packages.

By default, RobotWin eval setup installs CUDA PyTorch wheels from the PyTorch
CUDA index and then filters `torch`/`torchvision`/`nvidia-*` from the upstream
requirements to avoid a second resolver pass replacing them. On this machine
the CUDA wheels are large but install correctly through the local proxy. To make
a lightweight CPU-only environment instead, skip curobo and request CPU torch:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --torch-backend cpu --no-curobo
```

The setup also filters optional non-rollout packages such as `azure`, `wandb`,
and `openai` by default. `open3d` is not filtered because RoboTwin imports it
through the environment utility modules. To install the optional packages
anyway:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --with-optional
```

Observed setup notes on this machine:

- The script detected and used the local proxy at `http://127.0.0.1:7890`.
- Eval/runtime Python dependencies installed successfully in
  `eval/sim/robotwin/robotwin_uv/.venv`.
- `sapien==3.0.0b1` imports `pkg_resources`, so the script pins
  `setuptools<81`.
- RoboTwin assets are present after retry:
  `assets/background_texture`, `assets/embodiments`, and `assets/objects`.
- `open3d==0.19.0` was installed explicitly; without it, `import envs` fails.
- CUDA PyTorch is installed and visible to Python:
  `torch 2.4.1+cu121`, `torch.cuda.is_available() == True`.
- curobo is installed from RoboTwin's expected `v0.7.8` source checkout under
  `eval/sim/robotwin/RoboTwin/envs/curobo`.
- Full RoboTwin environment import now reaches `Base_Task` successfully.
  RoboTwin still prints `missing pytorch3d`, but this did not block importing
  `envs._base_task` or a sample task module.
- During earlier attempts, Hugging Face/Xet downloads from
  `cas-bridge.xethub.hf.co` timed out or reset while fetching
  `background_texture.zip`, `embodiments.zip`, and `objects.zip`.
- The upstream RoboTwin asset script can continue after a failed download, so
  our wrapper now validates that `assets/embodiments`, `assets/objects`, and
  `assets/background_texture` exist before reporting success.
- If Conda activation fails with
  `QT_XCB_GL_INTEGRATION: unbound variable`, prefer the default local `uv`
  environment. The setup script contains a workaround for strict shell mode,
  but using the local venv avoids Conda activation hooks entirely.
- If an install is interrupted halfway through, remove
  `eval/sim/robotwin/robotwin_uv/` and rerun the setup script. Package cache can
  be cleaned with `uv cache clean` if a large wheel download was interrupted.
- CUDA PyTorch and curobo are required by the default RoboTwin evaluation path.
  A lightweight CPU torch environment is useful only for import/debug checks,
  not for the full HY-VLA RoboTwin rollout.

If only the Python environment is needed, skip assets:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --no-assets
```

If the Python environment already exists and only assets should be retried:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --no-install
```

If a future partial download is intentional and the target task is known not to
need the missing assets, the strict asset check can be bypassed:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --no-install --allow-partial-assets
```

Pre-collected trajectory datasets are large and are opt-in:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --download-dataset
```

Start the server on the model machine:

```bash
./build/vla-server --bind tcp://*:5555 /path/to/hy_vla_robotwin.gguf
```

Run the RobotWin client from the RobotWin Python environment:

```bash
bash eval/run_robotwin_hy_vla_client.sh \
  -e <robotwin_gym_env_id> \
  -t /path/to/hy_vla_checkpoint_or_tokenizer \
  -a tcp://<server-host>:5555 \
  -m <robotwin_registration_module>
```

If RobotWin is cloned locally but not importable, add it to Python path:

```bash
bash eval/run_robotwin_hy_vla_client.sh \
  -e <robotwin_gym_env_id> \
  -t /path/to/hy_vla_checkpoint_or_tokenizer \
  -p /path/to/RobotWin \
  -m <robotwin_registration_module>
```

The adapter defaults to HY-VLA RobotWin dimensions:

```text
state_dim=20
action_dim=20
image_size=224
```

If the observation keys in your RobotWin version differ from the defaults,
override them:

```bash
FRONT_KEY=observation.images.front \
WRIST_KEY=observation.images.wrist \
STATE_KEY=observation.state \
TASK_KEY=instruction \
bash eval/run_robotwin_hy_vla_client.sh \
  -e <robotwin_gym_env_id> \
  -t /path/to/hy_vla_checkpoint_or_tokenizer \
  -m <robotwin_registration_module>
```

Current default observation key candidates include:

- front image: `images.front`, `images.image`, `observation.images.front`,
  `observation.images.image`, `pixels.image`, `front_image`, `image`
- wrist image: `images.wrist`, `images.image2`, `observation.images.wrist`,
  `observation.images.image2`, `pixels.image2`, `wrist_image`, `image2`
- state: `observation.state`, `state`, `robot_state`, `agent_pos`, `qpos`
- task text: `task_description`, `instruction`, `language_instruction`, `task`

### Native RoboTwin HY-VLA C++ Runner

For local HY-VLA RoboTwin deployment, prefer the native runner when the RoboTwin
checkout exists under `eval/sim/robotwin/RoboTwin`. It follows the original
HY-VLA Python deployment protocol more closely than the generic gym-style
adapter:

- three camera views: head, left wrist, right wrist;
- EE state/action protocol, with `TASK_ENV.take_action(..., action_type="ee")`;
- `rel_abs` action decoding, `action_chunk_size=20`, `exc_action_size=7`,
  `img_history_size=6`, `img_history_interval=5` from the original HY-VLA
  RoboTwin config;
- full HY-VLA server defaults: `VLA_HY_VLA_TEXT_LAYERS=32`,
  `VLA_HY_VLA_VISION_LAYERS=all`, CUDA vision frontend enabled, and
  `VLA_HY_VLA_VIDEO_HISTORY=6` when image history is enabled;
- `max_steps=0`, meaning use RoboTwin's task `step_lim`.

Build `vla-server` first:

```bash
cmake --build build --target vla-server -j"$(nproc)"
```

Run one full `place_empty_cup/demo_clean` episode with stable timing stats:

```bash
GGML_CUDA_DISABLE_GRAPHS=1 \
eval/sim/robotwin/robotwin_uv/.venv/bin/python \
  eval/client/run_robotwin_native_hy_vla.py \
  --model /home/xuling/robotic_dataset/models/hy_vla_full_q4_K_vlmvisionstable.gguf \
  --tokenizer /home/xuling/robotic_dataset/HY-VLA \
  --episodes 1 \
  --max-steps 0 \
  --output-dir outputs/robotwin_hy_vla_native_place_empty_cup_aligned \
  --hy-vla-text-layers 32 \
  --hy-vla-vision-layers all \
  --no-hy-vla-vision-cpu-sideload \
  --no-hy-vla-cuda-oom-fallback-cpu
```

The native runner sets `GGML_CUDA_DISABLE_GRAPHS=1` for the spawned
`vla-server` by default. This is intentional on the local 8GB RTX 3070 Ti
Laptop GPU: full HY-VLA vision plus the Q4_K resident model and
RoboTwin/curobo leave little VRAM headroom. With CUDA graph replay enabled,
ggml can fail during `cudaGraphInstantiate` because CUDA graphs need extra
instantiation memory. Disabling CUDA graphs uses the normal CUDA kernel path:
it is slower than graph replay, but it lets the full CUDA vision deployment run
stably on this machine.

If you start `vla-server` manually or use another wrapper, keep the same setting:

```bash
export GGML_CUDA_DISABLE_GRAPHS=1
```

Only override it, for example with `GGML_CUDA_DISABLE_GRAPHS=0`, on GPUs with
enough free VRAM after confirming repeated forward passes do not hit CUDA graph
OOM.

For multi-episode RoboTwin runs, there are two supported modes:

- Default mode: the runner asks the RoboTwin expert policy to find
  expert-solvable seeds and language instructions.
- Fixed-seed mode: pass a known seed list and `--skip-expert-check`. This is
  preferred for reproducing official Python results or for avoiding expert
  planner work while a resident HY-VLA `vla-server` is using the same GPU.

Keeping expert planning separate from model serving matters on 8GB GPUs because
Curobo/MAGMA planner work can compete with the resident HY-VLA server for VRAM.
The final validated RoboTwin comparison uses fixed Python reference seeds.

Timing columns are averaged after warmup:

```text
warmup_env_steps=7
warmup_forward_calls=1
```

The summary table uses these definitions:

```text
inf (ms)  = one server-side model forward latency after warmup
step (ms) = inf / na, the amortized per-action inference latency
VRAM      = sampled peak GPU memory above the runner baseline
```

On this machine, the aligned local run wrote:

```text
outputs/robotwin_hy_vla_native_place_empty_cup_aligned/summary.md
outputs/robotwin_hy_vla_native_place_empty_cup_aligned/summary.csv
outputs/robotwin_hy_vla_native_place_empty_cup_aligned/summary.json
outputs/robotwin_hy_vla_native_place_empty_cup_aligned/vla_server.log
```

Current validated fixed-seed full CUDA vision result, measured with CUDA graphs
disabled, using the same 20 successful seeds from the official Python RoboTwin
run:

```md
| Model | Backbone | na | SR (%) | step (ms) | inf (ms) | vision (ms) | action inf (ms) | VRAM (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HY-VLA-CPP | Hunyuan-VL / HY-VLA | 20 | 100.0 [83.9, 100.0] | 67.0 | 1340.3 | 691.7 | 598.9 | 6850 |
```

The command pattern is:

```bash
SEEDS=100000,100001,100002,100004,100005,100007,100008,100009,100010,100011,100012,100013,100014,100015,100016,100017,100018,100019,100020,100021

GGML_CUDA_DISABLE_GRAPHS=1 \
eval/sim/robotwin/robotwin_uv/.venv/bin/python \
  eval/client/run_robotwin_native_hy_vla.py \
  --model /home/xuling/robotic_dataset/models/hy_vla_full_q4_K_vlmvisionstable.gguf \
  --tokenizer /home/xuling/robotic_dataset/HY-VLA \
  --task-name place_empty_cup \
  --task-config demo_clean \
  --seed-list "$SEEDS" \
  --episodes 20 \
  --skip-expert-check \
  --max-steps 120 \
  --output-dir outputs/robotwin_hy_vla_q4k_cpp_fixedseeds_20ep_noexpert \
  --start-server-after-env \
  --addr tcp://127.0.0.1:5568 \
  --bind tcp://*:5568 \
  --hy-vla-text-layers 32 \
  --hy-vla-vision-layers all \
  --no-hy-vla-cuda-oom-fallback-cpu \
  --action-noise-mode torch_cuda_seed
```

The final output from the validated run is stored under:

```text
outputs/robotwin_hy_vla_q4k_cpp_fixedseeds_20ep_noexpert_20260623_221232/
```

The earlier 20s-class inference delay was caused by the CPU vision sidecar
path. Moving the full vision frontend to CUDA brings the server-side forward
to about 1.3s on this laptop GPU for the validated 20-episode run. In the
current full CUDA path, the remaining cost is split mainly between the MEM
video encoder (`vision`) and action denoising (`action inf`).

The server reports `chunk_size=40` because the released HY-VLA RoboTwin
checkpoint stores rel/abs action tokens; the effective RobotWin action chunk
reported as `na` is the original `action_chunk_size=20`.

Do not re-enable CPU vision sidecar for timing unless debugging memory pressure.
Do not enable CUDA graphs on this 8GB setup until allocator/VRAM planning has
been improved. A future optimization path is to make the MEM video attention
numerically match the Python reference with fewer CUDA launches, or replace it
with a dedicated CUDA kernel.

For Python/C++ parity against a dumped Python prefix, remember that the standard
protobuf request carries a modality mask but not a prefix pad mask. A full
Python prefix containing padded language rows will not compare correctly unless
those rows are compacted away, or explicit prefix-pad-mask support is added.
With compacted sample000 from the remote Python reference, the standard
`vla-server` path reached `MAE=0.00198`, `RMSE=0.00404`, `max_abs=0.0447`
against Python `sample_actions()`.

## Notes

- Large model/checkpoint/data files are ignored by `.gitignore`.
- CUDA `.run` installers are not part of the source project and should not be
  committed.
- The VLM-only chat example/server path has been removed; the remaining serving
  code is focused on embodied VLA models.
