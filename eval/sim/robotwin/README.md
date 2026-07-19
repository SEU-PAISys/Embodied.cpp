# RoboTwin / HY-VLA Evaluation

This directory contains the setup helper for running HY-VLA C++ inference on
RoboTwin. RoboTwin itself is not vendored in the repository; it is cloned by the
setup script into this directory when needed.

## Setup

Run from the repository root:

```bash
bash eval/sim/robotwin/setup_robotwin.sh
```

The script clones RoboTwin into `eval/sim/robotwin/RoboTwin`, creates a local
`uv` environment under `eval/sim/robotwin/robotwin_uv/.venv`, installs the
evaluation runtime dependencies, applies small SAPIEN/MPLib runtime patches,
and downloads simulation assets.

If downloads are slow, pass the proxy explicitly:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --proxy http://127.0.0.1:7890
```

Useful setup modes:

```bash
# Configure only the Python environment, without assets.
bash eval/sim/robotwin/setup_robotwin.sh --no-assets

# Reuse the Python environment and retry only assets.
bash eval/sim/robotwin/setup_robotwin.sh --no-install

# Use an existing conda environment instead of the local uv venv.
bash eval/sim/robotwin/setup_robotwin.sh --env-backend conda --conda-env embodied-cpp

# Install the complete upstream environment, including training-heavy extras.
bash eval/sim/robotwin/setup_robotwin.sh --mode full
```

The default setup installs CUDA PyTorch and RoboTwin's curobo dependency. These
are required by the normal RoboTwin task/expert path. A CPU-only environment is
only useful for lightweight import checks:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --torch-backend cpu --no-curobo
```

## Native HY-VLA C++ Runner

For HY-VLA, prefer the native runner:

```text
eval/client/run_robotwin_eval.py
```

It follows the original HY-VLA RoboTwin deployment protocol more closely than
the generic gym-style client:

- three camera views: head, left wrist, right wrist;
- EE action protocol with `TASK_ENV.take_action(..., action_type="ee")`;
- `rel_abs` action decoding;
- `action_chunk_size=20`, `exc_action_size=7`, `img_history_size=6`,
  `img_history_interval=5`;
- full HY-VLA server defaults: `VLA_HY_VLA_TEXT_LAYERS=32`,
  `VLA_HY_VLA_VISION_LAYERS=all`, CUDA vision frontend, and video history.

Build the server first:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMODEL_BUILD_VLA_HY_VLA=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=<your-arch>
cmake --build build --target vla-hy-vla-server -j"$(nproc)"
```

Run one `place_empty_cup/demo_clean` episode:

```bash
GGML_CUDA_DISABLE_GRAPHS=1 \
eval/sim/robotwin/robotwin_uv/.venv/bin/python \
  eval/client/run_robotwin_eval.py \
  --model /path/to/hy_vla_full_q4_K_vlmvisionstable.gguf \
  --tokenizer /path/to/HY-VLA \
  --task-name place_empty_cup \
  --task-config demo_clean \
  --episodes 1 \
  --max-steps 0 \
  --output-dir outputs/robotwin_hy_vla_place_empty_cup_1ep \
  --hy-vla-text-layers 32 \
  --hy-vla-vision-layers all \
  --no-hy-vla-cuda-oom-fallback-cpu
```

`--max-steps 0` means "use RoboTwin's task `step_lim`"; it does not mean one
step.

## Reproducible Fixed-Seed Run

The original Python evaluation first uses an expert policy to find stable
seeds. On 8GB GPUs, running the resident HY-VLA server and the expert planner
at the same time can cause memory pressure. For reproducible comparison, use a
known successful Python seed list and skip the expert check:

```bash
SEEDS=100000,100001,100002,100004,100005,100007,100008,100009,100010,100011,100012,100013,100014,100015,100016,100017,100018,100019,100020,100021

GGML_CUDA_DISABLE_GRAPHS=1 \
eval/sim/robotwin/robotwin_uv/.venv/bin/python \
  eval/client/run_robotwin_eval.py \
  --model /path/to/hy_vla_full_q4_K_vlmvisionstable.gguf \
  --tokenizer /path/to/HY-VLA \
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

Validated local result:

```md
| Model | Backbone | na | SR (%) | step (ms) | inf (ms) | vision (ms) | action inf (ms) | VRAM (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HY-VLA-CPP | Hunyuan-VL / HY-VLA | 20 | 100.0 [83.9, 100.0] | 735.9 | 1340.3 | 691.7 | 598.9 | 6850 |
```

Timing definitions:

```text
inf (ms)  = one server-side model forward latency after warmup
step (ms) = wall-clock time per environment step, amortized over action chunk replay
VRAM      = sampled peak GPU memory above the runner baseline
```

## Runtime Notes

Keep CUDA graphs disabled on the local 8GB RTX 3070 Ti Laptop GPU:

```bash
export GGML_CUDA_DISABLE_GRAPHS=1
```

Full HY-VLA vision plus the resident Q4_K model and RoboTwin/curobo leave little
VRAM headroom. CUDA graph instantiation can fail or become unstable because it
needs extra memory. Only re-enable CUDA graphs on GPUs with enough free VRAM
after repeated forward passes have been verified.

The server may report `chunk_size=40` because the released HY-VLA RoboTwin
checkpoint stores rel/abs action tokens. The effective RoboTwin action chunk
reported as `na` is the original `action_chunk_size=20`.

For Python/C++ parity against a dumped Python prefix, the standard protobuf
request carries a modality mask but not a prefix pad mask. A full Python prefix
containing padded language rows will not compare correctly unless those rows are
compacted away, or explicit prefix-pad-mask support is added.

## Generic Adapter

The repository also keeps a generic adapter path:

```text
eval/run_robotwin_hy_vla_client.sh
eval/client/run_robotwin_eval_simple.py
adapter/sim/robotwin.py
```

Use it only when you need to connect to a registered gym-style RobotWin
environment manually. For HY-VLA reproduction, the native runner above is the
recommended path.
