# Simulator Dataset / Environment Install Notes

This note records the practical setup points for the LIBERO and RoboTwin
evaluation environments used by embodied.cpp.

## Common prerequisites

- Install `uv` and make sure it is on `PATH`.
- Keep enough disk space free. A CUDA-enabled Python environment is large:
  - LIBERO env: about 9-10 GB.
- Use Python 3.10. The setup scripts create local `uv` virtual environments:
  - `eval/sim/libero/libero_uv/.venv`
  - `eval/sim/robotwin/robotwin_uv/.venv`
- If downloads look stuck, they are often just fetching large PyTorch/CUDA
  wheels. Packages such as `torch`, `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`,
  and `triton` can take many minutes.

## LIBERO setup

Run:

```bash
bash eval/sim/libero/setup_libero.sh
```

The script clones or reuses `eval/sim/libero/LIBERO`, recreates
`eval/sim/libero/libero_uv/.venv`, installs LIBERO and the pinned runtime
dependencies, then writes a default LIBERO config.

Important fixes / checks:

- The venv directory must be removed before recreating it. If `uv venv` reports
  that `.venv` already exists, remove `eval/sim/libero/libero_uv` and rerun.
- `~/.libero/config.yaml` must exist before non-interactive eval runs.
  Otherwise the first `import libero.libero` prompts for a dataset path and can
  fail with `EOFError`.
- A working default `~/.libero/config.yaml` should point to:

```yaml
assets: <repo>/eval/sim/libero/LIBERO/libero/libero/./assets
bddl_files: <repo>/eval/sim/libero/LIBERO/libero/libero/./bddl_files
benchmark_root: <repo>/eval/sim/libero/LIBERO/libero/libero
datasets: <repo>/eval/sim/libero/LIBERO/libero/libero/../datasets
init_states: <repo>/eval/sim/libero/LIBERO/libero/libero/./init_files
```

- Robosuite may warn that `macros_private.py` is missing. It is safe to create
  it by copying `robosuite/macros.py` to `robosuite/macros_private.py` inside
  the venv site-packages directory, or by running the setup script once in a
  fresh venv.
- `uv pip check` may report version metadata conflicts after the LIBERO install.
  The project pins older `numpy`, `gymnasium`, `pyarrow`, and `torchvision`
  versions for LIBERO compatibility, while some packages pulled by `lerobot`
  declare newer requirements. Treat this as a warning if the LIBERO imports and
  eval entrypoints work.

Useful verification:

```bash
eval/sim/libero/libero_uv/.venv/bin/python - <<'PY'
import torch
import robosuite
from libero.libero import benchmark
bench = benchmark.get_benchmark("libero_object")()
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("robosuite", robosuite.__version__)
print("tasks", bench.n_tasks)
PY

eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py --help
```

Expected result includes `tasks 10`.

LIBERO-LONG note:

- Upstream LIBERO names the 90-task long suite `libero_90`.
- This repository also accepts `libero_long` and `libero-long` as aliases in
  the direct client.
- To verify that the installed LIBERO checkout exposes the long suite:

```bash
eval/sim/libero/libero_uv/.venv/bin/python - <<'PY'
from libero.libero import benchmark
bench = benchmark.get_benchmark("libero_90")()
print("tasks", bench.n_tasks)
PY
```

Expected result includes `tasks 90`.

Example direct smoke test:

```bash
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --arch lingbot_va \
  --libero-suite long \
  --task-id 0 \
  --n-episodes 1 \
  --vla-addr tcp://<server-host>:5555 \
  --tokenizer /path/to/lingbot-va-tokenizer
```

## Running with the environments

Activate LIBERO:

```bash
source eval/sim/libero/libero_uv/.venv/bin/activate
```

Activate RoboTwin:

```bash
source eval/sim/robotwin/robotwin_uv/.venv/bin/activate
```

For actual evaluation, start the matching model server separately
and then run the relevant client script from the corresponding venv.

## RobotWin / HY-VLA setup

RobotWin is not vendored in this repository. Install or clone RobotWin in its
own Python environment, then run the HY-VLA client against an already-running
`vla-hy-vla-server`.

The helper script for a local setup is:

```bash
bash eval/sim/robotwin/setup_robotwin.sh
```

It clones the upstream RoboTwin repository into `eval/sim/robotwin/RoboTwin`,
creates `eval/sim/robotwin/robotwin_uv/.venv` with `uv`, installs the
evaluation/runtime requirements from `script/requirements.txt`, applies the
small upstream sapien/mplib runtime patches, and downloads simulation assets
through `script/_download_assets.sh`.

This mirrors the LIBERO setup pattern: each simulator gets its own
source checkout and its own local virtual environment under `eval/sim/<name>/`.

The script defaults to eval mode and installs the pieces needed for RoboTwin
simulation import, including CUDA PyTorch and curobo. It still skips
training-only extras such as pytorch3d. If a complete upstream installation is
required:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --mode full
```

To use an existing conda env instead of the local uv venv:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --env-backend conda --conda-env embodied-cpp
```

If package downloads are slow, pass the local proxy explicitly:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --proxy http://127.0.0.1:7890
```

The setup script auto-detects common local proxy ports, including `7890`, when
no proxy environment variable is set. This is useful because RobotWin eval
requirements still pull large wheels such as `torch`, `open3d`, `sapien`, and
CUDA runtime packages.

The default RobotWin setup uses CUDA PyTorch wheels from the PyTorch CUDA index
and filters `torch`/`torchvision`/`nvidia-*` from upstream requirements after
that explicit installation. This avoids a second resolver pass replacing the
chosen CUDA wheels. If only a lightweight CPU environment is required:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --torch-backend cpu --no-curobo
```

The default eval setup also filters optional non-rollout packages such as
`azure`, `wandb`, and `openai`. `open3d` is required by RoboTwin environment
utilities and is installed as part of the eval requirements. To install the
optional packages:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --with-optional
```

Observed setup status on this machine:

- Local proxy auto-detection found `http://127.0.0.1:7890`.
- Dependency installation completed in
  `eval/sim/robotwin/robotwin_uv/.venv`.
- `sapien==3.0.0b1` requires the deprecated `pkg_resources` module, so the
  helper installs `setuptools<81`.
- Asset download succeeded after retry. The required directories are present:
  `assets/background_texture`, `assets/embodiments`, and `assets/objects`.
- `open3d==0.19.0` was installed explicitly. It is required for importing the
  RoboTwin `envs` package.
- CUDA PyTorch is installed and working:
  `torch 2.4.1+cu121`, `torch.cuda.is_available() == True`.
- curobo is installed from RoboTwin's expected `v0.7.8` source checkout under
  `eval/sim/robotwin/RoboTwin/envs/curobo`.
- `envs._base_task` and a sample RoboTwin task module import successfully.
  RoboTwin still prints `missing pytorch3d`, but that warning did not block the
  checked environment import path.
- During earlier attempts, Hugging Face/Xet requests to
  `cas-bridge.xethub.hf.co` timed out or reset while downloading
  `background_texture.zip`, `embodiments.zip`, and `objects.zip`.
- The wrapper now avoids the upstream false-success case by checking for
  `assets/embodiments`, `assets/objects`, and `assets/background_texture`.
- The shell can fail inside Conda's Qt activation hook with
  `QT_XCB_GL_INTEGRATION: unbound variable` when scripts use strict
  `set -u`. The RobotWin setup script temporarily disables `nounset` while
  activating Conda environments. Prefer the local `uv` environment if the
  Conda activation stack is unstable.
- If an interrupted install leaves a half-created environment, remove
  `eval/sim/robotwin/robotwin_uv/` and rerun the setup script. To clean package
  cache after an interrupted large wheel download, use `uv cache clean` or
  remove the relevant temporary files under the user cache directory.
- A CPU-only or incomplete PyTorch install is not enough for the default
  RoboTwin path. RoboTwin imports curobo/planning utilities through the task
  setup path, and the HY-VLA Python reference uses CUDA tensors during action
  sampling. The default setup therefore installs CUDA PyTorch and curobo even
  though this repository only runs inference.
- `open3d` is needed for RoboTwin environment utilities. It looks optional from
  a pure inference perspective, but `import envs` can fail without it.

Useful retry modes:

```bash
# Configure only the Python environment, without assets.
bash eval/sim/robotwin/setup_robotwin.sh --no-assets

# Reuse the Python environment and retry only assets/dataset steps.
bash eval/sim/robotwin/setup_robotwin.sh --no-install

# Continue despite missing assets, only if the target env does not need them.
bash eval/sim/robotwin/setup_robotwin.sh --no-install --allow-partial-assets
```

The pre-collected trajectory dataset can be large, so it is not downloaded by
default. To request it:

```bash
bash eval/sim/robotwin/setup_robotwin.sh --download-dataset
```

Start the server with the HY-VLA GGUF on the machine that owns the model:

```bash
./build/vla-hy-vla-server --bind tcp://*:5555 /path/to/hy_vla_robotwin.gguf
```

Then run the client from the RobotWin Python environment:

```bash
bash eval/run_robotwin_hy_vla_client.sh \
  -e <robotwin_gym_env_id> \
  -t /path/to/hy_vla_checkpoint_or_tokenizer \
  -a tcp://<server-host>:5555 \
  -m <robotwin_registration_module>
```

If the RobotWin package is not already importable, add its source path:

```bash
bash eval/run_robotwin_hy_vla_client.sh \
  -e <robotwin_gym_env_id> \
  -t /path/to/hy_vla_checkpoint_or_tokenizer \
  -p /path/to/RobotWin \
  -m <robotwin_registration_module>
```

The RobotWin adapter defaults to HY-VLA's RobotWin dimensions:

```text
state_dim=20
action_dim=20
image_size=224
```

If your RobotWin observation keys differ from the defaults, override them with
environment variables:

```bash
FRONT_KEY=observation.images.front \
WRIST_KEY=observation.images.wrist \
STATE_KEY=observation.state \
TASK_KEY=instruction \
bash eval/run_robotwin_hy_vla_client.sh -e <env_id> -t <tokenizer> -m <module>
```

The client accepts common observation layouts by default, including
`images.front`, `images.wrist`, `observation.images.front`,
`observation.images.wrist`, `pixels.image`, `pixels.image2`, `state`, and
`observation.state`.

Native HY-VLA C++ RoboTwin deployment:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMODEL_BUILD_VLA_HY_VLA=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=<your-arch>
cmake --build build --target vla-hy-vla-server -j"$(nproc)"

eval/sim/robotwin/robotwin_uv/.venv/bin/python \
  eval/client/run_robotwin_native_hy_vla.py \
  --model /home/xuling/robotic_dataset/models/hy_vla_full_q4_K_vlmvisionstable.gguf \
  --tokenizer /home/xuling/robotic_dataset/HY-VLA \
  --episodes 1 \
  --max-steps 0 \
  --output-dir outputs/robotwin_hy_vla_native_place_empty_cup_aligned
```

This runner directly uses the RoboTwin task classes and follows the original
HY-VLA Python deployment protocol where possible: three cameras, raw EE
state/action, `action_type="ee"`, `rel_abs` action decoding,
`action_chunk_size=20`, `exc_action_size=7`, `img_history_size=6`, and
`img_history_interval=5`. Stable timing excludes `warmup_env_steps=7` and
`warmup_forward_calls=1`. The validated C++ path runs the full CUDA vision
frontend, including the HY-VLA video-history/MEM path, rather than the older
CPU vision sidecar path.

Important runtime notes:

- For this 8GB RTX 3070 Ti Laptop setup, keep CUDA graphs disabled:

```bash
export GGML_CUDA_DISABLE_GRAPHS=1
```

The native runner sets this for the spawned `vla-hy-vla-server` by default. Full
HY-VLA vision plus RoboTwin/curobo leaves little VRAM headroom, and CUDA graph
instantiation can fail or become unstable. Disable it long-term unless a larger
GPU has been verified with repeated forward passes.

- `--max-steps 0` means "use the RoboTwin task's own `step_lim`"; it does not
  mean stop after one step.
- The original Python evaluation first uses an expert policy to find
  solvable/stable seeds. Keeping a resident `vla-hy-vla-server` on an 8GB GPU while
  also running expert/curobo seed search can make the planner path fail. For
  exact reproducibility, use a known successful seed list from the Python
  reference and pass `--skip-expert-check`.
- Use the same action noise mode as the Python reference for parity-sensitive
  runs:

```bash
--action-noise-mode torch_cuda_seed
```

- The summary table uses the following timing definitions:
  - `inf (ms)`: one server-side model forward latency after warmup.
  - `step (ms)`: wall-clock time per environment step, amortized over action
    chunk replay.
  - `vision (ms)` and `action inf (ms)`: optional internal breakdowns.

Example fixed-seed 20-episode command:

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

Observed final local result for `place_empty_cup/demo_clean`, using the 20
known-good Python reference seeds and skipping expert check:

```md
| Model | Backbone | na | SR (%) | step (ms) | inf (ms) | vision (ms) | action inf (ms) | VRAM (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HY-VLA-CPP | Hunyuan-VL / HY-VLA | 20 | 100.0 [83.9, 100.0] | 735.9 | 1340.3 | 691.7 | 598.9 | 6850 |
```

Output files:

```text
outputs/robotwin_hy_vla_q4k_cpp_fixedseeds_20ep_noexpert_20260623_221232/summary.md
outputs/robotwin_hy_vla_q4k_cpp_fixedseeds_20ep_noexpert_20260623_221232/summary.csv
outputs/robotwin_hy_vla_q4k_cpp_fixedseeds_20ep_noexpert_20260623_221232/summary.json
outputs/robotwin_hy_vla_q4k_cpp_fixedseeds_20ep_noexpert_20260623_221232/vla_server.log
```
