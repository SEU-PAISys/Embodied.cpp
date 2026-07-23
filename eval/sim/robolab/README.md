# RoboLab / Cosmos3 Integration Notes

RoboLab is the Isaac Sim based DROID evaluation benchmark used for the
experimental Cosmos3 backend.

There are two deliberately separate paths:

- **Native Embodied.cpp path**: the target implementation.  RoboLab provides
  tasks, cameras, simulator stepping, and metrics; Embodied.cpp provides the
  WAM server and C++ Cosmos3 forward pass.
- **PyTorch baseline path**: a reference and capture path only.  Use it to
  compare official behavior or capture tensors while bringing up C++/CUDA
  operators.  Do not make the native path depend on it.

The native runtime path is:

```text
RoboLab observation
  -> eval/client/run_robolab_eval.py
  -> serving/wam-server.cpp
  -> models/cosmos3.cpp + kernels/cosmos3/
  -> action chunk
  -> RoboLab simulator
```

The PyTorch baseline path is:

```text
RoboLab observation
  -> RoboLab policies/cosmos3/client.py (OpenPI WebSocket)
  -> cosmos-framework action_policy_server_robolab (PyTorch/CUDA)
  -> action chunk
  -> RoboLab simulator
```

## Install RoboLab

```bash
RUN_SMOKE_TEST=0 eval/sim/robolab/setup_robolab.sh
```

Run the optional Isaac Sim smoke test only after explicitly selecting the GPU:

```bash
ROBOLAB_GPU=0 eval/sim/robolab/setup_robolab.sh
```

The checkout and uv cache default to this directory and are git-ignored. Use
`ROBOLAB_DIR` and `UV_CACHE_DIR` to share existing installations.

## Run Native Embodied.cpp Cosmos3

Use the native WAM entry point.  This path must not invoke
`RoboLab/policies/cosmos3/run.py` or `policies/cosmos3/client.py`.

```bash
cd /path/to/Embodied.cpp
python3 eval/client/run_robolab_eval.py \
  --conf eval/conf/robolab_cosmos3_eval.yaml
```

For a shared lab server, keep machine-specific model paths, tokenizer paths,
output directories, and GPU IDs in a git-ignored local config such as
`eval/conf/robolab_cosmos3_eval.local.yaml`.

The config defaults to one RoboLab task and one episode while the native
forward path is being completed:

```yaml
robolab:
  tasks:
    - BananaOnPlateTask
  num_envs: 1
  num_runs: 1
```

The fast transport smoke test does not launch Isaac Sim:

```bash
python3 eval/client/run_robolab_eval.py \
  --conf eval/conf/robolab_cosmos3_eval.yaml \
  --smoke-wam-only
```

## Run PyTorch Baseline

Use the project entry point rather than invoking the external framework
directly.  This is not the native Embodied.cpp runtime.

```bash
COSMOS_FRAMEWORK_DIR=/path/to/cosmos-framework \
BUNDLE_DIR=/path/to/robolab_full_w8 \
POLICY_GPU=0 SIM_GPU=1 \
eval/run_robolab_cosmos3_pytorch.sh
```

`POLICY_GPU` and `SIM_GPU` must be distinct physical GPUs.  The wrapper passes
the selected task list to RoboLab and stores replay data, boundary tensors and
calibration amax values below its run directory.

For a ten-task, one-episode evaluation:

```bash
TASKS="BananaInBowlTask BananaOnPlateTask BagelsOnPlateTask \
AppleAndYogurtInBowlTask BBQSauceInBinTask PlasticBottlesInSquarePailTask \
BowlInBinTask CannedFoodInBinTask ClampInRightBinTask CoffeePotInBinTask" \
NUM_ENVS=1 NUM_RUNS=1 \
COSMOS_FRAMEWORK_DIR=/path/to/cosmos-framework \
BUNDLE_DIR=/path/to/robolab_full_w8 \
POLICY_GPU=0 SIM_GPU=1 \
eval/run_robolab_cosmos3_pytorch.sh
```

The external `cosmos-framework` checkout must include the RoboLab OpenPI server
and `examples/robolab_quant/pipeline.sh` integration.

The wrapper defaults to headless execution, `VIDEO_MODE=none`, and
`RENDERING_TYPE=performance` to avoid GUI/video overhead while preserving the
policy input cameras.  Do not lower camera resolution for benchmark runs unless
you are intentionally running a visual-input ablation.
