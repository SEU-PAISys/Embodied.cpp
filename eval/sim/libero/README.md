# LIBERO Evaluation

This directory contains the setup helper and local runtime for LIBERO-based
Embodied.cpp evaluation. The setup script clones the upstream LIBERO checkout
and creates a local Python environment; neither is committed to this repository.

## Setup

Run once from the repository root:

```bash
bash eval/sim/libero/setup_libero.sh
```

The script creates:

```text
eval/sim/libero/LIBERO/
eval/sim/libero/libero_uv/.venv/
```

For installation diagnostics and dependency notes, see
[`eval/sim/INSTALL_NOTES.md`](../INSTALL_NOTES.md).

## Run an Evaluation

Start the matching `vla-server` or `wam-lingbot-server` first, then run a
configuration with the local LIBERO Python environment:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_pi05_eval.yaml
```

Available checked-in configurations are:

| Model | Configuration |
|---|---|
| pi0.5 | [`libero_pi05_eval.yaml`](../../conf/libero_pi05_eval.yaml) |
| SmolVLA | [`libero_smolvla_eval.yaml`](../../conf/libero_smolvla_eval.yaml) |
| GR00T N1.7 | [`libero_groot_n1_eval.yaml`](../../conf/libero_groot_n1_eval.yaml) |
| LingBot-VA | [`libero_lingbot_va_eval.yaml`](../../conf/libero_lingbot_va_eval.yaml) |
| Xiaomi-Robotics-0 | [`libero_xr0_eval.yaml`](../../conf/libero_xr0_eval.yaml) |
| TurboVLA | [`libero_turbovla_eval.yaml`](../../conf/libero_turbovla_eval.yaml) |
| X-VLA | [`libero_xvla_eval.yaml`](../../conf/libero_xvla_eval.yaml) |

## Suites and Tasks

The direct client accepts these short suite names:

| Suite | Canonical name | Tasks |
|---|---|---|
| `spatial` | `libero_spatial` | 10 |
| `object` | `libero_object` | 10 |
| `goal` | `libero_goal` | 10 |
| `10` | `libero_10` | 10 |
| `long` | `libero_90` | 90 |

Use `--task-id 0..9` for the four 10-task suites and `--task-id 0..89` for
`long`. Command-line options override the selected YAML configuration; for
example, this runs only task 0 from the pi0.5 configuration:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_pi05_eval.yaml \
  --task-id 0
```

## Headless Linux

On a Linux server without a desktop session, select EGL before launching the
client:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_pi05_eval.yaml
```

See [`eval/conf/README.md`](../../conf/README.md) for model-specific evaluation
settings and configuration fields.
