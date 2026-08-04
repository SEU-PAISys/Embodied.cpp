# SmolVLA validation

For a detailed Chinese implementation and validation report, see
[SMOLVLA_TECHNICAL_REPORT_ZH.md](SMOLVLA_TECHNICAL_REPORT_ZH.md).

This document records the validation protocol for the Embodied.cpp SmolVLA
runtime. Generated videos, per-episode JSON, logs, and model files remain local
under `outputs/` and `checkpoints/`; they are intentionally not committed.

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
- NVIDIA driver: 576.88
- Runtime: WSL2, Linux 5.15.146.1
- C++ toolchain: GCC 13.3.0, CMake 3.28.3, Ninja 1.11.1
- LIBERO client: Python 3.10.20
- Official LeRobot evaluator: Python 3.12.3
- CUDA target: compute capability 8.9

## Full acceptance matrix

The completed acceptance run used all ten tasks in each suite and ten episodes
per task (400 episodes per implementation, 800 total). Both implementations
used relative control, one replayed action per request, and ten flow-matching
steps.

| Suite | Embodied.cpp | Official LeRobot | Delta |
|---|---:|---:|---:|
| LIBERO-Spatial | 79/100 | 69/100 | +10 pp |
| LIBERO-Object | 86/100 | 87/100 | -1 pp |
| LIBERO-Goal | 80/100 | 76/100 | +4 pp |
| LIBERO-10 | 39/100 | 33/100 | +6 pp |
| **Total** | **284/400 (71.0%)** | **265/400 (66.25%)** | **+4.75 pp** |

This full run established functional coverage and baseline parity. It predates
the client-side per-suite action-noise seed derivation added during final PR
hardening, so it should not be used as a bit-for-bit reproducibility artifact.
The smaller protocol below is rerun after seed changes.

## PR smoke protocol

The PR smoke protocol selects task 0 from each of `libero_spatial`,
`libero_object`, `libero_goal`, and `libero_10`, with three episodes per task.
It runs 12 C++ and 12 official episodes. The generated report distinguishes
`selected_protocol_complete` from the four full-benchmark acceptance flags.

| Suite | Embodied.cpp | Official LeRobot | Delta |
|---|---:|---:|---:|
| LIBERO-Spatial task 0 | 1/3 | 1/3 | 0 pp |
| LIBERO-Object task 0 | 3/3 | 2/3 | +33.3 pp |
| LIBERO-Goal task 0 | 3/3 | 2/3 | +33.3 pp |
| LIBERO-10 task 0 | 0/3 | 0/3 | 0 pp |

The selected protocol completed and the C++ result was within the five-point
tolerance of the locally executed official baseline in every selected suite.
Three episodes are only a smoke sample; these rates must not be interpreted as
suite-level estimates or compared directly with published full-suite scores.

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_smolvla_acceptance.py \
  --run all --task-ids 0 --episodes 3 \
  --seed 1000 --noise-seed 1000
```

Each C++ result records the environment seed, base action-noise seed, and the
derived seed for every episode. Resume checks reject results from a different
model, suite, task, seed, noise seed, episode count, or action replay horizon.

## Automated checks

The lightweight regression suite covers deterministic action-noise generation,
suite/task/episode seed separation, C++ and official resume detection, report
semantics, Wilson intervals, and policy/mmproj converter dtype handling.

```bash
PYTHONPATH=third_party/llama.cpp/gguf-py \
  python -m unittest discover -s tests -v
```

Clean builds are checked in three independent build directories:

1. CPU-only SmolVLA server.
2. CUDA SmolVLA server for compute capability 8.9.
3. CUDA all-model build containing pi0.5, SmolVLA, HY-VLA, GR00T N1, and
   LingBot-VA server targets.

The only compiler warning observed is the pre-existing deprecated ZeroMQ
`setsockopt` call in `serving/server.cpp`.

## Numerical parity

Numerical parity uses the same two 512x512 images, 8-D state, 48 language
tokens and attention mask, and a fixed 50x32 action-noise tensor for both the
official PyTorch policy and the C++ server. The comparison reports maximum and
mean absolute action error and rejects NaN or infinity. This isolates model
math from simulator and random-number-generator differences.

The final fixed-input comparison produced maximum absolute error 0.027861,
mean absolute error 0.001906, and RMSE 0.003258 over the 50x7 real action
dimensions. Repeating the same C++ request produced zero maximum absolute
difference. Neither output contained NaN or infinity.

```bash
python scripts/check_smolvla_parity.py \
  --inputs /path/to/inputs.npz \
  --reference /path/to/python_action.npy \
  --output outputs/smolvla_parity.json
```
