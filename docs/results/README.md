# Evaluation results

Results use the [common benchmark standard](../../eval/VLA_BENCHMARK_STANDARD.md).
Model storage type is not necessarily compute precision. Compare Python and
C++ only with identified weights, matching inputs and timing boundaries;
unmeasured phases are `null`/`n/a`, not zero.

| Model | Validation and evidence |
|---|---|
| Xiaomi-Robotics-0 | [Validation report](../../eval/XR0_LIBERO_VALIDATION.md) |
| TurboVLA | [Validation report](../../eval/TURBOVLA_LIBERO_VALIDATION.md) |
| X-VLA | [Validation report](../../eval/XVLA_LIBERO_VALIDATION.md) |

These supplement the existing model results linked from the project
[README](../../README.md), rather than redefine their measurement protocols.
See the [current release audit](release_audit_20260907.md) for the full target
matrix, raw-result hashes and unresolved acceptance checks. Historical
[takeover measurements](takeover_20260903.md) retain their original limitations
and must not be substituted for matched comparisons.

## Public Python/C++ entry point

All three models use `eval/client/run_sim_client_direct.py`, selecting
`--implementation cpp` or `--implementation python`. Both support
`--profile-output`; action replay does not count as a new inference request.
Python XR0/X-VLA use `--hf-dir` plus their model-specific precision options.
TurboVLA Python uses existing local assets:

```bash
python eval/client/run_sim_client_direct.py \
  --arch turbovla --implementation python \
  --turbovla-checkpoint /path/to/turbovla.pth \
  --turbovla-official-root /path/to/official-source \
  --turbovla-bert-path /path/to/bert \
  --turbovla-norm-gguf /path/to/matching.gguf \
  --turbovla-precision fp32 --libero-suite spatial --task-ids 0 1 2 3 4 5 6 7 8 9 \
  --observation-size 256 --n-episodes 10 --seed 42 --no-video \
  --output-dir outputs/turbovla-python-fp32 \
  --profile-output outputs/turbovla-python-fp32/profile.json
```

`fp32` now selects FP32 vision as well as FP32 policy; historical runs that
used BF16 vision autocast are mixed precision, not this new baseline.
Run all four suites in distinct directories, then use the public aggregator
with `--require-full-matrix --episodes-per-task 10`. Coverage passing alone is
not a quality, parity or performance acceptance result.
