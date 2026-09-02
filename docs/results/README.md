# LIBERO Evaluation Reports

Historical LIBERO reports for **TurboVLA**, **Xiaomi-Robotics-0** and
**X-VLA**, using the same public evaluation, profiling and aggregation
entry points as other supported LIBERO models. These archived results are
not a fresh validation of every later commit. TurboVLA's local Python sweep
used a different checkpoint protocol; see its report before comparing rates.

| Model | Params | Scope | Official ref | Ours (best) | Verdict |
|---|---|---|---|---|---|
| [TurboVLA](turbovla_libero.md) | 0.2B | full LIBERO, 400 eps | 97.7% avg | **96.5%** (q8_0) / 96.25% (bf16) | within ~1 pp of official |
| [Xiaomi-Robotics-0](xr0_libero.md) | 4.7B | full LIBERO, 2000 eps/config | 98.7% avg | **98.45%** (bf16) / 98.55% (q4_k) | matches official |
| [X-VLA](xvla_libero.md) | 0.9B | full LIBERO, 400 eps | 99.8% | **99.2%** (bf16) / 99.8% (q8_0) | matches official |

Upstream snapshot at time of writing:

| Upstream | Status |
|---|---|
| SEU-PAISys/Embodied.cpp `main@1dad33f` | base for this integration PR; no later upstream commits at submission time |
| XiaomiRobotics/Xiaomi-Robotics-0 | latest: post-training code open-sourced 2026-04-27 |
| H-EmbodVis/TurboVLA | latest: checkpoints released 2026-07-31 (LIBERO avg 97.7%) |
| 2toinf/X-VLA | accepted to ICLR 2026; natively integrated into LeRobot |

Raw episode logs and videos are kept out of git (`outputs/` is ignored);
the tables in these documents are generated from those logs.

## Running and aggregating

Activate the project LIBERO environment and run commands from the repository
root. Prepare GGUFs using [the conversion instructions](../../scripts/README.md),
then start the matching server. XR0 and X-VLA additionally need the original
HF tokenizer snapshot: pass `--tokenizer /path/to/matching/snapshot`.

The three checked-in YAMLs are object-suite examples (20 episodes/task,
seed 42), not exact reproductions of the archived four-suite reports:

| Model | Archived episodes/task | Suites | Seed recorded in report | Replayed actions |
|---|---:|---|---|---:|
| XR0 | 50 | spatial, object, goal, 10 | 7 for the Python reference; C++ seed not recorded | 10 |
| TurboVLA | 10 | spatial, object, goal, 10 | 7 | 12 |
| X-VLA | 10 | spatial, object, goal, 10 | 7 | 30 |

For example, a new X-VLA run can use the reported episode/seed settings:

```bash
run=xvla-bf16-seed7-rerun1
for suite in spatial object goal 10; do
  python eval/client/run_sim_client_direct.py \
    --conf libero_xvla_eval.yaml --libero-suite "$suite" \
    --n-episodes 10 --seed 7 --tokenizer /path/to/xvla-hf-snapshot \
    --output-dir "outputs/$run" \
    --profile-output "outputs/$run/profiles/$suite.json"
done
python scripts/aggregate_eval_summary.py \
  --outputs outputs --out-dir outputs/reports
```

Use a new run name for each model, weight precision, seed or protocol change;
do not overwrite a previous run. For XR0 choose its YAML and 50 episodes/task;
for TurboVLA choose its YAML, omit `--tokenizer`, and restart the server with
the correct **suite-specific checkpoint** before evaluating each suite.
The C++ XR0 historical seed and complete asset/environment metadata are not
archived, so a new explicitly seeded run is not an exact historical replay.

The runner writes `<run>/<arch>/libero_<suite>/task_N/result.json` and
`summary.txt`. The aggregator groups these as `run:<run>` without inferring
precision, keeps them separate from archived and smoke results, and rejects
duplicate task records. Inspect task coverage, skipped episodes and recorded
protocols before promoting a result into this directory. A generated summary
alone does not certify a full benchmark or numerical parity. Use
`--outputs outputs` (the parent of named runs) to preserve run identity.

The old `run_libero_eval_xr0.py` command is a compatibility wrapper around
this runner; its output now uses the same per-task layout, not the old
standalone `result_<task>.txt` files.

## Performance evidence

The new-model cells in README section 1.2 remain **Pending** until matched
Python/C++ measurements are committed. Previously quoted TurboVLA/XR0
timings and resident-memory values do not have supporting measurements in
these reports. X-VLA's historical timing note is retained in its report but
mixes Python FP32 query time and C++ round-trip time; it is not a BF16 speedup.

Historical per-call measurements were recorded on 2026-08-18 in the
workspace artifact `outputs/eval_20260818_corrected/EVALUATION_REPORT.md`
(raw logs stay out of git). Environment: NVIDIA GeForce RTX 4060 Laptop
(8188 MiB), driver 610.62, WSL2 Ubuntu, Python 3.10.20, PyTorch
2.5.1+cu124, working commit `a548ccb`, seed 7. Measured values:
TurboVLA C++ server-phase 62.0 ms (mean, std 16.4, p50 57.6 / p95 100.0 /
p99 111.9) vs Python/ZMQ round-trip 150.6 ms (std 49.4), whole-card VRAM
~1303 MiB, resident weights 0.36 GiB (XR0/TurboVLA checkpoint hashes are
listed in that report); Xiaomi-Robotics-0 C++ 2170.4 ms vs Python 2200.8 ms
(std ~44-46), whole-card VRAM ~7629 MiB, resident weights 7.29 GiB.
These numbers cannot be turned into the unified `Python → C++ BF16` ratio
the README table requires: the Python timings are ZMQ wall round-trips
while the C++ values are server-phase only, XR0's C++ side ran its vision
tower on the CPU (`VLA_XR0_CLIP_GPU` was unusable due to a CUDA
`ggml_im2col` crash), and no Python-side VRAM baseline exists. Re-running a
controlled benchmark (same machine/dtype/inputs, `bench_models.py` phases,
p50/p95/p99, and process VRAM for both sides) is the path to filling the
Pending cells.

A comparable result must record:

- Code revisions, GPU/CPU, driver/runtime versions, build flags, checkpoint
  hashes, precision and tokenizer snapshot, plus commands and seeds.
- Identical image/state/prompt inputs, resolution, action replay cadence and
  model settings; distinguish fixed-input benchmarks from closed-loop rollouts.
- Warmup and repeated sample counts, mean, standard deviation and p50/p95/p99.
  Compare identical timing boundaries: client round-trip includes preprocessing
  and transport, while server phase timing does not. Record synchronization
  and whether transfers are included for both implementations.
- Process peak/resident VRAM versus whole-device memory versus weight-buffer
  size, with the same metric for Python and C++; do not mix these in a ratio.
- XR0's `VLA_XR0_CLIP_GPU` environment setting (unset defaults to CPU vision).

`bench_models.py` measures client round-trip and server phases separately;
the suite profiler deduplicates queued actions by request sequence and saves
metrics beside each profile JSON. Missing VRAM keeps `table_ready` false:
do not turn unavailable measurements into zero usage or a claimed speedup.
