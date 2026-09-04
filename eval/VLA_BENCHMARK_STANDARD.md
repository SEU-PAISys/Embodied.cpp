# Unified VLA benchmark standard

This document defines when a Python-to-Embodied.cpp result is comparable and
may be promoted to the README performance table. It applies equally to pi0.5,
SmolVLA, GR00T N1.7, HY-VLA, Xiaomi-Robotics-0, TurboVLA, X-VLA, and future VLA
runtimes.

## Claim levels

| Level | Meaning | May enter the README ratio table? |
|---|---|---|
| Published reference | A paper or upstream report gives a normalized result, but this repository does not contain the raw samples and full run manifest | Yes, only when marked as a published reference |
| Reproduced comparison | Python and C++ were measured locally under the complete protocol below | Yes |
| Integration evidence | Build, fixed-input parity, smoke rollout, or one implementation was measured | No |
| Pending | A matching checkpoint, protocol field, or raw evidence is missing | No numeric ratio |

## Performance protocol

Python and C++ must use the same checkpoint identity, input fixture, device,
batch size, model settings, and effective compute precision. Record checkpoint
SHA-256 values and distinguish three independent properties: source checkpoint,
GGUF storage type, and runtime compute type.

Checkpoint identity includes inference metadata, not only tensor names or
payloads. Before a full sweep, compare tokenizer/preprocessing settings,
normalization and per-instruction layouts against the source checkpoint and
the paired baseline. Preserve these fields during quantization. Two GGUFs
with identical tensors but different padding layouts are different inference
configurations; a tensor-name count cannot certify equivalence.

1. Use batch size 1, at least 5 untimed warm-up requests, and at least 100
   timed requests.
2. Time the same deployment boundary on both sides: raw CPU observations to a
   complete CPU action chunk. Include preprocessing, host/device transfers,
   synchronization, and action read-back. Exclude model loading, fixture
   construction, simulator time, and queued-action replay.
3. Preserve every timed sample and report population standard deviation plus
   mean, p50, p95, and p99. Do not combine samples from different runs to make
   a better ratio.
4. Report latency per request and per **generated model action**. For request
   `i`, the latter is `request_latency_i / generated_chunk_size_i`. The existing
   `inf/n_a` value instead divides by replayed environment actions and remains a
   control-loop metric; it is not the paper's per-generated-action metric when
   replay and output chunk sizes differ.
5. Measure GPU memory in a separate untimed phase with the same process-level
   method on both sides. Do not compare PyTorch allocator bytes, model file
   size, resident weight buffers, whole-device usage, and process peak as if
   they were the same quantity.

The LIBERO profiler writes raw latency and memory samples together with
mean/std/p50/p95/p99. `step_ms` is the client control-loop cost, `inf_ms` is the
server request cost, `model_step_ms` is replay-amortized, and
`generated_action_step_ms` divides that server cost by the generated horizon.
It must not be mixed with the full CPU-input-to-CPU-action deployment latency:
normalize the fixed-boundary benchmark's own samples by its output horizon
when reporting deployment cost per generated action.

## Closed-loop success protocol

Success comparisons require the same benchmark version, suite, task set,
episodes per task, environment initial states, control mode, action replay,
model horizon, image resolution, prompt construction, and stochastic-noise
protocol. Record explicit environment and action-noise seeds. Report raw
success counts and 95% Wilson intervals; skipped or aborted episodes are not
successful trials and must remain visible in the evidence.

LIBERO success rates may be compared only within the same LIBERO task scope.
HY-VLA RoboTwin results, for example, do not belong in a LIBERO success table.
A smoke run proves integration, not a suite success rate.

## Evidence package

Each promoted run must keep, without overwriting an earlier run:

- command line and configuration;
- repository revision, hardware, driver, CUDA, Python/PyTorch, and build flags;
- checkpoint/tokenizer hashes and precision manifest;
- raw latency and VRAM samples;
- per-episode records, seeds, skipped counts, and Wilson intervals;
- a short note describing synchronization and inclusions in the timing boundary.

Generated logs, videos, and checkpoints stay under ignored `outputs/` and
`checkpoints/` paths. A compact, auditable summary may be committed under
`docs/results/`.

## Reference interpretation

The [original Embodied.cpp paper](https://arxiv.org/html/2607.02501#S4.T3)
reports pi0.5, GR00T N1.7, and HY-VLA latency per generated action and
independently normalized success and VRAM for Python, C++ BF16, 8-bit, 6-bit,
and 4-bit configurations. Those normalized values are valid published
references. The repository does not contain their absolute raw latency samples
or a complete rerun manifest, so they must not be presented as fresh local
measurements.

See [THREE_MODEL_VALIDATION.md](THREE_MODEL_VALIDATION.md) for the current
evidence ledger and the exact gaps that still block like-for-like comparisons.
