# VLA validation and comparison ledger

This ledger places the three newly integrated runtimes beside the existing VLA
runtimes under the same evidence rules. “Validated” always names a bounded
scope; it never means that every precision and benchmark has been completed.

The acceptance rules are in [VLA_BENCHMARK_STANDARD.md](VLA_BENCHMARK_STANDARD.md).

| Model | Closed-loop evidence | Matched Python/C++ performance | Precision coverage | Current status / missing evidence |
|---|---|---|---|---|
| pi0.5 | Published normalized closed-loop result | Published normalized BF16/8/6/4-bit result | Paper matrix | Absolute raw samples and a current local rerun manifest are not in this repository |
| GR00T N1.7 | Published normalized closed-loop result; local single-task integration sample | Published normalized BF16/8/6/4-bit result | Paper matrix | Absolute raw samples and a current local rerun are missing |
| HY-VLA | Published normalized RoboTwin result | Published normalized BF16/8/6/4-bit result | Paper matrix | Benchmark differs from LIBERO; absolute raw samples and a current local rerun are missing |
| SmolVLA | Historical 400-vs-400 LIBERO comparison: 284/400 C++ vs 265/400 official; final seed protocol only has a 12-vs-12 smoke | Pending: recorded latency scopes differ; a same-method process-memory sample is 1117 vs 1385 MiB | One unlabelled GGUF configuration | Upstream reference limitation: final-protocol full scale and an explicit precision manifest are absent; not a work item for this three-model PR |
| Xiaomi-Robotics-0 | Historical 2000 episodes per BF16/Q8_0/Q4_K variant; paired BF16/Q6_K task-0 gate is 10/10 each under the final seed protocol | Matched BF16 policy + F16 vision: C++ 53.56 ms vs Python 143.63 ms; Q6_K is 69.33 ms and 4476 MiB vs BF16 8824 MiB | BF16, Q8_0, Q4_K full historical results; Q6_K fixed-boundary + one-task evidence | Q6_K misses strict `atol=0.005` parity by 1/960 values and lacks a full suite; historical full runs predate the final explicit seed manifest |
| TurboVLA | Fresh 400 episodes per BF16/Q8/Q4 storage variant; shared-loop two-task Python/C++ control is 20/20 on both sides; correct-layout Q6 378/400 vs BF16 382/400 | Matched BF16: C++ 27.24 ms vs Python 28.39 ms, 5 warm-ups + 100 samples | BF16, Q8_0, Q4_0 and Q6_K storage sweeps; Q6 has BF16 residency and no runtime-memory saving | Full matched Python success sweep remains pending; the old-layout 350/400 run is retained only as configuration-failure evidence |
| X-VLA | Historical four-suite results plus repaired-source F32 object-suite 100/100; historical-source Q6 392/400 under legacy observation-derived noise; explicit-noise paired gate 10/10 each. **Gate A (2026-09-05)**: the interim HF snapshot package `3f16a4b6…` (weights + config + locally modified remote code, loaded together) is not a valid release baseline for the tested LIBERO protocol — 901/903 tensors differ from official `2toINF/X-VLA-Libero` and it scores 0/10 under the official evaluator where the official package scores 10/10 (package-level A/B; pure-weight attribution not claimed); with official weights both the C++ server (official-derived GGUF `2c828fe6…`) and the Python public entry score 10/10 independently (see [gate-A report](../docs/results/xvla_checkpoint_gate_a_20260905.md)) | New same-source F32/BF16 parity and repeated timings available; no consistent BF16 latency advantage, 3.9% higher process VRAM — **but all of these used the invalid snapshot package and must be redone against official weights before any release claim** | BF16, Q8_0, Q4_0 and repaired-source F32 C++ results; historical-source Q6 full integration observation, not matched non-inferiority | Historical-source normalized row remains Pending; matched full success comparisons must be rerun on official weights (`260cc588…` / GGUF `2c828fe6…`) under the derive-episode-noise protocol; Q6 noise protocol differs from the paired gate |

## Published baseline values

The original paper's normalized deployment table uses Python as 1.00 and
reports the following BF16 C++ latency/VRAM ratios: pi0.5 0.90/0.60, GR00T N1.7
0.72/0.93, and HY-VLA 0.48/0.68. The same paper also reports 8-bit, 6-bit, and
4-bit variants. These are upstream published baselines, not measurements
reproduced by this integration branch.

## Evidence already suitable for comparison

The current matched Xiaomi-Robotics-0 and TurboVLA latency runs use one RTX
4090, identical raw CPU fixtures on both sides, the same checkpoint and compute
precision, five warm-ups, 100 timed requests, complete CPU action read-back,
and a separate process-memory phase. Their raw samples, hashes, and commands are
summarized in
[the 2026-09-03 follow-up](../docs/results/v2_followup_20260903.md) and its
[machine-readable evidence](../docs/results/v2_followup_20260903_evidence.json).

These request-level measurements are not cross-model rankings. To compare the
cost of producing one action across different output horizons, use the new
`generated_action_step_ms` samples from the shared profiler. Keep
`model_step_ms` only for the separate question of control-loop replay cost.

## Next experiments, in order

1. Extend the [new same-source X-VLA pair](../docs/results/xvla_matched_followup_20260904.md)
   from fixed-input parity and repeated deployment timings to matched success
   comparisons. It has no consistent BF16 latency or memory advantage; the
   historical checkpoint's normalized row remains Pending.
   **Root cause closed (2026-09-05)**: the interim HF snapshot package
   `3f16a4b6…` (weights + config + locally modified remote code, loaded
   together via `trust_remote_code`) is NOT a valid release baseline for the
   tested LIBERO protocol (901/903 tensors differ from official
   `2toINF/X-VLA-Libero`; 0/10 official-evaluator episodes vs 10/10 for the
   official package; package-level A/B, pure-weight attribution not
   claimed). All snapshot-derived parity/timing/VRAM numbers are quarantined;
   matched full comparisons must be rerun on official weights
   (`260cc588…` / GGUF `2c828fe6…`) under the derive-episode-noise protocol —
   see
   [gate-A report](../docs/results/xvla_checkpoint_gate_a_20260905.md).
   Historical partial progress below is retained for the audit trail: a
   standalone copy of the denoise loop
   is bit-exact against official `generate_actions()` with the same seed/inputs
   (`scripts/verify_xvla_denoise_equiv.py`, max abs error 0.0).
2. The corrected-layout TurboVLA Q6 full sweep is now complete at 378/400
   against same-protocol BF16 382/400. The [artifact audit](../docs/results/q6_artifact_audit_20260904.md)
   identifies why the earlier 350/400 run is not that result. X-VLA's 392/400 full integration
   observation uses legacy noise; a final-protocol paired comparison remains
   separate. Both retain the [same-batch Q6 report](../docs/results/turbovla_xvla_q6_followup_20260904.md).
   XR0 Q6_K has a
   [fixed-boundary and one-task report](../docs/results/xr0_q6_followup_20260904.md),
   but its strict parity miss and missing full suite remain open.
3. Rerun full success comparisons only after the checkpoint, seed, replay,
   noise, and suite manifests are frozen. A task-level smoke should remain
   labelled as integration evidence.

The original-model rows are comparison references for integration structure and
evidence quality, not work items for this three-model PR. No original-model
weights are currently present on the validation server and this PR does not
download or rerun them. Published values remain reference evidence rather than
being silently mixed with fresh RTX 4090 measurements.
