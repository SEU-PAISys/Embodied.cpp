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
| Xiaomi-Robotics-0 | Current F32 policy/F16 vision: C++ 396/400, Python 393/400; all 400 first-noise pairs match | Archived BF16/F16 comparison; new serial three-batch matrix in progress | F32 current full pair complete; BF16/Q8_0/Q6_K/Q5_K/Q4_K current-protocol coverage queued | Zero skips in completed pair; new quantized sweeps and controlled timing remain pending |
| TurboVLA | Current public Python FP32 386/400 and BF16 385/400, zero skips; earlier C++ F32 390/400 and BF16 382/400 remain separately sourced | Archived BF16 comparison; new serial three-batch matrix in progress | Four storage-quantized files rebuilt from the same metadata-complete source; full coverage queued | Historical Python 392/400 used BF16-autocast vision, not full FP32; old Q8 missing metadata is not a release artifact |
| X-VLA | Official phase2 strict recount: Python F32/BF16 391/400 each; C++ F32 387/400, BF16 390/400; quantized rows in linked audit | New controlled matrix uses official weights; no interim-snapshot ratio is promoted | Official F32/BF16/Q8_0/Q6_K/Q4_0/Q4_K full-run records audited for coverage | Strict full-BF16 parity still fails 0.005; mixed-precision diagnostic is not a waiver. New CUDA valid/error/valid recovery passed |

## Published baseline values

The original paper's normalized deployment table uses Python as 1.00 and
reports the following BF16 C++ latency/VRAM ratios: pi0.5 0.90/0.60, GR00T N1.7
0.72/0.93, and HY-VLA 0.48/0.68. The same paper also reports 8-bit, 6-bit, and
4-bit variants. These are upstream published baselines, not measurements
reproduced by this integration branch.

## Evidence already suitable for comparison

The archived Xiaomi-Robotics-0 and TurboVLA latency runs use one RTX
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

## Remaining release gates

1. Finish the serial 24-configuration, three-batch performance matrix. Report
   all batches separately; do not use concurrent rollout timing as that matrix.
2. Finish the queued 10-configuration / 4000-episode missing coverage.
   Reconcile the explicit precision inventory in addition to strict suite coverage.
3. Retain the X-VLA full-BF16 numerical failure until resolved. Different
   activation precision must remain explicit; no threshold or baseline waiver.
4. Publish unified tables only after raw evidence, source/weight fingerprints,
   protocol identity, numerical checks and coverage have been reconciled.

The [current audit](../docs/results/release_audit_20260907.md) and
[independently checked completed runs](../docs/results/completed_runs_audit_20260908.json)
supersede outdated task-status descriptions in historical follow-up reports.
Historical interim X-VLA snapshot measurements remain quarantined; official
phase2 records are a different evidence set, not rehabilitation of that snapshot.

The original-model rows are comparison references for integration structure and
evidence quality, not work items for this three-model PR. This PR does not download or rerun original-model weights. Published values remain reference evidence rather than
being silently mixed with fresh RTX 4090 measurements.
