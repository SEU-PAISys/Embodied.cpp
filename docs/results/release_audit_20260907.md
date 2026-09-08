# Three-model release evidence audit — 2026-09-07

Base `344622d7` plus the current uncommitted fixes. This report reconciles
existing results before deciding what to rerun; it is not release approval.
The [common standard](../../eval/VLA_BENCHMARK_STANDARD.md) remains authoritative.

## Durable source evidence

Archived existing JSON, summary, log, fixture and action files (no weights,
videos or core dumps) from the server's phase1/2/3/4 and later full-run trees:
`outputs/release_20260907/existing-evidence.tar.gz`, 4,900,585 bytes, SHA256
`cb18c925df7ee8f989010f4ed27c7033223b70da0ef6b4565633c0f49357742f`.
The archive is deliberately ignored by Git. Per-result hashes and recounts are
in the [machine-readable audit](release_coverage_audit_20260907.json).

Strict validation uses the repaired public aggregator, checking the actual
four suites × ten unique tasks × ten episode IDs, recomputed success/skip
counts and summary agreement. The explicit inventory contains 20 runs / 8000
episodes; **all pass coverage**, with zero skipped episodes. This does not
automatically prove weight, compute precision, initial-state or noise identity.

## Complete target inventory, including unresolved cells

| Model | Implementation / requested policy or storage | Recounted success | Evidence restriction |
|---|---|---:|---|
| X-VLA | Python F32 | 391/400 | Official phase2; paired first-noise checks pass |
| X-VLA | Python BF16 | 391/400 | Same |
| X-VLA | C++ F32 | 387/400 | Same |
| X-VLA | C++ BF16 matrices / F32 activations | 390/400 | Strict full-BF16 parity remains failed |
| X-VLA | C++ Q8_0 storage | 388/400 | Phase2; BF16 residency, not native Q8 compute |
| X-VLA | C++ Q6_K storage | 393/400 | Same |
| X-VLA | C++ Q4_0 storage | 391/400 | Same |
| X-VLA | C++ Q4_K storage | 393/400 | Same |
| TurboVLA | Python F32 policy / BF16-autocast vision (historically labelled FP32) | 392/400 | Not a pure F32 baseline; corrected loader requires a new run |
| TurboVLA | Python BF16 | 385/400 | Current public-entry rerun confirmed; see completed-run audit below |
| TurboVLA | C++ F32 | 390/400 | Same |
| TurboVLA | C++ BF16 | 382/400 | Same |
| TurboVLA | C++ Q8_0 storage | Historical evidence, not re-audited here | Must reconcile layout, source and protocol |
| TurboVLA | C++ Q6_K storage | Historical evidence, not re-audited here | Same |
| TurboVLA | C++ Q4_0 storage | Historical evidence, not re-audited here | Same |
| TurboVLA | C++ Q4_K storage | Historical evidence, not re-audited here | Same |
| XR0 | Python F32-policy | 393/400 | Corrected relative-control public rerun; all 400 noise pairs verified |
| XR0 | Python BF16-policy | 394/400 | Recorded checksums; C++ counterpart lacks them |
| XR0 | C++ F32-policy | 396/400 | Corrected relative-control public rerun; all 400 noise pairs verified |
| XR0 | C++ BF16-policy | 393/400 | Missing first-noise checksum; cannot certify pairing from seed alone |
| XR0 | C++ Q8_0 | 392/400 | Same evidence gap; native quantized matrices |
| XR0 | C++ Q6_K | 392/400 | Same; historical fixed-input gate remains separate |
| XR0 | C++ Q5_K | 398/400 | Same; not silently omitted from release scope |
| XR0 | C++ Q4_K | 395/400 | Same |

Quantized/storage rows compare with their model's declared unquantized Python
baseline; no unsupported Python low-bit implementation is invented. XR0's
vision dtype remains a separate axis. Original models are upstream references
and compatibility targets, not extra weight downloads or full evaluation jobs.

## X-VLA paired records

Official phase2 manifest identifies HF SHA256 `260cc58869125b826e93bcaa60bca3ea37bcc7aaf893d368a991ca14fde9f0c8`
and unquantized GGUF `2c828fe612c76db99ebf0fe66d1baec7ffcfcea0be904721dcad415b22e8d9b3`.
For each precision, all 400 Python/C++ episode keys, recorded derived seeds
and first-request checksums match. F32 has 7 Python-only successes and 3
C++-only successes; BF16 has 7 and 6. These are observed paired differences,
not proof of equivalence or non-inferiority. Initial-state/source identity
must still be tied to the retained driver/source manifest, not assumed from
the noise check.

## Numerical diagnosis, not a changed acceptance threshold

On the new server binary, the historical BF16 fixture reproduces max error
0.010524869 against official full BF16. A diagnostic Python control with
BF16-rounded linear/token/image-projection weights and F32 activations gives
0.001904577 against C++. Biases/norms/convolutions remain F32 in that control.
It is **not** a replacement official baseline or an exact GGML emulator.

Three additional real initial observations (spatial task 4, object task 0,
goal task 0), through the shared parser with complete 20-D proprioception,
produce these aggregate errors against C++ BF16:

| Python strategy | Max abs | Mean abs | RMSE | Values >0.005 | First-arm gripper flips |
|---|---:|---:|---:|---:|---:|
| Official full BF16 | 0.011243463 | 0.000780777 | 0.001564855 | 41/1800 | 0/90 |
| Official F32 with the same explicit noise | 0.000889659 | 0.000061871 | 0.000116693 | 0/1800 | 0/90 |
| Selective BF16 weights / F32 activations diagnostic | 0.000597954 | 0.000060094 | 0.000111176 | 0/1800 | 0/90 |

These observations support precision-strategy differences rather than justify
relaxing a threshold. Strict full-BF16 parity remains **FAIL**. Mid-trajectory
and gripper-boundary coverage is not established by these initial observations.
The mixed-precision release policy has been explicitly put to the user; no
silent waiver is applied.

Real server sequence valid → domain 999 error → valid returned identical valid
actions, and SIGTERM exited with code 0. The temporary server was stopped in
`finally`; no other GPU process was terminated.

## Remaining work

### Controlled matrix completed; new numerical failure retained

All 72 batches (24 configurations × 3) passed sample-count, finite-output,
fixture-hash and process-memory-source checks. See the
[complete measurement table](controlled_performance_20260908.md) and
[numerical audit](controlled_performance_20260908.json). No samples were pooled
or discarded. Archive: `outputs/release_20260907/performance-evidence.tar.gz`,
2,155,501 bytes; SHA256
`5019406e4059a33c31d8885db1bb603e7f1002fccbd383a8c39342e2d54c4be8`.

On this real initial observation TurboVLA BF16 fails its existing 0.01
tolerance: max error **0.134080887**, reproduced in all three batches.
TurboVLA F32 max error is **0.001003385**. Cross-storage/runtime diagnostics
are in progress; successful closed-loop runs do not erase this discrepancy.
X-VLA BF16 also remains failed (max **0.009681821** against 0.005 on this
fixture). These BF16 rows are not release acceptance or acceleration claims.

XR0 F32 latency was stable around 94 ms versus Python 146–160 ms across the
three batches. TurboVLA has substantial between-batch variation; X-VLA F32
C++ was 114–117 ms versus Python 109–111 ms, so no F32 speedup is claimed.

### Completed current-protocol full runs

Independent local strict validation of the archived 1,600 episodes confirms:
XR0 C++ F32/F16 396/400; XR0 Python F32/F16 393/400; TurboVLA Python full
FP32 386/400; TurboVLA Python BF16 385/400. Every run contains 40 unique tasks,
400 counted episodes and zero skips. All 400 XR0 paired first-request noise
seeds, checksums, dtypes and device labels match.

The [per-file and protocol audit](completed_runs_audit_20260908.json) records
the independent recount. Raw local archive:
`outputs/release_20260907/completed-full-evidence.tar.gz`, 2,889,782 bytes,
SHA256 `11d73004b49c88aa69173abc693aa9dc6a5b98e7741e032b167c27782bdf741e`.
These are success-rate results, not controlled latency batches. The new
TurboVLA FP32 386/400 replaces neither the historical mixed-precision 392/400
nor its label: they are distinct precision experiments.

Clean CPU and CUDA builds passed with XR0/TurboVLA/X-VLA/Smol enabled;
targets were vla-server, xr0-parity and xvla-parity. Each build reported the
existing `setsockopt` deprecation warning, not zero warnings. New CUDA
artifact inference and serial 3-batch measurements are being validated.

### 2026-09-08: rejected XR0 rerun configuration

The new F32 paired run `xr0-f32-full-v1` completed 800 episodes with zero
skips, but both sides achieved only **11/400**. Its driver incorrectly chose
`control_mode=absolute`; the retained successful historical XR0 runs use
`relative`, which is also the public runner's XR0 default. The standalone
XR0 reference contained the same incorrect hardcoded override and has been
corrected with a regression assertion. This run is failed configuration
evidence, **not a model regression or an accepted performance comparison**.
The relative-control paired smoke test subsequently passed **2/2 on each
side** on spatial task 0. The replacement full run subsequently completed
396/400 C++ and 393/400 Python as independently audited above.

The local archive `outputs/release_20260907/new-evidence.tar.gz` contains the
failed full run and the XR0/X-VLA numerical probes (3,901,032 bytes; SHA256
`b420c394f1ac8cfbbdbcbf08abd256e3d5785d7f16e593eb7b81807c5b3bbd9b`).
Local strict aggregation independently passed all 80 task records. The
server manifest lacks its final coverage marker; do not claim that its
original end-to-end driver completed successfully. Its server shutdown was 0.

TurboVLA's public Python entry and request profiling now have four passing
CPU regression tests; the shared integration suite passes 41 tests on Linux.
The newly corrected FP32 vision path passed 2/2 real-weight episodes through
the public entry and produced a complete profile. Its full FP32/BF16 runs
subsequently completed 386/400 and 385/400. The shared result-aggregation
regression suite passes 24 tests.

The paired XR0 smoke and TurboVLA smoke records are archived locally as
`outputs/release_20260907/smoke-evidence.tar.gz` (16,418 bytes; SHA256
`e2e6230ec39e529af50c2a51bc860ce412a54e77f62d49ba8e6a6efe42bf089b`).
These small runs establish functioning entry points, not release success
rates or benchmark speedups. Concurrent GPU jobs must not be treated as a
controlled performance batch.

Reconcile all runtime/source identities, close the XR0 changed-noise protocol
with real inputs and affected full sweeps, audit TurboVLA storage runs, and run
the unified repeated performance matrix using the corrected precision-capable
benchmark. Old 8-D X-VLA fixtures remain diagnostic: new full-state fixtures
must not be labelled as that same timing batch. Only then update normalized
performance rows and complete clean-build/public-command acceptance.
