# TurboVLA and X-VLA Q6_K follow-up

Date: 2026-09-04. This is a storage-quantization bring-up and bounded quality
gate, not a full-suite acceptance or a new Python speedup claim.

## Artifacts and runtime

The public `scripts/quantize_vla_gguf.py` now accepts `q6_k`, reusing the
existing GGML quantizer. Tensor selection follows the existing Q8/Q4 policy,
with Q6_K's 256-value row alignment. Ineligible tensors keep their source type.
Both runtimes dequantize selected matrices to **BF16 residency** at load time;
this is not native 6-bit execution.

| Model | Selected matrices | Q6 file bytes | Source SHA-256 | Q6 SHA-256 |
|---|---:|---:|---|---|
| TurboVLA | 215 | 209746432 | `055613848ce918c95a6f2564020a90fa0f2e40c2fa490881e5b895c8aa95c1df` | `c253c24c1015165b44ce6b6914db3bd360e1f6419df57ac12dfa7ab5c16f639f` |
| X-VLA | 267 | 999042848 | `e8c89688042bca1cbbfcf70f9369b7605a348c03395e22bb55e6fde0c4cced79` | `471630d4f37f180d3381cbba9df623ba8254a5c16085a211cbac60116de565da` |

TurboVLA uses the corrected per-instruction-layout BF16 GGUF. X-VLA uses the
historical GGUF (F32 storage, BF16 default residency), not the mismatched
available HF snapshot. X-VLA's Python-normalized comparison remains Pending.

## Same-batch fixed-input performance

One isolated RTX 4090, driver 580.95.05; sequential BF16 then Q6 runs for each
model. Each run uses 5 warm-ups, 100 timed requests and 20 separate untimed
memory requests. The boundary includes raw CPU observations, preprocessing,
ZMQ transport, transfers, GPU synchronization and the complete CPU action
chunk; it excludes model loading and the simulator.

| Model / storage | Mean ms | Std | p50 | p95 | p99 | Process peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| TurboVLA BF16 | 24.584 | 1.184 | 24.543 | 26.360 | 27.204 | 880 |
| TurboVLA Q6_K | 25.642 | 0.737 | 25.689 | 26.462 | 27.888 | 880 |
| X-VLA original F32 / BF16 resident | 95.910 | 2.031 | 95.478 | 98.229 | 106.574 | 2370 |
| X-VLA Q6_K / BF16 resident | 95.388 | 1.162 | 95.275 | 97.454 | 98.458 | 2370 |

Q6 saves disk space, **not GPU memory**, in these two runtimes. TurboVLA was
4.3% slower in this batch; X-VLA's 0.5% difference does not establish a speedup.
These fresh C++-only runs must not be divided by an older Python sample to
manufacture a better ratio. The previous matched Python/C++ report remains
separate. Neither batch reproduced the former periodic 150–300 ms spikes.

For deployment cost per generated action, divide these raw samples by 12
(TurboVLA) or 30 (X-VLA), not by a simulator's replay setting.

## Fixed-input output differences

Two 256px views and the same raw 8-D state were used within each pair.
TurboVLA returns 12×7 actions; X-VLA returns 30×20 with fixed explicit noise,
domain 3, and 224px preprocessing. All outputs were finite.

| Q6 versus its same-source C++ BF16 baseline | Max abs | Mean abs | Values > 0.005 | Values > 0.01 |
|---|---:|---:|---:|---:|
| TurboVLA | 2.000000 | 0.056508 | 37/84 | 35/84 |
| X-VLA | 0.005393 | 0.000774 | 7/600 | 0/600 |

TurboVLA Q6 is not numerically equivalent to BF16 on this fixture. X-VLA's
smaller error is not a Python parity result and does not prove full-suite
quality. No tolerance was relaxed to turn either result into a pass.

## Paired closed-loop gates

Both precisions used the same task set, ten episodes per task, environment
seed 42, native 256px two-view rendering, no video, and the public shared
client/profiler. TurboVLA used relative control and 12-action replay, with no
stochastic-noise input. X-VLA used absolute ee6d control, 30-action replay,
domain 3 and explicit per-episode noise derived from seed 42.

| Scope | BF16 resident baseline | Q6_K storage / BF16 resident |
|---|---:|---:|
| TurboVLA goal task 2 | 10/10 | 10/10 |
| TurboVLA goal task 3 | 9/10 | 10/10 |
| TurboVLA goal task 5 | 10/10 | 10/10 |
| TurboVLA selected tasks total | 29/30 | 30/30 |
| X-VLA object task 0 | 10/10 | 10/10 |

All 80 episodes completed, with zero skipped episodes. These TurboVLA tasks
were selected because of earlier Q4 sensitivity, before seeing Q6 outcomes.
The fixed-fixture discrepancy did not imply failure on these tasks, but neither
does this small gate establish Q6 superiority or full-suite equivalence.

## Evidence and reproduction

Local raw evidence: `outputs/vla_unified_20260904/`, including four boundary
JSONs, complete final action arrays, and the run scripts. Artifacts are created
in a unique server `/tmp` directory rather than the nearly full `/home` mount.
The [machine-readable summary](q6_and_xvla_evidence_20260904.json) includes raw
latency samples, action comparisons, hashes and all 80 episode records.

Relevant source/build SHA-256:

- Quantizer: `f3e6f426e3abc4ef94d7209e2724741b5415683eb694f1b8272a625cb7f5d4c5`
- Profiler: `972daee56683a77200c239961f157629b7e517541a907d6aab9c6c7c38acd597`
- Boundary benchmark: `ad18ee124708e2fd679f250b1bf60c45cc465bf5ba15b5c51dc42d2d9d78037d`
- Server binary: `6cb5c31fa4d59b232b91283d97b0092e73e77b92c4e360c799f4763b61d58959`

The server checkout is `19176e1` plus the validated working-tree changes;
do not identify its binary using the checkout revision alone. The local
checkout remains `517b972` plus uncommitted changes; nothing was pushed.


## Full-suite success rates (2026-09-04, seed 42, 256 px, 400 episodes each)

| Suite | TurboVLA Q6_K | X-VLA Q6_K |
|---|---:|---:|
| spatial | 96/100 | 98/100 |
| object | 100/100 | 100/100 |
| goal | 69/100 | 96/100 |
| libero_10 | 85/100 | 98/100 |
| **total** | **350/400 (87.50%)** | **392/400 (98.00%)** |

X-VLA Q6_K lands within 1.2 pp of its BF16 reference (99.2%) and is acceptable
as the formal 6-bit storage variant. TurboVLA Q6_K shows the same goal-suite
sensitivity already recorded for Q4_K/Q4_0 (68%/83%) — the loss is
model/quantizer-format specific and non-linear in bit width (Q8_0 at 94.75%
is loss-free), so Q8_0 remains the recommended quantized configuration for
TurboVLA. The missing padding-array concern was cleared by re-conversion:
tensor name sets are identical (673 = 673) and the checkpoint metadata carries
a single text_padding_length = 21, so no per-instruction data was ever lost.
