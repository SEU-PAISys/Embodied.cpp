# SmolVLA Q8_0 LIBERO-Object follow-up (2026-09-30)

This follow-up records the requested SmolVLA Q8_0 quantization
experiment. Its claim level is **integration evidence**, not a normalized
Table 3 comparison. It uses the full policy-model quantization scope: every eligible
main-model matrix is stored and kept resident as native Q8_0, while tensors
that GGML does not quantize (for example biases and normalization vectors)
retain their required floating-point types.

## Technical-report mapping

The [2026-08-09 technical-report revision](https://arxiv.org/abs/2607.02501v3)
reports VLA deployment results as three ratios relative to a matched Python
baseline: inference latency per generated action step, success rate, and VRAM.
This run records the corresponding raw C++ 8-bit fields, while preserving the
measurement-boundary caveats below.

| Model | Configuration | Inference latency | Success rate | VRAM |
|---|---|---:|---:|---:|
| SmolVLA | C++ Q8_0, full policy-model scope | 5.187 ms/generated action (server metric) | 83.00% (166/200) | 1029 MiB peak whole-device use |

The normalized Table 3 ratios are intentionally not reported. The latency is
the server's request duration divided by the 50-action generated horizon, not
a fixed raw-observation-to-CPU-action deployment boundary; the VRAM source is
whole-device use rather than process-attributed peak memory. A valid ratio
requires a Python baseline measured with the same checkpoint, task/episode
matrix, seeds, timing boundary, hardware, and memory method. The repository's
earlier SmolVLA Python figures use a different protocol and measurement
boundary. Dividing those values would not be a like-for-like result.

## Protocol

| Field | Value |
|---|---|
| Benchmark | LIBERO-Object |
| Tasks | 0--9 |
| Episodes | 20 per task, 200 total |
| Environment seed | 1000 |
| Action-noise seed | 1000, derived per episode |
| Control | relative; one replayed action per request |
| Flow steps | 10 |
| Generated action horizon | 50 |
| Observation / model image | 360x360 / 512x512 |
| Reset settling | 10 steps, action `[0, 0, 0, 0, 0, 0, -1]` |
| Video | disabled |

The checked-in configuration is
[`eval/conf/libero_smolvla_object_q8_eval.yaml`](conf/libero_smolvla_object_q8_eval.yaml).

## Success result

All 200 requested episodes were evaluable; no episode was skipped.

| Task | Success / episodes | Success rate |
|---:|---:|---:|
| 0 | 12 / 20 | 60.00% |
| 1 | 16 / 20 | 80.00% |
| 2 | 19 / 20 | 95.00% |
| 3 | 19 / 20 | 95.00% |
| 4 | 19 / 20 | 95.00% |
| 5 | 16 / 20 | 80.00% |
| 6 | 20 / 20 | 100.00% |
| 7 | 9 / 20 | 45.00% |
| 8 | 17 / 20 | 85.00% |
| 9 | 19 / 20 | 95.00% |
| **Total** | **166 / 200** | **83.00%** |

The total Wilson 95% interval is 77.18%--87.57%.

## Raw deployment measurements

The first five inference requests were excluded as profiler warm-up samples.

| Metric | Samples | Mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| Server inference request | 34,966 | 259.330 ms | 245.021 ms | 315.640 ms | 344.102 ms |
| Generated action step (request / 50) | 34,966 | 5.187 ms | 4.900 ms | 6.313 ms | 6.882 ms |
| Environment step wall time | 34,966 | 273.684 ms | 259.537 ms | 330.123 ms | 359.992 ms |
| Device VRAM | 17,515 | 1014.37 MiB | 1009 MiB | 1029 MiB | 1029 MiB |

Peak observed device VRAM was 1029 MiB. The profiler recorded
`device_total_fallback`, not process-attributed VRAM, so this raw value is kept
separate from normalized Table 3 reporting.

## Quantization and startup evidence

| Field | Value |
|---|---:|
| Quantization type | Q8_0 |
| Scope | `main_model_full` |
| Quantized tensors | 452 |
| Selected quantized bytes | 450,225,280 |
| Source tensor bytes | 1,045,070,008 |
| Output tensor bytes | 639,845,688 |
| Native resident Q8_0 tensors | 452 |
| Native resident Q8_0 bytes | 450,225,280 |

The runtime startup evidence agrees with the quantizer manifest: source storage
and native resident type are both Q8_0. The full-policy quantizer is
[`scripts/quantize_smolvla_full_gguf.py`](../scripts/quantize_smolvla_full_gguf.py),
following the repository's HY-VLA quantizer pattern.

Artifact SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| Source policy | `ddfbc78bafcdad9ae17ce3ed1596f7126b5d691e8241ec1f2535a228c48697dd` |
| Q8_0 policy | `dbe036a7886f9b911162b48383819d994b88a3b3e572b4ccb2008831b5397bc5` |
| mmproj | `d4b9f78400b80dee86b58ad1b112636ef37a1fdb46e6076447d60fc8683d1140` |
| Tokenizer tree | `025bfe9eeedead50c7ea5fc815ef01532c29e5373e5cb745fae076fec09353ae` |
| Experiment server binary | `e45398c0bf2cd069a166e504a309683535988cc9f888e4f41257959034981210` |

## Environment and evidence handling

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (`sm_89`)
- CUDA / driver: 12.0 / 576.88
- Build: Release, `-DGGML_CUDA=ON`, g++ 13.3.0
- Runtime: WSL2 Linux 5.15.146.1, Python 3.10.21
- Base repository revision: `dcedfec5721768661bd81f4196a0325d55f814f9`

The experiment ran from an uncommitted working tree based on the revision
above. The retained server binary is therefore identified explicitly by hash;
the capture manifest's dirty-path count alone is not a source patch hash. The
PR records the tested Q8/runtime path after removing opt-in diagnostic-only
instrumentation that was disabled during the rollout.

Raw task results, profiler samples, copied config, quantizer manifest, and
startup capture remain under
`outputs/smolvla_q8_full_10x20_20260930_v1/`. Generated outputs and model
artifacts remain ignored, consistent with
[`eval/VLA_BENCHMARK_STANDARD.md`](VLA_BENCHMARK_STANDARD.md); this compact
follow-up is the checked-in audit record. The strict capture and aggregation
utilities reject missing tasks, incomplete episode counts, protocol drift,
non-finite/negative timing samples, missing raw samples, and non-native Q8_0
startup evidence.
