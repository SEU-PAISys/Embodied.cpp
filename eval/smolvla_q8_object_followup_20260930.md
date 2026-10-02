# SmolVLA Q8_0 LIBERO-Object follow-up (2026-09-30)

This follow-up records the requested SmolVLA Q8_0 quantization experiment and
its matched Python baseline. It provides Table 3-style normalized latency and
success values plus a same-source isolated-device VRAM ratio. The Q8_0
run uses the full policy-model
quantization scope: every eligible
main-model matrix is stored and kept resident as native Q8_0, while tensors
that GGML does not quantize (for example biases and normalization vectors)
retain their required floating-point types.

## Technical-report-style mapping

The [2026-08-09 technical-report revision](https://arxiv.org/abs/2607.02501v3)
reports VLA deployment results as three ratios relative to the same model's
Python baseline, which is set to 1.00. SmolVLA's matched results are:

| Model | Latency Python | Latency C++ 8-bit | Success Python | Success C++ 8-bit | VRAM Python | VRAM C++ 8-bit |
|---|---:|---:|---:|---:|---:|---:|
| SmolVLA | 1.00 | **0.63** | 1.00 | **0.97** | 1.00 | **0.62** |

The unrounded values are 6.160599 / 9.815109 = 0.627665 for latency,
83.00 / 86.00 = 0.965116 for success rate, and 1029 / 1671 = 0.615799
for VRAM. Lower latency and VRAM are better; higher success rate is better.

| Backend | Inference latency | Success rate | Peak device VRAM |
|---|---:|---:|---:|
| Python baseline | 9.815 ms/generated action | 86.00% (172/200) | 1671 MiB |
| C++ Q8_0, full policy-model scope | 6.161 ms/generated action | 83.00% (166/200) | 1029 MiB |

Latency was remeasured on 2026-10-02 with the same fixed-input deployment
boundary on both sides: two CPU CHW images, raw 8-D state, exact instruction,
and exact 50x32 action noise through preprocessing/tokenization, host-to-device
transfer, model generation, postprocessing/action denormalization, and complete
50x7 CPU action readback. C++ additionally includes its public local ZMQ
transport. Each backend used five untimed warmups and 100 timed requests; every
raw sample is retained. The table divides each request by the generated horizon
of 50, not by replayed environment actions.

VRAM used a separate 20-request untimed phase. WSL exposed the correct process
PID and GPU UUID but reported per-process `used_memory` as `N/A`, so both
isolated runs use the same `nvidia-smi memory.used` whole-device fallback on the
same otherwise-idle GPU. The evidence explicitly records
`process_level_vram=false`; the 0.62 value is a same-source device ratio and is
not relabelled as process-attributed memory.

The compact checked-in evidence, including all 100+100 latency samples and all
VRAM samples, is
[`eval/smolvla_q8_fixed_boundary_20261002.json`](smolvla_q8_fixed_boundary_20261002.json).
Its pair gate verifies the backend pair, measurement plan, boundary and script
hashes, fixture/instruction/checkpoint/tokenizer identities, output shape and
horizon, GPU UUID, and memory source before writing normalized values.

### Fixed-boundary performance protocol

| Field | Value |
|---|---|
| Boundary | raw CPU observation to complete 50x7 CPU action chunk |
| Fixed noise | exact 50x32 float32 array |
| Warmup / timed requests | 5 / 100 per backend |
| Memory phase | 20 additional untimed requests per backend |
| Python request mean / p50 / p95 / p99 | 490.755 / 489.818 / 511.586 / 518.767 ms |
| Q8 request mean / p50 / p95 / p99 | 308.030 / 301.968 / 384.434 / 393.719 ms |
| Python generated-action mean | 9.815109 ms |
| Q8 generated-action mean | 6.160599 ms |
| Repository revision | `3a3468cfe634e88d93d08aafcae89bd3ef0fb5a2` |
| Script SHA-256 | `dee5573089d882ec5fcade6c99ede546ab810c22526c894c68bbca471175a892` |

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

## Matched Python baseline

The Python baseline uses the same 10 tasks and 20 episodes per task. It produced
172 successes from 200 episodes (86.00%; Wilson 95%: 80.51%--90.13%).

| Task | Success / episodes | Success rate |
|---:|---:|---:|
| 0 | 17 / 20 | 85.00% |
| 1 | 18 / 20 | 90.00% |
| 2 | 17 / 20 | 85.00% |
| 3 | 19 / 20 | 95.00% |
| 4 | 17 / 20 | 85.00% |
| 5 | 10 / 20 | 50.00% |
| 6 | 18 / 20 | 90.00% |
| 7 | 20 / 20 | 100.00% |
| 8 | 20 / 20 | 100.00% |
| 9 | 16 / 20 | 80.00% |
| **Total** | **172 / 200** | **86.00%** |

After five warm-up requests per task, the baseline retained 33,866 latency
samples. Its mean generated-action latency was 9.445762 ms (p50 9.332979 ms,
p95 10.044731 ms, p99 11.544108 ms). Peak whole-device VRAM was 2658.5 MiB.
These closed-loop profiler values remain useful diagnostics, but they are not
the Python denominator for the normalized deployment row above: their timing
starts inside the policy and their memory includes the simulator process.
One raw request sample from task 1 (sample 2274, 180864.500452 ms) was excluded
from latency aggregation because the user-requested host process suspension
occurred inside that timing interval. The raw sample remains in
`python_profile.json`; its exact task/index/value/reason and both filtered and
unfiltered summaries are recorded in `baseline_result.json`. No threshold-based
filter is applied, and no success or VRAM sample was removed.

The reproducible baseline runner is
[`scripts/profile_smolvla_python_baseline.py`](../scripts/profile_smolvla_python_baseline.py).
The final aggregation explicitly passed
`--exclude-latency-sample "1:2274:user-requested host process suspension"`.
Exclusions require an exact task/index and reason; there is no implicit filter.

## Historical closed-loop deployment diagnostics

The following retained Q8 rollout measurements use the simulator/control-loop
profiler and are not used for the strict fixed-boundary latency or VRAM ratios
above. In particular, the former 0.58 latency indicator derived from these
values is superseded by the 0.627665 fixed-boundary result.

The first five inference requests were excluded as profiler warm-up samples.

| Metric | Samples | Mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| Server inference request | 34,966 | 259.330 ms | 245.021 ms | 315.640 ms | 344.102 ms |
| Server-only generated action step (request / 50) | 34,966 | 5.187 ms | 4.900 ms | 6.313 ms | 6.882 ms |
| Client `get_action` request | 34,966 | 273.684 ms | 259.537 ms | 330.123 ms | 359.992 ms |
| Deployment generated action step (client request / 50) | 34,966 | 5.474 ms | 5.191 ms | 6.602 ms | 7.200 ms |
| Device VRAM | 17,515 | 1014.37 MiB | 1009 MiB | 1029 MiB | 1029 MiB |

Peak observed device VRAM was 1029 MiB. The profiler recorded
`device_total_fallback`, not process-attributed VRAM. The superseded historical
comparison paired it with Python whole-device total-minus-free memory rather
than PyTorch's process allocator peak.

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
| Source F32 policy GGUF | `ddfbc78bafcdad9ae17ce3ed1596f7126b5d691e8241ec1f2535a228c48697dd` |
| Source Python policy tree | `4c18d39c74af6f7d73ee0e9565bdbf390bd0f2d63e9e501a574f3977a18b506a` |
| Q8_0 policy | `dbe036a7886f9b911162b48383819d994b88a3b3e572b4ccb2008831b5397bc5` |
| mmproj | `d4b9f78400b80dee86b58ad1b112636ef37a1fdb46e6076447d60fc8683d1140` |
| Tokenizer tree | `01614d37024f17471bb6106b3aacbebc70d0de19773c2feec1cd6ad9b8638ebc` |
| Experiment server binary | `e45398c0bf2cd069a166e504a309683535988cc9f888e4f41257959034981210` |
| Fixed-boundary script | `dee5573089d882ec5fcade6c99ede546ab810c22526c894c68bbca471175a892` |

## Environment and evidence handling

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (`sm_89`)
- CUDA / driver: 12.0 / 576.88
- Build: Release, `-DGGML_CUDA=ON`, g++ 13.3.0
- Runtime: WSL2 Linux 5.15.146.1, Python 3.10.21
- Base repository revision: `dcedfec5721768661bd81f4196a0325d55f814f9`
- Python baseline packages: PyTorch 2.5.1 (CUDA 12.4), Gymnasium 0.29.1,
  LeRobot 0.4.3, LIBERO 0.1.0
- Python baseline source revision:
  `d43d643fd8b2a9004908f529f73d641508e1712e` plus the runner added by this PR
- Python policy tree SHA-256:
  `4c18d39c74af6f7d73ee0e9565bdbf390bd0f2d63e9e501a574f3977a18b506a`
- Python processor/config tree SHA-256:
  `01614d37024f17471bb6106b3aacbebc70d0de19773c2feec1cd6ad9b8638ebc`

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
