# Xiaomi-Robotics-0 Q6_K follow-up

Date: 2026-09-04. This is a precision bring-up and one-task quality gate, not a
full LIBERO success-rate result.

## Artifact

`scripts/quantize_xr0_gguf.py` quantized 332 large VLM/DiT matrices from the
audited 8.60 GB XR0 policy and retained embeddings, norms, biases, and the small
action heads at their source types.

| Field | Value |
|---|---|
| GGUF type | `Q6_K` (`general.file_type=18`, `xr0.weight_dtype=Q6_K`) |
| File size | 3,999,876,096 bytes |
| SHA-256 | `f4c2f2c5a6e5ebba8e50973729573de2a38559aef06e3e22719bcffb2cca84d5` |
| Vision projector | Existing F16 `xr0-mmproj.gguf`, GPU enabled |
| Runtime device | NVIDIA GeForce RTX 4090, one isolated GPU |

The 4 GB model was created under `/tmp` because the server's `/home` filesystem
had only 11 GB free. It was deleted after the evidence was copied; no model
artifact is committed.

## Fixed-input numerical gate

The existing 256×256 two-view fixture, 8-D raw state, fixed 30×32 action noise,
and Python reference were reused.

| Comparison | Max abs | Mean abs | Values over 0.005 |
|---|---:|---:|---:|
| Q6_K vs Python reference | 0.007068 | 0.000169 | 1/960 (0.10%) |
| Q6_K vs current C++ BF16 | 0.017947 | 0.000352 | 14/960 (1.46%) |

The runtime loaded and returned a finite 30×32 action chunk, but the existing
strict `atol=0.005` parity gate **did not pass**. The Q6_K result must therefore
remain a separate quantized-quality result; it is not labelled numerically
equivalent to BF16.

## Matched fixed-boundary performance

The Q6_K run reused the exact BF16/Python fixture
SHA-256 `20b98c06ab9344bf319453473b58091d8bbb7ed907504ffa91b53ce7fd2da1a0`.
Both measurements cover raw CPU observations through a complete CPU action
chunk, use 5 warm-ups plus 100 timed calls, and sample process VRAM in a
separate 20-request phase.

| Policy | mean | std | p50 | p95 | p99 | Process peak |
|---|---:|---:|---:|---:|---:|---:|
| BF16 policy + F16 vision | 53.56 ms | 5.30 | 51.80 | 64.11 | 64.48 | 8824 MiB |
| Q6_K big matrices + F16 vision | 69.33 ms | 1.01 | 69.36 | 70.86 | 72.25 | 4476 MiB |

Against BF16, Q6_K used 0.51× process memory (49.27% lower) but took 1.29× the
latency (29.44% higher) on this RTX 4090. It is a memory option, not a latency
acceleration claim.

## Paired closed-loop gate

BF16 and Q6_K each ran LIBERO-object task 0 for 10 episodes with environment
seed 42, derived per-episode action-noise seed 42, relative control, replay 10,
no video, and the same public client.

| Policy | Success | Environment steps | Measured requests | Server mean / p99 | Per generated action mean / p99 | Process peak |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 10/10 | 1340 | 133 | 59.071 / 61.765 ms | 1.969 / 2.059 ms | 8824 MiB |
| Q6_K | 10/10 | 1411 | 139 | 56.143 / 59.916 ms | 1.871 / 1.997 ms | 4476 MiB |

Both policies passed this bounded task gate with no skipped episodes. Their
trajectories and request counts differ, so the rollout timings are descriptive
and do not replace the fixed-input performance comparison. Ten episodes of one
task also do not establish a suite-wide Q6_K success rate.

## Evidence and implementation identity

Raw local evidence is retained under the ignored directory
`outputs/vla_unified_20260904/codex-xr0-q6-evidence/`.

| File | SHA-256 |
|---|---|
| `xr0-q6-k-boundary.json` | `a115029806f054aa3b0e06b9992fffe4e55b4f27623ad1143df7de33df13f14c` |
| `rollout-q6-profile.json` | `0cf286f968b3f0f1ab646ca89ff8a1f773531c12edf4031a8d848b3b2c3e4e71` |
| `rollout-bf16-profile.json` | `092a8d950a960ea6136762f455eb1abf56a9023e273698bfd95e88df7d806e95` |
| Q6_K task result | `c10e901f9bcbadeb549b8721d03298154a3253defb1ec637eecca6565daf4700` |
| BF16 task result | `092cb9b9c3c7ca844f9db831c230ee34bb81a8e15d3080b52a6d5b135066c447` |

The local and server copies used identical source hashes:

- `models/xr0.cpp`: `d22e3fd19fc5c557925648eb23593d353468a81373e6f08ca78309db0113ab07`
- `scripts/quantize_xr0_gguf.py`: `e5ba55d22bfa46ec2db8f9b6df713d2a0e377ed6c038f122cb5ad158abb7daac`
- `scripts/bench_vla_boundary.py`: `ad18ee124708e2fd679f250b1bf60c45cc465bf5ba15b5c51dc42d2d9d78037d`
- `eval/client/libero_profile.py`: `972daee56683a77200c239961f157629b7e517541a907d6aab9c6c7c38acd597`
- server `build/vla-server`: `6cb5c31fa4d59b232b91283d97b0092e73e77b92c4e360c799f4763b61d58959`

Next: define an acceptance threshold for quantized checkpoints, then run the
full frozen suite protocol if Q6_K is intended as a published deployment tier.
