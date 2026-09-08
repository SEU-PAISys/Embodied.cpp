# X-VLA — Full LIBERO Evaluation

Runtime: Embodied.cpp C++/GGML · Reference: official 2toinf/X-VLA
PyTorch implementation.

## 2026-09-03 validation and source warning

Recounted seed-42 BF16-resident results: spatial 96/100, object 99/100,
goal 95/100, libero_10 97/100 (**387/400, 96.75%**). Existing Q8_0/Q4_0
storage runs cover **object only**, at 98/100 and 99/100, and still use BF16
residency. These are not fresh full-suite quantized-compute sweeps.

A separate fresh object run of the repaired source with actual F32 residency
completed **100/100**, with zero skipped episodes. A separate historical-GGUF
run reached **99/100** with `VLA_XVLA_F32_WEIGHTS=1`, seed 42; the source audit
shows that these are different checkpoints, not same-weight replications.
The former file labelled
`f32` had two incorrectly ordered per-domain matrices and was actually run
with BF16 residency; its 0/100 score did not establish an F32 precision failure.
Conversion now preserves F32 values; a regression checks both matrix transposes.

The historical source HF snapshot was not found in the Windows workspace,
WSL project/cache or server project/cache. The current server HF snapshot is
a different set of weights. Consequently **Python-normalized performance
remains Pending**. See the [controlled repair and evidence](takeover_20260903.md).
The following seed-7 archive is retained separately, not relabelled as this run.

The [Q6_K follow-up](turbovla_xvla_q6_followup_20260904.md) compares the
historical GGUF against its own Q6 storage derivative: object task 0 is 10/10
on both sides, and GPU memory is unchanged. This does not resolve the absent
historical HF baseline.

A separately approved [new HF/GGUF pair](xvla_matched_followup_20260904.md)
now passes F32 and BF16 fixed-input Python/C++ parity and has repeated matched
deployment measurements. It does not show a consistent BF16 latency advantage
or a BF16 process-memory reduction, and is not a full success-rate rerun.

## Archived protocol (seed 7)

- Suites: LIBERO-spatial / object / goal / libero_10, all 10 tasks each
- Episodes: 10 per task → 100 per suite, 400 total · seed 7
- Observation: 256×256 dual-view
- Conversion: `scripts/convert_xvla_to_gguf.py`; parity tooling:
  `tools/xvla_parity.cpp` + `scripts/parity_xvla_reference.py`

## Results

| Suite | Official PyTorch | C++ bf16 | C++ q8_0 | C++ q4_k |
|---|---|---|---|---|
| spatial | 100/100 | 99/100 | 100/100 | 100/100 |
| object | 100/100 | 100/100 | 100/100 | 97/100 |
| goal | 99/100 | 98/100 | 99/100 | 100/100 |
| libero_10 | 100/100 | 100/100 | 100/100 | 98/100 |
| **total** | **399/400 = 99.8%** | **397/400 = 99.2%** | **399/400 = 99.8%** | 395/400 = 98.8% |

## Findings

1. C++ bf16 lands within two episodes of the PyTorch reference over 400
   rollouts; per-task differences are scattered with no systematic
   suite-level gap.
2. q8_0 reproduces the PyTorch result exactly (399/400) and is the
   recommended quantized configuration; q4_k gives up ~1 pp.
3. Reference latency (fixed-input benchmark): official PyTorch fp32 CUDA
   query ≈ 841 ms mean vs the C++ direct path at ≈ 490 ms wall RT on the
   same machine (`outputs/xvla_py_signal`, `outputs/compare_20260818`).
   These historical values have different timing boundaries; the C++ dtype,
   sample count and full machine metadata are not archived here. They are
   not a controlled BF16 speedup and are excluded from README's ratio table.

## Upstream note

X-VLA was accepted to ICLR 2026 and is now natively integrated into
LeRobot (`lerobot/xvla-base`, 0.9B). The runtime here targets the
original 2toinf/X-VLA inference stack and its LIBERO checkpoints.


## Numerical parity status (2026-09-08)

Measured against the official Libero weights (HF `260cc588…`) on fixed inputs:

| C++ path | Python reference | max abs action error | verdict |
|---|---|---:|---|
| C++ BF16 matmul / F32 activations (current default) | official all-BF16 | ~0.0105 | **exceeds** the 0.005 fixed-input threshold |
| C++ (as above) | matched matmul-rounding + F32-activation strategy | ~0.0019 | passes, but is **not** element-wise equivalence with the official all-BF16 path |
| C++ F32 | official F32 | within threshold | passes |

The repository therefore records X-VLA C++ BF16 as a **mixed-precision
deployment** (BF16 matmul, F32 activations). It is not claimed to be
element-wise equivalent to official all-BF16, and the 0.005 threshold is not
relaxed to make it pass. Closing this gap requires an all-BF16 activation path
in the C++ runtime; until then, success-rate parity (C++ BF16 97.50% vs Python
BF16 97.75% over 400 episodes) is reported as behavioural evidence, and strict
numerical parity for BF16 remains **not passed**.
