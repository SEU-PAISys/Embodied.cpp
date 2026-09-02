# X-VLA — Full LIBERO Evaluation

Runtime: Embodied.cpp C++/GGML · Reference: official 2toinf/X-VLA
PyTorch implementation.

## Protocol

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
