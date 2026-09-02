# Xiaomi-Robotics-0 — Full LIBERO Evaluation

Runtime: Embodied.cpp C++/GGML · Reference: official Xiaomi-Robotics-0
PyTorch stack (`deploy/server.py` + `eval_libero/main.py`,
paper avg **98.7%**).

## Protocol

- Suites: LIBERO-spatial / object / goal / libero_10, all 10 tasks each
- Episodes: 50 per task → 500 per suite, 2000 total per configuration
- Step budgets follow the official per-suite protocol:
  spatial 220 / object 280 / goal 300 / libero_10 520
- Official client CLI uses tyro function-style flags (`--args.*`);
  C++ side runs `eval/client/run_sim_client_direct.py --arch xr0`

## Results

| Suite | Paper | Official PyTorch | C++ f32 | C++ bf16 | C++ q8_0 | C++ q4_k |
|---|---|---|---|---|---|---|
| object | 100% | 99.4% | 99.2% | 99.4% | 99.4% | **100.0%** |
| spatial | 98.8% | 99.0% | 98.8% | 98.2% | 98.6% | 98.8% |
| goal | 98.8% | 97.4% | 97.8% | 98.4% | 98.4% | 98.0% |
| libero_10 | 97.2% | 97.2% | 96.8% | 97.8% | 96.2% | 97.4% |
| **avg** | **98.7%** | **98.25%** | **98.15%** | **98.45%** | **98.15%** | **98.55%** |

## Findings

1. The C++ runtime matches the official implementation across the entire
   precision spectrum: all five variants land within ±0.3 pp of the
   PyTorch reference, and per-task failure patterns coincide on the hard
   tasks (goal t2/t3, libero_10 t6).
2. **q4_k is the efficiency pick**: weights shrink 8.6 GB → 2.99 GB while
   scoring 98.55%; across 2000 episodes only one per-task deviation
   exceeds 2 pp vs bf16.
3. f32 adds nothing over bf16 (98.15% ≈ 98.45%), confirming bf16 as the
   right default for this runtime.
4. Numeric parity of the runtime is max-abs-error 3.3e-4 vs PyTorch in
   bf16; k-quant checkpoints are produced by
   `scripts/quantize_xr0_gguf.py` (q8_0/q6_k/q5_k/q4_k; big-matmul-only,
   norms and embeddings stay high-precision).

## Per-task success rates (official PyTorch run, seed 7)

```
object : 98 100 98 98 100 100 100 100 100 100
spatial: 100 100 100 100 96 94 100 100 100 100
goal   : 98 100 92 88 100 98 100 100 100 98
10     : 98 98 100 94 100 100 86 100 98 98
```

## Reproduction pointers

- Official reference server/client: Xiaomi-Robotics-0 repo.
- C++ serving: `build/vla-server --bind tcp://*:PORT mmproj.gguf
  model.gguf`; `VLA_XR0_F32_WEIGHTS=1` upcasts matmuls to f32 at load.
- Client: `eval/client/run_sim_client_direct.py --arch xr0 ...`
  (requires `protoc` on PATH and a LIBERO checkout on `PYTHONPATH`; see
  `eval/sim/libero/setup_libero.sh`).
