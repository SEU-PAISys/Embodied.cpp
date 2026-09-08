# X-VLA: new same-source Python/C++ comparison

Date: 2026-09-04. The user approved a new matched pair because the original HF
source for the historical GGUF could not be found. This report does **not**
replace the archived seed-7 results or claim to recover the lost checkpoint.

## Source identity and numerical gate

The existing HF snapshot was converted into a separate F32-storage GGUF.
Replaying the converter's source-to-runtime mapping against the serialized
file verified all **902 tensors**, with zero mismatches or non-finite values.
This checks source identity and serialization; the numerical gate below also
tests the conversion through actual inference.

- HF `model.safetensors` SHA-256: `3f16a4b67a1d2675fc1cb0350c6d5617522e452f47359e729d74d76b8aa4835b`
- New GGUF SHA-256: `02941fd1bea58b3b1c3b45d0f279fdd050c00dba283d5fb694662505bb3f4bd1`
- F32 fixture SHA-256: `43dfdbd13df8a593ae06e6bacea559d9ae93de76d1d49d653421da09a6c6c233`
- BF16 fixture SHA-256: `418ef4bfde92537d51e89611ba248848704c93bd8362a2e08eb7fb3f05c82a95`

The fixtures share two 256px views, an 8-D raw state padded to 20, instruction,
domain 3 and 30×20 output. Each precision uses its corresponding seed-42 CUDA
noise on both sides. F32 and BF16 therefore have separate fixture hashes and
must not be treated as identical-noise cross-precision experiments.

| Same-source Python versus C++ | Max abs action error | Mean abs | Values > 0.005 |
|---|---:|---:|---:|
| F32 | 0.00020331 | 0.00003998 | 0/600 |
| BF16 configuration | 0.00310403 | 0.00068251 | 0/600 |

Both pass the existing 0.005 fixed-input gate. This is not a new full-suite
Python/C++ success comparison. In BF16 mode, C++ retains F32 activations and
normalization with BF16 matrix weights; Python uses BF16 model/input tensors.
The compute paths are not bitwise-identical despite sharing source weights.

## Timing and process VRAM

One isolated RTX 4090 (driver 580.95.05), 5 warm-ups, 100 timed requests and 20
separate untimed memory requests per run. All samples cover raw CPU input to
complete CPU actions, including preprocessing, uploads, synchronization and
read-back; C++ additionally includes ZMQ. The official Python
`generate_actions(steps=10)` is used without modifying upstream code.
Seed restoration is inside the Python timed call, and fixture noise is checked
against the official generator before timing. No memory sampler runs during
the timed phase.

| Configuration | Mean ms | Std | p50 | p95 | p99 | Process peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| F32 Python | 107.881 | 0.740 | 108.000 | 108.735 | 109.565 | 4140 |
| F32 C++ | 113.679 | 1.023 | 113.680 | 115.210 | 115.695 | 3944 |
| BF16 Python, round 1 | 103.264 | 22.820 | 89.743 | 144.473 | 145.654 | 2280 |
| BF16 C++, round 1 | 95.812 | 1.236 | 95.265 | 98.437 | 99.392 | 2370 |
| BF16 C++, round 2 | 89.871 | 6.704 | 93.805 | 97.898 | 98.439 | 2370 |
| BF16 Python, round 2 | 92.276 | 0.526 | 92.132 | 93.439 | 94.156 | 2280 |
| BF16 Python, round 3 | 93.181 | 0.531 | 93.036 | 94.021 | 95.168 | 2280 |
| BF16 C++, round 3 | 95.326 | 0.895 | 95.380 | 96.782 | 97.334 | 2370 |

Order was Python→C++ in rounds 1 and 3, C++→Python in round 2. All rounds are
reported; no pooling or best-run selection is used to improve the ratio.
The cause of the first Python batch's larger variation was not isolated.

F32 C++ was **5.4% slower**, with **4.7% lower process memory**. BF16 C++/Python
mean ratios were approximately **0.928, 0.974, 1.023** across the three rounds:
there is no consistent latency advantage. BF16 C++ used **3.9% more** process
VRAM in every round. These results do not support the old large X-VLA speedup
or memory-reduction claims. The README's historical-normalized row remains
Pending; this independently sourced comparison is linked separately.

## Environment and reproducibility

Python inference used the existing project environment
`/home/xuling/yangzhixiao/xr0_pyenv2`; the shared C++ client used the existing
LIBERO environment. Exact Torch/Transformers/NumPy versions, commands, raw
samples, source hashes, and per-episode Q6 records are in the
[evidence JSON](q6_and_xvla_evidence_20260904.json).

The official environment already contains import shims for unused FastAPI,
uvicorn and json_numpy web-serving imports. The benchmark does not use those
routes or replace model inference. The LIBERO environment lacks these imports;
its initial official-model launch failed before inference. No global packages
or upstream source files were modified to bypass that failure.

PIL image preprocessing is pinned separately from tokenizer selection: a
valid fast-tokenizer snapshot need not contain `merges.txt`. The new predictor
test checks in-call preprocessing, state padding, domain, bool/int dtypes,
seed mismatch rejection, repeated actions and preservation of caller RNG state.

Raw local artifacts and reproducible run scripts are under
`outputs/vla_unified_20260904/`. New model files were temporary derivatives;
the original HF snapshot and historical GGUF were preserved.

Final verification: 50 related regression tests passed without skips; changed
Markdown links and `git diff --check` passed. The complete raw archive is
`outputs/vla_unified_20260904/q6-xvla-evidence-complete.tar.gz`, SHA-256
`ada0c1dda65150cb3d2586788841f86ea6372acb6bbb875cabf9f0f55a05b0a4`.
After verifying the local copy, the dedicated server temporary directory
(4.5 GiB, including all three generated GGUFs) was removed. These derivatives
can be regenerated from preserved sources; raw evidence remains local. No GPU
compute processes remained. No commit or push was performed.
