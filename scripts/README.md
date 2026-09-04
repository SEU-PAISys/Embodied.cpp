# Conversion and Quantization Scripts

This directory contains the public tools for converting supported upstream
checkpoints to GGUF and for creating selected quantized variants. Run commands
from the repository root.

## Prerequisites

Initialize the vendored `llama.cpp` tree before conversion so the Python GGUF
package is available:

```bash
./patches/init_third_party.sh
```

Conversion scripts require a Python environment with `torch`, `numpy`, and
`safetensors`. Q8_0/Q4_0 storage quantization uses vendored `gguf-py` directly;
Q6_K additionally loads `libggml-base.so` from a configured build. Use the
library from the same build configuration that will run the resulting model.

Each converter accepts `--help`. Run its `--dry-run` mode first where available
to validate paths and tensor mappings before writing a large GGUF file.

## Choose a Workflow

| Model | Conversion scripts | Optional quantization |
|---|---|---|
| pi0.5 | `convert_pi05_to_gguf.py`, `convert_pi05_mmproj_to_gguf.py` | Output type selected during conversion |
| GR00T N1.7 | `prepare_groot_n1_backbone.py` | Output type selected during conversion |
| HY-VLA | `convert_hy_vla_to_gguf.py` | `quantize_hy_vla_gguf.py` |
| LingBot-VA | `convert_lingbot_va_to_gguf.py` | `quantize_lingbot_wan_gguf.py` |
| Cosmos3-Nano | `convert_cosmos3_full_w8_to_gguf.py` | Use the upstream full_w8 bundle |
| SmolVLA | `convert_smolvla_to_gguf.py`, `convert_smolvla_mmproj_to_gguf.py` | Output type selected during conversion |
| Xiaomi-Robotics-0 | `convert_xr0_to_gguf.py` | `quantize_xr0_gguf.py` (q8_0/q6_k/q5_k/q4_k) |
| TurboVLA | `convert_turbovla_to_gguf.py` | `quantize_vla_gguf.py` (q8_0/q6_k/q4_0; storage quantization) |
| X-VLA | `convert_xvla_to_gguf.py` | `quantize_vla_gguf.py` (q8_0/q6_k/q4_0; storage quantization) |

Place final artifacts under the `checkpoints/` layout shown in the top-level
README, then use the matching build and evaluation configuration.

## pi0.5

pi0.5 uses two GGUF files: the policy checkpoint and the PaliGemma vision
projector. Convert both from the same LeRobot checkpoint directory:

```bash
python scripts/convert_pi05_to_gguf.py \
  --ckpt <PI05_CHECKPOINT> \
  --out checkpoints/pi05/pi05.gguf \
  --outtype bf16

python scripts/convert_pi05_mmproj_to_gguf.py \
  --ckpt <PI05_CHECKPOINT> \
  --out checkpoints/pi05/pi05-mmproj.gguf \
  --outtype bf16
```

The checkpoint directory must include `model.safetensors`, `config.json`, and
the policy processor metadata required for normalization statistics.

## GR00T N1.7

`prepare_groot_n1_backbone.py` produces the complete three-file deployment:
the Qwen3-VL text backbone, the matching mmproj, and the GR00T action head.

```bash
python scripts/prepare_groot_n1_backbone.py \
  --checkpoint <GROOT_CHECKPOINT> \
  --output-dir <PREPARED_QWEN3VL_DIR> \
  --gguf-dir checkpoints/groot-n1 \
  --outtype q8_0 \
  --ggml-lib <BUILD_DIR>/bin/libggml-base.so
```

Use `--reuse-prepared` on subsequent quantization runs to reuse the prepared
local Qwen3-VL files. The script writes:

```text
qwen3vl-backbone-<TYPE>.gguf
qwen3vl-mmproj-<TYPE>.gguf
groot-n1.7-libero-object-action-head-<TYPE>.gguf
```

## HY-VLA

Convert a HY-VLA checkpoint into a combined GGUF. Use `--scope full` for a
deployable model; the narrower scopes are intended for mapping validation.

```bash
python scripts/convert_hy_vla_to_gguf.py \
  --ckpt <HY_VLA_CHECKPOINT> \
  --out checkpoints/hy-vla/hy_vla_full_bf16.gguf \
  --scope full
```

Create a selective quantized variant with:

```bash
python scripts/quantize_hy_vla_gguf.py \
  --input checkpoints/hy-vla/hy_vla_full_bf16.gguf \
  --output checkpoints/hy-vla/hy_vla_full_q4_k.gguf \
  --qtype q4_K \
  --ggml-lib <BUILD_DIR>/bin/libggml-base.so
```

The quantizer keeps statistics, normalization tensors, biases, and small
projections in their source dtype.

## LingBot-VA

The LingBot converter accepts an upstream checkpoint root and can write the
transformer, text-encoder, and VAE-related modules selected by `--modules`.
Inspect the current module choices before conversion:

```bash
python scripts/convert_lingbot_va_to_gguf.py --help
```

Use `--dry-run` to validate the checkpoint layout first. Quantize only the Wan
transformer matmul weights after conversion:

```bash
python scripts/quantize_lingbot_wan_gguf.py \
  --input <LINGBOT_TRANSFORMER_GGUF> \
  --output <LINGBOT_TRANSFORMER_Q4_K_GGUF> \
  --qtype q4_K \
  --ggml-lib <BUILD_DIR>/bin/libggml-base.so
```

## Cosmos3-Nano

Convert the official RoboLab `full_w8` bundle with the Wan VAE encoder. The
encoder is required by the native RoboLab evaluation path.

```bash
python scripts/convert_cosmos3_full_w8_to_gguf.py \
  <COSMOS3_FULL_W8_BUNDLE> \
  --out checkpoints/cosmos3/cosmos3_robolab_full_w8_with_vae_encoder.gguf \
  --include-vae-encoder
```

Use the resulting GGUF with `eval/conf/robolab_cosmos3_eval.yaml`.

## SmolVLA

SmolVLA uses two GGUF files: the policy file containing the SmolLM2 backbone,
flow-matching action expert, connector, and normalization statistics, plus a
SigLIP projector file. Convert both files from the same LeRobot checkpoint:

```bash
python scripts/convert_smolvla_to_gguf.py \
  --ckpt <SMOLVLA_CHECKPOINT> \
  --out checkpoints/smolvla/smolvla.gguf

python scripts/convert_smolvla_mmproj_to_gguf.py \
  --ckpt <SMOLVLA_CHECKPOINT> \
  --out checkpoints/smolvla/mmproj-smolvla.gguf
```

Run `--help` to inspect optional dtype and validation flags. The converter
checks the VLM/action-expert layer topology and preserves the serialized
processor metadata required by the LIBERO client.

## Xiaomi-Robotics-0

Xiaomi-Robotics-0 uses a policy GGUF (Qwen3-VL-4B backbone + DiT
flow-matching action head) plus a llama.cpp Qwen3-VL mmproj file converted
from the same checkpoint:

```bash
python scripts/convert_xr0_to_gguf.py \
  --checkpoint <XIAOMI_ROBOTICS_0_HF_DIR> \
  --output checkpoints/xr0/xr0.gguf \
  --mmproj checkpoints/xr0/xr0-mmproj.gguf
```

Keep the source HF snapshot for client-side tokenization and pass
`--tokenizer <XIAOMI_ROBOTICS_0_HF_DIR>` to the eval client. The converter
writes GGUFs, not `checkpoints/xr0/hf`; that YAML path is only an example.

K-quantize selected big matmul weights in the backbone and DiT action head
while keeping norms, embeddings and other unselected tensors at source precision:

```bash
python scripts/quantize_xr0_gguf.py \
  --input checkpoints/xr0/xr0.gguf \
  --output checkpoints/xr0/xr0-q8_0.gguf \
  --outtype q8_0        # q8_0 | q6_k | q5_k | q4_k
```

Parity tools: `tools/xr0_parity.cpp`, `scripts/parity_xr0_reference.py`,
`scripts/parity_xr0_compare.py`.

## TurboVLA

TurboVLA converts to one self-contained GGUF (DINOv3 ViT + BERT +
bidirectional cross-attn fusion + ACT decoder). The bundled WordPiece vocab
is required so the server can tokenize raw instructions:

```bash
python scripts/convert_turbovla_to_gguf.py \
  --ckpt <TURBOVLA_CHECKPOINT.pth> \
  --vocab <TURBOVLA_VOCAB.TXT> \
  --out checkpoints/turbovla/turbovla.gguf
```

Parity scripts: `scripts/parity_turbovla_reference.py`,
`scripts/parity_turbovla_cpp.py` (final-action acceptance `atol=0.01`).

Use `--norm-gguf` to read the exact converted normalization arrays. The
reference explicitly selects `model_state_dict` (or `--checkpoint-key` for
an EMA release). Transformers **4.57.1** was validated for the post-final-LN
DINO features used by this runtime; 4.57.6 changes `hidden_states[-1]` to
pre-LN features, and the reference now refuses that semantic mismatch.
The reference fixture records the dependency versions and weight hashes.

For optional stage diagnostics, start the server with
`VLA_TURBOVLA_DUMP_DIR=<existing parity directory>`, then pass `--stages` to
the comparison script. Each request replaces the dumps. Disable this variable
for performance measurements. Reconvert historical GGUFs missing
`turbovla.pad_layout_instr` / `turbovla.pad_layout_len` to preserve the
checkpoint's per-instruction BERT encoding lengths.

`scripts/rollout_turbovla_reference.py` connects the official model to the
same LIBERO adapter and episode loop as the C++ client: native 256px dual
views, 8-D state, 12-action replay, relative control. This is a shared-protocol
model comparison, not the official release-policy CLI. The common C++
runner's `--no-video` option retains results without writing videos.

## X-VLA

X-VLA converts to a single policy GGUF (Florence-2 DaViT vision + BART
encoder + domain-conditioned flow head):

```bash
python scripts/convert_xvla_to_gguf.py \
  --hf-dir <XVLA_HF_SNAPSHOT> \
  --output checkpoints/xvla/xvla-libero.gguf
```

Parity and rollout tooling: `tools/xvla_parity.cpp`,
`scripts/parity_xvla_reference.py`, `scripts/rollout_xvla_reference.py`.

Pass `--tokenizer <XVLA_HF_SNAPSHOT>` to the eval client. It must contain the
matching BART tokenizer assets (`vocab.json`, `merges.txt` and tokenizer
configuration, or a compatible fast-tokenizer snapshot). Conversion does
not create the example `checkpoints/xvla/hf` directory. TurboVLA instead
uses the WordPiece vocabulary embedded in its GGUF; it needs no client tokenizer.

X-VLA conversion retains source F32 values and refuses an existing output.
`VLA_XVLA_F32_WEIGHTS=1` is a **runtime** residency choice, not a converter
rounding switch. A file named `f32` does not prove F32 computation or correct
per-domain matrix layout; see the [controlled repair evidence](../docs/results/takeover_20260903.md).

## Fixed-input deployment timing

`scripts/bench_vla_boundary.py` supports all three public C++ model clients and
the TurboVLA/XR0 references plus the official X-VLA Python model. It consumes a saved fixture (two native
256px CHW views, raw 8-D state, instruction; XR0 also needs matched seeded
30×32 noise, X-VLA fixed 30×20 noise), performs warmup, and records raw samples
plus mean/std/p50/p95/p99. TurboVLA and X-VLA images must be float values in [0, 1].
Use `--backend cpp --server-pid <PID>` for process VRAM, or `--backend python`
with the reference's model paths. `--n 100 --warmup 5` is the default.
Process memory uses 20 **additional untimed** requests after latency sampling
(`--memory-requests 20`); no GPU-memory query runs in the timing window.

The timing boundary is raw CPU input to full CPU actions (TurboVLA 12×7;
XR0 30×32, five flow steps), including preprocessing and device transfers.
For the deployed XR0 F16 mmproj, use `--xr0-vision-dtype f16` on the Python
reference while keeping its text/action policy BF16; record both precisions.
C++ X-VLA uses 30×20 output, 224px preprocessing and `--domain-id 3` by default.
Its Python path uses the official HF model and processor. Use only a GGUF
converted from that exact HF snapshot: an unrelated historical GGUF is not a
valid denominator. `--xvla-precision bf16|f32` selects Python precision; select
the matching C++ residency separately (`VLA_XVLA_F32_WEIGHTS=1` for F32).
The fixture's explicit 30×20 noise must match `--xvla-noise-seed` (default 42)
on the Python device/dtype; a mismatch is rejected before timing. The original
historical X-VLA checkpoint comparison remains Pending even when a new pair
is benchmarked. The Python branch requires the official snapshot's optional
import dependencies as well as compatible Transformers/PyTorch versions.
C++ additionally includes ZMQ transport, so label results as deployment/API
latency, not GPU-only speedup. Do not benchmark while another evaluation is
running. JSON and final actions use new output files; original results are
never overwritten. Unavailable per-process VRAM remains null, not zero.

## Verify Outputs

For TurboVLA and X-VLA **storage** quantization, use:

```bash
python scripts/quantize_vla_gguf.py \
  --input checkpoints/turbovla/turbovla.gguf \
  --output checkpoints/turbovla/turbovla-q8_0.gguf --outtype q8_0
```

The script uses vendored `gguf-py` codecs (and its Python dependencies),
preserves metadata array types, and refuses existing output paths. Supported
outputs are `q8_0`, `q4_0`, and `q6_k`; inputs must be original F32/BF16 GGUFs.
Q6_K reuses the shared GGML quantizer and additionally requires PyTorch and a
built `libggml-base.so` (override its path with `--ggml-lib`). It selects matrix
rows divisible by 256; ineligible tensors retain their original type. Q8_0 and
Q4_0 do not require PyTorch or that shared library.
Both runtimes dequantize these files to **BF16 residency by default**.
Smaller files alone do not establish native low-bit inference or VRAM savings.

Confirm that the generated files are in the expected `checkpoints/` directory,
then load them with the matching server from the top-level README. For a
simulator-level check, use the configuration in `eval/conf/` for the target
model and benchmark.
