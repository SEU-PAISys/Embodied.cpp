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
`safetensors`. Quantization scripts additionally load `libggml-base.so` from a
configured build. Use the library from the same build configuration that will
run the resulting model.

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

## Verify Outputs

Confirm that the generated files are in the expected `checkpoints/` directory,
then load them with the matching server from the top-level README. For a
simulator-level check, use the configuration in `eval/conf/` for the target
model and benchmark.
