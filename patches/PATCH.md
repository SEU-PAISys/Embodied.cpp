# llama.cpp Patch Series

Embodied.cpp keeps `third_party/llama.cpp` out of Git and reconstructs it from a
pinned upstream revision plus an ordered patch series. Run:

```bash
./patches/init_third_party.sh
```

The script checks out `LLAMA_REF` (currently `b9016`) and applies each patch
idempotently. A patch that is already present is detected with
`git apply --reverse --check`.

## Application Order

Patches are order-dependent and must remain in this sequence:

1. `llama.cpp-pi05.patch`
   - Adds the PaliGemma vision projector used by pi0.5.
   - Carries the existing shared MTMD batching and vision-runtime compatibility
     changes on which the current pi0.5 path depends.
2. `llama.cpp-groot-n1.patch`
   - Preserves the Qwen3-VL `embedding_output_pre_norm` GGUF metadata.
   - Allows GR00T to consume the layer-16 decoder residual before final RMSNorm.
   - Matches Qwen3-VL absolute-position interpolation to Hugging Face
     `align_corners=True` behavior.
   - Adds the opt-in `MTMD_DUMP_PREPROCESSED_DIR` validation dump.
3. `llama.cpp-cuda-parity.patch`
   - Adds the `GGML_CUDA_FAST_MATH` build option.
   - Makes `GGML_CUDA_FORCE_CUBLAS` cover the MMF and MMVF kernels.
   - Makes `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F` disable TF32 on the cuBLAS
     handle for strict tensor comparison.

The first patch is the established pre-GR00T project patch. The latter two are
kept separate so Qwen3-VL compatibility and optional CUDA numerical controls can
be reviewed or upstreamed independently.

## Overrides

The default series can be replaced without editing the setup script:

```bash
# Backward-compatible single patch
LLAMA_PATCH=/path/to/combined.patch ./patches/init_third_party.sh

# Ordered colon-separated patch list
LLAMA_PATCHES=/path/one.patch:/path/two.patch ./patches/init_third_party.sh
```

A missing patch or a patch that does not apply to `LLAMA_REF` is a hard error.
The setup must not silently build a partially patched runtime.

## Updating a Patch

1. Keep `LLAMA_REF` fixed while changing a patch.
2. Start from a clean checkout of that revision.
3. Apply all earlier patches in order.
4. Make only the changes owned by the patch being updated.
5. Export the delta with `git diff --binary`.
6. Apply the complete series to another clean checkout and build the affected
   targets.

Do not commit files under `third_party/llama.cpp`; that directory is generated
and ignored. The tracked patch files are the source of truth.

A clean application check can be run with a temporary worktree:

```bash
root=$PWD
check_tree=$(mktemp -d)
git -C third_party/llama.cpp worktree add --detach "$check_tree" HEAD
for patch in \
    patches/llama.cpp-pi05.patch \
    patches/llama.cpp-groot-n1.patch \
    patches/llama.cpp-cuda-parity.patch; do
    git -C "$check_tree" apply --check "$root/$patch"
    git -C "$check_tree" apply "$root/$patch"
done
git -C third_party/llama.cpp worktree remove --force "$check_tree"
```

After changing the GR00T patches, rebuild and run the focused validators:

```bash
cmake --build build-cuda --target vla-groot-n1-server -j2

GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1 \
python scripts/validate_groot_n1_backbone.py --max-relative-error 0.05

GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1 \
python scripts/validate_groot_n1_official.py \
  --cpp-weight-dtype bf16 --max-relative-error 0.05
```

## Scope Rules

- Prefer opt-in metadata, build options, or environment flags over global
  behavior changes.
- Keep validation-only output behind an environment variable.
- Put unrelated model families in separate patches.
- Changes generally useful to llama.cpp should be proposed upstream; keep the
  local patch until the pinned upstream revision includes them.
- Regenerate and revalidate the series whenever `LLAMA_REF` changes.

## Licensing

llama.cpp is distributed under the MIT License, which permits modification and
redistribution provided its copyright and license notice are retained. The
Embodied.cpp patch contributions carry Apache-2.0 headers; applying them does
not replace or remove llama.cpp's MIT license.
