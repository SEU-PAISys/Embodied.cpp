# TurboVLA — Full LIBERO Evaluation

Runtime: Embodied.cpp C++/GGML direct V+L→A inference · Reference:
official H-EmbodVis/TurboVLA PyTorch implementation (paper avg **97.7%**).

Current acceptance status is tracked in the [shared results index](../docs/results/README.md)
and [release audit](../docs/results/release_audit_20260907.md). The historical
Python run labelled FP32 (392/400 in the later audit) used BF16-autocast
vision. It is a mixed-precision result, not the corrected all-FP32 public
Python baseline. The corrected public entry has now completed **FP32
386/400 (96.5%)** and **BF16 385/400 (96.25%)**, with four suites × ten tasks ×
ten episodes and zero skips; seed 42, native 256px, relative control and
12-action replay. See the [independent audit](../docs/results/completed_runs_audit_20260908.json).
These success rates do not establish performance acceleration.

## Historical seed 7 primary results and 2026-09-03 follow-up

The first table below preserves the historical seed 7 primary results; the
seed 42 post-fix rerun is a separate table, not a replacement. The latter uses
native 256px dual views, raw 8-D state, 12-action replay, ten episodes per task,
and no skipped episodes. One verified
joint/all-four-suite checkpoint is reused for every suite. The raw checkpoint
SHA256 is `787c01bd8b328a5948b756aab92f8058a1e0802845a0e1f24506291b9cda59cf`;
all 669 mapped weights match the historical GGUF. The reconversion preserves
the checkpoint's per-instruction text lengths.

The [V2 performance audit](../docs/results/v2_followup_20260903.md) adds matched post-allocator
100-call benchmarks and a 20-episode-per-side shared-loop control (both 20/20;
C++ 2.147 vs Python 3.637 weighted get_action ms/step). Those measurements
have their own scope and do not replace the success-rate tables below.

The [Q6_K follow-up](../docs/results/turbovla_xvla_q6_followup_20260904.md) adds a separate
30-episode-per-side targeted gate and same-batch latency/VRAM evidence.
Q6 storage still loads as BF16 and is not a native 6-bit inference result.

| Suite | PyTorch¹ | C++ bf16 | C++ q8_0 | C++ q4_k |
|---|---|---|---|---|
| spatial | 98/100 | 99/100 | 100/100 | 95/100 |
| object | 100/100 | 100/100 | 100/100 | 85/100 |
| goal | 97/100 | 97/100 | 99/100 | 82/100 |
| libero_10 | 94/100 | 89/100 | 87/100 | 82/100 |
| **total** | **389/400 = 97.25%** | **385/400 = 96.25%** | **386/400 = 96.50%** | 344/400 = 86.00% |

### Post-fix re-validation (seed 42, per-variant GGUF files, 256 px)

After the attention-mask / padding-metadata / rendering fixes, the three
original variants were re-run over 1200 episodes and, after the builtin
padding-layout fallback landed, all five variants' goal suites were re-run
again (goal rows below use those corrected numbers; per-variant totals are
400 episodes each, 2000 across the five variants):

| Suite | BF16 file | Q8_0 file | Q4_0 file | Q4_K file | Q6_K file |
|---|---:|---:|---:|---:|---:|
| spatial | 98/100 | 97/100 | 96/100 | 100/100 | 96/100 |
| object | 99/100 | 100/100 | 97/100 | 98/100 | 100/100 |
| goal | 97/100 | 97/100 | 83/100 | 86/100 | 98/100 |
| libero_10 | 88/100 | 85/100 | 87/100 | 86/100 | 85/100 |
| **total** | **382/400 (95.50%)** | **379/400 (94.75%)** | **363/400 (90.75%)** | **370/400 (92.50%)** | **379/400 (94.75%)** |

The goal-suite losses recorded on 09-03 (Q4_K 68%, Q6_K 69%) were caused by the
same missing per-instruction padding metadata as the BF16 goal regression, not
by quantization. With the builtin padding-layout fallback
(`models/turbo_builtin_pad_layout.inc`) every variant's goal suite recovers
(97/97/83/86/98). Quantization-accuracy ordering is now monotonic in bit width:
Q8_0 and Q6_K show no detectable difference against BF16 in this sample
(goal 97 vs 98; a paired exact test on the corrected-layout Q6 run gives
p = 0.424, which means the difference was not resolved, not that the
variants are equivalent) and are the recommended quantized configurations;
Q4_0 and Q4_K carry a bounded goal/q4 cost.

These Q8_0/Q4_0 files are dequantized to **BF16 residency**; they do not imply
native low-bit execution or lower runtime VRAM. Q4_0 has a material goal-suite
loss. The old pre-fix BF16 sweep was 328/400, but the full rerun also changes
rendering, text-length metadata and quantization tooling; it is not an isolated
one-variable experiment. Same-360px targeted runs isolate the mask fix.

See the [takeover evidence](../docs/results/takeover_20260903.md) for the defect, three numerical
parity cases, same-protocol Python controls and measured deployment costs.
The following seed-7 archive is a separate experiment, not current-code proof.

## Archived protocol (seed 7)

- Suites: LIBERO-spatial / object / goal / libero_10, all 10 tasks each
- Episodes: 10 per task → 100 per suite, 400 total · seed 7
- Action execution: 12-step open-loop chunks · observation 256×256 dual-view
- Checkpoints: official suite-specific weights from HuggingFace,
  converted to self-contained GGUF via `scripts/convert_turbovla_to_gguf.py`

## Results

| Suite | PyTorch¹ | C++ bf16 | C++ q8_0 | C++ q4_k |
|---|---|---|---|---|
| spatial | 98/100 | 99/100 | 100/100 | 95/100 |
| object | 100/100 | 100/100 | 100/100 | 85/100 |
| goal | 97/100 | 97/100 | 99/100 | 82/100 |
| libero_10 | 94/100 | 89/100 | 87/100 | 82/100 |
| **total** | **389/400 = 97.25%** | **385/400 = 96.25%** | **386/400 = 96.50%** | 344/400 = 86.00% |

¹ Local PyTorch sweep reused the `object` checkpoint across suites
(only conversion available at run time); treat it as an approximate
reference. The authoritative baseline is the official 97.7% average.

## Findings

1. C++ bf16 reaches **96.25%**, within ~1 pp of both the local PyTorch
   sweep and the official paper average; q8_0 is statistically identical
   (**96.50%**) and is the recommended quantized configuration.
   (Phrasing kept from the original seed-7 run; the paired exact test on the
   corrected-layout comparison is reported in the post-fix section above and
   should be read as "difference not resolved", not as proof of equivalence.)
2. q4_k loses ~10 pp concentrated in goal/object; not recommended for
   this model.
3. The residual gap vs the references concentrates in libero_10
   long-horizon episodes.
4. Focused Object validation at 50 episodes scored 98.0% with amortized
   client-side policy latency ≈ **11.9 ms/episode**.

## Numerical agreement with the reference

Fixed-input comparison against the official PyTorch forward pass
(two 256×256×3 images, state dim 1, BERT tokens 8 valid + 3 PAD;
output 12×7 action chunk):

| Stage (PyTorch shape) | max abs | mean abs | cosine |
|---|---:|---:|---:|
| DINOv3 patch tokens `[1,2,256,768]` | 0.05425 | 0.00468 | 0.99992 |
| vision proj + view emb `[1,512,256]` | 0.06452 | 0.01009 | 0.99992 |
| BERT hidden `[1,21,768]` | 0.43326 | 0.01804 | 0.99634 |
| text projection `[1,21,256]` | 1.59619 | 0.07546 | 0.99639 |
| fused vision `[1,512,256]` | 0.10255 | 0.00768 | 0.99991 |
| fused text `[1,21,256]` | 0.01745 | 0.00207 | 0.99997 |
| state tokens `[1,2,256]` | 0.01829 | 0.00238 | 1.00000 |
| **final env action `[1,12,7]`** | **0.00418** | **0.00062** | — |

Final-action agreement is well within the acceptance threshold
(`atol=0.01`); intermediate deviations sit in individual BERT/text tokens
and do not propagate to decoded actions.

## Reproduction pointers

- Conversion: `scripts/convert_turbovla_to_gguf.py --vocab ...`
- Serving: `serving/vla-server` (WordPiece tokenizer built in)
- Parity: `scripts/parity_turbovla_cpp.py`, `scripts/parity_turbovla_reference.py`
- Closed-loop client: `eval/client/run_sim_client_direct.py --arch turbovla`
