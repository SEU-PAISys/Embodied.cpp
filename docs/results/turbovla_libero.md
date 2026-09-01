# TurboVLA — Full LIBERO Evaluation

Runtime: Embodied.cpp C++/GGML direct V+L→A inference · Reference:
official H-EmbodVis/TurboVLA PyTorch implementation (paper avg **97.7%**).

## Protocol

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
