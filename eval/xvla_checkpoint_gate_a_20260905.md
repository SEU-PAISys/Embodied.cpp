# X-VLA checkpoint Gate A: the interim HF snapshot package is not a valid LIBERO release baseline

Date: 2026-09-05. This report answers the review question in
`CODEX_REVIEW_HANDOFF_20260904.md` §12–§13. Scope note (Codex audit): the
A/B below loads each model directory via `trust_remote_code`, so weights,
config and remote code change together — the experiment bounds the
**package**, not weights in isolation; pure-weight attribution is not
claimed and is not needed for the decision to switch to official weights.

## Method

Official upstream evaluation only (`2toinf/X-VLA` @ `6bc2513`):

- Server: `deploy.py` loads the model with `torch_dtype=float32` (F32) and
  serves FastAPI `/act`; preprocessing is the official `XVLAProcessor`
  (CLIPImageProcessor 256→224, ImageNet norm), `generate_actions(steps=10)`
  with the official global-`torch.randn` noise start; no external noise.
- Client: official `evaluation/libero/libero_client.py` — 256px views,
  agentview double-flip, 20-dim proprio (`[pos3, ori6d6, grip0]` + zero legacy,
  proprio[:9] updated from the predicted last action), `domain_id=3`,
  full-chunk open-loop execution, gripper binarized at 0.5, seed-42 init
  states, 10 settle steps, `use_delta=False`.
- Suite: `libero_spatial`, all 10 tasks × 1 episode (`--eval_time 1`).

Server environment: transformers 4.51.3 (NOT 4.57.6, which returns pre-final-LN
DINO features), torch 2.14/CUDA on GPU 1. Client environment: the pinned LIBERO
venv (torch 2.5.1+cu124, robosuite 1.4.0, numpy 1.26.4).

## Weight identity

Per-tensor comparison (`compare_tensors.py`, safetensors, float64 diffs):

| File | SHA-256 | vs official Libero |
|---|---|---|
| Official `2toINF/X-VLA-Libero` | `260cc58869125b826e93bcaa60bca3ea37bcc7aaf893d368a991ca14fde9f0c8` | — |
| Official `2toINF/X-VLA-Pt` | `433acffc…` | different release (foundation) |
| Server snapshot `~/yangzhixiao/xvla_hf` | `3f16a4b67a1d2675fc1cb0350c6d5617522e452f47359e729d74d76b8aa4835b` | **901/903 tensors differ** (action_decoder.fc.weight max diff 1.91; MLP weights 0.27–0.57; 2 tensors equal) |

All three files have identical byte size (3,519,068,172) — same tensor set and
layout; the snapshot is not a serialization artifact but a different set of
weights. The snapshot's `modeling_xvla.py` / `processing_xvla.py` /
`action_hub.py` are byte-identical to the official Libero repo; its
`transformer.py` (+bf16-safe matmul) and `modeling_florence2.py` (+timm
removal) were locally modified; its `config.json` is a 1945-byte Pt-style
trimmed config (official Libero: 6261 bytes) with no provenance (no model
card, no HF cache entry, no download record).

## Closed-loop results (official evaluator)

| Weights | libero_spatial (10 tasks × 1 ep) | Result |
|---|---|---|
| Official `260cc588…` (positive control) | per-task successes all logged | **10/10 = 100%** |
| Snapshot `3f16a4b6…` | same tasks, same seeds, same rig | **0/10 = 0%** |

Official claim for X-VLA-Libero is 98.1%; the 100% positive control validates
the rig. The snapshot fails every episode under the identical protocol.

## Same-source pair rebuilt on official weights

Converted with `scripts/convert_xvla_to_gguf.py` from the official weights:
GGUF SHA-256 `2c828fe612c76db99ebf0fe66d1baec7ffcfcea0be904721dcad415b22e8d9b3`.

Protocol: `run_sim_client_direct.py`, spatial task 0, 10 episodes,
seed 42 / noise-seed 42, no derive, 256px, no video:

| Side | Weights | Result |
|---|---|---|
| C++ server + GGUF `2c828fe6…` | official | **10/10 = 100%** |
| Python public entry (`--implementation python --hf-dir`) | official `260cc588…` | **10/10 = 100%** |

**Noise-protocol correction (Codex audit, 2026-09-05)**: without
`--derive-episode-noise` the two sides did NOT share one noise sequence —
the C++ client falls back to observation-hash noise while the Python
reference client re-created a NumPy RNG from the constructor seed 42. The
10/10 pairs above therefore validate each implementation independently
against the official weights; they are not a same-noise matched pair. The
noise-selection asymmetry has been fixed (both sides now gate on the
reset-time seed; a bare reset uses observation-hash on both sides), and the
main matrix uses `--derive-episode-noise`, under which both sides consume
element-identical episode-derived noise (unit-tested; per-request
`noise_checksum` is now recorded in both clients' inference profiles).

## Conclusions

1. The `3f16a4b6…` snapshot **package** (weights + config + locally modified
   remote code, loaded together via `trust_remote_code`) is not a valid
   release baseline for the tested LIBERO protocol: 0/10 official-evaluator
   episodes vs 10/10 for the official package, with 901/903 tensors
   differing. This is a package-level A/B — the experiment does NOT isolate
   weights from config/code, and pure-weight attribution is not claimed.
   Statistically, if the snapshot truly matched the official 98.1%
   per-episode success rate, observing 10 consecutive failures has
   probability (1 − 0.981)^10 ≈ 6.1 × 10⁻¹⁸.
2. Gate A verdict per the review stands: stop all noise-side investigation
   of the old 0% results and switch the baseline to official weights.
3. The snapshot and its derived GGUF (`02941fd1…`) are quarantined; all
   snapshot-derived parity/timing/VRAM/latency numbers must not be pooled
   with historical valid-weight numbers and must be redone on official
   weights before any release claim.
4. The full 400+400 matched matrix is being rebuilt on the official weights
   (`260cc588…` / GGUF `2c828fe6…`) under the derive-episode-noise protocol,
   with aggregation-level unique-(suite, task) coverage gating
   (`aggregate_eval_summary.py --require-full-matrix`).
5. Closed (P2): snapshot provenance — "unknown origin, not a release
   baseline". The locally modified `transformer.py`/`modeling_florence2.py`
   traveled inside the snapshot package and are part of why the package
   fails; official-repo code is the reference for all rebuild work.

Raw evidence on the server (temp, regenerable): `/tmp/xvla_gatea/` —
`gateA_official/`, `gateA_ours/`, `step2/`, `phase1/` (server+client logs,
per-episode results, parity runs), `official_libero/` (weights + config),
`xvla-libero-official.gguf`, `tensor_compare_output.txt`, driver scripts
`gate_a_run.sh`, `step2_matched.sh`, `phase1_preflight.sh`.
Local copy of the official safetensors: `.gate_a/libero_model.safetensors`.

## Phase 1 preflight on official weights (derive-episode-noise protocol)

After the Codex plan round (2026-09-05), the noise-selection asymmetry was
fixed (`XVLAReferenceClient._resolve_noise` now gates on the reset-time seed,
mirroring the C++ client; both clients record per-request `noise_mode`,
`noise_seed` and `noise_checksum` in their inference profiles and per-episode
records). Preflight — official weights, spatial task 0, 10 episodes,
seed 42 / noise-seed 42 / `--derive-episode-noise`, 256px:

| Group | Success | Derived seeds match other side | First-request noise checksum |
|---|---|---|---|
| Python F32 | 10/10 | yes (all 10 episodes) | `34623cb6f8c09153` |
| C++ F32 | 10/10 | yes | `34623cb6f8c09153` |
| Python BF16 | 10/10 | yes | `34623cb6f8c09153` |
| C++ BF16 | 10/10 | yes | `34623cb6f8c09153` |

Fixed-input parity (existing `bench_vla_boundary.py` fixture methodology,
real LIBERO initial observation of spatial task 0):

| Precision | max abs | mean abs | values > 0.005 |
|---|---:|---:|---:|
| F32 | 0.000646 | 0.000048 | 0/600 |
| BF16 | 0.010525 | 0.001150 | 49/600 |

The BF16 figure exceeds the historical 0.005 gate. That gate was calibrated
on the invalid snapshot's weights; the C++/Python BF16 compute paths are
documented as non-identical (C++ keeps F32 activations and normalization with
BF16 matrix weights, Python uses full-BF16 tensors), so the fixed-input
divergence magnitude is weight-dependent. A synthetic-noise fixture gave
0.0167 and the real-observation fixture 0.0105 — the same order, so this is
not fixture pathology. Both BF16 sides pass 10/10 closed-loop, which rules
out protocol-level breakage (a mapping bug produces O(1) errors). The BF16
divergence is recorded here and flagged for the reviewer to re-confirm or
re-calibrate the BF16 gate; the F32 gate passes with a wide margin.

Aggregation gating added in this round: `aggregate_eval_summary.py
--require-full-matrix` hard-fails on duplicate (already), missing, or
unexpected (suite, task) keys — the 394/400 lesson is now enforced in the
shared aggregation path, with unit tests (37 integration + 13 aggregation
tests pass).
