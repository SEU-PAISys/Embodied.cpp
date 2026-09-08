# Q6 artifact and experiment audit — 2026-09-04

This audit supersedes the interpretation appended in commit `6184bd4`, not
the original raw results. The upstream comparison rules remain those in
[VLA_BENCHMARK_STANDARD.md](../../eval/VLA_BENCHMARK_STANDARD.md).

## What was verified

- Local branch at audit start: `integ-upstream`, `6184bd4`, clean. Its parent
  `566b6c4` committed 29 files; `6184bd4` changed only two reports.
- Server remained at `19176e1` plus working-tree modifications. A checkout
  revision alone does not identify its built executable.
- No GPU compute processes were present at audit start. New controlled runs
  use one GPU and terminate only the server child they own.
- All 40 unique task JSONs per model contain ten counted episodes each, with
  zero skipped episodes. Summing the episode success flags reproduces
  TurboVLA 350/400 and X-VLA 392/400.

## TurboVLA: missing metadata, not missing tensors

The checkpoint `/home/xuling/yangzhixiao/turbovla_hf/object_local_787c01bd.pth`
contains `model_config.text.padding_length_by_instruction`: 40 instructions,
with lengths 21 for spatial/long tasks, 14 for object, and 11 for goal.
The scalar fallback `padding_length=21` exists **alongside** this map.

| Artifact | SHA-256 | Per-instruction layout |
|---|---|---|
| Correct BF16 source | `055613848ce918c95a6f2564020a90fa0f2e40c2fa490881e5b895c8aa95c1df` | Present, 40 entries |
| Q6 in the earlier paired gate | `c253c24c1015165b44ce6b6914db3bd360e1f6419df57ac12dfa7ab5c16f639f` | Present |
| Q6 in the newly reported 400 episodes | `74943e33f861c0aaf9d5b3c3440ed0c514e49440eeeb9f8d19d6214c241e0b1c` | Absent; scalar 21 only |

Both files have 673 tensors. This does not establish equality of GGUF
metadata: `turbovla.pad_layout_instr` and `turbovla.pad_layout_len` are
key-value arrays, not tensor names. The converter already exports the map
and the quantizer already preserves it; using an old source artifact bypassed
both protections. The existing Q6 codec test explicitly checks preservation.

The new sequential control has now reproduced **0/10 with the old layout and
10/10 with the correct layout**, with seed 42, the same GPU, runtime, two
256px views, replay 12 and ten episodes of goal task 2. Both child servers
exited after their respective tests. This new run also verifies that **all
673 tensor payloads are byte-identical**, including their types and shapes.
Metadata differences are the two layout arrays, GGUF key count, and the
descriptive source/norm-stats-source strings. Thus this task's failure is a
configuration problem, not evidence of a Q6-only numerical regression.

The corrected-layout 400-episode sweep completed at **378/400 (94.50%)**:
spatial 96, object 99, goal 98 and libero_10 85. The same-protocol BF16 result
is 382/400 (95.50%): 98, 99, 97 and 88. Paired by episode, the contingency is
373 both-success, 9 BF16-only, 5 Q6-only and 13 both-fail; two-sided exact
McNemar p=0.424. The overlapping Wilson intervals are [91.813%, 96.340%] and
[93.000%, 97.135%]. This does not prove equivalence, but it removes the claimed
goal-specific Q6 collapse. The existing native Q6 quantizer tests passed
**4/4** on the server, including metadata-array and element-type preservation.

## X-VLA: valid count, different noise protocol

The new Q6 file matches the earlier artifact:
`471630d4f37f180d3381cbba9df623ba8254a5c16085a211cbac60116de565da`.
All task results use seed 42, 256px rendering and replay 30, but
`derive_episode_noise=false`, `noise_seed=null`. The earlier paired gate used
explicit episode-derived noise. The 392/400 observation remains valid for its
recorded legacy protocol; comparing it to a different-seed baseline does not
establish an acceptable 1.2-percentage-point loss or formal non-inferiority.
It also does not evaluate the newly matched HF checkpoint.

## Raw evidence and next checks

Server originals remain under `/home/xuling/yangzhixiao/rerun_results/`
(`turbo-q6_k`, `xvla-q6_k`) and `bench_results/`. The actual launch script is
`/home/xuling/yangzhixiao/q6_eval.sh`; it did not request derived episode noise
and enabled default video capture. These runs are not fixed-input latency
benchmarks.

The 80 JSONs, logs and launch script were copied locally to
`outputs/vla_unified_20260904/codex-q6-existing-evidence-20260904.tar.gz`,
SHA-256 `895f07ad2bde7e0d979104b68ce235394e57332c7525eaf0d5fb0621d938ccc3`.
No weights or credentials are included.

Controlled-run driver: `outputs/vla_unified_20260904/audit_q6_resume.py`.
Server run directory: `/tmp/codex-q6-audited-kjj_fak1`. Its manifest records
commands, model/source/binary hashes, metadata differences, episode results
and child-server shutdown. All three phases are complete and every owned child
server exited. The corrected evidence package is stored locally at
`outputs/vla_unified_20260904/codex-q6-corrected-evidence-20260904.tar.gz`,
SHA-256 `6e66557ef4b12180f0301d4140ffcd1cf021325d60b4ba17ee0edb7daa128b6c`.
It contains no checkpoint or credential.
