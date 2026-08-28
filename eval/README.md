# Evaluation Helpers

The original vla.cpp multi-dataset evaluation harness has been trimmed to the
datasets and model families maintained in embodied.cpp. Current evaluation code
targets LIBERO and RoboTwin with `pi05`, `groot_n1`, `hy_vla`, and `lingbot_va`
paths.

The adapter boundary now owns simulator-specific observation parsing:

```text
simulator / dataset observation
  -> adapter.typed_io.EmbodiedObservation
  -> adapter.pipeline.AdapterPipeline
  -> client request payload
```

Important paths:

- `client/`: direct and server-backed evaluation clients.
- `client/run_pi05_smoke.py`: simulator-independent deterministic pi0.5
  validation client for Jetson and other CUDA hosts. Its checked fixture is
  `fixtures/pi05_smoke.json`. See `JETSON_ORIN_PI05_VALIDATION.md` for the
  Jetson AGX Orin build, functional, latency, and resource report.
- `client/run_robolab_eval.py`: native Cosmos3 RoboLab entry point. It launches
  the C++ `wam-server`, registers RoboLab DROID tasks, and sends observations to
  WAM over ZMQ/protobuf without calling `RoboLab/policies/cosmos3/run.py`.
- `conf/robolab_cosmos3_eval.yaml`: official RoboLab Cosmos3 rollout defaults
  forwarded to C++ as `cosmos3.*` WAM step params.
- `../scripts/check_cosmos3_native_readiness.py`: lightweight readiness entry
  point.  It always runs the native boundary guard and can optionally run the
  server-side visual block0 capture alignment gate.  The default guard also
  checks that the YAML policy defaults, WAM step params, and C++ validation
  keys stay aligned.
- CMake target `cosmos3-native-readiness`: runs the default local-safe readiness
  guard from a configured build directory.
- CMake target `cosmos3-layer0-trace-gate`: server-side Cosmos3 regression gate
  for the CUDA build. It depends on `wam-server`, uses GPU 2, refreshes the
  native WAM layer0 trace and PyTorch/vLLM reference capture, then compares them
  with L1 `0.01` and Linf `0.05` thresholds.
- `client/lingbot_world_client.py`: ZeroMQ/Protobuf client for
  `wam-lingbot-server`.
- `sim/libero/`: LIBERO setup and local runtime.
- `sim/robotwin/`: RoboTwin setup and HY-VLA native evaluation path.
- `sim/robolab/`: RoboLab setup and Cosmos3 PyTorch integration notes.
- `run_robolab_cosmos3_pytorch.sh`: registered entry point for starting the
  Cosmos3 PyTorch server and RoboLab rollout through its OpenPI WebSocket ABI.
- `../adapter/sim/libero.py`: LIBERO typed I/O adapter.
- `../adapter/sim/robotwin.py`: RoboTwin typed I/O adapter.

## Deterministic pi0.5 smoke validation

Validate the checked fixture without loading a model or importing inference
frameworks:

```bash
python3 eval/client/run_pi05_smoke.py validate
```

For a functional run, start the pi0.5 server on the validation-only loopback
port with phase timing enabled:

```bash
build/pi05-sm87/vla-server \
  --bind tcp://127.0.0.1:15555 \
  --timing-detail phase \
  /path/to/pi05-mmproj.gguf \
  /path/to/pi05.gguf
```

Run the smoke client on the same Jetson host as `vla-server`; the validation
endpoint is intentionally restricted to loopback addresses.

Then send one warm-up request followed by three identical measured requests:

```bash
python3 eval/client/run_pi05_smoke.py run \
  --protocol functional \
  --endpoint tcp://127.0.0.1:15555 \
  --output /path/to/results/runtime/pi05-functional.json
```

The request uses two generated 224x224 RGB images, fixed tokens and state, and
a complete deterministic 1600-value float32 noise payload. The client checks
the response ID, `[50, 32]` action shape, finite values, and bit-identical
outputs, and records client/server timing summaries. It needs only `pyzmq` at
runtime and encodes the existing `serving/vla.proto` wire contract without a
Python protobuf installation. Keep `pyzmq` in the isolated validation
environment rather than changing system Python packages.

The current pi0.5 implementation reports aggregate inference timing but does
not populate the prefill and denoise phase fields. See
`JETSON_ORIN_PI05_VALIDATION.md` for how those fields are interpreted in the
Jetson results.

Run the GR00T N1.7 LIBERO-object sample after starting
`vla-server` on `tcp://127.0.0.1:5555`:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_groot_n1_eval.yaml
```

On the RTX 5060 Laptop validation machine, task 0
(`pick_up_the_alphabet_soup_and_place_it_in_the_basket`) completed successfully
in 144 environment steps with the all-BF16 strict-parity configuration. This is
a single integration sample, not a benchmark success-rate claim.

GR00T N1.7 loads its action-head matmul weights as BF16 by default, matching the
released `*-bf16.gguf` checkpoints. Set `VLA_GROOT_WEIGHT_DTYPE` only when you
want to override that runtime dtype, for example `VLA_GROOT_WEIGHT_DTYPE=q4_K`
for a quantized action-head path.

GR00T N1.7 disables backbone flash attention by default for validation-oriented
runs. Set `VLA_GROOT_FLASH_ATTN=1` before starting `vla-server` to opt into the
faster fused path.

For SmolVLA, the checked-in `libero_smolvla_eval.yaml` is an integration
configuration rather than a benchmark claim. During bring-up on an RTX 4060
Laptop, one seed each for LIBERO-object tasks 0, 1, and 2 completed
successfully at steps 149, 133, and 155 respectively.

## SmolVLA acceptance benchmark

The acceptance protocol is deliberately larger than the integration sample:
`libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`, with all ten
tasks and ten episodes per task. This is 400 episodes for the C++ runtime and
another 400 episodes for the official LeRobot SmolVLA implementation. Both use
relative control, fixed LIBERO initial states, one replayed action per model
request, and ten flow-matching steps.

With the C++ SmolVLA server already listening on `tcp://127.0.0.1:5566`, run:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_smolvla_acceptance.py --run cpp
```

Run the official GPU baseline from the LeRobot environment:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_smolvla_acceptance.py --run official \
  --lerobot-eval /root/smolvla-port/bin/lerobot-eval
```

The runner skips tasks that already have complete result files, so interruption
only causes the currently incomplete task to be replayed. C++ resume validation
checks the architecture, suite, task, environment seed, action-noise seed, and action
replay horizon. Official results additionally carry a sidecar `run_metadata.json`
with the suite, task, episode count, seed, policy path, action horizon, and flow
step count; stale or metadata-less official results are rerun and excluded from
aggregation, so results from a different protocol are not mixed.
Regenerate the comparison without launching rollouts with `--run none`. It writes
`outputs/smolvla_acceptance/report.json` and a Markdown table beside it. A suite
passes the parity guard when its C++ success rate is no more than five percentage
points below the locally executed official baseline; change this explicit
tolerance with `--tolerance` if an acceptance contract specifies another value.

The three successful one-episode tasks documented above are not a complete
benchmark result and do not satisfy these acceptance criteria.

For a PR smoke benchmark, select one task per suite and three episodes:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_smolvla_acceptance.py \
  --run all --task-ids 0 --episodes 3
```

This sets `selected_protocol_complete` when all 24 rollouts (12 C++ and 12
official) exist. It intentionally does not set the full-benchmark acceptance
flags, which require all ten tasks and ten episodes per task.

Run the lightweight regression suite with the repository's `gguf-py` package
on `PYTHONPATH`:

```bash
PYTHONPATH=third_party/llama.cpp/gguf-py \
  python -m unittest discover -s tests -v
```
