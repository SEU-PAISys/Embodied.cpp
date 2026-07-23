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

Run the GR00T N1.7 LIBERO-object sample after starting
`vla-groot-n1-server` on `tcp://127.0.0.1:5555`:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf eval/conf/libero_groot_n1_eval.yaml
```

On the RTX 5060 Laptop validation machine, task 0
(`pick_up_the_alphabet_soup_and_place_it_in_the_basket`) completed successfully
in 144 environment steps with the all-BF16 strict-parity configuration. This is
a single integration sample, not a benchmark success-rate claim.
