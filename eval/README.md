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
- `client/lingbot_world_client.py`: ZeroMQ/Protobuf client for
  `wam-lingbot-server`.
- `sim/libero/`: LIBERO setup and local runtime.
- `sim/robotwin/`: RoboTwin setup and HY-VLA native evaluation path.
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
