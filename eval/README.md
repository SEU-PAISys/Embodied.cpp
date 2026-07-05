# Evaluation Helpers

The original vla.cpp multi-dataset evaluation harness has been trimmed to the
datasets and model families maintained in embodied.cpp. Current evaluation code
targets LIBERO and RoboTwin with `pi05`, `hy_vla`, and `lingbot_va` paths.

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
