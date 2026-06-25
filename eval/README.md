# Evaluation helpers

The original vla.cpp multi-model LIBERO/Simpler/ALOHA evaluation harness has
been removed from this project. It referenced model families that are not part
of embodied.cpp.

Current retained helper:

- `client/lingbot_world_client.py`: ZeroMQ/Protobuf client for
  `lingbot-world-server`.

New evaluation scripts should target the supported embodied.cpp architectures:
`pi05`, `hy_vla`, and `lingbot_va`.
