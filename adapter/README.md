# Adapter Boundary

`adapter/` is the first runtime boundary that separates embodied I/O from the
model-specific code inherited from `vla.cpp`.

The intent is to make sensors, datasets, simulators, and robot platforms speak a
typed embodied interface first, then translate that interface into the current
model ABI only at the edge.

Current pieces:

- `typed_io.h` defines typed observations, images, tensors, action chunks, and
  deployment commands.
- `adapter.h` defines input and output adapter interfaces.
- `VlaModelInputAdapter` bridges typed observations into the existing
  `vla::Inputs` struct so the current models keep running during the refactor.
- `DirectActionOutputAdapter` wraps action chunks into deployment commands.
- `typed_io.py` and `pipeline.py` provide the same boundary for Python
  simulator and dataset evaluation code while preserving existing client input
  dictionaries during migration.
- `sim/robotwin.py` contains the RobotWin observation/action adapter. The old
  `eval/utils/sim_adapters/robotwin.py` path remains as a compatibility import.
- `sim/libero.py` contains the LIBERO parser registry and simulator adapter.
  The old `eval/utils/sim_adapters/libero.py` path remains as a compatibility
  import.

This maps to the third design principle in the report:
`Typed embodied I/O and deployment adapters`.

Near-term migration path:

1. Move simulator-specific parsing from `eval/client` and
   `eval/utils/sim_adapters` behind this boundary.
2. Let `serving/server.cpp` decode wire requests into `adapter::Observation`,
   then call `VlaModelInputAdapter` instead of filling `vla::Inputs` directly.
3. Add deployment adapters for ROS/ROS2, Isaac Sim, and dataset replay without
   changing model implementations.
