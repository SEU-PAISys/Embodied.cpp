# Adapter Boundary

`adapter/` is the first runtime boundary that separates embodied I/O from the
model-specific code inherited from `vla.cpp`.

The intent is to make sensors, datasets, simulators, and robot platforms speak a
typed embodied interface first, then translate that interface into the current
model ABI only at the edge. It is a source-level boundary only, so it does not
require extra runtime configuration or a separate service.

Current C++ path:

```text
serving/server.cpp
  -> adapter::Observation
  -> VlaModelInputAdapter
  -> vla::Inputs
  -> model predict()
```

Current Python evaluation path:

```text
simulator / dataset observation
  -> adapter.typed_io.EmbodiedObservation
  -> adapter.pipeline.AdapterPipeline
  -> existing client request dictionaries
```

Current pieces:

- `typed_io.h` and `typed_io.py` define typed observations, images, tensors,
  action chunks, and deployment commands.
- `adapter.h` and `adapter.cpp` define C++ input and output adapter interfaces.
- `VlaModelInputAdapter` bridges typed observations into the existing
  `vla::Inputs` struct so the current models keep running during the refactor.
- `DirectActionOutputAdapter` wraps action chunks into deployment commands.
- `pipeline.py` provides the same boundary for Python simulator and dataset
  evaluation code while preserving existing client input dictionaries during
  migration.
- `sim/robotwin.py` contains the RobotWin observation/action adapter.
- `sim/libero.py` contains the LIBERO parser registry and simulator adapter.

This maps to the third design principle in the report:
`Typed embodied I/O and deployment adapters`.

Near-term migration path:

1. Keep simulator-specific parsing in `adapter/sim` rather than in eval clients.
2. Let `serving/server.cpp` decode wire requests into `adapter::Observation`,
   then call `VlaModelInputAdapter` instead of filling `vla::Inputs` directly.
3. Add deployment adapters for ROS/ROS2, Isaac Sim, and dataset replay without
   changing model implementations.
