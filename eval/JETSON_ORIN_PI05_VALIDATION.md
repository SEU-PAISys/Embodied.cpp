# Jetson AGX Orin pi0.5 Validation

This report records the reproducible pi0.5 validation performed for the
Jetson platform work. It covers build compatibility, deterministic functional
behavior, latency, and resource observations. It does not change public APIs,
model behavior, protobuf schemas, or CUDA kernel algorithms.

## Environment

| Item | Value |
| --- | --- |
| Device | NVIDIA Jetson AGX Orin Developer Kit 64GB |
| Architecture | aarch64, CUDA compute capability 8.7 |
| OS | Ubuntu 22.04, JetPack 6 / L4T r36.4.4 |
| CUDA | 12.6 |
| cuDNN | 9.3 |
| CMake | 3.22.1 |
| Compiler | GCC 11.4 |
| Protobuf | 3.12.4, user-local and process-scoped |
| Power mode | MAXN |
| Server and model build source-tree commit | `c4be965d0111c5480a7fdce10ce2dcffacbc169c` |
| llama.cpp commit | `846262d7875dcabf502a150fa3d7b9c770dde7eb` |

The host used an RT kernel and had pre-existing system swap usage. All
application services in the agreed test scope were stopped during each
measurement window and restored afterwards. The results below are therefore
maintenance-window characterization data, not a fully controlled benchmark.

## Model artifacts

The official `SEU-PAISys/Embodied.cpp/pi05_libero_finetuned_v044` artifacts
were used without conversion.

| File | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `pi05.gguf` | 6,114,779,744 | `caee27a696914aeec6e3ca81f4ebac64ea057dc0ac3438ca533ff44fd133e1b8` |
| `pi05-mmproj.gguf` | 832,380,544 | `494353054dcd180a40d14d32d923b0fe7024f0f30a2d530a2b062afb40668700` |
| `tokenizer.model` | 4,264,023 | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |

## Reproduction

Initialize the pi0.5-only llama.cpp patch profile, then configure an explicit
Jetson AGX Orin architecture. JetPack 6 commonly provides CMake 3.22, so the
special `native` architecture value is not used here.

```bash
LLAMA_PATCH_PROFILE=pi05 ./patches/init_third_party.sh

cmake -S . -B build/pi05-sm87 \
  -DCMAKE_BUILD_TYPE=Release \
  -DMODEL_BUILD_VLA_PI05=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_PREFIX_PATH=/path/to/user-local/protobuf/usr \
  -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build/pi05-sm87 --target vla-server --parallel 8
```

Eight build jobs were used in the dedicated validation window. Use a lower
job count on a shared Jetson host when CPU or memory contention matters.

Create a separate validation environment instead of installing packages into
the system or an existing model-inference Python environment:

```bash
python3 -m venv /path/to/smoke-venv
/path/to/smoke-venv/bin/pip install pyzmq==26.4.0
```

Start the server on a validation-only loopback port:

```bash
LD_LIBRARY_PATH=/path/to/user-local/protobuf/usr/lib/aarch64-linux-gnu \
build/pi05-sm87/vla-server \
  --bind tcp://127.0.0.1:15555 \
  --timing-detail phase \
  /path/to/pi05-mmproj.gguf \
  /path/to/pi05.gguf
```

Run the smoke client on the same Jetson host as `vla-server`; the client
intentionally rejects non-loopback validation endpoints.

If protobuf is provided by a user-local prefix, expose its library directory
to this server process only, for example with an inline `LD_LIBRARY_PATH`.
Do not add the prefix to the host-wide linker configuration.

Run the checked functional protocol and the longer latency protocol:

```bash
/path/to/smoke-venv/bin/python eval/client/run_pi05_smoke.py run \
  --protocol functional \
  --endpoint tcp://127.0.0.1:15555 \
  --output /path/to/pi05-functional.json

/path/to/smoke-venv/bin/python eval/client/run_pi05_smoke.py run \
  --protocol performance \
  --endpoint tcp://127.0.0.1:15555 \
  --output /path/to/pi05-performance.json
```

The functional protocol uses one warm-up and three measured requests. The
performance protocol uses three warm-ups and twenty measured requests. Warm-up
requests are excluded from the reported summaries.

## Build results

- `vla-server` built successfully for aarch64 with explicit `sm_87` CUDA code.
- The CUDA library contained 134 `sm_87` cubin entries.
- All inspected dynamic dependencies resolved, including the process-local
  protobuf installation used by this validation workspace.
- The final server size was 205,776 bytes; its SHA-256 was
  `0bce4ba0b4da8c319b024470ee9e46a5e9465a2714127a1a347eb913eef23a7c`.
- The resumed eight-job build completed in 7 minutes 46.95 seconds with a
  maximum build-process RSS of 845,300 KiB. This is diagnostic data because an
  earlier two-job build was interrupted and the resumed build used cached
  objects.

## Functional results

Two complete runs were performed with independent server restarts. Both runs
passed all checks:

- CUDA device 0 was selected for the model and vision encoder.
- No CPU fallback or CUDA error was observed.
- Every response had the matching request ID.
- Every action tensor had shape `[50, 32]` and contained only finite values.
- All measured responses were bit-identical within and across both runs.
- The common action SHA-256 was
  `f68f87822991f318eaeaad2d95a61f018f2485df61f3cfb0c8c7bb0859007fb0`.
- Maximum absolute and relative repeatability differences were both zero.

Model load time was 11.059 seconds in the first run and 12.065 seconds in the
second run.

After the unit tests gained schema-drift guards and the smoke client gained
deterministic JSON fixture identification and stricter response validation, a
third functional-only run rechecked the final candidate client. The client
SHA-256 was
`16dea2bc8da6beb6b6aaf44292f38b4ad73f299450174213699643caa2ca1ee0`.
One warm-up and three measured requests passed with deterministic fixture
SHA-256 `d353f486fa05fad51cfa84f721781136b16a338ce3d9167b89b52cb37856b97d`.
All action values passed the finite checks, and all timing values passed the
finite and non-negative checks. The three measured outputs had the same action
SHA-256 shown above, and maximum absolute and relative differences remained
zero. This functional recheck did not repeat the performance protocol because
request encoding, transport, model, and server code were unchanged.

## Latency results

The following table reports the second run's twenty measured requests in
milliseconds. Client wall time includes transport and client-side overhead;
the server values come from the existing response timing fields.

| Timing | Min | P50 | P95 | Max | Mean | Stddev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Client wall | 496.708 | 496.962 | 497.196 | 497.507 | 496.988 | 0.158 |
| Server total | 496.006 | 496.262 | 496.471 | 496.757 | 496.283 | 0.148 |
| Vision | 82.234 | 82.325 | 82.566 | 82.785 | 82.358 | 0.117 |
| Inference | 413.021 | 413.153 | 413.245 | 413.291 | 413.156 | 0.057 |

The independent first run produced a server-total P50 of 495.979 ms, P95 of
496.224 ms, and mean of 496.009 ms.

The pi0.5 implementation currently populates aggregate inference timing but
does not populate the prefill and denoise phase fields. Their recorded zeros
mean that phase-level data is unavailable; they must not be interpreted as
zero-cost phases. Model code was intentionally left unchanged.

## Memory and resource observations

The second run provided the published resource sample:

| Observation | Value |
| --- | ---: |
| System RAM used, minimum | 10,985 MiB |
| System RAM used, peak | 14,637 MiB |
| Observed system RAM increase | 3,652 MiB |
| Server maximum RSS (`time -v`) | 1,187,716 KiB |
| Server RSS after performance protocol | 585,840 KiB |
| Server PSS after performance protocol | 571,981 KiB |
| Server process swap | 0 KiB |
| GPU utilization peak | 99% |
| CPU temperature peak | 65.968 C |
| GPU temperature peak | 66.000 C |
| `VDD_GPU_SOC` instantaneous peak | 30,065 mW |
| `VDD_CPU_CV` instantaneous peak | 2,849 mW |

System-wide swap peaked at 1,232 MiB and was already in use before the server
started; the server itself did not swap. Jetson unified memory means the
system RAM increase is a host-level observation, not a model-only allocation.
The rail readings are instantaneous `tegrastats` samples and are not total
device energy measurements.

## Evidence and limitations

- Raw logs, JSON results, `tegrastats` samples, process memory snapshots, and
  environment manifests are retained in the validation workspace on the test
  device. Host addresses and unrelated service details are intentionally not
  included in this repository.
- The original system Python and existing inference Python environments were
  not modified. `pyzmq` was installed only in the isolated smoke-test virtual
  environment.
- No simulator rollout or task-success benchmark was run. This report verifies
  model loading, deterministic request/response behavior, CUDA execution, and
  maintenance-window performance only.
- The host RT kernel, pre-existing system swap, and absence of an external
  power meter limit the strength of absolute performance and power claims.
