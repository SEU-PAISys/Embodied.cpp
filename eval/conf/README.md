# Evaluation Configs

YAML files in this directory provide benchmark defaults for eval clients.
Command-line arguments always override values loaded from `--conf`.

Example:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf libero_pi05_eval.yaml
```

Run only one task from the same config:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf libero_pi05_eval.yaml \
  --task-id 0
```

Use explicit paths for machine-local assets such as tokenizers when needed.

The C++ GR00T N1.7 path uses the tokenizer embedded in its Qwen3-VL GGUF, so its
LIBERO configuration needs no tokenizer path:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf libero_groot_n1_eval.yaml
```

For a 200-episode `libero_object` run and automatic table metrics, use the
tracked config and explicit overrides:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf libero_groot_n1_eval.yaml \
  --task-ids 0 1 2 3 4 5 6 7 8 9 --n-episodes 20 \
  --profile-output outputs/profiles/groot_n1_libero_object.json
```

The three integrated runtimes ship equivalent LIBERO configurations:

```bash
# Xiaomi-Robotics-0 (start vla-server with xr0-mmproj.gguf + xr0.gguf first)
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py --conf libero_xr0_eval.yaml

# TurboVLA (single-GGUF checkpoint)
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py --conf libero_turbovla_eval.yaml

# X-VLA (single-GGUF checkpoint; --tokenizer points at a BartTokenizerFast snapshot)
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py --conf libero_xvla_eval.yaml
```

These are per-model defaults rather than full-suite reproductions of the
committed reports: each selects 10 object tasks, 20 episodes per task and
seed 42. All three render at 256x256; `image_size` is a separate model-side
resize (X-VLA: 224). See the shared
[run and aggregation instructions](../../docs/results/README.md#running-and-aggregating)
for multi-suite runs and historical protocol differences. XR0 and X-VLA
also require `--tokenizer` to point to an existing matching HF snapshot;
the GGUF converter does not create the example tokenizer directories.

LingBot-VA on LIBERO uses the dedicated `wam-lingbot-server` protocol. Start
the CUDA server separately with the transformer, text encoder, and VAE encoder
GGUFs, then run:

```bash
eval/sim/libero/libero_uv/.venv/bin/python \
  eval/client/run_sim_client_direct.py \
  --conf libero_lingbot_va_eval.yaml
```

The LingBot-VA LIBERO config mirrors the upstream Python policy defaults:
`image_size=128`, `max_steps=800`, `n_episodes=50`, `max_length=512`,
untruncated cache updates, and server-side quantile action postprocessing.
For parity with the Python rollout, run the server in official denoise order
(`VLA_LINGBOT_PREDICT_OFFICIAL_ORDER=1`) and leave
`VLA_LINGBOT_PREDICT_MIXED` unset; the mixed path changes the denoise ordering
and now also requires `VLA_LINGBOT_ALLOW_MIXED_PREDICT=1`.
Leave experimental C++ implementation optimizations such as
`VLA_LINGBOT_RUNTIME_KV_DEVICE`,
`VLA_LINGBOT_OFFICIAL_CUDA_SELF_ATTN_DEVICE_X`, and
`VLA_LINGBOT_RUNTIME_KV_DEVICE_SELF_CTX_D2D` unset until checksum/parity tests
and LIBERO rollout success show they are equivalent to the Python official
path. Runtime KV self-attention itself is not an optional optimization in
official-order: it is the C++ bridge for Python's attention-cache semantics and
is enabled by default for official LingBot-VA requests. The runtime KV capacity
uses the Python `create_empty_cache()` formula from `attn_window`,
latent-token-per-chunk, and action-token-per-chunk; the default unconditional
text branch uses the official empty negative prompt token sequence (`[1]` for
the LingBot T5 tokenizer).

HY-VLA on RoboTwin uses the native RoboTwin client:

```bash
eval/sim/robotwin/robotwin_uv/.venv/bin/python \
  eval/client/run_robotwin_eval.py \
  --conf robotwin_hy_vla_eval.yaml
```

The HY-VLA RoboTwin config mirrors the upstream Python policy defaults:
`blend_mode=rel_abs`, `exec_action_size=7`, `img_history_size=6`, and
`img_history_interval=5`.

SmolVLA on LIBERO uses the dedicated `libero_smolvla_eval.yaml` configuration.
It expects the policy and SigLIP mmproj GGUFs described in
[`scripts/README.md`](../../scripts/README.md), uses one replayed action per
model request, and applies the serialized processor's trailing task-prompt
newline automatically in the direct client.

Latency-aware RoboTwin videos can be enabled for presentation/debug runs:

```bash
eval/sim/robotwin/robotwin_uv/.venv/bin/python \
  eval/client/run_robotwin_eval.py \
  --conf robotwin_hy_vla_eval.yaml \
  --latency-video \
  --latency-video-dir outputs/robotwin_hy_vla_latency_video
```

These videos differ from RoboTwin's native eval videos. Native videos record
simulator action frames only, so model forward latency is not visible. The
latency-aware path writes the same head-camera rollout frames, but inserts
still frames after each model forward according to `latency_ms_total`,
`latency_video_fps`, and `latency_video_speed`. This makes Python/C++ latency
differences visible while leaving the environment action sequence unchanged.

For the offline latency-video pipeline, keep native rollout/video behavior
unchanged and only record per-forward timing:

```bash
eval/sim/robotwin/robotwin_uv/.venv/bin/python \
  eval/client/run_robotwin_eval.py \
  --conf robotwin_hy_vla_eval.yaml \
  --forward-timing \
  --forward-timing-jsonl outputs/robotwin_hy_vla_native/forward_timing.jsonl
```

The JSONL schema is `robotwin_forward_timing.v1`; each line is one real model
forward and includes task, episode, seed, instruction, forward index,
`steps_before_forward`, `take_action_cnt_before_forward`, chunk sizes, and
latency fields. A separate video post-processing script can read RoboTwin's
native video plus this JSONL to insert latency frames offline.
