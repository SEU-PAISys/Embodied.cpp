# LingBot-VA vla.cpp 执行进度 / LingBot-VA vla.cpp Progress Log

本文档记录 LingBot-VA 接入 `vla.cpp` 的实际执行过程。规划和设计决策见 `LINGBOT_VA_PORT_PLAN.md`；本文只记录已经做过的动作、结果、问题和下一步。

This document records the actual execution progress of integrating LingBot-VA into `vla.cpp`. Planning and design decisions live in `LINGBOT_VA_PORT_PLAN.md`; this file records completed actions, results, issues, and next steps.

## 当前执行顺序 / Current Execution Order

更新时间：2026-06-17 00:00:00 CST

Updated: 2026-06-17 00:00:00 CST

- [x] 1. 将 streaming smoke 整理成可复用的 transformer executor 骨架。
- [x] 1. Refactor the streaming smoke path into a reusable transformer executor skeleton.
- [ ] 2. 补齐真实 RoPE / grid id / position id 路径，尽量贴近 LingBot-VA Python 原版。（进行中：smoke/parity 路径已完成，真实 `predict()` 动态 grid 尚未接入。）
- [ ] 2. Add the real RoPE / grid id / position id path, matching the original LingBot-VA Python implementation as closely as possible. (In progress: smoke/parity path is done; dynamic grids from real `predict()` are not wired yet.)
- [x] 3. 接入 diffusion / flow sampling loop 与 latent/action 状态管理。（已接入真实 latent 输入桥和 action flow loop；仍需更完整的会话级 world-latent cache。）
- [x] 3. Integrate the diffusion / flow sampling loop and latent/action state management. (Real latent input bridge and action flow loop are wired; richer session-level world-latent caching remains.)
- [x] 4. 将 executor 接到真实 `predict()` 和 vla.cpp ZMQ/protobuf 请求。（world-server 可接收真实 latent、lang tokens，并返回 LIBERO 7 维 action chunk。）
- [x] 4. Connect the executor to real `predict()` and vla.cpp ZMQ/protobuf requests. (The world-server accepts real latents and lang tokens, and returns LIBERO 7-D action chunks.)
- [x] 5. 补齐 VAE encoder/decoder 相关 C++/ggml 支持。（encoder/decoder parity 已通过；后续是性能优化和自动接入真实图像输入。）
- [x] 5. Add C++/ggml support for the VAE encoder/decoder path. (Encoder/decoder parity passes; remaining work is performance tuning and automatic real-image input wiring.)
- [x] 6. 补齐 UMT5 text encoder 相关 C++/ggml 支持。（24-block final norm parity 已通过，并接入 predict/server。）
- [x] 6. Add C++/ggml support for the UMT5 text encoder path. (24-block final-norm parity passes and it is wired into predict/server.)
- [x] 7. 搭建真实输入链路：图像输入 -> VAE encoder -> latent -> WanTransformer -> action postprocess -> LIBERO/robot action。（CPU/F32 VAE bridge smoke 已跑通；后续优化性能和 LIBERO client 集成。）
- [x] 7. Build the real input path: image input -> VAE encoder -> latent -> WanTransformer -> action postprocess -> LIBERO/robot action. (CPU/F32 VAE bridge smoke passes; next is performance optimization and LIBERO client integration.)
- [ ] 8. 在完整结构可运行后推进量化、block window 内存策略和 8 GB 部署优化。
- [ ] 8. After the full structure runs, proceed with quantization, block-window memory policy, and 8 GB deployment optimization.

## 2026-06-17 00:00 CST - Multi-Block Q8 Resident CUDA Mixed Predict Smoke

中文：

- 将 C++-only e2e parity harness 用于多 block smoke，验证 `lingbot-world-server` 可以从 1 block 扩到 2/3 blocks。
- 修复 `scripts/run_lingbot_va_e2e_parity.py` 的可观测性：
  - server ready 之后继续 tee `lingbot-world-server` stdout；
  - 日志写入 `<out-dir>/cpp/server.log`，同时打印到终端；
  - `--cuda-self-attn` 自动设置 `VLA_LINGBOT_PREDICT_MIXED=1`，避免误入 separate dense no-cache path。
- 验证 2-block 和 3-block 均通过：
  - UMT5 text encoder: 1 block；
  - VAE image bridge: real image -> latent；
  - WanTransformer mixed video/action predict；
  - CUDA flex self-attn；
  - Q8_0 resident block cache；
  - terminal predict step 写入 mode=1 pred cache。

验证命令：

```text
eval/sim/libero/libero_uv/.venv/bin/python scripts/run_lingbot_va_e2e_parity.py \
  --mode cpp \
  --out-dir /tmp/lingbot_e2e_blocks3_q8 \
  --num-fixtures 1 \
  --blocks 3 \
  --text-blocks 1 \
  --video-steps 1 \
  --action-steps 1 \
  --resident-dtype q8_0 \
  --resident-blocks 3 \
  --cuda-self-attn \
  --addr tcp://127.0.0.1:6063
```

关键结果：

```text
resident block cache store: block=0 dtype=q8_0 tensors=27 bytes=166.05 MiB
resident block cache store: block=1 dtype=q8_0 tensors=27 bytes=166.05 MiB
resident block cache store: block=2 dtype=q8_0 tensors=27 bytes=166.05 MiB
runtime KV self-attn session=1 block=0 mode=1 q=128 k=128 used=128 pred=128
runtime KV self-attn session=1 block=1 mode=1 q=128 k=128 used=128 pred=128
runtime KV self-attn session=1 block=2 mode=1 q=128 k=128 used=128 pred=128
predict bridge ok blocks=3 mode=cuda-flex-self-attn window=1 video_steps=1 action_steps=1
chunk=[16,7] checksum=0.535181746 max=0.137325346
server total=11029.2ms
```

结论：

- 当前 C++ LingBot path 已经从 1-block smoke 推进到 3-block Q8 resident CUDA mixed predict smoke。
- 这说明 multi-block resident block cache、mixed forward、CUDA self-attn、runtime pred-cache 写入可以在同一次 world-server 请求中协同工作。
- 下一步可以继续扩大到 5/10/30 blocks；同时要注意 `resident-blocks=N` 会以约 166 MiB/block 的 Q8_0 CPU resident 成本增长。

English:

- Used the C++-only e2e parity harness for multi-block smoke and verified that `lingbot-world-server` scales from 1 block to 2/3 blocks.
- Improved observability in `scripts/run_lingbot_va_e2e_parity.py`:
  - keeps teeing `lingbot-world-server` stdout after the ready banner;
  - writes the log to `<out-dir>/cpp/server.log` and prints it to the terminal;
  - automatically sets `VLA_LINGBOT_PREDICT_MIXED=1` when `--cuda-self-attn` is used, avoiding the separate dense no-cache path.
- Verified both 2-block and 3-block runs:
  - UMT5 text encoder: 1 block;
  - VAE image bridge: real image -> latent;
  - WanTransformer mixed video/action predict;
  - CUDA flex self-attention;
  - Q8_0 resident block cache;
  - terminal predict step writes mode=1 pred cache.

Verification command:

```text
eval/sim/libero/libero_uv/.venv/bin/python scripts/run_lingbot_va_e2e_parity.py \
  --mode cpp \
  --out-dir /tmp/lingbot_e2e_blocks3_q8 \
  --num-fixtures 1 \
  --blocks 3 \
  --text-blocks 1 \
  --video-steps 1 \
  --action-steps 1 \
  --resident-dtype q8_0 \
  --resident-blocks 3 \
  --cuda-self-attn \
  --addr tcp://127.0.0.1:6063
```

Key result:

```text
resident block cache store: block=0 dtype=q8_0 tensors=27 bytes=166.05 MiB
resident block cache store: block=1 dtype=q8_0 tensors=27 bytes=166.05 MiB
resident block cache store: block=2 dtype=q8_0 tensors=27 bytes=166.05 MiB
runtime KV self-attn session=1 block=0 mode=1 q=128 k=128 used=128 pred=128
runtime KV self-attn session=1 block=1 mode=1 q=128 k=128 used=128 pred=128
runtime KV self-attn session=1 block=2 mode=1 q=128 k=128 used=128 pred=128
predict bridge ok blocks=3 mode=cuda-flex-self-attn window=1 video_steps=1 action_steps=1
chunk=[16,7] checksum=0.535181746 max=0.137325346
server total=11029.2ms
```

Conclusion:

- The C++ LingBot path has moved from a 1-block smoke to a 3-block Q8 resident CUDA mixed predict smoke.
- Multi-block resident block cache, mixed forward, CUDA self-attention, and runtime pred-cache writes now work together in one world-server request.
- Next step: scale to 5/10/30 blocks, while tracking the Q8_0 CPU resident cost of about 166 MiB per resident block.

## 2026-06-14 / June 14, 2026

### 1. 建立移植规划文档 / Created Port Planning Document

动作：

Action:

- 新建并持续更新 `LINGBOT_VA_PORT_PLAN.md`。
- Created and iteratively updated `LINGBOT_VA_PORT_PLAN.md`.
- 将文档格式确定为中英双语。
- Established bilingual Chinese/English formatting for retained planning and note documents.

结果：

Result:

- 已明确当前路线是静态分析和 vla.cpp port planning 先行。
- The current route is static analysis and vla.cpp port planning first.
- 已明确不做 action-only shortcut。
- The action-only shortcut has been rejected.

### 2. 确定顶层工程决策 / Locked Top-Level Engineering Decisions

动作：

Action:

- 与用户逐项讨论并记录核心决策。
- Discussed and recorded core decisions with the user one by one.

结果：

Result:

```text
1. 通讯协议主线使用 vla.cpp 现有 ZMQ/protobuf。
   Main communication protocol is vla.cpp's existing ZMQ/protobuf.

2. 最终范围包含 UMT5、VAE 和 LingBot-VA world-model backbone 的完整 C++/ggml 承载。
   Final scope includes full C++/ggml hosting for UMT5, VAE, and the LingBot-VA world-model backbone.

3. 第一真实 end-to-end 目标是本机 RTX 3070 Ti Laptop 8 GB VRAM 可运行。
   First real end-to-end target is local RTX 3070 Ti Laptop 8 GB VRAM execution.

4. 量化由本项目把关，以本地 8 GB 可运行和约 3%-5% 整体损失为工程目标。
   Quantization is owned by this project, targeting local 8 GB execution and roughly 3%-5% overall loss.
```

## 2026-06-16 17:46 CST - Pred Cache and `clear_pred_cache()` Semantics

中文：

- 补齐 LingBot-VA 原版 cache 生命周期的下一层语义：
  - 普通 `_infer()` 的最后一次 terminal forward 使用 `update_cache=1` 写入 pred cache。
  - 后续 `_compute_kv_cache()` 开始前调用 `clear_pred_cache()`，清掉预测 cache。
  - `_compute_kv_cache()` 再用 `update_cache=2` 写入真实观测/action history 对应的 persistent history cache。
- C++ 对应实现：
  - `vla::Inputs::lingbot_clear_pred_cache`
  - `runtime_kv_clear_pred_for_session(model, session)`
  - 普通 predict 的 terminal step 自动切换为 `cache_mode=1`，除非设置 `VLA_LINGBOT_DISABLE_PRED_CACHE=1`。
  - `compute_kv_cache=True` 时 server 设置：

```text
lingbot_cache_mode = 2
lingbot_clear_pred_cache = true
```

验证：

```text
cmake --build build -j2

Runtime KV + pred-cache three-stage LIBERO smoke:
VLA_LINGBOT_PREDICT_CUDA_SELF_ATTN=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1

request 1 predict:
mode=0 q=128 k=128 used=0 pred=0
mode=1 q=128 k=128 used=128 pred=128

request 2 compute_kv_cache:
runtime KV clear_pred session=1 blocks=1 pred_before=128 pred_after=0
mode=2 q=128 k=128 used=128 pred=0
mode=2 q=128 k=256 used=256 pred=0

request 3 predict:
mode=0 q=128 k=384 used=256 pred=0
mode=1 q=128 k=384 used=384 pred=128
```

结论：

- 原版 `_infer()` 写 pred cache、`_compute_kv_cache()` 清 pred 并写真实 history cache 的生命周期已经在 C++ CUDA self-attn 路径闭环。
- 当前仍是 1-block 验证；下一步要扩大到多 block，并让 cache mask 更贴近原版 flex/block-sparse 语义。

English:

- Completed the next layer of the original LingBot-VA cache lifecycle:
  - the last terminal forward of normal `_infer()` writes pred cache with `update_cache=1`;
  - `_compute_kv_cache()` begins by calling `clear_pred_cache()`;
  - `_compute_kv_cache()` then writes real observation/action-history cache with `update_cache=2`.
- C++ implementation:
  - `vla::Inputs::lingbot_clear_pred_cache`
  - `runtime_kv_clear_pred_for_session(model, session)`
  - normal predict automatically switches the terminal step to `cache_mode=1`, unless `VLA_LINGBOT_DISABLE_PRED_CACHE=1` is set.
  - `compute_kv_cache=True` makes the server set:

```text
lingbot_cache_mode = 2
lingbot_clear_pred_cache = true
```

Verification:

```text
cmake --build build -j2

Runtime KV + pred-cache three-stage LIBERO smoke:
VLA_LINGBOT_PREDICT_CUDA_SELF_ATTN=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1

request 1 predict:
mode=0 q=128 k=128 used=0 pred=0
mode=1 q=128 k=128 used=128 pred=128

request 2 compute_kv_cache:
runtime KV clear_pred session=1 blocks=1 pred_before=128 pred_after=0
mode=2 q=128 k=128 used=128 pred=0
mode=2 q=128 k=256 used=256 pred=0

request 3 predict:
mode=0 q=128 k=384 used=256 pred=0
mode=1 q=128 k=384 used=384 pred=128
```

Conclusion:

- The original lifecycle of `_infer()` writing pred cache and `_compute_kv_cache()` clearing pred cache before writing real history cache is now closed in the C++ CUDA self-attention path.
- This is still a 1-block validation. Next step: scale to multi-block and make the cache attention mask closer to the original flex/block-sparse semantics.

## 2026-06-16 17:25 CST - Runtime KV Pool Wired into CUDA Self-Attention

中文：

- 将上一阶段的 `compute_kv_cache` 控制流继续向真实原版语义推进：把 runtime KV pool 接入 WanTransformer 的 CUDA flex self-attention 路径。
- 新增/修改内容：
  - `vla::Inputs` 增加 `lingbot_session_id` 和 `lingbot_cache_mode`：
    - `0`: normal temporary query，临时写入当前 K/V，attention 后 restore。
    - `1`: prediction cache update，保留为 pred cache。
    - `2`: persistent history/cache prefill，保留为历史 cache。
  - `lingbot-world-server` 在 reset 后为同一个用户 session 生成新的 runtime session id，避免 episode reset 后复用旧 KV。
  - `LingBotKVCache` 增加 compact valid K/V 的能力。
  - 新增全局 runtime KV store，key 为 `(model pointer, runtime session id, block id)`。
  - `real_qkv_to_cuda_context()` 现在可以使用持久 KV：
    - `compute_kv_cache` 请求以 `mode=2` 写入历史 KV；
    - 后续 predict 以 `mode=0` 临时插入当前 K/V，query attend 到历史+当前 K/V，然后 restore 当前 K/V；
    - CUDA kernel 使用 `seq_q != seq_k` 路径。

验证：

```text
cmake --build build -j2

Runtime KV three-stage LIBERO smoke with:
VLA_LINGBOT_PREDICT_CUDA_SELF_ATTN=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1

request 1 predict:
runtime KV self-attn session=1 block=0 mode=0 q=128 k=128 used=0 pred=0
chunk1 [16,7]

request 2 compute_kv_cache:
runtime KV self-attn session=1 block=0 mode=2 q=128 k=128 used=128 pred=0
runtime KV self-attn session=1 block=0 mode=2 q=128 k=256 used=256 pred=0
status: kv_cache_updated

request 3 predict:
runtime KV self-attn session=1 block=0 mode=0 q=128 k=384 used=256 pred=0
chunk2 [16,7]
```

结论：

- 这说明 C++ runtime 现在已经不只是“模拟 cache 控制流”，而是真的把历史 K/V 保存在 session/block 级 KV pool 里，并在后续 CUDA self-attention 中使用。
- 当前接入范围是 CUDA flex self-attn 路径；dense ggml self-attn 仍是 fallback，不使用 runtime KV pool。
- 下一步：
  - 把 `mode=1` pred-cache update 与原版 `clear_pred_cache()` 流程完整压测。
  - 从 1 block 扩展到多 block / 30 block，并检查 KV pool 内存与 eviction 策略。
  - 将当前 dense “history+current 全可见” cache attention mask 进一步贴近原版 flex/block-sparse mask。

English:

- Advanced the previous `compute_kv_cache` control-flow bridge toward the original runtime semantics by wiring a real runtime KV pool into the CUDA flex self-attention path of the WanTransformer.
- Added/changed:
  - `vla::Inputs` now carries `lingbot_session_id` and `lingbot_cache_mode`:
    - `0`: normal temporary query; insert current K/V temporarily, attend, then restore.
    - `1`: prediction cache update; keep entries as pred cache.
    - `2`: persistent history/cache prefill; keep entries as history cache.
  - `lingbot-world-server` creates a new runtime session id after each reset, so old KV entries are not reused across episodes.
  - `LingBotKVCache` can now compact valid K/V entries.
  - Added a global runtime KV store keyed by `(model pointer, runtime session id, block id)`.
  - `real_qkv_to_cuda_context()` can now use persistent KV:
    - `compute_kv_cache` writes history KV with `mode=2`;
    - later predict calls insert current K/V temporarily with `mode=0`, attend to history+current K/V, then restore the temporary entries;
    - the CUDA kernel path now supports `seq_q != seq_k`.

Verification:

```text
cmake --build build -j2

Runtime KV three-stage LIBERO smoke with:
VLA_LINGBOT_PREDICT_CUDA_SELF_ATTN=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1

request 1 predict:
runtime KV self-attn session=1 block=0 mode=0 q=128 k=128 used=0 pred=0
chunk1 [16,7]

request 2 compute_kv_cache:
runtime KV self-attn session=1 block=0 mode=2 q=128 k=128 used=128 pred=0
runtime KV self-attn session=1 block=0 mode=2 q=128 k=256 used=256 pred=0
status: kv_cache_updated

request 3 predict:
runtime KV self-attn session=1 block=0 mode=0 q=128 k=384 used=256 pred=0
chunk2 [16,7]
```

Conclusion:

- The C++ runtime now does more than simulate cache control flow: it stores historical K/V in a session/block-level KV pool and uses that pool in later CUDA self-attention calls.
- Current wiring covers the CUDA flex self-attention path. The dense ggml self-attention path remains a fallback and does not use the runtime KV pool.
- Next steps:
  - Validate `mode=1` pred-cache update and the original `clear_pred_cache()` flow.
  - Scale from 1 block to multi-block / 30-block runs and inspect memory/eviction behavior.
  - Replace the current dense “history+current all visible” cache attention mask with a closer original flex/block-sparse mask.

## 2026-06-16 17:25 CST - `compute_kv_cache` Control-Flow Bridge

中文：

- 继续复现 LingBot-VA 原版 server/eval 控制流。
- 原版 LIBERO 流程是：

```text
infer(obs, prompt) -> returns action chunk
execute chunk in env
infer(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action)
next infer(...)
```

- 在 `lingbot.proto` 中新增：

```proto
bool compute_kv_cache = 10;
bool imagine = 11;
```

- `LingBotWorldClient` 新增 `update_cache(obs_or_sequence, action_chunk, imagine=False)`：
  - 支持单帧 obs，也支持多帧 `key_frame_list`。
  - 多帧图像按 camera 打包成 `[F,H,W,3]`，server 侧继续转成 `[views,3,F,H,W]`。
  - 自动携带上一段 action chunk 对应的 `[7,4,4] action_condition`。
- `lingbot-world-server` 新增 `compute_kv_cache` 分支：
  - 仍走 UMT5 / VAE / action-condition / Wan bridge 输入链路。
  - 返回 `kv_cache_updated ...` 状态，不把结果当动作 chunk 返回。

验证：

```text
cmake --build build -j2
python -m py_compile eval/client/lingbot_world_client.py eval/client/run_sim_client_direct.py

三段式 LIBERO smoke:
1. predict chunk1 -> [16,7]
2. update_cache([obs0, obs1], chunk1) -> kv_cache_updated
3. predict chunk2 -> [16,7]
4. env.step(chunk2[0]) accepted

server log:
request 1 VAE input=[3,1,128,128]
request 2 compute_kv_cache VAE input=[3,2,128,128]
request 2 action condition shape=[7,4,4]
request 3 VAE input=[3,1,128,128]
```

边界说明：

- 这一步完成的是原版 `compute_kv_cache=True` 的协议和控制流骨架，以及多帧 VAE/action-history 输入链路。
- 真正的每层 attention KV pool 持久化还没有接入主 `exec_wan_mixed_forward_tensors()`；当前仍会执行 forward 来验证输入路径。
- 下一步应把 `LingBotKVCache` 从 smoke 结构升级为 session/block 级 cache，并接入 `exec_block_one()` / `exec_block_one_cuda_self_attn()` 的 Q/K/V attention 路径。

English:

- Continued matching the original LingBot-VA server/evaluation control flow.
- The original LIBERO loop is:

```text
infer(obs, prompt) -> returns action chunk
execute chunk in env
infer(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action)
next infer(...)
```

- Added to `lingbot.proto`:

```proto
bool compute_kv_cache = 10;
bool imagine = 11;
```

- Added `LingBotWorldClient.update_cache(obs_or_sequence, action_chunk, imagine=False)`:
  - accepts either a single observation or a multi-frame `key_frame_list`;
  - packs multi-frame images per camera as `[F,H,W,3]`, which the server converts to `[views,3,F,H,W]`;
  - carries the previous action chunk as a `[7,4,4] action_condition`.
- Added a `compute_kv_cache` branch in `lingbot-world-server`:
  - it runs the UMT5 / VAE / action-condition / Wan bridge input path;
  - it returns a `kv_cache_updated ...` status instead of an action chunk.

Verification:

```text
cmake --build build -j2
python -m py_compile eval/client/lingbot_world_client.py eval/client/run_sim_client_direct.py

Three-stage LIBERO smoke:
1. predict chunk1 -> [16,7]
2. update_cache([obs0, obs1], chunk1) -> kv_cache_updated
3. predict chunk2 -> [16,7]
4. env.step(chunk2[0]) accepted

server log:
request 1 VAE input=[3,1,128,128]
request 2 compute_kv_cache VAE input=[3,2,128,128]
request 2 action condition shape=[7,4,4]
request 3 VAE input=[3,1,128,128]
```

Boundary:

- This step implements the protocol/control-flow skeleton for the original `compute_kv_cache=True` path and validates the multi-frame VAE/action-history input bridge.
- True per-layer persistent attention KV pools are not yet wired into the main `exec_wan_mixed_forward_tensors()` path; the current branch still executes forward to validate inputs.
- Next step: promote `LingBotKVCache` from smoke infrastructure into session/block-level runtime cache and connect it to the Q/K/V attention path in `exec_block_one()` / `exec_block_one_cuda_self_attn()`.

## 2026-06-16 17:38 CST - Original Action-History Condition Bridge

中文：

- 将上一阶段的 raw `state` 调试路径升级为更贴近 LingBot-VA 原版的 action-history condition 路径。
- 原版 LIBERO eval 的关键语义是：

```text
ret = model.infer(dict(obs=first_obs, prompt=prompt))
action = ret["action"]                 # [7,F,H]
...
model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))
```

- 这里的 `state=action` 命名容易误导；它实际传入的是上一段 action chunk/history，随后由原版 `preprocess_action()` 转成 `[1,30,F,H,1]` action latent condition。
- 为避免把 raw robot pose 和 action history 混在一起，新增了显式协议字段：

```proto
Tensor action_condition = 9;  // F32 [C,F,H], LIBERO uses [7,4,4]
```

- `lingbot-world-server` 现在会解析 `StepRequest.action_condition`，并传给 `vla::Inputs::lingbot_action_condition`。
- `LingBotWorldClient` 现在会维护上一段 `[16,7]` action chunk，并在下一次请求前转成 `[7,4,4]` action condition。
- `lingbot_va.cpp` 新增：
  - `preprocess_libero_action_history_to_action_condition()`
  - `zero_unused_libero_action_channels()`
- 模型侧优先级：
  - 如果请求带 `action_condition`，按原版 `preprocess_action()` 公式归一化并注入第 0 action frame。
  - 如果没有显式 condition，默认注入 zero action condition，对应原版 `frame_st_id == 0` 的启动行为。
  - 旧的 raw `state` 注入只保留为调试后门，需要手动设置 `VLA_LINGBOT_DISABLE_ZERO_ACTION_COND=1 VLA_LINGBOT_STATE_COND=1`。

验证：

```text
cmake --build build -j2

VLA_LINGBOT_ACTION_STATE_SMOKE=1:
action condition smoke ok:
state_checksum=3.97472095
history_checksum=-0.558737218
max_diff=2.50111043e-12

LIBERO two-request smoke:
request 1:
using default zero action condition for frame 0
predict bridge ok ... chunk=[16,7] checksum=3.53711823 max=0.201614022

request 2:
using LingBot action history condition shape=[7,4,4] checksum=1.99741524 max=0.344420314
predict bridge ok ... chunk=[16,7] checksum=3.49582323 max=0.201160431

env.step(action) accepted for both requests
```

English:

- Upgraded the previous raw-`state` debug path into an action-history condition path that more closely matches the original LingBot-VA implementation.
- The original LIBERO evaluation uses this key semantic:

```text
ret = model.infer(dict(obs=first_obs, prompt=prompt))
action = ret["action"]                 # [7,F,H]
...
model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))
```

- The name `state=action` is misleading here. It is actually the previous action chunk/history, which the original `preprocess_action()` converts into a `[1,30,F,H,1]` action latent condition.
- To avoid mixing raw robot pose with action history, the LingBot protocol now has an explicit field:

```proto
Tensor action_condition = 9;  // F32 [C,F,H], LIBERO uses [7,4,4]
```

- `lingbot-world-server` parses `StepRequest.action_condition` and forwards it through `vla::Inputs::lingbot_action_condition`.
- `LingBotWorldClient` keeps the previous `[16,7]` action chunk and converts it into `[7,4,4]` before the next request.
- Added in `lingbot_va.cpp`:
  - `preprocess_libero_action_history_to_action_condition()`
  - `zero_unused_libero_action_channels()`
- Runtime priority:
  - If the request provides `action_condition`, normalize it with the original `preprocess_action()` formula and inject it into action frame 0.
  - If no explicit condition is provided, inject a zero action condition by default, matching the original `frame_st_id == 0` bootstrap behavior.
  - The old raw-`state` injection remains only as a debug backdoor and requires `VLA_LINGBOT_DISABLE_ZERO_ACTION_COND=1 VLA_LINGBOT_STATE_COND=1`.

Verification:

```text
cmake --build build -j2

VLA_LINGBOT_ACTION_STATE_SMOKE=1:
action condition smoke ok:
state_checksum=3.97472095
history_checksum=-0.558737218
max_diff=2.50111043e-12

LIBERO two-request smoke:
request 1:
using default zero action condition for frame 0
predict bridge ok ... chunk=[16,7] checksum=3.53711823 max=0.201614022

request 2:
using LingBot action history condition shape=[7,4,4] checksum=1.99741524 max=0.344420314
predict bridge ok ... chunk=[16,7] checksum=3.49582323 max=0.201160431

env.step(action) accepted for both requests
```

## 2026-06-16 17:06 CST - Action/State Condition Bridge and LIBERO Smoke

中文：

- 补齐了 LingBot-VA 动作条件输入的 C++ 路径：
  - 新增 `preprocess_libero_state_to_action_condition()`，按 LingBot/LIBERO 7D action quantile 规则把 7D 输入归一化到内部 30-channel action condition。
  - 新增 `apply_action_condition_frame0()`，把 action condition 写入 action tensor 的第 0 帧，并在 flow loop 更新后重新固定该条件帧。
  - 将 runtime action tensor 从 smoke 风格的 `[1,30,1,16,1]` 调整为更贴近原版 action latent 的 `[1,30,4,4,1]`；对外输出仍 flatten 成 16-step action chunk。
- 加入 `VLA_LINGBOT_ACTION_STATE_SMOKE=1` parity 检查，对齐 Python reference 的 quantile normalization 公式。
- 重要语义结论：LIBERO client 当前传入的 `state` 是机器人 7D pose/action-like observation，不一定等价于 LingBot-VA 原版 `preprocess_action(action_cond)` 期望的 action history tensor。直接默认注入会产生异常大的归一化值，因此 runtime 默认关闭。
- 当前策略：
  - 默认不启用 state/action condition 注入，避免误把 raw robot pose 当 action history。
  - 只有设置 `VLA_LINGBOT_STATE_COND=1` 时，才会把 `Inputs.state` 注入 action condition。
  - 后续如果要完全复现原版，需要在 client 侧维护真实 action history / previous action condition，而不是直接使用 raw pose。

验证：

```text
cmake --build build -j2

VLA_LINGBOT_ACTION_STATE_SMOKE=1:
state action condition smoke ok:
max_diff=2.50111043e-12
checksum=3.97472095

LIBERO single-step default smoke, without VLA_LINGBOT_STATE_COND:
server loaded chunk_size=16 max_action_dim=30 real_action_dim=7 state_dim=7
UMT5 tokens=13 blocks=1
VAE image bridge ok views=2 input=[3,1,128,128] latent=[1,48,1,8,16]
predict bridge ok blocks=1 mode=ggml-dense window=1 video_steps=1 action_steps=1 action_postprocess=libero_quantiles chunk=[16,7]
returned action_dim=7
env.step(action) accepted

Default server log does not contain:
using LIBERO state action condition
```

风险记录：

```text
When forced on with raw LIBERO pose:
using LIBERO state action condition checksum=82.930439 max=21.3017216
```

这说明公式实现没问题，但输入语义不能默认假设正确。

English:

- Added the C++ path for LingBot-VA action-conditioning input:
  - `preprocess_libero_state_to_action_condition()` normalizes a 7D LIBERO-style vector into the internal 30-channel action-condition space using LingBot/LIBERO action quantiles.
  - `apply_action_condition_frame0()` writes the condition into frame 0 of the action tensor and re-pins that frame after flow updates.
  - The runtime action tensor layout changed from the smoke-style `[1,30,1,16,1]` to the more faithful action-latent layout `[1,30,4,4,1]`; externally it is still flattened into a 16-step action chunk.
- Added `VLA_LINGBOT_ACTION_STATE_SMOKE=1` parity coverage against the Python reference quantile-normalization formula.
- Important semantic finding: the current LIBERO client passes a 7D robot pose/action-like observation as `state`, which is not guaranteed to match the action-history tensor expected by the original LingBot-VA `preprocess_action(action_cond)`. Injecting it by default can create very large normalized values, so runtime injection is disabled by default.
- Current policy:
  - Do not inject state/action condition by default.
  - Inject `Inputs.state` only when `VLA_LINGBOT_STATE_COND=1` is explicitly set.
  - For full original parity later, the client should maintain a true action-history / previous-action condition instead of directly using raw pose.

Verification:

```text
cmake --build build -j2

VLA_LINGBOT_ACTION_STATE_SMOKE=1:
state action condition smoke ok:
max_diff=2.50111043e-12
checksum=3.97472095

LIBERO single-step default smoke, without VLA_LINGBOT_STATE_COND:
server loaded chunk_size=16 max_action_dim=30 real_action_dim=7 state_dim=7
UMT5 tokens=13 blocks=1
VAE image bridge ok views=2 input=[3,1,128,128] latent=[1,48,1,8,16]
predict bridge ok blocks=1 mode=ggml-dense window=1 video_steps=1 action_steps=1 action_postprocess=libero_quantiles chunk=[16,7]
returned action_dim=7
env.step(action) accepted

Default server log does not contain:
using LIBERO state action condition
```

Risk note:

```text
When forced on with raw LIBERO pose:
using LIBERO state action condition checksum=82.930439 max=21.3017216
```

This means the formula implementation is correct, but the input semantics must not be assumed correct by default.

## 2026-06-16 00:55:00 CST - VAE Encoder GGUF Baseline / VAE Encoder GGUF 基线

中文：

- 开始推进第 5 步：VAE encoder/decoder C++/ggml 支持。
- 确认 LingBot-VA 原版使用 `diffusers.AutoencoderKLWan`，外层通过 `WanVAEStreamingWrapper` 调用：
  - 输入视频 chunk 先按 `patch_size=2` 做 spatial patchify；
  - 进入 `vae.encoder`；
  - 经过 `quant_conv` 输出 latent 分布参数。
- VAE 权重总计 196 个 tensor：
  - `encoder.* + quant_conv.*`: 86 个 tensor，约 0.56 GiB；
  - decoder/post path: 110 个 tensor，约 2.07 GiB；
  - 完整 VAE 约 2.63 GiB。
- 更新 `scripts/convert_lingbot_va_to_gguf.py`：
  - 新增 `--modules vae_encoder`，写入 `encoder.* + quant_conv.*`；
  - 新增 `--modules vae`，预留写入完整 VAE；
  - 对 5D Conv3D 权重继续 flatten 成 GGUF 可表达的 2D tensor；
  - 新增 `lingbot_va.vae.tensor_shapes` metadata，保留 flatten 前的原始 shape，例如 `[out, in, kt, kh, kw]`；
  - 新增 VAE config metadata：`z_dim/base_dim/dim_mult/latents_mean/latents_std/scale_factor_*` 等。
- 生成 VAE encoder-only GGUF：

```text
/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf
size: 571M
scope: encoder_quant_conv
```

## 2026-06-16 00:58:37 CST - Server Protocol Note for Latents / Latent 服务协议注意事项

中文：

- 用户提醒：LingBot-VA 是 world-action model，不是标准 action-only VLA；除了 `action_chunk`，模型内部还涉及 video/world latent、latent cache、reset/session 状态和可能的 predicted latent 输出。
- 当前 `vla-server` / `src/serving/vla.proto` 是标准 VLA 风格：
  - input: images / language tokens / state / noise；
  - output: action_chunk；
  - 没有显式 latent tensor、latent dtype、latent shape、world-state cache、reset/session id、decode-world 等字段。
- 工程原则：
  - VAE encoder/UMT5/WanTransformer 的 C++ 实现可以继续推进，不被 server 协议阻塞；
  - 但不能把 latent 数据流硬塞进 `action_chunk` 或 `precomputed_img_emb`，否则后续评估、复现和性能优化都会变得混乱；
  - 如果后续需要频繁传输 latent，优先考虑新增 LingBot 专用 server 或专用 protobuf/multipart ZMQ 路径。
- 候选方案：
  - 方案 A：扩展现有 `vla.proto`，加入 optional latent fields、session/reset 字段；
  - 方案 B：新增 `lingbot-world-server`，使用 protobuf metadata + raw binary multipart 传输 latent，服务端持有 KV/cache/world state；
  - 方案 C：保持现有 `vla-server` 只输出 action，latent 完全由服务端内部 VAE/WorldModel 管理，客户端不直接传 latent。
- 当前暂定：继续完成模型结构；等真实 VAE/UMT5 接入后，根据 latent 是否跨进程传输、传输频率、显存/内存占用，再决定是否新增 server。

English:

- User note: LingBot-VA is a world-action model, not a standard action-only VLA. Besides `action_chunk`, it has video/world latents, latent caches, reset/session state, and potentially predicted-latent outputs.
- Current `vla-server` / `src/serving/vla.proto` follows a standard VLA interface:
  - input: images / language tokens / state / noise;
  - output: action_chunk;
  - no explicit fields for latent tensors, latent dtype, latent shape, world-state cache, reset/session id, or world decoding.
- Engineering rule:
  - C++ implementation of VAE encoder / UMT5 / WanTransformer can continue without being blocked by the server protocol;
  - but latent data flow should not be forced into `action_chunk` or `precomputed_img_emb`, because that would make evaluation, reproduction, and performance work messy;
  - if latents need frequent cross-process transfer, prefer a LingBot-specific server or a dedicated protobuf/multipart ZMQ path.
- Candidate designs:
  - Option A: extend the existing `vla.proto` with optional latent fields and session/reset fields;
  - Option B: add `lingbot-world-server`, using protobuf metadata + raw binary multipart transfer for latents, with KV/cache/world state owned by the server;
  - Option C: keep the current `vla-server` action-only, with VAE/WorldModel latents managed entirely inside the server.
- Current tentative decision: continue model implementation first; after real VAE/UMT5 integration, decide whether to add a new server based on whether latents cross process boundaries, transfer frequency, and memory/VRAM pressure.

## 2026-06-16 01:00:42 CST - LingBot World Server Skeleton / LingBot World Server 骨架

中文：

- 用户决策更新：如果 LingBot-VA 的 world/action 模型需要高效 latent 传输、专用 cache 管理或专用 kernel，就大方实现；这些可以作为方法和工程贡献的 difference。
- 新增专用协议：

```text
src/serving/lingbot.proto
```

- 协议新增：
  - `Tensor`: `name/dtype/shape/raw bytes`；
  - `ResetRequest`: `session_id/instruction/clear_cache`；
  - `StepRequest`: `session_id/input_latents/action_noise/state/return_world_latents/action_horizon`；
  - `LingBotResponse`: `action_chunk/output_latents/latency/error/status`。
- 新增专用服务：

```text
src/serving/lingbot-world-server.cpp
```

- 新 server 当前能力：
  - ZMQ REP server，默认端口 `tcp://*:5557`；
  - 支持 `ping/reset/step`；
  - 校验 latent tensor 的 dtype、shape、raw byte size；
  - 当前 step 暂时调用 `LingBotVAModelArch::predict()` bridge 返回 action；
  - 后续可以把 `input_latents/output_latents/session_id` 直接接到 VAE/WanTransformer/KV cache/world state。
- 更新 `CMakeLists.txt`：
  - 生成 `lingbot.pb.{h,cc}`；
  - 新增 target `lingbot-world-server`。

English:

- User decision update: if LingBot-VA's world/action model needs efficient latent transfer, dedicated cache management, or custom kernels, implement them directly. These can become method/engineering differences.
- Added a dedicated protocol:

```text
src/serving/lingbot.proto
```

- New protocol fields:
  - `Tensor`: `name/dtype/shape/raw bytes`;
  - `ResetRequest`: `session_id/instruction/clear_cache`;
  - `StepRequest`: `session_id/input_latents/action_noise/state/return_world_latents/action_horizon`;
  - `LingBotResponse`: `action_chunk/output_latents/latency/error/status`.
- Added a dedicated server:

```text
src/serving/lingbot-world-server.cpp
```

- Current server capability:
  - ZMQ REP server, default bind `tcp://*:5557`;
  - supports `ping/reset/step`;
  - validates latent tensor dtype, shape, and raw byte size;
  - current step path temporarily calls `LingBotVAModelArch::predict()` bridge and returns actions;
  - later, `input_latents/output_latents/session_id` can be wired directly into VAE/WanTransformer/KV cache/world state.
- Updated `CMakeLists.txt`:
  - generates `lingbot.pb.{h,cc}`;
  - adds target `lingbot-world-server`.

Verification:

```text
cmake --build build -j$(nproc)

env VLA_LINGBOT_PREDICT_BLOCKS=1 \
    VLA_LINGBOT_PREDICT_VIDEO_STEPS=1 \
    VLA_LINGBOT_PREDICT_ACTION_STEPS=1 \
    ./build/lingbot-world-server \
      --bind tcp://127.0.0.1:5998 \
      /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf

Python protobuf/ZMQ smoke:
ping -> request_id=1 status=hello-lingbot error=''
step -> request_id=2 status=step_with_latents_bridge error=''
        chunk_size=16 action_dim=30 action_len=480
        checksum=9.77859516069293 latency_ms_total=2821.9384765625
```

## 2026-06-16 01:11:40 CST - VAE Encoder Block Smokes / VAE Encoder Block Smoke

中文：

- 继续推进 VAE encoder C++/ggml 执行骨架。
- 新增 host-side VAE spatial patchify：
  - 对齐原版 `WanVAEStreamingWrapper.encode_chunk()` 中的 `patchify(x, patch_size=2)`；
  - 输入 `[B,C,F,H,W]`；
  - 输出 `[B,C*patch*patch,F,H/patch,W/patch]`。
- 新增 `vae_norm_silu_to_conv_layout()`：
  - `ggml_conv_3d` 输出布局是 `[W,H,D,C]`；
  - `ggml_group_norm` 把 `ne2` 当 channel；
  - 因此图中显式做 `[W,H,D,C] -> [W,H,C,D] -> group_norm/gamma/SiLU -> [W,H,D,C]`。
- 新增 VAE smoke 开关：
  - `VLA_LINGBOT_VAE_PATCHIFY_SMOKE=1`
  - `VLA_LINGBOT_VAE_NORM_SMOKE=1`
  - `VLA_LINGBOT_VAE_RESNET_SMOKE=1`
- `VLA_LINGBOT_VAE_RESNET_SMOKE` 覆盖：
  - `encoder.conv_in`
  - `down_blocks.0.resnets.0.norm1`
  - `down_blocks.0.resnets.0.conv1`
  - `down_blocks.0.resnets.0.norm2`
  - `down_blocks.0.resnets.0.conv2`
  - residual add
- 当前结论：
  - VAE encoder 的 patchify、Conv3D、GroupNorm、SiLU、ResNet residual 主路径可用 ggml 组合；
  - 后续需要重点处理 downsampler/time_conv、mid attention、完整 causal streaming cache；
  - 高效版本可以在当前 smoke 验证语义后，针对 Conv3D causal cache / fused norm-activation-conv 写专用 CUDA kernel。

English:

- Continued the C++/ggml execution skeleton for the VAE encoder.
- Added host-side VAE spatial patchify:
  - matches `patchify(x, patch_size=2)` from original `WanVAEStreamingWrapper.encode_chunk()`;
  - input `[B,C,F,H,W]`;
  - output `[B,C*patch*patch,F,H/patch,W/patch]`.
- Added `vae_norm_silu_to_conv_layout()`:
  - `ggml_conv_3d` outputs `[W,H,D,C]`;
  - `ggml_group_norm` treats `ne2` as channel;
  - the graph explicitly uses `[W,H,D,C] -> [W,H,C,D] -> group_norm/gamma/SiLU -> [W,H,D,C]`.
- Added VAE smoke toggles:
  - `VLA_LINGBOT_VAE_PATCHIFY_SMOKE=1`
  - `VLA_LINGBOT_VAE_NORM_SMOKE=1`
  - `VLA_LINGBOT_VAE_RESNET_SMOKE=1`
- `VLA_LINGBOT_VAE_RESNET_SMOKE` covers:
  - `encoder.conv_in`
  - `down_blocks.0.resnets.0.norm1`
  - `down_blocks.0.resnets.0.conv1`
  - `down_blocks.0.resnets.0.norm2`
  - `down_blocks.0.resnets.0.conv2`
  - residual add
- Current conclusion:
  - VAE encoder patchify, Conv3D, GroupNorm, SiLU, and the ResNet residual main path can be composed with ggml;
  - next difficult pieces are downsampler/time_conv, mid attention, and full causal streaming cache;
  - after semantics are verified by smoke tests, efficient CUDA kernels can target Conv3D causal cache and fused norm-activation-conv.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_PATCHIFY_SMOKE=1
VLA_LINGBOT_VAE_NORM_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf
./build/vla-server --bind tcp://127.0.0.1:5999 /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf

Result:
VAE patchify smoke ok:
input=[1,3,2,4,6]
output=[1,12,2,2,3]
checksum=0.992649427

VAE conv+norm+silu smoke ok:
input=[4,4,3,12]
out=[4,4,160,3]
checksum=1289.07544
max=1.50236797

VLA_LINGBOT_VAE_RESNET_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf
./build/vla-server --bind tcp://127.0.0.1:5999 /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf

Result:
VAE first resnet smoke ok:
input=[4,4,3,12]
out=[4,4,3,160]
checksum=-1078.12145
max=2.5500493
```

## 2026-06-16 01:24:13 CST - VAE Down Block 0 Smoke / VAE Down Block 0 Smoke

中文：

- 继续推进 VAE encoder stage 级别的 C++/ggml 骨架。
- 新增 `VLA_LINGBOT_VAE_DOWN_BLOCK0_SMOKE=1`。
- 覆盖 `encoder.down_blocks.0` 的完整第一 stage：
  - `encoder.conv_in`
  - `down_blocks.0.resnets.0`
  - `down_blocks.0.resnets.1`
  - `down_blocks.0.downsampler.resample.1`
- 关键 layout：
  - ResNet 主路径保持 VAE/Conv3D layout `[W,H,D,C]`；
  - spatial downsampler 使用 `ggml_conv_2d`，输入前转换 `[W,H,D,C] -> [W,H,C,D]`，把 temporal depth 当作 Conv2D batch；
  - Conv2D 输出 `[W/2,H/2,C,D]` 后再转换回 `[W/2,H/2,D,C]`。
- 当前结论：
  - 第一组 VAE encoder stage 的 ResNet + spatial downsample 可以用 ggml 组合；
  - 下一步重点是 `down_blocks.1/2`，其中包含 `downsampler.time_conv.weight [C,C,3,1,1]`，这会进入 temporal/causal downsample 语义；
  - temporal streaming cache 是高效 kernel 的强候选点。

English:

- Continued the stage-level C++/ggml skeleton for the VAE encoder.
- Added `VLA_LINGBOT_VAE_DOWN_BLOCK0_SMOKE=1`.
- Covered the full first encoder stage `encoder.down_blocks.0`:
  - `encoder.conv_in`
  - `down_blocks.0.resnets.0`
  - `down_blocks.0.resnets.1`
  - `down_blocks.0.downsampler.resample.1`
- Key layout:
  - the ResNet main path stays in VAE/Conv3D layout `[W,H,D,C]`;
  - the spatial downsampler uses `ggml_conv_2d`, converting `[W,H,D,C] -> [W,H,C,D]` so temporal depth acts as the Conv2D batch;
  - Conv2D output `[W/2,H/2,C,D]` is converted back to `[W/2,H/2,D,C]`.
- Current conclusion:
  - the first VAE encoder stage, including ResNets and spatial downsample, can be composed with ggml;
  - next focus is `down_blocks.1/2`, which include `downsampler.time_conv.weight [C,C,3,1,1]` and temporal/causal downsample semantics;
  - temporal streaming cache is a strong custom-kernel candidate.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_DOWN_BLOCK0_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf
./build/vla-server --bind tcp://127.0.0.1:5999 /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf

Result:
VAE down block0 smoke ok:
input=[4,4,3,12]
out=[2,2,3,160]
checksum=-81.1262203
max=6.42128372
```

- 更新 `src/models/lingbot_va.cpp`：
  - 新增 `validate_vae_encoder_tensors()`；
  - 新增 `VLA_LINGBOT_VAE_SMOKE=1`；
  - 新增 `VLA_LINGBOT_VAE_GGUF=/path/to/vae.gguf`。

English:

- Started step 5: C++/ggml support for the VAE encoder/decoder path.
- Confirmed that original LingBot-VA uses `diffusers.AutoencoderKLWan`, wrapped by `WanVAEStreamingWrapper`:
  - video chunks are spatially patchified with `patch_size=2`;
  - then passed through `vae.encoder`;
  - then passed through `quant_conv` to produce latent distribution parameters.
- VAE weights contain 196 tensors:
  - `encoder.* + quant_conv.*`: 86 tensors, about 0.56 GiB;
  - decoder/post path: 110 tensors, about 2.07 GiB;
  - full VAE: about 2.63 GiB.
- Updated `scripts/convert_lingbot_va_to_gguf.py`:
  - added `--modules vae_encoder` for `encoder.* + quant_conv.*`;
  - added `--modules vae` as the full-VAE path;
  - flattened 5D Conv3D weights into GGUF-compatible 2D tensors;
  - added `lingbot_va.vae.tensor_shapes` metadata to preserve original pre-flatten shapes such as `[out, in, kt, kh, kw]`;
  - added VAE config metadata such as `z_dim/base_dim/dim_mult/latents_mean/latents_std/scale_factor_*`.
- Generated a VAE encoder-only GGUF:

```text
/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf
size: 571M
scope: encoder_quant_conv
```

- Updated `src/models/lingbot_va.cpp`:
  - added `validate_vae_encoder_tensors()`;
  - added `VLA_LINGBOT_VAE_SMOKE=1`;
  - added `VLA_LINGBOT_VAE_GGUF=/path/to/vae.gguf`.
- Added a first executable VAE Conv3D smoke:
  - `VLA_LINGBOT_VAE_CONV_SMOKE=1`;
  - reads `vae.encoder.conv_in.weight`;
  - restores the flattened Conv3D kernel as ggml shape `[3,3,3,1920]`;
  - runs `ggml_conv_3d` on a deterministic input.

Verification:

```text
python -m py_compile scripts/convert_lingbot_va_to_gguf.py

eval/sim/libero/libero_uv/.venv/bin/python scripts/convert_lingbot_va_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long \
  --dry-run \
  --modules vae_encoder

Result:
dry-run ok: transformer tensor map covers 841 tensors
dry-run ok: VAE tensor map covers 86 tensors

eval/sim/libero/libero_uv/.venv/bin/python scripts/convert_lingbot_va_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long \
  --modules vae_encoder \
  --out /home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf
./build/vla-server --bind tcp://127.0.0.1:5999 /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf

Result:
VAE GGUF metadata ok:
scope=encoder_quant_conv
tensors=86
shapes=86
570.82 MiB
F32=86
other=0
shape sample:
first=vae.encoder.conv_in.bias:160
last=vae.quant_conv.weight:96,96,1,1,1

VLA_LINGBOT_VAE_CONV_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf
./build/vla-server --bind tcp://127.0.0.1:5999 /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf

Result:
VAE conv_in smoke ok:
input=[4,4,3,12]
kernel=[3,3,3,1920]
out=[4,4,3,160]
checksum=0.164595325
max=0.018979881
```

## 2026-06-16 00:35:32 CST - Runtime Predict Bridge / 运行时 Predict 桥接

中文：

- 将 `LingBotVAModelArch::predict()` 从占位异常改成可执行的 WanTransformer runtime bridge。
- 当前 `predict()` 已经接入：
  - BF16 GGUF 读取并临时转 F32 的 common weights / block weights；
  - mixed video/action token packing；
  - WanTransformer block window 执行；
  - ggml dense self-attention 路径；
  - CUDA flex/block-sparse self-attention 路径；
  - video/action `FlowMatchScheduler` 更新；
  - 服务端协议需要的 `chunk=[cfg.n_suffix, cfg.max_action_dim]` 扁平动作输出。
- 将 LingBot 当前默认服务端输出形状设为 `chunk_size=16, action_dim=30`，避免此前 `cfg.n_suffix=0` 导致服务端返回空动作块。
- 将服务端 state 维度先设为 7，方便后续和现有 ZMQ/protobuf 客户端的 proprio state 请求对齐；当前 bridge 暂时还不消费 state。
- 新增 `VLA_LINGBOT_PREDICT_SMOKE=1`，模型加载阶段会直接调用一次 `predict()`，验证 runtime 入口和服务端输出形状。
- 当前 bridge 仍然是 WanTransformer 主干入口，不是完整 LingBot-VA 端到端：
  - UMT5 文本编码未接入；
  - VAE 图像/视频 latent 编码未接入；
  - 真实 LingBot action postprocess / unnormalize / mask 未接入；
  - 当前 latent/action 初值仍使用 synthetic/deterministic 或 request noise。

English:

- Replaced the placeholder `LingBotVAModelArch::predict()` with an executable WanTransformer runtime bridge.
- The current `predict()` path now wires:
  - BF16 GGUF reads with temporary F32 conversion for common/block weights;
  - mixed video/action token packing;
  - WanTransformer block-window execution;
  - ggml dense self-attention;
  - CUDA flex/block-sparse self-attention;
  - video/action `FlowMatchScheduler` updates;
  - the server-facing flat action chunk shaped as `chunk=[cfg.n_suffix, cfg.max_action_dim]`.
- Set LingBot's current server output shape to `chunk_size=16, action_dim=30`, fixing the previous `cfg.n_suffix=0` empty-output state.
- Set the server-facing state dimension to 7 for compatibility with existing ZMQ/protobuf proprio-state requests; the current bridge does not consume state yet.
- Added `VLA_LINGBOT_PREDICT_SMOKE=1`, which calls `predict()` during model load to validate the runtime entrypoint and output shape.
- This bridge is still the WanTransformer backbone path, not full LingBot-VA end-to-end inference:
  - UMT5 text encoding is not wired yet;
  - VAE image/video latent encoding is not wired yet;
  - real LingBot action postprocess / unnormalize / mask is not wired yet;
  - latent/action initial values still come from synthetic deterministic tensors or request noise.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_PREDICT_SMOKE=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1
./build/vla-server --bind tcp://127.0.0.1:5999 /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf

Result:
predict bridge ok blocks=1 mode=ggml-dense window=1 video_steps=1 action_steps=1
chunk=[16,30] checksum=9.79705236 max=0.225708961
vla-server loaded. chunk_size=16 action_dim=30

VLA_LINGBOT_PREDICT_SMOKE=1
VLA_LINGBOT_PREDICT_CUDA_SELF_ATTN=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1
./build/vla-server --bind tcp://127.0.0.1:5999 /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf

Result:
predict bridge ok blocks=1 mode=cuda-flex-self-attn window=1 video_steps=1 action_steps=1
chunk=[16,30] checksum=-4.16295406 max=0.158636853
vla-server loaded. chunk_size=16 action_dim=30

ZMQ/protobuf request smoke:
request_id=42
error=''
chunk_size=16
action_dim=30
action_len=480
checksum=9.77859516069293
latency_ms_total=2749.998779296875
```

### 3. 梳理 Python 原版推理链路 / Mapped Original Python Inference Flow

动作：

Action:

- 阅读 LingBot-VA 本地代码仓库：
- Inspected the local LingBot-VA repository:

```text
/home/xuling/robotic_code/embodied.cpp/lingbot-va-main
```

- 重点阅读：
- Key files inspected:

```text
wan_va/wan_va_server.py
wan_va/modules/model.py
wan_va/modules/utils.py
wan_va/utils/scheduler.py
wan_va/utils/sever_utils.py
evaluation/libero/client.py
evaluation/libero/launch_server.sh
evaluation/libero/launch_client.sh
```

结果：

Result:

- 已记录 `reset -> infer -> compute_kv_cache` 的控制流。
- Recorded the `reset -> infer -> compute_kv_cache` control flow.
- 已确认 LingBot-VA 原版使用 Python WebSocket，移植时仅作为参考。
- Confirmed the original LingBot-VA uses Python WebSocket, which is reference-only for the port.
- 已确认 LIBERO 输入是两路 128x128 RGB 图像。
- Confirmed LIBERO input uses two 128x128 RGB images.
- 已确认推理核心包含 video flow loop、action flow loop 和 KV cache update。
- Confirmed the inference core includes video flow loop, action flow loop, and KV cache update.

### 4. 发现原仓库脚本问题 / Found Original Script Issue

动作：

Action:

- 检查 `evaluation/libero/launch_server.sh`。
- Checked `evaluation/libero/launch_server.sh`.

结果：

Result:

- 发现疑似换行错误：
- Found a likely line-break bug:

```bash
--save_root $save_rootSTART=0
```

- 预期应为：
- Expected form:

```bash
--save_root $save_root

START=0
```

状态：

Status:

- 暂未修改 LingBot-VA 原仓库。等需要运行 Python reference 时再修。
- The original LingBot-VA repository has not been modified yet. Fix this when running the Python reference becomes necessary.

### 5. 新增权重检查工具 / Added Weight Inspection Tool

动作：

Action:

- 新增脚本：
- Added script:

```text
scripts/inspect_lingbot_va_weights.py
```

- 脚本设计为不依赖完整 safetensors 下载。
- The script is designed to work before full safetensors download completes.
- 标准库即可读取 config/index；如果安装了 `safetensors`，可进一步读取 shard header。
- It reads config/index with the standard library; if `safetensors` is installed, it can also inspect shard headers.

验证：

Verification:

```bash
python -m py_compile scripts/inspect_lingbot_va_weights.py
python scripts/inspect_lingbot_va_weights.py
```

结果：

Result:

```text
text_encoder:
  architecture: UMT5EncoderModel
  model_type: umt5
  num_layers: 24
  d_model: 4096
  expected tensors: 242
  expected size: 10.58 GiB
  expected shards: 3
  current missing shards: 3

transformer:
  class: WanTransformer3DModel
  num_layers: 30
  heads: 24
  head_dim: 128
  ffn_dim: 14336
  expected tensors: 841
  expected size: 9.48 GiB
  expected shards: 3
  current missing shards: 3
  config attn_mode: flex

vae:
  class: AutoencoderKLWan
  in_channels: 12
  out_channels: 12
  patch_size: 2
  safetensors files: none found yet

tokenizer:
  current snapshot directory is missing
```

说明：

Notes:

- `transformer/config.json` 当前 `attn_mode` 是 `flex`，推理时需要改为 `torch` 或 `flashattn`。
- `transformer/config.json` currently has `attn_mode=flex`; inference requires `torch` or `flashattn`.
- 当前 HF snapshot 只有 config/index，完整 safetensors 尚未进入 snapshot。
- The current HF snapshot has config/index files only; full safetensors are not yet present in the snapshot.

### 6. 更新规划文档 / Updated Planning Document

动作：

Action:

- 将 inspector 当前结果和 Phase 0-7 路线写入 `LINGBOT_VA_PORT_PLAN.md`。
- Added inspector results and the Phase 0-7 roadmap to `LINGBOT_VA_PORT_PLAN.md`.

结果：

Result:

- 当前路线图包括：
- Current roadmap includes:

```text
Phase 0: code/reference path freeze
Phase 1: weight inventory and GGUF baseline
Phase 2: minimal C++ WanTransformer3DModel forward
Phase 3: flow loop, action path, and state management
Phase 4: UMT5 C++/ggml support
Phase 5: VAE Encoder C++/ggml support
Phase 6: 8 GB deployment optimization and quantization
Phase 7: report and release preparation
```

## 当前状态 / Current Status

```text
Phase 0: in progress, mostly documented
Phase 1: inspector skeleton implemented; waiting for full weights for tensor inventory
Phase 2+: not started
```

### 7. 新增 GGUF metadata converter 骨架 / Added GGUF Metadata Converter Skeleton

动作：

Action:

- 新增脚本：
- Added script:

```text
scripts/convert_lingbot_va_to_gguf.py
```

- 当前支持 `--metadata-only`，用于在完整权重下载前生成只含 metadata 的 GGUF。
- It currently supports `--metadata-only`, which creates a metadata-only GGUF before full weights are downloaded.
- 自动使用仓库内的 `third_party/llama.cpp/gguf-py`，不要求 base Python 额外安装 `gguf`。
- It automatically uses the vendored `third_party/llama.cpp/gguf-py`, so base Python does not need an extra `gguf` install.

验证：

Verification:

```bash
python -m py_compile scripts/convert_lingbot_va_to_gguf.py
python scripts/convert_lingbot_va_to_gguf.py --metadata-only --out /tmp/lingbot_va.metadata.gguf
```

结果：

Result:

```text
writing metadata-only GGUF: /tmp/lingbot_va.metadata.gguf
text_shards_present: no
transformer_shards_present: no
vae_safetensors_present: no
tokenizer_present: no
```

### 8. 注册 LingBot-VA C++ 架构骨架 / Registered LingBot-VA C++ Architecture Skeleton

动作：

Action:

- 更新 `src/arch.h`，新增 `Arch::LINGBOT_VA` 和 `lingbot_va_create(...)`。
- Updated `src/arch.h` with `Arch::LINGBOT_VA` and `lingbot_va_create(...)`.
- 更新 `src/model.cpp`，支持从 `lingbot_va.architecture = lingbot_va` metadata 检测架构并分发。
- Updated `src/model.cpp` to detect `lingbot_va.architecture = lingbot_va` metadata and dispatch it.
- 新增 `src/models/lingbot_va.cpp` runtime 骨架。
- Added `src/models/lingbot_va.cpp` runtime skeleton.
- 更新 `CMakeLists.txt`，将 `src/models/lingbot_va.cpp` 加入 `vla_core`。
- Updated `CMakeLists.txt` to include `src/models/lingbot_va.cpp` in `vla_core`.

验证：

Verification:

```bash
cmake --build build -j2
./build/vla-server /tmp/lingbot_va.metadata.gguf
```

结果：

Result:

```text
vla: arch = lingbot_va
vla(lingbot_va): metadata skeleton loaded from /tmp/lingbot_va.metadata.gguf
vla-server: loaded. chunk_size=0 action_dim=30 n_lang=512 hidden=3072
vla(lingbot_va): runtime is not implemented yet
vla-server: bound to tcp://*:5555. ready.
```

说明：

Notes:

- 当前 runtime 只用于验证 metadata 和架构检测链路，尚不能执行真实推理。
- The current runtime only validates metadata and architecture detection; it cannot perform real inference yet.
- 测试结束后已停止 `vla-server`，未继续占用 5555 端口。
- `vla-server` was stopped after the test and is no longer occupying port 5555.

### 9. 新增 Transformer spec dump 工具 / Added Transformer Spec Dump Tool

动作：

Action:

- 新增脚本：
- Added script:

```text
scripts/dump_lingbot_va_spec.py
```

- 该脚本只依赖 config/index，不需要完整 safetensors。
- The script depends only on config/index files and does not require complete safetensors.
- 输出 `WanTransformer3DModel` 的 config-derived shape、主要 tensor shape、以及暂定 GGUF tensor 命名。
- It outputs config-derived shapes, major tensor shapes, and tentative GGUF tensor names for `WanTransformer3DModel`.

验证：

Verification:

```bash
python -m py_compile scripts/dump_lingbot_va_spec.py
python scripts/dump_lingbot_va_spec.py > /tmp/lingbot_va_transformer_spec.md
python scripts/dump_lingbot_va_spec.py | rg 'Full generated|missing|not concretely|extra'
cmake --build build -j2
```

结果：

Result:

```text
Full generated mapping rows: 841
Concrete mapped tensors missing from index: 0
Index tensors not concretely listed by this draft: 0
```

关键 shape：

Key shapes:

```text
layers: 30
heads: 24
head_dim: 128
inner_dim: 3072
ffn_dim: 14336
text_dim: 4096
patch_size: [1, 2, 2]
latent patch input dim: 192
latent patch output dim: 192
action_dim: 30
```

疑点记录：

Open issue:

- `transformer` index 中同时存在 `patch_embedding.weight/bias` 和 `patch_embedding_mlp.weight/bias`。
- The `transformer` index contains both `patch_embedding.weight/bias` and `patch_embedding_mlp.weight/bias`.
- 当前 Python `WanTransformer3DModel.forward()` 使用的是 `patch_embedding_mlp`，没有看到 `patch_embedding` 被实际调用。
- Current Python `WanTransformer3DModel.forward()` uses `patch_embedding_mlp`; no active use of `patch_embedding` has been observed.
- 暂定策略：converter 和 GGUF 命名先保守保留这组 tensor，标记为 `wvm.patch_embd_legacy.*`，后续通过完整权重加载和 Python reference 确认是否 unused。
- Tentative policy: keep these tensors conservatively in converter/GGUF naming as `wvm.patch_embd_legacy.*`, then confirm whether they are unused through full-weight loading and Python reference checks.

需要用户确认：

User confirmation needed:

- 对于“源码暂未使用但 checkpoint/index 中存在”的 tensor，是否默认保守写入 GGUF，直到确认不会影响兼容性后再移除？
- For tensors that appear in checkpoint/index but are not currently used by source code, should we conservatively write them into GGUF by default until compatibility checks prove they can be removed?

用户确认结果：

User decision:

- 已确认采用保守保留策略。
- Confirmed: use the conservative retention policy.
- 只要 tensor 出现在 checkpoint/index 中，converter 默认写入 GGUF；如果当前 forward 暂未使用，则在命名或注释中标记为 legacy/TBD。
- If a tensor appears in checkpoint/index, the converter should write it into GGUF by default; if current forward does not use it, mark it as legacy/TBD in naming or comments.
- 后续只有在完整权重加载、Python reference 和 C++ 对齐验证都表明该 tensor 可安全移除时，才考虑裁剪。
- Only consider removing such tensors later after full-weight loading, Python reference checks, and C++ alignment prove they are safe to drop.

### 10. 核对完整权重目录 / Verified Completed Weight Directory

动作：

Action:

- 用户提供新的本地权重目录：
- User provided the completed local weight directory:

```text
/home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long
```

- 与 Hugging Face 仓库 `robbyant/lingbot-va-posttrain-libero-long` 的远端文件清单进行对比。
- Compared it against the remote file list from Hugging Face repo `robbyant/lingbot-va-posttrain-libero-long`.
- 使用远端 `resolve/main/...` HEAD 请求逐项核对本地文件大小。
- Used remote `resolve/main/...` HEAD requests to compare local file sizes.
- 使用 `safetensors` 读取本地 shard headers，确认 tensor header 可解析。
- Used `safetensors` to read local shard headers and confirm tensor metadata is parseable.

结果：

Result:

```text
Local model/runtime files: complete
Remote repo files: 19
Local files: 16
Missing local files: 3 non-runtime files
Extra local files: 0
Size mismatches among common files: 0
```

缺失的三个远端文件：

Missing remote files:

```text
.gitattributes
README.md
assets/teaser.png
```

说明：

Notes:

- 这三个文件是仓库说明或展示文件，不影响模型加载和转换。
- These three files are repository metadata or visual assets and do not affect model loading or conversion.
- 所有模型运行相关文件均存在，且本地大小与 Hugging Face 远端大小一致。
- All model/runtime files are present, and local sizes match the Hugging Face remote sizes.

本地运行相关文件：

Local runtime-relevant files:

```text
text_encoder/config.json
text_encoder/model-00001-of-00003.safetensors
text_encoder/model-00002-of-00003.safetensors
text_encoder/model-00003-of-00003.safetensors
text_encoder/model.safetensors.index.json
tokenizer/special_tokens_map.json
tokenizer/spiece.model
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
transformer/config.json
transformer/diffusion_pytorch_model-00001-of-00003.safetensors
transformer/diffusion_pytorch_model-00002-of-00003.safetensors
transformer/diffusion_pytorch_model-00003-of-00003.safetensors
transformer/diffusion_pytorch_model.safetensors.index.json
vae/config.json
vae/diffusion_pytorch_model.safetensors
```

safetensors header 检查：

safetensors header checks:

```text
text_encoder:
  readable tensors: 242
  dtype: BF16
  estimated tensor bytes: 10.58 GiB

transformer:
  readable tensors: 841
  dtype: BF16
  estimated tensor bytes: 9.48 GiB

vae:
  readable tensors: 196
  dtype: F32
  estimated tensor bytes: 2.63 GiB
```

额外注意：

Additional note:

- `transformer/config.json` 当前仍为 `attn_mode: flex`。原版 Python 推理前需要改成 `torch` 或 `flashattn`；vla.cpp 转换/移植时也要记录这一点。
- `transformer/config.json` still has `attn_mode: flex`. Original Python inference requires changing it to `torch` or `flashattn`; the vla.cpp conversion/port should also record this.

待用户确认：

Waiting for user confirmation:

- 是否需要把缺失的 `.gitattributes`、`README.md`、`assets/teaser.png` 也下载到本地目录，使本地目录与 Hugging Face repo 逐文件完全一致？
- Whether to download the missing `.gitattributes`, `README.md`, and `assets/teaser.png` into the local directory so it exactly matches the Hugging Face repo file-for-file.

用户确认结果：

User decision:

- 不下载这三个非运行文件；它们不影响后续实验。
- Do not download these three non-runtime files; they do not affect the following experiments.

计划调整：

Plan adjustment:

- 完整参数文件已经到位，Phase 1 从“等待权重”切换为“实际权重清单、tensor map、GGUF 转换器实现”。
- Full weights are now available, so Phase 1 switches from "waiting for weights" to "actual weight inventory, tensor map, and GGUF converter implementation".
- 后续应先做参数转换和命名映射，再深入写 C++ loader / ggml graph。
- Next, implement parameter conversion and naming maps before going deep into the C++ loader / ggml graph.
- 原因：C++ 驱动层需要依赖 GGUF 中稳定的 tensor 名称、shape、dtype 和必要的 layout 转换。
- Reason: the C++ runtime depends on stable GGUF tensor names, shapes, dtypes, and required layout transforms.

重新接上的 full-weight 检查：

Reconnected full-weight checks:

```bash
eval/sim/libero/libero_uv/.venv/bin/python scripts/inspect_lingbot_va_weights.py \
  /home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long

python scripts/dump_lingbot_va_spec.py \
  /home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long
```

结果：

Result:

```text
text_encoder readable tensors: 242 / 242
transformer readable tensors: 841 / 841
transformer tensor map coverage: 841 / 841
VAE safetensors readable tensors: 196
```

### 11. 接入 Transformer tensor map 到 converter / Added Transformer Tensor Map to Converter

动作：

Action:

- 更新 `scripts/convert_lingbot_va_to_gguf.py`。
- Updated `scripts/convert_lingbot_va_to_gguf.py`.
- 新增 `--dry-run`，用于校验完整权重目录、shard 状态和 transformer tensor map 覆盖。
- Added `--dry-run` to validate full checkpoint status, shard availability, and transformer tensor map coverage.
- 新增 `--modules transformer` 的流式写入路径，但尚未实际写大 GGUF 文件。
- Added streaming write support for `--modules transformer`, but did not write the large GGUF file yet.
- 将 `torch/numpy/safetensors` 改成只在真正写 tensor 时才依赖；metadata-only 和 dry-run 更轻量。
- Deferred `torch/numpy/safetensors` dependency to actual tensor writing; metadata-only and dry-run remain lightweight.

验证：

Verification:

```bash
python -m py_compile scripts/convert_lingbot_va_to_gguf.py

eval/sim/libero/libero_uv/.venv/bin/python scripts/convert_lingbot_va_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long \
  --dry-run

python scripts/convert_lingbot_va_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long \
  --metadata-only \
  --out /tmp/lingbot_va.fullpath.metadata.gguf
```

结果：

Result:

```text
dry-run ok: transformer tensor map covers 841 tensors
text_shards_present: yes
transformer_shards_present: yes
vae_safetensors_present: yes
tokenizer_present: yes

/tmp/lingbot_va.fullpath.metadata.gguf generated successfully
```

当前未执行的大文件操作：

Large operation not yet executed:

```bash
eval/sim/libero/libero_uv/.venv/bin/python scripts/convert_lingbot_va_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long \
  --modules transformer \
  --out /home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long/lingbot_va_transformer_bf16.gguf
```

说明：

Notes:

- 该命令预计会生成约 9.5 GiB 的 transformer-only BF16 GGUF。
- This command is expected to generate a transformer-only BF16 GGUF of about 9.5 GiB.
- 等用户确认后再执行，避免未经确认占用磁盘和时间。
- Wait for user confirmation before running it to avoid consuming disk and time unexpectedly.

### 12. 生成 Transformer-only BF16 GGUF / Generated Transformer-only BF16 GGUF

动作：

Action:

- 用户确认生成 BF16 GGUF，并要求保存到 `/home/xuling/robotic_dataset/models/`，与原始参数目录平行。
- User confirmed generating the BF16 GGUF and requested saving it under `/home/xuling/robotic_dataset/models/`, parallel to the original checkpoint directory.
- 输出路径：
- Output path:

```text
/home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

执行命令：

Command:

```bash
eval/sim/libero/libero_uv/.venv/bin/python scripts/convert_lingbot_va_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long \
  --modules transformer \
  --out /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

首次验证发现的问题：

Issue found on first validation:

```text
gguf_init_from_file_ptr: tensor 'wvm.patch_embd_legacy.weight'
has invalid number of dimensions: 5 > 4
```

原因：

Cause:

```text
patch_embedding.weight shape = [3072, 48, 1, 2, 2]
GGUF / ggml tensor metadata supports up to 4 dimensions.
```

修复：

Fix:

- 更新 `scripts/convert_lingbot_va_to_gguf.py`，当 tensor 维度超过 4 时，将其 reshape 为 `[shape[0], -1]` 再写入 GGUF。
- Updated `scripts/convert_lingbot_va_to_gguf.py`; tensors with more than 4 dimensions are reshaped to `[shape[0], -1]` before writing to GGUF.
- 本次触发的是 legacy/TBD tensor `wvm.patch_embd_legacy.weight`：
- This was triggered by legacy/TBD tensor `wvm.patch_embd_legacy.weight`:

```text
[3072, 48, 1, 2, 2] -> [3072, 192]
```

重新生成结果：

Regeneration result:

```text
streaming transformer tensors: 841 tensors from 3 shards
  shard diffusion_pytorch_model-00001-of-00003.safetensors: 405 tensors
  shard diffusion_pytorch_model-00002-of-00003.safetensors: 403 tensors
  shard diffusion_pytorch_model-00003-of-00003.safetensors: 33 tensors
  note: flattening wvm.patch_embd_legacy.weight from [3072, 48, 1, 2, 2] to [3072, 192] for GGUF
```

输出文件：

Output file:

```text
/home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
size: 9.5 GiB
```

验证：

Verification:

```bash
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

Result:

```text
vla: arch = lingbot_va
vla(lingbot_va): metadata skeleton loaded from /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
vla-server: loaded. chunk_size=0 action_dim=30 n_lang=512 hidden=3072
vla-server: bound to tcp://*:5555. ready.
```

说明：

Notes:

- GGUF 文件现在可被 `vla-server` 读取并识别为 `lingbot_va`。
- The GGUF file can now be read by `vla-server` and detected as `lingbot_va`.
- runtime 仍是骨架，尚不能真实推理。
- The runtime is still a skeleton and cannot perform real inference yet.
- 验证完成后已停止 `vla-server`，未占用 5555 端口。
- `vla-server` was stopped after validation and is not occupying port 5555.

### 13. C++ Transformer GGUF Loader 校验 / C++ Transformer GGUF Loader Validation

动作：

Action:

- 更新 `src/models/lingbot_va.cpp`，从只读基础 metadata 的骨架 loader，推进到 transformer GGUF 张量清单校验 loader。
- Updated `src/models/lingbot_va.cpp` from a basic metadata-only skeleton loader to a transformer GGUF tensor-list validation loader.
- loader 现在会读取 LingBot-VA transformer 配置 metadata，并校验 converter 生成的 841 个 transformer tensor 名称是否全部存在。
- The loader now reads LingBot-VA transformer config metadata and verifies that all 841 transformer tensor names emitted by the converter exist in the GGUF.
- 这一步仍然不分配/上传完整权重，不执行真实 forward，目的是先低风险确认 GGUF 文件结构和 C++ 侧命名规则完全对齐。
- This step still does not allocate/upload full weights or run real forward; the goal is to confirm GGUF structure and C++ naming rules with low risk first.

编译：

Build:

```bash
cmake --build build -j2
```

结果：

Result:

```text
[100%] Built target vla-server
```

运行校验：

Runtime validation:

```bash
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

Result:

```text
vla: arch = lingbot_va
vla(lingbot_va): transformer tensor metadata ok: 841 tensors, 9.48 GiB, BF16=841 F32=0 other=0
vla(lingbot_va): metadata loaded from /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
vla(lingbot_va): transformer=30 layers, hidden=3072, heads=24, head_dim=128, ffn=14336, patch=[1,2,2], action_dim=30, attn_mode_config=flex
vla(lingbot_va): runtime forward is not implemented yet (next milestone: shape-only/minimal WanTransformer3DModel forward)
vla-server: loaded. chunk_size=0  action_dim=30  n_lang=512  hidden=3072  expert_h=0  timing_detail=none
vla-server: bound to tcp://*:5555. ready.
```

说明：

Notes:

- transformer-only BF16 GGUF 与当前 C++ loader 的命名表已经完全对齐。
- The transformer-only BF16 GGUF is fully aligned with the current C++ loader naming table.
- 841 个 transformer tensor 全部是 BF16；这符合当前“先原样/低损失转换，再设计量化策略”的路线。
- All 841 transformer tensors are BF16, matching the current plan of first doing a faithful low-loss conversion before designing quantization.
- 验证完成后已停止 `vla-server`，未占用 5555 端口。
- `vla-server` was stopped after validation and is not occupying port 5555.

### 14. 可选 CPU 权重加载路径 / Optional CPU Weight Loading Path

动作：

Action:

- 更新 `src/models/lingbot_va.cpp`，为 LingBot-VA transformer 增加 GGUF 原始 tensor 读取能力。
- Updated `src/models/lingbot_va.cpp` with raw GGUF tensor reading for the LingBot-VA transformer.
- 新增 opt-in 开关：
- Added an opt-in switch:

```bash
VLA_LINGBOT_LOAD_WEIGHTS=1 ./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

当前策略：

Current policy:

- 默认启动仍然只做 metadata/tensor-list validation，不加载 9.48 GiB transformer 权重。
- Default startup still performs metadata/tensor-list validation only and does not load the 9.48 GiB transformer weights.
- 设置 `VLA_LINGBOT_LOAD_WEIGHTS=1` 时，会把 841 个 BF16 transformer tensor 加载到 CPU resident buffer。
- When `VLA_LINGBOT_LOAD_WEIGHTS=1` is set, the 841 BF16 transformer tensors are loaded into a CPU resident buffer.
- 目前先放在 CPU buffer，而不是默认放 CUDA buffer，避免在 8GB 显存机器上启动即 OOM。
- The first resident buffer target is CPU, not CUDA, to avoid immediate OOM on the 8GB VRAM machine.

验证：

Verification:

```bash
cmake --build build -j2
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

Result:

```text
[100%] Built target vla-server
vla(lingbot_va): transformer tensor metadata ok: 841 tensors, 9.48 GiB, BF16=841 F32=0 other=0
vla(lingbot_va): transformer weights not resident yet; set VLA_LINGBOT_LOAD_WEIGHTS=1 to test CPU loading
vla-server: bound to tcp://*:5555. ready.
```

未执行的重型测试：

Heavy test not run:

- 当前机器 `free -h` 显示可用内存约 10 GiB；完整 transformer 权重为 9.48 GiB，边界较近。
- Current `free -h` shows about 10 GiB available memory; full transformer weights are 9.48 GiB, close to the edge.
- 因此本轮未强行执行 `VLA_LINGBOT_LOAD_WEIGHTS=1`，避免桌面会话换页严重或 OOM。
- Therefore this turn did not force-run `VLA_LINGBOT_LOAD_WEIGHTS=1`, to avoid severe swapping or OOM in the desktop session.

### 15. Transformer Typed Weight Layout / Transformer Typed Weight Layout

动作：

Action:

- 继续更新 `src/models/lingbot_va.cpp`，把 841 个 transformer tensor 从“扁平字符串列表”整理为 typed C++ weight layout。
- Continued updating `src/models/lingbot_va.cpp` by organizing the 841 transformer tensors from a flat string list into a typed C++ weight layout.
- 当前 layout 覆盖：
- Current layout covers:

```text
patch_embd_mlp
patch_embd_legacy
action_embd
output_proj
action_out
output_scale_shift
cond
action_cond
30 x blocks:
  scale_shift
  cross_norm
  self_attn q/k/v/o + q_norm/k_norm
  cross_attn q/k/v/o + q_norm/k_norm
  ffn_up
  ffn_down
```

说明：

Notes:

- typed layout 只在 `VLA_LINGBOT_LOAD_WEIGHTS=1` 时创建和加载，不影响默认轻量启动。
- The typed layout is created and loaded only when `VLA_LINGBOT_LOAD_WEIGHTS=1` is set, so default lightweight startup is unaffected.
- 这一步是实现 ggml forward 前的结构准备：后续每个 C++ forward 子模块可以直接引用清晰的 typed weight pointer。
- This prepares the structure for the ggml forward implementation: each later C++ forward submodule can refer to clear typed weight pointers.
- layout 内部会检查创建出的 tensor 数量是否等于 converter/validator 期望的 841 个。
- The layout internally checks that the number of created tensors equals the 841 tensors expected by the converter/validator.

验证：

Verification:

```bash
cmake --build build -j2
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

Result:

```text
[100%] Built target vla-server
vla(lingbot_va): transformer tensor metadata ok: 841 tensors, 9.48 GiB, BF16=841 F32=0 other=0
vla(lingbot_va): transformer weights not resident yet; set VLA_LINGBOT_LOAD_WEIGHTS=1 to test CPU loading
vla-server: bound to tcp://*:5555. ready.
```

服务状态：

Service status:

- 验证后已停止 `vla-server`。
- `vla-server` was stopped after validation.

### 16. WanTransformer No-Alloc Shape Smoke / WanTransformer No-Alloc Shape Smoke

动作：

Action:

- 在 `src/models/lingbot_va.cpp` 中新增 `VLA_LINGBOT_SHAPE_SMOKE=1` 调试路径。
- Added a `VLA_LINGBOT_SHAPE_SMOKE=1` debug path in `src/models/lingbot_va.cpp`.
- 该路径会创建 LingBot-VA typed weight layout，并搭建一个 no-alloc ggml graph，但不分配真实权重数据、不执行计算。
- This path creates the LingBot-VA typed weight layout and builds a no-alloc ggml graph, without allocating real weight data or executing computation.
- graph 覆盖 latent-mode 和 action-mode 两条路径：
- The graph covers both latent-mode and action-mode paths:

```text
latent/action input projection
text projection
time projection
30 x transformer blocks:
  norm
  self attention
  cross attention to text
  FFN
output projection / action output projection
```

验证命令：

Verification command:

```bash
cmake --build build -j2
VLA_LINGBOT_SHAPE_SMOKE=1 ./build/vla-server \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

Result:

```text
[100%] Built target vla-server
vla(lingbot_va): transformer tensor metadata ok: 841 tensors, 9.48 GiB, BF16=841 F32=0 other=0
vla(lingbot_va): VLA_LINGBOT_SHAPE_SMOKE is set; building no-alloc WanTransformer shape graph
vla(lingbot_va): shape smoke graph ok: latent_out=[192,16] action_out=[30,8]
vla-server: bound to tcp://*:5555. ready.
```

当前简化点：

Current simplifications:

- 还没有接入 Wan 3D RoPE；shape smoke 先验证非 RoPE 的 attention shape flow。
- Wan 3D RoPE is not wired yet; the shape smoke first validates the non-RoPE attention shape flow.
- 还没有接入 AdaLN 的 timestep scale/shift/gate；time projection 已建图，但暂时未参与 block 内调制。
- AdaLN timestep scale/shift/gate is not wired yet; time projection is in the graph but not yet used for block modulation.
- FFN 暂时使用 `ggml_gelu`，后续需要改成更贴近 Diffusers `gelu-approximate` 的形式。
- 已复核：Diffusers `gelu-approximate` 在这里实际对应 PyTorch `GELU(approximate="tanh")`，`ggml_gelu` 与该方向一致，暂不需要改成 sigmoid quick GELU。
- Rechecked: Diffusers `gelu-approximate` here maps to PyTorch `GELU(approximate="tanh")`; `ggml_gelu` is aligned with that direction, so it should not be replaced with sigmoid quick GELU for now.
- 还没有实现真实 KV cache、真实权重 resident compute 或 Python parity。
- Real KV cache, resident-weight compute, and Python parity are not implemented yet.

服务状态：

Service status:

- 验证后已停止 `vla-server`。
- `vla-server` was stopped after validation.

### 17. AdaLN Shape Wiring and RoPE Design Check / AdaLN Shape Wiring and RoPE Design Check

动作：

Action:

- 在 `src/models/lingbot_va.cpp` 的 shape graph 中接入 timestep AdaLN 的 6 路调制：
- Wired the 6-way timestep AdaLN modulation into the shape graph in `src/models/lingbot_va.cpp`:

```text
shift_msa
scale_msa
gate_msa
c_shift_msa
c_scale_msa
c_gate_msa
```

- 同时接入最终 output norm 的 `output_scale_shift`。
- Also wired the final output norm `output_scale_shift`.

验证：

Verification:

```bash
cmake --build build -j2
VLA_LINGBOT_SHAPE_SMOKE=1 ./build/vla-server \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

Result:

```text
[100%] Built target vla-server
vla(lingbot_va): shape smoke graph ok: latent_out=[192,16] action_out=[30,8]
vla-server: bound to tcp://*:5555. ready.
```

RoPE 判断：

RoPE assessment:

- LingBot-VA/Wan 的 3D RoPE 在 Python 中按 `f/h/w` 三个轴分别计算：
- LingBot-VA/Wan 3D RoPE computes three axes separately in Python:

```text
f_dim = head_dim - 2 * (head_dim // 3) = 44
h_dim = head_dim // 3 = 42
w_dim = head_dim // 3 = 42
```

- ggml 已有 `ggml_rope_multi`，但其 MROPE 注释说明 theta index 在 sections 之间不重置；Wan RoPE 对每个轴按各自维度重新计算频率。
- ggml provides `ggml_rope_multi`, but its MROPE notes say theta indices do not reset across sections; Wan RoPE recomputes frequencies per axis using each axis dimension.
- 因此这里不能盲目套 `ggml_rope_multi`；需要做 Python parity 验证，或实现一个 Wan-specific 3D RoPE 路径。
- Therefore we should not blindly use `ggml_rope_multi`; this needs Python parity validation or a Wan-specific 3D RoPE path.
- 这是目前第一个明确的“可能需要自定义/差异化设计”的模块。
- This is the first clearly identified module that may need custom/differentiated design.

服务状态：

Service status:

- 验证后已停止 `vla-server`。
- `vla-server` was stopped after validation.

### 18. RoPE Parity Harness and Wan-Specific ggml Shape Path / RoPE Parity Harness and Wan-Specific ggml Shape Path

动作：

Action:

- 新增 `scripts/check_lingbot_va_rope.py`，用于比较 LingBot-VA/Wan Python RoPE 与 ggml-style MROPE 复用候选。
- Added `scripts/check_lingbot_va_rope.py` to compare LingBot-VA/Wan Python RoPE with ggml-style MROPE reuse candidates.
- 脚本不加载模型权重，只比较 RoPE 数学和排列。
- The script does not load model weights; it only compares RoPE math and layout.

验证命令：

Verification command:

```bash
eval/sim/libero/libero_uv/.venv/bin/python scripts/check_lingbot_va_rope.py
```

结果：

Result:

```text
head_dim: 128
wan dims: f=44 h=42 w=42
pair sections: f=22 h=21 w=21

## latent
max |Wan freq - MROPE-no-reset freq|: 2.9938394
max |Wan apply - normal(no-reset-freq) apply|: 5.9081898
max |Wan apply - NEOX(no-reset-freq) apply|: 5.8436852

## action
max |Wan freq - MROPE-no-reset freq|: 0.99794647
max |Wan apply - normal(no-reset-freq) apply|: 3.4008822
max |Wan apply - NEOX(no-reset-freq) apply|: 3.655869
```

判断：

Assessment:

- 直接复用 `ggml_rope_multi` 的 MROPE 风格并不等价；差异不是浮点噪声，而是数学/排列规则差异。
- Directly reusing `ggml_rope_multi` in an MROPE-style layout is not equivalent; the difference is mathematical/layout-level, not floating-point noise.
- action grid 还包含 fractional frame offset，这进一步增加了直接套 int-position RoPE 的风险。
- The action grid also includes fractional frame offsets, further increasing the risk of direct int-position RoPE reuse.

后续实现选择：

Implementation choice:

- 已在 `src/models/lingbot_va.cpp` 中用 ggml 基础算子接入 Wan-specific RoPE shape path：
- Wired a Wan-specific RoPE shape path in `src/models/lingbot_va.cpp` using basic ggml ops:

```text
reshape_4d -> view even/odd pair components -> repeat cos/sin -> elementwise rotate -> concat -> reshape_3d
```

- 这不是直接调用 `ggml_rope_multi`，但仍然复用 ggml 基础算子，不需要新增 ggml/CUDA 算子。
- This does not directly call `ggml_rope_multi`, but it still reuses basic ggml ops and does not require a new ggml/CUDA op.
- 如果后续真实计算发现这一路性能太慢，再考虑 fused Wan 3D RoPE kernel；那时可以作为工程优化/difference 候选。
- If real compute later shows this path is too slow, we can consider a fused Wan 3D RoPE kernel; that would be a candidate engineering optimization/difference.

验证：

Verification:

```bash
cmake --build build -j2
VLA_LINGBOT_SHAPE_SMOKE=1 ./build/vla-server \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

Result:

```text
[100%] Built target vla-server
vla(lingbot_va): shape smoke graph ok: latent_out=[192,16] action_out=[30,8]
vla-server: bound to tcp://*:5555. ready.
```

服务状态：

Service status:

- 验证后已停止 `vla-server`。
- `vla-server` was stopped after validation.

### 19. One-Block Real CPU Compute Smoke / One-Block Real CPU Compute Smoke

动作：

Action:

- 在 `src/models/lingbot_va.cpp` 中新增 `VLA_LINGBOT_COMPUTE_SMOKE=1` 调试路径。
- Added a `VLA_LINGBOT_COMPUTE_SMOKE=1` debug path in `src/models/lingbot_va.cpp`.
- 该路径只加载 latent path 的必要权重和第 0 个 transformer block，不加载完整 30 层。
- This path only loads the weights needed by the latent path plus transformer block 0, not the full 30-layer model.
- 覆盖的真实计算链路：
- Real compute path covered:

```text
GGUF raw tensor read
BF16 -> F32 smoke conversion
CPU resident weight buffer
synthetic latent/text/time/rope inputs
patch embedding
text/time projections
block0 self-attn + Wan-specific RoPE
block0 cross-attn
block0 FFN
final output norm/projection
ggml_backend_graph_compute on CPU
```

首次问题：

First issue:

```text
binary_op: unsupported types: dst: f32, src0: f32, src1: bf16
```

原因：

Cause:

- GGUF transformer 权重都是 BF16，而 smoke 输入和中间激活是 F32；CPU binary add 不支持部分 `F32 + BF16` 混合路径。
- GGUF transformer weights are BF16 while smoke inputs and activations are F32; CPU binary add does not support some `F32 + BF16` mixed paths.

修复：

Fix:

- compute smoke 专用路径把读取到的 BF16 权重转换为 F32 resident tensors。
- The compute-smoke-only path converts BF16 weights into F32 resident tensors.
- 这不是最终部署策略，只是为了先验证真实 compute 链路。
- This is not the final deployment strategy; it is only to validate the real compute path first.

验证命令：

Verification command:

```bash
cmake --build build -j2
VLA_LINGBOT_COMPUTE_SMOKE=1 ./build/vla-server \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

Result:

```text
[100%] Built target vla-server
vla(lingbot_va): compute smoke loaded 967.95 MiB into CPU F32 buffer (42 tensors)
vla(lingbot_va): compute smoke ok: out=[192,2] checksum=-8.471 max_abs=0.773172
vla-server: bound to tcp://*:5555. ready.
```

判断：

Assessment:

- GGUF -> C++ tensor -> resident CPU buffer -> ggml graph compute 这一条链路已经跑通。
- The GGUF -> C++ tensor -> resident CPU buffer -> ggml graph compute chain is now working.
- 完整模型不能走全 F32 resident；完整 transformer F32 会接近 19 GiB，只能作为 smoke/debug 策略。
- The full model cannot use full F32 resident weights; the full transformer would be about 19 GiB in F32, so this is smoke/debug only.
- 后续 8GB 目标需要 BF16 resident、量化 resident、分层流式加载，或三者结合。
- The later 8GB target needs BF16 resident, quantized resident, per-layer streaming, or a combination.
- CPU BF16 混合算子限制是后续 dtype 策略必须处理的问题。
- CPU BF16 mixed-op limitations must be handled in the later dtype strategy.

阶段性 dtype 决策：

Stage dtype decision:

- 当前 smoke/debug 阶段继续使用 BF16 GGUF 作为权重来源，但在小规模真实计算 smoke 中统一转换为 F32 resident tensors。
- In the current smoke/debug stage, keep BF16 GGUF as the weight source, but convert to F32 resident tensors for small real-compute smoke tests.
- 这样做的目标是验证结构、shape、命名、GGUF 读取、ggml graph 和 Python parity，而不是提前确定最终部署 dtype。
- The goal is to validate structure, shape, naming, GGUF reading, ggml graph execution, and Python parity, not to decide the final deployment dtype yet.
- 最终 BF16/量化/分层加载策略在结构正确后系统设计。
- Final BF16/quantization/layer-streaming strategy will be designed systematically after correctness is established.

服务状态：

Service status:

- 验证后已停止 `vla-server`。
- `vla-server` was stopped after validation.

### 20. Latent + Action One-Block Compute Smoke / Latent + Action One-Block Compute Smoke

动作：

Action:

- 扩展 `VLA_LINGBOT_COMPUTE_SMOKE=1`，从只验证 latent path，升级为同时验证 latent path 和 action path。
- Extended `VLA_LINGBOT_COMPUTE_SMOKE=1` from latent-path only to both latent and action paths.
- 新增加载：
- Newly loaded:

```text
wvm.action_embd.*
wvm.action_out.*
wvm.action_cond.*
```

- 两条路径共享 block0 self-attn/cross-attn/FFN，并分别使用自己的 input/condition/output projection。
- Both paths share block0 self-attn/cross-attn/FFN while using their own input/condition/output projections.

验证命令：

Verification command:

```bash
cmake --build build -j2
VLA_LINGBOT_COMPUTE_SMOKE=1 ./build/vla-server \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

Result:

```text
[100%] Built target vla-server
vla(lingbot_va): compute smoke loaded 1307.79 MiB into CPU F32 buffer (56 tensors)
vla(lingbot_va): compute smoke latent ok: out=[192,2] checksum=-8.471 max_abs=0.773172
vla(lingbot_va): compute smoke action ok: out=[30,2] checksum=-1.86309 max_abs=0.857539
vla-server: bound to tcp://*:5555. ready.
```

判断：

Assessment:

- latent/action 两条核心 transformer 一层路径均已可真实执行。
- Both latent and action one-block transformer paths can now execute real computation.
- 当前仍是 smoke/debug，所有读取的 BF16 权重在 resident buffer 内转换为 F32。
- This remains smoke/debug only; all loaded BF16 weights are converted to F32 in the resident buffer.
- 下一步应转向 Python parity，而不是继续扩大 C++ 层数。
- The next step should move to Python parity instead of simply adding more C++ layers.

服务状态：

Service status:

- 验证后已停止 `vla-server`。
- `vla-server` was stopped after validation.

### 21. One-Block Python Parity Harness / One-Block Python Parity Harness

动作：

Action:

- 新增 `VLA_LINGBOT_SMOKE_DUMP_DIR`，让 C++ compute smoke 可以把 latent/action 输出和中间张量 dump 到 F32 binary 文件。
- Added `VLA_LINGBOT_SMOKE_DUMP_DIR` so the C++ compute smoke can dump latent/action outputs and intermediate tensors as F32 binary files.
- 新增 `scripts/run_lingbot_va_oneblock_parity.py`，用 PyTorch 从同一个 GGUF 读取权重，复刻 C++ one-block smoke 的合成输入和计算路径。
- Added `scripts/run_lingbot_va_oneblock_parity.py`, which reads weights from the same GGUF in PyTorch and mirrors the C++ one-block smoke synthetic inputs and compute path.
- 修正了两个 parity 基础问题：
- Fixed two parity-foundation issues:

```text
1. gguf-py tensor.shape is in ggml ne order; PyTorch parity needs reversed shape.
2. C++ .f32 dump is ggml ne0-contiguous; Python compare must reshape as [ne1, ne0].T.
```

验证命令：

Verification command:

```bash
rm -rf /tmp/lingbot_va_parity && mkdir -p /tmp/lingbot_va_parity
cmake --build build -j2
VLA_LINGBOT_COMPUTE_SMOKE=1 \
VLA_LINGBOT_SMOKE_DUMP_DIR=/tmp/lingbot_va_parity \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf

eval/sim/libero/libero_uv/.venv/bin/python \
  scripts/run_lingbot_va_oneblock_parity.py \
  --dump-dir /tmp/lingbot_va_parity
```

当前结果摘要：

Current result summary:

```text
latent_x_emb:        max_abs_diff=0.0826  mean_abs_diff=0.00375
latent_text:         max_abs_diff=0.0358  mean_abs_diff=0.00767
latent_t_hidden:     max_abs_diff=0.00814 mean_abs_diff=0.000240
latent_timestep_proj:max_abs_diff=0.0390  mean_abs_diff=0.00345
latent_block_out:    max_abs_diff=24.23   mean_abs_diff=0.247
latent final:        max_abs_diff=1.878   mean_abs_diff=0.486

action_x_emb:        max_abs_diff=0.00449 mean_abs_diff=0.000556
action_text:         max_abs_diff=0.0358  mean_abs_diff=0.00767
action_t_hidden:     max_abs_diff=0.00769 mean_abs_diff=0.000281
action_timestep_proj:max_abs_diff=0.0363  mean_abs_diff=0.00378
action_block_out:    max_abs_diff=54.34   mean_abs_diff=0.345
action final:        max_abs_diff=0.799   mean_abs_diff=0.159
```

判断：

Assessment:

- GGUF 读取、BF16->F32 转换、基础 linear 布局、text/time projection 已基本对齐。
- GGUF reading, BF16->F32 conversion, basic linear layout, and text/time projections are mostly aligned.
- 主要偏差从 `block_out` 开始放大，说明下一步应拆 block 内部：
- The main divergence starts at `block_out`, so the next step is to split the block internals:

```text
AdaLN n1
self-attn output
post-self residual
cross-attn norm/output
FFN input/output
```

- 当前不应继续盲目增加层数；先定位 block 内第一个明显偏差点。
- We should not blindly add more layers yet; first locate the first major divergence inside block0.

服务状态：

Service status:

- 验证后已停止 `vla-server`。
- `vla-server` was stopped after validation.

### 22. Block-Internal Parity Fixes / Block-Internal Parity Fixes

动作：

Action:

- 扩展 C++ compute smoke dump，加入 block0 内部检查点：
- Extended C++ compute smoke dumps with block0-internal checkpoints:

```text
n1
self_q / self_k / self_v
self_attn
post_self
n2
cross_attn
post_cross
n3
ff
```

- 扩展 Python parity 脚本，逐点比较这些 tensor。
- Extended the Python parity script to compare these tensors point by point.

发现与修复：

Findings and fixes:

1. 中间 tensor dump 需要 `ggml_set_output`。
   Intermediate tensor dumps need `ggml_set_output`.

```text
否则 ggml graph allocator 可能复用中间 buffer，导致 dump 出来的值不是该节点最终值。
Otherwise ggml's graph allocator may reuse intermediate buffers, so dumped values may not be the final value of that node.
```

2. attention merge 排列必须贴近 Python/Wan 原版。
   Attention merge layout must follow the native Python/Wan path.

```text
Final C++ path now flattens attention context as [token, heads, head_dim],
matching the usual PyTorch/Wan semantics before the output projection.
```

3. 合成输入生成顺序需要匹配 ggml memory order。
   Synthetic input generation must match ggml memory order.

```text
C++ fills ggml memory with ne0-contiguous order.
Python parity now reshapes input as [ne1, ne0].T for logical [ne0, ne1].
```

当前 parity 结果摘要：

Current parity summary:

```text
latent_x_emb:         max=1.49e-08 mean=5.99e-10
latent_t_hidden:      max=2.61e-08 mean=7.81e-10
latent_timestep_proj: max=1.25e-06 mean=5.32e-08
latent_n1:            max=3.58e-06 mean=1.11e-07
latent_self_q/k/v:    max <= 2.86e-06
latent_self_attn:     max=0.0161 mean=0.00370
latent_cross_attn:    max=0.0156 mean=0.00129
latent_ff:            max=0.185  mean=0.0167
latent_block_out:     max=0.173  mean=0.00188
latent final:         max=0.0124 mean=0.00351

action_x_emb:         max=1.49e-08 mean=4.28e-11
action_t_hidden:      max=2.24e-08 mean=7.71e-10
action_timestep_proj: max=8.34e-07 mean=5.16e-08
action_n1:            max=2.86e-06 mean=6.80e-08
action_self_q/k/v:    max <= 1.91e-06
action_self_attn:     max=0.0102 mean=0.00213
action_cross_attn:    max=0.00887 mean=0.00149
action_ff:            max=0.548  mean=0.0615
action_block_out:     max=0.535  mean=0.00560
action final:         max=0.00888 mean=0.00239
```

判断：

Assessment:

- one-block latent/action smoke 已基本与 Python parity 对齐。
- The one-block latent/action smoke is now mostly aligned with Python parity.
- 线性层、权重读取、BF16->F32、AdaLN、q/k/v projection 都已经高度一致。
- Linear layers, weight reading, BF16->F32 conversion, AdaLN, and q/k/v projection are highly aligned.
- 剩余差异主要来自 attention/FFN 的浮点执行细节；attention context 排列已经改为 Python/Wan native 路径。
- Remaining differences mostly come from attention/FFN floating-point execution details; the attention context layout now follows the Python/Wan native path.

关键决策：

Key decision:

- C++ attention context 的 flatten 排列已改为贴近 Python/Wan 原版，即 `[token, heads, head_dim]` 语义。
- The C++ attention context flattening has been changed to follow the native Python/Wan semantics, i.e. `[token, heads, head_dim]`.
- 原因：后续全模型 parity、权重解释、调试和技术报告表述都会更清晰。
- Reason: this will make full-model parity, weight interpretation, debugging, and technical-report writing clearer later.

## 下一步 / Next Steps

```text
1. Expand compute smoke from block0 to configurable N blocks while keeping F32 debug resident weights.
2. Design the 8GB VRAM execution policy:
   CPU resident weights, staged GPU uploads, quantized resident weights, or per-block streaming.
3. Keep UMT5 and VAE as required later modules:
   first align transformer execution, then add text encoder and VAE operators/weights.
4. Before real action prediction, define the staged test inputs:
   synthetic shape test -> Python parity on saved intermediate tensors -> LIBERO smoke run.
```

## 关键标注：Attention Context 排列 / Important Note: Attention Context Layout

中文：

- 我们的目标是让 C++ attention context 的 flatten 语义贴近 Python/Wan 原版。
- 也就是每个 token 的 context 按 `[heads, head_dim]` 展开，再进入 `o` projection。
- 这个点很重要，因为后续全模型 parity、调试中间张量、解释权重含义、以及 technical report 表述都会围绕这个排列展开。
- 需要特别注意：`ggml_permute` 的参数语义和 PyTorch `permute` 不完全同向直观，必须通过 parity 验证，而不能只看参数顺序猜测。
- C++ merge 写法已校正：ggml 的中间 context 先形成稳定的 q-d-h flatten，再显式重排成 Python/Wan native `[heads, head_dim]` flatten。

English:

- The target is for the C++ attention context flattening to follow the native Python/Wan semantics.
- For each token, the context should be flattened as `[heads, head_dim]` before the output projection.
- This matters for full-model parity, intermediate tensor debugging, weight interpretation, and the technical report.
- Be careful: `ggml_permute` does not read exactly like PyTorch `permute`; the layout must be verified with parity, not guessed from argument order alone.
- The C++ merge expression has been corrected: the ggml intermediate context is first materialized in the stable q-d-h flattening, then explicitly reordered to the Python/Wan native `[heads, head_dim]` flattening.

验证结果：

Verification result:

```text
Build: cmake --build build -j2
Smoke: VLA_LINGBOT_COMPUTE_SMOKE=1 with BF16 GGUF -> F32 debug weights

latent out checksum=12.094, max_abs=1.64022
action out checksum=2.11393, max_abs=0.562055

latent_self_attn: max_abs_diff=0.0161, mean_abs_diff=0.00370
action_self_attn: max_abs_diff=0.0102, mean_abs_diff=0.00213
latent final:     max_abs_diff=0.0124, mean_abs_diff=0.00351
action final:     max_abs_diff=0.00888, mean_abs_diff=0.00239
```

## 多 Block Compute Smoke / Multi-Block Compute Smoke

中文：

- compute smoke 已从固定 `wvm.blk.0` 扩展为可配置的前 N 个 transformer block。
- 默认仍然是 1 个 block，保持旧的 one-block smoke 行为。
- 新增环境变量：

```bash
VLA_LINGBOT_SMOKE_BLOCKS=2
```

- Python parity 脚本同步新增参数：

```bash
scripts/run_lingbot_va_oneblock_parity.py --blocks 2
```

- dump 文件名保持不变，但当 `blocks > 1` 时，`latent_self_attn` / `action_self_attn` 等中间 dump 表示“最后一个执行 block”的 trace。例如 `VLA_LINGBOT_SMOKE_BLOCKS=2` 时，它们对应 `wvm.blk.1`。
- 这个设计方便逐步扩大验证范围：1 block -> 2 blocks -> 更多 blocks，同时保持 dump 和 parity 脚本简单。

English:

- The compute smoke path now supports a configurable number of leading transformer blocks instead of being hardcoded to `wvm.blk.0`.
- The default remains 1 block, preserving the previous one-block smoke behavior.
- New environment variable:

```bash
VLA_LINGBOT_SMOKE_BLOCKS=2
```

- The Python parity script now accepts:

```bash
scripts/run_lingbot_va_oneblock_parity.py --blocks 2
```

- Dump names are unchanged, but when `blocks > 1`, intermediate dumps such as `latent_self_attn` / `action_self_attn` refer to the last executed block. For example, with `VLA_LINGBOT_SMOKE_BLOCKS=2`, they correspond to `wvm.blk.1`.
- This lets us expand verification gradually: 1 block -> 2 blocks -> more blocks, while keeping the dump/parity workflow simple.

验证结果：

Verification result:

```text
Build: cmake --build build -j2

1 block:
loaded 1307.79 MiB F32 debug weights, 56 tensors
latent final:      max_abs_diff=0.0124, mean_abs_diff=0.00351
action final:      max_abs_diff=0.00888, mean_abs_diff=0.00239
latent_self_attn:  max_abs_diff=0.0161, mean_abs_diff=0.00370
action_self_attn:  max_abs_diff=0.0102, mean_abs_diff=0.00213

2 blocks:
loaded 1932.09 MiB F32 debug weights, 83 tensors
latent final:      max_abs_diff=0.0243, mean_abs_diff=0.00580
action final:      max_abs_diff=0.0125, mean_abs_diff=0.00419
latent_self_attn:  max_abs_diff=0.103,  mean_abs_diff=0.0224
action_self_attn:  max_abs_diff=0.108,  mean_abs_diff=0.0239
```

判断：

Assessment:

- 多 block 路径已经跑通，2 block parity 没有出现排列级别错误或数值爆炸。
- The multi-block path is working; 2-block parity shows no layout-level error or numerical blow-up.
- 误差随 block 数增加而累积是预期现象，后续扩到更多 block 时需要持续观察 final/action 输出和最后一个 block 的 attention/FFN。
- Some error accumulation with more blocks is expected; as we expand to more blocks, we should keep monitoring final/action outputs and the last block's attention/FFN.

## 加速推进 Block 数 / Faster Block Scaling

中文：

- 按协同决策，block 数推进不再只逐个增加，而是先 `1, 2, 3`，稳定后跳到 `5, 7`，再跳到 `10, 15, 20`。
- 当前已经完成到 20 blocks，也就是 30 层 transformer 主干的前 2/3。
- 20 blocks 仍然没有出现排列级别错误或数值爆炸，说明 attention/context 排列、block 内结构和权重读取逻辑整体是稳定的。
- 暂时没有继续直接跑 30 blocks，主要原因是当前 smoke 使用 BF16 GGUF -> F32 调试权重常驻 CPU。20 blocks 已加载 13.17 GiB F32 权重，完整 30 blocks 预计接近 19 GiB，会把验证重点从“模型逻辑正确性”混到“本机内存压力测试”。

English:

- Following the joint decision, block scaling is no longer strictly one-by-one: we used `1, 2, 3`, then jumped to `5, 7`, then to `10, 15, 20` after the results remained stable.
- We have now verified 20 blocks, i.e. the first two thirds of the 30-layer transformer trunk.
- 20 blocks still show no layout-level error or numerical blow-up, which suggests that the attention/context layout, block internals, and weight-loading logic are generally stable.
- We did not immediately run 30 blocks because the current smoke path keeps BF16 GGUF weights as F32 debug tensors on CPU. 20 blocks already load 13.17 GiB of F32 weights; full 30 blocks is expected to approach 19 GiB, which would turn the check into a local memory stress test rather than a clean model-logic verification.

推进结果：

Scaling results:

```text
3 blocks:
loaded 2556.39 MiB, 110 tensors
latent final:      max_abs_diff=0.0377, mean_abs_diff=0.00797
action final:      max_abs_diff=0.0105, mean_abs_diff=0.00305
latent_self_attn:  max_abs_diff=0.118,  mean_abs_diff=0.0226
action_self_attn:  max_abs_diff=0.223,  mean_abs_diff=0.0460

5 blocks:
loaded 3804.99 MiB, 164 tensors
latent final:      max_abs_diff=0.0457, mean_abs_diff=0.00764
action final:      max_abs_diff=0.00800, mean_abs_diff=0.00256
latent_self_attn:  max_abs_diff=0.143,  mean_abs_diff=0.0313
action_self_attn:  max_abs_diff=0.0754, mean_abs_diff=0.0140

7 blocks:
loaded 5053.59 MiB, 218 tensors
latent final:      max_abs_diff=0.0381, mean_abs_diff=0.00707
action final:      max_abs_diff=0.0129, mean_abs_diff=0.00478
latent_self_attn:  max_abs_diff=0.136,  mean_abs_diff=0.0271
action_self_attn:  max_abs_diff=0.174,  mean_abs_diff=0.0328

10 blocks:
loaded 6926.49 MiB, 299 tensors
latent final:      max_abs_diff=0.0296, mean_abs_diff=0.00737
action final:      max_abs_diff=0.0167, mean_abs_diff=0.00567
latent_self_attn:  max_abs_diff=0.114,  mean_abs_diff=0.0213
action_self_attn:  max_abs_diff=0.220,  mean_abs_diff=0.0412

15 blocks:
loaded 10048.00 MiB, 434 tensors
latent final:      max_abs_diff=0.0646, mean_abs_diff=0.0185
action final:      max_abs_diff=0.0215, mean_abs_diff=0.00673
latent_self_attn:  max_abs_diff=0.266,  mean_abs_diff=0.0506
action_self_attn:  max_abs_diff=0.336,  mean_abs_diff=0.0721

20 blocks:
loaded 13169.50 MiB, 569 tensors
latent final:      max_abs_diff=0.0951, mean_abs_diff=0.0295
action final:      max_abs_diff=0.0365, mean_abs_diff=0.0118
latent_self_attn:  max_abs_diff=0.614,  mean_abs_diff=0.122
action_self_attn:  max_abs_diff=0.426,  mean_abs_diff=0.0923
```

下一步判断：

Next judgement:

- 如果继续验证完整 30 blocks，建议先改造 debug 权重策略，例如 BF16 常驻、分 block 加载、或只 dump final，避免 F32 全量常驻。
- If we continue to full 30-block verification, we should first improve the debug weight strategy, such as BF16-resident weights, per-block loading, or final-only dumps, to avoid fully resident F32 debug weights.

## 30 Block Final-Only 尝试 / 30-Block Final-Only Attempt

中文：

- 按计划先尝试最小改动方案 A：`VLA_LINGBOT_SMOKE_FINAL_ONLY=1`，只保留最终 `latent/action` 输出，不 dump 中间 tensor。
- 判断“稳定”的标准是：同一组合成输入、同一份 GGUF 权重，分别跑 Python parity 路径和 C++/ggml 路径，然后比较最终输出和必要中间张量。final-only 只比较最终 `latent/action`。
- 1 block final-only 回归通过，结果与之前一致。
- 30 block final-only 能成功加载全部 F32 debug 权重，但在 600 秒 timeout 内没有完成计算和 final dump。
- 这说明 A 路径不是逻辑失败，而是当前机器上 F32 全量常驻 + CPU/swap 计算太重，不适合作为常规完整 30 block 验证方案。

English:

- We first tried plan A with the smallest code change: `VLA_LINGBOT_SMOKE_FINAL_ONLY=1`, keeping only the final `latent/action` outputs and skipping intermediate tensor dumps.
- “Stable” means running the same synthetic inputs and the same GGUF weights through both the Python parity path and the C++/ggml path, then comparing final outputs and necessary intermediate tensors. In final-only mode we compare only final `latent/action`.
- The 1-block final-only regression passed and matched the previous result.
- The 30-block final-only run successfully loaded all F32 debug weights, but did not finish computation or write final dumps within a 600-second timeout.
- This is not a model-logic failure. It means the F32 fully-resident CPU/swap path is too heavy on this machine and should not be the regular full 30-block verification path.

结果：

Result:

```text
1 block final-only:
loaded 1307.79 MiB, 56 tensors
latent final: max_abs_diff=0.0124, mean_abs_diff=0.00351
action final: max_abs_diff=0.00888, mean_abs_diff=0.00239

30 blocks final-only:
loaded 19412.51 MiB, 839 tensors
no final dump within 600 seconds
```

下一步：

Next step:

- 转向更适合 8GB 目标的完整 30 block 策略：优先 BF16 resident 或 per-block streaming；如果仍然不够，再提高量化程度。
- Move to a full 30-block strategy suitable for the 8GB target: prioritize BF16-resident weights or per-block streaming; if still insufficient, increase the quantization level.

## Per-Block F32 Streaming Smoke

中文：

- 新增 `VLA_LINGBOT_SMOKE_STREAM=1` 路径。
- 这不是 BF16 resident 实现，而是 per-block streaming 的 F32 debug smoke：
  - GGUF 原始权重仍是 BF16。
  - 读取当前 block 时临时 BF16 -> F32。
  - 公共权重常驻 F32，当前 block 权重临时 F32，block 计算完成后释放。
- 这样可以先验证“分 block 执行”不会破坏 parity，同时避免 30 blocks 全量 F32 常驻。
- 这一步的意义是把完整 30 block 验证从“19.4 GiB F32 常驻 + swap”改为“约 683 MiB 公共 F32 + 约 624 MiB 当前 block F32”。

English:

- Added the `VLA_LINGBOT_SMOKE_STREAM=1` path.
- This is not the BF16-resident implementation yet. It is a per-block streaming F32 debug smoke:
  - The GGUF source weights remain BF16.
  - The current block is temporarily converted BF16 -> F32 when loaded.
  - Common weights stay resident as F32, while each block's F32 weights are loaded temporarily and released after the block.
- This first verifies that per-block execution preserves parity, while avoiding fully resident F32 weights for all 30 blocks.
- The practical effect is moving full 30-block verification from “19.4 GiB resident F32 + swap” to “about 683 MiB common F32 + about 624 MiB current-block F32”.

实现注意：

Implementation notes:

- streaming 路径分为三段：embedding graph -> per-block graph loop -> output-head graph。
- The streaming path is split into three stages: embedding graph -> per-block graph loop -> output-head graph.
- 中间曾出现过错误：在 `ggml_gallocr_free()` 后读取输出 tensor，导致读到失效 buffer。修复为先 `ggml_backend_tensor_get()`，再释放 graph allocator。
- A bug was found and fixed: reading output tensors after `ggml_gallocr_free()` caused invalid-buffer reads. The fix is to call `ggml_backend_tensor_get()` before freeing the graph allocator.
- 1 block streaming parity 已与原 full-graph smoke 对齐。
- 1-block streaming parity now matches the previous full-graph smoke.

验证结果：

Verification result:

```text
1 block streaming:
common weights: 683.49 MiB F32, 29 tensors
block weights:  624.30 MiB F32, 27 tensors per block
latent final:   max_abs_diff=0.0124, mean_abs_diff=0.00351
action final:   max_abs_diff=0.00888, mean_abs_diff=0.00239

5 blocks streaming:
latent final:   max_abs_diff=0.0457, mean_abs_diff=0.00764
action final:   max_abs_diff=0.00800, mean_abs_diff=0.00256

10 blocks streaming:
latent final:   max_abs_diff=0.0296, mean_abs_diff=0.00737
action final:   max_abs_diff=0.0167, mean_abs_diff=0.00567

30 blocks streaming:
completed all 30 transformer blocks
latent checksum: python=-1.55760, cpp=-2.70260
latent final:   max_abs_diff=0.246, mean_abs_diff=0.0586
action checksum: python=2.10819, cpp=2.09344
action final:   max_abs_diff=0.0346, mean_abs_diff=0.0159
```

判断：

Assessment:

- 完整 30 block transformer 主干已经通过 per-block F32 streaming 跑通。
- The full 30-block transformer trunk now runs through the per-block F32 streaming path.
- 30 层累积后 latent diff 明显大于前 20 blocks，但没有发散；action diff 仍较小。
- After 30 layers, latent diff is noticeably larger than at 20 blocks but does not diverge; action diff remains small.
- 这证明当前主要瓶颈已经从“结构/排列能不能跑通”转为“数值误差控制、BF16/量化执行策略、真实输入接入”。
- This shows the main bottleneck has shifted from “can the structure/layout run” to “numerical-error control, BF16/quantized execution strategy, and real-input integration”.

下一步：

Next step:

- 继续做 BF16 resident 或量化 block buffer，把当前的 “BF16 GGUF -> F32 临时 block” 改造成更接近最终部署的执行路径。
- Next, implement BF16-resident or quantized block buffers, replacing the current temporary “BF16 GGUF -> F32 block” path with an execution path closer to final deployment.

## 量化设计决策 / Quantization Design Decision

中文：

- 暂不急着为 LingBot-VA 自行抽象一套新的 dtype 系统。
- 先参考 vla.cpp 现有模型实现：
  - `smolvla.cpp` 使用 `pending_f32` / `pending_bf16`，小权重和数值敏感权重走 F32，大矩阵可走 BF16。
  - `bitvla.cpp` 使用 `mk_f32` / `mk_mm` 区分普通 F32 权重和 matmul 权重，并通过 `read_convert()` 根据目标 tensor dtype 装载。
  - `pi0.cpp` 也使用 `read_convert(t->type, ...)`，让 GGUF/tensor type 驱动权重装载，而不是每个量化档手写一套 C++。
- LingBot-VA 后续应采用同类风格：把权重分成 F32 side weights 和 matmul weights；matmul weights 的 dtype 尽量由 GGUF/tensor type 决定。
- 目标是避免 “BF16/Q8/Q6 每换一种 GGUF 就改一次 C++”。

English:

- Do not rush into inventing a new dtype abstraction for LingBot-VA.
- First follow the existing vla.cpp patterns:
  - `smolvla.cpp` uses `pending_f32` / `pending_bf16`, keeping small or numerically sensitive weights in F32 while allowing large matrices to use BF16.
  - `bitvla.cpp` uses `mk_f32` / `mk_mm` to separate ordinary F32 weights from matmul weights, and loads data via `read_convert()` according to the target tensor dtype.
  - `pi0.cpp` also uses `read_convert(t->type, ...)`, letting GGUF/tensor type drive loading instead of hand-writing a C++ path for every quantization variant.
- LingBot-VA should follow this style: split weights into F32 side weights and matmul weights; matmul dtype should be driven by GGUF/tensor type where possible.
- The goal is to avoid modifying C++ every time we test a BF16/Q8/Q6 GGUF.

## Adaptive Block-Window Streaming

时间 / Time: `2026-06-15`

中文：

- 在 per-block streaming 基础上新增 adaptive block-window streaming。
- 目的：不要固定每次只加载 1 个 block，而是根据窗口大小一次加载多个连续 block，减少反复读盘/分配/未来 CPU-GPU 传输压力。
- 新增控制：

```bash
VLA_LINGBOT_BLOCK_WINDOW=2      # 手动指定每窗 2 blocks
VLA_LINGBOT_BLOCK_WINDOW=auto   # 自动估算窗口
VLA_LINGBOT_BLOCK_BUDGET_MB=4096
```

- 当前默认仍是 `window=1`，保持旧 streaming smoke 行为不变。
- `auto` 当前基于 F32 debug 权重估算每个 block 字节数，并预留 768 MiB 激活/graph buffer。
- 当后续 matmul weights 变成 BF16/Q8/Q6 后，同样的 window 机制可以自然加载更多 blocks；如果估算窗口 `>= 30`，就等价于一次加载完整 transformer block 主干。

English:

- Added adaptive block-window streaming on top of per-block streaming.
- Purpose: avoid always loading only one block at a time; instead load multiple consecutive blocks per window to reduce repeated disk reads, allocation overhead, and future CPU-GPU transfer pressure.
- New controls:

```bash
VLA_LINGBOT_BLOCK_WINDOW=2      # manually use 2 blocks per window
VLA_LINGBOT_BLOCK_WINDOW=auto   # estimate window size automatically
VLA_LINGBOT_BLOCK_BUDGET_MB=4096
```

- The default remains `window=1`, preserving the previous streaming smoke behavior.
- `auto` currently estimates block size using the F32 debug-weight footprint and reserves 768 MiB for activations / graph buffers.
- When matmul weights later become BF16/Q8/Q6, the same window mechanism can naturally load more blocks. If the estimated window is `>= 30`, it becomes equivalent to loading the whole transformer trunk at once.

验证结果：

Verification result:

```text
Default window=1, 1 block:
latent final: max_abs_diff=0.0124, mean_abs_diff=0.00351
action final: max_abs_diff=0.00888, mean_abs_diff=0.00239

Manual window=2, 5 blocks:
loads stream blocks 0-1 and 2-3 as 1248.60 MiB windows
latent final: max_abs_diff=0.0457, mean_abs_diff=0.00764
action final: max_abs_diff=0.00800, mean_abs_diff=0.00256

Auto window, 10 blocks, budget=4096 MiB:
auto block window=5
loads stream blocks 0-4 and 5-9 as 3121.50 MiB windows
latent final: max_abs_diff=0.0296, mean_abs_diff=0.00737
action final: max_abs_diff=0.0167, mean_abs_diff=0.00567
```

判断：

Assessment:

- block-window streaming 已经跑通，结果与 one-block streaming / Python parity 对齐。
- Block-window streaming is working and remains aligned with one-block streaming / Python parity.
- 这为后续 BF16/量化 GGUF 提供统一执行器：量化程度越高，自动窗口越大；压缩足够时自然退化为 full-window/full-resident block trunk。
- This provides a unified executor for future BF16/quantized GGUF files: higher compression leads to larger automatic windows; sufficient compression naturally degenerates into a full-window/full-resident block trunk.

## Transformer Executor 骨架整理 / Transformer Executor Skeleton Refactor

时间 / Time: `2026-06-15 13:36:26 CST`

中文：

- 将已验证的 streaming smoke 路径整理为更接近正式推理的 transformer executor 骨架。
- 当前仍然使用 synthetic inputs 和 F32 debug weights，但执行阶段已经拆成未来 `predict()` 可复用的结构：

```text
LingBotTransformerExecutor
  -> exec_embedding_stage()
  -> exec_block_window()
  -> exec_output_stage()
```

- 新增/整理的核心结构：

```text
LingBotExecState
  seq
  x
  text
  t_hidden
  timestep_proj

LingBotTransformerExecutor
  model metadata
  common weights
  dump_dir
```

- `VLA_LINGBOT_SMOKE_STREAM=1` 仍然保留为验证入口，但内部不再只是散落的 smoke helper，而是沿着 executor stage 执行。
- 这个重构不改变数值路径，目标是为后续接真实输入、`predict()`、BF16/量化 block-window 权重做准备。

English:

- Refactored the verified streaming smoke path into a transformer executor skeleton closer to real inference.
- It still uses synthetic inputs and F32 debug weights for now, but the execution stages are now split into a structure that future `predict()` can reuse:

```text
LingBotTransformerExecutor
  -> exec_embedding_stage()
  -> exec_block_window()
  -> exec_output_stage()
```

- Core structures added / organized:

```text
LingBotExecState
  seq
  x
  text
  t_hidden
  timestep_proj

LingBotTransformerExecutor
  model metadata
  common weights
  dump_dir
```

- `VLA_LINGBOT_SMOKE_STREAM=1` remains the verification entrypoint, but internally it now follows executor stages instead of loose smoke helpers.
- This refactor does not change the numerical path. Its purpose is to prepare for real inputs, `predict()`, and BF16/quantized block-window weights.

验证结果：

Verification result:

```text
Build: cmake --build build -j2

10 blocks, VLA_LINGBOT_BLOCK_WINDOW=auto, budget=4096 MiB:
auto block window=5
latent final: max_abs_diff=0.0296, mean_abs_diff=0.00737
action final: max_abs_diff=0.0167, mean_abs_diff=0.00567
```

## 后续完整 C++ 化范围 / Remaining Full C++ Port Scope

时间 / Time: `2026-06-15 13:17:21 CST`

中文：

以下模块后续都需要完成，不能只停留在 transformer smoke：

1. UMT5 text encoder
   - 作用：把语言指令编码成 text embedding，供 LingBot transformer 的 cross-attention 使用。
   - 当前状态：未实现；smoke 目前使用 synthetic text input。

2. VAE encoder / decoder
   - 作用：把真实视觉 observation 编码到 latent，或把 latent 解码回视觉空间。
   - 当前状态：未实现；smoke 目前使用 synthetic latent/patch input。

3. 真实输入预处理
   - 内容：LIBERO observation 图像处理、resize/crop/normalize、state/action normalization、time/noise step、mesh/grid ids。
   - 当前状态：未实现完整真实链路。

4. 真实 Wan-style RoPE / position ids
   - 作用：替换 smoke 中的 `rope_cos=1, rope_sin=0` 简化输入，接入真实 3D RoPE 和 action grid。
   - 当前状态：已有 C++ Wan RoPE primitive 路径雏形，但真实位置输入尚未完整接入。

5. diffusion / flow matching sampling loop
   - 作用：从单次 transformer forward 扩展为完整 denoising / flow update 推理循环。
   - 当前状态：未实现；当前只验证 transformer forward。

6. ZMQ/protobuf `predict()` 接口适配
   - 作用：让 `vla-server` 收到真实请求后调用 LingBot-VA 完整推理并返回 action。
   - 当前状态：`predict()` 仍未实现完整 LingBot-VA 推理。

7. 权重量化与高效执行策略
   - 内容：adaptive block-window execution、BF16/quantized matmul weights、F32 side weights、8GB 目标部署。
   - 当前状态：已验证 per-block F32 streaming；BF16/量化 resident/window 策略尚未完成。

English:

The following modules must all be completed later; the project should not stop at transformer smoke tests:

1. UMT5 text encoder
   - Purpose: encode language instructions into text embeddings for LingBot transformer cross-attention.
   - Current status: not implemented; smoke currently uses synthetic text input.

2. VAE encoder / decoder
   - Purpose: encode real visual observations into latents, or decode latents back to visual space.
   - Current status: not implemented; smoke currently uses synthetic latent/patch input.

3. Real input preprocessing
   - Scope: LIBERO observation image handling, resize/crop/normalize, state/action normalization, time/noise step, mesh/grid ids.
   - Current status: the full real-input path is not implemented yet.

4. Real Wan-style RoPE / position ids
   - Purpose: replace the smoke simplification `rope_cos=1, rope_sin=0` with real 3D RoPE and action-grid positions.
   - Current status: a C++ Wan RoPE primitive path exists, but real position inputs are not fully wired.

5. Diffusion / flow matching sampling loop
   - Purpose: expand from a single transformer forward into the complete denoising / flow-update inference loop.
   - Current status: not implemented; current work only verifies transformer forward.

6. ZMQ/protobuf `predict()` integration
   - Purpose: allow `vla-server` to receive real requests, run full LingBot-VA inference, and return actions.
   - Current status: `predict()` does not yet implement full LingBot-VA inference.

7. Weight quantization and efficient execution
   - Scope: adaptive block-window execution, BF16/quantized matmul weights, F32 side weights, and 8GB-target deployment.
   - Current status: per-block F32 streaming is verified; BF16/quantized resident/window execution remains to be done.

## 2026-06-15 16:22:28 CST / June 15, 2026 16:22:28 CST

### 第二阶段开始：真实 RoPE / grid id 路径 / Started Stage 2: Real RoPE / Grid-ID Path

中文：

- 已将当前执行顺序写入本文档顶部，并把第 1 步 executor 骨架标记为已完成。
- 开始第 2 步：将 smoke 中的 identity RoPE（`cos=1, sin=0`）替换为更贴近 LingBot-VA Python 原版的 Wan-style RoPE。
- 对照 Python 原版确认关键逻辑：
  - `wan_va/utils/utils.py::get_mesh_id()` 生成 `f/h/w/t` grid id。
  - `wan_va/modules/model.py::WanRotaryPosEmbed` 将 `head_dim=128` 分为 `f_dim=44, h_dim=42, w_dim=42`，再分别生成 RoPE 频率。
  - action grid 使用 `action=True`，其特点是 `f` 带 fractional offset，`h/w=-1`。
- 在 `src/models/lingbot_va.cpp` 中新增共享函数 `build_lingbot_rope()`。
- `LingBotExecState` 新增 `rope_cos` / `rope_sin`，streaming executor 的每条 latent/action 路径保存自己的 RoPE 输入。
- full compute smoke 和 streaming smoke 都已从 identity RoPE 切换到真实 LingBot/Wan-style synthetic RoPE。
- 在 `scripts/run_lingbot_va_oneblock_parity.py` 中同步新增 `lingbot_rope()`，保证 PyTorch parity 和 C++ smoke 使用同一套最小真实 grid 语义。

说明：

- 当前 smoke 的 `seq=2` 不是完整真实视频/action 形状，所以这里使用“等价长度的最小 grid”：
  - latent：两个时间 token。
  - action：两个 `action=True` token，覆盖 action grid 的 fractional `f` 和 `h/w=-1` 语义。
- 这一步验证的是 RoPE 数学和张量排列已经进入 C++ executor，不代表真实 `predict()` 的动态 grid 已经完成。

验证：

```text
Build:
cmake --build build -j2

1 block streaming smoke:
latent checksum=11.9305, max_abs=1.63704
action checksum=2.10607, max_abs=0.561802

1 block parity:
latent max_abs_diff=0.042242289, mean_abs_diff=0.012459938
action max_abs_diff=0.0088200346, mean_abs_diff=0.0027387885

10 blocks streaming smoke, auto window:
auto block window=5
latent checksum=-1.66691, max_abs=1.41224
action checksum=0.467869, max_abs=0.727219

10 blocks parity:
latent max_abs_diff=0.048735037, mean_abs_diff=0.01251052
action max_abs_diff=0.019792199, mean_abs_diff=0.0060399272
```

后续：

- 将真实 `predict()` 的输入形状、frame offset、latent/action sampling timestep 接入 grid id 生成。
- 后续接 VAE/action loop 时，需要把 `get_mesh_id()` 的完整参数语义补齐，而不是继续使用 smoke 的最小 synthetic grid。

English:

- Added the current execution order to the top of this document and marked Stage 1, the executor skeleton, as complete.
- Started Stage 2 by replacing identity RoPE (`cos=1, sin=0`) in smoke tests with a Wan-style RoPE path closer to the original LingBot-VA Python implementation.
- Verified the relevant Python reference:
  - `wan_va/utils/utils.py::get_mesh_id()` generates `f/h/w/t` grid ids.
  - `wan_va/modules/model.py::WanRotaryPosEmbed` splits `head_dim=128` into `f_dim=44, h_dim=42, w_dim=42` and builds axis-specific RoPE frequencies.
  - The action grid uses `action=True`, with fractional `f` offsets and sentinel `h/w=-1`.
- Added shared `build_lingbot_rope()` in `src/models/lingbot_va.cpp`.
- Added `rope_cos` / `rope_sin` to `LingBotExecState`, so each latent/action path in the streaming executor carries its own RoPE inputs.
- Switched both the full compute smoke and streaming smoke from identity RoPE to the real LingBot/Wan-style synthetic RoPE.
- Added matching `lingbot_rope()` in `scripts/run_lingbot_va_oneblock_parity.py`, keeping PyTorch parity aligned with the C++ smoke path.

Notes:

- The smoke path still uses `seq=2`, not the full real video/action tensor shape. Therefore this stage uses a minimal grid with equivalent length:
  - latent: two temporal tokens.
  - action: two `action=True` tokens, covering fractional `f` and `h/w=-1` behavior.
- This verifies that RoPE math and tensor layout are now part of the C++ executor. It does not mean dynamic grids from real `predict()` are fully wired yet.

Verification:

```text
Build:
cmake --build build -j2

1-block streaming smoke:
latent checksum=11.9305, max_abs=1.63704
action checksum=2.10607, max_abs=0.561802

1-block parity:
latent max_abs_diff=0.042242289, mean_abs_diff=0.012459938
action max_abs_diff=0.0088200346, mean_abs_diff=0.0027387885

10-block streaming smoke, auto window:
auto block window=5
latent checksum=-1.66691, max_abs=1.41224
action checksum=0.467869, max_abs=0.727219

10-block parity:
latent max_abs_diff=0.048735037, mean_abs_diff=0.01251052
action max_abs_diff=0.019792199, mean_abs_diff=0.0060399272
```

Next:

- Wire real `predict()` input shapes, frame offsets, and latent/action sampling timesteps into grid-id generation.
- When VAE/action-loop integration starts, replace the smoke-only minimal synthetic grid with the complete `get_mesh_id()` parameter semantics.

## 2026-06-15 16:40:30 CST / June 15, 2026 16:40:30 CST

### 真实 grid spec 抽象 / Real Grid-Spec Abstraction

中文：

- 从 LingBot-VA Python 原版 `_prepare_latent_input()` 中确认真实 grid 规则：
  - latent: `get_mesh_id(F / patch_f, H / patch_h, W / patch_w, t=0, f_w=1, f_shift=frame_st_id, action=False)`。
  - action: `get_mesh_id(F, H, W, t=1, f_w=1, f_shift=frame_st_id, action=True)`。
- 在 `src/models/lingbot_va.cpp` 中新增 `LingBotGridSpec`：

```text
f, h, w, t, f_w, f_shift, action
```

- `build_lingbot_rope()` 已从 `seq + action_mode` 改为接收 `LingBotGridSpec`。
- smoke 仍使用最小 synthetic shape，但现在通过明确的 grid spec 表达：
  - latent smoke: `f=seq, h=1, w=1, t=0`
  - action smoke: `f=1, h=seq, w=1, t=1, action=True`
- `scripts/run_lingbot_va_oneblock_parity.py` 同步新增 Python 侧 `GridSpec` 和同样的 RoPE 生成规则。
- 新增 `scripts/check_lingbot_va_grid_rope.py`，直接加载 LingBot 原版 `get_mesh_id()` 源文件，避免导入整个 Python 包导致 diffusers 版本问题。

验证：

```text
python scripts/check_lingbot_va_grid_rope.py

smoke_latent:  grid_max_diff=0, rope_max_diff=2.98e-08
smoke_action:  grid_max_diff=0, rope_max_diff=2.98e-08
latent_realish: grid_max_diff=0, rope_max_diff=2.31e-07
action_realish: grid_max_diff=0, rope_max_diff=2.94e-07

cmake --build build -j2

10 blocks streaming smoke, auto window=5:
latent max_abs_diff=0.048735037, mean_abs_diff=0.01251052
action max_abs_diff=0.019796968, mean_abs_diff=0.0060399086
```

结论：

- C++ grid/RoPE 规则已经可以表达 LingBot 原版 `get_mesh_id()` 的核心参数语义。
- 这一步仍属于输入语义准备；真实 `predict()` 还需要把 `frame_st_id`、latent/action 实际形状和 sampling timestep 传入这些 spec。

English:

- Confirmed the real grid rules from the original LingBot-VA Python `_prepare_latent_input()`:
  - latent: `get_mesh_id(F / patch_f, H / patch_h, W / patch_w, t=0, f_w=1, f_shift=frame_st_id, action=False)`.
  - action: `get_mesh_id(F, H, W, t=1, f_w=1, f_shift=frame_st_id, action=True)`.
- Added `LingBotGridSpec` in `src/models/lingbot_va.cpp`:

```text
f, h, w, t, f_w, f_shift, action
```

- Changed `build_lingbot_rope()` from `seq + action_mode` to `LingBotGridSpec`.
- Smoke tests still use minimal synthetic shapes, but now express them explicitly as grid specs:
  - latent smoke: `f=seq, h=1, w=1, t=0`
  - action smoke: `f=1, h=seq, w=1, t=1, action=True`
- Added matching Python-side `GridSpec` and RoPE generation in `scripts/run_lingbot_va_oneblock_parity.py`.
- Added `scripts/check_lingbot_va_grid_rope.py`, which loads the original LingBot `get_mesh_id()` source file directly to avoid importing the full Python package and its diffusers-version-sensitive dependencies.

Verification:

```text
python scripts/check_lingbot_va_grid_rope.py

smoke_latent:  grid_max_diff=0, rope_max_diff=2.98e-08
smoke_action:  grid_max_diff=0, rope_max_diff=2.98e-08
latent_realish: grid_max_diff=0, rope_max_diff=2.31e-07
action_realish: grid_max_diff=0, rope_max_diff=2.94e-07

cmake --build build -j2

10-block streaming smoke, auto window=5:
latent max_abs_diff=0.048735037, mean_abs_diff=0.01251052
action max_abs_diff=0.019796968, mean_abs_diff=0.0060399086
```

Conclusion:

- The C++ grid/RoPE rules can now represent the core parameter semantics of the original LingBot `get_mesh_id()`.
- This is still input-semantics preparation. Real `predict()` must still pass `frame_st_id`, actual latent/action shapes, and sampling timesteps into these specs.

## 2026-06-15 19:10:22 CST / June 15, 2026 19:10:22 CST

### ONNX 算子审计 / ONNX Operator Audit

中文：

- 新增 `scripts/export_lingbot_va_onnx_audit.py`。
- 生成 LingBot-VA 代表性 ONNX skeleton，而不是导出完整 22GB 模型：
  - `wan_rope.onnx`
  - `wan_attention.onnx`
  - `wan_block.onnx`
  - `flow_step.onnx`
  - `causal_conv3d_skeleton.onnx`
- 输出目录：

```text
artifacts/lingbot_va_onnx/
```

- 生成双语审计报告：

```text
artifacts/lingbot_va_onnx/LINGBOT_VA_ONNX_AUDIT.md
```

- 当前分类结论：
  - WanTransformer 的 linear / norm / FFN / self-attn / cross-attn / RoPE / flow step 第一版都可以用 ggml 拼接。
  - UMT5 encoder 理论上也可以用 ggml 拼接，主要风险是权重规模、量化和加载策略。
  - VAE conv/resnet/up/downsample 理论上可用 ggml 拼接，但工程量较大。
  - 潜在新算子候选主要是 `Flex/block-sparse attention mask` 和 `WanCausalConv3d streaming cache` 的高效实现。
- Netron 已启动查看 `wan_block.onnx`：

```text
http://127.0.0.1:8088
```

注意：

- 为了使用 ONNX/Netron，LIBERO uv venv 中安装了 `onnx` 和 `netron`，同时 `protobuf` 被升级到 `7.35.1`。

English:

- Added `scripts/export_lingbot_va_onnx_audit.py`.
- Generated representative LingBot-VA ONNX skeletons instead of exporting the full 22GB model:
  - `wan_rope.onnx`
  - `wan_attention.onnx`
  - `wan_block.onnx`
  - `flow_step.onnx`
  - `causal_conv3d_skeleton.onnx`
- Output directory:

```text
artifacts/lingbot_va_onnx/
```

- Generated a bilingual audit report:

```text
artifacts/lingbot_va_onnx/LINGBOT_VA_ONNX_AUDIT.md
```

- Current classification:
  - WanTransformer linear / norm / FFN / self-attn / cross-attn / RoPE / flow step can be implemented with ggml composition for the first version.
  - UMT5 encoder should also be expressible with ggml; the main risks are model size, quantization, and loading strategy.
  - VAE conv/resnet/up/downsample should be expressible with ggml, but it is a larger implementation effort.
  - The main custom-kernel candidates are efficient `Flex/block-sparse attention mask` and `WanCausalConv3d streaming cache` paths.
- Netron is running for `wan_block.onnx`:

```text
http://127.0.0.1:8088
```

Note:

- To enable ONNX/Netron, `onnx` and `netron` were installed into the LIBERO uv venv, and `protobuf` was upgraded to `7.35.1`.

## 2026-06-15 20:08:20 CST / June 15, 2026 20:08:20 CST

### FlowMatchScheduler C++ helper / FlowMatchScheduler C++ 辅助实现

中文：

- 在 `src/models/lingbot_va.cpp` 中新增 `LingBotFlowScheduler`。
- 对齐 LingBot Python 原版 `wan_va/utils/scheduler.py::FlowMatchScheduler` 的推理路径：
  - `set_timesteps()`
  - shifted sigma schedule
  - `timestep = sigma * num_train_timesteps`
  - `step(): sample + model_output * (sigma_next - sigma)`
- 新增 `VLA_LINGBOT_FLOW_SMOKE=1` smoke 开关。
- 新增 `scripts/check_lingbot_va_flow_scheduler.py`，直接加载 Python 原版 scheduler 文件并打印 reference 数值。

验证：

```text
cmake --build build -j2

Python reference:
video  steps=20 shift=5    sigma0=1 sigma_last=0.0147929005 t0=1000 t_last=14.7929001 checksum=0.264304847 max_abs=0.0329403169
action steps=50 shift=0.05 sigma0=0.999999762 sigma_last=0.000150127613 t0=999.999756 t_last=0.15012762 checksum=0.264304847 max_abs=0.0329403169

C++ smoke:
video  steps=20 shift=5    sigma0=1 sigma_last=0.0147928994 t0=1000 t_last=14.7928994 checksum=0.264304846 max_abs=0.0329403169
action steps=50 shift=0.05 sigma0=1 sigma_last=0.000150127608 t0=1000 t_last=0.150127608 checksum=0.264304839 max_abs=0.0329403207
```

说明：

- 这一步只完成 scheduler 数学和 smoke 验证，还没有把 transformer forward 放进 video/action flow loop。
- `timeout` 结束 `vla-server` 是因为 smoke 后 server 正常进入 ready 状态；不是 smoke 失败。

English:

- Added `LingBotFlowScheduler` in `src/models/lingbot_va.cpp`.
- Matched the inference path of the original LingBot Python `wan_va/utils/scheduler.py::FlowMatchScheduler`:
  - `set_timesteps()`
  - shifted sigma schedule
  - `timestep = sigma * num_train_timesteps`
  - `step(): sample + model_output * (sigma_next - sigma)`
- Added the `VLA_LINGBOT_FLOW_SMOKE=1` smoke switch.
- Added `scripts/check_lingbot_va_flow_scheduler.py`, which loads the original Python scheduler file directly and prints reference values.

Verification:

```text
cmake --build build -j2

Python reference:
video  steps=20 shift=5    sigma0=1 sigma_last=0.0147929005 t0=1000 t_last=14.7929001 checksum=0.264304847 max_abs=0.0329403169
action steps=50 shift=0.05 sigma0=0.999999762 sigma_last=0.000150127613 t0=999.999756 t_last=0.15012762 checksum=0.264304847 max_abs=0.0329403169

C++ smoke:
video  steps=20 shift=5    sigma0=1 sigma_last=0.0147928994 t0=1000 t_last=14.7928994 checksum=0.264304846 max_abs=0.0329403169
action steps=50 shift=0.05 sigma0=1 sigma_last=0.000150127608 t0=1000 t_last=0.150127608 checksum=0.264304839 max_abs=0.0329403207
```

Notes:

- This step only completes scheduler math and smoke verification. The transformer forward is not yet wired into the video/action flow loops.
- The `timeout` termination happened because `vla-server` reached the normal ready state after the smoke; it was not a smoke failure.

## 2026-06-15 20:20:53 CST / June 15, 2026 20:20:53 CST

### Synthetic transformer flow-loop smoke / 合成输入 transformer flow-loop smoke

中文：

- 新增 `VLA_LINGBOT_FLOW_LOOP_SMOKE=1`。
- 将已完成的 WanTransformer streaming executor 与 `LingBotFlowScheduler` 串起来，形成第一版 video/action flow-loop skeleton。
- 当前 smoke 仍然使用 synthetic 输入：
  - synthetic latent/action sample
  - synthetic text embedding
  - synthetic time features with timestep injection
  - 默认只跑少量 steps 和少量 blocks，避免调试时重复加载/计算完整 30 block x 70 steps。
- 新增可调环境变量：

```text
VLA_LINGBOT_FLOW_LOOP_BLOCKS
VLA_LINGBOT_FLOW_LOOP_VIDEO_STEPS
VLA_LINGBOT_FLOW_LOOP_ACTION_STEPS
```

验证：

```text
cmake --build build -j2

1 video step + 1 action step + 1 block:
video  checksum=-11.3782059 max_abs=1.53592467
action checksum=-1.6575622  max_abs=0.548838615

default 2 video steps + 2 action steps + 1 block:
video  checksum=-10.4148079 max_abs=1.53314626
action checksum=-1.65648539 max_abs=0.548632026
```

说明：

- 这一步把控制流从“单次 transformer forward”推进到了“scheduler loop 调用 transformer forward”。
- 这仍不是正式 `predict()`：
  - 没有真实 UMT5 text encoder。
  - 没有真实 VAE image-to-latent。
  - 没有真实 action preprocess。
  - 没有 KV cache。
  - 还没有接 ZMQ/protobuf 请求输出真实 action。
- 当前实现为了复用现有 `exec_block_window()`，flow-loop smoke 会额外跑一个 dummy 分支；正式 executor 后续需要拆成单分支，避免多余计算。

English:

- Added `VLA_LINGBOT_FLOW_LOOP_SMOKE=1`.
- Connected the existing WanTransformer streaming executor with `LingBotFlowScheduler`, forming the first video/action flow-loop skeleton.
- The current smoke still uses synthetic inputs:
  - synthetic latent/action sample
  - synthetic text embedding
  - synthetic time features with timestep injection
  - by default only a few steps and blocks are executed, avoiding repeated full 30-block x 70-step debug runs.
- Added tunable environment variables:

```text
VLA_LINGBOT_FLOW_LOOP_BLOCKS
VLA_LINGBOT_FLOW_LOOP_VIDEO_STEPS
VLA_LINGBOT_FLOW_LOOP_ACTION_STEPS
```

Verification:

```text
cmake --build build -j2

1 video step + 1 action step + 1 block:
video  checksum=-11.3782059 max_abs=1.53592467
action checksum=-1.6575622  max_abs=0.548838615

default 2 video steps + 2 action steps + 1 block:
video  checksum=-10.4148079 max_abs=1.53314626
action checksum=-1.65648539 max_abs=0.548632026
```

Notes:

- This advances the control flow from a single transformer forward to a scheduler loop that calls transformer forward.
- This is still not real `predict()`:
  - no real UMT5 text encoder
  - no real VAE image-to-latent
  - no real action preprocessing
  - no KV cache
  - no ZMQ/protobuf real action output yet
- The smoke currently reuses `exec_block_window()` and therefore computes one extra dummy branch. The formal executor should later split this into a single-branch path to avoid redundant work.

## 2026-06-15 20:28:56 CST - Single-Branch Transformer Forward Skeleton / 单分支 Transformer Forward 骨架

中文：

- 将 flow-loop smoke 中的 dummy 双分支执行方式替换为正式的单分支 transformer forward skeleton。
- 新增 `exec_forward_one_streaming()`：
  - `exec_embedding_stage()`
  - configurable N x `exec_block_one()`
  - `exec_output_one()`
- `run_flow_loop_smoke()` 现在直接调用单分支 forward，不再为了复用双分支 `exec_block_window()` 而额外计算 dummy latent/action 分支。
- 这一步符合后续真实预测路径的调用形态：video/world branch 和 action branch 可以分别调用同一个单分支 executor。
- 推进策略调整：对于简单且容易定位的工程重构，可以直接做大步修改，然后用 build + smoke + parity 回归兜底。

验证：

```text
cmake --build build -j2

VLA_LINGBOT_FLOW_LOOP_SMOKE=1 VLA_LINGBOT_FLOW_LOOP_BLOCKS=1
video  checksum=-10.4148079 max_abs=1.53314626
action checksum=-1.65648539 max_abs=0.548632026

VLA_LINGBOT_COMPUTE_SMOKE=1 VLA_LINGBOT_SMOKE_STREAM=1 VLA_LINGBOT_SMOKE_BLOCKS=1
streaming latent checksum=11.9305 max_abs=1.63704
streaming action checksum=2.10607 max_abs=0.561802

Python parity, 1 block, final only:
latent max_abs_diff=0.042242289 mean_abs_diff=0.012459938
action max_abs_diff=0.0088189691 mean_abs_diff=0.0027387722
```

## 2026-06-16 17:59 CST - LIBERO LingBot 原版式 chunk rollout + cache update smoke

中文：

- 继续补齐 LIBERO client 到 `lingbot-world-server` 的真实输入链路。
- 更新 `eval/client/lingbot_world_client.py`：
  - 新增公开方法 `predict_chunk(obs)`，用于一次性获取 LingBot 的完整 action chunk。
  - `update_cache(obs_or_sequence, action_chunk, imagine=False)` 继续负责向 server 发送 `compute_kv_cache=true` 请求。
  - 新增 `max_cache_frames` 限制，当前默认 2 帧。
- 更新 `eval/client/run_sim_client_direct.py`：
  - 新增 `--max-steps`，用于短程 smoke，不再每次必须跑完整 episode。
  - 新增 `--lingbot-max-cache-frames`，把 LingBot cache update 的多帧数量暴露成命令行参数。
  - `--arch lingbot_va` 时改成更接近原版 LingBot LIBERO rollout 的流程：
    1. `predict_chunk(obs)` 获取 `[16,7]` action chunk；
    2. 按 `action_per_frame=4` reshape；
    3. 第一段 chunk 跳过 bootstrap frame 0；
    4. 执行低层 action；
    5. 收集 key frames；
    6. 调用 `update_cache(key_frames, chunk, imagine=False)` 写入 runtime KV history。
- 当前保留一个明确的工程边界：
  - C++ VAE 多帧路径在 3 帧 cache update 时仍会触发 down block1 AvgDown3D shortcut 失败。
  - 因此 client 暂时默认只发送最近 2 帧给 `compute_kv_cache`。
  - 这不是最终原版复现方案，后续需要继续修 VAE 3/4-frame streaming path。

验证：

```text
eval/sim/libero/libero_uv/.venv/bin/python -m py_compile \
  eval/client/run_sim_client_direct.py \
  eval/client/lingbot_world_client.py

env \
  VLA_LINGBOT_RESIDENT_BLOCK_CACHE=1 \
  VLA_LINGBOT_RESIDENT_BLOCK_DTYPE=q8_0 \
  VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX=1 \
  VLA_LINGBOT_PREDICT_CUDA_SELF_ATTN=1 \
  VLA_LINGBOT_PREDICT_TEXT_ENCODER=1 \
  VLA_LINGBOT_PREDICT_TEXT_BLOCKS=1 \
  VLA_LINGBOT_PREDICT_BLOCKS=1 \
  VLA_LINGBOT_PREDICT_VIDEO_STEPS=1 \
  VLA_LINGBOT_PREDICT_ACTION_STEPS=1 \
  VLA_LINGBOT_TEXT_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf \
  VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf \
  ./build/lingbot-world-server --bind tcp://127.0.0.1:6044 \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf

eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --arch lingbot_va \
  --task libero_object \
  --task-id 0 \
  --n-episodes 1 \
  --max-steps 13 \
  --vla-addr tcp://127.0.0.1:6044 \
  --recv-timeout-ms 900000 \
  --output-dir /tmp/lingbot_rollout_smoke
```

结果：

```text
Episode finished after 13 steps.
Final reward: 0.00
Average inference time per step: 6398.59 ms
Skipped: 0/1
```

server 侧关键确认：

```text
predict: mode=0, then mode=1 writes pred cache
runtime KV clear_pred session=2 blocks=1 pred_before=128 pred_after=0
cache_update: input=[3,2,128,128], action_condition=[7,4,4]
cache_update: mode=2 writes history cache, used grows 128 -> 256
next predict: mode=0 reads history cache, k=384 used=256
next predict: mode=1 writes pred cache again
```

结论：

- `LIBERO env -> LingBotWorldClient -> lingbot-world-server -> VAE image bridge -> UMT5 cache -> WanTransformer runtime KV -> LIBERO action postprocess -> env.step` 的短程闭环已经打通。
- 短程 smoke 不是评估效果，只验证链路稳定性；这里 reward 为 0 是因为 `--max-steps 13` 主动截断。
- 下一步优先级：修复 C++ VAE 多帧 3/4-frame streaming path，然后把 `--lingbot-max-cache-frames` 从 2 提升到原版需要的长度。

English:

- Continued wiring the real LIBERO client path into `lingbot-world-server`.
- Updated `eval/client/lingbot_world_client.py`:
  - added public `predict_chunk(obs)` to fetch a full LingBot action chunk;
  - kept `update_cache(obs_or_sequence, action_chunk, imagine=False)` as the `compute_kv_cache=true` request path;
  - added `max_cache_frames`, currently defaulting to 2 frames.
- Updated `eval/client/run_sim_client_direct.py`:
  - added `--max-steps` for short smoke tests;
  - added `--lingbot-max-cache-frames` to expose the LingBot cache-update frame limit;
  - when `--arch lingbot_va`, the rollout now follows the original LingBot-style chunk flow:
    1. fetch `[16,7]` through `predict_chunk(obs)`;
    2. reshape with `action_per_frame=4`;
    3. skip bootstrap frame 0 for the first chunk;
    4. execute low-level actions;
    5. collect key frames;
    6. call `update_cache(key_frames, chunk, imagine=False)` to write runtime KV history.
- Current boundary:
  - the C++ VAE multi-frame path still fails for 3-frame cache updates in the down block1 AvgDown3D shortcut;
  - the client therefore temporarily sends only the most recent 2 frames to `compute_kv_cache`;
  - this is not the final original-reproduction behavior and must be fixed in the VAE 3/4-frame streaming path.

Verification:

```text
eval/sim/libero/libero_uv/.venv/bin/python -m py_compile \
  eval/client/run_sim_client_direct.py \
  eval/client/lingbot_world_client.py

eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --arch lingbot_va \
  --task libero_object \
  --task-id 0 \
  --n-episodes 1 \
  --max-steps 13 \
  --vla-addr tcp://127.0.0.1:6044 \
  --recv-timeout-ms 900000 \
  --output-dir /tmp/lingbot_rollout_smoke
```

Result:

```text
Episode finished after 13 steps.
Final reward: 0.00
Average inference time per step: 6398.59 ms
Skipped: 0/1
```

Important server-side confirmation:

```text
predict: mode=0, then mode=1 writes pred cache
runtime KV clear_pred session=2 blocks=1 pred_before=128 pred_after=0
cache_update: input=[3,2,128,128], action_condition=[7,4,4]
cache_update: mode=2 writes history cache, used grows 128 -> 256
next predict: mode=0 reads history cache, k=384 used=256
next predict: mode=1 writes pred cache again
```

Conclusion:

- The short path `LIBERO env -> LingBotWorldClient -> lingbot-world-server -> VAE image bridge -> UMT5 cache -> WanTransformer runtime KV -> LIBERO action postprocess -> env.step` now works end-to-end.
- The smoke result is not a task-success evaluation; reward is 0 because `--max-steps 13` intentionally truncates the episode.
- Next priority: fix the C++ VAE 3/4-frame streaming path, then raise `--lingbot-max-cache-frames` from 2 to the original required length.

## 2026-06-16 18:08 CST - 修复 C++ VAE 3-frame cache update / Fixed C++ VAE 3-frame cache update

中文：

- 修复了上一节记录的 C++ VAE 3-frame cache update 失败点。
- 根因：
  - `vae_encoder_down_path_execute()` 给 temporal downsample 传的是逐帧 chunks：`[1,1,1,...]`。
  - 对 3 帧输入，main temporal path 只输出 1 帧，但 shortcut `AvgDown3D(factor_t=2)` 输出 `ceil(3/2)=2` 帧。
  - 因此 block1 处出现 `AvgDown3D shortcut failed`。
- 修改：
  - 新增 `temporal_down_chunks(n)` 分组策略：

```text
n=1 -> [1]
n=2 -> [1,1]
n=3 -> [1,2]
n=4 -> [1,2,1]
n=5 -> [1,2,2]
```

  - block1/block2 的 temporal downsample 改用该分组，使 main path 的时间维输出与 shortcut 的 `ceil(T/2)` 对齐。
  - `LingBotWorldClient.max_cache_frames` 默认从 2 提到 3。
  - `run_sim_client_direct.py --lingbot-max-cache-frames` 默认从 2 提到 3。

验证：

```text
cmake --build build -j2

VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1 \
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

VAE full encoder smoke 结果：

```text
input=[8,8,5,12]
block0=[4,4,5,160]
block1=[2,2,3,320]
block2=[1,1,2,640]
out=[1,1,2,96]
checksum=-1477.40445 max=16.5658226
```

真实 LIBERO world-server smoke：

```text
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --arch lingbot_va \
  --task libero_object \
  --task-id 0 \
  --n-episodes 1 \
  --max-steps 13 \
  --lingbot-max-cache-frames 3 \
  --vla-addr tcp://127.0.0.1:6044 \
  --recv-timeout-ms 900000 \
  --output-dir /tmp/lingbot_rollout_smoke
```

结果：

```text
Episode finished after 13 steps.
Skipped: 0/1
```

server 侧关键确认：

```text
VAE image bridge ok views=2 input=[3,3,128,128] latent=[1,48,1,8,16]
runtime KV clear_pred session=1 blocks=1 pred_before=128 pred_after=0
cache_update mode=2 writes history cache
next predict reads history cache
```

结论：

- C++ VAE 3-frame cache update 已可用。
- 当前 default cache frame 数已提升到 3。
- 后续可继续验证 4-frame 原版 key-frame window；如果 4 帧稳定，再把默认值提升到 4。

English:

- Fixed the C++ VAE 3-frame cache-update failure recorded in the previous section.
- Root cause:
  - `vae_encoder_down_path_execute()` passed per-frame chunks `[1,1,1,...]` into temporal downsampling.
  - For a 3-frame input, the main temporal path produced only 1 frame, while the `AvgDown3D(factor_t=2)` shortcut produced `ceil(3/2)=2` frames.
  - This caused the block1 shortcut shape mismatch.
- Change:
  - Added `temporal_down_chunks(n)`:

```text
n=1 -> [1]
n=2 -> [1,1]
n=3 -> [1,2]
n=4 -> [1,2,1]
n=5 -> [1,2,2]
```

  - block1/block2 temporal downsampling now uses this grouping, keeping the main path aligned with shortcut `ceil(T/2)`.
  - Raised `LingBotWorldClient.max_cache_frames` default from 2 to 3.
  - Raised `run_sim_client_direct.py --lingbot-max-cache-frames` default from 2 to 3.

Verification:

```text
cmake --build build -j2

VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1 \
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

VAE full encoder smoke:

```text
input=[8,8,5,12]
block0=[4,4,5,160]
block1=[2,2,3,320]
block2=[1,1,2,640]
out=[1,1,2,96]
checksum=-1477.40445 max=16.5658226
```

Real LIBERO world-server smoke:

```text
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --arch lingbot_va \
  --task libero_object \
  --task-id 0 \
  --n-episodes 1 \
  --max-steps 13 \
  --lingbot-max-cache-frames 3 \
  --vla-addr tcp://127.0.0.1:6044 \
  --recv-timeout-ms 900000 \
  --output-dir /tmp/lingbot_rollout_smoke
```

Result:

```text
Episode finished after 13 steps.
Skipped: 0/1
```

Important server-side confirmation:

```text
VAE image bridge ok views=2 input=[3,3,128,128] latent=[1,48,1,8,16]
runtime KV clear_pred session=1 blocks=1 pred_before=128 pred_after=0
cache_update mode=2 writes history cache
next predict reads history cache
```

Conclusion:

- The C++ VAE 3-frame cache-update path now works.
- The default cache frame count is now 3.
- Next, validate the 4-frame original key-frame window; if stable, raise the default to 4.

## 2026-06-16 18:12 CST - 4-frame key-frame cache window smoke

中文：

- 在 3-frame 修复后，继续验证更接近原版 rollout 的 4-frame key-frame cache window。
- 运行方式：
  - 第一段 action chunk 会跳过 bootstrap frame 0，只能产生 3 个 key frames；
  - 第二段完整 action chunk 会产生 4 个 key frames；
  - 因此使用 `--max-steps 29 --lingbot-max-cache-frames 4` 触发第二段后的 4-frame cache update。
- 验证通过后，将默认值提升为：
  - `LingBotWorldClient.max_cache_frames = 4`
  - `run_sim_client_direct.py --lingbot-max-cache-frames = 4`

验证：

```text
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --arch lingbot_va \
  --task libero_object \
  --task-id 0 \
  --n-episodes 1 \
  --max-steps 29 \
  --lingbot-max-cache-frames 4 \
  --vla-addr tcp://127.0.0.1:6044 \
  --recv-timeout-ms 900000 \
  --output-dir /tmp/lingbot_rollout_smoke
```

结果：

```text
Episode finished after 29 steps.
Skipped: 0/1
Average inference time per step: 10496.46 ms
```

server 侧关键确认：

```text
VAE image bridge ok views=2 input=[3,4,128,128] latent=[1,48,1,8,16]
runtime KV clear_pred session=1 blocks=1 pred_before=128 pred_after=0
cache_update mode=2 writes history cache, used grows 384 -> 512
next predict reads history cache, k=640 used=512
```

结论：

- 4-frame key-frame cache window 已通过短程 smoke。
- 当前 LIBERO LingBot client 默认已经切到 4-frame cache update。
- 后续重点从“输入链路是否能跑通”转向：
  - 更完整 block 数 / 更完整 flow step 的 rollout；
  - 真实 mask 语义与原版 cache attention 的进一步对齐；
  - 量化与 resident window 策略。

English:

- After fixing the 3-frame path, validated the more original-like 4-frame key-frame cache window.
- Test shape:
  - the first action chunk skips bootstrap frame 0, so it only produces 3 key frames;
  - the second full action chunk produces 4 key frames;
  - therefore `--max-steps 29 --lingbot-max-cache-frames 4` triggers a 4-frame cache update after the second chunk.
- After validation, raised defaults to:
  - `LingBotWorldClient.max_cache_frames = 4`
  - `run_sim_client_direct.py --lingbot-max-cache-frames = 4`

Verification:

```text
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --arch lingbot_va \
  --task libero_object \
  --task-id 0 \
  --n-episodes 1 \
  --max-steps 29 \
  --lingbot-max-cache-frames 4 \
  --vla-addr tcp://127.0.0.1:6044 \
  --recv-timeout-ms 900000 \
  --output-dir /tmp/lingbot_rollout_smoke
```

Result:

```text
Episode finished after 29 steps.
Skipped: 0/1
Average inference time per step: 10496.46 ms
```

Important server-side confirmation:

```text
VAE image bridge ok views=2 input=[3,4,128,128] latent=[1,48,1,8,16]
runtime KV clear_pred session=1 blocks=1 pred_before=128 pred_after=0
cache_update mode=2 writes history cache, used grows 384 -> 512
next predict reads history cache, k=640 used=512
```

Conclusion:

- The 4-frame key-frame cache window passes the short smoke.
- The LIBERO LingBot client now defaults to 4-frame cache updates.
- The next focus should move from “can the input path run” to:
  - rollout with more blocks / more flow steps;
  - closer cache-attention mask semantics versus the Python original;
  - quantization and resident-window strategy.

## 2026-06-16 18:28 CST - 验证策略切换：整体优先 parity / Switch to end-to-end-first parity

中文：

- 根据当前推进节奏，验证策略从“小 smoke 逐点推进”切换为“整体优先”：
  1. 先整体启动 C++ LingBot path；
  2. 优先比较最终 action chunk；
  3. 只有 final parity 超阈值时，才打开中间 dump 并二分定位。
- 新增 C++ predict 终点 dump 开关：

```text
VLA_LINGBOT_PREDICT_DUMP_DIR=/tmp/lingbot_predict_dump
```

- 该开关会在每次 `LingBotVAModelArch::predict()` 结束时写出：

```text
lingbot_predict_action_chunk.f32
lingbot_predict_action_chunk.shape.txt
lingbot_predict_action_tokens_raw.f32
lingbot_predict_action_tokens_raw.shape.txt
lingbot_predict_action_sample_final.f32
lingbot_predict_action_sample_final.shape.txt
lingbot_predict_latent_sample_final.f32
lingbot_predict_latent_sample_final.shape.txt
lingbot_predict_last_action_pred.f32
lingbot_predict_last_action_pred.shape.txt
lingbot_predict_last_latent_pred.f32
lingbot_predict_last_latent_pred.shape.txt
```

验证：

```text
rm -rf /tmp/lingbot_predict_dump && mkdir -p /tmp/lingbot_predict_dump

VLA_LINGBOT_PREDICT_SMOKE=1 \
VLA_LINGBOT_PREDICT_DUMP_DIR=/tmp/lingbot_predict_dump \
VLA_LINGBOT_PREDICT_BLOCKS=1 \
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1 \
VLA_LINGBOT_PREDICT_ACTION_STEPS=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
predict bridge ok blocks=1 mode=ggml-dense window=1 video_steps=1 action_steps=1
action_postprocess=libero_quantiles chunk=[16,7] checksum=4.62243509 max=0.150351882
```

结论：

- C++ 侧已经具备“整体最终输出 dump”能力。
- 后续真正 parity 应该新建/补齐 Python reference harness，让 Python 原版在同一输入、同一 random seed/noise、同一 timestep/action history 下 dump final action chunk，然后直接与 `lingbot_predict_action_chunk.f32` 比较。
- 中间层 smoke 保留，但只作为 final parity 失败后的定位工具。

English:

- Validation strategy is now switched from “many small smokes” to “end-to-end first”:
  1. start the full C++ LingBot path first;
  2. compare the final action chunk first;
  3. only enable intermediate dumps and binary-search localization when final parity exceeds the threshold.
- Added a C++ predict endpoint dump switch:

```text
VLA_LINGBOT_PREDICT_DUMP_DIR=/tmp/lingbot_predict_dump
```

- At the end of each `LingBotVAModelArch::predict()`, this writes:

```text
lingbot_predict_action_chunk.f32
lingbot_predict_action_chunk.shape.txt
lingbot_predict_action_tokens_raw.f32
lingbot_predict_action_tokens_raw.shape.txt
lingbot_predict_action_sample_final.f32
lingbot_predict_action_sample_final.shape.txt
lingbot_predict_latent_sample_final.f32
lingbot_predict_latent_sample_final.shape.txt
lingbot_predict_last_action_pred.f32
lingbot_predict_last_action_pred.shape.txt
lingbot_predict_last_latent_pred.f32
lingbot_predict_last_latent_pred.shape.txt
```

Verification:

```text
VLA_LINGBOT_PREDICT_SMOKE=1 \
VLA_LINGBOT_PREDICT_DUMP_DIR=/tmp/lingbot_predict_dump \
VLA_LINGBOT_PREDICT_BLOCKS=1 \
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1 \
VLA_LINGBOT_PREDICT_ACTION_STEPS=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
predict bridge ok blocks=1 mode=ggml-dense window=1 video_steps=1 action_steps=1
action_postprocess=libero_quantiles chunk=[16,7] checksum=4.62243509 max=0.150351882
```

Conclusion:

- The C++ side can now dump final end-to-end prediction outputs.
- True parity should next add/complete a Python reference harness that runs the original Python path with the same input, random seed/noise, timestep schedule, and action history, then compares directly against `lingbot_predict_action_chunk.f32`.
- Intermediate smokes remain available, but only as localization tools after final parity fails.

## 2026-06-16 18:54 CST - Python 原版整体 reference 可行性检查 / Python reference feasibility check

中文：

- 新增 `scripts/run_lingbot_va_e2e_parity.py`，用于整体优先 parity。
- 当前能力：
  - 生成 deterministic LIBERO-like fixture：
    - 两路 128x128 RGB 图像；
    - 7D robot state；
    - prompt；
    - `[16,30]` action noise。
  - 驱动 C++ `lingbot-world-server`；
  - 通过 `action_noise` 控制 C++ 初始 action sample；
  - 写出 C++ final dump；
  - 预留 Python 原版 reference 模式；
  - 支持最终 action chunk 比较。
- 同步更新 `eval/client/lingbot_world_client.py`：
  - `predict_chunk(obs, action_noise=None)`；
  - `_add_action_noise()`，通过 protobuf `StepRequest.action_noise` 传给 C++。

C++ harness 验证：

```text
eval/sim/libero/libero_uv/.venv/bin/python scripts/run_lingbot_va_e2e_parity.py \
  --mode cpp \
  --out-dir /tmp/lingbot_e2e_parity \
  --blocks 1 \
  --text-blocks 1 \
  --video-steps 1 \
  --action-steps 1
```

输出确认：

```text
lingbot_predict_action_chunk      shape=(16, 7)  checksum=5.91723728  max=0.233619809
client_action_chunk               shape=(16, 7)  checksum=5.91723728  max=0.233619809
lingbot_predict_action_tokens_raw shape=(30,16)  checksum=4.95251560  max=0.289243698
lingbot_predict_action_sample_final shape=(1,30,4,4,1) checksum=4.95251560
```

Python 原版 reference 环境检查：

- LIBERO venv:
  - `diffusers==0.30.1`
  - 缺 `diffusers.pipelines.wan`，不能 import 原版 `wan_va_server.py`。
- SimplerEnv venv:
  - `diffusers==0.36.0`，有 Wan pipeline；
  - 补装了轻依赖：

```text
easydict
einops
websockets
accelerate
ftfy
```

Python 原版运行结果：

- GPU 路径：
  - VAE/text/transformer shard 可以开始加载；
  - 在把完整 transformer 搬到 3070Ti 8GB CUDA 时 OOM：

```text
torch.OutOfMemoryError: CUDA out of memory
```

- CPU fallback 路径：
  - VAE/text/transformer 可以加载到 CPU；
  - 但完整 UMT5 / original full forward 太慢，超过当前交互节奏；
  - 已手动中断，栈停在 UMT5 self-attention CPU forward。

结论：

- “Python 原版 LIBERO 全量整体 parity”在当前本机不是一个高效日常回归路径。
- 这与我们之前的判断一致：原版 Python 22GB 权重 + 完整 transformer 在 8GB GPU 上不适合作为主验证闭环。
- 后续验证策略调整为：
  1. C++ 全链路 rollout 作为主验证；
  2. C++ final dump 持续保留；
  3. Python 原版只用于局部 reference 或离线大机器 reference；
  4. 若需要 final parity，优先构建“轻量 reference contract”：固定 latent/text/action_noise，只比较 Wan/action branch，而不是每次跑完整 UMT5+VAE+30block Python 原版。

English:

- Added `scripts/run_lingbot_va_e2e_parity.py` for end-to-end-first parity.
- Current capability:
  - generates a deterministic LIBERO-like fixture:
    - two 128x128 RGB views;
    - 7D robot state;
    - prompt;
    - `[16,30]` action noise;
  - drives the C++ `lingbot-world-server`;
  - sends controlled `action_noise` into C++;
  - writes C++ final dumps;
  - reserves a Python-original reference mode;
  - supports final action-chunk comparison.
- Updated `eval/client/lingbot_world_client.py`:
  - `predict_chunk(obs, action_noise=None)`;
  - `_add_action_noise()` through protobuf `StepRequest.action_noise`.

C++ harness verification:

```text
eval/sim/libero/libero_uv/.venv/bin/python scripts/run_lingbot_va_e2e_parity.py \
  --mode cpp \
  --out-dir /tmp/lingbot_e2e_parity \
  --blocks 1 \
  --text-blocks 1 \
  --video-steps 1 \
  --action-steps 1
```

Confirmed outputs:

```text
lingbot_predict_action_chunk      shape=(16, 7)  checksum=5.91723728  max=0.233619809
client_action_chunk               shape=(16, 7)  checksum=5.91723728  max=0.233619809
lingbot_predict_action_tokens_raw shape=(30,16)  checksum=4.95251560  max=0.289243698
lingbot_predict_action_sample_final shape=(1,30,4,4,1) checksum=4.95251560
```

Python-original reference environment check:

- LIBERO venv:
  - `diffusers==0.30.1`;
  - missing `diffusers.pipelines.wan`, so it cannot import the original `wan_va_server.py`.
- SimplerEnv venv:
  - `diffusers==0.36.0`, includes Wan pipeline;
  - installed small missing dependencies:

```text
easydict
einops
websockets
accelerate
ftfy
```

Python-original execution result:

- GPU path:
  - VAE/text/transformer shards start loading;
  - moving the full transformer to the 8GB 3070Ti CUDA device OOMs:

```text
torch.OutOfMemoryError: CUDA out of memory
```

- CPU fallback path:
  - VAE/text/transformer load on CPU;
  - full UMT5/original forward is too slow for the interactive loop;
  - manually interrupted while inside UMT5 self-attention CPU forward.

Conclusion:

- Full Python-original LIBERO end-to-end parity is not an efficient daily regression path on this local machine.
- This matches the earlier expectation: the 22GB Python model and full transformer do not fit well into an 8GB GPU workflow.
- Updated validation strategy:
  1. use C++ full-link rollout as the primary validation;
  2. keep C++ final dumps always available;
  3. use Python original only for local reference slices or offline reference on a larger machine;
  4. if final parity is needed locally, build a lightweight reference contract first: fixed latent/text/action_noise and compare the Wan/action branch, instead of running full UMT5+VAE+30-block Python original every time.

## 2026-06-16 19:13 CST - 10-fixture parity harness phase 1: C++ baseline

中文：

- 开始实施“10 个固定随机原始输入 + Python 分块 reference + C++ final output”的 parity 方案。
- 先完成第 1 阶段：固定 10 个 deterministic LIBERO-like fixtures，并用 C++ full-link path 跑出 final action dump。
- 更新 `scripts/run_lingbot_va_e2e_parity.py`：
  - 新增 `--num-fixtures N`，默认 10；
  - 新增 `--mode fixtures`，只生成 fixtures；
  - `--mode cpp` 现在一次启动 `lingbot-world-server`，循环跑完所有 fixture；
  - 每个样本输出到：

```text
<out>/cpp/sample_000/
<out>/cpp/sample_001/
...
```

  - 每个 sample 保存：

```text
client_action_chunk.f32
lingbot_predict_action_chunk.f32
lingbot_predict_action_tokens_raw.f32
lingbot_predict_action_sample_final.f32
lingbot_predict_latent_sample_final.f32
lingbot_predict_last_action_pred.f32
lingbot_predict_last_latent_pred.f32
```

- fixtures 包含：
  - 两路 128x128 RGB 图像；
  - prompt；
  - 7D robot state；
  - `[16,30]` action noise；
  - 变化的图像 seed、state、prompt、action noise phase。

验证命令：

```text
eval/sim/libero/libero_uv/.venv/bin/python -m py_compile \
  scripts/run_lingbot_va_e2e_parity.py \
  eval/client/lingbot_world_client.py

eval/sim/libero/libero_uv/.venv/bin/python scripts/run_lingbot_va_e2e_parity.py \
  --mode fixtures \
  --out-dir /tmp/lingbot_modular_parity \
  --num-fixtures 10

eval/sim/libero/libero_uv/.venv/bin/python scripts/run_lingbot_va_e2e_parity.py \
  --mode cpp \
  --out-dir /tmp/lingbot_modular_parity_cpp \
  --num-fixtures 10 \
  --blocks 1 \
  --text-blocks 1 \
  --video-steps 1 \
  --action-steps 1
```

C++ 10-fixture baseline：

```text
000 checksum=5.91723728 max=0.233619809 prompt=pick up the alphabet soup and place it in the basket
001 checksum=6.48227501 max=0.273430109 prompt=pick up the tomato sauce and place it in the basket
002 checksum=6.81425381 max=0.327453732 prompt=put the cream cheese in the basket
003 checksum=6.83273888 max=0.298802018 prompt=pick up the butter and place it in the basket
004 checksum=7.40984344 max=0.303530097 prompt=put the chocolate pudding in the basket
005 checksum=5.97485733 max=0.248710632 prompt=pick up the orange juice and place it in the basket
006 checksum=6.37658262 max=0.307757974 prompt=put the salad dressing in the basket
007 checksum=6.77864361 max=0.302134395 prompt=pick up the ketchup and place it in the basket
008 checksum=6.76662254 max=0.332090855 prompt=put the black bowl on the plate
009 checksum=6.71696377 max=0.282952428 prompt=pick up the mug and place it on the tray
```

结论：

- “10 个随机原始输入”的固定样本集已经落地。
- C++ full-link final action baseline 已经生成。
- 下一阶段接 Python 分块 reference：
  1. VAE encoder: image fixture -> latent；
  2. UMT5 encoder: prompt/token fixture -> text embedding；
  3. Wan/action branch: fixed latent/text/action_noise/action_condition -> final action；
  4. 对每个 sample 汇总 max/mean diff。

English:

- Started implementing the “10 fixed random raw inputs + Python modular reference + C++ final output” parity plan.
- Phase 1 is complete: generate 10 deterministic LIBERO-like fixtures and run the C++ full-link path to produce final action dumps.
- Updated `scripts/run_lingbot_va_e2e_parity.py`:
  - added `--num-fixtures N`, default 10;
  - added `--mode fixtures`;
  - `--mode cpp` now starts `lingbot-world-server` once and loops over all fixtures;
  - each sample is written under:

```text
<out>/cpp/sample_000/
<out>/cpp/sample_001/
...
```

- Each sample stores:

```text
client_action_chunk.f32
lingbot_predict_action_chunk.f32
lingbot_predict_action_tokens_raw.f32
lingbot_predict_action_sample_final.f32
lingbot_predict_latent_sample_final.f32
lingbot_predict_last_action_pred.f32
lingbot_predict_last_latent_pred.f32
```

- Fixtures include:
  - two 128x128 RGB views;
  - prompt;
  - 7D robot state;
  - `[16,30]` action noise;
  - varied image seeds, state, prompt, and action-noise phase.

Verification commands:

```text
eval/sim/libero/libero_uv/.venv/bin/python scripts/run_lingbot_va_e2e_parity.py \
  --mode cpp \
  --out-dir /tmp/lingbot_modular_parity_cpp \
  --num-fixtures 10 \
  --blocks 1 \
  --text-blocks 1 \
  --video-steps 1 \
  --action-steps 1
```

C++ 10-fixture baseline:

```text
000 checksum=5.91723728 max=0.233619809
001 checksum=6.48227501 max=0.273430109
002 checksum=6.81425381 max=0.327453732
003 checksum=6.83273888 max=0.298802018
004 checksum=7.40984344 max=0.303530097
005 checksum=5.97485733 max=0.248710632
006 checksum=6.37658262 max=0.307757974
007 checksum=6.77864361 max=0.302134395
008 checksum=6.76662254 max=0.332090855
009 checksum=6.71696377 max=0.282952428
```

Conclusion:

- The fixed 10-random-raw-input fixture set is now available.
- The C++ full-link final-action baseline has been generated.
- Next phase: add Python modular references:
  1. VAE encoder: image fixture -> latent;
  2. UMT5 encoder: prompt/token fixture -> text embedding;
  3. Wan/action branch: fixed latent/text/action_noise/action_condition -> final action;
  4. aggregate max/mean diff for each sample.

## 2026-06-16 15:35 CST / June 16, 2026 15:35 CST

### 真实 latent 输入与 LIBERO action postprocess / Real Latent Input and LIBERO Action Postprocess

中文：

- 真实链路中，action postprocess 不会阻塞 server 接收 latent 和执行模型，但会影响最终闭环是否可执行。
- 原因：LingBot-VA 内部 action 是 30 channel 归一化空间；LIBERO 实际只使用 `used_action_channel_ids=[0..6]` 的 7 维动作。
- 根据原版 `wan_va/configs/va_libero_cfg.py` 和 `wan_va_server.py`，C++ `predict()` 现在默认执行 LIBERO quantile unnormalize：

```text
real_action = (raw_action + 1) / 2 * (q99 - q01 + 1e-6) + q01
used channels = 0..6
```

- 默认返回 `[chunk_size=16, action_dim=7]`。
- 调试内部 30 维 raw action 时，可设置：

```bash
VLA_LINGBOT_ACTION_POSTPROCESS=raw
```

- `lingbot-world-server` 已改为根据 `predict()` 的实际返回长度填写 `response.action_dim`，不再固定写 `cfg.max_action_dim=30`。
- `StepRequest.input_latents` 已支持传入 F32 5D latent，优先识别名称：

```text
video_latent, world_latent, latent, vae_latent
```

验证：

```text
cmake --build build -j2

world-server request with video_latent shape [1,48,2,4,4]:
status step_with_latents_bridge session=16 step=1 latent_count=1 latent_bytes=6144 lang_tokens=4 used_latent=video_latent action_dim=7
len 112
chunk_size 16
action_dim 7
checksum 5.206195831298828
first7 [-0.01246, 0.082298, 0.030565, 0.055808, 0.007724, 0.004577, 0.146259]

model log:
using caller LingBot latent shape=[1,48,2,4,4] checksum=0.857021251
predict bridge ok ... action_postprocess=libero_quantiles chunk=[16,7] checksum=5.20619589 max=0.149389386
```

下一步：

- 将真实图像输入接入 `lingbot-world-server`，调用已完成 parity 的 VAE encoder 得到 latent。
- 将 world latent 输出/cache 设计成会话级状态，减少重复传输。
- 在 LIBERO 客户端侧增加 LingBot world-server 请求路径，完成端到端 smoke。

English:

- In the real path, action postprocess does not block latent ingestion or model execution, but it determines whether the final closed loop is executable.
- Reason: LingBot-VA internally denoises actions in a normalized 30-channel space, while LIBERO uses only the 7 channels from `used_action_channel_ids=[0..6]`.
- Following the original `wan_va/configs/va_libero_cfg.py` and `wan_va_server.py`, C++ `predict()` now performs LIBERO quantile unnormalization by default:

```text
real_action = (raw_action + 1) / 2 * (q99 - q01 + 1e-6) + q01
used channels = 0..6
```

- The default response shape is `[chunk_size=16, action_dim=7]`.
- For internal 30-D raw action debugging, set:

```bash
VLA_LINGBOT_ACTION_POSTPROCESS=raw
```

- `lingbot-world-server` now reports `response.action_dim` from the actual `predict()` output size instead of hard-coding `cfg.max_action_dim=30`.
- `StepRequest.input_latents` now accepts F32 5D latents, preferring tensor names:

```text
video_latent, world_latent, latent, vae_latent
```

Verification:

```text
cmake --build build -j2

world-server request with video_latent shape [1,48,2,4,4]:
status step_with_latents_bridge session=16 step=1 latent_count=1 latent_bytes=6144 lang_tokens=4 used_latent=video_latent action_dim=7
len 112
chunk_size 16
action_dim 7
checksum 5.206195831298828
first7 [-0.01246, 0.082298, 0.030565, 0.055808, 0.007724, 0.004577, 0.146259]

model log:
using caller LingBot latent shape=[1,48,2,4,4] checksum=0.857021251
predict bridge ok ... action_postprocess=libero_quantiles chunk=[16,7] checksum=5.20619589 max=0.149389386
```

Next:

- Wire real image input into `lingbot-world-server` and call the parity-checked VAE encoder to produce latents.
- Design session-level world-latent output/cache to reduce repeated tensor transfer.
- Add a LingBot world-server request path on the LIBERO client side and run an end-to-end smoke.

## 2026-06-16 16:05 CST / June 16, 2026 16:05 CST

### 真实图像输入链路 / Real Image Input Path

中文：

- 在 `src/serving/lingbot.proto` 的 `StepRequest` 中新增：

```proto
repeated Tensor input_images = 8;
```

- `lingbot-world-server` 现在支持在没有 `input_latents` 时接收 `input_images`，并将图像统一转换为 LingBot 专用 video tensor：

```text
server input accepted:
- U8  [F,H,W,3]
- U8  [3,F,H,W]
- F32 [F,H,W,3]
- F32 [3,F,H,W]

model input layout:
[views, 3, F, H, W], value range [-1, 1]
```

- `vla::Inputs` 新增 LingBot 专用 video 字段：

```text
lingbot_video
lingbot_video_views
lingbot_video_c/f/h/w
```

- `LingBotVAModelArch::predict()` 现在支持在没有 caller latent、但有 `lingbot_video` 时，通过环境变量指定的 VAE GGUF 自动编码 latent：

```bash
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf
```

- 新增 runtime wrapper：

```text
encode_lingbot_video_to_latent()
```

它复现 Python `_encode_obs()` 的关键后处理：

```text
VAE quant_conv output: 96 channels
mu = first 48 channels
mu_norm = (mu - latents_mean) / latents_std
multi-view concat: concatenate views along latent width
```

- 修复了 VAE encoder down path 中两个 smoke 专用假设：
  - patch 后空间尺寸从硬编码 `8x8` 改为动态 `W/H`；
  - temporal chunks 从 `{1,4}` / `{1,2}` 改为按实际时间维动态生成。

验证：

```text
cmake --build build -j2

world-server env:
VLA_LINGBOT_TEXT_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf
VLA_LINGBOT_RESIDENT_BLOCK_DTYPE=q8_0
VLA_LINGBOT_PREDICT_TEXT_ENCODER=1
VLA_LINGBOT_PREDICT_TEXT_BLOCKS=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1

request:
2 camera tensors, each U8 [1,128,128,3]

response:
status step_with_images_vae_bridge session=17 step=1 latent_count=0 latent_bytes=0 image_count=2 image_bytes=98304 lang_tokens=4 used_latent=none action_dim=7
len 112
chunk_size 16
action_dim 7
checksum 8.838333129882812
first7 [-0.084016, 0.279661, 0.318267, 0.040039, 0.024382, 0.025883, -0.007118]

model log:
VAE image bridge ok views=2 input=[3,1,128,128] latent=[1,48,1,8,16] checksum=-1328.3256
using VAE-encoded LingBot latent shape=[1,48,1,8,16] checksum=-1328.3256
predict bridge ok ... action_postprocess=libero_quantiles chunk=[16,7] checksum=8.83833314 max=0.321843624
```

当前限制：

- VAE image bridge 目前是 CPU/F32 路径，本次 smoke 约 `8000.9ms`，只证明链路正确，不代表最终性能。
- `return_world_latents` 暂时还不能返回模型内部生成的 VAE latent，因为编码发生在 `predict()` 内部；后续可以把 VAE encode 封成 server 可调用组件，顺便做 session-level latent cache。
- 下一步建议优先做 LIBERO client 到 `lingbot-world-server` 的真实请求路径，然后再优化 VAE CUDA/cache。

English:

- Added a new `StepRequest` field in `src/serving/lingbot.proto`:

```proto
repeated Tensor input_images = 8;
```

- `lingbot-world-server` now accepts `input_images` when no `input_latents` are provided, and converts images into the LingBot-specific video tensor:

```text
server input accepted:
- U8  [F,H,W,3]
- U8  [3,F,H,W]
- F32 [F,H,W,3]
- F32 [3,F,H,W]

model input layout:
[views, 3, F, H, W], value range [-1, 1]
```

- Added LingBot-specific video fields to `vla::Inputs`:

```text
lingbot_video
lingbot_video_views
lingbot_video_c/f/h/w
```

- `LingBotVAModelArch::predict()` now auto-encodes video into latents when caller latents are absent and `lingbot_video` is present, using:

```bash
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf
```

- Added runtime wrapper:

```text
encode_lingbot_video_to_latent()
```

It reproduces the key Python `_encode_obs()` postprocessing:

```text
VAE quant_conv output: 96 channels
mu = first 48 channels
mu_norm = (mu - latents_mean) / latents_std
multi-view concat: concatenate views along latent width
```

- Fixed two smoke-only assumptions in the VAE encoder down path:
  - patch-space `W/H` is now dynamic instead of hard-coded to `8x8`;
  - temporal chunks are generated from the actual time dimension instead of hard-coded `{1,4}` / `{1,2}`.

Verification:

```text
cmake --build build -j2

world-server env:
VLA_LINGBOT_TEXT_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf
VLA_LINGBOT_RESIDENT_BLOCK_DTYPE=q8_0
VLA_LINGBOT_PREDICT_TEXT_ENCODER=1
VLA_LINGBOT_PREDICT_TEXT_BLOCKS=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1

request:
2 camera tensors, each U8 [1,128,128,3]

response:
status step_with_images_vae_bridge session=17 step=1 latent_count=0 latent_bytes=0 image_count=2 image_bytes=98304 lang_tokens=4 used_latent=none action_dim=7
len 112
chunk_size 16
action_dim 7
checksum 8.838333129882812
first7 [-0.084016, 0.279661, 0.318267, 0.040039, 0.024382, 0.025883, -0.007118]

model log:
VAE image bridge ok views=2 input=[3,1,128,128] latent=[1,48,1,8,16] checksum=-1328.3256
using VAE-encoded LingBot latent shape=[1,48,1,8,16] checksum=-1328.3256
predict bridge ok ... action_postprocess=libero_quantiles chunk=[16,7] checksum=8.83833314 max=0.321843624
```

Current limitations:

- The VAE image bridge is currently CPU/F32; the smoke took about `8000.9ms`, so it proves correctness of the path, not final performance.
- `return_world_latents` cannot yet return the internally generated VAE latent because encoding happens inside `predict()`; next we can expose VAE encode as a server-side component and add session-level latent caching.
- Recommended next step: wire the LIBERO client to `lingbot-world-server` with real image requests, then optimize VAE CUDA/cache.

## 2026-06-16 16:35 CST / June 16, 2026 16:35 CST

### LIBERO client 接入 LingBot world-server / LIBERO Client Integration with LingBot world-server

中文：

- 新增 `eval/client/lingbot_world_client.py`。
- `LingBotWorldClient` 使用 `src/serving/lingbot.proto`，直接连接 `lingbot-world-server`，不走标准 `vla.proto`。
- 支持：

```text
reset -> ResetRequest(clear_cache=true)
get_action -> StepRequest(input_images + state + lang_tokens)
action queue -> replay predicted chunk rows
```

- `run_sim_client_direct.py` 新增 `--arch lingbot_va`。
- 当 `--arch lingbot_va` 时：
  - 直接使用 LIBERO raw observation；
  - 默认 image keys 从 LeRobot 的 `observation.images.*` 自动切换为 raw `image image2`；
  - `--vla-addr` 指向 `lingbot-world-server`；
  - `--tokenizer` 指向 LingBot 本地 tokenizer，默认尝试：

```text
/home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long/tokenizer
```

- 图像处理路径：

```text
LIBERO obs["pixels"]["image"] / obs["pixels"]["image2"]
-> flip [::-1, ::-1] to match existing LIBERO adapters/video orientation
-> resize to 128x128
-> Tensor U8 [1,128,128,3]
-> StepRequest.input_images
```

- 状态处理路径：

```text
eef pos(3) + quat->axis_angle(3) + gripper qpos first scalar(1)
-> 7-D state
```

验证：

```text
py_compile:
eval/sim/libero/libero_uv/.venv/bin/python -m py_compile \
  eval/client/lingbot_world_client.py \
  eval/client/run_sim_client_direct.py

run_sim_client_direct.py --help:
--arch includes lingbot_va
```

单步真实 LIBERO smoke：

```text
env:
libero_object/task_0

client:
LingBotWorldClient(
  vla_addr='tcp://127.0.0.1:6041',
  tokenizer_name='/home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long/tokenizer',
  image_size=128,
  image_keys=['image','image2'],
  n_action_steps=1,
)

obs pixels:
{'image': (256, 256, 3), 'image2': (256, 256, 3)}

task:
pick up the alphabet soup and place it in the basket

action:
shape (7,)
[-0.1204114556312561, 0.173681378364563, 0.07192730903625488,
 0.06249073147773743, 0.03263281285762787, -0.01537749171257019,
 0.06489086151123047]

env.step(action):
reward 0.0
done False
truncated False
info keys ['task', 'task_id', 'done', 'is_success']
```

server log:

```text
UMT5 encode host ok: tokens=13 blocks=1 checksum=12.3281671 max=1.82354021
VAE image bridge ok views=2 input=[3,1,128,128] latent=[1,48,1,8,16] checksum=31.6570007
using VAE-encoded LingBot latent shape=[1,48,1,8,16] checksum=31.6570007
predict bridge ok blocks=1 mode=ggml-dense window=1 video_steps=1 action_steps=1
action_postprocess=libero_quantiles chunk=[16,7] checksum=3.98670784 max=0.173681378
lingbot-world-server: rid=1 served=1 total=9242.0ms input_latents=0
```

可直接运行的最短命令形态：

```bash
source eval/sim/libero/libero_uv/.venv/bin/activate

python eval/client/run_sim_client_direct.py \
  --arch lingbot_va \
  --task libero_object \
  --task-id 0 \
  --n-episodes 1 \
  --n-action-steps 1 \
  --vla-addr tcp://127.0.0.1:5557 \
  --tokenizer /home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long/tokenizer \
  --recv-timeout-ms 900000 \
  --output-dir /tmp/libero_lingbot_smoke
```

当前限制：

- 本次 smoke 使用 server 侧 `VLA_LINGBOT_PREDICT_BLOCKS=1`、`VLA_LINGBOT_PREDICT_VIDEO_STEPS=1`、`VLA_LINGBOT_PREDICT_ACTION_STEPS=1`，还不是完整 LingBot 推理质量。
- VAE image bridge 仍是 CPU/F32，单请求约 9.2s。
- 现在只验证了 “client -> server -> VAE -> Wan -> action -> env.step” 单步链路，下一步需要跑完整 episode smoke，并根据失败情况调动作尺度/步数/cache。

English:

- Added `eval/client/lingbot_world_client.py`.
- `LingBotWorldClient` uses `src/serving/lingbot.proto` and connects directly to `lingbot-world-server`, not the standard `vla.proto` server.
- It supports:

```text
reset -> ResetRequest(clear_cache=true)
get_action -> StepRequest(input_images + state + lang_tokens)
action queue -> replay predicted chunk rows
```

- Added `--arch lingbot_va` to `run_sim_client_direct.py`.
- With `--arch lingbot_va`:
  - the client consumes raw LIBERO observations directly;
  - default image keys switch from LeRobot `observation.images.*` to raw `image image2`;
  - `--vla-addr` points to `lingbot-world-server`;
  - `--tokenizer` points to the local LingBot tokenizer, defaulting to:

```text
/home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long/tokenizer
```

- Image path:

```text
LIBERO obs["pixels"]["image"] / obs["pixels"]["image2"]
-> flip [::-1, ::-1] to match existing LIBERO adapters/video orientation
-> resize to 128x128
-> Tensor U8 [1,128,128,3]
-> StepRequest.input_images
```

- State path:

```text
eef pos(3) + quat->axis_angle(3) + first gripper qpos scalar(1)
-> 7-D state
```

Verification:

```text
py_compile:
eval/sim/libero/libero_uv/.venv/bin/python -m py_compile \
  eval/client/lingbot_world_client.py \
  eval/client/run_sim_client_direct.py

run_sim_client_direct.py --help:
--arch includes lingbot_va
```

Single-step real LIBERO smoke:

```text
env:
libero_object/task_0

client:
LingBotWorldClient(
  vla_addr='tcp://127.0.0.1:6041',
  tokenizer_name='/home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long/tokenizer',
  image_size=128,
  image_keys=['image','image2'],
  n_action_steps=1,
)

obs pixels:
{'image': (256, 256, 3), 'image2': (256, 256, 3)}

task:
pick up the alphabet soup and place it in the basket

action:
shape (7,)
[-0.1204114556312561, 0.173681378364563, 0.07192730903625488,
 0.06249073147773743, 0.03263281285762787, -0.01537749171257019,
 0.06489086151123047]

env.step(action):
reward 0.0
done False
truncated False
info keys ['task', 'task_id', 'done', 'is_success']
```

Server log:

```text
UMT5 encode host ok: tokens=13 blocks=1 checksum=12.3281671 max=1.82354021
VAE image bridge ok views=2 input=[3,1,128,128] latent=[1,48,1,8,16] checksum=31.6570007
using VAE-encoded LingBot latent shape=[1,48,1,8,16] checksum=31.6570007
predict bridge ok blocks=1 mode=ggml-dense window=1 video_steps=1 action_steps=1
action_postprocess=libero_quantiles chunk=[16,7] checksum=3.98670784 max=0.173681378
lingbot-world-server: rid=1 served=1 total=9242.0ms input_latents=0
```

Shortest runnable command shape:

```bash
source eval/sim/libero/libero_uv/.venv/bin/activate

python eval/client/run_sim_client_direct.py \
  --arch lingbot_va \
  --task libero_object \
  --task-id 0 \
  --n-episodes 1 \
  --n-action-steps 1 \
  --vla-addr tcp://127.0.0.1:5557 \
  --tokenizer /home/xuling/robotic_dataset/models/linbot-va-posttrain-libero-long/tokenizer \
  --recv-timeout-ms 900000 \
  --output-dir /tmp/libero_lingbot_smoke
```

Current limitations:

- This smoke used server-side `VLA_LINGBOT_PREDICT_BLOCKS=1`, `VLA_LINGBOT_PREDICT_VIDEO_STEPS=1`, and `VLA_LINGBOT_PREDICT_ACTION_STEPS=1`, so it is not full-quality LingBot inference yet.
- The VAE image bridge is still CPU/F32 and takes about 9.2s per request.
- We have verified the single-step “client -> server -> VAE -> Wan -> action -> env.step” path. Next we need a full-episode smoke and then tune action scale, inference steps, and caches based on observed failures.

## 2026-06-16 15:50 CST - Predict Cache for UMT5 Text and Common Weights

中文：

- 增加 LingBot `predict()` 内部缓存，降低连续 world-server step 的重复开销。
- 新增 UMT5 text cache：
  - key: `VLA_LINGBOT_TEXT_GGUF` 路径 + `VLA_LINGBOT_PREDICT_TEXT_BLOCKS` + token id 序列
  - value: UMT5 final hidden
  - 可用 `VLA_LINGBOT_TEXT_CACHE_DISABLE=1` 关闭
- 新增 common transformer weight cache：
  - 缓存 `make_compute_smoke_common_weights()` 加载的 common 权重
  - 避免每个 `predict()` 反复读取约 683 MiB F32 common 权重
  - 可用 `VLA_LINGBOT_COMMON_CACHE_DISABLE=1` 关闭
- 当前缓存是进程内、单模型、单线程 REQ 服务路径使用的工程缓存；如果后面 world server 改成多线程，需要给缓存加 mutex 或改成 session-aware cache。

Verification:

```text
cmake --build build -j2

world-server two repeated StepRequest calls:
rid 301 server_ms 4038.4 checksum 11.558194293640554
rid 302 server_ms 2130.8 checksum 11.558194293640554

server log:
common weight cache store
UMT5 text cache store: tokens=4 blocks=1 bytes=0.06 MiB
...
common weight cache hit
UMT5 text cache hit: tokens=4 blocks=1 checksum=3.19882102
```

- 结论：连续相同语言 token 的 step 请求已经能复用 UMT5 text hidden 和 common transformer 权重；第二次请求从约 4.0s 降到约 2.1s。
- 仍然存在的大瓶颈：每个 WanTransformer block window 仍会重复加载 block 权重；后续应做 block weight window cache 或直接转向 resident/quantized weight execution。

English:

- Added internal LingBot `predict()` caches to reduce repeated work across consecutive world-server steps.
- Added a UMT5 text cache:
  - key: `VLA_LINGBOT_TEXT_GGUF` path + `VLA_LINGBOT_PREDICT_TEXT_BLOCKS` + token id sequence
  - value: UMT5 final hidden
  - can be disabled with `VLA_LINGBOT_TEXT_CACHE_DISABLE=1`
- Added a common transformer weight cache:
  - caches the weights loaded by `make_compute_smoke_common_weights()`
  - avoids re-reading about 683 MiB of F32 common weights on every `predict()`
  - can be disabled with `VLA_LINGBOT_COMMON_CACHE_DISABLE=1`
- This is currently an in-process, single-model cache for the single-threaded REQ serving path. If the world server becomes multi-threaded, this should be protected by a mutex or replaced with a session-aware cache.

Verification:

```text
cmake --build build -j2

world-server two repeated StepRequest calls:
rid 301 server_ms 4038.4 checksum 11.558194293640554
rid 302 server_ms 2130.8 checksum 11.558194293640554

server log:
common weight cache store
UMT5 text cache store: tokens=4 blocks=1 bytes=0.06 MiB
...
common weight cache hit
UMT5 text cache hit: tokens=4 blocks=1 checksum=3.19882102
```

- Conclusion: consecutive step requests with the same language tokens now reuse both the UMT5 text hidden and common transformer weights; the second request drops from about 4.0s to about 2.1s.
- Remaining bottleneck: each WanTransformer block window still reloads block weights. Next we should add a block weight window cache or move directly toward resident/quantized weight execution.

## 2026-06-16 15:59 CST - Resident WanTransformer Block Cache

中文：

- 按更激进的 resident execution 方向，新增 WanTransformer block resident cache。
- 新增开关：
  - `VLA_LINGBOT_RESIDENT_BLOCK_CACHE=1`
  - `VLA_LINGBOT_RESIDENT_BLOCK_CACHE_DISABLE=1`
  - `VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX=N`
- 行为：
  - 第一次使用某个 block 时，从 GGUF 读取并转成 F32 `SmokeWeights`。
  - 后续 forward / predict / step 中相同 block 直接复用常驻 CPU buffer。
  - 默认最多常驻 2 个 block，避免在 8GB 本地机器上把内存吃满。
  - 超过 `VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX` 的 block 走临时加载路径。
- 这不是最终量化 resident 版本，但已经是正式 resident execution 的第一步：把反复加载 block 权重的 I/O 和 BF16->F32 转换从热路径拿掉。

Verification:

```text
cmake --build build -j2

env:
VLA_LINGBOT_RESIDENT_BLOCK_CACHE=1
VLA_LINGBOT_PREDICT_TEXT_ENCODER=1
VLA_LINGBOT_PREDICT_TEXT_BLOCKS=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1

world-server two repeated StepRequest calls:
rid 401 server_ms 3333.3 checksum 11.558194293640554
rid 402 server_ms 749.6  checksum 11.558194293640554

server log:
common weight cache store
UMT5 text cache store
resident block cache store: block=0 tensors=27
resident block cache hit: block=0
...
common weight cache hit
UMT5 text cache hit
resident block cache hit: block=0
resident block cache hit: block=0
```

- 阶段性延迟变化：
  - 无缓存 world-server bridge: 约 4.0s
  - UMT5 text + common cache: 约 2.1s
  - UMT5 text + common + resident block cache: 约 0.75s
- 仍然需要完成：
  - resident block cache 目前是 CPU F32，内存占用约每 block 624 MiB。
  - 下一阶段应做 BF16/量化 resident block，或直接将 block 权重放入 ggml/CUDA quantized buffer。
  - 多线程服务前需要给 resident cache 加锁或改成明确的 session/model cache 管理器。

English:

- Added a WanTransformer block resident cache as the first aggressive step toward resident execution.
- New switches:
  - `VLA_LINGBOT_RESIDENT_BLOCK_CACHE=1`
  - `VLA_LINGBOT_RESIDENT_BLOCK_CACHE_DISABLE=1`
  - `VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX=N`
- Behavior:
  - The first use of a block reads it from GGUF and converts it into F32 `SmokeWeights`.
  - Later forward / predict / step calls reuse the resident CPU buffer for the same block.
  - The default maximum is 2 resident blocks to avoid exhausting local memory on the 8GB target machine.
  - Blocks beyond `VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX` fall back to temporary loading.
- This is not the final quantized resident implementation, but it is the first resident execution step: repeated block weight I/O and BF16->F32 conversion are removed from the hot path.

Verification:

```text
cmake --build build -j2

env:
VLA_LINGBOT_RESIDENT_BLOCK_CACHE=1
VLA_LINGBOT_PREDICT_TEXT_ENCODER=1
VLA_LINGBOT_PREDICT_TEXT_BLOCKS=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1

world-server two repeated StepRequest calls:
rid 401 server_ms 3333.3 checksum 11.558194293640554
rid 402 server_ms 749.6  checksum 11.558194293640554

server log:
common weight cache store
UMT5 text cache store
resident block cache store: block=0 tensors=27
resident block cache hit: block=0
...
common weight cache hit
UMT5 text cache hit
resident block cache hit: block=0
resident block cache hit: block=0
```

- Stage latency trend:
  - No-cache world-server bridge: about 4.0s
  - UMT5 text + common cache: about 2.1s
  - UMT5 text + common + resident block cache: about 0.75s
- Still pending:
  - The resident block cache is CPU F32 and costs about 624 MiB per block.
  - Next stage should implement BF16/quantized resident blocks or move block weights directly into ggml/CUDA quantized buffers.
  - Before multi-threaded serving, the resident cache needs locks or a dedicated session/model cache manager.

## 2026-06-16 16:08 CST - Q8_0 Resident WanTransformer Block

中文：

- 将 resident block cache 从纯 F32 推进到可选 Q8_0 resident block。
- 新增开关：

```text
VLA_LINGBOT_RESIDENT_BLOCK_DTYPE=f32
VLA_LINGBOT_RESIDENT_BLOCK_DTYPE=q8_0
```

- 实现细节：
  - 只量化 block 内大线性层的 `weight`。
  - bias、norm weight、scale/shift table 继续使用 F32，降低精度风险。
  - GGUF 中 BF16 权重先按 tensor 读成临时 F32，再用 `ggml_quantize_chunk(..., GGML_TYPE_Q8_0, ...)` 写入 Q8_0 resident tensor。
  - `ggml_mul_mat` 直接使用 Q8_0 weight + F32 activation。
  - resident cache key 现在包含 GGUF path 和 dtype，避免同进程里 F32/Q8_0 误复用。

Verification:

```text
cmake --build build -j2

env:
VLA_LINGBOT_RESIDENT_BLOCK_CACHE=1
VLA_LINGBOT_RESIDENT_BLOCK_DTYPE=q8_0
VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX=1
VLA_LINGBOT_PREDICT_TEXT_ENCODER=1
VLA_LINGBOT_PREDICT_TEXT_BLOCKS=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1

world-server repeated StepRequest:
rid 501 server_ms 3436.0 checksum 11.36243001371622
rid 502 server_ms 504.6  checksum 11.36243001371622
repeat_diff_max 0.0
repeat_diff_mean 0.0

server log:
resident block cache store: block=0 dtype=q8_0 tensors=27 bytes=166.05 MiB
resident block cache hit: block=0 dtype=q8_0
```

- 与 F32 resident block 对比：

```text
F32 resident block bytes: 624.30 MiB
Q8_0 resident block bytes: 166.05 MiB

F32 checksum: 11.558194160461426
Q8_0 checksum: 11.36242961883545

action max_diff: 0.0060691535
action mean_diff: 0.0016720617
action relative_l2: 0.0197539509
```

- 结论：Q8_0 resident block 已经可以跑通 world-server step，内存从每 block 约 624 MiB 降到约 166 MiB，当前 1-block smoke 的 action 相对 L2 差异约 1.98%。这是后续 Q6/Q4 或 CUDA resident quantized block 的基线。

English:

- Advanced the resident block cache from F32-only to optional Q8_0 resident blocks.
- New switch:

```text
VLA_LINGBOT_RESIDENT_BLOCK_DTYPE=f32
VLA_LINGBOT_RESIDENT_BLOCK_DTYPE=q8_0
```

- Implementation details:
  - Only large linear layer weights inside a block are quantized.
  - Biases, norm weights, and scale/shift tables remain F32 to reduce accuracy risk.
  - BF16 weights from GGUF are read into temporary F32, then converted into resident Q8_0 tensors with `ggml_quantize_chunk(..., GGML_TYPE_Q8_0, ...)`.
  - `ggml_mul_mat` directly consumes Q8_0 weights with F32 activations.
  - The resident cache key now includes both GGUF path and dtype, avoiding accidental F32/Q8_0 reuse in the same process.

Verification:

```text
cmake --build build -j2

env:
VLA_LINGBOT_RESIDENT_BLOCK_CACHE=1
VLA_LINGBOT_RESIDENT_BLOCK_DTYPE=q8_0
VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX=1
VLA_LINGBOT_PREDICT_TEXT_ENCODER=1
VLA_LINGBOT_PREDICT_TEXT_BLOCKS=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1

world-server repeated StepRequest:
rid 501 server_ms 3436.0 checksum 11.36243001371622
rid 502 server_ms 504.6  checksum 11.36243001371622
repeat_diff_max 0.0
repeat_diff_mean 0.0

server log:
resident block cache store: block=0 dtype=q8_0 tensors=27 bytes=166.05 MiB
resident block cache hit: block=0 dtype=q8_0
```

- Compared with F32 resident block:

```text
F32 resident block bytes: 624.30 MiB
Q8_0 resident block bytes: 166.05 MiB

F32 checksum: 11.558194160461426
Q8_0 checksum: 11.36242961883545

action max_diff: 0.0060691535
action mean_diff: 0.0016720617
action relative_l2: 0.0197539509
```

- Conclusion: Q8_0 resident block execution now works through world-server step. Memory drops from about 624 MiB/block to about 166 MiB/block, with about 1.98% relative L2 action difference in the current 1-block smoke. This is the baseline for later Q6/Q4 or CUDA resident quantized blocks.

## 2026-06-16 16:17 CST - Real LingBot Latent Input Bridge

中文：

- 完成第一段真实输入链路：`lingbot-world-server StepRequest.input_latents` -> `vla::Inputs` -> `LingBotVAModelArch::predict()` -> WanTransformer latent branch。
- 扩展 `vla::Inputs`，新增 LingBot 专用 B,C,F,H,W latent view：

```text
lingbot_latent
lingbot_latent_b / c / f / h / w
```

- `lingbot-world-server` 现在会从 `StepRequest.input_latents` 中选择 F32 5D latent tensor：
  - 优先名称：`video_latent` / `world_latent` / `latent` / `vae_latent`
  - 如果没有名称匹配，则使用第一个 F32 5D tensor
- `predict()` 现在会优先使用 caller-provided latent：
  - shape layout: `[B, C, F, H, W]`
  - `C` 必须等于 LingBot transformer `in_channels`
  - `F/H/W` 必须能被 patch size `[patch_t, patch_h, patch_w]` 整除
  - 如果没有传 latent，才回退到 deterministic smoke latent
- world-server response status 会显示 `used_latent=<name>`，方便确认请求是否真的使用了真实 latent。

Verification:

```text
cmake --build build -j2

env:
VLA_LINGBOT_RESIDENT_BLOCK_CACHE=1
VLA_LINGBOT_RESIDENT_BLOCK_DTYPE=q8_0
VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX=1
VLA_LINGBOT_PREDICT_TEXT_ENCODER=1
VLA_LINGBOT_PREDICT_TEXT_BLOCKS=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1
```

- Sent two `StepRequest`s with `video_latent` shape `[1,48,2,4,4]`, same language tokens, but different latent scales:

```text
rid 701 scale=0.02
status step_with_latents_bridge ... used_latent=video_latent
checksum 11.36242961883545

rid 702 scale=0.04
status step_with_latents_bridge ... used_latent=video_latent
checksum 11.769811630249023

latent_sensitivity max_diff 0.0106786564
latent_sensitivity mean_diff 0.0030687298
latent_sensitivity rel_l2 0.0369107909
```

- Server log confirms real latent usage:

```text
using caller LingBot latent shape=[1,48,2,4,4] checksum=0.857021251
using caller LingBot latent shape=[1,48,2,4,4] checksum=1.7140425
```

- 结论：LingBot 的 world/video latent 已经不再只能用 deterministic smoke 输入；world-server 可以传真实 latent tensor，并且该 latent 会实际影响 action 输出。
- 仍待完成：
  - `LIBERO image -> VAE encoder -> video_latent` 的直接链路。
  - `return_world_latents` 当前仍只是回传输入 latent，后续要返回 flow 更新后的 world latent。
  - robot state / action history 还没有完整映射到 LingBot 原版 action condition。

English:

- Finished the first real-input bridge: `lingbot-world-server StepRequest.input_latents` -> `vla::Inputs` -> `LingBotVAModelArch::predict()` -> WanTransformer latent branch.
- Extended `vla::Inputs` with a LingBot-specific B,C,F,H,W latent view:

```text
lingbot_latent
lingbot_latent_b / c / f / h / w
```

- `lingbot-world-server` now selects a F32 5D latent tensor from `StepRequest.input_latents`:
  - Preferred names: `video_latent` / `world_latent` / `latent` / `vae_latent`
  - If none match, it uses the first F32 5D tensor
- `predict()` now prefers the caller-provided latent:
  - shape layout: `[B, C, F, H, W]`
  - `C` must equal the LingBot transformer `in_channels`
  - `F/H/W` must be divisible by patch size `[patch_t, patch_h, patch_w]`
  - deterministic smoke latent is used only when no caller latent is provided
- The world-server response status reports `used_latent=<name>` to make request debugging explicit.

Verification:

```text
cmake --build build -j2

env:
VLA_LINGBOT_RESIDENT_BLOCK_CACHE=1
VLA_LINGBOT_RESIDENT_BLOCK_DTYPE=q8_0
VLA_LINGBOT_RESIDENT_BLOCK_CACHE_MAX=1
VLA_LINGBOT_PREDICT_TEXT_ENCODER=1
VLA_LINGBOT_PREDICT_TEXT_BLOCKS=1
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_PREDICT_VIDEO_STEPS=1
VLA_LINGBOT_PREDICT_ACTION_STEPS=1
```

- Sent two `StepRequest`s with `video_latent` shape `[1,48,2,4,4]`, identical language tokens, and different latent scales:

```text
rid 701 scale=0.02
status step_with_latents_bridge ... used_latent=video_latent
checksum 11.36242961883545

rid 702 scale=0.04
status step_with_latents_bridge ... used_latent=video_latent
checksum 11.769811630249023

latent_sensitivity max_diff 0.0106786564
latent_sensitivity mean_diff 0.0030687298
latent_sensitivity rel_l2 0.0369107909
```

- Server log confirms real latent usage:

```text
using caller LingBot latent shape=[1,48,2,4,4] checksum=0.857021251
using caller LingBot latent shape=[1,48,2,4,4] checksum=1.7140425
```

- Conclusion: LingBot world/video latent input is no longer limited to deterministic smoke data. The world server can now pass a real latent tensor, and that latent affects the action output.
- Still pending:
  - Direct `LIBERO image -> VAE encoder -> video_latent` path.
  - `return_world_latents` currently echoes input latents; later it should return flow-updated world latents.
  - Robot state / action history is not yet fully mapped to the original LingBot action condition.

## 2026-06-16 14:42 CST - VAE Encoder Full Parity Fixed / VAE Encoder 全链路对齐完成

中文：

- 修复了 LingBot/Wan VAE encoder 的完整结构遗漏：原版 encoder 是
  `conv_in -> down_blocks[0..3] -> mid_block -> norm_out/conv_out -> quant_conv`，
  之前 C++ 路径只执行到 `down_blocks.2`，现在补齐了 `down_blocks.3`。
- `down_blocks.3` 没有空间/时间下采样，但仍然包含两个 640 通道 residual block，并且
  `WanResidualDownBlock.forward()` 最后还会加外层 `avg_shortcut(x_copy)`；在该 block 中
  这个 shortcut 等价于 identity。C++ 现在已经显式执行 `block3_out += block3_input`。
- 修复了 streaming helper 中几个原地复用 `std::vector` 时的 alias 问题：
  - `vae_causal_conv3d_stream_host()`
  - `vae_causal_conv3d_cached_ggml_execute()`
  - `vae_norm_silu_host()`
- 当前 VAE encoder full smoke 已经和 Python `AutoencoderKLWan._encode()` 对齐到 F32 数值噪声级别。

验证结果：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1
VLA_LINGBOT_VAE_DUMP_DIR=/tmp/lingbot_vae_cpp_dump
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf

block0 diff max/mean 9.5367431640625e-06 2.3197615917069925e-07
block1 diff max/mean 4.76837158203125e-06 3.7423396292979305e-07
block2 diff max/mean 3.6209821701049805e-06 6.960523251109407e-07
block3 diff max/mean 7.62939453125e-06 1.2763906624968513e-06
mid_res0 diff max/mean 7.62939453125e-06 1.2520570180640789e-06
mid_attn diff max/mean 7.62939453125e-06 1.234358933288604e-06
mid_res1 diff max/mean 7.62939453125e-06 1.2509684665928944e-06
final diff max/mean 7.62939453125e-06 1.1537777027115226e-06
final sum cpp/py -1477.4044189453125 -1477.4041748046875
```

## 2026-06-16 14:53 CST - VAE Decoder Full Parity Fixed / VAE Decoder 全链路对齐完成

中文：

- 对 VAE decoder 做了 Python reference parity 验证，参考路径为
  `AutoencoderKLWan._decode()`。
- 修复了 decoder full smoke 的结构性时间维错误：Python `_decode()` 会逐 latent frame 调用
  decoder，`upsample3d` 的第一个 chunk 不做 temporal `time_conv`，后续 chunk 才通过
  causal temporal conv 扩展时间维。
- C++ 现在维护 decoder 的 streaming chunk 列表。对于 2 个 latent frame：

```text
latent chunks: [1, 1]
after upsample3d block0: [1, 2]
after upsample3d block1: [1, 4]
final decoded frames: 5
```

- 新增/接入：
  - `vae_decoder_temporal_upsample_stream_execute()`
  - `vae_dup_up3d_stream_shortcut()`
  - decoder full smoke 的 final patch/image dump
- 当前 decoder full 输出已经和 Python `_decode()` 对齐到 F32 数值噪声级别。

验证结果：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_DECODER_FULL_SMOKE=1
VLA_LINGBOT_VAE_DUMP_DIR=/tmp/lingbot_vae_decoder_cpp_dump
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf

C++:
latent=[1,1,2,48]
patch_out=[8,8,5,12]
image=[16,16,5,3]
checksum=-220.768235
max=0.287907928

Python `_decode()` comparison:
sizes 3840 3840
py sum/max -220.76821899414062 0.2879076302051544
cpp sum/max -220.7682342529297 0.2879079282283783
diff max/mean 1.8514692783355713e-06 2.1063910082830262e-07
```

## 2026-06-16 15:05 CST - UMT5 Phase 1: Text Encoder GGUF + Metadata Smoke / UMT5 第一阶段：Text Encoder GGUF 与元数据验证

中文：

- 开始 UMT5 encoder 的 C++ 化工作。计划按 3-5 个大块推进，每个大块完成后做一次 parity/验证，整体完成后做最终 parity。
- 第一阶段完成内容：
  - `scripts/convert_lingbot_va_to_gguf.py` 新增 `--modules text_encoder`。
  - 新增 UMT5 tensor map：
    - `shared.weight -> text.token_embd.weight`
    - `encoder.final_layer_norm.weight -> text.final_norm.weight`
    - 每层 `attn_norm/q/k/v/o/relative_attention_bias/ffn_norm/wi_0/wi_1/wo`
  - GGUF metadata 新增 UMT5 关键配置：
    - `layers=24`
    - `d_model=4096`
    - `d_ff=10240`
    - `d_kv=64`
    - `heads=64`
    - `vocab_size=256384`
    - relative attention buckets / max distance / layer norm epsilon / FFN activation
  - C++ 侧新增 `validate_text_encoder_tensors()` 和
    `VLA_LINGBOT_TEXT_METADATA_SMOKE=1`。
- 已生成 UMT5 BF16 GGUF：

```text
/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf
size: 11G
```

验证结果：

```text
dry-run ok: text_encoder tensor map covers 242 tensors (layers=24 d_model=4096)

VLA_LINGBOT_TEXT_METADATA_SMOKE=1
VLA_LINGBOT_TEXT_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf

text_encoder GGUF metadata ok:
layers=24
d_model=4096
d_ff=10240
heads=64
d_kv=64
vocab=256384
tensors=242
size=10.58 GiB
BF16=242
F32=0
shape sample:
first=text.token_embd.weight:256384,4096
last=text.blk.23.ffn.wo.weight:4096,10240
```

下一阶段：

- 实现 UMT5 token embedding + RMSNorm + relative position bucket/bias helper。
- 做第一块 parity：Python UMT5 embedding / position bias / first attention 前后张量，与 C++ smoke 对比。

English:

- Started the UMT5 encoder C++ port. The plan is to move in 3-5 larger blocks, with parity/verification after each block and a final end-to-end parity check.
- Phase 1 completed:
  - Added `--modules text_encoder` to `scripts/convert_lingbot_va_to_gguf.py`.
  - Added the UMT5 tensor map:
    - `shared.weight -> text.token_embd.weight`
    - `encoder.final_layer_norm.weight -> text.final_norm.weight`
    - per-layer `attn_norm/q/k/v/o/relative_attention_bias/ffn_norm/wi_0/wi_1/wo`
  - Added UMT5 metadata to GGUF:
    - `layers=24`
    - `d_model=4096`
    - `d_ff=10240`
    - `d_kv=64`
    - `heads=64`
    - `vocab_size=256384`
    - relative attention buckets / max distance / layer norm epsilon / FFN activation
  - Added C++ `validate_text_encoder_tensors()` and
    `VLA_LINGBOT_TEXT_METADATA_SMOKE=1`.
- Generated the UMT5 BF16 GGUF:

```text
/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf
size: 11G
```

Verification:

```text
dry-run ok: text_encoder tensor map covers 242 tensors (layers=24 d_model=4096)

VLA_LINGBOT_TEXT_METADATA_SMOKE=1
VLA_LINGBOT_TEXT_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf

text_encoder GGUF metadata ok:
layers=24
d_model=4096
d_ff=10240
heads=64
d_kv=64
vocab=256384
tensors=242
size=10.58 GiB
BF16=242
F32=0
shape sample:
first=text.token_embd.weight:256384,4096
last=text.blk.23.ffn.wo.weight:4096,10240
```

Next phase:

- Implement UMT5 token embedding + RMSNorm + relative position bucket/bias helpers.
- Run the first parity checkpoint: compare Python UMT5 embedding / position bias / first-attention tensors against C++ smoke outputs.

## 2026-06-16 15:17 CST - UMT5 Phase 2: Block0 Attention + FFN Parity / UMT5 第二阶段：第 0 层 Attention + FFN 对齐

中文：

- 新增 UMT5 第 0 层 smoke：
  - `VLA_LINGBOT_TEXT_ATTN0_SMOKE=1`
  - `VLA_LINGBOT_TEXT_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf`
  - 可选 dump：`VLA_LINGBOT_TEXT_DUMP_DIR=/tmp/lingbot_umt5_attn0_cpp`
- 该 smoke 只读取第 0 层所需权重，不加载完整 10.58GiB UMT5 到常驻内存：
  - attention norm
  - q/k/v/o
  - relative attention bias
  - FFN norm
  - wi_0 / wi_1 / wo
- 已实现 host/F32 参考执行：
  - UMT5 RMSNorm
  - bidirectional relative position bucket
  - relative attention bias
  - no-scale self-attention
  - gated GELU FFN: `gelu_new(wi_0(x)) * wi_1(x) -> wo`
  - residual add
- 本阶段使用 `seq=4, d_model=4096` 的确定性 hidden states 做 parity，不涉及 tokenizer/embedding row lookup。

验证结果：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_TEXT_ATTN0_SMOKE=1
VLA_LINGBOT_TEXT_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf
VLA_LINGBOT_TEXT_DUMP_DIR=/tmp/lingbot_umt5_attn0_cpp

C++ smoke:
seq=4 dim=4096 heads=64
norm_checksum=1.82051608
q_checksum=1.47725896
bias_checksum=-1224.53149
attn_residual_checksum=-157.959608
block0_checksum=25991.3281
max=13687.0273

Python manual reference from original safetensors:
norm diff max/mean              3.2708048820495605e-06 1.423101423370099e-07
q diff max/mean                 1.914799213409424e-06 2.1948537209937058e-07
k diff max/mean                 1.329183578491211e-05 1.4821339391346555e-06
v diff max/mean                 2.09808349609375e-05 1.0962407941406127e-06
position_bias diff max/mean     0.0 0.0
context diff max/mean           1.8864870071411133e-05 9.111403187489486e-07
attn_out diff max/mean          0.00136566162109375 1.0484469385119155e-05
residual diff max/mean          0.00136566162109375 1.048476951837074e-05
block0_ffn_norm diff max/mean   3.1948089599609375e-05 8.414648391408264e-07
block0_ffn_gate diff max/mean   0.000492095947265625 2.3360224076895975e-05
block0_ffn_linear diff max/mean 0.00042724609375 1.6487758330185898e-05
block0_ffn_hidden diff max/mean 0.0067901611328125 4.924112727167085e-05
block0_ffn_out diff max/mean    0.3076171875 0.0017194414976984262
block0_out diff max/mean        0.3076171875 0.0017179702408611774
```

说明：

- `position_bias` 完全一致，说明 relative bucket 逻辑正确。
- attention 内部 Q/K/V/context 已经是 1e-5 量级。
- FFN 最终 max diff 较大但 mean 很小，且输出最大值约 `1.37e4`，主要来自超大 F32 矩阵乘法的累加顺序差异。
- 下一阶段应把单层 host 公式泛化成可复用 executor，并开始多层 streaming/block-window 运行；再根据内存压力决定是否落 ggml 图或 CUDA/量化路径。

English:

- Added a UMT5 layer-0 smoke:
  - `VLA_LINGBOT_TEXT_ATTN0_SMOKE=1`
  - `VLA_LINGBOT_TEXT_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf`
  - optional dump: `VLA_LINGBOT_TEXT_DUMP_DIR=/tmp/lingbot_umt5_attn0_cpp`
- The smoke reads only the tensors needed by layer 0 instead of keeping the full 10.58GiB UMT5 resident:
  - attention norm
  - q/k/v/o
  - relative attention bias
  - FFN norm
  - wi_0 / wi_1 / wo
- Implemented host/F32 reference execution:
  - UMT5 RMSNorm
  - bidirectional relative position bucket
  - relative attention bias
  - no-scale self-attention
  - gated GELU FFN: `gelu_new(wi_0(x)) * wi_1(x) -> wo`
  - residual add
- This phase uses deterministic hidden states with `seq=4, d_model=4096`; tokenizer/embedding row lookup is not included yet.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_TEXT_ATTN0_SMOKE=1
VLA_LINGBOT_TEXT_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf
VLA_LINGBOT_TEXT_DUMP_DIR=/tmp/lingbot_umt5_attn0_cpp

C++ smoke:
seq=4 dim=4096 heads=64
norm_checksum=1.82051608
q_checksum=1.47725896
bias_checksum=-1224.53149
attn_residual_checksum=-157.959608
block0_checksum=25991.3281
max=13687.0273

Python manual reference from original safetensors:
norm diff max/mean              3.2708048820495605e-06 1.423101423370099e-07
q diff max/mean                 1.914799213409424e-06 2.1948537209937058e-07
k diff max/mean                 1.329183578491211e-05 1.4821339391346555e-06
v diff max/mean                 2.09808349609375e-05 1.0962407941406127e-06
position_bias diff max/mean     0.0 0.0
context diff max/mean           1.8864870071411133e-05 9.111403187489486e-07
attn_out diff max/mean          0.00136566162109375 1.0484469385119155e-05
residual diff max/mean          0.00136566162109375 1.048476951837074e-05
block0_ffn_norm diff max/mean   3.1948089599609375e-05 8.414648391408264e-07
block0_ffn_gate diff max/mean   0.000492095947265625 2.3360224076895975e-05
block0_ffn_linear diff max/mean 0.00042724609375 1.6487758330185898e-05
block0_ffn_hidden diff max/mean 0.0067901611328125 4.924112727167085e-05
block0_ffn_out diff max/mean    0.3076171875 0.0017194414976984262
block0_out diff max/mean        0.3076171875 0.0017179702408611774
```

Notes:

- `position_bias` matches exactly, confirming the relative bucket logic.
- Q/K/V/context are already around the 1e-5 level.
- The final FFN max diff is larger but the mean is small, and the output max is about `1.37e4`; this is mainly F32 accumulation-order drift in very large matrix multiplications.
- Next phase should generalize the single-layer host formula into a reusable executor and start multi-layer streaming/block-window execution; then decide whether to move the path to ggml graphs or CUDA/quantized kernels based on memory pressure.

English:

- Added Python reference parity verification for the VAE decoder using
  `AutoencoderKLWan._decode()`.
- Fixed a structural temporal-shape bug in the decoder full smoke. Python `_decode()`
  calls the decoder one latent frame at a time; the first `upsample3d` chunk does not run
  temporal `time_conv`, while later chunks use causal temporal conv and expand the time axis.
- C++ now keeps a streaming chunk list through decoder upsampling. For two latent frames:

```text
latent chunks: [1, 1]
after upsample3d block0: [1, 2]
after upsample3d block1: [1, 4]
final decoded frames: 5
```

- Added/wired:
  - `vae_decoder_temporal_upsample_stream_execute()`
  - `vae_dup_up3d_stream_shortcut()`
  - final patch/image dumps for decoder full smoke
- The decoder full output now matches Python `_decode()` at F32 numerical-noise level.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_DECODER_FULL_SMOKE=1
VLA_LINGBOT_VAE_DUMP_DIR=/tmp/lingbot_vae_decoder_cpp_dump
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf

C++:
latent=[1,1,2,48]
patch_out=[8,8,5,12]
image=[16,16,5,3]
checksum=-220.768235
max=0.287907928

Python `_decode()` comparison:
sizes 3840 3840
py sum/max -220.76821899414062 0.2879076302051544
cpp sum/max -220.7682342529297 0.2879079282283783
diff max/mean 1.8514692783355713e-06 2.1063910082830262e-07
```

English:

- Fixed a missing structural part in the LingBot/Wan VAE encoder. The original encoder path is
  `conv_in -> down_blocks[0..3] -> mid_block -> norm_out/conv_out -> quant_conv`; the C++ path
  previously stopped after `down_blocks.2`, and now includes `down_blocks.3`.
- `down_blocks.3` has no spatial/temporal downsampling, but it still contains two 640-channel
  residual blocks and `WanResidualDownBlock.forward()` adds an outer `avg_shortcut(x_copy)`.
  In this block the shortcut is identity-like, so C++ now explicitly applies `block3_out += block3_input`.
- Fixed aliasing bugs in streaming helpers when the same `std::vector` is used as input and output:
  - `vae_causal_conv3d_stream_host()`
  - `vae_causal_conv3d_cached_ggml_execute()`
  - `vae_norm_silu_host()`
- The current VAE encoder full smoke now matches Python `AutoencoderKLWan._encode()` at F32 numerical-noise level.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1
VLA_LINGBOT_VAE_DUMP_DIR=/tmp/lingbot_vae_cpp_dump
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf

block0 diff max/mean 9.5367431640625e-06 2.3197615917069925e-07
block1 diff max/mean 4.76837158203125e-06 3.7423396292979305e-07
block2 diff max/mean 3.6209821701049805e-06 6.960523251109407e-07
block3 diff max/mean 7.62939453125e-06 1.2763906624968513e-06
mid_res0 diff max/mean 7.62939453125e-06 1.2520570180640789e-06
mid_attn diff max/mean 7.62939453125e-06 1.234358933288604e-06
mid_res1 diff max/mean 7.62939453125e-06 1.2509684665928944e-06
final diff max/mean 7.62939453125e-06 1.1537777027115226e-06
final sum cpp/py -1477.4044189453125 -1477.4041748046875
```

## 2026-06-16 13:27 CST - VAE parity fix round / VAE parity 修复轮次

中文：

- 修复了 C++ VAE RMSNorm 的核心轴布局问题。
  - ggml 的 `ggml_permute()` 参数语义不是 PyTorch 风格的“新轴取旧轴”，而是“旧轴放到新轴位置”。
  - 因此原来的 `[W,H,T,C] -> [C,W,H,T]` 实际排列错误，导致 gamma 不能正确按通道广播。
  - 已把 `vae_norm_silu_to_conv_layout()` 改成正确的通道优先 RMSNorm 路径。
- 将 VAE norm gamma 的 ggml tensor 布局统一为 `[C,1,1,1]`。
- Decoder direct/full 路径已接入 `DupUp3D` shortcut 的 host 实现。
  - 当前 direct smoke 仍不是 Python `decode()` 的 streaming 语义，所以 latent T=2 仍输出 direct path 的 8 帧，而 Python reference `decode()` 输出 5 帧。
- 新增 host helper：
  - `vae_avg_down3d_host()`
  - `vae_dup_up3d_host()`
  - `vae_add_same_shape()`
- 已验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1:
VAE full encoder smoke ok
out=[1,1,5,96]
checksum=-312179.269
max=1694.11938

VLA_LINGBOT_VAE_DECODER_FULL_SMOKE=1:
VAE decoder full smoke ok
patch_out=[8,8,8,12]
checksum=67057.296
max=24.9252377
image=[16,16,8,3]
```

- 重要待修复：
  - Python 原版 encoder/decoder reference 使用 `_encode()` / `_decode()` 的 streaming cache 路径。
  - `downsample3d` 的 time conv 是 stride=2；当前 C++ CUDA time-conv executor 主要是 stride=1 语义。
  - decoder `upsample3d` 第一块有 `first_chunk=True` 裁剪语义，必须实现后才会得到 Python reference 的时间长度。
  - encoder block0 的 `AvgDown3D` shortcut 需要以 `conv_in` 后的 160 通道特征为输入，当前 block0 C++ graph 把 `conv_in + block0` 合在一起，后续要拆开或在 graph 内部接 shortcut。

English:

- Fixed the core axis-layout bug in the C++ VAE RMSNorm path.
  - ggml `ggml_permute()` uses “old axis goes to new axis position” semantics, not PyTorch-style “new axis reads old axis”.
  - The previous `[W,H,T,C] -> [C,W,H,T]` conversion was therefore wrong and prevented channel-wise gamma broadcasting.
  - `vae_norm_silu_to_conv_layout()` now uses the corrected channel-first RMSNorm layout.
- Standardized VAE norm gamma ggml tensors to `[C,1,1,1]`.
- Added the host `DupUp3D` shortcut into the decoder direct/full path.
  - The current direct smoke is still not Python `decode()` streaming semantics; latent T=2 therefore still produces 8 direct-path frames, while Python reference `decode()` produces 5 frames.
- Added host helpers:
  - `vae_avg_down3d_host()`
  - `vae_dup_up3d_host()`
  - `vae_add_same_shape()`
- Verified:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1:
VAE full encoder smoke ok
out=[1,1,5,96]
checksum=-312179.269
max=1694.11938

VLA_LINGBOT_VAE_DECODER_FULL_SMOKE=1:
VAE decoder full smoke ok
patch_out=[8,8,8,12]
checksum=67057.296
max=24.9252377
image=[16,16,8,3]
```

- Remaining critical fixes:
  - The Python reference encoder/decoder uses `_encode()` / `_decode()` streaming cache semantics.
  - `downsample3d` time conv uses stride=2; the current C++ CUDA time-conv executor is mostly stride=1.
  - Decoder `upsample3d` has `first_chunk=True` trimming in the first chunk; this is required to match Python reference time lengths.
  - Encoder block0 `AvgDown3D` shortcut must consume the 160-channel `conv_in` output. The current block0 C++ graph fuses `conv_in + block0`, so it must be split or wired inside the graph.

## 2026-06-16 13:35 CST - VAE shortcut and stride kernel / VAE shortcut 与 stride kernel

中文：

- 将 encoder `conv_in` 拆成独立 executor：
  - `vae_encoder_conv_in_ggml_execute()`
  - encoder block0 现在按 `conv_in -> block0 main -> AvgDown3D(conv_in) -> add` 的结构执行。
- Decoder direct/full path 已加入每个 up block 的 `DupUp3D(x_copy)` shortcut。
- 新增 CUDA batched causal conv stride 入口：
  - `lingbot_causal_conv1d_cache_f32_batched_stride()`
  - 用于后续实现 Python `downsample3d` 的 stride=2 time conv。
- 新增 C++ WHDC stride wrapper：
  - `vae_time_conv_cuda_whdc_execute_io_stride()`
- 已验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_CAUSAL_CONV_VAE_SMOKE=1:
VAE-like cuda smoke T=4 Cin=32 Cout=32 K=3 max_diff=1.49011612e-08 cache_diff=0
VAE-like stride cuda smoke T=4 Tout=2 stride=2 max_diff=1.49011612e-08 cache_diff=0

VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1:
VAE full encoder smoke ok
out=[1,1,5,96]
checksum=-312032.076
max=1692.45007
```

- 注意：
  - 当前 encoder full smoke 仍是 direct path，不是 Python `_encode()` streaming path。
  - stride kernel 已经可用，但还没有替换 direct smoke；下一步应新开 streaming encode executor，而不是强行改 direct smoke。

English:

- Split encoder `conv_in` into a standalone executor:
  - `vae_encoder_conv_in_ggml_execute()`
  - encoder block0 now follows `conv_in -> block0 main -> AvgDown3D(conv_in) -> add`.
- Added `DupUp3D(x_copy)` shortcuts to the decoder direct/full up-block path.
- Added a CUDA batched causal-conv stride entry point:
  - `lingbot_causal_conv1d_cache_f32_batched_stride()`
  - This is required for the Python `downsample3d` stride=2 time conv.
- Added a C++ WHDC stride wrapper:
  - `vae_time_conv_cuda_whdc_execute_io_stride()`
- Verified:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_CAUSAL_CONV_VAE_SMOKE=1:
VAE-like cuda smoke T=4 Cin=32 Cout=32 K=3 max_diff=1.49011612e-08 cache_diff=0
VAE-like stride cuda smoke T=4 Tout=2 stride=2 max_diff=1.49011612e-08 cache_diff=0

VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1:
VAE full encoder smoke ok
out=[1,1,5,96]
checksum=-312032.076
max=1692.45007
```

- Note:
  - The current encoder full smoke is still the direct path, not the Python `_encode()` streaming path.
  - The stride kernel is available now, but it should be wired through a new streaming encode executor rather than forced into the direct smoke.

## 2026-06-16 14:05 CST - Encoder parity narrowed to mid/tail streaming / Encoder parity 定位到 mid/tail streaming

中文：

- 修复了 VAE RMSNorm 的重复缩放问题。
  - `ggml_rms_norm()` 已经按 `sqrt(mean(x^2))` 归一化，等价于 Python `F.normalize(...)*sqrt(C)`。
  - 之前 C++ 又额外乘了一次 `sqrt(C)`，导致 resnet 内部放大约 12.6 倍。
- 修复了 `WanCausalConv3d` 的 temporal causal padding：
  - 3x3x3 conv 现在走 `ggml_pad_ext()` 的 `[W/H 双侧 pad, T 左侧 pad] + conv3d padding=0`。
- 修复了 encoder spatial downsample：
  - Python 是 `ZeroPad2d((0,1,0,1)) + Conv2d(stride=2, padding=0)`。
  - C++ 已改成右/底补零，不再用对称 padding。
- 修复了 `downsample3d` streaming time stage：
  - 第一块 cache 为空时直接返回 spatial resample。
  - 后续块使用上一块最后一帧拼接当前 chunk，再做无 padding stride=2 time conv。
  - 当前先用 host 实现，后续可 CUDA 化。
- 分段 parity 结果：

```text
patch_input diff max ~= 6.5e-09
conv_in diff max ~= 1.5e-08
block0 diff max ~= 9.5e-06
block1 diff max ~= 4.8e-06
block2 diff max ~= 3.7e-06
```

- 剩余偏差：
  - C++ mid/tail 目前把 block2 的 T=2 拼起来做 no-cache full execution。
  - Python `_encode()` 是逐 chunk 调用 mid_block 和 tail conv，并使用 feat_cache。
  - 已验证 C++ mid_res0 与 Python no-cache full mid_res0 对齐，但与 Python `_encode()` streaming mid_res0 不同。
  - 所以下一步必须实现 encoder mid/tail streaming executor，而不是继续修基础算子。

English:

- Fixed duplicate scaling in VAE RMSNorm.
  - `ggml_rms_norm()` already normalizes by `sqrt(mean(x^2))`, which matches Python `F.normalize(...)*sqrt(C)`.
  - The C++ path had an extra `sqrt(C)` multiply, amplifying resnet activations by about 12.6x.
- Fixed `WanCausalConv3d` temporal causal padding:
  - 3x3x3 conv now uses `ggml_pad_ext()` with spatial two-sided padding and temporal left padding, followed by conv3d padding=0.
- Fixed encoder spatial downsample:
  - Python uses `ZeroPad2d((0,1,0,1)) + Conv2d(stride=2, padding=0)`.
  - C++ now pads only right/bottom instead of symmetric padding.
- Fixed `downsample3d` streaming time stage:
  - The first cache-empty chunk returns the spatial resample directly.
  - Later chunks concatenate the previous final frame with the current chunk and run valid stride=2 time conv.
  - This is currently a host implementation; it can be CUDA-specialized later.
- Segment parity results:

```text
patch_input diff max ~= 6.5e-09
conv_in diff max ~= 1.5e-08
block0 diff max ~= 9.5e-06
block1 diff max ~= 4.8e-06
block2 diff max ~= 3.7e-06
```

- Remaining mismatch:
  - C++ mid/tail currently concatenates block2 T=2 and runs no-cache full execution.
  - Python `_encode()` calls mid_block and tail conv chunk-by-chunk with feat_cache.
  - C++ mid_res0 matches Python no-cache full mid_res0, but differs from Python `_encode()` streaming mid_res0.
  - Next step: implement an encoder mid/tail streaming executor rather than changing basic operators.

## 2026-06-16 11:46:22 CST - VAE Encoder Mid/Tail Main Path / VAE Encoder Mid/Tail 主链路

中文：

- 继续推进第 5 步：VAE encoder C++/ggml 支持。
- 新增并接通 VAE encoder 尾部主链路：
  - `vae_encoder_down_path_execute()`：复用已验证的 `down_blocks.0/1/2`；
  - `vae_mid_resnet_ggml_execute()`：执行 `encoder.mid_block.resnets.{0,1}`；
  - `vae_mid_attention_host_execute()`：执行 `encoder.mid_block.attentions.0` 的 group norm、QKV、dense self-attention、proj、residual；
  - `vae_encoder_tail_ggml_execute()`：执行 `encoder.norm_out -> silu -> encoder.conv_out -> quant_conv`；
  - `vae_encoder_mid_tail_execute()`：把 mid block 和 tail 串成可复用 helper。
- 新增 smoke 开关：
  - `VLA_LINGBOT_VAE_ENCODER_TAIL_SMOKE=1`
  - `VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1`
- 当前 mid attention 是 host dense reference 实现，优先保证语义链路正确；后续可以替换为 ggml/CUDA 高效版本。
- `quant_conv` 输出维度为 `[W,H,T,96]`，对应 VAE encoder 的 latent 分布参数输出。

English:

- Continued step 5: C++/ggml support for the VAE encoder.
- Added and wired the VAE encoder tail main path:
  - `vae_encoder_down_path_execute()`: reuses the verified `down_blocks.0/1/2` path;
  - `vae_mid_resnet_ggml_execute()`: executes `encoder.mid_block.resnets.{0,1}`;
  - `vae_mid_attention_host_execute()`: executes group norm, QKV, dense self-attention, projection, and residual for `encoder.mid_block.attentions.0`;
  - `vae_encoder_tail_ggml_execute()`: executes `encoder.norm_out -> silu -> encoder.conv_out -> quant_conv`;
  - `vae_encoder_mid_tail_execute()`: packages the mid block and tail as a reusable helper.
- Added smoke switches:
  - `VLA_LINGBOT_VAE_ENCODER_TAIL_SMOKE=1`
  - `VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1`
- The current mid attention is a host dense reference implementation, prioritizing semantic correctness first; it can later be replaced by a ggml/CUDA optimized path.
- `quant_conv` output shape is `[W,H,T,96]`, corresponding to the VAE encoder latent distribution parameters.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_ENCODER_TAIL_SMOKE=1:
input=[1,1,5,640]
mid_res0_checksum=834.075077
attn_checksum=389.741571
mid_res1_checksum=769.677094
out=[1,1,5,96]
checksum=-2094.47117
max=13.7824039

VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1:
input=[8,8,5,12]
block0=[4,4,5,160]
block1=[2,2,5,320]
block2=[1,1,5,640]
mid_res0_checksum=950.83252
attn_checksum=850.363439
mid_res1_checksum=1466.25349
out=[1,1,5,96]
checksum=-2966.09307
max=15.7149105
```

## 2026-06-16 11:54:37 CST - CUDA VAE Mid Attention / CUDA VAE Mid Attention 高效路径

中文：

- 将 VAE encoder `mid_block.attentions.0` 从默认 host dense reference 升级为默认 CUDA 执行路径。
- 新增 CUDA 入口：

```text
lingbot_vae_mid_attn_f32()
```

- CUDA 路径覆盖：
  - WHDC 输入上的 group norm；
  - QKV linear；
  - dense self-attention；
  - projection linear；
  - residual add 并写回 WHDC。
- C++ 侧新增：
  - `vae_mid_attention_cuda_execute()`
  - `vae_mid_attention_execute()`
- 默认行为：
  - 编译了 `VLA_LINGBOT_FLEX_CUDA_KERNELS` 时，mid attention 默认走 CUDA；
  - 设置 `VLA_LINGBOT_VAE_MID_ATTN_HOST=1` 可强制使用 host dense reference；
  - 如果 CUDA path 失败，会打印 warning 并回退到 host path。
- 这一步把 VAE mid attention 的主要计算从 CPU loop 移到 GPU，为后续更大输入、更高吞吐和进一步 kernel fusion 留出了空间。

English:

- Upgraded VAE encoder `mid_block.attentions.0` from the default host dense reference path to a default CUDA execution path.
- Added CUDA entry:

```text
lingbot_vae_mid_attn_f32()
```

- CUDA path covers:
  - group norm on WHDC input;
  - QKV linear;
  - dense self-attention;
  - projection linear;
  - residual add and WHDC write-back.
- Added C++ side helpers:
  - `vae_mid_attention_cuda_execute()`
  - `vae_mid_attention_execute()`
- Default behavior:
  - when `VLA_LINGBOT_FLEX_CUDA_KERNELS` is compiled, mid attention uses CUDA by default;
  - set `VLA_LINGBOT_VAE_MID_ATTN_HOST=1` to force the host dense reference;
  - if the CUDA path fails, it prints a warning and falls back to the host path.
- This moves the main VAE mid-attention compute from CPU loops to GPU, leaving room for larger inputs, higher throughput, and further kernel fusion.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_ENCODER_TAIL_SMOKE=1, default CUDA:
attn_checksum=389.741546
out checksum=-2094.47135
max=13.7824059

VLA_LINGBOT_VAE_ENCODER_TAIL_SMOKE=1, forced host:
attn_checksum=389.741571
out checksum=-2094.47117
max=13.7824039

VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1, default CUDA:
attn_checksum=850.363467
out checksum=-2966.09299
max=15.7149105
```

## 2026-06-16 12:03:59 CST - VAE Decoder Full GGUF and Mid Path / VAE Decoder 完整 GGUF 与 Mid Path

中文：

- 继续推进第 5 步：VAE decoder C++/ggml 支持。
- 生成完整 VAE F32 GGUF：

```text
/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf
size: 2.7G
scope: full
tensors: 196
```

- 更新 VAE validator：
  - `encoder_quant_conv` scope 仍要求 86 tensors；
  - `full` scope 现在明确要求 196 tensors。
- 将 VAE mid attention helper 改成可传 prefix，因此 encoder 和 decoder 共享同一套 CUDA/host attention 路径：
  - encoder prefix: `vae.encoder.mid_block.attentions.0`
  - decoder prefix: `vae.decoder.mid_block.attentions.0`
- 新增 decoder 起始段 smoke：
  - `vae_decoder_post_quant_conv_in_ggml_execute()`
  - `run_vae_decoder_mid_smoke()`
  - 环境变量：`VLA_LINGBOT_VAE_DECODER_MID_SMOKE=1`
- 当前 decoder 已跑通：

```text
latent [1,1,5,48]
-> post_quant_conv
-> decoder.conv_in [1,1,5,1024]
-> decoder.mid_block.resnets.0
-> decoder.mid_block.attentions.0
-> decoder.mid_block.resnets.1
```

- 尚未完成：
  - `decoder.up_blocks.0/1/2/3`
  - upsampler spatial path；
  - upsampler temporal `time_conv` path；
  - `decoder.norm_out -> decoder.conv_out`；
  - final unpatchify/image reconstruction path。

English:

- Continued step 5: C++/ggml support for the VAE decoder.
- Generated the full VAE F32 GGUF:

```text
/home/xuling/robotic_dataset/models/lingbot_va_vae_full_f32.gguf
size: 2.7G
scope: full
tensors: 196
```

- Updated the VAE validator:
  - `encoder_quant_conv` scope still expects 86 tensors;
  - `full` scope now explicitly expects 196 tensors.
- Made the VAE mid-attention helper prefix-configurable, so encoder and decoder share the same CUDA/host attention path:
  - encoder prefix: `vae.encoder.mid_block.attentions.0`
  - decoder prefix: `vae.decoder.mid_block.attentions.0`
- Added a decoder starting-path smoke:
  - `vae_decoder_post_quant_conv_in_ggml_execute()`
  - `run_vae_decoder_mid_smoke()`
  - env switch: `VLA_LINGBOT_VAE_DECODER_MID_SMOKE=1`
- Current decoder path that passes smoke:

```text
latent [1,1,5,48]
-> post_quant_conv
-> decoder.conv_in [1,1,5,1024]
-> decoder.mid_block.resnets.0
-> decoder.mid_block.attentions.0
-> decoder.mid_block.resnets.1
```

- Still not complete:
  - `decoder.up_blocks.0/1/2/3`;
  - upsampler spatial path;
  - upsampler temporal `time_conv` path;
  - `decoder.norm_out -> decoder.conv_out`;
  - final unpatchify/image reconstruction path.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_DECODER_MID_SMOKE=1
VAE GGUF metadata ok: scope=full tensors=196 shapes=196 2688.17 MiB F32=196 other=0
input=[1,1,5,48]
conv_in=[1,1,5,1024] checksum=-5.91592272
mid_res0_checksum=-325.087057
attn_checksum=-286.842035
mid_res1_checksum=-281.203261
max=9.59588909
```

## 2026-06-16 12:49:33 CST - VAE Decoder Up-Block ResNet Trunk / VAE Decoder Up-Block ResNet 主干

中文：

- 继续推进 VAE decoder。
- 新增通用 ResNet executor：

```text
vae_resnet_ggml_execute()
```

- 该 helper 支持：
  - `norm1 -> silu -> conv1`
  - `norm2 -> silu -> conv2`
  - optional `conv_shortcut`
  - residual add
- 新增 decoder up-block ResNet 主干：
  - `vae_decoder_mid_execute()`
  - `vae_decoder_up_block_resnets_execute()`
  - `run_vae_decoder_up_resnets_smoke()`
- 新增 smoke 开关：

```text
VLA_LINGBOT_VAE_DECODER_UP_RESNETS_SMOKE=1
```

- 当前已验证的 decoder 链路：

```text
latent [1,1,5,48]
-> post_quant_conv
-> decoder.conv_in
-> decoder.mid_block
-> decoder.up_blocks.0.resnets.{0,1,2}
-> decoder.up_blocks.1.resnets.{0,1,2}
-> decoder.up_blocks.2.resnets.{0,1,2}
-> decoder.up_blocks.3.resnets.{0,1,2}
```

- 注意：这一步只验证 up block 的 ResNet 主干，不包含 spatial upsampler 和 temporal `time_conv`。实际 decoder 的空间/时间尺寸恢复仍需继续实现。

English:

- Continued the VAE decoder implementation.
- Added a generic ResNet executor:

```text
vae_resnet_ggml_execute()
```

- This helper supports:
  - `norm1 -> silu -> conv1`
  - `norm2 -> silu -> conv2`
  - optional `conv_shortcut`
  - residual add
- Added the decoder up-block ResNet trunk:
  - `vae_decoder_mid_execute()`
  - `vae_decoder_up_block_resnets_execute()`
  - `run_vae_decoder_up_resnets_smoke()`
- Added smoke switch:

```text
VLA_LINGBOT_VAE_DECODER_UP_RESNETS_SMOKE=1
```

- Currently verified decoder path:

```text
latent [1,1,5,48]
-> post_quant_conv
-> decoder.conv_in
-> decoder.mid_block
-> decoder.up_blocks.0.resnets.{0,1,2}
-> decoder.up_blocks.1.resnets.{0,1,2}
-> decoder.up_blocks.2.resnets.{0,1,2}
-> decoder.up_blocks.3.resnets.{0,1,2}
```

- Note: this step validates only the up-block ResNet trunk. The spatial upsampler and temporal `time_conv` are not included yet, so real decoder spatial/temporal size restoration still needs to be implemented.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_DECODER_UP_RESNETS_SMOKE=1
mid_checksum=-281.203261
up0=[1,1,5,1024] checksum=-229.31379
up1=[1,1,5,1024] checksum=518.653208
up2=[1,1,5,512] checksum=109.53156
up3=[1,1,5,256] checksum=-285.630988
max=4.59653711
```

## 2026-06-16 12:56:15 CST - Full VAE Decoder Smoke / 完整 VAE Decoder Smoke

中文：

- 按用户要求，将 decoder 剩余模块一次性补齐后统一验证。
- 新增支持 `C_in != C_out` 的 temporal conv executor：
  - `LingBotVaeCudaTimeConvWeightsIO`
  - `vae_time_conv_cuda_whdc_execute_io()`
- 新增 decoder spatial upsampler：
  - `vae_decoder_spatial_upsample_ggml_execute()`
  - 当前使用 `ggml_interpolate(..., GGML_SCALE_MODE_NEAREST)` 做 2x spatial upsample，然后接 `upsampler.resample.1` 2D conv。
- 新增 decoder temporal upsampler：
  - `vae_decoder_temporal_upsample_execute()`
  - 当前根据权重形状 `[2*C, C, 3, 1, 1]`，将 temporal conv 输出的 `2*C` 通道拆成两个连续时间步，实现 T x2。
- 新增 decoder tail：
  - `vae_decoder_tail_ggml_execute()`
  - 执行 `decoder.norm_out -> silu -> decoder.conv_out`。
- 新增 unpatchify：
  - `vae_unpatchify_ps2_rgb()`
  - 将 patchified `[W,H,T,12]` 还原成 RGB `[2W,2H,T,3]`。
- 新增完整 decoder executor 和 smoke：
  - `vae_decoder_full_execute()`
  - `run_vae_decoder_full_smoke()`
  - 环境变量：`VLA_LINGBOT_VAE_DECODER_FULL_SMOKE=1`
- 当前完整 decoder C++ 链路可执行：

```text
latent [W,H,T,48]
-> post_quant_conv
-> decoder.conv_in
-> decoder.mid_block
-> up_blocks.0 resnets -> temporal x2 -> spatial x2
-> up_blocks.1 resnets -> temporal x2 -> spatial x2
-> up_blocks.2 resnets -> spatial x2
-> up_blocks.3 resnets
-> decoder.norm_out / conv_out
-> unpatchify RGB
```

- 重要说明：temporal upsample 的 `2*C -> 2*T x C` 重排是根据权重形状和 decoder 对称结构做出的工程推断；还需要后续 Python reference parity 来确认是否完全贴合 diffusers `AutoencoderKLWan` 原版。

English:

- Per user request, completed the remaining decoder modules in one pass and then validated them together.
- Added a temporal conv executor that supports `C_in != C_out`:
  - `LingBotVaeCudaTimeConvWeightsIO`
  - `vae_time_conv_cuda_whdc_execute_io()`
- Added decoder spatial upsampler:
  - `vae_decoder_spatial_upsample_ggml_execute()`
  - currently uses `ggml_interpolate(..., GGML_SCALE_MODE_NEAREST)` for 2x spatial upsampling, followed by `upsampler.resample.1` 2D conv.
- Added decoder temporal upsampler:
  - `vae_decoder_temporal_upsample_execute()`
  - based on the weight shape `[2*C, C, 3, 1, 1]`, it splits the `2*C` temporal-conv output into two consecutive time steps, implementing T x2.
- Added decoder tail:
  - `vae_decoder_tail_ggml_execute()`
  - executes `decoder.norm_out -> silu -> decoder.conv_out`.
- Added unpatchify:
  - `vae_unpatchify_ps2_rgb()`
  - restores patchified `[W,H,T,12]` into RGB `[2W,2H,T,3]`.
- Added full decoder executor and smoke:
  - `vae_decoder_full_execute()`
  - `run_vae_decoder_full_smoke()`
  - env switch: `VLA_LINGBOT_VAE_DECODER_FULL_SMOKE=1`
- Current full decoder C++ path is executable:

```text
latent [W,H,T,48]
-> post_quant_conv
-> decoder.conv_in
-> decoder.mid_block
-> up_blocks.0 resnets -> temporal x2 -> spatial x2
-> up_blocks.1 resnets -> temporal x2 -> spatial x2
-> up_blocks.2 resnets -> spatial x2
-> up_blocks.3 resnets
-> decoder.norm_out / conv_out
-> unpatchify RGB
```

- Important note: the temporal upsample `2*C -> 2*T x C` rearrangement is an engineering inference from the tensor shape and decoder symmetry. It still needs Python reference parity against the original diffusers `AutoencoderKLWan`.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_DECODER_FULL_SMOKE=1
latent=[1,1,2,48]
patch_out=[8,8,8,12]
patch checksum=1525.88022
patch max=2.54167962
image=[16,16,8,3]
image checksum=1525.88022
image max=2.54167962

Regression:
VLA_LINGBOT_VAE_ENCODER_FULL_SMOKE=1 still passes with full VAE GGUF.
out=[1,1,5,96]
checksum=-2966.09299
max=15.7149105
```

## 2026-06-16 13:10:15 CST - VAE Python Reference Parity Round 1 / VAE Python Reference Parity 第一轮

中文：

- 按用户建议，先做整体 Python reference parity，再用二分思路定位问题。
- 为了加载原版 `AutoencoderKLWan`，在 SimplerEnv venv 中升级：

```text
diffusers: 0.30.1 -> 0.36.0
transformers: 4.51.3 -> 4.55.4
```

- Python 原版已能加载：

```text
diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan
```

- 整体验证结果：当前 C++ VAE smoke 路径不能通过 Python parity，原因不是单纯数值误差，而是结构/调用语义不一致。
- 关键发现：
  - 原版 VAE 使用 `WanRMS_norm`，语义是 channel RMSNorm：

```text
F.normalize(x, dim=channel) * sqrt(C) * gamma
```

    当前 C++ VAE helper 使用了 `ggml_group_norm(..., groups=32)`，这是错误的 parity 目标。
  - 原版 VAE 是 `is_residual=True` 结构：
    - encoder 使用 `WanResidualDownBlock`，包含 `avg_shortcut = AvgDown3D(...)`；
    - decoder 使用 `WanResidualUpBlock`，包含 `avg_shortcut = DupUp3D(...)`；
    - 当前 C++ smoke 只覆盖了主 ResNet/采样路径，漏掉 residual shortcut sampling 分支。
  - 原版 `WanAttentionBlock` 的 attention 是按每个时间帧内部的空间 tokens 做 attention：

```text
[B,C,T,H,W] -> [B*T,C,H,W] -> attention over H*W tokens
```

    当前 C++ mid attention smoke 在 WHDC 上把 `W*H*T` 都当成 token，会在 `H*W > 1` 或 `T > 1` 时偏离原版。
  - 原版 encoder parity 不能直接调用 `vae.encoder(x)` 做长序列对齐；真实路径是 `vae._encode()`，会按 chunk/cache 处理。
  - 原版 decoder 也不能直接调用 `vae.decoder(z)` 做长序列对齐；真实路径是 `vae._decode()/decode()`，逐 latent timestep 调 decoder，并用 `feat_cache` 管理 causal conv / residual upsample。
- Python reference checksum：

```text
vae._encode(raw video, frames=1) -> [1,96,1,1,1]
checksum=-725.0696456776932
max=15.910210609436035

vae._encode(raw video, frames=2) -> [1,96,1,1,1]
checksum=-727.4727365113795
max=15.952753067016602

vae._encode(raw video, frames=5) -> [1,96,2,1,1]
checksum=-1477.4044320359826
max=16.56582260131836

vae.decode(latent T=1) -> [1,3,1,16,16]
checksum=-22.99990927684121
max=0.22897066175937653

vae.decode(latent T=2) -> [1,3,5,16,16]
checksum=-220.76817390450742
max=0.28790757060050964

vae.decode(latent T=5) -> [1,3,17,16,16]
checksum=-903.5327487214236
max=0.3386855721473694
```

- 结论：
  - 当前 VAE C++ 链路是“可执行 smoke”，但不是 Python reference parity 实现。
  - 下一步必须先修 C++ 结构语义，而不是继续调小误差。
- 下一步修复顺序：
  1. 将 VAE norm 从 GroupNorm 改成 WanRMSNorm。
  2. 将 mid attention 改成 per-frame spatial attention。
  3. 补 `AvgDown3D` 和 `DupUp3D` residual shortcut，并改造 encoder/decoder block executor。
  4. 将 encoder parity 改为 `_encode()` 的 chunk/cache 路径。
  5. 将 decoder parity 改为 `_decode()/decode()` 的 per-latent-step streaming 路径。

English:

- Following the user's suggestion, started with whole-path Python reference parity and then localized issues with a bisection-style approach.
- To load the original `AutoencoderKLWan`, upgraded the SimplerEnv venv:

```text
diffusers: 0.30.1 -> 0.36.0
transformers: 4.51.3 -> 4.55.4
```

- The Python reference now loads:

```text
diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan
```

- Whole-path verification result: the current C++ VAE smoke path does not pass Python parity. This is not a small numerical drift; it is a structural/calling-semantics mismatch.
- Key findings:
  - The original VAE uses `WanRMS_norm`, i.e. channel RMSNorm:

```text
F.normalize(x, dim=channel) * sqrt(C) * gamma
```

    The current C++ VAE helper uses `ggml_group_norm(..., groups=32)`, which is the wrong parity target.
  - The original VAE is an `is_residual=True` structure:
    - encoder uses `WanResidualDownBlock`, including `avg_shortcut = AvgDown3D(...)`;
    - decoder uses `WanResidualUpBlock`, including `avg_shortcut = DupUp3D(...)`;
    - current C++ smokes cover only the main ResNet/sampling path and miss the residual shortcut sampling branch.
  - The original `WanAttentionBlock` attends over spatial tokens inside each time frame:

```text
[B,C,T,H,W] -> [B*T,C,H,W] -> attention over H*W tokens
```

    The current C++ mid-attention smoke treats `W*H*T` as one token axis, which diverges from the original when `H*W > 1` or `T > 1`.
  - Encoder parity should not call `vae.encoder(x)` directly for long sequences; the true path is `vae._encode()`, with chunk/cache handling.
  - Decoder parity should not call `vae.decoder(z)` directly for long sequences; the true path is `vae._decode()/decode()`, which calls the decoder one latent timestep at a time and manages `feat_cache` for causal conv / residual upsample.
- Python reference checksums:

```text
vae._encode(raw video, frames=1) -> [1,96,1,1,1]
checksum=-725.0696456776932
max=15.910210609436035

vae._encode(raw video, frames=2) -> [1,96,1,1,1]
checksum=-727.4727365113795
max=15.952753067016602

vae._encode(raw video, frames=5) -> [1,96,2,1,1]
checksum=-1477.4044320359826
max=16.56582260131836

vae.decode(latent T=1) -> [1,3,1,16,16]
checksum=-22.99990927684121
max=0.22897066175937653

vae.decode(latent T=2) -> [1,3,5,16,16]
checksum=-220.76817390450742
max=0.28790757060050964

vae.decode(latent T=5) -> [1,3,17,16,16]
checksum=-903.5327487214236
max=0.3386855721473694
```

- Conclusion:
  - The current VAE C++ path is executable as a smoke path, but it is not yet a Python-reference parity implementation.
  - The next step is to fix C++ structural semantics first, not tune small numeric differences.
- Next fix order:
  1. Replace VAE GroupNorm with WanRMSNorm.
  2. Change mid attention to per-frame spatial attention.
  3. Add `AvgDown3D` and `DupUp3D` residual shortcuts and update encoder/decoder block executors.
  4. Change encoder parity to the `_encode()` chunk/cache path.
  5. Change decoder parity to the `_decode()/decode()` per-latent-step streaming path.

## 2026-06-16 01:32:52 CST - Custom Streaming Causal Conv CUDA Kernel

中文：

- 开始加入 LingBot-VA 的第一块明确“difference”型自定义 CUDA kernel：streaming causal conv cache。
- 位置：
  - `src/kernels/lingbot/lingbot_flex_attn_cuda.h`
  - `src/kernels/lingbot/lingbot_flex_attn_cuda.cu`
  - `src/models/lingbot_va.cpp`
- 新增公开 CUDA API：

```text
lingbot_causal_conv1d_cache_f32(...)
lingbot_causal_conv1d_cache_cuda_smoke(...)
lingbot_causal_conv1d_cache_vae_smoke(...)
```

- 这个 kernel 的目标是服务 LingBot-VA/Wan VAE streaming 路径里的 temporal causal conv cache，而不是只依赖 ggml 拼接。
- 目前实现的是 F32 reference-grade kernel，先验证语义、cache 更新和布局；后续再继续做真实 VAE 接入、融合、BF16/量化版本。
- 已接入两个 smoke 开关：
  - `VLA_LINGBOT_CAUSAL_CONV_CUDA_SMOKE=1`
  - `VLA_LINGBOT_CAUSAL_CONV_VAE_SMOKE=1`

验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_CAUSAL_CONV_CUDA_SMOKE=1:
max_diff=2.98023224e-08
cache_diff=0

VLA_LINGBOT_CAUSAL_CONV_VAE_SMOKE=1:
T=4 Cin=32 Cout=32 K=3
max_diff=1.49011612e-08
cache_diff=0
```

说明：

- `timeout` 退出码 124 是因为 smoke 成功后 `vla-server` 进入监听状态，被计时器截停；不是 kernel 失败。
- 下一步把这个 kernel 对齐到 Python 原版 `WanVAEStreamingWrapper` / `WanCausalConv3d`，再接真实 VAE encoder/downsampler 的时序缓存路径。

English:

- Added the first explicit LingBot-VA "difference" CUDA kernel: streaming causal convolution with cache update.
- Files touched:
  - `src/kernels/lingbot/lingbot_flex_attn_cuda.h`
  - `src/kernels/lingbot/lingbot_flex_attn_cuda.cu`
  - `src/models/lingbot_va.cpp`
- New public CUDA API:

```text
lingbot_causal_conv1d_cache_f32(...)
lingbot_causal_conv1d_cache_cuda_smoke(...)
lingbot_causal_conv1d_cache_vae_smoke(...)
```

- The target is the temporal causal-conv cache path used by LingBot-VA/Wan VAE streaming, instead of relying only on ggml composition.
- The current kernel is an F32 reference-grade implementation for validating semantics, layout, and cache updates. Real VAE wiring, fusion, BF16, and quantized variants remain next steps.
- Added two smoke switches:
  - `VLA_LINGBOT_CAUSAL_CONV_CUDA_SMOKE=1`
  - `VLA_LINGBOT_CAUSAL_CONV_VAE_SMOKE=1`

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_CAUSAL_CONV_CUDA_SMOKE=1:
max_diff=2.98023224e-08
cache_diff=0

VLA_LINGBOT_CAUSAL_CONV_VAE_SMOKE=1:
T=4 Cin=32 Cout=32 K=3
max_diff=1.49011612e-08
cache_diff=0
```

Notes:

- Exit code 124 from `timeout` is expected because the smoke succeeds and then `vla-server` keeps listening until the timer stops it.
- Next step: align this kernel with the Python `WanVAEStreamingWrapper` / `WanCausalConv3d` path and then wire it into real VAE encoder/downsampler temporal-cache execution.

## 2026-06-16 01:38:46 CST - Real VAE Time-Conv Weights on Custom CUDA Kernel

中文：

- 将 custom streaming causal conv CUDA kernel 从 synthetic/VAE-like smoke 推进到真实 VAE 权重 smoke。
- 新增开关：

```text
VLA_LINGBOT_VAE_TIME_CONV_CUDA_SMOKE=1
```

- 该路径会读取 `VLA_LINGBOT_VAE_GGUF` 指向的 VAE encoder GGUF，并验证两处真实 encoder temporal downsampler：

```text
vae.encoder.down_blocks.1.downsampler.time_conv
vae.encoder.down_blocks.2.downsampler.time_conv
```

- 对应 Python 原版方向：
  - `WanVAEStreamingWrapper.encode_chunk()`
  - encoder 内部的 `WanCausalConv3d`
  - `feat_cache` / `feat_idx` streaming cache
- 目前 C++ 侧已经确认：
  - 真实 GGUF 权重可读；
  - `[Cout, Cin, K, 1, 1]` 权重可以按 temporal 1D causal conv 解释；
  - CUDA 输出和 C++ CPU reference 对齐；
  - cache 更新语义正确。

验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_TIME_CONV_CUDA_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1.downsampler.time_conv:
T=4 C=320 max_diff=1.49011612e-08 cache_diff=0

vae.encoder.down_blocks.2.downsampler.time_conv:
T=3 C=640 max_diff=1.3038516e-08 cache_diff=0
```

说明：

- 这是第一个已经使用 LingBot-VA 真实参数验证通过的 custom CUDA kernel，不再只是 toy/kernel-only smoke。
- 下一步需要把 temporal cache 抽成 VAE encoder runtime state，并接入真实 downsampler 执行路径。

English:

- Advanced the custom streaming causal-conv CUDA kernel from synthetic/VAE-like smoke to real VAE weights.
- New switch:

```text
VLA_LINGBOT_VAE_TIME_CONV_CUDA_SMOKE=1
```

- This path reads the VAE encoder GGUF from `VLA_LINGBOT_VAE_GGUF` and validates the two real encoder temporal downsamplers:

```text
vae.encoder.down_blocks.1.downsampler.time_conv
vae.encoder.down_blocks.2.downsampler.time_conv
```

- Corresponding Python path:
  - `WanVAEStreamingWrapper.encode_chunk()`
  - encoder-side `WanCausalConv3d`
  - `feat_cache` / `feat_idx` streaming cache
- C++ side confirmed:
  - real GGUF weights are readable;
  - `[Cout, Cin, K, 1, 1]` weights map cleanly to temporal 1D causal convolution;
  - CUDA output matches the C++ CPU reference;
  - cache update semantics are correct.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_TIME_CONV_CUDA_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1.downsampler.time_conv:
T=4 C=320 max_diff=1.49011612e-08 cache_diff=0

vae.encoder.down_blocks.2.downsampler.time_conv:
T=3 C=640 max_diff=1.3038516e-08 cache_diff=0
```

Notes:

- This is the first custom CUDA kernel validated with real LingBot-VA parameters, not just a toy/kernel-only smoke.
- Next step: promote temporal cache into a VAE encoder runtime state and wire it into the real downsampler execution path.

## 2026-06-16 01:41:15 CST - World Server Session Cache and Latent Echo Bridge

中文：

- 继续推进 server-side difference：增强 `lingbot-world-server` 的 latent/world-model 协议骨架。
- 文件：
  - `src/serving/lingbot-world-server.cpp`
- 新增轻量 session cache：
  - 按 `session_id` 记录 step 计数；
  - 记录最近一次输入 latent 的数量；
  - 记录最近一次输入 latent 的总字节数；
  - `reset(clear_cache=true)` 会清理对应 session cache。
- `StepRequest.return_world_latents=true` 时，当前先启用 latent echo bridge：
  - 将输入 latent tensor 原样放入 `LingBotResponse.output_latents`；
  - 这不是最终 world-latent 更新结果；
  - 目的是先验证 protobuf/ZMQ 的 latent 二进制传输链路，后面替换成 VAE/WorldModel 输出时客户端协议不用再重写。
- response `status` 现在包含 session 运行信息：

```text
step_with_latents_bridge session=42 step=1 latent_count=1 latent_bytes=96
```

验证：

```text
cmake --build build -j$(nproc)

./build/lingbot-world-server --bind tcp://127.0.0.1:6001 \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf

reset:
request_id=10 status=reset_clear_cache

step:
request_id=11
status=step_with_latents_bridge session=42 step=1 latent_count=1 latent_bytes=96
chunk_size=16
action_dim=30
action_values=480
output_latents=1
latent_echo_bytes=96
latent_echo_equal=true
```

说明：

- 这一步把 LingBot world/action 服务和标准 VLA action 服务区分开：它已经能显式承载 latent tensor、session cache 和未来 world latent 输出。
- 当前 action 仍走已有 `predict()` bridge；真实 VAE encoder / Wan world latent update 还没有接入 server step。

English:

- Advanced the server-side difference by strengthening the latent/world-model protocol skeleton in `lingbot-world-server`.
- File:
  - `src/serving/lingbot-world-server.cpp`
- Added a lightweight session cache:
  - tracks step count per `session_id`;
  - tracks the latest input latent count;
  - tracks the latest input latent byte size;
  - `reset(clear_cache=true)` clears the corresponding session cache.
- When `StepRequest.return_world_latents=true`, the server now enables a latent echo bridge:
  - input latent tensors are copied into `LingBotResponse.output_latents`;
  - this is not the final world-latent update result;
  - it validates the protobuf/ZMQ binary latent transport path before replacing the echo with VAE/WorldModel output.
- Response `status` now carries session runtime information:

```text
step_with_latents_bridge session=42 step=1 latent_count=1 latent_bytes=96
```

Verification:

```text
cmake --build build -j$(nproc)

./build/lingbot-world-server --bind tcp://127.0.0.1:6001 \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf

reset:
request_id=10 status=reset_clear_cache

step:
request_id=11
status=step_with_latents_bridge session=42 step=1 latent_count=1 latent_bytes=96
chunk_size=16
action_dim=30
action_values=480
output_latents=1
latent_echo_bytes=96
latent_echo_equal=true
```

Notes:

- This separates the LingBot world/action server from a standard VLA action-only server: it can now explicitly carry latent tensors, session cache state, and future world-latent outputs.
- The action path still uses the existing `predict()` bridge. Real VAE encoder and Wan world-latent update are not wired into `step` yet.

## 2026-06-16 01:47:10 CST - Device-Side VAE Temporal Cache Streaming Smoke

中文：

- 将 VAE temporal causal conv 的验证从“一次 kernel 调用”推进到“跨 chunk streaming cache”。
- 新增 `LingBotVaeCudaTemporalConvCache`：
  - device-side `past` / `next` cache；
  - 每个 chunk 调用 `lingbot_causal_conv1d_cache_f32(...)`；
  - step 后交换 cache；
  - 最后可读回 cache 验证。
- 新增开关：

```text
VLA_LINGBOT_VAE_TIME_CONV_STREAM_SMOKE=1
```

- 该 smoke 使用真实 VAE encoder GGUF 权重，模拟 Python 原版 `WanVAEStreamingWrapper` 的 `feat_cache` 语义：
  - 同一个 temporal conv cache 跨多个 chunk 保持；
  - streaming 输出与一次性完整序列 CPU reference 对齐；
  - 最终 cache 也与完整 reference 对齐。

验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_TIME_CONV_STREAM_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1.downsampler.time_conv:
C=320 chunks=2+3 max_diff=1.49011612e-08 cache_diff=0

vae.encoder.down_blocks.2.downsampler.time_conv:
C=640 chunks=1+2+2 max_diff=1.11758709e-08 cache_diff=0
```

说明：

- 这一步已经不是单算子 correctness，而是验证了跨 chunk streaming state 的生命周期。
- 后续接真实 VAE encoder 时，可以把这个 state 放进 LingBot world-server 的 session cache 或模型 runtime state 里。

English:

- Advanced VAE temporal causal-conv validation from a single kernel call to cross-chunk streaming cache execution.
- Added `LingBotVaeCudaTemporalConvCache`:
  - device-side `past` / `next` cache;
  - each chunk calls `lingbot_causal_conv1d_cache_f32(...)`;
  - cache is swapped after each step;
  - final cache can be copied back for validation.
- New switch:

```text
VLA_LINGBOT_VAE_TIME_CONV_STREAM_SMOKE=1
```

- This smoke uses real VAE encoder GGUF weights and models the Python `WanVAEStreamingWrapper` `feat_cache` semantics:
  - the same temporal-conv cache persists across chunks;
  - streaming output matches a full-sequence CPU reference;
  - final cache also matches the full reference.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_TIME_CONV_STREAM_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1.downsampler.time_conv:
C=320 chunks=2+3 max_diff=1.49011612e-08 cache_diff=0

vae.encoder.down_blocks.2.downsampler.time_conv:
C=640 chunks=1+2+2 max_diff=1.11758709e-08 cache_diff=0
```

Notes:

- This validates more than single-op correctness: it verifies the lifetime of streaming state across chunks.
- When wiring the real VAE encoder, this state can live in the LingBot world-server session cache or model runtime state.

## 2026-06-16 01:52:37 CST - Batched Spatial-Lane VAE Time-Conv CUDA Kernel

中文：

- 将 temporal causal conv CUDA kernel 从单条时间线扩展到 batched/spatial-lane 版本。
- 文件：
  - `src/kernels/lingbot/lingbot_flex_attn_cuda.h`
  - `src/kernels/lingbot/lingbot_flex_attn_cuda.cu`
  - `src/models/lingbot_va.cpp`
- 新增 CUDA API：

```text
lingbot_causal_conv1d_cache_f32_batched(...)
```

- 新增 C++ runtime smoke 结构：

```text
LingBotVaeCudaTemporalConvBatchedCache
```

- 语义：
  - 每个 lane 对应一个空间位置，例如 `B * H * W`；
  - 每个 lane 有独立 temporal cache；
  - 所有 lane 共享同一组 VAE `time_conv` 权重；
  - 支持跨 chunk streaming cache 更新。
- 新增开关：

```text
VLA_LINGBOT_VAE_TIME_CONV_BATCHED_STREAM_SMOKE=1
```

验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_TIME_CONV_BATCHED_STREAM_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1.downsampler.time_conv:
lanes=6 C=320 chunks=2+3 max_diff=1.49011612e-08 cache_diff=0

vae.encoder.down_blocks.2.downsampler.time_conv:
lanes=4 C=640 chunks=1+2+2 max_diff=1.11758709e-08 cache_diff=0
```

说明：

- 这一步比单 lane streaming smoke 更接近真实 VAE：真实 video latent 的每个空间位置都需要一条 temporal causal conv 轨迹。
- 下一步可以把真实 VAE downsampler 的 tensor layout 映射到 batched kernel 的 `[lane, T, C]` 输入格式。

English:

- Extended the temporal causal-conv CUDA kernel from a single timeline to a batched/spatial-lane version.
- Files:
  - `src/kernels/lingbot/lingbot_flex_attn_cuda.h`
  - `src/kernels/lingbot/lingbot_flex_attn_cuda.cu`
  - `src/models/lingbot_va.cpp`
- New CUDA API:

```text
lingbot_causal_conv1d_cache_f32_batched(...)
```

- New C++ runtime smoke state:

```text
LingBotVaeCudaTemporalConvBatchedCache
```

- Semantics:
  - each lane represents one spatial position, such as `B * H * W`;
  - each lane owns an independent temporal cache;
  - all lanes share the same VAE `time_conv` weights;
  - cross-chunk streaming cache updates are supported.
- New switch:

```text
VLA_LINGBOT_VAE_TIME_CONV_BATCHED_STREAM_SMOKE=1
```

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_TIME_CONV_BATCHED_STREAM_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1.downsampler.time_conv:
lanes=6 C=320 chunks=2+3 max_diff=1.49011612e-08 cache_diff=0

vae.encoder.down_blocks.2.downsampler.time_conv:
lanes=4 C=640 chunks=1+2+2 max_diff=1.11758709e-08 cache_diff=0
```

Notes:

- This is closer to the real VAE than the single-lane streaming smoke: every spatial position in a video latent needs its own temporal causal-conv trajectory.
- Next step: map the real VAE downsampler tensor layout into the batched kernel input format `[lane, T, C]`.

## 2026-06-16 11:16:34 CST - VAE WHDC Layout Bridge to Batched CUDA Time-Conv

中文：

- 补齐真实 VAE 接入前最容易出错的一步：ggml VAE tensor layout 到 CUDA lane-major layout 的映射。
- 新增 layout helper：

```text
vae_ggml_whdc_index(...)
vae_pack_whdc_to_lanes(...)
vae_unpack_lanes_to_whdc(...)
vae_write_lanes_chunk_to_whdc(...)
```

- 新增 smoke：

```text
VLA_LINGBOT_VAE_TIME_CONV_LAYOUT_SMOKE=1
```

- 验证路径：

```text
ggml-style WHDC memory
  -> pack to [lane, T, C]
  -> batched CUDA temporal causal conv with per-lane cache
  -> unpack back to WHDC
  -> compare against full-sequence CPU reference
```

- 这里的 WHDC 使用 ggml 的真实内存顺序：
  - `W` 是最快维；
  - `C` 是最慢维；
  - index = `w + W * (h + H * (t + T * c))`。

验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_TIME_CONV_LAYOUT_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1.downsampler.time_conv:
WHDC=[3,2,5,320] chunks=2+3 max_diff=1.49011612e-08 cache_diff=0

vae.encoder.down_blocks.2.downsampler.time_conv:
WHDC=[2,2,5,640] chunks=1+2+2 max_diff=1.11758709e-08 cache_diff=0
```

说明：

- 这一步确认了真实 VAE downsampler 中的 `[W,H,D,C]` tensor 可以稳定映射到 batched CUDA kernel 的 `[lane,T,C]` 输入格式。
- 下一步可以把该 layout bridge 接到 VAE downsampler 的正式执行骨架里，替换 smoke-only 调用。

English:

- Filled in the highest-risk bridge before real VAE integration: mapping ggml VAE tensor layout into CUDA lane-major layout.
- Added layout helpers:

```text
vae_ggml_whdc_index(...)
vae_pack_whdc_to_lanes(...)
vae_unpack_lanes_to_whdc(...)
vae_write_lanes_chunk_to_whdc(...)
```

- Added smoke:

```text
VLA_LINGBOT_VAE_TIME_CONV_LAYOUT_SMOKE=1
```

- Validation path:

```text
ggml-style WHDC memory
  -> pack to [lane, T, C]
  -> batched CUDA temporal causal conv with per-lane cache
  -> unpack back to WHDC
  -> compare against full-sequence CPU reference
```

- WHDC uses the real ggml memory order:
  - `W` is the fastest dimension;
  - `C` is the slowest dimension;
  - index = `w + W * (h + H * (t + T * c))`.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_TIME_CONV_LAYOUT_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1.downsampler.time_conv:
WHDC=[3,2,5,320] chunks=2+3 max_diff=1.49011612e-08 cache_diff=0

vae.encoder.down_blocks.2.downsampler.time_conv:
WHDC=[2,2,5,640] chunks=1+2+2 max_diff=1.11758709e-08 cache_diff=0
```

Notes:

- This confirms that real VAE downsampler `[W,H,D,C]` tensors can be mapped into the batched CUDA kernel input format `[lane,T,C]`.
- Next step: wire this layout bridge into the formal VAE downsampler executor instead of calling it only from smoke tests.

## 2026-06-16 11:21:35 CST - Reusable WHDC CUDA Time-Conv Executor Helper

中文：

- 将上一阶段的 WHDC layout smoke 重构为可复用执行 helper，减少 smoke-only 代码。
- 新增结构：

```text
LingBotVaeCudaTimeConvWeights
```

- 作用：
  - 从 VAE GGUF 读取真实 `time_conv.weight` / `time_conv.bias`；
  - 保留 host 权重用于 parity reference；
  - 上传并持有 device-side 权重 `d_w` / `d_b`；
  - 通过 RAII 风格析构释放 CUDA buffer。

- 新增正式执行 helper：

```text
vae_time_conv_cuda_whdc_execute(...)
```

- 作用：
  - 输入 ggml-style `[W,H,D,C]` host tensor；
  - 自动 pack 成 `[lane,T,C]` chunk；
  - 调用 batched CUDA temporal causal conv；
  - 使用 `LingBotVaeCudaTemporalConvBatchedCache` 维护跨 chunk cache；
  - 将输出写回 `[W,H,D,C]`。

- `VLA_LINGBOT_VAE_TIME_CONV_LAYOUT_SMOKE=1` 现在验证的是这个正式 helper，而不是重复写在 smoke 里的临时代码。

验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_TIME_CONV_LAYOUT_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1.downsampler.time_conv:
WHDC=[3,2,5,320] chunks=2+3 max_diff=1.49011612e-08 cache_diff=0

vae.encoder.down_blocks.2.downsampler.time_conv:
WHDC=[2,2,5,640] chunks=1+2+2 max_diff=1.11758709e-08 cache_diff=0
```

说明：

- 这一步把 VAE temporal downsampler 的 CUDA 路径从验证代码推进为可复用执行单元。
- 下一步可以在 VAE downsampler executor 中直接调用 `vae_time_conv_cuda_whdc_execute(...)`。

English:

- Refactored the previous WHDC layout smoke into a reusable executor helper, reducing smoke-only code.
- New structure:

```text
LingBotVaeCudaTimeConvWeights
```

- Responsibilities:
  - reads real `time_conv.weight` / `time_conv.bias` from the VAE GGUF;
  - keeps host weights for parity references;
  - uploads and owns device-side weights `d_w` / `d_b`;
  - releases CUDA buffers via RAII-style destruction.

- New formal executor helper:

```text
vae_time_conv_cuda_whdc_execute(...)
```

- Responsibilities:
  - accepts a ggml-style `[W,H,D,C]` host tensor;
  - packs each chunk to `[lane,T,C]`;
  - calls the batched CUDA temporal causal-conv kernel;
  - uses `LingBotVaeCudaTemporalConvBatchedCache` for cross-chunk cache;
  - writes output back to `[W,H,D,C]`.

- `VLA_LINGBOT_VAE_TIME_CONV_LAYOUT_SMOKE=1` now validates this formal helper instead of duplicated temporary smoke code.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_TIME_CONV_LAYOUT_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1.downsampler.time_conv:
WHDC=[3,2,5,320] chunks=2+3 max_diff=1.49011612e-08 cache_diff=0

vae.encoder.down_blocks.2.downsampler.time_conv:
WHDC=[2,2,5,640] chunks=1+2+2 max_diff=1.11758709e-08 cache_diff=0
```

Notes:

- This moves the VAE temporal downsampler CUDA path from validation-only code toward a reusable execution unit.
- The next step can call `vae_time_conv_cuda_whdc_execute(...)` directly from a VAE downsampler executor.

## 2026-06-16 11:26:33 CST - VAE Downsampler Executor Skeleton

中文：

- 将 VAE downsampler 从单独验证 time_conv 推进到正式的 two-stage executor smoke。
- 新增 helper：

```text
vae_spatial_downsample_ggml_execute(...)
```

- 新增 smoke：

```text
VLA_LINGBOT_VAE_DOWNSAMPLER_EXEC_SMOKE=1
```

- 当前执行路径：

```text
input WHDC
  -> ggml CPU spatial downsample: downsampler.resample.1
  -> WHDC output
  -> CUDA temporal causal conv: downsampler.time_conv
  -> WHDC output
```

- 验证范围：
  - `vae.encoder.down_blocks.1.downsampler`
  - `vae.encoder.down_blocks.2.downsampler`
- 注意：block0 只有 spatial downsample；block1/block2 有 spatial + temporal，因此本 smoke 重点覆盖 block1/block2。

验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_DOWNSAMPLER_EXEC_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1.downsampler:
input=[4,4,5,320]
chunks=2+3
spatial_checksum=-9.38012464
final_checksum=-35.4428392
max_diff=5.96046448e-08
cache_diff=0

vae.encoder.down_blocks.2.downsampler:
input=[2,2,5,640]
chunks=1+2+2
spatial_checksum=-4.72477473
final_checksum=-32.1068568
max_diff=2.98023224e-08
cache_diff=0
```

说明：

- `resample.1` 目前仍走 ggml CPU 路径；
- `time_conv` 已走自定义 CUDA streaming cache kernel；
- 这是 VAE encoder 真正接入前的混合执行骨架。
- 下一步可以把 VAE down block 的 ResNet + downsampler 按 block1/block2 接入这个 executor。

English:

- Advanced the VAE downsampler from standalone time-conv validation to a formal two-stage executor smoke.
- New helper:

```text
vae_spatial_downsample_ggml_execute(...)
```

- New smoke:

```text
VLA_LINGBOT_VAE_DOWNSAMPLER_EXEC_SMOKE=1
```

- Current execution path:

```text
input WHDC
  -> ggml CPU spatial downsample: downsampler.resample.1
  -> WHDC output
  -> CUDA temporal causal conv: downsampler.time_conv
  -> WHDC output
```

- Coverage:
  - `vae.encoder.down_blocks.1.downsampler`
  - `vae.encoder.down_blocks.2.downsampler`
- Note: block0 only has spatial downsampling; block1/block2 have spatial + temporal, so this smoke focuses on block1/block2.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_DOWNSAMPLER_EXEC_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1.downsampler:
input=[4,4,5,320]
chunks=2+3
spatial_checksum=-9.38012464
final_checksum=-35.4428392
max_diff=5.96046448e-08
cache_diff=0

vae.encoder.down_blocks.2.downsampler:
input=[2,2,5,640]
chunks=1+2+2
spatial_checksum=-4.72477473
final_checksum=-32.1068568
max_diff=2.98023224e-08
cache_diff=0
```

Notes:

- `resample.1` still runs on the ggml CPU path;
- `time_conv` now runs through the custom CUDA streaming-cache kernel;
- this is the mixed-execution skeleton needed before wiring the real VAE encoder.
- Next step: wire block1/block2 VAE down-block ResNets plus downsampler into this executor.

## 2026-06-16 11:31:52 CST - VAE Down Block 1/2 Executor Connected

中文：

- 按“直接连接”的要求，将 VAE encoder 的 block1/block2 从 ResNet 到 downsampler 串成完整 executor smoke。
- 新增 helper：

```text
vae_down_block_resnets_spatial_ggml_execute(...)
```

- 新增 smoke：

```text
VLA_LINGBOT_VAE_DOWN_BLOCK_EXEC_SMOKE=1
```

- 当前执行路径：

```text
input WHDC
  -> ResNet 0
       norm1 -> conv1 -> norm2 -> conv2
       + conv_shortcut for channel change
  -> ResNet 1
       norm1 -> conv1 -> norm2 -> conv2
  -> spatial downsample: downsampler.resample.1
  -> CUDA temporal causal conv: downsampler.time_conv
  -> output WHDC
```

- 覆盖的真实权重：
  - `vae.encoder.down_blocks.1`
    - input C=160, output C=320
    - includes `conv_shortcut`
  - `vae.encoder.down_blocks.2`
    - input C=320, output C=640
    - includes `conv_shortcut`

验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_DOWN_BLOCK_EXEC_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1:
input=[4,4,5,160]
outC=320
chunks=2+3
spatial_checksum=141.168652
final_checksum=-157.002885
max_diff=1.66893005e-06
cache_diff=0

vae.encoder.down_blocks.2:
input=[2,2,5,320]
outC=640
chunks=1+2+2
spatial_checksum=30.1212753
final_checksum=-77.9615336
max_diff=5.96046448e-07
cache_diff=0
```

说明：

- ResNet 与 spatial downsample 目前仍走 ggml CPU；
- temporal `time_conv` 走自定义 CUDA streaming cache kernel；
- 这一步已经把 block1/block2 的 VAE down block 结构接起来，下一步可以继续向完整 VAE encoder 串接：
  - block0 output -> block1 executor -> block2 executor -> later encoder tail / quant_conv。

English:

- Per the request to connect directly, wired VAE encoder block1/block2 from ResNets through the downsampler into a complete executor smoke.
- New helper:

```text
vae_down_block_resnets_spatial_ggml_execute(...)
```

- New smoke:

```text
VLA_LINGBOT_VAE_DOWN_BLOCK_EXEC_SMOKE=1
```

- Current execution path:

```text
input WHDC
  -> ResNet 0
       norm1 -> conv1 -> norm2 -> conv2
       + conv_shortcut for channel change
  -> ResNet 1
       norm1 -> conv1 -> norm2 -> conv2
  -> spatial downsample: downsampler.resample.1
  -> CUDA temporal causal conv: downsampler.time_conv
  -> output WHDC
```

- Real weights covered:
  - `vae.encoder.down_blocks.1`
    - input C=160, output C=320
    - includes `conv_shortcut`
  - `vae.encoder.down_blocks.2`
    - input C=320, output C=640
    - includes `conv_shortcut`

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_DOWN_BLOCK_EXEC_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

vae.encoder.down_blocks.1:
input=[4,4,5,160]
outC=320
chunks=2+3
spatial_checksum=141.168652
final_checksum=-157.002885
max_diff=1.66893005e-06
cache_diff=0

vae.encoder.down_blocks.2:
input=[2,2,5,320]
outC=640
chunks=1+2+2
spatial_checksum=30.1212753
final_checksum=-77.9615336
max_diff=5.96046448e-07
cache_diff=0
```

Notes:

- ResNets and spatial downsampling still run through ggml CPU;
- temporal `time_conv` runs through the custom CUDA streaming-cache kernel;
- this connects the VAE down-block structure for block1/block2. The next step is chaining the full VAE encoder:
  - block0 output -> block1 executor -> block2 executor -> later encoder tail / quant_conv.

## 2026-06-16 11:36:53 CST - VAE Encoder Down Path Connected

中文：

- 继续向完整 VAE encoder 推进，将下采样主干直接串起来：

```text
conv_in + down_blocks.0
  -> down_blocks.1 ResNets + spatial downsample + CUDA time_conv
  -> down_blocks.2 ResNets + spatial downsample + CUDA time_conv
```

- 新增 helper：

```text
vae_down_block0_ggml_execute(...)
vae_down_block_with_time_execute(...)
```

- 新增 smoke：

```text
VLA_LINGBOT_VAE_ENCODER_DOWN_PATH_SMOKE=1
```

- 当前执行特点：
  - block0 全部走 ggml CPU；
  - block1/block2 的 ResNet + spatial downsample 走 ggml CPU；
  - block1/block2 的 temporal `time_conv` 走自定义 CUDA streaming cache kernel；
  - block 之间真实传递 WHDC tensor。

验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_ENCODER_DOWN_PATH_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

input=[8,8,5,12]
block0=[4,4,5,160] checksum=-1141.40832
block1=[2,2,5,320] checksum=-223.803484
block2=[1,1,5,640] checksum=-41.7274002
max=3.21255159
```

说明：

- 这一步已经把 VAE encoder 的 down path 从独立 block smoke 推进到连续执行路径。
- 下一步需要补 encoder tail：
  - mid block ResNet；
  - mid block attention；
  - conv_out；
  - quant_conv。

English:

- Advanced toward the full VAE encoder by directly chaining the downsampling trunk:

```text
conv_in + down_blocks.0
  -> down_blocks.1 ResNets + spatial downsample + CUDA time_conv
  -> down_blocks.2 ResNets + spatial downsample + CUDA time_conv
```

- New helpers:

```text
vae_down_block0_ggml_execute(...)
vae_down_block_with_time_execute(...)
```

- New smoke:

```text
VLA_LINGBOT_VAE_ENCODER_DOWN_PATH_SMOKE=1
```

- Current execution properties:
  - block0 runs fully through ggml CPU;
  - block1/block2 ResNets + spatial downsampling run through ggml CPU;
  - block1/block2 temporal `time_conv` runs through the custom CUDA streaming-cache kernel;
  - real WHDC tensors are passed between blocks.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_VAE_ENCODER_DOWN_PATH_SMOKE=1
VLA_LINGBOT_VAE_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_vae_encoder_f32.gguf

input=[8,8,5,12]
block0=[4,4,5,160] checksum=-1141.40832
block1=[2,2,5,320] checksum=-223.803484
block2=[1,1,5,640] checksum=-41.7274002
max=3.21255159
```

Notes:

- This moves the VAE encoder down path from isolated block smokes to continuous execution.
- Next tail work:
  - mid-block ResNets;
  - mid-block attention;
  - conv_out;
  - quant_conv.

## 2026-06-15 23:21:42 CST - Real Wan Q/K/V to Flex CUDA Bridge / 真实 Wan Q/K/V 接入 Flex CUDA 桥接

中文：

- 按照当前推进目标，把真实 WanTransformer block 里的 self-attention Q/K/V、`LingBotKVCache` 和 `LingBotBlockSparseTable` 接到同一条可验证链路里。
- 扩展 `LingBotAttentionTrace`，现在可以暴露 RoPE 后的 head-layout 张量：

```text
qh / kh / vh
```

- 新增 host 侧张量布局转换：

```text
hds_tensor_to_shd()
```

  将 ggml 的 `[head_dim, heads, seq]` 转成 CUDA kernel 使用的 `[seq, heads, head_dim]`。

- 新增 `VLA_LINGBOT_QKV_CUDA_SMOKE=1`：
  - 加载 `wvm.blk.0.self_attn.*` 真实权重。
  - 用真实 block 的 Q/K/V projection + norm + RoPE 生成 Q/K/V。
  - 把真实 K/V 写入 `LingBotKVCache`。
  - 用 `LingBotBlockSparseTable + token_mask` 调用 `lingbot_flex_attn_f32_masked()`。
  - 与 CPU dense masked attention reference 做数值对齐。

验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_QKV_CUDA_SMOKE=1 ./build/vla-server \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
real Wan block Q/K/V flex CUDA smoke ok:
seq=27
heads=24
head_dim=128
table_blocks=15
max_row=4
max_diff=1.60932541e-06
checksum=-248.36446

KV cache:
total=27
used=27
pred=0
```

说明：

- 这一步证明了真实 Wan block 的 Q/K/V 已经可以进入 CUDA block-sparse attention，并且 block table / token mask / KV cache pool 三者可以协同工作。
- 这仍然是桥接 smoke，不是最终 runtime forward 替换。完整 forward 里要把 dense ggml attention 替换为 CUDA block-sparse executor，还需要把混合 video/action 序列的正式 RoPE position ids 和 streaming KV 生命周期接入。

English:

- Connected real self-attention Q/K/V from a WanTransformer block, `LingBotKVCache`, and `LingBotBlockSparseTable` into one verifiable bridge path.
- Extended `LingBotAttentionTrace` to expose post-RoPE head-layout tensors:

```text
qh / kh / vh
```

- Added host-side layout conversion:

```text
hds_tensor_to_shd()
```

  This converts ggml `[head_dim, heads, seq]` into the CUDA kernel layout `[seq, heads, head_dim]`.

- Added `VLA_LINGBOT_QKV_CUDA_SMOKE=1`:
  - loads real `wvm.blk.0.self_attn.*` weights
  - builds real Q/K/V through projection + norm + RoPE
  - writes real K/V into `LingBotKVCache`
  - calls `lingbot_flex_attn_f32_masked()` with `LingBotBlockSparseTable + token_mask`
  - compares CUDA output against a CPU dense masked attention reference

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_QKV_CUDA_SMOKE=1 ./build/vla-server \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
real Wan block Q/K/V flex CUDA smoke ok:
seq=27
heads=24
head_dim=128
table_blocks=15
max_row=4
max_diff=1.60932541e-06
checksum=-248.36446

KV cache:
total=27
used=27
pred=0
```

Notes:

- This proves that real Wan block Q/K/V can drive the CUDA block-sparse attention path, with block table, token mask, and KV cache pool all participating.
- This is still a bridge smoke, not the final runtime forward replacement. The full forward still needs dense ggml attention replaced by the CUDA block-sparse executor, plus official mixed video/action RoPE position ids and streaming KV lifetime management.

## 2026-06-15 23:33:25 CST - CUDA Self-Attention Block Executor Smoke / CUDA Self-Attention Block 执行器 Smoke

中文：

- 在上一轮“真实 Q/K/V -> CUDA flex attention”的桥接基础上，继续向正式 block executor 推进。
- 新增可复用 CUDA attention helper：

```text
run_lingbot_cuda_attention()
real_qkv_to_cuda_context()
shd_to_hidden_seq()
```

- 新增 `exec_block_one_cuda_self_attn()`：
  - 第一段 ggml graph 计算 block 内 `n1 -> real Q/K/V -> RoPE Q/K/V`。
  - 中间用 `LingBotKVCache + LingBotBlockSparseTable + token_mask` 调用 CUDA block-sparse self-attention。
  - 将 CUDA 输出 `[seq, heads, head_dim]` 转回 `[hidden, seq]`。
  - 第二段 ggml graph 继续执行 self-attn `o` projection、gate residual、cross-attention 和 FFN。

- 新增 `VLA_LINGBOT_CUDA_BLOCK_SMOKE=1`，用于验证“一个真实 block 的 self-attn 已经由 CUDA flex path 接管，block 剩余部分继续用 ggml 执行”。

验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_CUDA_BLOCK_SMOKE=1 ./build/vla-server \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
CUDA self-attn block smoke ok:
seq=27
hidden=3072
table_blocks=15
max_diff_vs_dense=3.26955426
dense_checksum=2.31445353
cuda_checksum=200.155173
```

说明：

- `max_diff_vs_dense` 较大是预期内的，因为 dense block 使用全 attention，而 CUDA block 使用 LingBot flex/block-sparse mask。这个 smoke 的目标不是 dense parity，而是验证 CUDA self-attn 可以被嵌入真实 block 执行链路。
- 底层 Q/K/V CUDA attention 回归仍然通过：

```text
VLA_LINGBOT_QKV_CUDA_SMOKE=1:
max_diff=1.60932541e-06
checksum=-248.36446
```

下一步：

- 将 `exec_block_one_cuda_self_attn()` 从 smoke-only 推进到 streaming executor 的可选执行路径。
- 构造正式 mixed video/action 序列的 RoPE position ids，替代当前 smoke-only 的 `smoke_grid_spec(seq, false)`。
- 之后再考虑 cross-attention 和 multi-block/window 级别的 CUDA/flex 执行策略。

English:

- Continued from the real Q/K/V -> CUDA flex attention bridge toward a formal block executor.
- Added reusable CUDA attention helpers:

```text
run_lingbot_cuda_attention()
real_qkv_to_cuda_context()
shd_to_hidden_seq()
```

- Added `exec_block_one_cuda_self_attn()`:
  - first ggml graph computes `n1 -> real Q/K/V -> post-RoPE Q/K/V`
  - CUDA block-sparse self-attention runs through `LingBotKVCache + LingBotBlockSparseTable + token_mask`
  - CUDA output `[seq, heads, head_dim]` is converted back to `[hidden, seq]`
  - second ggml graph continues self-attn `o` projection, gated residual, cross-attention, and FFN

- Added `VLA_LINGBOT_CUDA_BLOCK_SMOKE=1`, validating that one real block can use the CUDA flex path for self-attention while the remaining block operations continue through ggml.

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_CUDA_BLOCK_SMOKE=1 ./build/vla-server \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
CUDA self-attn block smoke ok:
seq=27
hidden=3072
table_blocks=15
max_diff_vs_dense=3.26955426
dense_checksum=2.31445353
cuda_checksum=200.155173
```

Notes:

- The large `max_diff_vs_dense` is expected because the dense block uses full attention while the CUDA block uses the LingBot flex/block-sparse mask. This smoke validates execution-path integration, not dense parity.
- The lower-level Q/K/V CUDA attention regression still passes:

```text
VLA_LINGBOT_QKV_CUDA_SMOKE=1:
max_diff=1.60932541e-06
checksum=-248.36446
```

Next:

- Promote `exec_block_one_cuda_self_attn()` from smoke-only into an optional streaming executor path.
- Build official mixed video/action RoPE position ids instead of the current smoke-only `smoke_grid_spec(seq, false)`.
- Then evaluate CUDA/flex handling for cross-attention and multi-block/window execution.

## 2026-06-15 23:47:48 CST - Optional CUDA Self-Attention Streaming Path / 可选 CUDA Self-Attention Streaming 路径

中文：

- 将 `exec_block_one_cuda_self_attn()` 从 smoke-only 推进到 `exec_forward_one_streaming()` 的可选执行路径。
- 新增环境变量：

```text
VLA_LINGBOT_CUDA_SELF_ATTN_STREAM=1
```

- 默认路径保持不变，仍走原来的 ggml dense block。只有显式设置该环境变量时，streaming forward 的 block self-attention 才会切到 CUDA flex/block-sparse path。
- 新增统一 helper：

```text
build_smoke_flex_meta()
```

  用于复用当前 smoke 的 LingBot flex mask 配置，避免多处重复写 latent/action meta。

- 当前 streaming CUDA path 仍使用 smoke flex meta，因此要求序列长度为 27。若 seq 不匹配，会直接报错提示需要设置：

```text
VLA_LINGBOT_FLOW_LOOP_LATENT_SEQ=27
VLA_LINGBOT_FLOW_LOOP_ACTION_SEQ=27
```

验证：

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_FLOW_LOOP_SMOKE=1 \
VLA_LINGBOT_CUDA_SELF_ATTN_STREAM=1 \
VLA_LINGBOT_FLOW_LOOP_BLOCKS=1 \
VLA_LINGBOT_FLOW_LOOP_VIDEO_STEPS=1 \
VLA_LINGBOT_FLOW_LOOP_ACTION_STEPS=1 \
VLA_LINGBOT_FLOW_LOOP_LATENT_SEQ=27 \
VLA_LINGBOT_FLOW_LOOP_ACTION_SEQ=27 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
flow-loop smoke video ok:
forward_steps=2
updates=1
blocks=1
terminal_t=0
checksum=-97.9486761
max_abs=1.57970333

flow-loop smoke action ok:
forward_steps=2
updates=1
blocks=1
terminal_t=0
checksum=-6.44306976
max_abs=0.218735814
```

错误保护验证：

```text
VLA_LINGBOT_CUDA_SELF_ATTN_STREAM=1 with default seq=2:
requires seq=27, got seq=2
```

说明：

- 这一步说明 CUDA flex self-attention 已经不只是孤立 block smoke，而是可以进入 flow-loop streaming forward。
- 当前仍是 smoke flex meta；正式路径还需要接入原版 LingBot 混合 video/action token 序列与对应 RoPE position ids。

English:

- Promoted `exec_block_one_cuda_self_attn()` from smoke-only into the optional `exec_forward_one_streaming()` path.
- Added environment variable:

```text
VLA_LINGBOT_CUDA_SELF_ATTN_STREAM=1
```

- The default path remains unchanged and still uses the original ggml dense block. The CUDA flex/block-sparse self-attention path is enabled only when this environment variable is explicitly set.
- Added shared helper:

```text
build_smoke_flex_meta()
```

  This reuses the current smoke LingBot flex mask configuration and removes repeated latent/action meta construction.

- The current streaming CUDA path still uses the smoke flex meta, so it requires sequence length 27. If the sequence length does not match, it reports a direct error and suggests:

```text
VLA_LINGBOT_FLOW_LOOP_LATENT_SEQ=27
VLA_LINGBOT_FLOW_LOOP_ACTION_SEQ=27
```

Verification:

```text
cmake --build build -j$(nproc)

VLA_LINGBOT_FLOW_LOOP_SMOKE=1 \
VLA_LINGBOT_CUDA_SELF_ATTN_STREAM=1 \
VLA_LINGBOT_FLOW_LOOP_BLOCKS=1 \
VLA_LINGBOT_FLOW_LOOP_VIDEO_STEPS=1 \
VLA_LINGBOT_FLOW_LOOP_ACTION_STEPS=1 \
VLA_LINGBOT_FLOW_LOOP_LATENT_SEQ=27 \
VLA_LINGBOT_FLOW_LOOP_ACTION_SEQ=27 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
flow-loop smoke video ok:
forward_steps=2
updates=1
blocks=1
terminal_t=0
checksum=-97.9486761
max_abs=1.57970333

flow-loop smoke action ok:
forward_steps=2
updates=1
blocks=1
terminal_t=0
checksum=-6.44306976
max_abs=0.218735814
```

Guard check:

```text
VLA_LINGBOT_CUDA_SELF_ATTN_STREAM=1 with default seq=2:
requires seq=27, got seq=2
```

Notes:

- CUDA flex self-attention is now able to participate in flow-loop streaming forward, not just isolated block smoke.
- This still uses the smoke flex meta. The formal path still needs LingBot's original mixed video/action token sequence and matching RoPE position ids.

## 2026-06-15 23:55:24 CST - Mixed Video/Action RoPE Order / 混合 Video/Action RoPE 顺序

中文：

- 对照 LingBot Python 原版 `wan_va/modules/model.py`：

```text
hidden_states = cat([
  latent_hidden_states,
  condition_latent_hidden_states,
  action_hidden_states,
  condition_action_hidden_states
])

full_grid_id = cat([latent_grid_id] * 2 + [action_grid_id] * 2)
```

- 在 C++ 中新增 mixed RoPE helper：

```text
build_grid_ids_from_spec()
build_lingbot_rope_from_grid_ids()
build_smoke_mixed_rope()
run_mixed_rope_smoke()
```

- 当前 smoke mixed order 对齐原版训练路径：

```text
[latent_noisy, latent_clean, action_noisy, action_clean, padding]
```

  对应 RoPE grid 顺序：

```text
[latent_grid, latent_grid, action_grid, action_grid, pad]
```

- CUDA streaming 路径现在会在 `VLA_LINGBOT_CUDA_SELF_ATTN_STREAM=1` 时覆盖为 mixed RoPE，而不是之前的临时 linear latent-only RoPE。

验证 mixed RoPE：

```text
VLA_LINGBOT_MIXED_ROPE_SMOKE=1 ./build/vla-server \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
mixed rope smoke ok:
tokens=27
pairs=64
diff_vs_linear=1.99996167
pad_deviation=0
checksum_cos=1687.56117
checksum_sin=41.8716942
```

验证 mixed RoPE + CUDA streaming flow-loop：

```text
VLA_LINGBOT_FLOW_LOOP_SMOKE=1 \
VLA_LINGBOT_CUDA_SELF_ATTN_STREAM=1 \
VLA_LINGBOT_FLOW_LOOP_BLOCKS=1 \
VLA_LINGBOT_FLOW_LOOP_VIDEO_STEPS=1 \
VLA_LINGBOT_FLOW_LOOP_ACTION_STEPS=1 \
VLA_LINGBOT_FLOW_LOOP_LATENT_SEQ=27 \
VLA_LINGBOT_FLOW_LOOP_ACTION_SEQ=27 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
CUDA self-attn streaming uses mixed LingBot RoPE order [latent, latent, action, action, pad], seq=27

flow-loop smoke video ok:
checksum=-98.1044218
max_abs=1.57587051

flow-loop smoke action ok:
checksum=-6.43549839
max_abs=0.218735814
```

说明：

- 这一步把 RoPE 顺序从临时 smoke-only 方案推进到贴近 LingBot 原版 mixed video/action sequence 的方案。
- 当前仍然只是 smoke 形状：latent tokens=8, action tokens=4, padding=3。后续要把这个 helper 泛化成运行时从真实 latent/action shape 自动生成。

English:

- Matched the LingBot Python reference in `wan_va/modules/model.py`:

```text
hidden_states = cat([
  latent_hidden_states,
  condition_latent_hidden_states,
  action_hidden_states,
  condition_action_hidden_states
])

full_grid_id = cat([latent_grid_id] * 2 + [action_grid_id] * 2)
```

- Added C++ mixed RoPE helpers:

```text
build_grid_ids_from_spec()
build_lingbot_rope_from_grid_ids()
build_smoke_mixed_rope()
run_mixed_rope_smoke()
```

- Current smoke mixed order now follows the original training path:

```text
[latent_noisy, latent_clean, action_noisy, action_clean, padding]
```

  With matching RoPE grid order:

```text
[latent_grid, latent_grid, action_grid, action_grid, pad]
```

- The CUDA streaming path now overrides the previous temporary linear latent-only RoPE with mixed RoPE when `VLA_LINGBOT_CUDA_SELF_ATTN_STREAM=1` is set.

Mixed RoPE verification:

```text
VLA_LINGBOT_MIXED_ROPE_SMOKE=1 ./build/vla-server \
  /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
mixed rope smoke ok:
tokens=27
pairs=64
diff_vs_linear=1.99996167
pad_deviation=0
checksum_cos=1687.56117
checksum_sin=41.8716942
```

Mixed RoPE + CUDA streaming flow-loop verification:

```text
VLA_LINGBOT_FLOW_LOOP_SMOKE=1 \
VLA_LINGBOT_CUDA_SELF_ATTN_STREAM=1 \
VLA_LINGBOT_FLOW_LOOP_BLOCKS=1 \
VLA_LINGBOT_FLOW_LOOP_VIDEO_STEPS=1 \
VLA_LINGBOT_FLOW_LOOP_ACTION_STEPS=1 \
VLA_LINGBOT_FLOW_LOOP_LATENT_SEQ=27 \
VLA_LINGBOT_FLOW_LOOP_ACTION_SEQ=27 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
CUDA self-attn streaming uses mixed LingBot RoPE order [latent, latent, action, action, pad], seq=27

flow-loop smoke video ok:
checksum=-98.1044218
max_abs=1.57587051

flow-loop smoke action ok:
checksum=-6.43549839
max_abs=0.218735814
```

Notes:

- This moves RoPE ordering from a temporary smoke-only linear layout toward LingBot's original mixed video/action sequence.
- This is still a smoke shape: latent tokens=8, action tokens=4, padding=3. Next step is generalizing this helper to derive the sequence from real latent/action runtime shapes.

## 2026-06-16 00:06:03 CST - WanTransformer Mixed Forward Executor / WanTransformer 混合 Forward 执行器

中文：

- 按照“放大步子”的要求，新增了更接近完整 `WanTransformer3DModel.forward_train()` 的 mixed forward executor smoke。
- 新增核心结构：

```text
LingBotMixedForwardState
```

  保存：

```text
latent_seq / action_seq / padded_seq / total_seq
mixed hidden_states
mixed t_hidden
mixed timestep_proj
mixed RoPE
flex meta / self block table / token mask
```

- 新增执行构造：

```text
build_mixed_forward_state()
exec_mixed_output_heads()
run_wan_mixed_forward_smoke()
```

- 现在 C++ mixed forward 已经覆盖原版 WanTransformer 的核心 forward 结构：

```text
latent_noisy embedding
latent_clean embedding
action_noisy embedding
action_clean embedding
concat -> [latent_noisy, latent_clean, action_noisy, action_clean, pad]
mixed RoPE
flex mask / block table
N x WanTransformer block
split latent/action
latent output head / action output head
```

- 支持两种 self-attention 模式：

```text
ggml dense self-attn
CUDA flex/block-sparse self-attn
```

验证 1：dense mixed forward：

```text
VLA_LINGBOT_MIXED_FORWARD_SMOKE=1 \
VLA_LINGBOT_MIXED_BLOCKS=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
Wan mixed forward smoke ok:
blocks=1
mode=ggml-dense
latent_tokens=[192,8]
action_tokens=[30,4]
total_seq=128
pad=104
latent_checksum=-8.15840236
latent_max=1.27248931
action_checksum=0.589102721
action_max=0.291250408
```

验证 2：CUDA flex self-attn mixed forward：

```text
VLA_LINGBOT_MIXED_FORWARD_SMOKE=1 \
VLA_LINGBOT_MIXED_CUDA_SELF_ATTN=1 \
VLA_LINGBOT_MIXED_BLOCKS=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
Wan mixed forward smoke ok:
blocks=1
mode=cuda-flex-self-attn
latent_tokens=[192,8]
action_tokens=[30,4]
total_seq=128
pad=104
latent_checksum=-27.9108505
latent_max=1.42296028
action_checksum=-0.115073762
action_max=0.130424768
```

验证 3：CUDA flex self-attn multi-block：

```text
VLA_LINGBOT_MIXED_FORWARD_SMOKE=1 \
VLA_LINGBOT_MIXED_CUDA_SELF_ATTN=1 \
VLA_LINGBOT_MIXED_BLOCKS=2 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
Wan mixed forward block 1/2 complete
Wan mixed forward block 2/2 complete

Wan mixed forward smoke ok:
blocks=2
mode=cuda-flex-self-attn
latent_checksum=-9.69070096
latent_max=1.1171937
action_checksum=-0.121305337
action_max=0.140111506
```

当前状态：

- `WanTransformer3DModel` 的 C++ 端已经从“单分支 smoke / block smoke”推进到“mixed forward executor smoke”。
- 这已经覆盖 transformer 主体的 forward 结构，但仍然还不是最终 `vla-server` runtime predict：
  - 输入仍是 synthetic latent/action tensor。
  - UMT5 text encoder / VAE encoder 还没有接入。
  - output 后的 unpatchify/action postprocess 还没有串入正式接口。
  - 30 block 的显存/量化/streaming window 策略还需要继续做。

English:

- Following the request to move faster, added a mixed forward executor smoke much closer to the original `WanTransformer3DModel.forward_train()`.
- Added core state:

```text
LingBotMixedForwardState
```

  It stores:

```text
latent_seq / action_seq / padded_seq / total_seq
mixed hidden_states
mixed t_hidden
mixed timestep_proj
mixed RoPE
flex meta / self block table / token mask
```

- Added execution builders:

```text
build_mixed_forward_state()
exec_mixed_output_heads()
run_wan_mixed_forward_smoke()
```

- The C++ mixed forward now covers the core original WanTransformer forward structure:

```text
latent_noisy embedding
latent_clean embedding
action_noisy embedding
action_clean embedding
concat -> [latent_noisy, latent_clean, action_noisy, action_clean, pad]
mixed RoPE
flex mask / block table
N x WanTransformer block
split latent/action
latent output head / action output head
```

- It supports two self-attention modes:

```text
ggml dense self-attn
CUDA flex/block-sparse self-attn
```

Verification 1: dense mixed forward:

```text
VLA_LINGBOT_MIXED_FORWARD_SMOKE=1 \
VLA_LINGBOT_MIXED_BLOCKS=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
Wan mixed forward smoke ok:
blocks=1
mode=ggml-dense
latent_tokens=[192,8]
action_tokens=[30,4]
total_seq=128
pad=104
latent_checksum=-8.15840236
latent_max=1.27248931
action_checksum=0.589102721
action_max=0.291250408
```

Verification 2: CUDA flex self-attn mixed forward:

```text
VLA_LINGBOT_MIXED_FORWARD_SMOKE=1 \
VLA_LINGBOT_MIXED_CUDA_SELF_ATTN=1 \
VLA_LINGBOT_MIXED_BLOCKS=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
Wan mixed forward smoke ok:
blocks=1
mode=cuda-flex-self-attn
latent_tokens=[192,8]
action_tokens=[30,4]
total_seq=128
pad=104
latent_checksum=-27.9108505
latent_max=1.42296028
action_checksum=-0.115073762
action_max=0.130424768
```

Verification 3: CUDA flex self-attn multi-block:

```text
VLA_LINGBOT_MIXED_FORWARD_SMOKE=1 \
VLA_LINGBOT_MIXED_CUDA_SELF_ATTN=1 \
VLA_LINGBOT_MIXED_BLOCKS=2 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
Wan mixed forward block 1/2 complete
Wan mixed forward block 2/2 complete

Wan mixed forward smoke ok:
blocks=2
mode=cuda-flex-self-attn
latent_checksum=-9.69070096
latent_max=1.1171937
action_checksum=-0.121305337
action_max=0.140111506
```

Current status:

- The C++ `WanTransformer3DModel` path has moved from single-branch/block smoke to a mixed forward executor smoke.
- This now covers the main transformer forward structure, but is still not the final `vla-server` runtime predict:
  - inputs are still synthetic latent/action tensors
  - UMT5 text encoder / VAE encoder are not wired in
  - output unpatchify/action postprocess are not wired into the runtime interface
  - 30-block memory/quantization/streaming-window strategy still needs follow-up work

## 2026-06-16 00:14:56 CST - Mixed Forward Windowing and Detokenized Outputs / Mixed Forward 窗口加载与输出张量还原

中文：

- 继续完善 `WanTransformer3DModel` mixed forward executor。
- `run_wan_mixed_forward_smoke()` 现在支持 block window/streaming 加载，不再必须一次性加载所有 block：

```text
VLA_LINGBOT_BLOCK_WINDOW=1
VLA_LINGBOT_BLOCK_WINDOW=auto
VLA_LINGBOT_BLOCK_BUDGET_MB=<MiB>
```

- mixed forward 输出现在不只停留在 token 形态，还会还原成张量：

```text
latent tokens -> projected_latent_tokens_to_tensor()
action tokens -> action_tokens_to_tensor()
```

- 这一步把输出路径从：

```text
latent/action output head -> tokens
```

推进到：

```text
latent/action output head -> tokens -> latent/action tensors
```

验证 1：dense mixed forward + tensor output：

```text
VLA_LINGBOT_MIXED_FORWARD_SMOKE=1 \
VLA_LINGBOT_MIXED_BLOCKS=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
blocks=1
mode=ggml-dense
window=1
latent_tokens=[192,8]
action_tokens=[30,4]
total_seq=128
pad=104
latent_tensor=[1,48,2,4,4]
latent_tensor_checksum=-8.15840236
latent_tensor_max=1.27248931
action_tensor=[1,30,2,2,1]
action_tensor_checksum=0.589102721
action_tensor_max=0.291250408
```

验证 2：CUDA flex self-attn + window=1：

```text
VLA_LINGBOT_MIXED_FORWARD_SMOKE=1 \
VLA_LINGBOT_MIXED_CUDA_SELF_ATTN=1 \
VLA_LINGBOT_MIXED_BLOCKS=2 \
VLA_LINGBOT_BLOCK_WINDOW=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
Wan mixed forward block 1/2 complete (CUDA flex self-attn, window=1)
Wan mixed forward block 2/2 complete (CUDA flex self-attn, window=1)

latent_tensor=[1,48,2,4,4]
latent_tensor_checksum=-9.69070096
latent_tensor_max=1.1171937
action_tensor=[1,30,2,2,1]
action_tensor_checksum=-0.121305337
action_tensor_max=0.140111506
```

验证 3：CUDA flex self-attn + auto window：

```text
VLA_LINGBOT_MIXED_FORWARD_SMOKE=1 \
VLA_LINGBOT_MIXED_CUDA_SELF_ATTN=1 \
VLA_LINGBOT_MIXED_BLOCKS=2 \
VLA_LINGBOT_BLOCK_WINDOW=auto \
VLA_LINGBOT_BLOCK_BUDGET_MB=4096 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
auto block window=2
Wan mixed forward block 1/2 complete (CUDA flex self-attn, window=2)
Wan mixed forward block 2/2 complete (CUDA flex self-attn, window=2)

latent_tensor_checksum=-9.69070096
action_tensor_checksum=-0.121305337
```

当前状态更新：

- mixed forward 已经具备：
  - 原版顺序的 mixed sequence
  - mixed RoPE
  - flex mask/table
  - dense 或 CUDA flex self-attn block loop
  - block window streaming 加载
  - latent/action output heads
  - latent/action tensor detokenize
- 下一步应把 synthetic latent/action tensor 替换成 runtime 输入，并把 flow scheduler step 接到 mixed forward 输出张量上。

English:

- Continued completing the `WanTransformer3DModel` mixed forward executor.
- `run_wan_mixed_forward_smoke()` now supports block window/streaming loading instead of always loading all blocks at once:

```text
VLA_LINGBOT_BLOCK_WINDOW=1
VLA_LINGBOT_BLOCK_WINDOW=auto
VLA_LINGBOT_BLOCK_BUDGET_MB=<MiB>
```

- Mixed forward outputs are now detokenized back into tensors:

```text
latent tokens -> projected_latent_tokens_to_tensor()
action tokens -> action_tokens_to_tensor()
```

- The output path has advanced from:

```text
latent/action output head -> tokens
```

to:

```text
latent/action output head -> tokens -> latent/action tensors
```

Verification 1: dense mixed forward + tensor output:

```text
VLA_LINGBOT_MIXED_FORWARD_SMOKE=1 \
VLA_LINGBOT_MIXED_BLOCKS=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
blocks=1
mode=ggml-dense
window=1
latent_tokens=[192,8]
action_tokens=[30,4]
total_seq=128
pad=104
latent_tensor=[1,48,2,4,4]
latent_tensor_checksum=-8.15840236
latent_tensor_max=1.27248931
action_tensor=[1,30,2,2,1]
action_tensor_checksum=0.589102721
action_tensor_max=0.291250408
```

Verification 2: CUDA flex self-attn + window=1:

```text
VLA_LINGBOT_MIXED_FORWARD_SMOKE=1 \
VLA_LINGBOT_MIXED_CUDA_SELF_ATTN=1 \
VLA_LINGBOT_MIXED_BLOCKS=2 \
VLA_LINGBOT_BLOCK_WINDOW=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
Wan mixed forward block 1/2 complete (CUDA flex self-attn, window=1)
Wan mixed forward block 2/2 complete (CUDA flex self-attn, window=1)

latent_tensor=[1,48,2,4,4]
latent_tensor_checksum=-9.69070096
latent_tensor_max=1.1171937
action_tensor=[1,30,2,2,1]
action_tensor_checksum=-0.121305337
action_tensor_max=0.140111506
```

Verification 3: CUDA flex self-attn + auto window:

```text
VLA_LINGBOT_MIXED_FORWARD_SMOKE=1 \
VLA_LINGBOT_MIXED_CUDA_SELF_ATTN=1 \
VLA_LINGBOT_MIXED_BLOCKS=2 \
VLA_LINGBOT_BLOCK_WINDOW=auto \
VLA_LINGBOT_BLOCK_BUDGET_MB=4096 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
auto block window=2
Wan mixed forward block 1/2 complete (CUDA flex self-attn, window=2)
Wan mixed forward block 2/2 complete (CUDA flex self-attn, window=2)

latent_tensor_checksum=-9.69070096
action_tensor_checksum=-0.121305337
```

Updated status:

- Mixed forward now has:
  - original-order mixed sequence
  - mixed RoPE
  - flex mask/table
  - dense or CUDA flex self-attn block loop
  - block-window streaming loading
  - latent/action output heads
  - latent/action tensor detokenization
- Next step should replace synthetic latent/action tensors with runtime inputs and connect FlowMatchScheduler steps to the mixed forward output tensors.

## 2026-06-16 00:25:55 CST - Mixed Flow Loop on WanTransformer Forward / 基于 WanTransformer Forward 的 Mixed Flow Loop

中文：

- 将 mixed forward 从 smoke 单次调用抽成可复用函数：

```text
exec_wan_mixed_forward_tensors()
LingBotWanMixedForwardResult
```

- 新函数输入：

```text
latent tensor
action tensor
latent timestep
action timestep
block count
self-attn mode
window size
```

- 新函数输出：

```text
latent prediction tokens
action prediction tokens
latent prediction tensor
action prediction tensor
shape / split / padding metadata
```

- 新增 `VLA_LINGBOT_MIXED_FLOW_LOOP_SMOKE=1`：
  - 同时维护 video latent sample 和 action sample。
  - 每一步调用 `exec_wan_mixed_forward_tensors()`。
  - 用 video/action 各自的 `FlowMatchScheduler` 更新 sample。
  - 这比旧 `VLA_LINGBOT_FLOW_LOOP_SMOKE` 更接近 LingBot 原版，因为 video/action 在同一个 mixed WanTransformer forward 中共同推理。

验证 1：dense mixed flow-loop：

```text
VLA_LINGBOT_MIXED_FLOW_LOOP_SMOKE=1 \
VLA_LINGBOT_MIXED_FLOW_BLOCKS=1 \
VLA_LINGBOT_MIXED_FLOW_VIDEO_STEPS=1 \
VLA_LINGBOT_MIXED_FLOW_ACTION_STEPS=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
Wan mixed flow-loop smoke ok:
blocks=1
mode=ggml-dense
window=1
video_steps=1
action_steps=1
video_updates=1
action_updates=1
latent_checksum=17.7702648
latent_max=1.61915302
action_checksum=2.74183325
action_max=0.264272511
last_latent_pred_checksum=63.6524193
last_action_pred_checksum=30.189331
```

验证 2：CUDA flex self-attn mixed flow-loop：

```text
VLA_LINGBOT_MIXED_FLOW_LOOP_SMOKE=1 \
VLA_LINGBOT_MIXED_CUDA_SELF_ATTN=1 \
VLA_LINGBOT_MIXED_FLOW_BLOCKS=1 \
VLA_LINGBOT_MIXED_FLOW_VIDEO_STEPS=1 \
VLA_LINGBOT_MIXED_FLOW_ACTION_STEPS=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
Wan mixed flow-loop smoke ok:
blocks=1
mode=cuda-flex-self-attn
window=1
latent_checksum=-22.4845991
latent_max=1.59936035
action_checksum=1.69353803
action_max=0.142777264
last_latent_pred_checksum=-56.4726009
last_action_pred_checksum=0.932488563
```

验证 3：CUDA flex self-attn + 2 blocks + auto window：

```text
VLA_LINGBOT_MIXED_FLOW_LOOP_SMOKE=1 \
VLA_LINGBOT_MIXED_CUDA_SELF_ATTN=1 \
VLA_LINGBOT_MIXED_FLOW_BLOCKS=2 \
VLA_LINGBOT_MIXED_FLOW_VIDEO_STEPS=1 \
VLA_LINGBOT_MIXED_FLOW_ACTION_STEPS=1 \
VLA_LINGBOT_BLOCK_WINDOW=auto \
VLA_LINGBOT_BLOCK_BUDGET_MB=4096 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

结果：

```text
auto block window=2

Wan mixed flow-loop smoke ok:
blocks=2
mode=cuda-flex-self-attn
window=2
video_updates=1
action_updates=1
latent_checksum=-28.9377623
latent_max=1.51444483
action_checksum=1.75259212
action_max=0.120767057
last_latent_pred_checksum=3.49940212
last_action_pred_checksum=0.910482666
```

当前状态：

- WanTransformer3DModel C++ 主体现在已经具备：
  - mixed forward
  - mixed output tensor restore
  - mixed flow-loop iterative denoising
  - dense or CUDA flex self-attn
  - block window/auto window loading
- 仍待接入：
  - 真实 runtime image/state/text 输入
  - UMT5/VAE 或外部预处理输入桥接
  - 正式 `predict()` / `vla-server` action response
  - 30-block + BF16/量化部署策略

English:

- Extracted the mixed forward into a reusable function:

```text
exec_wan_mixed_forward_tensors()
LingBotWanMixedForwardResult
```

- Inputs:

```text
latent tensor
action tensor
latent timestep
action timestep
block count
self-attn mode
window size
```

- Outputs:

```text
latent prediction tokens
action prediction tokens
latent prediction tensor
action prediction tensor
shape / split / padding metadata
```

- Added `VLA_LINGBOT_MIXED_FLOW_LOOP_SMOKE=1`:
  - maintains video latent sample and action sample together
  - calls `exec_wan_mixed_forward_tensors()` at every step
  - updates samples with separate video/action `FlowMatchScheduler`s
  - this is closer to the original LingBot path than the old single-branch `VLA_LINGBOT_FLOW_LOOP_SMOKE`, because video/action are inferred together in one mixed WanTransformer forward

Verification 1: dense mixed flow-loop:

```text
VLA_LINGBOT_MIXED_FLOW_LOOP_SMOKE=1 \
VLA_LINGBOT_MIXED_FLOW_BLOCKS=1 \
VLA_LINGBOT_MIXED_FLOW_VIDEO_STEPS=1 \
VLA_LINGBOT_MIXED_FLOW_ACTION_STEPS=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
Wan mixed flow-loop smoke ok:
blocks=1
mode=ggml-dense
window=1
video_steps=1
action_steps=1
video_updates=1
action_updates=1
latent_checksum=17.7702648
latent_max=1.61915302
action_checksum=2.74183325
action_max=0.264272511
last_latent_pred_checksum=63.6524193
last_action_pred_checksum=30.189331
```

Verification 2: CUDA flex self-attn mixed flow-loop:

```text
VLA_LINGBOT_MIXED_FLOW_LOOP_SMOKE=1 \
VLA_LINGBOT_MIXED_CUDA_SELF_ATTN=1 \
VLA_LINGBOT_MIXED_FLOW_BLOCKS=1 \
VLA_LINGBOT_MIXED_FLOW_VIDEO_STEPS=1 \
VLA_LINGBOT_MIXED_FLOW_ACTION_STEPS=1 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
Wan mixed flow-loop smoke ok:
blocks=1
mode=cuda-flex-self-attn
window=1
latent_checksum=-22.4845991
latent_max=1.59936035
action_checksum=1.69353803
action_max=0.142777264
last_latent_pred_checksum=-56.4726009
last_action_pred_checksum=0.932488563
```

Verification 3: CUDA flex self-attn + 2 blocks + auto window:

```text
VLA_LINGBOT_MIXED_FLOW_LOOP_SMOKE=1 \
VLA_LINGBOT_MIXED_CUDA_SELF_ATTN=1 \
VLA_LINGBOT_MIXED_FLOW_BLOCKS=2 \
VLA_LINGBOT_MIXED_FLOW_VIDEO_STEPS=1 \
VLA_LINGBOT_MIXED_FLOW_ACTION_STEPS=1 \
VLA_LINGBOT_BLOCK_WINDOW=auto \
VLA_LINGBOT_BLOCK_BUDGET_MB=4096 \
./build/vla-server /home/xuling/robotic_dataset/models/lingbot_va_transformer_bf16.gguf
```

Result:

```text
auto block window=2

Wan mixed flow-loop smoke ok:
blocks=2
mode=cuda-flex-self-attn
window=2
video_updates=1
action_updates=1
latent_checksum=-28.9377623
latent_max=1.51444483
action_checksum=1.75259212
action_max=0.120767057
last_latent_pred_checksum=3.49940212
last_action_pred_checksum=0.910482666
```

Current status:

- The C++ WanTransformer3DModel core now has:
  - mixed forward
  - mixed output tensor restore
  - mixed flow-loop iterative denoising
  - dense or CUDA flex self-attn
  - block window/auto-window loading
- Still pending:
  - real runtime image/state/text inputs
  - UMT5/VAE or external preprocessed input bridge
  - formal `predict()` / `vla-server` action response
  - 30-block + BF16/quantized deployment strategy

## 2026-06-15 20:33:38 CST - Python-Like Terminal Flow Step / 对齐 Python 原版 Terminal Flow Step

中文：

- 将 `run_flow_loop_smoke()` 进一步整理成 branch-level flow skeleton。
- 新增 `LingBotFlowBranchConfig`，把 video/action 分支的关键参数显式化：
  - label
  - action mode
  - inference steps
  - scheduler shift
  - input dimension
  - sequence length
- 新增 `run_synthetic_flow_branch()`，统一执行：
  - scheduler `set_timesteps()`
  - synthetic sample 初始化
  - repeated `exec_forward_one_streaming()`
  - scheduler sample update
  - checksum / max_abs 汇报
- 对齐 LingBot-VA Python 原版 `_infer()` 的控制流：在 scheduler timesteps 后追加 terminal `t=0`。
- terminal step 会运行 transformer forward，但不会再更新 sample。后续接真实 KV cache 时，这一步对应 Python 里的最后一步 `update_cache=1` 语义。

验证：

```text
cmake --build build -j2

1 block, default 2 video updates + 2 action updates:
video  forward_steps=3 updates=2 blocks=1 terminal_t=0 checksum=-10.4148079 max_abs=1.53314626
action forward_steps=3 updates=2 blocks=1 terminal_t=0 checksum=-1.65648539 max_abs=0.548632026

3 blocks, 1 video update + 1 action update:
loaded flow loop blocks 0-2: 1872.90 MiB CPU F32 debug buffer
video  forward_steps=2 updates=1 blocks=3 terminal_t=0 checksum=-0.594179096 max_abs=2.16361713
action forward_steps=2 updates=1 blocks=3 terminal_t=0 checksum=-0.474481421 max_abs=0.355469406
```

说明：

- 1-block checksum 与上一版一致，因为 terminal step 只 forward 不 update。
- 这一步不是性能优化；当前仍是 CPU F32 debug 路径。它的价值是让 C++ 控制流更贴近 Python 原版，方便后续接 KV cache、真实 latent/action 状态和 ZMQ/protobuf `predict()`。

English:

- Further refactored `run_flow_loop_smoke()` into a branch-level flow skeleton.
- Added `LingBotFlowBranchConfig`, making the video/action branch parameters explicit:
  - label
  - action mode
  - inference steps
  - scheduler shift
  - input dimension
  - sequence length
- Added `run_synthetic_flow_branch()`, which uniformly runs:
  - scheduler `set_timesteps()`
  - synthetic sample initialization
  - repeated `exec_forward_one_streaming()`
  - scheduler sample update
  - checksum / max_abs reporting
- Matched the original LingBot-VA Python `_infer()` control flow by appending a terminal `t=0` after scheduler timesteps.
- The terminal step runs transformer forward but does not update the sample. Once real KV cache is wired, this maps to the Python final-step `update_cache=1` behavior.

Verification:

```text
cmake --build build -j2

1 block, default 2 video updates + 2 action updates:
video  forward_steps=3 updates=2 blocks=1 terminal_t=0 checksum=-10.4148079 max_abs=1.53314626
action forward_steps=3 updates=2 blocks=1 terminal_t=0 checksum=-1.65648539 max_abs=0.548632026

3 blocks, 1 video update + 1 action update:
loaded flow loop blocks 0-2: 1872.90 MiB CPU F32 debug buffer
video  forward_steps=2 updates=1 blocks=3 terminal_t=0 checksum=-0.594179096 max_abs=2.16361713
action forward_steps=2 updates=1 blocks=3 terminal_t=0 checksum=-0.474481421 max_abs=0.355469406
```

Notes:

- The 1-block checksum is unchanged from the previous version because the terminal step only forwards and does not update.
- This is not a performance optimization; it is still the CPU F32 debug path. The value is that the C++ control flow is now closer to the original Python implementation, making later KV cache, real latent/action state, and ZMQ/protobuf `predict()` integration cleaner.

## 2026-06-15 20:38:23 CST - Dynamic Seq and Explicit Grid Input / 动态 Seq 与显式 Grid 输入

中文：

- 将 `exec_embedding_stage()` 从固定 `seq=2` 的 smoke 入口升级为可根据 `raw_input` 长度动态推导 token 序列长度。
- `exec_embedding_stage()` 新增可选 `LingBotGridSpec` 输入：
  - 若外部传入 grid，则检查 `grid.seq() == seq`。
  - 若未传入 grid，则继续使用原有 smoke minimal grid，保持默认 smoke 行为不变。
- `exec_forward_one_streaming()` 同步新增可选 grid 参数。
- `LingBotFlowBranchConfig` 新增 `grid` 字段，flow-loop smoke 现在显式传入 video/action grid。
- 新增调试环境变量：

```text
VLA_LINGBOT_FLOW_LOOP_LATENT_SEQ
VLA_LINGBOT_FLOW_LOOP_ACTION_SEQ
```

验证：

```text
cmake --build build -j2

Dynamic seq smoke:
VLA_LINGBOT_FLOW_LOOP_BLOCKS=1
VLA_LINGBOT_FLOW_LOOP_VIDEO_STEPS=1
VLA_LINGBOT_FLOW_LOOP_ACTION_STEPS=1
VLA_LINGBOT_FLOW_LOOP_LATENT_SEQ=3
VLA_LINGBOT_FLOW_LOOP_ACTION_SEQ=4

video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-13.7852552 max_abs=1.55784786
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-4.74378143 max_abs=0.549534678

Default regression:
video  forward_steps=3 updates=2 blocks=1 terminal_t=0 checksum=-10.4148079 max_abs=1.53314626
action forward_steps=3 updates=2 blocks=1 terminal_t=0 checksum=-1.65648539 max_abs=0.548632026
```

意义：

- executor 不再被 smoke 的 `seq=2` 形状绑定。
- 后续真实 VAE latent shape 和 action chunk shape 可以通过 `raw_input + LingBotGridSpec` 接入同一条 forward 路径。
- 这一步仍然没有接真实 VAE/UMT5，只是先把 transformer executor 的输入接口扩成可承载真实形状。

English:

- Upgraded `exec_embedding_stage()` from a fixed `seq=2` smoke entrypoint to one that derives token sequence length dynamically from `raw_input`.
- Added an optional `LingBotGridSpec` input to `exec_embedding_stage()`:
  - If a grid is provided, `grid.seq() == seq` is checked.
  - If no grid is provided, the previous smoke minimal grid is used, preserving default smoke behavior.
- Added the same optional grid parameter to `exec_forward_one_streaming()`.
- Added a `grid` field to `LingBotFlowBranchConfig`; flow-loop smoke now passes explicit video/action grids.
- Added debug environment variables:

```text
VLA_LINGBOT_FLOW_LOOP_LATENT_SEQ
VLA_LINGBOT_FLOW_LOOP_ACTION_SEQ
```

Verification:

```text
cmake --build build -j2

Dynamic seq smoke:
VLA_LINGBOT_FLOW_LOOP_BLOCKS=1
VLA_LINGBOT_FLOW_LOOP_VIDEO_STEPS=1
VLA_LINGBOT_FLOW_LOOP_ACTION_STEPS=1
VLA_LINGBOT_FLOW_LOOP_LATENT_SEQ=3
VLA_LINGBOT_FLOW_LOOP_ACTION_SEQ=4

video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-13.7852552 max_abs=1.55784786
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-4.74378143 max_abs=0.549534678

Default regression:
video  forward_steps=3 updates=2 blocks=1 terminal_t=0 checksum=-10.4148079 max_abs=1.53314626
action forward_steps=3 updates=2 blocks=1 terminal_t=0 checksum=-1.65648539 max_abs=0.548632026
```

Meaning:

- The executor is no longer tied to the smoke-only `seq=2` shape.
- Future real VAE latent shapes and action chunk shapes can enter the same forward path through `raw_input + LingBotGridSpec`.
- This still does not wire real VAE/UMT5; it prepares the transformer executor input interface for real shapes.

## 2026-06-15 20:46:19 CST - Real Timestep Embedding / 真实 Timestep Embedding

中文：

- 补齐 WanTransformer 的真实 timestep feature 生成。
- 新增 C++ helper：

```text
fill_timestep_embedding(out, dim=256, seq, timestep)
```

- 对齐 LingBot-VA Python 原版 `WanTimeTextImageEmbedding` 中的 diffusers `Timesteps` 配置：

```text
num_channels=256
flip_sin_to_cos=True
downscale_freq_shift=0
max_period=10000
```

- `exec_embedding_stage()` 不再使用 deterministic fake `time_raw`，而是根据当前 scheduler timestep 生成 sinusoidal timestep embedding。
- `run_transformer_compute_smoke()` 也同步改为使用 `t=0` 的真实 timestep embedding。
- `scripts/run_lingbot_va_oneblock_parity.py` 同步新增 Python reference `timestep_embedding()`，避免 parity 继续使用旧 fake time feature。

验证：

```text
cmake --build build -j2

1-block streaming smoke, t=0:
latent checksum=-4.23685 max_abs=1.27981
action checksum=1.76394 max_abs=0.833068

Python parity, 1 block, final only:
latent max_abs_diff=0.05009947 mean_abs_diff=0.007071042
action max_abs_diff=0.0074115098 mean_abs_diff=0.0028003657

Flow-loop smoke, 1 block, 1 video update + 1 action update:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-1.16708725 max_abs=1.01988125
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.47451801 max_abs=0.250955105
```

说明：

- smoke checksum 与上一版不同是预期行为，因为 `time_raw` 从 fake deterministic tensor 变成了真实 sinusoidal timestep feature。
- 当前 timestep embedding 在 host/C++ 侧生成，然后作为 `time_raw` 输入 ggml 图；这不需要新增 ggml 算子。
- 这一步补上了 WanTransformer3DModel 中一个原先缺失的真实模型语义。

English:

- Added the real timestep feature generation path for WanTransformer.
- Added the C++ helper:

```text
fill_timestep_embedding(out, dim=256, seq, timestep)
```

- Matched the diffusers `Timesteps` configuration used by the original LingBot-VA Python `WanTimeTextImageEmbedding`:

```text
num_channels=256
flip_sin_to_cos=True
downscale_freq_shift=0
max_period=10000
```

- `exec_embedding_stage()` no longer uses deterministic fake `time_raw`; it now generates sinusoidal timestep embeddings from the current scheduler timestep.
- `run_transformer_compute_smoke()` was also switched to real timestep embeddings with `t=0`.
- Added a matching Python reference `timestep_embedding()` in `scripts/run_lingbot_va_oneblock_parity.py`, so parity no longer uses the old fake time feature.

Verification:

```text
cmake --build build -j2

1-block streaming smoke, t=0:
latent checksum=-4.23685 max_abs=1.27981
action checksum=1.76394 max_abs=0.833068

Python parity, 1 block, final only:
latent max_abs_diff=0.05009947 mean_abs_diff=0.007071042
action max_abs_diff=0.0074115098 mean_abs_diff=0.0028003657

Flow-loop smoke, 1 block, 1 video update + 1 action update:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-1.16708725 max_abs=1.01988125
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.47451801 max_abs=0.250955105
```

Notes:

- The smoke checksums changed as expected because `time_raw` changed from a fake deterministic tensor to real sinusoidal timestep features.
- The timestep embedding is generated on the host/C++ side and then fed into the ggml graph as `time_raw`; no new ggml operator is required.
- This fills one previously missing real-model semantic piece of WanTransformer3DModel.

## 2026-06-15 20:54:00 CST - Patchify / Unpatchify Layout Helpers / Patchify 与 Unpatchify 排列辅助

中文：

- 补齐 WanTransformer3DModel 接真实 latent/action 张量前所需的 host-side layout helper。
- 新增 5D shape 描述：

```text
LingBotTensor5DShape { b, c, f, h, w }
```

- 新增 latent 输入 patchify：

```text
patchify_latent_tokens()

Python 对齐:
b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)
```

- 新增 latent 输入 patchify 的反向 roundtrip helper：

```text
unpatchify_latent_tokens()
```

- 新增 transformer latent output projection 的还原：

```text
projected_latent_tokens_to_tensor()

Python 对齐:
b l (n c) -> b (l n) c
data_seq_to_patch(...) -> b c f h w
```

- 新增 action 输入/输出 token helper：

```text
action_tensor_to_tokens()
action_tokens_to_tensor()

Python 对齐:
b c f h w -> b (f h w) c
```

- 新增 smoke 开关：

```text
VLA_LINGBOT_PATCH_SMOKE=1
```

验证：

```text
cmake --build build -j2

VLA_LINGBOT_PATCH_SMOKE=1:
latent shape=[1,48,2,4,6] tokens=[192,12] checksum=1.38428171
projected latent shape=[1,48,2,4,6] checksum=2.76856342
action shape=[1,30,2,3,1] tokens=[30,6] checksum=3.90200628

Flow-loop regression:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-1.16708725 max_abs=1.01988125
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.47451801 max_abs=0.250955105
```

意义：

- C++ 侧现在具备把真实 VAE latent `[B,C,F,H,W]` 转成 WanTransformer token 输入的基础排列能力。
- C++ 侧也具备把 latent output projection 还原回 scheduler 需要的 latent tensor 形状的基础排列能力。
- action branch 的 `[B,C,F,H,W] <-> token` 基础排列也已覆盖。
- 当前 helper 是 host-side 实现，后续可以根据性能需要迁移到 ggml graph 或 CUDA kernel。

English:

- Added host-side layout helpers required before wiring real latent/action tensors into WanTransformer3DModel.
- Added a 5D shape descriptor:

```text
LingBotTensor5DShape { b, c, f, h, w }
```

- Added latent input patchify:

```text
patchify_latent_tokens()

Python equivalent:
b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)
```

- Added the reverse roundtrip helper for latent input patchify:

```text
unpatchify_latent_tokens()
```

- Added reconstruction for transformer latent output projection:

```text
projected_latent_tokens_to_tensor()

Python equivalent:
b l (n c) -> b (l n) c
data_seq_to_patch(...) -> b c f h w
```

- Added action input/output token helpers:

```text
action_tensor_to_tokens()
action_tokens_to_tensor()

Python equivalent:
b c f h w -> b (f h w) c
```

- Added the smoke switch:

```text
VLA_LINGBOT_PATCH_SMOKE=1
```

Verification:

```text
cmake --build build -j2

VLA_LINGBOT_PATCH_SMOKE=1:
latent shape=[1,48,2,4,6] tokens=[192,12] checksum=1.38428171
projected latent shape=[1,48,2,4,6] checksum=2.76856342
action shape=[1,30,2,3,1] tokens=[30,6] checksum=3.90200628

Flow-loop regression:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-1.16708725 max_abs=1.01988125
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.47451801 max_abs=0.250955105
```

Meaning:

- The C++ side can now convert real VAE latent `[B,C,F,H,W]` tensors into WanTransformer token input layout.
- The C++ side can also reconstruct latent output projection into the latent tensor shape expected by the scheduler.
- The action branch `[B,C,F,H,W] <-> token` layout is also covered.
- These helpers are currently host-side implementations; later they can be moved into a ggml graph or CUDA kernel if performance requires it.

## 2026-06-15 21:00:37 CST - Tensor-State Flow Loop Skeleton / Tensor 状态 Flow Loop 骨架

中文：

- 将 `run_synthetic_flow_branch()` 从 flat token sample 升级为 tensor-state sample。
- flow-loop smoke 现在维护 synthetic `[B,C,F,H,W]` sample：
  - video/latent: `[1, 48, seq * patch_t, patch_h, patch_w]`
  - action: `[1, action_dim, 1, seq, 1]`
- 每一个 forward step 现在执行更接近真实路径的流程：

```text
sample tensor
  -> patchify_latent_tokens() / action_tensor_to_tokens()
  -> exec_forward_one_streaming()
  -> projected_latent_tokens_to_tensor() / action_tokens_to_tensor()
  -> FlowMatchScheduler step on tensor sample
```

- terminal `t=0` step 仍然只 forward，不做 scheduler update。
- `LingBotFlowBranchConfig` 新增 `tensor_shape` 字段，后续真实 VAE latent/action state 可以沿用同一结构。

验证：

```text
cmake --build build -j2

Patch smoke:
latent shape=[1,48,2,4,6] tokens=[192,12] checksum=1.38428171
projected latent shape=[1,48,2,4,6] checksum=2.76856342
action shape=[1,30,2,3,1] tokens=[30,6] checksum=3.90200628

Tensor-state flow loop, default seq=2:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-0.638099637 max_abs=1.03857756
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.41036641 max_abs=0.261282086

Tensor-state flow loop, latent_seq=3, action_seq=4:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-2.61395121 max_abs=1.0602833
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=3.25923621 max_abs=0.257275224
```

说明：

- flow-loop checksum 与上一版 flat-token sample 不同是预期行为，因为 sample 状态现在保存在 5D tensor layout 中，并且每一步经过 patchify/tokenize 排列。
- 这一步使 flow loop 更贴近真实 LingBot-VA 推理：scheduler 更新的是 tensor 状态，而不是直接更新 flat token buffer。

English:

- Upgraded `run_synthetic_flow_branch()` from a flat-token sample to a tensor-state sample.
- Flow-loop smoke now maintains synthetic `[B,C,F,H,W]` samples:
  - video/latent: `[1, 48, seq * patch_t, patch_h, patch_w]`
  - action: `[1, action_dim, 1, seq, 1]`
- Each forward step now follows a path closer to real inference:

```text
sample tensor
  -> patchify_latent_tokens() / action_tensor_to_tokens()
  -> exec_forward_one_streaming()
  -> projected_latent_tokens_to_tensor() / action_tokens_to_tensor()
  -> FlowMatchScheduler step on tensor sample
```

- The terminal `t=0` step still only forwards and does not run a scheduler update.
- Added `tensor_shape` to `LingBotFlowBranchConfig`, so future real VAE latent/action state can reuse the same structure.

Verification:

```text
cmake --build build -j2

Patch smoke:
latent shape=[1,48,2,4,6] tokens=[192,12] checksum=1.38428171
projected latent shape=[1,48,2,4,6] checksum=2.76856342
action shape=[1,30,2,3,1] tokens=[30,6] checksum=3.90200628

Tensor-state flow loop, default seq=2:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-0.638099637 max_abs=1.03857756
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.41036641 max_abs=0.261282086

Tensor-state flow loop, latent_seq=3, action_seq=4:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-2.61395121 max_abs=1.0602833
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=3.25923621 max_abs=0.257275224
```

Notes:

- The flow-loop checksums changed from the previous flat-token sample version as expected, because the sample state now lives in a 5D tensor layout and is patchified/tokenized at every step.
- This makes the flow loop closer to real LingBot-VA inference: the scheduler updates tensor state, not a flat token buffer.

## 2026-06-15 21:13:49 CST - KV Cache State Skeleton / KV Cache 状态骨架

中文：

- 新增 host-side `LingBotKVCache`，对齐 LingBot-VA Python 原版 `WanAttention` 的 cache 状态管理语义。
- 已覆盖的 cache 语义：
  - `init()`：创建 `[batch, total_tokens, heads, head_dim]` 的 K/V buffer。
  - `allocate_slots()`：优先使用空槽；容量不足时按最老 `id` 淘汰。
  - `update()`：写入 K/V，设置 `mask/id/is_pred`。
  - `restore()`：模拟 `update_cache=0` 的临时写入后回滚。
  - `clear_pred()`：清理 `update_cache=1` 标记的 prediction cache。
  - `clear()`：释放 cache 状态。
- 新增 smoke 开关：

```text
VLA_LINGBOT_KV_SMOKE=1
```

验证：

```text
cmake --build build -j2

VLA_LINGBOT_KV_SMOKE=1:
kv smoke ok: total=4 heads=24 head_dim=128 used=3 pred=0 next_id=3 checksum_k=-0.0832768185

Flow-loop regression:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-0.638099637 max_abs=1.03857756
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.41036641 max_abs=0.261282086
```

## 2026-06-16 15:35 CST - UMT5 Encoder Host Parity Milestone

中文：

- 完成了 `lingbot_va_text_encoder_bf16.gguf` 的 UMT5 text encoder 权重读取和主干验证。
- 新增局部 token embedding 行读取路径：按 token id 从 `text.token_embd.weight` 只读取需要的 BF16 行，再转 F32；避免把整张 embedding 表展开成约 4GB F32。
- 新增 UMT5 block host executor：
  - RMSNorm
  - Q/K/V/O self-attention
  - bidirectional relative position bucket + relative attention bias
  - gated GELU FFN (`wi_0`, `wi_1`, `wo`)
  - final encoder RMSNorm
- 新增 smoke 开关：
  - `VLA_LINGBOT_TEXT_BLOCKS_SMOKE=1`
  - `VLA_LINGBOT_TEXT_EMBEDDING_SMOKE=1`
  - `VLA_LINGBOT_TEXT_EMBED_BLOCKS_SMOKE=1`
  - `VLA_LINGBOT_TEXT_BLOCKS=N`
  - `VLA_LINGBOT_TEXT_TOKEN_IDS=1,42,1234,32000`
- 2-block deterministic hidden parity:

```text
block0_out        max=0.0078125 mean=3.98648444e-05
block1_out        max=0.0078125 mean=4.15743234e-05
final_norm        max=3.42726707e-07 mean=2.59136499e-08
```

- token embedding + 2-block parity:

```text
token_embedding   max=0 mean=0
block0_out        max=0.005859375 mean=8.45967406e-06
block1_out        max=0.005859375 mean=9.48594788e-06
final_norm        max=1.78813934e-07 mean=1.12167644e-08
```

- token embedding + full 24-block UMT5 parity:

```text
full24_pre_final  max=0.015625 mean=0.000130681554
full24_final_norm max=2.14576721e-06 mean=2.27741708e-07
```

- 结论：UMT5 encoder 的结构逻辑已经完整对齐 Python reference。当前实现是 host/F32 parity 骨架，主要用于确认权重布局、公式和层间数据流；后续仍要迁移到正式 ggml/CUDA/量化执行路径。

English:

- Finished UMT5 text encoder weight loading and backbone validation from `lingbot_va_text_encoder_bf16.gguf`.
- Added partial token embedding row reads: selected BF16 rows are read from `text.token_embd.weight` by token id and converted to F32, avoiding a full ~4GB F32 embedding expansion.
- Added a UMT5 block host executor covering:
  - RMSNorm
  - Q/K/V/O self-attention
  - bidirectional relative position buckets + relative attention bias
  - gated GELU FFN (`wi_0`, `wi_1`, `wo`)
  - final encoder RMSNorm
- Added smoke switches:
  - `VLA_LINGBOT_TEXT_BLOCKS_SMOKE=1`
  - `VLA_LINGBOT_TEXT_EMBEDDING_SMOKE=1`
  - `VLA_LINGBOT_TEXT_EMBED_BLOCKS_SMOKE=1`
  - `VLA_LINGBOT_TEXT_BLOCKS=N`
  - `VLA_LINGBOT_TEXT_TOKEN_IDS=1,42,1234,32000`
- 2-block deterministic hidden parity:

```text
block0_out        max=0.0078125 mean=3.98648444e-05
block1_out        max=0.0078125 mean=4.15743234e-05
final_norm        max=3.42726707e-07 mean=2.59136499e-08
```

- token embedding + 2-block parity:

```text
token_embedding   max=0 mean=0
block0_out        max=0.005859375 mean=8.45967406e-06
block1_out        max=0.005859375 mean=9.48594788e-06
final_norm        max=1.78813934e-07 mean=1.12167644e-08
```

- token embedding + full 24-block UMT5 parity:

```text
full24_pre_final  max=0.015625 mean=0.000130681554
full24_final_norm max=2.14576721e-06 mean=2.27741708e-07
```

- Conclusion: the UMT5 encoder structure is now fully aligned with the Python reference. This is still a host/F32 parity skeleton for validating tensor layout, formulas, and inter-layer data flow; the next step is migrating it into the production ggml/CUDA/quantized execution path.

## 2026-06-16 15:44 CST - UMT5 Predict / World-Server Bridge

中文：

- 将 UMT5 host encoder 从独立 smoke 接入到 LingBot `predict()` 桥接路径。
- 新增 `LingBotTransformerExecutor::text_raw_override` 和动态 `text_seq`，WanTransformer cross-attention 不再固定使用 2 个假 text token。
- `VLA_LINGBOT_PREDICT_TEXT_ENCODER=1` 时，`predict()` 会：
  - 读取 `VLA_LINGBOT_TEXT_GGUF`
  - 从 `Inputs::lang_tokens` 按行读取 token embedding
  - 执行 UMT5 encoder block
  - 执行 final RMSNorm
  - 将 UMT5 输出送入 WanTransformer 的 text condition / cross-attention 路径
- 新增 `VLA_LINGBOT_PREDICT_TEXT_BLOCKS=N`，用于控制 predict smoke 里 UMT5 的 block 数；默认走完整层数。
- 扩展 `src/serving/lingbot.proto`：

```proto
repeated int32 lang_tokens = 7;
```

- `lingbot-world-server` 现在会校验 `StepRequest.lang_tokens`，并传入 `vla::Inputs`。
- world-server 响应状态中加入 `lang_tokens=<N>`，方便检查请求是否真的带入语言 token。

Verification:

```text
cmake --build build -j2

VLA_LINGBOT_PREDICT_SMOKE=1
VLA_LINGBOT_PREDICT_TEXT_ENCODER=1
VLA_LINGBOT_PREDICT_TEXT_BLOCKS=2
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_TEXT_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf

UMT5 encode host ok:
tokens=4 blocks=2 checksum=1.53220735 max=1.00777638

predict bridge ok:
chunk=[16,30] checksum=10.8645738 max=0.220107228
```

- 还做了 world-server 协议级 smoke：临时用 `protoc --python_out=/tmp/lingbot_proto_py` 生成 Python protobuf，发送带 `lang_tokens=[1,42,1234,32000]` 的 `StepRequest`。

```text
response:
request_id 101
status step_bridge session=7 step=1 latent_count=0 latent_bytes=0
error <empty>
chunk_size 16 action_dim 30 actions 480
latency_ms_total 4040.43
latency_ms_inference 2090.448
checksum 11.558194293640554
max_abs 0.21994079649448395
```

- 当前限制：UMT5 predict bridge 仍是 host/F32 路径，适合结构打通和 reference 验证；后续需要迁移到 ggml/CUDA/量化路径，并加入缓存以避免每个 step 重算 text encoder。

English:

- Connected the UMT5 host encoder from standalone smoke tests into the LingBot `predict()` bridge.
- Added `LingBotTransformerExecutor::text_raw_override` and dynamic `text_seq`; WanTransformer cross-attention no longer uses a hard-coded pair of dummy text tokens.
- When `VLA_LINGBOT_PREDICT_TEXT_ENCODER=1`, `predict()` now:
  - opens `VLA_LINGBOT_TEXT_GGUF`
  - reads token embedding rows from `Inputs::lang_tokens`
  - runs UMT5 encoder blocks
  - applies final RMSNorm
  - feeds the UMT5 output into the WanTransformer text condition / cross-attention path
- Added `VLA_LINGBOT_PREDICT_TEXT_BLOCKS=N` to control the number of UMT5 blocks in predict smoke; default is the full encoder depth.
- Extended `src/serving/lingbot.proto`:

```proto
repeated int32 lang_tokens = 7;
```

- `lingbot-world-server` now validates `StepRequest.lang_tokens` and forwards them through `vla::Inputs`.
- The world-server response status now includes `lang_tokens=<N>` for easier request debugging.

Verification:

```text
cmake --build build -j2

VLA_LINGBOT_PREDICT_SMOKE=1
VLA_LINGBOT_PREDICT_TEXT_ENCODER=1
VLA_LINGBOT_PREDICT_TEXT_BLOCKS=2
VLA_LINGBOT_PREDICT_BLOCKS=1
VLA_LINGBOT_TEXT_GGUF=/home/xuling/robotic_dataset/models/lingbot_va_text_encoder_bf16.gguf

UMT5 encode host ok:
tokens=4 blocks=2 checksum=1.53220735 max=1.00777638

predict bridge ok:
chunk=[16,30] checksum=10.8645738 max=0.220107228
```

- Also ran a world-server protocol smoke. A temporary Python protobuf was generated with `protoc --python_out=/tmp/lingbot_proto_py`, then a `StepRequest` with `lang_tokens=[1,42,1234,32000]` was sent.

```text
response:
request_id 101
status step_bridge session=7 step=1 latent_count=0 latent_bytes=0
error <empty>
chunk_size 16 action_dim 30 actions 480
latency_ms_total 4040.43
latency_ms_inference 2090.448
checksum 11.558194293640554
max_abs 0.21994079649448395
```

- Current limitation: the UMT5 predict bridge still runs as a host/F32 path for structural integration and reference validation. Next steps are moving it to ggml/CUDA/quantized execution and adding caching so text encoding is not recomputed every step.

说明：

- 当前完成的是 cache 状态管理 skeleton，尚未把 cache K/V 接进 ggml attention 计算图。
- 下一步如果继续接 attention，需要决定 dense cache 拼接 + mask 的第一版策略，还是直接向 flex/block-sparse mask 靠近。这是一个需要单独处理的实现点。

English:

- Added host-side `LingBotKVCache`, matching the cache state-management semantics of the original LingBot-VA Python `WanAttention`.
- Covered cache semantics:
  - `init()`: creates `[batch, total_tokens, heads, head_dim]` K/V buffers.
  - `allocate_slots()`: uses free slots first; evicts the oldest `id` entries when capacity is insufficient.
  - `update()`: writes K/V and sets `mask/id/is_pred`.
  - `restore()`: simulates rollback after temporary writes for `update_cache=0`.
  - `clear_pred()`: clears prediction cache entries marked by `update_cache=1`.
  - `clear()`: releases cache state.
- Added the smoke switch:

```text
VLA_LINGBOT_KV_SMOKE=1
```

Verification:

```text
cmake --build build -j2

VLA_LINGBOT_KV_SMOKE=1:
kv smoke ok: total=4 heads=24 head_dim=128 used=3 pred=0 next_id=3 checksum_k=-0.0832768185

Flow-loop regression:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-0.638099637 max_abs=1.03857756
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.41036641 max_abs=0.261282086
```

Notes:

- This implements the cache state-management skeleton only. Cache K/V are not yet wired into the ggml attention graph.
- The next attention integration step needs a focused implementation choice: start with dense cache concatenation + mask, or move directly toward the flex/block-sparse mask path.

## 2026-06-15 21:27:40 CST - Flex / Block-Sparse Mask Semantics Skeleton / Flex 与 Block-Sparse Mask 语义骨架

中文：

- 根据用户决策，跳过 dense cache 拼接优先路线，直接朝 LingBot 原版 `flex` / block-sparse attention mask 路径推进。
- 新增 `LingBotFlexMaskMeta`，保存 Python 原版 `FlexAttnFunc.init_mask()` 里的核心元信息：
  - `seq_ids`
  - `frame_ids`
  - `noise_ids`
  - `text_seq_ids`
  - `window_size`
  - `block_size`
- 新增 `build_flex_mask_meta()`，对齐 Python 原版：

```text
latent_seq_id / action_seq_id
latent_frame_id / action_frame_id
noise_ids = [latent_noise, latent_clean, action_noise, action_clean]
padded_length -> -1 sentinel
```

- 新增 self-attention mask 规则：

```text
clean2clean: noise(q)=1, noise(k)=1, frame(k) <= frame(q)
noise2clean: noise(q)=0, noise(k)=1, frame(k) < frame(q)
noise2noise: noise(q)=0, noise(k)=0, frame(k) == frame(q)
seq_mask: seq(q) == seq(k), both >= 0
window: abs(frame(q) - frame(k)) <= window_size
```

- 新增 cross-attention mask 规则：

```text
seq(q) == text_seq(k), both >= 0
```

- 新增 block occupancy 统计，作为后续 block-sparse kernel / ggml mask 的基准。
- 新增 smoke 开关：

```text
VLA_LINGBOT_FLEX_MASK_SMOKE=1
```

验证：

```text
Python reference:
tokens=27 self_allowed=184 cross_allowed=12288 occupied_blocks=15

cmake --build build -j2

VLA_LINGBOT_FLEX_MASK_SMOKE=1:
flex mask smoke ok: tokens=27 self_allowed=184 cross_allowed=12288 occupied_blocks=15 block_size=4

Flow-loop regression:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-0.638099637 max_abs=1.03857756
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.41036641 max_abs=0.261282086
```

说明：

- 当前完成的是 flex/block-sparse mask 的规则和 block occupancy skeleton，尚未接入 attention 计算图。
- 这一步是高效路径的前置基准：后续可以基于 `LingBotFlexMaskMeta` 生成 block-sparse attention 所需的 block table / mask table。

English:

- Following the user's decision, skipped the dense cache-concatenation-first route and moved directly toward the original LingBot `flex` / block-sparse attention mask path.
- Added `LingBotFlexMaskMeta`, storing the core metadata from the original Python `FlexAttnFunc.init_mask()`:
  - `seq_ids`
  - `frame_ids`
  - `noise_ids`
  - `text_seq_ids`
  - `window_size`
  - `block_size`
- Added `build_flex_mask_meta()`, matching the Python reference:

```text
latent_seq_id / action_seq_id
latent_frame_id / action_frame_id
noise_ids = [latent_noise, latent_clean, action_noise, action_clean]
padded_length -> -1 sentinel
```

- Added self-attention mask rules:

```text
clean2clean: noise(q)=1, noise(k)=1, frame(k) <= frame(q)
noise2clean: noise(q)=0, noise(k)=1, frame(k) < frame(q)
noise2noise: noise(q)=0, noise(k)=0, frame(k) == frame(q)
seq_mask: seq(q) == seq(k), both >= 0
window: abs(frame(q) - frame(k)) <= window_size
```

- Added cross-attention mask rule:

```text
seq(q) == text_seq(k), both >= 0
```

- Added block occupancy counting as a future baseline for block-sparse kernels or ggml masks.
- Added the smoke switch:

```text
VLA_LINGBOT_FLEX_MASK_SMOKE=1
```

Verification:

```text
Python reference:
tokens=27 self_allowed=184 cross_allowed=12288 occupied_blocks=15

cmake --build build -j2

VLA_LINGBOT_FLEX_MASK_SMOKE=1:
flex mask smoke ok: tokens=27 self_allowed=184 cross_allowed=12288 occupied_blocks=15 block_size=4

Flow-loop regression:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-0.638099637 max_abs=1.03857756
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.41036641 max_abs=0.261282086
```

Notes:

- This implements the flex/block-sparse mask rules and block occupancy skeleton only. It is not yet wired into the attention graph.
- This is the baseline for the efficient path: later we can generate block tables / mask tables for block-sparse attention from `LingBotFlexMaskMeta`.

## 2026-06-15 21:37:12 CST - Block-Sparse Table Skeleton / Block-Sparse 表骨架

中文：

- 在 `LingBotFlexMaskMeta` 基础上继续推进高效 attention 路线。
- 新增 `LingBotBlockSparseTable`，使用 CSR 风格结构保存 block-sparse 关系：

```text
q_blocks
kv_blocks
block_size
row_ptr
col_idx
```

- 新增 self-attention block table：

```text
build_self_block_table()
```

- 新增 cross-attention block table：

```text
build_cross_block_table()
```

- 新增 `max_row_nnz()`，用于统计每个 query block 最多连接多少 kv blocks，后续可用于 kernel launch / workspace 估计。
- `VLA_LINGBOT_FLEX_MASK_SMOKE=1` 现在同时验证：
  - token-level mask 允许数量。
  - self block table occupancy。
  - cross block table occupancy。
  - CSR row_ptr/col_idx 基本结构。

验证：

```text
cmake --build build -j2

VLA_LINGBOT_FLEX_MASK_SMOKE=1:
flex mask smoke ok:
tokens=27
self_allowed=184
cross_allowed=12288
self_blocks=15
self_max_row=4
cross_blocks=768
cross_max_row=128
block_size=4

Flow-loop regression:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-0.638099637 max_abs=1.03857756
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.41036641 max_abs=0.261282086
```

说明：

- self attention 已经能生成稀疏 block table。
- cross attention 在当前 smoke 中仍然较密：batch=1、text_seq=512、block_size=4，因此有效 query block 会连接 128 个 text kv blocks。
- 这一步仍未接入 CUDA/ggml attention kernel，但已经形成后续 kernel 可消费的 `row_ptr/col_idx` 结构。

English:

- Continued the efficient attention path on top of `LingBotFlexMaskMeta`.
- Added `LingBotBlockSparseTable`, a CSR-style structure for block-sparse relations:

```text
q_blocks
kv_blocks
block_size
row_ptr
col_idx
```

- Added the self-attention block table:

```text
build_self_block_table()
```

- Added the cross-attention block table:

```text
build_cross_block_table()
```

- Added `max_row_nnz()`, which reports the maximum number of kv blocks connected to any query block. This can later feed kernel launch / workspace estimates.
- `VLA_LINGBOT_FLEX_MASK_SMOKE=1` now verifies:
  - token-level allowed mask counts
  - self block-table occupancy
  - cross block-table occupancy
  - basic CSR `row_ptr/col_idx` structure

Verification:

```text
cmake --build build -j2

VLA_LINGBOT_FLEX_MASK_SMOKE=1:
flex mask smoke ok:
tokens=27
self_allowed=184
cross_allowed=12288
self_blocks=15
self_max_row=4
cross_blocks=768
cross_max_row=128
block_size=4

Flow-loop regression:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-0.638099637 max_abs=1.03857756
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.41036641 max_abs=0.261282086
```

Notes:

- Self-attention can now generate a sparse block table.
- Cross-attention is still dense-ish in this smoke: with batch=1, text_seq=512, and block_size=4, each valid query block connects to 128 text kv blocks.
- This is still not wired into a CUDA/ggml attention kernel, but the `row_ptr/col_idx` structure is now available for a future kernel path.

## 2026-06-15 21:47:14 CST - CUDA Block-Sparse Attention Smoke Kernel / CUDA Block-Sparse Attention Smoke Kernel

中文：

- 新增 LingBot 专用 CUDA block-sparse attention kernel 文件：

```text
src/kernels/lingbot/lingbot_flex_attn_cuda.h
src/kernels/lingbot/lingbot_flex_attn_cuda.cu
```

- 将 kernel 加入 CUDA 静态库构建，并定义：

```text
VLA_LINGBOT_FLEX_CUDA_KERNELS
```

- 新增 C API：

```text
lingbot_flex_attn_f32(...)
lingbot_flex_attn_cuda_smoke(...)
```

- 第一版 kernel 使用 `row_ptr / col_idx` 的 block table 遍历允许的 kv blocks。
- smoke 内部使用显式 token mask 做 token-level 过滤，并与 CPU dense masked attention reference 对齐。
- `VLA_LINGBOT_FLEX_MASK_SMOKE=1` 现在会同时验证：
  - flex mask token 规则
  - block-sparse table
  - CUDA block-sparse attention smoke kernel

验证：

```text
cmake --build build -j2

VLA_LINGBOT_FLEX_MASK_SMOKE=1:
flex mask smoke ok:
tokens=27
self_allowed=184
cross_allowed=12288
self_blocks=15
self_max_row=4
cross_blocks=768
cross_max_row=128
block_size=4

CUDA attention smoke:
max_diff=4.47034836e-08

Flow-loop regression:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-0.638099637 max_abs=1.03857756
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.41036641 max_abs=0.261282086
```

说明：

- 这是第一版正确性 smoke kernel，不是最终高性能 kernel。
- 当前 kernel 已经证明 `row_ptr/col_idx` 可以驱动 CUDA block-sparse attention 并与 dense masked reference 对齐。
- 目前还没有替换 WanTransformer 主 forward 中的 ggml dense attention；下一步需要把真实 Q/K/V layout、cache K/V pool 和 block table 接进 kernel 调用。

English:

- Added LingBot-specific CUDA block-sparse attention kernel files:

```text
src/kernels/lingbot/lingbot_flex_attn_cuda.h
src/kernels/lingbot/lingbot_flex_attn_cuda.cu
```

- Added the kernel to the CUDA static library build and defined:

```text
VLA_LINGBOT_FLEX_CUDA_KERNELS
```

- Added C APIs:

```text
lingbot_flex_attn_f32(...)
lingbot_flex_attn_cuda_smoke(...)
```

- The first kernel version traverses allowed kv blocks through `row_ptr / col_idx`.
- The smoke path uses an explicit token mask for token-level filtering and compares against a CPU dense masked-attention reference.
- `VLA_LINGBOT_FLEX_MASK_SMOKE=1` now verifies:
  - flex mask token rules
  - block-sparse table
  - CUDA block-sparse attention smoke kernel

Verification:

```text
cmake --build build -j2

VLA_LINGBOT_FLEX_MASK_SMOKE=1:
flex mask smoke ok:
tokens=27
self_allowed=184
cross_allowed=12288
self_blocks=15
self_max_row=4
cross_blocks=768
cross_max_row=128
block_size=4

CUDA attention smoke:
max_diff=4.47034836e-08

Flow-loop regression:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-0.638099637 max_abs=1.03857756
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.41036641 max_abs=0.261282086
```

Notes:

- This is a first correctness smoke kernel, not the final high-performance kernel.
- It proves that `row_ptr/col_idx` can drive CUDA block-sparse attention and match a dense masked reference.
- It has not yet replaced the ggml dense attention in the WanTransformer main forward. The next step is wiring real Q/K/V layout, cache K/V pools, and block tables into the kernel call.

## 2026-06-15 22:47:18 CST - Host-Driven Flex CUDA Smoke / 模型侧驱动的 Flex CUDA Smoke

中文：

- 将 CUDA block-sparse attention kernel 从内部固定 smoke 推进到由 `lingbot_va.cpp` 侧真实 mask/table 元信息驱动。
- 新增 CUDA public API：

```text
lingbot_flex_attn_f32_masked(...)
```

- 该 API 接收：

```text
Q / K / V
row_ptr / col_idx
token_mask
seq_q / seq_k / heads / head_dim / block_size
```

- 在 `lingbot_va.cpp` 中新增：

```text
build_self_token_mask()
dense_masked_attention_ref()
run_lingbot_host_driven_flex_cuda_smoke()
```

- 现在 `VLA_LINGBOT_FLEX_MASK_SMOKE=1` 会执行两层 CUDA 验证：
  - kernel 文件内部固定 toy smoke。
  - LingBot 模型侧生成的 `LingBotFlexMaskMeta + LingBotBlockSparseTable + token_mask` 驱动的 CUDA smoke。

验证：

```text
cmake --build build -j2

VLA_LINGBOT_FLEX_MASK_SMOKE=1:
flex mask smoke ok:
tokens=27
self_allowed=184
cross_allowed=12288
self_blocks=15
self_max_row=4
cross_blocks=768
cross_max_row=128
block_size=4

internal CUDA smoke:
max_diff=4.47034836e-08

host-driven LingBot CUDA smoke:
seq=27 heads=2 head_dim=8 max_diff=2.23517418e-08

Flow-loop regression:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-0.638099637 max_abs=1.03857756
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.41036641 max_abs=0.261282086
```

说明：

- 现在已经证明 LingBot C++ 侧生成的 mask/table/token mask 可以直接驱动 CUDA block-sparse attention kernel，并与 dense masked reference 对齐。
- 这仍然没有替换 WanTransformer 主 forward 的 ggml dense attention。
- 下一步是把真实 block 内的 Q/K/V layout、KV cache pool 和 `LingBotBlockSparseTable` 接到 `lingbot_flex_attn_f32_masked()`。

English:

- Advanced the CUDA block-sparse attention kernel from an internal fixed smoke to a host-driven smoke using real mask/table metadata generated in `lingbot_va.cpp`.
- Added the CUDA public API:

```text
lingbot_flex_attn_f32_masked(...)
```

- The API accepts:

```text
Q / K / V
row_ptr / col_idx
token_mask
seq_q / seq_k / heads / head_dim / block_size
```

- Added the following in `lingbot_va.cpp`:

```text
build_self_token_mask()
dense_masked_attention_ref()
run_lingbot_host_driven_flex_cuda_smoke()
```

- `VLA_LINGBOT_FLEX_MASK_SMOKE=1` now runs two CUDA checks:
  - the fixed toy smoke inside the kernel file
  - a LingBot host-driven CUDA smoke using `LingBotFlexMaskMeta + LingBotBlockSparseTable + token_mask`

Verification:

```text
cmake --build build -j2

VLA_LINGBOT_FLEX_MASK_SMOKE=1:
flex mask smoke ok:
tokens=27
self_allowed=184
cross_allowed=12288
self_blocks=15
self_max_row=4
cross_blocks=768
cross_max_row=128
block_size=4

internal CUDA smoke:
max_diff=4.47034836e-08

host-driven LingBot CUDA smoke:
seq=27 heads=2 head_dim=8 max_diff=2.23517418e-08

Flow-loop regression:
video  forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=-0.638099637 max_abs=1.03857756
action forward_steps=2 updates=1 blocks=1 terminal_t=0 checksum=1.41036641 max_abs=0.261282086
```

Notes:

- This proves that mask/table/token-mask metadata generated by the LingBot C++ path can directly drive the CUDA block-sparse attention kernel and match a dense masked reference.
- This still has not replaced the ggml dense attention in the WanTransformer main forward.
- The next step is wiring real per-block Q/K/V layout, KV cache pools, and `LingBotBlockSparseTable` into `lingbot_flex_attn_f32_masked()`.

English:

- Replaced the dummy dual-branch execution in the flow-loop smoke with a formal single-branch transformer forward skeleton.
- Added `exec_forward_one_streaming()`:
  - `exec_embedding_stage()`
  - configurable N x `exec_block_one()`
  - `exec_output_one()`
- `run_flow_loop_smoke()` now calls the single-branch forward directly, instead of computing an extra dummy latent/action branch through `exec_block_window()`.
- This better matches the future real prediction path: the video/world branch and the action branch can call the same single-branch executor independently.
- Process adjustment: simple engineering refactors with easy failure localization can move in larger steps, backed by build + smoke + parity regression checks.

Verification:

```text
cmake --build build -j2

VLA_LINGBOT_FLOW_LOOP_SMOKE=1 VLA_LINGBOT_FLOW_LOOP_BLOCKS=1
video  checksum=-10.4148079 max_abs=1.53314626
action checksum=-1.65648539 max_abs=0.548632026

VLA_LINGBOT_COMPUTE_SMOKE=1 VLA_LINGBOT_SMOKE_STREAM=1 VLA_LINGBOT_SMOKE_BLOCKS=1
streaming latent checksum=11.9305 max_abs=1.63704
streaming action checksum=2.10607 max_abs=0.561802

Python parity, 1 block, final only:
latent max_abs_diff=0.042242289 mean_abs_diff=0.012459938
action max_abs_diff=0.0088189691 mean_abs_diff=0.0027387722
```
## 2026-06-16 19:43 CST - Parity harness incident: Python modular reference froze the machine

中文：

- 当时已经完成的工作：
  - C++ `predict()` 新增两个模块边界 dump：
    - `lingbot_predict_text_emb_raw.f32/.shape.txt`
    - `lingbot_predict_vae_latent_raw.f32/.shape.txt`
  - `scripts/run_lingbot_va_e2e_parity.py` 新增：
    - `--mode python-modular`
    - `--mode compare-modular`
    - UMT5 text encoder modular reference
    - Wan VAE encoder modular reference
  - 语法检查已通过。
  - 2-sample C++ server/client check 在重启前已跑通过，并确认每个 sample 目录有 text/VAE raw dump。
  - 10-sample C++ baseline 在重启前已跑通过，checksum 与前一次一致。

- 卡死发生的位置：
  - 执行 `python-modular` 的 CPU reference：

```bash
eval/sim/simpler/simpler_uv/.venv/bin/python scripts/run_lingbot_va_e2e_parity.py \
  --mode python-modular \
  --out-dir /tmp/lingbot_modular_parity_cpp_check \
  --num-fixtures 2 \
  --text-blocks 1 \
  --python-device cpu \
  --python-dtype f32
```

  - 最后一条可见输出停在 `UMT5EncoderModel.from_pretrained(...)` 加载 checkpoint shards 阶段：
    - `Loading checkpoint shards: 33%|...| 1/3`

- 排查结论：
  - 重启后没有残留进程。
  - 上一轮 journal 没有看到 OOM killer、CUDA Xid、kernel panic 等明确错误。
  - 结合现象，更像 Python CPU reference 在加载 UMT5 F32 权重时造成系统内存/交换分区压力过大，桌面无响应，随后手动长按电源键关机。
  - `/tmp` 中前一次 parity 输出被清理，需要后续重跑。
  - 源码改动仍在；但 build 产物时间戳较早，后续需要重新 `cmake --build build -j2`。

- 后续规避策略：
  - 不再用 F32 CPU 路径加载完整 UMT5 reference。
  - Python modular reference 改成更安全的方式：
    - 默认 `bf16`；
    - 优先只跑 VAE 或只跑 text 单模块；
    - 对 UMT5 reference 增加模块选择和内存限制；
    - 必要时只比较 GGUF/C++ 与 safetensors 的小切片或单 block，而不是整模型一次加载。

English:

- Completed before the freeze:
  - Added C++ predict boundary dumps:
    - `lingbot_predict_text_emb_raw.f32/.shape.txt`
    - `lingbot_predict_vae_latent_raw.f32/.shape.txt`
  - Added to `scripts/run_lingbot_va_e2e_parity.py`:
    - `--mode python-modular`
    - `--mode compare-modular`
    - UMT5 text encoder modular reference
    - Wan VAE encoder modular reference
  - Syntax checks passed.
  - A 2-sample C++ server/client check succeeded before reboot and confirmed text/VAE raw dumps per sample.
  - A 10-sample C++ baseline succeeded before reboot with the same checksums as the previous run.

- Freeze point:
  - The machine froze during CPU Python modular reference, specifically while loading UMT5 checkpoint shards:
    - last visible line: `Loading checkpoint shards: 33%|...| 1/3`

- Diagnosis:
  - No leftover processes after reboot.
  - Previous journal did not show OOM killer, CUDA Xid, or kernel panic.
  - Most likely the F32 CPU UMT5 reference path pushed RAM/swap too hard and made the desktop unresponsive, followed by manual power-button shutdown.
  - `/tmp` parity outputs from that run were cleared.
  - Source edits survived, but binaries need rebuilding before reuse.

- Next safety policy:
  - Avoid full F32 CPU UMT5 reference loading.
  - Use lower-memory modular parity:
    - default to BF16;
    - run VAE-only or text-only modules;
    - add explicit module selection and memory limits;
    - compare small/block-local slices when needed instead of loading the full Python model at once.

## 2026-06-16 20:00 CST - LingBot Python low-VRAM single-sample GPU path

中文：

- 在原版 LingBot repo `/home/xuling/robotic_code/embodied.cpp/lingbot-va-main` 中新增低显存路径：
  - `wan_va/modules/model.py`
    - 给 `WanTransformer3DModel` 增加 `enable_blockwise_gpu_offload()`
    - WanTransformer blocks 保持在 CPU，每个 block forward 时临时搬到 GPU，用完搬回 CPU
  - `wan_va/wan_va_server.py`
    - 新增环境变量 `LINGBOT_BLOCKWISE_GPU_OFFLOAD=1`
    - 启用后 transformer 从 CPU 加载，并开启 block-wise GPU offload
  - `tools/low_vram_single_gpu_sample.py`
    - 新增 staged 单样本 GPU runner
    - 阶段 1：UMT5 text encoder block-wise GPU，输出 text embedding 后释放
    - 阶段 2：VAE GPU encode，输出 latent 后释放
    - 阶段 3：WanTransformer CPU resident + per-block GPU offload

- 重要修正：
  - 完整 UMT5 BF16 一次性 `.to(cuda)` 在 8GB 3070Ti 上 OOM：

```text
torch.OutOfMemoryError: Tried to allocate 32.00 MiB; only 69.50 MiB free
```

  - 因此 text encoder 也必须 block-wise GPU offload，不能整模型搬 GPU。

- 验证 1：只跑 text + VAE，成功：

```bash
timeout 420 eval/sim/simpler/simpler_uv/.venv/bin/python \
  /home/xuling/robotic_code/embodied.cpp/lingbot-va-main/tools/low_vram_single_gpu_sample.py \
  --skip-transformer --dtype bf16 --device cuda:0
```

结果：

```text
[stage:text] hidden shape=(1, 512, 4096) checksum=-2385.09375
[mem] after text unload: allocated=0.008 GiB reserved=0.020 GiB
[stage:vae] latent shape=(1, 48, 1, 8, 16) checksum=-187.387375
[mem] after vae unload: allocated=0.008 GiB reserved=0.023 GiB
```

- 验证 2：单样本 + WanTransformer 1 block block-wise GPU，成功：

```bash
timeout 600 eval/sim/simpler/simpler_uv/.venv/bin/python \
  /home/xuling/robotic_code/embodied.cpp/lingbot-va-main/tools/low_vram_single_gpu_sample.py \
  --dtype bf16 --device cuda:0 --blocks 1
```

结果：

```text
[stage:transformer] truncating blocks 30 -> 1
[mem] before transformer forward: allocated=0.342 GiB reserved=0.350 GiB
[stage:transformer] video_out shape=(1, 128, 48) checksum=-40.8582916
[stage:transformer] action_out shape=(1, 16, 30) checksum=5.18078613
[mem] after transformer unload: allocated=0.012 GiB reserved=0.021 GiB
```

- 当前结论：
  - Python reference 可以走 GPU，但必须 staged + block-wise。
  - 1 block 链路已跑通。
  - 下一步按 `--blocks 2/4/8/16/30` 扩展，并观察系统 RAM 与 GPU 显存。

English:

- Added a low-VRAM path in the original LingBot repo:
  - `wan_va/modules/model.py`
    - Added `WanTransformer3DModel.enable_blockwise_gpu_offload()`
    - Transformer blocks stay on CPU and are moved to GPU only for their forward call.
  - `wan_va/wan_va_server.py`
    - Added `LINGBOT_BLOCKWISE_GPU_OFFLOAD=1`
    - Loads transformer on CPU and enables block-wise GPU offload.
  - `tools/low_vram_single_gpu_sample.py`
    - New staged single-sample GPU runner.
    - Stage 1: UMT5 text encoder block-wise GPU, then unload.
    - Stage 2: VAE GPU encode, then unload.
    - Stage 3: CPU-resident WanTransformer with per-block GPU offload.

- Important finding:
  - Full UMT5 BF16 `.to(cuda)` does not fit on the 8GB 3070Ti:

```text
torch.OutOfMemoryError: Tried to allocate 32.00 MiB; only 69.50 MiB free
```

  - Therefore UMT5 also needs block-wise GPU offload.

- Verification 1: text + VAE only succeeded.
- Verification 2: single sample with 1 WanTransformer block succeeded.
- Current conclusion:
  - Python reference can run on GPU, but only with staged loading and block-wise offload.
  - The 1-block path is working.
  - Next step: expand to `--blocks 2/4/8/16/30` while monitoring RAM and VRAM.

## 2026-06-16 20:17 CST - Full 30-block Python staged/block-wise GPU path succeeded

中文：

- 将低显存策略继续推进到“全线”：
  - UMT5 text encoder：CPU resident + block-wise GPU
  - VAE encoder：按需搬到 GPU encode，用完搬回 CPU
  - WanTransformer：CPU resident + per-block GPU
  - 单样本 staged runner 完整覆盖 text -> VAE -> WanTransformer video/action forward

- 代码更新：
  - `/home/xuling/robotic_code/embodied.cpp/lingbot-va-main/wan_va/modules/utils.py`
    - 新增 `BlockwiseCudaWrapper`
    - 新增 `enable_umt5_blockwise_gpu()`
  - `/home/xuling/robotic_code/embodied.cpp/lingbot-va-main/wan_va/wan_va_server.py`
    - `LINGBOT_BLOCKWISE_GPU_OFFLOAD=1` 时启用 UMT5 block-wise GPU
    - `_encode_obs()` 在该模式下临时将 VAE 搬到 GPU，编码完成后搬回 CPU 并清理 CUDA cache
  - `/home/xuling/robotic_code/embodied.cpp/lingbot-va-main/tools/low_vram_single_gpu_sample.py`
    - 已可跑完整 30-block 单样本 staged GPU path

- 分级验证结果：

```text
--blocks 2:
video_out checksum=-22.8283539
action_out checksum=-4.06570435

--blocks 4:
video_out checksum=92.0187759
action_out checksum=-0.815895081

--blocks 8:
video_out checksum=52.9975815
action_out checksum=4.58625793

--blocks 16:
video_out checksum=-280.71225
action_out checksum=9.71612549

--blocks 30:
video_out checksum=350.997589
action_out checksum=-14.1728287
```

- 完整 30-block 命令：

```bash
timeout 1200 eval/sim/simpler/simpler_uv/.venv/bin/python \
  /home/xuling/robotic_code/embodied.cpp/lingbot-va-main/tools/low_vram_single_gpu_sample.py \
  --dtype bf16 --device cuda:0 --blocks 30
```

- 完整 30-block 显存结果：

```text
[mem] before transformer forward: allocated=0.342 GiB reserved=0.350 GiB
[stage:transformer] video_out shape=(1, 128, 48) checksum=350.997589
[stage:transformer] action_out shape=(1, 16, 30) checksum=-14.1728287
[mem] after transformer unload: allocated=0.012 GiB reserved=0.021 GiB
[done] low-vram single-sample run completed
```

- 当前限制：
  - 单样本 staged runner 已经完整跑通。
  - 原版 `VA_Server` 已接入 `LINGBOT_BLOCKWISE_GPU_OFFLOAD=1`，但还没有直接完整启动验证。
  - 原因是持久化 server 初始化时仍可能同时常驻 VAE、UMT5、WanTransformer 的 CPU 权重，RAM 压力比 staged runner 大。
  - 后续如果要把 Python 原版 server 也做成稳定低内存版本，需要进一步做 lazy load / module unload，而不只是 block-wise GPU。

English:

- Extended the low-VRAM strategy to the full line:
  - UMT5 text encoder: CPU resident + block-wise GPU.
  - VAE encoder: moved to GPU only for encoding, then moved back to CPU.
  - WanTransformer: CPU resident + per-block GPU.
  - The single-sample staged runner now covers text -> VAE -> WanTransformer video/action forward.

- Code updates:
  - Added `BlockwiseCudaWrapper` and `enable_umt5_blockwise_gpu()` in `wan_va/modules/utils.py`.
  - Wired `LINGBOT_BLOCKWISE_GPU_OFFLOAD=1` into `wan_va/wan_va_server.py`.
  - VAE in `_encode_obs()` is temporarily moved to GPU in this mode and moved back to CPU after encoding.
  - `tools/low_vram_single_gpu_sample.py` now runs the complete 30-block staged GPU path.

- Validation:
  - `--blocks 2/4/8/16/30` all succeeded.
  - Full 30-block video/action forward completed on the local 8GB 3070Ti.

- Current limitation:
  - The staged single-sample runner is fully validated.
  - The persistent original `VA_Server` path is wired but not fully launched yet.
  - A stable low-memory Python server will likely need lazy loading/module unload in addition to block-wise GPU offload, because keeping VAE + UMT5 + WanTransformer CPU weights resident at the same time can still pressure system RAM.

## 2026-06-16 20:27 CST - Re-generated C++ 10-sample baseline for parity

中文：

- 重新编译 C++：

```bash
cmake --build build -j2
```

- 重新跑 10 个 deterministic C++ samples，输出保存到持久目录：

```bash
eval/sim/libero/libero_uv/.venv/bin/python scripts/run_lingbot_va_e2e_parity.py \
  --mode cpp \
  --out-dir /home/xuling/robotic_dataset/parity_runs/lingbot_cpp_10_baseline \
  --num-fixtures 10 \
  --blocks 1 \
  --text-blocks 1 \
  --video-steps 1 \
  --action-steps 1
```

- 10 个样本全部跑通，action checksum：

```text
000 checksum=5.91723728 max=0.233619809
001 checksum=6.48227501 max=0.273430109
002 checksum=6.81425381 max=0.327453732
003 checksum=6.83273888 max=0.298802018
004 checksum=7.40984344 max=0.303530097
005 checksum=5.97485733 max=0.248710632
006 checksum=6.37658262 max=0.307757974
007 checksum=6.77864361 max=0.302134395
008 checksum=6.76662254 max=0.332090855
009 checksum=6.71696377 max=0.282952428
```

- 文件完整性检查通过。每个 sample 目录都有：

```text
client_action_chunk.f32
lingbot_predict_action_chunk.f32
lingbot_predict_text_emb_raw.f32
lingbot_predict_vae_latent_raw.f32
lingbot_predict_last_action_pred.f32
lingbot_predict_last_latent_pred.f32
```

- 当前目录大小约 `4.2M`：

```text
/home/xuling/robotic_dataset/parity_runs/lingbot_cpp_10_baseline
```

- 下一步：
  - 将 Python staged/block-wise runner 接入 parity harness。
  - 用同一批 fixtures 逐个生成 Python staged 输出。
  - 对比 C++ dump 与 Python staged output。

English:

- Rebuilt C++ and regenerated the 10 deterministic C++ samples.
- Output directory:

```text
/home/xuling/robotic_dataset/parity_runs/lingbot_cpp_10_baseline
```

- All 10 samples completed and produced action chunks plus text/VAE/action/latent dumps.
- This directory is now the C++ reference baseline for the next staged Python parity step.

## 2026-06-16 21:45 CST - Sample000 Python/C++ parity debugging: action path mostly aligned

中文：

- 建立了 sample000 的 step-level trace：
  - C++: `/home/xuling/robotic_dataset/parity_runs/lingbot_cpp_sample000_flashattn`
  - Python: `/home/xuling/robotic_dataset/parity_runs/lingbot_python_sample000_fixture_ctxtrace`
- 修复了 C++ `time_proj` 路径：原版 Python 是 `time_proj(SiLU(temb))`，C++ 之前漏掉了 `SiLU`。修复后 `latent/action timestep_proj` 降到约 `3e-4` mean diff。
- 修复了 C++ Wan self-attention 的主要布局问题：原手写 `mul_mat + softmax + merge` 的维度排列不匹配 PyTorch SDPA；改为参考 vla.cpp 现有模型使用 `ggml_flash_attn_ext`。
- 修复后，step000 中：
  - `self_attn` 从巨大偏差降到接近 BF16/F32 数值误差。
  - `cross_attn` 基本对齐。
  - `last_action_pred` 对比 Python reference：

```text
mean_abs_diff ~= 0.0059968
max_abs_diff  ~= 0.05152
```

- 仍需继续关注：
  - video/latent 输出的布局与 unpatchify/seq 表达还需要单独对齐。
  - FFN 仍有约 `0.02` mean diff，可能来自 Python BF16 matmul/cast 与 C++ F32 compute 的数值路径差异，后续可进一步做 BF16/quantized execution parity。

English:

- Added step-level trace files for sample000:
  - C++: `/home/xuling/robotic_dataset/parity_runs/lingbot_cpp_sample000_flashattn`
  - Python: `/home/xuling/robotic_dataset/parity_runs/lingbot_python_sample000_fixture_ctxtrace`
- Fixed the C++ `time_proj` path. Python uses `time_proj(SiLU(temb))`; C++ previously skipped that SiLU. After the fix, latent/action `timestep_proj` is down to roughly `3e-4` mean diff.
- Fixed the main Wan self-attention layout issue. The old hand-written `mul_mat + softmax + merge` path did not match PyTorch SDPA layout; it now uses `ggml_flash_attn_ext`, following existing vla.cpp model patterns.
- After the fix:
  - `self_attn` is close to expected BF16/F32 numerical tolerance.
  - `cross_attn` is also closely aligned.
  - `last_action_pred` vs Python reference:

```text
mean_abs_diff ~= 0.0059968
max_abs_diff  ~= 0.05152
```

- Remaining work:
  - Align video/latent output layout and unpatchify/sequence representation.
  - FFN still has about `0.02` mean diff, likely from Python BF16 matmul/cast versus C++ F32 compute; this can be revisited when implementing BF16/quantized execution parity.
