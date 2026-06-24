# LingBot-VA vla.cpp 移植规划 / LingBot-VA vla.cpp Port Plan

本文档记录当前对 LingBot-VA 的理解，以及后续将其接入 `vla.cpp` 的工程路线。当前阶段刻意不依赖完整 safetensors 下载；完整 tensor 清单可以在权重下载完成后再补。

This document records the current understanding of LingBot-VA and the planned engineering route for adding it to `vla.cpp`. It intentionally does not depend on a complete safetensors download yet; the full tensor inventory can be added after the weights finish downloading.

## 当前决策 / Current Decision

我们采用路线 B：

We will follow route B:

- 先做静态分析和 vla.cpp 移植规划。
- First do static analysis and vla.cpp port planning.
- 保留原版 Python LingBot-VA LIBERO 推理路径，作为之后的参考验证目标。
- Keep the original Python LingBot-VA LIBERO inference path as a later reference target.
- 不从 action-only 简化版开始。LingBot-VA 的动作生成和 video latent、flow steps、transformer KV cache 强耦合，过早裁掉 video 部分容易得到误导性的工程结果。
- Do not start with an action-only shortcut. LingBot-VA action generation is coupled to video latents, flow steps, and the transformer KV cache, so a small action-only port would likely be misleading.

接口协议决策：

Interface protocol decision:

- 后续 LingBot-VA 接入以 `vla.cpp` 现有 ZMQ/protobuf 通讯为标准。
- Future LingBot-VA integration should use the existing `vla.cpp` ZMQ/protobuf communication path as the standard interface.
- 原始 LingBot-VA Python WebSocket 协议只作为理解和参考，不作为主线实现目标。
- The original LingBot-VA Python WebSocket protocol is only a reference for understanding, not the primary implementation target.
- 如果后续发现 `src/serving/vla.proto` 缺少表达 LingBot-VA reset、KV cache update 或 request mode 的字段，再按 vla.cpp 风格扩展 protobuf。
- If `src/serving/vla.proto` later lacks fields for LingBot-VA reset, KV cache update, or request mode semantics, extend the protobuf in the vla.cpp style.

模型承载决策：

Model hosting decision:

- 最终目标是在 `vla.cpp` 内完整、高质量地承载 LingBot-VA 所需的 UMT5、VAE、世界模型 Transformer、flow scheduler、KV cache 和 action postprocess。
- The final goal is to host LingBot-VA's required UMT5, VAE, world-model Transformer, flow scheduler, KV cache, and action postprocess fully and well inside `vla.cpp`.
- 分阶段实现仍然保留：第一阶段可以先用 Python 预计算 prompt embeddings 和 image/video latents，以便更快对齐 WanTransformer3DModel 主干。
- Staged implementation is still useful: the first stage may use Python-precomputed prompt embeddings and image/video latents to align the WanTransformer3DModel backbone faster.
- 但 Python 预计算只是调试和对齐手段，不是最终产品形态。
- Python precomputation is only a debugging and alignment method, not the final product shape.
- 补齐 UMT5 和 VAE 的 C++/ggml 执行能力是本项目的重要工作点，也可以成为 technical report 中区别于原作的核心工程贡献之一。
- Adding robust C++/ggml execution support for UMT5 and VAE is an important project goal and can become a core engineering contribution for the technical report.

本地部署目标：

Local deployment target:

- 第一次真实 end-to-end 运行的目标硬件是本机 NVIDIA GeForce RTX 3070 Ti Laptop GPU，8 GB VRAM。
- The first real end-to-end run targets the local NVIDIA GeForce RTX 3070 Ti Laptop GPU with 8 GB VRAM.
- 效率目标以“部署效率和运行效率尽量高”为方向，但最低验收线是本地 8 GB GPU 可以实际运行。
- The efficiency goal is to make deployment and runtime as efficient as practical, with the minimum acceptance target being actual execution on the local 8 GB GPU.
- 这意味着不能假设 BF16 全量 LingBot-VA 权重常驻显存；后续需要认真设计量化、CPU/GPU offload、分模块加载、低显存 KV cache 和可能的分阶段执行。
- This means we cannot assume the full BF16 LingBot-VA weights can stay resident in VRAM; quantization, CPU/GPU offload, staged module loading, low-memory KV cache, and staged execution must be considered seriously.

量化目标：

Quantization target:

- 量化策略由本项目维护，不简单机械套用已有模型的一刀切策略。
- The quantization strategy will be owned by this project, not mechanically copied as a one-size-fits-all policy from existing models.
- 可以复用 `vla.cpp` / `llama.cpp` / `ggml` 已有的 GGUF、量化 dtype、CUDA backend 和 matmul kernel，但 LingBot-VA 的模块级策略需要单独设计。
- Existing `vla.cpp` / `llama.cpp` / `ggml` facilities such as GGUF, quantized dtypes, CUDA backend, and matmul kernels can be reused, but LingBot-VA needs its own module-level strategy.
- 目标是在压缩倍率和精度损失之间取得平衡：优先保证本地 8 GB 可运行，同时尽量把整体精度或成功率损失控制在约 3%-5% 以内。
- The goal is to balance compression ratio and accuracy loss: prioritize local 8 GB execution while trying to keep overall accuracy or success-rate loss within roughly 3%-5%.
- 3%-5% 先作为工程目标，不作为未经实验验证的承诺；后续需要用 LIBERO 评测和模块级数值对齐来校准。
- The 3%-5% range is an engineering target, not an unverified promise; it must be calibrated later with LIBERO evaluation and module-level numerical comparisons.
- 量化不应在 FP16/BF16 GGUF 正确性基线之前过早定死。先建立正确性基线，再基于模块大小、误差敏感度和显存峰值确定 Transformer、UMT5、VAE、KV cache 的量化/offload 策略。
- Quantization should not be finalized before the FP16/BF16 GGUF correctness baseline. First establish the correctness baseline, then choose quantization/offload strategies for Transformer, UMT5, VAE, and KV cache based on module size, error sensitivity, and peak memory.

本地 LingBot-VA 代码仓库：

Local LingBot-VA code repo:

```text
/home/xuling/robotic_code/embodied.cpp/lingbot-va-main
```

当前 Hugging Face snapshot 路径：

Current Hugging Face snapshot path:

```text
/home/xuling/.cache/huggingface/hub/models--robbyant--lingbot-va-posttrain-libero-long/snapshots/0e89d1e753019988aba484e8da2dc0810e264d9f
```

截至本文档创建时，该 snapshot 中已有 config 和 index 文件，但 snapshot 树下还没有完整 safetensors payload。完整 tensor 清单应等待下载完成后再做。

At the time of this note, that snapshot contains config and index files, but the complete safetensors payload is not yet present in the snapshot tree. Full tensor inventory should wait until the download is complete.

## Python LIBERO 运行结构 / Python LIBERO Runtime Shape

LingBot-VA 原版使用 Python WebSocket server/client 分离架构，和当前 `vla.cpp` 的 ZMQ/protobuf 协议不同。移植时以 vla.cpp 的 ZMQ/protobuf 为标准，WebSocket 仅作为参考链路。

The original LingBot-VA uses a Python WebSocket server/client split, separate from the current `vla.cpp` ZMQ/protobuf protocol. The port should standardize on vla.cpp's ZMQ/protobuf path; WebSocket is only a reference path.

高层调用链：

High-level process:

```text
evaluation/libero/client.py
  -> WebsocketClientPolicy.infer(...)
  -> WebsocketPolicyServer._handler(...)
  -> DistributedModelWrapper.infer(...)
  -> VA_Server.infer(...)
  -> VA_Server._reset / _infer / _compute_kv_cache
```

关键文件：

Important files:

```text
evaluation/libero/client.py
wan_va/wan_va_server.py
wan_va/modules/model.py
wan_va/modules/utils.py
wan_va/utils/scheduler.py
wan_va/utils/sever_utils.py
wan_va/utils/Simple_Remote_Infer/deploy/websocket_client_policy.py
wan_va/utils/Simple_Remote_Infer/deploy/websocket_policy_server.py
wan_va/configs/va_libero_cfg.py
```

vla.cpp 侧当前协议文件：

Current vla.cpp-side protocol file:

```text
src/serving/vla.proto
```

当前 `PredictRequest` 已有字段：

Current `PredictRequest` fields include:

```text
images
lang_tokens
state
noise
request_id
precomputed_img_emb
precomputed_img_emb_n_views
attention_mask
```

当前 `PredictResponse` 已有字段：

Current `PredictResponse` fields include:

```text
request_id
action_chunk
chunk_size
action_dim
latency fields
error
```

LingBot-VA 的 reset 和 `_compute_kv_cache` 语义可能无法直接用现有字段完整表达。是否扩展 proto 要等我们完成最小 C++ forward 和状态管理设计后再定。

LingBot-VA reset and `_compute_kv_cache` semantics may not be fully expressible with the current fields. Whether to extend the proto should be decided after the minimal C++ forward path and state-management design are clearer.

## LIBERO Client 流程 / LIBERO Client Flow

`evaluation/libero/client.py` 创建 LIBERO `OffScreenRenderEnv`，提取两路 RGB 观测，然后反复调用远端 policy。

`evaluation/libero/client.py` creates a LIBERO `OffScreenRenderEnv`, extracts two RGB observations, then repeatedly calls the remote policy.

发送给 server 的 observation keys：

Observation keys sent to the server:

```text
observation.images.agentview_rgb
observation.images.eye_in_hand_rgb
```

图像经过提取和 resize 后是 128x128 RGB array。

The images are 128x128 RGB arrays after extraction and resize.

单个 episode 的调用顺序：

Episode sequence:

```text
1. model.infer({reset=True, prompt=task_language})
2. model.infer({obs=first_obs, prompt=task_language})
3. apply returned action chunk in the environment
4. model.infer({
     obs=key_frame_list,
     compute_kv_cache=True,
     imagine=False,
     state=action
   })
5. repeat until done or env timestep reaches 800
```

返回的 action 需要满足：

The returned action is expected to satisfy:

```text
action.shape[2] % 4 == 0
```

随后 client 在 frame/action-step 的双层循环中把所有动作逐步应用到环境。

The client then applies all actions in nested frame/action-step loops.

## Server 模型组件 / Server Model Components

`VA_Server.__init__` 从模型目录加载四个主要组件：

`VA_Server.__init__` loads four major components from the model directory:

```text
tokenizer/      -> T5TokenizerFast
text_encoder/  -> UMT5EncoderModel
vae/           -> diffusers AutoencoderKLWan
transformer/   -> WanTransformer3DModel
```

server 还会创建两个 flow scheduler：

The server also creates two flow schedulers:

```text
scheduler        = FlowMatchScheduler(shift=snr_shift)
action_scheduler = FlowMatchScheduler(shift=action_snr_shift)
```

LIBERO 相关的重要配置：

Important LIBERO config values:

```text
height = 128
width = 128
frame_chunk_size = 4
action_dim = 30
action_per_frame = 4
used_action_channel_ids = [0, 1, 2, 3, 4, 5, 6]
num_inference_steps = 20
action_num_inference_steps = 50
guidance_scale = 5
action_guidance_scale = 1
patch_size = (1, 2, 2)
param_dtype = bfloat16
```

动作归一化使用 `va_libero_cfg.norm_stat` 中的 quantiles。模型内部预测 30 个 action channel，但反归一化后只把前 7 维返回给 LIBERO。

Action normalization uses quantiles from `va_libero_cfg.norm_stat`. The model internally predicts 30 action channels, but only the first 7 are sent back to LIBERO after denormalization.

## Server 推理流程 / Server Inference Flow

### Reset

`VA_Server._reset(prompt)`：

`VA_Server._reset(prompt)`:

```text
1. clears transformer and streaming VAE cache
2. computes latent dimensions
3. creates transformer KV cache
4. builds action mask and normalization tensors
5. encodes the language prompt with tokenizer + UMT5 text encoder
```

普通 LIBERO 双相机配置下：

For normal LIBERO config with two cameras:

```text
latent_height = height / 16 = 8
latent_width  = width / 16 * 2 = 16
```

### First / Next Action Chunk

`VA_Server._infer(obs, frame_st_id)`：

`VA_Server._infer(obs, frame_st_id)`:

```text
1. If frame_st_id == 0:
     encode current observations with streaming VAE
     save as init_latent

2. Initialize random video latents:
     [1, 48, frame_chunk_size, latent_height, latent_width]

3. Initialize random action latents:
     [1, action_dim, frame_chunk_size, action_per_frame, 1]

4. Run video flow loop:
     default 20 steps + padded final 0 step
     call transformer(..., action_mode=False)
     update video latents with FlowMatchScheduler.step(...)

5. Run action flow loop:
     default 50 steps + padded final 0 step
     call transformer(..., action_mode=True)
     update action latents with action_scheduler.step(...)

6. Zero unused action channels
7. Denormalize and return the selected 7 action channels
```

由于 `guidance_scale=5`，classifier-free guidance 会作用于 video path。action path 的 `action_guidance_scale=1`，通常不做额外 CFG mixing。

Classifier-free guidance applies to the video path because `guidance_scale=5`. The action path has `action_guidance_scale=1`, so it normally does not apply extra CFG mixing.

### KV Cache Update

一个 action chunk 执行完后，client 会把关键帧和刚执行过的 action chunk 发回 server。`_compute_kv_cache(obs)`：

After an action chunk is executed, the client sends key frames and the executed action chunk back to the server. `_compute_kv_cache(obs)`:

```text
1. clears predicted-cache flags
2. encodes the observed key frames through the VAE
3. preprocesses and normalizes executed actions
4. calls transformer(..., update_cache=2, action_mode=False)
5. calls transformer(..., update_cache=2, action_mode=True)
6. advances frame_st_id
```

这也是为什么不能把 LingBot-VA 当成一个无状态 action head 来处理。

This is why the model should not be treated as a stateless action head.

## Transformer 结构 / Transformer Structure

`WanTransformer3DModel` 定义在 `wan_va/modules/model.py`。

`WanTransformer3DModel` is defined in `wan_va/modules/model.py`.

配置级 shape：

Config-level shape:

```text
num_layers = 30
num_attention_heads = 24
attention_head_dim = 128
inner_dim = 3072
ffn_dim = 14336
in_channels = 48
out_channels = 48
action_dim = 30
text_dim = 4096
```

主要层：

Main layers:

```text
patch_embedding_mlp: latent patches -> inner_dim
action_embedder:     action channels -> inner_dim
condition_embedder:  timestep + text projection
condition_embedder_action: copied condition_embedder for action path
blocks[30]:          self-attn + cross-attn + FFN
proj_out:            inner_dim -> latent patch output
action_proj_out:     inner_dim -> action_dim
```

Attention 模式：

Attention modes:

```text
attn_mode = torch      uses torch scaled_dot_product_attention
attn_mode = flashattn  uses flash attention
attn_mode = flex       training-only path with custom block masks
```

README 中明确说明，推理时 `transformer/config.json` 必须使用 `"torch"` 或 `"flashattn"`；`"flex"` 是训练路径，不适用于 eval。

For inference, README says `transformer/config.json` must use `"torch"` or `"flashattn"`. `"flex"` is a training path and is not valid for evaluation.

## vla.cpp 移植边界 / vla.cpp Port Boundary

完整移植大概率需要新增这些部分：

A full port likely needs these new pieces:

```text
src/models/lingbot_va.cpp
scripts/convert_lingbot_va_to_gguf.py
new arch enum / registration in src/arch.h and src/model.cpp
optional src/kernels/lingbot_va/* only if ggml ops are insufficient
```

converter 至少要映射三组权重：

The converter must map at least three weight groups:

```text
text_encoder/    UMT5 encoder
vae/             AutoencoderKLWan encoder path, at minimum
transformer/     WanTransformer3DModel
```

converter tensor 保留策略：

Converter tensor retention policy:

- checkpoint/index 中存在的 tensor 默认保守写入 GGUF，即使当前推理 forward 暂未使用。
- Tensors present in checkpoint/index are written into GGUF conservatively by default, even if the current inference forward path does not use them.
- 对于这种 tensor，命名或注释中标记为 legacy/TBD，避免误认为已确认的 active path。
- Such tensors should be marked as legacy/TBD in names or comments so they are not mistaken for confirmed active-path tensors.
- 只有在完整权重加载、Python reference 和 C++ 数值对齐都确认可安全移除后，才考虑裁剪。
- Only consider pruning them after full-weight loading, Python reference checks, and C++ numerical alignment prove they are safe to remove.

最终 `vla.cpp` 侧需要能够覆盖完整链路：

The final `vla.cpp` side should cover the complete chain:

```text
RGB images
  -> VAE image/video encoder
prompt text or tokens
  -> tokenizer / UMT5 text encoder
image/video latents + prompt embeddings
  -> LingBot-VA world-model Transformer
  -> FlowMatchScheduler denoise loops
  -> KV cache update / reuse
  -> action denormalization
  -> action_chunk response
```

第一个 C++ 里程碑不应该是完整 LIBERO success。更合适的第一个目标是：

The first C++ milestone should not be full LIBERO success. A better milestone is:

```text
Given synthetic/preprocessed tensors and a prompt embedding,
run one WanTransformer3DModel forward in latent mode and action mode,
with output shapes matching Python.
```

之后再依次实现：

Only after that should we implement:

```text
1. FlowMatchScheduler loop
2. action normalization / postprocessing
3. VAE image encoding in C++/ggml
4. text tokenization + UMT5 encoding in C++/ggml
5. KV cache semantics
6. vla-server protocol integration
```

这里的阶段拆分是工程降风险策略，不是最终范围裁剪。UMT5 和 VAE 仍然属于必须补齐并优化的最终目标。

This staged split is an engineering risk-reduction strategy, not a reduction of final scope. UMT5 and VAE remain required final targets and should be implemented and optimized.

## 分阶段执行计划 / Phased Execution Plan

### Phase 0: 代码结构和参考链路固化 / Code Structure and Reference Path Freeze

目标：

Goal:

- 固化 LingBot-VA Python 原版推理链路、关键配置、shape、状态语义。
- Freeze the original LingBot-VA Python inference path, key configs, shapes, and state semantics.
- 不依赖完整权重即可推进，主要基于 config、index、源码和小型 dry-run。
- Progress without complete weights by relying on configs, indexes, source code, and small dry-runs.

交付物：

Deliverables:

```text
LINGBOT_VA_PORT_PLAN.md
LingBot-VA call graph / shape notes
local bug list for original scripts
```

验证方式：

Verification:

- 文档能解释 `reset -> infer -> compute_kv_cache` 的完整控制流。
- The document can explain the full `reset -> infer -> compute_kv_cache` control flow.
- 明确哪些信息必须等完整权重下载后再补。
- It clearly separates what must wait for the full weight download.

状态：

Status:

```text
In progress; current document is the first version.
```

### Phase 1: 权重清单和 GGUF 基线 / Weight Inventory and GGUF Baseline

目标：

Goal:

- 权重下载完成后，建立完整 safetensors tensor inventory。
- After the weights finish downloading, build the full safetensors tensor inventory.
- 先做 FP16/BF16 GGUF 正确性基线，不急着量化。
- Build an FP16/BF16 GGUF correctness baseline before quantization.

交付物：

Deliverables:

```text
scripts/inspect_lingbot_va_weights.py
scripts/convert_lingbot_va_to_gguf.py
LingBot-VA tensor name mapping table
module size / dtype / shape report
```

关键检查：

Key checks:

- `text_encoder/`、`vae/`、`transformer/` 的 tensor 名称、shape、dtype 全部可枚举。
- Tensors under `text_encoder/`, `vae/`, and `transformer/` can be fully enumerated with names, shapes, and dtypes.
- GGUF metadata 能被 `src/model.cpp` 检测出 `lingbot_va` 架构。
- GGUF metadata can be detected by `src/model.cpp` as the `lingbot_va` architecture.
- GGUF tensor 命名和 C++ loader 设计一一对应。
- GGUF tensor names match the C++ loader design.

风险：

Risks:

- LingBot-VA 权重分布在多个 diffusers/transformers 子模块中，不能简单当成单个 `model.safetensors`。
- LingBot-VA weights are spread across multiple diffusers/transformers submodules and cannot be treated as one simple `model.safetensors`.
- VAE 和 UMT5 的 tensor layout 可能需要额外转置或重命名。
- VAE and UMT5 tensor layouts may need extra transpose or renaming steps.

当前进展：

Current progress:

- 已新增 `scripts/inspect_lingbot_va_weights.py`，可在完整权重下载前读取 config/index，并报告缺失 shard。
- Added `scripts/inspect_lingbot_va_weights.py`, which can read config/index files before the full weights are downloaded and report missing shards.
- 当前 snapshot 中 `text_encoder` 和 `transformer` 已有 config/index，但 safetensors shards 尚未出现在 snapshot 树下。
- The current snapshot has config/index files for `text_encoder` and `transformer`, but safetensors shards are not yet present in the snapshot tree.
- 当前 snapshot 中 `vae/config.json` 已存在，但 VAE safetensors 文件尚未出现；`tokenizer/` 目录也尚未出现。
- The current snapshot has `vae/config.json`, but no VAE safetensors file yet; the `tokenizer/` directory is also not present yet.

当前 inspector 输出摘要：

Current inspector output summary:

```text
text_encoder:
  architecture: UMT5EncoderModel
  model_type: umt5
  num_layers: 24
  d_model: 4096
  expected tensors: 242
  expected size: 10.58 GiB
  expected shards: 3, currently missing: 3

transformer:
  class: WanTransformer3DModel
  num_layers: 30
  heads: 24
  head_dim: 128
  ffn_dim: 14336
  expected tensors: 841
  expected size: 9.48 GiB
  expected shards: 3, currently missing: 3
  config attn_mode: flex, must be changed to torch or flashattn for inference

vae:
  class: AutoencoderKLWan
  in_channels: 12
  out_channels: 12
  patch_size: 2
  safetensors files: none found yet

tokenizer:
  directory: missing in current snapshot
```

### Phase 2: 最小 C++ 主干 Forward / Minimal C++ Backbone Forward

目标：

Goal:

- 在 C++/ggml 中实现 `WanTransformer3DModel` 的最小 forward。
- Implement the minimal `WanTransformer3DModel` forward in C++/ggml.
- 初期可以接受 Python 预计算的 prompt embeddings 和 image/video latents。
- Initially allow Python-precomputed prompt embeddings and image/video latents.

范围：

Scope:

```text
patch_embedding_mlp
action_embedder
time embedding / timestep projection
3D rotary embedding
self-attention
cross-attention
FFN
proj_out
action_proj_out
```

交付物：

Deliverables:

```text
src/models/lingbot_va.cpp
arch registration in src/arch.h and src/model.cpp
minimal synthetic-input test path
```

验证方式：

Verification:

- 输入 synthetic tensors，C++ latent-mode 和 action-mode 输出 shape 与 Python 一致。
- With synthetic tensors, C++ latent-mode and action-mode output shapes match Python.
- 使用 Python dump 的中间 tensor 做小规模数值对齐。
- Use Python-dumped intermediate tensors for small numerical comparisons.

风险：

Risks:

- LingBot-VA 的 3D RoPE、timestep conditioning、cache update 语义比普通 LLM 更复杂。
- LingBot-VA's 3D RoPE, timestep conditioning, and cache update semantics are more complex than a normal LLM.
- ggml 中若缺少某些 reshape/permute/attention 组合的高效路径，需要额外优化。
- If ggml lacks efficient paths for certain reshape/permute/attention combinations, extra optimization may be needed.

### Phase 3: Flow Loop、Action Path 和状态管理 / Flow Loop, Action Path, and State Management

目标：

Goal:

- 在 C++ 侧实现 LingBot-VA 的 video flow loop、action flow loop、action normalization/postprocess。
- Implement LingBot-VA's video flow loop, action flow loop, and action normalization/postprocess on the C++ side.
- 设计 vla.cpp 风格的 per-session state，用于保存 prompt embeddings、init latent、frame id、KV cache。
- Design vla.cpp-style per-session state for prompt embeddings, init latent, frame id, and KV cache.

交付物：

Deliverables:

```text
FlowMatchScheduler C++ implementation
LingBot-VA runtime state struct
action denormalization path
stateful predict path
```

协议问题：

Protocol issue:

- 现有 `PredictRequest` 没有显式 `reset` 或 `compute_kv_cache` 字段。
- The current `PredictRequest` has no explicit `reset` or `compute_kv_cache` field.
- 需要在完成 state 设计后决定是否扩展 `src/serving/vla.proto`。
- After state design, decide whether to extend `src/serving/vla.proto`.

验证方式：

Verification:

- 用 Python 预计算 latents/prompt embeddings，C++ 输出 action chunk shape 和范围正确。
- With Python-precomputed latents/prompt embeddings, C++ action chunk shape and value range are correct.
- 同一随机种子下，对齐若干步 flow update 的中间输出。
- With the same random seed, compare intermediate outputs for several flow update steps.

### Phase 4: UMT5 C++/ggml 支持 / UMT5 C++/ggml Support

目标：

Goal:

- 将 `text_encoder/` 的 UMT5 encoder 接入 vla.cpp。
- Bring the `text_encoder/` UMT5 encoder into vla.cpp.
- 让 C++ 服务从 token 或 prompt 侧生成 prompt embeddings。
- Let the C++ service generate prompt embeddings from tokens or prompt-side inputs.

交付物：

Deliverables:

```text
UMT5 tensor mapping
UMT5 encoder graph in C++/ggml
tokenizer integration strategy
prompt embedding cache
```

设计原则：

Design principles:

- UMT5 只在 reset/prompt 变化时执行，不是每个 env step 都高频执行。
- UMT5 runs only on reset or prompt changes, not at every environment step.
- 可以优先考虑 CPU/offload 或更激进量化，以减少 8 GB GPU 压力。
- CPU/offload or more aggressive quantization can be considered first to reduce 8 GB GPU pressure.

验证方式：

Verification:

- 同一 prompt 下，C++ prompt embeddings 与 Python UMT5 输出做数值对齐。
- For the same prompt, compare C++ prompt embeddings against Python UMT5 output.

### Phase 5: VAE Encoder C++/ggml 支持 / VAE Encoder C++/ggml Support

目标：

Goal:

- 将 `vae/` 的 AutoencoderKLWan encoder 路径接入 vla.cpp。
- Bring the `vae/` AutoencoderKLWan encoder path into vla.cpp.
- 优先实现推理所需的 image/video encoding；decoder 可后置。
- Prioritize image/video encoding needed for inference; decoder can be deferred.

交付物：

Deliverables:

```text
VAE encoder tensor mapping
WanVAEStreamingWrapper equivalent state
image/video preprocessing path
VAE latent normalization path
```

设计原则：

Design principles:

- LIBERO 推理主要需要 encoder，不一定一开始实现 decoder。
- LIBERO inference mainly needs the encoder; decoder is not required at first.
- VAE 是卷积/3D 卷积密集模块，量化策略应和 Transformer 分开设计。
- VAE is convolution/3D-convolution heavy, so its quantization strategy should be separate from the Transformer.
- 可考虑 VAE CPU/offload、分块执行、FP16 保留敏感层等策略。
- Consider VAE CPU/offload, chunked execution, and keeping sensitive layers in FP16.

验证方式：

Verification:

- 同一组 LIBERO 双相机图像，C++ VAE latent 与 Python `_encode_obs()` 输出对齐。
- For the same LIBERO two-camera images, compare C++ VAE latents against Python `_encode_obs()` output.

### Phase 6: 8GB 部署优化和量化 / 8 GB Deployment Optimization and Quantization

目标：

Goal:

- 在本地 RTX 3070 Ti Laptop 8 GB VRAM 上跑通真实 end-to-end。
- Run real end-to-end inference on the local RTX 3070 Ti Laptop with 8 GB VRAM.
- 在压缩倍率和成功率之间取得平衡，目标整体损失约 3%-5% 以内。
- Balance compression ratio and success rate, targeting roughly 3%-5% overall loss.

候选策略：

Candidate strategies:

```text
Transformer: start with Q8 / mixed Q6-Q8, then test Q4 on less sensitive layers
UMT5: cache prompt embeddings; consider CPU/offload or aggressive quantization
VAE: encoder-only, chunked execution, CPU/offload, selective FP16
KV cache: bounded window, low-memory allocation, possible lower precision
Runtime: module-by-module loading/offload if needed
```

交付物：

Deliverables:

```text
memory profiling report
quantization comparison table
8 GB runnable model recipe
recommended GGUF variants
```

验证方式：

Verification:

- 记录 peak VRAM、latency、LIBERO success rate、与 BF16 baseline 的差异。
- Record peak VRAM, latency, LIBERO success rate, and difference from BF16 baseline.
- 至少覆盖 smoke test 和小规模 LIBERO benchmark。
- Cover at least smoke tests and a small LIBERO benchmark.

### Phase 7: 报告化和发布整理 / Report and Release Preparation

目标：

Goal:

- 将工程贡献整理成 technical report 的可写内容。
- Turn engineering contributions into technical-report material.
- 形成可复现安装、转换、运行、评测流程。
- Provide reproducible install, conversion, run, and evaluation workflows.

潜在贡献点：

Potential contributions:

```text
vla.cpp world-model runtime extension
UMT5 and VAE execution support in C++/ggml
low-VRAM deployment strategy for video-action world models
module-wise quantization/offload recipe
ZMQ/protobuf integration for stateful world-model inference
```

交付物：

Deliverables:

```text
technical report outline
reproducible scripts
model conversion docs
evaluation notes
release checklist
```

## 近期执行顺序 / Near-Term Execution Order

在完整权重下载期间，可以先做这些不依赖大 safetensors 的工作：

While the full weights are downloading, we can do the following work that does not depend on large safetensors:

```text
1. Write a compact LingBot-VA shape/spec note from config and source.
2. Draft scripts/inspect_lingbot_va_weights.py against index/config files.
3. Draft scripts/convert_lingbot_va_to_gguf.py metadata structure.
4. Add placeholder arch registration plan for lingbot_va, without compiling it yet.
5. Design the minimal C++ forward test interface.
6. Decide the proto extension only after stateful runtime design is concrete.
```

第一批真正改代码的建议顺序：

Suggested order for the first real code changes:

```text
1. Add a non-invasive weight/config inspector.
2. Add GGUF metadata writer skeleton for lingbot_va.
3. Add architecture detection support for lingbot_va.
4. Add src/models/lingbot_va.cpp skeleton that fails loudly with "not implemented".
5. Fill in WanTransformer3DModel minimal forward layer by layer.
```

这样做的好处是每一步都可验证、可回滚，也能持续积累 report 中需要的工程证据。

This keeps each step verifiable and reversible, while continuously collecting engineering evidence for the report.

## 已知本地问题 / Known Local Issues

`evaluation/libero/launch_server.sh` 疑似存在换行错误：

`evaluation/libero/launch_server.sh` appears to have a line break bug:

```bash
--save_root $save_rootSTART=0
```

更可能应该是：

This should likely be:

```bash
--save_root $save_root

START=0
```

当我们决定运行原版 Python reference 时，应直接修复这个问题。

We should fix this when we decide to run the original Python reference.

## 硬件风险 / Hardware Risk

发布的 LingBot-VA 模型明显大于 SmolVLA/pi0。README 中的显存建议大致是：

The released model is much larger than SmolVLA/pi0. README memory guidance says roughly:

```text
18 GB VRAM for image-to-video-action inference with offload
24 GB VRAM for RoboTwin evaluation with offload
```

本地 RTX 3070 Ti Laptop GPU 只有 8 GB VRAM，所以原版 Python LIBERO 推理大概率无法在本机直接跑通，除非做大量 CPU offload、量化或专门的小型 debug path。与此同时，本项目的第一真实运行目标已经明确为本地 8 GB GPU 可运行，因此低显存部署不是附加优化，而是核心设计约束。

The local RTX 3070 Ti Laptop GPU has 8 GB VRAM, so original Python LIBERO inference is unlikely to run locally without substantial CPU offload, quantization, or a smaller debug path. At the same time, the first real run target is explicitly local 8 GB GPU execution, so low-memory deployment is not an optional optimization but a core design constraint.

## 待决策事项 / Pending Decisions

当前核心路线已定。后续待决策项会随着权重清单、baseline forward 和显存 profiling 的结果继续补充。

The core route is currently decided. Future pending decisions will be added as tensor inventory, baseline forward, and memory profiling results become available.

## 已定事项 / Decided Items

- 通讯协议主线：使用 vla.cpp 现有 ZMQ/protobuf。原始 WebSocket 只作为参考。
- Main communication protocol: use vla.cpp's existing ZMQ/protobuf path. The original WebSocket path is reference-only.
- 最终范围：vla.cpp 需要完整承载 UMT5、VAE 和 LingBot-VA 世界模型主干；Python 预计算只用于阶段性对齐。
- Final scope: vla.cpp should fully host UMT5, VAE, and the LingBot-VA world-model backbone; Python precomputation is only for staged alignment.
- 第一次真实 end-to-end 运行目标：本机 RTX 3070 Ti Laptop 8 GB VRAM 可运行。
- First real end-to-end run target: runnable on the local RTX 3070 Ti Laptop with 8 GB VRAM.
- 量化目标：以本地 8 GB 可运行为硬约束，在压缩倍率和精度损失之间平衡，尽量将整体损失控制在约 3%-5% 以内；具体策略由实验结果驱动。
- Quantization target: use local 8 GB execution as a hard constraint, balance compression ratio and accuracy loss, and try to keep overall loss within roughly 3%-5%; the concrete strategy should be driven by experiments.

## 等权重下载完成后再做 / Deferred Until Weights Finish Downloading

- 完整 safetensors tensor inventory。
- Full safetensors tensor inventory.
- Python module tensor name 到 GGUF name 的映射表。
- Tensor-name mapping table from Python modules to GGUF names.
- 按 dtype 和 module 估算精确内存占用。
- Exact memory estimate by dtype and module.
- Python 与 C++ forward pass 的数值对齐比较。
- Numerical comparison between Python and C++ forward passes.
