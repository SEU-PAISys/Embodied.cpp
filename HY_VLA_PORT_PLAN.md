# Hy-Embodied-0.5-VLA vla.cpp 移植规划 / Hy-Embodied-0.5-VLA vla.cpp Port Plan

本文档记录 Hy-Embodied-0.5-VLA 接入 `vla.cpp` 的模型理解、工程决策和分阶段路线。执行记录和实验结果写入 `HY_VLA_PROGRESS.md`。

This document records the model understanding, engineering decisions, and staged route for integrating Hy-Embodied-0.5-VLA into `vla.cpp`. Execution logs and experiment results live in `HY_VLA_PROGRESS.md`.

## 当前目标 / Current Goal

中文：

- 将 `/home/xuling/robotic_dataset/HY-VLA` 中的完整 Hy-VLA checkpoint 转换为 `vla.cpp` 可加载的 C++/GGML/GGUF 形态。
- 在 `vla.cpp` 现有 ZMQ/protobuf server 框架内运行 Hy-VLA 推理，优先复现 RoboTwin 风格的动作 chunk 输出。
- 先建立 Python reference parity，再逐步替换为 C++ 模块，避免一开始就把 VLM、vision、MEM、flow、action decode 全部耦合在一起调试。
- 最终目标是在本机 8GB GPU 环境上跑通可用 smoke，并进一步推进量化/offload，使其有机会进入完整评测链路。

English:

- Convert the full Hy-VLA checkpoint under `/home/xuling/robotic_dataset/HY-VLA` into a `vla.cpp`-loadable C++/GGML/GGUF form.
- Run Hy-VLA inference inside the existing `vla.cpp` ZMQ/protobuf server framework, prioritizing RoboTwin-style action chunk output.
- Establish Python reference parity first, then replace modules with C++ one by one, instead of debugging VLM, vision, MEM, flow, and action decode all at once.
- The final target is a usable smoke path on the local 8GB GPU, followed by quantization/offload work toward full evaluation.

## 本地资源 / Local Resources

源码仓库：

Source repository:

```text
/home/xuling/robotic_code/embodied.cpp/Hy-Embodied-0.5-VLA-main
```

模型 checkpoint：

Model checkpoint:

```text
/home/xuling/robotic_dataset/HY-VLA
```

checkpoint 文件：

Checkpoint files:

```text
config.json
model.safetensors          # about 9.05 GB
norm_stats.pkl
tokenizer.json
tokenizer_config.json
special_tokens_map.json
chat_template.jinja
preprocessor_config.json
```

关键配置快照：

Key config snapshot:

```text
chunk_size = 40
n_action_steps = 40
max_state_dim = 32
max_action_dim = 32
resize_imgs_with_padding = [224, 224]
tokenizer_max_length = 64
proj_width = 1024
num_steps = 10
use_cache = true
attention_implementation = eager
use_video_encoder = true
spacetime_layer_stride = 4
past_drop_layer = null
visual_segment_isolation = false

image_features:
  observation.images.top_head
  observation.images.hand_left
  observation.images.hand_right

VLM text_config:
  hidden_size = 2048
  num_hidden_layers = 32
  num_attention_heads = 16
  num_key_value_heads = 4
  vocab_size = 120818

Action expert:
  hidden_size = proj_width = 1024
  intermediate_size = 2048
```

## 模型结构理解 / Model Structure Understanding

Hy-VLA 不是 LingBot-VA 那种 world/video latent diffusion backbone，而是：

Hy-VLA is not a LingBot-VA-style world/video latent diffusion backbone. It is:

```text
multi-view RGB images + language + robot state
        │
        ├── Hy-Embodied-0.5 / HunYuan-VL-MoT VLM tower
        │       images -> visual tokens
        │       language -> text tokens
        │       MoT text/vision routed transformer
        │
        ├── MEM compact memory encoder
        │       K-frame per-camera history -> current-frame visual tokens
        │
        └── action expert tower
                state token + noisy action tokens + timestep
                │
                ▼
        dual-tower shared attention
                │
                ▼
        flow matching Euler sampler
                │
                ▼
        normalized continuous action chunk
                │
                ▼
        norm_stats / relative pose decode -> robot action
```

核心源码：

Core source files:

```text
hy_vla/modeling_hy_vla.py
hy_vla/modeling_dual_tower.py
hy_vla/space_time_attention.py
robotwin_eval/policy_wrapper.py
robotwin_eval/transforms.py
```

### Prefix / Suffix 语义 / Prefix and Suffix Semantics

中文：

- `prefix` 是图像和语言上下文：特殊 tokens、多视角视觉 tokens、语言 tokens。
- `suffix` 是动作生成段：state token、noisy action tokens、timestep embedding。
- 推理时先对 prefix 做一次 forward 并写 KV cache；flow sampling 的 10 个 Euler steps 只重复构造 suffix，并 attend 到 prefix cache。

English:

- `prefix` is the image/language context: special tokens, multi-view visual tokens, and language tokens.
- `suffix` is the action-generation segment: state token, noisy action tokens, and timestep embedding.
- In inference, prefix is forwarded once into a KV cache; the 10 Euler flow steps rebuild only the suffix and attend to the cached prefix.

### Dual-Tower Shared Attention / 双塔共享 Attention

中文：

- VLM tower hidden size 通常为 2048。
- action expert hidden size 为 1024。
- 每层先分别做 VLM/action 的 layernorm 和 QKV projection，再把 Q/K/V 在 sequence 维拼接。
- RoPE 和 attention 是共享计算；attention 输出再按 prefix/suffix 切回各自 tower，分别走各自 `o_proj` 和 MLP。
- MoT 参数路由依赖 modality mask：text token 走 text 参数，vision token 走 vision 参数。

English:

- The VLM tower hidden size is typically 2048.
- The action expert hidden size is 1024.
- Each layer first applies tower-specific layernorm and QKV projections, then concatenates Q/K/V along the sequence dimension.
- RoPE and attention are shared; the attention output is split back into each tower and passed through each tower's `o_proj` and MLP.
- MoT routing depends on the modality mask: text tokens use text parameters and vision tokens use vision parameters.

## Server 接入决策 / Server Integration Decision

中文：

- 主线使用 `vla.cpp-main/src/serving/server.cpp` 和现有 `src/serving/vla.proto`。
- 当前 `PredictRequest` 已经能表达：
  - images
  - lang_tokens
  - state
  - noise
  - request_id
  - precomputed_img_emb
  - attention_mask
- Hy-VLA 的第一阶段可以复用这条协议：
  - `images`: 按固定顺序传 `top_head / hand_left / hand_right` 当前帧；
  - `state`: 传 32 维 padded/normalized state；
  - `lang_tokens`: 传 tokenizer 后的 prompt tokens；
  - `noise`: 可选，用于 parity 固定 flow 初始噪声。
- 但 MEM video encoder 需要 K-frame history。现有 server 只有 image list，没有 camera/history metadata。短期可以约定固定顺序 `camera0_t0..tK, camera1_t0..tK, camera2_t0..tK`，或在 `Inputs` 增加 Hy-VLA 专用 image-history view；如果后续需要通用表达，再扩展 proto。

English:

- The main route uses `vla.cpp-main/src/serving/server.cpp` and the existing `src/serving/vla.proto`.
- Current `PredictRequest` can already express:
  - images
  - lang_tokens
  - state
  - noise
  - request_id
  - precomputed_img_emb
  - attention_mask
- The first Hy-VLA stage can reuse this protocol:
  - `images`: current frames in fixed `top_head / hand_left / hand_right` order;
  - `state`: 32-D padded/normalized state;
  - `lang_tokens`: tokenizer output prompt tokens;
  - `noise`: optional fixed flow initial noise for parity.
- MEM video encoder needs K-frame history. The existing server has an image list but no camera/history metadata. Short term, use a fixed order such as `camera0_t0..tK, camera1_t0..tK, camera2_t0..tK`, or add a Hy-VLA-specific image-history view in `Inputs`; extend proto only if a generic expression becomes necessary.

## 分阶段路线 / Staged Route

### 阶段 0：环境和 Python baseline / Stage 0: Environment and Python Baseline

中文：

- 建立 Hy-VLA conda/uv 环境，确认可以 import `hy_vla`。
- 跑 `scripts/quick_start.py` 或等价最小 smoke。
- 固定一个 deterministic fixture：
  - 3 路 RGB 输入；
  - 32 维 state；
  - prompt；
  - fixed Gaussian action noise。
- 输出 Python reference trace：
  - tokenizer ids；
  - image preprocessing 后张量；
  - prefix embeddings；
  - prefix KV cache；
  - suffix embeddings；
  - 每步 Euler `x_t` / `v_t`；
  - final normalized action chunk；
  - postprocessed RoboTwin action。

English:

- Build the Hy-VLA conda/uv environment and confirm `hy_vla` imports.
- Run `scripts/quick_start.py` or an equivalent minimal smoke.
- Create a deterministic fixture:
  - 3 RGB views;
  - 32-D state;
  - prompt;
  - fixed Gaussian action noise.
- Dump Python reference traces:
  - tokenizer ids;
  - preprocessed image tensors;
  - prefix embeddings;
  - prefix KV cache;
  - suffix embeddings;
  - per-Euler-step `x_t` / `v_t`;
  - final normalized action chunk;
  - postprocessed RoboTwin action.

### 阶段 1：权重审计和 GGUF 转换 / Stage 1: Weight Audit and GGUF Conversion

中文：

- 写 `scripts/inspect_hy_vla_weights.py`，读取 `model.safetensors` 的 tensor 名称、shape、dtype、模块归属。
- 写 `scripts/convert_hy_vla_to_gguf.py`，先转 metadata-only，再转完整 tensor。
- 初始 GGUF 可以按模块拆分，降低调试复杂度：
  - VLM text/MoT tower；
  - vision tower / HYViT2；
  - action expert；
  - action/state projection heads；
  - norm stats。
- 保留 safetensors -> GGUF tensor name mapping，便于 parity 对账。

English:

- Add `scripts/inspect_hy_vla_weights.py` to read tensor names, shapes, dtypes, and module ownership from `model.safetensors`.
- Add `scripts/convert_hy_vla_to_gguf.py`, starting with metadata-only and then full tensors.
- Initial GGUFs may be split by module to reduce debugging complexity:
  - VLM text/MoT tower;
  - vision tower / HYViT2;
  - action expert;
  - action/state projection heads;
  - norm stats.
- Keep a safetensors-to-GGUF tensor-name map for parity accounting.

### 阶段 2：Action Expert Suffix First / Stage 2: Action Expert Suffix First

中文：

- 先不接 vision tower，使用 Python 预计算 prefix KV cache 或 synthetic prefix。
- 在 C++ 中实现：
  - state projection；
  - action input projection；
  - sinusoidal timestep embedding；
  - action-time MLP；
  - expert tower layers；
  - `action_out_proj`；
  - 10-step Euler flow loop。
- 目标：固定 prefix KV + fixed noise 下，C++ `v_t` 和 final normalized action chunk 对齐 Python。

English:

- Do not wire the vision tower first. Use Python-precomputed prefix KV cache or a synthetic prefix.
- Implement in C++:
  - state projection;
  - action input projection;
  - sinusoidal timestep embedding;
  - action-time MLP;
  - expert tower layers;
  - `action_out_proj`;
  - 10-step Euler flow loop.
- Goal: with fixed prefix KV and fixed noise, align C++ `v_t` and final normalized action chunk with Python.

### 阶段 3：Dual-Tower Prefix KV / Stage 3: Dual-Tower Prefix KV

中文：

- 实现 VLM text embedding 和 MoT decoder text/vision routing 的 shared-attention forward。
- 先支持 text-only prefix，再接 precomputed visual tokens。
- 实现 prefix KV cache：
  - `fill_kv_cache=true` 路径；
  - denoise step 中 suffix attend prefix cache；
  - 避免每个 Euler step 重算图像和语言。
- 目标：prefix hidden / KV cache parity。

English:

- Implement VLM text embedding and MoT decoder text/vision routed shared-attention forward.
- Support text-only prefix first, then precomputed visual tokens.
- Implement prefix KV cache:
  - `fill_kv_cache=true` path;
  - suffix attending to prefix cache in denoise steps;
  - avoid recomputing image/language for every Euler step.
- Goal: prefix hidden / KV cache parity.

### 阶段 4：Vision Tower 和 MEM / Stage 4: Vision Tower and MEM

中文：

- 实现 Hy-ViT2 image preprocessing 和 vision encoder。
- 先支持单帧 3-view path。
- 再实现 MEM video encoder：
  - K-frame `[B,K,C,H,W]` 输入；
  - SpaceTimeBlock；
  - causal time attention；
  - spatial attention；
  - 只保留 current-frame tokens。
- 如果 MEM 太大，允许先通过 precomputed visual tokens 建立 server/action 闭环。

English:

- Implement Hy-ViT2 image preprocessing and vision encoder.
- Support single-frame 3-view path first.
- Then implement MEM video encoder:
  - K-frame `[B,K,C,H,W]` input;
  - SpaceTimeBlock;
  - causal time attention;
  - spatial attention;
  - keep only current-frame tokens.
- If MEM is too large, use precomputed visual tokens first to close the server/action loop.

### 阶段 5：Action Postprocess 和 RoboTwin Client / Stage 5: Action Postprocess and RoboTwin Client

中文：

- 读取 `norm_stats.pkl` 或转换后的 GGUF metadata。
- 实现：
  - qpos normalization；
  - action unnormalization；
  - 20D relative action -> 16D dual-arm pose；
  - quaternion layout conversion。
- 添加或适配 RoboTwin client，让 `server.cpp` 返回的 chunk 能被 simulator 消费。

English:

- Read `norm_stats.pkl` or converted GGUF metadata.
- Implement:
  - qpos normalization;
  - action unnormalization;
  - 20-D relative action -> 16-D dual-arm pose;
  - quaternion layout conversion.
- Add or adapt a RoboTwin client so chunks returned by `server.cpp` can be consumed by the simulator.

### 阶段 6：Server 集成 / Stage 6: Server Integration

中文：

- 新增 `src/models/hy_vla.cpp` 并注册到 `model_load()`。
- 优先复用 `src/serving/server.cpp`，只在必要时扩展 `vla::Inputs` 或 proto。
- smoke 顺序：
  1. text/state/noise only；
  2. precomputed visual tokens；
  3. single-frame 3-view images；
  4. K-frame MEM history；
  5. RoboTwin short rollout。

English:

- Add `src/models/hy_vla.cpp` and register it in `model_load()`.
- Reuse `src/serving/server.cpp` first, extending `vla::Inputs` or proto only when necessary.
- Smoke order:
  1. text/state/noise only;
  2. precomputed visual tokens;
  3. single-frame 3-view images;
  4. K-frame MEM history;
  5. RoboTwin short rollout.

### 阶段 7：低显存和量化 / Stage 7: Low-Memory and Quantization

中文：

- 9.05GB safetensors 不能假设全量常驻 8GB GPU。
- 借鉴 LingBot resident/window 策略：
  - CPU resident；
  - per-layer/per-block streaming；
  - Q8_0 baseline；
  - Q6/Q4 试验；
  - prefix KV cache 压缩；
  - action expert 优先常驻，vision/VLM 分阶段执行。
- 精度目标仍以整体损失约 3%-5% 作为工程参考，最终用 RoboTwin/LIBERO 类任务成功率校准。

English:

- The 9.05GB safetensors cannot be assumed to fit fully resident on an 8GB GPU.
- Reuse ideas from the LingBot resident/window strategy:
  - CPU resident;
  - per-layer/per-block streaming;
  - Q8_0 baseline;
  - Q6/Q4 experiments;
  - compressed prefix KV cache;
  - keep the action expert resident first, stage vision/VLM execution.
- Keep roughly 3%-5% total loss as an engineering reference, calibrated later by RoboTwin/LIBERO-style success rates.

## 初始文件计划 / Initial File Plan

中文：

- `HY_VLA_PORT_PLAN.md`：规划和设计决策。
- `HY_VLA_PROGRESS.md`：执行记录和实验结果。
- `scripts/inspect_hy_vla_weights.py`：权重审计。
- `scripts/convert_hy_vla_to_gguf.py`：GGUF 转换。
- `scripts/run_hy_vla_e2e_parity.py`：Python/C++ parity harness。
- `src/models/hy_vla.cpp`：C++ 模型实现。
- `eval/client/hy_vla_client.py` 或复用现有 client adapter：RoboTwin/vla-server 接入。

English:

- `HY_VLA_PORT_PLAN.md`: planning and design decisions.
- `HY_VLA_PROGRESS.md`: execution logs and experiment results.
- `scripts/inspect_hy_vla_weights.py`: weight audit.
- `scripts/convert_hy_vla_to_gguf.py`: GGUF conversion.
- `scripts/run_hy_vla_e2e_parity.py`: Python/C++ parity harness.
- `src/models/hy_vla.cpp`: C++ model implementation.
- `eval/client/hy_vla_client.py` or reuse an existing client adapter: RoboTwin/vla-server integration.

## 当前风险 / Current Risks

中文：

- Hy-VLA 依赖指定 transformers fork；Python reference 环境必须先稳定。
- vision tower 和 MEM video encoder 比 action expert 更复杂，应避免一开始就把它们放在 parity 主路径里。
- `server.cpp` 可以承载标准推理请求，但 K-frame history 的表达可能需要约定固定顺序或扩展输入结构。
- 本机 8GB GPU 显存不足以直接全量 BF16 常驻，必须早期设计 streaming/quantization。

English:

- Hy-VLA depends on a pinned transformers fork; the Python reference environment must be stabilized first.
- The vision tower and MEM video encoder are more complex than the action expert, so they should not be the first parity bottleneck.
- `server.cpp` can host standard inference requests, but K-frame history may require fixed ordering conventions or input-structure extensions.
- The local 8GB GPU cannot hold the full BF16 model resident, so streaming/quantization must be planned early.
