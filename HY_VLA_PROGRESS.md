# Hy-Embodied-0.5-VLA vla.cpp 执行进度 / Hy-Embodied-0.5-VLA vla.cpp Progress Log

本文档记录 Hy-Embodied-0.5-VLA 接入 `vla.cpp` 的实际执行过程。规划和设计决策见 `HY_VLA_PORT_PLAN.md`；本文只记录已经做过的动作、结果、问题和下一步。

This document records the actual execution progress of integrating Hy-Embodied-0.5-VLA into `vla.cpp`. Planning and design decisions live in `HY_VLA_PORT_PLAN.md`; this file records completed actions, results, issues, and next steps.

## 当前执行顺序 / Current Execution Order

更新时间：2026-06-17 CST

Updated: 2026-06-17 CST

- [x] 1. 阅读 Hy-VLA 源码并建立模型结构理解。
- [x] 1. Read the Hy-VLA source and establish the model-structure understanding.
- [x] 2. 确认本地 checkpoint 布局和关键 config。
- [x] 2. Confirm the local checkpoint layout and key config values.
- [x] 3. 建立 Python reference 环境并跑通 deterministic fixture / CPU load smoke。（完整 forward 尚未跑。）
- [x] 3. Build the Python reference environment and run deterministic fixture / CPU load smoke. (Full forward has not been run yet.)
- [x] 4. 编写 safetensors 权重审计脚本，生成 tensor inventory。
- [x] 4. Add a safetensors weight-audit script and generate a tensor inventory.
- [x] 5. 编写 GGUF 转换脚本，先 metadata-only，再模块化 tensor 转换。
- [x] 5. Add a GGUF conversion script, starting with metadata-only and then modular tensor conversion.
- [x] 6. 搭建 action expert suffix-first C++ parity path。
- [x] 6. Build the action-expert suffix-first C++ parity path.
- [x] 7. 接入 dual-tower prefix KV cache 的 text-only debug/parity 路径。
- [x] 7. Wire the text-only debug/parity path for the dual-tower prefix KV cache.
- [ ] 8. 接入 vision tower / MEM video encoder。
- [ ] 8. Wire the vision tower / MEM video encoder.
- [ ] 9. 复用 `src/serving/server.cpp` 打通 vla.cpp server 请求链路。
- [ ] 9. Reuse `src/serving/server.cpp` to close the vla.cpp server request path.
- [ ] 10. 推进 RoboTwin smoke、低显存执行和量化。
- [ ] 10. Proceed to RoboTwin smoke, low-memory execution, and quantization.

## 2026-06-17 - Initial Planning and Checkpoint Audit

中文：

- 确认本地 Hy-VLA 源码路径：

```text
/home/xuling/robotic_code/embodied.cpp/Hy-Embodied-0.5-VLA-main
```

- 确认本地 checkpoint 路径：

```text
/home/xuling/robotic_dataset/HY-VLA
```

- checkpoint 是自包含布局，包含：

```text
config.json
model.safetensors
norm_stats.pkl
tokenizer.json
tokenizer_config.json
special_tokens_map.json
chat_template.jinja
preprocessor_config.json
```

- `model.safetensors` 大小约 9.05GB。
- 从 `config.json` 读取到关键配置：

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
image_features = top_head / hand_left / hand_right
VLM text hidden = 2048
VLM text layers = 32
VLM text heads = 16
VLM text KV heads = 4
VLM vocab = 120818
Action expert hidden = 1024
Action expert FFN = 2048
```

- 初步 server 决策：
  - 优先复用 `src/serving/server.cpp`。
  - 当前 `PredictRequest` 已支持 images/lang_tokens/state/noise，足以承载第一阶段 fixed-current-frame smoke。
  - MEM K-frame history 可能需要固定 image ordering 约定，或后续扩展 `vla::Inputs` / proto。

English:

- Confirmed the local Hy-VLA source path:

```text
/home/xuling/robotic_code/embodied.cpp/Hy-Embodied-0.5-VLA-main
```

- Confirmed the local checkpoint path:

```text
/home/xuling/robotic_dataset/HY-VLA
```

- The checkpoint is self-contained and includes:

```text
config.json
model.safetensors
norm_stats.pkl
tokenizer.json
tokenizer_config.json
special_tokens_map.json
chat_template.jinja
preprocessor_config.json
```

- `model.safetensors` is about 9.05GB.
- Key values read from `config.json`:

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
image_features = top_head / hand_left / hand_right
VLM text hidden = 2048
VLM text layers = 32
VLM text heads = 16
VLM text KV heads = 4
VLM vocab = 120818
Action expert hidden = 1024
Action expert FFN = 2048
```

- Initial server decision:
  - Prefer reusing `src/serving/server.cpp`.
  - Current `PredictRequest` already supports images/lang_tokens/state/noise, enough for the first fixed-current-frame smoke.
  - MEM K-frame history may require a fixed image-ordering convention, or a later `vla::Inputs` / proto extension.

下一步：

Next:

- 等用户的 Hy-VLA Python 环境安装完成后，先跑 Python quick_start 或 deterministic fixture。
- Add `scripts/inspect_hy_vla_weights.py` to produce the first tensor inventory.

## 2026-06-17 - Python Environment Repair

中文：

- 用户在 `hy-vla` conda 环境中遇到：
  - `flash-attn` 预编译 wheel 安装后运行期失败；
  - `pip install -e .` 失败；
  - 初始依赖解析把 torch 装成 `2.12.0+cu130`，本机 NVIDIA driver/CUDA 12.8 无法运行 cu130。
- 已完成修复：
  - 修复 Hy-VLA `pyproject.toml` 中不存在的 license 文件名：

```text
License.txt -> LICENSE
```

  - 清理不兼容的 torch/cu13 依赖。
  - 安装 PyTorch CUDA 12.8 匹配版本：

```text
torch 2.7.1+cu128
torchvision 0.22.1+cu128
```

  - 安装核心推理依赖：

```text
numpy 1.26.4
transformers 4.57.0 from commit 9293856c419762ebf98fbe2bd9440f9ce7069f1a
timm 1.0.21
safetensors 0.8.0
huggingface-hub 0.36.2
scipy 1.15.3
opencv-python-headless 4.11.0.86
```

  - 预编译 `flash-attn` wheel 需要 `GLIBC_2.32`，但系统是 `GLIBC_2.31`。源码编译也产生了同样的 GLIBC 依赖。
  - 为本地 Python reference/parity 增加慢速 torch fallback：

```text
hy_vla/utils/torch_flash_attn_fallback.py
hy_vla/hunyuan_vl_mot/modeling_hunyuan_vl_mot.py
hy_vla/space_time_attention.py
```

- 当前验证结果：

```text
HyVLA imports ok
torch 2.7.1+cu128 12.8 True
fallback out (1, 4, 2, 8) torch.bfloat16 True
config ok 40 40 32 32 True
tokenizer ok torch.Size([1, 64])
```

- 说明：
  - 现在环境已满足 “导入 Hy-VLA + 读取本地 config/tokenizer + CUDA torch 可用”。
  - `flash-attn` fallback 是慢速 parity/debug 路径，不是正式性能路径。
  - 尚未运行完整 `HyVLA.from_pretrained()`，因为它会加载约 9GB 权重并可能触发本机 8GB GPU/系统内存压力。

English:

- The user hit several issues in the `hy-vla` conda environment:
  - the prebuilt `flash-attn` wheel installed but failed at runtime;
  - `pip install -e .` failed;
  - dependency resolution initially installed `torch 2.12.0+cu130`, which cannot run on the local NVIDIA driver/CUDA 12.8 setup.
- Completed fixes:
  - Fixed the non-existent license file name in Hy-VLA `pyproject.toml`:

```text
License.txt -> LICENSE
```

  - Removed the incompatible torch/cu13 dependency set.
  - Installed the CUDA 12.8-matching PyTorch stack:

```text
torch 2.7.1+cu128
torchvision 0.22.1+cu128
```

  - Installed core inference dependencies:

```text
numpy 1.26.4
transformers 4.57.0 from commit 9293856c419762ebf98fbe2bd9440f9ce7069f1a
timm 1.0.21
safetensors 0.8.0
huggingface-hub 0.36.2
scipy 1.15.3
opencv-python-headless 4.11.0.86
```

  - The prebuilt `flash-attn` wheel requires `GLIBC_2.32`, while this system has `GLIBC_2.31`. Building from source produced the same GLIBC dependency.
  - Added a slow torch fallback for local Python reference/parity:

```text
hy_vla/utils/torch_flash_attn_fallback.py
hy_vla/hunyuan_vl_mot/modeling_hunyuan_vl_mot.py
hy_vla/space_time_attention.py
```

- Current verification:

```text
HyVLA imports ok
torch 2.7.1+cu128 12.8 True
fallback out (1, 4, 2, 8) torch.bfloat16 True
config ok 40 40 32 32 True
tokenizer ok torch.Size([1, 64])
```

- Notes:
  - The environment now supports importing Hy-VLA, reading local config/tokenizer, and using CUDA torch.
  - The `flash-attn` fallback is a slow parity/debug path, not the final performance path.
- Full `HyVLA.from_pretrained()` has not been run yet because it loads about 9GB of weights and may stress the local 8GB GPU/system memory setup.

## 2026-06-17 - Deterministic Fixture and Python Load Smoke

中文：

- 新增 Python reference harness：

```text
scripts/run_hy_vla_reference.py
```

- 默认 `fixture` 模式不会加载 9GB 模型权重，只做：
  - 导入 Hy-VLA；
  - 读取本地 config；
  - 读取本地 tokenizer；
  - 生成 deterministic fixture；
  - dump prompt tokens / attention mask / state / fixed action noise。
- fixture 内容：

```text
prompt = "pick up the bottle"
history = 6
image_size = 224
images = 3 cameras x 6 frames, deterministic uint8 formula -> float [0,1]
state = linspace(-0.25, 0.25, 32)
noise = sin(i * 0.017) * 0.05, shape [1,40,32]
```

- fixture smoke 命令：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_reference.py \
  --mode fixture \
  --out-dir /tmp/hy_vla_reference_fixture
```

- fixture 结果：

```text
chunk_size = 40
n_action_steps = 40
max_state_dim = 32
max_action_dim = 32
token_checksum = 7286535
state_checksum = 0.0
noise_checksum = 5.798263072967529
torch = 2.7.1+cu128
cuda_available = true
```

- 产物：

```text
/tmp/hy_vla_reference_fixture/hy_vla_fixture.npz
/tmp/hy_vla_reference_fixture/lang_tokens.int32
/tmp/hy_vla_reference_fixture/lang_attention_mask.int32
/tmp/hy_vla_reference_fixture/state.float32
/tmp/hy_vla_reference_fixture/noise.float32
/tmp/hy_vla_reference_fixture/summary.json
```

- CPU load smoke 命令：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_reference.py \
  --mode load \
  --out-dir /tmp/hy_vla_reference_load \
  --device cpu \
  --dtype bf16
```

- CPU load smoke 结果：

```text
[modeling_hy_vla] Tokenizer model_path: /home/xuling/robotic_dataset/HY-VLA
[modeling_hy_vla] VLM AutoConfig loaded from embedded vlm_config_dict (nested hunyuan_vl_mot schema).
[hy-vla-reference] model loaded
```

- 结论：
  - Python reference 环境已经可以导入 Hy-VLA、读取本地 checkpoint config/tokenizer、生成 deterministic fixture，并在 CPU 上完成完整模型权重加载。
  - 完整 forward 尚未跑。原因是当前 fallback attention 是 dense torch 慢速实现，完整 Hy-VLA forward 会经过 vision/MEM、32 层 dual tower 和 10 步 Euler sampling；在本机 8GB GPU 上预计 OOM，在 CPU 上可能非常慢。
  - 下一步建议先做 safetensors tensor inventory 和 suffix-first/action-expert reference trace，而不是直接跑完整 vision+MEM forward。

English:

- Added the Python reference harness:

```text
scripts/run_hy_vla_reference.py
```

- The default `fixture` mode does not load the 9GB model weights. It only:
  - imports Hy-VLA;
  - reads the local config;
  - reads the local tokenizer;
  - creates a deterministic fixture;
  - dumps prompt tokens / attention mask / state / fixed action noise.
- Fixture contents:

```text
prompt = "pick up the bottle"
history = 6
image_size = 224
images = 3 cameras x 6 frames, deterministic uint8 formula -> float [0,1]
state = linspace(-0.25, 0.25, 32)
noise = sin(i * 0.017) * 0.05, shape [1,40,32]
```

- Fixture smoke command:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_reference.py \
  --mode fixture \
  --out-dir /tmp/hy_vla_reference_fixture
```

- Fixture result:

```text
chunk_size = 40
n_action_steps = 40
max_state_dim = 32
max_action_dim = 32
token_checksum = 7286535
state_checksum = 0.0
noise_checksum = 5.798263072967529
torch = 2.7.1+cu128
cuda_available = true
```

- Artifacts:

```text
/tmp/hy_vla_reference_fixture/hy_vla_fixture.npz
/tmp/hy_vla_reference_fixture/lang_tokens.int32
/tmp/hy_vla_reference_fixture/lang_attention_mask.int32
/tmp/hy_vla_reference_fixture/state.float32
/tmp/hy_vla_reference_fixture/noise.float32
/tmp/hy_vla_reference_fixture/summary.json
```

- CPU load smoke command:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_reference.py \
  --mode load \
  --out-dir /tmp/hy_vla_reference_load \
  --device cpu \
  --dtype bf16
```

- CPU load smoke result:

```text
[modeling_hy_vla] Tokenizer model_path: /home/xuling/robotic_dataset/HY-VLA
[modeling_hy_vla] VLM AutoConfig loaded from embedded vlm_config_dict (nested hunyuan_vl_mot schema).
[hy-vla-reference] model loaded
```

- Conclusion:
  - The Python reference environment can now import Hy-VLA, read the local checkpoint config/tokenizer, generate a deterministic fixture, and load the full model weights on CPU.
  - Full forward has not been run yet. The current fallback attention is a slow dense torch implementation, and a full Hy-VLA forward includes vision/MEM, the 32-layer dual tower, and 10 Euler sampling steps. It is expected to OOM on the local 8GB GPU and may be very slow on CPU.
  - Next step should be safetensors tensor inventory and suffix-first/action-expert reference tracing, rather than immediately running full vision+MEM forward.

## 2026-06-17 权重清点 / Weight Inventory

中文：

- 新增 HY-VLA 权重清点脚本：

```text
scripts/inspect_hy_vla_weights.py
```

- 脚本只读取 `model.safetensors` header，不加载 8.5GB tensor payload。
- 运行命令：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/inspect_hy_vla_weights.py \
  --ckpt /home/xuling/robotic_dataset/HY-VLA \
  --json-out /tmp/hy_vla_weight_inventory.json
```

- checkpoint 文件布局：

```text
/home/xuling/robotic_dataset/HY-VLA/model.safetensors  ~= 9.05 GB on disk
/home/xuling/robotic_dataset/HY-VLA/config.json
/home/xuling/robotic_dataset/HY-VLA/norm_stats.pkl
/home/xuling/robotic_dataset/HY-VLA/tokenizer.json
/home/xuling/robotic_dataset/HY-VLA/preprocessor_config.json
```

- safetensors 结果：

```text
tensors = 1628
estimated tensor bytes = 8.43 GiB
dtype = BF16 for all tensors
```

- 主要模块体积：

```text
vlm.text.layers.mlp   tensors=192, params=2,415,919,104, bytes=4.50 GiB
vlm.text.layers.attn  tensors=320, params=671,096,832,   bytes=1.25 GiB
vlm.text.lm_head      tensors=1,   params=247,435,264,   bytes=471.95 MiB
vlm.vision            tensors=335, params=450,371,568,   bytes=859.02 MiB
expert.layers.attn    tensors=320, params=335,552,512,   bytes=640.02 MiB
expert.layers.mlp     tensors=192, params=402,653,184,   bytes=768.00 MiB
```

- action/suffix 相关 projection 很小，适合优先做 C++ parity：

```text
state_proj            tensors=2, params=33,792,    bytes=66.00 KiB
action_in_proj        tensors=2, params=33,792,    bytes=66.00 KiB
action_time_mlp_in    tensors=2, params=2,098,176, bytes=4.00 MiB
action_time_mlp_out   tensors=2, params=1,049,600, bytes=2.00 MiB
action_out_proj       tensors=2, params=32,800,    bytes=64.06 KiB
```

- layer 结构确认：

```text
model.dual_tower.vlm.model.language_model.model.layers: 32 layers, 0..31
model.dual_tower.expert.model.layers:                  32 layers, 0..31
```

- `norm_stats.pkl` 内容：

```text
qpos_mean/action_mean 等统计只覆盖原始 20 维 robot/action 空间；
模型内部会 pad 到 max_state_dim=max_action_dim=32。
```

- 转换判断：
  - checkpoint 是单文件 BF16 safetensors，GGUF 转换路径比 LingBot shard 权重简单。
  - 第一阶段可以优先导出 action expert + suffix projections 的小 GGUF，用 deterministic fixture 做 suffix-first parity。
  - 完整模型 GGUF 需要同时覆盖 VLM text tower、vision tower/MEM、expert tower 和 tokenizer metadata。

English:

- Added the HY-VLA weight inspection script:

```text
scripts/inspect_hy_vla_weights.py
```

- The script reads only `model.safetensors` headers and does not load the 8.5GB tensor payload.
- Command:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/inspect_hy_vla_weights.py \
  --ckpt /home/xuling/robotic_dataset/HY-VLA \
  --json-out /tmp/hy_vla_weight_inventory.json
```

- Checkpoint layout:

```text
/home/xuling/robotic_dataset/HY-VLA/model.safetensors  ~= 9.05 GB on disk
/home/xuling/robotic_dataset/HY-VLA/config.json
/home/xuling/robotic_dataset/HY-VLA/norm_stats.pkl
/home/xuling/robotic_dataset/HY-VLA/tokenizer.json
/home/xuling/robotic_dataset/HY-VLA/preprocessor_config.json
```

- Safetensors summary:

```text
tensors = 1628
estimated tensor bytes = 8.43 GiB
dtype = BF16 for all tensors
```

- Main module sizes:

```text
vlm.text.layers.mlp   tensors=192, params=2,415,919,104, bytes=4.50 GiB
vlm.text.layers.attn  tensors=320, params=671,096,832,   bytes=1.25 GiB
vlm.text.lm_head      tensors=1,   params=247,435,264,   bytes=471.95 MiB
vlm.vision            tensors=335, params=450,371,568,   bytes=859.02 MiB
expert.layers.attn    tensors=320, params=335,552,512,   bytes=640.02 MiB
expert.layers.mlp     tensors=192, params=402,653,184,   bytes=768.00 MiB
```

- Action/suffix projections are small, so they are the best first C++ parity target:

```text
state_proj            tensors=2, params=33,792,    bytes=66.00 KiB
action_in_proj        tensors=2, params=33,792,    bytes=66.00 KiB
action_time_mlp_in    tensors=2, params=2,098,176, bytes=4.00 MiB
action_time_mlp_out   tensors=2, params=1,049,600, bytes=2.00 MiB
action_out_proj       tensors=2, params=32,800,    bytes=64.06 KiB
```

- Layer structure confirmed:

```text
model.dual_tower.vlm.model.language_model.model.layers: 32 layers, 0..31
model.dual_tower.expert.model.layers:                  32 layers, 0..31
```

- `norm_stats.pkl`:

```text
qpos_mean/action_mean and related statistics cover the original 20-D robot/action space;
the model pads internally to max_state_dim=max_action_dim=32.
```

- Conversion notes:
  - The checkpoint is a single-file BF16 safetensors checkpoint, so GGUF conversion is simpler than LingBot's sharded layout.
  - Stage 1 should export an action-expert + suffix-projection GGUF first and use the deterministic fixture for suffix-first parity.
  - The full GGUF must later cover the VLM text tower, vision tower/MEM, expert tower, and tokenizer metadata.

## 2026-06-17 GGUF 转换第一版 / First GGUF Conversion

中文：

- 新增 HY-VLA GGUF 转换脚本：

```text
scripts/convert_hy_vla_to_gguf.py
```

- 当前支持两个 scope：

```text
--scope metadata       # 只写 config / norm stats metadata
--scope action-expert  # 写 expert tower + suffix/action projections
```

- `metadata` smoke：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/convert_hy_vla_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/HY-VLA \
  --scope metadata \
  --out /tmp/hy_vla_metadata.gguf
```

- `action-expert` dry-run：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/convert_hy_vla_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/HY-VLA \
  --scope action-expert \
  --dry-run \
  --out /tmp/hy_vla_action_expert_bf16.gguf
```

- dry-run 结果：

```text
scope=action-expert
tensors=651
estimated_tensor_bytes=1414.5 MiB
tensor map validated
```

- 已生成第一份真实 HY-VLA GGUF：

```text
/home/xuling/robotic_dataset/models/hy_vla_action_expert_bf16.gguf
size ~= 1.4G
```

- GGUF reader 验证：

```text
fields = 28
tensors = 657
hy_vla.n_layers = 32
hy_vla.expert_hidden = 1024
hy_vla.written_tensor_count = 651
first_tensor = norm.state_mean, F32, shape=[32]
last_tensor = action_out_proj.bias, BF16, shape=[32]
```

- 说明：
  - 657 个 tensor 中，651 个来自 expert tower 和 suffix/action projections，6 个来自 `norm_stats.pkl`。
  - 这份 GGUF 还不包含 VLM text tower、vision tower/MEM、token embeddings/lm_head，因此不是完整模型文件。
  - 下一步 C++ 侧可以先写 `hy_vla.cpp` 的 loader 和 suffix/action expert graph，用 deterministic fixture 做 parity。

English:

- Added the HY-VLA GGUF conversion script:

```text
scripts/convert_hy_vla_to_gguf.py
```

- Current scopes:

```text
--scope metadata       # config / norm stats metadata only
--scope action-expert  # expert tower + suffix/action projections
```

- `metadata` smoke:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/convert_hy_vla_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/HY-VLA \
  --scope metadata \
  --out /tmp/hy_vla_metadata.gguf
```

- `action-expert` dry-run:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/convert_hy_vla_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/HY-VLA \
  --scope action-expert \
  --dry-run \
  --out /tmp/hy_vla_action_expert_bf16.gguf
```

- Dry-run result:

```text
scope=action-expert
tensors=651
estimated_tensor_bytes=1414.5 MiB
tensor map validated
```

- First real HY-VLA GGUF generated:

```text
/home/xuling/robotic_dataset/models/hy_vla_action_expert_bf16.gguf
size ~= 1.4G
```

- GGUF reader verification:

```text
fields = 28
tensors = 657
hy_vla.n_layers = 32
hy_vla.expert_hidden = 1024
hy_vla.written_tensor_count = 651
first_tensor = norm.state_mean, F32, shape=[32]
last_tensor = action_out_proj.bias, BF16, shape=[32]
```

- Notes:
  - Of the 657 tensors, 651 are expert/projection weights and 6 come from `norm_stats.pkl`.
  - This GGUF does not yet include the VLM text tower, vision tower/MEM, token embeddings, or lm_head, so it is not a complete model file.
  - The next C++ step is a `hy_vla.cpp` loader plus suffix/action-expert graph, using the deterministic fixture for parity.

## 2026-06-17 全量 GGUF 与 C++ suffix graph / Full GGUF and C++ Suffix Graph

中文：

- 按完整 C++ 复现目标，扩展 `scripts/convert_hy_vla_to_gguf.py`：

```text
--scope full
```

- `full` scope 会写入 checkpoint 的全部 1628 个 tensor：
  - expert tower 和 action/suffix projections 使用 C++ 友好的短名，例如 `expert.blk.0.attn_q_v.weight`；
  - 其余 VLM text / vision / MEM 参数也全部写入 GGUF；
  - 对超过 llama.cpp C GGUF tensor name 长度限制的原始名字做稳定短名；
  - `hy_vla.source_tensor_names` metadata 保留原始 PyTorch tensor 名，方便后续追溯。

- 生成的全量 GGUF：

```text
/home/xuling/robotic_dataset/models/hy_vla_full_bf16.gguf
size ~= 8.5G
written_tensor_count = 1628
total tensors = 1634  # 1628 weights + 6 norm stats
```

- 修复过的问题：
  - 第一版 full GGUF 保留了过长 tensor name，Python GGUF reader 能读，但 llama.cpp C `gguf_init_from_file` 会报：

```text
tensor name ... is too long: >= 64
```

  - 已改为短名 + 原名 metadata 映射后，C++ 可正常打开。

- 新增 C++ HY-VLA 架构注册：

```text
src/arch.h
src/model.cpp
CMakeLists.txt
src/models/hy_vla.cpp
```

- 当前 `src/models/hy_vla.cpp` 状态：
  - 能识别 `hy_vla.architecture=hy_vla`；
  - 能从全量 GGUF 中加载 suffix/action expert 所需短名权重；
  - resident-load expert `_v` branch、q/k norm、state/action/time/action_out projections；
  - 实现 suffix-only graph：

```text
state + noisy action + timestep
  -> embed_suffix
  -> expert blocks
  -> output norm
  -> action_out_proj
  -> Euler update
```

- C++ smoke：

```text
/tmp/hy_vla_smoke
```

- full suffix-only smoke 结果：

```text
out n = 1280
sum ~= 80.237353717
min ~= -0.380104363
max ~= 0.891912103
```

- 1 layer / 1 step debug smoke：

```text
VLA_HY_VLA_DEBUG_LAYERS=1 VLA_HY_VLA_DEBUG_STEPS=1 /tmp/hy_vla_smoke
```

C++:

```text
sum ~= 54.985826242
min ~= -0.104139753
max ~= 0.387882203
```

Python reference, same 1 block / 1 step suffix-only path:

```text
suffix_embs      sum ~= 3654.63525390625
suffix_out_1blk  sum ~= 898.4085693359375
v_t_1blk         sum ~= -432.3961181640625
x_1blk_1step     sum ~= 49.037872314453125
```

- 当前 parity 结论：
  - C++ loader 和 suffix/action graph 已经能运行；
  - 1-block/1-step parity 尚未完全对齐；
  - 已修正一个明确语义问题：HY-VLA suffix action tokens 不是 causal mask，而是 action chunk 内双向可见，state token 只看自己；
  - 剩余差异需要下一步加入 C++/Python 中间 tensor trace，重点查：
    - projection dtype/cast；
    - q/k RMSNorm 和 RoPE layout；
    - `ggml_flash_attn_ext` 的 GQA/mask layout；
    - expert `_v` branch 权重路径。

English:

- Extended `scripts/convert_hy_vla_to_gguf.py` for full-model conversion:

```text
--scope full
```

- The `full` scope writes all 1628 checkpoint tensors:
  - expert/action tensors use C++-friendly short names such as `expert.blk.0.attn_q_v.weight`;
  - all remaining VLM text / vision / MEM tensors are also written;
  - long PyTorch names are shortened to satisfy llama.cpp's C GGUF tensor-name limit;
  - `hy_vla.source_tensor_names` preserves the original PyTorch names for traceability.

- Full GGUF:

```text
/home/xuling/robotic_dataset/models/hy_vla_full_bf16.gguf
size ~= 8.5G
written_tensor_count = 1628
total tensors = 1634  # 1628 weights + 6 norm stats
```

- Fixed issue:
  - The first full GGUF kept very long tensor names. Python could read it, but llama.cpp C failed with:

```text
tensor name ... is too long: >= 64
```

  - The converter now writes shortened names plus original-name metadata, and C++ can open the file.

- Added HY-VLA C++ architecture wiring:

```text
src/arch.h
src/model.cpp
CMakeLists.txt
src/models/hy_vla.cpp
```

- Current `src/models/hy_vla.cpp` status:
  - detects `hy_vla.architecture=hy_vla`;
  - loads the suffix/action expert tensors from the full GGUF;
  - resident-loads expert `_v` branch, q/k norms, and state/action/time/action_out projections;
  - implements a suffix-only graph:

```text
state + noisy action + timestep
  -> embed_suffix
  -> expert blocks
  -> output norm
  -> action_out_proj
  -> Euler update
```

- C++ smoke:

```text
/tmp/hy_vla_smoke
```

- Full suffix-only smoke:

```text
out n = 1280
sum ~= 80.237353717
min ~= -0.380104363
max ~= 0.891912103
```

- 1 layer / 1 step debug smoke:

```text
VLA_HY_VLA_DEBUG_LAYERS=1 VLA_HY_VLA_DEBUG_STEPS=1 /tmp/hy_vla_smoke
```

C++:

```text
sum ~= 54.985826242
min ~= -0.104139753
max ~= 0.387882203
```

Python reference, same 1-block / 1-step suffix-only path:

```text
suffix_embs      sum ~= 3654.63525390625
suffix_out_1blk  sum ~= 898.4085693359375
v_t_1blk         sum ~= -432.3961181640625
x_1blk_1step     sum ~= 49.037872314453125
```

- Current parity conclusion:
  - The C++ loader and suffix/action graph are runnable;
  - 1-block/1-step parity is not exact yet;
  - One confirmed semantic issue was fixed: HY-VLA suffix action tokens are bidirectionally visible within the action chunk, while the state token only sees itself;
  - Next step is to add Python/C++ intermediate tensor traces, focusing on:
    - projection dtype/casts;
    - q/k RMSNorm and RoPE layout;
    - `ggml_flash_attn_ext` GQA/mask layout;
    - expert `_v` branch tensor routing.

## 2026-06-17 1-block/1-step parity 修复 / 1-block/1-step Parity Fix

中文：

- 1-block/1-step 对不齐的主要原因已经定位：
  - HY-VLA 的 dual-tower forward 中，q/k projection 按各 tower 路由；
  - 但是 q/k layernorm 不是用 expert tower 自己的 norm；
  - Python 原版明确使用 VLM tower 的 `query_layernorm/key_layernorm` 处理所有 tower 的 q/k：

```text
vlm_layer = models[0].layers[layer_idx]
q_normed.append(vlm_layer.self_attn.query_layernorm(q_part))
k_normed.append(vlm_layer.self_attn.key_layernorm(k_part))
```

- C++ 第一版误用了 expert tower 的 q/k norm：

```text
expert.blk.*.attn_q_norm.weight
expert.blk.*.attn_k_norm.weight
```

- 已修复为加载 VLM q/k norm：

```text
dt.vlm.model.lm.model.layers.*.attn.query_layernorm.weight
dt.vlm.model.lm.model.layers.*.attn.key_layernorm.weight
```

- 同时确认并记录：
  - suffix action chunk mask 不是 causal mask；
  - state token 只看自己；
  - action tokens 处于同一个 attention block，彼此双向可见。

- RoPE 也补齐为 HY-VLA 原版 dynamic alpha 形式：

```text
base = rope_theta * alpha ** (head_dim / (head_dim - 2))
alpha = 1000
```

- converter 已新增 metadata：

```text
hy_vla.rope_dynamic_alpha = 1000
```

- 已重新生成全量 GGUF：

```text
/home/xuling/robotic_dataset/models/hy_vla_full_bf16.gguf
tensors = 1634
hy_vla.rope_theta = 10000
hy_vla.rope_dynamic_alpha = 1000
hy_vla.written_tensor_count = 1628
```

- 修复后的 1-block/1-step 对比：

Python reference:

```text
v_t_1blk sum ~= -432.396118164
x_1blk_1step sum ~= 49.037872315
```

C++:

```text
VLA_HY_VLA_DEBUG_OUTPUT=vt VLA_HY_VLA_DEBUG_LAYERS=1 VLA_HY_VLA_DEBUG_STEPS=1 /tmp/hy_vla_smoke
v_t_1blk sum ~= -432.831216629

VLA_HY_VLA_DEBUG_LAYERS=1 VLA_HY_VLA_DEBUG_STEPS=1 /tmp/hy_vla_smoke
x_1blk_1step sum ~= 49.081385296
```

- 当前结论：
  - 1-block/1-step parity 已从明显不对齐修到约 `0.09%` checksum 级差异；
  - 剩余差异主要可能来自 BF16/F32 cast 和 CUDA flash attention 累积细节；
  - suffix/action expert 的结构路由已经基本确认正确。

- full suffix-only smoke：

```text
/tmp/hy_vla_smoke
out n = 1280
sum ~= 78.757642817
min ~= -0.428616613
max ~= 0.912010074
```

English:

- The main 1-block/1-step mismatch has been identified:
  - HY-VLA routes q/k projections per tower;
  - but q/k layernorm does not use the expert tower's own q/k norm;
  - the Python reference applies the VLM tower's `query_layernorm/key_layernorm` to q/k from all towers:

```text
vlm_layer = models[0].layers[layer_idx]
q_normed.append(vlm_layer.self_attn.query_layernorm(q_part))
k_normed.append(vlm_layer.self_attn.key_layernorm(k_part))
```

- The first C++ version incorrectly used the expert tower q/k norms:

```text
expert.blk.*.attn_q_norm.weight
expert.blk.*.attn_k_norm.weight
```

- Fixed C++ to load the VLM q/k norms instead:

```text
dt.vlm.model.lm.model.layers.*.attn.query_layernorm.weight
dt.vlm.model.lm.model.layers.*.attn.key_layernorm.weight
```

- Also confirmed:
  - the suffix action chunk mask is not causal;
  - the state token sees only itself;
  - action tokens share one attention block and are bidirectionally visible.

- RoPE metadata is now aligned with the original HY-VLA dynamic-alpha formula:

```text
base = rope_theta * alpha ** (head_dim / (head_dim - 2))
alpha = 1000
```

- The converter now writes:

```text
hy_vla.rope_dynamic_alpha = 1000
```

- Regenerated full GGUF:

```text
/home/xuling/robotic_dataset/models/hy_vla_full_bf16.gguf
tensors = 1634
hy_vla.rope_theta = 10000
hy_vla.rope_dynamic_alpha = 1000
hy_vla.written_tensor_count = 1628
```

- Fixed 1-block/1-step comparison:

Python reference:

```text
v_t_1blk sum ~= -432.396118164
x_1blk_1step sum ~= 49.037872315
```

C++:

```text
VLA_HY_VLA_DEBUG_OUTPUT=vt VLA_HY_VLA_DEBUG_LAYERS=1 VLA_HY_VLA_DEBUG_STEPS=1 /tmp/hy_vla_smoke
v_t_1blk sum ~= -432.831216629

VLA_HY_VLA_DEBUG_LAYERS=1 VLA_HY_VLA_DEBUG_STEPS=1 /tmp/hy_vla_smoke
x_1blk_1step sum ~= 49.081385296
```

- Current conclusion:
  - 1-block/1-step parity improved from a clear mismatch to roughly `0.09%` checksum-level difference;
  - the remaining drift is likely BF16/F32 cast and CUDA flash-attention accumulation detail;
  - suffix/action expert routing is now structurally aligned.

- Full suffix-only smoke:

```text
/tmp/hy_vla_smoke
out n = 1280
sum ~= 78.757642817
min ~= -0.428616613
max ~= 0.912010074
```

## 2026-06-17 server/lang 前置通路 / Server and Language Prefix Prep

中文：

- 修复 HY-VLA C++ config 暴露：

```text
cfg.n_lang = hy_vla.tokenizer_max_length = 64
cfg.n_img = 0  # 当前 HY-VLA C++ 仍是 suffix/text-prep 阶段，vision/MEM 尚未接入
```

- 修改 `src/serving/server.cpp`：
  - 原来所有模型都强制要求 `images` 或 `precomputed_img_emb`；
  - 现在仅当 `cfg.n_img > 0` 时强制要求图像输入；
  - 因此 HY-VLA 当前阶段可以通过标准 `vla-server` 跑 language/state/noise smoke。

- 新增 HY-VLA GGUF row-fetch 能力：
  - HY-VLA checkpoint 没有单独 `embed_tokens.weight`；
  - config 中 `tie_word_embeddings=true`；
  - 因此 text token embedding 从 tied `lm_head.weight` 逐行读取：

```text
dual_tower.vlm.model.language_model.lm_head.weight
```

- 新增 debug 输出：

```text
VLA_HY_VLA_DEBUG_OUTPUT=lang
```

该模式只读取 lang token embeddings，不跑 suffix/action graph。

- lang row-fetch smoke：

```text
VLA_HY_VLA_DEBUG_OUTPUT=lang /tmp/hy_vla_lang_smoke
```

结果：

```text
lang n = 8192  # 4 tokens * 2048 hidden
sum ~= 0.033759270
min ~= -0.671875000
max ~= 0.408203125
```

- `vla-server` HY-VLA 加载 smoke：

```text
timeout 5 ./build/vla-server \
  --bind tcp://127.0.0.1:6099 \
  /home/xuling/robotic_dataset/models/hy_vla_full_bf16.gguf
```

结果：

```text
vla: arch = hy_vla
vla(hy_vla): loaded suffix expert weights 0.69 GiB
vla-server: loaded. chunk_size=40 action_dim=32 n_lang=64 hidden=2048 expert_h=1024
vla-server: bound to tcp://127.0.0.1:6099. ready.
```

- 当前状态：
  - server 侧已能加载 HY-VLA；
  - language token embedding row-fetch 已打通；
  - 下一步是接 VLM text-only prefix tower / prefix KV；
  - vision/MEM 仍保留为后续独立阶段。

English:

- Fixed HY-VLA C++ config exposure:

```text
cfg.n_lang = hy_vla.tokenizer_max_length = 64
cfg.n_img = 0  # current HY-VLA C++ path is suffix/text-prep only; vision/MEM is not wired yet
```

- Updated `src/serving/server.cpp`:
  - previously every model required `images` or `precomputed_img_emb`;
  - now image input is required only when `cfg.n_img > 0`;
  - this allows the current HY-VLA stage to use the standard `vla-server` path for language/state/noise smoke tests.

- Added HY-VLA GGUF row-fetch support:
  - the checkpoint has no separate `embed_tokens.weight`;
  - config has `tie_word_embeddings=true`;
  - token embeddings are fetched from the tied `lm_head.weight`:

```text
dual_tower.vlm.model.language_model.lm_head.weight
```

- Added debug output:

```text
VLA_HY_VLA_DEBUG_OUTPUT=lang
```

This mode only reads language token embeddings and does not run the suffix/action graph.

- Language row-fetch smoke:

```text
VLA_HY_VLA_DEBUG_OUTPUT=lang /tmp/hy_vla_lang_smoke
```

Result:

```text
lang n = 8192  # 4 tokens * 2048 hidden
sum ~= 0.033759270
min ~= -0.671875000
max ~= 0.408203125
```

- `vla-server` HY-VLA load smoke:

```text
timeout 5 ./build/vla-server \
  --bind tcp://127.0.0.1:6099 \
  /home/xuling/robotic_dataset/models/hy_vla_full_bf16.gguf
```

Result:

```text
vla: arch = hy_vla
vla(hy_vla): loaded suffix expert weights 0.69 GiB
vla-server: loaded. chunk_size=40 action_dim=32 n_lang=64 hidden=2048 expert_h=1024
vla-server: bound to tcp://127.0.0.1:6099. ready.
```

- Current status:
  - the server can load HY-VLA;
  - language token embedding row-fetch is working;
  - the next step is wiring the VLM text-only prefix tower / prefix KV;
  - vision/MEM remains a later dedicated stage.

## 2026-06-17 text-prefix KV + suffix 联合注意力 / Text-Prefix KV + Suffix Joint Attention

中文：

- 在 `src/models/hy_vla.cpp` 中新增可选 VLM text tower resident load：

```text
VLA_HY_VLA_TEXT_LAYERS=N
```

- 新增 text-only prefix layer graph：
  - token embedding 仍从 tied `lm_head.weight` row-fetch；
  - VLM text block 使用 VLM tower 的 q/k/v/o、MLP、RMSNorm；
  - RoPE 使用 HY-VLA dynamic alpha 设置；
  - prefix K/V 可以导出给 suffix/action expert denoise step。

- 新增 suffix-attend-prefix debug path：

```text
VLA_HY_VLA_DEBUG_OUTPUT=joint_vt
VLA_HY_VLA_DEBUG_OUTPUT=joint
```

含义：
  - `joint_vt`：返回 1-step denoise 的 `v_t`，suffix action expert attends to prefix K/V；
  - `joint`：返回 `x_t + dt * v_t`；
  - 当前 debug fixture 使用 4 个固定 text tokens、零 state、零 noise；
  - prefix K/V 先由 VLM text tower 填充，suffix 再拼接 prefix K/V + suffix K/V 做 full attention。

- 修复一个 GGML graph allocation 细节：
  - `joint_layers=1` 时 prefix self-attention mask 不参与最终输出，只需要该层 prefix K/V；
  - 因此 graph allocator 不会给 `t_pref_mask` 分配 buffer；
  - input 写入改为只写已进入 graph、已分配 buffer 的 tensor。

- 新增 Python parity 小脚本：

```text
scripts/run_hy_vla_joint_parity.py
```

用于运行同一个 tiny fixture 的 suffix-only / joint-prefix-KV Python reference。

- Python vs C++ 1-block/1-step text-prefix joint parity：

Python reference：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_joint_parity.py --mode joint --layers 1 --dtype bf16

prefix_emb sum ~= 0.033759236336
joint_v_t_1blk sum ~= -400.298828125
joint_x_1step_1blk sum ~= 40.029884338
```

C++：

```text
VLA_HY_VLA_TEXT_LAYERS=1 \
VLA_HY_VLA_DEBUG_LAYERS=1 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=joint_vt \
/tmp/hy_vla_lang_smoke

joint_v_t_1blk sum ~= -400.200045970

VLA_HY_VLA_TEXT_LAYERS=1 \
VLA_HY_VLA_DEBUG_LAYERS=1 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=joint \
/tmp/hy_vla_lang_smoke

joint_x_1step_1blk sum ~= 40.020005376
```

- 同一零 state/零 noise fixture 的 suffix-only sanity check：

Python：

```text
suffix_v_t_1blk sum ~= -458.661315918
suffix_x_1step_1blk sum ~= 45.866130829
```

C++：

```text
VLA_HY_VLA_DEBUG_OUTPUT=vt VLA_HY_VLA_DEBUG_LAYERS=1 VLA_HY_VLA_DEBUG_STEPS=1 /tmp/hy_vla_lang_smoke
suffix_v_t_1blk sum ~= -458.594855341

VLA_HY_VLA_DEBUG_LAYERS=1 VLA_HY_VLA_DEBUG_STEPS=1 /tmp/hy_vla_lang_smoke
suffix_x_1step_1blk sum ~= 45.859486419
```

- 当前结论：
  - text-prefix KV + suffix/action expert joint attention 已在 C++ 路径跑通；
  - 1-block/1-step joint checksum 差异约 `0.025%`；
  - suffix-only 零输入 checksum 差异约 `0.014%`；
  - 这些差异处在 BF16/F32 cast、CPU/PyTorch 与 CUDA/GGML flash-attention accumulation 差异的合理范围内；
  - 下一步可继续扩展到更多 layers，然后接 vision/MEM prefix。

English:

- Added optional resident loading for the VLM text tower in `src/models/hy_vla.cpp`:

```text
VLA_HY_VLA_TEXT_LAYERS=N
```

- Added a text-only prefix layer graph:
  - token embeddings are still row-fetched from the tied `lm_head.weight`;
  - the VLM text block uses the VLM tower q/k/v/o, MLP, and RMSNorm weights;
  - RoPE uses the HY-VLA dynamic-alpha setting;
  - prefix K/V can be exported for the suffix/action expert denoise step.

- Added suffix-attend-prefix debug modes:

```text
VLA_HY_VLA_DEBUG_OUTPUT=joint_vt
VLA_HY_VLA_DEBUG_OUTPUT=joint
```

Meaning:
  - `joint_vt` returns the 1-step denoise `v_t`, where the suffix action expert attends to prefix K/V;
  - `joint` returns `x_t + dt * v_t`;
  - the current debug fixture uses 4 fixed text tokens, zero state, and zero noise;
  - prefix K/V is filled by the VLM text tower, and suffix attention then uses concatenated prefix K/V + suffix K/V.

- Fixed a GGML graph-allocation detail:
  - when `joint_layers=1`, the prefix self-attention mask is not part of the final graph because only that layer's prefix K/V is needed;
  - therefore the graph allocator does not assign a buffer to `t_pref_mask`;
  - input writes now skip tensors that are not present in the allocated graph.

- Added a Python parity helper:

```text
scripts/run_hy_vla_joint_parity.py
```

It runs the same tiny fixture through the suffix-only and joint-prefix-KV Python reference paths.

- Python vs C++ 1-block/1-step text-prefix joint parity:

Python reference:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_joint_parity.py --mode joint --layers 1 --dtype bf16

prefix_emb sum ~= 0.033759236336
joint_v_t_1blk sum ~= -400.298828125
joint_x_1step_1blk sum ~= 40.029884338
```

C++:

```text
VLA_HY_VLA_TEXT_LAYERS=1 \
VLA_HY_VLA_DEBUG_LAYERS=1 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=joint_vt \
/tmp/hy_vla_lang_smoke

joint_v_t_1blk sum ~= -400.200045970

VLA_HY_VLA_TEXT_LAYERS=1 \
VLA_HY_VLA_DEBUG_LAYERS=1 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=joint \
/tmp/hy_vla_lang_smoke

joint_x_1step_1blk sum ~= 40.020005376
```

- Suffix-only sanity check on the same zero-state/zero-noise fixture:

Python:

```text
suffix_v_t_1blk sum ~= -458.661315918
suffix_x_1step_1blk sum ~= 45.866130829
```

C++:

```text
VLA_HY_VLA_DEBUG_OUTPUT=vt VLA_HY_VLA_DEBUG_LAYERS=1 VLA_HY_VLA_DEBUG_STEPS=1 /tmp/hy_vla_lang_smoke
suffix_v_t_1blk sum ~= -458.594855341

VLA_HY_VLA_DEBUG_LAYERS=1 VLA_HY_VLA_DEBUG_STEPS=1 /tmp/hy_vla_lang_smoke
suffix_x_1step_1blk sum ~= 45.859486419
```

- Current conclusion:
  - text-prefix KV + suffix/action expert joint attention is now running in C++;
  - 1-block/1-step joint checksum drift is about `0.025%`;
  - suffix-only zero-input checksum drift is about `0.014%`;
  - these differences are consistent with BF16/F32 casts and PyTorch CPU vs GGML/CUDA flash-attention accumulation details;
  - the next step is to scale to more layers, then wire the vision/MEM prefix path.

## 2026-06-17 多层 prefix-KV parity 与稳定 VLM 短名 / Multi-Layer Prefix-KV Parity and Stable VLM Names

中文：

- 将 text-prefix KV + suffix/action expert joint path 扩展到 2 层 parity 点。

Python reference：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_joint_parity.py --mode joint --layers 2 --dtype bf16

prefix_emb sum ~= 0.033759236336
joint_v_t_2blk sum ~= -372.319610596
joint_x_1step_2blk sum ~= 37.231960297
```

C++：

```text
VLA_HY_VLA_TEXT_LAYERS=2 \
VLA_HY_VLA_DEBUG_LAYERS=2 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=joint_vt \
/tmp/hy_vla_lang_smoke

joint_v_t_2blk sum ~= -371.737192906

VLA_HY_VLA_TEXT_LAYERS=2 \
VLA_HY_VLA_DEBUG_LAYERS=2 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=joint \
/tmp/hy_vla_lang_smoke

joint_x_1step_2blk sum ~= 37.173719895
```

- 当前 2 层 checksum 差异约 `0.16%`，比 1 层略大，但没有出现层间 KV 错位或 mask 结构错误的迹象。

- 更新 `scripts/convert_hy_vla_to_gguf.py`：
  - full GGUF 现在给 VLM text tower 写稳定短名；
  - 同时包含普通 text 分支和 `_v` vision 分支参数：

```text
vlm.blk.{i}.attn_norm.weight
vlm.blk.{i}.attn_norm_v.weight
vlm.blk.{i}.attn_q/attn_k/attn_v/attn_o.weight
vlm.blk.{i}.attn_q_v/attn_k_v/attn_v_v/attn_o_v.weight
vlm.blk.{i}.ffn_norm.weight
vlm.blk.{i}.ffn_norm_v.weight
vlm.blk.{i}.ffn_gate/ffn_up/ffn_down.weight
vlm.blk.{i}.ffn_gate_v/ffn_up_v/ffn_down_v.weight
```

- 更新 `src/models/hy_vla.cpp` loader：
  - VLM text layer 支持新稳定短名；
  - action expert 的 q/k layernorm 仍使用 VLM tower 的 q/k norm，并支持新旧 GGUF 名称 fallback；
  - 旧 `hy_vla_full_bf16.gguf` 和新稳定短名 GGUF 都能加载。

- 新生成 full GGUF：

```text
/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf
size ~= 8.5G
tensors = 1628 checkpoint tensors + norm stats
```

- 新 GGUF 验证：

```text
VLA_HY_VLA_TEXT_LAYERS=2 \
VLA_HY_VLA_DEBUG_LAYERS=2 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=joint_vt \
VLA_HY_VLA_GGUF=/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf \
/tmp/hy_vla_lang_smoke

joint_v_t_2blk sum ~= -371.737192906
```

- 新 GGUF server load smoke：

```text
timeout 8 env VLA_HY_VLA_TEXT_LAYERS=2 ./build/vla-server \
  --bind tcp://127.0.0.1:6099 \
  /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

vla-server: loaded. chunk_size=40 action_dim=32 n_lang=64 hidden=2048 expert_h=1024
vla-server: bound to tcp://127.0.0.1:6099. ready.
```

- vision/MEM 前置结论：
  - visual patch tokens 在 VLM tower 中走 `_v` 分支；
  - vision_start / vision_end / row split tokens 和 language tokens 走普通 text 分支；
  - 因此 precomputed visual tokens 不能简单当作全 text prefix 处理；
  - 下一步应实现 prefix modality routing：按 prefix segment 对 text/vision branch 做 projection/MLP，再按原序拼回 shared attention。

English:

- Extended the text-prefix KV + suffix/action expert joint path to a 2-layer parity point.

Python reference:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_joint_parity.py --mode joint --layers 2 --dtype bf16

prefix_emb sum ~= 0.033759236336
joint_v_t_2blk sum ~= -372.319610596
joint_x_1step_2blk sum ~= 37.231960297
```

C++:

```text
VLA_HY_VLA_TEXT_LAYERS=2 \
VLA_HY_VLA_DEBUG_LAYERS=2 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=joint_vt \
/tmp/hy_vla_lang_smoke

joint_v_t_2blk sum ~= -371.737192906

VLA_HY_VLA_TEXT_LAYERS=2 \
VLA_HY_VLA_DEBUG_LAYERS=2 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=joint \
/tmp/hy_vla_lang_smoke

joint_x_1step_2blk sum ~= 37.173719895
```

- The current 2-layer checksum drift is about `0.16%`, larger than the 1-layer point but with no sign of layer-wise KV misalignment or mask-structure errors.

- Updated `scripts/convert_hy_vla_to_gguf.py`:
  - full GGUF now writes stable short names for the VLM text tower;
  - both the normal text branch and the `_v` vision branch are named explicitly:

```text
vlm.blk.{i}.attn_norm.weight
vlm.blk.{i}.attn_norm_v.weight
vlm.blk.{i}.attn_q/attn_k/attn_v/attn_o.weight
vlm.blk.{i}.attn_q_v/attn_k_v/attn_v_v/attn_o_v.weight
vlm.blk.{i}.ffn_norm.weight
vlm.blk.{i}.ffn_norm_v.weight
vlm.blk.{i}.ffn_gate/ffn_up/ffn_down.weight
vlm.blk.{i}.ffn_gate_v/ffn_up_v/ffn_down_v.weight
```

- Updated the `src/models/hy_vla.cpp` loader:
  - VLM text layers support the new stable short names;
  - action expert q/k layernorm still uses the VLM tower q/k norm and now supports old/new GGUF name fallback;
  - both the old `hy_vla_full_bf16.gguf` and the new stable-name GGUF can load.

- New full GGUF:

```text
/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf
size ~= 8.5G
tensors = 1628 checkpoint tensors + norm stats
```

- New GGUF verification:

```text
VLA_HY_VLA_TEXT_LAYERS=2 \
VLA_HY_VLA_DEBUG_LAYERS=2 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=joint_vt \
VLA_HY_VLA_GGUF=/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf \
/tmp/hy_vla_lang_smoke

joint_v_t_2blk sum ~= -371.737192906
```

- New GGUF server load smoke:

```text
timeout 8 env VLA_HY_VLA_TEXT_LAYERS=2 ./build/vla-server \
  --bind tcp://127.0.0.1:6099 \
  /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

vla-server: loaded. chunk_size=40 action_dim=32 n_lang=64 hidden=2048 expert_h=1024
vla-server: bound to tcp://127.0.0.1:6099. ready.
```

- Vision/MEM prerequisite conclusion:
  - visual patch tokens use the `_v` branch inside the VLM tower;
  - vision_start / vision_end / row split tokens and language tokens use the normal text branch;
  - therefore precomputed visual tokens must not simply be treated as an all-text prefix;
  - the next step should implement prefix modality routing: project/MLP text and vision segments through their own branches, then reassemble them in original order for shared attention.

## 2026-06-17 prefix modality routing 骨架 / Prefix Modality Routing Skeleton

中文：

- 在 `src/models/hy_vla.cpp` 中新增 VLM prefix `_v` branch resident loading：
  - 新稳定短名 GGUF 中检测到 `vlm.blk.0.attn_norm_v.weight` 时自动启用；
  - 旧 GGUF 没有这些稳定短名时保持 text-only 行为；
  - server load 日志现在显示：

```text
prefix_vision=yes/no
```

- 新增两段式 prefix layer builder：

```text
text segment   -> VLM normal text branch
vision segment -> VLM _v branch
concat Q/K/V over sequence
shared RoPE + attention
split attention output back to text/vision branch
branch-specific o_proj + MLP
concat output
```

- 新增 debug 模式：

```text
VLA_HY_VLA_DEBUG_OUTPUT=mixed_prefix
```

当前该模式只用于 C++ 内部 smoke：
  - `lang_tokens` 作为 text segment；
  - `precomputed_img_emb` 作为 synthetic/precomputed visual segment；
  - `Inputs::n_img_views` 在这个 debug 模式下临时解释为 visual token count；
  - 正式 server/proto 语义后续会单独整理，不直接复用这个临时约定。

- 临时 C++ harness：

```text
/tmp/hy_vla_mixed_prefix_smoke
```

使用 4 个固定 text tokens + 3 个 synthetic visual tokens。

C++ smoke：

```text
VLA_HY_VLA_TEXT_LAYERS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=mixed_prefix \
VLA_HY_VLA_GGUF=/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf \
/tmp/hy_vla_mixed_prefix_smoke

mixed_prefix_1blk n = 14336
sum ~= 19.039878845
min ~= -51.071376801
max ~= 73.893173218
```

2 层 C++ smoke：

```text
VLA_HY_VLA_TEXT_LAYERS=2 \
VLA_HY_VLA_DEBUG_LAYERS=2 \
VLA_HY_VLA_DEBUG_OUTPUT=mixed_prefix \
VLA_HY_VLA_GGUF=/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf \
/tmp/hy_vla_mixed_prefix_smoke

mixed_prefix_2blk n = 14336
sum ~= -167.390667194
min ~= -51.076030731
max ~= 73.892051697
```

- Python reference 脚本新增：

```text
scripts/run_hy_vla_joint_parity.py --mode mixed_prefix
```

- 新增 raw F32 tensor 比较工具：

```text
scripts/compare_f32_tensors.py
```

Python 1 层结果：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_joint_parity.py --mode mixed_prefix --layers 1 --dtype bf16

prefix_emb sum ~= 0.033759236336
mixed_prefix_1blk sum ~= 18.617565155
min ~= -51.0
max ~= 74.0
```

- C++ / Python full tensor diff：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/compare_f32_tensors.py \
  /tmp/hy_cpp_mixed_prefix_1blk.f32 \
  /tmp/hy_py_mixed_prefix_1blk.f32 \
  --shape 7 2048 \
  --segments 4 3

all mean_abs ~= 0.004689124
all max_abs  ~= 0.267238617
all rmse     ~= 0.008203761

text segment mean_abs ~= 0.001978203
text segment max_abs  ~= 0.150421143

vision segment mean_abs ~= 0.008303687
vision segment max_abs  ~= 0.267238617
```

- 当前结论：
  - C++ mixed-prefix branch routing 已跑通；
  - VLM `_v` branch 权重加载、text/vision segment 分支投影、shared attention、分支回写都能执行；
  - mixed-prefix 1 层 full tensor `mean_abs_diff ~= 0.00469`，vision segment 误差略高于 text segment；
  - extrema 与 Python 接近；
  - 下一步可以把 mixed prefix KV 接入 suffix/action joint path，或进一步按 block internals dump 定位 vision segment 的剩余差异。

English:

- Added resident loading for the VLM prefix `_v` branch in `src/models/hy_vla.cpp`:
  - it is enabled automatically when the new stable-name GGUF has `vlm.blk.0.attn_norm_v.weight`;
  - old GGUF files without these stable names keep the text-only behavior;
  - server load logs now report:

```text
prefix_vision=yes/no
```

- Added a two-segment prefix layer builder:

```text
text segment   -> VLM normal text branch
vision segment -> VLM _v branch
concat Q/K/V over sequence
shared RoPE + attention
split attention output back to text/vision branch
branch-specific o_proj + MLP
concat output
```

- Added a debug mode:

```text
VLA_HY_VLA_DEBUG_OUTPUT=mixed_prefix
```

This mode is currently for internal C++ smoke tests only:
  - `lang_tokens` forms the text segment;
  - `precomputed_img_emb` forms a synthetic/precomputed visual segment;
  - `Inputs::n_img_views` is temporarily interpreted as the visual-token count in this debug mode;
  - the final server/proto semantics will be cleaned up separately.

- Temporary C++ harness:

```text
/tmp/hy_vla_mixed_prefix_smoke
```

It uses 4 fixed text tokens plus 3 synthetic visual tokens.

C++ smoke:

```text
VLA_HY_VLA_TEXT_LAYERS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=mixed_prefix \
VLA_HY_VLA_GGUF=/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf \
/tmp/hy_vla_mixed_prefix_smoke

mixed_prefix_1blk n = 14336
sum ~= 19.039878845
min ~= -51.071376801
max ~= 73.893173218
```

2-layer C++ smoke:

```text
VLA_HY_VLA_TEXT_LAYERS=2 \
VLA_HY_VLA_DEBUG_LAYERS=2 \
VLA_HY_VLA_DEBUG_OUTPUT=mixed_prefix \
VLA_HY_VLA_GGUF=/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf \
/tmp/hy_vla_mixed_prefix_smoke

mixed_prefix_2blk n = 14336
sum ~= -167.390667194
min ~= -51.076030731
max ~= 73.892051697
```

- Added Python reference support:

```text
scripts/run_hy_vla_joint_parity.py --mode mixed_prefix
```

- Added a raw F32 tensor comparison helper:

```text
scripts/compare_f32_tensors.py
```

Python 1-layer result:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_joint_parity.py --mode mixed_prefix --layers 1 --dtype bf16

prefix_emb sum ~= 0.033759236336
mixed_prefix_1blk sum ~= 18.617565155
min ~= -51.0
max ~= 74.0
```

- C++ / Python full tensor diff:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/compare_f32_tensors.py \
  /tmp/hy_cpp_mixed_prefix_1blk.f32 \
  /tmp/hy_py_mixed_prefix_1blk.f32 \
  --shape 7 2048 \
  --segments 4 3

all mean_abs ~= 0.004689124
all max_abs  ~= 0.267238617
all rmse     ~= 0.008203761

text segment mean_abs ~= 0.001978203
text segment max_abs  ~= 0.150421143

vision segment mean_abs ~= 0.008303687
vision segment max_abs  ~= 0.267238617
```

- Current conclusion:
  - C++ mixed-prefix branch routing is now running;
  - VLM `_v` branch weight loading, text/vision branch projection, shared attention, and branch-specific output/MLP all execute;
  - mixed-prefix 1-layer full tensor `mean_abs_diff ~= 0.00469`, with the vision segment slightly higher than the text segment;
  - extrema are close to Python;
  - the next step can be wiring mixed prefix K/V into the suffix/action joint path, or dumping block internals to localize the remaining vision-segment drift.

## 2026-06-17 mixed-prefix KV 接入 action suffix / Mixed-Prefix KV to Action Suffix

中文：

- 在 `src/models/hy_vla.cpp` 中新增 mixed-prefix action debug：

```text
VLA_HY_VLA_DEBUG_OUTPUT=mixed_joint_vt
VLA_HY_VLA_DEBUG_OUTPUT=mixed_joint
```

含义：
  - text segment 走 VLM normal branch；
  - visual segment 走 VLM `_v` branch；
  - prefix K/V 由 mixed-prefix graph 产生；
  - suffix/action expert attend 到 mixed prefix K/V；
  - `mixed_joint_vt` 返回 action velocity `v_t`；
  - `mixed_joint` 返回 `x_t + dt * v_t`。

- C++ 1 层 mixed_joint smoke：

```text
VLA_HY_VLA_TEXT_LAYERS=1 \
VLA_HY_VLA_DEBUG_LAYERS=1 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=mixed_joint_vt \
VLA_HY_VLA_GGUF=/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf \
VLA_HY_VLA_DUMP_F32=/tmp/hy_cpp_mixed_joint_vt_1blk.f32 \
/tmp/hy_vla_mixed_prefix_smoke

mixed_joint_v_t_1blk sum ~= -344.428991805
min ~= -2.527388573
max ~= 0.454899967
```

- Python reference：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_joint_parity.py \
  --mode mixed_joint \
  --layers 1 \
  --dtype bf16 \
  --dump-f32 /tmp/hy_py_mixed_joint_vt_1blk.f32

mixed_joint_v_t_1blk sum ~= -344.640808105
mixed_joint_x_1step_1blk sum ~= 34.464080811
```

- C++ / Python action tensor diff：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/compare_f32_tensors.py \
  /tmp/hy_cpp_mixed_joint_vt_1blk.f32 \
  /tmp/hy_py_mixed_joint_vt_1blk.f32 \
  --shape 40 32

mean_abs ~= 0.002029480
max_abs  ~= 0.018661261
rmse     ~= 0.002908587
```

- 2 层 C++ mixed_joint smoke：

```text
VLA_HY_VLA_TEXT_LAYERS=2 \
VLA_HY_VLA_DEBUG_LAYERS=2 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=mixed_joint_vt \
VLA_HY_VLA_GGUF=/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf \
/tmp/hy_vla_mixed_prefix_smoke

mixed_joint_v_t_2blk sum ~= -333.879115215
min ~= -2.678010702
max ~= 0.423819631
```

- 当前结论：
  - mixed text/vision prefix K/V 已能接入 suffix/action expert；
  - 1 层 action `v_t` parity 很稳，`mean_abs_diff ~= 0.00203`；
  - 这比 mixed-prefix hidden tensor 的 checksum 更能说明 action 路径结构正确；
  - 下一步可以把 debug-only `precomputed_img_emb` 约定整理为正式 precomputed visual token path，或者开始实现 Hy-ViT2/MEM 的 C++ visual token 生成。

English:

- Added mixed-prefix action debug modes in `src/models/hy_vla.cpp`:

```text
VLA_HY_VLA_DEBUG_OUTPUT=mixed_joint_vt
VLA_HY_VLA_DEBUG_OUTPUT=mixed_joint
```

Meaning:
  - the text segment uses the VLM normal branch;
  - the visual segment uses the VLM `_v` branch;
  - prefix K/V is produced by the mixed-prefix graph;
  - the suffix/action expert attends to that mixed prefix K/V;
  - `mixed_joint_vt` returns action velocity `v_t`;
  - `mixed_joint` returns `x_t + dt * v_t`.

- C++ 1-layer mixed_joint smoke:

```text
VLA_HY_VLA_TEXT_LAYERS=1 \
VLA_HY_VLA_DEBUG_LAYERS=1 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=mixed_joint_vt \
VLA_HY_VLA_GGUF=/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf \
VLA_HY_VLA_DUMP_F32=/tmp/hy_cpp_mixed_joint_vt_1blk.f32 \
/tmp/hy_vla_mixed_prefix_smoke

mixed_joint_v_t_1blk sum ~= -344.428991805
min ~= -2.527388573
max ~= 0.454899967
```

- Python reference:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_joint_parity.py \
  --mode mixed_joint \
  --layers 1 \
  --dtype bf16 \
  --dump-f32 /tmp/hy_py_mixed_joint_vt_1blk.f32

mixed_joint_v_t_1blk sum ~= -344.640808105
mixed_joint_x_1step_1blk sum ~= 34.464080811
```

- C++ / Python action tensor diff:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/compare_f32_tensors.py \
  /tmp/hy_cpp_mixed_joint_vt_1blk.f32 \
  /tmp/hy_py_mixed_joint_vt_1blk.f32 \
  --shape 40 32

mean_abs ~= 0.002029480
max_abs  ~= 0.018661261
rmse     ~= 0.002908587
```

- 2-layer C++ mixed_joint smoke:

```text
VLA_HY_VLA_TEXT_LAYERS=2 \
VLA_HY_VLA_DEBUG_LAYERS=2 \
VLA_HY_VLA_DEBUG_STEPS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=mixed_joint_vt \
VLA_HY_VLA_GGUF=/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf \
/tmp/hy_vla_mixed_prefix_smoke

mixed_joint_v_t_2blk sum ~= -333.879115215
min ~= -2.678010702
max ~= 0.423819631
```

- Current conclusion:
  - mixed text/vision prefix K/V can now feed the suffix/action expert;
  - 1-layer action `v_t` parity is stable, with `mean_abs_diff ~= 0.00203`;
  - this is a stronger signal for action-path structural correctness than the mixed-prefix hidden checksum alone;
  - the next step can be formalizing the debug-only `precomputed_img_emb` convention into a proper precomputed visual-token path, or starting the C++ Hy-ViT2/MEM visual token generator.

## 2026-06-17 server precomputed visual-token path / Server Precomputed Visual-Token Path

中文：

- 更新 `src/serving/server.cpp` 的 `precomputed_img_emb` 校验：
  - 对传统模型保持原语义：`size == n_views * cfg.n_img * hidden`；
  - 当 `cfg.n_img == 0` 且请求提供 `precomputed_img_emb` 时，改为 token-style 语义：

```text
precomputed token count = precomputed_img_emb_size / hidden
Inputs::n_img_views = precomputed token count
```

- 这让当前 HY-VLA 阶段可以通过标准 `vla.proto` / `vla-server` 传入 precomputed visual tokens，而不需要绕过 server 调 `predict()`。

- 新增 server smoke 脚本：

```text
scripts/smoke_hy_vla_server_precomputed.py
```

该脚本：
  - 动态编译/加载 `src/serving/vla.proto`；
  - 发送 4 个固定 text tokens；
  - 发送 3 个 synthetic visual tokens，shape 等价于 `[3, 2048]`；
  - state 使用 32 维零向量；
  - 用于验证 server -> protobuf -> `Inputs::precomputed_img_emb` -> HY-VLA mixed path。

- server smoke 命令：

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  VLA_HY_VLA_DEBUG_OUTPUT=mixed_joint_vt \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6105 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py --addr tcp://127.0.0.1:6105
```

结果：

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= -344.428955078
min ~= -2.527388573
max ~= 0.454899967
```

- 该结果与 direct C++ mixed_joint dump 对齐，说明标准 ZMQ/protobuf server 链路已能承载 HY-VLA 的 precomputed visual-token mixed action path。

English:

- Updated `precomputed_img_emb` validation in `src/serving/server.cpp`:
  - traditional models keep the old semantics: `size == n_views * cfg.n_img * hidden`;
  - when `cfg.n_img == 0` and the request provides `precomputed_img_emb`, the server now uses token-style semantics:

```text
precomputed token count = precomputed_img_emb_size / hidden
Inputs::n_img_views = precomputed token count
```

- This allows the current HY-VLA stage to pass precomputed visual tokens through standard `vla.proto` / `vla-server`, instead of bypassing the server and calling `predict()` directly.

- Added a server smoke script:

```text
scripts/smoke_hy_vla_server_precomputed.py
```

The script:
  - dynamically compiles/loads `src/serving/vla.proto`;
  - sends 4 fixed text tokens;
  - sends 3 synthetic visual tokens, equivalent to shape `[3, 2048]`;
  - uses a 32-D zero state vector;
  - verifies the server -> protobuf -> `Inputs::precomputed_img_emb` -> HY-VLA mixed path.

- Server smoke command:

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  VLA_HY_VLA_DEBUG_OUTPUT=mixed_joint_vt \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6105 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py --addr tcp://127.0.0.1:6105
```

Result:

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= -344.428955078
min ~= -2.527388573
max ~= 0.454899967
```

- The result matches the direct C++ mixed_joint dump, confirming that the standard ZMQ/protobuf server path can carry HY-VLA precomputed visual-token mixed action inference.

## 2026-06-17 routed mixed-prefix path / Routed Mixed-Prefix Path

中文：

- 新增 generic routed prefix builder：

```text
build_hy_prefix_routed_layer(...)
```

该路径把完整 prefix embeddings 和 modality mask 一起输入：

```text
modality = 0 -> text branch
modality = 1 -> vision `_v` branch
```

实现上会把连续相同 modality 的 token 切成 runs；每个 run 走对应的 VLM branch，然后按原始 token 顺序拼回 Q/K/V，做共享 RoPE + attention，再按 runs 拆回 branch-specific output/MLP。这比之前固定的 `text segment + vision segment` 更接近真实 HY-VLA prefix，因为真实输入可能是 text/image token 混排。

- 新增 debug outputs：

```text
routed_prefix
routed_joint_vt
routed_joint
```

- direct C++ 等价性验证：

```text
mixed_joint_vt:  n=1280 sum ~= -344.428991805
routed_joint_vt: n=1280 sum ~= -344.428991805
diff: mean_abs = 0, max_abs = 0
```

这说明 routed path 在 `text tokens + visual tokens` 这个确定性 fixture 上与旧 mixed path 数学等价。

- routed server smoke：

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  VLA_HY_VLA_DEBUG_OUTPUT=routed_joint_vt \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6106 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py \
  --addr tcp://127.0.0.1:6106 \
  --routed \
  --prefix-f32 /tmp/hy_routed_prefix_7x2048.f32 \
  --modality 0 0 0 0 1 1 1
```

结果：

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= -344.428955078
min ~= -2.527388573
max ~= 0.454899967
```

- 结论：
  - server -> protobuf -> full-prefix embeddings -> modality routing -> suffix/action expert 链路已跑通；
  - 该路径是后续接入真实 Hy-ViT2/MEM visual token generator 和真实 tokenizer prefix 的更通用接口；
  - 当前仍是 debug/precomputed-token 阶段，下一步应继续补齐真实视觉 encoder 或正式 action decode/postprocess。

English:

- Added a generic routed prefix builder:

```text
build_hy_prefix_routed_layer(...)
```

This path takes full prefix embeddings plus a modality mask:

```text
modality = 0 -> text branch
modality = 1 -> vision `_v` branch
```

The implementation groups contiguous tokens with the same modality into runs, routes each run through the corresponding VLM branch, concatenates Q/K/V back in the original token order, applies shared RoPE + attention, then splits the result back into branch-specific output/MLP paths. This is closer to the real HY-VLA prefix layout than the earlier fixed `text segment + vision segment` path, because real inputs may interleave text and image tokens.

- Added debug outputs:

```text
routed_prefix
routed_joint_vt
routed_joint
```

- Direct C++ equivalence check:

```text
mixed_joint_vt:  n=1280 sum ~= -344.428991805
routed_joint_vt: n=1280 sum ~= -344.428991805
diff: mean_abs = 0, max_abs = 0
```

This confirms that the routed path is mathematically equivalent to the previous mixed path on the deterministic `text tokens + visual tokens` fixture.

- Routed server smoke:

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  VLA_HY_VLA_DEBUG_OUTPUT=routed_joint_vt \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6106 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py \
  --addr tcp://127.0.0.1:6106 \
  --routed \
  --prefix-f32 /tmp/hy_routed_prefix_7x2048.f32 \
  --modality 0 0 0 0 1 1 1
```

Result:

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= -344.428955078
min ~= -2.527388573
max ~= 0.454899967
```

- Conclusion:
  - the server -> protobuf -> full-prefix embeddings -> modality routing -> suffix/action expert path is now working;
  - this is the more general interface for later wiring in the real Hy-ViT2/MEM visual token generator and the real tokenizer prefix;
  - the current stage is still a debug/precomputed-token path, so the next work should be either the real visual encoder or the final action decode/postprocess path.

## 2026-06-17 default action postprocess / Default Action Postprocess

中文：

- 更新 `src/models/hy_vla.cpp` 默认推理路径：
  - 输入 state 的前 `real_state_dim=20` 维使用 `norm.state_mean/std` 做归一化；
  - 默认输出 action chunk 的前 `real_action_dim=20` 维使用 `norm.action_mean/std` 做反归一化；
  - padded 维度仍保留在 32 维输出中，和当前 `vla-server` action_dim 元数据一致。

- 为 parity/debug 保留 raw 输出：

```text
VLA_HY_VLA_RAW_ACTION=1
```

debug outputs（例如 `routed_joint_vt`、`mixed_joint_vt`、`joint_vt`、`vt`）仍直接返回 raw tensor，不走默认 postprocess。

- 重新构建通过：

```text
cmake --build build -j2
```

- 重新验证 routed debug server smoke，checksum 未变：

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= -344.428955078
min ~= -2.527388573
max ~= 0.454899967
```

English:

- Updated the default inference path in `src/models/hy_vla.cpp`:
  - the first `real_state_dim=20` state dimensions are normalized with `norm.state_mean/std`;
  - the first `real_action_dim=20` action dimensions are unnormalized with `norm.action_mean/std` by default;
  - padded dimensions remain in the 32-D output, matching the current `vla-server` action_dim metadata.

- Raw output remains available for parity/debug:

```text
VLA_HY_VLA_RAW_ACTION=1
```

Debug outputs such as `routed_joint_vt`, `mixed_joint_vt`, `joint_vt`, and `vt` still return raw tensors directly and do not use the default postprocess path.

- Rebuilt successfully:

```text
cmake --build build -j2
```

- Re-ran the routed debug server smoke; checksum is unchanged:

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= -344.428955078
min ~= -2.527388573
max ~= 0.454899967
```

## 2026-06-17 default routed prefix inference / Default Routed Prefix Inference

中文：

- 将 routed prefix path 接入默认 `predict()` 路径：
  - 当未设置 `VLA_HY_VLA_DEBUG_OUTPUT`；
  - 且请求提供 `precomputed_img_emb`；
  - 且 `attention_mask_n == n_img_views`；
  - HY-VLA 会把 `precomputed_img_emb` 解释为完整 prefix embeddings，把 `attention_mask` 解释为 modality mask：`0=text, 1=vision`。

- 默认 routed path 现在执行：

```text
full prefix embeddings + modality mask
  -> routed text/vision VLM prefix layers
  -> per-layer prefix K/V
  -> suffix/action expert denoise loop
  -> action postprocess unless VLA_HY_VLA_RAW_ACTION=1
```

- 新增 parity/debug 环境变量：

```text
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

用于让默认路径在 deterministic parity 时跳过 state normalize / action unnormalize。正式运行默认不设置它们。

- 更新 `scripts/smoke_hy_vla_server_precomputed.py`：
  - 新增 `--zero-noise`；
  - 新增 `--chunk` / `--action-dim`；
  - 可向 server 发送全零 action noise，使默认 denoise 输出可复现。

- raw deterministic routed smoke：

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  VLA_HY_VLA_RAW_STATE=1 \
  VLA_HY_VLA_RAW_ACTION=1 \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6107 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py \
  --addr tcp://127.0.0.1:6107 \
  --routed \
  --prefix-f32 /tmp/hy_routed_prefix_7x2048.f32 \
  --modality 0 0 0 0 1 1 1 \
  --zero-noise
```

结果：

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= 34.442897797
min ~= -0.045489997
max ~= 0.252738863
```

该结果与前面的 `routed_joint_vt` 一步结果一致：`x_1 = x_0 + dt * v_t`，其中 `x_0=0`、`dt=-0.1`、`v_t` checksum 约为 `-344.428955078`。

- 默认 postprocess routed smoke：

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6108 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py \
  --addr tcp://127.0.0.1:6108 \
  --routed \
  --prefix-f32 /tmp/hy_routed_prefix_7x2048.f32 \
  --modality 0 0 0 0 1 1 1 \
  --zero-noise
```

结果：

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= 246.459594727
min ~= -0.045678273
max ~= 1.256989002
```

- 结论：
  - HY-VLA 现在具备不依赖 debug output 的 routed prefix-conditioned server inference path；
  - precomputed full-prefix embeddings 可以作为真实 Hy-ViT2/MEM 接入前的稳定接口；
  - 下一步可以继续接真实 visual token generator，或继续扩展 routed path 到更多 layers/steps 的显存与性能验证。

English:

- Wired the routed prefix path into the default `predict()` path:
  - when `VLA_HY_VLA_DEBUG_OUTPUT` is not set;
  - and the request provides `precomputed_img_emb`;
  - and `attention_mask_n == n_img_views`;
  - HY-VLA interprets `precomputed_img_emb` as full prefix embeddings and `attention_mask` as the modality mask: `0=text, 1=vision`.

- The default routed path now executes:

```text
full prefix embeddings + modality mask
  -> routed text/vision VLM prefix layers
  -> per-layer prefix K/V
  -> suffix/action expert denoise loop
  -> action postprocess unless VLA_HY_VLA_RAW_ACTION=1
```

- Added parity/debug environment variables:

```text
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

These allow deterministic parity runs to bypass state normalization and action unnormalization in the default path. Normal runs should leave them unset.

- Updated `scripts/smoke_hy_vla_server_precomputed.py`:
  - added `--zero-noise`;
  - added `--chunk` / `--action-dim`;
  - the script can now send an all-zero action noise tensor to make default denoise output reproducible.

- Raw deterministic routed smoke:

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  VLA_HY_VLA_RAW_STATE=1 \
  VLA_HY_VLA_RAW_ACTION=1 \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6107 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py \
  --addr tcp://127.0.0.1:6107 \
  --routed \
  --prefix-f32 /tmp/hy_routed_prefix_7x2048.f32 \
  --modality 0 0 0 0 1 1 1 \
  --zero-noise
```

Result:

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= 34.442897797
min ~= -0.045489997
max ~= 0.252738863
```

This matches the earlier one-step `routed_joint_vt` result: `x_1 = x_0 + dt * v_t`, with `x_0=0`, `dt=-0.1`, and `v_t` checksum around `-344.428955078`.

- Default postprocess routed smoke:

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6108 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py \
  --addr tcp://127.0.0.1:6108 \
  --routed \
  --prefix-f32 /tmp/hy_routed_prefix_7x2048.f32 \
  --modality 0 0 0 0 1 1 1 \
  --zero-noise
```

Result:

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= 246.459594727
min ~= -0.045678273
max ~= 1.256989002
```

- Conclusion:
  - HY-VLA now has a routed prefix-conditioned server inference path that does not depend on debug outputs;
  - precomputed full-prefix embeddings provide a stable interface before wiring the real Hy-ViT2/MEM visual token generator;
  - next steps can be wiring the real visual token generator, or scaling the routed path to more layers/steps for memory and performance validation.

## 2026-06-17 HY prefix builder and frontend inventory / HY Prefix Builder and Frontend Inventory

中文：

- 新增 frontend inspection 脚本：

```text
scripts/inspect_hy_vla_frontend.py
```

该脚本无重依赖，直接读取 HY-VLA checkpoint 目录中的：

```text
config.json
preprocessor_config.json
tokenizer.json
tokenizer_config.json
chat_template.jinja
```

并输出 C++ 前端需要的 tokenizer / processor 常量。

- 本地 `/home/xuling/robotic_dataset/HY-VLA` 关键结果：

```text
patch_size = 16
merge_size = 2
resize_imgs_with_padding = [224, 224]
derived_grid_hw = [14, 14]
derived_merged_tokens_per_image = 49

bos_token_id = 120000
user_token_id = 120006
assistant_token_id = 120007
image token / placeholder no 669 = 120687
video token / placeholder no 670 = 120688
vision_start / placeholder no 666 = 120684
vision_end   / placeholder no 667 = 120685
vision_split / placeholder no 671 = 120689
```

- 额外验证默认 routed path 可扩展到 2 层 / 2 步：

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=2 \
  VLA_HY_VLA_DEBUG_LAYERS=2 \
  VLA_HY_VLA_DEBUG_STEPS=2 \
  VLA_HY_VLA_RAW_STATE=1 \
  VLA_HY_VLA_RAW_ACTION=1 \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6109 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py \
  --addr tcp://127.0.0.1:6109 \
  --routed \
  --prefix-f32 /tmp/hy_routed_prefix_7x2048.f32 \
  --modality 0 0 0 0 1 1 1 \
  --zero-noise
```

结果：

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= 55.042877197
min ~= -0.071270898
max ~= 0.440175295
```

- 新增 C++ host-side HY prefix builder：

```text
build_hy_vla_prefix_from_visual_tokens(...)
```

当默认 `predict()` 路径收到 `precomputed_img_emb` 但没有 `attention_mask` 时，HY-VLA 会把 `precomputed_img_emb` 解释为纯 visual patch tokens，并在 C++ 里构造 Python `embed_prefix()` 对应的 prefix：

```text
<bos><hy_User>
for each image:
    <vision_start>
    7 rows * (7 visual patch tokens + <vision_split>)
    <vision_end>
language tokens
```

生成的 modality mask 为：

```text
visual patch token -> 1 / vision branch
bos/user/vision_start/vision_split/vision_end/lang -> 0 / text branch
```

默认每张图 49 个 visual tokens，可用以下环境变量覆盖：

```text
VLA_HY_VLA_VIS_TOKENS_PER_IMAGE
```

- 新 prefix builder server smoke：

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  VLA_HY_VLA_RAW_STATE=1 \
  VLA_HY_VLA_RAW_ACTION=1 \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6110 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py \
  --addr tcp://127.0.0.1:6110 \
  --n-vis 147 \
  --zero-noise
```

这里 `147 = 3 images * 49 visual tokens/image`，C++ builder 自动生成完整 prefix，总 prefix 长度为：

```text
2 + 3 * (1 + 7 * (7 + 1) + 1) + 4 = 180 tokens
```

结果：

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= 24.536556244
min ~= -0.041626994
max ~= 0.165981978
```

- 结论：
  - C++ server 现在不仅支持 full-prefix embeddings，也支持更接近真实前端的 visual-patch-token 输入；
  - 真实 HyViT2/MEM 接入后，只需要产出每张图的 49 个 visual tokens，即可复用该 prefix builder；
  - 下一步可以开始迁移 HyViT2 visual encoder + MLP merger，或先用 Python 预计算真实 visual patch tokens 做 C++/Python prefix parity。

English:

- Added a frontend inspection script:

```text
scripts/inspect_hy_vla_frontend.py
```

The script is dependency-light and reads the HY-VLA checkpoint metadata directly:

```text
config.json
preprocessor_config.json
tokenizer.json
tokenizer_config.json
chat_template.jinja
```

It reports the tokenizer / processor constants required by the C++ frontend.

- Key local results for `/home/xuling/robotic_dataset/HY-VLA`:

```text
patch_size = 16
merge_size = 2
resize_imgs_with_padding = [224, 224]
derived_grid_hw = [14, 14]
derived_merged_tokens_per_image = 49

bos_token_id = 120000
user_token_id = 120006
assistant_token_id = 120007
image token / placeholder no 669 = 120687
video token / placeholder no 670 = 120688
vision_start / placeholder no 666 = 120684
vision_end   / placeholder no 667 = 120685
vision_split / placeholder no 671 = 120689
```

- Also verified that the default routed path scales to 2 layers / 2 steps:

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=2 \
  VLA_HY_VLA_DEBUG_LAYERS=2 \
  VLA_HY_VLA_DEBUG_STEPS=2 \
  VLA_HY_VLA_RAW_STATE=1 \
  VLA_HY_VLA_RAW_ACTION=1 \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6109 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py \
  --addr tcp://127.0.0.1:6109 \
  --routed \
  --prefix-f32 /tmp/hy_routed_prefix_7x2048.f32 \
  --modality 0 0 0 0 1 1 1 \
  --zero-noise
```

Result:

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= 55.042877197
min ~= -0.071270898
max ~= 0.440175295
```

- Added a C++ host-side HY prefix builder:

```text
build_hy_vla_prefix_from_visual_tokens(...)
```

When the default `predict()` path receives `precomputed_img_emb` without an `attention_mask`, HY-VLA now interprets `precomputed_img_emb` as pure visual patch tokens and constructs the Python `embed_prefix()` layout in C++:

```text
<bos><hy_User>
for each image:
    <vision_start>
    7 rows * (7 visual patch tokens + <vision_split>)
    <vision_end>
language tokens
```

The generated modality mask is:

```text
visual patch token -> 1 / vision branch
bos/user/vision_start/vision_split/vision_end/lang -> 0 / text branch
```

The default is 49 visual tokens per image; it can be overridden with:

```text
VLA_HY_VLA_VIS_TOKENS_PER_IMAGE
```

- New prefix-builder server smoke:

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  VLA_HY_VLA_RAW_STATE=1 \
  VLA_HY_VLA_RAW_ACTION=1 \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6110 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py \
  --addr tcp://127.0.0.1:6110 \
  --n-vis 147 \
  --zero-noise
```

Here `147 = 3 images * 49 visual tokens/image`; the C++ builder constructs the full prefix automatically. The full prefix length is:

```text
2 + 3 * (1 + 7 * (7 + 1) + 1) + 4 = 180 tokens
```

Result:

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= 24.536556244
min ~= -0.041626994
max ~= 0.165981978
```

- Conclusion:
  - the C++ server now supports both full-prefix embeddings and the more realistic visual-patch-token input;
  - once the real HyViT2/MEM frontend is wired, it only needs to produce 49 visual tokens per image to reuse this prefix builder;
  - next steps can be migrating the HyViT2 visual encoder + MLP merger, or using Python-precomputed real visual patch tokens for C++/Python prefix parity.

## 2026-06-18 routed prefix dump hooks / Routed Prefix Dump Hooks

中文：

- 为默认 routed path 增加 prefix dump 环境变量：

```text
VLA_HY_VLA_DUMP_PREFIX_F32=/tmp/hy_cpp_built_prefix_180x2048.f32
VLA_HY_VLA_DUMP_PREFIX_MASK_I32=/tmp/hy_cpp_built_prefix_180.i32
```

这两个 dump 在进入 routed VLM prefix graph 之前写出：
  - full prefix embeddings，shape `[n_prefix, hidden]`，float32；
  - routed modality mask，shape `[n_prefix]`，int32，`0=text branch`、`1=vision branch`。

- 同时修正 `scripts/inspect_hy_vla_frontend.py` 的 special-token inventory，补上 placeholder no 671：

```text
vision_split / placeholder no 671 = 120689
```

- dump smoke 命令：

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  VLA_HY_VLA_RAW_STATE=1 \
  VLA_HY_VLA_RAW_ACTION=1 \
  VLA_HY_VLA_DUMP_PREFIX_F32=/tmp/hy_cpp_built_prefix_180x2048.f32 \
  VLA_HY_VLA_DUMP_PREFIX_MASK_I32=/tmp/hy_cpp_built_prefix_180.i32 \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6111 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py \
  --addr tcp://127.0.0.1:6111 \
  --n-vis 147 \
  --zero-noise
```

action 输出保持上一轮结果：

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= 24.536556244
min ~= -0.041626994
max ~= 0.165981978
```

- dump 校验：

```text
prefix_floats = 368640
tokens = 180
hidden = 2048
prefix_sum ~= 19.391033173
mask_len = 180
mask_sum = 147
first20 mask = [0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1]
```

mask runs 符合 HY-VLA prefix 结构：

```text
<bos><hy_User><vision_start>
7 vision patch tokens + <vision_split>
...
<vision_end>
```

- 结论：
  - C++ prefix builder 现在可以产出可独立比较的 full prefix embedding 和 modality mask；
  - 后续 Python reference 若能 dump `embed_prefix()` 输出，即可直接做 prefix-level parity，而不必只依赖 action checksum。

English:

- Added prefix dump environment variables for the default routed path:

```text
VLA_HY_VLA_DUMP_PREFIX_F32=/tmp/hy_cpp_built_prefix_180x2048.f32
VLA_HY_VLA_DUMP_PREFIX_MASK_I32=/tmp/hy_cpp_built_prefix_180.i32
```

These dumps are written before entering the routed VLM prefix graph:
  - full prefix embeddings, shape `[n_prefix, hidden]`, float32;
  - routed modality mask, shape `[n_prefix]`, int32, where `0=text branch` and `1=vision branch`.

- Also fixed `scripts/inspect_hy_vla_frontend.py` special-token inventory by adding placeholder no 671:

```text
vision_split / placeholder no 671 = 120689
```

- Dump smoke command:

```text
env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  VLA_HY_VLA_RAW_STATE=1 \
  VLA_HY_VLA_RAW_ACTION=1 \
  VLA_HY_VLA_DUMP_PREFIX_F32=/tmp/hy_cpp_built_prefix_180x2048.f32 \
  VLA_HY_VLA_DUMP_PREFIX_MASK_I32=/tmp/hy_cpp_built_prefix_180.i32 \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6111 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmstable.gguf

python scripts/smoke_hy_vla_server_precomputed.py \
  --addr tcp://127.0.0.1:6111 \
  --n-vis 147 \
  --zero-noise
```

The action output remains unchanged from the previous run:

```text
response request_id=1 chunk=40 action_dim=32 n=1280
sum ~= 24.536556244
min ~= -0.041626994
max ~= 0.165981978
```

- Dump validation:

```text
prefix_floats = 368640
tokens = 180
hidden = 2048
prefix_sum ~= 19.391033173
mask_len = 180
mask_sum = 147
first20 mask = [0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1]
```

The mask runs match the HY-VLA prefix layout:

```text
<bos><hy_User><vision_start>
7 vision patch tokens + <vision_split>
...
<vision_end>
```

- Conclusion:
  - the C++ prefix builder can now produce independently comparable full prefix embeddings and modality masks;
  - once the Python reference can dump `embed_prefix()`, we can run prefix-level parity directly instead of relying only on action checksums.

## 2026-06-18 HY prefix builder parity / HY Prefix Builder Parity

中文：

- 新增轻量 Python reference 脚本：

```text
scripts/run_hy_vla_prefix_parity.py
```

该脚本不实例化完整 HY-VLA PyTorch 模型，只从 safetensors 读取：

```text
model.dual_tower.vlm.model.language_model.lm_head.weight
```

然后按 Python `embed_prefix()` 的结构构造同一套 synthetic fixture：

```text
<bos><hy_User>
for each image:
    <vision_start>
    7 rows * (7 visual patch tokens + <vision_split>)
    <vision_end>
language tokens
```

这样避免了本机 CPU 上完整 `HyVLA.from_pretrained()` 过慢的问题。

- 生成 Python reference dump：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python \
  scripts/run_hy_vla_prefix_parity.py \
  --dump-f32 /tmp/hy_py_prefix_180x2048.f32 \
  --dump-mask-i32 /tmp/hy_py_prefix_180.i32
```

输出：

```text
hy_prefix: shape=(180, 2048)
sum ~= 19.391033173
min ~= -0.671875
max ~= 0.408203125
hy_prefix_mask: shape=(180,) sum=147
first20=[0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1]
```

- 与 C++ dump 对比：

```text
python scripts/compare_f32_tensors.py \
  /tmp/hy_cpp_built_prefix_180x2048.f32 \
  /tmp/hy_py_prefix_180x2048.f32 \
  --shape 180 2048
```

结果：

```text
lhs sum=19.391033172607 min=-0.671875000000 max=0.408203125000
rhs sum=19.391033172607 min=-0.671875000000 max=0.408203125000
all mean_abs=0.000000000000 max_abs=0.000000000000 rmse=0.000000000000
```

mask 对比：

```text
mask_equal True
cpp_sum 147
py_sum 147
len 180
```

- 结论：
  - C++ host-side HY prefix builder 与轻量 Python reference 达到 bit-exact parity；
  - special token embedding、synthetic visual token 插入顺序、row split、vision_start/end、language token 拼接和 modality mask 全部对齐；
  - 下一步可以用同一机制接入 Python 真实 HyViT2/MEM visual tokens，或开始迁移 C++ HyViT2 visual encoder。

English:

- Added a lightweight Python reference script:

```text
scripts/run_hy_vla_prefix_parity.py
```

The script does not instantiate the full HY-VLA PyTorch model. It only reads the embedding tensor from safetensors:

```text
model.dual_tower.vlm.model.language_model.lm_head.weight
```

Then it constructs the same synthetic fixture using the Python `embed_prefix()` layout:

```text
<bos><hy_User>
for each image:
    <vision_start>
    7 rows * (7 visual patch tokens + <vision_split>)
    <vision_end>
language tokens
```

This avoids the slow full `HyVLA.from_pretrained()` path on the local CPU.

- Generated the Python reference dump:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python \
  scripts/run_hy_vla_prefix_parity.py \
  --dump-f32 /tmp/hy_py_prefix_180x2048.f32 \
  --dump-mask-i32 /tmp/hy_py_prefix_180.i32
```

Output:

```text
hy_prefix: shape=(180, 2048)
sum ~= 19.391033173
min ~= -0.671875
max ~= 0.408203125
hy_prefix_mask: shape=(180,) sum=147
first20=[0,0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1]
```

- Compared against the C++ dump:

```text
python scripts/compare_f32_tensors.py \
  /tmp/hy_cpp_built_prefix_180x2048.f32 \
  /tmp/hy_py_prefix_180x2048.f32 \
  --shape 180 2048
```

Result:

```text
lhs sum=19.391033172607 min=-0.671875000000 max=0.408203125000
rhs sum=19.391033172607 min=-0.671875000000 max=0.408203125000
all mean_abs=0.000000000000 max_abs=0.000000000000 rmse=0.000000000000
```

Mask comparison:

```text
mask_equal True
cpp_sum 147
py_sum 147
len 180
```

- Conclusion:
  - the C++ host-side HY prefix builder is bit-exact with the lightweight Python reference;
  - special token embeddings, synthetic visual-token insertion order, row splits, vision_start/end, language-token concatenation, and modality masks are all aligned;
  - next steps can use the same mechanism with real Python-precomputed HyViT2/MEM visual tokens, or start migrating the C++ HyViT2 visual encoder.

## 2026-06-19 HyViT2 vision inventory and stable GGUF names / HyViT2 Vision Inventory and Stable GGUF Names

中文：

- 新增视觉权重检查脚本：

```text
scripts/inspect_hy_vla_vision_weights.py
```

该脚本直接读取 HY-VLA `model.safetensors`，输出 HyViT2/Merger 视觉前端的紧凑 inventory。

- 本地 checkpoint 视觉前端统计：

```text
visual tensor_count = 335
visual params = 450,371,568

vision_tower.patch_embed params = 885,888
vision_tower.pos_embed params = 18,874,368
vision_tower.blocks params = 411,466,608
merger.proj params = 6,557,696
merger.pooler params = 12,587,008
```

- 推断出的视觉结构：

```text
HyViT2 width = 1152
layers = 27
heads = 16
head_dim = 72
patch_size = 16
input image = 224x224
patch grid = 14x14
merged visual tokens/image = 49
merger out channels = 2048
```

- 关键 tensor：

```text
vision_tower.patch_embed.proj.weight  [1152, 3, 16, 16]  BF16
vision_tower.pos_embed                [1, 16384, 1152]   BF16
merger.proj1.weight                   [2048, 1152]       BF16
merger.proj2.weight                   [2048, 2048]       BF16
merger.pooler.predictor.0.weight      [2048, 4096]       BF16
merger.pooler.predictor.2.weight      [2048, 2048]       BF16
```

- 更新 `scripts/convert_hy_vla_to_gguf.py`：
  - 新增 `_vision_tensor_map()`；
  - full scope 现在为全部视觉前端生成稳定短名；
  - 避免 C++ 视觉 loader 后续依赖 hash fallback 名称。

稳定短名示例：

```text
vision.patch_embed.weight
vision.patch_embed.bias
vision.pos_embed
vision.blk.0.norm1.weight
vision.blk.0.attn_qkv.weight
vision.blk.0.attn_proj.weight
vision.blk.0.ffn_fc1.weight
vision.blk.26.ffn_fc2.bias
vision.merger.proj1.weight
vision.merger.proj2.weight
vision.merger.pooler.fc0.weight
vision.merger.pooler.fc2.weight
```

- dry-run 验证：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python \
  scripts/convert_hy_vla_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/HY-VLA \
  --scope full \
  --dry-run
```

结果：

```text
scope=full tensors=1628 estimated_tensor_bytes=8633.9 MiB
dry-run: tensor map validated; no GGUF was written
```

- 生成新的 full GGUF：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python \
  scripts/convert_hy_vla_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/HY-VLA \
  --scope full \
  --out /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf
```

输出文件：

```text
/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf
size = 8.5 GiB / 9,053,646,048 bytes
```

- C++ loader smoke：

```text
timeout 45 env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6112 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf
```

结果：

```text
vla(hy_vla): loaded resident weights 0.87 GiB (suffix expert + text_layers=1 prefix_vision=yes; full GGUF may contain VLM/vision too)
vla-server: bound to tcp://127.0.0.1:6112. ready.
```

- GGUF stable-name 检查：

```text
vision.patch_embed.weight           True
vision.pos_embed                    True
vision.blk.0.attn_qkv.weight        True
vision.blk.26.ffn_fc2.bias          True
vision.merger.pooler.fc0.weight     True
vlm.blk.0.attn_norm.weight          True
vision_count = 335
```

- 结论：
  - HY-VLA full GGUF 现在同时拥有 stable VLM text/vision branch 名称和 stable HyViT2/Merger 视觉前端名称；
  - 下一步 C++ 视觉 loader 可以直接按 `vision.*` 名称加载 patch_embed / pos_embed / ViT blocks / merger；
  - 当前磁盘剩余约 19 GiB，继续生成更多 full GGUF 前需要注意空间。

English:

- Added a vision weight inspection script:

```text
scripts/inspect_hy_vla_vision_weights.py
```

The script reads the HY-VLA `model.safetensors` directly and reports a compact inventory for the HyViT2/Merger visual frontend.

- Local checkpoint visual frontend statistics:

```text
visual tensor_count = 335
visual params = 450,371,568

vision_tower.patch_embed params = 885,888
vision_tower.pos_embed params = 18,874,368
vision_tower.blocks params = 411,466,608
merger.proj params = 6,557,696
merger.pooler params = 12,587,008
```

- Inferred visual architecture:

```text
HyViT2 width = 1152
layers = 27
heads = 16
head_dim = 72
patch_size = 16
input image = 224x224
patch grid = 14x14
merged visual tokens/image = 49
merger out channels = 2048
```

- Key tensors:

```text
vision_tower.patch_embed.proj.weight  [1152, 3, 16, 16]  BF16
vision_tower.pos_embed                [1, 16384, 1152]   BF16
merger.proj1.weight                   [2048, 1152]       BF16
merger.proj2.weight                   [2048, 2048]       BF16
merger.pooler.predictor.0.weight      [2048, 4096]       BF16
merger.pooler.predictor.2.weight      [2048, 2048]       BF16
```

- Updated `scripts/convert_hy_vla_to_gguf.py`:
  - added `_vision_tensor_map()`;
  - full scope now emits stable short names for the whole visual frontend;
  - this avoids requiring the future C++ visual loader to depend on hash fallback names.

Stable-name examples:

```text
vision.patch_embed.weight
vision.patch_embed.bias
vision.pos_embed
vision.blk.0.norm1.weight
vision.blk.0.attn_qkv.weight
vision.blk.0.attn_proj.weight
vision.blk.0.ffn_fc1.weight
vision.blk.26.ffn_fc2.bias
vision.merger.proj1.weight
vision.merger.proj2.weight
vision.merger.pooler.fc0.weight
vision.merger.pooler.fc2.weight
```

- Dry-run validation:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python \
  scripts/convert_hy_vla_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/HY-VLA \
  --scope full \
  --dry-run
```

Result:

```text
scope=full tensors=1628 estimated_tensor_bytes=8633.9 MiB
dry-run: tensor map validated; no GGUF was written
```

- Generated a new full GGUF:

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python \
  scripts/convert_hy_vla_to_gguf.py \
  --ckpt /home/xuling/robotic_dataset/HY-VLA \
  --scope full \
  --out /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf
```

Output file:

```text
/home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf
size = 8.5 GiB / 9,053,646,048 bytes
```

- C++ loader smoke:

```text
timeout 45 env \
  VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  ./build/vla-server \
    --bind tcp://127.0.0.1:6112 \
    /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf
```

Result:

```text
vla(hy_vla): loaded resident weights 0.87 GiB (suffix expert + text_layers=1 prefix_vision=yes; full GGUF may contain VLM/vision too)
vla-server: bound to tcp://127.0.0.1:6112. ready.
```

- GGUF stable-name check:

```text
vision.patch_embed.weight           True
vision.pos_embed                    True
vision.blk.0.attn_qkv.weight        True
vision.blk.26.ffn_fc2.bias          True
vision.merger.pooler.fc0.weight     True
vlm.blk.0.attn_norm.weight          True
vision_count = 335
```

- Conclusion:
  - the HY-VLA full GGUF now has stable VLM text/vision branch names and stable HyViT2/Merger visual frontend names;
  - the next C++ visual loader can load patch_embed / pos_embed / ViT blocks / merger directly via `vision.*` names;
  - disk free space is now around 19 GiB, so further full-GGUF generation should be done carefully.

## 2026-06-19 C++ HyViT2 resident loader / C++ HyViT2 Resident Loader

中文：

- 在 `src/models/hy_vla.cpp` 中新增 HyViT2/Merger 视觉前端 resident 权重结构：
  - `HyVisionW`
  - `HyVisionBlockW`
- 新增按需加载开关：

```text
VLA_HY_VLA_VISION_LAYERS
```

- 行为：
  - 不设置：不加载视觉前端 resident 权重，保持现有 suffix/prefix parity 路径的内存占用；
  - `VLA_HY_VLA_VISION_LAYERS=0`：加载 patch embedding、position embedding、merger/pooler base 权重；
  - `VLA_HY_VLA_VISION_LAYERS=N`：加载 base + 前 N 个 HyViT2 blocks；
  - `VLA_HY_VLA_VISION_LAYERS=all`：加载 base + 全部 27 个 HyViT2 blocks。
- loader 使用 stable GGUF 名称，例如：

```text
vision.patch_embed.weight
vision.pos_embed
vision.blk.0.attn_qkv.weight
vision.blk.0.attn_proj.weight
vision.blk.0.ffn_fc1.weight
vision.merger.pooler.fc0.weight
```

- 如果设置了 `VLA_HY_VLA_VISION_LAYERS`，但 GGUF 缺少 stable vision frontend tensor，loader 会直接报错并提示使用：

```text
hy_vla_full_bf16_vlmvisionstable.gguf
```

- 编译验证：

```text
cmake --build build -j2
```

结果：通过。

- C++ loader smoke：

```text
timeout 45 env VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  ./build/vla-server --bind tcp://127.0.0.1:6113 \
  /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf

timeout 45 env VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_VISION_LAYERS=0 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  ./build/vla-server --bind tcp://127.0.0.1:6114 \
  /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf

timeout 60 env VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_VISION_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  ./build/vla-server --bind tcp://127.0.0.1:6115 \
  /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf
```

结果：

```text
vision_frontend=no  vision_layers=0  resident=0.87 GiB
vision_frontend=yes vision_layers=0  resident=0.98 GiB
vision_frontend=yes vision_layers=1  resident=1.01 GiB
```

- 说明：
  - 这一阶段完成的是视觉前端权重 resident loader；
  - patch embedding、HyViT2 block forward、pos-embed resize/interpolation、merger/pooler graph 尚未接入；
  - 现有 routed prefix / deterministic fixture 路径不受影响。

English:

- Added resident weight structs for the HyViT2/Merger visual frontend in `src/models/hy_vla.cpp`:
  - `HyVisionW`
  - `HyVisionBlockW`
- Added an opt-in loader switch:

```text
VLA_HY_VLA_VISION_LAYERS
```

- Behavior:
  - unset: do not load resident visual-frontend weights, preserving the current suffix/prefix parity memory footprint;
  - `VLA_HY_VLA_VISION_LAYERS=0`: load patch embedding, position embedding, and merger/pooler base weights;
  - `VLA_HY_VLA_VISION_LAYERS=N`: load base + the first N HyViT2 blocks;
  - `VLA_HY_VLA_VISION_LAYERS=all`: load base + all 27 HyViT2 blocks.
- The loader uses stable GGUF names such as:

```text
vision.patch_embed.weight
vision.pos_embed
vision.blk.0.attn_qkv.weight
vision.blk.0.attn_proj.weight
vision.blk.0.ffn_fc1.weight
vision.merger.pooler.fc0.weight
```

- If `VLA_HY_VLA_VISION_LAYERS` is set but the GGUF lacks stable vision-frontend tensors, the loader fails early and recommends:

```text
hy_vla_full_bf16_vlmvisionstable.gguf
```

- Build validation:

```text
cmake --build build -j2
```

Result: passed.

- C++ loader smoke:

```text
timeout 45 env VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  ./build/vla-server --bind tcp://127.0.0.1:6113 \
  /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf

timeout 45 env VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_VISION_LAYERS=0 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  ./build/vla-server --bind tcp://127.0.0.1:6114 \
  /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf

timeout 60 env VLA_HY_VLA_TEXT_LAYERS=1 \
  VLA_HY_VLA_VISION_LAYERS=1 \
  VLA_HY_VLA_DEBUG_LAYERS=1 \
  VLA_HY_VLA_DEBUG_STEPS=1 \
  ./build/vla-server --bind tcp://127.0.0.1:6115 \
  /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf
```

Results:

```text
vision_frontend=no  vision_layers=0  resident=0.87 GiB
vision_frontend=yes vision_layers=0  resident=0.98 GiB
vision_frontend=yes vision_layers=1  resident=1.01 GiB
```

- Notes:
  - this stage completes the resident loader for visual-frontend weights;
  - patch embedding, HyViT2 block forward, pos-embed resize/interpolation, and merger/pooler graph are not wired yet;
  - existing routed-prefix / deterministic-fixture paths remain unchanged.

## 2026-06-19 HyViT2 block helper and 224-pos cache / HyViT2 Block Helper and 224-Pos Cache

中文：

- 在 `src/models/hy_vla.cpp` 中新增 HyViT2 block graph helper：

```text
build_hy_vision_block(...)
```

- 该 helper 对齐 Python `_ViTBlock` 主干顺序：

```text
LayerNorm -> QKV attention -> proj residual -> LayerNorm -> GELU MLP residual
```

- 在视觉前端 loader 中新增 generated resident tensor：

```text
vision.pos_embed_14x14.f32
```

- 该 tensor 由 GGUF 中的 `vision.pos_embed` 128x128 表通过 bilinear resize 得到，用于 224x224 输入对应的 14x14 patch tokens。
- 编译验证：

```text
cmake --build build -j2
```

结果：通过。

- loader smoke 复测：

```text
VLA_HY_VLA_VISION_LAYERS=0 -> resident=0.98 GiB
VLA_HY_VLA_VISION_LAYERS=1 -> resident=1.01 GiB
```

- 说明：
  - HyViT2 block helper 目前尚未接入默认 `predict()`；
  - 真实 `images -> patch embedding -> HyViT2 blocks -> merger -> 49 visual tokens` forward 仍是下一步；
  - `precomputed_img_emb` 继续表示 2048 维 merger 后 visual/prefix tokens，不改 server 现有校验规则。

English:

- Added a HyViT2 block graph helper in `src/models/hy_vla.cpp`:

```text
build_hy_vision_block(...)
```

- The helper follows the Python `_ViTBlock` backbone order:

```text
LayerNorm -> QKV attention -> proj residual -> LayerNorm -> GELU MLP residual
```

- Added a generated resident tensor in the visual frontend loader:

```text
vision.pos_embed_14x14.f32
```

- This tensor is bilinearly resized from the 128x128 `vision.pos_embed` table in GGUF and targets 14x14 patch tokens for 224x224 inputs.
- Build validation:

```text
cmake --build build -j2
```

Result: passed.

- Loader smoke re-check:

```text
VLA_HY_VLA_VISION_LAYERS=0 -> resident=0.98 GiB
VLA_HY_VLA_VISION_LAYERS=1 -> resident=1.01 GiB
```

- Notes:
  - the HyViT2 block helper is not wired into default `predict()` yet;
  - the real `images -> patch embedding -> HyViT2 blocks -> merger -> 49 visual tokens` forward path is still the next step;
  - `precomputed_img_emb` continues to mean 2048-d post-merger visual/prefix tokens, so the existing server validation remains unchanged.

## 2026-06-19 Vision patch/VIT debug smoke / Vision Patch/VIT Debug Smoke

中文：

- 新增 224x224 HY-VLA 图像预处理 helper：

```text
preprocess_hy_image_chw_224(...)
```

- 当前版本要求输入已经是 224x224，像素按训练配置归一化为 `[-1, 1]`。后续还需要补完整 resize-with-pad。
- 新增 patch embedding graph helper：

```text
build_hy_vision_patch_embed_224(...)
```

- 新增 debug 输出：

```text
VLA_HY_VLA_DEBUG_OUTPUT=vision_patch
VLA_HY_VLA_DEBUG_OUTPUT=vision_vit
```

- `vision_patch` 返回 224x224 单图经过 patch embedding + 14x14 pos 后的 196x1152 tensor。
- `vision_vit` 在 `vision_patch` 基础上继续跑已加载的前 N 个 HyViT2 blocks；N 由 `VLA_HY_VLA_VISION_LAYERS` 和 `VLA_HY_VLA_DEBUG_LAYERS` 共同限制。
- 遇到并修复的问题：
  - CUDA `ggml_conv_2d` im2col 路径不接受 BF16 conv 中间输出；
  - `vision.patch_embed.weight` resident dtype 改为 F32；
  - HyViT2 q/k/v 从 packed qkv view 切出后需要 `ggml_cont()`，否则 `ggml_reshape_3d` 会因为 non-contiguous view 断言失败。
- Smoke 输入：一张全零 `RGB_U8` 224x224 图。
- `vision_patch` smoke：

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=0
VLA_HY_VLA_DEBUG_OUTPUT=vision_patch
```

结果：

```text
n = 225792 = 196 * 1152
sum = 16667.13671875
mean = 0.07381632924079895
min = -19.202808380126953
max = 24.6516056060791
```

- `vision_vit` 1-block smoke：

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=1
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_OUTPUT=vision_vit
```

结果：

```text
n = 225792 = 196 * 1152
sum = 23125.54296875
mean = 0.10241967439651489
min = -15.453181266784668
max = 31.0695743560791
```

- 当前状态：
  - C++ 已能从 `images` 请求跑通 HY-VLA vision patch embedding 和前 1 个 HyViT2 block；
  - 下一步是实现/验证 merger + normalized 2x pooler，将 196x1152 压到 49x2048；
  - 再下一步才是把该 49-token 输出接入现有 HY prefix builder/default predict 路径。

English:

- Added a 224x224 HY-VLA image preprocessing helper:

```text
preprocess_hy_image_chw_224(...)
```

- The current version expects already-224x224 input and normalizes pixels to `[-1, 1]` according to the training config. Full resize-with-pad is still pending.
- Added a patch embedding graph helper:

```text
build_hy_vision_patch_embed_224(...)
```

- Added debug outputs:

```text
VLA_HY_VLA_DEBUG_OUTPUT=vision_patch
VLA_HY_VLA_DEBUG_OUTPUT=vision_vit
```

- `vision_patch` returns the 196x1152 tensor after patch embedding + 14x14 positional embedding for one 224x224 image.
- `vision_vit` continues from `vision_patch` through the loaded first N HyViT2 blocks; N is bounded by both `VLA_HY_VLA_VISION_LAYERS` and `VLA_HY_VLA_DEBUG_LAYERS`.
- Issues found and fixed:
  - CUDA `ggml_conv_2d` im2col does not accept a BF16 conv intermediate;
  - `vision.patch_embed.weight` resident dtype was changed to F32;
  - HyViT2 q/k/v views sliced from packed qkv need `ggml_cont()` before `ggml_reshape_3d`.
- Smoke input: one all-zero `RGB_U8` 224x224 image.
- `vision_patch` smoke:

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=0
VLA_HY_VLA_DEBUG_OUTPUT=vision_patch
```

Result:

```text
n = 225792 = 196 * 1152
sum = 16667.13671875
mean = 0.07381632924079895
min = -19.202808380126953
max = 24.6516056060791
```

- `vision_vit` 1-block smoke:

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=1
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_OUTPUT=vision_vit
```

Result:

```text
n = 225792 = 196 * 1152
sum = 23125.54296875
mean = 0.10241967439651489
min = -15.453181266784668
max = 31.0695743560791
```

- Current state:
  - C++ can now run HY-VLA vision patch embedding and the first HyViT2 block from an `images` request;
  - next step: implement/validate the merger + normalized 2x pooler to reduce 196x1152 to 49x2048;
  - after that, wire the 49-token output into the existing HY prefix builder/default predict path.

## 2026-06-19 Vision merger and image default path / Vision Merger and Image Default Path

中文：

- 新增 HyViT2/Merger graph helper：

```text
build_hy_vision_merger_14(...)
run_hy_vision_frontend_224(...)
```

- `build_hy_vision_merger_14(...)` 对齐 Python `_HYViT2MLPProjector` 的主干：

```text
proj1 -> normalized 2x pooler -> GELU -> proj2
```

- 其中 normalized 2x pooler 按 7x7 个 2x2 cell 展开，每个 cell 做：

```text
new_x = 4 patch tokens
pooled_x = mean(new_x) repeated to 4 tokens
score = predictor(concat(new_x, pooled_x))
out = sum(new_x * softmax(score, dim=4-patch-axis))
```

- 新增 debug 输出：

```text
VLA_HY_VLA_DEBUG_OUTPUT=vision_merger
```

- `vision_merger` 返回 49x2048 visual tokens，可直接接入 HY prefix builder。
- Smoke 输入：一张全零 `RGB_U8` 224x224 图。
- `vision_merger` 0-block smoke：

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=0
VLA_HY_VLA_DEBUG_OUTPUT=vision_merger
```

结果：

```text
n = 100352 = 49 * 2048
sum = 4057.60205078125
mean = 0.040433693677186966
min = -7.22869873046875
max = 25.05078125
```

- `vision_merger` 1-block smoke：

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=1
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_OUTPUT=vision_merger
```

结果：

```text
n = 100352 = 49 * 2048
sum = 4021.906982421875
mean = 0.040077995508909225
min = -6.72869873046875
max = 24.30078125
```

- 默认 `predict()` 路径已接入 `images`：
  - 如果请求没有 `precomputed_img_emb`，但有 `images`，且 `VLA_HY_VLA_VISION_LAYERS` 已启用；
  - C++ 会先跑 `run_hy_vision_frontend_224(...)` 得到每张图 49 个 2048 维 visual tokens；
  - 然后复用已有 `build_hy_vla_prefix_from_visual_tokens(...)` 和 routed prefix/action path。
- image-only 默认路径 smoke：

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=0
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_STEPS=1
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

请求只带：

```text
images = 1 x RGB_U8 224x224 all-zero image
lang_tokens = [120000, 120001, 120020, 7]
state = 32 zeros
noise = 40 x 32 zeros
```

结果：

```text
chunk = 40
action_dim = 32
n = 1280
sum = 26.71707534790039
mean = 0.02087271586060524
min = -0.04124606028199196
max = 0.1796891838312149
server timing: total=278.0 ms, vision=238.7 ms, inference=31.9 ms
```

- 当前 caveat：
  - 只支持输入已经是 224x224 的图像；
  - 尚未实现 resize-with-pad；
  - full 27-block vision path 尚未在 8GB 本机做 smoke；
  - 还未做 Python/C++ vision merger parity。

English:

- Added HyViT2/Merger graph helpers:

```text
build_hy_vision_merger_14(...)
run_hy_vision_frontend_224(...)
```

- `build_hy_vision_merger_14(...)` follows the Python `_HYViT2MLPProjector` backbone:

```text
proj1 -> normalized 2x pooler -> GELU -> proj2
```

- The normalized 2x pooler expands the 14x14 grid into 7x7 2x2 cells. Each cell performs:

```text
new_x = 4 patch tokens
pooled_x = mean(new_x) repeated to 4 tokens
score = predictor(concat(new_x, pooled_x))
out = sum(new_x * softmax(score, dim=4-patch-axis))
```

- Added debug output:

```text
VLA_HY_VLA_DEBUG_OUTPUT=vision_merger
```

- `vision_merger` returns 49x2048 visual tokens, directly compatible with the HY prefix builder.
- Smoke input: one all-zero `RGB_U8` 224x224 image.
- `vision_merger` 0-block smoke:

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=0
VLA_HY_VLA_DEBUG_OUTPUT=vision_merger
```

Result:

```text
n = 100352 = 49 * 2048
sum = 4057.60205078125
mean = 0.040433693677186966
min = -7.22869873046875
max = 25.05078125
```

- `vision_merger` 1-block smoke:

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=1
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_OUTPUT=vision_merger
```

Result:

```text
n = 100352 = 49 * 2048
sum = 4021.906982421875
mean = 0.040077995508909225
min = -6.72869873046875
max = 24.30078125
```

- The default `predict()` path now accepts `images`:
  - if the request has no `precomputed_img_emb`, but has `images`, and `VLA_HY_VLA_VISION_LAYERS` is enabled;
  - C++ first runs `run_hy_vision_frontend_224(...)` to produce 49 visual tokens per image;
  - then it reuses the existing `build_hy_vla_prefix_from_visual_tokens(...)` and routed prefix/action path.
- Image-only default path smoke:

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=0
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_STEPS=1
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

Request contains only:

```text
images = 1 x RGB_U8 224x224 all-zero image
lang_tokens = [120000, 120001, 120020, 7]
state = 32 zeros
noise = 40 x 32 zeros
```

Result:

```text
chunk = 40
action_dim = 32
n = 1280
sum = 26.71707534790039
mean = 0.02087271586060524
min = -0.04124606028199196
max = 0.1796891838312149
server timing: total=278.0 ms, vision=238.7 ms, inference=31.9 ms
```

- Current caveats:
  - input images must already be 224x224;
  - resize-with-pad is not implemented yet;
  - the full 27-block vision path has not been smoked on the local 8GB GPU;
  - Python/C++ vision merger parity is still pending.

## 2026-06-19 C++ resize-with-pad for HY images / C++ Resize-With-Pad for HY Images

中文：

- `preprocess_hy_image_chw_224(...)` 已从“只接受 224x224”扩展为支持任意正尺寸 RGB 图。
- C++ 逻辑按 Python `resize_with_pad(img, 224, 224, pad_value=0, mode="bilinear")` 对齐：

```text
ratio = max(width / 224, height / 224)
resized_height = int(height / ratio)
resized_width = int(width / ratio)
bilinear resize with align_corners=False style sampling
center pad with pixel value 0 in [0, 1]
normalize to [-1, 1]
```

- 新增非方形图默认路径 smoke：

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=0
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_STEPS=1
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

请求：

```text
images = 1 x RGB_U8 320x240 gradient image
lang_tokens = [120000, 120001, 120020, 7]
state = 32 zeros
noise = 40 x 32 zeros
```

结果：

```text
chunk = 40
action_dim = 32
n = 1280
sum = 26.376846313476562
mean = 0.020606910809874535
min = -0.04079088196158409
max = 0.176898792386055
server timing: total=253.2 ms, vision=216.1 ms, inference=31.8 ms
```

- 当前 caveat 更新：
  - 已支持任意正尺寸输入图像；
  - bilinear resize 需要后续和 Python `F.interpolate(..., align_corners=False)` 做数值 parity；
  - full 27-block vision path 尚未在 8GB 本机做 smoke；
  - Python/C++ vision merger parity 仍待做。

English:

- `preprocess_hy_image_chw_224(...)` was extended from “224x224 only” to arbitrary positive RGB image sizes.
- The C++ logic follows Python `resize_with_pad(img, 224, 224, pad_value=0, mode="bilinear")`:

```text
ratio = max(width / 224, height / 224)
resized_height = int(height / ratio)
resized_width = int(width / ratio)
bilinear resize with align_corners=False style sampling
center pad with pixel value 0 in [0, 1]
normalize to [-1, 1]
```

- Added a non-square image default-path smoke:

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=0
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_STEPS=1
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

Request:

```text
images = 1 x RGB_U8 320x240 gradient image
lang_tokens = [120000, 120001, 120020, 7]
state = 32 zeros
noise = 40 x 32 zeros
```

Result:

```text
chunk = 40
action_dim = 32
n = 1280
sum = 26.376846313476562
mean = 0.020606910809874535
min = -0.04079088196158409
max = 0.176898792386055
server timing: total=253.2 ms, vision=216.1 ms, inference=31.8 ms
```

- Updated caveats:
  - arbitrary positive input image sizes are now supported;
  - bilinear resize still needs numeric parity against Python `F.interpolate(..., align_corners=False)`;
  - the full 27-block vision path has not been smoked on the local 8GB GPU;
  - Python/C++ vision merger parity is still pending.

## 2026-06-19 Full 27-block HyViT2 smoke / Full 27-Block HyViT2 Smoke

中文：

- 在本机 3070Ti Laptop 8GB 上完成 `VLA_HY_VLA_VISION_LAYERS=all` smoke。
- 配置：

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=all
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_STEPS=1
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

- 加载结果：

```text
resident = 1.75 GiB
vision_layers = 27
```

- 请求：

```text
images = 1 x RGB_U8 224x224 all-zero image
lang_tokens = [120000, 120001, 120020, 7]
state = 32 zeros
noise = 40 x 32 zeros
```

- 结果：

```text
chunk = 40
action_dim = 32
n = 1280
sum = 26.401439666748047
mean = 0.020626123994588852
min = -0.039936281740665436
max = 0.1785019487142563
server timing: total=281.9 ms, vision=241.8 ms, inference=33.3 ms
```

- 结论：
  - C++ 已能加载并运行完整 27-block HyViT2 visual frontend；
  - 当前 smoke 的 text/action 仍是 1 block / 1 step debug 配置；
  - 下一阶段应做 Python/C++ vision frontend parity，尤其是 resize、pos embedding resize、pooler softmax axis 和 GELU erf 对齐。

English:

- Completed a `VLA_HY_VLA_VISION_LAYERS=all` smoke on the local 3070Ti Laptop 8GB GPU.
- Configuration:

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=all
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_STEPS=1
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

- Load result:

```text
resident = 1.75 GiB
vision_layers = 27
```

- Request:

```text
images = 1 x RGB_U8 224x224 all-zero image
lang_tokens = [120000, 120001, 120020, 7]
state = 32 zeros
noise = 40 x 32 zeros
```

- Result:

```text
chunk = 40
action_dim = 32
n = 1280
sum = 26.401439666748047
mean = 0.020626123994588852
min = -0.039936281740665436
max = 0.1785019487142563
server timing: total=281.9 ms, vision=241.8 ms, inference=33.3 ms
```

- Conclusion:
  - C++ can now load and run the complete 27-block HyViT2 visual frontend;
  - this smoke still uses 1 text/action block and 1 flow step for the rest of the model;
  - the next stage should run Python/C++ vision-frontend parity, especially for resize, pos-embedding resize, pooler softmax axis, and GELU erf alignment.

## 2026-06-19 Vision frontend parity and QKV bias fix / Vision Frontend Parity and QKV Bias Fix

中文：

- 新增两个 parity 工具：

```text
scripts/run_hy_vla_vision_reference.py
scripts/run_hy_vla_vision_cpp_debug.py
```

- `run_hy_vla_vision_reference.py` 默认使用 `--load-mode visual-only`，只实例化 Python HY visual frontend，并只加载：

```text
model.dual_tower.vlm.model.visual.*
```

这样可以避免在本机 8GB GPU 上加载完整 HyVLA Python 模型导致 OOM。

- `run_hy_vla_vision_cpp_debug.py` 通过 `vla-server` 请求 C++ debug 输出，并 dump raw f32。
- 支持的对齐点：

```text
vision_patch
vision_vit
vision_merger
```

- Python 完整模型搬到 CUDA 的尝试失败，原因是本机 8GB GPU 显存不足：

```text
torch.OutOfMemoryError: CUDA out of memory
```

- 改用 visual-only reference 后，`vision_patch` Python dump 成功。
- `vision_patch` parity：

```text
C++ default vs Python bf16:
mean_abs = 0.001013011788
max_abs  = 0.091701507568

C++ default vs Python f32:
mean_abs = 0.000000018800
max_abs  = 0.000013351440
```

结论：图像预处理、patch embedding、14x14 pos resize 在 F32 语义下基本对齐；bf16 差异是 dtype 路径造成的。

- `vision_merger` 0-block parity：

```text
C++ BF16 resident vs Python f32:
mean_abs = 0.001899844967
max_abs  = 0.071424484253

C++ F32 resident vs Python f32:
mean_abs = 0.000163134813
max_abs  = 0.002740859985
```

- 发现并修复 HyViT2 block 的一个结构 bug：

```text
C++ 原来漏加 attn.qkv.bias
Python _ViTAttention 使用 self.qkv(x)，包含 bias
```

修复：

```text
qkv = mm_w(qkv_w, x_norm) + qkv_b
```

- 修复前 `vision_vit` 1-block F32 parity：

```text
mean_abs = 0.126228794456
max_abs  = 2.684199333191
```

- 修复后 `vision_vit` 1-block F32 parity：

```text
mean_abs = 0.000083761741
max_abs  = 0.002095699310
```

- 修复后 `vision_merger` 1-block F32 parity：

```text
mean_abs = 0.000187682861
max_abs  = 0.002531051636
```

- qkv bias 修复后，重新跑 full 27-block HyViT2 + 1 text/action block + 1 flow step smoke：

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=all
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_STEPS=1
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

结果：

```text
chunk = 40
action_dim = 32
n = 1280
sum = 33.47308349609375
mean = 0.026150846853852272
min = -0.04530525952577591
max = 0.2177203893661499
server timing: total=596.6 ms, vision=556.9 ms, inference=26.9 ms
```

English:

- Added two parity tools:

```text
scripts/run_hy_vla_vision_reference.py
scripts/run_hy_vla_vision_cpp_debug.py
```

- `run_hy_vla_vision_reference.py` defaults to `--load-mode visual-only`, instantiating only the Python HY visual frontend and loading only:

```text
model.dual_tower.vlm.model.visual.*
```

This avoids OOM from moving the full Python HyVLA model onto the local 8GB GPU.

- `run_hy_vla_vision_cpp_debug.py` requests C++ debug outputs through `vla-server` and dumps raw f32.
- Supported alignment points:

```text
vision_patch
vision_vit
vision_merger
```

- Attempting to move the full Python model to CUDA failed due to local 8GB VRAM:

```text
torch.OutOfMemoryError: CUDA out of memory
```

- With the visual-only reference, the Python `vision_patch` dump succeeds.
- `vision_patch` parity:

```text
C++ default vs Python bf16:
mean_abs = 0.001013011788
max_abs  = 0.091701507568

C++ default vs Python f32:
mean_abs = 0.000000018800
max_abs  = 0.000013351440
```

Conclusion: image preprocessing, patch embedding, and 14x14 positional resize are essentially aligned under F32 semantics; the bf16 gap comes from dtype-path differences.

- `vision_merger` 0-block parity:

```text
C++ BF16 resident vs Python f32:
mean_abs = 0.001899844967
max_abs  = 0.071424484253

C++ F32 resident vs Python f32:
mean_abs = 0.000163134813
max_abs  = 0.002740859985
```

- Found and fixed one HyViT2 block structure bug:

```text
C++ missed attn.qkv.bias
Python _ViTAttention uses self.qkv(x), which includes bias
```

Fix:

```text
qkv = mm_w(qkv_w, x_norm) + qkv_b
```

- `vision_vit` 1-block F32 parity before the fix:

```text
mean_abs = 0.126228794456
max_abs  = 2.684199333191
```

- `vision_vit` 1-block F32 parity after the fix:

```text
mean_abs = 0.000083761741
max_abs  = 0.002095699310
```

- `vision_merger` 1-block F32 parity after the fix:

```text
mean_abs = 0.000187682861
max_abs  = 0.002531051636
```

- After the qkv-bias fix, reran the full 27-block HyViT2 + 1 text/action block + 1 flow step smoke:

```text
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=all
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_STEPS=1
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

Result:

```text
chunk = 40
action_dim = 32
n = 1280
sum = 33.47308349609375
mean = 0.026150846853852272
min = -0.04530525952577591
max = 0.2177203893661499
server timing: total=596.6 ms, vision=556.9 ms, inference=26.9 ms
```

## 2026-06-19 Full 27-block vision parity / Full 27-Block Vision Parity

中文：

- 将 visual frontend parity 从 1-block 扩展到完整 27-block HyViT2。
- Python reference 使用 visual-only 模式：

```text
scripts/run_hy_vla_vision_reference.py \
  --mode vision_merger \
  --layers 27 \
  --image-kind zero \
  --height 224 \
  --width 224 \
  --device cuda \
  --dtype f32 \
  --dump-f32 /tmp/hy_py_vision_merger27_zero_224_f32.f32
```

Python 输出：

```text
shape = 49 x 2048
sum = 919.059387207031
mean = 0.009158356115
min = -5.016430854797
max = 44.085346221924
```

- C++ 使用 F32 resident 权重：

```text
VLA_HY_VLA_F32_WEIGHTS=1
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=all
VLA_HY_VLA_DEBUG_OUTPUT=vision_merger
VLA_HY_VLA_DEBUG_LAYERS=27
```

C++ 加载结果：

```text
resident = 3.43 GiB
vision_layers = 27
```

C++ 输出：

```text
shape = 49 x 2048
sum = 918.967712402344
mean = 0.009157442488
min = -5.017387390137
max = 44.075050354004
```

比较结果：

```text
mean_abs = 0.000470250729
max_abs  = 0.018554687500
rmse     = 0.000670485198
```

- 结论：
  - 完整 C++ HyViT2 27-block + merger 在 F32 resident 路径下已经和 Python visual-only reference 对齐到较小误差；
  - 后续默认 BF16 resident 路径的差异主要应视为 dtype/compute policy 差异；
  - 下一步可以把 parity 扩展到真实 image prefix + routed VLM prefix + action expert。

English:

- Extended visual frontend parity from 1 block to the full 27-block HyViT2.
- Python reference used visual-only mode:

```text
scripts/run_hy_vla_vision_reference.py \
  --mode vision_merger \
  --layers 27 \
  --image-kind zero \
  --height 224 \
  --width 224 \
  --device cuda \
  --dtype f32 \
  --dump-f32 /tmp/hy_py_vision_merger27_zero_224_f32.f32
```

Python output:

```text
shape = 49 x 2048
sum = 919.059387207031
mean = 0.009158356115
min = -5.016430854797
max = 44.085346221924
```

- C++ used F32 resident weights:

```text
VLA_HY_VLA_F32_WEIGHTS=1
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=all
VLA_HY_VLA_DEBUG_OUTPUT=vision_merger
VLA_HY_VLA_DEBUG_LAYERS=27
```

C++ load result:

```text
resident = 3.43 GiB
vision_layers = 27
```

C++ output:

```text
shape = 49 x 2048
sum = 918.967712402344
mean = 0.009157442488
min = -5.017387390137
max = 44.075050354004
```

Comparison:

```text
mean_abs = 0.000470250729
max_abs  = 0.018554687500
rmse     = 0.000670485198
```

- Conclusion:
  - the complete C++ HyViT2 27-block + merger path is closely aligned with the Python visual-only reference under the F32 resident path;
  - default BF16 resident differences should mainly be treated as dtype/compute-policy differences;
  - the next step can extend parity to real image prefix + routed VLM prefix + action expert.

## 2026-06-19 Real-image prefix parity / Real-Image Prefix Parity

中文：

- 扩展 `scripts/run_hy_vla_prefix_parity.py`：

```text
--visual-f32 /path/to/visual_tokens.f32
```

- 现在 prefix parity 脚本可以读取真实 post-merger visual tokens，而不是只能使用合成 sin visual fixture。
- 使用 Python 27-block visual-only 输出构建真实 image prefix：

```text
/tmp/hy_py_vision_merger27_zero_224_f32.f32
```

生成 Python prefix：

```text
scripts/run_hy_vla_prefix_parity.py \
  --visual-f32 /tmp/hy_py_vision_merger27_zero_224_f32.f32 \
  --lang-token 120000 120001 120020 7 \
  --dump-f32 /tmp/hy_py_prefix_realvision_zero_224.f32 \
  --dump-mask-i32 /tmp/hy_py_prefix_realvision_zero_224.i32
```

Python prefix：

```text
shape = 64 x 2048
sum = 926.303649902344
mask_sum = 49
```

- C++ 默认 image path 使用：

```text
VLA_HY_VLA_F32_WEIGHTS=1
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=all
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_STEPS=1
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
VLA_HY_VLA_DUMP_PREFIX_F32=/tmp/hy_cpp_prefix_realvision_zero_224.f32
VLA_HY_VLA_DUMP_PREFIX_MASK_I32=/tmp/hy_cpp_prefix_realvision_zero_224.i32
```

请求：

```text
images = 1 x RGB_U8 224x224 all-zero image
lang_tokens = [120000, 120001, 120020, 7]
state = 32 zeros
noise = 40 x 32 zeros
```

C++ action smoke：

```text
action_sum = 33.520606994628906
n = 1280
server timing: total=182.4 ms, vision=149.6 ms, inference=26.2 ms
```

Prefix 对比：

```text
shape = 64 x 2048
mean_abs = 0.000360035716
max_abs  = 0.018554687500
rmse     = 0.000586674490
```

分段观察：

```text
special/text token segments: mean_abs = 0, max_abs = 0
visual token segments: small non-zero diff inherited from 27-block visual frontend
mask_equal = True
mask_sum = 49
```

- 结论：
  - C++ image frontend 输出接入 HY prefix builder 的顺序、special token 插入和 modality mask 均正确；
  - prefix 非零误差来自 visual frontend 数值误差，而不是 prefix builder 拼接错误。

English:

- Extended `scripts/run_hy_vla_prefix_parity.py`:

```text
--visual-f32 /path/to/visual_tokens.f32
```

- The prefix parity script can now consume real post-merger visual tokens instead of only the synthetic sin visual fixture.
- Used the Python 27-block visual-only output:

```text
/tmp/hy_py_vision_merger27_zero_224_f32.f32
```

Generated Python prefix:

```text
scripts/run_hy_vla_prefix_parity.py \
  --visual-f32 /tmp/hy_py_vision_merger27_zero_224_f32.f32 \
  --lang-token 120000 120001 120020 7 \
  --dump-f32 /tmp/hy_py_prefix_realvision_zero_224.f32 \
  --dump-mask-i32 /tmp/hy_py_prefix_realvision_zero_224.i32
```

Python prefix:

```text
shape = 64 x 2048
sum = 926.303649902344
mask_sum = 49
```

- C++ default image path used:

```text
VLA_HY_VLA_F32_WEIGHTS=1
VLA_HY_VLA_TEXT_LAYERS=1
VLA_HY_VLA_VISION_LAYERS=all
VLA_HY_VLA_DEBUG_LAYERS=1
VLA_HY_VLA_DEBUG_STEPS=1
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
VLA_HY_VLA_DUMP_PREFIX_F32=/tmp/hy_cpp_prefix_realvision_zero_224.f32
VLA_HY_VLA_DUMP_PREFIX_MASK_I32=/tmp/hy_cpp_prefix_realvision_zero_224.i32
```

Request:

```text
images = 1 x RGB_U8 224x224 all-zero image
lang_tokens = [120000, 120001, 120020, 7]
state = 32 zeros
noise = 40 x 32 zeros
```

C++ action smoke:

```text
action_sum = 33.520606994628906
n = 1280
server timing: total=182.4 ms, vision=149.6 ms, inference=26.2 ms
```

Prefix comparison:

```text
shape = 64 x 2048
mean_abs = 0.000360035716
max_abs  = 0.018554687500
rmse     = 0.000586674490
```

Segment observation:

```text
special/text token segments: mean_abs = 0, max_abs = 0
visual token segments: small non-zero diff inherited from the 27-block visual frontend
mask_equal = True
mask_sum = 49
```

- Conclusion:
  - the C++ image frontend to HY prefix builder connection has the correct ordering, special-token insertion, and modality mask;
  - the non-zero prefix difference comes from visual-frontend numeric drift, not from prefix-builder assembly.

## Routed-prefix MoT attention parity fix / Routed-prefix MoT 注意力对齐修复

中文：

- 新增本地直连 debug 工具：

```text
src/tools/hy_vla_direct_debug.cpp
build/hy-vla-direct-debug
```

- 原因：
  - 当前 Codex 沙箱禁止 ZMQ `bind()`，所以 `vla-server` 无法在沙箱内完成 TCP/IPC parity 请求；
  - direct debug 工具绕过 server/ZMQ，直接调用 `vla::model_load()` 和 `vla::predict()`，用于中间张量 parity；
  - 这不改变正式 server 路径，只是一个本地验证入口。

- Python reference：

```text
scripts/run_hy_vla_routed_prefix_reference.py
```

- 该脚本只加载 `HunYuanVLMoT` 内部 text decoder 所需层，不再实例化完整 HyVLA policy，避免完整模型在本机 CPU/GPU 上加载过慢或 OOM。

- 发现的问题：
  - C++ routed-prefix 原实现只做了全序列 causal attention；
  - Python 原版 MoT attention 实际是：

```text
1. all-prefix causal attention
2. 对视觉段再做 segment-local bidirectional attention
3. 用第 2 步结果覆盖视觉段 attention output
```

  - Python 的视觉段不是简单的连续 `mask == 1` token。`_modality_mask_to_segments()` 会把中间单个 `vision_split` 文本 token 包进视觉段里，只有长度大于等于 2 的 text run 才作为视觉段分隔。

- C++ 修复：
  - 新增 `make_visual_override_segments()`，按 Python `_modality_mask_to_segments()` 规则生成视觉 override 段；
  - `build_hy_prefix_routed_layer()` 保留全序列 causal attention；
  - 对 visual override segment 再跑一次无 mask 的 segment-local attention；
  - 将 visual segment attention output 拼回原序列，再继续按 modality run 分别走 text/vision output projection 和 FFN。

- 验证命令：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_routed_prefix_reference.py \
  --layers 1 \
  --dtype f32 \
  --device cpu \
  --prefix-f32 /tmp/hy_py_prefix_realvision_zero_224.f32 \
  --mask-i32 /tmp/hy_py_prefix_realvision_zero_224.i32 \
  --dump-f32 /tmp/hy_py_routed_prefix_realvision_zero_224_1blk.f32

VLA_HY_VLA_F32_WEIGHTS=1 \
VLA_HY_VLA_TEXT_LAYERS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=routed_prefix \
VLA_HY_VLA_DEBUG_LAYERS=1 \
./build/hy-vla-direct-debug \
  --model /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf \
  --prefix-f32 /tmp/hy_py_prefix_realvision_zero_224.f32 \
  --mask-i32 /tmp/hy_py_prefix_realvision_zero_224.i32 \
  --out-f32 /tmp/hy_cpp_routed_prefix_realvision_zero_224_1blk.f32
```

- 修复前：

```text
Python routed-prefix 1 block sum = 2426.781982421875
C++ routed-prefix 1 block sum    = 1426.713867187500
mean_abs = 0.251100361347
max_abs  = 13.542633056641
rmse     = 0.474474340677
```

- 修复后：

```text
Python routed-prefix 1 block sum = 2426.781982421875
C++ routed-prefix 1 block sum    = 2426.781982421875
mean_abs = 0.000000189075
max_abs  = 0.000038146973
rmse     = 0.000000334193
```

- 结论：
  - routed-prefix 1 block / real visual prefix parity 已经基本完全对齐；
  - 之前的不对齐主因是 MoT visual segment bidirectional attention override 缺失，不是 GGUF 权重或 prefix builder 错误。

English:

- Added a local direct debug tool:

```text
src/tools/hy_vla_direct_debug.cpp
build/hy-vla-direct-debug
```

- Motivation:
  - the current Codex sandbox rejects ZMQ `bind()`, so `vla-server` cannot complete TCP/IPC parity requests inside the sandbox;
  - the direct debug tool bypasses server/ZMQ and calls `vla::model_load()` plus `vla::predict()` directly for intermediate tensor parity;
  - this does not change the production server path.

- Python reference:

```text
scripts/run_hy_vla_routed_prefix_reference.py
```

- This script loads only the required `HunYuanVLMoT` inner text-decoder layers instead of instantiating the full HyVLA policy, avoiding slow full-model CPU load or local OOM.

- Root cause:
  - the previous C++ routed-prefix implementation only performed full-prefix causal attention;
  - the Python MoT attention actually does:

```text
1. full-prefix causal attention
2. segment-local bidirectional attention for visual segments
3. overwrite visual-segment attention outputs with step 2
```

  - Python visual segments are not simply contiguous `mask == 1` runs. `_modality_mask_to_segments()` includes single `vision_split` text tokens inside the visual segment and only treats text runs of length >= 2 as separators.

- C++ fix:
  - added `make_visual_override_segments()` mirroring Python `_modality_mask_to_segments()`;
  - kept full-prefix causal attention in `build_hy_prefix_routed_layer()`;
  - reran unmasked segment-local attention for visual override segments;
  - stitched the overridden attention output back into sequence order before text/vision output projection and FFN routing.

- Result after the fix:

```text
mean_abs = 0.000000189075
max_abs  = 0.000038146973
rmse     = 0.000000334193
```

- Conclusion:
  - routed-prefix 1 block with a real visual prefix is now essentially aligned;
  - the mismatch was caused by a missing MoT visual-segment bidirectional attention override, not by GGUF weights or prefix assembly.

## Routed joint 1-block parity with real visual prefix / 真实视觉 Prefix 的 Routed Joint 单层对齐

中文：

- 新增轻量 Python reference：

```text
scripts/run_hy_vla_routed_joint_reference.py
```

- 目的：
  - 避免加载完整 HyVLA policy；
  - 只加载 VLM text decoder、action expert decoder、state/action/time/action_out projection；
  - 复用原版 `HyDualTower.forward()` 的 shared-attention/action-expert 逻辑；
  - 用同一份 real visual prefix 和 modality mask 验证 `routed_joint_vt` / `routed_joint`。

- Python `routed_joint_vt`：

```text
/home/xuling/anaconda3/envs/hy-vla/bin/python scripts/run_hy_vla_routed_joint_reference.py \
  --layers 1 \
  --dtype f32 \
  --device cpu \
  --mode routed_joint_vt \
  --prefix-f32 /tmp/hy_py_prefix_realvision_zero_224.f32 \
  --mask-i32 /tmp/hy_py_prefix_realvision_zero_224.i32 \
  --dump-f32 /tmp/hy_py_routed_joint_vt_realvision_zero_224_1blk.f32
```

```text
Python routed_joint_vt_1blk sum = -335.157287597656
```

- C++ `routed_joint_vt`：

```text
VLA_HY_VLA_F32_WEIGHTS=1 \
VLA_HY_VLA_TEXT_LAYERS=1 \
VLA_HY_VLA_DEBUG_OUTPUT=routed_joint_vt \
VLA_HY_VLA_DEBUG_LAYERS=1 \
./build/hy-vla-direct-debug \
  --model /home/xuling/robotic_dataset/models/hy_vla_full_bf16_vlmvisionstable.gguf \
  --prefix-f32 /tmp/hy_py_prefix_realvision_zero_224.f32 \
  --mask-i32 /tmp/hy_py_prefix_realvision_zero_224.i32 \
  --out-f32 /tmp/hy_cpp_routed_joint_vt_realvision_zero_224_1blk.f32
```

```text
C++ routed_joint_vt_1blk sum = -335.214965820312
mean_abs = 0.000145934609
max_abs  = 0.003317594528
rmse     = 0.000329048053
```

- `routed_joint` 一步 Euler 输出：

```text
Python routed_joint_1blk sum = 33.515727996826
C++ routed_joint_1blk sum    = 33.521499633789
mean_abs = 0.000014593488
max_abs  = 0.000331759453
rmse     = 0.000032904933
```

- 结论：
  - real visual prefix -> routed prefix KV -> suffix/action expert -> action_out_proj 的 1 block 链路已经稳定对齐；
  - `v_t` 的剩余误差约 `1e-4` mean_abs，主要来自 Python dual-tower eager 路径里的 BF16 cast / softmax 数值细节；
  - 一步 Euler 后 `x_1` 误差缩小 10 倍，符合 `dt=-0.1`。

English:

- Added lightweight Python reference:

```text
scripts/run_hy_vla_routed_joint_reference.py
```

- Purpose:
  - avoid loading the full HyVLA policy;
  - load only the VLM text decoder, action expert decoder, state/action/time/action_out projections;
  - reuse the upstream `HyDualTower.forward()` shared-attention/action-expert logic;
  - validate `routed_joint_vt` / `routed_joint` with the same real visual prefix and modality mask.

- `routed_joint_vt` result:

```text
Python sum = -335.157287597656
C++ sum    = -335.214965820312
mean_abs  = 0.000145934609
max_abs   = 0.003317594528
rmse      = 0.000329048053
```

- One-step Euler `routed_joint` result:

```text
Python sum = 33.515727996826
C++ sum    = 33.521499633789
mean_abs  = 0.000014593488
max_abs   = 0.000331759453
rmse      = 0.000032904933
```

- Conclusion:
  - the real-visual-prefix -> routed prefix KV -> suffix/action expert -> action_out_proj 1-block path is now stable;
  - the remaining `v_t` drift is around `1e-4` mean_abs, likely from BF16 cast / softmax numeric details in the Python dual-tower eager path;
  - the one-step Euler output shrinks the error by 10x, matching `dt=-0.1`.

## Routed action prefix mask correction / Routed Action Prefix Mask 修正

中文：

- 进一步定位 2-block `routed_joint_vt` 时发现：
  - `routed_prefix` debug 对齐的是 vendor `HunYuanVLMoTTextModel` 的 MoT visual-segment override；
  - 真实 `HyVLAFlowMatching.sample_actions()` 走的是 `HyDualTower.forward()`，prefix prefill 前由 `_apply_visual_segment_mask()` 改写 2D attention mask；
  - 当前模型配置 `visual_segment_isolation=False`，因此真实 action path 是 patch-only visual mask，不是 full visual segment override。

- 修正：
  - C++ `routed_prefix` debug 继续保留 vendor MoT full-segment override，用于验证 VLM text decoder parity；
  - C++ `routed_joint_vt`、`routed_joint` 和默认 routed action path 改用 patch-only prefix mask；
  - action path 的 routed prefix 不再额外执行 visual segment attention override re-run；
  - `scripts/run_hy_vla_routed_joint_reference.py` 也补上 patch-only visual mask，以匹配 `sample_actions()`。

- 2-block 修正前：

```text
routed_joint_vt_2blk mean_abs = 0.004370612092
routed_joint_vt_2blk max_abs  = 0.053586483002
```

- 2-block 修正后：

```text
Python routed_joint_vt_2blk sum = -347.895599365234
C++ routed_joint_vt_2blk sum    = -347.880493164062
mean_abs = 0.000244787399
max_abs  = 0.004205822945
rmse     = 0.000535910716
```

- 1-block 复测：

```text
Python routed_joint_vt_1blk sum = -335.157287597656
C++ routed_joint_vt_1blk sum    = -335.210510253906
mean_abs = 0.000136175717
max_abs  = 0.002888202667
rmse     = 0.000324120454
```

- 结论：
  - HY-VLA action path 的 prefix attention 语义现在与原版 `sample_actions()` 对齐；
  - routed-prefix debug 和 routed-action path 现在刻意区分两种参考语义：
    - `routed_prefix`: vendor VLM text decoder / MoT full-segment override；
    - `routed_joint*`: HyDualTower action inference / patch-only visual mask。

English:

- While checking 2-block `routed_joint_vt`, we found that:
  - `routed_prefix` debug matches the vendor `HunYuanVLMoTTextModel` MoT visual-segment override;
  - real `HyVLAFlowMatching.sample_actions()` uses `HyDualTower.forward()` with `_apply_visual_segment_mask()` before prefix prefill;
  - the current checkpoint has `visual_segment_isolation=False`, so the real action path uses patch-only visual masking, not full visual-segment override.

- Fix:
  - kept C++ `routed_prefix` debug on the vendor MoT full-segment override for VLM text-decoder parity;
  - changed C++ `routed_joint_vt`, `routed_joint`, and the default routed action path to use patch-only prefix masks;
  - disabled the extra visual-segment attention override re-run in the routed action path;
  - updated `scripts/run_hy_vla_routed_joint_reference.py` to apply the same patch-only visual mask as `sample_actions()`.

- 2-block result after the fix:

```text
Python sum = -347.895599365234
C++ sum    = -347.880493164062
mean_abs  = 0.000244787399
max_abs   = 0.004205822945
rmse      = 0.000535910716
```

- 1-block recheck:

```text
Python sum = -335.157287597656
C++ sum    = -335.210510253906
mean_abs  = 0.000136175717
max_abs   = 0.002888202667
rmse      = 0.000324120454
```

- Conclusion:
  - HY-VLA action-path prefix attention now matches upstream `sample_actions()`;
  - the debug routes intentionally keep two reference semantics separate:
    - `routed_prefix`: vendor VLM text decoder / MoT full-segment override;
    - `routed_joint*`: HyDualTower action inference / patch-only visual mask.

## Routed action 5/10-block parity / Routed Action 5/10 层对齐

中文：

- 在 patch-only prefix mask 修正后，继续提升 `routed_joint_vt` block 数：

```text
5 blocks:
Python sum = -316.557312011719
C++ sum    = -316.506286621094
mean_abs  = 0.000173904598
max_abs   = 0.002747535706
rmse      = 0.000375561824

10 blocks:
Python sum = -349.126770019531
C++ sum    = -348.889892578125
mean_abs  = 0.000348882459
max_abs   = 0.009839773178
rmse      = 0.000941350707
```

- 结论：
  - 5/10 层 action path 没有出现新的结构性偏差；
  - 误差随层数增加缓慢增长，符合 BF16/F32、softmax 和 matmul 实现差异的预期；
  - 下一步应验证完整 10-step Euler denoise loop，而不只是单步 `v_t`。

English:

- After the patch-only prefix-mask correction, we increased `routed_joint_vt` depth:

```text
5 blocks:
Python sum = -316.557312011719
C++ sum    = -316.506286621094
mean_abs  = 0.000173904598
max_abs   = 0.002747535706
rmse      = 0.000375561824

10 blocks:
Python sum = -349.126770019531
C++ sum    = -348.889892578125
mean_abs  = 0.000348882459
max_abs   = 0.009839773178
rmse      = 0.000941350707
```

- Conclusion:
  - the 5/10-layer action path shows no new structural mismatch;
  - drift grows slowly with depth, consistent with BF16/F32, softmax, and matmul implementation differences;
  - the next check should be the full 10-step Euler denoise loop, not only one-step `v_t`.

## Full 10-step routed sample parity / 完整 10 步 Routed Sample 对齐

中文：

- 扩展 `scripts/run_hy_vla_routed_joint_reference.py`：

```text
--mode routed_sample
```

- 该模式复用同一份 routed prefix KV，按原版 `sample_actions()` / `denoise_step()` 逻辑执行完整 10-step Euler loop。
- C++ 对照使用默认 routed action path，即不设置 `VLA_HY_VLA_DEBUG_OUTPUT`，并设置：

```text
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

- 2-block 完整 10-step loop：

```text
Python routed_sample_2blk sum = 72.427719116211
C++ routed_sample_2blk sum    = 72.427291870117
mean_abs = 0.000033512304
max_abs  = 0.001002252102
rmse     = 0.000094260235
```

- 5-block 完整 10-step loop：

```text
Python routed_sample_5blk sum = 67.509391784668
C++ routed_sample_5blk sum    = 67.520950317383
mean_abs = 0.000049195376
max_abs  = 0.000911414623
rmse     = 0.000115979121
```

- 10-block 完整 10-step loop：

```text
Python routed_sample_10blk sum = 61.557670593262
C++ routed_sample_10blk sum    = 61.553356170654
mean_abs = 0.000073886171
max_abs  = 0.001326173544
rmse     = 0.000154524954
```

- 结论：
  - 默认 routed action path 的 denoise loop、time schedule、Euler update 和 prefix KV 复用均已与轻量 Python reference 对齐；
  - 2/5 block 完整 loop 误差非常小，比单步 `v_t` 更小，说明误差在 Euler 积分后没有发散；
  - 下一步可以继续提升到 10 blocks / 更多 blocks，或开始把 direct parity 迁回标准 `vla-server` 请求路径。

English:

- Extended `scripts/run_hy_vla_routed_joint_reference.py`:

```text
--mode routed_sample
```

- This mode reuses the routed prefix KV and runs the full 10-step Euler loop following upstream `sample_actions()` / `denoise_step()`.
- The C++ comparison uses the default routed action path, without `VLA_HY_VLA_DEBUG_OUTPUT`, plus:

```text
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

- 2-block full 10-step loop:

```text
Python sum = 72.427719116211
C++ sum    = 72.427291870117
mean_abs  = 0.000033512304
max_abs   = 0.001002252102
rmse      = 0.000094260235
```

- 5-block full 10-step loop:

```text
Python sum = 67.509391784668
C++ sum    = 67.520950317383
mean_abs  = 0.000049195376
max_abs   = 0.000911414623
rmse      = 0.000115979121
```

- 10-block full 10-step loop:

```text
Python sum = 61.557670593262
C++ sum    = 61.553356170654
mean_abs  = 0.000073886171
max_abs   = 0.001326173544
rmse      = 0.000154524954
```

- Conclusion:
  - the default routed action path now matches the lightweight Python reference for denoise loop, time schedule, Euler update, and prefix KV reuse;
  - 2/5-block full-loop drift is very small and does not grow through Euler integration;
  - next steps are raising to 10+ blocks or moving the direct parity path back through standard `vla-server` requests.

## Direct image path parity without ZMQ / 无 ZMQ 的 Direct Image Path 对齐

中文：

- 扩展 `hy-vla-direct-debug`：

```text
--image-kind zero|gradient
--height H
--width W
--lang-tokens 120000,120001,120020,7
```

- 现在 direct debug 工具可以绕过 ZMQ，直接测试：

```text
RGB image
  -> C++ vision frontend
  -> HY prefix builder
  -> routed action path
  -> full 10-step Euler output
```

- 5-block zero-image full sample：

```text
VLA_HY_VLA_F32_WEIGHTS=1
VLA_HY_VLA_TEXT_LAYERS=5
VLA_HY_VLA_VISION_LAYERS=all
VLA_HY_VLA_DEBUG_LAYERS=5
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1
```

C++ image path output:

```text
sum = 67.520957946777
```

与 C++ precomputed-prefix routed sample 对比：

```text
mean_abs = 0.000000347784
max_abs  = 0.000021308661
rmse     = 0.000000986484
```

image path dump 的 prefix 与 Python real-visual prefix 对比：

```text
shape = 64 x 2048
mean_abs = 0.000000364951
max_abs  = 0.000022888184
rmse     = 0.000000570390
mask_equal = True
mask_sum = 49
```

- 结论：
  - direct image frontend -> prefix builder -> routed action 的 C++ 端到端路径已闭合；
  - image path 与 precomputed prefix path 的 action 输出基本一致；
  - 这绕过了当前沙箱里的 ZMQ bind 限制，但验证的是同一个 `vla::predict()` 模型路径。

English:

- Extended `hy-vla-direct-debug`:

```text
--image-kind zero|gradient
--height H
--width W
--lang-tokens 120000,120001,120020,7
```

- The direct debug tool can now test, without ZMQ:

```text
RGB image
  -> C++ vision frontend
  -> HY prefix builder
  -> routed action path
  -> full 10-step Euler output
```

- 5-block zero-image full sample:

```text
C++ image path sum = 67.520957946777
```

Compared with the C++ precomputed-prefix routed sample:

```text
mean_abs = 0.000000347784
max_abs  = 0.000021308661
rmse     = 0.000000986484
```

Image-path dumped prefix vs Python real-visual prefix:

```text
mean_abs = 0.000000364951
max_abs  = 0.000022888184
rmse     = 0.000000570390
mask_equal = True
mask_sum = 49
```

- Conclusion:
  - the direct C++ image frontend -> prefix builder -> routed action path is closed end to end;
  - image-path and precomputed-prefix action outputs are effectively identical;
  - this bypasses the current sandbox ZMQ bind limitation while testing the same `vla::predict()` model path.

## 20/32-block full-sample scaling / 20/32 层完整 Sample 扩展验证

中文：

- 继续把 precomputed real-visual prefix 的 full 10-step sample 提升到 20 layers：

```text
20 blocks, F32 resident:
Python sum = 38.086845397949
C++ sum    = 38.109008789062
mean_abs  = 0.000096737072
max_abs   = 0.001212924719
rmse      = 0.000178962902
```

- 32 layers F32 resident 结果：
  - C++ direct run 被系统杀掉，退出码 `137`；
  - 这是内存压力导致，符合 32 层 F32 resident 权重过大的预期；
  - Python F32 reference 也被中断，避免继续占资源。

- 32 layers BF16 resident C++ full sample：

```text
resident weights = 6.44 GiB
C++ sum = 42.206855773926
```

- 32 layers BF16 Python reference：

```text
Python sum = 41.334106445312
C++ sum    = 42.206855773926
mean_abs  = 0.002378383186
max_abs   = 0.023860216141
rmse      = 0.004617416300
```

- 32 layers BF16 resident direct image path：

```text
VLA_HY_VLA_TEXT_LAYERS=32
VLA_HY_VLA_VISION_LAYERS=all
resident weights = 7.32 GiB
C++ image-path sum = 42.008277893066
```

- 32-layer BF16 image path vs C++ precomputed-prefix path：

```text
action output:
mean_abs = 0.000580863969
max_abs  = 0.011278100312
rmse     = 0.001157656661

dumped prefix vs Python real-visual prefix:
mean_abs = 0.002205668483
max_abs  = 0.058364868164
rmse     = 0.003484458663
mask_equal = True
mask_sum = 49
```

- 结论：
  - 20-layer F32 full sample parity 仍然稳定；
  - 32-layer F32 resident 在本机/沙箱内不适合，内存压力会触发 kill；
  - 32-layer BF16 resident 能跑通完整 precomputed-prefix path 和 direct-image path；
  - full-depth BF16 与 Python BF16 reference 的误差升高到 `mean_abs ~= 0.00238`，属于完整 32 层 BF16/CPU eager/ggml 数值累计，需要后续在 GPU 或更细 block dump 下继续评估。

English:

- Raised the full 10-step sample with precomputed real-visual prefix to 20 layers:

```text
20 blocks, F32 resident:
Python sum = 38.086845397949
C++ sum    = 38.109008789062
mean_abs  = 0.000096737072
max_abs   = 0.001212924719
rmse      = 0.000178962902
```

- 32-layer F32 resident result:
  - the C++ direct run was killed with exit code `137`;
  - this is memory pressure, consistent with full-depth F32 resident weights;
  - the Python F32 reference was interrupted to avoid wasting resources.

- 32-layer BF16 resident C++ full sample:

```text
resident weights = 6.44 GiB
C++ sum = 42.206855773926
```

- 32-layer BF16 Python reference:

```text
Python sum = 41.334106445312
C++ sum    = 42.206855773926
mean_abs  = 0.002378383186
max_abs   = 0.023860216141
rmse      = 0.004617416300
```

- 32-layer BF16 resident direct image path:

```text
VLA_HY_VLA_TEXT_LAYERS=32
VLA_HY_VLA_VISION_LAYERS=all
resident weights = 7.32 GiB
C++ image-path sum = 42.008277893066
```

- 32-layer BF16 image path vs C++ precomputed-prefix path:

```text
action output:
mean_abs = 0.000580863969
max_abs  = 0.011278100312
rmse     = 0.001157656661

dumped prefix vs Python real-visual prefix:
mean_abs = 0.002205668483
max_abs  = 0.058364868164
rmse     = 0.003484458663
mask_equal = True
mask_sum = 49
```

- Conclusion:
  - 20-layer F32 full-sample parity remains stable;
  - 32-layer F32 resident is not suitable on this local/sandbox setup because of memory pressure;
  - 32-layer BF16 resident runs both the precomputed-prefix and direct-image paths;
  - full-depth BF16 drift rises to `mean_abs ~= 0.00238`, which should be evaluated further on GPU or with finer block-level dumps.

## Gradient-image direct path parity / Gradient 图像 Direct Path 对齐

中文：

- 在 zero image 之外，新增 gradient image 检查：

```text
image-kind = gradient
height = 224
width = 224
layers = 5
dtype/resident = F32
```

- Python visual reference：

```text
vision_merger27_gradient_224_f32 sum = 1291.546386718750
```

- C++ direct image dumped prefix vs Python real-visual prefix：

```text
shape = 64 x 2048
mean_abs = 0.000001088001
max_abs  = 0.000030517578
rmse     = 0.000001746029
mask_equal = True
mask_sum = 49
```

- 5-block full sample：

```text
Python routed_sample_5blk sum = 70.075614929199
C++ direct image sum          = 70.083251953125
mean_abs = 0.000049054262
max_abs  = 0.001010119915
rmse     = 0.000123153193
```

- 结论：
  - 非零图像输入下，C++ resize/normalize/vision frontend/prefix builder/action path 仍然稳定；
  - 这比 zero image 更能说明 image path 没有依赖全零输入的偶然对齐。

English:

- Added a gradient-image check in addition to the zero-image fixture:

```text
image-kind = gradient
height = 224
width = 224
layers = 5
dtype/resident = F32
```

- Python visual reference:

```text
vision_merger27_gradient_224_f32 sum = 1291.546386718750
```

- C++ direct-image dumped prefix vs Python real-visual prefix:

```text
shape = 64 x 2048
mean_abs = 0.000001088001
max_abs  = 0.000030517578
rmse     = 0.000001746029
mask_equal = True
mask_sum = 49
```

- 5-block full sample:

```text
Python routed_sample_5blk sum = 70.075614929199
C++ direct image sum          = 70.083251953125
mean_abs = 0.000049054262
max_abs  = 0.001010119915
rmse     = 0.000123153193
```

- Conclusion:
  - with a non-zero image input, C++ resize/normalization/vision frontend/prefix builder/action path remains stable;
  - this is a stronger signal than the all-zero image fixture and rules out accidental zero-input alignment.

## Non-square image resize parity fix / 非方图像 Resize 对齐修复

中文：

- 180x260 gradient image 曾暴露 C++/Python 图像预处理差异：

```text
image-kind = gradient
height = 180
width = 260
layers = 5
dtype/resident = F32
```

- 根因：
  - Python resize-with-pad 使用 double precision 计算比例，并用 `int(cur_width / ratio)` 截断；
  - 对 180x260 输入，Python 得到 resized width = 223；
  - C++ 原先使用 float 计算，得到 resized width = 224；
  - 这个 1-pixel 差异会进入 padding、patch embedding、vision prefix，最后放大到 action 输出。

- 修复：
  - `preprocess_hy_image_chw_224()` 中将 resize ratio 和采样坐标计算改为 double，再按 Python 语义截断；
  - 保持 bilinear 采样、normalize、pad value 与 Python reference 一致。

- 修复后 `vision_pixels`：

```text
shape = 3 x 224 x 224
mean_abs = 0.000000017455
max_abs  = 0.000000476837
rmse     = 0.000000050554
```

- 修复后 `vision_patch`：

```text
shape = 1 x 576 x 392
mean_abs = 0.000000199807
max_abs  = 0.000014305115
rmse     = 0.000000420146
```

- 修复后 C++ direct-image dumped prefix vs Python real-visual prefix：

```text
shape = 64 x 2048
mean_abs = 0.000001115579
max_abs  = 0.000026702881
rmse     = 0.000001769921
mask_equal = True
```

- 修复后 5-block full sample：

```text
Python routed_sample_5blk sum = 64.707588195801
C++ direct image sum          = 64.703399658203
mean_abs = 0.000044254986
max_abs  = 0.000964015722
rmse     = 0.000108005894
```

- 结论：
  - 非方图像路径已恢复到和 224x224 gradient fixture 同量级的误差；
  - 这个修复对真实机器人相机输入很重要，因为真实输入通常不是 1:1 方图。

English:

- A 180x260 gradient image exposed a C++/Python preprocessing mismatch:

```text
image-kind = gradient
height = 180
width = 260
layers = 5
dtype/resident = F32
```

- Root cause:
  - Python resize-with-pad computes the resize ratio in double precision and truncates with `int(cur_width / ratio)`;
  - for a 180x260 input, Python gets resized width = 223;
  - the previous C++ path used float arithmetic and got resized width = 224;
  - this one-pixel difference propagates through padding, patch embedding, the vision prefix, and finally the action output.

- Fix:
  - changed `preprocess_hy_image_chw_224()` to compute the resize ratio and sampling coordinates in double precision, then truncate with Python-equivalent semantics;
  - kept bilinear sampling, normalization, and pad values aligned with the Python reference.

- Fixed `vision_pixels`:

```text
shape = 3 x 224 x 224
mean_abs = 0.000000017455
max_abs  = 0.000000476837
rmse     = 0.000000050554
```

- Fixed `vision_patch`:

```text
shape = 1 x 576 x 392
mean_abs = 0.000000199807
max_abs  = 0.000014305115
rmse     = 0.000000420146
```

- Fixed C++ direct-image dumped prefix vs Python real-visual prefix:

```text
shape = 64 x 2048
mean_abs = 0.000001115579
max_abs  = 0.000026702881
rmse     = 0.000001769921
mask_equal = True
```

- Fixed 5-block full sample:

```text
Python routed_sample_5blk sum = 64.707588195801
C++ direct image sum          = 64.703399658203
mean_abs = 0.000044254986
max_abs  = 0.000964015722
rmse     = 0.000108005894
```

- Conclusion:
  - the non-square image path is now back to the same error scale as the 224x224 gradient fixture;
  - this matters for real robot camera inputs, which are often not square.

## vla-server image-path smoke / vla-server 图像路径 Smoke

中文：

- 在 direct debug 之外，验证了 HY-VLA 可以通过通用 `vla-server` 的 Protobuf/ZMQ 路径接收图像并返回 action：

```text
server = ./build/vla-server
bind = tcp://127.0.0.1:6128
backend = CPU (CUDA_VISIBLE_DEVICES=-1)
image-kind = gradient
height = 180
width = 260
layers = 5
state = zero raw state
noise = zero action noise
```

- 请求路径：

```text
Python smoke client
  -> vla.proto PredictRequest
  -> RGB_U8 image bytes
  -> ZMQ tcp://127.0.0.1:6128
  -> src/serving/server.cpp decode_image()
  -> Inputs::images + lang_tokens + state + noise
  -> src/models/hy_vla.cpp predict()
  -> PredictResponse action_chunk
```

- `vision_pixels` server smoke：

```text
server vs C++ direct-debug:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

server vs Python reference:
mean_abs = 0.000000017455
max_abs  = 0.000000476837
rmse     = 0.000000050554
```

- 5-block action server smoke：

```text
server action sum = 64.703399658203

server vs C++ direct-debug:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

server vs Python reference:
mean_abs = 0.000044254986
max_abs  = 0.000964015722
rmse     = 0.000108005894
```

- CPU server timing：

```text
total = 1632.3 ms
vision = 887.9 ms
inference = 712.0 ms
other = 32.4 ms
```

- CUDA server smoke：

```text
backend = CUDA
server action sum = 64.705039978027

CUDA server vs CUDA direct-debug:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

CUDA server vs Python reference:
mean_abs = 0.000050669372
max_abs  = 0.000971972942
rmse     = 0.000109866553
```

- CUDA server timing：

```text
total = 237.2 ms
vision = 139.5 ms
inference = 88.3 ms
other = 9.4 ms
```

- 10-block CUDA server scaling smoke：

```text
backend = CUDA
resident weights = 6.66 GiB
image-kind = gradient
height = 180
width = 260
server action sum = 54.442058563232

10-block CUDA server vs 10-block CUDA direct-debug:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

total = 309.0 ms
vision = 142.9 ms
inference = 154.1 ms
other = 12.0 ms
```

- 20-block BF16 resident CUDA server scaling smoke：

```text
backend = CUDA
resident weights = 5.17 GiB
image-kind = gradient
height = 180
width = 260
server action sum = 33.079795837402

20-block BF16 CUDA server vs 20-block BF16 CUDA direct-debug:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

total = 473.3 ms
vision = 249.4 ms
inference = 205.3 ms
other = 18.7 ms
```

- 32-block BF16 resident CUDA boundary：

```text
backend = CUDA
text_layers = 32
vision_layers = all
requested CUDA buffer = 7497.99 MiB
result = cudaMalloc failed: out of memory
```

- 新增 OOM fallback 开关：

```text
VLA_HY_VLA_CUDA_OOM_FALLBACK_CPU=1
```

- 行为：
  - 默认不改变：CUDA resident allocation 失败时仍然 `model_load` 失败；
  - 设置该开关后，如果 CUDA resident allocation 失败，会释放 CUDA backend，改用 CPU backend 重新分配同一个 weight context；
  - 这不是性能路径，而是为了在 8GB 本机上继续做 full-depth smoke/parity。

- 32-block + full vision fallback 验证：

```text
text_layers = 32
vision_layers = all
initial backend = CUDA
CUDA allocation = OOM at 7497.99 MiB
fallback backend = CPU (8 threads)
resident weights = 7.32 GiB
debug_output = vision_pixels

fallback vs C++ direct vision_pixels:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

fallback vs Python vision_pixels:
mean_abs = 0.000000017455
max_abs  = 0.000000476837
rmse     = 0.000000050554
```

- 结论：
  - 本机 3070Ti 8GB 可以运行 20-block BF16 resident serving smoke；
  - 32-block BF16 resident 在当前“一次性 resident 全量权重”策略下无法稳定装入；
  - fallback 可以让 full-depth CPU smoke/parity 继续进行；
  - 下一步若要在 8GB 本机跑完整 32-block CUDA，需要实现 block streaming / partial resident cache / quantized resident weights 之一。

## 32-block CUDA + CPU vision sidecar / 32 层 CUDA + CPU 视觉 Sidecar

中文：

- 新增低显存 direct-image 路径：

```text
VLA_HY_VLA_VISION_CPU_SIDELOAD=1
```

- 设计：
  - 主 HY-VLA backend 仍使用 CUDA；
  - 32 层 VLM text/prefix/action expert 权重保留在 CUDA resident buffer；
  - HyViT2/merger 视觉前端权重单独加载到 CPU sidecar backend；
  - direct image 请求先用 CPU sidecar 生成 49 个 visual tokens，再进入 CUDA routed prefix/action path。

- 5-block F32 sidecar sanity：

```text
sidecar vision weights = 1.68 GiB
main CUDA resident = 3.18 GiB

sidecar vs full-CUDA direct image:
mean_abs = 0.000011646365
max_abs  = 0.000104188919
rmse     = 0.000017025246

sidecar vs Python reference:
mean_abs = 0.000049871625
max_abs  = 0.000946372747
rmse     = 0.000108406930
```

- 32-block BF16 direct-image sidecar：

```text
image-kind = gradient
height = 180
width = 260
sidecar vision weights = 0.88 GiB
main CUDA resident = 6.44 GiB
action sum = 51.184600830078
```

- 32-block BF16 vla-server sidecar：

```text
server action sum = 51.184600830078

server vs direct-debug:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

total = 1862.8 ms
vision = 881.3 ms
inference = 838.0 ms
other = 143.5 ms
```

- 32-block BF16 zero-224 parity：

```text
sidecar zero-224 action sum = 42.146926879883

sidecar vs Python BF16 32-block reference:
mean_abs = 0.002407078398
max_abs  = 0.022368669510
rmse     = 0.004628079478

sidecar vs previous C++ full-resident direct-image:
mean_abs = 0.000626550813
max_abs  = 0.011215291917
rmse     = 0.001159808249

sidecar vs previous C++ precomputed-prefix path:
mean_abs = 0.000477877271
max_abs  = 0.005932569504
rmse     = 0.000736353220
```

- 结论：
  - 本机 3070Ti 8GB 已经可以跑 HY-VLA 32-block direct-image CUDA serving path；
  - 当前低显存策略不是完整 per-block streaming，而是先拆出视觉前端 sidecar；
  - 下一步可以继续做 text/action block streaming 或 quantized resident weights，把 CPU vision sidecar 也迁回 CUDA/量化路径。

English:

- Added a low-memory direct-image path:

```text
VLA_HY_VLA_VISION_CPU_SIDELOAD=1
```

- Design:
  - the main HY-VLA backend remains CUDA;
  - the 32-layer VLM text/prefix/action expert weights stay in the CUDA resident buffer;
  - the HyViT2/merger visual frontend is loaded into a separate CPU sidecar backend;
  - direct-image requests first produce 49 visual tokens with the CPU sidecar, then enter the CUDA routed prefix/action path.

- 5-block F32 sidecar sanity:

```text
sidecar vision weights = 1.68 GiB
main CUDA resident = 3.18 GiB

sidecar vs full-CUDA direct image:
mean_abs = 0.000011646365
max_abs  = 0.000104188919
rmse     = 0.000017025246

sidecar vs Python reference:
mean_abs = 0.000049871625
max_abs  = 0.000946372747
rmse     = 0.000108406930
```

- 32-block BF16 direct-image sidecar:

```text
image-kind = gradient
height = 180
width = 260
sidecar vision weights = 0.88 GiB
main CUDA resident = 6.44 GiB
action sum = 51.184600830078
```

- 32-block BF16 vla-server sidecar:

```text
server action sum = 51.184600830078

server vs direct-debug:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

total = 1862.8 ms
vision = 881.3 ms
inference = 838.0 ms
other = 143.5 ms
```

- 32-block BF16 zero-224 parity:

```text
sidecar zero-224 action sum = 42.146926879883

sidecar vs Python BF16 32-block reference:
mean_abs = 0.002407078398
max_abs  = 0.022368669510
rmse     = 0.004628079478

sidecar vs previous C++ full-resident direct-image:
mean_abs = 0.000626550813
max_abs  = 0.011215291917
rmse     = 0.001159808249

sidecar vs previous C++ precomputed-prefix path:
mean_abs = 0.000477877271
max_abs  = 0.005932569504
rmse     = 0.000736353220
```

- Conclusion:
  - the local 3070Ti 8GB can now run the HY-VLA 32-block direct-image CUDA serving path;
  - this low-memory strategy is not full per-block streaming yet; it first splits out the visual frontend sidecar;
  - the next step is text/action block streaming or quantized resident weights, so the CPU vision sidecar can eventually move back to CUDA or a quantized path.

- 注意：
  - 在当前 Codex 沙箱里，ZMQ `bind(tcp://127.0.0.1:...)` 需要提升权限；
  - server 和 client 必须在同一权限/网络视角运行，否则 client 可能连接后收不到响应；
  - 早先一次 CUDA server 5-block action smoke 在 300s client timeout 内未返回，但 server 关闭时显示 `served=0`，根因是 client 在沙箱网络视角下请求没有到达 server；
  - 使用同一提升权限/网络视角运行 server 和 client 后，CUDA server action smoke 已通过。

English:

- In addition to direct-debug parity, HY-VLA was verified through the generic `vla-server` Protobuf/ZMQ image-action path:

```text
server = ./build/vla-server
bind = tcp://127.0.0.1:6128
backend = CPU (CUDA_VISIBLE_DEVICES=-1)
image-kind = gradient
height = 180
width = 260
layers = 5
state = zero raw state
noise = zero action noise
```

- Request path:

```text
Python smoke client
  -> vla.proto PredictRequest
  -> RGB_U8 image bytes
  -> ZMQ tcp://127.0.0.1:6128
  -> src/serving/server.cpp decode_image()
  -> Inputs::images + lang_tokens + state + noise
  -> src/models/hy_vla.cpp predict()
  -> PredictResponse action_chunk
```

- `vision_pixels` server smoke:

```text
server vs C++ direct-debug:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

server vs Python reference:
mean_abs = 0.000000017455
max_abs  = 0.000000476837
rmse     = 0.000000050554
```

- 5-block action server smoke:

```text
server action sum = 64.703399658203

server vs C++ direct-debug:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

server vs Python reference:
mean_abs = 0.000044254986
max_abs  = 0.000964015722
rmse     = 0.000108005894
```

- CPU server timing:

```text
total = 1632.3 ms
vision = 887.9 ms
inference = 712.0 ms
other = 32.4 ms
```

- CUDA server smoke:

```text
backend = CUDA
server action sum = 64.705039978027

CUDA server vs CUDA direct-debug:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

CUDA server vs Python reference:
mean_abs = 0.000050669372
max_abs  = 0.000971972942
rmse     = 0.000109866553
```

- CUDA server timing:

```text
total = 237.2 ms
vision = 139.5 ms
inference = 88.3 ms
other = 9.4 ms
```

- 10-block CUDA server scaling smoke:

```text
backend = CUDA
resident weights = 6.66 GiB
image-kind = gradient
height = 180
width = 260
server action sum = 54.442058563232

10-block CUDA server vs 10-block CUDA direct-debug:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

total = 309.0 ms
vision = 142.9 ms
inference = 154.1 ms
other = 12.0 ms
```

- 20-block BF16 resident CUDA server scaling smoke:

```text
backend = CUDA
resident weights = 5.17 GiB
image-kind = gradient
height = 180
width = 260
server action sum = 33.079795837402

20-block BF16 CUDA server vs 20-block BF16 CUDA direct-debug:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

total = 473.3 ms
vision = 249.4 ms
inference = 205.3 ms
other = 18.7 ms
```

- 32-block BF16 resident CUDA boundary:

```text
backend = CUDA
text_layers = 32
vision_layers = all
requested CUDA buffer = 7497.99 MiB
result = cudaMalloc failed: out of memory
```

- Added OOM fallback switch:

```text
VLA_HY_VLA_CUDA_OOM_FALLBACK_CPU=1
```

- Behavior:
  - default behavior is unchanged: CUDA resident allocation failure still makes `model_load` fail;
  - with the switch enabled, a CUDA resident allocation failure releases the CUDA backend and reallocates the same weight context on the CPU backend;
  - this is not a performance path; it is meant to keep full-depth smoke/parity possible on the local 8GB machine.

- 32-block + full-vision fallback verification:

```text
text_layers = 32
vision_layers = all
initial backend = CUDA
CUDA allocation = OOM at 7497.99 MiB
fallback backend = CPU (8 threads)
resident weights = 7.32 GiB
debug_output = vision_pixels

fallback vs C++ direct vision_pixels:
mean_abs = 0.000000000000
max_abs  = 0.000000000000
rmse     = 0.000000000000

fallback vs Python vision_pixels:
mean_abs = 0.000000017455
max_abs  = 0.000000476837
rmse     = 0.000000050554
```

- Conclusion:
  - the local 3070Ti 8GB can run the 20-block BF16 resident serving smoke;
  - 32-block BF16 resident does not fit reliably with the current all-resident weight strategy;
  - fallback keeps full-depth CPU smoke/parity available;
  - to run full 32-block on local 8GB CUDA, the next engineering step is block streaming, partial resident cache, or quantized resident weights.

- Notes:
  - in the current Codex sandbox, ZMQ `bind(tcp://127.0.0.1:...)` requires elevated execution;
  - the server and client must run under the same permission/network view, otherwise the client may connect but receive no response;
  - an earlier CUDA server 5-block action smoke did not return before the 300s client timeout, but the server reported `served=0` when stopped; the root cause was that the sandboxed client request did not reach the elevated server;
  - once both server and client were run under the same elevated/network view, the CUDA server action smoke passed.

## 2026-06-21 Remote Python reference and 5-sample full-depth parity / 远端 Python Reference 与 5 样本全深度对齐

- 远端 Python reference 环境：
  - source: `/autodl-fs/data/Hy-Embodied-0.5-VLA-main`
  - checkpoint: `/autodl-fs/data/HY-VLA`
  - env: Python 3.10, PyTorch 2.7.1+cu128, `flash-attn==2.7.4.post1`

- 对比了本地源码与远端源码的关键文件 hash：

```text
modeling_hy_vla.py       local == remote
modeling_dual_tower.py   local == remote
configuration_hy_vla.py  local == remote
modeling_hunyuan_vl_mot.py differs only in flash-attn import fallback
```

- 具体差异：
  - 本地 `modeling_hunyuan_vl_mot.py` 保留 `hy_vla.utils.torch_flash_attn_fallback`，用于本地无可用 `flash-attn` 时的慢速 debug/parity；
  - 远端代码直接 `from flash_attn import ...`；
  - 因远端已安装可用 `flash-attn`，实际数学路径与原版一致；
  - `sample_actions()` 所在的 `modeling_hy_vla.py` 本地与远端完全一致。

- 确认了一个 reference harness 问题：
  - HY-VLA 原版 `sample_actions()` 中先执行 `x_t = noise`，随后 Euler loop 中执行 in-place `x_t += dt * v_t`；
  - 因此如果 parity 脚本在调用 `sample_actions(..., noise=noise)` 后再保存 `noise`，保存到的其实已经是最终 action；
  - 这不是原模型明显错误，而是原版实现的 in-place 更新语义；
  - 远端 reference v2 已改为 `noise=noise.clone()`，确保 dump 的 `noise.f32` 是真实初始噪声。

- 远端生成了 5 个统一随机 reference 样本：
  - output: `/autodl-fs/data/hy_vla_cpp_parity_5_v2`
  - local copy: `/tmp/hy_vla_cpp_parity_5_v2`
  - 每个样本包含 `prefix.f32`, `mask.i32`, `lang_tokens.i32`, `lang_mask.i32`, `state.f32`, `noise.f32`, `python_action.f32`。

- C++ direct debug 工具新增固定输入：
  - `--state-f32 STATE.f32`
  - `--noise-f32 NOISE.f32`
  - 这允许本地 C++ 使用远端 Python dump 的同一 state/noise 做 end-to-end action parity。

- 修复 C++ routed graph 容量：
  - 3-camera full prefix 加 32 层 routed action graph 会超过原 `ggml_new_graph_custom(..., 32768, ...)` 节点容量；
  - routed default path 已提升到 `131072`，避免 `cgraph->n_nodes < cgraph->size` assert。

- 第一轮 full-prefix 对齐结果较差：

```text
full prefix length = 240
mean_abs_avg = 0.0579574
max_abs_max  = 1.8389853
rmse_avg     = 0.1313536
```

- 根因不是 flow/action expert，而是 prefix padding 语义：
  - Python `sample_actions()` 使用 `prefix_pad_masks` 把 padded language tokens 从 attention 中屏蔽；
  - 当前 C++ direct `--prefix-f32` 模式把传入 prefix 视为全部有效，没有单独表达 prefix pad mask；
  - 因此 full-prefix 模式会让 padded language token 参与 attention。

- 临时 parity workaround：
  - 保留固定视觉/结构 token；
  - 只保留 `lang_mask == 1` 的有效 language tokens；
  - 生成 `prefix_compact.f32` 和 `mask_compact.i32`。

```text
sample000: 240 -> 181 tokens, valid language tokens = 5
sample001: 240 -> 183 tokens, valid language tokens = 7
sample002: 240 -> 180 tokens, valid language tokens = 4
sample003: 240 -> 183 tokens, valid language tokens = 7
sample004: 240 -> 181 tokens, valid language tokens = 5
```

- 5-sample compact-prefix, full 32-block, 10-step Euler, Python remote vs local C++ parity：

```text
sample000: mean_abs=0.001982694 max_abs=0.044725403 rmse=0.004044611
sample001: mean_abs=0.001703647 max_abs=0.016301960 rmse=0.002646090
sample002: mean_abs=0.001772186 max_abs=0.022888482 rmse=0.003347196
sample003: mean_abs=0.001161071 max_abs=0.015270472 rmse=0.001833013
sample004: mean_abs=0.002310109 max_abs=0.074430212 rmse=0.007241702

summary:
mean_abs_avg = 0.001785941
max_abs_max  = 0.074430212
rmse_avg     = 0.003822523
```

- 当前结论：
  - 服务器原始代码没有发现明显数学错误；
  - 之前 `noise == action` 是 parity dump 脚本保存时机错误；
  - full-prefix mismatch 是 C++ direct prefix 输入暂缺 pad-mask 语义；
  - 去除 padded language tokens 后，HY-VLA full-depth action path 已在 5 个远端随机样本上达到稳定低误差。

- 后续工程项：
  - 为 C++ prefix input 增加显式 `prefix_pad_mask` / attention mask 支持，替代 compact-prefix workaround；
  - 将同样的 5-sample parity 迁回标准 `vla-server` protobuf 请求路径；
  - 若要做更严格报告，补充 F32 GGUF 或 Python BF16/F32 controlled run，区分 BF16 权重、CUDA kernel 和 mask 语义带来的误差来源。

---

- Remote Python reference environment:
  - source: `/autodl-fs/data/Hy-Embodied-0.5-VLA-main`
  - checkpoint: `/autodl-fs/data/HY-VLA`
  - env: Python 3.10, PyTorch 2.7.1+cu128, `flash-attn==2.7.4.post1`

- Key local-vs-remote source hashes were checked:

```text
modeling_hy_vla.py       local == remote
modeling_dual_tower.py   local == remote
configuration_hy_vla.py  local == remote
modeling_hunyuan_vl_mot.py differs only in flash-attn import fallback
```

- The only source difference found:
  - the local `modeling_hunyuan_vl_mot.py` keeps `hy_vla.utils.torch_flash_attn_fallback` for slow local debug/parity when `flash-attn` is unavailable;
  - the remote code imports `flash_attn` directly;
  - since the remote environment has working `flash-attn`, the effective math path matches the upstream path;
  - `modeling_hy_vla.py`, which contains `sample_actions()`, is identical locally and remotely.

- One reference-harness issue was confirmed:
  - upstream `sample_actions()` does `x_t = noise`, then updates `x_t` in place with `x_t += dt * v_t` inside the Euler loop;
  - therefore, saving `noise` after calling `sample_actions(..., noise=noise)` saves the final action, not the initial noise;
  - this is not an obvious model bug; it is the in-place update semantics of the original implementation;
  - remote reference v2 now calls `sample_actions(..., noise=noise.clone())`, so `noise.f32` is the true initial action noise.

- Five shared random reference samples were generated remotely:
  - output: `/autodl-fs/data/hy_vla_cpp_parity_5_v2`
  - local copy: `/tmp/hy_vla_cpp_parity_5_v2`
  - each sample contains `prefix.f32`, `mask.i32`, `lang_tokens.i32`, `lang_mask.i32`, `state.f32`, `noise.f32`, and `python_action.f32`.

- The C++ direct debug tool now supports fixed raw inputs:
  - `--state-f32 STATE.f32`
  - `--noise-f32 NOISE.f32`
  - This lets local C++ consume the exact state/noise dumped by the remote Python reference.

- The C++ routed graph capacity was raised:
  - 3-camera full-prefix plus 32 routed action layers exceeded the old `ggml_new_graph_custom(..., 32768, ...)` capacity;
  - the routed default path now uses `131072` nodes to avoid the `cgraph->n_nodes < cgraph->size` assertion.

- The first full-prefix run was poor:

```text
full prefix length = 240
mean_abs_avg = 0.0579574
max_abs_max  = 1.8389853
rmse_avg     = 0.1313536
```

- Root cause: prefix padding semantics, not the flow/action expert:
  - Python `sample_actions()` uses `prefix_pad_masks` to mask padded language tokens out of attention;
  - the current C++ direct `--prefix-f32` mode treats every prefix row as valid and has no separate prefix pad mask;
  - therefore padded language tokens were incorrectly visible in full-prefix mode.

- Temporary parity workaround:
  - keep fixed visual/structure tokens;
  - keep only language tokens where `lang_mask == 1`;
  - write `prefix_compact.f32` and `mask_compact.i32`.

```text
sample000: 240 -> 181 tokens, valid language tokens = 5
sample001: 240 -> 183 tokens, valid language tokens = 7
sample002: 240 -> 180 tokens, valid language tokens = 4
sample003: 240 -> 183 tokens, valid language tokens = 7
sample004: 240 -> 181 tokens, valid language tokens = 5
```

- Five-sample compact-prefix, full 32-block, 10-step Euler parity between remote Python and local C++:

```text
sample000: mean_abs=0.001982694 max_abs=0.044725403 rmse=0.004044611
sample001: mean_abs=0.001703647 max_abs=0.016301960 rmse=0.002646090
sample002: mean_abs=0.001772186 max_abs=0.022888482 rmse=0.003347196
sample003: mean_abs=0.001161071 max_abs=0.015270472 rmse=0.001833013
sample004: mean_abs=0.002310109 max_abs=0.074430212 rmse=0.007241702

summary:
mean_abs_avg = 0.001785941
max_abs_max  = 0.074430212
rmse_avg     = 0.003822523
```

- Current conclusion:
  - no obvious math error was found in the remote original model code;
  - the earlier `noise == action` observation came from the parity dump script saving a mutated noise tensor;
  - full-prefix mismatch came from missing pad-mask semantics in the C++ direct prefix input;
  - after removing padded language tokens, the HY-VLA full-depth action path reaches stable low error on five remote random samples.

- Follow-up engineering items:
  - add explicit `prefix_pad_mask` / attention-mask support to the C++ prefix input, replacing the compact-prefix workaround;
  - rerun the same 5-sample parity through the standard `vla-server` protobuf request path;
  - for a stricter report, add an F32 GGUF or controlled Python BF16/F32 run to separate BF16 weight, CUDA kernel, and mask-semantics error sources.

## 2026-06-22 Full HY-VLA Native Runner + MEM Smoke

- The native RoboTwin runner now defaults to a full HY-VLA server configuration:
  - `--hy-vla-text-layers 32`
  - `--hy-vla-vision-layers all`
  - `--hy-vla-vision-cpu-sideload`
  - `--hy-vla-cuda-oom-fallback-cpu`
  - `VLA_HY_VLA_VIDEO_HISTORY=<img_history_size>` when history is enabled.

- The C++ HyViT2 frontend now has a MEM path controlled by
  `VLA_HY_VLA_VIDEO_HISTORY`.
  - Input order is interpreted as camera-major history frames, matching the
    native runner keys: `top_head.0..5`, `hand_left.0..5`, `hand_right.0..5`.
  - Each K-frame camera group is patch-embedded as `K * 196` tokens.
  - Blocks `3, 7, 11, ...` use the MEM SpaceTimeBlock-style time attention.
  - Only the current-frame tokens are kept before the 14x14 -> 7x7 merger.

- Full-load smoke on the local 8GB RTX 3070 Ti Laptop GPU succeeds with CPU
  vision sidecar:

```text
text_layers=32
vision_layers=27/all
vision_sidecar=cpu
CPU vision sidecar weights: 0.88 GiB
resident CUDA weights:      6.44 GiB
```

- Synthetic end-to-end full server request also succeeds:

```text
input: 3 cameras x 6 frames
output chunk: (40, 32)
first request total: 23143.6 ms
vision: 21186.9 ms
inference/action: 1759.0 ms
```

- Remote Python reference source/weight paths were located on the 4090 server:
  - source: `/autodl-fs/data/Hy-Embodied-0.5-VLA-main`
  - checkpoint: `/autodl-fs/data/HY-VLA`
  - reference samples: `/autodl-fs/data/hy_vla_cpp_parity_5_v2`

- `sample000` was copied locally to:
  - `outputs/hy_vla_remote_parity/sample000`

- Standard `vla-server` precomputed-prefix parity observations:
  - Full prefix `[240, 2048]` with Python padded language rows is not directly
    comparable because protobuf lacks a prefix pad mask.
  - Full prefix + raw action, normalized C++ state:

```text
MAE=0.208343
RMSE=0.548625
max_abs=3.749833
```

  - Full prefix + `VLA_HY_VLA_RAW_STATE=1` + `VLA_HY_VLA_RAW_ACTION=1`:

```text
MAE=0.078191
RMSE=0.164127
max_abs=0.956009
```

  - Compact prefix (`240 -> 181`, keeping only valid language tokens) +
    raw state/action:

```text
cpp_sum=50.1177597
py_sum=49.7745743
MAE=0.00198269
RMSE=0.00404461
max_abs=0.0447254
```

- Current conclusion:
  - the full-depth routed prefix/action path remains consistent with the remote
    Python reference when prefix padding semantics are matched;
  - direct full-prefix protobuf parity still needs explicit prefix-pad-mask
    support;
  - the next strict parity target is the new C++ MEM vision frontend itself,
    compared against Python `prepare_images -> embed_image(5D)`.

## 2026-06-23 MEM Vision / Action Parity Update

- Added strict C++ dump hooks for HY-VLA MEM vision parity:
  - `VLA_HY_VLA_DUMP_VIT_TOKENS_F32`: pre-merger current-frame ViT tokens
    `[num_groups,196,1152]`.
  - `VLA_HY_VLA_DUMP_VISUAL_TOKENS_F32`: post-merger visual tokens
    `[num_groups,49,2048]`.

- Added Python reference harness:
  - `scripts/run_hy_vla_mem_vision_reference.py`
  - It runs the original server-side HY-VLA Python source/checkpoint and dumps
    either pre-merger ViT tokens or post-merger visual tokens for deterministic
    3-camera x 6-frame synthetic inputs.

- Fixed C++ MEM vision replication issues:
  - build/free a separate ggml graph per camera/history group to avoid stale
    graph state and NaNs across multiple camera groups;
  - non-MEM HyViT2 blocks now run frame-independently over each 196-token
    frame, matching Python `(B*K,N,D)` batch semantics;
  - MEM SpaceTimeBlock now keeps time attention only along the K-frame axis and
    runs the following spatial attention frame-independently over 196 patches,
    matching Python `_time_softmax_on_v(...)` then `_space_attn(...)`.

- Vision parity against the unmodified Python source on the remote server:

```text
layers   cpp_sum      py_sum       MAE       RMSE      max_abs    corr
l0       1694.993     1706.870     0.002379  0.003173  0.093414   0.999986
l1       12894.501    12925.905    0.005517  0.007444  0.294632   0.999989
l4       17835.934    17870.674    0.006420  0.008641  0.298874   0.999988
l8       14476.997    14480.177    0.009586  0.012702  0.370777   0.999984
l12      28797.123    28782.537    0.013949  0.018457  0.600327   0.999984
full27   7716.319     7684.254     0.018081  0.023482  0.518370   0.999874
```

- Action parity with the C++-generated full MEM prefix (`[180,2048]`) and
  identical prefix mask, using `text_layers=1`:

```text
raw C++ action vs Python routed_sample:
MAE=0.000410
RMSE=0.000551
max_abs=0.002438
corr=0.999992
```

- Important interpretation:
  - C++ default server output denormalizes actions before returning them.
  - Python routed parity harness dumps raw/normalized model actions.
  - Comparing default C++ output to raw Python output gives a large false
    mismatch; set `VLA_HY_VLA_RAW_ACTION=1` for structural parity checks.
  - Python routed parity harness also uses raw zero state. Set
    `VLA_HY_VLA_RAW_STATE=1` together with `VLA_HY_VLA_RAW_ACTION=1` when
    comparing against it; otherwise the C++ default path normalizes state before
    inference and the sample trajectory is not the same fixture.

- Full-depth routed action parity with the C++-generated full MEM prefix:

```text
text_layers=32, raw state/action
cpp_sum=-5.400590
python_sum=-3.981240
MAE=0.016222
RMSE=0.038267
max_abs=0.344828
corr=0.991891
```

- Full image-path parity (`3 cameras x 6 frames`, full MEM vision + full
  32-layer text/action path) matches the precomputed-prefix result:

```text
VLA_HY_VLA_TEXT_LAYERS=32
VLA_HY_VLA_VISION_LAYERS=27
VLA_HY_VLA_VISION_CPU_SIDELOAD=1
VLA_HY_VLA_VIDEO_HISTORY=6
VLA_HY_VLA_RAW_STATE=1
VLA_HY_VLA_RAW_ACTION=1

cpp_sum=-5.400590
python_sum=-3.981240
MAE=0.016222
RMSE=0.038267
corr=0.991891
local timing: total=20675.6 ms, vision=19507.3 ms, action=1093.6 ms
resident CUDA weights=6.44 GiB, CPU vision sidecar=0.88 GiB
```

- Additional debug hooks added for parity work:
  - `VLA_HY_VLA_DEBUG_TIMESTEP=<float>` controls the timestep used by
    single-step debug paths such as `routed_joint_vt`.
  - `VLA_HY_VLA_DUMP_DENOISE_STEPS_F32=<path>` dumps the per-step sampled
    action tensor from the routed default denoise loop.

- Build check:

```text
cmake --build build --target vla-server -j$(nproc)
# passed
```
