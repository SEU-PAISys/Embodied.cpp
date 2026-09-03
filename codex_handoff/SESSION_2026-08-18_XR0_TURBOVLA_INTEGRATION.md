# 会话交接：XR0 & TurboVLA 集成、LIBERO 评估与已知问题

生成时间：2026-08-18
用途：供 Codex 在接手本仓库后，快速恢复当前进度、理解已完成的集成、并继续推进后续工作。

---

## 1. 一句话现状

仓库已在既有模型（pi0.5 / HY-VLA / OpenVLA / LingBot-VA）基础上**集成并评估了两个新 VLA**：**Xiaomi-Robotics-0（XR0，Qwen3-VL 骨干 + DiT flow head）** 和 **TurboVLA（DINOv3 + BERT + 双向交叉注意力 + ACT 解码器）**。两者都完成了 LIBERO 多 episode 冒烟评估，但尚无干净 commit；且集成过程中对共享 wire 协议的改动**导致旧模型 server（pi05/OpenVLA/HY-VLA）需重新编译才能配套当前客户端**。

---

## 2. 当前 Git 状态（接手前必须看）

```bash
cd /home/yozaxeen/projects/Embodied.cpp
git status --short          # 156 项改动，均为未提交
git log --oneline -5        # HEAD = b1083a5 (openvla-work 分支)
git branch -vv              # openvla-work * 领先，main 落后上游 7 个
```

- 分支：工作集中在 `openvla-work`（HEAD `b1083a5`，Merge upstream-main into openvla-work）。
- **尚未提交**：XR0/TurboVLA 的模型实现、转换脚本、proto 改动、README、评估脚本全部停留在工作区。
- `.gitignore` 已正确排除 `build/ checkpoints/ outputs/ *.gguf *.pth .venv/` 等大文件。
- 工作区两个调试残留文件 `gdb.txt`、`gdb_cmds.txt` 未入库，建议删除或 add 进 .gitignore。

> ⚠️ 接手第一件事：决定这些未提交改动是归入一个新 commit 还是整理后提交。**不要 `git reset --hard` / `git clean`**（见 codex_handoff 规则）。

---

## 3. 仓库模型全貌（6 个）

| 模型 | 视觉编码器 | 骨干 | 动作头 | chunk | 权重规模 | 状态 |
|---|---|---|---|---|---|---|
| pi0.5 | Qwen3-VL CLIP (GPU) | Qwen3-VL-4B (36层) | DiT flow (16层) | 12 | 6.5G | ✅ 已跑通 LIBERO |
| HY-VLA | SigLIP | Qwen3-VL 变体 | 离散 action token | 1 | 1.3B / 9.05GB bf16 | 旧二进制，需重编 |
| OpenVLA | SigLIP | Vicuna-7B | 256-bin 离散 token | 7 | 13.4GB | 旧二进制，需重编 |
| LingBot-VA | Qwen3-VL (CLIP) | Qwen3-VL | （WAM 用） | - | - | 旧二进制，需重编 |
| **XR0** | Qwen3-VL CLIP (**CPU**) | Qwen3-VL-4B (36层) | DiT flow (16层) | 30 | 8.0GB f16 + 836MB mmproj | ✅ 已跑通，但慢 |
| **TurboVLA** | DINOv3 | BERT（WordPiece, 固定 21 token） | ACT（6层双向交叉注意力） | 12 | 215.5M / 431MB | ✅ 已跑通，最快 |

arch 预设（chunk/n_action_steps/max_length 等）见 [eval/client/vla_cpp_client.py](file:///d:/桌面/embodied_cpp/Embodied.cpp/eval/client/vla_cpp_client.py#L38) 的 `ARCH_PRESETS`。

---

## 4. 本次会话完成的工作

### 4.1 XR0 集成
- 模型实现：[models/xr0.cpp](file:///d:/桌面/embodied_cpp/Embodied.cpp/models/xr0.cpp)
- 转换脚本：`scripts/convert_xr0_to_gguf.py`，数值校验（parity）全套通过，README 记录的 max abs err 3.3e-4。
- 关键修复（都在 C++ / 第三方）：
  - **CLIP CUDA warmup 崩溃**：`ggml_im2col` 分段错误 → 在 xr0.cpp 强制 CLIP 走 CPU（`cp.use_gpu=false`、`cp.flash_attn_type=DISABLED`、`cp.warmup=false`）。
  - **meta 缓冲溢出（deepstack qwen3vl 图节点 >8192）**：在 `third_party/llama.cpp/tools/mtmd/clip.h/clip.cpp` 加 `max_nodes` 字段，xr0.cpp 设 `cp.max_nodes=65536`，`mtmd.cpp` 聚合初始化补值。
- 评估：LIBERO object/task_0 ×5 ep，0/5 成功（任务难，正常），单步推理 91–158ms；单次 predict **~2.5s，其中视觉编码占 ~1.5s（CPU CLIP 是最大瓶颈）**。

### 4.2 TurboVLA 集成
- 模型实现：[models/turbovla.cpp](file:///d:/桌面/embodied_cpp/Embodied.cpp/models/turbovla.cpp)
- 转换脚本：`scripts/convert_turbovla_to_gguf.py`（单文件 GGUF，`--vocab` 必填，BERT vocab.txt 打包进 GGUF）。
- **独特集成模式**：客户端发原始文本 `language_text`，服务端用捆绑的 WordPiece 分词（`use_server_tokenizer=True`），BERT 固定 padding 到 21 token。
- 评估：LIBERO object/task_0 ×5 ep，0/5，单步推理 **11.6–18.5ms**（比 XR0 快约 8×）；单次 predict server 侧 ~60ms。

### 4.3 协议改动（关键 / 破坏性）
- [serving/vla.proto](file:///d:/桌面/embodied_cpp/Embodied.cpp/serving/vla.proto#L53)：为支持 TurboVLA 服务端分词，新增 `language_text`（field 10）。
- **后果**：旧二进制（pi05 8/6、OpenVLA/HY-VLA 7/28 构建）解析新消息错位 → **`std::bad_alloc` 崩溃**。
- **修复/验证**：重新编译 `vla-pi05-server` 后崩溃消失，LIBERO 冒烟跑通（3 ep，~101-109ms/step）。

> 🤖 结论（对 Codex 最重要）：**下次动 CMake/build 时，需同时重编所有启用的模型 server**，否则会出现「一个能跑、别的崩」的隐性故障。

### 4.4 LIBERO 成功率观察（重要，供 Codex 排查）

本次所有模型在 LIBERO 上的**成功率均为 0**：

| 模型 | 套件 / 任务 | episodes × steps | 成功率 | 单步推理 |
|---|---|---|---|---|
| pi05（重编后） | object / task_0 | 5 ep × 80 步 | 0/5 | ~101–109 ms |
| XR0 | object / task_0 | 5 ep × 80 步 | 0/5 | ~91–158 ms |
| TurboVLA | object / task_0 | 5 ep × 80 步 | 0/5 | ~12–18 ms |
| TurboVLA | spatial / task_0 | 3 ep × 200 步 | 0/3 | ~17–20 ms |

- 视频检查（`outputs/tv_spatial_t0_3ep/.../episode_000000.mp4`，201 帧）：画面平均亮度 123→130 缓慢变化，说明**机械臂在动、动作是有效的**，但不是停滞，而是从未达到任务完成态。
- 结论：**动作正常产生，但所有模型都拿不到 success**。可能原因（按可能性排序）：
  1. **权重未对 LIBERO 微调**：这些通用 checkpoints（除 pi05`_libero_finetuned_v044` 外）不是 LIBERO 专用权重，pick-and-place 成功率可能本就极低。
  2. **适配/归一化不一致**：action 反归一化、state 维度/pad、图像预处理与各模型官方 eval 不一致，导致动作语义偏移（最值得 Codex 优先核对）。
  3. 任务本身难度（object 全是 pick-X-put-in-basket，spatial 若 baseline 也不高则正常）。
- **建议 Codex 首查项**：对比 `adapter/sim/libero.py` + 各模型归一化 stats 是否符合模型官方 eval 脚本（尤其 action 的 mean/std 标度与 `un_normalize_action`）。

### 4.5 实测性能对比（同一台 RTX 4060 8GB，同机）

| 指标 | pi05（重编后） | XR0 | TurboVLA |
|---|---|---|---|
| 闭环单步推理 | 101–109 ms | 91–158 ms | 11.6–18.5 ms |
| 单次 predict server 总耗时 | 449 ms | 2513 ms | 60 ms |
| 视觉编码 | 57 ms（GPU） | 1490 ms（**CPU**） | 58 ms |
| 骨干+动作头 | 370 ms | 1010 ms | 49 ms |

---

## 5. 命令速查（快速恢复）

```bash
cd /home/yozaxeen/projects/Embodied.cpp

# 构建（当前 build 目录已启用了所有 MODEL_BUILD_* ；重编单个模型 server）
cmake --build build --target vla-pi05-server -j$(nproc)

# 先起 server（一次只跑一个，避免显存/端口冲突；端口默认 5555，可用 --bind 换）
./build/vla-pi05-server --bind tcp://*:5555 \
  checkpoints/pi05/pi05_libero_finetuned_v044/pi05-mmproj.gguf \
  checkpoints/pi05/pi05_libero_finetuned_v044/pi05.gguf
./build/vla-xr0-server --bind tcp://*:5556 \
  checkpoints/xr0/xr0-mmproj.gguf checkpoints/xr0/xr0.gguf
./build/vla-turbovla-server checkpoints/turbovla/turbovla.gguf   # 单文件 GGUF

# LIBERO 评估
eval/sim/libero/libero_uv/.venv/bin/python eval/client/run_sim_client_direct.py \
  --arch pi05 --libero-suite spatial --task-id 0 --n-episodes 30 --max-steps 0 --seed 42 \
  --tokenizer checkpoints/pi05/pi05_libero_finetuned_v044 \
  --vla-addr tcp://127.0.0.1:5555
# turbovla 无需 --tokenizer（服务端分词）
```

注意：`--tokenizer` 对 pi05 传**目录**（其下 `tokenizer.model`），组装 eval 客户端时对 XR0 传 `checkpoints/xr0/hf`。

---

## 6. 遗留的开放问题（给 Codex 的任务线索）

按优先级排序：

1. **提交工作区改动**：把 XR0/TurboVLA/协议改动整理成一个干净的 commit（参考 7 节文件清单），并删除 `gdb.txt`/`gdb_cmds.txt`。
2. **新旧 server 协调**：README 和构建文档需注明「改 proto 后所有 server 需重编」。可选：CI 或一键 `make all-server-bins`。
3. **XR0 性能债**：CPU CLIP（~1.5s/predict）是最大瓶颈。风险点：llama.cpp Qwen3VL CLIP 的 CUDA warmup 崩溃（`ggml_im2col`）——修复它或在 README 记录。
4. **TurboVLA 验证缺口**：只有转换脚本，**无 parity 数值校验 / 无 diag / 无量化脚本**。对比 openvla/hy_vla 都有 `quantize_*`，XR0 有完整 parity。建议补 TurboVLA parity + 量化。
5. **bench 覆盖**：[eval/client/bench_models.py](file:///d:/桌面/embodied_cpp/Embodied.cpp/eval/client/bench_models.py#L21) 覆盖 pi05/openvla/hy_vla/xr0，**缺 turbovla**。
6. **显存/COEXIST**：XR0 f16 8GB + TurboVLA 1.3GB 已把 8GB 显存吃满。无量化头寸。一次只跑一个模型最稳。

### 6.1 开源化待办（合入前按此清单核对）

目标：让仓库 clone 后可直接让别人 build + 跑通，不含本机调试残留。

- [ ] **提交工作区改动**（156 项未提交），按第 7 节文件清单归类提交。
- [ ] `.gitignore` 已追加忽略 `/gdb.txt`、`/gdb_cmds.txt`（本次已加），确认无需再漏。
- [ ] 确认 `checkpoints/`、`outputs/`、`build/`、`*.gguf`、`*.pth`、`.venv*/` 均不入库（.gitignore 已覆盖，报 `git status` 复核）。
- [ ] README「Known Limitations」补充：XR0 的 CPU CLIP 限制、TurboVLA 无 parity/量化验证、改 proto 后需重编所有 server。
- [ ] 补齐 TurboVLA parity 数值校验 + 量化脚本（如第 6 节第 4 条）。
- [ ] 补齐 [eval/client/bench_models.py](file:///d:/桌面/embodied_cpp/Embodied.cpp/eval/client/bench_models.py) 的 `turbovla` 配置。
- [ ] 确认删除的旧文件（smolvla/robolab/cosmos3/groot）不在 README 或依赖中有残留引用（`grep` 复核）。
- [ ] 跑一次干净构建：`rm -rf build && cmake -S . -B build -DMODEL_BUILD_* && cmake --build build`，确保文档里每个 server target 都能编译。
- [ ] （可选）提交前本地验证：3 个模型 server 各能起、至少各跑 1 个 LIBERO smoke episode 不崩溃。

---

## 7. 本次涉及的文件清单（未提交）

- **新增**：`models/xr0.cpp`、`models/turbovla.cpp`、`scripts/convert_xr0_to_gguf.py`、`scripts/convert_turbovla_to_gguf.py`、`scripts/parity_xr0_*.py`、`scripts/diag_xr0_*.py`、`scripts/bisect_xr0.py`、`tools/xr0_parity.cpp`、`eval/client/bench_models.py`、`eval/client/bench_xr0.py`、`eval/client/run_libero_eval*.py`、`eval/client/run_robotwin_native_hy_vla.py`。
- **修改**：`CMakeLists.txt`、`serving/vla.proto`、`serving/server.cpp`、`runtime/model.{h,cpp}`、`adapter/*`、`eval/client/run_sim_client_direct.py`、`eval/client/vla_cpp_client.py`、`adapter/sim/libero.py`、`README.md`、`third_party/llama.cpp/tools/mtmd/clip.{h,cpp}`、`third_party/llama.cpp/tools/mtmd/mtmd.cpp`（后三个为 CLIP max_nodes 补丁）。
- **删除**：一批 smolvla/robolab/cosmos3/groot 相关旧文件（见 `git status`，多为分支整合已删除项）。

---

## 8. 交接包已有的约定（继续沿用）

- 开发纪律见 `codex_handoff/AGENTS.md`（用户是电子工程大二学生，追求正确性 + 学习，不放水）。
- 环境/仓库规则见 `codex_handoff/docs/06_ENVIRONMENT_AND_REPO_RULES.md`。
- 研究方向候选见 `codex_handoff/docs/07_RESEARCH_DIRECTION.md`（量化敏感性 / 尾延迟 profiling / 视觉特征缓存 / 动态 action chunking）。
- 本仓库的正式 README（模型支持矩阵、构建、评估、转换、许可）见根目录 `README.md`，本次新模型的相关章节已补全。