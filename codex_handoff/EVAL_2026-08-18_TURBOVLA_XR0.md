# TurboVLA / XR0 评测记录（2026-08-18）

## 结论

此前 handoff 的 LIBERO 评测确实是真实运行，但它使用了旧闭环协议、过期 XR0 server，且 object/spatial episode 被提前截断。本轮已恢复 LIBERO checkout 和可复用的 WSL `xr0env`，用当前重编 server、256×256、完整 suite 步数重新跑通闭环。

- **XR0：数值推理链路已通过。** 当前源码重编后的 C++ 输出与 PyTorch 参考在 `--cpp-vision` 条件下 `max abs err=0.002419`、`mean abs err=0.000057`，阈值 `5e-3` 下 PASS。
- **TurboVLA：C++ 加载和合成推理已通过，但还没有数值 parity。** 输出稳定为 `(12, 7)`，没有 NaN/Inf；预热后 server 延迟约 68–123 ms，端到端约 124–234 ms。
- **XR0 server 崩溃已定位并修复。** 当时 `vla-xr0-server` 是前一天的旧可执行文件，而 adapter/protobuf 已经是新构建产物；旧 server 与新共享库混用导致 `language_text` 判断异常，随后在 adapter 路径触发 `std::bad_alloc`。重建 `vla-xr0-server` 后，官方格式的 prompt/token/noise 请求已恢复。
- **多 task 协议 smoke：XR0 6/6，TurboVLA 6/6。** 这里的成功率只表示请求返回、shape 正确且无 NaN/Inf，不等同于 LIBERO 环境任务完成率。

## 参考成绩

这些是官方仓库声明/附带日志中的参考值，不是本地 C++ 结果：

| 模型 | 官方 LIBERO 参考 | 评测协议要点 |
|---|---:|---|
| XR0 | 98.7% 平均成功率 | 256×256，30-step flow chunk，每 10 步 replan |
| TurboVLA | 97.7% 平均成功率 | 256×256，12-step action chunk，每 12 步 replan |

XR0 的附带 `merged_results.json` 还记录了：Object 98.8%、Spatial 98.8%、Goal 98.8%、LIBERO-10 97.2%。

## 本地证据

### XR0

- GGUF：`checkpoints/xr0/xr0.gguf`，架构 `xr0`，action chunk 30、action dim 32、5 个 flow steps，LIBERO action stats 已打包。
- 当前 build 重新编译了 `xr0-parity` 和 `vla-xr0-server`，使用 CUDA 主干 + CPU CLIP；模型常驻约 7.29 GiB。需要注意：只构建 TurboVLA 不会自动刷新 XR0 可执行文件，启动前应确认目标文件时间戳与本轮源码一致。
- 有效 parity 命令：

  ```bash
  ./build/xr0-parity checkpoints/xr0/xr0-mmproj.gguf checkpoints/xr0/xr0.gguf checkpoints/xr0
  python3 scripts/parity_xr0_compare.py --parity-dir checkpoints/xr0 --atol 5e-3
  ```

- 当前结果：`PARITY: PASS`；最大误差出现在 gripper 维度，`0.002419`，仍低于 `0.005`。
- server smoke：重建后模型加载、adapter、vision、DiT 推理均通过；6 个不同 task 输入都返回 `(30, 32)` finite action chunk。单请求端到端约 2.0–5.0 s，其中 vision 约 1.4–1.7 s、inference 约 0.5–0.7 s（当前 RTX 4060 Laptop + CPU CLIP 配置）。

### TurboVLA

- checkpoint：`checkpoints/turbovla/object.pth`，864,588,190 bytes，`global_step=75000`，`suite=object`。
- checkpoint metadata 表明：训练是四套件混合训练，但该文件的官方评测入口是 `libero_object`；不能直接拿它声称 spatial/goal 的单套件结果。
- 配置：DINOv3 ViT-B、BERT-base、2 views、256×256、state 8、action 7、horizon 12；40 条 LIBERO 指令的 padding 长度为 11/14/21，C++ runtime 使用总容量 21 并对 padding 做 mask。
- GGUF 已生成：`checkpoints/turbovla/object.gguf`，约 411.5 MiB；包含 672 个 tensor、约 215.5M 参数和 WordPiece vocab。
- 合成输入 smoke：8 个不同随机双视图/状态/任务，96 条 action 全部 finite；输出范围和每维统计正常，gripper 在该 checkpoint/输入上稳定输出 `+1`。
- 多 task 协议 smoke：6 个不同 task 输入均返回 `(12, 7)` finite action chunk，成功率 `6/6 = 100%`；server latency 约 44–77 ms。
- 延迟统计（8 chunks）：
  - server：mean 69.83 ms，median 61.65 ms，p95 111.31 ms；
  - client wall：mean 135.79 ms，median 107.12 ms，p95 255.61 ms。
- 当前缺口：没有 TurboVLA 的 PyTorch↔C++ 数值 parity；LIBERO 闭环已经有本轮工程回归结果，但还不是大样本 benchmark。

## 本轮真实 LIBERO 闭环结果（2026-08-18）

评测命令使用 `eval/client/run_sim_client_direct.py`，`--max-steps 0`（使用 suite 上限）、`--observation-size 256`、seed 42；TurboVLA 使用 `checkpoints/turbovla/object.gguf`，XR0 使用 `checkpoints/xr0/xr0.gguf` + `checkpoints/xr0/hf` tokenizer。

| 模型 | suite/task | episode | steps | success | 平均推理 |
|---|---|---:|---:|---:|---:|
| TurboVLA | object/task_0 | 1 | 129 | 1/1 | 26.46 ms/step |
| TurboVLA | object/task_1 | 1 | 280 | 0/1 | 20.52 ms/step |
| TurboVLA | object/task_2 | 1 | 156 | 1/1 | 20.10 ms/step |
| TurboVLA | object/task_3 | 1 | 162 | 1/1 | 16.69 ms/step |
| TurboVLA | object/task_4 | 1 | 280 | 0/1 | 14.14 ms/step |
| **TurboVLA 合计** | **object/task_0–4** | **5** | — | **3/5 = 60%** | — |
| XR0 | object/task_0 | 1 | 134 | 1/1 | 303.44 ms/step |
| XR0 | object/task_1 | 1 | 120 | 1/1 | 316.40 ms/step |
| XR0 | object/task_2 | 1 | 116 | 1/1 | 286.20 ms/step |
| **XR0 合计** | **object/task_0–2** | **3** | — | **3/3 = 100%** | — |

这些是每个 task 只跑 1 episode 的工程回归结果，不是官方 benchmark 的统计置信度；但它们证明当前 C++ server、客户端 adapter、LIBERO 环境、图像/状态预处理和动作闭环已经能完成真实任务。TurboVLA 的 object checkpoint 不应据此宣称 spatial/goal 成绩。

## 为什么此前的 0/5 不足以定性

此前 handoff 中的 0/5 是真实 LIBERO 的单任务小样本，但不是官方完整基准，而且当时：

1. LIBERO direct evaluator 默认渲染尺寸是 360，而官方 XR0/TurboVLA 协议是 256；现已加入 `--observation-size`，默认 256。
2. XR0 client 原先 replay 30 个 action，官方 LIBERO 每 10 步 replan；现已改为默认 10。
3. 旧 build 与当前源码并不总是同一轮产物；当前 XR0 parity 已用重编二进制重新确认。
4. 当时 object 只跑到 80 步、spatial 只跑到 200 步；这比当前适配器定义的官方上限（object 280、spatial 220）短，不能排除任务尚未完成就被截断。
5. 当时的 WSL home 虚拟环境和 checkout 后来被清掉；本轮已从 archive 恢复 checkout，并复用现有 `xr0env` 补齐最小仿真依赖。

## 下一步最小闭环

1. 为 TurboVLA 增加固定输入 parity harness；没有 parity 之前只把 Turbo smoke 视为“能跑”，不视为“数值正确”。
2. 在当前已恢复环境上扩大 episode 数量：先把 object/task_0–4 各跑到多 episode，再评估四套件；TurboVLA 继续只评估 `libero_object` checkpoint。
3. 扩大到每 task 多 episode，并记录 success rate、每步/每 chunk 延迟、动作范围、跳过 episode 和视频；当前本轮已保存视频到 `outputs/turbovla/...` 和 `outputs/xr0/...`。
