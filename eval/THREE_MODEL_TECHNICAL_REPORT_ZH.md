# Embodied.cpp 三模型（XR0 · TurboVLA · X-VLA）移植与 LIBERO 验证报告

> 对标 `eval/SMOLVLA_TECHNICAL_REPORT_ZH.md` 的结构，汇总 XR0（Xiaomi-Robotics-0）、
> TurboVLA、X-VLA 三个 runtime 的移植验证结果。数据来源为本分支既有评测产物
> （2026-08-18 ~ 09-03）。V2 复核补入同机 20 回合对照，并撤回错误的 X-VLA 权重/性能结论；
> [复核报告](v2_followup_20260903.md)与[原始证据摘要](v2_followup_20260903_evidence.json)为更正依据。

## 一、项目概述

三个 VLA 模型以统一整体形式并入 `integ-upstream` 分支：共享 `vla-server`/ZMQ 服务、
共享 LIBERO 闭环 client（`eval/client/run_sim_client_direct.py`）与 GGUF 权重转换链。
本文档汇总三者的 LIBERO 成功率（C++ 与官方 PyTorch 双侧）、数值一致性（parity）、
wall-clock 与显存样本。所有评测在本地/服务器 RTX 4090 或标注设备上完成。

## 二、系统组成与数据流

- 服务端：`serving/vla-server`（单进程多 arch：xr0/turbovla/xvla），GGUF 权重加载
  （F32/BF16 常驻；k-quant 文件经 `ggml_get_type_traits()->to_float` 反量化后常驻 BF16）。
- 客户端：`eval/client/run_sim_client_direct.py --arch <arch>` 驱动 LIBERO 闭环
  （robosuite），动作以 open-loop chunk 回放（turbo 12 步、xr0 30 步、xvla 30 步）。
- 精度档：GGUF 存储精度（Q8_0/Q4_0/Q4_K）≠ 常驻计算精度（反量化后 BF16 常驻），
  除 XR0 F32 档与 X-VLA `VLA_XVLA_F32_WEIGHTS=1` 的 F32 常驻配置。

## 三、模型架构与权重转换

### 3.1 原始检查点身份

| 模型 | 官方来源 | 关键文件 | SHA256（节选） |
|---|---|---|---|
| XR0 | Xiaomi-Robotics-0 官方 stack | xr0.gguf + xr0-mmproj.gguf | 主文件 `07dd9256…`（Codex 审计）|
| TurboVLA | H-EmbodVis/TurboVLA `libero.pth` | turbovla_all4.gguf | `787c01bd…`（669 张量全匹配）|
| X-VLA | 2toINF/X-VLA-Libero | xvla-libero.gguf | F32 存储 902 张量 |

TurboVLA 单一 joint checkpoint 跨四 suite 复用（官方 PyTorch 参考同法）。
X-VLA 现存 HF 与历史主 GGUF **权重不同**：生产转换映射下 902 个张量仅 1 个完全一致，
901 个不同，独立 F32 原始字节抽查也不同。此前“872/902 一致，其余为读取伪影”的说法撤回。

### 3.2 转换过程没有替换模型

- XR0：616/616 主模型张量 + 316/316 vision 张量与 HF 快照逐字节一致
  （`xr0-source-audit-bytes.json`，修正 uint16 视角误报后的正确结果）。
- TurboVLA：上传 `object.pth` 与本地哈希一致；与 GGUF 的 669 张量全匹配。
- X-VLA：本次源权重与转换映射检查均为有限数值，**未复现**“30 个 NaN bias”转换器缺陷。
  可疑旧产物不恢复使用，但其原因不能据误读认定。F32 计算需要
  `VLA_XVLA_F32_WEIGHTS=1` 并核对服务器实际日志，不能只看文件名。

### 3.3 运行时架构分发

`serving/vla-server` 按 GGUF `general.architecture` 分发至 `models/{xr0,turbovla,xvla}.cpp`；
k-quant 源文件统一经 `ggml_get_type_traits()->to_float` 反量化（Q8_0/Q4_0/Q4_K/Q6_K 全支持；
Q4_K 转换经 215 张量逐一离线验证，最大误差 0.067）。

## 四、实现设计与复用边界

复用仓库通用基础设施（proto/ZMQ、LIBERO client、GGUF 转换链、parity 工具）；
模型专用实现（DaViT/BERT 融合、flow-matching unrolled 图、CLIP+F16 vision）
在各 `models/*.cpp`。CUDA graph allocator 生命周期已统一提升至模型对象
（100 次同输入批次的 p99：TurboVLA 223→28 ms、XR0 258→64、X-VLA 248→94）。
另有 500 请求持续运行验证，不将有限样本解释为任意负载下永久无长尾。

## 五、端到端流程完整性

### 5.1 已贯通的流程

三模型均贯通：官方权重 → GGUF → vla-server → LIBERO 闭环（四 suite 全 task）
→ 逐 episode JSON/视频 → 聚合统计。渲染统一 256×256 双视角（client 默认已修正）。

### 5.2 LIBERO 成功率（C++ 与官方 PyTorch 双侧）

**XR0 历史归档**（50 ep/task，2000 total/档；官方 PyTorch seed 7，归档 C++ 种子未记录）：

| Suite | 官方 PyTorch | C++ bf16 | C++ q8_0 | C++ q4_k |
|---|---:|---:|---:|---:|
| object | 99.4% | 99.4% | 99.4% | 100.0% |
| spatial | 99.0% | 98.2% | 98.6% | 98.8% |
| goal | 97.4% | 98.4% | 98.4% | 98.0% |
| libero_10 | 97.2% | 97.8% | 96.2% | 97.4% |
| **总** | **98.25%** | **98.45%** | 98.15% | 98.55% |

另一次 seed 42 产物复点数：BF16 1961/2000（98.05%）、Q8_0 1958/2000（97.90%）、
Q4_K 1957/2000（97.85%）。此前本表将历史逐 suite 行与新批次总数混在一起，现已分开；
逐项数据见 [XR0 报告](xr0_libero.md)。

**TurboVLA**（10 ep/task，400 total/档；seed7 批次，256px）：

| Suite | PyTorch¹ | C++ bf16 | C++ q8_0 | C++ q4_k |
|---|---:|---:|---:|---:|
| spatial | 98/100 | 99/100 | 100/100 | 95/100 |
| object | 100/100 | 100/100 | 100/100 | 85/100 |
| goal | 97/100 | 97/100 | 99/100 | 82/100 |
| libero_10 | 94/100 | 89/100 | 87/100 | 82/100 |
| **总** | **389/400 = 97.25%** | **385/400 = 96.25%** | **386/400 = 96.50%** | 344/400 = 86.00% |

¹ 官方单 checkpoint 跨 suite 的近似参考；权威基线为官方均值 97.7%。
mask/padding/渲染修复后的 seed42 复验（1200 episodes）：95.50/94.75/90.75（Q4_0），
与 seed7 批次一致性见 `takeover_20260903.md`。

**X-VLA**（10 ep/task，400 total/档；seed7 批次）：

| Suite | 官方 PyTorch | C++ bf16 | C++ q8_0 | C++ q4_k |
|---|---:|---:|---:|---:|
| spatial | 100/100 | 99/100 | 100/100 | 100/100 |
| object | 100/100 | 100/100 | 100/100 | 97/100 |
| goal | 99/100 | 98/100 | 99/100 | 100/100 |
| libero_10 | 100/100 | 100/100 | 100/100 | 98/100 |
| **总** | **399/400 = 99.8%** | **397/400 = 99.2%** | **399/400 = 99.8%** | 395/400 = 98.8% |

**官方权重重建矩阵（2026-09-05，HF 260cc588 / GGUF 2c828fe6，seed42+派生噪声，
8 配置 × 400ep，全部严格 4×10 覆盖 PASS）**：

| 配置 | 成功率 | | 配置 | 成功率 |
|---|---:|---|---|---:|
| C++ BF16 | 390/400 (97.50%) | | Python BF16 | 391/400 (97.75%) |
| C++ F32 | 387/400 (96.75%) | | Python F32 | 391/400 (97.75%) |
| C++ Q8_0 | 388/400 (97.00%) | | C++ Q6_K | 393/400 (98.25%) |
| C++ Q4_0 | 391/400 (97.75%) | | C++ Q4_K | 393/400 (98.25%) |

历史 seed7 批次互相印证（差异 ≤1.7pp）。性能（官方权重，同卡 4090）：
C++ BF16 96.4 ms vs Python F32 110.0 ms（**0.88**）；VRAM 2370 vs 4142 MiB
（**0.57**）；C++ F32 115.0 ms（1.05）。历史快照（3f16…）派生的
parity/延迟/显存数字全部隔离作废。

X-VLA 修复后 seed42 复验：bf16 387/400（96.75%，256px 口径）；object 精度探针
Q8_0 98/100 · Q4_0 99/100 · Q4_K 100/100 · **F32 常驻 99/100**（主 GGUF +
`VLA_XVLA_F32_WEIGHTS=1`）。mask/padding 修复与 256 口径影响见 takeover 报告。

### 5.3 数值一致性（parity）

| 模型 | 固定输入 parity | 最终动作误差 | 阈值 |
|---|---|---:|---|
| TurboVLA | 8 阶段中间层 + 最终动作 | **0.00418**（短指令）/ 0.00281–0.00530（修复后复验）| atol 0.01 ✅ |
| XR0 | 主模型+vision 双路 | 0.00936 | atol 0.01 ✅ |
| X-VLA | 新同源 HF/GGUF 固定输入 | F32 **0.000203**；BF16 **0.003104** | atol 0.005 ✅ |

TurboVLA 满长指令（SEP@20）mask 缺陷修复后，短/满长指令 parity 全部通过；
逐层 dump 定位记录于 takeover 报告（文本塔为误差爆点，视觉塔次之）。

### 5.4 C++ 与官方 PyTorch 的 wall-clock 样本和显存

严格区分公共 `get_action()`、episode 总 wall、固定观测前向与完整输入/输出测速。
前两种即使除以同样的步数也不等价；同机只是必要条件，不是充分条件。

新补 TurboVLA 公共闭环：RTX 4090、同权重 BF16、seed 42、spatial task 0/4 各 10 回合，
双方均 20/20 成功。相同 `get_action()` 边界下，按环境步数加权 C++ **2.147**、
Python **3.637 ms/step**，约低 41%。这是两任务修复版对照，不替换 seed 7 主表。
XR0 历史官方 2000 回合缺时间戳，无法推算 per-step；历史 X-VLA 缺匹配权重，
另建的新同源 HF/GGUF 对照已有固定输入 parity 和重复性能测试，但没有完整成功率对照。

固定原始 CPU 观测 → 完整 CPU 动作块（5 warmup + 100 calls，同卡 RTX 4090）：
XR0 双侧 BF16 policy/F16 vision：C++ **53.56**、Python **143.63 ms**；
TurboVLA 双侧 BF16：C++ **27.24**、Python **28.39 ms**。
进程采样峰值分别 8824/9790、880/924 MiB。
历史 X-VLA C++ BF16 为 **84.41 ms / 2370 MiB**，因对应 HF 权重缺失，
其 Python 比值仍为 **Pending**。新同源对照的三轮 BF16 顺序交叉测试没有稳定的
C++ 延迟优势，且 C++ 进程显存高 3.9%，不能替代历史行或宣称加速。
115.9 ms 旧 Python 短测只有 20 次前向且不含预处理/读回，日志的 3.437 GiB
为 allocator 峰值，不能把 XR0 的 9790 MiB 套给它。
完整 std/p50/p95/p99、抽样限制与原始来源见[复核报告](v2_followup_20260903.md)。

### 5.5 构建与测试

五组 Python 回归共 **50 项全通过、0 skip**，Linux 安装安全 **7 项全通过**；
量化回归包含真实 Q6 codec、元数据数组保持和 Q4_K 215 张量逐一验证，
parity 回归（SEP@10/13/18/20 mask 用例）、真实 symlink 安装测试全部通过；
三个独立构建目录（CPU/CUDA 单模型/CUDA 全模型）此前已验证。

## 六、与其他模型一致的映射

与 SmolVLA/pi0.5/GR00T N1.7/HY-VLA 相同的映射规则：`models/<arch>.cpp` +
`serving/vla-server` 分发 + `eval/conf/libero_<arch>_eval.yaml` 配置 +
`eval/client/run_sim_client_direct.py` 闭环。三模型的 conf yaml 与 client
参数化（--tokenizer/--observation-size/--vla-addr）已对齐该映射。

## 七、遗留与后续

1. XR0 官方 Python 闭环 per-step：官方 2000ep 产物仅含 rollout 视频/JSON，**无时间戳
   记录**，per-step 不可推算——如需该口径需重跑官方评测并显式计时（后续工作）。
2. TurboVLA 同机公共闭环最小对照已补，完整 Python 四 suite 重跑未补。
3. XR0/X-VLA 的 `--derive-episode-noise` 现已真正控制请求噪声并有回归测试，
   默认关闭保留历史序列；新协议全量成功率未重跑。TurboVLA ACT 无 action-noise 输入，
   LingBot 使用独立噪声选项，均不接受此开关。
4. TurboVLA Q4/Q6 档的 goal 损失源于 GGUF 缺失逐指令 padding 元数据（与 BF16
   的 goal 回归同源），并非量化固有退化。runtime 内置 padding 表修复后：
   正确布局 Q6 全量 **378/400（94.50%）**（我们独立复验 379/400，±1 episode 噪声级），
   同协议 BF16 **382/400（95.50%）**，goal 98/100，逐回合双侧精确检验 p=0.424，
   不称无损或等价；Q4_K 同法升至 **92.50%**（goal 86）。详见
   [Q6 产物审计](q6_artifact_audit_20260904.md)。imatrix 只是一种候选方向，
   在有校准实验前不作为已定位的根因或既定方案。
