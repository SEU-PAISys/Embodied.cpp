# Embodied.cpp 三模型（XR0 · TurboVLA · X-VLA）移植与 LIBERO 验证报告

> 对标 `eval/SMOLVLA_TECHNICAL_REPORT_ZH.md` 的结构，汇总 XR0（Xiaomi-Robotics-0）、
> TurboVLA、X-VLA 三个 runtime 的移植验证结果。数据来源为本分支既有评测产物
> （2026-08-18 ~ 09-03），未做新的重跑；各表注明测量日期与协议。

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
X-VLA 历史 HF 快照与主 GGUF 的张量级一致性经 Codex 审计（872/902 直接一致，
差异集中于 30 个 bias 的读取伪影与 2 个 per-domain 转置，见 §X-VLA）。

### 3.2 转换过程没有替换模型

- XR0：616/616 主模型张量 + 316/316 vision 张量与 HF 快照逐字节一致
  （`xr0-source-audit-bytes.json`，修正 uint16 视角误报后的正确结果）。
- TurboVLA：上传 `object.pth` 与本地哈希一致；与 GGUF 的 669 张量全匹配。
- X-VLA：转换器 f32 路径存在 30 个 bias NaN 缺陷（已定位，重转产物废弃）；
  正确 F32 配置为主 GGUF + `VLA_XVLA_F32_WEIGHTS=1`。

### 3.3 运行时架构分发

`serving/vla-server` 按 GGUF `general.architecture` 分发至 `models/{xr0,turbovla,xvla}.cpp`；
k-quant 源文件统一经 `ggml_get_type_traits()->to_float` 反量化（Q8_0/Q4_0/Q4_K/Q6_K 全支持；
Q4_K 转换经 215 张量逐一离线验证，最大误差 0.067）。

## 四、实现设计与复用边界

复用仓库通用基础设施（proto/ZMQ、LIBERO client、GGUF 转换链、parity 工具）；
模型专用实现（DaViT/BERT 融合、flow-matching unrolled 图、CLIP+F16 vision）
在各 `models/*.cpp`。CUDA graph allocator 生命周期已统一提升至模型对象
（修复逐请求释放导致的 150–300 ms 周期尖峰：turbo p99 223→27 ms、XR0 258→67、xvla 248→94）。

## 五、端到端流程完整性

### 5.1 已贯通的流程

三模型均贯通：官方权重 → GGUF → vla-server → LIBERO 闭环（四 suite 全 task）
→ 逐 episode JSON/视频 → 聚合统计。渲染统一 256×256 双视角（client 默认已修正）。

### 5.2 LIBERO 成功率（C++ 与官方 PyTorch 双侧）

**XR0**（50 ep/task，2000 total/档；官方 PyTorch seed7）：

| Suite | 官方 PyTorch | C++ bf16 | C++ q8_0 | C++ q4_k |
|---|---:|---:|---:|---:|
| object | 99.4% | 99.4% | 99.4% | 100.0% |
| spatial | 99.0% | 98.2% | 98.6% | 98.8% |
| goal | 97.4% | 98.4% | 98.4% | 98.0% |
| **总** | — | **98.05%** | 97.90% | 97.85% |

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

X-VLA 修复后 seed42 复验：bf16 387/400（96.75%，256px 口径）；object 精度探针
Q8_0 98/100 · Q4_0 99/100 · Q4_K 100/100 · **F32 常驻 99/100**（主 GGUF +
`VLA_XVLA_F32_WEIGHTS=1`）。mask/padding 修复与 256 口径影响见 takeover 报告。

### 5.3 数值一致性（parity）

| 模型 | 固定输入 parity | 最终动作误差 | 阈值 |
|---|---|---:|---|
| TurboVLA | 8 阶段中间层 + 最终动作 | **0.00418**（短指令）/ 0.00281–0.00530（修复后复验）| atol 0.01 ✅ |
| XR0 | 主模型+vision 双路 | 0.00936 | atol 0.01 ✅ |
| X-VLA | AutoModel vs C++ | 诊断性（详见 takeover §2）| — |

TurboVLA 满长指令（SEP@20）mask 缺陷修复后，短/满长指令 parity 全部通过；
逐层 dump 定位记录于 takeover 报告（文本塔为误差爆点，视觉塔次之）。

### 5.4 C++ 与官方 PyTorch 的 wall-clock 样本和显存

口径：C++ = LIBERO 闭环 client 逐 environment-step 摊销 wall（日志
"Average inference time per step"，全量 log 提取）；Python 官方 = 评测 episode
总 wall ÷ env steps（TurboVLA 为 8/23 本地 4060 Laptop 样本，**与 4090 C++ 不同机，
比值仅作量级参考**）。VRAM 为 nvidia-smi 进程峰值（双侧同口径）。

| 模型 | C++ 闭环 ms/step（bf16/q8/q4）| Python 官方 ms/step | 进程显存 C++ / Python |
|---|---|---|---|
| XR0 | 17.83 / 15.07 / 14.88（Q4_K）| 官方闭环 per-step 待从 2000ep log 推算 | 8824 / 9790 MiB |
| TurboVLA | 5.75 / 5.82 / 2.24*（Q4_K）| 64.5（4060 Laptop，**不同机**）| 880 / 待测 |
| X-VLA | f32 常驻 4.90（object）；bf16 全量 log 待归档提取 | 官方无 LIBERO evaluator（固定 obs：115.9 ms/forward，4090 同机）| 2370 / 9790 MiB |

*TurboVLA Q4_K 的 2.24 受失败 episode 早停偏置，不与 bf16 直接比较。
X-VLA Python 无官方 LIBERO evaluator，其延迟采用固定观测前向样本并如实标注。

端到端部署口径（固定观测 → 动作块，100 calls，4090 同机，p99 干净）：
XR0 69.5 ms（0.48）/ TurboVLA 29.8 ms（1.05，小模型对称开销主导，模型层 13 ms）/
X-VLA 101.2 ms（0.87）。详见 `docs/results/three_model_final_matrix.md`。

### 5.5 构建与测试

集成测试 42 项、Linux 安装安全 7 项、量化回归（Q4_K 215 张量逐一验证）、
parity 回归（SEP@10/13/18/20 mask 用例）、真实 symlink 安装测试全部通过；
三个独立构建目录（CPU/CUDA 单模型/CUDA 全模型）此前已验证。

## 六、与其他模型一致的映射

与 SmolVLA/pi0.5/GR00T N1.7/HY-VLA 相同的映射规则：`models/<arch>.cpp` +
`serving/vla-server` 分发 + `eval/conf/libero_<arch>_eval.yaml` 配置 +
`eval/client/run_sim_client_direct.py` 闭环。三模型的 conf yaml 与 client
参数化（--tokenizer/--observation-size/--vla-addr）已对齐该映射。

## 七、遗留与后续

1. XR0 官方 Python 闭环 per-step 从 2000ep log 推算（零成本，待做）。
2. TurboVLA 官方 Python 的 4090 同机闭环样本（需 turbo_git evaluate 依赖链）。
3. 三模型 client 的 action-noise seed 派生/resume 校验（对齐 SmolVLA PR 机制）。
4. TurboVLA Q4 档 goal 敏感：如需 4bit 可用，方向为 imatrix/敏感层高保真量化（后续工作）。
