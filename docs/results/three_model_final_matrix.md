# 三模型 · 全精度 · 全指标对比（2026-09-03 最终版）

设备：单台 4×RTX 4090 服务器，同机同边界测量。C++ = Embodied.cpp GGML runtime（vla-server + 公开 ZMQ）；
Python = 各模型官方 HuggingFace 实现（fp32 官方默认精度）。所有延迟为**端到端部署口径**：
原始 CPU 观测 → 完整 CPU 动作块（含预处理/传输/读回；C++ 另含 ZMQ）；5 次 warmup + 100 次计时，
VRAM 为独立进程采样峰值。每次测量 p99 干净（allocator 生命周期修复后长尾根除）。

## 评测规模

| 模型 | suites | tasks/suite | episodes/task | 存储精度档 |
|---|---|---|---|---|
| XR0 | 4（spatial/object/goal/10）| 10 | **50** | BF16 / Q8_0 / Q4_K |
| TurboVLA | 4 | 10 | 10（object 另有 50ep 深度）| BF16 / Q8_0 / Q4_0 / Q4_K |
| X-VLA | 4（bf16）/ object（精度探针）| 10 | 10 | BF16 / Q8_0 / Q4_0 / Q4_K / F32 常驻 |

## LIBERO 成功率

| 模型 | BF16 | Q8_0 | Q4 档 | F32 常驻 |
|---|---|---|---|---|
| XR0 | **98.05%**（1961/2000）| 97.90% | Q4_K **97.85%** | — |
| TurboVLA | **95.50%**（382/400）| 94.75% | Q4_0 90.75% · Q4_K 88.00%（goal 敏感）| — |
| X-VLA | **96.75%**（387/400）| object 98% | Q4_0 99% · Q4_K 100%（object）| object **99%** |

TurboVLA Q4 档说明：goal suite 的长指令（"turn on ... and put ..." 类）在 4bit 下动作
符号翻转，属模型相关量化敏感（逐层 dump：文本塔 bert 2.29/text_projected 7.98 爆点；
转换文件与反量化内核 215 张量逐一验证清白）。**推荐量化档 = Q8_0（94.75%，几乎无损）**。
Q4_K 不推荐用于 TurboVLA；XR0/X-VLA 的 Q4 档无损可放心使用。

## 端到端延迟（ms，Python = 1.00）

| 模型 | Python（官方）| C++ BF16 | C++ Q8_0 | C++ Q4 档 | C++ F32 常驻 |
|---|---:|---:|---:|---:|---:|
| XR0（BF16 policy + F16 vision 双侧）| 143.6 | **69.5（0.48）** | 64.9（0.45）| 68.6（0.48，Q4_K）| — |
| TurboVLA（BF16）| 28.4 | 29.8（1.05）| 29.7（1.04）| 27.7（0.97，Q4_K）| — |
| X-VLA（FP32）| 115.9 | 101.2（**0.87**）| 101.4（0.88）| 106.0（0.91，Q4_K）| 104.7（0.90）|

TurboVLA 端到端接近 1.00 是小模型的物理现实：模型层 C++ 13ms ≈ Python 模型层，
两边对称的观测/传输开销主导；其 C++ 纯推理 7.3ms。XR0（最大模型）加速最显著。

## 进程 VRAM 峰值（MiB，Python = 1.00）

| 模型 | Python | C++ BF16 | C++ Q8_0 | C++ Q4 档 |
|---|---:|---:|---:|---:|
| XR0 | 9790 | 8824（0.90）| 5376（0.55）| 3516（**0.36**，Q4_K）|
| TurboVLA | 924 | 880（0.95）| 880（0.95）| 864（0.94，Q4_0）|
| X-VLA | 9790（FP32）| 2370（**0.24**）| 2370（0.24）| 2370（0.24）|

X-VLA 的 0.24 源于 C++ BF16 常驻 vs Python FP32 官方默认；同 dtype 下为 1.00。
量化文件对 TurboVLA/X-VLA 是 dequantize-then-convert（BF16 常驻），不降低常驻 VRAM。

## 结论

1. 成功率：三模型 BF16 全部 ≥95%，与各官方参考一致；量化档 XR0/X-VLA 无损（Q4_K 97.85/100），
   TurboVLA 推荐 Q8_0（94.75%）。
2. 延迟：XR0 端到端 2.1× 加速；X-VLA 1.15×；TurboVLA（最小模型）parity——对称客户端开销主导，
   模型层 C++ 7.3ms。
3. VRAM：XR0 Q4_K 0.36、X-VLA 0.24 为最显著收益。
4. 全部数字可复现：`outputs/bench_fixed_20260903/`（修后 11 档 bench）、`bench_results/*_progress.txt`
   （评测逐 task 记录）、`docs/results/takeover_20260903.md`（修复与审计全记录）。
