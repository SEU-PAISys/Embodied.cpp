# 三模型规范对标 · 闭环 ms/step 矩阵（v2，对标 SmolVLA 5.4 口径）

口径对齐 `eval/SMOLVLA_TECHNICAL_REPORT_ZH.md` §5.4：C++ = LIBERO 闭环 client
`get_action()`/RPC 逐 environment-step 摊销 wall（eval 日志 "Average inference
time per step"，全量 log 提取）；Python = 官方评测 episode 总 wall ÷ env steps。
VRAM = nvidia-smi 进程峰值。**延迟比值要求双侧同机**。

## C++ 闭环 ms/step（4090，全量 log 提取）

| 模型 | BF16 | Q8_0 | Q4 档 | log 数 |
|---|---:|---:|---:|---|
| XR0 | 17.83 | 15.07 | 14.88（Q4_K）| 2000+/精度 |
| TurboVLA（256px）| 5.75 | 5.82 | 2.24*（Q4_K）| ~400/精度 |
| X-VLA | —（见下）| — | — | — |

*X-VLA BF16 闭环全量批次缺 log（96.75% 批次在 8/31，log 位置待归档提取）；
f32 常驻 object 4.90 ms/step（100 log）。TurboVLA Q4_K 的 2.24 受失败 episode
早停偏置，不可直接与 bf16 比较（成功率 88% vs 95.5%，失败回合步数短）。

## Python 官方侧现状（口径/设备）

| 模型 | 数据 | 设备 | 状态 |
|---|---|---|---|
| TurboVLA | 64.5 ms/env-step（object，100ep 推算：1807s/28000 步）| **4060 Laptop（WSL）** | ❌ 不同机，需 4090 重跑 |
| XR0 | official PyTorch 2000ep seed7（成功率有，per-step 待从 log 推算）| 待查 | 待核 |
| X-VLA | 官方无 LIBERO evaluator | — | 闭环比值不可得，用固定 obs bench 标注 |

## 与 SmolVLA 规范的差距清单

1. Python 官方闭环 4090 重跑（TurboVLA 需解决 turbo_git evaluate 依赖；
   XR0 官方 stack 部署待查；X-VLA 无官方 evaluator——只能固定 obs 标注）。
2. 三模型 client 的 action-noise seed 派生/resume 校验（SmolVLA PR 机制）未确认。
3. 报告格式对齐 VALIDATION/TECHNICAL_REPORT（双侧对照段+诚实标注）。
