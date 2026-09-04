# 三模型规范对标 · 闭环 ms/step 矩阵（v2，对标 SmolVLA 5.4 口径）

口径对齐 `eval/SMOLVLA_TECHNICAL_REPORT_ZH.md` §5.4：C++ = LIBERO 闭环 client
`get_action()`/RPC 逐 environment-step 摊销 wall（eval 日志 "Average inference
time per step"，全量 log 提取）；Python = 官方评测 episode 总 wall ÷ env steps。
这两个计时边界并不相同，即使同机也不能直接相除。新补齐的对照见下节，双侧均为
公共 `get_action()`。本文不把 allocator 显存与 nvidia-smi 进程显存混用。

## V2 复核：同机同边界 TurboVLA 对照

| 实现 | 成功回合 | 环境步数 | 按步数加权 get_action ms/step |
|---|---:|---:|---:|
| C++ | 20/20 | 2049 | 2.147 |
| 官方 Python 模型接入公共闭环 | 20/20 | 2061 | 3.637 |

RTX 4090、同权重 BF16、seed 42、spatial task 0/4 各 10 回合、256px、12 步回放。
含首次推理和动作队列，不含环境推进；不是 episode 总 wall。有限两任务样本约低 41%，
不替换 seed 7 主表。来源、逐回合记录、舍入误差见[复核报告](v2_followup_20260903.md#2-补齐-turbovla-同机公共闭环对照)。

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
| TurboVLA | 历史 64.5 ms/env-step 为 episode 总 wall；新公共闭环 3.637 ms/step | 历史 4060 Laptop / 新 RTX 4090 | 只对新公共闭环计算比值 |
| XR0 | 官方 PyTorch 2000ep seed7 产物无时间戳 | — | 不可推算；需重新计时 |
| X-VLA | 未建立同权重 Python 公共闭环 | — | 比值 Pending；不能混用固定 obs 前向 |

## 与 SmolVLA 规范的差距清单

1. TurboVLA 已补最小同机公共闭环；全套 Python 重跑仍未完成。XR0 耗时需新采集，
   X-VLA 尚缺匹配权重，不能据“无独立 evaluator”推断无法接公共入口。
2. XR0/X-VLA opt-in 现已作用于实际请求噪声并有回归测试；默认保持历史协议，
   新协议全量成功率尚未验收。TurboVLA ACT 没有随机 action-noise 输入。
3. 原 C++ 聚合行是各自历史批次，并非此新 20 回合对照，不据此跨批次计算比值。
