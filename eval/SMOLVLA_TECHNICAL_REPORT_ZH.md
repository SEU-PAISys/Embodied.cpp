# Embodied.cpp SmolVLA 移植与 LIBERO 验证报告

> 报告日期：2026-08-04  
> 实现分支：`feat/smolvla-libero`  
> 对比基线：`main` / `2fb77dd`  
> 内容范围：模型架构、权重转换、C++ 推理、服务接入、LIBERO 评测、数值一致性、构建与测试

## 一、项目概述

本项目为 Embodied.cpp 增加 LeRobot SmolVLA 的 C++/GGML 推理能力，并建立从 Hugging Face/LeRobot policy checkpoint 到 LIBERO 仿真结果的完整工具链。主要交付包括：

- SmolVLA policy 与视觉塔的 GGUF 转换器；
- SigLIP、SmolLM2 和 flow-matching action expert 的 C++ 推理路径；
- 可独立构建和启动的 `vla-smolvla-server`；
- LIBERO 四套件、多 episode、断点恢复与确定性 seed 支持；
- C++ 与官方 LeRobot evaluator 的同协议比较；
- 固定输入数值 parity、自动化测试和 CPU/CUDA 构建验证。

历史完整实验覆盖每个实现 400 episodes；最终代码另以 12 episodes 的确定性 smoke 验证 seed、恢复和报告协议。两组实验承担不同用途，不能把 smoke 成功率解释为完整套件成绩。

## 二、系统组成与数据流

完整执行链如下：

`LeRobot checkpoint` → `policy/mmproj GGUF 转换` → `SmolVLA C++ runtime` → `ZeroMQ 服务` → `LIBERO client/adapter` → `episode 结果` → `官方基线对照与汇总报告`

模型文件、生成视频、逐 episode JSON 和运行日志保留在本地 `checkpoints/` 与 `outputs/`，不进入源码提交。仓库只保存运行时、转换器、配置、测试和可复现说明。

## 三、模型架构与权重转换

### 3.1 原始检查点身份

本地原始检查点位于 `/root/checkpoints/smolvla_libero`，包含：

- `config.json`
- `model.safetensors`
- `policy_preprocessor.json`
- `policy_preprocessor_step_5_normalizer_processor.safetensors`
- `policy_postprocessor.json`
- `policy_postprocessor_step_1_unnormalizer_processor.safetensors`

关键配置为：

| 字段 | 实际值 | 含义 |
|---|---|---|
| `type` | `smolvla` | LeRobot policy 类型明确为 SmolVLA |
| `vlm_model_name` | `HuggingFaceTB/SmolVLM2-500M-Instruct` | 使用 SmolVLM2 视觉语言骨干 |
| `load_vlm_weights` | `true` | 使用真实 VLM 权重 |
| `train_expert_only` | `true` | LIBERO 微调训练动作专家 |
| `expert_width_multiplier` | `0.5` | 动作专家隐藏宽度为 480 |
| `chunk_size` | `50` | 一次生成 50 步动作块 |
| `num_steps` | `10` | flow-matching 去噪 10 步 |
| state/action | `8` / `7` | LIBERO 实际状态和动作维度 |
| 最大 state/action | `32` / `32` | 模型内部补齐维度 |

因此输入转换器的源文件本身就是 SmolVLA LIBERO policy，而不是本仓库内某个 C++ 模型的权重。

### 3.2 转换过程没有替换模型

`scripts/convert_smolvla_to_gguf.py` 直接读取以下 SmolVLA 特有前缀：

- `model.vlm_with_expert.vlm.model.text_model`
- `model.vlm_with_expert.vlm.lm_head.weight`
- `model.vlm_with_expert.lm_expert`
- `model.state_proj`
- `model.action_in_proj` / `model.action_out_proj`
- `model.action_time_mlp_in` / `model.action_time_mlp_out`

`scripts/convert_smolvla_mmproj_to_gguf.py` 则读取：

- `model.vlm_with_expert.vlm.model.vision_model`
- `model.vlm_with_expert.vlm.model.connector.modality_projection`

转换器仅改变存储格式和张量命名，没有用 π0.5、GR00T 或其他模型权重填充 SmolVLA。视觉塔通过 mtmd 的 SigLIP 路径导出为 identity-proxy，真实的 pixel-shuffle 与 `12288 -> 960` connector 仍取自 SmolVLA policy GGUF，并在 `models/smolvla.cpp` 中执行。

最终复验使用的权重哈希为：

| 文件 | SHA-256 |
|---|---|
| `smolvla-parity-rope10k.gguf` | `ddfbc78bafcdad9ae17ce3ed1596f7126b5d691e8241ec1f2535a228c48697dd` |
| `mmproj-smolvla-fixed2.gguf` | `d4b9f78400b80dee86b58ad1b112636ef37a1fdb46e6076447d60fc8683d1140` |

哈希用于标识报告对应的本地制品；GGUF 和原始模型权重按项目约定不提交 Git。

### 3.3 运行时架构分发

运行时的实际分发链为：

`smolvla.architecture` → `Arch::SMOLVLA` → `smolvla_create()` → `vla-smolvla-server`

如果构建时未启用 `MODEL_BUILD_VLA_SMOLVLA`，程序会明确报错并要求重新配置。`smolvla_create()` 还会再次检查 `smolvla.architecture == "smolvla"`；错误或缺失元数据会导致加载失败。

实际服务日志记录了：

- policy：`/root/checkpoints/smolvla/smolvla-parity-rope10k.gguf`
- mmproj：`/root/checkpoints/smolvla/mmproj-smolvla-fixed2.gguf`
- 架构：`vla: arch = smolvla`
- VLM：32 层、hidden 960
- action expert：32 层、hidden 480
- action chunk / flow steps：50 / 10
- backend：CUDA
- vision model：`lerobot_smolvla_libero` / SigLIP

## 四、实现设计与复用边界

### 4.1 仓库通用基础设施

实现沿用以下仓库通用基础设施：

- `ModelArchBase`、`VLAInput`、服务协议和模型工厂接口；
- 本仓库通用的 GGUF/ggml backend 加载模式；
- `models/pi05.cpp` 的文件骨架和错误处理风格；
- 转换器的 CLI、统计张量命名和 GGUF writer 使用方式；
- llama.cpp/mtmd 已纳入仓库的第三方能力；
- 官方 LeRobot SmolVLA 模块作为架构和张量语义参考。

源文件顶部和转换器注释记录了相关参考关系。

### 4.2 SmolVLA 专用实现

与 π0.5 不同、需要单独实现的主要逻辑包括：

- SmolVLM2 文本骨干和 `lm_expert` 的真实张量映射；
- plain Gemma-style RMSNorm，而不是 π0.5 的 AdaRMSNorm；
- 独立 `state_proj` 生成 prefix state token；
- SigLIP 输出的 pixel-shuffle 与真实 connector；
- VLM prefix KV 与动作专家之间的 cross-attention；
- action/time MLP 融合及 10 步 flow-matching 更新；
- MEAN_STD 状态归一化和动作反归一化；
- 固定 action-noise 输入、按 suite/task/episode 派生 seed；
- SmolVLA 官方基线调度、断点恢复、完整性判定和报告生成。

SmolVLA 运行时沿用项目模型适配器的组织形式，但算法路径和权重结构独立于其他模型实现。

### 4.3 许可证与依赖

新增 C++、Python 和测试文件使用项目统一的 Apache-2.0 许可证头及 `SEU-PAISys` 版权声明。模型结构以官方 LeRobot SmolVLA 实现和 checkpoint 配置为语义依据；GGML、GGUF 与 mtmd 能力由仓库现有 llama.cpp 依赖提供。

## 五、端到端流程完整性

### 5.1 已贯通的流程

| 阶段 | 对应实现 | 状态 |
|---|---|---|
| 原始 policy 获取 | LeRobot SmolVLA LIBERO checkpoint | 已验证 |
| policy 转换 | `scripts/convert_smolvla_to_gguf.py` | 已完成 |
| vision/mmproj 转换 | `scripts/convert_smolvla_mmproj_to_gguf.py` | 已完成 |
| 架构识别 | `runtime/arch.h`、`runtime/model.cpp` | 已完成 |
| C++ 推理 | `models/smolvla.cpp` | 已完成 |
| 独立服务目标 | `vla-smolvla-server` | 已完成 |
| 请求预处理 | tokenizer、双图像、state、mask、noise | 已完成 |
| LIBERO 适配 | relative control、action replay | 已完成 |
| 多 episode/多 suite 调度 | `run_smolvla_acceptance.py` | 已完成 |
| 断点恢复与协议校验 | seed/model/task/episode/action horizon 校验 | 已完成 |
| 官方 LeRobot 基线 | 同 checkpoint 独立运行 | 已完成 |
| 汇总报告 | JSON + Markdown、Wilson 区间、容差判定 | 已完成 |
| 数值 parity | 相同输入、相同 noise 的动作对比 | 已完成 |
| CPU/CUDA/全模型构建 | 三种干净构建 | 已完成 |
| 自动化回归测试 | 12 项 | 已通过 |

### 5.2 完整 benchmark 结果

历史完整验收覆盖四个 LIBERO suite，每套件 10 tasks、每 task 10 episodes，即每个实现 400 episodes：

| Suite | Embodied.cpp SmolVLA | 官方 LeRobot SmolVLA | 差值 |
|---|---:|---:|---:|
| LIBERO-Spatial | 79/100 | 69/100 | +10 pp |
| LIBERO-Object | 86/100 | 87/100 | -1 pp |
| LIBERO-Goal | 80/100 | 76/100 | +4 pp |
| LIBERO-10 | 39/100 | 33/100 | +6 pp |
| **合计** | **284/400（71.0%）** | **265/400（66.25%）** | **+4.75 pp** |

这组结果证明完整 LIBERO benchmark 的任务覆盖和运行链路已经打通。它早于最终的分套件 action-noise seed 派生修复，因此不能作为最终代码的逐 bit 可复现制品。

最终代码采用最小 PR smoke 协议重跑了四个 suite 的 task 0，每个 3 episodes：

| Suite | C++ | 官方基线 |
|---|---:|---:|
| LIBERO-Spatial task 0 | 1/3 | 1/3 |
| LIBERO-Object task 0 | 3/3 | 2/3 |
| LIBERO-Goal task 0 | 3/3 | 2/3 |
| LIBERO-10 task 0 | 0/3 | 0/3 |

该 smoke 的 `selected_protocol_complete=true`，且四个选中任务均满足预设的单侧容差：C++ 成功率不得比官方基线低超过 5 个百分点；C++ 高于官方的结果不会因此失败。由于样本只有 3 episodes，它只用于最终代码冒烟和可复现性验证，不代表 suite 成功率。

### 5.3 数值一致性

固定输入 parity 使用完全相同的两张 512×512 图像、8 维 state、48 个 language token/mask，以及固定的 50×32 action-noise：

| 指标 | 结果 |
|---|---:|
| 最大绝对误差 | 0.0322541595 |
| 平均绝对误差 | 0.00202303892 |
| RMSE | 0.00328528439 |
| 同一 C++ 请求重复最大差异 | 0 |
| NaN / Inf | 均无 |

这比单看仿真成功率更直接地证明 C++ 执行的是与官方 SmolVLA 对应的数学路径。

### 5.4 构建与测试

已完成的干净构建：

- CPU-only SmolVLA：212/212 build steps；
- CUDA SmolVLA（compute capability 8.9）：352/352；
- CUDA 全模型：372/372，并生成 π0.5、SmolVLA、HY-VLA、GR00T N1、LingBot-VA 五个服务目标。

唯一观察到的编译警告是 `serving/server.cpp` 中已有的 ZeroMQ deprecated `setsockopt`。

2026-08-04 回归测试命令：

```bash
cd /mnt/c/embodied.cpp/Embodied.cpp
PYTHONPATH=/mnt/c/embodied.cpp/Embodied.cpp/third_party/llama.cpp/gguf-py \
  /root/smolvla-port/bin/python -m unittest discover -s tests -v
```

结果为 **12/12 通过**，覆盖 converter dtype、固定 noise/action、seed 隔离、恢复判定、报告语义和 Wilson 区间。

## 六、文件结构与命名一致性

### 6.1 与其他模型一致的映射

| 职责 | 现有模型惯例 | SmolVLA 实现 | 结论 |
|---|---|---|---|
| 模型源文件 | `models/<model>.cpp` | `models/smolvla.cpp` | 一致 |
| 架构枚举 | `Arch::<MODEL>` | `Arch::SMOLVLA` | 一致 |
| 工厂函数 | `<model>_create()` | `smolvla_create()` | 一致 |
| CMake 开关 | `MODEL_BUILD_VLA_<MODEL>` | `MODEL_BUILD_VLA_SMOLVLA` | 一致 |
| 服务目标 | `vla-<model>-server` | `vla-smolvla-server` | 一致 |
| policy 转换器 | `convert_<model>_to_gguf.py` | `convert_smolvla_to_gguf.py` | 一致 |
| mmproj 转换器 | `convert_<model>_mmproj_to_gguf.py` | `convert_smolvla_mmproj_to_gguf.py` | 与 π0.5 一致 |
| LIBERO 配置 | `libero_<model>_eval.yaml` | `libero_smolvla_eval.yaml` | 一致 |
| 客户端 arch 名 | 小写下划线 | `smolvla` | 一致 |
| 输出目录 | `outputs/<model>_*` | `outputs/smolvla_*` | 一致 |

`smolvla` 不拆成 `smol_vla`，与 LeRobot policy 类型及官方品牌写法一致；展示文本使用 `SmolVLA`，编译宏使用 `SMOLVLA`，符合项目现有的“机器名小写、类型名大写、展示名保留品牌大小写”规则。

### 6.2 分支文件范围

相对 `main`，当前分支修改 21 个文件，新增约 2802 行、删除 17 行。文件按职责分布在已有目录中，没有建立与项目惯例冲突的顶层目录：

- 核心：`models/`、`runtime/`、`CMakeLists.txt`
- 转换：`scripts/`
- 评测：`eval/client/`、`eval/conf/`、`adapter/sim/`
- 测试：`tests/`
- 文档：根 README 与 `eval/`

结构审查未发现需要在 PR 前重命名或迁移的文件。

## 七、限制与优化方向

当前实现存在以下限制和后续优化空间：

1. **最终 seed 协议没有重新跑满 800 个对照 episodes。** 完整历史结果与最终 smoke 分别承担“覆盖证明”和“最终代码复现证明”，不能混为同一统计实验。
2. **小样本 smoke 不能估计成功率。** 例如 3/3 不代表真实成功率为 100%，0/3 也不能证明模型在该 suite 完全失败。
3. **pixel-shuffle 仍在 host 侧执行。** 该步骤只是连续内存重排；主要的 `12288 -> 960` connector GEMM 和 state projection 已移入 GGML/CUDA graph。后续可继续缓存固定形状的 graph，减少每请求建图与分配开销。
4. **模型制品未纳入 Git。** 复现者必须使用同一原始 checkpoint，并记录转换参数和 SHA-256。

## 八、复现实验清单

- [x] 准备 SmolVLA LIBERO policy checkpoint
- [x] 转换 policy 与 mmproj GGUF
- [x] 通过 GGUF 架构元数据加载 SmolVLA runtime
- [x] 完成端到端 LIBERO 全任务历史验收
- [x] 完成官方 LeRobot 基线对比
- [x] 完成最终 seed 协议最小复验
- [x] 完成固定输入数值 parity
- [x] 完成 12 项自动化测试
- [x] 完成 CPU、CUDA 和全模型干净构建
- [x] 记录目录、命名、许可证和运行说明
- [ ] 如需发表最终统计结果，以最终 seed 协议重跑完整 800-episode 对照

## 九、总结

该分支为 Embodied.cpp 建立了完整的 SmolVLA C++ 推理与 LIBERO 验证能力。权重转换、视觉语言前缀、动作专家、flow-matching、仿真客户端、官方基线、恢复协议、统计报告和构建测试均已连通。固定输入 parity 与完整历史 benchmark 表明实现具备可用的数值和任务级行为；最终确定性 smoke 则覆盖了最新 seed 与恢复逻辑。connector 和 state projection 移入后端 graph 后，三个确定性任务样例的平均推理耗时由 1029.91 ms/step 降至 274.23 ms/step，均保持成功。后续可缓存固定形状 graph，并在需要正式发布最终成功率时按最新协议重跑完整对照实验。
