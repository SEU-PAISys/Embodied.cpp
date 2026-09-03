# 10 周暑期执行路线

建议每周 25–30 个有效小时，每周至少休息一天。

## 第 1 周：干净环境与 Python 数据分析

交付：

- WSL2 中 GPU 可见；
- 项目级 `venv`；
- CSV 性能分析脚本；
- mean/std/p50/p95/p99/success rate；
- Git 仓库与 README。

Codex 用法：

- 环境盘点和诊断可以用 Sol medium；
- 不允许 Codex 自动修改驱动或全局环境；
- Python 主体先自己写，Codex 只 review。

## 第 2 周：完整 PyTorch 训练闭环

交付：

- CIFAR-10 或 MNIST 训练；
- checkpoint 保存和恢复；
- evaluate 与 benchmark 脚本；
- 三组受控实验；
- 显存和训练时间记录。

Codex 用法：

- 数据流与 shape 解释：Sol/Terra medium；
- 小函数 review：Terra medium；
- 不让 Codex一次生成完整仓库。

## 第 3 周：Tiny Transformer

交付：

- embedding、Q/K/V、causal attention、multi-head、MLP、残差；
- 每层 shape 记录；
- 序列长度对延迟/显存的影响表。

Codex 用法：

- 先让 Codex检查数学和 shape，不直接写；
- 遇到 mask 或维度 bug 用 Sol high。

## 第 4 周：C++、CMake、Git、GDB

交付：

- 小型 C++ runtime demo；
- library + executable + test；
- `std::chrono` 计时；
- GDB 跟踪一次调用；
- 分支、diff、commit。

Codex 用法：

- CMake 链接错误：Sol high；
- 简单函数测试：Terra medium；
- 每次最多修改 1–3 个源文件。

## 第 5 周：CUDA 与 GGUF 推理

交付：

- vector add、transpose、naive matmul；
- 区分 kernel、传输、同步和端到端时间；
- 体验 GGUF 量化推理；
- 记录模型加载、显存和速度。

Codex 用法：

- CUDA 同步、错误检查和 timing review：Sol high；
- 重复格式化或日志：Luna low/medium。

## 第 6 周：Embodied.cpp CPU 构建

交付：

- 固定 commit；
- 初始化依赖；
- 编译一个 CPU server target；
- 保存完整构建命令和错误日志；
- 找到 `main()` 和启动路径。

Codex 用法：

- 首次架构扫描：Sol high，严格只读；
- 编译根因：Sol high；
- 不允许直接修改 third-party。

## 第 7 周：CUDA 构建与模型权重

交付：

- 确认 compute capability；
- 构建一个 CUDA target；
- 使用适合 8 GB 显存的量化权重；
- 记录加载时间、内存和 OOM 情况。

Codex 用法：

- CMake/CUDA/链接问题：Sol high；
- 不自动安装 Linux GPU driver；
- 所有安装命令先解释和确认。

## 第 8 周：一个 smoke test 与调用链

交付：

- 一个 episode 或最小请求；
- client → ZeroMQ/Protobuf → C++ server → inference → action；
- 请求和响应字段表；
- 一张 Mermaid 或手绘流程图。

Codex 用法：

- 先只读映射；
- 一个线程查 serving，一个线程查 model 仅在用户已经熟悉后再考虑；
- 初学阶段不要 Ultra。

## 第 9 周：首个 PR——请求级延迟统计

交付：

- request_id；
- inference_ms 与 total_ms；
- warm-up；
- CSV 或结构化日志；
- mean/std/p50/p95/p99；
- 构建与 smoke test 不回归。

Codex 用法：

- 计划和定位：Sol high；
- 实现：Sol medium 或 Terra medium；
- 独立 review：Sol high；
- 必须人工看 diff。

## 第 10 周：量化敏感性小研究

交付：

- BF16/Q8/Q6/Q4 等可行组合；
- 模型大小、加载时间、显存、平均/尾延迟；
- 离线动作误差、平滑性、闭环成功率；
- 至少一个模块级消融；
- 4–6 页报告。

Codex 用法：

- 实验设计：Sol high；
- 批量汇总：Luna/Terra；
- 结果解释和漏洞审查：Sol high。
