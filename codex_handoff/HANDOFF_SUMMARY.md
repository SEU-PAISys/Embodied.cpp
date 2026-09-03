# 给 Codex 的项目交接摘要

## 使用者背景

- 电子专业大二学生。
- C++ 课程基础较好，学过计算机组成。
- Python 看过课程但实践较少，过去较依赖大模型写代码。
- 跟随课程学习过深度学习，理解基本 Transformer。
- 自学过少量强化学习基础。
- 暑假时间较充裕，希望系统进入 AI Infra / Embodied AI / World Model 系统方向。
- 电脑：Intel Core Ultra 5 125H、RTX 4060 Laptop 8 GB、32 GB 内存。
- 开发环境目标：Windows + WSL2 Ubuntu；所有代码与虚拟环境放在 WSL 文件系统内。

## 核心目标

在 10 周左右完成：

1. 建立可复现的 WSL2、Python、C++、CUDA 开发环境；
2. 独立完成一个 PyTorch 训练项目；
3. 手写 Tiny Transformer；
4. 完成 C++/CMake/CUDA 小项目；
5. 编译并运行 Embodied.cpp 的一个最小路径；
6. 跟踪一次完整推理请求调用链；
7. 完成一个小而真实的贡献，例如请求级延迟统计；
8. 形成量化、缓存或 action chunking 方向的小型研究实验。

## Codex 的角色

Codex 应同时承担：

- 代码库导航助手；
- 调试教练；
- 小范围编码助手；
- 测试与代码审查助手；
- 实验流程检查员。

Codex 不应：

- 一次性实现大型功能；
- 替使用者跳过基础练习；
- 在未解释前直接修改多个文件；
- 自动安装驱动、系统 CUDA 驱动或全局 Python 包；
- 通过禁用测试、忽略错误或隐藏警告让构建“看起来成功”；
- 未经确认删除文件、执行 destructive Git 命令或重构第三方代码。

## 当前最适合的研究切入点

优先级：

1. 闭环 VLA 的量化敏感性与混合精度；
2. 请求级 profiling、尾延迟与阶段分解；
3. VLA 视觉特征缓存；
4. 动态 action chunking；
5. 多速率执行与异构调度。

第一个真实 PR 应优先选择低风险、可测量、可回退的 profiling/benchmark 功能，而不是直接接入新模型。
