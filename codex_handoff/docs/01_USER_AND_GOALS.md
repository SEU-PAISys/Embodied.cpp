# 使用者画像与成功标准

## 技术起点

### 已具备

- C++ 基础语法、类、基础数据结构；
- 计算机组成基础；
- Transformer 的基本概念；
- 少量强化学习概念；
- 能阅读教程和跟随命令操作。

### 当前短板

- Python 与 PyTorch 缺少完整实践闭环；
- 对环境隔离、依赖版本和调试流程不熟；
- 对中大型 C++ 工程、CMake、GDB、CUDA profiling 不熟；
- 容易把大任务直接交给 AI，自己缺少 diff 审查和根因分析；
- 尚未形成严谨的实验记录、重复测量和消融习惯。

## 暑假结束的成功标准

### Python/PyTorch

- 独立完成数据加载、训练、评测、checkpoint、benchmark；
- 能解释常见 tensor shape、dtype 和 device；
- 能看懂 traceback 并定位首个根因。

### C++/系统

- 能使用 Git 分支、CMake、Ninja、GDB；
- 能跟踪函数调用链；
- 能解释对象生命周期、引用/指针与资源所有权；
- 能正确测量 CPU/GPU 延迟。

### Embodied.cpp

- 固定仓库 commit 并可复现构建；
- 至少成功编译一个 server target；
- 至少运行一个 smoke test 或最小请求；
- 画出 client → serving → adapter/runtime/model → action 的路径；
- 完成一个小型、可验证、可回退的贡献；
- 得到一份可信的基线数据。

### 科研能力

- 能提出一个清楚的研究问题；
- 有 baseline、指标、变量控制、重复实验；
- 能区分工程贡献和论文贡献；
- 能写 4–6 页技术报告。
