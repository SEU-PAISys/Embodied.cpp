# 分阶段任务清单

Codex 每次只能领取一个最小任务。任务完成后更新状态，不自动推进下一项。

## P0：环境与安全

- [ ] 盘点 WSL、Python、CUDA、Git、CMake 状态；
- [ ] 输出环境报告，不做任何修改；
- [ ] 建立项目级虚拟环境；
- [ ] 验证 PyTorch GPU；
- [ ] 设置 Git 用户信息；
- [ ] 建立备份与 checkpoint 习惯。

验收：`scripts/check_environment.sh` 输出已保存。

## P1：Python/PyTorch

- [ ] CSV 性能统计脚本；
- [ ] PyTorch 数据加载；
- [ ] 训练循环；
- [ ] checkpoint；
- [ ] evaluate；
- [ ] benchmark；
- [ ] 三组实验；
- [ ] 周报告。

## P2：Tiny Transformer

- [ ] embedding；
- [ ] scaled dot-product attention；
- [ ] causal mask；
- [ ] multi-head reshape；
- [ ] Transformer block；
- [ ] 训练；
- [ ] 序列长度实验。

## P3：C++/CMake/CUDA

- [ ] CMake library/executable；
- [ ] 单元测试；
- [ ] chrono benchmark；
- [ ] GDB；
- [ ] CUDA vector add；
- [ ] CUDA transpose；
- [ ] CUDA naive matmul；
- [ ] 正确的同步与错误检查。

## P4：Embodied.cpp 构建

- [ ] 固定 commit；
- [ ] 运行环境检查；
- [ ] 初始化依赖；
- [ ] CMake configure；
- [ ] 编译单一 CPU target；
- [ ] 记录首个错误；
- [ ] 找到 main；
- [ ] 启动无权重路径或帮助信息；
- [ ] CUDA target；
- [ ] 权重加载。

## P5：调用链和 smoke test

- [ ] 客户端入口；
- [ ] protobuf 定义；
- [ ] ZeroMQ 连接；
- [ ] 服务端 handler；
- [ ] adapter；
- [ ] model inference；
- [ ] action 输出；
- [ ] 一个 smoke test；
- [ ] 流程图。

## P6：首个贡献

默认选题：request-level latency metrics。

- [ ] 设计指标和 timing boundary；
- [ ] 找到插入位置；
- [ ] 实现 warm-up；
- [ ] 结构化输出；
- [ ] Python 汇总脚本；
- [ ] 单元/集成验证；
- [ ] overhead 评估；
- [ ] 文档；
- [ ] PR review。

## P7：小型研究

- [ ] 固定模型/任务/seed；
- [ ] baseline；
- [ ] 量化配置；
- [ ] 离线误差；
- [ ] 动作平滑性；
- [ ] 延迟和显存；
- [ ] 闭环成功率；
- [ ] 模块级消融；
- [ ] 失败案例；
- [ ] 报告。
