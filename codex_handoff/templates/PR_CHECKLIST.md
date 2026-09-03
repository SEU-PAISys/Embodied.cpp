# PR 检查清单

## 目标

- [ ] 一个清楚、可测的目标
- [ ] 没有无关重构
- [ ] 默认行为未意外改变

## 代码

- [ ] 修改范围尽可能小
- [ ] ownership/lifetime 清楚
- [ ] 错误处理合理
- [ ] 无明显 UB、数据竞争或同步错误
- [ ] 没有硬编码本机路径
- [ ] 没有 secrets/weights/build artifacts

## 测试

- [ ] 构建通过
- [ ] 最窄相关测试通过
- [ ] smoke test 通过
- [ ] 失败路径考虑
- [ ] 记录完整命令

## 性能

- [ ] warm-up
- [ ] 重复测量
- [ ] CPU/GPU 同步正确
- [ ] timing boundary 解释
- [ ] overhead 评估
- [ ] mean/std/p50/p95/p99

## 文档

- [ ] 说明如何启用
- [ ] 输出格式
- [ ] 回退方式
- [ ] 已知限制

## 人工审查

- [ ] 已阅读 `git diff`
- [ ] Codex `/review`
- [ ] 能解释关键改动
- [ ] 学长/维护者 review
