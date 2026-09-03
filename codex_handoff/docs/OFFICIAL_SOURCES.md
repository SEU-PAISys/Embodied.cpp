# 官方资料索引（2026-07）

以下地址用于核对 Codex 的当前行为。产品会更新，实际界面以 `/model`、`/status` 和官方文档为准。

- Codex 模型选择：
  https://developers.openai.com/codex/models
- Codex CLI：
  https://developers.openai.com/codex/cli
- Codex 配置基础：
  https://developers.openai.com/codex/config-basic
- Codex 配置参考：
  https://developers.openai.com/codex/config-reference
- Codex 最佳实践：
  https://developers.openai.com/codex/learn/best-practices
- Codex 价格与使用限制：
  https://developers.openai.com/codex/pricing

关键点：

- GPT-5.6 Sol：复杂、开放、需要判断和打磨的工作；
- GPT-5.6 Terra：日常通用工作；
- GPT-5.6 Luna：明确、重复、高吞吐任务；
- 官方默认 Power：Sol + medium；
- 推理强度应从满足需求的最低档开始；
- High/Extra High 用于多步骤、多权衡的困难任务；
- Max/Ultra 不适合大多数普通任务；
- 使用 `AGENTS.md`、项目级 `.codex/config.toml` 和保守权限提高一致性。
