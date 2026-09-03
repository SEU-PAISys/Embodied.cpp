# Codex 交接包：Embodied.cpp 暑期入门

这个交接包用于把你的背景、目标、工作边界、学习方式和任务路线交给 Codex。

## 你现在要做的事

1. 将整个文件夹保存到一个长期位置，例如：

   ```bash
   ~/codex-handoff/embodied-cpp
   ```

2. 先阅读：

   - `HANDOFF_SUMMARY.md`
   - `docs/03_CODEX_MODEL_PLAYBOOK.md`
   - `docs/04_TASK_BACKLOG.md`

3. 当你已经克隆 `Embodied.cpp` 后，在交接包目录执行：

   ```bash
   bash scripts/install_into_repo.sh ~/workspace/Embodied.cpp
   ```

   脚本不会覆盖现有文件；若目标文件已存在，会跳过并提示你手动合并。

4. 进入仓库：

   ```bash
   cd ~/workspace/Embodied.cpp
   codex
   ```

5. 在 Codex 中先执行：

   ```text
   /status
   /permissions
   /model
   ```

6. 将 `prompts/00_FIRST_SESSION.txt` 的内容作为第一条任务发送。

## 推荐的默认配置

- 模型：`gpt-5.6-sol`
- 推理：`medium`
- 文件权限：`workspace-write`
- 命令审批：`on-request`
- 网络搜索：`cached`

这是为了兼顾学习、质量、速度和可控性。遇到复杂的 CMake/CUDA、跨文件调用链或实验设计任务，再临时切换到 `Sol + high`。

## 最重要的原则

Codex 是结对老师和工程助手，不是替你完成暑假的外包团队。

每个任务必须满足：

- 一次只做一个可验证的小目标；
- 先解释，再修改；
- 修改后运行验证；
- 你必须查看 `git diff`；
- 你能够口头解释关键代码；
- 任何系统安装、驱动、全局 Python、删除文件、重写 Git 历史的操作必须由你确认。
