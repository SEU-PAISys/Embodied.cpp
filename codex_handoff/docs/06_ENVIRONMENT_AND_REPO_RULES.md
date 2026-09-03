# 环境与仓库规则

## 环境隔离

- Windows 只负责桌面、NVIDIA Windows 驱动与 VS Code。
- 开发在 WSL2 Ubuntu 中进行。
- 源码放在 `~/workspace`，不放 `/mnt/c` 或 `/mnt/d` 进行高频编译。
- 每个 Python 项目一个 `venv`。
- 不向系统 Python 安装科研包。
- 不在 WSL 安装 Linux NVIDIA 显卡驱动。
- 编译 CUDA 程序时才安装 Toolkit，并先确认版本与路径。

## 仓库固定

第一次进入项目：

```bash
git rev-parse HEAD
git status --short
git submodule status
```

把 commit 写入实验记录。进行论文或 benchmark 时不要随意 `git pull`。

## 分支纪律

```bash
git switch -c learning/<task-name>
```

修改前：

```bash
git status
git diff
```

修改后：

```bash
git diff --check
git diff
```

不要：

```bash
git reset --hard
git clean -fdx
git push --force
```

除非你明确理解后果且做了备份。

## 大文件

禁止提交：

- GGUF/模型权重；
- checkpoints；
- datasets；
- build 目录；
- 仿真视频；
- Python 虚拟环境；
- API keys；
- `.env`；
- 大型日志。

## 构建失败记录

并行构建日志混乱时：

```bash
cmake --build <build-dir> -j1 2>&1 | tee build_error.log
```

记录：

- 当前 commit；
- 完整命令；
- 编译器、CMake、CUDA 版本；
- 第一个真实 error；
- 修改前后差异；
- 验证命令。

## 实验可复现

每次实验记录：

- 日期；
- commit；
- machine/GPU；
- driver/toolkit；
- Python 环境；
- 权重版本与 checksum；
- 命令行；
- seed；
- warm-up；
- 重复次数；
- 原始日志路径；
- 汇总脚本版本。
