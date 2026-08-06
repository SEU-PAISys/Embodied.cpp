# SmolVLA 项目交接说明

> 更新日期：2026-08-06
> 工作区：`C:\embodied.cpp\Embodied.cpp`
> 分支：`smolvla-dev-rebase`
> Draft PR：[SEU-PAISys/Embodied.cpp#6](https://github.com/SEU-PAISys/Embodied.cpp/pull/6)

## 当前结论

SmolVLA 的权重转换、C++/GGML 推理、ZeroMQ 服务、LIBERO 四套件评测、多 episode 统计、断点恢复、官方 LeRobot 对照、固定输入 parity、CPU/CUDA 构建和测试均已打通。

性能优化提交 `4988ea8` 已把原来位于 CPU 的 `12288 -> 960` 视觉 connector GEMM 和每请求读取/计算的 `state_proj` 移入与 VLM/action expert 相同的 GGML backend graph，并让权重常驻后端。host 侧只保留成本较低的 pixel-shuffle 内存重排。

优化后的完整 C++ LIBERO 仿真已完成，共 40 tasks、400 episodes，最终为 **279/400（69.75%）**，全任务平均推理时间约 **291.52 ms/step**。runner 和 server 均已停止，GPU 已释放。

## Git 与 PR

基于 `origin/dev` (`490913d`) 本地重放后的提交：

```text
3b8cd45 docs(smolvla): align notes with dev server layout
64dd7f5 fix(smolvla): use 64-bit GGUF seeks
62b3e8b docs(smolvla): add project handoff notes
62b507e perf(smolvla): move input projections into backend graph
0b6dbdf docs(smolvla): add detailed Chinese technical report
e3d09c9 docs(smolvla): document setup and validation results
4522e73 test(smolvla): add reproducible LIBERO acceptance workflow
0b6b2ee feat(smolvla): add C++ runtime and GGUF converters
```

提交作者为 `刘唱 <3496280049@qq.com>`。现有 PR #6 的目标仍为
`SEU-PAISys/Embodied.cpp:main`，来源仍为 `kchzhlc:feat/smolvla-libero`；本地
重放分支尚未推送，GitHub 上的 PR 文件列表因此仍不是最新 `dev` 结构。

下一位 agent 接手后应先执行：

```powershell
cd C:\embodied.cpp\Embodied.cpp
git status -sb
git log --oneline -6
```

## 模型、环境与权重

原始 policy：`/root/checkpoints/smolvla_libero`。

关键配置：

```text
type=smolvla
vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Instruct
chunk_size=50, num_steps=10, n_action_steps=1
real_state_dim=8, real_action_dim=7
max_state_dim=32, max_action_dim=32
```

C++ 运行制品：

```text
/root/checkpoints/smolvla/mmproj-smolvla-fixed2.gguf
/root/checkpoints/smolvla/smolvla-parity-rope10k.gguf
```

SHA-256：

```text
d4b9f78400b80dee86b58ad1b112636ef37a1fdb46e6076447d60fc8683d1140  mmproj-smolvla-fixed2.gguf
ddfbc78bafcdad9ae17ce3ed1596f7126b5d691e8241ec1f2535a228c48697dd  smolvla-parity-rope10k.gguf
```

主要环境：Windows + WSL2，RTX 4060 Laptop 8 GiB，CUDA compute capability 8.9。LIBERO Python 位于 `/root/Embodied.cpp/eval/sim/libero/libero_uv/.venv/bin/python`；SmolVLA/parity Python 位于 `/root/smolvla-port/bin/python`。

## 关键文件

| 文件 | 用途 |
|---|---|
| `models/smolvla.cpp` | SigLIP 后处理、SmolLM2、state token、action expert、flow-matching graph |
| `runtime/arch.h`、`runtime/model.cpp` | 架构注册、GGUF 检测与工厂分发 |
| `scripts/convert_smolvla_to_gguf.py` | policy GGUF 转换 |
| `scripts/convert_smolvla_mmproj_to_gguf.py` | SigLIP/mmproj GGUF 转换 |
| `eval/client/vla_cpp_client.py` | 请求、noise、动作队列和耗时解析 |
| `eval/client/run_sim_client_direct.py` | LIBERO 单套件/任务执行 |
| `eval/client/run_smolvla_acceptance.py` | 四套件调度、恢复和报告 |
| `eval/client/reproducibility.py` | suite/task/episode seed 派生 |
| `eval/conf/libero_smolvla_benchmark.yaml` | 完整 benchmark 配置 |
| `scripts/check_smolvla_parity.py` | 固定输入动作 parity |
| `tests/test_smolvla_*.py` | converter、seed、恢复与报告测试 |

运行时链路为：`smolvla.architecture` → `Arch::SMOLVLA` → `smolvla_create()` → 统一 `vla-server`。SmolVLA 的 10 个 flow steps 已经位于一个 policy graph 中，并非逐 step 零散调用；但每个请求仍会重建固定形状 graph 和 allocator，这是后续可能的优化点。

## 最新完整 C++ 仿真

结果目录：`outputs/smolvla_optimized_full/cpp`。汇总：`outputs/smolvla_optimized_full/report.json`。

| Suite | 成功数 | 成功率 | task 平均耗时的均值 |
|---|---:|---:|---:|
| LIBERO-Spatial | 72/100 | 72% | 285.88 ms/step |
| LIBERO-Object | 88/100 | 88% | 290.44 ms/step |
| LIBERO-Goal | 77/100 | 77% | 293.03 ms/step |
| LIBERO-10 | 42/100 | 42% | 296.73 ms/step |
| **合计** | **279/400** | **69.75%** | **291.52 ms/step** |

协议为每套件 tasks 0–9、每 task 10 episodes、environment seed 1000、action-noise base seed 1000、relative control、`n_action_steps=1`、10 flow steps。

重要：本次只运行 C++。新 `report.json` 中的 `official` 列是 runner 从既有 `outputs/smolvla_acceptance/official` 读取的历史结果，不是本次重跑。历史官方基线为 Spatial 69/100、Object 87/100、Goal 76/100、LIBERO-10 33/100，合计 265/400（66.25%）。因此可说新 C++ 比既有本地官方结果高 3.5 个百分点，但不能称为本次同机重新运行的 A/B 测试。

## 性能与正确性

旧版典型速度约 1.03 秒/step；新版完整运行平均约 0.292 秒/step，约快 3.5 倍。三个完全相同 seed/noise-seed 的定向样例为：

| 样例 | 修改前 | 修改后 | 加速 | 结果 |
|---|---:|---:|---:|---|
| Object task 0 episode 0 | 1050.60 | 258.04 ms/step | 4.07x | 成功 → 成功 |
| Spatial task 0 episode 0 | 998.46 | 291.65 ms/step | 3.42x | 成功 → 成功 |
| Goal task 0 episode 0 | 1040.66 | 273.01 ms/step | 3.81x | 成功 → 成功 |

优化前完整历史结果为 284/400，优化后为 279/400，相差 1.25 个百分点。固定输入 parity 仍通过：

```text
shape=50x7
max_abs_error=0.0322541595
mean_abs_error=0.00202303892
RMSE=0.00328528439
repeat_max_abs_error=0
NaN/Inf=false
```

固定参考位于 `/root/smolvla_parity/inputs.npz` 和 `/root/smolvla_parity/python_action.npy`。12/12 Python 回归测试通过；优化后 CPU 和 CUDA SmolVLA target 均构建通过。

测试命令：

```bash
cd /mnt/c/embodied.cpp/Embodied.cpp
PYTHONPATH=/mnt/c/embodied.cpp/Embodied.cpp/third_party/llama.cpp/gguf-py \
  /root/smolvla-port/bin/python -m unittest discover -s tests -v
```

构建目录：

```text
/root/embodied-pr-build-cpu
/root/embodied-pr-build-cuda-smol
```

性能运行使用的 CUDA binary 在最后一次仅修改启动提示文字之前构建，因此旧日志可能显示 `connector matmul on host`；实际 binary 已包含 backend connector 优化，并同时打印 `connector resident backend`。重新增量构建即可同步提示文字。

## 启动与恢复命令

服务：

```bash
tmux new-session -d -s smolvla-full-server \
  /root/embodied-pr-build-cuda-smol/vla-server \
  --bind tcp://127.0.0.1:5566 \
  /root/checkpoints/smolvla/mmproj-smolvla-fixed2.gguf \
  /root/checkpoints/smolvla/smolvla-parity-rope10k.gguf
```

完整 C++ benchmark：

```bash
/root/Embodied.cpp/eval/sim/libero/libero_uv/.venv/bin/python \
  /mnt/c/embodied.cpp/Embodied.cpp/eval/client/run_smolvla_acceptance.py \
  --run cpp --episodes 10 --task-ids 0 1 2 3 4 5 6 7 8 9 \
  --seed 1000 --noise-seed 1000 \
  --cpp-config /mnt/c/embodied.cpp/Embodied.cpp/eval/conf/libero_smolvla_benchmark.yaml \
  --cpp-output /mnt/c/embodied.cpp/Embodied.cpp/outputs/smolvla_optimized_full/cpp \
  --report /mnt/c/embodied.cpp/Embodied.cpp/outputs/smolvla_optimized_full/report.json
```

runner 按 task 检查完整性；不加 `--force` 会跳过已完成的 10-episode task。`outputs/`、checkpoint、视频和日志均为 Git ignored 本地制品，不应提交 PR。

## 当前运行状态与下一步

截至本文档写入时：完整 runner 已结束，server 已停止，相关 tmux 会话不存在，GPU 显存约 10 MiB。

建议下一位 agent：

1. 重新增量构建 CUDA target，使日志文字与源码一致。
2. 检查并提交本交接文档，若需要则推送到 PR #6。
3. 更新 PR 描述中的完整 C++ 结果为 279/400、291.52 ms/step，并注明官方列是历史基线。
4. 做最终 code review，重点检查 graph 生命周期、显存和异常路径。
5. 若继续优化，优先评估固定形状 compute context、graph 和 gallocr 缓存；每次修改必须跑 parity 和少量同 seed task。
6. 老师确认后再把 Draft PR 转为 Ready for review。

相关文档：`eval/SMOLVLA_TECHNICAL_REPORT_ZH.md`、`eval/SMOLVLA_VALIDATION.md`、`eval/README.md` 和根目录 `README.md`。
