# Model Graph Spec / 模型图规范

This directory uses a two-layer graph convention.

本目录采用双层图谱规范。

## Layer A: Block Graph / A 层：Block 图

Block graphs describe module-level data flow. They are used to compare model structure, token topology, and VLA versus world-action design differences.

Block 图描述模块级数据流，用来比较模型结构、token 组织方式，以及传统 VLA 与 world-action model 的差异。

Recommended node granularity:

- input stream: image, language, robot state, action history, noise, timestep
- encoder/front-end: vision encoder, text encoder, VAE encoder
- repeated block stack: transformer blocks, action expert blocks, world-action blocks
- head/postprocess: velocity head, action head, VAE decoder, action postprocess
- cache/state: prefix KV cache, latent cache, resident weights

推荐节点粒度：

- 输入流：图像、语言、机器人状态、动作历史、噪声、时间步
- 编码器/前端：视觉编码器、文本编码器、VAE encoder
- 重复 block 栈：transformer blocks、action expert blocks、world-action blocks
- 输出头/后处理：velocity head、action head、VAE decoder、action postprocess
- 缓存/状态：prefix KV cache、latent cache、resident weights

## Layer B: Operator Graph / B 层：Operator 图

Operator graphs expand one selected block or kernel-sensitive path. They are used for C++/GGML/CUDA implementation review, parity analysis, and kernel redesign discussion.

Operator 图只展开某个选中的 block 或 kernel 敏感路径，用于 C++/GGML/CUDA 实现审查、parity 分析和 kernel 设计讨论。

Recommended node granularity:

- tensor preparation: reshape, transpose, cast, concat, split
- normalization/modulation: RMSNorm, LayerNorm, AdaLN, timestep modulation
- projection: q/k/v/o projection, MLP up/down/gate projection
- attention: RoPE, mask, flash attention, cross attention
- residual/update: add, residual merge, Euler update
- implementation boundary: ggml op group, custom CUDA kernel, CPU fallback

推荐节点粒度：

- 张量准备：reshape、transpose、cast、concat、split
- 归一化/调制：RMSNorm、LayerNorm、AdaLN、timestep modulation
- 投影：q/k/v/o projection、MLP up/down/gate projection
- 注意力：RoPE、mask、flash attention、cross attention
- 残差/更新：add、residual merge、Euler update
- 实现边界：ggml op group、自定义 CUDA kernel、CPU fallback

## Node ID Convention / 节点 ID 约定

Use stable dotted IDs so a block node can be expanded by an operator graph.

使用稳定的点分 ID，让 block 图里的节点可以被 operator 图展开。

```text
<model>.<path>.<block_or_stage>
<model>.<path>.<block_or_stage>.<operator>
```

Examples / 示例：

```text
hy_vla.action_expert.blocks
hy_vla.action_expert.block_00.self_attn
hy_vla.action_expert.block_00.mlp

lingbot_va.wan.blocks
lingbot_va.wan.block_00.world_self_attn
lingbot_va.wan.block_00.action_cross_attn
```

## Edge Semantics / 连线语义

- solid arrow: main tensor/data flow
- dotted arrow: optional path, cache path, or debug-only path
- labeled arrow: condition, modulation, cross-attention source, or iteration

- 实线箭头：主张量/主数据流
- 点线箭头：可选路径、缓存路径或调试路径
- 带标签箭头：条件输入、调制输入、cross-attention 来源或循环迭代

## Metadata Header / 元数据头

Every new graph should start with a short metadata block before Mermaid.

每个新图建议在 Mermaid 前写一个简短 metadata。

```markdown
---
model: Hy-Embodied-0.5-VLA
graph_type: block
granularity: module/block
scope: full_model
cpp_file: models/hy_vla.cpp
python_ref: Hy-Embodied-0.5-VLA-main
status: draft
---
```

For operator graphs:

```markdown
---
model: Hy-Embodied-0.5-VLA
graph_type: operator
granularity: grouped ggml/cuda ops
scope: hy_vla.action_expert.block_00
expands: hy_vla.action_expert.blocks
cpp_file: models/hy_vla.cpp
status: draft
---
```

## Directory Layout / 目录布局

```text
artifacts/model_graphs/
├── GRAPH_SPEC.md
├── README.md
├── block/       # Layer A block graphs
└── operator/    # Layer B operator graphs
```

