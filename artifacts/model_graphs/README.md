# Model Graphs / 模型图

This directory stores report-ready Mermaid diagrams for VLA and world-action model visualization.

本目录用于保存可插入报告的 Mermaid 图，服务 VLA 和 world-action model 的结构对比、实现分析和 kernel 讨论。

## Convention / 规范

We use a two-layer graph convention. See `GRAPH_SPEC.md` for details.

我们采用双层图谱规范，细节见 `GRAPH_SPEC.md`。

- `block/`: module/block-level graphs for model structure and data flow.
- `operator/`: per-model ONNX folders with full and partitioned operator graphs.

- `block/`：模块/block 粒度图，用于模型结构和数据流对比。
- `operator/`：按模型分目录保存整体和分块 ONNX 算子图。

## Files / 文件

New two-layer graphs:

- `block/hy_vla_block_graph.md`: HY-VLA full-model block graph.
- `block/lingbot_va_block_graph.md`: LingBot-VA full-model block graph.
- `block/openvla_block_graph.md`: OpenVLA action-token generation block graph from Python source.
- `block/gr00t_n1d7_block_graph.md`: Isaac GR00T N1.7 Qwen3-VL + DiT flow block graph from Python source.
- `block/pi0_openpi_block_graph.md`: openpi pi0 block graph from Python source.
- `block/pi05_openpi_block_graph.md`: openpi pi0.5 block graph from Python source.
- `operator/README.md`: operator artifact layout and naming convention.
- `operator/pi0/full_inference_structural.onnx`: experimental structural ONNX for pi0-style full inference.
- `operator/pi0/full_inference_structural_onnx_audit.md`: export command and operator-count audit for the pi0 ONNX graph.

Deprecated flat graphs:

These were early sketches and should not be used as references for the six-model block graph set.

这些是早期草图，不作为当前六模型 block 图体系的参考。

- `hy_vla_module_dag.md`
- `lingbot_va_module_dag.md`
- `vla_vs_world_action.md`

## Preview / 预览

Most Markdown viewers that support Mermaid can render these files directly, including GitHub-style Markdown renderers and many VS Code Mermaid extensions.

多数支持 Mermaid 的 Markdown 预览器可以直接渲染这些文件，包括 GitHub 风格 Markdown 预览和 VS Code Mermaid 插件。

## Export / 导出

Install Mermaid CLI:

```bash
npm install -g @mermaid-js/mermaid-cli
```

Export one diagram to SVG:

```bash
mmdc -i artifacts/model_graphs/block/hy_vla_block_graph.md \
     -o artifacts/model_graphs/block/hy_vla_block_graph.svg
```

Export one diagram to PNG:

```bash
mmdc -i artifacts/model_graphs/block/pi05_openpi_block_graph.md \
     -o artifacts/model_graphs/block/pi05_openpi_block_graph.png \
     -b transparent
```

Batch export all Mermaid Markdown files to SVG:

```bash
find artifacts/model_graphs/block -name '*.md' \
  ! -name 'README.md' \
  ! -name 'GRAPH_SPEC.md' \
  -print | while read -r f; do
  mmdc -i "$f" -o "${f%.md}.svg"
done
```

## Style / 风格

- Block graphs use stable dotted IDs such as `hy_vla.action_expert.blocks`.
- Operator graphs expand one block/path, such as `hy_vla.action_expert.block_00.self_attn`.
- Avoid expanding the full model into primitive add/mul/matmul nodes in one diagram.

- Block 图使用稳定点分 ID，例如 `hy_vla.action_expert.blocks`。
- Operator 图只展开一个 block/路径，例如 `hy_vla.action_expert.block_00.self_attn`。
- 不要把整个模型在一张图里展开成 add/mul/matmul 级别。
