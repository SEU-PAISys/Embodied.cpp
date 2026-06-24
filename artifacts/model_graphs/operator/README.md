# Operator Graph Artifacts / 算子图产物

This directory stores model-level and partition-level ONNX graphs.

本目录保存模型整体和模块分块的 ONNX 算子图。

## Layout / 目录结构

Each model owns one subdirectory.

每个模型一个子目录。

```text
operator/
├── pi0/
├── pi05/
├── openvla/
├── gr00t_n1d7/
├── hy_vla/
└── lingbot_va/
```

## Naming / 命名

Use one full graph plus several partition graphs.

每个模型保留一个整体图，并按主要模块拆分若干分块图。

```text
full_inference.onnx
full_inference_onnx_audit.md

01_prefix_or_vision.onnx
01_prefix_or_vision_onnx_audit.md

02_action_or_policy.onnx
02_action_or_policy_onnx_audit.md

03_postprocess_or_flow.onnx
03_postprocess_or_flow_onnx_audit.md
```

The exact partition names can follow the model structure.

具体分块名可以按模型结构调整。

Examples / 示例：

```text
pi0/
  full_inference_structural.onnx
  01_prefix_prefill.onnx
  02_action_expert_step.onnx
  03_euler_loop.onnx
  partitioned_onnx_audit.md
  01_image_encoder_original_random.onnx
  02_language_embed_original_random.onnx
  03_prefix_pack_original_random.onnx
  04_prefix_prefill_original_random.onnx
  05_suffix_embed_original_random.onnx
  original_random_onnx_audit.md

lingbot_va/
  full_inference.onnx
  01_umt5_text_encoder.onnx
  02_vae_encoder.onnx
  03_wan_latent_stack.onnx
  04_wan_action_stack.onnx
  05_action_postprocess.onnx
```

## Purpose / 用途

- Full ONNX preserves the complete data-flow source of truth.
- Partition ONNX files improve readability for module-level operator analysis.
- Audit files record export commands, input/output shapes, and operator counts.

- 整体 ONNX 保留完整数据流事实底稿。
- 分块 ONNX 提升可读性，适合模块级算子分析。
- 审计文件记录导出命令、输入输出形状和算子统计。
