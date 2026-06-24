# LingBot-VA ONNX Operator Audit / LingBot-VA ONNX 算子审计

This report uses small representative ONNX skeletons plus checkpoint metadata. It is not a full 22GB model export.

本文使用小型代表性 ONNX skeleton 和权重 metadata，不是完整 22GB 模型导出。

## Exported ONNX Graphs / 已导出的 ONNX 图

- `wan_rope`: `/home/xuling/robotic_code/embodied.cpp/vla.cpp-main/artifacts/lingbot_va_onnx/wan_rope.onnx`
  ops: Add=1, Concat=1, Constant=6, Gather=2, Mul=4, Reshape=2, Sub=1, Unsqueeze=2
- `wan_attention`: `/home/xuling/robotic_code/embodied.cpp/vla.cpp-main/artifacts/lingbot_va_onnx/wan_attention.onnx`
  ops: Add=8, Cast=2, Concat=2, Constant=21, Div=3, Gather=4, Identity=1, MatMul=6, Mul=16, ReduceMean=2, Reshape=8, Shape=1, Slice=1, Softmax=1, Sqrt=5, Sub=2, Transpose=4, Unsqueeze=4
- `wan_block`: `/home/xuling/robotic_code/embodied.cpp/vla.cpp-main/artifacts/lingbot_va_onnx/wan_block.onnx`
  ops: Add=25, Cast=6, Concat=2, Constant=45, Div=4, Gather=4, Identity=2, LayerNormalization=3, MatMul=15, Mul=28, ReduceMean=2, Reshape=8, Shape=2, Slice=2, Softmax=2, Split=1, Sqrt=8, Squeeze=7, Sub=2, Tanh=1, Transpose=5, Unsqueeze=7
- `flow_step`: `/home/xuling/robotic_code/embodied.cpp/vla.cpp-main/artifacts/lingbot_va_onnx/flow_step.onnx`
  ops: Add=1, Mul=1, Sub=1
- `causal_conv3d_skeleton`: `/home/xuling/robotic_code/embodied.cpp/vla.cpp-main/artifacts/lingbot_va_onnx/causal_conv3d_skeleton.onnx`
  ops: Concat=1, Conv=1, Mul=1, Sigmoid=1

## Combined ONNX Ops / 汇总 ONNX 算子

- `Constant`: 72
- `Mul`: 50
- `Add`: 35
- `MatMul`: 21
- `Reshape`: 18
- `Unsqueeze`: 13
- `Sqrt`: 13
- `Gather`: 10
- `Transpose`: 9
- `Cast`: 8
- `Div`: 7
- `Squeeze`: 7
- `Sub`: 6
- `Concat`: 6
- `ReduceMean`: 4
- `Identity`: 3
- `Shape`: 3
- `Slice`: 3
- `Softmax`: 3
- `LayerNormalization`: 3
- `Split`: 1
- `Tanh`: 1
- `Conv`: 1
- `Sigmoid`: 1

## Checkpoint Metadata / 权重元数据

- Transformer: layers=30, hidden=3072, ffn=14336, attn_mode=flex
- VAE: class=AutoencoderKLWan, base_dim=160, z_dim=48, patch_size=2, temporal_downsample=[False, True, True]
- UMT5: layers=24, d_model=4096, d_ff=10240, heads=64, buckets=32
- Transformer tensor top prefixes: {'blocks': 810, 'condition_embedder': 10, 'condition_embedder_action': 10, 'action_embedder': 2, 'action_proj_out': 2, 'patch_embedding': 2, 'patch_embedding_mlp': 2, 'proj_out': 2}
- Text encoder tensor top prefixes: {'encoder': 241, 'shared': 1}
- VAE tensor top prefixes: {'decoder': 108, 'encoder': 84, 'post_quant_conv': 2, 'quant_conv': 2}

## Portability Classification / 可移植性分类

| Module / 模块 | First port strategy / 第一版策略 | Notes / 说明 |
|---|---|---|
| WanTransformer linear / norm / FFN | ggml composition | MatMul/Add/LayerNorm/RMSNorm/GELU/Silu are already available or easy to compose. |
| Wan self-attention and cross-attention | ggml composition first | Use ggml_flash_attn_ext + masks/cache. Custom windowed attention may be useful later. |
| Wan 3D RoPE | ggml composition/custom helper | We already implemented C++ real-op RoPE composition; no new backend op required for first version. |
| FlowMatchScheduler step | host/C++ + ggml simple ops | sample + velocity * delta_sigma; no custom op needed. |
| UMT5 encoder | ggml composition, high memory risk | Transformer encoder with relative position bias and gated GELU. Quantization/loading strategy matters more than new ops. |
| VAE conv/resnet/up/downsample | ggml composition possible | Conv3D/GroupNorm/SiLU/residual blocks likely map to ggml, but implementation is sizable. |
| WanCausalConv3d streaming cache | possible custom kernel later | Can be emulated by concat+conv first; high-performance streaming causal conv may deserve a specialized path. |
| Flex/block-sparse attention mask | possible custom kernel later | Training flex mask is complex; inference cache path may use dense attention first. Efficient long-window world-model attention is a strong custom-kernel candidate. |

## Netron / 可视化

Open any `.onnx` file in this directory with Netron, for example:

```bash
netron /home/xuling/robotic_code/embodied.cpp/vla.cpp-main/artifacts/lingbot_va_onnx/wan_block.onnx
```

可以用 Netron 打开本目录下任意 `.onnx` 文件查看算子图。
