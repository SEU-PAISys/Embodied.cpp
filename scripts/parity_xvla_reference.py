#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""X-VLA parity reference: pure-PyTorch replay of the WidowX checkpoint.

Deliberately avoids transformers model classes (their 5.x API breaks the
vendored Florence-2 code); every weight is read straight from
model.safetensors and the forward pass re-implements modeling_xvla.py /
transformer.py / the DaViT half of modeling_florence2.py in fp32.

Modes:
  python parity_xvla_reference.py --hf-dir DIR --out-dir DIR      # generate
  python parity_xvla_reference.py --hf-dir DIR --out-dir DIR --compare out.bin
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

IMG = 224
ACTIONS = 30
DIM = 20
TEXT_LEN = 50
STEPS = 10
GRIP = (9, 19)


# ------------------------------- DaViT --------------------------------------
def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows, B, ws, H, W):
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class Dwt:  # weight dict accessor with prefix
    def __init__(self, sd, prefix):
        self.sd = sd
        self.p = prefix

    def t(self, suffix):
        return torch.from_numpy(self.sd[self.p + "." + suffix]).float()


def dwconv2d(x, w, b):
    """x [B, C, H, W], depthwise 3x3 stride1 pad1."""
    return F.conv2d(x, w, b, padding=1, groups=w.shape[0])


def spatial_attn(x, size, w: Dwt, heads, ws):
    """x [B, L, C] -> [B, L, C]."""
    B, L, C = x.shape
    H, W = size
    h = x.view(B, H, W, C)
    pad_r = (ws - W % ws) % ws
    pad_b = (ws - H % ws) % ws
    h = F.pad(h, (0, 0, 0, pad_r, 0, pad_b))
    Hp, Wp = h.shape[1], h.shape[2]
    hw = window_partition(h, ws).view(-1, ws * ws, C)
    qkv = F.linear(hw, w.t("qkv.weight"), w.t("qkv.bias"))
    q, k, v = qkv.reshape(hw.shape[0], hw.shape[1], 3, heads, C // heads).permute(2, 0, 3, 1, 4).unbind(0)
    o = F.scaled_dot_product_attention(q, k, v)
    o = o.transpose(1, 2).reshape(hw.shape[0], ws * ws, C)
    o = F.linear(o, w.t("proj.weight"), w.t("proj.bias"))
    o = window_reverse(o.view(-1, ws, ws, C), B, ws, Hp, Wp)
    if pad_r or pad_b:
        o = o[:, :H, :W, :]
    return o.reshape(B, L, C)


def channel_attn(x, size, w: Dwt, groups):
    B, L, C = x.shape
    dg = C // groups
    qkv = F.linear(x, w.t("qkv.weight"), w.t("qkv.bias"))
    q, k, v = qkv.reshape(B, L, 3, groups, dg).permute(2, 0, 3, 1, 4)  # [B,G,L,dg]
    q = q * (float(L) ** -0.5)
    attn = q.transpose(-1, -2) @ k          # [B,G,dg,dg]
    attn = attn.softmax(dim=-1)
    o = (attn @ v.transpose(-1, -2)).transpose(-1, -2)  # [B,G,L,dg]
    o = o.transpose(1, 2).reshape(B, L, C)
    return F.linear(o, w.t("proj.weight"), w.t("proj.bias"))


def davit_block(x, size, w: Dwt, kind, cfg, tag=None):
    import os
    dd = os.environ.get("VLA_XVLA_DUMP_DIR")
    attn = "window_attn" if kind == "spatial_block" else "channel_attn"
    B, L, C = x.shape
    h = x.view(B, size[0], size[1], C).permute(0, 3, 1, 2)
    h = dwconv2d(h, w.t("conv1.fn.dw.weight"), w.t("conv1.fn.dw.bias"))
    # PreNorm(None, DepthWiseConv2d): residual dwconv.
    x = x + h.flatten(2).transpose(1, 2)

    if dd and tag:
        x.detach().numpy().tofile(f"{dd}/xvla_cpp_dwc1_{tag[:2]}.f32")
    n = F.layer_norm(x, (C,), w.t(f"{attn}.norm.weight"), w.t(f"{attn}.norm.bias"))
    if dd and tag:
        n.detach().numpy().tofile(f"{dd}/xvla_cpp_ln1_{tag[:2]}.f32")
    inner = Dwt(w.sd, f"{w.p}.{attn}.fn")
    if kind == "spatial_block":
        a = spatial_attn(n, size, inner, cfg["heads"], cfg["window"])
    else:
        a = channel_attn(n, size, inner, cfg["groups"])
    if dd and tag:
        a.detach().numpy().tofile(f"{dd}/xvla_cpp_attnout_{tag}.f32")
    x = x + a
    if dd and tag:
        x.detach().numpy().tofile(f"{dd}/xvla_cpp_blkdir_{tag}.f32")

    h = x.view(B, size[0], size[1], C).permute(0, 3, 1, 2)
    h = dwconv2d(h, w.t("conv2.fn.dw.weight"), w.t("conv2.fn.dw.bias"))
    # PreNorm(None, DepthWiseConv2d): residual dwconv.
    x = x + h.flatten(2).transpose(1, 2)

    n = F.layer_norm(x, (C,), w.t("ffn.norm.weight"), w.t("ffn.norm.bias"))
    f = F.linear(n, w.t("ffn.fn.net.fc1.weight"), w.t("ffn.fn.net.fc1.bias"))
    f = F.gelu(f)
    f = F.linear(f, w.t("ffn.fn.net.fc2.weight"), w.t("ffn.fn.net.fc2.bias"))
    return x + f


def davit(sd, pixel_values, cfg):
    """pixel_values [B, 3, H, W] -> tokens [B, P, 2048]."""
    x = pixel_values
    size = (x.shape[2], x.shape[3])
    B = x.shape[0]
    for s in range(4):
        cw = Dwt(sd, f"vlm.vision_tower.convs.{s}")
        if s == 0:
            img = x
            if cfg["prenorm"][s]:
                img = F.layer_norm(img.transpose(1, 3), (img.shape[1],),
                                   cw.t("norm.weight"), cw.t("norm.bias")).transpose(1, 3)
        else:
            # x holds tokens [B, L, C_prev]; optional pre-norm, then to image
            t = x
            if cfg["prenorm"][s]:
                t = F.layer_norm(t, (t.shape[-1],), cw.t("norm.weight"), cw.t("norm.bias"))
            img = t.reshape(B, size[0], size[1], -1).permute(0, 3, 1, 2)
        x = F.conv2d(img, cw.t("proj.weight"), cw.t("proj.bias"),
                     stride=cfg["stride"][s], padding=cfg["padding"][s])
        size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        import os
        if os.environ.get("VLA_XVLA_DUMP_DIR"):
            x.detach().numpy().tofile(
                os.environ["VLA_XVLA_DUMP_DIR"] + f"/xvla_cpp_preln{s}.f32")
        if not cfg["prenorm"][s]:
            x = F.layer_norm(x, (x.shape[-1],), cw.t("norm.weight"), cw.t("norm.bias"))
        import os
        if os.environ.get("VLA_XVLA_DUMP_DIR"):
            x.detach().numpy().tofile(
                os.environ["VLA_XVLA_DUMP_DIR"] + f"/xvla_cpp_preblk{s}.f32")
        for j in range(cfg["depths"][s]):
            for kind in ("spatial_block", "channel_block"):
                bw = Dwt(sd, f"vlm.vision_tower.blocks.{s}.{j}.{kind}")
                sub = {"heads": cfg["heads_s"][s], "groups": cfg["groups"][s],
                       "window": cfg["window"]}
                tag = f"{kind[:2]}{s}" if (s == 0) else None
                x = davit_block(x, size, bw, kind, sub, tag=tag)
        if os.environ.get("VLA_XVLA_DUMP_DIR"):
            x.detach().numpy().tofile(
                os.environ["VLA_XVLA_DUMP_DIR"] + f"/xvla_cpp_stage{s}.f32")
    return x


def encode_image(sd, pixel_values, cfg):
    """[B, 3, 224, 224] -> [B, 50, 1024]."""
    import os
    dd = os.environ.get("VLA_XVLA_DUMP_DIR")
    x = davit(sd, pixel_values, cfg)                       # [B, P, 2048]
    B, P, D = x.shape
    h = w = int(math.sqrt(P))
    x = x.view(B, h, w, D)
    col = torch.from_numpy(sd["vlm.image_pos_embed.column_embeddings.weight"]).float()
    row = torch.from_numpy(sd["vlm.image_pos_embed.row_embeddings.weight"]).float()
    pe = torch.cat([col[:w].unsqueeze(0).repeat(h, 1, 1),
                    row[:h].unsqueeze(1).repeat(1, w, 1)], dim=-1)  # [h,w,D]
    x = x + pe.unsqueeze(0)
    if dd:
        x.detach().numpy().tofile(dd + "/xvla_cpp_tokpe.f32")
    temporal = torch.from_numpy(sd["vlm.visual_temporal_embed.pos_idx_to_embed"]).float()[0]
    x = x + temporal
    if dd:
        x.detach().numpy().tofile(dd + "/xvla_cpp_toktmp.f32")
    spatial = x.mean(dim=(1, 2))                           # [B,D]
    feats = torch.cat([spatial.unsqueeze(1), x.reshape(B, P, D)], dim=1)  # [B,1+P,D]
    if dd:
        spatial.detach().numpy().tofile(dd + "/xvla_cpp_spmean.f32")
        feats.detach().numpy().tofile(dd + "/xvla_cpp_vfeats.f32")
    proj = torch.from_numpy(sd["vlm.image_projection"]).float()
    x = feats @ proj
    return F.layer_norm(x, (x.shape[-1],),
                        torch.from_numpy(sd["vlm.image_proj_norm.weight"]).float(),
                        torch.from_numpy(sd["vlm.image_proj_norm.bias"]).float())


def bart_encode(sd, merged, pos_emb):
    """merged [B, T, D] through the 12 post-LN encoder layers.

    The official encoder adds learned positions (id = index + 2) over the
    WHOLE merged sequence before layernorm_embedding; `merged` must arrive
    WITHOUT any positional embedding yet.
    """
    merged = merged + pos_emb[2:2 + merged.shape[1]].unsqueeze(0)
    x = F.layer_norm(merged, (merged.shape[-1],),
                     torch.from_numpy(sd["vlm.language_model.model.encoder.layernorm_embedding.weight"]).float(),
                     torch.from_numpy(sd["vlm.language_model.model.encoder.layernorm_embedding.bias"]).float())
    for i in range(12):
        p = f"vlm.language_model.model.encoder.layers.{i}"
        def lin(t, sfx):
            return torch.from_numpy(sd[f"{p}.{t}{sfx}"]).float()
        q = F.linear(x, lin("self_attn.q_proj", ".weight"), lin("self_attn.q_proj", ".bias"))
        k = F.linear(x, lin("self_attn.k_proj", ".weight"), lin("self_attn.k_proj", ".bias"))
        v = F.linear(x, lin("self_attn.v_proj", ".weight"), lin("self_attn.v_proj", ".bias"))
        heads, hd = 16, x.shape[-1] // 16
        q = q.view(q.shape[0], q.shape[1], heads, hd).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], heads, hd).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], heads, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        o = F.linear(o, lin("self_attn.out_proj", ".weight"), lin("self_attn.out_proj", ".bias"))
        x = F.layer_norm(x + o, (x.shape[-1],),
                         lin("self_attn_layer_norm", ".weight"), lin("self_attn_layer_norm", ".bias"))
        f = F.gelu(F.linear(x, lin("fc1", ".weight"), lin("fc1", ".bias")))
        f = F.linear(f, lin("fc2", ".weight"), lin("fc2", ".bias"))
        x = F.layer_norm(x + f, (x.shape[-1],),
                         lin("final_layer_norm", ".weight"), lin("final_layer_norm", ".bias"))
    return x


def timestep_embedding(t, dim, max_period=100):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(0, half, dtype=t.dtype) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DomainLinear(torch.nn.Module):
    def __init__(self, sd, prefix):
        super().__init__()
        self.fc = torch.from_numpy(sd[f"{prefix}.fc.weight"]).float()        # [D, in*out]
        self.bias = torch.from_numpy(sd[f"{prefix}.bias.weight"]).float()    # [D, out]

    def forward(self, x, domain_id):
        rows = self.fc[domain_id]                                            # [B, in*out]
        b = self.bias[domain_id]                                             # [B, out]
        # Official DomainAwareLinear views the flat row directly as
        # [input_size, output_size] (in-major), NO transpose.
        Wm = rows.view(-1, x.shape[-1], b.shape[-1])
        return torch.matmul(x, Wm) + b.unsqueeze(1)


def action_transformer(sd, domain_id, vlm_features, aux_visual, x_t, proprio, t, cfg,
                       aux_pad_slots=0):
    H = cfg["hidden"]
    aenc = DomainLinear(sd, "transformer.action_encoder")
    adec = DomainLinear(sd, "transformer.action_decoder")
    time_emb = timestep_embedding(t, cfg["dim_time"])                        # [B, 32]
    B, T, _ = x_t.shape
    time_tokens = time_emb.unsqueeze(1).expand(B, T, cfg["dim_time"])
    proprio_tokens = proprio.unsqueeze(1).expand(B, T, proprio.shape[-1])
    action_tokens = torch.cat([x_t, proprio_tokens, time_tokens], dim=-1)
    x = aenc(action_tokens, domain_id)                                       # [B, T, H]
    vlm_proj_w = torch.from_numpy(sd["transformer.vlm_proj.weight"]).float()
    vlm_proj_b = torch.from_numpy(sd["transformer.vlm_proj.bias"]).float()
    aux_proj_w = torch.from_numpy(sd["transformer.aux_visual_proj.weight"]).float()
    aux_proj_b = torch.from_numpy(sd["transformer.aux_visual_proj.bias"]).float()
    # Valid aux views go through the projection; masked-out pad slots keep
    # zero features whose projection is exactly the bias term.
    aux_proj = F.linear(aux_visual, aux_proj_w, aux_proj_b)
    if aux_pad_slots > 0:
        blk = aux_proj_b.view(1, 1, -1).repeat(1, aux_visual.shape[1], 1)
        aux_proj = torch.cat([aux_proj] + [blk] * aux_pad_slots, dim=1)
    x = torch.cat([x, F.linear(vlm_features, vlm_proj_w, vlm_proj_b),
                   aux_proj], dim=1)
    seq_len = x.shape[1]
    pos = torch.from_numpy(sd["transformer.pos_emb"]).float()[0, :seq_len]
    x = x + pos
    prompts = torch.from_numpy(sd["transformer.soft_prompt_hub.weight"]).float()
    sp = prompts[domain_id].view(-1, cfg["soft_prompts"], H)
    x = torch.cat([x, sp], dim=1)
    for i in range(cfg["depth"]):
        p = f"transformer.blocks.{i}"
        def lin(sfx):
            return torch.from_numpy(sd[f"{p}.{sfx}"]).float()
        h = F.layer_norm(x, (H,), lin("norm1.weight"), lin("norm1.bias"))
        qkv = F.linear(h, lin("attn.qkv.weight"), lin("attn.qkv.bias"))
        heads, hd = cfg["num_heads"], H // cfg["num_heads"]
        q, k, v = qkv.reshape(x.shape[0], x.shape[1], 3, heads, hd).permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(x.shape[0], x.shape[1], H)
        o = F.linear(o, lin("attn.proj.weight"), lin("attn.proj.bias"))
        x = x + o
        h = F.layer_norm(x, (H,), lin("norm2.weight"), lin("norm2.bias"))
        f = F.linear(h, lin("mlp.fc1.weight"), lin("mlp.fc1.bias"))
        f = F.gelu(f, approximate="tanh")
        f = F.linear(f, lin("mlp.fc2.weight"), lin("mlp.fc2.bias"))
        x = x + f
    normed = F.layer_norm(x[:, :T], (H,),
                          torch.from_numpy(sd["transformer.norm.weight"]).float(),
                          torch.from_numpy(sd["transformer.norm.bias"]).float())
    return adec(normed, domain_id)


def generate_actions(sd, inputs, cfg):
    tok_emb = torch.from_numpy(sd["vlm.language_model.model.shared.weight"]).float()
    pos_emb = torch.from_numpy(sd["vlm.language_model.model.encoder.embed_positions.weight"]).float()

    ids = torch.from_numpy(inputs["lang"]).long().unsqueeze(0)
    txt = tok_emb[ids]
    pixels = torch.from_numpy(inputs["pixels"]).float()                      # [V,H,W,3]
    # CLIPImageProcessor: rescale 1/255 then ImageNet mean/std normalize
    pixels = pixels / 255.0
    pixels = (pixels - torch.tensor([0.485, 0.456, 0.406])) / \
             torch.tensor([0.229, 0.224, 0.225])
    pixels = pixels.permute(0, 3, 1, 2)
    img_feats = encode_image(sd, pixels, cfg)                                # [V,50,1024]
    merged = torch.cat([img_feats[:1], txt], dim=1)
    enc = bart_encode(sd, merged, pos_emb)
    aux = img_feats[1:]
    # The HF processor pads views to num_views=3; masked-out view slots keep
    # zero features whose projection is exactly the bias term. Pass the pad
    # count through and let action_transformer append bias-only blocks AFTER
    # the linear projection.
    P = img_feats.shape[1]
    aux = aux.reshape(1, -1, img_feats.shape[-1])
    n_pad_slots = 2 - aux.shape[1] // P

    import os
    dd = os.environ.get("VLA_XVLA_DUMP_DIR")
    if dd:
        img_feats.reshape(-1).numpy().tofile(f"{dd}/xvla_cpp_img_feats.f32")
        enc.reshape(-1).numpy().tofile(f"{dd}/xvla_cpp_enc_out.f32")

    proprio = torch.from_numpy(inputs["proprio"]).float().unsqueeze(0)
    noise = torch.from_numpy(inputs["noise"]).float().unsqueeze(0)
    did = torch.tensor([inputs["domain_id"]], dtype=torch.long)

    pm = proprio.clone()
    pm[..., GRIP[0]] = 0.0
    pm[..., GRIP[1]] = 0.0

    action = torch.zeros_like(noise)
    for i in range(STEPS, 0, -1):
        t = torch.full((1,), i / STEPS, dtype=torch.float32)
        x_t = noise * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        x_t[..., GRIP[0]] = 0.0
        x_t[..., GRIP[1]] = 0.0
        action = action_transformer(sd, did, enc, aux, x_t, pm, t, cfg,
                                    aux_pad_slots=max(0, n_pad_slots))
    action[..., GRIP[0]] = torch.sigmoid(action[..., GRIP[0]])
    action[..., GRIP[1]] = torch.sigmoid(action[..., GRIP[1]])
    return action[0].numpy()


def build_inputs(out_dir: Path, views: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(views, IMG, IMG, 3), dtype=np.uint8)
    lang = rng.integers(10, 50000, size=(TEXT_LEN,), dtype=np.int64)
    lang[0] = 0          # <s>
    lang[-1] = 2         # </s>
    lang[1:-1][15:] = 1  # pads after 14 real tokens
    state = rng.normal(0, 0.1, size=(DIM,)).astype(np.float32)
    noise = rng.normal(0, 1, size=(ACTIONS, DIM)).astype(np.float32)
    domain_id = 5

    bin_path = out_dir / "xvla_parity_inputs.bin"
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<iii", views, domain_id, TEXT_LEN))
        f.write(lang.astype(np.int32).tobytes())
        f.write(pixels.tobytes())
        f.write(state.astype(np.float32).tobytes())
        f.write(noise.astype(np.float32).tobytes())
    print(f"wrote {bin_path}")
    return {
        "pixels": pixels,
        "lang": lang.astype(np.int64),
        "proprio": state,
        "noise": noise,
        "domain_id": domain_id,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--compare", type=Path, default=None,
                    help="path to xvla_parity_out.bin from the C++ harness")
    ap.add_argument("--atol", type=float, default=0.02)
    args = ap.parse_args()

    from safetensors.numpy import load_file
    sd = load_file(str(args.hf_dir / "model.safetensors"))

    import json
    mcfg = json.loads((args.hf_dir / "config.json").read_text(encoding="utf-8"))
    fcfg = mcfg["florence_config"]["vision_config"]
    cfg = {
        "depths": fcfg["depths"],
        "stride": fcfg["patch_stride"],
        "padding": fcfg["patch_padding"],
        "prenorm": fcfg["patch_prenorm"],
        "heads_s": fcfg["num_heads"],
        "groups": fcfg["num_groups"],
        "window": fcfg["window_size"],
        "hidden": mcfg["hidden_size"],
        "depth": mcfg["depth"],
        "num_heads": mcfg["num_heads"],
        "dim_time": mcfg["dim_time"],
        "soft_prompts": mcfg["len_soft_prompts"],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs(args.out_dir, views=args.views)
    ref = generate_actions(sd, inputs, cfg)
    np.save(args.out_dir / "xvla_parity_ref.npy", ref)
    print(f"reference shape={ref.shape} first8={ref.reshape(-1)[:8].tolist()}")

    if args.compare:
        cpp = np.fromfile(args.compare, dtype="<f4").reshape(ref.shape)
        delta = np.abs(cpp - ref)
        worst = tuple(int(i) for i in np.unravel_index(int(delta.argmax()), delta.shape))
        print(f"max_abs={delta.max():.8f} mean_abs={delta.mean():.8f} worst={worst}")
        print(f"pytorch[0][:8]={ref.reshape(-1)[:8].tolist()}")
        print(f"cpp[0][:8]={cpp.reshape(-1)[:8].tolist()}")
        if not np.isfinite(cpp).all() or float(delta.max()) > args.atol:
            raise SystemExit(f"FAIL: expected max_abs <= {args.atol}")
        print(f"PASS: max_abs <= {args.atol}")


if __name__ == "__main__":
    main()
