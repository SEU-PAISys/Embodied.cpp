#!/usr/bin/env python3
# Copyright 2026 SEU-PAISys
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a PyTorch reference for the Embodied.cpp Xiaomi-Robotics-0 parity test.

Runs the HuggingFace MiBoT implementation (float32, CPU) on fixed inputs and
writes flat binary files that tools/xr0_parity.cpp consumes / compares
against:

  xr0_parity_inputs.bin   fixed test inputs (tokens + 2 uint8 images +
                          state + noise), layout:
     [int32 n_lang][int32 lang_tokens * n_lang]
     [uint8 pixels 2 * 3 * 256 * 256 (HWC RGB, view-major)]
     [float32 state 32][float32 noise 30*32]
  xr0_parity_ref.bin      float32 actions [30, 32] (world units, i.e. after
                          processor.decode_action)
  xr0_parity_ref_vision.bin
                          float32 [2 * 64 * (4 * 2560)]: llama.cpp clip
                          equivalent = [main | ds0 | ds1 | ds2] per token,
                          view-major, matching the C++ img_feats buffer.
  xr0_parity_ref_kv.bin   float32 K/V of backbone layers 20 and 35 captured
                          after the prefill: per layer [n_seq, n_kv_heads,
                          head_dim] * 2 (K then V), layer 20 then layer 35.

NOTE: The default is --dtype float32 (needs ~20 GiB RAM).  A bfloat16
reference introduces ~1e-2 quantization noise that will fail the 5e-3
tolerance check in parity_xr0_compare.py, so only use it as a RAM-strapped
sanity check, not for the final verification.

Usage (WSL, 20+ GiB RAM):
  python scripts/parity_xr0_reference.py \
      --checkpoint checkpoints/xr0/hf \
      --out-dir   checkpoints/xr0
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# Patch for transformers >= 5.x: the cached MiBoT modeling code uses
# rope_type="default" which no longer exists in ROPE_INIT_FUNCTIONS,
# and the "linear" replacement expects a "factor" key in rope_parameters
# that the old Qwen3VL config does not provide.  We supply a custom
# init that simply returns standard sinusoidal inverse frequencies
# with the model's rope_theta (equivalent to the old "default" path).
import torch
import transformers.modeling_rope_utils as _rope_utils

def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
    base = getattr(config, "rope_theta", 10000.0)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    dim = head_dim
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64,
                                              device=device).float() / dim))
    return inv_freq, 1.0

if "default" not in _rope_utils.ROPE_INIT_FUNCTIONS:
    _rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

sys.path.insert(0, str(REPO_ROOT / "eval" / "client"))
from vla_cpp_client import _xr0_prompt  # noqa: E402

IMG = 256            # LIBERO resolution; 256/32 = 8 -> 8x8 = 64 tokens/view
N_VIEWS = 2
ACTION_DIM = 32
CHUNK = 30
STATE_DIM = 32
DIT_KV_LAYERS = (20, 35)


def _sfx(name: str) -> str:
    return name.replace(".bin", f"_{IMG}.bin")


def main() -> None:
    global IMG
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--img", type=int, default=IMG, help="image side (multiple of 32)")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="float32",
                    help="reference weights dtype. float32 is the ground truth "
                         "used by parity_xr0_compare.py (needs ~20 GiB RAM for "
                         "the 4.7B model). bfloat16 is a low-RAM fallback but "
                         "its ~1e-2 quantization noise will fail the 5e-3 "
                         "tolerance, so it is only useful for smoke checks.")
    ap.add_argument("--cpp-vision", action="store_true",
                    help="replace the ViT outputs with the C++ runtime dump "
                         "(xr0_out_vision_*.bin) so the reference exercises "
                         "backbone+DiT in isolation")
    args = ap.parse_args()
    IMG = args.img

    from transformers import AutoModel, AutoProcessor

    torch.set_grad_enabled(False)

    print(f"loading model from {args.checkpoint} ({args.dtype}, CPU) ...")
    model = AutoModel.from_pretrained(
        str(args.checkpoint), trust_remote_code=True,
        attn_implementation="eager",
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        str(args.checkpoint), trust_remote_code=True, use_fast=False)

    # ---- deterministic inputs ------------------------------------------------
    rng = np.random.default_rng(args.seed)
    images_u8 = [
        rng.integers(0, 256, size=(IMG, IMG, 3), dtype=np.uint8)
        for _ in range(N_VIEWS)
    ]
    language = "Pick up the red block."
    state = rng.standard_normal(STATE_DIM).astype(np.float32)
    pads = (IMG // 32) * (IMG // 32)
    # the HF processor expands one <|image_pad|> per view itself; the C++
    # runtime receives pre-expanded tokens, so build both variants here and
    # verify they tokenize identically.
    prompt_raw = _xr0_prompt(language, N_VIEWS, 1)
    prompt = _xr0_prompt(language, N_VIEWS, pads)

    from PIL import Image
    pil_images = [Image.fromarray(im) for im in images_u8]
    vl_inputs = processor(text=[prompt_raw], images=pil_images,
                          videos=None, padding=True, return_tensors="pt")
    lang = vl_inputs["input_ids"][0].numpy().astype(np.int32)
    lang_expanded = np.asarray(
        __import__("transformers").AutoTokenizer.from_pretrained(
            str(args.checkpoint), use_fast=False)(
                prompt, add_special_tokens=False)["input_ids"],
        dtype=np.int32)
    if lang_expanded.tolist() != lang.tolist():
        raise SystemExit(
            f"token mismatch between processor-expanded and pre-expanded "
            f"prompts: {len(lang)} vs {len(lang_expanded)}")
    print(f"prompt tokens: {len(lang)}  pixel_values: "
          f"{tuple(vl_inputs['pixel_values'].shape)}  "
          f"grid: {vl_inputs['image_grid_thw'].tolist()}")

    torch.manual_seed(args.seed)
    noise = torch.randn(1, CHUNK, ACTION_DIM, dtype=torch.float32)

    inputs = {k: v for k, v in vl_inputs.items()}
    if not args.cpp_vision:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.vlm.visual.dtype)
    inputs["state"] = torch.from_numpy(state).reshape(1, 1, -1).to(model.dtype)
    # The official preprocessor_config stores identical per-timestep stats for
    # libero_all, so get_action_mask() returns a (batch, 10, 32) mask whose rows
    # are all equal.  The GGUF/converter and C++ runtime use chunk=CHUNK (30),
    # so replicate row 0 across CHUNK rows to make the HF policy generate the
    # same 30-step action chunk (and the same redrawn noise) as the C++ side.
    base_mask = processor.get_action_mask("libero_all")  # (1, 10, 32)
    inputs["action_mask"] = base_mask[:, 0:1, :].expand(1, CHUNK, -1).to(model.dtype)
    inputs["num_steps"] = 5
    inputs["seed"] = args.seed  # HF forward re-seeds and redraws the noise

    # ---- reference forward ----------------------------------------------------
    # run the ViT in fp32 (it is only 0.6B params) so vision-side comparisons
    # are not polluted by bf16 accumulation noise
    if not args.cpp_vision:
        model.vlm.visual.to(torch.float32)

    cpp_vision = None
    if args.cpp_vision:
        vpath = out_dir_vision = args.out_dir / (
            "xr0_out_vision_256.bin" if IMG == 256 else
            f"xr0_out_vision_{IMG}.bin")
        per_tok = 4 * 2560
        cpp_vision = torch.from_numpy(
            np.fromfile(vpath, dtype="<f4").reshape(2, -1, per_tok).copy())
        n_tok_cpp = cpp_vision.shape[1]
        orig_gif = model.vlm.model.get_image_features

        def gif_override(pixel_values, image_grid_thw):
            main = [cpp_vision[v][:, k * 2560:(k + 1) * 2560] for v, k in
                    [(v, 0) for v in range(2)]]
            ds = [cpp_vision.reshape(-1, 4 * 2560)[:, (1 + k) * 2560:(2 + k) * 2560]
                  for k in range(3)]
            return main, ds

        model.vlm.model.get_image_features = gif_override
        # free the (now unused) vision tower to fit the fp32 model
        if hasattr(model.vlm.model, "visual"):
            del model.vlm.model.visual
        if args.dtype == "float32":
            model.to(torch.float32)
            # TimestepEmbedder caches a construction-time dtype that does not
            # follow Module.to(); keep it in sync for the fp32 reference
            model.t_embedder.dtype = torch.float32

    kv_capture = {}
    pos_capture = {}
    block_capture = {}
    block_hooks = []

    def make_block_hook(idx: int):
        def hook(mod, inp, out):
            # the ViT runs once per view (cu_seqlens); capture every call
            block_capture.setdefault(idx, []).append(
                out.detach().float().clone().cpu())
        return hook

    if not args.cpp_vision:
        for blk_idx in (5, 11, 17):
            block_hooks.append(model.vlm.visual.blocks[blk_idx].register_forward_hook(
                make_block_hook(blk_idx)))

    text_capture = {}
    text_hooks = []

    emb_capture = {}

    # capture the post-visual-scatter inputs_embeds that Qwen3VLModel.forward
    # passes into language_model.forward (the top-level MiBoT call passes
    # input_ids+pixel_values, not inputs_embeds, so Qwen3VLModel.forward is
    # what performs the masked_scatter; the resulting embeds then flow into
    # language_model.forward as the explicit inputs_embeds kwarg).
    orig_lm_forward = model.vlm.language_model.forward

    def lm_emb_capture(*a, **kw):
        if 0 not in emb_capture and "inputs_embeds" in kw:
            emb_capture[0] = kw["inputs_embeds"].detach().float().clone()
        return orig_lm_forward(*a, **kw)

    model.vlm.language_model.forward = lm_emb_capture

    def make_text_hook(idx: int):
        def hook(mod, inp, out):
            # out is hidden_states [1, seq, hidden] for decoder layer idx
            text_capture.setdefault(idx, []).append(out.detach().float().clone())
        return hook

    for l_idx in (0, 1, 2, 5):
        text_hooks.append(model.vlm.language_model.layers[l_idx].register_forward_hook(
            make_text_hook(l_idx)))

    orig_vlm_forward = model.vlm.forward

    def vlm_forward_with_kv(*a, **kw):
        kw["use_cache"] = True
        out = orig_vlm_forward(*a, **kw)
        pkv = out.past_key_values
        for layer in DIT_KV_LAYERS:
            if hasattr(pkv, "key_cache"):          # transformers <= 4.56
                k, v = pkv.key_cache[layer][0], pkv.value_cache[layer][0]
            else:                                   # transformers >= 4.57
                ent = pkv.layers[layer]
                k, v = ent.keys[0], ent.values[0]
            kv_capture[layer] = (k.float().clone(), v.float().clone())
        if 0 not in pos_capture and out.position_ids is not None:
            pos_capture[0] = out.position_ids.detach().clone()  # [3, 1, seq]
        return out

    # DiT-internal capture: per-layer outputs and the action_output_layer
    # input (pre-projection hidden states of the trailing chunk rows).
    dit_capture = {}
    dit_out_hook = []
    dit_in_capture = {}
    for blk_idx in range(model.dit.config.num_hidden_layers):
        def make_dit_hook(idx: int):
            def hook(mod, inp, out):
                # capture every dit_forward call (5 steps)
                dit_capture.setdefault(idx, []).append(
                    out.detach().float().clone().cpu())
            return hook
        dit_out_hook.append(model.dit.layers[blk_idx].register_forward_hook(
            make_dit_hook(blk_idx)))

    # capture the DiT module input (the concat [sink, state, action] hidden)
    def make_dit_in_hook(mod, inp, out):
        dit_in_capture.setdefault("inp", []).append(
            inp[0].detach().float().clone().cpu())

    dit_in_hook = model.dit.register_forward_hook(make_dit_in_hook)

    orig_dit_forward = model.dit_forward
    dit_input_capture = {}

    # capture action_projector OUTPUT (= a_emb) per step, and sink weight
    a_proj_capture = {}
    orig_a_proj = model.action_projector.forward

    def a_proj_forward(x):
        out = orig_a_proj(x)
        a_proj_capture.setdefault("out", []).append(out.detach().float().clone().cpu())
        return out

    model.action_projector.forward = a_proj_forward

    def dit_forward_with_capture(noisy_action, t, action_mask, state_embed,
                                 position_embeds, past_key_values, attn_mask):
        # capture the pre-projector values (projectors are Module children of
        # the top-level model, not of DiT, so wrap here rather than hooking)
        dit_input_capture.setdefault("state_embed", []).append(
            state_embed.detach().float().clone().cpu())
        dit_input_capture.setdefault("noisy_action", []).append(
            noisy_action.detach().float().clone().cpu())
        return orig_dit_forward(noisy_action, t, action_mask, state_embed,
                                position_embeds, past_key_values, attn_mask)

    model.dit_forward = dit_forward_with_capture

    # capture the action_output_layer input (hidden states of trailing rows)
    aol_input_capture = {}
    aol_input_capture["inp"] = []

    def make_aol_hook(mod, inp, out):
        aol_input_capture["inp"].append(inp[0].detach().float().clone().cpu())

    aol_hook = model.action_output_layer.register_forward_hook(make_aol_hook)

    model.vlm.forward = vlm_forward_with_kv
    if args.cpp_vision:
        print("dtypes:", {k: (str(v.dtype), tuple(v.shape)) for k, v in inputs.items()
                          if hasattr(v, "dtype")},
              "model.dtype:", model.dtype)
    outputs = model(**inputs)
    model.vlm.forward = orig_vlm_forward
    model.vlm.language_model.forward = orig_lm_forward

    # Decode with the canonical (32,) stats that the C++ runtime uses.  All
    # per-timestep rows are identical, and the GGUF applies a plain (32,)
    # mean/std, so decode here exactly like the runtime: a * std[0] + mean[0].
    if hasattr(processor, "action_config"):
        stats = processor.action_config["libero_all"]
        d_mean = torch.tensor(stats["mean"], dtype=torch.float32).reshape(-1, 32)[0]
        d_std  = torch.tensor(stats["std"],  dtype=torch.float32).reshape(-1, 32)[0]
    else:
        mask = inputs["action_mask"][0, 0]  # (32,)
        d_std  = mask      # decode_mask ≈ 1 where active
        d_mean = torch.zeros_like(mask)
    actions_world = (outputs.actions[0].float() * d_std + d_mean).numpy()  # [30, 32]

    # vision features in the C++ clip output layout
    if not args.cpp_vision:
        img_embeds, ds_lists = model.vlm.get_image_features(
            vl_inputs["pixel_values"].to(model.vlm.visual.dtype),
            vl_inputs["image_grid_thw"])
        n_tok = img_embeds[0].shape[0]
        vision = np.zeros((N_VIEWS, n_tok, 4, img_embeds[0].shape[1]), dtype=np.float32)
        for v in range(N_VIEWS):
            vision[v, :, 0, :] = img_embeds[v].float().numpy()
            for k, ds in enumerate(ds_lists):
                # NOTE: unlike image_embeds, deepstack features are NOT split
                # per image by HF; each entry holds all views concatenated
                ds_all = ds.reshape(N_VIEWS, n_tok, -1)
                vision[v, :, 1 + k, :] = ds_all[v].float().numpy()
    else:
        vision = cpp_vision.numpy()

    print("actions[0, :7] (world):", actions_world[0, :7])

    # ---- write artifacts -------------------------------------------------------
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ("xr0_parity_inputs.bin" if IMG == 256 else _sfx("xr0_parity_inputs.bin")), "wb") as f:
        f.write(struct.pack("<i", len(lang)))
        f.write(lang.astype("<i4").tobytes())
        for im in images_u8:
            f.write(np.ascontiguousarray(im).tobytes())
        f.write(state.astype("<f4").tobytes())
        f.write(noise.numpy().reshape(-1).astype("<f4").tobytes())

    with open(out / ("xr0_parity_ref.bin" if IMG == 256 else _sfx("xr0_parity_ref.bin")), "wb") as f:
        f.write(actions_world.astype("<f4").tobytes())

    with open(out / ("xr0_parity_ref_vision.bin" if IMG == 256 else _sfx("xr0_parity_ref_vision.bin")), "wb") as f:
        f.write(vision.reshape(-1).astype("<f4").tobytes())

    with open(out / ("xr0_parity_ref_kv.bin" if IMG == 256 else _sfx("xr0_parity_ref_kv.bin")), "wb") as f:
        for layer in DIT_KV_LAYERS:
            k, v = kv_capture[layer]
            # torch [heads, seq, hd] -> C++ rope layout [hd, heads, seq]
            f.write(k.permute(2, 0, 1).contiguous().float().numpy().astype("<f4").tobytes())
            f.write(v.permute(2, 0, 1).contiguous().float().numpy().astype("<f4").tobytes())

    for h in block_hooks:
        h.remove()
    for h in text_hooks:
        h.remove()
    model.dit_forward = orig_dit_forward
    for h in dit_out_hook:
        h.remove()
    aol_hook.remove()
    dit_in_hook.remove()
    model.action_projector.forward = orig_a_proj
    if a_proj_capture.get("out"):
        with open(out / "xr0_parity_ref_dit_aemb.bin", "wb") as f:
            for blk in a_proj_capture["out"]:
                # torch [1, chunk, hidden] -> [hidden, chunk]
                f.write(blk.squeeze(0).permute(1, 0).contiguous().numpy()
                        .astype("<f4").tobytes())
    # write sink.weight flat [hidden]
    with open(out / "xr0_parity_ref_sink.bin", "wb") as f:
        f.write(model.sink.weight[0].detach().float().numpy().astype("<f4").tobytes())
    if dit_in_capture.get("inp"):
        with open(out / "xr0_parity_ref_dit_in.bin", "wb") as f:
            for blk in dit_in_capture["inp"]:
                # torch [1, 1+1+chunk, hidden] -> [hidden, 1+1+chunk]
                f.write(blk.squeeze(0).permute(1, 0).contiguous().numpy()
                        .astype("<f4").tobytes())
    # DiT-internal reference artifacts (5 steps x per-layer, [hidden, seq])
    for idx, runs in dit_capture.items():
        with open(out / f"xr0_parity_ref_dit_{idx}.bin", "wb") as f:
            for blk in runs:
                # torch [1, seq, hidden] -> ggml [hidden, seq]
                f.write(blk.squeeze(0).permute(1, 0).contiguous().numpy()
                        .astype("<f4").tobytes())
    if "state_embed" in dit_input_capture:
        with open(out / "xr0_parity_ref_dit_state.bin", "wb") as f:
            for blk in dit_input_capture["state_embed"]:
                # torch [1, 1, hidden] -> flat [hidden]
                f.write(blk.squeeze(0).squeeze(0).contiguous().numpy()
                        .astype("<f4").tobytes())
    if "noisy_action" in dit_input_capture:
        with open(out / "xr0_parity_ref_dit_noisy.bin", "wb") as f:
            for blk in dit_input_capture["noisy_action"]:
                # torch [1, chunk, action_dim] -> [action_dim, chunk]
                f.write(blk.squeeze(0).permute(1, 0).contiguous().numpy()
                        .astype("<f4").tobytes())
    if aol_input_capture["inp"]:
        with open(out / "xr0_parity_ref_dit_aol.bin", "wb") as f:
            for blk in aol_input_capture["inp"]:
                # torch [1, chunk, hidden] -> [hidden, chunk]
                f.write(blk.squeeze(0).permute(1, 0).contiguous().numpy()
                        .astype("<f4").tobytes())
    stem = "xr0_parity_ref_block" if IMG == 256 else _sfx("xr0_parity_ref_block")
    for idx, per_view in block_capture.items():
        for view, blk in enumerate(per_view):
            with open(out / f"{stem}_{idx}_v{view}.bin", "wb") as f:
                # torch [n_pos, n_embd] -> ggml layout [n_embd, n_pos]
                f.write(blk.squeeze(0).permute(1, 0).contiguous().numpy()
                        .astype("<f4").tobytes())
    tstem = "xr0_parity_ref_vlmh" if IMG == 256 else _sfx("xr0_parity_ref_vlmh")
    for idx, runs in text_capture.items():
        # keep the main forward (first call); get_image_features does not
        # re-run the language model, so there is exactly one entry
        blk = runs[0]
        with open(out / f"{tstem}_{idx}.bin", "wb") as f:
            f.write(blk.squeeze(0).permute(1, 0).contiguous().numpy()
                    .astype("<f4").tobytes())
    if 0 in emb_capture:
        with open(out / ("xr0_parity_ref_emb.bin" if IMG == 256
                         else _sfx("xr0_parity_ref_emb.bin")), "wb") as f:
            f.write(emb_capture[0].squeeze(0).permute(1, 0).contiguous().numpy()
                    .astype("<f4").tobytes())
    if 0 in pos_capture:
        with open(out / ("xr0_parity_ref_posids.bin" if IMG == 256
                         else _sfx("xr0_parity_ref_posids.bin")), "wb") as f:
            # [3, seq] -> C++ layout: 3 blocks of seq
            f.write(pos_capture[0].squeeze(1).numpy().astype("<i4").tobytes())

    print(f"wrote reference artifacts to {out}")
    print(f"  n_lang={len(lang)} n_tok_view={cpp_vision.shape[1] if args.cpp_vision else n_tok} "
          f"actions_absmax={np.abs(actions_world).max():.4f}")


if __name__ == "__main__":
    main()
