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

"""Convert Xiaomi-Robotics-0 (MiBoT) checkpoints to GGUF for Embodied.cpp.

Xiaomi-Robotics-0 = Qwen3-VL-4B-Instruct backbone + 16-layer DiT flow-matching
action head that cross-attends directly to the backbone KV cache of layers
20..35 (start_index = 36 - 16).  This script writes TWO files:

  xr0-mmproj.gguf -- Qwen3-VL vision tower in llama.cpp mmproj format, so the
                     C++ runtime can reuse llama.cpp's clip implementation
                     (conv3d-split patch embed, absolute-position-embedding
                     interpolation, vision MRoPE, patch merger, and the three
                     deepstack mergers).  It is produced by llama.cpp's own
                     convert_hf_to_gguf.py, driven through a temporary
                     "HF view" of the checkpoint:

                       - config.json rewritten so the top level looks like a
                         plain Qwen3VLForConditionalGeneration model;
                       - model.safetensors.index.json keys remapped
                         vlm.model.* -> model.* (vision side only);
                       - weight files hard-linked (no copy).

  xr0.gguf        -- text backbone + DiT head + action statistics, in the
                     Embodied.cpp private namespace ("xr0.*" KVs, "vlm.*" /
                     "dit.*" tensors), loaded by models/xr0.cpp directly
                     (same pattern as pi0.5 / HY-VLA).

Reference upstream tensor names (XiaomiRobotics/Xiaomi-Robotics-0-*):

  vlm.model.language_model.embed_tokens.weight        [151936, 2560]
  vlm.model.language_model.layers.{i}.input_layernorm.weight
  vlm.model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
  vlm.model.language_model.layers.{i}.self_attn.{q,k}_norm.weight   [128]
  vlm.model.language_model.layers.{i}.post_attention_layernorm.weight
  vlm.model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight
  vlm.model.visual.*                                  (-> mmproj)

  dit.layers.{i}.adaln_table                          [6, 1024]
  dit.layers.{i}.attn.qkv_proj.{weight,bias}
  dit.layers.{i}.attn.{q,k}_norm.weight               [128]
  dit.layers.{i}.attn.o_proj.weight
  dit.layers.{i}.mlp.{gate,up,down}_proj.weight
  dit.layers.{i}.{input,middle,post,final}_layernorm.weight
  dit.sink.weight                                      [1, 1024]
  state_projector.layers.{0,2}.weight                 ([1024,32],[1024,1024])
  action_projector.layers.{0,2}.weight
  action_output_layer.layers.{0,2}.weight             ([32,1024],[32,32])
  t_embedder.mlp.{0,2}.weight                          ([1024,256],[1024,1024])
  t_projector.layers.0.{weight,bias}                   ([6144,1024],[6144])

Action normalisation statistics come from preprocessor_config.json
action_config.<robot>.{mean,std} (mean/std per padded action dim, mask =
std > 1e-5); decoding is a * std + mean (processing_mibot.MiBotProcessor).

Usage (inside WSL / a venv with torch + gguf installed):

  python scripts/convert_xr0_to_gguf.py \
      --checkpoint /path/to/Xiaomi-Robotics-0-LIBERO \
      --output checkpoints/xr0/xr0.gguf \
      --mmproj  checkpoints/xr0/xr0-mmproj.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf

REPO_ROOT = Path(__file__).resolve().parents[1]
LLAMA_CPP = REPO_ROOT / "third_party" / "llama.cpp"

ARCH = "xr0"

VLM_PREFIX = "vlm.model.language_model"
DIT_PREFIX = "dit"


def _bf16_to_u16_bytes(t: torch.Tensor) -> np.ndarray:
    assert t.element_size() == 2, "BF16 storage requires a 2-byte tensor"
    return t.view(torch.uint16).contiguous().cpu().numpy()


def _f32_np(t: torch.Tensor) -> np.ndarray:
    if t.dtype == torch.bfloat16:      # numpy cannot reinterpre bf16
        t = t.to(torch.float32)
    return t.contiguous().cpu().numpy().astype(np.float32, copy=False)


def _add_tensor(writer: gguf.GGUFWriter, name: str, t: torch.Tensor,
                dtype: str = "bf16") -> None:
    t = t.contiguous()
    if dtype == "f32":
        writer.add_tensor(name, _f32_np(t), raw_dtype=gguf.GGMLQuantizationType.F32)
    else:
        if t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        writer.add_tensor(name, _bf16_to_u16_bytes(t),
                          raw_dtype=gguf.GGMLQuantizationType.BF16)


class SafetensorsSet:
    """Multi-file safetensors reader with lazy per-tensor loading."""

    def __init__(self, ckpt_dir: Path):
        self.ckpt_dir = ckpt_dir
        index_path = ckpt_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            self.weight_map = index["weight_map"]
        else:
            single = ckpt_dir / "model.safetensors"
            if not single.exists():
                raise SystemExit(f"no safetensors found in {ckpt_dir}")
            self.weight_map = None
            self._single = str(single)
        self._handles: dict[str, safe_open] = {}

    def keys(self) -> set[str]:
        if self.weight_map is not None:
            return set(self.weight_map)
        with safe_open(self._single, framework="pt", device="cpu") as f:
            return set(f.keys())

    def _open(self, fname: str) -> safe_open:
        if fname not in self._handles:
            self._handles[fname] = safe_open(
                str(self.ckpt_dir / fname), framework="pt", device="cpu")
        return self._handles[fname]

    def get_tensor(self, name: str) -> torch.Tensor:
        fname = self.weight_map[name] if self.weight_map is not None else \
            os.path.basename(self._single)
        return self._open(fname).get_tensor(name)

    def close(self) -> None:
        self._handles.clear()


MMPROJ_DRIVER = '''
import sys
from pathlib import Path

sys.path.insert(0, {llama_cpp!r})
import convert_hf_to_gguf as ch


@ch.ModelBase.register("MiBoTForActionGeneration")
class MiBoTVisionModel(ch.Qwen3VLVisionModel):
    """Qwen3-VL vision tower of Xiaomi-Robotics-0 (vlm.model.visual.*)."""

    def modify_tensors(self, data_torch, name, bid):
        if not name.startswith("vlm.model.visual."):
            return ()
        name = name.replace("vlm.model.visual.", "model.visual.", 1)
        return super().modify_tensors(data_torch, name, bid)


sys.argv = [sys.argv[0]] + {argv!r}
ch.main()
'''


def _write_hf_view(ckpt_dir: Path, view_dir: Path) -> None:
    """Materialise a Qwen3VL view of MiBoT for the mmproj driver.

    The safetensors keep their original tensor names (vlm.model.*), so the
    weight index is copied verbatim; the driver class strips the prefix.
    """
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    vlm_cfg = cfg["vlm_config"]

    view_cfg = dict(vlm_cfg)
    view_cfg["architectures"] = ["MiBoTForActionGeneration"]
    view_cfg["model_type"] = "qwen3_vl"
    (view_dir / "config.json").write_text(json.dumps(view_cfg, indent=2))

    for fname in ("preprocessor_config.json", "processor_config.json",
                  "model.safetensors.index.json"):
        src = ckpt_dir / fname
        if src.exists():
            (view_dir / fname).write_bytes(src.read_bytes())

    if (ckpt_dir / "model.safetensors.index.json").exists():
        index = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())
        weight_files = sorted(set(index["weight_map"].values()))
    else:
        weight_files = ["model.safetensors"]

    for fname in weight_files:
        src = ckpt_dir / fname
        dst = view_dir / fname
        if dst.exists():
            continue
        try:
            os.link(src, dst)          # hard link: free, same volume
        except OSError:
            import shutil
            shutil.copy2(src, dst)     # fallback across volumes


def _build_mmproj(ckpt_dir: Path, out_path: Path, outtype: str) -> None:
    converter = LLAMA_CPP / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise SystemExit(f"llama.cpp converter not found at {converter}")
    with tempfile.TemporaryDirectory(prefix="xr0-hf-view-") as tmp:
        view_dir = Path(tmp) / "view"
        view_dir.mkdir()
        _write_hf_view(ckpt_dir, view_dir)
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        argv = [str(view_dir), "--mmproj", "--outtype", outtype,
                "--outfile", str(out_path)]
        driver = Path(tmp) / "mibot_mmproj_driver.py"
        driver.write_text(MMPROJ_DRIVER.format(
            llama_cpp=str(LLAMA_CPP), argv=argv), encoding="utf-8")
        cmd = [sys.executable, str(driver)]
        print("+", " ".join(argv), flush=True)
        subprocess.run(cmd, check=True, cwd=str(LLAMA_CPP))


def _write_main(sf: SafetensorsSet, keys: set[str], ckpt_dir: Path,
                 out_path: Path, outtype: str) -> None:
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    text = cfg["vlm_config"]["text_config"]
    dit = cfg["dit_config"]
    vcfg = cfg["vlm_config"]["vision_config"]

    n_layers = text["num_hidden_layers"]
    hidden = text["hidden_size"]
    inter = text["intermediate_size"]
    n_q_heads = text["num_attention_heads"]
    n_kv_heads = text["num_key_value_heads"]
    head_dim = text["head_dim"]
    rope_theta = text["rope_theta"]
    mrope_section = text["rope_scaling"]["mrope_section"]
    vocab_size = text["vocab_size"]

    dit_hidden = dit["hidden_size"]
    dit_layers = dit["num_hidden_layers"]
    dit_kv_heads = dit["num_key_value_heads"]
    dit_heads = dit_hidden // dit["head_dim"]
    action_dim = cfg["action_dim"]
    action_length = cfg["action_length"]
    state_dim = cfg["state_dim"]

    # ---- sanity checks against the actual checkpoint ----------------------
    emb = sf.get_tensor(f"{VLM_PREFIX}.embed_tokens.weight")
    assert tuple(emb.shape) == (vocab_size, hidden), emb.shape
    q0 = sf.get_tensor(f"{VLM_PREFIX}.layers.0.self_attn.q_proj.weight")
    assert tuple(q0.shape) == (n_q_heads * head_dim, hidden), q0.shape
    k0 = sf.get_tensor(f"{VLM_PREFIX}.layers.0.self_attn.k_proj.weight")
    assert tuple(k0.shape) == (n_kv_heads * head_dim, hidden), k0.shape
    qkv0 = sf.get_tensor(f"{DIT_PREFIX}.layers.0.attn.qkv_proj.weight")
    assert tuple(qkv0.shape) == (3 * dit_hidden, dit_hidden), qkv0.shape
    adaln0 = sf.get_tensor(f"{DIT_PREFIX}.layers.0.adaln_table")
    assert tuple(adaln0.shape) == (6, dit_hidden), adaln0.shape
    assert sf.get_tensor("sink.weight").shape == (1, dit_hidden)

    # ---- action statistics ------------------------------------------------
    # NOTE: the shipped preprocessor_config stores per-timestep stats shaped
    # (10, 32), but all rows are identical and MiBotProcessor.decode_action
    # broadcasts them as a plain (32,) vector; row 0 is the canonical one.
    pre = json.loads((ckpt_dir / "preprocessor_config.json").read_text())
    action_config = pre.get("action_config", {})
    robot = "libero_all"
    if robot not in action_config:
        raise SystemExit(
            f"action_config has no '{robot}' entry; available: "
            f"{sorted(action_config)}")
    stats = action_config[robot]
    action_mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1, 32)[0]
    action_std  = np.asarray(stats["std"],  dtype=np.float32).reshape(-1, 32)[0]
    assert action_mean.size == action_dim and action_std.size == action_dim

    print(f"resolved cfg: vlm {n_layers}L hidden={hidden} heads={n_q_heads}q/"
          f"{n_kv_heads}kv x{head_dim} rope_theta={rope_theta} "
          f"mrope={mrope_section} vocab={vocab_size}")
    print(f"              dit {dit_layers}L hidden={dit_hidden} heads={dit_heads} "
          f"chunk={action_length} action_dim={action_dim} state_dim={state_dim}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out_path}")
    w = gguf.GGUFWriter(str(out_path), arch=ARCH)
    matmul = "f32" if outtype == "f32" else "bf16"

    def u32(k, v): w.add_uint32(f"xr0.{k}", int(v))
    def f32(k, v): w.add_float32(f"xr0.{k}", float(v))
    def arr_f32(k, v): w.add_array(f"xr0.{k}", [float(x) for x in v])

    w.add_string("xr0.architecture", ARCH)
    u32("vlm.hidden", hidden)
    u32("vlm.intermediate", inter)
    u32("vlm.n_q_heads", n_q_heads)
    u32("vlm.n_kv_heads", n_kv_heads)
    u32("vlm.head_dim", head_dim)
    u32("vlm.n_layers", n_layers)
    u32("vlm.vocab_size", vocab_size)
    f32("vlm.rms_norm_eps", text["rms_norm_eps"])
    f32("vlm.rope_theta", rope_theta)
    arr_f32("vlm.mrope_sections", mrope_section)
    u32("vlm.image_token_id", cfg["vlm_config"]["image_token_id"])
    u32("vlm.deepstack_start", 0)   # deepstack injects after decoder layers 0..2

    u32("dit.hidden", dit_hidden)
    u32("dit.n_layers", dit_layers)
    u32("dit.n_heads", dit_heads)
    u32("dit.n_kv_heads", dit_kv_heads)
    u32("dit.head_dim", dit["head_dim"])
    f32("dit.rms_norm_eps", 1e-6)   # Qwen3VLTextRMSNorm default
    f32("dit.rope_theta", rope_theta)
    u32("dit.kv_start_layer", n_layers - dit_layers)

    u32("action_length", action_length)
    u32("action_dim", action_dim)
    u32("state_dim", state_dim)
    u32("num_steps", 5)             # flow-matching steps (HF default)
    w.add_array("xr0.action_mean", action_mean.tolist())
    w.add_array("xr0.action_std", action_std.tolist())

    # ---- tensors -----------------------------------------------------------
    _add_tensor(w, "token_embd.weight", emb, matmul)
    for i in range(n_layers):
        p = f"{VLM_PREFIX}.layers.{i}"
        _add_tensor(w, f"vlm.blk.{i}.attn_norm.weight",
                    sf.get_tensor(f"{p}.input_layernorm.weight"), "f32")
        _add_tensor(w, f"vlm.blk.{i}.attn_q.weight",
                    sf.get_tensor(f"{p}.self_attn.q_proj.weight"), matmul)
        _add_tensor(w, f"vlm.blk.{i}.attn_k.weight",
                    sf.get_tensor(f"{p}.self_attn.k_proj.weight"), matmul)
        _add_tensor(w, f"vlm.blk.{i}.attn_v.weight",
                    sf.get_tensor(f"{p}.self_attn.v_proj.weight"), matmul)
        _add_tensor(w, f"vlm.blk.{i}.attn_o.weight",
                    sf.get_tensor(f"{p}.self_attn.o_proj.weight"), matmul)
        _add_tensor(w, f"vlm.blk.{i}.attn_q_norm.weight",
                    sf.get_tensor(f"{p}.self_attn.q_norm.weight"), "f32")
        _add_tensor(w, f"vlm.blk.{i}.attn_k_norm.weight",
                    sf.get_tensor(f"{p}.self_attn.k_norm.weight"), "f32")
        _add_tensor(w, f"vlm.blk.{i}.ffn_norm.weight",
                    sf.get_tensor(f"{p}.post_attention_layernorm.weight"), "f32")
        _add_tensor(w, f"vlm.blk.{i}.ffn_gate.weight",
                    sf.get_tensor(f"{p}.mlp.gate_proj.weight"), matmul)
        _add_tensor(w, f"vlm.blk.{i}.ffn_up.weight",
                    sf.get_tensor(f"{p}.mlp.up_proj.weight"), matmul)
        _add_tensor(w, f"vlm.blk.{i}.ffn_down.weight",
                    sf.get_tensor(f"{p}.mlp.down_proj.weight"), matmul)

    for i in range(dit_layers):
        p = f"{DIT_PREFIX}.layers.{i}"
        _add_tensor(w, f"dit.blk.{i}.input_layernorm.weight",
                    sf.get_tensor(f"{p}.input_layernorm.weight"), "f32")
        _add_tensor(w, f"dit.blk.{i}.middle_layernorm.weight",
                    sf.get_tensor(f"{p}.middle_layernorm.weight"), "f32")
        _add_tensor(w, f"dit.blk.{i}.post_layernorm.weight",
                    sf.get_tensor(f"{p}.post_layernorm.weight"), "f32")
        _add_tensor(w, f"dit.blk.{i}.final_layernorm.weight",
                    sf.get_tensor(f"{p}.final_layernorm.weight"), "f32")
        _add_tensor(w, f"dit.blk.{i}.adaln_table",
                    sf.get_tensor(f"{p}.adaln_table"), "f32")
        _add_tensor(w, f"dit.blk.{i}.attn_qkv.weight",
                    sf.get_tensor(f"{p}.attn.qkv_proj.weight"), matmul)
        _add_tensor(w, f"dit.blk.{i}.attn_qkv.bias",
                    sf.get_tensor(f"{p}.attn.qkv_proj.bias"), "f32")
        _add_tensor(w, f"dit.blk.{i}.attn_q_norm.weight",
                    sf.get_tensor(f"{p}.attn.q_norm.weight"), "f32")
        _add_tensor(w, f"dit.blk.{i}.attn_k_norm.weight",
                    sf.get_tensor(f"{p}.attn.k_norm.weight"), "f32")
        _add_tensor(w, f"dit.blk.{i}.attn_o.weight",
                    sf.get_tensor(f"{p}.attn.o_proj.weight"), matmul)
        _add_tensor(w, f"dit.blk.{i}.ffn_gate.weight",
                    sf.get_tensor(f"{p}.mlp.gate_proj.weight"), matmul)
        _add_tensor(w, f"dit.blk.{i}.ffn_up.weight",
                    sf.get_tensor(f"{p}.mlp.up_proj.weight"), matmul)
        _add_tensor(w, f"dit.blk.{i}.ffn_down.weight",
                    sf.get_tensor(f"{p}.mlp.down_proj.weight"), matmul)

    _add_tensor(w, "dit.sink.weight", sf.get_tensor("sink.weight"), "f32")
    _add_tensor(w, "dit.state_proj.0.weight",
                sf.get_tensor("state_projector.layers.0.weight"), matmul)
    _add_tensor(w, "dit.state_proj.2.weight",
                sf.get_tensor("state_projector.layers.2.weight"), matmul)
    _add_tensor(w, "dit.action_proj.0.weight",
                sf.get_tensor("action_projector.layers.0.weight"), matmul)
    _add_tensor(w, "dit.action_proj.2.weight",
                sf.get_tensor("action_projector.layers.2.weight"), matmul)
    _add_tensor(w, "dit.action_out.0.weight",
                sf.get_tensor("action_output_layer.layers.0.weight"), matmul)
    _add_tensor(w, "dit.action_out.2.weight",
                sf.get_tensor("action_output_layer.layers.2.weight"), matmul)
    _add_tensor(w, "dit.t_embedder.0.weight",
                sf.get_tensor("t_embedder.mlp.0.weight"), matmul)
    _add_tensor(w, "dit.t_embedder.2.weight",
                sf.get_tensor("t_embedder.mlp.2.weight"), matmul)
    _add_tensor(w, "dit.t_proj.weight",
                sf.get_tensor("t_projector.layers.0.weight"), matmul)
    _add_tensor(w, "dit.t_proj.bias",
                sf.get_tensor("t_projector.layers.0.bias"), "f32")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Xiaomi-Robotics-0-* model dir (safetensors + configs)")
    ap.add_argument("--output", required=True, type=Path,
                    help="output main GGUF (text backbone + DiT + stats)")
    ap.add_argument("--mmproj", required=True, type=Path,
                    help="output mmproj GGUF (Qwen3-VL vision tower)")
    ap.add_argument("--outtype", default="bf16", choices=["bf16", "f16", "f32"],
                    help="weight storage type (default bf16)")
    ap.add_argument("--skip-mmproj", action="store_true",
                    help="only write the main GGUF (mmproj already built)")
    args = ap.parse_args()

    ckpt_dir = args.checkpoint.resolve()
    if not (ckpt_dir / "config.json").exists():
        raise SystemExit(f"config.json not found in {ckpt_dir}")

    if not args.skip_mmproj:
        args.mmproj.parent.mkdir(parents=True, exist_ok=True)
        _build_mmproj(ckpt_dir, args.mmproj,
                      "f32" if args.outtype == "f32" else "f16")

    sf = SafetensorsSet(ckpt_dir)
    try:
        _write_main(sf, sf.keys(), ckpt_dir, args.output, args.outtype)
    finally:
        sf.close()

    print("done.")
    print(f"  main gguf:    {args.output}")
    print(f"  mmproj gguf:  {args.mmproj}")


if __name__ == "__main__":
    main()
