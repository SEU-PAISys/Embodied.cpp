"""Numerical equivalence check: official generate_actions() vs the reference
client's explicit-noise denoise loop, with the SAME initial 30x20 noise.

Both paths must run on the same device/dtype with identical inputs. The
official path draws x1 from the global RNG after torch.manual_seed(S); the
explicit path reproduces that exact draw (same randn shape/order) and feeds it
through the hand-written loop from XVLAReferenceClient._predict_chunk.
Saves max/mean absolute error as JSON evidence.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

MODEL = "/home/xuling/yangzhixiao/xvla_hf"
SEED = 1234
DOMAIN = 3

from transformers import AutoImageProcessor, AutoModel, AutoProcessor  # noqa: E402

device = torch.device("cuda")
dtype = torch.bfloat16
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
proc_img = AutoImageProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to(device, dtype=dtype).eval()

rng = np.random.default_rng(7)
img_a = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
img_b = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
state = rng.uniform(-1, 1, 20).astype(np.float32)
task = "put the bowl on the stove"

inputs = processor(images=[img_a, img_b], language_instruction=task)
inputs.update(proprio=torch.from_numpy(state[None]),
              domain_id=torch.tensor([DOMAIN], dtype=torch.long))
inputs = {k: v.to(device, dtype=dtype if v.is_floating_point() else v.dtype)
          for k, v in inputs.items()}

# ---- Path A: official generate_actions (global-RNG noise) ----
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
with torch.inference_mode():
    action_a = model.generate_actions(
        inputs["input_ids"], inputs["image_input"], inputs["image_mask"],
        inputs["domain_id"], inputs["proprio"], steps=10)
action_a_np = action_a[0].float().cpu().numpy()

# ---- Path B: explicit-noise hand-written loop (XVLAReferenceClient logic) ----
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
x1 = torch.randn((1, 30, 20), device=device, dtype=dtype)
with torch.inference_mode():
    enc = model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
    action = torch.zeros_like(x1)
    for index in range(10, 0, -1):
        t = torch.full((1,), index / 10, device=device, dtype=dtype)
        x_t = x1 * t[:, None, None] + action * (1 - t[:, None, None])
        proprio_m, x_t_m = model.action_space.preprocess(inputs["proprio"], x_t)
        action = model.transformer(domain_id=inputs["domain_id"],
                                   action_with_noise=x_t_m, proprio=proprio_m, t=t, **enc)
    action_b_np = model.action_space.postprocess(action)[0].float().cpu().numpy()

diff = np.abs(action_a_np.astype(np.float64) - action_b_np.astype(np.float64))
result = {
    "seed": SEED, "domain_id": DOMAIN, "dtype": str(dtype),
    "shape": list(action_a_np.shape),
    "max_abs_error": float(diff.max()), "mean_abs_error": float(diff.mean()),
    "finite": bool(np.isfinite(action_a_np).all() and np.isfinite(action_b_np).all()),
}
out = Path("/home/xuling/yangzhixiao/xvla_denoise_equiv.json")
out.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
ok = result["max_abs_error"] <= 0.01 and result["finite"]
print("VERDICT:", "PASS (<=0.01)" if ok else "FAIL")
sys.exit(0 if ok else 1)
