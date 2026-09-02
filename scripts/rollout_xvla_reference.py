"""Official-PyTorch baseline rollout for X-VLA on LIBERO.

Mirrors eval/client/run_sim_client_direct.py's protocol (same vendored
LiberoEnv, seeds, init states, per-suite step caps, 30-step open-loop chunk
replay and absolute ee control) so results are directly comparable with the
C++ server numbers. The policy is the official HF PyTorch checkpoint loaded
with trust_remote_code (XVLAForCausalLM + XVLAProcessor).

Example:
    python scripts/rollout_xvla_reference.py \
        --model-path ../xlva_ref/libero \
        --libero-suite object --task-id 0 --n-episodes 10 --seed 7
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(EVAL))

import numpy as np
import torch

import sim.libero  # noqa: F401  side-effect: registers gymnasium envs
from adapter.sim.libero import _rotate6d_to_axisangle
from sim.libero.libero_env import LiberoEnv, TASK_SUITE_MAX_STEPS

LIBERO_SUITE_ALIASES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "10": "libero_10",
    "long": "libero_90",
}


def build_obs(model_inputs_source: dict, processor, task: str, device: str):
    """Package the gym observation exactly like the official client:
    agentview flipped 180 deg, wrist raw, proprio [pos3 ori6d6 grip1]+zeros."""
    pixels = model_inputs_source["pixels"]
    main = np.flip(np.flip(np.asarray(pixels["image"]), 0), 1)
    views = [main]
    if "image2" in pixels:
        views.append(np.asarray(pixels["image2"]))

    eef = model_inputs_source["robot_state"]["eef"]
    pos = np.asarray(eef["pos"], dtype=np.float64).reshape(-1)[:3]
    mat = np.asarray(eef["mat"], dtype=np.float64).reshape(3, 3)
    ori6d = np.concatenate([mat[:3, 0], mat[:3, 1]])
    cur = np.concatenate([pos, ori6d, [0.0]]).astype(np.float32)
    proprio = np.concatenate([cur, np.zeros_like(cur)])[None]        # [1, 20]

    inputs = processor(images=views, language_instruction=task)
    inputs["proprio"] = torch.as_tensor(proprio, dtype=torch.float32)
    inputs["domain_id"] = torch.tensor([3], dtype=torch.long)
    return {k: v.to(device) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()}


def chunk_to_env_action(row: np.ndarray) -> np.ndarray:
    """Absolute ee6d row [pos3, rot6d6, grip1] -> env action, mirroring
    XVLALIBEROParser.parse_action."""
    a = np.asarray(row, dtype=np.float64).reshape(-1)[:10]
    ori = _rotate6d_to_axisangle(a[3:9])
    grip = 1.0 if a[9] > 0.5 else -1.0
    return np.concatenate([a[0:3], ori, [grip]]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--task", type=str, default="libero_object")
    ap.add_argument("--libero-suite", type=str, default=None,
                    help="Short name: spatial | object | goal | 10 | long.")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--observation-size", type=int, default=256)
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "outputs" / "xvla_py_baseline")
    ap.add_argument("--record-video", action="store_true")
    ap.add_argument("--denoise-steps", type=int, default=10)
    args = ap.parse_args()

    suite = LIBERO_SUITE_ALIASES.get(args.libero_suite or args.task,
                                     args.libero_suite or args.task)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading HF X-VLA from {args.model_path} on {device} ...", flush=True)
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device).to(torch.float32).eval()
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        processor = None
    if processor is None:
        raise SystemExit("failed to load XVLAProcessor from the checkpoint dir")

    output_dir = args.output_dir / suite / f"task_{args.task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = LiberoEnv(
        task_suite_name=suite,
        task_id=args.task_id,
        seed=args.seed,
        observation_width=args.observation_size,
        observation_height=args.observation_size,
        output_video_dir=output_dir if args.record_video else None,
        control_mode="absolute",
    )

    max_steps = TASK_SUITE_MAX_STEPS.get(suite, 500)
    success_count, skipped, times_ms = 0, 0, []
    for episode in range(args.n_episodes):
        print(f"*** {suite}/task_{args.task_id} episode {episode + 1}/{args.n_episodes}",
              flush=True)
        obs, info = env.reset()
        # deterministic noise draw per (episode, query); the official flow
        # head samples x1 internally with torch.randn
        torch.manual_seed(args.seed * 1000 + episode)
        done = truncated = False
        step_id, queue = 0, []
        t_run = []
        while True:
            if not queue:
                t0 = time.time()
                inputs = build_obs(obs, processor, env.task_description, device)
                with torch.no_grad():
                    chunk = model.generate_actions(**inputs,
                                                   steps=args.denoise_steps)
                times_ms.append((time.time() - t0) * 1000.0)
                queue = chunk.squeeze(0).float().cpu().numpy().tolist()
            action = chunk_to_env_action(queue.pop(0))
            obs, reward, done, truncated, info = env.step(action)
            step_id += 1
            if done or truncated or step_id >= max_steps:
                break
        success_count += float(info.get("is_success", 0.0))
        print(f"- finished after {step_id} steps, success={info.get('is_success')}",
              flush=True)

    env.close()
    rate = success_count / max(1, args.n_episodes)
    avg_ms = sum(times_ms) / len(times_ms) if times_ms else 0.0
    summary = (
        f"Arch: xvla-pytorch-official\n"
        f"Task: {suite}/task_{args.task_id}\n"
        f"Success rate: {rate:.2%}  ({int(success_count)}/{args.n_episodes})\n"
        f"Average query time: {avg_ms:.2f} ms\n"
    )
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
