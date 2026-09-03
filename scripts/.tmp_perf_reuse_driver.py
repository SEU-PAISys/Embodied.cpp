#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path("/home/xuling/yangzhixiao/Embodied.cpp")
PY = ROOT / "eval/sim/libero/libero_uv/.venv/bin/python"
RAW = Path("/tmp/codex-padding-dvhsmtju")
OUT = Path(tempfile.mkdtemp(prefix="boundary-reuse-final-", dir=RAW))
PB = "/home/xuling/yangzhixiao/vla_pb_lib"
WRAP = "import vla_pb2,sys,runpy;sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"

runs = [
    ("turbovla-bf16", "turbovla", [str(RAW / "turbovla-layout-bf16.gguf")], {},
     RAW / "validated-object/turbovla_parity_inputs.npz", []),
    ("turbovla-q8_0", "turbovla", [str(RAW / "turbovla-layout-q8_0.gguf")], {},
     RAW / "validated-object/turbovla_parity_inputs.npz", []),
    ("turbovla-q4_0", "turbovla", [str(RAW / "turbovla-layout-q4_0.gguf")], {},
     RAW / "validated-object/turbovla_parity_inputs.npz", []),
    ("xr0-bf16", "xr0", ["checkpoints/xr0/xr0-mmproj.gguf", "checkpoints/xr0/xr0.gguf"],
     {"VLA_XR0_CLIP_GPU": "1"}, RAW / "boundary-a1hou9zj/xr0-fixture.npz",
     ["--hf-dir", "checkpoints/xr0/hf", "--xr0-vision-dtype", "f16"]),
    ("xr0-q8_0", "xr0", ["checkpoints/xr0/xr0-mmproj.gguf", "checkpoints/xr0/xr0-q8_0.gguf"],
     {"VLA_XR0_CLIP_GPU": "1"}, RAW / "boundary-a1hou9zj/xr0-fixture.npz",
     ["--hf-dir", "checkpoints/xr0/hf", "--xr0-vision-dtype", "f16"]),
    ("xr0-q4_k", "xr0", ["checkpoints/xr0/xr0-mmproj.gguf", "checkpoints/xr0/xr0-q4_k.gguf"],
     {"VLA_XR0_CLIP_GPU": "1"}, RAW / "boundary-a1hou9zj/xr0-fixture.npz",
     ["--hf-dir", "checkpoints/xr0/hf", "--xr0-vision-dtype", "f16"]),
    ("xvla-bf16", "xvla", ["checkpoints/xvla/xvla-libero.gguf"], {},
     RAW / "boundary-a1hou9zj/xvla-fixture.npz", ["--hf-dir", "/home/xuling/yangzhixiao/xvla_tok_libero"]),
    ("xvla-q8_0", "xvla", ["checkpoints/xvla/xvla-libero-q8_0.gguf"], {},
     RAW / "boundary-a1hou9zj/xvla-fixture.npz", ["--hf-dir", "/home/xuling/yangzhixiao/xvla_tok_libero"]),
    ("xvla-q4_0", "xvla", ["checkpoints/xvla/xvla-libero-q4_0.gguf"], {},
     RAW / "boundary-a1hou9zj/xvla-fixture.npz", ["--hf-dir", "/home/xuling/yangzhixiao/xvla_tok_libero"]),
    ("xvla-f32", "xvla", [str(RAW / "xvla-domain-fixed-f32.gguf")], {"VLA_XVLA_F32_WEIGHTS": "1"},
     RAW / "boundary-a1hou9zj/xvla-fixture.npz", ["--hf-dir", "/home/xuling/yangzhixiao/xvla_tok_libero"]),
]

def wait_ready(process, log):
    for _ in range(180):
        if process.poll() is not None:
            raise RuntimeError(log.read_text(errors="replace"))
        if log.exists() and "ready" in log.read_text(errors="replace"):
            return
        time.sleep(1)
    raise TimeoutError("server did not become ready")

gpu_processes = subprocess.run(
    ["nvidia-smi", "-i", "2", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
    text=True, capture_output=True, check=True).stdout.strip()
if gpu_processes:
    raise SystemExit(f"GPU 2 is not idle:\n{gpu_processes}")

selected = [runs[0], runs[3], runs[6]]
for index, (name, arch, model_args, extra_env, fixture, client_args) in enumerate(selected):
    port = 6300 + index
    output = OUT / f"{name}.json"
    log = OUT / f"{name}.server.log"
    env = os.environ | {"CUDA_VISIBLE_DEVICES": "2", "OMP_NUM_THREADS": "4"} | extra_env
    with log.open("w") as stream:
        server = subprocess.Popen([str(ROOT / "build/vla-server"), *model_args,
                                   "--bind", f"tcp://127.0.0.1:{port}"],
                                  cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    try:
        wait_ready(server, log)
        client_env = env | {"PYTHONPATH": f"{PB}:{ROOT / 'eval'}"}
        subprocess.run([str(PY), "-c", WRAP, "scripts/bench_vla_boundary.py",
                        "--arch", arch, "--fixture", str(fixture), "--n", "500",
                        "--warmup", "5", "--memory-requests", "0", "--backend", "cpp",
                        "--server-pid", str(server.pid), "--address", f"tcp://127.0.0.1:{port}",
                        "--output", str(output), *client_args], cwd=ROOT, env=client_env, check=True)
        data = json.loads(output.read_text())
        print(name, json.dumps({"latency_ms": data["latency_ms"],
                                "sampled_process_peak_mib": data["sampled_process_peak_mib"]}), flush=True)
    finally:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

(OUT / "manifest.json").write_text(json.dumps({"runs": [r[0] for r in selected], "n": 500, "complete": True}, indent=2))
print(f"OUTPUT={OUT}")
