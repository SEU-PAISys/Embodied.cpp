#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Smoke-test HY-VLA precomputed visual tokens through vla-server."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import zmq


os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def load_pb():
    if "vla_pb2" in sys.modules:
        return sys.modules["vla_pb2"]
    proto_file = Path(os.environ.get(
        "VLA_CPP_PROTO",
        Path(__file__).resolve().parents[1] / "serving" / "vla.proto",
    ))
    tmpdir = Path(tempfile.mkdtemp(prefix="hy-vla-pb-"))
    protoc = os.environ.get("VLA_CPP_PROTOC")
    if protoc is None:
        protoc = "/usr/bin/protoc" if Path("/usr/bin/protoc").exists() else "protoc"
    subprocess.check_call([
        protoc,
        f"--proto_path={proto_file.parent}",
        f"--python_out={tmpdir}",
        str(proto_file),
    ])
    sys.path.insert(0, str(tmpdir))
    import vla_pb2
    return vla_pb2


def bf16_roundtrip(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32, copy=False).view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    rounded = (u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addr", default="tcp://127.0.0.1:6099")
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--n-vis", type=int, default=3)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=40)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--zero-noise", action="store_true",
                        help="Send a deterministic all-zero action noise tensor.")
    parser.add_argument("--noise-f32", type=Path, default=None,
                        help="Raw float32 noise/action state tensor [chunk, action_dim].")
    parser.add_argument("--routed", action="store_true",
                        help="Send full prefix embeddings plus attention_mask as 0=text/1=vision modality mask.")
    parser.add_argument("--prefix-f32", type=Path, default=None,
                        help="Raw float32 full-prefix embeddings for --routed.")
    parser.add_argument("--mask-i32", type=Path, default=None,
                        help="Raw int32 0=text / 1=vision modality mask for --routed.")
    parser.add_argument("--modality", nargs="*", type=int, default=None,
                        help="0=text / 1=vision modality mask for --routed.")
    parser.add_argument("--dump-f32", type=Path, default=None,
                        help="Dump response action_chunk/debug tensor as raw float32.")
    args = parser.parse_args()

    pb = load_pb()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    sock.connect(args.addr)

    req = pb.PredictRequest()
    req.request_id = 1
    req.lang_tokens.extend([120000, 120001, 120020, 7])
    req.state.extend([0.0] * args.state_dim)
    if args.noise_f32 is not None:
        noise = np.fromfile(args.noise_f32, dtype=np.float32)
        expected = args.chunk * args.action_dim
        if noise.size != expected:
            raise SystemExit(f"{args.noise_f32} has {noise.size} floats, expected {expected}")
        req.noise.extend(noise.astype(np.float32).tolist())
    elif args.zero_noise:
        req.noise.extend([0.0] * (args.chunk * args.action_dim))
    vals = np.sin(np.arange(args.n_vis * args.hidden, dtype=np.float32) * np.float32(0.013))
    vals = bf16_roundtrip(vals * np.float32(0.02))
    if args.routed:
        if args.prefix_f32 is None:
            raise SystemExit("--routed requires --prefix-f32")
        full = np.fromfile(args.prefix_f32, dtype=np.float32)
        if full.size % args.hidden != 0:
            raise SystemExit(f"{args.prefix_f32} size {full.size} is not divisible by hidden={args.hidden}")
        n_pref = full.size // args.hidden
        if args.mask_i32 is not None:
            modality = np.fromfile(args.mask_i32, dtype=np.int32).tolist()
        else:
            modality = args.modality
        if not modality:
            raise SystemExit("--routed requires --mask-i32 or --modality")
        if len(modality) != n_pref:
            raise SystemExit(f"modality length {len(modality)} != prefix token count {n_pref}")
        req.precomputed_img_emb.extend(full.astype(np.float32).tolist())
        req.precomputed_img_emb_n_views = int(n_pref)
        req.attention_mask.extend([int(v) for v in modality])
    else:
        req.precomputed_img_emb.extend(vals.astype(np.float32).tolist())
        req.precomputed_img_emb_n_views = args.n_vis

    sock.send(req.SerializeToString())
    resp = pb.PredictResponse()
    resp.ParseFromString(sock.recv())
    if resp.error:
        raise RuntimeError(resp.error)
    arr = np.asarray(resp.action_chunk, dtype=np.float32)
    print(
        f"response request_id={resp.request_id} chunk={resp.chunk_size} action_dim={resp.action_dim} "
        f"n={arr.size} sum={float(arr.sum()):.12f} min={float(arr.min()):.12f} max={float(arr.max()):.12f}"
    )
    if args.dump_f32 is not None:
        args.dump_f32.parent.mkdir(parents=True, exist_ok=True)
        arr.astype(np.float32, copy=False).tofile(args.dump_f32)
        print(f"dumped_f32={args.dump_f32}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
