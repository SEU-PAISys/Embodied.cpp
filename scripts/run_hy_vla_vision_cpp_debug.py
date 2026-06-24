#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Request HY-VLA image-path tensors from vla-server and dump float32.

The server decides whether it returns an action chunk or a debug tensor via
VLA_HY_VLA_DEBUG_OUTPUT. Use --mode action for the normal inference path.
"""

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
    tmpdir = Path(tempfile.mkdtemp(prefix="hy-vla-vision-pb-"))
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


def make_image(kind: str, h: int, w: int) -> np.ndarray:
    if kind == "zero":
        return np.zeros((h, w, 3), dtype=np.uint8)
    if kind == "gradient":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        arr[..., 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        arr[..., 2] = 17
        return arr
    raise ValueError(f"unknown image kind: {kind}")


def deterministic_image(cam: int, frame: int, h: int, w: int) -> np.ndarray:
    y, x, c = np.meshgrid(
        np.arange(h, dtype=np.uint16),
        np.arange(w, dtype=np.uint16),
        np.arange(3, dtype=np.uint16),
        indexing="ij",
    )
    arr = (x * (3 + cam * 7 + frame) + y * (5 + cam + frame * 3) + c * 37 + cam * 41 + frame * 19) % 256
    return np.ascontiguousarray(arr.astype(np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addr", default="tcp://127.0.0.1:6125")
    parser.add_argument("--timeout-ms", type=int, default=300_000)
    parser.add_argument(
        "--mode",
        choices=["action", "vision_pixels", "vision_patch", "vision_vit", "vision_merger"],
        default="action",
    )
    parser.add_argument("--image-kind", choices=["zero", "gradient"], default="zero")
    parser.add_argument("--mem-history", type=int, default=1,
                        help="Send 3 camera-major K-frame videos: top_head, hand_left, hand_right.")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=40)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--zero-noise", action="store_true",
                        help="Send deterministic all-zero action noise.")
    parser.add_argument("--dump-f32", type=Path, required=True)
    args = parser.parse_args()

    pb = load_pb()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    sock.connect(args.addr)

    req = pb.PredictRequest()
    req.request_id = 1
    if args.mem_history > 1:
        for cam in range(3):
            for frame in range(args.mem_history):
                image = deterministic_image(cam, frame, args.height, args.width)
                img = req.images.add()
                img.encoding = pb.Image.RGB_U8
                img.height = args.height
                img.width = args.width
                img.data = image.tobytes()
    else:
        image = make_image(args.image_kind, args.height, args.width)
        img = req.images.add()
        img.encoding = pb.Image.RGB_U8
        img.height = args.height
        img.width = args.width
        img.data = image.tobytes()
    req.lang_tokens.extend([120000, 120001, 120020, 7])
    req.state.extend([0.0] * args.state_dim)
    if args.zero_noise:
        req.noise.extend([0.0] * (args.chunk * args.action_dim))

    sock.send(req.SerializeToString())
    resp = pb.PredictResponse()
    resp.ParseFromString(sock.recv())
    if resp.error:
        raise RuntimeError(resp.error)
    arr = np.asarray(resp.action_chunk, dtype=np.float32)
    args.dump_f32.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(args.dump_f32)
    print(
        f"{args.mode}: n={arr.size} sum={float(arr.sum()):.12f} "
        f"mean={float(arr.mean()):.12f} min={float(arr.min()):.12f} max={float(arr.max()):.12f}"
    )
    print(f"dumped_f32={args.dump_f32}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
