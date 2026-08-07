#!/usr/bin/env python3
"""Compare a fixed SmolVLA request against a PyTorch action reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zmq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from client.vla_cpp_client import _load_pb


def make_request(pb, inputs: np.lib.npyio.NpzFile, request_id: int):
    req = pb.PredictRequest(request_id=request_id)
    for key in ("image", "image2"):
        chw = np.ascontiguousarray(inputs[key], dtype=np.float32)
        if chw.shape != (3, 512, 512):
            raise ValueError(f"{key}: expected (3, 512, 512), got {chw.shape}")
        hwc = np.ascontiguousarray(np.transpose(chw, (1, 2, 0)))
        image = req.images.add()
        image.encoding = pb.Image.F32_RGB_01
        image.height = 512
        image.width = 512
        image.data = hwc.tobytes()
    tokens = np.asarray(inputs["input_ids"], dtype=np.int32).reshape(-1)
    mask = np.asarray(inputs["attention_mask"], dtype=np.uint32).reshape(-1)
    state = np.asarray(inputs["state"], dtype=np.float32).reshape(-1)
    noise = np.asarray(inputs["noise"], dtype=np.float32)
    if tokens.size != 48 or mask.size != 48:
        raise ValueError("input_ids and attention_mask must each contain 48 values")
    if state.size > 32:
        raise ValueError(f"state has {state.size} values; maximum is 32")
    if noise.shape != (50, 32):
        raise ValueError(f"noise: expected (50, 32), got {noise.shape}")
    padded_state = np.zeros(32, dtype=np.float32)
    padded_state[: state.size] = state
    req.lang_tokens.extend(tokens.tolist())
    req.attention_mask.extend(mask.tolist())
    req.state.extend(padded_state.tolist())
    req.noise.extend(noise.reshape(-1).tolist())
    return req


def predict(sock, pb, request) -> np.ndarray:
    sock.send(request.SerializeToString())
    body = sock.recv()
    response = pb.PredictResponse()
    response.ParseFromString(body)
    if response.error:
        raise RuntimeError(response.error)
    action = np.asarray(response.action_chunk, dtype=np.float32)
    return action.reshape(response.chunk_size, response.action_dim)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--server", default="tcp://127.0.0.1:5566")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-abs", type=float, default=0.05)
    parser.add_argument("--mean-abs", type=float, default=0.005)
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    args = parser.parse_args()

    pb = _load_pb()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    sock.connect(args.server)
    try:
        with np.load(args.inputs) as inputs:
            first = predict(sock, pb, make_request(pb, inputs, 0))[:, :7]
            second = predict(sock, pb, make_request(pb, inputs, 1))[:, :7]
    finally:
        sock.close()

    reference = np.asarray(np.load(args.reference), dtype=np.float32)
    if reference.shape != first.shape:
        raise ValueError(f"reference shape {reference.shape} != C++ shape {first.shape}")
    diff = np.abs(first - reference)
    repeat_diff = np.abs(first - second)
    report = {
        "shape": list(first.shape),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(first - reference)))),
        "repeat_max_abs_error": float(repeat_diff.max()),
        "cpp_nan": bool(np.isnan(first).any()),
        "cpp_inf": bool(np.isinf(first).any()),
    }
    report["passed"] = (
        not report["cpp_nan"]
        and not report["cpp_inf"]
        and report["max_abs_error"] <= args.max_abs
        and report["mean_abs_error"] <= args.mean_abs
        and report["repeat_max_abs_error"] == 0.0
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
