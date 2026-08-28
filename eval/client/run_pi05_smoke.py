#!/usr/bin/env python3
"""Deterministic, simulator-independent pi0.5 smoke client.

The client intentionally avoids Torch, Transformers, LIBERO, and the Python
protobuf runtime. It encodes the small subset of serving/vla.proto used by the
smoke request directly and only imports pyzmq when a request is sent.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import platform
import statistics
import struct
import sys
import tempfile
import time
from typing import Iterable, Sequence
from urllib.parse import urlsplit


FIXTURE_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
RGB_U8 = 1


class SmokeValidationError(RuntimeError):
    """Raised when the fixture, protocol, or server response is invalid."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pack_f32(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _pack_i32(values: Sequence[int]) -> bytes:
    return struct.pack(f"<{len(values)}i", *values)


def _splitmix64(state: int) -> tuple[int, int]:
    mask = (1 << 64) - 1
    state = (state + 0x9E3779B97F4A7C15) & mask
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return state, (value ^ (value >> 31)) & mask


def generate_noise(seed: int, count: int) -> list[float]:
    """Generate deterministic approximately-normal float32 values.

    Each sample is the sum of twelve 24-bit dyadic uniforms minus six
    (Irwin-Hall). Unlike Box-Muller, this uses no platform libm operations, so
    the serialized float32 payload is stable across x86_64 and aarch64.
    """
    if seed < 0 or seed >= 1 << 64:
        raise SmokeValidationError("noise seed must be a uint64")
    if count <= 0:
        raise SmokeValidationError("noise count must be positive")
    state = seed
    values: list[float] = []
    scale = float(1 << 24)
    for _ in range(count):
        total = 0.0
        for _ in range(12):
            state, bits = _splitmix64(state)
            total += float(bits >> 40) / scale
        values.append(struct.unpack("<f", struct.pack("<f", total - 6.0))[0])
    return values


def generate_image(width: int, height: int, view: int) -> bytes:
    """Generate one HWC RGB_U8 image using the fixture's integer grid."""
    if width <= 0 or height <= 0:
        raise SmokeValidationError("image dimensions must be positive")
    if view not in (0, 1):
        raise SmokeValidationError(f"unsupported image view {view}; expected 0 or 1")
    data = bytearray(width * height * 3)
    offset = 0
    for y in range(height):
        for x in range(width):
            if view == 0:
                rgb = (x & 0xFF, y & 0xFF, (x + y) & 0xFF)
            else:
                rgb = ((3 * x + 5 * y) & 0xFF, (x ^ y) & 0xFF, (255 - x) & 0xFF)
            data[offset : offset + 3] = bytes(rgb)
            offset += 3
    return bytes(data)


def generate_language_tokens(spec: dict) -> list[int]:
    if spec.get("algorithm") != "fixed_prefix_then_pad_i32_v1":
        raise SmokeValidationError("unsupported language token algorithm")
    count = _positive_int(spec, "count", "language")
    pad_id = _nonnegative_int(spec, "pad_token_id", "language")
    prefix = spec.get("prefix_token_ids")
    if not isinstance(prefix, list) or not prefix:
        raise SmokeValidationError("language.prefix_token_ids must be a non-empty list")
    if len(prefix) > count:
        raise SmokeValidationError("language prefix exceeds the configured token count")
    tokens = []
    for index, value in enumerate(prefix):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SmokeValidationError(
                f"language.prefix_token_ids[{index}] must be a non-negative integer"
            )
        tokens.append(value)
    return tokens + [pad_id] * (count - len(tokens))


def _positive_int(mapping: dict, key: str, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SmokeValidationError(f"{context}.{key} must be a positive integer")
    return value


def _nonnegative_int(mapping: dict, key: str, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SmokeValidationError(f"{context}.{key} must be a non-negative integer")
    return value


def _expect_hash(actual: str, expected: object, context: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise SmokeValidationError(f"{context}.sha256 must be a 64-character digest")
    if actual != expected:
        raise SmokeValidationError(
            f"{context} hash mismatch: expected {expected}, generated {actual}"
        )


def load_fixture(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    try:
        fixture = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SmokeValidationError(f"invalid fixture JSON: {exc}") from exc
    if fixture.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise SmokeValidationError(
            f"unsupported fixture schema_version={fixture.get('schema_version')!r}"
        )

    contract = fixture.get("model_contract")
    request = fixture.get("request")
    protocols = fixture.get("protocols")
    if not isinstance(contract, dict) or not isinstance(request, dict):
        raise SmokeValidationError("fixture requires model_contract and request objects")
    if not isinstance(protocols, dict):
        raise SmokeValidationError("fixture requires a protocols object")
    if contract.get("architecture") != "pi05":
        raise SmokeValidationError("model_contract.architecture must be pi05")

    width = _positive_int(contract, "image_width", "model_contract")
    height = _positive_int(contract, "image_height", "model_contract")
    image_count = _positive_int(contract, "image_count", "model_contract")
    state_dim = _positive_int(contract, "state_dim", "model_contract")
    chunk_size = _positive_int(contract, "chunk_size", "model_contract")
    action_dim = _positive_int(contract, "action_dim", "model_contract")
    if image_count != 2:
        raise SmokeValidationError("this fixture version requires exactly two images")

    image_specs = request.get("images")
    if not isinstance(image_specs, list) or len(image_specs) != image_count:
        raise SmokeValidationError("request.images does not match model_contract.image_count")
    images: list[bytes] = []
    for index, image_spec in enumerate(image_specs):
        if not isinstance(image_spec, dict):
            raise SmokeValidationError(f"request.images[{index}] must be an object")
        if image_spec.get("algorithm") != "pi05_rgb_grid_u8_v1":
            raise SmokeValidationError(f"request.images[{index}] has unknown algorithm")
        view = _nonnegative_int(image_spec, "view", f"request.images[{index}]")
        if view != index:
            raise SmokeValidationError("image view order must be 0, 1")
        image = generate_image(width, height, view)
        _expect_hash(_sha256(image), image_spec.get("sha256"), f"request.images[{index}]")
        images.append(image)

    language_spec = request.get("language")
    if not isinstance(language_spec, dict):
        raise SmokeValidationError("request.language must be an object")
    tokens = generate_language_tokens(language_spec)
    _expect_hash(
        _sha256(_pack_i32(tokens)),
        language_spec.get("sha256_le_i32"),
        "request.language",
    )

    state_values = request.get("state_f32")
    if not isinstance(state_values, list) or len(state_values) != state_dim:
        raise SmokeValidationError(f"request.state_f32 must contain {state_dim} values")
    state = [float(value) for value in state_values]
    if not all(math.isfinite(value) for value in state):
        raise SmokeValidationError("request.state_f32 contains NaN or Inf")
    _expect_hash(
        _sha256(_pack_f32(state)),
        request.get("state_sha256_le_f32"),
        "request.state_f32",
    )

    noise_spec = request.get("noise")
    if not isinstance(noise_spec, dict):
        raise SmokeValidationError("request.noise must be an object")
    if noise_spec.get("algorithm") != "splitmix64_irwin_hall_f32_v1":
        raise SmokeValidationError("request.noise has unknown algorithm")
    noise_count = _positive_int(noise_spec, "count", "request.noise")
    expected_noise_count = chunk_size * action_dim
    if noise_count != expected_noise_count:
        raise SmokeValidationError(
            f"noise count {noise_count} != chunk_size * action_dim {expected_noise_count}"
        )
    noise = generate_noise(_nonnegative_int(noise_spec, "seed", "request.noise"), noise_count)
    _expect_hash(
        _sha256(_pack_f32(noise)),
        noise_spec.get("sha256_le_f32"),
        "request.noise",
    )

    for name, protocol in protocols.items():
        if not isinstance(protocol, dict):
            raise SmokeValidationError(f"protocols.{name} must be an object")
        _nonnegative_int(protocol, "warmup_requests", f"protocols.{name}")
        _positive_int(protocol, "measured_requests", f"protocols.{name}")
    if "functional" not in protocols or "performance" not in protocols:
        raise SmokeValidationError("fixture requires functional and performance protocols")

    prepared = {
        "fixture_sha256": _sha256(raw),
        "width": width,
        "height": height,
        "images": images,
        "tokens": tokens,
        "state": state,
        "noise": noise,
        "chunk_size": chunk_size,
        "action_dim": action_dim,
    }
    return fixture, prepared


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise SmokeValidationError("protobuf varint cannot encode a negative value")
    encoded = bytearray()
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _field_varint(number: int, value: int) -> bytes:
    return _encode_varint(number << 3) + _encode_varint(value)


def _field_bytes(number: int, value: bytes) -> bytes:
    return _encode_varint((number << 3) | 2) + _encode_varint(len(value)) + value


def _pack_nonnegative_int32_varints(values: Sequence[int]) -> bytes:
    payload = bytearray()
    for value in values:
        if value < 0 or value > 0x7FFFFFFF:
            raise SmokeValidationError("language tokens must be non-negative int32 values")
        payload.extend(_encode_varint(value))
    return bytes(payload)


def _encode_image(width: int, height: int, data: bytes) -> bytes:
    return b"".join(
        (
            _field_varint(1, RGB_U8),
            _field_varint(2, height),
            _field_varint(3, width),
            _field_bytes(4, data),
        )
    )


def encode_request(prepared: dict, request_id: int) -> bytes:
    if request_id < 0 or request_id >= 1 << 64:
        raise SmokeValidationError("request_id must be a uint64")
    fields: list[bytes] = []
    for image in prepared["images"]:
        fields.append(
            _field_bytes(
                1,
                _encode_image(prepared["width"], prepared["height"], image),
            )
        )
    fields.extend(
        (
            _field_bytes(2, _pack_nonnegative_int32_varints(prepared["tokens"])),
            _field_bytes(3, _pack_f32(prepared["state"])),
            _field_bytes(4, _pack_f32(prepared["noise"])),
            _field_varint(5, request_id),
        )
    )
    return b"".join(fields)


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise SmokeValidationError("malformed protobuf varint")


def _iter_fields(data: bytes) -> Iterable[tuple[int, int, bytes | int]]:
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        number = key >> 3
        wire_type = key & 7
        if number == 0:
            raise SmokeValidationError("invalid protobuf field number 0")
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
            yield number, wire_type, value
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise SmokeValidationError("truncated protobuf fixed64 field")
            yield number, wire_type, data[offset:end]
            offset = end
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            end = offset + length
            if end > len(data):
                raise SmokeValidationError("truncated protobuf length-delimited field")
            yield number, wire_type, data[offset:end]
            offset = end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise SmokeValidationError("truncated protobuf fixed32 field")
            yield number, wire_type, data[offset:end]
            offset = end
        else:
            raise SmokeValidationError(f"unsupported protobuf wire type {wire_type}")


def decode_response(data: bytes) -> dict:
    response = {
        "request_id": 0,
        "action_chunk": [],
        "chunk_size": 0,
        "action_dim": 0,
        "latency_ms_total": 0.0,
        "latency_ms_inference": 0.0,
        "latency_ms_prefill": 0.0,
        "latency_ms_denoise": 0.0,
        "latency_ms_vision": 0.0,
        "error": "",
    }
    timing_fields = {
        5: "latency_ms_total",
        6: "latency_ms_inference",
        8: "latency_ms_prefill",
        9: "latency_ms_denoise",
        10: "latency_ms_vision",
    }
    for number, wire_type, value in _iter_fields(data):
        if number == 1 and wire_type == 0:
            response["request_id"] = int(value)
        elif number == 2 and wire_type == 2:
            assert isinstance(value, bytes)
            if len(value) % 4:
                raise SmokeValidationError("packed action_chunk length is not divisible by 4")
            response["action_chunk"].extend(
                struct.unpack(f"<{len(value) // 4}f", value)
            )
        elif number == 2 and wire_type == 5:
            assert isinstance(value, bytes)
            response["action_chunk"].append(struct.unpack("<f", value)[0])
        elif number == 3 and wire_type == 0:
            response["chunk_size"] = int(value)
        elif number == 4 and wire_type == 0:
            response["action_dim"] = int(value)
        elif number in timing_fields and wire_type == 5:
            assert isinstance(value, bytes)
            response[timing_fields[number]] = struct.unpack("<f", value)[0]
        elif number == 7 and wire_type == 2:
            assert isinstance(value, bytes)
            response["error"] = value.decode("utf-8", errors="replace")
    return response


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise SmokeValidationError("cannot summarize an empty measurement set")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise SmokeValidationError("measurement values must be finite and non-empty")
    return {
        "count": len(values),
        "min": min(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
        "stddev": statistics.pstdev(values),
    }


def _validate_response(response: dict, request_id: int, prepared: dict) -> None:
    if response["error"]:
        raise SmokeValidationError(f"server returned error: {response['error']}")
    if response["request_id"] != request_id:
        raise SmokeValidationError(
            f"response request_id={response['request_id']} != expected {request_id}"
        )
    expected_chunk = prepared["chunk_size"]
    expected_dim = prepared["action_dim"]
    if response["chunk_size"] != expected_chunk or response["action_dim"] != expected_dim:
        raise SmokeValidationError(
            "response shape metadata "
            f"[{response['chunk_size']}, {response['action_dim']}] != "
            f"[{expected_chunk}, {expected_dim}]"
        )
    actions = response["action_chunk"]
    if len(actions) != expected_chunk * expected_dim:
        raise SmokeValidationError(
            f"action_chunk has {len(actions)} values; expected {expected_chunk * expected_dim}"
        )
    numeric = actions + [
        response["latency_ms_total"],
        response["latency_ms_inference"],
        response["latency_ms_prefill"],
        response["latency_ms_denoise"],
        response["latency_ms_vision"],
    ]
    if not all(math.isfinite(value) for value in numeric):
        raise SmokeValidationError("response contains NaN or Inf")


def _difference(reference: Sequence[float], current: Sequence[float]) -> tuple[float, float]:
    if len(reference) != len(current):
        raise SmokeValidationError("cannot compare action outputs with different lengths")
    max_absolute = 0.0
    max_relative = 0.0
    for expected, actual in zip(reference, current):
        absolute = abs(actual - expected)
        relative = absolute / max(abs(expected), abs(actual), 1e-30)
        max_absolute = max(max_absolute, absolute)
        max_relative = max(max_relative, relative)
    return max_absolute, max_relative


def _request_record(response: dict, client_wall_ms: float, phase: str, index: int) -> dict:
    actions = response["action_chunk"]
    return {
        "phase": phase,
        "index": index,
        "request_id": response["request_id"],
        "client_wall_ms": client_wall_ms,
        "server_total_ms": response["latency_ms_total"],
        "server_vision_ms": response["latency_ms_vision"],
        "server_inference_ms": response["latency_ms_inference"],
        "server_prefill_ms": response["latency_ms_prefill"],
        "server_denoise_ms": response["latency_ms_denoise"],
        "chunk_size": response["chunk_size"],
        "action_dim": response["action_dim"],
        "action_sha256_le_f32": _sha256(_pack_f32(actions)),
    }


def _validate_loopback_endpoint(endpoint: str) -> None:
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise SmokeValidationError(f"invalid validation endpoint {endpoint!r}") from exc
    if parsed.scheme != "tcp" or parsed.path or parsed.query or parsed.fragment or port is None:
        raise SmokeValidationError("validation endpoint must be tcp://<loopback-host>:<port>")
    host = parsed.hostname
    is_loopback = host == "localhost"
    if host and not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise SmokeValidationError("validation endpoint must use a loopback address")
    if port < 1024:
        raise SmokeValidationError("validation endpoint must use an unprivileged high port")


def run_protocol(
    fixture: dict,
    prepared: dict,
    endpoint: str,
    protocol_name: str,
    timeout_ms: int,
    request_id_base: int,
) -> dict:
    _validate_loopback_endpoint(endpoint)
    if timeout_ms <= 0:
        raise SmokeValidationError("timeout_ms must be positive")
    protocol = fixture["protocols"].get(protocol_name)
    if not isinstance(protocol, dict):
        raise SmokeValidationError(f"unknown protocol {protocol_name!r}")
    warmup_count = int(protocol["warmup_requests"])
    measured_count = int(protocol["measured_requests"])

    try:
        import zmq
    except ImportError as exc:
        raise SmokeValidationError(
            "pyzmq is required only for transport. Install it in the isolated "
            "validation environment; do not modify the system Python environment."
        ) from exc

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(endpoint)
    records: list[dict] = []
    measured_actions: list[list[float]] = []
    try:
        total_requests = warmup_count + measured_count
        for sequence in range(total_requests):
            phase = "warmup" if sequence < warmup_count else "measured"
            index = sequence if phase == "warmup" else sequence - warmup_count
            request_id = request_id_base + sequence
            payload = encode_request(prepared, request_id)
            started = time.perf_counter()
            try:
                socket.send(payload)
                body = socket.recv()
            except zmq.Again as exc:
                raise SmokeValidationError(
                    f"request {request_id} timed out after {timeout_ms} ms"
                ) from exc
            except zmq.ZMQError as exc:
                raise SmokeValidationError(
                    f"request {request_id} failed in ZeroMQ transport: {exc}"
                ) from exc
            client_wall_ms = (time.perf_counter() - started) * 1000.0
            response = decode_response(body)
            _validate_response(response, request_id, prepared)
            records.append(_request_record(response, client_wall_ms, phase, index))
            if phase == "measured":
                measured_actions.append(response["action_chunk"])
    finally:
        socket.close(linger=0)
        context.term()

    reference = measured_actions[0]
    differences = [_difference(reference, values) for values in measured_actions]
    for record, (absolute, relative) in zip(
        (item for item in records if item["phase"] == "measured"),
        differences,
    ):
        record["max_abs_diff_from_first"] = absolute
        record["max_rel_diff_from_first"] = relative

    measured_records = [item for item in records if item["phase"] == "measured"]
    hashes = [item["action_sha256_le_f32"] for item in measured_records]
    exact_repeat = len(set(hashes)) == 1 and all(
        absolute == 0.0 and relative == 0.0 for absolute, relative in differences
    )
    metrics = {
        key: summarize_values([float(item[key]) for item in measured_records])
        for key in (
            "client_wall_ms",
            "server_total_ms",
            "server_vision_ms",
            "server_inference_ms",
            "server_prefill_ms",
            "server_denoise_ms",
        )
    }
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "status": "pass" if exact_repeat else "fail",
        "failure": None if exact_repeat else "measured action outputs are not bit-identical",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "endpoint": endpoint,
        "protocol": protocol_name,
        "fixture_name": fixture.get("name"),
        "fixture_sha256": prepared["fixture_sha256"],
        "warmup_requests": warmup_count,
        "measured_requests": measured_count,
        "exact_repeatability": exact_repeat,
        "metrics_ms": metrics,
        "requests": records,
    }


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _fixture_summary(fixture: dict, prepared: dict) -> dict:
    return {
        "status": "valid",
        "fixture_name": fixture.get("name"),
        "fixture_sha256": prepared["fixture_sha256"],
        "image_sha256": [_sha256(image) for image in prepared["images"]],
        "language_sha256_le_i32": _sha256(_pack_i32(prepared["tokens"])),
        "state_sha256_le_f32": _sha256(_pack_f32(prepared["state"])),
        "noise_sha256_le_f32": _sha256(_pack_f32(prepared["noise"])),
        "request_shape": {
            "images": len(prepared["images"]),
            "image_height": prepared["height"],
            "image_width": prepared["width"],
            "language_tokens": len(prepared["tokens"]),
            "state": len(prepared["state"]),
            "noise": len(prepared["noise"]),
        },
        "expected_response_shape": [prepared["chunk_size"], prepared["action_dim"]],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_fixture = Path(__file__).resolve().parents[1] / "fixtures" / "pi05_smoke.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=default_fixture)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="validate and summarize the fixture offline")
    run = subparsers.add_parser("run", help="send the deterministic request protocol")
    run.add_argument("--protocol", choices=("functional", "performance"), default="functional")
    run.add_argument("--endpoint", default="tcp://127.0.0.1:15555")
    run.add_argument("--timeout-ms", type=int, default=900_000)
    run.add_argument("--request-id-base", type=int, default=10_000)
    run.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        fixture, prepared = load_fixture(args.fixture)
        if args.command == "validate":
            print(json.dumps(_fixture_summary(fixture, prepared), indent=2, sort_keys=True))
            return 0
        result = run_protocol(
            fixture,
            prepared,
            args.endpoint,
            args.protocol,
            args.timeout_ms,
            args.request_id_base,
        )
        _write_json_atomic(args.output, result)
        print(json.dumps({
            "status": result["status"],
            "output": str(args.output.resolve()),
            "fixture_sha256": result["fixture_sha256"],
            "exact_repeatability": result["exact_repeatability"],
        }, indent=2, sort_keys=True))
        return 0 if result["status"] == "pass" else 1
    except (OSError, SmokeValidationError, ValueError) as exc:
        print(f"pi05-smoke: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
