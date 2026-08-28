from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import re
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "eval" / "client" / "run_pi05_smoke.py"
FIXTURE = REPO / "eval" / "fixtures" / "pi05_smoke.json"
PROTO = REPO / "serving" / "vla.proto"

spec = importlib.util.spec_from_file_location("run_pi05_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(smoke)


def _proto_message_body(source: str, name: str) -> str:
    match = re.search(rf"\bmessage\s+{re.escape(name)}\s*\{{", source)
    if match is None:
        raise AssertionError(f"message {name} not found in {PROTO}")
    start = match.end()
    depth = 1
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index]
    raise AssertionError(f"message {name} is missing its closing brace")


def _proto_fields(source: str, message: str) -> dict[str, tuple[str, str, int]]:
    body = _proto_message_body(source, message)
    fields: dict[str, tuple[str, str, int]] = {}
    pattern = re.compile(
        r"^\s*(?:(repeated)\s+)?([A-Za-z_]\w*)\s+([A-Za-z_]\w*)"
        r"\s*=\s*(\d+)\s*;",
        re.MULTILINE,
    )
    for match in pattern.finditer(body):
        cardinality, field_type, name, number = match.groups()
        fields[name] = (cardinality or "singular", field_type, int(number))
    return fields


class Pi05SmokeTests(unittest.TestCase):
    def test_manual_codec_contract_matches_proto_schema(self) -> None:
        source = PROTO.read_text(encoding="utf-8")
        expected = {
            "Image": {
                "encoding": ("singular", "Encoding", 1),
                "height": ("singular", "uint32", 2),
                "width": ("singular", "uint32", 3),
                "data": ("singular", "bytes", 4),
            },
            "PredictRequest": {
                "images": ("repeated", "Image", 1),
                "lang_tokens": ("repeated", "int32", 2),
                "state": ("repeated", "float", 3),
                "noise": ("repeated", "float", 4),
                "request_id": ("singular", "uint64", 5),
            },
            "PredictResponse": {
                "request_id": ("singular", "uint64", 1),
                "action_chunk": ("repeated", "float", 2),
                "chunk_size": ("singular", "uint32", 3),
                "action_dim": ("singular", "uint32", 4),
                "latency_ms_total": ("singular", "float", 5),
                "latency_ms_inference": ("singular", "float", 6),
                "error": ("singular", "string", 7),
                "latency_ms_prefill": ("singular", "float", 8),
                "latency_ms_denoise": ("singular", "float", 9),
                "latency_ms_vision": ("singular", "float", 10),
            },
        }
        for message, contract in expected.items():
            fields = _proto_fields(source, message)
            with self.subTest(message=message):
                self.assertEqual(
                    {name: fields.get(name) for name in contract},
                    contract,
                )
        image_body = _proto_message_body(source, "Image")
        rgb_u8 = re.search(r"\bRGB_U8\s*=\s*(\d+)\s*;", image_body)
        self.assertIsNotNone(rgb_u8)
        assert rgb_u8 is not None
        self.assertEqual(int(rgb_u8.group(1)), smoke.RGB_U8)

    def test_fixture_is_complete_and_deterministic(self) -> None:
        fixture, prepared = smoke.load_fixture(FIXTURE)
        self.assertEqual(fixture["name"], "pi05-jetson-smoke-v1")
        self.assertEqual(len(prepared["images"]), 2)
        self.assertEqual(len(prepared["images"][0]), 224 * 224 * 3)
        self.assertEqual(len(prepared["tokens"]), 200)
        self.assertEqual(
            prepared["tokens"][:7],
            [2, 18075, 908, 573, 3118, 28660, 108],
        )
        self.assertEqual(prepared["tokens"][7:], [0] * 193)
        self.assertEqual(len(prepared["state"]), 32)
        self.assertEqual(len(prepared["noise"]), 50 * 32)
        self.assertEqual(prepared["noise"], smoke.generate_noise(20260828, 1600))

    def test_fixture_sha_is_independent_of_json_line_endings(self) -> None:
        raw = FIXTURE.read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        crlf = lf.replace(b"\n", b"\r\n")
        self.assertNotEqual(lf, crlf)
        with tempfile.TemporaryDirectory() as directory:
            lf_path = Path(directory) / "fixture-lf.json"
            crlf_path = Path(directory) / "fixture-crlf.json"
            lf_path.write_bytes(lf)
            crlf_path.write_bytes(crlf)
            _, lf_prepared = smoke.load_fixture(lf_path)
            _, crlf_prepared = smoke.load_fixture(crlf_path)
        self.assertEqual(
            lf_prepared["fixture_sha256"],
            crlf_prepared["fixture_sha256"],
        )

    def test_request_contains_expected_top_level_fields(self) -> None:
        _, prepared = smoke.load_fixture(FIXTURE)
        request = smoke.encode_request(prepared, 12345)
        fields = list(smoke._iter_fields(request))
        self.assertEqual([number for number, _, _ in fields], [1, 1, 2, 3, 4, 5])
        self.assertEqual(fields[-1], (5, 0, 12345))
        token_payload = fields[2][2]
        self.assertIsInstance(token_payload, bytes)
        decoded_tokens = []
        offset = 0
        while offset < len(token_payload):
            value, offset = smoke._read_varint(token_payload, offset)
            decoded_tokens.append(value)
        self.assertEqual(decoded_tokens, prepared["tokens"])
        self.assertEqual(len(fields[3][2]), 32 * 4)
        self.assertEqual(len(fields[4][2]), 1600 * 4)

    def test_response_decoder_accepts_packed_actions_and_unknown_fields(self) -> None:
        action = [1.25, -2.5]
        body = b"".join(
            (
                smoke._field_varint(1, 77),
                smoke._field_bytes(2, struct.pack("<2f", *action)),
                smoke._field_varint(3, 1),
                smoke._field_varint(4, 2),
                smoke._encode_varint((5 << 3) | 5),
                struct.pack("<f", 12.5),
                smoke._encode_varint((6 << 3) | 5),
                struct.pack("<f", 8.5),
                smoke._encode_varint((8 << 3) | 5),
                struct.pack("<f", 3.5),
                smoke._encode_varint((9 << 3) | 5),
                struct.pack("<f", 5.0),
                smoke._encode_varint((10 << 3) | 5),
                struct.pack("<f", 4.0),
                smoke._field_bytes(99, b"ignored"),
            )
        )
        response = smoke.decode_response(body)
        self.assertEqual(response["request_id"], 77)
        self.assertEqual(response["action_chunk"], action)
        self.assertEqual(response["chunk_size"], 1)
        self.assertEqual(response["action_dim"], 2)
        self.assertEqual(response["latency_ms_total"], 12.5)
        self.assertEqual(response["latency_ms_inference"], 8.5)
        self.assertEqual(response["latency_ms_prefill"], 3.5)
        self.assertEqual(response["latency_ms_denoise"], 5.0)
        self.assertEqual(response["latency_ms_vision"], 4.0)

    def test_response_validation_rejects_shape_mismatch(self) -> None:
        response = {
            "request_id": 5,
            "error": "",
            "chunk_size": 49,
            "action_dim": 32,
            "action_chunk": [0.0] * (49 * 32),
            "latency_ms_total": 1.0,
            "latency_ms_inference": 1.0,
            "latency_ms_prefill": 0.0,
            "latency_ms_denoise": 0.0,
            "latency_ms_vision": 0.0,
        }
        with self.assertRaises(smoke.SmokeValidationError):
            smoke._validate_response(
                response,
                5,
                {"chunk_size": 50, "action_dim": 32},
            )

    def test_response_validation_rejects_nonfinite_or_negative_values(self) -> None:
        baseline = {
            "request_id": 5,
            "error": "",
            "chunk_size": 1,
            "action_dim": 1,
            "action_chunk": [0.0],
            "latency_ms_total": 1.0,
            "latency_ms_inference": 1.0,
            "latency_ms_prefill": 0.0,
            "latency_ms_denoise": 0.0,
            "latency_ms_vision": 0.0,
        }
        invalid_values = (
            ("action_chunk", math.nan),
            ("latency_ms_total", math.nan),
            ("latency_ms_inference", math.inf),
            ("latency_ms_vision", -0.1),
        )
        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                response = baseline.copy()
                response["action_chunk"] = list(baseline["action_chunk"])
                if field == "action_chunk":
                    response[field][0] = value
                else:
                    response[field] = value
                with self.assertRaises(smoke.SmokeValidationError):
                    smoke._validate_response(
                        response,
                        5,
                        {"chunk_size": 1, "action_dim": 1},
                    )

    def test_summary_uses_population_standard_deviation(self) -> None:
        summary = smoke.summarize_values([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["p50"], 2.0)
        self.assertAlmostEqual(summary["p95"], 2.9)
        self.assertAlmostEqual(summary["stddev"], (2.0 / 3.0) ** 0.5)

    def test_endpoint_must_be_loopback_and_unprivileged(self) -> None:
        smoke._validate_loopback_endpoint("tcp://127.0.0.1:15555")
        smoke._validate_loopback_endpoint("tcp://localhost:25555")
        smoke._validate_loopback_endpoint("tcp://[::1]:35555")
        with self.assertRaises(smoke.SmokeValidationError):
            smoke._validate_loopback_endpoint("tcp://192.0.2.1:15555")
        with self.assertRaises(smoke.SmokeValidationError):
            smoke._validate_loopback_endpoint("tcp://127.0.0.1:22")

    def test_functional_protocol_requires_exact_repeatability(self) -> None:
        fixture, prepared = smoke.load_fixture(FIXTURE)

        class FakeSocket:
            def __init__(self) -> None:
                self.request_id = 0

            def setsockopt(self, *_args) -> None:
                pass

            def connect(self, endpoint: str) -> None:
                self.endpoint = endpoint

            def send(self, payload: bytes) -> None:
                request_fields = list(smoke._iter_fields(payload))
                self.request_id = int(request_fields[-1][2])

            def recv(self) -> bytes:
                fixed32 = lambda number, value: (
                    smoke._encode_varint((number << 3) | 5) + struct.pack("<f", value)
                )
                return b"".join(
                    (
                        smoke._field_varint(1, self.request_id),
                        smoke._field_bytes(2, struct.pack("<1600f", *([0.25] * 1600))),
                        smoke._field_varint(3, 50),
                        smoke._field_varint(4, 32),
                        fixed32(5, 10.0),
                        fixed32(6, 8.0),
                        fixed32(8, 3.0),
                        fixed32(9, 5.0),
                        fixed32(10, 2.0),
                    )
                )

            def close(self, linger: int = 0) -> None:
                self.linger = linger

        fake_socket = FakeSocket()
        fake_context = SimpleNamespace(
            socket=lambda _kind: fake_socket,
            term=lambda: None,
        )
        fake_zmq = SimpleNamespace(
            Context=lambda: fake_context,
            REQ=1,
            LINGER=2,
            RCVTIMEO=3,
            SNDTIMEO=4,
            Again=TimeoutError,
            ZMQError=RuntimeError,
        )
        with mock.patch.dict(sys.modules, {"zmq": fake_zmq}):
            result = smoke.run_protocol(
                fixture,
                prepared,
                "tcp://127.0.0.1:15555",
                "functional",
                1000,
                10000,
            )
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["exact_repeatability"])
        self.assertEqual(result["warmup_requests"], 1)
        self.assertEqual(result["measured_requests"], 3)
        self.assertEqual(len(result["requests"]), 4)
        self.assertEqual(result["metrics_ms"]["server_total_ms"]["p50"], 10.0)
        self.assertEqual(result["metrics_ms"]["server_inference_ms"]["p50"], 8.0)
        self.assertEqual(result["metrics_ms"]["server_prefill_ms"]["p50"], 3.0)
        self.assertEqual(result["metrics_ms"]["server_denoise_ms"]["p50"], 5.0)
        self.assertEqual(result["metrics_ms"]["server_vision_ms"]["p50"], 2.0)


if __name__ == "__main__":
    unittest.main()
