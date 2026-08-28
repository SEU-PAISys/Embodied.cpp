from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "eval" / "client" / "run_pi05_smoke.py"
FIXTURE = REPO / "eval" / "fixtures" / "pi05_smoke.json"

spec = importlib.util.spec_from_file_location("run_pi05_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(smoke)


class Pi05SmokeTests(unittest.TestCase):
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
                smoke._field_bytes(99, b"ignored"),
            )
        )
        response = smoke.decode_response(body)
        self.assertEqual(response["request_id"], 77)
        self.assertEqual(response["action_chunk"], action)
        self.assertEqual(response["chunk_size"], 1)
        self.assertEqual(response["action_dim"], 2)
        self.assertEqual(response["latency_ms_total"], 12.5)

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
            smoke._validate_loopback_endpoint("tcp://30.78.54.119:15555")
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


if __name__ == "__main__":
    unittest.main()
