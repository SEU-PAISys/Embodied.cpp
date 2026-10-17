from __future__ import annotations

import sys
from collections import deque
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from client.reproducibility import derive_episode_noise_seed, generate_action_noise
from client.reproducibility import noise_checksum
try:
    from client.vla_cpp_client import VlaCppClient
except ModuleNotFoundError as exc:
    if exc.name != "zmq":
        raise
    # The tests exercise the request/profile path through __new__, so a
    # lightweight import stub keeps this focused regression independent of the
    # optional runtime transport package.
    sys.modules["zmq"] = SimpleNamespace()
    from client.vla_cpp_client import VlaCppClient


def _fake_smolvla_client():
    client = VlaCppClient.__new__(VlaCppClient)
    client.arch = "smolvla"
    client.image_size = 512
    client.max_state_dim = 32
    client.max_length = 48
    client.n_action_steps = 1
    client.image_keys = ["observation.images.image", "observation.images.image2"]
    client.real_action_dim = 7
    client.noise_chunk_size = 50
    client.noise_action_dim = 32
    client.use_server_tokenizer = False
    client._initial_noise_seed = None
    client._episode_noise_seed = None
    client._noise_rng = None
    client._noise_meta = None
    client._step = 0
    client._inference_sequence = 0
    client._last_response = None
    client._last_inference_profile = None
    client._action_queue = deque(maxlen=1)

    class Request:
        def __init__(self):
            self.images = SimpleNamespace(add=lambda: SimpleNamespace())
            self.lang_tokens = []
            self.attention_mask = []
            self.state = []
            self.noise = []

        def SerializeToString(self):
            return b"request"

    response = SimpleNamespace(
        error="",
        request_id=0,
        chunk_size=50,
        action_dim=7,
        action_chunk=np.zeros(50 * 7, dtype=np.float32),
        latency_ms_total=10.0,
        latency_ms_vision=2.0,
        latency_ms_inference=8.0,
        latency_ms_prefill=3.0,
        latency_ms_denoise=5.0,
        ParseFromString=lambda body: None,
    )
    client.pb = SimpleNamespace(
        PredictRequest=Request,
        PredictResponse=lambda: response,
        Image=SimpleNamespace(RGB_U8=0, F32_RGB_01=1),
    )
    client.sock = MagicMock()
    client.sock.recv.return_value = b"response"
    client.tok = lambda *args, **kwargs: {
        "input_ids": np.array([[0, 1, 2]], dtype=np.int32),
        "attention_mask": np.array([[1, 1, 1]], dtype=np.uint32),
    }
    observations = {
        "observation.images.image": np.zeros((3, 512, 512), dtype=np.float32),
        "observation.images.image2": np.zeros((3, 512, 512), dtype=np.float32),
        "observation.state": np.zeros(32, dtype=np.float32),
        "task": "pick up the bowl",
    }
    return client, observations


class ReproducibilityTests(unittest.TestCase):
    def test_episode_seed_is_stable_and_suite_specific(self) -> None:
        seed = derive_episode_noise_seed(1000, "libero_spatial", 0, 0)
        self.assertEqual(seed, derive_episode_noise_seed(1000, "libero_spatial", 0, 0))
        self.assertNotEqual(seed, derive_episode_noise_seed(1000, "libero_object", 0, 0))
        self.assertNotEqual(seed, derive_episode_noise_seed(1000, "libero_spatial", 1, 0))
        self.assertNotEqual(seed, derive_episode_noise_seed(1000, "libero_spatial", 0, 1))

    def test_episode_seed_rejects_negative_components(self) -> None:
        with self.assertRaises(ValueError):
            derive_episode_noise_seed(-1, "libero_spatial", 0, 0)

    def test_fixed_seed_reproduces_noise_and_action_payload(self) -> None:
        first = generate_action_noise(np.random.default_rng(123), 50, 32)
        second = generate_action_noise(np.random.default_rng(123), 50, 32)
        different = generate_action_noise(np.random.default_rng(124), 50, 32)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        self.assertEqual(first.shape, (50, 32))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(first.flags.c_contiguous)

    def test_noise_dimensions_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            generate_action_noise(np.random.default_rng(0), 0, 32)

    def test_generic_smolvla_seeded_noise_profile_is_complete_and_stable(self) -> None:
        client, observations = _fake_smolvla_client()
        client.reset(noise_seed=11)
        client._predict_chunk(observations)
        first = client.get_last_inference_profile()
        self.assertEqual(first["noise_mode"], "derived")
        self.assertEqual(first["noise_seed"], 11)
        self.assertEqual(first["noise_device"], "cpu")
        self.assertEqual(first["noise_dtype"], "float32")
        self.assertRegex(first["noise_checksum"], r"^[0-9a-f]{16}$")

        client.reset(noise_seed=11)
        client._predict_chunk(observations)
        second = client.get_last_inference_profile()
        self.assertEqual(first["noise_checksum"], second["noise_checksum"])

    def test_generic_smolvla_explicit_noise_profile_records_canonical_payload(self) -> None:
        client, observations = _fake_smolvla_client()
        explicit = np.arange(50 * 32, dtype=np.float64).reshape(50, 32)
        observations = dict(observations, action_noise=explicit)
        client.reset()
        client._predict_chunk(observations)
        profile = client.get_last_inference_profile()
        self.assertEqual(profile["noise_mode"], "explicit")
        self.assertIsNone(profile["noise_seed"])
        self.assertEqual(profile["noise_device"], "cpu")
        self.assertEqual(profile["noise_dtype"], "float32")
        self.assertEqual(
            profile["noise_checksum"],
            noise_checksum(explicit.astype(np.float32)),
        )


if __name__ == "__main__":
    unittest.main()
