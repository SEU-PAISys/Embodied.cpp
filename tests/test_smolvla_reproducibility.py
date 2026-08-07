from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))

from client.reproducibility import derive_episode_noise_seed, generate_action_noise


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


if __name__ == "__main__":
    unittest.main()
