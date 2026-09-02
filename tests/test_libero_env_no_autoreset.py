"""Regression coverage for LIBERO environment episode handling.

A successful step must terminate the episode itself instead of silently
resetting into the next one, which would otherwise skip the terminal
observation that the evaluation client records for success detection.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np

    from eval.sim.libero.libero_env import LiberoEnv
except ModuleNotFoundError as exc:  # pragma: no cover - optional eval deps
    raise unittest.SkipTest(
        "LIBERO simulator dependencies are unavailable; install the evaluation "
        "requirements (see eval/sim/INSTALL_NOTES.md) to run this test"
    ) from exc


class _SuccessfulEnv:
    """Minimal stand-in whose step always reports success."""

    def step(self, action):
        return {}, 1.0, False, {}

    def check_success(self):
        return True


class LiberoEnvNoAutoResetTest(unittest.TestCase):
    def test_successful_step_does_not_advance_episode(self):
        env = LiberoEnv.__new__(LiberoEnv)
        env._env = _SuccessfulEnv()
        env._step_id = 0
        env._max_episode_steps = 10
        env.task = "test"
        env.task_id = 0
        env._format_raw_obs = lambda raw: raw
        env._append_video_frame = lambda obs: None
        env._finalize_episode_video = lambda: None
        env.reset = lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("step() must not reset the next episode")
        )

        _, _, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])


if __name__ == "__main__":
    unittest.main()
