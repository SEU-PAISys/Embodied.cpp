from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval" / "sim" / "libero" / "libero_env.py"


def _module(name: str, **attributes):
    result = types.ModuleType(name)
    result.__dict__.update(attributes)
    result.__path__ = []
    return result


class _GymEnv:
    def reset(self, seed=None):
        self.test_reset_seed = seed


_spaces = types.SimpleNamespace(
    Box=lambda **kwargs: kwargs,
    Dict=lambda value: value,
    Text=lambda **kwargs: kwargs,
)
_gym = _module("gymnasium", Env=_GymEnv, spaces=_spaces)
_gym_envs = _module("gymnasium.envs")
_registration = _module("gymnasium.envs.registration", register=lambda **kwargs: None)
_gym_envs.registration = _registration
_imageio_v2 = _module("imageio.v2", get_writer=lambda *args, **kwargs: None)
_imageio = _module("imageio", v2=_imageio_v2)
_torch = _module("torch", load=lambda *args, **kwargs: None)
_benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {})
_libero_core = _module(
    "libero.libero", benchmark=_benchmark, get_libero_path=lambda name: ""
)
_libero_envs = _module("libero.libero.envs", OffScreenRenderEnv=type("OffScreenRenderEnv", (), {}))
_libero = _module("libero")

_external_stubs = {
    "gymnasium": _gym,
    "gymnasium.envs": _gym_envs,
    "gymnasium.envs.registration": _registration,
    "imageio": _imageio,
    "imageio.v2": _imageio_v2,
    "torch": _torch,
    "libero": _libero,
    "libero.libero": _libero_core,
    "libero.libero.envs": _libero_envs,
}
_spec = importlib.util.spec_from_file_location("libero_env_settling_test_target", SOURCE)
assert _spec is not None and _spec.loader is not None
LIBERO_ENV = importlib.util.module_from_spec(_spec)
with patch.dict(sys.modules, _external_stubs):
    _spec.loader.exec_module(LIBERO_ENV)


class _SettlingActionEnv:
    def __init__(self):
        self.actions = []
        self.robots = []

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        return {"reset": True}

    def step(self, action):
        self.actions.append(list(action))
        return {"step": len(self.actions)}, 0.0, False, {}


class LiberoSettlingActionTests(unittest.TestCase):
    def _make_env(self, **kwargs):
        inner = _SettlingActionEnv()
        with patch.object(LIBERO_ENV, "_get_suite", return_value=object()), \
             patch.object(LIBERO_ENV.LiberoEnv, "_make_envs_task", return_value=inner):
            env = LIBERO_ENV.LiberoEnv(
                "libero_object", 0, init_states=False, num_steps_wait=10, **kwargs
            )
        env._format_raw_obs = lambda observation: observation
        env._append_video_frame = lambda observation: None
        env._finalize_episode_video = lambda: None
        return env, inner

    def test_default_remains_zero_and_explicit_minus_one_is_used_for_all_settling_steps(self):
        default_env, default_inner = self._make_env()
        self.assertEqual(default_env.settling_gripper_action, 0.0)
        default_env.reset(seed=3)
        self.assertEqual(len(default_inner.actions), 10)
        self.assertTrue(all(action == [0.0] * 7 for action in default_inner.actions))

        smolvla_env, smolvla_inner = self._make_env(settling_gripper_action=-1.0)
        smolvla_env.reset(seed=3)
        expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
        self.assertEqual(len(smolvla_inner.actions), 10)
        self.assertTrue(all(action == expected for action in smolvla_inner.actions))

    def test_constructor_rejects_non_finite_or_out_of_range_settling_action(self):
        for value in (float("nan"), float("inf"), -1.01, 1.01):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "settling_gripper_action"
            ):
                self._make_env(settling_gripper_action=value)


if __name__ == "__main__":
    unittest.main()
