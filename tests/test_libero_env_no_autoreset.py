import numpy as np

from eval.sim.libero.libero_env import LiberoEnv


class _SuccessfulEnv:
    def step(self, action):
        return {}, 1.0, False, {}

    def check_success(self):
        return True


def test_successful_step_does_not_advance_episode():
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

    assert terminated and not truncated and info["is_success"]
