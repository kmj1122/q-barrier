import numpy as np
import sys
import types

sys.modules.setdefault("safety_gymnasium", types.SimpleNamespace(make=None))

from envs.mujoco.mujoco import MujocoWrapper


class _FakeStepEnv:
    def step(self, action):
        return (
            np.array([1.0, 2.0], dtype=np.float32),
            3.0,
            1.5,
            False,
            False,
            {},
        )


def test_step_returns_soft_reset_observation_when_inner_episode_ends():
    wrapper = object.__new__(MujocoWrapper)
    wrapper.env = _FakeStepEnv()
    wrapper.step_count = 0
    wrapper.time_limit = 1
    wrapper.reward_normalization_factor = 1.0
    wrapper.current_cost_to_go = 5.0
    wrapper.initial_cost_to_go = 10.0
    wrapper.current_episode_count = 0
    wrapper.max_episodes = 3
    wrapper.task_type = "goal"
    wrapper.current_velocity_threshold = None

    def fake_soft_reset():
        wrapper.current_cost_to_go = np.float32(wrapper.initial_cost_to_go)
        wrapper.step_count = 0
        return np.array([42.0, 99.0], dtype=np.float32)

    def fake_get_obs(env_obs):
        return {
            "observation": np.asarray(env_obs, dtype=np.float32),
            "cost_to_go": np.float32(wrapper.current_cost_to_go),
        }

    def fake_get_info(soft_reset, cost, original_reward=None):
        return {
            "soft_reset": soft_reset,
            "cost": cost,
            "cost_limit": np.float32(wrapper.initial_cost_to_go),
        }

    wrapper._soft_reset = fake_soft_reset
    wrapper._get_obs = fake_get_obs
    wrapper._get_info = fake_get_info

    obs, reward, terminated, truncated, info = MujocoWrapper.step(
        wrapper, np.array([0.0], dtype=np.float32)
    )

    assert reward == 3.0
    assert not terminated
    assert not truncated
    assert info["soft_reset"] == 1
    np.testing.assert_allclose(obs["observation"], np.array([42.0, 99.0], dtype=np.float32))
    assert obs["cost_to_go"] == np.float32(10.0)
