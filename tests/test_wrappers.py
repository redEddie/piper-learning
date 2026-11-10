"""Fast CPU regression for timeout/termination bookkeeping and slot isolation."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jp
from mujoco_playground._src.mjx_env import State
import numpy as np
import pytest

from piper_sim_lab.wrappers import wrap_for_training


class ToyTask:
    """Count physics steps and use a random goal to expose stale reset info."""

    def reset(self, key):
        goal = jax.random.uniform(key)
        zero = jp.float32(0)
        return State(
            data=jp.zeros(1), obs=jp.array([goal, 0.0]), reward=zero, done=zero,
            metrics={"probe": zero},
            info={"goal": goal, "previous_action": zero,
                  "AutoResetWrapper_preserve_info": {"curriculum": zero}},
        )

    def step(self, state, action):
        terminate = ~jp.all(jp.isfinite(action))
        action = jp.nan_to_num(action)[0]
        info = dict(state.info)
        info["previous_action"] = action
        preserved = dict(info["AutoResetWrapper_preserve_info"])
        preserved["curriculum"] = preserved["curriculum"] + 1
        info["AutoResetWrapper_preserve_info"] = preserved
        return state.replace(
            data=state.data + 1, obs=jp.array([info["goal"], action]),
            reward=jp.float32(1), done=terminate.astype(jp.float32),
            metrics={"probe": jp.float32(3)}, info=info,
        )


def test_timeout_keeps_completed_metrics_and_starts_a_clean_episode():
    env = wrap_for_training(ToyTask(), episode_length=2, action_repeat=1)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.split(jax.random.PRNGKey(10), 2))
    goals = np.asarray(state.info["goal"]).copy()
    for _ in range(2):
        state = step(state, jp.full((2, 1), 0.4))
    np.testing.assert_array_equal(state.done, [1, 1])
    np.testing.assert_array_equal(state.info["truncation"], [1, 1])
    np.testing.assert_array_equal(state.info["episode_done"], [1, 1])
    np.testing.assert_array_equal(state.info["episode_metrics"]["length"], [2, 2])
    np.testing.assert_array_equal(state.info["episode_metrics"]["sum_reward"], [2, 2])
    np.testing.assert_array_equal(state.info["episode_metrics"]["probe"], [6, 6])
    np.testing.assert_array_equal(state.info["AutoResetWrapper_done_count"], [1, 1])
    np.testing.assert_array_equal(state.data, np.zeros((2, 1)))
    np.testing.assert_array_equal(state.info["previous_action"], [0, 0])
    np.testing.assert_array_equal(state.obs[:, 0], state.info["goal"])
    assert np.all(goals != np.asarray(state.info["goal"]))
    np.testing.assert_array_equal(
        state.info["AutoResetWrapper_preserve_info"]["curriculum"], [2, 2],
    )
    state = step(state, jp.full((2, 1), 0.2))
    np.testing.assert_array_equal(state.done, [0, 0])
    np.testing.assert_array_equal(state.info["truncation"], [0, 0])
    np.testing.assert_array_equal(state.info["steps"], [1, 1])
    np.testing.assert_array_equal(state.info["episode_done"], [0, 0])
    np.testing.assert_array_equal(state.info["episode_metrics"]["length"], [1, 1])
    np.testing.assert_array_equal(state.info["episode_metrics"]["sum_reward"], [1, 1])
    np.testing.assert_array_equal(state.data, np.ones((2, 1)))


def test_one_termination_does_not_reset_other_slot_or_become_a_timeout():
    env = wrap_for_training(ToyTask(), episode_length=3, action_repeat=1)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.split(jax.random.PRNGKey(20), 2))
    state = step(state, jp.full((2, 1), 0.1))
    goals = np.asarray(state.info["goal"]).copy()
    state = step(state, jp.array([[jp.nan], [0.7]]))
    np.testing.assert_array_equal(state.done, [1, 0])
    np.testing.assert_array_equal(state.info["truncation"], [0, 0])
    np.testing.assert_array_equal(state.info["episode_done"], [1, 0])
    np.testing.assert_array_equal(state.info["AutoResetWrapper_done_count"], [1, 0])
    np.testing.assert_array_equal(state.info["episode_metrics"]["sum_reward"], [2, 2])
    np.testing.assert_array_equal(state.data[:, 0], [0, 2])
    np.testing.assert_allclose(state.info["previous_action"], [0, 0.7])
    assert state.info["goal"][0] != goals[0]
    assert state.info["goal"][1] == goals[1]
    np.testing.assert_array_equal(state.obs[:, 0], state.info["goal"])
    state = step(state, jp.full((2, 1), 0.3))
    np.testing.assert_array_equal(state.done, [0, 1])
    np.testing.assert_array_equal(state.info["truncation"], [0, 1])
    np.testing.assert_array_equal(state.info["episode_metrics"]["length"], [1, 3])
    np.testing.assert_array_equal(state.info["episode_metrics"]["sum_reward"], [1, 3])
    np.testing.assert_array_equal(state.info["AutoResetWrapper_done_count"], [1, 1])
    np.testing.assert_array_equal(state.data[:, 0], [1, 0])


def test_termination_on_last_step_has_no_timeout_bootstrap_mask():
    env = wrap_for_training(ToyTask(), episode_length=1, action_repeat=1)
    state = env.reset(jax.random.split(jax.random.PRNGKey(3), 2))
    state = jax.jit(env.step)(state, jp.array([[jp.nan], [0.0]]))
    np.testing.assert_array_equal(state.done, [1, 1])
    np.testing.assert_array_equal(state.info["truncation"], [0, 1])
    np.testing.assert_array_equal(state.info["episode_done"], [1, 1])


def test_action_repeat_preserves_episode_length_and_reward_accounting():
    env = wrap_for_training(ToyTask(), episode_length=4, action_repeat=2)
    state = env.reset(jax.random.split(jax.random.PRNGKey(3), 1))
    step = jax.jit(env.step)
    state = step(state, jp.zeros((1, 1)))
    np.testing.assert_array_equal(state.data[:, 0], [2])
    np.testing.assert_array_equal(state.reward, [2])
    state = step(state, jp.zeros((1, 1)))
    np.testing.assert_array_equal(state.info["episode_metrics"]["length"], [4])
    np.testing.assert_array_equal(state.info["episode_metrics"]["sum_reward"], [4])
    np.testing.assert_array_equal(state.info["truncation"], [1])


def test_invalid_episode_length_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        wrap_for_training(ToyTask(), episode_length=0)
