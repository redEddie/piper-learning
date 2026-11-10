"""CPU coverage for fail-fast checks without a robot or GPU compilation."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jp
from mujoco_playground._src.mjx_env import State
import numpy as np
import pytest

from piper_sim_lab import wrappers


class FlagTask:
    def __init__(self, reset_overflow=False):
        self.reset_overflow = reset_overflow

    def reset(self, key):
        del key
        zero = jp.float32(0)
        return State(
            data=jp.zeros(1), obs=jp.zeros(1), reward=zero, done=zero,
            metrics={"nonfinite": zero, "capacity_overflow": jp.float32(self.reset_overflow)},
            info={},
        )

    def step(self, state, action):
        nonfinite = (action[0] == 1).astype(jp.float32)
        overflow = (action[0] == 2).astype(jp.float32)
        return state.replace(
            data=state.data + 1, reward=jp.float32(1),
            done=jp.maximum(nonfinite, overflow),
            metrics={"nonfinite": nonfinite, "capacity_overflow": overflow},
        )


def test_healthy_steps_do_not_call_host_failure_callback(monkeypatch):
    calls = []
    monkeypatch.setattr(wrappers, "_raise_physics_error", lambda *flags: calls.append(flags))
    env = wrappers.wrap_for_training(FlagTask(), episode_length=2)
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 3))
    step = jax.jit(env.step)
    for _ in range(3):
        state = step(state, jp.zeros((3, 1)))
    jax.block_until_ready(state)
    jax.effects_barrier()
    assert calls == []
    np.testing.assert_array_equal(state.info["AutoResetWrapper_done_count"], [1, 1, 1])


@pytest.mark.parametrize("action, metric", [(1, "nonfinite"), (2, "capacity_overflow")])
def test_failure_in_one_slot_rejects_step_before_autoreset(action, metric):
    env = wrappers.wrap_for_training(FlagTask(), episode_length=2)
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(1), 2))
    with pytest.raises(Exception, match="Invalid Piper physics: " + metric):
        jax.block_until_ready(jax.jit(env.step)(state, jp.array([[0.0], [float(action)]])))


def test_initial_reset_overflow_is_rejected():
    env = wrappers.wrap_for_training(FlagTask(reset_overflow=True))
    with pytest.raises(Exception, match="Invalid Piper physics: capacity_overflow"):
        jax.block_until_ready(jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(2), 2)))


def test_explicit_opt_out_keeps_failure_metadata_for_reset_tests():
    env = wrappers.wrap_for_training(FlagTask(), strict_physics=False)
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(3), 2))
    state = jax.jit(env.step)(state, jp.array([[0.0], [2.0]]))
    jax.block_until_ready(state)
    np.testing.assert_array_equal(state.done, [0, 1])
    np.testing.assert_array_equal(state.metrics["capacity_overflow"], [0, 1])
    np.testing.assert_array_equal(state.data[:, 0], [1, 0])


def test_additional_vmap_does_not_raise_on_healthy_flags():
    env = wrappers.wrap_for_training(FlagTask())
    keys = jax.random.split(jax.random.PRNGKey(4), 4).reshape(2, 2, 2)
    state = jax.jit(jax.vmap(env.reset))(keys)
    state = jax.jit(jax.vmap(env.step))(state, jp.zeros((2, 2, 1)))
    jax.block_until_ready(state)
    jax.effects_barrier()
    np.testing.assert_array_equal(state.done, np.zeros((2, 2)))
