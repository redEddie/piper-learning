"""CPU checks for task semantics; no GPU, rendering, or training required."""

import os
from types import SimpleNamespace

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from piper_sim_lab.env import ARM_JOINTS, PiperReach, TARGET_BODY, TCP_SITE, default_config
from piper_sim_lab.wrappers import wrap_for_training


@pytest.fixture(scope="module")
def env():
    return PiperReach()


@pytest.fixture(scope="module")
def reset(env):
    return jax.jit(env.reset)


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(env.step)


def test_seeded_resets_are_reproducible_and_goals_use_forward_kinematics(env, reset):
    first = reset(jax.random.PRNGKey(4))
    repeated = reset(jax.random.PRNGKey(4))
    other = reset(jax.random.PRNGKey(5))
    np.testing.assert_array_equal(first.obs["state"], repeated.obs["state"])
    assert not np.allclose(first.data.mocap_pos, other.data.mocap_pos)
    assert first.obs["state"].shape == (30,)
    assert np.all(np.isfinite(first.obs["state"]))
    assert float(first.metrics["capacity_overflow"]) == 0.0
    # Position commands accumulate independently of qpos. Identical physical
    # state/goal/history with different targets must be distinguishable.
    different_ctrl = first.data.replace(
        ctrl=first.data.ctrl.at[env._arm_ctrl_ids].add(0.1)
    )
    changed_obs = env._get_obs(different_ctrl, jp.zeros(6))["state"]
    np.testing.assert_array_equal(changed_obs[:18], first.obs["state"][:18])
    np.testing.assert_allclose(changed_obs[18:24] - first.obs["state"][18:24], 0.1, atol=1e-7)
    np.testing.assert_array_equal(changed_obs[-6:], first.obs["state"][-6:])
    # Use native MuJoCo as an independent FK oracle for the sampled goal.
    native = mujoco.MjData(env.mj_model)
    native.qpos[:] = np.asarray(first.data.qpos)
    for index, name in enumerate(ARM_JOINTS):
        native.qpos[env.mj_model.joint(name).qposadr[0]] = first.info["goal_qpos"][index]
    mujoco.mj_forward(env.mj_model, native)
    target_id = env.mj_model.body(TARGET_BODY).mocapid[0]
    np.testing.assert_allclose(
        first.data.mocap_pos[target_id], native.site_xpos[env.mj_model.site(TCP_SITE).id], atol=2e-6,
    )


def test_tcp_matches_the_center_of_distal_finger_pads(env):
    native = mujoco.MjData(env.mj_model)
    native.qpos[:] = [0.0, 0.7, -0.9, 0.0, 0.3, 0.0, 0.025, -0.025]
    mujoco.mj_forward(env.mj_model, native)
    pad_centers = []
    for body_name in ("link7", "link8"):
        body_id = env.mj_model.body(body_name).id
        ids = np.flatnonzero(
            (env.mj_model.geom_bodyid == body_id)
            & (env.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
        )
        distal = ids[np.argmax(np.abs(env.mj_model.geom_pos[ids, 1]))]
        pad_centers.append(native.geom_xpos[distal])
    np.testing.assert_allclose(
        native.site_xpos[env.mj_model.site(TCP_SITE).id], np.mean(pad_centers, axis=0), atol=1e-6,
    )


def test_control_limits_fingers_and_finite_physics(env, reset, step):
    state = reset(jax.random.PRNGKey(12))
    # Put the targets at their limits, then request movement beyond those limits.
    actuator_ids = np.array([env.mj_model.actuator(name).id for name in ARM_JOINTS])
    limits = env.mj_model.actuator_ctrlrange[actuator_ids]
    state = state.replace(data=state.data.replace(
        ctrl=state.data.ctrl.at[actuator_ids].set(jp.array(limits[:, 1]))
    ))
    state = step(state, jp.full((6,), 10.0))
    np.testing.assert_allclose(state.data.ctrl[actuator_ids], limits[:, 1], atol=1e-6)
    np.testing.assert_array_equal(state.obs["state"][-6:], np.ones(6))
    gripper = env.mj_model.actuator("gripper").id
    assert float(state.data.ctrl[gripper]) == pytest.approx(0.025)
    state = reset(jax.random.PRNGKey(12))
    for _ in range(4):
        state = step(state, jp.array([0.1, -0.1, 0.1, 0.0, 0.0, 0.0]))
        assert np.all(np.isfinite(state.data.qpos))
        assert np.all(np.isfinite(state.data.qvel))
        assert np.all(np.isfinite(state.obs["state"]))
        assert np.isfinite(state.reward)
        assert float(state.done) == 0.0
        assert float(state.metrics["capacity_overflow"]) == 0.0


def test_success_is_a_metric_and_nonfinite_actions_end_episode(env, reset, step):
    state = reset(jax.random.PRNGKey(8))
    # Brax EvalWrapper adds this key before scanning env.step. Trace the exact
    # contract without another expensive physics compilation.
    extended = state.replace(metrics={**state.metrics, "reward": jp.float32(0.0)})
    traced = jax.eval_shape(env.step, extended, jp.zeros(6))
    assert traced.metrics.keys() == extended.metrics.keys()
    next_state = step(state, jp.zeros(6))
    target_id = env.mj_model.body(TARGET_BODY).mocapid[0]
    tcp_id = env.mj_model.site(TCP_SITE).id
    # Set the goal to the position reached by this exact next control step.
    state = state.replace(data=state.data.replace(
        mocap_pos=state.data.mocap_pos.at[target_id].set(next_state.data.site_xpos[tcp_id])
    ))
    reached = step(state, jp.zeros(6))
    assert float(reached.metrics["success"]) == 1.0
    assert float(reached.done) == 0.0
    failed = step(state, jp.array([jp.nan, 0, 0, 0, 0, 0]))
    assert float(failed.done) == 1.0
    assert float(failed.metrics["nonfinite"]) == 1.0
    assert float(failed.metrics["capacity_overflow"]) == 0.0
    assert float(failed.metrics["success"]) == 0.0
    assert np.isfinite(failed.reward)
    assert np.all(np.isfinite(failed.obs["state"]))


def test_config_overrides_do_not_mutate_the_callers_config():
    config = default_config()
    with pytest.raises(ValueError, match="integer multiple"):
        PiperReach(config, {"ctrl_dt": 0.051})
    assert config.ctrl_dt == 0.05
    with pytest.raises(ValueError, match="six finite"):
        PiperReach(config_overrides={"nominal_qpos": [0.0]})
    for name in ("batch_capacity", "contacts_per_env", "max_constraints_per_env", "ccd_pairs_per_env"):
        with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
            PiperReach(config_overrides={name: 0})

    # Inspect Warp sizing and bit semantics without constructing any GPU data.
    stub = object.__new__(PiperReach)
    stub._config = default_config()
    assert stub._make_data_kwargs() == {}
    stub._config.impl = "warp"
    stub._config.batch_capacity = 4096
    assert stub._make_data_kwargs() == {
        "naconmax": 655360, "njmax": 512, "naccdmax": 16384,
    }
    for bits, expected in ((0, False), ((1 << 9) | (1 << 10), False),
                           ((1 << 9) | 1, True), (1 << 11, True)):
        data = SimpleNamespace(_impl=SimpleNamespace(overflow=jp.int32(bits)))
        assert bool(stub._capacity_overflow(data)) is expected


def test_batched_timeout_resets_goals_and_observation_history(env):
    # This test intentionally injects a NaN to inspect the terminal/reset
    # contract; production training enables the outer fail-fast safety guard.
    wrapped = wrap_for_training(env, episode_length=2, action_repeat=1, strict_physics=False)
    batched_reset = jax.jit(wrapped.reset)
    batched_step = jax.jit(wrapped.step)
    state = batched_reset(jax.random.split(jax.random.PRNGKey(17), 2))
    initial_goals = np.asarray(state.data.mocap_pos).copy()
    assert not np.allclose(initial_goals[0], initial_goals[1])
    action = jp.full((2, 6), 0.1)
    state = batched_step(state, action)
    np.testing.assert_array_equal(state.done, np.zeros(2))
    state = batched_step(state, action)
    np.testing.assert_array_equal(state.done, np.ones(2))
    np.testing.assert_array_equal(state.info["truncation"], np.ones(2))
    np.testing.assert_array_equal(state.info["AutoResetWrapper_done_count"], np.ones(2))
    np.testing.assert_array_equal(state.obs["state"][:, -6:], np.zeros((2, 6)))
    assert not np.allclose(initial_goals, state.data.mocap_pos)
    target_id = env.mj_model.body(TARGET_BODY).mocapid[0]
    tcp_id = env.mj_model.site(TCP_SITE).id
    for index in range(2):
        native = mujoco.MjData(env.mj_model)
        native.qpos[:] = np.asarray(state.data.qpos[index])
        for j, name in enumerate(ARM_JOINTS):
            native.qpos[env.mj_model.joint(name).qposadr[0]] = state.info["goal_qpos"][index, j]
        mujoco.mj_forward(env.mj_model, native)
        np.testing.assert_allclose(
            state.data.mocap_pos[index, target_id], native.site_xpos[tcp_id], atol=2e-6,
        )
    np.testing.assert_allclose(
        state.obs["state"][:, 15:18],
        state.data.mocap_pos[:, target_id] - state.data.site_xpos[:, tcp_id], atol=1e-6,
    )
    state = batched_step(state, action)
    np.testing.assert_array_equal(state.done, np.zeros(2))
    np.testing.assert_array_equal(state.info["steps"], np.ones(2))
    # Reuse the same compiled signatures to check asynchronous reset isolation.
    state = batched_reset(jax.random.split(jax.random.PRNGKey(24), 2))
    old_goals = np.asarray(state.data.mocap_pos).copy()
    old_goal_qpos = np.asarray(state.info["goal_qpos"]).copy()
    state = batched_step(state, action.at[0, 0].set(jp.nan))
    np.testing.assert_array_equal(state.done, [1, 0])
    np.testing.assert_array_equal(state.info["truncation"], [0, 0])
    np.testing.assert_array_equal(state.info["AutoResetWrapper_done_count"], [1, 0])
    np.testing.assert_array_equal(state.data.mocap_pos[1], old_goals[1])
    np.testing.assert_array_equal(state.info["goal_qpos"][1], old_goal_qpos[1])
    np.testing.assert_array_equal(state.obs["state"][0, -6:], np.zeros(6))
    np.testing.assert_allclose(state.obs["state"][1, -6:], np.full(6, 0.1))
    assert not np.allclose(state.data.mocap_pos[0], old_goals[0])
