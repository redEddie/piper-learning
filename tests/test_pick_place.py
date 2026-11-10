"""Pick-and-place environment contract on the CPU (JAX) physics path.

Grasping itself is not exercised here: the CPU MJX collision path differs from
the MuJoCo-Warp path used for training, and a scripted grasp was verified on
Warp (docs/evidence). These tests cover shapes, finiteness, the task-space
action mapping and the task registry.
"""
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest

from piper_sim_lab.pick_place import ACTION_SIZE, PiperPickPlace
from piper_sim_lab.tasks import make_env


CPU = {"impl": "jax", "bowl_segments": 8}


@pytest.fixture(scope="module")
def env():
    # Empty-hand starts only, so the placement checks do not depend on which
    # reset stage the key draws. Stage-wise resets are tested separately.
    return PiperPickPlace(config_overrides={**CPU, "reset_stage_probs": [1.0, 0.0, 0.0, 0.0]})


def test_reset_shapes_and_placement(env):
    state = jax.jit(env.reset)(jax.random.key(0))
    assert state.obs["state"].shape == (51,)
    assert state.obs["privileged"].shape == (58,)
    assert bool(jp.all(jp.isfinite(state.obs["state"])))
    c = env._config
    block = np.asarray(state.data.xpos[env._block_id])
    bowl = np.asarray(state.data.mocap_pos[env._bowl_mocap_id])
    assert c.block_x[0] <= block[0] <= c.block_x[1] and c.block_y[0] <= block[1] <= c.block_y[1]
    assert c.bowl_x[0] <= bowl[0] <= c.bowl_x[1] and c.bowl_y[0] <= bowl[1] <= c.bowl_y[1]
    assert abs(block[2] - c.block_half) < 1e-4
    assert float(state.metrics["stage"]) == 0.0 and float(state.done) == 0.0


def test_jacobian_matches_mujoco(env):
    state = jax.jit(env.reset)(jax.random.key(1))
    jac = np.asarray(env._tcp_jacobian(state.data))
    model = env.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(state.data.qpos)
    data.mocap_pos[:] = np.asarray(state.data.mocap_pos)
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, env._tcp_id)
    expected = np.concatenate((jacp, jacr))[:, env._qvel_ids]
    assert jac.shape == (6, 6)
    np.testing.assert_allclose(jac, expected, atol=1e-5)
    fd = np.asarray(env._fd_jacobian(state.data))
    np.testing.assert_allclose(fd, expected, atol=5e-3)


def test_task_space_action_moves_tcp(env):
    step = jax.jit(env.step)
    state = jax.jit(env.reset)(jax.random.key(2))
    start = np.asarray(state.data.site_xpos[env._tcp_id])
    for _ in range(10):
        state = step(state, jp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    moved = np.asarray(state.data.site_xpos[env._tcp_id]) - start
    # 10 steps of 1 cm/step along +y; servo lag allows some shortfall.
    assert 0.06 < moved[1] < 0.11
    assert abs(moved[0]) < 0.02 and abs(moved[2]) < 0.02
    assert float(state.done) == 0.0 and bool(jp.isfinite(state.reward))


def test_gripper_target_is_rate_limited(env):
    step = jax.jit(env.step)
    state = jax.jit(env.reset)(jax.random.key(3))
    state = step(state, jp.array([0.0] * 6 + [-1.0]))
    target = float(state.data.ctrl[env._gripper_ctrl_id])
    assert abs(target - (env._config.gripper_max - env._config.gripper_rate)) < 1e-6


def test_gripper_force_cap_and_binary_action(env):
    """Closed = squeeze with gripper_force; the binary action maps sign to open/closed."""
    model = env.mj_model
    np.testing.assert_allclose(model.actuator_forcerange[env._gripper_ctrl_id], [-3.0, 3.0])
    step = jax.jit(env.step)
    state = jax.jit(env.reset)(jax.random.key(8))
    state = step(state, jp.array([0.0] * 6 + [0.3]))       # positive: stay open
    assert abs(float(state.data.ctrl[env._gripper_ctrl_id]) - env._config.gripper_max) < 1e-6
    for _ in range(2):                                      # non-positive: close at gripper_rate per step
        state = step(state, jp.array([0.0] * 6 + [-0.2]))
    target = float(state.data.ctrl[env._gripper_ctrl_id])
    assert abs(target - (env._config.gripper_max - 2 * env._config.gripper_rate)) < 1e-6
    for _ in range(3):                                      # ... and saturates at fully closed (0)
        state = step(state, jp.array([0.0] * 6 + [-1.0]))
    assert abs(float(state.data.ctrl[env._gripper_ctrl_id])) < 1e-6


def test_random_rollout_is_finite(env):
    step = jax.jit(env.step)
    rng = jax.random.key(4)
    state = jax.jit(env.reset)(rng)
    for _ in range(5):
        rng, key = jax.random.split(rng)
        state = step(state, jax.random.uniform(key, (ACTION_SIZE,), minval=-1.0, maxval=1.0))
    assert bool(jp.all(jp.isfinite(state.obs["state"])))
    assert float(state.metrics["nonfinite"]) == 0.0


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_stage_wise_reset(stage):
    """Stage 1: above the block, open. Stage 2: holding it lifted. Stage 3: holding it above the bowl."""
    probs = [0.0] * 4
    probs[stage] = 1.0
    env_s = PiperPickPlace(config_overrides={**CPU, "reset_stage_probs": probs})
    c = env_s._config
    state = jax.jit(env_s.reset)(jax.random.key(6))
    block = np.asarray(state.data.xpos[env_s._block_id])
    tcp = np.asarray(state.data.site_xpos[env_s._tcp_id])
    assert float(state.info["reset_stage"]) == stage
    if stage == 1:
        assert abs(block[2] - c.block_half) < 1e-4
        assert abs((tcp - block)[2] - c.pregrasp_height) < 0.01 and np.linalg.norm((tcp - block)[:2]) < 0.01
        assert abs(float(state.data.ctrl[env_s._gripper_ctrl_id]) - c.gripper_max) < 1e-6
    else:
        assert np.linalg.norm(tcp - block) < 0.01 and block[2] - c.block_half > c.lift_height
        assert float(state.data.ctrl[env_s._gripper_ctrl_id]) == 0.0
        bowl = np.asarray(state.data.mocap_pos[env_s._bowl_mocap_id])
        if stage == 2:
            assert float(state.metrics["stage"]) == 1.0  # grasp predicate holds at the start
        else:
            assert np.linalg.norm((block - bowl)[:2]) < c.bowl_r_rim - c.block_half
            assert block[2] - c.block_half > bowl[2] + c.bowl_floor + c.bowl_height
            assert float(state.metrics["stage"]) == 2.0  # pre_inside holds at the start


def test_three_entry_stage_probs_still_accepted():
    env_s = PiperPickPlace(config_overrides={**CPU, "reset_stage_probs": [0.5, 0.25, 0.25]})
    np.testing.assert_allclose(np.asarray(env_s._stage_probs), [0.5, 0.25, 0.25, 0.0])


def test_stage_obs_option(env):
    """The stage one-hot occupies state[40:44]; stage_obs=False removes it (paper-style actor input)."""
    state = jax.jit(env.reset)(jax.random.key(7))
    np.testing.assert_allclose(np.asarray(state.obs["state"][40:44]), [1.0, 0.0, 0.0, 0.0])
    env_n = PiperPickPlace(config_overrides={**CPU, "stage_obs": False, "reset_stage_probs": [1.0, 0.0, 0.0, 0.0]})
    state = jax.jit(env_n.reset)(jax.random.key(7))
    assert state.obs["state"].shape == (47,)
    assert state.obs["privileged"].shape == (54,)


def test_task_registry():
    assert isinstance(make_env({"task": "pick_place", "impl": "jax", "bowl_segments": 8}), PiperPickPlace)
    assert make_env({"impl": "jax"}).action_size == 6
    with pytest.raises(ValueError):
        make_env({"task": "unknown"})


def test_config_validation():
    with pytest.raises(ValueError):
        PiperPickPlace(config_overrides={"impl": "jax", "jacobian": "exact"})
    with pytest.raises(ValueError):
        PiperPickPlace(config_overrides={"impl": "jax", "bowl_r_rim": 0.01})
    with pytest.raises(ValueError):  # wider than the 70 mm gripper opening
        PiperPickPlace(config_overrides={"impl": "jax", "block_half": 0.034})
    with pytest.raises(ValueError):
        PiperPickPlace(config_overrides={"impl": "jax", "block_rgba": [1.0, 0.0, 0.0]})


def test_block_variant_changes_only_the_object():
    """configs/pick_place_block30.yaml: the 3 cm block is a config override, not a new class."""
    env30 = PiperPickPlace(config_overrides={
        **CPU, "reset_stage_probs": [1.0, 0.0, 0.0, 0.0], "block_half": 0.015, "block_mass": 0.02,
        "block_rgba": [0.9, 0.1, 0.1, 1.0]})
    geom = env30.mj_model.geom(env30.mj_model.body_geomadr[env30._block_id])
    np.testing.assert_allclose(geom.size, [0.015] * 3)
    np.testing.assert_allclose(env30.mj_model.body_mass[env30._block_id], 0.02, rtol=1e-6)
    state = jax.jit(env30.reset)(jax.random.key(5))
    assert state.obs["state"].shape == (51,)
    assert abs(float(state.data.xpos[env30._block_id][2]) - 0.015) < 1e-4
    # The reset table heights follow block_half.
    assert env30._reset_table["pregrasp"].shape[0] > 0


DR = {**CPU, "stage_obs": False, "size_obs": True, "obs_randomization": True, "physics_randomization": True,
      "reset_stage_probs": [1.0, 0.0, 0.0, 0.0]}


def test_bowl_arm_collision_flag():
    """The flag adds arm-bowl contacts; bowl-floor/base pairs stay excluded."""
    for flag in (False, True):
        env = PiperPickPlace(config_overrides={**CPU, "bowl_arm_collision": flag, "reset_table_grid": 2, "reset_table_yaws": 2})
        m = env.mj_model
        d = mujoco.MjData(m)
        d.qpos[env._qpos_ids] = env._config.nominal_qpos
        mujoco.mj_forward(m, d)
        link6 = d.xpos[env._link6_id].copy()
        d.mocap_pos[env._bowl_mocap_id] = [link6[0] + 0.07, link6[1], link6[2] - 0.03]
        mujoco.mj_forward(m, d)
        bodies = {(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name) for c in d.contact[:d.ncon]}
        arm_bowl = {pair for pair in bodies if "bowl" in pair and ({"link5", "link6"} & set(pair))}
        assert bool(arm_bowl) == flag
        assert not {pair for pair in bodies if set(pair) in ({"world", "bowl"}, {"base_link", "bowl"})}


def test_randomized_observation_layout_and_bounds():
    env = PiperPickPlace(config_overrides=DR)
    c = env._config
    state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(0), 16))
    assert state.obs["state"].shape == (16, 48) and state.obs["privileged"].shape == (16, 65)
    p = state.info["perception"]
    assert np.all((np.asarray(p["obj_delay"]) >= 0) & (np.asarray(p["obj_delay"]) <= c.obj_pose_delay_max / c.ctrl_dt + 1e-6))
    assert np.all(np.asarray(p["hold_substeps"]) <= round(c.action_delay_max / c.sim_dt))
    assert np.all(np.abs(np.asarray(p["size_scale"]) - 1) <= c.size_noise + 1e-6)
    assert np.all(np.abs(np.asarray(p["bowl_offset"])) <= c.static_pos_noise_cap + 1e-6)
    step = jax.jit(jax.vmap(env.step))
    for _ in range(3):
        state = step(state, jp.zeros((16, ACTION_SIZE)))
    obs, priv = np.asarray(state.obs["state"]), np.asarray(state.obs["privileged"])
    assert np.isfinite(obs).all() and np.isfinite(priv).all()
    # The block barely moves while the arm holds still, so the error is the noise.
    err = obs[:, 25:28] - priv[:, 55:58]
    assert np.all(np.abs(err) <= c.obj_pos_noise_cap + 2e-3) and np.std(err) > 1e-3
    np.testing.assert_allclose(obs[:, 31:34] - priv[:, 61:64], np.asarray(p["bowl_offset"]), atol=1e-6)
    np.testing.assert_allclose(obs[:, 40], c.block_half * np.asarray(p["size_scale"]), rtol=1e-6)


def test_randomization_off_reproduces_true_state():
    """size_obs without obs_randomization: the actor sees the true values (eval with DR off)."""
    env = PiperPickPlace(config_overrides={**DR, "obs_randomization": False})
    state = jax.jit(env.reset)(jax.random.key(1))
    state = jax.jit(env.step)(state, jp.zeros(ACTION_SIZE))
    obs, priv = np.asarray(state.obs["state"]), np.asarray(state.obs["privileged"])
    np.testing.assert_allclose(obs[25:28], priv[55:58], atol=1e-6)
    np.testing.assert_allclose(obs[31:34], priv[61:64], atol=1e-6)
    assert obs[40] == pytest.approx(env._config.block_half)
    assert int(state.info["perception"]["hold_substeps"]) == 0


def test_physics_randomization_fn_ranges():
    env = PiperPickPlace(config_overrides=DR)
    c = env._config
    model, axes = env.physics_randomization_fn(env.mjx_model, jax.random.split(jax.random.key(0), 32))
    mass = np.asarray(model.body_mass[:, env._block_id])
    assert mass.min() >= c.block_mass_range[0] - 1e-6 and mass.max() <= c.block_mass_range[1] + 1e-6 and mass.std() > 0
    fric = np.asarray(model.geom_friction[:, env._block_geom_id, 0])
    assert np.all(np.abs(fric - 1.0) <= c.block_friction_scale + 1e-6)
    table = np.asarray(model.geom_friction[:, env._floor_geom_id, 0])
    assert table.min() >= c.table_friction_range[0] - 1e-6 and table.max() <= c.table_friction_range[1] + 1e-6
    kp = np.asarray(model.actuator_gainprm[:, env._gripper_ctrl_id, 0]) / c.gripper_kp
    assert kp.min() >= c.gripper_gain_scale[0] - 1e-6 and kp.max() <= c.gripper_gain_scale[1] + 1e-6
    np.testing.assert_allclose(-np.asarray(model.actuator_biasprm[:, env._gripper_ctrl_id, 1]), kp * c.gripper_kp, rtol=1e-6)
    assert axes.body_mass == 0 and axes.geom_friction == 0 and axes.dof_damping is None
    # Other bodies and geoms keep their nominal values.
    others = np.delete(np.asarray(model.body_mass), env._block_id, axis=1)
    np.testing.assert_allclose(others, np.broadcast_to(np.delete(env.mj_model.body_mass, env._block_id), others.shape), rtol=1e-6)


def test_action_delay_holds_previous_command():
    env = PiperPickPlace(config_overrides={**DR, "physics_randomization": False})
    state = jax.jit(env.reset)(jax.random.key(2))
    ctrl0 = state.data.ctrl
    new = ctrl0.at[env._gripper_ctrl_id].set(0.0)
    held, _ = jax.jit(lambda d, u: env._step_physics(d, u, jp.int32(env.n_substeps)))(state.data, new)
    free, _ = jax.jit(lambda d, u: env._step_physics(d, u, jp.int32(0)))(state.data, new)
    width = lambda d: float(d.qpos[env._finger_qpos_ids[0]] - d.qpos[env._finger_qpos_ids[1]])
    assert width(held) > width(free) + 1e-3          # holding the old (open) command the whole step
    assert float(held.ctrl[env._gripper_ctrl_id]) == 0.0
