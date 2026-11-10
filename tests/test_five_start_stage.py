"""Physical prefix provenance, transformed velocities, and measured geometry."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
import yaml

from piper_sim_lab.tasks import make_env
from piper_sim_lab.position_random_stage import ASSET_DIR
from piper_sim_lab.five_start_stage import native_features
from piper_sim_lab import stage_reward_v2 as reward


@pytest.fixture(scope='module')
def env():
    cfg = yaml.safe_load(Path('configs/show103/stage-reward-v2-five-starts-4096-plus2-4-6-8.yaml').read_text())
    return make_env(dict(cfg['environment'], impl='jax', batch_capacity=1))


def test_four_original_banks_are_preserved(env):
    with np.load(ASSET_DIR/'show103-position-random-train-v1.npz') as old:
        np.testing.assert_array_equal(env.reset_bank['qpos'][:4], old['qpos'])
        np.testing.assert_array_equal(env.reset_bank['ctrl'][:4], old['ctrl'])
    np.testing.assert_allclose(env._stage_probs, [.2]*5)


def test_prefix_memories_are_derived_from_physical_trace(env):
    for split in ('train', 'eval'):
        with np.load(ASSET_DIR/f'show103-five-start-{split}-v2.npz') as bank:
            for index in (0, 63):
                rows = bank['prefix_features'][index, :bank['prefix_steps'][index]]
                fs = reward.Features(*(jp.asarray(rows[:, k], dtype=jp.bool_ if 13 <= k < 20 else jp.float32) for k in range(21)))
                @jax.jit
                def replay(features):
                    initial = jax.tree.map(lambda a: a[0], features)
                    def tick(carry, after):
                        memory, before = carry
                        return (reward.advance_memory(memory, before, after), after), None
                    return jax.lax.scan(tick, (reward.reset_memory(), initial), jax.tree.map(lambda a: a[1:], features))[0][0]
                memory = replay(fs)
                assert memory.qualified_release and not memory.invalid_release
                for name, actual in zip(memory._fields, memory):
                    np.testing.assert_allclose(actual, bank['memory_'+name][index], atol=2e-6)


def test_native_features_match_jax_geometry_and_velocity(env):
    d = mujoco.MjData(env.mj_model)
    d.qpos[:] = env.reset_bank['qpos'][4][0]
    d.qvel[:] = np.asarray(env._release_qvel[0])
    mujoco.mj_forward(env.mj_model, d)
    expected = native_features(env, d, .05)
    jac, jacr = np.zeros((3, env.mj_model.nv)), np.zeros((3, env.mj_model.nv))
    mujoco.mj_jacSite(env.mj_model, d, jac, jacr, env._tcp_id)
    view = SimpleNamespace(**{n: jp.asarray(getattr(d, n)) for n in
        ('qpos', 'qvel', 'xpos', 'xmat', 'site_xpos', 'geom_xpos', 'geom_xmat')})
    view.site_xmat = jp.asarray(d.site_xmat.reshape(-1, 3, 3))
    with patch.object(env, 'contacts', lambda _: {k: jp.asarray(v) for k, v in env._native_contacts(d).items()}), \
         patch('piper_sim_lab.stage_reward_env.support.jac', return_value=(jp.asarray(jac.T), jp.asarray(jacr.T))):
        actual = env.features(view, jp.array(.05))
    np.testing.assert_allclose(actual, expected, atol=2e-5)


def test_fifth_rotation_preserves_relative_release_motion(env):
    qpos, ctrl, index = env._reset_pose(jax.random.key(77), jp.array(4))
    velocity, goal, memory = env._reset_aux(jp.array(4), index, qpos, ctrl)
    d = mujoco.MjData(env.mj_model)
    d.qpos[:], d.qvel[:] = qpos, velocity
    mujoco.mj_forward(env.mj_model, d)
    rotated = native_features(env, d, goal)
    d.qpos[:] = env.reset_bank['qpos'][4][int(index)]
    d.qvel[:] = env._release_qvel[index]
    mujoco.mj_forward(env.mj_model, d)
    original = native_features(env, d, goal)
    assert goal == .05 and memory.qualified_release
    np.testing.assert_allclose([rotated.cube_bowl_xy, rotated.cube_relative_speed, rotated.tcp_floor_height],
        [original.cube_bowl_xy, original.cube_relative_speed, original.tcp_floor_height], atol=2e-5)
    _, goal, memory = env._reset_aux(jp.array(2), index, qpos, ctrl)
    assert not memory.lifted and not memory.qualified_release
