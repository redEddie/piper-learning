"""Position sampling must preserve physical grasps and held-out bank identity."""
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
import yaml

from piper_sim_lab.tasks import make_env
from piper_sim_lab.position_random_stage import ASSET_DIR

ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT/'configs/show103/stage-reward-v1-4096-random-positions-2-4-6-8.yaml').read_text())


@pytest.fixture(scope='module')
def env():
    return make_env(dict(config()['environment'], impl='jax', batch_capacity=1))


def test_train_and_eval_poses_are_independent_and_sealed(env):
    import hashlib
    import json
    names = ['show103-position-random-train-v1.npz', 'show103-position-random-eval-v1.npz']
    metas = [json.loads((ASSET_DIR/n).with_suffix('.json').read_text()) for n in names]
    assert metas[0]['seed'] != metas[1]['seed']
    assert metas[0]['sha256'] != metas[1]['sha256']
    for name, meta in zip(names, metas):
        assert hashlib.sha256((ASSET_DIR/name).read_bytes()).hexdigest() == meta['sha256']
        assert meta['reward_history_prefilled'] is False
    assert env._bank_qpos.shape == (4, 64, env.mj_model.nq)


@pytest.mark.parametrize('stage', range(4))
def test_random_positions_and_continuous_yaw_keep_contacts(env, stage):
    keys = jax.random.split(jax.random.key(6000+stage), 128)
    draw = jax.jit(jax.vmap(lambda k: env._reset_pose(k, jp.array(stage, jp.int32))))
    qpos, ctrl, index = jax.tree.map(np.asarray, draw(keys))
    again = jax.tree.map(np.asarray, draw(keys))
    np.testing.assert_array_equal(qpos, again[0])
    assert len(np.unique(index)) > 30
    cube = qpos[:, env._block_qpos_adr:env._block_qpos_adr+2]
    bowl = qpos[:, env._bowl_qpos_adr:env._bowl_qpos_adr+2]
    assert cube.std(axis=0).min() > .001
    assert bowl.std(axis=0).min() > .001
    if stage < 3:
        assert np.linalg.norm(cube-bowl, axis=1).min() >= .14-1e-6
    else:
        assert np.linalg.norm(cube-bowl, axis=1).max() < .003
    if stage == 0:
        np.testing.assert_allclose(ctrl[:, env._arm_ctrl_ids], np.tile(np.asarray(env._nominal), (128, 1)), atol=1e-6)
    data = mujoco.MjData(env.mj_model)
    for i in range(16):
        data.qpos[:] = qpos[i]
        data.ctrl[:] = ctrl[i]
        mujoco.mj_forward(env.mj_model, data)
        contact = env._native_contacts(data)
        assert not any(contact['bowl'+str(j)] for j in range(1, 9))
        if stage >= 2:
            assert contact['left'] and contact['right']
            assert np.linalg.norm(data.site_xpos[env._tcp_id]-data.xpos[env._block_id]) < .005


def test_bank_rejects_changed_geometry():
    with pytest.raises(ValueError, match='geometry differs'):
        make_env(dict(config()['environment'], impl='jax', base_height=.016, batch_capacity=1))


def test_reward_control_and_training_settings_match_fixed_experiment():
    old = yaml.safe_load((ROOT/'configs/show103/stage-reward-v1-4096-2-4-6-8.yaml').read_text())
    new = config()
    assert new['training'] == old['training']
    assert new['network'] == old['network']
    changed = {k for k in new['environment'] if new['environment'].get(k) != old['environment'].get(k)}
    assert changed <= {'task', 'position_reset_bank', 'evaluation_position_reset_bank',
        'position_yaw_range', 'block_x', 'block_y', 'bowl_x', 'bowl_y',
        'spawn_max_radius', 'spawn_min_distance', 'spawn_candidates'}
