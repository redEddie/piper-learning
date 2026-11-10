"""Physical reset/contact and frame calculations for the new reward adapter."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jp
import mujoco
import numpy as np
import pytest
import yaml

from piper_sim_lab.tasks import make_env


@pytest.fixture(scope='module')
def env():
    config = yaml.safe_load((Path(__file__).parents[1]/'configs/show103/stage-reward-v1-4096.yaml').read_text())
    return make_env(dict(config['environment'], impl='jax', batch_capacity=1))


def native_features(env, data, goal=0.):
    mujoco.mj_forward(env.mj_model, data)
    jacp, jacr = np.zeros((3, env.mj_model.nv)), np.zeros((3, env.mj_model.nv))
    mujoco.mj_jacSite(env.mj_model, data, jacp, jacr, env._tcp_id)
    view = SimpleNamespace(**{n: jp.asarray(getattr(data, n)) for n in
        ('qpos', 'qvel', 'xpos', 'xmat', 'site_xpos', 'geom_xpos', 'geom_xmat')})
    view.site_xmat = jp.asarray(data.site_xmat.reshape(-1, 3, 3))
    with patch.object(env, 'contacts', lambda _: {k: jp.asarray(v) for k, v in env._native_contacts(data).items()}), \
         patch('piper_sim_lab.stage_reward_env.support.jac', return_value=(jp.asarray(jacp.T), jp.asarray(jacr.T))):
        return env.features(view, jp.asarray(goal))


def test_four_starts_use_real_bilateral_contacts_without_near_release_reset(env):
    bank = env.reset_bank
    assert len(bank['qpos']) == 4
    for i in (2, 3):
        d = mujoco.MjData(env.mj_model)
        d.qpos[:] = bank['qpos'][i]
        d.ctrl[:] = bank['ctrl'][i]
        f = native_features(env, d)
        assert f.left_contact and f.right_contact
        assert f.cube_bottom_table > .02
    assert f.cube_center_above_rim > 0
    assert f.tcp_floor_height > .10  # ordinary over-bowl start, not release-height shortcut


def test_rotated_cube_bottom_uses_full_shape(env):
    d = mujoco.MjData(env.mj_model)
    d.qpos[:] = env.reset_bank['qpos'][0]
    start = env._block_qpos_adr
    angle = np.pi/4
    d.qpos[start:start+7] = [.30, -.12, .20, np.cos(angle/2), np.sin(angle/2), 0, 0]
    f = native_features(env, d)
    assert float(f.cube_bottom_table) == pytest.approx(.20-.025*np.sqrt(2), abs=1e-6)


def test_bowl_translation_and_rotation_are_removed_from_cube_velocity(env):
    d = mujoco.MjData(env.mj_model)
    d.qpos[:] = env.reset_bank['qpos'][0]
    bp = np.array([.25, .12, 0.])
    cp = np.array([.28, .15, .12])
    d.qpos[env._block_qpos_adr:env._block_qpos_adr+3] = cp
    d.qpos[env._bowl_qpos_adr:env._bowl_qpos_adr+7] = np.r_[bp, 1., 0., 0., 0.]
    velocity = np.array([.1, -.2, .03])
    omega = np.array([0., 0., 2.])
    b, c = env._bowl_qvel_adr, env._block_qvel_adr
    d.qvel[b:b+6] = np.r_[velocity, omega]
    d.qvel[c:c+3] = velocity+np.cross(omega, cp-bp)
    f = native_features(env, d)
    assert float(f.cube_relative_speed) < 1e-6
    assert float(f.cube_bowl_xy) == pytest.approx(np.linalg.norm((cp-bp)[:2]), abs=1e-6)


def test_opening_intent_uses_latched_final_goal(env):
    d = mujoco.MjData(env.mj_model)
    d.qpos[:] = env.reset_bank['qpos'][2]
    d.ctrl[env._gripper_ctrl_id] = .01  # moving reference differs from final target
    assert native_features(env, d, .05).opening_intent
    assert not native_features(env, d, 0.).opening_intent
