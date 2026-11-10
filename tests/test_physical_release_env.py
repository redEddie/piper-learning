"""Physical measurements and actual release prefix replay for the v4 task."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jp
import mujoco
import numpy as np
import pytest
import yaml
from piper_sim_lab.tasks import make_env
from piper_sim_lab.five_start_stage import placement_measurements,native_features


@pytest.fixture(scope='module')
def env():
    cfg=yaml.safe_load(Path('configs/show103/stage-reward-v4-transformer-lstm-physical-release-plus2.yaml').read_text())
    return make_env(dict(cfg['environment'],impl='jax',batch_capacity=1))


def test_measured_cube_containment_requires_geometry_and_floor_support(env):
    c=env._config;eye=np.eye(3);bowl=np.zeros(3)
    cube=np.array([0.,0.,c.bowl_floor+c.block_half])
    height,over,inside=placement_measurements(c,cube,bowl,eye,eye,True,xp=np)
    assert height==c.block_half and over and inside
    assert not placement_measurements(c,cube,bowl,eye,eye,False,xp=np)[2]
    assert not placement_measurements(c,cube+np.array([.1,0,0]),bowl,eye,eye,True,xp=np)[1]
    assert not placement_measurements(c,cube+np.array([0,0,.15]),bowl,eye,eye,True,xp=np)[2]


def test_prefix_replay_uses_actual_airborne_measurements_and_preserves_observation_layout(env):
    assert env.observation_size==dict(state=(49,),privileged=(66,))
    m=env._release_memory
    assert np.asarray(m.released).all() and not np.asarray(m.qualified_release).any()
    assert not np.asarray(m.invalid_release).any() and not np.asarray(m.first_floor_contact).any()
    assert np.all(np.asarray(m.release_center_height)>0)
    assert env.reward_settings.drop_weight==2.
    assert env.reset_bank['runtime_history_model'].startswith('stage_reward_v4')


def test_new_native_and_runtime_features_match(env):
    data=mujoco.MjData(env.mj_model);data.qpos[:]=env.reset_bank['qpos'][4][0]
    data.qvel[:]=np.asarray(env._release_qvel[0]);mujoco.mj_forward(env.mj_model,data)
    expected=native_features(env,data,.05)
    jac,jacr=np.zeros((3,env.mj_model.nv)),np.zeros((3,env.mj_model.nv))
    mujoco.mj_jacSite(env.mj_model,data,jac,jacr,env._tcp_id)
    view=SimpleNamespace(**{n:jp.asarray(getattr(data,n)) for n in
        ('qpos','qvel','xpos','xmat','site_xpos','geom_xpos','geom_xmat')})
    view.site_xmat=jp.asarray(data.site_xmat.reshape(-1,3,3))
    with patch.object(env,'contacts',lambda _:{k:jp.asarray(v) for k,v in env._native_contacts(data).items()}),patch(
        'piper_sim_lab.stage_reward_env.support.jac',return_value=(jp.asarray(jac.T),jp.asarray(jacr.T))):
        actual=env.features(view,jp.array(.05))
    assert len(actual)==30
    np.testing.assert_allclose(actual,expected,atol=2e-5)
