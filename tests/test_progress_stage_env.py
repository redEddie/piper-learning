"""Native geometry, conditional physical prefix, and independent face axes."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import mujoco
import numpy as np
import jax.numpy as jp
import pytest
import yaml
from piper_sim_lab.tasks import make_env
from piper_sim_lab.five_start_stage import native_features,grasp_measurements


@pytest.fixture(scope='module')
def env():
    c=yaml.safe_load(Path('configs/show103/stage-reward-v5-transformer-lstm-progress-plus2.yaml').read_text())
    return make_env(dict(c['environment'],impl='jax',batch_capacity=1))


def test_face_penalty_uses_nearest_3d_face_and_cube_width(env):
    eye=np.eye(3)
    for axis in (np.array([1.,0.,0.]),np.array([-1.,0.,0.]),np.array([0.,0.,1.])):
        _,width,angle=grasp_measurements(env._config,eye,axis,.05,xp=np)
        assert angle==0. and width==.05
    _,width,angle=grasp_measurements(env._config,eye,np.array([1.,1.,0.])/np.sqrt(2),.05,xp=np)
    assert angle==pytest.approx(np.pi/4) and width==pytest.approx(.05*np.sqrt(2))


def test_prefix_credit_is_measured_not_a_fabricated_landing(env):
    m=env._release_memory
    assert np.asarray(m.approach_complete).all() and np.asarray(m.grasp_complete).all()
    assert np.asarray(m.transport_complete).all()
    assert not np.asarray(m.place_complete).any() and not np.asarray(m.home_complete).any()
    assert np.asarray(m.released).all() and not np.asarray(m.first_floor_contact).any()
    assert env.observation_size==dict(state=(49,),privileged=(66,))
    assert env.reward_settings.home_error==.2


def test_native_and_runtime_new_measurements_are_identical(env):
    data=mujoco.MjData(env.mj_model);data.qpos[:]=env.reset_bank['qpos'][3][0]
    mujoco.mj_forward(env.mj_model,data)
    expected=native_features(env,data,0.)
    a,b=np.zeros((3,env.mj_model.nv)),np.zeros((3,env.mj_model.nv))
    mujoco.mj_jacSite(env.mj_model,data,a,b,env._tcp_id)
    view=SimpleNamespace(**{n:jp.asarray(getattr(data,n)) for n in ('qpos','qvel','xpos','xmat','site_xpos','geom_xpos','geom_xmat')})
    view.site_xmat=jp.asarray(data.site_xmat.reshape(-1,3,3))
    with patch.object(env,'contacts',lambda _:{k:jp.asarray(v) for k,v in env._native_contacts(data).items()}),patch('piper_sim_lab.stage_reward_env.support.jac',return_value=(jp.asarray(a.T),jp.asarray(b.T))):
        actual=env.features(view,jp.array(0.))
    assert len(actual)==33
    np.testing.assert_allclose(actual,expected,atol=3e-5)
