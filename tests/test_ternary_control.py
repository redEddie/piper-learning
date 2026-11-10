"""Latched gripper, observable command state, baseline preservation and FK parity."""
import ast
from pathlib import Path
import subprocess
from unittest import mock
import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
import yaml
from piper_sim_lab.pick_place import PiperPickPlace
from piper_sim_lab.tasks import make_env
from piper_sim_lab.ternary_control import gripper_command
from piper_sim_lab.control_lab.hierarchy_env import fk_floor

ROOT = Path(__file__).resolve().parents[1]


def test_hold_continues_to_final_goal_and_repeated_commands_are_idempotent():
    def sequence(actions):
        goal = ref = .05
        result=[]
        for a in actions:
            goal,ref=gripper_command(a,goal,ref,.05,.01,.5)
            result.append([float(goal),float(ref)])
        return np.array(result)
    once=sequence([-1.,0.,0.,0.,0.,0.])
    repeated=sequence([-1.]*6)
    np.testing.assert_allclose(once,repeated,atol=1e-7)
    np.testing.assert_allclose(once[:,0],0.)
    np.testing.assert_allclose(once[:,1],[.04,.03,.02,.01,0.,0.],atol=1e-7)
    # Holding does not freeze at .04 m or replace a force-closing goal by the measured aperture.
    assert once[-1,1] < 1e-7


@pytest.mark.parametrize('a,previous,expected',[(-.5,.05,0),(.5,0,.05),(.49,0,0),(-.49,.05,.05),(0.,.05,.05),(0.,0.,0.)])
def test_thresholds_and_zero_crossing(a,previous,expected):
    goal,_=gripper_command(a,previous,.025,.05,.01,.5)
    assert float(goal)==pytest.approx(expected)


@pytest.fixture(scope='module')
def native_env():
    c=yaml.safe_load((ROOT/'configs/show100/fresh-50m.yaml').read_text())['environment']
    tables={k:np.zeros((1,6)) for k in ('pregrasp','grasp','preinside')}
    with mock.patch('mujoco_playground._src.mjx_env.put_model',return_value=None), \
         mock.patch.object(PiperPickPlace,'_build_reset_table',return_value=tables):
        return make_env(dict(c,impl='jax'))


def test_final_goal_observable_independently_of_actual_aperture(native_env):
    env=native_env;data=mujoco.MjData(env._mj_model)
    data.qpos[:]=np.asarray(env._initial_qpos)
    data.qpos[env._finger_qpos_ids]=[.03,-.03]
    data.ctrl[:]=np.asarray(env._initial_ctrl)
    data.qpos[env._block_qpos_adr:env._block_qpos_adr+7]=[.3,0,.025,1,0,0,0]
    mujoco.mj_forward(env._mj_model,data)
    p=env._predicates(data)
    perception=env._reset_perception(data,jax.random.key(1))
    action=jp.arange(7,dtype=jp.float32)/10
    obs0,_=env._get_obs(data,action,p,perception,jax.random.key(2),gripper_goal=jp.float32(0))
    obs1,_=env._get_obs(data,action,p,perception,jax.random.key(2),gripper_goal=jp.float32(.05))
    assert obs0['state'].shape==(49,)
    assert obs0['privileged'].shape==(66,)
    assert float(obs0['state'][24])==pytest.approx(.06)
    assert float(obs1['state'][24])==pytest.approx(.06)
    assert float(obs0['state'][41])==0 and float(obs1['state'][41])==1
    np.testing.assert_array_equal(obs0['state'][-7:],action)
    np.testing.assert_array_equal(obs1['state'][-7:],action)


def test_tcp_geometry_and_runtime_fk_jacobian_agree(native_env):
    env=native_env;model=env._mj_model;data=mujoco.MjData(model)
    np.testing.assert_allclose(model.site_pos[env._tcp_id],[0,0,.1385],atol=1e-8)
    np.testing.assert_allclose(model.actuator_forcerange[env._gripper_ctrl_id],[-3.,3.])
    assert len([i for i in range(model.ngeom) if '_bridge_' in (model.geom(i).name or '')])==8
    jacp,jacr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
    rng=np.random.default_rng(100)
    for _ in range(30):
        q=rng.uniform(np.asarray(env._lower),np.asarray(env._upper))
        data.qpos[:]=np.asarray(env._initial_qpos);data.qpos[env._qpos_ids]=q
        mujoco.mj_forward(model,data)
        expected=data.site_xpos[env._tcp_id]-np.array([0,0,.015])
        np.testing.assert_allclose(env.kin.forward(q)['tcp_position'],expected,atol=1e-7)
        mujoco.mj_jacSite(model,data,jacp,jacr,env._tcp_id)
        np.testing.assert_allclose(env.kin.jacobian(q),np.r_[jacp[:,env._qvel_ids],jacr[:,env._qvel_ids]],atol=2e-5)
        z,normal=fk_floor(env.kin,jp.asarray(q))
        np.testing.assert_allclose(z,expected[2],atol=2e-7)
        np.testing.assert_allclose(normal,jacp[2,env._qvel_ids],atol=2e-7)


def test_original_reset_reward_predicates_and_randomization_preserved():
    original=subprocess.check_output(['git','show','b40fa4df0c8ef301421edab65250f9e3ed70037a:src/piper_sim_lab/pick_place.py'],cwd=ROOT,text=True)
    current=(ROOT/'src/piper_sim_lab/pick_place.py').read_text()
    def methods(source):
        cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='PiperPickPlace')
        return {x.name:x for x in cls.body if isinstance(x,ast.FunctionDef)}
    a,b=methods(original),methods(current)
    for name in ['reset','_predicates','physics_randomization_fn']:
        assert ast.dump(a[name])==ast.dump(b[name]),name
    targets={'reach','lift','carry','inside','action_penalty','previous_action','rate_penalty','reward'}
    def reward_nodes(step):
        return {x.targets[0].id:ast.dump(x.value) for x in step.body if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id in targets}
    assert reward_nodes(a['step'])==reward_nodes(b['step'])
