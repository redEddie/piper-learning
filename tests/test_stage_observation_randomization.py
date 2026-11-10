"""Delayed held-cube observations must coexist with clean physical reward state."""
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
import yaml

from piper_sim_lab.tasks import make_env

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def env():
    cfg=yaml.safe_load((ROOT/'configs/show108/stage-reward-v5-mlp-lstm-observation-dr-wide-4096-plus2.yaml').read_text())
    return make_env(dict(cfg['environment'],obs_randomization=True,impl='jax',batch_capacity=1))


def held_data(env):
    data=mujoco.MjData(env.mj_model)
    data.qpos[:]=env.reset_bank['qpos'][2][0]
    data.ctrl[:]=env.reset_bank['ctrl'][2][0]
    mujoco.mj_forward(env.mj_model,data)
    return data


def test_delayed_held_cube_does_not_replace_clean_truth_or_command_memory(env):
    data=held_data(env)
    pose=env._block_pose(data)
    older=pose.at[2].add(-.15)
    oldest=pose.at[2].add(-.25)
    perception=env._reset_perception(data,jax.random.key(42))
    previous_robot=env._robot_features(data).at[18].add(-.08)
    perception=dict(perception,block_hist=jp.stack([older,oldest,oldest]),obj_delay=jp.float32(1.5),
        robot_delay=jp.float32(.5),robot_prev=previous_robot)
    action=jp.arange(7,dtype=jp.float32)/10
    before=data.qpos.copy()
    obs,updated=env._get_obs(data,action,env._predicates(data),perception,jax.random.key(43),gripper_goal=jp.float32(0.))
    actor,critic=np.asarray(obs['state']),np.asarray(obs['privileged'])
    assert actor.shape==(49,) and critic.shape==(66,)
    delayed=np.asarray((older[:3]+oldest[:3])/2-env._base_offset)
    assert np.max(np.abs(actor[25:28]-delayed)) <= env._config.obj_pos_noise_cap+1e-6
    assert critic[58]-actor[27] > .16  # actual held height versus stale observed height
    np.testing.assert_allclose(critic[-10:-7],np.asarray(pose[:3]-env._base_offset),atol=1e-6)
    np.testing.assert_allclose(actor[18:21],np.asarray(env._robot_features(data)[18:21])+[-.04,0,0],atol=1e-6)
    np.testing.assert_allclose(actor[34:37],actor[25:28]-actor[18:21],atol=1e-6)
    np.testing.assert_allclose(actor[37:40],actor[25:28]-actor[31:34],atol=1e-6)
    assert actor[41]==0  # CLOSE goal stays observable even with stale object/robot state
    np.testing.assert_array_equal(actor[-7:],action)
    np.testing.assert_array_equal(data.qpos,before)
    np.testing.assert_allclose(updated['block_hist'][0],pose,atol=1e-6)


def test_all_perception_draws_active_and_robot_shape_preserved(env):
    data=held_data(env)
    draws=jax.vmap(lambda k:env._reset_perception(data,k))(jax.random.split(jax.random.key(44),256))
    for name in ['obj_delay','bowl_delay','robot_delay','size_scale']:
        assert np.std(draws[name])>1e-3
    assert np.asarray(draws['hold_substeps']).max()>0
    assert np.asarray(draws['hold_substeps']).max()<=10
    assert np.std(draws['bowl_offset'])>1e-3
    assert np.abs(draws['bowl_offset']).max()<=.015+1e-6
    # No new episode-static cube bias is introduced.
    assert set(draws)=={'block_hist','robot_prev','obj_delay','robot_delay','hold_substeps','size_scale','bowl_offset','bowl_hist','bowl_delay'}


def test_bowl_has_independent_delayed_stream_and_clean_critic(env):
    data=held_data(env)
    bowl=env._bowl_pos(data)
    perception=env._reset_perception(data,jax.random.key(50))
    history=jp.stack([bowl-jp.array([x,0.,0.]) for x in [.01,.03,.05,.07]])
    perception=dict(perception,bowl_hist=history,bowl_delay=jp.float32(2.5),bowl_offset=jp.zeros(3))
    obs,updated=env._get_obs(data,jp.zeros(7),env._predicates(data),perception,jax.random.key(51),gripper_goal=jp.float32(0.))
    expected=np.asarray(bowl-env._base_offset)-[.04,0.,0.]
    np.testing.assert_allclose(obs['state'][31:34],expected,atol=1e-6)
    np.testing.assert_allclose(obs['privileged'][-4:-1],bowl-env._base_offset,atol=1e-6)
    np.testing.assert_allclose(updated['bowl_hist'][0],bowl,atol=1e-6)
    # The oldest supported endpoint (300ms) must select the actual 3-step-old position.
    perception=dict(perception,bowl_delay=jp.float32(3.))
    obs,_=env._get_obs(data,jp.zeros(7),env._predicates(data),perception,jax.random.key(51))
    np.testing.assert_allclose(obs['state'][31:34],np.asarray(bowl-env._base_offset)-[.05,0.,0.],atol=1e-6)


@pytest.mark.parametrize('stage',range(5))
def test_wide_spawns_keep_coupled_grasps_inside_requested_rectangle(env,stage):
    draw=jax.jit(jax.vmap(lambda k:env._reset_pose(k,jp.int32(stage))))
    qpos,ctrl,_=jax.tree.map(np.asarray,draw(jax.random.split(jax.random.key(1080+stage),1024)))
    assert np.isfinite(qpos).all() and np.isfinite(ctrl).all()
    cube=qpos[:,env._block_qpos_adr:env._block_qpos_adr+2]
    bowl=qpos[:,env._bowl_qpos_adr:env._bowl_qpos_adr+2]
    for xy in (cube,bowl):
        assert np.all(xy >= [.1-1e-6,-.3-1e-6])
        assert np.all(xy <= [.5+1e-6,.3+1e-6])
    if stage==0:
        for xy in (cube,bowl):
            assert xy[:,0].min()<.12 and xy[:,0].max()>.48
            assert xy[:,1].min()<-.28 and xy[:,1].max()>.28
    if stage<3: assert np.linalg.norm(cube-bowl,axis=-1).min()>=.14-1e-6
    if stage in (2,3):
        data=mujoco.MjData(env.mj_model)
        for q,u in zip(qpos[:16],ctrl[:16]):
            data.qpos[:]=q;data.ctrl[:]=u
            mujoco.mj_forward(env.mj_model,data)
            contact=env._native_contacts(data)
            assert contact['left'] and contact['right']
