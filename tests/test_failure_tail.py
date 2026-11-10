import jax
import jax.numpy as jp
import numpy as np
import pytest
from mujoco_playground._src.mjx_env import State
from brax.training.agents.ppo.losses import compute_gae

from piper_sim_lab.quadratic_release import (
    QuadraticRelease, EVENT_METRICS, finish_normal_step, absorb_failure,
    penalty_memory,penalty_update,pending_cost,remaining_cost)
from piper_sim_lab.wrappers import wrap_for_training
from piper_sim_lab.quadratic_penalty_audit import Ledger


class ToyFailure(QuadraticRelease):
    def __init__(self,horizon=5,cost=3.,fast=False):
        self.finite_horizon_steps=horizon;self.cost=cost;self.fast=fast
        if fast:self.failure_discount=.99

    def reset(self,key):
        z=jp.array(0.,jp.float32)
        metrics={k:z for k in EVENT_METRICS}
        metrics.update(nonfinite=z,capacity_overflow=z,penalty_cost=z,failure_cost=z,
            failure_onset=z,failure_absorbing=z,penalty_positive_base=z,controller_interruption=z)
        if self.fast:metrics.update(failure_tail_cost=z,failure_early_reset=z)
        return State(data=jp.zeros(1),obs=jp.zeros(2),reward=z,done=z,metrics=metrics,
            info=dict(task_failed=jp.array(False),controller_failed=jp.array(False),failure_absorbing=jp.array(False),
                failure_cost=z,failure_settled=jp.array(False),penalty_total=z,penalty_episode_steps=jp.array(0,jp.int32)))

    def _normal_step(self,state,action):
        fail=action[0]>.5
        metrics=dict(state.metrics,gentle_reward=jp.array(10.),reward_reach=jp.array(1.),
            release_bonus_release=jp.array(2.),success=jp.array(1.),
            action_penalty=fail.astype(jp.float32)*.1,
            action_rate_penalty=fail.astype(jp.float32)*.2,
            gentle_impact_cost=fail.astype(jp.float32)*.3,
            gentle_recontact_cost=fail.astype(jp.float32)*.4)
        updated=state.replace(data=state.data+1,obs=state.obs+1,reward=jp.array(12.),
            metrics=metrics,info=dict(state.info,task_failed=fail))
        return finish_normal_step(state,updated,jp.array(self.cost),self.finite_horizon_steps,
                                  early_reset=self.fast)

    def step(self,state,action):
        if self.fast:return self._normal_step(state,action)
        # CPU-only absorbing reference: no native physics/vmap branch effects.
        def done(_):
            return state.replace(reward=jp.array(0.),done=jp.array(1.),
                metrics=dict(state.metrics,penalty_cost=jp.array(0.)))
        return jax.lax.cond(state.info['penalty_episode_steps']>=self.finite_horizon_steps,done,
            lambda _:jax.lax.cond(state.info['failure_absorbing'],
                lambda _:absorb_failure(state,self.finite_horizon_steps),
                lambda _:self._normal_step(state,action),None),None)


def test_no_early_reset_other_slot_unaffected_final_step_is_true_termination():
    env=wrap_for_training(ToyFailure(),episode_length=5)
    state=jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(1),2));step=jax.jit(env.step)
    rewards=[]
    for i in range(5):
        state=step(state,jp.array([[1. if i==1 else 0.],[0.]]));rewards.append(np.asarray(state.reward))
        if i==1:failed_data=np.asarray(state.data[0]).copy()
        if 1<=i<4:
            np.testing.assert_array_equal(state.data[0],failed_data)
            assert state.info['penalty_episode_steps'][0]==i+1
            assert state.info['AutoResetWrapper_done_count'][0]==0
            assert state.done[0]==0
            assert state.metrics['success'][0]==0
            assert state.metrics['release_bonus_release'][0]==0
        if i>=2:assert state.metrics['action_penalty'][0]==0
    np.testing.assert_array_equal(state.done,[1,1])
    np.testing.assert_array_equal(state.info['truncation'],[0,0])
    np.testing.assert_array_equal(state.info['AutoResetWrapper_done_count'],[1,1])
    np.testing.assert_array_equal(state.info['episode_metrics']['length'],[5,5])
    np.testing.assert_allclose(np.array(rewards)[:,0],[9,-4,-3,-3,-3],atol=1e-6)
    np.testing.assert_allclose(state.info['episode_metrics']['sum_reward'],[-4,45],atol=1e-6)
    assert state.info['episode_metrics']['failure_onset'][0]==1
    assert state.info['failure_absorbing'][0]==False
    np.testing.assert_array_equal(state.data,jp.zeros((2,1)))


def test_actions_do_not_change_failure_cost_or_frozen_state():
    task=ToyFailure(horizon=20,cost=3.391)
    state=task.step(task.reset(jax.random.key(0)),jp.ones(1))
    frozen=np.asarray(state.data).copy()
    for i in range(18):
        state=jax.jit(task.step)(state,jp.array([(-1.)**i]))
        assert float(state.reward)==pytest.approx(-3.391)
        np.testing.assert_array_equal(state.data,frozen)
        assert state.metrics['reward_reach']==0
        assert state.metrics['release_bonus_release']==0


def test_absorbing_failure_charges_last_step_once_and_never_after_horizon():
    task=ToyFailure(horizon=3,cost=.6)
    state=task.step(task.reset(jax.random.key(0)),jp.ones(1))
    state=task.step(state,jp.zeros(1));assert state.done==0
    state=task.step(state,jp.zeros(1));assert state.done==1
    assert float(state.reward)==pytest.approx(-.6)
    total=float(state.info['penalty_total'])
    state=task.step(state,jp.ones(1));assert state.done==1 and state.reward==0
    assert float(state.info['penalty_total'])==total


def test_before_grasp_failure_still_has_minimum_unfinished_cost():
    task=ToyFailure(horizon=3,cost=0.)
    state=task.step(task.reset(jax.random.key(0)),jp.ones(1))
    assert float(state.info['failure_cost'])==pytest.approx(.6)
    assert float(state.reward)==pytest.approx(-1.6)
    state=task.step(state,jp.zeros(1));assert float(state.reward)==pytest.approx(-.6)


def test_horizon_mismatch_cannot_silently_cancel_tail():
    with pytest.raises(ValueError,match='horizon'):
        wrap_for_training(ToyFailure(horizon=5),episode_length=3)


def test_gae_includes_all_tail_costs_and_last_step():
    task=ToyFailure(horizon=5,cost=3.)
    state=task.reset(jax.random.key(0));rewards=[];term=[]
    for i in range(5):
        state=task.step(state,jp.array([1. if i==1 else 0.]))
        rewards.append(state.reward);term.append(state.done)
    r=jp.stack(rewards)[:,None];t=jp.stack(term)[:,None]
    values,_=compute_gae(truncation=jp.zeros_like(t),termination=t,rewards=r,
        values=jp.zeros_like(r),bootstrap_value=jp.zeros(1),lambda_=1.,discount=.99)
    expected=sum(.99**i*float(v) for i,v in enumerate(rewards))
    assert float(values[0,0])==pytest.approx(expected,abs=1e-5)


def test_jax_penalty_ledger_matches_independent_reference_and_cannot_reset():
    base=dict(carried=jp.array(True),lost_contact_time=jp.array(0.),released=jp.array(False),release_paid=jp.array(False))
    m=penalty_memory(base);ledger=Ledger(active=True)
    for i,(xy,z,grip,released) in enumerate([(.1,.17,True,False),(.1,.17,False,False),
            (.05,.055,False,True),(0.,.025,False,True),(0.,.055,True,True)]):
        f=dict(xy=jp.array(xy),tcp_height=jp.array(z),left=jp.array(grip),right=jp.array(grip))
        new=dict(m,released=jp.array(released),lost_contact_time=jp.array(
            0. if grip else (.02 if released else .005)))
        first=bool(m['carried']) and not grip and float(m['lost_contact_time'])==0 and not bool(m['released'])
        ledger.observe(carried=True,first_loss=first,release_event=not bool(m['released']) and released,
            release_paid=False,xy=xy,tcp_height=z)
        m=penalty_update(m,new,f)
        assert float(pending_cost(m,f))==pytest.approx(ledger.cost(xy,z),abs=1e-6)
    assert float(pending_cost(m,f))==pytest.approx(3.3911466667,abs=1e-6)


@pytest.mark.parametrize('failure_step',[1,5,19,20])
def test_fast_reset_has_same_discounted_return_as_absorbing_reference(failure_step):
    slow=ToyFailure(horizon=20,cost=3.)
    fast=ToyFailure(horizon=20,cost=3.,fast=True)
    a=slow.reset(jax.random.key(0));b=fast.reset(jax.random.key(0))
    ra=[];rb=[]
    for i in range(20):
        action=jp.array([1. if i+1==failure_step else 0.])
        a=slow.step(a,action);ra.append(float(a.reward))
        if i<failure_step:
            b=fast.step(b,action);rb.append(float(b.reward))
    assert b.done==1
    va=sum(.99**i*v for i,v in enumerate(ra));vb=sum(.99**i*v for i,v in enumerate(rb))
    assert va==pytest.approx(vb,abs=2e-5)


def test_fast_reset_immediately_frees_slot_without_bootstrapping_next_episode():
    env=wrap_for_training(ToyFailure(horizon=5,cost=3.,fast=True),episode_length=5)
    st=jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(5),2));step=jax.jit(env.step)
    rewards=[];dones=[]
    for i in range(5):
        st=step(st,jp.array([[1. if i==1 else 0.],[0.]]))
        rewards.append(st.reward);dones.append(st.done)
        if i==1:
            np.testing.assert_array_equal(st.done,[1,0])
            assert st.info['penalty_episode_steps'][0]==0
            assert st.info['truncation'][0]==0
            assert st.info['episode_metrics']['length'][0]==2
            assert float(st.metrics['failure_tail_cost'][0])==pytest.approx(8.821197,abs=1e-5)
        if i==2:
            assert st.info['penalty_episode_steps'][0]==1
            assert st.data[0,0]==1
    r=jp.stack(rewards);t=jp.stack(dones)
    values,_=compute_gae(truncation=jp.zeros_like(t),termination=t,rewards=r,
        values=jp.zeros_like(r),bootstrap_value=jp.zeros(2),lambda_=1.,discount=.99)
    expected=9.+.99*(-4.-8.821197)
    assert float(values[0,0])==pytest.approx(expected,abs=1e-5)


def test_remaining_cost_has_no_off_by_one_and_handles_no_discount():
    assert float(remaining_cost(3.,0))==0.
    assert float(remaining_cost(3.,1))==pytest.approx(2.97)
    assert float(remaining_cost(3.,190,gamma=1.))==570.


def test_training_config_rejects_discount_mismatch(tmp_path):
    import yaml
    from pathlib import Path
    from piper_sim_lab.train import load_config
    cfg=yaml.safe_load(Path('configs/show103/quadratic-fast-reset.yaml').read_text())
    cfg['training']['discounting']=.98
    path=tmp_path/'mismatch.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='failure_discount'):
        load_config(path)


def test_fast_reset_terminal_settlement_is_idempotent_without_reset():
    task=ToyFailure(horizon=20,cost=3.,fast=True)
    st=task.step(task.reset(jax.random.key(0)),jp.ones(1))
    total=float(st.info['penalty_total'])
    for action in [0.,1.,0.]:
        st=task.step(st,jp.array([action]))
        assert st.done==1 and st.reward==0
        assert st.metrics['failure_tail_cost']==0 and st.metrics['failure_onset']==0
        assert float(st.info['penalty_total'])==total


def test_fast_settlement_rejects_action_repeat_discount_mismatch():
    with pytest.raises(ValueError,match='action_repeat'):
        wrap_for_training(ToyFailure(fast=True),episode_length=5,action_repeat=2)


def test_controller_interruption_does_not_receive_task_failure_debt():
    task=ToyFailure(horizon=20,cost=3.,fast=True)
    old=task.reset(jax.random.key(0))
    updated=old.replace(reward=jp.array(12.),info=dict(old.info,controller_failed=jp.array(True)))
    st=finish_normal_step(old,updated,jp.array(3.),20,early_reset=True)
    assert st.done==1 and st.reward==9
    assert st.metrics['failure_tail_cost']==0 and st.metrics['failure_onset']==0
    assert st.metrics['controller_interruption']==1
    assert not st.info['failure_settled'] and st.info['failure_cost']==0
