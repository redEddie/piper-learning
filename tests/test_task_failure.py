import jax
import jax.numpy as jp
import numpy as np
import pytest

from piper_sim_lab.task_failure import DEFAULTS, task_failure_memory, task_failure_update


def memory(stage=0, carried=False):
    return task_failure_memory(dict(carried=jp.array(carried)),jp.array(stage))


def features(**changes):
    f=dict(left=False,right=False,elevated=False,table=True,support=False,table_speed=0.)
    f.update(changes)
    return {k:jp.array(v) for k,v in f.items()}


@jax.jit
def advance(t, f, n):
    def substep(_,old):
        return task_failure_update(old,old,f,.005,DEFAULTS)
    return jax.lax.fori_loop(0,n,substep,t)


@pytest.mark.parametrize('stage',[0,1])
def test_initial_table_cube_is_not_drop_and_grasp_times_out_at_12_seconds(stage):
    t=advance(memory(stage),features(),2399)
    assert not t['tf_failed'] and not t['tf_lifted']
    t=advance(t,features(),1)
    assert t['tf_failed'] and t['tf_reason']==1


def test_brief_contact_and_failed_attempt_allow_retry_before_deadline():
    t=advance(memory(),features(left=True,right=True),10)
    assert not t['tf_grasped']
    t=advance(t,features(),1000)
    assert not t['tf_failed']
    t=advance(t,features(left=True,right=True),20)
    assert t['tf_grasped'] and not t['tf_lifted']
    t=advance(t,features(left=True,right=True),1500)
    assert not t['tf_failed']  # A table grasp does not require lifting first.


@pytest.mark.parametrize('stage',[2,3])
def test_held_initializations_are_not_subject_to_grasp_deadline(stage):
    t=advance(memory(stage),features(table=False),4000)
    assert not t['tf_failed']


def test_confirmed_lift_then_outside_landing_requires_continuous_100ms():
    t=advance(memory(),features(left=True,right=True,table=False,elevated=True),20)
    assert t['tf_grasped'] and t['tf_lifted']
    t=advance(t,features(table=False,table_speed=.3),120)
    assert not t['tf_failed']  # Even a long free flight is not a landing.
    t=advance(t,features(table_speed=.03),100)
    assert not t['tf_failed']  # Moving contact does not establish stable landing.
    t=advance(t,features(),19)
    assert not t['tf_failed']
    t=advance(t,features(table=False),1)
    t=advance(t,features(),19)
    assert not t['tf_failed']  # Interrupted contact resets the dwell timer.
    t=advance(t,features(),1)
    assert t['tf_failed'] and t['tf_reason']==2
    t=advance(t,features(support=True,table=False),4000)
    assert t['tf_reason']==2  # Reason and failure cannot be repaired after onset.


def test_normal_bowl_release_and_landing_do_not_fail():
    t=advance(memory(3,True),features(table=False,table_speed=.2),100)
    t=advance(t,features(table=False,support=True),4000)
    assert not t['tf_failed']


def test_never_lifted_grasp_loss_on_table_remains_retryable():
    t=advance(memory(),features(left=True,right=True),20)
    t=advance(t,features(),3000)
    assert not t['tf_failed'] and t['tf_grasped'] and not t['tf_lifted']


def test_batched_task_monitors_do_not_leak_between_slots():
    initial=jax.tree.map(lambda a,b:jp.stack([a,b]),memory(),memory(3,True))
    fs=jax.tree.map(lambda a,b:jp.stack([a,b]),features(),features(table=False,support=True))
    result=jax.jit(jax.vmap(advance,in_axes=(0,0,None)))(initial,fs,2400)
    np.testing.assert_array_equal(result['tf_reason'],[1,0])


def test_deadline_contact_confirmation_wins_over_timeout():
    t=advance(memory(),features(),2380)
    t=advance(t,features(left=True,right=True),20)
    assert t['tf_grasped'] and not t['tf_failed']
