import jax
import jax.numpy as jp
import numpy as np
import pytest

from piper_sim_lab.quadratic_release import replace_base_with_potential
from piper_sim_lab.quadratic_release import failure_cost, remaining_cost


@jax.jit
def shaped(scores, events, costs, terminal):
    def step(previous, x):
        score,event,cost,done=x
        r,p,f=replace_base_with_potential(score+event-cost,score,score,previous,done)
        return p,(r,f)
    return jax.lax.scan(step,jp.array(0.),(scores,events,costs,terminal))[1]


@pytest.mark.parametrize('pattern',['wait','oscillate','regrasp','large_raw_score'])
def test_waiting_cycles_and_regrasp_cannot_create_discounted_shaping_return(pattern):
    n=200
    raw={'wait':np.full(n,3.),'oscillate':np.tile([0.,3.],100),
         'regrasp':np.tile([3.,3.,0.,0.],50),'large_raw_score':np.full(n,100.)}[pattern]
    done=np.zeros(n);done[-1]=1
    r,f=map(np.asarray,shaped(jp.asarray(raw),jp.zeros(n),jp.zeros(n),jp.asarray(done)))
    assert np.dot(.99**np.arange(n),f)==pytest.approx(0.,abs=2e-5)
    assert r.sum()<=1e-5
    assert np.max(np.abs(f))<=4.+1e-6


@pytest.mark.parametrize('last',[0,9,119,199])
def test_true_early_termination_cancels_potential_without_discarding_actual_cost(last):
    n=last+1;raw=jp.full(n,3.);events=jp.zeros(n);costs=jp.full(n,.6)
    done=jp.zeros(n).at[-1].set(1.)
    r,f=map(np.asarray,shaped(raw,events,costs,done))
    assert np.dot(.99**np.arange(n),f)==pytest.approx(0.,abs=2e-5)
    assert np.dot(.99**np.arange(n),r)==pytest.approx(-.6*sum(.99**np.arange(n)),abs=2e-5)


def test_events_and_action_costs_preserved_and_total_never_exceeds_event_budget():
    n=200;raw=jp.full(n,3.)
    events=jp.zeros(n).at[10].set(3.).at[20].set(1.).at[30].set(2.).at[33].set(2.).at[40].set(3.)
    costs=jp.full(n,.01);done=jp.zeros(n).at[-1].set(1.)
    r,f=map(np.asarray,shaped(raw,events,costs,done))
    np.testing.assert_allclose(r-f,np.asarray(events-costs),atol=1e-6)
    assert r.sum()<=11.
    assert np.dot(.99**np.arange(n),r)==pytest.approx(
        np.dot(.99**np.arange(n),np.asarray(events-costs)),abs=2e-5)


def test_failure_base_already_removed_is_not_subtracted_again():
    # The failure path has already replaced base/event payments with -cost-tail.
    r,p,f=replace_base_with_potential(jp.array(-33.4),jp.array(0.),jp.array(3.),
                                     jp.array(2.5),jp.array(1.))
    assert float(r)==pytest.approx(-35.9,abs=1e-5)
    assert p==0 and f==-2.5


def test_returning_to_recorded_pose_and_grasp_does_not_pay_positive_shaping():
    raw=jp.tile(jp.array([0.,3.,0.,3.]),50)
    r,f=map(np.asarray,shaped(raw,jp.zeros(200),jp.zeros(200),jp.zeros(200).at[-1].set(1.)))
    assert f[1]>0
    assert np.all(f[2:]<=0.)
    assert float(np.maximum(f,0.).sum())<=4.


def test_per_second_cost_scales_timeout_floor_and_remaining_settlement_consistently():
    old=failure_cost(jp.array(0.))
    new=failure_cost(jp.array(0.),minimum=.6*.1)
    assert float(new)==pytest.approx(float(old)*.1)
    assert float(remaining_cost(new,80))==pytest.approx(float(remaining_cost(old,80))*.1,abs=1e-5)
