"""Every joint earns distant-home progress; release and final completion stay v2."""
import math

import jax
import jax.numpy as jp
import numpy as np
import pytest

from piper_sim_lab import stage_reward_v2 as v2, stage_reward_v3 as v3

ZERO=jp.zeros(7)


def features(errors, **overrides):
    base=v2.Features(joint_error_a=max(errors),joint_speed=0.,clearance=.15,
        bowl_floor_contact=True,cube_table_contact=False,robot_cube_contact=False,
        robot_bowl_contact=False,opening_intent=True)
    base=base._replace(**overrides)
    return v3.Features(*base,*errors)


def released_memory():
    return v3.reset_memory()._replace(released=jp.array(True),qualified_release=jp.array(True),support_time=jp.array(.1))


@pytest.mark.parametrize('joint',range(6))
def test_improving_any_joint_earns_reward_even_when_worst_joint_stays_far(joint):
    errors=np.full(6,math.radians(120.))
    before=features(errors)
    errors[joint]=math.radians(110.)
    after=features(errors)
    m=released_memory()
    assert before.joint_error_a == after.joint_error_a
    a=v3.reward_terms(m,m,before,ZERO,ZERO)
    b=v3.reward_terms(m,m,after,ZERO,ZERO)
    assert float(b['reward']) > float(a['reward'])+1e-4
    assert float(v2.main_score(m,v2.Features(*before[:21]))[0]) == 4.5
    assert float(v2.main_score(m,v2.Features(*after[:21]))[0]) == 4.5


def test_progress_has_no_one_radian_dead_zone_and_incomplete_score_is_bounded():
    m=released_memory()
    values=[float(v3.main_score(m,features([math.radians(e)]*6))[0]) for e in (180,120,90,60,45,20,3,0)]
    assert np.all(np.diff(values)>0)
    assert 4.5 < values[0] < values[-1] <= 4.990001
    assert not v3.main_score(m,features([0.]*6))[2]


@pytest.mark.parametrize('error,speed,clearance,touch,expected',[
    (2.9,.049,.081,False,True),(3.1,.049,.081,False,False),
    (2.9,.051,.081,False,False),(2.9,.049,.079,False,False),
    (2.9,.049,.081,True,False)])
def test_original_completion_thresholds_and_hold_time_match_v2(error,speed,clearance,touch,expected):
    f=features([math.radians(error)]*6,joint_speed=speed,clearance=clearance,robot_cube_contact=touch)
    original=v2.Features(*f[:21]);a,b=released_memory(),released_memory()
    advance=jax.jit(v3.advance_memory)
    for i in range(40):
        a=v2.advance_memory(a,original,original)
        b=advance(b,f,f)
        for x,y in zip(a,b):
            np.testing.assert_allclose(x,y,atol=1e-7)
        assert bool(v2.main_score(a,original)[2]) == bool(v3.main_score(b,f)[2])
        if i<39:
            assert not bool(v3.main_score(b,f)[2])
    assert bool(v3.main_score(b,f)[2]) == expected
    if expected:
        assert float(v3.main_score(b,f)[0]) == 5.


@pytest.mark.parametrize('phase',range(4))
def test_earlier_phases_are_numerically_unchanged(phase):
    m=v3.reset_memory()._replace(grasp_time=jp.array(.1 if phase else 0.),
        lifted=jp.array(phase>=2),bowl_reached=jp.array(phase>=3))
    f=features([2.]*6,tcp_cube_distance=.01 if phase else .3,
        left_contact=phase>0,right_contact=phase>0,cube_bowl_xy=.01,
        tcp_floor_height=.055,cube_bottom_table=.05)
    a=v2.reward_terms(m,m,v2.Features(*f[:21]),ZERO,ZERO)
    b=v3.reward_terms(m,m,f,ZERO,ZERO)
    assert int(b['section']) == phase
    for key in a:
        np.testing.assert_equal(a[key],b[key])


@pytest.mark.parametrize('failure',['invalid_release','outside_drop'])
def test_failures_cannot_earn_dense_home_reward(failure):
    m=released_memory()._replace(**{failure:jp.array(True)})
    out=v3.reward_terms(m,m,features([0.]*6),ZERO,ZERO,v3.RewardSettings(drop_weight=0.))
    assert out['main'] == 0. and out['reward'] == 0. and not out['complete']


def test_nonfinite_extra_joint_measurement_is_failure():
    f=features([1.]*6)._replace(home_error_3=float('nan'))
    m=released_memory()
    out=v3.reward_terms(m,m,f,ZERO,ZERO)
    assert out['nonfinite'] and out['reward'] == -1. and not out['complete']
