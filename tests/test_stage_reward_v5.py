"""Progress, carryover, quality, linear home error and action-only penalties."""
import math
import jax
import jax.numpy as jp
import numpy as np
import pytest
from piper_sim_lab import stage_reward_v5 as r

Z=jp.zeros(7)


def f(**kw):
    defaults=dict(tcp_cube_distance=.22,cube_bowl_xy=.25,cube_center_above_rim=-.1,
        cube_table_contact=False,cube_over_opening=True,finger_width=.05,cube_grasp_width=.05,
        left_contact=False,right_contact=False,robot_cube_contact=False,robot_bowl_contact=False,
        joint_error_a=1.2,**{'home_error_'+str(i):1.2 for i in range(6)})
    defaults.update(kw)
    return r.Features(**defaults)


def m(stage):
    result=r.reset_memory()._replace(approach_complete=jp.array(stage>=1),grasp_complete=jp.array(stage>=2),
        transport_complete=jp.array(stage>=3),place_complete=jp.array(stage>=4),
        place_quality=jp.array(1. if stage>=4 else 0.),
        approach_gap_start=jp.array(.2),transport_xy_start=jp.array(.2),transport_height_start=jp.array(.1),
        opening_width_start=jp.array(.05),opening_width_goal=jp.array(.06),home_distance_start=jp.array(1.))
    if stage>=2:result=result._replace(lifted=jp.array(True))
    if stage>=3:result=result._replace(bowl_reached=jp.array(True))
    if stage>=4:result=result._replace(released=jp.array(True),qualified_release=jp.array(True),
        first_floor_contact=jp.array(True),freefall_height=jp.array(.02),support_time=jp.array(.1))
    return result


def test_approach_is_linear_and_completion_is_not_added_twice():
    a=m(0)
    for gap,expected in [(.2,0.),(.15,.25),(.1,.5),(.05,.75),(0.,1.)]:
        assert float(r.main_score(a,f(tcp_cube_distance=.02+gap))[0])==pytest.approx(expected,abs=1e-6)
    goal=f(tcp_cube_distance=.02,cube_bottom_table=0.)
    b=r.advance_memory(a,goal,goal)
    assert b.approach_complete and r.main_score(b,goal)[0]==1.


def test_grasp_lift_cannot_be_farmed_without_contact_and_transition_is_continuous():
    a=m(1)
    goal=f(left_contact=True,right_contact=True,cube_bottom_table=.02)
    half=a._replace(grasp_time=jp.array(.05))
    assert float(r.main_score(half,goal._replace(cube_bottom_table=.01))[0])==pytest.approx(1.375)
    assert r.main_score(a,goal)[0]==1.
    ready=a._replace(grasp_time=jp.array(.1))
    assert r.main_score(ready,goal)[0]==2.
    b=r.advance_memory(ready,goal,goal)
    assert b.grasp_complete and r.main_score(b,goal)[0]==2.


def test_transport_progress_and_carryover():
    a=m(2)
    assert float(r.main_score(a,f(cube_bowl_xy=.15,cube_center_above_rim=-.05))[0])==pytest.approx(2.5)
    goal=f(cube_bowl_xy=.05,cube_center_above_rim=0.,left_contact=True)
    assert r.main_score(a,goal)[0]==3.
    b=r.advance_memory(a,goal,goal)
    assert b.transport_complete and r.main_score(b,goal)[0]==3.


@pytest.mark.parametrize('angle,cost',[(0.,0.),(math.pi/12,1/6),(math.pi/6,1/3),(math.pi/4,.5),(math.pi/3,.5)])
def test_face_penalty_is_bounded_and_only_in_first_two_stages(angle,cost):
    for stage in range(5):
        a=m(stage);out=r.reward_terms(a,a,f(face_angle=angle),Z,Z)
        assert float(out['face_cost'])==pytest.approx(cost if stage<2 else 0.,abs=1e-6)
        assert out['face_alignment']==0.


def test_closed_goal_penalty_is_independent_of_physical_release_and_only_stage4():
    for stage in range(5):
        a=m(stage);features=f(finger_width=.055)
        closed=r.reward_terms(a,a,features,Z,Z)
        opened=r.reward_terms(a,a,features._replace(opening_intent=True),Z,Z)
        assert float(closed['closed_cost'])==pytest.approx(.1 if stage==3 else 0.)
        assert opened['closed_cost']==0. and opened['opening_bonus']==0.
        if stage==3:
            assert float(opened['reward']-closed['reward'])==pytest.approx(.1,abs=1e-6)
            assert float(opened['progress'])==pytest.approx(.25,abs=1e-6)


@pytest.mark.parametrize('height,quality',[(.02,1.),(.07,1.),(.08,.5),(.0999,.5)])
def test_place_quality_carries_over_without_bonus_double_count(height,quality):
    a=m(3)._replace(released=jp.array(True),qualified_release=jp.array(True),first_floor_contact=jp.array(True),
        freefall_height=jp.array(height),support_time=jp.array(.1))
    goal=f(finger_width=.06,cube_inside_bowl=True,bowl_floor_contact=True,cube_relative_speed=0.,opening_intent=True)
    assert float(r.main_score(a,goal)[0])==pytest.approx(3.+quality)
    b=r.advance_memory(a,goal,goal)
    assert b.place_complete and float(b.place_quality)==quality
    assert float(r.main_score(b,goal)[0])==pytest.approx(3.+quality)
    out=r.reward_terms(a,b,goal,Z,Z)
    assert out['landing_bonus']==out['opening_bonus']==out['dwell_bonus']==0.


def test_high_drop_success_has_no_home_or_placement_credit():
    a=m(3)._replace(released=jp.array(True),qualified_release=jp.array(True),first_floor_contact=jp.array(True),
        freefall_height=jp.array(.12),high_drop_success=jp.array(True))
    goal=f(finger_width=.06,cube_inside_bowl=True,bowl_floor_contact=True)
    b=r.advance_memory(a,goal,goal)
    assert b.high_drop_success and not b.place_complete and not b.home_complete
    assert r.main_score(b,goal)[0]==3.5 and r.main_score(b,goal)[2]


def test_home_progress_is_linear_normalized_and_max_error_is_weighted_more():
    a=m(4)
    for remaining,expected in [(1.,0.),(.75,.2),(.5,.4),(.25,.6),(0.,.8)]:
        features=f(**{'home_error_'+str(i):.2+remaining for i in range(6)})
        assert float(r.main_score(a,features)[0])==pytest.approx(4.+expected,abs=1e-6)
    base=f(home_error_0=2.,**{'home_error_'+str(i):.5 for i in range(1,6)})
    worst=base._replace(home_error_0=1.9);other=base._replace(home_error_1=.4)
    assert float(r.home_distance(base)-r.home_distance(worst)) > 10*float(r.home_distance(base)-r.home_distance(other))


def test_home_needs_full_hold_and_does_not_add_an_extra_one_on_completion():
    a=m(4)._replace(home_time=jp.array(.1))
    goal=f(cube_inside_bowl=True,bowl_floor_contact=True,cube_relative_speed=0.,clearance=.15,
        joint_error_a=.2,joint_speed=.01,finger_width=.1,
        **{'home_error_'+str(i):.2 for i in range(6)})
    assert float(r.main_score(a,goal)[0])==pytest.approx(4.9)
    a=a._replace(home_time=jp.array(.195))
    b=r.advance_memory(a,goal,goal)
    assert b.home_complete and r.main_score(b,goal)[0]==5.
    assert r.reward_terms(a,b,goal,Z,Z)['reward']==5.
    mid=b._replace(place_quality=jp.array(.5))
    assert r.main_score(mid,goal)[0]==4.5


def test_stage_reference_is_frozen_and_failures_reset_credit():
    a=r.initialize(m(4),f())
    b=r.advance_memory(a,f(),f(home_error_0=2.))
    assert float(a.home_distance_start)==float(b.home_distance_start)
    bad=b._replace(invalid_release=jp.array(True),outside_drop=jp.array(True))
    out=r.reward_terms(b,bad,f(),Z,Z)
    assert out['reward']==-2. and not out['complete']
    assert r.reset_memory().place_quality==0.


def test_nonfinite_measurements_never_earn_progress_reward():
    a=m(0)
    out=r.reward_terms(a,a,f(face_angle=float('nan')),Z,Z)
    assert out['nonfinite'] and out['reward']==-1. and not out['complete']
