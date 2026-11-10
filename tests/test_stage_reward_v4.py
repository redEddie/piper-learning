"""Confirmed physical-release rules, height boundaries, independent home credit."""
import math

import jax
import jax.numpy as jp
import numpy as np
import pytest

from piper_sim_lab import stage_reward_v3 as v3, stage_reward_v4 as r

ZERO=jp.zeros(7)


def held(height=.095,**kwargs):
    options=dict(tcp_cube_distance=.005,cube_bottom_table=.1,cube_bowl_xy=.005,
        cube_center_above_rim=height-.067,tcp_floor_height=.125,
        cube_table_contact=False,left_contact=True,right_contact=True,robot_cube_contact=True,
        cube_center_floor_height=height,cube_over_opening=True)
    options.update(kwargs)
    return r.Features(**options)


def carrying():
    return r.reset_memory()._replace(grasp_time=jp.array(.1),lifted=jp.array(True),bowl_reached=jp.array(True))


def release(drop_height):
    before=held(drop_height+.025)
    detached=before._replace(left_contact=False,right_contact=False,robot_cube_contact=False,opening_intent=True)
    m=carrying();step=jax.jit(r.advance_memory)
    for i in range(4):m=step(m,before if i==0 else detached,detached)
    floor=detached._replace(cube_center_floor_height=.025,bowl_floor_contact=True,cube_inside_bowl=True)
    return m,step(m,detached,floor),floor


@pytest.mark.parametrize('height,bonus,terminal',[(.02,1.,False),(.07,1.,False),(.0701,.5,False),(.0999,.5,False),(.1,0.,True),(.15,0.,True)])
def test_drop_height_rewards_and_high_drop_success(height,bonus,terminal):
    old,m,f=release(height)
    out=r.reward_terms(old,m,f,ZERO,ZERO)
    assert m.released and m.qualified_release and not m.invalid_release
    assert float(m.freefall_height)==pytest.approx(height,abs=2e-6)
    assert float(out['landing_bonus'])==bonus
    assert float(out['opening_bonus'])==.5
    assert bool(out['terminal_success'])==terminal and bool(out['complete'])==terminal
    assert out['drop_cost']==0.
    if terminal:assert out['main'] < 5.  # no added final-home reward for high drops


def test_outside_drop_costs_two_once_and_is_not_success():
    before=held();free=before._replace(left_contact=False,right_contact=False,
        robot_cube_contact=False,opening_intent=True,cube_over_opening=False,cube_bowl_xy=.2)
    old=carrying();m=old
    for i in range(4):
        previous=m;m=r.advance_memory(m,before if i==0 else free,free)
    out=r.reward_terms(previous,m,free,ZERO,ZERO)
    assert m.invalid_release and m.outside_drop and out['drop_cost']==2.
    assert out['reward']==-2. and not out['complete'] and not out['terminal_success']
    assert r.reward_terms(m,m,free,ZERO,ZERO)['drop_cost']==0.


def test_opening_goal_requires_physical_separation_and_can_be_latched():
    m=carrying();f=held(opening_intent=True)
    assert r.reward_terms(m,m,f,ZERO,ZERO)['opening_bonus']==0.
    m=m._replace(released=jp.array(True))
    free=f._replace(left_contact=False,right_contact=False,robot_cube_contact=False)
    assert r.reward_terms(m,m,free,ZERO,ZERO)['opening_bonus']==.5
    assert r.reward_terms(m,m,free._replace(opening_intent=False),ZERO,ZERO)['opening_bonus']==0.


def test_no_preparation_decay_or_dwell_bonus_and_away_cost_remains():
    m=carrying()._replace(prepare_time=jp.array(100.))
    f=held(tcp_floor_height=.065)
    assert float(r.main_score(m,f)[0])==pytest.approx(float(r.main_score(m._replace(prepare_time=jp.array(0.)),f)[0]))
    after=m._replace(target_count=jp.array(2))
    assert r.reward_terms(m,after,f,ZERO,ZERO)['dwell_bonus']==0.
    near=m._replace(best_carry_xy=jp.array(.05))
    far=f._replace(cube_bowl_xy=.15)
    assert float(r.reward_terms(near,near,far,ZERO,ZERO)['away_cost'])==pytest.approx(.5)


def test_home_progress_is_independent_of_clearance_but_completion_is_not():
    _,m,f=release(.02)
    m=m._replace(support_time=jp.array(.1))
    f=f._replace(clearance=.02,joint_error_a=2.,**{'home_error_'+str(i):2. for i in range(6)})
    main,_,complete=r.main_score(m,f)
    oldmain=v3.main_score(r.legacy(m),v3.Features(*f[:27]))[0]
    assert float(main-oldmain)==pytest.approx(.5*math.exp(-2.),abs=1e-6)
    assert not complete
    # Even an A pose cannot finish without the original80mm clearance.
    at_a=f._replace(joint_error_a=0.,joint_speed=0.,**{'home_error_'+str(i):0. for i in range(6)})
    assert not r.main_score(m._replace(home_time=jp.array(.2)),at_a)[2]


def test_low_drop_still_requires_stationary_home_hold_for_completion():
    _,m,f=release(.02)
    m=m._replace(support_time=jp.array(.1));f=f._replace(clearance=.15,joint_error_a=0.,joint_speed=0.)
    for i in range(40):
        m=r.advance_memory(m,f,f)
        if i<39:assert not r.main_score(m,f)[2]
    assert r.main_score(m,f)[2]
    assert r.main_score(m,f)[0]==5.
    assert r.reward_terms(m,m,f,ZERO,ZERO)['reward']==6.5


def test_actual_landing_is_required_not_just_detachment_or_tcp_height():
    m,_,f=release(.02)
    assert m.released and not m.qualified_release
    assert r.reward_terms(m,m,f._replace(cube_inside_bowl=False,bowl_floor_contact=False),ZERO,ZERO)['landing_bonus']==0.


def test_nonfinite_added_geometry_is_failure():
    _,m,f=release(.02)
    out=r.reward_terms(m,m,f._replace(cube_center_floor_height=float('nan')),ZERO,ZERO)
    assert out['nonfinite'] and out['reward']==-1. and not out['complete']


def test_transient_floor_contact_inside_catchment_is_not_an_outside_miss():
    m,_,f=release(.02)
    transient=f._replace(bowl_floor_contact=False,cube_inside_bowl=False,cube_table_contact=True)
    m2=r.advance_memory(m,transient,transient)
    assert not m2.invalid_release and not m2.outside_drop
    assert not m2.qualified_release
    landed=r.advance_memory(m2,transient,f)
    assert landed.qualified_release and not landed.invalid_release
