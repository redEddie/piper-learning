from types import SimpleNamespace

import jax.numpy as jp
import pytest

from piper_sim_lab.gentle_placement import DEFAULTS as G
from piper_sim_lab.controlled_release import DEFAULTS, release_memory, release_update, release_reward, release_step_reward
from piper_sim_lab.balanced_release import bonuses

C=SimpleNamespace(**dict({**G,**DEFAULTS},release_balanced_rewards=True,
    release_require_stable_target=True,release_require_landing_for_retreat=True),
    bowl_r_bottom=.042,block_half=.025,reach_scale=5.)


def f(tcp=.055,held=True,support=False,speed=0.,xy=0.,height=.03,clear=0.,touch=False):
    return {k:jp.asarray(v) for k,v in dict(above=True,left=held,right=held,
        elevated=held,support=support,in_region=False,xy=xy,height=height,speed=speed,
        downward_speed=0.,robot_cube=held,robot_bowl=touch,clearance=clear,
        tcp_height=tcp,tcp_rel=jp.array([0.,0.,tcp]),opening_intent=not held,
        grip_reference=.05 if not held else 0.).items()}


def advance(t,b,a,n):
    for _ in range(n):t=release_update(t,b,a,.005,C);b=a
    return t


def test_two_nonoverlapping_dwell_windows_then_no_more_even_after_reentry():
    a=f();initial=release_memory(a,C)
    t=advance(initial,a,a,19);assert t['target_count']==0
    t=advance(t,a,a,1);assert t['target_count']==1
    t=advance(t,a,a,20);assert t['target_count']==2
    t=advance(t,a,a,100);assert t['target_count']==2
    b=f(tcp=.10);t=advance(t,a,b,40);t=advance(t,b,a,100)
    assert t['target_count']==2
    assert float(bonuses(initial,t)['dwell'])==1.


def test_interrupted_short_dwell_does_not_combine_across_reentry():
    a=f();b=f(tcp=.10);t=release_memory(a,C)
    for _ in range(3):
        t=advance(t,b,a,10);t=advance(t,a,b,10)
    assert t['target_count']==0


def test_initial_proximity_no_credit_and_oscillation_cannot_refill_progress():
    a=f(tcp=.10);b=f();initial=release_memory(a,C)
    t=advance(initial,a,a,20)
    assert t['progress_credit']==0
    t=advance(t,a,b,20);first=float(t['progress_credit'])
    for _ in range(3):t=advance(t,b,a,20);t=advance(t,a,b,20)
    assert float(t['progress_credit'])==pytest.approx(first)
    assert 0.<first<=3.
    at_target=release_memory(b,C)
    assert advance(at_target,b,b,40)['progress_credit']==0


def test_landing_is_only_floor_contact_and_relative_speed_for_100ms():
    a=f();out=f(held=False);t=advance(release_memory(a,C),a,out,5)
    # Deliberately outside the old XY/height predicates: must still count landing.
    landed=f(held=False,support=True,xy=.04,height=.02)
    t=advance(t,out,landed,19);assert not t['supported'];assert not t['landing_paid']
    t=advance(t,landed,landed,1)
    assert t['supported'] and t['landing_paid']


def test_wall_contact_or_fast_floor_contact_does_not_count_landing():
    a=f();out=f(held=False);t=advance(release_memory(a,C),a,out,5)
    wall=f(held=False,support=False)
    t=advance(t,out,wall,60);assert not t['landing_paid']
    fast=f(held=False,support=True,speed=.021)
    t=advance(t,wall,fast,60);assert not t['landing_paid']


def test_broken_floor_contact_resets_stability_timer_and_bounce_pays_once():
    a=f();out=f(held=False);initial=release_memory(a,C)
    t=advance(initial,a,out,5);land=f(held=False,support=True)
    t=advance(t,out,land,10);t=advance(t,land,out,1)
    t=advance(t,out,land,10);assert not t['landing_paid']
    t=advance(t,land,land,10);assert t['landing_paid']
    for _ in range(3):t=advance(t,land,out,5);t=advance(t,out,land,25)
    assert float(bonuses(initial,t)['landing'])==2.


def test_release_regrasp_and_later_clearance_cannot_repeat_bonuses():
    a=f();out=f(held=False);initial=release_memory(a,C)
    t=advance(initial,a,out,5);land=f(held=False,support=True,clear=.08)
    t=advance(t,out,land,60)
    paid=bonuses(initial,t)
    assert float(paid['release'])==2. and float(paid['landing'])==2.
    assert float(paid['withdrawal'])==3.
    grasp=f();t=advance(t,land,grasp,20);t=advance(t,grasp,out,5)
    t=advance(t,out,land,100)
    assert t['recontact_seen']
    assert float(bonuses(initial,t)['withdrawal'])==3.
    assert float(bonuses(initial,t)['release'])==2.
    p=dict(tcp_block=jp.array(.1),placed=jp.array(True),inside=jp.array(True))
    assert float(release_reward(t,land,p,C)[2])==0.


def test_bad_first_drop_cannot_unlock_events_by_later_regrasp():
    high=f(tcp=.12);t=release_memory(high,C);out=f(tcp=.12,held=False)
    t=advance(t,high,out,5)
    a=f();t=advance(t,out,a,20);t=advance(t,a,f(held=False),5)
    land=f(held=False,support=True,clear=.08);t=advance(t,land,land,100)
    assert not t['release_paid'] and not t['landing_paid'] and not t['withdrawal_paid']


@pytest.mark.parametrize('original',[.25,1.8,2.7])
def test_original_approach_grasp_transport_reward_is_preserved_before_placing(original):
    a=f(tcp=.15,held=False);a['above']=jp.array(False)
    old=release_memory(a,C);new=advance(old,a,a,20)
    p=dict(tcp_block=jp.array(.1),placed=jp.array(False),inside=jp.array(False))
    r,*_=release_step_reward(old,new,a,p,C,original,.002,.003,.1)
    assert float(r)==pytest.approx(original-.005)
