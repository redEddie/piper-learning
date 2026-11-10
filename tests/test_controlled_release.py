from types import SimpleNamespace
import jax.numpy as jp
import pytest
from piper_sim_lab.gentle_placement import DEFAULTS as G
from piper_sim_lab.controlled_release import DEFAULTS,release_memory,release_update,release_reward,ControlledRelease
from piper_sim_lab.smooth_grip import MOVE,GRIP

C=SimpleNamespace(**dict(G,**DEFAULTS),bowl_r_bottom=.042,block_half=.025,reach_scale=5.)
C3=SimpleNamespace(**dict(vars(C),release_require_stable_target=True,release_require_landing_for_retreat=True))

def f(tcp=.055,held=True,support=False,opening=False,clear=0.,touch=False,speed=0.):
    return {k:jp.asarray(v) for k,v in dict(above=True,left=held,right=held,elevated=held,
        support=support,height=0. if support else max(tcp-.025,0.),xy=0.,speed=speed,downward_speed=speed,
        in_region=support,robot_cube=held or touch,robot_bowl=False,clearance=clear,
        tcp_height=tcp,tcp_rel=jp.array([0.,0.,tcp+.003]),opening_intent=opening,
        grip_reference=.05 if opening else 0.).items()}

def advance(t,b,a,n=5):
    for _ in range(n):t=release_update(t,b,a,.005,C);b=a
    return t

def test_release_uses_tcp_height_without_requiring_prior_floor_support():
    before=f();t=release_memory(before);after=f(held=False,opening=True)
    t=advance(t,before,after)
    assert t['qualified_release'] and not t['supported'] and not t['premature']
    assert float(t['release_tcp_height'])==pytest.approx(.055)
    assert float(t['release_height'])==pytest.approx(.030)

def test_first_high_release_cannot_be_repaired_by_later_lowering_or_regrasp():
    high=f(tcp=.12);t=release_memory(high);out=f(tcp=.12,held=False,opening=True)
    t=advance(t,high,out)
    again=f();t=advance(t,out,again);t=advance(t,again,f(held=False,opening=True))
    assert t['premature'] and not t['qualified_release']
    assert float(t['release_tcp_height'])==pytest.approx(.12)

def test_slip_without_opening_intent_gets_no_qualified_release():
    before=f();t=advance(release_memory(before),before,f(held=False,opening=False))
    assert t['premature'] and not t['qualified_release']

def test_fast_arm_or_object_at_detach_is_not_a_controlled_release():
    before=f(tcp=.053);after=f(tcp=.055,held=False,opening=True)
    t=advance(release_memory(before),before,after)
    assert t['premature']
    before=f(speed=.5);t=advance(release_memory(before),before,f(held=False,opening=True))
    assert t['premature']

def test_opening_unlock_preserves_closing_and_preseparation_interlocks():
    obj=SimpleNamespace(_finger_qpos_ids=[0,1],_config=C)
    data=SimpleNamespace(qpos=jp.array([.029,-.029]))
    info=dict(coord_phase=GRIP,actuated_goal=.05,grip_start_width=.05)
    assert not ControlledRelease._arm_motion_allowed(obj,data,info)
    data.qpos=jp.array([.031,-.031])
    assert ControlledRelease._arm_motion_allowed(obj,data,info)
    info['actuated_goal']=0.
    assert not ControlledRelease._arm_motion_allowed(obj,data,info)
    info['coord_phase']=MOVE
    assert ControlledRelease._arm_motion_allowed(obj,data,info)

def test_release_then_clear_retreat_outscores_holding_in_height_band():
    before=f();t=release_memory(before);t=advance(t,before,before,30)
    p=dict(tcp_block=jp.array(.012),placed=jp.array(True),inside=jp.array(False))
    holding=float(release_reward(t,before,p,C)[0])
    opened=f(held=False,opening=True);t=advance(t,before,opened)
    released=float(release_reward(t,opened,p,C)[0])
    far=f(held=False,support=True,opening=True,clear=.08)
    p['inside']=jp.array(True)
    assert holding<released<float(release_reward(t,far,p,C)[0])
    touch=f(held=False,support=True,opening=True,clear=.08,touch=True)
    t=advance(t,opened,touch);t=advance(t,touch,far,60)
    assert t['recontact_seen'] and t['clear_time']==0
    assert float(release_reward(t,far,p,C)[2])==0.

def test_height_dwell_requires_actual_slow_motion_not_fast_band_crossing():
    before=f(tcp=.052);after=f(tcp=.0522)
    t=release_update(release_memory(before),before,after,.005,C3)
    assert t['target_time']==0
    stable=release_update(t,after,after,.005,C3)
    assert float(stable['target_time'])==pytest.approx(.005)

def test_withdrawal_bonus_requires_stable_landing_not_just_in_bowl_volume():
    before=f();opened=f(held=False,opening=True,clear=.08)
    t=advance(release_memory(before),before,opened)
    p=dict(tcp_block=jp.array(.1),placed=jp.array(True),inside=jp.array(True))
    assert float(release_reward(t,opened,p,C3)[3])==0.
    landed=f(held=False,opening=True,support=True,clear=.08)
    for _ in range(21):t=release_update(t,landed,landed,.005,C3)
    assert float(release_reward(t,landed,p,C3)[3])==pytest.approx(1.)
