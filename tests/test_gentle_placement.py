"""Physical event ordering, dense placement incentives and anti-cheating."""
from types import SimpleNamespace
import jax.numpy as jp
import numpy as np
import pytest
from piper_sim_lab.gentle_placement import DEFAULTS,gentle_memory,gentle_update,gentle_reward

C=SimpleNamespace(**DEFAULTS,reach_scale=5.,grasp_distance=.02)


def f(h=.1,held=True,support=False,width=.05,clearance=0.,touch=False):
    return {k:jp.asarray(v) for k,v in dict(above=True,distance=max(h,0.),height=h,xy=0.,
        speed=0.,downward_speed=0.,width=width,span=.05,left=held,right=held,
        support=support,in_region=(-.001<=h<=.005),elevated=held and h+.003>.005,
        robot_cube=held or touch,robot_bowl=False,clearance=clearance).items()}


def update(t,b,a,n=1):
    for _ in range(n):t=gentle_update(t,b,a,.005,C);b=a
    return t


def p(inside=False):
    return dict(tcp_block=jp.array(.012),placed=jp.array(True),inside=jp.array(inside))


def test_lowering_and_support_have_no_20mm_reward_hole():
    scores=[]
    before=f();t=gentle_memory(before)
    for h in [.04,.025,.0201,.0199,.01,.005,0.]:
        feature=f(h=h,support=h==0.)
        t=update(t,before,feature);before=feature
        scores.append(float(gentle_reward(t,feature,p(),C)[0]))
    assert np.all(np.diff(scores)>=0)
    assert scores[4]>scores[2]


def supported_release():
    high=f();t=gentle_memory(high);floor=f(h=0.,support=True)
    t=update(t,high,floor,21)
    assert t['supported']
    released=f(h=0.,held=False,support=True,width=.1)
    t=update(t,floor,released,5)
    assert t['qualified_release'] and not t['premature']
    return t,released


def test_release_then_retreat_outscore_supported_holding():
    high=f();held=f(h=0.,support=True);t=update(gentle_memory(high),high,held,21)
    held_score=float(gentle_reward(t,held,p(),C)[0])
    t,released=supported_release()
    release_score=float(gentle_reward(t,released,p(),C)[0])
    far=f(h=0.,held=False,support=True,width=.1,clearance=.08)
    far_score=float(gentle_reward(t,far,p(True),C)[0])
    assert held_score<release_score<far_score
    assert release_score==pytest.approx(7.) and far_score==pytest.approx(8.)


def test_early_drop_and_regrasp_cannot_create_gentle_quality():
    high=f();t=gentle_memory(high)
    falling=f(h=.04,held=False,width=.1)
    t=update(t,high,falling,5)
    floor=f(h=0.,held=False,support=True,width=.1)
    t=update(t,falling,floor,25)
    again=f(h=0.,held=True,support=True)
    for _ in range(3):t=update(t,floor,again,20);t=update(t,again,floor,20)
    score,_,quality,retreat=gentle_reward(t,floor,p(True),C)
    assert t['premature'] and not t['qualified_release']
    assert float(score)==pytest.approx(3.5)
    assert float(quality)==float(retreat)==0.


def test_post_release_touch_latches_strict_failure_and_scores_no_retreat():
    t,released=supported_release()
    touch=f(h=0.,held=False,support=True,width=.1,clearance=.08,touch=True)
    t=update(t,released,touch,1)
    assert t['recontact_seen'] and t['recontacts']==1
    assert float(gentle_reward(t,touch,p(True),C)[3])==0
    far=f(h=0.,held=False,support=True,width=.1,clearance=.08)
    t=update(t,touch,far,60)
    assert t['clear_time']==0


def test_out_and_back_gives_no_extra_withdrawal_score():
    t,released=supported_release()
    far=f(h=0.,held=False,support=True,width=.1,clearance=.08)
    score=float(gentle_reward(t,far,p(True),C)[0])
    for _ in range(3):
        t=update(t,released,far);t=update(t,far,released)
    assert float(gentle_reward(t,far,p(True),C)[0])==score
    assert float(gentle_reward(t,released,p(True),C)[0])<score


def test_unwitnessed_landing_is_disqualified_and_later_touches_are_counted():
    fall=f(h=.1,held=False);t=gentle_memory(fall)
    floor=f(h=0.,held=False,support=True,width=.1)
    t=update(t,fall,floor,25)
    assert t['premature'] and not t['carried']
    touch=f(h=0.,held=True,support=True)
    t=update(t,floor,touch)
    assert t['recontact_seen']
    assert float(gentle_reward(t,touch,p(True),C)[0])==pytest.approx(3.5)
