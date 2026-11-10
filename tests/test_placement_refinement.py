"""Placing-only reward timing and anti-duplication, independent of the simulator."""
from types import SimpleNamespace
import jax
import jax.numpy as jp
import numpy as np
import pytest
from piper_sim_lab.placement_refinement import DEFAULTS,placement_memory,placement_update,placement_terms
C=SimpleNamespace(**DEFAULTS,inside_reward=5.)
C3=SimpleNamespace(**dict(vars(C),place_episode_tracking=True,place_base_reward=1.5))

def f(h=.1,held=True,support=False,speed=0.,opening=.05,above=True,down=0.,xy=0.):
    return {k:jp.asarray(v) for k,v in dict(above=above,distance=np.hypot(xy,max(h,0)),
        height=h,xy=xy,speed=speed,downward_speed=down,width=opening,span=.05,
        left=held,right=held,support=support,in_region=(abs(xy)<=.017 and -.001<=h<=.005)).items()}


def advance(t,b,a,n=1):
    for _ in range(n):
        t=placement_update(t,b,a,.005,C);b=a
    return t


def test_hold_no_progress_and_phase_entry_no_free_credit():
    a=f();t=placement_memory(a);new=advance(t,a,a,100)
    assert float(new['progress_credit'])==0
    inactive=f(h=.3,above=False);t=placement_memory(inactive)
    new=advance(t,inactive,a)
    assert float(new['progress_credit'])==0
    assert float(new['best_distance'])==pytest.approx(.1)


def test_lowering_and_returning_cannot_repeat_progress():
    a=f();t=placement_memory(a);low=f(h=.02)
    t=advance(t,a,low);credit=float(t['progress_credit'])
    for _ in range(5):
        t=advance(t,low,a);t=advance(t,a,low)
    assert float(t['progress_credit'])==pytest.approx(credit)
    assert credit==pytest.approx(.08)


def test_drop_consumes_record_without_progress_or_gentle_credit():
    a=f();t=placement_memory(a);fall=f(h=.04,held=False,speed=.8,down=.8)
    t=advance(t,a,fall,5)
    floor=f(h=0,held=False,support=True,down=0,opening=.1)
    t=advance(t,fall,floor,40)
    assert bool(t['premature']) and not bool(t['qualified_release'])
    assert float(t['progress_credit'])==0
    assert float(t['impact_peak'])==pytest.approx(.8)
    assert float(t['opening_best'])==0
    terms=placement_terms(placement_memory(a),t,jp.array(True),C)
    assert float(terms['placement_reward'])==0


def test_supported_release_and_opening_reward_once():
    a=f();t=placement_memory(a);floor=f(h=0,support=True)
    t=advance(t,a,floor,21)
    assert bool(t['supported']) and not bool(t['released'])
    openf=f(h=0,held=False,support=True,opening=.08)
    t=advance(t,floor,openf,5)
    assert bool(t['qualified_release']) and not bool(t['premature'])
    assert float(t['opening_best'])==1
    credit=float(t['progress_credit'])
    for _ in range(3):
        t=advance(t,openf,floor,5);t=advance(t,floor,openf,5)
    assert float(t['opening_best'])==1
    assert float(t['progress_credit'])==pytest.approx(credit)
    assert float(placement_terms(placement_memory(a),t,jp.array(True),C)['placement_reward'])==5


def test_impact_uses_precontact_and_only_peak_increments():
    a=f(down=.8);t=placement_memory(a);floor=f(h=0,support=True,down=0)
    new=advance(t,a,floor)
    assert float(new['impact_peak'])==pytest.approx(.8)
    x=placement_terms(t,new,jp.array(False),C);assert float(x['impact'])==pytest.approx(8.)
    bounced=f(h=.001,support=False,down=.4)
    again=advance(new,floor,bounced);again=advance(again,bounced,floor)
    assert float(placement_terms(new,again,jp.array(False),C)['impact'])==0


def test_jit_state_update_has_same_contract():
    a=f();b=f(h=.05);t=placement_memory(a)
    update=jax.jit(lambda t:placement_update(t,a,b,.005,C))
    result=update(t)
    assert float(result['progress_credit'])==pytest.approx(.05)


def test_native_contact_sensor_addresses_and_floor_geometry():
    from pathlib import Path
    from unittest import mock
    import mujoco,yaml
    from piper_sim_lab.tasks import make_env
    from piper_sim_lab.pick_place import PiperPickPlace
    root=Path(__file__).resolve().parents[1]
    c=yaml.safe_load((root/'configs/show101/refine-8m.yaml').read_text())['environment']
    tables={k:np.zeros((1,6)) for k in ('pregrasp','grasp','preinside')}
    with mock.patch('mujoco_playground._src.mjx_env.put_model',return_value=None), \
         mock.patch.object(PiperPickPlace,'_build_reset_table',return_value=tables):
        env=make_env(dict(c,impl='warp'))
    assert env._mj_model.nsensor==3
    assert set(env._sensor_adr.values())=={0,1,2}
    data=mujoco.MjData(env._mj_model);data.qpos[:]=np.asarray(env._initial_qpos)
    data.qpos[env._block_qpos_adr:env._block_qpos_adr+7]=[.3,0,.0279,1,0,0,0]
    data.qpos[env._bowl_qpos_adr:env._bowl_qpos_adr+7]=[.3,0,0,1,0,0,0]
    data.ctrl[:]=np.asarray(env._initial_ctrl);mujoco.mj_forward(env._mj_model,data)
    contacts=env._place_contacts(data)
    assert contacts['support'] and not contacts['left'] and not contacts['right']
    features=env._place_features(data)
    assert abs(float(features['height'])+.0001)<1e-6


def test_v2_retains_basic_completion_credit_for_imperfect_placing():
    c=SimpleNamespace(**dict(vars(C),place_base_reward=1.5,place_hold_bridge=True,place_contact_activation=True))
    a=f();t=placement_memory(a)
    terms=placement_terms(t,t,jp.array(True),c,jp.array(True))
    assert float(terms['placement_reward'])==1.5
    terms=placement_terms(t,t,jp.array(True),c,jp.array(False))
    assert float(terms['placement_reward'])==0


def episode_feature(**kwargs):
    value=f(**kwargs)
    value['elevated']=jp.asarray(float(value['height'])+.003>C3.place_carry_clearance)
    return value


def episode_advance(t,b,a,n=1):
    for _ in range(n):
        t=placement_update(t,b,a,.005,C3);b=a
    return t


def test_episode_drop_before_phase_entry_cannot_be_repaired_by_recontact():
    high=episode_feature(above=False)
    t=placement_memory(high,True)
    fall=episode_feature(h=.04,held=False,above=False,speed=1.,down=1.)
    t=episode_advance(t,high,fall,4)
    assert bool(t['premature']) and bool(t['released'])
    floor=episode_feature(h=0,held=False,above=False,support=True,opening=.1)
    t=episode_advance(t,fall,floor,30)
    assert float(t['impact_peak'])==pytest.approx(1.)
    late_touch=episode_feature(h=0,above=True,support=True)
    t=episode_advance(t,floor,late_touch,30)
    before=t
    for _ in range(3):
        t=episode_advance(t,late_touch,floor,30)
        t=episode_advance(t,floor,late_touch,30)
    assert bool(t['premature']) and not bool(t['qualified_release'])
    assert float(t['opening_best'])==0
    assert float(t['progress_credit'])==0
    terms=placement_terms(before,t,jp.array(True),C3,jp.array(True))
    assert float(terms['quality'])==0
    # Basic task completion remains; no extra credit is minted by touching again.
    assert float(terms['placement_reward'])==1.5
    assert float(terms['opening'])==float(terms['progress'])==0


def test_unwitnessed_fall_then_floor_touch_cannot_create_qualified_release():
    high=episode_feature(above=False,held=False,down=1.2)
    t=placement_memory(high,True)
    floor=episode_feature(h=0,held=False,support=True,above=False,opening=.1)
    t=episode_advance(t,high,floor,30)
    assert bool(t['floor_seen']) and bool(t['premature'])
    assert float(t['impact_peak'])==pytest.approx(1.2)
    touch=episode_feature(h=0,support=True)
    t=episode_advance(t,floor,touch,30)
    t=episode_advance(t,touch,floor,30)
    assert not bool(t['carried']) and not bool(t['qualified_release'])
    assert float(t['opening_best'])==0


def test_normal_supported_release_is_valid_without_old_phase_predicate():
    high=episode_feature(above=False)
    initial=placement_memory(high,True)
    floor=episode_feature(h=0,support=True,above=False)
    t=episode_advance(initial,high,floor,21)
    opened=episode_feature(h=0,support=True,above=False,held=False,opening=.08)
    t=episode_advance(t,floor,opened,5)
    assert bool(t['qualified_release']) and not bool(t['premature'])
    assert not bool(t['active'])
    assert float(placement_terms(initial,t,jp.array(True),C3,jp.array(True))['placement_reward'])==5


def test_table_brush_without_lifting_does_not_disqualify_later_pick():
    table=episode_feature(h=-.003,above=False)
    t=placement_memory(table,True)
    empty=episode_feature(h=-.003,above=False,held=False)
    t=episode_advance(t,table,empty,30)
    assert not bool(t['carried']) and not bool(t['premature'])
    high=episode_feature(above=False)
    t=episode_advance(t,empty,high,5)
    assert bool(t['carried'])
    floor=episode_feature(h=0,support=True)
    t=episode_advance(t,high,floor,21)
    t=episode_advance(t,floor,episode_feature(h=0,support=True,held=False,opening=.08),5)
    assert bool(t['qualified_release']) and not bool(t['premature'])


def test_subthreshold_contact_gap_does_not_count_as_release():
    high=episode_feature(above=False)
    t=placement_memory(high,True)
    gap=episode_feature(above=False,held=False)
    t=episode_advance(t,high,gap,3)
    assert not bool(t['released'])
    t=episode_advance(t,gap,high)
    assert float(t['lost_contact_time'])==0
    t=episode_advance(t,high,gap,4)
    assert bool(t['released']) and bool(t['premature'])


def test_episode_tracking_jit_matches_eager():
    high=episode_feature(above=False)
    gap=episode_feature(above=False,held=False)
    initial=placement_memory(high,True)
    update=jax.jit(lambda t:placement_update(t,high,gap,.005,C3))
    result=initial
    for _ in range(4):result=update(result)
    assert bool(result['premature'])
