"""Physical-event bookkeeping and incentives for the proposed formula only."""
import jax
import jax.numpy as jp
import numpy as np
import pytest

from piper_sim_lab.stage_reward import (
    Features, RewardSettings, advance_memory, main_score, reset_memory,
    reward_terms, start_interval,
)

C = RewardSettings()
ZERO = np.zeros(7, np.float32)
STEP = jax.jit(advance_memory)
HELD = Features(tcp_cube_distance=.005, cube_bottom_table=.08,
    cube_center_above_rim=.038, tcp_floor_height=.105,
    joint_error_a=1., left_contact=True, right_contact=True,
    cube_table_contact=False, robot_cube_contact=True)
ABOVE = HELD._replace(cube_bowl_xy=.005, tcp_floor_height=.12)
TARGET = ABOVE._replace(cube_bottom_table=.03, cube_center_above_rim=-.012,
                       tcp_floor_height=.055)
FLIGHT = TARGET._replace(left_contact=False, right_contact=False,
    robot_cube_contact=False, opening_intent=True, clearance=.01)
LANDED = FLIGHT._replace(bowl_floor_contact=True, cube_bottom_table=.003,
    clearance=.04, tcp_cube_distance=.08)
HOME = LANDED._replace(clearance=.2, joint_error_a=0., joint_speed=0.,
                       tcp_cube_distance=.4, tcp_floor_height=.36)


def ticks(memory, f, n, before=None):
    before = f if before is None else before
    for _ in range(n):
        memory = STEP(memory, before, f)
        before = f
    return memory


def ready():
    return ticks(ticks(reset_memory(), HELD, 20), ABOVE, 1)


def release(memory=None, before=TARGET, after=FLIGHT):
    return ticks(ready() if memory is None else memory, after, 4, before)


def terms(memory, f, old=None, action=ZERO, previous=ZERO):
    return reward_terms(memory if old is None else old, memory, f, action, previous)


def test_grasp_requires_real_bilateral_time_and_lift():
    memory = ticks(reset_memory(), HELD, 19)
    assert not memory.lifted
    memory = ticks(memory, HELD, 1)
    assert memory.lifted
    assert int(main_score(memory, HELD)[1]) == 2
    for f in [HELD._replace(left_contact=False), HELD._replace(cube_bottom_table=.019)]:
        assert not ticks(reset_memory(), f, 40).lifted


def test_main_sections_have_cumulative_baselines_without_legacy_five():
    near = Features(tcp_cube_distance=.01)
    assert 0 <= float(main_score(reset_memory(), Features())[0]) <= 1
    assert float(main_score(reset_memory(), near)[0]) == 1
    carried = ticks(reset_memory(), HELD, 20)
    assert 2 <= float(main_score(carried, HELD)[0]) <= 3
    m = ready()
    assert float(main_score(m, TARGET)[0]) == pytest.approx(3.5)
    m = release(m)
    assert 4 <= float(main_score(m, FLIGHT)[0]) < 5


@pytest.mark.parametrize('height,xy,opening,valid', [
    (.05,.017,True,True), (.06,.017,True,True), (.049,.005,True,False),
    (.061,.005,True,False), (.055,.0171,True,False), (.055,.005,False,False)])
def test_first_detachment_height_alignment_and_open_intent(height,xy,opening,valid):
    before = TARGET._replace(tcp_floor_height=height,cube_bowl_xy=xy)
    after = FLIGHT._replace(opening_intent=opening)
    m = release(before=before,after=after)
    assert bool(m.qualified_release) == valid
    assert bool(m.invalid_release) != valid


def test_release_needs_twenty_ms_and_anchors_first_loss_not_late_height():
    m = ticks(ready(), FLIGHT, 1, TARGET)
    assert not m.released
    m = ticks(m, FLIGHT._replace(tcp_floor_height=.3), 3, FLIGHT)
    assert m.qualified_release
    assert float(m.candidate_height) == pytest.approx(.055)


def test_release_speed_is_measured_not_a_hidden_gate():
    before = TARGET._replace(tcp_relative_speed=.5,cube_relative_speed=2.)
    m = release(before=before)
    assert m.qualified_release
    assert float(m.candidate_tcp_speed) == pytest.approx(.5)
    assert float(m.candidate_cube_speed) == pytest.approx(2.)
    impact = ticks(m, LANDED, 1, FLIGHT._replace(cube_downward_speed=1.2))
    assert float(impact.impact_peak) == pytest.approx(1.2)


def test_high_drop_late_recontact_cannot_become_qualified_release():
    m = release(before=TARGET._replace(tcp_floor_height=.15))
    m = ticks(m, TARGET, 40)
    m = release(m)
    m = ticks(m, HOME, 100)
    assert not m.qualified_release
    assert float(terms(m, HOME)['main']) == 0
    assert not terms(m, HOME)['complete']


def test_outside_drop_removes_historical_stage_credit():
    m = release()
    outside = FLIGHT._replace(cube_table_contact=True,cube_relative_speed=0.)
    m = ticks(m, outside, 20)
    assert m.outside_drop
    assert float(terms(m, outside)['main']) == 0


def test_dwell_two_nonoverlapping_windows_and_no_roundtrip_refill():
    m = ready()
    old = m
    m = ticks(m, TARGET, 19)
    assert int(m.target_count) == 0
    m = ticks(m, TARGET, 1)
    assert int(m.target_count) == 1
    assert float(terms(m,TARGET,old)['dwell_bonus']) == .5
    old = m
    m = ticks(m, TARGET, 20)
    assert int(m.target_count) == 2
    assert float(terms(m,TARGET,old)['dwell_bonus']) == .5
    old = m
    for _ in range(3):
        m = ticks(m, ABOVE, 25)
        m = ticks(m, TARGET, 25)
    assert float(terms(m,TARGET,old)['dwell_bonus']) == 0
    m = release(m)
    assert int(ticks(m,TARGET,100).target_count) == 2


def test_fast_target_pass_does_not_earn_dwell():
    moving = TARGET._replace(tcp_relative_speed=.0101)
    assert int(ticks(ready(),moving,100).target_count) == 0


def test_xy_is_per_policy_step_only_while_carrying():
    m = ready()
    assert float(terms(m,TARGET)['xy_alignment']) == .5
    assert float(terms(m,TARGET._replace(cube_bowl_xy=.0171))['xy_alignment']) == 0
    m = release(m)
    assert float(terms(m,LANDED)['xy_alignment']) == 0
    assert float(terms(m,LANDED)['face_alignment']) == 0
    far = LANDED._replace(tcp_cube_distance=10.)
    assert float(terms(m,LANDED)['main']) == float(terms(m,far)['main'])


def test_any_five_ms_postrelease_contact_charges_full_point_five():
    m = start_interval(release())
    old = m
    touching = FLIGHT._replace(robot_bowl_contact=True)
    m = ticks(m,touching,1)
    m = ticks(m,FLIGHT,19,touching)
    assert float(terms(m,FLIGHT,old)['recontact_cost']) == .5
    m = ticks(start_interval(m),FLIGHT,20)
    assert float(terms(m,FLIGHT)['recontact_cost']) == 0
    # A transient historical touch does not permanently forbid recovery/home.
    m = ticks(m,HOME,60)
    assert terms(m,HOME)['complete']


def test_grasp_contacts_do_not_count_as_postrelease_recontact():
    m = ticks(start_interval(reset_memory()),HELD,40)
    assert float(terms(m,HELD)['recontact_cost']) == 0


@pytest.mark.parametrize('change', [dict(joint_error_a=C.home_error+.001),
    dict(joint_speed=C.home_speed+.001), dict(clearance=.079),
    dict(cube_relative_speed=.021)])
def test_home_needs_landing_clearance_pose_and_velocity(change):
    f = HOME._replace(**change)
    m = ticks(release(),f,100)
    assert not terms(m,f)['complete']
    assert float(terms(m,f)['main']) < 5


def test_home_needs_two_hundred_ms_and_completion_is_not_irrevocable():
    m = ticks(release(),HOME,20)
    assert float(m.home_time) == 0.
    assert not terms(m,HOME)['complete']
    m = ticks(m,HOME,39)
    assert not terms(m,HOME)['complete']
    m = ticks(m,HOME,1)
    assert terms(m,HOME)['complete']
    assert float(terms(m,HOME)['main']) == 5
    disturbed = HOME._replace(cube_relative_speed=.03)
    m = ticks(m,disturbed,1,HOME)
    assert not terms(m,disturbed)['complete']


def test_face_fades_into_grasp_credit_without_downward_reward_jump():
    f = HELD._replace(cube_bottom_table=0.)
    m = reset_memory()
    values = []
    for i in range(21):
        mi = m._replace(grasp_time=jp.float32(i*.005))
        r = terms(mi,f)
        values.append(float(r['main']+r['face_alignment']))
    assert np.diff(values).min() >= -1e-6
    assert max(values) <= 1.5+1e-6


def test_cube_face_symmetry_and_no_transport_height_cost():
    f = Features(tcp_cube_distance=0.)
    m = reset_memory()
    assert float(terms(m,f)['face_alignment']) == .5
    assert float(terms(m,f._replace(yaw_error=np.pi/2))['face_alignment']) == .5
    assert float(terms(m,f._replace(yaw_error=np.pi/4))['face_alignment']) == 0.
    m = ticks(m,HELD,20)
    low = terms(m,HELD._replace(tcp_floor_height=.04))
    high = terms(m,HELD._replace(tcp_floor_height=.8))
    assert float(low['reward']) == float(high['reward'])


def test_cost_units_caps_and_original_action_coefficients():
    f = Features(cube_horizontal_speed=.01)
    assert float(terms(reset_memory(),f)['cube_push_cost']) == pytest.approx(.1)
    assert float(terms(reset_memory(),f._replace(cube_horizontal_speed=1.))['cube_push_cost']) == .5
    r = terms(reset_memory(),f,action=np.ones(7),previous=-np.ones(7))
    assert float(r['action_cost']) == pytest.approx(.007)
    assert float(r['action_rate_cost']) == pytest.approx(.28)


def test_reset_and_nonfinite_handling():
    m = ticks(release(),HOME,60)
    assert terms(m,HOME)['complete']
    fresh = reset_memory()
    assert not fresh.released and int(fresh.target_count) == 0
    assert not terms(fresh,Features())['complete']
    assert float(terms(fresh,Features(tcp_cube_distance=np.nan))['reward']) == -1
    assert float(terms(fresh,Features(),action=np.full(7,np.nan))['reward']) == -1


def test_formula_vmaps_and_jits_over_4096_worlds():
    f = Features(*(jp.full((4096,),v) for v in TARGET))
    m = jax.tree.map(lambda v:jp.full((4096,),v),ready())
    a = jp.zeros((4096,7))
    m = jax.jit(jax.vmap(advance_memory))(m,f,f)
    result = jax.jit(jax.vmap(reward_terms))(m,m,f,a,a)
    assert result['reward'].shape == (4096,)
    np.testing.assert_allclose(result['main'],3.5)
    assert np.isfinite(result['reward']).all()


def test_settings_reject_invalid_timing_and_thresholds():
    with pytest.raises(ValueError):RewardSettings(release_height_max=.01)
    with pytest.raises(ValueError):RewardSettings(physics_dt=.007)
    with pytest.raises(ValueError):RewardSettings(action_weight=np.nan)
