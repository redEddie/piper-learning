"""Counterexamples from the failed policy and unchanged low-release rules."""
import jax
import jax.numpy as jp
import numpy as np
import pytest

from piper_sim_lab import stage_reward as v1
from piper_sim_lab import stage_reward_v2 as v2

ZERO = jp.zeros(7)
HELD = v2.Features(tcp_cube_distance=.005, cube_bottom_table=.05, cube_bowl_xy=.15,
    cube_center_above_rim=.03, tcp_floor_height=.12, left_contact=True,
    right_contact=True, cube_table_contact=False, robot_cube_contact=True,
    cube_bottom_above_rim=.005)
TARGET = HELD._replace(cube_bowl_xy=.005, tcp_floor_height=.055,
                       cube_center_above_rim=-.012, cube_bottom_above_rim=-.037)


def held_memory():
    return v2.reset_memory()._replace(grasp_time=jp.array(.1), lifted=jp.array(True), bowl_reached=jp.array(True))


def terms(f, memory=None):
    m = held_memory() if memory is None else memory
    return v2.reward_terms(m, m, f, ZERO, ZERO)


def test_leaving_bowl_cannot_keep_height_only_three_point_five():
    good = terms(TARGET)
    bad = terms(TARGET._replace(cube_bowl_xy=.4))
    assert good['main'] > 3.4
    assert bad['main'] < 2.1
    assert bad['xy_alignment'] == 0


def test_prepare_reward_decreases_continuously_with_current_xy():
    distances = np.linspace(.017, .4, 101)
    values = np.array([terms(TARGET._replace(cube_bowl_xy=x))['main'] for x in distances])
    assert np.all(np.diff(values) <= 1e-6)
    assert values[0] > values[-1]+1.


def test_carry_above_rim_preferred_but_low_inside_opening_remains_rewarded():
    m = held_memory()._replace(bowl_reached=jp.array(False))
    high = terms(HELD, m)['main']
    low = terms(HELD._replace(cube_bottom_above_rim=-.04), m)['main']
    assert high > low+.1
    assert terms(TARGET)['main'] > terms(HELD)['main']


@pytest.mark.parametrize('height,xy,valid', [(.055,.005,True), (.15,.005,False), (.055,.1,False)])
def test_drop_qualification_and_costs_stay_v1(height, xy, valid):
    before = TARGET._replace(tcp_floor_height=height, cube_bowl_xy=xy)
    after = before._replace(left_contact=False, right_contact=False, robot_cube_contact=False,
                          opening_intent=True, clearance=.01)
    m2 = held_memory()
    m1 = v1.Memory(*m2[:len(v1.Memory._fields)])
    step = jax.jit(v2.advance_memory)
    for i in range(4):
        b = before if i == 0 else after
        m1 = v1.advance_memory(m1, v1.Features(*b[:20]), v1.Features(*after[:20]))
        m2 = step(m2, b, after)
    assert bool(m2.qualified_release) == valid
    for a, b in zip(m1, m2):
        np.testing.assert_allclose(a, b)
    r = terms(after, m2)
    assert float(r['main']) >= 4. if valid else float(r['main']) == 0.
    assert float(r['recontact_cost']) == 0.


def test_no_new_collision_cost_or_open_command_penalty():
    a = terms(HELD)
    b = terms(HELD._replace(robot_bowl_contact=True))
    assert float(a['reward']) == float(b['reward'])
    assert float(b['recontact_cost']) == 0.


def test_nonfinite_extra_geometry_cannot_earn_reward():
    r = terms(TARGET._replace(cube_bottom_above_rim=float('nan')))
    assert r['nonfinite'] and r['reward'] == -1.


@pytest.mark.parametrize('distance,cost', [(.10,0.), (.15,.5), (.25,1.), (.35,1.5)])
def test_away_cost_uses_best_distance_with_ten_cm_threshold(distance, cost):
    m = held_memory()._replace(best_carry_xy=jp.array(.05))
    r = terms(HELD._replace(cube_bowl_xy=distance), m)
    assert float(r['away_cost']) == pytest.approx(cost, abs=2e-6)


def test_best_distance_updates_only_after_measured_grasp_and_survives_retreat():
    m = v2.reset_memory()
    step = jax.jit(v2.advance_memory)
    for _ in range(20):
        m = step(m, HELD, HELD)
    assert float(m.best_carry_xy) == pytest.approx(.15)
    near = HELD._replace(cube_bowl_xy=.05)
    m = step(m, HELD, near)
    m = step(m, near, HELD)
    assert float(m.best_carry_xy) == pytest.approx(.05)
    free = HELD._replace(left_contact=False, right_contact=False)
    assert float(terms(free, m)['away_cost']) == 0.


def test_waiting_at_target_cannot_farm_xy_or_prepare_bonus():
    m = held_memory()._replace(prepare_time=jp.array(1.))
    r = terms(TARGET, m)
    assert float(r['main']) < 3.01
    assert float(r['xy_alignment']) == 0.


def test_invalid_drop_cost_is_one_shot():
    before = held_memory()
    after = before._replace(invalid_release=jp.array(True), released=jp.array(True))
    r = v2.reward_terms(before, after, TARGET, ZERO, ZERO)
    assert r['drop_cost'] == .5
    assert v2.reward_terms(after, after, TARGET, ZERO, ZERO)['drop_cost'] == 0.


@pytest.mark.parametrize('failure', ['invalid_release', 'outside_drop'])
def test_zero_drop_cost_removes_only_direct_penalty_and_preserves_failure(failure):
    old = held_memory()
    new = old._replace(**{failure: jp.array(True), 'released': jp.array(True)})
    paid = v2.reward_terms(old, new, TARGET, ZERO, ZERO)
    free = v2.reward_terms(old, new, TARGET, ZERO, ZERO, v2.RewardSettings(drop_weight=0.))
    assert float(free['drop_cost']) == 0.
    assert float(free['reward']-paid['reward']) == pytest.approx(.5)
    assert float(free['main']) == 0. and not bool(free['complete'])
    for key in ('away_cost', 'action_cost', 'action_rate_cost', 'section', 'nonfinite'):
        np.testing.assert_equal(free[key], paid[key])


def test_zero_drop_cost_preserves_normal_release_reward_and_qualification():
    old = held_memory()
    free = TARGET._replace(left_contact=False, right_contact=False,
        robot_cube_contact=False, opening_intent=True)
    a, b = old, old
    for i in range(4):
        before = TARGET if i == 0 else free
        a = v2.advance_memory(a, before, free)
        b = v2.advance_memory(b, before, free, v2.RewardSettings(drop_weight=0.))
    for x,y in zip(a,b):
        np.testing.assert_equal(x,y)
    assert bool(b.qualified_release)
    paid = v2.reward_terms(old,a,free,ZERO,ZERO)
    unpaid = v2.reward_terms(old,b,free,ZERO,ZERO,v2.RewardSettings(drop_weight=0.))
    for key in paid:
        np.testing.assert_equal(paid[key],unpaid[key])
