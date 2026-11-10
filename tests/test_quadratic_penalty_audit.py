import pytest

from piper_sim_lab.quadratic_penalty_audit import Ledger, errors, components, apply_cost, discounted


def observe(ledger, **kwargs):
    values=dict(carried=True,first_loss=False,release_event=False,
                release_paid=False,xy=0.,tcp_height=.055)
    values.update(kwargs)
    ledger.observe(**values)


def test_quadratic_cost_really_quadruples_when_error_doubles():
    a=components(.025,0.)['horizontal']
    b=components(.05,0.)['horizontal']
    c=components(.10,0.)['horizontal']
    assert a==pytest.approx(.2)
    assert b==pytest.approx(4*a)
    assert c==pytest.approx(4*b)


def test_center100mm_is83mm_outside_target_and_not_saturated():
    xy,z=errors(.100,.055)
    assert xy==pytest.approx(.083) and z==0.
    assert sum(components(xy,z).values())==pytest.approx(2.80448)
    assert sum(components(*errors(.117,.055)).values())==pytest.approx(3.8)


def test_vertical_and_unfinished_parameters_are_preserved():
    assert sum(components(*errors(0.,.055)).values())==pytest.approx(.6)
    assert sum(components(*errors(0.,.10)).values())==pytest.approx(1.)
    assert sum(components(*errors(0.,.17)).values())==pytest.approx(1.1866666667)


def test_no_new_cost_before_confirmed_carry():
    ledger=Ledger()
    observe(ledger,carried=False,xy=.2,tcp_height=.3)
    assert ledger.cost(.2,.3)==0.


def test_wrong_release_keeps_first_loss_error_after_cube_enters_bowl():
    ledger=Ledger()
    observe(ledger,first_loss=True,xy=.1,tcp_height=.17)
    old=ledger.cost(.1,.17)
    observe(ledger,release_event=True,xy=.05,tcp_height=.055)
    assert ledger.locked
    assert ledger.cost(0.,.025)==pytest.approx(old)


def test_regrasp_and_empty_lowering_cannot_reset_failed_release_debt():
    ledger=Ledger()
    observe(ledger,first_loss=True,xy=0.,tcp_height=.17)
    observe(ledger,release_event=True,xy=0.,tcp_height=.17)
    for _ in range(5):
        observe(ledger,xy=0.,tcp_height=.055)
        observe(ledger,carried=False,xy=0.,tcp_height=.055)
    assert ledger.cost(0.,.055)==pytest.approx(1.1866666667)


def test_transient_loss_is_not_a_confirmed_release_and_new_loss_reanchors():
    ledger=Ledger()
    observe(ledger,first_loss=True,xy=.1,tcp_height=.17)
    observe(ledger,xy=0.,tcp_height=.10)
    assert not ledger.locked
    observe(ledger,first_loss=True,xy=0.,tcp_height=.10)
    observe(ledger,release_event=True,xy=0.,tcp_height=.09)
    assert ledger.cost(0.,.055)==pytest.approx(1.)


def test_qualified_release_removes_cost_for_withdrawal():
    ledger=Ledger()
    observe(ledger,first_loss=True)
    observe(ledger,release_event=True,release_paid=True)
    observe(ledger,carried=False,release_paid=True,xy=.3,tcp_height=.3)
    assert ledger.cost(.3,.3)==0.


def test_prior_state_floor_truncates_quad_cost_and_removes_far_distance_signal():
    state=2.4779547392
    cost=sum(components(*errors(.1,.055)).values())
    assert state-apply_cost(cost,state,floor_state_score=False)<0.
    assert state-apply_cost(cost,state,floor_state_score=True)==0.
    larger=sum(components(*errors(.2,.055)).values())
    assert state-apply_cost(larger,state,floor_state_score=True)==0.


def test_raw_negative_rewards_have_an_early_terminal_counterexample():
    state=2.4779547392
    cost=sum(components(*errors(.1,.17)).values())
    step=state-cost-.001
    assert step<0.
    assert discounted([step]*10)>discounted([step]*200)
