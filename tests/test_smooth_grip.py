from types import SimpleNamespace
import jax.numpy as jp
from piper_sim_lab.smooth_grip import accepted_goal,stroke_complete,MOVE,BRAKE,GRIP

C=SimpleNamespace(gripper_max=.05,grip_stable_speed=.005,grip_stable_seconds=.1)


def test_busy_requests_do_not_reverse_or_restart_current_stroke():
    assert float(accepted_goal(1.,0.,BRAKE,.05,.5))==0
    assert float(accepted_goal(1.,0.,GRIP,.05,.5))==0
    assert float(accepted_goal(0.,.05,MOVE,.05,.5))>0
    assert float(accepted_goal(-1.,.05,MOVE,.05,.5))==0


def test_cube_contact_width_is_not_required_to_reach_zero():
    assert stroke_complete(jp.full(20,.05),0.,0.,.1,.5,C)


def test_opening_can_settle_below_nominal_full_width_but_must_make_progress():
    assert stroke_complete(jp.full(20,.09),.05,.05,.05,.5,C)
    assert not stroke_complete(jp.full(20,.05),.05,.05,.05,.5,C)


def test_contact_chatter_envelope_is_accepted_but_a_still_moving_stroke_is_not():
    assert stroke_complete(.05+jp.tile(jp.array([-.0001,.0001]),10),0.,0.,.1,.5,C)
    assert not stroke_complete(jp.linspace(.06,.05,20),0.,0.,.1,.5,C)
    assert not stroke_complete(jp.full(20,.05),0.,0.,.1,.05,C)
