from types import SimpleNamespace
import jax.numpy as jp
from piper_sim_lab.direct_gripper import direct_goal
from piper_sim_lab.smooth_grip import SmoothGripPickPlace


def test_request_reversal_needs_no_phase_or_arm_state_and_hold_latches_final_goal():
    env=SimpleNamespace(independent_gripper=True,
        _config=SimpleNamespace(gripper_max=.05,gripper_threshold=.5))
    # Starting closed: an opening pulse is accepted immediately, even with no
    # coordinator fields. Hold retains its FINAL target while physically opening.
    ref,info=SmoothGripPickPlace._gripper_command(env,None,jp.array([0.]*6+[1.]),{'gripper_goal':jp.array(0.)})
    assert ref==info['gripper_goal'] and ref>0
    for command in [0.,.49,-.49]:
        ref,new=SmoothGripPickPlace._gripper_command(env,None,jp.array([0.]*6+[command]),info)
        assert ref==info['gripper_goal']==new['gripper_goal']
    ref,new=SmoothGripPickPlace._gripper_command(env,None,jp.array([0.]*6+[-1.]),info)
    assert ref==0 and new['gripper_goal']==0


def test_direct_mode_does_not_add_a_phase_observation():
    env=SimpleNamespace(independent_gripper=True)
    state=object()
    assert SmoothGripPickPlace._augment(env,state) is state


def test_threshold_boundaries_remain_three_state_commands():
    assert direct_goal(-.5,.05,.05,.5)==0
    assert direct_goal(.5,0.,.05,.5)>.049
    assert direct_goal(0.,.05,.05,.5)>.049
