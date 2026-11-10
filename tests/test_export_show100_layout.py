"""Portable metadata must describe the existing 49D latched-goal observation."""
from pathlib import Path
import runpy

EXPORT=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/deployment/export_policy.py'))


def test_show100_and_stage_reward_include_final_goal_before_previous_action():
    layout=EXPORT['observation_layout'](dict(stage_obs=False,size_obs=True,gripper_goal_obs=True))
    assert layout[-2]==dict(name='gripper_final_goal_normalized',start=41,stop=42)
    assert layout[-1]==dict(name='previous_normalized_policy_action',start=42,stop=49)


def test_legacy_show76_remains_48_dimensions():
    layout=EXPORT['observation_layout'](dict(stage_obs=False,size_obs=True))
    assert layout[-1]==dict(name='previous_normalized_policy_action',start=41,stop=48)
    assert not any(x['name']=='gripper_final_goal_normalized' for x in layout)


def test_stage_observation_keeps_existing_order():
    layout=EXPORT['observation_layout'](dict(stage_obs=True,size_obs=True,gripper_goal_obs=True))
    assert layout[-4]['name']=='stage_one_hot'
    assert layout[-1]['stop']==53


def test_tcp_metadata_uses_resolved_override_without_mutating_model_default():
    original=[0.,0.,.1358]
    assert EXPORT['tcp_local_position'](original,{'tcp_local_z':.1385})==[0.,0.,.1385]
    assert EXPORT['tcp_local_position'](original,{})==original
    assert original==[0.,0.,.1358]
