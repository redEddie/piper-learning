"""Price synthetic state histories through the real five-section equations.

No policy training, MuJoCo rollout, robot I/O or physical feasibility claim.
All comparisons keep a 20s horizon; completion does not terminate early.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS','cpu')
import jax
import jax.numpy as jp
import numpy as np

from piper_sim_lab.stage_reward import (
    Features, RewardSettings, advance_memory, reset_memory, reward_terms, start_interval,
)

C = RewardSettings()
HORIZON = 200
HELD = Features(tcp_cube_distance=.005,cube_bottom_table=.08,
    cube_center_above_rim=.038,tcp_floor_height=.105,joint_error_a=1.,
    left_contact=True,right_contact=True,cube_table_contact=False,robot_cube_contact=True)
ABOVE = HELD._replace(cube_bowl_xy=.005,tcp_floor_height=.12)
TARGET = ABOVE._replace(cube_bottom_table=.03,cube_center_above_rim=-.012,tcp_floor_height=.055)
FLIGHT = TARGET._replace(left_contact=False,right_contact=False,
    robot_cube_contact=False,opening_intent=True,clearance=.01)
LANDED = FLIGHT._replace(bowl_floor_contact=True,cube_bottom_table=.003,
    clearance=.04,tcp_cube_distance=.08)
HOME = LANDED._replace(clearance=.2,joint_error_a=0.,joint_speed=0.,
    tcp_cube_distance=.4,tcp_floor_height=.36)


def action(moving=False,gripper=0.):
    return np.array([.3 if moving else 0.]*6+[gripper],np.float32)


def packet(f,n=1,moving=False,gripper=0.):
    return [(f,action(moving,gripper)) for _ in range(n)]


def normal():
    path=[]
    for d in np.linspace(.35,.015,20):
        path+=packet(Features(tcp_cube_distance=float(d)),moving=True)
    for i in range(10):
        f=HELD._replace(cube_bottom_table=float(.03*i/9))
        path+=packet(f,moving=True,gripper=-1. if i==0 else 0.)
    for xy in np.linspace(.25,.005,20):
        path+=packet(ABOVE._replace(cube_bowl_xy=float(xy)),moving=True)
    for height in np.linspace(.12,.055,10):
        path+=packet(TARGET._replace(tcp_floor_height=float(height),tcp_relative_speed=.02),moving=True)
    path+=packet(TARGET,2)
    path+=packet(FLIGHT,2,gripper=1.)
    path+=packet(LANDED,2)
    for i in range(24):
        fraction=i/23
        f=HOME._replace(clearance=.04+.16*fraction,joint_error_a=1.-fraction,
                        joint_speed=.1 if i<23 else 0.)
        path+=packet(f,moving=i<23)
    path+=packet(HOME,HORIZON-len(path))
    assert len(path)==HORIZON
    return path


def paths():
    base=normal()
    cases=dict(normal_complete=base,
        wait_empty_a=packet(Features(),HORIZON),
        wait_before_lift=base[:20]+packet(HELD._replace(cube_bottom_table=0.),HORIZON-20),
        wait_during_transport=base[:30]+packet(HELD,HORIZON-30),
        wait_at_release_target=base[:62]+packet(TARGET,HORIZON-62),
        wait_8s_before_release=(base[:62]+packet(TARGET,80)+base[62:])[:HORIZON])
    high=list(base)
    high[61]=(TARGET._replace(tcp_floor_height=.15),action(True))
    cases['high_drop_then_return_low']=high
    outside=list(base)
    outside[61]=(TARGET._replace(cube_bowl_xy=.1),action(True))
    cases['outside_drop_then_return']=outside
    roundtrip=packet(ABOVE._replace(cube_bowl_xy=.15),20,True)+packet(TARGET,20)
    cases['roundtrip_after_dwell']=(base[:62]+roundtrip+base[62:])[:HORIZON]
    cases['regrasp_after_release']=(base[:64]+packet(TARGET,40)+base[64:])[:HORIZON]
    contact=list(base)
    contact[70]=(contact[70][0]._replace(robot_bowl_contact=True),contact[70][1])
    cases['one_recontact_then_recover']=contact
    push=list(base)
    for i in range(16,20):
        push[i]=(push[i][0]._replace(cube_horizontal_speed=.01),push[i][1])
    cases['normal_with_small_cube_push']=push
    return cases


@jax.jit
def rollout(features,actions):
    def policy_step(carry,inputs):
        memory,before,previous=carry
        f,a=inputs
        def physics_step(state,_):
            m,b=state
            return (advance_memory(m,b,f,C),f),None
        (new,_),_=jax.lax.scan(physics_step,(start_interval(memory),before),None,length=20)
        terms=reward_terms(memory,new,f,a,previous,C)
        sample=dict(terms,target_count=new.target_count,qualified_release=new.qualified_release,
                    invalid_release=new.invalid_release,outside_drop=new.outside_drop)
        return (new,f,a),sample
    initial=Features(*(v[0] for v in features))
    return jax.lax.scan(policy_step,(reset_memory(),initial,jp.zeros(7)),(features,actions))[1]


def audit():
    reports={}
    for name,path in paths().items():
        fs=Features(*(jp.asarray([f[i] for f,_ in path]) for i in range(len(Features()))))
        actions=jp.asarray(np.stack([a for _,a in path]))
        tr=jax.tree.map(np.asarray,jax.block_until_ready(rollout(fs,actions)))
        assert np.isfinite(tr['reward']).all(),name
        completed=np.flatnonzero(tr['complete'])
        reports[name]=dict(
            discounted_return=float(np.sum(tr['reward']*.99**np.arange(HORIZON))),
            undiscounted_return=float(tr['reward'].sum()),
            first_completion_s=float((completed[0]+1)*C.policy_dt) if len(completed) else None,
            final_complete=bool(tr['complete'][-1]),
            main_max=float(tr['main'].max()),reward_max=float(tr['reward'].max()),
            face_max=float(tr['face_alignment'].max()),
            xy_max=float(tr['xy_alignment'].max()),
            dwell_total=float(tr['dwell_bonus'].sum()),
            recontact_total=float(tr['recontact_cost'].sum()),
            action_total=float(tr['action_cost'].sum()),
            action_rate_total=float(tr['action_rate_cost'].sum()),
            qualified_release=bool(tr['qualified_release'][-1]),
            invalid_release=bool(tr['invalid_release'][-1]))
    good=reports['normal_complete']['discounted_return']
    must_beat=['wait_empty_a','wait_before_lift','wait_during_transport',
               'wait_at_release_target','wait_8s_before_release',
               'high_drop_then_return_low','outside_drop_then_return',
               'roundtrip_after_dwell','regrasp_after_release']
    assertions={f'normal_beats_{name}':good>reports[name]['discounted_return'] for name in must_beat}
    assertions.update(
        normal_finishes=reports['normal_complete']['final_complete'],
        all_dwell_episode_caps=all(r['dwell_total']<=1.+1e-6 for r in reports.values()),
        main_score_at_most_five=all(r['main_max']<=5.+1e-6 for r in reports.values()),
        normal_has_no_recontact_cost=reports['normal_complete']['recontact_total']==0.,
        single_contact_charges_point_five=reports['one_recontact_then_recover']['recontact_total']==.5,
        contact_can_recover=reports['one_recontact_then_recover']['final_complete'],
        high_drop_does_not_requalify=not reports['high_drop_then_return_low']['qualified_release'])
    assertions['small_push_completion_beats_no_motion']=(
        reports['normal_with_small_cube_push']['discounted_return'] >
        reports['wait_empty_a']['discounted_return'])
    assertions['small_push_completion_beats_grasp_wait']=(
        reports['normal_with_small_cube_push']['discounted_return'] >
        reports['wait_before_lift']['discounted_return'])
    return dict(status='passed' if all(assertions.values()) else 'failed',
        settings=asdict(C),horizon_seconds=20.,discount=.99,assertions=assertions,
        scenarios=reports,scope='Synthetic measured-state histories, not physically feasible robot rollouts or learned performance.',
        simulator_connected=False,training=False,real_robot=False,plane_modified=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=audit()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['status']!='passed':raise SystemExit(1)
