"""Production reward/event replay + isolated squared-penalty proposal; no learning."""
from pathlib import Path
from types import SimpleNamespace
import json
import math

import jax
import jax.numpy as jp
import numpy as np
import yaml

from piper_sim_lab.gentle_placement import DEFAULTS as G
from piper_sim_lab.controlled_release import DEFAULTS, release_memory, release_update, release_step_reward
from piper_sim_lab.quadratic_penalty_audit import Ledger, errors, components, apply_cost, discounted

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/show103/quadratic-penalty'
C=SimpleNamespace(**dict({**G,**DEFAULTS,'reach_scale':5.,'action_cost':.001,'action_rate_cost':.01},
    **yaml.safe_load((ROOT/'configs/show103/balanced-release.yaml').read_text())['environment']))
DT=.005
BOOLS={'above','left','right','elevated','support','in_region','robot_cube','robot_bowl','opening_intent'}


def ramp(t,start,length):
    return float(np.clip((t-start)/length,0.,1.))


def scene(kind,t):
    # Main comparison cases all begin held,100mm horizontally offset, TCP170mm.
    x=.1*(1.-ramp(t,0.,2.5));z=.17-.115*ramp(t,2.5,2.5)
    release=math.inf
    if kind=='far_wait':x,z=.1,.17
    if kind=='align_only':z=.17
    if kind=='complete':release=6.
    if kind=='late_release':release=14.
    if kind in {'bad_high_drop','bad_drop_then_lower','bad_drop_regrasp'}:
        z=.17;release=3.1
    if kind=='bad_outside_drop':x,z=.1,.17;release=.6
    if kind in {'xy_roundtrip','height_roundtrip'} and t>=5.:
        phase=(t-5.)%5.;tri=phase/2.5 if phase<2.5 else (5.-phase)/2.5
        if kind=='xy_roundtrip':x=.1*tri
        else:z=.055+.115*tri
    if kind in {'complete','late_release'} and t>=release:
        z=.055+.145*ramp(t,release+.4,3.)
    if kind in {'bad_drop_then_lower','bad_drop_regrasp'} and t>=3.5:
        z=.17-(.145 if kind=='bad_drop_regrasp' else .115)*ramp(t,3.5,2.9)
    regrasp=kind=='bad_drop_regrasp' and 6.5<=t<8.
    if regrasp:z=.025+.03*ramp(t,6.5,.6)
    if kind=='bad_drop_regrasp' and t>=8.:z=.055+.145*ramp(t,8.4,3.)
    held=t<release or regrasp
    if held:
        cube_x=x;bottom=z-.025
    else:
        cube_x=.1 if kind=='bad_outside_drop' else 0.
        initial_z=.17 if kind.startswith('bad_') else .055
        if kind=='bad_drop_regrasp' and t>=8.:initial_z=.055;release=8.
        elapsed=t-release
        bottom=max(initial_z-.025-.5*9.81*elapsed**2,
                   -.003 if kind=='bad_outside_drop' else 0.)
    support=not held and bottom<=1e-9 and cube_x<=.017
    hand=np.array([x+.012,0.,z+.003])
    cube=np.array([cube_x,0.,bottom+.025+.003])
    opening=t>=release-.1 and not regrasp
    clear=min(np.linalg.norm(hand-cube)-.05,np.linalg.norm(hand)-.075)
    touch=kind=='good_release_recontact' and 12.<=t<13.
    f=dict(above=held and cube_x<.05,left=held,right=held,
        elevated=held and bottom+.003>.005,support=support,
        in_region=cube_x<=.017 and -.001<=bottom<=.005,
        height=bottom,xy=cube_x,speed=0.,downward_speed=0.,
        robot_cube=held,robot_bowl=touch,clearance=clear,
        tcp_height=z,tcp_rel=hand,opening_intent=opening,
        grip_reference=.05 if opening else 0.,tcp_block=np.linalg.norm(hand-cube))
    return f,hand,cube


def features(kind):
    frames=[];hands=[];cubes=[]
    for i in range(4001):
        f,h,b=scene(kind,i*DT);frames.append(f);hands.append(h);cubes.append(b)
    for i in range(1,len(frames)):
        velocity=(cubes[i]-cubes[i-1])/DT
        frames[i]['speed']=np.linalg.norm(velocity)
        frames[i]['downward_speed']=max(-velocity[2],0.)
    arrays={k:jp.asarray(np.stack([f[k] for f in frames]),dtype=jp.bool_ if k in BOOLS else jp.float32)
            for k in frames[0]}
    action=np.zeros((200,7),np.float32)
    for i in range(200):
        j=i*20
        action[i,:3]=np.clip((hands[j+20]-hands[j])/.1/C.max_lin_vel,-1.,1.)
        action[i,6]=1. if frames[j+1]['opening_intent'] else -1.
    return arrays,jp.asarray(action)


@jax.jit
def production(fs,actions):
    initial=jax.tree.map(lambda v:v[0],fs)
    memory=release_memory(initial,C)
    rest=jax.tree.map(lambda v:v[1:].reshape((200,20)+v.shape[1:]),fs)
    def control(carry,inputs):
        old,before,previous=carry;subframes,action=inputs
        def sub(carry,after):
            m,b=carry;n=release_update(m,b,after,DT,C)
            first=m['carried']&~(after['left']|after['right'])&(m['lost_contact_time']==0)&~m['released']
            return (n,after),dict(carried=n['carried'],first_loss=first,
                release_event=~m['released']&n['released'],release_paid=n['release_paid'])
        (new,f),events=jax.lax.scan(sub,(old,before),subframes)
        near=f['tcp_block']<.02
        grasp=near&f['left']&(f['height']+.003>.02)
        placed=(f['xy']<.017)&(f['height']+.025<.067)&(f['height']+.028>0.)
        p=dict(tcp_block=f['tcp_block'],placed=placed,inside=placed&~near)
        reach=1.-jp.tanh(5.*p['tcp_block'])
        original=reach+grasp+grasp*(1.-jp.tanh(5.*f['xy']))+5.*placed+2.*p['inside']
        ac=.001*jp.sum(action**2);rate=.01*jp.sum((action-previous)**2)
        r,score,quality,retreat,impact,touch,bonus=release_step_reward(
            old,new,f,p,C,original,ac,rate,.1)
        return (new,f,action),dict(reward=r,score=score,events=events,bonus=bonus,
            action_cost=ac,rate_cost=rate,quality=quality,impact=impact,touch=touch)
    (final,_,_),tr=jax.lax.scan(control,(memory,initial,jp.zeros(7)),(rest,actions))
    return final,tr


def evaluate(kind):
    fs,actions=features(kind);final,tr=jax.device_get(production(fs,actions))
    events={k:v.reshape(-1) for k,v in tr['events'].items()}
    xy=np.asarray(fs['xy'])[1:];height=np.asarray(fs['tcp_height'])[1:]
    ledger=Ledger(active=bool(fs['left'][0]&fs['right'][0]))
    costs=[];snapshots=[]
    for i in range(4000):
        ledger.observe(**{k:events[k][i] for k in events},xy=xy[i],tcp_height=height[i])
        costs.append(ledger.cost(xy[i],height[i]));snapshots.append(ledger.locked)
    raw=np.asarray(costs).reshape(200,20)[:,-1]
    floor=np.array([apply_cost(c,s,floor_state_score=True) for c,s in zip(raw,tr['score'])])
    baseline=np.asarray(tr['reward']);raw_reward=baseline-raw;floor_reward=baseline-floor
    report=dict(baseline_return=discounted(baseline),raw_return=discounted(raw_reward),
        state_floor_return=discounted(floor_reward),raw_cost_max=float(raw.max()),
        floor_cost_max=float(floor.max()),floor_saturated_steps=int((raw>floor+1e-6).sum()),
        progress_bonus=float(tr['bonus']['progress'].sum()),dwell_bonus=float(tr['bonus']['dwell'].sum()),
        release_bonus=float(tr['bonus']['release'].sum()),landing_bonus=float(tr['bonus']['landing'].sum()),
        withdrawal_bonus=float(tr['bonus']['withdrawal'].sum()),
        error_locked_after_bad_release=ledger.locked,release_paid=bool(final['release_paid']),
        landing_paid=bool(final['landing_paid']),withdrawal_paid=bool(final['withdrawal_paid']))
    return report,dict(raw=raw_reward,floor=floor_reward,cost=raw,score=np.asarray(tr['score']))


def main():
    names=['far_wait','align_only','align_lower_wait','complete','late_release',
        'xy_roundtrip','height_roundtrip','bad_high_drop','bad_drop_then_lower',
        'bad_drop_regrasp','bad_outside_drop']
    results={};traces={}
    for name in names:results[name],traces[name]=evaluate(name)
    checks={}
    for mode,key in [('raw','raw_return'),('floor','state_floor_return')]:
        checks[mode]=dict(
            alignment_beats_far_wait=results['align_only'][key]>results['far_wait'][key],
            alignment_and_lowering_beats_alignment_only=results['align_lower_wait'][key]>results['align_only'][key],
            completion_beats_target_wait=results['complete'][key]>results['align_lower_wait'][key],
            fast_completion_beats_late=results['complete'][key]>results['late_release'][key],
            late_release_beats_target_wait=results['late_release'][key]>results['align_lower_wait'][key],
            xy_roundtrip_does_not_improve_return=results['xy_roundtrip'][key]<=results['align_lower_wait'][key],
            height_roundtrip_does_not_improve_return=results['height_roundtrip'][key]<=results['align_lower_wait'][key],
            high_drop_worse_than_held_high=results['bad_high_drop'][key]<results['align_only'][key],
            empty_lowering_cannot_clear_cost=np.allclose(traces['bad_high_drop']['cost'][60:],traces['bad_drop_then_lower']['cost'][60:]),
            regrasp_cannot_clear_cost=np.allclose(traces['bad_high_drop']['cost'][90:],traces['bad_drop_regrasp']['cost'][90:]))
    # Actual learner terminates on controller failure. Prefix-vs-full identifies
    # accounting vulnerability; these synthetic paths do NOT prove a policy can
    # intentionally trigger that failure from the same physical state.
    exits={}
    for name in ['far_wait','bad_outside_drop']:
        exits[name]={}
        for mode in ['raw','floor']:
            full=discounted(traces[name][mode]);early=discounted(traces[name][mode][:10])
            exits[name][mode]=dict(full20s=full,terminated_at1s=early,termination_advantage=early-full)
    grid=[]
    for mm in [0,17,30,50,75,100,150,200,300]:
        ex,ez=errors(mm/1000.,.055);c=components(ex,ez)
        state=1.-math.tanh(.06)+2.-math.tanh(5.*mm/1000.)
        grid.append(dict(center_distance_mm=mm,error_mm=ex*1000,**c,
            total=sum(c.values()),held_score=state,raw_net=state-sum(c.values()),
            floor_net=max(state-sum(c.values()),0.)))
    assert all(all(v.values()) for v in checks.values())
    assert exits['far_wait']['raw']['termination_advantage']>0
    assert results['far_wait']['floor_saturated_steps']==200
    assert results['bad_drop_regrasp']['release_bonus']==0
    out=dict(scope='Audit-only proposal composed with production5ms events and100ms rewards. Synthetic features, not a physics rollout or learning result.',
        formula='0.6 +0.8*(max(dxy-0.017,0)/0.050)^2 +0.8*ez/(0.040+ez);ez=distance(TCP height,[0.050,0.060]).',
        rule='Activate after confirmed carry. End only on first qualified release. Wrong first release locks first-loss XY/height errors; opening/regrasp cannot clear it.',
        modes=dict(raw='Uncapped squared penalty; state reward may become negative.',
            floor='Earlier proposal: new cost cannot exceed positive state score. Existing action/rate/impact/contact penalties remain.'),
        horizon_s=20,discount=.99,initial_conditions='All principal cases start held at XY100mm offset andTCP170mm above inner floor.',
        scenario_definitions=dict(align='XY100->0mm in2.5s',lower='TCP170->55mm during2.5-5s',
            complete='Release6s, ballistic landing then withdrawal6.4-9.4s',late_release='Same alignment/lowering, wait at goal until14s, finish withdrawal17.4s',
            bad_high_drop='Align then release atTCP170mm after3.1s',bad_outside_drop='Release atXY100mm,TCP170mm after0.6s; no bowl-floor support'),
        grid=grid,results=results,trajectory_checks=checks,early_termination_checks=exits,
        outcome='Standard alignment/completion ordering passes, but raw negative rewards permit early-exit accounting advantage; prior state floor truncates the quadratic penalty. Not ready for training as an unqualified fix.',
        limitations=['Synthetic trajectories do not establish collision-free feasibility.','Early-exit comparison tests reward accounting, not physical reachability of intentional controller faults.','Bad regrasp is an event stress case, not a validated motion.','No universal no-exploit or convergence proof.'],
        training_run=False,real_robot=False,registered_training_environment_changed=False)
    DOC.mkdir(parents=True,exist_ok=True)
    (DOC/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
    np.savez_compressed(DOC/'traces.npz',**{n+'_'+k:v for n,tr in traces.items() for k,v in tr.items()})
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
