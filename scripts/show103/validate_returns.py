"""Synthetic ordering audit of actual reward functions, including action costs.

This is incentive validation, not a robot/plant trajectory feasibility claim.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import jax
import jax.numpy as jp
import numpy as np
from piper_sim_lab.gentle_placement import DEFAULTS,gentle_memory,gentle_update,gentle_reward

C=SimpleNamespace(**DEFAULTS,reach_scale=5.,grasp_distance=.02)

def feature(h,held=True,support=False,clear=0.,touch=False,down=0.):
    return {k:jp.asarray(v) for k,v in dict(above=True,distance=max(h,0.),height=h,xy=0.,
        speed=0.,downward_speed=down,width=.05 if held else .1,span=.05,left=held,right=held,
        support=support,in_region=(-.001<=h<=.005),elevated=held and h+.003>.005,
        robot_cube=held or touch,robot_bowl=False,clearance=clear).items()}

high=feature(.128)
@jax.jit
def frame(memory,before,after,action,previous):
    old=memory
    def sub(t,i):
        new=gentle_update(t,jax.tree.map(lambda b,a:jp.where(i==0,b,a),before,after),after,.005,C)
        return new,None
    memory,_=jax.lax.scan(sub,memory,jp.arange(20))
    placed=after['height']<.045
    predicates=dict(tcp_block=jp.where(after['left'],.012,.10),placed=placed,
        inside=placed&~after['left'])
    reward=gentle_reward(memory,after,predicates,C)[0]
    penalty=.001*jp.sum(action**2)+.01*jp.sum((action-previous)**2)
    impact=2.*(jp.minimum(memory['impact_peak']/.1,1.)-jp.minimum(old['impact_peak']/.1,1.))
    recontact=2.*(memory['recontact_seconds']-old['recontact_seconds'])/.1
    return memory,reward-penalty-impact-recontact

def rollout(kind):
    memory=gentle_memory(high);before=high;previous=jp.zeros(7);rewards=[]
    for i in range(200):
        if kind=='wait_high':
            after=high;action=jp.array([0.,0.,0.,0.,0.,0.,-1.])
        elif kind.startswith('drop'):
            if i<5:after=high
            elif i==5:after=feature(.08,held=False,down=1.)
            else:after=feature(0.,held=False,support=True,clear=.08,touch=kind=='drop_recontact' and i%10<5)
            action=jp.array([0.,0.,0.,0.,0.,0.,-1. if i<5 else 1.])
        else:
            h=max(.128-.005*(i+1),0.)
            supported=h==0.
            released=kind!='hold_supported' and i>=28
            clearance=min(max((i-33)*.008,0.),.08) if released else 0.
            if kind=='out_and_back' and i>=45:clearance=.08 if i%20<10 else 0.
            after=feature(h,held=not released,support=supported,clear=clearance,down=.05 if h>0 else 0.)
            action=jp.array([0.,0.,-.5 if h>0 else (.5 if released and clearance<.08 else 0.),0.,0.,0.,1. if released else -1.])
        memory,r=frame(memory,before,after,action,previous)
        rewards.append(float(r));before=after;previous=action
    return dict(discounted_return=float(np.dot(.99**np.arange(200),rewards)),
        undiscounted_return=sum(rewards),min_reward=min(rewards),max_reward=max(rewards),
        qualified_release=bool(memory['qualified_release']),premature=bool(memory['premature']),
        recontacts=float(memory['recontacts']))

results={k:rollout(k) for k in ['wait_high','hold_supported','gentle_retreat','drop','drop_recontact','out_and_back']}
good=results['gentle_retreat']['discounted_return']
assert all(good>results[k]['discounted_return'] for k in results if k!='gentle_retreat')
assert not results['drop_recontact']['qualified_release']
assert max(x['max_reward'] for x in results.values())<=8.
path=Path('docs/show103/returns.json');path.write_text(json.dumps(dict(
    scope='Synthetic20s trajectories through actual reward/event functions; gamma.99, action cost.001 and action-rate cost.01. Not a feasibility or learning result.',
    results=results,status='passed',real_robot=False),indent=2)+'\n')
print(json.dumps(results),flush=True)
