"""No training/physics: replay features through production 5ms event/reward code."""
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import yaml

from piper_sim_lab.gentle_placement import DEFAULTS as G
from piper_sim_lab.controlled_release import (
    DEFAULTS, release_memory, release_update, release_step_reward)

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT/'configs/show103/balanced-release.yaml').read_text())
C = SimpleNamespace(**dict({**G, **DEFAULTS, 'reach_scale': 5.,
                            'action_cost': .001, 'action_rate_cost': .01},
                           **CFG['environment']))
DT = .005


def feature(tcp=.153, held=True, support=False, cube_height=None,
            touch=False, speed=0., down=0., xy=0., clear=0., opening=False):
    h = tcp-.025 if cube_height is None else cube_height
    return {k: np.asarray(v, dtype=np.float32 if k not in BOOLS else bool)
            for k, v in dict(above=True, left=held, right=held,
                elevated=held, support=support, height=h, xy=xy, speed=speed,
                downward_speed=down, in_region=(xy<=.017 and -.001<=h<=.005),
                robot_cube=held, robot_bowl=touch, clearance=clear,
                tcp_height=tcp, tcp_rel=[0., 0., tcp+.003],
                opening_intent=opening, grip_reference=.05 if opening else 0.).items()}


BOOLS = {'above', 'left', 'right', 'elevated', 'support', 'in_region',
         'robot_cube', 'robot_bowl', 'opening_intent'}


def trajectory(kind):
    frames=[];actions=[]
    delay=10. if kind=='late_complete' else 0.
    for i in range(4000):
        time=(i+1)*DT; t=time-delay
        tcp=.153-.098*np.clip(t/2.,0.,1.)
        if kind=='wait_high': tcp=.153
        if kind=='oscillate' and t>2.:
            tcp=.055 if int((t-2.)/.5)%2==0 else .095
        release=kind not in {'wait_high','wait_target','oscillate'} and t>=3.
        high=kind in {'high_drop','high_drop_regrasp'}
        if high: tcp=.12;release=t>=.5
        landing=(.7 if high else 3.2)
        support=release and t>=landing
        if kind=='bounce' and 3.2<=t<3.6:
            support=int((t-3.2)/.05)%2==0
        regrasp=kind in {'regrasp','high_drop_regrasp'} and 6.<=t<7.
        held=not release or regrasp
        if release and not high:
            tcp=.055+.145*np.clip((t-3.4)/1.,0.,1.)
        if regrasp: tcp=.055;support=False
        touch=kind in {'recontact','regrasp','high_drop_regrasp'} and 6.<=t<7.
        if kind=='return_clearance' and t>=6.:
            clear=.08 if int((t-6.)/.5)%2==0 else 0.
        else: clear=.08*np.clip((t-3.4)/1.,0.,1.) if release else 0.
        h=0. if support else (max(.03*(1.-(t-3.)/.2),0.) if release and not high else tcp-.025)
        # Impact label is the precontact relative downward speed, as in production.
        down=.7 if release and not support else 0.
        f=feature(tcp=tcp,held=held,support=support,cube_height=h,
                  touch=touch,speed=down,down=down,clear=clear,opening=release and not regrasp)
        frames.append(f)
        action=np.zeros(7,np.float32);action[6]=1. if release and not regrasp else -1.
        if 0.<t<2. and not high and kind!='wait_high':action[2]=-.5
        if release and 3.4<t<4.4 and not high:action[2]=.5
        actions.append(action)
    return jax.tree.map(lambda *x: jp.asarray(np.stack(x)), *frames), jp.asarray(actions)[::20]


@jax.jit
def evaluate(features, actions):
    initial=jax.tree.map(lambda x:x[0],features)
    memory=release_memory(initial,C)
    def control(carry, inputs):
        old,before,previous_action=carry
        subframes,action=inputs
        def sub(carry,after):
            t,b=carry
            return (release_update(t,b,after,DT,C),after),None
        (new,f),_=jax.lax.scan(sub,(old,before),subframes)
        placed=(f['xy']<.017)&(f['height']<.045)
        p=dict(tcp_block=jp.where(f['left'],.012,.1),placed=placed,
               inside=placed&~f['left'])
        # Normal original reward is only used by the production fallback.
        reach=1.-jp.tanh(5.*p['tcp_block'])
        lift=f['left'].astype(jp.float32)
        original=reach+lift+lift*(1.-jp.tanh(5.*f['xy']))+5.*placed+2.*p['inside']
        action_cost=C.action_cost*jp.sum(action**2)
        rate=C.action_rate_cost*jp.sum((action-previous_action)**2)
        r,score,quality,retreat,impact,touch,credits=release_step_reward(
            old,new,f,p,C,original,action_cost,rate,.1)
        return (new,f,action),dict(reward=r,score=score,quality=quality,
            impact=impact,touch=touch,credits=credits)
    frames=jax.tree.map(lambda x:x.reshape((200,20)+x.shape[1:]),features)
    (final,_,_),trace=jax.lax.scan(control,(memory,initial,jp.zeros(7)),(frames,actions))
    return final,trace


def run():
    results={}
    for kind in ['wait_high','wait_target','oscillate','high_drop','complete',
                 'late_complete','recontact','return_clearance','regrasp',
                 'high_drop_regrasp','bounce']:
        final,tr=jax.device_get(evaluate(*trajectory(kind)))
        r=tr['reward']
        results[kind]=dict(discounted_return=float(np.dot(.99**np.arange(200),r)),
            total_return=float(r.sum()),max_step_reward=float(r.max()),
            bonuses={k:float(v.sum()) for k,v in tr['credits'].items()},
            target_count=int(final['target_count']),
            qualified=bool(final['qualified_release']),recontact=bool(final['recontact_seen']),
            events={k:bool(final[k+'_paid']) for k in ['release','landing','withdrawal']})
    good=results['complete']['discounted_return']
    assert all(good>v['discounted_return'] for k,v in results.items() if k!='complete')
    assert results['late_complete']['discounted_return']>results['wait_target']['discounted_return']
    for row in results.values():
        b=row['bonuses']
        assert -.00001<=b['progress']<=3.00001
        assert 0.<=b['dwell']<=1.00001
        assert b['release']<=2. and b['landing']<=2. and b['withdrawal']<=3.
    assert results['wait_target']['target_count']==2
    assert results['oscillate']['target_count']==2
    assert results['regrasp']['bonuses']['release']==2.
    assert not any(results['high_drop_regrasp']['events'].values())
    assert results['return_clearance']['bonuses']['withdrawal']==3.
    assert results['bounce']['bonuses']['landing']==2.
    report=dict(scope='Synthetic features through production 5ms event updates and 100ms reward composition; not physics or learned success.',
        timestep_s=.1,horizon_s=20.,discount=.99,training_run=False,real_robot=False,
        costs='Original action .001, action-rate .01, impact and recontact weights2 retained.',
        assumptions='Held TCP descends153->55mm in2s; releases at3s; floor contact3.2s; clearance80mm at4.4s then held200ms. Delayed version waits10s first.',
        results=results,status='passed')
    dest=ROOT/'docs/show103/balanced-rewards';dest.mkdir(exist_ok=True)
    (dest/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':run()
