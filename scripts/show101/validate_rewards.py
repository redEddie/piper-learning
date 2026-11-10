"""Discounted scripted reward audit, not a learned-policy success evaluation."""
from pathlib import Path
from types import SimpleNamespace
import json
import jax
import jax.numpy as jp
import numpy as np
from piper_sim_lab.placement_refinement import DEFAULTS,placement_memory,placement_update,placement_terms
C=SimpleNamespace(**DEFAULTS,inside_reward=5.)
ROOT=Path(__file__).resolve().parents[2]

def trajectory(kind):
    times=np.arange(1601)*.005
    if kind=='wait':
        h=np.full(times.shape,.1);held=np.ones(len(times),bool);support=np.zeros(len(times),bool);down=np.zeros(len(times));width=np.full(len(times),.05)
    elif kind=='drop':
        elapsed=np.maximum(times-.5,0.);h=np.maximum(.1-.5*9.81*elapsed**2,0.)
        support=h<=0;held=times<.5;down=np.where(support,0.,9.81*elapsed);width=np.where(held,.05,.08)
    else:
        start=3. if kind=='delayed_gentle' else 0.
        elapsed=np.maximum(times-start,0.)
        h=np.where(elapsed<.8,.1-.1*elapsed,np.maximum(.02-.02*(elapsed-.8),0.))
        support=h<=1e-8;down=np.where(elapsed<.8,.1,np.where(support,0.,.02))
        down=np.where(times<start,0.,down)
        if kind=='closed_on_floor':held=np.ones(len(times),bool);width=np.full(len(times),.05)
        else:held=elapsed<2.1;width=np.where(held,.05,.08)
    speed=np.where(support,0.,down)
    d=dict(above=(h>.042)&held,distance=h,height=h,xy=np.zeros(len(times)),speed=speed,
        downward_speed=down,width=width,span=np.full(len(times),.05),left=held,right=held,
        support=support,in_region=(h<=.005))
    return {k:jp.asarray(v,dtype=jp.bool_ if v.dtype==bool else jp.float32) for k,v in d.items()},held


def audit(kind,c=C):
    f,held=trajectory(kind);initial={k:v[0] for k,v in f.items()};memory=placement_memory(initial)
    before={k:v[:-1] for k,v in f.items()};after={k:v[1:] for k,v in f.items()}
    @jax.jit
    def update(memory):
        def step(t,pair):
            a,b=pair;t=placement_update(t,a,b,.005,c);return t,t
        return jax.lax.scan(step,memory,(before,after))
    final,trace=jax.block_until_ready(update(memory));trace=jax.tree.map(np.asarray,trace)
    idx=np.arange(19,1600,20);prev_idx=np.r_[-1,idx[:-1]]
    prev={k:np.r_[np.asarray(memory[k])[None],trace[k][idx[:-1]]] for k in memory}
    cur={k:trace[k][idx] for k in memory}
    hf=np.asarray(f['height'])[1:][idx];heldf=held[1:][idx]
    near=heldf;distance=np.where(near,.01,.06)
    reach=1-np.tanh(5*distance);lift=heldf&(hf+.003>.02)
    carry=lift.astype(float) # xy=0 in these placing-only examples
    placed=hf<.042;inside=placed&~near
    original=reach+lift+carry+5*placed+2*inside
    terms=placement_terms(jax.tree.map(jp.asarray,prev),jax.tree.map(jp.asarray,cur),jp.asarray(placed),c,jp.asarray(inside))
    bridge=np.where(heldf&~cur['released'],(1.-lift.astype(float))*2.,0.) if c.place_hold_bridge else 0.
    reward=original-5*placed+np.asarray(terms['placement_reward'])+np.asarray(terms['progress'])-np.asarray(terms['impact'])+np.asarray(terms['opening'])+bridge
    # Same original action costs, approximating a down/up command and a single opening packet.
    a=np.zeros((80,7))
    if kind not in ['wait','drop']:
        a[:,2]=np.clip(-np.asarray(f['downward_speed'])[1:][idx]/.1,-1.,1.)
    releases=np.flatnonzero(np.diff(heldf.astype(int))<0)+1
    if len(releases):a[releases[0],6]=1
    cost=.001*np.square(a).sum(1)+.01*np.square(np.diff(np.r_[np.zeros((1,7)),a],axis=0)).sum(1)
    reward-=cost
    return dict(discounted_return=float((reward*.99**np.arange(80)).sum()),undiscounted_return=float(reward.sum()),
        progress_total=float(np.asarray(terms['progress']).sum()),impact_total=float(np.asarray(terms['impact']).sum()),
        opening_total=float(np.asarray(terms['opening']).sum()),final_quality=float(np.asarray(terms['quality'])[-1]),
        qualified_release=bool(final['qualified_release']),premature_release=bool(final['premature']),impact_peak=float(final['impact_peak']))

results={k:audit(k) for k in ['wait','drop','gentle','delayed_gentle','closed_on_floor']}
assert results['gentle']['discounted_return']>results['delayed_gentle']['discounted_return']>results['wait']['discounted_return']>results['drop']['discounted_return']
assert results['gentle']['discounted_return']>results['closed_on_floor']['discounted_return']
assert results['drop']['progress_total']==0 and results['drop']['final_quality']==0
assert results['gentle']['opening_total']<=1.00001
v2=SimpleNamespace(**dict(vars(C),place_base_reward=1.5,place_hold_bridge=True,place_contact_activation=True))
results_v2={k:audit(k,v2) for k in results}
assert results_v2['gentle']['discounted_return']>results_v2['delayed_gentle']['discounted_return']>results_v2['drop']['discounted_return']>results_v2['wait']['discounted_return']
assert results_v2['gentle']['discounted_return']>results_v2['closed_on_floor']['discounted_return']
(ROOT/'docs/show101/reward-audit-v2.json').write_text(json.dumps(dict(scope='scripted trajectories; basic completion floor 1.5 and hold bridge only inside placing',results=results_v2),indent=2)+'\n')
(ROOT/'docs/show101/reward-audit.json').write_text(json.dumps(dict(scope='scripted feature trajectories, not physical rollouts; includes original action costs and gamma=.99',results=results),indent=2)+'\n')
print(json.dumps(dict(v1=results,v2=results_v2)),flush=True)
