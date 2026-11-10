"""Validate 1,024 production slots, fresh resets and gripper/controller state."""
import json
import argparse
import functools
import time
from pathlib import Path
import yaml
from piper_sim_lab.runtime import configure
configure('gpu')
import jax
import jax.numpy as jp
import numpy as np
from piper_sim_lab.tasks import make_env
from piper_sim_lab.wrappers import wrap_for_training

root=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--config',default='configs/show101/refine-v2-8m.yaml')
ap.add_argument('--label',default='v2')
args=ap.parse_args()
c=yaml.safe_load((root/args.config).read_text())
e=make_env(dict(c['environment'],batch_capacity=1024))
w=wrap_for_training(e,episode_length=8,randomization_fn=functools.partial(e.physics_randomization_fn, rng=jax.random.split(jax.random.key(107),1024)))
start=time.monotonic()
input_dir=root/f'runs/show101/baseline-input-{args.label}';input_dir.mkdir(parents=True,exist_ok=True)
resolved=dict(c,resolved_environment=dict(e._config.to_dict(),task='show76_place_refine'))
(input_dir/'config.json').write_text(json.dumps(resolved,indent=2)+'\n')
s=jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(100),1024))
assert s.obs['state'].shape==(1024,49)
assert s.obs['privileged'].shape==(1024,66)

@jax.jit
def rollout(s):
    def f(carry,i):
        s,key=carry
        key,k=jax.random.split(key)
        a=jax.random.uniform(k,(1024,7),minval=-1.,maxval=1.)
        old_goal=s.info['gripper_goal']
        new=w.step(s,a)
        boundary=new.info['episode_done'].astype(bool)
        goal=jp.where(a[:,6]<=-.5,0.,jp.where(a[:,6]>=.5,e._config.gripper_max,old_goal))
        expected=jp.where(boundary,jp.where(new.info['reset_stage']>=2,0.,e._config.gripper_max),goal)
        discrepancy=jp.max(jp.abs(expected-new.info['gripper_goal']))
        obs_error=jp.max(jp.abs(new.obs['state'][:,41]-new.info['gripper_goal']/e._config.gripper_max))
        q=new.data.qpos[:,e._qpos_ids]
        ledger=new.info['placement']
        reset_ledger_error=jp.max(jp.where(boundary,ledger['progress_credit']+ledger['impact_peak']+ledger['opening_best']+
            ledger['released'].astype(jp.float32)+ledger['premature'].astype(jp.float32)+ledger['qualified_release'].astype(jp.float32),0.))
        raw_error=jp.max(jp.where(boundary[:,None],0.,jp.abs(new.obs['state'][:,-7:]-a)))
        pose_a_error=jp.max(jp.where((boundary & (new.info['reset_stage']==0))[:,None],
            jp.maximum(jp.abs(q-jp.asarray(e._config.nominal_qpos))-.05001,0.),0.))
        trace=new.info['tracking_trace']
        result=dict(reset_ledger_error=reset_ledger_error,raw_action_error=raw_error,goal_error=discrepancy,observation_error=obs_error,A_reset_error=pose_a_error,
            resets=jp.sum(boundary),reset_counts=jp.bincount(new.info['reset_stage'],weights=boundary.astype(jp.int32),length=4),
            nonfinite=jp.max(new.metrics['nonfinite']),overflow=jp.max(new.metrics['capacity_overflow']),
            controller_failed=jp.sum(new.metrics['controller_failed']),
            max_reference_acceleration=jp.max(trace[...,21]),
            min_reference_floor=jp.min(jp.where(boundary[:,None],jp.inf,trace[...,20])))
        return (new,key),result
    return jax.lax.scan(f,(s,jax.random.key(101)),jp.arange(24))
_,out=jax.block_until_ready(rollout(s));out=jax.tree.map(np.asarray,out)
summary={k:float(v.max()) for k,v in out.items() if k not in ['resets','reset_counts','controller_failed','min_reference_floor']}
summary.update(environments=1024,transitions=24576,reset_events=int(out['resets'].sum()),
 reset_counts=out['reset_counts'].sum(0).astype(int).tolist(),controller_failed_transitions=int(out['controller_failed'].sum()),
 min_reference_floor=float(out['min_reference_floor'].min()),elapsed_seconds=time.monotonic()-start,
 actor_shape=list(s.obs['state'].shape),critic_shape=list(s.obs['privileged'].shape),
 initial_reset_counts=np.bincount(np.asarray(s.info['reset_stage']),minlength=4).tolist())
(root/f'docs/show101/integration-probe-{args.label}.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
for key in ['goal_error','observation_error','A_reset_error','nonfinite','overflow','reset_ledger_error','raw_action_error']:
    assert summary[key]<1e-5,(key,summary[key])
assert summary['max_reference_acceleration']<=8.001
assert summary['min_reference_floor']>=-2.1e-5
assert summary['reset_events']==3072
