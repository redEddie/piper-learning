"""Explicit close/open request probe under concurrent arm commands; no learning."""
import argparse, functools, json, hashlib
from pathlib import Path
import yaml
from piper_sim_lab.runtime import configure
configure('gpu')
import jax
import jax.numpy as jp
import numpy as np
from piper_sim_lab.tasks import make_env
from piper_sim_lab.wrappers import wrap_for_training

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--config',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
cfg=yaml.safe_load(args.config.read_text());n=64
env=make_env(dict(cfg['environment'],batch_capacity=n,reset_stage_probs=[.5,0.,.5,0.]))
w=wrap_for_training(env,episode_length=200,randomization_fn=functools.partial(
    env.physics_randomization_fn,rng=jax.random.split(jax.random.key(1087),n)))
state=jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(1088),n))
held=np.asarray(state.info['reset_stage']>=2)
initial_width=np.asarray(state.data.qpos[:,env._finger_qpos_ids[0]]-state.data.qpos[:,env._finger_qpos_ids[1]])

@jax.jit
def run(state):
    def step(carry,i):
        old,alive=carry
        # Empty slots: close at0.5s while moving, open at3s. Held slots:
        # request opening at0.5s. Observe only the initial episode: an intended
        # cube release may subsequently end in an outside-table task failure.
        command=jp.where(jp.asarray(held),jp.where(i<5,0.,1.),
                         jp.where(i<5,0.,jp.where(i<30,-1.,1.)))
        a=jp.tile(jp.array([.1,0.,0.,0.,0.,.05,0.]),(n,1)).at[:,6].set(command)
        st=w.step(old,a)
        return (st,alive&~st.done.astype(bool)),dict(goal=st.metrics['coord_accepted_goal'],width=st.metrics['coord_width'],
            phase=st.metrics['coord_phase_metric'],completed=st.metrics['coord_completed'],valid=alive,
            locked_speed=st.metrics['grip_arm_locked_reference_speed'],
            fault=st.metrics['coord_reference_fault'],failed=st.metrics['controller_failed'],
            reason=st.metrics['coord_reason'],done=st.done,
            nonfinite=st.metrics['nonfinite'],overflow=st.metrics['capacity_overflow'])
    return jax.lax.scan(step,(state,jp.ones(n,bool)),jp.arange(65))[1]

tr=jax.tree.map(np.asarray,jax.block_until_ready(run(state)))
valid=tr['valid'];empty=~held
close=((tr['goal'][5:30]<.001)&valid[5:30]).any(0)&empty
open_=((tr['goal'][30:]>.049)&valid[30:]).any(0)&empty
closed=((tr['width'][5:30]<.01)&valid[5:30]).any(0)&empty
opened=((tr['width'][30:]>.09)&valid[30:]).any(0)&empty
held_open=((tr['goal'][5:]>.049)&valid[5:]).any(0)&held
held_separated=((tr['width']>=initial_width[None]+.01)&valid).any(0)&held
out=dict(environments=n,steps=65,empty_environments=int(empty.sum()),held_environments=int(held.sum()),
    close_accepted=int(close.sum()),open_accepted=int(open_.sum()),
    actually_closed_below10mm=int(closed.sum()),actually_opened_above90mm=int(opened.sum()),
    held_open_accepted=int(held_open.sum()),held_width_increased10mm=int(held_separated.sum()),
    operations_completed=float((tr['completed']*valid).sum()),
    phase_counts={str(int(k)):int(((tr['phase']==k)&valid).sum()) for k in np.unique(tr['phase'])},
    max_locked_arm_reference_speed=float(tr['locked_speed'].max()),
    controller_failure_events=int(((tr['failed']>0)&valid).sum()),reference_faults=int(((tr['fault']>0)&valid).sum()),
    empty_reset_events=int(((tr['done']>0)&valid&empty[None]).sum()),nonfinite=float(tr['nonfinite'].max()),overflow=float(tr['overflow'].max()),
    config=str(args.config.resolve()),config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),training=False,real_robot=False,
    scope='Fixed forced requests while requesting arm motion; verifies command routing/interlock, not learned policy behavior.')
print(json.dumps(out),flush=True)
assert out['close_accepted']==out['open_accepted']==int(empty.sum())
assert out['actually_closed_below10mm']==out['actually_opened_above90mm']==int(empty.sum())
assert out['held_open_accepted']==out['held_width_increased10mm']==int(held.sum())
assert out['controller_failure_events']==out['reference_faults']==out['empty_reset_events']==0
assert out['max_locked_arm_reference_speed']<1e-4
assert out['nonfinite']==out['overflow']==0
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(out,indent=2)+'\n')
