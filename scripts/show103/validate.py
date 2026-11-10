"""Fixed-action GPU validation, including serialized gripping and full resets."""
from pathlib import Path
import functools,json,time,argparse,hashlib
import yaml
from piper_sim_lab.runtime import configure
configure('gpu')
import jax
import jax.numpy as jp
import numpy as np
from piper_sim_lab.tasks import make_env
from piper_sim_lab.wrappers import wrap_for_training

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--num-envs',type=int,default=64);ap.add_argument('--steps',type=int,default=240);ap.add_argument('--tag',default='integration')
ap.add_argument('--config',type=Path,default=ROOT/'configs/show103/refine-8m.yaml')
ap.add_argument('--doc-dir',type=Path,default=ROOT/'docs/show103');args=ap.parse_args()
c=yaml.safe_load(args.config.read_text())
n=args.num_envs;env=make_env(dict(c['environment'],batch_capacity=n))
w=wrap_for_training(env,episode_length=200,randomization_fn=functools.partial(env.physics_randomization_fn,rng=jax.random.split(jax.random.key(10031),n)))
state=jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(10030),n))
assert state.obs['state'].shape==(n,50);assert state.obs['privileged'].shape==(n,67)
@jax.jit
def run(state):
    def step(st,i):
        # Static cube task, with alternating requested open/close and arm motion.
        command=jp.where((i//16)%2==0,-1.,1.)
        a=jp.tile(jp.array([.1,0.,0.,0.,0.,.05,command]),(n,1))
        st=w.step(st,a);tr=st.info['coord_trace'];reset=st.info['episode_done']
        return st,dict(jerk=jp.max(st.metrics['reference_jerk']),failure=st.metrics['controller_failed'],
            reason=st.metrics['coord_reason'],width=st.metrics['coord_width'],goal=st.metrics['coord_accepted_goal'],
            operation_phase=st.metrics['coord_phase_metric'],nonfinite=jp.max(st.metrics['nonfinite']),overflow=jp.max(st.metrics['capacity_overflow']),
            grip_arm_speed=jp.max(st.metrics['grip_arm_locked_reference_speed']),reset=jp.sum(reset),
            overall_grip_arm_speed=jp.max(st.metrics['grip_arm_reference_speed']),
            phase=st.info['coord_phase'],completed=st.metrics['coord_completed'],rejected=st.metrics['coord_rejected'],
            near=st.info['near_reset'],reset_stage=st.info['reset_stage'],
            placement_bad_initial_credit=jp.max(jp.where(reset>0,st.info['placement']['clear_time'],0.)),
            max_v=jp.max(tr[:,:,5]),max_a=jp.max(tr[:,:,6]),
            reset_backup_error=jp.max(jp.where(reset[:,None,None]>0,jp.abs(st.info['stop_backup'][:,:,6:]),0.)))
    return jax.lax.scan(step,state,jp.arange(args.steps))
started=time.monotonic();final,tr=jax.block_until_ready(run(state));t=jax.tree.map(np.asarray,tr)
summary=dict(environments=n,policy_transitions=n*args.steps,reset_events=int(t['reset'].sum()),
    max_reference_jerk=float(t['jerk'].max()),max_reference_speed=float(t['max_v'].max()),max_reference_acceleration=float(t['max_a'].max()),
    max_reference_speed_during_grip=float(t['overall_grip_arm_speed'].max()),controller_failed_transitions=int(t['failure'].sum()),
    failure_codes=np.unique(t['reason']).tolist(),operations_completed=float(t['completed'].sum()),rejected_proposals=float(t['rejected'].sum()),
    nonfinite=float(t['nonfinite'].max()),overflow=float(t['overflow'].max()),reset_backup_error=float(t['reset_backup_error'].max()),
    elapsed_seconds=time.monotonic()-started,training=False,real_robot=False,
    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
summary['failed_events']=[dict(step=int(i),world=int(e),code=float(t['reason'][i,e]),width_mm=float(t['width'][i,e]*1000),goal_mm=float(t['goal'][i,e]*2000),phase=float(t['operation_phase'][i,e])) for i,e in np.argwhere(t['failure']>0)]
summary.update(near_reset_policy_steps=int(t['near'].sum()),near_table=env.near_table_report,
    near_reset_slots=int(t['near'][0].sum()),placement_bad_initial_credit=float(t['placement_bad_initial_credit'].max()),
    reset_stage_probs=list(c['environment']['reset_stage_probs']),
    conditional_near_fraction=float(env._config.gentle_near_fraction),
    max_reference_speed_while_locked=float(t['grip_arm_speed'].max()),
    max_reference_speed_during_any_grip=float(t['overall_grip_arm_speed'].max()),
    task=c['environment']['task'])
args.doc_dir.mkdir(parents=True,exist_ok=True)
(args.doc_dir/f'{args.tag}.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
assert summary['nonfinite']==summary['overflow']==0
assert summary['max_reference_jerk']<=env.settings.jerk+.2
assert summary['max_reference_speed']<=env.settings.velocity+1e-5
assert summary['max_reference_acceleration']<=env.settings.acceleration+.001
assert summary['max_reference_speed_while_locked']<1e-4
assert summary['controller_failed_transitions']==0
