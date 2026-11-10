"""Direct-gripper acceptance, simultaneous motion and arm-bound validation."""
import argparse,functools,hashlib,json,time
from pathlib import Path
import yaml
from piper_sim_lab.runtime import configure
configure('gpu')
import jax
import jax.numpy as jp
import numpy as np
from piper_sim_lab.tasks import make_env
from piper_sim_lab.direct_gripper import direct_goal
from piper_sim_lab.wrappers import wrap_for_training

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--config',type=Path,required=True)
ap.add_argument('--doc-dir',type=Path,required=True)
ap.add_argument('--num-envs',type=int,default=1024)
ap.add_argument('--steps',type=int,default=240)
args=ap.parse_args();cfg=yaml.safe_load(args.config.read_text());n=args.num_envs
assert cfg['environment']['task']=='direct_gripper_release'
env=make_env(dict(cfg['environment'],batch_capacity=n))
w=wrap_for_training(env,episode_length=200,randomization_fn=functools.partial(
    env.physics_randomization_fn,rng=jax.random.split(jax.random.key(10031),n)))
state=jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(10030),n))
assert state.obs['state'].shape==(n,49) and state.obs['privileged'].shape==(n,66)
assert not any(k.startswith('coord_') or k in ['grip_reference','grip_motion_unlocked','grip_width_window'] for k in state.info)

@jax.jit
def run(state):
    def step(old,i):
        # First three packets exercise an opening pulse, immediate reversal,
        # then hold while the physical fingers are still responding.
        command=jp.where(i==0,1.,jp.where(i==1,-1.,jp.where(i==2,0.,jp.where((i//16)%2==0,-1.,1.))))
        action=jp.tile(jp.array([.1,0.,0.,0.,0.,.05,command]),(n,1))
        expected=direct_goal(command,old.info['gripper_goal'],env._config.gripper_max,env._config.gripper_threshold)
        old_width=old.data.qpos[:,env._finger_qpos_ids[0]]-old.data.qpos[:,env._finger_qpos_ids[1]]
        st=w.step(old,action)
        accepted=st.metrics['gripper_goal_open']*env._config.gripper_max
        reset=st.info['episode_done']>0
        width=st.data.qpos[:,env._finger_qpos_ids[0]]-st.data.qpos[:,env._finger_qpos_ids[1]]
        return st,dict(goal_error=jp.abs(accepted-expected),direct_error=st.metrics['gripper_direct_error'],
            simultaneous=(st.metrics['arm_reference_speed']>.005)&(jp.abs(width-old_width)>.001)&~reset,
            failure=st.metrics['controller_failed'],nonfinite=st.metrics['nonfinite'],overflow=st.metrics['capacity_overflow'],
            reference_fault=st.metrics['coord_reference_fault'],speed=st.metrics['arm_reference_speed'],
            acceleration=st.metrics['arm_reference_acceleration'],jerk=st.metrics['reference_jerk'],
            reset=reset,reset_potential=jp.where(reset,st.info['base_potential'],0.),
            finite_reward=jp.isfinite(st.reward),accepted=accepted,raw=command)
    return jax.lax.scan(step,state,jp.arange(args.steps))

started=time.monotonic();final,tr=jax.block_until_ready(run(state));tr=jax.tree.map(np.asarray,tr)
out=dict(environments=n,policy_transitions=n*args.steps,actor_observation=49,critic_observation=66,
    coordinator_state_absent=True,goal_mismatch_transitions=int((tr['goal_error']>1e-6).sum()),
    max_direct_reference_error=float(tr['direct_error'].max()),
    simultaneous_arm_and_gripper_transitions=int(tr['simultaneous'].sum()),
    immediate_open_accepted=int((tr['accepted'][0]>.049).sum()),
    immediate_reverse_accepted=int((tr['accepted'][1]<1e-6).sum()),
    hold_retains_closed_goal=int((tr['accepted'][2]<1e-6).sum()),
    controller_failed_transitions=int((tr['failure']>0).sum()),reference_faults=int((tr['reference_fault']>0).sum()),
    max_reference_speed=float(tr['speed'].max()),max_reference_acceleration=float(tr['acceleration'].max()),
    max_reference_jerk=float(tr['jerk'].max()),nonfinite=float(tr['nonfinite'].max()),overflow=float(tr['overflow'].max()),
    reset_events=int(tr['reset'].sum()),max_reset_potential=float(np.abs(tr['reset_potential']).max()),
    all_rewards_finite=bool(tr['finite_reward'].all()),elapsed_seconds=time.monotonic()-started,
    config=str(args.config.resolve()),config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
    training=False,real_robot=False,scope='Fixed actions; native force-limited actuator dynamics. Not learned success or hardware safety certification.')
checks=dict(no_coordinator=out['coordinator_state_absent'],all_goals_accepted=out['goal_mismatch_transitions']==0,
    no_gripper_ramp=out['max_direct_reference_error']<1e-6,
    first_open=out['immediate_open_accepted']==n,immediate_reversal=out['immediate_reverse_accepted']==n,
    hold_latched=out['hold_retains_closed_goal']==n,simultaneous_motion=out['simultaneous_arm_and_gripper_transitions']>0,
    controller_ok=out['controller_failed_transitions']==out['reference_faults']==0,
    bounds=out['max_reference_speed']<=env.settings.velocity+1e-5 and out['max_reference_acceleration']<=env.settings.acceleration+.001 and out['max_reference_jerk']<=env.settings.jerk+.2,
    physics_ok=out['nonfinite']==out['overflow']==0 and out['all_rewards_finite'],
    reset_clean=out['max_reset_potential']==0.)
out['checks']=checks;out['status']='passed' if all(checks.values()) else 'failed'
args.doc_dir.mkdir(parents=True,exist_ok=True)
(args.doc_dir/'controller-validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
assert all(checks.values()),checks
