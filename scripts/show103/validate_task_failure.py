"""GPU physics/contact and fast-reset validation; fixed actions, no learning."""
from pathlib import Path
import argparse,functools,json,hashlib
import yaml
from piper_sim_lab.runtime import configure
configure('gpu')
import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from piper_sim_lab.tasks import make_env
from piper_sim_lab.wrappers import wrap_for_training
from piper_sim_lab.quadratic_release import remaining_cost

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--config',type=Path,default=ROOT/'configs/show103/quadratic-fast-reset.yaml')
ap.add_argument('--doc-dir',type=Path,default=ROOT/'docs/show103/task-failure')
args=ap.parse_args()
cfg=yaml.safe_load(args.config.read_text())
n=64
# A controlled diagnostic fixture: all resets start empty-handed, not training.
env=make_env(dict(cfg['environment'],batch_capacity=n,reset_stage_probs=[1.,0.,0.,0.]))
w=wrap_for_training(env,episode_length=200,randomization_fn=functools.partial(
    env.physics_randomization_fn,rng=jax.random.split(jax.random.key(1077),n)))
state=jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(1078),n))
expected=(49,66) if cfg['environment']['task']=='direct_gripper_release' else (50,67)
assert state.obs['state'].shape==(n,expected[0]) and state.obs['privileged'].shape==(n,expected[1])


@jax.jit
def fixture(state):
    def prepare(data,i):
        # Slot0: previously lifted cube, now in free fall above table, outside bowl.
        # Slot1: previously lifted cube released onto the bowl's interior floor.
        cube=data.xpos[env._block_id]
        bowl=data.xpos[env._bowl_id]
        target=jp.where(i==1,bowl+jp.array([0.,0.,env._config.bowl_floor+.025]),
                        cube+jp.array([0.,0.,.10]))
        target=jp.where((i<2),target,cube)
        quat=jp.where(i<2,jp.array([1.,0.,0.,0.]),
                      data.qpos[env._block_qpos_adr+3:env._block_qpos_adr+7])
        qpos=jax.lax.dynamic_update_slice(data.qpos,jp.concatenate([target,quat]),(env._block_qpos_adr,))
        return mjx.forward(env._mjx_model,data.replace(qpos=qpos))
    data=jax.vmap(prepare)(state.data,jp.arange(n))
    m=state.info['placement'];held=jp.arange(n)<2
    m=dict(m,carried=m['carried']|held,p_active=m['p_active']|held,
           tf_grasped=m['tf_grasped']|held,tf_lifted=m['tf_lifted']|held)
    return state.replace(data=data,info=dict(state.info,placement=m))


state=jax.block_until_ready(fixture(state))


@jax.jit
def run(state):
    def step(old,i):
        # Slot2 is a deliberate controller interruption, not a task failure.
        inject=(i==20)&(jp.arange(n)==2)
        old=old.replace(info=dict(old.info,
            controller_failed=old.info['controller_failed']|inject,
            controller_failure_reason=jp.where(inject,128,old.info['controller_failure_reason'])))
        st=w.step(old,jp.zeros((n,7)))
        reset=st.info['episode_done']>0
        onset=st.metrics['failure_onset']>0
        expected=remaining_cost(st.metrics['failure_cost'],200-old.info['penalty_episode_steps']-1,.99)
        return st,dict(reset=reset,onset=onset,reason=st.metrics['task_failure_reason'],
            controller=st.metrics['controller_interruption'],tail=st.metrics['failure_tail_cost'],
            settlement_error=jp.where(onset,jp.abs(st.metrics['failure_tail_cost']-expected),0.),
            immediate_reset=jp.where(onset|inject,reset,True),
            clean_reset=jp.where(onset,(st.info['penalty_episode_steps']==0)&
                (st.info.get('base_potential',jp.zeros(n))==0),True),
            absorbing=st.info['failure_absorbing'],
            positive=jp.where(onset,st.metrics['penalty_positive_base'],0.),
            invalid=st.metrics['nonfinite'],overflow=st.metrics['capacity_overflow'],
            finite_reward=jp.isfinite(st.reward))
    return jax.lax.scan(step,state,jp.arange(240))


final,tr=jax.block_until_ready(run(state));tr=jax.tree.map(np.asarray,tr)
# The first reset of each slot distinguishes original fixture from subsequent episodes.
first0=int(np.flatnonzero(tr['reset'][:,0])[0]);first1=int(np.flatnonzero(tr['reset'][:,1])[0])
out=dict(environments=n,policy_transitions=n*240,
    table_drop_first_failure_s=(first0+1)*.1,
    table_drop_reason=int(tr['reason'][first0,0]),
    no_failure_during_initial_100ms_flight=bool(not tr['onset'][0,0]),
    bowl_landing_first_reset_s=(first1+1)*.1,
    bowl_landing_task_failure_before_reset=bool(tr['onset'][:first1+1,1].any()),
    first_grasp_timeout_s=(int(np.flatnonzero(tr['reason'][:,3]==1)[0])+1)*.1,
    grasp_timeouts=int(((tr['reason']==1)&tr['onset']).sum()),
    outside_drops=int(((tr['reason']==2)&tr['onset']).sum()),
    controller_interruptions=int(tr['controller'].sum()),
    injected_controller_tail=float(tr['tail'][20,2]),
    injected_controller_task_failure=bool(tr['onset'][20,2]),
    max_settlement_error=float(tr['settlement_error'].max()),
    all_failures_reset_immediately=bool(tr['immediate_reset'].all()),
    all_failed_slots_start_fresh=bool(tr['clean_reset'].all()),
    absorbing_slots=int(tr['absorbing'].sum()),positive_failure_reward_base=float(tr['positive'].max()),
    physical_nonfinite=float(tr['invalid'].max()),capacity_overflow=float(tr['overflow'].max()),
    all_rewards_finite=bool(tr['finite_reward'].all()),
    training_run=False,real_robot=False,
    scope='Fixed actions and constructed physical scenes; no learned-policy success-rate evaluation.')
assert out['table_drop_reason']==2 and .2<=out['table_drop_first_failure_s']<=1.
assert out['no_failure_during_initial_100ms_flight']
assert not out['bowl_landing_task_failure_before_reset'] and first1==199
assert out['grasp_timeouts']>=60 and out['outside_drops']==1
assert out['first_grasp_timeout_s']==12.
assert out['injected_controller_tail']==0 and not out['injected_controller_task_failure']
assert out['controller_interruptions']==1
assert out['max_settlement_error']<1e-4
assert out['all_failures_reset_immediately'] and out['all_failed_slots_start_fresh']
assert out['absorbing_slots']==out['positive_failure_reward_base']==0
assert out['physical_nonfinite']==out['capacity_overflow']==0 and out['all_rewards_finite']
out['config']=str(args.config.resolve())
out['config_sha256']=hashlib.sha256(args.config.read_bytes()).hexdigest()
out['controller_limits']=dict(velocity=env.settings.velocity,acceleration=env.settings.acceleration,jerk=env.settings.jerk)
doc=args.doc_dir;doc.mkdir(parents=True,exist_ok=True)
(doc/'integration.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
