"""First-episode evaluation with the same fast resets and limits as training.

Terminal physics labels are copied to metrics before autoreset. Later episodes
in the same slot never contribute to the reported first-episode success rate.
"""
import argparse,functools,json
from pathlib import Path
from piper_sim_lab.runtime import configure,write_json

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--run',type=Path,required=True)
ap.add_argument('--checkpoint',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--num-envs',type=int,default=128)
ap.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028])
ap.add_argument('--modes',nargs='+',default=['empty_hand','pregrasp','grasp','preinside','near','mixed'],
                choices=['empty_hand','pregrasp','grasp','preinside','near','mixed'])
args=ap.parse_args()
configure('gpu')
import jax
import jax.numpy as jp
import numpy as np
from piper_sim_lab.quadratic_release import QuadraticRelease
from piper_sim_lab.checkpoint import load_policy
from piper_sim_lab.wrappers import wrap_for_training

cfg=json.loads((args.run/'config.json').read_text())
task_name=cfg['resolved_environment']['task']
assert task_name in ('quadratic_release','direct_gripper_release')
if task_name=='direct_gripper_release':
    from piper_sim_lab.direct_gripper import DirectGripperRelease as EvaluationBase
else:
    EvaluationBase=QuadraticRelease


class EvaluationTask(EvaluationBase):
    def snapshot(self,state):
        p=self._predicates(state.data);f=self._place_features(state.data)
        m=state.info['placement']
        labels=dict(inside=p['inside'],near=p['pre_grasp'],grasp=p['grasp'],above=p['pre_inside'],
            table_height=f['tcp_height'],xy=f['xy'],width=f['width'],
            supported=m['supported'],released=m['released'],qualified=m['release_paid'],
            recontact=m['recontact_seen'],grasp_confirmed=m['tf_grasped'],
            task_failed=state.info['task_failed'],task_reason=state.info['task_failure_reason'],
            controller_failed=state.info['controller_failed'],
            controller_reason=state.info['controller_failure_reason'],
            goal=state.info['gripper_goal'],phase=state.info.get('coord_phase',jp.array(-1)))
        return state.replace(metrics=dict(state.metrics,**{'eval_'+k:jp.asarray(v,jp.float32) for k,v in labels.items()}))

    def reset(self,rng):
        return self.snapshot(super().reset(rng))

    def step(self,state,action):
        return self.snapshot(super().step(state,action))


args.output.mkdir(parents=True,exist_ok=False)
reports={}
for mode in args.modes:
    c=dict(cfg['resolved_environment'],batch_capacity=args.num_envs)
    c.pop('task')
    if mode!='mixed':
        stage={'empty_hand':0,'pregrasp':1,'grasp':2,'preinside':3,'near':3}[mode]
        c['reset_stage_probs']=[float(i==stage) for i in range(4)]
        if stage==3:c['gentle_near_fraction']=float(mode=='near')
    env=EvaluationTask(c)
    policy=load_policy(args.checkpoint,observation_size=env.observation_size,action_size=7,deterministic=True)
    for seed in args.seeds:
        w=wrap_for_training(env,episode_length=cfg['training']['episode_length'],
            randomization_fn=functools.partial(env.physics_randomization_fn,
                rng=jax.random.split(jax.random.key(seed+7),args.num_envs)))
        state=jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(seed),args.num_envs))
        stages=np.asarray(state.info['reset_stage'])
        initial_goal=np.asarray(state.metrics['eval_goal'])
        initial_width=np.asarray(state.metrics['eval_width'])
        keys=[k for k in state.metrics if k.startswith('eval_')]
        keys+=['controlled_success','gentle_success','penalty_cost','failure_tail_cost','reward_reach',
               'gentle_reward','gentle_impact_cost','gentle_recontact_cost','action_penalty','action_rate_penalty',
               'reference_jerk','coord_reference_fault','coord_timeout']

        @jax.jit
        def rollout(state):
            def step(carry,i):
                old,alive=carry
                action,_=policy(old.obs,jax.random.fold_in(jax.random.PRNGKey(seed),i))
                st=w.step(old,action)
                sample={k:st.metrics[k] for k in keys}
                sample.update(valid=alive,done=st.done,reward=st.reward,gripper_action=action[:,6])
                return (st,alive&~st.done.astype(bool)),sample
            return jax.lax.scan(step,(state,jp.ones(args.num_envs,bool)),jp.arange(cfg['training']['episode_length']))[1]

        tr=jax.tree.map(np.asarray,jax.block_until_ready(rollout(state)))
        alive=tr['valid'];last=alive.sum(0)-1;idx=np.arange(args.num_envs)
        def final(k):return tr[k][last,idx]
        def ever(k):return ((tr[k]>0)&alive).any(0)
        failed=final('eval_task_failed')>0;controller=final('eval_controller_failed')>0
        ordinary=(final('eval_inside')>0)&~failed&~controller
        gentle=(final('controlled_success')>0)&~failed&~controller
        opening=((tr['gripper_action']>=env._config.gripper_threshold)&alive).any(0)
        closing=((tr['gripper_action']<=-env._config.gripper_threshold)&alive).any(0)
        previous_goal=np.concatenate([initial_goal[None],tr['eval_goal'][:-1]],axis=0)
        switched=((tr['eval_goal']!=previous_goal)&alive).any(0)
        names=['empty_hand','pregrasp','grasp','preinside']
        def summarize(mask):
            n=int(mask.sum())
            result=dict(episodes=n,ordinary_complete=int((ordinary&mask).sum()),
                ordinary_rate=float(ordinary[mask].mean()) if n else None,
                gentle_complete=int((gentle&mask).sum()),gentle_rate=float(gentle[mask].mean()) if n else None,
                grasp_timeout=int(((final('eval_task_reason')==1)&mask).sum()),
                outside_drop=int(((final('eval_task_reason')==2)&mask).sum()),
                controller_interruption=int((controller&mask).sum()),
                approached=int((ever('eval_near')&mask).sum()),grasped=int((ever('eval_grasp_confirmed')&mask).sum()),
                lifted=int((ever('eval_grasp')&mask).sum()),above_bowl=int((ever('eval_above')&mask).sum()),
                qualified_release=int((ever('eval_qualified')&mask).sum()),
                supported_landing=int((ever('eval_supported')&mask).sum()),
                postrelease_recontact=int((ever('eval_recontact')&mask).sum()))
            result.update(open_requested_episodes=int((opening&mask).sum()),
                close_requested_episodes=int((closing&mask).sum()),
                accepted_goal_switched_episodes=int((switched&mask).sum()))
            return result
        report=dict(mode=mode,seed=seed,overall=summarize(np.ones(args.num_envs,bool)),
            by_initialization={k:summarize(stages==i) for i,k in enumerate(names)},
            mean_duration_s=float(((last+1)*env.dt).mean()),
            mean_return=float((tr['reward']*alive).sum(0).mean()),
            mean_failure_settlement=float((tr['failure_tail_cost']*alive).sum(0).mean()),
            max_reference_jerk=float(np.where(alive,tr['reference_jerk'],0.).max()),
            reference_faults=int(ever('coord_reference_fault').sum()),
            final_tcp_height_mm_median=float(np.median(final('eval_table_height'))*1000),
            controller_reason_counts={str(int(k)):int(((final('eval_controller_reason')==k)&controller).sum())
                                      for k in np.unique(final('eval_controller_reason')[controller])},
            scope='First episode per slot only; terminal labels preserved before autoreset. No later episode or post-termination success credited.')
        np.savez_compressed(args.output/f'{mode}-s{seed}-trace.npz',reset_stage=stages,
                            initial_goal=initial_goal,initial_width=initial_width,**tr)
        write_json(args.output/f'{mode}-s{seed}.json',report);reports[f'{mode}-s{seed}']=report
        print(json.dumps(dict(mode=mode,seed=seed,overall=report['overall'])),flush=True)
        rollout.clear_cache()
        del w,state,tr
aggregate={}
for mode in args.modes:
    rows=[r['overall'] for r in reports.values() if r['mode']==mode]
    totals={k:sum(r[k] for r in rows) for k in rows[0] if not k.endswith('_rate')}
    totals.update(ordinary_rate=totals['ordinary_complete']/totals['episodes'],
                  gentle_rate=totals['gentle_complete']/totals['episodes'])
    aggregate[mode]=totals
write_json(args.output/'evaluation.json',dict(aggregate=aggregate,reports=reports,checkpoint=str(args.checkpoint),
    learning=False,real_robot=False,termination_handling='First true termination only; same autorestart as training.'))
print(json.dumps(dict(aggregate=aggregate)),flush=True)
