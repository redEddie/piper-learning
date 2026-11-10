"""Inspect home rewards/DLS and run simulation-only feedback control probes.

Feedback actions in this script are diagnostic controllers, never policy
outputs or a training-environment override. No robot APIs are imported.
"""
import argparse
import json
from pathlib import Path

from piper_sim_lab.runtime import configure


def offline(args,cfg,env):
    import mujoco
    from mujoco import mjx
    import numpy as np
    from piper_sim_lab.five_start_stage import native_features
    model=env.mj_model;data=mujoco.MjData(model)
    nominal=np.asarray(env._nominal,dtype=float)
    scales=np.array([cfg['environment']['max_lin_vel']]*3+[cfg['environment']['max_ang_vel']]*3)
    damping=cfg['environment']['ik_damping']
    paths=sorted((args.run/'evaluation/2u').glob('post_release-s*.npz'))
    rejected=total=failures=0;norms=[];directions=[];home_direction=[];samples=[];jac_errors=[]
    first_score=[];last_score=[];gates=[]
    for path in paths:
        with np.load(path) as d:
            active=d['active']&~d['failed']
            total+=int(active.sum());rejected+=int(((d['metric_arm_rejected']>0)&active).sum())
            failures+=int(np.any(d['metric_controller_failed']>0,axis=0).sum())
            norms.extend(np.max(np.abs(d['action'][-50:,:,:6]),axis=-1).reshape(-1).tolist())
            first_score.extend(d['metric_stage_home_joint_progress'][0].tolist());last_score.extend(d['metric_stage_home_joint_progress'][-1].tolist())
            gates.extend(((d['metric_settled']>0)&(d['metric_clearance']>=.08)&(d['metric_stage_recontact_cost']==0)&active).mean(axis=0).tolist())
            for t in (1,5,10,20,50,100,199):
                for b in range(d['qpos'].shape[1]):
                    if not active[t,b]:continue
                    data.qpos[:]=d['qpos'][t-1,b];data.qvel[:]=d['qvel'][t-1,b]
                    mujoco.mj_forward(model,data)
                    p,r=np.zeros((3,model.nv)),np.zeros((3,model.nv))
                    mujoco.mj_jacSite(model,data,p,r,env._tcp_id)
                    jac=np.concatenate([p,r])[:,env._qvel_ids]
                    q=data.qpos[env._qpos_ids].copy();e=q-nominal
                    a=d['action'][t,b,:6]
                    desired=np.clip(jac.T@np.linalg.solve(jac@jac.T+damping**2*np.eye(6),a*scales),-1,1)
                    derivative=float(e@desired)
                    progress_derivative=float((-np.sign(e)*np.exp(-np.abs(e))/6)@desired)
                    directions.append([t,derivative,progress_derivative,float(np.linalg.norm(desired))])
                    home=-.5*e
                    projected=jac.T@np.linalg.solve(jac@jac.T+damping**2*np.eye(6),jac@home)
                    home_direction.append(float(e@projected))
                    if len(jac_errors)<8:
                        jac_errors.append(float(np.abs(np.asarray(env._tcp_jacobian(mjx.put_data(model,data,impl='jax')))-jac).max()))
                    if t==199 and b==0:
                        samples.append(dict(seed_trace=path.name,q=q.tolist(),singular_values=np.linalg.svd(jac,compute_uv=False).tolist(),
                            action=a.tolist(),desired_joint_velocity=desired.tolist(),actual_joint_velocity=d['qvel'][t,b,env._qvel_ids].tolist()))
    data.qpos[:]=env.reset_bank['qpos'][0][0];data.qpos[env._qpos_ids]=nominal;data.qvel[:]=0
    mujoco.mj_forward(model,data)
    p,r=np.zeros((3,model.nv)),np.zeros((3,model.nv));mujoco.mj_jacSite(model,data,p,r,env._tcp_id)
    jac=np.concatenate([p,r])[:,env._qvel_ids];sv=np.linalg.svd(jac,compute_uv=False)
    result=dict(scope='offline analysis of saved deterministic policy rollouts; no training or robot control',
        policy_control_intervals=total,qp_rejected_intervals=rejected,controller_failure_episodes=failures,
        late_max_abs_arm_action_median=float(np.median(norms)),late_max_abs_arm_action_p95=float(np.quantile(norms,.95)),
        native_vs_training_jacobian_max_error=max(jac_errors),
        sampled_direction_counts=dict(samples=len(directions),requested_L2_error_increasing=sum(x[1]>1e-8 for x in directions),
            requested_L2_error_decreasing=sum(x[1]<-1e-8 for x in directions),requested_home_progress_increasing=sum(x[2]>1e-8 for x in directions),
            home_directed_projected_commands_decrease_L2=sum(x< -1e-8 for x in home_direction)),
        mean_exponential_joint_progress_first_median=float(np.median(first_score)),mean_exponential_joint_progress_final_median=float(np.median(last_score)),
        home_bonus_gate_active_episode_fraction_median=float(np.median(gates)),
        at_A=dict(tcp_position=data.site_xpos[env._tcp_id].tolist(),singular_values=sv.tolist(),rank_at_1e_6=int((sv>1e-6).sum()),
            damped_singular_direction_gains=(sv**2/(sv**2+damping**2)).tolist()),
        final_policy_samples=samples)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def physics(args,cfg,env):
    import functools
    import jax
    import jax.numpy as jp
    import numpy as np
    from mujoco_playground import wrapper
    from piper_sim_lab.recurrent import load_actor_checkpoint,zero_carry
    n=args.num_envs;groups=4;count=n*groups
    actor,params,norm=load_actor_checkpoint(args.run/'training',args.checkpoint,env.observation_size,7)
    physical_keys=jp.tile(jax.random.split(jax.random.PRNGKey(2033),n),(groups,1))
    dr=wrapper.BraxDomainRandomizationVmapWrapper(env,functools.partial(env.physics_randomization_fn,rng=physical_keys))
    modes=jp.repeat(jp.arange(groups),n);scales=jp.array([env.settings.linear_scale]*3+[env.settings.angular_scale]*3)

    @jax.jit
    def experiment():
        keys=jp.tile(jax.random.split(jax.random.PRNGKey(2026),n),(groups,1))
        state=dr.reset(keys);memory=zero_carry(count)
        def prefix(c,_):
            state,memory=c
            memory,logits=actor.step(norm,params,memory,state.obs,state.done>0)
            return (dr.step(state,actor.dist.mode(logits)),memory),None
        (state,memory),_=jax.lax.scan(prefix,(state,memory),None,length=50)
        initial=state.data.qpos[:,env._qpos_ids]
        def step(c,_):
            state,memory,failed=c
            memory,logits=actor.step(norm,params,memory,state.obs,failed)
            learned=actor.dist.mode(logits)
            jac=jax.vmap(env._tcp_jacobian)(state.data)
            error=state.data.qpos[:,env._qpos_ids]-env._nominal
            goal_velocity=jp.clip(-.5*error,-.5,.5)
            projected=jp.einsum('bij,bj->bi',jac,goal_velocity)
            rhs=jp.einsum('bij,bj->bi',jp.swapaxes(jac,1,2)@jac+env.settings.damping**2*jp.eye(6),goal_velocity)
            compensated=jp.einsum('bij,bj->bi',jp.linalg.pinv(jp.swapaxes(jac,1,2),rtol=1e-6),rhs)
            raw=jp.where((modes==2)[:,None],projected,compensated)/scales
            home=raw/jp.maximum(1.,jp.max(jp.abs(raw),axis=-1,keepdims=True))
            action=jp.concatenate([home,jp.ones((count,1))],axis=-1)
            action=jp.where((modes==0)[:,None],learned,action)
            action=jp.where((modes==1)[:,None],jp.tile(jp.array([0.,0.,0.,0.,0.,0.,1.]),(count,1)),action)
            active=~failed;updated=dr.step(state,action);failed=failed|(updated.done>0)
            sample=dict(qpos=updated.data.qpos,qvel=updated.data.qvel,action=action,
                features=jp.stack(updated.info['reward_features'],axis=-1),
                tcp=updated.data.site_xpos[:,env._tcp_id],cube=updated.data.xpos[:,env._block_id],bowl=updated.data.xpos[:,env._bowl_id],
                goal=updated.info['gripper_goal'],reference=updated.data.ctrl[:,env._gripper_ctrl_id],
                width=updated.data.qpos[:,env._finger_qpos_ids[0]]-updated.data.qpos[:,env._finger_qpos_ids[1]],
                reward=updated.reward,active=active,failed=failed,metrics=updated.metrics)
            return (updated,memory,failed),sample
        (_,_,_),trace=jax.lax.scan(step,(state,memory,state.done>0),None,length=args.steps)
        return initial,trace
    initial,trace=jax.block_until_ready(experiment());trace=jax.tree.map(np.asarray,trace)
    metrics=trace.pop('metrics');flat=dict(trace,**{'metric_'+k:v for k,v in metrics.items()})
    np.savez_compressed(args.output.with_suffix('.npz'),**flat)
    records=[]
    names=['learned_policy','zero_arm_open','joint_descent_projected_to_TCP','joint_descent_DLS_compensated_TCP']
    for g,name in enumerate(names):
        sl=slice(g*n,(g+1)*n);valid=trace['active'][:,sl]&~trace['failed'][:,sl]
        err=metrics['home_error'][:,sl]
        records.append(dict(controller=name,episodes=n,
            start_max_error_deg_median=float(np.rad2deg(np.median(np.max(np.abs(np.asarray(initial)[sl]-np.asarray(env._nominal)),axis=-1)))),
            best_max_error_deg_median=float(np.rad2deg(np.median(np.min(np.where(valid,err,np.inf),axis=0)))),
            final_max_error_deg_median=float(np.rad2deg(np.median(err[-1]))),
            completed=int(np.any((metrics['success'][:,sl]>0)&valid,axis=0).sum()),
            failed=int(np.any(trace['failed'][:,sl],axis=0).sum()),
            qp_rejected_intervals=int(((metrics['arm_rejected'][:,sl]>0)&trace['active'][:,sl]).sum()),
            controller_failure_episodes=int(np.any(metrics['controller_failed'][:,sl]>0,axis=0).sum()),
            nonfinite=int(np.any(metrics['nonfinite'][:,sl]>0,axis=0).sum()),overflow=int(np.any(metrics['capacity_overflow'][:,sl]>0,axis=0).sum())))
    result=dict(scope='Diagnostic simulation controllers, not learned policies or training modifications. Same randomized post-release starts, same5s learned prefix and same DLS/QP/limits.',
        seed=2026,envs_per_controller=n,learned_prefix_seconds=5.,probe_seconds=args.steps*.1,records=records,
        real_robot=False,training=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['offline','physics'],default='offline')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--num-envs',type=int,default=16)
    p.add_argument('--steps',type=int,default=200)
    args=p.parse_args();configure('gpu' if args.phase=='physics' else 'cpu')
    from piper_sim_lab.tasks import make_env
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((args.run/'training/config.json').read_text())
    if args.phase=='physics':
        from piper_sim_lab.experiment_guard import require_idle_gpu
        require_idle_gpu()
        args.checkpoint=args.checkpoint or sorted((args.run/'training/checkpoints').iterdir())[-1]
    options=dict(cfg['resolved_environment'],impl='warp' if args.phase=='physics' else 'jax',
        batch_capacity=args.num_envs*4 if args.phase=='physics' else 1)
    if args.phase=='physics':
        options.update(reset_stage_probs=[0.,0.,0.,0.,1.],position_reset_bank=options['evaluation_position_reset_bank'])
    env=make_env(options)
    (physics if args.phase=='physics' else offline)(args,cfg,env)


if __name__=='__main__':main()
