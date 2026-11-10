"""GPU fixtures: low landing, high successful termination, and immediate miss.

Uses recorded airborne release prefixes with deterministic test scene offsets.
These are physics fixtures, not learned rollouts, training resets or robot runs.
"""
import argparse
import json
from pathlib import Path

from piper_sim_lab.runtime import configure


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();configure('gpu')
    import functools
    import jax
    import jax.numpy as jp
    import numpy as np
    import yaml
    from mujoco import mjx
    from mujoco_playground import wrapper
    from piper_sim_lab.tasks import make_env
    from piper_sim_lab.first_episode import latch
    from piper_sim_lab.experiment_guard import require_idle_gpu
    require_idle_gpu()
    cfg=yaml.safe_load(Path('configs/show103/stage-reward-v4-transformer-lstm-physical-release-plus2.yaml').read_text())
    n=16;count=3*n
    env=make_env(dict(cfg['environment'],batch_capacity=count,fixed_reset_settle_steps=0,
        reset_stage_probs=[0.,0.,0.,0.,1.],position_reset_bank=cfg['environment']['evaluation_position_reset_bank']))
    keys=jp.tile(jax.random.split(jax.random.PRNGKey(2026),n),(3,1))
    dr=wrapper.BraxDomainRandomizationVmapWrapper(env,functools.partial(env.physics_randomization_fn,
        rng=jp.tile(jax.random.split(jax.random.PRNGKey(2033),n),(3,1))))

    @jax.jit
    def experiment():
        state=dr.reset(keys)
        offset=jp.concatenate([jp.zeros(n),jp.full(n,.15),jp.zeros(n)])
        side=jp.concatenate([jp.zeros(2*n),jp.full(n,.2)])
        q=state.data.qpos.at[:,env._block_qpos_adr+2].add(offset).at[:,env._block_qpos_adr].add(side)
        high=(jp.arange(count)>=n)&(jp.arange(count)<2*n)
        # Move the high-drop fixture arm above the airborne cube so that the
        # cube does not land on the gripper. This is not a policy action.
        arm=env._nominal.at[5].set(.5)
        q=q.at[:,env._qpos_ids].set(jp.where(high[:,None],arm,q[:,env._qpos_ids]))
        velocity=state.data.qvel.at[:,env._qvel_ids].set(jp.where(high[:,None],0.,state.data.qvel[:,env._qvel_ids]))
        ctrl=state.data.ctrl.at[:,env._arm_ctrl_ids].set(jp.where(high[:,None],arm,state.data.ctrl[:,env._arm_ctrl_ids]))
        data=jax.vmap(lambda d:mjx.kinematics(env._mjx_model,d))(state.data.replace(qpos=q,qvel=velocity,ctrl=ctrl))
        f=jax.vmap(env.features)(data,state.info['gripper_goal'])
        post=jp.concatenate([ctrl[:,env._arm_ctrl_ids],jp.zeros((count,12))],axis=-1)
        post=jp.where(high[:,None],post,state.info['post_state'])
        backup=jp.where(high[:,None,None],jp.repeat(post[:,None,:],env.stop_steps,axis=1),state.info['stop_backup'])
        state=state.replace(data=data,info=dict(state.info,reward_features=f,post_state=post,stop_backup=backup,
            upstream_goal=jp.where(high[:,None],arm,state.info['upstream_goal']),
            upstream_velocity=jp.where(high[:,None],0.,state.info['upstream_velocity'])))
        def step(c,_):
            state,terminal,failed=c
            updated=dr.step(state,jp.tile(jp.array([0.,0.,0.,0.,0.,0.,1.]),(count,1)))
            state,terminal,failed,active=latch(state,updated,terminal,failed)
            out=dict(terminal=terminal,failed=failed,active=active,done=state.done,reward=state.reward,
                features=jp.stack(state.info['reward_features'],axis=-1),
                success=state.metrics['success'],height=state.metrics['stage_freefall_height'],
                opening=state.metrics['stage_opening_bonus'],landing=state.metrics['stage_landing_bonus'],
                drop_cost=state.metrics['stage_drop_cost'],qualified=state.metrics['qualified_release'],
                nonfinite=state.metrics['nonfinite'],overflow=state.metrics['capacity_overflow'])
            return (state,terminal,failed),out
        return jax.lax.scan(step,(state,jp.zeros(count,bool),jp.zeros(count,bool)),None,length=40)[1]
    trace=jax.tree.map(np.asarray,jax.block_until_ready(experiment()))
    np.savez_compressed(args.output.with_suffix('.npz'),**trace)
    records=[]
    for group,name in enumerate(('recorded_low_release','raised_cube_high_drop','sideways_miss')):
        sl=slice(group*n,(group+1)*n)
        record=dict(case=name,worlds=n,terminal=int(trace['terminal'][-1,sl].sum()),
            successes=int((trace['success'][-1,sl]>0).sum()),failures=int(trace['failed'][-1,sl].sum()),
            freefall_height_min_max=trace['height'][-1,sl].min().item(),
            freefall_height_max=float(trace['height'][-1,sl].max()),
            final_opening_bonus_min_max=[float(trace['opening'][-1,sl].min()),float(trace['opening'][-1,sl].max())],
            final_landing_bonus_min_max=[float(trace['landing'][-1,sl].min()),float(trace['landing'][-1,sl].max())])
        records.append(record)
    print(json.dumps(records,indent=2),flush=True)
    assert records[0]['successes']==records[0]['failures']==records[0]['terminal']==0
    assert np.all(trace['landing'][-1,:n]==1.)
    assert records[1]['successes']==records[1]['terminal']==n and records[1]['failures']==0
    assert np.all(trace['landing'][-1,n:2*n]==0.) and np.all(trace['height'][-1,n:2*n]>=.1-1e-6)
    assert records[2]['failures']==records[2]['terminal']==n and records[2]['successes']==0
    assert np.all(trace['drop_cost'][0,2*n:]==2.)
    assert not trace['nonfinite'].any() and not trace['overflow'].any()
    out=dict(status='passed',scope=__doc__,records=records,first_miss_done_step=1,
        successful_terminal_is_not_failure=True,training=False,real_robot=False)
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
