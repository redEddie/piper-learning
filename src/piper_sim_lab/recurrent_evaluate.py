"""Paired held-out empty-hand episodes for the matched MLP/LSTM experiment."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--num-envs',type=int,default=128)
    ap.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028])
    ap.add_argument('--conditions',nargs='+',choices=['dr','nodr'],default=['dr','nodr'])
    ap.add_argument('--platform',choices=['cpu','gpu'],default='gpu')
    args=ap.parse_args()
    from piper_sim_lab.runtime import configure,write_json
    configure(args.platform)
    if args.platform=='gpu':
        from piper_sim_lab.experiment_guard import require_idle_gpu
        require_idle_gpu()
    import jax
    import jax.numpy as jp
    import numpy as np
    from flax import serialization
    from brax.training.acme import running_statistics as stats
    from piper_sim_lab.recurrent import ActorAdapter,zero_carry
    from piper_sim_lab.tasks import make_env
    from mujoco_playground import wrapper as pg_wrapper
    cfg=json.loads((args.run/'config.json').read_text())
    checkpoint=args.checkpoint or sorted((args.run/'checkpoints').iterdir())[-1]
    args.output.mkdir(parents=True,exist_ok=False)
    restored=serialization.msgpack_restore((checkpoint/'learner.msgpack').read_bytes())
    params=jax.tree.map(jp.asarray,restored['params']['policy'])
    reports=[]
    for condition in args.conditions:
        config=dict(cfg['resolved_environment'],batch_capacity=args.num_envs,
                    reset_stage_probs=[1.,0.,0.,0.])
        if condition=='nodr':
            config.update(obs_randomization=False,physics_randomization=False)
        if args.platform=='cpu':config['impl']='jax'
        env=make_env(config)
        actor=ActorAdapter(env.observation_size,env.action_size,cfg['actor'],cfg['network'])
        norm=serialization.from_state_dict(stats.init_state(
            {k:jp.zeros(v) for k,v in env.observation_size.items()}),restored['normalizer'])
        rows=[]
        for seed in args.seeds:
            v_reset,v_step=jax.vmap(env.reset),jax.vmap(env.step)
            if config.get('physics_randomization'):
                dr=pg_wrapper.BraxDomainRandomizationVmapWrapper(env,functools.partial(
                    env.physics_randomization_fn,rng=jax.random.split(jax.random.key(seed+7),args.num_envs)))
                v_reset,v_step=dr.reset,dr.step
            state=jax.jit(v_reset)(jax.random.split(jax.random.key(seed),args.num_envs))
            scene_hash = hashlib.sha256(np.asarray(state.data.qpos).tobytes() + np.asarray(state.data.mocap_pos).tobytes()).hexdigest()
            @jax.jit
            def rollout(state):
                def f(c,_):
                    state,memory,failed,ever=c
                    memory,logits=actor.step(norm,params,memory,state.obs,failed)
                    action=actor.dist.mode(logits)
                    nxt=v_step(state,action)
                    failed=failed | (nxt.done>0) | (nxt.metrics['capacity_overflow']>0) | (nxt.metrics['nonfinite']>0)
                    finite=jp.all(jp.isfinite(nxt.data.qpos),axis=-1)&jp.all(jp.isfinite(nxt.data.qvel),axis=-1)
                    failed=failed | ~finite
                    success=(nxt.metrics['success']>0)&~failed
                    ever=ever|success
                    sample=(success,failed,nxt.data.qpos[:8],nxt.data.mocap_pos[:8],nxt.metrics[env.PROGRESS_METRIC])
                    return (nxt,memory,failed,ever),sample
                return jax.lax.scan(f,(state,zero_carry(args.num_envs),jp.zeros(args.num_envs,bool),
                    jp.zeros(args.num_envs,bool)),None,length=cfg['training']['episode_length'])
            started=time.monotonic()
            (final,_,failed,ever),trajectory=jax.block_until_ready(rollout(state))
            success,failed_trace,qpos,mocap,progress=[np.asarray(x) for x in trajectory]
            valid=~np.asarray(failed)
            final_success=success[-1]&valid
            ever=np.asarray(ever)&valid
            row={'condition':condition,'seed':seed,'episodes':args.num_envs,
                'initial_scene_sha256':scene_hash,
                'final_successes':int(final_success.sum()),'ever_successes':int(ever.sum()),
                'failures':int((~valid).sum()),'final_success_rate':float(final_success.mean()),
                'ever_success_rate':float(ever.mean()),'elapsed_compile_and_rollout_seconds':time.monotonic()-started,
                'final_success_per_episode':final_success.tolist(),'ever_success_per_episode':ever.tolist(),
                'valid_per_episode':valid.tolist()}
            rows.append(row)
            np.savez_compressed(args.output/f'trajectory-{condition}-s{seed}.npz',qpos=qpos,mocap_pos=mocap,
                                success=success,failed=failed_trace,progress=progress)
            write_json(args.output/f'{condition}-s{seed}.json',row)
            print(json.dumps({k:v for k,v in row.items() if 'per_episode' not in k}),flush=True)
        count=sum(r['episodes'] for r in rows);wins=sum(r['final_successes'] for r in rows)
        p=wins/count;z=1.96;den=1+z*z/count
        mid=(p+z*z/(2*count))/den;half=z*((p*(1-p)/count+z*z/(4*count*count))**.5)/den
        reports.append({'condition':condition,'episodes':count,'final_successes':wins,
            'final_success_rate':p,'wilson_95_interval':[mid-half,mid+half],
            'ever_success_rate':sum(r['ever_successes'] for r in rows)/count,
            'failures':sum(r['failures'] for r in rows),'by_seed':rows,'environment':config})
    write_json(args.output/'evaluation.json',{'actor':cfg['actor'],'training_seed':cfg['seed'],
        'checkpoint':str(checkpoint.resolve()),'train_steps':int(restored['step']),
        'reset_distribution':'empty hand; all episodes stage 0','success_definition':env.SUCCESS_DEFINITION,
        'confidence_interval_scope':'episode-level Wilson; does not quantify training-seed variability',
        'conditions':reports})


if __name__=='__main__':main()
