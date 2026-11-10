"""First-episode, per-start evaluation of the measured SHOW-103 reward task."""
import argparse
import functools
import json
from pathlib import Path

from piper_sim_lab.runtime import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--starts', nargs='+', choices=['initial_a', 'pre_grasp', 'held', 'over_bowl', 'post_release', 'mixed'])
    args = parser.parse_args()
    configure('gpu')
    import jax
    import jax.numpy as jp
    import numpy as np
    from mujoco_playground import wrapper
    from piper_sim_lab.tasks import make_env
    from piper_sim_lab.checkpoint import load_policy
    from piper_sim_lab.recurrent import load_actor_checkpoint, zero_carry
    from piper_sim_lab.first_episode import latch

    if args.num_envs <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('positive num-envs and distinct seeds required')
    cfg = json.loads((args.run/'config.json').read_text())
    recurrent = cfg.get('learner') == 'recurrent_ppo'
    if cfg['resolved_environment']['task'] not in ('stage_reward_pick_place', 'stage_reward_pick_place_random_positions', 'stage_reward_pick_place_five_starts_v2'):
        parser.error('requires stage_reward_pick_place checkpoint/config')
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    names = ['initial_a', 'pre_grasp', 'held', 'over_bowl']
    if cfg['resolved_environment']['task'] == 'stage_reward_pick_place_five_starts_v2':
        names.append('post_release')
    for name in args.starts or names:
        if name != 'mixed' and name not in names:
            parser.error(f'{name} is unavailable in this checkpoint configuration')
        config = dict(cfg['resolved_environment'], batch_capacity=args.num_envs)
        if config.get('position_reset_bank'):
            config['position_reset_bank'] = config['evaluation_position_reset_bank']
        if name != 'mixed':
            config['reset_stage_probs'] = [float(i == names.index(name)) for i in range(len(names))]
        env = make_env(config)
        if recurrent:
            actor, actor_params, normalizer = load_actor_checkpoint(args.run, args.checkpoint, env.observation_size, 7)
        else:
            policy = load_policy(args.checkpoint, observation_size=env.observation_size,
                                 action_size=7, deterministic=True)
        for seed in args.seeds:
            physical_keys = jax.random.split(jax.random.key(seed+7), args.num_envs)
            dr = wrapper.BraxDomainRandomizationVmapWrapper(env,
                functools.partial(env.physics_randomization_fn, rng=physical_keys))

            @jax.jit
            def evaluate(keys):
                initial = dr.reset(keys)
                def step(carry, _):
                    state, terminal, failed, memory = carry
                    if recurrent:
                        memory, logits = actor.step(normalizer, actor_params, memory, state.obs, terminal)
                        action = actor.dist.mode(logits)
                    else:
                        action, _ = policy(state.obs, jax.random.key(0))
                    action=jp.where(terminal[:,None],0.,action)
                    updated = dr.step(state, action)
                    state,terminal,failed,active=latch(state,updated,terminal,failed)
                    f = state.info['reward_features']
                    sample = dict(action=action, observations=state.obs['state'],
                        qpos=state.data.qpos, qvel=state.data.qvel,
                        tcp=state.data.site_xpos[:, env._tcp_id], cube=state.data.xpos[:, env._block_id],
                        bowl=state.data.xpos[:, env._bowl_id], goal=state.info['gripper_goal'],
                        reference=state.data.ctrl[:, env._gripper_ctrl_id],
                        width=state.data.qpos[:, env._finger_qpos_ids[0]]-state.data.qpos[:, env._finger_qpos_ids[1]],
                        reward=state.reward, features=jp.stack(f, axis=-1),
                        memory_lifted=state.info['reward_memory'].lifted,
                        memory_bowl_reached=state.info['reward_memory'].bowl_reached,
                        reset_pose_index=state.info['reset_pose_index'],
                        measured_5ms=state.info['reward_trace'], failed=failed, terminal=terminal, active=active,
                        metrics=state.metrics)
                    return (state, terminal, failed, memory), sample
                failed = initial.done > 0
                return jax.lax.scan(step, (initial, failed, failed, zero_carry(args.num_envs)), None, length=cfg['training']['episode_length'])

            (_, _, failed, _), trace = jax.block_until_ready(evaluate(jax.random.split(jax.random.key(seed), args.num_envs)))
            trace = jax.tree.map(np.asarray, trace)
            valid = ~np.asarray(failed)
            metrics = trace.pop('metrics')
            flat = dict(trace, **{'metric_'+k: v for k, v in metrics.items()})
            np.savez_compressed(args.output/f'{name}-s{seed}.npz', **flat)
            row = dict(start=name, seed=seed, episodes=args.num_envs,
                final_complete=int(((metrics['success'][-1] > 0) & valid).sum()),
                ever_complete=int((np.any(metrics['success'] > 0, axis=0) & valid).sum()),
                final_inside=int(((metrics['inside_success'][-1] > 0) & valid).sum()),
                qualified_release=int(((metrics['qualified_release'][-1] > 0) & valid).sum()),
                settled=int(((metrics['settled'][-1] > 0) & valid).sum()),
                retreated=int((valid & (metrics['settled'][-1] > 0) &
                    (metrics['clearance'][-1] >= .08) &
                    (metrics['stage_recontact_cost'][-1] == 0)).sum()),
                failures=int((~valid).sum()),
                nonfinite=int(np.any(metrics['nonfinite'] > 0, axis=0).sum()),
                overflow=int(np.any(metrics['capacity_overflow'] > 0, axis=0).sum()),
                controller_failures=int(np.any(metrics['controller_failed'] > 0, axis=0).sum()),
                timeout_without_completion=int((valid & (metrics['success'][-1] == 0)).sum()),
                checkpoint=str(args.checkpoint.resolve()), dynamics_dr=bool(config['physics_randomization']),
                observation_dr=bool(config['obs_randomization']), reset_stage_probs=config['reset_stage_probs'])
            if config.get('position_reset_bank'):
                row.update(position_bank=config['position_reset_bank'],
                           position_bank_sha256=env.reset_bank['sha256'],
                           position_bank_split=env.reset_bank['split'])
            sections = metrics['stage_section'][-1].astype(int)
            row['final_section_counts'] = {str(i): int((sections == i).sum()) for i in range(-1, 5)}
            row['progress_counts'] = {
                'approached': int((np.any(metrics['stage_section'] >= 1, axis=0) & valid).sum()),
                'grasped_and_lifted': int((np.any(metrics['stage_section'] >= 2, axis=0) & valid).sum()),
                'reached_bowl': int((np.any(metrics['stage_section'] >= 3, axis=0) & valid).sum()),
                'qualified_release': int((np.any(metrics['qualified_release'] > 0, axis=0) & valid).sum()),
                'settled': int((np.any(metrics['settled'] > 0, axis=0) & valid).sum()),
            }
            # Local achievements survive later failures. Exclude the interval
            # containing the first fault; no post-fault measurements count.
            prefix_valid = trace['active'] & ~trace['failed']
            # The command causing the first failure was genuinely issued.
            # Include it when measuring OPEN exploration, while achievements
            # still exclude that failing interval and all subsequent samples.
            row['open_command_episodes'] = int(np.any(
                (trace['action'][...,6] >= config['gripper_threshold']) & trace['active'], axis=0).sum())
            row['invalid_release_episodes'] = int(np.any(
                (metrics['release_invalid'] > 0) & trace['active'], axis=0).sum())
            measured = trace['measured_5ms']
            near = np.any((measured[..., 0] <= .02) | (measured[..., 13] > 0) | (measured[..., 14] > 0), axis=-1)
            local_events = dict(initial_a=near, pre_grasp=trace['memory_lifted'],
                held=trace['memory_bowl_reached'], over_bowl=metrics['qualified_release'] > 0,
                post_release=metrics['success'] > 0)
            row['local_success'] = int(np.any(local_events[name] & prefix_valid, axis=0).sum()) if name != 'mixed' else None
            row['local_success_rate'] = row['local_success']/args.num_envs if name != 'mixed' else None
            row['local_success_definition'] = dict(initial_a='TCP-cube <=20mm or finger contact',
                pre_grasp='bilateral contact100ms and cube bottom >=20mm',
                held='after lift, finger contact, XY<=50mm and cube center above rim',
                over_bowl='qualified low first release with OPEN intention and20ms detachment',
                post_release='settled100ms, clearance80mm, stationary A within3deg for200ms').get(name)
            if config.get('five_start_physical_release'):
                if name=='over_bowl':
                    row['local_success_definition']='actual detachment then floor-supported whole-cube containment; high drops may succeed and terminate'
                row['high_drop_success']=int(np.any((metrics['stage_terminal_success']>0)&~trace['failed'],axis=0).sum())
                row['home_complete']=int(np.any((metrics['success']>0)&(metrics['stage_terminal_success']==0)&~trace['failed'],axis=0).sum())
            if config.get('five_start_progress_reward'):
                names_to_metric=dict(initial_a='approach_complete',pre_grasp='grasp_complete',
                    held='transport_complete',over_bowl='place_complete',post_release='home_complete')
                if name!='mixed':
                    event=metrics['stage_'+names_to_metric[name]]>0
                    if name=='over_bowl':event=event|(metrics['stage_terminal_success']>0)
                    row['local_success']=int(np.any(event&prefix_valid,axis=0).sum())
                    row['local_success_rate']=row['local_success']/args.num_envs
                if name=='over_bowl':row['local_success_definition']='full target opening and100ms supported whole-cube containment; >=10cm inside-drop terminal exception'
                if name=='post_release':row['local_success_definition']='all joints within0.2rad,speed<=0.05rad/s,clearance80mm,no contact,supported settling,hold200ms'
                row['mean_final_stage_progress']=float(metrics['stage_progress'][-1].mean())
                row['mean_final_completed_credit']=float(metrics['stage_completed_credit'][-1].mean())
            reports.append(row)
            (args.output/'summary.json').write_text(json.dumps(dict(reports=reports,
                success_definition=env.SUCCESS_DEFINITION, first_episode_only=True,
                training_seed=cfg['seed'], evaluation_seeds=args.seeds, real_robot=False), indent=2)+'\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
