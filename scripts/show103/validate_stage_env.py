"""Actual GPU-Warp/DR acceptance for SHOW-103; never starts policy training."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from piper_sim_lab.runtime import configure, GpuSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, required=True)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/show103/stage-reward-v1-4096.yaml'))
    parser.add_argument('--bank-split', choices=['train', 'eval'], default='train')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    configure('gpu')
    import functools
    import jax
    import jax.numpy as jp
    import numpy as np
    import yaml
    from mujoco_playground import wrapper
    from piper_sim_lab.tasks import make_env

    n = args.num_envs
    config_path = args.config
    cfg = yaml.safe_load(config_path.read_text())
    if args.bank_split == 'eval':
        cfg['environment']['position_reset_bank'] = cfg['environment']['evaluation_position_reset_bank']
    env = make_env(dict(cfg['environment'], batch_capacity=n))
    reward_terms, advance_memory = env.reward_model.reward_terms, env.reward_model.advance_memory
    start_interval, Features = env.reward_model.start_interval, env.reward_model.Features
    batched = wrapper.BraxDomainRandomizationVmapWrapper(env,
        functools.partial(env.physics_randomization_fn, rng=jax.random.split(jax.random.key(2033), n)))
    reset = jax.jit(batched.reset)
    step = jax.jit(batched.step)
    started = time.monotonic()
    print(f'Created {n} GPU physics worlds; compiling reset', flush=True)
    with GpuSampler() as gpu:
        state = jax.block_until_ready(reset(jax.random.split(jax.random.key(2026), n)))
        assert state.obs['state'].shape == (n, 49)
        assert state.obs['privileged'].shape == (n, 66)
        assert not any(k in state.info for k in ('coord_phase', 'grip_reference', 'actuated_goal'))
        counts = np.bincount(np.asarray(state.info['reset_stage']), minlength=len(env._stage_probs)).tolist()
        diversity = {}
        if cfg['environment'].get('position_reset_bank'):
            positions = np.asarray(state.data.qpos)
            labels = np.asarray(state.info['reset_stage'])
            for start in range(len(env._stage_probs)):
                group = positions[labels == start]
                cube = group[:, env._block_qpos_adr:env._block_qpos_adr+2]
                bowl = group[:, env._bowl_qpos_adr:env._bowl_qpos_adr+2]
                diversity[str(start)] = dict(cube_xy_std=cube.std(axis=0).tolist(),
                    bowl_xy_std=bowl.std(axis=0).tolist(), samples=len(group),
                    cube_xy_min=cube.min(0).tolist(),cube_xy_max=cube.max(0).tolist(),
                    bowl_xy_min=bowl.min(0).tolist(),bowl_xy_max=bowl.max(0).tolist())
                assert cube.std(axis=0).min() > 1e-3 and bowl.std(axis=0).min() > 1e-3
                if env._config.position_bound_yaw:
                    for xy,x,y in [(cube,env._config.block_x,env._config.block_y),
                                   (bowl,env._config.bowl_x,env._config.bowl_y)]:
                        # Native/GPU settling can move the dynamic objects a few mm.
                        assert np.all(xy >= np.array([x[0],y[0]])-.003)
                        assert np.all(xy <= np.array([x[1],y[1]])+.003)
                if start < 3:
                    assert np.linalg.norm(cube-bowl, axis=1).min() >= .139
        perception_summary={'enabled':bool(env._config.obs_randomization)}
        if env._config.obs_randomization:
            perception=state.info['perception']
            names=['obj_delay','robot_delay','hold_substeps','size_scale']
            if env._config.bowl_pose_delay_max>0:
                names.append('bowl_delay')
            for key in names:
                values=np.asarray(perception[key])
                perception_summary[key]=dict(min=float(values.min()),max=float(values.max()),std=float(values.std()))
                assert values.std()>0
            actor=np.asarray(state.obs['state']);critic=np.asarray(state.obs['privileged'])
            true_cube=np.asarray(state.data.xpos)[:,env._block_id]-np.asarray(env._base_offset)
            true_bowl=np.asarray(state.data.xpos)[:,env._bowl_id]-np.asarray(env._base_offset)
            error=actor[:,25:28]-true_cube
            assert np.abs(error).max()<=env._config.obj_pos_noise_cap+1e-6
            assert error.std()>1e-3
            np.testing.assert_allclose(critic[:,-10:-7],true_cube,atol=1e-6)
            np.testing.assert_allclose(actor[:,31:34]-true_bowl,np.asarray(perception['bowl_offset']),atol=1e-6)
            np.testing.assert_allclose(actor[:,34:37],actor[:,25:28]-actor[:,18:21],atol=1e-6)
            np.testing.assert_allclose(actor[:,37:40],actor[:,25:28]-actor[:,31:34],atol=1e-6)
            perception_summary.update(cube_position_error_std_mm=(error.std(0)*1000).tolist(),
                cube_position_error_abs_max_mm=float(np.abs(error).max()*1000),
                clean_critic_geometry_verified=True,observed_relative_positions_consistent=True)
        reset_contacts = int(np.asarray(state.metrics['reset_contact_invalid']).sum())
        assert reset_contacts == 0, f'{reset_contacts} held resets lack bilateral contact'
        post = np.asarray(state.info['reset_stage']) == 4
        progress_reward=cfg['environment'].get('five_start_progress_reward',False)
        if progress_reward:
            assert np.asarray(state.info['reward_memory'].grasp_complete)[np.asarray(state.info['reset_stage'])>=2].all()
            assert np.asarray(state.info['reward_memory'].transport_complete)[np.asarray(state.info['reset_stage'])>=3].all()
            if not np.asarray(state.info['reward_memory'].place_complete)[post].all():
                print('Post-release reset not yet complete:',dict(
                    completed=int(np.asarray(state.info['reward_memory'].place_complete)[post].sum()),total=int(post.sum()),
                    support_time_quantiles=np.quantile(np.asarray(state.info['reward_memory'].support_time)[post],[0,.1,.5,1]).tolist(),
                    width_quantiles=np.quantile(np.asarray(state.info['reward_features'].finger_width)[post],[0,.5,1]).tolist()),flush=True)
            assert np.asarray(state.info['reward_memory'].place_complete)[post].all()
        else:
            assert not np.asarray(state.info['reward_memory'].lifted)[~post].any()
            assert not np.asarray(state.info['reward_memory'].grasp_time)[~post].any()
        assert not np.asarray(state.info['reward_memory'].released)[~post].any()
        assert np.asarray(state.info['reward_memory'].qualified_release)[post].all()
        assert not np.asarray(state.info['reward_memory'].invalid_release).any()
        assert not np.asarray(state.info['reward_memory'].outside_drop)[post].any()
        assert not np.asarray(state.info['reward_features'].left_contact)[post].any()
        assert not np.asarray(state.info['reward_features'].right_contact)[post].any()
        checks = dict(nonfinite=0, capacity_overflow=0, controller_failed=0,
                      max_speed=0., max_acceleration=0., max_jerk=0.,
                      max_gripper_reference_step=0., max_reward_replay_error=0.,
                      reward_memory_replay_error=0., all_finite=True)
        for i in range(args.steps):
            # Simultaneous small arm request with opening, immediate reversal and HOLD.
            grip = 1. if i == 2 else (-1. if i == 3 else 0.)
            action = jp.tile(jp.array([.02 if i >= 2 else 0., 0., 0., 0., 0., 0., grip]), (n, 1))
            old = state
            state = jax.block_until_ready(step(state, action))
            for key in ('nonfinite', 'capacity_overflow', 'controller_failed'):
                checks[key] += int(np.asarray(state.metrics[key]).sum())
            for name, key in [('max_speed', 'arm_reference_speed'),
                              ('max_acceleration', 'arm_reference_acceleration'),
                              ('max_jerk', 'reference_jerk')]:
                checks[name] = max(checks[name], float(np.asarray(state.metrics[key]).max()))
            checks['max_gripper_reference_step'] = max(checks['max_gripper_reference_step'],
                float(np.abs(np.asarray(state.data.ctrl[:, env._gripper_ctrl_id]-old.data.ctrl[:, env._gripper_ctrl_id])).max()))
            checks['all_finite'] &= bool(np.isfinite(np.asarray(state.data.qpos)).all() and np.isfinite(np.asarray(state.data.qvel)).all())
            assert np.all(np.asarray(state.info['reward_substeps']) == 20)
            if cfg['environment'].get('five_start_dense_home'):
                actual_errors=np.stack([np.asarray(getattr(state.info['reward_features'],'home_error_'+str(j))) for j in range(6)],axis=-1)
                expected_errors=np.abs(np.asarray(state.data.qpos)[:,env._qpos_ids]-np.asarray(env._nominal))
                np.testing.assert_allclose(actual_errors,expected_errors,atol=2e-6)
                if progress_reward:
                    excess=np.maximum(expected_errors-env.reward_settings.home_error,0.)
                    dist=.7*excess.max(axis=-1)+.3*excess.mean(axis=-1)
                    reference=np.asarray(state.info['reward_memory'].home_distance_start)
                    expected_progress=np.where(reference>1e-6,np.clip(1.-dist/np.maximum(reference,1e-6),0.,1.),np.where(dist<=1e-6,1.,0.))
                else:
                    expected_progress=np.exp(-expected_errors/env.reward_settings.home_progress_scale).mean(axis=-1)
                np.testing.assert_allclose(np.asarray(state.metrics['stage_home_joint_progress']),expected_progress,atol=2e-6)
            # Replay measured per-world 5ms features through the original pure equations.
            memory = jax.tree.map(lambda x: x[0], old.info['reward_memory'])
            before = jax.tree.map(lambda x: x[0], old.info['reward_features'])
            memory = start_interval(memory)
            for row in np.asarray(state.info['reward_trace'][0]):
                after = Features(*(jp.asarray(row[k], dtype=jp.bool_ if isinstance(Features()[k],bool) else jp.float32)
                                   for k in range(len(Features._fields))))
                memory = advance_memory(memory, before, after, env.reward_settings)
                before = after
            actual = jax.tree.map(lambda x: x[0], state.info['reward_memory'])
            errors = []
            for a, b in zip(memory, actual):
                a, b = np.asarray(a).astype(float), np.asarray(b).astype(float)
                assert not np.isnan(a).any() and not np.isnan(b).any()
                errors.append(0. if np.array_equal(a, b) else float(np.max(np.abs(a-b))))
            checks['reward_memory_replay_error'] = max(checks['reward_memory_replay_error'],
                max(errors))
            terms = reward_terms(jax.tree.map(lambda x: x[0], old.info['reward_memory']),
                memory, before, action[0], old.obs['state'][0, -7:], env.reward_settings)
            checks['max_reward_replay_error'] = max(checks['max_reward_replay_error'],
                abs(float(terms['reward'])-float(state.reward[0])))
            if i in (0, args.steps-1):
                print(f'{n} worlds: step {i+1}/{args.steps}, diagnostics {checks}', flush=True)
        # Fresh resets must clear episode reward ledgers in all worlds.
        fresh = jax.block_until_ready(reset(jax.random.split(jax.random.key(2027), n)))
        assert np.asarray(fresh.metrics['reset_contact_invalid']).sum() == 0
        fresh_post = np.asarray(fresh.info['reset_stage']) == 4
        assert not np.asarray(fresh.info['reward_memory'].released)[~fresh_post].any()
        assert not np.asarray(fresh.info['reward_memory'].target_count)[~fresh_post].any()
        assert np.asarray(fresh.info['reward_memory'].qualified_release)[fresh_post].all()
    checks['passed'] = (all(checks[k] == 0 for k in ('nonfinite', 'capacity_overflow', 'controller_failed'))
        and checks['all_finite'] and checks['max_speed'] <= 1.+1e-5
        and checks['max_acceleration'] <= 5.+1e-3 and checks['max_jerk'] <= 100.+.2
        and checks['max_gripper_reference_step'] <= env._config.gripper_rate+1e-6
        and checks['reward_memory_replay_error'] < 1e-5 and checks['max_reward_replay_error'] < 1e-4)
    result = dict(status='passed' if checks['passed'] else 'failed', environments=n,
        steps=args.steps, policy_transitions=n*args.steps, reset_group_counts=counts,
        invalid_held_resets=reset_contacts, actor=49, critic=66, action=7,
        reward_model=env.reward_model.__name__,
        checks=checks, elapsed_seconds=time.monotonic()-started, gpu=gpu.report(),
        reset_bank=env.reset_bank, reward_settings=asdict(env.reward_settings),
        initial_position_diversity=diversity, bank_split=args.bank_split,
        perception_randomization=perception_summary,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        scope='Actual GPU-Warp physics with dynamics DR, fixed actions; not policy training or success evaluation',
        training=False, real_robot=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('gpu', 'reset_bank')}, indent=2), flush=True)
    if not checks['passed']:
        raise RuntimeError('GPU physics acceptance failed; inspect output before training')


if __name__ == '__main__':
    main()
