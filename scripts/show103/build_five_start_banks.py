"""Append actual low-release prefixes to the preserved four-start reset banks."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from piper_sim_lab.runtime import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--splits', nargs='+', choices=['train', 'eval'], default=['train', 'eval'])
    parser.add_argument('--config', type=Path, default=Path('configs/show103/stage-reward-v1-4096-random-positions-2-4-6-8.yaml'))
    parser.add_argument('--source-prefix', default='show103-position-random')
    parser.add_argument('--output-prefix', default='show103-five-start')
    args = parser.parse_args()
    if args.count < 4 or any(Path(s).name != s for s in (args.source_prefix, args.output_prefix)):
        parser.error('require at least four poses and filename prefixes')
    configure('cpu')
    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    import yaml
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from piper_sim_lab.tasks import make_env
    from piper_sim_lab.position_random_stage import ASSET_DIR
    from piper_sim_lab.five_start_stage import native_features
    from piper_sim_lab import stage_reward_v2 as reward

    cfg = yaml.safe_load(args.config.read_text())
    # Warp contact sensors are part of this native C model as in GPU execution.
    # This script advances C MuJoCo only; no policy or GPU rollout is executed.
    env = make_env(dict(cfg['environment'], impl='warp', batch_capacity=1,
        task='stage_reward_pick_place_random_positions', reset_stage_probs=[.25]*4,
        position_reset_bank=f'{args.source_prefix}-train-v1.npz', obs_randomization=False))
    # Prefix generation uses the original 21 physical measurements/reward-v2
    # replay, irrespective of the eventual learner's reward version.
    env._dense_home = False
    model, c = env.mj_model, env._config
    lower, upper = np.asarray(env._lower), np.asarray(env._upper)
    aid = np.asarray(env._qpos_ids)
    control = np.asarray(env._arm_ctrl_ids)
    cset = reward.RewardSettings()

    @jax.jit
    def replay(trace, length):
        def tick(carry, pair):
            memory, before = carry
            row, i = pair
            f = reward.Features(*(row[k].astype(jp.bool_) if 13 <= k < 20 else row[k] for k in range(21)))
            new = reward.advance_memory(memory, before, f, cset)
            memory = jax.tree.map(lambda a, b: jp.where(i < length, a, b), new, memory)
            return (memory, f), None
        before = reward.Features(*(trace[0, k].astype(jp.bool_) if 13 <= k < 20 else trace[0, k] for k in range(21)))
        return jax.lax.scan(tick, (reward.reset_memory(), before), (trace[1:], jp.arange(1, len(trace))))[0][0]

    def prefix(qpos, ctrl):
        data, ik = mujoco.MjData(model), mujoco.MjData(model)
        data.qpos[:] = qpos
        data.ctrl[:] = ctrl
        mujoco.mj_forward(model, data)
        target = data.site_xpos[env._tcp_id].copy()
        target[2] = data.xpos[env._bowl_id, 2]+c.bowl_floor+.055
        orientation = data.site_xmat[env._tcp_id].reshape(3, 3).copy()
        def residual(q):
            ik.qpos[:] = data.qpos
            ik.qpos[aid] = q
            mujoco.mj_kinematics(model, ik)
            return np.r_[5*(ik.site_xpos[env._tcp_id]-target),
                Rotation.from_matrix(orientation@ik.site_xmat[env._tcp_id].reshape(3, 3).T).as_rotvec()]
        result = least_squares(residual, np.clip(ctrl[control], lower+1e-6, upper-1e-6),
            bounds=(lower, upper), max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if np.linalg.norm(residual(result.x)) > .001:
            return None, 'low IK unreachable'
        initial, destination = ctrl[control].copy(), result.x
        delta = destination-initial
        # Quintic endpoints have zero velocity/acceleration; choose duration
        # from exact polynomial extrema and the existing v/a/jerk limits.
        distance = np.max(np.abs(delta))
        duration = max(1.5, 1.875*distance, np.sqrt(5.774*distance/5), (60*distance/100)**(1/3))
        steps = int(np.ceil(duration/.005))
        trace = [native_features(env, data, 0.)]
        def advance(q, reference, goal):
            data.ctrl[control] = q
            data.ctrl[env._gripper_ctrl_id] = reference
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            f = native_features(env, data, goal)
            trace.append(f)
            return f
        for _ in range(25):
            advance(initial, 0., 0.)
        for i in range(1, steps+1):
            u = i/steps
            blend = 10*u**3-15*u**4+6*u**5
            advance(initial+delta*blend, 0., 0.)
        for _ in range(60):
            f = advance(destination, 0., 0.)
        if not (f.left_contact and f.right_contact and .05 <= f.tcp_floor_height <= .06 and f.cube_bowl_xy <= .017):
            return None, f'low hold not valid: height={f.tcp_floor_height:.4f}, xy={f.cube_bowl_xy:.4f}'
        lost, first_height, first_xy = 0, None, None
        for i in range(1, 161):
            before = trace[-1]
            f = advance(destination, min(c.gripper_max, i*c.gripper_rate/20), c.gripper_max)
            if not (f.left_contact or f.right_contact):
                if lost == 0:
                    first_height, first_xy = before.tcp_floor_height, before.cube_bowl_xy
                lost += 1
            else:
                lost = 0
            if lost >= 4:
                break
        else:
            return None, 'no release'
        if not (.05 <= first_height <= .06 and first_xy <= .017):
            return None, f'first-loss geometry: height={first_height:.4f}, xy={first_xy:.4f}'
        return dict(qpos=data.qpos.copy(), ctrl=data.ctrl.copy(), qvel=data.qvel.copy(),
            features=np.array(trace, dtype=np.float32), duration=duration,
            release_height=first_height, release_xy=first_xy), None

    for split in args.splits:
        parent = ASSET_DIR/f'{args.source_prefix}-{split}-v1.npz'
        metadata = json.loads(parent.with_suffix('.json').read_text())
        destination = ASSET_DIR/f'{args.output_prefix}-{split}-v2.npz'
        if destination.exists() or destination.with_suffix('.json').exists():
            raise FileExistsError(destination)
        with np.load(parent) as bank:
            qpos, ctrl = bank['qpos'].copy(), bank['ctrl'].copy()
        if args.count > qpos.shape[1]:
            raise ValueError('requested more release poses than the source bank contains')
        records, rejected = [], []
        for index in range(len(qpos[3])):
            item, reason = prefix(qpos[3, index], ctrl[3, index])
            if item is None:
                rejected.append(dict(index=index, reason=reason))
            else:
                xy = item['qpos'][env._block_qpos_adr:env._block_qpos_adr+2]
                if np.any(xy < [metadata['requested_xy_x'][0], metadata['requested_xy_y'][0]]) or np.any(xy > [metadata['requested_xy_x'][1], metadata['requested_xy_y'][1]]):
                    rejected.append(dict(index=index, reason='released cube outside requested XY rectangle'))
                    continue
                item['parent_index'] = index
                records.append(item)
            print(split, index, 'valid' if item else reason, flush=True)
            if len(records) >= args.count:
                break
        if len(records) < args.count:
            raise RuntimeError(f'{split}: only {len(records)} valid releases; rejected={rejected}')
        # Preserve the exact four original banks. Allow a smaller fifth sample
        # count only for an explicit diagnostic build, never silently duplicate.
        if args.count != qpos.shape[1]:
            qpos, ctrl = qpos[:, :args.count], ctrl[:, :args.count]
        lengths = np.array([len(r['features']) for r in records])
        traces = np.stack([np.pad(r['features'], ((0, lengths.max()-len(r['features'])), (0, 0)), mode='edge') for r in records])
        memories = jax.block_until_ready(jax.vmap(replay)(jp.asarray(traces), jp.asarray(lengths)))
        if not np.all(memories.qualified_release) or np.any(memories.invalid_release) or np.any(memories.outside_drop):
            raise RuntimeError('Physical prefix failed reward-history replay')
        payload = dict(qpos=np.concatenate([qpos, np.array([r['qpos'] for r in records])[None]]),
            ctrl=np.concatenate([ctrl, np.array([r['ctrl'] for r in records])[None]]),
            release_qvel=np.array([r['qvel'] for r in records]), prefix_features=traces, prefix_steps=lengths,
            **{'memory_'+name: np.asarray(value) for name, value in zip(reward.Memory._fields, memories)})
        np.savez_compressed(destination, **payload)
        records_summary=[]
        for stage, positions in enumerate(payload['qpos']):
            cube=positions[:,env._block_qpos_adr:env._block_qpos_adr+2]
            bowl=positions[:,env._bowl_qpos_adr:env._bowl_qpos_adr+2]
            records_summary.append(dict(stage=stage,count=len(positions),
                cube_xy_min=cube.min(0).tolist(),cube_xy_max=cube.max(0).tolist(),
                bowl_xy_min=bowl.min(0).tolist(),bowl_xy_max=bowl.max(0).tolist(),
                bilateral_held_contacts_verified=stage in (2,3)))
        metadata.update(version='show103-five-start-v2', poses_per_stage=args.count,
            records=records_summary,
            parent_bank_sha256=metadata['sha256'], sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            prefix_settings=asdict(cset), reward_history_prefilled=False,
            reward_history_source='Replay of recorded 5ms native grasp/lower/OPEN/release prefix; no fabricated timers',
            fifth_start='20ms after first valid loss of both finger contacts; physical velocity preserved',
            release_records=[{k: v for k, v in r.items() if k not in ('qpos', 'qvel', 'ctrl', 'features')} for r in records],
            rejected=rejected)
        destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(destination, flush=True)


if __name__ == '__main__':
    main()
