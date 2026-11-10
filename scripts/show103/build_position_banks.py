"""Build independent, contact-validated training/evaluation position reset banks."""
import argparse
import hashlib
import json
from pathlib import Path

from piper_sim_lab.runtime import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--train-seed', type=int, default=41030)
    parser.add_argument('--eval-seed', type=int, default=51030)
    parser.add_argument('--config', type=Path, default=Path('configs/show103/stage-reward-v1-4096-2-4-6-8.yaml'))
    parser.add_argument('--output-prefix', default='show103-position-random')
    parser.add_argument('--x-range', nargs=2, type=float, default=[.25, .35])
    parser.add_argument('--y-range', nargs=2, type=float, default=[-.20, .20])
    parser.add_argument('--max-radius', type=float, default=.4)
    parser.add_argument('--attempts-per-pose', type=int, default=80)
    args = parser.parse_args()
    if args.count < 4 or args.train_seed == args.eval_seed:
        parser.error('at least four poses and independent train/eval seeds required')
    if (Path(args.output_prefix).name != args.output_prefix or
        not args.x_range[0] < args.x_range[1] or not args.y_range[0] < args.y_range[1] or
        args.max_radius <= 0 or args.attempts_per_pose < 1):
        parser.error('require a filename prefix, ordered XY ranges and positive radius/attempt limit')
    configure('cpu')
    import mujoco
    import numpy as np
    import yaml
    from scipy.spatial.transform import Rotation
    from piper_sim_lab.tasks import make_env
    from piper_sim_lab.position_random_stage import ASSET_DIR, geometry_fingerprint

    config = yaml.safe_load(args.config.read_text())
    env = make_env(dict(config['environment'], impl='jax', batch_capacity=1,
        task='stage_reward_pick_place', position_reset_bank='', evaluation_position_reset_bank='',
        reset_stage_probs=[.25]*4, obs_randomization=False))
    c, model = env._config, env.mj_model
    data = mujoco.MjData(model)
    lower, upper = np.asarray(env._lower), np.asarray(env._upper)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    prototypes = np.asarray(env.reset_bank['ctrl'])[:, env._arm_ctrl_ids]
    goals = []
    for q in prototypes:
        mujoco.mj_resetData(model, data)
        data.qpos[env._qpos_ids] = q
        mujoco.mj_forward(model, data)
        goals.append(data.site_xmat[env._tcp_id].reshape(3, 3).copy())
    pivot = data.xanchor[model.joint('joint1').id, :2].copy().tolist()

    def solve(xy, height, stage):
        q = prototypes[stage].copy()
        target = np.r_[xy, height]
        goal = goals[stage]
        for _ in range(220):
            data.qpos[:] = model.qpos0
            data.qpos[env._qpos_ids] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            rotation = data.site_xmat[env._tcp_id].reshape(3, 3)
            pos_error = target-data.site_xpos[env._tcp_id]
            rot_error = .5*sum(np.cross(rotation[:, i], goal[:, i]) for i in range(3))
            if np.linalg.norm(pos_error) < 1e-4 and np.linalg.norm(rot_error) < 1e-3:
                if np.linalg.norm(Rotation.from_matrix(goal@rotation.T).as_rotvec()) < 1e-3:
                    return q
            mujoco.mj_jacSite(model, data, jacp, jacr, env._tcp_id)
            jac = np.r_[jacp, jacr][:, env._qvel_ids]
            error = np.r_[pos_error, rot_error]
            q = np.clip(q+.5*jac.T@np.linalg.solve(jac@jac.T+1e-4*np.eye(6), error), lower, upper)
        return None

    physical_names = ('robot_model', 'base_height', 'tcp_local_z', 'block_half', 'block_mass',
        'bowl_r_bottom', 'bowl_r_rim', 'bowl_height', 'bowl_floor', 'bowl_wall', 'bowl_mass',
        'gripper_max', 'gripper_force', 'gripper_kp', 'finger_bridge')
    physical_config = {k: c[k] for k in physical_names}
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for split, seed in [('train', args.train_seed), ('eval', args.eval_seed)]:
        path = ASSET_DIR/f'{args.output_prefix}-{split}-v1.npz'
        if path.exists() or path.with_suffix('.json').exists():
            raise FileExistsError(path)
        rng = np.random.default_rng(seed)
        positions, controls, records = [], [], []
        for stage in range(4):
            qs, us, cube_xy, bowl_xy = [], [], [], []
            attempts = 0
            while len(qs) < args.count:
                attempts += 1
                if attempts > args.count*args.attempts_per_pose:
                    raise RuntimeError(f'{split} stage {stage}: insufficient valid IK/contact poses')
                target_xy = rng.uniform([args.x_range[0], args.y_range[0]], [args.x_range[1], args.y_range[1]])
                if np.linalg.norm(target_xy) > args.max_radius:
                    continue
                q = prototypes[0] if stage == 0 else solve(target_xy, [0., .065, .10, .125][stage], stage)
                if q is None:
                    continue
                mujoco.mj_resetData(model, data)
                data.qpos[env._qpos_ids] = q
                held = stage >= 2
                data.qpos[env._finger_qpos_ids] = [c.block_half, -c.block_half] if held else [c.gripper_max, -c.gripper_max]
                mujoco.mj_kinematics(model, data)
                tcp = data.site_xpos[env._tcp_id].copy()
                axis = data.site_xmat[env._tcp_id].reshape(3, 3)[:, env._closing_axis]
                yaw = np.arctan2(axis[1], axis[0]) if stage > 0 else 0.
                cube = tcp if held else np.r_[target_xy, c.block_half]
                data.qpos[env._block_qpos_adr:env._block_qpos_adr+7] = np.r_[cube, np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
                if stage == 3:
                    bowl = cube[:2]
                else:
                    for _ in range(1000):
                        bowl = rng.uniform([args.x_range[0], args.y_range[0]], [args.x_range[1], args.y_range[1]])
                        if np.linalg.norm(bowl) <= args.max_radius and np.linalg.norm(bowl-cube[:2]) >= c.spawn_min_distance:
                            break
                    else:
                        raise RuntimeError('could not sample non-overlapping bowl')
                data.qpos[env._bowl_qpos_adr:env._bowl_qpos_adr+7] = np.r_[bowl, 0., 1., 0., 0., 0.]
                data.ctrl[env._arm_ctrl_ids] = q
                data.ctrl[env._gripper_ctrl_id] = 0. if held else c.gripper_max
                for _ in range(60):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                touch = env._native_contacts(data)
                robot_bowl = any(touch['bowl'+str(i)] for i in range(1, 9))
                if robot_bowl or not np.isfinite(data.qpos).all():
                    continue
                measured_cube = data.xpos[env._block_id, :2]
                measured_bowl = data.xpos[env._bowl_id, :2]
                if any(np.any(xy < [args.x_range[0], args.y_range[0]]) or
                       np.any(xy > [args.x_range[1], args.y_range[1]]) for xy in (measured_cube, measured_bowl)):
                    continue
                if held:
                    cube_bottom = data.xpos[env._block_id, 2]-c.block_half*np.abs(data.xmat[env._block_id].reshape(3, 3)[2]).sum()
                    if not (touch['left'] and touch['right']) or touch['table'] or cube_bottom <= .02:
                        continue
                elif touch['left'] or touch['right']:
                    continue
                qs.append(data.qpos.copy()); us.append(data.ctrl.copy())
                cube_xy.append(data.xpos[env._block_id, :2].copy())
                bowl_xy.append(data.xpos[env._bowl_id, :2].copy())
            positions.append(qs); controls.append(us)
            records.append(dict(stage=stage, count=len(qs), attempts=attempts,
                cube_xy_min=np.min(cube_xy, axis=0).tolist(), cube_xy_max=np.max(cube_xy, axis=0).tolist(),
                bowl_xy_min=np.min(bowl_xy, axis=0).tolist(), bowl_xy_max=np.max(bowl_xy, axis=0).tolist(),
                bilateral_held_contacts_verified=bool(stage >= 2)))
            print(split, records[-1], flush=True)
        np.savez_compressed(path, qpos=np.asarray(positions), ctrl=np.asarray(controls))
        metadata = dict(version='show103-position-random-v1', split=split, seed=seed,
            poses_per_stage=args.count, native_stabilization_seconds=.3, records=records,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), geometry_sha256=geometry_fingerprint(model),
            physical_config=physical_config, joint1_pivot_world_xy=pivot,
            sampling='Coupled native-validated arm/cube pose bank plus continuous yaw; independent continuous empty cube/bowl draws; over-bowl is coupled',
            requested_xy_x=args.x_range, requested_xy_y=args.y_range, max_radius=args.max_radius,
            minimum_free_cube_bowl_distance=c.spawn_min_distance, reward_history_prefilled=False,
            source_config=str(args.config), source_config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())
        path.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(path, flush=True)


if __name__ == '__main__':
    main()
