"""Audit saved pre-grasp trajectories and reproduce reward-v1 incentive gaps.

This reads existing evaluation traces. Synthetic comparisons isolate score
inputs; they do not claim the compared states are feasible robot trajectories.
No training, reward changes, model promotion, or robot I/O.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def trajectory_report(directory):
    samples = []
    for path in sorted(directory.glob('pre_grasp-s*.npz')):
        with np.load(path) as archive:
            names = ('active', 'failed', 'bowl', 'qvel', 'metric_stage_section',
                'features', 'metric_stage_main', 'measured_5ms',
                'metric_stage_cube_push_cost', 'metric_stage_recontact_cost',
                'metric_qualified_release')
            # NPZ entries decompress on access: load once outside the world loop.
            d = {name: archive[name] for name in names}
            valid = d['active'].astype(bool) & ~d['failed'].astype(bool)
            bowl = d['bowl']
            drift = np.linalg.norm(bowl[..., :2] - bowl[:1, :, :2], axis=-1)
            speed = np.linalg.norm(d['qvel'][..., 14:17], axis=-1)
            section = d['metric_stage_section']
            far_high = valid & (section == 3) & (d['features'][..., 2] > .05) & (d['metric_stage_main'] >= 3.49)
            for world in range(valid.shape[1]):
                mask = valid[:, world]
                if not np.any(mask):
                    raise ValueError(f'No valid interval: {path}:{world}')
                indices = np.flatnonzero(mask & (drift[:, world] > .005))
                samples.append(dict(seed=int(path.stem.split('-s')[-1]), world=world,
                    grasp_lift=bool(np.any(mask & (section[:, world] >= 2))),
                    max_bowl_drift_mm=float(1000*drift[mask, world].max()),
                    max_bowl_speed_m_s=float(speed[mask, world].max()),
                    first_5mm_drift_section=int(section[indices[0], world]) if len(indices) else None,
                    robot_bowl_contact=bool(np.any((d['measured_5ms'][:, world, :, 18] > 0) & mask[:, None])),
                    held_cube_bowl_floor_contact=bool(np.any(
                        (d['measured_5ms'][:, world, :, 15] > 0) &
                        ((d['measured_5ms'][:, world, :, 13] > 0) | (d['measured_5ms'][:, world, :, 14] > 0)) & mask[:, None])),
                    far_high_seconds=float(.1*far_high[:, world].sum()),
                    push_cost=float(d['metric_stage_cube_push_cost'][mask, world].sum()),
                    recontact_cost=float(d['metric_stage_recontact_cost'][mask, world].sum()),
                    qualified_release=bool(np.any(mask & (d['metric_qualified_release'][:, world] > 0))),
                    failure=bool(np.any(d['failed'][:, world]))))
    if len(samples) != 384:
        raise ValueError(f'Expected 384 pre-grasp episodes, got {len(samples)}')
    drift = np.array([s['max_bowl_drift_mm'] for s in samples])
    moved = [s for s in samples if s['max_bowl_drift_mm'] > 5]
    return dict(episodes=len(samples), grasp_lift=sum(s['grasp_lift'] for s in samples),
        bowl_moved_over_5mm=len(moved), bowl_moved_over_10mm=int((drift > 10).sum()),
        bowl_moved_over_20mm=int((drift > 20).sum()),
        bowl_drift_mm_p50_p90_max=np.percentile(drift, [50, 90, 100]).tolist(),
        robot_bowl_contact=sum(s['robot_bowl_contact'] for s in samples),
        held_cube_bowl_floor_contact=sum(s['held_cube_bowl_floor_contact'] for s in samples),
        first_5mm_drift_sections={str(i): sum(s['first_5mm_drift_section'] == i for s in moved) for i in range(-1, 5)},
        far_high_reward_episodes=sum(s['far_high_seconds'] > 0 for s in samples),
        far_high_seconds_mean=float(np.mean([s['far_high_seconds'] for s in samples])),
        moved_bowl_with_zero_recontact_cost=sum(s['recontact_cost'] == 0 for s in moved),
        qualified_release=sum(s['qualified_release'] for s in samples), failures=sum(s['failure'] for s in samples),
        first_recorded_case=samples[0])


def formula_report():
    import jax.numpy as jp
    from piper_sim_lab.stage_reward import Features, advance_memory, reset_memory, reward_terms

    zero = jp.zeros(7)
    held = Features(tcp_cube_distance=.005, cube_bottom_table=.05, cube_bowl_xy=.15,
        cube_center_above_rim=.02, tcp_floor_height=.12, left_contact=True,
        right_contact=True, cube_table_contact=False, robot_cube_contact=True,
        tcp_relative_speed=.02)
    memory = reset_memory()._replace(grasp_time=jp.array(.1), lifted=jp.array(True))

    def price(m, f):
        updated = advance_memory(m, f, f)
        result = reward_terms(m, updated, f, zero, zero)
        return {k: float(result[k]) for k in ('main', 'reward', 'cube_push_cost', 'recontact_cost', 'xy_alignment')}

    pairs = dict(
        held_robot_bowl_contact=dict(clean=price(memory, held),
            touching=price(memory, held._replace(robot_bowl_contact=True))),
        carry_below_rim=dict(above=price(memory, held),
            below=price(memory, held._replace(cube_center_above_rim=-.03, cube_bottom_table=.005))),
        moving_after_confirmed_grasp=dict(slow=price(memory, held._replace(cube_horizontal_speed=0.)),
            fast=price(memory, held._replace(cube_horizontal_speed=.3))),
        historical_bowl_reached=dict(
            aligned=price(memory._replace(bowl_reached=jp.array(True)), held._replace(cube_bowl_xy=.005, tcp_floor_height=.055)),
            far=price(memory._replace(bowl_reached=jp.array(True)), held._replace(cube_bowl_xy=.4, tcp_floor_height=.055))))
    checks = dict(
        held_bowl_contact_has_no_cost=pairs['held_robot_bowl_contact']['clean']['reward'] == pairs['held_robot_bowl_contact']['touching']['reward'],
        low_carry_has_same_score=pairs['carry_below_rim']['above']['reward'] == pairs['carry_below_rim']['below']['reward'],
        postgrasp_speed_has_no_push_cost=pairs['moving_after_confirmed_grasp']['fast']['cube_push_cost'] == 0.,
        far_after_reaching_keeps_3_point_5=abs(pairs['historical_bowl_reached']['far']['main'] - 3.5) < 1e-6)
    if not all(checks.values()):
        raise AssertionError(checks)
    return dict(scope='Isolated synthetic score-input comparisons, not physical rollout predictions', pairs=pairs,
                reproduced_gaps=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(campaign=str(args.campaign.resolve()),
        reward_source_sha256=hashlib.sha256(Path('src/piper_sim_lab/stage_reward.py').read_bytes()).hexdigest(),
        measurement_notes=[
            'Bowl displacement baseline is first recorded policy sample (t=0.1s), not the unrecorded reset pose',
            'Retain achievements before failure; exclude failing and subsequent intervals',
            'Robot-bowl contact is measured by 5ms sensors; cube-bowl rim contact has no dedicated recorded sensor',
            'pre_grasp is a reset group; its later trajectory can reach transport and release-preparation sections'],
        checkpoints={str(u): trajectory_report(args.campaign/'evaluation'/f'{u}u') for u in (2, 4, 6, 8)},
        formula=formula_report(),
        proposed_changes_not_implemented=[
            'Keep grasp/lift achievement separate from subsequent trajectory quality metrics',
            'Measure held-cube/bowl rim contact as well as robot/bowl contact',
            'Charge unintended bowl disturbance during grasp/carry; allow normal settling motion after release',
            'Retain current XY progress in release preparation so old bowl-reached history cannot maintain far-away credit',
            'Distinguish carrying below the rim outside the opening from safely lowering inside the opening',
            'Validate normal transport and release against low-carry impact, bowl pushing and far-away 3.5-score histories'],
        next_training_user_decision=dict(parent_checkpoint=str((args.campaign/'training/checkpoints/000033554432').resolve()),
            parent_units=4, reset_probabilities=[.2]*5,
            fifth_reset='post-release settling, retreat and return to A',
            status='reward review first; five-reset implementation and validation pending; no new training started'),
        reward_changed=False, training_started=False, real_robot=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
