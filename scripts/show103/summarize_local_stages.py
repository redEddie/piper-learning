"""Summarize local stage achievements from saved, first-episode physical traces."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


STARTS = ('initial_a', 'pre_grasp', 'held', 'over_bowl')
STAGES = ('approach', 'grasp_lift', 'transport', 'qualified_release')


def inspect(path):
    with np.load(path) as d:
        # Exclude the failing control interval and everything after it. A later
        # failure must not erase an achievement in an earlier valid interval.
        valid = d['active'].astype(bool) & ~d['failed'].astype(bool)
        f = d['measured_5ms'].transpose(0, 2, 1, 3).reshape(-1, valid.shape[1], 20)
        tick_valid = np.repeat(valid, 20, axis=0)
        near = (f[..., 0] <= .02) | (f[..., 13] > 0) | (f[..., 14] > 0)
        both = (f[..., 13] > 0) & (f[..., 14] > 0)
        index = np.arange(len(f))[:, None]
        last_not_both = np.maximum.accumulate(np.where(both, -1, index), axis=0)
        lifted_now = ((index - last_not_both) >= 20) & (f[..., 1] >= .02)
        lifted = np.maximum.accumulate(lifted_now, axis=0)
        bowl = lifted & ((f[..., 13] > 0) | (f[..., 14] > 0)) & (f[..., 2] <= .05) & (f[..., 3] >= 0)
        events = {
            'approach': np.any(near & tick_valid, axis=0),
            'grasp_lift': np.any(lifted_now & tick_valid, axis=0),
            'transport': np.any(bowl & tick_valid, axis=0),
            'qualified_release': np.any((d['metric_qualified_release'] > 0) & valid, axis=0),
            'settled': np.any((d['metric_settled'] > 0) & valid, axis=0),
            'complete': np.any((d['metric_success'] > 0) & valid, axis=0),
        }
        # Independently compare 5ms measurements with the stored reward section.
        for name, threshold in [('grasp_lift', 2), ('transport', 3)]:
            recorded = np.any((d['metric_stage_section'] >= threshold) & valid, axis=0)
            if not np.array_equal(events[name], recorded):
                raise ValueError(f'{path}: measured {name} disagrees with reward section')
        main = d['metric_stage_main']
        peak = np.max(np.where(valid, main, -np.inf), axis=0)
        release_ready = (d['metric_stage_dwell_bonus'] > 0) & valid
        opened = (d['action'][..., 6] >= .5) & valid
        # stage_section includes sticky bowl-reached history, as does the reward.
        opened_after_bowl = opened & (d['metric_stage_section'] >= 3)
        return dict(episodes=valid.shape[1], events=events, peak=peak,
                    release_ready=np.any(release_ready, axis=0),
                    opened_after_bowl=np.any(opened_after_bowl, axis=0),
                    failures=int(np.any(d['failed'], axis=0).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for unit in (2, 4, 6, 8):
        for start, stage in zip(STARTS, STAGES):
            parts = [inspect(p) for p in sorted((args.campaign/'evaluation'/f'{unit}u').glob(f'{start}-s*.npz'))]
            if len(parts) != 3:
                raise ValueError(f'Expected three seed traces: {unit}u/{start}')
            episodes = sum(p['episodes'] for p in parts)
            success = sum(int(p['events'][stage].sum()) for p in parts)
            peaks = np.concatenate([p['peak'] for p in parts])
            rows.append(dict(checkpoint_units=unit, start=start, local_stage=stage,
                episodes=episodes, local_success=success, local_success_rate=success/episodes,
                qualified_release=sum(int(p['events']['qualified_release'].sum()) for p in parts),
                settled=sum(int(p['events']['settled'].sum()) for p in parts),
                complete=sum(int(p['events']['complete'].sum()) for p in parts),
                release_ready_dwell=sum(int(p['release_ready'].sum()) for p in parts),
                opened_after_bowl=sum(int(p['opened_after_bowl'].sum()) for p in parts),
                mean_episode_peak_main=float(peaks.mean()), median_episode_peak_main=float(np.median(peaks)),
                failures=sum(p['failures'] for p in parts)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(campaign=str(args.campaign.resolve()), seeds=[2026, 2027, 2028],
        duration_seconds=20, physics_dt_seconds=.005,
        failure_handling='exclude failing interval and subsequent intervals; retain earlier achievements',
        definitions=dict(approach='TCP-cube distance <=20mm or finger contact',
            grasp_lift='bilateral contact >=100ms and cube bottom >=20mm above table',
            transport='after grasp/lift, finger contact, bowl XY <=50mm and cube center above rim',
            qualified_release='OPEN intention; first detachment at TCP height 50-60mm and XY <=17mm; no finger contact >=20ms'),
        stage_5=dict(name='post_release_settle_retreat_home', local_success_rate=None,
            reason='No post-release reset was evaluated; no qualified releases occurred in the existing four starts',
            qualified_release_episodes=sum(r['qualified_release'] for r in rows)), rows=rows)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
