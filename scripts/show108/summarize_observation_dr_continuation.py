"""Compare a v5 observation-DR parent and two continuation checkpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from piper_sim_lab.actor_comparison import STARTS, load_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--parent-units', type=int, default=16)
    parser.add_argument('--parent-checkpoint-units', type=int, default=2)
    parser.add_argument('--original-parent-run', type=Path,
                        help='Root containing the original-condition parent evaluation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent_units = args.parent_units
    budgets = (parent_units, parent_units+2, parent_units+4)
    stem = f'continuation-{budgets[-1]}u'
    campaign = json.loads((args.run/'campaign.json').read_text())
    if campaign['status'] != 'completed':
        raise ValueError('Training, checkpoint evaluation and rendering must finish first')
    config = json.loads((args.run/'training/config.json').read_text())
    if not config['resolved_environment']['obs_randomization']:
        raise ValueError('Expected the observation-DR continuation configuration')
    continued, samples = {}, {}
    continued[parent_units], samples[parent_units] = load_evaluation(
        args.parent_run, args.parent_checkpoint_units)
    for additional in (2, 4):
        units = parent_units+additional
        continued[units], samples[units] = load_evaluation(args.run, additional)
        for start in STARTS:
            if samples[units][start]['bank'] != samples[parent_units][start]['bank']:
                raise ValueError('Parent and child evaluation banks differ')
            np.testing.assert_array_equal(samples[units][start]['case_ids'],
                                          samples[parent_units][start]['case_ids'])
    original, original_samples = {}, {}
    original_parent = args.original_parent_run or args.run/'original-conditions'
    original[parent_units], original_samples[parent_units] = load_evaluation(
        original_parent, parent_units)
    for units in budgets[1:]:
        original[units], original_samples[units] = load_evaluation(
            args.run/'original-conditions', units)
        for start in STARTS:
            if original_samples[units][start]['bank'] != original_samples[parent_units][start]['bank']:
                raise ValueError('Original-condition evaluation banks differ')
            np.testing.assert_array_equal(original_samples[units][start]['case_ids'],
                                          original_samples[parent_units][start]['case_ids'])
    rows = []
    conditions = dict(observation_dr_wide_spawn=continued,
                      original_observations_narrow_spawn=original)
    for condition, checkpoints in conditions.items():
        for units, stages in checkpoints.items():
            for start, metrics in stages.items():
                rows.append(dict(condition=condition, cumulative_units=units,
                                 additional_units=units-parent_units, start=start, **metrics))
    checkpoint_records = []
    for additional in (2, 4):
        checkpoint = args.run/'training/checkpoints'/f'{additional*8388608:012d}'
        metadata = json.loads((checkpoint/'metadata.json').read_text())
        if metadata['cumulative_transitions'] != (parent_units+additional)*8388608:
            raise ValueError('Cumulative checkpoint metadata differs from the budget')
        checkpoint_records.append(dict(additional_units=additional,
            cumulative_units=parent_units+additional, path=str(checkpoint.resolve()),
            learner_sha256=hashlib.sha256((checkpoint/'learner.msgpack').read_bytes()).hexdigest(),
            metadata=metadata))
    gpu = json.loads((args.run/'training/gpu.json').read_text())
    gpu_samples = gpu.pop('samples')
    gpu.update(sample_count=len(gpu_samples),
        median_board_memory_mib=float(np.median([s['board_memory_mib'] for s in gpu_samples])),
        median_utilization_percent=float(np.median([s['utilization_percent'] for s in gpu_samples])))
    result = dict(status='completed', reward='stage_reward_v5_joint_A',
        source_commit=campaign['git_commit'], conditions=conditions,
        rows=rows, checkpoints=checkpoint_records,
        training=json.loads((args.run/'training/summary.json').read_text()),
        gpu=gpu,
        evaluation_seeds=[2026, 2027, 2028], episodes_per_start=384,
        success='Episode achievement before failure, full A excludes high-drop exception',
        original_condition_normalizer='Each checkpoint uses its own saved normalizer',
        training_seeds=1, real_robot=False)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/f'{stem}-results.json').write_text(
        json.dumps(result, indent=2, allow_nan=False)+'\n')
    with (args.output/f'{stem}-results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = f'| 지표 | 부모{budgets[0]}배 | 누적{budgets[1]}배 | 누적{budgets[2]}배 |'
    report = [f'# 관측 DR MLP+LSTM{budgets[0]}→{budgets[2]}배 결과', '',
        f'관측 DR·확장 스폰의 부모{parent_units}배에서 추가2/4배가 누적{budgets[1]}/{budgets[2]}배다.', '',
        '각 초기화384회,seed2026/2027/2028,최대20초다. 각 조건 안에서 bank와case ID를 대조했다.', '',
        '## 같은 관측 DR·확장 스폰 조건', '',
        header,
        '| --- | ---: | ---: | ---: |']
    labels = ['① 접근', '② 파지·들기', '③ 운반', '④ 개방·놓기', '⑤ 안착·후퇴·관절 A']
    for label, start in zip(labels, STARTS):
        values = [continued[u][start] for u in budgets]
        cells = [f"{v['local_success_rate']*100:.1f}% ({v['local_success']}/384)" for v in values]
        report.append('| '+label+' | '+' | '.join(cells)+' |')
    for condition, checkpoints in conditions.items():
        heading = ('새 관측 DR·확장 스폰' if condition == 'observation_dr_wide_spawn'
                   else '기존 관측·좁은 스폰')
        report += ['', '## '+heading+' 전체 완료와 A 오차', '',
            header, '| --- | ---: | ---: | ---: |']
        for label, start in zip(labels, STARTS):
            cells = [f"{checkpoints[u][start]['full_home_success_rate']*100:.1f}%"
                     f" ({checkpoints[u][start]['full_home_complete']}/384)" for u in budgets]
            report.append('| '+label+' 시작 실제 A 전체 | '+' | '.join(cells)+' |')
        errors = [f"{checkpoints[u]['post_release']['final_max_A_error_deg_median']:.2f}°"
                  for u in budgets]
        report.append('| ⑤ 마지막 최대 관절 A 오차 중앙값 | '+' | '.join(errors)+' |')
    report += ['', '## 해석 범위', '',
        '각 조건의 관측·스폰·평가 bank가 다르므로 조건 사이의 차이를 한 요소의 효과로 분리할 수 없다.',
        '성공은 달성 이력이고 마지막 오차는 종료 상태 지표다. 학습seed는1개다.',
        '단계 내④ 성공에는 높은낙하 종료 예외가 포함될 수 있으며 실제 A 전체 집계에서는 제외했다.', '',
        f'[전체 JSON]({stem}-results.json) · [CSV]({stem}-results.csv) · [학습 조건]({stem}.md)']
    (args.output/f'{stem}-results.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({str(u): {s: continued[u][s]['local_success_rate']
                              for s in STARTS} for u in budgets}, indent=2))


if __name__ == '__main__':
    main()
