"""Checkpoints, stage denominators and the all-starts promotion criterion."""
from copy import deepcopy
from pathlib import Path
import runpy

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
campaign = runpy.run_path(str(ROOT/'scripts/show103/run_stage_campaign.py'))


def evaluations():
    result = {}
    for units in (2, 4, 6, 8):
        rows = []
        for start in campaign['STARTS']:
            count = 64 if units == 2 else 80
            if units == 4:
                count = 60 if start == 'initial_a' else 128
            for seed in (2026, 2027, 2028):
                row = dict(start=start, seed=seed, episodes=128, final_complete=count,
                    ever_complete=count, final_inside=count, qualified_release=count,
                    settled=count, retreated=count, failures=0, controller_failures=0,
                    nonfinite=0, overflow=0, timeout_without_completion=128-count)
                rows.append(row)
        result[units] = dict(reports=rows)
    return result


def test_all_starts_need_fifty_percent_and_budget_ties_are_explicit():
    result = campaign['compare_checkpoints'](evaluations())
    assert result['best_candidate_units'] == 6
    assert result['threshold_met']
    assert not result['checkpoints'][1]['all_starts_at_least_50_percent']
    assert result['checkpoints'][1]['mean_start_success_rate'] > .8
    assert len(result['rows']) == 16
    assert all(row['episodes'] == 384 for row in result['rows'])


def test_physics_failure_is_not_eligible_for_selection():
    data = evaluations()
    data[6]['reports'][0]['overflow'] = 1
    result = campaign['compare_checkpoints'](data)
    assert result['best_candidate_units'] == 8
    assert not result['checkpoints'][2]['physics_valid']


def test_missing_or_duplicate_seed_is_rejected():
    data = evaluations()
    data[2]['reports'].pop()
    with pytest.raises(ValueError, match='missing/duplicate'):
        campaign['compare_checkpoints'](data)
    data = evaluations()
    data[2]['reports'].append(deepcopy(data[2]['reports'][0]))
    with pytest.raises(ValueError, match='missing/duplicate'):
        campaign['compare_checkpoints'](data)


def test_campaign_only_changes_budget_checkpoint_frequency_and_logging():
    old = yaml.safe_load((ROOT/'configs/show103/stage-reward-v1-4096.yaml').read_text())
    new = yaml.safe_load((ROOT/'configs/show103/stage-reward-v1-4096-2-4-6-8.yaml').read_text())
    assert old['environment'] == new['environment']
    assert old['network'] == new['network']
    t = new['training']
    assert t['num_timesteps'] == 67108864
    assert t['num_evals'] == 5
    step = t['batch_size']*t['num_minibatches']*t['unroll_length']
    assert t['num_timesteps']//4//step*step == 16777216
    assert {k: v for k, v in t.items() if k not in ('num_timesteps', 'num_evals', 'log_training_metrics', 'training_metrics_steps')} == {
        k: v for k, v in old['training'].items() if k not in ('num_timesteps', 'num_evals')}


def local_evaluations():
    result = evaluations()
    for unit, data in result.items():
        for row in data['reports']:
            row.update(local_success=128, local_success_definition='stage-local', final_complete=0)
        for seed in (2026, 2027, 2028):
            row = deepcopy(data['reports'][0])
            row.update(start='post_release', seed=seed, local_success=64 if unit < 6 else 96)
            data['reports'].append(row)
    return result


def test_local_comparison_uses_achievements_and_additional_budget():
    result = campaign['compare_local_checkpoints'](local_evaluations())
    assert len(result['rows']) == 20 and result['threshold_met']
    assert result['best_candidate_units'] == 6
    row = result['rows'][0]
    assert row['additional_units'] == 2 and row['cumulative_units'] == 6
    assert row['episodes'] == 384 and row['local_success_rate'] == 1.
    assert row['final_complete'] == 0


def test_local_comparison_rejects_missing_fifth_and_overflow():
    data = local_evaluations()
    data[6]['reports'][0]['overflow'] = 1
    assert campaign['compare_local_checkpoints'](data)['best_candidate_units'] == 8
    data[2]['reports'].pop()
    with pytest.raises(ValueError, match='missing/duplicate'):
        campaign['compare_local_checkpoints'](data)


def test_short_continuation_compares_additional_two_after_parent_eight():
    data = local_evaluations()
    result = campaign['compare_local_checkpoints']({2:data[2]}, parent_units=8, units_to_compare=[2])
    assert len(result['rows']) == 5
    assert result['best_candidate_units'] == 2
    assert all(r['cumulative_units'] == 10 and r['additional_units'] == 2 for r in result['rows'])
