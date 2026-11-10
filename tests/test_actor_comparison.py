"""Matched protocol and uncertainty guards; no GPU or training required."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from piper_sim_lab.actor_comparison import (load_evaluation, paired_interval,
    practical_target_reached, validate_configs, STARTS)

ROOT = Path(__file__).resolve().parents[1]


def configs():
    base = ROOT/'configs/show108'
    return [yaml.safe_load((base/f'stage-reward-v5-{actor}-4096-2-to16.yaml').read_text())
            for actor in ('mlp-lstm', 'transformer-lstm')]


def test_published_configs_are_matched_except_encoder():
    a, b = configs()
    assert validate_configs(a, b) == list(range(2, 17, 2))
    assert a['training']['num_timesteps'] == b['training']['num_timesteps'] == 134217728


@pytest.mark.parametrize('field', ['reward', 'optimizer', 'seed', 'frequency', 'interface'])
def test_confounding_changes_are_rejected(field):
    a, b = configs()
    if field == 'reward':
        a['environment']['five_start_drop_weight'] = 0.
    elif field == 'optimizer':
        a['training']['learning_rate'] = .0001
    elif field == 'seed':
        a['seed'] = 8
    elif field == 'frequency':
        a['training']['num_evals'] = b['training']['num_evals'] = 5
    else:
        a['actor_architecture']['hidden_sizes'][-1] = 32
    with pytest.raises(ValueError):
        validate_configs(a, b)


def test_paired_bootstrap_preserves_pairing_and_sign():
    x = np.array([0, 1]*192)
    same = paired_interval(x, x)
    assert same['difference_mlp_minus_transformer'] == 0.
    assert same['ci95'] == [0., 0.]
    better = paired_interval(np.ones(384), np.zeros(384))
    assert better['difference_mlp_minus_transformer'] == 1.
    assert better['ci95'] == [1., 1.]
    errors = paired_interval(np.full(384, 10.), np.full(384, 20.), 'median')
    assert errors['ci95'] == [-10., -10.]
    with pytest.raises(ValueError, match='matching'):
        paired_interval([1, 0], [1])
    with pytest.raises(ValueError, match='finite'):
        paired_interval([np.nan], [0])


def test_equally_poor_policies_do_not_reproduce_historical_target():
    baseline = json.loads((ROOT/'docs/show103/mjx-3090/progress-reward-v5-continuation-16u.json').read_text())['continued']
    metrics = {name: dict(local_success_rate=0., full_home_success_rate=0.,
                          final_max_A_error_deg_median=90.) for name in STARTS}
    assert not practical_target_reached(metrics, baseline)
    matched = deepcopy(metrics)
    for name, row in baseline.items():
        matched[name] = dict(local_success_rate=row['local_success_rate'],
            full_home_success_rate=row['home_complete']/row['episodes'],
            final_max_A_error_deg_median=row['final_max_A_error_deg_median'])
    assert practical_target_reached(matched, baseline)


def test_high_drop_successes_cannot_substitute_for_full_A_return():
    baseline = json.loads((ROOT/'docs/show103/mjx-3090/progress-reward-v5-continuation-16u.json').read_text())['continued']
    metrics = {name: dict(local_success_rate=1., full_success_rate=1., full_home_success_rate=0.,
        final_max_A_error_deg_median=10.) for name in STARTS}
    assert not practical_target_reached(metrics, baseline)


def test_missing_evaluation_cases_are_rejected_before_analysis(tmp_path):
    folder = tmp_path/'evaluation/2u'; folder.mkdir(parents=True)
    (folder/'summary.json').write_text(json.dumps(dict(reports=[])))
    with pytest.raises(ValueError, match='missing or duplicate'):
        load_evaluation(tmp_path, 2)
