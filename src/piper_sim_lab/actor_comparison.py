"""Matched evaluation analysis; uncertainty is conditional on trained policies.

Evaluation seeds are not independent training repetitions. In particular, two
unsuccessful policies are not evidence of reproducing the historical target.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

STARTS = ('initial_a', 'pre_grasp', 'held', 'over_bowl', 'post_release')
SEEDS = (2026, 2027, 2028)
UNIT_STEPS = 8388608


def validate_configs(mlp, transformer):
    configs = [deepcopy(mlp), deepcopy(transformer)]
    for c, actor in zip(configs, ('mlp_lstm', 'transformer_lstm')):
        if c.pop('actor') != actor or c.get('learner') != 'recurrent_ppo':
            raise ValueError('comparison requires the two recurrent encoder actors')
        arch = c.pop('actor_architecture')
        feature_size = arch['hidden_sizes'][-1] if actor == 'mlp_lstm' else arch['token_width']
        if feature_size != 64:
            raise ValueError('both encoder interfaces must have width64')
        env, training = c['environment'], c['training']
        if (training['num_envs'] != 4096 or env['reset_stage_probs'] != [.2]*5 or
            not env.get('five_start_progress_reward')):
            raise ValueError('comparison requires v5,4096 worlds and uniform five starts')
    if configs[0] != configs[1]:
        raise ValueError('only actor and actor_architecture may differ')
    training = mlp['training']
    steps = training['num_timesteps']
    if steps % (2*UNIT_STEPS) or steps <= 0:
        raise ValueError('budget must contain whole2u checkpoints')
    units = list(range(2, steps//UNIT_STEPS+1, 2))
    if training['num_evals'] != len(units)+1:
        raise ValueError('checkpoint frequency must be every2u')
    return units


def paired_interval(mlp, transformer, statistic='mean', draws=4000, seed=108):
    """Paired bootstrap CI using the same reset case indices in both policies."""
    a, b = np.asarray(mlp, float), np.asarray(transformer, float)
    if a.ndim != 1 or a.shape != b.shape or not a.size or not np.isfinite([a, b]).all():
        raise ValueError('paired samples must be finite matching nonempty vectors')
    if draws <= 0 or statistic not in ('mean', 'median'):
        raise ValueError('invalid bootstrap settings')
    reduce = np.mean if statistic == 'mean' else np.median
    indices = np.random.default_rng(seed).integers(0, len(a), size=(draws, len(a)))
    samples = reduce(a[indices], axis=1)-reduce(b[indices], axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return dict(difference_mlp_minus_transformer=float(reduce(a)-reduce(b)),
                ci95=[float(low), float(high)], paired_cases=len(a), bootstrap_draws=draws)


def practical_target_reached(metrics, historical, success_margin=.05, error_margin_deg=3.):
    checks = [metrics[n]['local_success_rate'] >= historical[n]['local_success_rate']-success_margin
              for n in STARTS]
    checks.append(metrics['initial_a']['full_home_success_rate'] >=
                  historical['initial_a']['home_complete']/historical['initial_a']['episodes']-success_margin)
    checks.append(metrics['post_release']['final_max_A_error_deg_median'] <=
                  historical['post_release']['final_max_A_error_deg_median']+error_margin_deg)
    return bool(all(checks))


def load_evaluation(root, units):
    root = Path(root)
    folder = root/'evaluation'/f'{units}u'
    reports = json.loads((folder/'summary.json').read_text())['reports']
    expected = {(n, s) for n in STARTS for s in SEEDS}
    identities = [(r['start'], r['seed']) for r in reports]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError('missing or duplicate start/seed evaluations')
    metrics, samples = {}, {}
    event_names = dict(initial_a='approach_complete', pre_grasp='grasp_complete',
        held='transport_complete', over_bowl='place_complete', post_release='home_complete')
    for name in STARTS:
        rows = sorted((r for r in reports if r['start'] == name), key=lambda r: r['seed'])
        if any(r['episodes'] != 128 for r in rows):
            raise ValueError('comparison requires128 episodes per evaluation seed')
        values = {k: sum(r[k] for r in rows) for k in ('episodes', 'local_success', 'final_complete',
            'failures', 'nonfinite', 'overflow', 'controller_failures', 'home_complete', 'high_drop_success')}
        if len({r['position_bank_sha256'] for r in rows}) != 1:
            raise ValueError('evaluation banks differ across seeds')
        local, full, full_home, error, best, case_ids = [], [], [], [], [], []
        for row in rows:
            with np.load(folder/f'{name}-s{row["seed"]}.npz') as trace:
                prefix = trace['active'] & ~trace['failed']
                event = trace['metric_stage_'+event_names[name]] > 0
                if name == 'over_bowl':
                    event = event | (trace['metric_stage_terminal_success'] > 0)
                successes = np.any(event & prefix, axis=0)
                complete = (trace['metric_success'][-1] > 0) & ~trace['failed'][-1]
                home_complete = complete & (trace['metric_stage_terminal_success'][-1] == 0)
                if int(successes.sum()) != row['local_success'] or int(complete.sum()) != row['final_complete']:
                    raise ValueError('summary counts disagree with recorded first-episode events')
                local.extend(successes); full.extend(complete); full_home.extend(home_complete)
                e = trace['metric_home_error']*180/np.pi
                error.extend(e[-1]); best.extend(e.min(axis=0))
                poses = trace['reset_pose_index'][0]
                case_ids.extend((row['seed'], i, int(pose)) for i, pose in enumerate(poses))
        values.update(local_success_rate=values['local_success']/values['episodes'],
            full_success_rate=values['final_complete']/values['episodes'],
            full_home_complete=int(np.sum(full_home)),
            full_home_success_rate=float(np.mean(full_home)),
            final_max_A_error_deg_median=float(np.median(error)),
            best_max_A_error_deg_median=float(np.median(best)))
        metrics[name] = values
        samples[name] = dict(local=np.asarray(local), full=np.asarray(full), full_home=np.asarray(full_home), error=np.asarray(error),
                             case_ids=np.asarray(case_ids), bank=rows[0]['position_bank_sha256'])
    return metrics, samples


def compare_runs(mlp_root, transformer_root, units, historical, success_margin=.05, error_margin_deg=3.):
    reports = []
    for u in units:
        mlp, a = load_evaluation(mlp_root, u)
        transformer, b = load_evaluation(transformer_root, u)
        differences = {}
        for name in STARTS:
            if (a[name]['bank'] != b[name]['bank'] or
                not np.array_equal(a[name]['case_ids'], b[name]['case_ids'])):
                raise ValueError('paired cases or evaluation banks differ')
            interval = paired_interval(a[name]['local'], b[name]['local'])
            interval['noninferiority_supported'] = interval['ci95'][0] >= -success_margin
            differences[name] = interval
        full = paired_interval(a['initial_a']['full_home'], b['initial_a']['full_home'])
        full['noninferiority_supported'] = full['ci95'][0] >= -success_margin
        error = paired_interval(a['post_release']['error'], b['post_release']['error'], 'median')
        error['noninferiority_supported'] = error['ci95'][1] <= error_margin_deg
        valid = all(r['nonfinite'] == r['overflow'] == 0 for group in (mlp, transformer) for r in group.values())
        supported = valid and full['noninferiority_supported'] and error['noninferiority_supported'] and all(
            r['noninferiority_supported'] for r in differences.values())
        target = practical_target_reached(mlp, historical, success_margin, error_margin_deg)
        reports.append(dict(units=u, mlp_lstm=mlp, transformer_lstm=transformer,
            local_success_differences=differences, full_home_success_difference=full, A_error_difference_deg=error,
            finite_evaluations=valid, noninferiority_to_fresh_control_supported=bool(supported),
            historical_practical_target_point_estimates_reached=target,
            reproduction_supported_on_this_evaluation=bool(supported and target)))
    return dict(checkpoints=reports, success_margin=success_margin, A_error_margin_deg=error_margin_deg,
        uncertainty='Paired95% bootstrap across reset cases, conditional on one trained policy per actor. Not training-seed uncertainty.',
        conclusion_scope='Historical16u has a different reward history; compare fresh matched actors for architecture evidence. Both poor policies do not establish target reproduction.',
        historical_target=historical, real_robot=False)
