"""Train the fixed SHOW-103 curriculum, then compare four saved checkpoints.

The 2/4/6/8 multipliers refer to 8,388,608 transitions, not SHOW-76's old 1x.
No automatic difficulty promotion and no robot interfaces are used.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

STARTS = ('initial_a', 'pre_grasp', 'held', 'over_bowl')
FIVE_STARTS = STARTS + ('post_release',)
UNITS = (2, 4, 6, 8)
UNIT_STEPS = 8388608


def compare_local_checkpoints(evaluations, parent_units=4, seeds=(2026, 2027, 2028), environments=128, threshold=.5, units_to_compare=UNITS):
    rows, checkpoints = [], []
    for units in units_to_compare:
        reports = evaluations[units]['reports']
        expected = {(start, seed) for start in FIVE_STARTS for seed in seeds}
        actual = [(r['start'], r['seed']) for r in reports]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f'{units}: missing/duplicate local-stage evaluation start/seed')
        rates, physics_valid = [], True
        for index, start in enumerate(FIVE_STARTS):
            selected = [r for r in reports if r['start'] == start]
            if any(r['episodes'] != environments for r in selected):
                raise ValueError('local evaluation denominator differs')
            episodes = sum(r['episodes'] for r in selected)
            count = sum(r['local_success'] for r in selected)
            if not 0 <= count <= episodes:
                raise ValueError('invalid local success count')
            rate = count/episodes
            row = dict(additional_units=units, cumulative_units=parent_units+units,
                additional_transitions=UNIT_STEPS*units, cumulative_transitions=UNIT_STEPS*(parent_units+units),
                stage=index+1, start=start, episodes=episodes, local_success=count,
                local_success_rate=rate, meets_50_percent=rate >= threshold,
                local_success_definition=selected[0]['local_success_definition'])
            for key in ('final_complete', 'qualified_release', 'settled', 'failures', 'controller_failures', 'nonfinite', 'overflow'):
                row[key] = sum(r[key] for r in selected)
            for key in ('open_command_episodes', 'invalid_release_episodes', 'high_drop_success', 'home_complete'):
                if all(key in r for r in selected):
                    row[key] = sum(r[key] for r in selected)
            physics_valid &= row['nonfinite'] == row['overflow'] == 0
            rows.append(row)
            rates.append(rate)
        checkpoints.append(dict(units=units, additional_units=units, cumulative_units=parent_units+units,
            min_local_success_rate=min(rates), mean_local_success_rate=sum(rates)/len(rates),
            physics_valid=physics_valid, all_stages_at_least_50_percent=physics_valid and min(rates) >= threshold))
    eligible = [c for c in checkpoints if c['physics_valid']]
    best = max(eligible, key=lambda c: (c['min_local_success_rate'], c['mean_local_success_rate'], -c['units'])) if eligible else None
    return dict(unit_transitions=UNIT_STEPS, parent_units=parent_units, checkpoints=checkpoints, rows=rows,
        threshold=threshold, best_candidate_units=best['units'] if best else None,
        threshold_met=bool(best and best['all_stages_at_least_50_percent']),
        selection='Highest minimum stage-local success rate, then mean, then smaller additional budget; no automatic promotion',
        seeds=list(seeds), environments_per_seed=environments, difficulty_promoted=False, real_robot=False,
        scope='Five conditional initializations, local achievements before failure; training seed7 is one repetition')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def compare_checkpoints(evaluations, seeds=(2026, 2027, 2028), environments=128, threshold=.5, initial_geometry='fixed-v1'):
    rows, checkpoints = [], []
    for units in UNITS:
        reports = evaluations[units]['reports']
        expected = {(start, seed) for start in STARTS for seed in seeds}
        actual = [(r['start'], r['seed']) for r in reports]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f'{units}: missing/duplicate evaluation start/seed')
        rates, valid_physics = [], True
        for start in STARTS:
            selected = [r for r in reports if r['start'] == start]
            if any(r['episodes'] != environments for r in selected):
                raise ValueError('evaluation denominator differs')
            episodes = sum(r['episodes'] for r in selected)
            successes = sum(r['final_complete'] for r in selected)
            if not 0 <= successes <= episodes:
                raise ValueError('invalid success count')
            rate = successes/episodes
            row = dict(checkpoint_units=units, training_transitions=UNIT_STEPS*units,
                       start=start, episodes=episodes, final_complete=successes,
                       final_complete_rate=rate, meets_50_percent=rate >= threshold)
            for key in ('ever_complete', 'final_inside', 'qualified_release', 'settled', 'retreated',
                        'failures', 'controller_failures', 'nonfinite', 'overflow', 'timeout_without_completion'):
                row[key] = sum(r[key] for r in selected)
            progress = {k: sum(r.get('progress_counts', {}).get(k, 0) for r in selected)
                        for k in ('approached', 'grasped_and_lifted', 'reached_bowl', 'qualified_release', 'settled')}
            row['progress_rates'] = {k: v/episodes for k, v in progress.items()}
            valid_physics &= row['nonfinite'] == row['overflow'] == 0
            rows.append(row)
            rates.append(rate)
        checkpoints.append(dict(units=units, transitions=units*UNIT_STEPS,
            min_start_success_rate=min(rates), mean_start_success_rate=sum(rates)/len(rates),
            physics_valid=valid_physics,
            all_starts_at_least_50_percent=valid_physics and min(rates) >= threshold))
    valid = [c for c in checkpoints if c['physics_valid']]
    best = max(valid, key=lambda c: (c['min_start_success_rate'], c['mean_start_success_rate'], -c['units'])) if valid else None
    return dict(unit_transitions=UNIT_STEPS, checkpoints=checkpoints, rows=rows,
        threshold=threshold, seeds=list(seeds), environments_per_seed=environments,
        best_candidate_units=best['units'] if best else None,
        threshold_met=bool(best and best['all_starts_at_least_50_percent']),
        selection='Highest minimum per-start final completion; then highest mean; then lower training budget',
        difficulty_promoted=False, real_robot=False,
        initial_geometry=initial_geometry,
        scope=f'{initial_geometry} geometry and dynamics DR. Evaluation seeds are not independent training repetitions.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish-dir', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--artifact-prefix', default='show103')
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--parent-units', type=int, default=0)
    parser.add_argument('--units', nargs='+', type=int, default=list(UNITS),
                        help='Evenly spaced additional budgets, e.g. 2 for a short ablation')
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text())
    train = config['training']
    units_requested = args.units
    if not units_requested or units_requested[0] <= 0 or units_requested != [units_requested[0]*i for i in range(1,len(units_requested)+1)]:
        raise ValueError('checkpoint units must be positive and evenly spaced from zero')
    if train['num_timesteps'] != UNIT_STEPS*units_requested[-1] or train['num_envs'] != 4096 or train['num_evals'] != len(units_requested)+1:
        raise ValueError('campaign requires 4,096 worlds and matching budget/evaluation frequency')
    increment = train['batch_size']*train['num_minibatches']*train['unroll_length']*train['action_repeat']
    epoch = math.ceil(train['num_timesteps']/(len(units_requested)*increment))*increment
    expected_steps = [epoch*i for i in range(1, len(units_requested)+1)]
    if expected_steps != [UNIT_STEPS*u for u in units_requested]:
        raise ValueError('Brax checkpoint rounding would differ from requested budgets')
    five = config['environment']['task'] == 'stage_reward_pick_place_five_starts_v2'
    if not five and units_requested != list(UNITS):
        raise ValueError('custom checkpoint budgets are supported only for five-start local evaluation')
    recurrent = config.get('learner') == 'recurrent_ppo'
    if recurrent and bool(args.warm_start) != (args.parent_units > 0):
        raise ValueError('recurrent continuation requires both a checkpoint and positive parent units')
    if five and not recurrent and (not args.warm_start or args.parent_units != 4 or config['environment']['reset_stage_probs'] != [.2]*5):
        raise ValueError('five-start continuation requires parent4u and uniform20% resets')
    if args.warm_start:
        required = ('learner.msgpack','metadata.json') if recurrent else ('commit_success.txt',)
        if not all((args.warm_start/name).exists() for name in required):
            raise ValueError('missing/incomplete warm-start checkpoint')
        if recurrent:
            meta = json.loads((args.warm_start/'metadata.json').read_text())
            if meta.get('cumulative_transitions',meta['step']) != args.parent_units*UNIT_STEPS:
                raise ValueError('parent units differ from the recurrent checkpoint')
    geometry = 'five-start-random-v2' if five else ('random-positions-v1' if config['environment']['task'] == 'stage_reward_pick_place_random_positions' else 'fixed-v1')
    plan = dict(checkpoint_transitions=expected_steps, unit_transitions=UNIT_STEPS,
        checkpoint_units=units_requested, starts=list(FIVE_STARTS if five else STARTS), evaluation_seeds=[2026, 2027, 2028],
        environments_per_seed=128, threshold=.5, total_training_transitions=train['num_timesteps'],
        initial_geometry=geometry,
        dynamics_dr=config['environment'].get('physics_randomization', False),
        observation_dr=config['environment'].get('obs_randomization', False),
        order='one fresh training run; after completion evaluate all four checkpoints; compare',
        real_robot=False)
    if five:
        plan.update(parent_units=args.parent_units, warm_start=str(args.warm_start.resolve()) if args.warm_start else None,
            checkpoint_units_mean='additional to preserved parent4u; cumulative6/8/10/12u',
            criterion='stage-local success; all five >=50%',
            order='warm-start parent4u; save additional2/4/6/8u; evaluate all four with five starts and compare local success')
    if recurrent:
        plan.update(learner='recurrent_ppo', actor=config['actor'],
            checkpoint_units_mean='additional to parent' if args.warm_start else 'fresh model cumulative units',
            order='restore full recurrent learner' if args.warm_start else 'fresh recurrent actor')
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None,[str(repo/'src'),os.environ.get('PYTHONPATH','')])),
                       PYTHONUNBUFFERED='1')
    environment.pop('LD_LIBRARY_PATH', None)
    status = dict(plan, status='starting', started_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        output=str(root), child_pid=None, shared_runtime=__import__('piper_runtime.identity',fromlist=['identity']).identity())
    write_json(root/'campaign.json', status)
    shutil.copyfile(args.config, root/'training.yaml')

    def run(command, phase, log_path, progress=None):
        status.update(status=phase, command=command)
        with log_path.open('w') as log:
            process = subprocess.Popen(command, cwd=repo, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = process.pid
            write_json(root/'campaign.json', status)
            while process.poll() is None:
                if progress and progress.exists():
                    # Progress callback rows can arrive out of order; use the maximum completed budget.
                    values = []
                    for line in progress.read_text().splitlines():
                        try:
                            values.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # concurrent append may leave one incomplete last line
                    if values:
                        latest = max(values, key=lambda r: r['steps'])
                        status['training_transitions_observed'] = latest['steps']
                        status['training_elapsed_seconds_observed'] = latest['elapsed_seconds']
                status['last_checked_utc'] = datetime.now(timezone.utc).isoformat()
                write_json(root/'campaign.json', status)
                time.sleep(5)
        status['child_pid'] = None
        if process.returncode:
            raise RuntimeError(f'{phase} exited {process.returncode}; see {log_path}')

    try:
        training_command = [sys.executable, '-u', '-m', 'piper_sim_lab.recurrent_train' if recurrent else 'piper_sim_lab.train', '--config', str(root/'training.yaml'),
             '--output', str(root/'training'), '--platform', 'gpu']
        if recurrent:
            training_command.extend(['--actor', config['actor']])
        if args.warm_start:
            training_command.extend(['--warm-start', str(args.warm_start.resolve())])
        run(training_command,
            'training', root/'training.log', root/'training/metrics.jsonl')
        evaluations = {}
        for units, steps in zip(units_requested, expected_steps):
            checkpoint = root/'training/checkpoints'/f'{steps:012d}'
            required = ('learner.msgpack','metadata.json') if recurrent else ('ppo_network_config.json','commit_success.txt')
            if not all((checkpoint/name).exists() for name in required):
                raise RuntimeError(f'incomplete/missing checkpoint: {checkpoint}')
            destination = root/'evaluation'/f'{units}u'
            run([sys.executable, '-u', str(repo/'scripts/show103/evaluate_stage_env.py'),
                 '--run', str(root/'training'), '--checkpoint', str(checkpoint),
                 '--output', str(destination)], f'evaluating_{units}u', root/f'evaluation-{units}u.log')
            evaluations[units] = json.loads((destination/'summary.json').read_text())
            status.setdefault('evaluated_units', []).append(units)
            if five:
                # Publish each completed checkpoint immediately, before all
                # four comparisons are ready.
                interim = dict(additional_units=units, cumulative_units=args.parent_units+units,
                    reports=evaluations[units]['reports'], criterion='stage-local success')
                write_json(root/f'local-stages-plus{units}u.json', interim)
                if args.publish_dir:
                    write_json(args.publish_dir/f'{args.artifact_prefix}-plus{units}u-local-stages.json', interim)
            write_json(root/'campaign.json', status)
        comparison = compare_local_checkpoints(evaluations, parent_units=args.parent_units, units_to_compare=units_requested) if five else compare_checkpoints(evaluations, initial_geometry=geometry)
        comparison.update(training=str(root/'training'), git_commit=status['git_commit'])
        write_json(root/'comparison.json', comparison)
        fields = [k for k in comparison['rows'][0] if k != 'progress_rates']
        with (root/'comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: v for k, v in row.items() if k != 'progress_rates'} for row in comparison['rows'])
        # Record the first A/held case, without selecting successful scenes.
        for units in sorted({units_requested[-1], comparison['best_candidate_units']} - {None}):
            for start in (FIVE_STARTS if five else ('initial_a', 'held')):
                destination = root/'videos'/f'{units}u-{start}-first.mp4'
                run([sys.executable, str(repo/'scripts/show103/render_stage_env.py'),
                     '--run', str(root/'training'),
                     '--trace', str(root/'evaluation'/f'{units}u'/f'{start}-s2026.npz'),
                     '--output', str(destination), '--scene', '0'],
                    f'rendering_{units}u_{start}', root/f'render-{units}u-{start}.log')
        if args.publish_dir:
            args.publish_dir.mkdir(parents=True, exist_ok=True)
            suffix = '-four-checkpoints' if len(units_requested) == 4 else '-checkpoints'
            shutil.copyfile(root/'comparison.json', args.publish_dir/(args.artifact_prefix+suffix+'.json'))
            shutil.copyfile(root/'comparison.csv', args.publish_dir/(args.artifact_prefix+suffix+'.csv'))
            for video in (root/'videos').glob('*.mp4'):
                shutil.copyfile(video, args.publish_dir/(args.artifact_prefix+'-'+video.name))
        status.update(status='completed', threshold_met=comparison['threshold_met'],
            comparison=str(root/'comparison.json'), best_candidate_units=comparison['best_candidate_units'],
            ended_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(status='failed', error=str(error), ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(root/'campaign.json', status)
        raise
    write_json(root/'campaign.json', status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
