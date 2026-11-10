"""Run two fresh, matched final-v5 actors sequentially on one GPU.

Each actor uses the established recurrent learner/campaign. The original
Transformer16u checkpoint is an immutable practical target, not the fresh
architecture control. No reward edits or automatic difficulty promotion.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from piper_sim_lab.actor_comparison import compare_runs, validate_configs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def resource_summary(root):
    training = root/'training'
    summary = json.loads((training/'summary.json').read_text())
    gpu = json.loads((training/'gpu.json').read_text())
    samples = gpu.get('samples', [])
    utilization = [x['utilization_percent'] for x in samples]
    return dict(training=summary, peak_board_memory_mib=gpu['peak_board_memory_mib'],
        mean_gpu_utilization_percent=sum(utilization)/len(utilization) if utilization else None,
        gpu_scope=gpu['scope'])


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish-dir', type=Path)
    parser.add_argument('--mlp-config', type=Path,
        default=repo/'configs/show108/stage-reward-v5-mlp-lstm-4096-2-to16.yaml')
    parser.add_argument('--transformer-config', type=Path,
        default=repo/'configs/show108/stage-reward-v5-transformer-lstm-4096-2-to16.yaml')
    parser.add_argument('--historical-target', type=Path,
        default=repo/'docs/show103/mjx-3090/progress-reward-v5-continuation-16u.json')
    parser.add_argument('--success-margin', type=float, default=.05)
    parser.add_argument('--A-error-margin-deg', type=float, default=3.)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if not 0 < args.success_margin < 1 or not 0 < args.A_error_margin_deg < 180:
        parser.error('comparison margins must be finite positive tolerances')
    sources = dict(mlp_lstm=args.mlp_config.resolve(), transformer_lstm=args.transformer_config.resolve())
    configs = {actor: yaml.safe_load(path.read_text()) for actor, path in sources.items()}
    units = validate_configs(configs['mlp_lstm'], configs['transformer_lstm'])
    historical = json.loads(args.historical_target.read_text())['continued']
    protocol = dict(actors=list(sources), fresh=True, checkpoint_units=units,
        transitions_per_actor=configs['mlp_lstm']['training']['num_timesteps'],
        total_training_transitions=2*configs['mlp_lstm']['training']['num_timesteps'],
        worlds=4096, reset_probabilities=[.2]*5, training_seed=configs['mlp_lstm']['seed'],
        evaluation_seeds=[2026, 2027, 2028], episodes_per_start=384,
        success_margin=args.success_margin, A_error_margin_deg=args.A_error_margin_deg,
        uncertainty='One training seed per actor; paired evaluation uncertainty only',
        primary_full_completion='Actual A-return completion; exclude the high-drop terminal exception',
        historical_target='Published Transformer16u, different v2/v3/v5 training history',
        order='MLP+lastLSTM train/evaluate/videos, then fresh Transformer+lastLSTM, then matched comparison',
        reward_changed=False, control_changed=False, real_robot=False)
    if args.plan_only:
        print(json.dumps(protocol, indent=2))
        return
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_files = [*repo.joinpath('src').rglob('*.py'),
                    *repo.joinpath('scripts/show103').glob('*.py'), *repo.joinpath('scripts/show108').glob('*.py')]
    bank_dir = repo/'src/piper_sim_lab/assets/reset_banks'
    source_files += [bank_dir/configs['mlp_lstm']['environment'][key]
                     for key in ('position_reset_bank', 'evaluation_position_reset_bank')]
    fingerprints = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    status = dict(protocol, status='starting', started_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        source_sha256=fingerprints, shared_runtime=__import__('piper_runtime.identity',fromlist=['identity']).identity(), root=str(root), child_pid=None, completed_actors=[])
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None,[str(repo/'src'),os.environ.get('PYTHONPATH','')])), PYTHONUNBUFFERED='1')
    environment.pop('LD_LIBRARY_PATH', None)
    write_json(root/'comparison-campaign.json', status)
    for actor, path in sources.items():
        dest = root/'configs'/path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        sources[actor] = dest
    write_json(root/'protocol.json', protocol)
    shutil.copyfile(args.historical_target, root/'historical-target.json')
    def check_source():
        changed = [n for n, digest in fingerprints.items()
                   if not (repo/n).exists() or hashlib.sha256((repo/n).read_bytes()).hexdigest() != digest]
        if changed:
            raise RuntimeError(f'source changed during matched run: {changed}')
    try:
        for actor, config in sources.items():
            check_source()
            command = [sys.executable, '-u', str(repo/'scripts/show103/run_stage_campaign.py'),
                '--config', str(config), '--output', str(root/actor), '--units', *map(str, units)]
            if args.publish_dir:
                command += ['--publish-dir', str(args.publish_dir.resolve()), '--artifact-prefix', f'show108-{actor}-fresh-v5']
            status.update(status='running_actor', actor=actor, command=command)
            with (root/f'{actor}.log').open('w') as log:
                process = subprocess.Popen(command, cwd=repo, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                status['child_pid'] = process.pid
                while process.poll() is None:
                    child = root/actor/'campaign.json'
                    if child.exists():
                        state = json.loads(child.read_text())
                        status.update(actor_phase=state['status'],
                            transitions_observed=state.get('training_transitions_observed', 0),
                            evaluated_units=state.get('evaluated_units', []))
                    status['last_checked_utc'] = datetime.now(timezone.utc).isoformat()
                    write_json(root/'comparison-campaign.json', status)
                    time.sleep(5)
            status['child_pid'] = None
            if process.returncode:
                raise RuntimeError(f'{actor} campaign exited{process.returncode}; see {root/actor/"campaign.json"}')
            check_source()
            status['completed_actors'].append(actor)
            write_json(root/'comparison-campaign.json', status)
        status['status'] = 'comparing'
        write_json(root/'comparison-campaign.json', status)
        result = compare_runs(root/'mlp_lstm', root/'transformer_lstm', units, historical,
                             args.success_margin, args.A_error_margin_deg)
        result.update(protocol=protocol, source_commit=status['source_commit'],
            resources={a: resource_summary(root/a) for a in sources})
        write_json(root/'matched-comparison.json', result)
        fields = ['units', 'actor', 'start', 'episodes', 'local_success', 'local_success_rate',
                  'final_complete', 'full_success_rate', 'full_home_complete', 'full_home_success_rate',
                  'final_max_A_error_deg_median', 'best_max_A_error_deg_median']
        with (root/'matched-comparison.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for point in result['checkpoints']:
                for actor in sources:
                    for start, row in point[actor].items():
                        writer.writerow(dict(units=point['units'], actor=actor, start=start,
                                             **{k: row[k] for k in fields if k in row}))
        if args.publish_dir:
            args.publish_dir.mkdir(parents=True, exist_ok=True)
            for name in ('matched-comparison.json', 'matched-comparison.csv'):
                shutil.copyfile(root/name, args.publish_dir/('show108-'+name))
        status.update(status='completed', ended_utc=datetime.now(timezone.utc).isoformat(),
            comparison=str(root/'matched-comparison.json'),
            final_reproduction_supported_on_this_evaluation=result['checkpoints'][-1]['reproduction_supported_on_this_evaluation'])
    except BaseException as error:
        status.update(status='failed', error=str(error), ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(root/'comparison-campaign.json', status)
        raise
    write_json(root/'comparison-campaign.json', status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
