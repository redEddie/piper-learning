#!/usr/bin/env python3
"""Evaluate a pick-place checkpoint from empty hands on fixed held-out seeds."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--randomization', choices=['both', 'dr', 'nodr'], default='both')
    parser.add_argument('--reuse', action='store_true', help='Validate and reuse complete reports')
    args = parser.parse_args()
    if args.num_envs <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error('num-envs must be positive and seeds must be distinct')
    run = args.run.resolve()
    checkpoint = (args.checkpoint or sorted(p for p in (run / 'checkpoints').iterdir() if p.is_dir() and p.name.isdigit())[-1]).resolve()
    original = json.loads((run / 'config.json').read_text())
    if original['environment'].get('task') != 'pick_place':
        parser.error('This evaluator requires a pick_place run')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(repo / 'src'))
    environment.pop('LD_LIBRARY_PATH', None)
    groups, reports = [], []
    modes = ['dr', 'nodr'] if args.randomization == 'both' else [args.randomization]
    for mode in modes:
        config = json.loads(json.dumps(original))
        effective = config.get('resolved_environment', config['environment'])
        effective.update(reset_stage_probs=[1, 0, 0, 0], obs_randomization=(mode == 'dr'), physics_randomization=(mode == 'dr'))
        path = args.output / f'evaluation-{mode}-config.json'
        path.write_text(json.dumps(config, indent=2) + '\n')
        selected = []
        for seed in args.seeds:
            directory = args.output / f'eval-empty-{mode}-s{seed}'
            report_path = directory / 'evaluation.json'
            if report_path.exists() and args.reuse:
                report = json.loads(report_path.read_text())
            else:
                if directory.exists():
                    raise FileExistsError(f'Output already exists: {directory}; inspect it before reuse')
                command = [sys.executable, '-c', 'from piper_sim_lab.evaluate import main; main()',
                           '--checkpoint', str(checkpoint), '--config', str(path), '--output', str(directory),
                           '--num-envs', str(args.num_envs), '--seed', str(seed), '--platform', 'gpu']
                print(f'Evaluating {mode}, seed {seed}', flush=True)
                with (args.output / f'{directory.name}.log').open('w') as log:
                    subprocess.run(command, cwd=repo, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads(report_path.read_text())
            e = report['effective_environment']
            if (Path(report['checkpoint']).resolve() != checkpoint or report['seed'] != seed
                    or report['num_envs'] != args.num_envs or e['reset_stage_probs'] != [1, 0, 0, 0]
                    or e['obs_randomization'] != (mode == 'dr') or e['physics_randomization'] != (mode == 'dr')
                    or e['impl'] != 'warp' or report['provenance']['platform'] != 'gpu'):
                raise ValueError(f'Evaluation contract mismatch: {report_path}')
            value = report['results']['trained']
            if not value['finite'] or value['failure_rate'] != 0:
                raise ValueError(f'Invalid physics in {report_path}')
            row = {'randomization': mode, 'seed': seed, **value}
            selected.append(row)
            reports.append(row)
        group = {'randomization': mode, 'episodes': args.num_envs * len(selected)}
        for metric in ['final_success_rate', 'ever_success_rate']:
            values = [r[metric] for r in selected]
            group[metric + '_mean'] = statistics.mean(values)
            group[metric + '_evaluation_seed_sd'] = statistics.stdev(values) if len(values) > 1 else None
        groups.append(group)
    summary = {'checkpoint': str(checkpoint), 'reset_stage_probs': [1, 0, 0, 0],
               'evaluation_seeds': args.seeds, 'num_envs_per_seed': args.num_envs,
               'groups': groups, 'per_seed': reports,
               'scope': 'Simulation; evaluation seeds are not independent training repetitions'}
    (args.output / 'empty-hand-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
