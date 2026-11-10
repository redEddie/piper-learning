"""Run the approved unchanged continuation and evaluate saved checkpoints.

Evaluation is read-only and deterministic; it never supplies training actions.
Other GPU processes are left untouched while waiting to start training.
"""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',default='runs/show101/mlp-place-v2-continue-8m')
parser.add_argument('--source',default='runs/show101/mlp-place-v2-8m/checkpoints/000008388608')
parser.add_argument('--config',default='configs/show101/refine-v2-8m.yaml')
parser.add_argument('--source-steps',type=int,default=58720256)
parser.add_argument('--evaluation-prefix',default='continuation')
parser.add_argument('--baseline-evaluation',action='store_true')
args=parser.parse_args()
RUN = ROOT / args.run
LOGS = ROOT / 'runs/show101/logs'
SOURCE = ROOT / args.source
PREFIX = args.evaluation_prefix


def report_evaluation(path, steps):
    data = json.loads((path / 'evaluation.json').read_text())
    modes = {}
    for mode, aggregate in data['aggregate'].items():
        rows = [r['placing'] for k, r in data['reports'].items()
                if k.startswith(mode + '-')]
        released = sum(r['release_episodes'] for r in rows)
        contacted = sum(r['floor_contact_episodes'] for r in rows)
        modes[mode] = dict(aggregate,
            supported_then_released=sum(r['qualified_release_episodes'] for r in rows),
            release_episodes=released,
            release_height_mean_mm=(sum((r['detected_release_height_mm_mean'] or 0)
                * r['release_episodes'] for r in rows) / released if released else None),
            contact_peak_mean_m_s=(sum((r['peak_contact_speed_m_s_mean'] or 0)
                * r['floor_contact_episodes'] for r in rows) / contacted if contacted else None))
    row = dict(additional_steps=steps, lineage_steps=args.source_steps + steps, modes=modes)
    with (LOGS / f'{PREFIX}-evaluations.jsonl').open('a') as handle:
        handle.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def evaluate(steps, final=False, baseline=False):
    name = f'baseline-{PREFIX}-evaluation' if baseline else (f'final-{PREFIX}-evaluation' if final else f'interim-{PREFIX}-{steps}-evaluation')
    output = ROOT / 'runs/show101' / name
    command = [sys.executable, '-u', 'scripts/show100/evaluate.py',
        '--run', str(RUN), '--checkpoint', str(SOURCE if baseline else RUN / 'checkpoints' / f'{steps:012d}'),
        '--output', str(output), '--modes', 'empty_hand', 'mixed', 'preinside',
        '--seeds', *(['2026', '2027', '2028'] if final or baseline else ['2026'])]
    print(f'Evaluating additional checkpoint {steps}', flush=True)
    with (LOGS / f'{name}.log').open('w') as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
    report_evaluation(output, steps)


def main():
    if RUN.exists():
        raise FileExistsError(f'Preserving existing output: {RUN}')
    from piper_sim_lab.experiment_guard import compute_pids
    deadline = time.monotonic() + 600
    while compute_pids():
        if time.monotonic() >= deadline:
            raise RuntimeError('GPU remained occupied; no training was started')
        print('Waiting for existing GPU work to finish; other processes untouched', flush=True)
        time.sleep(15)
    command = [sys.executable, '-u', 'scripts/show101/run_refine.py',
        '--config', args.config, '--output', str(RUN),
        '--warm-start', str(SOURCE)]
    completed = set()
    with (LOGS / f'{RUN.name}.log').open('w') as handle:
        child = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        print(f'Training started pid={child.pid}', flush=True)
        if args.baseline_evaluation:
            while not (RUN/'config.json').exists():
                if child.poll() is not None:
                    raise RuntimeError('Training exited before writing its resolved config')
                time.sleep(2)
            evaluate(0,baseline=True)
        while True:
            for steps in (2097152, 4194304, 6291456):
                ready = RUN / 'checkpoints' / f'{steps:012d}' / 'commit_success.txt'
                if steps not in completed and ready.exists():
                    evaluate(steps)
                    completed.add(steps)
            result = child.poll()
            if result is not None:
                if result:
                    raise RuntimeError(f'Training exited with {result}; inspect its log')
                break
            time.sleep(5)
    summary = json.loads((RUN / 'summary.json').read_text())
    if summary['status'] != 'passed':
        raise RuntimeError('Training summary did not pass')
    evaluate(8388608, final=True)
    print('Approved continuation and all checkpoint evaluations complete', flush=True)


if __name__ == '__main__':
    main()
