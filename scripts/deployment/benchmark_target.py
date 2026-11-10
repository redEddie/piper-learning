"""Benchmark exported NumPy policies on a target, without hardware access.

Uses synthetic export replay inputs, not camera/robot observations. Writes all
latency samples and action traces so summary statistics can be reproduced.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from piper_policy_runtime import Policy
from piper_policy_runtime.check import check


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(samples):
    return {
        'count': len(samples),
        'p50': float(np.quantile(samples, .50)),
        'p95': float(np.quantile(samples, .95)),
        'p99': float(np.quantile(samples, .99)),
        'max': float(np.max(samples)),
        'mean': float(np.mean(samples)),
    }


def action_trace(policy, observations, resets):
    """Replay one slot; retain the original episode reset markers."""
    policy.reset(1)
    config = policy.metadata['environment']
    trace = []
    for step, (obs, reset) in enumerate(zip(observations, resets)):
        action = policy.step(obs, bool(reset))
        trace.append({
            'step': step, 'reset': bool(reset), 'normalized_action': action.tolist(),
            'proposed_base_linear_velocity_m_s': (action[:3] * config['max_lin_vel']).tolist(),
            'proposed_base_angular_velocity_rad_s': (action[3:6] * config['max_ang_vel']).tolist(),
            'gripper_intent': ('open' if action[6] > 0 else 'close') if config['gripper_binary'] else 'continuous',
            'hardware_command_sent': False,
        })
    return trace


def benchmark(release, output, rounds=5, samples=2000, warmup=256):
    if rounds < 1 or samples < 2 or warmup < 0:
        raise ValueError('Invalid benchmark size')
    output.mkdir(parents=True, exist_ok=False)
    names = json.loads((release / 'release.json').read_text())['policies']
    if not names or len(set(names)) != len(names):
        raise ValueError('Release must list distinct exported policies')
    policies, inputs, results = {}, {}, {}
    for name in names:
        directory = release / 'policies' / name
        policy = policies[name] = Policy(directory)
        with np.load(directory / 'golden.npz', allow_pickle=False) as data:
            obs, resets = data['observations'][:, 0].copy(), data['resets'][:, 0].copy()
        inputs[name] = obs, resets
        trace = action_trace(policy, obs, resets)
        (output / f'{name}-synthetic-actions.json').write_text(json.dumps({
            'input_source': 'synthetic golden export vectors, slot 0',
            'not_a_closed_loop_rollout': True, 'actions': trace,
        }, indent=2) + '\n')
        results[name] = {
            'parity': check(directory),
            'weights_sha256': policy.metadata['weights_sha256'],
            'policy_json_sha256': digest(directory / 'policy.json'),
            'golden_sha256': digest(directory / 'golden.npz'),
            'training_steps': policy.metadata['training_step'],
            'rounds': [],
        }
    # Rotate order to reduce a fixed warm-cache/order advantage. No affinity,
    # performance mode, GPU clocks, or concurrent process settings are changed.
    for round_id in range(rounds):
        order = names[round_id % len(names):] + names[:round_id % len(names)]
        for name in order:
            policy = policies[name]
            obs, resets = inputs[name]
            policy.reset(1)
            for step in range(warmup):
                index = step % len(obs)
                policy.step(obs[index], bool(resets[index]) or index == 0)
            times = []
            for step in range(samples):
                index = step % len(obs)
                reset = bool(resets[index]) or index == 0
                start = time.perf_counter_ns()
                policy.step(obs[index], reset)
                times.append((time.perf_counter_ns() - start) / 1e6)
            results[name]['rounds'].append({'round': round_id, 'order': order.index(name),
                                           'latency_ms': summary(times), 'samples_ms': times})
    for result in results.values():
        result['aggregate_latency_ms'] = summary([t for r in result['rounds'] for t in r['samples_ms']])
    forbidden = ['jax', 'mujoco', 'torch', 'piper_sdk', 'can', 'pyrealsense2']
    imported = [name for name in forbidden if name in sys.modules]
    if imported:
        raise AssertionError(f'Unexpected hardware/training imports: {imported}')
    report = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'machine': platform.machine(),
        'python': platform.python_version(), 'python_executable': sys.executable,
        'numpy': np.__version__, 'cpu_count': os.cpu_count(),
        'thread_settings': {name: os.environ.get(name) for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
        'script_sha256': digest(__file__), 'release_path': str(release.resolve()),
        'release_manifest_sha256': digest(release / 'release.json'),
        'batch_size': 1, 'warmup_per_round': warmup, 'samples_per_round': samples,
        'rounds': rounds, 'timing_scope': 'warm NumPy Policy.step only; includes normalization and recurrent state update; excludes input construction, model loading, vision, IK, SDK, scheduling and I/O',
        'input_source': 'synthetic golden export vectors; not live camera or robot data',
        'hardware_commands_sent': False, 'forbidden_imports_found': imported,
        'policies': results,
    }
    (output / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    return {name: {'parity_error': r['parity']['max_action_abs_error'],
                   'latency_ms': r['aggregate_latency_ms']} for name, r in results.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--warmup', type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(benchmark(args.release, args.output, args.rounds, args.samples, args.warmup), indent=2))
