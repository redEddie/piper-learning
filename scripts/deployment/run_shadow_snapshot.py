#!/usr/bin/env python3
"""One snapshot: camera+passive CAN -> existing vision -> file-only RL proposals.

All results use a new directory. No robot SDK/enable/control command is invoked.
The original perception source/config/calibration are never edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--adapter', type=Path, default=Path.home() / 'robot/adapters/shadow-20261021-r1')
    ap.add_argument('--vision-project', type=Path, default=Path.home() / 'songjisu/show33')
    ap.add_argument('--perception-python', type=Path, default=Path.home() / 'songjisu/envs/perception/bin/python')
    ap.add_argument('--policy-python', type=Path, default=Path.home() / 'robot/envs/policy-r1/bin/python')
    ap.add_argument('--policies', type=Path, default=Path.home() / 'robot/releases/thor-20261021-r1/policies')
    ap.add_argument('--can', default='can_piper')
    args = ap.parse_args()
    if args.output.exists(): ap.error('Choose a new output directory')
    output = args.output.resolve()
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    records = []
    started = time.time_ns()

    def stage(name, command, acceptable=(0,), timeout=180):
        begin = time.time_ns()
        result = subprocess.run([str(x) for x in command], text=True, capture_output=True,
                                env=environment, timeout=timeout)
        record = {'name': name, 'command': [str(x) for x in command],
                  'started_wall_time_ns': begin, 'ended_wall_time_ns': time.time_ns(),
                  'duration_s': (time.time_ns() - begin) / 1e9, 'returncode': result.returncode}
        records.append(record)
        output.mkdir(parents=True, exist_ok=True)
        (output / f'{name}.log').write_text(result.stdout + result.stderr)
        if result.returncode not in acceptable:
            raise RuntimeError(f'{name} failed; inspect {name}.log')

    report = {'schema_version': 1, 'mode': 'single_camera_snapshot_diagnostic_dry_run',
              'hardware_commands_sent': False, 'control_ready': False, 'stages': records}
    try:
        stage('capture', [args.perception_python, args.adapter / 'capture_shadow_inputs.py',
                          '--output', output, '--can', args.can], timeout=40)
        shutil.copy2(args.adapter / 'task.json', output / 'task.json')
        shutil.copy2(args.vision_project / 'calib/T_base_cam.json', output / 'calibration_snapshot.json')
        camera = output / 'camera'
        common = ['--run', camera, '--objects', output / 'task.json']
        for script in ['segment', 'pose_geom', 'export_poses']:
            command = [args.perception_python, args.vision_project / f'scripts/{script}.py'] + common
            if script != 'segment': command += ['--calib', output / 'calibration_snapshot.json']
            if script == 'export_poses': command += ['--estimator', 'geometric']
            stage(script, command)
        policy_common = [args.policy_python, '-m', 'piper_policy_runtime.shadow',
                         '--vision', camera / 'object_poses.json', '--robot-state', output / 'robot_feedback.json',
                         '--calibration', output / 'calibration_snapshot.json',
                         '--model-xml', args.adapter / 'assets/piper.xml', '--policies', args.policies,
                         '--block-id', 'green_block', '--bowl-id', 'bowl', '--virtual-target-at-measured']
        stage('strict_gate', policy_common + ['--output', output / 'strict'], acceptable=(0, 2))
        stage('diagnostic', policy_common + ['--output', output / 'diagnostic', '--diagnostic-only'])
        diagnostic = json.loads((output / 'diagnostic/report.json').read_text())
        capture = json.loads((output / 'capture_report.json').read_text())
        report.update(status='diagnostic_completed', policy_evaluated=diagnostic['policy_evaluated'],
                      strict_status=json.loads((output / 'strict/report.json').read_text())['status'],
                      warnings=diagnostic['warnings'], snapshot_age_at_policy_s=diagnostic['snapshot_age_at_evaluation_s'],
                      capture_can_tx_packets_delta=capture['tx_packets_delta'])
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        report.update(status='failed', error=str(error))
    report['duration_s'] = (time.time_ns() - started) / 1e9
    output.mkdir(parents=True, exist_ok=True)
    (output / 'pipeline_report.json').write_text(json.dumps(report, indent=2) + '\n')
    manifest = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.rglob('*') if p.is_file()}
    (output / 'evidence-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report['status'] == 'failed': raise SystemExit(1)


if __name__ == '__main__': main()
