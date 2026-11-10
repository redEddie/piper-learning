"""Unattended fresh training and final first-episode evaluation, simulation only."""
import argparse,hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--name',default='qp-v1-a5-j100')
parser.add_argument('--config',type=Path,default=ROOT/'configs/show103/qp-v1-a5-j100-fresh-50m.yaml')
parser.add_argument('--preflight',action='store_true')
args=parser.parse_args()
if not re.fullmatch(r'[a-z0-9-]+',args.name):parser.error('name must be lowercase letters, numbers and hyphens')
NAME=args.name
CONFIG=args.config.resolve()
CFG=yaml.safe_load(CONFIG.read_text())
STEPS=int(CFG['training']['num_timesteps'])
RUN=ROOT/f'runs/show103/{NAME}-fresh-50m'
EVAL=ROOT/f'runs/show103/{NAME}-final-evaluation'
DOC=ROOT/f'docs/show103/{NAME}'
STATUS=DOC/'status.json'
LOG=ROOT/'runs/show103/logs'


def write_status(**fields):
    record=dict(updated_unix=time.time(),**fields)
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(STATUS)


def main():
    from piper_sim_lab.experiment_guard import require_idle_gpu
    require_idle_gpu()
    if RUN.exists() or EVAL.exists():raise FileExistsError('Fresh run/evaluation paths must be unused')
    DOC.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    if args.preflight:
        if CFG['environment']['task']=='direct_gripper_release':
            validation_jobs=[('direct-controller','scripts/show103/validate_direct_gripper.py',
                ['--num-envs','1024','--steps','240','--doc-dir',str(DOC)])]
        else:
            validation_jobs=[
                ('gripper-switch','scripts/show103/validate_gripper_switch.py',
                    ['--output',str(DOC/'gripper-switch.json')]),
                ('controller','scripts/show103/validate.py',
                    ['--num-envs','1024','--steps','240','--tag','controller-validation','--doc-dir',str(DOC)])]
        validation_jobs.append(('task-failure','scripts/show103/validate_task_failure.py',
                               ['--doc-dir',str(DOC/'task-failure')]))
        for label,script,extra in validation_jobs:
            with (LOG/f'{NAME}-{label}-validation.log').open('x') as handle:
                child=subprocess.Popen([sys.executable,'-u',script,'--config',str(CONFIG),*extra],
                    cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
                write_status(phase='validation',validation_step=label,validation_pid=child.pid,
                    config=str(CONFIG),real_robot=False)
                code=child.wait()
                if code:
                    write_status(phase='failed',error=f'{label} validation failed: {code}',real_robot=False)
                    raise RuntimeError(f'{label} validation failed: {code}')
    # Validation was performed with exactly this config before launching us.
    validation=json.loads((DOC/'controller-validation.json').read_text())
    task_validation=json.loads((DOC/'task-failure/integration.json').read_text())
    assert validation['environments']==1024
    assert validation['controller_failed_transitions']==validation['nonfinite']==validation['overflow']==0
    assert task_validation['controller_limits']==dict(velocity=1.,acceleration=5.,jerk=100.)
    assert task_validation['all_failures_reset_immediately'] and task_validation['max_settlement_error']<1e-4
    config_hash=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    if args.preflight:
        checked_reports=[validation,task_validation]
        if CFG['environment']['task']=='direct_gripper_release':
            assert validation['status']=='passed' and all(validation['checks'].values())
        else:
            grip_validation=json.loads((DOC/'gripper-switch.json').read_text())
            checked_reports.append(grip_validation)
            assert grip_validation['close_accepted']==grip_validation['open_accepted']==grip_validation['empty_environments']
            assert grip_validation['held_open_accepted']==grip_validation['held_width_increased10mm']==grip_validation['held_environments']
        for checked in checked_reports:
            assert checked['config_sha256']==config_hash
    protocol=dict(random_initialization=True,restored_policy=False,restored_value=False,restored_normalizer=False,
        num_envs=1024,num_timesteps=STEPS,seed=CFG['seed'],
        config=str(CONFIG),config_sha256=config_hash,
        reference_limits=dict(velocity_rad_s=1.,acceleration_rad_s2=5.,jerk_rad_s3=100.),
        action_scales=dict(linear_m_s=CFG['environment']['max_lin_vel'],angular_rad_s=CFG['environment']['max_ang_vel']),reset_probs=[.4,.2,.2,.2],
        bounded_base_potential=CFG['environment'].get('bounded_base_potential',False),
        unfinished_cost_per_second=CFG['environment'].get('unfinished_cost_per_second',False),
        near_fraction=.05,grasp_deadline_s=12.,task=CFG['environment']['task'],
        independent_gripper=CFG['environment']['task']=='direct_gripper_release',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        training_log=str(LOG/f'{NAME}-training.log'),evaluation_output=str(EVAL),
        prior_runs_preserved=True,real_robot=False,plane_modified=False,
        checkpoint_policy='Save about every8.39m. After final evaluation keep final and preceding checkpoints from this run; never delete prior experiments.')
    (DOC/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    started=time.monotonic();train=None
    try:
        with (LOG/f'{NAME}-training.log').open('x') as handle:
            train=subprocess.Popen([sys.executable,'-u','scripts/smooth_grip/train.py',
                '--config',str(CONFIG),'--output',str(RUN)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
            print(json.dumps(dict(training_pid=train.pid,run=str(RUN),warm_start=None)),flush=True)
            while True:
                metric=RUN/'metrics.jsonl';rows=[]
                if metric.exists():
                    for line in metric.read_text().splitlines():
                        try:rows.append(json.loads(line))
                        except json.JSONDecodeError:pass
                latest=rows[-1] if rows else {}
                if latest.get('eval/episode_coord_reference_fault',0)>0:
                    train.terminate();train.wait();raise RuntimeError('Reference constraint fault in own training run')
                write_status(phase='training',training_pid=train.pid,run=str(RUN),
                    elapsed_seconds=time.monotonic()-started,latest=latest,target_transitions=STEPS)
                code=train.poll()
                if code is not None:
                    if code:raise RuntimeError(f'Training failed: {code}')
                    break
                time.sleep(10)
        summary=json.loads((RUN/'summary.json').read_text())
        assert summary['policy_changed'] and summary['status']=='passed'
        checkpoint=RUN/f'checkpoints/{STEPS:012d}'
        assert (checkpoint/'commit_success.txt').exists()
        write_status(phase='evaluation',checkpoint=str(checkpoint),elapsed_seconds=time.monotonic()-started)
        with (LOG/f'{NAME}-evaluation.log').open('x') as handle:
            evaluation=subprocess.Popen([sys.executable,'-u','scripts/show103/evaluate_quadratic.py',
                '--run',str(RUN),'--checkpoint',str(checkpoint),'--output',str(EVAL)],
                cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
            write_status(phase='evaluation',evaluation_pid=evaluation.pid,checkpoint=str(checkpoint),
                         elapsed_seconds=time.monotonic()-started)
            code=evaluation.wait()
            if code:raise RuntimeError(f'Evaluation failed: {code}')
        report=json.loads((EVAL/'evaluation.json').read_text())
        shutil.copyfile(EVAL/'evaluation.json',DOC/'evaluation.json')
        checkpoints=sorted(p for p in (RUN/'checkpoints').iterdir() if p.is_dir() and p.name.isdigit())
        removed=[]
        for old in checkpoints[:-2]:
            # Strictly scoped to this newly created run, after final evaluation.
            assert old.parent==RUN/'checkpoints'
            shutil.rmtree(old);removed.append(str(old))
        write_status(phase='completed',aggregate=report['aggregate'],checkpoint=str(checkpoint),
            training_summary=summary,removed_own_intermediate_checkpoints=removed,
            elapsed_seconds=time.monotonic()-started,real_robot=False,plane_modified=False)
        print(json.dumps(dict(completed=True,aggregate=report['aggregate'])),flush=True)
    except BaseException as exc:
        if train is not None and train.poll() is None:
            train.terminate();train.wait()
        write_status(phase='failed',error=repr(exc),elapsed_seconds=time.monotonic()-started,real_robot=False)
        raise


if __name__=='__main__':main()
