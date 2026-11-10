"""Fresh PPO run; preserve checkpoints, evaluate basic and qualified completion."""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/show103/balanced-fresh-16m'
EVAL=ROOT/'runs/show103/balanced-fresh'
LOG=ROOT/'runs/show103/logs'


def evaluate(label,checkpoint,final=False):
    with (LOG/('balanced-fresh-'+label+'.log')).open('w') as handle:
        subprocess.run([sys.executable,'-u','scripts/show103/evaluate.py',
            '--run',str(RUN),'--checkpoint',str(checkpoint),
            '--output',str(EVAL/label),'--seeds',
            *map(str,[2026,2027,2028] if final else [2026]),
            '--modes',*(['empty_hand','mixed','preinside','near'] if final else ['empty_hand','near'])],
            cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    report=json.loads((EVAL/label/'evaluation.json').read_text())
    print(json.dumps(dict(evaluation=label,aggregate=report['aggregate'])),flush=True)


def main():
    from piper_sim_lab.experiment_guard import require_idle_gpu
    require_idle_gpu()
    if RUN.exists():raise FileExistsError(RUN)
    doc=ROOT/'docs/show103/balanced-fresh';doc.mkdir(exist_ok=False)
    (doc/'protocol.json').write_text(json.dumps(dict(
        random_initialization=True,restored_policy=False,restored_value=False,
        restored_normalizer=False,seed=7,num_envs=1024,num_timesteps=16777216,
        config='configs/show103/balanced-fresh-16m.yaml',
        reference_checkpoint='none',reset_probs=[.4,.2,.2,.2],
        near_fraction=.05,real_robot=False,
        reward_audit='docs/show103/balanced-rewards/audit.json',
        scope='Reward/reset/controller settings fixed throughout the run.'),indent=2)+'\n')
    with (LOG/'balanced-fresh-training.log').open('w') as handle:
        train=subprocess.Popen([sys.executable,'-u','scripts/smooth_grip/train.py',
            '--config','configs/show103/balanced-fresh-16m.yaml','--output',str(RUN)],
            cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        print(json.dumps(dict(training_pid=train.pid,run=str(RUN),warm_start=None)),flush=True)
        while True:
            metric=RUN/'metrics.jsonl'
            if metric.exists():
                rows=[]
                for line in metric.read_text().splitlines():
                    try:rows.append(json.loads(line))
                    except json.JSONDecodeError:pass
                if rows and rows[-1].get('eval/episode_coord_reference_fault',0)>0:
                    train.terminate();train.wait()
                    raise RuntimeError('Own run stopped: reference constraint fault')
            # User preference: finish the full run before external evaluations.
            code=train.poll()
            if code is not None:
                if code:raise RuntimeError(f'Training failed:{code}')
                break
            time.sleep(5)
    evaluate('final-16m',RUN/'checkpoints/000016777216',final=True)
    print('Fresh training and final evaluations complete',flush=True)


if __name__=='__main__':main()
