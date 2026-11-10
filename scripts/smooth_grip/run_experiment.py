"""Run checked smooth-grip training, frozen evaluations and checkpoint reviews."""
from pathlib import Path
import json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/smooth-grip/refine-16m'
LOG=ROOT/'runs/smooth-grip/logs';SOURCE=ROOT/'runs/smooth-grip/bootstrap/000050331648'
def evaluate(label,checkpoint,seeds):
    out=ROOT/'runs/smooth-grip'/label
    with (LOG/(label+'.log')).open('w') as handle:
        subprocess.run([sys.executable,'-u','scripts/smooth_grip/evaluate.py','--run',str(RUN),
            '--checkpoint',str(checkpoint),'--output',str(out),'--seeds',*map(str,seeds),
            '--modes','empty_hand','mixed','preinside'],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    data=json.loads((out/'evaluation.json').read_text());print(json.dumps(dict(evaluation=label,aggregate=data['aggregate'])),flush=True)
    with (LOG/'experiment-evaluations.jsonl').open('a') as handle:handle.write(json.dumps(dict(evaluation=label,aggregate=data['aggregate']))+'\n')


def main():
    if RUN.exists():raise FileExistsError(RUN)
    from piper_sim_lab.experiment_guard import compute_pids
    deadline=time.monotonic()+600
    while compute_pids():
        if time.monotonic()>deadline:raise RuntimeError('GPU is still occupied; no training started')
        print('Waiting for existing GPU processes to finish; no processes changed',flush=True);time.sleep(15)
    with (LOG/'training.log').open('w') as log:
        train=subprocess.Popen([sys.executable,'-u','scripts/smooth_grip/train.py','--config','configs/smooth_grip/refine-16m.yaml',
            '--output',str(RUN),'--warm-start',str(SOURCE)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        print(f'Training started pid={train.pid}',flush=True)
        while not (RUN/'config.json').exists():
            if train.poll() is not None:raise RuntimeError('Training failed before config write')
            time.sleep(2)
        # This read-only comparison also records the exact resolved new config.
        evaluate('frozen-baseline',SOURCE,[2026,2027,2028])
        done=set()
        while True:
            metrics=RUN/'metrics.jsonl'
            if metrics.exists():
                rows=[json.loads(x) for x in metrics.read_text().splitlines()]
                if rows and rows[-1].get('eval/episode_coord_reference_fault',0)>0:
                    train.terminate();train.wait()
                    raise RuntimeError('A reference constraint invariant failed; training stopped for inspection')
            for step in [4194304,8388608,12582912]:
                cp=RUN/'checkpoints'/f'{step:012d}'
                if step not in done and (cp/'commit_success.txt').exists():
                    evaluate(f'interim-{step}',cp,[2026]);done.add(step)
            code=train.poll()
            if code is not None:
                if code:raise RuntimeError(f'Training failed with code{code}')
                break
            time.sleep(5)
    summary=json.loads((RUN/'summary.json').read_text())
    if summary['status']!='passed':raise RuntimeError('Training did not update the policy')
    evaluate('final-16m',RUN/'checkpoints/000016777216',[2026,2027,2028])
    print('Training and all evaluations complete',flush=True)


if __name__=='__main__':main()
