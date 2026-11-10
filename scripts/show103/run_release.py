"""Validate, warm-start the preserved smooth policy, and evaluate SHOW-103."""
from pathlib import Path
import json,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/show103/release-v2-8m';LOG=ROOT/'runs/show103/logs';EVAL=ROOT/'runs/show103/release-v2'
SOURCE=ROOT/'runs/show103/refine-8m/checkpoints/000008388608'


def evaluate(label,checkpoint,seeds,modes):
    with (LOG/('release-v2-'+label+'.log')).open('w') as handle:
        subprocess.run([sys.executable,'-u','scripts/show103/evaluate.py','--run',str(RUN),
            '--checkpoint',str(checkpoint),'--output',str(EVAL/label),
            '--seeds',*map(str,seeds),'--modes',*modes],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    data=json.loads((EVAL/label/'evaluation.json').read_text())
    print(json.dumps(dict(evaluation=label,aggregate=data['aggregate'])),flush=True)


def main():
    if RUN.exists():raise FileExistsError(RUN)
    from piper_sim_lab.experiment_guard import require_idle_gpu
    require_idle_gpu()
    validation=json.loads((ROOT/'docs/show103/release-v2/integration.json').read_text())
    positive=json.loads((ROOT/'docs/show103/positive-release.json').read_text())
    returns=json.loads((ROOT/'docs/show103/release-v2/returns.json').read_text())
    assert validation['reset_stage_probs']==[.4,.2,.2,.2]
    assert validation['controller_failed_transitions']==validation['max_reference_speed_while_locked']==0
    assert positive['gentle_final_successes']>0 and returns['status']=='passed'
    print('Validated TCP50-60mm release, opening interlock, returns and1024 environments',flush=True)
    with (LOG/'release-training.log').open('w') as handle:
        train=subprocess.Popen([sys.executable,'-u','scripts/smooth_grip/train.py',
            '--config','configs/show103/release-8m.yaml','--output',str(RUN),
            '--warm-start',str(SOURCE)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        print(f'Training started pid={train.pid}',flush=True)
        while not (RUN/'config.json').exists():
            if train.poll() is not None:raise RuntimeError('Training failed before config')
            time.sleep(2)
        evaluate('frozen-0',SOURCE,[2026],['empty_hand','preinside','near'])
        done=set()
        while True:
            metrics=RUN/'metrics.jsonl'
            if metrics.exists():
                rows=[json.loads(x) for x in metrics.read_text().splitlines()]
                if rows and rows[-1].get('eval/episode_coord_reference_fault',0)>0:
                    train.terminate();train.wait()
                    raise RuntimeError('Reference constraint fault; stopped own training for inspection')
            for step in [2097152,4194304,6291456]:
                cp=RUN/'checkpoints'/f'{step:012d}'
                if step not in done and (cp/'commit_success.txt').exists():
                    evaluate(f'interim-{step}',cp,[2026],['empty_hand','near']);done.add(step)
            code=train.poll()
            if code is not None:
                if code:raise RuntimeError(f'Training failed:{code}')
                break
            time.sleep(5)
    summary=json.loads((RUN/'summary.json').read_text())
    if summary['status']!='passed':raise RuntimeError('No parameter update')
    evaluate('final-8m',RUN/'checkpoints/000008388608',[2026,2027,2028],
        ['empty_hand','mixed','preinside','near'])
    print('Training and final evaluations complete',flush=True)


if __name__=='__main__':main()
