"""Evaluate placing refinement after its training process has exited."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--train-pid',type=int,required=True);args=ap.parse_args()
root=Path(__file__).resolve().parents[2];run=root/'runs/show101/mlp-place-8m'
while True:
    try:os.kill(args.train_pid,0)
    except ProcessLookupError:break
    time.sleep(10)
if not(run/'summary.json').exists() or json.loads((run/'summary.json').read_text())['status']!='passed':
    raise RuntimeError('placing refinement training did not complete')
subprocess.run([sys.executable,'-u','scripts/show100/evaluate.py','--run',str(run),
 '--checkpoint',str(run/'checkpoints/000008388608'),'--output',str(root/'runs/show101/final-evaluation'),
 '--modes','empty_hand','mixed','preinside'],cwd=root,check=True)
env=dict(os.environ,JAX_PLATFORMS='cpu')
subprocess.run([sys.executable,'-u','scripts/show100/render.py','--evaluation',str(root/'runs/show101/final-evaluation'),
 '--output',str(root/'runs/show101/videos')],cwd=root,env=env,check=True)
print('SHOW-101 final evaluation and videos complete',flush=True)
