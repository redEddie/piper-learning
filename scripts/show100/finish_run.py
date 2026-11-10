"""Wait for the fresh run, then evaluate its final checkpoint and render videos."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--train-pid',type=int,required=True)
a=ap.parse_args()
root=Path(__file__).resolve().parents[2]
run=root/'runs/show100/mlp-fresh-50m'
while True:
    try: os.kill(a.train_pid,0)
    except ProcessLookupError: break
    time.sleep(10)
summary=run/'summary.json'
if not summary.exists() or json.loads(summary.read_text())['status']!='passed':
    raise RuntimeError('fresh training did not complete successfully; evaluation not started')
checkpoint=run/'checkpoints/000050331648'
if not checkpoint.is_dir():raise FileNotFoundError(checkpoint)
print('Training complete; starting frozen final evaluation',flush=True)
subprocess.run([sys.executable,'-u','scripts/show100/evaluate.py','--run',str(run),'--checkpoint',str(checkpoint),
 '--output',str(root/'runs/show100/final-evaluation')],cwd=root,check=True)
print('New-policy evaluation complete; starting original baseline evaluation',flush=True)
release=Path('/home/pi_lab/robot/releases/thor-show76-v1-8x-20261023-r2')
subprocess.run([sys.executable,'-u','scripts/show100/evaluate.py','--run',str(release/'artifacts/training'),
 '--checkpoint',str(release/'artifacts/training/checkpoints/000083886080'),
 '--output',str(root/'runs/show100/baseline-evaluation'),'--modes','empty_hand'],cwd=root,check=True)
print('Evaluation complete; starting recorded-pose rendering',flush=True)
env=dict(os.environ,JAX_PLATFORMS='cpu')
subprocess.run([sys.executable,'-u','scripts/show100/render.py','--evaluation',str(root/'runs/show100/final-evaluation'),
 '--output',str(root/'runs/show100/videos')],cwd=root,env=env,check=True)
print('Final evaluation and videos complete',flush=True)
