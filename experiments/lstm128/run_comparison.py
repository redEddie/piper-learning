"""Exclusive, cooperative queue. Never terminates or renices another process."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/lstm128'
ENV=dict(os.environ, PYTHONPATH=str(ROOT/'src'), PYTHONUNBUFFERED='1',
         XLA_PYTHON_CLIENT_PREALLOCATE='false',
         JAX_COMPILATION_CACHE_DIR=str(ROOT/'.cache/lstm-jax'),
         WARP_CACHE_PATH=str(ROOT/'.cache/lstm-warp'))
ENV.pop('LD_LIBRARY_PATH',None)
PYTHON='/data/envs/venv/piper-sim-lab/bin/python'


def pids():
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return [int(x.strip()) for x in result.splitlines() if x.strip().isdigit()]


def log(row):
    row['time_kst']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
    with (OUT/'queue.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def run(name,command):
    # A quiet minute avoids jumping into the gap between existing evaluation jobs.
    quiet=0
    while quiet<6:
        active=pids()
        quiet=quiet+1 if not active else 0
        log({'job':name,'status':'waiting_idle_gpu','pids':active,'quiet_samples':quiet})
        time.sleep(10)
    with (OUT/f'{name}.log').open('x') as f:
        process=subprocess.Popen([PYTHON,*command],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT)
        log({'job':name,'status':'started','pid':process.pid})
        while process.poll() is None:
            active=pids()
            with (OUT/'occupancy.jsonl').open('a') as h:
                h.write(json.dumps({'time':time.time(),'job':name,'own_pid':process.pid,'pids':active})+'\n')
            time.sleep(10)
        log({'job':name,'status':'finished','returncode':process.returncode})
        if process.returncode:raise SystemExit(process.returncode)


for kind in ['mlp','lstm']:
    run(f'train-{kind}', ['-m','piper_sim_lab.recurrent_train','--config','configs/pick_place_lstm_comparison.yaml',
        '--output',f'runs/matched-{kind}-s7','--actor',kind])
    run(f'eval-{kind}', ['-m','piper_sim_lab.recurrent_evaluate','--run',f'runs/matched-{kind}-s7',
        '--output',f'runs/matched-{kind}-s7/evaluation'])
log({'status':'all_jobs_complete'})
