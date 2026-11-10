"""Create honest pilot results and actual-state simulation images after the queue."""
from pathlib import Path
import json
import os
os.environ['MPLBACKEND']='Agg'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['MUJOCO_GL']='egl'
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/lstm128/results'
OUT.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
report={'scope':'one independently trained seed (7) per actor; 384 paired evaluation episodes per condition',
        'models':{},'paired_comparisons':{},'existing_brax_mlp_reference':read(ROOT/'experiments/lstm128/existing-mlp-reference.json')}
for kind in ['mlp','lstm']:
    run=ROOT/f'runs/matched-{kind}-s7'
    report['models'][kind]={'training':read(run/'summary.json'),'evaluation':read(run/'evaluation/evaluation.json')}
for condition in ['dr','nodr']:
    vals={}
    for kind in ['mlp','lstm']:
        rows=next(x for x in report['models'][kind]['evaluation']['conditions'] if x['condition']==condition)
        vals[kind]=np.concatenate([x['final_success_per_episode'] for x in rows['by_seed']])
    mlp_eval=next(x for x in report['models']['mlp']['evaluation']['conditions'] if x['condition']==condition)
    lstm_eval=next(x for x in report['models']['lstm']['evaluation']['conditions'] if x['condition']==condition)
    assert mlp_eval['environment']==lstm_eval['environment'], 'Evaluation environments differ'
    assert [r['initial_scene_sha256'] for r in mlp_eval['by_seed']]==[r['initial_scene_sha256'] for r in lstm_eval['by_seed']], 'Initial scenes differ'
    a,b=vals['mlp'],vals['lstm']
    report['paired_comparisons'][condition]={'lstm_minus_mlp_percentage_points':float(100*(b.mean()-a.mean())),
        'both_succeed':int((a&b).sum()),'only_mlp':int((a&~b).sum()),'only_lstm':int((~a&b).sum()),
        'both_fail':int((~a&~b).sum()),'episodes':len(a)}
mlp=report['models']['mlp']['training'];lstm=report['models']['lstm']['training']
report['lstm_to_mlp_steady_state_throughput_ratio']=lstm['steady_state_sps']/mlp['steady_state_sps']
occupancy=[json.loads(x) for x in (ROOT/'experiments/lstm128/occupancy.jsonl').read_text().splitlines()]
overlap=[x for x in occupancy if any(p!=x['own_pid'] for p in x['pids'])]
queue=[json.loads(x) for x in (ROOT/'experiments/lstm128/queue.jsonl').read_text().splitlines()]
from datetime import datetime
for kind in ['mlp','lstm']:
    started=next(x for x in queue if x.get('job')==f'train-{kind}' and x.get('status')=='started')
    ended=next(x for x in queue if x.get('job')==f'train-{kind}' and x.get('status')=='finished')
    report['models'][kind]['training']['queue_process_walltime_seconds']=(datetime.fromisoformat(ended['time_kst'])-datetime.fromisoformat(started['time_kst'])).total_seconds()
report['occupancy']={'sample_interval_seconds':10,'external_overlap_samples':overlap,
    'note':'Read-only checks; trainer yields before next iteration if an external PID appears.'}
(OUT/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

fig,axs=plt.subplots(1,3,figsize=(14,4.6))
for j,condition in enumerate(['dr','nodr']):
    ax=axs[j]; ys=[];errs=[]
    for kind in ['mlp','lstm']:
        r=next(x for x in report['models'][kind]['evaluation']['conditions'] if x['condition']==condition)
        ys.append(100*r['final_success_rate']);lo,hi=r['wilson_95_interval']
        errs.append([max(0.,ys[-1]-lo*100),max(0.,hi*100-ys[-1])])
    ax.bar(['MLP','LSTM-128'],ys,color=['#377eb8','#ff7f00'],yerr=np.array(errs).T,capsize=5)
    for i,y in enumerate(ys):ax.text(i,y+5,f'{y:.1f}%',ha='center')
    ax.set(ylim=(0,108),ylabel='Final success (%)',title=f'{condition.upper()}: empty-hand, 384 episodes')
    ref=next(x for x in report['existing_brax_mlp_reference']['conditions'] if x['condition']==condition)
    ax.axhline(100*ref['final_success_rate'],color='#555555',linestyle='--',label='Existing Brax MLP (different learner)')
    ax.legend(fontsize=8,loc='lower right')
    ax.grid(axis='y',alpha=.2)
sps=[report['models'][k]['training']['steady_state_sps'] for k in ['mlp','lstm']]
axs[2].bar(['MLP','LSTM-128'],sps,color=['#377eb8','#ff7f00'])
for i,y in enumerate(sps):axs[2].text(i,y,f'{y:,.0f}',ha='center',va='bottom')
axs[2].set(ylabel='Environment transitions / second',title='Collection + PPO update; JIT excluded')
fig.suptitle('Matched 16.78M-step pilot | one training seed per actor | error bars: episode Wilson 95% CI')
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(11,4))
for kind in ['mlp','lstm']:
    rows=[json.loads(x) for x in (ROOT/f'runs/matched-{kind}-s7/metrics.jsonl').read_text().splitlines()]
    x=np.array([r['steps']/1e6 for r in rows]);reward=np.array([r['rollout_reward_mean'] for r in rows])
    size=32; xx=x[:len(x)//size*size].reshape(-1,size).mean(1)
    yy=reward[:len(x)//size*size].reshape(-1,size).mean(1)
    axs[0].plot(xx,yy,label=kind.upper())
    durations=np.array([r['collection_seconds']+r['update_seconds'] for r in rows[1:]])
    axs[1].plot(x[1:],durations,alpha=.5,label=kind.upper())
axs[0].set(xlabel='Training transitions (million)',ylabel='Mean reward per control step',title='Training only: mixed-stage resets (32-update mean)')
axs[1].set(xlabel='Training transitions (million)',ylabel='Seconds per 16,384-transition update',title='Synchronized execution; first iteration excluded')
for a in axs:a.legend();a.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'learning-curves.png',dpi=150);plt.close(fig)

from piper_sim_lab.tasks import make_env
import mujoco
fig,axs=plt.subplots(2,3,figsize=(12,6.8))
for row,kind in enumerate(['mlp','lstm']):
    run=ROOT/f'runs/matched-{kind}-s7'
    cfg=read(run/'config.json');ecfg=dict(cfg['resolved_environment'],impl='jax',batch_capacity=1)
    env=make_env(ecfg);model=env.mj_model;data=mujoco.MjData(model)
    tr=np.load(run/'evaluation/trajectory-dr-s2026.npz')
    # Prefer a success among captured episodes; otherwise explicitly show a failed one.
    candidates=np.flatnonzero(tr['success'][-1,:tr['qpos'].shape[1]])
    idx=int(candidates[0]) if len(candidates) else 0
    success=bool(tr['success'][-1,idx]) and not bool(tr['failed'][-1,idx])
    camera=mujoco.MjvCamera();camera.lookat[:]=[.15,0,.25]
    camera.distance=1.25;camera.azimuth=135;camera.elevation=-25
    with mujoco.Renderer(model,height=360,width=480) as renderer:
        for col,t in enumerate([0,39,79]):
            data.qpos[:]=tr['qpos'][t,idx];data.mocap_pos[:]=tr['mocap_pos'][t,idx]
            mujoco.mj_forward(model,data);renderer.update_scene(data,camera)
            axs[row,col].imshow(renderer.render());axs[row,col].axis('off')
            axs[row,col].set_title(f'{kind.upper()} | t={(t+1)*.1:.1f}s | episode {idx}'+
                (f" | {'success' if success else 'failure'}" if col==2 else ''))
fig.suptitle('Actual policy trajectories | DR evaluation seed 2026 | one selected episode per actor')
fig.tight_layout();fig.savefig(OUT/'policy-rollouts.png',dpi=150);plt.close(fig)
print(json.dumps({'result_directory':str(OUT),'paired':report['paired_comparisons'],
    'steady_state_sps':{'mlp':mlp['steady_state_sps'],'lstm':lstm['steady_state_sps']}}))
