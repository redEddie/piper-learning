"""Build an auditable summary; failed rollouts are not valid tracking data."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/qp-jerk';OUT.mkdir(exist_ok=True,parents=True)
BASE=ROOT/'runs/qp-jerk/cpu';raw=json.loads((BASE/'results.json').read_text());rows=[]
for limit in [None,100.,300.,1000.]:
    records=[r for r in raw['results'] if r['jerk_limit']==limit]
    synthetic=[r for r in records if not r['case'].startswith('replay')]
    replays=[r for r in records if r['case'].startswith('replay')]
    assert all(r['completed'] for r in synthetic)
    times=np.concatenate([np.load(BASE/f"{r['case']}-j{float(limit or 0)}.npz")['controller_ms'][20:] for r in synthetic])
    sine=[r for r in synthetic if r['case'].endswith('sine0.5')]
    rows.append(dict(jerk_limit_rad_s3=limit,cpu_ms_p50=float(np.median(times)),cpu_ms_p95=float(np.percentile(times,95)),cpu_ms_p99=float(np.percentile(times,99)),
        actual_lag_0_5hz_ms_mean=float(np.mean([r['actual_phase_lag_ms'] for r in sine])),
        reference_lag_0_5hz_ms_mean=float(np.mean([r['reference_phase_lag_ms'] for r in sine])),
        per_pose_sine={r['case']:dict(actual_lag_ms=r['actual_phase_lag_ms'],actual_gain=r['actual_gain']) for r in sine},
        maximum_reference_jerk=float(max(r['reference_jerk_max'] for r in synthetic)),
        synthetic_complete=sum(r['completed'] for r in synthetic),synthetic_cases=len(synthetic),
        replay_controller_complete=sum(r['completed'] for r in replays),replay_cases=len(replays),
        replay_failures=dict(Counter(r['failure']['reason'] for r in replays if r['failure'])),
        replay_failure_times_s=[r['failure']['time_s'] for r in replays if r['failure']] ))
for row in rows:
    row['delta_cpu_p50_ms']=row['cpu_ms_p50']-rows[0]['cpu_ms_p50']
    row['delta_lag_0_5hz_ms']=row['actual_lag_0_5hz_ms_mean']-rows[0]['actual_lag_0_5hz_ms_mean']

steps={}
for limit in [None,100.,300.,1000.]:
    trace=np.load(BASE/f'A-step-j{float(limit or 0)}.npz')
    y=trace['actual'][:,0]-trace['actual'][0,0];t=trace['t'];mask=t>=1.
    low=np.flatnonzero(mask&(y>=.003));high=np.flatnonzero(mask&(y>=.027));outside=np.flatnonzero(mask&(np.abs(y-.03)>.0006))
    steps[str(limit)]=dict(actual_final_gain=float(np.mean(y[-100:])/.03),actual_overshoot_percent=float(max(0.,y[mask].max()/.03-1)*100),
        actual_rise_10_90_ideal_ms=float((t[high[0]]-t[low[0]])*1000) if len(low) and len(high) else None,
        actual_settling_2percent_ideal_ms=float((t[outside[-1]+1]-1)*1000) if len(outside) and outside[-1]<len(t)-1 else (0. if not len(outside) else None))

trace=np.load(BASE/'replay-0-j100.0.npz');v=float(trace['v'][-1,4]);a=float(trace['a'][-1,4]);minimum=v+.005*a-100*.005**2
conflict=dict(case='replay-0',time_s=.24,joint=5,previous_velocity_rad_s=v,previous_acceleration_rad_s2=a,
    minimum_next_velocity_allowed_by_jerk=minimum,maximum_velocity_rad_s=1.5,
    explanation='The greedy controller reaches a state whose positive acceleration cannot decrease fast enough to respect both jerk and next-step speed. Anticipatory braking is required; no constraints were relaxed.')
gpu=json.loads((ROOT/'runs/qp-jerk/gpu/results.json').read_text());probe=json.loads((ROOT/'runs/qp-jerk/gpu/integration.json').read_text())
summary=dict(scope='Matched10Hz TCP inputs;200Hz offline controller; current SHOW100 geometry, no optimization training or hardware I/O',
    cpu_rows=rows,step_response_ideal_target=steps,conflict_example=conflict,
    gpu=gpu,integration_probe=probe,default_jerk_enabled=False,
    conclusion='Arithmetic overhead is small; merely adding a jerk box to the current single-step QP is not deployable on the replayed actions because it does not preserve future feasibility.',
    manifest=raw['manifest'],source_baseline_commit='229ea13',training=False,real_robot=False)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');colors=['#4263eb','#e8590c','#099268','#ae3ec9']
labels=['기존 QP','jerk 100','jerk 300','jerk 1,000']
fig,axes=plt.subplots(2,2,figsize=(13,9))
for limit,label,color in zip([None,100.,300.,1000.],labels,colors):
    d=np.load(BASE/f'A-step-j{float(limit or 0)}.npz');t=d['t'];y=d['actual'][:,0]-d['actual'][0,0]
    axes[0,0].plot(t,y,label=label,color=color)
    axes[0,1].plot(t,d['j'][:,0],label=label,color=color)
axes[0,0].axhline(.03,color='black',ls='--',label='목표 0.03rad');axes[0,0].set_xlim(.8,4.)
axes[0,0].set_title('급변 입력: 실제 관절1 위치',fontproperties=font);axes[0,0].set_ylabel('rad')
axes[0,1].set_xlim(.9,1.6);axes[0,1].set_title('같은 장면: 관절1 참조 jerk',fontproperties=font);axes[0,1].set_ylabel('rad/s³')
axes[1,0].bar(labels,[r['actual_lag_0_5hz_ms_mean'] for r in rows],color=colors)
axes[1,0].set_title('0.5Hz 실제 추종 위상 지연 · 두 자세 평균',fontproperties=font);axes[1,0].set_ylabel('ms')
axes[1,1].bar(labels,[r['cpu_ms_p50'] for r in rows],color=colors)
axes[1,1].set_title('CPU 제어 주기 연산시간 중앙값',fontproperties=font);axes[1,1].set_ylabel('ms')
for ax in axes.flat:
    ax.grid(alpha=.2)
    for tick in ax.get_xticklabels():tick.set_fontproperties(font)
for ax in axes[0]:ax.legend(prop=font)
fig.suptitle('jerk 제약 추가 비교 — 기록 동작은 기존12/12, jerk 적용은 모두0/12 완주',fontproperties=font)
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'comparison.png',dpi=150)
print(json.dumps(dict(rows=rows,conflict=conflict,steps=steps),ensure_ascii=False),flush=True)
