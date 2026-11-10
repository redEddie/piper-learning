"""Summarize equal20s frozen evaluations and the new-controller learning curve."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[2];DOC=ROOT/'docs/smooth-grip'
original=json.loads((ROOT/'runs/smooth-grip/original-20s/evaluation.json').read_text())
base_path=ROOT/'runs/smooth-grip/frozen-baseline/evaluation.json'
base=json.loads(base_path.read_text()) if base_path.exists() else None
points=[];all_results={}
sources=[('original-20s',None),('frozen-baseline',0),('interim-4194304',4194304),
    ('interim-8388608',8388608),('candidate-8m',None),('interim-12582912',12582912),('final-16m',16777216)]
for name,steps in sources:
    path=ROOT/'runs/smooth-grip'/name/'evaluation.json'
    if not path.exists():continue
    d=json.loads(path.read_text());mode_results={}
    for mode,a in d['aggregate'].items():
        rows=[r for k,r in d['reports'].items() if k.startswith(mode+'-')]
        n=a['episodes'];result=dict(a,success_ever=sum(r['success_ever'] for r in rows),success_at8s=sum(r['success_at_8s'] for r in rows),
            failures={k:sum(r['overall']['failures'][k] for r in rows) for k in rows[0]['overall']['failures']},
            same_initial_scenes=all(r['initial_scene_sha256']==original['reports'][f"{mode}-s{r['seed']}"]['initial_scene_sha256'] for r in rows))
        coordinators=[r['coordinator'] for r in rows if 'coordinator' in r]
        if coordinators:
            result['coordinator']=dict(max_reference_jerk=max(r['max_reference_jerk'] for r in coordinators),
                max_reference_speed=max(r['max_reference_speed'] for r in coordinators),
                max_reference_acceleration=max(r['max_reference_acceleration'] for r in coordinators),
                max_reference_speed_during_grip=max(r['max_reference_speed_during_grip'] for r in coordinators),
                operations_completed=sum(r['operations_completed'] for r in coordinators),
                rejected_proposals=sum(r['rejected_proposals'] for r in coordinators),timeouts=sum(r['timeouts'] for r in coordinators))
        mode_results[mode]=result
    all_results[name]=mode_results
    if steps is not None:
        paired={mode:d['reports'][mode+'-s2026']['overall']['final_success_rate'] for mode in ['empty_hand','mixed','preinside']}
        points.append(dict(additional_steps=steps,paired128_rates=paired,aggregate=d['aggregate']))
bootstrap=json.loads((DOC/'bootstrap.json').read_text());source=Path(bootstrap['source'])
changed=[name for name,h in bootstrap['source_sha256'].items() if hashlib.sha256((source/name).read_bytes()).hexdigest()!=h]
repeatability=None
repeat_path=ROOT/'runs/smooth-grip/candidate-8m/empty_hand-s2026.json'
if repeat_path.exists():
    before=json.loads((ROOT/'runs/smooth-grip/interim-8388608/empty_hand-s2026.json').read_text())
    after=json.loads(repeat_path.read_text())
    with np.load(ROOT/'runs/smooth-grip/interim-8388608/empty_hand-s2026-trace.npz') as a,np.load(repeat_path.with_name('empty_hand-s2026-trace.npz')) as b:
        repeatability=dict(checkpoint_additional_transitions=8388608,mode='empty_hand',seed=2026,episodes=128,
            same_initial_qpos_sha256=before['initial_scene_sha256']==after['initial_scene_sha256'],
            successes=[before['overall']['final_successes'],after['overall']['final_successes']],
            first_action_max_difference=float(np.max(np.abs(a['action'][0]-b['action'][0]))),
            first_step_actual_qpos_max_difference=float(np.max(np.abs(a['qpos'][0]-b['qpos'][0]))),
            note='Frozen policy and identical first action/initial poses; tiny GPU physics differences grow through contact. Fixed seeds do not give bitwise-repeatable trajectories.')
out=dict(scope='All episodes20s; old SHOW100 published8s rate is not substituted into this comparison. Intermediate learning curve uses same128 seed2026 scenes, final tables384 scenes per mode.',
    results=all_results,points=points,repeatability=repeatability,source_checkpoint_changed_files=changed,real_robot=False)
(DOC/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');fig,ax=plt.subplots(figsize=(10,5.6))
for mode,label,color in [('empty_hand','빈손 A','#4263eb'),('mixed','혼합 50/20/20/10','#099268'),('preinside','놓기 직전','#e8590c')]:
    ax.plot([p['additional_steps']/1e6 for p in points],[p['paired128_rates'][mode]*100 for p in points],'-o',label=label,color=color)
ax.set_xlabel('SHOW-100에서 추가학습 전이 (백만)',fontproperties=font);ax.set_ylabel('20초 종료 전체 완료율 (%)',fontproperties=font)
ax.set_title('새 제어기 학습 경과 · 동일 seed2026 · 시작 조건별128회',fontproperties=font);ax.set_ylim(0,102);ax.grid(alpha=.2);ax.legend(prop=font)
fig.tight_layout();fig.savefig(DOC/'learning-curve.png',dpi=150)
print(json.dumps({k:{m:r['final_success_rate'] for m,r in v.items()} for k,v in all_results.items()},ensure_ascii=False),flush=True)
