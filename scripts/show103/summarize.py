"""Summarize actual learning separately from synthetic incentive validation."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['support','release','stable'],default='support');args=ap.parse_args()
DOC=ROOT/'docs/show103';RUN=ROOT/'runs/show103'
if args.variant=='release':DOC=DOC/'release-v2';RUN=RUN/'release-v2'
if args.variant=='stable':DOC=DOC/'stable-v3';RUN=RUN/'stable-v3'
impact_limit=.1 if args.variant=='support' else 1.
sources=[('frozen-0',0),('interim-2097152',2097152),('interim-4194304',4194304),
    ('interim-6291456',6291456),('final-8m',8388608)]
if args.variant=='stable':sources=[('frozen-0',0),('interim-2097152',2097152),('final-4m',4194304)]
results={};points=[]
for label,steps in sources:
    path=RUN/label/'evaluation.json'
    if not path.exists():continue
    data=json.loads(path.read_text());modes={};paired={}
    for mode,a in data['aggregate'].items():
        rows=[(k,r) for k,r in data['reports'].items() if k.startswith(mode+'-')]
        heights=[];tcp_heights=[];speeds=[];post_landing=0;actual_gentle_before=0;faults={}
        for name,r in rows:
            with np.load(RUN/label/(name+'-trace.npz')) as tr:
                valid=~(tr['controller_failed'].astype(bool)|tr['nonfinite'].astype(bool)|tr['overflow'].astype(bool)).any(0)
                released=tr['place_released'][-1].astype(bool)&valid
                floors=tr['place_floor_seen'][-1].astype(bool)&valid
                heights.extend(tr['place_release_height'][-1,released]*1000)
                if 'release_tcp_height' in tr:tcp_heights.extend(tr['release_tcp_height'][-1,released]*1000)
                speeds.extend(tr['place_impact_peak'][-1,floors])
                gt=tr['gentle_trace'].transpose(1,0,2,3).reshape(len(valid),-1,10)
                landed=np.maximum.accumulate(gt[:,:,2]>0,axis=1)
                released_or_early=(gt[:,:,5]>0)|(gt[:,:,6]>0)
                touching=(gt[:,:,7]>0)|(gt[:,:,8]>0)
                post_landing+=int(((landed&released_or_early&touching).any(1)&valid).sum())
                actual_gentle_before+=int((tr['place_qualified'][-1]&~tr['place_premature'][-1]&valid&(tr['place_impact_peak'][-1]<=impact_limit)).sum())
            for key,n in r['overall']['failures'].items():faults[key]=faults.get(key,0)+n
        mode_result=dict(a,qualified_release=sum(r['placing']['qualified_release_episodes'] for _,r in rows),
            gentle_before_withdrawal=actual_gentle_before,
            post_release_recontact_episodes=sum(r['placing']['recontact_episodes'] for _,r in rows),
            post_landing_recontact_episodes=post_landing,
            release_height_mm_median=float(np.median(heights)) if heights else None,
            release_height_mm_mean=float(np.mean(heights)) if heights else None,
            release_tcp_height_mm_mean=float(np.mean(tcp_heights)) if tcp_heights else None,
            release_tcp_height_mm_median=float(np.median(tcp_heights)) if tcp_heights else None,
            release_tcp_in_band=int(sum(50<=v<=60 for v in tcp_heights)),
            precontact_downward_speed_m_s_median=float(np.median(speeds)) if speeds else None,
            failures=faults,max_reference_jerk=max(r['coordinator']['max_reference_jerk'] for _,r in rows),
            gripper_timeouts=sum(r['coordinator']['timeouts'] for _,r in rows),
            max_reference_speed_during_grip=max(r['coordinator']['max_reference_speed_during_grip'] for _,r in rows))
        mode_result['opening_diagnostics']={str(r['seed']):r['opening_exploration'] for _,r in rows if 'opening_exploration' in r}
        modes[mode]=mode_result
        first=data['reports'][mode+'-s2026']
        paired[mode]=dict(basic=first['overall']['final_success_rate'],gentle=first['placing']['gentle_final_success_rate'],
            height=first['placing']['detected_release_height_mm_mean'] if args.variant=='support' else first['placing']['release_tcp_height_mm_mean'],recontact=first['placing']['recontact_episodes']/128)
    results[label]=modes;points.append(dict(additional_steps=steps,paired_seed2026=paired))
protocol=json.loads((DOC/'protocol.json').read_text());source=Path(protocol['source_checkpoint'])
changed=[name for name,h in protocol['source_checkpoint_sha256'].items() if hashlib.sha256((source/name).read_bytes()).hexdigest()!=h]
manifest=json.loads((ROOT/'docs/show100/source-manifest.json').read_text())
changed_source=[n for n,h in manifest['files'].items() if hashlib.sha256((Path(manifest['source'])/n).read_bytes()).hexdigest()!=h]
assert not changed and not changed_source
out=dict(scope='MLP,1024 training envs; same20s horizon. Basic inside and target-qualified release+contact-free withdrawal are distinct. Intermediate128 and final384 per mode; learning curves use paired seed2026/128.',
    target='support-before-release' if args.variant=='support' else 'TCP50-60mm above bowl inner floor',
    results=results,points=points,preservation=dict(source_checkpoint_changed_files=changed,original_source_changed_files=changed_source,original_source_files_checked=len(manifest['files'])),real_robot=False)
(DOC/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,axes=plt.subplots(2,2,figsize=(12,8))
for mode,label,color in [('empty_hand','A 빈손','#4263eb'),('near','낮은 놓기 초기화','#e8590c')]:
    subset=[p for p in points if mode in p['paired_seed2026']];x=[p['additional_steps']/1e6 for p in subset]
    for ax,key,scale in [(axes[0,0],'basic',100),(axes[0,1],'gentle',100),(axes[1,0],'height',1),(axes[1,1],'recontact',100)]:
        ax.plot(x,[p['paired_seed2026'][mode][key]*scale for p in subset],'-o',color=color,label=label)
for ax,title in zip(axes.flat,['일반 완료율 (%)','목표 해제·철수 완료율 (%)','평균 해제 '+('큐브 바닥' if args.variant=='support' else 'TCP')+' 높이 (mm)','해제 후 재접촉 에피소드 (%)']):
    ax.set_title(title,fontproperties=font);ax.set_xlabel('추가 학습 전이 (백만)',fontproperties=font);ax.grid(alpha=.2);ax.legend(prop=font)
axes[0,0].set_ylim(0,102);axes[0,1].set_ylim(0,102);axes[1,1].set_ylim(0,102)
fig.suptitle('SHOW-103 · 동일 seed2026 · 조건별128회 · 20초',fontproperties=font);fig.tight_layout();fig.savefig(DOC/'learning.png',dpi=150)
print(json.dumps(results,ensure_ascii=False),flush=True)
