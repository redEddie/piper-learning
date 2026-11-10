"""Report the fixed episode-wide placing run against its SHOW-100 source."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[2];DOCS=ROOT/'docs/show101'
RUN=ROOT/'runs/show101/mlp-place-v3-episode-8m'
baseline_path=ROOT/'runs/show101/baseline-episode-v3-evaluation/evaluation.json'
baseline=json.loads(baseline_path.read_text())
points=[];final=None
for steps,name in [(0,'baseline-episode-v3-evaluation'),
    (2097152,'interim-episode-v3-2097152-evaluation'),
    (4194304,'interim-episode-v3-4194304-evaluation'),
    (6291456,'interim-episode-v3-6291456-evaluation'),
    (8388608,'final-episode-v3-evaluation')]:
    path=ROOT/'runs/show101'/name/'evaluation.json'
    if not path.exists():continue
    data=json.loads(path.read_text());point=dict(additional_steps=steps,lineage_steps=50331648+steps,modes={})
    for mode in ('empty_hand','mixed','preinside'):
        r=data['reports'][f'{mode}-s2026'];q=r['placing']
        point['modes'][mode]=dict(episodes=r['overall']['episodes'],basic=r['overall']['final_successes'],
            gentle=q['gentle_final_successes'],supported_release=q['qualified_release_episodes'],
            release_height_mm=q['detected_release_height_mm_mean'],contact_mean_m_s=q['peak_contact_speed_m_s_mean'],
            contact_median_m_s=q['peak_contact_speed_m_s_median'],
            same_initial_scene=r['initial_scene_sha256']==baseline['reports'][f'{mode}-s2026']['initial_scene_sha256'])
    points.append(point)
    if steps==8388608:final=data


def placing_summary(data):
    result={}
    for mode in data['aggregate']:
        rows=[r['placing'] for k,r in data['reports'].items() if k.startswith(mode+'-')]
        release=sum(r['release_episodes'] for r in rows);floor=sum(r['floor_contact_episodes'] for r in rows)
        result[mode]=dict(supported_release=sum(r['qualified_release_episodes'] for r in rows),release_episodes=release,
            release_height_mean_mm=sum((r['detected_release_height_mm_mean'] or 0)*r['release_episodes'] for r in rows)/release if release else None,
            contact_peak_mean_m_s=sum((r['peak_contact_speed_m_s_mean'] or 0)*r['floor_contact_episodes'] for r in rows)/floor if floor else None)
    return result


source=json.loads((DOCS/'source-checkpoint.json').read_text());parent=Path(source['source'])
changed=[n for n,h in source['files'].items() if hashlib.sha256((parent/n).read_bytes()).hexdigest()!=h]
result=dict(scope='Same seed2026/128 per mode at intermediate checkpoints; baseline/final aggregates use384 per mode. Episode tracking fix enabled.',
    points=points,baseline_384=baseline['aggregate'],baseline_placing_384=placing_summary(baseline),source_checkpoint_changed_files=changed,real_robot=False)
if final is not None:
    errors={k:sum(r['overall']['failures'][k] for r in final['reports'].values()) for k in ('controller','physics')}
    result.update(final_384=final['aggregate'],final_placing_384=placing_summary(final),final_errors=errors,
        training=json.loads((RUN/'summary.json').read_text()))
(DOCS/'v3-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
x=np.array([p['additional_steps'] for p in points])/1e6
fig,axs=plt.subplots(2,2,figsize=(13,9))
for mode,label,color in [('empty_hand','빈손 A','#4263eb'),('preinside','놓기 직전','#099268')]:
    rows=[p['modes'][mode] for p in points]
    axs[0,0].plot(x,[100*r['basic']/r['episodes'] for r in rows],'o-',color=color,label=label+' 일반')
    axs[0,0].plot(x,[100*r['gentle']/r['episodes'] for r in rows],'s--',color=color,label=label+' 부드러운 놓기')
    axs[0,1].plot(x,[r['supported_release'] for r in rows],'o-',color=color,label=label)
    axs[1,0].plot(x,[r['release_height_mm'] for r in rows],'o-',color=color,label=label)
    axs[1,1].plot(x,[r['contact_mean_m_s'] for r in rows],'o-',color=color,label=label)
titles=['완료율 (%)','지지 후 해제 횟수 / 128회','평균 실제 해제 높이 (내부 바닥 기준 mm)','접촉 직전 최대 하강속도 평균 (m/s)']
for ax,title in zip(axs.flat,titles):
    ax.set_title(title,fontproperties=font);ax.set_xlabel('이번 추가 전이 (백만)',fontproperties=font)
    ax.grid(alpha=.2);ax.legend(prop=font)
axs[0,0].set_ylim(-2,102)
fig.suptitle('에피소드 전체 놓기 판정 수정 후 SHOW-100에서 재학습 · 동일 128개 장면 비교',fontproperties=font)
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(DOCS/'v3-progress.png',dpi=150)
print(json.dumps({k:v for k,v in result.items() if k not in ['points','training']},ensure_ascii=False))
