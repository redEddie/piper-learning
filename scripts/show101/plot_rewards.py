"""Explain original and placing-refinement rewards under explicit fixed geometry."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
root=Path(__file__).resolve().parents[2];out=root/'docs/show101'
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'axes.unicode_minus':False,'font.size':11})
reach=1-np.tanh(5*.01);released_reach=1-np.tanh(5*.06)
z=np.linspace(3.001,160,1400)
lift=(z>20).astype(float);placed=(z<45).astype(float)
old=reach+2*lift+5*placed
v1=reach+2*lift
v2=np.full(z.shape,reach+2)
labels=['기존 SHOW-76','놓기 첫 안(미채택)','놓기 보완안(839만 추가학습)'];colors=['#4263eb','#e8590c','#099268']
fig,axs=plt.subplots(1,2,figsize=(15,6.3),gridspec_kw={'width_ratios':[1.2,1]})
a=axs[0]
for y,label,col in zip([old,v1,v2],labels,colors):a.plot(z,y,label=label,color=col,lw=2.7)
a.axvline(20,color='#495057',ls='--',lw=1.3);a.axvline(45,color='#868e96',ls=':',lw=1.3)
a.text(21,8.65,'들기·운반 판정 경계: 20mm',fontproperties=font,fontsize=10)
a.text(46,7.25,'기존 배치 판정 경계: 45mm',fontproperties=font,fontsize=10)
a.scatter([19,21],[reach,reach+2],c=colors[1],zorder=4)
a.annotate('첫 안: 내려놓는 중 약 2점 감소',xy=(20,1.9),xytext=(52,1.0),fontproperties=font,fontsize=11,arrowprops=dict(arrowstyle='->',color=colors[1]),color=colors[1])
a.set(xlim=(0,160),ylim=(0,9.3));a.set_xlabel('물체 아랫면의 탁자 기준 높이(mm) · 내려놓을수록 왼쪽',fontproperties=font);a.set_ylabel('100ms 한 스텝의 지속 보상',fontproperties=font)
a.set_title('① 물체를 잡고 있을 때: 높이에 따른 보상',fontproperties=font,pad=12)
a.legend(prop=font,loc='upper right');a.grid(alpha=.2)
b=axs[1];x=np.arange(4);width=.25
vals=[[reach+2,reach+5,released_reach+7,released_reach+7],
      [reach+2,reach,released_reach+2,released_reach+7],
      [reach+2,reach+2,released_reach+3.5,released_reach+7]]
for i,(v,col) in enumerate(zip(vals,colors)):
 bars=b.bar(x+(i-1)*width,v,width,color=col,alpha=.9)
 b.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
b.set_xticks(x,['100mm에서\n잡고 대기','바닥에서\n잡고 대기','높은 곳 낙하로\n기본 완료','지지 후\n부드럽게 해제'],fontproperties=font)
b.set_ylabel('100ms 한 스텝의 지속 보상',fontproperties=font);b.set_ylim(0,9.3);b.grid(axis='y',alpha=.2)
b.set_title('② 완료 후 보상: 기본 완료와 놓기 품질',fontproperties=font,pad=12)
fig.suptitle('매 100ms마다 받는 보상 비교 — 스테이지 전체에 한 번 주는 점수가 아님',fontproperties=font,fontsize=15)
fig.text(.05,.03,'단순화: 50mm 큐브·내부 바닥 3mm·테두리 70mm·수평오차 0·TCP 거리 잡을 때 10mm/해제 후 60mm.\n진척·충격·개방의 일회성 항과 행동 비용 제외. 오른쪽 부드러운 해제는 충격 속도 0으로 가정.',fontproperties=font,fontsize=10,color='#495057')
fig.tight_layout(rect=(0,.12,1,.92));fig.savefig(out/'reward-height-and-outcome.png',dpi=160);plt.close(fig)
first=json.loads((out/'reward-audit.json').read_text())['results'];second=json.loads((out/'reward-audit-v2.json').read_text())['results']
keys=['wait','drop','gentle','delayed_gentle','closed_on_floor']
fig,ax=plt.subplots(figsize=(12,5.5));x=np.arange(5)
for i,(d,col,label) in enumerate([(first,colors[1],labels[1]),(second,colors[2],labels[2])]):
 vals=[d[k]['discounted_return'] for k in keys];bars=ax.bar(x+(i-.5)*.32,vals,.32,label=label,color=col);ax.bar_label(bars,fmt='%.1f',padding=3)
ax.set_xticks(x,['위에서 대기','높은 곳 낙하','빠른 부드러운 놓기','3초 대기 후\n부드러운 놓기','바닥에서 계속 잡기'],fontproperties=font)
ax.set_ylabel('8초 할인 누적 보상(γ=0.99)',fontproperties=font);ax.set_title('스크립트 경로 수치 검증 — 실제 학습 성공률이 아님',fontproperties=font)
ax.legend(prop=font);ax.grid(axis='y',alpha=.2)
fig.text(.03,.025,'진척·접촉 직전 충격·개방·기존 행동 비용 포함. 정해진 특징 경로를 계산한 것으로 실제 물리 궤적 재생과 구분.',fontproperties=font,fontsize=10,color='#495057')
fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'discounted-placement-returns.png',dpi=160);plt.close(fig)
print(str(out/'reward-height-and-outcome.png'))
