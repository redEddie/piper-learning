"""Render frozen-policy evaluation records with gripper raw/goal/reference labels."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse
import json
from pathlib import Path
from unittest import mock
import numpy as np
import mujoco
import mediapy as media
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from piper_sim_lab.tasks import make_env
from piper_sim_lab.pick_place import PiperPickPlace
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--evaluation',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--seed',type=int,default=2026)
ap.add_argument('--mode',choices=['empty_hand','mixed','preinside','near'],default='empty_hand')
ap.add_argument('--scene',type=int,help='Render this recorded case instead of automatic selection')
ap.add_argument('--audit-note',default='',help='Visible qualification when auditing a saved success label')
args=ap.parse_args();args.output.mkdir(exist_ok=False,parents=True)
report=json.loads((args.evaluation/f'{args.mode}-s{args.seed}.json').read_text())
meta=json.loads((args.evaluation/'evaluation.json').read_text())
config=dict(meta['environment'],impl='jax')
is_refinement=True
is_smooth=True
issue_label='SHOW-103 젠틀하게 내려놓기'
tables={k:np.zeros((1,6)) for k in ('pregrasp','grasp','preinside','near')}
with mock.patch('mujoco_playground._src.mjx_env.put_model',return_value=None), \
     mock.patch.object(PiperPickPlace,'_build_reset_table',return_value=tables):
    from piper_sim_lab.gentle_placement import GentlePlacement
    with mock.patch.object(GentlePlacement,'_build_reset_table',return_value=tables):
        env=make_env(config)
model=env._mj_model;data=mujoco.MjData(model)
trace=np.load(args.evaluation/f'{args.mode}-s{args.seed}-trace.npz')
model.mat_reflectance[:]=0
model.vis.global_.offwidth=640;model.vis.global_.offheight=480
option=mujoco.MjvOption();option.sitegroup[:]=0
fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font=ImageFont.truetype(fontpath,22,index=1);small=ImageFont.truetype(fontpath,17,index=1)
media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
def camera(view):
    c=mujoco.MjvCamera()
    if view=='overview':c.lookat[:]=[.19,0,.16];c.distance=1.18;c.azimuth=135;c.elevation=-30
    else:c.lookat[:]=[.29,0,.06];c.distance=.94;c.azimuth=95;c.elevation=-65
    return c

def tile(renderer,t,e,view):
    data.qpos[:]=trace['qpos'][t,e];data.mocap_pos[:]=trace['mocap'][t,e]
    mujoco.mj_kinematics(model,data);mujoco.mj_camlight(model,data)
    renderer.update_scene(data,camera=camera(view),scene_option=option)
    im=Image.fromarray(renderer.render().copy());d=ImageDraw.Draw(im)
    d.rectangle((0,0,640,89 if is_refinement else 65),fill=(17,25,36))
    a=trace['action'][t,e,6];command='닫기' if a<=-.5 else '열기' if a>=.5 else '유지'
    status='기본 완료' if trace['inside'][t,e] else '수행 중'
    if 'gentle_success' in trace and trace['gentle_success'][t,e]>0:
        status='목표 높이·철수 완료' if config.get('task')=='controlled_release' else ('기존 판정: 부드러운 완료' if args.audit_note else '부드러운 완료')
    d.text((12,2),f'사례 #{e} · {status} · 정책 요청 {command} (a={a:+.2f})',font=font,fill=(100,232,164))
    width=trace['width'][t,e]*1000;goal=trace['goal'][t,e]*2000
    d.text((12,32),f'수락 목표 {goal:.0f}mm · 실제 폭 {width:.1f}mm',font=small,fill=(220,230,240))
    if is_smooth:
        ct=trace['coord_trace'][t,e];phase=int(ct[-1,0]);phase_name=['이동','감속','개폐'][phase]
        gt=trace['gentle_trace'][t,e]
        tcp_label=f"TCP {trace['tcp_height'][t,e]*1000:.1f}mm · " if 'tcp_height' in trace else ''
        d.text((12,57),f'{phase_name} · {tcp_label}큐브 바닥 {gt[-1,0]*1000:.1f}mm',font=small,fill=(220,230,240))
    elif is_refinement:
        peak=trace['place_impact_peak'][t,e]
        d.text((12,57),f'접촉 전 최대 하강속도 {peak:.3f}m/s',font=small,fill=(220,230,240))
    return im

def frame(renderer,t,specs,title):
    rows=(len(specs)+1)//2
    im=Image.new('RGB',(1280,64+rows*480+48),(13,20,30));d=ImageDraw.Draw(im)
    d.text((18,12),title,font=font,fill='white')
    for i,(e,view) in enumerate(specs):im.paste(tile(renderer,t,e,view),((i%2)*640,64+(i//2)*480))
    start_label={'empty_hand':'A 자세 빈손 시작','mixed':'혼합 초기화','preinside':'기존 놓기 직전 시작','near':'낮은 놓기 직전 시작'}[args.mode]
    d=ImageDraw.Draw(im);d.text((18,im.height-35),f'{start_label} · 1배속 · t={(t+1)*.1:.1f}s · 3상태 그리퍼 · DLS+QP · 감속 후 개폐',font=small,fill=(182,196,211))
    return np.asarray(im)

success=np.flatnonzero(np.asarray(report['categories'])=='complete').tolist()
if 'gentle_success' in trace:
    valid=~(trace['controller_failed'].astype(bool)|trace['nonfinite'].astype(bool)|trace['overflow'].astype(bool)).any(0)
    gentle=np.flatnonzero((trace['gentle_success'][-1]>0)&valid).tolist()
    if gentle:success=gentle
scene=args.scene if args.scene is not None else (success[0] if success else 0)
if not 0<=scene<trace['qpos'].shape[1]:raise ValueError('scene index outside recorded cases')
videos=[('first-four.mp4',[(e,'close') for e in range(4)],f'{issue_label} | 평가 기록의 첫 4개 장면'),
        ('two-views.mp4',[(scene,'overview'),(scene,'close')],f'{issue_label} | 사례 #{scene} · '+(args.audit_note or '전체 및 작업부'))]
with mujoco.Renderer(model,height=480,width=640) as renderer:
    for name,specs,title in videos:
        with media.VideoWriter(args.output/name,shape=(64+480*((len(specs)+1)//2)+48,1280),fps=10,crf=20) as writer:
            for t in range(len(trace['qpos'])):
                f=frame(renderer,t,specs,title);writer.add_image(f)
                if t in (0,39,79,119,len(trace['qpos'])-1):Image.fromarray(f).save(args.output/f'{Path(name).stem}-{t:02d}.jpg')
        print(name,flush=True)
(args.output/'metadata.json').write_text(json.dumps(dict(evaluation=str(args.evaluation.resolve()),
    seed=args.seed,mode=args.mode,scene_selection='explicit --scene or first valid final gentle-labeled success when present, otherwise first basic success, or scene 0',
    audit_note=args.audit_note,
    first_four=list(range(4)),two_views_scene=scene,playback='recorded poses at 10Hz without interpolation',
    learning=False,real_robot=False),indent=2)+'\n')
