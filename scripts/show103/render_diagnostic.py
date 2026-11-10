"""Render recorded physical validation, explicitly not a learned policy."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('JAX_PLATFORMS','cpu')
from pathlib import Path
from unittest import mock
import numpy as np
import mujoco,yaml,mediapy as media,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from piper_sim_lab.tasks import make_env
from piper_sim_lab.pick_place import PiperPickPlace
from piper_sim_lab.gentle_placement import GentlePlacement

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/show103/diagnostic-release-video'
OUT.mkdir(exist_ok=False)
tables={k:np.zeros((1,6)) for k in ['pregrasp','grasp','preinside','near']}
c=yaml.safe_load((ROOT/'configs/show103/release-8m.yaml').read_text())['environment']
with mock.patch('mujoco_playground._src.mjx_env.put_model',return_value=None),\
     mock.patch.object(PiperPickPlace,'_build_reset_table',return_value=tables),\
     mock.patch.object(GentlePlacement,'_build_reset_table',return_value=tables):
    env=make_env(dict(c,impl='jax'))
m=env._mj_model;d=mujoco.MjData(m);m.mat_reflectance[:]=0
m.vis.global_.offwidth=640;m.vis.global_.offheight=480
tr=np.load(ROOT/'runs/show103/positive-release.npz')
d.qpos[:]=tr['qpos'][0,0];mujoco.mj_kinematics(m,d);bowl=d.xpos[env._bowl_id].copy()
fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font=ImageFont.truetype(fontpath,22,index=1);small=ImageFont.truetype(fontpath,18,index=1)
option=mujoco.MjvOption();option.sitegroup[:]=0
media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
with mujoco.Renderer(m,height=480,width=640) as renderer:
    with media.VideoWriter(OUT/'diagnostic.mp4',shape=(592,1280),fps=10,crf=20) as writer:
        for i in range(120):
            d.qpos[:]=tr['qpos'][i,0];mujoco.mj_kinematics(m,d);mujoco.mj_camlight(m,d)
            frame=Image.new('RGB',(1280,592),(13,20,30));draw=ImageDraw.Draw(frame)
            draw.text((18,12),'SHOW-103 | 진단용 동작 · 학습 정책 아님',font=font,fill='white')
            for view in range(2):
                cam=mujoco.MjvCamera()
                cam.lookat[:]=[.19,float(bowl[1])*.3,.15] if view==0 else [float(bowl[0]),float(bowl[1]),.08]
                cam.distance=1.1 if view==0 else .7;cam.azimuth=135 if view==0 else 95;cam.elevation=-30 if view==0 else -65
                renderer.update_scene(d,camera=cam,scene_option=option)
                im=Image.fromarray(renderer.render().copy());di=ImageDraw.Draw(im);di.rectangle((0,0,640,84),fill=(17,25,36))
                di.text((12,4),f"TCP-내부 바닥 {tr['tcp_height'][i,0]*1000:.1f}mm · 실제 개방폭 {tr['width'][i,0]*1000:.1f}mm",font=small,fill=(220,230,240))
                di.text((12,32),f"목표 높이 해제 {bool(tr['qualified'][i,0])} · 재접촉 {bool(tr['recontact'][i,0])}",font=small,fill=(100,232,164))
                di.text((12,57),f"목표 해제·철수 완료 {bool(tr['gentle'][i,0])}",font=small,fill=(100,232,164))
                frame.paste(im,(view*640,64))
            ImageDraw.Draw(frame).text((18,557),f'기록된 시뮬레이션 동작 · 1배속 · {(i+1)*.1:.1f}s · 100mm 개방 유지',font=small,fill=(182,196,211))
            writer.add_image(np.asarray(frame))
            if i in [29,44,79,119]:frame.save(OUT/f'frame-{i:03}.jpg')
print(OUT/'diagnostic.mp4',flush=True)
