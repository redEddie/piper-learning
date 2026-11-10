"""Render one explicitly chosen recorded SHOW-103 episode at real-time speed."""
import argparse
import json
from pathlib import Path

from piper_sim_lab.runtime import configure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', type=int, default=0)
    parser.add_argument('--diagnostic', action='store_true', help='Show release height/XY and actual contact diagnostics')
    parser.add_argument('--label', default='', help='Visible checkpoint/start label')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    configure('cpu')
    import imageio_ffmpeg
    import mediapy as media
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from piper_sim_lab.tasks import make_env

    run_config = json.loads((args.run/'config.json').read_text())
    config = run_config['resolved_environment']
    five_start = config['task'] == 'stage_reward_pick_place_five_starts_v2'
    if five_start:
        args.diagnostic = True
        names = dict(initial_a='① 접근', pre_grasp='② 파지·들기', held='③ 운반',
                     over_bowl='④ 정렬·하강·해제', post_release='⑤ 안착·후퇴·A 복귀')
        start = args.trace.stem.split('-s')[0]
        unit_kind = '학습' if run_config.get('learner') == 'recurrent_ppo' and not run_config.get('continuation_checkpoint') else '추가'
        args.label = args.label or f'{unit_kind} {args.trace.parent.name.removesuffix("u")}배 / {names.get(start, start)}'
    env = make_env(dict(config, impl='jax', batch_capacity=1))
    model, data = env.mj_model, mujoco.MjData(env.mj_model)
    trace = np.load(args.trace)
    if not 0 <= args.scene < trace['qpos'].shape[1]:
        parser.error('scene outside recorded cases')
    frame_count = len(trace['qpos'])
    # Evaluation retains post-fault measurements for diagnostics. A v2 video
    # ends at the first terminal sample, matching the trained episode boundary.
    if five_start and 'failed' in trace:
        failures = np.flatnonzero(trace['terminal' if 'terminal' in trace else 'failed'][:, args.scene])
        if len(failures):
            frame_count = min(frame_count, int(failures[0])+1)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.20, 0., .15]
    camera.distance, camera.azimuth, camera.elevation = 1.10, 135, -30
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    model.mat_reflectance[:] = 0
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc' if args.diagnostic else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 16)
    header = 128 if args.diagnostic else 64
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene = args.scene
    with mujoco.Renderer(model, height=480, width=640) as renderer, \
         media.VideoWriter(args.output, shape=(480+header, 640), fps=10, crf=20) as video:
        for i in range(frame_count):
            data.qpos[:] = trace['qpos'][i, scene]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            frame = Image.new('RGB', (640, 480+header), (15, 24, 35))
            frame.paste(Image.fromarray(renderer.render().copy()), (0, header))
            draw = ImageDraw.Draw(frame)
            draw.text((8, 3), f'SHOW-103 reward v1 | case {scene} | t={(i+1)*.1:.1f}s | 1x', font=font, fill='white')
            draw.text((8, 23), f"raw grip {trace['action'][i,scene,6]:+.2f} | goal {2000*trace['goal'][i,scene]:.0f}mm | width {1000*trace['width'][i,scene]:.1f}mm", font=font, fill='white')
            draw.text((8, 43), f"section {int(trace['metric_stage_section'][i,scene])} | qualified {int(trace['metric_qualified_release'][i,scene])} | complete {int(trace['metric_success'][i,scene])}", font=font, fill='white')
            if args.diagnostic:
                draw.rectangle((0, 0, 640, header-1), fill=(15, 24, 35))
                a = float(trace['action'][i, scene, 6])
                command = '열기' if a >= .5 else '닫기' if a <= -.5 else '유지'
                features = trace['features'][i, scene]
                height, xy = features[4]*1000, features[2]*1000
                lines = [
                    (f'{args.label or args.trace.stem} | 사례 {scene} | {(i+1)*.1:.1f}초 | 1배속', 'white'),
                    (f'그리퍼 요청 {a:+.2f} ({command}) | 열기 기준 +0.50 이상', 'white'),
                    (f"최종 목표 {trace['goal'][i,scene]*2000:.0f}mm | 참조 {trace['reference'][i,scene]*2000:.0f}mm | 실제 폭 {trace['width'][i,scene]*1000:.1f}mm", 'white'),
                    (f'TCP 높이 {height:.1f}mm (목표 50~60mm) | 큐브-그릇 XY {xy:.1f}mm (목표 ≤17mm)', (110, 231, 172) if 50 <= height <= 60 and xy <= 17 else (255, 189, 105)),
                    (f"양쪽 접촉 {int(features[13])}/{int(features[14])} | 주보상 {trace['metric_stage_main'][i,scene]:.2f} | 총보상 {trace['reward'][i,scene]:.2f}", 'white'),
                    (f"구간 {int(trace['metric_stage_section'][i,scene])} | 합격 해제 {int(trace['metric_qualified_release'][i,scene])} | 완료 {int(trace['metric_success'][i,scene])} | 실패 {int(trace['failed'][i,scene])}", 'white'),
                ]
                if config.get('five_start_physical_release'):
                    drop=trace['metric_stage_freefall_height'][i,scene]*1000
                    lines[3]=(f'TCP 높이 {height:.1f}mm | 큐브-그릇 XY {xy:.1f}mm | 큐브 낙하 {drop:.1f}mm','white')
                    home=trace['metric_home_error'][i,scene]*180/np.pi
                    lines[5]=(f"구간 {int(trace['metric_stage_section'][i,scene])} | A 오차 {home:.1f}도 | 성공 {int(trace['metric_success'][i,scene])} | 실패 {int(trace['failed'][i,scene])}",'white')
                if config.get('five_start_progress_reward'):
                    credit=trace['metric_stage_completed_credit'][i,scene];progress=trace['metric_stage_progress'][i,scene]
                    face=trace['metric_stage_face_cost'][i,scene];closed=trace['metric_stage_closed_cost'][i,scene]
                    lines[4]=(f"기본 {credit:.2f} + 진행 {progress:.2f} | 면 {face:.2f} 닫힘 {closed:.2f} 감점 | 총 {trace['reward'][i,scene]:.2f}",'white')
                for j, (line, color) in enumerate(lines):
                    draw.text((8, 2+21*j), line, font=font, fill=color)
            video.add_image(np.asarray(frame))
            if i == frame_count-1:
                frame.save(args.output.with_suffix('.png'))
    args.output.with_suffix('.json').write_text(json.dumps(dict(trace=str(args.trace.resolve()),
        scene=scene, selection='explicit scene index; default 0 is first recorded case, no success selection',
        fps=10, control_dt_s=.1, duration_seconds=frame_count*.1,
        rendered_frames=frame_count, recorded_frames=len(trace['qpos']),
        stopped_at_first_failure=bool(five_start and trace['failed'][frame_count-1,scene]),
        stopped_at_terminal_success=bool('terminal' in trace and trace['terminal'][frame_count-1,scene] and not trace['failed'][frame_count-1,scene]),
        final_complete=bool(trace['metric_success'][frame_count-1,scene]),
        diagnostic=args.diagnostic, label=args.label,
        real_robot=False), indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
