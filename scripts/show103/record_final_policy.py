"""Record final-policy first episodes; render separately on CPU without robot I/O."""
import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('phase', choices=['capture', 'render'])
ap.add_argument('--run', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()
cfg = json.loads((args.run / 'config.json').read_text())
checkpoint = args.run / 'checkpoints/000050331648'
args.output.mkdir(parents=True, exist_ok=True)

if args.phase == 'capture':
    from piper_sim_lab.runtime import configure
    configure('gpu')
    import functools
    import jax
    import jax.numpy as jp
    import numpy as np
    from piper_sim_lab.quadratic_release import QuadraticRelease
    from piper_sim_lab.checkpoint import load_policy
    from piper_sim_lab.wrappers import wrap_for_training

    class RecordedTask(QuadraticRelease):
        def snapshot(self, st):
            f = self._place_features(st.data)
            v = dict(qpos=st.data.qpos, mocap=st.data.mocap_pos,
                width=f['width'], tcp_height=f['tcp_height'], xy=f['xy'],
                goal=st.info['gripper_goal'], phase=st.info['coord_phase'],
                reason=st.info['task_failure_reason'], reward=st.reward,
                controller_failed=st.info['controller_failed'],
                inside=self._predicates(st.data)['inside'])
            preserve = dict(st.info.get('AutoResetWrapper_preserve_info', {}), visual=v)
            return st.replace(info=dict(st.info, AutoResetWrapper_preserve_info=preserve))

        def reset(self, key):
            return self.snapshot(super().reset(key))

        def step(self, st, action):
            return self.snapshot(super().step(st, action))

    for mode, probs in [('empty_hand', [1., 0., 0., 0.]), ('grasp', [0., 0., 1., 0.])]:
        if (args.output / f'{mode}.npz').exists():
            print(json.dumps(dict(mode=mode, reuse_recording=True)), flush=True)
            continue
        c = dict(cfg['resolved_environment'], batch_capacity=128, reset_stage_probs=probs)
        c.pop('task')
        env = RecordedTask(c)
        policy = load_policy(checkpoint, observation_size=env.observation_size,
                             action_size=7, deterministic=True)
        w = wrap_for_training(env, episode_length=200,
            randomization_fn=functools.partial(env.physics_randomization_fn,
                rng=jax.random.split(jax.random.key(2033), 128)))
        state = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(2026), 128))

        @jax.jit
        def rollout(state):
            def step(carry, i):
                old, alive = carry
                a, _ = policy(old.obs, jax.random.fold_in(jax.random.PRNGKey(2026), i))
                st = w.step(old, a)
                v = st.info['AutoResetWrapper_preserve_info']['visual']
                sample = {k: val[:4] for k, val in v.items()}
                sample.update(action=a[:4], valid=alive[:4], done=st.done[:4])
                return (st, alive & ~st.done.astype(bool)), sample
            return jax.lax.scan(step, (state, jp.ones(128, bool)), jp.arange(200))[1]

        tr = jax.tree.map(np.asarray, jax.block_until_ready(rollout(state)))
        np.savez_compressed(args.output / f'{mode}.npz', **tr)
        print(json.dumps(dict(mode=mode, first_four_duration=tr['valid'].sum(0).tolist(),
            recorded=True)), flush=True)
        rollout.clear_cache()
    comparisons = {}
    for mode in ['empty_hand', 'grasp']:
        tr = np.load(args.output / f'{mode}.npz')
        previous = np.load(args.run.parent / 'qp-v1-a5-j100-final-evaluation' / f'{mode}-s2026-trace.npz')
        # GPU contact physics can diverge numerically across compilations. Keep
        # the measured difference; do not pretend this is the original trace.
        comparisons[mode] = {recorded: float(np.abs(tr[recorded][tr['valid']] -
            previous[name][:, :4][tr['valid']]).max())
            for name, recorded in [('eval_width', 'width'), ('eval_xy', 'xy'),
                ('eval_table_height', 'tcp_height'), ('eval_task_reason', 'reason')]}
        np.testing.assert_allclose(tr['width'][0], previous['eval_width'][0, :4], atol=1e-6)
    (args.output / 'metadata.json').write_text(json.dumps(dict(checkpoint=str(checkpoint.resolve()),
        modes=['empty_hand', 'grasp'], seed=2026, batch=128, cases=[0, 1, 2, 3],
        displayed_case=0, playback='1x; 10Hz recorded poses; freeze after first termination',
        saved_evaluation_max_abs_difference=comparisons,
        note='Fresh recording with the same frozen policy/config/seed; GPU contact physics may differ numerically.',
        learning=False, real_robot=False), indent=2) + '\n')
else:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['MUJOCO_GL'] = 'egl'
    from unittest import mock
    import numpy as np
    import mujoco
    import mediapy as media
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    from piper_sim_lab.tasks import make_env
    from piper_sim_lab.pick_place import PiperPickPlace
    from piper_sim_lab.gentle_placement import GentlePlacement

    tables = {k: np.zeros((1, 6)) for k in ['pregrasp', 'grasp', 'preinside', 'near']}
    with mock.patch('mujoco_playground._src.mjx_env.put_model', return_value=None), \
         mock.patch.object(PiperPickPlace, '_build_reset_table', return_value=tables), \
         mock.patch.object(GentlePlacement, '_build_reset_table', return_value=tables):
        env = make_env(dict(cfg['resolved_environment'], impl='jax'))
    m = env._mj_model
    d = mujoco.MjData(m)
    m.mat_reflectance[:] = 0
    m.vis.global_.offwidth = 640
    m.vis.global_.offheight = 480
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    fontpath = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(fontpath, 21, index=1)
    small = ImageFont.truetype(fontpath, 17, index=1)
    traces = [np.load(args.output / f'{mode}.npz') for mode in ['empty_hand', 'grasp']]
    labels = ['빈손 A 자세 시작', '물체를 잡은 상태에서 시작']
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    with mujoco.Renderer(m, height=480, width=640) as renderer:
        with media.VideoWriter(args.output / 'final-policy.mp4', shape=(592, 1280), fps=10, crf=19) as writer:
            for t in range(200):
                frame = Image.new('RGB', (1280, 592), (13, 20, 30))
                draw = ImageDraw.Draw(frame)
                draw.text((16, 12), '최종 5,033만 정책 · 평가 시드 2026 · 첫 번째 사례 (성공 장면 선별 없음)', font=font, fill='white')
                for column, (tr, label) in enumerate(zip(traces, labels)):
                    last = int(tr['valid'][:, 0].sum()) - 1
                    i = min(t, last)
                    d.qpos[:] = tr['qpos'][i, 0]
                    d.mocap_pos[:] = tr['mocap'][i, 0]
                    mujoco.mj_kinematics(m, d)
                    mujoco.mj_camlight(m, d)
                    cam = mujoco.MjvCamera()
                    cam.lookat[:] = [.19, 0, .14]
                    cam.distance = 1.05
                    cam.azimuth = 135
                    cam.elevation = -35
                    renderer.update_scene(d, camera=cam, scene_option=option)
                    tile = Image.fromarray(renderer.render().copy())
                    td = ImageDraw.Draw(tile)
                    td.rectangle((0, 0, 640, 115), fill=(17, 25, 36))
                    a = tr['action'][i, 0, 6]
                    command = '닫기' if a <= -.5 else '열기' if a >= .5 else '유지'
                    ended = t >= last
                    reason = int(tr['reason'][i, 0])
                    status = ('파지 시간초과 종료' if reason == 1 else '과제 종료') if ended else '수행 중'
                    td.text((12, 3), f'{label} · {status}', font=font, fill=(100, 232, 164))
                    td.text((12, 33), f"정책 {command} ({a:+.2f}) · 목표 {tr['goal'][i,0]*2000:.0f} / 실제 {tr['width'][i,0]*1000:.0f}mm", font=small, fill='white')
                    td.text((12, 60), f"TCP-그릇 바닥 {tr['tcp_height'][i,0]*1000:.0f}mm · 큐브-그릇 XY {tr['xy'][i,0]*1000:.0f}mm", font=small, fill='white')
                    td.text((12, 87), f"{['이동','감속','개폐'][int(tr['phase'][i,0])]} · t={(i+1)*.1:.1f}s · 스텝 보상 {tr['reward'][i,0]:+.2f}", font=small, fill='white')
                    frame.paste(tile, (column * 640, 64))
                ImageDraw.Draw(frame).text((16, 557), '시뮬레이션 · 1배속 · 종료 후 마지막 자세 정지 표시 · 실제 로봇 구동 없음', font=small, fill=(182, 196, 211))
                writer.add_image(np.asarray(frame))
                if t in [0, 39, 79, 119, 199]:
                    frame.save(args.output / f'frame-{t:03}.jpg')
    print(str((args.output / 'final-policy.mp4').resolve()), flush=True)
