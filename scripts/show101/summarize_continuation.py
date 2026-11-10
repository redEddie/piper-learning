"""Summarize fixed-seed continuation evaluations and preserve comparison scope."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs/show101'
BASELINE = ROOT / 'runs/show101/final-v2-evaluation/evaluation.json'
FINAL = ROOT / 'runs/show101/final-continuation-evaluation/evaluation.json'


def main():
    baseline = json.loads(BASELINE.read_text())
    sources = [(0, baseline)]
    for steps in (2097152, 4194304, 6291456):
        path = ROOT / f'runs/show101/interim-continuation-{steps}-evaluation/evaluation.json'
        if path.exists():
            sources.append((steps, json.loads(path.read_text())))
    if FINAL.exists():
        sources.append((8388608, json.loads(FINAL.read_text())))
    points = []
    for steps, data in sources:
        point = dict(additional_steps=steps, lineage_steps=58720256+steps, modes={})
        for mode in ('empty_hand', 'mixed', 'preinside'):
            report = data['reports'][f'{mode}-s2026']
            placing = report['placing']
            point['modes'][mode] = dict(
                episodes=report['overall']['episodes'],
                basic_successes=report['overall']['final_successes'],
                gentle_successes=placing['gentle_final_successes'],
                supported_then_released=placing['qualified_release_episodes'],
                release_height_mean_mm=placing['detected_release_height_mm_mean'],
                peak_contact_speed_mean_m_s=placing['peak_contact_speed_m_s_mean'],
                peak_contact_speed_median_m_s=placing['peak_contact_speed_m_s_median'],
                initial_scene_matches_baseline=(report['initial_scene_sha256'] ==
                    baseline['reports'][f'{mode}-s2026']['initial_scene_sha256']))
        points.append(point)
    protocol = json.loads((DOCS / 'continuation-protocol.json').read_text())
    parent = Path(protocol['source_checkpoint'])
    changed = [name for name, digest in protocol['source_files_sha256'].items()
               if hashlib.sha256((parent/name).read_bytes()).hexdigest() != digest]
    run = ROOT / 'runs/show101/mlp-place-v2-continue-8m'
    old = ROOT / 'runs/show101/mlp-place-v2-8m'
    same_config = (json.loads((run/'config.json').read_text()) ==
                   json.loads((old/'config.json').read_text()))
    result = dict(scope='seed2026 fixed 128 scenes per mode at each checkpoint; final aggregates separately use 384 scenes',
                  measurement_warning='Original gentle/qualified-release labels have a late-activation false-positive defect also used by reward. See continuation-success-label-audit.json. Reported labels are raw and are not validated gentle-place success rates.',
                  points=points, unchanged_run_configuration=same_config,
                  source_checkpoint_changed_files=changed, real_robot=False)
    if FINAL.exists():
        final = json.loads(FINAL.read_text())
        result.update(baseline_384=baseline['aggregate'], final_384=final['aggregate'])
        for name, evaluation in [('baseline', baseline), ('final', final)]:
            summaries = {}
            for mode in ('empty_hand', 'mixed', 'preinside'):
                rows = [r['placing'] for key, r in evaluation['reports'].items() if key.startswith(mode+'-')]
                releases = sum(r['release_episodes'] for r in rows)
                contacts = sum(r['floor_contact_episodes'] for r in rows)
                summaries[mode] = dict(
                    supported_then_released=sum(r['qualified_release_episodes'] for r in rows),
                    release_episodes=releases, floor_contact_episodes=contacts,
                    release_height_mean_mm=sum((r['detected_release_height_mm_mean'] or 0)*r['release_episodes'] for r in rows)/releases if releases else None,
                    peak_contact_speed_mean_m_s=sum((r['peak_contact_speed_m_s_mean'] or 0)*r['floor_contact_episodes'] for r in rows)/contacts if contacts else None)
            result[name+'_placing_384'] = summaries
        result['training'] = json.loads((run/'summary.json').read_text())
    (DOCS/'continuation-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    x = np.array([p['additional_steps'] for p in points])/1e6
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for mode, label, color in [('empty_hand', '빈손·A', '#4263eb'), ('preinside', '놓기 직전', '#099268')]:
        rows = [p['modes'][mode] for p in points]
        basic = [100*r['basic_successes']/r['episodes'] for r in rows]
        gentle = [100*r['gentle_successes']/r['episodes'] for r in rows]
        axes[0,0].plot(x, basic, 'o-', color=color, label=label+' 일반 완료')
        axes[0,0].plot(x, gentle, 's--', color=color, label=label+' 부드러운 놓기')
        axes[0,1].plot(x, [r['supported_then_released'] for r in rows], 'o-', color=color, label=label)
        axes[1,0].plot(x, [r['release_height_mean_mm'] for r in rows], 'o-', color=color, label=label)
        axes[1,1].plot(x, [r['peak_contact_speed_mean_m_s'] for r in rows], 'o-', color=color, label=label)
    titles = ['기존 판정 완료율 (%) · 부드러운 놓기 값은 오판 포함', '기존 지지 후 해제 판정 수 / 128회 · 오판 포함',
              '평균 해제 높이 (그릇 내부 바닥 기준 mm)', '접촉 직전 최대 하강속도의 평균 (m/s)']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontproperties=font)
        ax.set_xlabel('이번 추가학습 전이 (백만)', fontproperties=font)
        ax.grid(alpha=.2)
        ax.legend(prop=font, fontsize=9)
    axes[0,0].set_ylim(-2,102)
    fig.suptitle('보상·설정 유지 추가학습 — 기존 놓기 판정 오류 발견, 수치는 참고용', fontproperties=font)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(DOCS/'continuation-progress.png', dpi=150)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
