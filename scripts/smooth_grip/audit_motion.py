"""Audit recorded reference bounds, actuation coordination and completion time.

Reference bounds are distinct from actual joint motion. No robot I/O is used.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--evaluation', type=Path, required=True)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
evaluation = json.loads((args.evaluation / 'evaluation.json').read_text())
reports = {}
for name, report in evaluation['reports'].items():
    with np.load(args.evaluation / f'{name}-trace.npz') as trace:
        valid = ~(trace['controller_failed'].astype(bool) |
                  trace['nonfinite'].astype(bool) | trace['overflow'].astype(bool)).any(0)
        final = trace['inside'][-1].astype(bool) & valid
        first = (trace['inside'].argmax(0)[final] + 1) * .1
        # Each row is a recorded 5ms sample of the actual joint positions.
        actual = trace['tracking'][..., 12:18].transpose(1, 0, 2, 3).reshape(len(valid), -1, 6)
        actual_jerk = np.diff(actual, n=3, axis=1) / .005**3
        result = dict(
            final_successes=int(final.sum()),
            completion_time_seconds_median=float(np.median(first)) if len(first) else None,
            completion_time_seconds_p90=float(np.quantile(first, .9)) if len(first) else None,
            actual_joint_jerk_estimate=dict(
                method='Third difference of recorded actual joint positions at5ms; valid episodes, excluding first100ms; not a hard bound',
                rms=float(np.sqrt(np.mean(actual_jerk[valid, 20:]**2))),
                absolute_p99=float(np.quantile(np.abs(actual_jerk[valid, 20:]), .99))),
        )
        if 'coord_trace' in trace:
            ct = trace['coord_trace']
            grip = (ct[..., 0] == 2) & valid[None, :, None]
            actual_speed = ct[..., 12][grip]
            result.update(
                reference_limit_violating_episodes=int((((trace['controller_reason'].astype(int) & 63) != 0).any(0)).sum()),
                reference_speed_during_grip_max=float(ct[..., 5][grip].max()),
                actual_max_joint_speed_during_grip=dict(
                    p50=float(np.quantile(actual_speed, .5)),
                    p95=float(np.quantile(actual_speed, .95)),
                    p99=float(np.quantile(actual_speed, .99)),
                    maximum=float(actual_speed.max())),
                failure_reasons={str(int(reason)): int((trace['controller_reason'][-1] == reason).sum())
                                 for reason in np.unique(trace['controller_reason'][-1])},
            )
            if name == 'empty_hand-s2026':
                scene = int(np.flatnonzero(final)[0])
                detail = ct[:, scene].reshape(-1, 13)
                seconds = (np.arange(len(detail)) + 1) * .005
                font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
                fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
                axes[0].step(seconds, detail[:, 0], where='post', color='#7048e8')
                axes[0].set_yticks([0, 1, 2], ['MOVE', 'BRAKE', 'GRIP'])
                axes[0].set_title(f'시뮬레이션 사례 #{scene}: 감속 → 개폐 → 이동', fontproperties=font)
                axes[1].plot(seconds, detail[:, 3] * 1000, label='실제 개방폭')
                axes[1].plot(seconds, detail[:, 2] * 2000, '--', label='작동 목표')
                axes[1].set_ylabel('개방폭 (mm)', fontproperties=font)
                axes[1].legend(prop=font, loc='upper right')
                axes[2].plot(seconds, detail[:, 5], label='팔 참조속도 최대')
                axes[2].plot(seconds, detail[:, 12], alpha=.6, label='팔 실제속도 최대')
                axes[2].set_ylabel('rad/s')
                axes[2].set_xlabel('시간 (초)', fontproperties=font)
                axes[2].legend(prop=font, loc='upper right')
                for ax in axes:
                    ax.grid(alpha=.2)
                fig.tight_layout()
                fig.savefig(args.output / 'coordination.png', dpi=150)
                plt.close(fig)
        reports[name] = result
(args.output / 'motion.json').write_text(json.dumps(dict(
    evaluation=str(args.evaluation.resolve()),
    scope='Same recorded scenes and20s horizon. Completion time is first inside among final successes, not independently verified gentle placement.',
    reports=reports, real_robot=False), ensure_ascii=False, indent=2) + '\n')
print(json.dumps(reports, ensure_ascii=False), flush=True)
