"""Read-only audit of positive placement labels; not a replacement reward.

Saved 100 ms traces expose activation after an earlier open-and-fall sequence.
The heuristic flags cases for review and is not a calibrated success metric.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['final-v2-evaluation', 'interim-continuation-2097152-evaluation',
         'interim-continuation-4194304-evaluation',
         'interim-continuation-6291456-evaluation', 'final-continuation-evaluation']
rows = []
for name in NAMES:
    directory = ROOT/'runs/show101'/name
    evaluation = json.loads((directory/'evaluation.json').read_text())
    for key, report in evaluation['reports'].items():
        if not report['placing']['qualified_release_episodes']:
            continue
        trace = np.load(directory/f'{key}-trace.npz')
        for episode in np.flatnonzero(trace['place_qualified'][-1]):
            def first(field):
                indices = np.flatnonzero(trace[field][:, episode] > 0)
                return int(indices[0]) if len(indices) else None
            active, inside = first('place_active'), first('inside')
            release = first('place_released')
            start = max(0, inside-10) if inside is not None else 0
            width = float(trace['width'][inside, episode]*1000) if inside is not None else None
            fall = float((trace['block'][start:inside+1, episode, 2].max()-
                         trace['block'][inside, episode, 2])*1000) if inside is not None else None
            opened = bool(np.any(trace['action'][start:inside+1, episode, 6] >= .5)) if inside is not None else False
            late = bool(inside is not None and active-inside >= 2 and
                        width >= 75 and fall >= 30 and opened)
            rows.append(dict(evaluation=name, recording=key, episode=int(episode),
                raw_gentle_final=bool(trace['gentle_success'][-1, episode]),
                first_inside_seconds=(inside+1)*.1 if inside is not None else None,
                first_placing_activation_seconds=(active+1)*.1,
                recorded_release_seconds=(release+1)*.1,
                opening_before_first_inside=opened,
                width_at_first_inside_mm=width,
                descent_in_previous_second_mm=fall,
                recorded_peak_contact_speed_m_s=float(trace['place_impact_peak'][-1, episode]),
                late_activation_after_open_descent=late))
gentle = [r for r in rows if r['raw_gentle_final']]
output = dict(scope='Audit of all positive qualified-release labels in saved evaluations; no retraining, reward changes, or new physical rollout',
    finding='Placing ledger activates after prior open/descent, allowing later incidental finger contact to appear as supported release and omitting earlier impact.',
    reward_uses_same_ledger=True,
    raw_gentle_positive_records=len(gentle),
    raw_gentle_records_flagged_for_late_activation=sum(r['late_activation_after_open_descent'] for r in gentle),
    limitation='Trace heuristic and example video support a classification error; this does not establish a new validated success rate or assess false negatives.',
    records=rows, real_robot=False)
path=ROOT/'docs/show101/continuation-success-label-audit.json'
path.write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k!='records'}, ensure_ascii=False))
