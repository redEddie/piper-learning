"""Summarize fresh-run frozen evaluations without treating summed metrics as rates."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/show103/balanced-fresh-16m'
EVAL=ROOT/'runs/show103/balanced-fresh'
DOC=ROOT/'docs/show103/balanced-fresh'


def main():
    points=[]
    for path in sorted(EVAL.glob('*/evaluation.json')):
        data=json.loads(path.read_text());label=path.parent.name
        steps=16777216 if label=='final-16m' else int(label.split('-')[-1])
        by_mode={}
        for mode,aggregate in data['aggregate'].items():
            rows=[r for key,r in data['reports'].items() if key.startswith(mode+'-')]
            n=aggregate['episodes']
            by_mode[mode]=dict(aggregate,
                qualified_releases=sum(r['placing']['qualified_release_episodes'] for r in rows),
                qualified_release_rate=sum(r['placing']['qualified_release_episodes'] for r in rows)/n,
                recontact_episodes=sum(r['placing']['recontact_episodes'] for r in rows),
                floor_contact_episodes=sum(r['placing']['floor_contact_episodes'] for r in rows),
                failures={name:sum(r['overall']['failures'][name] for r in rows)
                          for name in rows[0]['overall']['failures']})
        points.append(dict(transitions=steps,evaluation=label,conditions=by_mode))
    points.sort(key=lambda x:x['transitions'])
    metrics=[json.loads(s) for s in (RUN/'metrics.jsonl').read_text().splitlines()]
    status='running' if not (RUN/'summary.json').exists() else json.loads((RUN/'summary.json').read_text())['status']
    out=dict(random_initialization=True,restored_checkpoint=None,training_status=status,
        real_robot=False,configuration='configs/show103/balanced-fresh-16m.yaml',
        definitions=dict(basic='Final original geometric inside predicate; control/physics faults count as failure.',
            controlled='Final qualified first release at TCP50-60mm, floor contact+relative speed<=20mm/s for100ms, no recontact, clearance80mm for200ms.',
            rates='Full-episode outcomes from each initialization; not isolated stage skill rates.',
            qualification='First release geometry/velocity qualification; later recontact may still invalidate final goal.'),
        points=points,training_metrics=[{k:d[k] for k in ['steps','elapsed_seconds','training/sps',
            'training/v_loss','training/policy_loss','training/kl_mean',
            'eval/episode_reward','eval/episode_coord_reference_fault',
            'eval/episode_controlled_success'] if k in d} for d in metrics])
    DOC.mkdir(exist_ok=True)
    (DOC/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
