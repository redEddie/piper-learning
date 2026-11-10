"""Compare full final evaluations at16m and cumulative50m, never metric sums."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/show103/balanced-to-50m'
RUN=ROOT/'runs/show103/balanced-continue-32m'
BEFORE=ROOT/'runs/show103/balanced-fresh/final-16m/evaluation.json'
AFTER=ROOT/'runs/show103/balanced-to-50m/final-50m/evaluation.json'


def main():
    a=json.loads(BEFORE.read_text());b=json.loads(AFTER.read_text())
    training=json.loads((RUN/'summary.json').read_text())
    assert training['status']=='passed' and training['policy_changed']
    assert training['policy_changes'][-1]['steps']==33554432
    rows={}
    for mode,r in b['aggregate'].items():
        assert r['episodes']==a['aggregate'][mode]['episodes']==384
        reports=[v for k,v in b['reports'].items() if k.startswith(mode+'-')]
        rows[mode]=dict(before=a['aggregate'][mode],after=r,
            ordinary_change_percentage_points=100*(r['final_success_rate']-a['aggregate'][mode]['final_success_rate']),
            controlled_change_percentage_points=100*(r['controlled_final_success_rate']-a['aggregate'][mode]['controlled_final_success_rate']),
            qualified_releases=sum(v['placing']['qualified_release_episodes'] for v in reports),
            release_episodes=sum(v['placing']['release_episodes'] for v in reports),
            recontact_episodes=sum(v['placing']['recontact_episodes'] for v in reports),
            failures={k:sum(v['overall']['failures'][k] for v in reports) for k in reports[0]['overall']['failures']})
    out=dict(parent_transitions=16777216,additional_transitions=33554432,
        cumulative_transitions=50331648,restored_policy=True,
        optimizer_and_rng_restarted=True,rewards_resets_controller_unchanged=True,
        intermediate_evaluations=False,training_seconds=training['elapsed_seconds'],
        real_robot=False,results=rows,
        definitions=dict(ordinary='Original geometric inside predicate at final step.',
            controlled='Qualified first release TCP50-60mm; stable inner-floor contact and relative linear speed<=20mm/s for100ms; no robot recontact;80mm point clearance for200ms.',
            rates='Final full-episode outcomes for384 episodes per start condition; controller/physics failures count as failures.',
            limitations='One training seed; three evaluation seeds are not independent training replications. Additional training and optimizer restart occur together.'))
    (DOC/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
