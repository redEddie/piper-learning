"""Validate bounded best-score shaping and replay old first-episode rewards.

This checks accounting/incentives, not physical feasibility or learned success.
No policy training or robot I/O is performed.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import yaml

import jax
import jax.numpy as jp
import numpy as np

from piper_sim_lab.quadratic_release import replace_base_with_potential
from piper_sim_lab.quadratic_penalty_audit import Ledger, discounted

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/show103/qp-v1-a5-j100-fresh-50m'
OUT=ROOT/'docs/show103/bounded-base-potential'
OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('reward_scenes',ROOT/'scripts/show103/audit_quadratic_penalty.py')
scenes=importlib.util.module_from_spec(spec)
spec.loader.exec_module(scenes)
cfg=json.loads((RUN/'config.json').read_text())
candidate=yaml.safe_load((ROOT/'configs/show103/qp-v1-a5-j100-bounded-base.yaml').read_text())
scenes.C=SimpleNamespace(**{**cfg['resolved_environment'],**candidate['environment']})


@jax.jit
def reprice(rewards,scores,done,valid):
    def step(previous,x):
        old,score,terminal,alive=x
        new,potential,shaping=replace_base_with_potential(old,score,score,previous,terminal)
        return potential,(jp.where(alive,new,0.),jp.where(alive,shaping,0.))
    return jax.lax.scan(step,jp.zeros_like(rewards[0]),(rewards,scores,done,valid))[1]


def synthetic(name):
    fs,actions=scenes.features(name)
    final,tr=jax.device_get(scenes.production(fs,actions))
    events={k:v.reshape(-1) for k,v in tr['events'].items()}
    ledger=Ledger(active=True)
    cost=[]
    for i in range(4000):
        ledger.observe(**{k:events[k][i] for k in events},
            xy=float(fs['xy'][i+1]),tcp_height=float(fs['tcp_height'][i+1]))
        cost.append(ledger.cost(float(fs['xy'][i+1]),float(fs['tcp_height'][i+1])))
    cost=np.asarray(cost).reshape(200,20)[:,-1]
    old=np.asarray(tr['reward'])-cost
    done=np.zeros(200,bool);done[-1]=True
    new,shaping=jax.device_get(reprice(jp.asarray(np.asarray(tr['reward'])-.1*cost),tr['score'],jp.asarray(done),jp.ones(200,bool)))
    bonus={k:float(np.asarray(v).sum()) for k,v in tr['bonus'].items()}
    return dict(old_total=float(old.sum()),new_total=float(new.sum()),
        old_discounted=discounted(old),new_discounted=discounted(new),
        shaping_discounted=discounted(shaping),positive_shaping_sum=float(np.maximum(shaping,0.).sum()),
        bonuses=bonus,release_paid=bool(final['release_paid']),landing_paid=bool(final['landing_paid']),
        withdrawal_paid=bool(final['withdrawal_paid']))


names=['far_wait','align_only','align_lower_wait','complete','late_release',
       'xy_roundtrip','height_roundtrip','bad_high_drop','bad_drop_then_lower','bad_drop_regrasp']
synthetic_results={name:synthetic(name) for name in names}
checks=dict(
    complete_beats_wait=synthetic_results['complete']['new_discounted']>synthetic_results['align_lower_wait']['new_discounted'],
    fast_beats_late=synthetic_results['complete']['new_discounted']>synthetic_results['late_release']['new_discounted'],
    late_beats_wait=synthetic_results['late_release']['new_discounted']>synthetic_results['align_lower_wait']['new_discounted'],
    alignment_beats_far_wait=synthetic_results['align_only']['new_discounted']>synthetic_results['far_wait']['new_discounted'],
    lowering_beats_aligned_wait=synthetic_results['align_lower_wait']['new_discounted']>synthetic_results['align_only']['new_discounted'],
    xy_roundtrip_no_gain=synthetic_results['xy_roundtrip']['new_discounted']<=synthetic_results['align_lower_wait']['new_discounted'],
    height_roundtrip_no_gain=synthetic_results['height_roundtrip']['new_discounted']<=synthetic_results['align_lower_wait']['new_discounted'],
    positive_increment_budget=all(r['positive_shaping_sum']<=4.+1e-5 for r in synthetic_results.values()),
    net_total_upper_bound=all(r['new_total']<=11.+1e-5 for r in synthetic_results.values()),
    shaping_terminal_accounting=all(abs(r['shaping_discounted'])<2e-4 for r in synthetic_results.values()))

replay={}
for mode in ['empty_hand','pregrasp','grasp','preinside','near','mixed']:
    old_totals=[];new_totals=[];old_discounted=[];new_discounted=[]
    for seed in [2026,2027,2028]:
        tr=np.load(RUN.parent/'qp-v1-a5-j100-final-evaluation'/f'{mode}-s{seed}-trace.npz')
        repriced=tr['reward']+.9*(tr['penalty_cost']+tr['failure_tail_cost'])
        new,shaping=jax.device_get(reprice(jp.asarray(repriced),jp.asarray(tr['gentle_reward']),
            jp.asarray(tr['done']),jp.asarray(tr['valid'])))
        old=tr['reward']*tr['valid']
        old_totals.extend(old.sum(0));new_totals.extend(new.sum(0))
        old_discounted.extend((old*.99**np.arange(200)[:,None]).sum(0))
        new_discounted.extend((new*.99**np.arange(200)[:,None]).sum(0))
        assert np.max(new.sum(0))<=11.+1e-4
        assert np.max(np.abs((shaping*.99**np.arange(200)[:,None]).sum(0)))<2e-4
    replay[mode]=dict(episodes=len(old_totals),old_total_mean=float(np.mean(old_totals)),
        new_total_mean=float(np.mean(new_totals)),new_total_max=float(np.max(new_totals)),
        old_discounted_mean=float(np.mean(old_discounted)),new_discounted_mean=float(np.mean(new_discounted)))

out=dict(status='passed' if all(checks.values()) else 'failed',
    formula='Phi=episode-record max(clip(raw_state_score,0,4)); F=.99*Phi_next-Phi_previous; Phi_reset=Phi_terminal=0. Replace repeated raw state score with F.',
    upper_bound='Positive shaping increments <=4; one-shot events <=11; undiscounted episode NET reward <=11, because total shaping <=0 and all retained costs are nonnegative.',
    notes=['Same frozen trajectories repriced; this does not predict the newly trained policy.',
        'Synthetic scenarios use production5ms event and100ms reward functions, not physical rollouts.',
        'Unfinished XY/height cost is a per-second rate integrated over100ms (0.1x per step), including failure min and tail. Other action/contact costs are retained. Large negative costs remain possible.',
        'True termination cancels potential at finite horizon, task failure and controller interruption.',
        'No reward for leaving and returning to an already recorded score; existing action/action-rate penalties retained.'],
    checks=checks,synthetic=synthetic_results,frozen_trajectory_replay=replay,
    training=False,real_robot=False)
(OUT/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
assert all(checks.values()),checks
