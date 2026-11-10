"""Use runtime failure accounting to close the saved numerical counterexample."""
from pathlib import Path
import json
import jax.numpy as jp
import numpy as np
from mujoco_playground._src.mjx_env import State

from piper_sim_lab.quadratic_release import EVENT_METRICS,finish_normal_step,absorb_failure
from piper_sim_lab.quadratic_penalty_audit import discounted

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/show103/failure-tail'
DATA=ROOT/'docs/show103/quadratic-penalty/traces.npz'


def failure_path(rewards,scores,costs,fail_step,fast=False):
    index=fail_step-1;z=jp.array(0.,jp.float32)
    metrics={k:z for k in EVENT_METRICS}
    metrics.update(nonfinite=z,capacity_overflow=z,penalty_cost=z,
        failure_cost=z,failure_onset=z,failure_absorbing=z,penalty_positive_base=z)
    old=State(data=jp.zeros(1),obs=jp.zeros(1),reward=z,done=z,metrics=metrics,
        info=dict(task_failed=jp.array(False),controller_failed=jp.array(False),failure_absorbing=jp.array(False),
            failure_cost=z,penalty_episode_steps=jp.array(index,jp.int32),penalty_total=z))
    # Saved wait/drop cases at these steady steps have only the .001 raw-action
    # cost. The onset suppresses state/bonus credit and keeps actual action cost.
    updated=old.replace(reward=jp.array(rewards[index]+costs[index]),
        metrics=dict(metrics,gentle_reward=jp.array(scores[index]),action_penalty=jp.array(.001)),
        info=dict(old.info,task_failed=jp.array(True)))
    state=finish_normal_step(old,updated,jp.array(costs[index]),200,early_reset=fast)
    result=list(rewards[:index])+[float(state.reward)]
    if fast:
        assert state.done==1
        return np.asarray(result),float(state.info['failure_cost'])
    while int(state.info['penalty_episode_steps'])<200:
        state=absorb_failure(state,200);result.append(float(state.reward))
    assert len(result)==200 and state.done==1
    return np.asarray(result),float(state.info['failure_cost'])


def main():
    out={}
    with np.load(DATA) as a:
        for name in ['far_wait','bad_outside_drop']:
            r=a[name+'_raw'];scores=a[name+'_score'];costs=a[name+'_cost']
            full=discounted(r);rows=[]
            for k in [1,10,50,100,199]:
                frozen,c=failure_path(r,scores,costs,k)
                fast,_=failure_path(r,scores,costs,k,fast=True)
                before=discounted(r[:k])
                flat50=before-50.*.99**(k-1)
                after=discounted(frozen)
                settled=discounted(fast)
                rows.append(dict(failure_time_s=k*.1,old_zero_tail_return=before,
                    fixed_minus50_return=flat50,absorbing_tail_return=after,
                    fast_reset_return=settled,fast_absorbing_difference=settled-after,
                    fast_failure_step_reward=float(fast[-1]),remaining_steps=200-k,
                    original20s_return=full,old_escape_advantage=before-full,
                    flat50_escape_advantage=flat50-full,new_escape_advantage=after-full,
                    repeated_cost_per_step=c,max_failure_onset_or_tail_cost=float(-frozen[k-1:].min())))
                assert after<=full+1e-4
                assert abs(settled-after)<2e-4
                assert -frozen[k:].max()>=c-1e-6 if k<200 else True
            assert all(rows[i]['absorbing_tail_return']<=rows[i+1]['absorbing_tail_return'] for i in range(len(rows)-1))
            out[name]=rows
    report=dict(scope='Hypothetical task-failure events on saved synthetic counterexamples recomputed with actual runtime finish_normal_step/absorb_failure functions; not learning or a physical-failure feasibility test.',
        rule='Runtime: zero failure-step positive scores, charge current cost plus analytic discounted remaining cost, then reset immediately. Pure absorbing reference repeats the same cost until200 steps for equivalence checking.',
        horizon_steps=200,policy_dt_s=.1,gamma=.99,results=out,
        interpretation='Fixed-50 does not cover early large-distance failures. Exact tail settlement has the same discounted return as the absorbing reference and avoids occupying failed slots for the rest of the horizon.',
        termination='Failure true-terminates immediately after settlement; natural finite end is step200. Discount/horizon/action-repeat mismatches rejected. Physics NaN/overflow still abort through RuntimeSafety.',
        learning_caveat='Return equivalence does not mean identical GAE(lambda<1), loss variance or gradient distribution. Settlement can be large; it is not silently clipped to50.',
        training_run=False,real_robot=False,status='passed_for_saved_counterexamples')
    DOC.mkdir(parents=True,exist_ok=True)
    (DOC/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
