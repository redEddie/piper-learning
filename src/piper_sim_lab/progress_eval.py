"""Report original task predicates without changing rewards or transitions."""
import numpy as np

STAGES=('empty_hand','pregrasp','grasp','preinside')
FAILURES=('approach','grasp_lift','held_start_lost','transport','place_release',
          'completion_lost','controller','physics')


def summarize_progress(reset_stage,initial,trace,ctrl_dt=.1):
    stage=np.asarray(reset_stage);n=len(stage)
    physics=(trace['nonfinite'].astype(bool)|trace['overflow'].astype(bool)).any(0)
    controller=trace['controller_failed'].astype(bool).any(0)
    valid=~(physics|controller)
    ever={k:np.asarray(initial[k]).astype(bool)|np.asarray(trace[k]).astype(bool).any(0)
          for k in ('pre_grasp','grasp','pre_inside','inside')}
    final=np.asarray(trace['inside'][-1]).astype(bool)&valid
    final_grasp=np.asarray(trace['grasp'][-1]).astype(bool)
    streak=np.zeros(n,int);longest=np.zeros(n,int)
    for flag in np.asarray(trace['grasp']).astype(bool):
        streak=np.where(flag,streak+1,0);longest=np.maximum(longest,streak)
    categories=[];details=[]
    for i in range(n):
        if physics[i]:category,detail='physics','invalid_physics'
        elif controller[i]:category,detail='controller','reference_controller_stopped'
        elif final[i]:category,detail='complete','final_inside'
        elif ever['inside'][i]:category,detail='completion_lost','inside_then_lost_at_horizon'
        elif stage[i]>=2 and not ever['grasp'][i]:category,detail='held_start_lost','held_reset_never_met_grasp_predicate'
        elif not ever['pre_grasp'][i]:category,detail='approach','never_within_original_near_threshold'
        elif not ever['grasp'][i]:category,detail='grasp_lift','near_but_no_grasp_and_lift'
        elif not ever['pre_inside'][i]:
            category='transport'
            detail='transport_not_above' if final_grasp[i] else 'transport_grasp_condition_lost'
        else:
            category='place_release'
            detail='place_grasp_still_true' if final_grasp[i] else 'place_inside_condition_not_met'
        categories.append(category);details.append(detail)
    def row(mask):
        count=int(mask.sum())
        counts={k:sum(c==k and bool(m) for c,m in zip(categories,mask)) for k in ('complete',)+FAILURES}
        assert sum(counts.values())==count
        return dict(episodes=count,final_successes=int(final[mask].sum()),
            final_success_rate=float(final[mask].mean()) if count else None,
            failures={k:counts[k] for k in FAILURES},
            reached={k:int((v&mask&valid).sum()) for k,v in ever.items()})
    # Rates describe observed predicates among eligible starts, not learned
    # independent skills. Exclude abilities supplied by reset construction.
    eligible={
        'approach':(stage<2)&~np.asarray(initial['pre_grasp']).astype(bool),
        'grasp_lift':(stage<2)&ever['pre_grasp'],
        'transport':(stage!=3)&ever['grasp']&~np.asarray(initial['pre_inside']).astype(bool),
        'place_release':(stage==3)|ever['pre_inside'],
    }
    targets=dict(approach='pre_grasp',grasp_lift='grasp',transport='pre_inside',place_release='inside')
    operations={}
    for name,mask in eligible.items():
        wins=mask&ever[targets[name]]&valid
        operations[name]=dict(eligible=int(mask.sum()),reached=int(wins.sum()),
            rate=float(wins.sum()/mask.sum()) if mask.sum() else None)
    return dict(overall=row(np.ones(n,bool)),
        by_initialization={name:row(stage==i) for i,name in enumerate(STAGES)},
        operation_predicate_rates=operations,categories=categories,details=details,
        max_grasp_predicate_duration_seconds=(longest*ctrl_dt).tolist(),
        definitions=dict(final='original inside predicate at final step, controller/physics failures counted as failures',
            failure='exclusive furthest observed progress; not proof of physical cause',
            operations='original 100ms predicates; reset-supplied abilities excluded where applicable'))
