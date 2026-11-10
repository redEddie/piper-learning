"""Frozen MLP evaluation: A-start completion, reset groups and post-success chatter."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import time
from piper_sim_lab.runtime import configure,write_json
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--run',type=Path,required=True)
ap.add_argument('--checkpoint',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--num-envs',type=int,default=128)
ap.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
ap.add_argument('--modes',nargs='+',choices=['empty_hand','mixed','preinside','near'],default=['empty_hand','mixed'])
args=ap.parse_args()
configure('gpu')
import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground import wrapper
from piper_sim_lab.checkpoint import load_policy
from piper_sim_lab.tasks import make_env
from piper_sim_lab.progress_eval import summarize_progress
cfg=json.loads((args.run/'config.json').read_text())
args.output.mkdir(parents=True,exist_ok=False)
reports={}
for mode in args.modes:
    c=dict(cfg['resolved_environment'],batch_capacity=args.num_envs)
    if mode=='empty_hand': c['reset_stage_probs']=[1.,0.,0.,0.]
    if mode=='preinside': c.update(reset_stage_probs=[0.,0.,0.,1.],gentle_near_fraction=0.)
    if mode=='near': c.update(reset_stage_probs=[0.,0.,0.,1.],gentle_near_fraction=1.)
    env=make_env(c)
    policy=load_policy(args.checkpoint,observation_size=env.observation_size,action_size=7,deterministic=True)
    # Distribution metadata only; executed actions remain the deterministic
    # policy above. These extra reads do not change actions or the controller.
    diagnostic_policy=load_policy(args.checkpoint,observation_size=env.observation_size,action_size=7,deterministic=False) if c['task']=='controlled_release' else None
    # Physical parameters are random draws per slot; object/perception reset per episode.
    dr=wrapper.BraxDomainRandomizationVmapWrapper(env,functools.partial(env.physics_randomization_fn,
        rng=jax.random.split(jax.random.key(args.seeds[0]+7),args.num_envs)))
    @jax.jit
    def rollout(state):
        def step(carry,_):
            state,previous_goal=carry
            distribution_sample={}
            if diagnostic_policy is not None:
                _,extra=diagnostic_policy(state.obs,jax.random.key(103))
                fchoice=jax.vmap(env._place_features)(state.data)
                distribution_sample=dict(policy_distribution=extra['distribution_params'],
                    choice_tcp_height=fchoice['tcp_height'],choice_xy=fchoice['xy'],
                    choice_phase=state.info['coord_phase'],choice_goal=state.info['gripper_goal'],
                    choice_held=state.info['placement']['carried']&~state.info['placement']['released']&(fchoice['left']|fchoice['right']))
            action,_=policy(state.obs,jax.random.key(0))
            state=dr.step(state,action)
            p=jax.vmap(env._predicates)(state.data)
            if c.get('gripper_goal_obs',False):
                goal=state.info['gripper_goal']
                hold=state.metrics['gripper_hold']
            else:
                goal=jp.where(action[:,6]>0.,env._config.gripper_max,0.)
                hold=jp.zeros(args.num_envs)
            sample=dict(**p,**distribution_sample,reward=state.reward,nonfinite=state.metrics['nonfinite'],
                overflow=state.metrics['capacity_overflow'],controller_failed=state.info.get('controller_failed',jp.zeros(args.num_envs,bool)),
                controller_reason=state.info.get('controller_failure_reason',jp.zeros(args.num_envs,jp.int32)),action=action,
                goal=goal,goal_switch=(goal!=previous_goal).astype(jp.float32),
                hold=hold,qpos=state.data.qpos,qvel=state.data.qvel,
                ctrl=state.data.ctrl,tcp=state.data.site_xpos[:,env._tcp_id],
                block=state.data.xpos[:,env._block_id],bowl=state.data.xpos[:,env._bowl_id],
                mocap=state.data.mocap_pos,tracking=state.info.get('tracking_trace',jp.zeros((args.num_envs,env.n_substeps,24))))
            if 'coord_phase' in state.info:
                sample.update(coord_phase=state.info['coord_phase'],coord_trace=state.info['coord_trace'],operations_completed=state.info['operations_completed'],candidate_rejections=state.info['candidate_rejections'])
            if 'placement' in state.info:
                memory=state.info['placement']
                sample.update(gentle_success=state.metrics['gentle_success'],place_active=memory['active'],place_had_grip_contact=memory['had_contact'],
                    place_qualified=memory['qualified_release'],place_premature=memory['premature'],
                    place_impact_peak=memory['impact_peak'],place_supported=memory['supported'],
                    place_released=memory['released'],place_release_height=memory['release_height'],
                    place_floor_seen=memory['floor_seen'],place_quality=state.metrics['gentle_quality'],
                    gentle_trace=state.info['gentle_trace'],near_reset=state.info['near_reset'],
                    recontacts=memory['recontacts'],recontact_seen=memory['recontact_seen'],
                    withdrawal_time=memory['clear_time'],
                    release_tcp_height=memory.get('release_tcp_height',jp.zeros(args.num_envs)),
                    locked_grip_reference_speed=state.metrics['grip_arm_locked_reference_speed'],
                    opening_arm_unlocked=state.info['grip_motion_unlocked'])
                if c['task']=='controlled_release':
                    sample['tcp_height']=jax.vmap(env._place_features)(state.data)['tcp_height']
            return (state,goal),sample
        return jax.lax.scan(step,(state,state.data.ctrl[:,env._gripper_ctrl_id]),None,length=cfg['training']['episode_length'])
    for seed in args.seeds:
        # Set DR key per evaluation seed, matching the original SHOW-76 evaluator.
        dr=wrapper.BraxDomainRandomizationVmapWrapper(env,functools.partial(env.physics_randomization_fn,
            rng=jax.random.split(jax.random.key(seed+7),args.num_envs)))
        # Clear compilation so the closed-over DR model is re-read for each seed.
        # Preserve the same shapes and 80-step horizon.
        rollout.clear_cache()
        state=jax.jit(dr.reset)(jax.random.split(jax.random.key(seed),args.num_envs))
        initial=jax.tree.map(np.asarray,jax.vmap(env._predicates)(state.data))
        stages=np.asarray(state.info['reset_stage'])
        started=time.monotonic()
        _,trace=jax.block_until_ready(rollout(state));tr=jax.tree.map(np.asarray,trace)
        report=summarize_progress(stages,initial,tr,env.dt)
        inside=tr['inside'].astype(bool)
        ever_seen=np.maximum.accumulate(inside,axis=0)
        # Count transitions after success was already reached at the preceding step.
        post=np.r_[np.zeros((1,args.num_envs),bool),ever_seen[:-1]]
        valid=~(tr['controller_failed'].astype(bool)|tr['nonfinite'].astype(bool)|tr['overflow'].astype(bool)).any(0)
        post &= valid[None,:]
        post_seconds=post.sum(0)*env.dt
        switches=(tr['goal_switch']*post).sum(0)
        count=post.sum()
        width_ranges=[]
        for episode in range(args.num_envs):
            if post_seconds[episode]>=.5:
                widths=tr['width'][post[:,episode],episode]
                width_ranges.append(float(np.ptp(widths)*1000))
        report['post_success']=dict(scope='after first inside predicate; valid physics/controller episodes; ranges require >=0.5s observation',
            mean_actual_width_range_mm=float(np.mean(width_ranges)) if width_ranges else None,
            median_actual_width_range_mm=float(np.median(width_ranges)) if width_ranges else None,
            episodes_with_observation=int((post_seconds>0).sum()),
            observed_seconds=float(post_seconds.sum()),target_switches=int(switches.sum()),
            target_switches_per_second=float(switches.sum()/post_seconds.sum()) if count else None,
            hold_fraction=float((tr['hold']*post).sum()/count) if count else None,
            target_switches_by_episode=switches.astype(int).tolist(),
            observed_seconds_by_episode=post_seconds.tolist())
        if 'gentle_success' in tr:
            gentle=(tr['gentle_success'][-1]>0)&valid
            floor=tr['place_floor_seen'][-1].astype(bool)&valid
            released=tr['place_released'][-1].astype(bool)&valid
            peak=tr['place_impact_peak'][-1]
            report['placing']=dict(gentle_final_successes=int(gentle.sum()),gentle_final_success_rate=float(gentle.mean()),
                basic_final_successes=report['overall']['final_successes'],
                by_initialization={name:dict(episodes=int((stages==i).sum()),gentle_successes=int((gentle&(stages==i)).sum())) for i,name in enumerate(['empty_hand','pregrasp','grasp','preinside'])},
                floor_contact_episodes=int(floor.sum()),release_episodes=int(released.sum()),
                qualified_release_episodes=int((tr['place_qualified'][-1]&valid).sum()),
                premature_release_episodes=int((tr['place_premature'][-1]&valid).sum()),
                peak_contact_speed_m_s_mean=float(peak[floor].mean()) if floor.any() else None,
                peak_contact_speed_m_s_median=float(np.median(peak[floor])) if floor.any() else None,
                peak_contact_speed_m_s_p90=float(np.quantile(peak[floor],.9)) if floor.any() else None,
                detected_release_height_mm_mean=float(tr['place_release_height'][-1,released].mean()*1000) if released.any() else None,
                measurement='5ms per-world contacts; pre-contact relative body-linear velocity projected on bowl floor normal')
            report['placing'].update(recontact_episodes=int((tr['recontact_seen'][-1]&valid).sum()),
                recontact_events=int(tr['recontacts'][-1,valid].sum()),
                near_reset_episodes=int(tr['near_reset'][0].sum()),
                gentle_before_withdrawal=int((tr['place_qualified'][-1]&~tr['place_premature'][-1]&valid&(peak<=c.get('release_impact_limit',.1))).sum()),
                release_tcp_height_mm_mean=float(tr['release_tcp_height'][-1,released].mean()*1000) if released.any() else None,
                release_tcp_height_mm_median=float(np.median(tr['release_tcp_height'][-1,released])*1000) if released.any() else None,
                criterion='TCP50-60mm release then landing and contact-free withdrawal' if c['task']=='controlled_release' else 'supported release then contact-free withdrawal')
            if 'locked_grip_reference_speed' in tr:
                report['max_locked_grip_reference_speed']=float(tr['locked_grip_reference_speed'].max())
        if 'coord_trace' in tr:
            ct=tr['coord_trace']
            report['coordinator']=dict(max_reference_jerk=float(ct[...,7].max()),max_reference_speed=float(ct[...,5].max()),max_reference_acceleration=float(ct[...,6].max()),max_reference_speed_during_grip=float(np.where(ct[...,0]==2,ct[...,5],0.).max()),operations_completed=int(tr['operations_completed'][-1].sum()),rejected_proposals=int(tr['candidate_rejections'][-1].sum()),phase_fraction={str(i):float((ct[...,0]==i).mean()) for i in range(3)},timeouts=int((ct[...,11]>0).any(0).any(-1).sum()))
        if 'policy_distribution' in tr:
            from scipy.special import ndtr
            mask=tr['choice_held']&(tr['choice_phase']==0)&(tr['choice_goal']==0)&(tr['choice_xy']<=.017)&(tr['choice_tcp_height']>=.05)&(tr['choice_tcp_height']<=.06)
            loc=tr['policy_distribution'][...,6][mask]
            scale=np.logaddexp(0.,tr['policy_distribution'][...,13][mask])+.001
            prob=ndtr((loc-np.arctanh(.5))/scale)
            report['opening_exploration']=dict(scope='Conditional probabilities at recorded deterministic-evaluation states: held, aligned, TCP50-60mm, MOVE and closed target. Not observed stochastic-training frequencies.',
                state_samples=int(mask.sum()),opening_probability_mean=float(prob.mean()) if len(prob) else None,
                opening_probability_median=float(np.median(prob)) if len(prob) else None,
                opening_probability_max=float(prob.max()) if len(prob) else None,
                raw_location_median=float(np.median(loc)) if len(loc) else None,
                raw_scale_median=float(np.median(scale)) if len(scale) else None,
                deterministic_open_fraction=float((loc>=np.arctanh(.5)).mean()) if len(loc) else None,
                deterministic_upward_fraction=float((tr['action'][...,2][mask]>0).mean()) if mask.any() else None)
        index8=min(79,len(tr['inside'])-1)
        report['success_at_8s']=int((tr['inside'][index8]&valid).sum())
        report['success_ever']=int((tr['inside'].any(0)&valid).sum())
        report.update(seed=seed,elapsed_seconds=time.monotonic()-started,
            initial_scene_sha256=hashlib.sha256(np.asarray(state.data.qpos).tobytes()).hexdigest(),
            max_tracking_gap=float(tr['tracking'][...,18].max()) if c.get('control_mode')=='qp' else None,
            max_reference_acceleration=float(tr['tracking'][...,21].max()) if c.get('control_mode')=='qp' else None,
            min_reference_floor_distance=float(tr['tracking'][...,20].min()) if c.get('control_mode')=='qp' else None)
        name=f'{mode}-s{seed}'
        np.savez_compressed(args.output/f'{name}-trace.npz',reset_stage=stages,**tr,
            **{f'initial_{k}':v for k,v in initial.items()})
        write_json(args.output/f'{name}.json',report);reports[name]=report
        print(json.dumps(dict(evaluation=name,overall=report['overall'],by_initialization=report['by_initialization'],post_success=report['post_success']['target_switches_per_second'])),flush=True)
aggregate={}
for mode in args.modes:
    rows=[v['overall'] for k,v in reports.items() if k.startswith(mode+'-')]
    n=sum(x['episodes'] for x in rows);s=sum(x['final_successes'] for x in rows)
    aggregate[mode]=dict(episodes=n,final_successes=s,final_success_rate=s/n)
    placing=[v['placing'] for k,v in reports.items() if k.startswith(mode+'-') and 'placing' in v]
    if placing:
        gs=sum(x['gentle_final_successes'] for x in placing)
        aggregate[mode].update(gentle_final_successes=gs,gentle_final_success_rate=gs/n)
        if c['task']=='controlled_release':aggregate[mode].update(controlled_final_successes=gs,controlled_final_success_rate=gs/n)
write_json(args.output/'evaluation.json',dict(checkpoint=str(args.checkpoint.resolve()),environment=dict(cfg['resolved_environment'],batch_capacity=args.num_envs),
    seeds=args.seeds,aggregate=aggregate,reports=reports,learning=False,real_robot=False))
print(json.dumps(dict(aggregate=aggregate)),flush=True)
