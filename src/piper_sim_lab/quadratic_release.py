"""Opt-in squared placement costs and fast failure reset with tail settlement.

Old task/configurations are unchanged. Task failure receives the analytic
discounted cost of the remaining finite horizon, then resets immediately.
The pure absorbing reference is retained only for equivalence tests. No training
or real-robot I/O is performed by this module.
"""
import jax.numpy as jp
import math
import mujoco

from .controlled_release import ControlledRelease
from .task_failure import DEFAULTS as TASK_DEFAULTS, task_failure_memory, task_failure_update


EVENT_METRICS = (
    'reward_reach','reward_lift','reward_carry','reward_inside',
    'gentle_reward','gentle_quality','gentle_retreat','gentle_success','controlled_success','success',
    'release_bonus_progress','release_bonus_dwell','release_bonus_release',
    'release_bonus_landing','release_bonus_withdrawal','gentle_recontacts',
    'gentle_contact_seconds','gentle_impact_cost','gentle_recontact_cost',
    'action_penalty','action_rate_penalty','coord_completed','coord_rejected',
    'gripper_goal_switch')


def penalty_errors(f):
    return (jp.maximum(f['xy']-.017,0.),
            jp.maximum(.05-f['tcp_height'],jp.maximum(f['tcp_height']-.06,0.)))


def penalty_memory(base):
    z=jp.array(0.,jp.float32);no=jp.array(False)
    return dict(base,p_active=base['carried'],p_qualified=no,p_locked=no,
        p_detach_xy=z,p_detach_z=z,p_locked_xy=z,p_locked_z=z)


def penalty_update(old,new,after):
    ex,ez=penalty_errors(after)
    first=(old['carried']&~(after['left']|after['right'])&
           (old['lost_contact_time']==0)&~old['released'])
    dx=jp.where(first,ex,old['p_detach_xy']);dz=jp.where(first,ez,old['p_detach_z'])
    invalid=(~old['released']&new['released']&~new['release_paid']&~old['p_locked'])
    return dict(new,p_active=old['p_active']|new['carried'],
        p_qualified=old['p_qualified']|new['release_paid'],
        p_locked=old['p_locked']|invalid,p_detach_xy=dx,p_detach_z=dz,
        p_locked_xy=jp.where(invalid,dx,old['p_locked_xy']),
        p_locked_z=jp.where(invalid,dz,old['p_locked_z']))


def pending_cost(memory,f):
    ex,ez=penalty_errors(f)
    ex=jp.where(memory['p_locked'],memory['p_locked_xy'],ex)
    ez=jp.where(memory['p_locked'],memory['p_locked_z'],ez)
    cost=.6+.8*(ex/.05)**2+.8*ez/(.04+ez)
    return jp.where(memory['p_active']&~memory['p_qualified'],cost,0.)


def failure_cost(pending,minimum=.6):
    # Failure before carry or after a release is still not a task completion.
    return jp.maximum(pending,minimum)


def remaining_cost(cost,remaining,gamma=.99):
    """Cost AFTER the current failure step, expressed at that step's discount."""
    if gamma==1.:
        return cost*remaining
    # gamma is a static config float. Compute log before float32 rounding so
    # long remaining horizons do not amplify the rounded .99 logarithm error.
    return cost*gamma*(-jp.expm1(remaining*math.log(gamma)))/(1.-gamma)


def replace_base_with_potential(reward, credited_base, raw_score, previous, done,
                                gamma=.99, maximum=4.):
    """Replace repeated state payments with discounted potential shaping.

    Potential records the episode's best score, so returning to an old pose
    never produces another positive increment. Reset potential is zero.
    Every true termination, including task/controller
    failures and the finite horizon, has zero successor potential. Thus the
    shaping return telescopes to zero under the learner's discount; it cannot
    be farmed by waiting, cycling, or ending an episode early. Event bonuses
    and all costs in ``reward - credited_base`` are retained exactly.
    """
    potential=jp.where(done>0,0.,jp.maximum(previous,jp.clip(raw_score,0.,maximum)))
    shaping=gamma*potential-previous
    return reward-credited_base+shaping,potential,shaping


def clear_events(metrics):
    return {k:(jp.zeros_like(v) if k in EVENT_METRICS else v) for k,v in metrics.items()}


def absorb_failure(state,horizon):
    """Pure function shared by the runtime and numerical/GAE boundary tests."""
    info=dict(state.info)
    finished=info['penalty_episode_steps']>=horizon
    charge=jp.where(finished,0.,info['failure_cost'])
    steps=jp.minimum(info['penalty_episode_steps']+1,horizon)
    info.update(penalty_episode_steps=steps,
                penalty_total=info['penalty_total']+charge)
    metrics=dict(clear_events(state.metrics),penalty_cost=charge,
        failure_cost=info['failure_cost'],failure_onset=jp.array(0.),
        failure_absorbing=jp.array(1.),penalty_positive_base=jp.array(0.))
    return state.replace(reward=-charge,done=(steps>=horizon).astype(jp.float32),
                         info=info,metrics=metrics)


def finish_normal_step(state,updated,cost,horizon,*,early_reset=False,gamma=.99,failure_min_cost=.6):
    """Shared runtime bookkeeping; tests use this with a minimal task."""
    failed=updated.info['task_failed']
    interrupted=updated.info['controller_failed'] & ~failed
    fixed=failure_cost(cost,failure_min_cost)
    event_cost=sum(updated.metrics[k] for k in (
        'action_penalty','action_rate_penalty','gentle_impact_cost','gentle_recontact_cost'))
    reward=jp.where(failed,-fixed-event_cost,updated.reward-cost)
    charge=jp.where(failed,fixed,cost)
    steps=state.info['penalty_episode_steps']+1
    tail=jp.where(failed,remaining_cost(fixed,jp.maximum(horizon-steps,0),gamma),0.) if early_reset else jp.array(0.)
    reward-=tail
    info=dict(updated.info,failure_absorbing=failed,failure_cost=jp.where(failed,fixed,0.),
        penalty_episode_steps=steps,penalty_total=state.info['penalty_total']+charge+tail)
    metrics={k:jp.where(failed,jp.zeros_like(v),v) if k in EVENT_METRICS else v
             for k,v in updated.metrics.items()}
    for k in ('action_penalty','action_rate_penalty','gentle_impact_cost','gentle_recontact_cost'):
        metrics[k]=updated.metrics[k]
    metrics.update(penalty_cost=charge,failure_cost=info['failure_cost'],
        failure_onset=failed.astype(jp.float32),failure_absorbing=failed.astype(jp.float32),
        controller_interruption=interrupted.astype(jp.float32),
        penalty_positive_base=jp.where(failed,0.,updated.metrics['gentle_reward']))
    if early_reset:
        info['failure_absorbing']=jp.array(False)
        already=state.info.get('failure_settled',jp.array(False))|(state.info['penalty_episode_steps']>=horizon)
        info['failure_settled']=state.info.get('failure_settled',jp.array(False))|failed
        info['failure_cost']=jp.where(already,state.info['failure_cost'],info['failure_cost'])
        info['penalty_episode_steps']=jp.where(already,state.info['penalty_episode_steps'],steps)
        info['penalty_total']=jp.where(already,state.info['penalty_total'],info['penalty_total'])
        reward=jp.where(already,0.,reward)
        metrics.update(failure_absorbing=jp.array(0.),failure_tail_cost=tail,
                       failure_early_reset=failed.astype(jp.float32),failure_cost=info['failure_cost'])
        for key in (*EVENT_METRICS,'penalty_cost','failure_tail_cost','failure_onset',
                    'failure_early_reset','penalty_positive_base'):
            if key in metrics:metrics[key]=jp.where(already,0.,metrics[key])
    invalid=(updated.metrics['nonfinite']>0)|(updated.metrics['capacity_overflow']>0)
    done=((steps>=horizon)|invalid|interrupted|(failed|already if early_reset else False)).astype(jp.float32)
    return updated.replace(info=info,metrics=metrics,reward=reward,done=done)


class QuadraticRelease(ControlledRelease):
    def __init__(self,config_overrides=None):
        options=dict(config_overrides or {})
        horizon=options.pop('penalty_horizon_steps',200)
        gamma=options.pop('failure_discount',.99)
        bounded_base=options.pop('bounded_base_potential',False)
        potential_max=options.pop('base_potential_max',4.)
        cost_per_second=options.pop('unfinished_cost_per_second',False)
        if not isinstance(cost_per_second,bool):
            raise ValueError('unfinished_cost_per_second must be boolean')
        if not isinstance(bounded_base,bool):
            raise ValueError('bounded_base_potential must be boolean')
        if not math.isfinite(potential_max) or potential_max<=0:
            raise ValueError('base_potential_max must be positive finite')
        self.task_failure_settings={k:float(options.pop(k,v)) for k,v in TASK_DEFAULTS.items()}
        if any(not math.isfinite(v) or v<=0 for v in self.task_failure_settings.values()):
            raise ValueError('task failure thresholds must be positive finite')
        duration=int(horizon)*float(options.get('ctrl_dt',.1))
        if self.task_failure_settings['task_grasp_deadline']>=duration:
            raise ValueError('task_grasp_deadline must be below the episode duration')
        if not 0.<gamma<=1.:
            raise ValueError('failure_discount must be in (0,1]')
        if int(horizon)!=horizon or horizon<=0:
            raise ValueError('positive integer penalty_horizon_steps required')
        options.setdefault('release_balanced_rewards',True)
        options.setdefault('release_require_stable_target',True)
        options.setdefault('release_require_landing_for_retreat',True)
        if not options['release_balanced_rewards']:
            raise ValueError('quadratic release requires balanced release rewards')
        super().__init__(options)
        self.finite_horizon_steps=int(horizon)
        self.failure_discount=float(gamma)
        self.bounded_base_potential=bounded_base
        self.base_potential_max=float(potential_max)
        self.unfinished_cost_per_second=cost_per_second
        with self._config.unlocked():
            self._config.penalty_horizon_steps=int(horizon)
            self._config.episode_length=int(horizon)
            self._config.failure_discount=float(gamma)
            self._config.bounded_base_potential=bounded_base
            self._config.base_potential_max=float(potential_max)
            self._config.unfinished_cost_per_second=cost_per_second
            for key,value in self.task_failure_settings.items():
                self._config[key]=value

    def _configure_task_spec(self,spec):
        super()._configure_task_spec(spec)
        if self._config.impl!='jax':
            sensor=spec.add_sensor(name='task_cube_table',type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=mujoco.mjtObj.mjOBJ_BODY,objname='block',
                reftype=mujoco.mjtObj.mjOBJ_GEOM,refname='floor')
            sensor.intprm=[1,0,1]

    def _place_contacts(self,data):
        contacts=super()._place_contacts(data)
        if self._config.impl!='jax':
            adr=int(self._mj_model.sensor('task_cube_table').adr[0])
            contacts['table']=data.sensordata[adr]>0
        else:
            contact=data._impl.contact
            ga,gb=contact.geom[:,0],contact.geom[:,1]
            ba,bb=self._geom_body[ga],self._geom_body[gb]
            active=(contact.dist<=0)&(contact.efc_address>=0)
            contacts['table']=jp.any(active&(((ga==self._floor_geom_id)&(bb==self._block_id))|
                                                         ((gb==self._floor_geom_id)&(ba==self._block_id))))
        return contacts

    def _place_features(self,data):
        features=super()._place_features(data)
        features['table_speed']=jp.linalg.norm(data.qvel[self._block_qvel_adr:self._block_qvel_adr+3])
        return features

    def _reset_controller(self,state):
        state=super()._reset_controller(state)
        z=jp.array(0.,jp.float32)
        placement=task_failure_memory(penalty_memory(state.info['placement']),state.info['reset_stage'])
        state=state.replace(info=dict(state.info,placement=placement,
            task_failed=jp.array(False),task_failure_reason=jp.array(0,jp.int32),
            failure_absorbing=jp.array(False),failure_cost=z,
            failure_settled=jp.array(False),penalty_episode_steps=jp.array(0,jp.int32),penalty_total=z),
            metrics=dict(state.metrics,penalty_cost=z,failure_cost=z,
                failure_onset=z,failure_absorbing=z,penalty_positive_base=z,
                failure_tail_cost=z,failure_early_reset=z,controller_interruption=z,
                task_failure_reason=z,task_grasp_timeout=z,task_outside_drop=z,task_grasp_confirmed=z))
        if self.bounded_base_potential:
            state=state.replace(info=dict(state.info,base_potential=z),
                metrics=dict(state.metrics,base_raw_score=z,base_potential=z,base_shaping=z,
                    gripper_request_open=z,gripper_request_close=z,
                    empty_close_request=z,held_open_request=z))
        return state

    def _task_substep(self,previous,current,action,task):
        old=task[0]
        new,trace,index=super()._task_substep(previous,current,action,task)
        features=self._place_features(current)
        new=penalty_update(old,new,features)
        return task_failure_update(old,new,features,self.sim_dt,self.task_failure_settings),trace,index

    def _normal_step(self,state,action):
        updated=super().step(state,action)
        memory=updated.info['placement'];reason=memory['tf_reason']
        updated=updated.replace(info=dict(updated.info,task_failed=memory['tf_failed'],task_failure_reason=reason),
            metrics=dict(updated.metrics,task_failure_reason=reason.astype(jp.float32),
                task_grasp_timeout=(reason==1).astype(jp.float32),
                task_outside_drop=(reason==2).astype(jp.float32),
                task_grasp_confirmed=memory['tf_grasped'].astype(jp.float32)))
        scale=self.dt if self.unfinished_cost_per_second else 1.
        cost=scale*pending_cost(updated.info['placement'],self._place_features(updated.data))
        final=finish_normal_step(state,updated,cost,self.finite_horizon_steps,
                                 early_reset=True,gamma=self.failure_discount,failure_min_cost=.6*scale)
        if self.bounded_base_potential:
            already=(state.info['failure_settled']|
                     (state.info['penalty_episode_steps']>=self.finite_horizon_steps))
            # Task failure already suppresses the positive base and all event
            # credits. Do not subtract that base a second time on termination.
            raw=updated.metrics['gentle_reward']
            credited=jp.where(updated.info['task_failed']|already,0.,raw)
            reward,potential,shaping=replace_base_with_potential(
                final.reward,credited,raw,state.info['base_potential'],final.done,
                self.failure_discount,self.base_potential_max)
            final=final.replace(reward=reward,
                info=dict(final.info,base_potential=potential),
                metrics=dict(final.metrics,base_raw_score=raw,
                    base_potential=potential,base_shaping=shaping,
                    penalty_positive_base=jp.maximum(shaping,0.),
                    gripper_request_open=(action[6]>=self._config.gripper_threshold).astype(jp.float32),
                    gripper_request_close=(action[6]<=-self._config.gripper_threshold).astype(jp.float32),
                    empty_close_request=((action[6]<=-self._config.gripper_threshold)&
                        (state.info['reset_stage']<2)).astype(jp.float32),
                    held_open_request=((action[6]>=self._config.gripper_threshold)&
                        (state.info['reset_stage']>=2)).astype(jp.float32)))
        return final

    def step(self,state,action):
        # One normal batched physics call; the existing AutoResetWrapper handles
        # failed slots immediately. No absorbed/dummy tail physics is run.
        return self._normal_step(state,action)
