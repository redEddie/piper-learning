"""SHOW103 revised target: TCP50-60mm above bowl inner floor, then withdraw.

This is controlled low-height release, not support-before-release. The actor
still has50 observations and7 outputs. Opening unlock uses measured width,
not simulation-only contacts; contacts remain reward/evaluation labels.
"""
import jax.numpy as jp
from .gentle_placement import GentlePlacement
from .smooth_grip import MOVE,GRIP
from .balanced_release import proximity

DEFAULTS=dict(release_tcp_min=.05,release_tcp_max=.06,release_tcp_speed=.03,
    release_object_speed=.2,release_width_margin=.01,release_flight_seconds=.5,
    release_impact_limit=1.,gentle_near_min=.03,gentle_near_max=.05,
    release_require_stable_target=False,release_target_hold_speed=.01,
    release_require_landing_for_retreat=False, release_balanced_rewards=False)


def release_memory(f, c=None):
    z=jp.array(0.);no=jp.array(False)
    carried=f['left']&f['right']&f['elevated']&~f['support']
    return dict(active=f['above'],carried=carried,had_contact=f['left']|f['right'],
        support_time=z,supported=no,floor_seen=f['support'],impact_peak=z,
        lost_contact_time=z,released=no,qualified_release=no,premature=f['support']&~carried,
        release_height=z,release_tcp_height=z,detach_candidate=no,detach_height=z,
        detach_tcp=z,release_time=z,target_time=z,recontact_seen=no,post_touch=no,
        recontacts=z,recontact_seconds=z,clear_time=z,
        progress_best=(proximity(f,c) if c is not None else z),
        progress_credit=z,target_count=z,release_paid=no,landing_paid=no,withdrawal_paid=no)


def release_update(t,before,after,dt,c):
    both=after['left']&after['right']
    grip=after['left']|after['right']
    carried=t['carried']|(~t['floor_seen']&both&after['elevated'])
    active=t['active']|before['above']|after['above']
    lost=carried&~grip
    first_loss=lost&(t['lost_contact_time']==0)&~t['released']
    tcp_speed=jp.linalg.norm(after['tcp_rel']-before['tcp_rel'])/dt
    opening=after['opening_intent']|(after['grip_reference']>before['grip_reference']+1e-6)
    valid=(carried & opening &
        (after['tcp_height']>=c.release_tcp_min)&(after['tcp_height']<=c.release_tcp_max)&
        (after['xy']<=c.bowl_r_bottom-c.block_half)&(tcp_speed<=c.release_tcp_speed)&
        (before['speed']<=c.release_object_speed))
    candidate=jp.where(first_loss,valid,t['detach_candidate'])
    anchor_h=jp.where(first_loss,before['height'],t['detach_height'])
    anchor_tcp=jp.where(first_loss,after['tcp_height'],t['detach_tcp'])
    lost_time=jp.where(lost,t['lost_contact_time']+dt,0.)
    event=lost&~t['released']&(lost_time>=c.place_release_seconds-1e-6)
    premature=t['premature']|(event&~candidate)|(after['support']&~before['support']&~carried)
    qualified=(t['qualified_release']|(event&candidate))&~premature
    support_now=after['support']&(after['speed']<=c.place_stable_speed)
    if not c.release_balanced_rewards:
        support_now &= after['in_region']
    support_time=jp.where(support_now,t['support_time']+dt,0.)
    impact=jp.maximum(t['impact_peak'],jp.where(after['support']&~before['support'],before['downward_speed'],0.))
    post_release=t['released']|(t['floor_seen']&~t['carried'])
    touch=post_release&(after['robot_cube']|after['robot_bowl'])
    recontact=t['recontact_seen']|touch
    clear=(qualified&~recontact&support_now&(impact<=c.release_impact_limit)&
        (after['clearance']>=c.gentle_clearance))
    target=((after['tcp_height']>=c.release_tcp_min)&(after['tcp_height']<=c.release_tcp_max)&
        (after['xy']<=c.bowl_r_bottom-c.block_half)&grip)
    if c.release_require_stable_target:
        target=target&(tcp_speed<=c.release_target_hold_speed)
    new = dict(t,active=active,carried=carried,had_contact=t['had_contact']|grip,
        support_time=support_time,supported=support_time>=c.place_support_seconds-1e-6,
        floor_seen=t['floor_seen']|after['support'],impact_peak=impact,
        lost_contact_time=lost_time,released=t['released']|event,
        qualified_release=qualified,premature=premature,
        release_height=jp.where(event,anchor_h,t['release_height']),
        release_tcp_height=jp.where(event,anchor_tcp,t['release_tcp_height']),
        detach_candidate=candidate,detach_height=anchor_h,detach_tcp=anchor_tcp,
        release_time=jp.where(t['released']|event,t['release_time']+dt,0.),
        target_time=jp.where(target,t['target_time']+dt,0.),
        recontact_seen=recontact,post_touch=touch,recontacts=t['recontacts']+(touch&~t['post_touch']),
        recontact_seconds=t['recontact_seconds']+touch*dt,
        clear_time=jp.where(clear,t.get('clear_time',jp.array(0.))+dt,0.))
    if c.release_balanced_rewards:
        from .balanced_release import update_credits
        new=update_credits(t,new,after,c)
    return new


def release_reward(t,f,p,c):
    if c.release_balanced_rewards:
        from .balanced_release import reward
        return reward(t,f,p,c)
    reach=1.-jp.tanh(c.reach_scale*p['tcp_block'])
    hold=t['carried']&~t['released']&(f['left']|f['right'])
    error=jp.maximum(c.release_tcp_min-f['tcp_height'],jp.maximum(f['tcp_height']-c.release_tcp_max,0.))
    align=jp.exp(-(f['xy']/c.gentle_alignment_scale)**2)
    proximity=(1.-jp.tanh(error/c.gentle_lower_scale))*align
    dwell=jp.minimum(t['target_time']/c.place_support_seconds,1.)
    holding=reach+(2.-jp.tanh(c.reach_scale*f['xy']))+3.*proximity+dwell
    quality=(t['qualified_release']&~t['premature']&~t['recontact_seen']).astype(jp.float32)
    quality*=1.-jp.clip((t['impact_peak']-c.release_impact_limit)/.5,0.,1.)
    basic=1.5*p['placed']+2.*p['inside']
    flight=t['released']&(t['release_time']<=c.release_flight_seconds)
    valid_progress=p['placed']|flight
    released=jp.where(quality>0.,valid_progress*(3.5+3.5*quality),basic)
    retreat=jp.clip(f['clearance']/c.gentle_clearance,0.,1.)*quality*p['placed']
    retreat*=~f['robot_cube']&~f['robot_bowl']
    if c.release_require_landing_for_retreat:
        retreat*=t['supported']
    score=jp.where(hold&~t['premature'],holding,released+retreat)
    active=t['active']&(hold|valid_progress)
    return score,active,quality,retreat


def release_step_reward(old,new,f,p,c,original,action_penalty,rate_penalty,dt):
    """Production reward composition, also used by the numerical audit."""
    score,active,quality,retreat=release_reward(new,f,p,c)
    score=jp.where(active|p['placed'],score,original)
    impact=c.gentle_impact_weight*(jp.minimum(new['impact_peak']/c.release_impact_limit,1.)-
        jp.minimum(old['impact_peak']/c.release_impact_limit,1.))
    touch=c.gentle_recontact_weight*(new['recontact_seconds']-old['recontact_seconds'])/dt
    from .balanced_release import bonuses
    credits=bonuses(old,new)
    bonus=sum(credits.values()) if c.release_balanced_rewards else jp.array(0.)
    reward=score+bonus-action_penalty-rate_penalty-impact-touch
    return reward,score,quality,retreat,impact,touch,credits


class ControlledRelease(GentlePlacement):
    def __init__(self,config_overrides=None):
        super().__init__(config_overrides,task_defaults=DEFAULTS)
        c=self._config
        if not 0<c.release_tcp_min<c.release_tcp_max:
            raise ValueError('positive ordered TCP release height band required')

    def _place_features(self,data):
        f=super()._place_features(data)
        rb=data.xmat[self._bowl_id].reshape(3,3)
        local=rb.T@(data.site_xpos[self._tcp_id]-data.xpos[self._bowl_id])
        f.update(tcp_rel=local,tcp_height=local[2]-self._config.bowl_floor,
            grip_reference=data.ctrl[self._gripper_ctrl_id],
            opening_intent=data.ctrl[self._gripper_ctrl_id]>(f['width']/2.+.0005))
        return f

    def _arm_motion_allowed(self,data,info):
        width=data.qpos[self._finger_qpos_ids[0]]-data.qpos[self._finger_qpos_ids[1]]
        opening=(info['coord_phase']==GRIP)&(info['actuated_goal']>0.)
        separated=width>=info['grip_start_width']+self._config.release_width_margin
        return (info['coord_phase']==MOVE)|(opening&separated)

    def _reset_controller(self,state):
        state=super()._reset_controller(state)
        memory=release_memory(self._place_features(state.data),self._config);memory['clear_time']=jp.array(0.)
        return state.replace(info=dict(state.info,placement=memory),metrics=dict(state.metrics,
            controlled_success=jp.array(0.),release_tcp_height=jp.array(0.),
            **{name:jp.array(0.) for name in ['release_bonus_progress','release_bonus_dwell',
                'release_bonus_release','release_bonus_landing','release_bonus_withdrawal',
                'release_target_count','release_progress_credit']}))

    def _task_substep(self,previous,current,action,task):
        memory,trace,index=task;f=self._place_features(current)
        memory=release_update(memory,self._place_features(previous),f,self.sim_dt,self._config)
        trace=trace.at[index].set(jp.array([f['height'],f['downward_speed'],f['support'],
            f['left'],f['right'],memory['qualified_release'],memory['premature'],
            f['robot_cube'],f['robot_bowl'],f['clearance']]))
        return memory,trace,index+1

    def step(self,state,action):
        # Call the unchanged smooth controller/base task, then replace only
        # placing scores. The earlier support-only experiment stays available.
        from .smooth_grip import SmoothGripPickPlace
        updated=SmoothGripPickPlace.step(self,state,action)
        c=self._config;old=state.info['placement'];new=updated.info['placement']
        f=self._place_features(updated.data);p=self._predicates(updated.data)
        original=updated.reward+updated.metrics['action_penalty']+updated.metrics['action_rate_penalty']
        reward,score,quality,retreat,impact,touch,credits=release_step_reward(old,new,f,p,c,
            original,updated.metrics['action_penalty'],updated.metrics['action_rate_penalty'],self.dt)
        invalid=(updated.metrics['nonfinite']>0)|(updated.metrics['capacity_overflow']>0)
        success=p['inside']&(new['clear_time']>=c.gentle_clear_seconds-1e-6)&(quality>=1.-1e-6)&~invalid&~updated.info['controller_failed']
        if c.release_balanced_rewards:
            success=new['supported']&(new['clear_time']>=c.gentle_clear_seconds-1e-6)&(quality>=1.-1e-6)&~invalid&~updated.info['controller_failed']
        metrics=dict(updated.metrics,gentle_success=success.astype(jp.float32),controlled_success=success.astype(jp.float32),
            gentle_quality=quality,gentle_retreat=retreat,gentle_active=new['active'].astype(jp.float32),
            gentle_supported=new['supported'].astype(jp.float32),gentle_released=new['released'].astype(jp.float32),
            gentle_premature=new['premature'].astype(jp.float32),gentle_recontacts=new['recontacts']-old['recontacts'],
            gentle_contact_seconds=new['recontact_seconds']-old['recontact_seconds'],gentle_impact_peak=new['impact_peak'],
            gentle_release_height=new['release_height'],release_tcp_height=new['release_tcp_height'],
            gentle_near_reset=updated.info['near_reset'].astype(jp.float32),gentle_reward=score,
            gentle_impact_cost=impact,gentle_recontact_cost=touch,
            release_target_count=new['target_count'],release_progress_credit=new['progress_credit'],
            **{'release_bonus_'+k:v for k,v in credits.items()})
        return updated.replace(reward=jp.where(invalid,-1.,reward),metrics=metrics)
