"""Placing-only refinement of SHOW-100, using SHOW-87 v2 measurements.

No action guards. The 49/66 observations, reset distribution, reach/lift/carry
rewards and basic success predicate stay unchanged. Contact readings are reward
labels only, never actor inputs. Pure memory updates run every 5 ms.
"""
import jax.numpy as jp
import mujoco
import numpy as np
from piper_sim_lab.ternary_control import TernaryControlledPickPlace

DEFAULTS=dict(place_progress_weight=20.,place_impact_weight=10.,place_opening_weight=1.,
    place_impact_scale=.1,place_support_seconds=.1,place_stable_speed=.02,
    place_height_tolerance=.005,place_release_seconds=.02,place_opening_margin=.01,
    place_base_reward=0.,place_hold_bridge=False,place_contact_activation=False,
    place_episode_tracking=False,place_carry_clearance=.005)


def placement_memory(f,episode_tracking=False):
    z=jp.array(0.,jp.float32);no=jp.array(False)
    carried=(f['left']|f['right'])&f['elevated']&~f['support'] if episode_tracking else no
    return dict(active=f['above'],best_distance=f['distance'],progress_credit=z,
        support_time=z,supported=no,had_contact=f['left']|f['right'],
        carried=carried,lost_contact_time=z,released=no,qualified_release=no,
        premature=f['support']&~carried if episode_tracking else no,
        impact_peak=z,opening_best=z,floor_seen=f['support'] if episode_tracking else no,release_height=z)


def placement_update(t,before,after,dt,c):
    """Episode ledgers prevent progress/opening/impact reward duplication."""
    active=t['active']|before['above']|after['above']
    episode=getattr(c,'place_episode_tracking',False)
    # Phase entry gates progress only. Release/support/impact events must not
    # disappear just because the policy did not pass the old pre_inside test.
    monitor=jp.array(True) if episode else active
    carried=t['carried']
    if episode:
        before_carry=(before['left']|before['right'])&before['elevated']&~before['support']
        after_carry=(after['left']|after['right'])&after['elevated']&~after['support']
        carried=carried|(~t['floor_seen']&(before_carry|after_carry))
    # Starting a placing phase must not mint progress from earlier transport.
    best=jp.where(t['active'],jp.minimum(t['best_distance'],after['distance']),after['distance'])
    gripping=after['left']|after['right']
    support_now=after['support']&after['in_region']&(after['speed']<=c.place_stable_speed+1e-6)
    support_time=jp.where(monitor&support_now,t['support_time']+dt,0.)
    supported=support_time>=c.place_support_seconds-1e-6
    had_contact=t['had_contact']|(monitor&gripping)
    release_witness=carried if episode else had_contact
    lost_time=jp.where(monitor&release_witness&~gripping,t['lost_contact_time']+dt,0.)
    release=monitor&release_witness&~t['released']&(lost_time>=c.place_release_seconds-1e-6)
    unheld_landing=(after['support']&~before['support']&~carried) if episode else jp.array(False)
    premature=t['premature']|(release&~t['supported'])|unheld_landing
    qualified=(t['qualified_release']|(release&t['supported']&~t['premature']))&~premature
    eligible=t['active'] & ~(premature if episode else t['premature']) & (gripping|t['supported']) & ~t['released']
    progress=t['progress_credit']+jp.where(eligible,jp.maximum(t['best_distance']-best,0.),0.)
    # The velocity BEFORE first contact is used, not a slowed post-impact sample.
    new_impact=jp.where(monitor&after['support']&~before['support'],before['downward_speed'],0.)
    impact=jp.maximum(t['impact_peak'],new_impact)
    opening=jp.clip((after['width']-after['span'])/c.place_opening_margin,0.,1.)
    useful=active&t['supported']&support_now&~premature&(carried if episode else jp.array(True))
    opening_best=jp.where(useful,jp.maximum(t['opening_best'],opening),t['opening_best'])
    return dict(active=active,best_distance=best,progress_credit=progress,
        support_time=support_time,supported=supported,had_contact=had_contact,carried=carried,
        lost_contact_time=lost_time,released=t['released']|release,qualified_release=qualified,
        premature=premature,impact_peak=impact,opening_best=opening_best,
        floor_seen=t['floor_seen']|(monitor&after['support']),
        release_height=jp.where(release,before['height'],t['release_height']))


def placement_terms(old,new,placed,c,inside=None):
    quality=(new['qualified_release']&~new['premature']).astype(jp.float32)/(1+(new['impact_peak']/c.place_impact_scale)**2)
    basic=placed if inside is None else inside
    floor=getattr(c,'place_base_reward',0.)
    return dict(progress=c.place_progress_weight*(new['progress_credit']-old['progress_credit']),
        impact=c.place_impact_weight*(new['impact_peak']-old['impact_peak']),
        opening=c.place_opening_weight*(new['opening_best']-old['opening_best']),
        quality=quality,placement_reward=floor*basic.astype(jp.float32)+(c.inside_reward-floor)*placed.astype(jp.float32)*quality)


class PlacementControlledPickPlace(TernaryControlledPickPlace):
    def __init__(self,config_overrides=None):
        super().__init__(config_overrides=config_overrides,task_defaults=DEFAULTS)
        for name in DEFAULTS:
            if name in ('place_base_reward','place_hold_bridge','place_contact_activation','place_episode_tracking'):continue
            if not np.isfinite(self._config[name]) or self._config[name]<=0:
                raise ValueError(f'{name} must be positive finite')
        if not 0<=self._config.place_base_reward<=self._config.inside_reward:
            raise ValueError('place_base_reward must be in [0, inside_reward]')
        self._sensor_adr={name:int(np.asarray(self._mj_model.sensor('place_'+name).adr).item())
                          for name in ('left','right','support')} if self._config.impl!='jax' else {}
        if not self._config.bowl_free:
            raise ValueError('placing refinement requires the original free bowl')
        self._bowl_qvel_adr=int(self._mj_model.jnt_dofadr[self._mj_model.body_jntadr[self._bowl_id]])
        self._geom_body=jp.asarray(self._mj_model.geom_bodyid)
        self._finger_bodies=[self._mj_model.body(n).id for n in ('link7','link8')]

    def _configure_task_spec(self,spec):
        spec.body('bowl').geoms[0].name='place_interior_floor'
        if self._config.impl=='jax':return
        for name,obj,ref,is_geom in [('left','link7','block',False),('right','link8','block',False),
                                    ('support','block','place_interior_floor',True)]:
            s=spec.add_sensor(name='place_'+name,type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=mujoco.mjtObj.mjOBJ_BODY,objname=obj,
                reftype=mujoco.mjtObj.mjOBJ_GEOM if is_geom else mujoco.mjtObj.mjOBJ_BODY,refname=ref)
            s.intprm=[1,0,1]

    def _place_contacts(self,data):
        if self._config.impl!='jax':
            return {name:data.sensordata[adr]>0 for name,adr in self._sensor_adr.items()}
        contact=data._impl.contact
        ga,gb=contact.geom[:,0],contact.geom[:,1]
        ba,bb=self._geom_body[ga],self._geom_body[gb]
        active=(contact.dist<=0)&(contact.efc_address>=0)
        def pair(a,b):return jp.any(active&(((ba==a)&(bb==b))|((ba==b)&(bb==a))))
        floor=int(self._mj_model.body_geomadr[self._bowl_id])
        return dict(left=pair(self._block_id,self._finger_bodies[0]),
            right=pair(self._block_id,self._finger_bodies[1]),
            support=jp.any(active&(((ga==floor)&(bb==self._block_id))|((gb==floor)&(ba==self._block_id)))))

    def _place_features(self,data):
        c=self._config;obj=data.xpos[self._block_id];bowl=data.xpos[self._bowl_id]
        rb=data.xmat[self._bowl_id].reshape(3,3);ro=data.xmat[self._block_id].reshape(3,3);rt=data.site_xmat[self._tcp_id].reshape(3,3)
        local=rb.T@(obj-bowl)
        height=local[2]-c.block_half*jp.sum(jp.abs(rb[:,2]@ro))-c.bowl_floor
        xy=jp.linalg.norm(local[:2]);distance=jp.hypot(xy,jp.maximum(height,0.))
        relative_velocity=data.qvel[self._block_qvel_adr:self._block_qvel_adr+3]-data.qvel[self._bowl_qvel_adr:self._bowl_qvel_adr+3]
        contact=self._place_contacts(data)
        above=self._predicates(data)['pre_inside']
        if c.place_contact_activation:
            above=above|((contact['left']|contact['right'])&(xy<c.bowl_r_rim-c.block_half))
        return dict(contact,above=above,distance=distance,
            elevated=(obj[2]-c.block_half*jp.sum(jp.abs(ro[2,:])))>c.place_carry_clearance,
            height=height,xy=xy,speed=jp.linalg.norm(relative_velocity),
            downward_speed=jp.maximum(-(rb[:,2]@relative_velocity),0.),
            width=data.qpos[self._finger_qpos_ids[0]]-data.qpos[self._finger_qpos_ids[1]],
            span=2*c.block_half*jp.sum(jp.abs(rt[:,self._closing_axis]@ro)),
            in_region=(xy<=c.bowl_r_bottom-c.block_half)&(height>=-.001)&(height<=c.place_height_tolerance))

    def _reset_controller(self,state):
        state=super()._reset_controller(state)
        memory=placement_memory(self._place_features(state.data),self._config.place_episode_tracking)
        zero=jp.array(0.,jp.float32)
        metrics={k:zero for k in ['reward_place_progress','penalty_place_impact','reward_place_opening','reward_place_hold_bridge',
            'place_quality','place_supported','place_premature_release','place_impact_peak','place_release_height','place_floor_seen','gentle_success']}
        return state.replace(info=dict(state.info,placement=memory),metrics=dict(state.metrics,**metrics))

    def _start_task_scan(self,data,info):return info['placement']
    def _task_substep(self,previous,current,action,task):
        return placement_update(task,self._place_features(previous),self._place_features(current),self.sim_dt,self._config)
    def _finish_task_scan(self,task):return dict(placement=task)

    def step(self,state,action):
        updated=super().step(state,action)
        p=self._predicates(updated.data);old=state.info['placement'];new=updated.info['placement']
        terms=placement_terms(old,new,p['placed'],self._config,p['inside'])
        bridge=jp.array(0.,jp.float32)
        if self._config.place_hold_bridge:
            contact=self._place_contacts(updated.data)
            held=new['active']&~new['released']&(contact['left']|contact['right'])
            if self._config.place_episode_tracking:
                held=held&new['carried']&~new['premature']
            bridge=held.astype(jp.float32)*(1.-p['grasp'].astype(jp.float32))*(2.-jp.tanh(self._config.reach_scale*p['block_bowl_xy']))
        original_placed=self._config.inside_reward*p['placed'].astype(jp.float32)
        reward=updated.reward-original_placed+terms['placement_reward']+terms['progress']-terms['impact']+terms['opening']+bridge
        invalid=(updated.metrics['nonfinite']>0)|(updated.metrics['capacity_overflow']>0)
        reward=jp.where(invalid,-1.,reward)
        gentle=p['inside']&new['qualified_release']&new['supported']&(new['impact_peak']<=self._config.place_impact_scale)
        return updated.replace(reward=reward,metrics=dict(updated.metrics,
            reward_inside=terms['placement_reward']+self._config.release_bonus*p['inside'].astype(jp.float32),
            reward_place_progress=terms['progress'],reward_place_hold_bridge=bridge,penalty_place_impact=terms['impact'],
            reward_place_opening=terms['opening'],place_quality=terms['quality'],
            place_supported=new['supported'].astype(jp.float32),place_premature_release=new['premature'].astype(jp.float32),
            place_impact_peak=new['impact_peak'],place_release_height=new['release_height'],
            place_floor_seen=new['floor_seen'].astype(jp.float32),gentle_success=(gentle&~invalid).astype(jp.float32)))
