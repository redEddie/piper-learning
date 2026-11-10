"""SHOW-103: supported placement and withdrawal with SHOW100 smooth control.

Contact labels are reward/evaluation information, never actor inputs. The source
reward/reset/control implementations are retained as separate registered tasks.
"""
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx
from .smooth_grip import SmoothGripPickPlace
from .pick_place import PiperPickPlace
from .placement_refinement import (
    PlacementControlledPickPlace, DEFAULTS as OLD_DEFAULTS,
    placement_memory, placement_update,
)

DEFAULTS = dict(OLD_DEFAULTS, place_episode_tracking=True,
    place_contact_activation=True, gentle_near_fraction=.5,
    gentle_near_min=.005, gentle_near_max=.020,
    gentle_lower_scale=.04, gentle_alignment_scale=.02,
    gentle_clearance=.08, gentle_clear_seconds=.2,
    gentle_recontact_weight=2., gentle_impact_weight=2.)


def gentle_memory(f):
    t = placement_memory(f, True)
    return dict(t, recontact_seen=jp.array(False), post_touch=jp.array(False),
                recontacts=jp.array(0.), recontact_seconds=jp.array(0.),
                clear_time=jp.array(0.))


def gentle_update(t, before, after, dt, c):
    new = placement_update(t, before, after, dt, c)
    post_release = t['released'] | (t['floor_seen'] & ~t['carried'])
    touch = post_release & (after['robot_cube'] | after['robot_bowl'])
    recontact = t['recontact_seen'] | touch
    safe = (new['qualified_release'] & ~new['premature'] & ~recontact &
            after['support'] & after['in_region'] &
            (after['speed'] <= c.place_stable_speed) &
            (after['clearance'] >= c.gentle_clearance))
    return dict(new, recontact_seen=recontact, post_touch=touch,
        recontacts=t['recontacts']+(touch & ~t['post_touch']).astype(jp.float32),
        recontact_seconds=t['recontact_seconds']+touch.astype(jp.float32)*dt,
        clear_time=jp.where(safe, t['clear_time']+dt, 0.))


def gentle_reward(t, f, p, c):
    """Dense score: holding/lowering <=7, supported release7, clear retreat<=8.

    Premature release remains disqualified for the entire episode. Current
    distance (rather than cumulative retreat events) prevents a return trip
    from minting more withdrawal reward.
    """
    reach = 1.-jp.tanh(c.reach_scale*p['tcp_block'])
    hold = t['carried'] & ~t['released'] & (f['left'] | f['right'])
    align = jp.exp(-(f['xy']/c.gentle_alignment_scale)**2)
    lower = (1.-jp.tanh(jp.maximum(f['height'], 0.)/c.gentle_lower_scale))*align
    support = jp.minimum(t['support_time']/c.place_support_seconds, 1.)
    holding = reach + (2.-jp.tanh(c.reach_scale*f['xy'])) + 3.*lower + support
    # A safe contact gets the full score. Hard contacts degrade it smoothly.
    quality = (t['qualified_release'] & ~t['premature']).astype(jp.float32)
    quality *= 1.-jp.clip((t['impact_peak']-c.place_impact_scale)/.2, 0., 1.)
    basic = 1.5*p['placed'].astype(jp.float32)+2.*p['inside'].astype(jp.float32)
    released = jp.where(quality > 0.,
        p['placed'].astype(jp.float32)*(3.5+3.5*quality), basic)
    retreat = jp.clip(f['clearance']/c.gentle_clearance, 0., 1.)
    retreat *= quality*p['placed'].astype(jp.float32)*(~f['robot_cube'] & ~f['robot_bowl'])
    placing = jp.where(hold & ~t['premature'], holding, released+retreat)
    active = t['active'] & (hold | p['placed'])
    return placing, active, quality, retreat


class GentlePlacement(SmoothGripPickPlace):
    def __init__(self, config_overrides=None, task_defaults=None):
        defaults=dict(DEFAULTS);defaults.update(task_defaults or {})
        super().__init__(config_overrides, task_defaults=defaults)
        c = self._config
        if not c.bowl_free or not 0 <= c.gentle_near_fraction <= 1:
            raise ValueError('free bowl and near fraction in[0,1] required')
        if not 0 < c.gentle_near_min <= c.gentle_near_max:
            raise ValueError('positive ordered near heights required')
        self._sensor_adr = {n:int(self._mj_model.sensor('place_'+n).adr[0])
                            for n in ['left', 'right', 'support']} if c.impl != 'jax' else {}
        self._cube_adrs = [int(self._mj_model.sensor('gentle_cube_'+str(i)).adr[0])
                           for i in range(1,7)] if c.impl != 'jax' else []
        self._bowl_adrs = [int(self._mj_model.sensor('gentle_bowl_'+str(i)).adr[0])
                           for i in range(1,9)] if c.impl != 'jax' else []
        self._bowl_qvel_adr = int(self._mj_model.jnt_dofadr[self._mj_model.body_jntadr[self._bowl_id]])
        self._geom_body = jp.asarray(self._mj_model.geom_bodyid)
        self._finger_bodies = [self._mj_model.body(n).id for n in ['link7','link8']]

    def _configure_task_spec(self, spec):
        PlacementControlledPickPlace._configure_task_spec(self, spec)
        if self._config.impl == 'jax':
            return
        for obj, indices in [('cube',range(1,7)), ('bowl',range(1,9))]:
            for i in indices:
                sensor=spec.add_sensor(name=f'gentle_{obj}_{i}',type=mujoco.mjtSensor.mjSENS_CONTACT,
                    objtype=mujoco.mjtObj.mjOBJ_BODY,objname=f'link{i}',
                    reftype=mujoco.mjtObj.mjOBJ_BODY,refname='block' if obj=='cube' else 'bowl')
                sensor.intprm=[1,0,1]

    def _place_contacts(self, data):
        contact = PlacementControlledPickPlace._place_contacts(self, data)
        if self._config.impl != 'jax':
            contact.update(robot_cube=contact['left'] | contact['right'] |
                jp.any(data.sensordata[jp.array(self._cube_adrs)]>0),
                robot_bowl=jp.any(data.sensordata[jp.array(self._bowl_adrs)]>0))
        else:
            a,b=data._impl.contact.geom[:,0],data._impl.contact.geom[:,1]
            ba,bb=self._geom_body[a],self._geom_body[b]
            robot=jp.array([self._mj_model.body(f'link{i}').id for i in range(1,9)])
            ra=jp.any(ba[:,None]==robot,axis=1);rb=jp.any(bb[:,None]==robot,axis=1)
            active=(data._impl.contact.dist<=0)&(data._impl.contact.efc_address>=0)
            for key,body in [('robot_cube',self._block_id),('robot_bowl',self._bowl_id)]:
                contact[key]=jp.any(active&((ra&(bb==body))|(rb&(ba==body))))
        return contact

    def _place_features(self, data):
        f = PlacementControlledPickPlace._place_features(self, data)
        # Contact with both fingers establishes the carry witness; a one-sided
        # brush against a resting cube cannot create a successful carry/release.
        f['elevated'] &= f['left'] & f['right']
        tcp=data.site_xpos[self._tcp_id]
        f['clearance']=jp.minimum(jp.linalg.norm(tcp-data.xpos[self._block_id])-.05,
            jp.linalg.norm(tcp-data.xpos[self._bowl_id])-self._config.bowl_r_rim)
        return f

    def _build_reset_table(self):
        tables=super()._build_reset_table()
        from .gentle_reset import build_near_table
        tables['near']=build_near_table(self, tables['preinside'])
        return tables

    def reset(self, rng):
        state=PiperPickPlace.reset(self, rng)
        key=jax.random.fold_in(rng, 103)
        k1,k2,k3,k4=jax.random.split(key,4)
        near=(state.info['reset_stage']==3)&(jax.random.uniform(k1)<self._config.gentle_near_fraction)
        table=self._reset_table['near']
        q=table[jax.random.randint(k2,(),0,len(table))]
        qpos=state.data.qpos.at[self._qpos_ids].set(q)
        kin=mjx.kinematics(self._mjx_model,state.data.replace(qpos=qpos))
        tcp=kin.site_xpos[self._tcp_id]
        axis=kin.site_xmat[self._tcp_id][:,self._closing_axis]
        yaw=jp.arctan2(axis[1],axis[0])
        block=jp.concatenate([tcp,jp.array([jp.cos(yaw/2),0.,0.,jp.sin(yaw/2)])])
        qpos=jax.lax.dynamic_update_slice(qpos,block,(self._block_qpos_adr,))
        bowl=jp.concatenate([tcp[:2],jp.array([0.,1.,0.,0.,0.])])
        qpos=jax.lax.dynamic_update_slice(qpos,bowl,(self._bowl_qpos_adr,))
        qpos=qpos.at[self._finger_qpos_ids].set(jp.array([self._config.block_half,-self._config.block_half]))
        ctrl=state.data.ctrl.at[self._arm_ctrl_ids].set(q).at[self._gripper_ctrl_id].set(0.)
        data=mjx.forward(self._mjx_model,state.data.replace(
            qpos=jp.where(near,qpos,state.data.qpos),ctrl=jp.where(near,ctrl,state.data.ctrl)))
        predicates=self._predicates(data)
        percep=self._reset_perception(data,k3)
        obs,percep=self._get_obs(data,jp.zeros(7),predicates,percep,k4)
        obs=jax.tree.map(lambda x,y:jp.where(near,x,y),obs,state.obs)
        percep=jax.tree.map(lambda x,y:jp.where(near,x,y),percep,state.info['perception'])
        state=state.replace(data=data,obs=obs,info=dict(state.info,perception=percep,near_reset=near))
        return self._reset_controller(state)

    def _reset_controller(self, state):
        state=super()._reset_controller(state)
        memory=gentle_memory(self._place_features(state.data))
        names=['gentle_success','gentle_quality','gentle_retreat','gentle_active',
            'gentle_supported','gentle_released','gentle_premature','gentle_recontacts',
            'gentle_contact_seconds','gentle_impact_peak','gentle_release_height',
            'gentle_near_reset','gentle_reward','gentle_impact_cost','gentle_recontact_cost']
        metrics=dict(state.metrics,**{n:jp.array(0.) for n in names})
        return state.replace(info=dict(state.info,placement=memory,gentle_trace=jp.zeros((self.n_substeps,10))),metrics=metrics)

    def _start_task_scan(self,data,info):
        return info['placement'],jp.zeros((self.n_substeps,10)),jp.array(0,jp.int32)

    def _task_substep(self,previous,current,action,task):
        memory,trace,index=task
        f=self._place_features(current)
        memory=gentle_update(memory,self._place_features(previous),f,self.sim_dt,self._config)
        trace=trace.at[index].set(jp.array([f['height'],f['downward_speed'],f['support'],
            f['left'],f['right'],memory['qualified_release'],memory['premature'],
            f['robot_cube'],f['robot_bowl'],f['clearance']]))
        return memory,trace,index+1

    def _finish_task_scan(self,task):
        return dict(placement=task[0],gentle_trace=task[1])

    def step(self,state,action):
        updated=super().step(state,action)
        c=self._config;old=state.info['placement'];new=updated.info['placement']
        f=self._place_features(updated.data);p=self._predicates(updated.data)
        placing,active,quality,retreat=gentle_reward(new,f,p,c)
        original=updated.reward+updated.metrics['action_penalty']+updated.metrics['action_rate_penalty']
        score=jp.where(active,placing,original)
        # An unsupported release must not keep earning reach rewards after
        # insertion. A recontact cannot restore its lost quality eligibility.
        score=jp.where(p['placed'],placing,score)
        impact=c.gentle_impact_weight*(jp.minimum(new['impact_peak']/c.place_impact_scale,1.)-
            jp.minimum(old['impact_peak']/c.place_impact_scale,1.))
        recontact=c.gentle_recontact_weight*(new['recontact_seconds']-old['recontact_seconds'])/self.dt
        reward=score-updated.metrics['action_penalty']-updated.metrics['action_rate_penalty']-impact-recontact
        invalid=(updated.metrics['nonfinite']>0)|(updated.metrics['capacity_overflow']>0)
        reward=jp.where(invalid,-1.,reward)
        success=p['inside']&(new['clear_time']>=c.gentle_clear_seconds-1e-6)&(quality>=1.-1e-6)&~invalid&~updated.info['controller_failed']
        metrics=dict(updated.metrics,gentle_success=success.astype(jp.float32),gentle_quality=quality,
            gentle_retreat=retreat,gentle_active=new['active'].astype(jp.float32),gentle_supported=new['supported'].astype(jp.float32),
            gentle_released=new['released'].astype(jp.float32),gentle_premature=new['premature'].astype(jp.float32),
            gentle_recontacts=new['recontacts']-old['recontacts'],gentle_contact_seconds=new['recontact_seconds']-old['recontact_seconds'],
            gentle_impact_peak=new['impact_peak'],gentle_release_height=new['release_height'],
            gentle_near_reset=updated.info['near_reset'].astype(jp.float32),gentle_reward=score,
            gentle_impact_cost=impact,gentle_recontact_cost=recontact)
        return updated.replace(reward=reward,metrics=metrics)
