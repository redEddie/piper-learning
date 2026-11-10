"""Batched hierarchical controllers with the SHOW-76 plant, rewards and DR.

Only Ruckig crosses to CPU, once per 10 Hz packet (20 trajectory samples).
The joint QP has an exact scalar-dual GPU solution, checked against OSQP.
Controller state is explicit in info and resets with each environment. It is
not added to the 48D actor observation: partial observability is intentional.
"""
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jp
from mujoco import mjx
from piper_sim_lab.pick_place import PiperPickPlace, default_config
from piper_policy_runtime.kinematics import PiperKinematics
from .controllers import Settings, floor_distance
from .hierarchy import bounds, ruckig_packet, floor_minimum


def project_jax(target, lo, hi, normal, minimum):
    maximum = jp.dot(normal,jp.where(normal>=0,hi,lo))
    feasible = jp.all(lo<=hi) & (maximum>=minimum-1e-6)
    def value(multiplier):
        return jp.clip(target+multiplier*normal,lo,hi)
    upper = jax.lax.fori_loop(0,40,lambda _,r: jp.where(jp.dot(normal,value(r))<minimum,2*r,r),jp.array(1.,target.dtype))
    def bisect(_,pair):
        l,r = pair
        mid = (l+r)*.5
        small = jp.dot(normal,value(mid))<minimum
        return jp.where(small,mid,l),jp.where(small,r,mid)
    _, multiplier = jax.lax.fori_loop(0,32,bisect,(jp.array(0.,target.dtype),upper))
    clipped = value(0.)
    v = jp.where(jp.dot(normal,clipped)>=minimum,clipped,value(multiplier))
    return v, feasible


from piper_runtime.control.jax_kinematics import fk_floor


def batch_ruckig(start, old, target, hold, s):
    lead=np.broadcast_shapes(start.shape[:-1],old.shape[:-1],target.shape[:-1],hold.shape)
    starts=np.broadcast_to(start,lead+(18,)).reshape(-1,18)
    olds=np.broadcast_to(old,lead+(6,)).reshape(-1,6)
    targets=np.broadcast_to(target,lead+(6,)).reshape(-1,6)
    holds=np.broadcast_to(hold,lead).reshape(-1)
    n=round(s.policy_dt/s.dt)
    plans=[]; failures=[]
    for st,ov,tv,h in zip(starts,olds,targets,holds):
        try:
            plan=ruckig_packet(st.astype(float),ov.astype(float),tv.astype(float),h,s)
            failed=not np.isfinite(plan).all()
        except (ValueError,RuntimeError):
            failed=True
            plan=np.tile(np.r_[st[:6],np.zeros(12)],(n,1))
        plans.append(plan);failures.append(failed)
    return (np.asarray(plans,dtype=start.dtype).reshape(lead+(n,18)),
            np.asarray(failures,dtype=bool).reshape(lead))


class HierarchicalPickPlace(PiperPickPlace):
    def __init__(self,config_overrides=None, task_defaults=None):
        config=default_config()
        config.update(dict(control_mode='dls',extra_delay_ticks=0,surrogate_tau=.04,
            control_acceleration=8.,control_jerk=100.,control_floor=0.))
        if task_defaults:
            config.update(task_defaults)
        super().__init__(config=config,config_overrides=config_overrides)
        c=self._config
        self.mode=c.control_mode
        if self.mode not in ('dls','delay','surrogate','qp','ruckig'):
            raise ValueError(self.mode)
        if not 0<=c.extra_delay_ticks<=60 or c.surrogate_tau<=0:
            raise ValueError('delay must be 0..60 ticks; tau positive')
        self.settings=Settings(dt=self.sim_dt,policy_dt=self.dt,linear_scale=c.max_lin_vel,
            angular_scale=c.max_ang_vel,damping=c.ik_damping,velocity=c.max_joint_vel,
            lag=c.max_target_lag,acceleration=c.control_acceleration,jerk=c.control_jerk,
            tcp_floor=c.control_floor,base_height=c.base_height).with_scene_frames(self.frames)
        self.kin=PiperKinematics(Path(self._xml_path).parent/'piper.xml')

    def reset(self,rng):
        state=super().reset(rng)
        return self._reset_controller(state)

    def _reset_controller(self,state):
        q=state.data.ctrl[self._arm_ctrl_ids]
        zero=jp.zeros(6,q.dtype)
        return state.replace(info=dict(state.info,post_state=jp.concatenate([q,zero,zero]),
            upstream_goal=q,upstream_velocity=zero,delay_queue=jp.tile(q,(61,1)),
            controller_failed=jp.array(False),controller_intervention=jp.array(0.,q.dtype),
            controller_failure_reason=jp.array(0,jp.int32)))

    def _start_task_scan(self, data, info):
        return ()

    def _task_substep(self, previous, current, action, task):
        return task

    def _finish_task_scan(self, task):
        return {}

    def _advance_control(self,data,ctrl,action,hold,info):
        if self.mode=='dls':
            data,overflow= self._step_physics(data,ctrl,hold)
            return data,overflow,{}
        s=self.settings
        hold=jp.array(0,jp.int32) if hold is None else hold
        j=self._tcp_jacobian(data)
        twist=action[:6]*jp.array([s.linear_scale]*3+[s.angular_scale]*3)
        desired=jp.clip(j.T@jp.linalg.solve(j@j.T+s.damping**2*jp.eye(6),twist),-s.velocity,s.velocity)
        q=data.qpos[self._qpos_ids]
        goal=jp.clip(jp.clip(info['upstream_goal']+s.policy_dt*desired,q-s.lag,q+s.lag),self._lower,self._upper)
        previous_ctrl=data.ctrl
        st=info['post_state']
        failed=info['controller_failed']
        if self.mode=='ruckig':
            plan,plan_failed=jax.pure_callback(lambda *a:batch_ruckig(*a,s),
                (jax.ShapeDtypeStruct((self.n_substeps,18),st.dtype),jax.ShapeDtypeStruct((),jp.bool_)),
                st,info['upstream_velocity'],desired,hold,vmap_method='broadcast_all')
            failed=failed|plan_failed
        else:
            plan=jp.zeros((self.n_substeps,18),st.dtype)

        def substep(carry,i):
            current,st,queue,failed,overflow,interventions,reason,task=carry
            code=jp.array(0,jp.int32)
            ref,v,a=jp.split(st,3)
            target=jp.where(i<hold,info['upstream_velocity'],desired)
            if self.mode=='delay':
                target_goal=jp.where(i<hold,info['upstream_goal'],goal)
                queue=jp.concatenate([target_goal[None],queue[:-1]],axis=0)
                candidate=queue[self._config.extra_delay_ticks]
                vn=(candidate-ref)/s.dt
                an=(vn-v)/s.dt
                ok=jp.array(True)
            elif self.mode=='ruckig':
                candidate,vn,an=jp.split(plan[i],3)
                measured=current.qpos[self._qpos_ids]
                z,_=fk_floor(self.kin,candidate,s.floor_normal_base)
                ok=(jp.all(candidate>=self._lower-s.tolerance)&jp.all(candidate<=self._upper+s.tolerance)
                    &jp.all(jp.abs(candidate-measured)<=s.lag+s.tolerance)&(floor_distance(z,s)>=-s.tolerance))
                code=(8*(jp.any(candidate<self._lower-s.tolerance)|jp.any(candidate>self._upper+s.tolerance)).astype(jp.int32)
                    +16*jp.any(jp.abs(candidate-measured)>s.lag+s.tolerance).astype(jp.int32)
                    +32*(floor_distance(z,s)<-s.tolerance).astype(jp.int32)
                    +128*plan_failed.astype(jp.int32))
            else:
                lo,hi=bounds(current.qpos[self._qpos_ids],ref,v,self._lower,self._upper,s,xp=jp)
                z,normal=fk_floor(self.kin,ref,s.floor_normal_base)
                minimum=floor_minimum(z,s,xp=jp)
                if self.mode=='qp':
                    preferred=(target+s.smooth_weight*v)/(1+s.smooth_weight)
                else:
                    desired_a=jp.clip((target-v)/self._config.surrogate_tau,-s.acceleration,s.acceleration)
                    limited_a=jp.clip(desired_a,a-s.jerk*s.dt,a+s.jerk*s.dt)
                    preferred=v+s.dt*limited_a
                vn,ok=project_jax(preferred,lo,hi,normal,minimum)
                candidate=ref+s.dt*vn
                next_z,_=fk_floor(self.kin,candidate,s.floor_normal_base)
                ok=ok&(floor_distance(next_z,s)>=-s.tolerance)
                code=(jp.any(lo>hi).astype(jp.int32)
                    +2*(jp.dot(normal,jp.where(normal>=0,hi,lo))<minimum-1e-6).astype(jp.int32)
                    +4*(floor_distance(next_z,s)<-s.tolerance).astype(jp.int32))
                an=(vn-v)/s.dt
                interventions+=jp.any(jp.abs(vn-preferred)>s.tolerance).astype(st.dtype)
            code+=64*(~jp.all(jp.isfinite(candidate))).astype(jp.int32)
            reason=jp.where(reason==0,code,reason)
            failed=failed|~ok|~jp.all(jp.isfinite(candidate))
            proposed=jp.concatenate([candidate,vn,an])
            st=jp.where(failed,st,proposed)
            applied=jp.where(i<hold,previous_ctrl,ctrl).at[self._arm_ctrl_ids].set(st[:6])
            previous=current
            current=mjx.step(self._mjx_model,current.replace(ctrl=applied))
            task=self._task_substep(previous,current,action,task)
            return (current,st,queue,failed,overflow|self._capacity_overflow(current),interventions,reason,task),None
        (data,st,queue,failed,overflow,interventions,reason,task),_=jax.lax.scan(substep,
            (data,st,info['delay_queue'],failed,self._capacity_overflow(data),jp.array(0.,st.dtype),info['controller_failure_reason'],self._start_task_scan(data,info)),jp.arange(self.n_substeps))
        # Gripper target bookkeeping matches the original action-delay path.
        data=data.replace(ctrl=ctrl.at[self._arm_ctrl_ids].set(st[:6]))
        return data,overflow,dict(post_state=st,upstream_goal=goal,upstream_velocity=desired,
            delay_queue=queue,controller_failed=failed,controller_intervention=interventions/self.n_substeps,
            controller_failure_reason=reason, **self._finish_task_scan(task))
