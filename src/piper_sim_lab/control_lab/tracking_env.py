"""200Hz execution of the recovering QP; task code remains inherited."""
import jax
import jax.numpy as jp
from mujoco import mjx
from .hierarchy_env import fk_floor
from .hierarchy import floor_minimum
from .controllers import floor_distance
from .tracking_qp import tracking_problem,project_tracking


def advance_tracking(env,data,ctrl,action,hold,info):
    s,p=env.settings,env.tracking_settings
    hold=jp.array(0,jp.int32) if hold is None else hold
    j=env._tcp_jacobian(data)
    twist=action[:6]*jp.array([s.linear_scale]*3+[s.angular_scale]*3)
    desired=jp.clip(j.T@jp.linalg.solve(j@j.T+s.damping**2*jp.eye(6),twist),-s.velocity,s.velocity)
    previous_ctrl=data.ctrl
    def substep(carry,i):
        current,st,active,timer,failed,reason,overflow,task=carry
        ref,v,previous_acceleration=jp.split(st,3)
        measured=current.qpos[env._qpos_ids]
        qdot=current.qvel[env._qvel_ids]
        target=jp.where(i<hold,info['upstream_velocity'],desired)
        preferred,lo,hi,slo,shi,next_active,next_timer=tracking_problem(
            measured,qdot,ref,v,target,env._lower,env._upper,s,p,active,timer,xp=jp,
            previous_acceleration=previous_acceleration)
        z,normal=fk_floor(env.kin,ref,s.floor_normal_base)
        minimum=floor_minimum(z,s,xp=jp)
        vn,ok=project_tracking(preferred,lo,hi,slo,shi,p.weight,normal,minimum)
        candidate=ref+s.dt*vn
        next_z,_=fk_floor(env.kin,candidate,s.floor_normal_base)
        distance=floor_distance(next_z,s)
        timeout=jp.any(next_timer>p.timeout)
        code=(jp.any(lo>hi).astype(jp.int32)
              +2*(jp.dot(normal,jp.where(normal>=0,hi,lo))<minimum-1e-6).astype(jp.int32)
              +4*(distance < -s.tolerance).astype(jp.int32)
              +64*(~jp.all(jp.isfinite(candidate))).astype(jp.int32)
              +256*timeout.astype(jp.int32))
        failed=failed|~ok|(code!=0)
        reason=jp.where(reason==0,code,reason)
        an=(vn-v)/s.dt
        st=jp.where(failed,st,jp.concatenate([candidate,vn,an]))
        active=jp.where(failed,active,next_active)
        timer=jp.where(failed,timer,next_timer)
        applied=jp.where(i<hold,previous_ctrl,ctrl).at[env._arm_ctrl_ids].set(st[:6])
        previous=current
        current=mjx.step(env._mjx_model,current.replace(ctrl=applied))
        task=env._task_substep(previous,current,action,task)
        gap=jp.max(jp.abs(st[:6]-current.qpos[env._qpos_ids]))
        # A one-step plant prediction is not an outer-cap guarantee. Detect
        # unexpected actual tracking excursions and stop further references.
        actual_cap=gap>s.lag+p.extra+s.tolerance
        failed=failed|actual_cap
        reason=jp.where(reason==0,512*actual_cap.astype(jp.int32),reason)
        row=jp.concatenate([st[:6],st[6:12],current.qpos[env._qpos_ids],
            jp.array([gap,jp.maximum(gap-s.lag,0.),distance,jp.max(jp.abs(an)),
                      code.astype(ref.dtype),jp.sum(active).astype(ref.dtype)])])
        return (current,st,active,timer,failed,reason,overflow|env._capacity_overflow(current),task),row
    (data,st,active,timer,failed,reason,overflow,task),trace=jax.lax.scan(substep,
        (data,info['post_state'],info['tracking_recovering'],info['tracking_timer'],
         info['controller_failed'],info['controller_failure_reason'],env._capacity_overflow(data),env._start_task_scan(data,info)),
        jp.arange(env.n_substeps))
    data=data.replace(ctrl=ctrl.at[env._arm_ctrl_ids].set(st[:6]))
    return data,overflow,dict(post_state=st,upstream_goal=st[:6],upstream_velocity=desired,
        controller_failed=failed,controller_failure_reason=reason,
        controller_intervention=jp.mean(trace[:,19]>0),
        tracking_recovering=active,tracking_timer=timer,tracking_trace=trace,**env._finish_task_scan(task))
