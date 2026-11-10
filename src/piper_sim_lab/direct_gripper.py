"""Direct latched gripper position targets, independent of the six-axis QP.

No MOVE/BRAKE/GRIP phases, grip slew filter, interlock, stroke completion,
busy-request rejection, or grip timeout. MuJoCo's existing actuator dynamics
and force limits remain the simulated hardware. Observation/action delay DR
is retained; it is not a gripper coordinator.
"""
import jax
import jax.numpy as jp
from mujoco import mjx

from .quadratic_release import QuadraticRelease
from .control_lab.smooth_plan import checked_plan
from .control_lab.hierarchy_env import fk_floor
from .control_lab.controllers import floor_distance


def direct_goal(action,old_goal,maximum,threshold):
    return jp.where(action<=-threshold,0.,jp.where(action>=threshold,maximum,old_goal))


class DirectGripperRelease(QuadraticRelease):
    independent_gripper=True

    def __init__(self,config_overrides=None):
        options=dict(config_overrides or {})
        if options.pop('independent_gripper',True) is not True:
            raise ValueError('direct_gripper_release requires independent_gripper=True')
        super().__init__(options)
        with self._config.unlocked():
            self._config.independent_gripper=True


def reset_arm(env,state):
    z=jp.array(0.,jp.float32)
    # The superclass has initialized only the arm tracking state and gripper
    # goal. In particular no coordinator state or phase observation is created.
    return state.replace(info=dict(state.info,
        stop_backup=jp.tile(state.info['post_state'],(env.stop_steps,1)),
        arm_trace=jp.zeros((env.n_substeps,9))),
        metrics=dict(state.metrics,reference_jerk=z,arm_reference_speed=z,
            arm_reference_acceleration=z,arm_rejected=z,
            # Legacy monitoring aliases refer solely to arm reference faults.
            coord_reference_fault=z,coord_reason=z,coord_timeout=z,
            gripper_direct_error=z))


def advance_arm(env,data,ctrl,action,hold,info):
    s=env.settings;p=env.tracking_settings
    hold=jp.array(0,jp.int32) if hold is None else hold
    jac=env._tcp_jacobian(data)
    twist=action[:6]*jp.array([s.linear_scale]*3+[s.angular_scale]*3)
    desired=jp.clip(jac.T@jp.linalg.solve(jac@jac.T+s.damping**2*jp.eye(6),twist),-s.velocity,s.velocity)
    candidate,ok=checked_plan(env.kin,data.qpos[env._qpos_ids],data.qvel[env._qvel_ids],
        info['post_state'],desired,info['upstream_velocity'],hold,env._lower,env._upper,
        s,p,env.n_substeps,env.stop_steps)
    accept=ok&~info['controller_failed']
    backup=jp.concatenate([info['stop_backup'],jp.tile(info['stop_backup'][-1],(env.n_substeps,1))])
    plan=jp.where(accept,candidate,backup)
    old_gripper_reference=data.ctrl[env._gripper_ctrl_id]
    goal=ctrl[env._gripper_ctrl_id]

    def step(carry,i):
        current,st,failed,reason,overflow,task=carry
        planned=jp.where(failed,st,plan[i])
        jerk=(planned[12:]-st[12:])/s.dt
        distance=floor_distance(fk_floor(env.kin,planned[:6],s.floor_normal_base)[0],s)
        code=(jp.any((planned[:6]<env._lower-s.tolerance)|(planned[:6]>env._upper+s.tolerance)).astype(jp.int32)
            +2*(distance < -s.tolerance).astype(jp.int32)
            +4*(jp.max(jp.abs(planned[6:12]))>s.velocity+1e-5).astype(jp.int32)
            +8*(jp.max(jp.abs(planned[12:]))>s.acceleration+1e-3).astype(jp.int32)
            +16*(jp.max(jp.abs(jerk))>s.jerk+.2).astype(jp.int32)
            +32*(~jp.all(jp.isfinite(planned))).astype(jp.int32))
        failed=failed|(code!=0);reason=jp.where(reason==0,code,reason)
        st=jp.where(failed,st,planned)
        # Apply the requested goal directly after the common communication
        # delay draw. There is no arm-dependent gripper gate or ramp.
        gripper_reference=jp.where(i<hold,old_gripper_reference,goal)
        applied=ctrl.at[env._arm_ctrl_ids].set(st[:6]).at[env._gripper_ctrl_id].set(gripper_reference)
        previous=current
        current=mjx.step(env._mjx_model,current.replace(ctrl=applied))
        task=env._task_substep(previous,current,action,task)
        gap=jp.max(jp.abs(st[:6]-current.qpos[env._qpos_ids]))
        cap=gap>s.lag+p.extra+s.tolerance
        code=code+64*cap.astype(jp.int32)
        failed=failed|cap;reason=jp.where(reason==0,code,reason)
        tracking=jp.concatenate([st[:6],st[6:12],current.qpos[env._qpos_ids],
            jp.array([gap,jp.maximum(gap-s.lag,0.),distance,jp.max(jp.abs(st[12:])),code.astype(st.dtype),0.])])
        width=current.qpos[env._finger_qpos_ids[0]]-current.qpos[env._finger_qpos_ids[1]]
        detail=jp.array([jp.max(jp.abs(st[6:12])),jp.max(jp.abs(st[12:])),jp.max(jp.abs(jerk)),
            accept,~ok,code.astype(st.dtype),gripper_reference,goal,width],dtype=st.dtype)
        return (current,st,failed,reason,overflow|env._capacity_overflow(current),task),(tracking,detail)

    initial=(data,info['post_state'],info['controller_failed'],info['controller_failure_reason'],
        env._capacity_overflow(data),env._start_task_scan(data,info))
    final,(tracking,detail)=jax.lax.scan(step,initial,jp.arange(env.n_substeps))
    current,st,failed,reason,overflow,task=final
    return current,overflow,dict(post_state=st,upstream_goal=st[:6],
        upstream_velocity=jp.where(~info['controller_failed'],desired,0.),
        stop_backup=plan[env.n_substeps:],controller_failed=failed,
        controller_failure_reason=reason,controller_intervention=(~ok).astype(jp.float32),
        tracking_trace=tracking,arm_trace=detail,**env._finish_task_scan(task))


def arm_metrics(env,updated):
    tr=updated.info['arm_trace'];reason=updated.info['controller_failure_reason']
    return dict(reference_jerk=jp.max(tr[:,2]),arm_reference_speed=jp.max(tr[:,0]),
        arm_reference_acceleration=jp.max(tr[:,1]),arm_rejected=jp.max(tr[:,4]),
        coord_reference_fault=((reason&63)!=0).astype(jp.float32),
        coord_reason=reason.astype(jp.float32),coord_timeout=jp.array(0.),
        gripper_direct_error=jp.abs(updated.data.ctrl[env._gripper_ctrl_id]-updated.info['gripper_goal']))
