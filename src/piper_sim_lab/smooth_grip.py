"""SHOW-100 reward/resets with bounded reference plans and serialized gripping.

MOVE -> BRAKE -> GRIP -> MOVE. Raw policy actions remain unchanged; accepted
gripper goal and phase are observed. Requests during a busy phase do not restart
or reverse an operation. New arm plans include a checked complete-stop tail.
"""
import math
import jax
import jax.numpy as jp
from mujoco import mjx
from piper_sim_lab.ternary_control import TernaryControlledPickPlace,gripper_command
from piper_sim_lab.control_lab.smooth_plan import checked_plan
from piper_sim_lab.control_lab.hierarchy_env import fk_floor
from piper_sim_lab.control_lab.controllers import floor_distance

MOVE=0;BRAKE=1;GRIP=2
DEFAULTS=dict(max_joint_vel=.75,control_acceleration=4.,control_jerk=300.,max_lin_vel=.05,max_ang_vel=.5,
    smooth_stop_steps=64,grip_arm_stable_speed=.02,grip_arm_stable_seconds=.05,
    grip_stable_speed=.005,grip_stable_seconds=.1,grip_timeout=3.,brake_timeout=2.)


def accepted_goal(action,old_goal,phase,maximum,threshold):
    requested=jp.where(action<=-threshold,0.,jp.where(action>=threshold,maximum,old_goal))
    return jp.where(phase==MOVE,requested,old_goal)


def stroke_complete(window,reference,goal,start_width,elapsed,c):
    width=window[-1]
    stable=(jp.max(window)-jp.min(window))<=c.grip_stable_speed*c.grip_stable_seconds
    opened=(width>=start_width+.002)|(jp.abs(width-2*c.gripper_max)<.002)
    actual_end=jp.where(goal>.5*c.gripper_max,opened,width<2*c.gripper_max-.002)
    return stable&(elapsed>=c.grip_stable_seconds-1e-6)&(jp.abs(reference-goal)<1e-6)&actual_end


class SmoothGripPickPlace(TernaryControlledPickPlace):
    def __init__(self,config_overrides=None,task_defaults=None):
        defaults=dict(DEFAULTS)
        defaults.update(task_defaults or {})
        super().__init__(config_overrides=config_overrides,task_defaults=defaults)
        c=self._config
        for name in DEFAULTS:
            if not math.isfinite(c[name]) or c[name]<=0:raise ValueError(f'{name} must be positive finite')
        if int(c.smooth_stop_steps)!=c.smooth_stop_steps:raise ValueError('integer stop horizon required')
        self.stop_steps=int(c.smooth_stop_steps)
        self.width_samples=int(round(c.grip_stable_seconds/self.sim_dt))

    def _augment(self,state):
        if getattr(self,'independent_gripper',False):return state
        phase=(state.info['coord_phase']/2.).astype(jp.float32)[None]
        obs={name:jp.concatenate([v[:42],phase,v[42:]]) for name,v in state.obs.items()}
        return state.replace(obs=obs)

    def _reset_controller(self,state):
        state=super()._reset_controller(state)
        if getattr(self,'independent_gripper',False):
            from .direct_gripper import reset_arm
            return reset_arm(self,state)
        z=jp.array(0.,jp.float32);held=state.info['reset_stage']>=2
        info=dict(state.info,stop_backup=jp.tile(state.info['post_state'],(self.stop_steps,1)),
            coord_phase=jp.where(held,GRIP,MOVE).astype(jp.int32),coord_timer=z,coord_stable=z,
            arm_stable=z,actuated_goal=state.info['gripper_goal'],
            grip_reference=state.data.ctrl[self._gripper_ctrl_id],candidate_rejections=z,
            grip_start_width=state.data.qpos[self._finger_qpos_ids[0]]-state.data.qpos[self._finger_qpos_ids[1]],
            grip_width_window=jp.full((self.width_samples,),state.data.qpos[self._finger_qpos_ids[0]]-state.data.qpos[self._finger_qpos_ids[1]]),
            operations_completed=z,coord_trace=jp.zeros((self.n_substeps,13)))
        info['grip_motion_unlocked']=jp.array(False)
        metrics=dict(state.metrics,coord_busy=z,coord_grip=z,coord_completed=z,coord_rejected=z,
            coord_timeout=z,reference_jerk=z,grip_arm_reference_speed=z,coord_reason=z,coord_width=z,
            coord_accepted_goal=state.info['gripper_goal'],coord_phase_metric=z,coord_reference_fault=z,
            grip_arm_locked_reference_speed=z)
        return self._augment(state.replace(info=info,metrics=metrics))

    def _gripper_command(self,data,action,info):
        if getattr(self,'independent_gripper',False):
            from .direct_gripper import direct_goal
            goal=direct_goal(action[6],info['gripper_goal'],self._config.gripper_max,self._config.gripper_threshold)
            return goal,dict(gripper_goal=goal)
        goal=accepted_goal(action[6],info['gripper_goal'],info['coord_phase'],self._config.gripper_max,self._config.gripper_threshold)
        return info['grip_reference'],dict(gripper_goal=goal)

    def _arm_motion_allowed(self,data,info):
        """Subclasses may release an opening interlock; closing stays locked."""
        return info['coord_phase']==MOVE

    def _advance_control(self,data,ctrl,action,hold,info):
        if getattr(self,'independent_gripper',False):
            from .direct_gripper import advance_arm
            return advance_arm(self,data,ctrl,action,hold,info)
        c=self._config;s=self.settings;p=self.tracking_settings
        hold=jp.array(0,jp.int32) if hold is None else hold
        goal=accepted_goal(action[6],info['gripper_goal'],info['coord_phase'],c.gripper_max,c.gripper_threshold)
        request=(info['coord_phase']==MOVE)&(goal!=info['actuated_goal'])
        phase=jp.where(request,BRAKE,info['coord_phase'])
        timer=jp.where(request,0.,info['coord_timer'])
        jac=self._tcp_jacobian(data)
        twist=action[:6]*jp.array([s.linear_scale]*3+[s.angular_scale]*3)
        desired=jp.clip(jac.T@jp.linalg.solve(jac@jac.T+s.damping**2*jp.eye(6),twist),-s.velocity,s.velocity)
        candidate,ok=checked_plan(self.kin,data.qpos[self._qpos_ids],data.qvel[self._qvel_ids],info['post_state'],
            desired,info['upstream_velocity'],hold,self._lower,self._upper,s,p,self.n_substeps,self.stop_steps)
        normal=~request&self._arm_motion_allowed(data,info)&~info['controller_failed']
        opening_unlocked=normal&(info['coord_phase']==GRIP)
        accept=normal&ok
        backup=jp.concatenate([info['stop_backup'],jp.tile(info['stop_backup'][-1],(self.n_substeps,1))])
        plan=jp.where(accept,candidate,backup)

        def step(carry,i):
            current,st,phase,timer,stable,arm_stable,actuated,gref,start_width,window,completed,failed,reason,overflow,task=carry
            already_failed=failed
            planned=jp.where(already_failed,st,plan[i]);old_a=st[12:];jerk=(planned[12:]-old_a)/s.dt
            distance=floor_distance(fk_floor(self.kin,planned[:6],s.floor_normal_base)[0],s)
            code=(jp.any((planned[:6]<self._lower-s.tolerance)|(planned[:6]>self._upper+s.tolerance)).astype(jp.int32)
                +2*(distance < -s.tolerance).astype(jp.int32)
                +4*(jp.max(jp.abs(planned[6:12]))>s.velocity+1e-5).astype(jp.int32)
                +8*(jp.max(jp.abs(planned[12:]))>s.acceleration+1e-3).astype(jp.int32)
                +16*(jp.max(jp.abs(jerk))>s.jerk+.2).astype(jp.int32)
                +32*(~jp.all(jp.isfinite(planned))).astype(jp.int32))
            failed=failed|(code!=0);reason=jp.where(reason==0,code,reason)
            st=jp.where(failed,st,planned)
            # Only actuate once reference p/v/a and measured arm velocity settle.
            stopped=(jp.max(jp.abs(st[6:]))<1e-4)&(jp.max(jp.abs(current.qvel[self._qvel_ids]))<=c.grip_arm_stable_speed)
            arm_stable=jp.where((phase==BRAKE)&stopped,arm_stable+s.dt,0.)
            start=(phase==BRAKE)&(arm_stable>=c.grip_arm_stable_seconds-1e-6)&(i>=hold)
            start=start&~failed
            phase=jp.where(start,GRIP,phase);timer=jp.where(start,0.,timer)
            actuated=jp.where(start,goal,actuated)
            current_width=current.qpos[self._finger_qpos_ids[0]]-current.qpos[self._finger_qpos_ids[1]]
            start_width=jp.where(start,current_width,start_width)
            rate=c.gripper_rate*s.dt/self.dt
            gref=jp.where((phase==GRIP)&~failed,jp.clip(actuated,gref-rate,gref+rate),gref)
            old_width=current.qpos[self._finger_qpos_ids[0]]-current.qpos[self._finger_qpos_ids[1]]
            applied=ctrl.at[self._arm_ctrl_ids].set(st[:6]).at[self._gripper_ctrl_id].set(gref)
            previous=current
            current=mjx.step(self._mjx_model,current.replace(ctrl=applied))
            task=self._task_substep(previous,current,action,task)
            width=current.qpos[self._finger_qpos_ids[0]]-current.qpos[self._finger_qpos_ids[1]]
            width_speed=jp.abs(width-old_width)/s.dt
            window=jp.where(phase==GRIP,jp.concatenate([window[1:],width[None]]),jp.full_like(window,width))
            # Contact chatter can have large instantaneous derivatives while
            # staying within a submillimetre envelope. Check the entire100ms.
            width_range=jp.max(window)-jp.min(window)
            stable=jp.where((phase==GRIP)&(width_range<=c.grip_stable_speed*c.grip_stable_seconds),timer+s.dt,0.)
            done=(phase==GRIP)&~failed&stroke_complete(window,gref,actuated,start_width,timer+s.dt,c)
            completed=completed+done.astype(jp.float32)
            phase=jp.where(done,MOVE,phase);stable=jp.where(done,0.,stable)
            timer=jp.where(phase==MOVE,0.,timer+s.dt)
            timeout=((phase==GRIP)&(timer>c.grip_timeout))|((phase==BRAKE)&(timer>c.brake_timeout))
            gap=jp.max(jp.abs(st[:6]-current.qpos[self._qpos_ids]))
            cap=gap>s.lag+p.extra+s.tolerance
            code=code+64*cap.astype(jp.int32)+128*timeout.astype(jp.int32)
            failed=failed|cap|timeout;reason=jp.where(reason==0,code,reason)
            row=jp.concatenate([st[:6],st[6:12],current.qpos[self._qpos_ids],
                jp.array([gap,jp.maximum(gap-s.lag,0.),distance,jp.max(jp.abs(st[12:])),code.astype(st.dtype),0.])])
            detail=jp.array([phase,goal,actuated,width,width_speed,jp.max(jp.abs(st[6:12])),
                jp.max(jp.abs(st[12:])),jp.max(jp.abs(jerk)),done,accept,normal&~ok,timeout,
                jp.max(jp.abs(current.qvel[self._qvel_ids]))],dtype=st.dtype)
            return (current,st,phase,timer,stable,arm_stable,actuated,gref,start_width,window,completed,failed,reason,
                overflow|self._capacity_overflow(current),task),(row,detail)
        initial=(data,info['post_state'],phase,timer,info['coord_stable'],info['arm_stable'],info['actuated_goal'],
            info['grip_reference'],info['grip_start_width'],info['grip_width_window'],info['operations_completed'],info['controller_failed'],info['controller_failure_reason'],self._capacity_overflow(data),self._start_task_scan(data,info))
        final,(trace,detail)=jax.lax.scan(step,initial,jp.arange(self.n_substeps))
        data,st,phase,timer,stable,arm_stable,actuated,gref,start_width,window,completed,failed,reason,overflow,task=final
        return data,overflow,dict(post_state=st,upstream_goal=st[:6],upstream_velocity=jp.where(normal,desired,0.),
            stop_backup=plan[self.n_substeps:],coord_phase=phase,coord_timer=timer,coord_stable=stable,
            arm_stable=arm_stable,actuated_goal=actuated,grip_reference=gref,grip_start_width=start_width,grip_width_window=window,
            operations_completed=completed,candidate_rejections=info['candidate_rejections']+(normal&~ok).astype(jp.float32),
            controller_failed=failed,controller_failure_reason=reason,controller_intervention=(normal&~ok).astype(jp.float32),
            tracking_trace=trace,coord_trace=detail,grip_motion_unlocked=opening_unlocked,
            **self._finish_task_scan(task))

    def step(self,state,action):
        updated=super().step(state,action)
        if getattr(self,'independent_gripper',False):
            from .direct_gripper import arm_metrics
            return updated.replace(metrics=dict(updated.metrics,**arm_metrics(self,updated)),
                done=jp.maximum(updated.done,updated.info['controller_failed'].astype(jp.float32)))
        tr=updated.info['coord_trace']
        metrics=dict(updated.metrics,coord_busy=(updated.info['coord_phase']!=MOVE).astype(jp.float32),
            coord_grip=(updated.info['coord_phase']==GRIP).astype(jp.float32),
            coord_completed=updated.info['operations_completed']-state.info['operations_completed'],
            coord_rejected=updated.info['candidate_rejections']-state.info['candidate_rejections'],
            coord_timeout=jp.max(tr[:,11]),reference_jerk=jp.max(tr[:,7]),
            coord_reason=updated.info['controller_failure_reason'].astype(jp.float32),
            coord_width=tr[-1,3],coord_accepted_goal=updated.info['gripper_goal'],
            coord_phase_metric=updated.info['coord_phase'].astype(jp.float32),
            coord_reference_fault=((updated.info['controller_failure_reason']&63)!=0).astype(jp.float32),
            grip_arm_reference_speed=jp.max(jp.where(tr[:,0]==GRIP,tr[:,5],0.)),
            grip_arm_locked_reference_speed=jp.max(jp.where((tr[:,0]==GRIP)&~updated.info['grip_motion_unlocked'],tr[:,5],0.)))
        return self._augment(updated.replace(metrics=metrics,done=jp.maximum(updated.done,updated.info['controller_failed'].astype(jp.float32))))
