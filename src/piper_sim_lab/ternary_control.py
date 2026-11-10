"""SHOW-76 task with a latched three-command gripper and recovering DLS+QP.

The original reward, reset distribution, success predicate and raw PPO action
are retained. A hold command retains the FINAL goal, not the ramped actuator
reference and not the measured width. No task-dependent action overrides.
"""
from pathlib import Path
import math
import jax.numpy as jp
from piper_policy_runtime.kinematics import PiperKinematics
from piper_sim_lab.pick_place import PiperPickPlace, default_config
from piper_sim_lab.scene_frames import SceneFrames
from piper_sim_lab.control_lab.controllers import Settings
from piper_sim_lab.control_lab.hierarchy_env import HierarchicalPickPlace
from piper_sim_lab.control_lab.tracking_qp import TrackingSettings
from piper_sim_lab.control_lab.tracking_env import advance_tracking


def gripper_command(action, previous_goal, previous_reference, maximum, rate, threshold):
    goal = jp.where(action <= -threshold, 0.,
                    jp.where(action >= threshold, maximum, previous_goal))
    reference = jp.clip(goal, previous_reference-rate, previous_reference+rate)
    return goal, reference


class TernaryControlledPickPlace(HierarchicalPickPlace):
    def __init__(self, config_overrides=None, task_defaults=None):
        config = default_config()
        config.update(dict(control_mode='qp', control_acceleration=8., control_floor=0.,control_jerk=0.,
            tracking_extra=.02, tracking_margin=.02, tracking_horizon=.05,
            tracking_weight=100., tracking_timeout=.5, gripper_threshold=.5, gripper_mode="ternary_latched",
            gripper_goal_obs=True, finger_bridge=True, tcp_local_z=.1385))
        if task_defaults:
            config.update(task_defaults)
        PiperPickPlace.__init__(self, config=config, config_overrides=config_overrides)
        c = self._config
        if c.control_mode != 'qp' or not c.gripper_goal_obs or c.gripper_mode != 'ternary_latched':
            raise ValueError('SHOW-100 requires QP and observable gripper final goal')
        if not 0 < c.gripper_threshold < 1:
            raise ValueError('gripper_threshold must be in (0,1)')
        if not math.isfinite(c.tcp_local_z) or c.tcp_local_z <= 0:
            raise ValueError('positive finite TCP local z required')
        if not math.isfinite(c.control_jerk) or c.control_jerk<0:
            raise ValueError('control_jerk must be finite and nonnegative; zero disables jerk limits')
        if not 0 < c.tracking_margin < c.max_target_lag:
            raise ValueError('tracking margin must be in (0, max_target_lag)')
        self.mode = 'qp'
        self.frames = SceneFrames.from_config(c)
        self.settings = Settings(dt=self.sim_dt, policy_dt=self.dt,
            linear_scale=c.max_lin_vel, angular_scale=c.max_ang_vel,
            damping=c.ik_damping, velocity=c.max_joint_vel, lag=c.max_target_lag,
            acceleration=c.control_acceleration, tcp_floor=c.control_floor,
            qp_jerk=c.control_jerk>0,jerk=c.control_jerk if c.control_jerk>0 else 100.,
            base_height=c.base_height).with_scene_frames(self.frames)
        self.kin = PiperKinematics(Path(self._xml_path).parent / 'piper.xml')
        self.kin.tcp_local_position = tuple(self._mj_model.site_pos[self._tcp_id])
        self.tracking_settings = TrackingSettings(extra=c.tracking_extra, margin=c.tracking_margin,
            horizon=c.tracking_horizon, weight=c.tracking_weight, timeout=c.tracking_timeout)

    def _reset_controller(self, state):
        state = super()._reset_controller(state)
        goal = state.data.ctrl[self._gripper_ctrl_id]
        z = jp.array(0., jp.float32)
        return state.replace(info=dict(state.info, gripper_goal=goal,
            tracking_recovering=jp.zeros(6, bool), tracking_timer=jp.zeros(6),
            tracking_trace=jp.zeros((self.n_substeps, 24))),
            metrics=dict(state.metrics, controller_failed=z, gripper_goal_switch=z,
                gripper_hold=z, gripper_goal_open=goal/self._config.gripper_max))

    def _gripper_command(self, data, action, info):
        goal, reference = gripper_command(action[6], info['gripper_goal'],
            data.ctrl[self._gripper_ctrl_id], self._config.gripper_max,
            self._config.gripper_rate, self._config.gripper_threshold)
        return reference, dict(gripper_goal=goal)

    def _advance_control(self, data, ctrl, action, hold, info):
        return advance_tracking(self, data, ctrl, action, hold, info)

    def step(self, state, action):
        previous_goal = state.info['gripper_goal']
        updated = super().step(state, action)
        return updated.replace(metrics=dict(updated.metrics,
            controller_failed=updated.info['controller_failed'].astype(jp.float32),
            gripper_goal_switch=(updated.info['gripper_goal'] != previous_goal).astype(jp.float32),
            gripper_hold=(jp.abs(action[6]) < self._config.gripper_threshold).astype(jp.float32),
            gripper_goal_open=updated.info['gripper_goal']/self._config.gripper_max))
