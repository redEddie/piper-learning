"""A single Piper reaching task, vectorized by Playground's training wrapper.

The robot, its contact geometry, equality constraint, and position actuators
come from the pinned Menagerie asset. Task markers are added in memory only.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
import numpy as np

from piper_sim_lab.assets import get_model_path


ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
TCP_SITE = "piper_tcp"
TARGET_BODY = "reach_target"
# Both distal finger pads have their centers at this link6-local z coordinate:
# finger body z=.13503 + rotated pad offset z=-.045. The two pads are symmetric.
TCP_LOCAL_POS = (0.0, 0.0, 0.09003)
# MuJoCo Warp OverflowType bits 9/10 indicate solver/line-search iteration
# limits, not exhausted contact/constraint buffers. Treat every other bit,
# including future unknown bits, as an invalid capacity transition.
_SOLVER_LIMIT_BITS = (1 << 9) | (1 << 10)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.05,
        sim_dt=0.005,
        episode_length=100,
        action_repeat=1,
        impl="jax",
        batch_capacity=1,
        contacts_per_env=160,
        max_constraints_per_env=512,
        ccd_pairs_per_env=4,
        solver_iterations=8,
        solver_ls_iterations=4,
        action_scale=0.05,
        gripper_opening=0.025,
        nominal_qpos=[0.0, 0.7, -0.9, 0.0, 0.3, 0.0],
        reset_joint_noise=0.05,
        goal_joint_radius=[0.45, 0.3, 0.35, 0.4, 0.3, 0.4],
        success_distance=0.03,
        reach_scale=5.0,
        action_cost=0.001,
        action_rate_cost=0.01,
    )


class PiperReach(mjx_env.MjxEnv):
    """Six joint-delta actions with fixed opening; success does not end an episode.

    Observation order is arm qpos (6), arm qvel (6), TCP position (3),
    target-minus-TCP (3), arm control-target-minus-qpos (6), and the previous
    clipped action (6). The tracking error exposes the accumulated actuator
    targets needed to predict the next control step. Targets are
    kinematically reachable positions sampled around the nominal joint pose;
    this is not a guarantee of a collision-free path to every target.
    """

    PROGRESS_METRIC = "distance"
    SUCCESS_DEFINITION = "TCP distance < success_distance (m); position-only reaching, not grasping"

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        config = default_config() if config is None else deepcopy(config)
        super().__init__(config, config_overrides)
        self._validate_config()
        self._xml_path = str(get_model_path())
        spec = mujoco.MjSpec.from_file(self._xml_path)
        spec.option.timestep = self.sim_dt
        spec.option.iterations = self._config.solver_iterations
        spec.option.ls_iterations = self._config.solver_ls_iterations
        spec.body("link6").add_site(
            name=TCP_SITE, pos=TCP_LOCAL_POS, size=[0.008, 0.008, 0.008],
            rgba=[0.1, 0.5, 1.0, 1.0], type=mujoco.mjtGeom.mjGEOM_SPHERE,
        )
        target = spec.worldbody.add_body(name=TARGET_BODY, mocap=True)
        target.add_site(
            name="target_marker", size=[self._config.success_distance] * 3,
            rgba=[0.2, 0.9, 0.3, 0.35], type=mujoco.mjtGeom.mjGEOM_SPHERE,
        )
        self._mj_model = spec.compile()
        self._mjx_model = mjx_env.put_model(self._mj_model, impl=self._config.impl)
        model = self._mj_model
        joints = np.array([model.joint(name).id for name in ARM_JOINTS])
        self._qpos_ids = np.array(model.jnt_qposadr[joints])
        self._qvel_ids = np.array(model.jnt_dofadr[joints])
        self._arm_ctrl_ids = np.array([model.actuator(name).id for name in ARM_JOINTS])
        self._gripper_ctrl_id = model.actuator("gripper").id
        self._finger_qpos_ids = np.array([
            model.jnt_qposadr[model.joint(name).id] for name in ("joint7", "joint8")
        ])
        self._tcp_id = model.site(TCP_SITE).id
        self._target_id = int(model.body(TARGET_BODY).mocapid[0])
        joint_ranges = model.jnt_range[joints]
        ctrl_ranges = model.actuator_ctrlrange[self._arm_ctrl_ids]
        self._lower = jp.array(np.maximum(joint_ranges[:, 0], ctrl_ranges[:, 0]))
        self._upper = jp.array(np.minimum(joint_ranges[:, 1], ctrl_ranges[:, 1]))
        nominal = np.asarray(self._config.nominal_qpos, dtype=np.float32)
        if np.any(nominal < np.asarray(self._lower)) or np.any(nominal > np.asarray(self._upper)):
            raise ValueError("nominal_qpos must lie inside joint and actuator limits")
        opening = self._config.gripper_opening
        gripper_limits = model.actuator_ctrlrange[self._gripper_ctrl_id]
        if not gripper_limits[0] <= opening <= gripper_limits[1]:
            raise ValueError("gripper_opening is outside the actuator range")
        self._nominal = jp.array(nominal)
        self._goal_radius = jp.array(self._config.goal_joint_radius, dtype=jp.float32)
        self._initial_qpos = jp.array(model.qpos0, dtype=jp.float32)
        self._initial_qpos = self._initial_qpos.at[self._qpos_ids].set(self._nominal)
        self._initial_qpos = self._initial_qpos.at[self._finger_qpos_ids].set(
            jp.array([opening, -opening], dtype=jp.float32)
        )
        self._initial_ctrl = jp.zeros(model.nu, dtype=jp.float32)
        self._initial_ctrl = self._initial_ctrl.at[self._arm_ctrl_ids].set(self._nominal)
        self._initial_ctrl = self._initial_ctrl.at[self._gripper_ctrl_id].set(opening)

    def _validate_config(self) -> None:
        c = self._config
        for name in (
            "solver_iterations", "solver_ls_iterations", "batch_capacity",
            "contacts_per_env", "max_constraints_per_env", "ccd_pairs_per_env",
        ):
            if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(c.ctrl_dt) or not np.isfinite(c.sim_dt) or min(c.ctrl_dt, c.sim_dt) <= 0:
            raise ValueError("ctrl_dt and sim_dt must be positive and finite")
        if not np.isclose(c.ctrl_dt / c.sim_dt, round(c.ctrl_dt / c.sim_dt)):
            raise ValueError("ctrl_dt must be an integer multiple of sim_dt")
        for name in ("nominal_qpos", "goal_joint_radius"):
            value = np.asarray(c[name], dtype=float)
            if value.shape != (6,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain six finite values")
        if np.any(np.asarray(c.goal_joint_radius) < 0):
            raise ValueError("goal_joint_radius must be nonnegative")
        for name in ("action_scale", "success_distance", "reach_scale"):
            if not np.isfinite(c[name]) or c[name] <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("reset_joint_noise", "action_cost", "action_rate_cost", "gripper_opening"):
            if not np.isfinite(c[name]) or c[name] < 0:
                raise ValueError(f"{name} must be nonnegative and finite")

    def _make_data_kwargs(self) -> dict[str, int]:
        """Warp contact/CCD arrays are shared; constraint rows are per world.

        These bounds come from docs/evidence/warp-capacity-cpu-audit.json and
        apply to the pinned Piper geometry, collision masks and elliptic cone.
        The caller must set batch_capacity to its actual vectorized batch size.
        """
        if self._config.impl != "warp":
            return {}
        return {
            "naconmax": self._config.contacts_per_env * self._config.batch_capacity,
            "njmax": self._config.max_constraints_per_env,
            "naccdmax": self._config.ccd_pairs_per_env * self._config.batch_capacity,
        }

    def _capacity_overflow(self, data: mjx.Data) -> jax.Array:
        if self._config.impl != "warp":
            return jp.array(False)
        # The per-world field is scalar inside vmap. Do not default a missing
        # field to zero: incompatible Warp APIs must fail instead of hiding it.
        bits = jp.asarray(data._impl.overflow, dtype=jp.int32)
        return jp.any(jp.bitwise_and(bits, ~_SOLVER_LIMIT_BITS) != 0)

    def _step_physics(self, data: mjx.Data, ctrl: jax.Array) -> tuple[mjx.Data, jax.Array]:
        if self._config.impl != "warp":
            return mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps), jp.array(False)

        def substep(carry, _):
            current, overflow = carry
            current = mjx.step(self._mjx_model, current.replace(ctrl=ctrl))
            return (current, overflow | self._capacity_overflow(current)), None

        # Same number of physics calls as mjx_env.step, with a boolean carried
        # through the scan so a transient overflow cannot vanish at the last
        # substep. Include the input flag, including any overflow during reset.
        return jax.lax.scan(
            substep, (data, self._capacity_overflow(data)), None,
            length=self.n_substeps,
        )[0]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, initial_key, goal_key = jax.random.split(rng, 3)
        initial_joints = jp.clip(
            self._nominal + jax.random.uniform(initial_key, (6,), minval=-1.0, maxval=1.0)
            * self._config.reset_joint_noise,
            self._lower, self._upper,
        )
        goal_joints = jp.clip(
            self._nominal + jax.random.uniform(goal_key, (6,), minval=-1.0, maxval=1.0)
            * self._goal_radius,
            self._lower, self._upper,
        )
        data = mjx_env.make_data(
            self._mj_model,
            qpos=self._initial_qpos.at[self._qpos_ids].set(initial_joints),
            qvel=jp.zeros(self._mj_model.nv, dtype=jp.float32),
            ctrl=self._initial_ctrl.at[self._arm_ctrl_ids].set(initial_joints),
            impl=self._config.impl,
            **self._make_data_kwargs(),
        )
        # Kinematics is sufficient to construct the goal, without a physics step.
        goal_data = mjx.kinematics(
            self._mjx_model,
            data.replace(qpos=self._initial_qpos.at[self._qpos_ids].set(goal_joints)),
        )
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._target_id].set(goal_data.site_xpos[self._tcp_id]),
        )
        data = mjx.forward(self._mjx_model, data)
        capacity_overflow = self._capacity_overflow(data)
        zero = jp.zeros((), dtype=jp.float32)
        distance = jp.linalg.norm(data.mocap_pos[self._target_id] - data.site_xpos[self._tcp_id])
        metrics = {
            "distance": distance,
            "success": ((distance < self._config.success_distance) & ~capacity_overflow).astype(jp.float32),
            "reward_reach": zero,
            "action_penalty": zero,
            "action_rate_penalty": zero,
            "nonfinite": zero,
            "capacity_overflow": capacity_overflow.astype(jp.float32),
        }
        info = {"rng": rng, "goal_qpos": goal_joints}
        return mjx_env.State(
            data=data, obs=self._get_obs(data, jp.zeros(6, dtype=jp.float32)),
            reward=zero, done=capacity_overflow.astype(jp.float32), metrics=metrics, info=info,
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.asarray(action, dtype=jp.float32)
        if action.shape != (self.action_size,):
            raise ValueError(f"expected action shape (6,), got {action.shape}")
        nonfinite = ~jp.all(jp.isfinite(action))
        nonfinite |= ~jp.all(jp.isfinite(state.data.qpos))
        nonfinite |= ~jp.all(jp.isfinite(state.data.qvel))
        nonfinite |= ~jp.all(jp.isfinite(state.data.ctrl))
        action = jp.clip(jp.nan_to_num(action), -1.0, 1.0)
        # A failed state still produces finite observations/reward for the PPO
        # terminal transition. The training wrapper resets it when done is true.
        safe_data = state.data.replace(
            qpos=jp.where(jp.isfinite(state.data.qpos), state.data.qpos, self._initial_qpos),
            qvel=jp.nan_to_num(state.data.qvel, posinf=0.0, neginf=0.0),
            ctrl=jp.where(jp.isfinite(state.data.ctrl), state.data.ctrl, self._initial_ctrl),
        )
        joint_targets = jp.clip(
            safe_data.ctrl[self._arm_ctrl_ids] + action * self._config.action_scale,
            self._lower, self._upper,
        )
        ctrl = safe_data.ctrl.at[self._arm_ctrl_ids].set(joint_targets)
        ctrl = ctrl.at[self._gripper_ctrl_id].set(self._config.gripper_opening)
        data, capacity_overflow = self._step_physics(safe_data, ctrl)
        # MJX step leaves derived positions at the integration input; update the
        # TCP to the resulting qpos before computing observations and rewards.
        data = mjx.kinematics(self._mjx_model, data)
        distance = jp.linalg.norm(data.mocap_pos[self._target_id] - data.site_xpos[self._tcp_id])
        nonfinite |= ~jp.all(jp.isfinite(data.qpos)) | ~jp.all(jp.isfinite(data.qvel))
        nonfinite |= ~jp.isfinite(distance)
        invalid = nonfinite | capacity_overflow
        reach = 1.0 - jp.tanh(self._config.reach_scale * distance)
        action_penalty = self._config.action_cost * jp.sum(jp.square(action))
        # Cached autoreset restores obs but not arbitrary info. Keeping the
        # action history in obs makes both Playground reset modes consistent.
        previous_action = jp.nan_to_num(state.obs["state"][-6:])
        rate_penalty = self._config.action_rate_cost * jp.sum(jp.square(action - previous_action))
        reward = jp.where(invalid, -1.0, reach - action_penalty - rate_penalty)
        # Training/evaluation wrappers may add metrics (for example "reward").
        # Preserve their keys so scans retain the same state pytree structure.
        metrics = dict(state.metrics)
        metrics.update({
            "distance": jp.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0),
            "success": ((distance < self._config.success_distance) & ~invalid).astype(jp.float32),
            "reward_reach": jp.where(invalid, 0.0, reach),
            "action_penalty": action_penalty,
            "action_rate_penalty": rate_penalty,
            "nonfinite": nonfinite.astype(jp.float32),
            "capacity_overflow": capacity_overflow.astype(jp.float32),
        })
        return state.replace(
            data=data, obs=self._get_obs(data, action), reward=reward,
            done=invalid.astype(jp.float32), metrics=metrics,
        )

    def _get_obs(self, data: mjx.Data, previous_action: jax.Array) -> dict[str, jax.Array]:
        tcp = data.site_xpos[self._tcp_id]
        obs = jp.concatenate((
            data.qpos[self._qpos_ids], data.qvel[self._qvel_ids], tcp,
            data.mocap_pos[self._target_id] - tcp,
            data.ctrl[self._arm_ctrl_ids] - data.qpos[self._qpos_ids],
            previous_action,
        ))
        return {"state": jp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)}

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 6

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
