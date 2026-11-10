"""Piper pick-and-place: a block on the table goes inside a bowl.

Objects are added to the pinned Menagerie model in memory. The block is a
free body; the bowl is a mocap body (kinematic, immovable) built from boxes
so it stays MJX-safe. The policy acts in task space: an end-effector twist in
the base frame plus a gripper width, mapped to joint targets by damped
least-squares velocity IK (SHOW-24). Predicates follow the Dex-One2Many Doll
chain pre_grasp -> grasp -> pre_inside -> inside (SHOW-20).
"""

from __future__ import annotations

from copy import deepcopy
import itertools
import math
from typing import Any

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import support as mjx_support
from mujoco_playground._src import mjx_env
import numpy as np

from piper_sim_lab.assets import get_model_path
from piper_sim_lab.env import ARM_JOINTS, _SOLVER_LIMIT_BITS

# Grasp point (TCP site on link6) and the link6 axis the fingers close along, per model.
# menagerie (v0): centre of the fingertip plates, link6 z 0.12, fingers along y.
# piper_v100: midpoint of the fingertips, link6 z 0.1358 (SHOW-79 measurement), fingers along x.
TCP_SITE = "piper_grasp"
ROBOT_MODELS = {
    "menagerie": {"tcp": (0.0, 0.0, 0.12), "closing_axis": 1},
    "piper_v100": {"tcp": (0.0, 0.0, 0.1358), "closing_axis": 0},
}
BLOCK_BODY = "block"
BOWL_BODY = "bowl"
N_STAGES = 4
ACTION_SIZE = 7


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.1,
        sim_dt=0.005,
        episode_length=80,
        # Robot model (assets.MODEL_DIRS) and the mounting plate under its base.
        # Positions in the observation are in the robot base frame, so the
        # plate shifts table-frame z by -base_height. v1: piper_v100, 0.015.
        robot_model="menagerie",
        base_height=0.0,
        finger_bridge=False,   # opt-in mesh-derived connector gap repair
        tcp_local_z=0.0,        # 0 retains the original model TCP
        gripper_goal_obs=False, # opt-in normalized final goal before previous action
        # Floor geom priority. The block has priority 1 so pad contacts use its
        # (randomized) friction; table_priority 2 makes block-table contacts use
        # the table's friction instead, so table friction randomization applies.
        table_priority=0,
        action_repeat=1,
        impl="jax",
        batch_capacity=1,
        # Warp's contact and CCD buffers are shared by the whole batch
        # (naconmax / naccdmax = per-env value * batch_capacity). Box-box pairs
        # (block vs. bowl floor + 12 wall boxes + finger pads) go through the
        # GJK/EPA CCD path, and a 5 cm cube resting in the bowl touches several
        # wall boxes at once: with 4 CCD pairs per env the deterministic eval,
        # where every env reaches "inside" at the same step, raised the CCD
        # overflow bit and aborted pp-full-03. 16 pairs per env left no overflow
        # in a 128-env rollout; the contact bound is raised for headroom.
        contacts_per_env=384,
        max_constraints_per_env=1024,
        ccd_pairs_per_env=16,
        solver_iterations=8,
        # SHOW-62: budget 4 can stop Newton before contact forces converge.
        solver_ls_iterations=20,
        # Top-down hover: grasp point at (0.30, 0, 0.07) m with the gripper pointing
        # at the table, 0.22 rad inside the joint5 limit (wrist singularity avoided).
        nominal_qpos=[0.0, 1.743, -1.087, 0.0, 1.002, 0.0],
        reset_joint_noise=0.05,
        # Objects (metres, kg). The team's block is a 5 cm green PLA cube of
        # about 80 g (SHOW-20); configs/pick_place_block30.yaml keeps the
        # earlier 3 cm / 20 g red block. The block must fit the 70 mm gripper
        # opening (2 * gripper_max) and the bowl floor (bowl_r_bottom).
        block_half=0.025,
        block_mass=0.08,
        block_rgba=[0.1, 0.8, 0.2, 1.0],
        bowl_r_bottom=0.055,
        bowl_r_rim=0.075,
        bowl_height=0.05,
        bowl_wall=0.004,
        bowl_floor=0.004,
        bowl_segments=12,
        # Placement regions in the base frame: block on the robot's right (-y),
        # bowl on its left (+y), so they never overlap.
        block_x=[0.25, 0.40],
        block_y=[-0.15, -0.02],
        bowl_x=[0.25, 0.35],
        bowl_y=[0.12, 0.20],
        # Minimum block-bowl centre distance at an empty-hand start (m). 0 keeps
        # the v0 behaviour (disjoint ranges). With overlapping ranges the bowl is
        # drawn from spawn_candidates candidates and the first one far enough
        # from the block is used (the farthest one if none is).
        spawn_min_distance=0.0,
        spawn_candidates=8,
        # Objects farther than this from the robot base (xy, m) are out of
        # top-down reach and are not spawned. 0 disables the limit (v0).
        spawn_max_radius=0.0,
        # Task-space action: unit action -> max end-effector velocity.
        max_lin_vel=0.10,
        max_ang_vel=1.0,
        gripper_max=0.035,
        # Gripper semantics (decided 2026-10-21): the gripper is either open or
        # closed, and when closed it squeezes the object with a per-task force
        # cap, like the real Piper's GripperCtrl(angle, effort). In MuJoCo the
        # cap is the actuator force range; kp is high enough that the servo
        # saturates at the cap within a few millimetres of error, so the grip
        # force is gripper_force whatever the object size. The target still
        # moves at most gripper_rate per control step (finger speed).
        # gripper_force is the per-finger normal force in newtons: an 80 g PLA
        # cube needs about 0.8 N per finger at friction 0.5, so 3 N is a 4x
        # margin and well below the asset's 10 N. Map it to the real effort
        # value with a force gauge (SHOW-63).
        gripper_force=3.0,
        gripper_kp=400.0,
        gripper_rate=0.01,         # m per control step; max change of the finger target
        gripper_binary=True,       # action[6] > 0 opens, <= 0 closes; False keeps the linear width map
        # Velocity IK.
        jacobian="mjx",            # "mjx" (mjx.jac) or "fd" (finite differences through kinematics)
        ik_damping=0.05,
        max_joint_vel=1.5,
        max_target_lag=0.15,       # rad; bound on actuator target minus measured joint
        # Stage-wise resets (SHOW-50): probabilities of starting with an empty
        # hand (stage 0), the gripper 3 cm above the block (pre_grasp), or the
        # block already held 8 cm above the table (grasp). The pre-grasp and
        # grasp starts come from a table of top-down joint configurations solved
        # with MuJoCo IK at construction time.
        # Four start stages (the paper resets at every stage): empty hand,
        # pre_grasp, grasp (block held above its own position), pre_inside
        # (block held above the bowl). A 3-entry list is accepted for old
        # run configs and means no pre_inside starts.
        reset_stage_probs=[0.4, 0.2, 0.2, 0.2],
        pregrasp_height=0.03,
        lift_reset_height=0.08,
        preinside_height=0.03,     # block bottom above the bowl rim at a pre_inside start
        reset_table_grid=4,
        # Second IK seed for the reset table: a top-down arm configuration (the v0
        # start pose), used when the start pose itself is far from top-down.
        reset_ik_seed=[0.0, 1.743, -1.087, 0.0, 1.002, 0.0],
        reset_ik_flip=False,       # also accept the grasp turned 180 degrees (v1); False keeps v0 tables unchanged
        reset_table_yaws=8,
        # Observation. The paper's actor observation is node/edge features only
        # (proprioception, object pose/geometry, relative poses) with no stage
        # input. stage_obs=True appends the stage one-hot (4) to obs['state'].
        # With the v0 resets/reward the stage input was decisive (pp-full-04
        # 82.8% vs pp-full-05 0%); with pre_inside resets and the placed/release
        # reward the stage-free policy reaches 93.0% (pp-full-08,
        # configs/pick_place_nostage.yaml). At deployment the stage is computed
        # by the same predicates from estimated poses, width and TCP.
        stage_obs=True,
        # Reward and predicates. inside_reward is paid while the block is in the
        # bowl, held or not; release_bonus is added once the hand has let go
        # (TCP farther than grasp_distance), which is also the success test.
        reach_scale=5.0,
        inside_reward=5.0,
        release_bonus=2.0,
        grasp_distance=0.02,
        lift_height=0.02,
        action_cost=0.001,
        action_rate_cost=0.01,
        # Bowl-arm collision. False (v0) lets the arm links pass through the
        # bowl wall; True adds the arm capsules (and palm) to the bowl's
        # collision partners. Defaults keep old run configs reproducible.
        bowl_arm_collision=False,
        # Free bowl. False (v0) keeps the bowl a mocap body lifted 2 mm off the
        # table; True gives it a free joint and bowl_mass so the arm or the block
        # can push or tip it, and it rests on the table (table friction applies,
        # the floor has the higher priority). Requires bowl_arm_collision.
        bowl_free=False,
        bowl_mass=0.11,
        # Domain randomization, Dex-One2Many Table 10. All off by default so old
        # run configs evaluate unchanged; configs/pick_place_dr.yaml turns them on.
        # size_obs appends the perceived block half-size (1) to obs['state'] and
        # the true block position/rotation, bowl position and half-size (10) to
        # obs['privileged'], so the critic keeps clean state.
        size_obs=False,
        # Actor-observation errors and delays. Without size_obs the size term has
        # nowhere to go (lets older policies be evaluated under the other errors).
        obs_randomization=False,
        obj_pos_noise=0.01,        # m, Gaussian per axis per step, block
        obj_pos_noise_cap=0.03,
        static_pos_noise=0.005,    # m, Gaussian per axis, bowl, drawn once per episode
        static_pos_noise_cap=0.015,
        obj_rot_noise=5.0,         # deg, rotation about a uniform random axis per step
        size_noise=0.10,           # perceived size scale 1 +- 10 %, once per episode
        obj_pose_delay_max=0.15,   # s, block pose observation delay, once per episode
        bowl_pose_delay_max=0.0,   # s, opt-in separate bowl stream; legacy observations unchanged
        robot_state_delay_max=0.05,  # s, proprioception delay
        action_delay_max=0.05,     # s, the previous command is held this long into the step
        # Physical properties, sampled once per environment slot by
        # physics_randomization_fn (Playground's per-env model batching).
        physics_randomization=False,
        block_mass_range=[0.064, 0.096],   # kg, 80 g +- 20 % (the paper's m +- dm)
        block_friction_scale=0.10,         # block friction 1 +- 10 %
        table_friction_range=[0.3, 0.6],
        gripper_gain_scale=[0.8, 1.25],    # gripper stiffness and damping multipliers
    )


class PiperPickPlace(mjx_env.MjxEnv):
    """Task-space actions (7): EE linear velocity 3, angular velocity 3, gripper 1.

    Observation ``obs['state']`` (51): arm qpos 6, arm qvel 6, arm
    control-target-minus-qpos 6, TCP position 3,
    TCP rotation vector 3, gripper width 1, block position 3, block rotation
    vector 3, bowl position 3, block-minus-TCP 3, block-minus-bowl 3,
    stage one-hot 4 (omitted when ``stage_obs`` is false), perceived block
    half-size 1 (only with ``size_obs``), previous action 7.
    ``obs['privileged']`` adds block linear velocity 3 and the predicate
    flags 4 for the critic, plus the true block position 3, rotation vector 3,
    bowl position 3 and half-size 1 with ``size_obs``.

    With ``obs_randomization`` the robot-state and object terms of
    ``obs['state']`` are what a deployed robot would perceive: delayed robot
    state, a delayed and noisy block pose, a noisy bowl position and a noisy
    block size. Reward, predicates and stage use the true state.
    """

    PROGRESS_METRIC = "block_bowl_xy"
    SUCCESS_DEFINITION = "block centre inside the bowl floor radius, below the rim, not held (inside predicate)"

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        config = default_config() if config is None else deepcopy(config)
        super().__init__(config, config_overrides)
        self._validate_config()
        c = self._config
        self._xml_path = str(get_model_path(c.robot_model))
        robot = ROBOT_MODELS[c.robot_model]
        self._closing_axis = robot["closing_axis"]
        self._base_offset = jp.array([0.0, 0.0, c.base_height], dtype=jp.float32)
        spec = mujoco.MjSpec.from_file(self._xml_path)
        spec.body("base_link").pos = list(np.asarray(spec.body("base_link").pos) + [0.0, 0.0, c.base_height])
        if c.finger_bridge:
            from piper_sim_lab.finger_collision import add_finger_bridges
            self.finger_bridge_report = add_finger_bridges(spec)
        spec.geom("floor").priority = c.table_priority
        spec.option.timestep = self.sim_dt
        spec.option.iterations = c.solver_iterations
        spec.option.ls_iterations = c.solver_ls_iterations
        spec.body("link6").add_site(
            name=TCP_SITE, pos=(robot["tcp"] if c.tcp_local_z == 0 else (0., 0., c.tcp_local_z)), size=[0.008, 0.008, 0.008],
            rgba=[0.1, 0.5, 1.0, 1.0], type=mujoco.mjtGeom.mjGEOM_SPHERE,
        )
        self._add_block(spec)
        self._add_bowl(spec)
        self._configure_task_spec(spec)
        self._set_collision_bits(spec, c.bowl_arm_collision, c.bowl_free)
        gripper = spec.actuator("gripper")
        gripper.gainprm[0] = c.gripper_kp
        gripper.biasprm[1] = -c.gripper_kp
        gripper.forcerange[:] = [-c.gripper_force, c.gripper_force]
        self._mj_model = spec.compile()
        self._mjx_model = mjx_env.put_model(self._mj_model, impl=c.impl)
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
        self._link6_id = model.body("link6").id
        self._block_id = model.body(BLOCK_BODY).id
        self._block_geom_id = int(model.body_geomadr[self._block_id])
        self._floor_geom_id = int(model.geom("floor").id)
        self._gripper_dof_ids = np.array([model.jnt_dofadr[model.joint(n).id] for n in ("joint7", "joint8")])
        block_joint = model.body_jntadr[self._block_id]
        self._block_qpos_adr = int(model.jnt_qposadr[block_joint])
        self._block_qvel_adr = int(model.jnt_dofadr[block_joint])
        self._bowl_id = model.body(BOWL_BODY).id
        self._bowl_mocap_id = int(model.body_mocapid[self._bowl_id])
        if c.bowl_free:
            bowl_joint = model.body_jntadr[self._bowl_id]
            self._bowl_qpos_adr = int(model.jnt_qposadr[bowl_joint])

        joint_ranges = model.jnt_range[joints]
        ctrl_ranges = model.actuator_ctrlrange[self._arm_ctrl_ids]
        self._lower = jp.array(np.maximum(joint_ranges[:, 0], ctrl_ranges[:, 0]))
        self._upper = jp.array(np.minimum(joint_ranges[:, 1], ctrl_ranges[:, 1]))
        nominal = np.asarray(c.nominal_qpos, dtype=np.float32)
        if np.any(nominal < np.asarray(self._lower)) or np.any(nominal > np.asarray(self._upper)):
            raise ValueError("nominal_qpos must lie inside joint and actuator limits")
        gripper_limits = model.actuator_ctrlrange[self._gripper_ctrl_id]
        if not gripper_limits[0] <= c.gripper_max <= gripper_limits[1]:
            raise ValueError("gripper_max is outside the actuator range")
        self._nominal = jp.array(nominal)
        self._initial_qpos = jp.array(model.qpos0, dtype=jp.float32)
        self._initial_qpos = self._initial_qpos.at[self._qpos_ids].set(self._nominal)
        self._initial_qpos = self._initial_qpos.at[self._finger_qpos_ids].set(
            jp.array([c.gripper_max, -c.gripper_max], dtype=jp.float32)
        )
        self._initial_ctrl = jp.zeros(model.nu, dtype=jp.float32)
        self._initial_ctrl = self._initial_ctrl.at[self._arm_ctrl_ids].set(self._nominal)
        self._initial_ctrl = self._initial_ctrl.at[self._gripper_ctrl_id].set(c.gripper_max)
        self._bowl_rim_z = c.bowl_floor + c.bowl_height
        probs = np.zeros(getattr(self, 'reset_stage_count', 4), dtype=np.float32)
        probs[:len(c.reset_stage_probs)] = np.asarray(c.reset_stage_probs, dtype=np.float32)
        self._stage_probs = jp.array(probs / probs.sum())
        self._reset_table = {name: jp.array(v) for name, v in self._build_reset_table().items()}

    # ------------------------------------------------------------------ model
    def _add_block(self, spec: mujoco.MjSpec) -> None:
        c = self._config
        body = spec.worldbody.add_body(name=BLOCK_BODY, pos=[0.3, -0.1, c.block_half])
        body.add_freejoint()
        # priority 1: block contacts use the block's friction (MuJoCo otherwise
        # takes the larger of the two), so friction randomization of the block
        # reaches block-pad and block-table contacts. Nominal values are all 1.0.
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[c.block_half] * 3, mass=c.block_mass,
            rgba=list(c.block_rgba), condim=4, friction=[1.0, 0.005, 0.0001], priority=1,
        )

    def _add_bowl(self, spec: mujoco.MjSpec) -> None:
        """Floor disc plus a ring of leaning wall boxes (see assets/objects/block_bowl/make_bowl.py).

        The kinematic (mocap) bowl is lifted 2 mm above the table plane so it
        never generates bowl-table contacts. With ``bowl_free`` the bowl has a
        free joint, rests on the table and weighs ``bowl_mass`` (spread over the
        boxes by volume).
        """
        c = self._config
        if c.bowl_free:
            body = spec.worldbody.add_body(name=BOWL_BODY, pos=[0.3, 0.15, 0.0])
            body.add_freejoint()
        else:
            body = spec.worldbody.add_body(name=BOWL_BODY, pos=[0.3, 0.15, 0.002], mocap=True)
        rgba = [0.2, 0.4, 0.9, 1.0]
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[c.bowl_r_bottom + c.bowl_wall, c.bowl_r_bottom + c.bowl_wall, c.bowl_floor / 2],
            pos=[0.0, 0.0, c.bowl_floor / 2], rgba=rgba,
        )
        n = c.bowl_segments
        seg = 2 * math.pi / n
        tilt = math.atan2(c.bowl_r_rim - c.bowl_r_bottom, c.bowl_height)
        length = math.hypot(c.bowl_r_rim - c.bowl_r_bottom, c.bowl_height)
        half_w = (c.bowl_r_rim + c.bowl_wall) * math.tan(seg / 2) * 1.05
        z0 = c.bowl_floor + (c.bowl_wall / 2) * math.sin(tilt)
        r0 = c.bowl_r_bottom + (c.bowl_wall / 2) / math.cos(tilt)
        r_mid = r0 + (length / 2) * math.sin(tilt)
        z_mid = z0 + (length / 2) * math.cos(tilt)
        for i in range(n):
            a = seg * i
            cz, sz, cy, sy = math.cos(a / 2), math.sin(a / 2), math.cos(tilt / 2), math.sin(tilt / 2)
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[c.bowl_wall / 2, half_w, length / 2],
                pos=[r_mid * math.cos(a), r_mid * math.sin(a), z_mid],
                quat=[cz * cy, -sz * sy, cz * sy, sz * cy], rgba=rgba,
            )
        if c.bowl_free:
            geoms = [g for g in spec.geoms if g.parent is not None and g.parent.name == BOWL_BODY]
            volumes = [float(np.prod(g.size[:3])) for g in geoms]
            for g, v in zip(geoms, volumes):
                g.mass = c.bowl_mass * v / sum(volumes)

    @staticmethod
    def _set_collision_bits(spec: mujoco.MjSpec, bowl_arm: bool = False, bowl_free: bool = False) -> None:
        """Limit collision pairs: bit 1 = table/arm, bit 2 = bowl.

        The bowl (2/2) meets only geoms carrying bit 2: the block (3/3) and the
        finger pads (3/3). Arm capsules and the table plane keep 1/1, so the
        kinematic bowl never pairs with the table or the arm links. Fewer
        convex pairs keep the MJX step cheap; the arm passing through the bowl
        wall is the v0 limitation.

        ``bowl_arm=True`` gives the bowl conaffinity 3, so it also meets the arm
        capsules (bit 1). Bowl-floor and bowl-base pairs are excluded: the bowl
        is kinematic and lifted 2 mm, and the base cannot reach it.

        ``bowl_free=True`` (with ``bowl_arm``) keeps the bowl-table pair so the
        free bowl rests on the table; only the base exclusion remains.
        """
        for geom in spec.geoms:
            if geom.contype == 0 and geom.conaffinity == 0:
                continue                                    # visual meshes
            parent = geom.parent.name if geom.parent is not None else ""
            if parent == BOWL_BODY:
                geom.contype, geom.conaffinity = 2, (3 if bowl_arm else 2)
            elif parent in (BLOCK_BODY, "link7", "link8"):
                geom.contype, geom.conaffinity = 3, 3
        if bowl_arm:
            for other in (("base_link",) if bowl_free else ("world", "base_link")):
                spec.add_exclude(bodyname1=other, bodyname2=BOWL_BODY)

    def _build_reset_table(self) -> dict[str, np.ndarray]:
        """Top-down arm configurations over the block region for stage-wise resets.

        Solved with MuJoCo's own Jacobian (CPU) at construction: for each grid
        point, yaw and height, damped least-squares IK drives the grasp site to
        the target with the gripper pointing at the table. Configurations that
        do not converge are dropped.
        """
        c = self._config
        model = self._mj_model
        data = mujoco.MjData(model)
        lower, upper = np.asarray(self._lower), np.asarray(self._upper)
        # (height, x range, y range) per start stage; pre_inside hovers over the bowl region
        # with the block bottom preinside_height above the rim.
        targets = {
            "pregrasp": (c.block_half + c.pregrasp_height, c.block_x, c.block_y),
            "grasp": (c.lift_reset_height, c.block_x, c.block_y),
            "preinside": (0.002 + self._bowl_rim_z + c.block_half + c.preinside_height, c.bowl_x, c.bowl_y),
        }
        yaws = np.linspace(-np.pi / 2, np.pi / 2, c.reset_table_yaws, endpoint=False)
        flip = np.diag([-1.0, -1.0, 1.0])
        seeds = [np.asarray(self._nominal, dtype=np.float64), np.asarray(c.reset_ik_seed, dtype=np.float64)]
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        table = {name: [] for name in targets}
        for name, (z, x_range, y_range) in targets.items():
            for x in np.linspace(x_range[0], x_range[1], c.reset_table_grid):
                for y in np.linspace(y_range[0], y_range[1], c.reset_table_grid):
                    for yaw in yaws:
                        rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
                        desired = rot @ np.diag([1.0, -1.0, -1.0])            # z axis down, yaw about z
                        # A two-finger grasp is the same turned 180 degrees about the
                        # vertical; try both (reset_ik_flip), from the start pose and a top-down seed.
                        goals = (desired, flip @ desired) if c.reset_ik_flip else (desired,)
                        for goal, seed in itertools.product(goals, seeds):
                            q = seed.copy()
                            for _ in range(200):
                                data.qpos[:] = model.qpos0
                                data.qpos[self._qpos_ids] = q
                                mujoco.mj_kinematics(model, data)
                                mujoco.mj_comPos(model, data)
                                current = data.site_xmat[self._tcp_id].reshape(3, 3)
                                err_pos = np.array([x, y, z]) - data.site_xpos[self._tcp_id]
                                err_rot = 0.5 * sum(np.cross(current[:, i], goal[:, i]) for i in range(3))
                                err = np.concatenate((err_pos, err_rot))
                                if np.linalg.norm(err_pos) < 1e-4 and np.linalg.norm(err_rot) < 1e-3:
                                    break
                                mujoco.mj_jacSite(model, data, jacp, jacr, self._tcp_id)
                                jac = np.concatenate((jacp, jacr))[:, self._qvel_ids]
                                q = np.clip(q + 0.5 * (jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)), lower, upper)
                            if np.linalg.norm(err_pos) < 2e-3 and np.linalg.norm(err_rot) < 2e-2:
                                table[name].append(q.astype(np.float32))
                                break
        for name, rows in table.items():
            if not rows:
                raise ValueError(f"no reachable {name} reset configuration; check the placement ranges and heights")
        return {name: np.stack(rows) for name, rows in table.items()}

    def _validate_config(self) -> None:
        c = self._config
        if c.robot_model not in ROBOT_MODELS:
            raise ValueError(f"robot_model must be one of {sorted(ROBOT_MODELS)}")
        for name in ("base_height", "spawn_min_distance", "spawn_max_radius"):
            if not np.isfinite(c[name]) or c[name] < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(c.table_priority, bool) or not isinstance(c.table_priority, int) or c.table_priority < 0:
            raise ValueError("table_priority must be a nonnegative integer")
        if c.bowl_free and not c.bowl_arm_collision:
            raise ValueError("bowl_free requires bowl_arm_collision")
        if not c.bowl_mass > 0:
            raise ValueError("bowl_mass must be positive")
        for name in (
            "solver_iterations", "solver_ls_iterations", "batch_capacity",
            "contacts_per_env", "max_constraints_per_env", "ccd_pairs_per_env", "bowl_segments",
            "spawn_candidates",
        ):
            if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(c.ctrl_dt) or not np.isfinite(c.sim_dt) or min(c.ctrl_dt, c.sim_dt) <= 0:
            raise ValueError("ctrl_dt and sim_dt must be positive and finite")
        if not np.isclose(c.ctrl_dt / c.sim_dt, round(c.ctrl_dt / c.sim_dt)):
            raise ValueError("ctrl_dt must be an integer multiple of sim_dt")
        value = np.asarray(c.nominal_qpos, dtype=float)
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise ValueError("nominal_qpos must contain six finite values")
        for name in ("block_x", "block_y", "bowl_x", "bowl_y"):
            lo, hi = np.asarray(c[name], dtype=float)
            if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
                raise ValueError(f"{name} must be a finite [low, high] range")
        for name in (
            "block_half", "block_mass", "bowl_r_bottom", "bowl_r_rim", "bowl_height", "bowl_wall",
            "bowl_floor", "max_lin_vel", "max_ang_vel", "gripper_max", "gripper_kp", "gripper_rate", "gripper_force", "ik_damping", "max_joint_vel", "max_target_lag",
            "reach_scale", "grasp_distance", "lift_height", "inside_reward", "release_bonus", "preinside_height",
        ):
            if not np.isfinite(c[name]) or c[name] <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if c.bowl_r_rim < c.bowl_r_bottom:
            raise ValueError("bowl_r_rim must be at least bowl_r_bottom")
        # The closed predicate (width < 2 * block_half + 4 mm) must be reachable
        # from the open gripper, and the inside predicate needs room on the floor.
        if 2 * c.block_half + 0.004 >= 2 * c.gripper_max:
            raise ValueError("block is too wide for the gripper opening (2 * gripper_max)")
        if c.block_half >= c.bowl_r_bottom:
            raise ValueError("block_half must be smaller than bowl_r_bottom")
        rgba = np.asarray(c.block_rgba, dtype=float)
        if rgba.shape != (4,) or not np.isfinite(rgba).all() or np.any(rgba < 0) or np.any(rgba > 1):
            raise ValueError("block_rgba must be four values in [0, 1]")
        probs = np.asarray(c.reset_stage_probs, dtype=float)
        allowed = ((3,), (4,), (5,)) if getattr(self, 'reset_stage_count', 4) == 5 else ((3,), (4,))
        if probs.shape not in allowed or np.any(probs < 0) or not np.isfinite(probs).all() or probs.sum() <= 0:
            raise ValueError(f"reset_stage_probs must have one of {[s[0] for s in allowed]} entries, nonnegative with positive sum")
        for name in ("reset_table_grid", "reset_table_yaws"):
            if isinstance(c[name], bool) or not isinstance(c[name], int) or c[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("pregrasp_height", "lift_reset_height", "preinside_height"):
            if not np.isfinite(c[name]) or c[name] <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if c.jacobian not in ("mjx", "fd"):
            raise ValueError("jacobian must be 'mjx' or 'fd'")
        for name in ("reset_joint_noise", "action_cost", "action_rate_cost",
                     "obj_pos_noise", "obj_pos_noise_cap", "static_pos_noise", "static_pos_noise_cap",
                     "obj_rot_noise", "size_noise", "obj_pose_delay_max", "bowl_pose_delay_max", "robot_state_delay_max",
                     "action_delay_max", "block_friction_scale"):
            if not np.isfinite(c[name]) or c[name] < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if c.size_noise >= 1.0 or c.block_friction_scale >= 1.0:
            raise ValueError("size_noise and block_friction_scale must be below 1")
        if c.obj_pose_delay_max > 2 * c.ctrl_dt or c.robot_state_delay_max > c.ctrl_dt:
            raise ValueError("observation delays are limited to 2 (block) / 1 (robot) control steps")
        if c.bowl_pose_delay_max > 3 * c.ctrl_dt:
            raise ValueError("bowl observation delay is limited to 3 control steps")
        if c.action_delay_max > c.ctrl_dt:
            raise ValueError("action_delay_max must not exceed ctrl_dt")
        for name in ("block_mass_range", "table_friction_range", "gripper_gain_scale"):
            lo, hi = np.asarray(c[name], dtype=float)
            if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo <= hi):
                raise ValueError(f"{name} must be a positive [low, high] range")

    # ---------------------------------------------------------------- physics
    def _make_data_kwargs(self) -> dict[str, int]:
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
        bits = jp.asarray(data._impl.overflow, dtype=jp.int32)
        return jp.any(jp.bitwise_and(bits, ~_SOLVER_LIMIT_BITS) != 0)

    def _step_physics(
        self, data: mjx.Data, ctrl: jax.Array, hold_substeps: jax.Array | None = None,
    ) -> tuple[mjx.Data, jax.Array]:
        """Advance one control step. The first ``hold_substeps`` substeps keep the
        previous command (``data.ctrl``): the action-execution delay."""
        if hold_substeps is None:
            if self._config.impl != "warp":
                return mjx_env.step(self._mjx_model, data, ctrl, self.n_substeps), jp.array(False)
            hold_substeps = jp.zeros((), dtype=jp.int32)
        previous = data.ctrl

        def substep(carry, i):
            current, overflow = carry
            applied = jp.where(i < hold_substeps, previous, ctrl)
            current = mjx.step(self._mjx_model, current.replace(ctrl=applied))
            return (current, overflow | self._capacity_overflow(current)), None

        data, overflow = jax.lax.scan(
            substep, (data, self._capacity_overflow(data)), jp.arange(self.n_substeps),
        )[0]
        return data.replace(ctrl=ctrl), overflow

    # --------------------------------------------------------------------- IK
    def _tcp_jacobian(self, data: mjx.Data) -> jax.Array:
        """(6, 6) Jacobian of the TCP site w.r.t. the arm joints: rows = linear xyz, angular xyz."""
        if self._config.jacobian == "mjx":
            jacp, jacr = mjx_support.jac(
                self._mjx_model, data, data.site_xpos[self._tcp_id], self._link6_id,
            )
            jac = jp.concatenate((jacp.T, jacr.T), axis=0)          # (6, nv)
            return jac[:, self._qvel_ids]
        return self._fd_jacobian(data)

    def _fd_jacobian(self, data: mjx.Data, eps: float = 1e-4) -> jax.Array:
        base_pos = data.site_xpos[self._tcp_id]
        base_mat = data.site_xmat[self._tcp_id]
        qpos_ids = jp.asarray(self._qpos_ids)

        def column(i):
            qpos = data.qpos.at[qpos_ids[i]].add(eps)
            moved = mjx.kinematics(self._mjx_model, data.replace(qpos=qpos))
            dpos = (moved.site_xpos[self._tcp_id] - base_pos) / eps
            # First-order rotation vector of the small relative rotation R_new R_old^T
            # (its skew part); the quaternion route loses precision in float32 here.
            rel = moved.site_xmat[self._tcp_id] @ base_mat.T
            drot = jp.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / (2.0 * eps)
            return jp.concatenate((dpos, drot))

        return jax.vmap(column)(jp.arange(6)).T                      # (6, 6)

    def _joint_targets(self, data: mjx.Data, action: jax.Array) -> jax.Array:
        c = self._config
        twist = jp.concatenate((action[:3] * c.max_lin_vel, action[3:6] * c.max_ang_vel))
        jac = self._tcp_jacobian(data)
        gram = jac @ jac.T + (c.ik_damping ** 2) * jp.eye(6, dtype=jac.dtype)
        qvel = jac.T @ jp.linalg.solve(gram, twist)
        qvel = jp.clip(qvel, -c.max_joint_vel, c.max_joint_vel)
        # Accumulate on the actuator target (as the reach task does) so the
        # position servos keep pulling toward the commanded motion, but keep the
        # target within max_target_lag of the measured joints to avoid wind-up
        # when the arm is blocked by contact.
        qpos = data.qpos[self._qpos_ids]
        target = data.ctrl[self._arm_ctrl_ids] + qvel * c.ctrl_dt
        target = jp.clip(target, qpos - c.max_target_lag, qpos + c.max_target_lag)
        return jp.clip(target, self._lower, self._upper)

    def _bowl_pos(self, data: mjx.Data) -> jax.Array:
        if self._config.bowl_free:
            return data.xpos[self._bowl_id]
        return data.mocap_pos[self._bowl_mocap_id]

    # ------------------------------------------------------------- predicates
    def _predicates(self, data: mjx.Data) -> dict[str, jax.Array]:
        c = self._config
        tcp = data.site_xpos[self._tcp_id]
        block = data.xpos[self._block_id]
        bowl = self._bowl_pos(data)
        width = data.qpos[self._finger_qpos_ids[0]] - data.qpos[self._finger_qpos_ids[1]]
        tcp_block = jp.linalg.norm(block - tcp)
        block_bowl_xy = jp.linalg.norm((block - bowl)[:2])
        height = block[2] - c.block_half
        near = tcp_block < c.grasp_distance
        closed = width < 2 * c.block_half + 0.004
        lifted = height > c.lift_height
        grasped = near & closed & lifted
        above = grasped & (block_bowl_xy < c.bowl_r_rim - c.block_half) & (block[2] > bowl[2] + self._bowl_rim_z)
        placed = (
            (block_bowl_xy < c.bowl_r_bottom - c.block_half)
            & (block[2] < bowl[2] + self._bowl_rim_z)
            & (block[2] > bowl[2])
        )
        in_bowl = placed & ~near                                   # released: the success test
        return {
            "tcp_block": tcp_block, "block_bowl_xy": block_bowl_xy, "height": height, "width": width,
            "pre_grasp": near, "grasp": grasped, "pre_inside": above, "placed": placed, "inside": in_bowl,
        }

    @staticmethod
    def _stage(p: dict[str, jax.Array]) -> jax.Array:
        # Unchanged from v0 so policies trained before the placed/released split keep their stage input.
        return jp.where(p["inside"], 3, jp.where(p["pre_inside"], 2, jp.where(p["grasp"], 1, 0)))

    # ------------------------------------------------------------------- env
    def reset(self, rng: jax.Array) -> mjx_env.State:
        c = self._config
        rng, stage_key, joint_key, block_key, yaw_key, bowl_key, pre_key, grasp_key, inside_key, offset_key = jax.random.split(rng, 10)
        stage = jax.random.choice(stage_key, 4, p=self._stage_probs)
        noise = jax.random.uniform(joint_key, (6,), minval=-1.0, maxval=1.0) * c.reset_joint_noise
        pre_table, grasp_table, inside_table = (self._reset_table[k] for k in ("pregrasp", "grasp", "preinside"))
        pre_joints = pre_table[jax.random.randint(pre_key, (), 0, pre_table.shape[0])]
        grasp_joints = grasp_table[jax.random.randint(grasp_key, (), 0, grasp_table.shape[0])]
        inside_joints = inside_table[jax.random.randint(inside_key, (), 0, inside_table.shape[0])]
        arm_joints = jp.where(stage == 0, self._nominal, jp.where(stage == 1, pre_joints,
                              jp.where(stage == 2, grasp_joints, inside_joints)))
        arm_joints = jp.clip(arm_joints + noise, self._lower, self._upper)
        # Forward kinematics of the arm alone gives the grasp site pose that the
        # pre-grasp and grasp starts place the block against.
        arm_qpos = self._initial_qpos.at[self._qpos_ids].set(arm_joints)
        kin = mjx.kinematics(self._mjx_model, mjx_env.make_data(
            self._mj_model, qpos=arm_qpos, qvel=jp.zeros(self._mj_model.nv, dtype=jp.float32),
            ctrl=self._initial_ctrl, impl=c.impl, **self._make_data_kwargs(),
        ))
        tcp = kin.site_xpos[self._tcp_id]
        closing_axis = kin.site_xmat[self._tcp_id][:, self._closing_axis]   # finger closing direction
        gripper_yaw = jp.arctan2(closing_axis[1], closing_axis[0])
        # v0 (no reach limit, no minimum distance) draws one sample each, so old runs replay exactly.
        v1_spawn = c.spawn_max_radius > 0 or c.spawn_min_distance > 0
        n_draw = c.spawn_candidates if v1_spawn else 1
        block_candidates = jax.random.uniform(
            block_key, (n_draw, 2), minval=jp.array([c.block_x[0], c.block_y[0]]),
            maxval=jp.array([c.block_x[1], c.block_y[1]]),
        ) if v1_spawn else jax.random.uniform(
            block_key, (2,), minval=jp.array([c.block_x[0], c.block_y[0]]),
            maxval=jp.array([c.block_x[1], c.block_y[1]]),
        )[None]
        reach = jp.float32(c.spawn_max_radius if c.spawn_max_radius > 0 else jp.inf)
        block_r = jp.linalg.norm(block_candidates, axis=1)
        free_xy = block_candidates[jp.where(jp.any(block_r <= reach), jp.argmax(block_r <= reach), jp.argmin(block_r))]
        free_yaw = jax.random.uniform(yaw_key, (), minval=-jp.pi, maxval=jp.pi)
        block_pos = jp.where(
            stage == 0, jp.concatenate((free_xy, jp.array([c.block_half]))),
            jp.where(stage == 1, jp.array([tcp[0], tcp[1], c.block_half]), tcp),
        )
        yaw = jp.where(stage == 0, free_yaw, gripper_yaw)
        block_qpos = jp.concatenate((block_pos, jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])))
        held = stage >= 2
        finger = jp.where(held, c.block_half, c.gripper_max)
        gripper_ctrl = jp.where(held, 0.0, c.gripper_max)
        candidates = jax.random.uniform(
            bowl_key, (n_draw, 2), minval=jp.array([c.bowl_x[0], c.bowl_y[0]]),
            maxval=jp.array([c.bowl_x[1], c.bowl_y[1]]),
        ) if v1_spawn else jax.random.uniform(
            bowl_key, (2,), minval=jp.array([c.bowl_x[0], c.bowl_y[0]]),
            maxval=jp.array([c.bowl_x[1], c.bowl_y[1]]),
        )[None]
        gap = jp.linalg.norm(candidates - block_pos[:2], axis=1)
        in_reach = jp.linalg.norm(candidates, axis=1) <= reach
        far_enough = (gap >= c.spawn_min_distance) & in_reach
        pick = jp.where(jp.any(far_enough), jp.argmax(far_enough), jp.argmax(jp.where(in_reach, gap, -1.0)))
        bowl_xy = candidates[pick]
        # pre_inside start: the bowl sits under the held block, offset by less
        # than the clearance to the rim so the block is above the opening.
        clearance = jp.maximum(c.bowl_r_rim - c.block_half - 0.01, 0.0)
        offset = jax.random.uniform(offset_key, (2,), minval=-clearance, maxval=clearance) / jp.sqrt(2.0)
        bowl_xy = jp.where(stage == 3, tcp[:2] + offset, bowl_xy)
        qpos = arm_qpos.at[self._finger_qpos_ids].set(jp.array([finger, -finger]))
        qpos = jax.lax.dynamic_update_slice(qpos, block_qpos.astype(qpos.dtype), (self._block_qpos_adr,))
        if c.bowl_free:
            bowl_qpos = jp.concatenate((bowl_xy, jp.array([0.0, 1.0, 0.0, 0.0, 0.0])))
            qpos = jax.lax.dynamic_update_slice(qpos, bowl_qpos.astype(qpos.dtype), (self._bowl_qpos_adr,))
        ctrl = self._initial_ctrl.at[self._arm_ctrl_ids].set(arm_joints).at[self._gripper_ctrl_id].set(gripper_ctrl)
        data = mjx_env.make_data(
            self._mj_model, qpos=qpos, qvel=jp.zeros(self._mj_model.nv, dtype=jp.float32),
            ctrl=ctrl, impl=c.impl, **self._make_data_kwargs(),
        )
        if not c.bowl_free:
            data = data.replace(mocap_pos=data.mocap_pos.at[self._bowl_mocap_id].set(
                jp.concatenate((bowl_xy, jp.array([0.002]))).astype(data.mocap_pos.dtype)
            ))
        data = mjx.forward(self._mjx_model, data)
        capacity_overflow = self._capacity_overflow(data)
        p = self._predicates(data)
        rng, percep_key, obs_key = jax.random.split(rng, 3)
        percep = self._reset_perception(data, percep_key)
        zero = jp.zeros((), dtype=jp.float32)
        metrics = {
            "tcp_block": p["tcp_block"], "block_bowl_xy": p["block_bowl_xy"], "block_height": p["height"],
            "stage": self._stage(p).astype(jp.float32),
            "grasped": p["grasp"].astype(jp.float32),
            "success": (p["inside"] & ~capacity_overflow).astype(jp.float32),
            "reward_reach": zero, "reward_lift": zero, "reward_carry": zero, "reward_inside": zero,
            "action_penalty": zero, "action_rate_penalty": zero,
            "nonfinite": zero, "capacity_overflow": capacity_overflow.astype(jp.float32),
        }
        return mjx_env.State(
            data=data, obs=self._get_obs(data, jp.zeros(ACTION_SIZE, dtype=jp.float32), p, percep, obs_key)[0],
            reward=zero, done=capacity_overflow.astype(jp.float32), metrics=metrics,
            info={"rng": rng, "reset_stage": stage, "perception": percep},
        )

    def _configure_task_spec(self, spec):
        """Optional sensors; base task geometry and dynamics are unchanged."""

    def _gripper_command(self, data, action, info):
        """Original binary/continuous mapping, unchanged for existing configs."""
        c = self._config
        if c.gripper_binary:
            desired = jp.where(action[6] > 0.0, c.gripper_max, 0.0)
        else:
            desired = c.gripper_max * (action[6] + 1.0) / 2.0
        previous = data.ctrl[self._gripper_ctrl_id]
        return jp.clip(desired, previous-c.gripper_rate, previous+c.gripper_rate), {}

    def _advance_control(self, data, ctrl, action, hold, info):
        data, overflow = self._step_physics(data, ctrl, hold)
        return data, overflow, {}

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        c = self._config
        action = jp.asarray(action, dtype=jp.float32)
        if action.shape != (ACTION_SIZE,):
            raise ValueError(f"expected action shape ({ACTION_SIZE},), got {action.shape}")
        nonfinite = ~jp.all(jp.isfinite(action))
        for field in (state.data.qpos, state.data.qvel, state.data.ctrl):
            nonfinite |= ~jp.all(jp.isfinite(field))
        action = jp.clip(jp.nan_to_num(action), -1.0, 1.0)
        safe_data = state.data.replace(
            qpos=jp.where(jp.isfinite(state.data.qpos), state.data.qpos, self._initial_qpos),
            qvel=jp.nan_to_num(state.data.qvel, posinf=0.0, neginf=0.0),
            ctrl=jp.where(jp.isfinite(state.data.ctrl), state.data.ctrl, self._initial_ctrl),
        )
        joint_targets = self._joint_targets(safe_data, action)
        gripper_target, gripper_info = self._gripper_command(safe_data, action, state.info)
        ctrl = safe_data.ctrl.at[self._arm_ctrl_ids].set(joint_targets)
        ctrl = ctrl.at[self._gripper_ctrl_id].set(gripper_target)
        percep = state.info["perception"]
        rng, obs_key = jax.random.split(state.info["rng"])
        hold = percep["hold_substeps"] if c.obs_randomization else None
        data, capacity_overflow, control_info = self._advance_control(safe_data, ctrl, action, hold, state.info)
        data = mjx.kinematics(self._mjx_model, data)
        p = self._predicates(data)
        nonfinite |= ~jp.all(jp.isfinite(data.qpos)) | ~jp.all(jp.isfinite(data.qvel))
        nonfinite |= ~jp.isfinite(p["tcp_block"]) | ~jp.isfinite(p["block_bowl_xy"])
        invalid = nonfinite | capacity_overflow

        # Stage-shaped reward v0 (SHOW-49 replaces this with the scene-graph version).
        reach = 1.0 - jp.tanh(c.reach_scale * p["tcp_block"])
        lift = p["grasp"].astype(jp.float32)
        carry = lift * (1.0 - jp.tanh(c.reach_scale * p["block_bowl_xy"]))
        inside = c.inside_reward * p["placed"].astype(jp.float32) + c.release_bonus * p["inside"].astype(jp.float32)
        action_penalty = c.action_cost * jp.sum(jp.square(action))
        previous_action = jp.nan_to_num(state.obs["state"][-ACTION_SIZE:])
        rate_penalty = c.action_rate_cost * jp.sum(jp.square(action - previous_action))
        reward = jp.where(invalid, -1.0, reach + lift + carry + inside - action_penalty - rate_penalty)

        metrics = dict(state.metrics)
        metrics.update({
            "tcp_block": jp.nan_to_num(p["tcp_block"]), "block_bowl_xy": jp.nan_to_num(p["block_bowl_xy"]),
            "block_height": jp.nan_to_num(p["height"]),
            "stage": self._stage(p).astype(jp.float32),
            "grasped": p["grasp"].astype(jp.float32),
            "success": (p["inside"] & ~invalid).astype(jp.float32),
            "reward_reach": jp.where(invalid, 0.0, reach), "reward_lift": lift, "reward_carry": carry,
            "reward_inside": inside, "action_penalty": action_penalty, "action_rate_penalty": rate_penalty,
            "nonfinite": nonfinite.astype(jp.float32), "capacity_overflow": capacity_overflow.astype(jp.float32),
        })
        obs, percep = self._get_obs(data, action, p, percep, obs_key, gripper_goal=gripper_info.get("gripper_goal"))
        info = dict(state.info, rng=rng, perception=percep, **control_info, **gripper_info)
        return state.replace(
            data=data, obs=obs, reward=reward,
            done=invalid.astype(jp.float32), metrics=metrics, info=info,
        )

    # ------------------------------------------------------------ observation
    def _robot_features(self, data: mjx.Data) -> jax.Array:
        """Proprioception (25): arm qpos, qvel, target-minus-qpos, TCP position and rotation, gripper width."""
        width = data.qpos[self._finger_qpos_ids[0]] - data.qpos[self._finger_qpos_ids[1]]
        return jp.concatenate((
            data.qpos[self._qpos_ids], data.qvel[self._qvel_ids],
            data.ctrl[self._arm_ctrl_ids] - data.qpos[self._qpos_ids],
            # The TCP site has no local rotation, so its orientation is link6's.
            data.site_xpos[self._tcp_id] - self._base_offset, _quat_to_rotvec(data.xquat[self._link6_id]), width[None],
        ))

    def _block_pose(self, data: mjx.Data) -> jax.Array:
        return jp.concatenate((data.xpos[self._block_id], data.xquat[self._block_id]))

    def _reset_perception(self, data: mjx.Data, key: jax.Array) -> dict[str, jax.Array]:
        """Per-episode perception state: pose history for the delays and the episode's draws.

        Without ``obs_randomization`` the draws are neutral (no delay, no bowl
        offset, size scale 1), so the observation is the true state.
        """
        c = self._config
        k_obj, k_rob, k_act, k_size, k_bowl = jax.random.split(key, 5)
        on = jp.float32(1.0 if c.obs_randomization else 0.0)
        u = lambda k, hi: on * jax.random.uniform(k, (), minval=0.0, maxval=hi)
        bowl_offset = jp.clip(
            jax.random.normal(k_bowl, (3,)) * c.static_pos_noise, -c.static_pos_noise_cap, c.static_pos_noise_cap,
        )
        pose = self._block_pose(data)
        perception = {
            "block_hist": jp.stack([pose, pose, pose]),            # t, t-1, t-2 (true)
            "robot_prev": self._robot_features(data),              # t-1 (true)
            "obj_delay": u(k_obj, c.obj_pose_delay_max / c.ctrl_dt),        # control steps
            "robot_delay": u(k_rob, c.robot_state_delay_max / c.ctrl_dt),   # control steps
            "hold_substeps": jp.round(u(k_act, c.action_delay_max / self.sim_dt)).astype(jp.int32),
            "size_scale": 1.0 + on * jax.random.uniform(k_size, (), minval=-c.size_noise, maxval=c.size_noise),
            "bowl_offset": on * bowl_offset,
        }
        if c.bowl_pose_delay_max > 0:
            steps = int(np.ceil(c.bowl_pose_delay_max/c.ctrl_dt))
            perception.update(bowl_hist=jp.tile(self._bowl_pos(data)[None], (steps+1, 1)),
                bowl_delay=u(jax.random.fold_in(key, 108), c.bowl_pose_delay_max/c.ctrl_dt))
        return perception

    def _get_obs(
        self, data: mjx.Data, previous_action: jax.Array, p: dict[str, jax.Array],
        percep: dict[str, jax.Array], key: jax.Array, gripper_goal=None,
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        c = self._config
        robot_now = self._robot_features(data)
        pose_now = self._block_pose(data)
        hist = jp.concatenate((pose_now[None], percep["block_hist"][:2]))
        # Delays: interpolate between stored control steps (sub-step latency).
        robot = robot_now + jp.minimum(percep["robot_delay"], 1.0) * (percep["robot_prev"] - robot_now)
        d = percep["obj_delay"]
        i = jp.clip(jp.floor(d), 0, 1).astype(jp.int32)
        f = jp.clip(d - i, 0.0, 1.0)
        a, b = hist[i], hist[i + 1]
        block = a[:3] + f * (b[:3] - a[:3])
        qb = b[3:] * jp.where(jp.dot(a[3:], b[3:]) < 0, -1.0, 1.0)
        quat = a[3:] + f * (qb - a[3:])
        quat = quat / jp.maximum(jp.linalg.norm(quat), 1e-8)
        bowl = self._bowl_pos(data)
        if c.bowl_pose_delay_max > 0:
            steps = int(np.ceil(c.bowl_pose_delay_max/c.ctrl_dt))
            bowl_hist = jp.concatenate((bowl[None], percep['bowl_hist'][:steps]))
            delay = percep['bowl_delay']
            index = jp.clip(jp.floor(delay), 0, steps-1).astype(jp.int32)
            fraction = jp.clip(delay-index, 0., 1.)
            bowl = bowl_hist[index]+fraction*(bowl_hist[index+1]-bowl_hist[index])
        bowl = bowl + percep["bowl_offset"] - self._base_offset
        block = block - self._base_offset
        if c.obs_randomization:
            k_pos, k_axis, k_angle = jax.random.split(key, 3)
            block = block + jp.clip(jax.random.normal(k_pos, (3,)) * c.obj_pos_noise, -c.obj_pos_noise_cap, c.obj_pos_noise_cap)
            axis = jax.random.normal(k_axis, (3,))
            axis = axis / jp.maximum(jp.linalg.norm(axis), 1e-8)
            angle = jax.random.uniform(k_angle, (), minval=-1.0, maxval=1.0) * jp.deg2rad(c.obj_rot_noise)
            quat = _quat_mul(jp.concatenate((jp.cos(angle / 2)[None], axis * jp.sin(angle / 2))), quat)
        tcp = robot[18:21]
        parts = [robot, block, _quat_to_rotvec(quat), bowl, block - tcp, block - bowl]
        if c.stage_obs:
            parts.append(jax.nn.one_hot(self._stage(p), N_STAGES, dtype=jp.float32))
        if c.size_obs:
            parts.append((c.block_half * percep["size_scale"])[None])
        if c.gripper_goal_obs:
            goal = data.ctrl[self._gripper_ctrl_id] if gripper_goal is None else gripper_goal
            parts.append((goal / c.gripper_max)[None])
        state = jp.concatenate(parts + [previous_action])
        flags = jp.stack([p["pre_grasp"], p["grasp"], p["pre_inside"], p["inside"]]).astype(jp.float32)
        block_vel = data.qvel[self._block_qvel_adr:self._block_qvel_adr + 3]
        privileged = [state, block_vel, flags]
        if c.size_obs:
            privileged += [pose_now[:3] - self._base_offset, _quat_to_rotvec(pose_now[3:]),
                           self._bowl_pos(data) - self._base_offset,
                           jp.array([c.block_half], dtype=jp.float32)]
        privileged = jp.concatenate(privileged)
        percep = dict(percep, block_hist=hist, robot_prev=robot_now)
        if c.bowl_pose_delay_max > 0:
            percep['bowl_hist'] = bowl_hist
        clean = lambda x: jp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return {"state": clean(state), "privileged": clean(privileged)}, percep

    # ------------------------------------------------- physics randomization
    def physics_randomization_fn(self, model: mjx.Model, rng: jax.Array):
        """Per-slot model parameters for Playground's ``BraxDomainRandomizationVmapWrapper``.

        ``rng`` has one key per environment slot (Brax passes ``num_envs`` keys);
        each slot keeps its draw for the whole run. Varies block mass (inertia
        scaled with it), block friction, table friction, and the gripper servo's
        stiffness and damping (Table 10's hand stiffness/damping).
        """
        c = self._config
        bid, gid, fid, aid = self._block_id, self._block_geom_id, self._floor_geom_id, self._gripper_ctrl_id

        @jax.vmap
        def draw(key):
            k_mass, k_fric, k_table, k_kp, k_kv = jax.random.split(key, 5)
            mass = jax.random.uniform(k_mass, (), minval=c.block_mass_range[0], maxval=c.block_mass_range[1])
            body_mass = model.body_mass.at[bid].set(mass)
            body_inertia = model.body_inertia.at[bid].set(model.body_inertia[bid] * mass / model.body_mass[bid])
            fric = jax.random.uniform(k_fric, (), minval=1 - c.block_friction_scale, maxval=1 + c.block_friction_scale)
            table = jax.random.uniform(k_table, (), minval=c.table_friction_range[0], maxval=c.table_friction_range[1])
            geom_friction = model.geom_friction.at[gid, 0].multiply(fric).at[fid, 0].set(table)
            kp = jax.random.uniform(k_kp, (), minval=c.gripper_gain_scale[0], maxval=c.gripper_gain_scale[1])
            kv = jax.random.uniform(k_kv, (), minval=c.gripper_gain_scale[0], maxval=c.gripper_gain_scale[1])
            gainprm = model.actuator_gainprm.at[aid, 0].multiply(kp)
            biasprm = model.actuator_biasprm.at[aid, 1].multiply(kp).at[aid, 2].multiply(kv)
            return body_mass, body_inertia, geom_friction, gainprm, biasprm

        body_mass, body_inertia, geom_friction, gainprm, biasprm = draw(rng)
        fields = {
            "body_mass": body_mass, "body_inertia": body_inertia, "geom_friction": geom_friction,
            "actuator_gainprm": gainprm, "actuator_biasprm": biasprm,
        }
        in_axes = jax.tree_util.tree_map(lambda _: None, model)
        in_axes = in_axes.tree_replace({name: 0 for name in fields})
        return model.tree_replace(fields), in_axes

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


def _quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Hamilton product of (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_to_rotvec(quat: jax.Array) -> jax.Array:
    """Rotation vector of a unit quaternion (w, x, y, z); safe at zero rotation."""
    w = jp.clip(quat[0], -1.0, 1.0)
    axis = quat[1:]
    sin_half = jp.linalg.norm(axis)
    angle = 2.0 * jp.arctan2(sin_half, w)
    scale = jp.where(sin_half > 1e-8, angle / jp.maximum(sin_half, 1e-8), 2.0)
    return axis * scale


