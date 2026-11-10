"""SHOW-103 measured five-section reward with fixed, physically settled starts.

Uses SHOW-100 observations and rate-limited ternary gripper, and the existing
GPU DLS/QP checked-plan controller. No legacy task reward or grip coordinator.
Contact sensors belong to each world; the shared Warp contact buffer is unused.
"""
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import support
from mujoco_playground._src import mjx_env
import numpy as np

from .ternary_control import TernaryControlledPickPlace
from .direct_gripper import reset_arm, advance_arm, arm_metrics, direct_goal
from .stage_reward import Features
from . import stage_reward


DEFAULTS = dict(control_acceleration=5., control_jerk=100., max_joint_vel=1.,
                max_lin_vel=.1, max_ang_vel=.5, smooth_stop_steps=64,
                reset_joint_noise=0., obs_randomization=False,
                fixed_reset_settle_steps=20, episode_length=200,
                position_reset_bank='', evaluation_position_reset_bank='',
                position_yaw_range=.2617993877991494, position_bound_yaw=False, five_start_drop_weight=.5,
                five_start_dense_home=False, five_start_physical_release=False,
                five_start_progress_reward=False)


class StageRewardPickPlace(TernaryControlledPickPlace):
    reward_model = stage_reward
    SUCCESS_DEFINITION = "qualified low release, supported settling, 80mm clearance and stationary return to A"

    def __init__(self, config_overrides=None):
        super().__init__(config_overrides=config_overrides, task_defaults=DEFAULTS)
        c = self._config
        if not c.bowl_free or c.stage_obs or not c.size_obs:
            raise ValueError('stage reward requires free bowl, size observation and no stage input')
        if c.reset_joint_noise != 0:
            raise ValueError('independent joint noise would invalidate the contact-validated reset poses')
        if type(c.position_bound_yaw) is not bool:
            raise ValueError('position_bound_yaw must be boolean')
        if self.n_substeps != 20 or self.dt != .1 or self.sim_dt != .005:
            raise ValueError('reward v1 requires 20 physical samples per 100ms')
        self.stop_steps = int(c.smooth_stop_steps)
        self.reward_settings = self.reward_model.RewardSettings()
        self._bank_qpos = jp.asarray(self.reset_bank['qpos'], jp.float32)
        self._bank_ctrl = jp.asarray(self.reset_bank['ctrl'], jp.float32)
        self._sensor_addresses = {n: int(self._mj_model.sensor('stage_'+n).adr[0])
                                  for n in self._contact_pairs()} if c.impl == 'warp' else {}
        self._bowl_qvel_adr = int(self._mj_model.jnt_dofadr[self._mj_model.body_jntadr[self._bowl_id]])
        self._bowl_geoms = np.flatnonzero(self._mj_model.geom_bodyid == self._bowl_id)
        self._robot_bodies = [self._mj_model.body('link'+str(i)).id for i in range(1, 9)]

    @staticmethod
    def _contact_pairs():
        pairs = dict(left=('link7', 'block'), right=('link8', 'block'),
                     support=('block', 'stage_bowl_floor'), table=('block', 'floor'))
        for obj in ('block', 'bowl'):
            pairs.update({obj+str(i): ('link'+str(i), obj) for i in range(1, 9)})
        return pairs

    def _configure_task_spec(self, spec):
        spec.body('bowl').geoms[0].name = 'stage_bowl_floor'
        if self._config.impl != 'warp':
            return
        for name, (a, b) in self._contact_pairs().items():
            is_geom = name in ('support', 'table')
            sensor = spec.add_sensor(name='stage_'+name, type=mujoco.mjtSensor.mjSENS_CONTACT,
                objtype=mujoco.mjtObj.mjOBJ_BODY, objname=a,
                reftype=mujoco.mjtObj.mjOBJ_GEOM if is_geom else mujoco.mjtObj.mjOBJ_BODY, refname=b)
            sensor.intprm = [1, 0, 1]

    def _build_reset_table(self):
        """Resolve a deterministic four-pose bank in native MuJoCo, without fake reward history."""
        model, c = self._mj_model, self._config
        lower, upper = np.asarray(self._lower), np.asarray(self._upper)
        desired = np.diag([1., -1., -1.])
        data = mujoco.MjData(model)
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

        def solve(target):
            from scipy.optimize import least_squares
            from scipy.spatial.transform import Rotation
            def residual(q, goal):
                data.qpos[:] = model.qpos0
                data.qpos[self._qpos_ids] = q
                mujoco.mj_kinematics(model, data)
                rotation = data.site_xmat[self._tcp_id].reshape(3, 3)
                return np.r_[5*(target-data.site_xpos[self._tcp_id]),
                             Rotation.from_matrix(goal@rotation.T).as_rotvec()]
            seeds = [np.asarray(c.reset_ik_seed), np.asarray(c.nominal_qpos)]
            goals = []
            for yaw in (0., -np.pi/2, np.pi/2, np.pi, -np.pi/4, np.pi/4):
                rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
                goals.append(rz@desired)
            for goal in goals:
                for seed in seeds:
                    result = least_squares(lambda q: residual(q, goal),
                        np.clip(seed, lower+1e-6, upper-1e-6), bounds=(lower, upper),
                        max_nfev=400, ftol=1e-11, xtol=1e-11, gtol=1e-11)
                    if np.linalg.norm(residual(result.x, goal)) < 1e-4:
                        return result.x
            raise ValueError(f'fixed reset target unreachable: {target}')

        targets = [None, np.array([.30, -.12, .065]),
                   np.array([.30, -.12, .10]), np.array([.25, .12, .125])]
        arms = [np.asarray(c.nominal_qpos)] + [solve(t) for t in targets[1:]]
        positions, controls, records = [], [], []
        for stage, q in enumerate(arms):
            mujoco.mj_resetData(model, data)
            data.qpos[self._qpos_ids] = q
            held = stage >= 2
            data.qpos[self._finger_qpos_ids] = [c.block_half, -c.block_half] if held else [c.gripper_max, -c.gripper_max]
            mujoco.mj_kinematics(model, data)
            tcp = data.site_xpos[self._tcp_id].copy()
            axis = data.site_xmat[self._tcp_id].reshape(3, 3)[:, self._closing_axis]
            yaw = np.arctan2(axis[1], axis[0])
            block = tcp if held else np.array([.30, -.12, c.block_half])
            data.qpos[self._block_qpos_adr:self._block_qpos_adr+7] = np.r_[block, np.cos(yaw/2) if stage else 1., 0., 0., np.sin(yaw/2) if stage else 0.]
            data.qpos[self._bowl_qpos_adr:self._bowl_qpos_adr+7] = [.25, .12, 0., 1., 0., 0., 0.]
            data.ctrl[self._arm_ctrl_ids] = q
            data.ctrl[self._gripper_ctrl_id] = 0. if held else c.gripper_max
            # Native contact stabilization creates a valid physical pose, not timers/credits.
            for _ in range(60):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            contacts = self._native_contacts(data)
            if held and not (contacts['left'] and contacts['right']):
                raise ValueError(f'fixed held reset {stage} lacks bilateral contact: {contacts}')
            positions.append(data.qpos.copy()); controls.append(data.ctrl.copy())
            records.append(dict(stage=stage, tcp_world=data.site_xpos[self._tcp_id].tolist(),
                block_world=data.xpos[self._block_id].tolist(), bowl_world=data.xpos[self._bowl_id].tolist(),
                contacts=contacts, stabilization_seconds=.3))
        self.reset_bank = dict(version='show103-fixed-v1', qpos=np.array(positions).tolist(),
            ctrl=np.array(controls).tolist(), records=records,
            note='Native settled poses; each MJX reset settles another 100ms with its actual per-world dynamics; reward memory starts empty')
        return {name: np.array([q]) for name, q in zip(('pregrasp', 'grasp', 'preinside'), arms[1:])}

    def _native_contacts(self, data):
        model = self._mj_model
        body = model.geom_bodyid
        def body_pair(a, b):
            return any(ct.dist <= 0 and {int(body[ct.geom1]), int(body[ct.geom2])} == {model.body(a).id, model.body(b).id}
                       for ct in data.contact)
        def geom_pair(name):
            g = model.geom(name).id
            block = model.body('block').id
            return any(ct.dist <= 0 and ((ct.geom1 == g and body[ct.geom2] == block) or
                                        (ct.geom2 == g and body[ct.geom1] == block)) for ct in data.contact)
        result = dict(left=body_pair('link7', 'block'), right=body_pair('link8', 'block'),
                      support=geom_pair('stage_bowl_floor'), table=geom_pair('floor'))
        result.update({obj+str(i): body_pair('link'+str(i), obj) for obj in ('block', 'bowl') for i in range(1, 9)})
        return result

    def contacts(self, data):
        if self._config.impl == 'warp':
            return {n: data.sensordata[a] > 0 for n, a in self._sensor_addresses.items()}
        contact = data._impl.contact
        ga, gb = contact.geom[:, 0], contact.geom[:, 1]
        body = jp.asarray(self._mj_model.geom_bodyid)
        ba, bb = body[ga], body[gb]
        active = (contact.dist <= 0) & (contact.efc_address >= 0)
        result = {}
        for n, (a, b) in self._contact_pairs().items():
            ia = self._mj_model.body(a).id
            if n in ('support', 'table'):
                ib = self._mj_model.geom(b).id
                mask = ((ba == ia) & (gb == ib)) | ((bb == ia) & (ga == ib))
            else:
                ib = self._mj_model.body(b).id
                mask = ((ba == ia) & (bb == ib)) | ((bb == ia) & (ba == ib))
            result[n] = jp.any(active & mask)
        return result

    def features(self, data, goal):
        c = self._config
        cube, bowl, tcp = data.xpos[self._block_id], data.xpos[self._bowl_id], data.site_xpos[self._tcp_id]
        rc, rb = data.xmat[self._block_id].reshape(3, 3), data.xmat[self._bowl_id].reshape(3, 3)
        local = rb.T@(cube-bowl)
        bowl_v = data.qvel[self._bowl_qvel_adr:self._bowl_qvel_adr+3]
        bowl_w = rb@data.qvel[self._bowl_qvel_adr+3:self._bowl_qvel_adr+6]
        cube_v = data.qvel[self._block_qvel_adr:self._block_qvel_adr+3]
        jacp, _ = support.jac(self._mjx_model, data, tcp, self._link6_id)
        tcp_v = jacp.T@data.qvel
        cube_rel = cube_v-bowl_v-jp.cross(bowl_w, cube-bowl)
        tcp_rel = tcp_v-bowl_v-jp.cross(bowl_w, tcp-bowl)
        contact = self.contacts(data)
        # Exact point-to-box distances for the cube and all 13 bowl collision boxes.
        cube_gap = jp.linalg.norm(jp.maximum(jp.abs(rc.T@(tcp-cube))-c.block_half, 0.))
        gs = self._bowl_geoms
        delta = jp.einsum('gji,gj->gi', data.geom_xmat[gs].reshape(-1, 3, 3), tcp-data.geom_xpos[gs])
        bowl_gap = jp.min(jp.linalg.norm(jp.maximum(jp.abs(delta)-self._mjx_model.geom_size[gs], 0.), axis=1))
        closing = data.site_xmat[self._tcp_id][:, self._closing_axis]
        yaw_error = jp.arctan2(closing[1], closing[0])-jp.arctan2(rc[1, 0], rc[0, 0])
        return Features(tcp_cube_distance=jp.linalg.norm(tcp-cube),
            cube_bottom_table=cube[2]-c.block_half*jp.sum(jp.abs(rc[2])),
            cube_bowl_xy=jp.linalg.norm(local[:2]),
            cube_center_above_rim=local[2]-self._bowl_rim_z,
            tcp_floor_height=(rb.T@(tcp-bowl))[2]-c.bowl_floor,
            tcp_relative_speed=jp.linalg.norm(tcp_rel), cube_relative_speed=jp.linalg.norm(cube_rel),
            cube_horizontal_speed=jp.linalg.norm(cube_v[:2]),
            cube_downward_speed=jp.maximum(-rb[:, 2]@cube_rel, 0.), yaw_error=yaw_error,
            clearance=jp.minimum(cube_gap, bowl_gap),
            joint_error_a=jp.max(jp.abs(data.qpos[self._qpos_ids]-self._nominal)),
            joint_speed=jp.max(jp.abs(data.qvel[self._qvel_ids])),
            left_contact=contact['left'], right_contact=contact['right'], bowl_floor_contact=contact['support'],
            cube_table_contact=contact['table'],
            robot_cube_contact=jp.any(jp.stack([contact['block'+str(i)] for i in range(1, 9)])),
            robot_bowl_contact=jp.any(jp.stack([contact['bowl'+str(i)] for i in range(1, 9)])),
            opening_intent=goal > c.gripper_max/2)

    def _reset_pose(self, key, stage):
        return self._bank_qpos[stage], self._bank_ctrl[stage], jp.array(0, jp.int32)

    def _reset_aux(self, stage, index, qpos, ctrl):
        return jp.zeros(self._mj_model.nv), ctrl[self._gripper_ctrl_id], self.reward_model.reset_memory()

    def reset(self, rng):
        rng, choice, perception, observation = jax.random.split(rng, 4)
        stage = jax.random.choice(choice, len(self._stage_probs), p=self._stage_probs)
        qpos, ctrl, pose_index = self._reset_pose(choice, stage)
        qvel, goal, memory = self._reset_aux(stage, pose_index, qpos, ctrl)
        data = mjx_env.make_data(self._mj_model, qpos=qpos,
            qvel=qvel, ctrl=ctrl,
            impl=self._config.impl, **self._make_data_kwargs())
        data = mjx.forward(self._mjx_model, data)
        before = self.features(data, goal)
        def settle(carry, _):
            data, overflow, memory, before = carry
            post_release = stage == 4
            reference = jp.clip(goal, data.ctrl[self._gripper_ctrl_id]-self._config.gripper_rate/20,
                                data.ctrl[self._gripper_ctrl_id]+self._config.gripper_rate/20)
            data = data.replace(ctrl=data.ctrl.at[self._gripper_ctrl_id].set(
                jp.where(post_release, reference, data.ctrl[self._gripper_ctrl_id])))
            data = mjx.step(self._mjx_model, data)
            data = mjx.kinematics(self._mjx_model, data)
            after = self.features(data, goal)
            advanced = self.reward_model.advance_memory(memory, before, after, self.reward_settings)
            track_history=post_release | jp.asarray(self._config.five_start_progress_reward)
            memory = jax.tree.map(lambda a, b: jp.where(track_history, a, b), advanced, memory)
            return (data, overflow | self._capacity_overflow(data), memory, after), None
        (data, overflow, memory, _), _ = jax.lax.scan(settle, (data, self._capacity_overflow(data), memory, before),
            None, length=int(self._config.fixed_reset_settle_steps))
        data = mjx.kinematics(self._mjx_model, data)
        p = self._predicates(data)
        percep = self._reset_perception(data, perception)
        obs, percep = self._get_obs(data, jp.zeros(7), p, percep, observation, gripper_goal=goal)
        zero = jp.array(0., jp.float32)
        metrics = {n: zero for n in ('tcp_block', 'block_bowl_xy', 'block_height', 'stage', 'grasped',
            'success', 'reward_reach', 'reward_lift', 'reward_carry', 'reward_inside',
            'action_penalty', 'action_rate_penalty', 'nonfinite', 'capacity_overflow')}
        metrics['capacity_overflow'] = overflow.astype(jp.float32)
        state = mjx_env.State(data=data, obs=obs, reward=zero, done=overflow.astype(jp.float32),
            metrics=metrics, info=dict(rng=rng, reset_stage=stage, reset_pose_index=pose_index, perception=percep))
        state = reset_arm(self, super()._reset_controller(state))
        state = state.replace(info=dict(state.info, gripper_goal=goal))
        features = self.features(data, goal)
        if hasattr(self.reward_model,'initialize'):
            memory=self.reward_model.initialize(memory,features,self.reward_settings)
        terms = self.reward_model.reward_terms(memory, memory, features, jp.zeros(7), jp.zeros(7), self.reward_settings)
        metrics = dict(state.metrics, **self._reward_metrics(terms, memory, features),
            gripper_goal_open=goal/self._config.gripper_max,
            inside_success=p['inside'].astype(jp.float32),
            reset_contact_invalid=((stage >= 2) & (stage < 4) & ~(features.left_contact & features.right_contact)).astype(jp.float32))
        return state.replace(info=dict(state.info, reward_memory=memory, reward_features=features,
            reward_trace=jp.zeros((20, len(self.reward_model.Features._fields))), reward_substeps=jp.array(0, jp.int32)), metrics=metrics)

    def _advance_control(self, data, ctrl, action, hold, info):
        return advance_arm(self, data, ctrl, action, hold, info)

    def _start_task_scan(self, data, info):
        return (self.reward_model.start_interval(info['reward_memory']), info['reward_features'],
                jp.zeros((20, len(self.reward_model.Features._fields))), jp.array(0, jp.int32), info['gripper_goal'])

    def _task_substep(self, previous, current, action, task):
        memory, before, trace, index, goal = task
        goal = direct_goal(action[6], goal, self._config.gripper_max, self._config.gripper_threshold)
        current = mjx.kinematics(self._mjx_model, current)
        after = self.features(current, goal)
        memory = self.reward_model.advance_memory(memory, before, after, self.reward_settings)
        return memory, after, trace.at[index].set(jp.asarray(after, jp.float32)), index+1, goal

    def _finish_task_scan(self, task):
        return dict(reward_memory=task[0], reward_features=task[1], reward_trace=task[2], reward_substeps=task[3])

    @staticmethod
    def _reward_metrics(terms, memory, features):
        metrics = {'stage_'+n: jp.asarray(v, jp.float32) for n, v in terms.items()}
        metrics.update(qualified_release=memory.qualified_release.astype(jp.float32),
            settled=(memory.support_time >= .1-1e-6).astype(jp.float32),
            release_height=memory.candidate_height, release_tcp_speed=memory.candidate_tcp_speed,
            release_cube_speed=memory.candidate_cube_speed, landing_speed=memory.impact_peak,
            release_invalid=memory.invalid_release.astype(jp.float32),
            tcp_floor_height=features.tcp_floor_height, cube_xy=features.cube_bowl_xy,
            clearance=features.clearance, home_error=features.joint_error_a,
            home_speed=features.joint_speed)
        return metrics

    def step(self, state, action):
        updated = super().step(state, action)
        memory, features = updated.info['reward_memory'], updated.info['reward_features']
        terms = self.reward_model.reward_terms(state.info['reward_memory'], memory, features, action,
            state.obs['state'][-7:], self.reward_settings)
        invalid = (updated.metrics['nonfinite'] > 0) | (updated.metrics['capacity_overflow'] > 0) | terms['nonfinite']
        failed = updated.info['controller_failed']
        metrics = dict(updated.metrics, **self._reward_metrics(terms, memory, features),
            **arm_metrics(self, updated), success=(terms['complete'] & ~invalid & ~failed).astype(jp.float32),
            inside_success=self._predicates(updated.data)['inside'].astype(jp.float32),
            nonfinite=jp.maximum(updated.metrics['nonfinite'], terms['nonfinite'].astype(jp.float32)))
        # Reward is replaced, not added to the inherited SHOW-76 calculation.
        for name in ('reward_reach', 'reward_lift', 'reward_carry', 'reward_inside'):
            metrics[name] = jp.array(0., jp.float32)
        metrics.update(action_penalty=terms['action_cost'], action_rate_penalty=terms['action_rate_cost'])
        return updated.replace(reward=jp.where(invalid | failed, -1., terms['reward']),
            done=jp.maximum(updated.done, (invalid | failed).astype(jp.float32)), metrics=metrics)
