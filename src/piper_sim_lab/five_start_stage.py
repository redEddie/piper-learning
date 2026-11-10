"""Five position-random starts with physically recorded post-release history."""
from dataclasses import asdict, replace
import math

import jax
import jax.numpy as jp
import numpy as np

from . import stage_reward_v2
from .position_random_stage import ASSET_DIR, PositionRandomStageRewardPickPlace


_CORNERS=np.array([(x,y,z) for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])


def grasp_measurements(config,cube_rotation,closing_axis,width,xp=jp):
    alignment=xp.abs(cube_rotation.T@closing_axis)
    best=xp.argmax(alignment)
    # atan2 is well-conditioned near perfect alignment; acos(dot) loses
    # precision in float32 when the true angle is almost zero.
    angle=xp.arctan2(xp.linalg.norm(xp.cross(cube_rotation[:,best],closing_axis)),alignment[best])
    return width,2*config.block_half*xp.sum(alignment),angle


def placement_measurements(config,cube,bowl,cube_rotation,bowl_rotation,supported,xp=jp):
    """Whole-cube containment in the bowl frustum, plus measured centre height.

    2mm radial/rim and5mm floor tolerances accommodate contact penetration.
    Miss detection uses the centre within the bowl cross-section at its height;
    actual inside credit
    additionally requires floor support and whole-cube containment.
    """
    local=(cube-bowl)@bowl_rotation
    corners=((xp.asarray(_CORNERS)*config.block_half)@cube_rotation.T+cube-bowl)@bowl_rotation
    heights=corners[:,2]-config.bowl_floor
    radius=config.bowl_r_bottom+(config.bowl_r_rim-config.bowl_r_bottom)*xp.clip(heights/config.bowl_height,0.,1.)
    inside=(supported & xp.all(xp.linalg.norm(corners[:,:2],axis=-1)<=radius+.002) &
            (xp.min(heights)>=-.005) & (xp.max(heights)<=config.bowl_height+.002))
    centre_height=local[2]-config.bowl_floor
    centre_radius=config.bowl_r_bottom+(config.bowl_r_rim-config.bowl_r_bottom)*xp.clip(centre_height/config.bowl_height,0.,1.)
    return centre_height,xp.linalg.norm(local[:2])<=centre_radius,inside


class FiveStartStageRewardPickPlace(PositionRandomStageRewardPickPlace):
    reset_stage_count = 5
    reward_model = stage_reward_v2

    def __init__(self, config_overrides=None):
        options=dict(config_overrides or {})
        self._progress_reward=options.get('five_start_progress_reward',False)
        if type(self._progress_reward) is not bool:
            raise ValueError('five_start_progress_reward must be boolean')
        self._physical_release=options.get('five_start_physical_release',False)
        if type(self._physical_release) is not bool:
            raise ValueError('five_start_physical_release must be boolean')
        self._dense_home = (config_overrides or {}).get('five_start_dense_home', False)
        if type(self._dense_home) is not bool:
            raise ValueError('five_start_dense_home must be boolean')
        if self._dense_home:
            from . import stage_reward_v3
            self.reward_model = stage_reward_v3
        if self._physical_release:
            if not self._dense_home:
                raise ValueError('physical release requires dense_home')
            from . import stage_reward_v4
            self.reward_model=stage_reward_v4
            options.setdefault('five_start_drop_weight',2.)
            self.SUCCESS_DEFINITION=('Cube inside supported bowl: drops>=10cm succeed and terminate; '
                'lower drops continue to settled80mm-clearance stationary A within3deg for200ms')
        if self._progress_reward:
            if not self._physical_release:
                raise ValueError('progress reward requires physical_release')
            from . import stage_reward_v5
            self.reward_model=stage_reward_v5
            self.SUCCESS_DEFINITION=('Sequential measured progress; >=10cm inside drop succeeds and terminates; '
                'lower drops require full opening,stable containment and A within0.2rad,clearance80mm,speed<=0.05rad/s for200ms')
        super().__init__(config_overrides=options)
        weight = self._config.five_start_drop_weight
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('five_start_drop_weight must be finite and nonnegative')
        # This cost never participates in prefix memory updates. The recorded
        # physics history and all release/failure predicates remain identical.
        self.reward_settings = replace(self.reward_settings, drop_weight=weight)
        if self._progress_reward:
            self.reward_settings=replace(self.reward_settings,max_gripper_width=2*self._config.gripper_max)

    def _build_reset_table(self):
        table = super()._build_reset_table()
        if self.reset_bank.get('prefix_settings') != asdict(stage_reward_v2.RewardSettings()):
            raise ValueError('post-release history settings differ from the current reward')
        path = ASSET_DIR/self._config.position_reset_bank
        with np.load(path) as bank:
            self._release_qvel = jp.asarray(bank['release_qvel'])
            if self._physical_release:
                if np.any(bank['prefix_features'][...,15]>0):
                    raise ValueError('v4 migration requires the recorded airborne prefix without floor contact')
                with jax.default_device(jax.devices('cpu')[0]):
                    replayed=self.reward_model.replay_prefixes(jp.asarray(bank['prefix_features']),
                        jp.asarray(bank['prefix_steps']),self._config.bowl_height,self._config.bowl_r_rim,self._config.bowl_r_bottom)
                    replayed=jax.tree.map(np.asarray,replayed)
                if not np.all(replayed.released) or np.any(replayed.invalid_release):
                    raise ValueError('physical release prefix replay failed')
                self._release_memory=jax.tree.map(jp.asarray,replayed)
                self.reset_bank['runtime_history_model']=self.reward_model.__name__.split('.')[-1]+' replayed from original measured airborne prefixes'
            else:
                self._release_memory = self.reward_model.Memory(*(jp.asarray(bank['memory_'+n]) for n in self.reward_model.Memory._fields))
            if self._release_qvel.shape != (self._bank_count(), self._mj_model.nv):
                raise ValueError('malformed post-release velocity bank')
            if not np.all(bank['memory_qualified_release']) or np.any(bank['memory_invalid_release']):
                raise ValueError('post-release bank must contain actual qualified releases')
            if not np.all(bank['prefix_steps'] > 0):
                raise ValueError('post-release bank lacks measured prefix history')
        self.reset_bank['reward_history_source'] = '5ms native physics prefixes stored with the checksum-validated bank'
        return table

    def _bank_count(self):
        return len(self.reset_bank['qpos'][0])

    def _reset_aux(self, stage, index, qpos, ctrl):
        velocity, goal, memory = super()._reset_aux(stage, index, qpos, ctrl)
        angle = qpos[self._qpos_ids[0]]-self._bank_qpos[4, index, self._qpos_ids[0]]
        co, si = jp.cos(angle), jp.sin(angle)
        rotation = jp.array([[co, -si], [si, co]])
        recorded = self._release_qvel[index]
        for address in (self._block_qvel_adr, self._bowl_qvel_adr):
            recorded = recorded.at[address:address+2].set(rotation@recorded[address:address+2])
        memory = self.reward_model.Memory(*(jp.where(stage == 4, v[index], empty) for v, empty in zip(self._release_memory, memory)))
        return jp.where(stage == 4, recorded, velocity), jp.where(stage == 4, self._config.gripper_max, goal), memory

    def features(self, data, goal):
        base = super().features(data, goal)
        rc = data.xmat[self._block_id].reshape(3, 3)
        rb = data.xmat[self._bowl_id].reshape(3, 3)
        bottom = base.cube_center_above_rim-self._config.block_half*jp.sum(jp.abs((rb.T@rc)[2]))
        features = stage_reward_v2.Features(*base, bottom)
        if self._dense_home:
            errors = jp.abs(data.qpos[self._qpos_ids]-self._nominal)
            if self._physical_release:
                measured=placement_measurements(self._config,data.xpos[self._block_id],
                    data.xpos[self._bowl_id],rc,rb,features.bowl_floor_contact)
                if self._progress_reward:
                    grip=grasp_measurements(self._config,rc,data.site_xmat[self._tcp_id][:,self._closing_axis],
                        data.qpos[self._finger_qpos_ids[0]]-data.qpos[self._finger_qpos_ids[1]])
                    return self.reward_model.Features(*features,*errors,*measured,*grip)
                return self.reward_model.Features(*features,*errors,*measured)
            return self.reward_model.Features(*features, *errors)
        return features

    @staticmethod
    def _reward_metrics(terms, memory, features):
        metrics = PositionRandomStageRewardPickPlace._reward_metrics(terms, memory, features)
        return dict(metrics, task_failed=(memory.invalid_release | memory.outside_drop).astype(jp.float32))

    def step(self, state, action):
        updated = super().step(state, action)
        failed = updated.info['reward_memory'].invalid_release | updated.info['reward_memory'].outside_drop
        terminal_success=updated.metrics.get('stage_terminal_success',jp.array(0.))
        if self._physical_release:
            updated=updated.replace(metrics=dict(updated.metrics,
                inside_success=updated.info['reward_features'].cube_inside_bowl.astype(jp.float32)))
        return updated.replace(done=jp.maximum(updated.done, jp.maximum(failed.astype(jp.float32),terminal_success)))


def native_features(env, data, goal):
    """Equivalent C-MuJoCo measurements for reset-prefix generation only."""
    import mujoco
    c, model = env._config, env.mj_model
    cube, bowl, tcp = data.xpos[env._block_id], data.xpos[env._bowl_id], data.site_xpos[env._tcp_id]
    rc, rb = data.xmat[env._block_id].reshape(3, 3), data.xmat[env._bowl_id].reshape(3, 3)
    local = rb.T@(cube-bowl)
    bv = data.qvel[env._bowl_qvel_adr:env._bowl_qvel_adr+3]
    bw = rb@data.qvel[env._bowl_qvel_adr+3:env._bowl_qvel_adr+6]
    cv = data.qvel[env._block_qvel_adr:env._block_qvel_adr+3]
    jac, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac, jacr, env._tcp_id)
    tv = jac@data.qvel
    cr, tr = cv-bv-np.cross(bw, cube-bowl), tv-bv-np.cross(bw, tcp-bowl)
    if env._sensor_addresses:
        touch = {n: data.sensordata[a] > 0 for n, a in env._sensor_addresses.items()}
    else:
        touch = env._native_contacts(data)
    cg = np.linalg.norm(np.maximum(np.abs(rc.T@(tcp-cube))-c.block_half, 0.))
    gs = env._bowl_geoms
    delta = np.einsum('gji,gj->gi', data.geom_xmat[gs].reshape(-1, 3, 3), tcp-data.geom_xpos[gs])
    bg = np.min(np.linalg.norm(np.maximum(np.abs(delta)-model.geom_size[gs], 0.), axis=1))
    closing = data.site_xmat[env._tcp_id].reshape(3, 3)[:, env._closing_axis]
    features = stage_reward_v2.Features(np.linalg.norm(tcp-cube),
        cube[2]-c.block_half*np.abs(rc[2]).sum(), np.linalg.norm(local[:2]), local[2]-env._bowl_rim_z,
        (rb.T@(tcp-bowl))[2]-c.bowl_floor, np.linalg.norm(tr), np.linalg.norm(cr),
        np.linalg.norm(cv[:2]), max(-rb[:, 2]@cr, 0.),
        np.arctan2(closing[1], closing[0])-np.arctan2(rc[1, 0], rc[0, 0]), min(cg, bg),
        np.max(np.abs(data.qpos[env._qpos_ids]-env._nominal)), np.max(np.abs(data.qvel[env._qvel_ids])),
        touch['left'], touch['right'], touch['support'], touch['table'],
        any(touch['block'+str(i)] for i in range(1, 9)), any(touch['bowl'+str(i)] for i in range(1, 9)),
        goal > c.gripper_max/2, local[2]-env._bowl_rim_z-c.block_half*np.abs((rb.T@rc)[2]).sum())
    if env._dense_home:
        if env._physical_release:
            extra=placement_measurements(c,cube,bowl,rc,rb,features.bowl_floor_contact,xp=np)
            if env._progress_reward:
                grip=grasp_measurements(c,rc,data.site_xmat[env._tcp_id].reshape(3,3)[:,env._closing_axis],
                    data.qpos[env._finger_qpos_ids[0]]-data.qpos[env._finger_qpos_ids[1]],xp=np)
                return env.reward_model.Features(*features,*np.abs(data.qpos[env._qpos_ids]-np.asarray(env._nominal)),*extra,*grip)
            return env.reward_model.Features(*features,*np.abs(data.qpos[env._qpos_ids]-np.asarray(env._nominal)),*extra)
        return env.reward_model.Features(*features, *np.abs(data.qpos[env._qpos_ids]-np.asarray(env._nominal)))
    return features
