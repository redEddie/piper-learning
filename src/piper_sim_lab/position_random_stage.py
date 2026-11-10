"""Position-random SHOW-103 resets; reward/observations/controller are unchanged.

Native-validated coupled arm/cube poses are sampled, continuously rotated about
joint1, and paired with an independently sampled bowl except at over-bowl starts.
Empty-A cube and bowl XY are continuous independent draws with clearance checks.
Evaluation uses a separately generated pose bank, selected by the evaluator.
"""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from .scene_frames import quat_multiply
from .stage_reward_env import StageRewardPickPlace

ASSET_DIR = Path(__file__).with_name('assets')/'reset_banks'


def geometry_fingerprint(model):
    digest = hashlib.sha256()
    for name in ('geom_size', 'geom_pos', 'geom_quat', 'body_pos', 'body_quat', 'site_pos', 'jnt_range'):
        digest.update(name.encode())
        digest.update(np.asarray(getattr(model, name), dtype='<f8').tobytes())
    return digest.hexdigest()


class PositionRandomStageRewardPickPlace(StageRewardPickPlace):
    def _build_reset_table(self):
        name = self._config.position_reset_bank
        if not name or Path(name).name != name:
            raise ValueError('position_reset_bank must name a bundled reset-bank NPZ')
        path = ASSET_DIR/name
        metadata = json.loads(path.with_suffix('.json').read_text())
        if metadata['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('position reset bank checksum mismatch')
        if metadata['geometry_sha256'] != geometry_fingerprint(self._mj_model):
            raise ValueError('position reset bank geometry differs from the current model')
        for key, expected in metadata['physical_config'].items():
            if self._config[key] != expected:
                raise ValueError(f'position reset bank setting mismatch: {key}')
        with np.load(path) as bank:
            qpos, ctrl = bank['qpos'], bank['ctrl']
        if qpos.shape[:2] != ctrl.shape[:2] or qpos.shape[0] != getattr(self, 'reset_stage_count', 4) or qpos.shape[2] != self._mj_model.nq:
            raise ValueError('malformed four-start position reset bank')
        self.reset_bank = dict(metadata, qpos=qpos.tolist(), ctrl=ctrl.tolist(), source_asset=name)
        return {n: ctrl[i][:, self._arm_ctrl_ids]
                for i, n in enumerate(('unused', 'pregrasp', 'grasp', 'preinside')) if i > 0}

    def _reset_pose(self, key, stage):
        c = self._config
        pick, angle, cube_key, bowl_key = jax.random.split(jax.random.fold_in(key, 1031), 4)
        index = jax.random.randint(pick, (), 0, self._bank_qpos.shape[1])
        qpos, ctrl = self._bank_qpos[stage, index], self._bank_ctrl[stage, index]
        joint = int(self._qpos_ids[0]); actuator = int(self._arm_ctrl_ids[0])
        lo = jp.maximum(-c.position_yaw_range, self._lower[0]-jp.minimum(qpos[joint], ctrl[actuator]))
        hi = jp.minimum(c.position_yaw_range, self._upper[0]-jp.maximum(qpos[joint], ctrl[actuator]))
        theta = jax.random.uniform(angle, (), minval=lo, maxval=hi)
        if c.position_bound_yaw:
            # Keep coupled held/release poses in the requested XY rectangle.
            # Zero is an explicit fallback to the native-validated bank pose.
            angles = jp.concatenate([jax.random.uniform(angle, (32,), minval=lo, maxval=hi), jp.zeros(1)])
            co, si = jp.cos(angles), jp.sin(angles)
            rotations = jp.stack([jp.stack([co, -si], -1), jp.stack([si, co], -1)], -2)
            pivot = jp.asarray(self.reset_bank['joint1_pivot_world_xy'])
            cube_xy = pivot+jp.einsum('nij,j->ni', rotations, qpos[self._block_qpos_adr:self._block_qpos_adr+2]-pivot)
            bowl_xy = pivot+jp.einsum('nij,j->ni', rotations, qpos[self._bowl_qpos_adr:self._bowl_qpos_adr+2]-pivot)
            def valid_xy(xy, x, y):
                return (jp.all(xy >= jp.array([x[0], y[0]]), -1) &
                        jp.all(xy <= jp.array([x[1], y[1]]), -1) &
                        (jp.linalg.norm(xy, axis=-1) <= c.spawn_max_radius))
            valid = valid_xy(cube_xy, c.block_x, c.block_y)
            valid &= (stage < 3) | valid_xy(bowl_xy, c.bowl_x, c.bowl_y)
            theta = jp.where(jp.any(valid), angles[jp.argmax(valid)], jp.nan)
        theta = jp.where(stage == 0, 0., theta)  # empty start remains the exact A arm pose
        co, si = jp.cos(theta), jp.sin(theta)
        rotation = jp.array([[co, -si], [si, co]])
        pivot = jp.asarray(self.reset_bank['joint1_pivot_world_xy'])
        quaternion = jp.array([jp.cos(theta/2), 0., 0., jp.sin(theta/2)])
        qpos = qpos.at[joint].add(theta)
        ctrl = ctrl.at[actuator].add(theta)
        for address in (self._block_qpos_adr, self._bowl_qpos_adr):
            qpos = qpos.at[address:address+2].set(pivot+rotation@(qpos[address:address+2]-pivot))
            qpos = qpos.at[address+3:address+7].set(quat_multiply(quaternion, qpos[address+3:address+7], xp=jp))
        # No independent perturbation of a held cube relative to its fingers.
        count = c.spawn_candidates
        low = jp.array([c.block_x[0], c.block_y[0]])
        high = jp.array([c.block_x[1], c.block_y[1]])
        cube_candidates = jax.random.uniform(cube_key, (count, 2), minval=low, maxval=high)
        cube_valid = jp.linalg.norm(cube_candidates, axis=1) <= c.spawn_max_radius
        cube = cube_candidates[jp.argmax(cube_valid)]
        cube = jp.where(jp.any(cube_valid), cube, jp.full(2, jp.nan))
        old_cube = qpos[self._block_qpos_adr:self._block_qpos_adr+2]
        cube = jp.where(stage == 0, cube, old_cube)
        qpos = qpos.at[self._block_qpos_adr:self._block_qpos_adr+2].set(cube)
        low, high = jp.array([c.bowl_x[0], c.bowl_y[0]]), jp.array([c.bowl_x[1], c.bowl_y[1]])
        candidates = jax.random.uniform(bowl_key, (count, 2), minval=low, maxval=high)
        valid = ((jp.linalg.norm(candidates, axis=1) <= c.spawn_max_radius) &
                 (jp.linalg.norm(candidates-cube, axis=1) >= c.spawn_min_distance))
        # Reject an invalid candidate set explicitly; never count an overlap as a valid reset.
        bowl = candidates[jp.argmax(valid)]
        bowl = jp.where(jp.any(valid), bowl, jp.full(2, jp.nan))
        old_bowl = qpos[self._bowl_qpos_adr:self._bowl_qpos_adr+2]
        bowl = jp.where(stage >= 3, old_bowl, bowl)
        qpos = qpos.at[self._bowl_qpos_adr:self._bowl_qpos_adr+2].set(bowl)
        return qpos, ctrl, index
