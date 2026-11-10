"""Current-position shaping for five-start continuation; v1 remains reproducible.

No new collision classifier. Keep v1 release qualification, add measured best
carry distance and bounded preparation credit. Invalid drops cost0.5 once and
the five-start environment terminates immediately. Extra geometry is the cube
bottom along the bowl up axis, including actual orientation.
"""
from collections import namedtuple
from dataclasses import dataclass, fields
import math

import jax.numpy as jp

from . import stage_reward as v1
from .stage_reward import start_interval

Features = namedtuple('Features', v1.Features._fields + ('cube_bottom_above_rim',),
                      defaults=tuple(v1.Features()) + (.02,))
Memory = namedtuple('Memory', v1.Memory._fields + ('best_carry_xy', 'prepare_time'),
                    defaults=tuple(v1.Memory()) + (float('inf'), 0.))


def reset_memory():
    return Memory(*(jp.asarray(v) for v in Memory()))


@dataclass(frozen=True)
class RewardSettings(v1.RewardSettings):
    carry_margin: float = .005
    carry_height_scale: float = .04
    release_xy_scale: float = .04
    away_distance: float = .10
    away_weight: float = .5
    prepare_credit_seconds: float = 1.
    drop_weight: float = .5

    def __post_init__(self):
        # v1 requires every field to be positive; only this new event cost
        # may be disabled. Validate v1 and the remaining v2 fields unchanged.
        if not math.isfinite(self.drop_weight) or self.drop_weight < 0:
            raise ValueError('drop_weight must be finite and nonnegative')
        base_fields = {f.name for f in fields(v1.RewardSettings)}
        v1.RewardSettings(**{name:getattr(self, name) for name in base_fields})
        for f in fields(self):
            if f.name not in base_fields and f.name != 'drop_weight':
                value = getattr(self, f.name)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{f.name} must be positive finite')


def advance_memory(old, before, after, c=RewardSettings()):
    measured = v1.advance_memory(v1.Memory(*old[:len(v1.Memory._fields)]), v1.Features(*before[:20]), v1.Features(*after[:20]), c)
    gripping = after.left_contact | after.right_contact
    carrying = gripping & ~measured.released & ((measured.grasp_time >= c.grasp_seconds-1e-6) | measured.lifted)
    best = jp.where(carrying, jp.minimum(old.best_carry_xy, after.cube_bowl_xy), old.best_carry_xy)
    ready = (carrying & measured.bowl_reached & (after.cube_bowl_xy <= c.alignment_radius) &
             (after.tcp_floor_height >= c.release_height_min) & (after.tcp_floor_height <= c.release_height_max))
    return Memory(*measured, best, old.prepare_time+jp.where(ready, c.physics_dt, 0.))


def main_score(memory, f, c=RewardSettings()):
    main, phase, complete = v1.main_score(memory, v1.Features(*f[:20]), c)
    xy = jp.maximum(f.cube_bowl_xy, 0.)
    xy_progress = 1. - jp.tanh(c.reach_scale * xy)
    centered = jp.exp(-(jp.maximum(xy-c.alignment_radius, 0.)/c.release_xy_scale)**2)
    above_rim = 1. - jp.tanh(jp.maximum(c.carry_margin-f.cube_bottom_above_rim, 0.)/c.carry_height_scale)
    # Above the sides during transport, or centered in the opening while
    # lowering. No prescribed path, output override, or collision classifier.
    carry_quality = jp.maximum(above_rim, centered)
    transport = 2. + xy_progress * carry_quality
    height_error = jp.maximum(c.release_height_min-f.tcp_floor_height,
                              jp.maximum(f.tcp_floor_height-c.release_height_max, 0.))
    low = 1. - jp.tanh(height_error/c.lower_scale)
    remaining_credit = jp.clip(1.-memory.prepare_time/c.prepare_credit_seconds, 0., 1.)
    prepare = transport + .5 * low * centered * remaining_credit
    main = jp.where(phase == 2, transport, jp.where(phase == 3, prepare, main))
    return main, phase, complete


def reward_terms(old, new, f, action, previous_action, c=RewardSettings()):
    terms = v1.reward_terms(v1.Memory(*old[:len(v1.Memory._fields)]),
        v1.Memory(*new[:len(v1.Memory._fields)]), v1.Features(*f[:20]), action, previous_action, c)
    main, phase, complete = main_score(new, f, c)
    invalid = terms['nonfinite'] | ~jp.isfinite(f.cube_bottom_above_rim)
    gripping = f.left_contact | f.right_contact
    carrying = gripping & ~new.released & ((new.grasp_time >= c.grasp_seconds-1e-6) | new.lifted)
    lost_distance = jp.maximum(f.cube_bowl_xy-new.best_carry_xy, 0.)
    away = jp.where(carrying & (lost_distance >= c.away_distance-1e-6),
                    c.away_weight*lost_distance/c.away_distance, 0.)
    failed = new.invalid_release | new.outside_drop
    previous_failed = old.invalid_release | old.outside_drop
    drop = c.drop_weight*(failed & ~previous_failed)
    # Remove the perpetual +0.5 XY supplement. Current XY remains in main;
    # low preparation credit decays over one second of real target occupancy.
    reward = terms['reward'] - terms['main'] - terms['xy_alignment'] + main - away - drop
    return dict(terms, main=main, section=phase, reward=jp.where(invalid, -1., reward),
                xy_alignment=jp.array(0.), away_cost=away, drop_cost=drop,
                best_carry_xy=jp.where(jp.isfinite(new.best_carry_xy), new.best_carry_xy, f.cube_bowl_xy),
                prepare_time=new.prepare_time, complete=complete & ~invalid, nonfinite=invalid)
