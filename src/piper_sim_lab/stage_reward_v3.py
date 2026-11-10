"""Per-joint, continuous A shaping; v2 physics memory/completion stay unchanged.

The extra six absolute errors are reward measurements only. Actor49 and
critic66 stay unchanged. Stored v2 release prefixes remain valid because no
memory transition or release/home-completion threshold changes.
"""
from collections import namedtuple

import jax.numpy as jp

from . import stage_reward_v2 as v2
from .stage_reward_v2 import (Memory, RewardSettings, advance_memory,
                              reset_memory, start_interval)

Features = namedtuple('Features', v2.Features._fields + tuple('home_error_'+str(i) for i in range(6)),
                      defaults=tuple(v2.Features()) + (0.,)*6)


def home_progress(f, c=RewardSettings()):
    """Every joint contributes throughout its range, including beyond1rad."""
    errors = jp.abs(jp.asarray(f[-6:]))
    return jp.mean(jp.exp(-errors/c.home_progress_scale))


def main_score(memory, f, c=RewardSettings()):
    main, phase, complete = v2.main_score(memory, f, c)
    land = jp.clip(memory.support_time/c.support_seconds, 0., 1.)
    clear = jp.clip(f.clearance/c.clearance, 0., 1.)
    touch = memory.contact_window | f.robot_cube_contact | f.robot_bowl_contact
    # Preserve the original landing/clearance/contact gates and the0.5 home
    # budget. Only replace the worst-joint, clipped progress within stage5.
    retreat = (4. + .25*land + .25*clear*~touch + .5*home_progress(f,c)*
               (land >= 1.-1e-6)*(clear >= 1.)*~touch)
    retreat = jp.where(complete, 5., jp.minimum(retreat, 4.99))
    return jp.where(phase == 4, retreat, main), phase, complete


def reward_terms(old, new, f, action, previous_action, c=RewardSettings()):
    terms = v2.reward_terms(old, new, f, action, previous_action, c)
    main, phase, complete = main_score(new, f, c)
    invalid = terms['nonfinite'] | ~jp.all(jp.isfinite(jp.asarray(f[-6:])))
    reward = terms['reward'] - terms['main'] + main
    return dict(terms, main=main, section=phase,
                reward=jp.where(invalid, -1., reward), complete=complete & ~invalid,
                nonfinite=invalid, home_joint_progress=home_progress(f,c))
