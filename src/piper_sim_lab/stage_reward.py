"""SHOW-103 five-section reward mathematics; no simulator or training side effects.

Four reset groups are distinct from these five physical progress sections.
Call start_interval once per policy step, advance_memory for each measured
physics substep, then reward_terms once with the pre/post policy memories.
Inputs are true simulation measurements for rewards, not new actor observations.
"""
from dataclasses import dataclass, fields
import math
from typing import NamedTuple

import jax.numpy as jp


@dataclass(frozen=True)
class RewardSettings:
    physics_dt: float = .005
    policy_dt: float = .1
    approach_distance: float = .02
    reach_scale: float = 5.
    grasp_seconds: float = .1
    lift_height: float = .02
    bowl_entry_radius: float = .05
    alignment_radius: float = .017
    release_height_min: float = .05
    release_height_max: float = .06
    lower_scale: float = .04
    release_seconds: float = .02
    target_speed: float = .01
    target_seconds: float = .1
    support_speed: float = .02
    support_seconds: float = .1
    clearance: float = .08
    home_error: float = math.radians(3.)
    home_speed: float = .05
    home_seconds: float = .2
    home_progress_scale: float = 1.
    push_speed_scale: float = .05
    face_distance_scale: float = .03
    face_weight: float = .5
    xy_weight: float = .5
    dwell_weight: float = .5
    push_weight: float = .5
    recontact_weight: float = .5
    action_weight: float = .001
    action_rate_weight: float = .01

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{field.name} must be positive finite')
        if self.release_height_min >= self.release_height_max:
            raise ValueError('release height interval must be ordered')
        if abs(self.policy_dt / self.physics_dt - round(self.policy_dt / self.physics_dt)) > 1e-6:
            raise ValueError('policy_dt must contain an integral number of physics steps')


class Features(NamedTuple):
    """SI units; bowl positions/speeds are relative to its interior-floor frame.

    joint_error_a and joint_speed are maxima of absolute values over six joints.
    cube_table_contact excludes the bowl floor. opening_intent includes an
    already latched OPEN target while its physical stroke is still in progress.
    """
    tcp_cube_distance: float = .35
    cube_bottom_table: float = 0.
    cube_bowl_xy: float = .25
    cube_center_above_rim: float = -.042
    tcp_floor_height: float = .35
    tcp_relative_speed: float = 0.
    cube_relative_speed: float = 0.
    cube_horizontal_speed: float = 0.
    cube_downward_speed: float = 0.
    yaw_error: float = 0.
    clearance: float = 0.
    joint_error_a: float = 0.
    joint_speed: float = 0.
    left_contact: bool = False
    right_contact: bool = False
    bowl_floor_contact: bool = False
    cube_table_contact: bool = True
    robot_cube_contact: bool = False
    robot_bowl_contact: bool = False
    opening_intent: bool = False


class Memory(NamedTuple):
    grasp_time: float = 0.
    lifted: bool = False
    bowl_reached: bool = False
    lost_time: float = 0.
    release_candidate: bool = False
    candidate_height: float = 0.
    candidate_tcp_speed: float = 0.
    candidate_cube_speed: float = 0.
    released: bool = False
    qualified_release: bool = False
    invalid_release: bool = False
    support_time: float = 0.
    outside_time: float = 0.
    outside_drop: bool = False
    home_time: float = 0.
    target_time: float = 0.
    target_count: int = 0
    contact_window: bool = False
    impact_peak: float = 0.
    nonfinite_seen: bool = False


def reset_memory():
    # No synthetic grasp/landing timer or previous-episode achievement credit.
    return Memory(*(jp.asarray(v) for v in Memory()))


def start_interval(memory):
    """Clear only the 100ms contact accumulator, never episode reward ledgers."""
    return memory._replace(contact_window=jp.array(False))


def finite_features(f):
    return jp.all(jp.isfinite(jp.stack([jp.asarray(v, jp.float32) for v in f])))


def measured(f):
    return Features(*(jp.asarray(v) for v in f))


def advance_memory(old, before, after, c=RewardSettings()):
    """Consume a real 5ms transition. Speed limits do not veto OPEN/release."""
    old = Memory(*(jp.asarray(v) for v in old))
    before, after = measured(before), measured(after)
    dt = c.physics_dt
    both = after.left_contact & after.right_contact
    gripping = after.left_contact | after.right_contact
    grasp_time = jp.where(both & ~old.released, old.grasp_time + dt, 0.)
    grasped = grasp_time >= c.grasp_seconds - 1e-6
    lifted = old.lifted | (grasped & (after.cube_bottom_table >= c.lift_height))
    bowl_reached = old.bowl_reached | (
        lifted & gripping & (after.cube_bowl_xy <= c.bowl_entry_radius) &
        (after.cube_center_above_rim >= 0.))

    lost = lifted & ~gripping
    first_loss = lost & (old.lost_time == 0.) & ~old.released
    candidate_now = (
        (before.tcp_floor_height >= c.release_height_min) &
        (before.tcp_floor_height <= c.release_height_max) &
        (before.cube_bowl_xy <= c.alignment_radius) &
        (before.opening_intent | after.opening_intent))
    candidate = jp.where(first_loss, candidate_now, old.release_candidate)
    lost_time = jp.where(lost, old.lost_time + dt, 0.)
    release = lost & ~old.released & (lost_time >= c.release_seconds - 1e-6)
    released = old.released | release
    qualified = old.qualified_release | (release & candidate)
    invalid = old.invalid_release | (release & ~candidate)

    stable = after.bowl_floor_contact & (after.cube_relative_speed <= c.support_speed)
    support_time = jp.where(qualified & stable, old.support_time + dt, 0.)
    landed = support_time >= c.support_seconds - 1e-6
    outside_time = jp.where(
        released & after.cube_table_contact & ~after.bowl_floor_contact &
        (after.cube_relative_speed <= c.support_speed), old.outside_time + dt, 0.)
    outside_drop = old.outside_drop | (outside_time >= c.support_seconds - 1e-6)

    # Count even the first detached sample, not just samples after 20ms release.
    # Bilateral finger contact during a normal grasp is excluded by this gate.
    touch = (old.released | lost) & (after.robot_cube_contact | after.robot_bowl_contact)
    contact_window = old.contact_window | touch
    clear = (after.clearance >= c.clearance) & ~touch
    at_home = ((after.joint_error_a <= c.home_error) &
               (after.joint_speed <= c.home_speed))
    previously_landed = old.support_time >= c.support_seconds - 1e-6
    home_time = jp.where(landed & previously_landed & clear & at_home,
                         old.home_time + dt, 0.)

    target = (lifted & bowl_reached & gripping & ~released & ~invalid &
              (after.cube_bowl_xy <= c.alignment_radius) &
              (after.tcp_floor_height >= c.release_height_min) &
              (after.tcp_floor_height <= c.release_height_max) &
              (after.tcp_relative_speed <= c.target_speed))
    target_time = jp.where(target, old.target_time + dt, 0.)
    windows = jp.maximum(
        jp.floor((target_time + 1e-6) / c.target_seconds) -
        jp.floor((old.target_time + 1e-6) / c.target_seconds), 0.)
    count = jp.minimum(old.target_count + windows.astype(jp.int32), 2)
    impact = jp.maximum(old.impact_peak, jp.where(
        after.bowl_floor_contact & ~before.bowl_floor_contact,
        before.cube_downward_speed, 0.))
    return Memory(
        grasp_time, lifted, bowl_reached, lost_time, candidate,
        jp.where(first_loss, before.tcp_floor_height, old.candidate_height),
        jp.where(first_loss, before.tcp_relative_speed, old.candidate_tcp_speed),
        jp.where(first_loss, before.cube_relative_speed, old.candidate_cube_speed),
        released, qualified, invalid, support_time, outside_time, outside_drop,
        home_time, target_time, count, contact_window, impact,
        old.nonfinite_seen | ~finite_features(before) | ~finite_features(after))


def section(memory, f, c=RewardSettings()):
    f = measured(f)
    near = (f.tcp_cube_distance <= c.approach_distance) | f.left_contact | f.right_contact
    phase = jp.where(memory.qualified_release, 4,
                     jp.where(memory.lifted, jp.where(memory.bowl_reached, 3, 2),
                              jp.where(near, 1, 0)))
    return jp.where(memory.invalid_release | memory.outside_drop, -1, phase)


def main_score(memory, f, c=RewardSettings()):
    f = measured(f)
    phase = section(memory, f, c)
    grasp = jp.clip(memory.grasp_time / c.grasp_seconds, 0., 1.)
    lift = jp.clip(f.cube_bottom_table / c.lift_height, 0., 1.)
    approach = 1. - jp.tanh(c.reach_scale * jp.maximum(f.tcp_cube_distance, 0.))
    grasp_lift = 1. + .5 * grasp + .5 * grasp * lift
    transport = 2. + (1. - jp.tanh(c.reach_scale * jp.maximum(f.cube_bowl_xy, 0.)))
    error = jp.maximum(c.release_height_min - f.tcp_floor_height,
                       jp.maximum(f.tcp_floor_height - c.release_height_max, 0.))
    # Stop at 3.5 while holding; 4 requires a real, qualified detachment.
    prepare = 3. + .5 * (1. - jp.tanh(error / c.lower_scale))
    land_progress = jp.clip(memory.support_time / c.support_seconds, 0., 1.)
    clear_progress = jp.clip(f.clearance / c.clearance, 0., 1.)
    touch = memory.contact_window | f.robot_cube_contact | f.robot_bowl_contact
    settled = land_progress >= 1. - 1e-6
    home_progress = jp.clip(1. - f.joint_error_a / c.home_progress_scale, 0., 1.)
    # Before settling, hand clearance can still earn up to .25: withdrawal
    # need not wait motionless for the cube to land. A progress requires landing.
    retreat = (4. + .25 * land_progress + .25 * clear_progress * ~touch +
               .5 * home_progress * settled * (clear_progress >= 1.) * ~touch)
    complete = (memory.qualified_release & ~memory.outside_drop & settled & ~touch &
                (clear_progress >= 1.) & (memory.home_time >= c.home_seconds - 1e-6))
    # Getting close to A at high speed cannot equal a completed home hold.
    retreat = jp.where(complete, 5., jp.minimum(retreat, 4.99))
    score = jp.where(phase == 0, approach, jp.where(phase == 1, grasp_lift,
        jp.where(phase == 2, transport, jp.where(phase == 3, prepare, retreat))))
    return jp.where(phase < 0, 0., score), phase, complete & (phase == 4)


def reward_terms(old, new, f, action, previous_action, c=RewardSettings()):
    """One 100ms reward. No +5/+2 legacy score or extra event reward is added."""
    old = Memory(*(jp.asarray(v) for v in old))
    new = Memory(*(jp.asarray(v) for v in new))
    f = measured(f)
    a = jp.asarray(action, jp.float32)
    previous = jp.asarray(previous_action, jp.float32)
    if a.shape != (7,) or previous.shape != (7,):
        raise ValueError('expected seven normalized current/previous policy actions')
    invalid = (new.nonfinite_seen | ~finite_features(f) |
               ~jp.all(jp.isfinite(a)) | ~jp.all(jp.isfinite(previous)))
    a = jp.clip(jp.nan_to_num(a), -1., 1.)
    previous = jp.clip(jp.nan_to_num(previous), -1., 1.)
    main, phase, complete = main_score(new, f, c)
    grasp_fraction = jp.clip(new.grasp_time / c.grasp_seconds, 0., 1.)
    before_grasp = (phase >= 0) & (phase <= 1) & ~new.released
    align = .5 * (1. + jp.cos(4. * f.yaw_error))
    # Fade the auxiliary into the equal .5 grasp-confirmation credit, so
    # confirming a good grasp does not lose a .5 face bonus in one step.
    face = (c.face_weight * align * jp.exp(-jp.maximum(f.tcp_cube_distance, 0.) /
            c.face_distance_scale) * (1. - grasp_fraction) * before_grasp)
    push = (c.push_weight * jp.clip(f.cube_horizontal_speed / c.push_speed_scale, 0., 1.) *
            before_grasp * (grasp_fraction < 1.))
    xy = c.xy_weight * ((phase == 2) | (phase == 3)) * (
        f.cube_bowl_xy <= c.alignment_radius) * ~new.released
    dwell = c.dwell_weight * jp.maximum(new.target_count - old.target_count, 0)
    recontact = c.recontact_weight * new.contact_window
    action_cost = c.action_weight * jp.sum(a ** 2)
    rate_cost = c.action_rate_weight * jp.sum((a - previous) ** 2)
    value = main + face + xy + dwell - push - recontact - action_cost - rate_cost
    return dict(reward=jp.where(invalid, -1., value), main=main, section=phase,
                face_alignment=face, xy_alignment=xy, dwell_bonus=dwell,
                cube_push_cost=push, recontact_cost=recontact,
                action_cost=action_cost, action_rate_cost=rate_cost,
                complete=complete & ~invalid, nonfinite=invalid)
