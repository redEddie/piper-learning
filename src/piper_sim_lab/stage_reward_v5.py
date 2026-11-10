"""Progress-based five-stage reward with latched completed-stage credit.

No one-shot completion reward is added. On transition the achieved progress is
stored in the next baseline. Only explicit unwanted actions incur penalties.
All reference distances are frozen at stage entry (or physical reset end).
"""
from collections import namedtuple
from dataclasses import dataclass
import math

import jax
import jax.numpy as jp

from . import stage_reward_v4 as v4

Features=namedtuple('Features',v4.Features._fields+('finger_width','cube_grasp_width','face_angle'),
    defaults=tuple(v4.Features())+(.05,.05,0.))
_EXTRA=('approach_complete','grasp_complete','transport_complete','place_complete','home_complete',
        'approach_gap_start','transport_xy_start','transport_height_start',
        'opening_width_start','opening_width_goal','home_distance_start','place_quality')
_DEFAULTS=(False,False,False,False,False,-1.,-1.,-1.,-1.,-1.,-1.,0.)
Memory=namedtuple('Memory',v4.Memory._fields+_EXTRA,defaults=tuple(v4.Memory())+_DEFAULTS)


@dataclass(frozen=True)
class RewardSettings(v4.RewardSettings):
    home_error: float = .2
    face_angle_cap: float = math.pi/4
    closed_weight: float = .1
    opening_margin: float = .01
    max_gripper_width: float = .1
    home_max_weight: float = .7
    home_mean_weight: float = .3

    def __post_init__(self):
        super().__post_init__()
        if abs(self.home_max_weight+self.home_mean_weight-1.)>1e-6:
            raise ValueError('home distance weights must sum to one')


def reset_memory():
    return Memory(*(jp.asarray(x) for x in Memory()))


def start_interval(memory):
    return memory._replace(contact_window=jp.array(False))


def physical(memory):
    return v4.Memory(*memory[:len(v4.Memory._fields)])


def section(memory):
    phase=jp.where(memory.place_complete,4,jp.where(memory.transport_complete,3,
        jp.where(memory.grasp_complete,2,jp.where(memory.approach_complete,1,0))))
    return jp.where(memory.invalid_release | memory.outside_drop,-1,phase)


def home_distance(f,c=RewardSettings()):
    errors=jp.maximum(jp.asarray(f[21:27])-c.home_error,0.)
    return c.home_max_weight*jp.max(errors)+c.home_mean_weight*jp.mean(errors)


def fraction(remaining,start):
    return jp.where(start>1e-6,jp.clip(1.-remaining/jp.maximum(start,1e-6),0.,1.),
                    jp.where(remaining<=1e-6,1.,0.))


def initialize(memory,f,c=RewardSettings()):
    """Use the actual post-settle observation as the current stage reference."""
    phase=section(memory)
    return memory._replace(
        approach_gap_start=jp.where(phase==0,jp.maximum(f.tcp_cube_distance-c.approach_distance,0.),memory.approach_gap_start),
        transport_xy_start=jp.where(phase==2,jp.maximum(f.cube_bowl_xy-c.bowl_entry_radius,0.),memory.transport_xy_start),
        transport_height_start=jp.where(phase==2,jp.maximum(-f.cube_center_above_rim,0.),memory.transport_height_start),
        opening_width_start=jp.where(phase==3,f.finger_width,memory.opening_width_start),
        opening_width_goal=jp.where(phase==3,jp.minimum(f.cube_grasp_width+c.opening_margin,c.max_gripper_width),memory.opening_width_goal),
        home_distance_start=jp.where(phase==4,home_distance(f,c),memory.home_distance_start))


def advance_memory(old,before,after,c=RewardSettings()):
    old=Memory(*(jp.asarray(x) for x in old))
    before=Features(*(jp.asarray(x) for x in before));after=Features(*(jp.asarray(x) for x in after))
    base=v4.advance_memory(physical(old),v4.Features(*before[:30]),v4.Features(*after[:30]),c)
    approach=old.approach_complete | (after.tcp_cube_distance<=c.approach_distance) | after.left_contact | after.right_contact | base.lifted
    grasp=old.grasp_complete | base.lifted
    transport=old.transport_complete | base.bowl_reached
    open_enter=transport & ((~old.transport_complete) | (old.opening_width_start<0.))
    width_start=jp.where(open_enter,jp.where(old.transport_complete,before.finger_width,after.finger_width),old.opening_width_start)
    width_goal=jp.where(open_enter,jp.minimum(
        jp.where(old.transport_complete,before.cube_grasp_width,after.cube_grasp_width)+c.opening_margin,c.max_gripper_width),old.opening_width_goal)
    settled=base.support_time>=c.support_seconds-1e-6
    opened=after.finger_width>=width_goal-1e-5
    place=old.place_complete | (transport & base.qualified_release & after.cube_inside_bowl & settled & opened & ~base.high_drop_success)
    quality=jp.where(place & ~old.place_complete,
        jp.where(base.freefall_height<=c.low_drop_height+1e-6,1.,.5),old.place_quality)
    home=old.home_complete | (place & (base.home_time>=c.home_seconds-1e-6) & settled &
        (after.joint_error_a<=c.home_error) & (after.joint_speed<=c.home_speed) &
        (after.clearance>=c.clearance) & ~base.contact_window)
    entering_transport=grasp & ((~old.grasp_complete) | (old.transport_xy_start<0.))
    transport_f=jp.where(old.grasp_complete,before.cube_bowl_xy,after.cube_bowl_xy)
    transport_h=jp.where(old.grasp_complete,before.cube_center_above_rim,after.cube_center_above_rim)
    entering_home=place & ((~old.place_complete) | (old.home_distance_start<0.))
    result=Memory(*base,approach,grasp,transport,place,home,
        jp.where(old.approach_gap_start<0.,jp.maximum(before.tcp_cube_distance-c.approach_distance,0.),old.approach_gap_start),
        jp.where(entering_transport,jp.maximum(transport_f-c.bowl_entry_radius,0.),old.transport_xy_start),
        jp.where(entering_transport,jp.maximum(-transport_h,0.),old.transport_height_start),
        width_start,width_goal,
        jp.where(entering_home,home_distance(after,c),old.home_distance_start),quality)
    return result


def main_score(memory,f,c=RewardSettings()):
    phase=section(memory)
    approach=fraction(jp.maximum(f.tcp_cube_distance-c.approach_distance,0.),memory.approach_gap_start)
    g=jp.clip(memory.grasp_time/c.grasp_seconds,0.,1.)
    lift=jp.clip(f.cube_bottom_table/c.lift_height,0.,1.)
    grasp=.5*g+.5*g*lift
    xy=fraction(jp.maximum(f.cube_bowl_xy-c.bowl_entry_radius,0.),memory.transport_xy_start)
    height=fraction(jp.maximum(-f.cube_center_above_rim,0.),memory.transport_height_start)
    carry=.8*xy+.2*height
    opening=jp.clip((f.finger_width-memory.opening_width_start)/jp.maximum(
        memory.opening_width_goal-memory.opening_width_start,1e-6),0.,1.)
    settled=jp.clip(memory.support_time/c.support_seconds,0.,1.)
    low=memory.first_floor_contact & (memory.freefall_height<=c.low_drop_height+1e-6)
    place=.5*opening+.5*settled*low
    home=.8*fraction(home_distance(f,c),memory.home_distance_start)+.2*jp.clip(memory.home_time/c.home_seconds,0.,1.)
    base=jp.where(phase==0,0.,jp.where(phase==1,1.,jp.where(phase==2,2.,
        jp.where(phase==3,3.,3.+memory.place_quality))))
    progress=jp.where(phase==0,approach,jp.where(phase==1,grasp,jp.where(phase==2,carry,jp.where(phase==3,place,home))))
    base=jp.where(memory.home_complete,4.+memory.place_quality,base)
    progress=jp.where(memory.home_complete,0.,progress)
    valid=phase>=0
    return jp.where(valid,base+progress,0.),phase,(memory.home_complete | memory.high_drop_success) & valid,base,progress


def reward_terms(old,new,f,action,previous_action,c=RewardSettings()):
    f=Features(*(jp.asarray(x) for x in f))
    main,phase,complete,base,progress=main_score(new,f,c)
    action=jp.asarray(action,jp.float32);previous=jp.asarray(previous_action,jp.float32)
    invalid=(new.nonfinite_seen | ~jp.all(jp.isfinite(jp.asarray(f,jp.float32))) |
        ~jp.all(jp.isfinite(action)) | ~jp.all(jp.isfinite(previous)))
    action=jp.clip(jp.nan_to_num(action),-1.,1.);previous=jp.clip(jp.nan_to_num(previous),-1.,1.)
    face=c.face_weight*jp.clip(f.face_angle/c.face_angle_cap,0.,1.)*((phase==0)|(phase==1))
    closed=c.closed_weight*(phase==3)*~f.opening_intent
    carrying=(f.left_contact | f.right_contact) & ~new.released & ((new.grasp_time>=c.grasp_seconds-1e-6)|new.lifted)
    delta=jp.maximum(f.cube_bowl_xy-new.best_carry_xy,0.)
    away=jp.where(carrying & (delta>=c.away_distance-1e-6),c.away_weight*delta/c.away_distance,0.)
    failed=new.invalid_release | new.outside_drop
    old_failed=old.invalid_release | old.outside_drop
    drop=c.drop_weight*(failed & ~old_failed)
    recontact=c.recontact_weight*new.contact_window
    action_cost=c.action_weight*jp.sum(action**2)
    rate_cost=c.action_rate_weight*jp.sum((action-previous)**2)
    reward=main-face-closed-away-drop-recontact-action_cost-rate_cost
    return dict(reward=jp.where(invalid,-1.,reward),main=main,section=phase,
        completed_credit=base,progress=progress,face_cost=face,closed_cost=closed,
        face_alignment=jp.array(0.),xy_alignment=jp.array(0.),dwell_bonus=jp.array(0.),cube_push_cost=jp.array(0.),
        opening_bonus=jp.array(0.),landing_bonus=jp.array(0.),away_cost=away,drop_cost=drop,recontact_cost=recontact,
        action_cost=action_cost,action_rate_cost=rate_cost,complete=complete & ~invalid,nonfinite=invalid,
        terminal_success=new.high_drop_success & ~invalid,freefall_height=new.freefall_height,
        approach_complete=new.approach_complete,grasp_complete=new.grasp_complete,
        transport_complete=new.transport_complete,place_complete=new.place_complete,home_complete=new.home_complete,
        home_joint_progress=fraction(home_distance(f,c),new.home_distance_start),home_distance=home_distance(f,c),
        best_carry_xy=jp.where(jp.isfinite(new.best_carry_xy),new.best_carry_xy,f.cube_bowl_xy))


@jax.jit
def replay_prefixes(rows,lengths,bowl_height,rim_radius,bottom_radius):
    base=v4.replay_prefixes(rows,lengths,bowl_height,rim_radius,bottom_radius)
    extras=tuple(jp.full(base.released.shape,x,dtype=jp.bool_ if isinstance(x,bool) else jp.float32) for x in _DEFAULTS)
    return Memory(*base,*extras)._replace(approach_complete=base.lifted,grasp_complete=base.lifted,
        transport_complete=base.bowl_reached)
