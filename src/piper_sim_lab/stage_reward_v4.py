"""Physical release/landing rewards; independent home progress, unchanged home goal.

Keep v1-v3 reproducible. The new five-start task enables this module explicitly.
Drop height measures the cube centre's descent from its free-flight peak to
first bowl-floor support, in the bowl frame; it is not TCP height.
"""
from collections import namedtuple
from dataclasses import dataclass

import jax
import jax.numpy as jp

from . import stage_reward as v1, stage_reward_v2 as v2, stage_reward_v3 as v3

Features=namedtuple('Features',v3.Features._fields+(
    'cube_center_floor_height','cube_over_opening','cube_inside_bowl'),
    defaults=tuple(v3.Features())+(.1,True,False))
Memory=namedtuple('Memory',v2.Memory._fields+(
    'release_center_height','fall_peak_height','first_floor_contact','freefall_height','high_drop_success'),
    defaults=tuple(v2.Memory())+(0.,0.,False,0.,False))


@dataclass(frozen=True)
class RewardSettings(v2.RewardSettings):
    drop_weight: float = 2.
    low_drop_height: float = .07
    high_drop_height: float = .10
    opening_weight: float = .5
    low_landing_weight: float = 1.
    middle_landing_weight: float = .5

    def __post_init__(self):
        super().__post_init__()
        if self.low_drop_height >= self.high_drop_height:
            raise ValueError('drop height thresholds must be ordered')


def reset_memory():
    return Memory(*(jp.asarray(v) for v in Memory()))


def start_interval(memory):
    return memory._replace(contact_window=jp.array(False))


def legacy(memory):
    return v2.Memory(*memory[:len(v2.Memory._fields)])._replace(prepare_time=jp.array(0.))


def advance_memory(old,before,after,c=RewardSettings()):
    old=Memory(*(jp.asarray(v) for v in old))
    before=Features(*(jp.asarray(v) for v in before))
    after=Features(*(jp.asarray(v) for v in after))
    base=v2.advance_memory(legacy(old),v2.Features(*before[:21]),v2.Features(*after[:21]),c)
    gripping=after.left_contact | after.right_contact
    lost=base.lifted & ~gripping
    first_loss=lost & (old.lost_time == 0.) & ~old.released
    release_height=jp.where(first_loss,before.cube_center_floor_height,old.release_center_height)
    peak=jp.where(first_loss,before.cube_center_floor_height,old.fall_peak_height)
    peak=jp.where(lost & ~old.first_floor_contact,jp.maximum(peak,after.cube_center_floor_height),peak)
    tentative_reset=gripping & ~old.released
    had_floor=old.first_floor_contact & ~tentative_reset
    first_support=lost & after.bowl_floor_contact & ~had_floor
    floor_contact=had_floor | first_support
    height=jp.where(tentative_reset,0.,jp.where(first_support,
        jp.maximum(peak-after.cube_center_floor_height,0.),old.freefall_height))
    released=base.released
    # Floor/side contacts can overlap for a penetrating/tilting cube during
    # landing. A real miss is a centre outside the bowl catchment. Await full
    # containment for credit rather than calling transient contacts a miss.
    miss=released & ~after.cube_over_opening
    invalid=old.invalid_release | miss
    outside=old.outside_drop | miss
    qualified=old.qualified_release | (released & after.cube_inside_bowl & after.bowl_floor_contact)
    stable=after.bowl_floor_contact & (after.cube_relative_speed <= c.support_speed)
    support=jp.where(qualified & stable,old.support_time+c.physics_dt,0.)
    landed=support >= c.support_seconds-1e-6
    previously_landed=old.support_time >= c.support_seconds-1e-6
    no_touch=~base.contact_window
    clear=(after.clearance >= c.clearance) & no_touch
    at_home=(after.joint_error_a <= c.home_error) & (after.joint_speed <= c.home_speed)
    home_time=jp.where(landed & previously_landed & clear & at_home,old.home_time+c.physics_dt,0.)
    high=(old.high_drop_success | (qualified & floor_contact &
          (height >= c.high_drop_height-1e-6))) & ~invalid & ~outside
    base=base._replace(qualified_release=qualified,invalid_release=invalid,outside_drop=outside,
        support_time=support,home_time=home_time,prepare_time=jp.array(0.),
        target_time=jp.array(0.),target_count=jp.array(0,jp.int32))
    return Memory(*base,release_height,peak,floor_contact,height,high)


def main_score(memory,f,c=RewardSettings()):
    core=v3.Features(*f[:27])
    main,phase,home_complete=v3.main_score(legacy(memory),core,c)
    land=jp.clip(memory.support_time/c.support_seconds,0.,1.)
    clear=jp.clip(f.clearance/c.clearance,0.,1.)
    touch=memory.contact_window | f.robot_cube_contact | f.robot_bowl_contact
    # Home credit no longer depends on clearance or withdrawal. Recontact is
    # penalized separately; final home completion retains ALL original gates.
    retreat=(4.+.25*land+.25*clear*~touch+
             .5*v3.home_progress(core,c)*(land >= 1.-1e-6))
    retreat=jp.where(home_complete,5.,jp.minimum(retreat,4.99))
    main=jp.where(phase==4,retreat,main)
    high=memory.high_drop_success & ~memory.invalid_release & ~memory.outside_drop
    return main,phase,home_complete | high


def reward_terms(old,new,f,action,previous_action,c=RewardSettings()):
    old=Memory(*(jp.asarray(v) for v in old));new=Memory(*(jp.asarray(v) for v in new))
    f=Features(*(jp.asarray(v) for v in f))
    terms=v3.reward_terms(legacy(old),legacy(new),v3.Features(*f[:27]),action,previous_action,c)
    main,phase,complete=main_score(new,f,c)
    invalid=terms['nonfinite'] | ~jp.all(jp.isfinite(jp.asarray(f,jp.float32)))
    detached=new.released & ~(f.left_contact | f.right_contact)
    opening=c.opening_weight*detached*f.opening_intent*(phase >= 3)
    inside=new.qualified_release & new.first_floor_contact & f.cube_inside_bowl
    # <=7cm gets1, (7cm,10cm) gets0.5, >=10cm gets0 and successful termination.
    landing=jp.where(new.freefall_height <= c.low_drop_height+1e-6,c.low_landing_weight,
        jp.where(new.freefall_height < c.high_drop_height-1e-6,c.middle_landing_weight,0.))*inside
    reward=terms['reward']-terms['main']-terms['dwell_bonus']+main+opening+landing
    return dict(terms,main=main,section=phase,reward=jp.where(invalid,-1.,reward),
        dwell_bonus=jp.array(0.),opening_bonus=opening,landing_bonus=landing,
        freefall_height=new.freefall_height,terminal_success=new.high_drop_success & ~invalid,
        complete=complete & ~invalid,nonfinite=invalid,
        home_joint_progress=v3.home_progress(v3.Features(*f[:27]),c))


@jax.jit
def replay_prefixes(rows,lengths,bowl_height,rim_radius,bottom_radius):
    """Replay the original airborne native prefix under the new memory rules.

    Callers verify there were no bowl-floor contacts in these prefixes. Thus
    inside=False is exact, centre height and opening XY are stored measurements,
    and six omitted joint errors do not participate in any memory transition.
    """
    def replay(record,length):
        centre=record[:,3]+bowl_height
        radius=bottom_radius+(rim_radius-bottom_radius)*jp.clip(centre/bowl_height,0.,1.)
        fs=Features(*(record[:,i]>.5 if 13<=i<20 else record[:,i] for i in range(21)),
            *(jp.zeros(record.shape[0]) for _ in range(6)),centre,
            record[:,2]<=radius,jp.zeros(record.shape[0],bool))
        initial=jax.tree.map(lambda a:a[0],fs)
        def step(carry,i):
            memory,before=carry;after=jax.tree.map(lambda a:a[i],fs)
            updated=advance_memory(memory,before,after)
            memory=jax.tree.map(lambda a,b:jp.where(i<length,a,b),updated,memory)
            return (memory,after),None
        return jax.lax.scan(step,(reset_memory(),initial),jp.arange(1,record.shape[0]))[0][0]
    return jax.vmap(replay)(rows,lengths)
