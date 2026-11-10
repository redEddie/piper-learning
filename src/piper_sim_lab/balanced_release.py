"""SHOW-103 agreed reward revision; opt-in to preserve previous experiments."""
import jax.numpy as jp


def proximity(f, c):
    error = jp.maximum(c.release_tcp_min - f['tcp_height'],
                       jp.maximum(f['tcp_height'] - c.release_tcp_max, 0.))
    return ((1. - jp.tanh(error / c.gentle_lower_scale)) *
            jp.exp(-(f['xy'] / c.gentle_alignment_scale) ** 2))


def eligible(t, c):
    return (t['qualified_release'] & ~t['premature'] & ~t['recontact_seen'] &
            (t['impact_peak'] <= c.release_impact_limit))


def update_credits(old, new, f, c):
    # Every ledger survives contact loss, regrasp and returning to a prior pose.
    held = (new['active'] & new['carried'] & ~new['released'] &
            ~new['premature'] & (f['left'] | f['right']))
    best = jp.maximum(old['progress_best'], jp.where(held, proximity(f, c), 0.))
    progress = old['progress_credit'] + 3. * (best - old['progress_best'])
    old_windows = jp.floor((old['target_time'] + 1e-6) / c.place_support_seconds)
    new_windows = jp.floor((new['target_time'] + 1e-6) / c.place_support_seconds)
    windows = jp.where(held, jp.maximum(new_windows - old_windows, 0.), 0.)
    count = jp.minimum(old['target_count'] + windows, 2.)
    valid = eligible(new, c)
    release_paid = old['release_paid'] | valid
    landing_paid = old['landing_paid'] | (valid & new['supported'])
    withdrawal_paid = old['withdrawal_paid'] | (
        valid & new['supported'] &
        (new['clear_time'] >= c.gentle_clear_seconds - 1e-6))
    return dict(new, progress_best=best, progress_credit=progress,
                target_count=count, release_paid=release_paid,
                landing_paid=landing_paid, withdrawal_paid=withdrawal_paid)


def reward(t, f, p, c):
    hold = t['carried'] & ~t['released'] & (f['left'] | f['right'])
    holding = 1. - jp.tanh(c.reach_scale * p['tcp_block'])
    holding += 2. - jp.tanh(c.reach_scale * f['xy'])
    quality = eligible(t, c).astype(jp.float32)
    # Legacy geometric completion is retained only for ordinary drop credit.
    basic = 1.5 * p['placed'] + p['inside']
    flight = t['released'] & (t['release_time'] <= c.release_flight_seconds)
    stable = t['supported'] & f['support'] & (f['speed'] <= c.place_stable_speed)
    valid_progress = flight | stable
    released = jp.where(quality > 0., 4. * valid_progress, basic)
    retreat = (jp.clip(f['clearance'] / c.gentle_clearance, 0., 1.) *
               quality * stable * ~f['robot_cube'] * ~f['robot_bowl'])
    score = jp.where(hold & ~t['premature'], holding, released + retreat)
    active = t['active'] & (hold | valid_progress)
    return score, active, quality, retreat


def bonuses(old, new):
    return dict(
        progress=new['progress_credit'] - old['progress_credit'],
        dwell=.5 * (new['target_count'] - old['target_count']),
        release=2. * (new['release_paid'].astype(jp.float32) - old['release_paid']),
        landing=2. * (new['landing_paid'].astype(jp.float32) - old['landing_paid']),
        withdrawal=3. * (new['withdrawal_paid'].astype(jp.float32) - old['withdrawal_paid']))
