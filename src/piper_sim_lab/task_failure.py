"""Simulation-only task termination labels, sampled at each physics substep.

Contact labels are not actor observations. Controller faults are deliberately
absent here: they remain interruptions of the rollout, not failed-task events.
"""
import jax.numpy as jp


DEFAULTS = dict(task_grasp_deadline=12., task_grasp_confirm_seconds=.1,
                task_drop_confirm_seconds=.1, task_drop_stable_speed=.02)
GRASP_TIMEOUT = 1
OUTSIDE_DROP = 2


def task_failure_memory(base, stage):
    z = jp.array(0., jp.float32)
    return dict(base, tf_elapsed=z, tf_grasp_time=z, tf_landing_time=z,
                tf_requires_grasp=stage < 2, tf_grasped=base['carried'],
                tf_lifted=base['carried'], tf_failed=jp.array(False),
                tf_reason=jp.array(0, jp.int32))


def task_failure_update(old, new, f, dt, settings):
    both = f['left'] & f['right']
    grasp_time = jp.where(both, old['tf_grasp_time'] + dt, 0.)
    grasped = old['tf_grasped'] | (grasp_time >= settings['task_grasp_confirm_seconds'] - 1e-6)
    lifted = old['tf_lifted'] | (grasped & both & f['elevated'] & ~f['table'])
    # A cube initially resting on the table is not a dropped cube. Normal
    # release flight has no table contact; a bowl-floor landing has support.
    landed_outside = (lifted & ~f['left'] & ~f['right'] & f['table'] &
                      ~f['support'] & (f['table_speed'] <= settings['task_drop_stable_speed']))
    landing_time = jp.where(landed_outside, old['tf_landing_time'] + dt, 0.)
    elapsed = old['tf_elapsed'] + dt
    timeout = (old['tf_requires_grasp'] & ~grasped &
               (elapsed >= settings['task_grasp_deadline'] - 1e-5))
    dropped = landing_time >= settings['task_drop_confirm_seconds'] - 1e-6
    reason = jp.where(timeout, GRASP_TIMEOUT, jp.where(dropped, OUTSIDE_DROP, 0))
    return dict(new, tf_elapsed=elapsed, tf_grasp_time=grasp_time,
                tf_grasped=grasped, tf_lifted=lifted, tf_landing_time=landing_time,
                tf_failed=old['tf_failed'] | timeout | dropped,
                tf_reason=jp.where(old['tf_failed'], old['tf_reason'], reason))
