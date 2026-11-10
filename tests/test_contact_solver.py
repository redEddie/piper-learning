"""Regression for the held-block ejection caused by a low line-search budget.

Exercise actual C-engine contact dynamics, without a policy or GPU. Both
block sizes previously ejected under the task's old default budget of 4.
"""
import mujoco
import numpy as np
import pytest

from piper_sim_lab.pick_place import PiperPickPlace


@pytest.mark.parametrize("half,mass", [(0.015, 0.02), (0.025, 0.08)])
def test_default_solver_holds_block_without_ejection(half, mass):
    env = PiperPickPlace(config_overrides={
        "impl": "jax", "block_half": half, "block_mass": mass,
    })
    model = env.mj_model
    data = mujoco.MjData(model)
    joints = np.asarray(env._reset_table["grasp"])[0]
    data.qpos[env._qpos_ids] = joints
    data.qpos[env._finger_qpos_ids] = [half, -half]
    data.ctrl[env._arm_ctrl_ids] = joints
    data.ctrl[env._gripper_ctrl_id] = 0.0
    mujoco.mj_kinematics(model, data)
    axis = data.site_xmat[env._tcp_id].reshape(3, 3)[:, 1]
    yaw = np.arctan2(axis[1], axis[0])
    adr = env._block_qpos_adr
    data.qpos[adr:adr + 7] = [
        *data.site_xpos[env._tcp_id], np.cos(yaw / 2), 0, 0, np.sin(yaw / 2),
    ]
    peak_speed = 0.0
    for _ in range(round(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        velocity = data.qvel[env._block_qvel_adr:env._block_qvel_adr + 3]
        peak_speed = max(peak_speed, float(np.linalg.norm(velocity)))
    mujoco.mj_kinematics(model, data)
    block = data.qpos[adr:adr + 3]
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert peak_speed < 0.3, "held block acquired an implausible transient speed"
    assert np.linalg.norm(block - data.site_xpos[env._tcp_id]) < 0.01
    assert block[2] > 0.05, "block fell from the lifted starting pose"
