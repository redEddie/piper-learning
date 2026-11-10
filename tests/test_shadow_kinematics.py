"""Portable FK/IK parity against the C engine and actual training controller."""
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from piper_policy_runtime.kinematics import PiperKinematics, quaternion_to_rotvec


ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "src/piper_sim_lab/assets/agilex_piper/piper.xml"


def c_model(xml=XML):
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.body("link6").add_site(name="parity_tcp", pos=[0, 0, .12], size=[.001]*3)
    model = spec.compile()
    return model, mujoco.MjData(model)


def c_values(model, data, q):
    joint_ids = np.array([model.joint(f"joint{i}").id for i in range(1, 7)])
    dofs = model.jnt_dofadr[joint_ids]
    data.qpos[model.jnt_qposadr[joint_ids]] = q
    mujoco.mj_forward(model, data)
    site = model.site("parity_tcp").id
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, linear, angular, site)
    return (data.site_xpos[site].copy(), data.xquat[model.body("link6").id].copy(),
            np.r_[linear, angular][:, dofs])


def test_fk_quaternion_sign_and_analytic_jacobian_match_c_engine():
    kinematics = PiperKinematics(XML)
    model, data = c_model()
    random = np.random.default_rng(72)
    states = np.r_[np.zeros((1, 6)), kinematics.lower[None], kinematics.upper[None],
                   random.uniform(kinematics.lower, kinematics.upper, (80, 6))]
    signs = []
    for q in states:
        position, quaternion, jacobian = c_values(model, data, q)
        actual = kinematics.forward(q)
        np.testing.assert_allclose(actual["tcp_position"], position, atol=2e-14)
        # Component-wise comparison catches an R->q conversion flipping sign.
        np.testing.assert_allclose(actual["link6_quaternion_wxyz"], quaternion, atol=2e-14)
        np.testing.assert_allclose(kinematics.jacobian(q), jacobian, atol=2e-14)
        signs.append(quaternion[0])
    assert min(signs) < -.1 and max(signs) > .1


def test_vendor_v1_tcp_base_frame_and_jacobian_match_c_engine():
    xml = ROOT / 'src/piper_sim_lab/assets/agilex_piper_v100/piper.xml'
    kinematics = PiperKinematics(xml)
    np.testing.assert_array_equal(kinematics.tcp_local_position, [0, 0, .1358])
    assert kinematics.robot_model == 'piper_v100'
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.body('base_link').pos = [0, 0, .015]
    spec.body('link6').add_site(name='parity_tcp', pos=[0, 0, .1358], size=[.001]*3)
    model = spec.compile()
    data = mujoco.MjData(model)
    random = np.random.default_rng(76)
    for q in random.uniform(kinematics.lower, kinematics.upper, (100, 6)):
        position, quaternion, jacobian = c_values(model, data, q)
        actual = kinematics.forward(q)
        np.testing.assert_allclose(actual['tcp_position'], position - [0, 0, .015], atol=2e-14)
        np.testing.assert_allclose(actual['link6_quaternion_wxyz'], quaternion, atol=2e-14)
        np.testing.assert_allclose(kinematics.jacobian(q), jacobian, atol=2e-14)


def test_vendor_mount_translation_is_rejected(tmp_path):
    xml = ROOT / 'src/piper_sim_lab/assets/agilex_piper_v100/piper.xml'
    tree = ET.parse(xml)
    tree.getroot().find(".//body[@name='mount']").set('pos', '0 0 .1')
    modified = tmp_path / 'wrong-mount.xml'
    tree.write(modified)
    with pytest.raises(ValueError, match='mount must have an identity'):
        PiperKinematics(modified)


def test_nonzero_joint_pivots_refs_and_nonunit_xml_quaternions(tmp_path):
    root = ET.parse(XML).getroot()
    root.find("compiler").set("meshdir", str(XML.parent/"assets"))
    random = np.random.default_rng(61)
    for i in range(1, 7):
        body = root.find(f".//body[@name='link{i}']")
        original = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        body.set("quat", " ".join(map(str, original*2.7)))
        joint = body.find("joint")
        joint.set("pos", " ".join(map(str, random.uniform(-.01, .01, 3))))
        joint.set("axis", " ".join(map(str, random.normal(size=3))))
        joint.set("ref", str(random.uniform(-.2, .2)))
    modified = tmp_path/"modified.xml"
    ET.ElementTree(root).write(modified)
    kinematics = PiperKinematics(modified)
    model, data = c_model(modified)
    for q in random.uniform(kinematics.lower, kinematics.upper, (20, 6)):
        position, quaternion, jacobian = c_values(model, data, q)
        actual = kinematics.forward(q)
        np.testing.assert_allclose(actual["tcp_position"], position, atol=2e-14)
        np.testing.assert_allclose(actual["link6_quaternion_wxyz"], quaternion, atol=2e-14)
        np.testing.assert_allclose(kinematics.jacobian(q), jacobian, atol=2e-14)


def test_proposals_match_training_joint_targets_and_preserve_target_memory():
    # Call the real controller using the C engine's TCP Jacobian.  No physics,
    # environment construction, training or GPU kernels are needed.
    import jax.numpy as jnp
    from piper_sim_lab.pick_place import PiperPickPlace, _quat_to_rotvec

    kinematics = PiperKinematics(XML)
    model, data = c_model()
    random = np.random.default_rng(91)
    parameters = dict(ctrl_dt=.1, max_lin_vel=.10, max_ang_vel=1.,
                      ik_damping=.05, max_joint_vel=1.5, max_target_lag=.15)
    context = SimpleNamespace(_config=SimpleNamespace(**parameters),
                              _qpos_ids=np.arange(6), _arm_ctrl_ids=np.arange(6),
                              _lower=jnp.asarray(kinematics.lower),
                              _upper=jnp.asarray(kinematics.upper))
    for _ in range(25):
        q = random.uniform(kinematics.lower, kinematics.upper)
        current_target = q + random.uniform(-.3, .3, 6)
        action = random.uniform(-2., 2., 7)
        _, quaternion, jacobian = c_values(model, data, q)
        context._tcp_jacobian = lambda _: jnp.asarray(jacobian)
        state = SimpleNamespace(qpos=jnp.asarray(q), ctrl=jnp.asarray(current_target))
        expected = PiperPickPlace._joint_targets(context, state, jnp.clip(jnp.asarray(action), -1., 1.))
        actual = kinematics.propose(q, current_target, action, **parameters)
        np.testing.assert_allclose(actual["joint_target_rad"], expected, atol=8e-7, rtol=3e-6)
        np.testing.assert_allclose(quaternion_to_rotvec(quaternion), _quat_to_rotvec(jnp.asarray(quaternion)), atol=8e-7)
        assert not actual["hardware_commands_sent"]
        assert np.max(np.abs(actual["joint_velocity_radps"])) <= parameters["max_joint_vel"]
    # A zero velocity command retains the servo target within its anti-windup
    # band. Replacing this with q + dq would change the trained controller.
    q = (kinematics.lower+kinematics.upper)/2
    target = q + .04
    result = kinematics.propose(q, target, np.zeros(7))
    np.testing.assert_allclose(result["joint_target_rad"], target)


def test_gripper_binary_zero_closes_and_width_is_twice_finger_travel():
    kinematics = PiperKinematics(XML)
    q = (kinematics.lower+kinematics.upper)/2
    action = np.zeros(7)
    closed = kinematics.propose(q, q, action, current_gripper_finger_target=.025)
    assert closed["gripper_finger_target_m"] == pytest.approx(.015)
    assert closed["gripper_width_target_m"] == pytest.approx(.03)
    action[6] = 1.
    opened = kinematics.propose(q, q, action, current_gripper_finger_target=.025)
    assert opened["gripper_width_target_m"] == pytest.approx(.07)
    action[6] = 0.
    linear = kinematics.propose(q, q, action, gripper_binary=False,
                                current_gripper_finger_target=.02)
    assert linear["gripper_width_target_m"] == pytest.approx(.035)


@pytest.mark.parametrize("body_name,attribute,value", [
    ("link2", "euler", "0 0 1"),
    ("link4", "mocap", "true"),
    ("base_link", "pos", "1 0 0"),
])
def test_unsupported_model_transforms_fail_closed(tmp_path, body_name, attribute, value):
    root = ET.parse(XML).getroot()
    root.find(f".//body[@name='{body_name}']").set(attribute, value)
    modified = tmp_path/"unsupported.xml"
    ET.ElementTree(root).write(modified)
    with pytest.raises(ValueError):
        PiperKinematics(modified)


def test_invalid_command_inputs_and_zero_axis_are_rejected(tmp_path):
    kinematics = PiperKinematics(XML)
    q = (kinematics.lower+kinematics.upper)/2
    with pytest.raises(ValueError):
        kinematics.forward(np.full(6, np.nan))
    with pytest.raises(ValueError):
        kinematics.propose(q, q, np.zeros(6))
    with pytest.raises(ValueError):
        kinematics.propose(q, q, np.zeros(7), ik_damping=0)
    with pytest.raises(ValueError):
        kinematics.propose(q, q, np.zeros(7), current_gripper_finger_target=.1)
    root = ET.parse(XML).getroot()
    root.find(".//joint[@name='joint2']").set("axis", "0 0 0")
    modified = tmp_path/"zero_axis.xml"
    ET.ElementTree(root).write(modified)
    with pytest.raises(ValueError, match="axis"):
        PiperKinematics(modified)


def test_runtime_kinematics_does_not_import_simulation_or_robot_stack():
    code = """import sys
from piper_policy_runtime.kinematics import PiperKinematics
PiperKinematics(sys.argv[1]).propose([0,1.5,-1,0,0,0], [0,1.5,-1,0,0,0], [0]*7)
assert not any(name in sys.modules for name in ('jax', 'mujoco', 'piper_sdk', 'torch'))
"""
    subprocess.run([sys.executable, "-c", code, str(XML)], check=True,
                   env=dict(os.environ, PYTHONPATH=os.environ.get('PYTHONPATH','')))
