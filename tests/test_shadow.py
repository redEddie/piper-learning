"""Shadow adapter rejects incomplete live inputs and preserves diagnostic gates."""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from piper_policy_runtime.kinematics import PiperKinematics
from piper_policy_runtime.shadow import assemble, run

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / 'src/piper_sim_lab/assets/agilex_piper/piper.xml'


@pytest.fixture
def inputs():
    vision = {'frame': 'piper_base_link', 'units': 'm',
              'camera_to_base': {'T': np.eye(4).tolist()}, 'objects': [
                  {'id': 'block', 'pos': [.3, -.1, .025], 'quat_wxyz': [1, 0, 0, 0],
                   'size': {'type': 'box', 'half_extents': [.025] * 3}},
                  {'id': 'bowl', 'pos': [.3, .16, 0], 'size': {'type': 'bowl'}}]}
    robot = {'q_rad': [0, 1.7, -1, 0, 1, 0], 'qvel_rad_s': [0] * 6,
             'gripper_full_width_m': .07, 'telemetry_fresh_at_frame': True}
    calib = {'trusted': False, 'T_base_cam': np.eye(4).tolist()}
    env = {'gripper_max': .035, 'block_x': [.25, .4], 'block_y': [-.15, -.02],
           'bowl_x': [.25, .35], 'bowl_y': [.12, .2]}
    return vision, robot, calib, PiperKinematics(XML), env


def test_measured_width_and_relative_position_are_not_replaced(inputs):
    v, r, c, k, e = inputs
    r['gripper_full_width_m'] = .0963
    obs, target, warnings, provenance = assemble(v, r, c, k, e, virtual_target=True)
    assert obs.shape == (48,)
    assert obs[24] == pytest.approx(.0963)  # Remains measured, beyond training max.
    np.testing.assert_allclose(obs[34:37], obs[25:28] - obs[18:21], atol=2e-8)
    np.testing.assert_allclose(obs[37:40], obs[25:28] - obs[31:34], atol=2e-8)
    np.testing.assert_array_equal(obs[12:18], np.zeros(6))
    assert 'camera_to_base_calibration_untrusted' in warnings
    assert 'measured_gripper_width_exceeds_training_range' in warnings
    assert 'virtual' in provenance['joint_target']


def test_v1_observation_uses_vendor_tcp_and_table_below_base(inputs):
    v, r, c, _, e = inputs
    model = PiperKinematics(ROOT / 'src/piper_sim_lab/assets/agilex_piper_v100/piper.xml')
    e.update(robot_model='piper_v100', base_height=.015, gripper_max=.05)
    v['objects'][1]['pos'][2] = -.012
    obs, _, warnings, provenance = assemble(v, r, c, model, e, virtual_target=True)
    np.testing.assert_allclose(obs[18:21], model.forward(r['q_rad'])['tcp_position'], atol=2e-8)
    assert 'bowl_below_training_table_z' not in warnings
    assert '0.1358' in provenance['tcp']


@pytest.mark.parametrize('failure', ['missing_velocity', 'stale', 'missing_bowl', 'wrong_frame', 'wrong_calibration', 'unknown_target'])
def test_real_input_gaps_are_rejected(inputs, failure):
    v, r, c, k, e = inputs
    if failure == 'missing_velocity': r['qvel_rad_s'][2] = None
    if failure == 'stale': r['telemetry_fresh_at_frame'] = False
    if failure == 'missing_bowl': v['objects'].pop()
    if failure == 'wrong_frame': v['frame'] = 'camera'
    if failure == 'wrong_calibration': c['T_base_cam'][0][3] = .2
    with pytest.raises(ValueError):
        assemble(v, r, c, k, e, virtual_target=failure != 'unknown_target')


def test_strict_blocks_while_diagnostic_records_proposals(tmp_path, inputs):
    exports = os.environ.get('PIPER_POLICY_EXPORTS')
    if not exports: pytest.skip('Needs three exported policies')
    v, r, c, _, _ = inputs
    camera = tmp_path / 'camera'
    (camera / 'color').mkdir(parents=True)
    (camera / 'depth').mkdir()
    for name in ['color/0.png', 'depth/0.png', 'meta.json']:
        (camera / name).write_bytes(b'test provenance fixture, not a real frame')
    vision_path = camera / 'object_poses.json'
    robot_path, calib_path = tmp_path / 'robot.json', tmp_path / 'calib.json'
    for path, data in [(vision_path, v), (robot_path, r), (calib_path, c)]:
        path.write_text(json.dumps(data))
    kwargs = dict(vision_path=vision_path, robot_path=robot_path, calibration_path=calib_path,
                  model_xml=XML, policies=Path(exports), block_id='block', bowl_id='bowl', virtual_target=True)
    strict = run(output=tmp_path / 'strict', **kwargs)
    assert strict['status'] == 'blocked' and not strict['policy_evaluated']
    assert not (tmp_path / 'strict/proposed_commands.json').exists()
    diagnostic = run(output=tmp_path / 'diagnostic', diagnostic_only=True, **kwargs)
    assert diagnostic['policy_evaluated'] and not diagnostic['control_ready']
    commands = json.loads((tmp_path / 'diagnostic/proposed_commands.json').read_text())
    assert len(commands) == 3
    for command in commands:
        assert not command['hardware_commands_sent']
        assert command['real_gripper_effort_command'] is None
        assert np.isfinite(command['proposed_simulation_command']['joint_target_rad']).all()
    # Diagnostic mode also refuses a missing task object instead of fabricating it.
    v['objects'].pop()
    vision_path.write_text(json.dumps(v))
    missing = run(output=tmp_path / 'missing', diagnostic_only=True, **kwargs)
    assert not missing['policy_evaluated']
