"""Feedback decoding and camera-time selection, with no hardware access."""
from pathlib import Path
import runpy
import struct

import pytest


capture = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                             "scripts/deployment/capture_shadow_inputs.py"))


def record(stamp, data):
    return {"host_monotonic_ns": stamp, "host_wall_time_ns": stamp + 1_000_000, "data": data}


def test_feedback_uses_only_past_samples_and_marks_missing_and_stale():
    now = 1_000_000_000
    records = {
        0x2A5: [record(now - 20_000_000, struct.pack(">ii", 90000, -45000)),
                record(now + 1, struct.pack(">ii", 0, 0))],
        0x251: [record(now - 200_000_000, struct.pack(">h", -1234))],
        0x2A8: [record(now - 10_000_000, struct.pack(">ihBB", 96300, 100, 64, 0))],
    }
    result = capture["select_feedback"](records, {0x2A5: 2}, now)
    assert result["q_rad"][:2] == pytest.approx([1.5707963267948966, -0.7853981633974483])
    assert result["q_rad"][2:] == [None] * 4
    assert result["qvel_rad_s"][0] == -1.234
    assert result["gripper_full_width_m"] == 0.0963
    assert result["feedback_components"]["0x251"]["stale"] is True
    assert result["feedback_components"]["0x2A6"]["missing"] is True
    assert result["telemetry_fresh_at_frame"] is False
    assert result["hardware_commands_sent"] is False


def test_complete_fresh_feedback_and_no_frame_reference():
    now = 1_000_000_000
    records = {can_id: [record(now, b"\x00" * 8)] for can_id in capture["FEEDBACK_IDS"]}
    result = capture["select_feedback"](records, {}, now)
    assert result["telemetry_fresh_at_frame"] is True
    missing = capture["select_feedback"](records, {}, None)
    assert missing["q_rad"] == [None] * 6
    assert missing["gripper_full_width_m"] is None
    assert missing["telemetry_fresh_at_frame"] is False
