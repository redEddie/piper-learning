#!/usr/bin/env python3
"""Capture RGB-D plus passive Piper CAN feedback; never transmit robot commands.

Run in the existing perception environment (pyrealsense2, OpenCV, NumPy).
The output camera directory is compatible with SHOW-33's --run argument.
Protocol units/byte order were checked against the installed Piper SDK v2
parser and feedback classes on Thor; the SDK is deliberately not imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import datetime
import json
from pathlib import Path
import socket
import struct
import sys
import threading
import time


JOINT_IDS = (0x2A5, 0x2A6, 0x2A7)
SPEED_IDS = tuple(range(0x251, 0x257))
GRIPPER_ID = 0x2A8
FEEDBACK_IDS = JOINT_IDS + SPEED_IDS + (GRIPPER_ID,)
CAN_FRAME = struct.Struct("=IB3x8s")


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def tx_packets(interface: str) -> int | None:
    try:
        return int((Path("/sys/class/net") / interface / "statistics/tx_packets").read_text())
    except (OSError, ValueError):
        return None


class PassiveCAN:
    """Only socket creation, bind and receive. No SDK, enable or query calls."""

    def __init__(self, interface: str):
        self.interface = interface
        self.records = defaultdict(lambda: deque(maxlen=2000))
        self.counts = Counter()
        self.errors: list[str] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.sock = None
        self.thread = None

    def start(self) -> None:
        try:
            self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            # Exact standard data-frame IDs; reject extended and RTR frames.
            filters = b"".join(struct.pack("=II", can_id, 0x800007FF | 0x40000000)
                               for can_id in FEEDBACK_IDS)
            self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
            self.sock.settimeout(0.1)
            self.sock.bind((self.interface,))
            self.thread = threading.Thread(target=self._receive, daemon=True)
            self.thread.start()
        except (OSError, AttributeError) as error:
            self.errors.append(f"CAN receive unavailable: {error}")
            self.stop()

    def _receive(self) -> None:
        while not self.stop_event.is_set():
            try:
                raw = self.sock.recv(CAN_FRAME.size)
                host_monotonic_ns = time.monotonic_ns()
                host_wall_time_ns = time.time_ns()
                if len(raw) != CAN_FRAME.size:
                    continue
                can_id, size, data = CAN_FRAME.unpack(raw)
                if can_id not in FEEDBACK_IDS or size < (8 if can_id in JOINT_IDS else 2):
                    continue
                if can_id == GRIPPER_ID and size < 7:
                    continue
                record = {"host_monotonic_ns": host_monotonic_ns,
                          "host_wall_time_ns": host_wall_time_ns,
                          "data": data[:size]}
                with self.lock:
                    self.records[can_id].append(record)
                    self.counts[can_id] += 1
            except socket.timeout:
                continue
            except OSError as error:
                if not self.stop_event.is_set():
                    self.errors.append(f"CAN receive failed: {error}")
                return

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
        if self.sock:
            self.sock.close()
            self.sock = None

    def feedback(self, frame_monotonic_ns: int | None, max_age_s: float = 0.1) -> dict:
        with self.lock:
            records = {key: list(value) for key, value in self.records.items()}
            counts = dict(self.counts)
        return select_feedback(records, counts, frame_monotonic_ns, max_age_s, self.errors)


def select_feedback(records: dict, counts: dict, frame_monotonic_ns: int | None,
                    max_age_s: float = 0.1, errors=()) -> dict:
    """Use last host-received sample at/before the frame, never a future sample.

    Stale values remain visible with explicit freshness flags. Missing values
    remain null, and must not be replaced with a measured zero downstream.
    """
    selected = {}
    components = {}
    for can_id in FEEDBACK_IDS:
        candidates = records.get(can_id, ())
        record = next((r for r in reversed(candidates)
                       if frame_monotonic_ns is not None
                       and r["host_monotonic_ns"] <= frame_monotonic_ns), None)
        selected[can_id] = record
        age = None if record is None else (frame_monotonic_ns - record["host_monotonic_ns"]) / 1e9
        components[f"0x{can_id:03X}"] = {
            "received_count": counts.get(can_id, 0), "missing": record is None,
            "host_wall_time_ns": None if record is None else record["host_wall_time_ns"],
            "host_monotonic_ns": None if record is None else record["host_monotonic_ns"],
            "age_at_frame_s": age, "stale": None if age is None else age > max_age_s,
            "fresh": age is not None and 0 <= age <= max_age_s,
        }
    q = []
    for can_id in JOINT_IDS:
        record = selected[can_id]
        q.extend([None, None] if record is None else
                 [value / 1000 * (3.141592653589793 / 180)
                  for value in struct.unpack(">ii", record["data"][:8])])
    qvel = [None if selected[can_id] is None else
            struct.unpack(">h", selected[can_id]["data"][:2])[0] * 0.001
            for can_id in SPEED_IDS]
    grip = selected[GRIPPER_ID]
    return {
        "schema_version": 1, "source": "passive_socketcan_receive_only",
        "reference_frame_host_monotonic_ns": frame_monotonic_ns,
        "q_rad": q, "qvel_rad_s": qvel,
        "gripper_full_width_m": None if grip is None else struct.unpack(">i", grip["data"][:4])[0] / 1e6,
        "gripper_status_code": None if grip is None else grip["data"][6],
        "freshness_limit_s": max_age_s,
        "telemetry_fresh_at_frame": all(c["fresh"] for c in components.values()),
        "feedback_components": components, "errors": list(errors),
        "time_alignment": "Host receipt before final color frame host receipt; not hardware timestamp synchronization.",
        "joint_speed_note": "0x251..0x256 motor-speed feedback, SDK unit 0.001 rad/s; hardware scaling validation remains pending.",
        "hardware_commands_sent": False,
    }


def frame_metadata(frame) -> dict:
    result = {}
    for name, getter in (("frame_number", frame.get_frame_number),
                         ("sensor_timestamp_ms", frame.get_timestamp),
                         ("timestamp_domain", frame.get_frame_timestamp_domain)):
        try:
            value = getter()
            result[name] = str(value) if name == "timestamp_domain" else value
        except RuntimeError as error:
            result[name] = None
            result.setdefault("metadata_errors", []).append(f"{name}: {error}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are refused")
    parser.add_argument("--can", default="can_piper")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.timeout_ms < 1:
        parser.error("frames and timeout must be positive; warmup must be nonnegative")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output already exists: {args.output}")
    report = {"schema_version": 1, "output": str(args.output.resolve()),
              "can_interface": args.can, "hardware_commands_sent": False,
              "capture_started_wall_time_ns": time.time_ns(),
              "capture_started_monotonic_ns": time.monotonic_ns(),
              "tx_packets_before": tx_packets(args.can), "errors": []}
    reader = PassiveCAN(args.can)
    pipeline = None
    pipeline_started = False
    frame_host_monotonic_ns = None
    frame_host_wall_time_ns = None
    try:
        # Optional imports stay here so pure protocol tests need no camera stack.
        import cv2
        import numpy as np
        import pyrealsense2 as rs

        reader.start()
        config = rs.config()
        if args.serial:
            config.enable_device(args.serial)
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        pipeline_started = True
        device = profile.get_device()
        scale = device.first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        for _ in range(args.warmup):
            pipeline.wait_for_frames(args.timeout_ms)
        depth_images, frame_samples = [], []
        for _ in range(args.frames):
            frameset = pipeline.wait_for_frames(args.timeout_ms)
            frame_host_monotonic_ns = time.monotonic_ns()
            frame_host_wall_time_ns = time.time_ns()
            aligned = align.process(frameset)
            color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("Aligned RGB-D frameset is incomplete")
            color = np.asanyarray(color_frame.get_data()).copy()
            depth_images.append(np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale)
            frame_samples.append({"host_monotonic_ns": frame_host_monotonic_ns,
                                  "host_wall_time_ns": frame_host_wall_time_ns,
                                  "color": frame_metadata(color_frame), "depth": frame_metadata(depth_frame)})
        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        stack = np.stack(depth_images)
        stack[stack == 0] = np.nan
        # All-invalid pixels are explicitly stored as zero (RealSense convention).
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            depth_m = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0)
        depth_mm = np.clip(np.round(depth_m * 1000), 0, 65535).astype(np.uint16)
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        camera = args.output / "camera"
        (camera / "color").mkdir(parents=True)
        (camera / "depth").mkdir()
        if not cv2.imwrite(str(camera / "color/0.png"), color):
            raise RuntimeError("Failed to save color PNG")
        if not cv2.imwrite(str(camera / "depth/0.png"), depth_mm):
            raise RuntimeError("Failed to save depth PNG")
        np.savetxt(camera / "cam_K.txt", K)
        meta = {
            "camera": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "stamp": datetime.datetime.fromtimestamp(frame_host_wall_time_ns / 1e9).astimezone().isoformat(),
            "width": intr.width, "height": intr.height, "K": K.tolist(),
            "distortion_model": str(intr.model), "distortion_coeffs": list(intr.coeffs),
            "depth_scale_m": scale, "depth_frames_median": args.frames,
            "depth_valid_ratio": float((depth_mm > 0).mean()), "gravity_imu": None,
            "host_wall_time_ns": frame_host_wall_time_ns,
            "host_monotonic_ns": frame_host_monotonic_ns,
            "frame_samples": frame_samples,
            "color": frame_samples[-1]["color"],
            "depth_capture_start_host_monotonic_ns": frame_samples[0]["host_monotonic_ns"],
            "depth_capture_end_host_monotonic_ns": frame_samples[-1]["host_monotonic_ns"],
            "depth_median_note": "Depth temporal median across listed frames; color is final frame, static scene assumed.",
            "time_alignment": "Host receipt times; RealSense sensor timestamp domain is reported separately.",
        }
        save_json(camera / "meta.json", meta)
        report["camera"] = {"directory": str(camera.resolve()), "meta": "camera/meta.json",
                            "host_monotonic_ns": frame_host_monotonic_ns}
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        if pipeline_started:
            try:
                pipeline.stop()
            except Exception as error:
                report["errors"].append(f"Camera stop failed: {error}")
        reader.stop()
        feedback = reader.feedback(frame_host_monotonic_ns)
        feedback["can_interface"] = args.can
        feedback["camera_frame_host_wall_time_ns"] = frame_host_wall_time_ns
        save_json(args.output / "robot_feedback.json", feedback)
        report["telemetry_fresh_at_frame"] = feedback["telemetry_fresh_at_frame"]
        report["can_errors"] = list(reader.errors)
        report["tx_packets_after"] = tx_packets(args.can)
        before, after = report["tx_packets_before"], report["tx_packets_after"]
        report["tx_packets_delta"] = None if before is None or after is None else after - before
        report["tx_counter_note"] = "Interface-global counters include unrelated processes; this program only receives CAN frames."
        report["capture_ended_wall_time_ns"] = time.time_ns()
        report["capture_ended_monotonic_ns"] = time.monotonic_ns()
        save_json(args.output / "capture_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
