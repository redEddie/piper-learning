#!/usr/bin/env python3
"""Stress Warp capacities using full-range Piper joint poses on one GPU.

These randomly overlapping configurations are capacity stress inputs, not valid
training initial states. No training, rendering, or physics settings are changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import traceback


IGNORED_OVERFLOW_MASK = (1 << 9) | (1 << 10)


def sample_failures(sample: dict, capacities: dict) -> list[str]:
    """Reject capacity/finite failures, while retaining solver-limit diagnostics."""
    failures = []
    if sample["overflow_bits_or"] & ~IGNORED_OVERFLOW_MASK:
        failures.append("critical_overflow")
    if sample["nonfinite_worlds"]:
        failures.append("nonfinite_state")
    if sample["contacts_total"] > capacities["naconmax"]:
        failures.append("contacts_exceed_capacity")
    if sample["constraints_max_per_world"] > capacities["njmax"]:
        failures.append("constraints_exceed_capacity")
    if sample["broadphase_pairs_total"] > capacities["naconmax"]:
        failures.append("broadphase_exceeds_capacity")
    return failures


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10,
                        help="Individual physics steps after forward; 0 checks static poses only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contacts-per-env", type=int, default=160)
    parser.add_argument("--max-constraints-per-env", type=int, default=512)
    parser.add_argument("--ccd-pairs-per-env", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for field in ("num_envs", "contacts_per_env", "max_constraints_per_env", "ccd_pairs_per_env"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    return args


def run(args, report):
    # Keep --help and result validation usable without initializing JAX/CUDA.
    from piper_sim_lab.runtime import configure
    configure("gpu")
    import jax
    import jax.numpy as jp
    import numpy as np
    from mujoco import mjx
    from mujoco.mjx.third_party.mujoco_warp._src.types import OverflowType
    from mujoco_playground._src import mjx_env
    from piper_sim_lab.assets import verify_assets
    from piper_sim_lab.env import ARM_JOINTS, PiperReach

    if jax.default_backend() != "gpu":
        raise RuntimeError("This check requires the CUDA backend")
    report["devices"] = [str(device) for device in jax.devices()]
    report["packages"] = {
        name: importlib.metadata.version(name)
        for name in ("mujoco", "mujoco-mjx", "mujoco-warp", "warp-lang", "jax", "jaxlib", "playground")
    }
    report["assets"] = verify_assets()
    overrides = dict(
        impl="warp", batch_capacity=args.num_envs,
        contacts_per_env=args.contacts_per_env,
        max_constraints_per_env=args.max_constraints_per_env,
        ccd_pairs_per_env=args.ccd_pairs_per_env,
    )
    env = PiperReach(config_overrides=overrides)
    model, mjx_model = env.mj_model, env.mjx_model
    capacities = env._make_data_kwargs()
    report["capacities"] = capacities
    report["environment_config"] = env._config.to_dict()
    report["overflow_bit_names"] = {
        str(int(bit)): bit.name for bit in OverflowType
        if int(bit) and int(bit) & (int(bit) - 1) == 0
    }
    from piper_sim_lab import env as env_module
    report["environment_source_sha256"] = hashlib.sha256(Path(env_module.__file__).read_bytes()).hexdigest()

    joints = np.array([model.joint(name).id for name in ARM_JOINTS])
    qpos_ids = np.array(model.jnt_qposadr[joints])
    actuator_ids = np.array([model.actuator(name).id for name in ARM_JOINTS])
    limits = np.asarray(model.jnt_range[joints])
    rng = np.random.default_rng(args.seed)
    joint_poses = rng.uniform(limits[:, 0], limits[:, 1], (args.num_envs, 6)).astype(np.float32)
    qpos = np.broadcast_to(np.asarray(env._initial_qpos), (args.num_envs, model.nq)).copy()
    qpos[:, qpos_ids] = joint_poses
    controls = np.broadcast_to(np.asarray(env._initial_ctrl), (args.num_envs, model.nu)).copy()
    controls[:, actuator_ids] = joint_poses
    report["input"] = {
        "distribution": "Independent uniform draws over all six model arm joint ranges",
        "joint_ranges": limits.tolist(), "initial_qvel": 0,
        "gripper_qpos": [env._config.gripper_opening, -env._config.gripper_opening],
        "arm_control": "Hold each sampled joint pose using original position actuators",
        "qpos_sha256": hashlib.sha256(qpos.tobytes()).hexdigest(),
        "num_envs": args.num_envs, "seed": args.seed,
    }

    def initialize(pose, control):
        data = mjx_env.make_data(
            model, qpos=pose, qvel=jp.zeros(model.nv), ctrl=control,
            impl="warp", **capacities,
        )
        return mjx.forward(mjx_model, data)

    def summarize(data):
        finite = jp.ones((args.num_envs,), dtype=bool)
        for name in ("qpos", "qvel", "qacc", "ctrl", "act", "site_xpos"):
            value = getattr(data, name)
            finite &= jp.all(jp.isfinite(value), axis=tuple(range(1, value.ndim)))
        overflow = data._impl.overflow
        critical = jp.bitwise_and(overflow, ~IGNORED_OVERFLOW_MASK)
        return {
            "contacts_total": jp.max(data._impl.nacon),
            "constraints_max_per_world": jp.max(data._impl.nefc),
            "constraints_total": jp.sum(data._impl.nefc),
            "broadphase_pairs_total": jp.max(data._impl.ncollision),
            "overflow_bits_or": jp.bitwise_or.reduce(overflow.reshape(-1)),
            "critical_overflow_worlds": jp.count_nonzero(critical),
            "solver_limit_worlds": jp.count_nonzero(jp.bitwise_and(overflow, IGNORED_OVERFLOW_MASK)),
            "nonfinite_worlds": jp.count_nonzero(~finite),
        }

    summarize_jit = jax.jit(summarize)

    def record(data, stage, elapsed):
        raw = jax.device_get(summarize_jit(data))
        sample = {key: int(value) for key, value in raw.items()}
        sample.update(stage=stage, call_seconds_including_first_compile=elapsed)
        sample["failures"] = sample_failures(sample, capacities)
        report["samples"].append(sample)
        print(json.dumps(sample), flush=True)

    begin = time.perf_counter()
    data = jax.jit(jax.vmap(initialize))(jp.asarray(qpos), jp.asarray(controls))
    jax.block_until_ready(data)
    record(data, "forward", time.perf_counter() - begin)
    step = jax.jit(jax.vmap(lambda row: mjx.step(mjx_model, row)))
    for index in range(args.steps):
        begin = time.perf_counter()
        data = step(data)
        jax.block_until_ready(data)
        record(data, f"physics_step_{index + 1}", time.perf_counter() - begin)
    report["finite_fields_checked"] = ["qpos", "qvel", "qacc", "ctrl", "act", "site_xpos"]
    report["physics_steps"] = args.steps
    report["aggregate"] = {
        "contacts_peak_total": max(sample["contacts_total"] for sample in report["samples"]),
        "constraints_peak_per_world": max(sample["constraints_max_per_world"] for sample in report["samples"]),
        "all_finite": all(sample["nonfinite_worlds"] == 0 for sample in report["samples"]),
        "critical_overflow_detected": any(sample["critical_overflow_worlds"] for sample in report["samples"]),
        "failed_samples": sum(bool(sample["failures"]) for sample in report["samples"]),
    }
    report["status"] = "passed" if not report["aggregate"]["failed_samples"] else "failed"


def main(argv=None):
    args = parse_args(argv)
    report = {
        "status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Full-range overlapping joint-pose capacity stress; not valid training states or a task-success test",
        "ignored_overflow_bits": {"9": "solver iteration limit", "10": "line-search iteration limit"},
        "ignored_overflow_mask": IGNORED_OVERFLOW_MASK,
        "samples": [],
    }
    try:
        run(args, report)
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
