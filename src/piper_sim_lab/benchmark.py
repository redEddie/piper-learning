"""Measure synchronized, headless batched physics with real episode resets."""
import argparse
from pathlib import Path
import time

from piper_sim_lab.runtime import configure, GpuSampler, provenance, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 128, 1024, 4096])
    parser.add_argument("--impl", choices=["jax", "warp"], default=None)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    if min(*args.num_envs, args.steps, args.repeats) <= 0:
        parser.error("All counts must be positive")
    args.impl = args.impl or ("warp" if args.platform == "gpu" else "jax")
    configure(args.platform)
    import jax
    import jax.numpy as jp
    import numpy as np
    from piper_sim_lab.wrappers import wrap_for_training
    from piper_sim_lab.env import PiperReach
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"provenance": provenance(), "profile": "PiperReach, full collisions, zero actions, full fresh autoreset, no rendering/PPO",
              "impl": args.impl, "results": []}
    for count in args.num_envs:
        task = PiperReach(config_overrides={"impl": args.impl, "batch_capacity": count})
        env = wrap_for_training(task, episode_length=100)
        with GpuSampler() as gpu:
            started = time.monotonic()
            state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), count))
            jax.block_until_ready(state)
            reset_seconds = time.monotonic() - started
            print(f"{count} env reset: {reset_seconds:.3f}s", flush=True)
            actions = jp.zeros((count, env.action_size))

            @jax.jit
            def rollout(state):
                def step(carry, _):
                    state = env.step(carry, actions)
                    failures = jp.stack([jp.sum(state.metrics[k]) for k in ("nonfinite", "capacity_overflow")])
                    return state, failures
                state, failures = jax.lax.scan(step, state, None, length=args.steps)
                return state, jp.sum(failures, axis=0)

            started = time.monotonic()
            state, failure_counts = rollout(state)
            jax.block_until_ready(state)
            total_failures = np.asarray(failure_counts).copy()
            first_seconds = time.monotonic() - started
            print(f"{count} env first rollout: {first_seconds:.3f}s", flush=True)
            times = []
            for _ in range(args.repeats):
                started = time.monotonic()
                state, failure_counts = rollout(state)
                jax.block_until_ready(state)
                total_failures += np.asarray(failure_counts)
                times.append(time.monotonic() - started)
            finite = bool(np.isfinite(np.asarray(state.data.qpos)).all() and np.isfinite(np.asarray(state.data.qvel)).all())
            resets = np.asarray(state.info["AutoResetWrapper_done_count"])
        row = {"environment": task._config.to_dict(), "num_envs": count, "steps_per_repeat": args.steps, "reset_first_call_seconds": reset_seconds,
               "rollout_first_call_seconds": first_seconds, "warm_seconds": times,
               "warm_transitions_per_second": count * args.steps / float(np.median(times)),
               "warm_physics_steps_per_second": count * args.steps * env.n_substeps / float(np.median(times)),
               "policy_dt_seconds": env.dt, "finite": finite, "nonfinite_transitions": int(total_failures[0]),
               "capacity_overflow_transitions": int(total_failures[1]), "minimum_episode_resets": int(resets.min()),
               "peak_board_memory_mib": gpu.report()["peak_board_memory_mib"]}
        report["results"].append(row)
        write_json(args.output / "benchmark.json", report)
        write_json(args.output / f"gpu-{count}.json", gpu.report())
        print(row, flush=True)
        if not finite or np.any(total_failures):
            raise FloatingPointError("Invalid physics transition or Warp capacity overflow")

