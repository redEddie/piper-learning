"""A thin configuration adapter around upstream Brax PPO, not a PPO fork."""
import argparse
import functools
import json
from pathlib import Path
import time

from piper_sim_lab.runtime import configure, GpuSampler, provenance, write_json


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    train = config["training"]
    for name in ("num_envs", "batch_size", "num_minibatches", "num_timesteps", "episode_length",
                 "num_eval_envs", "unroll_length", "num_updates_per_batch"):
        if type(train[name]) is not int or train[name] <= 0:
            raise ValueError(f"training.{name} must be a positive integer")
    if train["batch_size"] * train["num_minibatches"] % train["num_envs"]:
        raise ValueError("batch_size * num_minibatches must be divisible by num_envs")
    if train.get("action_repeat", 1) != 1:
        raise ValueError("Keep action_repeat=1; control rate is set with environment.ctrl_dt")
    if train["num_timesteps"] <= 0 or train["episode_length"] <= 0:
        raise ValueError("Positive timesteps and episode_length are required")
    if config["environment"].get("task") in ("quadratic_release","direct_gripper_release"):
        gamma=config["environment"].get("failure_discount",.99)
        if abs(gamma-train.get("discounting",.9))>1e-12:
            raise ValueError("failure_discount must match training.discounting for tail-cost equivalence")
        if config["environment"].get("penalty_horizon_steps",200)!=train["episode_length"]:
            raise ValueError("penalty_horizon_steps must match training.episode_length")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--warm-start", type=Path, help="Restore policy/value/normalizer; optimizer and RNG restart")
    args = parser.parse_args()
    configure(args.platform)
    import jax
    import numpy as np
    from brax.training.agents.ppo import networks, train as ppo
    from piper_sim_lab.wrappers import wrap_for_training
    from piper_sim_lab.tasks import make_env

    config = load_config(args.config)
    if config.get('learner') == 'recurrent_ppo':
        raise ValueError('Use python -m piper_sim_lab.recurrent_train for this recurrent configuration')
    if args.platform == "cpu":
        config["environment"]["impl"] = "jax"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config["environment"]["batch_capacity"] = config["training"]["num_envs"]
    env = make_env(config["environment"])
    eval_config = dict(config["environment"], batch_capacity=config["training"]["num_eval_envs"])
    # Keep the task name with the resolved settings so evaluation rebuilds the same environment.
    config["resolved_environment"] = dict(env._config.to_dict(), task=config["environment"].get("task", "reach"))
    write_json(output / "config.json", config)
    if hasattr(env, 'reset_bank'):
        write_json(output / 'reset-bank.json', env.reset_bank)
    if hasattr(env, 'reward_settings'):
        from dataclasses import asdict
        write_json(output / 'reward-settings.json', asdict(env.reward_settings))
    write_json(output / "provenance.json", provenance())
    write_json(output / "invocation.json", {"config": str(args.config.resolve()),
        "platform": args.platform, "warm_start": str(args.warm_start.resolve()) if args.warm_start else None})
    first_policy = None
    policy_changes = []
    started = time.monotonic()

    def on_policy(steps, make_policy, params):
        nonlocal first_policy
        leaves = [np.asarray(x).copy() for x in jax.tree.leaves(params[1])]
        if not all(np.isfinite(x).all() for x in leaves):
            raise FloatingPointError("Nonfinite policy parameters")
        if first_policy is None:
            first_policy = leaves
        delta = sum(float(np.sum((x - y) ** 2)) for x, y in zip(leaves, first_policy)) ** 0.5
        policy_changes.append({"steps": int(steps), "policy_l2_change": delta})

    def progress(steps, metrics):
        row = {"steps": int(steps), "elapsed_seconds": time.monotonic() - started,
               **{k: float(v) for k, v in metrics.items()}}
        if not all(np.isfinite(v) for v in row.values()):
            raise FloatingPointError(f"Nonfinite training metric: {row}")
        if any(v > 0 for k, v in row.items() if "capacity_overflow" in k or "nonfinite" in k):
            raise FloatingPointError(f"Invalid physics transition: {row}")
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    with GpuSampler() as gpu:
        _, params, metrics = ppo.train(
            environment=env,
            eval_env=make_env(eval_config),
            **config["training"],
            seed=config["seed"],
            network_factory=functools.partial(networks.make_ppo_networks, **config["network"]),
            wrap_env_fn=wrap_for_training,
            # Per-slot physical parameters (Table 10); the same function serves the eval env.
            randomization_fn=env.physics_randomization_fn if config["resolved_environment"].get("physics_randomization") else None,
            progress_fn=progress,
            policy_params_fn=on_policy,
            save_checkpoint_path=str(output / "checkpoints"),
            restore_checkpoint_path=str(args.warm_start.resolve()) if args.warm_start else None,
            max_devices_per_host=1,
        )
        jax.block_until_ready(params)
    write_json(output / "gpu.json", gpu.report())
    changed = policy_changes[-1]["policy_l2_change"] > 0
    write_json(output / "summary.json", {
        "status": "passed" if changed else "failed", "elapsed_seconds": time.monotonic() - started,
        "policy_changed": changed, "policy_changes": policy_changes,
        "checkpoint_semantics": "policy, value, observation normalizer; warm-start resets optimizer/RNG",
        "metrics": {k: float(v) for k, v in metrics.items()},
    })
    if not changed:
        raise RuntimeError("Training returned without updating the policy")


if __name__ == '__main__':
    main()

