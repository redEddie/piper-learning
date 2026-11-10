"""Compare a saved policy with random/zero actions on identical held-out seeds."""
import argparse
import functools
import json
from pathlib import Path

from piper_sim_lab.runtime import configure, provenance, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Training run's config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--platform", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error("num-envs must be positive")
    configure(args.platform)
    import jax
    import jax.numpy as jp
    import numpy as np
    from piper_sim_lab.checkpoint import load_policy
    from piper_sim_lab.tasks import make_env
    config = json.loads(args.config.read_text())
    if config["training"].get("action_repeat", 1) != 1:
        raise ValueError("Evaluation requires action_repeat=1")
    effective_env = dict(config.get("resolved_environment", config["environment"]))
    if args.platform == "cpu":
        effective_env["impl"] = "jax"
    effective_env["batch_capacity"] = args.num_envs
    env = make_env(effective_env)
    progress_key = env.PROGRESS_METRIC
    args.output.mkdir(parents=True, exist_ok=False)
    policy = load_policy(
        args.checkpoint, observation_size=env.observation_size,
        action_size=env.action_size, deterministic=True,
    )
    steps = config["training"]["episode_length"]
    report = {"seed": args.seed, "num_envs": args.num_envs, "steps": steps, "checkpoint": str(args.checkpoint.resolve()),
              "success_definition": env.SUCCESS_DEFINITION, "progress_metric": progress_key,
              "provenance": provenance(), "config": config, "effective_environment": effective_env, "results": {}}
    v_reset, v_step = jax.vmap(env.reset), jax.vmap(env.step)
    if effective_env.get("physics_randomization"):
        # Same per-slot physics draws as training (Playground's DR vmap), seeded by --seed.
        from mujoco_playground import wrapper as pg_wrapper
        dr_env = pg_wrapper.BraxDomainRandomizationVmapWrapper(env, functools.partial(
            env.physics_randomization_fn, rng=jax.random.split(jax.random.key(args.seed + 7), args.num_envs)))
        v_reset, v_step = dr_env.reset, dr_env.step
    for mode in ("zero", "random", "trained"):
        initial = jax.jit(v_reset)(jax.random.split(jax.random.key(args.seed), args.num_envs))

        @jax.jit
        def rollout(state, key):
            def step(carry, _):
                state, key = carry
                key, action_key = jax.random.split(key)
                if mode == "trained":
                    action, _ = policy(state.obs, action_key)
                elif mode == "random":
                    action = jax.random.uniform(action_key, (args.num_envs, env.action_size), minval=-1., maxval=1.)
                else:
                    action = jp.zeros((args.num_envs, env.action_size))
                state = v_step(state, action)
                sample = (state.metrics[progress_key], state.metrics["success"], state.done, state.data.qpos, state.data.mocap_pos,
                          jp.all(jp.isfinite(state.data.qpos)) & jp.all(jp.isfinite(state.data.qvel)))
                return (state, key), sample
            return jax.lax.scan(step, (state, key), None, length=steps)

        (_, _), trajectory = rollout(initial, jax.random.key(args.seed + 1))
        distances, successes, failures, qpos, mocap, physics_finite = [np.asarray(x) for x in jax.block_until_ready(trajectory)]
        finite = bool(np.isfinite(distances).all() and physics_finite.all())
        valid = (~np.any(failures > 0, axis=0)) & np.all(np.isfinite(distances), axis=0)
        report["results"][mode] = {
            "initial_progress": float(np.mean(np.asarray(initial.metrics[progress_key]))),
            "final_progress": float(np.mean(distances[-1, valid])) if valid.any() else None,
            "final_success_rate": float(np.mean((successes[-1] > 0) & valid)),
            "ever_success_rate": float(np.mean(np.any(successes > 0, axis=0) & valid)),
            "failure_rate": float(np.mean(~valid)), "finite": finite,
        }
        write_json(args.output / "evaluation.json", report)
        if not finite:
            raise FloatingPointError(f"Nonfinite evaluation: {mode}")
        if mode == "trained":
            np.savez_compressed(args.output / "trajectory.npz", qpos=qpos, mocap_pos=mocap, progress=distances, success=successes)
            if args.video:
                import mujoco
                import mediapy as media
                import imageio_ffmpeg
                media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
                model = env.mj_model
                data = mujoco.MjData(model)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [.15, 0, .25]
                camera.distance = 1.25
                camera.azimuth = 135
                camera.elevation = -25
                frames = []
                with mujoco.Renderer(model, height=480, width=640) as renderer:
                    for t in range(steps):
                        panels = []
                        for i in range(min(4, args.num_envs)):
                            data.qpos[:] = qpos[t, i]
                            data.mocap_pos[:] = mocap[t, i]
                            mujoco.mj_forward(model, data)
                            renderer.update_scene(data, camera=camera)
                            panels.append(renderer.render().copy())
                        while len(panels) < 4:
                            panels.append(np.zeros_like(panels[0]))
                        frames.append(np.concatenate([np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)], axis=0))
                media.write_video(str(args.output / "policy.mp4"), frames, fps=1/env.dt)
                media.write_image(str(args.output / "preview.png"), frames[-1])
        write_json(args.output / "evaluation.json", report)
    print(json.dumps(report, indent=2), flush=True)
