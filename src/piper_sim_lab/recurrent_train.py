"""MLP/LSTM/encoder-LSTM: sequence collection + upstream Brax PPO loss.

Run with python -m piper_sim_lab.recurrent_train. This does not change train.py.
All actors use this exact loop. Normalization is frozen through collection and
SGD, and updated afterward. All episode prefixes are replayed before each loss
with the current weights; BPTT spans unroll_length, with reset masks inside it.
"""
import argparse
import functools
import json
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--actor", choices=["mlp", "lstm", "transformer_lstm", "mlp_lstm"], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--platform", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument('--warm-start', type=Path, help='Restore matching full learner; restart simulator episodes')
    ap.add_argument("--allow-gpu-sharing", action="store_true", help="Only for explicit isolated smoke tests")
    args = ap.parse_args()
    from piper_sim_lab.runtime import configure
    configure(args.platform)
    import jax
    import jax.numpy as jp
    import numpy as np
    import optax
    from flax import serialization
    from brax.training import types
    from brax.training.acme import running_statistics as stats
    from brax.training.agents.ppo import losses
    from piper_sim_lab.recurrent import ActorAdapter, zero_carry, with_memory, write_history
    from piper_sim_lab.tasks import make_env
    from piper_sim_lab.wrappers import wrap_for_training
    from piper_sim_lab.runtime import write_json, provenance, GpuSampler
    from piper_sim_lab.train import load_config

    if args.platform == "gpu" and not args.allow_gpu_sharing:
        from piper_sim_lab.experiment_guard import require_idle_gpu
        require_idle_gpu()
    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    t = cfg["training"]
    n, horizon, episode = t["num_envs"], t["unroll_length"], t["episode_length"]
    if t.get("action_repeat", 1) != 1 or t.get("bootstrap_on_timeout", False):
        raise ValueError("This adapter requires action_repeat=1 and bootstrap_on_timeout=False")
    if t["batch_size"] * t["num_minibatches"] != n:
        raise ValueError("One rollout per update: batch_size * num_minibatches must equal num_envs")
    if t["num_timesteps"] % (n * horizon):
        raise ValueError("num_timesteps must be exactly divisible by num_envs * unroll_length")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg["environment"]["batch_capacity"] = n
    if args.platform == "cpu":
        cfg["environment"]["impl"] = "jax"
    raw_env = make_env(cfg["environment"])
    cfg["resolved_environment"] = dict(raw_env._config.to_dict(), task=cfg["environment"]["task"])
    cfg["actor"] = args.actor
    cfg["recurrent_width"] = 128
    cfg['continuation_checkpoint'] = str(args.warm_start.resolve()) if args.warm_start else None
    cfg['actor_architecture'] = cfg.get('actor_architecture') if args.actor in ('transformer_lstm', 'mlp_lstm') else None
    cfg["adapter"] = {"normalizer_update": "after SGD", "prefix_replay": "full current-episode prefix, detached", "tbptt": horizon}
    write_json(output / "config.json", cfg)
    write_json(output / "provenance.json", provenance())
    if hasattr(raw_env, 'reset_bank'):
        write_json(output/'reset-bank.json', raw_env.reset_bank)
    if hasattr(raw_env, 'reward_settings'):
        from dataclasses import asdict
        write_json(output/'reward-settings.json', asdict(raw_env.reward_settings))
    randomization = None
    if cfg["environment"].get("physics_randomization"):
        randomization = functools.partial(raw_env.physics_randomization_fn,
            rng=jax.random.split(jax.random.PRNGKey(args.seed + 7), n))
    env = wrap_for_training(raw_env, episode_length=episode, randomization_fn=randomization)
    actor = ActorAdapter(raw_env.observation_size, raw_env.action_size, args.actor, cfg["network"],
                         actor_config=cfg['actor_architecture'], exploration_config=cfg.get('exploration'))
    key = jax.random.PRNGKey(args.seed)
    key, pk, vk, ek = jax.random.split(key, 4)
    params = losses.PPONetworkParams(actor.init(pk), actor.base.value_network.init(vk))
    opt = optax.chain(optax.clip_by_global_norm(t["max_grad_norm"]), optax.adam(t["learning_rate"]))
    opt_state = opt.init(params)
    sizes = raw_env.observation_size
    normalizer = stats.init_state({k: jp.zeros(v) for k, v in sizes.items()})
    parent_step = 0
    if args.warm_start:
        from piper_sim_lab.recurrent import restore_learner_checkpoint
        restored = restore_learner_checkpoint(args.warm_start,
            dict(params=params, normalizer=normalizer, optimizer=opt_state,
                 rng=jax.random.key_data(key), step=0), cfg)
        params, normalizer, opt_state, key = (restored[k] for k in ('params','normalizer','optimizer','rng'))
        parent_metadata = json.loads((args.warm_start/'metadata.json').read_text())
        parent_step = parent_metadata.get('cumulative_transitions', int(restored['step']))
        write_json(output/'continuation.json', dict(checkpoint=str(args.warm_start.resolve()),
            parent_transitions=parent_step, restored=['policy','value','normalizer','optimizer','rng'],
            simulator='fresh episodes, empty recurrent history; not an uninterrupted simulation'))
    initial_policy = jax.tree.map(lambda x: np.asarray(x).copy(), params.policy)
    state = jax.jit(env.reset)(jax.random.split(ek, n))
    state = jax.tree.map(lambda x: jp.astype(jp.asarray(x), jp.asarray(x).dtype), state)
    history = jp.zeros((n, episode, sizes[actor.policy_key][-1]))
    lengths = jp.zeros(n, dtype=jp.int32)
    param_counts = {"actor": sum(x.size for x in jax.tree.leaves(params.policy)),
                    "critic": sum(x.size for x in jax.tree.leaves(params.value))}
    write_json(output / "parameters.json", param_counts)
    write_json(output/'initial-resets.json', {'counts':np.bincount(np.asarray(state.info['reset_stage']),
        minlength=len(cfg['environment']['reset_stage_probs'])).tolist(),
        'reset_stage_probs':cfg['environment']['reset_stage_probs'],
        'reward_phase_counts':np.bincount(np.asarray(state.metrics['stage_section'],dtype=int),minlength=5).tolist()})
    loss_fn = functools.partial(losses.compute_ppo_loss, ppo_network=actor.networks,
        entropy_cost=t["entropy_cost"], discounting=t["discounting"],
        gae_lambda=t["gae_lambda"], clipping_epsilon=t["clipping_epsilon"],
        reward_scaling=t["reward_scaling"])

    @jax.jit
    def collect(params, norm, state, history, lengths, key):
        carry = actor.replay(norm, params.policy, history, lengths)
        def f(c, _):
            state, memory, hist, lens, key = c
            key, sample_key = jax.random.split(key)
            obs = with_memory(state.obs, memory, lens == 0)
            next_memory, logits = actor.step(norm, params.policy, memory, state.obs, lens == 0)
            raw = actor.dist.sample_no_postprocessing(logits, sample_key)
            action = actor.dist.postprocess(raw)
            nxt = env.step(state, action)
            hist, lens = write_history(hist, lens, state.obs[actor.policy_key], nxt.done > 0)
            data = types.Transition(obs, action, nxt.reward, 1-nxt.done,
                with_memory(nxt.obs, next_memory, nxt.done > 0),
                {"policy_extras": {"raw_action": raw,
                  "log_prob": actor.dist.log_prob(logits, raw), "distribution_params": logits},
                 "state_extras": {"truncation": nxt.info["truncation"]}})
            ended = nxt.done > 0
            boundary = nxt.info['episode_metrics']
            metric = jp.stack([jp.mean(nxt.reward), jp.mean(nxt.metrics["success"]),
                               jp.max(nxt.metrics["capacity_overflow"]), jp.max(nxt.metrics["nonfinite"]),
                               jp.mean(ended*boundary.get('controller_failed',jp.zeros_like(nxt.done))),
                               jp.mean(ended*boundary.get('task_failed',jp.zeros_like(nxt.done))),
                               jp.mean(actor.dist.create_dist(logits).scale[...,6]),
                               jp.mean(action[...,6] >= raw_env._config.gripper_threshold),
                               jp.mean(action[...,6] <= -raw_env._config.gripper_threshold),
                               jp.mean(actor.dist.create_dist(logits).scale[...,:6])])
            return (nxt, next_memory, hist, lens, key), (data, metric)
        (state, _, history, lengths, key), (data, metric) = jax.lax.scan(
            f, (state, carry, history, lengths, key), None, length=horizon)
        return state, history, lengths, key, jax.tree.map(lambda x: jp.swapaxes(x,0,1), data), metric.mean(0)

    def recurrent_loss(params, norm, data, prefix, prefix_lengths, key):
        c, h = actor.replay(norm, params.policy, prefix, prefix_lengths)
        obs = dict(data.observation)
        obs["memory_c"] = obs["memory_c"].at[:, 0].set(c)
        obs["memory_h"] = obs["memory_h"].at[:, 0].set(h)
        return loss_fn(params, norm, data._replace(observation=obs), key)

    @jax.jit
    def optimize(params, opt_state, norm, data, prefix, prefix_lengths, key):
        def epoch(c, _):
            params, opt_state, key = c
            key, perm_key = jax.random.split(key)
            order = jax.random.permutation(perm_key, n).reshape(t["num_minibatches"], -1)
            def minibatch(c, indices):
                params, opt_state, key = c
                key, loss_key = jax.random.split(key)
                batch = jax.tree.map(lambda x: x[indices], data)
                (loss, metrics), grads = jax.value_and_grad(recurrent_loss, has_aux=True)(
                    params, norm, batch, prefix[indices], prefix_lengths[indices], loss_key)
                updates, opt_state = opt.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                metrics["grad_norm"] = optax.global_norm(grads)
                return (params, opt_state, key), metrics
            return jax.lax.scan(minibatch, (params, opt_state, key), order)
        (params, opt_state, key), metrics = jax.lax.scan(epoch,
            (params, opt_state, key), None, length=t["num_updates_per_batch"])
        norm = stats.update(norm, {k:data.observation[k] for k in sizes})
        return params, opt_state, norm, key, jax.tree.map(jp.mean, metrics)

    @jax.jit
    def initial_ratio(params, norm, data, prefix, prefix_lengths):
        c,h = actor.replay(norm, params.policy, prefix, prefix_lengths)
        obs = jax.tree.map(lambda x: jp.swapaxes(x,0,1), data.observation)
        obs["memory_c"] = obs["memory_c"].at[0].set(c)
        obs["memory_h"] = obs["memory_h"].at[0].set(h)
        logits = actor.networks.policy_network.apply(norm, params.policy, obs)
        old = jp.swapaxes(data.extras["policy_extras"]["log_prob"], 0,1)
        raw = jp.swapaxes(data.extras["policy_extras"]["raw_action"],0,1)
        return jp.max(jp.abs(actor.dist.log_prob(logits,raw) - old))

    def save(step):
        dest = output / "checkpoints" / f"{step:012d}"
        dest.mkdir(parents=True, exist_ok=False)
        # Full learner state. Runtime carry is reconstructed from raw episode history.
        (dest / "learner.msgpack").write_bytes(serialization.to_bytes({
            "params": params, "normalizer": normalizer, "optimizer": opt_state,
            "rng": jax.random.key_data(key), "step": step}))
        write_json(dest / "metadata.json", {"actor": args.actor, "width":128, "step":step,
             'actor_architecture': cfg['actor_architecture'], 'action_size':raw_env.action_size,
             'exploration': cfg.get('exploration'),
             'observation_size': {k:list(v) for k,v in sizes.items()},
             'parent_transitions': parent_step, 'cumulative_transitions': parent_step+step,
             "state": "learner checkpoint; continuation restores learner with fresh simulator episodes"})

    iterations = t["num_timesteps"] // (n*horizon)
    interval = max(1, iterations // max(1, t["num_evals"]-1))
    started = time.monotonic()
    collect_seconds = update_seconds = 0.
    cold_seconds = 0.
    max_ratio_error = 0.
    with GpuSampler() as gpu:
        for i in range(iterations):
            if args.platform == "gpu" and not args.allow_gpu_sharing:
                from piper_sim_lab.experiment_guard import compute_pids
                waited = 0.
                while (other_pids := compute_pids()):
                    print(json.dumps({"waiting_for_other_gpu_processes": other_pids, "iteration": i}), flush=True)
                    before_wait = time.monotonic()
                    time.sleep(10)
                    waited += time.monotonic() - before_wait
                if waited:
                    with (output / "gpu-yields.jsonl").open("a") as f:
                        f.write(json.dumps({"iteration": i, "wait_seconds": waited}) + "\n")
            prefix, prefix_lengths = history, lengths
            clock = time.monotonic()
            state, history, lengths, key, data, rollout_metrics = collect(
                params, normalizer, state, history, lengths, key)
            jax.block_until_ready((state, data, rollout_metrics))
            duration_collect = time.monotonic()-clock
            # Check ratios at start, a nonempty prefix, and after several optimizer updates.
            if i in (0, 1, 7):
                error = float(initial_ratio(params, normalizer, data, prefix, prefix_lengths))
                max_ratio_error = max(max_ratio_error, error)
                if error > 2e-4:
                    raise RuntimeError(f"Collection/sequence log-prob mismatch: {error}")
            clock = time.monotonic()
            params, opt_state, normalizer, key, metrics = optimize(
                params, opt_state, normalizer, data, prefix, prefix_lengths, key)
            jax.block_until_ready((params, opt_state, normalizer, metrics))
            duration_update = time.monotonic()-clock
            values = np.asarray(rollout_metrics)
            if values[2] > 0 or values[3] > 0:
                raise FloatingPointError("Invalid physics transition")
            if i == 0:
                cold_seconds = duration_collect + duration_update
            else:
                collect_seconds += duration_collect
                update_seconds += duration_update
            row = {"steps": (i+1)*n*horizon, "iteration": i+1,
                "elapsed_seconds": time.monotonic()-started,
                "collection_seconds": duration_collect, "update_seconds": duration_update,
                "rollout_reward_mean": float(values[0]), "rollout_success_mean":float(values[1]),
                'capacity_overflow':float(values[2]), 'nonfinite':float(values[3]),
                'controller_failure_per_transition':float(values[4]), 'task_failure_per_transition':float(values[5]),
                'gripper_pre_tanh_std_mean':float(values[6]),
                'gripper_open_command_per_transition':float(values[7]),
                'gripper_close_command_per_transition':float(values[8]),
                'arm_pre_tanh_std_mean':float(values[9]),
                **{k:float(v) for k,v in metrics.items()}}
            if not all(np.isfinite(v) for v in row.values()):
                raise FloatingPointError(row)
            with (output/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(row)+"\n")
            if i % 32 == 0 or (i+1)%interval == 0 or i == iterations-1:
                print(json.dumps(row), flush=True)
            if (i+1)%interval == 0 or i == iterations-1:
                save((i+1)*n*horizon)
    jax.block_until_ready(params)
    delta = sum(float(np.sum((np.asarray(x)-y)**2)) for x,y in
                zip(jax.tree.leaves(params.policy), jax.tree.leaves(initial_policy)))**.5
    if not np.isfinite(delta) or delta <= 0:
        raise RuntimeError("Policy did not learn finite parameter updates")
    summary = {"actor":args.actor,"seed":args.seed,"timesteps":t["num_timesteps"],
        "elapsed_seconds":time.monotonic()-started,"first_iteration_compile_and_execute_seconds":cold_seconds,
        "steady_state_collection_seconds":collect_seconds,"steady_state_update_seconds":update_seconds,
        "steady_state_sps":(iterations-1)*n*horizon/(collect_seconds+update_seconds),
        "policy_l2_change":delta,"max_initial_log_prob_error":max_ratio_error,"parameters":param_counts}
    write_json(output/"summary.json",summary)
    write_json(output/"gpu.json",gpu.report())
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    main()
