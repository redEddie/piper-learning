"""GPU proof that invalid high release ends and freshly resets immediately."""
import argparse
import functools
import json
from pathlib import Path

from piper_sim_lab.runtime import configure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--drop-weight', type=float, default=.5)
    args = p.parse_args()
    configure('gpu')
    import jax
    import jax.numpy as jp
    import numpy as np
    import yaml
    from piper_sim_lab.tasks import make_env
    from piper_sim_lab.wrappers import wrap_for_training
    cfg = yaml.safe_load(Path('configs/show103/stage-reward-v2-five-starts-4096-plus2-4-6-8.yaml').read_text())
    n = 128
    env = make_env(dict(cfg['environment'], batch_capacity=n, five_start_drop_weight=args.drop_weight, reset_stage_probs=[0.,0.,0.,1.,0.],
        position_reset_bank=cfg['environment']['evaluation_position_reset_bank']))
    wrapped = wrap_for_training(env, episode_length=200, randomization_fn=functools.partial(
        env.physics_randomization_fn, rng=jax.random.split(jax.random.key(2033), n)))
    # Pinned Brax EpisodeWrapper derives batch axes from legacy key shape (...,2).
    state = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(2026), n))
    step = jax.jit(wrapped.step)
    terminal_count = 0
    terminal_rewards = []
    failures = []
    for i in range(10):
        state = jax.block_until_ready(step(state, jp.tile(jp.array([0.,0.,0.,0.,0.,0.,1.]), (n,1))))
        ended = np.asarray(state.done) > 0
        terminal_count += int(ended.sum())
        terminal_rewards.extend(np.asarray(state.reward)[ended].tolist())
        if args.drop_weight == 0.:
            assert not np.asarray(state.metrics['stage_drop_cost']).any()
            assert (np.asarray(state.reward)[ended] >= -.02).all(), 'unexpected terminal drop penalty'
        assert not np.asarray(state.info['reward_memory'].invalid_release)[ended].any()
        assert not np.asarray(state.info['reward_memory'].released)[ended].any()
        assert np.allclose(np.asarray(state.data.ctrl[:,env._gripper_ctrl_id])[ended], 0.)
        assert not np.asarray(state.metrics['nonfinite']).any()
        assert not np.asarray(state.metrics['capacity_overflow']).any()
        failures.append(dict(step=i+1, newly_reset=int(ended.sum()),
            terminal_task_failure=float(np.asarray(state.info['episode_metrics']['task_failed'])[ended].max()) if ended.any() else 0.))
    assert terminal_count >= n, 'invalid high releases were not reset within1s'
    result = dict(status='passed', worlds=n, steps=10, terminal_resets=terminal_count,
        drop_weight=env.reward_settings.drop_weight,
        terminal_reward_range=[min(terminal_rewards),max(terminal_rewards)],
        action='stationary arm, rate-limited OPEN at125mm', records=failures,
        verified='terminal done/episode metrics retained; physical state and reward memory already fresh on the same transition',
        training=False, real_robot=False)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
