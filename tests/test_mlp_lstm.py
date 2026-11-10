"""Last-LSTM encoder: temporal behavior, PPO gradients and saved actor use."""
import json

from flax import serialization
import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.training.acme import running_statistics as stats

from piper_sim_lab.recurrent import ActorAdapter, load_actor_checkpoint, zero_carry

NETWORK = dict(policy_hidden_layer_sizes=[256, 256], value_hidden_layer_sizes=[256, 256],
               policy_obs_key='state', value_obs_key='privileged')
SIZES = dict(state=(49,), privileged=(66,))
ARCH = dict(hidden_sizes=[256, 64])


@pytest.fixture(scope='module')
def model():
    actor = ActorAdapter(SIZES, 7, 'mlp_lstm', NETWORK, actor_config=ARCH,
        exploration_config=dict(arm_min_std=.25, gripper_min_std=.5))
    return (actor, actor.init(jax.random.key(1)),
            stats.init_state({k: jp.zeros(v) for k, v in SIZES.items()}))


def observations(x, reset):
    return dict(state=x, privileged=jp.zeros(x.shape[:2]+(66,)),
                memory_c=jp.zeros(x.shape[:2]+(128,)), memory_h=jp.zeros(x.shape[:2]+(128,)),
                episode_start=reset)


def test_encoder_output_matches_interface_and_memory_is_last(model):
    _, params, _ = model
    p = params['params']
    assert p['encoder_0']['kernel'].shape == (49, 256)
    assert p['encoder_1']['kernel'].shape == (256, 64)
    assert p['distribution']['kernel'].shape == (128, 14)
    assert not any(k.startswith('head_') for k in p)


def test_step_sequence_reset_isolation_and_causality(model):
    actor, params, norm = model
    x = jax.random.normal(jax.random.key(2), (6, 2, 49))
    reset = jp.zeros((6, 2), bool).at[0].set(True).at[3, 1].set(True)
    carry, outputs = zero_carry(2), []
    for i in range(6):
        carry, logits = actor.step(norm, params, carry, dict(state=x[i]), reset[i])
        outputs.append(logits)
    seq = jax.jit(actor.sequence_logits)(norm, params, observations(x, reset))
    np.testing.assert_allclose(seq, jp.stack(outputs), atol=2e-5)
    _, fresh = actor.step(norm, params, zero_carry(1), dict(state=x[3, 1:2]), jp.ones(1, bool))
    np.testing.assert_allclose(seq[3, 1], fresh[0], atol=2e-5)
    no_reset = actor.sequence_logits(norm, params, observations(x, reset.at[3, 1].set(False)))
    np.testing.assert_allclose(seq[:, 0], no_reset[:, 0], atol=2e-5)
    assert float(jp.linalg.norm(seq[3, 1]-no_reset[3, 1])) > 1e-4
    changed = actor.sequence_logits(norm, params, observations(x.at[4:].add(100.), reset))
    np.testing.assert_array_equal(seq[:4], changed[:4])


def test_encoder_and_memory_learn_and_reset_cuts_temporal_gradient(model):
    actor, params, norm = model
    x = jax.random.normal(jax.random.key(3), (5, 1, 49))
    reset = jp.zeros((5, 1), bool).at[0].set(True)
    def objective(p, inputs, mask):
        return jp.square(actor.sequence_logits(norm, p, observations(inputs, mask))[-1]).sum()
    value, grads = jax.jit(jax.value_and_grad(objective))(params, x, reset)
    assert np.isfinite(float(value))
    for name in ('encoder_0', 'encoder_1', 'lstm'):
        values = jax.tree.leaves(grads['params'][name])
        assert all(np.isfinite(np.asarray(v)).all() for v in values)
        assert sum(float(jp.square(v).sum()) for v in values) > 1e-9
    dx = jax.jit(jax.grad(objective, argnums=1))(params, x, reset)
    assert float(jp.linalg.norm(dx[0])) > 1e-6
    cut = jax.jit(jax.grad(objective, argnums=1))(params, x, reset.at[3].set(True))
    np.testing.assert_allclose(cut[:3], 0., atol=1e-7)


def test_prefix_replay_reconstructs_current_weights_and_is_detached(model):
    actor, params, norm = model
    history = jax.random.normal(jax.random.key(4), (3, 6, 49))
    lengths = jp.array([0, 2, 6])
    params = jax.tree.map(lambda a: a+.003, params)
    carry = jax.jit(actor.replay)(norm, params, history, lengths)
    for b, length in enumerate(np.asarray(lengths)):
        c = zero_carry(1)
        for i in range(int(length)):
            c, _ = actor.step(norm, params, c, dict(state=history[b:b+1, i]), jp.array([False]))
        for actual, expected in zip(carry, c):
            np.testing.assert_allclose(actual[b], expected[0], atol=2e-5)
    def objective(h):
        return sum(jp.square(a).sum() for a in actor.replay(norm, params, h, lengths))
    np.testing.assert_array_equal(jax.grad(objective)(history), 0.)


def test_actor_cannot_read_privileged_state(model):
    actor, params, norm = model
    x = jp.ones((2, 49)); reset = jp.ones(2, bool)
    a = actor.step(norm, params, zero_carry(2), dict(state=x, privileged=jp.zeros((2, 66))), reset)[1]
    b = actor.step(norm, params, zero_carry(2), dict(state=x, privileged=jp.ones((2, 66))*1e5), reset)[1]
    np.testing.assert_array_equal(a, b)


def test_checkpoint_restores_architecture_distribution_and_actions(model, tmp_path):
    actor, params, norm = model
    config = dict(actor='mlp_lstm', recurrent_width=128, network=NETWORK,
        actor_architecture=ARCH, exploration=dict(arm_min_std=.25, gripper_min_std=.5))
    (tmp_path/'config.json').write_text(json.dumps(config))
    cp = tmp_path/'checkpoint'; cp.mkdir()
    metadata = dict(actor='mlp_lstm', width=128, action_size=7,
        observation_size={k: list(v) for k, v in SIZES.items()}, exploration=config['exploration'])
    (cp/'metadata.json').write_text(json.dumps(metadata))
    (cp/'learner.msgpack').write_bytes(serialization.to_bytes(dict(params=dict(policy=params), normalizer=norm)))
    loaded, p, n = load_actor_checkpoint(tmp_path, cp, SIZES, 7)
    obs = dict(state=jp.ones((2, 49))); reset = jp.ones(2, bool)
    x = actor.step(norm, params, zero_carry(2), obs, reset)[1]
    y = loaded.step(n, p, zero_carry(2), obs, reset)[1]
    np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(actor.dist.sample(x, jax.random.key(8)), loaded.dist.sample(y, jax.random.key(8)))
    config['actor_architecture'] = dict(hidden_sizes=[128, 64])
    (tmp_path/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='parameter shapes'):
        load_actor_checkpoint(tmp_path, cp, SIZES, 7)


@pytest.mark.parametrize('arch', [dict(hidden_sizes=[]), dict(hidden_sizes=[64, 0]),
    dict(hidden_sizes=[True, 64]), dict(hidden_sizes='64'), dict(heads=4)])
def test_invalid_encoder_architecture_is_rejected(arch):
    with pytest.raises(ValueError, match='MLP encoder'):
        ActorAdapter(SIZES, 7, 'mlp_lstm', NETWORK, actor_config=arch)
