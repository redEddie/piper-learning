"""Encoder-first LSTM parity, memory reset and export replay regressions."""
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict

from piper_sim_lab.mlp_lstm import MLPLSTMActor
from piper_policy_runtime import Policy
from piper_policy_runtime.check import check


@pytest.fixture
def exported(tmp_path):
    module = MLPLSTMActor(action_size=2, width=5, hidden_sizes=(8, 4))
    carry = (jnp.zeros((3, 5)), jnp.zeros((3, 5)))
    params = module.init(jax.random.key(108), carry, jnp.zeros((3, 6)))
    weights = {key: np.asarray(value) for key, value in flatten_dict(params, sep='/').items()}
    weights['normalizer/mean'] = np.linspace(-.2, .2, 6, dtype=np.float32)
    weights['normalizer/std'] = np.linspace(.2, 1., 6, dtype=np.float32)
    for name, suffix in [('input_kernel', 'i{gate}/kernel'),
                         ('hidden_kernel', 'h{gate}/kernel'), ('bias', 'h{gate}/bias')]:
        weights['runtime/lstm_' + name] = np.concatenate([
            weights['params/lstm/' + suffix.format(gate=gate)] for gate in 'ifgo'], axis=-1)
    np.savez_compressed(tmp_path / 'weights.npz', **weights)
    meta = dict(schema_version=1, actor='mlp_lstm', recurrent_width=5,
                encoder_sizes=[8, 4], head_sizes=[], observation_size=6, action_size=2,
                weights_sha256=hashlib.sha256((tmp_path / 'weights.npz').read_bytes()).hexdigest())
    (tmp_path / 'policy.json').write_text(json.dumps(meta))
    obs = np.random.default_rng(108).normal(0, .5, (12, 3, 6)).astype(np.float32)
    reset = np.zeros((12, 3), dtype=bool)
    reset[0] = True
    reset[5, 1] = True
    reset[9] = True
    logits, cells, hidden = [], [], []
    for sample, mask in zip(obs, reset):
        carry = tuple(jnp.where(mask[:, None], 0., value) for value in carry)
        normalized = (sample - weights['normalizer/mean']) / weights['normalizer/std']
        carry, output = module.apply(params, carry, normalized)
        logits.append(np.asarray(output))
        cells.append(np.asarray(carry[0]))
        hidden.append(np.asarray(carry[1]))
    vectors = dict(observations=obs, resets=reset, logits=np.asarray(logits),
                   memory_c=np.asarray(cells), memory_h=np.asarray(hidden),
                   actions=np.tanh(np.asarray(logits)[..., :2]))
    np.savez_compressed(tmp_path / 'golden.npz', **vectors)
    return tmp_path, vectors


def test_mlp_encoder_then_lstm_matches_flax_with_partial_resets(exported):
    directory, vectors = exported
    report = check(directory, tolerance=1e-5)
    assert report['actor'] == 'mlp_lstm'
    assert max(report['state_and_logit_max_abs_error'].values()) < 1e-5
    policy = Policy(directory)
    np.testing.assert_allclose(policy.step(vectors['observations'][0, 0], reset=True),
                               vectors['actions'][0, 0], atol=1e-5)


def test_invalid_input_preserves_encoder_first_lstm_memory(exported):
    directory, vectors = exported
    policy = Policy(directory)
    policy.step(vectors['observations'][0, 0])
    before = (policy.c.copy(), policy.h.copy())
    with pytest.raises(ValueError, match='Nonfinite observation'):
        policy.step(np.full(6, np.nan), reset=True)
    for actual, expected in zip((policy.c, policy.h), before):
        np.testing.assert_array_equal(actual, expected)


def test_replay_rejects_incorrect_memory_even_when_actions_match(exported):
    directory, vectors = exported
    vectors['memory_c'][6, 1, 0] += .1
    np.savez_compressed(directory / 'golden.npz', **vectors)
    with pytest.raises(AssertionError, match='recurrent/logit'):
        check(directory)
