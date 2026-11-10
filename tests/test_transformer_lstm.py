"""Behavioral memory, reset isolation, causality, gradients and checkpoint use."""
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.training.acme import running_statistics as stats

from piper_sim_lab.recurrent import ActorAdapter, zero_carry, with_memory, load_actor_checkpoint
from piper_sim_lab.transformer_lstm import TOKEN_GROUPS


NETWORK = dict(policy_hidden_layer_sizes=[256,256], value_hidden_layer_sizes=[256,256],
               policy_obs_key='state', value_obs_key='privileged')
SIZES = dict(state=(49,), privileged=(66,))


@pytest.fixture(scope='module')
def model():
    actor = ActorAdapter(SIZES, 7, 'transformer_lstm', NETWORK)
    params = actor.init(jax.random.key(1))
    norm = stats.init_state({k:jp.zeros(v) for k,v in SIZES.items()})
    return actor, params, norm


def sequence(actor, norm, params, x, reset):
    obs = dict(state=x, memory_c=jp.zeros(x.shape[:2]+(128,)),
               memory_h=jp.zeros(x.shape[:2]+(128,)), episode_start=reset)
    return actor.sequence_logits(norm, params, obs)


def test_role_tokens_partition_actor_only_and_memory_is_last(model):
    indices = [i for group in TOKEN_GROUPS.values() for i in group]
    assert sorted(indices) == list(range(49))
    assert len(indices) == len(set(indices))
    _, params, _ = model
    assert params['params']['distribution']['kernel'].shape == (128,14)
    assert not any(k.startswith('head_') for k in params['params'])


def test_streaming_equals_sequence_with_independent_episode_resets(model):
    actor, params, norm = model
    x = jax.random.normal(jax.random.key(2), (6,2,49))
    resets = jp.zeros((6,2),bool).at[0].set(True).at[3,1].set(True)
    carry, outputs = zero_carry(2), []
    for i in range(6):
        carry, y = actor.step(norm, params, carry, dict(state=x[i]), resets[i])
        outputs.append(y)
    actual = jax.jit(lambda p,x,r:sequence(actor,norm,p,x,r))(params,x,resets)
    np.testing.assert_allclose(actual, jp.stack(outputs), atol=2e-5)
    _, fresh = actor.step(norm, params, zero_carry(1), dict(state=x[3,1:2]), jp.ones(1,bool))
    np.testing.assert_allclose(actual[3,1], fresh[0], atol=2e-5)
    no_reset = sequence(actor,norm,params,x,resets.at[3,1].set(False))
    np.testing.assert_allclose(actual[:,0], no_reset[:,0], atol=2e-5)
    assert float(jp.linalg.norm(actual[3,1]-no_reset[3,1])) > 1e-4
    changed_future = sequence(actor,norm,params,x.at[4:].add(10.),resets)
    np.testing.assert_allclose(actual[:4], changed_future[:4], atol=2e-5)


def test_attention_and_lstm_receive_gradients_and_reset_cuts_time(model):
    actor, params, norm = model
    x = jax.random.normal(jax.random.key(3), (5,1,49))
    reset = jp.zeros((5,1),bool).at[0].set(True)
    def objective(p,x,r):
        return jp.square(sequence(actor,norm,p,x,r)[-1]).sum()
    value, gradients = jax.jit(jax.value_and_grad(objective))(params,x,reset)
    assert np.isfinite(float(value))
    for name in ('relation_0','relation_1','lstm'):
        leaves = jax.tree.leaves(gradients['params'][name])
        assert all(np.isfinite(np.asarray(a)).all() for a in leaves)
        assert sum(float(jp.square(a).sum()) for a in leaves) > 1e-9
    grad = jax.jit(jax.grad(objective, argnums=1))(params,x,reset)
    assert float(jp.linalg.norm(grad[0])) > 1e-6
    cut = jax.jit(jax.grad(objective, argnums=1))(params,x,reset.at[3].set(True))
    np.testing.assert_allclose(cut[:3],0.,atol=1e-7)


def test_current_weight_prefix_replay_matches_streaming(model):
    actor, params, norm = model
    history = jax.random.normal(jax.random.key(4), (3,6,49))
    lengths = jp.array([0,2,6])
    updated = jax.tree.map(lambda a:a+.003, params)
    carry = jax.jit(actor.replay)(norm,updated,history,lengths)
    for b,length in enumerate(np.asarray(lengths)):
        c = zero_carry(1)
        for i in range(int(length)):
            c,_ = actor.step(norm,updated,c,dict(state=history[b:b+1,i]),jp.array([False]))
        for actual,expected in zip(carry,c):
            np.testing.assert_allclose(actual[b],expected[0],atol=2e-5)
    def prefix_value(h):
        c = actor.replay(norm,updated,h,lengths)
        return sum(jp.square(a).sum() for a in c)
    np.testing.assert_array_equal(jax.grad(prefix_value)(history), 0.)


def test_critic_privileged_state_cannot_change_actor(model):
    actor, params, norm = model
    x = jp.ones((2,49)); reset=jp.ones(2,bool)
    a = actor.step(norm,params,zero_carry(2),dict(state=x,privileged=jp.zeros((2,66))),reset)[1]
    b = actor.step(norm,params,zero_carry(2),dict(state=x,privileged=jp.ones((2,66))*1e5),reset)[1]
    np.testing.assert_array_equal(a,b)


def test_saved_checkpoint_roundtrip_and_metadata_guard(model, tmp_path):
    actor, params, norm = model
    cfg=dict(actor='transformer_lstm',recurrent_width=128,network=NETWORK,actor_architecture=None)
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    path=tmp_path/'checkpoint';path.mkdir()
    metadata=dict(actor='transformer_lstm',width=128,observation_size={k:list(v) for k,v in SIZES.items()},action_size=7)
    (path/'metadata.json').write_text(json.dumps(metadata))
    (path/'learner.msgpack').write_bytes(serialization.to_bytes(dict(params=dict(policy=params),normalizer=norm)))
    loaded,p,n = load_actor_checkpoint(tmp_path,path,SIZES,7)
    obs=dict(state=jp.ones((2,49)));reset=jp.ones(2,bool)
    a=actor.step(norm,params,zero_carry(2),obs,reset)[1]
    b=loaded.step(n,p,zero_carry(2),obs,reset)[1]
    np.testing.assert_array_equal(a,b)
    metadata['action_size']=6;(path/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='action dimension'):
        load_actor_checkpoint(tmp_path,path,SIZES,7)


def test_invalid_role_layout_and_attention_dimensions_rejected():
    with pytest.raises(ValueError,match='actor49'):
        ActorAdapter(dict(state=(48,),privileged=(66,)),7,'transformer_lstm',NETWORK)
    with pytest.raises(ValueError,match='divisible'):
        ActorAdapter(SIZES,7,'transformer_lstm',NETWORK,actor_config=dict(heads=3))
