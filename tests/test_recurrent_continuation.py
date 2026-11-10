"""Continuation restores optimizer momentum, normalization and RNG, not just weights."""
from copy import deepcopy
import json

from flax import serialization
import jax
import jax.numpy as jp
import numpy as np
import optax
import pytest
from brax.training.acme import running_statistics as stats
from brax.training.agents.ppo.losses import PPONetworkParams

from piper_sim_lab.recurrent import restore_learner_checkpoint


def checkpoint(tmp_path):
    cfg = dict(actor='transformer_lstm', recurrent_width=128,
        actor_architecture=dict(token_width=64, heads=4, layers=2, feedforward_width=128),
        network=dict(policy_obs_key='state', value_obs_key='privileged'),
        training=dict(learning_rate=.0003, max_grad_norm=1.))
    params = PPONetworkParams(dict(w=jp.arange(6.,dtype=jp.float32).reshape(2,3)), dict(w=jp.ones(3)))
    optimizer = optax.chain(optax.clip_by_global_norm(1.),optax.adam(.0003))
    norm = stats.init_state(dict(state=jp.zeros(49),privileged=jp.zeros(66)))
    template = dict(params=params,normalizer=norm,optimizer=optimizer.init(params),
                    rng=jax.random.PRNGKey(7),step=0)
    grads = jax.tree.map(lambda a:jp.full_like(a,.1), params)
    updates, state = optimizer.update(grads, template['optimizer'],params)
    trained = dict(params=optax.apply_updates(params,updates),
        optimizer=state,normalizer=stats.update(norm,dict(state=jp.ones((3,49)),privileged=jp.ones((3,66)))),
        rng=jax.random.PRNGKey(13),step=67108864)
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    path=tmp_path/'checkpoints'/'000067108864';path.mkdir(parents=True)
    (path/'metadata.json').write_text(json.dumps(dict(actor=cfg['actor'],width=128,step=67108864)))
    (path/'learner.msgpack').write_bytes(serialization.to_bytes(trained))
    return cfg,path,template,trained,optimizer,grads


def test_restored_next_optimizer_update_matches_parent_and_rng(tmp_path):
    cfg,path,template,trained,opt,grads=checkpoint(tmp_path)
    restored=restore_learner_checkpoint(path,template,cfg)
    for a,b in zip(jax.tree.leaves(restored),jax.tree.leaves(trained)):
        np.testing.assert_array_equal(a,b)
    a=opt.update(grads,restored['optimizer'],restored['params'])
    b=opt.update(grads,trained['optimizer'],trained['params'])
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        np.testing.assert_array_equal(x,y)
    assert int(restored['normalizer'].count) == 3


@pytest.mark.parametrize('kind',['architecture','optimizer','shape','nonfinite'])
def test_mismatched_or_corrupt_learner_is_rejected(tmp_path,kind):
    cfg,path,template,trained,_,_=checkpoint(tmp_path)
    if kind=='architecture':
        cfg=deepcopy(cfg);cfg['actor_architecture']['token_width']=32
    elif kind=='optimizer':
        cfg=deepcopy(cfg);cfg['training']['learning_rate']=.1
    else:
        trained['params']=trained['params'].replace(policy=dict(w=jp.ones((1,3)) if kind=='shape' else jp.full((2,3),jp.nan)))
        (path/'learner.msgpack').write_bytes(serialization.to_bytes(trained))
    with pytest.raises(ValueError):
        restore_learner_checkpoint(path,template,cfg)
