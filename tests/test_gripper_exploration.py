"""Axis-specific exploration must stay on-policy and preserve deterministic actions."""
import json

from flax import serialization
import jax
import jax.numpy as jp
import numpy as np
import pytest
from brax.training.acme import running_statistics as stats

from piper_sim_lab.recurrent import ActorAdapter, load_actor_checkpoint, zero_carry

SIZES = dict(state=(49,),privileged=(66,))
NETWORK = dict(policy_hidden_layer_sizes=[16],value_hidden_layer_sizes=[16],
               policy_obs_key='state',value_obs_key='privileged')


def actor(exploration=None):
    return ActorAdapter(SIZES,7,'mlp',NETWORK,exploration_config=exploration)


def test_only_gripper_samples_widen_and_deterministic_actions_stay_identical():
    old,new=actor(),actor(dict(gripper_min_std=.4))
    logits=jp.concatenate([jp.zeros((65536,7)),jp.full((65536,7),-12.)],axis=-1)
    key=jax.random.key(3)
    a=old.dist.sample(logits,key);b=new.dist.sample(logits,key)
    np.testing.assert_array_equal(a[...,:6],b[...,:6])
    np.testing.assert_array_equal(old.dist.mode(logits),new.dist.mode(logits))
    assert np.isfinite(np.asarray(b)).all() and float(jp.max(jp.abs(b))) <= 1.
    assert float(jp.mean(a[...,6] >= .5)) == 0.
    assert .075 < float(jp.mean(b[...,6] >= .5)) < .095
    assert .075 < float(jp.mean(b[...,6] <= -.5)) < .095
    np.testing.assert_array_equal(new.dist.create_dist(logits).loc,old.dist.create_dist(logits).loc)


def test_collector_and_ppo_share_probabilities_and_policy_mean_gets_gradients():
    a=actor(dict(gripper_min_std=.4));d=a.networks.parametric_action_distribution
    logits=jp.array([.1,-.2,0.,.3,-.4,0.,.05]+[-3.]*7)
    raw=a.dist.sample_no_postprocessing(logits,jax.random.key(4))
    np.testing.assert_array_equal(a.dist.log_prob(logits,raw),d.log_prob(logits,raw))
    np.testing.assert_array_equal(a.dist.entropy(logits,jax.random.key(5)),d.entropy(logits,jax.random.key(5)))
    assert abs(float(d.create_dist(logits).kl_divergence(a.dist.create_dist(logits)))) < 1e-3
    gradient=jax.grad(lambda x:d.log_prob(x,raw))(logits)
    assert np.isfinite(np.asarray(gradient)).all() and abs(float(gradient[6])) > 1e-4


def test_arm_and_gripper_floors_widen_all_seven_axes_without_changing_means():
    old=actor();new=actor(dict(gripper_min_std=.4,arm_min_std=.2))
    logits=jp.concatenate([jp.linspace(-.3,.3,7),jp.full(7,-12.)])
    a,b=old.dist.create_dist(logits),new.dist.create_dist(logits)
    np.testing.assert_array_equal(a.loc,b.loc)
    np.testing.assert_allclose(b.scale[:6],.2,atol=1e-5)
    assert .4 <= float(b.scale[6]) < .401
    np.testing.assert_array_equal(old.dist.mode(logits),new.dist.mode(logits))
    np.testing.assert_allclose(b.scale-a.scale,jp.array([.199]*6+[.399]),atol=1e-7)


def test_checkpoint_reload_preserves_exploration_and_rejects_mismatch(tmp_path):
    options=dict(gripper_min_std=.4);a=actor(options)
    params=a.init(jax.random.key(7))
    norm=stats.init_state({k:jp.zeros(v) for k,v in SIZES.items()})
    cfg=dict(actor='mlp',network=NETWORK,recurrent_width=128,exploration=options)
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    path=tmp_path/'checkpoint';path.mkdir()
    metadata=dict(actor='mlp',width=128,exploration=options)
    (path/'metadata.json').write_text(json.dumps(metadata))
    (path/'learner.msgpack').write_bytes(serialization.to_bytes(dict(params=dict(policy=params),normalizer=norm)))
    loaded,p,n=load_actor_checkpoint(tmp_path,path,SIZES,7)
    obs=dict(state=jp.ones((2,49)));reset=jp.ones(2,bool)
    x=a.step(norm,params,zero_carry(2),obs,reset)[1]
    y=loaded.step(n,p,zero_carry(2),obs,reset)[1]
    np.testing.assert_array_equal(a.dist.sample(x,jax.random.key(8)),loaded.dist.sample(y,jax.random.key(8)))
    metadata['exploration']=dict(gripper_min_std=.1)
    (path/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='exploration metadata'):
        load_actor_checkpoint(tmp_path,path,SIZES,7)


@pytest.mark.parametrize('kind', ['transformer_lstm', 'mlp_lstm'])
def test_recurrent_encoder_upstream_ppo_loss_keeps_matching_ratios_and_gradients(kind):
    from brax.training import types
    from brax.training.agents.ppo import losses
    architecture = (dict(token_width=16,heads=2,layers=1,feedforward_width=32)
                    if kind == 'transformer_lstm' else dict(hidden_sizes=[32,16]))
    a=ActorAdapter(SIZES,7,kind,NETWORK,
        actor_config=architecture,
        exploration_config=dict(gripper_min_std=.4,arm_min_std=.2))
    p=losses.PPONetworkParams(a.init(jax.random.key(10)),a.base.value_network.init(jax.random.key(11)))
    norm=stats.init_state({k:jp.zeros(v) for k,v in SIZES.items()})
    obs=dict(state=jax.random.normal(jax.random.key(12),(3,2,49)),privileged=jp.ones((3,2,66)),
        memory_c=jp.zeros((3,2,128)),memory_h=jp.zeros((3,2,128)),
        episode_start=jp.zeros((3,2),bool).at[0].set(True))
    logits=a.sequence_logits(norm,p.policy,obs)
    raw=a.dist.sample_no_postprocessing(logits,jax.random.key(13))
    np.testing.assert_array_equal(a.dist.log_prob(logits,raw),a.networks.parametric_action_distribution.log_prob(logits,raw))
    data=types.Transition(obs,a.dist.postprocess(raw),jp.arange(6.).reshape(3,2),jp.ones((3,2)),obs,
        dict(policy_extras=dict(raw_action=raw,log_prob=a.dist.log_prob(logits,raw),distribution_params=logits),
             state_extras=dict(truncation=jp.zeros((3,2)))))
    data=jax.tree.map(lambda x:jp.swapaxes(x,0,1),data)
    def objective(params):
        return losses.compute_ppo_loss(params,norm,data,jax.random.key(14),ppo_network=a.networks)[0]
    value,grad=jax.jit(jax.value_and_grad(objective))(p)
    assert np.isfinite(float(value))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(grad))
    assert abs(float(grad.policy['params']['distribution']['bias'][6])) > 1e-6


@pytest.mark.parametrize('minimum',[float('nan'),float('inf'),-.1,0.])
def test_invalid_exploration_is_rejected(minimum):
    with pytest.raises(ValueError,match='gripper_min_std'):
        actor(dict(gripper_min_std=minimum))
