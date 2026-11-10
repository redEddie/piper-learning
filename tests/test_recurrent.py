"""Memory/reset, time-order gradients, and replay integrity (CPU only)."""
import jax
import jax.numpy as jnp
import numpy as np
from brax.training.acme import running_statistics as stats
from piper_sim_lab.recurrent import ActorAdapter, zero_carry, with_memory, write_history


def setup_actor(kind="lstm"):
    sizes={"state":(5,),"privileged":(8,)}
    actor=ActorAdapter(sizes,2,kind,{"policy_hidden_layer_sizes":[16,16],
        "value_hidden_layer_sizes":[16,16],"policy_obs_key":"state","value_obs_key":"privileged"})
    params=actor.init(jax.random.key(1))
    norm=stats.init_state({k:jnp.zeros(v) for k,v in sizes.items()})
    return actor,params,norm


def test_sequence_matches_step_and_resets_each_slot():
    actor,params,norm=setup_actor()
    x=jax.random.normal(jax.random.key(2),(7,3,5))
    reset=jnp.zeros((7,3),bool).at[0].set(True).at[3,1].set(True)
    carry=zero_carry(3)
    observations=[]; outputs=[]
    for i in range(7):
        observations.append(with_memory({"state":x[i]},carry,reset[i]))
        carry,y=actor.step(norm,params,carry,{"state":x[i]},reset[i])
        outputs.append(y)
    obs=jax.tree.map(lambda *xs:jnp.stack(xs),*observations)
    sequence=jax.jit(actor.sequence_logits)(norm,params,obs)
    np.testing.assert_allclose(sequence,jnp.stack(outputs),atol=1e-5)
    _,fresh=actor.step(norm,params,zero_carry(1),{"state":x[3,1:2]},jnp.ones(1,bool))
    np.testing.assert_allclose(sequence[3,1],fresh[0],atol=1e-5)
    # Resetting slot 1 must not reset other environments.
    reset_none=reset.at[3,1].set(False)
    altered=dict(obs,episode_start=reset_none)
    altered_y=actor.sequence_logits(norm,params,altered)
    np.testing.assert_allclose(sequence[:,0],altered_y[:,0],atol=1e-5)
    assert np.linalg.norm(np.asarray(sequence[3,1]-altered_y[3,1]))>1e-4


def test_replay_matches_prefix_after_parameter_update():
    actor,params,norm=setup_actor()
    hist=jax.random.normal(jax.random.key(3),(3,6,5));lens=jnp.array([0,2,6])
    for p in [params,jax.tree.map(lambda x:x+.01,params)]:
        carry=actor.replay(norm,p,hist,lens)
        for b,l in enumerate(np.asarray(lens)):
            c=zero_carry(1)
            for i in range(int(l)):
                c,_=actor.step(norm,p,c,{"state":hist[b:b+1,i]},jnp.array([False]))
            for a,v in zip(carry,c):np.testing.assert_allclose(a[b],v[0],atol=1e-5)


def test_history_clears_only_finished_slot():
    hist=jnp.ones((2,4,5));lens=jnp.array([2,1]);x=jnp.full((2,5),3.)
    hist,lens=write_history(hist,lens,x,jnp.array([True,False]))
    np.testing.assert_array_equal(lens,[0,2])
    np.testing.assert_array_equal(hist[0],0.)
    np.testing.assert_array_equal(hist[1,1],3.)


def test_temporal_gradient_stops_at_reset():
    actor,params,norm=setup_actor()
    x=jax.random.normal(jax.random.key(4),(5,1,5))
    carry=zero_carry(1)
    obs={"state":x,"memory_c":jnp.zeros((5,1,128)),"memory_h":jnp.zeros((5,1,128)),
         "episode_start":jnp.array([[True],[False],[False],[False],[False]])}
    def target(x,reset):
        return actor.sequence_logits(norm,params,dict(obs,state=x,episode_start=reset))[-1].sum()
    g=jax.grad(target)(x,obs['episode_start'])
    assert float(jnp.linalg.norm(g[0]))>1e-5
    g=jax.grad(target)(x,obs['episode_start'].at[3].set(True))
    np.testing.assert_allclose(g[:3],0.,atol=1e-7)


def test_actor_cannot_read_privileged_state():
    actor,params,norm=setup_actor()
    x=jnp.ones((3,5));reset=jnp.ones(3,bool)
    c=zero_carry(3)
    a=actor.step(norm,params,c,{"state":x,"privileged":jnp.zeros((3,8))},reset)[1]
    b=actor.step(norm,params,c,{"state":x,"privileged":jnp.full((3,8),1e6)},reset)[1]
    np.testing.assert_array_equal(a,b)


def test_checkpoint_roundtrip_preserves_action():
    from flax import serialization
    actor,params,norm=setup_actor()
    blob=serialization.to_bytes({'policy':params,'normalizer':norm})
    restored=serialization.msgpack_restore(blob)
    restored_norm=serialization.from_state_dict(norm,restored['normalizer'])
    x={'state':jnp.ones((2,5))};reset=jnp.ones(2,bool)
    before=actor.step(norm,params,zero_carry(2),x,reset)[1]
    after=actor.step(restored_norm,restored['policy'],zero_carry(2),x,reset)[1]
    np.testing.assert_array_equal(before,after)


def test_dynamic_prefix_replay_allows_sgd_without_differentiating_burn_in():
    actor,params,norm=setup_actor()
    hist=jax.random.normal(jax.random.key(5),(2,6,5));lengths=jnp.array([2,5])
    x=jax.random.normal(jax.random.key(6),(4,2,5))
    def objective(p):
        c,h=actor.replay(norm,p,hist,lengths)
        obs={'state':x,'memory_c':jnp.broadcast_to(c,(4,2,128)),
             'memory_h':jnp.broadcast_to(h,(4,2,128)), 'episode_start':jnp.zeros((4,2),bool)}
        return jnp.square(actor.sequence_logits(norm,p,obs)).mean()
    loss,g=jax.jit(jax.value_and_grad(objective))(params)
    assert np.isfinite(float(loss))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(g))
    assert sum(float(jnp.square(x).sum()) for x in jax.tree.leaves(g))>0
