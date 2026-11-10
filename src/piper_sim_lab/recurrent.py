"""Small stateful actor and sequence adapter; upstream Brax owns PPO loss/GAE.

Carry order is (cell, hidden). Raw episode prefixes are replayed with current
weights before each update, then detached for truncated BPTT. Only real actor
observations enter memory; privileged state remains exclusive to the critic.
"""
from flax import linen as nn
import jax
import jax.numpy as jnp
from brax.training import networks as base_networks
from brax.training.acme import running_statistics as stats
from brax.training.agents.ppo import networks as ppo_networks


class LSTMActor(nn.Module):
    action_size: int
    width: int = 128
    head_sizes: tuple = (256, 256)

    @nn.compact
    def __call__(self, carry, x):
        carry, x = nn.OptimizedLSTMCell(self.width, name="lstm")(carry, x)
        x = nn.LayerNorm(name="recurrent_norm")(x)
        for i, size in enumerate(self.head_sizes):
            x = nn.swish(nn.Dense(size, name=f"head_{i}")(x))
        return carry, nn.Dense(2 * self.action_size, name="distribution")(x)


def zero_carry(batch, width=128):
    return (jnp.zeros((batch, width)), jnp.zeros((batch, width)))


def reset_carry(carry, reset):
    return jax.tree.map(lambda x: jnp.where(reset[..., None], 0., x), carry)


class ActorAdapter:
    def __init__(self, sizes, action_size, kind, network_config, width=128, actor_config=None, exploration_config=None):
        if kind not in ("mlp", "lstm", "transformer_lstm", "mlp_lstm"):
            raise ValueError(kind)
        self.kind, self.width = kind, width
        self.base = ppo_networks.make_ppo_networks(
            sizes, action_size, preprocess_observations_fn=stats.normalize,
            **network_config)
        self.module = LSTMActor(action_size, width,
                                tuple(network_config["policy_hidden_layer_sizes"]))
        if kind == 'transformer_lstm':
            from .transformer_lstm import TransformerLSTMActor
            options = dict(token_width=64, heads=4, layers=2, feedforward_width=128)
            options.update(actor_config or {})
            if any(type(v) is not int or v <= 0 for v in options.values()):
                raise ValueError('Transformer dimensions must be positive integers')
            if options['token_width'] % options['heads']:
                raise ValueError('Transformer token width must be divisible by heads')
            if sizes[network_config['policy_obs_key']] != (49,):
                raise ValueError('Transformer role tokens require actor49')
            self.module = TransformerLSTMActor(action_size, width, **options)
        elif kind == 'mlp_lstm':
            from .mlp_lstm import MLPLSTMActor
            options = dict(hidden_sizes=(256, 64))
            options.update(actor_config or {})
            if set(options) != {'hidden_sizes'}:
                raise ValueError('MLP encoder accepts only hidden_sizes')
            hidden = options['hidden_sizes']
            if (not isinstance(hidden, (list, tuple)) or not hidden or
                any(type(v) is not int or v <= 0 for v in hidden)):
                raise ValueError('MLP encoder hidden_sizes must be positive integers')
            self.module = MLPLSTMActor(action_size, width, tuple(hidden))
        self.policy_key = network_config["policy_obs_key"]
        self.sizes = sizes
        if kind != "mlp":
            self.networks = self.base.replace(policy_network=base_networks.FeedForwardNetwork(
                init=self.init, apply=self.sequence_logits))
        else:
            self.networks = self.base
        self.dist = self.base.parametric_action_distribution
        if exploration_config:
            import math
            from brax.training.distribution import NormalTanhDistribution
            if ('gripper_min_std' not in exploration_config or
                set(exploration_config)-{'gripper_min_std','arm_min_std'} or action_size != 7):
                raise ValueError('Exploration requires gripper_min_std, optional arm_min_std and seven actions')
            minimum = exploration_config['gripper_min_std']
            arm_minimum = exploration_config.get('arm_min_std', .001)
            for name,value in (('gripper_min_std',minimum),('arm_min_std',arm_minimum)):
                if not math.isfinite(value) or value < .001:
                    raise ValueError(f'{name} must be finite and at least0.001')
            # Upstream NormalTanh accepts axis-specific scale minima.
            # PPO log_prob/entropy/KL and collection all
            # use this same distribution, without post-sampling noise.
            self.dist = NormalTanhDistribution(action_size,
                min_std=jnp.array([arm_minimum]*6+[minimum]))
            self.networks = self.networks.replace(parametric_action_distribution=self.dist)

    def init(self, key):
        if self.kind == "mlp":
            return self.base.policy_network.init(key)
        return self.module.init(key, zero_carry(1, self.width),
                                jnp.zeros((1, self.sizes[self.policy_key][-1])))

    def normalize_actor(self, normalizer, x):
        return stats.normalize(x, base_networks.normalizer_select(normalizer, self.policy_key))

    def step(self, normalizer, params, carry, obs, reset):
        carry = reset_carry(carry, reset)
        if self.kind == "mlp":
            return carry, self.base.policy_network.apply(normalizer, params, obs)
        return self.module.apply(params, carry,
                                 self.normalize_actor(normalizer, obs[self.policy_key]))

    def replay(self, normalizer, params, history, lengths):
        """Rebuild from the actual episode start, never from a stale parameter cache."""
        carry = zero_carry(history.shape[0], self.width)
        if self.kind == "mlp":
            return carry
        xs = jnp.swapaxes(self.normalize_actor(normalizer, history), 0, 1)

        def f(i, c):
            nxt, _ = self.module.apply(params, c, xs[i])
            return jax.tree.map(lambda a, b: jnp.where((i < lengths)[:, None], a, b), nxt, c)
        # No reverse differentiation is needed through the burn-in prefix.
        # A dynamic bound avoids evaluating padded memory steps at every update.
        carry = jax.lax.fori_loop(0, jnp.max(lengths), f, carry)
        return jax.tree.map(jax.lax.stop_gradient, carry)

    def sequence_logits(self, normalizer, params, obs):
        """Brax loss passes time-major [T,B,D] observations, not shuffled steps."""
        carry = (obs["memory_c"][0], obs["memory_h"][0])
        def f(c, item):
            x, reset = item
            return self.step(normalizer, params, c, {self.policy_key: x}, reset)
        _, logits = jax.lax.scan(f, carry, (obs[self.policy_key], obs["episode_start"]))
        return logits


def with_memory(obs, carry, reset):
    return {**obs, "memory_c": carry[0], "memory_h": carry[1], "episode_start": reset}


def write_history(history, lengths, x, done):
    """History is per slot, and excludes the fresh observation after autoreset."""
    history = history.at[jnp.arange(x.shape[0]), jnp.minimum(lengths, history.shape[1]-1)].set(x)
    lengths = jnp.where(done, 0, lengths + 1)
    history = jnp.where(done[:, None, None], 0., history)
    return history, lengths


def load_actor_checkpoint(run, checkpoint, sizes, action_size):
    """Restore this adapter's checkpoint; never interpret it as feed-forward PPO."""
    import json
    from pathlib import Path
    from flax import serialization
    cfg = json.loads((Path(run)/'config.json').read_text())
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint/'metadata.json').read_text())
    if metadata['actor'] != cfg['actor'] or metadata['width'] != cfg['recurrent_width']:
        raise ValueError('Recurrent actor metadata differs from the run configuration')
    if metadata.get('action_size', action_size) != action_size:
        raise ValueError('Recurrent action dimension mismatch')
    if metadata.get('observation_size', {k:list(v) for k,v in sizes.items()}) != {k:list(v) for k,v in sizes.items()}:
        raise ValueError('Recurrent observation layout mismatch')
    actor = ActorAdapter(sizes, action_size, cfg['actor'], cfg['network'], cfg['recurrent_width'],
                         cfg.get('actor_architecture'), cfg.get('exploration'))
    if metadata.get('exploration',cfg.get('exploration')) != cfg.get('exploration'):
        raise ValueError('Recurrent exploration metadata differs from the run configuration')
    restored = serialization.msgpack_restore((checkpoint/'learner.msgpack').read_bytes())
    params = jax.tree.map(jnp.asarray, restored['params']['policy'])
    reference = actor.init(jax.random.key(0))
    if (jax.tree.structure(params) != jax.tree.structure(reference) or
        any(a.shape != b.shape for a,b in zip(jax.tree.leaves(params), jax.tree.leaves(reference)))):
        raise ValueError('Recurrent parameter shapes differ from the specified architecture')
    normalizer = serialization.from_state_dict(stats.init_state({k:jnp.zeros(v) for k,v in sizes.items()}),
                                               restored['normalizer'])
    return actor, params, normalizer


def restore_learner_checkpoint(checkpoint, template, config):
    """Continue the same learner, with fresh simulator episodes and empty carry.

    Reward/budget/reset changes are allowed; architecture, observation layout
    and optimizer settings must match. Counters in the new run are additional.
    """
    import json
    from pathlib import Path
    from flax import serialization
    import numpy as np
    checkpoint = Path(checkpoint).resolve()
    parent = json.loads((checkpoint.parent.parent/'config.json').read_text())
    metadata = json.loads((checkpoint/'metadata.json').read_text())
    for key in ('actor', 'recurrent_width', 'actor_architecture', 'network'):
        if parent.get(key) != config.get(key):
            raise ValueError(f'Continuation configuration mismatch: {key}')
    if metadata['actor'] != config['actor'] or metadata['width'] != config['recurrent_width']:
        raise ValueError('Continuation checkpoint actor metadata mismatch')
    for key in ('learning_rate', 'max_grad_norm'):
        if parent['training'][key] != config['training'][key]:
            raise ValueError(f'Continuation optimizer mismatch: {key}')
    restored = serialization.from_bytes(template, (checkpoint/'learner.msgpack').read_bytes())
    if (jax.tree.structure(restored) != jax.tree.structure(template) or
        any(np.shape(a) != np.shape(b) for a,b in zip(jax.tree.leaves(restored), jax.tree.leaves(template)))):
        raise ValueError('Continuation learner state shape mismatch')
    if not all(np.isfinite(a).all() for a in jax.tree.leaves(restored)):
        raise ValueError('Continuation learner state contains nonfinite values')
    if int(restored['step']) != metadata['step']:
        raise ValueError('Continuation checkpoint step metadata mismatch')
    return jax.tree.map(jnp.asarray, restored)
