"""Spatial relation encoder followed by the final temporal LSTM.

Attention mixes roles within the CURRENT observation, not future time steps.
Temporal memory is exclusively (cell, hidden), carried by the recurrent PPO
adapter and reset per episode. No privileged features or stage labels enter.
"""
from flax import linen as nn
import jax.numpy as jp


# Exact partition of SHOW-103 actor49: proprio25, cube pose6, bowl3,
# cube-minus-TCP3, cube-minus-bowl3, half-size1, gripper goal1, previous action7.
TOKEN_GROUPS = {
    'robot': tuple(range(25)),
    'cube': tuple(range(25, 31)) + (40,),
    'bowl': tuple(range(31, 34)) + tuple(range(37, 40)),
    'interaction': tuple(range(34, 37)) + tuple(range(41, 49)),
}


class RelationBlock(nn.Module):
    token_width: int = 64
    heads: int = 4
    feedforward_width: int = 128

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm(name='attention_norm')(x)
        y = nn.MultiHeadDotProductAttention(num_heads=self.heads,
            qkv_features=self.token_width, out_features=self.token_width,
            dropout_rate=0., deterministic=True, name='attention')(y)
        x = x+y
        y = nn.LayerNorm(name='feedforward_norm')(x)
        gate = nn.silu(nn.Dense(self.feedforward_width, name='gate')(y))
        value = nn.Dense(self.feedforward_width, name='value')(y)
        return x+nn.Dense(self.token_width, name='output')(gate*value)


class TransformerLSTMActor(nn.Module):
    action_size: int
    width: int = 128
    token_width: int = 64
    heads: int = 4
    layers: int = 2
    feedforward_width: int = 128

    @nn.compact
    def __call__(self, carry, x):
        if x.shape[-1] != 49:
            raise ValueError('relation tokens require the exact SHOW-103 actor49 layout')
        values = [nn.Dense(self.token_width, name='token_'+name)(x[..., jp.array(indices)])
                  for name, indices in TOKEN_GROUPS.items()]
        tokens = jp.stack(values, axis=-2)
        summary = self.param('summary_token', nn.initializers.normal(.02), (1, self.token_width))
        summary = jp.broadcast_to(summary, x.shape[:-1]+(1, self.token_width))
        tokens = jp.concatenate([summary, tokens], axis=-2)
        roles = self.param('role_embeddings', nn.initializers.normal(.02), (5, self.token_width))
        tokens = tokens+roles
        for i in range(self.layers):
            tokens = RelationBlock(self.token_width, self.heads, self.feedforward_width,
                                   name=f'relation_{i}')(tokens)
        features = nn.LayerNorm(name='relation_norm')(tokens[..., 0, :])
        # The LSTM is the LAST nonlinear processing layer. Only the affine
        # distribution head follows it; no hidden MLP follows memory.
        carry, hidden = nn.OptimizedLSTMCell(self.width, name='lstm')(carry, features)
        return carry, nn.Dense(2*self.action_size, name='distribution')(hidden)
