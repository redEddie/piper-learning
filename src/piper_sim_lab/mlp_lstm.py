"""Flat observation encoder followed by the final temporal LSTM.

The encoder outputs64 features, matching the Transformer encoder interface.
Only an affine distribution head follows memory. Legacy LSTM actors and their
checkpoint layout remain separate under actor kind ``lstm``.
"""
from flax import linen as nn


class MLPLSTMActor(nn.Module):
    action_size: int
    width: int = 128
    hidden_sizes: tuple = (256, 64)

    @nn.compact
    def __call__(self, carry, x):
        for i, size in enumerate(self.hidden_sizes):
            x = nn.silu(nn.Dense(size, name=f'encoder_{i}')(x))
        features = nn.LayerNorm(name='encoder_norm')(x)
        carry, hidden = nn.OptimizedLSTMCell(self.width, name='lstm')(carry, features)
        return carry, nn.Dense(2*self.action_size, name='distribution')(hidden)
