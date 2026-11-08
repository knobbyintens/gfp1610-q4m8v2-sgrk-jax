"""Flax implementation of the SGRK recurrence."""

import jax
import jax.numpy as jnp
import flax.linen as nn


class SGRKLayer(nn.Module):
    channels: int = 64
    kernel_bank: int = 8
    sparsity: int = 4

    @nn.compact
    def __call__(self, h, temperature=1.0):
        gate = nn.Dense(self.kernel_bank, name="sparsity_gate")(h.transpose(0, 2, 1))
        mask = nn.sigmoid(gate / temperature)
        hard = (mask > 0.5).astype(jnp.float32)
        mask = hard + mask - jax.lax.stop_gradient(mask)
        kernels = [nn.Conv(features=self.channels, kernel_size=(2 * i + 1,), name=f"k{i}")
                   for i in range(self.kernel_bank)]
        mixed = jnp.stack([k(h.transpose(0, 2, 1)) for k in kernels], axis=1)
        return (mask[..., None] * mixed).sum(axis=1)
