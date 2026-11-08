"""Horizon-decay schedule for the JAX port."""

import jax.numpy as jnp


def horizon_decay(h, steps, half_life):
    factor = jnp.power(0.5, steps / half_life)
    return h * factor
