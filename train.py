"""Toy associative-recall training for the SGRK-JAX port."""

import jax
import jax.numpy as jnp

from sgrk.layer import SGRKLayer


def recall_task(batch=32, horizon=4096, channels=64):
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    layer = SGRKLayer(channels=channels, kernel_bank=8, sparsity=4)
    h = jax.random.normal(keys[0], (batch, channels, horizon))
    return layer.apply({"params": {}}, h)


if __name__ == "__main__":
    print(recall_task().shape)
