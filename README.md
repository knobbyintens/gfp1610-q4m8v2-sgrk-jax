# SGRK-JAX

Unofficial JAX / Flax port of **Sparse-Gated Recurrent Kernels for Long-Horizon Sequence Modeling**
(Vasquez, Boateng, Takeda and Lindqvist, preprint ID **2601.01234**).

The port follows the reference formulation: the kernel sparsity gate selects at
most *s* kernels per timestep, and the horizon-decay schedule rescales retained
channels by a learned half-life. The original PyTorch reference is not used at
runtime; everything here is re-implemented in JAX.

## Contents

* `sgrk/layer.py` — Flax `SGRKLayer` module
* `sgrk/decay.py` — horizon-decay schedule
* `train.py` — toy associative-recall training entry point

## Notes

This repository is an independent reimplementation of the preprint above.
We are not the authors and this is not an official release.
