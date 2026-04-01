"""
simulate.py — Poisson Pseudo-data Generation
==============================================

Generate observed histograms from the Poisson MAC:
  y_i ~ Poisson(N · [Aθ]_i + b_i)
"""

import numpy as np


def generate_poisson_data(A, theta, N, background=None, rng=None):
    """Generate a single Poisson pseudo-experiment.

    Parameters
    ----------
    A : ndarray (M, K) — template matrix
    theta : ndarray (K,) — branching ratios
    N : float — total events
    background : ndarray (M,) or None — background rates
    rng : numpy random generator

    Returns
    -------
    y : ndarray (M,) — observed bin counts
    lam : ndarray (M,) — true expected counts (for reference)
    """
    if rng is None:
        rng = np.random.default_rng()
    if background is None:
        background = np.zeros(A.shape[0])

    lam = N * (A @ theta) + background
    lam = np.maximum(lam, 0.0)
    y = rng.poisson(lam)
    return y, lam


def generate_multi_experiment(A, theta, N, background=None,
                               n_experiments=100, rng=None):
    """Generate multiple pseudo-experiments.

    Parameters
    ----------
    A, theta, N, background — as above
    n_experiments : int — number of experiments

    Returns
    -------
    Y : ndarray (n_experiments, M) — matrix of observed histograms
    lam : ndarray (M,) — true expected counts
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if background is None:
        background = np.zeros(A.shape[0])

    M = A.shape[0]
    lam = N * (A @ theta) + background
    lam = np.maximum(lam, 0.0)

    Y = rng.poisson(np.tile(lam, (n_experiments, 1)))
    return Y, lam
