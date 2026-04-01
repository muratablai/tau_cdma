"""
interference.py — Interference Matrix and Multiuser Efficiency
===============================================================

Implements:
  - R matrix: normalized Gram matrix under Poisson-weighted inner product
  - Multiuser efficiency η_k = 1/[R⁻¹]_kk  (Verdú 1998)
  - Interference-to-noise ratio INR_{j→k}
"""

import numpy as np
from numpy.linalg import inv


def interference_matrix(A, theta, N, background=None):
    """Normalized cross-correlation matrix R (the interference matrix).

    R_jk = <a_j, a_k>_W / (||a_j||_W · ||a_k||_W)

    where the inner product is Poisson-weighted: <u,v>_W = Σ_i u_i v_i / λ_i

    Parameters
    ----------
    A : ndarray (M, K) — template matrix
    theta : ndarray (K,) — branching ratios
    N : float — total events
    background : ndarray (M,) or None

    Returns
    -------
    R : ndarray (K, K) — interference matrix (diagonal = 1)
    """
    M, K = A.shape
    if background is None:
        background = np.zeros(M)

    lam = N * (A @ theta) + background
    W = 1.0 / np.maximum(lam, 1e-30)

    # Weighted Gram matrix
    AW = A.T * W  # (K, M)
    G = AW @ A    # (K, K) — unnormalized

    # Norms
    norms = np.sqrt(np.diag(G))
    norms = np.maximum(norms, 1e-30)

    # Normalize
    R = G / np.outer(norms, norms)
    # Force exact diagonal = 1
    np.fill_diagonal(R, 1.0)
    return R


def multiuser_efficiency(R):
    """Multiuser efficiency η_k = 1 / [R⁻¹]_kk.

    This measures how much the presence of other users (channels) degrades
    the estimation of channel k. η_k = 1 means no interference;
    η_k → 0 means channel k is overwhelmed.

    Parameters
    ----------
    R : ndarray (K, K) — interference matrix

    Returns
    -------
    eta : ndarray (K,) — multiuser efficiencies
    """
    try:
        R_inv = inv(R)
        eta = 1.0 / np.diag(R_inv)
    except np.linalg.LinAlgError:
        eta = np.zeros(R.shape[0])
    return eta


def inr_matrix(A, theta, N, background=None):
    """Interference-to-noise ratio matrix.

    INR_{j→k} = (BR_j / BR_k) · (||a_j||²_W / ||a_k||²_W) · R²_jk

    Decomposes into: power ratio × norm ratio × squared cross-correlation,
    following the standard CDMA convention (Verdú 1998, Ch. 3).

    Parameters
    ----------
    A, theta, N, background — as in interference_matrix

    Returns
    -------
    INR : ndarray (K, K) — INR[j,k] = interference from j into k
    """
    M, K = A.shape
    if background is None:
        background = np.zeros(M)

    lam = N * (A @ theta) + background
    W = 1.0 / np.maximum(lam, 1e-30)

    AW = A.T * W
    G = AW @ A
    norms_sq = np.diag(G)

    # Normalized cross-correlation
    norms = np.sqrt(np.maximum(norms_sq, 1e-30))
    R = G / np.outer(norms, norms)

    INR = np.zeros((K, K))
    for j in range(K):
        for k in range(K):
            if j != k and norms_sq[k] > 0:
                power_ratio = theta[j] / max(theta[k], 1e-30)
                norm_ratio = norms_sq[j] / max(norms_sq[k], 1e-30)
                INR[j, k] = power_ratio * norm_ratio * R[j, k]**2
    return INR
