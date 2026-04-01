"""
nmf.py — Blind Code Discovery via Poisson NMF
================================================

Implements:
  - Poisson NMF (KL divergence) via sklearn
  - Model selection (BIC) for number of channels K
  - Template recovery error metrics
"""

import numpy as np
from sklearn.decomposition import NMF


def poisson_nmf(Y, K, n_iter=2000, init='nndsvda', random_state=42):
    """Non-negative Matrix Factorization under Poisson (KL divergence).

    min_{A≥0, θ≥0} D_KL(Y || A·θ)

    Parameters
    ----------
    Y : ndarray (n_samples, M) — observed histograms (each row = one experiment)
        If 1D array of length M, treated as single sample.
    K : int — number of components
    n_iter : int — max iterations
    init : str — initialization method
    random_state : int

    Returns
    -------
    A_hat : ndarray (M, K) — recovered template matrix
    theta_hat : ndarray (n_samples, K) — recovered mixing coefficients
    reconstruction_err : float — final KL divergence
    """
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    # Ensure non-negative
    Y = np.maximum(Y, 0.0)

    # Choose init based on dimensions
    n_samples, M_dim = Y.shape
    if K > min(n_samples, M_dim):
        init = 'random'

    model = NMF(
        n_components=K,
        solver='mu',
        beta_loss='kullback-leibler',
        init=init,
        max_iter=n_iter,
        random_state=random_state,
        tol=1e-6,
    )

    theta_hat = model.fit_transform(Y)
    A_hat = model.components_.T  # (M, K) in our convention

    # Normalize columns of A_hat to sum to 1
    col_sums = np.sum(A_hat, axis=0, keepdims=True)
    col_sums = np.maximum(col_sums, 1e-30)
    A_hat = A_hat / col_sums
    # Compensate in theta
    theta_hat = theta_hat * col_sums.ravel()

    reconstruction_err = model.reconstruction_err_

    return A_hat, theta_hat, reconstruction_err


def nmf_model_selection(Y, K_range=range(2, 12), n_iter=1000):
    """Select optimal K using BIC-like criterion.

    BIC = -2·log L + k·log(n)
    For Poisson NMF: log L ≈ -D_KL(Y || Ŷ) (up to constants)

    Parameters
    ----------
    Y : ndarray (n_samples, M) — data
    K_range : iterable of int — K values to test

    Returns
    -------
    results : dict with:
        'K_values' : list of K tested
        'bic' : list of BIC scores
        'errors' : list of reconstruction errors
        'K_best' : K with lowest BIC
    """
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    n_samples, M = Y.shape
    n = n_samples * M

    bics = []
    errors = []

    for K in K_range:
        try:
            _, _, err = poisson_nmf(Y, K, n_iter=n_iter)
            # Number of free parameters: K*M + n_samples*K
            n_params = K * M + n_samples * K
            bic = 2 * err * n + n_params * np.log(n)
            bics.append(bic)
            errors.append(err)
        except Exception:
            bics.append(np.inf)
            errors.append(np.inf)

    K_values = list(K_range)
    K_best = K_values[np.argmin(bics)]

    return {
        'K_values': K_values,
        'bic': bics,
        'errors': errors,
        'K_best': K_best,
    }


def template_recovery_error(A_true, A_hat):
    """Measure how well recovered templates match true templates.

    Uses optimal permutation matching (Hungarian algorithm) to align
    recovered components with true templates.

    Parameters
    ----------
    A_true : ndarray (M, K) — true template matrix
    A_hat : ndarray (M, K_hat) — recovered template matrix

    Returns
    -------
    errors : dict with:
        'per_channel' : ndarray — ||a_k - â_σ(k)|| / ||a_k|| for best match
        'mean' : float — mean relative error
        'matching' : list of tuples — (true_idx, recovered_idx) pairs
    """
    K = A_true.shape[1]
    K_hat = A_hat.shape[1]
    K_match = min(K, K_hat)

    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    # Normalize columns
    A_true_n = A_true / np.maximum(np.linalg.norm(A_true, axis=0, keepdims=True), 1e-30)
    A_hat_n = A_hat / np.maximum(np.linalg.norm(A_hat, axis=0, keepdims=True), 1e-30)

    # Cost matrix: 1 - cosine similarity
    cost = 1.0 - (A_true_n.T @ A_hat_n)  # (K, K_hat)

    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost[:K_match, :K_hat])
    except ImportError:
        # Greedy matching fallback
        row_ind, col_ind = [], []
        used = set()
        for j in range(K_match):
            best_k = -1
            best_d = np.inf
            for k in range(K_hat):
                if k not in used and cost[j, k] < best_d:
                    best_d = cost[j, k]
                    best_k = k
            if best_k >= 0:
                row_ind.append(j)
                col_ind.append(best_k)
                used.add(best_k)

    # Compute relative errors for matched pairs
    per_channel = np.full(K, np.inf)
    matching = []
    for j, k in zip(row_ind, col_ind):
        norm_true = np.linalg.norm(A_true[:, j])
        if norm_true > 0:
            per_channel[j] = np.linalg.norm(A_true[:, j] - A_hat[:, k]) / norm_true
        matching.append((j, k))

    return {
        'per_channel': per_channel,
        'mean': np.mean(per_channel[per_channel < np.inf]),
        'matching': matching,
    }
