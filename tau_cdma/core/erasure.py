"""
erasure.py — Detector Acceptance as an Erasure Channel
=======================================================

Implements:
  - Random (Bernoulli) erasure masks
  - Geometric (structured) erasure masks for τ detector
  - FIM under erasure
  - Erasure sweep: CRB vs access fraction α
"""

import numpy as np
from .fisher import poisson_fim, crb


def random_erasure_masks(M, alpha, n_trials=100, rng=None):
    """Generate random Bernoulli erasure masks.

    Parameters
    ----------
    M : int — number of bins
    alpha : float — access fraction (probability each bin is kept)
    n_trials : int — number of random masks to generate
    rng : numpy random generator or None

    Returns
    -------
    masks : ndarray (n_trials, M) — binary masks
    """
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.binomial(1, alpha, size=(n_trials, M)).astype(float)


def geometric_erasure_mask(m_bins, m_tau=1776.93, m_pi=139.57,
                            low_cut=None, high_cut=None,
                            gap_center=None, gap_width=None):
    """Generate geometric erasure mask based on detector acceptance.

    Default model for τ visible mass:
      - low_cut: m_vis < ~m_π erased (below pion mass threshold)
      - high_cut: m_vis > m_τ erased (kinematic endpoint)
      - Optional gap: central-mass region with poor resolution

    Parameters
    ----------
    m_bins : ndarray — bin centers in MeV
    m_tau : float — τ mass (kinematic endpoint)
    m_pi : float — pion mass (low threshold)
    low_cut : float or None — low mass cutoff (default: 0.8 * m_pi)
    high_cut : float or None — high mass cutoff (default: m_tau)
    gap_center : float or None — center of a detector gap
    gap_width : float or None — width of detector gap

    Returns
    -------
    mask : ndarray (M,) — binary mask
    alpha : float — resulting access fraction
    """
    if low_cut is None:
        low_cut = 0.8 * m_pi
    if high_cut is None:
        high_cut = m_tau

    mask = np.ones(len(m_bins))
    mask[m_bins < low_cut] = 0.0
    mask[m_bins > high_cut] = 0.0

    if gap_center is not None and gap_width is not None:
        mask[np.abs(m_bins - gap_center) < gap_width / 2] = 0.0

    alpha = np.mean(mask)
    return mask, alpha


def fim_under_erasure(A, theta, N, background, mask):
    """Compute FIM with erasure mask applied.

    F_R = N² Ã^T W̃ Ã  where Ã = M_R · A, W̃ = diag(1/λ̃_i) on kept bins.

    Parameters
    ----------
    A : ndarray (M, K) — full template matrix
    theta : ndarray (K,) — branching ratios
    N : float — total events
    background : ndarray (M,) — background per bin
    mask : ndarray (M,) — binary erasure mask

    Returns
    -------
    F_R : ndarray (K, K) — Fisher information under erasure
    """
    A_eff = A * mask[:, None]
    b_eff = background * mask
    lam = N * (A_eff @ theta) + b_eff
    lam_safe = np.where(mask > 0, np.maximum(lam, 1e-30), 1.0)
    W = np.where(mask > 0, 1.0 / lam_safe, 0.0)
    F_R = N**2 * (A_eff.T * W) @ A_eff
    return F_R


def erasure_sweep(A, theta, N, background, alpha_values,
                  n_trials=50, mode='random', m_bins=None, rng=None):
    """Sweep access fraction α, computing CRB at each value.

    Parameters
    ----------
    A, theta, N, background — model parameters
    alpha_values : array — access fractions to test
    n_trials : int — number of random trials per α (for random mode)
    mode : 'random' or 'geometric'
    m_bins : ndarray — bin centers (required for geometric mode)
    rng : numpy RNG

    Returns
    -------
    results : dict with keys:
        'alpha' : array of α values
        'crb_mean' : ndarray (len(alpha), K) — mean CRB at each α
        'crb_std' : ndarray (len(alpha), K) — std of CRB (random only)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    M, K = A.shape
    crb_all = np.zeros((len(alpha_values), K))
    crb_std = np.zeros((len(alpha_values), K))

    for i, alpha in enumerate(alpha_values):
        if mode == 'random':
            crbs = []
            for _ in range(n_trials):
                mask = rng.binomial(1, alpha, M).astype(float)
                if np.sum(mask) < K + 1:
                    continue  # too few bins
                F_R = fim_under_erasure(A, theta, N, background, mask)
                try:
                    c = crb(F_R, regularize=True)
                    crbs.append(c)
                except:
                    pass
            if crbs:
                crbs = np.array(crbs)
                crb_all[i] = np.median(crbs, axis=0)
                crb_std[i] = np.std(crbs, axis=0)
            else:
                crb_all[i] = np.inf
                crb_std[i] = np.inf

        elif mode == 'geometric':
            if m_bins is None:
                raise ValueError("m_bins required for geometric erasure")
            # Use only the first M bins (m_bins might be longer than M)
            mb = m_bins[:M] if len(m_bins) >= M else m_bins
            m_range_width = mb[-1] - mb[0]
            erased = 1.0 - alpha
            low_cut = mb[0] + erased * 0.4 * m_range_width
            high_cut = mb[-1] - erased * 0.6 * m_range_width
            mask = np.ones(M)
            mask[mb < low_cut] = 0.0
            mask[mb > high_cut] = 0.0
            actual_alpha = np.mean(mask)

            if np.sum(mask) < K + 1:
                crb_all[i] = np.inf
            else:
                F_R = fim_under_erasure(A, theta, N, background, mask)
                crb_all[i] = crb(F_R, regularize=True)

    return {
        'alpha': np.array(alpha_values),
        'crb_mean': crb_all,
        'crb_std': crb_std,
    }
