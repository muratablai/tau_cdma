"""
aliasing.py — Aliasing: When Channels Become Indistinguishable
===============================================================

Implements:
  - Template distance d²_jk (per-event, Poisson-weighted)
  - Aliasing sweep: track pairwise distances and eigenvalues vs binning M
  - Aliasing threshold matrix M*_jk
  - Aliasing order: which pairs merge first as M decreases
"""

import numpy as np
from .fisher import poisson_fim, eigenvalue_spectrum


def template_distance_per_event(A, theta):
    """Per-event Poisson-weighted template distances.

    d²_jk = Σ_i (a_ji - a_ki)² / p_i

    where p_i = [Aθ]_i is the per-event mixture probability.
    This is the single-event chi-squared distinguishability,
    independent of N. It measures pure resolution/geometry.

    Parameters
    ----------
    A : ndarray (M, K) — template matrix (columns sum to 1)
    theta : ndarray (K,) — branching ratios

    Returns
    -------
    D : ndarray (K, K) — symmetric distance matrix
    """
    K = A.shape[1]
    p = A @ theta  # per-event mixture probability per bin
    W = 1.0 / np.maximum(p, 1e-30)
    D = np.zeros((K, K))
    for j in range(K):
        for k in range(j + 1, K):
            diff = A[:, j] - A[:, k]
            D[j, k] = D[k, j] = np.sum(diff**2 * W)
    return D


def template_distance(A, lam):
    """Pairwise Poisson-weighted template distances (count-scaled).

    d²_jk = Σ_i (a_ji - a_ki)² / λ_i

    where λ_i = N·[Aθ]_i + b_i. This scales as 1/N.
    For aliasing analysis, prefer template_distance_per_event.

    Parameters
    ----------
    A : ndarray (M, K) — template matrix
    lam : ndarray (M,) — expected counts per bin

    Returns
    -------
    D : ndarray (K, K) — symmetric distance matrix
    """
    K = A.shape[1]
    W = 1.0 / np.maximum(lam, 1e-30)
    D = np.zeros((K, K))
    for j in range(K):
        for k in range(j + 1, K):
            diff = A[:, j] - A[:, k]
            D[j, k] = D[k, j] = np.sum(diff**2 * W)
    return D


def aliasing_sweep(template_builder, M_values, theta, N,
                    background_density=0.01):
    """Sweep binning resolution M, tracking eigenvalue spectrum and distances.

    Parameters
    ----------
    template_builder : TauTemplates instance (will be rebuilt at each M)
    M_values : list of int — binning resolutions to sweep
    theta : ndarray (K,) — branching ratios
    N : float — total events
    background_density : float — background rate per MeV

    Returns
    -------
    results : list of dicts with keys:
        'M', 'eigvals', 'eigvecs', 'distances', 'distances_per_event', 'F', 'A'
    """
    from tau_cdma.tau.templates import TauTemplates

    results = []
    m_range = template_builder.m_range
    range_width = m_range[1] - m_range[0]
    sigma_det = template_builder.sigma_det

    for M in M_values:
        tb = TauTemplates(M=M, m_range=m_range, sigma_det=sigma_det)
        A = tb.A
        dm = range_width / M
        b = background_density * dm * np.ones(M)
        lam = N * (A @ theta) + b
        F = poisson_fim(A, theta, N, background=b)
        eigvals, eigvecs = eigenvalue_spectrum(F)
        D = template_distance(A, lam)
        D_pe = template_distance_per_event(A, theta)

        results.append({
            'M': M,
            'eigvals': eigvals.copy(),
            'eigvecs': eigvecs.copy(),
            'distances': D.copy(),
            'distances_per_event': D_pe.copy(),
            'F': F.copy(),
            'A': A.copy(),
        })
    return results


def aliasing_threshold_matrix(sweep_results, d_crit=0.1):
    """Identify aliasing thresholds M*_jk for each channel pair.

    M*_jk = min{M : d²_jk(M) ≥ d²_crit}

    Uses per-event distances (geometry only, independent of N).

    Parameters
    ----------
    sweep_results : output of aliasing_sweep
    d_crit : float — critical per-event distance threshold.
        d_crit = 0.1 means templates must differ by at least 10%
        in their chi-squared profile to be considered distinguishable.

    Returns
    -------
    M_star : ndarray (K, K) — aliasing threshold matrix
        M_star[j,k] = minimum M to distinguish channels j and k
        M_star[j,k] = inf if they're always aliased in the sweep range
    """
    K = sweep_results[0]['distances_per_event'].shape[0]
    M_star = np.full((K, K), np.inf)

    # Sort by M ascending (coarsest first)
    sorted_results = sorted(sweep_results, key=lambda r: r['M'])

    for j in range(K):
        for k in range(j + 1, K):
            for r in sorted_results:
                if r['distances_per_event'][j, k] >= d_crit:
                    M_star[j, k] = M_star[k, j] = r['M']
                    break
    return M_star


def aliasing_order(sweep_results):
    """Determine the aliasing order: which pairs merge first as M decreases.

    At each M, rank pairs by per-event distance (smallest = most aliased).
    Returns the ordering from most-easily-aliased to most-distinct.

    Parameters
    ----------
    sweep_results : output of aliasing_sweep

    Returns
    -------
    order : list of (j, k, d²_min) tuples, sorted by d² at coarsest binning
    distances_vs_M : dict mapping (j,k) → list of (M, d²) pairs
    """
    K = sweep_results[0]['distances_per_event'].shape[0]

    # Use the coarsest binning to establish the order
    coarsest = min(sweep_results, key=lambda r: r['M'])
    D = coarsest['distances_per_event']

    pairs = []
    for j in range(K):
        for k in range(j + 1, K):
            pairs.append((j, k, D[j, k]))

    # Sort by distance (smallest first = aliases first)
    pairs.sort(key=lambda x: x[2])

    # Also collect full distance vs M for each pair
    sorted_results = sorted(sweep_results, key=lambda r: r['M'])
    distances_vs_M = {}
    for j in range(K):
        for k in range(j + 1, K):
            distances_vs_M[(j, k)] = [
                (r['M'], r['distances_per_event'][j, k]) for r in sorted_results
            ]

    return pairs, distances_vs_M


def eigenvalue_collapse_diagnostic(sweep_results, threshold_fraction=0.01):
    """Identify which eigenvalue directions collapse at each M.

    Returns
    -------
    diagnostics : list of dicts with 'M', 'collapsed_indices',
                  'collapsed_eigvecs', 'min_eigval_ratio'
    """
    diagnostics = []
    for r in sweep_results:
        eigvals = r['eigvals']
        eigvecs = r['eigvecs']
        max_eig = eigvals[-1] if eigvals[-1] > 0 else 1.0
        ratios = eigvals / max_eig

        collapsed = np.where(ratios < threshold_fraction)[0]
        diagnostics.append({
            'M': r['M'],
            'collapsed_indices': collapsed,
            'collapsed_eigvecs': eigvecs[:, collapsed] if len(collapsed) > 0 else None,
            'min_eigval_ratio': ratios[0] if len(ratios) > 0 else 0.0,
            'eigval_ratios': ratios.copy(),
        })
    return diagnostics
