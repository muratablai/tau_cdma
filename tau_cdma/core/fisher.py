"""
fisher.py — Fisher Information for the Poisson MAC
====================================================

Implements:
  - Poisson FIM: F_jk = N² Σ_i a_ji a_ki / λ_i
  - Cramér-Rao bound (unconstrained and reduced/simplex)
  - Eigenvalue spectrum analysis
  - Information loss from erasure

Scaling note:
  The N² prefactor is cancelled by the N-dependence of λ_i in the
  signal-dominated regime (λ_i ≈ N·(Aθ)_i), giving effective F ∝ N.
  In background-dominated bins (λ_i ≈ b_i), F ∝ N² for those bins.
  For PID, where signal templates dominate the expected counts in the
  relevant bins, the effective scaling is linear in N.
"""

import numpy as np
from numpy.linalg import inv, eigh, cond


def poisson_fim(A, theta, N, background=None):
    """Compute Fisher Information Matrix for the Poisson MAC.

    F_jk = N² Σ_i a_ji a_ki / λ_i

    Parameters
    ----------
    A : ndarray (M, K) — template matrix
    theta : ndarray (K,) — branching ratios
    N : float — total number of events
    background : ndarray (M,) or None — background rates per bin

    Returns
    -------
    F : ndarray (K, K) — Fisher information matrix
    """
    M, K = A.shape
    if background is None:
        background = np.zeros(M)

    lam = N * (A @ theta) + background
    # Avoid division by zero
    lam_safe = np.maximum(lam, 1e-30)
    W = 1.0 / lam_safe

    F = N**2 * (A.T * W) @ A
    return F


def crb(F, regularize=False):
    """Cramér-Rao bound: diagonal of F⁻¹.

    For near-singular FIM (e.g. under erasure), uses eigendecomposition
    and sets CRB = inf for directions with insufficient Fisher information.

    Parameters
    ----------
    F : ndarray (K, K) — Fisher information matrix
    regularize : bool — add small ridge for numerical stability

    Returns
    -------
    bounds : ndarray (K,) — lower bounds on variance of each BR estimate.
        Returns inf for channels with effectively zero Fisher information.
    """
    K = F.shape[0]
    if regularize:
        F = F + 1e-12 * np.eye(K)

    # Eigendecomposition for numerical stability
    eigvals, eigvecs = np.linalg.eigh(F)

    # Threshold: eigenvalues below this are treated as zero
    # (machine precision relative to largest eigenvalue)
    max_eig = np.max(np.abs(eigvals))
    if max_eig == 0:
        return np.full(K, np.inf)
    threshold = max_eig * K * np.finfo(float).eps * 100  # conservative

    # Pseudoinverse via eigendecomposition
    eigvals_inv = np.zeros_like(eigvals)
    for i in range(len(eigvals)):
        if eigvals[i] > threshold:
            eigvals_inv[i] = 1.0 / eigvals[i]
        # else leave as 0 (direction has no info)

    F_pinv = (eigvecs * eigvals_inv) @ eigvecs.T
    bounds = np.diag(F_pinv).copy()  # .copy() to ensure writable

    # Channels whose CRB came from zero-info directions get inf
    # Detection: if any eigenvalue was zeroed, check if channel is affected
    zero_dirs = eigvals <= threshold
    if np.any(zero_dirs):
        # A channel k is unresolvable if it has significant weight
        # along a zero-eigenvalue direction
        for k in range(K):
            projection = np.sum(eigvecs[k, zero_dirs]**2)
            if projection > 0.01:  # channel has >1% weight in null space
                bounds[k] = np.inf

    # Safety: CRB variance must be non-negative
    bounds = np.where(bounds < 0, np.inf, bounds)

    return bounds


def reduced_fim(A, theta, N, background=None):
    """Compute FIM in the reduced (K-1)-dimensional parameterization.

    Since θ ∈ Δ^{K-1} (simplex), we drop the last component:
    θ_K = 1 - Σ_{k<K} θ_k.

    This gives the proper CRB for simplex-constrained estimation
    (Gorman & Hero 1990).

    Parameters
    ----------
    A : ndarray (M, K) — template matrix
    theta : ndarray (K,) — species fractions on the simplex
    N : float — total event count
    background : ndarray (M,) or None — background rates per bin

    Returns
    -------
    F_red : ndarray (K-1, K-1) — reduced Fisher information matrix
    """
    M, K = A.shape
    if background is None:
        background = np.zeros(M)

    lam = N * (A @ theta) + background
    lam_safe = np.maximum(lam, 1e-30)
    W = 1.0 / lam_safe

    # Reduced templates: ã_k = a_k - a_K for k = 0,...,K-2
    A_tilde = A[:, :-1] - A[:, -1:] # (M, K-1)
    F_red = N**2 * (A_tilde.T * W) @ A_tilde
    return F_red


def eigenvalue_spectrum(F):
    """Eigenvalues and eigenvectors of FIM, sorted ascending.

    Returns
    -------
    eigvals : ndarray — eigenvalues (ascending)
    eigvecs : ndarray — corresponding eigenvectors (columns)
    """
    eigvals, eigvecs = eigh(F)
    return eigvals, eigvecs


def condition_number(F):
    """Condition number κ(F) = λ_max / λ_min."""
    return cond(F)


def information_loss(F_full, F_partial):
    """Information loss matrix ΔF = F_full - F_partial.

    Parameters
    ----------
    F_full : ndarray (K, K) — Fisher information matrix with full observables
    F_partial : ndarray (K, K) — Fisher information matrix with partial access

    Returns
    -------
    delta : ndarray (K, K) — information loss matrix
    trace_loss : float — total scalar information lost (trace of ΔF)
    """
    delta = F_full - F_partial
    return delta, np.trace(delta)


def crb_relative_uncertainty(F, theta):
    """Relative uncertainty σ(BR_k)/BR_k from CRB.

    Parameters
    ----------
    F : ndarray (K, K) — FIM
    theta : ndarray (K,) — true branching ratios

    Returns
    -------
    rel_unc : ndarray (K,) — relative uncertainties
    """
    bounds = crb(F, regularize=True)
    return np.sqrt(np.maximum(bounds, 0.0)) / np.maximum(theta, 1e-30)


def constrained_crb(F, C=None, method='auto'):
    """Constrained Cramér-Rao bound (Gorman-Hero 1990 / Stoica-Ng 1998).

    When parameters are subject to equality constraints Cθ = d,
    the constrained CRB is tighter than the unconstrained CRB.

    Gorman-Hero (F positive definite):
      CRB_c = F⁻¹ − F⁻¹Cᵀ(CF⁻¹Cᵀ)⁻¹CF⁻¹

    Stoica-Ng (F singular):
      U = null(C), CRB_c = U(UᵀFU)⁻¹Uᵀ

    Parameters
    ----------
    F : ndarray (K, K) — Fisher information matrix
    C : ndarray (r, K) or None — constraint Jacobian (default: sum-to-one)
    method : str — 'gorman-hero', 'stoica-ng', or 'auto'

    Returns
    -------
    CRB_c : ndarray (K, K) — constrained CRB matrix (PSD)
    """
    from scipy.linalg import svd as scipy_svd

    K = F.shape[0]
    if C is None:
        C = np.ones((1, K))  # sum-to-one: [1, 1, ..., 1]

    eps = 1e-10

    # Check if F is positive definite
    eigvals_F = np.linalg.eigvalsh(F)
    is_pd = np.min(eigvals_F) > eps * np.max(np.abs(eigvals_F))

    if method == 'auto':
        method = 'gorman-hero' if is_pd else 'stoica-ng'

    if method == 'gorman-hero' and is_pd:
        Finv = np.linalg.inv(F)
        FinvCT = Finv @ C.T
        CFinvCT = C @ FinvCT
        correction = FinvCT @ np.linalg.inv(CFinvCT) @ FinvCT.T
        CRB_c = Finv - correction
    else:
        # Stoica-Ng: project into null space of C
        _, S, Vt = scipy_svd(C, full_matrices=True)
        r = np.sum(S > eps)
        U = Vt[r:].T  # (K, K-r)

        UtFU = U.T @ F @ U
        # Check if regularization would dominate smallest eigenvalue
        eigs_UtFU = np.linalg.eigvalsh(UtFU)
        min_eig = np.min(np.abs(eigs_UtFU))
        if min_eig > 0 and eps / min_eig > 0.01:
            import warnings
            warnings.warn(
                f"constrained_crb: regularization eps={eps:.1e} is {eps/min_eig:.1%} "
                f"of smallest eigenvalue {min_eig:.1e}. CRB may mask true rank deficiency.",
                stacklevel=2
            )
        UtFU += eps * np.eye(UtFU.shape[0])  # regularize
        CRB_c = U @ np.linalg.inv(UtFU) @ U.T

    # Ensure PSD (clip numerical noise)
    eigvals_c, eigvecs_c = np.linalg.eigh(CRB_c)
    eigvals_c = np.maximum(eigvals_c, 0.0)
    CRB_c = (eigvecs_c * eigvals_c) @ eigvecs_c.T

    return CRB_c


def crb_multiuser_efficiency(F, CRB_c=None):
    """CRB-based multiuser efficiency.

    η_k = 1 / ([F⁻¹]_kk · F_kk)

    This is algebraically equivalent to η_k = 1 / [R⁻¹]_kk from
    interference.py when R is invertible, since [R⁻¹]_kk = F_kk · [F⁻¹]_kk.
    Both compute the decorrelator asymptotic multiuser efficiency (AME).

    Parameters
    ----------
    F : ndarray (K, K) — Fisher information matrix
    CRB_c : ndarray (K, K) or None — constrained CRB (for η_c)

    Returns
    -------
    eta : ndarray (K,) — unconstrained CRB-based efficiency
    eta_c : ndarray (K,) or None — constrained efficiency
    """
    K = F.shape[0]
    F_diag = np.diag(F)

    try:
        Finv = np.linalg.inv(F + 1e-15 * np.eye(K))
        Finv_diag = np.diag(Finv)
    except np.linalg.LinAlgError:
        Finv_diag = np.full(K, np.inf)

    eta = np.where(
        (Finv_diag > 0) & (F_diag > 0),
        1.0 / (Finv_diag * F_diag),
        0.0
    )
    eta = np.clip(eta, 0.0, 1.0)

    eta_c = None
    if CRB_c is not None:
        CRBc_diag = np.diag(CRB_c)
        eta_c = np.where(
            (CRBc_diag > 0) & (F_diag > 0),
            1.0 / (CRBc_diag * F_diag),
            0.0
        )
        eta_c = np.clip(eta_c, 0.0, 1.0)

    return eta, eta_c
