"""
gamp.py — Generalized Approximate Message Passing for Poisson Template Mixtures
================================================================================

Implements GAMP (Rangan 2011) specialized to the Poisson output channel:
    y_i ~ Poisson(λ_i),  λ_i = N * (Aθ)_i + b_i

The algorithm iteratively estimates θ from observed counts y by passing
messages between a "prior" (simplex constraint) and a "channel" (Poisson
likelihood) through the linear mixing matrix A.

References:
    - Rangan (2011). Generalized AMP. IEEE Trans. IT.
    - Schniter, Rangan, Fletcher (2016). GLM-VAMP. arXiv:1612.01186.

Status: Algorithmic baseline for structured template inference.
"""

import numpy as np
from scipy.special import digamma


def _poisson_channel_estimation(y, lam_hat, tau_p):
    """Poisson channel output estimation function.
    
    Computes the posterior mean and variance of z_i given y_i ~ Pois(z_i)
    and prior z_i ~ N(lam_hat_i, tau_p_i).
    
    Uses Gaussian approximation to the Poisson posterior for tractability.
    
    Parameters
    ----------
    y : ndarray (M,) — observed counts
    lam_hat : ndarray (M,) — prior mean for rates
    tau_p : ndarray (M,) — prior variance for rates
    
    Returns
    -------
    z_hat : ndarray (M,) — posterior mean of rates
    tau_z : ndarray (M,) — posterior variance of rates
    """
    # Poisson Fisher information at the prior mean: 1/lambda
    lam_safe = np.maximum(lam_hat, 1e-10)
    
    # Posterior precision = prior precision + channel Fisher
    tau_z = 1.0 / (1.0 / np.maximum(tau_p, 1e-30) + 1.0 / lam_safe)
    
    # Posterior mean: Newton step from prior mean toward MLE
    # MLE of Poisson rate is y_i; posterior mean is a precision-weighted average
    z_hat = tau_z * (lam_hat / np.maximum(tau_p, 1e-30) + y / lam_safe)
    
    # Enforce positivity (rates must be positive)
    z_hat = np.maximum(z_hat, 1e-10)
    
    return z_hat, tau_z


def _simplex_projection(v):
    """Project v onto the probability simplex Δ^{K-1}.
    
    Implements the efficient O(K log K) algorithm of Duchi et al. (2008).
    """
    K = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    cond = u > cssv / np.arange(1, K + 1)
    if not np.any(cond):
        return np.ones(K) / K
    rho = np.max(np.where(cond))
    tau = cssv[rho] / (rho + 1.0)
    return np.maximum(v - tau, 0.0)


def _simplex_denoiser(r, tau_r):
    """Denoise θ under simplex prior using proximal projection.
    
    Given noisy observation r = θ + N(0, tau_r I), compute the
    posterior mean of θ ∈ Δ^{K-1}.
    
    For simplicity, uses the proximal (MAP) estimator:
        θ_hat = argmin_{θ ∈ Δ} ||θ - r||² / (2 tau_r)
    which is the simplex projection of r.
    
    Parameters
    ----------
    r : ndarray (K,) — effective observation
    tau_r : float — effective noise variance
    
    Returns
    -------
    theta_hat : ndarray (K,) — denoised estimate on simplex
    tau_theta : float — effective posterior variance (scalar approx)
    """
    theta_hat = _simplex_projection(r)
    
    # Approximate posterior variance: count active (non-boundary) components
    active = theta_hat > 1e-10
    n_active = max(np.sum(active), 1)
    # Variance reduction from projection: roughly tau_r * (1 - 1/K)
    tau_theta = tau_r * (n_active - 1) / n_active
    
    return theta_hat, tau_theta


def gamp_poisson(A, y, N, background=None, theta_init=None,
                 max_iter=200, tol=1e-6, damping=0.5, verbose=False):
    """GAMP for Poisson template mixture estimation.
    
    Estimates θ ∈ Δ^{K-1} from y_i ~ Poisson(N*(Aθ)_i + b_i).
    
    Parameters
    ----------
    A : ndarray (M, K) — template matrix (columns sum to 1)
    y : ndarray (M,) — observed bin counts
    N : float — total event count
    background : ndarray (M,) or None — background rates
    theta_init : ndarray (K,) or None — initial estimate
    max_iter : int — maximum iterations
    tol : float — convergence tolerance on θ
    damping : float — damping factor ∈ (0, 1] for stability
    verbose : bool — print convergence info
    
    Returns
    -------
    result : dict with keys:
        'theta' : ndarray (K,) — estimated branching ratios
        'converged' : bool
        'iterations' : int
        'mse_history' : list — ||Δθ|| per iteration
    """
    M, K = A.shape
    if background is None:
        background = np.zeros(M)
    
    # Scale templates by N
    NA = N * A  # (M, K)
    
    # Initialize
    if theta_init is None:
        theta_hat = np.ones(K) / K
    else:
        theta_hat = theta_init.copy()
    
    tau_theta = 1.0 / K  # scalar variance approximation
    
    # GAMP state variables
    s_hat = np.zeros(M)  # output channel "residual"
    
    # Column norms squared for variance propagation
    col_norms_sq = np.sum(NA**2, axis=0)  # (K,)
    row_norms_sq = np.sum(NA**2, axis=1)  # (M,)
    
    mse_history = []
    
    for it in range(max_iter):
        theta_old = theta_hat.copy()
        
        # --- Forward pass: prior → channel ---
        # Effective prior on z = NA @ θ
        lam_hat = NA @ theta_hat + background  # (M,)
        tau_p = tau_theta * row_norms_sq  # (M,) variance propagation
        tau_p = np.maximum(tau_p, 1e-30)
        
        # Channel estimation
        z_hat, tau_z = _poisson_channel_estimation(y, lam_hat, tau_p)
        
        # Onsager-corrected residual
        s_hat_new = (z_hat - lam_hat) / np.maximum(tau_p, 1e-30)
        s_hat = damping * s_hat_new + (1 - damping) * s_hat
        
        tau_s = (1.0 / np.maximum(tau_z, 1e-30) - 1.0 / np.maximum(tau_p, 1e-30))
        tau_s = np.maximum(tau_s, 1e-30)
        
        # --- Backward pass: channel → prior ---
        # Effective observation of θ
        tau_r = 1.0 / np.maximum(np.sum(tau_s[:, None] * NA**2, axis=0), 1e-30)  # (K,)
        tau_r_scalar = np.mean(tau_r)
        
        r = theta_hat + tau_r * (NA.T @ s_hat)  # (K,)
        
        # Denoise with simplex prior
        theta_hat_new, tau_theta_new = _simplex_denoiser(r, tau_r_scalar)
        
        # Damped update
        theta_hat = damping * theta_hat_new + (1 - damping) * theta_hat
        theta_hat = _simplex_projection(theta_hat)  # ensure on simplex
        tau_theta = tau_theta_new
        
        # Convergence check
        delta = np.linalg.norm(theta_hat - theta_old)
        mse_history.append(delta)
        
        if verbose and it % 20 == 0:
            lam_fit = N * A @ theta_hat + background
            poisson_ll = np.sum(y * np.log(np.maximum(lam_fit, 1e-30)) - lam_fit)
            print(f"  GAMP iter {it:3d}: ||Δθ|| = {delta:.2e}, "
                  f"log L = {poisson_ll:.1f}")
        
        if delta < tol:
            if verbose:
                print(f"  GAMP converged at iteration {it}")
            return {
                'theta': theta_hat,
                'converged': True,
                'iterations': it + 1,
                'mse_history': mse_history
            }
    
    if verbose:
        print(f"  GAMP did not converge after {max_iter} iterations "
              f"(||Δθ|| = {mse_history[-1]:.2e})")
    
    return {
        'theta': theta_hat,
        'converged': False,
        'iterations': max_iter,
        'mse_history': mse_history
    }
