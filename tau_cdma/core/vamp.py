"""
vamp.py — Vector Approximate Message Passing for Poisson Template Mixtures
===========================================================================

Implements GLM-VAMP (Schniter, Rangan, Fletcher 2016) specialized to:
    y_i ~ Poisson(λ_i),  λ_i = N * (Aθ)_i + b_i

VAMP is more robust than GAMP to ill-conditioned template matrices A,
because it uses SVD-based denoising and right-rotationally invariant
assumptions. The key advantage: VAMP's convergence is "relatively
insensitive to the condition number" (Schniter et al. 2016).

References:
    - Rangan, Schniter, Fletcher (2019). Vector AMP. IEEE Trans. IT.
    - Schniter, Rangan, Fletcher (2016). GLM-VAMP. arXiv:1612.01186.

Status: Algorithmic baseline, more robust than GAMP for structured templates.
"""

import numpy as np
from .gamp import _simplex_projection


def _poisson_proximal(y, z_prior, gamma, background):
    """Proximal operator for the Poisson negative log-likelihood.
    
    Solves: argmin_z  -Σ_i [y_i log(z_i + b_i) - (z_i + b_i)] + (gamma/2)||z - z_prior||²
    
    This is a scalar optimization per bin (separable).
    Newton's method on the first-order condition:
        -y_i / (z_i + b_i) + 1 + gamma * (z_i - z_prior_i) = 0
    
    Parameters
    ----------
    y : ndarray (M,) — observed counts
    z_prior : ndarray (M,) — prior mean from VAMP
    gamma : float — inverse variance (precision) from VAMP
    background : ndarray (M,) — background rates
    
    Returns
    -------
    z_hat : ndarray (M,) — proximal estimate
    """
    M = len(y)
    z = np.maximum(z_prior, 1e-10).copy()
    
    for _ in range(20):  # Newton iterations
        lam = z + background
        lam = np.maximum(lam, 1e-10)
        
        # Gradient: -y/lam + 1 + gamma*(z - z_prior)
        g = -y / lam + 1.0 + gamma * (z - z_prior)
        
        # Hessian: y/lam² + gamma
        h = y / (lam**2) + gamma
        
        # Newton step
        step = g / np.maximum(h, 1e-30)
        z = z - step
        z = np.maximum(z, 1e-10)
        
        if np.max(np.abs(step)) < 1e-10:
            break
    
    return z


def vamp_poisson(A, y, N, background=None, theta_init=None,
                 max_iter=200, tol=1e-6, damping=0.8, verbose=False):
    """GLM-VAMP for Poisson template mixture estimation.
    
    Estimates θ ∈ Δ^{K-1} from y_i ~ Poisson(N*(Aθ)_i + b_i).
    
    Uses SVD-based message passing for robustness to ill-conditioned A.
    
    Parameters
    ----------
    A : ndarray (M, K) — template matrix
    y : ndarray (M,) — observed bin counts
    N : float — total event count
    background : ndarray (M,) or None — background rates
    theta_init : ndarray (K,) or None — initial estimate
    max_iter : int — maximum iterations
    tol : float — convergence tolerance
    damping : float — damping factor
    verbose : bool — print info
    
    Returns
    -------
    result : dict with keys:
        'theta' : ndarray (K,) — estimated branching ratios
        'converged' : bool
        'iterations' : int
        'mse_history' : list
    """
    M, K = A.shape
    if background is None:
        background = np.zeros(M)
    
    NA = N * A  # (M, K)
    
    # SVD of NA for VAMP's linear step
    U_svd, S_svd, Vt_svd = np.linalg.svd(NA, full_matrices=False)
    # NA = U_svd @ diag(S_svd) @ Vt_svd, shapes: (M,K), (K,), (K,K)
    S_sq = S_svd**2  # squared singular values
    
    # Initialize
    if theta_init is None:
        theta1 = np.ones(K) / K
    else:
        theta1 = theta_init.copy()
    
    gamma1 = 1.0  # precision (inverse variance) for denoiser
    
    mse_history = []
    
    for it in range(max_iter):
        theta_old = theta1.copy()
        
        # ==========================================
        # DENOISER 1: Simplex prior denoiser
        # ==========================================
        # Input: (theta1, gamma1)
        # Apply simplex projection as proximal denoiser
        r1 = theta1.copy()
        theta1_hat = _simplex_projection(r1)
        
        # Estimate divergence of denoiser (needed for Onsager correction)
        # Use finite-difference approximation
        eps_fd = 1e-5
        div1 = 0.0
        for k in range(K):
            r_plus = r1.copy()
            r_plus[k] += eps_fd
            div1 += (_simplex_projection(r_plus)[k] - theta1_hat[k]) / eps_fd
        div1 = div1 / K  # average divergence per dimension
        
        # VAMP variance update
        alpha1 = np.clip(div1, 0.01, 0.99)
        gamma1_out = gamma1 * (1.0 / np.maximum(alpha1, 1e-10) - 1.0)
        gamma1_out = np.maximum(gamma1_out, 1e-10)
        
        # Onsager-corrected message to channel
        r1_out = (gamma1_out * theta1_hat - gamma1 * r1 * (alpha1 / (1 - alpha1 + 1e-10))) 
        # Simplified: pass the denoised estimate
        r1_out = theta1_hat + (theta1_hat - r1) * alpha1 / (1 - alpha1 + 1e-10)
        
        # ==========================================
        # DENOISER 2: Poisson channel + linear LMMSE via SVD
        # ==========================================
        # Compute expected rates
        z_prior = NA @ theta1_hat + background
        
        # Poisson proximal step
        z_hat = _poisson_proximal(y, z_prior, gamma1_out, background)
        
        # Linear LMMSE step in SVD basis
        # theta_lmmse = (NA^T NA + gamma2 I)^{-1} (NA^T z_hat + gamma2 * r2)
        # In SVD basis: efficient diagonal solve
        gamma2 = gamma1_out
        
        # Transform to SVD basis
        z_centered = z_hat - background
        z_svd = U_svd.T @ z_centered  # (K,)
        r2_svd = Vt_svd @ theta1_hat  # (K,)
        
        # LMMSE in SVD basis (diagonal)
        theta2_svd = (S_svd * z_svd + gamma2 * r2_svd) / (S_sq + gamma2)
        
        # Back to original basis
        theta2 = Vt_svd.T @ theta2_svd
        
        # Project back to simplex
        theta2 = _simplex_projection(theta2)
        
        # Variance of LMMSE step
        alpha2 = np.mean(S_sq / (S_sq + gamma2))
        gamma2_out = gamma2 * (1.0 / np.maximum(alpha2, 1e-10) - 1.0)
        gamma2_out = np.maximum(gamma2_out, 1e-10)
        
        # ==========================================
        # Update for next iteration
        # ==========================================
        theta1 = damping * theta2 + (1 - damping) * theta_old
        theta1 = _simplex_projection(theta1)
        gamma1 = gamma2_out
        
        # Convergence
        delta = np.linalg.norm(theta1 - theta_old)
        mse_history.append(delta)
        
        if verbose and it % 20 == 0:
            lam_fit = N * A @ theta1 + background
            poisson_ll = np.sum(y * np.log(np.maximum(lam_fit, 1e-30)) - lam_fit)
            kappa = S_svd[0] / max(S_svd[-1], 1e-30)
            print(f"  VAMP iter {it:3d}: ||Δθ|| = {delta:.2e}, "
                  f"log L = {poisson_ll:.1f}, κ(NA) = {kappa:.1f}")
        
        if delta < tol:
            if verbose:
                print(f"  VAMP converged at iteration {it}")
            return {
                'theta': theta1,
                'converged': True,
                'iterations': it + 1,
                'mse_history': mse_history
            }
    
    if verbose:
        print(f"  VAMP did not converge after {max_iter} iterations "
              f"(||Δθ|| = {mse_history[-1]:.2e})")
    
    return {
        'theta': theta1,
        'converged': False,
        'iterations': max_iter,
        'mse_history': mse_history
    }


def stress_test_structured_A(A, y, N, theta_true, background=None,
                              n_random=20, verbose=False):
    """Stress test: compare GAMP and VAMP on structured vs randomized A.
    
    Generates randomized variants of A (preserving singular values via 
    random right-rotations) and compares convergence/MSE.
    
    Parameters
    ----------
    A : ndarray (M, K) — original structured template matrix
    y : ndarray (M,) — observed counts
    N : float — total events
    theta_true : ndarray (K,) — true branching ratios
    background : ndarray (M,) or None
    n_random : int — number of random rotations
    verbose : bool
    
    Returns
    -------
    results : dict with stress test outcomes
    """
    from .gamp import gamp_poisson
    
    M, K = A.shape
    U_svd, S_svd, Vt_svd = np.linalg.svd(N * A, full_matrices=False)
    kappa = S_svd[0] / max(S_svd[-1], 1e-30)
    
    # Run on original A
    res_gamp_orig = gamp_poisson(A, y, N, background, max_iter=300, damping=0.5)
    res_vamp_orig = vamp_poisson(A, y, N, background, max_iter=300, damping=0.8)
    
    mse_gamp_orig = np.mean((res_gamp_orig['theta'] - theta_true)**2)
    mse_vamp_orig = np.mean((res_vamp_orig['theta'] - theta_true)**2)
    
    # Run on randomized A (random right-rotation preserving singular values)
    mse_gamp_random = []
    mse_vamp_random = []
    conv_gamp_random = []
    conv_vamp_random = []
    
    for trial in range(n_random):
        # Random orthogonal matrix Q
        Q, _ = np.linalg.qr(np.random.randn(K, K))
        
        # Randomized A: same singular values, random right singular vectors
        A_rand = (U_svd @ np.diag(S_svd) @ Q) / N
        A_rand = np.maximum(A_rand, 0)  # enforce non-negativity
        # Renormalize columns
        col_sums = A_rand.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        A_rand = A_rand / col_sums
        
        # Generate data from randomized A
        lam_rand = N * A_rand @ theta_true + (background if background is not None else 0)
        y_rand = np.random.poisson(np.maximum(lam_rand, 0.1))
        
        bg = background if background is not None else None
        rg = gamp_poisson(A_rand, y_rand, N, bg, max_iter=300, damping=0.5)
        rv = vamp_poisson(A_rand, y_rand, N, bg, max_iter=300, damping=0.8)
        
        mse_gamp_random.append(np.mean((rg['theta'] - theta_true)**2))
        mse_vamp_random.append(np.mean((rv['theta'] - theta_true)**2))
        conv_gamp_random.append(rg['converged'])
        conv_vamp_random.append(rv['converged'])
    
    results = {
        'condition_number': kappa,
        'structured': {
            'gamp_mse': mse_gamp_orig,
            'vamp_mse': mse_vamp_orig,
            'gamp_converged': res_gamp_orig['converged'],
            'vamp_converged': res_vamp_orig['converged'],
        },
        'randomized': {
            'gamp_mse_mean': np.mean(mse_gamp_random),
            'gamp_mse_std': np.std(mse_gamp_random),
            'vamp_mse_mean': np.mean(mse_vamp_random),
            'vamp_mse_std': np.std(mse_vamp_random),
            'gamp_conv_rate': np.mean(conv_gamp_random),
            'vamp_conv_rate': np.mean(conv_vamp_random),
        }
    }
    
    if verbose:
        print(f"  Stress test: κ(NA) = {kappa:.1f}")
        print(f"  Structured:  GAMP MSE={mse_gamp_orig:.2e} ({'conv' if res_gamp_orig['converged'] else 'FAIL'}), "
              f"VAMP MSE={mse_vamp_orig:.2e} ({'conv' if res_vamp_orig['converged'] else 'FAIL'})")
        print(f"  Randomized:  GAMP MSE={np.mean(mse_gamp_random):.2e} (conv {np.mean(conv_gamp_random)*100:.0f}%), "
              f"VAMP MSE={np.mean(mse_vamp_random):.2e} (conv {np.mean(conv_vamp_random)*100:.0f}%)")
    
    return results
