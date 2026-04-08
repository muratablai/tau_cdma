"""
robust.py — Robustness Tools for Template Mismatch and MAP Collapse Diagnostics
================================================================================

Implements:
  1. Godambe/sandwich covariance estimator for Poisson MLE under template mismatch
  2. Dominance-margin diagnostic for MAP collapse theorem robustness
  3. Poisson-specific KL-Fisher remainder with explicit constants

References:
    - White (1982). MLE of misspecified models. Econometrica.
    - Godambe (1960). Optimum estimating equations. Ann. Math. Statist.
    - Collaborator proposal: dominance-margin lemma for μ=0% robustness.

Status: Publishable robustness tools for the validated framework.
"""

import numpy as np
from .fisher import poisson_fim, constrained_crb


# ============================================================
# 1. Godambe / Sandwich Covariance Under Template Mismatch
# ============================================================

def godambe_sandwich(A_assumed, A_true, theta_star, N, background=None):
    """Sandwich (Godambe) covariance for Poisson MLE under template mismatch.
    
    When the assumed templates A_assumed differ from the true templates A_true,
    the MLE converges to the KL-projection θ* and its covariance is:
    
        Σ = H⁻¹ V H⁻¹  (the "sandwich")
    
    where:
        H = E_true[-∂²ℓ/∂θ∂θᵀ]  (expected Hessian under true model)
        V = E_true[(∂ℓ/∂θ)(∂ℓ/∂θ)ᵀ]  (score variance under true model)
    
    Under correct specification, H = V = F (Fisher), and Σ = F⁻¹.
    Under misspecification, H ≠ V, and the sandwich gives larger variance.
    
    Parameters
    ----------
    A_assumed : ndarray (M, K) — templates used by the estimator
    A_true : ndarray (M, K) — true templates generating the data
    theta_star : ndarray (K,) — pseudo-true parameter (KL-projection)
    N : float — total event count
    background : ndarray (M,) or None
    
    Returns
    -------
    result : dict with keys:
        'sandwich_cov' : ndarray (K, K) — sandwich covariance matrix
        'H' : ndarray (K, K) — expected Hessian (under true model, assumed templates)
        'V' : ndarray (K, K) — score variance (under true model, assumed templates)
        'F_true' : ndarray (K, K) — Fisher information (true model)
        'inflation' : ndarray (K,) — ratio of sandwich diagonal to true CRB
    """
    M, K = A_assumed.shape
    if background is None:
        background = np.zeros(M)
    
    # True expected counts
    lam_true = N * (A_true @ theta_star) + background
    lam_true = np.maximum(lam_true, 1e-30)
    
    # Assumed expected counts (what the estimator thinks)
    lam_assumed = N * (A_assumed @ theta_star) + background
    lam_assumed = np.maximum(lam_assumed, 1e-30)
    
    # H = expected negative Hessian of assumed log-likelihood under true model
    # V = score outer product under true model
    #
    # Derivation:
    # For the Poisson log-likelihood ℓ(θ) = Σ_i [y_i log λ^a_i - λ^a_i]:
    #   H_jk = -E_true[∂²ℓ/∂θ²] = N² Σ_i E[y_i] a^a_ji a^a_ki / (λ^a_i)²
    #        = N² Σ_i λ^t_i a^a_ji a^a_ki / (λ^a_i)²
    #   V_jk = Cov_true(∂ℓ/∂θ_j, ∂ℓ/∂θ_k) = N² Σ_i Var(y_i) a^a_ji a^a_ki / (λ^a_i)²
    #        = N² Σ_i λ^t_i a^a_ji a^a_ki / (λ^a_i)²
    #
    # For Poisson, H = V per-bin (first Bartlett identity). The sandwich becomes
    # non-trivial because H should use the *assumed model's curvature* (1/λ^a)
    # while V uses the *true model's score variance* (λ^t / (λ^a)²).
    #
    # Under correct specification (λ^t = λ^a), both reduce to the standard FIM.
    # Under misspecification, H ≠ V because 1/λ^a ≠ λ^t/(λ^a)².
    #
    # Note: the mean score is NOT zero under misspecification:
    #   E[score_j] = N Σ_i (λ^t_i - λ^a_i) a^a_ji / λ^a_i ≠ 0
    # This bias means θ_star is the KL projection, not the true θ.
    
    # H: assumed-model FIM evaluated at θ_star
    W_Ha = 1.0 / lam_assumed
    H_assumed = N**2 * (A_assumed.T * W_Ha) @ A_assumed
    
    # V: score variance under true model, using assumed-model score function
    W_Va = lam_true / (lam_assumed**2)
    V_mismatch = N**2 * (A_assumed.T * W_Va) @ A_assumed
    
    # Sandwich: Σ = H⁻¹ V H⁻¹
    try:
        H_inv = np.linalg.inv(H_assumed + 1e-12 * np.eye(K))
        sandwich = H_inv @ V_mismatch @ H_inv
    except np.linalg.LinAlgError:
        sandwich = np.full((K, K), np.inf)
    
    # True Fisher (for comparison)
    F_true = poisson_fim(A_true, theta_star, N, background)
    
    # Inflation ratio: how much worse is the sandwich vs the ideal CRB
    try:
        F_true_inv = np.linalg.inv(F_true + 1e-12 * np.eye(K))
        inflation = np.diag(sandwich) / np.maximum(np.diag(F_true_inv), 1e-30)
    except np.linalg.LinAlgError:
        inflation = np.full(K, np.inf)
    
    return {
        'sandwich_cov': sandwich,
        'H': H_assumed,
        'V': V_mismatch,
        'F_true': F_true,
        'inflation': inflation
    }


def template_mismatch_sensitivity(A, theta, N, background=None,
                                   epsilon=0.01, n_samples=100, seed=42):
    """Bootstrap template mismatch: perturb A and track diagnostics.
    
    Generates multiplicative perturbations A_pert = A * (1 + ε * Z) / norm
    and reports the distribution of η_k, λ_min, and CRB under mismatch.
    
    Parameters
    ----------
    A : ndarray (M, K) — nominal templates
    theta : ndarray (K,) — operating point
    N : float — total events
    background : ndarray (M,) or None
    epsilon : float — perturbation magnitude
    n_samples : int — number of bootstrap samples
    seed : int — random seed
    
    Returns
    -------
    result : dict with mismatch statistics
    """
    from .interference import multiuser_efficiency
    
    rng = np.random.default_rng(seed)
    M, K = A.shape
    
    eta_samples = np.zeros((n_samples, K))
    lam_min_samples = np.zeros(n_samples)
    crb_diag_samples = np.zeros((n_samples, K))
    
    for s in range(n_samples):
        # Multiplicative perturbation
        Z = rng.standard_normal((M, K))
        A_pert = A * (1.0 + epsilon * Z)
        A_pert = np.maximum(A_pert, 0.0)  # non-negativity
        # Renormalize columns
        col_sums = A_pert.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        A_pert = A_pert / col_sums
        
        # Compute diagnostics
        F_pert = poisson_fim(A_pert, theta, N, background)
        eigvals = np.linalg.eigvalsh(F_pert)
        lam_min_samples[s] = eigvals[0]
        
        # η from R matrix
        D = np.diag(F_pert)
        D_safe = np.maximum(D, 1e-30)
        D_sqrt_inv = 1.0 / np.sqrt(D_safe)
        R = F_pert * np.outer(D_sqrt_inv, D_sqrt_inv)
        try:
            R_inv = np.linalg.inv(R + 1e-12 * np.eye(K))
            eta_samples[s] = 1.0 / np.diag(R_inv)
            eta_samples[s] = np.clip(eta_samples[s], 0, 1)
        except np.linalg.LinAlgError:
            eta_samples[s] = 0.0
        
        # Constrained CRB
        CRB_c = constrained_crb(F_pert)
        crb_diag_samples[s] = np.diag(CRB_c)
    
    return {
        'eta_mean': np.mean(eta_samples, axis=0),
        'eta_std': np.std(eta_samples, axis=0),
        'eta_q05': np.percentile(eta_samples, 5, axis=0),
        'eta_q95': np.percentile(eta_samples, 95, axis=0),
        'lam_min_mean': np.mean(lam_min_samples),
        'lam_min_std': np.std(lam_min_samples),
        'lam_min_q05': np.percentile(lam_min_samples, 5),
        'crb_mean': np.mean(crb_diag_samples, axis=0),
        'crb_inflation': np.mean(crb_diag_samples, axis=0) / np.maximum(
            crb_diag_samples[0], 1e-30),  # relative to first sample
        'epsilon': epsilon,
        'n_samples': n_samples,
    }


# ============================================================
# 2. Dominance Margin Diagnostic for MAP Collapse Robustness
# ============================================================

def dominance_margin(A, theta, target_class=1, competitor_class=0):
    """Compute the log dominance margin for MAP collapse analysis.
    
    The dominance margin of class u against its best competitor:
        Δ_u(i) = log[max_{j≠u} π_j p_j(i)] - log[π_u p_u(i)]
    
    If Δ_u(i) > 0 for all bins i, class u is never predicted by MAP.
    The minimum margin γ = min_i Δ_u(i) quantifies robustness.
    
    Parameters
    ----------
    A : ndarray (M, K) — template matrix (columns = p(i|k))
    theta : ndarray (K,) — priors (branching ratios)
    target_class : int — class index u to check for collapse
    competitor_class : int or None — specific competitor (None = best)
    
    Returns
    -------
    result : dict with keys:
        'margin_per_bin' : ndarray (M,) — Δ_u(i) for each bin
        'min_margin' : float — γ = min_i Δ_u(i)
        'collapses' : bool — True if γ > 0 (MAP never predicts u)
        'margin_threshold' : float — max perturbation ε for which collapse persists
    """
    M, K = A.shape
    u = target_class
    
    # Weighted likelihoods: π_k * p(i|k) for each class
    weighted = theta[:, None] * A.T  # (K, M): weighted[k, i] = θ_k * A[i,k]
    
    # Score of target class
    score_u = weighted[u]  # (M,)
    
    # Max competitor score
    mask = np.ones(K, dtype=bool)
    mask[u] = False
    score_competitors = weighted[mask]  # (K-1, M)
    score_max_competitor = np.max(score_competitors, axis=0)  # (M,)
    
    # Only check bins where target class has nonzero probability
    # (Theorem condition: "for all i with p_b(i) > 0")
    active_bins = A[:, u] > 1e-15  # bins where target has nonzero template
    
    if not np.any(active_bins):
        # Target has zero probability everywhere — trivially collapsed
        return {
            'margin_per_bin': np.full(M, np.inf),
            'min_margin': np.inf,
            'collapses': True,
            'margin_threshold': 1.0,
        }
    
    # Log dominance margin (only for active bins)
    safe_u = np.maximum(score_u, 1e-300)
    safe_comp = np.maximum(score_max_competitor, 1e-300)
    margin = np.full(M, np.inf)  # inactive bins have infinite margin (irrelevant)
    margin[active_bins] = np.log(safe_comp[active_bins]) - np.log(safe_u[active_bins])
    
    min_margin = np.min(margin[active_bins])
    collapses = min_margin > 0
    
    # Robustness: under multiplicative perturbation (1+ε), collapse persists if
    # ε < tanh(γ/8) (from collaborator's lemma, sufficient condition)
    if min_margin > 0:
        margin_threshold = np.tanh(min_margin / 8.0)
    else:
        margin_threshold = 0.0
    
    return {
        'margin_per_bin': margin,
        'min_margin': min_margin,
        'collapses': collapses,
        'margin_threshold': margin_threshold,
    }


def dominance_margin_sweep(A, theta, target_class=1, M_values=None):
    """Sweep dominance margin over binning resolution M.
    
    Parameters
    ----------
    A : ndarray (M_max, K) — high-resolution templates
    theta : ndarray (K,) — priors
    target_class : int — class to check
    M_values : list of int or None — binning values to test
    
    Returns
    -------
    results : list of (M, margin_dict) tuples
    """
    M_max, K = A.shape
    if M_values is None:
        M_values = [10, 20, 50, 100, 200, 500]
    
    results = []
    for M_target in M_values:
        if M_target >= M_max:
            A_binned = A
        else:
            # Rebin by averaging groups of bins
            bin_size = M_max // M_target
            A_binned = np.zeros((M_target, K))
            for b in range(M_target):
                start = b * bin_size
                end = min(start + bin_size, M_max)
                A_binned[b] = A[start:end].sum(axis=0)
            # Renormalize
            col_sums = A_binned.sum(axis=0)
            col_sums[col_sums == 0] = 1.0
            A_binned = A_binned / col_sums
        
        dm = dominance_margin(A_binned, theta, target_class)
        results.append((M_target, dm))
    
    return results


# ============================================================
# 3. Poisson-Specific KL-Fisher Remainder
# ============================================================

def poisson_kl_fisher_remainder(lam, lam_prime):
    """Compute Poisson KL divergence with exact Fisher quadratic and remainder.
    
    For Poisson(λ') vs Poisson(λ):
        D_KL = λ' log(λ'/λ) - (λ' - λ)
             = δ²/(2λ) + R
    
    where δ = λ' - λ and R = δ³/(6ξ²) for some ξ between λ and λ'
    (from Taylor's theorem on g(x) = x log(x/λ) - (x-λ) with g'''(x) = -1/x²).
    
    Note: the correct coefficient is 1/6 (from g'''/3! = (-1/x²)/6),
    not 1/3 as sometimes stated. This is because the Taylor remainder
    involves the third derivative divided by 3! = 6.
    
    Parameters
    ----------
    lam : ndarray — reference Poisson means (must be > 0)
    lam_prime : ndarray — perturbed Poisson means (must be > 0)
    
    Returns
    -------
    result : dict with keys:
        'kl_exact' : ndarray — exact KL divergence per bin
        'fisher_quadratic' : ndarray — δ²/(2λ) per bin
        'remainder' : ndarray — exact remainder (KL - quadratic)
        'remainder_bound' : ndarray — |δ|³/(6 λ² (1-ρ)²) upper bound
        'relative_perturbation' : ndarray — |δ|/λ per bin
    """
    lam = np.asarray(lam, dtype=float)
    lam_prime = np.asarray(lam_prime, dtype=float)
    
    delta = lam_prime - lam
    
    # Exact KL
    kl_exact = lam_prime * np.log(np.maximum(lam_prime, 1e-300) / np.maximum(lam, 1e-300)) - delta
    kl_exact = np.maximum(kl_exact, 0.0)  # KL is non-negative
    
    # Fisher quadratic term
    fisher_quad = delta**2 / (2.0 * np.maximum(lam, 1e-30))
    
    # Exact remainder
    remainder = kl_exact - fisher_quad
    
    # Upper bound on |remainder|
    rho = np.abs(delta) / np.maximum(lam, 1e-30)
    rho_safe = np.minimum(rho, 0.99)  # ensure (1-ρ) > 0
    remainder_bound = np.abs(delta)**3 / (6.0 * lam**2 * (1.0 - rho_safe)**2)
    
    return {
        'kl_exact': kl_exact,
        'fisher_quadratic': fisher_quad,
        'remainder': remainder,
        'remainder_bound': remainder_bound,
        'relative_perturbation': rho,
    }


def poisson_mixture_kl_fisher_expansion(A, theta, theta_ref, N, background=None):
    """KL-Fisher expansion for the full Poisson template mixture.
    
    Computes D_KL(P_{θ'} || P_θ₀) = ½ h^T F(θ₀) h + R(θ₀, h)
    where h = θ' - θ₀, specialized to the Poisson bin model.
    
    Parameters
    ----------
    A : ndarray (M, K) — templates
    theta : ndarray (K,) — perturbed parameter θ'
    theta_ref : ndarray (K,) — reference parameter θ₀
    N : float — total events
    background : ndarray (M,) or None
    
    Returns
    -------
    result : dict with KL decomposition
    """
    M, K = A.shape
    if background is None:
        background = np.zeros(M)
    
    h = theta - theta_ref
    
    # Expected counts at reference and perturbed
    lam_ref = N * (A @ theta_ref) + background
    lam_pert = N * (A @ theta) + background
    
    # Per-bin KL-Fisher decomposition
    bin_result = poisson_kl_fisher_remainder(lam_ref, lam_pert)
    
    # Total KL = sum over bins
    kl_total = np.sum(bin_result['kl_exact'])
    
    # Fisher quadratic
    F = poisson_fim(A, theta_ref, N, background)
    fisher_quad_total = 0.5 * h @ F @ h
    
    # Total remainder
    remainder_total = kl_total - fisher_quad_total
    remainder_bound_total = np.sum(bin_result['remainder_bound'])
    
    return {
        'kl_total': kl_total,
        'fisher_quadratic': fisher_quad_total,
        'remainder': remainder_total,
        'remainder_bound': remainder_bound_total,
        'h_norm': np.linalg.norm(h),
        'max_relative_perturbation': np.max(bin_result['relative_perturbation']),
    }
