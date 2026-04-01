"""
shannon.py — Shannon Information Theory for the Poisson MAC
=============================================================

Complements fisher.py (estimation precision) with Shannon measures
(classification power). Together they give the complete information-
theoretic picture of the particle physics measurement channel.

Fisher answers:  'How precisely can we estimate continuous θ?' → CRB
Shannon answers: 'How reliably can we identify discrete K?'  → Fano

Key quantities:
  - I(K; Y): mutual information per event (classification capacity)
  - Bayes confusion matrix: optimal classifier performance ceiling
  - JSD(j,k): Jensen-Shannon distance between template pairs
  - Information budget: bits lost at each processing stage
  - Uncertainty decomposition: aliasing fraction vs statistical fraction

References:
  Cover & Thomas, Elements of Information Theory (Wiley, 2006)
  Verdú, Multiuser Detection (Cambridge, 1998), Ch. 5 (capacity)
  Fano, Transmission of Information (MIT Press, 1961)
"""

import numpy as np


def _entropy(p):
    """Shannon entropy H(p) = -Σ p_i log2 p_i (bits)."""
    p = np.asarray(p, dtype=float)
    mask = p > 1e-30
    return -np.sum(p[mask] * np.log2(p[mask]))


def template_entropy(A):
    """Shannon entropy of each template (column of A), viewed as a pmf.

    Parameters
    ----------
    A : ndarray (M, K) — template matrix (columns need not be normalized)

    Returns
    -------
    H : ndarray (K,) — entropy in bits per channel
    """
    K = A.shape[1]
    H = np.zeros(K)
    for k in range(K):
        a_k = A[:, k].copy()
        s = np.sum(a_k)
        if s > 0:
            a_k /= s
        H[k] = _entropy(a_k)
    return H


def classification_mi(A, theta):
    """Mutual information I(K; Y) for single-event channel classification.

    I(K; Y) = H(Y) - H(Y|K) = H(mixture) - Σ_k θ_k H(a_k)

    This measures how many bits about the channel identity K can be
    extracted from one event's detector measurement Y.

    Parameters
    ----------
    A : ndarray (M, K) — template matrix
    theta : ndarray (K,) — branching ratios (channel priors)

    Returns
    -------
    result : dict with keys:
        'MI': float — mutual information in bits
        'H_Y': float — entropy of the mixture distribution
        'H_Y_given_K': float — conditional entropy (avg template entropy)
        'H_K': float — prior entropy of channel identity
        'H_K_given_Y': float — posterior entropy after measurement
        'n_eff': float — effective channels per event (2^MI)
        'fano_bound': float — Fano lower bound on classification error
    """
    K = len(theta)
    theta = np.asarray(theta, dtype=float)

    # H(Y) — entropy of mixture
    mixture = A @ theta
    mix_sum = np.sum(mixture)
    if mix_sum > 0:
        mixture_norm = mixture / mix_sum
    else:
        mixture_norm = np.ones(A.shape[0]) / A.shape[0]
    H_Y = _entropy(mixture_norm)

    # H(Y|K) = Σ_k θ_k H(a_k)
    H_Y_given_K = 0.0
    for k in range(K):
        a_k = A[:, k].copy()
        s = np.sum(a_k)
        if s > 0:
            a_k /= s
        H_Y_given_K += theta[k] * _entropy(a_k)

    MI = H_Y - H_Y_given_K
    H_K = _entropy(theta)
    H_K_given_Y = H_K - MI

    # Fano inequality — tight form (C3):
    # H_b(P_e) + P_e·log₂(K−1) ≥ H(K|Y)
    # Solve for minimum P_e via binary search
    if K > 1 and H_K_given_Y > 0:
        rhs = H_K_given_Y
        lo, hi = 0.0, 1.0 - 1.0 / K  # max error is (K-1)/K
        for _ in range(100):
            mid = (lo + hi) / 2
            if mid < 1e-15 or mid > 1 - 1e-15:
                Hb = 0.0
            else:
                Hb = -mid * np.log2(mid) - (1 - mid) * np.log2(1 - mid)
            lhs = Hb + mid * np.log2(K - 1)
            if lhs < rhs:
                lo = mid
            else:
                hi = mid
        fano = lo
    else:
        fano = 0.0

    return {
        'MI': MI,
        'H_Y': H_Y,
        'H_Y_given_K': H_Y_given_K,
        'H_K': H_K,
        'H_K_given_Y': H_K_given_Y,
        'n_eff': 2.0 ** MI,
        'fano_bound': min(fano, 1.0),
    }


def bayes_confusion(A, theta):
    """Bayes-optimal (MAP) confusion matrix for single-event classification.

    For each mass bin i, compute the posterior P(k | Y=i) ∝ θ_k · a_k(i)
    and assign the event to argmax_k P(k|i). Average over the true
    distributions to get the per-channel accuracy.

    This gives the THEORETICAL CEILING for any classifier using the same
    features — no NN can exceed this accuracy.

    Parameters
    ----------
    A : ndarray (M, K) — template matrix
    theta : ndarray (K,) — branching ratios

    Returns
    -------
    result : dict with keys:
        'confusion': ndarray (K, K) — confusion[true, predicted], rows sum to 1
        'accuracy': ndarray (K,) — per-channel classification accuracy
        'overall': float — weighted overall accuracy
    """
    M, K = A.shape
    theta = np.asarray(theta, dtype=float)

    # Normalize templates
    A_norm = np.zeros_like(A)
    for k in range(K):
        s = np.sum(A[:, k])
        if s > 0:
            A_norm[:, k] = A[:, k] / s

    confusion = np.zeros((K, K))
    for i in range(M):
        # Posterior P(k | Y=i) ∝ θ_k · a_k(i)
        posts = theta * A_norm[i, :]
        total = np.sum(posts)
        if total < 1e-30:
            continue
        posts /= total
        # MAP decision: argmax returns lowest index on ties (deterministic
        # tie-breaking convention A5). For the μ=0% theorem, ties never occur
        # because e strictly dominates μ in every active bin.
        predicted = np.argmax(posts)

        # Weight by probability of bin i under each true channel
        for k in range(K):
            confusion[k, predicted] += A_norm[i, k]

    # Normalize rows to sum to 1
    for k in range(K):
        s = np.sum(confusion[k])
        if s > 0:
            confusion[k] /= s

    accuracy = np.diag(confusion)
    overall = np.sum(theta * accuracy)

    return {
        'confusion': confusion,
        'accuracy': accuracy,
        'overall': overall,
    }


def pairwise_jsd(A):
    """Jensen-Shannon divergence between all template pairs.

    JSD(j,k) = (1/2) D_KL(a_j || m) + (1/2) D_KL(a_k || m)
    where m = (a_j + a_k) / 2.

    JSD ∈ [0, 1] bits. JSD = 0 means identical templates (fully aliased).
    JSD = 1 means non-overlapping templates (perfectly distinguishable).

    Parameters
    ----------
    A : ndarray (M, K) — template matrix

    Returns
    -------
    JSD : ndarray (K, K) — symmetric JSD matrix (bits)
    """
    M, K = A.shape
    # Normalize templates
    A_norm = np.zeros_like(A)
    for k in range(K):
        s = np.sum(A[:, k])
        if s > 0:
            A_norm[:, k] = A[:, k] / s

    JSD = np.zeros((K, K))
    for j in range(K):
        for k in range(j + 1, K):
            m = 0.5 * (A_norm[:, j] + A_norm[:, k])
            kl_j = 0.0
            kl_k = 0.0
            for i in range(M):
                if A_norm[i, j] > 1e-30 and m[i] > 1e-30:
                    kl_j += A_norm[i, j] * np.log2(A_norm[i, j] / m[i])
                if A_norm[i, k] > 1e-30 and m[i] > 1e-30:
                    kl_k += A_norm[i, k] * np.log2(A_norm[i, k] / m[i])
            JSD[j, k] = JSD[k, j] = 0.5 * (kl_j + kl_k)

    return JSD


def kl_from_mixture(A, theta):
    """Per-channel KL divergence D_KL(a_k || mixture).

    Measures how distinguishable each channel is from the average
    observation. High D_KL → easy to identify; low D_KL → blends in.

    Parameters
    ----------
    A : ndarray (M, K) — template matrix
    theta : ndarray (K,) — branching ratios

    Returns
    -------
    DKL : ndarray (K,) — KL divergences in bits
    """
    K = len(theta)
    mixture = A @ theta
    mix_sum = np.sum(mixture)
    if mix_sum > 0:
        mixture_norm = mixture / mix_sum
    else:
        mixture_norm = np.ones(A.shape[0]) / A.shape[0]

    DKL = np.zeros(K)
    for k in range(K):
        a_k = A[:, k].copy()
        s = np.sum(a_k)
        if s > 0:
            a_k /= s
        mask = (a_k > 1e-30) & (mixture_norm > 1e-30)
        DKL[k] = np.sum(a_k[mask] * np.log2(a_k[mask] / mixture_norm[mask]))

    return DKL


def uncertainty_decomposition(crb_1d, crb_multi):
    """Decompose CRB variance into aliasing and statistical components.

    σ²_1D = σ²_aliasing + σ²_stats
    where σ²_stats ≈ σ²_multiD (with PID, aliasing removed)
          σ²_aliasing = σ²_1D - σ²_multiD

    Parameters
    ----------
    crb_1d : ndarray (K,) — CRB from 1D mass only
    crb_multi : ndarray (K,) — CRB from multi-dimensional features

    Returns
    -------
    result : dict with keys per channel:
        'aliasing_frac': ndarray (K,) — fraction due to aliasing
        'stats_frac': ndarray (K,) — fraction due to statistics
    """
    K = len(crb_1d)
    aliasing_frac = np.zeros(K)
    stats_frac = np.zeros(K)

    for k in range(K):
        s1 = crb_1d[k] if np.isfinite(crb_1d[k]) else np.inf
        sm = crb_multi[k] if np.isfinite(crb_multi[k]) else np.inf

        if np.isinf(s1):
            aliasing_frac[k] = 1.0
            stats_frac[k] = 0.0
        elif np.isinf(sm) or s1 <= sm:
            # Multi-D is worse or equal — no aliasing benefit
            aliasing_frac[k] = 0.0
            stats_frac[k] = 1.0
        else:
            aliasing_frac[k] = (s1 - sm) / s1
            stats_frac[k] = sm / s1

    return {
        'aliasing_frac': aliasing_frac,
        'stats_frac': stats_frac,
    }


def information_budget(A, theta, A_pid, theta_pid=None,
                       bg=None, bg_pid=None):
    """Compute the information budget across processing stages.

    Returns MI per event at each stage for a flow diagram:
      source → realistic detector → with PID

    Parameters
    ----------
    A : ndarray (M, K) — 1D mass templates
    theta : ndarray (K,) — branching ratios
    A_pid : ndarray (M+P, K) — augmented templates with PID
    theta_pid : ndarray or None — (same theta if None)
    bg, bg_pid : background arrays (unused, for API symmetry)

    Returns
    -------
    budget : dict with MI at each stage
    """
    if theta_pid is None:
        theta_pid = theta

    # Source entropy
    H_K = _entropy(theta)

    # 1D mass
    mi_1d = classification_mi(A, theta)

    # With PID
    mi_pid = classification_mi(A_pid, theta_pid)

    return {
        'H_source': H_K,
        'MI_1d': mi_1d['MI'],
        'MI_pid': mi_pid['MI'],
        'loss_to_aliasing': H_K - mi_1d['MI'],
        'gain_from_pid': mi_pid['MI'] - mi_1d['MI'],
        'residual_uncertainty': H_K - mi_pid['MI'],
        'efficiency_1d': mi_1d['MI'] / H_K if H_K > 0 else 0,
        'efficiency_pid': mi_pid['MI'] / H_K if H_K > 0 else 0,
    }
