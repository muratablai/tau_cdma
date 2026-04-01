"""
cascade.py — Cascade Decays as Concatenated Code Transformations
=================================================================

Implements:
  - Cascade FIM via Schur complement
  - Cascade spreading factor (BW, Gaussian, and Voigt regimes)
  - Data processing inequality verification
"""

import numpy as np
from numpy.linalg import inv, solve


def cascade_fim_schur(F_full, parent_idx, daughter_idx):
    """Compute effective FIM via Schur complement.

    F_eff = F_PP - F_PD · F_DD⁻¹ · F_PD^T  ⪯  F_PP

    Parameters
    ----------
    F_full : ndarray (n, n) — full FIM over all parameters
    parent_idx : array-like — indices of parent-level parameters
    daughter_idx : array-like — indices of daughter-level parameters

    Returns
    -------
    F_eff : ndarray — effective FIM for parent parameters
    F_PP : ndarray — parent-only block (for comparison)
    info_loss : float — tr(F_PP) - tr(F_eff), quantifying cascade info loss
    """
    parent_idx = np.array(parent_idx)
    daughter_idx = np.array(daughter_idx)

    F_PP = F_full[np.ix_(parent_idx, parent_idx)]
    F_DD = F_full[np.ix_(daughter_idx, daughter_idx)]
    F_PD = F_full[np.ix_(parent_idx, daughter_idx)]

    # F_eff = F_PP - F_PD · F_DD⁻¹ · F_PD^T
    try:
        correction = F_PD @ solve(F_DD, F_PD.T)
    except np.linalg.LinAlgError:
        # F_DD singular — no information at daughter level
        correction = np.zeros_like(F_PP)

    F_eff = F_PP - correction
    info_loss = np.trace(F_PP) - np.trace(F_eff)

    return F_eff, F_PP, info_loss


def cascade_sf(m_parent, gammas, regime='bw'):
    """Cascade spreading factor.

    Parameters
    ----------
    m_parent : float — parent mass (MeV)
    gammas : list of float — widths at each cascade stage (MeV)
    regime : str — 'bw' (Lorentzian/Breit-Wigner), 'gaussian', or 'voigt'

    Returns
    -------
    SF_cascade : float
    Gamma_eff : float — effective combined width
    """
    gammas = np.array(gammas)
    gammas = gammas[gammas > 0]  # ignore zero-width stages

    if len(gammas) == 0:
        return float('inf'), 0.0

    if regime == 'bw':
        # Lorentzian convolution: widths add linearly
        Gamma_eff = np.sum(gammas)
    elif regime == 'gaussian':
        # Gaussian convolution: widths add in quadrature
        Gamma_eff = np.sqrt(np.sum(gammas**2))
    elif regime == 'voigt':
        # General Voigt: separate BW and Gaussian contributions
        # Assume first width is BW (natural), rest are Gaussian (detector)
        Gamma_L = gammas[0]  # Lorentzian (natural width)
        if len(gammas) > 1:
            Gamma_G = np.sqrt(np.sum(gammas[1:]**2))
        else:
            Gamma_G = 0.0
        Gamma_eff = voigt_cascade_sf(Gamma_L, Gamma_G)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    SF = m_parent / Gamma_eff if Gamma_eff > 0 else float('inf')
    return SF, Gamma_eff


def voigt_cascade_sf(gamma_L, gamma_G):
    """Voigt FWHM approximation (Thompson et al. 1987).

    Γ_Voigt ≈ 0.5346·Γ_L + √(0.2166·Γ_L² + Γ_G²)

    Parameters
    ----------
    gamma_L : float — Lorentzian FWHM (sum of natural widths)
    gamma_G : float — Gaussian FWHM (quadrature sum of detector resolutions)

    Returns
    -------
    Gamma_Voigt : float — effective FWHM
    """
    return 0.5346 * gamma_L + np.sqrt(0.2166 * gamma_L**2 + gamma_G**2)


def cascade_tau_a1(N=1_000_000, M=200, sigma_det=20.0):
    """Demonstrate cascade bottleneck for τ→a₁ν→3πν.

    Constructs a simplified two-stage model:
      Stage 1: τ→a₁ν (template in τ visible mass)
      Stage 2: a₁→3π (sub-structure within the 3π system)

    Parameters
    ----------
    N : int — total event count (default 1e6)
    M : int — number of histogram bins (default 200)
    sigma_det : float — detector resolution in MeV (default 20)

    Returns
    -------
    dict with:
        'I1' : Fisher information at stage 1 (τ→a₁ν)
        'I2' : Fisher information at stage 2 (a₁→3π sub-structure)
        'bottleneck' : which stage limits information ('stage1' or 'stage2')
        'SF_cascade' : cascade spreading factor
    """
    from tau_cdma.tau.templates import M_TAU, M_A1, G_A1, M_RHO, G_RHO, M_PI0, bw_template

    m_bins = np.linspace(0, M_TAU, M)
    dm = m_bins[1] - m_bins[0]

    # Stage 1: τ→a₁ν template is the a₁ BW in τ visible mass
    a1_template = bw_template(m_bins, M_A1, G_A1, sigma_det=sigma_det)
    # Background: other τ decay modes (simplified as flat)
    bg_template = np.ones(M) / M

    # FIM for stage 1: distinguishing a₁ from background
    A1 = np.column_stack([a1_template, bg_template])
    theta1 = np.array([0.5, 0.5])
    b1 = 0.01 * dm * np.ones(M)
    lam1 = N * (A1 @ theta1) + b1
    W1 = 1.0 / np.maximum(lam1, 1e-30)
    F1 = N**2 * (A1.T * W1) @ A1
    I1 = np.trace(F1)

    # Stage 2: sub-structure within a₁→3π
    # The 3π Dalitz plot has structure from ρ(770) sub-channel
    # Simplified: two sub-channels (ρπ vs non-resonant 3π)
    m_3pi = np.linspace(3 * M_PI0, M_A1 + 2*G_A1, M)
    dm2 = m_3pi[1] - m_3pi[0] if M > 1 else 1.0

    rho_sub = bw_template(m_3pi, M_RHO, G_RHO, sigma_det=sigma_det*2)
    flat_sub = np.ones(M)
    flat_sub /= (np.sum(flat_sub) * dm2) if np.sum(flat_sub) > 0 else 1.0

    A2 = np.column_stack([rho_sub, flat_sub])
    theta2 = np.array([0.7, 0.3])
    b2 = 0.01 * dm2 * np.ones(M)
    # Effective N at stage 2: only events that went through a₁
    N2 = N * theta1[0]
    lam2 = N2 * (A2 @ theta2) + b2
    W2 = 1.0 / np.maximum(lam2, 1e-30)
    F2 = N2**2 * (A2.T * W2) @ A2
    I2 = np.trace(F2)

    SF_c, Gamma_eff = cascade_sf(M_TAU, [G_A1], regime='bw')

    return {
        'I1': I1,
        'I2': I2,
        'F1': F1,
        'F2': F2,
        'bottleneck': 'stage2' if I2 < I1 else 'stage1',
        'SF_cascade': SF_c,
        'Gamma_eff': Gamma_eff,
    }
