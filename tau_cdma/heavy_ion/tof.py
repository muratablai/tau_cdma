"""
tof.py — Time-of-Flight Templates and Multi-Detector Fusion
=============================================================

Implements TOF-based particle identification via m² measurement
and multi-detector (TPC+TOF) Fisher information fusion.

Functions:
  - tof_template()  — Gaussian template in m²_TOF space (Eq. 3.13)
  - joint_fisher()  — TPC+TOF fusion with partial coverage (Eq. 3.15-3.17)

TOF parameters from ALICE:
  Run 1: σ_t ≈ 80 ps overall
  Run 2: σ_t ≈ 56 ps intrinsic, ~60 ps overall in Pb-Pb
         (start-time well-determined from high multiplicity)
         Ref: arXiv:1809.00574
  Run 3: σ_t ≈ 68 ps intrinsic, FT0 start-time 17 ps (pp) / 4.4 ps (Pb-Pb)
         Overall ~70 ps pp, ~68 ps Pb-Pb (C7 correction applied)
"""

import numpy as np
from scipy.stats import norm


# =====================================================================
#  CONSTANTS
# =====================================================================

C_LIGHT_M_PS = 2.99792458e-4   # speed of light in m/ps

# Default ALICE TOF parameters (Run 2 Pb-Pb)
TOF_RESOLUTION_PS = 60.0       # overall time resolution (ps), Run 2 Pb-Pb
TOF_PATH_LENGTH_M = 3.8        # average path length (m), ALICE TOF
TOF_MATCH_EFF = 0.60           # average matching efficiency


# =====================================================================
#  6.8  TOF TEMPLATE
# =====================================================================

def tof_template(p, mass, sigma_t=TOF_RESOLUTION_PS,
                 L=TOF_PATH_LENGTH_M, m2_bin_edges=None):
    """Construct TOF m² template for a given species.

    The TOF measures flight time t, from which m² is computed:
      m² = p²(t²c²/L² - 1)

    The resolution in m² space (Eq. 3.13), from error propagation on t=L/(βc):
      σ(m²) = 2p² × c × σ_t / (β × L)
    Equivalently: σ(m²) = 2m²βγ² × c × σ_t / L

    Parameters
    ----------
    p : float — momentum in GeV/c
    mass : float — particle mass in GeV/c²
    sigma_t : float — time resolution in ps
    L : float — path length in m
    m2_bin_edges : ndarray (B+1,) or None — m² bin edges in GeV²

    Returns
    -------
    template : ndarray (B,) — probability per bin, sums to 1
    sigma_m2 : float — m² resolution in GeV²
    """
    m2_true = mass**2
    E = np.sqrt(p**2 + mass**2)
    beta = p / E
    gamma = E / mass if mass > 0 else 1e10

    # σ(m²) = 2 p² c σ_t / (β L)
    # Derived from m² = p²(c²t²/L² − 1), dm²/dt = 2p²c/(βL)
    sigma_m2 = 2.0 * p**2 * C_LIGHT_M_PS * sigma_t / (beta * L)

    if m2_bin_edges is None:
        m2_bin_edges = np.linspace(-0.5, 2.0, 101)

    cdf_vals = norm.cdf(m2_bin_edges, loc=m2_true, scale=max(sigma_m2, 1e-15))
    template = np.diff(cdf_vals)

    total = template.sum()
    if total > 1e-30:
        template /= total
    else:
        template[:] = 1.0 / len(template)

    return template, sigma_m2


def build_tof_template_matrix(p, masses, n_bins=100,
                               sigma_t=TOF_RESOLUTION_PS,
                               L=TOF_PATH_LENGTH_M):
    """Build TOF template matrix at momentum p.

    Parameters
    ----------
    p : float — momentum
    masses : list of float — species masses
    n_bins : int
    sigma_t : float — time resolution in ps
    L : float — path length in m

    Returns
    -------
    A_tof : ndarray (n_bins, K) — TOF template matrix
    m2_bin_edges : ndarray (n_bins+1,)
    """
    K = len(masses)
    m2_bin_edges = np.linspace(-0.2, 1.5, n_bins + 1)
    A_tof = np.zeros((n_bins, K))
    for k in range(K):
        A_tof[:, k], _ = tof_template(p, masses[k], sigma_t, L, m2_bin_edges)
    return A_tof, m2_bin_edges


# =====================================================================
#  6.9  JOINT FISHER (TPC + TOF FUSION)
# =====================================================================

def joint_fisher(F_TPC, F_TOF, eps_match=TOF_MATCH_EFF):
    """Multi-detector fusion: TPC + TOF with partial coverage.

    Eq. 3.15: F_joint = F_TPC + F_TOF (independence)
    Eq. 3.16: F_eff = ε × F_joint + (1−ε) × F_TPC
    Eq. 3.17: ΔI_TOF = ε × F_TOF

    Parameters
    ----------
    F_TPC : ndarray (K, K) — TPC Fisher information
    F_TOF : ndarray (K, K) — TOF Fisher information
    eps_match : float — TOF matching efficiency

    Returns
    -------
    F_eff : ndarray (K, K) — effective (population-averaged) Fisher info
    delta_I : ndarray (K, K) — information gain from TOF
    F_joint : ndarray (K, K) — full joint Fisher info (matched tracks only)
    """
    F_joint = F_TPC + F_TOF
    F_eff = eps_match * F_joint + (1.0 - eps_match) * F_TPC
    delta_I = eps_match * F_TOF
    return F_eff, delta_I, F_joint
