"""
bethe_bloch.py — Parametric Bethe-Bloch Templates for Heavy-Ion PID
=====================================================================

Extends the fixed histogram templates in templates.py with analytically
computed dE/dx templates parameterized by momentum, species mass, and
detector resolution. Uses the 5-parameter ALEPH formula as implemented
in AliRoot (AliExternalTrackParam::BetheBlochAleph).

Functions:
  - bethe_bloch()       — ALEPH 5-parameter dE/dx
  - make_tpc_template() — Gaussian TPC template at (p, mass, sigma)
  - separation_power()  — n_sigma between two species
  - find_crossings()    — exact dE/dx crossing momenta
  - make_bin_edges()    — adaptive bin edges covering all species
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# =====================================================================
#  PHYSICS CONSTANTS (PDG 2024)
# =====================================================================

# Particle masses in GeV/c^2
MASS = {
    'pi': 0.13957039,       # +/- 0.00000018
    'K':  0.493677,          # +/- 0.000015 (C6)
    'p':  0.93827208816,     # +/- 0.00000000029
}

# ALEPH Bethe-Bloch parameters
# Source: AliRoot, AliExternalTrackParam::BetheBlochAleph
ALEPH_PARAMS = {
    'P1': 0.0761760,
    'P2': 10.632,
    'P3': 1.3279e-5,
    'P4': 1.8631,
    'P5': 1.9479,
}


# =====================================================================
#  6.1  BETHE-BLOCH
# =====================================================================

def bethe_bloch(p, mass, params=None):
    """ALEPH 5-parameter Bethe-Bloch dE/dx.

    f(βγ) = (P₁/β^P₄) × [P₂ − β^P₄ − ln(P₃ + 1/(βγ)^P₅)]

    Parameters
    ----------
    p : float or ndarray — momentum in GeV/c
    mass : float — particle mass in GeV/c²
    params : dict or None — ALEPH parameters (defaults to ALEPH_PARAMS)

    Returns
    -------
    dEdx : float or ndarray — expected dE/dx in MIP units
    """
    if params is None:
        params = ALEPH_PARAMS
    P1, P2, P3, P4, P5 = (params[f'P{i}'] for i in range(1, 6))

    p = np.asarray(p, dtype=float)
    bg = p / mass                           # βγ
    beta = bg / np.sqrt(1.0 + bg**2)       # β

    dEdx = (P1 / np.power(beta, P4)) * (
        P2 - np.power(beta, P4) - np.log(P3 + np.power(1.0 / bg, P5))
    )

    return np.maximum(dEdx, 0.0)


# =====================================================================
#  6.2  MAKE TEMPLATE (TPC)
# =====================================================================

def make_tpc_template(p, mass, sigma, bin_edges, params=None):
    """Construct Gaussian TPC dE/dx template at given momentum.

    Template is the probability per bin for a particle of given mass
    at momentum p, with fractional dE/dx resolution sigma.

    Parameters
    ----------
    p : float — momentum in GeV/c
    mass : float — particle mass in GeV/c²
    sigma : float — fractional dE/dx resolution (e.g. 0.05 for 5%)
    bin_edges : ndarray (B+1,) — dE/dx bin edges in MIP units
    params : dict or None — ALEPH parameters

    Returns
    -------
    template : ndarray (B,) — probability per bin, sums to 1
    """
    mu = float(bethe_bloch(p, mass, params))
    sigma_abs = sigma * mu

    if sigma_abs < 1e-15:
        template = np.zeros(len(bin_edges) - 1)
        idx = np.searchsorted(bin_edges, mu) - 1
        idx = np.clip(idx, 0, len(template) - 1)
        template[idx] = 1.0
        return template

    cdf_vals = norm.cdf(bin_edges, loc=mu, scale=sigma_abs)
    template = np.diff(cdf_vals)

    total = template.sum()
    if total > 1e-30:
        template /= total
    else:
        template[:] = 1.0 / len(template)

    return template


def make_bin_edges(p, masses, sigma, n_bins=100, n_sigma_range=5):
    """Create dE/dx bin edges covering all species at momentum p.

    Parameters
    ----------
    p : float — momentum in GeV/c
    masses : list of float — species masses
    sigma : float — fractional resolution
    n_bins : int
    n_sigma_range : float — how many sigma to extend beyond peaks

    Returns
    -------
    bin_edges : ndarray (n_bins+1,)
    """
    dedx_vals = [float(bethe_bloch(p, m)) for m in masses]
    mu_min = min(dedx_vals)
    mu_max = max(dedx_vals)
    spread = max(sigma * mu_max, 0.1) * n_sigma_range

    lo = max(mu_min - spread, 0.01)
    hi = mu_max + spread
    return np.linspace(lo, hi, n_bins + 1)


# =====================================================================
#  SEPARATION POWER AND CROSSINGS
# =====================================================================

def separation_power(p, mass1, mass2, sigma, params=None):
    """Separation power n_sigma between two species at momentum p.

    n_sigma = |dE/dx_1 - dE/dx_2| / sigma_avg

    Parameters
    ----------
    p : float or ndarray
    mass1, mass2 : float — species masses in GeV/c²
    sigma : float — fractional resolution

    Returns
    -------
    nsig : float or ndarray
    """
    dedx1 = bethe_bloch(p, mass1, params)
    dedx2 = bethe_bloch(p, mass2, params)
    sigma_abs = sigma * 0.5 * (dedx1 + dedx2)
    return np.where(sigma_abs > 0, np.abs(dedx1 - dedx2) / sigma_abs, 0.0)


def find_crossings(mass1, mass2, p_lo=0.2, p_hi=5.0, params=None):
    """Find exact dE/dx crossing momenta between two species.

    Parameters
    ----------
    mass1, mass2 : float — species masses in GeV/c²
    p_lo, p_hi : float — momentum search range
    params : dict or None

    Returns
    -------
    crossings : list of float — crossing momenta in GeV/c
    """
    def diff(p):
        return float(bethe_bloch(p, mass1, params) - bethe_bloch(p, mass2, params))

    p_scan = np.linspace(p_lo, p_hi, 2000)
    d = np.array([diff(p) for p in p_scan])
    crossings = []
    for i in range(len(d) - 1):
        if d[i] * d[i + 1] < 0:
            crossings.append(brentq(diff, p_scan[i], p_scan[i + 1]))
    return crossings


def build_template_matrix(p, masses, sigma, n_bins=100, params=None):
    """Build the full template matrix A at momentum p.

    Parameters
    ----------
    p : float — momentum
    masses : list of float — species masses
    sigma : float — fractional resolution
    n_bins : int
    params : dict or None

    Returns
    -------
    A : ndarray (n_bins, K) — template matrix
    bin_edges : ndarray (n_bins+1,)
    """
    K = len(masses)
    bin_edges = make_bin_edges(p, masses, sigma, n_bins)
    A = np.zeros((n_bins, K))
    for k in range(K):
        A[:, k] = make_tpc_template(p, masses[k], sigma, bin_edges, params)
    return A, bin_edges
