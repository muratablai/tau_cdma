"""
spreading.py — Spreading Factor and Optimal Binning
=====================================================

Implements:
  - SF = m/Γ computation
  - Processing gain in dB
  - Optimal binning M_opt ∝ SF_k
"""

import numpy as np


def spreading_factor(mass, width):
    """Compute spreading factor SF = m/Γ.

    Parameters
    ----------
    mass : float — particle mass (MeV)
    width : float — total width (MeV)

    Returns
    -------
    SF : float (inf if width == 0, i.e., stable particle)
    """
    if width <= 0:
        return float('inf')
    return mass / width


def processing_gain_dB(sf):
    """Processing gain in dB: PG = 10 · log₁₀(SF)."""
    if sf <= 0 or np.isinf(sf):
        return float('inf') if sf > 0 else 0.0
    return 10.0 * np.log10(sf)


def optimal_binning(m_range_width, width_k):
    """Optimal number of bins for channel k.

    M_opt,k ≈ m_range / Γ_k

    Parameters
    ----------
    m_range_width : float — total mass range (MeV)
    width_k : float — natural width of channel k (MeV)

    Returns
    -------
    M_opt : float
    """
    if width_k <= 0:
        return float('inf')
    return m_range_width / width_k


# Standard particle table for reference
PARTICLE_TABLE = {
    'J/psi':  {'mass': 3096.9, 'width': 0.0929, 'SF': 33334.},
    'Z':      {'mass': 91188., 'width': 2495.,  'SF': 36.5},
    'W':      {'mass': 80369., 'width': 2085.,  'SF': 38.5},
    'rho':    {'mass': 775.26, 'width': 149.1,  'SF': 5.2},
    'f0_500': {'mass': 500.,   'width': 500.,   'SF': 1.0},
    'a1':     {'mass': 1230.,  'width': 420.,   'SF': 2.9},
    'top':    {'mass': 172570.,'width': 1420.,  'SF': 121.5},
}
