"""
templates.py — Visible-mass tau decay channel templates (v0.5.0)
================================================================

Constructs 1D visible-mass templates for the 7 tau decay channels.
Each template is a normalized PDF over visible invariant mass,
convolved with detector resolution.

Observable: visible invariant mass m_vis of all non-neutrino decay products.
  - Single-particle channels (e, mu, pi): m_vis = rest mass (delta function)
  - Multi-hadron channels (rho, a1, pi2pi0, other): m_vis = invariant mass
    of the hadronic system (Breit-Wigner resonance shape)

Detector model: uniform Gaussian resolution sigma_det = 15 MeV (benchmark
parameter, motivated by typical hadronic invariant mass resolution at
B-factory experiments; Belle II: sigma ~ 3-5 MeV for charged di-pion,
sigma ~ 10-20 MeV for multi-hadron channels including pi0 reconstruction).

Channels (PDG 2024):
  k=0: tau->e nu nubar    BR=17.82%  (delta at m_e = 0.511 MeV)
  k=1: tau->mu nu nubar   BR=17.39%  (delta at m_mu = 105.658 MeV)
  k=2: tau->pi nu         BR=10.82%  (delta at m_pi = 139.570 MeV)
  k=3: tau->rho nu        BR=25.49%  (BW: m_rho=775, Gamma=149 MeV)
  k=4: tau->a1 nu         BR= 8.99%  (BW: m_a1=1230, Gamma=420 MeV)
  k=5: tau->pi 2pi0 nu    BR= 9.26%  (BW: m~1050, Gamma~300 MeV)
  k=6: Other              BR=10.23%  (broad: m~1400, Gamma~500 MeV)

Changes from v0.4.9 (templates_legacy.py):
  - Leptonic channels use delta functions at rest mass instead of
    Michel energy spectra.
  - Default resolution: 15 MeV (was 20 MeV).
  - Default mass range: 0-2000 MeV (was 0-1800 MeV).
  - a1 mass: 1230 MeV (PDG 2024).
  - pi2pi0 template: distinct BW (m=1050, Gamma=300 MeV).
"""

import numpy as np


# === Physical constants (MeV) ===
M_TAU = 1776.93  # PDG 2024
M_E = 0.511
M_MU = 105.658
M_PI = 139.570
M_PI0 = 134.977
M_RHO = 775.11   # PDG 2024
G_RHO = 149.1
M_A1 = 1230.0    # PDG 2024
G_A1 = 420.0
M_PI2PI0 = 1050.0  # tau->pi- pi0 pi0 nu visible mass model
G_PI2PI0 = 300.0
M_OTHER = 1400.0   # broad multi-hadron
G_OTHER = 500.0

# Physical mass thresholds for hadronic channels
THRESH_RHO = M_PI + M_PI0       # pi+ pi0: ~274.5 MeV
THRESH_A1 = 2 * M_PI + M_PI0    # pi+ pi- pi0: ~414.1 MeV
THRESH_PI2PI0 = M_PI + 2 * M_PI0  # pi- pi0 pi0: ~409.5 MeV
THRESH_OTHER = 4 * M_PI           # multi-hadron: ~558.3 MeV

# PDG 2024 branching ratios
TAU_BR = np.array([0.1782, 0.1739, 0.1082, 0.2549, 0.0899, 0.0926, 0.1023])
TAU_LABELS = [
    r'$\tau\to e\nu\bar\nu$',
    r'$\tau\to \mu\nu\bar\nu$',
    r'$\tau\to \pi\nu$',
    r'$\tau\to \rho\nu$',
    r'$\tau\to a_1\nu$',
    r'$\tau\to \pi 2\pi^0\nu$',
    r'Other',
]
TAU_SHORT_LABELS = ['e', '\u03bc', '\u03c0', '\u03c1', 'a\u2081', '\u03c02\u03c0\u2070', 'other']


def delta_template(m_bins, m0, sigma_det=15.0):
    """Gaussian-smeared delta function at mass m0.

    Models the visible mass of a single identified particle: m_vis = m_rest.

    Parameters
    ----------
    m_bins : ndarray -- bin centers in MeV
    m0 : float -- particle rest mass in MeV
    sigma_det : float -- detector resolution in MeV

    Returns
    -------
    template : ndarray -- normalized template (sums to 1)
    """
    g = np.exp(-0.5 * ((m_bins - m0) / sigma_det)**2)
    total = np.sum(g)
    return g / total if total > 0 else g


def breit_wigner_smeared(m_bins, m0, gamma, sigma_det=15.0, threshold=None):
    """Breit-Wigner resonance convolved with Gaussian detector resolution.

    Models the visible invariant mass of a multi-hadron system produced
    through an intermediate resonance (e.g., rho->pipi, a1->3pi).

    Parameters
    ----------
    m_bins : ndarray -- bin centers in MeV
    m0 : float -- resonance mass in MeV
    gamma : float -- resonance width in MeV
    sigma_det : float -- detector resolution in MeV
    threshold : float or None -- physical mass threshold in MeV.
        If given, the BW is set to zero below this mass before smearing.

    Returns
    -------
    template : ndarray -- normalized template (sums to 1)
    """
    fine = np.linspace(0, 2500.0, 10000)
    df = fine[1] - fine[0]

    # Relativistic Breit-Wigner in s = m^2, converted to mass density
    # BW(s) = m0*Γ / ((s-s0)^2 + s0*Γ^2), with Jacobian ds = 2m dm
    s = fine**2
    s0 = m0**2
    bw = m0 * gamma / ((s - s0)**2 + s0 * gamma**2)
    bw = bw * 2 * fine  # Jacobian: density in m, not s

    # Apply physical threshold
    if threshold is not None:
        bw[fine < threshold] = 0.0

    bw_norm = bw / (np.sum(bw) * df) if np.sum(bw) > 0 else bw

    # Convolve with Gaussian resolution
    result = np.zeros(len(m_bins))
    for j in range(len(m_bins)):
        kern = np.exp(-0.5 * ((fine - m_bins[j]) / sigma_det)**2)
        result[j] = np.sum(bw_norm * kern) * df

    total = np.sum(result)
    return result / total if total > 0 else result


# Keep legacy functions accessible for backward compatibility
def bw_template(m_bins, m0, gamma, sigma_det=15.0, threshold=None):
    """Alias for breit_wigner_smeared (backward compatibility)."""
    return breit_wigner_smeared(m_bins, m0, gamma, sigma_det, threshold=threshold)


class TauTemplates:
    """Generate the 7-channel tau decay visible-mass template matrix.

    Uses physically correct observable: visible invariant mass.
    Single-particle channels (e, mu, pi) produce delta functions at rest mass.
    Multi-hadron channels produce Breit-Wigner resonance shapes.

    Parameters
    ----------
    M : int -- number of bins
    m_range : tuple -- (m_min, m_max) in MeV
    sigma_det : float -- detector mass resolution in MeV
        Default 15.0 MeV: Belle II-motivated benchmark parameter.
    """

    def __init__(self, M=200, m_range=(0.0, 2000.0), sigma_det=15.0):
        self.M = M
        self.m_range = m_range
        self.sigma_det = sigma_det
        self.K = 7
        self.BR = TAU_BR.copy()
        self.labels = TAU_LABELS
        self.short_labels = TAU_SHORT_LABELS

        dm = (m_range[1] - m_range[0]) / M
        self.m_bins = np.linspace(m_range[0] + 0.5 * dm,
                                   m_range[1] - 0.5 * dm,
                                   M)
        self.dm = dm
        self._A = None

    @property
    def A(self):
        """Template matrix A in R^{MxK}, each column normalized to sum to 1."""
        if self._A is None:
            self._A = self._build_templates()
        return self._A

    def _build_templates(self):
        m = self.m_bins
        sd = self.sigma_det
        A = np.zeros((self.M, self.K))

        # Single-particle channels: delta at rest mass, smeared
        A[:, 0] = delta_template(m, M_E, sigma_det=sd)       # tau->e nu nubar
        A[:, 1] = delta_template(m, M_MU, sigma_det=sd)      # tau->mu nu nubar
        A[:, 2] = delta_template(m, M_PI, sigma_det=sd)       # tau->pi nu

        # Multi-hadron channels: Breit-Wigner invariant mass, smeared
        # Physical thresholds enforced to remove sub-threshold support.
        A[:, 3] = breit_wigner_smeared(m, M_RHO, G_RHO, sigma_det=sd,
                                        threshold=THRESH_RHO)        # rho->pipi
        A[:, 4] = breit_wigner_smeared(m, M_A1, G_A1, sigma_det=sd,
                                        threshold=THRESH_A1)          # a1->3pi
        A[:, 5] = breit_wigner_smeared(m, M_PI2PI0, G_PI2PI0, sigma_det=sd,
                                        threshold=THRESH_PI2PI0)      # pi-pi0pi0
        A[:, 6] = breit_wigner_smeared(m, M_OTHER, G_OTHER, sigma_det=sd,
                                        threshold=THRESH_OTHER)        # other

        # Ensure non-negative and normalized
        A = np.maximum(A, 0.0)
        for k in range(self.K):
            total = np.sum(A[:, k])
            if total > 0:
                A[:, k] /= total

        return A

    def rebuild(self, M=None, sigma_det=None):
        """Rebuild templates with new parameters."""
        if M is not None:
            self.M = M
            dm = (self.m_range[1] - self.m_range[0]) / M
            self.m_bins = np.linspace(
                self.m_range[0] + 0.5 * dm,
                self.m_range[1] - 0.5 * dm,
                M)
            self.dm = dm
        if sigma_det is not None:
            self.sigma_det = sigma_det
        self._A = None
        return self

    def spreading_factors(self):
        """Compute spreading factor SF = m/Gamma for each resonance channel."""
        return {
            0: None,                   # e: single particle
            1: None,                   # mu: single particle
            2: float('inf'),           # pi: stable
            3: M_RHO / G_RHO,         # rho: 5.2
            4: M_A1 / G_A1,           # a1: 2.9
            5: M_PI2PI0 / G_PI2PI0,   # pi2pi0: 3.5
            6: M_OTHER / G_OTHER,      # other: 2.8
        }
