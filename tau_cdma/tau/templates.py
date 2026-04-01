"""
templates.py — Analytic τ decay channel templates
==================================================

Constructs 1D (m_vis) and optionally 2D templates for the 7 τ decay channels.
Each template is a normalized PDF over visible mass, convolved with detector
resolution where appropriate.

Channels (PDG 2024):
  k=0: τ→eνν̄    BR=0.178   (Michel spectrum)
  k=1: τ→μνν̄    BR=0.174   (Michel spectrum)
  k=2: τ→πν     BR=0.108   (delta at m_π, smeared)
  k=3: τ→ρν     BR=0.255   (Breit-Wigner, m=775, Γ=149 MeV)
  k=4: τ→a₁ν    BR=0.090   (Breit-Wigner, m=1230, Γ=420 MeV)
  k=5: τ→π2π⁰ν  BR=0.093   (similar to a₁ but shifted)
  k=6: Other     BR=0.102   (broad empirical)
"""

import numpy as np
from scipy.special import voigt_profile
from scipy.stats import norm


# === Physical constants (MeV) ===
M_TAU = 1776.93  # PDG 2024 (was 1776.86 pre-2024)
M_E = 0.511
M_MU = 105.658
M_PI = 139.570
M_PI0 = 134.977
M_RHO = 775.11    # C5: consistent charged ρ± value (PDG 2024)
G_RHO = 149.1
M_A1 = 1230.0
G_A1 = 420.0

# PDG 2024 branching ratios (C4 applied to π2π⁰ channel)
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
TAU_SHORT_LABELS = ['e', 'μ', 'π', 'ρ', 'a₁', 'π2π⁰', 'other']


def breit_wigner(m, m0, gamma):
    """Relativistic Breit-Wigner PDF (unnormalized).

    BW(m) = m0 * gamma / ((m² - m0²)² + m0² * gamma²)

    Parameters
    ----------
    m : ndarray — mass values in MeV
    m0 : float — resonance mass in MeV
    gamma : float — resonance width in MeV

    Returns
    -------
    bw : ndarray — unnormalized Breit-Wigner values
    """
    s = m**2
    s0 = m0**2
    return m0 * gamma / ((s - s0)**2 + s0 * gamma**2)


def michel_spectrum(x, m_lepton, rho=0.75, eta=0.0):
    """Michel spectrum for τ→ℓνν̄.

    Parameters
    ----------
    x : array — E_ℓ / E_max (scaled lepton energy, 0 to 1)
    m_lepton : float — lepton mass in MeV
    rho : float — Michel parameter (SM: 3/4)
    eta : float — eta parameter

    Returns
    -------
    dGamma/dx (unnormalized)
    """
    r = (m_lepton / M_TAU)**2
    x = np.clip(x, 1e-10, 1.0 - 1e-10)
    # Standard Michel spectrum
    spectrum = x**2 * (3 - 2*x) + rho * x**2 * (4*x/3 - 1)
    # eta term (small for e, relevant for μ)
    if eta != 0 and m_lepton > 1.0:
        spectrum += eta * (m_lepton / M_TAU) * (1 - x)
    return np.where(x > 0, spectrum, 0.0)


def michel_to_mvis(m_bins, m_lepton, sigma_det=20.0):
    """Convert Michel spectrum to visible mass distribution.

    For leptonic τ decays, the visible mass is approximately the lepton
    energy (since m_vis ≈ E_ℓ for a single track). The energy spectrum
    maps to an m_vis distribution via kinematics.

    Parameters
    ----------
    m_bins : ndarray — mass bin centers in MeV
    m_lepton : float — lepton mass in MeV (e: 0.511, μ: 105.66)
    sigma_det : float — detector resolution in MeV (default 20)

    Returns
    -------
    template : ndarray — normalized visible mass distribution
    """
    # m_vis for single lepton: effectively the lepton energy in τ rest frame
    # E_max ≈ m_τ/2 (when both neutrinos go opposite)
    E_max = M_TAU / 2.0
    dm = m_bins[1] - m_bins[0] if len(m_bins) > 1 else 1.0

    # Evaluate Michel spectrum over bin range
    x_vals = np.clip(m_bins / E_max, 0.01, 0.99)
    template = michel_spectrum(x_vals, m_lepton)
    template[m_bins < m_lepton] = 0.0
    template[m_bins > E_max] = 0.0

    # Gaussian smearing
    if sigma_det > 0:
        template = _gaussian_smear(template, m_bins, sigma_det)

    # Normalize
    total = np.sum(template) * dm
    if total > 0:
        template /= total
    return template


def bw_template(m_bins, m0, gamma, sigma_det=20.0):
    """Breit-Wigner resonance template convolved with detector resolution.

    Uses Voigt profile when sigma_det > 0.

    Parameters
    ----------
    m_bins : ndarray — mass bin centers in MeV
    m0 : float — resonance mass in MeV
    gamma : float — resonance width in MeV
    sigma_det : float — detector resolution in MeV (default 20)

    Returns
    -------
    template : ndarray — normalized template distribution
    """
    dm = m_bins[1] - m_bins[0] if len(m_bins) > 1 else 1.0

    if sigma_det > 0:
        # Voigt profile: convolution of Lorentzian (BW) with Gaussian (detector)
        # voigt_profile(x, sigma, gamma) where x = (m - m0), sigma = Gaussian σ,
        # gamma = Lorentzian half-width = Γ/2
        template = voigt_profile(m_bins - m0, sigma_det, gamma / 2.0)
    else:
        template = breit_wigner(m_bins, m0, gamma)

    # Restrict to physical range
    template[m_bins < 2 * M_PI] = 0.0  # below 2-pion threshold for hadronic
    template[m_bins > M_TAU] = 0.0

    total = np.sum(template) * dm
    if total > 0:
        template /= total
    return template


def delta_template(m_bins, m0, sigma_det=20.0):
    """Delta-function template (e.g., τ→πν) smeared by detector resolution.

    Parameters
    ----------
    m_bins : ndarray — mass bin centers in MeV
    m0 : float — particle mass in MeV
    sigma_det : float — detector resolution in MeV (default 20)

    Returns
    -------
    template : ndarray — normalized template distribution
    """
    dm = m_bins[1] - m_bins[0] if len(m_bins) > 1 else 1.0
    if sigma_det > 0:
        template = norm.pdf(m_bins, loc=m0, scale=sigma_det)
    else:
        template = np.zeros_like(m_bins)
        idx = np.argmin(np.abs(m_bins - m0))
        template[idx] = 1.0 / dm
    template[m_bins > M_TAU] = 0.0
    total = np.sum(template) * dm
    if total > 0:
        template /= total
    return template


def other_template(m_bins, sigma_det=20.0):
    """Broad empirical template for 'Other' τ decay modes.

    Models the sum of many small modes as a broad distribution.

    Parameters
    ----------
    m_bins : ndarray — mass bin centers in MeV
    sigma_det : float — detector resolution in MeV (default 20)

    Returns
    -------
    template : ndarray — normalized template distribution
    """
    dm = m_bins[1] - m_bins[0] if len(m_bins) > 1 else 1.0
    # Broad distribution peaking around 0.8 GeV with large width
    template = bw_template(m_bins, 800.0, 500.0, sigma_det=sigma_det)
    # Add some weight at high mass (multi-hadron modes)
    high_mass = bw_template(m_bins, 1400.0, 400.0, sigma_det=sigma_det)
    template = 0.6 * template + 0.4 * high_mass
    total = np.sum(template) * dm
    if total > 0:
        template /= total
    return template


def _gaussian_smear(template, m_bins, sigma):
    """Convolve template with Gaussian resolution using FFT."""
    dm = m_bins[1] - m_bins[0]
    n = len(m_bins)
    # Create Gaussian kernel
    kernel_x = np.arange(-n//2, n//2 + 1) * dm
    kernel = norm.pdf(kernel_x, 0, sigma)
    kernel /= np.sum(kernel)
    # Convolve via FFT
    from scipy.signal import fftconvolve
    result = fftconvolve(template, kernel, mode='same')
    return np.maximum(result, 0.0)


def voigt_fwhm(gamma_L, sigma_G):
    """Voigt profile FWHM approximation (Thompson et al. 1987).

    Parameters
    ----------
    gamma_L : float — Lorentzian FWHM (sum of BW widths)
    sigma_G : float — Gaussian FWHM (quadrature sum of resolutions)

    Returns
    -------
    Approximate FWHM of the Voigt profile
    """
    return 0.5346 * gamma_L + np.sqrt(0.2166 * gamma_L**2 + sigma_G**2)


class TauTemplates:
    """Generate the 7-channel τ decay template matrix.

    Parameters
    ----------
    M : int — number of bins
    m_range : tuple — (m_min, m_max) in MeV
    sigma_det : float — detector mass resolution in MeV
    """

    def __init__(self, M=200, m_range=(0.0, 1800.0), sigma_det=20.0):
        self.M = M
        self.m_range = m_range
        self.sigma_det = sigma_det
        self.K = 7
        self.BR = TAU_BR.copy()
        self.labels = TAU_LABELS
        self.short_labels = TAU_SHORT_LABELS

        # Bin centers
        self.m_bins = np.linspace(m_range[0] + 0.5 * (m_range[1] - m_range[0]) / M,
                                   m_range[1] - 0.5 * (m_range[1] - m_range[0]) / M,
                                   M)
        self.dm = self.m_bins[1] - self.m_bins[0]
        self._A = None

    @property
    def A(self):
        """Template matrix A ∈ ℝ^{M×K}, each column normalized to sum to 1."""
        if self._A is None:
            self._A = self._build_templates()
        return self._A

    def _build_templates(self):
        m = self.m_bins
        sd = self.sigma_det
        A = np.zeros((self.M, self.K))

        # k=0: τ→eνν̄ (Michel spectrum)
        A[:, 0] = michel_to_mvis(m, M_E, sigma_det=sd)

        # k=1: τ→μνν̄ (Michel spectrum, shifted by μ mass)
        A[:, 1] = michel_to_mvis(m, M_MU, sigma_det=sd)

        # k=2: τ→πν (delta at m_π)
        A[:, 2] = delta_template(m, M_PI, sigma_det=sd)

        # k=3: τ→ρν (BW at m_ρ, Γ_ρ)
        A[:, 3] = bw_template(m, M_RHO, G_RHO, sigma_det=sd)

        # k=4: τ→a₁ν (BW at m_a₁, Γ_a₁)
        A[:, 4] = bw_template(m, M_A1, G_A1, sigma_det=sd)

        # k=5: τ→π2π⁰ν (similar to a₁ but only charged π visible → lower m_vis)
        # The visible mass is m_π (charged pion) smeared, since 2π⁰ are neutral
        # In practice, if π⁰→γγ are reconstructed, this looks like a₁
        # Here: model as BW shifted lower + broader
        A[:, 5] = bw_template(m, 1100.0, 450.0, sigma_det=sd)

        # k=6: Other
        A[:, 6] = other_template(m, sigma_det=sd)

        # Ensure non-negative and normalized (probability per bin, Σ_i a_ki = 1)
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
            self.m_bins = np.linspace(
                self.m_range[0] + 0.5 * (self.m_range[1] - self.m_range[0]) / M,
                self.m_range[1] - 0.5 * (self.m_range[1] - self.m_range[0]) / M,
                M)
            self.dm = self.m_bins[1] - self.m_bins[0]
        if sigma_det is not None:
            self.sigma_det = sigma_det
        self._A = None
        return self

    def spreading_factors(self):
        """Compute spreading factor SF = m/Γ for each resonance channel.

        Returns dict mapping channel index to SF (None for non-resonance channels).
        """
        return {
            0: None,           # 3-body leptonic
            1: None,           # 3-body leptonic
            2: float('inf'),   # stable π
            3: M_RHO / G_RHO,  # ρ: 5.2
            4: M_A1 / G_A1,    # a₁: 2.9
            5: 1100.0 / 450.0, # π2π⁰: ~2.4
            6: None,           # mixed
        }
