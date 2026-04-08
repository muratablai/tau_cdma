"""
tau_7channel.py — τ lepton 7-channel benchmark configuration
==============================================================

Primary validation target for the framework.
Tests predictions P1-P4, P7, P8.
"""

import numpy as np
from tau_cdma.tau.templates import TauTemplates, TAU_BR


def default_config():
    """Default benchmark configuration."""
    return {
        'M': 200,                    # bins
        'm_range': (0.0, 2000.0),    # MeV (was 1800 in v0.4.9)
        'sigma_det': 15.0,           # MeV detector resolution (Belle II-motivated)
        'N': 1_000_000,              # total events
        'background_density': 0.01,  # per MeV
        'theta': TAU_BR.copy(),
    }


def setup_benchmark(config=None):
    """Set up the τ benchmark with templates, FIM, etc.

    Returns
    -------
    bench : dict with all precomputed quantities
    """
    if config is None:
        config = default_config()

    from tau_cdma.core.fisher import poisson_fim, crb, eigenvalue_spectrum
    from tau_cdma.core.interference import interference_matrix, multiuser_efficiency
    from tau_cdma.core.spreading import spreading_factor

    tb = TauTemplates(
        M=config['M'],
        m_range=config['m_range'],
        sigma_det=config['sigma_det'],
    )

    A = tb.A
    theta = config['theta']
    N = config['N']
    M = config['M']
    dm = (config['m_range'][1] - config['m_range'][0]) / M
    background = config['background_density'] * dm * np.ones(M)

    # Fisher information
    F = poisson_fim(A, theta, N, background)
    crb_vals = crb(F, regularize=True)
    eigvals, eigvecs = eigenvalue_spectrum(F)

    # Interference matrix
    R = interference_matrix(A, theta, N, background)
    eta = multiuser_efficiency(R)

    return {
        'templates': tb,
        'A': A,
        'theta': theta,
        'N': N,
        'M': M,
        'background': background,
        'F': F,
        'crb': crb_vals,
        'eigvals': eigvals,
        'eigvecs': eigvecs,
        'R': R,
        'eta': eta,
        'config': config,
    }
