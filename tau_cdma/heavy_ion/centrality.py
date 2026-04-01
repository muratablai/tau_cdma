"""
centrality.py — Centrality-Dependent Analysis for ALICE Pb-Pb
===============================================================

Centrality simultaneously changes four inputs to the Fisher information:
  N (multiplicity), σ (resolution), θ (species fractions), templates (flow).

This module provides:
  - ALICE Pb-Pb centrality configurations (Table 2 from Document 1)
  - momentum_sweep()    — F(p), η(p), CRB(p) over momentum grid
  - centrality_sweep()  — η_k(p,c) measurability landscape

Sources:
  PRL 116, 222302 (2016) — multiplicity
  PRC 101, 044907 (2020) — species ratios
  Eur. Phys. J. Plus 131, 168 (2016) — resolution
"""

import numpy as np

from tau_cdma.core.fisher import poisson_fim, constrained_crb, crb_multiuser_efficiency
from tau_cdma.heavy_ion.bethe_bloch import (
    MASS, make_tpc_template, make_bin_edges, bethe_bloch, build_template_matrix
)
from tau_cdma.heavy_ion.tof import (
    build_tof_template_matrix, joint_fisher,
    TOF_RESOLUTION_PS, TOF_PATH_LENGTH_M, TOF_MATCH_EFF
)


# =====================================================================
#  CENTRALITY CONFIGURATIONS
# =====================================================================

# ALICE Pb-Pb at sqrt(s_NN) = 5.02 TeV
# (label, dNch/deta, sigma_dEdx, K/pi, p/pi)
CENTRALITY_CONFIGS = [
    ('0-5%',   1943, 0.070, 0.170, 0.046),
    ('5-10%',  1585, 0.065, 0.165, 0.045),
    ('10-20%', 1180, 0.060, 0.155, 0.043),
    ('20-30%',  786, 0.058, 0.145, 0.042),
    ('30-40%',  512, 0.055, 0.140, 0.040),
    ('40-50%',  318, 0.053, 0.135, 0.038),
    ('50-60%',  183, 0.052, 0.130, 0.037),
    ('60-70%',   96, 0.051, 0.130, 0.036),
    ('70-80%',   44, 0.050, 0.130, 0.035),
]


def fractions_from_ratios(k_pi, p_pi):
    """Convert K/pi and p/pi ratios to species fractions (pi, K, p).

    Parameters
    ----------
    k_pi : float — K/pi ratio
    p_pi : float — p/pi ratio

    Returns
    -------
    theta : ndarray (3,) — [pi, K, p] fractions summing to 1
    """
    denom = 1.0 + k_pi + p_pi
    return np.array([1.0 / denom, k_pi / denom, p_pi / denom])


# =====================================================================
#  6.6  MOMENTUM SWEEP
# =====================================================================

def momentum_sweep(masses=None, sigma=0.05, theta=None, N=10000,
                   p_grid=None, background_frac=0.001, n_bins=100,
                   species_names=None, compute_tof=False,
                   sigma_t=TOF_RESOLUTION_PS, L=TOF_PATH_LENGTH_M,
                   eps_match=TOF_MATCH_EFF):
    """Sweep over momentum: compute F, eta, CRB at each p.

    Parameters
    ----------
    masses : list of float or None — species masses in GeV
    sigma : float — fractional dE/dx resolution
    theta : ndarray or None — species fractions
    N : float — total event count
    p_grid : ndarray or None — momentum values in GeV/c
    background_frac : float — background as fraction of signal
    n_bins : int — number of dE/dx bins
    species_names : list of str or None
    compute_tof : bool — also compute TOF and joint Fisher info
    sigma_t, L, eps_match : TOF parameters

    Returns
    -------
    results : dict
    """
    if masses is None:
        masses = [MASS['pi'], MASS['K'], MASS['p']]
    if species_names is None:
        species_names = ['pi', 'K', 'p']
    K = len(masses)

    if theta is None:
        theta = fractions_from_ratios(0.14, 0.04)

    if p_grid is None:
        p_grid = np.linspace(0.2, 5.0, 200)

    C = np.ones((1, K))

    n_p = len(p_grid)
    F_arr = np.zeros((n_p, K, K))
    CRB_arr = np.zeros((n_p, K))
    CRBc_arr = np.zeros((n_p, K))
    eta_arr = np.zeros((n_p, K))
    etac_arr = np.zeros((n_p, K))
    eig_arr = np.zeros((n_p, K))
    sep_power = np.zeros((n_p, K, K))

    if compute_tof:
        F_TOF_arr = np.zeros((n_p, K, K))
        F_joint_arr = np.zeros((n_p, K, K))
        F_eff_arr = np.zeros((n_p, K, K))
        delta_I_arr = np.zeros((n_p, K, K))
        eta_joint_arr = np.zeros((n_p, K))
        eta_eff_arr = np.zeros((n_p, K))

    for ip, p in enumerate(p_grid):
        # --- TPC ---
        A, bin_edges = build_template_matrix(p, masses, sigma, n_bins)
        B = A.shape[0]

        # Separation power
        dedx_vals = np.array([float(bethe_bloch(p, m)) for m in masses])
        for j in range(K):
            for k in range(j + 1, K):
                sigma_abs = sigma * 0.5 * (dedx_vals[j] + dedx_vals[k])
                if sigma_abs > 0:
                    sep_power[ip, j, k] = abs(dedx_vals[j] - dedx_vals[k]) / sigma_abs
                    sep_power[ip, k, j] = sep_power[ip, j, k]

        # Background
        bg = background_frac * N * np.ones(B) / B

        # Fisher information
        F = poisson_fim(A, theta, N, bg)
        F_arr[ip] = F

        # Eigenvalues
        eig_arr[ip] = np.sort(np.linalg.eigvalsh(F))

        # CRB (unconstrained)
        try:
            Finv = np.linalg.inv(F + 1e-15 * np.eye(K))
            CRB_arr[ip] = np.diag(Finv)
        except np.linalg.LinAlgError:
            CRB_arr[ip] = np.inf

        # Constrained CRB
        CRB_c = constrained_crb(F, C)
        CRBc_arr[ip] = np.diag(CRB_c)

        # CRB-based multiuser efficiency (C2)
        eta, eta_c = crb_multiuser_efficiency(F, CRB_c)
        eta_arr[ip] = eta
        etac_arr[ip] = eta_c if eta_c is not None else eta

        # --- TOF ---
        if compute_tof:
            A_tof, _ = build_tof_template_matrix(p, masses, n_bins, sigma_t, L)
            F_tof = poisson_fim(A_tof, theta, N, bg[:n_bins])
            F_TOF_arr[ip] = F_tof

            F_eff, dI, F_j = joint_fisher(F, F_tof, eps_match)
            F_joint_arr[ip] = F_j
            F_eff_arr[ip] = F_eff
            delta_I_arr[ip] = dI

            eta_j, _ = crb_multiuser_efficiency(F_j)
            eta_e, _ = crb_multiuser_efficiency(F_eff)
            eta_joint_arr[ip] = eta_j
            eta_eff_arr[ip] = eta_e

    results = {
        'p_grid': p_grid,
        'species_names': species_names,
        'masses': masses,
        'theta': theta,
        'sigma': sigma,
        'N': N,
        'F': F_arr,
        'CRB': CRB_arr,
        'CRB_c': CRBc_arr,
        'eta': eta_arr,
        'eta_c': etac_arr,
        'eigenvalues': eig_arr,
        'separation_power': sep_power,
    }

    if compute_tof:
        results.update({
            'F_TOF': F_TOF_arr,
            'F_joint': F_joint_arr,
            'F_eff': F_eff_arr,
            'delta_I': delta_I_arr,
            'eta_joint': eta_joint_arr,
            'eta_eff': eta_eff_arr,
        })

    return results


# =====================================================================
#  6.7  CENTRALITY SWEEP
# =====================================================================

def centrality_sweep(configs=None, p_grid=None, compute_tof=True,
                     n_bins=100, background_frac=0.001):
    """Sweep over centrality and momentum: measurability landscape.

    The primary output is η_k(p, c): multiuser efficiency as a function
    of momentum and centrality for each species.

    Parameters
    ----------
    configs : list of tuples or None — centrality configurations
    p_grid : ndarray or None
    compute_tof : bool
    n_bins : int
    background_frac : float

    Returns
    -------
    results : dict with arrays indexed [centrality, momentum, species]
    """
    if configs is None:
        configs = CENTRALITY_CONFIGS
    if p_grid is None:
        p_grid = np.linspace(0.2, 5.0, 200)

    masses = [MASS['pi'], MASS['K'], MASS['p']]
    K = len(masses)
    n_c = len(configs)
    n_p = len(p_grid)

    labels = []
    eta_all = np.zeros((n_c, n_p, K))
    etac_all = np.zeros((n_c, n_p, K))
    crb_all = np.zeros((n_c, n_p, K))
    crbc_all = np.zeros((n_c, n_p, K))
    eig_all = np.zeros((n_c, n_p, K))
    sep_all = np.zeros((n_c, n_p, K, K))

    if compute_tof:
        eta_joint_all = np.zeros((n_c, n_p, K))
        eta_eff_all = np.zeros((n_c, n_p, K))
        dI_trace_all = np.zeros((n_c, n_p))

    for ic, (label, dNch, sigma, k_pi, p_pi) in enumerate(configs):
        labels.append(label)
        theta = fractions_from_ratios(k_pi, p_pi)
        N = float(dNch)

        res = momentum_sweep(
            masses=masses, sigma=sigma, theta=theta, N=N,
            p_grid=p_grid, background_frac=background_frac,
            n_bins=n_bins, compute_tof=compute_tof
        )

        eta_all[ic] = res['eta']
        etac_all[ic] = res['eta_c']
        crb_all[ic] = res['CRB']
        crbc_all[ic] = res['CRB_c']
        eig_all[ic] = res['eigenvalues']
        sep_all[ic] = res['separation_power']

        if compute_tof:
            eta_joint_all[ic] = res['eta_joint']
            eta_eff_all[ic] = res['eta_eff']
            dI_trace_all[ic] = np.array([
                np.trace(res['delta_I'][i]) for i in range(n_p)
            ])

    output = {
        'centrality_labels': labels,
        'p_grid': p_grid,
        'species_names': ['pi', 'K', 'p'],
        'eta': eta_all,
        'eta_c': etac_all,
        'CRB': crb_all,
        'CRB_c': crbc_all,
        'eigenvalues': eig_all,
        'separation_power': sep_all,
    }

    if compute_tof:
        output.update({
            'eta_joint': eta_joint_all,
            'eta_eff': eta_eff_all,
            'delta_I_trace': dI_trace_all,
        })

    return output
