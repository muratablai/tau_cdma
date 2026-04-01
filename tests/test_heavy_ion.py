"""
test_heavy_ion.py — Consistency tests for heavy-ion PID extension
===================================================================

Verifies:
  - Bethe-Bloch produces physical dE/dx values
  - TPC templates are proper probability distributions
  - TOF templates are proper probability distributions
  - Constrained CRB <= unconstrained CRB
  - Joint Fisher >= TPC Fisher (information additivity)
  - Centrality configs have monotone multiplicity
  - BR sum conservation
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from tau_cdma.heavy_ion.bethe_bloch import (
    bethe_bloch, make_tpc_template, make_bin_edges,
    find_crossings, build_template_matrix, MASS, ALEPH_PARAMS
)
from tau_cdma.heavy_ion.tof import tof_template, build_tof_template_matrix, joint_fisher
from tau_cdma.core.fisher import poisson_fim, constrained_crb, crb_multiuser_efficiency
from tau_cdma.heavy_ion.centrality import (
    CENTRALITY_CONFIGS, fractions_from_ratios, momentum_sweep
)


def test_bethe_bloch_physical():
    """Bethe-Bloch should give positive dE/dx for all species at all momenta."""
    for name, mass in MASS.items():
        p_grid = np.linspace(0.2, 10.0, 100)
        dedx = bethe_bloch(p_grid, mass)
        assert np.all(dedx > 0), f"dE/dx < 0 for {name}"
        assert np.all(dedx < 100), f"dE/dx unreasonably large for {name}"
    print("  Bethe-Bloch physical values: pass")


def test_bethe_bloch_mass_ordering():
    """At low momentum, heavier particles have larger dE/dx."""
    p = 0.5  # GeV/c
    dedx_pi = float(bethe_bloch(p, MASS['pi']))
    dedx_K = float(bethe_bloch(p, MASS['K']))
    dedx_p = float(bethe_bloch(p, MASS['p']))
    assert dedx_p > dedx_K > dedx_pi, (
        f"Mass ordering violated: pi={dedx_pi:.3f}, K={dedx_K:.3f}, p={dedx_p:.3f}"
    )
    print("  Bethe-Bloch mass ordering: pass")


def test_crossings_exist():
    """pi/K and K/p crossings should exist in [0.2, 5.0] GeV/c."""
    piK = find_crossings(MASS['pi'], MASS['K'])
    Kp = find_crossings(MASS['K'], MASS['p'])
    assert len(piK) >= 1, "No pi/K crossing found"
    assert len(Kp) >= 1, "No K/p crossing found"
    assert 0.5 < piK[0] < 2.0, f"pi/K crossing at {piK[0]:.3f}, expected 0.5-2.0"
    assert 1.5 < Kp[0] < 4.0, f"K/p crossing at {Kp[0]:.3f}, expected 1.5-4.0"
    print(f"  Crossings: pi/K={piK[0]:.4f}, K/p={Kp[0]:.4f}: pass")


def test_tpc_template_normalization():
    """TPC templates should be proper probability distributions."""
    masses = list(MASS.values())
    for p in [0.3, 1.0, 2.0, 4.0]:
        A, _ = build_template_matrix(p, masses, 0.05, 100)
        for k in range(3):
            assert np.all(A[:, k] >= 0), f"Negative probability at p={p}, k={k}"
            total = np.sum(A[:, k])
            assert abs(total - 1.0) < 0.01, f"Sum={total} at p={p}, k={k}"
    print("  TPC template normalization: pass")


def test_tof_template_normalization():
    """TOF templates should be proper probability distributions."""
    masses = list(MASS.values())
    for p in [0.5, 1.0, 2.0]:
        A_tof, _ = build_tof_template_matrix(p, masses)
        for k in range(3):
            assert np.all(A_tof[:, k] >= 0), f"Negative TOF prob at p={p}, k={k}"
            total = np.sum(A_tof[:, k])
            assert abs(total - 1.0) < 0.01, f"TOF sum={total} at p={p}, k={k}"
    print("  TOF template normalization: pass")


def test_constrained_crb_tighter():
    """Constrained CRB should be <= unconstrained CRB (element-wise diagonal)."""
    masses = list(MASS.values())
    theta = fractions_from_ratios(0.14, 0.04)
    for p in [0.5, 1.0, 2.0]:
        A, _ = build_template_matrix(p, masses, 0.05, 100)
        N = 500
        bg = np.ones(100) * 0.001 * N / 100
        F = poisson_fim(A, theta, N, bg)
        CRB_c = constrained_crb(F)

        Finv = np.linalg.inv(F + 1e-15 * np.eye(3))
        CRB_unc = np.diag(Finv)
        CRB_con = np.diag(CRB_c)

        for k in range(3):
            assert CRB_con[k] <= CRB_unc[k] + 1e-10, (
                f"Constrained CRB > unconstrained at p={p}, k={k}: "
                f"{CRB_con[k]:.4e} > {CRB_unc[k]:.4e}"
            )
    print("  Constrained CRB <= unconstrained: pass")


def test_joint_fisher_additive():
    """Joint Fisher >= TPC Fisher (information never decreases)."""
    masses = list(MASS.values())
    theta = fractions_from_ratios(0.14, 0.04)
    for p in [0.5, 1.0, 2.0]:
        A, _ = build_template_matrix(p, masses, 0.05, 100)
        A_tof, _ = build_tof_template_matrix(p, masses, 100)
        N = 500
        bg = np.ones(100) * 0.001 * N / 100
        F_tpc = poisson_fim(A, theta, N, bg)
        F_tof = poisson_fim(A_tof, theta, N, bg)
        _, _, F_j = joint_fisher(F_tpc, F_tof)

        diff = F_j - F_tpc
        eigvals = np.linalg.eigvalsh(diff)
        assert np.all(eigvals >= -1e-10), (
            f"F_joint - F_TPC not PSD at p={p}: min_eig={eigvals[0]:.2e}"
        )
    print("  Joint Fisher additivity: pass")


def test_centrality_monotone():
    """Centrality configs should have monotone decreasing multiplicity."""
    mults = [c[1] for c in CENTRALITY_CONFIGS]
    for i in range(len(mults) - 1):
        assert mults[i] >= mults[i + 1], (
            f"Multiplicity not monotone: {mults[i]} < {mults[i + 1]}"
        )
    print("  Centrality multiplicity monotone: pass")


def test_fractions_sum_to_one():
    """Species fractions should sum to 1 for all centrality configs."""
    for label, _, _, k_pi, p_pi in CENTRALITY_CONFIGS:
        theta = fractions_from_ratios(k_pi, p_pi)
        assert abs(np.sum(theta) - 1.0) < 1e-10, (
            f"Fractions don't sum to 1 for {label}: {np.sum(theta)}"
        )
    print("  Species fractions sum to 1: pass")


def test_crb_multiuser_efficiency_bounds():
    """CRB-based eta should be in [0, 1]."""
    masses = list(MASS.values())
    theta = fractions_from_ratios(0.14, 0.04)
    for p in [0.5, 1.0, 2.0]:
        A, _ = build_template_matrix(p, masses, 0.05, 100)
        N = 500
        bg = np.ones(100) * 0.001 * N / 100
        F = poisson_fim(A, theta, N, bg)
        eta, _ = crb_multiuser_efficiency(F)
        for k in range(3):
            assert 0 <= eta[k] <= 1.0 + 1e-6, (
                f"eta[{k}]={eta[k]} out of bounds at p={p}"
            )
    print("  CRB multiuser efficiency bounds: pass")


def run_all_tests():
    """Run all heavy-ion consistency tests."""
    print("Running heavy-ion extension tests...")
    test_bethe_bloch_physical()
    test_bethe_bloch_mass_ordering()
    test_crossings_exist()
    test_tpc_template_normalization()
    test_tof_template_normalization()
    test_constrained_crb_tighter()
    test_joint_fisher_additive()
    test_centrality_monotone()
    test_fractions_sum_to_one()
    test_crb_multiuser_efficiency_bounds()
    print("\nAll heavy-ion tests passed!")


if __name__ == '__main__':
    run_all_tests()
