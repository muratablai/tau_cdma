"""
test_consistency.py — Mathematical consistency tests
=====================================================

Verifies:
  - FIM symmetry and positive semi-definiteness
  - Template normalization
  - Simplex constraint
  - Schur complement ⪯ property
  - Code rate bounds
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from tau_cdma.tau.templates import TauTemplates, TAU_BR
from tau_cdma.core.fisher import poisson_fim, crb, eigenvalue_spectrum, reduced_fim
from tau_cdma.core.interference import interference_matrix, multiuser_efficiency
from tau_cdma.core.cascade import cascade_fim_schur
from tau_cdma.core.emergent import parity_check_matrix, code_rate


def test_templates():
    """Templates should be non-negative and normalized."""
    tb = TauTemplates(M=200)
    A = tb.A
    dm = tb.dm

    # Non-negative
    assert np.all(A >= 0), "Templates must be non-negative"

    # Each column should sum to 1 (probability vector)
    for k in range(7):
        col_sum = np.sum(A[:, k])
        assert abs(col_sum - 1.0) < 0.01, f"Template {k} sum = {col_sum}, expected 1"

    # BR sum to 1
    assert abs(np.sum(TAU_BR) - 1.0) < 0.01, f"BR sum = {np.sum(TAU_BR)}"

    print("  templates: ✓")


def test_fim_symmetry():
    """FIM must be symmetric and positive semi-definite."""
    tb = TauTemplates(M=200)
    A = tb.A
    N = 1e6
    bg = 0.01 * tb.dm * np.ones(tb.M)

    F = poisson_fim(A, TAU_BR, N, bg)

    # Symmetric
    assert np.allclose(F, F.T, atol=1e-10), "FIM must be symmetric"

    # PSD: all eigenvalues ≥ 0
    eigvals, _ = eigenvalue_spectrum(F)
    assert np.all(eigvals >= -1e-10), f"FIM has negative eigenvalue: {eigvals[0]}"

    print("  FIM symmetry + PSD: ✓")


def test_reduced_fim():
    """Reduced FIM should be (K-1)×(K-1)."""
    tb = TauTemplates(M=200)
    A = tb.A
    N = 1e6
    bg = 0.01 * tb.dm * np.ones(tb.M)

    F_red = reduced_fim(A, TAU_BR, N, bg)
    assert F_red.shape == (6, 6), f"Reduced FIM shape {F_red.shape}, expected (6,6)"
    assert np.allclose(F_red, F_red.T, atol=1e-10), "Reduced FIM must be symmetric"

    print("  reduced FIM: ✓")


def test_interference_diagonal():
    """Interference matrix R should have 1s on diagonal."""
    tb = TauTemplates(M=200)
    A = tb.A
    N = 1e6
    bg = 0.01 * tb.dm * np.ones(tb.M)

    R = interference_matrix(A, TAU_BR, N, bg)
    assert np.allclose(np.diag(R), 1.0, atol=1e-10), "R diagonal must be 1"
    assert np.allclose(R, R.T, atol=1e-10), "R must be symmetric"

    print("  R matrix diagonal: ✓")


def test_multiuser_efficiency_bounds():
    """η_k should be in (0, 1]."""
    tb = TauTemplates(M=200)
    A = tb.A
    N = 1e6
    bg = 0.01 * tb.dm * np.ones(tb.M)

    R = interference_matrix(A, TAU_BR, N, bg)
    eta = multiuser_efficiency(R)

    for k in range(7):
        assert 0 < eta[k] <= 1.0 + 1e-10, f"η[{k}] = {eta[k]} out of bounds"

    print("  η bounds: ✓")


def test_schur_complement_ordering():
    """F_eff ⪯ F_PP (Schur complement reduces information)."""
    n = 6
    rng = np.random.default_rng(42)
    # Random PSD matrix
    X = rng.normal(0, 1, (n, n))
    F = X @ X.T + np.eye(n)

    F_eff, F_PP, loss = cascade_fim_schur(F, [0, 1, 2], [3, 4, 5])
    assert loss >= -1e-10, f"Info loss should be ≥ 0, got {loss}"

    # F_PP - F_eff should be PSD
    diff = F_PP - F_eff
    eigvals = np.linalg.eigvalsh(diff)
    assert np.all(eigvals >= -1e-10), "F_PP - F_eff must be PSD"

    print("  Schur complement ⪯: ✓")


def test_parity_check_ranks():
    """Verify rank(H) for each interaction type."""
    tests = {
        'strong': 9,
        'weak_vertex': 5,
        'weak_propagation': 3,
    }
    for interaction, expected_rank in tests.items():
        H, rank, _ = parity_check_matrix(interaction)
        assert rank == expected_rank, f"{interaction}: rank={rank}, expected {expected_rank}"

    print("  parity check ranks: ✓")


def test_code_rates():
    """Code rate R = (n - rank(H)) / n."""
    assert abs(code_rate('strong') - 0.0) < 1e-10
    assert abs(code_rate('weak_vertex') - 4/9) < 1e-10
    assert abs(code_rate('weak_propagation') - 2/3) < 1e-10

    print("  code rates: ✓")


def run_all_tests():
    """Run all consistency tests."""
    print("Running consistency tests...")
    test_templates()
    test_fim_symmetry()
    test_reduced_fim()
    test_interference_diagonal()
    test_multiuser_efficiency_bounds()
    test_schur_complement_ordering()
    test_parity_check_ranks()
    test_code_rates()
    print("\nAll tests passed! ✓")


if __name__ == '__main__':
    run_all_tests()
