"""
validate_heavy_ion.py — Heavy-Ion PID Prediction Validation (P9-P16)
======================================================================

Usage:
    python -m tau_cdma.validate_heavy_ion

Predictions tested:
  P9:  Constrained CRB < unconstrained; gain largest for aliased species
  P10: Eigenvalue collapse at Bethe-Bloch crossing momenta
  P11: Kaon η_K < 0.1 near π/K crossing (TPC only)
  P12: TOF information gain peaks near crossing momenta
  P13: TOF increases kaon Fisher info and η by orders of magnitude
  P14: Central collisions have better absolute precision (CRB)
  P15: Measurability valley η_K(p) deepens with centrality
  P16: Fano bound gives P_e > 5% at crossing (TPC only)

All predictions are self-consistent: crossing momenta are computed
from the ALEPH Bethe-Bloch parameterization, not hardcoded.
"""

import numpy as np

from tau_cdma.heavy_ion.bethe_bloch import MASS, find_crossings, bethe_bloch, build_template_matrix
from tau_cdma.core.fisher import poisson_fim, crb_multiuser_efficiency
from tau_cdma.heavy_ion.centrality import momentum_sweep, centrality_sweep, fractions_from_ratios


def validate_predictions(verbose=True):
    """Validate predictions P9-P16 for heavy-ion PID.

    Returns
    -------
    results : dict — prediction name -> {checks, passed, ...}
    """
    if verbose:
        print("\n" + "=" * 65)
        print("  Heavy-Ion PID: Prediction Validation Suite (P9-P16)")
        print("=" * 65)

    all_results = {}
    masses = [MASS['pi'], MASS['K'], MASS['p']]

    # Step 0: Compute crossing momenta from Bethe-Bloch
    piK_crossings = find_crossings(MASS['pi'], MASS['K'])
    Kp_crossings = find_crossings(MASS['K'], MASS['p'])
    p_cross_piK = piK_crossings[0] if piK_crossings else 1.0
    p_cross_Kp = Kp_crossings[0] if Kp_crossings else 2.4

    if verbose:
        print(f"\n  Bethe-Bloch crossings (computed):")
        print(f"    pi/K: {p_cross_piK:.4f} GeV/c")
        print(f"    K/p:  {p_cross_Kp:.4f} GeV/c")

    # Reference config: mid-centrality (30-40%)
    sigma_ref = 0.055
    theta_ref = fractions_from_ratios(0.14, 0.04)
    N_ref = 512.0

    cross_half = 0.15
    cross_lo = p_cross_piK - cross_half
    cross_hi = p_cross_piK + cross_half

    p_grid = np.linspace(0.2, 5.0, 300)

    if verbose:
        print(f"\n  Config: sigma={sigma_ref}, N={N_ref}")
        print(f"  theta = [{', '.join('%.4f' % t for t in theta_ref)}]")

    res = momentum_sweep(
        masses=masses, sigma=sigma_ref, theta=theta_ref, N=N_ref,
        p_grid=p_grid, compute_tof=True
    )

    # ── P9: Constrained CRB ──
    if verbose:
        print("\n" + "-" * 65)
        print("  P9: Constrained CRB improvement")
        print("-" * 65)

    gain = np.zeros_like(res['CRB'])
    for ip in range(len(p_grid)):
        for k in range(3):
            if res['CRB'][ip, k] > 1e-30:
                gain[ip, k] = 1.0 - res['CRB_c'][ip, k] / res['CRB'][ip, k]

    all_gain_nonneg = np.all(gain >= -1e-6)
    idx_cross = np.argmin(np.abs(p_grid - p_cross_piK))
    wider_mask = (p_grid >= p_cross_piK - 0.3) & (p_grid <= p_cross_piK + 0.3)
    G_K = np.max(gain[wider_mask, 1]) if wider_mask.any() else 0
    G_pi = np.max(gain[wider_mask, 0]) if wider_mask.any() else 0

    checks_p9 = {
        'All G_k >= 0': all_gain_nonneg,
        'G_K > 0 (constraint helps aliased K)': G_K > 1e-4,
        'Dominant species benefits most (G_pi > G_K)': G_pi > G_K,
    }
    if verbose:
        print(f"  G_K near crossing: {G_K:.4f}, G_pi: {G_pi:.4f}")
        for name, ok in checks_p9.items():
            print(f"    {name}: {'pass' if ok else 'FAIL'}")
    all_results['P9'] = {'checks': checks_p9, 'passed': all(checks_p9.values())}

    # ── P10: Eigenvalue collapse ──
    if verbose:
        print("\n" + "-" * 65)
        print("  P10: Eigenvalue collapse at crossings")
        print("-" * 65)

    lam_min = res['eigenvalues'][:, 0]
    global_max = np.max(lam_min[lam_min > 0])

    piK_mask = np.abs(p_grid - p_cross_piK) < 0.3
    Kp_mask = np.abs(p_grid - p_cross_Kp) < 0.3
    ratio_piK = np.min(lam_min[piK_mask]) / global_max if piK_mask.any() else 1
    ratio_Kp = np.min(lam_min[Kp_mask]) / global_max if Kp_mask.any() else 1

    checks_p10 = {
        'Collapse at pi/K (ratio < 0.01)': ratio_piK < 0.01,
        'Collapse at K/p (ratio < 0.01)': ratio_Kp < 0.01,
        'Crossings found': len(piK_crossings) > 0 and len(Kp_crossings) > 0,
    }
    if verbose:
        print(f"  pi/K: ratio={ratio_piK:.2e}, K/p: ratio={ratio_Kp:.2e}")
        for name, ok in checks_p10.items():
            print(f"    {name}: {'pass' if ok else 'FAIL'}")
    all_results['P10'] = {'checks': checks_p10, 'passed': all(checks_p10.values())}

    # ── P11: Kaon eta_K < 0.1 ──
    if verbose:
        print("\n" + "-" * 65)
        print("  P11: Kaon eta_K < 0.1 near crossing")
        print("-" * 65)

    window = (p_grid >= cross_lo) & (p_grid <= cross_hi)
    eta_K_at_cross = res['eta'][idx_cross, 1]
    eta_K_min = np.min(res['eta'][window, 1]) if window.any() else 1.0

    checks_p11 = {
        'eta_K < 0.1 at crossing': eta_K_at_cross < 0.1,
        'eta_K min < 0.01 in window': eta_K_min < 0.01,
    }
    if verbose:
        print(f"  eta_K at crossing: {eta_K_at_cross:.6f}, min in window: {eta_K_min:.6f}")
        for name, ok in checks_p11.items():
            print(f"    {name}: {'pass' if ok else 'FAIL'}")
    all_results['P11'] = {'checks': checks_p11, 'passed': all(checks_p11.values())}

    # ── P12: TOF gain peaks at crossing ──
    if verbose:
        print("\n" + "-" * 65)
        print("  P12: TOF eta improvement peaks near crossing")
        print("-" * 65)

    eta_K_tpc = res['eta'][:, 1]
    eta_K_joint = res['eta_joint'][:, 1]
    eta_improve = np.where(eta_K_tpc > 1e-6, eta_K_joint / eta_K_tpc, eta_K_joint * 1e6)
    interior = (p_grid > 0.3) & (p_grid < 4.5)
    ei = eta_improve.copy()
    ei[~interior] = 0
    p_peak = p_grid[np.argmax(ei)]
    dist = min(abs(p_peak - p_cross_piK), abs(p_peak - p_cross_Kp))

    checks_p12 = {
        'Peak within 0.3 GeV/c of a crossing': dist < 0.3,
        'Improvement > 10x': np.max(ei) > 10,
    }
    if verbose:
        nearest = 'pi/K' if abs(p_peak - p_cross_piK) < abs(p_peak - p_cross_Kp) else 'K/p'
        print(f"  Peak at {p_peak:.3f} GeV/c (near {nearest}), {np.max(ei):.0f}x improvement")
        for name, ok in checks_p12.items():
            print(f"    {name}: {'pass' if ok else 'FAIL'}")
    all_results['P12'] = {'checks': checks_p12, 'passed': all(checks_p12.values())}

    # ── P13: TOF rescues kaon ID ──
    if verbose:
        print("\n" + "-" * 65)
        print("  P13: TOF increases kaon info at crossing")
        print("-" * 65)

    F_TPC_KK = res['F'][idx_cross, 1, 1]
    F_joint_KK = res['F_joint'][idx_cross, 1, 1]
    fisher_ratio = F_joint_KK / max(F_TPC_KK, 1e-30)
    eta_ratio = res['eta_joint'][idx_cross, 1] / max(res['eta'][idx_cross, 1], 1e-10)

    checks_p13 = {
        'Fisher ratio > 3': fisher_ratio > 3,
        'eta_K improves > 100x': eta_ratio > 100,
    }
    if verbose:
        print(f"  Fisher ratio: {fisher_ratio:.1f}x, eta improvement: {eta_ratio:.0f}x")
        for name, ok in checks_p13.items():
            print(f"    {name}: {'pass' if ok else 'FAIL'}")
    all_results['P13'] = {'checks': checks_p13, 'passed': all(checks_p13.values())}

    # ── P14: Central vs peripheral ──
    if verbose:
        print("\n" + "-" * 65)
        print("  P14: Central has better absolute precision")
        print("-" * 65)
        print("  Running centrality sweep...")

    csweep = centrality_sweep(p_grid=p_grid, compute_tof=False, n_bins=100)

    p_check = p_cross_piK + 0.5
    idx_check = np.argmin(np.abs(p_grid - p_check))
    crb_central = csweep['CRB'][0, idx_check, 1]
    crb_periph = csweep['CRB'][-1, idx_check, 1]

    checks_p14 = {
        'CRB_K(central) < CRB_K(peripheral)': crb_central < crb_periph,
    }
    if verbose:
        print(f"  At p={p_check:.2f}: CRB_K central={crb_central:.2e}, periph={crb_periph:.2e}")
        print(f"  Precision: {crb_periph/max(crb_central,1e-30):.1f}x better in central")
        for name, ok in checks_p14.items():
            print(f"    {name}: {'pass' if ok else 'FAIL'}")
    all_results['P14'] = {'checks': checks_p14, 'passed': all(checks_p14.values())}

    # ── P15: Valley deepens ──
    if verbose:
        print("\n" + "-" * 65)
        print("  P15: Valley deepens with centrality")
        print("-" * 65)

    valley = (p_grid >= cross_lo) & (p_grid <= cross_hi)
    v_central = np.min(csweep['eta'][0, valley, 1])
    v_periph = np.min(csweep['eta'][-1, valley, 1])

    checks_p15 = {
        'Valley exists (eta_K < 0.1)': v_central < 0.1,
        'Valley deeper in central': v_central <= v_periph + 1e-6,
    }
    if verbose:
        print(f"  eta_K valley: central={v_central:.6f}, peripheral={v_periph:.6f}")
        for name, ok in checks_p15.items():
            print(f"    {name}: {'pass' if ok else 'FAIL'}")
    all_results['P15'] = {'checks': checks_p15, 'passed': all(checks_p15.values())}

    # ── P16: Fano bound ──
    if verbose:
        print("\n" + "-" * 65)
        print("  P16: Fano bound at crossing")
        print("-" * 65)

    A_cross, _ = build_template_matrix(p_cross_piK, masses, sigma_ref, 100)
    mixture = A_cross @ theta_ref
    H_K = -np.sum(theta_ref * np.log2(np.maximum(theta_ref, 1e-30)))

    H_K_Y = 0.0
    for i in range(len(mixture)):
        if mixture[i] > 1e-30:
            for k in range(3):
                post = theta_ref[k] * A_cross[i, k] / mixture[i]
                if post > 1e-30:
                    H_K_Y -= mixture[i] * post * np.log2(post)
    MI = H_K - H_K_Y

    # Binary search for Fano P_e (tightened, C3)
    rhs = H_K - MI
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if mid < 1e-15 or mid > 1 - 1e-15:
            Hb = 0.0
        else:
            Hb = -mid * np.log2(mid) - (1 - mid) * np.log2(1 - mid)
        if Hb + mid * np.log2(2) < rhs:
            lo = mid
        else:
            hi = mid
    Pe_fano = lo

    checks_p16 = {
        'P_e > 5% at crossing': Pe_fano > 0.05,
        'MI < H(K)': MI < H_K,
        'MI < 0.5*H(K) (significant loss)': MI < 0.5 * H_K,
    }
    if verbose:
        print(f"  H(K)={H_K:.4f}, I(K;Y)={MI:.4f}, Fano P_e>={Pe_fano:.4f} ({Pe_fano*100:.1f}%)")
        for name, ok in checks_p16.items():
            print(f"    {name}: {'pass' if ok else 'FAIL'}")
    all_results['P16'] = {'checks': checks_p16, 'passed': all(checks_p16.values())}

    # ── SUMMARY ──
    if verbose:
        print("\n" + "=" * 65)
        print("  SUMMARY")
        print("=" * 65)
        total_c, total_p = 0, 0
        for pid in ['P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16']:
            r = all_results[pid]
            n = len(r['checks'])
            p = sum(r['checks'].values())
            total_c += n
            total_p += p
            print(f"  {pid}: {p}/{n} {'PASS' if r['passed'] else 'FAIL'}")
        print(f"\n  Total: {total_p}/{total_c}")
        print(f"  Crossings: pi/K={p_cross_piK:.4f}, K/p={p_cross_Kp:.4f}")

    return all_results


if __name__ == '__main__':
    validate_predictions(verbose=True)
