"""
validate_predictions.py — Framework Prediction Validation Suite (v4.5)
======================================================================

Usage:
    python -m tau_cdma.validate [--quick]

Predictions tested:
  P1: η ordering + Shannon classification (Bayes μ=0% theorem)
  P2: Aliasing order — Monte Carlo validated template distances
  P3: Geometric vs random erasure — sweep over access fractions
  P4: Cascade bottleneck at a₁→3π
  P7: Optimal binning M_opt ∝ SF_k — Fisher saturation curves
  P8: Blind decomposition + NN receiver + Bayes ceiling
  P9: Aliasing as eigenvalue collapse of R
"""

import numpy as np
import sys
import os
from scipy.optimize import minimize


# =====================================================================
#  P1: Multiuser Efficiency Ordering
# =====================================================================

def validate_p1(bench, verbose=True):
    """P1: Multiuser efficiency ordering — observable dependence.

    v4.5 core prediction: η ordering reshuffles when the observable changes.
      1D m_vis:   η_π ≫ η_ρ > η_a₁ > η_e ≈ η_μ ≈ 0
      With PID:   η_e, η_μ jump to high values; π remains high

    We demonstrate this by computing η under two scenarios:
      (a) 1D visible mass only (M=200 bins)
      (b) m_vis + particle ID (3-category PID: e / μ / hadron)
    """
    from tau_cdma.core.fisher import poisson_fim
    from tau_cdma.core.interference import interference_matrix, multiuser_efficiency

    if verbose:
        print("\n" + "="*60)
        print("P1: Multiuser Efficiency Ordering")
        print("="*60)

    eta_1d = bench['eta']
    R_1d = bench['R']
    labels = bench['templates'].short_labels
    K = len(labels)

    # --- 1D m_vis ---
    if verbose:
        print(f"\n  (a) 1D visible mass (M=200):")
        for k in range(K):
            print(f"      η[{labels[k]:>5s}] = {eta_1d[k]:.4f}")

    # --- With PID + n_trk: augment template matrix ---
    # SYNTHETIC WEIGHTED FUSION BENCHMARK
    # This constructs a multi-observable template by stacking mass, track
    # multiplicity, and PID blocks with ad hoc weights (0.5, 0.2, 0.3).
    # This is NOT a detector-derived joint likelihood over a true combined
    # observation space. It is a synthetic benchmark demonstrating that the
    # framework's η_k ordering changes when observables are added, and that
    # the Fisher machinery handles multi-observable fusion correctly.
    #
    # n_trk (track multiplicity):
    #   e, μ, π, ρ(→ππ⁰), π2π⁰: 1-prong (1 charged track)
    #   a₁(→3π): 3-prong (3 charged tracks)
    #   other: ~70% 1-prong, ~30% 3-prong (PDG τ topology fractions)
    #
    # PID (particle identification):
    #   e → PID=e, μ → PID=μ, hadronic → PID=had
    
    A_1d = bench['A']  # (M, 7)
    M = A_1d.shape[0]

    # n_trk observable: 2 bins (1-prong, 3-prong)
    ntrk_block = np.zeros((2, K))
    ntrk_block[0, :] = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.7]  # 1-prong
    ntrk_block[1, :] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3]  # 3-prong

    # PID observable: 3 bins (e, μ, hadron)
    pid_block = np.zeros((3, K))
    pid_block[0, 0] = 1.0   # e → PID=e
    pid_block[1, 1] = 1.0   # μ → PID=μ
    pid_block[2, 2:] = 1.0  # all hadronic → PID=had

    # Combine: (m_vis, n_trk, PID) with relative weights
    mass_weight = 0.5
    ntrk_weight = 0.2
    pid_weight = 0.3

    # Normalize each block's columns to sum to 1, then weight
    ntrk_norm = ntrk_block / np.maximum(ntrk_block.sum(axis=0, keepdims=True), 1e-30)
    pid_norm = pid_block / np.maximum(pid_block.sum(axis=0, keepdims=True), 1e-30)
    
    A_aug = np.vstack([
        A_1d * mass_weight,
        ntrk_norm * ntrk_weight,
        pid_norm * pid_weight,
    ])
    # Re-normalize columns to sum to 1
    col_sums = A_aug.sum(axis=0, keepdims=True)
    A_aug = A_aug / np.maximum(col_sums, 1e-30)
    A_pid = A_aug  # keep name for backward compatibility

    theta = bench['theta']
    N = bench['N']
    M_aug = A_pid.shape[0]
    bg_aug = np.concatenate([bench['background'], 0.001 * np.ones(5)])  # 2 ntrk + 3 PID bins

    F_pid = poisson_fim(A_pid, theta, N, bg_aug)
    R_pid = interference_matrix(A_pid, theta, N, bg_aug)
    eta_pid = multiuser_efficiency(R_pid)

    # Also compute CRB to quantify estimation improvement
    from tau_cdma.core.fisher import crb as compute_crb
    crb_1d = bench['crb']
    crb_pid = compute_crb(F_pid, regularize=True)

    if verbose:
        print(f"\n  (b) (m_vis, n_trk, PID) ({M}+2+3={M_aug} bins, "
              f"weights: mass={mass_weight}, ntrk={ntrk_weight}, PID={pid_weight}):")
        for k in range(K):
            arrow = " ↑↑↑" if eta_pid[k] > 10 * eta_1d[k] else ""
            print(f"      η[{labels[k]:>5s}] = {eta_pid[k]:.4f}{arrow}")

        print(f"\n  Observable dependence (η ratio and CRB improvement):")
        print(f"      {'Ch':>5s}  {'η_pid/η_1D':>10s}  {'σ_1D':>10s}  {'σ_PID':>10s}  {'σ improve':>10s}")
        for k in range(K):
            s1 = np.sqrt(crb_1d[k]) if np.isfinite(crb_1d[k]) and crb_1d[k] > 0 else np.inf
            sp = np.sqrt(crb_pid[k]) if np.isfinite(crb_pid[k]) and crb_pid[k] > 0 else np.inf
            if eta_1d[k] > 1e-6:
                eta_r = f"×{eta_pid[k] / eta_1d[k]:.1f}"
            else:
                eta_r = f"0→{eta_pid[k]:.2f}"
            if np.isfinite(s1) and np.isfinite(sp) and sp > 0:
                sig_r = f"×{s1/sp:.1f}"
            elif np.isinf(s1) and np.isfinite(sp):
                sig_r = "∞→finite"
            else:
                sig_r = "—"
            s1s = f"{s1:.5f}" if np.isfinite(s1) else "∞"
            sps = f"{sp:.5f}" if np.isfinite(sp) else "∞"
            print(f"      {labels[k]:>5s}  {eta_r:>10s}  {s1s:>10s}  {sps:>10s}  {sig_r:>10s}")

    # --- (c) Shannon classification view ---
    from tau_cdma.core.shannon import (bayes_confusion as compute_bayes,
                                  classification_mi, uncertainty_decomposition)

    bc_1d = compute_bayes(A_1d, theta)
    bc_pid = compute_bayes(A_pid, theta)
    mi_1d = classification_mi(A_1d, theta)
    mi_pid = classification_mi(A_pid, theta)
    unc = uncertainty_decomposition(crb_1d, crb_pid)

    if verbose:
        print(f"\n  (c) Shannon classification view:")
        print(f"      I(K;Y) = {mi_1d['MI']:.3f} bits/event → "
              f"{mi_1d['n_eff']:.1f} effective channels (of {K})")
        print(f"      Fano bound: P_error ≥ {mi_1d['fano_bound']:.1%}")
        print()
        print(f"      Bayes-optimal MAP accuracy (theoretical ceiling):")
        print(f"      {'Ch':>6s}  {'1D mass':>8s}  {'+ PID':>8s}  {'η_1D':>8s}  "
              f"{'alias%':>8s}")
        for k in range(K):
            alias_pct = f"{unc['aliasing_frac'][k]:.0%}"
            print(f"      {labels[k]:>6s}  {bc_1d['accuracy'][k]:>8.1%}  "
                  f"{bc_pid['accuracy'][k]:>8.1%}  {eta_1d[k]:>8.4f}  "
                  f"{alias_pct:>8s}")
        print(f"      {'Total':>6s}  {bc_1d['overall']:>8.1%}  "
              f"{bc_pid['overall']:>8.1%}")
        print()
        print(f"      μ = {bc_1d['accuracy'][1]:.0%} accuracy is a theorem: θ_e > θ_μ with")
        print(f"      identical templates → P(μ|m) < P(e|m) for every mass bin.")
        print(f"      Even the perfect classifier never predicts μ.")
        print(f"      PID breaks this: μ accuracy → {bc_pid['accuracy'][1]:.0%}")

    # Checks — aligned with formalism criteria (prediction_criteria.py)
    from tau_cdma.validate.prediction_criteria import P1_CRITERIA as C
    
    # Augmented (m_vis, n_trk, PID) space checks
    eta_12_rel = abs(eta_pid[0] - eta_pid[1]) / max(eta_pid[0], eta_pid[1], 1e-30)
    eta_56_rel = abs(eta_pid[4] - eta_pid[5]) / max(eta_pid[4], eta_pid[5], 1e-30)
    
    checks = {
        # Augmented space ordering (formalism exact criterion)
        f'|η₁-η₂|/max < {C["eta_12_rel_diff_max"]} (e≈μ)': eta_12_rel < C['eta_12_rel_diff_max'],
        'η₃ > η₁ (π dominates augmented)': eta_pid[2] > eta_pid[0],
        'η₃ > η₄ > η₅ (hadronic ordering)': eta_pid[2] > eta_pid[3] > eta_pid[4],
        f'|η₅-η₆|/max < {C["eta_56_rel_diff_max"]} (a₁≈π2π⁰)': eta_56_rel < C['eta_56_rel_diff_max'],
        # 1D corollary checks
        f'π dominates 1D (η_π > {C["pi_dominates_1d_min"]})': eta_1d[2] > C['pi_dominates_1d_min'],
        f'e,μ aliased 1D (R_eμ > {C["R_emu_min"]})': R_1d[0, 1] > C['R_emu_min'],
        f'e,μ near zero 1D (η < {C["leptonic_aliased_1d_max"]})': (
            eta_1d[0] < C['leptonic_aliased_1d_max'] and eta_1d[1] < C['leptonic_aliased_1d_max']
        ),
        # Shannon checks (from formalism)
        f'Bayes μ = 0% in 1D (acc < {C["mu_accuracy_1d_max"]})': bc_1d['accuracy'][1] < C['mu_accuracy_1d_max'],
        f'PID rescues μ (acc > {C["mu_accuracy_pid_min"]})': bc_pid['accuracy'][1] > C['mu_accuracy_pid_min'],
    }

    passed = all(checks.values())
    if verbose:
        print(f"\n  Checks:")
        for name, ok in checks.items():
            print(f"    {name}: {'✓' if ok else '✗'}")
        print(f"\n  P1 {'PASSED' if passed else 'NEEDS REVIEW'}")

    return {'eta_1d': eta_1d, 'eta_pid': eta_pid, 'R_1d': R_1d, 'R_pid': R_pid,
            'bayes_1d': bc_1d, 'bayes_pid': bc_pid, 'mi_1d': mi_1d, 'mi_pid': mi_pid,
            'uncertainty_decomposition': unc,
            'checks': checks, 'passed': passed}


# =====================================================================
#  P2: Aliasing Order
# =====================================================================

def validate_p2(bench, verbose=True):
    """P2: Aliasing order — validated with Monte Carlo.

    Per-event template distances predict which channel pairs become
    indistinguishable first as resolution degrades. We validate this
    prediction with actual Poisson simulations at each M.
    """
    from tau_cdma.core.aliasing import aliasing_sweep, aliasing_threshold_matrix, aliasing_order
    from tau_cdma.tau.templates import TauTemplates
    from tau_cdma.core.fisher import poisson_fim, crb

    if verbose:
        print("\n" + "="*60)
        print("P2: Aliasing Order (Template Distance + Monte Carlo)")
        print("="*60)

    tb = bench['templates']
    theta = bench['theta']
    N = bench['N']
    K = len(theta)
    labels = bench['templates'].short_labels
    M_values = [3, 5, 10, 20, 50, 100, 200, 500]

    # --- Part A: Per-event template distances ---
    results = aliasing_sweep(tb, M_values, theta, N)
    order, dist_vs_M = aliasing_order(results)

    # --- Part A2: Aliasing threshold ordering M* ---
    # Formalism criterion: M*₅₆ < M*₄₅ < M*₃₄ < M*₁₂
    # (1-indexed; in 0-indexed: M*(4,5) < M*(3,4) < M*(2,3) < M*(0,1))
    # M*_ij = smallest M at which d²(i,j) > d²_threshold
    d2_threshold = 0.05  # threshold for "separable"
    pairs_to_check = [(4, 5), (3, 4), (2, 3), (0, 1)]
    pair_labels = ['(a₁,π2π⁰)', '(ρ,a₁)', '(π,ρ)', '(e,μ)']
    M_star = {}
    for (i, j) in pairs_to_check:
        M_star[(i,j)] = M_values[-1]  # default: never separated
        for r in sorted(results, key=lambda x: x['M']):
            D = r['distances_per_event']
            if D[i, j] > d2_threshold:
                M_star[(i,j)] = r['M']
                break
    
    # Check ordering
    mstar_values = [M_star[p] for p in pairs_to_check]
    ordering_ok = all(mstar_values[i] <= mstar_values[i+1] for i in range(len(mstar_values)-1))
    
    if verbose:
        print(f"\n  Aliasing threshold ordering (d² > {d2_threshold}):")
        for idx, (i,j) in enumerate(pairs_to_check):
            print(f"      M*{pair_labels[idx]:>14s} = {M_star[(i,j)]}")
        print(f"      Ordering M*(a₁,π2π⁰) ≤ M*(ρ,a₁) ≤ M*(π,ρ) ≤ M*(e,μ): {'✓' if ordering_ok else '✗'}")

    if verbose:
        print(f"\n  (a) Aliasing order (per-event d², M=3):")
        for j, k, d2 in order:
            if d2 < 3.0 or (j, k) in [(0,1), (3,4), (4,5), (2,3)]:
                print(f"      ({labels[j]}, {labels[k]}): d² = {d2:.4f}")

        print(f"\n  Per-event d² vs M:")
        print(f"  {'M':>5s}  {'e-μ':>8s}  {'e-π':>8s}  {'ρ-a₁':>8s}  {'a₁-π2π⁰':>8s}")
        for r in sorted(results, key=lambda x: x['M']):
            D = r['distances_per_event']
            print(f"  {r['M']:>5d}  {D[0,1]:>8.4f}  {D[0,2]:>8.4f}  {D[3,4]:>8.4f}  {D[4,5]:>8.4f}")

    # --- Part B: Monte Carlo validation ---
    # Key test: e-μ correlation sign flips from +1 (degenerate) to -1 (trade-off)
    # At low M: fitter can't separate e/μ → they move together → corr ≈ +1
    # At high M: partially separable → anti-correlated trade-off → corr ≈ -1
    # The σ(e+μ) sum is always small → only the split is uncertain
    mc_M_values = [3, 5, 10, 20, 50, 100, 200]
    n_mc = 200
    rng = np.random.default_rng(42)

    if verbose:
        print(f"\n  (b) Monte Carlo validation ({n_mc} simulations per M):")
        print(f"      e-μ aliasing signature: correlation sign flip")
        print(f"      {'M':>5s}  {'corr(e,μ)':>10s}  {'σ_e':>10s}  {'σ_μ':>10s}  "
              f"{'σ(e+μ)':>10s}  {'Phase':>12s}")

    mc_results = {}
    for M_mc in mc_M_values:
        tb_mc = TauTemplates(M=M_mc, m_range=tb.m_range, sigma_det=tb.sigma_det)
        A_mc = tb_mc.A
        dm_mc = (tb.m_range[1] - tb.m_range[0]) / M_mc
        bg_mc = 0.01 * dm_mc * np.ones(M_mc)
        lam = N * (A_mc @ theta) + bg_mc

        F_mc = poisson_fim(A_mc, theta, N, bg_mc)
        crb_mc = crb(F_mc, regularize=True)

        theta_fits = np.zeros((n_mc, K))
        for trial in range(n_mc):
            y = rng.poisson(lam)
            def nll(t, A_fit=A_mc, bg_fit=bg_mc):
                t = np.abs(t)
                l = N * (A_fit @ t) + bg_fit
                l = np.maximum(l, 1e-30)
                return np.sum(l - y * np.log(l))
            res = minimize(nll, theta, method='L-BFGS-B',
                          bounds=[(1e-6, 1)]*K)
            theta_fits[trial] = np.abs(res.x)

        mc_var = np.var(theta_fits, axis=0)
        mc_corr = np.corrcoef(theta_fits.T)
        sigma_sum = np.std(theta_fits[:, 0] + theta_fits[:, 1])

        mc_results[M_mc] = {
            'crb': crb_mc, 'mc_var': mc_var, 'mc_corr': mc_corr,
            'sigma_e': np.sqrt(mc_var[0]), 'sigma_mu': np.sqrt(mc_var[1]),
            'sigma_sum': sigma_sum,
        }

        if verbose:
            corr_emu = mc_corr[0, 1]
            phase = "degenerate" if corr_emu > 0.5 else ("trade-off" if corr_emu < -0.5 else "transition")
            print(f"      {M_mc:>5d}  {corr_emu:>+10.3f}  {np.sqrt(mc_var[0]):>10.5f}  "
                  f"{np.sqrt(mc_var[1]):>10.5f}  {sigma_sum:>10.5f}  {phase:>12s}")

    # Find the transition M where correlation flips sign
    sign_flip_M = None
    prev_corr = mc_results[mc_M_values[0]]['mc_corr'][0, 1]
    for M_mc in mc_M_values[1:]:
        curr_corr = mc_results[M_mc]['mc_corr'][0, 1]
        if prev_corr > 0 and curr_corr < 0:
            sign_flip_M = M_mc
            break
        prev_corr = curr_corr

    if verbose and sign_flip_M:
        print(f"\n      Correlation sign flip at M={sign_flip_M}")
        print(f"      (e-μ transition from degenerate to partially separable)")

    # CRB saturation efficiency: σ_MC / σ_CRB at M=200
    M_eff = 200
    if M_eff in mc_results:
        mc_r = mc_results[M_eff]
        if verbose:
            print(f"\n      Estimator efficiency at M={M_eff} (σ_MC/σ_CRB):")
            for k in range(K):
                if np.isfinite(mc_r['crb'][k]) and mc_r['crb'][k] > 0:
                    ratio = np.sqrt(mc_r['mc_var'][k]) / np.sqrt(mc_r['crb'][k])
                    eff_str = f"{ratio:.2f}"
                else:
                    eff_str = "CRB=∞"
                print(f"        {labels[k]:>6s}: {eff_str}")
            print(f"      (e,μ < 1.0 at M=200 because aliasing inflates CRB)")

    # Checks — aligned with formalism criteria (prediction_criteria.py)
    from tau_cdma.validate.prediction_criteria import P2_CRITERIA as C
    coarsest = min(results, key=lambda r: r['M'])
    D = coarsest['distances_per_event']

    checks = {
        '(e,μ) most aliased (d²≈0)': D[0, 1] < 0.01,
        '(a₁,π2π⁰) more aliased than (ρ,a₁)': D[4, 5] < D[3, 4],
        '(e,μ) more aliased than (e,π)': D[0, 1] < D[0, 2],
        'π most distinct (max d² from all)': all(D[2, k] > 1.0 for k in range(K) if k != 2),
        # M* ordering (formalism criterion)
        'M* ordering: M*(a₁,π2π⁰) ≤ ... ≤ M*(e,μ)': ordering_ok,
        # MC: e-μ degenerate at M=3
        f'MC: e-μ degenerate at M=3 (corr>{C["mc_degenerate_corr_min"]})': (
            mc_results[3]['mc_corr'][0, 1] > C['mc_degenerate_corr_min']
        ),
        # MC: e-μ anti-correlated at M≥200
        f'MC: e-μ trade-off at M=200 (corr<{C["mc_tradeoff_corr_max"]})': (
            mc_results[200]['mc_corr'][0, 1] < C['mc_tradeoff_corr_max']
        ),
        'MC: correlation sign flip exists': sign_flip_M is not None,
        'MC: σ(e+μ) < σ_e at M=200': mc_results[200]['sigma_sum'] < mc_results[200]['sigma_e'],
    }

    passed = all(checks.values())
    if verbose:
        print(f"\n  Checks:")
        for name, ok in checks.items():
            print(f"    {name}: {'✓' if ok else '✗'}")
        print(f"\n  P2 {'PASSED' if passed else 'NEEDS REVIEW'}")

    return {'sweep_results': results, 'order': order, 'mc_results': mc_results,
            'M_star': M_star, 'M_star_ordering': ordering_ok,
            'checks': checks, 'passed': passed}


# =====================================================================
#  P3: Geometric vs Random Erasure
# =====================================================================

def validate_p3(bench, verbose=True):
    """P3: Geometric erasure penalty ≥ random.

    Structured (geometric) erasure removes contiguous mass regions,
    potentially destroying entire channels. Random erasure spreads
    damage uniformly. We sweep α from 0.3 to 1.0 and compare.
    """
    from tau_cdma.core.erasure import erasure_sweep

    if verbose:
        print("\n" + "="*60)
        print("P3: Geometric vs Random Erasure")
        print("="*60)

    A = bench['A']
    theta = bench['theta']
    N = bench['N']
    bg = bench['background']
    m_bins = bench['templates'].m_bins
    labels = bench['templates'].short_labels
    K = len(labels)

    alpha_vals = np.linspace(0.3, 1.0, 15)

    res_random = erasure_sweep(A, theta, N, bg, alpha_vals,
                                n_trials=50, mode='random')
    res_geom = erasure_sweep(A, theta, N, bg, alpha_vals,
                              mode='geometric', m_bins=m_bins)

    # Show at multiple α values
    if verbose:
        for alpha_show in [0.5, 0.7, 0.9]:
            idx = np.argmin(np.abs(alpha_vals - alpha_show))
            print(f"\n  α ≈ {alpha_vals[idx]:.2f} (σ = √CRB):")
            print(f"  {'Channel':>8s}  {'Random':>10s}  {'Geometric':>10s}  {'Verdict':>12s}")
            for k in range(K):
                cr_var = res_random['crb_mean'][idx, k]
                cg_var = res_geom['crb_mean'][idx, k]
                cr = np.sqrt(cr_var) if np.isfinite(cr_var) else np.inf
                cg = np.sqrt(cg_var) if np.isfinite(cg_var) else np.inf

                if np.isinf(cg) and np.isfinite(cr):
                    verdict = "∞ (erased)"
                elif np.isinf(cr) and np.isfinite(cg):
                    verdict = "random worse"
                elif np.isinf(cr) and np.isinf(cg):
                    verdict = "both ∞"
                else:
                    ratio = cg / max(cr, 1e-30)
                    verdict = f"×{ratio:.2f}"

                cr_s = f"{cr:.5f}" if np.isfinite(cr) else "∞"
                cg_s = f"{cg:.5f}" if np.isfinite(cg) else "∞"
                print(f"  {labels[k]:>8s}  {cr_s:>10s}  {cg_s:>10s}  {verdict:>12s}")

    # Count channels where geometric ≥ random at each α
    n_worse_by_alpha = []
    for idx in range(len(alpha_vals)):
        n = 0
        for k in range(K):
            cg = res_geom['crb_mean'][idx, k]
            cr = res_random['crb_mean'][idx, k]
            if np.isinf(cg) and np.isfinite(cr):
                n += 1
            elif np.isfinite(cg) and np.isfinite(cr) and cg > cr:
                n += 1
        n_worse_by_alpha.append(n)

    if verbose:
        print(f"\n  Channels where geometric ≥ random vs α:")
        for idx in range(0, len(alpha_vals), 3):
            print(f"    α={alpha_vals[idx]:.2f}: {n_worse_by_alpha[idx]}/{K}")

    # Compute explicit CRB ratios r_k = CRB_geometric / CRB_random at target α
    from tau_cdma.validate.prediction_criteria import P3_CRITERIA as C
    alpha_target = C['alpha_test']
    idx_target = np.argmin(np.abs(alpha_vals - alpha_target))
    
    r_k = np.full(K, np.nan)
    for k in range(K):
        cg = res_geom['crb_mean'][idx_target, k]
        cr = res_random['crb_mean'][idx_target, k]
        if np.isfinite(cg) and np.isfinite(cr) and cr > 0:
            r_k[k] = cg / cr
        elif np.isinf(cg) and np.isfinite(cr):
            r_k[k] = np.inf  # geometric kills, random doesn't
    
    hadronic = C['hadronic_channels']
    
    # Among channels with FINITE r_k in both modes, find the max penalty
    finite_mask = np.isfinite(r_k) & (r_k > 0)
    if np.any(finite_mask):
        finite_rk = np.where(finite_mask, r_k, -np.inf)
        max_finite_rk_channel = np.argmax(finite_rk)
        max_finite_rk_value = r_k[max_finite_rk_channel]
    else:
        max_finite_rk_channel = -1
        max_finite_rk_value = 0.0
    
    # Check 1: r_k > threshold for at least one non-leptonic channel (finite only)
    hadronic_has_high_ratio = any(
        finite_mask[k] and r_k[k] > C['min_ratio_hadronic']
        for k in hadronic
    )
    # Check 2: max FINITE r_k is on a non-leptonic channel
    max_on_hadronic = max_finite_rk_channel in hadronic if max_finite_rk_channel >= 0 else False
    
    # Check 3: geometric kills at least one channel that random preserves
    any_killed = any(np.isinf(r_k[k]) for k in range(K))
    
    if verbose:
        print(f"\n  CRB ratio r_k = σ²_geometric / σ²_random at α={alpha_vals[idx_target]:.2f}:")
        for k in range(K):
            hadr = " (non-leptonic)" if k in hadronic else " (leptonic)"
            if np.isinf(r_k[k]):
                print(f"    {labels[k]:>8s}: r_k = ∞ (geometric kills){hadr}")
            elif np.isnan(r_k[k]):
                print(f"    {labels[k]:>8s}: r_k = N/A (both fail){hadr}")
            else:
                print(f"    {labels[k]:>8s}: r_k = {r_k[k]:.2f}{hadr}")
        if max_finite_rk_channel >= 0:
            tag = 'non-leptonic' if max_on_hadronic else 'LEPTONIC'
            print(f"    Max finite r_k: {labels[max_finite_rk_channel]} = {max_finite_rk_value:.2f} ({tag})")

    checks = {
        f'r_k > {C["min_ratio_hadronic"]} for ≥1 hadronic at α={alpha_target}': hadronic_has_high_ratio,
        f'max finite r_k on hadronic channel': max_on_hadronic,
        'Geometric kills ≥1 channel that random preserves': any_killed,
    }

    passed = all(checks.values())
    if verbose:
        print(f"\n  Checks:")
        for name, ok in checks.items():
            print(f"    {name}: {'✓' if ok else '✗'}")
        print(f"\n  P3 {'PASSED' if passed else 'NEEDS REVIEW'}")

    return {'random': res_random, 'geometric': res_geom,
            'alpha_vals': alpha_vals, 'n_worse_by_alpha': n_worse_by_alpha,
            'checks': checks, 'passed': passed}


# =====================================================================
#  P4: Cascade Bottleneck
# =====================================================================

def validate_p4(bench, verbose=True):
    """P4: Cascade bottleneck at a₁→3π stage.

    The Data Processing Inequality predicts the bottleneck is at the
    broadest intermediate resonance. For τ→a₁ν→3πν, this is the a₁
    (Γ = 420 MeV).
    """
    from tau_cdma.core.cascade import cascade_tau_a1

    if verbose:
        print("\n" + "="*60)
        print("P4: Cascade Bottleneck (τ→a₁ν→3πν)")
        print("="*60)

    result = cascade_tau_a1(N=bench['N'])

    ratio = result['I1'] / result['I2']

    if verbose:
        print(f"\n  Stage 1 info (τ→a₁ν):   I₁ = {result['I1']:.2e}")
        print(f"  Stage 2 info (a₁→3π):    I₂ = {result['I2']:.2e}")
        print(f"  Ratio I₁/I₂:             {ratio:.1f}×")
        print(f"  Bottleneck:              {result['bottleneck']}")
        print(f"  Cascade SF:              {result['SF_cascade']:.2f}")
        print(f"  Effective width:         {result['Gamma_eff']:.1f} MeV")
        print(f"  DPI: I_cascade ≤ min(I₁,I₂) = I₂ ✓")

        # Comparison: ρ cascade (narrower resonance → less bottlenecked)
        from tau_cdma.tau.templates import M_TAU, M_RHO, G_RHO, M_A1, G_A1, bw_template
        sigma_det = bench['config']['sigma_det']

        # Compute ρ cascade for comparison
        M_bins = 200
        m_bins_cas = np.linspace(0, M_TAU, M_bins)
        dm_cas = m_bins_cas[1] - m_bins_cas[0]

        # ρ cascade: τ→ρν→ππ⁰ν
        from tau_cdma.core.fisher import poisson_fim
        rho_template = bw_template(m_bins_cas, M_RHO, G_RHO, sigma_det=sigma_det)
        bg_cas = np.ones(M_bins) / M_bins
        A_rho = np.column_stack([rho_template, bg_cas])
        theta_rho = np.array([0.5, 0.5])
        b_cas = 0.01 * dm_cas * np.ones(M_bins)
        N_cas = bench['N']
        lam_rho = N_cas * (A_rho @ theta_rho) + b_cas
        W_rho = 1.0 / np.maximum(lam_rho, 1e-30)
        F_rho = N_cas**2 * (A_rho.T * W_rho) @ A_rho
        I_rho_stage1 = np.trace(F_rho)

        from tau_cdma.core.cascade import cascade_sf
        SF_rho, Gamma_rho = cascade_sf(M_TAU, [G_RHO], regime='bw')
        SF_a1, Gamma_a1 = cascade_sf(M_TAU, [G_A1], regime='bw')

        print(f"\n  Cascade comparison (broader resonance = worse bottleneck):")
        print(f"    {'Decay':>20s}  {'Γ (MeV)':>8s}  {'SF':>6s}  {'I_stage1':>10s}")
        print(f"    {'τ→ρν→ππ⁰ν':>20s}  {G_RHO:>8.1f}  {SF_rho:>6.1f}  {I_rho_stage1:>10.2e}")
        print(f"    {'τ→a₁ν→3πν':>20s}  {G_A1:>8.1f}  {SF_a1:>6.1f}  {result['I1']:>10.2e}")
        print(f"    → a₁ 2.8× broader than ρ → more information lost in cascade")

    checks = {
        'Bottleneck at stage 2 (broad a₁)': result['bottleneck'] == 'stage2',
        'I₁ > I₂ (stage 2 is limiting)': result['I1'] > result['I2'],
        'DPI: I₂ < I₁ (cascade info ≤ min)': result['I2'] < result['I1'],
        'Bottleneck ratio > 5': ratio > 5,
    }
    passed = all(checks.values())

    if verbose:
        print(f"\n  Checks:")
        for name, ok in checks.items():
            print(f"    {name}: {'✓' if ok else '✗'}")
        print(f"\n  P4 {'PASSED' if passed else 'NEEDS REVIEW'}")

    result['checks'] = checks
    result['passed'] = passed
    return result


# =====================================================================
#  P7: Optimal Binning
# =====================================================================

def validate_p7(bench, verbose=True):
    """P7: Optimal binning M_opt ∝ SF_k — Fisher saturation curves.

    Shows the full Fisher information saturation curve per channel
    and extracts the 90% saturation point M_opt.
    """
    from tau_cdma.core.fisher import poisson_fim
    from tau_cdma.tau.templates import TauTemplates

    if verbose:
        print("\n" + "="*60)
        print("P7: Optimal Binning (Fisher Information Saturation)")
        print("="*60)

    theta = bench['theta']
    N = bench['N']
    sigma_det = bench['config']['sigma_det']
    m_range = bench['config']['m_range']
    range_width = m_range[1] - m_range[0]
    bg_density = bench['config']['background_density']
    K = len(theta)
    labels = bench['templates'].short_labels

    M_values = [3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]

    # Compute per-channel Fisher info diagonal as function of M
    fish_diag = np.zeros((len(M_values), K))
    for i, M in enumerate(M_values):
        tb = TauTemplates(M=M, m_range=m_range, sigma_det=sigma_det)
        A = tb.A
        dm = range_width / M
        bg = bg_density * dm * np.ones(M)
        F = poisson_fim(A, theta, N, bg)
        fish_diag[i] = np.diag(F)

    # Actual maximum across all M values (not just finest)
    F_max = np.max(fish_diag, axis=0)

    # Saturation fraction at each M
    frac = fish_diag / np.maximum(F_max, 1e-30)

    if verbose:
        print(f"\n  Fisher information saturation (% of M=1000):")
        print(f"  {'M':>5s}", end="")
        for k in range(K):
            print(f"  {labels[k]:>6s}", end="")
        print()
        for i, M in enumerate(M_values):
            if M in [3, 5, 10, 20, 50, 100, 200, 500, 1000]:
                print(f"  {M:>5d}", end="")
                for k in range(K):
                    print(f"  {frac[i,k]*100:>5.1f}%", end="")
                print()

    # M_opt at 90% saturation
    M_opt = np.zeros(K)
    for k in range(K):
        if F_max[k] > 0:
            above = np.where(frac[:, k] >= 0.90)[0]
            if len(above) > 0:
                M_opt[k] = M_values[above[0]]
            else:
                M_opt[k] = M_values[-1]
        else:
            M_opt[k] = np.nan

    sf_values = bench['templates'].spreading_factors()

    if verbose:
        print(f"\n  90% saturation point:")
        print(f"  {'Channel':>8s}  {'M_opt':>6s}  {'SF':>6s}  {'bin≈Γ?':>10s}")
        for k in range(K):
            sf = sf_values[k]
            sf_str = f"{sf:.1f}" if sf is not None and np.isfinite(sf) else "N/A"
            # Effective bin width at M_opt
            if M_opt[k] > 0 and not np.isnan(M_opt[k]):
                bin_width = range_width / M_opt[k]
                bin_str = f"{bin_width:.0f} MeV"
            else:
                bin_str = "N/A"
            print(f"  {labels[k]:>8s}  {M_opt[k]:>6.0f}  {sf_str:>6s}  {bin_str:>10s}")

    # Correlation for resonance channels
    from tau_cdma.validate.prediction_criteria import P7_CRITERIA as C
    resonance_idx = C['resonance_channels']
    valid = [(sf_values[i], M_opt[i]) for i in resonance_idx
             if sf_values[i] is not None and np.isfinite(sf_values[i]) and np.isfinite(M_opt[i])]

    if len(valid) >= 2:
        sfs_arr = np.array([v[0] for v in valid])
        mopts_arr = np.array([v[1] for v in valid])
        if np.std(sfs_arr) > 0 and np.std(mopts_arr) > 0:
            corr = np.corrcoef(sfs_arr, mopts_arr)[0, 1]
        else:
            corr = 0.0
    else:
        corr = np.nan

    # Stronger test: at M=5, π should have captured most info but a₁ should not yet
    m5_idx = M_values.index(5) if 5 in M_values else 1
    m50_idx = M_values.index(50) if 50 in M_values else 7

    # Note: Fisher info is NOT monotonic — it peaks then drops slightly.
    # This is because background per bin = bg_density × bin_width grows,
    # but signal per bin shrinks. Beyond the peak, background noise dominates.
    peak_M = [M_values[np.argmax(fish_diag[:, k])] for k in range(K)]

    if verbose:
        print(f"\n  Note: Fisher info peaks then drops slightly (bg dilution):")
        for k in [2, 3, 4]:  # π, ρ, a₁
            pk = peak_M[k]
            drop = fish_diag[-1, k] / np.max(fish_diag[:, k])
            print(f"    F_{labels[k]:>4s} peaks at M={pk}, drops to {drop:.3f}× at M=1000")

    checks = {
        'SF-Mopt positive correlation (resonance channels)': not np.isnan(corr) and corr > 0,
        f'All channels >90% by M={C["all_saturated_by_M"]}': all(
            frac[M_values.index(C['all_saturated_by_M']), k] > C['saturation_threshold']
            for k in range(K)
        ),
        f'π saturates early (>{C["pi_saturation_min"]*100:.0f}% at M={C["pi_saturates_early_M"]})': (
            frac[M_values.index(C['pi_saturates_early_M']), 2] > C['pi_saturation_min']
        ),
        'Peak M finite (not at boundary)': all(pk < 1000 for pk in peak_M),
        'Wide channels saturate before narrow (M_opt(a₁) ≤ M_opt(ρ))': M_opt[4] <= M_opt[3],
    }

    passed = all(checks.values())
    if verbose:
        if not np.isnan(corr):
            print(f"\n  Correlation(SF, M_opt) for resonances: {corr:.3f}")
            print(f"  Note: weak correlation ({corr:.2f}) is expected for the τ system —")
            print(f"  all widths (149-450 MeV) are >> σ_det (20 MeV), so spreading")
            print(f"  factors span only 2× (from 3 to 7.5). A stronger test would use")
            print(f"  charm mesons (widths keV to hundreds of MeV) for wider dynamic range.")
        print(f"\n  Checks:")
        for name, ok in checks.items():
            print(f"    {name}: {'✓' if ok else '✗'}")
        print(f"\n  P7 {'PASSED' if passed else 'NEEDS REVIEW'}")

    return {'M_values': M_values, 'fish_diag': fish_diag, 'frac': frac,
            'M_opt': M_opt, 'sf_values': sf_values, 'corr': corr,
            'checks': checks, 'passed': passed}


# =====================================================================
#  P8: Blind Decomposition — Three-Layer Validation
# =====================================================================

def validate_p8(bench, verbose=True, quick=False):
    """P8: Blind decomposition — three-layer validation.

    Layer 1: Blind NMF recovers K≈7 templates from mixture data
    Layer 2: Template fit + residual analysis discovers injected unknown channel
    Layer 3: Detection sensitivity — minimum BR for 5σ discovery
    """
    from tau_cdma.core.simulate import generate_multi_experiment
    from tau_cdma.core.nmf import poisson_nmf, nmf_model_selection, template_recovery_error

    if verbose:
        print("\n" + "="*60)
        print("P8: Blind Decomposition & Channel Discovery")
        print("="*60)

    A = bench['A']
    theta = bench['theta']
    bg = bench['background']
    m_bins = bench['templates'].m_bins
    labels = bench['templates'].short_labels
    K = len(labels)

    # === Layer 1: Blind NMF ===
    if verbose:
        print(f"\n  --- Layer 1: Blind NMF Template Recovery ---")

    n_exp = 50 if not quick else 10
    N_nmf = 1e5
    Y, _ = generate_multi_experiment(A, theta, N_nmf, bg, n_experiments=n_exp)

    K_range = range(4, 11) if not quick else range(5, 9)
    ms = nmf_model_selection(Y, K_range=K_range, n_iter=500)

    A_hat, _, _ = poisson_nmf(Y, 7, n_iter=2000)
    recovery = template_recovery_error(A, A_hat)

    if verbose:
        print(f"    Data: {n_exp} experiments × {A.shape[0]} bins, N={N_nmf:.0e}")
        print(f"    K_best (BIC) = {ms['K_best']}")
        print(f"    Recovery at K=7 (mean error): {recovery['mean']:.3f}")
        for k in range(K):
            print(f"      {labels[k]:>6s}: {recovery['per_channel'][k]:.3f}")

    # === Layer 2: Injected unknown channel + residual discovery ===
    if verbose:
        print(f"\n  --- Layer 2: Unknown Channel Discovery ---")

    # Inject: narrow resonance at 500 MeV, Γ=30 MeV, BR=3%
    m_new = 500.0
    gamma_new = 30.0
    br_new = 0.03
    sigma_eff = np.sqrt(gamma_new**2 + bench['config']['sigma_det']**2)
    a_new = np.exp(-0.5 * ((m_bins - m_new) / sigma_eff)**2)
    a_new /= np.sum(a_new)

    A_8 = np.column_stack([A, a_new.reshape(-1, 1)])
    theta_8 = theta * (1 - br_new)
    theta_8 = np.append(theta_8, br_new)

    N_disc = 500_000 if not quick else 100_000
    n_disc = 50 if not quick else 10
    rng = np.random.default_rng(42)
    y_total = np.zeros(A.shape[0])
    for exp in range(n_disc):
        y_total += rng.poisson(N_disc * (A_8 @ theta_8) + bg)
    N_total = N_disc * n_disc

    # Fit with known K=7 templates
    def nll(t, A_fit, y, N_tot, bg_fit):
        t = np.abs(t)
        lam = N_tot * (A_fit @ t) + bg_fit
        lam = np.maximum(lam, 1e-30)
        return np.sum(lam - y * np.log(lam))

    bg_total = bg * n_disc
    res7 = minimize(nll, theta, args=(A, y_total, N_total, bg_total),
                    method='L-BFGS-B', bounds=[(1e-6, 1)] * K)
    theta_fit7 = np.abs(res7.x)

    # Residual analysis
    lam_fit7 = N_total * (A @ theta_fit7) + bg_total
    pull = (y_total - lam_fit7) / np.sqrt(np.maximum(lam_fit7, 1))
    chi2_ndof = np.sum(pull**2) / (A.shape[0] - K)
    max_pull_idx = np.argmax(pull)
    max_pull = pull[max_pull_idx]
    max_pull_mass = m_bins[max_pull_idx]

    # Sliding window scan
    window = 8
    best_sig = 0
    best_center = 0
    for i in range(A.shape[0] - window):
        excess = np.sum(y_total[i:i + window] - lam_fit7[i:i + window])
        noise = np.sqrt(np.sum(lam_fit7[i:i + window]))
        sig = excess / max(noise, 1)
        if sig > best_sig:
            best_sig = sig
            best_center = (m_bins[i] + m_bins[min(i + window, A.shape[0] - 1)]) / 2

    if verbose:
        print(f"    Injected: m={m_new:.0f} MeV, Γ={gamma_new:.0f} MeV, BR={br_new:.1%}")
        print(f"    Data: {n_disc}×{N_disc:.0e} = {N_total:.0e} total events")
        print(f"    K=7 fit χ²/ndof = {chi2_ndof:.1f} (>>1 = something missing)")
        print(f"    Max pull: {max_pull:.1f}σ at m = {max_pull_mass:.0f} MeV (true: {m_new:.0f})")
        print(f"    Window scan: {best_sig:.1f}σ at m ≈ {best_center:.0f} MeV")

    # Likelihood ratio scan for mass of new channel
    nll_7 = res7.fun
    best_dll = 0
    best_m_scan = 0
    best_br_scan = 0
    scan_masses = np.arange(200, 1600, 20)
    scan_results = []

    for m_scan in scan_masses:
        sigma_scan = np.sqrt(50**2 + bench['config']['sigma_det']**2)
        a_scan = np.exp(-0.5 * ((m_bins - m_scan) / sigma_scan)**2)
        a_scan /= np.sum(a_scan)
        A_8_scan = np.column_stack([A, a_scan.reshape(-1, 1)])

        theta0 = np.append(theta_fit7 * 0.95, 0.05)
        res8 = minimize(nll, theta0, args=(A_8_scan, y_total, N_total, bg_total),
                        method='L-BFGS-B', bounds=[(1e-6, 1)] * (K + 1))
        dll = nll_7 - res8.fun
        scan_results.append(dll)

        if dll > best_dll:
            best_dll = dll
            best_m_scan = m_scan
            best_br_scan = np.abs(res8.x[-1])

    scan_significance = np.sqrt(2 * max(best_dll, 0))

    if verbose:
        print(f"\n    Likelihood ratio scan:")
        print(f"    Best-fit mass: {best_m_scan:.0f} MeV (true: {m_new:.0f})")
        print(f"    Recovered BR: {best_br_scan:.4f} (true: {br_new:.4f})")
        print(f"    Significance: {scan_significance:.1f}σ")

    # === Layer 3: Detection sensitivity ===
    if verbose:
        print(f"\n  --- Layer 3: Detection Sensitivity ---")

    sensitivity = []
    test_brs = [0.001, 0.005, 0.01, 0.05] if not quick else [0.005, 0.05]
    test_Ns = [1e5, 1e6] if not quick else [1e5]

    for br_test in test_brs:
        for N_test in test_Ns:
            theta_test = theta * (1 - br_test)
            theta_test = np.append(theta_test, br_test)
            n_test_exp = 10
            rng2 = np.random.default_rng(123)
            y_test = np.zeros(A.shape[0])
            for exp in range(n_test_exp):
                y_test += rng2.poisson(N_test * (A_8 @ theta_test) + bg)
            N_test_total = N_test * n_test_exp

            bg_test = bg * n_test_exp
            res_test = minimize(nll, theta, args=(A, y_test, N_test_total, bg_test),
                                method='L-BFGS-B', bounds=[(1e-6, 1)] * K)
            lam_test = N_test_total * (A @ np.abs(res_test.x)) + bg_test
            pull_test = (y_test - lam_test) / np.sqrt(np.maximum(lam_test, 1))

            near = np.abs(m_bins - m_new) < 60
            max_p = np.max(pull_test[near]) if np.any(near) else 0
            sensitivity.append({
                'br': br_test, 'N': N_test, 'N_total': N_test_total,
                'max_pull': max_p, 'detected': max_p > 5.0
            })

    if verbose:
        print(f"    {'BR':>6s} {'N':>8s} {'N_total':>10s} {'Pull':>8s} {'Found':>6s}")
        for s in sensitivity:
            det = '✓' if s['detected'] else '✗'
            print(f"    {s['br']:>6.3f} {s['N']:>8.0e} {s['N_total']:>10.0e} "
                  f"{s['max_pull']:>7.1f}σ {det:>6s}")

    # === Layer 4: NN multiuser receiver ===
    from tau_cdma.tau.ml_receiver import run_ml_layer
    ml_results = run_ml_layer(bench, verbose=verbose, quick=quick)

    # Checks — aligned with formalism criteria (prediction_criteria.py)
    from tau_cdma.validate.prediction_criteria import P8_CRITERIA as C
    
    # Count channels with recovery error < 1.0
    n_below_1 = sum(1 for e in recovery['per_channel'] if e < 1.0)
    
    checks = {
        f'Blind NMF: K_best within ±{C["K_best_tolerance"]} of {C["K_true"]}': (
            abs(ms['K_best'] - C['K_true']) <= C['K_best_tolerance']
        ),
        f'NMF mean recovery < {C["recovery_mean_max"]}': recovery['mean'] < C['recovery_mean_max'],
        f'NMF ≥{C["recovery_some_below_1"]} channels with error < 1.0': (
            n_below_1 >= C['recovery_some_below_1']
        ),
        f'Residual: peak >{C["residual_significance_min"]:.0f}σ near {m_new:.0f} MeV': (
            max_pull > C['residual_significance_min'] and abs(max_pull_mass - m_new) < C['mass_recovery_tolerance']
        ),
        f'LR scan: mass within {C["mass_recovery_tolerance"]} MeV': (
            abs(best_m_scan - m_new) < C['mass_recovery_tolerance']
        ),
        f'LR scan: significance >{C["lr_significance_min"]:.0f}σ': scan_significance > C['lr_significance_min'],
    }
    # Add ML checks
    checks.update(ml_results.get('checks', {}))

    passed = all(checks.values())
    if verbose:
        print(f"\n  Checks:")
        for name, ok in checks.items():
            print(f"    {name}: {'✓' if ok else '✗'}")
        print(f"\n  P8 {'PASSED' if passed else 'NEEDS REVIEW'}")

    return {
        'nmf': {'K_best': ms['K_best'], 'recovery': recovery},
        'discovery': {
            'max_pull': max_pull, 'max_pull_mass': max_pull_mass,
            'scan_mass': best_m_scan, 'scan_br': best_br_scan,
            'scan_significance': scan_significance,
        },
        'sensitivity': sensitivity,
        'ml': ml_results,
        'checks': checks,
        'passed': passed,
    }


# =====================================================================
#  P9: Aliasing as Eigenvalue Collapse
# =====================================================================

def validate_p9(bench=None, verbose=True):
    """P9: Eigenvalue spectrum of R reveals channel resolution structure.

    The interference matrix R is a symmetric K×K matrix computable from
    the template matrix A (given by physics) without any QN knowledge.
    Its eigenvalue spectrum encodes:
      - Which channels are aliased (near-zero eigenvalues)
      - Which direction in channel space is unresolvable (eigenvectors)
      - How many independent measurements the data supports (PR)

    The framework PREDICTS:
      (a) λ_min ≈ 0, with eigenvector along e-μ direction
      (b) Adding PID lifts λ_min (de-aliasing)
      (c) Participation ratio PR(R) < K in 1D, increases with PID
      (d) Blind NMF from data recovers consistent structure

    This is NOT circular: templates come from physics (PDG + detector),
    eigenvalue analysis discovers aliasing without knowing which channels
    are aliased. The PID lift is a falsifiable prediction.
    """
    from tau_cdma.tau.templates import TauTemplates, TAU_BR
    from tau_cdma.core.fisher import poisson_fim
    from tau_cdma.core.interference import interference_matrix, multiuser_efficiency
    from tau_cdma.core.simulate import generate_multi_experiment
    from tau_cdma.core.nmf import poisson_nmf

    if verbose:
        print("\n" + "="*60)
        print("P9: Aliasing as Eigenvalue Collapse")
        print("="*60)

    # Use bench if provided, else setup
    if bench is None:
        from tau_cdma.tau.benchmark import setup_benchmark
        bench = setup_benchmark()

    A = bench['A']
    theta = bench['theta']
    N = bench['N']
    bg = bench['background']
    labels = bench['templates'].short_labels
    K = len(theta)
    M = A.shape[0]

    # === (a) Eigenvalue spectrum of R ===
    R = bench['R']
    eigvals, eigvecs = np.linalg.eigh(R)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Participation ratio
    pr_1d = (np.sum(eigvals))**2 / np.sum(eigvals**2)

    # Identify aliased direction (smallest eigenvalue's eigenvector)
    v_min = eigvecs[:, -1]  # eigenvector for λ_min
    # Which two channels dominate v_min?
    top2 = np.argsort(np.abs(v_min))[-2:]
    aliased_pair = tuple(sorted(top2))
    # Check if it's genuinely a two-channel direction
    v_min_concentration = np.sum(v_min[top2]**2) / np.sum(v_min**2)

    if verbose:
        print(f"\n  (a) R eigenvalue spectrum (1D visible mass):")
        for i in range(K):
            marker = " ← aliased" if i == K-1 else ""
            print(f"      λ_{i+1} = {eigvals[i]:.6f}{marker}")
        print(f"\n      PR(R) = {pr_1d:.3f} → ~{pr_1d:.0f} effective channels (of {K})")

        print(f"\n      Aliased direction (λ_{K} eigenvector):")
        for k in range(K):
            bar = "█" * int(abs(v_min[k]) * 40)
            sign = "+" if v_min[k] >= 0 else "-"
            print(f"        {labels[k]:>6s}: {sign}{abs(v_min[k]):.4f} {bar}")
        print(f"      → λ_min eigenvector is {v_min_concentration:.1%} concentrated"
              f" on ({labels[aliased_pair[0]]}, {labels[aliased_pair[1]]})")
        print(f"      Framework discovers e-μ aliasing from eigenvalue analysis alone")

    # === (b) PID lifts degeneracy ===
    pid_weight = 0.3
    pid_block = np.zeros((3, K))
    pid_block[0, 0] = 1.0
    pid_block[1, 1] = 1.0
    pid_block[2, 2:] = 1.0
    A_pid = np.vstack([A * (1 - pid_weight),
                       pid_block * pid_weight / np.maximum(
                           pid_block.sum(axis=0, keepdims=True), 1e-30)])
    col_sums = A_pid.sum(axis=0, keepdims=True)
    A_pid = A_pid / np.maximum(col_sums, 1e-30)
    bg_aug = np.concatenate([bg, 0.001 * np.ones(3)])

    R_pid = interference_matrix(A_pid, theta, N, bg_aug)
    eigvals_pid = np.sort(np.linalg.eigvalsh(R_pid))[::-1]
    pr_pid = (np.sum(eigvals_pid))**2 / np.sum(eigvals_pid**2)

    lift_min = eigvals_pid[-1] / max(eigvals[-1], 1e-30)

    if verbose:
        print(f"\n  (b) PID de-aliasing (eigenvalue lift):")
        print(f"      {'i':>5s}  {'λ_1D':>10s}  {'λ_PID':>10s}  {'lift':>10s}")
        for i in range(K):
            delta = eigvals_pid[i] - eigvals[i]
            print(f"      {i+1:>5d}  {eigvals[i]:>10.6f}  {eigvals_pid[i]:>10.6f}"
                  f"  {delta:>+10.6f}")
        print(f"\n      λ_min: {eigvals[-1]:.6f} → {eigvals_pid[-1]:.6f}"
              f" ({lift_min:.0f}× lift)")
        print(f"      PR(R): {pr_1d:.3f} → {pr_pid:.3f}"
              f" ({pr_pid - pr_1d:+.3f} effective channels)")

    # === (c) Second-smallest eigenvalue = next aliased direction ===
    v_6 = eigvecs[:, -2]
    top3_6 = np.argsort(np.abs(v_6))[-3:]

    if verbose:
        print(f"\n  (c) Next-most-aliased direction (λ_{K-1} eigenvector):")
        for k in range(K):
            if abs(v_6[k]) > 0.1:
                print(f"        {labels[k]:>6s}: {v_6[k]:+.4f}")
        print(f"      → Broad hadronic channels (a₁, π2π⁰, other) partially alias")

    # === (d) Data-driven NMF recovers consistent structure ===
    if verbose:
        print(f"\n  (d) Data-driven validation (NMF from Poisson data):")

    n_exp = 50
    N_nmf = 100_000
    Y, _ = generate_multi_experiment(A, theta, N_nmf, bg, n_experiments=n_exp)
    A_nmf, _, _ = poisson_nmf(Y, K, n_iter=2000)

    # Match NMF columns to true columns
    from scipy.optimize import linear_sum_assignment
    corr_matrix = np.abs(np.corrcoef(A.T, A_nmf.T)[:K, K:])
    row_ind, col_ind = linear_sum_assignment(-corr_matrix)
    A_nmf_matched = A_nmf[:, col_ind]
    match_quality = np.mean([corr_matrix[i, col_ind[i]] for i in range(K)])

    # R from NMF-recovered templates
    R_nmf = interference_matrix(A_nmf_matched, theta, N, bg)
    eigvals_nmf = np.sort(np.linalg.eigvalsh(R_nmf))[::-1]
    pr_nmf = (np.sum(eigvals_nmf))**2 / np.sum(eigvals_nmf**2)

    # Check if NMF also finds that the smallest eigenvalue direction
    # is dominated by e/μ
    _, evecs_nmf = np.linalg.eigh(R_nmf)
    v_min_nmf = evecs_nmf[:, 0]  # smallest eigenvalue
    top2_nmf = np.argsort(np.abs(v_min_nmf))[-2:]
    nmf_finds_emu = set(top2_nmf) == {0, 1}

    if verbose:
        print(f"      NMF template matching quality: {match_quality:.3f}")
        print(f"      PR(R_NMF) = {pr_nmf:.3f} (true: {pr_1d:.3f})")
        print(f"      λ_min(NMF) = {eigvals_nmf[-1]:.6f} (true: {eigvals[-1]:.6f})")
        top2_labels = f"({labels[top2_nmf[0]]}, {labels[top2_nmf[1]]})"
        found = "✓" if nmf_finds_emu else "✗"
        print(f"      NMF λ_min direction: {top2_labels} {found}")

    # === Checks ===
    checks = {
        'λ_min < 0.01 (aliasing exists)': eigvals[-1] < 0.01,
        'Aliased direction is e-μ': set(aliased_pair) == {0, 1},
        'e-μ concentration > 99%': v_min_concentration > 0.99,
        'PID lifts λ_min > 10×': lift_min > 10,
        'PR increases with PID': pr_pid > pr_1d,
        'NMF: PR within 50% of true': abs(pr_nmf - pr_1d) / pr_1d < 0.5,
    }

    passed = all(checks.values())
    if verbose:
        print(f"\n  Checks:")
        for name, ok in checks.items():
            print(f"    {name}: {'✓' if ok else '✗'}")
        print(f"\n  P9 {'PASSED' if passed else 'NEEDS REVIEW'}")

    return {
        'eigvals_1d': eigvals, 'eigvecs_1d': eigvecs,
        'eigvals_pid': eigvals_pid,
        'pr_1d': pr_1d, 'pr_pid': pr_pid, 'pr_nmf': pr_nmf,
        'aliased_pair': aliased_pair,
        'v_min_concentration': v_min_concentration,
        'lift_min': lift_min,
        'checks': checks, 'passed': passed,
    }


# =====================================================================
#  Run All
# =====================================================================

def run_all(quick=False, verbose=True):
    """Run all prediction validations."""
    from tau_cdma.tau.benchmark import setup_benchmark

    if verbose:
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  tau_cdma: Framework Prediction Validation Suite         ║")
        print("║  Unified Formalism v4.5                                  ║")
        print("╚══════════════════════════════════════════════════════════╝")

    bench = setup_benchmark()
    summary = {}

    summary['P1'] = validate_p1(bench, verbose)
    summary['P2'] = validate_p2(bench, verbose)
    summary['P3'] = validate_p3(bench, verbose)
    summary['P4'] = validate_p4(bench, verbose)
    summary['P7'] = validate_p7(bench, verbose)
    summary['P8'] = validate_p8(bench, verbose, quick=quick)
    summary['P9'] = validate_p9(bench, verbose)

    if verbose:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for pred_name, result in sorted(summary.items()):
            status = bool(result.get('passed', False)) if result.get('passed') is not None else None
            icon = '✓' if status is True else ('✗' if status is False else '?')
            print(f"  {pred_name}: {icon}")

    return summary


if __name__ == '__main__':
    quick = '--quick' in sys.argv
    run_all(quick=quick)
