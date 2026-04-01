#!/usr/bin/env python3
"""
Generate all publication figures for Paper 1, CPC Paper 2, and the Extension paper.

Usage:
    python generate_paper_figures.py [output_root]

Output structure:
    output_root/
        paper1_figures/     — 9 figures for the EPJ C letter
        cpc_figures/        — 8 figures for the CPC companion
        extension_figures/  — 4 figures for the optimal design extension
"""

import sys
import os
import numpy as np

# Ensure tau_cdma is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def generate_paper1_figures(output_dir):
    """Generate all 9 Paper 1 figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    from tau_cdma.tau.benchmark import setup_benchmark
    from tau_cdma.core.shannon import classification_mi
    from tau_cdma.core.cascade import cascade_tau_a1
    from tau_cdma.core.robust import dominance_margin
    from tau_cdma.core.aliasing import aliasing_sweep
    from tau_cdma.heavy_ion.centrality import momentum_sweep, centrality_sweep
    from tau_cdma.heavy_ion.bethe_bloch import bethe_bloch
    
    bench = setup_benchmark()
    A, theta, N = bench['A'], bench['theta'], bench['N']
    eta, R, eigvals = bench['eta'], bench['R'], bench['eigvals']
    labels = bench['templates'].short_labels
    
    # ── Fig 1: τ templates + eigenvalues + eigenvector (3 panels) ──
    print("  Fig 1: τ templates + eigenvalues + eigenvector")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    
    m_bins = bench['templates'].m_bins
    for k in range(A.shape[1]):
        ax1.plot(m_bins, theta[k] * A[:, k], label=labels[k], linewidth=1.2)
    ax1.set_xlabel('Visible mass [MeV]')
    ax1.set_ylabel('θ_k · A_mk')
    ax1.set_title('(a) Weighted templates')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    eigR = np.linalg.eigvalsh(R)
    colors = ['steelblue'] * 6 + ['red']
    ax2.bar(range(1, 8), sorted(eigR, reverse=True), color=colors)
    ax2.set_yscale('log')
    ax2.set_xlabel('Eigenvalue index')
    ax2.set_ylabel('λ_i(R)')
    ax2.set_title(f'(b) Eigenvalue spectrum of R')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Aliased eigenvector (smallest eigenvalue)
    eigvals_R, eigvecs_R = np.linalg.eigh(R)
    v_min = np.abs(eigvecs_R[:, 0])
    ax3.bar(range(7), v_min, color=['red' if x > 0.1 else 'steelblue' for x in v_min],
            tick_label=[l[:3] for l in labels])
    ax3.set_ylabel('|v_min|')
    ax3.set_title('(c) Most aliased eigenvector')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_tau_benchmark.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ── Fig 2: η_K vs momentum (ALICE) — LOG SCALE ──
    print("  Fig 2: η_K vs momentum (log scale)")
    res = momentum_sweep(compute_tof=True)
    p = res['p_grid']
    eta_hi = res['eta']
    eta_joint = res['eta_joint']
    p_piK, p_Kp = 0.9961, 2.3736
    species_names = res['species_names']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for k in range(3):
        ax1.semilogy(p, eta_hi[:, k], label=f'{species_names[k]} (TPC)', linewidth=1.5)
        ax1.semilogy(p, eta_joint[:, k], '--', label=f'{species_names[k]} (TPC+TOF)',
                     linewidth=1.5, alpha=0.7)
    ax1.axvline(p_piK, color='gray', ls=':', alpha=0.5)
    ax1.axvline(p_Kp, color='gray', ls='-.', alpha=0.5)
    ax1.set_xlabel('Momentum p [GeV/c]')
    ax1.set_ylabel('Multiuser efficiency η_k')
    ax1.set_title('(a) All species')
    ax1.set_ylim(1e-7, 2)
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(p, eta_hi[:, 1], 'b-', label='K (TPC only)', linewidth=2)
    ax2.semilogy(p, eta_joint[:, 1], 'b--', label='K (TPC+TOF)', linewidth=2)
    ax2.axvline(p_piK, color='red', ls=':', alpha=0.7)
    ax2.axvline(p_Kp, color='red', ls='-.', alpha=0.7)
    idx_piK = np.argmin(np.abs(p - p_piK))
    ax2.annotate(f'η_K = {eta_hi[idx_piK,1]:.1e}\n(TPC)',
                 xy=(p_piK, max(eta_hi[idx_piK,1], 1e-7)), xytext=(1.3, 1e-5),
                 arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
    ax2.annotate(f'η_K = {eta_joint[idx_piK,1]:.3f}\n(TPC+TOF)',
                 xy=(p_piK, eta_joint[idx_piK,1]), xytext=(1.5, 0.3),
                 arrowprops=dict(arrowstyle='->', color='blue'), fontsize=9, color='blue')
    ax2.set_xlabel('Momentum p [GeV/c]')
    ax2.set_ylabel('Kaon multiuser efficiency η_K')
    ax2.set_title(f'(b) Kaon zoom — 633× TOF rescue')
    ax2.set_ylim(1e-7, 2)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_eta_K_momentum.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ── Fig 3: Eigenvalue collapse at BB crossings — LOG SCALE ──
    print("  Fig 3: eigenvalue collapse at crossings (log scale)")
    eigenvalues = res['eigenvalues']
    masses = [0.13957039, 0.493677, 0.93827209]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    p_fine = np.linspace(0.2, 5.0, 500)
    for m, name, color in zip(masses, ['π', 'K', 'p'], ['blue', 'orange', 'green']):
        dedx = [bethe_bloch(pi, m) for pi in p_fine]
        ax1.plot(p_fine, dedx, label=name, color=color, linewidth=1.5)
    ax1.axvline(p_piK, color='red', ls=':', alpha=0.7, label=f'π/K ({p_piK:.3f})')
    ax1.axvline(p_Kp, color='red', ls='-.', alpha=0.7, label=f'K/p ({p_Kp:.3f})')
    ax1.set_xlabel('Momentum p [GeV/c]')
    ax1.set_ylabel('dE/dx [a.u.]')
    ax1.set_title('(a) ALEPH Bethe-Bloch')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    eig_ratio = eigenvalues.min(axis=1) / eigenvalues.max(axis=1)
    ax2.semilogy(p, eig_ratio, 'k-', linewidth=1.5)
    ax2.axvline(p_piK, color='red', ls=':', alpha=0.7, label='π/K')
    ax2.axvline(p_Kp, color='red', ls='-.', alpha=0.7, label='K/p')
    idx_piK = np.argmin(np.abs(p - p_piK))
    idx_Kp = np.argmin(np.abs(p - p_Kp))
    ax2.annotate(f'{eig_ratio[idx_piK]:.1e}',
                 xy=(p_piK, max(eig_ratio[idx_piK], 1e-10)), xytext=(1.5, 1e-3),
                 arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
    ax2.annotate(f'{eig_ratio[idx_Kp]:.1e}',
                 xy=(p_Kp, max(eig_ratio[idx_Kp], 1e-10)), xytext=(3.2, 1e-3),
                 arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
    ax2.set_xlabel('Momentum p [GeV/c]')
    ax2.set_ylabel('λ_min / λ_max')
    ax2.set_title('(b) FIM eigenvalue ratio')
    ax2.set_ylim(1e-10, 2)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_eigenvalue_collapse.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ── Fig 4: MAP confusion + per-channel accuracy + η-accuracy ──
    print("  Fig 4: MAP collapse theorem")
    from tau_cdma.core.shannon import bayes_confusion
    conf = bayes_confusion(A, theta)
    C = conf['confusion']
    acc_per = conf['accuracy']
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    im = ax1.imshow(C, cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks(range(7)); ax1.set_yticks(range(7))
    ax1.set_xticklabels([l[:3] for l in labels], fontsize=8)
    ax1.set_yticklabels([l[:3] for l in labels], fontsize=8)
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')
    ax1.set_title('(a) MAP confusion matrix')
    plt.colorbar(im, ax=ax1, shrink=0.8)
    
    ax2.barh(range(7), acc_per * 100, color=['red' if a < 0.01 else 'steelblue' for a in acc_per])
    ax2.set_yticks(range(7)); ax2.set_yticklabels([l[:3] for l in labels], fontsize=8)
    ax2.set_xlabel('Per-channel accuracy [%]')
    ax2.set_title('(b) Per-channel accuracy')
    ax2.axvline(0, color='black', linewidth=0.5)
    
    ax3.scatter(eta, acc_per * 100, c='steelblue', s=60, zorder=5)
    for k in range(7):
        ax3.annotate(labels[k][:3], (eta[k], acc_per[k]*100), fontsize=7,
                     textcoords="offset points", xytext=(5, 5))
    ax3.set_xlabel('Multiuser efficiency η_k')
    ax3.set_ylabel('MAP accuracy [%]')
    ax3.set_title(f'(c) η vs accuracy (r = 0.84)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_map_collapse.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ── Fig 5: Correlation sign flip ──
    print("  Fig 5: correlation sign flip")
    M_vals = [5, 10, 20, 50, 100, 200, 500]
    corrs = []
    from scipy.optimize import minimize as sp_minimize
    for M_test in M_vals:
        config = {'M': M_test, 'm_range': (0, 1800), 'sigma_det': 20.0,
                  'N': 1000000, 'background_density': 0.01,
                  'theta': list(theta)}
        b_test = setup_benchmark(config)
        A_t, th_t, N_t, bg_t = b_test['A'], b_test['theta'], b_test['N'], b_test['background']
        lam = N_t * A_t @ th_t + bg_t
        rng = np.random.default_rng(42)
        theta_hats = []
        for _ in range(50):
            y = rng.poisson(lam)
            def neg_ll(th):
                th_full = np.append(th, 1 - th.sum())
                if np.any(th_full < 0) or np.any(th_full > 1):
                    return 1e12
                mu = N_t * A_t @ th_full + bg_t
                mu = np.maximum(mu, 1e-30)
                return -np.sum(y * np.log(mu) - mu)
            x0 = th_t[:-1] + rng.normal(0, 0.001, size=len(th_t)-1)
            result = sp_minimize(neg_ll, x0, method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-8})
            th_hat = np.append(result.x, 1 - result.x.sum())
            theta_hats.append(th_hat)
        theta_hats = np.array(theta_hats)
        if theta_hats.shape[0] > 10:
            corr_em = np.corrcoef(theta_hats[:, 0], theta_hats[:, 1])[0, 1]
        else:
            corr_em = 0
        corrs.append(corr_em)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(M_vals, corrs, 'bo-', linewidth=2, markersize=8)
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Number of bins M')
    ax.set_ylabel('corr(θ̂_e, θ̂_μ)')
    ax.set_title('Correlation sign flip: e-μ estimators')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_correlation_sign_flip.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ── Fig 6: Cascade bottleneck ──
    print("  Fig 6: cascade bottleneck")
    cas = cascade_tau_a1(N=N)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    stages = ['Stage 1\n(τ→a₁ν)', 'Stage 2\n(a₁→3π)']
    vals = [cas['I1'], cas['I2']]
    ratio = cas['I1'] / cas['I2']
    ax.bar(stages, vals, color=['steelblue', 'coral'], width=0.5)
    ax.set_ylabel('Fisher information')
    ax.set_title(f'Cascade bottleneck: I₁/I₂ = {ratio:.1f}×')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_cascade_bottleneck.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ── Fig 7: Receiver hierarchy ──
    print("  Fig 7: receiver hierarchy")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = ['MAP\n(1D)', '1D NN', '5D NN']
    accs = [46.4, 46.7, 76.1]
    colors_rh = ['steelblue', 'steelblue', 'coral']
    bars = ax.bar(methods, accs, color=colors_rh, width=0.5)
    ax.axhline(33.1, color='red', ls='--', label='Fano floor (33.1%)', linewidth=1.5)
    ax.set_ylabel('Overall accuracy [%]')
    ax.set_title('Receiver hierarchy')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc}%', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig7_receiver_hierarchy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ── Fig 8: Information budget ──
    print("  Fig 8: information budget")
    mi_tau = classification_mi(A, theta)
    
    from tau_cdma.heavy_ion.bethe_bloch import build_template_matrix
    theta_hi_vec = np.array([0.8475, 0.1186, 0.0339])
    A_cross, _ = build_template_matrix(0.9961, [0.13957039, 0.493677, 0.93827209],
                                        sigma=0.055, n_bins=100)
    mi_alice = classification_mi(A_cross, theta_hi_vec)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.bar(['H(K)', 'I(K;X)', 'Fano\nfloor'],
            [mi_tau['H_K'], mi_tau['MI'], mi_tau['fano_bound'] * np.log2(7-1)],
            color=['gray', 'steelblue', 'red'], width=0.5)
    ax1.set_ylabel('Bits')
    ax1.set_title(f'τ decay: I = {mi_tau["MI"]:.2f} of {mi_tau["H_K"]:.2f} bits')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(['H(K)', 'I(K;X)', 'Fano\nfloor'],
            [mi_alice['H_K'], mi_alice['MI'], mi_alice['fano_bound'] * np.log2(3-1)],
            color=['gray', 'steelblue', 'red'], width=0.5)
    ax2.set_ylabel('Bits')
    ax2.set_title(f'ALICE at crossing: I = {mi_alice["MI"]:.2f} of {mi_alice["H_K"]:.2f} bits')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig8_information_budget.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ── Fig 9: Central vs peripheral ──
    print("  Fig 9: central vs peripheral")
    # Use the momentum sweep results already computed
    p_grid = res['p_grid']
    eta_K_tpc = res['eta'][:, 1]
    crb_K = res['CRB'][:, 1]
    
    # Also compute at different N (centrality proxy)
    from tau_cdma.core.fisher import poisson_fim, crb as compute_crb
    from tau_cdma.core.interference import interference_matrix as int_mat, multiuser_efficiency as mu_eff
    
    N_central = 1600
    N_periph = 100
    A_ref, _ = build_template_matrix(1.5, [0.13957039, 0.493677, 0.93827209],
                                      sigma=0.055, n_bins=100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # (a) η_K vs p for different centralities  
    for N_c, label, ls in [(N_central, 'Central (N=1600)', '-'), 
                            (N_periph, 'Peripheral (N=100)', '--')]:
        eta_arr = []
        for pi in p_grid[::5]:
            Ai, _ = build_template_matrix(pi, [0.13957039, 0.493677, 0.93827209],
                                           sigma=0.055, n_bins=100)
            Ri = int_mat(Ai, theta_hi_vec, N_c)
            ei = mu_eff(Ri)
            eta_arr.append(ei[1])
        ax1.semilogy(p_grid[::5], eta_arr, ls, label=label, linewidth=1.5)
    ax1.set_xlabel('Momentum p [GeV/c]')
    ax1.set_ylabel('η_K')
    ax1.set_title('(a) η_K: worse in central')
    ax1.set_ylim(1e-7, 2)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # (b) CRB_K vs p for different centralities
    for N_c, label, ls in [(N_central, 'Central (N=1600)', '-'),
                            (N_periph, 'Peripheral (N=100)', '--')]:
        crb_arr = []
        for pi in p_grid[::5]:
            Ai, _ = build_template_matrix(pi, [0.13957039, 0.493677, 0.93827209],
                                           sigma=0.055, n_bins=100)
            Fi = poisson_fim(Ai, theta_hi_vec, N_c)
            ci = compute_crb(Fi)
            crb_arr.append(np.sqrt(ci[1]))
        ax2.semilogy(p_grid[::5], crb_arr, ls, label=label, linewidth=1.5)
    ax2.set_xlabel('Momentum p [GeV/c]')
    ax2.set_ylabel('√CRB_K')
    ax2.set_title('(b) CRB_K: better in central (16.8×)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig9_central_peripheral.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  All 9 Paper 1 figures saved to {output_dir}/")


def generate_cpc_figures(output_dir):
    """Generate all 8 CPC Paper 2 figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    from tau_cdma.tau.benchmark import setup_benchmark
    from tau_cdma.core.aliasing import aliasing_sweep
    from tau_cdma.core.erasure import erasure_sweep
    from tau_cdma.core.cascade import cascade_tau_a1
    from tau_cdma.validate.tau_predictions import validate_p7
    from tau_cdma.heavy_ion.centrality import momentum_sweep
    from tau_cdma.plotting import (plot_templates, plot_interference_matrix,
                                    plot_multiuser_efficiency, plot_eigenvalue_sweep,
                                    plot_erasure_comparison, plot_cascade_info,
                                    plot_fisher_vs_M)
    
    bench = setup_benchmark()
    labels = bench['templates'].short_labels
    
    # Fig 1: templates
    print("  CPC Fig 1: templates")
    plot_templates(bench, os.path.join(output_dir, 'fig1_templates.png'))
    
    # Fig 2: R matrix
    print("  CPC Fig 2: R matrix")
    plot_interference_matrix(bench, os.path.join(output_dir, 'fig2_R_matrix.png'))
    
    # Fig 3: η bar chart
    print("  CPC Fig 3: multiuser efficiency")
    plot_multiuser_efficiency(bench, os.path.join(output_dir, 'fig3_eta.png'))
    
    # Fig 4: eigenvalue sweep
    print("  CPC Fig 4: eigenvalue sweep")
    tb = bench['templates']
    M_values = [5, 10, 20, 50, 100, 200, 500]
    sweep = aliasing_sweep(tb, M_values, bench['theta'], bench['N'])
    plot_eigenvalue_sweep(sweep, os.path.join(output_dir, 'fig4_eigenvalues.png'))
    
    # Fig 5: erasure comparison
    print("  CPC Fig 5: erasure comparison")
    alpha_vals = np.linspace(0.3, 1.0, 15)
    er = erasure_sweep(bench['A'], bench['theta'], bench['N'], bench['background'],
                       alpha_vals, n_trials=20, mode='random')
    eg = erasure_sweep(bench['A'], bench['theta'], bench['N'], bench['background'],
                       alpha_vals, mode='geometric', m_bins=bench['templates'].m_bins)
    plot_erasure_comparison(er, eg, labels, os.path.join(output_dir, 'fig5_erasure.png'))
    
    # Fig 6: cascade
    print("  CPC Fig 6: cascade bottleneck")
    cas = cascade_tau_a1(N=bench['N'])
    plot_cascade_info(cas, os.path.join(output_dir, 'fig6_cascade.png'))
    
    # Fig 7: optimal binning
    print("  CPC Fig 7: optimal binning")
    p7 = validate_p7(bench, verbose=False)
    plot_fisher_vs_M(p7['M_values'], p7['fish_diag'], labels,
                     os.path.join(output_dir, 'fig7_optimal_binning.png'))
    
    # Fig 8: Heavy-ion eigenvalue collapse (LOG SCALE)
    print("  CPC Fig 8: heavy-ion eigenvalue collapse (log scale)")
    res = momentum_sweep(compute_tof=True)
    p = res['p_grid']
    eigenvalues = res['eigenvalues']
    eig_ratio = eigenvalues.min(axis=1) / eigenvalues.max(axis=1)
    p_piK, p_Kp = 0.9961, 2.3736
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(p, eig_ratio, 'k-', linewidth=1.5)
    ax.axvline(p_piK, color='red', ls=':', alpha=0.7, label=f'π/K ({p_piK:.3f} GeV/c)')
    ax.axvline(p_Kp, color='red', ls='-.', alpha=0.7, label=f'K/p ({p_Kp:.3f} GeV/c)')
    idx1 = np.argmin(np.abs(p - p_piK))
    idx2 = np.argmin(np.abs(p - p_Kp))
    ax.annotate(f'{eig_ratio[idx1]:.1e}',
                xy=(p_piK, max(eig_ratio[idx1], 1e-10)), xytext=(1.5, 1e-3),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
    ax.annotate(f'{eig_ratio[idx2]:.1e}',
                xy=(p_Kp, max(eig_ratio[idx2], 1e-10)), xytext=(3.2, 1e-3),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
    ax.set_xlabel('Momentum p [GeV/c]')
    ax.set_ylabel('λ_min / λ_max')
    ax.set_title('Eigenvalue collapse at Bethe-Bloch crossings')
    ax.set_ylim(1e-10, 2)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig8_eigenvalue_collapse_momentum.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  All 8 CPC figures saved to {output_dir}/")


def generate_extension_figures(output_dir):
    """Generate all 4 extension paper figures via run_optimal_design.py."""
    os.makedirs(output_dir, exist_ok=True)
    
    # The extension figures are generated by run_optimal_design.py
    # Import and run it
    import subprocess
    result = subprocess.run(
        [sys.executable, 'run_optimal_design.py', output_dir],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        print(f"  All 4 extension figures saved to {output_dir}/")
    else:
        print(f"  ERROR generating extension figures: {result.stderr[:200]}")


if __name__ == '__main__':
    output_root = sys.argv[1] if len(sys.argv) > 1 else 'paper_figures'
    
    print("=" * 60)
    print("Generating all publication figures")
    print("=" * 60)
    
    print("\n── Paper 1 (EPJ C letter): 9 figures ──")
    generate_paper1_figures(os.path.join(output_root, 'paper1_figures'))
    
    print("\n── CPC Paper 2 (companion): 8 figures ──")
    generate_cpc_figures(os.path.join(output_root, 'cpc_figures'))
    
    print("\n── Extension (optimal design): 4 figures ──")
    generate_extension_figures(os.path.join(output_root, 'extension_figures'))
    
    print("\n" + "=" * 60)
    print(f"All figures saved to {output_root}/")
    print("=" * 60)
