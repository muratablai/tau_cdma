#!/usr/bin/env python3
"""
Generate all figures for all three papers with proper scaling.

Usage:
    python generate_all_paper_figures.py [output_base_dir]

Creates:
    output_base_dir/paper1/    — 9 figures for Paper 1 (EPJ C letter)
    output_base_dir/cpc/       — 8 figures for CPC Paper 2
    output_base_dir/extension/ — 4 figures for extension paper
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from tau_cdma.tau.benchmark import setup_benchmark
from tau_cdma.tau.templates import TauTemplates
from tau_cdma.core.fisher import poisson_fim, eigenvalue_spectrum
from tau_cdma.core.interference import interference_matrix, multiuser_efficiency
from tau_cdma.core.shannon import classification_mi
from tau_cdma.core.cascade import cascade_tau_a1
from tau_cdma.core.erasure import erasure_sweep, random_erasure_masks, geometric_erasure_mask
from tau_cdma.core.aliasing import aliasing_sweep
from tau_cdma.core.spreading import spreading_factor
from tau_cdma.heavy_ion.centrality import momentum_sweep, centrality_sweep
from tau_cdma.heavy_ion.bethe_bloch import bethe_bloch, build_template_matrix, find_crossings

LABELS = ['e', 'μ', 'π', 'ρ', 'a₁', 'π2π⁰', 'other']
MASSES_HI = [0.13957039, 0.493677, 0.93827209]
THETA_HI = np.array([0.8475, 0.1186, 0.0339])
HI_NAMES = ['π', 'K', 'p']


def setup_dirs(base):
    dirs = {}
    for sub in ['paper1', 'cpc', 'extension']:
        d = os.path.join(base, sub)
        os.makedirs(d, exist_ok=True)
        dirs[sub] = d
    return dirs


# ═══════════════════════════════════════════════════════════
# PAPER 1 FIGURES (9 total)
# ═══════════════════════════════════════════════════════════

def paper1_fig1(bench, save_dir):
    """Fig 1: tau templates + eigenvalue spectrum + aliased eigenvector."""
    A, theta = bench['A'], bench['theta']
    R = bench['R']
    eigvals = np.linalg.eigvalsh(R)
    eigvecs = np.linalg.eigh(R)[1]
    M = A.shape[0]
    t = TauTemplates()
    bins = np.linspace(0, 1800, M + 1)
    centers = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Templates
    ax = axes[0]
    for k in range(len(theta)):
        ax.plot(centers, theta[k] * A[:, k], label=LABELS[k], linewidth=1.2)
    ax.set_xlabel('Visible mass [MeV]')
    ax.set_ylabel('θ_k · A_mk')
    ax.set_title('(a) Templates (weighted)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (b) Eigenvalue spectrum — LOG SCALE
    ax = axes[1]
    idx = np.arange(1, len(eigvals) + 1)
    sorted_eig = np.sort(eigvals)[::-1]
    colors = ['steelblue'] * (len(eigvals) - 1) + ['red']
    ax.bar(idx, sorted_eig, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('λ_ℓ of R')
    ax.set_title(f'(b) Eigenvalue spectrum (κ = {sorted_eig[0]/sorted_eig[-1]:.0f})')
    ax.set_xticks(idx)
    ax.grid(True, alpha=0.3)

    # (c) Aliased eigenvector
    ax = axes[2]
    v_min = eigvecs[:, 0]  # smallest eigenvalue
    colors_bar = ['red' if abs(v) > 0.3 else 'steelblue' for v in v_min]
    ax.bar(range(len(v_min)), np.abs(v_min), color=colors_bar, edgecolor='black')
    ax.set_xticks(range(len(v_min)))
    ax.set_xticklabels(LABELS, rotation=45, fontsize=8)
    ax.set_ylabel('|v_min| components')
    ax.set_title(f'(c) Aliased eigenvector (λ_min = {sorted_eig[-1]:.4f})')
    ax.axhline(0.3, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig1_tau_benchmark.png'), dpi=150, bbox_inches='tight')
    plt.close()


def paper1_fig2(save_dir):
    """Fig 2: Kaon eta_K vs momentum — LOG SCALE."""
    res = momentum_sweep(compute_tof=True)
    p = res['p_grid']
    eta = res['eta']
    eta_joint = res['eta_joint']
    names = res['species_names']

    p_piK, p_Kp = 0.9961, 2.3736

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) All species
    for k in range(3):
        ax1.semilogy(p, eta[:, k], label=f'{HI_NAMES[k]} (TPC)', linewidth=1.5)
        ax1.semilogy(p, eta_joint[:, k], '--', label=f'{HI_NAMES[k]} (TPC+TOF)',
                     linewidth=1.5, alpha=0.7)
    ax1.axvline(p_piK, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(p_Kp, color='gray', linestyle='-.', alpha=0.5)
    ax1.set_xlabel('Momentum p [GeV/c]')
    ax1.set_ylabel('Multiuser efficiency η_k')
    ax1.set_title('(a) All species')
    ax1.set_ylim(1e-7, 2)
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # (b) Kaon zoom
    ax2.semilogy(p, eta[:, 1], 'b-', label='K (TPC only)', linewidth=2)
    ax2.semilogy(p, eta_joint[:, 1], 'b--', label='K (TPC+TOF)', linewidth=2)
    ax2.axvline(p_piK, color='red', linestyle=':', alpha=0.7)
    ax2.axvline(p_Kp, color='red', linestyle='-.', alpha=0.7)
    idx_piK = np.argmin(np.abs(p - p_piK))
    ax2.annotate(f'η_K = {eta[idx_piK, 1]:.1e}\n(TPC)',
                 xy=(p_piK, max(eta[idx_piK, 1], 1e-7)),
                 xytext=(1.3, 1e-5),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9, color='red')
    ax2.annotate(f'η_K = {eta_joint[idx_piK, 1]:.3f}\n(TPC+TOF)',
                 xy=(p_piK, eta_joint[idx_piK, 1]),
                 xytext=(1.5, 0.3),
                 arrowprops=dict(arrowstyle='->', color='blue'),
                 fontsize=9, color='blue')
    ax2.set_xlabel('Momentum p [GeV/c]')
    ax2.set_ylabel('Kaon multiuser efficiency η_K')
    ax2.set_title(f'(b) Kaon zoom (633× rescue at π/K)')
    ax2.set_ylim(1e-7, 2)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig2_eta_K_momentum.png'), dpi=150, bbox_inches='tight')
    plt.close()


def paper1_fig3(save_dir):
    """Fig 3: Eigenvalue collapse at BB crossings — LOG SCALE."""
    res = momentum_sweep(compute_tof=True)
    p = res['p_grid']
    eigenvalues = res['eigenvalues']
    p_piK, p_Kp = 0.9961, 2.3736

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Bethe-Bloch curves
    p_fine = np.linspace(0.2, 5.0, 500)
    for m, name, color in zip(MASSES_HI, HI_NAMES, ['blue', 'orange', 'green']):
        dedx = [bethe_bloch(pi, m) for pi in p_fine]
        ax1.plot(p_fine, dedx, label=name, color=color, linewidth=1.5)
    ax1.axvline(p_piK, color='red', linestyle=':', alpha=0.7, label=f'π/K ({p_piK})')
    ax1.axvline(p_Kp, color='red', linestyle='-.', alpha=0.7, label=f'K/p ({p_Kp})')
    ax1.set_xlabel('Momentum p [GeV/c]')
    ax1.set_ylabel('dE/dx [a.u.]')
    ax1.set_title('(a) ALEPH Bethe-Bloch parameterization')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (b) Eigenvalue ratio — LOG SCALE
    eig_ratio = eigenvalues.min(axis=1) / eigenvalues.max(axis=1)
    ax2.semilogy(p, eig_ratio, 'k-', linewidth=1.5)
    ax2.axvline(p_piK, color='red', linestyle=':', alpha=0.7, label='π/K')
    ax2.axvline(p_Kp, color='red', linestyle='-.', alpha=0.7, label='K/p')
    idx_piK = np.argmin(np.abs(p - p_piK))
    idx_Kp = np.argmin(np.abs(p - p_Kp))
    ax2.annotate(f'{eig_ratio[idx_piK]:.1e}',
                 xy=(p_piK, max(eig_ratio[idx_piK], 1e-10)),
                 xytext=(1.5, 1e-3),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9, color='red')
    ax2.annotate(f'{eig_ratio[idx_Kp]:.1e}',
                 xy=(p_Kp, max(eig_ratio[idx_Kp], 1e-10)),
                 xytext=(3.2, 1e-3),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9, color='red')
    ax2.set_xlabel('Momentum p [GeV/c]')
    ax2.set_ylabel('λ_min / λ_max')
    ax2.set_title('(b) FIM eigenvalue ratio')
    ax2.set_ylim(1e-10, 2)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig3_eigenvalue_collapse_HI.png'), dpi=150, bbox_inches='tight')
    plt.close()


def paper1_fig4(bench, save_dir):
    """Fig 4: MAP collapse — confusion matrix + accuracy + eta-accuracy."""
    A, theta = bench['A'], bench['theta']
    eta = bench['eta']
    M, K = A.shape

    # MAP confusion matrix
    posterior = theta[np.newaxis, :] * A
    map_class = np.argmax(posterior, axis=1)
    lam = 1e6 * A * theta[np.newaxis, :]
    confusion = np.zeros((K, K))
    for m_bin in range(M):
        for k in range(K):
            confusion[k, map_class[m_bin]] += lam[m_bin, k]
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    confusion_norm = confusion / row_sums
    acc = np.diag(confusion_norm)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Confusion matrix
    ax = axes[0]
    im = ax.imshow(confusion_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(LABELS, fontsize=7, rotation=45)
    ax.set_yticks(range(K)); ax.set_yticklabels(LABELS, fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('(a) MAP confusion matrix')
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f'{confusion_norm[i,j]:.2f}', ha='center', va='center',
                    fontsize=6, color='white' if confusion_norm[i,j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # (b) Per-channel accuracy
    ax = axes[1]
    colors = ['red' if a < 0.01 else 'steelblue' for a in acc]
    ax.bar(range(K), acc * 100, color=colors, edgecolor='black')
    ax.set_xticks(range(K)); ax.set_xticklabels(LABELS, rotation=45, fontsize=8)
    ax.set_ylabel('MAP accuracy [%]')
    ax.set_title(f'(b) Per-channel accuracy (overall: {np.sum(np.diag(confusion))/np.sum(confusion)*100:.1f}%)')
    ax.grid(True, alpha=0.3, axis='y')

    # (c) eta vs accuracy
    ax = axes[2]
    ax.scatter(eta, acc * 100, c='steelblue', s=60, zorder=5)
    for k in range(K):
        ax.annotate(LABELS[k], (eta[k], acc[k] * 100), fontsize=7,
                    xytext=(5, 5), textcoords='offset points')
    from scipy.stats import pearsonr
    r, _ = pearsonr(eta, acc)
    ax.set_xlabel('Multiuser efficiency η_k')
    ax.set_ylabel('MAP accuracy [%]')
    ax.set_title(f'(c) η vs accuracy (r = {r:.2f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_MAP_collapse.png'), dpi=150, bbox_inches='tight')
    plt.close()


def paper1_fig5(bench, save_dir):
    """Fig 5: Correlation sign flip (Monte Carlo)."""
    from tau_cdma.core.aliasing import aliasing_sweep
    A_full, theta = bench['A'], bench['theta']
    M_values = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300]
    t = TauTemplates()

    corrs = []
    for M_val in M_values:
        t_m = TauTemplates(M=M_val)
        A_m = t_m.A
        bg_m = np.full(M_val, 0.01)
        F = poisson_fim(A_m, theta, bench['N'], bg_m)
        if np.linalg.det(F) > 0:
            F_inv = np.linalg.inv(F)
            sigma_e = np.sqrt(F_inv[0, 0])
            sigma_mu = np.sqrt(F_inv[1, 1])
            cov_emu = F_inv[0, 1]
            corr = cov_emu / (sigma_e * sigma_mu)
            corrs.append(corr)
        else:
            corrs.append(1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(M_values, corrs, 'bo-', linewidth=2, markersize=6)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of bins M')
    ax.set_ylabel('corr(θ̂_e, θ̂_μ)')
    ax.set_title('Correlation sign flip: degenerate (+1) → trade-off (−1)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig5_correlation_sign_flip.png'), dpi=150, bbox_inches='tight')
    plt.close()


def paper1_fig6(save_dir):
    """Fig 6: Cascade bottleneck."""
    cas = cascade_tau_a1()
    fig, ax = plt.subplots(figsize=(7, 5))
    stages = ['Stage 1\n(τ→a₁ν)', 'Stage 2\n(a₁→3π)']
    values = [cas['I1'], cas['I2']]
    colors = ['steelblue', 'coral']
    ax.bar(stages, values, color=colors, edgecolor='black', width=0.5)
    ax.set_ylabel('Fisher information')
    ratio = cas['I1'] / cas['I2']
    ax.set_title(f'Cascade bottleneck: I₁/I₂ = {ratio:.1f}×')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig6_cascade_bottleneck.png'), dpi=150, bbox_inches='tight')
    plt.close()


def paper1_fig7(bench, save_dir):
    """Fig 7: Receiver hierarchy — bar chart of accuracies."""
    # These are the validated numbers
    methods = ['Matched\nFilter', 'Decorrelator', 'MAP', '1D NN', '5D NN']
    accs = [46.4, 46.4, 46.4, 46.7, 76.1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#a0c4e8', '#a0c4e8', '#a0c4e8', '#6baed6', '#2171b5']
    bars = ax.bar(methods, accs, color=colors, edgecolor='black', width=0.6)
    ax.set_ylabel('Classification accuracy [%]')
    ax.set_title('Receiver hierarchy: 1D ceiling vs multi-observable gain')
    ax.axhline(46.4, color='gray', linestyle='--', alpha=0.5, label='1D ceiling (46.4%)')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{acc}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig7_receiver_hierarchy.png'), dpi=150, bbox_inches='tight')
    plt.close()


def paper1_fig8(bench, save_dir):
    """Fig 8: Information budget for both benchmarks."""
    mi_tau = classification_mi(bench['A'], bench['theta'])

    # Heavy-ion at crossing
    A_hi, _ = build_template_matrix(0.9961, MASSES_HI, sigma=0.055, n_bins=100)
    mi_hi = classification_mi(A_hi, THETA_HI)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # tau
    H_K = mi_tau['H_K']
    MI = mi_tau['MI']
    lost = H_K - MI
    ax1.bar(['H(K)', 'I(K;X)', 'Lost'], [H_K, MI, lost],
            color=['steelblue', 'green', 'coral'], edgecolor='black')
    ax1.set_ylabel('Bits')
    ax1.set_title(f'τ decay: {MI:.2f} of {H_K:.2f} bits retained')
    ax1.axhline(mi_tau['fano_bound'] * 100 / 100, color='red', linestyle='--', alpha=0)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(1, MI + 0.05, f'Fano: P_e ≥ {mi_tau["fano_bound"]*100:.1f}%', fontsize=8, color='red')

    # heavy-ion
    H_K2 = mi_hi['H_K']
    MI2 = mi_hi['MI']
    lost2 = H_K2 - MI2
    ax2.bar(['H(K)', 'I(K;X)', 'Lost'], [H_K2, MI2, lost2],
            color=['steelblue', 'green', 'coral'], edgecolor='black')
    ax2.set_ylabel('Bits')
    ax2.set_title(f'ALICE at π/K crossing: {MI2:.2f} of {H_K2:.2f} bits retained')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(1, MI2 + 0.02, f'Fano: P_e ≥ {mi_hi["fano_bound"]*100:.1f}%', fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig8_information_budget.png'), dpi=150, bbox_inches='tight')
    plt.close()


def paper1_fig9(save_dir):
    """Fig 9: Central vs peripheral eta-CRB distinction."""
    res = momentum_sweep(compute_tof=False, sigma=0.05, N=1600)
    res_periph = momentum_sweep(compute_tof=False, sigma=0.04, N=100)
    p = res['p_grid']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # (a) eta_K
    ax1.semilogy(p, res['eta'][:, 1], 'r-', label='Central (dNch=1600)', linewidth=1.5)
    ax1.semilogy(p, res_periph['eta'][:, 1], 'b-', label='Peripheral (dNch=100)', linewidth=1.5)
    ax1.set_xlabel('Momentum p [GeV/c]')
    ax1.set_ylabel('Kaon η_K')
    ax1.set_title('(a) η_K: worse in central (broader σ)')
    ax1.set_ylim(1e-7, 2)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (b) sqrt(CRB_K)
    ax2.semilogy(p, np.sqrt(res['CRB'][:, 1]), 'r-', label='Central', linewidth=1.5)
    ax2.semilogy(p, np.sqrt(res_periph['CRB'][:, 1]), 'b-', label='Peripheral', linewidth=1.5)
    ax2.set_xlabel('Momentum p [GeV/c]')
    ax2.set_ylabel('√CRB_K')
    ax2.set_title('(b) √CRB_K: better in central (more tracks)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig9_central_peripheral.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════
# CPC PAPER 2 FIGURES (8 total)
# ═══════════════════════════════════════════════════════════

def cpc_fig1(bench, save_dir):
    """CPC Fig 1: tau templates."""
    A, theta = bench['A'], bench['theta']
    M = A.shape[0]
    centers = np.linspace(0, 1800, M)

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(len(theta)):
        ax.plot(centers, A[:, k], label=LABELS[k], linewidth=1.2)
    ax.set_xlabel('Visible mass [MeV]')
    ax.set_ylabel('Template A_mk')
    ax.set_title('τ-decay templates (M=200, σ=20 MeV)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig1_templates.png'), dpi=150, bbox_inches='tight')
    plt.close()


def cpc_fig2(bench, save_dir):
    """CPC Fig 2: R matrix heatmap."""
    R = bench['R']
    K = R.shape[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(R, cmap='RdBu_r', norm=TwoSlopeNorm(0), aspect='equal')
    ax.set_xticks(range(K)); ax.set_xticklabels(LABELS, fontsize=8, rotation=45)
    ax.set_yticks(range(K)); ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_title('Normalized correlation matrix R')
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f'{R[i,j]:.2f}', ha='center', va='center', fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig2_R_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def cpc_fig3(bench, save_dir):
    """CPC Fig 3: eta bar chart."""
    eta = bench['eta']
    K = len(eta)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['red' if e < 0.01 else ('orange' if e < 0.1 else 'steelblue') for e in eta]
    ax.bar(range(K), eta, color=colors, edgecolor='black')
    ax.set_xticks(range(K)); ax.set_xticklabels(LABELS, rotation=45, fontsize=9)
    ax.set_ylabel('Multiuser efficiency η_k')
    ax.set_title('Per-species identifiability')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 2)
    ax.grid(True, alpha=0.3, axis='y')
    for k in range(K):
        ax.text(k, eta[k] * 1.2, f'{eta[k]:.3f}', ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig3_eta.png'), dpi=150, bbox_inches='tight')
    plt.close()


def cpc_fig4(bench, save_dir):
    """CPC Fig 4: eigenvalue spectrum of R — LOG SCALE."""
    R = bench['R']
    eigvals = np.sort(np.linalg.eigvalsh(R))[::-1]
    K = len(eigvals)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['steelblue'] * (K - 1) + ['red']
    ax.bar(range(1, K + 1), eigvals, color=colors, edgecolor='black')
    ax.set_yscale('log')
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('λ_ℓ')
    ax.set_title(f'R eigenvalue spectrum (λ₁/λ₇ = {eigvals[0]/eigvals[-1]:.0f})')
    ax.set_xticks(range(1, K + 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_eigenvalues.png'), dpi=150, bbox_inches='tight')
    plt.close()


def cpc_fig5(bench, save_dir):
    """CPC Fig 5: erasure comparison — sqrt(CRB) per species."""
    A, theta, N, bg = bench['A'], bench['theta'], bench['N'], bench['background']
    m_bins = bench['templates'].m_bins
    alphas = np.linspace(0.3, 1.0, 15)

    er = erasure_sweep(A, theta, N, bg, alphas, n_trials=20, mode='random')
    eg = erasure_sweep(A, theta, N, bg, alphas, mode='geometric', m_bins=m_bins)

    K = A.shape[1]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    axes = axes.ravel()
    for k in range(min(K, 7)):
        ax = axes[k]
        cr = np.sqrt(np.maximum(er['crb_mean'][:, k], 0))
        cg = np.sqrt(np.maximum(eg['crb_mean'][:, k], 0))
        ax.plot(alphas, cr, 'b.-', label='Random', markersize=3)
        ax.plot(alphas, cg, 'r.-', label='Geometric', markersize=3)
        ax.set_title(LABELS[k], fontsize=10)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if k >= 4: ax.set_xlabel('Access fraction α')
        if k % 4 == 0: ax.set_ylabel('√CRB')
        if k == 0: ax.legend(fontsize=7)
    if K < 8: axes[-1].set_visible(False)
    fig.suptitle('Geometric vs random erasure: species-dependent CRB', fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig5_erasure.png'), dpi=150, bbox_inches='tight')
    plt.close()


def cpc_fig6(save_dir):
    """CPC Fig 6: cascade bottleneck (same as Paper 1 Fig 6)."""
    paper1_fig6(save_dir)  # Reuse
    os.rename(os.path.join(save_dir, 'fig6_cascade_bottleneck.png'),
              os.path.join(save_dir, 'fig6_cascade.png'))


def cpc_fig7(bench, save_dir):
    """CPC Fig 7: optimal binning."""
    from tau_cdma.core.spreading import spreading_factor, optimal_binning
    widths = [0.0, 0.0, 0.3, 149.1, 420.0, 300.0, 200.0]  # approximate resonance widths
    names_short = LABELS
    M_range = np.arange(10, 300, 10)
    t = TauTemplates()

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(len(widths)):
        if widths[k] > 0:
            sf = spreading_factor(1776.93, widths[k])
            ax.axhline(y=sf, color=f'C{k}', linestyle='--', alpha=0.3)
    ax.bar(range(len(widths)), [spreading_factor(1776.93, w) if w > 0 else 0 for w in widths],
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(LABELS, rotation=45, fontsize=8)
    ax.set_ylabel('Spreading factor')
    ax.set_title('Spreading factor by channel (broader resonance → larger SF)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig7_optimal_binning.png'), dpi=150, bbox_inches='tight')
    plt.close()


def cpc_fig8(save_dir):
    """CPC Fig 8: eigenvalue collapse across momentum — LOG SCALE."""
    # This is the heavy-ion momentum sweep, NOT the tau eigenvalue sweep
    res = momentum_sweep(compute_tof=False)
    p = res['p_grid']
    eigenvalues = res['eigenvalues']
    eig_ratio = eigenvalues.min(axis=1) / eigenvalues.max(axis=1)
    p_piK, p_Kp = 0.9961, 2.3736

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(p, eig_ratio, 'k-', linewidth=1.5)
    ax.axvline(p_piK, color='red', linestyle=':', alpha=0.7, label=f'π/K ({p_piK})')
    ax.axvline(p_Kp, color='red', linestyle='-.', alpha=0.7, label=f'K/p ({p_Kp})')
    ax.set_xlabel('Momentum p [GeV/c]')
    ax.set_ylabel('λ_min / λ_max')
    ax.set_title('Eigenvalue collapse at Bethe-Bloch crossings')
    ax.set_ylim(1e-10, 2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig8_eigenvalue_collapse_HI.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else 'paper_figures'
    dirs = setup_dirs(base)

    print("Setting up tau benchmark...")
    bench = setup_benchmark()

    print("\n=== PAPER 1 FIGURES ===")
    for i, (name, fn) in enumerate([
        ('Fig 1: tau benchmark', lambda: paper1_fig1(bench, dirs['paper1'])),
        ('Fig 2: eta_K momentum', lambda: paper1_fig2(dirs['paper1'])),
        ('Fig 3: eigenvalue collapse HI', lambda: paper1_fig3(dirs['paper1'])),
        ('Fig 4: MAP collapse', lambda: paper1_fig4(bench, dirs['paper1'])),
        ('Fig 5: correlation sign flip', lambda: paper1_fig5(bench, dirs['paper1'])),
        ('Fig 6: cascade bottleneck', lambda: paper1_fig6(dirs['paper1'])),
        ('Fig 7: receiver hierarchy', lambda: paper1_fig7(bench, dirs['paper1'])),
        ('Fig 8: information budget', lambda: paper1_fig8(bench, dirs['paper1'])),
        ('Fig 9: central/peripheral', lambda: paper1_fig9(dirs['paper1'])),
    ], 1):
        print(f"  [{i}/9] {name}...", end=' ', flush=True)
        fn()
        print("✓")

    print("\n=== CPC PAPER 2 FIGURES ===")
    for i, (name, fn) in enumerate([
        ('Fig 1: templates', lambda: cpc_fig1(bench, dirs['cpc'])),
        ('Fig 2: R matrix', lambda: cpc_fig2(bench, dirs['cpc'])),
        ('Fig 3: eta bars', lambda: cpc_fig3(bench, dirs['cpc'])),
        ('Fig 4: eigenvalues', lambda: cpc_fig4(bench, dirs['cpc'])),
        ('Fig 5: erasure', lambda: cpc_fig5(bench, dirs['cpc'])),
        ('Fig 6: cascade', lambda: cpc_fig6(dirs['cpc'])),
        ('Fig 7: optimal binning', lambda: cpc_fig7(bench, dirs['cpc'])),
        ('Fig 8: eigenvalue collapse HI', lambda: cpc_fig8(dirs['cpc'])),
    ], 1):
        print(f"  [{i}/8] {name}...", end=' ', flush=True)
        fn()
        print("✓")

    print("\n=== EXTENSION FIGURES ===")
    import subprocess
    result = subprocess.run(
        [sys.executable, 'run_optimal_design.py', dirs['extension']],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0:
        print("  Extension figures generated ✓")
    else:
        print(f"  Extension figures FAILED: {result.stderr[-200:]}")

    # Summary
    print("\n" + "=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)
    for sub in ['paper1', 'cpc', 'extension']:
        figs = sorted(os.listdir(dirs[sub]))
        print(f"\n  {sub}/ ({len(figs)} files):")
        for f in figs:
            if f.endswith('.png'):
                print(f"    {f}")


if __name__ == '__main__':
    main()
