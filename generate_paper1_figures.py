#!/usr/bin/env python3
"""Generate all 9 Paper 1 figures from the current tau_cdma code."""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tau_cdma.tau.benchmark import setup_benchmark
from tau_cdma.core.fisher import poisson_fim, eigenvalue_spectrum
from tau_cdma.core.interference import interference_matrix, multiuser_efficiency
from tau_cdma.core.shannon import classification_mi
from tau_cdma.core.cascade import cascade_tau_a1
from tau_cdma.core.robust import dominance_margin
from tau_cdma.heavy_ion.bethe_bloch import build_template_matrix, bethe_bloch
from tau_cdma.heavy_ion.tof import build_tof_template_matrix
from tau_cdma.heavy_ion.centrality import momentum_sweep


def make_paper_figures(outdir):
    os.makedirs(outdir, exist_ok=True)
    bench = setup_benchmark()
    A, theta, N = bench['A'], bench['theta'], bench['N']
    R, eta = bench['R'], bench['eta']
    labels = ['e', 'μ', 'π', 'ρ', 'a₁', 'π2π⁰', 'other']

    # ── Fig 1: τ templates + eigenvalue spectrum + aliased eigenvector ──
    print("  Fig 1: τ benchmark (3 panels)...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # (a) Templates
    ax = axes[0]
    m_bins = np.linspace(0, 1800, A.shape[0])
    for k in range(A.shape[1]):
        ax.plot(m_bins, theta[k] * A[:, k], label=labels[k])
    ax.set_xlabel('Visible mass (MeV)')
    ax.set_ylabel('θ_k · A(m)')
    ax.set_title('(a) Templates × BR')
    ax.legend(fontsize=7, ncol=2)
    
    # (b) Eigenvalue spectrum
    ax = axes[1]
    eigR = np.sort(np.linalg.eigvalsh(R))[::-1]
    ax.semilogy(range(1, len(eigR)+1), eigR, 'ko-', markersize=6)
    ax.semilogy([len(eigR)], [eigR[-1]], 'ro', markersize=8, label=f'λ₇={eigR[-1]:.4f}')
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('λ(R)')
    ax.set_title('(b) Eigenvalue spectrum of R')
    ax.legend()
    
    # (c) Aliased eigenvector
    ax = axes[2]
    _, vecs = np.linalg.eigh(R)
    aliased = np.abs(vecs[:, 0])
    ax.bar(range(len(aliased)), aliased**2, tick_label=labels)
    ax.set_ylabel('|v₇|²')
    ax.set_title('(c) Aliased eigenvector')
    ax.tick_params(axis='x', rotation=45)
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'paper_fig1.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("    ✓")

    # ── Fig 2: η_K(p) momentum sweep ──
    print("  Fig 2: η_K vs momentum...")
    masses = [0.13957039, 0.493677, 0.93827209]
    theta_hi = np.array([0.8475, 0.1186, 0.0339])
    hi_labels = ['π', 'K', 'p']
    p_grid = np.linspace(0.2, 5.0, 200)
    res = momentum_sweep(masses=masses, sigma=0.055, theta=theta_hi, N=512,
                         p_grid=p_grid, compute_tof=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    ax = axes[0]
    for k, lbl in enumerate(hi_labels):
        ax.semilogy(p_grid, res['eta'][:, k], label=f'{lbl} (TPC)')
        ax.semilogy(p_grid, res['eta_joint'][:, k], '--', label=f'{lbl} (TPC+TOF)')
    ax.set_xlabel('p (GeV/c)')
    ax.set_ylabel('η_k')
    ax.set_title('(a) All species')
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(1e-6, 2)
    
    ax = axes[1]
    ax.semilogy(p_grid, res['eta'][:, 1], 'b-', label='K (TPC)', linewidth=2)
    ax.semilogy(p_grid, res['eta_joint'][:, 1], 'b--', label='K (TPC+TOF)', linewidth=2)
    ax.axhline(0.0015, color='gray', ls=':', alpha=0.5)
    ax.annotate('η_K = 0.0015', xy=(1.0, 0.0015), fontsize=8, color='gray')
    ax.set_xlabel('p (GeV/c)')
    ax.set_ylabel('η_K')
    ax.set_title('(b) Kaon zoom: 633× TOF rescue')
    ax.legend()
    ax.set_ylim(1e-6, 2)
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'paper_fig2.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("    ✓")

    # ── Fig 3: Bethe-Bloch + eigenvalue collapse ──
    print("  Fig 3: eigenvalue collapse at crossings...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    ax = axes[0]
    p_fine = np.linspace(0.2, 5.0, 500)
    for k, (m, lbl) in enumerate(zip(masses, hi_labels)):
        dedx = [bethe_bloch(p, m) for p in p_fine]
        ax.plot(p_fine, dedx, label=lbl)
    ax.axvline(0.9961, color='gray', ls=':', alpha=0.7)
    ax.axvline(2.3736, color='gray', ls=':', alpha=0.7)
    ax.set_xlabel('p (GeV/c)')
    ax.set_ylabel('dE/dx (a.u.)')
    ax.set_title('(a) Bethe-Bloch parameterization')
    ax.legend()
    
    ax = axes[1]
    eig_ratios = []
    for p in p_grid:
        A_hi, _ = build_template_matrix(p, masses, sigma=0.055, n_bins=100)
        F_hi = poisson_fim(A_hi, theta_hi, 512)
        eigs = eigenvalue_spectrum(F_hi)
        eig_ratios.append(float(eigs[0]) / max(float(eigs[-1]), 1e-30))
    ax.semilogy(p_grid, [1.0/r for r in eig_ratios], 'k-')
    ax.axvline(0.9961, color='red', ls=':', alpha=0.7, label='π/K crossing')
    ax.axvline(2.3736, color='blue', ls=':', alpha=0.7, label='K/p crossing')
    ax.set_xlabel('p (GeV/c)')
    ax.set_ylabel('λ_min / λ_max')
    ax.set_title('(b) Eigenvalue ratio')
    ax.legend()
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'paper_fig3.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("    ✓")

    # ── Fig 4: MAP confusion + per-channel accuracy + η-accuracy ──
    print("  Fig 4: MAP collapse theorem...")
    M_bins, K_sp = A.shape
    posterior = theta[np.newaxis, :] * A
    map_class = np.argmax(posterior, axis=1)
    lam = N * A * theta[np.newaxis, :]
    
    confusion = np.zeros((K_sp, K_sp))
    for m in range(M_bins):
        for k in range(K_sp):
            confusion[k, map_class[m]] += lam[m, k]
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = confusion / np.maximum(row_sums, 1)
    
    acc_per_species = np.diag(confusion_norm)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    ax = axes[0]
    im = ax.imshow(confusion_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(K_sp)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(K_sp)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('(a) MAP confusion matrix')
    for i in range(K_sp):
        for j in range(K_sp):
            ax.text(j, i, f'{confusion_norm[i,j]:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if confusion_norm[i,j] > 0.5 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = axes[1]
    ax.bar(range(K_sp), acc_per_species * 100, tick_label=labels, color='steelblue')
    ax.set_ylabel('MAP accuracy (%)')
    ax.set_title('(b) Per-channel accuracy')
    ax.tick_params(axis='x', rotation=45)
    
    ax = axes[2]
    ax.scatter(eta, acc_per_species, c='steelblue', s=60, zorder=5)
    for k in range(K_sp):
        ax.annotate(labels[k], (eta[k], acc_per_species[k]), fontsize=8, 
                    xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('η_k')
    ax.set_ylabel('MAP accuracy')
    ax.set_title(f'(c) η vs accuracy (r={np.corrcoef(eta, acc_per_species)[0,1]:.2f})')
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'paper_fig4.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("    ✓")

    # ── Fig 5: Correlation sign flip ──
    print("  Fig 5: correlation sign flip...")
    M_sweep = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    n_trials = 200
    rng = np.random.default_rng(42)
    
    corr_vals = []
    for M_val in M_sweep:
        from tau_cdma.tau.templates import TauTemplates
        tb = TauTemplates(M=M_val)
        A_m = tb.A
        theta_m = tb.theta
        N_m = 100000
        estimates = []
        for _ in range(n_trials):
            lam_m = N_m * A_m @ theta_m + 0.01
            y = rng.poisson(lam_m)
            from scipy.optimize import minimize
            def neg_ll(th):
                th_full = np.append(th, 1 - th.sum())
                if np.any(th_full < 0) or np.any(th_full > 1):
                    return 1e10
                lam_hat = N_m * A_m @ th_full + 0.01
                return -np.sum(y * np.log(np.maximum(lam_hat, 1e-30)) - lam_hat)
            x0 = theta_m[:-1] + rng.normal(0, 0.001, len(theta_m)-1)
            res_opt = minimize(neg_ll, x0, method='Nelder-Mead', 
                              options={'maxiter': 5000, 'xatol': 1e-8})
            estimates.append(res_opt.x)
        estimates = np.array(estimates)
        if estimates.shape[1] >= 2:
            c = np.corrcoef(estimates[:, 0], estimates[:, 1])[0, 1]
        else:
            c = 0
        corr_vals.append(c)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(M_sweep, corr_vals, 'ko-', markersize=5)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Number of bins M')
    ax.set_ylabel('corr(θ̂_e, θ̂_μ)')
    ax.set_title('Correlation sign flip (200 MC trials)')
    ax.set_xscale('log')
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'paper_fig5.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("    ✓")

    # ── Fig 6: Cascade bottleneck ──
    print("  Fig 6: cascade bottleneck...")
    casc = cascade_tau_a1()
    
    fig, ax = plt.subplots(figsize=(6, 4))
    stages = ['Stage 1\n(τ→a₁ν)', 'Stage 2\n(a₁→3π)']
    values = [casc['I1'], casc['I2']]
    bars = ax.bar(stages, values, color=['steelblue', 'coral'], width=0.5)
    ax.set_ylabel('Fisher information')
    ax.set_title(f'Cascade bottleneck: I₁/I₂ = {casc["I1"]/casc["I2"]:.1f}×')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'paper_fig6.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("    ✓")

    # ── Fig 7: Receiver hierarchy ──
    print("  Fig 7: receiver hierarchy...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    methods = ['Matched\nfilter', 'Decorrelator', 'MAP', '1D NN', '5D NN']
    accs = [46.4, 46.4, 46.4, 46.7, 76.1]
    colors = ['#a0a0a0', '#a0a0a0', '#4878A8', '#5888B8', '#2E5E8E']
    bars = ax.bar(methods, accs, color=colors, width=0.6)
    ax.set_ylabel('Overall accuracy (%)')
    ax.set_title('Receiver hierarchy — τ decay benchmark')
    ax.set_ylim(0, 100)
    ax.axhline(33.1, color='red', ls='--', alpha=0.7, label='Fano floor (33.1%)')
    ax.legend()
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 1.5, f'{acc}%', 
                ha='center', fontsize=9)
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'paper_fig7.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("    ✓")

    # ── Fig 8: Information budget ──
    print("  Fig 8: information budget...")
    mi_tau = classification_mi(A, theta)
    
    A_cross, _ = build_template_matrix(0.9961, masses, sigma=0.055, n_bins=100)
    mi_hi = classification_mi(A_cross, theta_hi)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    ax = axes[0]
    H_K_tau = mi_tau['H_K']
    MI_tau = mi_tau['MI']
    lost_tau = H_K_tau - MI_tau
    ax.bar(['H(K)', 'I(K;X)', 'Lost'], [H_K_tau, MI_tau, lost_tau],
           color=['steelblue', 'seagreen', 'coral'])
    ax.set_ylabel('bits')
    ax.set_title(f'τ decay: {MI_tau:.2f} of {H_K_tau:.2f} bits\nFano ≥ {mi_tau["fano_bound"]*100:.1f}%')
    
    ax = axes[1]
    H_K_hi = mi_hi['H_K']
    MI_hi = mi_hi['MI']
    lost_hi = H_K_hi - MI_hi
    ax.bar(['H(K)', 'I(K;X)', 'Lost'], [H_K_hi, MI_hi, lost_hi],
           color=['steelblue', 'seagreen', 'coral'])
    ax.set_ylabel('bits')
    ax.set_title(f'ALICE at crossing: {MI_hi:.2f} of {H_K_hi:.2f} bits\nFano ≥ {mi_hi["fano_bound"]*100:.1f}%')
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'paper_fig8.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("    ✓")

    # ── Fig 9: Central vs peripheral ──
    print("  Fig 9: central vs peripheral...")
    from tau_cdma.heavy_ion.centrality import centrality_sweep, CENTRALITY_CONFIGS
    cent = centrality_sweep(p_grid=np.array([0.9961]))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    dNch = [c[1] for c in CENTRALITY_CONFIGS]
    eta_K_cent = cent['eta'][:, 0, 1]  # [centrality, p_idx=0, K=1]
    crb_K_cent = np.sqrt(cent['CRB_c'][:, 0, 1])
    
    ax = axes[0]
    ax.semilogx(dNch, eta_K_cent, 'ko-')
    ax.set_xlabel('dN_ch/dη')
    ax.set_ylabel('η_K')
    ax.set_title('(a) η_K worse in central')
    
    ax = axes[1]
    ax.loglog(dNch, crb_K_cent, 'ko-')
    ax.set_xlabel('dN_ch/dη')
    ax.set_ylabel('√CRB_K')
    ax.set_title('(b) CRB_K better in central (16.8× ratio)')
    
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'paper_fig9.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("    ✓")

    print(f"\nAll 9 paper figures saved to {outdir}/")
    return outdir


if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else 'paper_figures'
    make_paper_figures(outdir)
