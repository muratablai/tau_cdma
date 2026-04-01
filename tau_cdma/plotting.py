"""
plotting.py — Publication-quality figures for the τ-CDMA framework
====================================================================

Generates Figures 1–10 for the CPC paper.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (7, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def plot_templates(bench, save_path=None):
    """Fig 1: Seven τ decay templates overlaid."""
    tb = bench['templates']
    A = bench['A']
    m = tb.m_bins
    theta = bench['theta']

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set1(np.linspace(0, 1, 7))

    for k in range(7):
        ax.plot(m, A[:, k] * theta[k], label=tb.short_labels[k],
                color=colors[k], linewidth=1.5)

    # Total
    total = A @ theta
    ax.plot(m, total, 'k--', label='Total', linewidth=2, alpha=0.5)

    ax.set_xlabel(r'$m_{\rm vis}$ [MeV]')
    ax.set_ylabel(r'BR$_k \cdot a_k(m)$ [per MeV]')
    ax.set_title(r'$\tau$ Decay Channel Templates (1D visible mass)')
    ax.legend(ncol=2)
    ax.set_xlim(0, 1800)
    ax.set_ylim(bottom=0)

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_interference_matrix(bench, save_path=None):
    """Fig 2: Interference matrix R as heatmap."""
    R = bench['R']
    labels = bench['templates'].short_labels

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.abs(R), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(7))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(range(7))
    ax.set_yticklabels(labels)
    ax.set_title('Interference Matrix |R|')
    plt.colorbar(im, ax=ax, label=r'$|R_{jk}|$')

    # Annotate values
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f'{R[i,j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if abs(R[i,j]) > 0.5 else 'black')

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_multiuser_efficiency(bench, save_path=None):
    """Fig 3: Multiuser efficiency η_k bar chart."""
    eta = bench['eta']
    labels = bench['templates'].short_labels

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, 7))
    bars = ax.bar(range(7), eta, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(7))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r'Multiuser Efficiency $\eta_k$')
    ax.set_title('P1: Channel Separation Efficiency')
    ax.set_ylim(0, max(eta) * 1.2)

    for bar, val in zip(bars, eta):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_eigenvalue_sweep(sweep_results, save_path=None):
    """Fig 4: Eigenvalue spectrum vs binning M."""
    M_vals = [r['M'] for r in sweep_results]
    K = sweep_results[0]['eigvals'].shape[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    for ell in range(K):
        eigvals = [r['eigvals'][ell] for r in sweep_results]
        ax.plot(M_vals, eigvals, 'o-', label=f'λ_{ell+1}', markersize=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of bins M')
    ax.set_ylabel('FIM Eigenvalue')
    ax.set_title('P2: Eigenvalue Spectrum vs Binning Resolution')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_erasure_comparison(erasure_random, erasure_geom, labels, save_path=None):
    """Fig 6: CRB vs α for random and geometric erasure."""
    alpha = erasure_random['alpha']
    K = erasure_random['crb_mean'].shape[1]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True)
    axes = axes.ravel()

    for k in range(min(K, 7)):
        ax = axes[k]
        cr = np.sqrt(np.maximum(erasure_random['crb_mean'][:, k], 0))
        cg = np.sqrt(np.maximum(erasure_geom['crb_mean'][:, k], 0))

        ax.plot(alpha, cr, 'b.-', label='Random', markersize=3)
        ax.plot(alpha, cg, 'r.-', label='Geometric', markersize=3)
        ax.set_title(labels[k], fontsize=10)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if k >= 4:
            ax.set_xlabel(r'Access fraction $\alpha$')
        if k % 4 == 0:
            ax.set_ylabel(r'$\sqrt{\rm CRB}$')
        if k == 0:
            ax.legend(fontsize=7)

    # Remove unused subplot
    if K < 8:
        axes[-1].set_visible(False)

    fig.suptitle('P3: Random vs Geometric Erasure', fontsize=13, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_cascade_info(cascade_result, save_path=None):
    """Fig 8: Cascade information flow for τ→a₁→3π."""
    I1 = cascade_result['I1']
    I2 = cascade_result['I2']
    SF = cascade_result['SF_cascade']

    fig, ax = plt.subplots(figsize=(6, 4))
    stages = ['Stage 1\nτ→a₁ν', 'Stage 2\na₁→3π']
    values = [I1, I2]
    colors = ['steelblue', 'coral']

    bars = ax.bar(stages, values, color=colors, edgecolor='black', width=0.5)
    ax.set_ylabel('Fisher Information (trace)')
    ax.set_title(f'P4: Cascade Bottleneck (SF_cascade = {SF:.1f})')
    ax.set_yscale('log')

    # Arrow showing bottleneck
    bottleneck_idx = 0 if I1 < I2 else 1
    ax.annotate('BOTTLENECK', xy=(bottleneck_idx, values[bottleneck_idx]),
                xytext=(bottleneck_idx, values[bottleneck_idx] * 3),
                ha='center', fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_fisher_vs_M(M_values, fish_diag, labels, save_path=None):
    """Fig 9: Fisher information per channel vs M showing saturation."""
    fig, ax = plt.subplots(figsize=(7, 5))
    F_max = np.max(fish_diag, axis=0)
    frac = fish_diag / np.maximum(F_max, 1e-30)
    
    for k in range(min(fish_diag.shape[1], 7)):
        ax.plot(M_values, frac[:, k] * 100, 'o-', label=labels[k], markersize=4)

    ax.set_xscale('log')
    ax.set_xlabel('Number of bins M')
    ax.set_ylabel('Fisher Information (% of peak)')
    ax.set_title('P7: Fisher Information Saturation vs Binning Resolution')
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_p9_eigenvalue_collapse(p9_results, labels, save_path=None):
    """Fig 10: Eigenvalue collapse and PID rescue for P9."""
    eigvals_1d = p9_results['eigvals_1d']
    eigvals_pid = p9_results['eigvals_pid']
    K = len(eigvals_1d)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: eigenvalue spectra comparison
    ax = axes[0]
    idx = np.arange(1, K + 1)
    ax.semilogy(idx, eigvals_1d[::-1], 'bo-', label='1D mass only', markersize=6)
    ax.semilogy(idx, eigvals_pid[::-1], 'rs-', label='+ n_trk + PID', markersize=6)
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel(r'Eigenvalue $\lambda_\ell$ of R')
    ax.set_title('P9: Eigenvalue Collapse and PID Rescue')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(idx)

    # Right: aliased eigenvector
    ax = axes[1]
    v_min = p9_results['eigvecs_1d'][:, -1]  # smallest eigenvalue direction
    colors = ['red' if abs(v) > 0.3 else 'steelblue' for v in v_min]
    ax.bar(range(K), np.abs(v_min), color=colors, edgecolor='black')
    ax.set_xticks(range(K))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel(r'$|v_{\min}|$ (eigenvector components)')
    ax.set_title(f'Aliased direction ($\\lambda_{{min}}$ = {eigvals_1d[-1]:.4f})')
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5)

    fig.suptitle('P9: Eigenvalue Structure of Interference Matrix', fontsize=13, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
    return fig


def generate_all_figures(output_dir='paper/figures'):
    """Generate all publication figures."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    from tau_cdma.tau.benchmark import setup_benchmark
    from tau_cdma.core.aliasing import aliasing_sweep
    from tau_cdma.core.erasure import erasure_sweep
    from tau_cdma.core.cascade import cascade_tau_a1
    from tau_cdma.validate.tau_predictions import validate_p7, validate_p9

    bench = setup_benchmark()
    labels = bench['templates'].short_labels

    print("Generating figures...")

    # Fig 1
    plot_templates(bench, os.path.join(output_dir, 'fig01_templates.png'))
    print("  Fig 1: templates ✓")

    # Fig 2
    plot_interference_matrix(bench, os.path.join(output_dir, 'fig02_R_matrix.png'))
    print("  Fig 2: interference matrix ✓")

    # Fig 3
    plot_multiuser_efficiency(bench, os.path.join(output_dir, 'fig03_eta.png'))
    print("  Fig 3: multiuser efficiency ✓")

    # Fig 4
    tb = bench['templates']
    M_values = [5, 10, 20, 50, 100, 200, 500]
    sweep = aliasing_sweep(tb, M_values, bench['theta'], bench['N'])
    plot_eigenvalue_sweep(sweep, os.path.join(output_dir, 'fig04_eigenvalues.png'))
    print("  Fig 4: eigenvalue sweep ✓")

    # Fig 6
    alpha_vals = np.linspace(0.3, 1.0, 15)
    er = erasure_sweep(bench['A'], bench['theta'], bench['N'], bench['background'],
                       alpha_vals, n_trials=20, mode='random')
    eg = erasure_sweep(bench['A'], bench['theta'], bench['N'], bench['background'],
                       alpha_vals, mode='geometric', m_bins=bench['templates'].m_bins)
    plot_erasure_comparison(er, eg, labels, os.path.join(output_dir, 'fig06_erasure.png'))
    print("  Fig 6: erasure comparison ✓")

    # Fig 8
    cas = cascade_tau_a1(N=bench['N'])
    plot_cascade_info(cas, os.path.join(output_dir, 'fig08_cascade.png'))
    print("  Fig 8: cascade bottleneck ✓")

    # Fig 9 (from P7)
    p7 = validate_p7(bench, verbose=False)
    plot_fisher_vs_M(p7['M_values'], p7['fish_diag'], labels,
                  os.path.join(output_dir, 'fig09_optimal_binning.png'))
    print("  Fig 9: optimal binning ✓")

    # Fig 10 (from P9)
    p9 = validate_p9(bench, verbose=False)
    plot_p9_eigenvalue_collapse(p9, labels,
                  os.path.join(output_dir, 'fig10_eigenvalue_collapse.png'))
    print("  Fig 10: eigenvalue collapse ✓")

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == '__main__':
    generate_all_figures()
