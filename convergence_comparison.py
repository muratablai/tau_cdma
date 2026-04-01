"""
convergence_comparison.py — GAMP vs VAMP convergence for CPC Paper 2
=====================================================================

Generates convergence plots showing iteration-by-iteration recovery
for both algorithms on the tau benchmark.

Usage:
    python convergence_comparison.py [output_dir]
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_convergence_comparison(output_dir='.'):
    """Run GAMP and VAMP on tau benchmark, compare convergence."""
    from tau_cdma.tau.benchmark import setup_benchmark
    from tau_cdma.core.gamp import gamp_poisson
    from tau_cdma.core.vamp import vamp_poisson

    bench = setup_benchmark()
    A, theta_true = bench['A'], bench['theta']
    N = bench['N']
    labels = bench['templates'].short_labels

    np.random.seed(42)
    lam_true = N * A @ theta_true
    y = np.random.poisson(lam_true)

    print("Running GAMP (max 200 iterations, damping=0.5)...")
    gamp_result = gamp_poisson(A, y, N, max_iter=200, damping=0.5, verbose=False)
    print(f"  Converged: {gamp_result['converged']} in {gamp_result['iterations']} iters")

    print("Running VAMP (max 50 iterations)...")
    vamp_result = vamp_poisson(A, y, N, max_iter=50, verbose=False)
    print(f"  Converged: {vamp_result['converged']} in {vamp_result['iterations']} iters")

    gamp_hist = gamp_result['mse_history']
    vamp_hist = vamp_result['mse_history']
    theta_gamp = gamp_result['theta']
    theta_vamp = vamp_result['theta']

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.semilogy(range(1, len(gamp_hist)+1), gamp_hist, 'b-', lw=1.5,
                 label=f"GAMP ({gamp_result['iterations']} iters)")
    ax1.semilogy(range(1, len(vamp_hist)+1), vamp_hist, 'r-', lw=1.5,
                 label=f"VAMP ({vamp_result['iterations']} iters)")
    ax1.axhline(1e-6, color='gray', ls='--', alpha=0.5, label='Tol (1e-6)')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Update norm', fontsize=12)
    ax1.set_title('(a) Convergence: GAMP vs VAMP', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    x = np.arange(len(labels))
    w = 0.25
    ax2.bar(x-w, theta_true, w, label='True', color='#2196F3', alpha=0.8)
    ax2.bar(x, theta_gamp, w,
            label=f'GAMP (MSE={np.mean((theta_gamp-theta_true)**2):.1e})',
            color='#FF9800', alpha=0.8)
    ax2.bar(x+w, theta_vamp, w,
            label=f'VAMP (MSE={np.mean((theta_vamp-theta_true)**2):.1e})',
            color='#4CAF50', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Branching ratio', fontsize=12)
    ax2.set_title('(b) Parameter recovery', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 0.32)

    plt.tight_layout()
    outpath = os.path.join(output_dir, 'convergence_gamp_vamp.png')
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

    print(f"\n{'':20s}  {'GAMP':>10s}  {'VAMP':>10s}")
    print(f"{'Iterations':20s}  {gamp_result['iterations']:10d}  {vamp_result['iterations']:10d}")
    print(f"{'Converged':20s}  {str(gamp_result['converged']):>10s}  {str(vamp_result['converged']):>10s}")
    print(f"{'Final update':20s}  {gamp_hist[-1]:10.2e}  {vamp_hist[-1]:10.2e}")
    print(f"{'MSE(theta)':20s}  {np.mean((theta_gamp-theta_true)**2):10.2e}  {np.mean((theta_vamp-theta_true)**2):10.2e}")


if __name__ == '__main__':
    outdir = sys.argv[1] if len(sys.argv) > 1 else '.'
    os.makedirs(outdir, exist_ok=True)
    run_convergence_comparison(outdir)
