"""
benchmarks.py — Runtime and performance benchmarks for CPC Paper 2
===================================================================

Measures wall-clock time and memory for core operations across
parameter sweeps (M, N, K). Outputs JSON + console table.

Usage:
    python benchmarks.py              # full sweep
    python benchmarks.py --quick      # reduced grid
"""
import sys
import os
import time
import json
import tracemalloc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _time_call(func, *args, n_repeats=3, **kwargs):
    """Time a function call, return (result, median_seconds, peak_memory_MB)."""
    times = []
    peak_mem = 0
    result = None
    for _ in range(n_repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)
        peak_mem = max(peak_mem, peak)
    return result, np.median(times), peak_mem / 1e6


def benchmark_fisher_vs_M(N=1_000_000, K=7):
    """Benchmark FIM computation as a function of number of bins M."""
    from tau_cdma.core.fisher import poisson_fim
    M_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
    results = []
    for M in M_values:
        A = np.random.dirichlet(np.ones(M), size=K).T
        theta = np.ones(K) / K
        _, t, mem = _time_call(poisson_fim, A, theta, N)
        results.append({'M': M, 'K': K, 'N': N, 'time_s': round(t, 6), 'peak_MB': round(mem, 2)})
        print(f"  FIM: M={M:5d}, K={K} -> {t*1000:.2f} ms, {mem:.1f} MB")
    return results


def benchmark_fisher_vs_K(N=1_000_000, M=200):
    """Benchmark FIM computation as a function of number of species K."""
    from tau_cdma.core.fisher import poisson_fim
    K_values = [2, 3, 5, 7, 10, 15, 20, 30]
    results = []
    for K in K_values:
        A = np.random.dirichlet(np.ones(M), size=K).T
        theta = np.ones(K) / K
        _, t, mem = _time_call(poisson_fim, A, theta, N)
        results.append({'M': M, 'K': K, 'N': N, 'time_s': round(t, 6), 'peak_MB': round(mem, 2)})
        print(f"  FIM: M={M}, K={K:3d} -> {t*1000:.2f} ms, {mem:.1f} MB")
    return results


def benchmark_fisher_vs_N(M=200, K=7):
    """Benchmark FIM computation as a function of total events N."""
    from tau_cdma.core.fisher import poisson_fim
    N_values = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    results = []
    A = np.random.dirichlet(np.ones(M), size=K).T
    theta = np.ones(K) / K
    for N in N_values:
        _, t, mem = _time_call(poisson_fim, A, theta, N)
        results.append({'M': M, 'K': K, 'N': N, 'time_s': round(t, 6), 'peak_MB': round(mem, 2)})
        print(f"  FIM: N={N:.0e}, M={M}, K={K} -> {t*1000:.2f} ms, {mem:.1f} MB")
    return results


def benchmark_full_pipeline():
    """Benchmark the complete tau benchmark pipeline."""
    from tau_cdma.tau.benchmark import setup_benchmark
    from tau_cdma.core.shannon import classification_mi, bayes_confusion
    from tau_cdma.core.robust import dominance_margin
    from tau_cdma.core.cascade import cascade_tau_a1

    results = {}

    _, t, mem = _time_call(setup_benchmark)
    results['setup_benchmark'] = {'time_s': round(t, 4), 'peak_MB': round(mem, 2)}
    print(f"  setup_benchmark:   {t*1000:.1f} ms, {mem:.1f} MB")

    bench = setup_benchmark()
    A, theta = bench['A'], bench['theta']

    _, t, mem = _time_call(classification_mi, A, theta)
    results['classification_mi'] = {'time_s': round(t, 4), 'peak_MB': round(mem, 2)}
    print(f"  classification_mi: {t*1000:.1f} ms, {mem:.1f} MB")

    _, t, mem = _time_call(bayes_confusion, A, theta)
    results['bayes_confusion'] = {'time_s': round(t, 4), 'peak_MB': round(mem, 2)}
    print(f"  bayes_confusion:   {t*1000:.1f} ms, {mem:.1f} MB")

    _, t, mem = _time_call(dominance_margin, A, theta, target_class=1)
    results['dominance_margin'] = {'time_s': round(t, 4), 'peak_MB': round(mem, 2)}
    print(f"  dominance_margin:  {t*1000:.1f} ms, {mem:.1f} MB")

    _, t, mem = _time_call(cascade_tau_a1, N=1_000_000)
    results['cascade'] = {'time_s': round(t, 4), 'peak_MB': round(mem, 2)}
    print(f"  cascade_tau_a1:    {t*1000:.1f} ms, {mem:.1f} MB")

    return results


def benchmark_gamp_vamp():
    """Benchmark GAMP and VAMP convergence."""
    from tau_cdma.tau.benchmark import setup_benchmark
    from tau_cdma.core.gamp import gamp_poisson
    from tau_cdma.core.vamp import vamp_poisson

    bench = setup_benchmark()
    A, theta = bench['A'], bench['theta']
    N = bench['N']
    np.random.seed(42)
    lam = N * A @ theta
    y = np.random.poisson(lam)

    results = {}
    _, t, mem = _time_call(gamp_poisson, A, y, N, n_repeats=1)
    results['gamp'] = {'time_s': round(t, 4), 'peak_MB': round(mem, 2)}
    print(f"  GAMP:  {t*1000:.1f} ms, {mem:.1f} MB")

    _, t, mem = _time_call(vamp_poisson, A, y, N, n_repeats=1)
    results['vamp'] = {'time_s': round(t, 4), 'peak_MB': round(mem, 2)}
    print(f"  VAMP:  {t*1000:.1f} ms, {mem:.1f} MB")

    return results


def benchmark_validation_suite():
    """Benchmark the full validation suite runtime."""
    from tau_cdma.validate.tau_predictions import run_all
    from tau_cdma.validate.heavy_ion_predictions import validate_predictions

    results = {}
    _, t, mem = _time_call(run_all, quick=True, verbose=False, n_repeats=1)
    results['tau_validation'] = {'time_s': round(t, 2), 'peak_MB': round(mem, 2)}
    print(f"  Tau validation:       {t:.1f} s, {mem:.1f} MB")

    _, t, mem = _time_call(validate_predictions, verbose=False, n_repeats=1)
    results['heavy_ion_validation'] = {'time_s': round(t, 2), 'peak_MB': round(mem, 2)}
    print(f"  Heavy-ion validation: {t:.1f} s, {mem:.1f} MB")

    return results


def main():
    quick = '--quick' in sys.argv
    np.random.seed(42)

    all_results = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'mode': 'quick' if quick else 'full',
        'python': sys.version.split()[0],
        'numpy': np.__version__,
    }
    try:
        import scipy
        all_results['scipy'] = scipy.__version__
    except ImportError:
        pass

    print("=" * 60)
    print(f"tau_cdma performance benchmarks ({'quick' if quick else 'full'})")
    print("=" * 60)

    print("\n--- FIM vs M (bins) ---")
    all_results['fim_vs_M'] = benchmark_fisher_vs_M()

    if not quick:
        print("\n--- FIM vs K (species) ---")
        all_results['fim_vs_K'] = benchmark_fisher_vs_K()
        print("\n--- FIM vs N (events) ---")
        all_results['fim_vs_N'] = benchmark_fisher_vs_N()

    print("\n--- Full pipeline (tau benchmark) ---")
    all_results['pipeline'] = benchmark_full_pipeline()

    print("\n--- GAMP / VAMP ---")
    all_results['gamp_vamp'] = benchmark_gamp_vamp()

    print("\n--- Validation suite ---")
    all_results['validation'] = benchmark_validation_suite()

    outfile = 'benchmark_results.json'
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, data in all_results.get('pipeline', {}).items():
        print(f"  {name:25s}  {data['time_s']*1000:8.1f} ms  {data['peak_MB']:6.1f} MB")


if __name__ == '__main__':
    main()
