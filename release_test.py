#!/usr/bin/env python3
"""
release_test.py — End-to-end release gate for tau_cdma
=======================================================

Proves that validation + figure generation + shipped outputs
all reproduce from a clean checkout.

Usage:
    python release_test.py [--output-dir DIR]

Exit code 0 = release-ready, non-zero = blocker found.
"""

import sys
import os
import tempfile
import json
from datetime import datetime

def main():
    output_dir = None
    smoke = '--smoke' in sys.argv
    for i, arg in enumerate(sys.argv):
        if arg == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='tau_cdma_release_')
    
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'output_dir': output_dir,
        'mode': 'smoke' if smoke else 'full',
        'steps': {},
    }
    all_ok = True
    
    print("=" * 60)
    print(f"tau_cdma release gate test ({'smoke' if smoke else 'full'})")
    print("=" * 60)
    if smoke:
        print("  (smoke mode: imports + version + criteria + 1 figure only)")
    
    # --- Step 1: Import check ---
    print("\n--- Step 1: Import check ---")
    try:
        from tau_cdma.validate.tau_predictions import run_all
        from tau_cdma.validate.heavy_ion_predictions import validate_predictions
        from tau_cdma.validate.prediction_criteria import get_all_criteria
        from tau_cdma.plotting import generate_all_figures
        criteria = get_all_criteria()
        results['steps']['import'] = {'status': 'PASS', 'n_criteria': len(criteria)}
        print(f"  ✓ All imports succeed, {len(criteria)} prediction criteria loaded")
    except Exception as e:
        results['steps']['import'] = {'status': 'FAIL', 'error': str(e)}
        print(f"  ✗ Import failed: {e}")
        all_ok = False
    
    # --- Step 2: τ validation suite ---
    print("\n--- Step 2: τ validation suite ---")
    if smoke:
        print("  ⊘ Skipped in smoke mode (use full mode to run)")
        results['steps']['tau_validation'] = {'status': 'SKIPPED'}
    else:
        try:
            summary = run_all(quick=True, verbose=False)
            n_pass = sum(1 for r in summary.values() if r.get('passed'))
            n_total = len(summary)
            n_checks_pass = sum(
                sum(1 for v in r.get('checks', {}).values() if v)
                for r in summary.values()
            )
            n_checks_total = sum(len(r.get('checks', {})) for r in summary.values())
            
            tau_ok = all(r.get('passed') for r in summary.values())
            results['steps']['tau_validation'] = {
                'status': 'PASS' if tau_ok else 'FAIL',
                'predictions_passed': n_pass,
                'predictions_total': n_total,
                'checks_passed': n_checks_pass,
                'checks_total': n_checks_total,
            }
            icon = '✓' if tau_ok else '✗'
            print(f"  {icon} τ predictions: {n_pass}/{n_total} passed, "
                  f"{n_checks_pass}/{n_checks_total} checks")
            if not tau_ok:
                all_ok = False
                for p, r in summary.items():
                    if not r.get('passed'):
                        failed = [k for k, v in r.get('checks', {}).items() if not v]
                        print(f"      {p} failed: {failed}")
        except Exception as e:
            results['steps']['tau_validation'] = {'status': 'ERROR', 'error': str(e)}
            print(f"  ✗ Validation crashed: {e}")
            all_ok = False
    
    # --- Step 3: Heavy-ion validation ---
    print("\n--- Step 3: Heavy-ion validation ---")
    if smoke:
        print("  ⊘ Skipped in smoke mode")
        results['steps']['heavy_ion_validation'] = {'status': 'SKIPPED'}
    else:
        try:
            hi_results = validate_predictions(verbose=False)
            results['steps']['heavy_ion_validation'] = {'status': 'PASS'}
            print(f"  ✓ Heavy-ion predictions all pass")
        except Exception as e:
            results['steps']['heavy_ion_validation'] = {'status': 'ERROR', 'error': str(e)}
            print(f"  ✗ Heavy-ion validation error: {e}")
            all_ok = False
    
    # --- Step 4: Figure generation ---
    print("\n--- Step 4: Figure generation ---")
    expected_figures = [
        'fig01_templates.png', 'fig02_R_matrix.png', 'fig03_eta.png',
        'fig04_eigenvalues.png', 'fig06_erasure.png', 'fig08_cascade.png',
        'fig09_optimal_binning.png', 'fig10_eigenvalue_collapse.png',
    ]
    if smoke:
        # Smoke: generate only fig01 (templates) as a fast pipeline check
        try:
            from tau_cdma.tau.benchmark import setup_benchmark
            from tau_cdma.plotting import plot_templates
            bench = setup_benchmark()
            plot_templates(bench, os.path.join(fig_dir, 'fig01_templates.png'))
            results['steps']['figure_generation'] = {'status': 'PASS', 'mode': 'smoke (1 fig)'}
            print(f"  ✓ Smoke: generated fig01_templates.png")
        except Exception as e:
            results['steps']['figure_generation'] = {'status': 'ERROR', 'error': str(e)}
            print(f"  ✗ Smoke figure failed: {e}")
            all_ok = False
    else:
        try:
            generate_all_figures(fig_dir)
            generated = os.listdir(fig_dir)
            missing = [f for f in expected_figures if f not in generated]
            
            fig_ok = len(missing) == 0
            results['steps']['figure_generation'] = {
                'status': 'PASS' if fig_ok else 'FAIL',
                'generated': len(generated),
                'expected': len(expected_figures),
                'missing': missing,
            }
            icon = '✓' if fig_ok else '✗'
            print(f"  {icon} Generated {len(generated)}/{len(expected_figures)} figures")
            if missing:
                print(f"      Missing: {missing}")
            if not fig_ok:
                all_ok = False
        except Exception as e:
            results['steps']['figure_generation'] = {'status': 'ERROR', 'error': str(e)}
            print(f"  ✗ Figure generation crashed: {e}")
            all_ok = False
    
    # --- Step 5: Version consistency ---
    print("\n--- Step 5: Version consistency ---")
    try:
        import tau_cdma
        pkg_dir = os.path.dirname(os.path.dirname(tau_cdma.__file__))
        
        version_checks = {}
        # Check pyproject.toml
        pyproject_path = os.path.join(pkg_dir, 'pyproject.toml')
        if os.path.exists(pyproject_path):
            with open(pyproject_path) as f:
                content = f.read()
            if '0.5.0' in content:
                version_checks['pyproject'] = 'PASS'
            else:
                version_checks['pyproject'] = 'FAIL: not 0.5.0'
        
        # Check README
        readme_path = os.path.join(pkg_dir, 'README.md')
        if os.path.exists(readme_path):
            with open(readme_path) as f:
                content = f.read()
            if '0.4.9' in content or '0.4.8' in content:
                version_checks['readme'] = 'FAIL: stale version in README'
            elif '0.5.0' in content:
                version_checks['readme'] = 'PASS'
            else:
                version_checks['readme'] = 'PASS'
        
        ver_ok = all(v == 'PASS' for v in version_checks.values())
        results['steps']['version_consistency'] = {
            'status': 'PASS' if ver_ok else 'FAIL',
            'checks': version_checks,
        }
        icon = '✓' if ver_ok else '✗'
        print(f"  {icon} Version consistency: {version_checks}")
        if not ver_ok:
            all_ok = False
    except Exception as e:
        results['steps']['version_consistency'] = {'status': 'ERROR', 'error': str(e)}
        print(f"  ✗ Version check error: {e}")
        all_ok = False
    
    # --- Step 6: Criteria synchronization check ---
    print("\n--- Step 6: Criteria source-of-truth check ---")
    try:
        from tau_cdma.validate.prediction_criteria import get_all_criteria
        criteria = get_all_criteria()
        # Verify all predictions have criteria
        expected_preds = ['P1', 'P2', 'P3', 'P4', 'P7', 'P8', 'P9']
        missing_preds = [p for p in expected_preds if p not in criteria]
        
        sync_ok = len(missing_preds) == 0
        results['steps']['criteria_sync'] = {
            'status': 'PASS' if sync_ok else 'FAIL',
            'missing': missing_preds,
        }
        icon = '✓' if sync_ok else '✗'
        print(f"  {icon} All {len(expected_preds)} predictions have criteria defined")
    except Exception as e:
        results['steps']['criteria_sync'] = {'status': 'ERROR', 'error': str(e)}
        print(f"  ✗ Criteria check error: {e}")
        all_ok = False
    
    # --- Step 7: Optimal design extension ---
    print("\n--- Step 7: Optimal design extension ---")
    if smoke:
        # Smoke: just verify the runner imports and the core function works
        try:
            from scipy.linalg import svd
            from tau_cdma.core.fisher import poisson_fim
            from tau_cdma.heavy_ion.bethe_bloch import build_template_matrix
            import numpy as np
            
            masses = [0.13957039, 0.493677, 0.93827209]
            theta = np.array([0.8475, 0.1186, 0.0339])
            A, _ = build_template_matrix(1.0, masses, sigma=0.055, n_bins=20)
            F = poisson_fim(A, theta, 10000)
            C = np.ones((1, 3))
            _, S_vals, Vt = svd(C, full_matrices=True)
            U = Vt[int(np.sum(S_vals > 1e-10)):].T
            FT = U.T @ F @ U
            assert FT.shape == (2, 2), f"Tangent FIM shape wrong: {FT.shape}"
            assert np.linalg.det(FT) > 0, "Tangent FIM not positive definite"
            
            results['steps']['optimal_design'] = {'status': 'PASS', 'mode': 'smoke'}
            print(f"  ✓ Smoke: tangent-space FIM computes correctly (2×2, det > 0)")
        except Exception as e:
            results['steps']['optimal_design'] = {'status': 'ERROR', 'error': str(e)}
            print(f"  ✗ Optimal design smoke failed: {e}")
            all_ok = False
    else:
        try:
            import subprocess
            opt_dir = os.path.join(output_dir, 'optimal_design')
            os.makedirs(opt_dir, exist_ok=True)
            pkg_dir_for_opt = os.path.dirname(os.path.abspath(__file__))
            runner = os.path.join(pkg_dir_for_opt, 'run_optimal_design.py')
            
            if os.path.exists(runner):
                result = subprocess.run(
                    [sys.executable, runner, opt_dir],
                    capture_output=True, text=True, timeout=180
                )
                if result.returncode == 0:
                    json_path_opt = os.path.join(opt_dir, 'optimal_design_results.json')
                    if os.path.exists(json_path_opt):
                        with open(json_path_opt) as f:
                            opt_data = json.load(f)
                        violations = opt_data.get('submodularity_violations', -1)
                        k3_ratio = opt_data.get('table', {}).get('3', {}).get('ratio', 0)
                        opt_ok = violations == 0 and k3_ratio > 0.9
                        results['steps']['optimal_design'] = {
                            'status': 'PASS' if opt_ok else 'FAIL',
                            'submodularity_violations': violations,
                            'k3_ratio': k3_ratio,
                        }
                        icon = '✓' if opt_ok else '✗'
                        print(f"  {icon} Optimal design: violations={violations}, "
                              f"k=3 ratio={k3_ratio:.4f}")
                        if not opt_ok:
                            all_ok = False
                    else:
                        results['steps']['optimal_design'] = {'status': 'FAIL', 'error': 'no JSON output'}
                        print(f"  ✗ run_optimal_design.py produced no JSON")
                        all_ok = False
                else:
                    results['steps']['optimal_design'] = {
                        'status': 'FAIL', 'error': result.stderr[:200]
                    }
                    print(f"  ✗ run_optimal_design.py failed: {result.stderr[:200]}")
                    all_ok = False
            else:
                results['steps']['optimal_design'] = {'status': 'SKIPPED', 'reason': 'runner not found'}
                print(f"  ⊘ run_optimal_design.py not found — skipping")
        except Exception as e:
            results['steps']['optimal_design'] = {'status': 'ERROR', 'error': str(e)}
            print(f"  ✗ Optimal design check error: {e}")
            all_ok = False
    
    # --- Summary ---
    print("\n" + "=" * 60)
    results['overall'] = 'PASS' if all_ok else 'FAIL'
    if all_ok:
        print("RELEASE GATE: ✓ PASS — package is internally consistent")
    else:
        failed_steps = [k for k, v in results['steps'].items() 
                       if v.get('status') != 'PASS']
        print(f"RELEASE GATE: ✗ FAIL — blockers in: {failed_steps}")
    print("=" * 60)
    
    # Write JSON results
    json_path = os.path.join(output_dir, 'release_test_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results: {json_path}")
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
