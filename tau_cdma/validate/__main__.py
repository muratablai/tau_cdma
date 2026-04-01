"""
Entry point for: python -m tau_cdma.validate [--quick] [--verbose]

Runs the complete prediction validation suite.
"""
import sys


def main():
    quick = '--quick' in sys.argv
    verbose = '--verbose' in sys.argv or '-v' in sys.argv or '--quick' not in sys.argv
    
    from tau_cdma.validate.tau_predictions import run_all
    summary = run_all(quick=quick, verbose=verbose)
    
    # Also run heavy-ion predictions
    print("\n")
    from tau_cdma.validate.heavy_ion_predictions import validate_predictions
    hi_results = validate_predictions(verbose=verbose)
    
    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETE VALIDATION SUMMARY")
    print("=" * 60)
    
    n_pass = sum(1 for r in summary.values() if r.get('passed'))
    n_total = len(summary)
    print(f"  τ predictions: {n_pass}/{n_total} passed")
    print(f"  Heavy-ion predictions: see above")
    
    all_passed = all(r.get('passed') for r in summary.values())
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
