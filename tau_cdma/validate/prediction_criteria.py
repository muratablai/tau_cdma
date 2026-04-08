"""
prediction_criteria.py — Success Criteria for Tau-Side Predictions (v0.5.0)
============================================================================

Updated for physically correct visible-mass templates.

Key physics changes from v0.4.9:
  - Leptonic templates are delta functions at rest mass (not Michel spectra)
  - e and mu are well-separated (6.7 sigma), NOT aliased
  - Most aliased direction: hadronic cluster (a1, pi2pi0, other)
  - mu-pi near-far effect at 2.3 sigma separation
  - No zero-recall species at default resolution (sigma=15 MeV)
  - Cascade bottleneck I₁/I₂ = 1.4 (consistent PMF normalization + threshold)

Version: v0.5.0 (visible-mass benchmark rebuild)
"""

# =====================================================================
# P1: Multiuser Efficiency Structure
# =====================================================================
P1_CRITERIA = {
    'description': 'Multiuser efficiency reveals identifiability landscape',

    # Electron is isolated (delta at 0.5 MeV, far from everything)
    'eta_e_min': 0.99,              # eta_e > 0.99

    # mu-pi near-far pair: nearly equal eta but asymmetric confusion
    'eta_mu_pi_rel_diff_max': 0.05, # |eta_mu - eta_pi|/max < 5%
    'eta_mu_min': 0.8,              # eta_mu > 0.8 (not aliased, just partial overlap)
    'eta_pi_min': 0.8,              # eta_pi > 0.8

    # Hadronic cluster: a1 is most degraded
    'eta_a1_max': 0.15,             # eta_a1 < 0.15 (severely confused)
    'eta_a1_is_lowest': True,       # a1 has the worst eta among all species

    # Near-far demonstration: mu recall > pi recall (prior advantage)
    'mu_recall_gt_pi': True,        # mu accuracy > pi accuracy in MAP

    # No zero-recall in default benchmark
    'no_zero_recall': True,         # all species have recall > 0

    # Shannon classification
    'mi_min': 1.5,                  # I(K;X) > 1.5 bits
    'fano_max': 0.20,               # Fano floor < 20%
}

# =====================================================================
# P2: Aliasing Order
# =====================================================================
P2_CRITERIA = {
    'description': 'Template aliasing structure in visible-mass space',

    # Hadronic channels alias first (broad overlapping resonances)
    'hadronic_most_aliased': True,  # most aliased pair is among {rho, a1, pi2pi0, other}

    # e-mu are NOT aliased (well-separated in visible mass)
    'emu_not_aliased': True,        # d^2(e,mu) >> d^2(hadronic pairs)

    # Resolution degradation: hadronic pair correlation (no sign flip in v0.5.0)
    'hadronic_correlation_negative': True,  # most aliased pair anti-correlated at M=200
    'sum_stability': True,
}

# =====================================================================
# P3: Geometric Erasure Penalty
# =====================================================================
P3_CRITERIA = {
    'description': 'Central-window erasure produces species-dependent degradation',

    'alpha_test': 0.5,
    'min_ratio_hadronic': 1.2,
    'hadronic_channels': [2, 3, 4, 5, 6],
    'leptonic_channels': [0, 1],
    'max_ratio_on_hadronic': True,
}

# =====================================================================
# P4: Cascade Bottleneck
# =====================================================================
P4_CRITERIA = {
    'description': 'Cascade bottleneck: stage 2 retains less Fisher information than stage 1',

    # Bottleneck ratio I1/I2
    'bottleneck_min': 1.1,          # I1/I2 > 1.1 (stage 2 is the bottleneck)
    'bottleneck_max': 20.0,         # I1/I2 < 20 (sanity check)

    # Stage 2 is limiting
    'bottleneck_at_stage2': True,

    # Spreading factor range
    'sf_range': (2.0, 10.0),

    # Width ordering: broader resonance → worse bottleneck
    'width_ordering': True,
}

# =====================================================================
# P7: Optimal Binning
# =====================================================================
P7_CRITERIA = {
    'description': 'Optimal binning M_opt correlates with spreading factor SF_k',

    'resonance_channels': [3, 4, 5],  # rho, a1, pi2pi0
    'corr_positive': True,
    'saturation_threshold': 0.90,
    'all_saturated_by_M': 200,
    'pi_saturates_early_M': 5,
    'pi_saturation_min': 0.80,
    'broad_before_narrow': True,
}

# =====================================================================
# P8: Blind NMF Discovery
# =====================================================================
P8_CRITERIA = {
    'description': 'Blind NMF recovery and classification validation',

    # NMF blind recovery
    'K_best_tolerance': 3,          # |K_best - 7| <= 3
    'K_true': 7,

    # Template recovery: L1 normalized error
    # Broad hadronic channels with significant overlap achieve recovery errors
    # of ~1.0 per channel. The hadronic cluster (a1, pi2pi0, other) cannot be
    # perfectly separated (this validates the aliasing prediction).
    'recovery_mean_max': 1.5,
    'recovery_some_below_1': 2,     # at least 2 channels with error < 1.0
    'non_aliased_channels': [0, 1, 2, 3],  # e, mu, pi, rho (well-separated)

    # Template matching quality
    'template_match_min': 0.2,

    # Shannon classification
    'mu_accuracy_1d_min': 0.50,     # mu recall > 50% (no collapse in v0.5.0)
    'eta_map_corr_min': 0.5,        # corr(eta, MAP accuracy) > 0.5

    # NMF-based accuracy
    'nmf_accuracy_min': 0.20,

    # Discovery
    'residual_significance_min': 5.0,
    'mass_recovery_tolerance': 100,
    'lr_significance_min': 5.0,
}

# =====================================================================
# P9: Eigenvalue Collapse Diagnostic
# =====================================================================
P9_CRITERIA = {
    'description': 'Eigenvalue structure reveals hadronic confusion cluster',

    # Smallest eigenvalue: NOT near-zero (no complete degeneracy at sigma=15)
    'lambda_min_max': 0.15,         # lambda_min < 0.15 (partial aliasing)
    'lambda_min_min': 0.01,         # lambda_min > 0.01 (not fully collapsed)

    # Most aliased direction should involve hadronic channels
    'aliased_direction_hadronic': True,  # eigvec of lambda_min dominated by {a1, pi2pi0, other}
    'hadronic_concentration_min': 0.70,  # at least 70% weight on hadronic channels

    # Condition number: moderate (not astronomical)
    'kappa_max': 100.0,             # kappa(R) < 100
    'kappa_min': 10.0,              # kappa(R) > 10 (some aliasing exists)

    # Participation ratio
    'pr_min': 3.0,                  # PR > 3 effective channels
    'pr_max': 6.0,                  # PR < 6
}


def get_all_criteria():
    """Return all prediction criteria as a dict."""
    return {
        'P1': P1_CRITERIA,
        'P2': P2_CRITERIA,
        'P3': P3_CRITERIA,
        'P4': P4_CRITERIA,
        'P7': P7_CRITERIA,
        'P8': P8_CRITERIA,
        'P9': P9_CRITERIA,
    }


def format_criterion(pred_id, criterion_name, value, passed):
    """Format a criterion check result for display."""
    status = '\u2713' if passed else '\u2717'
    return f"    {criterion_name}: {value} {status}"
