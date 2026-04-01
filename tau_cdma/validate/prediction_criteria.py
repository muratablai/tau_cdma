"""
prediction_criteria.py — Centralized Success Criteria for Tau-Side Predictions
================================================================================

This file defines success criteria for the tau-decay predictions validated in
tau_predictions.py: P1, P2, P3, P4, P7, P8, and the eigenvalue-collapse check
(code-level P9, corresponding to aliasing aspects of formalism P1-P2).

The heavy-ion predictions (P10-P16) carry their own criteria internally in
heavy_ion_predictions.py. A future release will unify all criteria here.

Formalism P5 (coding gain), P6 (critical luminosity), and P9 (emergent quantum
numbers) are not yet implemented computationally.

Version: v4.5 (post-collaboration-review)
"""

# =====================================================================
# P1: Multiuser Efficiency Ordering (multi-dimensional observables)
# =====================================================================
P1_CRITERIA = {
    'description': 'Multiuser efficiency ordering in (m_vis, n_trk, PID) space',
    
    # Quantitative near-equality conditions (augmented space)
    'eta_12_rel_diff_max': 0.10,    # |η₁ − η₂|/max(η₁, η₂) < 0.1
    'eta_56_rel_diff_max': 0.20,    # |η₅ − η₆|/max(η₅, η₆) < 0.20
    
    # Ordering in augmented space: η₃ > η₁ > η₄ > η₅
    'pi_dominates_augmented': True,  # η₃ > η₁ (π has most distinctive template)
    'hadronic_ordering': True,       # η₃ > η₄ > η₅
    
    # 1D corollary (observable dependence)
    'pi_dominates_1d_min': 0.9,     # η_π > 0.9 in 1D
    'leptonic_aliased_1d_max': 0.05, # η_e, η_μ < 0.05 in 1D
    'R_emu_min': 0.99,              # R(e,μ) > 0.99 in 1D
    
    # Shannon classification
    'mu_accuracy_1d_max': 0.01,     # Bayes MAP μ accuracy < 1% in 1D (μ=0% theorem)
    'mu_accuracy_pid_min': 0.10,    # Bayes MAP μ accuracy > 10% with PID
    
    # Observable dependence
    'pid_eta_boost_min': 0.1,       # PID lifts e,μ η above this
}

# =====================================================================
# P2: Aliasing Order
# =====================================================================
P2_CRITERIA = {
    'description': 'Aliasing order: which channel pairs separate first as M increases',
    
    # Template distance ordering at coarsest binning
    'emu_most_aliased': True,       # (e,μ) pair has smallest d²
    'ordering_check': True,          # d²(a₁,π2π⁰) < d²(ρ,a₁) < d²(ρ,π) < d²(e,π)
    
    # M* ordering (non-strict: ≤ because τ hadronic widths are all O(100 MeV))
    'mstar_ordering_strict': False,  # ≤, not <
    
    # Monte Carlo correlation sign flip
    'mc_degenerate_corr_min': 0.9,  # corr(e,μ) > 0.9 at M=3
    'mc_tradeoff_corr_max': -0.9,   # corr(e,μ) < -0.9 at M=200
    'sign_flip_required': True,      # correlation must flip sign
    'sum_stability': True,           # σ(e+μ) < σ_e at high M
}

# =====================================================================
# P3: Geometric Erasure Penalty
# =====================================================================
P3_CRITERIA = {
    'description': 'Geometric (central-window) erasure produces strongly species-dependent '
                   'degradation: species outside the kept window are severely impacted',
    
    'alpha_test': 0.5,              # Test at α = 0.5 (formalism criterion)
    'min_ratio_hadronic': 1.2,      # r_k > 1.2 for at least one hadronic channel
    'hadronic_channels': [2, 3, 4, 5, 6], # all non-leptonic: π, ρ, a₁, π2π⁰, other
    'leptonic_channels': [0, 1],           # e, μ
    'max_ratio_on_hadronic': True,  # max_k(r_k) should be on a hadronic channel
    # Note: the central-window model (360–1260 MeV at α=0.5) removes both
    # low-mass and high-mass bins. Species below the low cut (e, μ, π) may be
    # killed (r_k = ∞), while species centered in the window (ρ) may improve.
    # This species-dependent pattern is the prediction, not a blanket
    # "geometric worse than random" claim.
}

# =====================================================================
# P4: Cascade Bottleneck
# =====================================================================
P4_CRITERIA = {
    'description': 'Cascade bottleneck: I₂(a₁→3π) < I₁(τ→a₁ν)',
    
    'ratio_min': 2.0,               # I₁/I₂ > 2 (bottleneck at stage 2)
    'bottleneck_stage': 2,           # argmin_s I_s = stage 2
}

# =====================================================================
# P7: Optimal Binning
# =====================================================================
P7_CRITERIA = {
    'description': 'Optimal binning M_opt correlates with spreading factor SF_k',
    
    # NOTE: The correlation criterion applies to resonance channels only,
    # because non-resonance channels (e, μ, π, other) do not have 
    # well-defined spreading factors. The τ system has limited SF dynamic
    # range (2.4-5.2), so the correlation is inherently weak. A stronger
    # test requires systems with wider SF range (e.g., charm mesons).
    'resonance_channels': [3, 4, 5],  # ρ, a₁, π2π⁰
    'corr_positive': True,             # Correlation must be positive
    'saturation_threshold': 0.90,      # 90% of peak Fisher info
    'all_saturated_by_M': 200,         # All channels reach 90% by M=200
    'pi_saturates_early_M': 5,         # π saturates >80% by M=5
    'pi_saturation_min': 0.80,
    
    # Physical ordering: broad channels saturate before narrow
    'broad_before_narrow': True,       # M_opt(a₁) ≤ M_opt(ρ)
}

# =====================================================================
# P8: Blind Recovery, NN Receiver, MAP Classification
# =====================================================================
P8_CRITERIA = {
    'description': 'Blind NMF recovery and classification validation',
    
    # NMF blind recovery
    'K_best_tolerance': 3,          # |K_best - 7| ≤ 3
    'K_true': 7,
    
    # Template recovery: L1 normalized error
    # NOTE: Blind NMF on heavily overlapping templates (e≈μ, and broad
    # hadronic channels with significant overlap) achieves recovery errors
    # of ~1.0 per channel. This is expected: NMF cannot separate components
    # that are not geometrically distinguishable in the nonnegative cone.
    # The criterion reflects achievable blind recovery, not perfect recovery.
    # The aliased e-μ pair cannot be recovered individually (this is itself
    # a validation of the aliasing prediction).
    'recovery_mean_max': 1.5,            # mean across all channels < 1.5
    'recovery_some_below_1': 2,          # at least 2 channels with error < 1.0
    'non_aliased_channels': [2, 3, 4, 5, 6],  # π, ρ, a₁, π2π⁰, other
    
    # Shannon classification
    'mu_accuracy_1d_max': 0.01,     # μ = 0% theorem validation
    'eta_map_corr_min': 0.7,        # corr(η, MAP accuracy) > 0.7
    
    # Discovery
    'residual_significance_min': 5.0,  # >5σ residual near injected signal
    'mass_recovery_tolerance': 100,     # within 100 MeV of true mass
    'lr_significance_min': 5.0,         # LR scan significance >5σ
}

# =====================================================================
# P9: Aliasing as Eigenvalue Collapse
# =====================================================================
P9_CRITERIA = {
    'description': 'Eigenvalue collapse of R reveals aliasing structure',
    
    'lambda_min_max': 0.01,         # λ_min < 0.01 (aliasing exists)
    'aliased_pair': {0, 1},         # e-μ direction
    'concentration_min': 0.99,      # >99% concentrated on aliased pair
    'pid_lift_min': 10,             # PID lifts λ_min by >10×
    'pr_increases_with_pid': True,  # Participation ratio increases
    'nmf_pr_tolerance': 0.5,        # NMF PR within 50% of true
}


def get_all_criteria():
    """Return all prediction criteria as a dictionary."""
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
    """Format a single criterion check result."""
    icon = '✓' if passed else '✗'
    return f"    {icon} [{pred_id}] {criterion_name}: {value}"
