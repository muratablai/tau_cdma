"""
tau_cdma: The Standard Model as a Communication Protocol
========================================================

Computational framework for treating particle physics reconstruction
as a CDMA multiuser detection problem.

Architecture
------------
The codebase has three layers, each building on the one below:

  tau_cdma/
    core/           <- Layer 0: Reusable CDMA-PID framework machinery
      fisher          Poisson FIM, CRB (unconstrained + constrained)
      interference    R-matrix, multiuser efficiency (Verdu 1998)
      shannon         Classification MI, Fano, Bayes confusion, JSD
      aliasing        Template aliasing analysis
      cascade         Cascade decay Schur complement
      erasure         Geometric vs random erasure
      emergent        Emergent quantum number discovery (polar codes)
      nmf             Blind decomposition (NMF)
      spreading       Spreading factor analysis (SF = m/Gamma)
      simulate        Monte Carlo event generation

    tau/            <- Layer 1: tau decay benchmark (P1-P4, P7-P8, eigenvalue collapse)
      templates       7-channel analytic templates (PDG 2024)
      ml_receiver     NN receiver hierarchy
      benchmark       Standard benchmark setup

    heavy_ion/      <- Layer 2: ALICE Pb-Pb pi/K/p PID (P9-P16)
      bethe_bloch     ALEPH 5-parameter dE/dx, TPC templates
      tof             TOF m^2 templates, TPC+TOF joint Fisher
      centrality      ALICE centrality configs, measurability landscape

    validate/       <- Prediction validation suites
      tau_predictions         P1-P4, P7-P8, eigenvalue collapse
      heavy_ion_predictions   P9-P16 (18 checks)

Extending
---------
To add a new benchmark system (e.g. Belle II, LHCb):
  1. Create tau_cdma/<experiment>/ with templates and physics
  2. Write sweep functions that feed into core.fisher
  3. Add tau_cdma/validate/<experiment>_predictions.py

Corrections (C1-C7, C8-C11)
--------------------
  C1: Wyner-Ziv Markov chain U-X-Y (en-dash, not arrow)
  C2: CRB-based multiuser efficiency (decorrelator AME)
  C3: Tightened Fano inequality with H_b(P_e)
  C4: tau->pi2pi0 BR = 0.0926 (PDG 2024)
  C5: rho(770) mass = 775.11 MeV, width = 149.1 MeV
  C6: K+/- mass uncertainty +/- 0.015 MeV
  C7: FT0 start-time 17 ps (pp), 4.4 ps (Pb-Pb)
  C8: tau mass = 1776.93 MeV (PDG 2024, was 1776.86)
  C9: TOF m^2 resolution: sigma(m^2) = 2p^2 c sigma_t / (beta L)
      (was beta^3 gamma^2 — incorrect relativistic factor)
  C10: TOF resolution = 60 ps (Run 2 Pb-Pb), was 80 ps (Run 1)
  C11: TOF path length = 3.8 m (ALICE standard), was 3.7 m
"""
__version__ = "0.5.0"
