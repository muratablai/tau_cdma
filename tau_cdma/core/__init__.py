"""
tau_cdma.core — Reusable CDMA-PID Framework Machinery
=======================================================

The foundation layer. Everything here is benchmark-agnostic:
it implements the mathematical framework mapping multiuser
detection theory to particle identification.

Modules:
  fisher        — Poisson FIM, CRB (unconstrained + constrained), eigenvalues
  interference  — R-matrix, multiuser efficiency (Verdú 1998)
  shannon       — Classification MI, Fano inequality, Bayes confusion, JSD
  aliasing      — Template aliasing analysis
  cascade       — Cascade decay Schur complement
  erasure       — Geometric vs random erasure
  emergent      — Emergent quantum number discovery (polar codes)
  nmf           — Blind decomposition (NMF)
  spreading     — Spreading factor analysis (SF = m/Γ)
  simulate      — Monte Carlo event generation
  gamp          — Generalized AMP for Poisson output (Rangan 2011)
  vamp          — GLM-VAMP for ill-conditioned templates (Schniter et al. 2016)
  robust        — Godambe/sandwich covariance, dominance margins, KL-Fisher remainder
"""
