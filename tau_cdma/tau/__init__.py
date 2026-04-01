"""
tau_cdma.tau — τ Lepton Decay Benchmark (Layer 1)
===================================================

The primary validation system: 7 τ decay channels with known
PDG branching ratios. This is where the CDMA ↔ PID analogy
was first established and predictions P1-P9 validated.

Modules:
  templates   — Analytic 7-channel τ decay templates (PDG 2024)
  ml_receiver — NN receiver hierarchy (matched filter → decorrelator → MMSE → ML)
  benchmark   — Standard benchmark setup (templates + BR + background)
"""
