"""
tau_cdma.heavy_ion — ALICE Pb-Pb Charged Hadron PID (Layer 2)
===============================================================

Extension to a real running experiment: π/K/p identification
in ALICE at LHC. Introduces momentum-dependent Bethe-Bloch
templates, TOF fusion, and centrality as a control parameter.
Predictions P9-P16.

Modules:
  bethe_bloch — ALEPH 5-parameter dE/dx, TPC templates, crossing finder
  tof         — TOF m² templates, TPC+TOF joint Fisher, partial coverage
  centrality  — ALICE Pb-Pb centrality configs, momentum/centrality sweeps

To add a new benchmark system (e.g. Belle II, LHCb):
  Create a sibling package at tau_cdma/<experiment>/ following
  this same pattern — define templates, physics parameters,
  and a sweep function that feeds into core.fisher.
"""
