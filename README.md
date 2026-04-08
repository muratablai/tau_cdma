# tau_cdma: Particle Identification through Fisher-Information Geometry

Computational framework mapping template-based particle identification to multiuser detection theory.

Implements predictions from the **Unified Formalism v4.5**, validated on tau-decay (7 channels) and ALICE Pb-Pb (pi/K/p) benchmarks.

## Installation

```bash
pip install -e .
# or just:
pip install numpy scipy matplotlib scikit-learn
```

## Quick Start

```python
# Run tau prediction validations (P1-P4, P7-P8 + eigenvalue collapse)
from tau_cdma.validate.tau_predictions import run_all
results = run_all(quick=True)

# Run heavy-ion predictions (P10-P16 in the formalism)
from tau_cdma.validate.heavy_ion_predictions import validate_predictions
hi_results = validate_predictions()
```

### Command line
```bash
# Full validation suite (tau + heavy-ion)
python -m tau_cdma.validate --quick

# Verbose output
python -m tau_cdma.validate
```

## Prediction Numbering

The formalism (Section 12) defines predictions P1-P4, P7-P8 plus eigenvalue collapse for tau decay, and P9-P16 for
heavy-ion PID. The code validators cover:

- **Tau side** (`tau_predictions.py`): P1, P2, P3, P4, P7, P8, plus a structural
  eigenvalue-collapse check (labeled P9 in the code but corresponding to the
  eigenvalue/aliasing aspects of P1-P2, not to formalism P9).
- **Heavy-ion side** (`heavy_ion_predictions.py`): P10-P16 (labeled P9-P16 in the
  code for historical reasons).
- **Not yet implemented**: Formalism P5 (coding gain), P6 (critical luminosity),
  and P9 (emergent quantum numbers / latent dimensionality).

Success criteria for the tau-side predictions are centralized in
`tau_cdma/validate/prediction_criteria.py`. The heavy-ion validator carries its
own criteria internally. A future release will unify all criteria into the
single-source-of-truth module.

## Package Structure

```
tau_cdma/
  core/                    # Benchmark-agnostic machinery
    fisher.py              # Poisson FIM, CRB, constrained CRB, eigenvalues
    interference.py        # R-matrix, multiuser efficiency
    shannon.py             # Classification MI, Fano, Bayes confusion, JSD
    gamp.py                # Generalized AMP for Poisson output
    vamp.py                # GLM-VAMP, robust to ill-conditioned A
    robust.py              # Godambe/sandwich, dominance margins, KL remainder
    aliasing.py            # Template aliasing analysis
    cascade.py             # Cascade decay bottleneck (Schur complement)
    erasure.py             # Geometric vs random erasure
    nmf.py                 # Blind decomposition (Poisson NMF)
    spreading.py           # Spreading factor (SF = m/Gamma)
    simulate.py            # Monte Carlo event generation
    emergent.py            # Emergent quantum number discovery

  tau/                     # Tau lepton decay benchmark (Layer 1)
    templates.py           # 7-channel visible mass templates
    benchmark.py           # Benchmark setup and configuration
    ml_receiver.py         # Neural network receiver hierarchy

  heavy_ion/               # ALICE PID benchmark (Layer 2)
    bethe_bloch.py         # ALEPH dE/dx parameterization
    tof.py                 # TOF m^2 resolution
    centrality.py          # Central vs peripheral analysis

  validate/                # Prediction validation suite
    prediction_criteria.py # Centralized criteria for tau-side predictions
    tau_predictions.py     # P1-P4, P7-P8, eigenvalue collapse validators
    heavy_ion_predictions.py # P10-P16 validators
    __main__.py            # CLI entry point
```

## Version

- Code: v0.5.1
- Framework: v4.5 (Unified Formalism)
- Python: >=3.10

## Extension: Optimal Observable Selection

The optimal-design extension demonstrates submodular D-optimal observable selection
for template-based PID. To reproduce all results and figures:

```bash
python run_optimal_design.py [output_dir]
```

This generates `optimal_design_results.json` and Figures 1–4 for the extension draft.
