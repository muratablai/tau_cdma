# Changelog

## v0.5.1 (2026-04-08)

### Fixed
- **Breit-Wigner Jacobian**: Hadronic templates now include the `ds = 2m dm`
  Jacobian factor when converting the relativistic BW from `s`-space to mass
  density. This is the correct physics for templates defined on a mass grid.
  Impact on benchmark: MAP 78.0%→78.2%, MI 1.894→1.911 bits. Qualitative
  conclusions unchanged; all 7 prediction families still pass.
- **pyproject.toml build backend**: Changed from
  `setuptools.backends._legacy:_Backend` (broken on most environments) to
  `setuptools.build_meta`.
- **Docstring in `core/fisher.py`**: Corrected `crb_multiuser_efficiency()`
  docstring to state that the CRB-based and R-matrix efficiencies are
  algebraically equivalent when R is invertible:
  `[R⁻¹]_kk = F_kk · [F⁻¹]_kk`.

### Note on submitted papers
Papers submitted to EPJ C and CPC use v0.5.0 benchmark numbers. The v0.5.1
Jacobian correction shifts hadronic-channel values modestly (see table below).
Paper numbers will be updated during the revision round if requested.

| Quantity | v0.5.0 | v0.5.1 |
|----------|--------|--------|
| η_ρ | 0.84 | 0.85 |
| η_a₁ | 0.12 | 0.12 |
| η_π2π⁰ | 0.29 | 0.30 |
| κ(R) | 35.8 | 35.5 |
| MAP overall | 78.0% | 78.2% |
| MI | 1.894 bits | 1.911 bits |
| Fano floor | 11.4% | 11.1% |
| η-acc r | 0.92 | 0.89 |
| Cascade I₁/I₂ | 1.39 | 1.56 |

## v0.5.0 (2026-04-07)

- Visible-mass template rebuild: delta functions at rest masses for leptonic
  channels, Breit-Wigner resonances with physical thresholds for hadronic.
- Seven prediction families, 41 individual checks, all passing.
- First submission to EPJ C (Papers 1 & 3) and CPC (Paper 2).
