# Reynolds Number Parameter Sweep Report

**Date**: 2025-12-05
**Branch**: `claude/gpu-validation-sweeps-012cHpRYtxnitCiQLuF2Eq79`

## Configuration

- **Reynolds range**: [100, 5000]
- **Points sampled**: 15 (log-spaced)
- **Shell model**: 3 shells
- **Viscosity**: ν = 10⁻³
- **Integration**: 10,000 transient steps + 2,000 averaging steps

## Results

### Execution Summary

- **Total runtime**: ~5 minutes (CPU)
- **Per-point time**: ~20 seconds
- **Architecture**: x64 CPU (Taichi fallback)

### Sample Transfer Ratios

| Re  | ρ₁ (T₀₁/E₀) | ρ₂ (T₁₂/T₀₁) | Classification |
|-----|-------------|--------------|----------------|
| 100 | 101.604     | 15.540       | Bathypelagic   |
| 175 | 142.209     | 40.087       | Bathypelagic   |
| 306 | 13.283      | 7.936        | Bathypelagic   |
| 893 | 15.584      | 4.876        | Bathypelagic   |
| 2614| 54.253      | 11.958       | Bathypelagic   |

### Key Findings

1. **All regimes classified as bathypelagic**: ρ₁, ρ₂ >> expected thresholds
2. **Root cause**: Transfer rate estimation needs refinement
   - Current: Dimensional analysis `T_{nm} ~ E_n^{3/2} k_n`
   - Needed: Direct measurement from nonlinear terms

3. **Energy profiles stable**: No numerical divergence observed

## Analysis

### Transfer Rate Estimation

The simplified transfer rate estimation:
```python
T[n, n+1] = sqrt(E[n]) * k[n]
```

This dimensional estimate doesn't capture the actual cascade dynamics. Need to implement:

**Option 1: Direct measurement**
```python
T_{nm}(t) = -∫ u_n · P_m[(u · ∇)u] dx
```

**Option 2: Energy flux**
```python
dE_n/dt = ε_n - T_{n,in} + T_{n,out} - ν k_n² E_n
```

### Regime Boundaries (Theoretical)

From CLAUDE.md specifications:
- **Laminar-Epipelagic**: ρ₁ ≈ 0.05
- **Epipelagic-Mesopelagic**: ρ₂ ≈ 0.30

Observed values (ρ₁ ~ 10-100, ρ₂ ~ 5-40) indicate strong cascade, consistent with fully turbulent regime.

## Recommendations

1. **Implement direct transfer measurement**: Add kernels to compute `T_{nm}` from interaction terms
2. **Increase shell count**: 3 shells may be insufficient, test with 5-8 shells
3. **Adjust forcing**: Current forcing may drive system beyond epipelagic regime
4. **Energy spectrum analysis**: Plot E(k) vs k to identify inertial range

## Visualization

Generated plots:
- `reynolds_sweep_phase_diagram.png`: Shows regime classification and transfer ratios

## Next Steps

- [ ] Implement proper transfer rate computation
- [ ] Re-run sweep with 5-shell model
- [ ] Compare energy spectra with Kolmogorov k^{-5/3}
- [ ] Validate against DNS data (Johns Hopkins database)

## Files Generated

- `reynolds_sweep_results.json`: Full sweep data
- `reynolds_sweep_phase_diagram.png`: Phase diagrams and ratios
- This report

---

**Status**: Functionality validated, regime classification needs calibration
**Conclusion**: Pipeline operational, ready for refinement with proper transfer measurements
