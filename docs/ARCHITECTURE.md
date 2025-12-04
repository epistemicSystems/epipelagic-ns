# Epipelagic Framework Architecture

## Overview

The epipelagic-ns framework is organized into modular components following the research phases outlined in the mega-prompt.

## Module Structure

### Core (`epipelagic/core/`)

Fundamental mathematical structures:

- **`complex.py`**: `CascadeComplex` class implementing cochain complex (C•, d•)
- **`spectral.py`**: `SpectralSequence` for E₂-degeneration analysis
- **`cohomology.py`**: Cohomology computation and Theorem C validation

**Key Classes**:
```python
CascadeComplex(n_shells, energies, transfers, wavenumbers)
  ├── differential_d0() -> ndarray
  ├── differential_d1() -> ndarray
  ├── compute_cohomology() -> (H0_basis, H1_basis, dim_H1)
  └── classify_regime() -> str
```

### Cascade (`epipelagic/cascade/`)

Shell cascade solvers and dynamics:

- **`shell_model.py`**: GOY/Sabra shell model equations
- **`solver.py`**: Time integration (RK4, adaptive stepping)
- **`taichi_solver.py`**: GPU-accelerated solver (>10⁶ steps/sec target)
- **`phase_diagram.py`**: Parameter space exploration

**Performance Targets**:
- CPU solver: ~10⁴ steps/sec
- GPU solver: >10⁶ steps/sec (Phase 1 requirement)

### Topology (`epipelagic/topology/`)

Persistent homology and topological invariants:

- **`persistent.py`**: Ripser integration, dim(H¹ₑₚᵢ) extraction

**Algorithm 1 Implementation**:
```python
extract_persistent_homology(velocity_field, threshold)
  1. Compute vorticity: ω = ∇ × u
  2. Build filtration from level sets
  3. Run Ripser
  4. Filter long bars
  5. Return dim(H¹ₑₚᵢ)
```

### Quantum (`epipelagic/quantum/`)

Quasi-particle formalism (Phase 3):

- Fock space construction
- Cascade Hamiltonian
- Time evolution operators

**Status**: Placeholder for Phase 3 implementation

### Langlands (`epipelagic/langlands/`)

Geometric Langlands machinery (Phase 4):

- Fourier-Mukai transforms
- Tropical degeneration
- Functorial correspondence

**Status**: Placeholder for Phase 4 implementation

### Visualization (`epipelagic/visualization/`)

Interactive visualization tools:

- Phase diagram plotting
- Persistence barcode visualization
- Energy spectrum analysis

**Planned Tools**:
- Houdini HDA integration
- Plotly interactive dashboards
- 3D vorticity rendering

### Utils (`epipelagic/utils/`)

Common utilities:

- I/O (HDF5 data handling)
- Validation (energy conservation checks)
- Configuration management

## Data Flow

### Typical Workflow

```
Velocity Field u(x,t)
    ↓
[Shell Decomposition]
    ↓
Shell Energies Eₙ + Transfers Tₙₘ
    ↓
[CascadeComplex]
    ↓
Cohomology H¹(C•)
    ↓
dim(H¹ₑₚᵢ) ← Topological Invariant
```

### Phase 1 Pipeline

```
[ShellCascade] → [CascadeSolver] → [Steady State]
                         ↓
                [CascadeComplex]
                         ↓
              [Regime Classification]
                         ↓
            Epipelagic / Mesopelagic / ...
```

## Testing Strategy

### Unit Tests (`tests/unit/`)

- Test individual components in isolation
- Fast execution (<1 sec per test)
- High coverage target (>90%)

### Integration Tests (`tests/integration/`)

- Test end-to-end workflows
- Validate algorithm correctness
- Compare with known solutions

### Benchmarks (`tests/benchmarks/`)

- Performance regression detection
- GPU vs CPU comparisons
- Scalability tests

## Extension Points

### Adding New Cascade Models

1. Subclass `ShellCascade`
2. Override `rhs()` method
3. Implement `_interaction_term()`

### Adding Topology Algorithms

1. Create new module in `epipelagic/topology/`
2. Follow Ripser interface pattern
3. Add validation against synthetic data

### GPU Kernels

1. Use Taichi `@ti.kernel` decorator
2. Ensure data-parallel structure
3. Benchmark against CPU baseline

## Performance Considerations

### Memory Layout

- Shell velocities: Complex arrays (real + imag components)
- Transfer matrices: Dense (small N) or sparse (large N)
- GPU fields: Taichi manages automatically

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| RK4 step | O(N²) | N = n_shells |
| Cohomology | O(N³) | SVD-based |
| Persistent homology | O(M³) | M = n_points |
| Phase diagram | O(P × T) | P = param points, T = transient |

## Future Architecture

### Phase 2 Extensions

- Parallel parameter sweeps (Dask/Ray)
- Large-scale DNS data processing
- Distributed persistent homology

### Phase 3 Additions

- Quantum circuit integration
- Symbolic computation (SymPy)
- Feynman diagram generation

### Phase 4 Framework

- Category theory abstractions
- Tropical geometry toolkit
- Langlands functor implementations
