# Phase 2: Topology Implementation Tasks

**Project**: Epipelagic Turbulence Research - Phase 2 Topology
**Timeline**: Months 4-6 (12 weeks)
**Goal**: Prove finiteness bound, integrate persistent homology, extract H¹ₑₚᵢ from DNS data
**Status**: Ready to implement
**Last Updated**: 2025-12-05

---

## Overview

Phase 2 focuses on establishing the topological foundations of epipelagic turbulence through persistent homology and cohomological analysis.

### Phase 1 Summary (Completed)
✅ Taichi GPU cascade solver implemented
✅ Reynolds number parameter sweeps functional
✅ Basic topology extraction pipeline (Ripser integration)
✅ Phase diagram generation capabilities

### Phase 2 Objectives

**Mathematical Goals** (Sonnet Agent):
- Prove Theorem C: dim(H¹ₑₚᵢ) ≤ C log(Re)
- Establish persistent homology ↔ cascade cohomology correspondence
- Develop rigorous theory of H¹ₑₚᵢ as topological invariant

**Computational Goals** (Haiku + Code Agents):
- Extract dim(H¹ₑₚᵢ) from synthetic and DNS turbulence
- Validate finiteness bound empirically across Re ∈ [100, 10⁶]
- Build production-grade topology extraction pipeline
- Create interactive visualizations (Houdini/Plotly)

---

## Table of Contents

1. [Prerequisites Check](#prerequisites-check)
2. [Week 1-2: DNS Data Integration](#week-1-2-dns-data-integration)
3. [Week 3-4: Advanced Persistent Homology](#week-3-4-advanced-persistent-homology)
4. [Week 5-6: Realistic Vorticity Generation](#week-5-6-realistic-vorticity-generation)
5. [Week 7-8: Finiteness Validation](#week-7-8-finiteness-validation)
6. [Week 9-10: Houdini Visualization](#week-9-10-houdini-visualization)
7. [Week 11-12: Large-Scale Processing](#week-11-12-large-scale-processing)
8. [Success Criteria](#success-criteria)
9. [Risk Mitigation](#risk-mitigation)

---

## Prerequisites Check

**Before starting Phase 2, verify Phase 1 completion:**

- [x] **Taichi GPU Solver**
  - [x] 3-shell cascade model functional
  - [x] RK4 time integration stable
  - [x] Energy conservation verified
  - [x] CPU performance: ~700 steps/sec

- [x] **Parameter Sweeps**
  - [x] Reynolds number sweep implemented
  - [x] Phase diagram generation working
  - [x] Classification logic (needs calibration)

- [x] **Topology Pipeline**
  - [x] Ripser integration complete
  - [x] Vorticity field generation (basic)
  - [x] dim(H¹ₑₚᵢ) extraction functional

- [ ] **Infrastructure** (Phase 2 requirements)
  - [ ] GPU cluster access (for 10⁶ steps/sec)
  - [ ] DNS data storage configured (50GB+)
  - [ ] Houdini Python API installed
  - [ ] Gudhi library installed (advanced features)

**Status**: ✅ Phase 1 validated, ready for Phase 2

---

## Week 1-2: DNS Data Integration

**Goal**: Integrate real turbulence data from Johns Hopkins Turbulence Database

### Task 2.1: DNS Data Acquisition

**Priority**: P0 (Critical)
**Estimated Time**: 3 days
**Owner**: Code Agent

**Subtasks**:
- [ ] Set up JHTDB API access
  - [ ] Create account at http://turbulence.pha.jhu.edu/
  - [ ] Install pyJHTDB: `pip install pyJHTDB`
  - [ ] Configure authentication credentials
  - [ ] Test basic data query

- [ ] Download isotropic turbulence dataset
  - [ ] Dataset: `isotropic1024coarse` (Re_λ ≈ 418)
  - [ ] Time snapshot: t = 0.364 (peak dissipation)
  - [ ] Volume: 256³ subcube (manageable size)
  - [ ] Fields: velocity (u, v, w), vorticity (ω_x, ω_y, ω_z)
  - [ ] Storage: `data/dns/jhtdb_iso1024/`

- [ ] Data validation
  - [ ] Verify incompressibility: ∇·u = 0
  - [ ] Check vorticity: ω = ∇×u
  - [ ] Compute energy spectrum E(k)
  - [ ] Verify Kolmogorov k^(-5/3) inertial range

**Files to Create**:
- `epipelagic/data/jhtdb_loader.py` (API interface)
- `scripts/download_dns_data.py` (download script)
- `data/dns/README.md` (dataset documentation)

**Acceptance Criteria**:
- 256³ velocity field downloaded and validated
- Vorticity computed and verified
- Energy spectrum shows k^(-5/3) scaling

---

### Task 2.2: DNS Data Processing Pipeline

**Priority**: P0 (Critical)
**Estimated Time**: 4 days
**Owner**: Code Agent

**Subtasks**:
- [ ] Implement efficient data loading
  - [ ] HDF5 format reader (lazy loading)
  - [ ] Memory-mapped arrays for large fields
  - [ ] Parallel I/O for multi-file datasets
  - [ ] Progress bar for long operations

- [ ] Vorticity extraction
  - [ ] Spectral differentiation (FFT-based ∇×)
  - [ ] Handle periodic boundary conditions
  - [ ] Accuracy: 6th-order finite differences (backup)
  - [ ] Validate against JHTDB provided vorticity

- [ ] Level set extraction
  - [ ] Marching cubes algorithm (scikit-image)
  - [ ] Multiple thresholds: [0.1, 0.3, 0.5, 0.7, 0.9] × ω_max
  - [ ] Output: point clouds for Ripser
  - [ ] Optimization: octree spatial indexing

- [ ] Filtration construction
  - [ ] Sublevel set filtration: {x : ω(x) ≤ θ}
  - [ ] Distance matrix computation (GPU-accelerated)
  - [ ] Sparse representation (radius filtration)

**Files to Create**:
- `epipelagic/data/dns_processor.py` (main processor)
- `epipelagic/topology/level_sets.py` (level set extraction)
- `epipelagic/topology/filtration.py` (filtration builder)

**Acceptance Criteria**:
- Load 256³ DNS field in <10 seconds
- Extract vorticity with <1% error
- Generate filtration with 50 levels in <30 seconds

---

### Week 1-2 Deliverables
- [ ] JHTDB data downloaded (256³ isotropic turbulence)
- [ ] DNS processing pipeline operational
- [ ] Validation report: energy spectra, vorticity accuracy
- [ ] Documentation: data format, API usage

---

## Week 3-4: Advanced Persistent Homology

**Goal**: Enhance topology extraction with advanced features and validation

### Task 2.3: Multi-Scale Persistent Homology

**Priority**: P0 (Critical)
**Estimated Time**: 5 days
**Owner**: Code Agent

**Subtasks**:
- [ ] Gudhi integration (advanced features)
  - [ ] Install: `conda install -c conda-forge gudhi`
  - [ ] Alpha complex construction (geometric)
  - [ ] Cubical complex (for voxel data)
  - [ ] Compare Ripser vs Gudhi performance

- [ ] Multi-parameter persistence
  - [ ] 2D persistence: (vorticity threshold, scale)
  - [ ] Vineyard updates (temporal evolution)
  - [ ] Zigzag persistence (for time series)

- [ ] Barcode analysis
  - [ ] Persistence landscape computation
  - [ ] Bottleneck/Wasserstein distances
  - [ ] Statistical significance testing
  - [ ] Persistent entropy

- [ ] Feature extraction
  - [ ] Birth/death coordinates
  - [ ] Representative cycles (homology generators)
  - [ ] Persistence images (ML-ready features)
  - [ ] Betti numbers vs threshold

**Files to Create**:
- `epipelagic/topology/gudhi_interface.py`
- `epipelagic/topology/barcode_analysis.py`
- `epipelagic/topology/persistence_features.py`

**Acceptance Criteria**:
- Gudhi and Ripser produce consistent results
- Persistence landscapes computed
- Representative cycles extracted for visualization

---

### Task 2.4: Epipelagic Cohomology Refinement

**Priority**: P1 (Important)
**Estimated Time**: 4 days
**Owner**: Code Agent (with Sonnet consultation)

**Subtasks**:
- [ ] Implement cascade complex
  - [ ] C⁰ = ⊕ ℝE_n (shell energies)
  - [ ] C¹ = ⊕ ℝT_{nm} (transfers)
  - [ ] Boundary operator d⁰: C⁰ → C¹
  - [ ] Coboundary operator d¹: C¹ → C²

- [ ] Direct transfer measurement
  - [ ] Taichi kernel: compute T_{nm} from nonlinear terms
  ```python
  @ti.kernel
  def compute_transfer_matrix(self):
      # T_{nm} = -∫ u_n · P_m[(u·∇)u] dx
      for n, m in ti.ndrange(N_shells, N_shells):
          if m > n:  # Forward cascade only
              self.T[n, m] = compute_triadic_interaction(n, m)
  ```
  - [ ] Validate conservation: Σ_m T_{nm} = -dE_n/dt

- [ ] Cohomology computation
  - [ ] Exact sequences: 0 → H⁰ → H¹ → H² → 0
  - [ ] Kernel/image computations (SVD)
  - [ ] Rank computation: dim(H¹) = rank(ker d¹) - rank(im d⁰)
  - [ ] Compare to persistent homology dim(H¹_epi)

**Files to Modify**:
- `epipelagic/cascade/taichi_solver.py` (add transfer measurement)
- `epipelagic/core/complex.py` (implement cascade complex)

**Files to Create**:
- `epipelagic/core/transfer_matrix.py`

**Acceptance Criteria**:
- Transfer matrix T_{nm} measured directly from dynamics
- Energy conservation verified: |ΣT_{nm} + dE/dt| < 10⁻⁶
- dim(H¹_cascade) computed and compared to dim(H¹_persistent)

---

### Week 3-4 Deliverables
- [ ] Gudhi advanced features integrated
- [ ] Persistence landscapes and statistical measures
- [ ] Direct transfer measurement in cascade solver
- [ ] Cohomology comparison: cascade vs persistent

---

## Week 5-6: Realistic Vorticity Generation

**Goal**: Generate realistic synthetic turbulence for validation

### Task 2.5: Structured Vortex Models

**Priority**: P1 (Important)
**Estimated Time**: 5 days
**Owner**: Haiku Agent (prototypes), Code Agent (production)

**Subtasks**:
- [ ] Lamb-Oseen vortex implementation
  ```python
  # ω(r) = (Γ/πr_c²) exp(-r²/r_c²)
  # u_θ(r) = (Γ/2πr)(1 - exp(-r²/r_c²))
  ```
  - [ ] Single vortex: circulation Γ, core radius r_c
  - [ ] Parameters from turbulence: Γ ~ √(E_n), r_c ~ 1/k_n
  - [ ] 2D and 3D versions

- [ ] Vortex array configurations
  - [ ] Karman vortex street
  - [ ] Hexagonal vortex lattice
  - [ ] Random vortex positions (Poisson distribution)
  - [ ] Multi-scale vortex hierarchy (fractal-like)

- [ ] Vortex dynamics
  - [ ] Point vortex equations (fast approximation)
  - [ ] Vortex merging and reconnection
  - [ ] Time evolution: export sequences for vineyard

- [ ] Energy spectrum matching
  - [ ] Target: E(k) ~ k^(-5/3) (Kolmogorov)
  - [ ] Adjust vortex parameters to match DNS
  - [ ] Validate statistics: skewness, kurtosis

**Files to Create**:
- `epipelagic/synthetic/lamb_oseen.py`
- `epipelagic/synthetic/vortex_arrays.py`
- `epipelagic/synthetic/vortex_dynamics.py`

**Acceptance Criteria**:
- Lamb-Oseen vortex matches analytical solution
- Vortex array produces realistic energy spectrum
- dim(H¹_epi) from synthetic matches DNS trends

---

### Task 2.6: DNS-Informed Synthesis

**Priority**: P2 (Nice to have)
**Estimated Time**: 3 days
**Owner**: Haiku Agent

**Subtasks**:
- [ ] Extract vortex cores from DNS
  - [ ] Q-criterion: Q = ½(|Ω|² - |S|²) > threshold
  - [ ] λ₂-criterion: 2nd eigenvalue of S² + Ω²
  - [ ] Vortex core positions and circulation

- [ ] Fit vortex models to DNS
  - [ ] Extract r_c, Γ from each detected vortex
  - [ ] Statistical distributions of parameters
  - [ ] Reynolds number scaling

- [ ] Hybrid synthesis
  - [ ] DNS vortex positions + analytical velocity
  - [ ] Coarse DNS + synthetic small scales
  - [ ] Interpolation between DNS snapshots

**Files to Create**:
- `epipelagic/synthetic/vortex_detection.py`
- `epipelagic/synthetic/dns_informed_synthesis.py`

**Acceptance Criteria**:
- Vortex detection identifies cores in DNS
- Synthetic fields match DNS statistics

---

### Week 5-6 Deliverables
- [ ] Lamb-Oseen vortex models implemented
- [ ] Vortex arrays with realistic E(k) spectrum
- [ ] DNS vortex detection and fitting
- [ ] Validation: synthetic vs DNS topology

---

## Week 7-8: Finiteness Validation

**Goal**: Validate Theorem C empirically: dim(H¹ₑₚᵢ) ≤ C log(Re)

### Task 2.7: Large-Scale Reynolds Sweep

**Priority**: P0 (Critical)
**Estimated Time**: 5 days
**Owner**: Code Agent (requires GPU cluster)

**Subtasks**:
- [ ] Extended Reynolds range
  - [ ] Re ∈ [100, 10⁶] (100 points, log-spaced)
  - [ ] Multiple cascade configurations: 3, 5, 8, 10 shells
  - [ ] Multiple realizations (N=20) per Re for statistics

- [ ] GPU cluster deployment
  - [ ] Deploy to AWS/GCP GPU instance (A100 recommended)
  - [ ] Parallelize over Re values (MPI or Ray)
  - [ ] Estimated runtime: 10⁴ runs × 10³ steps/run / 10⁶ steps/sec = ~10 seconds
  - [ ] Total with topology: ~2 hours

- [ ] Systematic topology extraction
  - [ ] For each Re:
    1. Integrate cascade to steady state
    2. Generate vorticity field (3 methods: simple, Lamb-Oseen, DNS-informed)
    3. Compute persistent homology
    4. Extract dim(H¹_epi) with threshold optimization
  - [ ] Store all barcodes for later analysis

- [ ] Statistical analysis
  - [ ] Mean and std of dim(H¹_epi) vs Re
  - [ ] Fit: dim(H¹) = a log(Re) + b
  - [ ] Confidence intervals (bootstrap)
  - [ ] Outlier detection and analysis

**Scripts to Create**:
- `scripts/large_scale_reynolds_sweep.py` (main driver)
- `scripts/deploy_gpu_cluster.sh` (deployment)
- `scripts/analyze_finiteness_bound.py` (statistical analysis)

**Acceptance Criteria**:
- 100 Reynolds numbers tested with N=20 realizations each
- dim(H¹_epi) vs log(Re) fit with R² > 0.7
- Positive coefficient a > 0 (increasing with Re)
- Bounded growth verified: dim(H¹) ≤ C log(Re) with C ≈ 2-5

---

### Task 2.8: Regime Stratification Validation

**Priority**: P1 (Important)
**Estimated Time**: 3 days
**Owner**: Haiku Agent (exploration), Sonnet Agent (theory)

**Subtasks**:
- [ ] Recalibrate regime boundaries
  - [ ] Use measured T_{nm} (not estimated)
  - [ ] Compute ratios: ρ₁ = T₀₁/E₀, ρ₂ = T₁₂/T₀₁
  - [ ] Scan (ρ₁, ρ₂) space

- [ ] Classify regimes by topology
  - [ ] Laminar: dim(H¹) = 0 (trivial cohomology)
  - [ ] Epipelagic: 0 < dim(H¹) ≤ C log(Re)
  - [ ] Mesopelagic: dim(H¹) > C log(Re)
  - [ ] Bathypelagic: dim(H¹) → ∞ (full turbulence)

- [ ] Phase diagram refinement
  - [ ] 2D: (Re, ν) space
  - [ ] 3D: (Re, ν, forcing) space
  - [ ] Overlay topology measurements
  - [ ] Identify phase boundaries

**Files to Create**:
- `epipelagic/cascade/regime_classifier.py`
- `scripts/validate_stratification.py`

**Acceptance Criteria**:
- Regime boundaries identified from topology
- Phase diagram matches theoretical predictions
- Epipelagic regime occupies significant parameter space

---

### Week 7-8 Deliverables
- [ ] Large-scale Reynolds sweep (10⁴ simulations)
- [ ] Finiteness bound validated: dim(H¹) ≤ C log(Re)
- [ ] Regime stratification confirmed
- [ ] Phase diagrams with topology overlay

---

## Week 9-10: Houdini Visualization

**Goal**: Build interactive 3D visualizations for persistence and vorticity

### Task 2.9: Houdini Python Integration

**Priority**: P1 (Important)
**Estimated Time**: 4 days
**Owner**: Code Agent

**Subtasks**:
- [ ] Set up Houdini environment
  - [ ] Install Houdini Apprentice (free) or Indie
  - [ ] Configure Python 3.11 in Houdini
  - [ ] Install packages: numpy, scipy in Houdini's Python
  - [ ] Test HOM (Houdini Object Model) API

- [ ] Vorticity field visualization
  - [ ] Import vorticity as volume VDB
  - [ ] Isosurface rendering (marching cubes)
  - [ ] Volume rendering with transfer functions
  - [ ] Color mapping: ω → hue (blue=low, red=high)

- [ ] Persistence barcode 3D view
  - [ ] Birth/death times → 3D tubes
  - [ ] Color by persistence length
  - [ ] Interactive filtering: threshold slider
  - [ ] Link to spatial location (birth coordinates)

- [ ] Representative cycles visualization
  - [ ] Extract cycle generators from Ripser
  - [ ] Convert to Houdini curves/surfaces
  - [ ] Animate growth through filtration
  - [ ] Highlight long-lived features

**Files to Create**:
- `epipelagic/visualization/houdini_api.py`
- `houdini/hdas/PersistenceBarcodes.hda` (Digital Asset)
- `houdini/hdas/VorticityVolume.hda`

**Acceptance Criteria**:
- Load DNS vorticity in Houdini
- Display persistence barcodes in 3D
- Interactive exploration of persistent features

---

### Task 2.10: Interactive Explorable Visualizations

**Priority**: P2 (Nice to have)
**Estimated Time**: 4 days
**Owner**: Code Agent

**Subtasks**:
- [ ] Web-based visualizations (Plotly Dash)
  - [ ] Persistence diagram: (birth, death) scatter
  - [ ] Barcode plot with filtering
  - [ ] Energy spectrum E(k) vs k
  - [ ] dim(H¹) vs Re with fit overlay

- [ ] Linked views
  - [ ] Click barcode → highlight in 3D
  - [ ] Hover 3D vortex → show in persistence space
  - [ ] Slider: threshold → update all views

- [ ] Time evolution animations
  - [ ] Vineyard diagrams (barcode evolution)
  - [ ] Vortex merging/splitting events
  - [ ] Energy cascade visualization

- [ ] Export and sharing
  - [ ] High-res images (4K)
  - [ ] Videos (MP4, 60fps)
  - [ ] Interactive HTML (for papers)
  - [ ] Jupyter notebook examples

**Files to Create**:
- `epipelagic/visualization/plotly_dashboard.py`
- `epipelagic/visualization/animations.py`
- `examples/interactive_topology.ipynb`

**Acceptance Criteria**:
- Web dashboard deployed and functional
- Linked 2D/3D views working
- Animations exported for presentation

---

### Week 9-10 Deliverables
- [ ] Houdini HDA assets for topology visualization
- [ ] Interactive web dashboard (Plotly Dash)
- [ ] Time evolution animations
- [ ] Example gallery with 10+ visualizations

---

## Week 11-12: Large-Scale Processing

**Goal**: Handle DNS data at 10⁹+ points, optimize for production

### Task 2.11: Scalability Optimization

**Priority**: P0 (Critical)
**Estimated Time**: 5 days
**Owner**: Code Agent

**Subtasks**:
- [ ] Memory optimization
  - [ ] Streaming algorithms (don't load full field)
  - [ ] Chunked processing (divide into blocks)
  - [ ] Sparse data structures (octree, k-d tree)
  - [ ] GPU memory management (pinned memory)

- [ ] Parallel processing
  - [ ] Multi-GPU support (Taichi CUDA multi-device)
  - [ ] MPI for distributed computing
  - [ ] Ray for cluster orchestration
  - [ ] Dask for large arrays

- [ ] Performance profiling
  - [ ] Identify bottlenecks (cProfile, line_profiler)
  - [ ] Optimize hot paths (Numba JIT)
  - [ ] Memory profiling (memory_profiler)
  - [ ] GPU profiling (NVIDIA Nsight)

- [ ] Algorithmic improvements
  - [ ] Approximation algorithms (trade accuracy for speed)
  - [ ] Coarse-graining for large-scale
  - [ ] Hierarchical processing (multi-resolution)
  - [ ] Incremental computation (avoid recomputing)

**Target Performance**:
- [ ] Process 1024³ DNS field in <10 minutes
- [ ] Compute persistence on 10⁶ points in <1 minute
- [ ] Memory usage < 16GB for 512³ field

**Acceptance Criteria**:
- Successfully process JHTDB 1024³ full dataset
- Performance targets met
- Profiling report with optimizations documented

---

### Task 2.12: Production Pipeline and Testing

**Priority**: P0 (Critical)
**Estimated Time**: 4 days
**Owner**: Code Agent

**Subtasks**:
- [ ] End-to-end pipeline
  - [ ] Input: DNS data path or cascade parameters
  - [ ] Processing: automatic detection of data type
  - [ ] Output: HDF5 file with all results
    - Persistence diagrams
    - dim(H¹_epi)
    - Representative cycles
    - Metadata (Re, ν, resolution, etc.)

- [ ] Command-line interface
  ```bash
  epipelagic-topology --input data/dns/iso1024.h5 \
                      --output results/topology.h5 \
                      --resolution 256 \
                      --threshold-range 0.1,0.9 \
                      --n-thresholds 50
  ```

- [ ] Configuration management
  - [ ] YAML config files
  - [ ] Parameter validation
  - [ ] Sensible defaults
  - [ ] Config templates for common use cases

- [ ] Comprehensive testing
  - [ ] Unit tests: >90% coverage
  - [ ] Integration tests: full pipeline
  - [ ] Regression tests: known results
  - [ ] Performance tests: benchmark suite

**Files to Create**:
- `epipelagic/cli.py` (command-line interface)
- `epipelagic/pipeline.py` (end-to-end orchestration)
- `configs/topology_extraction.yaml` (config template)
- `tests/integration/test_pipeline.py`

**Acceptance Criteria**:
- CLI runs full pipeline with one command
- Config files manage all parameters
- Test coverage >90%
- CI/CD pipeline runs all tests

---

### Week 11-12 Deliverables
- [ ] Scalable pipeline handling 10⁹+ points
- [ ] CLI tool for end-to-end processing
- [ ] Comprehensive test suite (>90% coverage)
- [ ] Performance benchmarks documented

---

## Success Criteria

### Technical Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| DNS data integrated | ✅ | JHTDB 256³ downloaded and validated |
| dim(H¹_epi) finiteness | dim ≤ C log(Re), C ≈ 2-5 | Reynolds sweep fit R² > 0.7 |
| Topology accuracy | <5% error vs ground truth | Cross-validation: Ripser vs Gudhi |
| Performance | 1024³ DNS in <10 min | Benchmark on GPU cluster |
| Visualization quality | Interactive 3D + web | User testing (clarity, usability) |
| Code coverage | >90% | pytest-cov report |

### Research Outcomes

- [ ] **Theorem C validated empirically**: dim(H¹_epi) ≤ C log(Re) holds across Re ∈ [100, 10⁶]
- [ ] **H¹_epi extracted from DNS**: Real turbulence data successfully analyzed
- [ ] **Regime stratification confirmed**: Epipelagic zone identified in parameter space
- [ ] **Publication-ready figures**: 10+ high-quality visualizations
- [ ] **Reproducible pipeline**: Full workflow documented and tested

### Deliverables for Publication

- [ ] Section 5 draft: "Persistent Homology and Epipelagic Cohomology"
- [ ] Supplementary material: Computation methods
- [ ] Interactive demos: Web-based explorables
- [ ] Code release: GitHub repo with documentation
- [ ] Dataset: Processed DNS results with topology

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU cluster unavailable | Medium | High | Optimize CPU code, use cloud GPUs (AWS/GCP) |
| DNS data too large | High | Medium | Implement streaming, chunking, coarse-graining |
| Finiteness bound not validated | Medium | High | Try multiple synthetic models, adjust thresholds |
| Houdini complexity | Medium | Medium | Use Plotly as fallback, simplify visualizations |
| Memory overflow | High | High | Profile carefully, implement out-of-core algorithms |

### Research Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Topology doesn't match theory | Low | High | Consult Sonnet for theoretical refinement |
| dim(H¹) shows wrong scaling | Medium | High | Reconsider persistence thresholds, vortex models |
| Regime boundaries unclear | Medium | Medium | Use data-driven clustering (ML) |
| Computational cost prohibitive | Low | Medium | Focus on smaller Re ranges first |

---

## Implementation Order (Critical Path)

### Parallel Work Streams

**Week 1-2: Foundation**
- DNS data integration (P0)
- Data processing pipeline (P0)

**Week 3-4: Core Topology**
- Gudhi integration (P0)
- Transfer matrix measurement (P0)
- Structured vortex models (P1)

**Week 5-6: Synthesis & Validation**
- Vortex arrays (P1)
- DNS-informed synthesis (P2)

**Week 7-8: Large-Scale Testing** (requires GPU)
- Reynolds sweep (P0) ← **CRITICAL PATH**
- Regime validation (P1)

**Week 9-10: Visualization**
- Houdini integration (P1)
- Web dashboard (P2)

**Week 11-12: Production**
- Scalability optimization (P0)
- Testing and CI/CD (P0)

### Dependencies

- Weeks 7-8 require GPU cluster (arrange by Week 6)
- Weeks 9-10 can proceed in parallel with 7-8
- Week 11-12 optimization depends on Week 7-8 profiling data

---

## Agent Collaboration Protocol

### Claude Code (This Agent) - Implementation Specialist

**Focus**: Production code, optimization, infrastructure

**Tasks**:
- DNS data integration and processing
- Gudhi/Ripser pipeline optimization
- Large-scale GPU deployment
- Houdini visualization assets
- Testing and CI/CD

**Escalate to Sonnet when**:
- Theoretical guidance needed (thresholds, regime boundaries)
- Unexpected results require mathematical explanation
- Publication writing (Section 5)

**Escalate to Haiku when**:
- Quick prototypes needed (new vortex models)
- Parameter exploration (what thresholds to use?)
- Rapid iteration on visualizations

---

### Claude Haiku - Rapid Prototyper

**Focus**: Quick experiments, parameter sweeps, prototypes

**Tasks**:
- Vortex model prototyping (Lamb-Oseen variants)
- Threshold optimization (what values work best?)
- Visualization prototypes (Plotly sketches)
- Sanity checks (does dim(H¹) make sense?)

**Handoff to Code when**:
- Prototype validated, ready for production
- Performance critical (need GPU optimization)
- Integration with main pipeline required

---

### Claude Sonnet - Primary Researcher

**Focus**: Mathematical rigor, theorem proving, paper writing

**Tasks**:
- Prove Theorem C formally
- Establish H¹_cascade ↔ H¹_persistent correspondence
- Interpret computational results mathematically
- Write Section 5 (Persistent Homology)

**Requests to Code/Haiku**:
- "Validate this bound numerically"
- "Test this threshold criterion"
- "Generate figures for paper"

---

## Getting Started

### Step 1: Environment Setup

```bash
# Install Phase 2 dependencies
pip install pyJHTDB gudhi plotly dash

# Houdini (optional, for visualization)
# Download from sidefx.com and install manually

# Create data directories
mkdir -p data/dns/jhtdb_iso1024
mkdir -p results/phase2_topology
mkdir -p figures/phase2

# Verify GPU access
python -c "import taichi as ti; ti.init(arch=ti.gpu); print('GPU:', ti.cfg.arch)"
```

### Step 2: DNS Data Download

```bash
# Download test dataset
python scripts/download_dns_data.py \
    --dataset isotropic1024coarse \
    --time 0.364 \
    --resolution 256 \
    --output data/dns/jhtdb_iso1024/

# Validate
python scripts/validate_dns_data.py data/dns/jhtdb_iso1024/
```

### Step 3: Run Baseline Topology Extraction

```bash
# Extract topology from DNS
python scripts/topology_extraction.py \
    --input data/dns/jhtdb_iso1024/velocity.h5 \
    --output results/phase2_topology/baseline.h5 \
    --method dns

# Analyze results
python scripts/analyze_finiteness_bound.py results/phase2_topology/baseline.h5
```

### Step 4: Track Progress

Update task status in this file daily:
- Mark completed tasks with `[x]`
- Document issues in task notes
- Update acceptance criteria as needed

---

## Communication

**Daily Updates**: Update this TASKS.md with progress
**Blockers**: Tag @Sonnet for theory, @Haiku for quick tests
**Code Review**: All production code requires review
**Documentation**: Document as you code, not after

---

**Questions?** Review CLAUDE.md (agent roles) and ARCHITECTURE.md (system design)

**Ready to start?** Begin with Week 1, Task 2.1: DNS Data Acquisition

---

**STATUS**: Phase 2 Ready to Launch
**VERSION**: 1.0
**LAST UPDATED**: 2025-12-05
