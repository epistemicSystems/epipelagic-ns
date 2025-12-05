# DNS Turbulence Data

This directory contains Direct Numerical Simulation (DNS) turbulence data for Phase 2 topology analysis.

## Directory Structure

```
dns/
├── README.md              # This file
├── jhtdb_iso1024/         # JHTDB isotropic turbulence data
│   └── velocity_256.h5    # 256³ velocity field
├── raw/                   # Raw downloaded data
└── processed/             # Processed data (vorticity, level sets, etc.)
```

## Data Sources

### 1. Johns Hopkins Turbulence Database (JHTDB)

**Primary source for Phase 2**

- **Website**: http://turbulence.pha.jhu.edu/
- **Dataset**: `isotropic1024coarse` (forced isotropic turbulence)
- **Resolution**: 1024³ (we use 256³ subcube)
- **Reynolds number**: Re_λ ≈ 418
- **Time snapshot**: t = 0.364 (peak dissipation)

**How to obtain data:**

```python
# Option 1: Create synthetic test data
from epipelagic.data import create_sample_jhtdb_file

create_sample_jhtdb_file(
    'data/dns/jhtdb_iso1024/velocity_256.h5',
    resolution=(256, 256, 256),
    reynolds_number=1000
)
```

```python
# Option 2: Download instructions
from epipelagic.data import download_jhtdb_data

instructions = download_jhtdb_data(
    'data/dns/jhtdb_iso1024/velocity_256.h5',
    method='manual'
)
```

### 2. Synthetic Turbulence

**For testing and validation**

```python
from epipelagic.data import generate_synthetic_turbulence
import h5py

# Generate synthetic field
velocity = generate_synthetic_turbulence(
    resolution=(256, 256, 256),
    reynolds_number=1000,
    energy_spectrum='kolmogorov'
)

# Save to HDF5
with h5py.File('data/dns/synthetic_Re1000.h5', 'w') as f:
    f.create_dataset('velocity', data=velocity, compression='gzip')
    f.attrs['reynolds_number'] = 1000
    f.attrs['type'] = 'synthetic'
```

## Data Format

All DNS data is stored in HDF5 format with the following structure:

```
filename.h5
├── /velocity              # Dataset: (nx, ny, nz, 3) float64
│   └── Velocity field u(x,y,z) = (ux, uy, uz)
├── /vorticity (optional)  # Dataset: (nx, ny, nz) float64
│   └── Vorticity magnitude |ω|
└── attributes:
    ├── reynolds_number    # Reynolds number Re
    ├── Re_lambda          # Taylor-microscale Reynolds number
    ├── resolution         # Grid resolution (nx, ny, nz)
    └── dataset_type       # 'dns', 'synthetic', etc.
```

## Usage Examples

### Load and Process DNS Data

```python
from epipelagic.data import DNSProcessor

# Initialize processor
processor = DNSProcessor('data/dns/jhtdb_iso1024/velocity_256.h5')

# Load velocity
velocity = processor.load_velocity()

# Compute vorticity (spectral method)
vorticity = processor.compute_vorticity(method='spectral')

# Extract level sets for persistent homology
level_sets = processor.extract_level_sets(vorticity, n_levels=50)

# Build filtration
filtration = processor.build_filtration(level_sets)

# Save processed data
processor.save_processed_data(
    'data/dns/processed/iso1024_processed.h5',
    vorticity=vorticity,
    filtration=filtration
)
```

### Validate DNS Data

```python
from epipelagic.data import validate_dns_data

# Run validation tests
results = validate_dns_data('data/dns/jhtdb_iso1024/velocity_256.h5')

# Check results
print(f"Incompressibility: max |∇·u| = {results['incompressibility']:.3e}")
print(f"Energy spectrum slope: {results['energy_spectrum_slope']:.3f}")
```

### Extract Topology

```python
from epipelagic.topology import extract_persistent_homology

# Load velocity
processor = DNSProcessor('data/dns/jhtdb_iso1024/velocity_256.h5')
velocity = processor.load_velocity()

# Extract epipelagic cohomology
result = extract_persistent_homology(
    velocity,
    threshold=0.5,
    max_dimension=2
)

print(f"dim(H¹_epi) = {result['dim_H1_epi']}")
print(f"Number of long bars: {len(result['long_bars'])}")
```

## Data Requirements

### Phase 2 Week 1-2 (Task 2.1-2.2)
- ✅ 256³ velocity field (manageable size)
- ✅ Re_λ ~ 400-1000 range
- ✅ Single time snapshot sufficient

### Phase 2 Week 7-8 (Task 2.7)
- 🔄 Large-scale Reynolds sweep
- 🔄 Multiple realizations (N=20) per Re
- 🔄 Re ∈ [100, 10⁶], 100 points

### Phase 2 Week 11-12 (Task 2.11)
- ⏳ Full 1024³ JHTDB dataset
- ⏳ Streaming/chunked processing
- ⏳ Multi-GPU support

## Disk Space Requirements

| Data Type | Resolution | Size | Status |
|-----------|-----------|------|--------|
| Synthetic 64³ | 64³ | ~10 MB | ✅ Available |
| Synthetic 256³ | 256³ | ~500 MB | ✅ Available |
| JHTDB 256³ | 256³ | ~500 MB | 🔄 Download |
| JHTDB 1024³ | 1024³ | ~30 GB | ⏳ Future |

**Current allocation**: ~50 GB for DNS data

## Citation

If you use JHTDB data, please cite:

```bibtex
@article{Li2008,
  title={A Public Turbulence Database Cluster and Applications to Study Lagrangian Evolution of Velocity Increments in Turbulence},
  author={Li, Yi and Perlman, Eric and Wan, Minping and others},
  journal={Journal of Turbulence},
  volume={9},
  pages={N31},
  year={2008}
}
```

## Troubleshooting

### pyJHTDB Installation Issues
The official `pyJHTDB` package has compatibility issues with numpy 2.x. We provide custom loaders instead.

### Large File Handling
For files > 10 GB:
- Use lazy loading: `DNSProcessor(path, lazy_load=True)`
- Process in chunks
- Use memory-mapped arrays

### Missing Data
If JHTDB download fails:
1. Use synthetic data for testing
2. Contact project maintainers for pre-processed datasets
3. Try manual download from JHTDB web portal

## Status

**Phase 2 Week 1-2**: ✅ Infrastructure complete
- [x] DNS data module created
- [x] Synthetic turbulence generator working
- [x] JHTDB loader implemented
- [x] Data validation pipeline ready
- [ ] Download 256³ JHTDB data (in progress)

**Next Steps**:
1. Generate synthetic test dataset
2. Validate full pipeline
3. Extract topology from test data
4. Proceed to Task 2.3 (Advanced Persistent Homology)

---

**Last Updated**: 2025-12-05
**Version**: 1.0
**Phase**: 2 (Topology Implementation)
