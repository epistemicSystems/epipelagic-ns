"""
Data loading and processing for DNS turbulence data.

This module provides interfaces for loading Direct Numerical Simulation (DNS)
data from various sources, including:
- Johns Hopkins Turbulence Database (JHTDB)
- Local HDF5 files
- Synthetic turbulence generators

For Phase 2, we support:
1. Custom JHTDB data loader (works with downloaded HDF5 files)
2. Data validation utilities
3. Preprocessing pipelines
"""

from .jhtdb_loader import JHTDBLoader, download_jhtdb_data, create_sample_jhtdb_file
from .dns_processor import DNSProcessor, validate_dns_data
from .synthetic import generate_synthetic_turbulence

__all__ = [
    'JHTDBLoader',
    'download_jhtdb_data',
    'create_sample_jhtdb_file',
    'DNSProcessor',
    'validate_dns_data',
    'generate_synthetic_turbulence',
]
