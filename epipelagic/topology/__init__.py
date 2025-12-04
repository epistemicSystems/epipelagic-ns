"""
Topological analysis tools for turbulent cascade dynamics.

This module provides:
- Persistent homology extraction from vorticity fields
- Epipelagic cohomology dimension computation
- Barcode visualization and analysis
"""

from epipelagic.topology.persistent import (
    extract_persistent_homology,
    compute_epipelagic_dimension,
    filter_long_bars,
)

__all__ = [
    "extract_persistent_homology",
    "compute_epipelagic_dimension",
    "filter_long_bars",
]
