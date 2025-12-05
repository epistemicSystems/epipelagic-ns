"""
Topological analysis tools for turbulent cascade dynamics.

This module provides:
- Persistent homology extraction from vorticity fields (Ripser)
- Advanced persistent homology (Gudhi)
- Barcode analysis (landscapes, distances, entropy)
- Feature extraction for machine learning
- Epipelagic cohomology dimension computation

Phase 2 Topology Implementation (Tasks 2.3-2.4):
- Gudhi integration with Alpha/Cubical/Rips complexes
- Persistence landscapes and statistical analysis
- Bottleneck and Wasserstein distances
- Persistence images and ML-ready features
"""

# Core persistent homology (Ripser-based)
from epipelagic.topology.persistent import (
    extract_persistent_homology,
    compute_epipelagic_dimension,
    filter_long_bars,
    compute_vorticity,
)

# Advanced persistent homology (Gudhi-based) - Task 2.3
from epipelagic.topology.gudhi_interface import (
    compute_alpha_complex_persistence,
    compute_cubical_complex_persistence,
    compute_rips_complex_persistence,
    compare_ripser_gudhi,
    persistence_to_diagrams,
)

# Barcode analysis - Task 2.3
from epipelagic.topology.barcode_analysis import (
    compute_persistence_landscape,
    landscape_distance,
    bottleneck_distance,
    wasserstein_distance,
    persistent_entropy,
    betti_curve,
    statistical_significance_test,
)

# Feature extraction - Task 2.3
from epipelagic.topology.persistence_features import (
    extract_birth_death_coordinates,
    compute_persistence_statistics,
    compute_persistence_image,
    compute_betti_numbers_vs_threshold,
    extract_epipelagic_features,
    create_feature_vector,
)

__all__ = [
    # Core Ripser functionality
    "extract_persistent_homology",
    "compute_epipelagic_dimension",
    "filter_long_bars",
    "compute_vorticity",
    # Gudhi advanced features
    "compute_alpha_complex_persistence",
    "compute_cubical_complex_persistence",
    "compute_rips_complex_persistence",
    "compare_ripser_gudhi",
    "persistence_to_diagrams",
    # Barcode analysis
    "compute_persistence_landscape",
    "landscape_distance",
    "bottleneck_distance",
    "wasserstein_distance",
    "persistent_entropy",
    "betti_curve",
    "statistical_significance_test",
    # Feature extraction
    "extract_birth_death_coordinates",
    "compute_persistence_statistics",
    "compute_persistence_image",
    "compute_betti_numbers_vs_threshold",
    "extract_epipelagic_features",
    "create_feature_vector",
]
