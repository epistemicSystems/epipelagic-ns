"""
Epipelagic Turbulence Research Framework
=========================================

A computational framework for studying turbulent cascade dynamics through
the lens of geometric Langlands correspondence and persistent homology.

Main modules:
- core: Fundamental mathematical structures (complexes, cohomology)
- cascade: Shell cascade solvers and energy transfer models
- topology: Persistent homology and topological invariants
- quantum: Quasi-particle formalism and Fock space dynamics
- langlands: Geometric Langlands machinery and functorial correspondence
- visualization: Interactive visualization tools
- utils: Utilities and helper functions
"""

__version__ = "0.1.0"
__author__ = "Epipelagic Research Team"
__license__ = "MIT"

# Core imports
from epipelagic.core import (
    CascadeComplex,
    SpectralSequence,
    compute_cohomology,
)

from epipelagic.cascade import (
    ShellCascade,
    CascadeSolver,
    compute_phase_diagram,
)

from epipelagic.topology import (
    extract_persistent_homology,
    compute_epipelagic_dimension,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core structures
    "CascadeComplex",
    "SpectralSequence",
    "compute_cohomology",
    # Cascade solvers
    "ShellCascade",
    "CascadeSolver",
    "compute_phase_diagram",
    # Topology
    "extract_persistent_homology",
    "compute_epipelagic_dimension",
]
