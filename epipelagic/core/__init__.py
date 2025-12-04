"""
Core mathematical structures for epipelagic turbulence framework.

This module provides fundamental structures:
- CascadeComplex: Cochain complex (C•, d•) for cascade dynamics
- SpectralSequence: Spectral sequence {Eᵣᵖ'ᑫ, dᵣ} from filtrations
- Cohomology computation and analysis tools
"""

from epipelagic.core.complex import CascadeComplex
from epipelagic.core.spectral import SpectralSequence
from epipelagic.core.cohomology import compute_cohomology, epipelagic_dimension

__all__ = [
    "CascadeComplex",
    "SpectralSequence",
    "compute_cohomology",
    "epipelagic_dimension",
]
