"""
Core mathematical structures for epipelagic turbulence framework.

This module provides fundamental structures:
- CascadeComplex: Cochain complex (C•, d•) for cascade dynamics
- SpectralSequence: Spectral sequence {Eᵣᵖ'ᑫ, dᵣ} from filtrations
- TransferMatrix: Energy transfer matrix measurement (Task 2.4)
- Cohomology computation and analysis tools
"""

from epipelagic.core.complex import CascadeComplex
from epipelagic.core.spectral import SpectralSequence
from epipelagic.core.cohomology import compute_cohomology, epipelagic_dimension
from epipelagic.core.transfer_matrix import (
    TransferMatrix,
    compute_transfer_matrix_spectral,
    compute_transfer_matrix_shell_model,
    validate_transfer_conservation,
)

__all__ = [
    "CascadeComplex",
    "SpectralSequence",
    "compute_cohomology",
    "epipelagic_dimension",
    # Task 2.4: Transfer matrix measurement
    "TransferMatrix",
    "compute_transfer_matrix_spectral",
    "compute_transfer_matrix_shell_model",
    "validate_transfer_conservation",
]
