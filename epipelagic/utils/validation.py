"""
Validation utilities for cascade simulations.
"""

import numpy as np
from typing import Tuple


def validate_energy_conservation(
    energies_initial: np.ndarray,
    energies_final: np.ndarray,
    forcing: float = 0.0,
    dissipation: float = 0.0,
    dt: float = 1.0,
    tolerance: float = 1e-6,
) -> Tuple[bool, float]:
    """
    Validate energy conservation in cascade simulation.

    Parameters
    ----------
    energies_initial : ndarray
        Initial shell energies
    energies_final : ndarray
        Final shell energies
    forcing : float
        Energy injection rate
    dissipation : float
        Energy dissipation rate
    dt : float
        Time interval
    tolerance : float
        Relative error tolerance

    Returns
    -------
    valid : bool
        True if energy conservation holds within tolerance
    relative_error : float
        Relative energy conservation error
    """
    E_initial = np.sum(energies_initial)
    E_final = np.sum(energies_final)

    # Expected energy change
    dE_expected = (forcing - dissipation) * dt

    # Actual energy change
    dE_actual = E_final - E_initial

    # Relative error
    if abs(dE_expected) > 1e-10:
        relative_error = abs(dE_actual - dE_expected) / abs(dE_expected)
    else:
        relative_error = abs(dE_actual - dE_expected)

    valid = relative_error < tolerance

    return valid, relative_error
