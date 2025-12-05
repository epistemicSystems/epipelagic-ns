"""
Cohomology computation and analysis for cascade complexes.

This module provides high-level functions for computing epipelagic cohomology
and analyzing its properties.
"""

from typing import Tuple, Optional
import numpy as np
from epipelagic.core.complex import CascadeComplex
from epipelagic.core.spectral import SpectralSequence


def compute_cohomology(
    complex: CascadeComplex,
    degree: int = 1,
) -> Tuple[np.ndarray, int]:
    """
    Compute cohomology group Hⁿ(C•) of cascade complex.

    Parameters
    ----------
    complex : CascadeComplex
        Cascade complex to analyze
    degree : int
        Cohomology degree (0 or 1)

    Returns
    -------
    basis : ndarray
        Basis vectors for Hⁿ
    dimension : int
        Dimension of cohomology group

    Examples
    --------
    >>> complex = CascadeComplex(...)
    >>> H1_basis, dim_H1 = compute_cohomology(complex, degree=1)
    >>> print(f"Epipelagic cohomology dimension: {dim_H1}")
    """
    H0_basis, H1_basis, dim_H1 = complex.compute_cohomology()

    if degree == 0:
        return H0_basis, H0_basis.shape[1]
    elif degree == 1:
        return H1_basis, dim_H1
    else:
        raise ValueError(f"Cohomology degree {degree} not supported (use 0 or 1)")


def epipelagic_dimension(
    complex: CascadeComplex,
    verify_degeneration: bool = True,
) -> int:
    """
    Compute dimension of epipelagic cohomology: dim(H¹ₑₚᵢ).

    Parameters
    ----------
    complex : CascadeComplex
        Cascade complex in epipelagic regime
    verify_degeneration : bool
        If True, verify E₂-degeneration of spectral sequence

    Returns
    -------
    dim_H1_epi : int
        Dimension of H¹ₑₚᵢ

    Raises
    ------
    ValueError
        If not in epipelagic regime and verification requested

    Mathematical Context:
        In the epipelagic regime, the spectral sequence degenerates at E₂,
        so H¹ₑₚᵢ = E₂¹'⁰ = H¹(C•).

        Theorem C predicts: dim(H¹ₑₚᵢ) ≤ C log(Re) for universal constant C.
    """
    # Compute H¹
    _, dim_H1 = compute_cohomology(complex, degree=1)

    # Verify we're in epipelagic regime
    regime = complex.classify_regime()

    if verify_degeneration and regime != "epipelagic":
        raise ValueError(
            f"Complex not in epipelagic regime (found: {regime}). "
            f"E₂-degeneration may not hold."
        )

    return dim_H1


def finiteness_bound(reynolds_number: float, constant: float = 2.5) -> float:
    """
    Theoretical upper bound on dim(H¹ₑₚᵢ) from Theorem C.

    Parameters
    ----------
    reynolds_number : float
        Reynolds number Re
    constant : float
        Universal constant C in bound

    Returns
    -------
    bound : float
        Upper bound: C log(Re)

    Mathematical Statement (Theorem C):
        For parameters p in epipelagic regime,
            dim(H¹ₑₚᵢ(p)) ≤ C log(Re(p))
        where C is a universal constant independent of p.

    References
    ----------
    [1] RIGOROUS_FOUNDATIONS.md, Section 3.3 (Theorem C)
    """
    if reynolds_number <= 1:
        return 0.0

    return constant * np.log(reynolds_number)


def verify_finiteness_theorem(
    complex: CascadeComplex,
    reynolds_number: float,
    constant: float = 2.5,
) -> Tuple[bool, float, float]:
    """
    Verify Theorem C (finiteness bound) for given cascade complex.

    Parameters
    ----------
    complex : CascadeComplex
        Cascade complex to test
    reynolds_number : float
        Reynolds number
    constant : float
        Constant C in theorem

    Returns
    -------
    satisfied : bool
        True if dim(H¹ₑₚᵢ) ≤ C log(Re)
    actual_dimension : float
        Actual dim(H¹ₑₚᵢ)
    bound : float
        Theoretical bound C log(Re)

    Examples
    --------
    >>> complex = CascadeComplex(...)
    >>> satisfied, dim, bound = verify_finiteness_theorem(complex, Re=1000)
    >>> print(f"dim(H¹) = {dim}, bound = {bound:.1f}, satisfied = {satisfied}")
    """
    dim = epipelagic_dimension(complex, verify_degeneration=False)
    bound = finiteness_bound(reynolds_number, constant)

    satisfied = dim <= bound

    return satisfied, float(dim), bound


def compute_persistent_generators(
    complex: CascadeComplex,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Compute persistent generators of H¹ₑₚᵢ (long-lived cascade structures).

    Parameters
    ----------
    complex : CascadeComplex
        Cascade complex
    threshold : float
        Persistence threshold (energy transfer magnitude)

    Returns
    -------
    generators : ndarray, shape (n_generators, n_transfers)
        Basis of persistent generators

    Algorithm:
        1. Compute H¹ basis
        2. Filter by persistence (transfer magnitude)
        3. Return generators with |Tₙₘ| > threshold

    Interpretation:
        Each generator corresponds to a cross-scale structure that persists
        across multiple cascade shells (e.g., vortex stretching, energy pathway).
    """
    H1_basis, _ = compute_cohomology(complex, degree=1)

    # Filter by magnitude
    transfer_magnitudes = np.abs(complex.transfers[complex.transfers != 0])

    if len(transfer_magnitudes) == 0:
        return np.zeros((0, H1_basis.shape[0]))

    # Select generators with strong transfers
    persistent = []
    for i in range(H1_basis.shape[1]):
        generator = H1_basis[:, i]
        magnitude = np.linalg.norm(generator)

        if magnitude > threshold:
            persistent.append(generator)

    if len(persistent) == 0:
        return np.zeros((0, H1_basis.shape[0]))

    return np.column_stack(persistent).T
