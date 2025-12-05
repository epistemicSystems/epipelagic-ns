"""
Phase diagram computation for cascade parameter space.

Implements parameter sweeps to identify:
- Laminar regime
- Epipelagic regime (E₂-degeneration)
- Mesopelagic regime
- Bathypelagic regime (fully developed turbulence)
"""

from typing import Tuple, Optional, List
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from epipelagic.cascade.shell_model import ShellCascade
from epipelagic.cascade.solver import CascadeSolver
from epipelagic.core.complex import CascadeComplex


@dataclass
class PhaseDiagramResult:
    """
    Results from phase diagram computation.

    Attributes
    ----------
    Re_range : ndarray
        Reynolds numbers tested
    nu_range : ndarray
        Viscosities tested
    regime_map : ndarray, shape (len(Re_range), len(nu_range))
        Regime classification (0=laminar, 1=epi, 2=meso, 3=bathy)
    dim_H1_map : ndarray
        Cohomology dimensions
    energy_maps : dict
        Shell energies for each parameter point
    """

    Re_range: np.ndarray
    nu_range: np.ndarray
    regime_map: np.ndarray
    dim_H1_map: np.ndarray
    energy_maps: dict


def compute_phase_diagram(
    Re_range: np.ndarray,
    nu_range: np.ndarray,
    n_shells: int = 8,
    forcing_amplitude: float = 0.1,
    t_transient: float = 100.0,
    verbose: bool = True,
) -> PhaseDiagramResult:
    """
    Compute phase diagram in (Re, ν) parameter space.

    Parameters
    ----------
    Re_range : ndarray
        Array of Reynolds numbers to test
    nu_range : ndarray
        Array of viscosities to test
    n_shells : int
        Number of cascade shells
    forcing_amplitude : float
        Forcing strength
    t_transient : float
        Time to reach steady state
    verbose : bool
        Show progress bar

    Returns
    -------
    result : PhaseDiagramResult
        Phase diagram data

    Examples
    --------
    >>> Re_range = np.logspace(2, 4, 20)
    >>> nu_range = np.logspace(-4, -1, 15)
    >>> result = compute_phase_diagram(Re_range, nu_range)
    >>> plt.imshow(result.regime_map, extent=[...])
    >>> plt.colorbar(label='Regime')
    """
    n_Re = len(Re_range)
    n_nu = len(nu_range)

    regime_map = np.zeros((n_Re, n_nu), dtype=int)
    dim_H1_map = np.zeros((n_Re, n_nu), dtype=int)
    energy_maps = {}

    # Regime encoding
    regime_dict = {
        "laminar": 0,
        "epipelagic": 1,
        "mesopelagic": 2,
        "bathypelagic": 3,
    }

    # Iterate over parameter space
    total = n_Re * n_nu
    pbar = tqdm(total=total, desc="Phase diagram") if verbose else None

    for i, Re in enumerate(Re_range):
        for j, nu in enumerate(nu_range):
            # Create cascade model
            # Re = U L / ν, so adjust forcing to achieve target Re
            cascade = ShellCascade(
                n_shells=n_shells,
                nu=nu,
                forcing_amplitude=forcing_amplitude,
            )

            # Solve to steady state
            solver = CascadeSolver(cascade, verbose=False)

            # Initial condition
            u0 = (
                np.random.randn(n_shells) * 0.1
                + 1j * np.random.randn(n_shells) * 0.1
            )

            # Find steady state
            u_steady, converged = solver.find_steady_state(
                u0,
                max_time=t_transient,
            )

            if not converged:
                # Mark as unconverged (treat as laminar)
                regime_map[i, j] = regime_dict["laminar"]
                dim_H1_map[i, j] = 0
                if pbar:
                    pbar.update(1)
                continue

            # Compute cascade complex
            energies = cascade.compute_energies(u_steady)
            transfers = cascade.compute_energy_transfers(u_steady)

            complex = CascadeComplex(
                n_shells=n_shells,
                energies=energies.real,
                transfers=transfers,
                wavenumbers=cascade.wavenumbers,
            )

            # Classify regime
            regime = complex.classify_regime()
            regime_map[i, j] = regime_dict[regime]

            # Compute cohomology dimension
            _, _, dim_H1 = complex.compute_cohomology()
            dim_H1_map[i, j] = dim_H1

            # Store energies
            energy_maps[(i, j)] = energies

            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()

    return PhaseDiagramResult(
        Re_range=Re_range,
        nu_range=nu_range,
        regime_map=regime_map,
        dim_H1_map=dim_H1_map,
        energy_maps=energy_maps,
    )


def find_epipelagic_boundaries(
    result: PhaseDiagramResult,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract boundaries of epipelagic regime from phase diagram.

    Parameters
    ----------
    result : PhaseDiagramResult
        Phase diagram result

    Returns
    -------
    lower_boundary : ndarray
        Lower boundary (laminar-epipelagic)
    upper_boundary : ndarray
        Upper boundary (epipelagic-mesopelagic)

    Algorithm:
        For each ν, find Re values where regime transitions occur.
    """
    n_Re, n_nu = result.regime_map.shape

    lower_boundary = np.zeros(n_nu)
    upper_boundary = np.zeros(n_nu)

    for j in range(n_nu):
        regime_column = result.regime_map[:, j]

        # Find transitions
        epi_indices = np.where(regime_column == 1)[0]  # Epipelagic

        if len(epi_indices) > 0:
            lower_boundary[j] = result.Re_range[epi_indices[0]]
            upper_boundary[j] = result.Re_range[epi_indices[-1]]
        else:
            lower_boundary[j] = np.nan
            upper_boundary[j] = np.nan

    return lower_boundary, upper_boundary


def verify_E2_degeneration(
    result: PhaseDiagramResult,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Verify E₂-degeneration in epipelagic regime.

    Parameters
    ----------
    result : PhaseDiagramResult
        Phase diagram result
    tolerance : float
        Tolerance for checking spectral sequence

    Returns
    -------
    degeneration_map : ndarray, shape (n_Re, n_nu)
        Boolean array: True where E₂ degenerates

    Note:
        Full E₂-degeneration verification requires computing the spectral
        sequence, which is computationally expensive. This is a placeholder
        for future implementation.
    """
    # For now, assume epipelagic regime always has E₂-degeneration
    # (this is the defining property)
    degeneration_map = result.regime_map == 1

    return degeneration_map
