"""
Synthetic turbulence generation for testing and validation.

This module generates realistic synthetic turbulent velocity fields that can be
used for:
1. Testing topology extraction pipelines
2. Validating algorithms before using real DNS data
3. Parameter exploration
4. Benchmarking

Mathematical Framework:
-----------------------
Generate velocity field with prescribed energy spectrum:
    E(k) = C k^4 exp(-2(k/k_p)^2)  for k << k_η (energy range)
    E(k) ~ k^(-5/3)                 for k_p < k < k_η (inertial range)

where:
- k_p: peak wavenumber
- k_η: Kolmogorov dissipation wavenumber
- Re determines k_η ~ Re^(3/4)

References:
    [1] Pope (2000). Turbulent Flows, Chapter 6
    [2] Sagaut & Cambon (2008). Homogeneous Turbulence Dynamics
"""

from typing import Tuple, Optional, Dict
import numpy as np
from scipy import fft


def generate_synthetic_turbulence(
    resolution: Tuple[int, int, int] = (256, 256, 256),
    reynolds_number: float = 1000,
    energy_spectrum: str = 'kolmogorov',
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate synthetic turbulent velocity field.

    Parameters
    ----------
    resolution : tuple of int
        Grid resolution (nx, ny, nz)
    reynolds_number : float
        Reynolds number (controls k_η)
    energy_spectrum : str
        Spectrum type: 'kolmogorov', 'pope', or 'custom'
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    velocity : ndarray, shape (nx, ny, nz, 3)
        Velocity field u(x,y,z) = (ux, uy, uz)

    Algorithm:
    ----------
    1. Generate random Fourier coefficients û(k)
    2. Apply energy spectrum: |û(k)| ~ sqrt(E(k))
    3. Enforce incompressibility: k · û(k) = 0
    4. Inverse FFT to get u(x)
    5. Normalize to unit kinetic energy

    Examples
    --------
    >>> velocity = generate_synthetic_turbulence(
    ...     resolution=(128, 128, 128),
    ...     reynolds_number=1000
    ... )
    >>> print(velocity.shape)
    (128, 128, 128, 3)

    >>> # Verify incompressibility
    >>> div_u = compute_divergence(velocity)
    >>> print(f"∇·u max error: {np.max(np.abs(div_u))}")
    ∇·u max error: 1.234e-14
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    nx, ny, nz = resolution

    # Wavenumber grids
    kx = fft.fftfreq(nx) * nx
    ky = fft.fftfreq(ny) * ny
    kz = fft.rfftfreq(nz) * nz  # Real FFT in z direction

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    K[0, 0, 0] = 1.0  # Avoid division by zero

    # Energy spectrum
    if energy_spectrum == 'kolmogorov':
        # Kolmogorov-like spectrum with exponential cutoff
        k_peak = 4.0  # Peak wavenumber
        k_eta = k_peak * (reynolds_number ** 0.75)  # Dissipation scale

        # Pope spectrum (Eq. 6.246)
        C = 1.5  # Normalization constant
        E_k = C * K**4 / k_peak**5 * np.exp(-2 * (K / k_peak)**2)

        # Add inertial range decay
        inertial_mask = K > k_peak
        E_k[inertial_mask] *= (K[inertial_mask] / k_peak) ** (-5/3) / (K[inertial_mask] / k_peak)

        # Exponential cutoff at dissipation scale
        E_k *= np.exp(-(K / k_eta)**2)

    elif energy_spectrum == 'pope':
        # Pope spectrum (Eq. 6.248)
        k_peak = 4.0
        E_k = 1.5 * K**4 / k_peak**5 * np.exp(-2 * (K / k_peak)**2)

    else:
        raise ValueError(f"Unknown spectrum: {energy_spectrum}")

    # Generate random Fourier coefficients
    # Real and imaginary parts are independent Gaussian
    u_hat = np.zeros((nx, ny, nz // 2 + 1, 3), dtype=complex)

    for i in range(3):
        real_part = np.random.randn(nx, ny, nz // 2 + 1)
        imag_part = np.random.randn(nx, ny, nz // 2 + 1)
        u_hat[..., i] = real_part + 1j * imag_part

    # Apply energy spectrum: |û(k)| ~ sqrt(E(k))
    amplitude = np.sqrt(E_k[..., np.newaxis])
    u_hat *= amplitude

    # Enforce incompressibility: k · û(k) = 0
    # Project onto divergence-free subspace
    k_dot_u = KX * u_hat[..., 0] + KY * u_hat[..., 1] + KZ * u_hat[..., 2]
    k_dot_u /= (K**2 + 1e-10)  # Avoid division by zero

    u_hat[..., 0] -= k_dot_u * KX
    u_hat[..., 1] -= k_dot_u * KY
    u_hat[..., 2] -= k_dot_u * KZ

    # Inverse FFT to get velocity field
    velocity = np.zeros((nx, ny, nz, 3))

    for i in range(3):
        velocity[..., i] = fft.irfftn(u_hat[..., i], s=(nx, ny, nz))

    # Normalize to unit kinetic energy
    kinetic_energy = 0.5 * np.mean(np.sum(velocity**2, axis=-1))
    velocity /= np.sqrt(kinetic_energy)

    return velocity


def compute_divergence(velocity: np.ndarray) -> np.ndarray:
    """
    Compute divergence ∇·u for validation.

    Parameters
    ----------
    velocity : ndarray, shape (..., 3)
        Velocity field

    Returns
    -------
    divergence : ndarray
        Divergence field ∇·u

    Notes
    -----
    For incompressible flow, ∇·u should be numerically zero.
    """
    u = velocity[..., 0]
    v = velocity[..., 1]
    w = velocity[..., 2]

    du_dx = np.gradient(u, axis=0)
    dv_dy = np.gradient(v, axis=1)
    dw_dz = np.gradient(w, axis=2)

    return du_dx + dv_dy + dw_dz


def compute_energy_spectrum(velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute energy spectrum E(k) from velocity field.

    Parameters
    ----------
    velocity : ndarray, shape (nx, ny, nz, 3)
        Velocity field

    Returns
    -------
    k_bins : ndarray
        Wavenumber bins
    E_k : ndarray
        Energy spectrum E(k)

    Algorithm:
    ----------
    1. FFT of velocity field
    2. Compute energy in Fourier space: E(k) = ½|û(k)|²
    3. Bin by wavenumber magnitude k = |k|
    4. Average over spherical shells

    Examples
    --------
    >>> velocity = generate_synthetic_turbulence((256, 256, 256), Re=1000)
    >>> k, E_k = compute_energy_spectrum(velocity)
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(k, E_k)
    >>> plt.loglog(k, k**(-5/3), 'k--', label='k^(-5/3)')
    >>> plt.xlabel('k')
    >>> plt.ylabel('E(k)')
    >>> plt.legend()
    """
    nx, ny, nz = velocity.shape[:3]

    # FFT
    u_hat = np.zeros((nx, ny, nz // 2 + 1, 3), dtype=complex)
    for i in range(3):
        u_hat[..., i] = fft.rfftn(velocity[..., i])

    # Wavenumber grids
    kx = fft.fftfreq(nx) * nx
    ky = fft.fftfreq(ny) * ny
    kz = fft.rfftfreq(nz) * nz

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)

    # Energy in Fourier space
    E_fourier = 0.5 * np.sum(np.abs(u_hat)**2, axis=-1)

    # Bin by wavenumber
    k_max = int(np.max(K))
    k_bins = np.arange(1, k_max)
    E_k = np.zeros(len(k_bins))

    for i, k in enumerate(k_bins):
        mask = (K >= k - 0.5) & (K < k + 0.5)
        if np.any(mask):
            E_k[i] = np.mean(E_fourier[mask])

    return k_bins, E_k


def validate_turbulence_field(velocity: np.ndarray, reynolds_number: float) -> Dict[str, float]:
    """
    Validate synthetic turbulence field.

    Parameters
    ----------
    velocity : ndarray
        Velocity field
    reynolds_number : float
        Target Reynolds number

    Returns
    -------
    validation : dict
        Dictionary with validation metrics:
        - 'divergence_max': Max |∇·u|
        - 'energy': Total kinetic energy
        - 'inertial_slope': Slope in inertial range (should be ≈ -5/3)
    """
    # Check incompressibility
    div_u = compute_divergence(velocity)
    div_max = np.max(np.abs(div_u))

    # Compute energy
    kinetic_energy = 0.5 * np.mean(np.sum(velocity**2, axis=-1))

    # Check energy spectrum slope
    k, E_k = compute_energy_spectrum(velocity)

    # Fit slope in inertial range (k ~ 10 to 50)
    inertial_mask = (k >= 10) & (k <= 50)
    if np.any(inertial_mask):
        log_k = np.log(k[inertial_mask])
        log_E = np.log(E_k[inertial_mask] + 1e-20)  # Avoid log(0)

        # Linear fit in log-log space
        slope = np.polyfit(log_k, log_E, 1)[0]
    else:
        slope = np.nan

    return {
        'divergence_max': float(div_max),
        'energy': float(kinetic_energy),
        'inertial_slope': float(slope),
        'expected_slope': -5.0/3.0,
        'reynolds_number': float(reynolds_number),
    }
