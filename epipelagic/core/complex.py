"""
Cascade Complex: Cochain complex (C•, d•) for turbulent cascade dynamics.

Mathematical Structure:
    C⁰ = ⊕ₙ ℝ Eₙ     (shell energies)
    C¹ = ⊕ₙ<ₘ ℝ Tₙₘ  (energy transfers)
    C² = 0            (higher cochains vanish in this formulation)

    d⁰: C⁰ → C¹ encodes energy conservation constraints
    d¹: C¹ → C² detects cascade closure

The cohomology H¹(C•) captures persistent cross-scale structures.
"""

from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class CascadeComplex:
    """
    Cochain complex for cascade dynamics.

    Attributes
    ----------
    n_shells : int
        Number of cascade shells
    energies : ndarray, shape (n_shells,)
        Shell energies E₀, E₁, ..., Eₙ₋₁
    transfers : ndarray, shape (n_shells, n_shells)
        Energy transfer matrix Tₙₘ (antisymmetric: Tₙₘ = -Tₘₙ)
    wavenumbers : ndarray, shape (n_shells,)
        Representative wavenumber for each shell kₙ
    """

    n_shells: int
    energies: np.ndarray
    transfers: np.ndarray
    wavenumbers: np.ndarray

    def __post_init__(self) -> None:
        """Validate complex structure."""
        assert self.energies.shape == (self.n_shells,), \
            f"Energy shape mismatch: {self.energies.shape} != ({self.n_shells},)"
        assert self.transfers.shape == (self.n_shells, self.n_shells), \
            f"Transfer shape mismatch: {self.transfers.shape}"
        assert self.wavenumbers.shape == (self.n_shells,), \
            f"Wavenumber shape mismatch"

        # Verify antisymmetry of transfer matrix
        antisym_error = np.max(np.abs(self.transfers + self.transfers.T))
        if antisym_error > 1e-10:
            raise ValueError(f"Transfer matrix not antisymmetric: max error = {antisym_error}")

    @classmethod
    def from_velocity_field(
        cls,
        velocity: np.ndarray,
        n_shells: int = 8,
        k_min: float = 1.0,
        k_ratio: float = 2.0,
        method: str = "phenomenological",
    ) -> "CascadeComplex":
        """
        Construct cascade complex from velocity field via shell decomposition.

        Parameters
        ----------
        velocity : ndarray, shape (nx, ny, nz, 3) or (nx, ny, 3)
            Velocity field u(x) with components (ux, uy, uz)
        n_shells : int
            Number of shells in decomposition
        k_min : float
            Minimum wavenumber (largest scale)
        k_ratio : float
            Ratio between consecutive shells kₙ₊₁/kₙ
        method : str
            Transfer computation method:
            - 'phenomenological': Simple model (fast, approximate)
            - 'spectral': Direct computation from nonlinear terms (slow, accurate)

        Returns
        -------
        complex : CascadeComplex
            Cascade complex extracted from velocity field

        Algorithm
        ---------
        1. FFT velocity field: û(k) = ℱ[u(x)]
        2. Define shells: Sₙ = {k : kₙ ≤ |k| < kₙ₊₁}
        3. Project to shells: ûₙ(k) = û(k) · 𝟙_{Sₙ}(k)
        4. Compute energies: Eₙ = ½∫|ûₙ|² dk
        5. Compute transfers: Tₙₘ = -∫ûₙ · 𝒫ₘ[(u·∇)u] dk
           (if method='spectral', otherwise use phenomenological model)
        """
        if method == "spectral":
            # Use direct transfer measurement (Task 2.4)
            from epipelagic.core.transfer_matrix import compute_transfer_matrix_spectral

            transfer_obj = compute_transfer_matrix_spectral(
                velocity, n_shells=n_shells, k_min=k_min, k_ratio=k_ratio
            )

            # Extract wavenumbers (computed from k_min and k_ratio)
            wavenumbers = k_min * k_ratio ** np.arange(n_shells)

            return cls(
                n_shells=n_shells,
                energies=transfer_obj.energies,
                transfers=transfer_obj.T,
                wavenumbers=wavenumbers,
            )

        # Otherwise, use phenomenological model (original implementation)
        from numpy.fft import fftn, ifftn, fftfreq

        # Determine dimensionality
        ndim = velocity.ndim - 1
        spatial_shape = velocity.shape[:-1]

        # Compute FFT
        u_hat = fftn(velocity, axes=tuple(range(ndim)))

        # Build wavenumber grid
        if ndim == 2:
            kx = fftfreq(spatial_shape[0], d=1.0) * 2 * np.pi
            ky = fftfreq(spatial_shape[1], d=1.0) * 2 * np.pi
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            k_mag = np.sqrt(KX**2 + KY**2)
        elif ndim == 3:
            kx = fftfreq(spatial_shape[0], d=1.0) * 2 * np.pi
            ky = fftfreq(spatial_shape[1], d=1.0) * 2 * np.pi
            kz = fftfreq(spatial_shape[2], d=1.0) * 2 * np.pi
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
            k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
        else:
            raise ValueError(f"Unsupported dimensionality: {ndim}")

        # Define shell boundaries
        wavenumbers = k_min * k_ratio ** np.arange(n_shells)
        k_bounds = np.concatenate([wavenumbers, [wavenumbers[-1] * k_ratio]])

        # Project to shells and compute energies
        energies = np.zeros(n_shells)
        u_shells = []

        for n in range(n_shells):
            # Shell indicator function
            mask = (k_mag >= k_bounds[n]) & (k_mag < k_bounds[n+1])

            # Project velocity field
            u_n_hat = u_hat.copy()
            for i in range(velocity.shape[-1]):
                u_n_hat[..., i] *= mask

            # Compute shell energy (in Fourier space)
            energy_density = np.sum(np.abs(u_n_hat)**2, axis=-1)
            energies[n] = 0.5 * np.sum(energy_density) / np.prod(spatial_shape)

            # Store shell velocity for transfer computation
            u_shells.append(u_n_hat)

        # Compute energy transfers (simplified phenomenological model)
        # Full calculation requires: Tₙₘ = -∫ûₙ·𝒫ₘ[(u·∇)u] dk
        # Here we use a phenomenological model based on shell energies
        transfers = np.zeros((n_shells, n_shells))

        for n in range(n_shells - 1):
            # Forward transfer (energy cascade)
            transfers[n, n+1] = -0.1 * energies[n] * wavenumbers[n]
            transfers[n+1, n] = -transfers[n, n+1]  # Antisymmetry

        return cls(
            n_shells=n_shells,
            energies=energies,
            transfers=transfers,
            wavenumbers=wavenumbers,
        )

    def differential_d0(self) -> np.ndarray:
        """
        Compute differential d⁰: C⁰ → C¹.

        Returns
        -------
        d0 : ndarray, shape (n_transfers, n_shells)
            Matrix representation of d⁰ where n_transfers = n_shells * (n_shells - 1) / 2

        Mathematical Definition:
            d⁰ maps shell energies to energy conservation constraints.
            For each transfer Tₙₘ, the constraint is ∂Eₙ/∂t + ∂Eₘ/∂t = -Tₙₘ
        """
        n_transfers = self.n_shells * (self.n_shells - 1) // 2
        d0 = np.zeros((n_transfers, self.n_shells))

        idx = 0
        for n in range(self.n_shells):
            for m in range(n + 1, self.n_shells):
                d0[idx, n] = 1.0
                d0[idx, m] = 1.0
                idx += 1

        return d0

    def differential_d1(self) -> np.ndarray:
        """
        Compute differential d¹: C¹ → C².

        Returns
        -------
        d1 : ndarray, shape (n_cochains2, n_transfers)
            Matrix representation of d¹

        In the epipelagic regime, d¹ should vanish (E₂-degeneration).
        """
        n_transfers = self.n_shells * (self.n_shells - 1) // 2
        # For now, C² is trivial, so d¹ maps to zero
        d1 = np.zeros((0, n_transfers))
        return d1

    def compute_cohomology(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Compute cohomology groups H⁰, H¹ of the cascade complex.

        Returns
        -------
        H0_basis : ndarray
            Basis for H⁰ = ker(d⁰)
        H1_basis : ndarray
            Basis for H¹ = ker(d¹) / im(d⁰)
        dim_H1 : int
            Dimension of epipelagic cohomology dim(H¹)

        Algorithm:
            1. Compute d⁰ and d¹
            2. Find ker(d⁰) and ker(d¹) via SVD
            3. Compute quotient: H¹ = ker(d¹) / im(d⁰)
        """
        d0 = self.differential_d0()
        d1 = self.differential_d1()

        # Compute kernel of d⁰
        _, S0, Vh0 = np.linalg.svd(d0, full_matrices=True)
        rank_d0 = np.sum(S0 > 1e-10)
        H0_basis = Vh0[rank_d0:].T  # Basis for ker(d⁰)

        # Compute kernel of d¹ (if d¹ is nontrivial)
        if d1.shape[0] > 0:
            _, S1, Vh1 = np.linalg.svd(d1, full_matrices=True)
            rank_d1 = np.sum(S1 > 1e-10)
            ker_d1_basis = Vh1[rank_d1:].T
        else:
            # If C² is trivial, ker(d¹) = C¹
            n_transfers = self.n_shells * (self.n_shells - 1) // 2
            ker_d1_basis = np.eye(n_transfers)

        # Compute H¹ = ker(d¹) / im(d⁰)
        # Image of d⁰ is the column space of d⁰
        U0, S0, _ = np.linalg.svd(d0, full_matrices=True)
        rank_d0 = np.sum(S0 > 1e-10)
        im_d0_basis = U0[:, :rank_d0]

        # Find orthogonal complement of im(d⁰) within ker(d¹)
        # Project ker(d¹) onto orthogonal complement of im(d⁰)
        if im_d0_basis.shape[1] > 0:
            # QR decomposition to find orthogonal complement
            combined = np.hstack([im_d0_basis, ker_d1_basis])
            Q, R = np.linalg.qr(combined)

            # H¹ basis is columns of Q corresponding to ker(d¹) that are orthogonal to im(d⁰)
            H1_basis = Q[:, rank_d0:]
            dim_H1 = H1_basis.shape[1]
        else:
            H1_basis = ker_d1_basis
            dim_H1 = ker_d1_basis.shape[1]

        return H0_basis, H1_basis, dim_H1

    def classify_regime(self) -> str:
        """
        Classify cascade regime based on transfer ratios.

        Returns
        -------
        regime : str
            One of: 'laminar', 'epipelagic', 'mesopelagic', 'bathypelagic'

        Classification Criteria:
            - Laminar: ρ₁ = T₀₁/E₀ < 0.05
            - Epipelagic: 0.05 ≤ ρ₁ < 0.30 and ρ₂ = T₁₂/T₀₁ < 0.50
            - Mesopelagic: ρ₁ ≥ 0.30 or ρ₂ ≥ 0.50
            - Bathypelagic: Fully developed turbulence (Re → ∞)
        """
        # Compute transfer ratios
        if self.energies[0] > 1e-10:
            rho1 = abs(self.transfers[0, 1]) / self.energies[0]
        else:
            rho1 = 0.0

        if abs(self.transfers[0, 1]) > 1e-10 and self.n_shells >= 3:
            rho2 = abs(self.transfers[1, 2]) / abs(self.transfers[0, 1])
        else:
            rho2 = 0.0

        # Classify
        if rho1 < 0.05:
            return "laminar"
        elif rho1 < 0.30 and rho2 < 0.50:
            return "epipelagic"
        elif rho1 < 0.80:
            return "mesopelagic"
        else:
            return "bathypelagic"

    def __repr__(self) -> str:
        """String representation of cascade complex."""
        regime = self.classify_regime()
        _, _, dim_H1 = self.compute_cohomology()
        return (
            f"CascadeComplex(n_shells={self.n_shells}, "
            f"regime='{regime}', dim(H¹)={dim_H1})"
        )
