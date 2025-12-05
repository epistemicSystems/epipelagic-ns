"""
Shell Cascade Model: Simplified dynamics for turbulent energy transfer.

Mathematical Model (GOY/Sabra Shell Model):
    du_n/dt = i k_n F_n(u) - ν k_n² u_n + f_n

where:
    F_n(u) = k_{n+1} (u_{n+1}* u_{n+2} + ε u_{n-1}* u_{n+1})
             + k_{n-1} (1-ε) u_{n-1}* u_{n-2}

Parameters:
    k_n = k_0 λⁿ  (wavenumbers, λ = 2 typically)
    ν = viscosity
    ε = ±1 (preserves/breaks symmetry)
    f_n = forcing (energy injection)

This captures essential cascade physics in a tractable model.
"""

from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class ShellCascade:
    """
    N-shell cascade model for turbulent energy transfer.

    Attributes
    ----------
    n_shells : int
        Number of shells in cascade
    k0 : float
        Fundamental wavenumber (largest scale)
    lambda_k : float
        Ratio between consecutive wavenumbers (typically 2.0)
    nu : float
        Viscosity
    epsilon : float
        Symmetry parameter (±1)
    forcing_shell : int
        Shell index for energy injection (typically 0 or 1)
    forcing_amplitude : float
        Amplitude of energy forcing

    Examples
    --------
    >>> cascade = ShellCascade(n_shells=8, nu=1e-3, forcing_amplitude=0.1)
    >>> u = np.random.randn(8) + 1j * np.random.randn(8)
    >>> dudt = cascade.rhs(u, t=0.0)
    """

    n_shells: int = 8
    k0: float = 1.0
    lambda_k: float = 2.0
    nu: float = 1e-3
    epsilon: float = 1.0
    forcing_shell: int = 1
    forcing_amplitude: float = 0.1

    def __post_init__(self) -> None:
        """Compute derived quantities."""
        # Wavenumbers: k_n = k0 * lambda^n
        self.wavenumbers = self.k0 * self.lambda_k ** np.arange(self.n_shells)

        # Forcing vector
        self.forcing = np.zeros(self.n_shells, dtype=complex)
        self.forcing[self.forcing_shell] = self.forcing_amplitude

    def rhs(self, u: np.ndarray, t: float) -> np.ndarray:
        """
        Right-hand side of shell cascade equations.

        Parameters
        ----------
        u : ndarray, shape (n_shells,), complex
            Shell velocities (complex)
        t : float
            Time (for time-dependent forcing, currently unused)

        Returns
        -------
        dudt : ndarray, shape (n_shells,), complex
            Time derivative du/dt

        Mathematical Form:
            du_n/dt = i k_n F_n(u) - ν k_n² u_n + f_n
        """
        dudt = np.zeros(self.n_shells, dtype=complex)

        for n in range(self.n_shells):
            # Nonlinear interaction term F_n(u)
            F_n = self._interaction_term(u, n)

            # Full RHS
            dudt[n] = (
                1j * self.wavenumbers[n] * F_n  # Nonlinear
                - self.nu * self.wavenumbers[n]**2 * u[n]  # Viscous dissipation
                + self.forcing[n]  # External forcing
            )

        return dudt

    def _interaction_term(self, u: np.ndarray, n: int) -> complex:
        """
        Compute nonlinear interaction term F_n(u).

        GOY/Sabra model:
            F_n = k_{n+1} (u_{n+1}* u_{n+2} + ε u_{n-1}* u_{n+1})
                  + k_{n-1} (1-ε) u_{n-1}* u_{n-2}

        (with appropriate boundary conditions at ends)
        """
        F = 0.0 + 0.0j

        # Forward interaction: u_{n+1}* u_{n+2}
        if n + 2 < self.n_shells:
            F += self.wavenumbers[n+1] * np.conj(u[n+1]) * u[n+2]

        # Mixed interaction: ε u_{n-1}* u_{n+1}
        if 0 <= n-1 and n+1 < self.n_shells:
            F += self.epsilon * self.wavenumbers[n+1] * np.conj(u[n-1]) * u[n+1]

        # Backward interaction: (1-ε) u_{n-1}* u_{n-2}
        if 0 <= n-2:
            F += (1 - self.epsilon) * self.wavenumbers[n-1] * np.conj(u[n-1]) * u[n-2]

        return F

    def compute_energies(self, u: np.ndarray) -> np.ndarray:
        """
        Compute shell energies E_n = |u_n|²/2.

        Parameters
        ----------
        u : ndarray, shape (n_shells,), complex
            Shell velocities

        Returns
        -------
        energies : ndarray, shape (n_shells,)
            Shell energies
        """
        return 0.5 * np.abs(u)**2

    def compute_energy_transfers(self, u: np.ndarray) -> np.ndarray:
        """
        Compute energy transfer matrix T_{nm}.

        Parameters
        ----------
        u : ndarray, shape (n_shells,), complex
            Shell velocities

        Returns
        -------
        transfers : ndarray, shape (n_shells, n_shells)
            Energy transfer matrix (antisymmetric)

        Algorithm:
            T_{nm} = -Im[ u_n* (∂u_n/∂t)_{transfer from m} ]

        This is computed from the nonlinear terms in the cascade equations.
        """
        transfers = np.zeros((self.n_shells, self.n_shells))

        for n in range(self.n_shells):
            # Contribution from interactions with neighboring shells
            for m in range(self.n_shells):
                if m == n:
                    continue

                # Estimate transfer from interaction structure
                interaction = self._interaction_transfer(u, n, m)
                transfers[n, m] = interaction
                transfers[m, n] = -interaction  # Antisymmetry

        return transfers

    def _interaction_transfer(self, u: np.ndarray, n: int, m: int) -> float:
        """Estimate energy transfer from shell n to shell m."""
        # Simplified model based on triad interactions
        if abs(n - m) == 1:
            # Adjacent shells have strong interaction
            return 0.1 * np.abs(u[n]) * np.abs(u[m]) * self.wavenumbers[min(n,m)]
        elif abs(n - m) == 2:
            # Next-nearest shells have weaker interaction
            return 0.01 * np.abs(u[n]) * np.abs(u[m]) * self.wavenumbers[min(n,m)]
        else:
            # Distant shells have negligible direct interaction
            return 0.0

    def compute_reynolds_number(self, u: np.ndarray) -> float:
        """
        Estimate Reynolds number from shell velocities.

        Re ≈ √(E₀) / (ν k₀)

        where E₀ is the large-scale energy.
        """
        E0 = self.compute_energies(u)[0]
        U = np.sqrt(2 * E0)  # Characteristic velocity
        L = 1.0 / self.k0  # Characteristic length

        Re = U * L / self.nu
        return Re

    def total_energy(self, u: np.ndarray) -> float:
        """Compute total energy: E_total = Σ E_n."""
        return np.sum(self.compute_energies(u))

    def energy_dissipation_rate(self, u: np.ndarray) -> float:
        """
        Compute total energy dissipation rate.

        ε_dissip = Σ_n ν k_n² |u_n|²
        """
        energies = self.compute_energies(u)
        dissipation = np.sum(2 * self.nu * self.wavenumbers**2 * energies)
        return dissipation

    def energy_injection_rate(self) -> float:
        """Compute energy injection rate from forcing."""
        # For complex forcing: ε_inject = 2 Re[f_n* u_n]
        # Here we return the maximum possible injection
        return self.forcing_amplitude**2 / (self.nu * self.wavenumbers[self.forcing_shell]**2)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ShellCascade(n_shells={self.n_shells}, "
            f"ν={self.nu:.2e}, k₀={self.k0}, λ={self.lambda_k})"
        )
