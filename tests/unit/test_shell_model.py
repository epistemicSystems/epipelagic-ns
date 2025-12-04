"""
Unit tests for shell cascade model.
"""

import pytest
import numpy as np
from epipelagic.cascade.shell_model import ShellCascade


class TestShellCascade:
    """Test suite for shell cascade model."""

    def test_initialization(self):
        """Test cascade model initialization."""
        cascade = ShellCascade(n_shells=8, nu=1e-3)

        assert cascade.n_shells == 8
        assert cascade.nu == 1e-3
        assert len(cascade.wavenumbers) == 8

    def test_wavenumber_scaling(self):
        """Test wavenumber geometric progression."""
        cascade = ShellCascade(n_shells=5, k0=1.0, lambda_k=2.0)

        expected = np.array([1, 2, 4, 8, 16])
        np.testing.assert_array_almost_equal(cascade.wavenumbers, expected)

    def test_rhs_shape(self):
        """Test RHS function returns correct shape."""
        cascade = ShellCascade(n_shells=6)
        u = np.random.randn(6) + 1j * np.random.randn(6)

        dudt = cascade.rhs(u, t=0.0)

        assert dudt.shape == u.shape
        assert dudt.dtype == np.complex128

    def test_energy_conservation(self):
        """Test energy conservation in absence of forcing/dissipation."""
        # No forcing or dissipation
        cascade = ShellCascade(
            n_shells=4,
            nu=0.0,  # No viscosity
            forcing_amplitude=0.0,  # No forcing
        )

        u = np.array([1.0, 0.5, 0.25, 0.125]) + 0j

        # Energy should be conserved (dE/dt ≈ 0)
        dudt = cascade.rhs(u, t=0.0)

        # Compute dE/dt = Re[u* du/dt]
        dE_dt = 2 * np.real(np.sum(np.conj(u) * dudt))

        assert abs(dE_dt) < 1e-10, "Energy should be conserved without forcing/dissipation"

    def test_energy_dissipation(self):
        """Test viscous dissipation rate."""
        cascade = ShellCascade(n_shells=4, nu=1e-2)
        u = np.ones(4) + 0j

        dissipation = cascade.energy_dissipation_rate(u)

        # Should be positive
        assert dissipation > 0

    def test_reynolds_number(self):
        """Test Reynolds number estimation."""
        cascade = ShellCascade(n_shells=5, nu=1e-3, k0=1.0)
        u = np.array([1.0, 0.5, 0.25, 0.125, 0.0625]) + 0j

        Re = cascade.compute_reynolds_number(u)

        # Should be positive and finite
        assert Re > 0
        assert np.isfinite(Re)

    def test_total_energy(self):
        """Test total energy computation."""
        cascade = ShellCascade(n_shells=3)
        u = np.array([1.0, 2.0, 3.0]) + 0j

        E_total = cascade.total_energy(u)
        E_expected = 0.5 * (1**2 + 2**2 + 3**2)

        assert abs(E_total - E_expected) < 1e-10
