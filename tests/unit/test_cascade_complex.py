"""
Unit tests for CascadeComplex class.
"""

import pytest
import numpy as np
from epipelagic.core.complex import CascadeComplex


class TestCascadeComplex:
    """Test suite for cascade complex operations."""

    def test_initialization(self):
        """Test basic initialization."""
        n_shells = 5
        energies = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        transfers = np.zeros((n_shells, n_shells))
        transfers[0, 1] = -0.1
        transfers[1, 0] = 0.1
        wavenumbers = 2.0 ** np.arange(n_shells)

        complex = CascadeComplex(
            n_shells=n_shells,
            energies=energies,
            transfers=transfers,
            wavenumbers=wavenumbers,
        )

        assert complex.n_shells == n_shells
        assert len(complex.energies) == n_shells

    def test_antisymmetry(self):
        """Test transfer matrix antisymmetry requirement."""
        n_shells = 3
        energies = np.ones(n_shells)
        transfers = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])  # Antisymmetric
        wavenumbers = np.arange(n_shells)

        # Should not raise
        complex = CascadeComplex(
            n_shells=n_shells,
            energies=energies,
            transfers=transfers,
            wavenumbers=wavenumbers,
        )

        # Test symmetric matrix (should fail)
        transfers_sym = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        with pytest.raises(ValueError, match="not antisymmetric"):
            CascadeComplex(
                n_shells=n_shells,
                energies=energies,
                transfers=transfers_sym,
                wavenumbers=wavenumbers,
            )

    def test_differential_d0(self):
        """Test d⁰ differential computation."""
        n_shells = 3
        energies = np.ones(n_shells)
        transfers = np.zeros((n_shells, n_shells))
        wavenumbers = np.arange(n_shells)

        complex = CascadeComplex(
            n_shells=n_shells,
            energies=energies,
            transfers=transfers,
            wavenumbers=wavenumbers,
        )

        d0 = complex.differential_d0()

        # Should have correct shape
        n_transfers = n_shells * (n_shells - 1) // 2
        assert d0.shape == (n_transfers, n_shells)

    def test_cohomology_computation(self):
        """Test cohomology group computation."""
        n_shells = 4
        energies = np.array([1.0, 0.5, 0.25, 0.125])
        transfers = np.zeros((n_shells, n_shells))

        # Simple cascade: only adjacent transfers
        transfers[0, 1] = -0.1
        transfers[1, 0] = 0.1
        transfers[1, 2] = -0.05
        transfers[2, 1] = 0.05

        wavenumbers = 2.0 ** np.arange(n_shells)

        complex = CascadeComplex(
            n_shells=n_shells,
            energies=energies,
            transfers=transfers,
            wavenumbers=wavenumbers,
        )

        H0_basis, H1_basis, dim_H1 = complex.compute_cohomology()

        # Basic checks
        assert dim_H1 >= 0
        assert H1_basis.shape[1] == dim_H1

    def test_regime_classification(self):
        """Test cascade regime classification."""
        n_shells = 5
        energies = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        wavenumbers = 2.0 ** np.arange(n_shells)

        # Laminar regime (weak transfer)
        transfers_lam = np.zeros((n_shells, n_shells))
        transfers_lam[0, 1] = -0.01 * energies[0]
        transfers_lam[1, 0] = 0.01 * energies[0]

        complex_lam = CascadeComplex(
            n_shells=n_shells,
            energies=energies,
            transfers=transfers_lam,
            wavenumbers=wavenumbers,
        )

        assert complex_lam.classify_regime() == "laminar"

        # Epipelagic regime (moderate transfer)
        transfers_epi = np.zeros((n_shells, n_shells))
        transfers_epi[0, 1] = -0.1 * energies[0]
        transfers_epi[1, 0] = 0.1 * energies[0]
        transfers_epi[1, 2] = -0.02 * energies[0]
        transfers_epi[2, 1] = 0.02 * energies[0]

        complex_epi = CascadeComplex(
            n_shells=n_shells,
            energies=energies,
            transfers=transfers_epi,
            wavenumbers=wavenumbers,
        )

        regime = complex_epi.classify_regime()
        assert regime in ["epipelagic", "mesopelagic"]

    def test_from_velocity_field_2d(self):
        """Test construction from 2D velocity field."""
        # Create simple vortex field
        nx, ny = 32, 32
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Vortex velocity field
        r = np.sqrt(X**2 + Y**2) + 1e-10
        u = -Y / r
        v = X / r

        velocity = np.stack([u, v], axis=-1)

        # Extract cascade complex
        complex = CascadeComplex.from_velocity_field(
            velocity,
            n_shells=4,
            k_min=1.0,
            k_ratio=2.0,
        )

        assert complex.n_shells == 4
        assert np.all(complex.energies >= 0)  # Energies non-negative
