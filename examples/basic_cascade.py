#!/usr/bin/env python3
"""
Basic 3-shell cascade example.

Demonstrates:
1. Creating a shell cascade model
2. Time integration
3. Energy spectrum visualization
4. Regime classification
"""

import numpy as np
import matplotlib.pyplot as plt
from epipelagic.cascade.shell_model import ShellCascade
from epipelagic.cascade.solver import CascadeSolver
from epipelagic.core.complex import CascadeComplex


def main():
    """Run basic cascade example."""
    print("=" * 60)
    print("EPIPELAGIC TURBULENCE: Basic 3-Shell Cascade Example")
    print("=" * 60)

    # Create 8-shell cascade
    print("\n[1] Creating shell cascade model...")
    cascade = ShellCascade(
        n_shells=8,
        nu=1e-3,  # Viscosity
        k0=1.0,
        lambda_k=2.0,  # Wavenumber ratio
        forcing_amplitude=0.1,
        forcing_shell=1,  # Force at large scales
    )

    print(f"    Wavenumbers: {cascade.wavenumbers}")
    print(f"    Viscosity: ν = {cascade.nu}")

    # Initial condition (small random perturbation)
    print("\n[2] Setting initial condition...")
    u0 = np.random.randn(8) * 0.01 + 1j * np.random.randn(8) * 0.01

    # Time integration
    print("\n[3] Integrating to steady state...")
    solver = CascadeSolver(cascade, dt=1e-3, verbose=True)

    u_steady, converged = solver.find_steady_state(
        u0,
        max_time=100.0,
        energy_tolerance=1e-6,
    )

    if converged:
        print("    ✓ Steady state reached")
    else:
        print("    ⚠ Did not fully converge")

    # Compute cascade complex
    print("\n[4] Computing cascade complex...")
    energies = cascade.compute_energies(u_steady)
    transfers = cascade.compute_energy_transfers(u_steady)

    complex = CascadeComplex(
        n_shells=8,
        energies=energies.real,
        transfers=transfers,
        wavenumbers=cascade.wavenumbers,
    )

    print(f"    Shell energies: {energies.real}")
    print(f"    Regime: {complex.classify_regime()}")

    # Compute cohomology
    print("\n[5] Computing epipelagic cohomology...")
    _, _, dim_H1 = complex.compute_cohomology()
    print(f"    dim(H¹_epi) = {dim_H1}")

    # Estimate Reynolds number
    Re = cascade.compute_reynolds_number(u_steady)
    print(f"    Reynolds number: Re ≈ {Re:.1f}")

    # Visualization
    print("\n[6] Creating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Energy spectrum
    axes[0].loglog(cascade.wavenumbers, energies.real, 'o-', linewidth=2, markersize=8)
    axes[0].loglog(
        cascade.wavenumbers,
        cascade.wavenumbers**(-5/3),
        '--',
        label='k⁻⁵/³ (Kolmogorov)',
        alpha=0.5,
    )
    axes[0].set_xlabel('Wavenumber k_n', fontsize=12)
    axes[0].set_ylabel('Energy E_n', fontsize=12)
    axes[0].set_title('Energy Spectrum', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Transfer matrix
    im = axes[1].imshow(
        transfers,
        cmap='RdBu_r',
        vmin=-np.max(np.abs(transfers)),
        vmax=np.max(np.abs(transfers)),
    )
    axes[1].set_xlabel('Shell index m', fontsize=12)
    axes[1].set_ylabel('Shell index n', fontsize=12)
    axes[1].set_title('Energy Transfer Matrix T_nm', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[1], label='Transfer rate')

    plt.tight_layout()
    plt.savefig('cascade_example.png', dpi=150)
    print("    Saved: cascade_example.png")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
