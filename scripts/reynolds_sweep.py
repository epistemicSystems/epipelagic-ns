#!/usr/bin/env python3
"""
Reynolds Number Parameter Sweep

Identifies epipelagic regime boundaries in (Re, ν) parameter space
by computing energy transfer ratios ρ₁ = T₀₁/E₀ and ρ₂ = T₁₂/T₀₁.

Phase boundaries:
- Laminar-Epipelagic: ρ₁ ≈ 0.05
- Epipelagic-Mesopelagic: ρ₂ ≈ 0.30

Usage:
    python scripts/reynolds_sweep.py
    python scripts/reynolds_sweep.py --re-min 100 --re-max 1000000 --n-points 50
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from epipelagic.cascade.taichi_solver import TAICHI_AVAILABLE, TaichiCascadeSolver


def compute_steady_state(solver, u0, n_transient=50000, n_average=10000, dt=1e-3):
    """
    Evolve to steady state and compute time-averaged quantities.

    Parameters
    ----------
    solver : TaichiCascadeSolver
        Cascade solver
    u0 : ndarray
        Initial condition
    n_transient : int
        Steps to reach steady state
    n_average : int
        Steps for time averaging
    dt : float
        Timestep

    Returns
    -------
    E_mean : ndarray
        Mean shell energies
    T_mean : ndarray
        Mean transfer rates (lower triangular matrix)
    """
    # Transient phase
    u = solver.integrate(u0, n_steps=n_transient, dt=dt)

    # Averaging phase
    E_sum = np.zeros(solver.n_shells)
    T_count = 0

    for _ in range(n_average):
        u = solver.integrate(u, n_steps=1, dt=dt)
        E_n = 0.5 * np.abs(u)**2
        E_sum += E_n
        T_count += 1

    E_mean = E_sum / T_count

    return E_mean, u


def compute_transfer_rates(E, k, nu):
    """
    Estimate transfer rates from energy profile.

    For simplicity, use dimensional analysis:
    T_{nm} ~ E_n^{3/2} k_n (approximate)

    Returns
    -------
    T : ndarray
        Transfer matrix approximation
    """
    n_shells = len(E)
    T = np.zeros((n_shells, n_shells))

    for n in range(n_shells - 1):
        # Forward transfer (approximate)
        T[n, n+1] = np.sqrt(E[n]) * k[n]

    return T


def classify_regime(E, k, nu):
    """
    Classify flow regime based on energy ratios.

    Returns
    -------
    regime : str
        One of: 'laminar', 'epipelagic', 'mesopelagic', 'bathypelagic'
    rho1 : float
        First transfer ratio
    rho2 : float
        Second transfer ratio
    """
    # Compute approximate transfer rates
    T = compute_transfer_rates(E, k, nu)

    # Transfer ratios
    if E[0] > 1e-10:
        rho1 = T[0, 1] / E[0] if len(E) > 1 else 0.0
    else:
        rho1 = 0.0

    if T[0, 1] > 1e-10:
        rho2 = T[1, 2] / T[0, 1] if len(E) > 2 else 0.0
    else:
        rho2 = 0.0

    # Classify
    if rho1 < 0.05:
        regime = 'laminar'
    elif rho2 < 0.30:
        regime = 'epipelagic'
    elif rho2 < 0.80:
        regime = 'mesopelagic'
    else:
        regime = 'bathypelagic'

    return regime, rho1, rho2


def reynolds_sweep(
    Re_min=100,
    Re_max=1e6,
    n_points=50,
    n_shells=5,
    nu_fixed=1e-3,
    output_dir="experiments/phase1_foundation",
):
    """
    Parameter sweep over Reynolds number.

    Parameters
    ----------
    Re_min, Re_max : float
        Reynolds number range
    n_points : int
        Number of points to sample
    n_shells : int
        Number of cascade shells
    nu_fixed : float
        Fixed viscosity
    output_dir : str
        Output directory

    Returns
    -------
    results : dict
        Sweep results
    """
    if not TAICHI_AVAILABLE:
        print("ERROR: Taichi not available")
        return None

    print("=" * 80)
    print("REYNOLDS NUMBER PARAMETER SWEEP")
    print("=" * 80)
    print(f"Re range: [{Re_min}, {Re_max}]")
    print(f"Points: {n_points}")
    print(f"Shells: {n_shells}")
    print(f"Viscosity: {nu_fixed}")
    print()

    # Reynolds number grid (log-spaced)
    Re_values = np.logspace(np.log10(Re_min), np.log10(Re_max), n_points)

    results = {
        "config": {
            "Re_min": Re_min,
            "Re_max": Re_max,
            "n_points": n_points,
            "n_shells": n_shells,
            "nu_fixed": nu_fixed,
        },
        "sweep": []
    }

    regime_counts = {'laminar': 0, 'epipelagic': 0, 'mesopelagic': 0, 'bathypelagic': 0}

    for Re in tqdm(Re_values, desc="Reynolds sweep"):
        # Adjust forcing to maintain Re
        # Re ~ forcing_amplitude / nu
        forcing_amplitude = Re * nu_fixed / 10.0  # Scale factor

        # Create solver
        solver = TaichiCascadeSolver(
            n_shells=n_shells,
            nu=nu_fixed,
            forcing_amplitude=forcing_amplitude,
            forcing_shell=1,
        )

        # Initial condition
        u0 = np.random.randn(n_shells) + 1j * np.random.randn(n_shells)
        u0 *= 0.01

        # Compute steady state
        try:
            E_mean, u_final = compute_steady_state(
                solver, u0,
                n_transient=10000,
                n_average=2000,
                dt=1e-3,
            )

            # Get wavenumbers
            k = solver.k.to_numpy()

            # Classify regime
            regime, rho1, rho2 = classify_regime(E_mean, k, nu_fixed)
            regime_counts[regime] += 1

            result = {
                "Re": float(Re),
                "nu": nu_fixed,
                "forcing": forcing_amplitude,
                "E": E_mean.tolist(),
                "regime": regime,
                "rho1": float(rho1),
                "rho2": float(rho2),
                "total_energy": float(np.sum(E_mean)),
            }

        except Exception as e:
            print(f"\nWarning: Failed at Re={Re}: {e}")
            result = {
                "Re": float(Re),
                "nu": nu_fixed,
                "forcing": forcing_amplitude,
                "regime": "failed",
                "error": str(e),
            }

        results["sweep"].append(result)

    # Summary
    print(f"\n{'=' * 80}")
    print("REGIME CLASSIFICATION SUMMARY")
    print(f"{'=' * 80}")
    for regime, count in regime_counts.items():
        percentage = 100 * count / n_points
        print(f"  {regime:15s}: {count:3d} points ({percentage:5.1f}%)")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "reynolds_sweep_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Generate plots
    plot_phase_diagram(results, output_path)

    return results


def plot_phase_diagram(results, output_dir):
    """Generate phase diagram plots."""
    sweep = [s for s in results["sweep"] if s.get("regime") != "failed"]

    if not sweep:
        print("No valid results to plot")
        return

    Re_values = np.array([s["Re"] for s in sweep])
    rho1_values = np.array([s.get("rho1", 0) for s in sweep])
    rho2_values = np.array([s.get("rho2", 0) for s in sweep])
    regimes = [s["regime"] for s in sweep]

    # Color map for regimes
    regime_colors = {
        'laminar': 'blue',
        'epipelagic': 'green',
        'mesopelagic': 'orange',
        'bathypelagic': 'red',
    }

    colors = [regime_colors.get(r, 'gray') for r in regimes]

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Regime vs Re
    ax = axes[0, 0]
    for regime, color in regime_colors.items():
        mask = np.array([r == regime for r in regimes])
        if mask.any():
            ax.scatter(Re_values[mask], np.ones(mask.sum()),
                      c=color, label=regime, s=50, alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('Reynolds Number', fontsize=12)
    ax.set_ylabel('Regime', fontsize=12)
    ax.set_title('Flow Regime Classification', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)

    # Plot 2: Transfer ratios vs Re
    ax = axes[0, 1]
    ax.scatter(Re_values, rho1_values, c=colors, s=30, alpha=0.7, label='ρ₁ = T₀₁/E₀')
    ax.scatter(Re_values, rho2_values, c=colors, s=30, alpha=0.7, marker='^', label='ρ₂ = T₁₂/T₀₁')
    ax.axhline(0.05, color='k', linestyle='--', alpha=0.5, label='Laminar-Epi boundary')
    ax.axhline(0.30, color='k', linestyle=':', alpha=0.5, label='Epi-Meso boundary')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Reynolds Number', fontsize=12)
    ax.set_ylabel('Transfer Ratio', fontsize=12)
    ax.set_title('Transfer Ratios vs Reynolds Number', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Total energy vs Re
    ax = axes[1, 0]
    total_energies = [s.get("total_energy", 0) for s in sweep]
    ax.scatter(Re_values, total_energies, c=colors, s=50, alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Reynolds Number', fontsize=12)
    ax.set_ylabel('Total Energy', fontsize=12)
    ax.set_title('Energy vs Reynolds Number', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Phase space (ρ₁, ρ₂)
    ax = axes[1, 1]
    ax.scatter(rho1_values, rho2_values, c=colors, s=50, alpha=0.7)
    ax.axvline(0.05, color='k', linestyle='--', alpha=0.5)
    ax.axhline(0.30, color='k', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('ρ₁ = T₀₁/E₀', fontsize=12)
    ax.set_ylabel('ρ₂ = T₁₂/T₀₁', fontsize=12)
    ax.set_title('Phase Space Diagram', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add regime labels
    ax.text(0.01, 0.1, 'Laminar', fontsize=10, color='blue', alpha=0.7)
    ax.text(0.1, 0.1, 'Epipelagic', fontsize=10, color='green', alpha=0.7)
    ax.text(0.1, 0.5, 'Mesopelagic', fontsize=10, color='orange', alpha=0.7)

    plt.tight_layout()

    plot_file = Path(output_dir) / "reynolds_sweep_phase_diagram.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Phase diagram saved to: {plot_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Reynolds number parameter sweep")
    parser.add_argument('--re-min', type=float, default=100, help='Minimum Reynolds number')
    parser.add_argument('--re-max', type=float, default=1e6, help='Maximum Reynolds number')
    parser.add_argument('--n-points', type=int, default=50, help='Number of points')
    parser.add_argument('--shells', type=int, default=5, help='Number of shells')
    parser.add_argument('--nu', type=float, default=1e-3, help='Viscosity')
    parser.add_argument('--output', type=str, default='experiments/phase1_foundation')

    args = parser.parse_args()

    results = reynolds_sweep(
        Re_min=args.re_min,
        Re_max=args.re_max,
        n_points=args.n_points,
        n_shells=args.shells,
        nu_fixed=args.nu,
        output_dir=args.output,
    )

    if results is None:
        sys.exit(1)

    # Count epipelagic points
    epi_count = sum(1 for s in results["sweep"] if s.get("regime") == "epipelagic")
    if epi_count > 0:
        print(f"\n✓ Epipelagic regime identified: {epi_count} points")
        sys.exit(0)
    else:
        print(f"\n⚠ WARNING: No epipelagic regime found")
        sys.exit(1)


if __name__ == "__main__":
    main()
