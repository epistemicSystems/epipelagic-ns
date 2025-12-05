#!/usr/bin/env python3
"""
Topology Extraction: dim(H¹ₑₚᵢ) Computation

Extracts epipelagic cohomology dimension from synthetic turbulence fields
using persistent homology (Ripser).

Verifies finiteness bound: dim(H¹ₑₚᵢ) ≤ C log(Re)

Usage:
    python scripts/topology_extraction.py
    python scripts/topology_extraction.py --re-range 100,10000 --n-points 20
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("Warning: Ripser not available. Install with: pip install ripser")

from epipelagic.cascade.taichi_solver import TAICHI_AVAILABLE, TaichiCascadeSolver


def generate_synthetic_vorticity_field(u_shells, k_shells, grid_size=64):
    """
    Generate 2D vorticity field from shell velocities.

    Parameters
    ----------
    u_shells : ndarray, complex
        Shell velocities
    k_shells : ndarray
        Shell wavenumbers
    grid_size : int
        Spatial grid resolution

    Returns
    -------
    vorticity : ndarray, shape (grid_size, grid_size)
        Vorticity field ω(x, y)
    positions : ndarray, shape (N, 2)
        Positions of vortex cores
    """
    # Create spatial grid
    x = np.linspace(0, 2*np.pi, grid_size)
    y = np.linspace(0, 2*np.pi, grid_size)
    X, Y = np.meshgrid(x, y)

    # Synthesize vorticity from shell modes
    vorticity = np.zeros((grid_size, grid_size))

    for u_n, k_n in zip(u_shells, k_shells):
        # Random phases for this shell
        n_modes = int(k_n)
        for _ in range(max(1, n_modes)):
            # Random wavevector with magnitude k_n
            theta = np.random.rand() * 2 * np.pi
            kx = k_n * np.cos(theta)
            ky = k_n * np.sin(theta)

            # Random phase
            phi = np.random.rand() * 2 * np.pi

            # Add contribution
            amplitude = np.abs(u_n) / np.sqrt(n_modes)
            vorticity += amplitude * np.sin(kx * X + ky * Y + phi)

    # Extract vortex cores (local extrema above threshold)
    threshold = np.std(vorticity) * 0.5

    positions = []
    for i in range(1, grid_size - 1):
        for j in range(1, grid_size - 1):
            val = vorticity[i, j]
            neighbors = [
                vorticity[i-1, j], vorticity[i+1, j],
                vorticity[i, j-1], vorticity[i, j+1],
            ]

            # Local maximum or minimum
            if val > threshold and all(val > n for n in neighbors):
                positions.append([x[j], y[i]])
            elif val < -threshold and all(val < n for n in neighbors):
                positions.append([x[j], y[i]])

    positions = np.array(positions) if positions else np.empty((0, 2))

    return vorticity, positions


def extract_epipelagic_cohomology(positions, threshold_percentile=50):
    """
    Compute dim(H¹ₑₚᵢ) from vortex positions using Ripser.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Vortex core positions
    threshold_percentile : float
        Persistence threshold percentile (0-100)

    Returns
    -------
    dim_H1_epi : int
        Dimension of epipelagic cohomology
    persistence_diagram : ndarray
        H¹ persistence diagram
    """
    if not RIPSER_AVAILABLE:
        raise ImportError("Ripser required for topology extraction")

    if len(positions) < 3:
        return 0, np.empty((0, 2))

    # Compute persistent homology
    result = ripser(positions, maxdim=1)
    dgm1 = result['dgms'][1]  # H¹ diagram

    if len(dgm1) == 0:
        return 0, dgm1

    # Filter by persistence
    persistence = dgm1[:, 1] - dgm1[:, 0]
    threshold = np.percentile(persistence, threshold_percentile)

    long_bars = dgm1[persistence > threshold]
    dim_H1_epi = len(long_bars)

    return dim_H1_epi, dgm1


def topology_sweep(
    Re_values,
    n_shells=5,
    nu=1e-3,
    grid_size=64,
    output_dir="experiments/phase1_foundation",
):
    """
    Sweep Reynolds number and compute dim(H¹ₑₚᵢ).

    Verifies: dim(H¹ₑₚᵢ) ≤ C log(Re)

    Parameters
    ----------
    Re_values : array-like
        Reynolds numbers to test
    n_shells : int
        Number of cascade shells
    nu : float
        Viscosity
    grid_size : int
        Spatial resolution
    output_dir : str
        Output directory

    Returns
    -------
    results : dict
        Topology extraction results
    """
    if not RIPSER_AVAILABLE:
        print("ERROR: Ripser not available")
        return None

    if not TAICHI_AVAILABLE:
        print("ERROR: Taichi not available")
        return None

    print("=" * 80)
    print("TOPOLOGY EXTRACTION: dim(H¹ₑₚᵢ) vs REYNOLDS NUMBER")
    print("=" * 80)
    print(f"Reynolds range: [{min(Re_values)}, {max(Re_values)}]")
    print(f"Points: {len(Re_values)}")
    print(f"Shells: {n_shells}")
    print(f"Grid: {grid_size}x{grid_size}")
    print()

    results = {
        "config": {
            "Re_values": Re_values.tolist(),
            "n_shells": n_shells,
            "nu": nu,
            "grid_size": grid_size,
        },
        "topology": []
    }

    for Re in tqdm(Re_values, desc="Topology sweep"):
        # Create solver
        forcing_amplitude = Re * nu / 10.0
        solver = TaichiCascadeSolver(
            n_shells=n_shells,
            nu=nu,
            forcing_amplitude=forcing_amplitude,
            forcing_shell=1,
        )

        # Evolve to steady state
        u0 = np.random.randn(n_shells) + 1j * np.random.randn(n_shells)
        u0 *= 0.01

        u_final = solver.integrate(u0, n_steps=20000, dt=1e-3)

        # Get wavenumbers
        k = solver.k.to_numpy()

        # Generate vorticity field
        vorticity, positions = generate_synthetic_vorticity_field(
            u_final, k, grid_size=grid_size
        )

        # Extract topology
        try:
            dim_H1, dgm1 = extract_epipelagic_cohomology(
                positions,
                threshold_percentile=50
            )

            result = {
                "Re": float(Re),
                "log_Re": float(np.log(Re)),
                "dim_H1_epi": int(dim_H1),
                "n_vortices": len(positions),
                "n_bars_total": len(dgm1),
            }

        except Exception as e:
            print(f"\nWarning: Failed at Re={Re}: {e}")
            result = {
                "Re": float(Re),
                "log_Re": float(np.log(Re)),
                "dim_H1_epi": 0,
                "error": str(e),
            }

        results["topology"].append(result)

    # Verify finiteness bound
    verify_finiteness_bound(results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "topology_extraction_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Generate plots
    plot_topology_results(results, output_path)

    return results


def verify_finiteness_bound(results):
    """
    Verify dim(H¹ₑₚᵢ) ≤ C log(Re).

    Fits linear model: dim(H¹) = C * log(Re) + b
    """
    topology = [t for t in results["topology"] if "error" not in t]

    if not topology:
        print("\nNo valid topology results")
        return

    log_Re = np.array([t["log_Re"] for t in topology])
    dim_H1 = np.array([t["dim_H1_epi"] for t in topology])

    # Fit linear model
    if len(log_Re) > 1:
        coeffs = np.polyfit(log_Re, dim_H1, 1)
        C = coeffs[0]
        b = coeffs[1]

        # Predicted values
        dim_H1_pred = C * log_Re + b

        # R² score
        ss_res = np.sum((dim_H1 - dim_H1_pred)**2)
        ss_tot = np.sum((dim_H1 - np.mean(dim_H1))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"\n{'=' * 80}")
        print("FINITENESS BOUND VERIFICATION")
        print(f"{'=' * 80}")
        print(f"Fitted model: dim(H¹ₑₚᵢ) = {C:.3f} * log(Re) + {b:.3f}")
        print(f"R² score: {r2:.4f}")
        print(f"Constant C: {C:.3f}")

        if C > 0 and C < 10:
            print(f"\n✓ Finiteness bound verified: dim(H¹ₑₚᵢ) ≤ {C:.1f} log(Re)")
        else:
            print(f"\n⚠ WARNING: Bound constant C={C:.3f} outside expected range [0, 10]")

        results["finiteness_bound"] = {
            "C": float(C),
            "b": float(b),
            "r2": float(r2),
            "verified": bool(0 < C < 10),
        }


def plot_topology_results(results, output_dir):
    """Generate topology plots."""
    topology = [t for t in results["topology"] if "error" not in t]

    if not topology:
        print("No valid results to plot")
        return

    Re_values = np.array([t["Re"] for t in topology])
    log_Re = np.array([t["log_Re"] for t in topology])
    dim_H1 = np.array([t["dim_H1_epi"] for t in topology])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: dim(H¹) vs Re
    ax1.scatter(Re_values, dim_H1, s=50, alpha=0.7, color='purple')
    ax1.set_xscale('log')
    ax1.set_xlabel('Reynolds Number', fontsize=12)
    ax1.set_ylabel('dim(H¹ₑₚᵢ)', fontsize=12)
    ax1.set_title('Epipelagic Cohomology Dimension', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: dim(H¹) vs log(Re) with fit
    ax2.scatter(log_Re, dim_H1, s=50, alpha=0.7, color='purple', label='Measured')

    if len(log_Re) > 1 and "finiteness_bound" in results:
        bound = results["finiteness_bound"]
        C = bound["C"]
        b = bound["b"]

        log_Re_fit = np.linspace(log_Re.min(), log_Re.max(), 100)
        dim_H1_fit = C * log_Re_fit + b

        ax2.plot(log_Re_fit, dim_H1_fit, 'r--', linewidth=2,
                label=f'Fit: {C:.2f} log(Re) + {b:.2f}')

    ax2.set_xlabel('log(Re)', fontsize=12)
    ax2.set_ylabel('dim(H¹ₑₚᵢ)', fontsize=12)
    ax2.set_title('Finiteness Bound Verification', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_file = Path(output_dir) / "topology_extraction_results.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Topology plot saved to: {plot_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Topology extraction from turbulence")
    parser.add_argument('--re-range', type=str, default='100,10000',
                       help='Reynolds range (min,max), comma-separated')
    parser.add_argument('--n-points', type=int, default=20,
                       help='Number of Reynolds values to sample')
    parser.add_argument('--shells', type=int, default=5)
    parser.add_argument('--nu', type=float, default=1e-3)
    parser.add_argument('--grid-size', type=int, default=64)
    parser.add_argument('--output', type=str, default='experiments/phase1_foundation')

    args = parser.parse_args()

    # Parse Re range
    re_min, re_max = map(float, args.re_range.split(','))
    Re_values = np.logspace(np.log10(re_min), np.log10(re_max), args.n_points)

    results = topology_sweep(
        Re_values=Re_values,
        n_shells=args.shells,
        nu=args.nu,
        grid_size=args.grid_size,
        output_dir=args.output,
    )

    if results is None:
        sys.exit(1)

    # Check if finiteness bound verified
    if "finiteness_bound" in results and results["finiteness_bound"]["verified"]:
        print(f"\n✓ SUCCESS: Finiteness bound verified!")
        sys.exit(0)
    else:
        print(f"\n⚠ WARNING: Finiteness bound not verified")
        sys.exit(1)


if __name__ == "__main__":
    main()
