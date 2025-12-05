#!/usr/bin/env python
"""
Compare cascade cohomology with persistent homology (Task 2.4).

This script demonstrates the correspondence between:
1. Cascade complex cohomology: dim(H¹_cascade)
2. Persistent homology from vorticity: dim(H¹_persistent)

Theoretical prediction (from CLAUDE.md):
    dim(H¹_epi) ≤ C log(Re)

This validates the epipelagic principle: persistent cross-scale
structures captured by both methods should agree.

Usage:
------
# Basic comparison on synthetic data
python scripts/compare_cohomology.py --reynolds 1000

# Compare across Reynolds sweep
python scripts/compare_cohomology.py --sweep --re-min 100 --re-max 5000

# Use real DNS data
python scripts/compare_cohomology.py --input data/dns/velocity.h5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from epipelagic.data import generate_synthetic_turbulence, DNSProcessor
from epipelagic.core.complex import CascadeComplex
from epipelagic.core.transfer_matrix import (
    compute_transfer_matrix_spectral,
    validate_transfer_conservation,
)
from epipelagic.topology import (
    extract_persistent_homology,
    extract_epipelagic_features,
    compute_cubical_complex_persistence,
    persistence_to_diagrams,
)


def compare_single_field(
    velocity: np.ndarray,
    reynolds_number: float,
    n_shells: int = 6,
    persistence_threshold: float = 0.5,
) -> dict:
    """
    Compare cohomology dimensions for a single velocity field.

    Parameters
    ----------
    velocity : ndarray
        Velocity field
    reynolds_number : float
        Reynolds number
    n_shells : int
        Number of cascade shells
    persistence_threshold : float
        Threshold for epipelagic features

    Returns
    -------
    results : dict
        Comparison results
    """
    print(f"\n{'='*80}")
    print(f"Reynolds Number: Re = {reynolds_number:.1f}")
    print(f"{'='*80}")

    # 1. CASCADE COHOMOLOGY (from transfer matrix)
    print("\n1. Computing cascade complex cohomology...")

    # Option A: Phenomenological (fast)
    cascade_phenom = CascadeComplex.from_velocity_field(
        velocity,
        n_shells=n_shells,
        method="phenomenological"
    )
    _, _, dim_H1_phenom = cascade_phenom.compute_cohomology()
    regime_phenom = cascade_phenom.classify_regime()

    print(f"   Phenomenological model:")
    print(f"     Regime: {regime_phenom}")
    print(f"     dim(H¹_cascade) = {dim_H1_phenom}")

    # Option B: Direct measurement (slow, accurate)
    print("\n   Computing direct transfer matrix...")
    cascade_spectral = CascadeComplex.from_velocity_field(
        velocity,
        n_shells=n_shells,
        method="spectral"
    )
    _, _, dim_H1_spectral = cascade_spectral.compute_cohomology()
    regime_spectral = cascade_spectral.classify_regime()

    print(f"   Spectral (direct) method:")
    print(f"     Regime: {regime_spectral}")
    print(f"     dim(H¹_cascade) = {dim_H1_spectral}")

    # Validate transfer conservation
    transfer_matrix = compute_transfer_matrix_spectral(velocity, n_shells=n_shells)
    wavenumbers = 2.0 ** np.arange(n_shells)
    validation = validate_transfer_conservation(
        transfer_matrix,
        viscosity=1.0 / reynolds_number,
        wavenumbers=wavenumbers
    )
    print(f"\n   Transfer validation:")
    print(f"     Antisymmetric: {validation['antisymmetric']}")
    print(f"     Conservative: {validation['conservative']}")
    print(f"     Forward cascade: {validation['cascade_direction']}")

    # 2. PERSISTENT HOMOLOGY (from vorticity)
    print("\n2. Computing persistent homology from vorticity...")

    # Compute vorticity
    processor = DNSProcessor("", lazy_load=False)
    processor._velocity = velocity
    vorticity = processor.compute_vorticity(method='spectral')

    print(f"   Vorticity range: [{vorticity.min():.3e}, {vorticity.max():.3e}]")

    # Method A: Ripser (fast, approximate)
    print("\n   Using Ripser (Vietoris-Rips)...")
    result_ripser = extract_persistent_homology(
        velocity,
        threshold=persistence_threshold,
        n_points=min(5000, velocity.size // 3)
    )
    dim_H1_ripser = result_ripser['dim_H1_epi']

    print(f"     dim(H¹_persistent) = {dim_H1_ripser}")
    print(f"     Total H¹ features: {len(result_ripser['dgms'][1])}")

    # Method B: Gudhi Cubical (optimal for voxel data)
    print("\n   Using Gudhi (Cubical complex)...")
    try:
        result_gudhi = compute_cubical_complex_persistence(
            vorticity,
            periodic=True
        )
        dgms_gudhi = persistence_to_diagrams(result_gudhi['persistence'])

        # Extract epipelagic features
        features_gudhi = extract_epipelagic_features(dgms_gudhi, threshold=persistence_threshold)
        dim_H1_gudhi = features_gudhi['dim_H1_epi']

        print(f"     dim(H¹_persistent) = {dim_H1_gudhi}")
        print(f"     Total H¹ features: {len(dgms_gudhi[1])}")

    except Exception as e:
        print(f"     Gudhi computation failed: {e}")
        dim_H1_gudhi = None

    # 3. COMPARISON
    print(f"\n3. Comparison:")
    print(f"   {'Method':<30} {'dim(H¹)':<10}")
    print(f"   {'-'*40}")
    print(f"   {'Cascade (phenomenological)':<30} {dim_H1_phenom:<10}")
    print(f"   {'Cascade (spectral)':<30} {dim_H1_spectral:<10}")
    print(f"   {'Persistent (Ripser)':<30} {dim_H1_ripser:<10}")
    if dim_H1_gudhi is not None:
        print(f"   {'Persistent (Gudhi cubical)':<30} {dim_H1_gudhi:<10}")

    # Theoretical bound: dim(H¹_epi) ≤ C log(Re)
    if reynolds_number > 1:
        C = 2.5  # Empirical constant
        theoretical_bound = C * np.log(reynolds_number)
        print(f"\n   Theoretical bound (C log Re): {theoretical_bound:.2f}")
        print(f"   C = {C:.2f}")

        # Check if bounds satisfied
        satisfies = []
        for name, val in [
            ("Cascade (spectral)", dim_H1_spectral),
            ("Persistent (Ripser)", dim_H1_ripser),
            ("Persistent (Gudhi)", dim_H1_gudhi),
        ]:
            if val is not None:
                satisfies.append(val <= theoretical_bound)
                status = "✓" if val <= theoretical_bound else "✗"
                print(f"     {status} {name}: {val} ≤ {theoretical_bound:.2f}")

    # Return results
    return {
        'reynolds_number': reynolds_number,
        'dim_H1_cascade_phenom': dim_H1_phenom,
        'dim_H1_cascade_spectral': dim_H1_spectral,
        'dim_H1_persistent_ripser': dim_H1_ripser,
        'dim_H1_persistent_gudhi': dim_H1_gudhi,
        'regime': regime_spectral,
        'transfer_validation': validation,
    }


def reynolds_sweep(
    re_values: np.ndarray,
    n_shells: int = 6,
    resolution: tuple = (64, 64, 64),
) -> list:
    """
    Compare cohomology across Reynolds number sweep.

    Parameters
    ----------
    re_values : ndarray
        Reynolds numbers to test
    n_shells : int
        Number of cascade shells
    resolution : tuple
        Grid resolution

    Returns
    -------
    results : list of dict
        Results for each Re
    """
    results = []

    print(f"\n{'='*80}")
    print(f"REYNOLDS NUMBER SWEEP")
    print(f"{'='*80}")
    print(f"Testing Re ∈ [{re_values.min():.0f}, {re_values.max():.0f}]")
    print(f"Grid resolution: {resolution[0]}³")
    print(f"Number of shells: {n_shells}")

    for Re in re_values:
        print(f"\n{'-'*80}")
        print(f"Generating turbulence at Re = {Re:.1f}...")

        # Generate synthetic turbulence
        velocity = generate_synthetic_turbulence(
            resolution=resolution,
            reynolds_number=Re,
        )

        # Compare cohomology
        result = compare_single_field(velocity, Re, n_shells=n_shells)
        results.append(result)

    return results


def plot_comparison(results: list, output_path: str = "cohomology_comparison.png"):
    """
    Plot comparison of cohomology dimensions vs Reynolds number.

    Parameters
    ----------
    results : list of dict
        Results from reynolds_sweep()
    output_path : str
        Output file path
    """
    # Extract data
    Re = np.array([r['reynolds_number'] for r in results])
    dim_cascade = np.array([r['dim_H1_cascade_spectral'] for r in results])
    dim_persistent = np.array([r['dim_H1_persistent_ripser'] for r in results])

    # Theoretical bound
    C = 2.5
    Re_theory = np.linspace(Re.min(), Re.max(), 100)
    bound_theory = C * np.log(Re_theory)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(Re, dim_cascade, 'o-', label='Cascade cohomology (spectral)', linewidth=2)
    ax.plot(Re, dim_persistent, 's-', label='Persistent homology (Ripser)', linewidth=2)
    ax.plot(Re_theory, bound_theory, 'k--', label=f'Bound: {C:.1f} log(Re)', linewidth=2)

    ax.set_xlabel('Reynolds number Re', fontsize=12)
    ax.set_ylabel('dim(H¹)', fontsize=12)
    ax.set_title('Epipelagic Cohomology: Cascade vs Persistent Homology', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n✓ Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare cascade and persistent cohomology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--reynolds', type=float, default=1000,
                        help='Reynolds number for single comparison (default: 1000)')
    parser.add_argument('--sweep', action='store_true',
                        help='Perform Reynolds number sweep')
    parser.add_argument('--re-min', type=float, default=100,
                        help='Minimum Reynolds number for sweep')
    parser.add_argument('--re-max', type=float, default=5000,
                        help='Maximum Reynolds number for sweep')
    parser.add_argument('--n-points', type=int, default=10,
                        help='Number of points in sweep')
    parser.add_argument('--n-shells', type=int, default=6,
                        help='Number of cascade shells')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Grid resolution (NxNxN)')
    parser.add_argument('--input', type=str, default=None,
                        help='Input DNS data file (HDF5)')
    parser.add_argument('--output', type=str, default='cohomology_comparison.png',
                        help='Output plot file')

    args = parser.parse_args()

    if args.sweep:
        # Reynolds sweep
        re_values = np.logspace(
            np.log10(args.re_min),
            np.log10(args.re_max),
            args.n_points
        )
        results = reynolds_sweep(
            re_values,
            n_shells=args.n_shells,
            resolution=(args.resolution,) * 3
        )

        # Plot results
        plot_comparison(results, output_path=args.output)

    else:
        # Single comparison
        if args.input:
            # Load DNS data
            print(f"Loading DNS data from {args.input}...")
            processor = DNSProcessor(args.input, lazy_load=False)
            velocity = processor.load_velocity()
            re = args.reynolds
        else:
            # Generate synthetic data
            print(f"Generating synthetic turbulence (Re = {args.reynolds})...")
            velocity = generate_synthetic_turbulence(
                resolution=(args.resolution,) * 3,
                reynolds_number=args.reynolds,
            )
            re = args.reynolds

        # Compare
        result = compare_single_field(
            velocity,
            re,
            n_shells=args.n_shells
        )

    print(f"\n{'='*80}")
    print("Comparison complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
