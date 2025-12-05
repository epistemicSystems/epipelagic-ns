#!/usr/bin/env python3
"""
GPU Performance Benchmark for Taichi Cascade Solver

Validates >10^6 steps/sec target for Phase 1 requirements.

Usage:
    python scripts/benchmark_gpu.py
    python scripts/benchmark_gpu.py --shells 3,5,8,10 --steps 100000
"""

import argparse
import json
import sys
from pathlib import Path
from time import time
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from epipelagic.cascade.taichi_solver import TAICHI_AVAILABLE, benchmark_taichi_solver


def run_comprehensive_benchmark(
    shell_counts=[3, 5, 8, 10, 12],
    n_steps=100000,
    dt=1e-3,
    output_dir="experiments/phase1_foundation",
):
    """
    Run comprehensive GPU benchmarks.

    Parameters
    ----------
    shell_counts : list of int
        Shell counts to test
    n_steps : int
        Number of integration steps
    dt : float
        Timestep
    output_dir : str
        Output directory for results

    Returns
    -------
    results : dict
        Benchmark results
    """
    if not TAICHI_AVAILABLE:
        print("ERROR: Taichi not available. Cannot run GPU benchmarks.")
        print("Install with: pip install taichi")
        return None

    print("=" * 80)
    print("EPIPELAGIC TURBULENCE: GPU PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"\nTarget: >10^6 steps/sec")
    print(f"Testing shell counts: {shell_counts}")
    print(f"Steps per benchmark: {n_steps}")
    print(f"Timestep: {dt}")
    print()

    results = {
        "config": {
            "shell_counts": shell_counts,
            "n_steps": n_steps,
            "dt": dt,
            "target_steps_per_sec": 1e6,
        },
        "benchmarks": []
    }

    for n_shells in shell_counts:
        print(f"\n{'─' * 80}")
        print(f"Benchmarking n_shells={n_shells}")
        print(f"{'─' * 80}")

        stats = benchmark_taichi_solver(
            n_shells=n_shells,
            n_steps=n_steps,
            dt=dt,
        )

        results["benchmarks"].append(stats)

        # Print results
        print(f"  Wall time:        {stats['wall_time']:.3f} sec")
        print(f"  Steps/sec:        {stats['steps_per_sec']:.2e}")
        print(f"  Target met:       {'✓ YES' if stats['target_met'] else '✗ NO'}")
        print(f"  Final energy:     {stats['final_energy']:.6f}")

        if stats['target_met']:
            speedup = stats['steps_per_sec'] / 1e6
            print(f"  Speedup vs target: {speedup:.2f}x")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    passed = [s for s in results["benchmarks"] if s['target_met']]
    failed = [s for s in results["benchmarks"] if not s['target_met']]

    print(f"Tests passed: {len(passed)}/{len(results['benchmarks'])}")

    if passed:
        max_perf = max(s['steps_per_sec'] for s in passed)
        print(f"Best performance: {max_perf:.2e} steps/sec")

    if failed:
        print(f"\n⚠ WARNING: {len(failed)} tests failed to meet target")
        for s in failed:
            deficit = 1e6 - s['steps_per_sec']
            print(f"  - n_shells={s['n_shells']}: {s['steps_per_sec']:.2e} steps/sec ({deficit:.2e} below target)")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "gpu_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Generate plots
    plot_results(results, output_path)

    return results


def plot_results(results, output_dir):
    """Generate performance plots."""
    benchmarks = results["benchmarks"]

    shell_counts = [b['n_shells'] for b in benchmarks]
    steps_per_sec = [b['steps_per_sec'] for b in benchmarks]
    target = results["config"]["target_steps_per_sec"]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Performance vs shell count
    ax1.plot(shell_counts, steps_per_sec, 'o-', linewidth=2, markersize=8, label='Measured')
    ax1.axhline(target, color='r', linestyle='--', linewidth=2, label=f'Target ({target:.0e})')
    ax1.set_xlabel('Number of Shells', fontsize=12)
    ax1.set_ylabel('Performance (steps/sec)', fontsize=12)
    ax1.set_title('GPU Performance vs Shell Count', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_yscale('log')

    # Highlight passing/failing
    for i, b in enumerate(benchmarks):
        color = 'green' if b['target_met'] else 'red'
        ax1.plot(b['n_shells'], b['steps_per_sec'], 'o', color=color, markersize=10, alpha=0.5)

    # Plot 2: Speedup factor
    speedup = [s / target for s in steps_per_sec]
    colors = ['green' if s >= 1 else 'red' for s in speedup]

    ax2.bar(range(len(shell_counts)), speedup, color=colors, alpha=0.7)
    ax2.axhline(1.0, color='k', linestyle='--', linewidth=1.5, label='Target')
    ax2.set_xlabel('Shell Count', fontsize=12)
    ax2.set_ylabel('Speedup vs Target', fontsize=12)
    ax2.set_title('Performance Relative to Target', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(shell_counts)))
    ax2.set_xticklabels(shell_counts)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)

    plt.tight_layout()

    plot_file = Path(output_dir) / "gpu_benchmark_performance.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Performance plot saved to: {plot_file}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Taichi GPU cascade solver")
    parser.add_argument(
        '--shells',
        type=str,
        default='3,5,8,10,12',
        help='Comma-separated shell counts to test (default: 3,5,8,10,12)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=100000,
        help='Number of steps per benchmark (default: 100000)'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=1e-3,
        help='Timestep (default: 1e-3)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='experiments/phase1_foundation',
        help='Output directory (default: experiments/phase1_foundation)'
    )

    args = parser.parse_args()

    # Parse shell counts
    shell_counts = [int(x) for x in args.shells.split(',')]

    # Run benchmark
    results = run_comprehensive_benchmark(
        shell_counts=shell_counts,
        n_steps=args.steps,
        dt=args.dt,
        output_dir=args.output,
    )

    if results is None:
        sys.exit(1)

    # Exit with error if any tests failed
    failed = [b for b in results["benchmarks"] if not b['target_met']]
    if failed:
        print(f"\n⚠ Benchmark FAILED: {len(failed)} test(s) below target")
        sys.exit(1)
    else:
        print(f"\n✓ All benchmarks PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
