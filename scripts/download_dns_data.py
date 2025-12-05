#!/usr/bin/env python
"""
Download and setup DNS data for Phase 2.

This script implements Task 2.1 from TASKS.md:
- Downloads or generates DNS turbulence data
- Validates data integrity
- Prepares for topology extraction

Usage:
------
# Generate synthetic test data
python scripts/download_dns_data.py --synthetic --resolution 256 --reynolds 1000

# Get JHTDB download instructions
python scripts/download_dns_data.py --dataset isotropic1024coarse --method manual

# Validate existing data
python scripts/download_dns_data.py --validate --input data/dns/test.h5
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from epipelagic.data import (
    create_sample_jhtdb_file,
    download_jhtdb_data,
    validate_dns_data,
    DNSProcessor,
)


def main():
    parser = argparse.ArgumentParser(
        description="Download and setup DNS turbulence data for Phase 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data source options
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate synthetic turbulence data'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='isotropic1024coarse',
        help='JHTDB dataset name (default: isotropic1024coarse)'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='manual',
        choices=['manual', 'curl', 'wget'],
        help='Download method (default: manual)'
    )

    # Data parameters
    parser.add_argument(
        '--resolution',
        type=int,
        default=256,
        help='Grid resolution (default: 256 for 256³)'
    )

    parser.add_argument(
        '--reynolds',
        type=float,
        default=1000,
        help='Reynolds number for synthetic data (default: 1000)'
    )

    parser.add_argument(
        '--time',
        type=float,
        default=0.364,
        help='Time snapshot for JHTDB data (default: 0.364)'
    )

    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for data file'
    )

    # Validation
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate existing data file'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Input file to validate'
    )

    # Processing
    parser.add_argument(
        '--process',
        action='store_true',
        help='Process data (compute vorticity, level sets)'
    )

    args = parser.parse_args()

    # Default output paths
    if args.output is None:
        if args.synthetic:
            args.output = f'data/dns/synthetic_Re{int(args.reynolds)}_{args.resolution}cubed.h5'
        else:
            args.output = f'data/dns/jhtdb_iso1024/velocity_{args.resolution}.h5'

    print("=" * 80)
    print("DNS Data Setup - Phase 2 Task 2.1")
    print("=" * 80)

    # Validate existing data
    if args.validate:
        if args.input is None:
            print("❌ Error: --input required for validation")
            sys.exit(1)

        print(f"\n📊 Validating: {args.input}")
        print("-" * 80)

        try:
            results = validate_dns_data(args.input)

            print("\n✓ Validation Results:")
            print(f"  Incompressibility: max |∇·u| = {results['incompressibility']:.3e}")
            print(f"  Energy spectrum slope: {results['energy_spectrum_slope']:.3f}")
            print(f"  Vorticity range: [{results['vorticity_min']:.3e}, {results['vorticity_max']:.3e}]")

            # Pass/fail criteria
            incomp_pass = results['incompressibility'] < 1e-4
            slope_pass = -2.0 < results['energy_spectrum_slope'] < -1.5

            if incomp_pass and slope_pass:
                print("\n✅ Data validation PASSED")
            else:
                print("\n⚠️  Data validation UNCERTAIN (see criteria above)")

        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            sys.exit(1)

        return

    # Generate synthetic data
    if args.synthetic:
        print(f"\n🔬 Generating synthetic turbulence")
        print(f"   Resolution: {args.resolution}³")
        print(f"   Reynolds number: {args.reynolds}")
        print(f"   Output: {args.output}")
        print("-" * 80)

        try:
            output_path = create_sample_jhtdb_file(
                args.output,
                resolution=(args.resolution,) * 3,
                reynolds_number=args.reynolds,
            )

            print(f"\n✅ Synthetic data created successfully!")
            print(f"\nNext steps:")
            print(f"  1. Validate: python {__file__} --validate --input {output_path}")
            print(f"  2. Process: python {__file__} --process --input {output_path}")
            print(f"  3. Extract topology: python scripts/topology_extraction.py --input {output_path}")

        except Exception as e:
            print(f"\n❌ Error generating synthetic data: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Download JHTDB data
    else:
        print(f"\n🌐 JHTDB Data Download")
        print(f"   Dataset: {args.dataset}")
        print(f"   Time: {args.time}")
        print(f"   Resolution: {args.resolution}³")
        print(f"   Method: {args.method}")
        print(f"   Output: {args.output}")
        print("-" * 80)

        try:
            instructions = download_jhtdb_data(
                args.output,
                dataset=args.dataset,
                time=args.time,
                resolution=(args.resolution,) * 3,
                method=args.method,
            )

            if args.method == 'manual':
                print("\n⚠️  Manual download required. Follow instructions above.")
                print("\n💡 TIP: For immediate testing, use synthetic data:")
                print(f"   python {__file__} --synthetic --resolution 128 --reynolds 1000")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)

    # Process data
    if args.process:
        input_file = args.input if args.input else args.output

        print(f"\n⚙️  Processing DNS data: {input_file}")
        print("-" * 80)

        try:
            processor = DNSProcessor(input_file, lazy_load=False)

            print("\n1. Loading velocity field...")
            velocity = processor.load_velocity()
            print(f"   ✓ Shape: {velocity.shape}")

            print("\n2. Computing vorticity...")
            vorticity = processor.compute_vorticity(method='spectral')
            print(f"   ✓ Range: [{vorticity.min():.3e}, {vorticity.max():.3e}]")

            print("\n3. Extracting level sets...")
            level_sets = processor.extract_level_sets(vorticity, n_levels=50)
            print(f"   ✓ Number of levels: {len(level_sets)}")

            print("\n4. Building filtration...")
            filtration = processor.build_filtration(level_sets, max_points=5000)
            print(f"   ✓ Filtration ready for persistent homology")

            # Save processed data
            output_processed = input_file.replace('.h5', '_processed.h5')
            if output_processed == input_file:
                output_processed = input_file.replace('.h5', '') + '_processed.h5'

            print(f"\n5. Saving processed data...")
            processor.save_processed_data(
                output_processed,
                vorticity=vorticity,
                filtration=filtration,
            )

            print(f"\n✅ Processing complete!")
            print(f"\nNext step:")
            print(f"  Extract topology: python scripts/topology_extraction.py --input {input_file}")

        except Exception as e:
            print(f"\n❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
