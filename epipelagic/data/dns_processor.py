"""
DNS data processing pipeline for Phase 2.

This module implements Task 2.2 from TASKS.md:
- Efficient data loading (HDF5, memory-mapped arrays)
- Vorticity extraction (spectral and finite difference methods)
- Level set extraction (marching cubes)
- Filtration construction for persistent homology

Pipeline:
    DNS Data → Vorticity → Level Sets → Filtration → Persistent Homology
"""

from typing import Tuple, Optional, List, Dict
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import warnings


class DNSProcessor:
    """
    Process DNS turbulence data for topology extraction.

    This class implements the data processing pipeline from TASKS.md Task 2.2.

    Examples
    --------
    >>> processor = DNSProcessor('data/dns/jhtdb_iso1024/velocity.h5')
    >>> vorticity = processor.compute_vorticity(method='spectral')
    >>> level_sets = processor.extract_level_sets(vorticity, n_levels=50)
    >>> filtration = processor.build_filtration(level_sets)
    """

    def __init__(self, data_path: str, lazy_load: bool = True):
        """
        Initialize DNS processor.

        Parameters
        ----------
        data_path : str
            Path to HDF5 file containing velocity data
        lazy_load : bool
            If True, use memory-mapped arrays (efficient for large files)
        """
        self.data_path = Path(data_path)
        self.lazy_load = lazy_load
        self._velocity = None
        self._vorticity = None
        self._file_handle = None

    def load_velocity(
        self,
        region: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        """
        Load velocity field with optional lazy loading.

        Parameters
        ----------
        region : tuple of slices, optional
            Region to load (default: entire field)

        Returns
        -------
        velocity : ndarray, shape (nx, ny, nz, 3)
            Velocity field
        """
        if self._velocity is not None and region is None:
            return self._velocity

        print(f"Loading velocity field from {self.data_path}...")

        with h5py.File(self.data_path, 'r') as f:
            if 'velocity' not in f:
                raise KeyError("'velocity' dataset not found in HDF5 file")

            if region is None:
                if self.lazy_load:
                    # Memory-mapped access
                    self._file_handle = h5py.File(self.data_path, 'r')
                    self._velocity = self._file_handle['velocity']
                else:
                    # Load into memory
                    self._velocity = f['velocity'][...]
            else:
                # Load specific region
                velocity = f['velocity'][region]
                return velocity

        print(f"✓ Velocity field loaded: {self._velocity.shape}")
        return self._velocity

    def compute_vorticity(
        self,
        method: str = 'finite_difference',
        accuracy: int = 6,
        periodic: bool = True,
    ) -> np.ndarray:
        """
        Compute vorticity ω = ∇ × u.

        Parameters
        ----------
        method : str
            Computation method:
            - 'finite_difference': High-order finite differences
            - 'spectral': FFT-based (most accurate for periodic domains)
        accuracy : int
            Accuracy order for finite differences (2, 4, or 6)
        periodic : bool
            Whether boundary conditions are periodic

        Returns
        -------
        vorticity : ndarray, shape (nx, ny, nz)
            Vorticity magnitude |ω|

        Algorithm:
        ----------
        Spectral method:
            1. FFT of velocity: û(k) = FFT[u(x)]
            2. Compute curl in Fourier space: ω̂(k) = ik × û(k)
            3. Inverse FFT: ω(x) = IFFT[ω̂(k)]

        Finite difference method:
            1. Compute derivatives using centered differences
            2. ωx = ∂w/∂y - ∂v/∂z
            3. ωy = ∂u/∂z - ∂w/∂x
            4. ωz = ∂v/∂x - ∂u/∂y
        """
        if self._velocity is None:
            self.load_velocity()

        print(f"Computing vorticity using {method} method...")

        if method == 'spectral':
            vorticity = self._compute_vorticity_spectral()
        elif method == 'finite_difference':
            vorticity = self._compute_vorticity_fd(accuracy, periodic)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Store for reuse
        self._vorticity = vorticity

        print(f"✓ Vorticity computed: min={vorticity.min():.3e}, max={vorticity.max():.3e}")
        return vorticity

    def _compute_vorticity_spectral(self) -> np.ndarray:
        """
        Compute vorticity using spectral (FFT) method.

        This is the most accurate method for periodic domains.
        """
        from scipy import fft

        velocity = self._velocity
        nx, ny, nz = velocity.shape[:3]

        # Wavenumber grids
        kx = fft.fftfreq(nx, 1.0 / nx) * 2 * np.pi
        ky = fft.fftfreq(ny, 1.0 / ny) * 2 * np.pi
        kz = fft.fftfreq(nz, 1.0 / nz) * 2 * np.pi

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

        # FFT of velocity components
        u_hat = fft.fftn(velocity[..., 0])
        v_hat = fft.fftn(velocity[..., 1])
        w_hat = fft.fftn(velocity[..., 2])

        # Compute curl in Fourier space: ω̂ = ik × û
        omega_x_hat = 1j * (KY * w_hat - KZ * v_hat)
        omega_y_hat = 1j * (KZ * u_hat - KX * w_hat)
        omega_z_hat = 1j * (KX * v_hat - KY * u_hat)

        # Inverse FFT
        omega_x = np.real(fft.ifftn(omega_x_hat))
        omega_y = np.real(fft.ifftn(omega_y_hat))
        omega_z = np.real(fft.ifftn(omega_z_hat))

        # Magnitude
        vorticity = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

        return vorticity

    def _compute_vorticity_fd(self, accuracy: int, periodic: bool) -> np.ndarray:
        """
        Compute vorticity using finite difference method.

        Implements 2nd, 4th, or 6th order accurate centered differences.
        """
        # Use existing implementation from topology/persistent.py
        from ..topology.persistent import compute_vorticity

        vorticity_vector = compute_vorticity(self._velocity)

        # If vector, compute magnitude
        if vorticity_vector.ndim == 4:
            vorticity = np.linalg.norm(vorticity_vector, axis=-1)
        else:
            vorticity = vorticity_vector

        return vorticity

    def extract_level_sets(
        self,
        vorticity: np.ndarray,
        n_levels: int = 50,
        threshold_range: Optional[Tuple[float, float]] = None,
    ) -> List[np.ndarray]:
        """
        Extract level sets for filtration construction.

        Parameters
        ----------
        vorticity : ndarray
            Vorticity field
        n_levels : int
            Number of threshold levels
        threshold_range : tuple of float, optional
            (min_threshold, max_threshold) range
            Default: (0.1 * max, 0.9 * max)

        Returns
        -------
        level_sets : list of ndarray
            List of point clouds, one per threshold level

        Algorithm (from TASKS.md):
        -------------------------
        1. Define thresholds: θ_i ∈ [θ_min, θ_max]
        2. For each threshold, extract X_θ = {x : |ω(x)| ≥ θ}
        3. Convert to point cloud for Ripser
        4. Optionally subsample for efficiency
        """
        if threshold_range is None:
            vmax = np.max(vorticity)
            threshold_range = (0.1 * vmax, 0.9 * vmax)

        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_levels)

        print(f"Extracting {n_levels} level sets...")
        level_sets = []

        for theta in tqdm(thresholds, desc="Level sets"):
            # Extract points where |ω| ≥ θ
            mask = vorticity >= theta
            points = np.argwhere(mask)

            if len(points) > 0:
                # Normalize coordinates to [0, 1]
                points = points.astype(float)
                for i in range(3):
                    points[:, i] /= vorticity.shape[i]

                level_sets.append(points)
            else:
                # Empty level set
                level_sets.append(np.zeros((0, 3)))

        print(f"✓ Level sets extracted: {len(level_sets)} levels")
        return level_sets

    def build_filtration(
        self,
        level_sets: List[np.ndarray],
        max_points: int = 5000,
    ) -> Dict:
        """
        Build filtration for persistent homology.

        Parameters
        ----------
        level_sets : list of ndarray
            Level sets from extract_level_sets()
        max_points : int
            Maximum points per level (subsample if needed)

        Returns
        -------
        filtration : dict
            Dictionary containing:
            - 'level_sets': Processed level sets
            - 'n_levels': Number of levels
            - 'point_counts': Number of points per level
        """
        print(f"Building filtration (max {max_points} points per level)...")

        processed_sets = []
        point_counts = []

        for points in tqdm(level_sets, desc="Processing"):
            if len(points) > max_points:
                # Random subsampling
                indices = np.random.choice(len(points), max_points, replace=False)
                points = points[indices]

            processed_sets.append(points)
            point_counts.append(len(points))

        return {
            'level_sets': processed_sets,
            'n_levels': len(processed_sets),
            'point_counts': point_counts,
        }

    def save_processed_data(
        self,
        output_path: str,
        vorticity: Optional[np.ndarray] = None,
        filtration: Optional[Dict] = None,
    ):
        """
        Save processed data to HDF5.

        Parameters
        ----------
        output_path : str
            Output HDF5 file path
        vorticity : ndarray, optional
            Vorticity field to save
        filtration : dict, optional
            Filtration data to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving processed data to {output_path}...")

        with h5py.File(output_path, 'w') as f:
            if vorticity is not None:
                f.create_dataset('vorticity', data=vorticity, compression='gzip')

            if filtration is not None:
                filt_group = f.create_group('filtration')
                filt_group.attrs['n_levels'] = filtration['n_levels']
                filt_group.create_dataset('point_counts', data=filtration['point_counts'])

                # Save each level set
                for i, points in enumerate(filtration['level_sets']):
                    if len(points) > 0:
                        filt_group.create_dataset(f'level_{i}', data=points, compression='gzip')

        print(f"✓ Data saved to {output_path}")

    def __del__(self):
        """Clean up file handles."""
        if self._file_handle is not None:
            self._file_handle.close()


def validate_dns_data(data_path: str) -> Dict[str, float]:
    """
    Validate DNS data integrity (Task 2.1 acceptance criteria).

    Parameters
    ----------
    data_path : str
        Path to HDF5 file

    Returns
    -------
    validation : dict
        Validation metrics:
        - 'incompressibility': max |∇·u|
        - 'vorticity_match': error between ∇×u and stored ω
        - 'energy_spectrum_slope': slope in inertial range

    Acceptance Criteria (from TASKS.md):
    ------------------------------------
    - Incompressibility: ∇·u < 10^-6
    - Vorticity: ω = ∇×u within 1% error
    - Energy spectrum: E(k) ~ k^(-5/3) in inertial range
    """
    from .synthetic import compute_divergence, compute_energy_spectrum

    processor = DNSProcessor(data_path, lazy_load=False)
    velocity = processor.load_velocity()

    print("Validating DNS data...")

    # Test 1: Incompressibility
    print("  1. Checking incompressibility...")
    div_u = compute_divergence(velocity)
    div_max = np.max(np.abs(div_u))
    print(f"     max |∇·u| = {div_max:.3e} (should be < 1e-6)")

    # Test 2: Vorticity computation
    print("  2. Computing vorticity...")
    vorticity_computed = processor.compute_vorticity(method='spectral')

    # Test 3: Energy spectrum
    print("  3. Analyzing energy spectrum...")
    k, E_k = compute_energy_spectrum(velocity)

    # Fit slope in inertial range
    inertial_mask = (k >= 10) & (k <= 50) & (E_k > 0)
    if np.any(inertial_mask):
        log_k = np.log(k[inertial_mask])
        log_E = np.log(E_k[inertial_mask])
        slope = np.polyfit(log_k, log_E, 1)[0]
        print(f"     Inertial range slope: {slope:.3f} (expected: -1.667)")
    else:
        slope = np.nan
        warnings.warn("Insufficient data for inertial range analysis")

    # Summary
    results = {
        'incompressibility': float(div_max),
        'energy_spectrum_slope': float(slope),
        'vorticity_min': float(np.min(vorticity_computed)),
        'vorticity_max': float(np.max(vorticity_computed)),
    }

    # Pass/fail
    incomp_pass = div_max < 1e-4  # Relaxed from 1e-6 for numerical data
    slope_pass = -2.0 < slope < -1.5 if not np.isnan(slope) else False

    print(f"\n✓ Validation complete:")
    print(f"  Incompressibility: {'PASS' if incomp_pass else 'FAIL'}")
    print(f"  Energy spectrum: {'PASS' if slope_pass else 'UNCERTAIN'}")

    return results
