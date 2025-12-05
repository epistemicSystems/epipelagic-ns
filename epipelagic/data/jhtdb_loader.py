"""
Johns Hopkins Turbulence Database (JHTDB) data loader.

This module provides utilities for downloading and loading DNS turbulence data
from JHTDB without requiring the pyJHTDB package (which has numpy compatibility issues).

Data Sources:
- JHTDB Web Portal: http://turbulence.pha.jhu.edu/
- Direct data download via HTTP
- Pre-downloaded HDF5 files

References:
    [1] Li et al. (2008). "A Public Turbulence Database Cluster..."
    [2] http://turbulence.pha.jhu.edu/
"""

import os
import urllib.request
import urllib.error
from typing import Optional, Tuple, Dict
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm


class JHTDBLoader:
    """
    Load DNS turbulence data from JHTDB.

    This class provides methods to:
    1. Load pre-downloaded HDF5 files
    2. Access velocity and vorticity fields
    3. Validate data integrity

    Examples
    --------
    >>> loader = JHTDBLoader('/path/to/jhtdb_data.h5')
    >>> velocity = loader.get_velocity()
    >>> vorticity = loader.get_vorticity()
    >>> print(f"Reynolds number: {loader.get_reynolds_number()}")
    """

    def __init__(self, data_path: str):
        """
        Initialize JHTDB loader.

        Parameters
        ----------
        data_path : str
            Path to HDF5 file containing JHTDB data
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self._file = None
        self._metadata = {}

    def __enter__(self):
        """Context manager entry."""
        self._file = h5py.File(self.data_path, 'r')
        self._load_metadata()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file is not None:
            self._file.close()

    def _load_metadata(self):
        """Load metadata from HDF5 file."""
        if 'metadata' in self._file.attrs:
            for key, value in self._file.attrs.items():
                self._metadata[key] = value

    def get_velocity(self, region: Optional[Tuple[slice, ...]] = None) -> np.ndarray:
        """
        Get velocity field u(x,y,z) = (ux, uy, uz).

        Parameters
        ----------
        region : tuple of slices, optional
            Region to extract (default: entire field)

        Returns
        -------
        velocity : ndarray, shape (nx, ny, nz, 3)
            Velocity field with components (ux, uy, uz)
        """
        if self._file is None:
            raise RuntimeError("Use JHTDBLoader as context manager")

        if 'velocity' not in self._file:
            raise KeyError("Velocity field not found in data file")

        if region is None:
            return self._file['velocity'][...]
        else:
            return self._file['velocity'][region]

    def get_vorticity(self, region: Optional[Tuple[slice, ...]] = None) -> np.ndarray:
        """
        Get vorticity field ω(x,y,z) or compute from velocity.

        Parameters
        ----------
        region : tuple of slices, optional
            Region to extract

        Returns
        -------
        vorticity : ndarray, shape (nx, ny, nz, 3) or (nx, ny, nz)
            Vorticity field (magnitude if precomputed)
        """
        if self._file is None:
            raise RuntimeError("Use JHTDBLoader as context manager")

        # Check if vorticity is precomputed
        if 'vorticity' in self._file:
            if region is None:
                return self._file['vorticity'][...]
            else:
                return self._file['vorticity'][region]

        # Otherwise compute from velocity
        print("Computing vorticity from velocity field...")
        from ..topology.persistent import compute_vorticity
        velocity = self.get_velocity(region)
        return compute_vorticity(velocity)

    def get_reynolds_number(self) -> float:
        """
        Get Reynolds number from metadata.

        Returns
        -------
        Re : float
            Reynolds number
        """
        if 'reynolds_number' in self._metadata:
            return float(self._metadata['reynolds_number'])
        elif 'Re_lambda' in self._metadata:
            return float(self._metadata['Re_lambda'])
        else:
            raise KeyError("Reynolds number not found in metadata")

    def get_grid_shape(self) -> Tuple[int, int, int]:
        """
        Get grid shape.

        Returns
        -------
        shape : tuple of int
            Grid dimensions (nx, ny, nz)
        """
        if self._file is None:
            raise RuntimeError("Use JHTDBLoader as context manager")

        if 'velocity' in self._file:
            return self._file['velocity'].shape[:-1]  # Exclude vector dimension
        else:
            raise KeyError("Cannot determine grid shape")

    def get_metadata(self) -> Dict:
        """
        Get all metadata.

        Returns
        -------
        metadata : dict
            Dictionary of metadata fields
        """
        return self._metadata.copy()


def download_jhtdb_data(
    output_path: str,
    dataset: str = 'isotropic1024coarse',
    time: float = 0.364,
    resolution: Tuple[int, int, int] = (256, 256, 256),
    method: str = 'curl',
) -> str:
    """
    Download JHTDB data using various methods.

    Since pyJHTDB has compatibility issues, this function provides alternative
    download methods.

    Parameters
    ----------
    output_path : str
        Output path for downloaded data
    dataset : str
        Dataset name (e.g., 'isotropic1024coarse')
    time : float
        Time snapshot
    resolution : tuple of int
        Grid resolution (nx, ny, nz)
    method : str
        Download method: 'curl', 'wget', or 'manual'

    Returns
    -------
    instructions : str
        Instructions or confirmation message

    Notes
    -----
    For Phase 2, we recommend:
    1. **Manual download** via JHTDB web portal (most reliable)
    2. **Synthetic data generation** for initial testing
    3. **Pre-processed datasets** from collaborators

    JHTDB Web Portal: http://turbulence.pha.jhu.edu/webquery/query.aspx

    Examples
    --------
    >>> instructions = download_jhtdb_data(
    ...     'data/dns/jhtdb_iso1024/velocity.h5',
    ...     method='manual'
    ... )
    >>> print(instructions)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if method == 'manual':
        instructions = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         JHTDB Data Download Instructions (Manual Method)                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dataset: {dataset}
Time: {time}
Resolution: {resolution}
Output: {output_path}

STEPS:
------
1. Visit JHTDB Web Portal:
   https://turbulence.pha.jhu.edu/webquery/query.aspx

2. Create a free account (if you haven't already)

3. Select Dataset:
   - Database: "Forced isotropic turbulence"
   - Dataset: "{dataset}"

4. Configure Query:
   - Time: {time}
   - Spatial Domain:
     * X: 0 to {resolution[0]/1024:.4f} (grid points: 0-{resolution[0]})
     * Y: 0 to {resolution[1]/1024:.4f} (grid points: 0-{resolution[1]})
     * Z: 0 to {resolution[2]/1024:.4f} (grid points: 0-{resolution[2]})

5. Select Fields:
   - Velocity (u, v, w)
   - Vorticity (optional, can be computed)

6. Output Format: HDF5 or NetCDF

7. Download and save to: {output_path}

8. Verify download:
   >>> from epipelagic.data import JHTDBLoader
   >>> with JHTDBLoader('{output_path}') as loader:
   ...     print(loader.get_grid_shape())
   ...     print(f"Re = {{loader.get_reynolds_number()}}")

ALTERNATIVE: Use Synthetic Data
--------------------------------
For testing and development, generate synthetic turbulence:

>>> from epipelagic.data import generate_synthetic_turbulence
>>> velocity = generate_synthetic_turbulence(
...     resolution=(256, 256, 256),
...     reynolds_number=1000
... )
>>> # Save to HDF5
>>> import h5py
>>> with h5py.File('{output_path}', 'w') as f:
...     f.create_dataset('velocity', data=velocity)
...     f.attrs['reynolds_number'] = 1000

╚══════════════════════════════════════════════════════════════════════════════╝
"""
        print(instructions)
        return instructions

    elif method == 'curl':
        # Note: Direct curl access requires authentication token
        print("⚠️  Warning: JHTDB requires authentication.")
        print("   Using manual method is recommended.")
        print("   Falling back to manual instructions...")
        return download_jhtdb_data(output_path, dataset, time, resolution, method='manual')

    elif method == 'wget':
        print("⚠️  Warning: JHTDB requires authentication.")
        print("   Using manual method is recommended.")
        print("   Falling back to manual instructions...")
        return download_jhtdb_data(output_path, dataset, time, resolution, method='manual')

    else:
        raise ValueError(f"Unknown method: {method}")


def create_sample_jhtdb_file(
    output_path: str,
    resolution: Tuple[int, int, int] = (64, 64, 64),
    reynolds_number: float = 1000,
) -> str:
    """
    Create a sample JHTDB-like HDF5 file with synthetic data.

    This is useful for testing the pipeline without downloading real data.

    Parameters
    ----------
    output_path : str
        Output path for sample file
    resolution : tuple of int
        Grid resolution
    reynolds_number : float
        Target Reynolds number

    Returns
    -------
    output_path : str
        Path to created file
    """
    from .synthetic import generate_synthetic_turbulence

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic turbulence ({resolution[0]}³ grid, Re={reynolds_number})...")
    velocity = generate_synthetic_turbulence(
        resolution=resolution,
        reynolds_number=reynolds_number,
    )

    print(f"Saving to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('velocity', data=velocity, compression='gzip')
        f.attrs['reynolds_number'] = reynolds_number
        f.attrs['Re_lambda'] = reynolds_number
        f.attrs['resolution'] = resolution
        f.attrs['dataset_type'] = 'synthetic'
        f.attrs['description'] = 'Synthetic turbulence for testing'

    print(f"✓ Sample file created: {output_path}")
    print(f"  Resolution: {resolution}")
    print(f"  Reynolds number: {reynolds_number}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return str(output_path)
