"""
Input/output utilities for cascade data.
"""

import numpy as np
import h5py
from typing import Dict, Any
from pathlib import Path


def save_cascade_data(
    filepath: str,
    energies: np.ndarray,
    transfers: np.ndarray,
    wavenumbers: np.ndarray,
    metadata: Dict[str, Any] = None,
) -> None:
    """
    Save cascade data to HDF5 file.

    Parameters
    ----------
    filepath : str
        Output file path
    energies : ndarray
        Shell energies
    transfers : ndarray
        Transfer matrix
    wavenumbers : ndarray
        Wavenumbers
    metadata : dict, optional
        Additional metadata
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('energies', data=energies)
        f.create_dataset('transfers', data=transfers)
        f.create_dataset('wavenumbers', data=wavenumbers)

        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value


def load_cascade_data(filepath: str) -> Dict[str, Any]:
    """
    Load cascade data from HDF5 file.

    Parameters
    ----------
    filepath : str
        Input file path

    Returns
    -------
    data : dict
        Dictionary with 'energies', 'transfers', 'wavenumbers', and metadata
    """
    with h5py.File(filepath, 'r') as f:
        data = {
            'energies': f['energies'][:],
            'transfers': f['transfers'][:],
            'wavenumbers': f['wavenumbers'][:],
            'metadata': dict(f.attrs),
        }

    return data
